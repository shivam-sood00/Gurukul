# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

SMP_ROOT = Path(__file__).resolve().parents[1] / "source/Gurukul/Gurukul/tasks/manager_based/smp"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def smp_modules():
    """Load pure SMP modules without importing Gurukul.tasks."""
    package_name = "_smp_diffusion_contract_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(SMP_ROOT)]
    sys.modules[package_name] = package
    profiles = _load_module(f"{package_name}.profiles", SMP_ROOT / "profiles.py")
    diffusion = _load_module(f"{package_name}.diffusion", SMP_ROOT / "diffusion.py")
    yield types.SimpleNamespace(diffusion=diffusion, profiles=profiles)
    for name in (f"{package_name}.diffusion", f"{package_name}.profiles", package_name):
        sys.modules.pop(name, None)


def _prior(diffusion, feature_dim=5, window_size=3, timesteps=4, *, dtype=torch.float32):
    model_config = diffusion.MotionDenoiserConfig(
        window_size=window_size,
        feature_dim=feature_dim,
        hidden_dim=8,
        num_layers=1,
        num_heads=2,
        mlp_ratio=2.0,
    )
    q_low = torch.linspace(-2.0, -1.0, feature_dim, dtype=dtype)
    q_high = torch.linspace(1.0, 3.0, feature_dim, dtype=dtype)
    return diffusion.SmpPrior(
        model_config,
        diffusion.DiffusionConfig(T=timesteps),
        q_low,
        q_high,
        ema_decay=0.5,
    )


def test_cosine_epsilon_scheduler_and_denoiser_contract(smp_modules):
    diffusion = smp_modules.diffusion
    assert diffusion.DiffusionConfig().T == 50
    scheduler = diffusion.CosineDDPMScheduler()
    assert scheduler.betas.shape == (50,)
    assert torch.all((scheduler.betas > 0.0) & (scheduler.betas < 1.0))
    assert torch.all(scheduler.alpha_bars[1:] < scheduler.alpha_bars[:-1])

    generator = torch.Generator().manual_seed(17)
    clean = torch.randn(2, 3, 5, generator=generator)
    noise = torch.randn(2, 3, 5, generator=generator)
    timesteps = torch.tensor([0, 25], dtype=torch.long)
    noised = scheduler.add_noise(clean, noise, timesteps)
    recovered = scheduler.predict_clean(noise, noised, timesteps)
    torch.testing.assert_close(recovered, clean, atol=2.0e-5, rtol=2.0e-5)

    config = diffusion.MotionDenoiserConfig(3, 5, hidden_dim=8, num_layers=1, num_heads=2)
    denoiser = diffusion.MotionDenoiser(config)
    assert denoiser(clean, timesteps).shape == clean.shape
    with pytest.raises(ValueError, match="trailing shape"):
        denoiser(clean[..., :-1], timesteps)
    with pytest.raises(TypeError, match="torch.long"):
        denoiser(clean, timesteps.float())
    with pytest.raises(ValueError, match="divisible"):
        diffusion.MotionDenoiserConfig(3, 5, hidden_dim=7, num_heads=2)
    with pytest.raises(TypeError, match="integer"):
        diffusion.DiffusionConfig(T=4.0)


@pytest.mark.parametrize(
    ("profile_name", "expected_parameters"),
    (("g1", 2_668_338), ("pm01", 2_664_648), ("go2", 2_654_178)),
)
def test_default_denoiser_architecture_is_profile_dimensioned(
    smp_modules,
    profile_name,
    expected_parameters,
):
    profile = smp_modules.profiles.get_profile(profile_name)
    model = smp_modules.diffusion.MotionDenoiser(
        smp_modules.diffusion.MotionDenoiserConfig(window_size=10, feature_dim=profile.feature_dim)
    )

    assert sum(parameter.numel() for parameter in model.parameters()) == expected_parameters
    assert model.preprocess_projection.kernel_size == (1,)
    assert model.postprocess_projection.kernel_size == (1,)
    assert len(model.blocks) == 2
    assert model.blocks[0].attention.num_heads == 4


def test_training_is_l1_and_ema_is_frozen_and_updated(smp_modules):
    diffusion = smp_modules.diffusion
    prior = _prior(diffusion)
    prior.train()
    assert not prior.ema_model.training
    assert not any(parameter.requires_grad for parameter in prior.ema_model.parameters())

    generator = torch.Generator().manual_seed(23)
    physical = torch.randn(2, 3, 5, generator=generator).clamp(-1.0, 1.0)
    noise = torch.randn(2, 3, 5, generator=generator)
    timesteps = torch.tensor([1, 2], dtype=torch.long)
    normalized = prior.normalize(physical)
    noised = prior.scheduler.add_noise(normalized, noise, timesteps)
    expected = torch.nn.functional.l1_loss(prior.model(noised, timesteps), noise)
    loss = prior.training_loss(physical, timesteps=timesteps, noise=noise)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    assert any(parameter.grad is not None for parameter in prior.model.parameters())
    assert all(parameter.grad is None for parameter in prior.ema_model.parameters())

    model_parameter = next(prior.model.parameters())
    ema_parameter = next(prior.ema_model.parameters())
    old_ema = ema_parameter.detach().clone()
    with torch.no_grad():
        model_parameter.add_(1.0)
    prior.update_ema()
    torch.testing.assert_close(ema_parameter, 0.5 * old_ema + 0.5 * model_parameter)
    assert prior.ema_updates.item() == 1


def test_sds_uses_independent_noise_and_scores_from_one_previous_snapshot(smp_modules):
    diffusion = smp_modules.diffusion
    prior = _prior(diffusion, feature_dim=4, window_size=2, timesteps=5)
    for parameter in prior.ema_model.parameters():
        parameter.data.zero_()
    physical = torch.zeros(3, 2, 4)
    timestep_values = (1, 2, 3)

    expected_generator = torch.Generator().manual_seed(31)
    expected_noise = torch.randn(3 * 3, 2, 4, generator=expected_generator).reshape(3, 3, 2, 4)
    expected_losses = expected_noise.square().mean(dim=(-2, -1))
    losses = prior.sds_losses(
        physical,
        timesteps=timestep_values,
        generator=torch.Generator().manual_seed(31),
    )
    torch.testing.assert_close(losses, expected_losses)

    scale = 1.7
    expected_first = torch.exp(-scale * losses.mean(dim=-1))
    first = prior.score_from_losses(losses, timestep_values, loss_scale=scale)
    torch.testing.assert_close(first, expected_first)
    assert torch.equal(prior.score_running_count[torch.tensor(timestep_values)], torch.full((3,), 3))

    previous_mean = losses.mean(dim=0)
    expected_second = torch.exp(-scale * (losses / previous_mean).mean(dim=-1))
    second = prior.score_from_losses(losses, timestep_values, loss_scale=scale, update_normalizer=False)
    torch.testing.assert_close(second, expected_second)
    assert torch.all((second >= 0.0) & (second <= 1.0))

    restored = _prior(diffusion, feature_dim=4, window_size=2, timesteps=5)
    restored.load_score_normalizer_state(prior.score_normalizer_state())
    assert torch.equal(restored.score_running_mean, prior.score_running_mean)
    assert torch.equal(restored.score_running_count, prior.score_running_count)


def test_seeded_ddpm_sampling_returns_finite_physical_windows(smp_modules):
    prior = _prior(smp_modules.diffusion, timesteps=3)
    first = prior.sample(2, generator=torch.Generator().manual_seed(47))
    second = prior.sample(2, generator=torch.Generator().manual_seed(47))
    assert first.shape == (2, 3, 5)
    assert torch.isfinite(first).all()
    assert torch.equal(first, second)

    physical = torch.linspace(-1.0, 1.0, 30).reshape(2, 3, 5)
    torch.testing.assert_close(prior.unnormalize(prior.normalize(physical)), physical)


def test_checkpoint_roundtrip_is_weights_only_safe_and_schema_strict(smp_modules, tmp_path):
    diffusion = smp_modules.diffusion
    profile = smp_modules.profiles.GO2_PROFILE
    prior = _prior(diffusion, feature_dim=profile.feature_dim, timesteps=3, dtype=torch.float64)
    prior.score_from_losses(
        torch.full((2, 2), 0.25, dtype=torch.float64),
        timesteps=(1, 2),
    )
    checkpoint_path = diffusion.save_smp_checkpoint(
        tmp_path / "go2_prior.pt",
        prior,
        profile,
        training_metadata={"step": 7, "loss": 0.25},
    )
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    assert payload["format"] == diffusion.SMP_CHECKPOINT_FORMAT

    loaded = diffusion.load_smp_checkpoint(
        checkpoint_path,
        expected_profile="unitree-go2",
        expected_control_fps=50.0,
    )
    assert loaded.profile is profile
    assert loaded.training_metadata == {"step": 7, "loss": 0.25}
    assert torch.equal(loaded.q_low, prior.q_low)
    assert torch.equal(loaded.q_high, prior.q_high)
    assert torch.equal(loaded.prior.score_running_count, prior.score_running_count)
    assert not any(parameter.requires_grad for parameter in loaded.prior.parameters())

    bad_order = dict(payload)
    bad_order["profile"] = dict(payload["profile"])
    bad_order["profile"]["joint_names"] = list(payload["profile"]["joint_names"])
    bad_order["profile"]["joint_names"][0:2] = reversed(bad_order["profile"]["joint_names"][0:2])
    bad_order_path = tmp_path / "bad_order.pt"
    torch.save(bad_order, bad_order_path)
    with pytest.raises(ValueError, match="joint_names"):
        diffusion.load_smp_checkpoint(bad_order_path, expected_profile="go2")
    with pytest.raises(ValueError, match="control-rate mismatch"):
        diffusion.load_smp_checkpoint(checkpoint_path, expected_control_fps=60.0)
    with pytest.raises(ValueError, match="profile mismatch"):
        diffusion.load_smp_checkpoint(checkpoint_path, expected_profile="g1")

    bad_dtype = dict(payload)
    bad_dtype["model_state_dict"] = dict(payload["model_state_dict"])
    state_key = next(iter(bad_dtype["model_state_dict"]))
    bad_dtype["model_state_dict"][state_key] = bad_dtype["model_state_dict"][state_key].float()
    bad_dtype_path = tmp_path / "bad_dtype.pt"
    torch.save(bad_dtype, bad_dtype_path)
    with pytest.raises(TypeError, match="dtype"):
        diffusion.load_smp_checkpoint(bad_dtype_path)


def test_prior_rejects_nonfinite_or_mismatched_motion_contracts(smp_modules):
    diffusion = smp_modules.diffusion
    prior = _prior(diffusion)
    with pytest.raises(TypeError, match="bounds use"):
        prior.normalize(torch.zeros(2, 3, 5, dtype=torch.float64))
    bad = torch.zeros(2, 3, 5)
    bad[0, 0, 0] = torch.nan
    with pytest.raises(ValueError, match="finite"):
        prior.score(bad, timesteps=(1, 2))
    with pytest.raises(ValueError, match="shape"):
        prior.score_from_losses(torch.ones(2, 3), timesteps=(1, 2))
    with pytest.raises(TypeError, match="integers"):
        prior.sds_losses(torch.zeros(2, 3, 5), timesteps=(1.5, 2.5))
