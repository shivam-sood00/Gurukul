# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
import importlib.util
import inspect
import sys
import types
from pathlib import Path
from types import SimpleNamespace

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


class _FakeOnPolicyRunner:
    """Small stand-in that records the superclass lifecycle calls."""

    def save(self, path: str, infos: dict | None = None) -> None:
        self.base_save_call = (path, infos)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        self.base_learn_call = (num_learning_iterations, init_at_random_ep_len)
        self.alg.update()

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ):
        self.base_load_call = (path, load_cfg, strict, map_location)
        return self.base_load_result


@pytest.fixture(scope="module")
def smp_modules():
    """Load SMP runtime modules without importing Isaac Lab or a real RSL-RL."""

    package_name = "_smp_runtime_contract_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(SMP_ROOT)]
    sys.modules[package_name] = package

    previous_rsl_rl = sys.modules.get("rsl_rl")
    previous_runners = sys.modules.get("rsl_rl.runners")
    rsl_rl = types.ModuleType("rsl_rl")
    runners = types.ModuleType("rsl_rl.runners")
    runners.OnPolicyRunner = _FakeOnPolicyRunner
    rsl_rl.runners = runners
    sys.modules["rsl_rl"] = rsl_rl
    sys.modules["rsl_rl.runners"] = runners

    loaded = {}
    try:
        for module_name in ("profiles", "features", "diffusion", "runtime", "runner"):
            qualified_name = f"{package_name}.{module_name}"
            loaded[module_name] = _load_module(qualified_name, SMP_ROOT / f"{module_name}.py")
        yield SimpleNamespace(**loaded)
    finally:
        for module_name in reversed(("profiles", "features", "diffusion", "runtime", "runner")):
            sys.modules.pop(f"{package_name}.{module_name}", None)
        sys.modules.pop(package_name, None)
        if previous_runners is None:
            sys.modules.pop("rsl_rl.runners", None)
        else:
            sys.modules["rsl_rl.runners"] = previous_runners
        if previous_rsl_rl is None:
            sys.modules.pop("rsl_rl", None)
        else:
            sys.modules["rsl_rl"] = previous_rsl_rl


def _reset_env(smp_modules, *, num_envs: int = 4):
    state = SimpleNamespace(
        gsi_pool=None,
        needs_history_prime=torch.zeros(num_envs, dtype=torch.bool),
    )
    env = SimpleNamespace(
        num_envs=num_envs,
        device="cpu",
        _smp_runtime_state=state,
    )
    return env, state


def test_event_terms_keep_env_ids_as_the_manager_positional_argument(smp_modules):
    for function in (smp_modules.runtime.initialize_smp, smp_modules.runtime.reset_smp_state):
        env_ids = inspect.signature(function).parameters["env_ids"]
        assert env_ids.default is inspect.Parameter.empty


def test_named_motion_root_may_use_a_coincident_fixed_dummy_root(smp_modules):
    profile = smp_modules.profiles.G1_PROFILE

    class Robot:
        def __init__(self):
            self.data = SimpleNamespace(
                root_link_pos_w=torch.zeros(2, 3),
                root_link_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(2, 1),
                body_link_pos_w=torch.zeros(2, 7, 3),
                body_link_quat_w=torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(2, 7, 4).clone(),
            )

        def find_joints(self, names, preserve_order):
            assert preserve_order
            return list(range(len(names))), list(names)

        def find_bodies(self, names, preserve_order):
            assert preserve_order
            if names == [profile.root_body_name]:
                return [1], list(names)
            return list(range(2, 2 + len(names))), list(names)

    robot = Robot()
    _, _, root_body_id = smp_modules.runtime._resolve_named_indices(robot, profile, "cpu")
    assert root_body_id == 1

    robot.data.body_link_pos_w[:, 1, 0] = 0.01
    with pytest.raises(RuntimeError, match="not coincident"):
        smp_modules.runtime._resolve_named_indices(robot, profile, "cpu")


def test_online_features_use_one_consistent_link_frame(smp_modules):
    body_link_pos = torch.tensor(
        [
            [[10.0, 20.0, 0.5], [11.0, 20.0, 0.1], [9.0, 20.0, 0.1]],
            [[-3.0, 4.0, 0.6], [-2.0, 4.0, 0.2], [-4.0, 4.0, 0.2]],
        ]
    )
    identity = torch.tensor([1.0, 0.0, 0.0, 0.0]).expand(2, 3, 4).clone()
    link_lin_vel = torch.tensor([[[1.0, 2.0, 3.0]] * 3, [[4.0, 5.0, 6.0]] * 3])
    link_ang_vel = torch.tensor([[[0.1, 0.2, 0.3]] * 3, [[0.4, 0.5, 0.6]] * 3])
    robot = SimpleNamespace(
        data=SimpleNamespace(
            body_link_pos_w=body_link_pos,
            body_link_quat_w=identity,
            body_link_lin_vel_w=link_lin_vel,
            body_link_ang_vel_w=link_ang_vel,
            # These incompatible legacy COM aliases must never be read.
            body_pos_w=torch.full_like(body_link_pos, 999.0),
            body_quat_w=torch.full_like(identity, 999.0),
            body_lin_vel_w=torch.full_like(link_lin_vel, 999.0),
            body_ang_vel_w=torch.full_like(link_ang_vel, 999.0),
            joint_pos=torch.arange(8, dtype=torch.float32).reshape(2, 4),
        )
    )
    origins = torch.tensor([[10.0, 20.0, 0.0], [-3.0, 4.0, 0.0]])
    env = SimpleNamespace(scene={"robot": robot})
    env.scene = SimpleNamespace(robot=robot, env_origins=origins)
    env.scene.__getitem__ = lambda name: robot

    class Scene(SimpleNamespace):
        def __getitem__(self, name):
            assert name == "robot"
            return self.robot

    env.scene = Scene(robot=robot, env_origins=origins)
    state = SimpleNamespace(
        root_body_id=0,
        joint_ids=torch.tensor([1, 3]),
        key_body_ids=torch.tensor([1, 2]),
    )

    root_pos, root_quat, joint_pos, key_body_pos, root_lin_vel, root_ang_vel = smp_modules.runtime._read_kinematics(
        env, state
    )

    torch.testing.assert_close(root_pos, torch.tensor([[0.0, 0.0, 0.5], [0.0, 0.0, 0.6]]))
    torch.testing.assert_close(root_quat, identity[:, 0])
    torch.testing.assert_close(joint_pos, torch.tensor([[1.0, 3.0], [5.0, 7.0]]))
    torch.testing.assert_close(key_body_pos, body_link_pos[:, 1:] - origins[:, None])
    torch.testing.assert_close(root_lin_vel, link_lin_vel[:, 0])
    torch.testing.assert_close(root_ang_vel, link_ang_vel[:, 0])


@pytest.mark.parametrize(
    ("env_ids", "expected"),
    (
        (None, (0, 1, 2, 3)),
        ([1, 3], (1, 3)),
        (slice(1, 4, 2), (1, 3)),
    ),
)
def test_reset_selects_none_list_and_slice(smp_modules, env_ids, expected):
    env, state = _reset_env(smp_modules)

    smp_modules.runtime.reset_smp_state(env, env_ids)

    selected = tuple(state.needs_history_prime.nonzero().flatten().tolist())
    assert selected == expected


@pytest.mark.parametrize(
    "env_ids",
    (
        [True, False, True, False],
        torch.tensor([False, True, False, True]),
    ),
)
def test_reset_rejects_boolean_indices(smp_modules, env_ids):
    env, _ = _reset_env(smp_modules)
    with pytest.raises(TypeError, match="booleans"):
        smp_modules.runtime.reset_smp_state(env, env_ids)


@pytest.mark.parametrize(
    "env_ids",
    (
        [1.0],
        [1.25],
        torch.tensor([2.0]),
    ),
)
def test_reset_rejects_float_indices(smp_modules, env_ids):
    env, _ = _reset_env(smp_modules)
    with pytest.raises(TypeError, match="integer"):
        smp_modules.runtime.reset_smp_state(env, env_ids)


@pytest.mark.parametrize("env_ids", ([-1], [4], torch.tensor([0, 9])))
def test_reset_rejects_out_of_range_indices(smp_modules, env_ids):
    env, _ = _reset_env(smp_modules)
    with pytest.raises(IndexError, match=r"\[0, 4\)"):
        smp_modules.runtime.reset_smp_state(env, env_ids)


def test_reset_rejects_duplicate_indices(smp_modules):
    env, _ = _reset_env(smp_modules)
    with pytest.raises(ValueError, match="duplicates"):
        smp_modules.runtime.reset_smp_state(env, [1, 1])


class _RewardHistory:
    def __init__(self, num_envs: int):
        self.num_envs = num_envs
        self.append_calls = 0
        self.fill_calls = 0

    def fill(self, *args) -> None:
        self.fill_calls += 1

    def append(self, *args) -> None:
        self.append_calls += 1

    def features(self) -> torch.Tensor:
        return torch.zeros(self.num_envs, 10, 1)


def _reward_env(num_envs: int = 3):
    history = _RewardHistory(num_envs)
    state = SimpleNamespace(
        history=history,
        needs_history_prime=torch.zeros(num_envs, dtype=torch.bool),
    )
    env = SimpleNamespace(num_envs=num_envs, _smp_runtime_state=state)
    return env, state, history


def _patch_reward_inputs(monkeypatch, runtime, reward: torch.Tensor) -> None:
    monkeypatch.setattr(
        runtime,
        "_read_kinematics",
        lambda env, state: tuple(torch.zeros(env.num_envs, 1) for _ in range(6)),
    )
    monkeypatch.setattr(runtime, "_score_prior", lambda state, windows: reward)


def test_reward_accepts_finite_unit_interval_vector(smp_modules, monkeypatch):
    env, _, history = _reward_env()
    expected = torch.tensor([0.0, 0.5, 1.0])
    _patch_reward_inputs(monkeypatch, smp_modules.runtime, expected)

    actual = smp_modules.runtime.smp_guidance_reward(env)

    assert torch.equal(actual, expected)
    assert history.append_calls == 1


@pytest.mark.parametrize(
    ("reward", "message"),
    (
        (torch.zeros(3, 1), "reward shape"),
        (torch.tensor([0.0, float("nan"), 1.0]), "non-finite"),
        (torch.tensor([0.0, float("inf"), 1.0]), "non-finite"),
        (torch.tensor([-0.01, 0.5, 1.0]), r"\[0, 1\]"),
        (torch.tensor([0.0, 0.5, 1.01]), r"\[0, 1\]"),
    ),
)
def test_reward_rejects_invalid_shape_finiteness_or_range(smp_modules, monkeypatch, reward, message):
    env, _, _ = _reward_env()
    _patch_reward_inputs(monkeypatch, smp_modules.runtime, reward)

    with pytest.raises(RuntimeError, match=message):
        smp_modules.runtime.smp_guidance_reward(env)


class _ChunkedPrior:
    def __init__(self):
        self.loss_batches = []
        self.score_calls = []

    def sds_losses(self, windows: torch.Tensor, *, timesteps):
        self.loss_batches.append((windows.detach().clone(), timesteps))
        marker = windows[:, 0, 0]
        offsets = torch.tensor([0.1, 0.2, 0.3], dtype=windows.dtype)
        return marker[:, None] + offsets[None]

    def score_from_losses(self, losses: torch.Tensor, **kwargs):
        self.score_calls.append((losses.detach().clone(), kwargs))
        return torch.full((losses.shape[0],), 0.25, dtype=losses.dtype)


def test_chunked_sds_losses_are_concatenated_before_one_score_update(smp_modules):
    prior = _ChunkedPrior()
    state = SimpleNamespace(
        checkpoint=SimpleNamespace(prior=prior),
        fixed_timesteps=(8, 15, 22),
        loss_scale=6.0,
        score_batch_size=2,
        update_score_normalizer=True,
    )
    windows = torch.zeros(5, 10, 1)
    windows[:, 0, 0] = torch.arange(5, dtype=torch.float32)

    reward = smp_modules.runtime._score_prior(state, windows)

    assert [batch.shape[0] for batch, _ in prior.loss_batches] == [2, 2, 1]
    assert all(timesteps == (8, 15, 22) for _, timesteps in prior.loss_batches)
    assert len(prior.score_calls) == 1
    losses, kwargs = prior.score_calls[0]
    expected = torch.arange(5, dtype=torch.float32)[:, None] + torch.tensor([0.1, 0.2, 0.3])
    torch.testing.assert_close(losses, expected)
    assert kwargs == {
        "timesteps": (8, 15, 22),
        "loss_scale": 6.0,
        "update_normalizer": True,
        "defer_normalizer_update": True,
    }
    assert torch.equal(reward, torch.full((5,), 0.25))

    state.update_score_normalizer = False
    smp_modules.runtime._score_prior(state, windows)
    assert prior.score_calls[-1][1]["update_normalizer"] is False


def test_runner_flushes_deferred_score_statistics_once_per_rollout(smp_modules):
    wrapped_env, state = _runner_env(smp_modules)
    prior = state.checkpoint.prior
    prior.score_from_losses(
        torch.tensor([[0.25, 0.5], [0.75, 1.0]]),
        timesteps=(1, 2),
        defer_normalizer_update=True,
        synchronize_distributed=False,
    )
    before = prior.score_running_count.clone()
    assert torch.equal(prior.score_pending_count[[1, 2]], torch.tensor([2, 2]))

    update_calls = []
    runner = smp_modules.runner.SmpOnPolicyRunner()
    runner.env = wrapped_env
    runner.alg = SimpleNamespace(update=lambda: update_calls.append(True))
    runner.learn(1, init_at_random_ep_len=True)

    assert runner.base_learn_call == (1, True)
    assert update_calls == [True]
    assert torch.equal(prior.score_running_count[[1, 2]], before[[1, 2]] + 2)
    assert torch.count_nonzero(prior.score_pending_count) == 0


def test_root_finite_difference_recovers_linear_and_angular_velocity(smp_modules):
    step_dt = 0.02
    frame_count = 8
    time = torch.arange(frame_count, dtype=torch.float64) * step_dt
    linear_velocity = torch.tensor([1.25, -0.5, 0.0], dtype=torch.float64)
    root_pos = time[None, :, None] * linear_velocity[None, None, :]

    yaw_rate = 0.7
    yaw = yaw_rate * time
    root_quat = torch.zeros(1, frame_count, 4, dtype=torch.float64)
    root_quat[0, :, 0] = torch.cos(0.5 * yaw)
    root_quat[0, :, 3] = torch.sin(0.5 * yaw)
    # Quaternion sign changes must not create angular-velocity spikes.
    root_quat[:, 1::2] *= -1.0

    linear, angular = smp_modules.runtime._finite_difference_root_motion(root_pos, root_quat, step_dt)

    torch.testing.assert_close(linear, linear_velocity.expand_as(linear), atol=1.0e-12, rtol=1.0e-12)
    expected_angular = torch.tensor([0.0, 0.0, yaw_rate], dtype=torch.float64).expand_as(angular)
    torch.testing.assert_close(angular, expected_angular, atol=1.0e-12, rtol=1.0e-12)


def _tiny_prior(diffusion):
    return diffusion.SmpPrior(
        diffusion.MotionDenoiserConfig(
            window_size=2,
            feature_dim=3,
            hidden_dim=8,
            num_layers=1,
            num_heads=2,
        ),
        diffusion.DiffusionConfig(T=4),
        torch.full((3,), -1.0),
        torch.full((3,), 1.0),
    )


def _runner_env(smp_modules, *, digest: str = "a" * 64):
    prior = _tiny_prior(smp_modules.diffusion)
    prior.score_from_losses(
        torch.tensor([[0.25, 0.5], [0.75, 1.0]]),
        timesteps=(1, 2),
        synchronize_distributed=False,
    )
    state = SimpleNamespace(
        checkpoint_sha256=digest,
        profile=smp_modules.profiles.GO2_PROFILE,
        checkpoint=SimpleNamespace(prior=prior),
    )
    manager_env = SimpleNamespace(_smp_runtime_state=state)
    return SimpleNamespace(env=manager_env), state


def test_runner_save_and_load_export_and_restore_normalizer(smp_modules):
    wrapped_env, state = _runner_env(smp_modules)
    runner = smp_modules.runner.SmpOnPolicyRunner()
    runner.env = wrapped_env
    caller_infos = {"iteration": 12}

    runner.save("policy.pt", caller_infos)

    saved_path, saved_infos = runner.base_save_call
    assert saved_path == "policy.pt"
    assert caller_infos == {"iteration": 12}
    assert saved_infos["iteration"] == 12
    payload = saved_infos[smp_modules.runner.SMP_RUNNER_STATE_KEY]
    assert payload["prior_sha256"] == "a" * 64
    assert payload["profile"] == state.profile.to_metadata()
    expected_normalizer = copy.deepcopy(payload["score_normalizer"])

    state.checkpoint.prior.score_running_mean.zero_()
    state.checkpoint.prior.score_running_count.zero_()
    runner.base_load_result = saved_infos
    loaded_infos = runner.load(
        "policy.pt",
        load_cfg={"resume": True},
        strict=False,
        map_location="cpu",
    )

    assert loaded_infos is saved_infos
    assert runner.base_load_call == ("policy.pt", {"resume": True}, False, "cpu")
    restored = state.checkpoint.prior.score_normalizer_state()
    assert torch.equal(restored["score_running_mean"], expected_normalizer["score_running_mean"])
    assert torch.equal(restored["score_running_count"], expected_normalizer["score_running_count"])


def test_runner_restore_rejects_prior_digest_and_profile_mismatch(smp_modules):
    wrapped_env, state = _runner_env(smp_modules)
    payload = smp_modules.runner._export_smp_state(wrapped_env)
    original = state.checkpoint.prior.score_normalizer_state()

    wrong_digest = copy.deepcopy(payload)
    wrong_digest["prior_sha256"] = "b" * 64
    with pytest.raises(ValueError, match="different prior file"):
        smp_modules.runner._restore_smp_state(wrapped_env, wrong_digest)

    wrong_profile = copy.deepcopy(payload)
    wrong_profile["profile"]["name"] = "g1"
    with pytest.raises(ValueError, match="profile"):
        smp_modules.runner._restore_smp_state(wrapped_env, wrong_profile)

    current = state.checkpoint.prior.score_normalizer_state()
    assert torch.equal(current["score_running_mean"], original["score_running_mean"])
    assert torch.equal(current["score_running_count"], original["score_running_count"])


def test_training_resume_requires_smp_normalizer_state(smp_modules):
    wrapped_env, _ = _runner_env(smp_modules)
    runner = smp_modules.runner.SmpOnPolicyRunner()
    runner.env = wrapped_env
    runner.base_load_result = None

    with pytest.raises(ValueError, match="training resume requires"):
        runner.load("legacy_policy.pt")
