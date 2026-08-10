
from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
REAL_TEACHER_PATH = REPO_ROOT / "source/Gurukul/Gurukul/tasks/manager_based/locomotion/velocity/real_teacher.py"
PRETRAIN_ATTENTION_PATH = REPO_ROOT / "source/Gurukul/Gurukul/tasks/manager_based/locomotion/velocity/pretrain_attention.py"
REAL_TEACHER_VIZ_PATH = REPO_ROOT / "source/Gurukul/Gurukul/tasks/manager_based/locomotion/velocity/real_teacher_viz.py"
ROUGH_ENV_CFG_PATH = REPO_ROOT / "source/Gurukul/Gurukul/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_go2/rough_env_cfg.py"
REAL_SPARSE_ENV_CFG_PATH = REPO_ROOT / "source/Gurukul/Gurukul/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_go2/rough_real_sparse_env_cfg.py"
REAL_BEAM_ENV_CFG_PATH = REPO_ROOT / "source/Gurukul/Gurukul/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_go2/rough_real_beam_env_cfg.py"
START_TERRAINS_CFG_PATH = REPO_ROOT / "source/Gurukul/Gurukul/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_go2/start_terrains_cfg.py"
TEACHER_CFG_PATH = REPO_ROOT / "source/Gurukul/Gurukul/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_go2/agents/rsl_rl_teacher_cfg.py"
REGISTRY_PATH = REPO_ROOT / "source/Gurukul/Gurukul/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_go2/__init__.py"
TRAIN_SCRIPT_PATH = REPO_ROOT / "scripts/reinforcement_learning/rsl_rl/train.py"
PRETRAIN_SCRIPT_PATH = REPO_ROOT / "scripts/tools/pretrain_real_attention.py"
REAL_DOCS_PATH = REPO_ROOT / "website/docs/tasks/velocity-locomotion/real.md"


def _load_module(module_name: str, path: Path):
    package_name = "_real_teacher_test_modules"
    if package_name not in sys.modules:
        package = types.ModuleType(package_name)
        package.__path__ = [str(REAL_TEACHER_PATH.parent)]
        sys.modules[package_name] = package
    qualified_name = f"{package_name}.{module_name}"
    spec = importlib.util.spec_from_file_location(qualified_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[qualified_name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def real_teacher_module():
    return _load_module("real_teacher_test_module", REAL_TEACHER_PATH)


@pytest.fixture(scope="module")
def pretrain_attention_module():
    return _load_module("pretrain_attention_test_module", PRETRAIN_ATTENTION_PATH)


@pytest.fixture(scope="module")
def real_teacher_viz_module():
    return _load_module("real_teacher_viz_test_module", REAL_TEACHER_VIZ_PATH)


def _make_obs(batch_size: int = 8, terrain_dim: int = 160):
    obs = {
        "real_teacher_proprio": torch.randn(batch_size, 48),
        "real_teacher_terrain": torch.randn(batch_size, terrain_dim),
        "real_teacher_privileged": torch.randn(batch_size, 50),
    }
    obs_groups = {
        "policy": ["real_teacher_proprio", "real_teacher_terrain", "real_teacher_privileged"],
        "critic": ["real_teacher_proprio", "real_teacher_terrain", "real_teacher_privileged"],
    }
    return obs, obs_groups


def _build_policy(real_teacher_module, obs, obs_groups, **overrides):
    kwargs = {
        "obs": obs,
        "obs_groups": obs_groups,
        "num_actions": 12,
        "actor_obs_normalization": False,
        "critic_obs_normalization": False,
        "actor_hidden_dims": [128, 64],
        "critic_hidden_dims": [128, 64],
        "terrain_scan_shape": (16, 10),
        "attention_embed_dim": 64,
        "attention_num_heads": 4,
        "privileged_hidden_dims": [64],
        "privileged_latent_dim": 64,
    }
    kwargs.update(overrides)
    return real_teacher_module.RealTeacherActorCritic(**kwargs)


def _make_pretrain_checkpoint(tmp_path: Path, pretrain_attention_module, scan_shape: tuple[int, int] = (4, 4)) -> Path:
    num_scan_values = int(scan_shape[0]) * int(scan_shape[1])
    model = pretrain_attention_module.TerrainAttentionPretrainer(
        scan_shape=scan_shape,
        attention_embed_dim=32,
        attention_num_heads=4,
        terrain_encoder_hidden_dim=24,
        activation="elu",
    )
    height_scans = torch.randn(6, num_scan_values)
    targets = pretrain_attention_module.build_traversability_targets(
        height_scans,
        scan_shape=scan_shape,
        raster_resolution=0.1,
        apply_forward_prior=False,
    )
    dataset = pretrain_attention_module.OfflineTerrainDataset(
        height_scans=height_scans,
        traversability_targets=targets,
        scan_shape=scan_shape,
        terrain_set="unit_test",
        raster_resolution=0.1,
        base_heights=torch.full((height_scans.shape[0],), 0.33),
        metadata={"sampler": "unit_test"},
    )
    checkpoint_path = tmp_path / "pretrained_attention.pt"
    pretrain_attention_module.save_attention_checkpoint(
        model=model,
        output_path=checkpoint_path,
        dataset=dataset,
        history=[{"epoch": 1.0, "loss": 0.5, "alignment": 0.25, "entropy": 1.0}],
    )
    return checkpoint_path


def test_real_teacher_forward_shapes(real_teacher_module):
    obs, obs_groups = _make_obs(batch_size=8)
    policy = _build_policy(real_teacher_module, obs, obs_groups)

    actions = policy.act(obs)
    values = policy.evaluate(obs)
    log_prob = policy.get_actions_log_prob(actions)

    assert actions.shape == (8, 12)
    assert values.shape == (8, 1)
    assert log_prob.shape == (8,)
    assert policy.last_actor_attention_weights.shape == (8, 4, 160)


def test_real_teacher_logging_stats(real_teacher_module):
    obs, obs_groups = _make_obs(batch_size=5)
    policy = _build_policy(real_teacher_module, obs, obs_groups)

    policy.act(obs)
    metrics = policy.pop_logging_stats()

    assert "Policy/real_teacher_actor_attention_entropy" in metrics
    assert "Policy/real_teacher_actor_attention_max" in metrics
    assert "Policy/real_teacher_actor_attention_top5_mass" in metrics
    assert "Policy/real_teacher_terrain_scan_std" in metrics
    assert "Policy/real_teacher_terrain_scan_abs_mean" in metrics
    assert metrics["Policy/real_teacher_actor_attention_entropy"] >= 0.0
    assert metrics["Policy/real_teacher_actor_attention_max"] > 0.0
    assert metrics["Policy/real_teacher_actor_attention_top5_mass"] > 0.0

    assert policy.pop_logging_stats() == {}


def test_real_teacher_attention_map_is_normalized(real_teacher_module):
    obs, obs_groups = _make_obs(batch_size=3)
    policy = _build_policy(real_teacher_module, obs, obs_groups)

    attention_map = policy.get_actor_attention_map(obs)
    assert attention_map.shape == (3, 4, 16, 10)
    head_sums = attention_map.reshape(3, 4, -1).sum(dim=-1)
    assert torch.allclose(head_sums, torch.ones_like(head_sums), atol=1.0e-5)


def test_pretrain_attention_traversability_prefers_flat_supported_regions(pretrain_attention_module):
    flat_map = torch.zeros(1, 5, 5)
    slope_map = torch.linspace(0.0, 0.3, 5).view(1, 5, 1).expand(1, 5, 5).clone()
    hole_map = flat_map.clone()
    hole_map[0, 2, 3] = -0.5

    flat_scores = pretrain_attention_module.compute_traversability_from_heightmap(
        flat_map,
        resolution_x=0.1,
        normalize_distribution=False,
    )
    slope_scores = pretrain_attention_module.compute_traversability_from_heightmap(
        slope_map,
        resolution_x=0.1,
        normalize_distribution=False,
    )
    hole_scores = pretrain_attention_module.compute_traversability_from_heightmap(
        hole_map,
        resolution_x=0.1,
        normalize_distribution=False,
    )

    center_index = 2 * 5 + 2
    assert float(flat_scores[0, center_index]) > float(slope_scores[0, center_index])
    assert float(flat_scores[0, center_index]) > float(hole_scores[0, center_index])


def test_pretrain_attention_checkpoint_roundtrip(tmp_path, pretrain_attention_module):
    checkpoint_path = _make_pretrain_checkpoint(tmp_path, pretrain_attention_module)

    attention_state_dict, metadata = pretrain_attention_module.load_attention_checkpoint(checkpoint_path)
    model, restored_metadata = pretrain_attention_module.load_attention_pretrainer_model(checkpoint_path)

    assert metadata["scan_shape"] == (4, 4)
    assert restored_metadata["sampler"] == "unit_test"
    assert "terrain_encoder.encoder.0.weight" in attention_state_dict
    assert "cross_attention.k_in_proj_weight" in attention_state_dict
    assert not any("v_in_proj" in key or "out_proj" in key for key in attention_state_dict)
    assert not any(key.startswith("context_norm.") for key in attention_state_dict)

    height_scans = torch.randn(2, 16)
    _, attention_weights = model(height_scans)
    assert attention_weights.shape == (2, 4, 16)
    assert torch.allclose(attention_weights.sum(dim=-1), torch.ones_like(attention_weights.sum(dim=-1)), atol=1.0e-5)


def test_real_teacher_pretrain_checkpoint_loading(tmp_path, real_teacher_module, pretrain_attention_module):
    checkpoint_path = _make_pretrain_checkpoint(tmp_path, pretrain_attention_module)
    attention_state_dict, metadata = pretrain_attention_module.load_attention_checkpoint(checkpoint_path)

    obs, obs_groups = _make_obs(batch_size=4, terrain_dim=16)
    policy = _build_policy(
        real_teacher_module,
        obs,
        obs_groups,
        actor_hidden_dims=[64, 32],
        critic_hidden_dims=[64, 32],
        terrain_scan_shape=(4, 4),
        attention_embed_dim=32,
        attention_num_heads=4,
        terrain_encoder_hidden_dim=24,
        privileged_hidden_dims=[32],
        privileged_latent_dim=32,
        pretrained_attention_checkpoint=str(checkpoint_path),
    )

    assert policy.pretrained_attention_metadata["scan_shape"] == (4, 4)
    assert policy.pretrained_attention_metadata["sampler"] == metadata["sampler"]
    assert torch.allclose(
        policy.actor.terrain_encoder.encoder[0].weight.detach().cpu(),
        attention_state_dict["terrain_encoder.encoder.0.weight"],
    )
    assert torch.allclose(
        policy.critic.terrain_encoder.encoder[0].weight.detach().cpu(),
        attention_state_dict["terrain_encoder.encoder.0.weight"],
    )
    embed_dim = int(policy.actor.cross_attention.embed_dim)
    assert torch.allclose(
        policy.actor.cross_attention.in_proj_weight[embed_dim : 2 * embed_dim].detach().cpu(),
        attention_state_dict["cross_attention.k_in_proj_weight"],
    )


def test_legacy_unsupervised_attention_layers_are_not_transferred(real_teacher_module):
    checkpoint = {
        "attention_state_dict": {
            "terrain_encoder.encoder.0.weight": torch.randn(4, 3),
            "context_norm.weight": torch.randn(4),
            "cross_attention.out_proj.weight": torch.randn(4, 4),
            "cross_attention.kv_in_proj_weight": torch.randn(8, 4),
        }
    }

    state_dict, _ = real_teacher_module._extract_attention_state_dict(checkpoint)

    assert set(state_dict) == {
        "terrain_encoder.encoder.0.weight",
        "cross_attention.k_in_proj_weight",
    }
    torch.testing.assert_close(
        state_dict["cross_attention.k_in_proj_weight"],
        checkpoint["attention_state_dict"]["cross_attention.kv_in_proj_weight"][:4],
    )


def test_offline_attention_exports_only_parameters_with_nonzero_gradients(pretrain_attention_module):
    torch.manual_seed(12)
    model = pretrain_attention_module.TerrainAttentionPretrainer(
        scan_shape=(4, 4),
        attention_embed_dim=16,
        attention_num_heads=4,
        terrain_encoder_hidden_dim=12,
    )
    scans = torch.randn(8, 16)
    targets = torch.softmax(torch.randn(8, 16), dim=-1)

    loss, _ = model.compute_loss(scans, targets)
    loss.backward()

    embed_dim = model.attention_embed_dim
    in_proj_grad = model.cross_attention.in_proj_weight.grad
    assert in_proj_grad is not None
    assert float(in_proj_grad[:embed_dim].abs().sum()) > 0.0
    assert float(in_proj_grad[embed_dim : 2 * embed_dim].abs().sum()) > 0.0
    assert float(in_proj_grad[2 * embed_dim :].abs().sum()) == 0.0
    assert model.cross_attention.out_proj.weight.grad is None
    assert model.context_norm.weight.grad is None

    exported = model.export_attention_state_dict()
    assert "cross_attention.k_in_proj_weight" in exported
    assert not any("v_in_proj" in key or "out_proj" in key for key in exported)


def test_real_teacher_pretrain_freezes_only_supervised_terrain_encoder(
    tmp_path, real_teacher_module, pretrain_attention_module
):
    checkpoint_path = _make_pretrain_checkpoint(tmp_path, pretrain_attention_module)
    obs, obs_groups = _make_obs(batch_size=4, terrain_dim=16)
    policy = _build_policy(
        real_teacher_module,
        obs,
        obs_groups,
        actor_hidden_dims=[64, 32],
        critic_hidden_dims=[64, 32],
        terrain_scan_shape=(4, 4),
        attention_embed_dim=32,
        attention_num_heads=4,
        terrain_encoder_hidden_dim=24,
        privileged_hidden_dims=[32],
        privileged_latent_dim=32,
        pretrained_attention_checkpoint=str(checkpoint_path),
        freeze_pretrained_attention=True,
    )

    assert not policy.actor.terrain_encoder.encoder[0].weight.requires_grad
    assert policy.actor.context_norm.weight.requires_grad
    assert policy.actor.cross_attention.out_proj.weight.requires_grad

    actor_output, _ = policy.actor(
        proprio=obs["real_teacher_proprio"],
        terrain_scan=obs["real_teacher_terrain"],
        privileged_state=obs["real_teacher_privileged"],
        need_attention=True,
    )
    loss = actor_output.square().mean()
    policy.zero_grad()
    loss.backward()

    embed_dim = int(policy.actor.cross_attention.embed_dim)
    kv_grad = policy.actor.cross_attention.in_proj_weight.grad[embed_dim:]
    q_grad = policy.actor.cross_attention.in_proj_weight.grad[:embed_dim]
    assert float(kv_grad.abs().sum()) > 0.0
    assert float(q_grad.abs().sum()) > 0.0
    assert float(policy.actor.cross_attention.out_proj.weight.grad.abs().sum()) > 0.0


def test_real_teacher_visualization_outputs(tmp_path, real_teacher_module, real_teacher_viz_module):
    patterns = real_teacher_viz_module.generate_synthetic_terrain_patterns(16, 10)
    obs = real_teacher_viz_module.build_demo_observations(patterns)
    obs_groups = {
        "policy": ["real_teacher_proprio", "real_teacher_terrain", "real_teacher_privileged"],
        "critic": ["real_teacher_proprio", "real_teacher_terrain", "real_teacher_privileged"],
    }
    policy = _build_policy(real_teacher_module, obs, obs_groups)
    saved_paths = real_teacher_viz_module.save_attention_visualizations(
        policy=policy,
        obs=obs,
        output_dir=tmp_path,
        sample_names=list(patterns.keys()),
    )
    assert len(saved_paths) == len(patterns)
    assert all(path.is_file() for path in saved_paths)


def test_real_teacher_attention_marker_overlay(real_teacher_viz_module):
    attention_map = torch.zeros(4, 16, 10)
    attention_map[0, 11, 4] = 1.0
    attention_map[1, 9, 5] = 0.7
    attention_map[2, 6, 6] = 0.45
    marker_indices, marker_scales, point_strength = real_teacher_viz_module.compute_attention_marker_overlay(attention_map)

    assert marker_indices.shape == (160,)
    assert marker_scales.shape == (160, 3)
    assert point_strength.shape == (160,)
    assert marker_indices[11 * 10 + 4].item() == 0
    assert marker_indices[9 * 10 + 5].item() == 1
    assert marker_indices[6 * 10 + 6].item() == 2
    assert float(point_strength[11 * 10 + 4]) > float(point_strength[0])
    assert float(point_strength[9 * 10 + 5]) > float(point_strength[0])
    assert torch.all(marker_scales[:, 0] > 0.0)


def test_real_teacher_attention_marker_overlay_accepts_flat_attention(real_teacher_viz_module):
    rows, cols = real_teacher_viz_module.resolve_scan_shape(187, scan_shape=(16, 10))
    assert (rows, cols) == (17, 11)

    attention_map = torch.zeros(4, 187)
    focus_index = 8 * cols + 6
    attention_map[2, focus_index] = 1.0
    marker_indices, marker_scales, point_strength = real_teacher_viz_module.compute_attention_marker_overlay(
        attention_map,
        scan_shape=(16, 10),
    )

    assert marker_indices.shape == (187,)
    assert marker_scales.shape == (187, 3)
    assert point_strength.shape == (187,)
    assert marker_indices[focus_index].item() == 2
    assert float(point_strength[focus_index]) > float(point_strength[0])
    assert torch.all(marker_scales[:, 0] > 0.0)


def test_real_teacher_point_overlay_renderer(real_teacher_viz_module):
    terrain_map = real_teacher_viz_module.generate_synthetic_terrain_patterns(16, 10)["gap"]
    attention_map = torch.zeros(4, 16, 10)
    attention_map[0, 11, 4] = 1.0
    attention_map[1, 9, 5] = 0.7
    attention_map[2, 6, 6] = 0.45
    overlay = real_teacher_viz_module.render_attention_point_overlay(terrain_map, attention_map)

    assert overlay.ndim == 3
    assert overlay.shape[2] == 3
    assert overlay.dtype == real_teacher_viz_module.np.uint8
    assert overlay.shape[0] > terrain_map.shape[0]
    assert overlay.shape[1] > terrain_map.shape[1]
    assert overlay.max() > overlay.min()
    assert ((overlay[..., 0] != overlay[..., 1]) | (overlay[..., 1] != overlay[..., 2])).any()


def test_real_teacher_config_wiring_present():
    rough_env_cfg = ROUGH_ENV_CFG_PATH.read_text()
    real_sparse_env_cfg = REAL_SPARSE_ENV_CFG_PATH.read_text()
    real_beam_env_cfg = REAL_BEAM_ENV_CFG_PATH.read_text()
    start_terrains_cfg = START_TERRAINS_CFG_PATH.read_text()
    teacher_cfg = TEACHER_CFG_PATH.read_text()
    registry_cfg = REGISTRY_PATH.read_text()
    train_script = TRAIN_SCRIPT_PATH.read_text()
    pretrain_attention = PRETRAIN_ATTENTION_PATH.read_text()

    assert PRETRAIN_SCRIPT_PATH.is_file()
    assert REAL_DOCS_PATH.is_file()
    assert "self.observations.teacher_full = Go2RoughFullTeacherObservationsCfg()" in rough_env_cfg
    assert "self.observations.real_teacher_proprio = Go2RoughRealTeacherProprioObservationsCfg()" in rough_env_cfg
    assert "self.observations.real_teacher_terrain = Go2RoughRealTeacherTerrainObservationsCfg()" in rough_env_cfg
    assert "self.observations.real_teacher_privileged = Go2RoughRealTeacherPrivilegedObservationsCfg()" in rough_env_cfg
    assert "self.scene.terrain.terrain_generator = START_SPARSE_TERRAINS_CFG.replace(curriculum=True)" in real_sparse_env_cfg
    assert "self.scene.terrain.terrain_generator = REAL_BEAM_GAP_TERRAINS_CFG.replace(curriculum=True)" in real_beam_env_cfg
    assert "self.rewards.feet_height_body.weight = 0.0" in real_sparse_env_cfg
    assert "self.rewards.feet_gait.weight = 0.0" in real_sparse_env_cfg
    assert "self.rewards.feet_air_time_variance.weight = 0.0" in real_sparse_env_cfg
    assert "self.rewards.joint_mirror.weight = 0.0" in real_sparse_env_cfg
    assert "self.rewards.feet_height_body.weight = 0.0" in real_beam_env_cfg
    assert "self.rewards.feet_gait.weight = 0.0" in real_beam_env_cfg
    assert "self.rewards.feet_air_time_variance.weight = 0.0" in real_beam_env_cfg
    assert "self.rewards.joint_mirror.weight = 0.0" in real_beam_env_cfg
    assert "self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)" in real_beam_env_cfg
    assert "self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)" in real_beam_env_cfg
    assert "self.commands.base_velocity.ranges.ang_vel_z = (-0.4, 0.4)" in real_beam_env_cfg
    assert "gap_x_range=(0.35, 0.8)" in start_terrains_cfg
    assert "gap_width_range=(0.35, 0.85)" in start_terrains_cfg
    assert "class RealTeacherActorCriticCfg" in teacher_cfg
    assert "pretrained_attention_checkpoint" in teacher_cfg
    assert "class UnitreeGo2RoughRealTeacherPretrainedPPORunnerCfg" in teacher_cfg
    assert "class UnitreeGo2RoughRealTeacherFrozenPPORunnerCfg" in teacher_cfg
    assert '"rsl_rl_real_teacher_cfg_entry_point"' in registry_cfg
    assert '"rsl_rl_real_teacher_pretrained_cfg_entry_point"' in registry_cfg
    assert '"rsl_rl_real_teacher_frozen_cfg_entry_point"' in registry_cfg
    assert "Using pretrained REAL attention checkpoint" in train_script
    assert "class TerrainAttentionPretrainer" in pretrain_attention
    assert "def compute_traversability_from_heightmap" in pretrain_attention
    assert "Gurukul-Isaac-Velocity-Rough-Unitree-Go2-Real-Sparse-v0" in registry_cfg
    assert "Gurukul-Isaac-Velocity-Rough-Unitree-Go2-Real-Beam-v0" in registry_cfg
