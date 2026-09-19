from __future__ import annotations

import ast
import hashlib
import importlib.util
import re
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
VELOCITY_ROOT = REPO_ROOT / "source/Gurukul/Gurukul/tasks/manager_based/locomotion/velocity"
PM01_ROOT = VELOCITY_ROOT / "config/humanoid/engineai_pm01"
REGISTRY = PM01_ROOT / "__init__.py"
ENV_CFG = PM01_ROOT / "amp_env_cfg.py"
AGENT_CFG = PM01_ROOT / "agents/rsl_rl_amp_ppo_cfg.py"
ALGORITHM = VELOCITY_ROOT / "pm01_amp_ppo.py"
DATA_LOADER = VELOCITY_ROOT / "pm01_amp_data.py"
MOTION = PM01_ROOT / "motion/locomotion.npz"
UPSTREAM = PM01_ROOT / "UPSTREAM.md"
LICENSE = PM01_ROOT / "LICENSE.engineai_amp.txt"
DOCS = REPO_ROOT / "website/docs/tasks/velocity-locomotion/pm01.md"

EXPECTED_SHA256 = "945fa07b32c66410d2904e9f9147f4d433a69f9d688965c366092e71a311f6b4"


def _load_data_module():
    spec = importlib.util.spec_from_file_location("pm01_amp_data_for_test", DATA_LOADER)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pm01_amp_source_and_registration_are_well_formed():
    for path in (REGISTRY, ENV_CFG, AGENT_CFG, ALGORITHM, DATA_LOADER):
        ast.parse(path.read_text())

    registry = REGISTRY.read_text()
    task_id = "Gurukul-Isaac-Velocity-Flat-EngineAI-PM01-AMP-v0"
    assert registry.count(task_id) == 1
    assert "EngineAiPm01AmpFlatEnvCfg" in registry
    assert "EngineAiPm01AmpFlatPPORunnerCfg" in registry

    env_source = ENV_CFG.read_text()
    assert "EngineAiPm01FlatEnvCfg" in env_source
    assert "PM01_AMP_JOINT_ASSET_CFG" in env_source
    assert "PM01_AMP_REFERENCE_ARM_POSE" in env_source
    assert "self.history_length = 5" in env_source
    assert "self.enable_corruption = False" in env_source


def test_official_pm01_amp_reference_is_bundled_and_attributed():
    assert hashlib.sha256(MOTION.read_bytes()).hexdigest() == EXPECTED_SHA256
    assert "83ba64bbb58a02e14483e52adce5f893f3f31cdf" in UPSTREAM.read_text()
    assert "engineai-robotics/engineai_amp" in UPSTREAM.read_text()
    assert "BSD 3-Clause License" in LICENSE.read_text()

    with np.load(MOTION, allow_pickle=False) as motion:
        assert set(motion.files) == {
            "fps",
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        }
        assert motion["fps"].tolist() == [100]
        assert motion["joint_pos"].shape == (2303, 23)
        assert motion["body_pos_w"].shape == (2303, 24, 3)
        assert np.count_nonzero(motion["joint_pos"][:, 2]) == 0
        for key in motion.files:
            assert np.all(np.isfinite(motion[key])), key


def test_pm01_amp_reference_history_matches_the_50hz_policy_period():
    data = _load_data_module()
    dataset = data.Pm01AmpMotionDataset(
        MOTION,
        history_length=5,
        policy_dt=0.02,
        device="cpu",
    )

    assert dataset.fps == 100.0
    assert dataset.frame_stride == 2
    assert dataset.frame_dim == 26
    assert dataset.observation_dim == 130
    torch.testing.assert_close(
        dataset.window_indices(torch.tensor([8, 10])),
        torch.tensor([[0, 2, 4, 6, 8], [2, 4, 6, 8, 10]]),
    )
    assert dataset.sample(7).shape == (7, 130)


def test_pm01_amp_runner_preserves_velocity_task_reward_and_checkpoint_contract():
    agent_source = AGENT_CFG.read_text()
    algorithm_source = ALGORITHM.read_text()
    docs_source = DOCS.read_text()

    assert "EngineAiPm01FlatPPORunnerCfg" in agent_source
    # Upstream parity: 2.0 * style per second of simulated time, applied from iteration 0.
    assert "amp_style_reward_weight: float = 2.0" in agent_source
    assert "amp_style_reward_warmup_iterations: int = 0" in agent_source
    assert "amp_style_reward_ramp_iterations: int = 0" in agent_source
    assert "amp_reference_min_speed: float = 0.15" in agent_source
    assert "amp_batch_size: int = 16_384" in agent_source
    assert "amp_discriminator_hidden_dims: list[int] = [256, 128]" in agent_source
    assert 'self.experiment_name = "engineai_pm01_official_amp_flat"' in agent_source
    assert "task_reward = rewards.clone()" in algorithm_source
    assert "self.amp_policy_dt * effective_style_weight" in algorithm_source
    assert 'log_values["Metrics/amp_style_weight"]' in algorithm_source
    assert 'log_values["Metrics/amp_style_raw"]' in algorithm_source
    assert 'obs["amp"]' in algorithm_source
    assert '"amp_discriminator_state_dict"' in algorithm_source
    assert "--task=Gurukul-Isaac-Velocity-Flat-EngineAI-PM01-AMP-v0" in docs_source
    assert "--num_envs 16384" in docs_source
    assert "--logger wandb" in docs_source


def test_pm01_amp_reference_sampling_excludes_the_standing_tail():
    data = _load_data_module()
    unfiltered = data.Pm01AmpMotionDataset(MOTION, history_length=5, policy_dt=0.02, min_planar_speed=0.0)
    walking = data.Pm01AmpMotionDataset(MOTION, history_length=5, policy_dt=0.02, min_planar_speed=0.15)

    # The released clip closes with ~8.5 s of standing still; filtering must drop it.
    assert walking.num_sampled_windows < unfiltered.num_sampled_windows
    assert 0.4 < walking.num_sampled_windows / unfiltered.num_sampled_windows < 0.75

    base_lin_vel = walking.features[:, -3:] / 7.0
    sampled = walking.sampled_latest_frames
    assert torch.linalg.vector_norm(base_lin_vel[sampled, :2], dim=-1).min() >= 0.15
    assert walking.sample(64).shape == (64, 130)


def _literal_dict(source: Path, name: str, keyword: str | None = None) -> dict:
    """Read a dict literal out of a module that cannot be imported without Isaac Sim."""
    for node in ast.walk(ast.parse(source.read_text())):
        if keyword is None:
            target = getattr(node, "target", None) or (getattr(node, "targets", [None])[0])
            if isinstance(node, (ast.Assign, ast.AnnAssign)) and getattr(target, "id", None) == name:
                return ast.literal_eval(node.value)
        elif isinstance(node, ast.keyword) and node.arg == keyword and isinstance(node.value, ast.Dict):
            return ast.literal_eval(node.value)
    raise AssertionError(f"{name or keyword} not found in {source}")


def _resolve_pose(pose: dict[str, float], joint_names: list[str]) -> dict[str, float]:
    """Mirror ``resolve_matching_names_values``: every joint matches at most one key."""
    resolved: dict[str, float] = {}
    for joint in joint_names:
        hits = [key for key in pose if re.fullmatch(key, joint)]
        assert len(hits) <= 1, f"'{joint}' matches multiple keys: {hits}"
        if hits:
            resolved[joint] = pose[hits[0]]
    return resolved


def test_pm01_amp_nominal_pose_resolves_without_colliding_with_the_asset_defaults():
    """Isaac Lab rejects a joint that two init_state keys both match, at env construction."""
    joint_names = _literal_dict(PM01_ROOT / "pm01_constants.py", "PM01_POLICY_JOINT_NAMES")
    asset_pose = _literal_dict(REPO_ROOT / "source/Gurukul/Gurukul/assets/engineai_pm01_official.py", "", "joint_pos")
    arm_pose = _literal_dict(PM01_ROOT / "pm01_constants.py", "PM01_AMP_REFERENCE_ARM_POSE")

    merged = {**asset_pose, **arm_pose}
    resolved = _resolve_pose(merged, joint_names)
    assert set(arm_pose).issubset(merged)
    # Every arm joint the prior cares about must actually pick up an override.
    for joint in joint_names:
        if "SHOULDER" in joint or "ELBOW" in joint:
            assert joint in resolved, joint


def test_pm01_amp_nominal_arm_pose_matches_the_reference_clip():
    """The discriminator reads absolute joint positions, so the nominal poses must agree."""
    joint_names = _literal_dict(PM01_ROOT / "pm01_constants.py", "PM01_POLICY_JOINT_NAMES")
    arm_pose = _resolve_pose(_literal_dict(PM01_ROOT / "pm01_constants.py", "PM01_AMP_REFERENCE_ARM_POSE"), joint_names)

    # EngineAI stores the 23 legacy joints in PhysX breadth-first articulation order.
    reference_columns = {
        "J13_SHOULDER_PITCH_L": 5,
        "J18_SHOULDER_PITCH_R": 6,
        "J14_SHOULDER_ROLL_L": 9,
        "J19_SHOULDER_ROLL_R": 10,
        "J15_SHOULDER_YAW_L": 13,
        "J20_SHOULDER_YAW_R": 14,
        "J16_ELBOW_PITCH_L": 17,
        "J21_ELBOW_PITCH_R": 18,
        "J17_ELBOW_YAW_L": 21,
        "J22_ELBOW_YAW_R": 22,
    }
    assert set(arm_pose) == set(reference_columns)

    with np.load(MOTION, allow_pickle=False) as motion:
        joint_pos = motion["joint_pos"][143:1451]  # the walking segment

    for joint, column in reference_columns.items():
        spread = float(joint_pos[:, column].std())
        offset = abs(arm_pose[joint] - float(joint_pos[:, column].mean()))
        # Within two standard deviations, versus 8-15 sigma before the alignment.
        assert offset <= 2.0 * spread + 0.02, (joint, offset, spread)
