from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK = (
    REPO_ROOT
    / "source/Gurukul/Gurukul/tasks/manager_based/beyondmimic/config"
    / "engineai_pm01_24dof"
)
MOTION = TASK / "motion/dance.npz"
WALKING_MOTION = TASK / "motion/walking_24dof.npz"
POLICY = TASK / "pretrained/policy.onnx"
CHECKPOINT = TASK / "pretrained/dance.pt"
DEPLOY_CONFIG = TASK / "pretrained/deploy_config.yaml"
HARDWARE_CONTRACT = TASK / "hardware/pm01_24dof_contract.yaml"
ASSET = (
    REPO_ROOT
    / "source/Gurukul/data/Robots/engineai/pm01_24dof/serial_pm01_edu.usd"
)
REGISTRY = TASK / "__init__.py"
ENV_CFG = TASK / "flat_env_cfg.py"
SIM2SIM = (
    REPO_ROOT
    / "engineai-sim2real/RL_policy_runner/sim2sim/run_pm01_beyondmimic_policy.py"
)


def test_official_pm01_24dof_artifacts_match_the_deployment_contract():
    contract = yaml.safe_load(DEPLOY_CONFIG.read_text())

    assert len(contract["joint_names"]) == 24
    assert len(set(contract["joint_names"])) == 24
    assert "J23_HEAD_YAW" in contract["joint_names"]
    assert contract["observation_names"] == [
        "command",
        "motion_anchor_ori_b",
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
        "actions",
    ]
    assert contract["observation_history_lengths"] == [1, 1, 1, 1, 1, 1]
    for key in (
        "default_joint_pos",
        "joint_stiffness",
        "joint_damping",
        "action_scale",
    ):
        assert len(contract[key]) == 24
    assert ASSET.stat().st_size > 1_000
    assert CHECKPOINT.stat().st_size > 1_000_000
    assert POLICY.stat().st_size > 100_000


@pytest.mark.skipif(not MOTION.exists(), reason="Optional EngineAI motion archive is not distributed.")
def test_official_pm01_dance_motion_is_finite_and_name_aligned():
    contract = yaml.safe_load(DEPLOY_CONFIG.read_text())
    with np.load(MOTION, allow_pickle=False) as motion:
        assert motion["fps"].tolist() == [50]
        assert motion["joint_pos"].shape == (1222, 24)
        assert motion["joint_vel"].shape == (1222, 24)
        assert motion["body_pos_w"].shape == (1222, 29, 3)
        assert motion["body_quat_w"].shape == (1222, 29, 4)
        assert motion["joint_names"].tolist() == contract["joint_names"]
        for key in (
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        ):
            assert np.all(np.isfinite(motion[key])), key


def test_official_pm01_onnx_is_129_observations_by_24_actions():
    ort = pytest.importorskip("onnxruntime")
    session = ort.InferenceSession(str(POLICY), providers=["CPUExecutionProvider"])

    assert session.get_inputs()[0].shape == [1, 129]
    assert session.get_outputs()[0].shape == [1, 24]


@pytest.mark.skipif(not WALKING_MOTION.exists(), reason="Optional EngineAI motion archive is not distributed.")
def test_pm01_walking_motion_adds_zero_head_yaw_in_policy_order():
    contract = yaml.safe_load(DEPLOY_CONFIG.read_text())
    with np.load(WALKING_MOTION, allow_pickle=False) as motion:
        assert motion["fps"].tolist() == [50]
        assert motion["joint_pos"].shape == (1152, 24)
        assert motion["joint_vel"].shape == (1152, 24)
        assert motion["body_pos_w"].shape == (1152, 29, 3)
        assert motion["body_quat_w"].shape == (1152, 29, 4)
        assert motion["joint_names"].tolist() == contract["joint_names"]
        assert motion["joint_names"][7] == "J23_HEAD_YAW"
        assert np.count_nonzero(motion["joint_pos"][:, 7]) == 0
        assert np.count_nonzero(motion["joint_vel"][:, 7]) == 0
        assert len(motion["body_names"]) == 29
        for key in (
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        ):
            assert np.all(np.isfinite(motion[key])), key


def test_pm01_hardware_contract_matches_the_official_deploy_parameters():
    deploy = yaml.safe_load(DEPLOY_CONFIG.read_text())
    hardware = yaml.safe_load(HARDWARE_CONTRACT.read_text())
    policy_to_hardware = hardware["policy_to_hardware_index"]
    hardware_to_policy = hardware["hardware_to_policy_index"]

    assert sorted(policy_to_hardware) == list(range(24))
    assert [hardware_to_policy[index] for index in policy_to_hardware] == list(range(24))
    assert [
        hardware["hardware_joint_names"][index] for index in policy_to_hardware
    ] == hardware["policy_joint_names"]
    assert hardware["policy_joint_names"] == deploy["joint_names"]
    assert hardware["control"]["observation_dim"] == 129
    assert hardware["control"]["action_dim"] == 24
    assert hardware["timing"]["policy_frequency_hz"] == 50
    assert hardware["walking_reference"]["head_yaw_policy_index"] == 7
    assert hardware["walking_reference"]["head_yaw_position_rad"] == 0.0
    for key in (
        "default_joint_pos",
        "joint_stiffness",
        "joint_damping",
        "action_scale",
    ):
        np.testing.assert_allclose(hardware[key], deploy[key], rtol=0, atol=1.0e-6)


def test_official_pm01_task_keeps_head_action_and_joint_velocity_noise():
    registry_source = REGISTRY.read_text()
    env_source = ENV_CFG.read_text()
    runner_source = SIM2SIM.read_text()

    assert "PM01-24DoF-Wo-State-Estimation-v0" in registry_source
    assert registry_source.count("PM01-24DoF-Walking-v0") == 1
    assert registry_source.count("WalkingBeyondMimicFlatWoStateEstimationEnvCfg") == 1
    assert env_source.count("class EngineAiPm0124DofWalkingBeyondMimicFlatEnvCfg") == 1
    assert "PM01_24DOF_POLICY_JOINT_NAMES" in env_source
    assert "walking_24dof.npz" in env_source
    assert 'joint_noise_scales={".*ANKLE.*": 3.0}' in env_source
    assert "self.rewards.joint_acc_l2 = None" in env_source
    assert "self.rewards.joint_torques_l2 = None" in env_source
    assert "input_meta.shape != [1, 129]" in runner_source
    assert "output_meta.shape != [1, 24]" in runner_source
