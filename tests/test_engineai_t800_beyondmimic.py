from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
T800_TASK = (
    REPO_ROOT / "source/Gurukul/Gurukul/tasks/manager_based/beyondmimic/config/engineai_t800"
)
MOTION = T800_TASK / "motion/dance_t800.npz"
DEPLOY_CONFIG = T800_TASK / "pretrained/deploy_config.yaml"
POLICY = T800_TASK / "pretrained/policy.onnx"
CHECKPOINT = T800_TASK / "pretrained/dance.pt"
T800_ASSET = REPO_ROOT / "source/Gurukul/data/Robots/engineai/t800/serial_t800.usd"
REGISTRY = T800_TASK / "__init__.py"
SIM2SIM = REPO_ROOT / "engineai-sim2real/RL_policy_runner/sim2sim/run_t800_beyondmimic_policy.py"


def test_t800_official_artifacts_have_the_expected_policy_contract():
    contract = yaml.safe_load(DEPLOY_CONFIG.read_text())
    joint_names = contract["joint_names"]

    assert len(joint_names) == 25
    assert len(set(joint_names)) == 25
    assert len(contract["default_joint_pos"]) == 25
    assert len(contract["joint_stiffness"]) == 25
    assert len(contract["joint_damping"]) == 25
    assert len(contract["action_scale"]) == 25
    assert contract["observation_names"] == [
        "command",
        "motion_anchor_ori_b",
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
        "actions",
    ]
    assert contract["observation_history_lengths"] == [1, 1, 1, 1, 1, 1]
    assert sum((50, 6, 3, 25, 25, 25)) == 134
    assert CHECKPOINT.stat().st_size > 1_000_000
    assert POLICY.stat().st_size > 100_000
    assert T800_ASSET.stat().st_size > 1_000


@pytest.mark.skipif(not MOTION.exists(), reason="Optional EngineAI motion archive is not distributed.")
def test_t800_motion_is_finite_and_matches_deployment_joint_order():
    contract = yaml.safe_load(DEPLOY_CONFIG.read_text())
    with np.load(MOTION, allow_pickle=False) as motion:
        assert motion["fps"].tolist() == [50]
        assert motion["joint_pos"].shape == (1054, 25)
        assert motion["joint_vel"].shape == (1054, 25)
        assert motion["body_pos_w"].shape == (1054, 30, 3)
        assert motion["body_quat_w"].shape == (1054, 30, 4)
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


def test_t800_onnx_shapes_match_native_sdk_observation_and_action_sizes():
    ort = pytest.importorskip("onnxruntime")
    session = ort.InferenceSession(str(POLICY), providers=["CPUExecutionProvider"])

    assert session.get_inputs()[0].shape == [1, 134]
    assert session.get_outputs()[0].shape == [1, 25]


def test_t800_tasks_and_sim2sim_entry_point_are_present():
    registry_source = REGISTRY.read_text()
    runner_source = SIM2SIM.read_text()

    assert "Gurukul-Isaac-BeyondMimic-Flat-EngineAI-T800-v0" in registry_source
    assert "Gurukul-Isaac-BeyondMimic-Flat-EngineAI-T800-Wo-State-Estimation-v0" in registry_source
    assert "Gurukul-Isaac-BeyondMimic-Flat-EngineAI-T800-Low-Freq-v0" in registry_source
    assert "ENGINEAI_NATIVE_SDK_ROOT" in runner_source
    assert 'choices=("native-sdk", "export")' in runner_source
    assert 'NATIVE_MODEL_RELATIVE = Path("assets/resource/t800.xml")' in runner_source
