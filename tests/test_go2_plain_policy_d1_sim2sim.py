from __future__ import annotations

from pathlib import Path

import numpy as np
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = (
    REPO_ROOT
    / "unitree-sim2real/RL_policy_runner/configs/Gurukul/"
    "go2_velocity_flat_v0_with_d1_motion.yaml"
)
RUNNER_PATH = REPO_ROOT / "unitree-sim2real/RL_policy_runner/sim2sim/run_rl_policy.py"


def test_plain_go2_policy_keeps_its_actor_contract_on_the_d1_model():
    cfg = yaml.safe_load(CONFIG_PATH.read_text())

    assert cfg["policy_path"].endswith(
        "unitree_go2_flat/2026-03-12_19-03-13/exported/policy.onnx"
    )
    assert cfg["xml_path"].endswith("go2_with_d1/scene_flat.xml")
    assert cfg["observation_layout"] == "isaac_45"
    assert cfg["num_obs"] == 45
    assert cfg["num_actions"] == 12
    assert cfg["num_joint_obs"] == 12
    assert cfg["num_motors"] == 19
    assert cfg["controlled_motor_indices"] == list(range(12))
    assert cfg["joint_obs_mapping"] == list(range(12))
    assert cfg["joint_mapping"] == list(range(12))

    # The actor sees only the original Go2 terms:
    # angular velocity, gravity, velocity command, leg q/dq, and prior leg action.
    assert 3 + 3 + 3 + 12 + 12 + 12 == cfg["num_obs"]


def test_d1_motion_is_independent_from_the_plain_go2_actor():
    cfg = yaml.safe_load(CONFIG_PATH.read_text())
    arm = cfg["arm_command"]
    motion = cfg["motion_visualization"]

    assert arm["enabled"] is True
    assert arm["source"] == "motion_reference"
    assert arm["indices"] == list(range(12, 19))
    assert arm["reference_indices"] == list(range(12, 19))
    assert arm["gripper_index"] == 18
    assert arm["model"] == "second_order_angle"
    assert motion["enabled"] is True
    assert motion["drive_command"] is False
    assert motion["motion_file"].endswith("go2_d1/wave_hello.npz")
    assert len(cfg["tracker_joint_names"]) == 19
    assert cfg["tracker_joint_names"][-1] == "arm_7_1_joint"

    np.testing.assert_allclose(
        cfg["default_angles"][12:18],
        np.deg2rad([0.0, -90.0, 90.0, 0.0, 0.0, 0.0]),
        atol=1.0e-10,
    )
    assert len(cfg["kps"]) == len(cfg["kds"]) == len(cfg["default_angles"]) == 19
    assert cfg["kps"][12:18] == [200.0, 200.0, 200.0, 50.0, 50.0, 50.0]
    assert cfg["kds"][12:18] == [5.0, 5.0, 4.0, 0.25, 0.25, 0.25]


def test_runner_registers_and_applies_the_independent_d1_motion_source():
    source = RUNNER_PATH.read_text()

    assert '"go2_velocity_flat_v0_with_d1_motion": {' in source
    assert '"flat_v0_with_d1_motion": "go2_velocity_flat_v0_with_d1_motion"' in source
    assert 'arm_command_source not in {"command", "motion_reference"}' in source
    assert "def apply_arm_command_source(self, cmd):" in source
    assert "reference_q = self.motion_runtime.current_joint_position_reference()" in source
    assert "reference_idx = int(arm_reference_indices[local_idx])" in source
    assert "rl_policy.apply_arm_command_source(cmd)" in source
