from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
GO2_APEX_CFG_ROOT = (
    REPO_ROOT
    / "source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2"
)
MOTION_PATH = GO2_APEX_CFG_ROOT / "motion/npz/go2_d1/top_down_can_pick_and_move.npz"


@pytest.mark.skipif(not MOTION_PATH.exists(), reason="Optional Go2+D1 motion data is not distributed.")
def test_can_pick_carry_drop_motion_contract():
    with np.load(MOTION_PATH, allow_pickle=False) as motion:
        assert float(motion["fps"]) == 50.0
        assert motion["joint_pos"].shape == motion["joint_vel"].shape == (1141, 18)
        assert motion["motion_name"].item() == "top_down_floor_can_pick_then_walk"
        assert motion["object_names"].tolist() == ["slim_cola_can"]
        assert motion["object_shapes"].tolist() == ["cylinder"]
        np.testing.assert_allclose(motion["object_size"], [[0.053, 0.053, 0.135]], atol=1.0e-6)
        np.testing.assert_allclose(motion["object_pos_w"][0, 0], [0.48, 0.0, 0.0675], atol=1.0e-6)
        assert motion["object_attached"].shape == (1141, 1)
        assert np.flatnonzero(np.diff(motion["object_attached"][:, 0].astype(np.int8))).tolist() == [389, 1129]
        assert int(motion["grasp_frame"]) == 350
        assert int(motion["grasp_hold_end_frame"]) == 390
        assert int(motion["carry_walk_end_frame"]) == 990
        assert int(motion["gripper_open_start_frame"]) == 1090
        assert int(motion["release_frame"]) == 1130
        assert int(motion["drop_end_frame"]) == 1140
        assert float(motion["drop_duration"]) == np.float32(0.2)
        np.testing.assert_allclose(motion["body_pos_w"][990, 0, 0], 1.0378104, atol=1.0e-6)
        assert motion["object_pos_w"][-1, 0, 2] < motion["object_pos_w"][1130, 0, 2]


def test_can_pick_carry_drop_has_dedicated_task_and_domain_randomization():
    env_source = (GO2_APEX_CFG_ROOT / "flat_d1_arm_tracker_env_cfg.py").read_text()
    registry_source = (GO2_APEX_CFG_ROOT / "__init__.py").read_text()
    runner_source = (GO2_APEX_CFG_ROOT / "agents/rsl_rl_ppo_cfg.py").read_text()
    command_source = (
        REPO_ROOT
        / "source/Gurukul/Gurukul/tasks/manager_based/go2_apex/mdp/commands.py"
    ).read_text()
    event_source = (
        REPO_ROOT
        / "source/Gurukul/Gurukul/tasks/manager_based/go2_apex/mdp/events.py"
    ).read_text()

    assert "class UnitreeGo2D1ArmApexCanPickCarryDropFlatTrackerEnvCfg" in env_source
    assert 'self.commands.motion.motion_files = (f"{motion_root}/top_down_can_pick_and_move.npz",)' in env_source
    assert "spawn=sim_utils.CylinderCfg(" in env_source
    assert "_CAN_OBJECT_SIZE = (0.053, 0.053, 0.135)" in env_source
    assert "_CAN_OBJECT_SIZE_SCALE_RANGE = (0.92, 1.08)" in env_source
    assert "_CAN_OBJECT_MASS_RANGE = (0.18, 0.32)" in env_source
    assert "_CAN_OBJECT_NOMINAL_MASS = 0.25" in env_source
    assert "self.commands.motion.object_size_preserve_grasp_face = False" in env_source
    assert '"preserve_grasp_face": False' in env_source
    assert '"x": (-0.01, 0.01)' in env_source
    assert '"y": (-0.01, 0.01)' in env_source
    assert '"static_friction_range": (1.05, 1.35)' in env_source
    assert '"dynamic_friction_range": (0.75, 1.00)' in env_source
    assert '"restitution_range": (0.0, 0.08)' in env_source
    assert 'self.actions.joint_pos.scale[r"^arm_3_joint$"] = 0.375' in env_source
    assert "self.episode_length_s = 22.82" in env_source
    assert "object_size_preserve_grasp_face: bool = True" in command_source
    assert "if self.cfg.object_size_preserve_grasp_face:" in command_source
    assert "preserve_grasp_face: bool = True" in event_source
    assert "if preserve_grasp_face:" in event_source

    task_id = "Gurukul-Isaac-Go2-D1-Arm-APEX-Can-Pick-Carry-Drop-Flat-Tracker-v0"
    assert task_id in registry_source
    assert "UnitreeGo2D1ArmApexCanPickCarryDropFlatTrackerPPORunnerCfg" in runner_source
    assert 'self.experiment_name = "unitree_go2_d1_arm_apex_can_pick_carry_drop"' in runner_source
