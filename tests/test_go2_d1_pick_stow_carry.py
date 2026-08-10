from __future__ import annotations

import importlib.util
import re
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
GO2_APEX_CFG_ROOT = (
    REPO_ROOT
    / "source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2"
)
MOTION_PATH = GO2_APEX_CFG_ROOT / "motion/npz/go2_d1/pick_stow_carry.npz"
ROBOT_ONLY_MOTION_PATH = GO2_APEX_CFG_ROOT / "motion/npz/go2_d1/pick_stow_carry_robot_only.npz"
GO2_D1_ASSET_ROOT = REPO_ROOT / "source/Gurukul/data/Robots/unitree/go2_with_d1"


def test_go2_d1_hardware_gripper_calibration():
    constants_path = (
        REPO_ROOT
        / "source/Gurukul/Gurukul/tasks/manager_based/go2_apex/constants.py"
    )
    spec = importlib.util.spec_from_file_location("go2_apex_constants", constants_path)
    assert spec is not None and spec.loader is not None
    constants = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(constants)

    to_joint = constants.go2_d1_gripper_hardware_angle_to_joint_position
    to_hardware = constants.go2_d1_gripper_joint_position_to_hardware_angle
    assert to_joint(68.6) == 0.0
    assert to_joint(-28.7) == 0.033
    assert to_joint(100.0) == 0.0
    assert to_joint(-100.0) == 0.033
    np.testing.assert_allclose(to_hardware(0.0), 68.6, atol=1.0e-12)
    np.testing.assert_allclose(to_hardware(0.033), -28.7, atol=1.0e-12)
    np.testing.assert_allclose(to_hardware(0.00665), 48.99257575757576, atol=1.0e-12)


@pytest.mark.skipif(not MOTION_PATH.exists(), reason="Optional Go2+D1 motion data is not distributed.")
def test_pick_stow_carry_motion_contract():
    with np.load(MOTION_PATH, allow_pickle=False) as motion:
        assert float(motion["fps"]) == 50.0
        assert motion["joint_pos"].shape == motion["joint_vel"].shape == (2341, 18)
        assert motion["skill"].shape == (2341, 1)
        assert set(np.unique(motion["skill"]).tolist()) == {0, 2}
        assert motion["body_names"].tolist() == ["base", "FL_foot", "FR_foot", "RL_foot", "RR_foot"]
        assert motion["arm_ee_pos_w"].shape == (2341, 3)
        np.testing.assert_allclose(motion["body_pos_w"][0, 0, :2], [0.0, 0.0], atol=1.0e-6)
        assert motion["object_names"].tolist() == ["tall_grasp_box"]
        assert motion["object_shapes"].tolist() == ["box"]
        np.testing.assert_allclose(motion["object_size"], [[0.135, 0.055, 0.255]], atol=1.0e-6)
        np.testing.assert_allclose(
            motion["object_pos_w"][0, 0], [2.0321982, 0.0, 0.1275], atol=1.0e-6
        )
        assert motion["object_pos_w"].shape == (2341, 1, 3)
        assert motion["object_quat_w"].shape == (2341, 1, 4)
        assert motion["object_attached"].shape == (2341, 1)
        assert np.flatnonzero(np.diff(motion["object_attached"][:, 0].astype(np.int8))).tolist() == [639, 2039]
        expected_markers = {
            "approach_walk_end_frame": 150,
            "approach_settle_end_frame": 250,
            "pregrasp_frame": 525,
            "reach_end_frame": 600,
            "grasp_frame": 600,
            "grasp_hold_end_frame": 640,
            "stow_end_frame": 940,
            "depart_end_frame": 1040,
            "carry_walk_end_frame": 1590,
            "final_settle_end_frame": 1690,
            "place_end_frame": 2000,
            "release_frame": 2040,
            "retract_end_frame": 2340,
        }
        assert {key: int(motion[key]) for key in expected_markers} == expected_markers
        gripper_joint_pos = motion["gripper_joint_pos"]
        assert motion["gripper_joint_names"].tolist() == ["arm_7_1_joint", "arm_7_2_joint"]
        assert gripper_joint_pos.shape == (2341, 2)
        np.testing.assert_allclose(gripper_joint_pos[0], 0.0)
        np.testing.assert_allclose(gripper_joint_pos[640:2000], 0.00565, atol=1.0e-9)
        np.testing.assert_allclose(gripper_joint_pos[2040:], 0.0)
        assert np.all(np.diff(gripper_joint_pos[600:640, 0]) >= 0.0)
        assert np.all(np.diff(gripper_joint_pos[2000:2040, 0]) <= 0.0)
        np.testing.assert_allclose(gripper_joint_pos[:, 0], gripper_joint_pos[:, 1])
        binary_gripper = (gripper_joint_pos[:, 0] > 0.0).astype(np.int8)
        assert np.unique(binary_gripper).tolist() == [0, 1]
        assert np.flatnonzero(np.diff(binary_gripper)).tolist() == [600, 2039]


@pytest.mark.skipif(not MOTION_PATH.exists(), reason="Optional Go2+D1 motion data is not distributed.")
def test_pick_stow_carry_robot_only_motion_has_no_object_channels():
    with np.load(MOTION_PATH, allow_pickle=False) as source, np.load(
        ROBOT_ONLY_MOTION_PATH, allow_pickle=False
    ) as robot_only:
        removed = tuple(str(key) for key in robot_only["removed_object_channels"].tolist())
        assert removed == tuple(key for key in source.files if key.startswith("object_"))
        assert not any(key.startswith("object_") for key in robot_only.files)
        assert str(robot_only["robot_only_source"]) == MOTION_PATH.name
        for key in source.files:
            if key.startswith("object_"):
                continue
            np.testing.assert_array_equal(robot_only[key], source[key])


def test_pick_stow_carry_has_dedicated_task_and_log_namespace():
    env_source = (GO2_APEX_CFG_ROOT / "flat_d1_arm_tracker_env_cfg.py").read_text()
    registry_source = (GO2_APEX_CFG_ROOT / "__init__.py").read_text()
    runner_source = (GO2_APEX_CFG_ROOT / "agents/rsl_rl_ppo_cfg.py").read_text()
    constants_source = (
        REPO_ROOT
        / "source/Gurukul/Gurukul/tasks/manager_based/go2_apex/constants.py"
    ).read_text()
    command_source = (
        REPO_ROOT
        / "source/Gurukul/Gurukul/tasks/manager_based/go2_apex/mdp/commands.py"
    ).read_text()
    asset_source = (
        REPO_ROOT / "source/Gurukul/Gurukul/assets/unitree.py"
    ).read_text()

    assert "class UnitreeGo2D1ArmApexPickStowCarryFlatTrackerEnvCfg" in env_source
    assert 'self.commands.motion.motion_files = (f"{motion_root}/pick_stow_carry.npz",)' in env_source
    assert "Gurukul-Isaac-Go2-D1-Arm-APEX-Pick-Stow-Carry-Flat-Tracker-v0" in registry_source
    assert "class UnitreeGo2D1ArmApexPickStowCarryFlatTrackerPPORunnerCfg" in runner_source
    assert 'self.experiment_name = "unitree_go2_d1_arm_apex_pick_stow_carry"' in runner_source
    assert "self.clip_actions = 6.6" in runner_source
    assert 'GO2_D1_ACTION_JOINT_NAMES = GO2_D1_MOTION_JOINT_NAMES + ("arm_7_1_joint",)' in constants_source
    assert "GO2_D1_PICK_STOW_ACTION_JOINT_NAMES = GO2_D1_ACTION_JOINT_NAMES" in constants_source
    assert 'self.commands.motion.gripper_joint_names = ("arm_7_1_joint",)' in env_source
    assert "self.commands.motion.gripper_binary = True" in env_source
    assert "self.commands.motion.gripper_binary_threshold = 0.0" in env_source
    assert "self.commands.motion.gripper_closed_position = 0.033" in env_source
    assert "self.episode_length_s = 46.82" in env_source
    assert "_PICK_OBJECT_SIZE = (0.135, 0.055, 0.255)" in env_source
    assert "_PICK_OBJECT_SIZE_SCALE_RANGE = (0.70, 0.90)" in env_source
    assert "_PICK_OBJECT_MASS_RANGE = (0.015, 0.040)" in env_source
    assert "_PICK_OBJECT_NOMINAL_MASS = 0.0275" in env_source
    assert "spawn=sim_utils.CuboidCfg(" in env_source
    assert "size=_PICK_OBJECT_SIZE" in env_source
    assert "pos=(2.0321982, 0.0, 0.1275)" in env_source
    assert 'r"^arm_7_1_joint$": 0.005' in env_source
    assert "self.actions.joint_pos.reference_residual_joint_names = ()" in env_source
    assert "self.actions.joint_pos.decap_joint_names = tuple(GO2_D1_MOTION_JOINT_NAMES)" in env_source
    assert 'self.actions.joint_pos.binary_joint_names = ("arm_7_1_joint",)' in env_source
    assert "self.actions.joint_pos.binary_action_threshold = 0.0" in env_source
    assert "self.actions.joint_pos.binary_closed_position = 0.033" in env_source
    assert 'r"^arm_2_joint$": 0.50' in env_source
    assert 'r"^arm_3_joint$": 0.35' in env_source
    assert "self.actions.joint_pos.servo_joint_names = D1_COMMAND_JOINT_NAMES" in env_source
    assert "self.rewards.imitate_gripper_joint_pos = None" in env_source
    assert "self.rewards.imitate_gripper_policy_targets = None" in env_source
    assert "self.rewards.imitate_gripper_action = None" in env_source
    assert "self.rewards.imitate_gripper_logit = None" in env_source
    assert "self.rewards.imitate_gripper_state = RewTerm(" in env_source
    assert "self.actions.joint_pos.joint_names = tracked_joint_names" in env_source
    assert "self.scene.object = RigidObjectCfg(" in env_source
    assert "self.rewards.imitate_object_pos = RewTerm(" in env_source
    assert "gripper_joint_pos = gripper_joint_pos * self._motion_gripper_position_scale" in command_source
    assert "gripper_closed_position: float | None = None" in command_source
    assert "if self.cfg.gripper_binary:" in command_source
    assert "gripper_joint_pos > float(self.cfg.gripper_binary_threshold)" in command_source
    assert '"arm_1_joint": 0.0' in asset_source
    assert '"arm_2_joint": math.radians(-90.0)' in asset_source
    assert '"arm_3_joint": math.radians(90.0)' in asset_source
    assert '"arm_4_joint": 0.0' in asset_source
    assert '"arm_5_joint": 0.0' in asset_source
    assert '"arm_6_joint": 0.0' in asset_source
    assert '"arm_7_1_joint": 0.0' in asset_source
    assert '"arm_7_2_joint": 0.0' in asset_source
    assert "soft_joint_pos_limit_factor=1.0" in asset_source
    assert "GO2_D1_GRIPPER_OPEN_HARDWARE_ANGLE_DEG = 68.6" in constants_source
    assert "GO2_D1_GRIPPER_CLOSED_HARDWARE_ANGLE_DEG = -28.7" in constants_source


def test_pick_stow_carry_uses_manipulation_data_and_contact_supervision():
    env_source = (GO2_APEX_CFG_ROOT / "flat_d1_arm_tracker_env_cfg.py").read_text()
    command_source = (
        REPO_ROOT
        / "source/Gurukul/Gurukul/tasks/manager_based/go2_apex/mdp/commands.py"
    ).read_text()
    event_source = (
        REPO_ROOT
        / "source/Gurukul/Gurukul/tasks/manager_based/go2_apex/mdp/events.py"
    ).read_text()
    observation_source = (
        REPO_ROOT
        / "source/Gurukul/Gurukul/tasks/manager_based/go2_apex/mdp/observations.py"
    ).read_text()
    reward_source = (
        REPO_ROOT
        / "source/Gurukul/Gurukul/tasks/manager_based/go2_apex/mdp/rewards.py"
    ).read_text()
    termination_source = (
        REPO_ROOT
        / "source/Gurukul/Gurukul/tasks/manager_based/go2_apex/mdp/terminations.py"
    ).read_text()
    train_source = (REPO_ROOT / "scripts/reinforcement_learning/rsl_rl/train.py").read_text()

    assert 'self.object_attached = torch.tensor(' in command_source
    assert '"object_attached": object_attached' in command_source
    assert "def object_target_pos_w(" in command_source
    assert "def object_target_quat_w(" in command_source
    assert "def reference_object_position_trajectory_b(" in observation_source
    assert "def reference_object_attachment_phase(" in observation_source
    assert "def filtered_contact_state(" in observation_source
    assert "def motion_object_position_error_huber(" in reward_source
    assert "cost = cost.clamp_max(float(max_cost))" in reward_source
    assert "def motion_object_up_axis_error_exp(" in reward_source
    assert "def motion_object_linear_velocity_error_exp(" in reward_source
    assert "def motion_object_attachment_offset_error_exp(" in reward_source
    assert "def motion_object_attached_bilateral_contact(" in reward_source
    assert "def motion_object_attached_without_bilateral_contact(" in reward_source
    assert "def motion_binary_action_state_match(" in reward_source
    assert "def _motion_object_phase_scale(" in reward_source
    assert "def motion_clip_end(" in termination_source
    assert "class object_grasp_contact_timeout(ManagerTermBase):" in termination_source
    assert "should_be_gripped = attached[:, object_index].bool()" in termination_source
    assert "missing_grasp = should_be_gripped & (~bilateral_contact)" in termination_source
    assert "self._ungripped_time_s >= timeout_s" in termination_source
    assert "self.scene.d1_left_gripper_object_contact = ContactSensorCfg(" in env_source
    assert "self.scene.d1_right_gripper_object_contact = ContactSensorCfg(" in env_source
    assert "self.events.reset_object = EventTerm(" in env_source
    assert "self.events.randomize_object_material = EventTerm(" in env_source
    assert "self.events.randomize_object_mass = EventTerm(" in env_source
    assert "self.events.randomize_object_size = EventTerm(" in env_source
    assert "func=mdp.randomize_rigid_body_size_scale" in env_source
    assert "func=mdp.reset_root_state_uniform_with_size_scale" in env_source
    assert "def randomize_rigid_body_size_scale(" in event_source
    assert 'scale_range=size_scale_range' in event_source
    assert "def reset_root_state_uniform_with_size_scale(" in event_source
    assert '"static_friction_range": (1.10, 1.40)' in env_source
    assert '"dynamic_friction_range": (0.80, 1.00)' in env_source
    assert '"mass_distribution_params": _PICK_OBJECT_MASS_RANGE' in env_source
    assert '"operation": "abs"' in env_source
    assert '"x": (-0.0025, 0.0025)' in env_source
    assert '"yaw": (-0.05, 0.05)' in env_source
    assert "self.rewards.attached_bilateral_gripper_contact = RewTerm(" in env_source
    assert "self.rewards.attached_without_bilateral_gripper_contact = RewTerm(" in env_source
    assert "self.observations.policy.reference_object_attachment = ObsTerm(" in env_source
    assert "self.actions.joint_pos.decap_joint_names = tuple(GO2_D1_MOTION_JOINT_NAMES)" in env_source
    assert "self.commands.motion.reset_random_frame_probability = 0.20" in env_source
    assert "self.commands.motion.reset_attached_frame_probability = 0.30" in env_source
    assert "self.commands.motion.reference_state_init = True" in env_source
    assert "self.commands.motion.reference_object_state_init = True" in env_source
    assert 'self.commands.motion.reference_object_asset_name = "object"' in env_source
    assert "self.commands.motion.terminate_episode_at_motion_end = True" in env_source
    assert "self.rewards.imitate_world_base_pos.weight = 2.0" in env_source
    assert "self.rewards.imitate_world_base_pos_huber = RewTerm(" in env_source
    assert "func=mdp.motion_world_base_position_error_huber" in env_source
    assert '"max_cost": 1.5' in env_source
    assert "self.rewards.track_command_lin_vel_xy.weight = 1.5" in env_source
    assert "robot_anchor_pos_w=root_pos[env_ids]" in command_source
    assert "robot_anchor_quat_w=root_ori[env_ids]" in command_source
    assert "def object_size_center_offset_w(" in command_source
    assert "size_center_offset_w = command.object_size_center_offset_w(aligned_ref_quat_w)" in observation_source
    assert "axis_angle_from_quat(quat_mul(next_quat_w, quat_inv(prev_quat_w)))" in command_source
    assert "root_lin_vel[env_ids] = quat_apply(orientations_delta, root_lin_vel[env_ids])" in command_source
    assert "reference_state_use_recorded_gripper: bool = True" in command_source
    assert "self.events.disable_pick_reference_collision_pairs = EventTerm(" in env_source
    assert '("Robot/Head_upper", "Robot/d1/Link4")' in env_source
    assert '("Robot/Head_upper", "Robot/d1/Link5")' in env_source
    assert '("Robot/Head_upper", "Robot/d1/Link6")' in env_source
    assert '("Robot/d1/Link6", "Object")' in env_source
    assert "def disable_collision_pairs(" in event_source
    assert "candidate.HasAPI(UsdPhysics.CollisionAPI)" in event_source
    assert "weight=12.0" in env_source
    assert "weight=-2.0" in env_source
    assert '"max_cost": 0.25' in env_source
    assert '"detached_scale": 0.10' in env_source
    assert "self.terminations.motion_clip_end = DoneTerm(" in env_source
    assert "self.terminations.object_grasp_contact_timeout = DoneTerm(" in env_source
    assert "_PICK_GRASP_TERMINATION_WARMUP_ITERATIONS = 250" in env_source
    assert "_PICK_GRASP_TERMINATION_RAMP_END_ITERATION = 1000" in env_source
    assert "_PICK_GRASP_TIMEOUT_START_S = 2.0" in env_source
    assert "_PICK_GRASP_TIMEOUT_END_S = 0.5" in env_source
    assert '"object_grasp_contact_timeout",' in env_source
    assert 'self.terminations.illegal_contact.params["sensor_cfg"].body_names = ["base"]' in env_source
    assert "self.terminations.bad_anchor_height = DoneTerm(" in env_source
    assert "self.terminations.bad_anchor_orientation = DoneTerm(" in env_source
    assert "self.rewards.bad_tracking_termination = RewTerm(" in env_source
    assert "weight=-200.0" in env_source
    assert 'params["steps_per_iteration"] = steps_per_iteration' in train_source
    assert 'params["resume_iteration"] = max(0, int(resume_iteration))' in train_source


def test_pick_playback_disables_training_phase_resets_by_default():
    play_source = (REPO_ROOT / "scripts/reinforcement_learning/rsl_rl/play.py").read_text()

    assert '"--motion-reset-curriculum"' in play_source
    assert "if starts_at_beginning and not args_cli.motion_reset_curriculum:" in play_source
    assert "motion_cfg.reset_random_frame_probability = 0.0" in play_source
    assert "motion_cfg.reset_attached_frame_probability = 0.0" in play_source
    assert "def _configure_play_resume_aware_terminations(" in play_source
    assert "_configure_play_resume_aware_terminations(env_cfg, resume_path)" in play_source


def test_pick_privileged_teacher_is_a_separate_task_with_distillation_pair():
    env_source = (GO2_APEX_CFG_ROOT / "flat_d1_arm_tracker_env_cfg.py").read_text()
    registry_source = (GO2_APEX_CFG_ROOT / "__init__.py").read_text()
    teacher_source = (GO2_APEX_CFG_ROOT / "agents/rsl_rl_teacher_cfg.py").read_text()
    distillation_source = (GO2_APEX_CFG_ROOT / "agents/rsl_rl_distillation_cfg.py").read_text()
    observation_source = (
        REPO_ROOT
        / "source/Gurukul/Gurukul/tasks/manager_based/go2_apex/mdp/observations.py"
    ).read_text()

    direct_block = env_source.split(
        "class UnitreeGo2D1ArmApexPickStowCarryFlatTrackerEnvCfg", maxsplit=1
    )[1].split("def _configure_pick_privileged_manipulation_observations", maxsplit=1)[0]
    teacher_block = env_source.split(
        "class UnitreeGo2D1ArmApexPickStowCarryPrivilegedTeacherEnvCfg", maxsplit=1
    )[1].split("class UnitreeGo2D1ArmApexPickStowCarryDistillationStudentEnvCfg", maxsplit=1)[0]

    assert "_configure_pick_privileged_manipulation_observations(self)" not in direct_block
    assert "_configure_pick_privileged_manipulation_observations(self)" in teacher_block
    assert "_PICK_PRIVILEGED_REFERENCE_TIME_OFFSETS = (-20, -10, -5, -2, 0, 1, 2, 5, 10, 20, 40)" in env_source
    assert "privileged.history_length = 5" in env_source
    assert "privileged.flatten_history_dim = True" in env_source
    for term_name in (
        "object_orientation",
        "object_angular_velocity",
        "arm_end_effector_pose",
        "gripper_object_geometry",
        "left_gripper_object_force",
        "right_gripper_object_force",
        "object_mass",
        "object_size_scale",
        "reference_object_orientation",
        "reference_arm_ee_pose",
    ):
        assert f"privileged.{term_name} = ObsTerm(" in env_source
    for function_name in (
        "object_orientation_b",
        "object_angular_velocity_b",
        "body_pose_b",
        "gripper_object_geometry_b",
        "filtered_contact_force_b",
        "rigid_body_mass",
        "reference_object_orientation_trajectory_b",
        "reference_arm_ee_pose_trajectory_b",
    ):
        assert f"def {function_name}(" in observation_source

    assert "Gurukul-Isaac-Go2-D1-Arm-APEX-Pick-Stow-Carry-Privileged-Teacher-v0" in registry_source
    assert "UnitreeGo2D1ArmApexPickStowCarryPrivilegedTeacherEnvCfg" in registry_source
    assert "UnitreeGo2D1ArmApexPickStowCarryPrivilegedTeacherPPORunnerCfg" in teacher_source
    assert 'self.obs_groups = {"actor": ["privileged"], "critic": ["privileged"]}' in teacher_source
    assert 'self.experiment_name = "unitree_go2_d1_arm_apex_pick_stow_carry_privileged_teacher"' in teacher_source
    assert "Gurukul-Isaac-Go2-D1-Arm-APEX-Pick-Stow-Carry-Distillation-Student-v0" in registry_source
    assert "UnitreeGo2D1ArmApexPickStowCarryDistillationStudentEnvCfg" in registry_source
    assert "UnitreeGo2D1ArmApexPickStowCarryDistillationRunnerCfg" in distillation_source
    assert 'obs_groups = {"student": ["policy"], "teacher": ["privileged"]}' in distillation_source
    assert "train_with_zero_decap: bool = True" in env_source


def test_pick_stow_carry_robot_only_task_has_no_object_surface():
    env_source = (GO2_APEX_CFG_ROOT / "flat_d1_arm_tracker_env_cfg.py").read_text()
    registry_source = (GO2_APEX_CFG_ROOT / "__init__.py").read_text()
    runner_source = (GO2_APEX_CFG_ROOT / "agents/rsl_rl_ppo_cfg.py").read_text()
    robot_only_block = env_source.split(
        "class UnitreeGo2D1ArmApexPickStowCarryRobotOnlyFlatTrackerEnvCfg", maxsplit=1
    )[1].split("class UnitreeGo2D1ArmApexPickStowCarryFlatTrackerEnvCfg", maxsplit=1)[0]

    assert 'self.commands.motion.motion_files = (f"{motion_root}/pick_stow_carry_robot_only.npz",)' in robot_only_block
    assert "self.commands.motion.reference_object_state_init = False" in robot_only_block
    assert "self.commands.motion.reference_object_asset_name = None" in robot_only_block
    assert "self.commands.motion.reset_attached_frame_probability = 0.0" in robot_only_block
    assert "self.commands.motion.gripper_binary = False" in robot_only_block
    assert "self.actions.joint_pos.binary_joint_names = ()" in robot_only_block
    assert "self.commands.motion.reset_random_frame_probability = 0.50" in robot_only_block
    assert "self.rewards.imitate_world_base_pos_huber = RewTerm(" in robot_only_block
    assert "self.scene.object" not in robot_only_block
    assert "object_position" not in robot_only_block
    assert "object_contact" not in robot_only_block
    assert "randomize_object" not in robot_only_block
    assert "Gurukul-Isaac-Go2-D1-Arm-APEX-Pick-Stow-Carry-Robot-Only-Flat-Tracker-v0" in registry_source
    assert "UnitreeGo2D1ArmApexPickStowCarryRobotOnlyFlatTrackerEnvCfg" in registry_source
    assert "UnitreeGo2D1ArmApexPickStowCarryRobotOnlyPPORunnerCfg" in runner_source
    assert 'self.experiment_name = "unitree_go2_d1_arm_apex_pick_stow_carry_robot_only"' in runner_source


@pytest.mark.skipif(
    not (GO2_D1_ASSET_ROOT / "urdf/go2_d1_vis.urdf").is_file(),
    reason="Optional Go2+D1 robot assets are not distributed.",
)
def test_go2_d1_gripper_zero_pose_matches_hardware_urdf():
    hardware_urdf = (
        REPO_ROOT
        / "source/Gurukul/data/Robots/unitree/go2_with_d1/urdf/go2_d1_vis.urdf"
    )
    isaac_overlay = (
        REPO_ROOT
        / "source/Gurukul/data/Robots/unitree/go2_with_d1/usd/go2_d1_center_gripper.usda"
    ).read_text()

    urdf_root = ET.parse(hardware_urdf).getroot()
    hardware_joints = {
        joint.attrib["name"]: joint
        for joint in urdf_root.findall("joint")
        if joint.attrib.get("name") in {"arm_7_1_joint", "arm_7_2_joint"}
    }
    hardware_y = {
        name: float(joint.find("origin").attrib["xyz"].split()[1])
        for name, joint in hardware_joints.items()
    }
    np.testing.assert_allclose(
        [hardware_y["arm_7_1_joint"], hardware_y["arm_7_2_joint"]],
        [-0.033778, 0.033022],
        atol=1.0e-9,
    )
    np.testing.assert_allclose(
        hardware_y["arm_7_2_joint"] - hardware_y["arm_7_1_joint"],
        0.0668,
        atol=1.0e-9,
    )

    local_anchor_matches = re.findall(
        r"physics:localPos0 = \([^,]+, ([^,]+), [^)]+\)",
        isaac_overlay,
    )
    np.testing.assert_allclose(
        [float(value) for value in local_anchor_matches[-2:]],
        [hardware_y["arm_7_1_joint"], hardware_y["arm_7_2_joint"]],
        atol=1.0e-9,
    )
    assert isaac_overlay.count("float physics:upperLimit = 0.033") == 2
    assert isaac_overlay.count("float physics:lowerLimit = 0") == 2
    assert "quatf physics:localRot0 = (-0.5001518, 0.49984998, -0.50015, -0.49984816)" in isaac_overlay
    assert "quatf physics:localRot0 = (-0.49984998, 0.5001482, 0.4998518, 0.50014997)" not in isaac_overlay
    assert "float physxMimicJoint:rotX:gearing = -1" in isaac_overlay
    assert isaac_overlay.count("float physics:lowerLimit = -90") == 2
    assert isaac_overlay.count("float physics:upperLimit = 90") == 2


@pytest.mark.skipif(
    not all(
        (GO2_D1_ASSET_ROOT / path).is_file()
        for path in ("usd/d1/viser_gripper_left.usda", "usd/d1/viser_gripper_right.usda")
    ),
    reason="Optional Go2+D1 robot assets are not distributed.",
)
def test_go2_d1_isaac_uses_hardware_viser_finger_geometry():
    asset_root = (
        REPO_ROOT / "source/Gurukul/data/Robots/unitree/go2_with_d1"
    )
    overlay = (asset_root / "usd/go2_d1_center_gripper.usda").read_text()
    left_mesh = (asset_root / "usd/d1/viser_gripper_left.usda").read_text()
    right_mesh = (asset_root / "usd/d1/viser_gripper_right.usda").read_text()

    assert "references = @./d1/viser_gripper_left.usda@</Finger>" in overlay
    assert "references = @./d1/viser_gripper_right.usda@</Finger>" in overlay
    assert overlay.count(
        "quatd xformOp:orient = (0.7071067811865476, 0.7071067811865475, 0, 0)"
    ) == 2
    assert overlay.count(
        "quatd xformOp:orient = (0.7071067811865476, -0.7071067811865475, 0, 0)"
    ) == 2
    assert overlay.count(
        'prepend apiSchemas = ["PhysicsCollisionAPI", "PhysicsMeshCollisionAPI"]'
    ) == 2
    assert "(-0.0105000036, -0.0225000009, -0.0165000074)" in left_mesh
    assert "(0.0568000004, 0.000250000012, 0.00950000715)" in left_mesh
    assert "(-0.0105000036, -0.000250000012, -0.0165000074)" in right_mesh
    assert "(0.0568000004, 0.0225000009, 0.00950000715)" in right_mesh


def test_all_go2_d1_control_paths_use_hardware_positive_gripper_coordinates():
    go2_d1_cfg_root = (
        REPO_ROOT
        / "source/Gurukul/Gurukul/tasks/manager_based/locomotion/velocity/config/"
        "quadruped_with_arm/unitree_go2_d1_arm"
    )
    control_sources = "\n".join(path.read_text() for path in go2_d1_cfg_root.glob("*.py"))
    actions_source = (
        REPO_ROOT
        / "source/Gurukul/Gurukul/tasks/manager_based/locomotion/velocity/mdp/actions.py"
    ).read_text()
    play_source = (REPO_ROOT / "scripts/reinforcement_learning/rsl_rl/play.py").read_text()

    assert "_D1_GRIPPER_OPEN_POS = (0.0, 0.0)" in control_sources
    assert "_D1_GRIPPER_CLOSED_POS = (0.033, 0.033)" in control_sources
    assert "(0.025, -0.025)" not in control_sources
    assert "arm_7_2_joint\": (-0.03, 0.0)" not in control_sources
    assert "self._gripper_targets[:, 1] = -gripper" not in actions_source
    assert "self._gripper_targets[:, 1] = gripper" in actions_source
    assert "class PositiveBinaryJointPositionAction(" in actions_source
    assert "close_mask = actions > float(self.cfg.threshold)" in actions_source
    assert "float(self.cfg.gripper_scale) * 0.5" not in actions_source
    assert "env._gripper_target_pos[:, 1] = -float(state.gripper)" not in play_source
    assert "env._gripper_target_pos[:, 1] = float(state.gripper)" in play_source


def test_replay_tool_supports_reference_object_geometry():
    replay_source = (REPO_ROOT / "scripts/tools/go2_apex/replay_npz.py").read_text()

    assert '"--disable-object-vis"' in replay_source
    assert "def make_reference_object_visualizer()" in replay_source
    assert '"box": sim_utils.CuboidCfg(' in replay_source
    assert 'unsupported_shapes = sorted(set(object_shapes) - {"box", "cylinder"})' in replay_source
    assert 'shape_to_marker_index = {"cylinder": 0, "box": 1}' in replay_source
    assert '"object_pos_w": object_pos_w' in replay_source
    assert "object_visualizer.visualize(" in replay_source
    assert '"gripper_joint_pos": gripper_joint_pos' in replay_source
    assert '"--gripper-position-scale"' in replay_source
    assert '"--gripper-closed-position"' in replay_source
    assert '"--binary-gripper"' in replay_source
    assert '"--object-size-scale-range"' in replay_source
    assert '"--object-height-scale-range"' in replay_source
    assert "default=(0.70, 0.90)" in replay_source
    assert '"--object-size-seed"' in replay_source
    assert '"--object-height-seed"' in replay_source
    assert '"--object-mass-range"' in replay_source
    assert '"--num-envs"' in replay_source
    assert "np.linspace(size_low, size_high, scene.num_envs)" in replay_source
    assert "np.linspace(mass_low, mass_high, scene.num_envs)" in replay_source
    assert "nominal_object_size.unsqueeze(0) * object_size_scale[:, None, None]" in replay_source
    assert "object_pos_w += grasp_face_offset_w" in replay_source
    assert "object_pos_w[..., 2] += grounded_height_offset.unsqueeze(0)" in replay_source
    assert "ReplayGo2NpzSceneCfg(" in replay_source
    assert "num_envs=args_cli.num_envs, env_spacing=args_cli.env_spacing" in replay_source
    assert "def gripper_replay_target_at(" in replay_source
    assert "if args_cli.binary_gripper:" in replay_source
    assert '"--camera-focus"' in replay_source
    assert 'default="base"' in replay_source
    assert '"--drive-arm-with-pd"' in replay_source
    assert "GO2_D1_PD_REPLAY_SIM_DT = 0.005" in replay_source
    assert "GO2_D1_PD_REPLAY_CONTROL_DECIMATION = 4" in replay_source
    assert "robot.set_joint_position_target(replay_joint_pos)" in replay_source
    assert "joint_ids=kinematic_joint_indices" in replay_source
    assert "for _ in range(GO2_D1_PD_REPLAY_CONTROL_DECIMATION):" in replay_source
    assert "sim.step(render=False)" in replay_source
    assert "print_arm_joint_error_summary(" in replay_source
    assert 'f"[Replay] Fully-open frame {frame}: q=[0, 0] m, "' in replay_source
    assert "f\"visual inner-pad opening={1000.0 * inner_pad_opening:.3f} mm\"" in replay_source
    assert "robot.data.body_pos_w[:, gripper_body_indices].mean(dim=(0, 1))" in replay_source
    assert "robot.data.root_pos_w.mean(dim=0)" in replay_source
    assert 'if joint_name == "arm_7_2_joint":' not in replay_source
    assert "scale[index] *= -1.0" not in replay_source
    assert "gripper_target_pos, gripper_target_vel = gripper_replay_target_at(" in replay_source
