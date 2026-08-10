"""Command-conditioned whole-body controller tasks for Go2+D1."""

# SPDX-License-Identifier: Apache-2.0

import copy

from isaaclab.managers import (
    CurriculumTermCfg,
    EventTermCfg,
    SceneEntityCfg,
)
from isaaclab.managers import (
    ObservationTermCfg as ObsTerm,
)
from isaaclab.managers import (
    RewardTermCfg as RewTerm,
)
from isaaclab.utils import configclass

import Gurukul.tasks.manager_based.locomotion.velocity.mdp as mdp
from Gurukul.assets.unitree import GO2_D1_ARM_JOINT_DEFAULTS

from ...quadruped.unitree_go2.mjlab_parity_constants import GO2_MJLAB_PARITY_ACTION_SCALE
from .flat_env_cfg import UnitreeGo2D1ArmFlatEnvCfg
from .rough_env_cfg import D1_ARM_HARDWARE_VELOCITY_LIMITS, UnitreeGo2D1ArmRoughEnvCfg

# Reset with the arm folded over the body to minimize the base disturbance. The
# Cartesian deployment route first lifts above the body keep-out, reaches forward
# while high, then lowers at the workspace-ready point.
GO2_D1_WBC_CARRY_POSE = (0.0, -1.15, 1.35, 0.0, -0.30, 0.0)
GO2_D1_WBC_WORKSPACE_READY_POSE = (0.0, 0.58, 0.02, 0.0, -0.38, 0.0)
GO2_D1_WBC_DEPLOYMENT_EE_WAYPOINTS = (
    (0.0911, 0.0, 0.3894),
    (0.4200, 0.0, 0.4200),
    (0.4159, 0.0, 0.3381),
)
_D1_STOW_READY_POSE = GO2_D1_WBC_CARRY_POSE
_D1_ASYNC_READY_POSE = GO2_D1_WBC_CARRY_POSE
GO2_D1_ARM_DEFAULT_JOINT_POS = GO2_D1_ARM_JOINT_DEFAULTS
_ARM_EE_COMMAND_ATTR = "_arm_ee_target_pos"
_ARM_EE_GOAL_ATTR = "_arm_ee_goal_pos"
GO2_D1_WBC_EE_POS_RANGE = ((0.10, 0.56), (-0.40, 0.40), (0.18, 0.65))
GO2_D1_WBC_WORKSPACE_ORIGIN = (0.0, 0.0, 0.08)
GO2_D1_WBC_REACH_RANGE = (0.12, 0.58)
GO2_D1_WBC_BODY_EXCLUSION_BOX = ((-0.30, 0.34), (-0.20, 0.20), (-0.02, 0.30))
GO2_D1_WBC_BODY_CLEARANCE = 0.07
GO2_D1_WBC_NEUTRAL_EE_POS = (0.42, 0.0, 0.34)
GO2_D1_WBC_BODY_HEIGHT_RANGE = (0.28, 0.39)
GO2_D1_WBC_BODY_NOMINAL_HEIGHT = 0.33
GO2_D1_WBC_BODY_PITCH_RANGE = (-0.16, 0.12)
GO2_D1_SAFE_ARM_JOINT_RANGES = {
    "arm_1_joint": (-1.30, 1.30),
    "arm_2_joint": (-1.25, 1.30),
    "arm_3_joint": (-0.65, 1.55),
    "arm_4_joint": (-0.45, 0.45),
    "arm_5_joint": (-0.60, 0.40),
    "arm_6_joint": (-1.00, 1.00),
}
GO2_D1_IK_ARM_JOINT_RANGES = {
    "arm_1_joint": (-2.12, 2.12),
    "arm_2_joint": (-1.41, 1.41),
    "arm_3_joint": (-1.41, 1.41),
    "arm_4_joint": (-2.12, 2.12),
    "arm_5_joint": (-1.41, 1.41),
    "arm_6_joint": (-2.12, 2.12),
}
GO2_D1_WBC_JOINT_CLIP = {
    r"^(FL|FR|RL|RR)_(hip|thigh|calf)_joint$": (-100.0, 100.0),
    r"^arm_[146]_joint$": (-2.12, 2.12),
    r"^arm_[235]_joint$": (-1.41, 1.41),
    r"^arm_7_1_joint$": (0.0, 0.03),
    r"^arm_7_2_joint$": (0.0, 0.03),
}
GO2_D1_MANIP_ARM_JOINT_RANGES = {
    "arm_1_joint": (-1.35, 1.35),
    "arm_2_joint": (-1.40, 1.40),
    "arm_3_joint": (-1.40, 1.40),
    "arm_4_joint": (-1.40, 1.40),
    "arm_5_joint": (-1.35, 1.35),
    "arm_6_joint": (-1.80, 1.80),
}
GO2_D1_SAFE_ARM_WAYPOINTS = (
    _D1_ASYNC_READY_POSE,
    GO2_D1_WBC_WORKSPACE_READY_POSE,
    # Position-IK solutions sampled across the boundary and interior of the
    # configured Cartesian workspace. Wrist rotation is varied independently
    # because it does not change the Link6 position target.
    (1.2800, 0.8830, 0.0911, 0.0742, -0.1725, -0.80),
    (-1.2800, 0.8826, 0.0913, -0.0751, -0.1704, 0.80),
    (1.0289, -0.1929, 0.2511, 0.0350, 0.0090, 0.55),
    (-0.0041, -0.5943, 0.4097, -0.0005, 0.3580, -0.55),
    (-1.0331, -0.1932, 0.2514, -0.0364, 0.0088, 0.55),
    (0.9163, 1.1643, -0.4040, 0.1559, -0.4272, -0.80),
    (-0.9182, 1.1643, -0.4043, -0.1563, -0.4255, 0.80),
    (0.9122, 0.9198, -0.5154, 0.1857, -0.5167, 0.55),
    (-0.9141, 0.9194, -0.5146, -0.1858, -0.5170, -0.55),
    (0.3906, 1.2638, -0.5852, 0.0487, -0.4742, -0.80),
    (-0.0010, 1.0937, -0.2654, 0.0001, -0.4073, 0.80),
    (-0.3925, 1.2638, -0.5851, -0.0483, -0.4749, -0.55),
    (-0.0010, 0.8311, -0.3667, 0.0002, -0.4897, 0.55),
)
GO2_D1_WBC_ARM_ACTION_SCALE = {
    r"^arm_1_joint$": 2.20,
    r"^arm_2_joint$": 3.00,
    r"^arm_3_joint$": 3.00,
    r"^arm_4_joint$": 2.20,
    r"^arm_5_joint$": 1.45,
    r"^arm_6_joint$": 2.20,
    r"^arm_7_[12]_joint$": 0.03,
}
GO2_D1_MJLAB_ACTION_SCALE = {
    **GO2_MJLAB_PARITY_ACTION_SCALE,
    **GO2_D1_WBC_ARM_ACTION_SCALE,
}


def _set_reward_joint_names(env_cfg, joint_names: list[str]) -> None:
    """Scope inherited posture/action regularizers to selected joints."""
    for reward_name in (
        "joint_torques_l2",
        "joint_acc_l2",
        "joint_pos_limits",
        "joint_power",
        "stand_still",
        "joint_pos_penalty",
    ):
        reward = getattr(env_cfg.rewards, reward_name, None)
        if reward is not None and "asset_cfg" in reward.params:
            reward.params["asset_cfg"].joint_names = joint_names
            if reward_name == "joint_pos_limits":
                reward.func = mdp.joint_pos_limits_with_soft_factor
                reward.params["soft_factor"] = 0.9


def _configure_go2_d1_wbc_policy_observations(env_cfg, arm_cfg: SceneEntityCfg, gripper_cfg: SceneEntityCfg) -> None:
    """Expose the deployable WBC command interface on the policy observation group."""
    env_cfg.observations.policy.velocity_commands = ObsTerm(
        func=mdp.velocity_posture_commands,
        params={"command_name": "base_velocity"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    env_cfg.observations.policy.arm_ee_command_pos = ObsTerm(
        func=mdp.arm_ee_target_pos_b,
        params={"target_attr": _ARM_EE_COMMAND_ATTR},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    env_cfg.observations.policy.arm_ee_command_error = ObsTerm(
        func=mdp.arm_ee_target_error_b,
        params={"asset_cfg": arm_cfg, "target_attr": _ARM_EE_COMMAND_ATTR},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    env_cfg.observations.policy.gripper_command_pos = ObsTerm(
        func=mdp.arm_joint_target_rel,
        params={"asset_cfg": gripper_cfg, "target_attr": "_gripper_target_pos"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    env_cfg.observations.policy.gripper_command_error = ObsTerm(
        func=mdp.arm_joint_target_error,
        params={"asset_cfg": gripper_cfg, "target_attr": "_gripper_target_pos"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )

    # Remove privileged / training-only terms from the deployable actor interface.
    for term_name in (
        "arm_ee_target_pos",
        "arm_ee_goal_pos",
        "arm_ee_target_error",
        "arm_ik_joint_target_pos",
        "arm_ik_joint_target_error",
        "arm_trajectory_state",
        "gripper_target_pos",
        "gripper_target_error",
        "arm_motion_state",
        "arm_apex_reference",
    ):
        if hasattr(env_cfg.observations.policy, term_name):
            setattr(env_cfg.observations.policy, term_name, None)


def _configure_go2_d1_wbc_critic_observations(env_cfg, arm_cfg: SceneEntityCfg, gripper_cfg: SceneEntityCfg) -> None:
    """Keep privileged IK and curriculum signals on the critic only."""
    env_cfg.observations.critic.arm_ee_command_pos = copy.deepcopy(env_cfg.observations.policy.arm_ee_command_pos)
    env_cfg.observations.critic.arm_ee_command_error = copy.deepcopy(env_cfg.observations.policy.arm_ee_command_error)
    env_cfg.observations.critic.gripper_command_pos = copy.deepcopy(env_cfg.observations.policy.gripper_command_pos)
    env_cfg.observations.critic.gripper_command_error = copy.deepcopy(env_cfg.observations.policy.gripper_command_error)
    env_cfg.observations.critic.arm_ee_interp_pos = ObsTerm(
        func=mdp.arm_ee_target_pos_b,
        params={"target_attr": "_arm_ee_target_pos"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    env_cfg.observations.critic.arm_ik_joint_target_pos = ObsTerm(
        func=mdp.arm_joint_target_rel,
        params={"asset_cfg": arm_cfg, "target_attr": "_arm_ik_joint_target_pos"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    env_cfg.observations.critic.arm_ik_joint_target_error = ObsTerm(
        func=mdp.arm_joint_target_error,
        params={"asset_cfg": arm_cfg, "target_attr": "_arm_ik_joint_target_pos"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    env_cfg.observations.critic.arm_trajectory_state = ObsTerm(
        func=mdp.arm_ee_trajectory_state,
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    env_cfg.observations.critic.arm_motion_state = ObsTerm(
        func=mdp.arm_motion_state,
        clip=(-100.0, 100.0),
        scale=1.0,
    )


def _configure_go2_d1_paper_low_level_critic_observations(env_cfg) -> None:
    """Add paper-style privileged critic observations for low-level loco-manipulation."""
    height_sensor_cfg = None
    foot_height_sensor_cfg = None
    if getattr(env_cfg.scene, "height_scanner_base", None) is not None:
        height_sensor_cfg = SceneEntityCfg("height_scanner_base")
    if getattr(env_cfg.scene, "height_scanner", None) is not None:
        foot_height_sensor_cfg = SceneEntityCfg("height_scanner")
    foot_contact_cfg = SceneEntityCfg("contact_forces", body_names=[env_cfg.foot_link_name])
    foot_asset_cfg = SceneEntityCfg("robot", body_names=[env_cfg.foot_link_name])

    env_cfg.observations.critic.base_height = ObsTerm(
        func=mdp.base_height,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[env_cfg.base_link_name]),
            "sensor_cfg": height_sensor_cfg,
        },
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    if foot_height_sensor_cfg is not None:
        env_cfg.observations.critic.feet_height_scan = ObsTerm(
            func=mdp.feet_heightmap_scan,
            params={
                "height_sensor_cfg": foot_height_sensor_cfg,
                "asset_cfg": foot_asset_cfg,
                "patch_radius": 0.10,
            },
            clip=(-100.0, 100.0),
            scale=1.0,
        )
    env_cfg.observations.critic.feet_air_stance_time = ObsTerm(
        func=mdp.feet_air_stance_time,
        params={"sensor_cfg": foot_contact_cfg},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    env_cfg.observations.critic.feet_contact_forces = ObsTerm(
        func=mdp.contact_forces_b,
        params={
            "sensor_cfg": foot_contact_cfg,
            "asset_cfg": SceneEntityCfg("robot", body_names=[env_cfg.base_link_name]),
            "normalize": 100.0,
        },
        clip=(-100.0, 100.0),
        scale=1.0,
    )


def configure_go2_d1_whole_body_controller(env_cfg) -> None:
    """Configure a low-level command policy for later hierarchical loco-manipulation."""
    arm_cfg = SceneEntityCfg("robot", joint_names=env_cfg.arm_joint_names, body_names=["Link6"], preserve_order=True)
    gripper_cfg = SceneEntityCfg("robot", joint_names=env_cfg.gripper_joint_names, preserve_order=True)

    # Low-level commands should look like what a later high-level policy can emit:
    # bounded base velocity/posture plus D1 end-effector goal position and gripper targets.
    env_cfg.commands.base_velocity = mdp.UniformVelocityBodyPostureCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 6.0),
        rel_standing_envs=0.25,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        roll_range=(0.0, 0.0),
        pitch_range=(0.0, 0.0),
        height_range=(GO2_D1_WBC_BODY_NOMINAL_HEIGHT, GO2_D1_WBC_BODY_NOMINAL_HEIGHT),
        nominal_height=GO2_D1_WBC_BODY_NOMINAL_HEIGHT,
        zero_posture_probability=0.20,
        ranges=mdp.UniformVelocityBodyPostureCommandCfg.Ranges(
            lin_vel_x=(-0.45, 0.55),
            lin_vel_y=(-0.20, 0.20),
            ang_vel_z=(-0.45, 0.45),
            heading=(0.0, 0.0),
        ),
    )

    env_cfg.scene.robot.init_state.joint_pos.update(
        {
            "arm_1_joint": GO2_D1_WBC_CARRY_POSE[0],
            "arm_2_joint": GO2_D1_WBC_CARRY_POSE[1],
            "arm_3_joint": GO2_D1_WBC_CARRY_POSE[2],
            "arm_4_joint": GO2_D1_WBC_CARRY_POSE[3],
            "arm_5_joint": GO2_D1_WBC_CARRY_POSE[4],
            "arm_6_joint": GO2_D1_WBC_CARRY_POSE[5],
            "arm_7_1_joint": 0.0,
            "arm_7_2_joint": 0.0,
        }
    )
    env_cfg.actions.joint_pos.joint_names = list(env_cfg.joint_names)
    env_cfg.actions.joint_pos.preserve_order = True
    env_cfg.actions.joint_pos.scale = {
        r"^(FL|FR|RL|RR)_hip_joint$": 0.125,
        r"^(FL|FR|RL|RR)_(thigh|calf)_joint$": 0.25,
        **GO2_D1_WBC_ARM_ACTION_SCALE,
    }
    env_cfg.actions.joint_pos.clip = dict(GO2_D1_WBC_JOINT_CLIP)

    _configure_go2_d1_wbc_policy_observations(env_cfg, arm_cfg, gripper_cfg)
    _configure_go2_d1_wbc_critic_observations(env_cfg, arm_cfg, gripper_cfg)
    _configure_go2_d1_paper_low_level_critic_observations(env_cfg)

    carry_pose = GO2_D1_WBC_CARRY_POSE
    arm_command_params = {
        "asset_cfg": arm_cfg,
        "start_motion_enabled": False,
        "smooth_motion": True,
        "interpolate": True,
        "visualize": True,
        "ee_pos_range": GO2_D1_WBC_EE_POS_RANGE,
        "workspace_origin": GO2_D1_WBC_WORKSPACE_ORIGIN,
        "reach_range": GO2_D1_WBC_REACH_RANGE,
        "body_exclusion_box": GO2_D1_WBC_BODY_EXCLUSION_BOX,
        "body_clearance": GO2_D1_WBC_BODY_CLEARANCE,
        # Sample complete task-space goals. Smooth cubic interpolation provides
        # continuous references without shrinking the workspace into a local walk.
        "max_pos_change": None,
        "max_pos_change_range": None,
        "motion_speed": 0.32,
        "motion_speed_range": (0.12, 0.32),
        "motion_primitives": ("workspace", "reach", "pick_place", "sweep", "stow"),
        "global_workspace_probability": 0.55,
        "min_arm_motion_difficulty": 0.05,
        "preserve_current_orientation": False,
        "neutral_pos": (0.42, 0.0, 0.34),
        "deployment_ee_waypoints": GO2_D1_WBC_DEPLOYMENT_EE_WAYPOINTS,
        "min_workspace_fraction": 0.0,
        "ik_controller_cfg": {"command_type": "position", "ik_method": "dls", "dls_damping": 0.06},
        "joint_position_ranges": GO2_D1_IK_ARM_JOINT_RANGES,
        "gripper_joint_names": tuple(env_cfg.gripper_joint_names),
        "gripper_open_pos": (0.0, 0.0),
        "gripper_closed_pos": (0.033, 0.033),
        "gripper_close_probability": 0.35,
    }
    env_cfg.events.reset_arm_joint_state = EventTermCfg(
        func=mdp.reset_joints_by_absolute_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=env_cfg.arm_joint_names + env_cfg.gripper_joint_names,
                preserve_order=True,
            ),
            "position_ranges": {
                **{joint_name: (value, value) for joint_name, value in zip(env_cfg.arm_joint_names, carry_pose)},
                **{joint_name: (0.0, 0.0) for joint_name in env_cfg.gripper_joint_names},
            },
            "velocity_range": (0.0, 0.0),
        },
    )
    env_cfg.events.reset_arm_command = EventTermCfg(
        func=mdp.random_arm_ik_motion,
        mode="reset",
        params={**arm_command_params, "reset_deployment": True},
    )
    env_cfg.events.randomize_arm_command = EventTermCfg(
        func=mdp.random_arm_ik_motion,
        mode="interval",
        interval_range_s=(3.0, 5.0),
        params=dict(arm_command_params),
    )
    env_cfg.events.advance_arm_command = EventTermCfg(
        func=mdp.continuous_arm_ik_tracking,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={"asset_cfg": arm_cfg, "apply_target": False},
    )

    _set_reward_joint_names(env_cfg, env_cfg.leg_joint_names)
    env_cfg.rewards.joint_pos_penalty.weight = -0.12
    env_cfg.rewards.joint_pos_penalty.params["stand_still_scale"] = 2.0
    env_cfg.rewards.joint_pos_penalty.params["posture_nominal_height"] = GO2_D1_WBC_BODY_NOMINAL_HEIGHT
    env_cfg.rewards.stand_still.params["posture_nominal_height"] = GO2_D1_WBC_BODY_NOMINAL_HEIGHT
    env_cfg.rewards.track_lin_vel_xy_exp.weight = 2.0
    env_cfg.rewards.track_ang_vel_z_exp.weight = 1.0
    env_cfg.rewards.track_base_roll_pitch_exp = RewTerm(
        func=mdp.track_base_roll_pitch_exp,
        weight=0.45,
        params={"command_name": "base_velocity", "std": 0.18},
    )
    env_cfg.rewards.track_base_height_command = RewTerm(
        func=mdp.track_base_height_command_exp,
        weight=0.35,
        params={
            "command_name": "base_velocity",
            "std": 0.12,
            "sensor_cfg": env_cfg.rewards.base_height_l2.params.get("sensor_cfg"),
        },
    )
    env_cfg.rewards.arm_ee_target_tracking = RewTerm(
        func=mdp.arm_ee_target_tracking_exp,
        weight=2.5,
        params={"asset_cfg": arm_cfg, "target_attr": _ARM_EE_COMMAND_ATTR, "std": 0.14},
    )
    env_cfg.rewards.arm_ik_joint_target_tracking = None
    env_cfg.rewards.arm_joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": arm_cfg},
    )
    env_cfg.rewards.arm_joint_default = None
    env_cfg.rewards.arm_ee_target_error = None
    env_cfg.rewards.gripper_target_tracking = RewTerm(
        func=mdp.arm_joint_target_tracking_exp,
        weight=0.30,
        params={"asset_cfg": gripper_cfg, "target_attr": "_gripper_target_pos", "std": 0.012},
    )
    env_cfg.rewards.gripper_default = RewTerm(
        func=mdp.joint_error_l2,
        weight=-0.01,
        params={"asset_cfg": gripper_cfg},
    )
    env_cfg.rewards.action_rate_l2.func = mdp.action_rate_l2_after_reset
    env_cfg.rewards.action_rate_l2.weight = -0.01
    env_cfg.rewards.action_smoothness_l2.weight = -0.005
    env_cfg.rewards.arm_action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2_selected,
        weight=-0.08,
        params={"action_indices": tuple(range(12, 20))},
    )
    env_cfg.rewards.arm_action_smoothness_l2 = RewTerm(
        func=mdp.action_smoothness_l2_selected,
        weight=-0.04,
        params={"action_indices": tuple(range(12, 20))},
    )
    env_cfg.rewards.arm_joint_velocity = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-5.0e-4,
        params={"asset_cfg": arm_cfg},
    )
    env_cfg.rewards.arm_joint_acceleration = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-1.0e-6,
        params={"asset_cfg": arm_cfg},
    )

    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.command_levels_ang_vel = None
    # Time-based schedule: brief locomotion warmup, stationary-base arm tracking,
    # then combined walking + EE tracking. Arm difficulty ramps linearly to full workspace.
    env_cfg.curriculum.loco_manipulation_training_stages = CurriculumTermCfg(
        func=mdp.loco_manipulation_training_stages,
        params={
            "stage_steps": (40_000, 100_000),
            "arm_difficulty_steps": (40_000, 160_000),
            "stage_iteration_bins": (0.10, 0.25),
            "arm_difficulty_iteration_bins": (0.10, 1.0),
            "walking_lin_vel_x": (-0.45, 0.55),
            "walking_lin_vel_y": (-0.20, 0.20),
            "walking_ang_vel_z": (-0.45, 0.45),
            "walking_roll": (0.0, 0.0),
            "walking_pitch": GO2_D1_WBC_BODY_PITCH_RANGE,
            "walking_height": GO2_D1_WBC_BODY_HEIGHT_RANGE,
            "nominal_height": GO2_D1_WBC_BODY_NOMINAL_HEIGHT,
            "command_name": "base_velocity",
            "stage1_reward_terms": ("arm_ee_target_tracking",),
            "stage2_reward_terms": ("arm_ee_target_tracking",),
            "arm_difficulty_reward_terms": ("arm_ee_target_tracking",),
        },
    )
    env_cfg.curriculum.go2_d1_wbc_arm_difficulty = CurriculumTermCfg(
        func=mdp.env_attr_curriculum_metric,
        params={"attr_name": "_loco_manip_arm_motion_difficulty"},
    )


def configure_go2_d1_wbc_hierarchical_runtime(env_cfg) -> None:
    """Move deployable WBC observations into a dedicated group for frozen-policy execution."""
    env_cfg.observations.wbc_policy = copy.deepcopy(env_cfg.observations.policy)
    env_cfg.actions.joint_pos = None
    env_cfg.actions.wbc_command = mdp.FrozenGo2D1WbcCommandActionCfg(
        asset_name="robot",
        joint_names=list(env_cfg.joint_names),
        wbc_obs_group="wbc_policy",
        policy_path=getattr(env_cfg, "wbc_policy_path", ""),
        velocity_scale=(1.0, 1.0, 1.0),
        ee_pos_scale=(1.0, 1.0, 1.0),
        ee_pos_range=GO2_D1_WBC_EE_POS_RANGE,
        workspace_origin=GO2_D1_WBC_WORKSPACE_ORIGIN,
        reach_range=GO2_D1_WBC_REACH_RANGE,
        body_exclusion_box=GO2_D1_WBC_BODY_EXCLUSION_BOX,
        body_clearance=GO2_D1_WBC_BODY_CLEARANCE,
        body_pitch_range=GO2_D1_WBC_BODY_PITCH_RANGE,
        body_height_range=GO2_D1_WBC_BODY_HEIGHT_RANGE,
        body_nominal_height=GO2_D1_WBC_BODY_NOMINAL_HEIGHT,
        deployment_ee_waypoints=GO2_D1_WBC_DEPLOYMENT_EE_WAYPOINTS,
        deployment_motion_speed=0.18,
        deployment_min_segment_duration_s=1.0,
        gripper_scale=0.033,
        joint_action_scales={
            r"^(FL|FR|RL|RR)_hip_joint$": 0.125,
            r"^(FL|FR|RL|RR)_(thigh|calf)_joint$": 0.25,
            **GO2_D1_WBC_ARM_ACTION_SCALE,
        },
        joint_position_ranges={
            **GO2_D1_IK_ARM_JOINT_RANGES,
            "arm_7_1_joint": (0.0, 0.03),
            "arm_7_2_joint": (0.0, 0.03),
        },
        use_default_offset=True,
        clip={".*": (-100.0, 100.0)},
        preserve_order=True,
    )
    env_cfg.observations.wbc_policy.actions = ObsTerm(
        func=mdp.frozen_wbc_low_level_actions,
        params={"action_name": "wbc_command"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    # The frozen WBC consumes Cartesian EE commands directly. The differential-IK
    # event only maintains privileged reference joints for stage-1 WBC training;
    # leaving it active here adds an unused per-step CUDA linear solve and can make
    # hierarchical playback fail while creating a cuSolver handle.
    env_cfg.events.advance_arm_command = None
    env_cfg.events.randomize_arm_command = None
    _disable_go2_d1_wbc_training_curricula(env_cfg)


def _disable_go2_d1_wbc_training_curricula(env_cfg) -> None:
    """Remove low-level WBC curricula and diagnostics from frozen hierarchical runtimes."""
    for term_name in (
        "loco_manipulation_training_stages",
        "go2_d1_wbc_arm_difficulty",
        "go2_d1_wbc_arm_difficulty_frontier",
        "go2_d1_leg_wbc_locomotion_score",
        "go2_d1_leg_wbc_posture_score",
        "go2_d1_leg_wbc_combined_posture_score",
        "go2_d1_leg_wbc_difficulty_score",
        "go2_d1_leg_wbc_velocity_diagnostics",
        "go2_d1_leg_wbc_arm_motion_diagnostics",
        "go2_d1_leg_wbc_action_diagnostics",
        "go2_d1_leg_wbc_joint_limit_diagnostics",
        "go2_d1_leg_wbc_payload_mass",
    ):
        if hasattr(env_cfg.curriculum, term_name):
            setattr(env_cfg.curriculum, term_name, None)


def configure_go2_d1_leg_wbc_async_arm_controller(env_cfg) -> None:
    """Train a leg-only Go2 controller against externally driven asynchronous D1 joint motion."""
    arm_cfg = SceneEntityCfg(
        "robot",
        joint_names=env_cfg.arm_joint_names,
        body_names=["Link6"],
        preserve_order=True,
    )

    env_cfg.commands.base_velocity = mdp.UniformVelocityBodyPostureCommandCfg(
        asset_name="robot",
        resampling_time_range=(4.0, 6.0),
        rel_standing_envs=0.20,
        rel_heading_envs=0.0,
        heading_command=False,
        heading_control_stiffness=0.5,
        debug_vis=True,
        roll_range=(0.0, 0.0),
        pitch_range=(0.0, 0.0),
        height_range=(GO2_D1_WBC_BODY_NOMINAL_HEIGHT, GO2_D1_WBC_BODY_NOMINAL_HEIGHT),
        nominal_height=GO2_D1_WBC_BODY_NOMINAL_HEIGHT,
        curriculum_difficulty_attr="_loco_manip_arm_motion_difficulty",
        curriculum_frontier_difficulty_attr="_loco_manip_arm_motion_frontier",
        zero_posture_probability=0.20,
        ranges=mdp.UniformVelocityBodyPostureCommandCfg.Ranges(
            lin_vel_x=(-1.50, 1.50),
            lin_vel_y=(-0.50, 0.50),
            ang_vel_z=(-1.50, 1.50),
            heading=(0.0, 0.0),
        ),
    )
    env_cfg.observations.policy.velocity_commands = ObsTerm(
        func=mdp.velocity_posture_commands,
        params={"command_name": "base_velocity"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    env_cfg.observations.critic.velocity_commands = copy.deepcopy(env_cfg.observations.policy.velocity_commands)
    env_cfg.observations.policy.joint_pos.params["asset_cfg"].joint_names = env_cfg.joint_names
    env_cfg.observations.policy.joint_vel.params["asset_cfg"].joint_names = env_cfg.leg_joint_names
    env_cfg.observations.critic.joint_pos.params["asset_cfg"].joint_names = env_cfg.joint_names
    env_cfg.observations.critic.joint_vel.params["asset_cfg"].joint_names = env_cfg.joint_names
    _configure_go2_d1_paper_low_level_critic_observations(env_cfg)

    env_cfg.actions.joint_pos.scale = {
        r"^(FL|FR|RL|RR)_hip_joint$": 0.125,
        r"^(FL|FR|RL|RR)_(thigh|calf)_joint$": 0.25,
    }
    env_cfg.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
    env_cfg.actions.joint_pos.joint_names = env_cfg.leg_joint_names

    ready = _D1_ASYNC_READY_POSE
    env_cfg.scene.robot.init_state.joint_pos.update(
        {
            "arm_1_joint": ready[0],
            "arm_2_joint": ready[1],
            "arm_3_joint": ready[2],
            "arm_4_joint": ready[3],
            "arm_5_joint": ready[4],
            "arm_6_joint": ready[5],
            "arm_7_1_joint": 0.0,
            "arm_7_2_joint": 0.0,
        }
    )
    env_cfg.events.reset_arm_joint_state = EventTermCfg(
        func=mdp.reset_joints_by_absolute_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=env_cfg.arm_joint_names + env_cfg.gripper_joint_names,
                preserve_order=True,
            ),
            "position_ranges": {
                **{joint_name: (value, value) for joint_name, value in zip(env_cfg.arm_joint_names, ready)},
                **{joint_name: (0.0, 0.0) for joint_name in env_cfg.gripper_joint_names},
            },
            "velocity_range": (0.0, 0.0),
        },
    )
    env_cfg.events.randomize_arm_payload_mass = EventTermCfg(
        func=mdp.randomize_end_effector_payload_mass,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["Link6"]),
            "payload_mass_range": (0.0, 0.4),
            "recompute_inertia": True,
            "difficulty_attr": "_loco_manip_arm_motion_difficulty",
        },
    )
    arm_motion_params = {
        "asset_cfg": arm_cfg,
        "joint_position_ranges": GO2_D1_SAFE_ARM_JOINT_RANGES,
        "nominal_joint_pos": ready,
        "safe_waypoints": GO2_D1_SAFE_ARM_WAYPOINTS,
        "waypoint_probability": 0.80,
        "waypoint_jitter_fraction": 0.04,
        "max_joint_change_range": (0.15, 2.60),
        "trajectory_duration_range": (0.35, 1.40),
        "joint_velocity_limits": D1_ARM_HARDWARE_VELOCITY_LIMITS,
        "max_velocity_fraction": 0.98,
        "velocity_fraction_range": (0.35, 0.98),
        "start_motion_enabled": False,
        "apply_target_when_disabled": True,
        "gripper_joint_names": tuple(env_cfg.gripper_joint_names),
        "gripper_open_pos": (0.0, 0.0),
        "gripper_closed_pos": (0.033, 0.033),
        "gripper_close_probability": 0.35,
    }
    env_cfg.events.reset_arm_async_motion = EventTermCfg(
        func=mdp.random_async_arm_joint_motion,
        mode="reset",
        params={**arm_motion_params, "reset_to_nominal": True},
    )
    env_cfg.events.randomize_arm_async_motion = EventTermCfg(
        func=mdp.random_async_arm_joint_motion,
        mode="interval",
        interval_range_s=(4.0, 6.0),
        params=dict(arm_motion_params),
    )
    env_cfg.events.advance_arm_async_motion = EventTermCfg(
        func=mdp.continuous_async_arm_joint_tracking,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={
            "asset_cfg": arm_cfg,
            "apply_target": True,
            "apply_gripper_target": True,
            "gripper_joint_names": tuple(env_cfg.gripper_joint_names),
        },
    )

    _set_reward_joint_names(env_cfg, env_cfg.leg_joint_names)
    env_cfg.rewards.joint_pos_penalty.weight = -0.12
    env_cfg.rewards.joint_pos_penalty.params["stand_still_scale"] = 2.0
    env_cfg.rewards.joint_pos_penalty.params["posture_nominal_height"] = GO2_D1_WBC_BODY_NOMINAL_HEIGHT
    env_cfg.rewards.stand_still.params["posture_nominal_height"] = GO2_D1_WBC_BODY_NOMINAL_HEIGHT
    env_cfg.rewards.track_lin_vel_xy_exp.weight = 2.0
    env_cfg.rewards.track_ang_vel_z_exp.weight = 1.0
    env_cfg.rewards.track_base_roll_pitch_exp = RewTerm(
        func=mdp.track_base_roll_pitch_exp,
        weight=0.60,
        params={"command_name": "base_velocity", "std": 0.18},
    )
    env_cfg.rewards.track_base_height_command = RewTerm(
        func=mdp.track_base_height_command_exp,
        weight=0.45,
        params={
            "command_name": "base_velocity",
            "std": 0.12,
            "sensor_cfg": env_cfg.rewards.base_height_l2.params.get("sensor_cfg"),
        },
    )
    env_cfg.rewards.arm_joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-1.0,
        params={"asset_cfg": arm_cfg},
    )

    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.command_levels_ang_vel = None
    # Preserve the proven locomotion skill while expanding posture and arm-motion difficulty only
    # after the preceding stationary/combined mode is tracking its commands. Alternating stage 1
    # and stage 2 prevents a time-triggered jump from changing arm motion, body posture, and walking
    # at once.
    env_cfg.curriculum.loco_manipulation_training_stages = CurriculumTermCfg(
        func=mdp.loco_manipulation_training_stages,
        params={
            "walking_lin_vel_x": (-1.50, 1.50),
            "walking_lin_vel_y": (-0.50, 0.50),
            "walking_ang_vel_z": (-1.50, 1.50),
            "walking_roll": (0.0, 0.0),
            "walking_pitch": GO2_D1_WBC_BODY_PITCH_RANGE,
            "walking_height": GO2_D1_WBC_BODY_HEIGHT_RANGE,
            "nominal_height": GO2_D1_WBC_BODY_NOMINAL_HEIGHT,
            "command_name": "base_velocity",
            "performance_based": True,
            "stage0_reward_terms": ("track_lin_vel_xy_exp", "track_ang_vel_z_exp"),
            "stage1_reward_terms": ("track_base_roll_pitch_exp", "track_base_height_command"),
            "stage2_reward_terms": ("track_base_roll_pitch_exp", "track_base_height_command"),
            "arm_difficulty_reward_terms": ("track_base_roll_pitch_exp", "track_base_height_command"),
            "stage0_threshold": 0.70,
            "stage1_threshold": 0.70,
            "stage2_threshold": 0.70,
            "stage_lower_threshold": 0.50,
            "stage2_lower_threshold": 0.50,
            "arm_difficulty_threshold": 0.70,
            "arm_difficulty_lower_threshold": 0.50,
            "arm_difficulty_step": 0.05,
            "alternate_arm_difficulty_stages": True,
            "arm_difficulty_replay_fraction": 0.25,
            "held_arm_replay_fraction": 0.10,
        },
    )
    env_cfg.curriculum.go2_d1_wbc_arm_difficulty = CurriculumTermCfg(
        func=mdp.env_attr_curriculum_metric,
        params={"attr_name": "_loco_manip_arm_motion_difficulty"},
    )
    env_cfg.curriculum.go2_d1_wbc_arm_difficulty_frontier = CurriculumTermCfg(
        func=mdp.env_attr_curriculum_metric,
        params={"attr_name": "_loco_manip_arm_motion_frontier"},
    )
    env_cfg.curriculum.go2_d1_leg_wbc_locomotion_score = CurriculumTermCfg(
        func=mdp.env_attr_curriculum_metric,
        params={"attr_name": "_loco_manip_stage0_score"},
    )
    env_cfg.curriculum.go2_d1_leg_wbc_posture_score = CurriculumTermCfg(
        func=mdp.env_attr_curriculum_metric,
        params={"attr_name": "_loco_manip_stage1_score"},
    )
    env_cfg.curriculum.go2_d1_leg_wbc_combined_posture_score = CurriculumTermCfg(
        func=mdp.env_attr_curriculum_metric,
        params={"attr_name": "_loco_manip_stage2_score"},
    )
    env_cfg.curriculum.go2_d1_leg_wbc_difficulty_score = CurriculumTermCfg(
        func=mdp.env_attr_curriculum_metric,
        params={"attr_name": "_loco_manip_arm_difficulty_score"},
    )
    env_cfg.curriculum.go2_d1_leg_wbc_velocity_diagnostics = CurriculumTermCfg(
        func=mdp.velocity_command_diagnostics,
        params={"command_name": "base_velocity", "planar_threshold": 0.2, "yaw_threshold": 0.05},
    )
    env_cfg.curriculum.go2_d1_leg_wbc_arm_motion_diagnostics = CurriculumTermCfg(
        func=mdp.async_arm_motion_diagnostics,
        params={"near_limit_threshold": 0.90},
    )
    env_cfg.curriculum.go2_d1_leg_wbc_action_diagnostics = CurriculumTermCfg(
        func=mdp.normalized_action_diagnostics,
        params={"action_name": "joint_pos", "saturation_threshold": 1.0},
    )
    env_cfg.curriculum.go2_d1_leg_wbc_joint_limit_diagnostics = CurriculumTermCfg(
        func=mdp.joint_target_limit_diagnostics,
        params={
            "action_name": "joint_pos",
            "soft_factor": 0.9,
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=env_cfg.leg_joint_names,
                preserve_order=True,
            ),
        },
    )
    env_cfg.curriculum.go2_d1_leg_wbc_payload_mass = CurriculumTermCfg(
        func=mdp.env_attr_curriculum_metric,
        params={"attr_name": "_arm_payload_mass"},
    )


def configure_go2_d1_leg_wbc_arm_hierarchical_runtime(env_cfg) -> None:
    """Use a frozen leg-only WBC while the high-level policy commands D1 arm joints."""
    env_cfg.observations.wbc_policy = copy.deepcopy(env_cfg.observations.policy)
    env_cfg.actions.joint_pos = None
    env_cfg.actions.wbc_command = mdp.FrozenGo2D1LegWbcArmCommandActionCfg(
        asset_name="robot",
        leg_joint_names=list(env_cfg.leg_joint_names),
        arm_joint_names=list(env_cfg.arm_joint_names),
        gripper_joint_names=list(env_cfg.gripper_joint_names),
        wbc_obs_group="wbc_policy",
        policy_path=getattr(env_cfg, "wbc_policy_path", ""),
        velocity_scale=(1.0, 1.0, 1.0),
        body_pitch_range=GO2_D1_WBC_BODY_PITCH_RANGE,
        body_height_range=GO2_D1_WBC_BODY_HEIGHT_RANGE,
        arm_joint_ranges=GO2_D1_MANIP_ARM_JOINT_RANGES,
        gripper_scale=0.033,
        leg_joint_action_scales={
            r"^(FL|FR|RL|RR)_hip_joint$": 0.125,
            r"^(FL|FR|RL|RR)_(thigh|calf)_joint$": 0.25,
        },
        preserve_order=True,
    )
    env_cfg.observations.wbc_policy.actions = ObsTerm(
        func=mdp.frozen_wbc_low_level_actions,
        params={"action_name": "wbc_command"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    for event_name in (
        "reset_arm_async_motion",
        "randomize_arm_async_motion",
        "advance_arm_async_motion",
        "reset_arm_command",
        "randomize_arm_command",
        "advance_arm_command",
    ):
        if hasattr(env_cfg.events, event_name):
            setattr(env_cfg.events, event_name, None)
    _disable_go2_d1_wbc_training_curricula(env_cfg)


def configure_go2_d1_leg_wbc_ee_hierarchical_runtime(env_cfg) -> None:
    """Use a frozen leg WBC while the high level commands velocity and Cartesian D1 goals."""
    env_cfg.observations.wbc_policy = copy.deepcopy(env_cfg.observations.policy)
    env_cfg.actions.joint_pos = None
    env_cfg.actions.wbc_command = mdp.FrozenGo2D1LegWbcEeCommandActionCfg(
        asset_name="robot",
        leg_joint_names=list(env_cfg.leg_joint_names),
        arm_joint_names=list(env_cfg.arm_joint_names),
        gripper_joint_names=list(env_cfg.gripper_joint_names),
        wbc_obs_group="wbc_policy",
        policy_path=getattr(env_cfg, "wbc_policy_path", ""),
        velocity_scale=(1.50, 0.50, 1.50),
        ee_pos_scale=(1.0, 1.0, 1.0),
        ee_pos_range=GO2_D1_WBC_EE_POS_RANGE,
        workspace_origin=GO2_D1_WBC_WORKSPACE_ORIGIN,
        reach_range=GO2_D1_WBC_REACH_RANGE,
        body_exclusion_box=GO2_D1_WBC_BODY_EXCLUSION_BOX,
        body_clearance=GO2_D1_WBC_BODY_CLEARANCE,
        body_pitch_range=GO2_D1_WBC_BODY_PITCH_RANGE,
        body_height_range=GO2_D1_WBC_BODY_HEIGHT_RANGE,
        body_nominal_height=GO2_D1_WBC_BODY_NOMINAL_HEIGHT,
        ee_body_name="Link6",
        grasp_body_names=("Link7_1", "Link7_2"),
        ee_nominal_quat=(0.7071067811865476, 0.0, 0.7071067811865475, 0.0),
        wrist_roll_range=(-1.5707963267948966, 1.5707963267948966),
        arm_joint_ranges=GO2_D1_IK_ARM_JOINT_RANGES,
        deployment_ee_waypoints=GO2_D1_WBC_DEPLOYMENT_EE_WAYPOINTS,
        deployment_motion_speed=0.18,
        deployment_min_segment_duration_s=1.0,
        dls_damping=0.06,
        gripper_scale=0.033,
        leg_joint_action_scales={
            r"^(FL|FR|RL|RR)_hip_joint$": 0.125,
            r"^(FL|FR|RL|RR)_(thigh|calf)_joint$": 0.25,
        },
        preserve_order=True,
    )
    env_cfg.observations.wbc_policy.actions = ObsTerm(
        func=mdp.frozen_wbc_low_level_actions,
        params={"action_name": "wbc_command"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    for event_name in (
        "reset_arm_async_motion",
        "randomize_arm_async_motion",
        "advance_arm_async_motion",
        "reset_arm_command",
        "randomize_arm_command",
        "advance_arm_command",
    ):
        if hasattr(env_cfg.events, event_name):
            setattr(env_cfg.events, event_name, None)
    _disable_go2_d1_wbc_training_curricula(env_cfg)


def configure_go2_d1_wbc_online_apex_arm(env_cfg) -> None:
    """Use online IK-generated references as a decaying arm action prior while tracking EE goals."""
    arm_cfg = SceneEntityCfg("robot", joint_names=env_cfg.arm_joint_names, body_names=["Link6"], preserve_order=True)

    base_action_cfg = env_cfg.actions.joint_pos
    env_cfg.actions.joint_pos = mdp.OnlineArmDecapJointPositionActionCfg(
        asset_name=base_action_cfg.asset_name,
        joint_names=list(env_cfg.joint_names),
        scale=copy.deepcopy(base_action_cfg.scale),
        offset=copy.deepcopy(base_action_cfg.offset),
        clip=copy.deepcopy(base_action_cfg.clip),
        preserve_order=True,
        use_default_offset=base_action_cfg.use_default_offset,
        decap_joint_names=list(env_cfg.arm_joint_names),
        decap_target_attr="_arm_ik_joint_target_pos",
        decap_lambda_start=1.0,
        decap_lambda_end=0.0,
        decap_decay_type="cosine",
        decap_steps_per_iteration=24,
        decap_warmup_iterations=100,
        decap_decay_start_iteration=0,
        decap_decay_end_iteration=2000,
        decap_prior_only=False,
    )

    apex_reference = ObsTerm(
        func=mdp.arm_apex_reference_state,
        params={
            "asset_cfg": arm_cfg,
            "target_attr": "_arm_ik_joint_target_pos",
            "ee_target_attr": "_arm_ee_target_pos",
            "ee_goal_attr": _ARM_EE_GOAL_ATTR,
            "include_joint_pos": True,
            "include_joint_error": True,
            "include_ee_target": False,
            "include_ee_error": True,
            "include_ee_goal": True,
            "include_trajectory": True,
            "add_noise": False,
        },
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    env_cfg.observations.critic.arm_apex_reference = apex_reference

    env_cfg.rewards.arm_ee_target_tracking.params["target_attr"] = _ARM_EE_COMMAND_ATTR
    env_cfg.rewards.arm_ik_joint_target_tracking = None


@configclass
class UnitreeGo2D1WholeBodyControllerFlatEnvCfg(UnitreeGo2D1ArmFlatEnvCfg):
    """Flat stage-1 Go2+D1 WBC for hierarchical loco-manipulation."""

    def __post_init__(self):
        super().__post_init__()
        configure_go2_d1_whole_body_controller(self)

        if self.__class__.__name__ == "UnitreeGo2D1WholeBodyControllerFlatEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class UnitreeGo2D1WholeBodyControllerFlatMjlabActionScaleEnvCfg(UnitreeGo2D1WholeBodyControllerFlatEnvCfg):
    """Flat Go2+D1 WBC with mjlab-derived action scales for legs and D1 arm joints."""

    def __post_init__(self):
        super().__post_init__()

        self.actions.joint_pos.scale = GO2_D1_MJLAB_ACTION_SCALE

        if self.__class__.__name__ == "UnitreeGo2D1WholeBodyControllerFlatMjlabActionScaleEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class UnitreeGo2D1WholeBodyControllerRoughEnvCfg(UnitreeGo2D1ArmRoughEnvCfg):
    """Rough stage-1 Go2+D1 WBC for hierarchical loco-manipulation."""

    def __post_init__(self):
        super().__post_init__()
        configure_go2_d1_whole_body_controller(self)

        if self.__class__.__name__ == "UnitreeGo2D1WholeBodyControllerRoughEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class UnitreeGo2D1LegWbcAsyncArmFlatEnvCfg(UnitreeGo2D1ArmFlatEnvCfg):
    """Flat stage-1 leg-only Go2 WBC with asynchronous scripted D1 arm disturbances."""

    def __post_init__(self):
        super().__post_init__()
        configure_go2_d1_leg_wbc_async_arm_controller(self)

        if self.__class__.__name__ == "UnitreeGo2D1LegWbcAsyncArmFlatEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class UnitreeGo2D1LegWbcAsyncArmRoughEnvCfg(UnitreeGo2D1ArmRoughEnvCfg):
    """Rough stage-1 leg-only Go2 WBC with asynchronous scripted D1 arm disturbances."""

    def __post_init__(self):
        super().__post_init__()
        configure_go2_d1_leg_wbc_async_arm_controller(self)

        if self.__class__.__name__ == "UnitreeGo2D1LegWbcAsyncArmRoughEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class UnitreeGo2D1WholeBodyControllerApexArmFlatEnvCfg(UnitreeGo2D1WholeBodyControllerFlatEnvCfg):
    """Flat Go2+D1 WBC with online IK APEX-style arm reference tracking."""

    def __post_init__(self):
        super().__post_init__()
        configure_go2_d1_wbc_online_apex_arm(self)

        if self.__class__.__name__ == "UnitreeGo2D1WholeBodyControllerApexArmFlatEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class UnitreeGo2D1WholeBodyControllerApexArmRoughEnvCfg(UnitreeGo2D1WholeBodyControllerRoughEnvCfg):
    """Rough Go2+D1 WBC with online IK APEX-style arm reference tracking."""

    def __post_init__(self):
        super().__post_init__()
        configure_go2_d1_wbc_online_apex_arm(self)

        if self.__class__.__name__ == "UnitreeGo2D1WholeBodyControllerApexArmRoughEnvCfg":
            self.disable_zero_weight_rewards()
