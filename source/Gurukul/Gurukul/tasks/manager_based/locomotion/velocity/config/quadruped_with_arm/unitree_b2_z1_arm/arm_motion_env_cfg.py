# SPDX-License-Identifier: Apache-2.0

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

import Gurukul.tasks.manager_based.locomotion.velocity.mdp as mdp
from Gurukul.assets.unitree import B2_Z1_LIDAR_POS


def _set_reward_joint_names(env_cfg, joint_names: list[str]) -> None:
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


def configure_z1_arm_joint_motion(env_cfg, *, visualize: bool = False) -> None:
    """Drive the Z1 arm through smooth joint-space primitives while the policy controls B2 legs."""
    controlled_arm_joint_names = getattr(env_cfg, "arm_joint_names", env_cfg.joint_names)
    arm_entity_cfg = SceneEntityCfg(
        "robot",
        joint_names=controlled_arm_joint_names,
        body_names=["link06"],
    )

    env_cfg.actions.joint_pos.scale = {
        r"^(FL|FR|RL|RR)_hip_joint$": 0.125,
        r"^(FL|FR|RL|RR)_(thigh|calf)_joint$": 0.25,
    }
    env_cfg.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
    env_cfg.actions.joint_pos.joint_names = env_cfg.leg_joint_names

    env_cfg.observations.policy.joint_pos.params["asset_cfg"].joint_names = env_cfg.joint_names
    env_cfg.observations.policy.joint_vel.params["asset_cfg"].joint_names = env_cfg.joint_names
    _set_reward_joint_names(env_cfg, env_cfg.leg_joint_names)
    env_cfg.events.randomize_reset_joints.params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=env_cfg.leg_joint_names
    )

    env_cfg.commands.base_velocity = mdp.UniformThresholdVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        ranges=mdp.UniformThresholdVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
            heading=(-3.14159, 3.14159),
        ),
    )

    ready = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    carry = (0.55, 1.20, -1.05, -0.25, 0.0, 0.0)
    forward = (0.55, 1.00, -1.15, 0.05, 0.0, 0.0)
    pick_low = (0.55, 1.55, -1.35, -0.25, 0.0, 0.0)
    pick_mid = (0.55, 1.25, -1.20, 0.05, 0.0, 0.0)
    pick_high = (0.55, 0.90, -0.90, 0.35, 0.0, 0.0)
    place_left = (0.85, 1.25, -1.15, 0.05, 0.0, 0.0)
    place_right = (0.35, 1.25, -1.15, 0.05, 0.0, 0.0)
    env_cfg.scene.robot.init_state.joint_pos.update(
        {
            "joint1": ready[0],
            "joint2": ready[1],
            "joint3": ready[2],
            "joint4": ready[3],
            "joint5": ready[4],
            "joint6": ready[5],
        }
    )

    env_cfg.events.randomize_push_robot.func = mdp.random_arm_joint_motion
    env_cfg.events.randomize_push_robot.interval_range_s = (2.0, 3.5)
    env_cfg.events.randomize_push_robot.params = {
        "asset_cfg": arm_entity_cfg,
        "visualize": visualize,
        "start_motion_enabled": False,
        "max_joint_change": 0.30,
        "max_joint_change_range": (0.12, 0.30),
        "motion_speed": 0.40,
        "motion_speed_range": (0.18, 0.40),
        "motion_primitives": ("move", "pick_forward", "pick_low", "pick_high", "place", "sweep", "stow"),
        "difficulty_motion_primitives": (
            ("move", "stow"),
            ("move", "pick_forward", "pick_low", "stow"),
            ("move", "pick_forward", "pick_low", "pick_high", "place", "sweep", "stow"),
        ),
        "joint_motion_library": {
            "stow": (ready,),
            "move": (
                ready,
                carry,
                forward,
                carry,
                ready,
            ),
            "pick_forward": (
                ready,
                carry,
                forward,
                pick_mid,
                carry,
                ready,
            ),
            "pick_low": (
                ready,
                carry,
                pick_low,
                carry,
                ready,
            ),
            "pick_high": (
                ready,
                carry,
                pick_high,
                carry,
                ready,
            ),
            "place": (
                ready,
                carry,
                place_left,
                carry,
                place_right,
                carry,
                ready,
            ),
            "pick_place": (
                ready,
                carry,
                pick_low,
                carry,
                place_left,
                carry,
                pick_mid,
                carry,
                place_right,
                ready,
            ),
            "sweep": (
                ready,
                (0.85, 1.35, -1.20, -0.05, 0.0, 0.0),
                carry,
                (0.35, 1.35, -1.20, -0.05, 0.0, 0.0),
                ready,
            ),
        },
    }
    env_cfg.events.reset_arm_joint_state = EventTermCfg(
        func=mdp.reset_joints_by_absolute_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=controlled_arm_joint_names + env_cfg.gripper_joint_names),
            "position_ranges": {
                **{joint_name: (0.0, 0.0) for joint_name in controlled_arm_joint_names},
                **{joint_name: (-1.2, -1.2) for joint_name in env_cfg.gripper_joint_names},
            },
            "velocity_range": (0.0, 0.0),
        },
    )
    env_cfg.events.reset_arm_joint_motion = EventTermCfg(
        func=mdp.random_arm_joint_motion,
        mode="reset",
        params=dict(env_cfg.events.randomize_push_robot.params),
    )

    env_cfg.events.continuous_arm_tracking = EventTermCfg(
        func=mdp.continuous_arm_joint_tracking,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={"asset_cfg": arm_entity_cfg},
    )

    env_cfg.rewards.track_base_roll_pitch_exp = RewTerm(
        func=mdp.track_base_roll_pitch_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": 0.20},
    )
    if getattr(env_cfg.rewards, "upward", None) is not None:
        env_cfg.rewards.upward.weight = min(float(env_cfg.rewards.upward.weight), 1.0)

    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.command_levels_ang_vel = None
    env_cfg.curriculum.loco_manipulation_training_stages = CurriculumTermCfg(
        func=mdp.loco_manipulation_training_stages,
        params={
            "stage_steps": (40_000, 80_000),
            "arm_difficulty_steps": (40_000, 120_000),
            "stage_iteration_bins": (1.0 / 3.0, 2.0 / 3.0),
            "arm_difficulty_iteration_bins": (1.0 / 3.0, 1.0),
            "walking_lin_vel_x": (-1.0, 1.0),
            "walking_lin_vel_y": (-0.5, 0.5),
            "walking_ang_vel_z": (-1.0, 1.0),
            "walking_roll": (0.0, 0.0),
            "walking_pitch": (-0.18, 0.12),
            "command_name": "base_velocity",
        },
    )


def configure_z1_arm_task_space_motion(env_cfg, *, visualize: bool = False) -> None:
    """Train B2 locomotion against broad Z1 task-space motion and commanded base bending."""
    controlled_arm_joint_names = getattr(env_cfg, "arm_joint_names", env_cfg.joint_names)
    arm_entity_cfg = SceneEntityCfg(
        "robot",
        joint_names=controlled_arm_joint_names,
        body_names=["link06"],
    )

    env_cfg.actions.joint_pos.scale = {
        r"^(FL|FR|RL|RR)_hip_joint$": 0.125,
        r"^(FL|FR|RL|RR)_(thigh|calf)_joint$": 0.25,
    }
    env_cfg.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
    env_cfg.actions.joint_pos.joint_names = env_cfg.leg_joint_names

    env_cfg.observations.policy.joint_pos.params["asset_cfg"].joint_names = env_cfg.joint_names
    env_cfg.observations.policy.joint_vel.params["asset_cfg"].joint_names = env_cfg.joint_names
    env_cfg.observations.policy.velocity_commands = ObsTerm(
        func=mdp.velocity_posture_commands,
        params={"command_name": "base_velocity"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    _set_reward_joint_names(env_cfg, env_cfg.leg_joint_names)
    env_cfg.events.randomize_reset_joints.params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=env_cfg.leg_joint_names
    )

    env_cfg.commands.base_velocity = mdp.UniformVelocityPostureCommandCfg(
        asset_name="robot",
        resampling_time_range=(10.0, 10.0),
        rel_standing_envs=0.02,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        roll_range=(0.0, 0.0),
        pitch_range=(0.0, 0.0),
        ranges=mdp.UniformVelocityPostureCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.0),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
            heading=(-3.14159, 3.14159),
        ),
    )

    ready = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    lidar_keepout = (
        (B2_Z1_LIDAR_POS[0] - 0.12, B2_Z1_LIDAR_POS[0] + 0.12),
        (B2_Z1_LIDAR_POS[1] - 0.14, B2_Z1_LIDAR_POS[1] + 0.14),
        (B2_Z1_LIDAR_POS[2] - 0.09, B2_Z1_LIDAR_POS[2] + 0.13),
    )
    env_cfg.scene.robot.init_state.joint_pos.update(
        {
            "joint1": ready[0],
            "joint2": ready[1],
            "joint3": ready[2],
            "joint4": ready[3],
            "joint5": ready[4],
            "joint6": ready[5],
        }
    )

    env_cfg.events.randomize_push_robot.func = mdp.random_arm_ik_motion
    env_cfg.events.randomize_push_robot.interval_range_s = (1.25, 2.25)
    env_cfg.events.randomize_push_robot.params = {
        "asset_cfg": arm_entity_cfg,
        "visualize": visualize,
        "start_motion_enabled": False,
        "smooth_motion": True,
        # Keep a side-inclusive trainable envelope while clamping centerline targets in front of the B2 trunk.
        "ee_pos_range": ((0.22, 0.95), (-0.65, 0.65), (0.24, 0.92)),
        "body_exclusion_box": ((-0.46, 0.50), (-0.30, 0.30), (-0.08, 0.44)),
        "extra_exclusion_boxes": (lidar_keepout,),
        "body_clearance": 0.10,
        "max_pos_change": 0.14,
        "max_pos_change_range": (0.06, 0.16),
        "motion_speed": 0.14,
        "motion_speed_range": (0.06, 0.18),
        "motion_primitives": ("workspace", "reach", "pick_place", "sweep", "stow"),
        "preserve_current_orientation": True,
        "neutral_pos": (0.58, 0.0, 0.56),
        "min_workspace_fraction": 0.35,
        "ik_controller_cfg": {"ik_method": "dls", "dls_damping": 0.08},
    }
    env_cfg.events.reset_arm_joint_state = EventTermCfg(
        func=mdp.reset_joints_by_absolute_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=controlled_arm_joint_names + env_cfg.gripper_joint_names),
            "position_ranges": {
                **{joint_name: (0.0, 0.0) for joint_name in controlled_arm_joint_names},
                **{joint_name: (-1.2, -1.2) for joint_name in env_cfg.gripper_joint_names},
            },
            "velocity_range": (0.0, 0.0),
        },
    )
    env_cfg.events.reset_arm_ik_motion = EventTermCfg(
        func=mdp.random_arm_ik_motion,
        mode="reset",
        params=dict(env_cfg.events.randomize_push_robot.params),
    )

    env_cfg.events.continuous_arm_tracking = EventTermCfg(
        func=mdp.continuous_arm_ik_tracking,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={"asset_cfg": arm_entity_cfg},
    )

    env_cfg.rewards.track_base_roll_pitch_exp = RewTerm(
        func=mdp.track_base_roll_pitch_exp,
        weight=0.75,
        params={"command_name": "base_velocity", "std": 0.20},
    )
    if getattr(env_cfg.rewards, "upward", None) is not None:
        env_cfg.rewards.upward.weight = min(float(env_cfg.rewards.upward.weight), 1.0)

    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.command_levels_ang_vel = None
    env_cfg.curriculum.loco_manipulation_training_stages = CurriculumTermCfg(
        func=mdp.loco_manipulation_training_stages,
        params={
            "stage_steps": (40_000, 80_000),
            "arm_difficulty_steps": (40_000, 160_000),
            "stage_iteration_bins": (1.0 / 3.0, 2.0 / 3.0),
            "arm_difficulty_iteration_bins": (1.0 / 3.0, 1.0),
            "walking_lin_vel_x": (-1.0, 1.0),
            "walking_lin_vel_y": (-0.5, 0.5),
            "walking_ang_vel_z": (-1.0, 1.0),
            "walking_roll": (-0.16, 0.16),
            "walking_pitch": (-0.30, 0.18),
            "command_name": "base_velocity",
        },
    )
