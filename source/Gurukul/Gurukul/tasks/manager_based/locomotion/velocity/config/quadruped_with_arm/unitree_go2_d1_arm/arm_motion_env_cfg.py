# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import CurriculumTermCfg, EventTermCfg, SceneEntityCfg

import Gurukul.tasks.manager_based.locomotion.velocity.mdp as mdp
from Gurukul.assets.unitree import GO2_D1_ARM_READY_POSE


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


def configure_d1_arm_joint_motion(env_cfg, *, visualize: bool = False) -> None:
    """Drive the D1 arm through smooth joint-space primitives while the policy controls Go2 legs."""
    controlled_arm_joint_names = getattr(env_cfg, "arm_joint_names", env_cfg.joint_names)
    arm_entity_cfg = SceneEntityCfg(
        "robot",
        joint_names=controlled_arm_joint_names,
        body_names=["Link6"],
    )

    env_cfg.actions.joint_pos.scale = {
        r"^(FL|FR|RL|RR)_hip_joint$": 0.125,
        r"^(FL|FR|RL|RR)_(thigh|calf)_joint$": 0.25,
    }
    env_cfg.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
    env_cfg.actions.joint_pos.joint_names = env_cfg.leg_joint_names

    env_cfg.observations.policy.joint_pos.params["asset_cfg"].joint_names = env_cfg.joint_names
    env_cfg.observations.policy.joint_vel.params["asset_cfg"].joint_names = env_cfg.leg_joint_names
    _set_reward_joint_names(env_cfg, env_cfg.leg_joint_names)

    # Low parallel pose matches the Go2+D1 asset default spawn configuration.
    ready = GO2_D1_ARM_READY_POSE
    carry = (0.0, -1.15, 1.35, 0.0, -0.30, 0.0)
    forward = (0.0, -0.85, 1.20, 0.0, -0.45, 0.0)
    pick_low = (0.0, -1.20, 1.45, 0.0, -0.30, 0.0)
    pick_mid = (0.0, -0.75, 1.20, 0.0, -0.45, 0.0)
    pick_high = (0.0, -0.35, 0.95, 0.0, -0.60, 0.0)
    place_left = (0.45, -0.75, 1.15, 0.0, -0.45, 0.0)
    place_right = (-0.45, -0.75, 1.15, 0.0, -0.45, 0.0)
    env_cfg.scene.robot.init_state.joint_pos.update(
        {
            "arm_1_joint": ready[0],
            "arm_2_joint": ready[1],
            "arm_3_joint": ready[2],
            "arm_4_joint": ready[3],
            "arm_5_joint": ready[4],
            "arm_6_joint": ready[5],
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
                (0.45, -1.05, 1.25, 0.0, -0.35, 0.0),
                carry,
                (-0.45, -1.05, 1.25, 0.0, -0.35, 0.0),
                ready,
            ),
        },
    }

    env_cfg.events.continuous_arm_tracking = EventTermCfg(
        func=mdp.continuous_arm_joint_tracking,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={"asset_cfg": arm_entity_cfg},
    )

    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.command_levels_ang_vel = None
    env_cfg.curriculum.loco_manipulation_training_stages = CurriculumTermCfg(
        func=mdp.loco_manipulation_training_stages,
        params={
            "stage_steps": (40_000, 40_000),
            "arm_difficulty_steps": (40_000, 120_000),
            "stage_iteration_bins": (0.10, 0.10),
            "arm_difficulty_iteration_bins": (0.10, 1.0),
            "walking_lin_vel_x": (-1.0, 1.0),
            "walking_lin_vel_y": (-0.5, 0.5),
            "walking_ang_vel_z": (-1.0, 1.0),
            "command_name": "base_velocity",
        },
    )
