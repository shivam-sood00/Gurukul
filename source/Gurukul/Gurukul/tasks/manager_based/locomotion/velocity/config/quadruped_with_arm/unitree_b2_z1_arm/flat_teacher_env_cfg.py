"""Flat B2+Z1 ArmMoving teacher task with privileged arm-motion observations."""
# SPDX-License-Identifier: Apache-2.0

import copy

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import Gurukul.tasks.manager_based.locomotion.velocity.mdp as mdp

from .flat_env_cfg_arm_moving import UnitreeB2Z1ArmFlatArmMovingEnvCfg
from .flat_env_cfg_arm_moving_wide import UnitreeB2Z1ArmFlatWideArmMovingEnvCfg


def configure_b2_z1_arm_moving_teacher_observations(env_cfg) -> None:
    """Add privileged teacher observations for B2+Z1 ArmMoving tasks."""
    env_cfg.observations.teacher = copy.deepcopy(env_cfg.observations.critic)
    arm_asset_cfg = SceneEntityCfg("robot", joint_names=env_cfg.arm_joint_names, preserve_order=True)
    foot_contact_cfg = SceneEntityCfg("contact_forces", body_names=[env_cfg.foot_link_name])
    non_foot_contact_cfg = SceneEntityCfg("contact_forces", body_names=[f"^(?!.*{env_cfg.foot_link_name}).*"])

    env_cfg.observations.teacher.root_pos_w = ObsTerm(func=mdp.root_pos_w, clip=(-100.0, 100.0), scale=1.0)
    env_cfg.observations.teacher.root_quat_w = ObsTerm(
        func=mdp.root_quat_w,
        params={"make_quat_unique": True},
        clip=(-1.0, 1.0),
        scale=1.0,
    )
    env_cfg.observations.teacher.root_lin_vel_w = ObsTerm(func=mdp.root_lin_vel_w, clip=(-100.0, 100.0), scale=1.0)
    env_cfg.observations.teacher.root_ang_vel_w = ObsTerm(func=mdp.root_ang_vel_w, clip=(-100.0, 100.0), scale=1.0)
    env_cfg.observations.teacher.arm_target_joint_pos = ObsTerm(
        func=mdp.arm_joint_target_rel,
        params={"asset_cfg": arm_asset_cfg, "target_attr": "_arm_joint_target_pos"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    env_cfg.observations.teacher.arm_goal_joint_pos = ObsTerm(
        func=mdp.arm_joint_target_rel,
        params={"asset_cfg": arm_asset_cfg, "target_attr": "_arm_joint_goal_pos"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    env_cfg.observations.teacher.arm_target_error = ObsTerm(
        func=mdp.arm_joint_target_error,
        params={"asset_cfg": arm_asset_cfg, "target_attr": "_arm_joint_target_pos"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    env_cfg.observations.teacher.arm_motion_state = ObsTerm(
        func=mdp.arm_motion_state,
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    env_cfg.observations.teacher.arm_trajectory_state = ObsTerm(
        func=mdp.arm_joint_trajectory_state,
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    env_cfg.observations.teacher.feet_contact_state = ObsTerm(
        func=mdp.body_contact_state,
        params={"sensor_cfg": foot_contact_cfg, "force_threshold": 1.0},
        clip=(0.0, 1.0),
        scale=1.0,
    )
    env_cfg.observations.teacher.feet_contact_force_norms = ObsTerm(
        func=mdp.body_contact_force_norms,
        params={"sensor_cfg": foot_contact_cfg, "scale": 100.0},
        clip=(0.0, 100.0),
        scale=1.0,
    )
    env_cfg.observations.teacher.non_foot_contact_state = ObsTerm(
        func=mdp.body_contact_state,
        params={"sensor_cfg": non_foot_contact_cfg, "force_threshold": 1.0},
        clip=(0.0, 1.0),
        scale=1.0,
    )
    env_cfg.observations.teacher.non_foot_contact_force_norms = ObsTerm(
        func=mdp.body_contact_force_norms,
        params={"sensor_cfg": non_foot_contact_cfg, "scale": 100.0},
        clip=(0.0, 100.0),
        scale=1.0,
    )


def configure_b2_z1_wide_arm_moving_teacher_observations(env_cfg) -> None:
    """Add privileged teacher observations for task-space B2+Z1 ArmMoving tasks."""
    env_cfg.observations.teacher = copy.deepcopy(env_cfg.observations.critic)
    ee_asset_cfg = SceneEntityCfg("robot", body_names=["link06"])
    foot_contact_cfg = SceneEntityCfg("contact_forces", body_names=[env_cfg.foot_link_name])
    non_foot_contact_cfg = SceneEntityCfg("contact_forces", body_names=[f"^(?!.*{env_cfg.foot_link_name}).*"])

    env_cfg.observations.teacher.root_pos_w = ObsTerm(func=mdp.root_pos_w, clip=(-100.0, 100.0), scale=1.0)
    env_cfg.observations.teacher.root_quat_w = ObsTerm(
        func=mdp.root_quat_w,
        params={"make_quat_unique": True},
        clip=(-1.0, 1.0),
        scale=1.0,
    )
    env_cfg.observations.teacher.root_lin_vel_w = ObsTerm(func=mdp.root_lin_vel_w, clip=(-100.0, 100.0), scale=1.0)
    env_cfg.observations.teacher.root_ang_vel_w = ObsTerm(func=mdp.root_ang_vel_w, clip=(-100.0, 100.0), scale=1.0)
    env_cfg.observations.teacher.arm_ee_target_pos_b = ObsTerm(
        func=mdp.arm_ee_target_pos_b,
        params={"target_attr": "_arm_ee_target_pos"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    env_cfg.observations.teacher.arm_ee_goal_pos_b = ObsTerm(
        func=mdp.arm_ee_target_pos_b,
        params={"target_attr": "_arm_ee_goal_pos"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    env_cfg.observations.teacher.arm_ee_target_error_b = ObsTerm(
        func=mdp.arm_ee_target_error_b,
        params={"asset_cfg": ee_asset_cfg, "target_attr": "_arm_ee_target_pos"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    env_cfg.observations.teacher.arm_motion_state = ObsTerm(
        func=mdp.arm_motion_state,
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    env_cfg.observations.teacher.arm_trajectory_state = ObsTerm(
        func=mdp.arm_ee_trajectory_state,
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    env_cfg.observations.teacher.feet_contact_state = ObsTerm(
        func=mdp.body_contact_state,
        params={"sensor_cfg": foot_contact_cfg, "force_threshold": 1.0},
        clip=(0.0, 1.0),
        scale=1.0,
    )
    env_cfg.observations.teacher.feet_contact_force_norms = ObsTerm(
        func=mdp.body_contact_force_norms,
        params={"sensor_cfg": foot_contact_cfg, "scale": 100.0},
        clip=(0.0, 100.0),
        scale=1.0,
    )
    env_cfg.observations.teacher.non_foot_contact_state = ObsTerm(
        func=mdp.body_contact_state,
        params={"sensor_cfg": non_foot_contact_cfg, "force_threshold": 1.0},
        clip=(0.0, 1.0),
        scale=1.0,
    )
    env_cfg.observations.teacher.non_foot_contact_force_norms = ObsTerm(
        func=mdp.body_contact_force_norms,
        params={"sensor_cfg": non_foot_contact_cfg, "scale": 100.0},
        clip=(0.0, 100.0),
        scale=1.0,
    )


@configclass
class UnitreeB2Z1ArmFlatArmMovingTeacherEnvCfg(UnitreeB2Z1ArmFlatArmMovingEnvCfg):
    """Teacher variant whose actor sees privileged base and scripted-arm state."""

    def __post_init__(self):
        super().__post_init__()
        configure_b2_z1_arm_moving_teacher_observations(self)

        if self.__class__.__name__ == "UnitreeB2Z1ArmFlatArmMovingTeacherEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class UnitreeB2Z1ArmFlatWideArmMovingTeacherEnvCfg(UnitreeB2Z1ArmFlatWideArmMovingEnvCfg):
    """Teacher variant whose actor sees privileged task-space arm target state."""

    def __post_init__(self):
        super().__post_init__()
        configure_b2_z1_wide_arm_moving_teacher_observations(self)

        if self.__class__.__name__ == "UnitreeB2Z1ArmFlatWideArmMovingTeacherEnvCfg":
            self.disable_zero_weight_rewards()
