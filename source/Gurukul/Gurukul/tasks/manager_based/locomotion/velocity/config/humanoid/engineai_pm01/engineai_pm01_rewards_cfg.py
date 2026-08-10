# SPDX-License-Identifier: Apache-2.0
"""Reward bundle matching engineai_amp ``PM01Rewards`` (Flat-PM01-v0), excluding AMP.

Also includes mjlab-style ``variable_posture``, torso angular-velocity, soft landing,
and self-collision terms (see ``unitree_g1/rough_env_cfg_v1.py``).
"""

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import Gurukul.tasks.manager_based.locomotion.velocity.mdp as mdp
import Gurukul.tasks.manager_based.locomotion.velocity.mdp.pm01_rewards as pm01_mdp
from Gurukul.tasks.manager_based.locomotion.velocity.velocity_env_cfg import RewardsCfg

from .pm01_constants import (
    PM01_BASE_HEIGHT_TARGET,
    PM01_POLICY_JOINT_ASSET_CFG,
    PM01_VARIABLE_POSTURE_STD_RUNNING,
    PM01_VARIABLE_POSTURE_STD_STANDING,
    PM01_VARIABLE_POSTURE_STD_WALKING,
)

_PM01_CURR_INTERVAL = 200 * 24

_FEET_CONTACT_SENSOR = SceneEntityCfg("contact_forces", body_names=["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"])
_FEET_ASSET = SceneEntityCfg("robot", body_names=["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"])
_ALL_JOINTS = SceneEntityCfg("robot", joint_names=[".*"])


@configclass
class EngineAiPm01RewardsCfg(RewardsCfg):
    """EngineAI-Lab PM01 velocity rewards (non-AMP)."""

    # --- Velocity tracking (sigma kernel, hybrid stand/walk) ---
    track_lin_vel_xy_exp = RewTerm(
        func=pm01_mdp.track_lin_vel_xy_yaw_frame_exp_sigma,
        weight=2.0,
        params={"command_name": "base_velocity", "sigma": 5.0},
    )
    track_ang_vel_z_exp = RewTerm(
        func=pm01_mdp.track_ang_vel_z_world_exp_sigma,
        weight=2.5,
        params={"command_name": "base_velocity", "sigma": 5.0},
    )

    # --- Disable legacy stack items replaced by PM01-specific terms ---
    lin_vel_z_l2 = RewTerm(func=mdp.lin_vel_z_l2, weight=0.0)
    ang_vel_xy_l2 = RewTerm(func=mdp.ang_vel_xy_l2, weight=0.0)
    flat_orientation_l2 = RewTerm(func=mdp.flat_orientation_l2, weight=0.0)
    base_height_l2 = RewTerm(
        func=mdp.base_height_l2,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["LINK_BASE"]),
            "sensor_cfg": SceneEntityCfg("height_scanner_base"),
            "target_height": 0.0,
        },
    )
    body_lin_acc_l2 = RewTerm(
        func=mdp.body_lin_acc_l2,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="LINK_BASE")},
    )

    pm01_base_orientation = RewTerm(func=pm01_mdp.base_orientation, weight=1.0)
    pm01_base_height_tracking = RewTerm(
        func=pm01_mdp.base_height_tracking,
        weight=0.4,
        params={"target_height": PM01_BASE_HEIGHT_TARGET},
    )

    pm01_feet_position = RewTerm(
        func=pm01_mdp.feet_position,
        weight=0.5,
        params={
            "asset_cfg": _FEET_ASSET,
            "command_name": "base_velocity",
            "stand_threshold": 0.1,
            "ankle_distance": 0.22,
            "base_height_target": PM01_BASE_HEIGHT_TARGET,
        },
    )
    pm01_feet_orientation = RewTerm(
        func=pm01_mdp.feet_orientation,
        weight=0.25,
        params={
            "asset_cfg": _FEET_ASSET,
            "command_name": "base_velocity",
            "stand_threshold": 0.1,
        },
    )

    # mjlab ``variable_posture`` replaces per-group ``joint_deviation_exp`` terms.
    pm01_pose = RewTerm(
        func=mdp.variable_posture,
        weight=1.0,
        params={
            "asset_cfg": PM01_POLICY_JOINT_ASSET_CFG,
            "command_name": "base_velocity",
            "std_standing": PM01_VARIABLE_POSTURE_STD_STANDING,
            "std_walking": PM01_VARIABLE_POSTURE_STD_WALKING,
            "std_running": PM01_VARIABLE_POSTURE_STD_RUNNING,
            "walking_threshold": 0.05,
            "running_threshold": 1.5,
        },
    )

    pm01_joint_deviation_waist = RewTerm(
        func=pm01_mdp.joint_deviation_exp,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=["J12_WAIST_YAW"]), "scale": 3.0, "tolerance": 0.0},
    )
    pm01_joint_deviation_legs = RewTerm(
        func=pm01_mdp.joint_deviation_exp,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_HIP_ROLL_.*", ".*_HIP_YAW_.*", ".*_ANKLE_ROLL_.*"]),
            "scale": 3.0,
        },
    )
    pm01_joint_deviation_arm_pitch = RewTerm(
        func=pm01_mdp.joint_deviation_exp,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_SHOULDER_PITCH_.*", ".*_ELBOW_PITCH_.*"]),
            "scale": 3.0,
        },
    )
    pm01_joint_deviation_arm_roll = RewTerm(
        func=pm01_mdp.joint_deviation_exp,
        weight=0.0,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_SHOULDER_ROLL_.*"]), "scale": 3.0},
    )
    pm01_joint_deviation_arm_yaw = RewTerm(
        func=pm01_mdp.joint_deviation_exp,
        weight=0.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*_SHOULDER_YAW_.*", ".*_ELBOW_YAW_.*"]),
            "scale": 10.0,
        },
    )

    pm01_body_ang_vel = RewTerm(
        func=mdp.body_ang_vel_xy_l2,
        weight=-0.05,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=["LINK_BASE"])},
    )
    pm01_soft_landing = RewTerm(
        func=mdp.soft_landing,
        weight=-1.0e-5,
        params={
            "sensor_cfg": _FEET_CONTACT_SENSOR,
            "command_name": "base_velocity",
            "command_threshold": 0.05,
        },
    )
    pm01_feet_contact_fixed = RewTerm(
        func=pm01_mdp.feet_contact_fixed,
        weight=0.25,
        params={
            "sensor_cfg": _FEET_CONTACT_SENSOR,
            "command_name": "base_velocity",
            "stand_threshold": 0.1,
            "force_threshold": 5.0,
        },
    )

    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        # Touchdown signal: penalize steps shorter than the threshold and
        # reward longer swing phases. The dense term below complements it.
        weight=10.0,
        params={
            "command_name": "base_velocity",
            "threshold": 0.5,
            "sensor_cfg": _FEET_CONTACT_SENSOR,
        },
    )
    pm01_feet_air_time_positive = RewTerm(
        func=mdp.feet_air_time_positive_biped,
        weight=1.25,
        params={
            "command_name": "base_velocity",
            "threshold": 0.5,
            "sensor_cfg": _FEET_CONTACT_SENSOR,
        },
    )

    feet_stumble = RewTerm(
        func=pm01_mdp.feet_stumble_pm01,
        weight=-1.0,
        params={
            "sensor_cfg": _FEET_CONTACT_SENSOR,
            "tangential_threshold": 2.0,
            "normal_threshold": 1.0,
        },
    )

    joint_pos_limits = RewTerm(func=mdp.joint_pos_limits, weight=-10.0, params={"asset_cfg": _ALL_JOINTS})

    pm01_energy_curriculum = RewTerm(
        func=pm01_mdp.energy_cost_with_curriculum,
        weight=-0.002,
        params={
            "asset_cfg": _ALL_JOINTS,
            "start_scale": 0.1,
            "power": 0.8,
            "interval_epochs": _PM01_CURR_INTERVAL,
        },
    )

    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.25,
        params={"sensor_cfg": _FEET_CONTACT_SENSOR, "asset_cfg": _FEET_ASSET},
    )

    joint_vel_l2 = RewTerm(func=mdp.joint_vel_l2, weight=-1.0e-5, params={"asset_cfg": _ALL_JOINTS})
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-1.25e-8, params={"asset_cfg": _ALL_JOINTS})

    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=0.0)
    action_smoothness_l2 = RewTerm(func=mdp.action_smoothness_l2, weight=0.0)

    pm01_action_rate_curriculum = RewTerm(
        func=pm01_mdp.action_rate_with_curriculum,
        weight=-0.03,
        params={"start_scale": 0.1, "power": 0.8, "interval_epochs": _PM01_CURR_INTERVAL},
    )
    pm01_action_smoothness_curriculum = RewTerm(
        func=pm01_mdp.action_smoothness_with_curriculum,
        weight=-0.02,
        params={"start_scale": 0.1, "power": 0.8, "interval_epochs": _PM01_CURR_INTERVAL},
    )

    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1.0e-6, params={"asset_cfg": _ALL_JOINTS})

    is_terminated = RewTerm(func=mdp.is_terminated, weight=-200.0)

    # --- Unused PM01 stack terms (weights zero; stripped by disable_zero_weight_rewards) ---
    upward = RewTerm(func=mdp.upward, weight=0.0)
