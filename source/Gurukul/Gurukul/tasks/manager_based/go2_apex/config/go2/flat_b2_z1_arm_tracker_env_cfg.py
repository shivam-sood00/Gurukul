from __future__ import annotations

import glob
import os

import torch
from isaaclab.envs.mdp import JointPositionAction
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import Gurukul.tasks.manager_based.go2_apex.mdp as mdp
from Gurukul.assets.unitree import UNITREE_B2_Z1_ARM_CFG
from Gurukul.tasks.manager_based.go2_apex.constants import (
    B2_Z1_ACTION_JOINT_NAMES,
    B2_Z1_ARM_JOINT_NAMES,
    B2_Z1_GRIPPER_JOINT_NAMES,
    FOOT_BODY_NAMES,
    GO2_ACTION_JOINT_NAMES,
)
from Gurukul.tasks.manager_based.go2_apex.tracking_env_cfg import ObservationsCfg

from .flat_tracker_env_cfg import (
    GO2_PRIVILEGED_TRACKER_REFERENCE_TIME_OFFSETS,
    GO2_TRACKER_ONE_STEP_REFERENCE_TIME_OFFSETS,
    UnitreeGo2ApexFlatTrackerEnvCfg,
    privileged_tracker_reference_params,
    tracker_reference_params,
)

B2_Z1_CONTROL_DECIMATION = 4
B2_Z1_SIM_DT = 0.005
B2_Z1_FIXED_WRIST_JOINT_NAMES = ("joint6",)
B2_Z1_ARM_TRACKED_JOINT_NAMES = tuple(
    joint_name for joint_name in B2_Z1_ARM_JOINT_NAMES if joint_name not in B2_Z1_FIXED_WRIST_JOINT_NAMES
)
B2_Z1_FIXED_WRIST_REFERENCE_JOINT_NAMES = GO2_ACTION_JOINT_NAMES + B2_Z1_ARM_TRACKED_JOINT_NAMES
B2_Z1_FIXED_WRIST_ACTION_JOINT_NAMES = B2_Z1_FIXED_WRIST_REFERENCE_JOINT_NAMES + B2_Z1_GRIPPER_JOINT_NAMES
B2_Z1_FIXED_WRIST_OBS_JOINT_NAMES = (
    B2_Z1_FIXED_WRIST_REFERENCE_JOINT_NAMES + B2_Z1_FIXED_WRIST_JOINT_NAMES + B2_Z1_GRIPPER_JOINT_NAMES
)
B2_STANDING_LEG_JOINT_POS = (
    -0.1,
    0.8,
    -1.5,
    0.1,
    0.8,
    -1.5,
    -0.1,
    0.8,
    -1.5,
    0.1,
    0.8,
    -1.5,
)


def b2_standing_leg_joint_position_l2(env, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize B2 leg joints for moving away from the standing B2 pose."""
    asset = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    target_pos = torch.tensor(B2_STANDING_LEG_JOINT_POS, dtype=joint_pos.dtype, device=joint_pos.device).unsqueeze(0)
    return torch.sum(torch.square(joint_pos - target_pos), dim=1)


@configclass
class B2Z1ArmApexTrackerPolicyObservationsCfg(ObservationsCfg.PolicyCfg):
    """Actor observations for B2+Z1 reference-motion tracking."""

    reference_motion = ObsTerm(
        func=mdp.reference_motion_state,
        params=tracker_reference_params(B2_Z1_ACTION_JOINT_NAMES),
        clip=(-100.0, 100.0),
        scale=1.0,
    )


@configclass
class B2Z1ArmApexTrackerCriticObservationsCfg(ObservationsCfg.CriticCfg):
    """Critic observations for B2+Z1 reference-motion tracking."""

    reference_motion = ObsTerm(
        func=mdp.reference_motion_state,
        params=tracker_reference_params(B2_Z1_ACTION_JOINT_NAMES),
        clip=(-100.0, 100.0),
        scale=1.0,
    )


@configclass
class B2Z1ArmApexOneStepFutureTrackerPolicyObservationsCfg(ObservationsCfg.PolicyCfg):
    """Actor observations for B2+Z1 one-step-future reference-motion tracking."""

    reference_motion = ObsTerm(
        func=mdp.reference_motion_state,
        params=tracker_reference_params(
            B2_Z1_ACTION_JOINT_NAMES,
            time_offsets=GO2_TRACKER_ONE_STEP_REFERENCE_TIME_OFFSETS,
        ),
        clip=(-100.0, 100.0),
        scale=1.0,
    )


@configclass
class B2Z1ArmApexOneStepFutureTrackerHistoryPolicyObservationsCfg(
    B2Z1ArmApexOneStepFutureTrackerPolicyObservationsCfg
):
    """Deployable B2+Z1 one-step tracker observations with flattened temporal history."""

    def __post_init__(self):
        super().__post_init__()
        self.history_length = 5
        self.flatten_history_dim = True


@configclass
class B2Z1ArmApexOneStepFutureTrackerCriticObservationsCfg(ObservationsCfg.CriticCfg):
    """Critic observations for B2+Z1 one-step-future reference-motion tracking."""

    reference_motion = ObsTerm(
        func=mdp.reference_motion_state,
        params=tracker_reference_params(
            B2_Z1_ACTION_JOINT_NAMES,
            time_offsets=GO2_TRACKER_ONE_STEP_REFERENCE_TIME_OFFSETS,
        ),
        clip=(-100.0, 100.0),
        scale=1.0,
    )


@configclass
class B2Z1ArmApexPrivilegedTrackerObservationsCfg(B2Z1ArmApexTrackerCriticObservationsCfg):
    """Clean full-state observations for a privileged B2+Z1 APEX tracker teacher."""

    reference_motion = ObsTerm(
        func=mdp.reference_motion_state,
        params=privileged_tracker_reference_params(B2_Z1_ACTION_JOINT_NAMES),
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    reference_feet_yaw_b = ObsTerm(
        func=mdp.reference_body_state_yaw_b,
        params={
            "command_name": "motion",
            "time_offsets": GO2_PRIVILEGED_TRACKER_REFERENCE_TIME_OFFSETS,
            "body_names": FOOT_BODY_NAMES,
            "include_pos": True,
            "include_lin_vel": True,
            "include_ang_vel": False,
        },
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    reference_body_yaw_b = ObsTerm(
        func=mdp.reference_body_state_yaw_b,
        params={
            "command_name": "motion",
            "time_offsets": GO2_PRIVILEGED_TRACKER_REFERENCE_TIME_OFFSETS,
            "include_pos": True,
            "include_lin_vel": False,
            "include_ang_vel": False,
        },
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    reference_base_height = ObsTerm(
        func=mdp.reference_base_height,
        params={"command_name": "motion"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    reference_base_height_error = ObsTerm(
        func=mdp.reference_base_height_error,
        params={"command_name": "motion"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    robot_body_pos_b = ObsTerm(
        func=mdp.robot_body_pos_b,
        params={"command_name": "motion"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    robot_body_ori_b = ObsTerm(
        func=mdp.robot_body_ori_b,
        params={"command_name": "motion"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    reference_base_pos_b = ObsTerm(
        func=mdp.reference_base_pos_b,
        params={"command_name": "motion"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    reference_base_ori_b = ObsTerm(
        func=mdp.reference_base_ori_b,
        params={"command_name": "motion"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    root_lin_vel_w = ObsTerm(
        func=mdp.robot_root_lin_vel_w,
        params={"asset_cfg": SceneEntityCfg("robot")},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    root_ang_vel_w = ObsTerm(
        func=mdp.robot_root_ang_vel_w,
        params={"asset_cfg": SceneEntityCfg("robot")},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    joint_torques = ObsTerm(
        func=mdp.joint_torques,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=B2_Z1_ACTION_JOINT_NAMES, preserve_order=True)},
        clip=(-100.0, 100.0),
        scale=0.05,
    )
    feet_contact_state = ObsTerm(
        func=mdp.feet_contact_state,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES), "force_threshold": 1.0},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    motion_id = ObsTerm(
        func=mdp.motion_id_one_hot,
        params={"command_name": "motion"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )


@configclass
class B2Z1ArmApexFixedWristGripperPolicyObservationsCfg(ObservationsCfg.PolicyCfg):
    """Actor observations for B2+Z1 tracking with a fixed wrist and commanded gripper."""

    gripper_command = ObsTerm(
        func=mdp.joint_position_command,
        params={"command_name": "gripper"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    reference_motion = ObsTerm(
        func=mdp.reference_motion_state,
        params=tracker_reference_params(B2_Z1_FIXED_WRIST_REFERENCE_JOINT_NAMES),
        clip=(-100.0, 100.0),
        scale=1.0,
    )


@configclass
class B2Z1ArmApexFixedWristGripperCriticObservationsCfg(ObservationsCfg.CriticCfg):
    """Critic observations for B2+Z1 tracking with a fixed wrist and commanded gripper."""

    gripper_command = ObsTerm(
        func=mdp.joint_position_command,
        params={"command_name": "gripper"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    reference_motion = ObsTerm(
        func=mdp.reference_motion_state,
        params=tracker_reference_params(B2_Z1_FIXED_WRIST_REFERENCE_JOINT_NAMES),
        clip=(-100.0, 100.0),
        scale=1.0,
    )


@configclass
class B2Z1ArmApexFixedWristGripperPrivilegedObservationsCfg(B2Z1ArmApexPrivilegedTrackerObservationsCfg):
    """Privileged observations for B2+Z1 tracking with a fixed wrist and commanded gripper."""

    gripper_command = ObsTerm(
        func=mdp.joint_position_command,
        params={"command_name": "gripper"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    reference_motion = ObsTerm(
        func=mdp.reference_motion_state,
        params=privileged_tracker_reference_params(B2_Z1_FIXED_WRIST_REFERENCE_JOINT_NAMES),
        clip=(-100.0, 100.0),
        scale=1.0,
    )


@configclass
class UnitreeB2Z1ArmApexFlatTrackerEnvCfg(UnitreeGo2ApexFlatTrackerEnvCfg):
    """B2+Z1 APEX tracker with leg and Z1 arm reference tracking."""

    def __post_init__(self):
        super().__post_init__()

        # Use the original 50 Hz tracker control rate while debugging B2+Z1 tracking regressions.
        self.decimation = B2_Z1_CONTROL_DECIMATION
        self.sim.dt = B2_Z1_SIM_DT
        self.sim.render_interval = self.decimation

        self.scene.robot = UNITREE_B2_Z1_ARM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.sim.physx.gpu_max_rigid_patch_count = 32 * 2**15

        motion_root = f"{os.path.dirname(__file__)}/motion/npz"
        b2_z1_motion_glob = f"{motion_root}/b2_z1_motions/**/*.npz"
        self.commands.motion.motion_files = tuple(
            motion_file
            for motion_file in sorted(glob.glob(b2_z1_motion_glob, recursive=True))
            if not motion_file.endswith("/walk4_subject1.npz")
        )
        self.commands.motion.joint_names = list(B2_Z1_ACTION_JOINT_NAMES)
        self.commands.motion.debug_vis_show_arm_ee_reference = True
        self.rewards.imitate_joint_pos.weight = 0.0
        self.rewards.imitate_joint_pos_legs = RewTerm(
            func=mdp.motion_joint_position_error_exp,
            weight=0.5,
            params={
                "command_name": "motion",
                "std": 0.05,
                "stand_still_vel_threshold": 1.0e-6,
                "stand_still_ang_vel_threshold": 1.0e-6,
                "stand_still_lin_vel_z_threshold": 1.0e-6,
                "joint_names": list(GO2_ACTION_JOINT_NAMES),
            },
        )
        self.rewards.imitate_joint_pos_arms = RewTerm(
            func=mdp.motion_joint_position_error_exp,
            weight=2.0,
            params={
                "command_name": "motion",
                "std": 0.1,
                "stand_still_vel_threshold": 1.0e-6,
                "stand_still_ang_vel_threshold": 1.0e-6,
                "stand_still_lin_vel_z_threshold": 1.0e-6,
                "joint_names": list(B2_Z1_ARM_JOINT_NAMES),
            },
        )
        self.rewards.imitate_arm_ee_pos = RewTerm(
            func=mdp.motion_arm_ee_position_error_exp,
            weight=2.0,
            params={
                "command_name": "motion",
                "sigma": 0.04,
                "asset_cfg": SceneEntityCfg("robot", body_names=["link06"]),
                "align_to_robot_yaw": True,
            },
        )
        self.rewards.imitate_arm_ee_lin_vel = None
        self.rewards.imitate_foot_pos = RewTerm(
            func=mdp.motion_foot_position_error_exp,
            weight=1.0,
            params={
                "command_name": "motion",
                "sigma": 0.04,
                "stand_still_vel_threshold": 1.0e-6,
                "stand_still_ang_vel_threshold": 1.0e-6,
                "stand_still_lin_vel_z_threshold": 1.0e-6,
            },
        )
        self.rewards.joint_acc_l2.weight = 0.0
        self.rewards.joint_torques_l2.weight = 0.0
        self.rewards.action_rate_l2.weight = 0.0
        self.rewards.action_smoothness_l2.weight = 0.0
        self.rewards.joint_acc_l2_legs = RewTerm(
            func=mdp.joint_acc_l2,
            weight=-2.5e-7,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(GO2_ACTION_JOINT_NAMES))},
        )
        self.rewards.joint_acc_l2_arms = RewTerm(
            func=mdp.joint_acc_l2,
            weight=-5.0e-7,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(B2_Z1_ARM_JOINT_NAMES))},
        )
        self.rewards.joint_torques_l2_legs = RewTerm(
            func=mdp.joint_torques_l2,
            weight=-1.0e-5,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(GO2_ACTION_JOINT_NAMES))},
        )
        self.rewards.joint_torques_l2_arms = RewTerm(
            func=mdp.joint_torques_l2,
            weight=-2.0e-5,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(B2_Z1_ARM_JOINT_NAMES))},
        )
        self.rewards.action_rate_l2_legs = RewTerm(
            func=mdp.action_rate_l2_selected,
            weight=-1.0e-2,
            params={"joint_names": list(GO2_ACTION_JOINT_NAMES)},
        )
        self.rewards.action_rate_l2_arms = RewTerm(
            func=mdp.action_rate_l2_selected,
            weight=-2.5e-2,
            params={"joint_names": list(B2_Z1_ARM_JOINT_NAMES)},
        )
        self.rewards.action_smoothness_l2_legs = RewTerm(
            func=mdp.action_smoothness_l2_selected,
            weight=-1.0e-2,
            params={"joint_names": list(GO2_ACTION_JOINT_NAMES)},
        )
        self.rewards.action_smoothness_l2_arms = RewTerm(
            func=mdp.action_smoothness_l2_selected,
            weight=-2.5e-2,
            params={"joint_names": list(B2_Z1_ARM_JOINT_NAMES)},
        )
        self.rewards.joint_pos_limits_legs = RewTerm(
            func=mdp.motion_joint_pos_limits_l1,
            weight=-2.0,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(GO2_ACTION_JOINT_NAMES))},
        )
        self.rewards.default_leg_joint_pos = RewTerm(
            func=b2_standing_leg_joint_position_l2,
            weight=-0.25e-3,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=list(GO2_ACTION_JOINT_NAMES)),
            },
        )
        self.rewards.feet_air_time = None
        self.rewards.feet_air_time_variance = None
        self.rewards.legs_distance = RewTerm(
            func=mdp.motion_legs_distance,
            weight=-1.5,
            params={
                "min_distance": 0.38,
                "asset_cfg": SceneEntityCfg("robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
            },
        )
        self.rewards.feet_contact_forces = RewTerm(
            func=mdp.motion_feet_contact_forces_penalty,
            weight=-1.0e-4,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
                "max_contact_force": 500.0,
            },
        )
        self.rewards.undesired_contacts = RewTerm(
            func=mdp.undesired_contacts,
            weight=-3.0,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[r"^(?!.*_foot$).+$"],
                ),
                "threshold": 1.0,
            },
        )
        self.rewards.airborne_contact.weight = 0.0
        self.rewards.track_command_ang_vel_z.weight = 0.0
        self.actions.joint_pos.joint_names = list(B2_Z1_ACTION_JOINT_NAMES)
        self.actions.joint_pos.preserve_order = True
        self.actions.joint_pos.scale = {
            r"^(FL|FR|RL|RR)_hip_joint$": 0.125,
            r"^(FL|FR|RL|RR)_(thigh|calf)_joint$": 0.25,
            r"^joint[1-6]$": 0.05,
        }
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}

        self.events.randomize_reset_joints.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=list(B2_Z1_ACTION_JOINT_NAMES)
        )
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (0.9, 1.15)
        self.events.randomize_rigid_body_mass_base.params["operation"] = "scale"
        self.events.randomize_rigid_body_mass_others.params["mass_distribution_params"] = (0.85, 1.15)
        self.events.randomize_com_positions.params["com_range"] = {
            "x": (-0.04, 0.04),
            "y": (-0.03, 0.03),
            "z": (-0.03, 0.03),
        }
        self.events.randomize_apply_external_force_torque.params["force_range"] = (-40.0, 40.0)
        self.events.randomize_apply_external_force_torque.params["torque_range"] = (-20.0, 20.0)
        self.events.randomize_z1_mount_joints = EventTerm(
            func=mdp.randomize_fixed_joint_defaults,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["z1_mount_x_joint", "z1_mount_z_joint"]),
                "position_ranges": {
                    "z1_mount_x_joint": (-0.003, 0.003),
                    "z1_mount_z_joint": (-0.002, 0.002),
                },
                "velocity_range": (0.0, 0.0),
            },
        )
        self.events.randomize_actuator_gains.params["asset_cfg"].joint_names = list(B2_Z1_ACTION_JOINT_NAMES)
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.85, 1.15)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.85, 1.15)
        self.observations.policy = B2Z1ArmApexTrackerPolicyObservationsCfg()
        self.observations.critic = B2Z1ArmApexTrackerCriticObservationsCfg()
        self.observations.privileged = B2Z1ArmApexPrivilegedTrackerObservationsCfg()

        for obs_group in (self.observations.policy, self.observations.critic, self.observations.privileged):
            obs_group.joint_pos.params["asset_cfg"].joint_names = list(B2_Z1_ACTION_JOINT_NAMES)
            obs_group.joint_vel.params["asset_cfg"].joint_names = list(B2_Z1_ACTION_JOINT_NAMES)

        for obs_group in (self.observations.critic, self.observations.privileged):
            obs_group.reference_joint_pos.params["joint_names"] = list(B2_Z1_ACTION_JOINT_NAMES)

        self.observations.policy.reference_motion.params["add_noise"] = True
        self.observations.policy.reference_motion.params["noise_std"] = 0.01


@configclass
class UnitreeB2Z1ArmApexFlatPrivilegedTrackerEnvCfg(UnitreeB2Z1ArmApexFlatTrackerEnvCfg):
    """Task surface for training a clean full-state B2+Z1 APEX tracker teacher."""

    pass


@configclass
class UnitreeB2Z1ArmApexFlatFixedWristGripperTrackerEnvCfg(UnitreeB2Z1ArmApexFlatTrackerEnvCfg):
    """B2+Z1 APEX tracker with randomized fixed wrist and commanded open/close gripper."""

    def __post_init__(self):
        super().__post_init__()

        self.commands.gripper = mdp.BinaryJointPositionCommandCfg(
            asset_name="robot",
            joint_names=list(B2_Z1_GRIPPER_JOINT_NAMES),
            resampling_time_range=(1.0, 3.0),
            low_position=(-1.2,),
            high_position=(0.0,),
            high_prob=0.5,
        )

        self.actions.joint_pos.class_type = JointPositionAction
        self.actions.joint_pos.joint_names = list(B2_Z1_FIXED_WRIST_ACTION_JOINT_NAMES)
        self.actions.joint_pos.preserve_order = True
        self.actions.joint_pos.scale = {
            r"^(FL|FR|RL|RR)_hip_joint$": 0.125,
            r"^(FL|FR|RL|RR)_(thigh|calf)_joint$": 0.25,
            r"^joint[1-5]$": 0.05,
            r"^jointGripper$": 0.05,
        }

        self.commands.motion.joint_names = list(B2_Z1_FIXED_WRIST_REFERENCE_JOINT_NAMES)
        self.observations.policy = B2Z1ArmApexFixedWristGripperPolicyObservationsCfg()
        self.observations.critic = B2Z1ArmApexFixedWristGripperCriticObservationsCfg()
        self.observations.privileged = B2Z1ArmApexFixedWristGripperPrivilegedObservationsCfg()

        for obs_group in (self.observations.policy, self.observations.critic, self.observations.privileged):
            obs_group.joint_pos.params["asset_cfg"].joint_names = list(B2_Z1_FIXED_WRIST_OBS_JOINT_NAMES)
            obs_group.joint_vel.params["asset_cfg"].joint_names = list(B2_Z1_FIXED_WRIST_OBS_JOINT_NAMES)

        for obs_group in (self.observations.critic, self.observations.privileged):
            obs_group.reference_joint_pos.params["joint_names"] = list(B2_Z1_FIXED_WRIST_REFERENCE_JOINT_NAMES)

        self.events.randomize_reset_joints.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=list(B2_Z1_FIXED_WRIST_ACTION_JOINT_NAMES)
        )
        self.events.randomize_fixed_wrist = EventTerm(
            func=mdp.randomize_fixed_joint_defaults,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=list(B2_Z1_FIXED_WRIST_JOINT_NAMES)),
                "position_ranges": {"joint6": (-1.57, 1.57)},
                "velocity_range": (0.0, 0.0),
            },
        )
        self.events.randomize_actuator_gains.params["asset_cfg"].joint_names = list(
            B2_Z1_FIXED_WRIST_ACTION_JOINT_NAMES
        )

        self.rewards.imitate_joint_pos_arms = None
        self.rewards.joint_acc_l2_arms.params["asset_cfg"].joint_names = list(B2_Z1_ARM_TRACKED_JOINT_NAMES)
        self.rewards.joint_torques_l2_arms.params["asset_cfg"].joint_names = list(B2_Z1_ARM_TRACKED_JOINT_NAMES)
        self.rewards.action_rate_l2_arms.params["joint_names"] = list(
            B2_Z1_ARM_TRACKED_JOINT_NAMES + B2_Z1_GRIPPER_JOINT_NAMES
        )
        self.rewards.action_smoothness_l2_arms.params["joint_names"] = list(
            B2_Z1_ARM_TRACKED_JOINT_NAMES + B2_Z1_GRIPPER_JOINT_NAMES
        )
        self.rewards.track_gripper_command = RewTerm(
            func=mdp.joint_position_command_tracking_exp,
            weight=1.0,
            params={
                "command_name": "gripper",
                "sigma": 0.01,
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=list(B2_Z1_GRIPPER_JOINT_NAMES), preserve_order=True
                ),
            },
        )

        self.observations.policy.reference_motion.params["add_noise"] = True
        self.observations.policy.reference_motion.params["noise_std"] = 0.01


@configclass
class UnitreeB2Z1ArmApexFlatFixedWristGripperPrivilegedTrackerEnvCfg(
    UnitreeB2Z1ArmApexFlatFixedWristGripperTrackerEnvCfg
):
    """Task surface for training a privileged fixed-wrist/gripper-command B2+Z1 tracker teacher."""

    pass


@configclass
class UnitreeB2Z1ArmApexFlatOneStepFutureTrackerEnvCfg(UnitreeB2Z1ArmApexFlatTrackerEnvCfg):
    """B2+Z1 one-step-future tracker student for distillation from the privileged tracker teacher."""

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy = B2Z1ArmApexOneStepFutureTrackerPolicyObservationsCfg()
        self.observations.critic = B2Z1ArmApexOneStepFutureTrackerCriticObservationsCfg()
        self.observations.privileged = B2Z1ArmApexPrivilegedTrackerObservationsCfg()

        for obs_group in (self.observations.policy, self.observations.critic, self.observations.privileged):
            obs_group.joint_pos.params["asset_cfg"].joint_names = list(B2_Z1_ACTION_JOINT_NAMES)
            obs_group.joint_vel.params["asset_cfg"].joint_names = list(B2_Z1_ACTION_JOINT_NAMES)

        for obs_group in (self.observations.critic, self.observations.privileged):
            obs_group.reference_joint_pos.params["joint_names"] = list(B2_Z1_ACTION_JOINT_NAMES)

        self.observations.policy.base_ang_vel.noise.n_min = -0.8
        self.observations.policy.base_ang_vel.noise.n_max = 0.8
        self.observations.policy.projected_gravity.noise.n_min = -0.12
        self.observations.policy.projected_gravity.noise.n_max = 0.12
        self.observations.policy.joint_pos.noise.n_min = -0.03
        self.observations.policy.joint_pos.noise.n_max = 0.03
        self.observations.policy.joint_vel.noise.n_min = -3.0
        self.observations.policy.joint_vel.noise.n_max = 3.0
        self.observations.policy.reference_motion.params["add_noise"] = True
        self.observations.policy.reference_motion.params["noise_std"] = 0.02


@configclass
class UnitreeB2Z1ArmApexFlatOneStepFutureTrackerHistoryEnvCfg(
    UnitreeB2Z1ArmApexFlatOneStepFutureTrackerEnvCfg
):
    """B2+Z1 one-step tracker student with deployable observation/reference history for distillation and sim2sim."""

    def __post_init__(self):
        super().__post_init__()

        self.observations.policy = B2Z1ArmApexOneStepFutureTrackerHistoryPolicyObservationsCfg()
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = list(B2_Z1_ACTION_JOINT_NAMES)
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = list(B2_Z1_ACTION_JOINT_NAMES)
        self.observations.policy.base_ang_vel.noise.n_min = -0.8
        self.observations.policy.base_ang_vel.noise.n_max = 0.8
        self.observations.policy.projected_gravity.noise.n_min = -0.12
        self.observations.policy.projected_gravity.noise.n_max = 0.12
        self.observations.policy.joint_pos.noise.n_min = -0.03
        self.observations.policy.joint_pos.noise.n_max = 0.03
        self.observations.policy.joint_vel.noise.n_min = -3.0
        self.observations.policy.joint_vel.noise.n_max = 3.0
        self.observations.policy.reference_motion.params["add_noise"] = True
        self.observations.policy.reference_motion.params["noise_std"] = 0.02
