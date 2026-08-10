from __future__ import annotations

import os

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import Gurukul.tasks.manager_based.go2_apex.mdp as mdp
from Gurukul.tasks.manager_based.go2_apex.constants import FOOT_BODY_NAMES, GO2_ACTION_JOINT_NAMES
from Gurukul.tasks.manager_based.go2_apex.tracking_env_cfg import ObservationsCfg

from .flat_env_cfg import UnitreeGo2ApexFlatEnvCfg

GO2_TRACKER_REFERENCE_TIME_OFFSETS = (0, 1, 2, 5, 10)
GO2_TRACKER_ONE_STEP_REFERENCE_TIME_OFFSETS = (0, 1)
GO2_PRIVILEGED_TRACKER_REFERENCE_TIME_OFFSETS = (-4, -2, 0, 1, 2, 5, 10)
GO2_TRACKER_REFERENCE_NOISE_STD = 0.01


def tracker_reference_params(
    joint_names=GO2_ACTION_JOINT_NAMES,
    time_offsets=GO2_TRACKER_REFERENCE_TIME_OFFSETS,
    add_noise: bool = False,
    noise_std: float = GO2_TRACKER_REFERENCE_NOISE_STD,
) -> dict:
    return {
        "command_name": "motion",
        "time_offsets": time_offsets,
        "joint_names": joint_names,
        "include_joint_pos": True,
        "include_joint_vel": False,
        "include_base_lin_vel": True,
        "include_base_ang_vel": True,
        "include_base_quat": True,
        "include_command_vel": False,
        "add_noise": add_noise,
        "noise_std": noise_std,
    }


def privileged_tracker_reference_params(
    joint_names=GO2_ACTION_JOINT_NAMES,
    time_offsets=GO2_PRIVILEGED_TRACKER_REFERENCE_TIME_OFFSETS,
) -> dict:
    params = tracker_reference_params(joint_names=joint_names, time_offsets=time_offsets)
    params.update(
        {
            "include_joint_vel": True,
            "include_command_vel": True,
            "add_noise": False,
        }
    )
    return params


@configclass
class Go2ApexTrackerPolicyObservationsCfg(ObservationsCfg.PolicyCfg):
    """Actor observations with configurable current/future reference motion features."""

    reference_motion = ObsTerm(
        func=mdp.reference_motion_state,
        params=tracker_reference_params(),
        clip=(-100.0, 100.0),
        scale=1.0,
    )


@configclass
class Go2ApexTrackerCriticObservationsCfg(ObservationsCfg.CriticCfg):
    """Critic observations with the same future reference bundle used by the actor."""

    reference_motion = ObsTerm(
        func=mdp.reference_motion_state,
        params=tracker_reference_params(),
        clip=(-100.0, 100.0),
        scale=1.0,
    )


@configclass
class Go2ApexPrivilegedTrackerObservationsCfg(Go2ApexTrackerCriticObservationsCfg):
    """Clean full-state observations for a privileged APEX tracker teacher."""

    reference_motion = ObsTerm(
        func=mdp.reference_motion_state,
        params=privileged_tracker_reference_params(),
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
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2_ACTION_JOINT_NAMES, preserve_order=True)},
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
class UnitreeGo2ApexFlatTrackerEnvCfg(UnitreeGo2ApexFlatEnvCfg):
    """Go2 APEX flat tracker that conditions the policy on reference motion data."""

    def __post_init__(self):
        super().__post_init__()
        motion_root = f"{os.path.dirname(__file__)}/motion/npz"
        self.commands.motion.motion_files = (
            # f"{motion_root}/animal_mocap/*.npz",
            # f"{motion_root}/gait_switch/*.npz",
            # f"{motion_root}/walk_these_ways/*.npz",
            f"{motion_root}/cross_morpho_replay_added/*.npz",
            # f"{motion_root}/learn_diverse_quad_loco/combined/*_diverse_combined.npz",
        )
        # Training samples a random motion id, then a random frame inside that motion on every reset/resample.
        self.commands.motion.sample_motions_uniformly = True
        self.commands.motion.use_adaptive_sampling = False
        self.commands.motion.reference_state_init = False
        self.commands.motion.pose_range = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.commands.motion.velocity_range = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.commands.motion.joint_position_range = (0.0, 0.0)
        self.commands.motion.use_reference_command_channels_directly = False
        self.commands.motion.command_velocity_frame = "base"
        self.commands.motion.command_lin_vel_x_offset_range = (-0.7, 0.1)
        self.commands.motion.command_lin_vel_y_offset_range = (0.0, 0.0)
        self.commands.motion.command_ang_vel_z_offset_range = (0.0, 0.0)
        self.commands.motion.command_lin_vel_x_clip = (-1.0, 1.5)
        self.commands.motion.command_lin_vel_y_clip = (-1.0e9, 1.0e9)
        self.commands.motion.command_ang_vel_z_clip = (-1.0e9, 1.0e9)
        self.commands.motion.adaptive_sampling_strategy = "tracking_error"
        self.commands.motion.adaptive_kernel_size = 5
        self.commands.motion.adaptive_alpha = 0.01
        self.commands.motion.adaptive_uniform_ratio = 0.2
        self.commands.motion.adaptive_tracking_error_weights = {
            "joint_pos": 1.0,
            "base_pos": 0.5,
            "base_orientation": 0.5,
            "command_lin_vel_xy": 0.25,
            "command_ang_vel_z": 0.25,
        }
        # Tracker resets currently use the deterministic Go2 reset state. Adaptive sampling still focuses hard
        # reference frames through the command phase without teleporting the robot into sampled reference states.
        self.events.randomize_reset_joints.params["position_range"] = (0.0, 0.0)
        self.events.randomize_reset_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.events.randomize_reset_base.params["velocity_range"] = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        # self.events.randomize_com_positions.params["com_range"] = {
        #     "x": (-0.015, 0.015),
        #     "y": (-0.015, 0.015),
        #     "z": (-0.015, 0.015),
        # }
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.9, 1.1)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.9, 1.1)
        self.events.randomize_push_robot.interval_range_s = (10.0, 15.0)
        self.events.randomize_push_robot.params = {
            "velocity_range": {
                "x": (-0.2, 0.2),
                "y": (-0.2, 0.2),
                "roll": (-0.15, 0.15),
                "pitch": (-0.15, 0.15),
                "yaw": (-0.2, 0.2),
            }
        }

        self.observations.policy = Go2ApexTrackerPolicyObservationsCfg()
        self.observations.critic = Go2ApexTrackerCriticObservationsCfg()
        self.observations.privileged = Go2ApexPrivilegedTrackerObservationsCfg()
        self.observations.policy.reference_motion.params["add_noise"] = True
        self.observations.policy.reference_motion.params["noise_std"] = 0.01


@configclass
class UnitreeGo2ApexFlatPrivilegedTrackerEnvCfg(UnitreeGo2ApexFlatTrackerEnvCfg):
    """Task surface for training a clean full-state APEX tracker teacher."""

    pass


@configclass
class Go2ApexOneStepFutureTrackerPolicyObservationsCfg(ObservationsCfg.PolicyCfg):
    """Actor observations with current and one-frame-future reference motion features."""

    reference_motion = ObsTerm(
        func=mdp.reference_motion_state,
        params=tracker_reference_params(time_offsets=GO2_TRACKER_ONE_STEP_REFERENCE_TIME_OFFSETS),
        clip=(-100.0, 100.0),
        scale=1.0,
    )


@configclass
class Go2ApexOneStepFutureTrackerHistoryPolicyObservationsCfg(Go2ApexOneStepFutureTrackerPolicyObservationsCfg):
    """Deployable one-step tracker observations with flattened temporal history."""

    def __post_init__(self):
        super().__post_init__()
        self.history_length = 5
        self.flatten_history_dim = True


@configclass
class Go2ApexOneStepFutureTrackerCriticObservationsCfg(ObservationsCfg.CriticCfg):
    """Critic observations with current and one-frame-future reference motion features."""

    reference_motion = ObsTerm(
        func=mdp.reference_motion_state,
        params=tracker_reference_params(time_offsets=GO2_TRACKER_ONE_STEP_REFERENCE_TIME_OFFSETS),
        clip=(-100.0, 100.0),
        scale=1.0,
    )


@configclass
class UnitreeGo2ApexFlatOneStepFutureTrackerEnvCfg(UnitreeGo2ApexFlatTrackerEnvCfg):
    """Go2 APEX tracker student variant with only one future reference frame.

    The deployable policy/critic groups use the one-step reference layout, but the privileged group remains the full
    teacher layout so distillation can load the full privileged tracker teacher checkpoint.
    """

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.command_lin_vel_x_clip = (-1.0, 1.5)
        self.observations.policy = Go2ApexOneStepFutureTrackerPolicyObservationsCfg()
        self.observations.critic = Go2ApexOneStepFutureTrackerCriticObservationsCfg()
        self.observations.privileged = Go2ApexPrivilegedTrackerObservationsCfg()
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
class UnitreeGo2ApexFlatOneStepFutureTrackerHistoryEnvCfg(UnitreeGo2ApexFlatOneStepFutureTrackerEnvCfg):
    """One-step tracker student with deployable observation/reference history for distillation and sim2sim."""

    def __post_init__(self):
        super().__post_init__()
        self.observations.policy = Go2ApexOneStepFutureTrackerHistoryPolicyObservationsCfg()
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
