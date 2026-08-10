from __future__ import annotations

import copy
import os

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import Gurukul.tasks.manager_based.go2_apex.mdp as mdp
from Gurukul.assets.unitree import UNITREE_GO2_D1_ARM_APEX_CFG
from Gurukul.tasks.manager_based.go2_apex.constants import (
    FOOT_BODY_NAMES,
    GO2_ACTION_JOINT_NAMES,
    GO2_D1_ACTION_JOINT_NAMES,
    GO2_D1_ARM_JOINT_NAMES,
    GO2_D1_MOTION_JOINT_NAMES,
    GO2_D1_PICK_STOW_ACTION_JOINT_NAMES,
)
from Gurukul.tasks.manager_based.go2_apex.tracking_env_cfg import ObservationsCfg

from .flat_tracker_env_cfg import (
    Go2ApexPrivilegedTrackerObservationsCfg,
    UnitreeGo2ApexFlatTrackerEnvCfg,
    privileged_tracker_reference_params,
    tracker_reference_params,
)

D1_ARM_HARDWARE_VELOCITY_LIMITS = (1.05, 1.05, 1.05, 1.73, 1.73, 1.73)
D1_ARM_NONBINDING_SIM_VELOCITY_LIMIT = 100.0
GO2_D1_SIM_DT = 0.005
GO2_D1_CONTROL_DECIMATION = 4
GO2_D1_CONTROL_FREQUENCY_HZ = 1.0 / (GO2_D1_SIM_DT * GO2_D1_CONTROL_DECIMATION)
D1_ARM_COMMAND_PERIOD_S = 0.1
D1_ARM_COMMAND_FREQUENCY_HZ = 1.0 / D1_ARM_COMMAND_PERIOD_S
D1_COMMAND_JOINT_NAMES = GO2_D1_ARM_JOINT_NAMES + ("arm_7_1_joint",)
_D1_NONADJACENT_SELF_COLLISION_FILTERS = (
    ("base_link", ("Link2", "Link3", "Link4", "Link5", "Link6", "Link7_1", "Link7_2")),
    ("Link1", ("Link3", "Link4", "Link5", "Link6", "Link7_1", "Link7_2")),
    # The coarse collision proxies for Link2 overlap Link4/Link5 in the valid
    # ready pose and hello reference. Filter those two known proxy overlaps.
    ("Link2", ("Link6", "Link7_1", "Link7_2")),
    ("Link3", ("Link5", "Link6", "Link7_1", "Link7_2")),
    ("Link4", ("Link6", "Link7_1", "Link7_2")),
    ("Link5", ("Link7_1", "Link7_2")),
)
_D1_SELF_COLLISION_SENSOR_NAMES = tuple(
    f"d1_nonadjacent_contact_{source_body.lower()}" for source_body, _ in _D1_NONADJACENT_SELF_COLLISION_FILTERS
)
_PICK_OBJECT_REFERENCE_TIME_OFFSETS = (0, 1, 2, 5, 10)
_PICK_PRIVILEGED_REFERENCE_TIME_OFFSETS = (-20, -10, -5, -2, 0, 1, 2, 5, 10, 20, 40)
_PICK_LEFT_OBJECT_CONTACT_SENSOR = "d1_left_gripper_object_contact"
_PICK_RIGHT_OBJECT_CONTACT_SENSOR = "d1_right_gripper_object_contact"
_PICK_OBJECT_SIZE = (0.135, 0.055, 0.255)
_PICK_OBJECT_SIZE_SCALE_RANGE = (0.70, 0.90)
_PICK_OBJECT_MASS_RANGE = (0.015, 0.040)
_PICK_OBJECT_NOMINAL_MASS = 0.0275
_PICK_GRASP_TERMINATION_WARMUP_ITERATIONS = 250
_PICK_GRASP_TERMINATION_RAMP_END_ITERATION = 1000
_PICK_GRASP_TIMEOUT_START_S = 2.0
_PICK_GRASP_TIMEOUT_END_S = 0.5
_CAN_OBJECT_SIZE = (0.053, 0.053, 0.135)
_CAN_OBJECT_SIZE_SCALE_RANGE = (0.92, 1.08)
_CAN_OBJECT_MASS_RANGE = (0.18, 0.32)
_CAN_OBJECT_NOMINAL_MASS = 0.25


@configclass
class Go2D1ArmApexTrackerPolicyObservationsCfg(ObservationsCfg.PolicyCfg):
    """Actor observations for Go2+D1 reference-motion tracking."""

    skill = ObsTerm(
        func=mdp.motion_skill,
        params={"command_name": "motion"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    reference_motion = ObsTerm(
        func=mdp.reference_motion_state,
        params=tracker_reference_params(GO2_D1_ACTION_JOINT_NAMES),
        clip=(-100.0, 100.0),
        scale=1.0,
    )


@configclass
class Go2D1ArmApexTrackerCriticObservationsCfg(Go2ApexPrivilegedTrackerObservationsCfg):
    """Simulator-only critic observations for Go2+D1 reference-motion tracking."""

    skill = ObsTerm(
        func=mdp.motion_skill,
        params={"command_name": "motion"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    reference_motion = ObsTerm(
        func=mdp.reference_motion_state,
        params=privileged_tracker_reference_params(GO2_D1_ACTION_JOINT_NAMES),
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    feet_contact_forces_b = ObsTerm(
        func=mdp.contact_forces_b,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=list(FOOT_BODY_NAMES)),
            "asset_cfg": SceneEntityCfg("robot"),
            "normalize": 100.0,
        },
        clip=(-100.0, 100.0),
        scale=1.0,
    )


@configclass
class Go2D1ArmApexPrivilegedTrackerObservationsCfg(Go2D1ArmApexTrackerCriticObservationsCfg):
    """Clean full-state observations with all tracked Go2+D1 joints."""


@configclass
class UnitreeGo2D1ArmApexFlatTrackerEnvCfg(UnitreeGo2ApexFlatTrackerEnvCfg):
    """Direct PPO Go2+D1 APEX tracker with deployable actor observations."""

    def __post_init__(self):
        super().__post_init__()

        # Keep policy inference and reference advancement at 50 Hz. Go2 consumes
        # each policy target, while the seven D1 packet targets cross a separate
        # 10 Hz zero-order-hold command boundary below.
        self.sim.dt = GO2_D1_SIM_DT
        self.decimation = GO2_D1_CONTROL_DECIMATION
        self.sim.render_interval = self.decimation
        if GO2_D1_CONTROL_FREQUENCY_HZ != 50.0:
            raise ValueError(
                "Go2+D1 APEX requires a 50 Hz policy control rate, got "
                f"{GO2_D1_CONTROL_FREQUENCY_HZ:g} Hz."
            )
        if GO2_D1_CONTROL_FREQUENCY_HZ % D1_ARM_COMMAND_FREQUENCY_HZ != 0.0:
            raise ValueError(
                "The D1 arm command rate must divide the Go2+D1 policy rate exactly; got "
                f"{D1_ARM_COMMAND_FREQUENCY_HZ:g} Hz and {GO2_D1_CONTROL_FREQUENCY_HZ:g} Hz."
            )

        self.scene.robot = UNITREE_GO2_D1_ARM_APEX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.sim.physx.gpu_max_rigid_patch_count = 32 * 2**15

        motion_root = f"{os.path.dirname(__file__)}/motion/npz"
        self.commands.motion.motion_files = (f"{motion_root}/go2_d1/**/*.npz",)
        self.commands.motion.joint_names = list(GO2_D1_MOTION_JOINT_NAMES)
        self.commands.motion.gripper_joint_names = ("arm_7_1_joint",)
        self.commands.motion.arm_ee_body_names = ("Link7_1", "Link7_2")
        self.commands.motion.arm_ee_orientation_body_name = "Link6"
        self.commands.motion.debug_vis_show_arm_ee_reference = True
        # Manipulation clips own their locomotion command channels. Do not inherit the
        # generic Go2 tracker's random forward-velocity offset: stationary clips such
        # as wave_hello must remain stationary during both training and playback.
        self.commands.motion.use_reference_command_channels_directly = True
        self.commands.motion.command_lin_vel_x_offset_range = (0.0, 0.0)
        self.commands.motion.command_lin_vel_y_offset_range = (0.0, 0.0)
        self.commands.motion.command_ang_vel_z_offset_range = (0.0, 0.0)

        # A stationary manipulation clip still needs to track its moving arm. The shared
        # locomotion reward deliberately falls back to the default pose at zero command,
        # so split the leg and arm terms and always use the motion reference here.
        self.rewards.imitate_joint_pos.weight = 0.0
        self.rewards.imitate_joint_pos_legs = RewTerm(
            func=mdp.motion_joint_position_error_exp,
            weight=2.0,
            params={
                "command_name": "motion",
                # At std=0.05 the reward was already ~0.04 at 0.09-rad RMS
                # error and supplied almost no useful correction gradient.
                "std": 0.12,
                "stand_still_vel_threshold": None,
                "stand_still_ang_vel_threshold": None,
                "stand_still_lin_vel_z_threshold": None,
                "joint_names": list(GO2_ACTION_JOINT_NAMES),
            },
        )
        self.rewards.imitate_joint_pos_arms = None
        self.rewards.imitate_arm_joint_pos_proximal = RewTerm(
            func=mdp.motion_joint_position_error_exp_per_joint,
            weight=1.5,
            params={
                "command_name": "motion",
                "std": 0.12,
                "joint_names": list(GO2_D1_ARM_JOINT_NAMES[:3]),
            },
        )
        self.rewards.imitate_arm_joint_pos_wrist = RewTerm(
            func=mdp.motion_joint_position_error_exp_per_joint,
            weight=2.5,
            params={
                "command_name": "motion",
                "std": 0.08,
                "joint_names": list(GO2_D1_ARM_JOINT_NAMES[3:]),
            },
        )
        self.rewards.imitate_arm_ee_pos = RewTerm(
            func=mdp.motion_arm_ee_position_error_exp,
            weight=2.0,
            params={
                "command_name": "motion",
                "sigma": 0.04,
                "asset_cfg": SceneEntityCfg("robot", body_names=["Link7_1", "Link7_2"]),
                "align_to_robot_yaw": True,
            },
        )
        self.rewards.imitate_arm_ee_orientation = RewTerm(
            func=mdp.motion_arm_ee_orientation_error_exp,
            weight=1.5,
            params={
                "command_name": "motion",
                "std": 0.2,
                "asset_cfg": SceneEntityCfg("robot", body_names=["Link6"]),
            },
        )
        self.rewards.imitate_gripper_joint_pos = RewTerm(
            func=mdp.motion_joint_position_error_exp,
            weight=1.0,
            params={
                "command_name": "motion",
                "std": 0.005,
                "stand_still_vel_threshold": None,
                "stand_still_ang_vel_threshold": None,
                "stand_still_lin_vel_z_threshold": None,
                "joint_names": ["arm_7_1_joint"],
            },
        )
        # Supervise the policy's own full-body targets before the temporary
        # DecAP motion prior is added. This ensures the learned policy, rather
        # than a permanent reference residual, owns the motion by the end of
        # the DecAP schedule.
        self.rewards.imitate_leg_policy_targets = RewTerm(
            func=mdp.motion_joint_action_error_l2,
            weight=-2.0,
            params={
                "command_name": "motion",
                "joint_names": list(GO2_ACTION_JOINT_NAMES),
            },
        )
        self.rewards.imitate_arm_policy_targets = None
        self.rewards.imitate_arm_policy_targets_proximal = RewTerm(
            func=mdp.motion_joint_action_error_l2,
            weight=-0.25,
            params={
                "command_name": "motion",
                "joint_names": list(GO2_D1_ARM_JOINT_NAMES[:3]),
                "normalize_by_action_scale": True,
            },
        )
        self.rewards.imitate_arm_policy_targets_wrist = RewTerm(
            func=mdp.motion_joint_action_error_l2,
            weight=-0.75,
            params={
                "command_name": "motion",
                "joint_names": list(GO2_D1_ARM_JOINT_NAMES[3:]),
                "normalize_by_action_scale": True,
            },
        )
        self.rewards.imitate_gripper_policy_targets = RewTerm(
            func=mdp.motion_joint_action_error_l2,
            weight=-2.0e-2,
            params={
                "command_name": "motion",
                "joint_names": ["arm_7_1_joint"],
                "normalize_by_action_scale": True,
            },
        )
        self.rewards.track_command_lin_vel_xy.weight = 3.0
        # Arm-only regularization suppresses high-frequency policy targets in
        # addition to the shared causal servo response; neither path hard-caps
        # target velocity.
        self.rewards.d1_arm_action_rate = RewTerm(
            func=mdp.action_rate_l2_selected,
            weight=-4.0e-2,
            params={"joint_names": list(GO2_D1_ARM_JOINT_NAMES)},
        )
        self.rewards.d1_arm_action_smoothness = RewTerm(
            func=mdp.action_smoothness_l2_selected,
            weight=-4.0e-2,
            params={"joint_names": list(GO2_D1_ARM_JOINT_NAMES)},
        )
        self.rewards.d1_arm_joint_velocity = RewTerm(
            func=mdp.joint_vel_l2,
            weight=-5.0e-4,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(GO2_D1_ARM_JOINT_NAMES))},
        )
        self.rewards.d1_gripper_action_rate = RewTerm(
            func=mdp.action_rate_l2_selected,
            weight=-2.0e-2,
            params={"joint_names": ["arm_7_1_joint"]},
        )
        self.rewards.d1_gripper_action_smoothness = RewTerm(
            func=mdp.action_smoothness_l2_selected,
            weight=-2.0e-2,
            params={"joint_names": ["arm_7_1_joint"]},
        )
        # The prismatic gripper coordinate is measured in metres and must never
        # enter rotary acceleration or torque norms. Keep the shared penalties
        # leg-only and log the explicit-servo rotary arm separately at a much
        # smaller acceleration weight.
        self.rewards.joint_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=list(GO2_ACTION_JOINT_NAMES)
        )
        self.rewards.joint_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=list(GO2_ACTION_JOINT_NAMES)
        )
        self.rewards.d1_arm_joint_acc_l2 = RewTerm(
            func=mdp.joint_acc_l2,
            weight=-1.0e-9,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(GO2_D1_ARM_JOINT_NAMES))},
        )
        self.rewards.d1_arm_joint_torques_l2 = RewTerm(
            func=mdp.joint_torques_l2,
            weight=-1.0e-5,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=list(GO2_D1_ARM_JOINT_NAMES))},
        )
        d1_arm_cfg = SceneEntityCfg(
            "robot",
            joint_names=list(GO2_D1_ARM_JOINT_NAMES),
            preserve_order=True,
        )
        self.rewards.d1_arm_joint_velocity_limits = RewTerm(
            func=mdp.joint_velocity_soft_limits_l2,
            weight=-0.25,
            params={
                "asset_cfg": d1_arm_cfg,
                "velocity_limits": D1_ARM_HARDWARE_VELOCITY_LIMITS,
                "max_excess_ratio": 2.0,
            },
        )
        for sensor_name, (source_body, target_bodies) in zip(
            _D1_SELF_COLLISION_SENSOR_NAMES,
            _D1_NONADJACENT_SELF_COLLISION_FILTERS,
            strict=True,
        ):
            setattr(
                self.scene,
                sensor_name,
                ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}/Robot/d1/{source_body}",
                    history_length=3,
                    filter_prim_paths_expr=[
                        f"{{ENV_REGEX_NS}}/Robot/d1/{target_body}" for target_body in target_bodies
                    ],
                ),
            )
        self.rewards.d1_nonadjacent_self_collision = RewTerm(
            func=mdp.filtered_contact_pair_violations,
            weight=-2.0,
            params={
                "sensor_names": _D1_SELF_COLLISION_SENSOR_NAMES,
                "force_threshold": 5.0,
            },
        )
        self.actions.joint_pos.joint_names = list(GO2_D1_ACTION_JOINT_NAMES)
        self.actions.joint_pos.preserve_order = True
        self.actions.joint_pos.scale = {
            r"^(FL|FR|RL|RR)_hip_joint$": 0.125,
            r"^(FL|FR|RL|RR)_(thigh|calf)_joint$": 0.25,
            r"^arm_[1-6]_joint$": 0.25,
            r"^arm_7_1_joint$": 0.005,
        }
        self.actions.joint_pos.clip = {
            r"^(FL|FR|RL|RR)_(hip|thigh|calf)_joint$": (-100.0, 100.0),
            r"^arm_[146]_joint$": (-2.35, 2.35),
            r"^arm_[235]_joint$": (-1.57, 1.57),
            r"^arm_7_1_joint$": (0.0, 0.033),
        }
        # This is the normal direct APEX-RL task, not a distillation student.
        # PPO sees the deployable policy group and learns all 19 targets while
        # original additive DecAP temporarily supplies the 18 leg/arm motion
        # errors. The gripper is learned directly throughout.
        self.actions.joint_pos.decap_joint_names = tuple(GO2_D1_MOTION_JOINT_NAMES)
        self.actions.joint_pos.decap_lambda_start = 1.0
        self.actions.joint_pos.decap_lambda_end = 0.0
        self.actions.joint_pos.decap_decay_type = "cosine"
        self.actions.joint_pos.decap_warmup_iterations = 100
        self.actions.joint_pos.decap_decay_start_iteration = 0
        self.actions.joint_pos.decap_decay_end_iteration = 1000
        self.actions.joint_pos.reference_residual_joint_names = ()
        # The D1 firmware accepts its seven-angle position packet at 10 Hz. Keep
        # the 50 Hz actor contract, but latch J1-J6 and the physical gripper only
        # every 100 ms. Do not add unverified quantization, latency, or filtering.
        self.actions.joint_pos.servo_joint_names = D1_COMMAND_JOINT_NAMES
        self.actions.joint_pos.servo_command_period_s = D1_ARM_COMMAND_PERIOD_S
        self.actions.joint_pos.servo_command_quantization = 0.0
        self.actions.joint_pos.servo_filter_enabled = False

        self.scene.robot.actuators["legs"].stiffness = 20.0
        self.scene.robot.actuators["legs"].damping = 0.5
        # The APEX asset uses the MuJoCo-tuned, explicit, torque-limited D1 PD
        # actuators, so Isaac and MuJoCo evaluate the same torque equation.
        # Hardware speed limits remain soft reward thresholds rather than
        # solver constraints. The nonbinding numerical velocity is repeated
        # here as an explicit task contract in case the shared asset changes.
        for actuator_name in ("arm_j1_j2", "arm_j3", "arm_j4", "arm_j5_j6"):
            actuator_cfg = self.scene.robot.actuators[actuator_name]
            actuator_cfg.velocity_limit = D1_ARM_NONBINDING_SIM_VELOCITY_LIMIT
            actuator_cfg.velocity_limit_sim = D1_ARM_NONBINDING_SIM_VELOCITY_LIMIT
        self.events.randomize_reset_joints.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=list(GO2_D1_ACTION_JOINT_NAMES)
        )
        self.events.randomize_d1_mount_x = EventTerm(
            func=mdp.randomize_fixed_joint_defaults,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=["base_link_joint"]),
                "position_ranges": {"base_link_joint": (-0.005, 0.005)},
                "velocity_range": (0.0, 0.0),
            },
        )
        # Randomize all Go2+D1 gains around the calibrated nominal values for
        # sim2real robustness. The flat tracker supplies a 0.9–1.1 scale range.
        self.events.randomize_actuator_gains.params["asset_cfg"].joint_names = list(GO2_D1_ACTION_JOINT_NAMES)
        # Keep every independently sampled PhysX material valid:
        # static friction must never be below dynamic friction.
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.8, 1.0)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.3, 0.8)
        self.observations.policy = Go2D1ArmApexTrackerPolicyObservationsCfg()
        self.observations.critic = Go2D1ArmApexTrackerCriticObservationsCfg()
        self.observations.privileged = Go2D1ArmApexPrivilegedTrackerObservationsCfg()
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = list(GO2_D1_ACTION_JOINT_NAMES)
        # D1 feedback exposes measured joint angles, but no deployable arm
        # velocity/effort signal. Keep leg velocities for locomotion and make
        # every arm-specific actor channel position based.
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = list(GO2_ACTION_JOINT_NAMES)
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = list(GO2_D1_ACTION_JOINT_NAMES)
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = list(GO2_D1_ACTION_JOINT_NAMES)
        self.observations.critic.reference_joint_pos.params["joint_names"] = list(GO2_D1_ACTION_JOINT_NAMES)
        self.observations.critic.joint_torques.params["asset_cfg"].joint_names = list(GO2_D1_ACTION_JOINT_NAMES)
        self.observations.privileged.joint_pos.params["asset_cfg"].joint_names = list(GO2_D1_ACTION_JOINT_NAMES)
        self.observations.privileged.joint_vel.params["asset_cfg"].joint_names = list(GO2_D1_ACTION_JOINT_NAMES)
        self.observations.privileged.reference_joint_pos.params["joint_names"] = list(GO2_D1_ACTION_JOINT_NAMES)
        self.observations.privileged.joint_torques.params["asset_cfg"].joint_names = list(GO2_D1_ACTION_JOINT_NAMES)


@configclass
class UnitreeGo2D1ArmApexDistillationStudentEnvCfg(UnitreeGo2D1ArmApexFlatTrackerEnvCfg):
    """Supervised Go2+D1 student with deployable observations and no DecAP prior."""

    train_with_zero_decap: bool = True

    def __post_init__(self):
        super().__post_init__()

        # Distillation supervises these direct 19-D actions from the privileged
        # teacher. Keep the student action contract explicit in this separate
        # task instead of changing the normal PPO task based on runner type.
        self.actions.joint_pos.decap_joint_names = ()
        self.actions.joint_pos.decap_lambda_start = 0.0
        self.actions.joint_pos.decap_lambda_end = 0.0
        self.actions.joint_pos.decap_decay_type = "constant"
        self.actions.joint_pos.decap_warmup_iterations = 0
        self.actions.joint_pos.decap_decay_start_iteration = 0
        self.actions.joint_pos.decap_decay_end_iteration = 1
        self.actions.joint_pos.reference_residual_joint_names = ()
        self.actions.joint_pos.servo_joint_names = D1_COMMAND_JOINT_NAMES


@configclass
class UnitreeGo2D1ArmApexOriginalDecapTeacherEnvCfg(UnitreeGo2D1ArmApexFlatTrackerEnvCfg):
    """Privileged Go2+D1 teacher with original additive DecAP at 50 Hz."""

    def __post_init__(self):
        super().__post_init__()

        # Teacher, direct PPO task, and student share the validated explicit,
        # torque-limited MuJoCo-parity plant. The teacher retains the original
        # additive DecAP schedule while exposing its privileged observation
        # group through the teacher runner.
        self.actions.joint_pos.decap_joint_names = tuple(GO2_D1_MOTION_JOINT_NAMES)
        self.actions.joint_pos.decap_lambda_start = 1.0
        self.actions.joint_pos.decap_lambda_end = 0.0
        self.actions.joint_pos.decap_decay_type = "cosine"
        self.actions.joint_pos.decap_warmup_iterations = 100
        self.actions.joint_pos.decap_decay_start_iteration = 0
        self.actions.joint_pos.decap_decay_end_iteration = 1000
        self.actions.joint_pos.reference_residual_joint_names = ()
        self.actions.joint_pos.servo_joint_names = D1_COMMAND_JOINT_NAMES


@configclass
class UnitreeGo2D1ArmApexPickStowCarryRobotOnlyFlatTrackerEnvCfg(UnitreeGo2D1ArmApexFlatTrackerEnvCfg):
    """Track the pick-stow-carry robot motion without spawning or observing an object."""

    def __post_init__(self):
        super().__post_init__()

        motion_root = f"{os.path.dirname(__file__)}/motion/npz/go2_d1"
        self.commands.motion.motion_files = (f"{motion_root}/pick_stow_carry_robot_only.npz",)
        self.commands.motion.start_at_motion_beginning_on_reset = True
        # Half the environments learn the complete approach from frame zero;
        # the other half sample all later arm, carry, and placement phases.
        self.commands.motion.reset_random_frame_probability = 0.50
        self.commands.motion.reset_attached_frame_probability = 0.0
        self.commands.motion.reference_state_init = True
        self.commands.motion.reference_object_state_init = False
        self.commands.motion.reference_object_asset_name = None
        self.commands.motion.terminate_episode_at_motion_end = True
        self.commands.motion.use_adaptive_sampling = False
        self.commands.motion.gripper_binary = False

        self.episode_length_s = 46.82
        self.scene.env_spacing = 12.0

        # Preserve the full recorded Go2+D1 trajectory, including the
        # continuous gripper coordinate, but keep the demonstrated arm reach
        # inside the runner's normalized action bound.
        self.actions.joint_pos.scale.pop(r"^arm_[1-6]_joint$", None)
        self.actions.joint_pos.scale.update(
            {
                r"^arm_[1456]_joint$": 0.25,
                r"^arm_2_joint$": 0.50,
                r"^arm_3_joint$": 0.35,
            }
        )
        self.actions.joint_pos.binary_joint_names = ()

        # Prevent the long stationary manipulation phase from rewarding a
        # policy that stops before reaching the demonstrated world position.
        self.rewards.imitate_world_base_pos.weight = 2.0
        self.rewards.imitate_world_base_pos_huber = RewTerm(
            func=mdp.motion_world_base_position_error_huber,
            weight=-2.0,
            params={
                "command_name": "motion",
                "delta": 0.15,
                "xy_scale": 1.0,
                "height_scale": 0.0,
                "max_cost": 1.5,
            },
        )
        self.rewards.track_command_lin_vel_xy.weight = 1.5

        self.terminations.motion_clip_end = DoneTerm(
            func=mdp.motion_clip_end,
            time_out=True,
            params={"command_name": "motion"},
        )
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = ["base"]
        self.terminations.bad_anchor_height = DoneTerm(
            func=mdp.bad_anchor_pos_z_only,
            params={"command_name": "motion", "threshold": 0.15},
        )
        self.terminations.bad_anchor_orientation = DoneTerm(
            func=mdp.bad_anchor_ori,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "command_name": "motion",
                "threshold": 0.5,
            },
        )
        self.rewards.bad_tracking_termination = RewTerm(
            func=mdp.is_terminated_term,
            weight=-200.0,
            params={
                "term_keys": ["illegal_contact", "bad_anchor_height", "bad_anchor_orientation"],
            },
        )


@configclass
class UnitreeGo2D1ArmApexPickStowCarryFlatTrackerEnvCfg(UnitreeGo2D1ArmApexFlatTrackerEnvCfg):
    """Contact-aware Go2+D1 tracker for the pick, stow, carry, and place sequence."""

    def __post_init__(self):
        super().__post_init__()

        motion_root = f"{os.path.dirname(__file__)}/motion/npz/go2_d1"
        self.commands.motion.motion_files = (f"{motion_root}/pick_stow_carry.npz",)
        self.commands.motion.gripper_joint_names = ("arm_7_1_joint",)
        # Convert the demonstrated close phase into a binary manipulation
        # command: 0=open and 1=fully close. The physical servo endpoints are
        # 0 and 33 mm; contact with the box stops the jaws before the latter.
        self.commands.motion.gripper_binary = True
        self.commands.motion.gripper_binary_threshold = 0.0
        self.commands.motion.gripper_open_position = 0.0
        self.commands.motion.gripper_closed_position = 0.033
        self.commands.motion.object_size_scale_attr = "_object_size_scale"
        self.commands.motion.object_nominal_size = _PICK_OBJECT_SIZE
        self.commands.motion.start_at_motion_beginning_on_reset = True
        # Reset curriculum: half of environments begin at frame zero, 20%
        # begin at a uniformly sampled frame, and 30% begin in a demonstrated
        # attached phase with robot and box initialized consistently.
        self.commands.motion.reset_random_frame_probability = 0.20
        self.commands.motion.reset_attached_frame_probability = 0.30
        self.commands.motion.reference_state_init = True
        self.commands.motion.reference_object_state_init = True
        self.commands.motion.reference_object_asset_name = "object"
        self.commands.motion.terminate_episode_at_motion_end = True
        self.commands.motion.use_adaptive_sampling = False

        # A failed policy can otherwise walk only part of the approach, stand
        # still through manipulation, and collect the stationary velocity
        # reward. Keep the exponential arrival bonus, but pair it with a
        # robust XY cost whose gradient remains useful when the base is far
        # from the reference path. Height failures are handled separately.
        self.rewards.imitate_world_base_pos.weight = 2.0
        self.rewards.imitate_world_base_pos_huber = RewTerm(
            func=mdp.motion_world_base_position_error_huber,
            weight=-2.0,
            params={
                "command_name": "motion",
                "delta": 0.15,
                "xy_scale": 1.0,
                "height_scale": 0.0,
                "max_cost": 1.5,
            },
        )
        self.rewards.track_command_lin_vel_xy.weight = 1.5

        # One episode covers the complete 2,341-frame sequence at 50 Hz.
        self.episode_length_s = 46.82
        self.scene.env_spacing = 12.0
        self.scene.replicate_physics = False
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.CuboidCfg(
                size=_PICK_OBJECT_SIZE,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=4,
                    disable_gravity=False,
                    enable_gyroscopic_forces=True,
                    linear_damping=0.04,
                    angular_damping=0.04,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=_PICK_OBJECT_NOMINAL_MASS),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.25,
                    dynamic_friction=0.9,
                    restitution=0.0,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.85, 0.06, 0.04),
                    metallic=0.25,
                    roughness=0.35,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(2.0321982, 0.0, 0.1275),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )
        self.scene.d1_left_gripper_object_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/d1/Link7_1",
            history_length=4,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )
        self.scene.d1_right_gripper_object_contact = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/d1/Link7_2",
            history_length=4,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )
        # The ground-grasp reference passes through the Go2 head's deliberately
        # conservative cylinder proxy (Link4/5 on pickup, Link4/5/6 on place).
        # It also brings Link6 through the placed box during retraction. Those
        # pairs make the supplied reference dynamically impossible while all
        # other robot self-collisions and the two finger-object contacts remain
        # active. Filter only the audited conflicting pairs.
        self.events.disable_pick_reference_collision_pairs = EventTerm(
            func=mdp.disable_collision_pairs,
            mode="prestartup",
            params={
                "prim_path_pairs": (
                    ("Robot/Head_upper", "Robot/d1/Link4"),
                    ("Robot/Head_upper", "Robot/d1/Link5"),
                    ("Robot/Head_upper", "Robot/d1/Link6"),
                    ("Robot/d1/Link6", "Object"),
                ),
            },
        )

        tracked_joint_names = list(GO2_D1_PICK_STOW_ACTION_JOINT_NAMES)
        self.actions.joint_pos.joint_names = tracked_joint_names
        # This clip reaches much farther than the hello motions. With the
        # shared 0.25-rad arm scale, J2 needed +11.71 and J3 needed -7.85
        # normalized action, beyond the runner's +/-6.6 bound. Use the
        # smallest task-specific scale increase that makes both reachable.
        self.actions.joint_pos.scale.pop(r"^arm_[1-6]_joint$", None)
        self.actions.joint_pos.scale.update(
            {
                r"^arm_[1456]_joint$": 0.25,
                r"^arm_2_joint$": 0.50,
                r"^arm_3_joint$": 0.35,
            }
        )
        # Keep the original additive DecAP schedule used by the successful
        # direct APEX tracker. The gripper servo remains a directly learned action.
        self.actions.joint_pos.decap_joint_names = tuple(GO2_D1_MOTION_JOINT_NAMES)
        self.actions.joint_pos.decap_lambda_start = 1.0
        self.actions.joint_pos.decap_lambda_end = 0.0
        self.actions.joint_pos.decap_decay_type = "cosine"
        self.actions.joint_pos.decap_warmup_iterations = 100
        self.actions.joint_pos.decap_decay_start_iteration = 0
        self.actions.joint_pos.decap_decay_end_iteration = 1000
        self.actions.joint_pos.reference_residual_joint_names = ()
        self.actions.joint_pos.binary_joint_names = ("arm_7_1_joint",)
        self.actions.joint_pos.binary_action_threshold = 0.0
        self.actions.joint_pos.binary_open_position = 0.0
        self.actions.joint_pos.binary_closed_position = 0.033
        self.actions.joint_pos.servo_joint_names = D1_COMMAND_JOINT_NAMES
        # The gripper earns only a small correct-state reward. Do not reward
        # jaw position, target magnitude, or logit magnitude: those objectives
        # were saturated in the failed run without producing a physical grasp.
        self.rewards.imitate_gripper_joint_pos = None
        self.rewards.imitate_gripper_policy_targets = None
        self.rewards.imitate_gripper_action = None
        self.rewards.imitate_gripper_logit = None
        self.rewards.imitate_gripper_state = RewTerm(
            func=mdp.motion_binary_action_state_match,
            weight=0.25,
            params={
                "command_name": "motion",
                "action_name": "joint_pos",
                "joint_names": ["arm_7_1_joint"],
                "action_threshold": 0.0,
                "physical_threshold": 0.0,
            },
        )
        self.rewards.imitate_object_pos = RewTerm(
            func=mdp.motion_object_position_error_exp,
            weight=4.0,
            params={
                "command_name": "motion",
                "sigma": 0.08,
                "object_cfg": SceneEntityCfg("object"),
                "detached_scale": 0.10,
                "attached_scale": 1.0,
            },
        )
        self.rewards.imitate_object_pos_huber = RewTerm(
            func=mdp.motion_object_position_error_huber,
            weight=-3.0,
            params={
                "command_name": "motion",
                "delta": 0.08,
                # A dropped box can become metres from its moving reference.
                # Keep the dense recovery gradient near the grasp without
                # allowing this one term to dominate an entire 47 s rollout.
                "max_cost": 0.25,
                "object_cfg": SceneEntityCfg("object"),
                "detached_scale": 0.10,
                "attached_scale": 1.0,
            },
        )
        self.rewards.imitate_object_up_axis = RewTerm(
            func=mdp.motion_object_up_axis_error_exp,
            weight=0.5,
            params={
                "command_name": "motion",
                "std": 0.25,
                "object_cfg": SceneEntityCfg("object"),
                "detached_scale": 0.10,
                "attached_scale": 1.0,
            },
        )
        self.rewards.imitate_object_linear_velocity = RewTerm(
            func=mdp.motion_object_linear_velocity_error_exp,
            weight=0.5,
            params={
                "command_name": "motion",
                "sigma": 0.35,
                "object_cfg": SceneEntityCfg("object"),
                "detached_scale": 0.10,
                "attached_scale": 1.0,
            },
        )
        self.rewards.imitate_attached_object_offset = RewTerm(
            func=mdp.motion_object_attachment_offset_error_exp,
            weight=8.0,
            params={
                "command_name": "motion",
                "sigma": 0.04,
                "ee_cfg": SceneEntityCfg("robot", body_names=["Link7_1", "Link7_2"]),
                "object_cfg": SceneEntityCfg("object"),
            },
        )
        object_contact_reward_params = {
            "command_name": "motion",
            "left_sensor_cfg": SceneEntityCfg(_PICK_LEFT_OBJECT_CONTACT_SENSOR, body_names="Link7_1"),
            "right_sensor_cfg": SceneEntityCfg(_PICK_RIGHT_OBJECT_CONTACT_SENSOR, body_names="Link7_2"),
            "force_threshold": 0.35,
        }
        self.rewards.attached_bilateral_gripper_contact = RewTerm(
            func=mdp.motion_object_attached_bilateral_contact,
            weight=12.0,
            params=copy.deepcopy(object_contact_reward_params),
        )
        self.rewards.attached_without_bilateral_gripper_contact = RewTerm(
            func=mdp.motion_object_attached_without_bilateral_contact,
            # Contact is sparse at initialization. The +12 contact reward,
            # +8 offset reward, and +4 position reward already make a real
            # grasp decisively better; a smaller miss cost avoids a critic
            # collapse before the first successful attached-reset samples.
            weight=-2.0,
            params=copy.deepcopy(object_contact_reward_params),
        )
        self.terminations.motion_clip_end = DoneTerm(
            func=mdp.motion_clip_end,
            time_out=True,
            params={"command_name": "motion"},
        )
        self.terminations.object_grasp_contact_timeout = DoneTerm(
            func=mdp.object_grasp_contact_timeout,
            params={
                **copy.deepcopy(object_contact_reward_params),
                # The contract is inactive during early exploration, then
                # tightens from a forgiving two-second miss window to 0.5 s.
                # Training injects rollout length and checkpoint iteration so
                # this schedule remains correct when a run is resumed.
                "warmup_iterations": _PICK_GRASP_TERMINATION_WARMUP_ITERATIONS,
                "ramp_end_iteration": _PICK_GRASP_TERMINATION_RAMP_END_ITERATION,
                "timeout_start_s": _PICK_GRASP_TIMEOUT_START_S,
                "timeout_end_s": _PICK_GRASP_TIMEOUT_END_S,
                "steps_per_iteration": 24,
                "resume_iteration": 0,
            },
        )
        # Hip contacts are common while the policy lowers the wrist toward the
        # box and are already penalized by ``undesired_contacts``. The inherited
        # base+hip termination reset roughly 90% of the grasp-curriculum run at
        # about 10.6 s, before the clip's 12.0 s pickup. Keep only a base
        # collision fatal so a true fall still ends the episode.
        self.terminations.illegal_contact.params["sensor_cfg"].body_names = ["base"]
        self.terminations.bad_anchor_height = DoneTerm(
            func=mdp.bad_anchor_pos_z_only,
            params={"command_name": "motion", "threshold": 0.15},
        )
        self.terminations.bad_anchor_orientation = DoneTerm(
            func=mdp.bad_anchor_ori,
            params={
                "asset_cfg": SceneEntityCfg("robot"),
                "command_name": "motion",
                "threshold": 0.5,
            },
        )
        self.rewards.bad_tracking_termination = RewTerm(
            func=mdp.is_terminated_term,
            weight=-200.0,
            params={
                "term_keys": [
                    "illegal_contact",
                    "bad_anchor_height",
                    "bad_anchor_orientation",
                    "object_grasp_contact_timeout",
                ],
            },
        )
        self.rewards.joint_acc_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=list(GO2_ACTION_JOINT_NAMES)
        )
        self.rewards.joint_torques_l2.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=list(GO2_ACTION_JOINT_NAMES)
        )
        self.events.randomize_reset_joints.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=tracked_joint_names
        )
        self.events.randomize_actuator_gains.params["asset_cfg"].joint_names = tracked_joint_names
        self.events.randomize_object_size = EventTerm(
            func=mdp.randomize_rigid_body_size_scale,
            mode="prestartup",
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "size_scale_range": _PICK_OBJECT_SIZE_SCALE_RANGE,
                "scale_attr_name": "_object_size_scale",
            },
        )
        self.events.reset_object = EventTerm(
            func=mdp.reset_root_state_uniform_with_size_scale,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "nominal_size": _PICK_OBJECT_SIZE,
                "scale_attr_name": "_object_size_scale",
                "pose_range": {
                    "x": (-0.0025, 0.0025),
                    "y": (-0.0025, 0.0025),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (-0.05, 0.05),
                },
                "velocity_range": {
                    "x": (0.0, 0.0),
                    "y": (0.0, 0.0),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
            },
        )
        self.events.randomize_object_material = EventTerm(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("object", body_names=".*"),
                # Keep every sampled static coefficient above every sampled
                # dynamic coefficient to satisfy the PhysX material contract.
                "static_friction_range": (1.10, 1.40),
                "dynamic_friction_range": (0.80, 1.00),
                "restitution_range": (0.0, 0.05),
                "num_buckets": 64,
            },
        )
        self.events.randomize_object_mass = EventTerm(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("object", body_names=".*"),
                "mass_distribution_params": _PICK_OBJECT_MASS_RANGE,
                "operation": "abs",
                "recompute_inertia": True,
            },
        )

        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = tracked_joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = list(GO2_ACTION_JOINT_NAMES)
        self.observations.policy.reference_motion.params = tracker_reference_params(GO2_D1_PICK_STOW_ACTION_JOINT_NAMES)
        self.observations.policy.object_position = ObsTerm(
            func=mdp.object_position_b,
            params={"object_cfg": SceneEntityCfg("object")},
            clip=(-10.0, 10.0),
            scale=1.0,
        )
        self.observations.policy.object_linear_velocity = ObsTerm(
            func=mdp.object_linear_velocity_b,
            params={"object_cfg": SceneEntityCfg("object")},
            clip=(-10.0, 10.0),
            scale=0.5,
        )
        self.observations.policy.object_up_axis = ObsTerm(
            func=mdp.object_up_axis_b,
            params={"object_cfg": SceneEntityCfg("object")},
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        self.observations.policy.reference_object_position = ObsTerm(
            func=mdp.reference_object_position_trajectory_b,
            params={
                "command_name": "motion",
                "time_offsets": _PICK_OBJECT_REFERENCE_TIME_OFFSETS,
            },
            clip=(-10.0, 10.0),
            scale=1.0,
        )
        self.observations.policy.reference_object_up_axis = ObsTerm(
            func=mdp.reference_object_up_axis_b,
            params={"command_name": "motion"},
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        self.observations.policy.reference_object_attachment = ObsTerm(
            func=mdp.reference_object_attachment_phase,
            params={
                "command_name": "motion",
                "time_offsets": _PICK_OBJECT_REFERENCE_TIME_OFFSETS,
            },
            clip=(0.0, 1.0),
            scale=1.0,
        )

        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = tracked_joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = tracked_joint_names
        self.observations.critic.reference_joint_pos.params["joint_names"] = tracked_joint_names
        self.observations.critic.joint_torques.params["asset_cfg"].joint_names = tracked_joint_names
        self.observations.critic.reference_motion.params = privileged_tracker_reference_params(
            GO2_D1_PICK_STOW_ACTION_JOINT_NAMES
        )
        self.observations.critic.object_position = copy.deepcopy(self.observations.policy.object_position)
        self.observations.critic.reference_object_position = copy.deepcopy(
            self.observations.policy.reference_object_position
        )
        self.observations.critic.object_linear_velocity = copy.deepcopy(
            self.observations.policy.object_linear_velocity
        )
        self.observations.critic.object_up_axis = copy.deepcopy(self.observations.policy.object_up_axis)
        self.observations.critic.reference_object_up_axis = copy.deepcopy(
            self.observations.policy.reference_object_up_axis
        )
        self.observations.critic.reference_object_attachment = copy.deepcopy(
            self.observations.policy.reference_object_attachment
        )
        self.observations.critic.left_gripper_object_contact = ObsTerm(
            func=mdp.filtered_contact_state,
            params={
                "sensor_cfg": SceneEntityCfg(_PICK_LEFT_OBJECT_CONTACT_SENSOR, body_names="Link7_1"),
                "force_threshold": 0.35,
            },
            clip=(0.0, 1.0),
            scale=1.0,
        )
        self.observations.critic.right_gripper_object_contact = ObsTerm(
            func=mdp.filtered_contact_state,
            params={
                "sensor_cfg": SceneEntityCfg(_PICK_RIGHT_OBJECT_CONTACT_SENSOR, body_names="Link7_2"),
                "force_threshold": 0.35,
            },
            clip=(0.0, 1.0),
            scale=1.0,
        )

        self.observations.privileged.joint_pos.params["asset_cfg"].joint_names = tracked_joint_names
        self.observations.privileged.joint_vel.params["asset_cfg"].joint_names = tracked_joint_names
        self.observations.privileged.reference_joint_pos.params["joint_names"] = tracked_joint_names
        self.observations.privileged.joint_torques.params["asset_cfg"].joint_names = tracked_joint_names
        self.observations.privileged.reference_motion.params = privileged_tracker_reference_params(
            GO2_D1_PICK_STOW_ACTION_JOINT_NAMES
        )
        self.observations.privileged.object_position = copy.deepcopy(self.observations.policy.object_position)
        self.observations.privileged.reference_object_position = copy.deepcopy(
            self.observations.policy.reference_object_position
        )
        self.observations.privileged.object_linear_velocity = copy.deepcopy(
            self.observations.policy.object_linear_velocity
        )
        self.observations.privileged.object_up_axis = copy.deepcopy(self.observations.policy.object_up_axis)
        self.observations.privileged.reference_object_up_axis = copy.deepcopy(
            self.observations.policy.reference_object_up_axis
        )
        self.observations.privileged.reference_object_attachment = copy.deepcopy(
            self.observations.policy.reference_object_attachment
        )
        self.observations.privileged.left_gripper_object_contact = copy.deepcopy(
            self.observations.critic.left_gripper_object_contact
        )
        self.observations.privileged.right_gripper_object_contact = copy.deepcopy(
            self.observations.critic.right_gripper_object_contact
        )


@configclass
class UnitreeGo2D1ArmApexCanPickCarryDropFlatTrackerEnvCfg(
    UnitreeGo2D1ArmApexPickStowCarryFlatTrackerEnvCfg
):
    """Track a top-down floor-can pickup, walking carry, and gravity-driven drop."""

    def __post_init__(self):
        super().__post_init__()

        motion_root = f"{os.path.dirname(__file__)}/motion/npz/go2_d1"
        self.commands.motion.motion_files = (f"{motion_root}/top_down_can_pick_and_move.npz",)
        self.commands.motion.object_nominal_size = _CAN_OBJECT_SIZE
        # The can is grasped from above, so size randomization should preserve
        # its ground plane without shifting its center toward a lateral face.
        self.commands.motion.object_size_preserve_grasp_face = False

        # The 1,141-frame clip includes the final 0.2 s of free fall after the
        # gripper opens. Match the command's inclusive clip duration at 50 Hz.
        self.episode_length_s = 22.82
        self.scene.env_spacing = 4.0
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            spawn=sim_utils.CylinderCfg(
                radius=0.5 * _CAN_OBJECT_SIZE[0],
                height=_CAN_OBJECT_SIZE[2],
                axis="Z",
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=4,
                    disable_gravity=False,
                    enable_gyroscopic_forces=True,
                    linear_damping=0.02,
                    angular_damping=0.02,
                ),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                mass_props=sim_utils.MassPropertiesCfg(mass=_CAN_OBJECT_NOMINAL_MASS),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.2,
                    dynamic_friction=0.9,
                    restitution=0.02,
                ),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.78, 0.04, 0.03),
                    metallic=0.55,
                    roughness=0.28,
                ),
            ),
            init_state=RigidObjectCfg.InitialStateCfg(
                pos=(0.48, 0.0, 0.0675),
                rot=(1.0, 0.0, 0.0, 0.0),
            ),
        )

        # J3 reaches farther in this top-down clip than in pick-stow-carry.
        # A 0.375-rad scale keeps the demonstrated -6.40 normalized target
        # inside the runner's +/-6.6 action bound.
        self.actions.joint_pos.scale[r"^arm_3_joint$"] = 0.375

        self.events.randomize_object_size.params["size_scale_range"] = _CAN_OBJECT_SIZE_SCALE_RANGE
        self.events.reset_object.params.update(
            {
                "nominal_size": _CAN_OBJECT_SIZE,
                "preserve_grasp_face": False,
                "pose_range": {
                    "x": (-0.01, 0.01),
                    "y": (-0.01, 0.01),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (-0.25, 0.25),
                },
            }
        )
        self.events.randomize_object_material.params.update(
            {
                "static_friction_range": (1.05, 1.35),
                "dynamic_friction_range": (0.75, 1.00),
                "restitution_range": (0.0, 0.08),
            }
        )
        self.events.randomize_object_mass.params["mass_distribution_params"] = _CAN_OBJECT_MASS_RANGE


def _configure_pick_privileged_manipulation_observations(env_cfg) -> None:
    """Install teacher-only manipulation state and extended temporal reference features."""
    privileged = env_cfg.observations.privileged
    privileged.reference_motion.params = privileged_tracker_reference_params(
        GO2_D1_PICK_STOW_ACTION_JOINT_NAMES,
        time_offsets=_PICK_PRIVILEGED_REFERENCE_TIME_OFFSETS,
    )
    privileged.reference_feet_yaw_b.params["time_offsets"] = _PICK_PRIVILEGED_REFERENCE_TIME_OFFSETS
    privileged.reference_body_yaw_b.params["time_offsets"] = _PICK_PRIVILEGED_REFERENCE_TIME_OFFSETS
    privileged.reference_object_position.params["time_offsets"] = _PICK_PRIVILEGED_REFERENCE_TIME_OFFSETS
    privileged.reference_object_attachment.params["time_offsets"] = _PICK_PRIVILEGED_REFERENCE_TIME_OFFSETS
    privileged.object_orientation = ObsTerm(
            func=mdp.object_orientation_b,
            params={"object_cfg": SceneEntityCfg("object")},
            clip=(-1.0, 1.0),
            scale=1.0,
        )
    privileged.object_angular_velocity = ObsTerm(
            func=mdp.object_angular_velocity_b,
            params={"object_cfg": SceneEntityCfg("object")},
            clip=(-20.0, 20.0),
            scale=0.25,
        )
    privileged.arm_end_effector_pose = ObsTerm(
            func=mdp.body_pose_b,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["Link6", "Link7_1", "Link7_2"],
                    preserve_order=True,
                )
            },
            clip=(-10.0, 10.0),
            scale=1.0,
        )
    privileged.gripper_object_geometry = ObsTerm(
            func=mdp.gripper_object_geometry_b,
            params={
                "gripper_cfg": SceneEntityCfg(
                    "robot",
                    body_names=["Link7_1", "Link7_2"],
                    preserve_order=True,
                ),
                "object_cfg": SceneEntityCfg("object"),
            },
            clip=(-10.0, 10.0),
            scale=1.0,
        )
    privileged.left_gripper_object_force = ObsTerm(
            func=mdp.filtered_contact_force_b,
            params={
                "sensor_cfg": SceneEntityCfg(_PICK_LEFT_OBJECT_CONTACT_SENSOR),
                "asset_cfg": SceneEntityCfg("robot"),
                "normalize": 20.0,
            },
            clip=(-10.0, 10.0),
            scale=1.0,
        )
    privileged.right_gripper_object_force = ObsTerm(
            func=mdp.filtered_contact_force_b,
            params={
                "sensor_cfg": SceneEntityCfg(_PICK_RIGHT_OBJECT_CONTACT_SENSOR),
                "asset_cfg": SceneEntityCfg("robot"),
                "normalize": 20.0,
            },
            clip=(-10.0, 10.0),
            scale=1.0,
        )
    privileged.object_mass = ObsTerm(
            func=mdp.rigid_body_mass,
            params={"asset_cfg": SceneEntityCfg("object"), "normalize": _PICK_OBJECT_NOMINAL_MASS},
            clip=(0.0, 10.0),
            scale=1.0,
        )
    privileged.object_size_scale = ObsTerm(
            func=mdp.environment_scalar_attribute,
            params={"attribute_name": "_object_size_scale", "default": 1.0},
            clip=(0.0, 10.0),
            scale=1.0,
        )
    privileged.reference_object_orientation = ObsTerm(
            func=mdp.reference_object_orientation_trajectory_b,
            params={
                "command_name": "motion",
                "time_offsets": _PICK_PRIVILEGED_REFERENCE_TIME_OFFSETS,
            },
            clip=(-1.0, 1.0),
            scale=1.0,
        )
    privileged.reference_arm_ee_pose = ObsTerm(
            func=mdp.reference_arm_ee_pose_trajectory_b,
            params={
                "command_name": "motion",
                "time_offsets": _PICK_PRIVILEGED_REFERENCE_TIME_OFFSETS,
            },
            clip=(-10.0, 10.0),
            scale=1.0,
        )
    # Five 50 Hz snapshots expose an 80 ms actual-state history span. Each
    # snapshot also carries the explicit reference window from -0.4 s to
    # +0.8 s, so the teacher can anticipate grasp and release transitions.
    privileged.history_length = 5
    privileged.flatten_history_dim = True
    privileged.enable_corruption = False


@configclass
class UnitreeGo2D1ArmApexPickStowCarryPrivilegedTeacherEnvCfg(
    UnitreeGo2D1ArmApexPickStowCarryFlatTrackerEnvCfg
):
    """Dedicated full-state manipulation teacher with the pick task's original DecAP schedule."""

    def __post_init__(self):
        super().__post_init__()
        _configure_pick_privileged_manipulation_observations(self)


@configclass
class UnitreeGo2D1ArmApexPickStowCarryDistillationStudentEnvCfg(
    UnitreeGo2D1ArmApexPickStowCarryFlatTrackerEnvCfg
):
    """Deployable manipulation student surface paired with the separate privileged teacher."""

    train_with_zero_decap: bool = True

    def __post_init__(self):
        super().__post_init__()
        _configure_pick_privileged_manipulation_observations(self)
        self.actions.joint_pos.decap_joint_names = ()
        self.actions.joint_pos.decap_lambda_start = 0.0
        self.actions.joint_pos.decap_lambda_end = 0.0
        self.actions.joint_pos.decap_decay_type = "constant"
        self.actions.joint_pos.decap_warmup_iterations = 0
        self.actions.joint_pos.decap_decay_start_iteration = 0
        self.actions.joint_pos.decap_decay_end_iteration = 1
