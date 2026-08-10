from __future__ import annotations

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg

##
# Pre-defined configs
##
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import Gurukul.tasks.manager_based.go2_apex.mdp as mdp
import Gurukul.tasks.manager_based.locomotion.velocity.mdp as velocity_mdp
from Gurukul.tasks.manager_based.go2_apex.constants import (
    FOOT_BODY_NAMES,
    GO2_ACTION_JOINT_NAMES,
    ILLEGAL_CONTACT_BODY_NAMES,
    VELOCITY_RANGE,
)

##
# Scene definition
##

@configclass
class MySceneCfg(InteractiveSceneCfg):
    """Configuration for the terrain scene with a legged robot."""

    # ground terrain
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        visual_material=sim_utils.MdlFileCfg(
            mdl_path="{NVIDIA_NUCLEUS_DIR}/Materials/Base/Architecture/Shingles_01.mdl",
            project_uvw=True,
        ),
    )
    # robots
    robot: ArticulationCfg = MISSING
    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DistantLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(color=(0.13, 0.13, 0.13), intensity=1000.0),
    )
    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0, debug_vis=True
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command specifications for the MDP."""

    gripper = None
    motion = mdp.MotionCommandCfg(
        asset_name="robot",
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=True,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
        command_lin_vel_x_offset_range=(-1.0, 1.0),
        command_lin_vel_y_offset_range=(-0.1, 0.1),
        command_lin_vel_x_clip=(0.0, 3.0),
        command_lin_vel_y_clip=(-0.1, 0.1),
        command_ang_vel_z_offset_range=(-1.5, 1.5),
        command_ang_vel_z_clip=(-1.5, 1.5),
        use_adaptive_sampling=True,
        reference_state_init=True,
        require_command_channels=True,
        debug_vis_show_pose_frames=False,
        debug_vis_show_velocity=True,
        debug_vis_simplified_foot_reference=True,
        debug_vis_foot_body_names=tuple(FOOT_BODY_NAMES),
        sample_motions_uniformly=True,
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    joint_pos = mdp.DecapJointPositionActionCfg(
        asset_name="robot",
        joint_names=[".*"],
        use_default_offset=True,
        decap_command_name="motion",
        decap_lambda_start=1.0,
        decap_lambda_end=0.0,
        decap_decay_type="cosine",

        # Linear/cosine schedule. These are PPO iteration counts, converted to env steps in the action term.
        decap_steps_per_iteration=24,
        decap_warmup_iterations=100,
        decap_decay_start_iteration=0,
        decap_decay_end_iteration=1000,

        # Exponential schedule. Used only when decap_decay_type="exp".
        decap_exp_gamma=0.99,
        decap_exp_k=100.0,
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        # Actor intentionally excludes imitation/reference observations.
        base_ang_vel = ObsTerm(
            func=mdp.base_ang_vel,
            noise=Unoise(n_min=-0.2, n_max=0.2),
            clip=(-100.0, 100.0),
            scale=0.25,
        )
        projected_gravity = ObsTerm(
            func=mdp.projected_gravity,
            noise=Unoise(n_min=-0.05, n_max=0.05),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        command_vel_x = ObsTerm(
            func=mdp.motion_command_vel_x,
            params={"command_name": "motion"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        command_vel_y = ObsTerm(
            func=mdp.motion_command_vel_y,
            params={"command_name": "motion"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        command_vel_z = ObsTerm(
            func=mdp.motion_command_vel_z,
            params={"command_name": "motion"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        command_yaw = ObsTerm(
            func=mdp.motion_command_yaw,
            params={"command_name": "motion"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2_ACTION_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-0.01, n_max=0.01),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2_ACTION_JOINT_NAMES, preserve_order=True)},
            noise=Unoise(n_min=-1.5, n_max=1.5),
            clip=(-100.0, 100.0),
            scale=0.05,
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0), scale=1.0)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0), scale=1.0)
        projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0), scale=1.0)
        command = ObsTerm(
            func=mdp.motion_command_vel,
            params={"command_name": "motion"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2_ACTION_JOINT_NAMES, preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=GO2_ACTION_JOINT_NAMES, preserve_order=True)},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)
        phase = ObsTerm(func=mdp.motion_phase, params={"command_name": "motion"}, clip=(-100.0, 100.0), scale=1.0)
        reference_joint_pos = ObsTerm(
            func=mdp.reference_joint_pos,
            params={"command_name": "motion", "joint_names": GO2_ACTION_JOINT_NAMES},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        reference_foot_pos_b = ObsTerm(
            func=mdp.reference_foot_pos_b,
            params={"command_name": "motion"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        reference_base_quat = ObsTerm(
            func=mdp.reference_base_quat,
            params={"command_name": "motion"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    # startup
    randomize_rigid_body_material = EventTerm(
        func=velocity_mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.0),
            "dynamic_friction_range": (0.3, 0.8),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    randomize_rigid_body_mass_base = EventTerm(
        func=velocity_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "mass_distribution_params": (-1.0, 2.0),
            "operation": "add",
            "recompute_inertia": True,
        },
    )

    randomize_rigid_body_mass_others = EventTerm(
        func=velocity_mdp.randomize_rigid_body_mass,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[r"^(?!.*base).*"]),
            "mass_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )

    randomize_com_positions = EventTerm(
        func=velocity_mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "com_range": {"x": (-0.05, 0.05), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    # reset
    randomize_apply_external_force_torque = EventTerm(
        func=velocity_mdp.apply_external_force_torque,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=["base"]),
            "force_range": (-10.0, 10.0),
            "torque_range": (-10.0, 10.0),
        },
    )

    randomize_reset_joints = EventTerm(
        func=velocity_mdp.reset_joints_by_offset,
        mode="reset",
        params={
            "position_range": (-0.1, 0.1),
            "velocity_range": (-0.1, 0.1),
        },
    )

    randomize_actuator_gains = EventTerm(
        func=velocity_mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    randomize_reset_base = EventTerm(
        func=velocity_mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.0, 0.2),
                "roll": (-0.25, 0.25),
                "pitch": (-0.25, 0.25),
                "yaw": (-1.57, 1.57),
            },
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.5, 0.5),
                "roll": (-0.5, 0.5),
                "pitch": (-0.5, 0.5),
                "yaw": (-0.5, 0.5),
            },
        },
    )

    # interval
    randomize_push_robot = EventTerm(
        func=velocity_mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(10.0, 15.0),
        params={"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}},
    )


@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    # Regularization / safety
    joint_acc_l2 = RewTerm(func=mdp.joint_acc_l2, weight=-2.5e-7)
    joint_torques_l2 = RewTerm(func=mdp.joint_torques_l2, weight=-1e-5)
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-1e-2)
    action_smoothness_l2 = RewTerm(func=velocity_mdp.action_smoothness_l2, weight=-1e-2)
    ang_vel_xy_l2 = RewTerm(func=mdp.motion_ang_vel_xy_l2, weight=-0.05)

    # Active Isaac Gym flat imitation / command-tracking terms
    imitate_joint_pos = RewTerm(
        func=mdp.motion_joint_position_error_exp,
        weight=1.0,
        params={
            "command_name": "motion",
            "std": 0.05,
            "stand_still_vel_threshold": 1.0e-6,
            "stand_still_ang_vel_threshold": 1.0e-6,
            "stand_still_lin_vel_z_threshold": 1.0e-6,
        },
    )
    imitate_base_orientation = RewTerm(
        func=mdp.motion_base_orientation_error_exp,
        weight=0.0,
        params={"command_name": "motion", "std": 0.1},
    )
    imitate_projected_gravity = RewTerm(
        func=mdp.motion_projected_gravity_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.1},
    )
    imitate_foot_pos = RewTerm(
        func=mdp.motion_foot_position_error_exp,
        weight=1.0,
        params={
            "command_name": "motion",
            "sigma": 0.01,
            "stand_still_vel_threshold": 1.0e-6,
            "stand_still_ang_vel_threshold": 1.0e-6,
            "stand_still_lin_vel_z_threshold": 1.0e-6,
        },
    )
    imitate_world_foot_pos = RewTerm(
        func=mdp.motion_world_foot_position_error_exp,
        weight=0.75,
        params={
            "command_name": "motion",
            "sigma": 0.04,
            "xy_scale": 0.5,
            "height_scale": 1.5,
        },
    )
    imitate_world_base_pos = RewTerm(
        func=mdp.motion_world_base_position_error_exp,
        weight=0.5,
        params={
            "command_name": "motion",
            "sigma": 0.09,
            "xy_scale": 0.5,
            "height_scale": 1.5,
        },
    )
    track_command_lin_vel_xy = RewTerm(
        func=mdp.motion_command_tracking_lin_vel_exp,
        weight=1.5,
        params={"command_name": "motion", "sigma": 0.25},
    )
    track_command_lin_vel_z = RewTerm(
        func=mdp.motion_command_tracking_lin_vel_z_exp,
        weight=0.5,
        params={"command_name": "motion", "sigma": 0.25},
    )
    track_command_ang_vel_z = RewTerm(
        func=mdp.motion_command_tracking_ang_vel_exp,
        weight=1.0,
        params={"command_name": "motion", "sigma": 0.25},
    )
    imitate_base_height = RewTerm(
        func=mdp.motion_base_height_error_l2,
        weight=-10.0,
        params={"command_name": "motion"},
    )
    feet_slip = RewTerm(
        func=mdp.motion_feet_slip_penalty,
        weight=-0.08,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
        },
    )
    impact_reduction = RewTerm(
        func=mdp.motion_impact_penalty,
        weight=-5e-3,
        params={"command_name": "motion"},
    )
    airborne_contact = RewTerm(
        func=mdp.motion_airborne_contact_penalty,
        weight=-0.75,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
            "height_threshold": 0.05,
            "vertical_velocity_threshold": 0.45,
            "contact_threshold": 1.0,
        },
    )
    reference_foot_contact = RewTerm(
        func=mdp.motion_reference_foot_contact_penalty,
        weight=-0.0,
        params={
            "command_name": "motion",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=FOOT_BODY_NAMES),
            "contact_threshold": 1.0,
        },
    )

    # Others
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=[r"^(?!.*_foot$).+$"],
            ),
            "threshold": 1.0,
        },
    )


@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    illegal_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=ILLEGAL_CONTACT_BODY_NAMES),
            "threshold": 1.0,
        },
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    pass


##
# Environment configuration
##


@configclass
class Go2ApexEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the Go2 APEX motion-imitation environment."""

    train_with_zero_decap: bool = False

    # Scene settings
    scene: MySceneCfg = MySceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def force_zero_decap(self) -> None:
        """Disable env-side DecAP prior injection for training ablations."""
        action_cfg = self.actions.joint_pos
        if not hasattr(action_cfg, "decap_lambda_start"):
            return
        action_cfg.decap_prior_only = False
        action_cfg.decap_lambda_start = 0.0
        action_cfg.decap_lambda_end = 0.0
        action_cfg.decap_decay_type = "constant"
        if hasattr(action_cfg, "decap_resume_iteration"):
            action_cfg.decap_resume_iteration = 0
        action_cfg.decap_warmup_steps = 0
        action_cfg.decap_decay_start_step = 0
        action_cfg.decap_decay_end_step = 1

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 4
        self.episode_length_s = 20.0
        # simulation settings
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 16 * 2**15
        # viewer settings
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"
        if self.train_with_zero_decap:
            self.force_zero_decap()
