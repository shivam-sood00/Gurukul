from isaaclab.utils import configclass
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg

import Gurukul.tasks.manager_based.locomotion.velocity.mdp as mdp
from Gurukul.tasks.manager_based.locomotion.velocity.velocity_env_cfg import (
    LocomotionVelocityRoughEnvCfg,
    ObservationsCfg,
)

##
# Pre-defined configs
##
# # use cloud assets
# from isaaclab_assets.robots.unitree import UNITREE_GO2_CFG  # isort: skip
# use local assets
from Gurukul.assets.unitree import UNITREE_GO2_CFG  # isort: skip


@configclass
class Go2RoughFullTeacherObservationsCfg(ObsGroup):
    """Repository privileged-state teacher subset with temporal history.

    The compatibility class name predates CTS paper-fidelity review. This group
    does not include every paper teacher signal (for example joint torque,
    joint acceleration, or full contact-force values).
    """

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0), scale=1.0)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0), scale=1.0)
    projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0), scale=1.0)
    velocity_commands = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "base_velocity"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)
    height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        clip=(-1.0, 1.0),
        scale=1.0,
    )
    root_lin_vel_w = ObsTerm(func=mdp.root_lin_vel_w, clip=(-100.0, 100.0), scale=1.0)
    root_ang_vel_w = ObsTerm(func=mdp.root_ang_vel_w, clip=(-100.0, 100.0), scale=1.0)
    root_quat_w = ObsTerm(
        func=mdp.root_quat_w,
        params={"make_quat_unique": True},
        clip=(-1.0, 1.0),
        scale=1.0,
    )
    feet_contact_state = ObsTerm(
        func=mdp.feet_contact_state,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "force_threshold": 1.0,
        },
        clip=(0.0, 1.0),
        scale=1.0,
    )
    feet_heightmap = ObsTerm(
        func=mdp.feet_heightmap_scan,
        params={
            "height_sensor_cfg": SceneEntityCfg("height_scanner"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "patch_radius": 0.1,
            "offset": 0.5,
        },
        clip=(-1.0, 1.0),
        scale=1.0,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True
        self.history_length = 5
        self.flatten_history_dim = True


@configclass
class Go2RoughCTSRoleObservationsCfg(ObsGroup):
    """Per-environment rollout role used by CTS; 1=teacher, 0=student."""

    is_teacher = ObsTerm(func=mdp.cts_teacher_role, params={"teacher_fraction": 0.75})

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class Go2RoughCTSStudentHistoryObservationsCfg(ObservationsCfg.PolicyCfg):
    """Five-observation deployable history from CTS §III-B."""

    def __post_init__(self):
        super().__post_init__()
        self.history_length = 5
        self.flatten_history_dim = True


@configclass
class Go2RoughRealTeacherProprioObservationsCfg(ObsGroup):
    """REAL-style teacher proprioceptive query observations."""

    base_lin_vel = ObsTerm(func=mdp.base_lin_vel, clip=(-100.0, 100.0), scale=1.0)
    base_ang_vel = ObsTerm(func=mdp.base_ang_vel, clip=(-100.0, 100.0), scale=1.0)
    projected_gravity = ObsTerm(func=mdp.projected_gravity, clip=(-100.0, 100.0), scale=1.0)
    velocity_commands = ObsTerm(
        func=mdp.generated_commands,
        params={"command_name": "base_velocity"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    joint_pos = ObsTerm(
        func=mdp.joint_pos_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    joint_vel = ObsTerm(
        func=mdp.joint_vel_rel,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*", preserve_order=True)},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    actions = ObsTerm(func=mdp.last_action, clip=(-100.0, 100.0), scale=1.0)

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class Go2RoughRealTeacherTerrainObservationsCfg(ObsGroup):
    """REAL-style teacher terrain scan observations."""

    height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        clip=(-1.0, 1.0),
        scale=1.0,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class Go2RoughRealTeacherPrivilegedObservationsCfg(ObsGroup):
    """REAL-style teacher-only privileged support observations."""

    root_lin_vel_w = ObsTerm(func=mdp.root_lin_vel_w, clip=(-100.0, 100.0), scale=1.0)
    root_ang_vel_w = ObsTerm(func=mdp.root_ang_vel_w, clip=(-100.0, 100.0), scale=1.0)
    root_quat_w = ObsTerm(
        func=mdp.root_quat_w,
        params={"make_quat_unique": True},
        clip=(-1.0, 1.0),
        scale=1.0,
    )
    feet_contact_state = ObsTerm(
        func=mdp.feet_contact_state,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "force_threshold": 1.0,
        },
        clip=(0.0, 1.0),
        scale=1.0,
    )
    feet_heightmap = ObsTerm(
        func=mdp.feet_heightmap_scan,
        params={
            "height_sensor_cfg": SceneEntityCfg("height_scanner"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "patch_radius": 0.1,
            "offset": 0.5,
        },
        clip=(-1.0, 1.0),
        scale=1.0,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class UnitreeGo2RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    base_link_name = "base"
    foot_link_name = ".*_foot"
    # fmt: off
    joint_names = [
        "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
        "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
        "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
        "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    ]
    # fmt: on

    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # ------------------------------Sence------------------------------
        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        # ------------------------------Observations------------------------------
        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.teacher_full = Go2RoughFullTeacherObservationsCfg()
        self.observations.teacher_full.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.teacher_full.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.teacher_full.feet_contact_state.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.observations.teacher_full.feet_heightmap.params["asset_cfg"].body_names = [self.foot_link_name]
        self.observations.real_teacher_proprio = Go2RoughRealTeacherProprioObservationsCfg()
        self.observations.real_teacher_proprio.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.real_teacher_proprio.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.real_teacher_terrain = Go2RoughRealTeacherTerrainObservationsCfg()
        self.observations.real_teacher_privileged = Go2RoughRealTeacherPrivilegedObservationsCfg()
        self.observations.real_teacher_privileged.feet_contact_state.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.observations.real_teacher_privileged.feet_heightmap.params["asset_cfg"].body_names = [self.foot_link_name]

        # ------------------------------Actions------------------------------
        # reduce action scale
        self.actions.joint_pos.scale = {".*_hip_joint": 0.125, "^(?!.*_hip_joint).*": 0.25}
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names

        # ------------------------------Events------------------------------
        self.events.randomize_reset_base.params = {
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
        }
        self.events.randomize_reset_joints.func = mdp.reset_joints_by_offset
        self.events.randomize_reset_joints.params = {"position_range": (-0.1, 0.1), "velocity_range": (-0.1, 0.1)}
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        self.events.randomize_rigid_body_mass_others.params["mass_distribution_params"] = (0.8, 1.2)
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.8, 1.2)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.8, 1.2)

        # ------------------------------Rewards------------------------------
        # General
        self.rewards.is_terminated.weight = 0

        # Root penalties
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = 0
        self.rewards.base_height_l2.weight = 0
        self.rewards.base_height_l2.params["target_height"] = 0.33
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        # Joint penalties
        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_acc_l2.weight = -2.5e-7
        # self.rewards.create_joint_deviation_l1_rewterm("joint_deviation_hip_l1", -0.2, [".*_hip_joint"])
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.joint_power.weight = -2e-5
        self.rewards.stand_still.weight = -2.0
        self.rewards.joint_pos_penalty.weight = -1.0
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]

        # Action penalties
        self.rewards.action_rate_l2.weight = -0.01

        # Contact sensor
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]

        # Velocity-tracking rewards
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5

        # Others
        self.rewards.feet_air_time.weight = 0.1
        self.rewards.feet_air_time.params["threshold"] = 0.5
        self.rewards.feet_air_time.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_air_time_variance.weight = -1.0
        self.rewards.feet_air_time_variance.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact.weight = 0
        self.rewards.feet_contact.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_contact_without_cmd.weight = 0.1
        self.rewards.feet_contact_without_cmd.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_stumble.weight = 0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.weight = -0.1
        self.rewards.feet_slide.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_slide.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height.weight = 0
        self.rewards.feet_height.params["target_height"] = 0.05
        self.rewards.feet_height.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_height_body.weight = -5.0
        self.rewards.feet_height_body.params["target_height"] = -0.2
        self.rewards.feet_height_body.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_gait.weight = 0.5
        self.rewards.feet_gait.params["synced_feet_pair_names"] = (("FL_foot", "RR_foot"), ("FR_foot", "RL_foot"))
        self.rewards.upward.weight = 1.0

        # Subclasses may re-enable or retune zero-weight terms. Prune only after
        # the concrete base config has finished all reward customization.
        if self.__class__.__name__ == "UnitreeGo2RoughEnvCfg":
            self.disable_zero_weight_rewards()

        # ------------------------------Terminations------------------------------
        # self.terminations.illegal_contact.params["sensor_cfg"].body_names = [self.base_link_name, ".*_hip"]
        self.terminations.illegal_contact = None

        # ------------------------------Curriculums------------------------------
        # self.curriculum.command_levels_lin_vel.params["range_multiplier"] = (0.2, 1.0)
        # self.curriculum.command_levels_ang_vel.params["range_multiplier"] = (0.2, 1.0)
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        # ------------------------------Commands------------------------------
        # self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        # self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        # self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)


@configclass
class UnitreeGo2RoughCTSEnvCfg(UnitreeGo2RoughEnvCfg):
    """Go2 rough velocity task with CTS role and student-history observation groups."""

    def __post_init__(self):
        super().__post_init__()
        # CTS encodes the current full state for the teacher (§II-B),
        # while only the student encoder receives an observation history.
        self.observations.teacher_full.history_length = 0
        self.observations.student_history = Go2RoughCTSStudentHistoryObservationsCfg()
        self.observations.student_history.base_lin_vel = None
        self.observations.student_history.height_scan = None
        self.observations.student_history.base_ang_vel.scale = 0.25
        self.observations.student_history.joint_vel.scale = 0.05
        self.observations.student_history.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.student_history.joint_vel.params["asset_cfg"].joint_names = self.joint_names
        self.observations.cts_role = Go2RoughCTSRoleObservationsCfg()

        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 0.5)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)

        # CTS reward tuning follows wty-yy/go2_rl_gym Go2 locomotion rewards.
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.base_height_l2.weight = 0.0
        self.rewards.correct_base_height.weight = -1.0
        self.rewards.correct_base_height.params["target_height"] = 0.38
        self.rewards.joint_torques_l2.weight = -1.0e-4
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_pos_limits.weight = -2.0
        self.rewards.joint_power.weight = -2.0e-5
        self.rewards.stand_still.weight = 0.0
        self.rewards.joint_pos_penalty.weight = 0.0
        self.rewards.joint_mirror.weight = 0.0
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.action_smoothness_l2.weight = -0.01
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["threshold"] = 0.1
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [".*(thigh|calf).*"]
        self.rewards.contact_forces.weight = 0.0
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_lin_vel_xy_exp.params["std"] = 0.5
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.track_ang_vel_z_exp.params["std"] = 0.5
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_air_time_variance.weight = 0.0
        self.rewards.feet_gait.weight = 0.0
        self.rewards.feet_contact_without_cmd.weight = 0.0
        self.rewards.feet_slide.weight = 0.0
        self.rewards.feet_height_body.weight = 0.0
        self.rewards.feet_regulation.weight = -0.05
        self.rewards.feet_regulation.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_regulation.params["target_base_height"] = 0.38
        self.rewards.hip_to_default.weight = -0.05
        self.rewards.hip_to_default.params["asset_cfg"].joint_names = ".*_hip_joint"
        self.rewards.upward.weight = 0.0

        self.curriculum.cts_lin_vel_z_weight = CurrTerm(
            func=mdp.linear_reward_weight_by_iteration,
            params={
                "term_name": "lin_vel_z_l2",
                "start_iteration": 0,
                "end_iteration": 1500,
                "start_weight": -2.0,
                "end_weight": 0.0,
            },
        )
        self.curriculum.cts_correct_base_height_weight = CurrTerm(
            func=mdp.linear_reward_weight_by_iteration,
            params={
                "term_name": "correct_base_height",
                "start_iteration": 0,
                "end_iteration": 5000,
                "start_weight": -1.0,
                "end_weight": -10.0,
            },
        )
        self.curriculum.cts_teacher_terrain_level = CurrTerm(
            func=mdp.cts_role_terrain_level,
            params={"role": "teacher", "teacher_fraction": 0.75},
        )
        self.curriculum.cts_student_terrain_level = CurrTerm(
            func=mdp.cts_role_terrain_level,
            params={"role": "student", "teacher_fraction": 0.75},
        )
        if self.__class__.__name__ == "UnitreeGo2RoughCTSEnvCfg":
            self.disable_zero_weight_rewards()
