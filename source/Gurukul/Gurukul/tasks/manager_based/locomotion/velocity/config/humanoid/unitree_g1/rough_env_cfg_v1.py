"""G1 rough velocity env: command / push / gain DR / torso-rate penalty aligned with mjlab G1 velocity."""

import math

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import Gurukul.tasks.manager_based.locomotion.velocity.mdp as mdp

from .rough_env_cfg import UnitreeG1RoughEnvCfg


@configclass
class UnitreeG1RoughEnvCfgV1(UnitreeG1RoughEnvCfg):
    """Parity tweaks vs mjlab G1 rough velocity plus base velocity defaults."""

    def __post_init__(self):
        super().__post_init__()

        # G1 mjlab attaches terrain scan to the pelvis while keeping torso_link as the controlled/upright body.
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/pelvis"
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/pelvis"
        foot_sensor_cfg = SceneEntityCfg("contact_forces", body_names=[self.foot_link_name])
        foot_asset_cfg = SceneEntityCfg("robot", body_names=[self.foot_link_name])

        # ``mjlab/tasks/velocity/velocity_env_cfg.py`` command head (Isaac has no ``rel_forward_envs`` on this cfg).
        self.commands.base_velocity.resampling_time_range = (3.0, 8.0)
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.rel_heading_envs = 0.3
        self.commands.base_velocity.ranges.lin_vel_x = (-1.0, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-1.0, 1.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

        # ``make_velocity_env_cfg`` initial sampling: xy/z/yaw offsets, no randomized root velocity.
        self.events.randomize_reset_base.params = {
            "pose_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (0.01, 0.05),
                "yaw": (-3.14, 3.14),
            },
            "velocity_range": {},
        }
        self.events.randomize_reset_joints.func = mdp.reset_joints_by_offset
        self.events.randomize_reset_joints.params = {"position_range": (0.0, 0.0), "velocity_range": (0.0, 0.0)}

        self.events.randomize_push_robot.interval_range_s = (1.0, 3.0)
        self.events.randomize_push_robot.params = {
            "velocity_range": {
                "x": (-0.5, 0.5),
                "y": (-0.5, 0.5),
                "z": (-0.4, 0.4),
                "roll": (-0.52, 0.52),
                "pitch": (-0.52, 0.52),
                "yaw": (-0.78, 0.78),
            }
        }

        # Match the mjlab G1 rough actor/critic observation surface where Isaac equivalents exist.
        self.observations.policy.base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            noise=Unoise(n_min=-0.5, n_max=0.5),
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        self.observations.policy.base_ang_vel.scale = 1.0
        self.observations.policy.joint_vel.scale = 1.0
        self.observations.policy.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        self.observations.critic.base_lin_vel = ObsTerm(
            func=mdp.base_lin_vel,
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        self.observations.critic.base_ang_vel.scale = 1.0
        self.observations.critic.joint_vel.scale = 1.0
        self.observations.critic.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        self.observations.critic.foot_height = ObsTerm(
            func=mdp.feet_height_observation,
            params={"asset_cfg": foot_asset_cfg, "height_sensor_name": "height_scanner"},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        self.observations.critic.foot_air_time = ObsTerm(
            func=mdp.feet_air_time_observation,
            params={"sensor_cfg": foot_sensor_cfg},
            clip=(0.0, 100.0),
            scale=1.0,
        )
        self.observations.critic.foot_contact = ObsTerm(
            func=mdp.feet_contact_observation,
            params={"sensor_cfg": foot_sensor_cfg, "force_threshold": 1.0},
            clip=(0.0, 1.0),
            scale=1.0,
        )
        self.observations.critic.foot_contact_forces = ObsTerm(
            func=mdp.feet_contact_force_observation,
            params={"sensor_cfg": foot_sensor_cfg},
            clip=(-1000.0, 1000.0),
            scale=1.0,
        )

        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.8, 1.2)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.8, 1.2)

        # mjlab G1 uses foot friction + torso COM randomization, but no mass DR or reset wrench.
        self.events.randomize_rigid_body_mass_base = None
        self.events.randomize_rigid_body_mass_others = None
        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_rigid_body_material.params["asset_cfg"].body_names = [self.foot_link_name]
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.3, 1.2)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.3, 1.2)
        self.events.randomize_rigid_body_material.params["restitution_range"] = (0.0, 0.0)
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_com_positions.params["com_range"] = {
            "x": (-0.025, 0.025),
            "y": (-0.025, 0.025),
            "z": (-0.03, 0.03),
        }

        # Mjlab uses a pelvis self-collision contact sensor for G1.
        self.scene.self_collision = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/pelvis",
            history_length=4,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Robot/.*"],
        )

        # ``make_velocity_env_cfg`` reward surface plus G1-specific mjlab tweaks.
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_linear_velocity_exp
        self.rewards.track_lin_vel_xy_exp.params["std"] = math.sqrt(0.25)
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.func = mdp.track_angular_velocity_exp
        self.rewards.track_ang_vel_z_exp.params["std"] = math.sqrt(0.5)

        # Clear Gurukul-only shaping terms so rough v1 keeps the mjlab reward surface.
        self.rewards.is_terminated.weight = 0.0
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.ang_vel_xy_l2.weight = 0.0
        self.rewards.flat_orientation_l2.weight = 0.0
        self.rewards.joint_torques_l2.weight = 0.0
        self.rewards.joint_acc_l2.weight = 0.0
        self.rewards.joint_pos_penalty.weight = 0.0
        self.rewards.joint_deviation_hip_l1.weight = 0.0
        self.rewards.joint_deviation_arms_l1.weight = 0.0
        self.rewards.joint_deviation_torso_l1.weight = 0.0
        self.rewards.upward.weight = 0.0

        self.rewards.joint_pos_limits.weight = -1.0
        self.rewards.action_rate_l2.weight = -0.1
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_slide.weight = -0.1
        self.rewards.feet_slide.func = mdp.feet_slip
        self.rewards.feet_slide.params["command_name"] = "base_velocity"
        self.rewards.feet_slide.params["command_threshold"] = 0.05

        # mjlab ``body_ang_vel`` on torso_link at -0.05.
        self.rewards.body_ang_vel = RewTerm(
            func=mdp.body_ang_vel_xy_l2,
            weight=-0.05,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=[self.base_link_name])},
        )
        self.rewards.angular_momentum = RewTerm(
            func=mdp.angular_momentum_l2,
            weight=-0.02,
            params={"asset_cfg": SceneEntityCfg("robot", body_names=".*")},
        )

        self.rewards.pose = RewTerm(
            func=mdp.variable_posture,
            weight=1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
                "command_name": "base_velocity",
                "std_standing": {".*": 0.05},
                "std_walking": {
                    r".*hip_pitch.*": 0.3,
                    r".*hip_roll.*": 0.15,
                    r".*hip_yaw.*": 0.15,
                    r".*knee.*": 0.35,
                    r".*ankle_pitch.*": 0.25,
                    r".*ankle_roll.*": 0.1,
                    r".*waist_yaw.*": 0.2,
                    r".*waist_roll.*": 0.08,
                    r".*waist_pitch.*": 0.1,
                    r".*shoulder_pitch.*": 0.15,
                    r".*shoulder_roll.*": 0.15,
                    r".*shoulder_yaw.*": 0.1,
                    r".*elbow.*": 0.15,
                    r".*wrist.*": 0.3,
                },
                "std_running": {
                    r".*hip_pitch.*": 0.5,
                    r".*hip_roll.*": 0.2,
                    r".*hip_yaw.*": 0.2,
                    r".*knee.*": 0.6,
                    r".*ankle_pitch.*": 0.35,
                    r".*ankle_roll.*": 0.15,
                    r".*waist_yaw.*": 0.3,
                    r".*waist_roll.*": 0.08,
                    r".*waist_pitch.*": 0.2,
                    r".*shoulder_pitch.*": 0.5,
                    r".*shoulder_roll.*": 0.2,
                    r".*shoulder_yaw.*": 0.15,
                    r".*elbow.*": 0.35,
                    r".*wrist.*": 0.3,
                },
                "walking_threshold": 0.05,
                "running_threshold": 1.5,
            },
        )
        self.rewards.upright = RewTerm(
            func=mdp.upright_exp,
            weight=1.0,
            params={"std": math.sqrt(0.2), "asset_cfg": SceneEntityCfg("robot", body_names=[self.base_link_name])},
        )
        self.rewards.foot_clearance = RewTerm(
            func=mdp.feet_clearance,
            weight=-2.0,
            params={
                "target_height": 0.1,
                "height_sensor_name": "height_scanner",
                "command_name": "base_velocity",
                "command_threshold": 0.05,
                "asset_cfg": foot_asset_cfg,
            },
        )
        self.rewards.foot_swing_height = RewTerm(
            func=mdp.feet_swing_height,
            weight=-0.25,
            params={
                "sensor_cfg": foot_sensor_cfg,
                "asset_cfg": foot_asset_cfg,
                "height_sensor_name": "height_scanner",
                "target_height": 0.1,
                "command_name": "base_velocity",
                "command_threshold": 0.05,
            },
        )
        self.rewards.soft_landing = RewTerm(
            func=mdp.soft_landing,
            weight=-1.0e-5,
            params={"sensor_cfg": foot_sensor_cfg, "command_name": "base_velocity", "command_threshold": 0.05},
        )
        self.rewards.self_collisions = RewTerm(
            func=mdp.self_collision_cost,
            weight=-1.0,
            params={"sensor_cfg": SceneEntityCfg("self_collision"), "force_threshold": 10.0},
        )

        # mjlab G1 terminates on orientation/terrain bounds, not torso ground contact.
        self.terminations.illegal_contact = None
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(70.0)},
        )

        if self.__class__.__name__ == "UnitreeG1RoughEnvCfgV1":
            self.disable_zero_weight_rewards()
