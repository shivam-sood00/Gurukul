"""Go2 rough velocity env aligned with mjlab Go2 defaults."""

import math

from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import Gurukul.tasks.manager_based.locomotion.velocity.mdp as mdp

from .mjlab_parity_constants import GO2_MJLAB_PARITY_ACTION_SCALE, UNITREE_GO2_VELOCITY_MJLAB_V1_CFG
from .rough_env_cfg import UnitreeGo2RoughEnvCfg


@configclass
class UnitreeGo2RoughEnvCfgV1(UnitreeGo2RoughEnvCfg):
    """Go2 rough velocity with mjlab-style PD, action scale, commands, and core reward weights."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_GO2_VELOCITY_MJLAB_V1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.scale = GO2_MJLAB_PARITY_ACTION_SCALE

        # ``mjlab/tasks/velocity/velocity_env_cfg.py`` UniformVelocityCommandCfg
        self.commands.base_velocity.resampling_time_range = (3.0, 8.0)
        self.commands.base_velocity.rel_standing_envs = 0.1
        self.commands.base_velocity.rel_heading_envs = 0.3
        self.commands.base_velocity.ranges.ang_vel_z = (-0.5, 0.5)

        self.observations.policy.height_scan = ObsTerm(
            func=mdp.height_scan,
            params={"sensor_cfg": SceneEntityCfg("height_scanner")},
            noise=Unoise(n_min=-0.1, n_max=0.1),
            clip=(-1.0, 1.0),
            scale=1.0,
        )

        # ``mjlab/tasks/velocity/config/go2/env_cfgs.py`` push + mass DR
        self.events.randomize_push_robot.interval_range_s = (15.0, 20.0)
        self.events.randomize_push_robot.params = {"velocity_range": {"x": (-0.25, 0.25), "y": (-0.25, 0.25)}}
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-0.5, 1.5)
        self.events.randomize_rigid_body_mass_others.params["mass_distribution_params"] = (0.9, 1.1)

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

        # ``make_velocity_env_cfg`` tracking + penalties (before Go2-specific mjlab tweaks)
        self.rewards.track_lin_vel_xy_exp.weight = 2.0
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_linear_velocity_exp
        self.rewards.track_lin_vel_xy_exp.params["std"] = math.sqrt(0.25)
        self.rewards.track_ang_vel_z_exp.weight = 2.0
        self.rewards.track_ang_vel_z_exp.func = mdp.track_angular_velocity_exp
        self.rewards.track_ang_vel_z_exp.params["std"] = math.sqrt(0.5)

        # Clear Gurukul-only shaping terms so rough v1 keeps the mjlab reward surface.
        self.rewards.lin_vel_z_l2.weight = 0.0
        self.rewards.ang_vel_xy_l2.weight = 0.0
        self.rewards.joint_torques_l2.weight = 0.0
        self.rewards.joint_acc_l2.weight = 0.0
        self.rewards.joint_power.weight = 0.0
        self.rewards.stand_still.weight = 0.0
        self.rewards.joint_pos_penalty.weight = 0.0
        self.rewards.joint_mirror.weight = 0.0
        self.rewards.undesired_contacts.weight = 0.0
        self.rewards.contact_forces.weight = 0.0
        self.rewards.feet_contact_without_cmd.weight = 0.0
        self.rewards.feet_height_body.weight = 0.0
        self.rewards.feet_gait.weight = 0.0
        self.rewards.upward.weight = 0.0

        self.rewards.action_rate_l2.weight = -0.1
        self.rewards.joint_pos_limits.weight = -1.0
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_air_time_variance.weight = 0.0
        self.rewards.feet_slide.weight = -0.1
        self.rewards.feet_slide.func = mdp.feet_slip
        self.rewards.feet_slide.params["command_name"] = "base_velocity"
        self.rewards.feet_slide.params["command_threshold"] = 0.05

        # Mjlab-style reward terms that are not present in the base Gurukul surface.
        foot_sensor_cfg = SceneEntityCfg("contact_forces", body_names=[self.foot_link_name])
        foot_asset_cfg = SceneEntityCfg("robot", body_names=[self.foot_link_name])
        self.rewards.pose = RewTerm(
            func=mdp.variable_posture,
            weight=1.0,
            params={
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.joint_names),
                "command_name": "base_velocity",
                "std_standing": {
                    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.05,
                    r".*(FR|FL|RR|RL)_calf_joint.*": 0.1,
                },
                "std_walking": {
                    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
                    r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
                },
                "std_running": {
                    r".*(FR|FL|RR|RL)_(hip|thigh)_joint.*": 0.3,
                    r".*(FR|FL|RR|RL)_calf_joint.*": 0.6,
                },
                "walking_threshold": 0.05,
                "running_threshold": 1.5,
            },
        )
        self.rewards.upright = RewTerm(
            func=mdp.upright_exp,
            weight=1.0,
            params={
                "std": math.sqrt(0.2),
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.base_link_name]),
                "terrain_sensor_cfg": SceneEntityCfg("height_scanner"),
            },
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
        self.terminations.illegal_contact = DoneTerm(
            func=mdp.illegal_contact,
            params={
                "sensor_cfg": SceneEntityCfg(
                    "contact_forces",
                    body_names=[r"^(base|[FR][LR]_(hip|thigh))$"],
                ),
                "threshold": 10.0,
            },
        )

        if self.__class__.__name__ == "UnitreeGo2RoughEnvCfgV1":
            self.disable_zero_weight_rewards()
