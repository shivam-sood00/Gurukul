"""G1 flat velocity: ref-style flat walking rewards on top of the mjlab-oriented v1 base."""

import math

from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import Gurukul.tasks.manager_based.locomotion.velocity.mdp as mdp

from .rough_env_cfg_v1 import UnitreeG1RoughEnvCfgV1


@configclass
class UnitreeG1FlatEnvCfgV1(UnitreeG1RoughEnvCfgV1):
    def __post_init__(self):
        super().__post_init__()

        foot_sensor_cfg = SceneEntityCfg("contact_forces", body_names=[self.foot_link_name])
        foot_asset_cfg = SceneEntityCfg("robot", body_names=[self.foot_link_name])

        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None

        self.observations.policy.base_lin_vel = None
        self.observations.policy.base_ang_vel.scale = 0.2
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.height_scan = None
        self.observations.policy.history_length = 5
        self.observations.critic.base_ang_vel.scale = 0.2
        self.observations.critic.joint_vel.scale = 0.05
        self.observations.critic.height_scan = None
        self.observations.critic.foot_height = None
        self.observations.critic.foot_air_time = None
        self.observations.critic.foot_contact = None
        self.observations.critic.foot_contact_forces = None
        self.observations.critic.history_length = 5

        self.events.randomize_push_robot.interval_range_s = (5.0, 5.0)
        self.events.randomize_push_robot.params = {"velocity_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5)}}

        self.commands.base_velocity.resampling_time_range = (10.0, 10.0)
        self.commands.base_velocity.rel_standing_envs = 0.02
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.3, 0.3)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.2, 0.2)

        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_lin_vel_xy_exp.func = mdp.track_lin_vel_xy_yaw_frame_exp
        self.rewards.track_lin_vel_xy_exp.params["std"] = math.sqrt(0.25)
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.track_ang_vel_z_exp.func = mdp.track_ang_vel_z_exp
        self.rewards.track_ang_vel_z_exp.params["std"] = math.sqrt(0.25)

        self.rewards.alive = RewTerm(func=mdp.is_alive, weight=0.15)

        self.rewards.is_terminated.weight = 0.0
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -5.0
        self.rewards.base_height_l2.weight = -10.0
        self.rewards.base_height_l2.params["target_height"] = 0.78

        self.rewards.joint_torques_l2.weight = 0.0
        self.rewards.joint_vel_l2.weight = -0.001
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_power.weight = -2.0e-5
        self.rewards.joint_pos_penalty.weight = 0.0
        self.rewards.joint_error.weight = 0.0
        self.rewards.joint_vel_limits.weight = 0.0
        self.rewards.stand_still.weight = 0.0
        self.rewards.joint_mirror.weight = 0.0
        self.rewards.action_mirror.weight = 0.0
        self.rewards.action_sync.weight = 0.0
        self.rewards.joint_deviation_hip_l1.weight = -1.0
        self.rewards.joint_deviation_hip_l1.params["asset_cfg"].joint_names = [
            ".*_hip_roll_joint",
            ".*_hip_yaw_joint",
        ]
        self.rewards.joint_deviation_arms_l1.weight = -0.1
        self.rewards.joint_deviation_arms_l1.params["asset_cfg"].joint_names = [
            ".*_shoulder_.*_joint",
            ".*_elbow_joint",
            ".*_wrist_.*",
        ]
        self.rewards.joint_deviation_torso_l1.weight = -1.0
        self.rewards.joint_deviation_torso_l1.params["asset_cfg"].joint_names = ["waist.*"]

        self.rewards.action_rate_l2.weight = -0.05
        self.rewards.applied_torque_limits.weight = 0.0
        self.rewards.action_smoothness_l2.weight = 0.0

        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = ["(?!.*ankle.*).*"]
        self.rewards.undesired_contacts.params["threshold"] = 1.0
        self.rewards.contact_forces.weight = 0.0

        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_gait.weight = 0.5
        self.rewards.feet_gait.func = mdp.feet_gait_pattern
        self.rewards.feet_gait.params = {
            "period": 0.8,
            "offset": [0.0, 0.5],
            "threshold": 0.55,
            "command_name": "base_velocity",
            "sensor_cfg": foot_sensor_cfg,
        }
        self.rewards.feet_contact.weight = 0.0
        self.rewards.feet_contact_without_cmd.weight = 0.0
        self.rewards.feet_stumble.weight = 0.0
        self.rewards.feet_edge.weight = 0.0
        self.rewards.feet_slide.weight = -0.2
        self.rewards.feet_slide.func = mdp.feet_slide
        self.rewards.feet_slide.params = {
            "asset_cfg": foot_asset_cfg,
            "sensor_cfg": foot_sensor_cfg,
        }
        self.rewards.feet_height.weight = 0.0
        self.rewards.feet_height_body.weight = 0.0
        self.rewards.feet_distance_y_exp.weight = 0.0
        self.rewards.upward.weight = 0.0

        self.rewards.body_ang_vel.weight = 0.0
        self.rewards.angular_momentum.weight = 0.0
        self.rewards.pose.weight = 0.0
        self.rewards.upright.weight = 0.0
        self.rewards.foot_clearance.weight = 1.0
        self.rewards.foot_clearance.func = mdp.foot_clearance_reward
        self.rewards.foot_clearance.params = {
            "std": 0.05,
            "tanh_mult": 2.0,
            "target_height": 0.1,
            "asset_cfg": foot_asset_cfg,
        }
        self.rewards.foot_swing_height.weight = 0.0
        self.rewards.soft_landing.weight = 0.0
        self.rewards.self_collisions.weight = 0.0

        self.curriculum.terrain_levels = None
        self.terminations.terrain_out_of_bounds = None
        self.terminations.base_height = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.2},
        )
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": 0.8},
        )

        if self.__class__.__name__ == "UnitreeG1FlatEnvCfgV1":
            self.disable_zero_weight_rewards()
