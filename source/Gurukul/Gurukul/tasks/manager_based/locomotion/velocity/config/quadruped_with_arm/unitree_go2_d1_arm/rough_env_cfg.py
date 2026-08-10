# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import Gurukul.tasks.manager_based.locomotion.velocity.mdp as mdp
from Gurukul.assets.unitree import UNITREE_GO2_D1_ARM_APEX_CFG  # isort: skip
from Gurukul.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from ..unitree_go2_airbot_arm.reset_env_cfg import configure_go2_like_resets


D1_ARM_HARDWARE_VELOCITY_LIMITS = (1.05, 1.05, 1.05, 1.73, 1.73, 1.73)
_D1_NONADJACENT_SELF_COLLISION_FILTERS = (
    ("base_link", ("Link2", "Link3", "Link4", "Link5", "Link6", "Link7_1", "Link7_2")),
    ("Link1", ("Link3", "Link4", "Link5", "Link6", "Link7_1", "Link7_2")),
    # Link2's coarse proxies overlap Link4/Link5 in valid folded poses, so the
    # calibrated USD filters those known proxy overlaps.
    ("Link2", ("Link6", "Link7_1", "Link7_2")),
    ("Link3", ("Link5", "Link6", "Link7_1", "Link7_2")),
    ("Link4", ("Link6", "Link7_1", "Link7_2")),
    ("Link5", ("Link7_1", "Link7_2")),
)


@configclass
class UnitreeGo2D1ArmRoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    base_link_name = "base"
    foot_link_name = ".*_foot"
    leg_joint_names = [
        "FR_hip_joint",
        "FR_thigh_joint",
        "FR_calf_joint",
        "FL_hip_joint",
        "FL_thigh_joint",
        "FL_calf_joint",
        "RR_hip_joint",
        "RR_thigh_joint",
        "RR_calf_joint",
        "RL_hip_joint",
        "RL_thigh_joint",
        "RL_calf_joint",
    ]
    arm_joint_names = [
        "arm_1_joint",
        "arm_2_joint",
        "arm_3_joint",
        "arm_4_joint",
        "arm_5_joint",
        "arm_6_joint",
    ]
    gripper_joint_names = [
        "arm_7_1_joint",
        "arm_7_2_joint",
    ]
    joint_names = leg_joint_names + arm_joint_names + gripper_joint_names
    actuated_joint_names = leg_joint_names + arm_joint_names + ["arm_7_1_joint"]

    def __post_init__(self):
        super().__post_init__()

        self.sim.physx.gpu_max_rigid_patch_count = 32 * 2**15
        # Use the same calibrated plant as Go2+D1 APEX: corrected D1
        # masses/inertias and joint conventions, filtered physical
        # self-collisions, and explicit torque-limited arm PD dynamics.
        self.scene.robot = UNITREE_GO2_D1_ARM_APEX_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.actuators["legs"].stiffness = 20.0
        self.scene.robot.actuators["legs"].damping = 0.5
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        self.observations.policy.base_lin_vel.scale = 2.0
        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_pos.scale = 1.0
        self.observations.policy.joint_vel.scale = 0.05
        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.policy.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        # D1 deployment reports measured arm positions but no arm velocity
        # signal. Keep the actor deployable and reserve full joint velocities
        # for the asymmetric critic.
        self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.leg_joint_names
        self.observations.critic.joint_pos.params["asset_cfg"].joint_names = self.joint_names
        self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names

        self.actions.joint_pos.scale = {
            r"^(FL|FR|RL|RR)_hip_joint$": 0.125,
            r"^(FL|FR|RL|RR)_(thigh|calf)_joint$": 0.25,
            r"^arm_[1-6]_joint$": 0.25,
            r"^arm_7_[12]_joint$": 0.01,
        }
        self.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
        self.actions.joint_pos.joint_names = self.joint_names

        configure_go2_like_resets(self)
        self.events.randomize_reset_joints.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.joint_names, preserve_order=True
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
        self.events.randomize_rigid_body_mass_base.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_rigid_body_mass_others.params["asset_cfg"].body_names = [
            f"^(?!.*{self.base_link_name}).*"
        ]
        # PhysX requires every sampled static-friction effort to be at least
        # the corresponding dynamic-friction effort.
        self.events.randomize_rigid_body_material.params["static_friction_range"] = (0.8, 1.0)
        self.events.randomize_rigid_body_material.params["dynamic_friction_range"] = (0.3, 0.8)
        self.events.randomize_com_positions.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_apply_external_force_torque.params["asset_cfg"].body_names = [self.base_link_name]
        self.events.randomize_actuator_gains.params["asset_cfg"] = SceneEntityCfg(
            "robot", joint_names=self.actuated_joint_names, preserve_order=True
        )
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.9, 1.1)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.9, 1.1)

        self.rewards.is_terminated.weight = 0
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = 0
        self.rewards.base_height_l2.weight = 0
        self.rewards.base_height_l2.params["target_height"] = 0.33
        self.rewards.base_height_l2.params["asset_cfg"].body_names = [self.base_link_name]
        self.rewards.body_lin_acc_l2.weight = 0
        self.rewards.body_lin_acc_l2.params["asset_cfg"].body_names = [self.base_link_name]

        self.rewards.joint_torques_l2.weight = -2.5e-5
        self.rewards.joint_vel_l2.weight = 0
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_pos_limits.weight = -5.0
        self.rewards.joint_vel_limits.weight = 0
        self.rewards.d1_arm_joint_velocity_limits = RewTerm(
            func=mdp.joint_velocity_soft_limits_l2,
            weight=-0.25,
            params={
                "asset_cfg": SceneEntityCfg(
                    "robot", joint_names=self.arm_joint_names, preserve_order=True
                ),
                "velocity_limits": D1_ARM_HARDWARE_VELOCITY_LIMITS,
                "max_excess_ratio": 2.0,
            },
        )
        self.rewards.joint_power.weight = -2e-5
        self.rewards.stand_still.weight = -2.0
        self.rewards.joint_pos_penalty.weight = -1.0
        self.rewards.joint_mirror.weight = -0.05
        self.rewards.joint_mirror.params["mirror_joints"] = [
            ["FR_(hip|thigh|calf).*", "RL_(hip|thigh|calf).*"],
            ["FL_(hip|thigh|calf).*", "RR_(hip|thigh|calf).*"],
        ]

        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.undesired_contacts.weight = -1.0
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.contact_forces.weight = -1.5e-4
        self.rewards.contact_forces.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.track_lin_vel_xy_exp.weight = 3.0
        self.rewards.track_ang_vel_z_exp.weight = 1.5
        self.rewards.feet_air_time.weight = 0.2
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
        for source_body, target_bodies in _D1_NONADJACENT_SELF_COLLISION_FILTERS:
            sensor_name = f"d1_nonadjacent_contact_{source_body.lower()}"
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
            setattr(
                self.rewards,
                sensor_name,
                RewTerm(
                    func=mdp.self_collision_cost,
                    weight=-2.0,
                    params={
                        "sensor_cfg": SceneEntityCfg(sensor_name, body_names=[source_body]),
                        "force_threshold": 5.0,
                    },
                ),
            )

        if self.__class__.__name__ == "UnitreeGo2D1ArmRoughEnvCfg":
            self.disable_zero_weight_rewards()

        self.terminations.illegal_contact = None
        self.curriculum.command_levels = None
