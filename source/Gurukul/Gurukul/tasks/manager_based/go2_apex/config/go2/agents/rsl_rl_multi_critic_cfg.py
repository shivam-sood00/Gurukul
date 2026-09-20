import copy

from isaaclab.utils import configclass

from . import rsl_rl_ppo_cfg as ppo
from . import rsl_rl_teacher_cfg as teacher
from .rsl_rl_ppo_cfg import UnitreeGo2ApexFlatPPORunnerCfg


@configclass
class UnitreeGo2ApexFlatMultiCriticRunnerCfg(UnitreeGo2ApexFlatPPORunnerCfg):
    """APEX independent critics with Gurukul's existing Go2 rewards."""

    multi_critic_recurrent: bool = False
    multi_critic_rnn_type: str = "lstm"
    multi_critic_rnn_hidden_dim: int = 256
    multi_critic_rnn_num_layers: int = 1
    multi_critic_std_range: list[float] = [0.01, 2.0]

    def __post_init__(self):
        super().__post_init__()
        self.class_name = "MultiCriticRunner"
        self.experiment_name = "unitree_go2_apex_flat_multi_critic"
        self.obs_groups = {"policy": ["policy"], "critic": ["critic"]}

        # Retain the policy config schema; the runner builds native separate models.
        self.policy.class_name = "MultiCriticActorCritic"
        self.policy.noise_std_type = "log"
        self.algorithm.class_name = "MultiCriticPPO"

        # Two value heads sharing the privileged critic observation.
        self.multi_critic_groups = [["critic"], ["critic"]]
        self.multi_critic_advantage_weights = [0.5, 0.5]
        self.multi_critic_reward_term_groups = [
            [
                "imitate_joint_pos",
                "imitate_base_orientation",
                "imitate_projected_gravity",
                "imitate_foot_pos",
                "imitate_world_foot_pos",
                "imitate_world_base_pos",
                "airborne_contact",
                "reference_foot_contact",
            ],
            [
                "track_command_lin_vel_xy",
                "track_command_lin_vel_z",
                "track_command_ang_vel_z",
                "imitate_base_height",
                "ang_vel_xy_l2",
                "joint_acc_l2",
                "joint_torques_l2",
                "action_rate_l2",
                "action_smoothness_l2",
                "feet_slip",
                "impact_reduction",
                "undesired_contacts",
            ],
        ]


@configclass
class UnitreeGo2ApexFlatTrackerMultiCriticRunnerCfg(UnitreeGo2ApexFlatMultiCriticRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_go2_apex_flat_tracker_multi_critic"
        self.clip_actions = None


@configclass
class UnitreeGo2ApexFlatOneStepFutureMultiCriticRunnerCfg(UnitreeGo2ApexFlatTrackerMultiCriticRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_go2_apex_flat_tracker_one_step_future_multi_critic"


@configclass
class UnitreeGo2ApexFlatHistoryMultiCriticRunnerCfg(UnitreeGo2ApexFlatOneStepFutureMultiCriticRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_go2_apex_flat_tracker_one_step_future_history_multi_critic"
        self.policy.actor_hidden_dims = [1024, 512, 256]
        self.policy.critic_hidden_dims = [1024, 512, 256]


@configclass
class UnitreeGo2ApexFlatPrivilegedMultiCriticRunnerCfg(UnitreeGo2ApexFlatTrackerMultiCriticRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_go2_apex_flat_privileged_tracker_multi_critic"
        self.obs_groups = {"policy": ["privileged"], "critic": ["privileged"]}
        self.multi_critic_groups = [["privileged"], ["privileged"]]
        self.policy.actor_hidden_dims = [1024, 512, 256]
        self.policy.critic_hidden_dims = [1024, 512, 256]


def configure_multi_critic(cfg):
    """Enable grouped PPO while retaining an existing task's policy and optimizer settings."""
    base = UnitreeGo2ApexFlatMultiCriticRunnerCfg()
    for name, value in vars(base).items():
        if name.startswith("multi_critic_"):
            setattr(cfg, name, copy.deepcopy(value))
    cfg.class_name = "MultiCriticRunner"
    cfg.experiment_name += "_multi_critic"
    cfg.policy.class_name = "MultiCriticActorCritic"
    cfg.policy.noise_std_type = "log"
    cfg.algorithm.class_name = "MultiCriticPPO"
    groups = cfg.obs_groups
    if not isinstance(groups, dict):
        groups = {"policy": ["policy"], "critic": ["critic"]}
    cfg.obs_groups = {
        "policy": groups.get("actor", groups.get("policy", ["policy"])),
        "critic": groups.get("critic", ["critic"]),
    }
    cfg.multi_critic_groups = [list(cfg.obs_groups["critic"]), list(cfg.obs_groups["critic"])]


def _configure_arm_multi_critic(cfg, family):
    """Add the arm family's explicit imitation/regularization reward partition."""
    configure_multi_critic(cfg)
    imitation, regularization = cfg.multi_critic_reward_term_groups
    imitation += ["imitate_joint_pos_legs", "imitate_arm_ee_pos"]
    if family == "go2_d1":
        imitation += [
            "imitate_arm_joint_pos_proximal",
            "imitate_arm_joint_pos_wrist",
            "imitate_arm_ee_orientation",
            "imitate_gripper_joint_pos",
            "imitate_leg_policy_targets",
            "imitate_arm_policy_targets_proximal",
            "imitate_arm_policy_targets_wrist",
            "imitate_gripper_policy_targets",
            "imitate_world_base_pos_huber",
            "imitate_gripper_state",
            "imitate_object_pos",
            "imitate_object_pos_huber",
            "imitate_object_up_axis",
            "imitate_object_linear_velocity",
            "imitate_attached_object_offset",
            "attached_bilateral_gripper_contact",
            "attached_without_bilateral_gripper_contact",
        ]
        regularization += [
            "d1_arm_action_rate",
            "d1_arm_action_smoothness",
            "d1_arm_joint_velocity",
            "d1_gripper_action_rate",
            "d1_gripper_action_smoothness",
            "d1_arm_joint_acc_l2",
            "d1_arm_joint_torques_l2",
            "d1_arm_joint_velocity_limits",
            "d1_nonadjacent_self_collision",
            "bad_tracking_termination",
        ]
    elif family == "b2_z1":
        imitation += ["imitate_joint_pos_arms", "track_gripper_command"]
        regularization += [
            "joint_acc_l2_legs",
            "joint_acc_l2_arms",
            "joint_torques_l2_legs",
            "joint_torques_l2_arms",
            "action_rate_l2_legs",
            "action_rate_l2_arms",
            "action_smoothness_l2_legs",
            "action_smoothness_l2_arms",
            "joint_pos_limits_legs",
            "default_leg_joint_pos",
            "legs_distance",
            "feet_contact_forces",
        ]
    else:
        raise ValueError(f"Unknown APEX arm family: {family}")


@configclass
class UnitreeGo2D1ArmApexFlatTrackerMultiCriticRunnerCfg(ppo.UnitreeGo2D1ArmApexFlatTrackerPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_arm_multi_critic(self, "go2_d1")


@configclass
class UnitreeGo2D1ArmApexOriginalDecapTeacherMultiCriticRunnerCfg(
    teacher.UnitreeGo2D1ArmApexOriginalDecapTeacherPPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        _configure_arm_multi_critic(self, "go2_d1")


@configclass
class UnitreeGo2D1ArmApexPickStowCarryMultiCriticRunnerCfg(ppo.UnitreeGo2D1ArmApexPickStowCarryFlatTrackerPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_arm_multi_critic(self, "go2_d1")


@configclass
class UnitreeGo2D1ArmApexPickStowCarryRobotOnlyMultiCriticRunnerCfg(
    ppo.UnitreeGo2D1ArmApexPickStowCarryRobotOnlyPPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        _configure_arm_multi_critic(self, "go2_d1")


@configclass
class UnitreeGo2D1ArmApexCanPickCarryDropMultiCriticRunnerCfg(
    ppo.UnitreeGo2D1ArmApexCanPickCarryDropFlatTrackerPPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        _configure_arm_multi_critic(self, "go2_d1")


@configclass
class UnitreeGo2D1ArmApexPickStowCarryPrivilegedTeacherMultiCriticRunnerCfg(
    teacher.UnitreeGo2D1ArmApexPickStowCarryPrivilegedTeacherPPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        _configure_arm_multi_critic(self, "go2_d1")


@configclass
class UnitreeB2Z1ArmApexFlatTrackerMultiCriticRunnerCfg(ppo.UnitreeB2Z1ArmApexFlatTrackerPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        _configure_arm_multi_critic(self, "b2_z1")


@configclass
class UnitreeB2Z1ArmApexFlatPrivilegedTrackerMultiCriticRunnerCfg(
    teacher.UnitreeB2Z1ArmApexFlatPrivilegedTrackerPPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        _configure_arm_multi_critic(self, "b2_z1")


@configclass
class UnitreeB2Z1ArmApexFlatOneStepFutureTrackerMultiCriticRunnerCfg(
    ppo.UnitreeB2Z1ArmApexFlatOneStepFutureTrackerPPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        _configure_arm_multi_critic(self, "b2_z1")


@configclass
class UnitreeB2Z1ArmApexFlatOneStepFutureTrackerHistoryMultiCriticRunnerCfg(
    ppo.UnitreeB2Z1ArmApexFlatOneStepFutureTrackerHistoryPPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        _configure_arm_multi_critic(self, "b2_z1")


@configclass
class UnitreeB2Z1ArmApexFlatFixedWristGripperTrackerMultiCriticRunnerCfg(
    ppo.UnitreeB2Z1ArmApexFlatFixedWristGripperTrackerPPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        _configure_arm_multi_critic(self, "b2_z1")


@configclass
class UnitreeB2Z1ArmApexFlatFixedWristGripperPrivilegedTrackerMultiCriticRunnerCfg(
    teacher.UnitreeB2Z1ArmApexFlatFixedWristGripperPrivilegedTrackerPPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        _configure_arm_multi_critic(self, "b2_z1")
