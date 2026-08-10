from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import UnitreeGo2ApexFlatPPORunnerCfg


@configclass
class UnitreeGo2ApexFlatMultiCriticRunnerCfg(UnitreeGo2ApexFlatPPORunnerCfg):
    """APEX-style multi-critic runner aligned with the IsaacGym reference setup."""

    def __post_init__(self):
        super().__post_init__()
        self.class_name = "MultiCriticRunner"
        self.experiment_name = "unitree_go2_apex_flat_multi_critic"
        self.obs_groups = {"policy": ["policy"], "critic": ["critic"]}

        # Use the custom multi-head actor-critic defined in scripts/reinforcement_learning/rsl_rl/multi_critic.py.
        self.policy.class_name = "MultiCriticActorCritic"
        self.algorithm.class_name = "MultiCriticPPO"

        # Two value heads sharing the privileged critic observation.
        self.multi_critic_groups = [["critic"], ["critic"]]
        self.multi_critic_reward_weights = [0.5, 0.5]
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
