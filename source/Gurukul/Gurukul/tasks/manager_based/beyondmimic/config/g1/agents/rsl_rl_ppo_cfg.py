from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg

from Gurukul.tasks.manager_based.go2_apex.config.go2.agents.rsl_rl_multi_critic_cfg import configure_multi_critic


@configclass
class UnitreeG1BeyondMimicFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 30000
    save_interval = 500
    experiment_name = "unitree_g1_beyondmimic_flat"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.005,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )


@configclass
class UnitreeG1ApexFlatMultiCriticRunnerCfg(UnitreeG1BeyondMimicFlatPPORunnerCfg):
    """Independent motion-tracking and regularization critics for the G1 APEX aliases."""

    def __post_init__(self):
        super().__post_init__()
        configure_multi_critic(self)
        self.experiment_name = "unitree_g1_apex_flat_multi_critic"
        self.multi_critic_reward_term_groups = [
            [
                "motion_global_anchor_pos",
                "motion_global_anchor_ori",
                "motion_body_pos",
                "motion_body_ori",
                "motion_body_lin_vel",
                "motion_body_ang_vel",
            ],
            ["joint_acc_l2", "joint_torques_l2", "action_rate_l2", "joint_pos_limits", "undesired_contacts"],
        ]
