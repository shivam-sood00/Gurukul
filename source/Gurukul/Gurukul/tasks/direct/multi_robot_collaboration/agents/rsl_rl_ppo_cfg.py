from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class Go2B2CollaborationPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config used after Isaac Lab's MARL-to-single-agent wrapper.

    The repository's RSL-RL training script converts DirectMARLEnv instances to a
    single-agent wrapper, so this policy sees concatenated Go2/B2 observations and
    emits concatenated Go2/B2 actions.
    """

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "go2_b2_collaboration_direct"
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
        entropy_coef=0.01,
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
class Go2B2HierarchicalCollaborationPPORunnerCfg(Go2B2CollaborationPPORunnerCfg):
    """PPO config for high-level velocity-command training over frozen locomotion policies."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "go2_b2_hierarchical_collaboration_direct"
