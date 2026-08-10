from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class StartActorCriticCfg(RslRlPpoActorCriticRecurrentCfg):
    """Configuration for START single-stage actor-critic."""

    class_name: str = "StartActorCritic"
    depth_obs_group: str = "depth_camera"
    depth_image_shape: tuple[int, int] = (60, 60)
    depth_backbone_channels: list[int] = [32, 64]
    depth_backbone_kernels: list[int] = [5, 3]
    depth_backbone_pool_kernel: int = 2
    depth_backbone_image_fc_dim: int = 128
    depth_backbone_latent_dim: int = 64

    terrain_heightmap_obs_group: str = "terrain_heightmap"
    feet_heightmap_obs_group: str = "feet_heightmap"
    base_velocity_obs_group: str = "base_velocity_gt"
    terrain_map_shape: tuple[int, int] | None = None

    tr_proprio_hidden_dim: int = 128
    tr_rnn_hidden_dim: int = 256
    tr_rnn_num_layers: int = 1
    tr_refine_base_channels: int = 16

    ie_prop_hidden_dim: int = 128
    ie_rnn_hidden_dim: int = 256
    ie_rnn_num_layers: int = 1
    ie_map_latent_dim: int = 128
    ie_transformer_dim: int = 256
    ie_transformer_heads: int = 8
    ie_transformer_layers: int = 2
    ie_transformer_ff_dim: int = 512
    explicit_latent_dim: int = 128
    implicit_latent_dim: int = 64

    critic_rnn_type: str = "gru"
    critic_rnn_hidden_dim: int = 256
    critic_rnn_num_layers: int = 1

    tr_rough_loss_weight: float = 1.0
    tr_refined_loss_weight: float = 1.0
    ie_kld_loss_weight: float = 1.0
    ie_base_vel_loss_weight: float = 1.0
    ie_body_map_loss_weight: float = 1.0
    ie_feet_map_loss_weight: float = 1.0
    ie_proprio_loss_weight: float = 1.0

    adasmpl_enabled: bool = True
    adasmpl_initial_probability: float = 1.0
    adasmpl_min_probability: float = 0.0
    adasmpl_max_probability: float = 1.0


@configclass
class StartPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """Configuration for START PPO with auxiliary losses and AdaSmpl."""

    class_name: str = "StartPPO"
    tr_loss_coef: float = 1.0
    ie_loss_coef: float = 1.0

    adasmpl_reward_window: int = 256
    adasmpl_min_episodes: int = 8
    adasmpl_cv_scale: float = 1.0
    # EMA alpha=1.0 matches the paper equation p_smpl = tanh(CV(R)).
    adasmpl_probability_ema_alpha: float = 1.0
    adasmpl_probability_min: float = 0.0
    adasmpl_probability_max: float = 1.0


@configclass
class UnitreeGo2RoughStartPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """START single-stage PPO config for Go2 sparse-foothold locomotion."""

    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    experiment_name = "unitree_go2_rough_start"
    obs_groups = {
        "policy": ["policy", "depth_camera"],
        "critic": ["critic"],
    }

    policy = StartActorCriticCfg(
        init_noise_std=0.1,
        noise_std_type="scalar",
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
    )

    algorithm = StartPpoAlgorithmCfg(
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
