from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class RslRlAMETerrainEncoderModelCfg(RslRlMLPModelCfg):
    class_name: str = "ame:AMETerrainEncoderModel"
    map_scan_dim: tuple[int, int, int] = (33, 21, 3)
    mha_dim: int = 64
    num_heads: int = 16
    cnn_downsample: bool = True
    attach_global: bool = False


@configclass
class UnitreeG1AMERoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    experiment_name = "unitree_g1_ame_rough"
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlAMETerrainEncoderModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlAMETerrainEncoderModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
    )
    algorithm = RslRlPpoAlgorithmCfg(
        value_loss_coef=1.0,
        use_clipped_value_loss=True,
        clip_param=0.2,
        entropy_coef=0.008,
        num_learning_epochs=5,
        num_mini_batches=4,
        learning_rate=1.0e-3,
        schedule="adaptive",
        gamma=0.99,
        lam=0.95,
        desired_kl=0.01,
        max_grad_norm=1.0,
    )
