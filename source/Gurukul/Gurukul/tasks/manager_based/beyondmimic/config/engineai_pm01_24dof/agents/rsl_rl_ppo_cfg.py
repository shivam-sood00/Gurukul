from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg


@configclass
class EngineAiPm0124DofBeyondMimicFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 2000
    experiment_name = "engineai_pm01_24dof_beyondmimic_flat"
    empirical_normalization = True
    obs_groups = {"actor": ["policy"], "critic": ["critic"]}
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=False,
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


LOW_FREQ_SCALE = 0.5


@configclass
class EngineAiPm0124DofBeyondMimicFlatLowFreqPPORunnerCfg(
    EngineAiPm0124DofBeyondMimicFlatPPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.num_steps_per_env = round(self.num_steps_per_env * LOW_FREQ_SCALE)
        self.algorithm.gamma = self.algorithm.gamma ** (1 / LOW_FREQ_SCALE)
        self.algorithm.lam = self.algorithm.lam ** (1 / LOW_FREQ_SCALE)


@configclass
class EngineAiPm0124DofWalkingBeyondMimicFlatPPORunnerCfg(
    EngineAiPm0124DofBeyondMimicFlatPPORunnerCfg
):
    experiment_name = "engineai_pm01_24dof_walking_beyondmimic_flat"


@configclass
class EngineAiPm01TabletopDeepMimicPPORunnerCfg(EngineAiPm0124DofBeyondMimicFlatPPORunnerCfg):
    experiment_name = "engineai_pm01_tabletop_deepmimic"


@configclass
class EngineAiPm01HeavyPushDeepMimicPPORunnerCfg(EngineAiPm0124DofBeyondMimicFlatPPORunnerCfg):
    experiment_name = "engineai_pm01_heavy_push_deepmimic"
