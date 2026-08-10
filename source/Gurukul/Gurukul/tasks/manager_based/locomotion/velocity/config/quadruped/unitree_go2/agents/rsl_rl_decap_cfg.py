from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from .rsl_rl_ppo_cfg import UnitreeGo2FlatPPORunnerCfg, UnitreeGo2RoughPPORunnerCfg


@configclass
class UnitreeGo2RoughDecAPPPORunnerCfg(UnitreeGo2RoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.class_name = "DecAPRunner"
        self.experiment_name = "unitree_go2_rough_decap"
        self.teacher_experiment_name = "unitree_go2_rough_teacher"
        self.teacher_load_run = ".*"
        self.teacher_load_checkpoint = "model_.*.pt"
        self.decap_lambda_start = 1.0
        self.decap_lambda_end = 0.0
        self.decap_decay_type = "linear"
        self.decap_decay_start_iteration = 0
        self.decap_warmup_iterations = 50
        self.decap_decay_end_iteration = 5000
        self.decap_adaptive_decay = True
        self.decap_adaptive_start_iteration = 10
        self.decap_adaptive_metric_ema_alpha = 0.1
        self.decap_adaptive_pause_drop_ratio = 0.1
        self.decap_adaptive_resume_ratio = 0.98
        self.decap_adaptive_pause_patience = 3
        self.decap_adaptive_resume_patience = 3
        self.obs_groups = {"policy": ["policy"], "critic": ["critic"], "teacher": ["critic"]}


@configclass
class UnitreeGo2FlatDecAPPPORunnerCfg(UnitreeGo2FlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.class_name = "DecAPRunner"
        self.experiment_name = "unitree_go2_flat_decap"
        self.teacher_experiment_name = "unitree_go2_flat_teacher"
        self.teacher_load_run = ".*"
        self.teacher_load_checkpoint = "model_.*.pt"
        self.decap_lambda_start = 1.0
        self.decap_lambda_end = 0.0
        self.decap_decay_type = "linear"
        self.decap_decay_start_iteration = 0
        self.decap_warmup_iterations = 50
        self.decap_decay_end_iteration = 1500
        self.decap_adaptive_decay = True
        self.decap_adaptive_start_iteration = 10
        self.decap_adaptive_metric_ema_alpha = 0.1
        self.decap_adaptive_pause_drop_ratio = 0.1
        self.decap_adaptive_resume_ratio = 0.98
        self.decap_adaptive_pause_patience = 3
        self.decap_adaptive_resume_patience = 3
        self.obs_groups = {"policy": ["policy"], "critic": ["critic"], "teacher": ["critic"]}


@configclass
class UnitreeGo2DepthBackboneActorModelCfg(RslRlMLPModelCfg):
    """Depth-CNN actor model config for the depth student PPO baseline."""

    class_name: str = "Gurukul.tasks.manager_based.locomotion.velocity.depth_student_teacher:DepthBackboneActorModel"
    depth_obs_group: str = "depth_camera"
    depth_image_shape: tuple[int, int] = (58, 87)
    depth_backbone_channels: list[int] = [32, 64]
    depth_backbone_kernels: list[int] = [5, 3]
    depth_backbone_pool_kernel: int = 2
    depth_backbone_image_fc_dim: int = 128
    depth_backbone_latent_dim: int = 32


@configclass
class UnitreeGo2RoughDepthActionPriorPPORunnerCfg(UnitreeGo2RoughDecAPPPORunnerCfg):
    """Depth student trained with PPO, teacher reward shaping, and teacher action priors."""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_go2_rough_depth_action_prior"
        self.teacher_experiment_name = "unitree_go2_rough_teacher"
        self.obs_groups = {
            "actor": ["policy", "depth_camera"],
            "critic": ["critic"],
            "teacher": ["critic"],
        }
        self.actor = UnitreeGo2DepthBackboneActorModelCfg(
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=False,
            stochastic=True,
            init_noise_std=1.0,
            noise_std_type="scalar",
            state_dependent_std=False,
        )
        self.critic = RslRlMLPModelCfg(
            class_name="MLPModel",
            hidden_dims=[512, 256, 128],
            activation="elu",
            obs_normalization=False,
            stochastic=False,
        )
        self.teacher_actor = {
            "class_name": "MLPModel",
            "hidden_dims": [512, 256, 128],
            "activation": "elu",
            "obs_normalization": False,
            "distribution_cfg": {
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        }

        self.decap_teacher_deterministic = True
        self.decap_action_reward_weight = 0.05
        self.decap_action_reward_sigma = 0.25
        self.decap_action_reward_mode = "exp_mse"
        self.decap_action_reward_use_mean = True
        self.decap_action_reward_scale_with_lambda = False
