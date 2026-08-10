from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import UnitreeGo2ApexFlatPPORunnerCfg


@configclass
class UnitreeGo2ApexFlatDecAPPPORunnerCfg(UnitreeGo2ApexFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.class_name = "DecAPRunner"
        self.experiment_name = "unitree_go2_apex_flat_decap"
        self.teacher_experiment_name = "unitree_go2_apex_flat_teacher"
        self.teacher_load_run = ".*"
        self.teacher_load_checkpoint = "model_.*.pt"
        self.decap_lambda_start = 1.0
        self.decap_lambda_end = 0.0
        self.decap_decay_type = "cosine"

        # Linear/cosine schedule. These are PPO iteration counts.
        self.decap_decay_start_iteration = 0
        self.decap_warmup_iterations = 50
        self.decap_decay_iterations = 3000
        self.decap_decay_end_iteration = None

        # Exponential schedule. Used only when decap_decay_type="exp".
        self.decap_exp_gamma = 0.99
        self.decap_exp_k = 100.0

        self.decap_adaptive_decay = False
        self.decap_adaptive_start_iteration = 10
        self.decap_adaptive_metric_ema_alpha = 0.1
        self.decap_adaptive_pause_drop_ratio = 0.1
        self.decap_adaptive_resume_ratio = 0.98
        self.decap_adaptive_pause_patience = 3
        self.decap_adaptive_resume_patience = 3
        self.obs_groups = {"actor": ["policy"], "critic": ["critic"], "teacher": ["critic"]}


@configclass
class UnitreeGo2ApexFlatDepthActionPriorPPORunnerCfg(UnitreeGo2ApexFlatDecAPPPORunnerCfg):
    """Depth-observation APEX PPO student with teacher actions as action priors."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_go2_apex_flat_depth_action_prior"
        self.teacher_experiment_name = "unitree_go2_apex_flat_teacher"
        self.obs_groups = {"actor": ["policy", "depth_camera"], "critic": ["critic"], "teacher": ["critic"]}
