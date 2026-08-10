from isaaclab.utils import configclass

from .rsl_rl_distillation_cfg import UnitreeGo2ApexFlatDistillationRunnerCfg


@configclass
class UnitreeGo2ApexFlatDepthDistillationRunnerCfg(UnitreeGo2ApexFlatDistillationRunnerCfg):
    experiment_name = "unitree_go2_apex_flat_depth_distill"
    teacher_experiment_name = "unitree_go2_apex_flat_teacher"
    obs_groups = {"student": ["policy", "depth_camera"], "teacher": ["critic"]}
