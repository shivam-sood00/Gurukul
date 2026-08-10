from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlDistillationStudentTeacherCfg

from .rsl_rl_distillation_cfg import UnitreeGo2RoughDistillationRunnerCfg


@configclass
class UnitreeGo2RoughDepthBackboneStudentTeacherCfg(RslRlDistillationStudentTeacherCfg):
    """Student-teacher config with a depth CNN backbone in the student path."""

    class_name = "StudentTeacherDepthBackbone"
    depth_obs_group: str = "depth_camera"
    depth_image_shape: tuple[int, int] = (58, 87)
    depth_backbone_channels: list[int] = [32, 64]
    depth_backbone_kernels: list[int] = [5, 3]
    depth_backbone_pool_kernel: int = 2
    depth_backbone_image_fc_dim: int = 128
    depth_backbone_latent_dim: int = 32


@configclass
class UnitreeGo2RoughDepthDistillationRunnerCfg(UnitreeGo2RoughDistillationRunnerCfg):
    experiment_name = "unitree_go2_rough_depth_distill"
    teacher_experiment_name = "unitree_go2_rough_teacher"
    obs_groups = {"policy": ["policy", "depth_camera"], "teacher": ["critic"]}
    policy = UnitreeGo2RoughDepthBackboneStudentTeacherCfg(
        init_noise_std=0.1,
        noise_std_type="scalar",
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[512, 256, 128],
        activation="elu",
    )

    def __post_init__(self):
        self.algorithm.class_name = "CombinedDistillation"
