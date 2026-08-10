from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlDistillationAlgorithmCfg, RslRlDistillationStudentTeacherCfg

from .rsl_rl_distillation_cfg import UnitreeGo2RoughDistillationRunnerCfg


@configclass
class StartDistillationAlgorithmCfg(RslRlDistillationAlgorithmCfg):
    class_name: str = "StartDistillation"
    reconstruction_loss_weight: float = 0.5
    adasmpl_reward_window: int = 256
    adasmpl_min_episodes: int = 8
    adasmpl_cv_scale: float = 1.0
    adasmpl_probability_ema_alpha: float = 0.2
    adasmpl_probability_min: float = 0.05
    adasmpl_probability_max: float = 1.0


@configclass
class UnitreeGo2RoughDepthBackboneRecurrentStudentTeacherCfg(RslRlDistillationStudentTeacherCfg):
    """START-style recurrent depth student-teacher config with terrain reconstruction + AdaSmpl."""

    class_name = "StudentTeacherDepthBackboneRecurrentSTART"
    depth_obs_group: str = "depth_camera"
    depth_image_shape: tuple[int, int] = (58, 87)
    depth_backbone_channels: list[int] = [32, 64]
    depth_backbone_kernels: list[int] = [5, 3]
    depth_backbone_pool_kernel: int = 2
    depth_backbone_image_fc_dim: int = 128
    depth_backbone_latent_dim: int = 32
    depth_recurrent_fusion_dim: int = 128
    rnn_type: str = "gru"
    rnn_hidden_dim: int = 256
    rnn_num_layers: int = 1
    terrain_heightmap_obs_group: str = "terrain_heightmap"
    terrain_reconstruction_hidden_dim: int = 256
    terrain_reconstruction_latent_dim: int = 64
    terrain_recon_rough_loss_weight: float = 1.0
    terrain_recon_refined_loss_weight: float = 1.0
    terrain_adasmpl_enabled: bool = True
    terrain_adasmpl_initial_prob: float = 1.0
    terrain_adasmpl_min_prob: float = 0.05
    terrain_adasmpl_max_prob: float = 1.0
    terrain_adasmpl_use_in_update: bool = True


@configclass
class UnitreeGo2RoughDepthDistillationRecurrentBackboneRunnerCfg(UnitreeGo2RoughDistillationRunnerCfg):
    experiment_name = "unitree_go2_rough_depth_distill_recurrent_backbone"
    teacher_experiment_name = "unitree_go2_rough_teacher"
    obs_groups = {"policy": ["policy", "depth_camera"], "teacher": ["critic"]}
    policy = UnitreeGo2RoughDepthBackboneRecurrentStudentTeacherCfg(
        init_noise_std=0.1,
        noise_std_type="scalar",
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = StartDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=8,
        max_grad_norm=1.0,
    )
