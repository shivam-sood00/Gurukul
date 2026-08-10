from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class CTSActorCriticCfg(RslRlPpoActorCriticCfg):
    """Policy config for concurrent teacher-student rough locomotion."""

    class_name: str = "CTSActorCritic"
    privileged_encoder_hidden_dims: list[int] = [512, 256]
    history_encoder_hidden_dims: list[int] = [512, 256]
    latent_dim: int = 32
    latent_norm: str | None = "l2"
    student_history_group: str = "student_history"
    teacher_obs_group: str = "teacher"
    role_obs_group: str = "cts_role"


@configclass
class CTSPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    """PPO config for concurrent teacher-student rough locomotion."""

    class_name: str = "CTSPPO"
    role_obs_group: str = "cts_role"
    actor_hidden_dims: list[int] = [512, 256, 128]
    critic_hidden_dims: list[int] = [512, 256, 128]
    privileged_encoder_hidden_dims: list[int] = [512, 256]
    history_encoder_hidden_dims: list[int] = [512, 256]
    latent_dim: int = 32
    latent_norm: str | None = "l2"
    activation: str = "elu"
    init_noise_std: float = 1.0
    noise_std_type: str = "scalar"
    actor_obs_normalization: bool = False
    critic_obs_normalization: bool = False
    student_history_group: str = "student_history"
    teacher_obs_group: str = "teacher"
    encoder_learning_rate: float = 1.0e-3
    num_encoder_epochs: int = 5
    teacher_loss_coef: float = 1.0
    student_loss_coef: float = 1.0
    encoder_loss_coef: float = 1.0


@configclass
class UnitreeGo2RoughCTSRunnerCfg(RslRlOnPolicyRunnerCfg):
    """CTS runner for Go2 rough velocity locomotion.

    Follows CTS §III-B: first 75% of environment slots use the teacher path and the remaining slots use the student
    path. The teacher uses the existing privileged full-state rough group; the student uses deployable policy
    observations plus a five-observation history.
    """

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 500
    experiment_name = "unitree_go2_rough_cts"
    obs_groups = {
        "policy": ["policy"],
        "teacher": ["teacher_full"],
        "student_history": ["student_history"],
        "critic": ["teacher_full"],
        "cts_role": ["cts_role"],
    }
    policy = CTSActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = CTSPpoAlgorithmCfg(
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
