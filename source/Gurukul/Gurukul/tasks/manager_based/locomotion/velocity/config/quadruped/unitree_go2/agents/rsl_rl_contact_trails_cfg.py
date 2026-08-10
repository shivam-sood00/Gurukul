from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlOnPolicyRunnerCfg,
    RslRlPpoActorCriticRecurrentCfg,
    RslRlPpoAlgorithmCfg,
)


@configclass
class ContactTrailActorCriticCfg(RslRlPpoActorCriticRecurrentCfg):
    """Actor-critic with policy-side contact trail memory."""

    class_name: str = "ContactTrailActorCritic"
    use_gru: bool = True
    cnn_latent_dim: int = 128
    proprio_hidden_dim: int = 128
    use_contact_quality_aux_loss: bool = False
    debug_save_maps: bool = False
    debug_save_interval: int = 500
    log_stats_interval: int = 24
    contact_trail_events_group: str = "contact_trail_events"
    contact_trail_pose_group: str = "contact_trail_pose"
    foot_pos_b_group: str = "foot_pos_b"
    contact_trail_cfg: dict | None = None


@configclass
class ContactTrailPpoAlgorithmCfg(RslRlPpoAlgorithmCfg):
    class_name: str = "ContactTrailPPO"
    contact_quality_loss_coef: float = 0.0


@configclass
class UnitreeGo2RoughContactTrailsPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO runner for Go2 contact trail locomotion."""

    num_steps_per_env = 24
    max_iterations = 10000
    save_interval = 100
    experiment_name = "unitree_go2_rough_contact_trails"
    obs_groups = {
        "policy": ["policy", "contact_trail_events", "contact_trail_pose", "foot_pos_b"],
        "critic": ["critic", "contact_trail_privileged"],
    }

    policy = ContactTrailActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        use_gru=True,
        use_contact_quality_aux_loss=False,
        debug_save_maps=False,
        debug_save_interval=500,
        log_stats_interval=24,
        contact_trail_cfg={
            "num_channels": 8,
            "grid_size": (40, 40),
            "resolution": 0.05,
            "decay": 0.985,
            "write_radius": 1,
            "write_mode": "learned",
            "use_warp": True,
            "write_only_on_contact": True,
            "contact_force_threshold": 1.0,
            "slip_velocity_scale": 0.5,
        },
    )

    algorithm = ContactTrailPpoAlgorithmCfg(
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
        contact_quality_loss_coef=0.0,
    )


@configclass
class UnitreeGo2RoughContactTrailsEngineeredPPORunnerCfg(UnitreeGo2RoughContactTrailsPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_go2_rough_contact_trails_engineered"
        if self.policy.contact_trail_cfg is None:
            self.policy.contact_trail_cfg = {}
        self.policy.contact_trail_cfg["write_mode"] = "engineered"
