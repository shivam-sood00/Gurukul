from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class UnitreeGo2ApexFlatPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 500
    experiment_name = "unitree_go2_apex_flat"
    policy = RslRlPpoActorCriticCfg(
        init_noise_std=1.0,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlPpoAlgorithmCfg(
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


@configclass
class UnitreeGo2ApexFlatTrackerPPORunnerCfg(UnitreeGo2ApexFlatPPORunnerCfg):
    """PPO config for the reference-conditioned Go2 APEX tracker."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_go2_apex_flat_tracker"
        self.clip_actions = None


@configclass
class UnitreeGo2ApexFlatOneStepFutureTrackerPPORunnerCfg(UnitreeGo2ApexFlatPPORunnerCfg):
    """PPO config for the one-step-future Go2 APEX tracker."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_go2_apex_flat_tracker_one_step_future"
        self.clip_actions = None


@configclass
class UnitreeGo2ApexFlatOneStepFutureTrackerHistoryPPORunnerCfg(UnitreeGo2ApexFlatOneStepFutureTrackerPPORunnerCfg):
    """PPO config for the history-augmented one-step-future Go2 APEX tracker."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_go2_apex_flat_tracker_one_step_future_history"
        self.policy.actor_hidden_dims = [1024, 512, 256]
        self.policy.critic_hidden_dims = [1024, 512, 256]


@configclass
class UnitreeGo2D1ArmApexFlatTrackerPPORunnerCfg(UnitreeGo2ApexFlatPPORunnerCfg):
    """PPO config for the reference-conditioned Go2+D1 APEX tracker."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_go2_d1_arm_apex_flat_tracker"
        self.obs_groups = {"policy": ["policy"], "critic": ["critic"]}
        # The walk-then-wave clip needs normalized targets up to +5.21 on D1 J2
        # and -4.68 on J3 with the 0.25-rad action scale. A +/-4 runner clip made
        # those poses unreachable and let entropy inflate only the clipped arm
        # dimensions. Keep a bounded margin around the complete reference range.
        self.clip_actions = 6.0


@configclass
class UnitreeGo2D1ArmApexPickStowCarryFlatTrackerPPORunnerCfg(
    UnitreeGo2D1ArmApexFlatTrackerPPORunnerCfg
):
    """PPO config for the dedicated Go2+D1 pick-stow-carry tracker."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_go2_d1_arm_apex_pick_stow_carry"
        # The pick task's dedicated J2/J3 scales put its full demonstrated
        # range inside this bound (J2 +5.85, J3 -5.61). The binary gripper is
        # thresholded before its 0/33 mm physical target and does not need a
        # large continuous-action range.
        self.clip_actions = 6.6


@configclass
class UnitreeGo2D1ArmApexPickStowCarryRobotOnlyPPORunnerCfg(UnitreeGo2D1ArmApexFlatTrackerPPORunnerCfg):
    """PPO config for object-free imitation of the pick-stow-carry robot trajectory."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_go2_d1_arm_apex_pick_stow_carry_robot_only"
        self.clip_actions = 6.6


@configclass
class UnitreeGo2D1ArmApexCanPickCarryDropFlatTrackerPPORunnerCfg(
    UnitreeGo2D1ArmApexPickStowCarryFlatTrackerPPORunnerCfg
):
    """PPO config for the Go2+D1 top-down can pick-carry-drop tracker."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_go2_d1_arm_apex_can_pick_carry_drop"


@configclass
class UnitreeB2Z1ArmApexFlatTrackerPPORunnerCfg(UnitreeGo2ApexFlatPPORunnerCfg):
    """PPO config for the reference-conditioned B2+Z1 APEX tracker."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_b2_z1_arm_apex_flat_tracker"
        self.clip_actions = None


@configclass
class UnitreeB2Z1ArmApexFlatFixedWristGripperTrackerPPORunnerCfg(UnitreeB2Z1ArmApexFlatTrackerPPORunnerCfg):
    """PPO config for B2+Z1 tracking with a fixed wrist and commanded gripper."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_b2_z1_arm_apex_flat_fixed_wrist_gripper_tracker"


@configclass
class UnitreeB2Z1ArmApexFlatOneStepFutureTrackerPPORunnerCfg(UnitreeB2Z1ArmApexFlatTrackerPPORunnerCfg):
    """PPO config for the one-step-future B2+Z1 APEX tracker."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_b2_z1_arm_apex_flat_tracker_one_step_future"


@configclass
class UnitreeB2Z1ArmApexFlatOneStepFutureTrackerHistoryPPORunnerCfg(
    UnitreeB2Z1ArmApexFlatOneStepFutureTrackerPPORunnerCfg
):
    """PPO config for the history-augmented one-step-future B2+Z1 APEX tracker."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_b2_z1_arm_apex_flat_tracker_one_step_future_history"
        self.policy.actor_hidden_dims = [1024, 512, 256]
        self.policy.critic_hidden_dims = [1024, 512, 256]
