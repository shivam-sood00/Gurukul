# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


@configclass
class UnitreeGo2AirbotArmRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "unitree_go2_airbot_arm_rough"
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
class UnitreeGo2AirbotArmFlatPPORunnerCfg(UnitreeGo2AirbotArmRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "unitree_go2_airbot_arm_flat"


##
# Fixed Arm Configurations (12 DOF - legs only)
##


@configclass
class UnitreeGo2AirbotArmFixedRoughPPORunnerCfg(RslRlOnPolicyRunnerCfg):
    """PPO config for GO2 with fixed arm on rough terrain - 12 DOF actions."""

    num_steps_per_env = 24
    max_iterations = 20000
    save_interval = 100
    experiment_name = "unitree_go2_airbot_arm_fixed_rough"
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
class UnitreeGo2AirbotArmFixedFlatPPORunnerCfg(UnitreeGo2AirbotArmFixedRoughPPORunnerCfg):
    """PPO config for GO2 with fixed arm on flat terrain - 12 DOF actions."""

    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 5000
        self.experiment_name = "unitree_go2_airbot_arm_fixed_flat"


@configclass
class UnitreeGo2AirbotArmIkRoughPPORunnerCfg(UnitreeGo2AirbotArmFixedRoughPPORunnerCfg):
    """PPO config for Go2 locomotion while the Airbot arm follows IK targets."""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_go2_airbot_arm_ik_rough"


@configclass
class UnitreeGo2AirbotArmIkFlatPPORunnerCfg(UnitreeGo2AirbotArmFixedFlatPPORunnerCfg):
    """PPO config for flat Go2 locomotion while the Airbot arm follows IK targets."""

    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_go2_airbot_arm_ik_flat"
