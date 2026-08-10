# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

import cusrl
from cusrl.environment.isaaclab import TrainerCfg


@dataclass
class UnitreeGo2AirbotArmRoughTrainerCfg(TrainerCfg):
    max_iterations = 20000
    save_interval = 100
    experiment_name = "unitree_go2_airbot_arm_rough"
    agent_factory = cusrl.ActorCritic.Factory(
        num_steps_per_update=24,
        actor_factory=cusrl.Actor.Factory(
            backbone_factory=cusrl.Mlp.Factory(
                hidden_dims=[512, 256, 128], activation_fn="ELU", ends_with_activation=True
            ),
            distribution_factory=cusrl.NormalDist.Factory(),
        ),
        critic_factory=cusrl.Value.Factory(
            backbone_factory=cusrl.Mlp.Factory(
                hidden_dims=[512, 256, 128], activation_fn="ELU", ends_with_activation=True
            ),
        ),
        optimizer_factory=cusrl.OptimizerFactory("AdamW", defaults={"lr": 1.0e-3}),
        sampler=cusrl.AutoMiniBatchSampler(num_epochs=5, num_mini_batches=4),
        hooks=[
            cusrl.hook.ValueComputation(),
            cusrl.hook.GeneralizedAdvantageEstimation(gamma=0.99, lamda=0.95),
            cusrl.hook.AdvantageNormalization(),
            cusrl.hook.ValueLoss(),
            cusrl.hook.OnPolicyPreparation(),
            cusrl.hook.PpoSurrogateLoss(),
            cusrl.hook.EntropyLoss(weight=0.008),
            cusrl.hook.GradientClipping(max_grad_norm=1.0),
            cusrl.hook.OnPolicyStatistics(sampler=cusrl.AutoMiniBatchSampler()),
            cusrl.hook.AdaptiveLRSchedule(desired_kl_divergence=0.01),
        ],
    )


@dataclass
class UnitreeGo2AirbotArmFlatTrainerCfg(UnitreeGo2AirbotArmRoughTrainerCfg):
    max_iterations = 5000
    experiment_name = "unitree_go2_airbot_arm_flat"


##
# Fixed Arm Configurations (12 DOF - legs only)
##


@dataclass
class UnitreeGo2AirbotArmFixedRoughTrainerCfg(TrainerCfg):
    """CusRL trainer config for GO2 with fixed arm on rough terrain - 12 DOF actions."""

    max_iterations = 20000
    save_interval = 100
    experiment_name = "unitree_go2_airbot_arm_fixed_rough"
    agent_factory = cusrl.ActorCritic.Factory(
        num_steps_per_update=24,
        actor_factory=cusrl.Actor.Factory(
            backbone_factory=cusrl.Mlp.Factory(
                hidden_dims=[512, 256, 128], activation_fn="ELU", ends_with_activation=True
            ),
            distribution_factory=cusrl.NormalDist.Factory(),
        ),
        critic_factory=cusrl.Value.Factory(
            backbone_factory=cusrl.Mlp.Factory(
                hidden_dims=[512, 256, 128], activation_fn="ELU", ends_with_activation=True
            ),
        ),
        optimizer_factory=cusrl.OptimizerFactory("AdamW", defaults={"lr": 1.0e-3}),
        sampler=cusrl.AutoMiniBatchSampler(num_epochs=5, num_mini_batches=4),
        hooks=[
            cusrl.hook.ValueComputation(),
            cusrl.hook.GeneralizedAdvantageEstimation(gamma=0.99, lamda=0.95),
            cusrl.hook.AdvantageNormalization(),
            cusrl.hook.ValueLoss(),
            cusrl.hook.OnPolicyPreparation(),
            cusrl.hook.PpoSurrogateLoss(),
            cusrl.hook.EntropyLoss(weight=0.008),
            cusrl.hook.GradientClipping(max_grad_norm=1.0),
            cusrl.hook.OnPolicyStatistics(sampler=cusrl.AutoMiniBatchSampler()),
            cusrl.hook.AdaptiveLRSchedule(desired_kl_divergence=0.01),
        ],
    )


@dataclass
class UnitreeGo2AirbotArmFixedFlatTrainerCfg(UnitreeGo2AirbotArmFixedRoughTrainerCfg):
    """CusRL trainer config for GO2 with fixed arm on flat terrain - 12 DOF actions."""

    max_iterations = 5000
    experiment_name = "unitree_go2_airbot_arm_fixed_flat"


@dataclass
class UnitreeGo2AirbotArmIkRoughTrainerCfg(UnitreeGo2AirbotArmFixedRoughTrainerCfg):
    """CusRL config for Go2 locomotion while the Airbot arm follows IK targets."""

    experiment_name = "unitree_go2_airbot_arm_ik_rough"


@dataclass
class UnitreeGo2AirbotArmIkFlatTrainerCfg(UnitreeGo2AirbotArmFixedFlatTrainerCfg):
    """CusRL config for flat Go2 locomotion while the Airbot arm follows IK targets."""

    experiment_name = "unitree_go2_airbot_arm_ik_flat"
