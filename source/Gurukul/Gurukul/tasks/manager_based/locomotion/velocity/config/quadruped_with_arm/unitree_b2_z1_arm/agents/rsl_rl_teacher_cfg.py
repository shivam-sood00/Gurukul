# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import RslRlPpoActorCriticCfg

from .rsl_rl_ppo_cfg import (
    UnitreeB2Z1ArmMovingFlatPPORunnerCfg,
    UnitreeB2Z1ArmMovingRoughPPORunnerCfg,
    UnitreeB2Z1WideArmMovingFlatPPORunnerCfg,
    UnitreeB2Z1WideArmMovingRoughPPORunnerCfg,
)


@configclass
class UnitreeB2Z1ArmMovingFlatTeacherPPORunnerCfg(UnitreeB2Z1ArmMovingFlatPPORunnerCfg):
    """Privileged flat ArmMoving teacher for B2+Z1."""

    def __post_init__(self):
        super().__post_init__()

        self.obs_groups = {"actor": ["teacher"], "critic": ["teacher"]}
        self.experiment_name = "unitree_b2_z1_arm_moving_flat_teacher"
        self.policy = RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=[1024, 512, 256],
            critic_hidden_dims=[1024, 512, 256],
            activation="elu",
        )


@configclass
class UnitreeB2Z1ArmMovingRoughTeacherPPORunnerCfg(UnitreeB2Z1ArmMovingRoughPPORunnerCfg):
    """Privileged rough ArmMoving teacher for B2+Z1."""

    def __post_init__(self):
        super().__post_init__()

        self.obs_groups = {"actor": ["teacher"], "critic": ["teacher"]}
        self.experiment_name = "unitree_b2_z1_arm_moving_rough_teacher"
        self.policy = RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=[1024, 512, 256],
            critic_hidden_dims=[1024, 512, 256],
            activation="elu",
        )


@configclass
class UnitreeB2Z1WideArmMovingFlatTeacherPPORunnerCfg(UnitreeB2Z1WideArmMovingFlatPPORunnerCfg):
    """Privileged flat wide ArmMoving teacher for B2+Z1."""

    def __post_init__(self):
        super().__post_init__()

        self.obs_groups = {"actor": ["teacher"], "critic": ["teacher"]}
        self.experiment_name = "unitree_b2_z1_wide_arm_moving_flat_teacher"
        self.policy = RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=[1024, 512, 256],
            critic_hidden_dims=[1024, 512, 256],
            activation="elu",
        )


@configclass
class UnitreeB2Z1WideArmMovingRoughTeacherPPORunnerCfg(UnitreeB2Z1WideArmMovingRoughPPORunnerCfg):
    """Privileged rough wide ArmMoving teacher for B2+Z1."""

    def __post_init__(self):
        super().__post_init__()

        self.obs_groups = {"actor": ["teacher"], "critic": ["teacher"]}
        self.experiment_name = "unitree_b2_z1_wide_arm_moving_rough_teacher"
        self.policy = RslRlPpoActorCriticCfg(
            init_noise_std=1.0,
            actor_obs_normalization=False,
            critic_obs_normalization=False,
            actor_hidden_dims=[1024, 512, 256],
            critic_hidden_dims=[1024, 512, 256],
            activation="elu",
        )
