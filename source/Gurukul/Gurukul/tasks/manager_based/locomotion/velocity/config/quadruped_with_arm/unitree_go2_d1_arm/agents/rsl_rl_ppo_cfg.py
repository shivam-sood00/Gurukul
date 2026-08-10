# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from ...unitree_go2_airbot_arm.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2AirbotArmFlatPPORunnerCfg,
    UnitreeGo2AirbotArmRoughPPORunnerCfg,
)


@configclass
class UnitreeGo2D1ArmRoughPPORunnerCfg(UnitreeGo2AirbotArmRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_go2_d1_arm_rough"


@configclass
class UnitreeGo2D1ArmFlatPPORunnerCfg(UnitreeGo2AirbotArmFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_go2_d1_arm_flat"


@configclass
class UnitreeGo2D1ArmMovingRoughPPORunnerCfg(UnitreeGo2D1ArmRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_go2_d1_arm_moving_rough"


@configclass
class UnitreeGo2D1ArmMovingFlatPPORunnerCfg(UnitreeGo2D1ArmFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_go2_d1_arm_moving_flat"


@configclass
class UnitreeGo2D1WbcRoughPPORunnerCfg(UnitreeGo2D1ArmRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 9000
        self.obs_groups = {"actor": ["policy"], "critic": ["critic"]}
        self.experiment_name = "unitree_go2_d1_wbc_rough"


@configclass
class UnitreeGo2D1WbcFlatPPORunnerCfg(UnitreeGo2D1ArmFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 7000
        self.obs_groups = {"actor": ["policy"], "critic": ["critic"]}
        self.experiment_name = "unitree_go2_d1_wbc_flat"


@configclass
class UnitreeGo2D1WbcFlatMjlabActionScalePPORunnerCfg(UnitreeGo2D1WbcFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_go2_d1_wbc_flat_mjlab_action_scale"


@configclass
class UnitreeGo2D1LegWbcAsyncArmRoughPPORunnerCfg(UnitreeGo2D1ArmRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 9000
        self.obs_groups = {"actor": ["policy"], "critic": ["critic"]}
        self.experiment_name = "unitree_go2_d1_leg_wbc_async_arm_rough"


@configclass
class UnitreeGo2D1LegWbcAsyncArmFlatPPORunnerCfg(UnitreeGo2D1ArmFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 7000
        self.obs_groups = {"actor": ["policy"], "critic": ["critic"]}
        self.experiment_name = "unitree_go2_d1_leg_wbc_async_arm_flat"


@configclass
class UnitreeGo2D1WbcApexArmRoughPPORunnerCfg(UnitreeGo2D1WbcRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_go2_d1_wbc_apex_arm_rough"


@configclass
class UnitreeGo2D1WbcApexArmFlatPPORunnerCfg(UnitreeGo2D1WbcFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_go2_d1_wbc_apex_arm_flat"


@configclass
class UnitreeGo2D1PickFlatPPORunnerCfg(UnitreeGo2D1ArmFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.num_steps_per_env = 48
        self.max_iterations = 8000
        self.obs_groups = {"actor": ["policy"], "critic": ["critic"]}
        self.policy.init_noise_std = 0.35
        self.policy.actor_obs_normalization = True
        self.policy.critic_obs_normalization = True
        self.algorithm.entropy_coef = 0.002
        self.algorithm.learning_rate = 5.0e-4
        self.algorithm.gamma = 0.995
        self.experiment_name = "unitree_go2_d1_pick_flat"


@configclass
class UnitreeGo2D1PickTeacherFlatPPORunnerCfg(UnitreeGo2D1PickFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.obs_groups = {"actor": ["teacher"], "critic": ["teacher"]}
        self.experiment_name = "unitree_go2_d1_pick_teacher_flat"


@configclass
class UnitreeGo2D1PickPlaceFlatPPORunnerCfg(UnitreeGo2D1ArmFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.num_steps_per_env = 64
        self.max_iterations = 12000
        self.obs_groups = {"actor": ["policy"], "critic": ["critic"]}
        self.policy.init_noise_std = 0.35
        self.policy.actor_obs_normalization = True
        self.policy.critic_obs_normalization = True
        self.algorithm.entropy_coef = 0.002
        self.algorithm.learning_rate = 5.0e-4
        self.algorithm.gamma = 0.995
        self.experiment_name = "unitree_go2_d1_pick_place_flat"


@configclass
class UnitreeGo2D1StationaryPickPlaceFlatPPORunnerCfg(UnitreeGo2D1PickPlaceFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 6000
        self.experiment_name = "unitree_go2_d1_stationary_pick_place_flat"


@configclass
class UnitreeGo2D1PickWbcHierarchicalFlatPPORunnerCfg(UnitreeGo2D1PickFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.num_steps_per_env = 64
        self.max_iterations = 10000
        self.experiment_name = "unitree_go2_d1_pick_wbc_hierarchical_flat"


@configclass
class UnitreeGo2D1PickLegWbcArmHierarchicalFlatPPORunnerCfg(UnitreeGo2D1PickFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.num_steps_per_env = 64
        self.max_iterations = 10000
        self.experiment_name = "unitree_go2_d1_pick_leg_wbc_arm_hierarchical_flat"


@configclass
class UnitreeGo2D1PickLegWbcEeHierarchicalFlatPPORunnerCfg(UnitreeGo2D1PickFlatPPORunnerCfg):
    use_beta_action_distribution: bool = True

    def __post_init__(self):
        super().__post_init__()

        self.num_steps_per_env = 64
        self.max_iterations = 10000
        self.obs_groups = {"actor": ["teacher"], "critic": ["teacher"]}
        self.clip_actions = 1.0
        # The first bounded-action run contracted before bilateral grasp behavior
        # emerged. Keep enough Beta entropy for close/contact/lift exploration.
        self.algorithm.entropy_coef = 0.002
        # Adaptive KL scheduling raised this task's learning rate by more than an
        # order of magnitude and erased the initially successful grasp behavior.
        self.algorithm.schedule = "fixed"
        self.algorithm.learning_rate = 3.0e-4
        self.experiment_name = "unitree_go2_d1_pick_leg_wbc_ee_teacher_flat"


@configclass
class UnitreeGo2D1PickLegWbcEeHierarchicalFastFlatPPORunnerCfg(
    UnitreeGo2D1PickLegWbcEeHierarchicalFlatPPORunnerCfg
):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_go2_d1_pick_leg_wbc_ee_teacher_fast_flat"
