# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from ...unitree_go2_airbot_arm.agents.rsl_rl_ppo_cfg import (
    UnitreeGo2AirbotArmFlatPPORunnerCfg,
    UnitreeGo2AirbotArmRoughPPORunnerCfg,
)


@configclass
class UnitreeB2Z1ArmRoughPPORunnerCfg(UnitreeGo2AirbotArmRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_b2_z1_arm_rough"


@configclass
class UnitreeB2Z1ArmFlatPPORunnerCfg(UnitreeGo2AirbotArmFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_b2_z1_arm_flat"


@configclass
class UnitreeB2Z1ArmMovingRoughPPORunnerCfg(UnitreeB2Z1ArmRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_b2_z1_arm_moving_rough"


@configclass
class UnitreeB2Z1ArmMovingFlatPPORunnerCfg(UnitreeB2Z1ArmFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_b2_z1_arm_moving_flat"


@configclass
class UnitreeB2Z1WideArmMovingRoughPPORunnerCfg(UnitreeB2Z1ArmRoughPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_b2_z1_wide_arm_moving_rough"


@configclass
class UnitreeB2Z1WideArmMovingFlatPPORunnerCfg(UnitreeB2Z1ArmFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_b2_z1_wide_arm_moving_flat"


@configclass
class UnitreeB2Z1ReachFlatPPORunnerCfg(UnitreeB2Z1ArmFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 6000
        self.experiment_name = "unitree_b2_z1_reach_flat"


@configclass
class UnitreeB2Z1PushFlatPPORunnerCfg(UnitreeB2Z1ArmFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 10000
        self.experiment_name = "unitree_b2_z1_push_flat"


@configclass
class UnitreeB2Z1RearrangeFlatPPORunnerCfg(UnitreeB2Z1ArmFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 14000
        self.experiment_name = "unitree_b2_z1_rearrange_flat"


@configclass
class UnitreeB2Z1PickThrowFlatPPORunnerCfg(UnitreeB2Z1ArmFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 16000
        self.experiment_name = "unitree_b2_z1_pick_throw_flat"


@configclass
class UnitreeB2Z1BadmintonFlatPPORunnerCfg(UnitreeB2Z1ArmFlatPPORunnerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 18000
        self.experiment_name = "unitree_b2_z1_badminton_flat"
