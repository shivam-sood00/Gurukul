# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from ...unitree_go2_airbot_arm.agents.cusrl_ppo_cfg import (
    UnitreeGo2AirbotArmFlatTrainerCfg,
    UnitreeGo2AirbotArmRoughTrainerCfg,
)


@dataclass
class UnitreeB2Z1ArmRoughTrainerCfg(UnitreeGo2AirbotArmRoughTrainerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_b2_z1_arm_rough"


@dataclass
class UnitreeB2Z1ArmFlatTrainerCfg(UnitreeGo2AirbotArmFlatTrainerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_b2_z1_arm_flat"


@dataclass
class UnitreeB2Z1ArmMovingRoughTrainerCfg(UnitreeB2Z1ArmRoughTrainerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_b2_z1_arm_moving_rough"


@dataclass
class UnitreeB2Z1ArmMovingFlatTrainerCfg(UnitreeB2Z1ArmFlatTrainerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_b2_z1_arm_moving_flat"


@dataclass
class UnitreeB2Z1WideArmMovingRoughTrainerCfg(UnitreeB2Z1ArmRoughTrainerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_b2_z1_wide_arm_moving_rough"


@dataclass
class UnitreeB2Z1WideArmMovingFlatTrainerCfg(UnitreeB2Z1ArmFlatTrainerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_b2_z1_wide_arm_moving_flat"


@dataclass
class UnitreeB2Z1ReachFlatTrainerCfg(UnitreeB2Z1ArmFlatTrainerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 6000
        self.experiment_name = "unitree_b2_z1_reach_flat"


@dataclass
class UnitreeB2Z1PushFlatTrainerCfg(UnitreeB2Z1ArmFlatTrainerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 10000
        self.experiment_name = "unitree_b2_z1_push_flat"


@dataclass
class UnitreeB2Z1RearrangeFlatTrainerCfg(UnitreeB2Z1ArmFlatTrainerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 14000
        self.experiment_name = "unitree_b2_z1_rearrange_flat"


@dataclass
class UnitreeB2Z1PickThrowFlatTrainerCfg(UnitreeB2Z1ArmFlatTrainerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 16000
        self.experiment_name = "unitree_b2_z1_pick_throw_flat"


@dataclass
class UnitreeB2Z1BadmintonFlatTrainerCfg(UnitreeB2Z1ArmFlatTrainerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 18000
        self.experiment_name = "unitree_b2_z1_badminton_flat"
