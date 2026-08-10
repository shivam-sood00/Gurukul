# SPDX-License-Identifier: Apache-2.0

from dataclasses import dataclass

from ...unitree_go2_airbot_arm.agents.cusrl_ppo_cfg import (
    UnitreeGo2AirbotArmFlatTrainerCfg,
    UnitreeGo2AirbotArmRoughTrainerCfg,
)


@dataclass
class UnitreeGo2D1ArmRoughTrainerCfg(UnitreeGo2AirbotArmRoughTrainerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_go2_d1_arm_rough"


@dataclass
class UnitreeGo2D1ArmFlatTrainerCfg(UnitreeGo2AirbotArmFlatTrainerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_go2_d1_arm_flat"


@dataclass
class UnitreeGo2D1ArmMovingRoughTrainerCfg(UnitreeGo2D1ArmRoughTrainerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_go2_d1_arm_moving_rough"


@dataclass
class UnitreeGo2D1ArmMovingFlatTrainerCfg(UnitreeGo2D1ArmFlatTrainerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_go2_d1_arm_moving_flat"


@dataclass
class UnitreeGo2D1WbcRoughTrainerCfg(UnitreeGo2D1ArmRoughTrainerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 9000
        self.experiment_name = "unitree_go2_d1_wbc_rough"


@dataclass
class UnitreeGo2D1WbcFlatTrainerCfg(UnitreeGo2D1ArmFlatTrainerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 7000
        self.experiment_name = "unitree_go2_d1_wbc_flat"


@dataclass
class UnitreeGo2D1WbcApexArmRoughTrainerCfg(UnitreeGo2D1WbcRoughTrainerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_go2_d1_wbc_apex_arm_rough"


@dataclass
class UnitreeGo2D1WbcApexArmFlatTrainerCfg(UnitreeGo2D1WbcFlatTrainerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_go2_d1_wbc_apex_arm_flat"


@dataclass
class UnitreeGo2D1PickFlatTrainerCfg(UnitreeGo2D1ArmFlatTrainerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 8000
        self.experiment_name = "unitree_go2_d1_pick_flat"


@dataclass
class UnitreeGo2D1PickTeacherFlatTrainerCfg(UnitreeGo2D1PickFlatTrainerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.obs_groups = {"actor": ["teacher"], "critic": ["teacher"]}
        self.experiment_name = "unitree_go2_d1_pick_teacher_flat"


@dataclass
class UnitreeGo2D1PickPlaceFlatTrainerCfg(UnitreeGo2D1ArmFlatTrainerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 12000
        self.experiment_name = "unitree_go2_d1_pick_place_flat"


@dataclass
class UnitreeGo2D1StationaryPickPlaceFlatTrainerCfg(UnitreeGo2D1PickPlaceFlatTrainerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 6000
        self.experiment_name = "unitree_go2_d1_stationary_pick_place_flat"


@dataclass
class UnitreeGo2D1PickWbcHierarchicalFlatTrainerCfg(UnitreeGo2D1PickFlatTrainerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.max_iterations = 10000
        self.experiment_name = "unitree_go2_d1_pick_wbc_hierarchical_flat"


@dataclass
class UnitreeGo2D1PickLegWbcEeTeacherFlatTrainerCfg(UnitreeGo2D1PickWbcHierarchicalFlatTrainerCfg):
    def __post_init__(self):
        super().__post_init__()

        self.obs_groups = {"actor": ["teacher"], "critic": ["teacher"]}
        self.experiment_name = "unitree_go2_d1_pick_leg_wbc_ee_teacher_flat"


@dataclass
class UnitreeGo2D1PickLegWbcEeTeacherFastFlatTrainerCfg(
    UnitreeGo2D1PickLegWbcEeTeacherFlatTrainerCfg
):
    def __post_init__(self):
        super().__post_init__()

        self.experiment_name = "unitree_go2_d1_pick_leg_wbc_ee_teacher_fast_flat"
