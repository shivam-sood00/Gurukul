"""Rough B2+Z1 ArmMoving teacher task with privileged arm-motion observations."""
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .flat_teacher_env_cfg import (
    configure_b2_z1_arm_moving_teacher_observations,
    configure_b2_z1_wide_arm_moving_teacher_observations,
)
from .rough_env_cfg_arm_moving import UnitreeB2Z1ArmRoughArmMovingEnvCfg
from .rough_env_cfg_arm_moving_wide import UnitreeB2Z1ArmRoughWideArmMovingEnvCfg


@configclass
class UnitreeB2Z1ArmRoughArmMovingTeacherEnvCfg(UnitreeB2Z1ArmRoughArmMovingEnvCfg):
    """Rough-terrain teacher variant whose actor sees privileged scripted-arm state."""

    def __post_init__(self):
        super().__post_init__()
        configure_b2_z1_arm_moving_teacher_observations(self)

        if self.__class__.__name__ == "UnitreeB2Z1ArmRoughArmMovingTeacherEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class UnitreeB2Z1ArmRoughWideArmMovingTeacherEnvCfg(UnitreeB2Z1ArmRoughWideArmMovingEnvCfg):
    """Rough-terrain teacher variant whose actor sees privileged task-space arm target state."""

    def __post_init__(self):
        super().__post_init__()
        configure_b2_z1_wide_arm_moving_teacher_observations(self)

        if self.__class__.__name__ == "UnitreeB2Z1ArmRoughWideArmMovingTeacherEnvCfg":
            self.disable_zero_weight_rewards()
