"""Flat-terrain locomotion task with broad task-space Z1 arm motion."""
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .arm_motion_env_cfg import configure_z1_arm_task_space_motion
from .flat_env_cfg import UnitreeB2Z1ArmFlatEnvCfg


@configclass
class UnitreeB2Z1ArmFlatWideArmMovingEnvCfg(UnitreeB2Z1ArmFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_z1_arm_task_space_motion(self)

        if self.__class__.__name__ == "UnitreeB2Z1ArmFlatWideArmMovingEnvCfg":
            self.disable_zero_weight_rewards()
