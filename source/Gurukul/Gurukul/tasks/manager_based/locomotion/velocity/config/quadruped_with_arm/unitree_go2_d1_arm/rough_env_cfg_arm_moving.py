"""Rough-terrain locomotion task with a moving D1 arm."""
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .arm_motion_env_cfg import configure_d1_arm_joint_motion
from .rough_env_cfg import UnitreeGo2D1ArmRoughEnvCfg


@configclass
class UnitreeGo2D1ArmRoughArmMovingEnvCfg(UnitreeGo2D1ArmRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_d1_arm_joint_motion(self)

        if self.__class__.__name__ == "UnitreeGo2D1ArmRoughArmMovingEnvCfg":
            self.disable_zero_weight_rewards()
