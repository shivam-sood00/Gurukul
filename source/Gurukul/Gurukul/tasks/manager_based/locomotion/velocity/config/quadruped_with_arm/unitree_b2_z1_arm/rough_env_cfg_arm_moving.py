"""Rough-terrain locomotion task with a moving Z1 arm."""
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .arm_motion_env_cfg import configure_z1_arm_joint_motion
from .rough_env_cfg import UnitreeB2Z1ArmRoughEnvCfg


@configclass
class UnitreeB2Z1ArmRoughArmMovingEnvCfg(UnitreeB2Z1ArmRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_z1_arm_joint_motion(self)

        if self.__class__.__name__ == "UnitreeB2Z1ArmRoughArmMovingEnvCfg":
            self.disable_zero_weight_rewards()
