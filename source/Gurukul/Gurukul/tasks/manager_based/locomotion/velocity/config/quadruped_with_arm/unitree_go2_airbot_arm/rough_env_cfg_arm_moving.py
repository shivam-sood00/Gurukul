# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .arm_ik_env_cfg import configure_realistic_arm_ik_motion
from .rough_env_cfg_fixed_arm import UnitreeGo2AirbotArmFixedRoughEnvCfg


@configclass
class UnitreeGo2AirbotArmRoughArmMovingEnvCfg(UnitreeGo2AirbotArmFixedRoughEnvCfg):
    """Rough-terrain locomotion task with a moving Airbot arm driven by IK."""

    def __post_init__(self):
        super().__post_init__()
        configure_realistic_arm_ik_motion(self)

        if self.__class__.__name__ == "UnitreeGo2AirbotArmRoughArmMovingEnvCfg":
            self.disable_zero_weight_rewards()
