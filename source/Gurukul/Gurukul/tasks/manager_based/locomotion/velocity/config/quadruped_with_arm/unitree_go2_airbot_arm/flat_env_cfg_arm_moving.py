"""Flat-terrain locomotion task with a moving Airbot arm driven by IK."""
# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .arm_ik_env_cfg import configure_realistic_arm_ik_motion
from .flat_env_cfg_fixed_arm import UnitreeGo2AirbotArmFixedFlatEnvCfg


@configclass
class UnitreeGo2AirbotArmFlatArmMovingEnvCfg(UnitreeGo2AirbotArmFixedFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        configure_realistic_arm_ik_motion(self)

        # If the class is the concrete env, disable zero-weight rewards as usual
        if self.__class__.__name__ == "UnitreeGo2AirbotArmFlatArmMovingEnvCfg":
            self.disable_zero_weight_rewards()
