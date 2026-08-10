# SPDX-License-Identifier: Apache-2.0

"""Unitree Go2 flat velocity task with a score-matching motion prior."""

from isaaclab.utils import configclass

from Gurukul.tasks.manager_based.locomotion.velocity.config.quadruped.unitree_go2.flat_env_cfg import (
    UnitreeGo2FlatEnvCfg,
)
from Gurukul.tasks.manager_based.smp.velocity_env_cfg import install_smp_terms


@configclass
class UnitreeGo2SmpFlatEnvCfg(UnitreeGo2FlatEnvCfg):
    """Unitree Go2 flat velocity tracking with a Go2 SMP prior."""

    def __post_init__(self) -> None:
        super().__post_init__()
        install_smp_terms(self, "go2")
        self.disable_zero_weight_rewards()
