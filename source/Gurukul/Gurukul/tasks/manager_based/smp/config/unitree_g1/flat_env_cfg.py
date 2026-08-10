# SPDX-License-Identifier: Apache-2.0

"""Unitree G1 flat velocity task with a score-matching motion prior."""

from isaaclab.utils import configclass

from Gurukul.tasks.manager_based.locomotion.velocity.config.humanoid.unitree_g1.flat_env_cfg import (
    UnitreeG1FlatEnvCfg,
)
from Gurukul.tasks.manager_based.smp.velocity_env_cfg import install_smp_terms


@configclass
class UnitreeG1SmpFlatEnvCfg(UnitreeG1FlatEnvCfg):
    """Unitree G1 flat velocity tracking with a G1 SMP prior."""

    def __post_init__(self) -> None:
        super().__post_init__()
        install_smp_terms(self, "g1")
        self.disable_zero_weight_rewards()
