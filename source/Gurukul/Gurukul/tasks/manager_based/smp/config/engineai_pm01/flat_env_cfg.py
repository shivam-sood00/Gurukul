# SPDX-License-Identifier: Apache-2.0

"""EngineAI PM01 flat velocity task with a score-matching motion prior."""

from isaaclab.utils import configclass

from Gurukul.tasks.manager_based.locomotion.velocity.config.humanoid.engineai_pm01.flat_env_cfg import (
    EngineAiPm01FlatEnvCfg,
)
from Gurukul.tasks.manager_based.smp.velocity_env_cfg import install_smp_terms


@configclass
class EngineAiPm01SmpFlatEnvCfg(EngineAiPm01FlatEnvCfg):
    """EngineAI PM01 flat velocity tracking with a PM01 SMP prior."""

    def __post_init__(self) -> None:
        super().__post_init__()
        install_smp_terms(self, "pm01")
        self.disable_zero_weight_rewards()
