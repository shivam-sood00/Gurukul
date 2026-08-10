# SPDX-License-Identifier: Apache-2.0

"""PPO runner for the EngineAI PM01 SMP velocity task."""

from isaaclab.utils import configclass

from Gurukul.tasks.manager_based.locomotion.velocity.config.humanoid.engineai_pm01.agents import (
    rsl_rl_ppo_cfg as pm01_ppo,
)


@configclass
class EngineAiPm01SmpFlatPPORunnerCfg(pm01_ppo.EngineAiPm01FlatPPORunnerCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.class_name = "SmpOnPolicyRunner"
        self.experiment_name = "engineai_pm01_smp_flat"
