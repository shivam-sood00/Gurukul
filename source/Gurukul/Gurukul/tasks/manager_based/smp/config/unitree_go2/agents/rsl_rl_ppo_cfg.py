# SPDX-License-Identifier: Apache-2.0

"""PPO runner for the Unitree Go2 SMP velocity task."""

from isaaclab.utils import configclass

from Gurukul.tasks.manager_based.locomotion.velocity.config.quadruped.unitree_go2.agents import (
    rsl_rl_ppo_cfg as go2_ppo,
)


@configclass
class UnitreeGo2SmpFlatPPORunnerCfg(go2_ppo.UnitreeGo2FlatPPORunnerCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.class_name = "SmpOnPolicyRunner"
        self.experiment_name = "unitree_go2_smp_flat"
