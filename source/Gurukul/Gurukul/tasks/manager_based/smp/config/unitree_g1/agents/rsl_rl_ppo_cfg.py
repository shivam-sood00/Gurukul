# SPDX-License-Identifier: Apache-2.0

"""PPO runner for the Unitree G1 SMP velocity task."""

from isaaclab.utils import configclass

from Gurukul.tasks.manager_based.locomotion.velocity.config.humanoid.unitree_g1.agents import (
    rsl_rl_ppo_cfg as g1_ppo,
)


@configclass
class UnitreeG1SmpFlatPPORunnerCfg(g1_ppo.UnitreeG1FlatPPORunnerCfg):
    def __post_init__(self) -> None:
        super().__post_init__()
        self.class_name = "SmpOnPolicyRunner"
        self.experiment_name = "unitree_g1_smp_flat"
