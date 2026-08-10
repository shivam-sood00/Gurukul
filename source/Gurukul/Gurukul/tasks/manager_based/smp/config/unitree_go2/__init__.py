# SPDX-License-Identifier: Apache-2.0

"""Unitree Go2 SMP task registration."""

import gymnasium as gym

from . import agents

gym.register(
    id="Gurukul-Isaac-SMP-Velocity-Flat-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2SmpFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2SmpFlatPPORunnerCfg",
    },
)
