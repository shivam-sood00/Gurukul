# SPDX-License-Identifier: Apache-2.0

"""EngineAI PM01 SMP task registration."""

import gymnasium as gym

from . import agents

gym.register(
    id="Gurukul-Isaac-SMP-Velocity-Flat-EngineAI-PM01-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:EngineAiPm01SmpFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:EngineAiPm01SmpFlatPPORunnerCfg",
    },
)
