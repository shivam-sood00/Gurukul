# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Direct MARL environments for heterogeneous multi-robot collaboration."""

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Gurukul-Go2-B2-Collaboration-Direct-v0",
    entry_point=f"{__name__}.go2_b2_collaboration_env:Go2B2CollaborationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_b2_collaboration_env:Go2B2CollaborationEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2B2CollaborationPPORunnerCfg",
    },
)

gym.register(
    id="Gurukul-Go2-B2-Hierarchical-Collaboration-Direct-v0",
    entry_point=f"{__name__}.go2_b2_collaboration_env:Go2B2CollaborationEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.go2_b2_collaboration_env:Go2B2HierarchicalCollaborationEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:Go2B2HierarchicalCollaborationPPORunnerCfg",
    },
)
