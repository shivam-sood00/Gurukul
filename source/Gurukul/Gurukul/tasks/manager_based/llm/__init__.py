"""Language-conditioned robot interaction tasks."""

import gymnasium as gym

gym.register(
    id="Gurukul-Isaac-LLM-Go2-D1-Press-Switch-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_d1_switch_env_cfg:Go2D1PressSwitchLlmEnvCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-LLM-Go2-D1-Open-Door-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_d1_service_env_cfg:Go2D1OpenDoorLlmEnvCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-LLM-Go2-D1-Pick-Place-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.go2_d1_service_env_cfg:Go2D1PickPlaceLlmEnvCfg",
    },
)
