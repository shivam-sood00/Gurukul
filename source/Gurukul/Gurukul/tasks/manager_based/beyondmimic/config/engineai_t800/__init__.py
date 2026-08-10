import gymnasium as gym

from . import agents


gym.register(
    id="Gurukul-Isaac-BeyondMimic-Flat-EngineAI-T800-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:EngineAiT800BeyondMimicFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:EngineAiT800BeyondMimicFlatPPORunnerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-BeyondMimic-Flat-EngineAI-T800-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_env_cfg:EngineAiT800BeyondMimicFlatWoStateEstimationEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:EngineAiT800BeyondMimicFlatPPORunnerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-BeyondMimic-Flat-EngineAI-T800-Low-Freq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:EngineAiT800BeyondMimicFlatLowFreqEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:EngineAiT800BeyondMimicFlatLowFreqPPORunnerCfg"
        ),
    },
)
