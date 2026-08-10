import gymnasium as gym

from . import agents


gym.register(
    id="Gurukul-Isaac-BeyondMimic-Flat-EngineAI-PM01-24DoF-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:EngineAiPm0124DofBeyondMimicFlatEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:EngineAiPm0124DofBeyondMimicFlatPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gurukul-Isaac-LocoManip-Tabletop-EngineAI-PM01-DeepMimic-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.loco_manipulation_env_cfg:EngineAiPm01TabletopDeepMimicEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:EngineAiPm01TabletopDeepMimicPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gurukul-Isaac-LocoManip-HeavyPush-EngineAI-PM01-DeepMimic-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.loco_manipulation_env_cfg:EngineAiPm01HeavyPushDeepMimicEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:EngineAiPm01HeavyPushDeepMimicPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gurukul-Isaac-LocoManip-Tabletop-EngineAI-PM01-AMP-v0",
    entry_point="Gurukul.tasks.manager_based.beyondmimic.amp_env:BeyondMimicAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manipulation_env_cfg:EngineAiPm01TabletopAmpEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_pm01_loco_manip_amp_cfg.yaml",
    },
)

gym.register(
    id="Gurukul-Isaac-LocoManip-HeavyPush-EngineAI-PM01-AMP-v0",
    entry_point="Gurukul.tasks.manager_based.beyondmimic.amp_env:BeyondMimicAmpEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manipulation_env_cfg:EngineAiPm01HeavyPushAmpEnvCfg",
        "skrl_amp_cfg_entry_point": f"{agents.__name__}:skrl_pm01_loco_manip_amp_cfg.yaml",
    },
)

gym.register(
    id="Gurukul-Isaac-BeyondMimic-Flat-EngineAI-PM01-24DoF-Wo-State-Estimation-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_env_cfg:EngineAiPm0124DofBeyondMimicFlatWoStateEstimationEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:EngineAiPm0124DofBeyondMimicFlatPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gurukul-Isaac-BeyondMimic-Flat-EngineAI-PM01-24DoF-Low-Freq-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_env_cfg:EngineAiPm0124DofBeyondMimicFlatLowFreqEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:EngineAiPm0124DofBeyondMimicFlatLowFreqPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gurukul-Isaac-BeyondMimic-Flat-EngineAI-PM01-24DoF-Walking-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_env_cfg:EngineAiPm0124DofWalkingBeyondMimicFlatEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "EngineAiPm0124DofWalkingBeyondMimicFlatPPORunnerCfg"
        ),
    },
)

gym.register(
    id=(
        "Gurukul-Isaac-BeyondMimic-Flat-EngineAI-"
        "PM01-24DoF-Walking-Wo-State-Estimation-v0"
    ),
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_env_cfg:"
            "EngineAiPm0124DofWalkingBeyondMimicFlatWoStateEstimationEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "EngineAiPm0124DofWalkingBeyondMimicFlatPPORunnerCfg"
        ),
    },
)
