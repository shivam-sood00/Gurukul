import gymnasium as gym

from . import agents, flat_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Gurukul-Isaac-BeyondMimic-Flat-Unitree-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeG1BeyondMimicFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1BeyondMimicFlatPPORunnerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-BeyondMimic-APEX-Flat-Unitree-G1-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.apex_tracker_env_cfg:UnitreeG1BeyondMimicApexFlatTrackerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1BeyondMimicFlatPPORunnerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-G1-APEX-Flat-Tracker-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.apex_tracker_env_cfg:UnitreeG1BeyondMimicApexFlatTrackerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeG1BeyondMimicFlatPPORunnerCfg",
    },
)
