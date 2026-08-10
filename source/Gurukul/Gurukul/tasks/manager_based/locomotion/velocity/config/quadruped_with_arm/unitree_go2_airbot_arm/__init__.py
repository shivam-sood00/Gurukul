# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

##
# Register Gym environments.
##

gym.register(
    id="Gurukul-Isaac-Velocity-Flat-Unitree-Go2-Airbot-Arm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2AirbotArmFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2AirbotArmFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2AirbotArmFlatTrainerCfg",
    },
)

# IK-arm variants: policy controls Go2 legs while the arm follows smooth end-effector targets.
gym.register(
    id="Gurukul-Isaac-Velocity-Flat-Unitree-Go2-Airbot-Arm-ArmMoving-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_arm_moving:UnitreeGo2AirbotArmFlatArmMovingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2AirbotArmIkFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2AirbotArmIkFlatTrainerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-Velocity-Rough-Unitree-Go2-Airbot-Arm-ArmMoving-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg_arm_moving:UnitreeGo2AirbotArmRoughArmMovingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2AirbotArmIkRoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2AirbotArmIkRoughTrainerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-Velocity-Rough-Unitree-Go2-Airbot-Arm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2AirbotArmRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2AirbotArmRoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2AirbotArmRoughTrainerCfg",
    },
)

##
# Register Fixed Arm (12 DOF) environments - arm joints fixed at zero
##

gym.register(
    id="Gurukul-Isaac-Velocity-Flat-Unitree-Go2-Airbot-Arm-Fixed-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_fixed_arm:UnitreeGo2AirbotArmFixedFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2AirbotArmFixedFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2AirbotArmFixedFlatTrainerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-Velocity-Rough-Unitree-Go2-Airbot-Arm-Fixed-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg_fixed_arm:UnitreeGo2AirbotArmFixedRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2AirbotArmFixedRoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2AirbotArmFixedRoughTrainerCfg",
    },
)
