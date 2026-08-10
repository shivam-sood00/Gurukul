# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

gym.register(
    id="Gurukul-Isaac-Velocity-Flat-Unitree-Go2-D1-Arm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2D1ArmFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2D1ArmFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2D1ArmFlatTrainerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-Velocity-Rough-Unitree-Go2-D1-Arm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2D1ArmRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2D1ArmRoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2D1ArmRoughTrainerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-Velocity-Flat-Unitree-Go2-D1-Arm-ArmMoving-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_arm_moving:UnitreeGo2D1ArmFlatArmMovingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2D1ArmMovingFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2D1ArmMovingFlatTrainerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-Velocity-Rough-Unitree-Go2-D1-Arm-ArmMoving-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg_arm_moving:UnitreeGo2D1ArmRoughArmMovingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2D1ArmMovingRoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2D1ArmMovingRoughTrainerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-WBC-Flat-Unitree-Go2-D1-Arm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.whole_body_controller_env_cfg:UnitreeGo2D1WholeBodyControllerFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2D1WbcFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2D1WbcFlatTrainerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-WBC-Flat-Unitree-Go2-D1-Arm-v0-MJLabActionScale",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.whole_body_controller_env_cfg:"
            "UnitreeGo2D1WholeBodyControllerFlatMjlabActionScaleEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2D1WbcFlatMjlabActionScalePPORunnerCfg"
        ),
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2D1WbcFlatTrainerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-WBC-Rough-Unitree-Go2-D1-Arm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.whole_body_controller_env_cfg:UnitreeGo2D1WholeBodyControllerRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2D1WbcRoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2D1WbcRoughTrainerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-LegWBC-AsyncArm-Flat-Unitree-Go2-D1-Arm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.whole_body_controller_env_cfg:UnitreeGo2D1LegWbcAsyncArmFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2D1LegWbcAsyncArmFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2D1WbcFlatTrainerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-LegWBC-AsyncArm-Rough-Unitree-Go2-D1-Arm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.whole_body_controller_env_cfg:UnitreeGo2D1LegWbcAsyncArmRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2D1LegWbcAsyncArmRoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2D1WbcRoughTrainerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-WBC-ApexArm-Flat-Unitree-Go2-D1-Arm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.whole_body_controller_env_cfg:UnitreeGo2D1WholeBodyControllerApexArmFlatEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2D1WbcApexArmFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2D1WbcApexArmFlatTrainerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-WBC-ApexArm-Rough-Unitree-Go2-D1-Arm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.whole_body_controller_env_cfg:UnitreeGo2D1WholeBodyControllerApexArmRoughEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2D1WbcApexArmRoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2D1WbcApexArmRoughTrainerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-Pick-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manipulation_env_cfg:UnitreeGo2D1PickFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2D1PickFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2D1PickFlatTrainerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-Pick-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manipulation_env_cfg:UnitreeGo2D1PickTeacherFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2D1PickTeacherFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2D1PickTeacherFlatTrainerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-PickPlace-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manipulation_env_cfg:UnitreeGo2D1PickPlaceFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2D1PickPlaceFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2D1PickPlaceFlatTrainerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-StationaryPickPlace-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.loco_manipulation_env_cfg:UnitreeGo2D1StationaryPickPlaceFlatEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2D1StationaryPickPlaceFlatPPORunnerCfg"
        ),
        "cusrl_cfg_entry_point": (
            f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2D1StationaryPickPlaceFlatTrainerCfg"
        ),
    },
)

gym.register(
    id="Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-Pick-Wbc-Hierarchical-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.wbc_hierarchical_env_cfg:UnitreeGo2D1PickWbcHierarchicalFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2D1PickWbcHierarchicalFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2D1PickWbcHierarchicalFlatTrainerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-Pick-LegWbcArm-Hierarchical-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.wbc_hierarchical_env_cfg:UnitreeGo2D1PickLegWbcArmHierarchicalFlatEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2D1PickLegWbcArmHierarchicalFlatPPORunnerCfg"
        ),
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2D1PickWbcHierarchicalFlatTrainerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-Pick-LegWbcEe-Hierarchical-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.wbc_hierarchical_env_cfg:UnitreeGo2D1PickLegWbcEeHierarchicalFlatEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2D1PickLegWbcEeHierarchicalFlatPPORunnerCfg"
        ),
        "cusrl_cfg_entry_point": (
            f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2D1PickLegWbcEeTeacherFlatTrainerCfg"
        ),
    },
)

gym.register(
    id="Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-Pick-LegWbcEe-Hierarchical-Fast-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.wbc_hierarchical_env_cfg:"
            "UnitreeGo2D1PickLegWbcEeHierarchicalFastFlatEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "UnitreeGo2D1PickLegWbcEeHierarchicalFastFlatPPORunnerCfg"
        ),
        "cusrl_cfg_entry_point": (
            f"{agents.__name__}.cusrl_ppo_cfg:"
            "UnitreeGo2D1PickLegWbcEeTeacherFastFlatTrainerCfg"
        ),
    },
)
