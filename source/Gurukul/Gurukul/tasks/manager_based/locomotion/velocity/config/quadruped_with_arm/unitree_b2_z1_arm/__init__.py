# SPDX-License-Identifier: Apache-2.0

import gymnasium as gym

from . import agents

gym.register(
    id="Gurukul-Isaac-Velocity-Flat-Unitree-B2-Z1-Arm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeB2Z1ArmFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2Z1ArmFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeB2Z1ArmFlatTrainerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-Velocity-Rough-Unitree-B2-Z1-Arm-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeB2Z1ArmRoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2Z1ArmRoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeB2Z1ArmRoughTrainerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-Velocity-Flat-Unitree-B2-Z1-Arm-ArmMoving-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_arm_moving:UnitreeB2Z1ArmFlatArmMovingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2Z1ArmMovingFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeB2Z1ArmMovingFlatTrainerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-Velocity-Flat-Unitree-B2-Z1-Arm-ArmMoving-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_teacher_env_cfg:UnitreeB2Z1ArmFlatArmMovingTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeB2Z1ArmMovingFlatTeacherPPORunnerCfg"
        ),
        "rsl_rl_teacher_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeB2Z1ArmMovingFlatTeacherPPORunnerCfg"
        ),
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:UnitreeB2Z1ArmMovingFlatDistillationRunnerCfg"
        ),
    },
)

gym.register(
    id="Gurukul-Isaac-Velocity-Rough-Unitree-B2-Z1-Arm-ArmMoving-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg_arm_moving:UnitreeB2Z1ArmRoughArmMovingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2Z1ArmMovingRoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeB2Z1ArmMovingRoughTrainerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-Velocity-Rough-Unitree-B2-Z1-Arm-ArmMoving-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_teacher_env_cfg:UnitreeB2Z1ArmRoughArmMovingTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeB2Z1ArmMovingRoughTeacherPPORunnerCfg"
        ),
        "rsl_rl_teacher_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeB2Z1ArmMovingRoughTeacherPPORunnerCfg"
        ),
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:UnitreeB2Z1ArmMovingRoughDistillationRunnerCfg"
        ),
    },
)

gym.register(
    id="Gurukul-Isaac-Velocity-Flat-Unitree-B2-Z1-Arm-WideArmMoving-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_arm_moving_wide:UnitreeB2Z1ArmFlatWideArmMovingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2Z1WideArmMovingFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeB2Z1WideArmMovingFlatTrainerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-Velocity-Flat-Unitree-B2-Z1-Arm-WideArmMoving-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_teacher_env_cfg:UnitreeB2Z1ArmFlatWideArmMovingTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeB2Z1WideArmMovingFlatTeacherPPORunnerCfg"
        ),
        "rsl_rl_teacher_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeB2Z1WideArmMovingFlatTeacherPPORunnerCfg"
        ),
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:UnitreeB2Z1WideArmMovingFlatDistillationRunnerCfg"
        ),
    },
)

gym.register(
    id="Gurukul-Isaac-Velocity-Rough-Unitree-B2-Z1-Arm-WideArmMoving-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg_arm_moving_wide:UnitreeB2Z1ArmRoughWideArmMovingEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2Z1WideArmMovingRoughPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeB2Z1WideArmMovingRoughTrainerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-Velocity-Rough-Unitree-B2-Z1-Arm-WideArmMoving-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_teacher_env_cfg:UnitreeB2Z1ArmRoughWideArmMovingTeacherEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeB2Z1WideArmMovingRoughTeacherPPORunnerCfg"
        ),
        "rsl_rl_teacher_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeB2Z1WideArmMovingRoughTeacherPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gurukul-Isaac-LocoManip-Flat-Unitree-B2-Z1-Reach-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manipulation_env_cfg:UnitreeB2Z1ReachFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2Z1ReachFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeB2Z1ReachFlatTrainerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-LocoManip-Flat-Unitree-B2-Z1-Push-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manipulation_env_cfg:UnitreeB2Z1PushFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2Z1PushFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeB2Z1PushFlatTrainerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-LocoManip-Flat-Unitree-B2-Z1-Rearrange-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manipulation_env_cfg:UnitreeB2Z1RearrangeFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2Z1RearrangeFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeB2Z1RearrangeFlatTrainerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-LocoManip-Flat-Unitree-B2-Z1-PickThrow-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manipulation_env_cfg:UnitreeB2Z1PickThrowFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2Z1PickThrowFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeB2Z1PickThrowFlatTrainerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-LocoManip-Flat-Unitree-B2-Z1-Badminton-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.loco_manipulation_env_cfg:UnitreeB2Z1BadmintonFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2Z1BadmintonFlatPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeB2Z1BadmintonFlatTrainerCfg",
    },
)
