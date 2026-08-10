import gymnasium as gym

from . import (
    agents,
    flat_b2_z1_arm_tracker_env_cfg,
    flat_d1_arm_tracker_env_cfg,
    flat_depth_distill_env_cfg,
    flat_env_cfg,
    flat_tracker_env_cfg,
)

##
# Register Gym environments.
##

gym.register(
    id="Gurukul-Isaac-Go2-APEX-Flat-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2ApexFlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2ApexFlatPPORunnerCfg",
        "rsl_rl_teacher_cfg_entry_point": f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeGo2ApexFlatTeacherPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:UnitreeGo2ApexFlatDistillationRunnerCfg"
        ),
        "rsl_rl_decap_cfg_entry_point": f"{agents.__name__}.rsl_rl_decap_cfg:UnitreeGo2ApexFlatDecAPPPORunnerCfg",
        "rsl_rl_multi_critic_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_multi_critic_cfg:UnitreeGo2ApexFlatMultiCriticRunnerCfg"
        ),
    },
)

gym.register(
    id="Gurukul-Isaac-Go2-APEX-Flat-Depth-Distill-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_depth_distill_env_cfg:UnitreeGo2ApexFlatDepthDistillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2ApexFlatPPORunnerCfg",
        "rsl_rl_teacher_cfg_entry_point": f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeGo2ApexFlatTeacherPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_depth_distillation_cfg:UnitreeGo2ApexFlatDepthDistillationRunnerCfg"
        ),
        "rsl_rl_decap_cfg_entry_point": f"{agents.__name__}.rsl_rl_decap_cfg:UnitreeGo2ApexFlatDecAPPPORunnerCfg",
        "rsl_rl_multi_critic_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_multi_critic_cfg:UnitreeGo2ApexFlatMultiCriticRunnerCfg"
        ),
    },
)

gym.register(
    id="Gurukul-Isaac-Go2-APEX-Flat-Depth-Action-Prior-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_depth_distill_env_cfg:UnitreeGo2ApexFlatDepthDistillEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_decap_cfg:UnitreeGo2ApexFlatDepthActionPriorPPORunnerCfg"
        ),
        "rsl_rl_teacher_cfg_entry_point": f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeGo2ApexFlatTeacherPPORunnerCfg",
        "rsl_rl_decap_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_decap_cfg:UnitreeGo2ApexFlatDepthActionPriorPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gurukul-Isaac-Go2-APEX-Flat-Tracker-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_tracker_env_cfg:UnitreeGo2ApexFlatTrackerEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2ApexFlatTrackerPPORunnerCfg",
        "rsl_rl_teacher_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeGo2ApexFlatPrivilegedTrackerPPORunnerCfg"
        ),
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:UnitreeGo2ApexFlatTrackerDistillationRunnerCfg"
        ),
    },
)

gym.register(
    id="Gurukul-Isaac-Go2-APEX-Flat-Privileged-Tracker-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_tracker_env_cfg:UnitreeGo2ApexFlatPrivilegedTrackerEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeGo2ApexFlatPrivilegedTrackerPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gurukul-Isaac-Go2-APEX-Flat-Tracker-One-Step-Future-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_tracker_env_cfg:UnitreeGo2ApexFlatOneStepFutureTrackerEnvCfg",
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2ApexFlatOneStepFutureTrackerPPORunnerCfg"
        ),
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:UnitreeGo2ApexFlatOneStepFutureTrackerDistillationRunnerCfg"
        ),
    },
)

gym.register(
    id="Gurukul-Isaac-Go2-APEX-Flat-Tracker-One-Step-Future-History-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_tracker_env_cfg:UnitreeGo2ApexFlatOneStepFutureTrackerHistoryEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2ApexFlatOneStepFutureTrackerHistoryPPORunnerCfg"
        ),
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:"
            "UnitreeGo2ApexFlatOneStepFutureTrackerHistoryDistillationRunnerCfg"
        ),
    },
)

gym.register(
    id="Gurukul-Isaac-Go2-D1-Arm-APEX-Flat-Tracker-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_d1_arm_tracker_env_cfg:UnitreeGo2D1ArmApexFlatTrackerEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2D1ArmApexFlatTrackerPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gurukul-Isaac-Go2-D1-Arm-APEX-Distillation-Student-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_d1_arm_tracker_env_cfg:"
            "UnitreeGo2D1ArmApexDistillationStudentEnvCfg"
        ),
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:"
            "UnitreeGo2D1ArmApexDistillationStudentRunnerCfg"
        ),
    },
)

gym.register(
    id="Gurukul-Isaac-Go2-D1-Arm-APEX-Original-DecAP-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_d1_arm_tracker_env_cfg:"
            "UnitreeGo2D1ArmApexOriginalDecapTeacherEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_teacher_cfg:"
            "UnitreeGo2D1ArmApexOriginalDecapTeacherPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gurukul-Isaac-Go2-D1-Arm-APEX-Pick-Stow-Carry-Flat-Tracker-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_d1_arm_tracker_env_cfg:"
            "UnitreeGo2D1ArmApexPickStowCarryFlatTrackerEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "UnitreeGo2D1ArmApexPickStowCarryFlatTrackerPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gurukul-Isaac-Go2-D1-Arm-APEX-Pick-Stow-Carry-Robot-Only-Flat-Tracker-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_d1_arm_tracker_env_cfg:"
            "UnitreeGo2D1ArmApexPickStowCarryRobotOnlyFlatTrackerEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "UnitreeGo2D1ArmApexPickStowCarryRobotOnlyPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gurukul-Isaac-Go2-D1-Arm-APEX-Can-Pick-Carry-Drop-Flat-Tracker-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_d1_arm_tracker_env_cfg:"
            "UnitreeGo2D1ArmApexCanPickCarryDropFlatTrackerEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "UnitreeGo2D1ArmApexCanPickCarryDropFlatTrackerPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gurukul-Isaac-Go2-D1-Arm-APEX-Pick-Stow-Carry-Privileged-Teacher-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_d1_arm_tracker_env_cfg:"
            "UnitreeGo2D1ArmApexPickStowCarryPrivilegedTeacherEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_teacher_cfg:"
            "UnitreeGo2D1ArmApexPickStowCarryPrivilegedTeacherPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gurukul-Isaac-Go2-D1-Arm-APEX-Pick-Stow-Carry-Distillation-Student-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_d1_arm_tracker_env_cfg:"
            "UnitreeGo2D1ArmApexPickStowCarryDistillationStudentEnvCfg"
        ),
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:"
            "UnitreeGo2D1ArmApexPickStowCarryDistillationRunnerCfg"
        ),
    },
)

gym.register(
    id="Gurukul-Isaac-B2-Z1-Arm-APEX-Flat-Tracker-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_b2_z1_arm_tracker_env_cfg:UnitreeB2Z1ArmApexFlatTrackerEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2Z1ArmApexFlatTrackerPPORunnerCfg"
        ),
        "rsl_rl_teacher_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeB2Z1ArmApexFlatPrivilegedTrackerPPORunnerCfg"
        ),
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:UnitreeB2Z1ArmApexFlatTrackerDistillationRunnerCfg"
        ),
    },
)

gym.register(
    id="Gurukul-Isaac-B2-Z1-Arm-APEX-Flat-Tracker-One-Step-Future-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_b2_z1_arm_tracker_env_cfg:UnitreeB2Z1ArmApexFlatOneStepFutureTrackerEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2Z1ArmApexFlatOneStepFutureTrackerPPORunnerCfg"
        ),
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:"
            "UnitreeB2Z1ArmApexFlatOneStepFutureTrackerDistillationRunnerCfg"
        ),
    },
)

gym.register(
    id="Gurukul-Isaac-B2-Z1-Arm-APEX-Flat-Tracker-One-Step-Future-History-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_b2_z1_arm_tracker_env_cfg:"
            "UnitreeB2Z1ArmApexFlatOneStepFutureTrackerHistoryEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeB2Z1ArmApexFlatOneStepFutureTrackerHistoryPPORunnerCfg"
        ),
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:"
            "UnitreeB2Z1ArmApexFlatOneStepFutureTrackerHistoryDistillationRunnerCfg"
        ),
    },
)

gym.register(
    id="Gurukul-Isaac-B2-Z1-Arm-APEX-Flat-Privileged-Tracker-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_b2_z1_arm_tracker_env_cfg:UnitreeB2Z1ArmApexFlatPrivilegedTrackerEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeB2Z1ArmApexFlatPrivilegedTrackerPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gurukul-Isaac-B2-Z1-Arm-APEX-Flat-Fixed-Wrist-Gripper-Tracker-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_b2_z1_arm_tracker_env_cfg:"
            "UnitreeB2Z1ArmApexFlatFixedWristGripperTrackerEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_ppo_cfg:"
            "UnitreeB2Z1ArmApexFlatFixedWristGripperTrackerPPORunnerCfg"
        ),
        "rsl_rl_teacher_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_teacher_cfg:"
            "UnitreeB2Z1ArmApexFlatFixedWristGripperPrivilegedTrackerPPORunnerCfg"
        ),
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:"
            "UnitreeB2Z1ArmApexFlatFixedWristGripperTrackerDistillationRunnerCfg"
        ),
    },
)

gym.register(
    id="Gurukul-Isaac-B2-Z1-Arm-APEX-Flat-Fixed-Wrist-Gripper-Privileged-Tracker-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": (
            f"{__name__}.flat_b2_z1_arm_tracker_env_cfg:"
            "UnitreeB2Z1ArmApexFlatFixedWristGripperPrivilegedTrackerEnvCfg"
        ),
        "rsl_rl_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_teacher_cfg:"
            "UnitreeB2Z1ArmApexFlatFixedWristGripperPrivilegedTrackerPPORunnerCfg"
        ),
    },
)
