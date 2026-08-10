from importlib.util import find_spec

import gymnasium as gym

from . import agents, rough_depth_distill_env_cfg

##
# Register Gym environments.
##

gym.register(
    id="Gurukul-Isaac-Velocity-Rough-Unitree-Go2-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg_v1:UnitreeGo2RoughEnvCfgV1",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg",
        "rsl_rl_teacher_cfg_entry_point": f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeGo2RoughTeacherPPORunnerCfg",
        "rsl_rl_real_teacher_cfg_entry_point": f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeGo2RoughRealTeacherPPORunnerCfg",
        "rsl_rl_real_teacher_pretrained_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeGo2RoughRealTeacherPretrainedPPORunnerCfg"
        ),
        "rsl_rl_real_teacher_frozen_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeGo2RoughRealTeacherFrozenPPORunnerCfg"
        ),
        "rsl_rl_full_teacher_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeGo2RoughFullTeacherPPORunnerCfg"
        ),
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:UnitreeGo2RoughDistillationRunnerCfg"
        ),
        "rsl_rl_distillation_recurrent_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_recurrent_cfg:UnitreeGo2RoughDistillationRecurrentRunnerCfg"
        ),
        "rsl_rl_cts_cfg_entry_point": f"{agents.__name__}.rsl_rl_cts_cfg:UnitreeGo2RoughCTSRunnerCfg",
        "rsl_rl_decap_cfg_entry_point": f"{agents.__name__}.rsl_rl_decap_cfg:UnitreeGo2RoughDecAPPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2RoughTrainerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-Velocity-Rough-Unitree-Go2-AME-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.ame_env_cfg:UnitreeGo2AMERoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ame_cfg:UnitreeGo2AMERoughPPORunnerCfg",
    },
)

if find_spec("Gurukul.assets.lidar") is not None:
    gym.register(
        id="Gurukul-Isaac-Velocity-Rough-Unitree-Go2-Lidar-v0",
        entry_point="isaaclab.envs:ManagerBasedRLEnv",
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.rough_lidar_env_cfg:UnitreeGo2RoughLidarEnvCfg",
            "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RoughLidarPointNetPPORunnerCfg",
        },
    )

gym.register(
    id="Gurukul-Isaac-Velocity-Flat-Unitree-Go2-v1",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg_v1:UnitreeGo2FlatEnvCfgV1",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
        "rsl_rl_teacher_cfg_entry_point": f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeGo2FlatTeacherPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:UnitreeGo2FlatDistillationRunnerCfg"
        ),
        "rsl_rl_distillation_recurrent_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_recurrent_cfg:UnitreeGo2FlatDistillationRecurrentRunnerCfg"
        ),
        "rsl_rl_decap_cfg_entry_point": f"{agents.__name__}.rsl_rl_decap_cfg:UnitreeGo2FlatDecAPPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2FlatTrainerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-Velocity-Flat-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2FlatEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
        "rsl_rl_teacher_cfg_entry_point": f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeGo2FlatTeacherPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:UnitreeGo2FlatDistillationRunnerCfg"
        ),
        "rsl_rl_distillation_recurrent_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_recurrent_cfg:UnitreeGo2FlatDistillationRecurrentRunnerCfg"
        ),
        "rsl_rl_decap_cfg_entry_point": f"{agents.__name__}.rsl_rl_decap_cfg:UnitreeGo2FlatDecAPPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2FlatTrainerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-Velocity-Flat-Unitree-Go2-v0-MJLabActionScale",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.flat_env_cfg:UnitreeGo2FlatMjlabActionScaleEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2FlatPPORunnerCfg",
        "rsl_rl_teacher_cfg_entry_point": f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeGo2FlatTeacherPPORunnerCfg",
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:UnitreeGo2FlatDistillationRunnerCfg"
        ),
        "rsl_rl_distillation_recurrent_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_recurrent_cfg:UnitreeGo2FlatDistillationRecurrentRunnerCfg"
        ),
        "rsl_rl_decap_cfg_entry_point": f"{agents.__name__}.rsl_rl_decap_cfg:UnitreeGo2FlatDecAPPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2FlatTrainerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-Velocity-Rough-Unitree-Go2-Depth-Distill-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_depth_distill_env_cfg:UnitreeGo2RoughDepthDistillEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg",
        "rsl_rl_teacher_cfg_entry_point": f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeGo2RoughTeacherPPORunnerCfg",
        "rsl_rl_real_teacher_cfg_entry_point": f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeGo2RoughRealTeacherPPORunnerCfg",
        "rsl_rl_real_teacher_pretrained_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeGo2RoughRealTeacherPretrainedPPORunnerCfg"
        ),
        "rsl_rl_real_teacher_frozen_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeGo2RoughRealTeacherFrozenPPORunnerCfg"
        ),
        "rsl_rl_full_teacher_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeGo2RoughFullTeacherPPORunnerCfg"
        ),
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_depth_distillation_cfg:UnitreeGo2RoughDepthDistillationRunnerCfg"
        ),
        "rsl_rl_depth_recurrent_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_depth_recurrent_distillation_cfg:UnitreeGo2RoughDepthDistillationRecurrentBackboneRunnerCfg"
        ),
        "rsl_rl_distillation_recurrent_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_recurrent_cfg:UnitreeGo2RoughDistillationRecurrentRunnerCfg"
        ),
        "rsl_rl_decap_cfg_entry_point": f"{agents.__name__}.rsl_rl_decap_cfg:UnitreeGo2RoughDecAPPPORunnerCfg",
        "rsl_rl_depth_action_prior_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_decap_cfg:UnitreeGo2RoughDepthActionPriorPPORunnerCfg"
        ),
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2RoughTrainerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-Velocity-Rough-Unitree-Go2-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2RoughEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg",
        "rsl_rl_teacher_cfg_entry_point": f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeGo2RoughTeacherPPORunnerCfg",
        "rsl_rl_real_teacher_cfg_entry_point": f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeGo2RoughRealTeacherPPORunnerCfg",
        "rsl_rl_real_teacher_pretrained_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeGo2RoughRealTeacherPretrainedPPORunnerCfg"
        ),
        "rsl_rl_real_teacher_frozen_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeGo2RoughRealTeacherFrozenPPORunnerCfg"
        ),
        "rsl_rl_full_teacher_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeGo2RoughFullTeacherPPORunnerCfg"
        ),
        "rsl_rl_distillation_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_cfg:UnitreeGo2RoughDistillationRunnerCfg"
        ),
        "rsl_rl_distillation_recurrent_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_distillation_recurrent_cfg:UnitreeGo2RoughDistillationRecurrentRunnerCfg"
        ),
        "rsl_rl_cts_cfg_entry_point": f"{agents.__name__}.rsl_rl_cts_cfg:UnitreeGo2RoughCTSRunnerCfg",
        "rsl_rl_decap_cfg_entry_point": f"{agents.__name__}.rsl_rl_decap_cfg:UnitreeGo2RoughDecAPPPORunnerCfg",
        "cusrl_cfg_entry_point": f"{agents.__name__}.cusrl_ppo_cfg:UnitreeGo2RoughTrainerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-Velocity-Rough-Unitree-Go2-CTS-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_env_cfg:UnitreeGo2RoughCTSEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_cts_cfg:UnitreeGo2RoughCTSRunnerCfg",
        "rsl_rl_cts_cfg_entry_point": f"{agents.__name__}.rsl_rl_cts_cfg:UnitreeGo2RoughCTSRunnerCfg",
    },
)

gym.register(
    id="Gurukul-Isaac-Velocity-Rough-Unitree-Go2-Real-Sparse-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_real_sparse_env_cfg:UnitreeGo2RoughRealSparseEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RoughPPORunnerCfg",
        "rsl_rl_teacher_cfg_entry_point": f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeGo2RoughTeacherPPORunnerCfg",
        "rsl_rl_real_teacher_cfg_entry_point": f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeGo2RoughRealTeacherPPORunnerCfg",
        "rsl_rl_real_teacher_pretrained_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeGo2RoughRealTeacherPretrainedPPORunnerCfg"
        ),
        "rsl_rl_real_teacher_frozen_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeGo2RoughRealTeacherFrozenPPORunnerCfg"
        ),
        "rsl_rl_full_teacher_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeGo2RoughFullTeacherPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Gurukul-Isaac-Velocity-Rough-Unitree-Go2-Real-Beam-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_real_beam_env_cfg:UnitreeGo2RoughRealBeamEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_ppo_cfg:UnitreeGo2RoughBeamPPORunnerCfg",
        "rsl_rl_teacher_cfg_entry_point": f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeGo2RoughTeacherPPORunnerCfg",
        "rsl_rl_real_teacher_cfg_entry_point": f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeGo2RoughRealBeamPPORunnerCfg",
        "rsl_rl_real_teacher_pretrained_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeGo2RoughRealBeamPretrainedPPORunnerCfg"
        ),
        "rsl_rl_real_teacher_frozen_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeGo2RoughRealBeamFrozenPPORunnerCfg"
        ),
        "rsl_rl_full_teacher_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_teacher_cfg:UnitreeGo2RoughFullTeacherPPORunnerCfg"
        ),
    },
)


gym.register(
    id="Gurukul-Isaac-Velocity-Rough-Unitree-Go2-Contact-Trails-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_contact_trails_env_cfg:UnitreeGo2RoughContactTrailsEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_contact_trails_cfg:UnitreeGo2RoughContactTrailsPPORunnerCfg",
        "rsl_rl_contact_trails_engineered_cfg_entry_point": (
            f"{agents.__name__}.rsl_rl_contact_trails_cfg:UnitreeGo2RoughContactTrailsEngineeredPPORunnerCfg"
        ),
    },
)

gym.register(
    id="Gurukul-Isaac-Velocity-Rough-Unitree-Go2-Start-v0",
    entry_point="isaaclab.envs:ManagerBasedRLEnv",
    disable_env_checker=True,
    kwargs={
        "env_cfg_entry_point": f"{__name__}.rough_start_env_cfg:UnitreeGo2RoughStartEnvCfg",
        "rsl_rl_cfg_entry_point": f"{agents.__name__}.rsl_rl_start_cfg:UnitreeGo2RoughStartPPORunnerCfg",
    },
)
