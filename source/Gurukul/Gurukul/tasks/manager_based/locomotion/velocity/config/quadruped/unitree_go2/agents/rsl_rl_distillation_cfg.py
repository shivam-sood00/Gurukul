# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)


@configclass
class UnitreeGo2RoughDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 120
    max_iterations = 5000
    save_interval = 100
    experiment_name = "unitree_go2_rough"
    # Distillation logs are written under `experiment_name`, but teacher checkpoints are loaded
    # from this dedicated experiment folder.
    teacher_experiment_name = "unitree_go2_rough_teacher"
    teacher_load_run = ".*"
    teacher_load_checkpoint = "model_.*.pt"
    # Teacher uses privileged critic observations, student uses deployable policy observations.
    obs_groups = {"policy": ["policy"], "teacher": ["critic"]}
    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=0.1,
        noise_std_type="scalar",
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[512, 256, 128],
        activation="elu",
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=15,
    )


@configclass
class UnitreeGo2FlatDistillationRunnerCfg(UnitreeGo2RoughDistillationRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_go2_flat"
        self.teacher_experiment_name = "unitree_go2_flat_teacher"
