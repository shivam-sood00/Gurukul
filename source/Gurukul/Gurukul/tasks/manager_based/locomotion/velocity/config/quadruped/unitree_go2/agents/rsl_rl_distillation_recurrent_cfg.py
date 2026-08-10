# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherRecurrentCfg,
)


@configclass
class UnitreeGo2RoughDistillationRecurrentRunnerCfg(RslRlDistillationRunnerCfg):
    """Opt-in recurrent student distillation config for rough-terrain Go2."""

    num_steps_per_env = 120
    max_iterations = 5000
    save_interval = 100
    experiment_name = "unitree_go2_rough_distill_recurrent"
    # Keep teacher loading behavior aligned with the default distillation config.
    teacher_experiment_name = "unitree_go2_rough_teacher"
    teacher_load_run = ".*"
    teacher_load_checkpoint = "model_.*.pt"
    # Student uses deployable policy observations; teacher uses privileged critic observations.
    obs_groups = {"policy": ["policy"], "teacher": ["critic"]}
    policy = RslRlDistillationStudentTeacherRecurrentCfg(
        init_noise_std=0.1,
        noise_std_type="scalar",
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[512, 256, 128],
        activation="elu",
        rnn_type="gru",
        rnn_hidden_dim=256,
        rnn_num_layers=1,
        # Teacher checkpoint comes from a non-recurrent PPO teacher by default.
        teacher_recurrent=False,
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=5.0e-4,
        gradient_length=24,
        max_grad_norm=1.0,
    )


@configclass
class UnitreeGo2FlatDistillationRecurrentRunnerCfg(UnitreeGo2RoughDistillationRecurrentRunnerCfg):
    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_go2_flat_distill_recurrent"
        self.teacher_experiment_name = "unitree_go2_flat_teacher"
