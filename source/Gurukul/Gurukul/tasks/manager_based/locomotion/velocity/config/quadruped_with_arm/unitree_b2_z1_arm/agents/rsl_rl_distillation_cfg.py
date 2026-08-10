# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)


@configclass
class UnitreeB2Z1ArmMovingFlatDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Distill the privileged flat ArmMoving teacher into the deployable 45D policy."""

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "unitree_b2_z1_arm_moving_flat_distill"
    teacher_experiment_name = "unitree_b2_z1_arm_moving_flat_teacher"
    teacher_load_run = ".*"
    teacher_load_checkpoint = "model_.*.pt"
    obs_groups = {"student": ["policy"], "teacher": ["teacher"]}
    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=0.1,
        noise_std_type="scalar",
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[1024, 512, 256],
        activation="elu",
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=15,
        loss_type="mse",
    )


@configclass
class UnitreeB2Z1ArmMovingRoughDistillationRunnerCfg(UnitreeB2Z1ArmMovingFlatDistillationRunnerCfg):
    """Distill the privileged rough ArmMoving teacher into the deployable policy."""

    experiment_name = "unitree_b2_z1_arm_moving_rough_distill"
    teacher_experiment_name = "unitree_b2_z1_arm_moving_rough_teacher"


@configclass
class UnitreeB2Z1WideArmMovingFlatDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    """Distill the privileged flat wide ArmMoving teacher into the deployable policy."""

    num_steps_per_env = 24
    max_iterations = 5000
    save_interval = 100
    experiment_name = "unitree_b2_z1_wide_arm_moving_flat_distill"
    teacher_experiment_name = "unitree_b2_z1_wide_arm_moving_flat_teacher"
    teacher_load_run = ".*"
    teacher_load_checkpoint = "model_.*.pt"
    obs_groups = {"student": ["policy"], "teacher": ["teacher"]}
    policy = RslRlDistillationStudentTeacherCfg(
        init_noise_std=0.1,
        noise_std_type="scalar",
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[512, 256, 128],
        teacher_hidden_dims=[1024, 512, 256],
        activation="elu",
    )
    algorithm = RslRlDistillationAlgorithmCfg(
        num_learning_epochs=2,
        learning_rate=1.0e-3,
        gradient_length=15,
        loss_type="mse",
    )
