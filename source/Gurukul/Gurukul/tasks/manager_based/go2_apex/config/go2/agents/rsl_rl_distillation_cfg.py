from isaaclab.utils import configclass

from isaaclab_rl.rsl_rl import (
    RslRlDistillationAlgorithmCfg,
    RslRlDistillationRunnerCfg,
    RslRlDistillationStudentTeacherCfg,
)


@configclass
class UnitreeGo2ApexFlatDistillationRunnerCfg(RslRlDistillationRunnerCfg):
    num_steps_per_env = 120
    max_iterations = 5000
    save_interval = 100
    experiment_name = "unitree_go2_apex_flat"
    teacher_experiment_name = "unitree_go2_apex_flat_teacher"
    teacher_load_run = ".*"
    teacher_load_checkpoint = "model_.*.pt"
    obs_groups = {"student": ["policy"], "teacher": ["critic"]}
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
        loss_type="mse",
    )


@configclass
class UnitreeGo2ApexFlatTrackerDistillationRunnerCfg(UnitreeGo2ApexFlatDistillationRunnerCfg):
    """Distill the privileged APEX tracker teacher into the deployable noisy tracker policy."""

    experiment_name = "unitree_go2_apex_flat_tracker_distill"
    teacher_experiment_name = "unitree_go2_apex_flat_privileged_tracker"
    obs_groups = {"student": ["policy"], "teacher": ["privileged"]}

    def __post_init__(self):
        super().__post_init__()
        self.policy.teacher_hidden_dims = [1024, 512, 256]


@configclass
class UnitreeGo2D1ArmApexDistillationStudentRunnerCfg(UnitreeGo2ApexFlatTrackerDistillationRunnerCfg):
    """Supervise the separate deployable D1 student from the original-DecAP teacher."""

    experiment_name = "unitree_go2_d1_arm_apex_flat_tracker_distill"
    teacher_experiment_name = "unitree_go2_d1_arm_apex_original_decap_teacher"
    obs_groups = {"student": ["policy"], "teacher": ["privileged"]}
    clip_actions = 6.0


@configclass
class UnitreeGo2D1ArmApexPickStowCarryDistillationRunnerCfg(UnitreeGo2ApexFlatTrackerDistillationRunnerCfg):
    """Distill the manipulation teacher into the pick task's deployable policy group."""

    experiment_name = "unitree_go2_d1_arm_apex_pick_stow_carry_distill"
    teacher_experiment_name = "unitree_go2_d1_arm_apex_pick_stow_carry_privileged_teacher"
    obs_groups = {"student": ["policy"], "teacher": ["privileged"]}
    clip_actions = 6.6

    def __post_init__(self):
        super().__post_init__()
        self.policy.student_hidden_dims = [1024, 512, 256]
        self.policy.teacher_hidden_dims = [1024, 512, 256]


@configclass
class UnitreeGo2ApexFlatOneStepFutureTrackerDistillationRunnerCfg(UnitreeGo2ApexFlatTrackerDistillationRunnerCfg):
    """Distill the full privileged tracker teacher into the one-step-future deployable student."""

    experiment_name = "unitree_go2_apex_flat_tracker_one_step_future_distill"
    teacher_experiment_name = "unitree_go2_apex_flat_privileged_tracker"


@configclass
class UnitreeGo2ApexFlatOneStepFutureTrackerHistoryDistillationRunnerCfg(
    UnitreeGo2ApexFlatTrackerDistillationRunnerCfg
):
    """Distill the full privileged tracker teacher into a one-step student with deployable observation history."""

    experiment_name = "unitree_go2_apex_flat_tracker_one_step_future_history_distill"
    teacher_experiment_name = "unitree_go2_apex_flat_privileged_tracker"

    def __post_init__(self):
        super().__post_init__()
        self.policy.student_hidden_dims = [1024, 512, 256]


@configclass
class UnitreeB2Z1ArmApexFlatTrackerDistillationRunnerCfg(UnitreeGo2ApexFlatDistillationRunnerCfg):
    """Distill the privileged B2+Z1 APEX tracker teacher into the deployable noisy tracker policy."""

    experiment_name = "unitree_b2_z1_arm_apex_flat_tracker_distill"
    teacher_experiment_name = "unitree_b2_z1_arm_apex_flat_privileged_tracker"
    obs_groups = {"student": ["policy"], "teacher": ["privileged"]}

    def __post_init__(self):
        super().__post_init__()
        self.policy.teacher_hidden_dims = [1024, 512, 256]


@configclass
class UnitreeB2Z1ArmApexFlatFixedWristGripperTrackerDistillationRunnerCfg(
    UnitreeB2Z1ArmApexFlatTrackerDistillationRunnerCfg
):
    """Distill the fixed-wrist/gripper-command B2+Z1 privileged tracker teacher."""

    experiment_name = "unitree_b2_z1_arm_apex_flat_fixed_wrist_gripper_tracker_distill"
    teacher_experiment_name = "unitree_b2_z1_arm_apex_flat_fixed_wrist_gripper_privileged_tracker"


@configclass
class UnitreeB2Z1ArmApexFlatOneStepFutureTrackerDistillationRunnerCfg(
    UnitreeB2Z1ArmApexFlatTrackerDistillationRunnerCfg
):
    """Distill the privileged B2+Z1 tracker teacher into the one-step-future student."""

    experiment_name = "unitree_b2_z1_arm_apex_flat_tracker_one_step_future_distill"
    teacher_experiment_name = "unitree_b2_z1_arm_apex_flat_privileged_tracker"


@configclass
class UnitreeB2Z1ArmApexFlatOneStepFutureTrackerHistoryDistillationRunnerCfg(
    UnitreeB2Z1ArmApexFlatTrackerDistillationRunnerCfg
):
    """Distill the privileged B2+Z1 tracker teacher into a one-step student with deployable observation history."""

    experiment_name = "unitree_b2_z1_arm_apex_flat_tracker_one_step_future_history_distill"
    teacher_experiment_name = "unitree_b2_z1_arm_apex_flat_privileged_tracker"

    def __post_init__(self):
        super().__post_init__()
        self.policy.student_hidden_dims = [1024, 512, 256]
