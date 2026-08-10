from isaaclab.utils import configclass

from .rsl_rl_ppo_cfg import (
    UnitreeGo2ApexFlatPPORunnerCfg,
    UnitreeGo2D1ArmApexFlatTrackerPPORunnerCfg,
    UnitreeGo2D1ArmApexPickStowCarryFlatTrackerPPORunnerCfg,
)


@configclass
class UnitreeGo2ApexFlatTeacherPPORunnerCfg(UnitreeGo2ApexFlatPPORunnerCfg):
    """Teacher uses privileged critic observations for both actor and critic."""

    obs_groups = {"actor": ["critic"], "critic": ["critic"]}

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_go2_apex_flat_teacher"


@configclass
class UnitreeGo2ApexFlatPrivilegedTrackerPPORunnerCfg(UnitreeGo2ApexFlatPPORunnerCfg):
    """Privileged APEX tracker teacher with clean full-state motion observations."""

    obs_groups = {"actor": ["privileged"], "critic": ["privileged"]}

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_go2_apex_flat_privileged_tracker"
        self.clip_actions = None
        self.policy.actor_hidden_dims = [1024, 512, 256]
        self.policy.critic_hidden_dims = [1024, 512, 256]


@configclass
class UnitreeGo2D1ArmApexOriginalDecapTeacherPPORunnerCfg(UnitreeGo2D1ArmApexFlatTrackerPPORunnerCfg):
    """Privileged Go2+D1 PPO teacher trained with environment-side original DecAP."""

    obs_groups = {"actor": ["privileged"], "critic": ["privileged"]}

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_go2_d1_arm_apex_original_decap_teacher"
        # The D1 PPO parent installs its deployable policy/critic mapping in
        # __post_init__, so replace it after the parent has finished.
        self.obs_groups = {"actor": ["privileged"], "critic": ["privileged"]}
        self.policy.actor_hidden_dims = [1024, 512, 256]
        self.policy.critic_hidden_dims = [1024, 512, 256]


@configclass
class UnitreeGo2D1ArmApexPickStowCarryPrivilegedTeacherPPORunnerCfg(
    UnitreeGo2D1ArmApexPickStowCarryFlatTrackerPPORunnerCfg
):
    """Full-state pick/stow/carry teacher for later supervised distillation."""

    obs_groups = {"actor": ["privileged"], "critic": ["privileged"]}

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_go2_d1_arm_apex_pick_stow_carry_privileged_teacher"
        self.obs_groups = {"actor": ["privileged"], "critic": ["privileged"]}
        self.policy.actor_hidden_dims = [1024, 512, 256]
        self.policy.critic_hidden_dims = [1024, 512, 256]


@configclass
class UnitreeB2Z1ArmApexFlatPrivilegedTrackerPPORunnerCfg(UnitreeGo2ApexFlatPPORunnerCfg):
    """Privileged B2+Z1 APEX tracker teacher with clean full-state motion observations."""

    obs_groups = {"actor": ["privileged"], "critic": ["privileged"]}

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_b2_z1_arm_apex_flat_privileged_tracker"
        self.clip_actions = None
        self.policy.actor_hidden_dims = [1024, 512, 256]
        self.policy.critic_hidden_dims = [1024, 512, 256]


@configclass
class UnitreeB2Z1ArmApexFlatFixedWristGripperPrivilegedTrackerPPORunnerCfg(
    UnitreeB2Z1ArmApexFlatPrivilegedTrackerPPORunnerCfg
):
    """Privileged B2+Z1 tracker teacher with fixed wrist and commanded gripper."""

    def __post_init__(self):
        super().__post_init__()
        self.experiment_name = "unitree_b2_z1_arm_apex_flat_fixed_wrist_gripper_privileged_tracker"
