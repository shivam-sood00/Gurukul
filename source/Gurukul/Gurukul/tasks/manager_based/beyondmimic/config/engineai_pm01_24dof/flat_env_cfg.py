from pathlib import Path

from isaaclab.utils import configclass

from Gurukul.assets.engineai_pm01_official import (
    ENGINEAI_PM01_24DOF_CFG,
    PM01_24DOF_ACTION_SCALE,
    PM01_24DOF_POLICY_JOINT_NAMES,
)
from Gurukul.tasks.manager_based.beyondmimic.mdp.noise import JointwiseUniformNoiseCfg
from Gurukul.tasks.manager_based.beyondmimic.tracking_env_cfg import BeyondMimicEnvCfg

from .agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE

_PM01_24DOF_TRACKED_BODIES = [
    "LINK_BASE",
    "LINK_HIP_ROLL_L",
    "LINK_KNEE_PITCH_L",
    "LINK_ANKLE_ROLL_L",
    "LINK_HIP_ROLL_R",
    "LINK_KNEE_PITCH_R",
    "LINK_ANKLE_ROLL_R",
    "LINK_TORSO_YAW",
    "LINK_SHOULDER_ROLL_L",
    "LINK_ELBOW_YAW_L",
    "LINK_SHOULDER_ROLL_R",
    "LINK_ELBOW_YAW_R",
    "LINK_HEAD_YAW",
]


@configclass
class EngineAiPm0124DofBeyondMimicFlatEnvCfg(BeyondMimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 10.0
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.scene.robot = ENGINEAI_PM01_24DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.actions.joint_pos.joint_names = list(PM01_24DOF_POLICY_JOINT_NAMES)
        self.actions.joint_pos.preserve_order = True
        self.actions.joint_pos.scale = dict(PM01_24DOF_ACTION_SCALE)

        self.commands.motion.motion_file = f"{Path(__file__).parent}/motion/dance.npz"
        self.commands.motion.anchor_body_name = "LINK_BASE"
        self.commands.motion.body_names = list(_PM01_24DOF_TRACKED_BODIES)
        self.commands.motion.debug_vis_body_names = [
            "LINK_BASE",
            "LINK_ANKLE_ROLL_L",
            "LINK_ANKLE_ROLL_R",
            "LINK_ELBOW_YAW_L",
            "LINK_ELBOW_YAW_R",
            "LINK_HEAD_YAW",
        ]

        self.observations.policy.joint_vel.noise = JointwiseUniformNoiseCfg(
            joint_noise_scales={".*ANKLE.*": 3.0},
            default_n_min=-0.5,
            default_n_max=0.5,
        )
        self.events.randomize_com_positions.params["asset_cfg"].body_names = "LINK_BASE"
        self.rewards.joint_acc_l2 = None
        self.rewards.joint_torques_l2 = None
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            r"^(?!LINK_ANKLE_ROLL_L$)(?!LINK_ANKLE_ROLL_R$)(?!LINK_ELBOW_YAW_L$)(?!LINK_ELBOW_YAW_R$).+$"
        ]
        self.terminations.ee_body_pos.params["body_names"] = [
            "LINK_ANKLE_ROLL_L",
            "LINK_ANKLE_ROLL_R",
            "LINK_ELBOW_YAW_L",
            "LINK_ELBOW_YAW_R",
        ]


@configclass
class EngineAiPm0124DofBeyondMimicFlatWoStateEstimationEnvCfg(
    EngineAiPm0124DofBeyondMimicFlatEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None


@configclass
class EngineAiPm0124DofBeyondMimicFlatLowFreqEnvCfg(EngineAiPm0124DofBeyondMimicFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE


@configclass
class EngineAiPm0124DofWalkingBeyondMimicFlatEnvCfg(EngineAiPm0124DofBeyondMimicFlatEnvCfg):
    """Official 24-DoF PM01 parameters tracking the retargeted walking motion."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.motion_file = f"{Path(__file__).parent}/motion/walking_24dof.npz"


@configclass
class EngineAiPm0124DofWalkingBeyondMimicFlatWoStateEstimationEnvCfg(
    EngineAiPm0124DofWalkingBeyondMimicFlatEnvCfg
):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None
