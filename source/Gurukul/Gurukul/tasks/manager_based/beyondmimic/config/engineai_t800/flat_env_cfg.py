from pathlib import Path

from isaaclab.utils import configclass

from Gurukul.assets.engineai import (
    ENGINEAI_T800_ACTION_SCALE,
    ENGINEAI_T800_CFG,
    ENGINEAI_T800_POLICY_JOINT_NAMES,
)
from Gurukul.tasks.manager_based.beyondmimic.mdp.noise import JointwiseUniformNoiseCfg
from Gurukul.tasks.manager_based.beyondmimic.tracking_env_cfg import BeyondMimicEnvCfg

from .agents.rsl_rl_ppo_cfg import LOW_FREQ_SCALE

_T800_TRACKED_BODIES = [
    "LINK_BASE",
    "LINK_HIP_ROLL_L",
    "LINK_KNEE_PITCH_L",
    "LINK_ANKLE_ROLL_L",
    "LINK_HIP_ROLL_R",
    "LINK_KNEE_PITCH_R",
    "LINK_ANKLE_ROLL_R",
    "LINK_WAIST_YAW",
    "LINK_SHOULDER_ROLL_L",
    "LINK_ELBOW_YAW_L",
    "LINK_WRIST_END_L",
    "LINK_SHOULDER_ROLL_R",
    "LINK_ELBOW_YAW_R",
    "LINK_WRIST_END_R",
    "LINK_HEAD_YAW",
]


@configclass
class EngineAiT800BeyondMimicFlatEnvCfg(BeyondMimicEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # Keep EngineAI's official tracking horizon for checkpoint parity.
        self.episode_length_s = 10.0
        self.scene.robot = ENGINEAI_T800_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.env_spacing = 3.5
        self.actions.joint_pos.joint_names = list(ENGINEAI_T800_POLICY_JOINT_NAMES)
        self.actions.joint_pos.preserve_order = True
        self.actions.joint_pos.scale = dict(ENGINEAI_T800_ACTION_SCALE)

        self.commands.motion.motion_file = f"{Path(__file__).parent}/motion/dance_t800.npz"
        self.commands.motion.anchor_body_name = "LINK_BASE"
        self.commands.motion.body_names = list(_T800_TRACKED_BODIES)
        self.commands.motion.debug_vis_body_names = [
            "LINK_BASE",
            "LINK_ANKLE_ROLL_L",
            "LINK_ANKLE_ROLL_R",
            "LINK_WRIST_END_L",
            "LINK_WRIST_END_R",
        ]

        self.observations.policy.joint_vel.noise = JointwiseUniformNoiseCfg(
            joint_noise_scales={".*ANKLE.*": 3.0},
            default_n_min=-0.5,
            default_n_max=0.5,
        )
        self.events.randomize_com_positions.params["asset_cfg"].body_names = "LINK_WAIST_YAW"
        self.events.randomize_com_positions.params["com_range"] = {
            "x": (-0.1, 0.1),
            "y": (-0.1, 0.1),
            "z": (-0.1, 0.1),
        }
        self.rewards.motion_global_anchor_pos.params["std"] = 0.45
        self.rewards.motion_body_pos.params["std"] = 0.45
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
            r"^(?!LINK_ANKLE_ROLL_L$)(?!LINK_ANKLE_ROLL_R$)(?!LINK_WRIST_END_L$)(?!LINK_WRIST_END_R$).+$"
        ]
        self.rewards.action_rate_l2.weight = -0.03
        self.terminations.anchor_pos.params["threshold"] = 0.35
        self.terminations.ee_body_pos.params["threshold"] = 0.35
        self.terminations.ee_body_pos.params["body_names"] = [
            "LINK_ANKLE_ROLL_L",
            "LINK_ANKLE_ROLL_R",
            "LINK_WRIST_END_L",
            "LINK_WRIST_END_R",
        ]


@configclass
class EngineAiT800BeyondMimicFlatWoStateEstimationEnvCfg(EngineAiT800BeyondMimicFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.observations.policy.motion_anchor_pos_b = None
        self.observations.policy.base_lin_vel = None
        self.rewards.action_rate_l2.weight = -0.075


@configclass
class EngineAiT800BeyondMimicFlatLowFreqEnvCfg(EngineAiT800BeyondMimicFlatEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.decimation = round(self.decimation / LOW_FREQ_SCALE)
        self.rewards.action_rate_l2.weight *= LOW_FREQ_SCALE
