# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from Gurukul.assets.engineai_pm01_official import ENGINEAI_PM01_24DOF_CFG
from Gurukul.tasks.manager_based.locomotion.velocity.velocity_env_cfg import LocomotionVelocityRoughEnvCfg

from .engineai_pm01_commands_actions_cfg import EngineAiPm01ActionsCfg, EngineAiPm01CommandsCfg
from .engineai_pm01_event_cfg import EngineAiPm01EventCfg
from .engineai_pm01_rewards_cfg import EngineAiPm01RewardsCfg
from .pm01_constants import PM01_POLICY_JOINT_ASSET_CFG

_HIST_LEN = 15


@configclass
class EngineAiPm01RoughEnvCfg(LocomotionVelocityRoughEnvCfg):
    """EngineAI-Lab-style PM01 rough-terrain velocity (full non-AMP stack)."""

    rewards: EngineAiPm01RewardsCfg = EngineAiPm01RewardsCfg()
    commands: EngineAiPm01CommandsCfg = EngineAiPm01CommandsCfg()
    actions: EngineAiPm01ActionsCfg = EngineAiPm01ActionsCfg()
    events: EngineAiPm01EventCfg = EngineAiPm01EventCfg()

    base_link_name = "LINK_BASE"
    foot_link_name = "LINK_ANKLE_ROLL_[LR]"

    def __post_init__(self) -> None:
        super().__post_init__()

        self.scene.robot = ENGINEAI_PM01_24DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        # Preserve the collision behavior used to train the velocity policy.
        self.scene.robot.spawn.articulation_props.enabled_self_collisions = False

        # Match the front three-quarter PM01 framing in EngineAI's official
        # engineai_rl_lab tracking environment.
        self.viewer.eye = (1.5, 1.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"

        # Match EngineAI's official 500 Hz low-level and 50 Hz policy rates.
        self.decimation = 10
        self.sim.dt = 0.002
        self.sim.render_interval = self.decimation

        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        if self.scene.height_scanner is not None:
            self.scene.height_scanner.update_period = self.decimation * self.sim.dt
        if getattr(self.scene, "height_scanner_base", None) is not None:
            self.scene.height_scanner_base.update_period = self.decimation * self.sim.dt
        if self.scene.contact_forces is not None:
            self.scene.contact_forces.update_period = 0.005

        self.observations.policy.base_lin_vel = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.observations.policy.joint_pos.params["asset_cfg"] = PM01_POLICY_JOINT_ASSET_CFG
        self.observations.policy.joint_vel.params["asset_cfg"] = PM01_POLICY_JOINT_ASSET_CFG
        self.observations.critic.joint_pos.params["asset_cfg"] = PM01_POLICY_JOINT_ASSET_CFG
        self.observations.critic.joint_vel.params["asset_cfg"] = PM01_POLICY_JOINT_ASSET_CFG

        self.observations.policy.base_ang_vel.scale = 0.25
        self.observations.policy.joint_vel.scale = 0.05

        for name in ("joint_pos", "joint_vel", "actions", "base_ang_vel", "projected_gravity"):
            pol = getattr(self.observations.policy, name)
            cri = getattr(self.observations.critic, name)
            if pol is not None:
                pol.history_length = _HIST_LEN
            if cri is not None:
                cri.history_length = _HIST_LEN

        self.events.randomize_apply_external_force_torque = None
        self.events.randomize_actuator_gains = None

        self.terminations.illegal_contact.params["sensor_cfg"].body_names = [
            self.base_link_name,
            "LINK_KNEE_PITCH.*",
            ".*SHOULDER.*",
            ".*ELBOW.*",
        ]

        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None

        if self.__class__.__name__ == "EngineAiPm01RoughEnvCfg":
            self.disable_zero_weight_rewards()
