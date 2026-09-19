# Copyright (c) 2025- Shenzhen Zhongqing Robot Technology Co., Ltd. ("EngineAI")
# Copyright (c) 2026, Gurukul contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""PM01 flat velocity following with EngineAI's adversarial walking prior."""

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

import Gurukul.tasks.manager_based.locomotion.velocity.mdp as mdp
from Gurukul.tasks.manager_based.locomotion.velocity.velocity_env_cfg import ObservationsCfg

from .flat_env_cfg import EngineAiPm01FlatEnvCfg
from .pm01_constants import PM01_AMP_JOINT_ASSET_CFG, PM01_AMP_REFERENCE_ARM_POSE


@configclass
class EngineAiPm01AmpObservationsCfg(ObservationsCfg):
    """Actor/critic observations plus the task-independent AMP state history."""

    @configclass
    class AmpCfg(ObsGroup):
        state = ObsTerm(
            func=mdp.pm01_amp_state,
            params={"asset_cfg": PM01_AMP_JOINT_ASSET_CFG},
        )

        def __post_init__(self) -> None:
            self.concatenate_terms = True
            self.enable_corruption = False
            self.history_length = 5
            self.flatten_history_dim = True

    amp: AmpCfg = AmpCfg()


@configclass
class EngineAiPm01AmpFlatEnvCfg(EngineAiPm01FlatEnvCfg):
    """Keep the PM01 velocity MDP and expose a five-frame AMP observation."""

    observations: EngineAiPm01AmpObservationsCfg = EngineAiPm01AmpObservationsCfg()

    def __post_init__(self) -> None:
        super().__post_init__()

        # The discriminator reads absolute joint positions, so a standing offset between the
        # robot's nominal pose and the pose EngineAI recorded the clip in is a constant, free
        # feature for the discriminator. The maintained PM01 nominal arm pose sits 8-15 sigma
        # from the reference on shoulder yaw, elbow yaw and shoulder roll, which is enough to
        # separate expert from policy on posture alone and pin the style reward near zero.
        # Adopt the reference arm pose here so the prior has to judge the gait instead. Legs
        # already sit within one sigma of the clip and keep the deployment stance untouched.
        #
        # This also moves the action offset (``use_default_offset=True``) and the
        # ``variable_posture`` target for this task only, so an AMP checkpoint has a different
        # nominal pose from the plain velocity checkpoint. Export and deploy them separately.
        # ``ArticulationCfg.replace`` is a shallow dataclass copy, so the rough config's robot
        # shares one ``init_state`` with every other PM01 task. Swap in a new one rather than
        # mutating it, or building an AMP config would move the plain velocity task's pose too.
        self.scene.robot.init_state = self.scene.robot.init_state.replace(
            joint_pos={**self.scene.robot.init_state.joint_pos, **PM01_AMP_REFERENCE_ARM_POSE}
        )

        self.disable_zero_weight_rewards()
