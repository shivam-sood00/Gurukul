from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

import Gurukul.tasks.manager_based.beyondmimic.mdp as mdp
from Gurukul.tasks.manager_based.beyondmimic.tracking_env_cfg import ObservationsCfg

from .flat_env_cfg import UnitreeG1BeyondMimicFlatEnvCfg

G1_APEX_REFERENCE_TIME_OFFSETS = (0, 1, 2, 5, 10)


def g1_apex_reference_params() -> dict:
    return {
        "command_name": "motion",
        "time_offsets": G1_APEX_REFERENCE_TIME_OFFSETS,
        "include_joint_pos": True,
        "include_joint_vel": False,
        "include_base_lin_vel": True,
        "include_base_ang_vel": True,
        "include_base_quat": True,
        "include_base_rotmat": False,
    }


@configclass
class G1ApexTrackerPolicyObservationsCfg(ObservationsCfg.PolicyCfg):
    """Actor observations with current/future reference motion features."""

    reference_motion = ObsTerm(
        func=mdp.reference_motion_state,
        params=g1_apex_reference_params(),
        clip=(-100.0, 100.0),
        scale=1.0,
    )


@configclass
class G1ApexTrackerCriticObservationsCfg(ObservationsCfg.CriticCfg):
    """Critic observations with the same reference-motion bundle used by the actor."""

    reference_motion = ObsTerm(
        func=mdp.reference_motion_state,
        params=g1_apex_reference_params(),
        clip=(-100.0, 100.0),
        scale=1.0,
    )


@configclass
class UnitreeG1BeyondMimicApexFlatTrackerEnvCfg(UnitreeG1BeyondMimicFlatEnvCfg):
    """G1 BeyondMimic task with APEX-style future reference observations."""

    def __post_init__(self):
        super().__post_init__()
        self.commands.motion.pose_range = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.commands.motion.velocity_range = {
            "x": (0.0, 0.0),
            "y": (0.0, 0.0),
            "z": (0.0, 0.0),
            "roll": (0.0, 0.0),
            "pitch": (0.0, 0.0),
            "yaw": (0.0, 0.0),
        }
        self.commands.motion.joint_position_range = (0.0, 0.0)
        self.observations.policy = G1ApexTrackerPolicyObservationsCfg()
        self.observations.critic = G1ApexTrackerCriticObservationsCfg()
