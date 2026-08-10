# SPDX-License-Identifier: Apache-2.0
"""Commands and 24-DoF actions for official PM01 velocity tasks."""

import math

from isaaclab.utils import configclass

import Gurukul.tasks.manager_based.locomotion.velocity.mdp as mdp
from Gurukul.assets.engineai_pm01_official import PM01_24DOF_ACTION_SCALE
from Gurukul.tasks.manager_based.locomotion.velocity.velocity_env_cfg import ActionsCfg, CommandsCfg

from .pm01_constants import PM01_POLICY_JOINT_NAMES


@configclass
class EngineAiPm01CommandsCfg(CommandsCfg):
    """Uniform velocity commands with EngineAI-style ranges and resampling."""

    base_velocity = mdp.UniformThresholdVelocityCommandCfg(
        asset_name="robot",
        resampling_time_range=(7.5, 7.5),
        rel_standing_envs=0.1,
        rel_heading_envs=1.0,
        heading_command=True,
        heading_control_stiffness=0.5,
        debug_vis=True,
        marker_height_offset=0.65,
        ranges=mdp.UniformThresholdVelocityCommandCfg.Ranges(
            lin_vel_x=(-1.0, 1.5),
            lin_vel_y=(-0.5, 0.5),
            ang_vel_z=(-1.0, 1.0),
            heading=(-math.pi, math.pi),
        ),
    )


@configclass
class EngineAiPm01ActionsCfg(ActionsCfg):
    """Joint position targets for all 24 official PM01 joints."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        use_default_offset=True,
        preserve_order=True,
        joint_names=PM01_POLICY_JOINT_NAMES,
        scale=dict(PM01_24DOF_ACTION_SCALE),
        clip={".*": (-100.0, 100.0)},
    )
