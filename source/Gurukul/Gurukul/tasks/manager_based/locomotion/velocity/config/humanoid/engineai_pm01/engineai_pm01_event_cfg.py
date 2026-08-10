# SPDX-License-Identifier: Apache-2.0
"""Event randomization matching engineai_amp PM01 (friction, default joint offset, push, reset)."""

import math

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import Gurukul.tasks.manager_based.locomotion.velocity.mdp as mdp
from Gurukul.tasks.manager_based.locomotion.velocity.velocity_env_cfg import EventCfg

from .pm01_constants import PM01_POLICY_JOINT_ASSET_CFG, PM01_PUSH_VELOCITY_RANGE


@configclass
class EngineAiPm01EventCfg(EventCfg):
    """Event terms for PM01; augments and overrides :class:`EventCfg` for the stack."""

    # Use `randomize_rigid_body_mass_fn`: Isaac Lab's `randomize_rigid_body_mass` is class-based and may not be
    # constructed before `apply(startup)` (timeline PLAY ordering).
    randomize_rigid_body_mass_base = EventTerm(
        func=mdp.randomize_rigid_body_mass_fn,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="LINK_BASE"),
            "mass_distribution_params": (-1.0, 3.0),
            "operation": "add",
            "recompute_inertia": True,
        },
    )
    randomize_rigid_body_mass_others = EventTerm(
        func=mdp.randomize_rigid_body_mass_fn,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "mass_distribution_params": (0.7, 1.3),
            "operation": "scale",
            "recompute_inertia": True,
        },
    )

    randomize_rigid_body_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.3, 1.6),
            "dynamic_friction_range": (0.3, 1.2),
            "restitution_range": (0.0, 0.5),
            "num_buckets": 64,
        },
    )

    add_joint_default_pos = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "asset_cfg": PM01_POLICY_JOINT_ASSET_CFG,
            "pos_distribution_params": (-0.01, 0.01),
            "operation": "add",
        },
    )

    randomize_com_positions = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="LINK_BASE"),
            "com_range": {"x": (-0.025, 0.025), "y": (-0.05, 0.05), "z": (-0.05, 0.05)},
        },
    )

    randomize_push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": PM01_PUSH_VELOCITY_RANGE},
    )

    randomize_reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (-0.5, 0.5), "y": (-0.5, 0.5), "yaw": (-math.pi, math.pi)},
            "velocity_range": {
                "x": (0.0, 0.0),
                "y": (0.0, 0.0),
                "z": (0.0, 0.0),
                "roll": (0.0, 0.0),
                "pitch": (0.0, 0.0),
                "yaw": (0.0, 0.0),
            },
        },
    )

    randomize_reset_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={"position_range": (0.8, 1.2), "velocity_range": (-0.5, 0.5)},
    )
