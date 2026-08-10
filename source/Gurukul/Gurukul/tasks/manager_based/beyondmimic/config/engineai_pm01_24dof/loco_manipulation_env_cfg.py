"""PM01 whole-body object interaction tasks.

DeepMimic variants follow arXiv:1804.02717 by combining explicit reference
tracking with task rewards. AMP variants follow arXiv:2104.02180 by removing
phase/reference observations and explicit imitation rewards; SKRL supplies the
learned transition-style reward. The heavy-object randomization follows the
compatible portions of arXiv:2310.03191.
"""

from __future__ import annotations

import copy
from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import Gurukul.tasks.manager_based.beyondmimic.mdp as mdp
import Gurukul.tasks.manager_based.locomotion.velocity.mdp as loco_mdp

from .flat_env_cfg import EngineAiPm0124DofBeyondMimicFlatEnvCfg, _PM01_24DOF_TRACKED_BODIES


_PM01_ARM_END_BODIES = ["LINK_ELBOW_YAW_L", "LINK_ELBOW_YAW_R"]
_MOTION_DIR = Path(__file__).parent / "motion"


def _dynamic_box_cfg(
    *,
    size: tuple[float, float, float],
    mass: float,
    init_pos: tuple[float, float, float],
    color: tuple[float, float, float],
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                disable_gravity=False,
                linear_damping=0.02,
                angular_damping=0.02,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.9,
                dynamic_friction=0.7,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos, rot=(1.0, 0.0, 0.0, 0.0)),
    )


def _table_cfg() -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.CuboidCfg(
            size=(0.90, 0.75, 0.06),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,
                dynamic_friction=0.8,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.30, 0.22, 0.14)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.70, 0.0, 0.63), rot=(1.0, 0.0, 0.0, 0.0)),
    )


@configclass
class EngineAiPm01LocoManipDeepMimicEnvCfg(EngineAiPm0124DofBeyondMimicFlatEnvCfg):
    """Shared PM01 DeepMimic-style object-interaction configuration."""

    object_size = (0.12, 0.12, 0.12)
    object_mass = 0.30
    object_init_pos = (0.58, 0.0, 0.72)
    object_color = (0.92, 0.32, 0.12)
    goal_offset = (0.98, 0.0, 0.72)
    object_pose_range = {"x": (-0.05, 0.05), "y": (-0.18, 0.18), "yaw": (-3.14, 3.14)}
    object_scale_range = (0.90, 1.10)
    object_mass_scale_range = (0.67, 1.50)
    object_static_friction_range = (0.75, 1.20)
    object_dynamic_friction_range = (0.45, 0.90)
    target_object_speed = 0.18
    object_goal_std = 0.30
    object_speed_std = 0.25
    standoff_distance = 0.42
    tabletop = True
    minimum_object_height = 0.54
    reference_motion = "dance.npz"

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 12.0
        self.scene.env_spacing = 3.5
        self.scene.replicate_physics = False
        self.commands.motion.motion_file = str(_MOTION_DIR / self.reference_motion)
        self.scene.object = _dynamic_box_cfg(
            size=self.object_size,
            mass=self.object_mass,
            init_pos=self.object_init_pos,
            color=self.object_color,
        )
        if self.tabletop:
            self.scene.table = _table_cfg()

        object_cfg = SceneEntityCfg("object")
        arm_ends_cfg = SceneEntityCfg("robot", body_names=list(_PM01_ARM_END_BODIES), preserve_order=True)

        self.observations.policy.object_position = ObsTerm(
            func=loco_mdp.object_position_b, params={"object_cfg": object_cfg}, clip=(-5.0, 5.0)
        )
        self.observations.policy.object_velocity = ObsTerm(
            func=loco_mdp.object_velocity_b, params={"object_cfg": object_cfg}, clip=(-5.0, 5.0), scale=0.5
        )
        self.observations.policy.object_goal_position = ObsTerm(
            func=loco_mdp.object_goal_position_b,
            params={"goal_offset": self.goal_offset},
            clip=(-5.0, 5.0),
        )
        self.observations.policy.object_to_goal = ObsTerm(
            func=loco_mdp.object_to_goal_b,
            params={"object_cfg": object_cfg, "goal_offset": self.goal_offset},
            clip=(-5.0, 5.0),
        )
        self.observations.policy.arm_end_position = ObsTerm(
            func=loco_mdp.body_position_b, params={"asset_cfg": arm_ends_cfg}, clip=(-5.0, 5.0)
        )
        self.observations.policy.object_standoff = ObsTerm(
            func=loco_mdp.base_to_object_standoff_b,
            params={"object_cfg": object_cfg, "standoff_distance": self.standoff_distance},
            clip=(-5.0, 5.0),
        )
        for term_name in (
            "object_position",
            "object_velocity",
            "object_goal_position",
            "object_to_goal",
            "arm_end_position",
            "object_standoff",
        ):
            setattr(
                self.observations.critic,
                term_name,
                copy.deepcopy(getattr(self.observations.policy, term_name)),
            )

        self.events.reset_object = EventTerm(
            func=loco_mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": object_cfg,
                "pose_range": self.object_pose_range,
                "velocity_range": {
                    "x": (-0.03, 0.03),
                    "y": (-0.03, 0.03),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (-0.05, 0.05),
                },
            },
        )
        self.events.randomize_object_scale = EventTerm(
            func=loco_mdp.randomize_rigid_body_scale,
            mode="prestartup",
            params={"asset_cfg": object_cfg, "scale_range": self.object_scale_range},
        )
        self.events.randomize_object_material = EventTerm(
            func=loco_mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("object", body_names=".*"),
                "static_friction_range": self.object_static_friction_range,
                "dynamic_friction_range": self.object_dynamic_friction_range,
                "restitution_range": (0.0, 0.05),
                "num_buckets": 64,
            },
        )
        self.events.randomize_object_mass = EventTerm(
            func=loco_mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("object", body_names=".*"),
                "mass_distribution_params": self.object_mass_scale_range,
                "operation": "scale",
                "recompute_inertia": True,
            },
        )

        self.rewards.arm_ends_to_object = RewTerm(
            func=loco_mdp.ee_to_object_tanh,
            weight=1.5,
            params={"ee_cfg": arm_ends_cfg, "object_cfg": object_cfg, "std": 0.35},
        )
        self.rewards.object_to_goal = RewTerm(
            func=loco_mdp.object_to_goal_exp,
            weight=5.0,
            params={"object_cfg": object_cfg, "goal_offset": self.goal_offset, "std": self.object_goal_std},
        )
        self.rewards.object_velocity_to_goal = RewTerm(
            func=loco_mdp.object_velocity_towards_goal_exp,
            weight=1.0,
            params={
                "object_cfg": object_cfg,
                "goal_offset": self.goal_offset,
                "target_speed": self.target_object_speed,
                "std": self.object_speed_std,
            },
        )
        self.rewards.object_upright = RewTerm(
            func=loco_mdp.object_upright_exp,
            weight=0.5,
            params={"object_cfg": object_cfg, "std": 0.35},
        )
        self.rewards.base_standoff = RewTerm(
            func=loco_mdp.base_to_object_standoff_exp,
            weight=1.0,
            params={
                "object_cfg": object_cfg,
                "standoff_distance": self.standoff_distance,
                "std": 0.35,
            },
        )
        self.rewards.base_faces_object = RewTerm(
            func=loco_mdp.base_faces_object_exp,
            weight=0.5,
            params={"object_cfg": object_cfg, "std": 0.45},
        )

        self.terminations.object_out_of_bounds = DoneTerm(
            func=loco_mdp.object_out_of_bounds,
            params={"object_cfg": object_cfg, "max_distance": 2.5},
        )
        self.terminations.object_below_minimum_height = DoneTerm(
            func=loco_mdp.object_below_minimum_height,
            params={"object_cfg": object_cfg, "minimum_height": self.minimum_object_height},
        )


@configclass
class EngineAiPm01TabletopDeepMimicEnvCfg(EngineAiPm01LocoManipDeepMimicEnvCfg):
    """Slide a light randomized object across a tabletop using PM01's arm-end links."""


@configclass
class EngineAiPm01HeavyPushDeepMimicEnvCfg(EngineAiPm01LocoManipDeepMimicEnvCfg):
    """Push a randomized 6--10 kg crate while tracking PM01 walking motion."""

    object_size = (0.45, 0.50, 0.42)
    object_mass = 10.0
    object_init_pos = (0.78, 0.0, 0.21)
    object_color = (0.15, 0.42, 0.78)
    goal_offset = (1.55, 0.0, 0.21)
    object_pose_range = {"x": (-0.12, 0.12), "y": (-0.30, 0.30), "yaw": (-0.35, 0.35)}
    object_scale_range = (0.85, 1.15)
    object_mass_scale_range = (0.60, 1.00)
    object_static_friction_range = (0.60, 1.20)
    object_dynamic_friction_range = (0.40, 0.90)
    target_object_speed = 0.28
    object_goal_std = 0.55
    object_speed_std = 0.30
    standoff_distance = 0.38
    tabletop = False
    minimum_object_height = -0.05
    reference_motion = "walking_24dof.npz"

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = 16.0
        self.scene.env_spacing = 4.0
        self.commands.motion.make_in_place = True
        self.rewards.motion_global_anchor_pos.weight = 0.15
        self.rewards.arm_ends_to_object.weight = 1.0
        self.rewards.object_to_goal.weight = 6.0
        self.rewards.object_velocity_to_goal.weight = 2.0
        self.rewards.object_upright.weight = 1.0
        self.rewards.base_standoff.weight = 1.5


class _EngineAiPm01AmpMixin:
    """Switch a DeepMimic task to phase-free AMP style learning."""

    amp_motion_command_name = "motion"
    amp_num_observations = 2
    amp_key_body_names = list(_PM01_24DOF_TRACKED_BODIES)

    def __post_init__(self):
        super().__post_init__()

        # AMP learns style from transitions. The actor is not synchronized to a
        # reference phase and receives no target motion or explicit tracking reward.
        for group in (self.observations.policy, self.observations.critic):
            group.command = None
            group.motion_anchor_pos_b = None
            group.motion_anchor_ori_b = None
        self.observations.critic.body_pos = None
        self.observations.critic.body_ori = None
        self.observations.policy.projected_gravity = ObsTerm(func=mdp.projected_gravity)
        self.observations.critic.projected_gravity = copy.deepcopy(self.observations.policy.projected_gravity)

        for reward_name in (
            "motion_global_anchor_pos",
            "motion_global_anchor_ori",
            "motion_body_pos",
            "motion_body_ori",
            "motion_body_lin_vel",
            "motion_body_ang_vel",
        ):
            setattr(self.rewards, reward_name, None)

        # Retain reference-state initialization, but use task-independent fall
        # tests rather than phase-synchronized pose-error terminations.
        self.terminations.anchor_pos = DoneTerm(
            func=loco_mdp.root_height_below_minimum, params={"minimum_height": 0.42}
        )
        self.terminations.anchor_ori = DoneTerm(func=loco_mdp.bad_orientation, params={"limit_angle": 0.9})
        self.terminations.ee_body_pos = None


@configclass
class EngineAiPm01TabletopAmpEnvCfg(_EngineAiPm01AmpMixin, EngineAiPm01TabletopDeepMimicEnvCfg):
    """Tabletop slide task regularized by an AMP prior learned from PM01 dance."""


@configclass
class EngineAiPm01HeavyPushAmpEnvCfg(_EngineAiPm01AmpMixin, EngineAiPm01HeavyPushDeepMimicEnvCfg):
    """Heavy crate push task regularized by an AMP prior learned from PM01 walking."""
