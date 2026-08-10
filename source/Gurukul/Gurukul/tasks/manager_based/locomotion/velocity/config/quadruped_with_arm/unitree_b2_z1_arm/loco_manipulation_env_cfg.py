# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import EventTermCfg, ObservationTermCfg as ObsTerm, RewardTermCfg as RewTerm, SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import Gurukul.tasks.manager_based.locomotion.velocity.mdp as mdp

from .flat_env_cfg import UnitreeB2Z1ArmFlatEnvCfg


def _training_object_cfg(
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


def _cone_object_cfg(
    *,
    radius: float,
    height: float,
    mass: float,
    init_pos: tuple[float, float, float],
    color: tuple[float, float, float],
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.ConeCfg(
            radius=radius,
            height=height,
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
                static_friction=0.75,
                dynamic_friction=0.55,
                restitution=0.35,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos, rot=(1.0, 0.0, 0.0, 0.0)),
    )


def _kinematic_box_cfg(
    *,
    prim_path: str,
    size: tuple[float, float, float],
    init_pos: tuple[float, float, float],
    color: tuple[float, float, float],
    opacity: float = 1.0,
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path=prim_path,
        spawn=sim_utils.CuboidCfg(
            size=size,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color, opacity=opacity),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos, rot=(1.0, 0.0, 0.0, 0.0)),
    )


@configclass
class UnitreeB2Z1LocoManipFlatEnvCfg(UnitreeB2Z1ArmFlatEnvCfg):
    """Shared flat B2+Z1 loco-manipulation setup with a trainable rigid object."""

    object_size = (0.35, 0.35, 0.35)
    object_mass = 3.0
    object_init_pos = (0.95, 0.0, 0.175)
    object_color = (0.1, 0.45, 0.85)
    goal_offset = (1.75, 0.0, 0.175)
    object_reset_xy = {"x": (-0.20, 0.20), "y": (-0.35, 0.35), "yaw": (-0.4, 0.4)}
    target_object_velocity = (0.25, 0.0)

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 12.0
        self.scene.env_spacing = 4.0
        self.scene.object = _training_object_cfg(
            size=self.object_size,
            mass=self.object_mass,
            init_pos=self.object_init_pos,
            color=self.object_color,
        )

        ee_cfg = SceneEntityCfg("robot", body_names=["link06"])
        object_cfg = SceneEntityCfg("object")

        self.observations.policy.object_position = ObsTerm(
            func=mdp.object_position_b,
            params={"object_cfg": object_cfg},
            clip=(-5.0, 5.0),
            scale=1.0,
        )
        self.observations.policy.object_goal_position = ObsTerm(
            func=mdp.object_goal_position_b,
            params={"goal_offset": self.goal_offset},
            clip=(-5.0, 5.0),
            scale=1.0,
        )
        self.observations.policy.object_to_goal = ObsTerm(
            func=mdp.object_to_goal_b,
            params={"goal_offset": self.goal_offset, "object_cfg": object_cfg},
            clip=(-5.0, 5.0),
            scale=1.0,
        )
        self.observations.policy.ee_position = ObsTerm(
            func=mdp.body_position_b,
            params={"asset_cfg": ee_cfg},
            clip=(-5.0, 5.0),
            scale=1.0,
        )

        self.observations.critic.object_position = copy.deepcopy(self.observations.policy.object_position)
        self.observations.critic.object_goal_position = copy.deepcopy(self.observations.policy.object_goal_position)
        self.observations.critic.object_to_goal = copy.deepcopy(self.observations.policy.object_to_goal)
        self.observations.critic.ee_position = copy.deepcopy(self.observations.policy.ee_position)

        self.events.reset_object = EventTermCfg(
            func=mdp.reset_root_state_uniform,
            mode="reset",
            params={
                "asset_cfg": object_cfg,
                "pose_range": self.object_reset_xy,
                "velocity_range": {
                    "x": (-0.05, 0.05),
                    "y": (-0.05, 0.05),
                    "z": (0.0, 0.0),
                    "roll": (0.0, 0.0),
                    "pitch": (0.0, 0.0),
                    "yaw": (-0.05, 0.05),
                },
            },
        )

        self.terminations.object_out_of_bounds = DoneTerm(
            func=mdp.object_out_of_bounds,
            params={"object_cfg": object_cfg, "max_distance": 3.0},
        )

        self.rewards.ee_to_object = RewTerm(
            func=mdp.ee_to_object_exp,
            weight=0.0,
            params={"ee_cfg": ee_cfg, "object_cfg": object_cfg, "std": 0.45},
        )
        self.rewards.object_to_goal = RewTerm(
            func=mdp.object_to_goal_exp,
            weight=0.0,
            params={"object_cfg": object_cfg, "goal_offset": self.goal_offset, "std": 0.75},
        )
        self.rewards.object_velocity = RewTerm(
            func=mdp.object_velocity_xy_exp,
            weight=0.0,
            params={"object_cfg": object_cfg, "target_velocity": self.target_object_velocity, "std": 0.35},
        )
        self.rewards.object_upright = RewTerm(
            func=mdp.object_upright_exp,
            weight=0.0,
            params={"object_cfg": object_cfg, "std": 0.35},
        )

        self.commands.base_velocity.ranges.lin_vel_x = (-0.6, 0.8)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.4, 0.4)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.8, 0.8)
        self.commands.base_velocity.heading_control_stiffness = 0.35


@configclass
class UnitreeB2Z1ReachFlatEnvCfg(UnitreeB2Z1LocoManipFlatEnvCfg):
    """Reach a randomized object with the Z1 while maintaining stable B2 posture."""

    object_size = (0.18, 0.18, 0.18)
    object_mass = 0.4
    object_init_pos = (0.85, 0.0, 0.28)
    object_color = (0.9, 0.35, 0.1)
    goal_offset = (0.85, 0.0, 0.28)
    object_reset_xy = {"x": (-0.30, 0.20), "y": (-0.45, 0.45), "z": (-0.03, 0.12), "yaw": (-3.14, 3.14)}

    def __post_init__(self):
        super().__post_init__()

        self.rewards.ee_to_object.weight = 5.0
        self.rewards.object_upright.weight = 0.25
        self.rewards.track_lin_vel_xy_exp.weight = 0.5
        self.rewards.track_ang_vel_z_exp.weight = 0.25
        self.rewards.stand_still.weight = -0.5
        self.disable_zero_weight_rewards()


@configclass
class UnitreeB2Z1PushFlatEnvCfg(UnitreeB2Z1LocoManipFlatEnvCfg):
    """Push a box forward while coordinating B2 base motion and Z1 contact."""

    object_size = (0.42, 0.42, 0.32)
    object_mass = 8.0
    object_init_pos = (0.95, 0.0, 0.16)
    goal_offset = (1.85, 0.0, 0.16)
    target_object_velocity = (0.35, 0.0)

    def __post_init__(self):
        super().__post_init__()

        self.rewards.ee_to_object.weight = 1.5
        self.rewards.object_velocity.weight = 2.0
        self.rewards.object_to_goal.weight = 3.0
        self.rewards.object_upright.weight = 0.5
        self.rewards.track_lin_vel_xy_exp.weight = 1.0
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.disable_zero_weight_rewards()


@configclass
class UnitreeB2Z1RearrangeFlatEnvCfg(UnitreeB2Z1LocoManipFlatEnvCfg):
    """Rearrange a chair/bin-sized object to a target region, inspired by ALORE-style large-object tasks."""

    object_size = (0.55, 0.45, 0.75)
    object_mass = 14.0
    object_init_pos = (1.05, 0.0, 0.375)
    object_color = (0.25, 0.65, 0.25)
    goal_offset = (2.15, 0.25, 0.375)
    object_reset_xy = {"x": (-0.20, 0.20), "y": (-0.45, 0.45), "yaw": (-0.8, 0.8)}
    target_object_velocity = (0.22, 0.05)

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 18.0
        self.rewards.ee_to_object.weight = 1.0
        self.rewards.object_velocity.weight = 1.0
        self.rewards.object_to_goal.weight = 4.0
        self.rewards.object_upright.weight = 1.0
        self.rewards.track_lin_vel_xy_exp.weight = 0.75
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.disable_zero_weight_rewards()


@configclass
class UnitreeB2Z1PickThrowFlatEnvCfg(UnitreeB2Z1LocoManipFlatEnvCfg):
    """Pick up a light object with the Z1 arm and throw it into a fixed bin."""

    object_size = (0.10, 0.10, 0.10)
    object_mass = 0.18
    object_init_pos = (0.80, 0.0, 0.05)
    object_color = (0.95, 0.52, 0.12)
    goal_offset = (1.55, 0.0, 0.12)
    object_reset_xy = {"x": (-0.12, 0.12), "y": (-0.20, 0.20), "z": (0.0, 0.04), "yaw": (-3.14, 3.14)}
    target_object_velocity = (0.75, 0.0)
    bin_center = (1.55, 0.0, 0.16)
    bin_half_size = (0.30, 0.30, 0.16)

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 10.0
        self.scene.env_spacing = 4.5
        self.scene.bin_floor = _kinematic_box_cfg(
            prim_path="{ENV_REGEX_NS}/BinFloor",
            size=(0.68, 0.68, 0.04),
            init_pos=(1.55, 0.0, 0.02),
            color=(0.08, 0.10, 0.12),
        )
        self.scene.bin_front_wall = _kinematic_box_cfg(
            prim_path="{ENV_REGEX_NS}/BinFrontWall",
            size=(0.68, 0.04, 0.36),
            init_pos=(1.55, 0.36, 0.20),
            color=(0.08, 0.10, 0.12),
        )
        self.scene.bin_back_wall = _kinematic_box_cfg(
            prim_path="{ENV_REGEX_NS}/BinBackWall",
            size=(0.68, 0.04, 0.36),
            init_pos=(1.55, -0.36, 0.20),
            color=(0.08, 0.10, 0.12),
        )
        self.scene.bin_left_wall = _kinematic_box_cfg(
            prim_path="{ENV_REGEX_NS}/BinLeftWall",
            size=(0.04, 0.76, 0.36),
            init_pos=(1.91, 0.0, 0.20),
            color=(0.08, 0.10, 0.12),
        )
        self.scene.bin_right_wall = _kinematic_box_cfg(
            prim_path="{ENV_REGEX_NS}/BinRightWall",
            size=(0.04, 0.76, 0.36),
            init_pos=(1.19, 0.0, 0.20),
            color=(0.08, 0.10, 0.12),
        )

        object_cfg = SceneEntityCfg("object")
        self.events.reset_object = EventTermCfg(
            func=mdp.randomize_pick_throw_scene,
            mode="reset",
            params={
                "object_cfg": object_cfg,
                "bin_asset_names": (
                    "bin_floor",
                    "bin_front_wall",
                    "bin_back_wall",
                    "bin_left_wall",
                    "bin_right_wall",
                ),
                "nominal_bin_center": self.bin_center,
                "object_pose_range": {
                    "x": (-0.22, 0.18),
                    "y": (-0.35, 0.35),
                    "z": (0.0, 0.08),
                    "roll": (-0.15, 0.15),
                    "pitch": (-0.15, 0.15),
                    "yaw": (-3.14, 3.14),
                },
                "bin_pose_range": {
                    "x": (-0.25, 0.35),
                    "y": (-0.45, 0.45),
                    "z": (-0.01, 0.04),
                    "yaw": (0.0, 0.0),
                },
                "object_velocity_range": {
                    "x": (-0.05, 0.05),
                    "y": (-0.05, 0.05),
                    "z": (0.0, 0.05),
                    "roll": (-0.05, 0.05),
                    "pitch": (-0.05, 0.05),
                    "yaw": (-0.05, 0.05),
                },
            },
        )
        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (-0.35, 0.35),
            "y": (-0.30, 0.30),
            "yaw": (-0.75, 0.75),
        }

        self.observations.policy.object_velocity = ObsTerm(
            func=mdp.object_velocity_b,
            params={"object_cfg": object_cfg},
            clip=(-5.0, 5.0),
            scale=0.5,
        )
        self.observations.critic.object_velocity = copy.deepcopy(self.observations.policy.object_velocity)

        self.rewards.ee_to_object.weight = 2.0
        self.rewards.object_lifted = RewTerm(
            func=mdp.object_lifted_exp,
            weight=3.0,
            params={"object_cfg": object_cfg, "target_height": 0.30, "std": 0.18},
        )
        self.rewards.object_throw_velocity = RewTerm(
            func=mdp.object_velocity_towards_goal_exp,
            weight=1.5,
            params={
                "object_cfg": object_cfg,
                "goal_offset": self.goal_offset,
                "target_speed": 0.75,
                "std": 0.50,
            },
        )
        self.rewards.object_to_goal = RewTerm(
            func=mdp.object_to_goal_3d_exp,
            weight=5.0,
            params={"object_cfg": object_cfg, "goal_offset": self.goal_offset, "std": 0.40},
        )
        self.rewards.object_in_bin = RewTerm(
            func=mdp.object_in_bin,
            weight=8.0,
            params={
                "object_cfg": object_cfg,
                "bin_center": self.bin_center,
                "bin_half_size": self.bin_half_size,
            },
        )
        self.rewards.object_velocity.weight = 0.0
        self.rewards.object_upright.weight = 0.0
        self.rewards.track_lin_vel_xy_exp.weight = 0.5
        self.rewards.track_ang_vel_z_exp.weight = 0.25
        self.rewards.stand_still.weight = -0.25
        self.terminations.object_out_of_bounds.params["max_distance"] = 3.5
        self.disable_zero_weight_rewards()


@configclass
class UnitreeB2Z1BadmintonFlatEnvCfg(UnitreeB2Z1LocoManipFlatEnvCfg):
    """Hit a lightweight shuttle proxy over a net into a randomized landing target."""

    object_size = (0.06, 0.06, 0.10)
    object_mass = 0.02
    object_init_pos = (0.75, 0.0, 0.45)
    object_color = (0.98, 0.96, 0.82)
    goal_offset = (2.15, 0.0, 0.08)
    object_reset_xy = {"x": (-0.10, 0.10), "y": (-0.35, 0.35), "z": (-0.10, 0.18), "yaw": (-3.14, 3.14)}
    target_object_velocity = (1.25, 0.0)
    target_center = (2.15, 0.0, 0.08)

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 8.0
        self.scene.env_spacing = 5.0
        self.scene.object = _cone_object_cfg(
            radius=0.035,
            height=0.10,
            mass=self.object_mass,
            init_pos=self.object_init_pos,
            color=self.object_color,
        )
        self.scene.badminton_net = _kinematic_box_cfg(
            prim_path="{ENV_REGEX_NS}/BadmintonNet",
            size=(0.04, 1.50, 0.55),
            init_pos=(1.40, 0.0, 0.30),
            color=(0.16, 0.18, 0.20),
        )
        self.scene.badminton_target = _kinematic_box_cfg(
            prim_path="{ENV_REGEX_NS}/BadmintonTarget",
            size=(0.55, 0.55, 0.02),
            init_pos=self.target_center,
            color=(0.15, 0.60, 0.28),
        )
        self.scene.badminton_racket_face = _kinematic_box_cfg(
            prim_path="{ENV_REGEX_NS}/BadmintonRacketFace",
            size=(0.010, 0.30, 0.22),
            init_pos=(0.95, 0.0, 0.55),
            color=(0.70, 0.86, 1.00),
            opacity=0.22,
        )
        self.scene.badminton_racket_frame_top = _kinematic_box_cfg(
            prim_path="{ENV_REGEX_NS}/BadmintonRacketFrameTop",
            size=(0.032, 0.36, 0.025),
            init_pos=(0.95, 0.0, 0.55),
            color=(0.06, 0.07, 0.08),
        )
        self.scene.badminton_racket_frame_bottom = _kinematic_box_cfg(
            prim_path="{ENV_REGEX_NS}/BadmintonRacketFrameBottom",
            size=(0.032, 0.36, 0.025),
            init_pos=(0.95, 0.0, 0.55),
            color=(0.06, 0.07, 0.08),
        )
        self.scene.badminton_racket_frame_left = _kinematic_box_cfg(
            prim_path="{ENV_REGEX_NS}/BadmintonRacketFrameLeft",
            size=(0.032, 0.025, 0.27),
            init_pos=(0.95, 0.0, 0.55),
            color=(0.06, 0.07, 0.08),
        )
        self.scene.badminton_racket_frame_right = _kinematic_box_cfg(
            prim_path="{ENV_REGEX_NS}/BadmintonRacketFrameRight",
            size=(0.032, 0.025, 0.27),
            init_pos=(0.95, 0.0, 0.55),
            color=(0.06, 0.07, 0.08),
        )
        self.scene.badminton_racket_string_v0 = _kinematic_box_cfg(
            prim_path="{ENV_REGEX_NS}/BadmintonRacketStringV0",
            size=(0.012, 0.006, 0.22),
            init_pos=(0.95, 0.0, 0.55),
            color=(0.95, 0.95, 0.92),
        )
        self.scene.badminton_racket_string_v1 = _kinematic_box_cfg(
            prim_path="{ENV_REGEX_NS}/BadmintonRacketStringV1",
            size=(0.012, 0.006, 0.22),
            init_pos=(0.95, 0.0, 0.55),
            color=(0.95, 0.95, 0.92),
        )
        self.scene.badminton_racket_string_v2 = _kinematic_box_cfg(
            prim_path="{ENV_REGEX_NS}/BadmintonRacketStringV2",
            size=(0.012, 0.006, 0.22),
            init_pos=(0.95, 0.0, 0.55),
            color=(0.95, 0.95, 0.92),
        )
        self.scene.badminton_racket_string_h0 = _kinematic_box_cfg(
            prim_path="{ENV_REGEX_NS}/BadmintonRacketStringH0",
            size=(0.012, 0.30, 0.006),
            init_pos=(0.95, 0.0, 0.55),
            color=(0.95, 0.95, 0.92),
        )
        self.scene.badminton_racket_string_h1 = _kinematic_box_cfg(
            prim_path="{ENV_REGEX_NS}/BadmintonRacketStringH1",
            size=(0.012, 0.30, 0.006),
            init_pos=(0.95, 0.0, 0.55),
            color=(0.95, 0.95, 0.92),
        )
        self.scene.badminton_racket_shaft = _kinematic_box_cfg(
            prim_path="{ENV_REGEX_NS}/BadmintonRacketShaft",
            size=(0.032, 0.032, 0.25),
            init_pos=(0.80, 0.0, 0.40),
            color=(0.06, 0.07, 0.08),
        )
        self.scene.badminton_racket_handle = _kinematic_box_cfg(
            prim_path="{ENV_REGEX_NS}/BadmintonRacketHandle",
            size=(0.055, 0.055, 0.18),
            init_pos=(0.70, 0.0, 0.28),
            color=(0.10, 0.12, 0.16),
        )

        object_cfg = SceneEntityCfg("object")
        ee_cfg = SceneEntityCfg("robot", body_names=["link06"])
        racket_parts = {
            "face": ((0.16, 0.0, 0.02), "badminton_racket_face"),
            "frame_top": ((0.16, 0.0, 0.155), "badminton_racket_frame_top"),
            "frame_bottom": ((0.16, 0.0, -0.115), "badminton_racket_frame_bottom"),
            "frame_left": ((0.16, -0.18, 0.02), "badminton_racket_frame_left"),
            "frame_right": ((0.16, 0.18, 0.02), "badminton_racket_frame_right"),
            "string_v0": ((0.166, -0.10, 0.02), "badminton_racket_string_v0"),
            "string_v1": ((0.166, 0.0, 0.02), "badminton_racket_string_v1"),
            "string_v2": ((0.166, 0.10, 0.02), "badminton_racket_string_v2"),
            "string_h0": ((0.166, 0.0, -0.05), "badminton_racket_string_h0"),
            "string_h1": ((0.166, 0.0, 0.09), "badminton_racket_string_h1"),
            "shaft": ((0.03, 0.0, -0.20), "badminton_racket_shaft"),
            "handle": ((-0.06, 0.0, -0.40), "badminton_racket_handle"),
        }
        for event_suffix, (pos_offset, object_name) in racket_parts.items():
            setattr(
                self.events,
                f"sync_badminton_racket_{event_suffix}",
                EventTermCfg(
                    func=mdp.sync_kinematic_object_to_body,
                    mode="interval",
                    interval_range_s=(0.0, 0.0),
                    params={
                        "object_cfg": SceneEntityCfg(object_name),
                        "body_cfg": ee_cfg,
                        "pos_offset": pos_offset,
                        "rot_offset_rpy": (0.0, 0.0, 0.0),
                    },
                ),
            )
        self.events.reset_object = EventTermCfg(
            func=mdp.randomize_pick_throw_scene,
            mode="reset",
            params={
                "object_cfg": object_cfg,
                "bin_asset_names": ("badminton_target",),
                "nominal_bin_center": self.target_center,
                "object_pose_range": {
                    "x": (-0.18, 0.18),
                    "y": (-0.45, 0.45),
                    "z": (-0.12, 0.18),
                    "roll": (-0.35, 0.35),
                    "pitch": (-0.35, 0.35),
                    "yaw": (-3.14, 3.14),
                },
                "bin_pose_range": {
                    "x": (-0.25, 0.35),
                    "y": (-0.55, 0.55),
                    "z": (0.0, 0.0),
                    "yaw": (0.0, 0.0),
                },
                "object_velocity_range": {
                    "x": (-0.10, 0.10),
                    "y": (-0.10, 0.10),
                    "z": (-0.05, 0.10),
                    "roll": (-0.10, 0.10),
                    "pitch": (-0.10, 0.10),
                    "yaw": (-0.10, 0.10),
                },
            },
        )
        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (-0.30, 0.30),
            "y": (-0.25, 0.25),
            "yaw": (-0.60, 0.60),
        }

        self.observations.policy.object_velocity = ObsTerm(
            func=mdp.object_velocity_b,
            params={"object_cfg": object_cfg},
            clip=(-8.0, 8.0),
            scale=0.4,
        )
        self.observations.critic.object_velocity = copy.deepcopy(self.observations.policy.object_velocity)

        self.rewards.ee_to_object.weight = 2.0
        self.rewards.object_lifted = RewTerm(
            func=mdp.object_lifted_exp,
            weight=2.0,
            params={"object_cfg": object_cfg, "target_height": 0.55, "std": 0.25},
        )
        self.rewards.object_net_clearance = RewTerm(
            func=mdp.object_clearance_exp,
            weight=3.0,
            params={"object_cfg": object_cfg, "obstacle_x": 1.40, "target_height": 0.62, "x_window": 0.28, "std": 0.25},
        )
        self.rewards.object_forward_velocity = RewTerm(
            func=mdp.object_velocity_towards_goal_exp,
            weight=2.0,
            params={
                "object_cfg": object_cfg,
                "goal_offset": self.goal_offset,
                "target_speed": 1.25,
                "std": 0.75,
            },
        )
        self.rewards.object_to_goal = RewTerm(
            func=mdp.object_to_goal_exp,
            weight=5.0,
            params={"object_cfg": object_cfg, "goal_offset": self.goal_offset, "std": 0.55},
        )
        self.rewards.object_velocity.weight = 0.0
        self.rewards.object_upright.weight = 0.0
        self.rewards.track_lin_vel_xy_exp.weight = 0.4
        self.rewards.track_ang_vel_z_exp.weight = 0.2
        self.rewards.stand_still.weight = -0.25
        self.terminations.object_out_of_bounds.params["max_distance"] = 4.0
        self.disable_zero_weight_rewards()
