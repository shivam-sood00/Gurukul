# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.managers import (
    CurriculumTermCfg as CurrTerm,
)
from isaaclab.managers import (
    EventTermCfg,
    SceneEntityCfg,
)
from isaaclab.managers import (
    ObservationTermCfg as ObsTerm,
)
from isaaclab.managers import (
    RewardTermCfg as RewTerm,
)
from isaaclab.managers import (
    TerminationTermCfg as DoneTerm,
)
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import Gurukul.tasks.manager_based.locomotion.velocity.mdp as mdp

from .flat_env_cfg import UnitreeGo2D1ArmFlatEnvCfg
from .rough_env_cfg import _D1_NONADJACENT_SELF_COLLISION_FILTERS

_GO2_CONTACT_FILTER_BODIES = (
    "base",
    "FL_hip",
    "FL_thigh",
    "FL_calf",
    "FL_foot",
    "FR_hip",
    "FR_thigh",
    "FR_calf",
    "FR_foot",
    "Head_upper",
    "Head_lower",
    "RL_hip",
    "RL_thigh",
    "RL_calf",
    "RL_foot",
    "RR_hip",
    "RR_thigh",
    "RR_calf",
    "RR_foot",
)
_D1_HARD_CONTACT_BODIES = ("base_link", "Link1", "Link2", "Link3", "Link4", "Link5", "Link6")
_D1_ARM_OBS_CONTACT_BODIES = ("base_link", "Link1", "Link2", "Link3", "Link4", "Link5", "Link6")
_D1_EE_BODY_NAMES = ("Link7_1", "Link7_2")
_D1_MANIPULATION_PREGRASP_POSE = (0.0, 0.52, 0.17, 0.0, -0.34, 0.0)
_D1_GRIPPER_OPEN_POS = (0.0, 0.0)
_D1_GRIPPER_CLOSED_POS = (0.033, 0.033)
_GO2_FRONT_CLEARANCE_X = 0.38
_TABLE_APPROACH_MARGIN = 0.05


def _set_reward_weight(cfg, name: str, weight: float) -> None:
    term = getattr(cfg.rewards, name, None)
    if term is not None and hasattr(term, "weight"):
        term.weight = weight


def _configure_pick_teacher_observations(cfg) -> None:
    """Add privileged pick-task observations for teacher training and later distillation."""
    object_cfg = SceneEntityCfg("object")
    ee_cfg = SceneEntityCfg("robot", body_names=list(_D1_EE_BODY_NAMES), preserve_order=True)
    root_asset_cfg = SceneEntityCfg("robot", body_names=[cfg.base_link_name])
    foot_contact_cfg = SceneEntityCfg("contact_forces", body_names=[cfg.foot_link_name])
    arm_hard_contact_sensor_names = tuple(
        f"d1_{body_name.lower()}_hard_contact" for body_name in _D1_ARM_OBS_CONTACT_BODIES
    )
    gripper_object_contact_cfg = SceneEntityCfg(
        "left_gripper_object_contact_forces",
        body_names="Link7_1",
    )
    right_gripper_object_contact_cfg = SceneEntityCfg(
        "right_gripper_object_contact_forces",
        body_names="Link7_2",
    )

    lift_margin = max(float(cfg.target_lift_height) - float(cfg.object_init_pos[2]), 0.05)
    cfg.observations.teacher = copy.deepcopy(cfg.observations.policy)
    cfg.observations.teacher.enable_corruption = False
    cfg.observations.teacher.base_height = ObsTerm(
        func=mdp.base_height,
        params={"asset_cfg": root_asset_cfg, "sensor_cfg": None},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    cfg.observations.teacher.feet_air_stance_time = ObsTerm(
        func=mdp.feet_air_stance_time,
        params={"sensor_cfg": foot_contact_cfg},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    cfg.observations.teacher.feet_contact_forces = ObsTerm(
        func=mdp.contact_forces_b,
        params={"sensor_cfg": foot_contact_cfg, "asset_cfg": root_asset_cfg, "normalize": 100.0},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    cfg.observations.teacher.arm_contact_forces = ObsTerm(
        func=mdp.contact_sensor_force_norms,
        params={"sensor_names": arm_hard_contact_sensor_names, "normalize": 60.0},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    cfg.observations.teacher.left_gripper_object_contact_state = ObsTerm(
        func=mdp.gripper_object_contact_state,
        params={
            "sensor_cfg": gripper_object_contact_cfg,
            "force_threshold": 0.35,
            "required_contacts": 1,
        },
        clip=(0.0, 1.0),
        scale=1.0,
    )
    cfg.observations.teacher.right_gripper_object_contact_state = ObsTerm(
        func=mdp.gripper_object_contact_state,
        params={
            "sensor_cfg": right_gripper_object_contact_cfg,
            "force_threshold": 0.35,
            "required_contacts": 1,
        },
        clip=(0.0, 1.0),
        scale=1.0,
    )
    cfg.observations.teacher.ee_to_object_distance = ObsTerm(
        func=mdp.body_to_object_distance,
        params={"body_cfg": ee_cfg, "object_cfg": object_cfg},
        clip=(0.0, 5.0),
        scale=1.0,
    )
    cfg.observations.teacher.object_lift_progress = ObsTerm(
        func=mdp.object_lift_progress,
        params={
            "object_cfg": object_cfg,
            "min_height": cfg.object_init_pos[2],
            "lift_margin": lift_margin,
        },
        clip=(0.0, 1.0),
        scale=1.0,
    )

    cfg.observations.critic = copy.deepcopy(cfg.observations.teacher)
    cfg.observations.critic.enable_corruption = False


def _can_object_cfg(
    *,
    radius: float,
    height: float,
    mass: float,
    init_pos: tuple[float, float, float],
    color: tuple[float, float, float],
) -> RigidObjectCfg:
    return RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CylinderCfg(
            radius=radius,
            height=height,
            axis="Z",
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=4,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                linear_damping=0.02,
                angular_damping=0.02,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            mass_props=sim_utils.MassPropertiesCfg(mass=mass),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.1,
                dynamic_friction=0.9,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=init_pos, rot=(1.0, 0.0, 0.0, 0.0)),
    )


def _validate_approach_geometry(cfg) -> None:
    """Catch table/object/standoff layouts that physically block the base approach."""
    table_half_x = 0.5 * float(cfg.table_size[0])
    object_edge_margin = table_half_x + float(cfg.object_table_offset[0]) - float(cfg.object_radius)
    table_clearance = (
        float(cfg.standoff_distance)
        - _GO2_FRONT_CLEARANCE_X
        - table_half_x
        - float(cfg.object_table_offset[0])
    )
    if object_edge_margin < 0.025:
        raise ValueError(
            f"{cfg.__class__.__name__} object is too close to/outside the table edge: "
            f"margin={object_edge_margin:.3f} m"
        )
    if table_clearance < _TABLE_APPROACH_MARGIN:
        raise ValueError(
            f"{cfg.__class__.__name__} standoff intersects the table approach footprint: "
            f"clearance={table_clearance:.3f} m"
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


def _configure_stationary_manipulation(cfg) -> None:
    """Turn off inherited velocity-command/gait terms for table-top manipulation."""
    cfg.observations.policy.velocity_commands = None
    cfg.observations.critic.velocity_commands = None

    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.commands.base_velocity.rel_standing_envs = 1.0
    cfg.commands.base_velocity.debug_vis = False
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)

    _set_reward_weight(cfg, "track_lin_vel_xy_exp", 0.0)
    _set_reward_weight(cfg, "track_ang_vel_z_exp", 0.0)
    _set_reward_weight(cfg, "stand_still", 0.0)
    _set_reward_weight(cfg, "feet_air_time", 0.0)
    _set_reward_weight(cfg, "feet_air_time_variance", 0.0)
    _set_reward_weight(cfg, "feet_contact_without_cmd", 0.0)
    _set_reward_weight(cfg, "feet_height_body", 0.0)
    _set_reward_weight(cfg, "feet_gait", 0.0)

    cfg.rewards.base_xy_velocity_l2 = RewTerm(func=mdp.base_xy_velocity_l2, weight=-0.5)
    cfg.rewards.flat_orientation_l2.weight = -2.0
    cfg.rewards.base_height_l2.weight = -1.0
    if cfg.rewards.joint_pos_penalty is not None:
        cfg.rewards.joint_pos_penalty.weight = -0.12
        cfg.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = cfg.leg_joint_names
        cfg.rewards.joint_pos_penalty.params["stand_still_scale"] = 3.0
        cfg.rewards.joint_pos_penalty.params["velocity_threshold"] = 0.05
        cfg.rewards.joint_pos_penalty.params["command_threshold"] = 0.02


def _configure_approach_manipulation(cfg, *, standoff_distance: float) -> None:
    """Use generated object-approach commands, then stabilize for arm manipulation."""
    cfg.commands.base_velocity.heading_command = False
    cfg.commands.base_velocity.rel_heading_envs = 0.0
    cfg.commands.base_velocity.rel_standing_envs = 1.0
    cfg.commands.base_velocity.debug_vis = False
    cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
    cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)

    cfg.rewards.track_lin_vel_xy_exp.weight = 0.6
    cfg.rewards.track_ang_vel_z_exp.weight = 0.35
    cfg.rewards.stand_still.weight = 0.0
    cfg.rewards.joint_pos_penalty.weight = -0.08
    cfg.rewards.joint_pos_penalty.params["asset_cfg"].joint_names = cfg.leg_joint_names
    cfg.rewards.joint_pos_penalty.params["stand_still_scale"] = 2.0
    cfg.rewards.joint_pos_penalty.params["velocity_threshold"] = 0.35
    cfg.rewards.joint_pos_penalty.params["command_threshold"] = 0.08
    cfg.rewards.feet_air_time.weight = 0.05
    cfg.rewards.feet_air_time_variance.weight = -0.3
    cfg.rewards.feet_contact_without_cmd.weight = 0.05
    cfg.rewards.feet_height_body.weight = 0.0
    cfg.rewards.feet_gait.weight = 0.15

    cfg.rewards.base_xy_velocity_near_standoff_l2 = RewTerm(
        func=mdp.base_xy_velocity_near_standoff_l2,
        weight=-0.5,
        params={"standoff_distance": standoff_distance, "std": 0.18},
    )
    cfg.rewards.flat_orientation_l2.weight = -1.5
    cfg.rewards.base_height_l2.weight = -0.6


def _configure_collision_safety(cfg, *, support_filter_paths: tuple[str, ...]) -> None:
    """Penalize hard body contacts while leaving gripper-object contact available."""
    cfg.scene.robot.spawn.articulation_props.enabled_self_collisions = True
    # The manipulation sensors below cover these same robot pairs as well as
    # object/support contacts, so remove the inherited pair-only terms instead
    # of charging the same self-contact twice.
    for source_body, _ in _D1_NONADJACENT_SELF_COLLISION_FILTERS:
        inherited_reward_name = f"d1_nonadjacent_contact_{source_body.lower()}"
        if hasattr(cfg.rewards, inherited_reward_name):
            setattr(cfg.rewards, inherited_reward_name, None)
        if hasattr(cfg.scene, inherited_reward_name):
            setattr(cfg.scene, inherited_reward_name, None)

    cfg.rewards.undesired_contacts.weight = -0.6
    cfg.rewards.undesired_contacts.params["sensor_cfg"].body_names = [
        "base",
        ".*_(hip|thigh|calf)",
    ]
    robot_filter_paths = tuple(f"{{ENV_REGEX_NS}}/Robot/{body_name}" for body_name in _GO2_CONTACT_FILTER_BODIES)
    arm_filter_paths = tuple(f"{{ENV_REGEX_NS}}/Robot/d1/{body_name}" for body_name in _D1_HARD_CONTACT_BODIES)
    filter_paths = (*robot_filter_paths, *arm_filter_paths, *support_filter_paths)
    for body_name in _D1_HARD_CONTACT_BODIES:
        sensor_name = f"d1_{body_name.lower()}_hard_contact"
        setattr(
            cfg.scene,
            sensor_name,
            ContactSensorCfg(
                prim_path=f"{{ENV_REGEX_NS}}/Robot/d1/{body_name}",
                history_length=4,
                filter_prim_paths_expr=list(filter_paths),
            ),
        )
        setattr(
            cfg.rewards,
            sensor_name,
            RewTerm(
                func=mdp.self_collision_cost,
                weight=-0.5,
                params={
                    "sensor_cfg": SceneEntityCfg(sensor_name, body_names=body_name),
                    "force_threshold": 8.0,
                },
            ),
        )


def _configure_compact_collision_safety(cfg) -> None:
    """Replace per-link filtered sensors with one low-overhead hard-link sensor."""
    for body_name in _D1_HARD_CONTACT_BODIES:
        sensor_name = f"d1_{body_name.lower()}_hard_contact"
        setattr(cfg.scene, sensor_name, None)
        setattr(cfg.rewards, sensor_name, None)

    # Isaac Lab filtered contact reporting is one-sensor-body-to-many only.
    # A multi-body sensor must therefore use net forces. For this fast-training
    # profile, every contact on the non-gripper D1 links is conservatively
    # undesired; the two finger-object sensors remain separate below.
    cfg.scene.d1_compact_hard_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/d1/(base_link|Link[1-6])",
        history_length=1,
    )
    cfg.rewards.d1_compact_hard_contact = RewTerm(
        func=mdp.self_collision_cost,
        weight=-0.5,
        params={
            "sensor_cfg": SceneEntityCfg(
                "d1_compact_hard_contact",
                body_names=list(_D1_HARD_CONTACT_BODIES),
                preserve_order=True,
            ),
            "force_threshold": 8.0,
        },
    )


def _configure_manipulation_gripper(cfg) -> None:
    """Use a reachable pre-grasp arm pose and a binary open/close gripper action."""
    arm_joint_defaults = dict(zip(cfg.arm_joint_names, _D1_MANIPULATION_PREGRASP_POSE))
    gripper_joint_defaults = dict(zip(cfg.gripper_joint_names, _D1_GRIPPER_OPEN_POS))
    cfg.scene.robot.init_state.joint_pos.update({**arm_joint_defaults, **gripper_joint_defaults})

    cfg.actions.joint_pos.scale.pop(r"^arm_[1-6]_joint$", None)
    cfg.actions.joint_pos.scale.pop(r"^arm_7_[12]_joint$", None)
    cfg.actions.joint_pos.scale.update(
        {
            r"^arm_[14]_joint$": 0.35,
            r"^arm_[235]_joint$": 0.50,
            r"^arm_6_joint$": 0.35,
        }
    )
    cfg.actions.joint_pos.joint_names = cfg.leg_joint_names + cfg.arm_joint_names
    cfg.actions.gripper_binary = mdp.PositiveBinaryJointPositionActionCfg(
        asset_name="robot",
        joint_names=cfg.gripper_joint_names,
        open_command_expr={r"^arm_7_[12]_joint$": _D1_GRIPPER_OPEN_POS[0]},
        close_command_expr={r"^arm_7_[12]_joint$": _D1_GRIPPER_CLOSED_POS[0]},
        threshold=0.0,
        clip={r"^arm_7_[12]_joint$": (0.0, 0.033)},
    )
    cfg.events.randomize_reset_joints.params["asset_cfg"] = SceneEntityCfg(
        "robot", joint_names=cfg.leg_joint_names, preserve_order=True
    )
    cfg.events.reset_manipulator_joints = EventTermCfg(
        func=mdp.reset_joints_by_absolute_uniform,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg(
                "robot",
                joint_names=cfg.arm_joint_names + cfg.gripper_joint_names,
                preserve_order=True,
            ),
            "position_ranges": {
                **{
                    joint_name: (joint_pos - 0.035, joint_pos + 0.035)
                    for joint_name, joint_pos in arm_joint_defaults.items()
                },
                **{
                    joint_name: (joint_pos, joint_pos)
                    for joint_name, joint_pos in gripper_joint_defaults.items()
                },
            },
            "velocity_range": (0.0, 0.0),
        },
    )
    cfg.scene.left_gripper_object_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/d1/Link7_1",
        history_length=4,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )
    cfg.scene.right_gripper_object_contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/d1/Link7_2",
        history_length=4,
        filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
    )


@configclass
class UnitreeGo2D1PickFlatEnvCfg(UnitreeGo2D1ArmFlatEnvCfg):
    """Pick up a nearby can-like cylinder with the Go2-mounted D1 arm."""

    object_radius = 0.035
    object_height = 0.12
    object_mass = 0.08
    table_center = (0.65, 0.0, 0.54)
    table_size = (0.38, 0.54, 0.04)
    table_surface_z = 0.56
    object_table_offset = (-0.12, 0.0)
    object_init_pos = (0.53, 0.0, 0.625)
    object_color = (0.85, 0.06, 0.04)
    target_lift_height = 0.74
    approach_distance_start = 0.65
    approach_distance_end = 0.65
    standoff_distance = 0.55
    manipulation_reach = 0.60
    validate_approach_geometry = True

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 10.0
        self.scene.env_spacing = 2.5
        self.scene.replicate_physics = False
        if self.validate_approach_geometry:
            _validate_approach_geometry(self)
        self.scene.pick_table = _kinematic_box_cfg(
            prim_path="{ENV_REGEX_NS}/PickTable",
            size=self.table_size,
            init_pos=self.table_center,
            color=(0.32, 0.34, 0.36),
        )
        self.scene.object = _can_object_cfg(
            radius=self.object_radius,
            height=self.object_height,
            mass=self.object_mass,
            init_pos=self.object_init_pos,
            color=self.object_color,
        )
        _configure_manipulation_gripper(self)
        _configure_collision_safety(self, support_filter_paths=("{ENV_REGEX_NS}/PickTable",))

        object_cfg = SceneEntityCfg("object")
        table_cfg = SceneEntityCfg("pick_table")
        ee_cfg = SceneEntityCfg("robot", body_names=list(_D1_EE_BODY_NAMES), preserve_order=True)
        gripper_joint_cfg = SceneEntityCfg("robot", joint_names=self.gripper_joint_names)
        gripper_object_contact_cfg = SceneEntityCfg("left_gripper_object_contact_forces", body_names="Link7_1")
        right_gripper_object_contact_cfg = SceneEntityCfg("right_gripper_object_contact_forces", body_names="Link7_2")

        self.events.reset_object = EventTermCfg(
            func=mdp.reset_pick_approach_scene,
            mode="reset",
            params={
                "object_cfg": object_cfg,
                "table_asset_name": "pick_table",
                "table_center": self.table_center,
                "object_center_z": self.object_init_pos[2],
                "object_table_offset": self.object_table_offset,
                "object_distance_range": (self.approach_distance_start, self.approach_distance_end),
                "object_pose_range": {
                    "x": (-0.020, 0.020),
                    "y": (-0.045, 0.045),
                    "z": (0.0, 0.004),
                    "roll": (-0.035, 0.035),
                    "pitch": (-0.035, 0.035),
                    "yaw": (-1.57, 1.57),
                },
                "object_velocity_range": {
                    "x": (-0.03, 0.03),
                    "y": (-0.03, 0.03),
                    "z": (0.0, 0.03),
                    "roll": (-0.05, 0.05),
                    "pitch": (-0.05, 0.05),
                    "yaw": (-0.05, 0.05),
                },
            },
        )
        self.events.randomize_object_scale = EventTermCfg(
            func=mdp.randomize_rigid_body_scale,
            mode="prestartup",
            params={"asset_cfg": object_cfg, "scale_range": (0.90, 1.10)},
        )
        self.events.randomize_object_material = EventTermCfg(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("object", body_names=".*"),
                "static_friction_range": (0.90, 1.30),
                "dynamic_friction_range": (0.60, 0.85),
                "restitution_range": (0.0, 0.05),
                "num_buckets": 64,
            },
        )
        self.events.randomize_object_mass = EventTermCfg(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("object", body_names=".*"),
                "mass_distribution_params": (0.65, 1.35),
                "operation": "scale",
                "recompute_inertia": True,
            },
        )
        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (-0.01, 0.01),
            "y": (-0.01, 0.01),
            "yaw": (-0.03, 0.03),
        }
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None
        self.curriculum.go2_d1_pick_approach = None

        self.observations.policy.object_position = ObsTerm(
            func=mdp.object_position_b,
            params={"object_cfg": object_cfg},
            clip=(-5.0, 5.0),
            scale=1.0,
        )
        self.observations.policy.object_geometry_points = ObsTerm(
            func=mdp.object_cuboid_points_b,
            params={
                "object_cfg": object_cfg,
                "half_size": (self.object_radius, self.object_radius, 0.5 * self.object_height),
            },
            clip=(-5.0, 5.0),
            scale=1.0,
        )
        self.observations.policy.table_geometry_points = ObsTerm(
            func=mdp.object_cuboid_points_b,
            params={
                "object_cfg": table_cfg,
                "half_size": tuple(0.5 * float(size) for size in self.table_size),
            },
            clip=(-5.0, 5.0),
            scale=1.0,
        )
        self.observations.policy.base_to_object_standoff = None
        self.observations.policy.object_velocity = ObsTerm(
            func=mdp.object_velocity_b,
            params={"object_cfg": object_cfg},
            clip=(-5.0, 5.0),
            scale=0.5,
        )
        self.observations.policy.object_height = ObsTerm(
            func=mdp.object_height,
            params={"object_cfg": object_cfg},
            clip=(0.0, 1.0),
            scale=2.0,
        )
        self.observations.policy.object_up_axis = ObsTerm(
            func=mdp.object_up_axis_b,
            params={"object_cfg": object_cfg},
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        self.observations.policy.ee_position = ObsTerm(
            func=mdp.body_position_b,
            params={"asset_cfg": ee_cfg},
            clip=(-5.0, 5.0),
            scale=1.0,
        )
        self.observations.policy.ee_to_object = ObsTerm(
            func=mdp.body_to_object_b,
            params={"body_cfg": ee_cfg, "object_cfg": object_cfg},
            clip=(-5.0, 5.0),
            scale=1.0,
        )
        self.observations.policy.loco_manipulation_skill = ObsTerm(
            func=mdp.loco_manipulation_skill,
            params={
                "manipulation_reach": self.manipulation_reach,
                "object_cfg": object_cfg,
            },
            clip=(0.0, 1.0),
            scale=1.0,
        )

        self.observations.critic.object_position = copy.deepcopy(self.observations.policy.object_position)
        self.observations.critic.object_geometry_points = copy.deepcopy(
            self.observations.policy.object_geometry_points
        )
        self.observations.critic.table_geometry_points = copy.deepcopy(
            self.observations.policy.table_geometry_points
        )
        self.observations.critic.base_to_object_standoff = None
        self.observations.critic.object_velocity = copy.deepcopy(self.observations.policy.object_velocity)
        self.observations.critic.object_height = copy.deepcopy(self.observations.policy.object_height)
        self.observations.critic.object_up_axis = copy.deepcopy(self.observations.policy.object_up_axis)
        self.observations.critic.ee_position = copy.deepcopy(self.observations.policy.ee_position)
        self.observations.critic.ee_to_object = copy.deepcopy(self.observations.policy.ee_to_object)
        self.observations.critic.loco_manipulation_skill = copy.deepcopy(
            self.observations.policy.loco_manipulation_skill
        )
        self.observations.policy.history_length = 5
        self.observations.policy.flatten_history_dim = True
        self.observations.critic.history_length = 5
        self.observations.critic.flatten_history_dim = True

        self.terminations.object_out_of_bounds = DoneTerm(
            func=mdp.object_out_of_bounds,
            params={"object_cfg": object_cfg, "max_distance": 1.5},
        )

        self.rewards.base_to_object_standoff = None
        self.rewards.base_faces_object = None
        self.rewards.ee_to_object = RewTerm(
            func=mdp.ee_to_object_in_manipulation_reach_tanh,
            weight=5.0,
            params={
                "ee_cfg": ee_cfg,
                "object_cfg": object_cfg,
                "std": 0.28,
                "manipulation_reach": self.manipulation_reach,
            },
        )
        self.rewards.base_approach_outside_arm_reach = RewTerm(
            func=mdp.base_to_object_standoff_outside_manipulation_reach_exp,
            weight=3.0,
            params={
                "standoff_distance": self.standoff_distance,
                "manipulation_reach": self.manipulation_reach,
                "std": 0.20,
                "object_cfg": object_cfg,
            },
        )
        self.rewards.gripper_close_near_object = RewTerm(
            func=mdp.gripper_close_near_object,
            weight=0.75,
            params={
                "ee_cfg": ee_cfg,
                "gripper_cfg": gripper_joint_cfg,
                "object_cfg": object_cfg,
                "near_std": 0.13,
                "open_joint_pos": _D1_GRIPPER_OPEN_POS,
                "closed_joint_pos": _D1_GRIPPER_CLOSED_POS,
            },
        )
        self.rewards.gripper_object_contact = RewTerm(
            func=mdp.gripper_object_contact,
            weight=2.0,
            params={
                "sensor_cfg": gripper_object_contact_cfg,
                "right_sensor_cfg": right_gripper_object_contact_cfg,
                "force_threshold": 0.35,
                "required_contacts": 2,
            },
        )
        self.rewards.object_lifted = RewTerm(
            func=mdp.object_lifted_above_height_exp,
            weight=1.5,
            params={
                "object_cfg": object_cfg,
                "min_height": self.object_init_pos[2],
                "target_height": self.target_lift_height,
                "std": 0.10,
            },
        )
        self.rewards.object_lifted_near_ee = RewTerm(
            func=mdp.object_lifted_near_ee_above_height_exp,
            weight=3.0,
            params={
                "ee_cfg": ee_cfg,
                "object_cfg": object_cfg,
                "min_height": self.object_init_pos[2],
                "target_height": self.target_lift_height,
                "std": 0.18,
            },
        )
        self.rewards.object_lifted_with_gripper_contact = RewTerm(
            func=mdp.object_lifted_with_gripper_contact_exp,
            weight=10.0,
            params={
                "sensor_cfg": gripper_object_contact_cfg,
                "right_sensor_cfg": right_gripper_object_contact_cfg,
                "object_cfg": object_cfg,
                "force_threshold": 0.35,
                "min_height": self.object_init_pos[2],
                "target_height": self.target_lift_height,
                "std": 0.10,
            },
        )
        self.rewards.object_hold_lifted = RewTerm(
            func=mdp.object_hold_lifted_with_gripper_contact,
            weight=6.0,
            params={
                "sensor_cfg": gripper_object_contact_cfg,
                "right_sensor_cfg": right_gripper_object_contact_cfg,
                "object_cfg": object_cfg,
                "force_threshold": 0.35,
                "min_height": self.object_init_pos[2],
                "lift_margin": self.target_lift_height - self.object_init_pos[2],
                "velocity_std": 0.35,
            },
        )
        self.rewards.object_lifted_without_gripper_contact = RewTerm(
            func=mdp.object_lifted_without_gripper_contact,
            weight=-8.0,
            params={
                "sensor_cfg": gripper_object_contact_cfg,
                "right_sensor_cfg": right_gripper_object_contact_cfg,
                "object_cfg": object_cfg,
                "force_threshold": 0.35,
                "min_height": self.object_init_pos[2],
                "lift_margin": self.target_lift_height - self.object_init_pos[2],
            },
        )
        self.rewards.object_xy_motion_before_lift = RewTerm(
            func=mdp.object_xy_motion_before_lift,
            weight=-1.0,
            params={
                "object_cfg": object_cfg,
                "min_height": self.object_init_pos[2],
                "lift_margin": self.target_lift_height - self.object_init_pos[2],
            },
        )
        self.rewards.object_vertical = RewTerm(
            func=mdp.object_vertical_exp,
            weight=0.5,
            params={"object_cfg": object_cfg, "std": 0.45},
        )
        self.rewards.object_still_xy = RewTerm(
            func=mdp.object_velocity_xy_exp,
            weight=0.0,
            params={"object_cfg": object_cfg, "target_velocity": (0.0, 0.0), "std": 0.35},
        )
        self.rewards.arm_ready_before_standoff = None
        _configure_stationary_manipulation(self)
        self.rewards.base_xy_velocity_l2 = RewTerm(
            func=mdp.base_xy_velocity_in_manipulation_reach_l2,
            weight=-1.0,
            params={
                "manipulation_reach": self.manipulation_reach,
                "object_cfg": object_cfg,
            },
        )
        self.rewards.base_xy_drift_in_arm_reach = RewTerm(
            func=mdp.base_xy_drift_in_manipulation_reach_l2,
            weight=-1.5,
            params={
                "manipulation_reach": self.manipulation_reach,
                "drift_std": 0.06,
                "object_cfg": object_cfg,
            },
        )
        self.disable_zero_weight_rewards()


@configclass
class UnitreeGo2D1PickTeacherFlatEnvCfg(UnitreeGo2D1PickFlatEnvCfg):
    """Privileged teacher variant of the stationary Go2+D1 pick task."""

    def __post_init__(self):
        super().__post_init__()

        _configure_pick_teacher_observations(self)


@configclass
class UnitreeGo2D1PickPlaceFlatEnvCfg(UnitreeGo2D1ArmFlatEnvCfg):
    """Pick up a can-like cylinder with the Go2-mounted D1 arm and place it in a shallow tray."""

    object_radius = 0.035
    object_height = 0.12
    object_mass = 0.08
    table_center = (0.72, 0.0, 0.54)
    table_size = (0.38, 0.62, 0.04)
    table_surface_z = 0.56
    object_init_pos = (0.64, -0.08, 0.625)
    object_color = (0.85, 0.06, 0.04)
    goal_offset = (0.64, 0.14, 0.64)
    tray_half_size = (0.09, 0.09, 0.08)
    tray_wall_thickness = 0.015
    tray_wall_height = 0.08
    tray_floor_thickness = 0.02
    target_lift_height = 0.73
    approach_distance_start = 0.72
    approach_distance_end = 1.05
    object_table_offset = (-0.08, -0.08)
    standoff_distance = 0.55
    validate_approach_geometry = True

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 22.0
        self.scene.env_spacing = 3.5
        self.scene.replicate_physics = False
        if self.validate_approach_geometry:
            _validate_approach_geometry(self)
        self.scene.pick_table = _kinematic_box_cfg(
            prim_path="{ENV_REGEX_NS}/PickTable",
            size=self.table_size,
            init_pos=self.table_center,
            color=(0.32, 0.34, 0.36),
        )
        self.scene.object = _can_object_cfg(
            radius=self.object_radius,
            height=self.object_height,
            mass=self.object_mass,
            init_pos=self.object_init_pos,
            color=self.object_color,
        )
        _configure_manipulation_gripper(self)
        tray_outer_x = 2.0 * (self.tray_half_size[0] + self.tray_wall_thickness)
        tray_outer_y = 2.0 * (self.tray_half_size[1] + self.tray_wall_thickness)
        tray_wall_z = self.table_surface_z + self.tray_floor_thickness + 0.5 * self.tray_wall_height
        self.scene.place_tray_floor = _kinematic_box_cfg(
            prim_path="{ENV_REGEX_NS}/PlaceTrayFloor",
            size=(tray_outer_x, tray_outer_y, self.tray_floor_thickness),
            init_pos=(
                self.goal_offset[0],
                self.goal_offset[1],
                self.table_surface_z + 0.5 * self.tray_floor_thickness,
            ),
            color=(0.08, 0.10, 0.12),
        )
        self.scene.place_tray_front_wall = _kinematic_box_cfg(
            prim_path="{ENV_REGEX_NS}/PlaceTrayFrontWall",
            size=(tray_outer_x, self.tray_wall_thickness, self.tray_wall_height),
            init_pos=(
                self.goal_offset[0],
                self.goal_offset[1] + self.tray_half_size[1] + 0.5 * self.tray_wall_thickness,
                tray_wall_z,
            ),
            color=(0.08, 0.10, 0.12),
        )
        self.scene.place_tray_back_wall = _kinematic_box_cfg(
            prim_path="{ENV_REGEX_NS}/PlaceTrayBackWall",
            size=(tray_outer_x, self.tray_wall_thickness, self.tray_wall_height),
            init_pos=(
                self.goal_offset[0],
                self.goal_offset[1] - self.tray_half_size[1] - 0.5 * self.tray_wall_thickness,
                tray_wall_z,
            ),
            color=(0.08, 0.10, 0.12),
        )
        self.scene.place_tray_left_wall = _kinematic_box_cfg(
            prim_path="{ENV_REGEX_NS}/PlaceTrayLeftWall",
            size=(self.tray_wall_thickness, tray_outer_y, self.tray_wall_height),
            init_pos=(
                self.goal_offset[0] + self.tray_half_size[0] + 0.5 * self.tray_wall_thickness,
                self.goal_offset[1],
                tray_wall_z,
            ),
            color=(0.08, 0.10, 0.12),
        )
        self.scene.place_tray_right_wall = _kinematic_box_cfg(
            prim_path="{ENV_REGEX_NS}/PlaceTrayRightWall",
            size=(self.tray_wall_thickness, tray_outer_y, self.tray_wall_height),
            init_pos=(
                self.goal_offset[0] - self.tray_half_size[0] - 0.5 * self.tray_wall_thickness,
                self.goal_offset[1],
                tray_wall_z,
            ),
            color=(0.08, 0.10, 0.12),
        )
        _configure_collision_safety(
            self,
            support_filter_paths=(
                "{ENV_REGEX_NS}/PickTable",
                "{ENV_REGEX_NS}/PlaceTrayFloor",
                "{ENV_REGEX_NS}/PlaceTrayFrontWall",
                "{ENV_REGEX_NS}/PlaceTrayBackWall",
                "{ENV_REGEX_NS}/PlaceTrayLeftWall",
                "{ENV_REGEX_NS}/PlaceTrayRightWall",
            ),
        )

        object_cfg = SceneEntityCfg("object")
        ee_cfg = SceneEntityCfg("robot", body_names=list(_D1_EE_BODY_NAMES), preserve_order=True)
        gripper_joint_cfg = SceneEntityCfg("robot", joint_names=self.gripper_joint_names, preserve_order=True)
        gripper_object_contact_cfg = SceneEntityCfg("left_gripper_object_contact_forces", body_names="Link7_1")
        right_gripper_object_contact_cfg = SceneEntityCfg(
            "right_gripper_object_contact_forces", body_names="Link7_2"
        )

        self.events.reset_object = EventTermCfg(
            func=mdp.reset_pick_place_approach_scene,
            mode="reset",
            params={
                "object_cfg": object_cfg,
                "table_asset_name": "pick_table",
                "table_center": self.table_center,
                "object_center_z": self.object_init_pos[2],
                "object_table_offset": self.object_table_offset,
                "object_distance_range": (self.approach_distance_start, self.approach_distance_end),
                "bin_asset_names": (
                    "place_tray_floor",
                    "place_tray_front_wall",
                    "place_tray_back_wall",
                    "place_tray_left_wall",
                    "place_tray_right_wall",
                ),
                "nominal_bin_center": self.goal_offset,
                "object_pose_range": {
                    "x": (-0.025, 0.025),
                    "y": (-0.055, 0.055),
                    "z": (0.0, 0.005),
                    "roll": (-0.05, 0.05),
                    "pitch": (-0.05, 0.05),
                    "yaw": (-1.57, 1.57),
                },
                "bin_pose_range": {
                    "x": (-0.045, 0.045),
                    "y": (-0.080, 0.080),
                    "z": (0.0, 0.008),
                    "yaw": (0.0, 0.0),
                },
                "object_velocity_range": {
                    "x": (-0.03, 0.03),
                    "y": (-0.03, 0.03),
                    "z": (0.0, 0.03),
                    "roll": (-0.05, 0.05),
                    "pitch": (-0.05, 0.05),
                    "yaw": (-0.05, 0.05),
                },
            },
        )
        self.events.randomize_object_scale = EventTermCfg(
            func=mdp.randomize_rigid_body_scale,
            mode="prestartup",
            params={"asset_cfg": object_cfg, "scale_range": (0.90, 1.10)},
        )
        self.events.randomize_object_material = EventTermCfg(
            func=mdp.randomize_rigid_body_material,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("object", body_names=".*"),
                "static_friction_range": (0.90, 1.30),
                "dynamic_friction_range": (0.60, 0.85),
                "restitution_range": (0.0, 0.05),
                "num_buckets": 64,
            },
        )
        self.events.randomize_object_mass = EventTermCfg(
            func=mdp.randomize_rigid_body_mass,
            mode="startup",
            params={
                "asset_cfg": SceneEntityCfg("object", body_names=".*"),
                "mass_distribution_params": (0.65, 1.35),
                "operation": "scale",
                "recompute_inertia": True,
            },
        )
        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (-0.02, 0.02),
            "y": (-0.02, 0.02),
            "yaw": (-0.06, 0.06),
        }
        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None
        self.curriculum.go2_d1_pick_place_approach = CurrTerm(
            func=mdp.go2_d1_pick_approach_curriculum,
            params={
                "distance_steps": (100_000, 360_000),
                "object_distance_start": self.approach_distance_start,
                "object_distance_end": self.approach_distance_end,
                "standoff_distance": self.standoff_distance,
                "stop_radius": 0.08,
                "max_lin_speed": 0.45,
                "max_lat_speed": 0.28,
                "max_yaw_rate": 0.75,
                "lin_gain": 1.2,
                "lat_gain": 1.3,
                "yaw_gain": 1.7,
                "command_name": "base_velocity",
                "object_cfg": object_cfg,
                "performance_based": True,
                "performance_threshold": 0.72,
                "performance_lower_threshold": 0.45,
                "performance_progress_step": 0.015,
                "performance_regress_step": 0.004,
                "min_steps_before_performance_update": 20_000,
            },
        )

        self.observations.policy.object_position = ObsTerm(
            func=mdp.object_position_b,
            params={"object_cfg": object_cfg},
            clip=(-5.0, 5.0),
            scale=1.0,
        )
        self.observations.policy.object_geometry_points = ObsTerm(
            func=mdp.object_cuboid_points_b,
            params={
                "object_cfg": object_cfg,
                "half_size": (self.object_radius, self.object_radius, 0.5 * self.object_height),
            },
            clip=(-5.0, 5.0),
            scale=1.0,
        )
        self.observations.policy.object_goal_position = ObsTerm(
            func=mdp.object_goal_position_b,
            params={"goal_offset": self.goal_offset},
            clip=(-5.0, 5.0),
            scale=1.0,
        )
        self.observations.policy.base_to_object_standoff = ObsTerm(
            func=mdp.base_to_object_standoff_b,
            params={"object_cfg": object_cfg, "standoff_distance": self.standoff_distance},
            clip=(-5.0, 5.0),
            scale=1.0,
        )
        self.observations.policy.object_to_goal = ObsTerm(
            func=mdp.object_to_goal_b,
            params={"goal_offset": self.goal_offset, "object_cfg": object_cfg},
            clip=(-5.0, 5.0),
            scale=1.0,
        )
        self.observations.policy.pick_place_stage = ObsTerm(
            func=mdp.pick_place_stage_state,
            params={
                "goal_offset": self.goal_offset,
                "bin_half_size": self.tray_half_size,
                "min_height": self.object_init_pos[2],
                "lift_margin": 0.08,
                "standoff_distance": self.standoff_distance,
                "ee_cfg": ee_cfg,
                "object_cfg": object_cfg,
                "sensor_cfg": gripper_object_contact_cfg,
                "right_sensor_cfg": right_gripper_object_contact_cfg,
                "force_threshold": 0.35,
                "required_contacts": 2,
            },
            clip=(0.0, 1.0),
            scale=1.0,
        )
        self.observations.policy.object_velocity = ObsTerm(
            func=mdp.object_velocity_b,
            params={"object_cfg": object_cfg},
            clip=(-5.0, 5.0),
            scale=0.5,
        )
        self.observations.policy.object_up_axis = ObsTerm(
            func=mdp.object_up_axis_b,
            params={"object_cfg": object_cfg},
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        self.observations.policy.ee_position = ObsTerm(
            func=mdp.body_position_b,
            params={"asset_cfg": ee_cfg},
            clip=(-5.0, 5.0),
            scale=1.0,
        )

        self.observations.critic.object_position = copy.deepcopy(self.observations.policy.object_position)
        self.observations.critic.object_geometry_points = copy.deepcopy(
            self.observations.policy.object_geometry_points
        )
        self.observations.critic.object_goal_position = copy.deepcopy(self.observations.policy.object_goal_position)
        self.observations.critic.base_to_object_standoff = copy.deepcopy(
            self.observations.policy.base_to_object_standoff
        )
        self.observations.critic.object_to_goal = copy.deepcopy(self.observations.policy.object_to_goal)
        self.observations.critic.pick_place_stage = copy.deepcopy(self.observations.policy.pick_place_stage)
        self.observations.critic.object_velocity = copy.deepcopy(self.observations.policy.object_velocity)
        self.observations.critic.object_up_axis = copy.deepcopy(self.observations.policy.object_up_axis)
        self.observations.critic.ee_position = copy.deepcopy(self.observations.policy.ee_position)
        self.observations.policy.history_length = 5
        self.observations.policy.flatten_history_dim = True
        self.observations.critic.history_length = 5
        self.observations.critic.flatten_history_dim = True

        self.terminations.object_out_of_bounds = DoneTerm(
            func=mdp.object_out_of_bounds,
            params={"object_cfg": object_cfg, "max_distance": 2.5},
        )
        self.terminations.object_placed = DoneTerm(
            func=mdp.object_released_in_bin_after_grasp,
            params={
                "object_cfg": object_cfg,
                "bin_center": self.goal_offset,
                "bin_half_size": self.tray_half_size,
                "min_height": self.object_init_pos[2],
                "lift_margin": 0.08,
                "sensor_cfg": gripper_object_contact_cfg,
                "right_sensor_cfg": right_gripper_object_contact_cfg,
                "force_threshold": 0.35,
                "required_contacts": 2,
                "max_speed": 0.20,
            },
        )

        self.rewards.base_to_object_standoff = RewTerm(
            func=mdp.base_to_object_standoff_exp,
            weight=1.0,
            params={"object_cfg": object_cfg, "standoff_distance": self.standoff_distance, "std": 0.32},
        )
        self.rewards.base_faces_object = RewTerm(
            func=mdp.base_faces_object_exp,
            weight=0.25,
            params={"object_cfg": object_cfg, "std": 0.50},
        )
        self.rewards.ee_to_object = RewTerm(
            func=mdp.ee_to_object_after_standoff_tanh,
            weight=2.5,
            params={
                "ee_cfg": ee_cfg,
                "object_cfg": object_cfg,
                "std": 0.35,
                "standoff_distance": self.standoff_distance,
                "standoff_std": 0.26,
            },
        )
        self.rewards.gripper_close_near_object = RewTerm(
            func=mdp.gripper_close_near_object,
            weight=0.75,
            params={
                "ee_cfg": ee_cfg,
                "gripper_cfg": gripper_joint_cfg,
                "object_cfg": object_cfg,
                "near_std": 0.13,
                "open_joint_pos": _D1_GRIPPER_OPEN_POS,
                "closed_joint_pos": _D1_GRIPPER_CLOSED_POS,
            },
        )
        self.rewards.gripper_object_contact = RewTerm(
            func=mdp.gripper_object_contact,
            weight=2.0,
            params={
                "sensor_cfg": gripper_object_contact_cfg,
                "right_sensor_cfg": right_gripper_object_contact_cfg,
                "force_threshold": 0.35,
                "required_contacts": 2,
            },
        )
        self.rewards.object_lifted = RewTerm(
            func=mdp.object_lifted_after_standoff_exp,
            weight=0.75,
            params={
                "object_cfg": object_cfg,
                "min_height": self.object_init_pos[2],
                "target_height": self.target_lift_height,
                "std": 0.12,
                "standoff_distance": self.standoff_distance,
                "standoff_std": 0.26,
            },
        )
        self.rewards.object_lifted_near_ee = RewTerm(
            func=mdp.object_lifted_near_ee_after_standoff_exp,
            weight=1.0,
            params={
                "ee_cfg": ee_cfg,
                "object_cfg": object_cfg,
                "min_height": self.object_init_pos[2],
                "target_height": self.target_lift_height,
                "std": 0.18,
                "standoff_distance": self.standoff_distance,
                "standoff_std": 0.26,
            },
        )
        self.rewards.object_lifted_with_gripper_contact = RewTerm(
            func=mdp.object_lifted_with_gripper_contact_exp,
            weight=10.0,
            params={
                "sensor_cfg": gripper_object_contact_cfg,
                "right_sensor_cfg": right_gripper_object_contact_cfg,
                "object_cfg": object_cfg,
                "force_threshold": 0.35,
                "min_height": self.object_init_pos[2],
                "target_height": self.target_lift_height,
                "std": 0.10,
            },
        )
        self.rewards.object_hold_lifted = RewTerm(
            func=mdp.object_hold_lifted_with_gripper_contact,
            weight=5.0,
            params={
                "sensor_cfg": gripper_object_contact_cfg,
                "right_sensor_cfg": right_gripper_object_contact_cfg,
                "object_cfg": object_cfg,
                "force_threshold": 0.35,
                "min_height": self.object_init_pos[2],
                "lift_margin": 0.08,
                "velocity_std": 0.35,
            },
        )
        self.rewards.object_lifted_without_gripper_contact = RewTerm(
            func=mdp.object_lifted_without_gripper_contact,
            weight=-8.0,
            params={
                "sensor_cfg": gripper_object_contact_cfg,
                "right_sensor_cfg": right_gripper_object_contact_cfg,
                "object_cfg": object_cfg,
                "force_threshold": 0.35,
                "min_height": self.object_init_pos[2],
                "lift_margin": 0.08,
            },
        )
        self.rewards.object_xy_motion_before_lift = RewTerm(
            func=mdp.object_xy_motion_before_lift,
            weight=-1.0,
            params={
                "object_cfg": object_cfg,
                "min_height": self.object_init_pos[2],
                "lift_margin": 0.08,
            },
        )
        self.rewards.object_vertical = RewTerm(
            func=mdp.object_vertical_exp,
            weight=0.5,
            params={"object_cfg": object_cfg, "std": 0.45},
        )
        self.rewards.object_to_goal = RewTerm(
            func=mdp.object_to_goal_after_lift_3d_exp,
            weight=6.0,
            params={
                "object_cfg": object_cfg,
                "goal_offset": self.goal_offset,
                "std": 0.24,
                "min_height": self.object_init_pos[2],
                "lift_margin": 0.08,
                "sensor_cfg": gripper_object_contact_cfg,
                "right_sensor_cfg": right_gripper_object_contact_cfg,
                "force_threshold": 0.35,
                "required_contacts": 2,
            },
        )
        self.rewards.object_velocity_to_goal = RewTerm(
            func=mdp.object_velocity_towards_goal_after_lift_exp,
            weight=1.0,
            params={
                "object_cfg": object_cfg,
                "goal_offset": self.goal_offset,
                "target_speed": 0.25,
                "std": 0.30,
                "min_height": self.object_init_pos[2],
                "lift_margin": 0.08,
                "sensor_cfg": gripper_object_contact_cfg,
                "right_sensor_cfg": right_gripper_object_contact_cfg,
                "force_threshold": 0.35,
                "required_contacts": 2,
            },
        )
        self.rewards.object_in_tray = RewTerm(
            func=mdp.object_in_bin_after_lift,
            weight=8.0,
            params={
                "object_cfg": object_cfg,
                "bin_center": self.goal_offset,
                "bin_half_size": self.tray_half_size,
                "min_height": self.object_init_pos[2],
                "lift_margin": 0.08,
                "sensor_cfg": gripper_object_contact_cfg,
                "right_sensor_cfg": right_gripper_object_contact_cfg,
                "force_threshold": 0.35,
                "required_contacts": 2,
            },
        )
        self.rewards.gripper_open_in_tray = RewTerm(
            func=mdp.gripper_open_after_grasp_near_goal,
            weight=2.0,
            params={
                "object_cfg": object_cfg,
                "bin_center": self.goal_offset,
                "bin_half_size": self.tray_half_size,
                "min_height": self.object_init_pos[2],
                "lift_margin": 0.08,
                "gripper_cfg": gripper_joint_cfg,
                "sensor_cfg": gripper_object_contact_cfg,
                "right_sensor_cfg": right_gripper_object_contact_cfg,
                "force_threshold": 0.35,
                "required_contacts": 2,
                "open_joint_pos": _D1_GRIPPER_OPEN_POS,
                "closed_joint_pos": _D1_GRIPPER_CLOSED_POS,
            },
        )
        self.rewards.object_released_in_tray = RewTerm(
            func=mdp.object_released_in_bin_after_grasp,
            weight=40.0,
            params={
                "object_cfg": SceneEntityCfg("object"),
                "bin_center": self.goal_offset,
                "bin_half_size": self.tray_half_size,
                "min_height": self.object_init_pos[2],
                "lift_margin": 0.08,
                "sensor_cfg": SceneEntityCfg(
                    "left_gripper_object_contact_forces", body_names="Link7_1"
                ),
                "right_sensor_cfg": SceneEntityCfg(
                    "right_gripper_object_contact_forces", body_names="Link7_2"
                ),
                "force_threshold": 0.35,
                "required_contacts": 2,
                "max_speed": 0.20,
            },
        )
        self.rewards.pick_place_stage_progress = RewTerm(
            func=mdp.pick_place_stage_progress,
            weight=2.0,
            params={
                "goal_offset": self.goal_offset,
                "bin_half_size": self.tray_half_size,
                "min_height": self.object_init_pos[2],
                "lift_margin": 0.08,
                "standoff_distance": self.standoff_distance,
                "ee_cfg": ee_cfg,
                "object_cfg": object_cfg,
                "sensor_cfg": gripper_object_contact_cfg,
                "right_sensor_cfg": right_gripper_object_contact_cfg,
                "force_threshold": 0.35,
                "required_contacts": 2,
            },
        )
        self.rewards.arm_ready_before_standoff = RewTerm(
            func=mdp.arm_joint_deviation_before_standoff_l2,
            weight=-0.30,
            params={
                "object_cfg": object_cfg,
                "standoff_distance": self.standoff_distance,
                "standoff_std": 0.26,
                "asset_cfg": SceneEntityCfg("robot", joint_names=self.arm_joint_names),
            },
        )
        _configure_approach_manipulation(self, standoff_distance=self.standoff_distance)
        self.disable_zero_weight_rewards()


@configclass
class UnitreeGo2D1StationaryPickPlaceFlatEnvCfg(UnitreeGo2D1PickPlaceFlatEnvCfg):
    """Stationary arm-only pick-place while the Go2 base keeps balance."""

    table_center = (0.65, 0.0, 0.54)
    table_size = (0.38, 0.62, 0.04)
    object_init_pos = (0.53, -0.08, 0.625)
    goal_offset = (0.53, 0.14, 0.64)
    approach_distance_start = 0.65
    approach_distance_end = 0.65
    object_table_offset = (-0.12, -0.08)
    standoff_distance = 0.55
    validate_approach_geometry = True

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 14.0
        self.scene.env_spacing = 2.5
        self.events.reset_object.params["object_distance_range"] = (
            self.approach_distance_start,
            self.approach_distance_end,
        )
        self.events.reset_object.params["object_pose_range"] = {
            "x": (-0.025, 0.025),
            "y": (-0.035, 0.035),
            "z": (0.0, 0.004),
            "roll": (-0.035, 0.035),
            "pitch": (-0.035, 0.035),
            "yaw": (-1.57, 1.57),
        }
        self.events.reset_object.params["bin_pose_range"] = {
            "x": (-0.035, 0.035),
            "y": (-0.050, 0.050),
            "z": (0.0, 0.006),
            "yaw": (0.0, 0.0),
        }
        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (-0.01, 0.01),
            "y": (-0.01, 0.01),
            "yaw": (-0.03, 0.03),
        }

        object_cfg = SceneEntityCfg("object")
        ee_cfg = SceneEntityCfg("robot", body_names=list(_D1_EE_BODY_NAMES), preserve_order=True)

        self.curriculum.go2_d1_pick_place_approach = None
        self.observations.policy.base_to_object_standoff = None
        self.observations.critic.base_to_object_standoff = None
        self.observations.policy.pick_place_stage.params["standoff_distance"] = None
        self.observations.critic.pick_place_stage.params["standoff_distance"] = None

        self.rewards.base_to_object_standoff = None
        self.rewards.base_faces_object = None
        self.rewards.base_xy_velocity_near_standoff_l2 = None
        self.rewards.arm_ready_before_standoff = None
        self.rewards.pick_place_stage_progress.params["standoff_distance"] = None
        self.rewards.ee_to_object = RewTerm(
            func=mdp.ee_to_object_tanh,
            weight=3.0,
            params={"ee_cfg": ee_cfg, "object_cfg": object_cfg, "std": 0.28},
        )
        self.rewards.object_lifted = RewTerm(
            func=mdp.object_lifted_above_height_exp,
            weight=0.75,
            params={
                "object_cfg": object_cfg,
                "min_height": self.object_init_pos[2],
                "target_height": self.target_lift_height,
                "std": 0.10,
            },
        )
        self.rewards.object_lifted_near_ee = RewTerm(
            func=mdp.object_lifted_near_ee_above_height_exp,
            weight=1.0,
            params={
                "ee_cfg": ee_cfg,
                "object_cfg": object_cfg,
                "min_height": self.object_init_pos[2],
                "target_height": self.target_lift_height,
                "std": 0.16,
            },
        )
        self.rewards.object_to_goal.weight = 8.0
        self.rewards.object_to_goal.params["std"] = 0.16
        self.rewards.object_velocity_to_goal.weight = 0.75
        self.rewards.object_velocity_to_goal.params["target_speed"] = 0.18
        self.rewards.object_velocity_to_goal.params["std"] = 0.24
        self.rewards.object_in_tray.weight = 12.0
        self.rewards.object_in_tray.params["bin_half_size"] = self.tray_half_size
        self.rewards.object_vertical.weight = 0.35

        _configure_stationary_manipulation(self)
        self.disable_zero_weight_rewards()
