"""Useful Go2+D1 service tasks for door opening and contact-gated pick-place."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import Gurukul.tasks.manager_based.locomotion.velocity.mdp as locomotion_mdp
from Gurukul.assets import ISAACLAB_ASSETS_DATA_DIR
from Gurukul.tasks.manager_based.locomotion.velocity.config.quadruped_with_arm.unitree_go2_d1_arm import (
    loco_manipulation_env_cfg as manipulation_cfg,
)

from . import mdp
from .go2_d1_switch_env_cfg import Go2D1PressSwitchLlmEnvCfg

DOOR_OPEN_THRESHOLD_RAD = 1.3089969390
OBJECT_INITIAL_POSITION_M = (0.88, -0.12, 0.625)
OBJECT_REQUIRED_LIFT_M = 0.08
TRAY_CENTER_M = (0.88, 0.16, 0.64)
TRAY_HALF_SIZE_M = (0.085, 0.085, 0.08)


@configclass
class Go2D1ServiceCameraObservationsCfg(ObsGroup):
    """Raw robot RGB retained separately from the frozen WBC observations."""

    rgb = ObsTerm(
        func=locomotion_mdp.image,
        params={
            "sensor_cfg": SceneEntityCfg("front_rgb_camera"),
            "data_type": "rgb",
            "normalize": False,
        },
    )

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class Go2D1DoorObservationsCfg(Go2D1ServiceCameraObservationsCfg):
    """Camera plus physical door state."""

    door = ObsTerm(
        func=mdp.hinged_door_state,
        params={
            "asset_cfg": SceneEntityCfg("door", joint_names=["door_hinge"]),
            "open_threshold_rad": DOOR_OPEN_THRESHOLD_RAD,
        },
    )


@configclass
class Go2D1PickPlaceObservationsCfg(Go2D1ServiceCameraObservationsCfg):
    """Camera plus the complete object pose and velocity."""

    object = ObsTerm(
        func=mdp.rigid_object_service_state,
        params={"asset_cfg": SceneEntityCfg("object")},
    )


def _remove_switch_scene(cfg: Go2D1PressSwitchLlmEnvCfg) -> None:
    cfg.scene.switch = None
    cfg.events.reset_switch = None
    cfg.rewards.switch_press_progress = None
    cfg.terminations.switch_pressed = None


def _configure_service_velocity_response(cfg: Go2D1PressSwitchLlmEnvCfg) -> None:
    """Apply a trackable service command in one 50 Hz control tick."""
    rate_limits = list(cfg.actions.wbc_command.normalized_action_rate_limits)
    rate_limits[:3] = [20.0, 20.0, 20.0]
    cfg.actions.wbc_command.normalized_action_rate_limits = tuple(rate_limits)


@configclass
class Go2D1OpenDoorLlmEnvCfg(Go2D1PressSwitchLlmEnvCfg):
    """Push a physical revolute door beyond a measured opening angle."""

    task_instruction = "Open the blue access door by pushing its gold lever until the hinge reaches 75 degrees."
    door_panel_body_name = "panel"
    door_handle_local_position_m = (-0.145, -0.72, -0.10)
    door_push_axis_local = (1.0, 0.0, 0.0)
    door_handle_normal_radius_m = 0.018
    door_precontact_clearance_m = 0.045
    door_clear_width_m = 0.90
    door_clear_height_m = 1.58
    door_required_robot_clearance_m = 0.55
    door_open_threshold_rad = DOOR_OPEN_THRESHOLD_RAD

    def __post_init__(self) -> None:
        super().__post_init__()
        _remove_switch_scene(self)
        _configure_service_velocity_response(self)

        self.episode_length_s = 35.0
        self.viewer.eye = (2.0, -2.2, 1.5)
        self.viewer.lookat = (0.65, -0.15, 0.55)
        self.scene.door = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Door",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Objects/service_door/hinged_door.usda",
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    solver_position_iteration_count=12,
                    solver_velocity_iteration_count=4,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    fix_root_link=True,
                    solver_position_iteration_count=12,
                    solver_velocity_iteration_count=4,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                # The handle starts at approximately (0.905, -0.22, 0.70) world,
                # well outside the reset finger midpoint. The robot must walk
                # before the handle enters its Cartesian workspace.
                pos=(1.05, 0.50, 0.0),
                joint_pos={"door_hinge": 0.0},
                joint_vel={"door_hinge": 0.0},
            ),
            actuators={
                "passive_hinge": ImplicitActuatorCfg(
                    joint_names_expr=["door_hinge"],
                    effort_limit_sim=25.0,
                    velocity_limit_sim=2.0,
                    stiffness=0.0,
                    damping=1.2,
                    friction=0.15,
                )
            },
        )
        self.events.reset_door = EventTerm(
            func=locomotion_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("door", joint_names=["door_hinge"]),
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
            },
        )
        self.observations.llm = Go2D1DoorObservationsCfg()
        self.rewards.door_open_progress = RewTerm(
            func=mdp.hinged_door_open_progress,
            weight=2.0,
            params={
                "asset_cfg": SceneEntityCfg("door", joint_names=["door_hinge"]),
                "open_threshold_rad": self.door_open_threshold_rad,
            },
        )
        self.terminations.door_opened = DoneTerm(
            func=mdp.hinged_door_opened,
            params={
                "asset_cfg": SceneEntityCfg("door", joint_names=["door_hinge"]),
                "open_threshold_rad": self.door_open_threshold_rad,
            },
        )


@configclass
class Go2D1PickPlaceLlmEnvCfg(Go2D1PressSwitchLlmEnvCfg):
    """Pick a can with bilateral finger contact and release it into a tray."""

    task_instruction = "Pick up the red can, lift it securely, and place and release it in the black tray."
    object_initial_position_m = OBJECT_INITIAL_POSITION_M
    object_required_lift_m = OBJECT_REQUIRED_LIFT_M
    tray_center_m = TRAY_CENTER_M
    tray_half_size_m = TRAY_HALF_SIZE_M
    table_center_m = (0.88, 0.02, 0.54)
    table_size_m = (0.50, 0.62, 0.04)
    table_surface_z_m = 0.56
    object_radius_m = 0.035
    object_height_m = 0.12

    def __post_init__(self) -> None:
        super().__post_init__()
        _remove_switch_scene(self)
        _configure_service_velocity_response(self)

        self.episode_length_s = 40.0
        self.scene.replicate_physics = False
        self.viewer.eye = (1.6, -1.8, 1.35)
        self.viewer.lookat = (0.45, 0.02, 0.55)
        self.scene.pick_table = manipulation_cfg._kinematic_box_cfg(
            prim_path="{ENV_REGEX_NS}/PickTable",
            size=self.table_size_m,
            init_pos=self.table_center_m,
            color=(0.32, 0.34, 0.36),
        )
        self.scene.object = manipulation_cfg._can_object_cfg(
            radius=self.object_radius_m,
            height=self.object_height_m,
            mass=0.08,
            init_pos=self.object_initial_position_m,
            color=(0.85, 0.06, 0.04),
        )

        wall_thickness = 0.015
        wall_height = 0.08
        floor_thickness = 0.02
        outer_x = 2.0 * (self.tray_half_size_m[0] + wall_thickness)
        outer_y = 2.0 * (self.tray_half_size_m[1] + wall_thickness)
        wall_z = self.table_surface_z_m + floor_thickness + 0.5 * wall_height
        tray_x, tray_y, _ = self.tray_center_m
        self.scene.place_tray_floor = manipulation_cfg._kinematic_box_cfg(
            prim_path="{ENV_REGEX_NS}/PlaceTrayFloor",
            size=(outer_x, outer_y, floor_thickness),
            init_pos=(tray_x, tray_y, self.table_surface_z_m + 0.5 * floor_thickness),
            color=(0.08, 0.10, 0.12),
        )
        self.scene.place_tray_front_wall = manipulation_cfg._kinematic_box_cfg(
            prim_path="{ENV_REGEX_NS}/PlaceTrayFrontWall",
            size=(outer_x, wall_thickness, wall_height),
            init_pos=(tray_x, tray_y + self.tray_half_size_m[1] + 0.5 * wall_thickness, wall_z),
            color=(0.08, 0.10, 0.12),
        )
        self.scene.place_tray_back_wall = manipulation_cfg._kinematic_box_cfg(
            prim_path="{ENV_REGEX_NS}/PlaceTrayBackWall",
            size=(outer_x, wall_thickness, wall_height),
            init_pos=(tray_x, tray_y - self.tray_half_size_m[1] - 0.5 * wall_thickness, wall_z),
            color=(0.08, 0.10, 0.12),
        )
        self.scene.place_tray_left_wall = manipulation_cfg._kinematic_box_cfg(
            prim_path="{ENV_REGEX_NS}/PlaceTrayLeftWall",
            size=(wall_thickness, outer_y, wall_height),
            init_pos=(tray_x + self.tray_half_size_m[0] + 0.5 * wall_thickness, tray_y, wall_z),
            color=(0.08, 0.10, 0.12),
        )
        self.scene.place_tray_right_wall = manipulation_cfg._kinematic_box_cfg(
            prim_path="{ENV_REGEX_NS}/PlaceTrayRightWall",
            size=(wall_thickness, outer_y, wall_height),
            init_pos=(tray_x - self.tray_half_size_m[0] - 0.5 * wall_thickness, tray_y, wall_z),
            color=(0.08, 0.10, 0.12),
        )
        self.scene.left_gripper_object_contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/d1/Link7_1",
            history_length=4,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )
        self.scene.right_gripper_object_contact_forces = ContactSensorCfg(
            prim_path="{ENV_REGEX_NS}/Robot/d1/Link7_2",
            history_length=4,
            filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"],
        )
        self.observations.llm = Go2D1PickPlaceObservationsCfg()
        self.rewards.object_lift_progress = RewTerm(
            func=mdp.object_lift_progress,
            weight=2.0,
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "initial_height_m": self.object_initial_position_m[2],
                "required_lift_m": self.object_required_lift_m,
            },
        )
        self.terminations.object_placed = DoneTerm(
            func=mdp.object_settled_in_region,
            params={
                "asset_cfg": SceneEntityCfg("object"),
                "center_m": self.tray_center_m,
                "half_size_m": self.tray_half_size_m,
                "max_speed_mps": 0.20,
            },
        )
