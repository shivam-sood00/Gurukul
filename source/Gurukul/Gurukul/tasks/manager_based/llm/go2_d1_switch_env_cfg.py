"""Go2+D1 RGB switch-press task prepared for a future LLM controller."""

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
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass

import Gurukul.tasks.manager_based.locomotion.velocity.mdp as locomotion_mdp
from Gurukul.assets import ISAACLAB_ASSETS_DATA_DIR
from Gurukul.tasks.manager_based.locomotion.velocity.config.quadruped_with_arm.unitree_go2_d1_arm import (
    wbc_hierarchical_env_cfg as hierarchical_cfg,
)
from Gurukul.tasks.manager_based.locomotion.velocity.config.quadruped_with_arm.unitree_go2_d1_arm import (
    whole_body_controller_env_cfg as controller_cfg,
)

from . import mdp

SWITCH_TRAVEL = 0.025
SWITCH_PRESSED_THRESHOLD = 0.020
GO2_D1_LLM_ACTION_ORDER = (
    "vx",
    "vy",
    "wz",
    "body_pitch",
    "body_height",
    "grasp_x",
    "grasp_y",
    "grasp_z",
    "wrist_roll",
    "gripper",
)


@configclass
class Go2D1LlmObservationsCfg(ObsGroup):
    """Non-concatenated sensor payload intended for an LLM/VLM adapter."""

    rgb = ObsTerm(
        func=locomotion_mdp.image,
        params={
            "sensor_cfg": SceneEntityCfg("front_rgb_camera"),
            "data_type": "rgb",
            "normalize": False,
        },
    )
    switch = ObsTerm(
        func=mdp.switch_state,
        params={
            "asset_cfg": SceneEntityCfg("switch", joint_names=["button_joint"]),
            "travel": SWITCH_TRAVEL,
            "pressed_threshold": SWITCH_PRESSED_THRESHOLD,
        },
    )

    def __post_init__(self) -> None:
        self.enable_corruption = False
        self.concatenate_terms = False


@configclass
class Go2D1PressSwitchLlmEnvCfg(controller_cfg.UnitreeGo2D1LegWbcAsyncArmFlatEnvCfg):
    """Press a spring-return switch using Cartesian D1 commands over a frozen Go2 leg policy."""

    task_instruction = "Press the red button until the switch activates."
    llm_observation_group = "llm"
    llm_action_order = GO2_D1_LLM_ACTION_ORDER
    # Frame metadata used by the text-only code-policy runner. The imported
    # button cylinder is 40 mm long along local +X; its face toward the robot
    # therefore lies 20 mm along -X from the moving button-link origin.
    switch_button_body_name = "button"
    switch_press_axis_local = (1.0, 0.0, 0.0)
    switch_button_half_length_m = 0.020
    switch_prepress_clearance_m = 0.050
    switch_travel_m = SWITCH_TRAVEL
    switch_pressed_threshold_m = SWITCH_PRESSED_THRESHOLD
    wbc_policy_path: str = ""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.episode_length_s = 30.0
        # GPU simulation requires Fabric to propagate moving articulation and
        # attached-camera transforms to the renderer. The current Isaac Sim
        # 5.1 camera path supports the Fabric hierarchy interface.
        self.sim.use_fabric = True
        self.viewer.eye = (1.8, -2.2, 1.4)
        self.viewer.lookat = (0.4, 0.0, 0.45)
        self.scene.num_envs = 1
        self.scene.env_spacing = 3.0

        self.scene.switch = ArticulationCfg(
            prim_path="{ENV_REGEX_NS}/Switch",
            spawn=sim_utils.UsdFileCfg(
                usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Objects/llm_switch/spring_button.usda",
                activate_contact_sensors=True,
                rigid_props=sim_utils.RigidBodyPropertiesCfg(
                    disable_gravity=False,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=2,
                    max_depenetration_velocity=1.0,
                ),
                articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                    fix_root_link=True,
                    solver_position_iteration_count=8,
                    solver_velocity_iteration_count=2,
                ),
            ),
            init_state=ArticulationCfg.InitialStateCfg(
                pos=(0.92, 0.0, 0.62),
                joint_pos={"button_joint": 0.0},
                joint_vel={"button_joint": 0.0},
            ),
            actuators={
                "button_spring": ImplicitActuatorCfg(
                    joint_names_expr=["button_joint"],
                    effort_limit_sim=80.0,
                    velocity_limit_sim=0.50,
                    stiffness=350.0,
                    damping=8.0,
                    friction=0.2,
                )
            },
        )
        self.scene.front_rgb_camera = CameraCfg(
            prim_path="{ENV_REGEX_NS}/Robot/base/front_rgb_camera",
            update_period=0.10,
            height=240,
            width=320,
            data_types=["rgb"],
            spawn=sim_utils.PinholeCameraCfg(
                focal_length=24.0,
                focus_distance=1.0,
                horizontal_aperture=20.955,
                clipping_range=(0.05, 10.0),
            ),
            offset=CameraCfg.OffsetCfg(
                pos=(0.34, 0.0, 0.12),
                rot=(1.0, 0.0, 0.0, 0.0),
                convention="world",
            ),
        )
        self.num_rerenders_on_reset = 2

        controller_cfg.configure_go2_d1_leg_wbc_ee_hierarchical_runtime(self)
        hierarchical_cfg._configure_external_high_level_velocity_command(self)
        hierarchical_cfg._configure_cartesian_pick_ready_reset(self)
        self.actions.wbc_command.policy_path = self.wbc_policy_path

        # This is an interactive control scene, not a training rollout. Start
        # with D1 deployed in the established collision-safe Cartesian posture
        # and let physics continue even if the robot tips over or the session
        # exceeds the inherited episode duration. Otherwise Isaac Lab silently
        # auto-resets the environment and makes the robot appear kinematic.
        self.scene.robot.init_state.joint_pos.update(
            {
                joint_name: joint_position
                for joint_name, joint_position in zip(
                    self.arm_joint_names,
                    controller_cfg.GO2_D1_WBC_WORKSPACE_READY_POSE,
                )
            }
        )
        self.commands.base_velocity.debug_vis = False
        self.terminations.time_out = None
        self.terminations.bad_orientation = None

        # Keep the first language task deterministic. Scene and command
        # randomization can be introduced later as explicit benchmark variants.
        for event_name in (
            "randomize_rigid_body_material",
            "randomize_rigid_body_mass_base",
            "randomize_rigid_body_mass_others",
            "randomize_com_positions",
            "randomize_apply_external_force_torque",
            "randomize_actuator_gains",
            "randomize_push_robot",
            "randomize_d1_mount_x",
            "randomize_arm_payload_mass",
        ):
            if hasattr(self.events, event_name):
                setattr(self.events, event_name, None)
        self.events.randomize_reset_base.params["pose_range"] = {}
        self.events.randomize_reset_base.params["velocity_range"] = {}
        self.events.randomize_reset_joints.params["position_range"] = (0.0, 0.0)
        self.events.randomize_reset_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.reset_switch = EventTerm(
            func=locomotion_mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "asset_cfg": SceneEntityCfg("switch", joint_names=["button_joint"]),
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
            },
        )

        self.observations.llm = Go2D1LlmObservationsCfg()
        self.rewards.switch_press_progress = RewTerm(
            func=mdp.switch_press_progress,
            weight=2.0,
            params={
                "asset_cfg": SceneEntityCfg("switch", joint_names=["button_joint"]),
                "pressed_threshold": SWITCH_PRESSED_THRESHOLD,
            },
        )
        self.terminations.switch_pressed = DoneTerm(
            func=mdp.switch_pressed,
            params={
                "asset_cfg": SceneEntityCfg("switch", joint_names=["button_joint"]),
                "pressed_threshold": SWITCH_PRESSED_THRESHOLD,
            },
        )

        self.disable_zero_weight_rewards()
