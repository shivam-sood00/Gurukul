"""Run useful Go2+D1 door-opening and contact-gated pick/place tasks."""

from __future__ import annotations

import argparse
import concurrent.futures
import math
import time
import traceback
from dataclasses import asdict
from typing import Any

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run the Go2+D1 useful service-skill benchmark.")
parser.add_argument("--base-policy", required=True, help="Exported Go2+D1 LegWBC-AsyncArm policy.pt.")
parser.add_argument(
    "--scenario",
    choices=("open-door", "pick", "pick-place"),
    default="open-door",
    help="Physical service skill to evaluate.",
)
parser.add_argument(
    "--controller",
    choices=("demo", "llm"),
    default="demo",
    help="Use the deterministic mechanics oracle or ask an LLM for validated actions.",
)
parser.add_argument("--llm-model", default=None, help="Hosted Hugging Face or local model ID for --controller llm.")
parser.add_argument("--llm-provider", default="auto", help="Hugging Face Inference Provider.")
parser.add_argument("--llm-base-url", default=None, help="Optional loopback OpenAI-compatible server URL.")
parser.add_argument("--llm-timeout", type=float, default=60.0, help="Timeout for one LLM request.")
parser.add_argument(
    "--llm-action-space",
    choices=("primitive", "semantic"),
    default="primitive",
    help="Use real velocity/IK/gripper primitives or the oracle semantic-skill ablation.",
)
parser.add_argument(
    "--llm-control-mode",
    choices=("one-shot", "reactive", "receding-horizon"),
    default="one-shot",
    help="Execute one horizon, react one action at a time, or retain a horizon while replanning after action outcomes.",
)
parser.add_argument(
    "--llm-max-requests",
    type=int,
    default=12,
    help="Maximum LLM calls in reactive/receding-horizon modes; one-shot always uses at most one.",
)
parser.add_argument(
    "--llm-start-delay",
    type=float,
    default=1.0,
    help="Simulation settling time before the LLM request.",
)
parser.add_argument("--max_steps", type=int, default=0, help="Maximum control steps; zero runs until closed.")
parser.add_argument("--print-interval", type=int, default=25, help="Steps between physical state diagnostics.")
parser.add_argument("--exit-on-success", action="store_true", help="Exit instead of retaining the successful scene.")
parser.add_argument("--dashboard", action="store_true", help="Publish the always-saved policy trace on loopback HTTP.")
parser.add_argument("--dashboard-port", type=int, default=8765, help="127.0.0.1 dashboard port.")
parser.add_argument(
    "--log-dir",
    default="logs/llm/go2_d1_service",
    help="Parent directory for timestamped dashboard runs.",
)
parser.add_argument("--log-interval", type=int, default=10, help="Steps between dashboard telemetry samples.")
parser.add_argument("--dashboard-refresh", type=float, default=0.5, help="Dashboard refresh interval in seconds.")
parser.add_argument(
    "--no-camera",
    action="store_true",
    help="Disable the robot RGB sensor for faster headless mechanics tests.",
)
parser.add_argument("--no-real-time", dest="real_time", action="store_false", help="Run without wall-clock pacing.")
parser.add_argument("--use_fabric", action="store_true", help="Enable Fabric scene I/O.")
parser.add_argument("--no-fabric", dest="use_fabric", action="store_false", help="Disable Fabric for CPU/debug use.")
parser.set_defaults(real_time=True, use_fabric=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = not args_cli.no_camera

if args_cli.controller == "llm" and not args_cli.llm_model:
    parser.error("--controller llm requires --llm-model.")
if args_cli.max_steps < 0:
    parser.error("--max_steps must be >= 0.")
if args_cli.print_interval <= 0 or args_cli.log_interval <= 0:
    parser.error("--print-interval and --log-interval must be > 0.")
if args_cli.llm_timeout <= 0.0 or args_cli.llm_start_delay < 0.0:
    parser.error("--llm-timeout must be > 0 and --llm-start-delay must be >= 0.")
if args_cli.llm_max_requests <= 0:
    parser.error("--llm-max-requests must be > 0.")
if not 1 <= args_cli.dashboard_port <= 65535:
    parser.error("--dashboard-port must be in [1, 65535].")
if args_cli.dashboard_refresh <= 0.0:
    parser.error("--dashboard-refresh must be > 0.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Isaac-dependent imports are valid after the application starts."""

import gymnasium as gym
import Gurukul.tasks  # noqa: F401
import torch
from code_policy import rotate_vector_wxyz, world_point_to_root
from live_dashboard import LiveDashboard
from service_policy import (
    ServiceSkillExecutor,
    build_service_prompt,
    build_service_reactive_prompt,
    build_service_replan_prompt,
    default_service_program,
    request_service_decision,
    request_service_program,
    single_skill_program,
)
from service_primitive_policy import (
    PrimitiveExecutor,
    PrimitiveResponseValidationError,
    apply_base_station_keep,
    build_primitive_prompt,
    build_primitive_repair_prompt,
    build_primitive_replan_prompt,
    request_primitive_decision,
    request_primitive_program,
    single_primitive_program,
)

from isaaclab_tasks.utils import parse_env_cfg

TASK_IDS = {
    "open-door": "Gurukul-Isaac-LLM-Go2-D1-Open-Door-v0",
    "pick": "Gurukul-Isaac-LLM-Go2-D1-Pick-Place-v0",
    "pick-place": "Gurukul-Isaac-LLM-Go2-D1-Pick-Place-v0",
}


def _rounded(value: Any, digits: int = 6) -> list[float]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().tolist()
    return [round(float(component), digits) for component in value]


def _add(a: Any, b: Any) -> tuple[float, float, float]:
    return tuple(float(a[axis]) + float(b[axis]) for axis in range(3))


def _contact_active(sensor: Any, force_threshold: float = 0.35) -> bool:
    data = sensor.data
    if data.force_matrix_w_history is not None:
        return bool(torch.any(torch.linalg.vector_norm(data.force_matrix_w_history[0], dim=-1) > force_threshold))
    return bool(torch.any(torch.linalg.vector_norm(data.net_forces_w_history[0], dim=-1) > force_threshold))


def _robot_state(robot: Any, action_term: Any) -> dict[str, Any]:
    root_position_w = _rounded(robot.data.root_pos_w[0])
    root_quaternion_w = _rounded(robot.data.root_quat_w[0])
    grasp_position_w = _rounded(robot.data.body_pos_w[0].index_select(0, action_term._grasp_body_ids).mean(dim=0))
    w, x, y, z = root_quaternion_w
    measured_pitch_rad = math.asin(max(-1.0, min(1.0, 2.0 * (w * y - z * x))))
    applied = action_term.processed_actions[0]
    pitch_low, pitch_high = action_term.cfg.body_pitch_range
    height_low, height_high = action_term.cfg.body_height_range
    pitch_command_rad = float(pitch_low) + 0.5 * (float(applied[3]) + 1.0) * (
        float(pitch_high) - float(pitch_low)
    )
    height_command_m = float(height_low) + 0.5 * (float(applied[4]) + 1.0) * (
        float(height_high) - float(height_low)
    )
    gripper_joint_pos = robot.data.joint_pos[0].index_select(0, action_term._gripper_joint_ids)
    measured_closed_fraction = float(
        torch.clamp(torch.mean(gripper_joint_pos) / float(action_term.cfg.gripper_scale), 0.0, 1.0).item()
    )
    commanded_closed_fraction = max(0.0, min(1.0, 0.5 * (float(applied[9]) + 1.0)))
    return {
        "root_pose_world": {
            "frame": "world",
            "position_m": root_position_w,
            "quaternion_wxyz": root_quaternion_w,
        },
        "grasp_midpoint": {
            "meaning": "midpoint of Link7_1 and Link7_2",
            "world_position_m": grasp_position_w,
            "root_position_m": _rounded(world_point_to_root(grasp_position_w, root_position_w, root_quaternion_w)),
        },
        "measured_root_velocity": {
            "linear_root_mps": _rounded(robot.data.root_lin_vel_b[0]),
            "angular_root_rps": _rounded(robot.data.root_ang_vel_b[0]),
        },
        "posture_and_gripper": {
            "measured_body_pitch_rad": round(measured_pitch_rad, 6),
            "commanded_body_pitch_rad": round(pitch_command_rad, 6),
            "measured_root_height_m": round(root_position_w[2], 6),
            "commanded_body_height_m": round(height_command_m, 6),
            "gripper_command_closed_fraction": round(commanded_closed_fraction, 6),
            "gripper_measured_closed_fraction": round(measured_closed_fraction, 6),
            "gripper_contract": "0=fully open, 1=fully closed",
        },
    }


def _door_state(env: Any, action_term: Any, handles: dict[str, int]) -> dict[str, Any]:
    robot = env.unwrapped.scene["robot"]
    door = env.unwrapped.scene["door"]
    cfg = env.unwrapped.cfg
    robot_state = _robot_state(robot, action_term)
    root_pose = robot_state["root_pose_world"]
    panel_position_w = _rounded(door.data.body_pos_w[0, handles["door_panel_body"]])
    panel_quaternion_w = _rounded(door.data.body_quat_w[0, handles["door_panel_body"]])
    handle_offset_w = rotate_vector_wxyz(panel_quaternion_w, cfg.door_handle_local_position_m)
    handle_center_w = _add(panel_position_w, handle_offset_w)
    push_direction_w = rotate_vector_wxyz(panel_quaternion_w, cfg.door_push_axis_local)
    precontact_distance = float(cfg.door_handle_normal_radius_m + cfg.door_precontact_clearance_m)
    precontact_w = tuple(handle_center_w[axis] - precontact_distance * push_direction_w[axis] for axis in range(3))
    angle_rad = float(door.data.joint_pos[0, handles["door_joint"]].item())
    projected_clear_width_m = float(cfg.door_clear_width_m) * (1.0 - max(math.cos(angle_rad), 0.0))
    grasp_to_handle_distance_m = math.dist(robot_state["grasp_midpoint"]["world_position_m"], handle_center_w)
    handle_center_b = _rounded(
        world_point_to_root(handle_center_w, root_pose["position_m"], root_pose["quaternion_wxyz"])
    )
    precontact_b = _rounded(world_point_to_root(precontact_w, root_pose["position_m"], root_pose["quaternion_wxyz"]))
    safe_travel_b = (0.34, 0.0, 0.60)
    safe_travel_error_m = math.dist(robot_state["grasp_midpoint"]["root_position_m"], safe_travel_b)
    precontact_error_m = math.dist(robot_state["grasp_midpoint"]["world_position_m"], precontact_w)
    workspace = action_term.cfg.ee_pos_range
    precontact_inside_workspace = all(
        float(workspace[axis][0]) <= float(precontact_b[axis]) <= float(workspace[axis][1]) for axis in range(3)
    )
    base_forward_error_m = max(float(handle_center_b[0]) - 0.42, 0.0)
    suggested_vx_mps = min(max(1.5 * base_forward_error_m, 0.40), 0.75)
    push_target_w = tuple(
        float(handle_center_w[axis]) + 0.12 * float(push_direction_w[axis]) for axis in range(3)
    )
    if angle_rad >= float(cfg.door_open_threshold_rad):
        planning_context = {
            "recommended_op": "stop",
            "reason": "door hinge already satisfies the success threshold",
        }
    elif safe_travel_error_m > 0.07:
        planning_context = {
            "recommended_op": "move_ee",
            "reason": "stow the arm before locomotion",
            "frame": "root",
            "position_m": list(safe_travel_b),
            "tolerance_m": 0.06,
            "timeout_s": 4.0,
        }
    elif base_forward_error_m > 0.05 or not precontact_inside_workspace:
        planning_context = {
            "recommended_op": "drive_base_to",
            "reason": "door precontact is outside the live IK workspace",
            "frame": "world",
            "position_m": _rounded(handle_center_w),
            "standoff_m": 0.42,
            "forward_error_m": round(base_forward_error_m, 6),
            "max_vx_mps": round(suggested_vx_mps, 6),
            "max_vy_mps": 0.25,
            "max_yaw_rps": 0.50,
            "body_pitch_rad": 0.0,
            "body_height_m": 0.33,
            "tolerance_m": 0.05,
            "timeout_s": 10.0,
        }
    elif precontact_error_m > 0.05 and angle_rad < math.radians(2.0):
        planning_context = {
            "recommended_op": "move_ee",
            "reason": "move the open finger midpoint to door precontact",
            "frame": "world",
            "position_m": _rounded(precontact_w),
            "tolerance_m": 0.05,
            "timeout_s": 5.0,
        }
    else:
        planning_context = {
            "recommended_op": "move_ee",
            "reason": "push through the live handle along its measured direction",
            "frame": "world",
            "position_m": _rounded(push_target_w),
            "tolerance_m": 0.05,
            "timeout_s": 5.0,
        }
    return {
        "scenario": "open-door",
        "frames": {
            "world": "fixed Isaac frame",
            "root": "moving Go2 base frame: +x forward, +y left, +z up",
            "quaternion_order": "wxyz",
        },
        "robot": robot_state,
        "door": {
            "hinge_angle_rad": round(angle_rad, 6),
            "hinge_angle_deg": round(math.degrees(angle_rad), 3),
            "open_threshold_rad": float(cfg.door_open_threshold_rad),
            "opened": angle_rad >= float(cfg.door_open_threshold_rad),
            "passage": {
                "frame_clear_width_m": float(cfg.door_clear_width_m),
                "frame_clear_height_m": float(cfg.door_clear_height_m),
                "projected_clear_width_m": round(projected_clear_width_m, 6),
                "required_robot_clearance_m": float(cfg.door_required_robot_clearance_m),
                "wide_enough_to_traverse": projected_clear_width_m >= float(cfg.door_required_robot_clearance_m),
            },
            "grasp_to_handle_distance_m": round(grasp_to_handle_distance_m, 6),
            "panel_pose_world": {
                "position_m": panel_position_w,
                "quaternion_wxyz": panel_quaternion_w,
            },
            "handle_center": {
                "world_position_m": _rounded(handle_center_w),
                "root_position_m": handle_center_b,
            },
            "precontact": {
                "world_position_m": _rounded(precontact_w),
                "root_position_m": precontact_b,
                "inside_live_ik_workspace": precontact_inside_workspace,
            },
            "push_direction": {
                "frame": "world",
                "world_unit_vector": _rounded(push_direction_w),
            },
        },
        "controller": {
            "cartesian_target": "finger midpoint in the live root frame",
            "grasp_workspace_root_m": [list(map(float, bounds)) for bounds in action_term.cfg.ee_pos_range],
            "safe_travel_error_m": round(safe_travel_error_m, 6),
            "planning_context": {
                **planning_context,
                "advisory_only": True,
                "feasibility": {
                    "door_precontact_inside_live_ik_workspace": precontact_inside_workspace,
                    "drive_base_to_available": True,
                    "safe_travel_pose_error_m": round(safe_travel_error_m, 6),
                },
            },
        },
    }


def _pick_place_state(
    env: Any,
    action_term: Any,
    memory: dict[str, bool],
    scenario: str,
) -> dict[str, Any]:
    robot = env.unwrapped.scene["robot"]
    obj = env.unwrapped.scene["object"]
    cfg = env.unwrapped.cfg
    robot_state = _robot_state(robot, action_term)
    root_pose = robot_state["root_pose_world"]
    object_position_w = _rounded(obj.data.root_pos_w[0])
    left_contact = _contact_active(env.unwrapped.scene.sensors["left_gripper_object_contact_forces"])
    right_contact = _contact_active(env.unwrapped.scene.sensors["right_gripper_object_contact_forces"])
    bilateral_contact = left_contact and right_contact
    lifted = object_position_w[2] >= float(cfg.object_initial_position_m[2] + cfg.object_required_lift_m)
    memory["lifted_once"] |= bilateral_contact and lifted
    tray_center = tuple(map(float, cfg.tray_center_m))
    tray_half_size = tuple(map(float, cfg.tray_half_size_m))
    in_tray = all(abs(object_position_w[axis] - tray_center[axis]) <= tray_half_size[axis] for axis in range(3))
    speed_mps = float(torch.linalg.vector_norm(obj.data.root_lin_vel_w[0]).item())
    grasp_to_object_distance_m = math.dist(robot_state["grasp_midpoint"]["world_position_m"], object_position_w)
    placement_center = (tray_center[0], tray_center[1], float(cfg.table_surface_z_m + 0.10))
    object_position_b = _rounded(
        world_point_to_root(object_position_w, root_pose["position_m"], root_pose["quaternion_wxyz"])
    )
    placement_center_b = _rounded(
        world_point_to_root(placement_center, root_pose["position_m"], root_pose["quaternion_wxyz"])
    )
    workspace = action_term.cfg.ee_pos_range

    def _inside_workspace(point_b: list[float]) -> bool:
        return all(float(workspace[axis][0]) <= point_b[axis] <= float(workspace[axis][1]) for axis in range(3))

    object_inside_workspace = _inside_workspace(object_position_b)
    placement_inside_workspace = _inside_workspace(placement_center_b)
    safe_travel_b = (0.34, 0.0, 0.60)
    safe_travel_error_m = math.dist(robot_state["grasp_midpoint"]["root_position_m"], safe_travel_b)
    commanded_closed_fraction = robot_state["posture_and_gripper"]["gripper_command_closed_fraction"]
    approach_target_b = placement_center_b if memory["lifted_once"] else object_position_b
    base_forward_error_m = max(float(approach_target_b[0]) - 0.42, 0.0)
    suggested_vx_mps = min(max(1.5 * base_forward_error_m, 0.40), 0.75)
    if scenario == "pick" and memory["lifted_once"] and lifted and bilateral_contact:
        planning_context = {
            "recommended_op": "stop",
            "reason": "the measured bilateral grasp and required lift satisfy pick success",
        }
    elif scenario == "pick-place" and memory["lifted_once"]:
        if in_tray and not (left_contact or right_contact):
            planning_context = {
                "recommended_op": "stop",
                "reason": "the lifted object is in the tray and released",
            }
        elif in_tray:
            planning_context = {
                "recommended_op": "set_gripper",
                "reason": "the lifted object is inside the tray but remains in finger contact",
                "closed_fraction": 0,
                "duration_s": 0.8,
            }
        elif not placement_inside_workspace:
            planning_context = {
                "recommended_op": "drive_base_to",
                "reason": "the placement center is outside the live IK workspace",
                "frame": "world",
                "position_m": list(placement_center),
                "standoff_m": 0.42,
                "forward_error_m": round(base_forward_error_m, 6),
                "max_vx_mps": round(suggested_vx_mps, 6),
                "max_vy_mps": 0.25,
                "max_yaw_rps": 0.50,
                "body_pitch_rad": 0.0,
                "body_height_m": 0.33,
                "tolerance_m": 0.05,
                "timeout_s": 10.0,
            }
        else:
            planning_context = {
                "recommended_op": "move_ee",
                "reason": "carry the verified grasp to the reachable placement center while remaining closed",
                "frame": "world",
                "position_m": list(placement_center),
                "tolerance_m": 0.05,
                "timeout_s": 5.0,
            }
    elif safe_travel_error_m > 0.07:
        planning_context = {
            "recommended_op": "move_ee",
            "reason": "put the open arm in a travel-safe configuration before locomotion",
            "frame": "root",
            "position_m": list(safe_travel_b),
            "tolerance_m": 0.06,
            "timeout_s": 4.0,
        }
    elif not object_inside_workspace:
        planning_context = {
            "recommended_op": "drive_base_to",
            "reason": "the object center is outside the live IK workspace",
            "frame": "world",
            "position_m": object_position_w,
            "standoff_m": 0.42,
            "forward_error_m": round(base_forward_error_m, 6),
            "max_vx_mps": round(suggested_vx_mps, 6),
            "max_vy_mps": 0.25,
            "max_yaw_rps": 0.50,
            "body_pitch_rad": 0.0,
            "body_height_m": 0.33,
            "tolerance_m": 0.05,
            "timeout_s": 10.0,
        }
    elif commanded_closed_fraction > 0.1 and not bilateral_contact:
        planning_context = {
            "recommended_op": "set_gripper",
            "reason": "open the gripper before approaching the reachable object",
            "closed_fraction": 0,
            "duration_s": 0.8,
        }
    elif grasp_to_object_distance_m > 0.05:
        planning_context = {
            "recommended_op": "move_ee",
            "reason": "align the open finger midpoint with the reachable object center",
            "frame": "world",
            "position_m": object_position_w,
            "tolerance_m": 0.04,
            "timeout_s": 5.0,
        }
    elif not bilateral_contact:
        planning_context = {
            "recommended_op": "set_gripper",
            "reason": "the finger midpoint is aligned but bilateral contact is not yet measured",
            "closed_fraction": 1,
            "duration_s": 0.8,
        }
    else:
        lift_target_w = [
            object_position_w[0],
            object_position_w[1],
            max(object_position_w[2] + float(cfg.object_required_lift_m), float(cfg.table_surface_z_m + 0.18)),
        ]
        planning_context = {
            "recommended_op": "move_ee",
            "reason": "bilateral contact is measured; lift while preserving the closed gripper",
            "frame": "world",
            "position_m": lift_target_w,
            "tolerance_m": 0.05,
            "timeout_s": 5.0,
        }
    return {
        "scenario": scenario,
        "frames": {
            "world": "fixed Isaac frame",
            "root": "moving Go2 base frame: +x forward, +y left, +z up",
            "quaternion_order": "wxyz",
        },
        "robot": robot_state,
        "object": {
            "name": "can",
            "radius_m": float(cfg.object_radius_m),
            "height_m": float(cfg.object_height_m),
            "required_lift_m": float(cfg.object_required_lift_m),
            "world_position_m": object_position_w,
            "root_position_m": object_position_b,
            "quaternion_world_wxyz": _rounded(obj.data.root_quat_w[0]),
            "linear_velocity_world_mps": _rounded(obj.data.root_lin_vel_w[0]),
            "speed_mps": round(speed_mps, 6),
            "grasp_to_object_distance_m": round(grasp_to_object_distance_m, 6),
            "left_finger_contact": left_contact,
            "right_finger_contact": right_contact,
            "bilateral_contact": bilateral_contact,
            "any_finger_contact": left_contact or right_contact,
            "lifted": lifted,
            "lifted_once": memory["lifted_once"],
            "in_tray": in_tray,
        },
        "tray": {
            "name": "tray",
            "center_world_m": list(tray_center),
            "half_size_m": list(tray_half_size),
            "placement_center_world_m": list(placement_center),
        },
        "controller": {
            "cartesian_target": "finger midpoint in the live root frame",
            "grasp_workspace_root_m": [list(map(float, bounds)) for bounds in workspace],
            "planning_context": {
                **planning_context,
                "advisory_only": True,
                "feasibility": {
                    "object_center_inside_live_ik_workspace": object_inside_workspace,
                    "placement_center_inside_live_ik_workspace": placement_inside_workspace,
                    "drive_base_to_available": True,
                    "safe_travel_pose_error_m": round(safe_travel_error_m, 6),
                },
            },
        },
    }


def _success(state: dict[str, Any], scenario: str) -> bool:
    if scenario == "open-door":
        return bool(state["door"]["opened"])
    obj = state["object"]
    if scenario == "pick":
        return bool(obj["lifted_once"] and obj["lifted"] and obj["bilateral_contact"])
    return bool(obj["lifted_once"] and obj["in_tray"] and not obj["any_finger_contact"] and obj["speed_mps"] <= 0.20)


def main() -> None:  # noqa: C901
    """Run one service scenario through a frozen leg WBC and Cartesian D1 controller."""
    task_id = TASK_IDS[args_cli.scenario]
    env_cfg = parse_env_cfg(task_id, device=args_cli.device, num_envs=1, use_fabric=args_cli.use_fabric)
    env_cfg.wbc_policy_path = args_cli.base_policy
    env_cfg.actions.wbc_command.policy_path = args_cli.base_policy
    if args_cli.no_camera:
        env_cfg.scene.front_rgb_camera = None
        env_cfg.observations.llm.rgb = None
    if args_cli.scenario == "open-door":
        env_cfg.terminations.door_opened = None
    else:
        env_cfg.terminations.object_placed = None
    env = gym.make(task_id, cfg=env_cfg)
    dashboard = None
    request_pool = None
    request_future = None
    request_started_at = None
    request_kind = None
    request_fn = None
    request_prompt = None
    request_repair_attempts = 0
    waiting_anchor_world_m = None
    request_count = 0
    executor = None
    initial_plan = None
    initial_request_completed = False
    execution_history: list[dict[str, Any]] = []
    replan_pending = False
    llm_stopped = False
    request_budget = 1 if args_cli.llm_control_mode == "one-shot" else args_cli.llm_max_requests
    success_latched = False
    memory = {"lifted_once": False}

    try:
        observations, _ = env.reset()
        action_term = env.unwrapped.action_manager.get_term("wbc_command")
        actions = action_term.processed_actions.detach().clone()
        actions[:, 0:3] = 0.0
        command_term = env.unwrapped.command_manager.get_term("base_velocity")
        handles: dict[str, int] = {}
        if args_cli.scenario == "open-door":
            door = env.unwrapped.scene["door"]
            handles["door_joint"] = int(door.find_joints("door_hinge")[0][0])
            handles["door_panel_body"] = int(door.find_bodies(env.unwrapped.cfg.door_panel_body_name)[0][0])
            state = _door_state(env, action_term, handles)
            initial_clearance_m = float(state["door"]["grasp_to_handle_distance_m"])
        else:
            state = _pick_place_state(env, action_term, memory, args_cli.scenario)
            initial_clearance_m = float(state["object"]["grasp_to_object_distance_m"])

        if initial_clearance_m < 0.25:
            raise RuntimeError(
                f"Invalid service-task reset: grasp-to-target clearance is only {initial_clearance_m:.3f} m; "
                "the robot must start away from the interaction target."
            )

        print(f"[INFO] Task: {task_id}")
        print(f"[INFO] Instruction: {env.unwrapped.cfg.task_instruction}")
        print(f"[INFO] Initial grasp-to-target clearance: {initial_clearance_m:.3f} m")
        if args_cli.no_camera:
            print("[INFO] Robot RGB disabled for this mechanics-only run.")
        else:
            print(f"[INFO] Robot RGB: {tuple(observations['llm']['rgb'].shape)}")
            print("[INFO] RGB is rendered for the robot-mounted camera but is not sent to the text LLM.")

        if args_cli.controller == "demo":
            program = default_service_program(args_cli.scenario)
            executor = ServiceSkillExecutor(program)
            print(f"[SERVICE] deterministic mechanics-isolation program: {program.summary}")
        else:
            program = None
            request_pool = concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="go2-service-llm")
            print(
                f"[LLM] action_space={args_cli.llm_action_space} control_mode={args_cli.llm_control_mode} "
                f"request_budget={request_budget}; "
                f"waiting {args_cli.llm_start_delay:.1f} s before the initial symbolic request"
            )

        backend = f"local {args_cli.llm_base_url}" if args_cli.llm_base_url else "hosted Hugging Face"
        dashboard = LiveDashboard(
            args_cli.log_dir,
            model=args_cli.llm_model or "deterministic-service-policy",
            provider=args_cli.llm_provider if args_cli.controller == "llm" else "none",
            backend=backend if args_cli.controller == "llm" else "local deterministic executor",
            refresh_s=args_cli.dashboard_refresh,
        )
        dashboard.update(
            status=executor.status if executor is not None else "waiting to request",
            task_id=task_id,
            scenario=args_cli.scenario,
            instruction=env.unwrapped.cfg.task_instruction,
            control_mode=args_cli.llm_control_mode if args_cli.controller == "llm" else "deterministic",
            action_space=args_cli.llm_action_space if args_cli.controller == "llm" else "deterministic",
            request_limit=request_budget if args_cli.controller == "llm" else 0,
            request_count=request_count,
            program=asdict(program) if program is not None else None,
            raw_response="Deterministic demo; no LLM request." if program is not None else None,
            observation=state,
        )
        print(f"[SERVICE] Trace HTML: {dashboard.html_path}")
        print(f"[SERVICE] Event log: {dashboard.jsonl_path}")
        if args_cli.dashboard:
            try:
                dashboard_url = dashboard.start_local_server(args_cli.dashboard_port)
            except OSError as exc:
                dashboard.close("dashboard server failed")
                dashboard = None
                raise RuntimeError(f"Could not publish the dashboard on 127.0.0.1:{args_cli.dashboard_port}.") from exc
            print(f"[SERVICE] Dashboard: {dashboard_url}")

        def policy_status() -> str:
            if executor is not None:
                return executor.status
            if request_future is not None:
                return f"{request_kind or 'LLM'} request pending"
            if llm_stopped:
                return "LLM stopped"
            if replan_pending:
                return "event-triggered replan queued"
            return "waiting for LLM"

        step = 0
        while simulation_app.is_running() and (args_cli.max_steps == 0 or step < args_cli.max_steps):
            step_started = time.time()
            if args_cli.scenario == "open-door":
                state = _door_state(env, action_term, handles)
            else:
                state = _pick_place_state(env, action_term, memory, args_cli.scenario)

            if (
                args_cli.controller == "llm"
                and not llm_stopped
                and request_future is None
                and executor is None
                and not initial_request_completed
                and request_count < request_budget
                and step * env.unwrapped.step_dt >= args_cli.llm_start_delay
            ):
                if args_cli.llm_action_space == "primitive":
                    if args_cli.llm_control_mode == "reactive":
                        prompt = build_primitive_replan_prompt(
                            env.unwrapped.cfg.task_instruction,
                            state,
                            args_cli.scenario,
                            None,
                            execution_history,
                        )
                        request_kind = "reactive_primitive"
                        request_fn = request_primitive_decision
                    else:
                        prompt = build_primitive_prompt(
                            env.unwrapped.cfg.task_instruction,
                            state,
                            args_cli.scenario,
                        )
                        request_kind = "initial_primitive_horizon"
                        request_fn = request_primitive_program
                elif args_cli.llm_control_mode == "reactive":
                    prompt = build_service_reactive_prompt(
                        env.unwrapped.cfg.task_instruction,
                        state,
                        args_cli.scenario,
                        execution_history,
                    )
                    request_kind = "reactive_decision"
                    request_fn = request_service_decision
                else:
                    prompt = build_service_prompt(env.unwrapped.cfg.task_instruction, state, args_cli.scenario)
                    request_kind = "initial_horizon"
                    request_fn = request_service_program
                request_count += 1
                request_prompt = prompt
                request_repair_attempts = 0
                waiting_anchor_world_m = tuple(state["robot"]["root_pose_world"]["position_m"])
                print(
                    f"[LLM] submitting {request_kind} {request_count}/{request_budget} "
                    f"for {args_cli.scenario} to {args_cli.llm_model}"
                )
                request_started_at = time.perf_counter()
                request_future = request_pool.submit(
                    request_fn,
                    prompt,
                    args_cli.scenario,
                    args_cli.llm_model,
                    args_cli.llm_provider,
                    args_cli.llm_base_url,
                    args_cli.llm_timeout,
                )
                if dashboard is not None:
                    dashboard.event(
                        "request_submitted",
                        request_kind=request_kind,
                        request_count=request_count,
                        prompt=prompt,
                        observation=state,
                    )
                    dashboard.update(
                        status=f"{request_kind} request pending",
                        request_kind=request_kind,
                        request_count=request_count,
                        prompt=prompt,
                        request_observation=state,
                    )

            if (
                args_cli.controller == "llm"
                and args_cli.llm_control_mode in ("reactive", "receding-horizon")
                and replan_pending
                and not success_latched
                and not llm_stopped
                and request_future is None
                and executor is None
            ):
                if request_count >= request_budget:
                    llm_stopped = True
                    replan_pending = False
                    actions[:, 0:3] = 0.0
                    print(f"[LLM][SAFE STOP] request budget exhausted ({request_count}/{request_budget})")
                    if dashboard is not None:
                        dashboard.event("request_budget_exhausted", request_count=request_count)
                        dashboard.update(status="request budget exhausted", request_count=request_count)
                else:
                    if args_cli.llm_action_space == "primitive":
                        prompt = build_primitive_replan_prompt(
                            env.unwrapped.cfg.task_instruction,
                            state,
                            args_cli.scenario,
                            initial_plan if args_cli.llm_control_mode == "receding-horizon" else None,
                            execution_history,
                        )
                        request_kind = (
                            "primitive_replan"
                            if args_cli.llm_control_mode == "receding-horizon"
                            else "reactive_primitive"
                        )
                        request_fn = request_primitive_decision
                    elif args_cli.llm_control_mode == "receding-horizon":
                        assert initial_plan is not None
                        prompt = build_service_replan_prompt(
                            env.unwrapped.cfg.task_instruction,
                            state,
                            args_cli.scenario,
                            initial_plan,
                            execution_history,
                        )
                        request_kind = "replan"
                        request_fn = request_service_decision
                    else:
                        prompt = build_service_reactive_prompt(
                            env.unwrapped.cfg.task_instruction,
                            state,
                            args_cli.scenario,
                            execution_history,
                        )
                        request_kind = "reactive_decision"
                        request_fn = request_service_decision
                    request_count += 1
                    replan_pending = False
                    request_prompt = prompt
                    request_repair_attempts = 0
                    waiting_anchor_world_m = tuple(state["robot"]["root_pose_world"]["position_m"])
                    print(
                        f"[LLM] submitting event-triggered replan {request_count}/{request_budget} "
                        f"to {args_cli.llm_model}"
                    )
                    request_started_at = time.perf_counter()
                    request_future = request_pool.submit(
                        request_fn,
                        prompt,
                        args_cli.scenario,
                        args_cli.llm_model,
                        args_cli.llm_provider,
                        args_cli.llm_base_url,
                        args_cli.llm_timeout,
                    )
                    if dashboard is not None:
                        dashboard.event(
                            "request_submitted",
                            request_kind=request_kind,
                            request_count=request_count,
                            prompt=prompt,
                            observation=state,
                            execution_history=execution_history,
                        )
                        dashboard.update(
                            status="replan request pending",
                            request_kind=request_kind,
                            request_count=request_count,
                            prompt=prompt,
                            request_observation=state,
                            execution_history=execution_history,
                        )

            if request_future is not None and request_future.done():
                latency_s = time.perf_counter() - request_started_at
                replacement_request_submitted = False
                try:
                    result = request_future.result()
                except Exception as exc:
                    actions[:, 0:3] = 0.0
                    raw_response = getattr(exc, "raw_response", None)
                    failed_usage = getattr(exc, "usage", None)
                    if dashboard is not None:
                        dashboard.event(
                            "request_failed",
                            request_kind=request_kind,
                            request_count=request_count,
                            error_type=type(exc).__name__,
                            error=str(exc),
                            latency_s=latency_s,
                            usage=failed_usage,
                            raw_response=raw_response,
                        )
                    can_repair = (
                        isinstance(exc, PrimitiveResponseValidationError)
                        and request_repair_attempts < 1
                        and request_count < request_budget
                        and request_fn is not None
                    )
                    if can_repair:
                        request_repair_attempts += 1
                        request_count += 1
                        request_kind = f"{request_kind}_repair"
                        request_prompt = build_primitive_repair_prompt(
                            raw_response or "",
                            str(exc),
                            single_step=request_fn is request_primitive_decision,
                            planning_context=state.get("controller", {}).get("planning_context"),
                        )
                        print(
                            f"[LLM] response validation failed; submitting one JSON repair "
                            f"{request_count}/{request_budget}: {exc}"
                        )
                        request_started_at = time.perf_counter()
                        request_future = request_pool.submit(
                            request_fn,
                            request_prompt,
                            args_cli.scenario,
                            args_cli.llm_model,
                            args_cli.llm_provider,
                            args_cli.llm_base_url,
                            args_cli.llm_timeout,
                        )
                        replacement_request_submitted = True
                        if dashboard is not None:
                            dashboard.event(
                                "request_submitted",
                                request_kind=request_kind,
                                request_count=request_count,
                                prompt=request_prompt,
                                observation=state,
                                repair_of_invalid_response=True,
                            )
                            dashboard.update(
                                status="JSON repair request pending",
                                request_kind=request_kind,
                                request_count=request_count,
                                prompt=request_prompt,
                                request_observation=state,
                                raw_response=None,
                                program=None,
                            )
                    else:
                        llm_stopped = True
                        replan_pending = False
                        print(f"[LLM][SAFE STOP] request or validation failed: {exc}")
                        if dashboard is not None:
                            dashboard.update(
                                status=f"request failed: {type(exc).__name__}",
                                latency_s=latency_s,
                                usage=failed_usage,
                                raw_response=raw_response,
                            )
                else:
                    initial_request_completed = True
                    if request_kind is not None and request_kind.startswith(
                        ("initial_horizon", "initial_primitive_horizon")
                    ):
                        initial_plan = result.program
                        if args_cli.llm_control_mode == "receding-horizon":
                            program = (
                                single_primitive_program(initial_plan)
                                if args_cli.llm_action_space == "primitive"
                                else single_skill_program(initial_plan)
                            )
                        else:
                            program = initial_plan
                    else:
                        program = result.program
                    executor = (
                        PrimitiveExecutor(program)
                        if args_cli.llm_action_space == "primitive"
                        else ServiceSkillExecutor(program)
                    )
                    print(f"[LLM] validated {request_kind}: {program.summary}")
                    print(f"[LLM] raw response: {result.raw_response}")
                    if dashboard is not None:
                        dashboard.event(
                            "request_completed",
                            request_kind=request_kind,
                            request_count=request_count,
                            latency_s=latency_s,
                            usage=result.usage,
                            raw_response=result.raw_response,
                            program=asdict(program),
                            initial_plan=asdict(initial_plan) if initial_plan is not None else None,
                        )
                        dashboard.update(
                            status=executor.status,
                            request_kind=request_kind,
                            request_count=request_count,
                            latency_s=latency_s,
                            usage=result.usage,
                            raw_response=result.raw_response,
                            program=asdict(program),
                            initial_plan=asdict(initial_plan) if initial_plan is not None else None,
                        )
                if not replacement_request_submitted:
                    request_future = None
                    request_kind = None
                    request_fn = None
                    request_prompt = None

            if executor is None and (request_future is not None or llm_stopped):
                actions[:, 0:3] = 0.0
                if waiting_anchor_world_m is not None:
                    apply_base_station_keep(actions, action_term, state, waiting_anchor_world_m)

            terminal_executor = None
            if executor is not None:
                transition = executor.apply(actions, action_term, state, env.unwrapped.step_dt)
                if transition:
                    print(transition)
                    if dashboard is not None:
                        dashboard.event("action_transition", step=step, transition=transition, status=executor.status)
                        dashboard.update(status=executor.status)
                if executor.finished or executor.failed:
                    terminal_executor = executor

            with torch.inference_mode():
                observations, _, terminated, truncated, _ = env.step(actions)
            step += 1
            if args_cli.scenario == "open-door":
                state = _door_state(env, action_term, handles)
            else:
                state = _pick_place_state(env, action_term, memory, args_cli.scenario)

            exit_after_success = False
            if not success_latched and _success(state, args_cli.scenario):
                success_latched = True
                actions[:, 0:3] = 0.0
                print(f"[SUCCESS] {args_cli.scenario} physical success contract satisfied at step={step}")
                if dashboard is not None:
                    dashboard.event("task_success", scenario=args_cli.scenario, step=step, state=state)
                    dashboard.update(status="physical task success")
                if args_cli.exit_on_success:
                    exit_after_success = True

            if (
                terminal_executor is not None
                and args_cli.controller == "llm"
                and args_cli.llm_control_mode in ("reactive", "receding-horizon")
            ):
                decision = terminal_executor.program.steps[0].op
                outcome = "failed" if terminal_executor.failed else "completed"
                if args_cli.scenario == "open-door":
                    progress = {
                        "hinge_angle_deg": state["door"]["hinge_angle_deg"],
                        "opened": state["door"]["opened"],
                    }
                else:
                    obj = state["object"]
                    progress = {
                        "bilateral_contact": obj["bilateral_contact"],
                        "lifted": obj["lifted"],
                        "lifted_once": obj["lifted_once"],
                        "in_tray": obj["in_tray"],
                        "speed_mps": obj["speed_mps"],
                    }
                history_entry = {
                    "action": decision,
                    "action_space": args_cli.llm_action_space,
                    "outcome": outcome,
                    "detail": terminal_executor.message,
                    "sim_time_s": round(step * env.unwrapped.step_dt, 4),
                    "physical_task_success": success_latched,
                    "progress": progress,
                }
                execution_history.append(history_entry)
                executor = None
                if success_latched or decision == "stop":
                    llm_stopped = True
                else:
                    replan_pending = True
                if dashboard is not None:
                    dashboard.event("action_outcome", **history_entry)
                    dashboard.update(
                        status=policy_status(),
                        execution_history=execution_history,
                        observation=state,
                    )

            if exit_after_success:
                break

            if step % args_cli.print_interval == 0:
                wbc_rms = float(torch.sqrt(torch.mean(torch.square(action_term.low_level_actions[0]))).item())
                actor_command = _rounded(command_term.vel_command_b[0, :3])
                if args_cli.scenario == "open-door":
                    detail = (
                        f"door={state['door']['hinge_angle_deg']:.1f}deg "
                        f"handle_root={state['door']['handle_center']['root_position_m']}"
                    )
                else:
                    obj = state["object"]
                    detail = (
                        f"object={obj['world_position_m']} contacts=L{int(obj['left_finger_contact'])}"
                        f"/R{int(obj['right_finger_contact'])} lifted_once={int(obj['lifted_once'])} "
                        f"in_tray={int(obj['in_tray'])} speed={obj['speed_mps']:.3f}m/s "
                        f"grasp={state['robot']['grasp_midpoint']['world_position_m']}"
                    )
                status = policy_status()
                print(f"[INFO] step={step} WBC_rms={wbc_rms:.3f} actor_cmd={actor_command} status={status!r} {detail}")

            if dashboard is not None and step % args_cli.log_interval == 0:
                status = policy_status()
                dashboard.update(status=status, observation=state)
                dashboard.sample(
                    observation=state,
                    telemetry={
                        "step": step,
                        "sim_time_s": round(step * env.unwrapped.step_dt, 4),
                        "policy_status": status,
                        "requested_action": _rounded(actions[0]),
                        "applied_action": _rounded(action_term.processed_actions[0]),
                        "physical_base_command": _rounded(command_term.vel_command_b[0, :3]),
                        "measured_root_velocity": state["robot"]["measured_root_velocity"],
                        "posture_and_gripper": state["robot"]["posture_and_gripper"],
                        "gripper_command_closed_fraction": state["robot"]["posture_and_gripper"][
                            "gripper_command_closed_fraction"
                        ],
                        "gripper_measured_closed_fraction": state["robot"]["posture_and_gripper"][
                            "gripper_measured_closed_fraction"
                        ],
                        "task_state": state.get("door", state.get("object")),
                    },
                )

            dones = terminated | truncated
            if torch.any(dones):
                print("[WARN] environment reset unexpectedly; service runner keeps success terminations disabled")

            if args_cli.real_time:
                remaining = env.unwrapped.step_dt - (time.time() - step_started)
                if remaining > 0.0:
                    time.sleep(remaining)
    finally:
        if request_pool is not None:
            request_pool.shutdown(wait=False, cancel_futures=True)
        if dashboard is not None:
            final_status = "physical task success" if success_latched else "run finished"
            dashboard.close(final_status)
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
