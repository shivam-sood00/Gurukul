"""Play the Go2+D1 switch scene with live robot RGB and a frozen locomotion policy."""

"""Launch Isaac Sim Simulator first."""

import argparse
import concurrent.futures
import contextlib
import json
import math
import os
import time
import traceback
from dataclasses import asdict

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Run the Go2+D1 RGB switch-press task.")
parser.add_argument(
    "--base-policy",
    required=True,
    help="Exported policy.pt (or export/run directory) for the Go2+D1 LegWBC-AsyncArm policy.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of parallel switch scenes.")
parser.add_argument(
    "--max_steps",
    type=int,
    default=0,
    help="Maximum play steps; 0 runs until the simulator or RGB window is closed.",
)
parser.add_argument("--print_interval", type=int, default=30, help="Steps between switch-state prints.")
parser.add_argument(
    "--base-command",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    metavar=("VX", "VY", "WZ"),
    help="Initial normalized base command; the default is a stationary [0, 0, 0] command.",
)
parser.add_argument("--rgb-env-index", type=int, default=0, help="Environment index shown in the RGB window.")
parser.add_argument("--rgb-window-scale", type=float, default=2.0, help="Live RGB window scale factor.")
parser.add_argument("--no-rgb-view", action="store_true", help="Disable the live robot-camera window.")
parser.add_argument(
    "--llm-model",
    default=None,
    help=(
        "Enable the text-only code policy with this hosted Hugging Face or local model ID; "
        "for example Qwen/Qwen3-4B-Instruct-2507."
    ),
)
parser.add_argument(
    "--llm-provider",
    default="auto",
    help="Hugging Face Inference Provider used for hosted requests (default: auto).",
)
parser.add_argument(
    "--llm-base-url",
    default=None,
    help="Optional local OpenAI-compatible server URL, for example http://127.0.0.1:8000.",
)
parser.add_argument("--llm-timeout", type=float, default=60.0, help="Timeout in seconds for one LLM request.")
parser.add_argument(
    "--llm-start-delay",
    type=float,
    default=1.0,
    help="Seconds to hold after reset before submitting the initial symbolic-state request.",
)
parser.add_argument(
    "--llm-max-requests",
    type=int,
    default=1,
    help="Maximum requests per run; 0 is unlimited. The hosted-credit-safe default is one request.",
)
parser.add_argument(
    "--llm-instruction",
    default=None,
    help="Optional task instruction override sent to the text model.",
)
parser.add_argument(
    "--print-llm-state",
    action="store_true",
    help="Print the exact symbolic JSON state sent with each LLM request.",
)
parser.add_argument(
    "--llm-dashboard",
    action="store_true",
    help="Write a self-refreshing live HTML trace and append-only JSONL event log.",
)
parser.add_argument(
    "--llm-log-dir",
    default="logs/llm/go2_d1_switch",
    help="Parent directory for timestamped dashboard runs.",
)
parser.add_argument(
    "--llm-log-interval",
    type=int,
    default=10,
    help="Control steps between dashboard/JSONL telemetry samples (default: 10, or 5 Hz at 50 Hz control).",
)
parser.add_argument(
    "--llm-dashboard-refresh",
    type=float,
    default=0.5,
    help="Browser auto-refresh interval in seconds.",
)
parser.add_argument(
    "--llm-dashboard-port",
    type=int,
    default=8765,
    help="Loopback HTTP port for the live dashboard (default: 8765; only 127.0.0.1 is exposed).",
)
parser.add_argument(
    "--llm-input-price-per-million",
    type=float,
    default=None,
    help="Optional current provider input price in USD per million tokens for cost estimates.",
)
parser.add_argument(
    "--llm-output-price-per-million",
    type=float,
    default=None,
    help="Optional current provider output price in USD per million tokens for cost estimates.",
)
parser.add_argument(
    "--reset-on-success",
    action="store_true",
    help="Reset the scene after pressing the switch; by default the interactive runner stops and holds at success.",
)
parser.add_argument(
    "--no-real-time",
    dest="real_time",
    action="store_false",
    help="Run as fast as possible instead of pacing the play loop to the environment step time.",
)
parser.add_argument(
    "--use_fabric",
    action="store_true",
    help="Enable Fabric scene I/O (the default and required for GPU-rendered transform updates).",
)
parser.add_argument(
    "--no-fabric",
    dest="use_fabric",
    action="store_false",
    help="Disable Fabric only for CPU/debug use; GPU robot and attached-camera visuals will remain stale.",
)
parser.set_defaults(real_time=True, use_fabric=True)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
if args_cli.max_steps < 0:
    parser.error("--max_steps must be >= 0.")
if args_cli.rgb_env_index < 0:
    parser.error("--rgb-env-index must be >= 0.")
if args_cli.rgb_window_scale <= 0.0:
    parser.error("--rgb-window-scale must be > 0.")
if args_cli.llm_timeout <= 0.0:
    parser.error("--llm-timeout must be > 0.")
if args_cli.llm_start_delay < 0.0:
    parser.error("--llm-start-delay must be >= 0.")
if args_cli.llm_max_requests < 0:
    parser.error("--llm-max-requests must be >= 0.")
if args_cli.llm_dashboard and not args_cli.llm_model:
    parser.error("--llm-dashboard requires --llm-model.")
if args_cli.llm_log_interval <= 0:
    parser.error("--llm-log-interval must be > 0.")
if args_cli.llm_dashboard_refresh <= 0.0:
    parser.error("--llm-dashboard-refresh must be > 0.")
if not 1 <= args_cli.llm_dashboard_port <= 65535:
    parser.error("--llm-dashboard-port must be in [1, 65535].")
if (args_cli.llm_input_price_per_million is None) != (args_cli.llm_output_price_per_million is None):
    parser.error("Provide both --llm-input-price-per-million and --llm-output-price-per-million, or neither.")
if args_cli.llm_input_price_per_million is not None and (
    args_cli.llm_input_price_per_million < 0.0 or args_cli.llm_output_price_per_million < 0.0
):
    parser.error("LLM token prices must be >= 0.")
if any(abs(value) > 1.0 for value in args_cli.base_command):
    parser.error("--base-command values must be normalized to [-1, 1].")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""The environment can be imported after Isaac Sim starts."""

import gymnasium as gym
import torch

try:
    if os.path.isdir("/usr/share/fonts/truetype/dejavu"):
        os.environ.setdefault("QT_QPA_FONTDIR", "/usr/share/fonts/truetype/dejavu")
    import cv2
except ModuleNotFoundError:
    cv2 = None

import Gurukul.tasks  # noqa: F401
from code_policy import (
    PolicyProgram,
    PolicyStep,
    build_policy_prompt,
    inverse_rotate_vector_wxyz,
    request_policy_program,
    root_point_to_world,
    rotate_vector_wxyz,
    world_point_to_root,
)
from live_dashboard import LiveDashboard

from isaaclab_tasks.utils import parse_env_cfg

TASK_ID = "Gurukul-Isaac-LLM-Go2-D1-Press-Switch-v0"
RGB_WINDOW_NAME = "Go2+D1 Robot RGB"


def _initial_hold_actions(env, base_command: tuple[float, float, float]) -> tuple[torch.Tensor, object]:
    """Hold the measured workspace-ready arm pose with the selected base command."""
    action_term = env.unwrapped.action_manager.get_term("wbc_command")
    actions = action_term.processed_actions.detach().clone()
    actions[:, 0:3] = torch.as_tensor(base_command, device=actions.device, dtype=actions.dtype)
    return actions, action_term


def _rounded(values, digits: int = 6) -> list[float]:
    """Convert a tensor or sequence to compact JSON-safe floats."""
    if isinstance(values, torch.Tensor):
        values = values.detach().cpu().tolist()
    return [round(float(value), digits) for value in values]


def _resolve_symbolic_frame_handles(env, action_term) -> dict[str, int]:
    """Resolve the exact rigid bodies and joint used by the symbolic state."""
    switch = env.unwrapped.scene["switch"]
    button_body_ids = switch.find_bodies(env.unwrapped.cfg.switch_button_body_name)[0]
    button_joint_ids = switch.find_joints("button_joint")[0]
    if len(button_body_ids) != 1 or len(button_joint_ids) != 1:
        raise RuntimeError(
            "The code policy requires exactly one button body and one button_joint; "
            f"found bodies={button_body_ids}, joints={button_joint_ids}."
        )
    return {
        "button_body_id": int(button_body_ids[0]),
        "button_joint_id": int(button_joint_ids[0]),
        "ee_body_id": int(action_term._ee_body_id),
    }


def _collect_symbolic_state(env, action_term, handles: dict[str, int], env_index: int) -> dict:
    """Collect frame-explicit geometry without sending an RGB image."""
    robot = env.unwrapped.scene["robot"]
    switch = env.unwrapped.scene["switch"]
    cfg = env.unwrapped.cfg

    root_position_w = _rounded(robot.data.root_pos_w[env_index])
    root_quaternion_w = _rounded(robot.data.root_quat_w[env_index])
    ee_position_w = _rounded(robot.data.body_pos_w[env_index, handles["ee_body_id"]])
    grasp_position_w = _rounded(
        robot.data.body_pos_w[env_index].index_select(0, action_term._grasp_body_ids).mean(dim=0)
    )
    button_pose_w = switch.data.body_pose_w[env_index, handles["button_body_id"]]
    button_center_w = _rounded(button_pose_w[:3])
    button_quaternion_w = _rounded(button_pose_w[3:7])

    press_axis_w = rotate_vector_wxyz(button_quaternion_w, cfg.switch_press_axis_local)
    half_length = float(cfg.switch_button_half_length_m)
    front_surface_w = tuple(button_center_w[axis] - half_length * press_axis_w[axis] for axis in range(3))
    prepress_w = tuple(
        front_surface_w[axis] - float(cfg.switch_prepress_clearance_m) * press_axis_w[axis] for axis in range(3)
    )
    joint_depth_m = float(switch.data.joint_pos[env_index, handles["button_joint_id"]].item())
    remaining_to_activation_m = max(float(cfg.switch_pressed_threshold_m) - joint_depth_m, 0.0)
    activation_surface_w = tuple(
        front_surface_w[axis] + remaining_to_activation_m * press_axis_w[axis] for axis in range(3)
    )

    ee_position_b = world_point_to_root(ee_position_w, root_position_w, root_quaternion_w)
    grasp_position_b = world_point_to_root(grasp_position_w, root_position_w, root_quaternion_w)
    button_center_b = world_point_to_root(button_center_w, root_position_w, root_quaternion_w)
    front_surface_b = world_point_to_root(front_surface_w, root_position_w, root_quaternion_w)
    prepress_b = world_point_to_root(prepress_w, root_position_w, root_quaternion_w)
    activation_surface_b = world_point_to_root(activation_surface_w, root_position_w, root_quaternion_w)
    press_axis_b = inverse_rotate_vector_wxyz(root_quaternion_w, press_axis_w)

    return {
        "schema_version": 1,
        "frames": {
            "world": "fixed Isaac world frame",
            "root": "floating Go2 base frame; +x forward, +y left, +z up",
            "quaternion_order": "wxyz",
        },
        "robot": {
            "root_pose_world": {
                "position_m": root_position_w,
                "quaternion_wxyz": root_quaternion_w,
            },
            "base_linear_velocity_root_mps": _rounded(robot.data.root_lin_vel_b[env_index]),
            "base_angular_velocity_root_rps": _rounded(robot.data.root_ang_vel_b[env_index]),
            "ee_link6_origin": {
                "world_position_m": ee_position_w,
                "root_position_m": _rounded(ee_position_b),
            },
            "grasp_midpoint": {
                "definition": "mean position of Link7_1 and Link7_2; this is the commanded Cartesian point",
                "world_position_m": grasp_position_w,
                "root_position_m": _rounded(grasp_position_b),
            },
        },
        "button": {
            "joint_depth_m": round(joint_depth_m, 6),
            "travel_m": float(cfg.switch_travel_m),
            "pressed_threshold_m": float(cfg.switch_pressed_threshold_m),
            "remaining_to_activation_m": round(remaining_to_activation_m, 6),
            "pressed": joint_depth_m >= float(cfg.switch_pressed_threshold_m),
            "body_center": {
                "world_position_m": button_center_w,
                "root_position_m": _rounded(button_center_b),
                "world_quaternion_wxyz": button_quaternion_w,
            },
            "press_axis": {
                "meaning": "positive direction moves the button into the housing",
                "world_unit_vector": _rounded(press_axis_w),
                "root_unit_vector": _rounded(press_axis_b),
            },
            "front_surface": {
                "world_position_m": _rounded(front_surface_w),
                "root_position_m": _rounded(front_surface_b),
            },
            "prepress": {
                "meaning": "collision-free point outside the button face",
                "world_position_m": _rounded(prepress_w),
                "root_position_m": _rounded(prepress_b),
            },
            "activation_surface": {
                "meaning": "front-surface location at the pressed threshold",
                "world_position_m": _rounded(activation_surface_w),
                "root_position_m": _rounded(activation_surface_b),
            },
        },
        "controller": {
            "cartesian_target": "grasp_midpoint in the current root frame",
            "grasp_workspace_root_m": [list(map(float, bounds)) for bounds in action_term.cfg.ee_pos_range],
            "base_velocity_scale_mps_mps_rps": list(map(float, action_term.cfg.velocity_scale)),
            "wrist_roll_range_rad": list(map(float, action_term.cfg.wrist_roll_range)),
        },
    }


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _walk_command(error: float, gain: float, minimum: float, maximum: float, tolerance: float) -> float:
    """Return zero inside tolerance and a learned-policy-compatible command outside it."""
    if abs(error) <= tolerance:
        return 0.0
    magnitude = _clamp(abs(error) * gain, minimum, maximum)
    return math.copysign(magnitude, error)


def _set_physical_base_velocity(actions, action_term, env_index: int, command_b: tuple[float, float, float]) -> None:
    """Convert a physical root/body-frame velocity to the normalized action contract."""
    for axis, value in enumerate(command_b):
        scale = float(action_term.cfg.velocity_scale[axis])
        actions[env_index, axis] = _clamp(float(value) / scale, -1.0, 1.0)


def _set_grasp_target_root(
    actions,
    action_term,
    env_index: int,
    target_b: tuple[float, float, float],
    wrist_roll_rad: float = 0.0,
    gripper: str = "open",
) -> None:
    """Map a physical root-frame grasp target into the normalized Cartesian action."""
    for axis, (lower, upper) in enumerate(action_term.cfg.ee_pos_range):
        normalized = 2.0 * (float(target_b[axis]) - float(lower)) / (float(upper) - float(lower)) - 1.0
        actions[env_index, 5 + axis] = _clamp(normalized, -1.0, 1.0)
    wrist_lower, wrist_upper = action_term.cfg.wrist_roll_range
    wrist_normalized = 2.0 * (wrist_roll_rad - float(wrist_lower)) / (float(wrist_upper) - float(wrist_lower)) - 1.0
    actions[env_index, 8] = _clamp(wrist_normalized, -1.0, 1.0)
    actions[env_index, 9] = 1.0 if gripper == "closed" else -1.0


class _CodePolicyExecutor:
    """Execute a validated language-model program through closed-loop primitives."""

    def __init__(self, program: PolicyProgram, env_index: int = 0):
        self.program = program
        self.env_index = env_index
        self.step_index = 0
        self.step_elapsed_s = 0.0
        self.finished = False
        self.failed = False
        self.message = "program ready"

    @property
    def status(self) -> str:
        if self.failed or self.finished:
            return self.message
        step = self.program.steps[self.step_index]
        return f"step {self.step_index + 1}/{len(self.program.steps)}: {step.op}"

    def _advance(self, detail: str) -> str:
        event = f"[LLM] completed {self.status}: {detail}"
        self.step_index += 1
        self.step_elapsed_s = 0.0
        if self.step_index >= len(self.program.steps):
            self.finished = True
            self.message = "program complete"
        return event

    def _fail(self, actions, detail: str) -> str:
        actions[self.env_index, 0:3] = 0.0
        self.failed = True
        self.message = f"program stopped: {detail}"
        return f"[LLM][SAFE STOP] {detail}"

    def apply(self, actions, action_term, state: dict, step_dt: float) -> str | None:
        """Apply one control tick and return a transition message when relevant."""
        if self.finished or self.failed:
            actions[self.env_index, 0:3] = 0.0
            return None

        step: PolicyStep = self.program.steps[self.step_index]
        self.step_elapsed_s += float(step_dt)
        root_pose = state["robot"]["root_pose_world"]
        root_position_w = root_pose["position_m"]
        root_quaternion_w = root_pose["quaternion_wxyz"]

        if step.op == "approach_button":
            surface_b = state["button"]["front_surface"]["root_position_m"]
            x_error = float(surface_b[0]) - step.standoff_m
            y_error = float(surface_b[1])
            heading_error = math.atan2(float(surface_b[1]), max(float(surface_b[0]), 1.0e-6))
            command_b = (
                _walk_command(x_error, gain=2.0, minimum=0.45, maximum=0.75, tolerance=0.04),
                _walk_command(y_error, gain=1.0, minimum=0.12, maximum=0.25, tolerance=0.04),
                _walk_command(heading_error, gain=1.2, minimum=0.20, maximum=0.60, tolerance=0.10),
            )
            _set_physical_base_velocity(actions, action_term, self.env_index, command_b)
            reached = abs(x_error) <= 0.04 and abs(y_error) <= 0.04 and abs(heading_error) <= 0.10
            if reached:
                actions[self.env_index, 0:3] = 0.0
                return self._advance(f"button face is {surface_b[0]:.3f} m forward in root")
            if self.step_elapsed_s >= step.timeout_s:
                return self._fail(actions, f"approach_button timed out after {step.timeout_s:.1f} s")
            return None

        if step.op == "move_grasp":
            actions[self.env_index, 0:3] = 0.0
            if step.target_position_m is None or step.target_frame is None:
                return self._fail(actions, "move_grasp has no validated target")
            if step.target_frame == "world":
                target_w = step.target_position_m
                target_b = world_point_to_root(target_w, root_position_w, root_quaternion_w)
                measured = state["robot"]["grasp_midpoint"]["world_position_m"]
            else:
                target_b = step.target_position_m
                target_w = root_point_to_world(target_b, root_position_w, root_quaternion_w)
                measured = state["robot"]["grasp_midpoint"]["root_position_m"]
            _set_grasp_target_root(
                actions,
                action_term,
                self.env_index,
                target_b,
                wrist_roll_rad=step.wrist_roll_rad,
                gripper=step.gripper,
            )
            target_for_error = target_w if step.target_frame == "world" else target_b
            error_m = math.sqrt(sum((float(measured[axis]) - target_for_error[axis]) ** 2 for axis in range(3)))
            if error_m <= step.tolerance_m:
                return self._advance(f"grasp error={error_m:.3f} m in {step.target_frame} frame")
            if self.step_elapsed_s >= step.timeout_s:
                return self._fail(
                    actions,
                    f"move_grasp timed out with {error_m:.3f} m {step.target_frame}-frame error",
                )
            return None

        if step.op == "press_button":
            actions[self.env_index, 0:3] = 0.0
            button = state["button"]
            front_w = button["front_surface"]["world_position_m"]
            axis_w = button["press_axis"]["world_unit_vector"]
            travel_m = float(button["remaining_to_activation_m"]) + step.overtravel_m
            target_w = tuple(float(front_w[axis]) + travel_m * float(axis_w[axis]) for axis in range(3))
            target_b = world_point_to_root(target_w, root_position_w, root_quaternion_w)
            _set_grasp_target_root(actions, action_term, self.env_index, target_b, gripper="open")
            if button["pressed"]:
                return self._advance("button crossed the physical joint threshold")
            if self.step_elapsed_s >= step.timeout_s:
                return self._fail(actions, f"press_button timed out after {step.timeout_s:.1f} s")
            return None

        if step.op == "hold":
            actions[self.env_index, 0:3] = 0.0
            if self.step_elapsed_s >= step.duration_s:
                return self._advance(f"held for {step.duration_s:.1f} s")
            return None

        actions[self.env_index, 0:3] = 0.0
        self.finished = True
        self.message = "program stopped"
        return "[LLM] stop primitive reached; base command is zero"


def _show_rgb_frame(
    observations,
    actions,
    action_term,
    command_term,
    robot,
    root_origin_xy,
    initial_rgb,
    llm_status: str,
    env_index: int,
    window_scale: float,
) -> int:
    """Show one robot-camera frame and return its OpenCV key code."""
    rgb = observations["llm"]["rgb"][env_index, ..., :3].detach().cpu()
    if rgb.dtype != torch.uint8:
        rgb = torch.clamp(rgb, 0.0, 1.0).mul(255.0).to(torch.uint8)
    frame = cv2.cvtColor(rgb.numpy(), cv2.COLOR_RGB2BGR)
    if window_scale != 1.0:
        frame = cv2.resize(
            frame,
            None,
            fx=window_scale,
            fy=window_scale,
            interpolation=cv2.INTER_NEAREST,
        )

    switch = observations["llm"]["switch"][env_index].detach().cpu().tolist()
    low_level = action_term.low_level_actions[env_index]
    wbc_rms = torch.sqrt(torch.mean(torch.square(low_level))).item()
    base_height = robot.data.root_pos_w[env_index, 2].item()
    requested_command = actions[env_index, :3].detach().cpu().tolist()
    applied_command = action_term.processed_actions[env_index, :3].detach().cpu().tolist()
    physical_command = command_term.vel_command_b[env_index, :3].detach().cpu().tolist()
    root_velocity = robot.data.root_lin_vel_b[env_index, :2].detach().cpu().tolist()
    root_yaw_rate = robot.data.root_ang_vel_b[env_index, 2].item()
    travel = torch.linalg.vector_norm(robot.data.root_pos_w[env_index, :2] - root_origin_xy[env_index]).item()
    rgb_delta = torch.mean(
        torch.abs(observations["llm"]["rgb"][env_index, ..., :3].float() - initial_rgb[env_index].float())
    ).item()
    leg_speed = torch.sqrt(torch.mean(torch.square(robot.data.joint_vel[env_index, action_term._leg_joint_ids]))).item()
    cv2.putText(
        frame,
        f"policy: RUN | arm: HOLD | norm req/applied vx: {requested_command[0]:+.2f}/{applied_command[0]:+.2f}",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"actor cmd: [{physical_command[0]:+.2f}, {physical_command[1]:+.2f}, {physical_command[2]:+.2f}]",
        (8, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"body vel: [{root_velocity[0]:+.2f}, {root_velocity[1]:+.2f}, {root_yaw_rate:+.2f}] | "
        f"travel: {travel:.2f}m | WBC: {wbc_rms:.2f} | qd: {leg_speed:.2f} | z: {base_height:.3f}",
        (8, 66),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"switch: {switch[0]:.3f}/{switch[1]:.0f} | RGB delta: {rgb_delta:.2f} | "
        "W/S vx  A/D vy  J/L yaw  X stop  Esc quit",
        (8, 88),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        frame,
        f"text policy: {llm_status} | P replan",
        (8, 110),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.43,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.imshow(RGB_WINDOW_NAME, frame)
    return cv2.waitKey(1) & 0xFF


def _apply_rgb_base_key(actions: torch.Tensor, key: int, increment: float = 0.10) -> bool:
    """Apply one persistent normalized base-command key; return false to exit."""
    if key in (27, ord("q")):
        return False
    if key == ord("w"):
        actions[:, 0] += increment
    elif key == ord("s"):
        actions[:, 0] -= increment
    elif key == ord("a"):
        actions[:, 1] += increment
    elif key == ord("d"):
        actions[:, 1] -= increment
    elif key == ord("j"):
        actions[:, 2] += increment
    elif key == ord("l"):
        actions[:, 2] -= increment
    elif key in (ord("x"), ord("0"), ord(" ")):
        actions[:, 0:3] = 0.0
    actions[:, 0:3].clamp_(-1.0, 1.0)
    return True


def main() -> None:  # noqa: C901
    """Play the frozen WBC from a zero-default base command while holding D1 ready."""
    env_cfg = parse_env_cfg(
        TASK_ID,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=args_cli.use_fabric,
    )
    env_cfg.wbc_policy_path = args_cli.base_policy
    env_cfg.actions.wbc_command.policy_path = args_cli.base_policy
    if not args_cli.reset_on_success:
        # The registered task terminates immediately at success. For a visual
        # interactive runner, retain the successful scene instead of making a
        # short forward rollout look stationary through rapid teleport/reset cycles.
        env_cfg.terminations.switch_pressed = None
    env = gym.make(TASK_ID, cfg=env_cfg)
    if args_cli.rgb_env_index >= env.unwrapped.num_envs:
        env.close()
        raise ValueError(f"--rgb-env-index {args_cli.rgb_env_index} is outside {env.unwrapped.num_envs} environments.")
    if args_cli.llm_model and env.unwrapped.num_envs != 1:
        env.close()
        raise ValueError("The text code policy currently requires --num_envs 1 so one state maps to one robot.")

    observations, _ = env.reset()
    initial_rgb = observations["llm"]["rgb"][..., :3].detach().clone()
    previous_rgb = initial_rgb.clone()
    rgb_update_counts = torch.zeros(env.unwrapped.num_envs, device=initial_rgb.device, dtype=torch.long)
    print(f"[INFO] Instruction: {env.unwrapped.cfg.task_instruction}")
    print(f"[INFO] Action order: {env.unwrapped.cfg.llm_action_order}")
    print(f"[INFO] RGB shape: {tuple(observations['llm']['rgb'].shape)}")
    print(
        f"[INFO] Rendering diagnostics: use_fabric={env.unwrapped.cfg.sim.use_fabric}, "
        f"render_mode={env.render_mode}, viewer_origin={env.unwrapped.cfg.viewer.origin_type}"
    )

    # The action term seeds this command from the measured reset grasp pose.
    # Reusing it avoids moving the arm to the normalized workspace midpoint.
    actions, action_term = _initial_hold_actions(env, tuple(args_cli.base_command))
    robot = env.unwrapped.scene["robot"]
    root_origin_xy = robot.data.root_pos_w[:, :2].detach().clone()
    command_term = env.unwrapped.command_manager.get_term("base_velocity")
    symbolic_handles = _resolve_symbolic_frame_handles(env, action_term)
    print(f"[INFO] Hold action: {actions[0].detach().cpu().tolist()}")
    print(
        "[INFO] The frozen leg policy runs continuously; HOLD refers only to the D1 arm target. "
        "The RGB overlay reports the normalized request and physical actor command separately."
    )

    rgb_view_enabled = not args_cli.no_rgb_view and cv2 is not None
    if not args_cli.no_rgb_view and cv2 is None:
        print("[WARN] OpenCV is unavailable; continuing without the live RGB window.")
    if rgb_view_enabled:
        try:
            cv2.namedWindow(RGB_WINDOW_NAME, cv2.WINDOW_NORMAL)
        except cv2.error as exc:
            print(f"[WARN] OpenCV HighGUI is unavailable; continuing without the live RGB window: {exc}")
            rgb_view_enabled = False
        else:
            print(
                "[INFO] RGB controls: W/S forward/back, A/D strafe, J/L yaw, X/space stop, "
                "P request/replan, Esc/q quit."
            )

    llm_enabled = bool(args_cli.llm_model)
    llm_instruction = args_cli.llm_instruction or env.unwrapped.cfg.task_instruction
    llm_pool = (
        concurrent.futures.ThreadPoolExecutor(max_workers=1, thread_name_prefix="go2-code-policy")
        if llm_enabled
        else None
    )
    llm_future = None
    llm_executor = None
    llm_request_count = 0
    llm_replan_requested = False
    llm_status = "disabled"
    llm_request_started_at = None
    llm_backend = f"local {args_cli.llm_base_url}" if args_cli.llm_base_url else "hosted Hugging Face"
    dashboard = None
    if args_cli.llm_dashboard:
        dashboard = LiveDashboard(
            args_cli.llm_log_dir,
            model=args_cli.llm_model,
            provider=args_cli.llm_provider,
            backend=llm_backend,
            refresh_s=args_cli.llm_dashboard_refresh,
            input_price_per_million=args_cli.llm_input_price_per_million,
            output_price_per_million=args_cli.llm_output_price_per_million,
        )
        dashboard.update(
            request_limit=args_cli.llm_max_requests or "unlimited",
            status="waiting to request",
            task_id=TASK_ID,
            scenario="press-switch",
            instruction=llm_instruction,
            control_mode="one-shot",
            action_space="semantic",
        )
        try:
            dashboard_url = dashboard.start_local_server(args_cli.llm_dashboard_port)
        except OSError as exc:
            dashboard.close("dashboard server failed")
            env.close()
            raise RuntimeError(
                f"Could not publish the dashboard on 127.0.0.1:{args_cli.llm_dashboard_port}. "
                "Choose another --llm-dashboard-port."
            ) from exc
        print(f"[LLM] Open dashboard in your browser: {dashboard_url}")
        print(f"[LLM] Live dashboard: {dashboard.html_path}")
        print(f"[LLM] Append-only event log: {dashboard.jsonl_path}")

    def request_budget_available() -> bool:
        return args_cli.llm_max_requests == 0 or llm_request_count < args_cli.llm_max_requests

    def submit_llm_request(state: dict) -> None:
        nonlocal llm_future, llm_request_count, llm_request_started_at, llm_status
        if llm_pool is None or llm_future is not None:
            return
        if not request_budget_available():
            llm_status = "request limit reached"
            print(f"[LLM] request not submitted: --llm-max-requests={args_cli.llm_max_requests} exhausted.")
            if dashboard is not None:
                dashboard.event(
                    "request_rejected",
                    reason="request limit reached",
                    request_limit=args_cli.llm_max_requests,
                )
                dashboard.update(status=llm_status)
            return
        prompt = build_policy_prompt(llm_instruction, state)
        if args_cli.print_llm_state:
            print("[LLM] symbolic state (no image sent):")
            print(json.dumps(state, indent=2, sort_keys=True))
        llm_request_count += 1
        source = args_cli.llm_base_url or f"Hugging Face provider={args_cli.llm_provider}"
        print(f"[LLM] submitting text-only request {llm_request_count} to {source}; model={args_cli.llm_model}")
        llm_request_started_at = time.perf_counter()
        if dashboard is not None:
            dashboard.event(
                "request_submitted",
                request_index=llm_request_count,
                model=args_cli.llm_model,
                provider=args_cli.llm_provider,
                source=source,
                prompt=prompt,
                observation=state,
            )
            dashboard.update(
                status=f"request {llm_request_count} pending",
                request_count=llm_request_count,
                prompt=prompt,
                request_observation=state,
                raw_response=None,
                program=None,
                usage=None,
                latency_s=None,
            )
        llm_future = llm_pool.submit(
            request_policy_program,
            prompt,
            args_cli.llm_model,
            args_cli.llm_provider,
            args_cli.llm_base_url,
            args_cli.llm_timeout,
        )
        llm_status = f"request {llm_request_count} pending"

    if llm_enabled:
        actions[:, 0:3] = 0.0
        print(
            f"[LLM] text-only code policy enabled with {args_cli.llm_model} via {llm_backend}. "
            "Only symbolic state is sent; RGB pixels are not included."
        )
        print(
            "[LLM] Safety contract: validated JSON primitives only, root/body-frame base velocity, "
            "world-to-root Cartesian targets, no eval/exec."
        )
        llm_status = "waiting to request"

    step = 0
    success_latched = torch.zeros(env.unwrapped.num_envs, device=env.unwrapped.device, dtype=torch.bool)
    while simulation_app.is_running() and (args_cli.max_steps == 0 or step < args_cli.max_steps):
        step_start = time.time()
        if llm_enabled:
            symbolic_state = _collect_symbolic_state(env, action_term, symbolic_handles, env_index=0)
            initial_request_due = (
                llm_request_count == 0
                and step * env.unwrapped.step_dt >= args_cli.llm_start_delay
                and llm_future is None
            )
            if llm_replan_requested or initial_request_due:
                actions[0, 0:3] = 0.0
                llm_executor = None
                submit_llm_request(symbolic_state)
                llm_replan_requested = False

            if llm_future is not None and llm_future.done():
                request_latency_s = (
                    time.perf_counter() - llm_request_started_at if llm_request_started_at is not None else None
                )
                try:
                    request_result = llm_future.result()
                except Exception as exc:
                    actions[0, 0:3] = 0.0
                    llm_status = f"request failed: {type(exc).__name__}"
                    print(f"[LLM][SAFE STOP] request or validation failed: {exc}")
                    if dashboard is not None:
                        dashboard.event(
                            "request_failed",
                            request_index=llm_request_count,
                            latency_s=request_latency_s,
                            error_type=type(exc).__name__,
                            error=str(exc),
                        )
                        dashboard.update(status=llm_status, latency_s=request_latency_s)
                else:
                    program = request_result.program
                    raw_response = request_result.raw_response
                    llm_executor = _CodePolicyExecutor(program)
                    llm_status = llm_executor.status
                    print(f"[LLM] validated program: {program.summary or '(no summary)'}")
                    for program_index, program_step in enumerate(program.steps, start=1):
                        print(f"[LLM]   {program_index}: {program_step}")
                    print(f"[LLM] raw response: {raw_response}")
                    if dashboard is not None:
                        serialized_program = asdict(program)
                        dashboard.event(
                            "request_completed",
                            request_index=llm_request_count,
                            latency_s=request_latency_s,
                            usage=request_result.usage,
                            raw_response=raw_response,
                            program=serialized_program,
                        )
                        dashboard.update(
                            status=llm_status,
                            latency_s=request_latency_s,
                            usage=request_result.usage,
                            raw_response=raw_response,
                            program=serialized_program,
                        )
                llm_future = None
                llm_request_started_at = None

            if llm_executor is not None:
                transition = llm_executor.apply(
                    actions,
                    action_term,
                    symbolic_state,
                    env.unwrapped.step_dt,
                )
                llm_status = llm_executor.status
                if transition:
                    print(transition)
                    if dashboard is not None:
                        dashboard.event(
                            "primitive_transition",
                            step=step,
                            sim_time_s=round(step * env.unwrapped.step_dt, 4),
                            transition=transition,
                            policy_status=llm_status,
                        )
                        dashboard.update(status=llm_status)

        with torch.inference_mode():
            observations, _, terminated, truncated, _ = env.step(actions)
        step += 1
        current_rgb = observations["llm"]["rgb"][..., :3]
        rgb_changed = torch.any(current_rgb != previous_rgb, dim=tuple(range(1, current_rgb.ndim)))
        rgb_update_counts += rgb_changed
        previous_rgb.copy_(current_rgb)
        switch_pressed = observations["llm"]["switch"][:, 1] > 0.5
        if not args_cli.reset_on_success:
            new_success = switch_pressed & ~success_latched
            if torch.any(new_success):
                successful_envs = torch.nonzero(new_success, as_tuple=False).flatten().detach().cpu().tolist()
                actions[new_success, 0:3] = 0.0
                success_latched |= new_success
                print(
                    f"[SUCCESS] switch pressed in envs={successful_envs} at step={step}; "
                    "stopping the base command and retaining the scene."
                )
                if dashboard is not None:
                    dashboard.event(
                        "task_success",
                        step=step,
                        sim_time_s=round(step * env.unwrapped.step_dt, 4),
                        environments=successful_envs,
                    )
        if dashboard is not None and step % args_cli.llm_log_interval == 0:
            live_switch = observations["llm"]["switch"][0].detach().cpu().tolist()
            telemetry = {
                "step": step,
                "sim_time_s": round(step * env.unwrapped.step_dt, 4),
                "policy_status": llm_status,
                "request_count": llm_request_count,
                "requested_action": _rounded(actions[0]),
                "applied_action": _rounded(action_term.processed_actions[0]),
                "physical_base_command": _rounded(command_term.vel_command_b[0, :3]),
                "measured_root_velocity": {
                    "linear_root_mps": _rounded(robot.data.root_lin_vel_b[0]),
                    "angular_root_rps": _rounded(robot.data.root_ang_vel_b[0]),
                },
                "switch": {
                    "normalized_depth": round(float(live_switch[0]), 6),
                    "pressed": bool(live_switch[1] > 0.5),
                },
                "wbc_action_rms": round(
                    torch.sqrt(torch.mean(torch.square(action_term.low_level_actions[0]))).item(),
                    6,
                ),
            }
            if hasattr(env.unwrapped, "_arm_grasp_goal_pos"):
                telemetry["grasp_goal_root_m"] = _rounded(env.unwrapped._arm_grasp_goal_pos[0])
            dashboard.update(status=llm_status, request_count=llm_request_count)
            dashboard.sample(observation=symbolic_state, telemetry=telemetry)
        if step % max(args_cli.print_interval, 1) == 0:
            switch_state = observations["llm"]["switch"][0].detach().cpu().tolist()
            wbc_rms = torch.sqrt(torch.mean(torch.square(action_term.low_level_actions[0]))).item()
            base_height = robot.data.root_pos_w[0, 2].item()
            root_velocity = robot.data.root_lin_vel_b[0, :2].detach().cpu().tolist()
            root_yaw_rate = robot.data.root_ang_vel_b[0, 2].item()
            travel = torch.linalg.vector_norm(robot.data.root_pos_w[0, :2] - root_origin_xy[0]).item()
            rgb_delta = torch.mean(torch.abs(current_rgb[0].float() - initial_rgb[0].float())).item()
            physical_command = command_term.vel_command_b[0, :3].detach().cpu().tolist()
            leg_speed = torch.sqrt(torch.mean(torch.square(robot.data.joint_vel[0, action_term._leg_joint_ids]))).item()
            print(
                f"[INFO] step={step} WBC_rms={wbc_rms:.3f} base_z={base_height:.3f} "
                f"actor_cmd=[{physical_command[0]:+.3f},{physical_command[1]:+.3f},"
                f"{physical_command[2]:+.3f}] "
                f"body_vel=[{root_velocity[0]:+.3f},{root_velocity[1]:+.3f},{root_yaw_rate:+.3f}] "
                f"travel={travel:.3f}m rgb_delta={rgb_delta:.2f} rgb_updates={rgb_update_counts[0].item()} "
                f"leg_qd_rms={leg_speed:.3f} "
                f"switch=[depth={switch_state[0]:.3f}, pressed={switch_state[1]:.0f}] "
                f"text_policy={llm_status!r}"
            )

        if rgb_view_enabled:
            try:
                key = _show_rgb_frame(
                    observations,
                    actions,
                    action_term,
                    command_term,
                    robot,
                    root_origin_xy,
                    initial_rgb,
                    llm_status,
                    env_index=args_cli.rgb_env_index,
                    window_scale=args_cli.rgb_window_scale,
                )
                if key == ord("p") and llm_enabled:
                    if request_budget_available():
                        llm_replan_requested = True
                        actions[0, 0:3] = 0.0
                        llm_status = "replan requested"
                    else:
                        print(f"[LLM] P ignored: --llm-max-requests={args_cli.llm_max_requests} exhausted.")
                if not _apply_rgb_base_key(actions, key):
                    print("[INFO] Closing play loop because the RGB window received ESC/q.")
                    break
            except cv2.error as exc:
                print(f"[WARN] OpenCV HighGUI is unavailable; disabling the live RGB window: {exc}")
                rgb_view_enabled = False

        dones = terminated | truncated
        if torch.any(dones):
            termination_manager = env.unwrapped.termination_manager
            for env_index in torch.nonzero(dones, as_tuple=False).flatten().detach().cpu().tolist():
                reasons = [
                    name for name, values in termination_manager.get_active_iterable_terms(env_index) if values[0] > 0.0
                ]
                print(f"[WARN] env={env_index} reset at step={step}; termination={reasons}")
                if dashboard is not None:
                    dashboard.event(
                        "scene_reset",
                        step=step,
                        environment=env_index,
                        termination=reasons,
                    )
            # Manager-based environments reset completed scenes internally.
            # Refresh the workspace-ready arm command for only those environments while
            # preserving the user's persistent base command across the automatic reset.
            persistent_base_commands = actions[dones, 0:3].clone()
            actions[dones] = action_term.processed_actions[dones]
            actions[dones, 0:3] = persistent_base_commands
            root_origin_xy[dones] = robot.data.root_pos_w[dones, :2]
            if llm_enabled:
                actions[dones, 0:3] = 0.0
                llm_executor = None
                llm_status = "scene reset; press P to replan"

        if args_cli.real_time:
            sleep_time = env.unwrapped.step_dt - (time.time() - step_start)
            if sleep_time > 0.0:
                time.sleep(sleep_time)

    if llm_pool is not None:
        llm_pool.shutdown(wait=False, cancel_futures=True)
    if dashboard is not None:
        dashboard.close(llm_status)
    env.close()
    if cv2 is not None:
        with contextlib.suppress(cv2.error):
            cv2.destroyAllWindows()


if __name__ == "__main__":
    try:
        main()
    except Exception:
        # Isaac Sim can suppress an unhandled traceback while its extensions
        # are shutting down, so report runner failures before closing the app.
        traceback.print_exc()
        raise
    finally:
        simulation_app.close()
