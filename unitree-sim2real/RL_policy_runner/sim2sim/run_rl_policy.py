import time
import sys
from pathlib import Path
import os
import numpy as np
import torch
import yaml
import threading
import select
import signal
import argparse
import subprocess
import glob
from typing import List, Optional

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py.core.channel import ChannelPublisher, ChannelSubscriber
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.utils.crc import CRC
from shared_depth import DepthFrameReader, depth_preview_uint8, preprocess_depth_image


def quat_to_rot(quat):
    w, x, y, z = quat
    
    x2, y2, z2 = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    
    R = np.array([
        [1 - 2 * (y2 + z2), 2 * (xy - wz), 2 * (xz + wy)],
        [2 * (xy + wz), 1 - 2 * (x2 + z2), 2 * (yz - wx)],
        [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (x2 + y2)]
    ], dtype=np.float32)
    
    return R


def yaw_from_quat_wxyz(quat):
    w, x, y, z = quat
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rot_z(yaw):
    c = np.cos(yaw)
    s = np.sin(yaw)
    return np.array(
        [
            [c, -s, 0.0],
            [s, c, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )


def quat_rotate_inverse(q, v):
    """Rotate vector v by inverse of quaternion q
    Args:
        q: quaternion [w, x, y, z]
        v: vector [x, y, z]
    Returns:
        rotated vector [x, y, z]
    """
    q_w = q[0]
    q_x = q[1]
    q_y = q[2]
    q_z = q[3]
    
    v_x = v[0]
    v_y = v[1]
    v_z = v[2]
    
    # a = v * (2.0 * q_w^2 - 1.0)
    a_x = v_x * (2.0 * q_w * q_w - 1.0)
    a_y = v_y * (2.0 * q_w * q_w - 1.0)
    a_z = v_z * (2.0 * q_w * q_w - 1.0)
    
    # cross = q_vec × v
    cross_x = q_y * v_z - q_z * v_y
    cross_y = q_z * v_x - q_x * v_z
    cross_z = q_x * v_y - q_y * v_x
    
    # b = cross * q_w * 2.0
    b_x = cross_x * q_w * 2.0
    b_y = cross_y * q_w * 2.0
    b_z = cross_z * q_w * 2.0
    
    # dot = q_vec · v
    dot = q_x * v_x + q_y * v_y + q_z * v_z
    
    # c = q_vec * dot * 2.0
    c_x = q_x * dot * 2.0
    c_y = q_y * dot * 2.0
    c_z = q_z * dot * 2.0
    
    # result = a - b + c
    return np.array([
        a_x - b_x + c_x,
        a_y - b_y + c_y,
        a_z - b_z + c_z
    ], dtype=np.float32)

LOGS_ROOT = REPO_ROOT.parent / "logs" / "rsl_rl"
DEFAULT_CONFIG_ARG = os.environ.get("RL_POLICY_CONFIG", "RL_policy_runner/configs/go2.yaml")
COLOR_ENABLED = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None
APEX_VIS_MAGIC = 2048.042
APEX_VIS_FLOAT_COUNT = 32
APEX_VIS_DEFAULT_BUFFER = "/tmp/Gurukul_go2_apex_motion_vis.mmap"
GO2_APEX_BASE_OBS_DIM = 46
GO2_APEX_TRACKER_BASE_OBS_DIM = 46
GO2_APEX_TRACKER_REFERENCE_FRAME_DIM = 22
GO2_APEX_TRACKER_JOINT_NAMES = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)
GO2_APEX_TRACKER_DEFAULT_REFERENCE_OFFSETS = (0, 1, 2, 5, 10)
GO2_APEX_TRACKER_REFERENCE_OFFSETS = GO2_APEX_TRACKER_DEFAULT_REFERENCE_OFFSETS
GO2_APEX_TRACKER_HISTORY_LENGTH = 1
GO2_APEX_TRACKER_SINGLE_OBS_DIM = 0
GO2_APEX_TRACKER_SKILL_DIM = 0
actor_joint_velocity_indices = np.arange(12, dtype=np.int32)
ISAAC_HISTORY_LENGTH = 1
ISAAC_HISTORY_SINGLE_OBS_DIM = 45
arm_command_cfg = {}
arm_command_enabled = False
arm_command_indices = np.zeros(0, dtype=np.int32)
arm_gripper_index = None
arm_lower_limits = np.zeros(0, dtype=np.float32)
arm_upper_limits = np.zeros(0, dtype=np.float32)
arm_max_joint_delta = np.float32(0.05)
arm_command_period_s = np.float32(0.04)
arm_command_velocity = True
arm_command_model = "linear_rate_limited"
arm_command_quantization = np.float32(0.0)
arm_command_natural_frequency_hz = np.float32(6.0)
arm_command_damping_ratio = np.float32(1.0)
arm_command_latency_s = np.float32(0.0)
arm_command_source = "command"
arm_reference_indices = np.zeros(0, dtype=np.int32)
reference_residual_action_indices = np.zeros(0, dtype=np.int32)
position_target_lower_limits = np.zeros(0, dtype=np.float32)
position_target_upper_limits = np.zeros(0, dtype=np.float32)


def _write_go2_apex_base_observation(
    output: torch.Tensor,
    *,
    base_ang_vel: np.ndarray,
    projected_gravity: np.ndarray,
    motion_command: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    previous_action: np.ndarray,
    ang_vel_scale: float,
    gravity_scale: float,
    cmd_scale: np.ndarray,
    dof_pos_scale: float,
    dof_vel_scale: float,
) -> int:
    """Write the current 46-D Go2 APEX actor contract in Isaac Lab term order."""
    named_fields = (
        ("base_ang_vel", base_ang_vel, 3, ang_vel_scale),
        ("projected_gravity", projected_gravity, 3, gravity_scale),
        ("motion_command", motion_command, 4, cmd_scale),
        ("joint_pos", joint_pos, 12, dof_pos_scale),
        ("joint_vel", joint_vel, 12, dof_vel_scale),
        ("previous_action", previous_action, 12, 1.0),
    )
    if output.numel() != GO2_APEX_BASE_OBS_DIM:
        raise ValueError(
            f"Go2 APEX base observation output must have {GO2_APEX_BASE_OBS_DIM} values; "
            f"got {output.numel()}."
        )

    idx = 0
    for name, raw_values, expected_size, scale in named_fields:
        values = np.asarray(raw_values, dtype=np.float32).reshape(-1)
        if values.size != expected_size:
            raise ValueError(f"{name} must have {expected_size} values; got {values.size}.")
        scale_values = np.asarray(scale, dtype=np.float32).reshape(-1)
        if scale_values.size not in (1, expected_size):
            raise ValueError(
                f"{name} scale must be scalar or have {expected_size} values; got {scale_values.size}."
            )
        end = idx + expected_size
        output[idx:end] = torch.as_tensor(
            values * scale_values,
            dtype=output.dtype,
            device=output.device,
        )
        idx = end
    return idx


def _validate_policy_observation_width(input_shape, actual_width: int) -> None:
    """Fail before inference when an exported policy has a known, incompatible input width."""
    if input_shape is None or len(input_shape) == 0:
        return
    exported_width = input_shape[-1]
    if isinstance(exported_width, (int, np.integer)) and exported_width > 0:
        if int(exported_width) != int(actual_width):
            raise ValueError(
                "Policy observation width mismatch: "
                f"export expects {int(exported_width)}, but the selected config builds {int(actual_width)}."
            )


def _ansi(text: object, code: str) -> str:
    if not COLOR_ENABLED:
        return str(text)
    return f"\033[{code}m{text}\033[0m"


def _status(label: str, value: object, code: str = "1;36") -> str:
    return f"{_ansi(f'[{label}]', code)} {value}"


def _print_runtime_summary(task_key: Optional[str], config_path: Path, selected_policy_path: str):
    title = _ansi("Gurukul Sim2Sim", "1;36")
    label = lambda value: _ansi(f"{value:<10}", "1;37")
    print(f"\n{title}")
    print("-" * 56)
    if task_key is not None:
        spec = TASK_SPECS[task_key]
        print(f"{label('Task')}: {_ansi(task_key, '1;33')}")
        print(f"{label('Gurukul')}: {spec['task_id']}")
        print(f"{label('Experiment')}: {spec['experiment']}")
    else:
        print(f"{label('Task')}: custom config")
    print(f"{label('Config')}: {config_path}")
    print(f"{label('Policy')}: {_ansi(selected_policy_path, '1;32')}")
    print("-" * 56)


def _format_float(value: object) -> str:
    value = float(value)
    if abs(value) >= 1.0e8:
        return "unbounded" if value > 0.0 else "-unbounded"
    return f"{value:.4g}"


def _format_array(values, precision: int = 3) -> str:
    arr = np.asarray(values).reshape(-1)
    return "[" + ", ".join(f"{float(value):.{precision}g}" for value in arr) + "]"


def _format_gain_summary(values) -> str:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return "[]"
    if np.allclose(arr, arr[0]):
        return _format_float(arr[0])
    return f"range[{_format_float(np.min(arr))}, {_format_float(np.max(arr))}]"


def _format_enabled(value: bool) -> str:
    return "enabled" if value else "disabled"


def _print_table(title: str, headers: List[str], rows: List[List[object]]):
    if not rows:
        return
    rendered_rows = [[str(item) for item in row] for row in rows]
    widths = [
        max(len(headers[col]), *(len(row[col]) for row in rendered_rows))
        for col in range(len(headers))
    ]
    title_text = _ansi(title, "1;36")
    print(f"\n{title_text}")
    print("-" * min(100, max(len(title), sum(widths) + 3 * (len(widths) - 1))))
    print(" | ".join(headers[col].ljust(widths[col]) for col in range(len(headers))))
    print("-+-".join("-" * width for width in widths))
    for row in rendered_rows:
        print(" | ".join(row[col].ljust(widths[col]) for col in range(len(headers))))


def _observation_rows() -> List[List[object]]:
    rows: List[List[object]] = []
    idx = 0
    active_num_obs = depth_base_obs_dim if depth_enabled else num_obs

    def add(name: str, size: int, scale: object = "1", source: str = ""):
        nonlocal idx
        rows.append([idx, idx + size, size, name, scale, source])
        idx += size

    if observation_layout == "go2_apex_base":
        add("base_ang_vel", 3, _format_float(ang_vel_scale), "imu gyro")
        add("projected_gravity", 3, _format_float(gravity_scale), "imu quaternion")
        add("command_vel_x", 1, _format_float(cmd_scale[0]), "keyboard/APEX")
        add("command_vel_y", 1, _format_float(cmd_scale[1]), "keyboard/APEX")
        add("command_vel_z", 1, _format_float(cmd_scale[2]), "APEX motion")
        add("command_yaw", 1, _format_float(cmd_scale[3]), "keyboard/APEX")
        add("joint_pos - default", 12, _format_float(dof_pos_scale), "low_state motor.q")
        add("joint_vel", 12, _format_float(dof_vel_scale), "low_state motor.dq")
        add("previous_action", 12, "1", "policy output")
        return rows

    if observation_layout == "go2_apex_tracker_history":
        history = GO2_APEX_TRACKER_HISTORY_LENGTH
        reference_dim = (
            GO2_APEX_TRACKER_SINGLE_OBS_DIM
            - GO2_APEX_TRACKER_BASE_OBS_DIM
            - GO2_APEX_TRACKER_SKILL_DIM
        )
        add("base_ang_vel history", 3 * history, _format_float(ang_vel_scale), "imu gyro")
        add("projected_gravity history", 3 * history, _format_float(gravity_scale), "imu quaternion")
        add("command_vel_x history", history, _format_float(cmd_scale[0]), "keyboard/APEX")
        add("command_vel_y history", history, _format_float(cmd_scale[1]), "keyboard/APEX")
        add("command_vel_z history", history, _format_float(cmd_scale[2]), "APEX motion")
        add("command_yaw history", history, _format_float(cmd_scale[3]), "keyboard/APEX")
        add("joint_pos - default history", num_joint_obs * history, _format_float(dof_pos_scale), "low_state motor.q")
        add(
            "actor joint_vel history",
            actor_joint_velocity_indices.size * history,
            _format_float(dof_vel_scale),
            "selected low_state motor.dq",
        )
        add("previous_action history", num_actions * history, "1", "policy output")
        if GO2_APEX_TRACKER_SKILL_DIM:
            add("skill history", GO2_APEX_TRACKER_SKILL_DIM * history, "1", "APEX motion")
        add("reference_motion history", reference_dim * history, "1", "APEX motion")
        return rows
    if observation_layout in {"isaac_history", "isaac_term_history"}:
        history = ISAAC_HISTORY_LENGTH
        add("base_ang_vel history", 3 * history, _format_float(ang_vel_scale), "imu gyro")
        add("projected_gravity history", 3 * history, _format_float(gravity_scale), "imu quaternion")
        command_size = 3 if observation_layout == "isaac_term_history" else 3 * history
        command_label = "command" if observation_layout == "isaac_term_history" else "command history"
        add(command_label, command_size, f"cmd_scale={_format_array(cmd_scale)}", "keyboard")
        add("joint_pos - default history", num_joint_obs * history, _format_float(dof_pos_scale), "low_state motor.q")
        add("joint_vel history", num_joint_obs * history, _format_float(dof_vel_scale), "low_state motor.dq")
        add("previous_action history", num_actions * history, "1", "policy output")
        return rows

    if active_num_obs == 39:
        add("projected_gravity", 3, _format_float(gravity_scale), "imu quaternion")
        add("joint_pos - default", 12, _format_float(dof_pos_scale), "low_state motor.q")
        add("joint_vel", 12, _format_float(dof_vel_scale), "low_state motor.dq")
        add("previous_action", 12, "1", "policy output")
    elif active_num_obs == 42:
        add("projected_gravity", 3, _format_float(gravity_scale), "imu quaternion")
        add("command", 3, f"cmd_scale={_format_array(cmd_scale)}", "keyboard/APEX")
        add("joint_pos - default", 12, _format_float(dof_pos_scale), "low_state motor.q")
        add("joint_vel", 12, _format_float(dof_vel_scale), "low_state motor.dq")
        add("previous_action", 12, "1", "policy output")
    elif active_num_obs == 48:
        add("base_lin_vel", 3, _format_float(lin_vel_scale), "sport mode state")
        add("base_ang_vel", 3, _format_float(ang_vel_scale), "imu gyro")
        add("projected_gravity", 3, _format_float(gravity_scale), "imu quaternion")
        add("command", 3, f"cmd_scale={_format_array(cmd_scale)}", "keyboard/APEX")
        add("joint_pos - default", 12, _format_float(dof_pos_scale), "low_state motor.q")
        add("joint_vel", 12, _format_float(dof_vel_scale), "low_state motor.dq")
        add("previous_action", 12, "1", "policy output")
    elif active_num_obs == 43:
        add("projected_gravity", 3, _format_float(gravity_scale), "imu quaternion")
        add("command", 3, f"cmd_scale={_format_array(cmd_scale)}", "keyboard/APEX")
        add("joint_pos - default", 12, _format_float(dof_pos_scale), "low_state motor.q")
        add("joint_vel", 12, _format_float(dof_vel_scale), "low_state motor.dq")
        add("previous_action", 12, "1", "policy output")
        add("phase", 1, "1", "runner")
    elif active_num_obs == 11 + 2 * num_joint_obs + num_actions:
        add("base_ang_vel", 3, _format_float(ang_vel_scale), "imu gyro")
        add("projected_gravity", 3, _format_float(gravity_scale), "imu quaternion")
        add("velocity_commands", 5, f"cmd_scale={_format_array(cmd_scale)}", "keyboard")
        add("joint_pos - default", num_joint_obs, _format_float(dof_pos_scale), "low_state motor.q")
        add("joint_vel", num_joint_obs, _format_float(dof_vel_scale), "low_state motor.dq")
        add("previous_action", num_actions, "1", "policy output")
    elif (
        active_num_obs in (
            45,
            GO2_APEX_TRACKER_BASE_OBS_DIM
            + GO2_APEX_TRACKER_SKILL_DIM
            + GO2_APEX_TRACKER_REFERENCE_FRAME_DIM * len(GO2_APEX_TRACKER_REFERENCE_OFFSETS),
        )
        or active_num_obs == 9 + 3 * num_actions
    ):
        add("base_ang_vel", 3, _format_float(ang_vel_scale), "imu gyro")
        add("projected_gravity", 3, _format_float(gravity_scale), "imu quaternion")
        if observation_layout == "mjlab_go2_velocity":
            add("joint_pos - default", num_actions, _format_float(dof_pos_scale), "low_state motor.q")
            add("joint_vel", num_actions, _format_float(dof_vel_scale), "low_state motor.dq")
            add("previous_action", num_actions, "1", "policy output")
            add("command", 3, f"cmd_scale={_format_array(cmd_scale)}", "keyboard/APEX")
        else:
            command_dim = 4 if observation_layout in {"go2_apex_tracker", "go2_apex_tracker_history"} else 3
            add("command", command_dim, f"cmd_scale={_format_array(cmd_scale)}", "keyboard/APEX")
            joint_pos_dim = (
                num_joint_obs
                if observation_layout in {"go2_apex_tracker", "go2_apex_tracker_history"}
                else num_actions
            )
            joint_vel_dim = (
                actor_joint_velocity_indices.size
                if observation_layout in {"go2_apex_tracker", "go2_apex_tracker_history"}
                else num_actions
            )
            add("joint_pos - default", joint_pos_dim, _format_float(dof_pos_scale), "low_state motor.q")
            add("actor joint_vel", joint_vel_dim, _format_float(dof_vel_scale), "selected low_state motor.dq")
            add("previous_action", num_actions, "1", "policy output")

        if active_num_obs > GO2_APEX_TRACKER_BASE_OBS_DIM and observation_layout in {
            "go2_apex_tracker",
            "go2_apex_tracker_history",
        }:
            if GO2_APEX_TRACKER_SKILL_DIM:
                add("skill", GO2_APEX_TRACKER_SKILL_DIM, "1", "APEX motion")
            for offset in GO2_APEX_TRACKER_REFERENCE_OFFSETS:
                add(f"ref[{offset}] joint_pos", len(GO2_APEX_TRACKER_JOINT_NAMES), "1", "APEX motion")
                add(f"ref[{offset}] base_lin_vel", 3, "1", "APEX motion")
                add(f"ref[{offset}] base_ang_vel", 3, "1", "APEX motion")
                add(f"ref[{offset}] base_quat_wxyz", 4, "1", "APEX motion")
    elif active_num_obs == 46:
        add("base_ang_vel", 3, _format_float(ang_vel_scale), "imu gyro")
        add("projected_gravity", 3, _format_float(gravity_scale), "imu quaternion")
        add("command", 3, f"cmd_scale={_format_array(cmd_scale)}", "keyboard")
        add("joint_pos - default", 12, _format_float(dof_pos_scale), "low_state motor.q")
        add("joint_vel", 12, _format_float(dof_vel_scale), "low_state motor.dq")
        add("previous_action", 12, "1", "policy output")
        add("skill_number", 1, "1", "keyboard")
    else:
        add(f"unsupported base layout ({active_num_obs})", active_num_obs, "", "")

    if depth_enabled:
        add(
            f"depth_image {depth_resize_shape[0]}x{depth_resize_shape[1]}",
            depth_image_dim,
            "normalized" if depth_normalize else "meters",
            depth_buffer_path,
        )
    return rows


def _print_policy_interface_summary():
    control_hz = 1.0 / (simulation_dt * control_decimation)
    _print_table(
        "Policy Interface",
        ["field", "value"],
        [
            ["observation_layout", observation_layout],
            ["num_obs", num_obs],
            ["num_actions", num_actions],
            ["control_type", control_type],
            ["control_dt", f"{simulation_dt * control_decimation:.4f}s ({control_hz:.1f} Hz)"],
            ["simulation_dt", f"{simulation_dt:.4f}s"],
            [
                "standup_duration",
                (
                    f"{standup_duration_s:.2f}s deployment tanh ramp"
                    if standup_duration_s > 0.0
                    else "disabled; direct policy handover"
                ),
            ],
            [
                "standup_start_kp",
                _format_gain_summary(standup_start_kps) if standup_duration_s > 0.0 else "not used",
            ],
            ["standup_kp", _format_gain_summary(standup_kps)],
            ["standup_kd", _format_gain_summary(standup_kds)],
            ["torque_limit", _format_gain_summary(torque_limit)],
            ["action_clip", "none" if action_clip is None else _format_float(action_clip)],
            ["action_target", "q_target = default_q + action * action_scale"],
            ["command_init", _format_array(cmd_init)],
            [
                "command_limits",
                (
                    f"x[{_format_float(cmd_lin_x_min)}, {_format_float(cmd_lin_x_max)}], "
                    f"y[{_format_float(cmd_lin_y_min)}, {_format_float(cmd_lin_y_max)}], "
                    f"yaw[{_format_float(cmd_ang_z_min)}, {_format_float(cmd_ang_z_max)}]"
                ),
            ],
            [
                "command_smoothing",
                (
                    f"step={_format_array(cmd_key_step)}, accel={_format_array(cmd_accel)}, "
                    f"decel={_format_array(cmd_decel)}, timeout={float(cmd_key_timeout):.2f}s"
                ),
            ],
            [
                "arm_command",
                (
                    f"{_format_enabled(arm_command_enabled)}"
                    if not arm_command_enabled
                    else (
                        (
                            f"source={arm_command_source}, model={arm_command_model}, "
                            f"indices={arm_command_indices.tolist()}, "
                            f"max_delta={_format_float(arm_max_joint_delta)} rad/"
                            f"{_format_float(arm_command_period_s)}s, "
                            f"dq_cmd={_format_enabled(arm_command_velocity)}"
                        )
                        if arm_command_model == "linear_rate_limited"
                        else (
                            (
                                f"source={arm_command_source}, model={arm_command_model}, "
                                f"indices={arm_command_indices.tolist()}, "
                                f"period={_format_float(arm_command_period_s)}s"
                            )
                            if arm_command_model == "sample_and_hold"
                            else (
                                f"source={arm_command_source}, model={arm_command_model}, "
                                f"indices={arm_command_indices.tolist()}, "
                                f"period={_format_float(arm_command_period_s)}s, "
                                f"quant={_format_array(arm_command_quantization)} rad/m, "
                                f"fn={_format_float(arm_command_natural_frequency_hz)}Hz, "
                                f"zeta={_format_float(arm_command_damping_ratio)}, "
                                f"latency={_format_float(arm_command_latency_s)}s"
                            )
                        )
                    )
                ),
            ],
        ],
    )

    _print_table(
        "Observation Order",
        ["start", "end", "size", "term", "scale", "source"],
        _observation_rows(),
    )

    action_rows = []
    controlled_lookup = {
        int(motor_idx): int(mapping[action_idx]) for action_idx, motor_idx in enumerate(controlled_motor_indices)
    }
    for unitree_idx in range(num_motors):
        joint_name = motor_names[unitree_idx]
        policy_idx = controlled_lookup.get(unitree_idx, "hold")
        action_rows.append(
            [
                unitree_idx,
                joint_name,
                policy_idx,
                _format_float(default_angles[unitree_idx]),
                "0" if policy_idx == "hold" else _format_float(action_scales[policy_idx]),
                _format_float(kps[unitree_idx]),
                _format_float(kds[unitree_idx]),
            ]
        )
    _print_table(
        "Action / Joint Mapping",
        ["motor", "joint", "policy_action", "default_q", "scale", "kp", "kd"],
        action_rows,
    )


TASK_SPECS = {
    "go2_velocity_flat_v0": {
        "task_id": "Gurukul-Isaac-Velocity-Flat-Unitree-Go2-v0",
        "config_path": "RL_policy_runner/configs/Gurukul/go2_velocity_flat_v0.yaml",
        "experiment": "unitree_go2_flat",
        "allow_run_selection": True,
        "notes": "45-D flat velocity actor.",
    },
    "go2_velocity_flat_v0_mjlab_action_scale": {
        "task_id": "Gurukul-Isaac-Velocity-Flat-Unitree-Go2-v0-MJLabActionScale",
        "config_path": "RL_policy_runner/configs/Gurukul/go2_velocity_flat_v0_mjlab_action_scale.yaml",
        "experiment": "unitree_go2_flat",
        "allow_run_selection": True,
        "notes": "45-D flat v0 velocity actor with mjlab parity action scale only.",
    },
    "go2_velocity_flat_v0_with_d1_motion": {
        "task_id": "Gurukul-Sim2Sim-Velocity-Flat-Unitree-Go2-With-D1-Motion-v0",
        "config_path": (
            "RL_policy_runner/configs/Gurukul/"
            "go2_velocity_flat_v0_with_d1_motion.yaml"
        ),
        "experiment": "unitree_go2_flat",
        "allow_run_selection": True,
        "notes": "Plain 45-D Go2 actor on the Go2+D1 model with independently replayed D1 motion.",
    },
    "go2_velocity_rough_v0": {
        "task_id": "Gurukul-Isaac-Velocity-Rough-Unitree-Go2-v0",
        "config_path": "RL_policy_runner/configs/Gurukul/go2_velocity_rough_v0.yaml",
        "experiment": "unitree_go2_rough",
        "allow_run_selection": True,
        "notes": "45-D rough velocity actor on terrain scene.",
    },
    "go2_velocity_rough_depth_distill_v0": {
        "task_id": "Gurukul-Isaac-Velocity-Rough-Unitree-Go2-Depth-Distill-v0",
        "config_path": "RL_policy_runner/configs/Gurukul/go2_velocity_rough_depth_distill_v0.yaml",
        "experiment": "unitree_go2_rough_depth_distill",
        "allow_run_selection": True,
        "notes": "45-D rough proprio obs plus a 58x87 depth image from the MuJoCo Go2 camera, using the stairs scene for now.",
    },
    "go2_velocity_flat_v1": {
        "task_id": "Gurukul-Isaac-Velocity-Flat-Unitree-Go2-v1",
        "config_path": "RL_policy_runner/configs/Gurukul/go2_velocity_flat_v1.yaml",
        "experiment": "unitree_go2_flat",
        "allow_run_selection": False,
        "notes": "MJLab-aligned PD/action scales. No dedicated v1 export is configured in this workspace.",
    },
    "b2_velocity_flat_v0": {
        "task_id": "Gurukul-Isaac-Velocity-Flat-Unitree-B2-v0",
        "config_path": "RL_policy_runner/configs/Gurukul/b2_velocity_flat_v0.yaml",
        "experiment": "unitree_b2_flat",
        "allow_run_selection": True,
        "notes": "45-D flat velocity actor on the Unitree B2 MuJoCo flat scene.",
    },
    "b2_z1_arm_velocity_flat_v0": {
        "task_id": "Gurukul-Isaac-Velocity-Flat-Unitree-B2-Z1-Arm-v0",
        "config_path": "RL_policy_runner/configs/Gurukul/b2_z1_arm_velocity_flat_v0.yaml",
        "experiment": "unitree_b2_z1_arm_flat",
        "allow_run_selection": True,
        "notes": "66-D flat velocity actor on the Unitree B2 + Z1 MuJoCo flat scene.",
    },
    "b2_z1_arm_moving_flat_distill_v0": {
        "task_id": "Gurukul-Isaac-Velocity-Flat-Unitree-B2-Z1-Arm-ArmMoving-Teacher-v0",
        "config_path": "RL_policy_runner/configs/Gurukul/b2_z1_arm_moving_flat_distill_v0.yaml",
        "experiment": "unitree_b2_z1_arm_moving_flat_distill",
        "allow_run_selection": True,
        "notes": "61-D distilled ArmMoving actor: 12 B2 leg actions, 19 B2+Z1 joint observations, 5-D velocity/posture command.",
    },
    "go2_apex_flat_v0": {
        "task_id": "Gurukul-Isaac-Go2-APEX-Flat-v0",
        "config_path": "RL_policy_runner/configs/Gurukul/go2_apex_flat_v0.yaml",
        "experiment": "unitree_go2_apex_flat",
        "allow_run_selection": True,
        "notes": (
            "Current 46-D APEX flat actor with a four-component motion command. "
            "The separate legacy-45 config is only for older exports."
        ),
    },
    "go2_apex_flat_tracker_v0": {
        "task_id": "Gurukul-Isaac-Go2-APEX-Flat-Tracker-v0",
        "config_path": "RL_policy_runner/configs/Gurukul/go2_apex_flat_tracker_v0.yaml",
        "experiment": "unitree_go2_apex_flat_tracker",
        "allow_run_selection": True,
        "notes": "APEX tracker actor with reference-motion observations and MuJoCo motion overlay.",
    },
    "go2_d1_arm_apex_flat_tracker_v0": {
        "task_id": "Gurukul-Isaac-Go2-D1-Arm-APEX-Flat-Tracker-v0",
        "config_path": "RL_policy_runner/configs/Gurukul/go2_d1_arm_apex_flat_tracker_v0.yaml",
        "experiment": "unitree_go2_d1_arm_apex_flat_tracker",
        "allow_run_selection": True,
        "notes": "Direct-PPO Go2+D1 APEX tracker with deployable observations and 19 position actions.",
    },
    "go2_d1_arm_apex_distillation_student_v0": {
        "task_id": "Gurukul-Isaac-Go2-D1-Arm-APEX-Distillation-Student-v0",
        "config_path": "RL_policy_runner/configs/Gurukul/go2_d1_arm_apex_flat_tracker_v0.yaml",
        "experiment": "unitree_go2_d1_arm_apex_flat_tracker_distill",
        "allow_run_selection": True,
        "notes": "Supervised Go2+D1 student distilled from the privileged original-DecAP teacher.",
    },
    "go2_apex_flat_tracker_one_step_future_v0": {
        "task_id": "Gurukul-Isaac-Go2-APEX-Flat-Tracker-One-Step-Future-v0",
        "config_path": "RL_policy_runner/configs/Gurukul/go2_apex_flat_tracker_one_step_future_v0.yaml",
        "experiment": "unitree_go2_apex_flat_tracker_one_step_future",
        "allow_run_selection": True,
        "notes": "APEX tracker actor with current plus one future reference frame and MuJoCo motion overlay.",
    },
    "go2_apex_flat_tracker_one_step_future_history_v0": {
        "task_id": "Gurukul-Isaac-Go2-APEX-Flat-Tracker-One-Step-Future-History-v0",
        "config_path": (
            "RL_policy_runner/configs/Gurukul/"
            "go2_apex_flat_tracker_one_step_future_history_v0.yaml"
        ),
        "experiment": "unitree_go2_apex_flat_tracker_one_step_future_history_distill",
        "allow_run_selection": True,
        "notes": "APEX one-step tracker with a 5-frame deployable observation/reference history.",
    },
    "b2_z1_arm_apex_flat_tracker_one_step_future_v0": {
        "task_id": "Gurukul-Isaac-B2-Z1-Arm-APEX-Flat-Tracker-One-Step-Future-v0",
        "config_path": (
            "RL_policy_runner/configs/Gurukul/"
            "b2_z1_arm_apex_flat_tracker_one_step_future_v0.yaml"
        ),
        "experiment": "unitree_b2_z1_arm_apex_flat_tracker_one_step_future_distill",
        "allow_run_selection": True,
        "notes": "B2+Z1 APEX one-step tracker with 18 action joints and current plus one future reference frame.",
    },
    "b2_z1_arm_apex_flat_tracker_one_step_future_history_v0": {
        "task_id": "Gurukul-Isaac-B2-Z1-Arm-APEX-Flat-Tracker-One-Step-Future-History-v0",
        "config_path": (
            "RL_policy_runner/configs/Gurukul/"
            "b2_z1_arm_apex_flat_tracker_one_step_future_history_v0.yaml"
        ),
        "experiment": "unitree_b2_z1_arm_apex_flat_tracker_one_step_future_history_distill",
        "allow_run_selection": True,
        "notes": "B2+Z1 APEX one-step tracker with a 5-frame deployable observation/reference history.",
    },
}

TASK_ALIASES = {
    "flat_v0": "go2_velocity_flat_v0",
    "flat_v0_mjlab_action_scale": "go2_velocity_flat_v0_mjlab_action_scale",
    "flat_v0_with_d1_motion": "go2_velocity_flat_v0_with_d1_motion",
    "rough_v0": "go2_velocity_rough_v0",
    "rough_depth_distill_v0": "go2_velocity_rough_depth_distill_v0",
    "flat_v1": "go2_velocity_flat_v1",
    "b2_flat_v0": "b2_velocity_flat_v0",
    "b2_z1_flat_v0": "b2_z1_arm_velocity_flat_v0",
    "b2_z1_arm_moving_distill_v0": "b2_z1_arm_moving_flat_distill_v0",
    "apex_flat_v0": "go2_apex_flat_v0",
    "apex_tracker_v0": "go2_apex_flat_tracker_v0",
    "go2_d1_apex_tracker_v0": "go2_d1_arm_apex_flat_tracker_v0",
    "go2_d1_apex_distillation_student_v0": "go2_d1_arm_apex_distillation_student_v0",
    "apex_tracker_one_step_future_v0": "go2_apex_flat_tracker_one_step_future_v0",
    "apex_tracker_one_step_future_history_v0": "go2_apex_flat_tracker_one_step_future_history_v0",
    "b2_z1_apex_tracker_one_step_future_v0": "b2_z1_arm_apex_flat_tracker_one_step_future_v0",
    "b2_z1_apex_tracker_one_step_future_history_v0": "b2_z1_arm_apex_flat_tracker_one_step_future_history_v0",
}

EXPERIMENT_NOTES = {
    "unitree_go2_airbot_arm_fixed_flat": "Go2 + Airbot arm family, not part of the kept plain-Go2 runner.",
    "unitree_go2_airbot_arm_fixed_rough": "Go2 + Airbot arm family, not part of the kept plain-Go2 runner.",
    "unitree_go2_airbot_arm_flat": "Go2 + Airbot arm family, not part of the kept plain-Go2 runner.",
    "unitree_go2_airbot_arm_rough": "Go2 + Airbot arm family, not part of the kept plain-Go2 runner.",
    "unitree_go2_apex_flat_multi_critic": "Multi-critic APEX variant is not wired into this trimmed runner.",
    "unitree_go2_flat_decap": "Decap family is not maintained in this MuJoCo runner.",
    "unitree_go2_flat_teacher": "Teacher-only family is not wired separately here.",
    "unitree_go2_rough_real_teacher": "REAL teacher family needs separate observation code.",
    "unitree_go2_rough_real_teacher_beam": "REAL beam teacher family needs separate observation code.",
    "unitree_go2_rough_real_teacher_pretrained": "REAL pretrained teacher family needs separate observation code.",
    "unitree_go2_rough_teacher": "Teacher/elevmap family uses observation terms not kept here.",
    "unitree_go2_rough_teacher_full": "Teacher full family uses observation terms not kept here.",
}

def _resolve_repo_path(path_str):
    path = Path(path_str)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def _vec3_from_cfg(values, default_values):
    """Return a float32 vec3, falling back to defaults if config shape is invalid."""
    arr = np.array(values if values is not None else default_values, dtype=np.float32).reshape(-1)
    if arr.size != 3:
        arr = np.array(default_values, dtype=np.float32)
    return arr


def _pair_int_from_cfg(values, default_values):
    """Return a pair of ints, falling back to defaults if config shape is invalid."""
    arr = np.array(values if values is not None else default_values, dtype=np.int32).reshape(-1)
    if arr.size != 2:
        arr = np.array(default_values, dtype=np.int32)
    return int(arr[0]), int(arr[1])


def _load_yaml_config(config_file: Path) -> dict:
    with open(config_file, "r", encoding="utf-8") as stream:
        return yaml.load(stream, Loader=yaml.FullLoader)


def _resolve_globbed_repo_paths(patterns) -> List[Path]:
    if patterns is None:
        return []
    if isinstance(patterns, (str, os.PathLike)):
        patterns = [patterns]

    paths: List[Path] = []
    for pattern in patterns:
        raw_path = Path(pattern).expanduser()
        search_pattern = str(raw_path if raw_path.is_absolute() else REPO_ROOT / raw_path)
        matches = [Path(path) for path in sorted(glob.glob(search_pattern, recursive=True))]
        for match in matches:
            if match.is_file():
                paths.append(match.resolve())
            elif match.is_dir():
                paths.extend(sorted(path.resolve() for path in match.rglob("*.npz")))
    return list(dict.fromkeys(paths))


def _resolve_task_key(task_name: str) -> str:
    candidate = task_name.strip()
    if candidate in TASK_SPECS:
        return candidate
    if candidate in TASK_ALIASES:
        return TASK_ALIASES[candidate]
    for task_key, spec in TASK_SPECS.items():
        if candidate == spec["task_id"]:
            return task_key
    raise ValueError(
        f"Unknown task '{task_name}'. Use --list-tasks to see the supported Gurukul tasks."
    )


def _discover_exported_runs(experiment_name: str) -> List[str]:
    experiment_dir = LOGS_ROOT / experiment_name
    if not experiment_dir.is_dir():
        return []
    runs = []
    for run_dir in sorted((path for path in experiment_dir.iterdir() if path.is_dir()), reverse=True):
        if (run_dir / "exported" / "policy.onnx").is_file():
            runs.append(run_dir.name)
    return runs


def _discover_unitree_families() -> List[str]:
    if not LOGS_ROOT.is_dir():
        return []
    return sorted(
        path.name for path in LOGS_ROOT.iterdir() if path.is_dir() and path.name.startswith("unitree_")
    )


def _resolve_task_config_path(task_key: str) -> Path:
    return _resolve_repo_path(TASK_SPECS[task_key]["config_path"])


def _default_policy_from_task(task_key: str) -> Optional[str]:
    cfg = _load_yaml_config(_resolve_task_config_path(task_key))
    policy_path = cfg.get("policy_path")
    return None if policy_path is None else str(policy_path)


def _resolve_run_policy(task_key: str, run_name: str) -> Path:
    spec = TASK_SPECS[task_key]
    if not spec.get("allow_run_selection", False):
        raise ValueError(
            f"Task '{task_key}' does not enable --run because no dedicated exported family is pinned for it yet. "
            "Use --policy-path with the exact ONNX you want."
        )

    experiment_name = spec["experiment"]
    resolved_run = run_name
    if run_name == "latest":
        exported_runs = _discover_exported_runs(experiment_name)
        if not exported_runs:
            raise FileNotFoundError(f"No exported policies found under {LOGS_ROOT / experiment_name}.")
        resolved_run = exported_runs[0]

    policy_file = LOGS_ROOT / experiment_name / resolved_run / "exported" / "policy.onnx"
    if not policy_file.is_file():
        raise FileNotFoundError(
            f"Exported policy not found for task '{task_key}' at {policy_file}. "
            "Use --list-tasks to inspect available run folder names."
        )
    return policy_file


def _training_motion_file_for_export(policy_file: Path) -> Optional[Path]:
    """Resolve the motion override recorded beside an exported RSL-RL policy."""
    env_cfg_path = policy_file.parent.parent / "params" / "env.yaml"
    if not env_cfg_path.is_file():
        return None
    for line in env_cfg_path.read_text().splitlines():
        stripped = line.strip()
        if not stripped.startswith("motion_file:"):
            continue
        value = stripped.split(":", maxsplit=1)[1].strip().strip("'\"")
        if not value or value.lower() in {"none", "null"}:
            return None
        motion_path = Path(value).expanduser()
        if not motion_path.is_absolute():
            motion_path = REPO_ROOT.parent / motion_path
        return motion_path.resolve() if motion_path.is_file() else None
    return None


def _ensure_policy_file_exists(policy_value: str, task_key: Optional[str] = None):
    policy_file = _resolve_repo_path(policy_value)
    if policy_file.is_file():
        return
    if "<" in policy_value and ">" in policy_value and task_key is not None:
        raise FileNotFoundError(
            f"Task '{task_key}' does not have a concrete default policy path configured yet. "
            "Pass --policy-path <.../policy.onnx> for that task."
        )
    raise FileNotFoundError(f"Policy file not found: {policy_file}")


def _format_task_listing() -> str:
    lines = [
        "Supported Gurukul Unitree sim2sim tasks:",
        "Viewer backends: native, viser, none. Append --viser to any task to launch MuJoCo headless with viser.",
        "",
    ]
    for task_key, spec in TASK_SPECS.items():
        lines.append(f"- {task_key}")
        lines.append(f"  task_id: {spec['task_id']}")
        lines.append(f"  config: {spec['config_path']}")
        lines.append(f"  experiment: {spec['experiment']}")
        default_policy = _default_policy_from_task(task_key)
        if default_policy is not None:
            lines.append(f"  default_policy: {default_policy}")
        if spec.get("notes"):
            lines.append(f"  notes: {spec['notes']}")
        exported_runs = _discover_exported_runs(spec["experiment"])
        if exported_runs:
            lines.append("  exported_runs:")
            for run_name in exported_runs:
                lines.append(f"    {run_name}")
        else:
            lines.append("  exported_runs: none found")

    supported_families = {spec["experiment"] for spec in TASK_SPECS.values()}
    other_families = [name for name in _discover_unitree_families() if name not in supported_families]
    if other_families:
        lines.append("")
        lines.append("Other Unitree experiment families found in ../logs/rsl_rl but not wired to this runner:")
        for family_name in other_families:
            lines.append(f"- {family_name}")
            note = EXPERIMENT_NOTES.get(family_name)
            if note:
                lines.append(f"  note: {note}")
            exported_runs = _discover_exported_runs(family_name)
            if exported_runs:
                lines.append("  exported_runs:")
                for run_name in exported_runs:
                    lines.append(f"    {run_name}")
            else:
                lines.append("  exported_runs: none found")
    return "\n".join(lines)


def _pd_gains_from_config(cfg, n_joints: int):
    """Scalar kps/kds = uniform; lists = per motor in runner motor order."""
    kp_raw = cfg["kps"]
    kd_raw = cfg["kds"]
    if isinstance(kp_raw, (list, tuple)) or isinstance(kd_raw, (list, tuple)):
        kps_a = np.array(kp_raw, dtype=np.float32).reshape(-1)
        kds_a = np.array(kd_raw, dtype=np.float32).reshape(-1)
        if kps_a.size != n_joints or kds_a.size != n_joints:
            raise ValueError(
                f"kps and kds as lists must have length {n_joints}; "
                f"got kps={kps_a.size}, kds={kds_a.size}"
            )
        return kps_a, kds_a
    return np.full(n_joints, np.float32(kp_raw), dtype=np.float32), np.full(
        n_joints, np.float32(kd_raw), dtype=np.float32
    )


def _standup_gain_array(raw, policy_gains: np.ndarray, name: str) -> np.ndarray:
    """Standup gains can be scalar, per-motor list, or "policy" to reuse runtime gains."""
    policy_gains = np.asarray(policy_gains, dtype=np.float32).reshape(-1)
    if isinstance(raw, str):
        if raw.strip().lower() in {"policy", "control", "configured"}:
            return policy_gains.copy()
        raise ValueError(f'{name} string value must be "policy"; got {raw!r}')

    arr = np.array(raw, dtype=np.float32).reshape(-1)
    if arr.size == 1:
        return np.full(policy_gains.size, arr.item(), dtype=np.float32)
    if arr.size != policy_gains.size:
        raise ValueError(f"{name} must be scalar or length {policy_gains.size}; got {arr.size}")
    return arr


gravity_vector = np.array([0, 0, -1], dtype=np.float32)


def _validate_mapping(mapping_array: np.ndarray, n_items: int, name: str = "joint_mapping"):
    if mapping_array.size != n_items:
        raise ValueError(f"{name} must have length {n_items}, got {mapping_array.size}")
    if set(mapping_array.tolist()) != set(range(n_items)):
        raise ValueError(
            f"{name} must be a permutation of 0..{n_items - 1}, got {mapping_array.tolist()}"
        )


def _apply_runtime_config(cfg: dict):
    global policy_path, xml_path, control_type, simulation_duration, simulation_dt, control_decimation
    global standup_duration_s, standup_angles, standup_start_kps, standup_kps, standup_kds
    global default_angles, lin_vel_scale, ang_vel_scale, gravity_scale, dof_pos_scale, dof_vel_scale
    global action_scale, cmd_scale, cmd_limits_cfg, cmd_smoothing_cfg, num_actions, num_obs, kps, kds
    global num_motors, num_joint_obs, controlled_motor_indices, joint_obs_mapping
    global observation_layout, cmd_init, torque_limit, action_clip, cmd_lin_x_min, cmd_lin_x_max, cmd_lin_y_min
    global cmd_lin_y_max, cmd_ang_z_min, cmd_ang_z_max, cmd_key_step, cmd_accel, cmd_decel
    global cmd_key_timeout, hip_action_scale, hip_joint_indices, action_scales, mapping, motor_names
    global default_angles_policy, default_angles_joint_obs, total_steps, depth_enabled, depth_base_obs_dim, depth_image_dim
    global depth_buffer_path, depth_raw_height, depth_raw_width, depth_crop_top, depth_crop_bottom
    global depth_crop_left, depth_crop_right, depth_resize_shape, depth_normalize, depth_max_distance
    global depth_visualize, depth_preview_scale, depth_wait_timeout
    global GO2_APEX_TRACKER_BASE_OBS_DIM, GO2_APEX_TRACKER_REFERENCE_FRAME_DIM, GO2_APEX_TRACKER_JOINT_NAMES
    global GO2_APEX_TRACKER_REFERENCE_OFFSETS, GO2_APEX_TRACKER_HISTORY_LENGTH, GO2_APEX_TRACKER_SINGLE_OBS_DIM
    global GO2_APEX_TRACKER_SKILL_DIM, actor_joint_velocity_indices
    global ISAAC_HISTORY_LENGTH, ISAAC_HISTORY_SINGLE_OBS_DIM
    global apex_motion_cfg
    global arm_command_cfg, arm_command_enabled, arm_command_indices, arm_gripper_index
    global arm_lower_limits, arm_upper_limits, arm_max_joint_delta, arm_command_period_s, arm_command_velocity
    global arm_command_model, arm_command_quantization, arm_command_natural_frequency_hz
    global arm_command_damping_ratio, arm_command_latency_s
    global arm_command_source, arm_reference_indices
    global reference_residual_action_indices
    global position_target_lower_limits, position_target_upper_limits

    policy_path = str(_resolve_repo_path(cfg["policy_path"]))
    xml_path = str(_resolve_repo_path(cfg["xml_path"]))
    control_type = cfg["control_type"]
    simulation_duration = float(cfg["simulation_duration"])
    simulation_dt = float(cfg["simulation_dt"])
    control_decimation = int(cfg["control_decimation"])
    standup_duration_s = float(cfg.get("standup_duration_s", 2.0))

    default_angles = np.array(cfg["default_angles"], dtype=np.float32)
    standup_angles = np.array(cfg.get("standup_angles", default_angles), dtype=np.float32).reshape(-1)
    lin_vel_scale = np.float32(cfg["lin_vel_scale"])
    ang_vel_scale = np.float32(cfg["ang_vel_scale"])
    gravity_scale = np.float32(cfg.get("gravity_scale", 1.0))
    dof_pos_scale = np.float32(cfg["dof_pos_scale"])
    dof_vel_scale = np.float32(cfg["dof_vel_scale"])
    action_scale = np.float32(cfg["action_scale"])
    cmd_scale = np.array(cfg["cmd_scale"], dtype=np.float32)
    cmd_limits_cfg = cfg.get("cmd_limits", {})
    cmd_smoothing_cfg = cfg.get("cmd_smoothing", {})

    num_actions = int(cfg["num_actions"])
    num_motors = int(cfg.get("num_motors", num_actions))
    num_joint_obs = int(cfg.get("num_joint_obs", num_motors))
    num_obs = int(cfg["num_obs"])
    motor_names = list(cfg.get("motor_names", []))
    if len(default_angles) != num_motors:
        raise ValueError(f"default_angles must have length num_motors={num_motors}; got {len(default_angles)}")
    if len(standup_angles) != num_motors:
        raise ValueError(f"standup_angles must have length num_motors={num_motors}; got {len(standup_angles)}")
    if len(motor_names) != num_motors:
        motor_names = [
            GO2_APEX_TRACKER_JOINT_NAMES[i] if i < len(GO2_APEX_TRACKER_JOINT_NAMES) else f"motor_{i}"
            for i in range(num_motors)
        ]
    kps, kds = _pd_gains_from_config(cfg, num_motors)
    standup_kps = _standup_gain_array(cfg.get("standup_kp", 30.0), kps, "standup_kp")
    standup_start_kps = _standup_gain_array(
        cfg.get("standup_start_kp", 60.0), standup_kps, "standup_start_kp"
    )
    standup_kds = _standup_gain_array(cfg.get("standup_kd", 1.0), kds, "standup_kd")
    observation_layout = cfg.get("observation_layout", "isaac_45")
    cmd_init = np.array(cfg["cmd_init"], dtype=np.float32)
    torque_limit = np.array(cfg["torque_limit"], dtype=np.float32)
    if torque_limit.ndim == 0:
        torque_limit = np.float32(torque_limit)
    elif torque_limit.size != num_motors:
        raise ValueError(f"torque_limit list must have length num_motors={num_motors}; got {torque_limit.size}")
    raw_action_clip = cfg.get("action_clip")
    action_clip = None if raw_action_clip is None else np.float32(raw_action_clip)
    position_target_lower_limits = np.asarray(
        cfg.get("position_target_lower_limits", [-np.inf] * num_motors),
        dtype=np.float32,
    ).reshape(-1)
    position_target_upper_limits = np.asarray(
        cfg.get("position_target_upper_limits", [np.inf] * num_motors),
        dtype=np.float32,
    ).reshape(-1)
    if (
        position_target_lower_limits.size != num_motors
        or position_target_upper_limits.size != num_motors
    ):
        raise ValueError(
            "position target limit lists must match num_motors="
            f"{num_motors}; got {position_target_lower_limits.size}/"
            f"{position_target_upper_limits.size}."
        )
    if np.any(position_target_lower_limits > position_target_upper_limits):
        raise ValueError("position_target_lower_limits must not exceed upper limits.")
    cmd_lin_x_min, cmd_lin_x_max = [np.float32(v) for v in cmd_limits_cfg.get("lin_vel_x", [0.0, 2.5])]
    cmd_lin_y_min, cmd_lin_y_max = [np.float32(v) for v in cmd_limits_cfg.get("lin_vel_y", [-0.5, 0.5])]
    cmd_ang_z_min, cmd_ang_z_max = [np.float32(v) for v in cmd_limits_cfg.get("ang_vel_z", [-1.5, 1.5])]
    cmd_key_step = _vec3_from_cfg(cmd_smoothing_cfg.get("key_step"), [0.15, 0.10, 0.20])
    cmd_accel = _vec3_from_cfg(cmd_smoothing_cfg.get("accel"), [1.8, 1.2, 3.2])
    cmd_decel = _vec3_from_cfg(cmd_smoothing_cfg.get("decel"), [2.4, 1.8, 4.5])
    cmd_key_timeout = np.float32(cmd_smoothing_cfg.get("key_timeout", 0.35))

    hip_action_scale = np.float32(cfg.get("hip_action_scale", action_scale))
    hip_joint_indices = np.array(cfg.get("hip_joint_indices", [0, 3, 6, 9]), dtype=np.int32)
    per_joint = cfg.get("action_scales_per_joint")
    if per_joint is not None:
        action_scales = np.array(per_joint, dtype=np.float32).reshape(-1)
        if action_scales.size != num_actions:
            raise ValueError(
                f"action_scales_per_joint must have length {num_actions}, got {action_scales.size}"
            )
    else:
        action_scales = np.full(num_actions, action_scale, dtype=np.float32)
        for hip_idx in hip_joint_indices:
            if 0 <= hip_idx < num_actions:
                action_scales[hip_idx] = hip_action_scale

    reference_residual_action_indices = np.array(
        cfg.get("reference_residual_action_indices", []),
        dtype=np.int32,
    ).reshape(-1)
    if np.any(reference_residual_action_indices < 0) or np.any(
        reference_residual_action_indices >= num_actions
    ):
        raise ValueError(
            "reference_residual_action_indices must be valid policy-action indices; "
            f"got {reference_residual_action_indices.tolist()} for {num_actions} actions."
        )
    if len(set(reference_residual_action_indices.tolist())) != reference_residual_action_indices.size:
        raise ValueError(
            "reference_residual_action_indices contains duplicates: "
            f"{reference_residual_action_indices.tolist()}"
        )

    mapping = np.array(cfg.get("joint_mapping", list(range(num_actions))), dtype=np.int32).reshape(-1)
    _validate_mapping(mapping, num_actions, "joint_mapping")
    controlled_motor_indices = np.array(
        cfg.get("controlled_motor_indices", list(range(num_actions))), dtype=np.int32
    ).reshape(-1)
    if controlled_motor_indices.size != num_actions:
        raise ValueError(
            f"controlled_motor_indices must have length num_actions={num_actions}; "
            f"got {controlled_motor_indices.size}"
        )
    if np.any(controlled_motor_indices < 0) or np.any(controlled_motor_indices >= num_motors):
        raise ValueError(
            f"controlled_motor_indices must be in [0, {num_motors - 1}], got {controlled_motor_indices.tolist()}"
        )
    joint_obs_mapping = np.array(cfg.get("joint_obs_mapping", list(range(num_joint_obs))), dtype=np.int32).reshape(-1)
    _validate_mapping(joint_obs_mapping, num_joint_obs, "joint_obs_mapping")
    actor_joint_velocity_indices = np.array(
        cfg.get("actor_joint_velocity_indices", list(range(num_joint_obs))),
        dtype=np.int32,
    ).reshape(-1)
    if np.any(actor_joint_velocity_indices < 0) or np.any(actor_joint_velocity_indices >= num_joint_obs):
        raise ValueError(
            f"actor_joint_velocity_indices must be in [0, {num_joint_obs - 1}], "
            f"got {actor_joint_velocity_indices.tolist()}"
        )
    if len(set(actor_joint_velocity_indices.tolist())) != actor_joint_velocity_indices.size:
        raise ValueError(
            "actor_joint_velocity_indices contains duplicates: "
            f"{actor_joint_velocity_indices.tolist()}"
        )

    arm_command_cfg = cfg.get("arm_command") or {}
    arm_command_enabled = bool(arm_command_cfg.get("enabled", False))
    if arm_command_enabled:
        arm_command_source = str(arm_command_cfg.get("source", "command"))
        if arm_command_source not in {"command", "motion_reference"}:
            raise ValueError(
                "arm_command.source must be 'command' or 'motion_reference', "
                f"got {arm_command_source!r}"
            )
        arm_command_indices = np.array(arm_command_cfg.get("indices", []), dtype=np.int32).reshape(-1)
        if arm_command_indices.size == 0:
            raise ValueError("arm_command.enabled=true requires non-empty arm_command.indices.")
        if np.any(arm_command_indices < 0) or np.any(arm_command_indices >= num_motors):
            raise ValueError(
                f"arm_command.indices must be in [0, {num_motors - 1}], got {arm_command_indices.tolist()}"
            )
        if len(set(arm_command_indices.tolist())) != arm_command_indices.size:
            raise ValueError(f"arm_command.indices contains duplicates: {arm_command_indices.tolist()}")
        arm_reference_indices = np.array(
            arm_command_cfg.get("reference_indices", []),
            dtype=np.int32,
        ).reshape(-1)
        if arm_command_source == "motion_reference":
            if arm_reference_indices.size != arm_command_indices.size:
                raise ValueError(
                    "arm_command.reference_indices must match arm_command.indices for "
                    "source='motion_reference'; "
                    f"got {arm_reference_indices.size}/{arm_command_indices.size}."
                )
            if np.any(arm_reference_indices < 0):
                raise ValueError(
                    "arm_command.reference_indices must be non-negative, "
                    f"got {arm_reference_indices.tolist()}"
                )
        elif arm_reference_indices.size:
            raise ValueError(
                "arm_command.reference_indices is only valid with source='motion_reference'."
            )
        raw_gripper_index = arm_command_cfg.get("gripper_index")
        arm_gripper_index = None if raw_gripper_index is None else int(raw_gripper_index)
        if arm_gripper_index is not None and not (0 <= arm_gripper_index < num_motors):
            raise ValueError(f"arm_command.gripper_index must be in [0, {num_motors - 1}], got {arm_gripper_index}")
        arm_lower_limits = np.array(
            arm_command_cfg.get("lower_limits", [-np.inf] * arm_command_indices.size), dtype=np.float32
        ).reshape(-1)
        arm_upper_limits = np.array(
            arm_command_cfg.get("upper_limits", [np.inf] * arm_command_indices.size), dtype=np.float32
        ).reshape(-1)
        if arm_lower_limits.size != arm_command_indices.size or arm_upper_limits.size != arm_command_indices.size:
            raise ValueError(
                "arm_command lower_limits/upper_limits must match arm_command.indices length "
                f"({arm_command_indices.size}); got {arm_lower_limits.size}/{arm_upper_limits.size}."
            )
        arm_command_model = str(arm_command_cfg.get("model", "linear_rate_limited"))
        arm_command_period_s = np.float32(arm_command_cfg.get("period_s", 0.04))
        if not np.isfinite(arm_command_period_s) or arm_command_period_s <= 0.0:
            raise ValueError(f"arm_command.period_s must be positive, got {arm_command_period_s}")
        arm_command_velocity = bool(arm_command_cfg.get("command_velocity", True))
        arm_max_joint_delta = np.float32(arm_command_cfg.get("max_joint_delta", 0.05))
        arm_command_quantization = np.asarray(
            arm_command_cfg.get("quantization_rad", 0.0),
            dtype=np.float32,
        )
        if arm_command_quantization.ndim == 0:
            arm_command_quantization = np.full(
                arm_command_indices.size,
                float(arm_command_quantization),
                dtype=np.float32,
            )
        else:
            arm_command_quantization = arm_command_quantization.reshape(-1)
            if arm_command_quantization.size != arm_command_indices.size:
                raise ValueError(
                    "arm_command.quantization_rad must be scalar or match arm_command.indices; "
                    f"got {arm_command_quantization.size}/{arm_command_indices.size}."
                )
        arm_command_natural_frequency_hz = np.float32(
            arm_command_cfg.get("natural_frequency_hz", 6.0)
        )
        arm_command_damping_ratio = np.float32(arm_command_cfg.get("damping_ratio", 1.0))
        arm_command_latency_s = np.float32(arm_command_cfg.get("latency_s", 0.0))
        if arm_command_model == "linear_rate_limited":
            if not np.isfinite(arm_max_joint_delta) or arm_max_joint_delta <= 0.0:
                raise ValueError(f"arm_command.max_joint_delta must be positive, got {arm_max_joint_delta}")
        elif arm_command_model == "sample_and_hold":
            if not np.all(np.isfinite(arm_command_quantization)) or np.any(arm_command_quantization < 0.0):
                raise ValueError(
                    f"arm_command.quantization_rad must be non-negative, got {arm_command_quantization}"
                )
        elif arm_command_model == "second_order_angle":
            if not np.all(np.isfinite(arm_command_quantization)) or np.any(arm_command_quantization < 0.0):
                raise ValueError(
                    f"arm_command.quantization_rad must be non-negative, got {arm_command_quantization}"
                )
            if not np.isfinite(arm_command_natural_frequency_hz) or arm_command_natural_frequency_hz <= 0.0:
                raise ValueError(
                    "arm_command.natural_frequency_hz must be positive, "
                    f"got {arm_command_natural_frequency_hz}"
                )
            if not np.isfinite(arm_command_damping_ratio) or arm_command_damping_ratio <= 0.0:
                raise ValueError(
                    f"arm_command.damping_ratio must be positive, got {arm_command_damping_ratio}"
                )
            if not np.isfinite(arm_command_latency_s) or arm_command_latency_s < 0.0:
                raise ValueError(f"arm_command.latency_s must be non-negative, got {arm_command_latency_s}")
        else:
            raise ValueError(
                "arm_command.model must be 'linear_rate_limited', 'sample_and_hold', or "
                "'second_order_angle', "
                f"got {arm_command_model!r}"
            )
    else:
        arm_command_source = "command"
        arm_reference_indices = np.zeros(0, dtype=np.int32)
        arm_command_indices = np.zeros(0, dtype=np.int32)
        arm_gripper_index = None
        arm_lower_limits = np.zeros(0, dtype=np.float32)
        arm_upper_limits = np.zeros(0, dtype=np.float32)
        arm_max_joint_delta = np.float32(0.05)
        arm_command_period_s = np.float32(0.04)
        arm_command_velocity = True
        arm_command_model = "linear_rate_limited"
        arm_command_quantization = np.zeros(0, dtype=np.float32)
        arm_command_natural_frequency_hz = np.float32(6.0)
        arm_command_damping_ratio = np.float32(1.0)
        arm_command_latency_s = np.float32(0.0)

    default_angles_policy = np.zeros_like(default_angles)
    for action_idx, motor_idx in enumerate(controlled_motor_indices):
        default_angles_policy[mapping[action_idx]] = default_angles[motor_idx]
    default_angles_joint_obs = np.zeros(num_joint_obs, dtype=np.float32)
    for sensor_idx, obs_idx in enumerate(joint_obs_mapping):
        default_angles_joint_obs[obs_idx] = default_angles[sensor_idx]

    depth_cfg = cfg.get("depth_camera") or {}
    depth_enabled = bool(depth_cfg.get("enabled", False))
    depth_base_obs_dim = int(depth_cfg.get("base_obs_dim", 45))
    depth_image_dim = 0
    depth_buffer_path = None
    depth_raw_height = 0
    depth_raw_width = 0
    depth_crop_top = 0
    depth_crop_bottom = 0
    depth_crop_left = 0
    depth_crop_right = 0
    depth_resize_shape = None
    depth_normalize = False
    depth_max_distance = 0.0
    depth_visualize = False
    depth_preview_scale = 4
    depth_wait_timeout = 10.0
    apex_motion_cfg = cfg.get("motion_visualization") or {}
    tracker_joint_names = cfg.get("tracker_joint_names")
    if tracker_joint_names is not None:
        GO2_APEX_TRACKER_JOINT_NAMES = tuple(str(name) for name in tracker_joint_names)
    if arm_command_source == "motion_reference":
        if not bool(apex_motion_cfg.get("enabled", False)):
            raise ValueError(
                "arm_command.source='motion_reference' requires motion_visualization.enabled=true."
            )
        if np.any(arm_reference_indices >= len(GO2_APEX_TRACKER_JOINT_NAMES)):
            raise ValueError(
                "arm_command.reference_indices exceed tracker_joint_names; "
                f"got {arm_reference_indices.tolist()} for "
                f"{len(GO2_APEX_TRACKER_JOINT_NAMES)} reference joints."
            )
    reference_offsets = apex_motion_cfg.get("reference_offsets")
    if reference_offsets is not None:
        GO2_APEX_TRACKER_REFERENCE_OFFSETS = tuple(int(offset) for offset in reference_offsets)
    else:
        GO2_APEX_TRACKER_REFERENCE_OFFSETS = GO2_APEX_TRACKER_DEFAULT_REFERENCE_OFFSETS
    if observation_layout == "go2_apex_base":
        if depth_enabled:
            raise ValueError("go2_apex_base sim2sim is proprioceptive-only; disable depth_camera.")
        if num_actions != 12 or num_joint_obs != 12:
            raise ValueError(
                "go2_apex_base requires the 12-action/12-joint Go2 contract; "
                f"got num_actions={num_actions}, num_joint_obs={num_joint_obs}."
            )
        if num_obs != GO2_APEX_BASE_OBS_DIM:
            raise ValueError(
                "go2_apex_base observation mismatch: "
                f"num_obs={num_obs}, expected {GO2_APEX_BASE_OBS_DIM}."
            )
        if cmd_scale.size != 4:
            raise ValueError(
                "go2_apex_base cmd_scale must contain vx, vy, vz, and yaw scales; "
                f"got {cmd_scale.size} values."
            )
        if not bool(apex_motion_cfg.get("enabled", False)):
            raise ValueError(
                "go2_apex_base requires motion_visualization.enabled=true to provide command_vel_z."
            )
    if observation_layout in {"go2_apex_tracker", "go2_apex_tracker_history"}:
        if depth_enabled:
            raise ValueError(f"{observation_layout} sim2sim is proprioceptive/reference-only; disable depth_camera.")
        GO2_APEX_TRACKER_BASE_OBS_DIM = (
            10 + num_joint_obs + actor_joint_velocity_indices.size + num_actions
        )
        GO2_APEX_TRACKER_REFERENCE_FRAME_DIM = len(GO2_APEX_TRACKER_JOINT_NAMES) + 10
        GO2_APEX_TRACKER_SKILL_DIM = 1 if bool(apex_motion_cfg.get("include_skill_observation", False)) else 0
        GO2_APEX_TRACKER_SINGLE_OBS_DIM = (
            GO2_APEX_TRACKER_BASE_OBS_DIM
            + GO2_APEX_TRACKER_SKILL_DIM
            + GO2_APEX_TRACKER_REFERENCE_FRAME_DIM * len(GO2_APEX_TRACKER_REFERENCE_OFFSETS)
        )
        GO2_APEX_TRACKER_HISTORY_LENGTH = int(cfg.get("history_length", 1))
        if observation_layout == "go2_apex_tracker_history":
            GO2_APEX_TRACKER_HISTORY_LENGTH = max(1, GO2_APEX_TRACKER_HISTORY_LENGTH)
            expected_tracker_obs = GO2_APEX_TRACKER_SINGLE_OBS_DIM * GO2_APEX_TRACKER_HISTORY_LENGTH
        else:
            GO2_APEX_TRACKER_HISTORY_LENGTH = 1
            expected_tracker_obs = GO2_APEX_TRACKER_SINGLE_OBS_DIM
        if num_obs != expected_tracker_obs:
            raise ValueError(
                "APEX tracker observation mismatch: "
                f"num_obs={num_obs}, observation_layout={observation_layout}, "
                f"reference_offsets={GO2_APEX_TRACKER_REFERENCE_OFFSETS}, "
                f"skill_dim={GO2_APEX_TRACKER_SKILL_DIM}, "
                f"history_length={GO2_APEX_TRACKER_HISTORY_LENGTH} imply {expected_tracker_obs}."
            )
        if not bool(apex_motion_cfg.get("enabled", False)):
            raise ValueError(f"{observation_layout} layout requires motion_visualization.enabled=true.")
    if observation_layout in {"isaac_history", "isaac_term_history"}:
        if depth_enabled:
            raise ValueError(f"{observation_layout} sim2sim is proprioceptive-only; disable depth_camera.")
        ISAAC_HISTORY_LENGTH = max(1, int(cfg.get("history_length", 1)))
        ISAAC_HISTORY_SINGLE_OBS_DIM = 9 + 2 * num_joint_obs + num_actions
        if observation_layout == "isaac_term_history":
            # Isaac Lab applies history to individual terms. PM01 intentionally
            # keeps the 3-D command current while all proprioceptive/action
            # terms carry history.
            expected_history_obs = (
                (ISAAC_HISTORY_SINGLE_OBS_DIM - 3) * ISAAC_HISTORY_LENGTH + 3
            )
        else:
            expected_history_obs = ISAAC_HISTORY_SINGLE_OBS_DIM * ISAAC_HISTORY_LENGTH
        if num_obs != expected_history_obs:
            raise ValueError(
                f"{observation_layout} observation mismatch: "
                f"num_obs={num_obs}, history_length={ISAAC_HISTORY_LENGTH}, "
                f"num_joint_obs={num_joint_obs}, num_actions={num_actions} imply {expected_history_obs}."
            )

    if depth_enabled:
        depth_buffer_path = str(depth_cfg.get("shared_buffer_path", "/tmp/Gurukul_go2_depth.mmap"))
        depth_raw_height, depth_raw_width = _pair_int_from_cfg(depth_cfg.get("raw_shape"), [60, 106])
        depth_crop_top = int(depth_cfg.get("crop_top", 0))
        depth_crop_bottom = int(depth_cfg.get("crop_bottom", 0))
        depth_crop_left = int(depth_cfg.get("crop_left", 0))
        depth_crop_right = int(depth_cfg.get("crop_right", 0))
        resize_raw = depth_cfg.get("resize")
        if resize_raw is not None:
            depth_resize_shape = _pair_int_from_cfg(resize_raw, [58, 87])
        else:
            cropped_h = depth_raw_height - depth_crop_top - depth_crop_bottom
            cropped_w = depth_raw_width - depth_crop_left - depth_crop_right
            depth_resize_shape = (cropped_h, cropped_w)
        depth_normalize = bool(depth_cfg.get("normalize", True))
        depth_max_distance = float(depth_cfg.get("max_distance", 0.0))
        depth_visualize = bool(depth_cfg.get("visualize", True))
        depth_preview_scale = int(depth_cfg.get("preview_scale", 4))
        depth_wait_timeout = float(depth_cfg.get("wait_timeout", 10.0))
        depth_image_dim = int(depth_resize_shape[0]) * int(depth_resize_shape[1])
        expected_num_obs = depth_base_obs_dim + depth_image_dim
        if num_obs != expected_num_obs:
            raise ValueError(
                f"Depth policy config mismatch: num_obs={num_obs}, "
                f"but base_obs_dim={depth_base_obs_dim} and depth_image_dim={depth_image_dim} "
                f"imply {expected_num_obs}."
            )

    total_steps = int(simulation_duration / simulation_dt)


class DepthPreviewWindow:
    def __init__(self, scale: int, normalized: bool, max_distance: float):
        self.scale = max(1, int(scale))
        self.normalized = normalized
        self.max_distance = float(max_distance)
        self.enabled = False
        self._pygame = None
        self._screen = None
        self._last_update = 0.0
        self._update_period = 0.1

        try:
            import pygame

            pygame.init()
            pygame.display.set_caption("Gurukul Depth Preview")
            self._pygame = pygame
            self.enabled = True
        except Exception as exc:
            print(f"Depth preview disabled: {exc}")

    def update(self, depth_image: np.ndarray):
        if not self.enabled:
            return
        if (time.monotonic() - self._last_update) < self._update_period:
            return

        try:
            pygame = self._pygame
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    self.close()
                    return

            preview = depth_preview_uint8(
                depth_image,
                normalized=self.normalized,
                max_distance=self.max_distance,
            )
            preview = np.repeat(np.repeat(preview, self.scale, axis=0), self.scale, axis=1)
            preview_rgb = np.repeat(preview[:, :, None], 3, axis=2)

            height, width = preview.shape
            if self._screen is None:
                self._screen = pygame.display.set_mode((width, height))

            surface = pygame.surfarray.make_surface(np.transpose(preview_rgb, (1, 0, 2)))
            self._screen.blit(surface, (0, 0))
            pygame.display.flip()
            self._last_update = time.monotonic()
        except Exception as exc:
            print(f"Depth preview disabled: {exc}")
            self.close()

    def close(self):
        if not self.enabled:
            return
        self.enabled = False
        try:
            self._pygame.display.quit()
            self._pygame.quit()
        except Exception:
            pass


class DeploymentArmCommandAdapter:
    """Shape MuJoCo arm targets like the configured hardware command loop.

    Z1 tasks retain their deployment linear interpolator. D1 tasks can use a
    pure sample-and-hold boundary or an absolute-angle model with command
    quantization, fixed transport latency, and a second-order response.
    """

    def __init__(self):
        self.indices = arm_command_indices.astype(np.int32).copy()
        self.gripper_index = arm_gripper_index
        self.lower_limits = arm_lower_limits.astype(np.float32).copy()
        self.upper_limits = arm_upper_limits.astype(np.float32).copy()
        self.period_s = float(arm_command_period_s)
        self.max_joint_delta = float(arm_max_joint_delta)
        self.command_velocity = bool(arm_command_velocity)
        self.model = str(arm_command_model)
        self.quantization = np.asarray(arm_command_quantization, dtype=np.float32).reshape(-1).copy()
        if self.quantization.size == 1:
            self.quantization = np.full(
                self.indices.size,
                float(self.quantization[0]),
                dtype=np.float32,
            )
        self.natural_frequency = 2.0 * np.pi * float(arm_command_natural_frequency_hz)
        self.damping_ratio = float(arm_command_damping_ratio)
        self.latency_s = float(arm_command_latency_s)
        self.desired_q = default_angles[self.indices].astype(np.float32).copy()
        self.segment_start_q = self.desired_q.copy()
        self.segment_target_q = self.desired_q.copy()
        self.command_q = self.desired_q.copy()
        self.command_dq = np.zeros_like(self.desired_q)
        self.elapsed_s = self.period_s
        self.initialized = False
        self.latency_steps = max(0, int(round(self.latency_s / float(simulation_dt))))
        self.target_history = np.repeat(
            self.desired_q[None, :],
            self.latency_steps + 1,
            axis=0,
        )
        self.target_history_index = 0

    def _measured_q(self, low_state):
        measured = np.array([low_state.motor_state[int(i)].q for i in self.indices], dtype=np.float32)
        if not np.all(np.isfinite(measured)):
            return self.command_q.copy()
        return measured

    def set_desired_from_cmd(self, cmd):
        for local_idx, motor_idx in enumerate(self.indices):
            self.desired_q[local_idx] = float(cmd.motor_cmd[int(motor_idx)].q)
        np.clip(self.desired_q, self.lower_limits, self.upper_limits, out=self.desired_q)
        if self.model == "second_order_angle":
            self.target_history_index = (self.target_history_index + 1) % len(self.target_history)
            self.target_history[self.target_history_index] = self.desired_q

    def _start_segment(self, measured_q):
        self.segment_start_q[:] = measured_q
        delta = np.clip(
            self.desired_q - measured_q,
            -self.max_joint_delta,
            self.max_joint_delta,
        )
        self.segment_target_q[:] = measured_q + delta
        self.command_dq[:] = delta / max(self.period_s, 1.0e-6)
        self.elapsed_s = 0.0
        self.initialized = True

    def apply(self, cmd, low_state, dt):
        if low_state is None:
            return
        measured_q = self._measured_q(low_state)
        if self.model == "sample_and_hold":
            self._apply_sample_and_hold(cmd, dt)
            return
        if self.model == "second_order_angle":
            self._apply_second_order_angle(cmd, measured_q, dt)
            return

        if not self.initialized:
            self.command_q[:] = measured_q
            self._start_segment(measured_q)

        self.elapsed_s += float(dt)
        if self.elapsed_s >= self.period_s:
            self._start_segment(measured_q)
            self.elapsed_s += float(dt)

        alpha = min(max(self.elapsed_s / self.period_s, 0.0), 1.0)
        self.command_q[:] = self.segment_start_q * (1.0 - alpha) + self.segment_target_q * alpha
        for local_idx, motor_idx in enumerate(self.indices):
            motor_cmd = cmd.motor_cmd[int(motor_idx)]
            motor_cmd.q = float(self.command_q[local_idx])
            motor_cmd.dq = float(self.command_dq[local_idx]) if self.command_velocity else 0.0

        if self.gripper_index is not None:
            gripper_cmd = cmd.motor_cmd[int(self.gripper_index)]
            gripper_cmd.dq = 0.0

    def _apply_sample_and_hold(self, cmd, dt):
        if not self.initialized:
            self.elapsed_s = self.period_s
            self.initialized = True

        self.elapsed_s += float(dt)
        while self.elapsed_s >= self.period_s:
            self.segment_target_q[:] = self.desired_q
            np.clip(self.segment_target_q, self.lower_limits, self.upper_limits, out=self.segment_target_q)
            positive_quantization = self.quantization > 0.0
            self.segment_target_q[positive_quantization] = (
                np.round(
                    self.segment_target_q[positive_quantization]
                    / self.quantization[positive_quantization]
                )
                * self.quantization[positive_quantization]
            )
            self.command_q[:] = self.segment_target_q
            self.command_dq[:] = 0.0
            self.elapsed_s -= self.period_s

        for local_idx, motor_idx in enumerate(self.indices):
            motor_cmd = cmd.motor_cmd[int(motor_idx)]
            motor_cmd.q = float(self.command_q[local_idx])
            motor_cmd.dq = 0.0

    def _apply_second_order_angle(self, cmd, measured_q, dt):
        if not self.initialized:
            self.command_q[:] = measured_q
            self.command_dq[:] = 0.0
            self.segment_target_q[:] = measured_q
            self.elapsed_s = max(self.period_s - float(dt), 0.0)
            self.initialized = True

        self.elapsed_s += float(dt)
        while self.elapsed_s >= self.period_s:
            delayed_index = (self.target_history_index - self.latency_steps) % len(self.target_history)
            self.segment_target_q[:] = self.target_history[delayed_index]
            np.clip(self.segment_target_q, self.lower_limits, self.upper_limits, out=self.segment_target_q)
            positive_quantization = self.quantization > 0.0
            self.segment_target_q[positive_quantization] = (
                np.round(
                    self.segment_target_q[positive_quantization]
                    / self.quantization[positive_quantization]
                )
                * self.quantization[positive_quantization]
            )
            self.elapsed_s -= self.period_s

        omega = self.natural_frequency
        acceleration = (
            omega * omega * (self.segment_target_q - self.command_q)
            - 2.0 * self.damping_ratio * omega * self.command_dq
        )
        self.command_dq += acceleration * float(dt)
        self.command_q += self.command_dq * float(dt)

        for local_idx, motor_idx in enumerate(self.indices):
            motor_cmd = cmd.motor_cmd[int(motor_idx)]
            motor_cmd.q = float(self.command_q[local_idx])
            # Match Isaac's explicit PD arm: the filtered position target moves,
            # but the actuator damping term uses zero desired velocity.
            motor_cmd.dq = 0.0

        if self.gripper_index is not None:
            gripper_cmd = cmd.motor_cmd[int(self.gripper_index)]
            gripper_cmd.dq = 0.0


class DepthCameraRuntime:
    def __init__(self):
        self.reader = DepthFrameReader(
            depth_buffer_path,
            expected_height=depth_raw_height,
            expected_width=depth_raw_width,
        )
        self.latest_depth = np.zeros(depth_resize_shape, dtype=np.float32)
        self.preview = (
            DepthPreviewWindow(
                scale=depth_preview_scale,
                normalized=depth_normalize,
                max_distance=depth_max_distance,
            )
            if depth_visualize
            else None
        )
        self._warned_missing = False

    def _process(self, raw_depth: np.ndarray) -> np.ndarray:
        processed = preprocess_depth_image(
            raw_depth,
            crop_top=depth_crop_top,
            crop_bottom=depth_crop_bottom,
            crop_left=depth_crop_left,
            crop_right=depth_crop_right,
            resize=depth_resize_shape,
            normalize=depth_normalize,
            max_distance=depth_max_distance,
            clip_to_max_distance=True,
        )
        np.copyto(self.latest_depth, processed)
        if self.preview is not None:
            self.preview.update(self.latest_depth)
        return self.latest_depth.reshape(-1)

    def get_flattened(self) -> Optional[np.ndarray]:
        frame = self.reader.read()
        if frame is None:
            if not self._warned_missing:
                print(
                    "Waiting for MuJoCo depth frames. "
                    f"Expected buffer: {depth_buffer_path}"
                )
                self._warned_missing = True
            return None
        self._warned_missing = False
        return self._process(frame.frame)

    def wait_until_ready(self, timeout_s: float) -> bool:
        deadline = time.time() + max(0.0, float(timeout_s))
        while time.time() < deadline:
            flattened = self.get_flattened()
            if flattened is not None:
                return True
            time.sleep(0.05)
        return False

    def close(self):
        self.reader.close()
        if self.preview is not None:
            self.preview.close()


class ApexMotionRuntime:
    def __init__(self, cfg: dict):
        self.enabled = bool(cfg.get("enabled", False))
        self.drive_command = bool(cfg.get("drive_command", True))
        self.shared_buffer_path = str(cfg.get("shared_buffer_path", APEX_VIS_DEFAULT_BUFFER))
        self.motion_files = _resolve_globbed_repo_paths(cfg.get("motion_file") or cfg.get("motion_files"))
        self.motions = []
        self.motion_names = []
        self.motion_id = 0
        self.local_frame = 0
        self.frame_accum = 0.0
        self.sequence = 0
        self.buffer = None

        if not self.enabled:
            return
        if not self.motion_files:
            raise FileNotFoundError(
                "APEX motion visualization is enabled, but no NPZ files matched "
                f"{cfg.get('motion_file') or cfg.get('motion_files')!r}."
            )

        for motion_file in self.motion_files:
            self.motions.append(self._load_motion(motion_file))
            self.motion_names.append(motion_file.name)

        Path(self.shared_buffer_path).parent.mkdir(parents=True, exist_ok=True)
        self.buffer = np.memmap(self.shared_buffer_path, dtype=np.float32, mode="w+", shape=(APEX_VIS_FLOAT_COUNT,))
        self.buffer[:] = 0.0
        self.buffer.flush()
        print(
            f"[Motion] MuJoCo APEX visualization enabled: {len(self.motions)} clip(s), "
            f"buffer={self.shared_buffer_path}"
        )
        self.print_status(force=True)

    @staticmethod
    def _load_motion(path: Path) -> dict:
        data = np.load(path)
        body_names = data["body_names"].tolist() if "body_names" in data else []
        joint_names = data["joint_names"].tolist() if "joint_names" in data else []
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
        joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
        if "gripper_joint_pos" in data:
            gripper_joint_names = (
                data["gripper_joint_names"].tolist()
                if "gripper_joint_names" in data
                else [f"gripper_{idx}" for idx in range(np.asarray(data["gripper_joint_pos"]).shape[1])]
            )
            gripper_joint_pos = np.asarray(data["gripper_joint_pos"], dtype=np.float32)
            gripper_joint_vel = (
                np.asarray(data["gripper_joint_vel"], dtype=np.float32)
                if "gripper_joint_vel" in data
                else np.zeros_like(gripper_joint_pos)
            )
            joint_names = [*joint_names, *gripper_joint_names]
            joint_pos = np.concatenate((joint_pos, gripper_joint_pos), axis=1)
            joint_vel = np.concatenate((joint_vel, gripper_joint_vel), axis=1)
        foot_indices = [idx for idx, name in enumerate(body_names) if "foot" in str(name).lower()]
        base_idx = body_names.index("base") if "base" in body_names else 0
        joint_name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
        tracker_joint_indices = [
            joint_name_to_idx[name] for name in GO2_APEX_TRACKER_JOINT_NAMES if name in joint_name_to_idx
        ]
        if not foot_indices:
            raise ValueError(f"Motion file {path} does not expose foot body names for visualization.")
        frame_count = int(joint_pos.shape[0])
        if "skill" in data:
            skill = np.asarray(data["skill"], dtype=np.float32).reshape(frame_count, -1)
            if skill.shape[1] != 1:
                raise ValueError(f"Motion file {path} skill must have one value per frame; got {skill.shape}.")
        else:
            skill = np.zeros((frame_count, 1), dtype=np.float32)
        return {
            "path": path,
            "fps": float(np.asarray(data["fps"]).item()),
            "base_idx": base_idx,
            "foot_indices": foot_indices[:4],
            "tracker_joint_indices": tracker_joint_indices,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "skill": skill,
            "body_pos_w": np.asarray(data["body_pos_w"], dtype=np.float32),
            "body_quat_w": np.asarray(data["body_quat_w"], dtype=np.float32),
            "body_lin_vel_w": np.asarray(data["body_lin_vel_w"], dtype=np.float32),
            "body_ang_vel_w": np.asarray(data["body_ang_vel_w"], dtype=np.float32),
            "command_lin_vel_xy": (
                np.asarray(data["command_lin_vel_xy"], dtype=np.float32)
                if "command_lin_vel_xy" in data
                else None
            ),
            "command_ang_vel_z": (
                np.asarray(data["command_ang_vel_z"], dtype=np.float32).reshape(-1, 1)
                if "command_ang_vel_z" in data
                else None
            ),
        }

    @property
    def active_motion(self):
        if not self.enabled or not self.motions:
            return None
        return self.motions[self.motion_id]

    def print_status(self, force: bool = False):
        motion = self.active_motion
        if motion is None:
            return
        total = motion["body_pos_w"].shape[0]
        if force:
            print(
                f"[Motion] Active reference: {self.motion_id + 1}/{len(self.motions)} "
                f"{self.motion_names[self.motion_id]} frame={self.local_frame}/{max(total - 1, 0)}"
            )

    def switch_motion(self, delta: int):
        if not self.enabled or len(self.motions) <= 1:
            return
        self.motion_id = (self.motion_id + int(delta)) % len(self.motions)
        self.local_frame = 0
        self.frame_accum = 0.0
        self.print_status(force=True)

    def update(self, dt: float, command_out: Optional[np.ndarray] = None):
        motion = self.active_motion
        if motion is None or self.buffer is None:
            return

        frame_count = motion["body_pos_w"].shape[0]
        self.frame_accum += float(dt) * motion["fps"]
        frame_step = int(self.frame_accum)
        if frame_step > 0:
            self.frame_accum -= frame_step
            self.local_frame = (self.local_frame + frame_step) % frame_count

        base_idx = motion["base_idx"]
        foot_indices = motion["foot_indices"]
        body_pos = motion["body_pos_w"][self.local_frame]
        body_quat = motion["body_quat_w"][self.local_frame]
        body_vel = motion["body_lin_vel_w"][self.local_frame]
        base_pos = body_pos[base_idx]
        base_yaw = yaw_from_quat_wxyz(body_quat[base_idx])
        foot_rel_w = body_pos[foot_indices] - base_pos
        foot_rel = foot_rel_w @ rot_z(-base_yaw).T

        if motion["command_lin_vel_xy"] is not None:
            ref_xy = motion["command_lin_vel_xy"][self.local_frame]
        else:
            ref_xy = body_vel[base_idx, :2]
        if motion["command_ang_vel_z"] is not None:
            ref_yaw = float(motion["command_ang_vel_z"][self.local_frame, 0])
        else:
            ref_yaw = 0.0

        if self.drive_command and command_out is not None:
            command_out[0] = np.clip(ref_xy[0], cmd_lin_x_min, cmd_lin_x_max)
            command_out[1] = np.clip(ref_xy[1], cmd_lin_y_min, cmd_lin_y_max)
            command_out[2] = np.clip(ref_yaw, cmd_ang_z_min, cmd_ang_z_max)

        self.sequence += 1
        self.buffer[:] = 0.0
        self.buffer[0] = APEX_VIS_MAGIC
        self.buffer[1] = 1.0
        self.buffer[2] = float(self.sequence)
        self.buffer[3] = 1.0
        self.buffer[4] = float(len(foot_indices))
        self.buffer[5] = float(self.motion_id)
        self.buffer[6] = float(self.local_frame)
        self.buffer[7] = float(frame_count)
        self.buffer[8:11] = np.array([ref_xy[0], ref_xy[1], ref_yaw], dtype=np.float32)
        flat_feet = foot_rel[:4].reshape(-1)
        self.buffer[16 : 16 + flat_feet.size] = flat_feet
        self.buffer.flush()

    def current_base_lin_vel_z(self) -> float:
        motion = self.active_motion
        if motion is None:
            return 0.0
        frame_count = motion["body_lin_vel_w"].shape[0]
        frame = min(max(int(self.local_frame), 0), max(frame_count - 1, 0))
        return float(motion["body_lin_vel_w"][frame, motion["base_idx"], 2])

    def current_skill(self) -> float:
        motion = self.active_motion
        if motion is None:
            return 0.0
        frame_count = motion["skill"].shape[0]
        frame = min(max(int(self.local_frame), 0), max(frame_count - 1, 0))
        return float(motion["skill"][frame, 0])

    def current_joint_position_reference(self) -> np.ndarray:
        """Return the current reference in configured tracker/action order."""
        motion = self.active_motion
        if motion is None:
            raise RuntimeError("Reference-residual actions require an active APEX motion runtime.")
        if len(motion["tracker_joint_indices"]) != len(GO2_APEX_TRACKER_JOINT_NAMES):
            raise RuntimeError(
                "Reference-residual action mapping requires every tracker joint; "
                f"got {len(motion['tracker_joint_indices'])}/{len(GO2_APEX_TRACKER_JOINT_NAMES)}."
            )
        frame_count = motion["joint_pos"].shape[0]
        frame = min(max(int(self.local_frame), 0), max(frame_count - 1, 0))
        return motion["joint_pos"][frame, motion["tracker_joint_indices"]].astype(
            np.float32,
            copy=False,
        )

    def reference_motion_features(self) -> np.ndarray:
        motion = self.active_motion
        if motion is None:
            raise RuntimeError("Tracker observation requested without an active APEX motion runtime.")
        if len(motion["tracker_joint_indices"]) != len(GO2_APEX_TRACKER_JOINT_NAMES):
            raise RuntimeError(
                "Active APEX motion does not contain the full Go2 tracker joint set; "
                f"got {len(motion['tracker_joint_indices'])}/{len(GO2_APEX_TRACKER_JOINT_NAMES)} joints."
            )

        frame_count = motion["body_pos_w"].shape[0]
        base_idx = motion["base_idx"]
        features = []
        for offset in GO2_APEX_TRACKER_REFERENCE_OFFSETS:
            frame = min(self.local_frame + int(offset), frame_count - 1)
            quat = motion["body_quat_w"][frame, base_idx].copy()
            norm = np.linalg.norm(quat)
            if norm > 1.0e-6:
                quat /= norm
            features.extend(
                [
                    motion["joint_pos"][frame, motion["tracker_joint_indices"]],
                    motion["body_lin_vel_w"][frame, base_idx],
                    motion["body_ang_vel_w"][frame, base_idx],
                    quat,
                ]
            )
        return np.concatenate(features).astype(np.float32)

    def close(self):
        if self.buffer is not None:
            self.buffer[3] = 0.0
            self.buffer.flush()
            self.buffer = None


class ApexTrackerDeltaLogger:
    """Record APEX tracker actor slices and per-step max deltas."""

    def __init__(self, path: Optional[str], max_steps: int = 1000):
        self.path = path
        self.max_steps = max(0, int(max_steps))
        self.enabled = bool(path) and self.max_steps > 0
        self.steps: List[int] = []
        self.values = {
            "base_ang_vel": [],
            "projected_gravity": [],
            "command": [],
            "joint_pos": [],
            "joint_vel": [],
            "prev_action": [],
            "skill": [],
            "reference_joint_pos": [],
            "reference_base_lin_vel": [],
            "reference_base_ang_vel": [],
            "reference_quat": [],
            "action": [],
        }
        self.deltas = {name: [] for name in self.values}
        self._prev = None

    @staticmethod
    def _slices(actor_obs: np.ndarray, action: np.ndarray) -> dict:
        actor_obs = np.asarray(actor_obs, dtype=np.float32).reshape(-1)
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        tracker_prefix_dim = GO2_APEX_TRACKER_BASE_OBS_DIM + GO2_APEX_TRACKER_SKILL_DIM
        if actor_obs.size >= tracker_prefix_dim and (
            actor_obs.size - tracker_prefix_dim
        ) % GO2_APEX_TRACKER_REFERENCE_FRAME_DIM == 0:
            reference_start = tracker_prefix_dim
            command = actor_obs[6:10]
            joint_pos_start = 10
            skill = actor_obs[GO2_APEX_TRACKER_BASE_OBS_DIM:tracker_prefix_dim]
        else:
            reference_start = 45
            command = actor_obs[6:9]
            joint_pos_start = 9
            skill = np.zeros(0, dtype=np.float32)
        joint_vel_start = joint_pos_start + num_joint_obs
        prev_action_start = joint_vel_start + actor_joint_velocity_indices.size

        reference = actor_obs[reference_start:]
        if reference.size % GO2_APEX_TRACKER_REFERENCE_FRAME_DIM != 0:
            reference = np.zeros(0, dtype=np.float32)
        reference_frames = (
            reference.reshape(-1, GO2_APEX_TRACKER_REFERENCE_FRAME_DIM)
            if reference.size
            else np.zeros((0, GO2_APEX_TRACKER_REFERENCE_FRAME_DIM), dtype=np.float32)
        )
        return {
            "base_ang_vel": actor_obs[0:3].copy(),
            "projected_gravity": actor_obs[3:6].copy(),
            "command": command.copy(),
            "joint_pos": actor_obs[joint_pos_start:joint_vel_start].copy(),
            "joint_vel": actor_obs[joint_vel_start:prev_action_start].copy(),
            "prev_action": actor_obs[prev_action_start : prev_action_start + num_actions].copy(),
            "skill": skill.copy(),
            "reference_joint_pos": reference_frames[:, :len(GO2_APEX_TRACKER_JOINT_NAMES)].reshape(-1).copy(),
            "reference_base_lin_vel": reference_frames[
                :, len(GO2_APEX_TRACKER_JOINT_NAMES) : len(GO2_APEX_TRACKER_JOINT_NAMES) + 3
            ].reshape(-1).copy(),
            "reference_base_ang_vel": reference_frames[
                :, len(GO2_APEX_TRACKER_JOINT_NAMES) + 3 : len(GO2_APEX_TRACKER_JOINT_NAMES) + 6
            ].reshape(-1).copy(),
            "reference_quat": reference_frames[
                :, len(GO2_APEX_TRACKER_JOINT_NAMES) + 6 : len(GO2_APEX_TRACKER_JOINT_NAMES) + 10
            ].reshape(-1).copy(),
            "action": action.copy(),
        }

    def record(self, step: int, actor_obs: np.ndarray, action: np.ndarray) -> None:
        if not self.enabled or len(self.steps) >= self.max_steps:
            return
        current = self._slices(actor_obs, action)
        self.steps.append(int(step))
        for name, value in current.items():
            self.values[name].append(value)
            if self._prev is None or self._prev[name].shape != value.shape:
                self.deltas[name].append(0.0)
            else:
                self.deltas[name].append(float(np.max(np.abs(value - self._prev[name]))))
        self._prev = {name: value.copy() for name, value in current.items()}

    def close(self) -> None:
        if not self.enabled:
            return
        output_path = Path(self.path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"steps": np.asarray(self.steps, dtype=np.int64)}
        for name, values in self.values.items():
            payload[f"{name}_values"] = np.stack(values) if values else np.zeros((0, 0), dtype=np.float32)
            payload[f"{name}_max_abs_delta"] = np.asarray(self.deltas[name], dtype=np.float32)
        np.savez(output_path, **payload)
        summary_parts = []
        for name in self.values:
            delta_values = payload[f"{name}_max_abs_delta"]
            max_delta = float(np.max(delta_values)) if delta_values.size else 0.0
            summary_parts.append(f"{name}={max_delta:.5g}")
        summary = ", ".join(summary_parts)
        print(f"[APEX delta] wrote {len(self.steps)} samples to {output_path}; max deltas: {summary}")


class ArmTrackingLogger:
    """Record the D1/Z1 target pipeline and measured arm state for gain tuning."""

    def __init__(self, path: Optional[str]):
        self.path = path
        self.enabled = bool(path)
        self.time_s = []
        self.desired_q = []
        self.command_q = []
        self.command_dq = []
        self.measured_q = []
        self.measured_dq = []
        self.kp = []
        self.kd = []
        self.tau_est = []

    def record(self, elapsed_s: float, cmd, low_state, adapter: Optional[DeploymentArmCommandAdapter]) -> None:
        if not self.enabled or adapter is None or low_state is None:
            return
        indices = adapter.indices
        self.time_s.append(float(elapsed_s))
        self.desired_q.append(adapter.desired_q.copy())
        self.command_q.append(
            np.asarray([cmd.motor_cmd[int(i)].q for i in indices], dtype=np.float32)
        )
        self.command_dq.append(adapter.command_dq.copy())
        self.measured_q.append(
            np.asarray([low_state.motor_state[int(i)].q for i in indices], dtype=np.float32)
        )
        self.measured_dq.append(
            np.asarray([low_state.motor_state[int(i)].dq for i in indices], dtype=np.float32)
        )
        self.kp.append(
            np.asarray([cmd.motor_cmd[int(i)].kp for i in indices], dtype=np.float32)
        )
        self.kd.append(
            np.asarray([cmd.motor_cmd[int(i)].kd for i in indices], dtype=np.float32)
        )
        self.tau_est.append(
            np.asarray(
                [getattr(low_state.motor_state[int(i)], "tau_est", np.nan) for i in indices],
                dtype=np.float32,
            )
        )

    def close(self, adapter: Optional[DeploymentArmCommandAdapter]) -> None:
        if not self.enabled:
            return
        output_path = Path(self.path).expanduser().resolve()
        output_path.parent.mkdir(parents=True, exist_ok=True)
        width = 0 if adapter is None else int(adapter.indices.size)

        def _stack(values):
            return (
                np.stack(values).astype(np.float32, copy=False)
                if values
                else np.zeros((0, width), dtype=np.float32)
            )

        indices = (
            np.zeros(0, dtype=np.int32)
            if adapter is None
            else adapter.indices.astype(np.int32, copy=True)
        )
        names = np.asarray(
            [
                motor_names[int(index)] if int(index) < len(motor_names) else f"motor_{int(index)}"
                for index in indices
            ],
            dtype=np.str_,
        )
        np.savez(
            output_path,
            time_s=np.asarray(self.time_s, dtype=np.float64),
            motor_indices=indices,
            joint_names=names,
            desired_q=_stack(self.desired_q),
            command_q=_stack(self.command_q),
            command_dq=_stack(self.command_dq),
            measured_q=_stack(self.measured_q),
            measured_dq=_stack(self.measured_dq),
            kp=_stack(self.kp),
            kd=_stack(self.kd),
            tau_est=_stack(self.tau_est),
        )
        print(f"[arm trace] wrote {len(self.time_s)} samples to {output_path}")


def _manual_mujoco_command(viewer: str = "native") -> str:
    scene_file = Path(str(xml_path)).name
    viewer_arg = "" if viewer == "native" else f" --viewer {viewer}"
    robot_arg = f"GURUKUL_MUJOCO_ROBOT={_infer_mujoco_robot_from_xml()} "
    return (
        f"MUJOCO_GL=glfw {robot_arg}GURUKUL_MUJOCO_SCENE={scene_file} "
        f"python unitree_mujoco/simulate_python/unitree_mujoco.py{viewer_arg}"
    )


def _infer_mujoco_robot_from_xml() -> str:
    path = Path(str(xml_path))
    for parent in [path.parent, *path.parents]:
        if parent.name in {
            "go2",
            "go2_with_d1",
            "b2",
            "b2_with_z1",
            "b2w",
            "h1",
            "go2w",
            "g1",
            "pm01",
        }:
            return parent.name
    return "b2" if "b2" in path.as_posix() else "go2"


def _launch_mujoco_subprocess(
    interface: Optional[str],
    viewer: str = "native",
    viser_host: Optional[str] = None,
    viser_port: Optional[int] = None,
) -> subprocess.Popen:
    env = os.environ.copy()
    env["GURUKUL_MUJOCO_ROBOT"] = _infer_mujoco_robot_from_xml()
    env["GURUKUL_MUJOCO_SCENE"] = str(xml_path)
    env["GURUKUL_INTERFACE"] = "lo" if interface is None else str(interface)
    env["GURUKUL_SIMULATE_DT"] = str(simulation_dt)
    env["GURUKUL_VIEWER"] = str(viewer)
    env["GURUKUL_ENABLE_DEPTH_CAMERA"] = "1" if depth_enabled else "0"
    env["GURUKUL_ENABLE_APEX_VIS_OVERLAY"] = (
        "1" if bool(apex_motion_cfg.get("enabled", False)) else "0"
    )
    if depth_enabled:
        env["GURUKUL_DEPTH_BUFFER_PATH"] = str(depth_buffer_path)
        env["GURUKUL_DEPTH_HEIGHT"] = str(depth_raw_height)
        env["GURUKUL_DEPTH_WIDTH"] = str(depth_raw_width)
    if viser_host is not None:
        env["GURUKUL_VISER_HOST"] = str(viser_host)
    if viser_port is not None:
        env["GURUKUL_VISER_PORT"] = str(viser_port)
    env["GURUKUL_APEX_VIS_BUFFER_PATH"] = str(
        apex_motion_cfg.get("shared_buffer_path", APEX_VIS_DEFAULT_BUFFER)
    )
    env["GURUKUL_APEX_VIS_FLOAT_COUNT"] = str(APEX_VIS_FLOAT_COUNT)
    default_gl = "egl" if viewer in ("viser", "none") else "glfw"
    env["MUJOCO_GL"] = env.get("GURUKUL_MUJOCO_GL", default_gl)
    cmd = [
        sys.executable,
        str(_resolve_repo_path("unitree_mujoco/simulate_python/unitree_mujoco.py")),
    ]
    proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), env=env, start_new_session=True)
    print(f"Started MuJoCo subprocess (pid={proc.pid}) with scene {xml_path}, viewer={viewer}")
    return proc


def _terminate_process_tree(proc: Optional[subprocess.Popen], timeout: float = 5.0) -> None:
    if proc is None or proc.poll() is not None:
        return

    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    except Exception:
        proc.terminate()

    try:
        proc.wait(timeout=timeout)
        return
    except subprocess.TimeoutExpired:
        pass

    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except ProcessLookupError:
        return
    except Exception:
        proc.kill()

    try:
        proc.wait(timeout=2.0)
    except subprocess.TimeoutExpired:
        print(f"[WARN] MuJoCo subprocess pid={proc.pid} did not exit after SIGKILL.")


def _runtime_lowcmd_types():
    if num_motors > 20:
        from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowCmd_ as LowCmd_default
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_ as RuntimeLowCmd
        from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_ as RuntimeLowState

        return RuntimeLowCmd, RuntimeLowState, LowCmd_default
    return LowCmd_, LowState_, unitree_go_msg_dds__LowCmd_


def _write_crc_if_present(cmd, crc: CRC) -> None:
    if hasattr(cmd, "crc"):
        cmd.crc = crc.Crc(cmd)


class RLPolicy:
    def __init__(
        self,
        motion_runtime: Optional[ApexMotionRuntime] = None,
        delta_logger: Optional[ApexTrackerDeltaLogger] = None,
    ):
        # State variables
        self.high_state = None
        self.low_state = None
        self.policy = None
        # "torch" (TorchScript .pt) or "onnx" (.onnx); set in load_policy
        self._policy_backend = None
        self.policy_session = None
        self._onnx_output_names = None
        
        # Pre-allocate all arrays to avoid repeated memory allocation
        self.lin_vel = np.zeros(3, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.projected_gravity = np.zeros(3, dtype=np.float32)
        self.dof_pos = np.zeros(num_joint_obs, dtype=np.float32)
        self.dof_vel = np.zeros(num_joint_obs, dtype=np.float32)
        self.torques = np.zeros(num_actions, dtype=np.float32)
        self.command = cmd_init.copy()
        
        # Counters
        self.step_counter = 0
        self.print_counter = 0
        
        # Action buffers
        self.prev_action = np.zeros(num_actions, dtype=np.float32)
        self.action_zero = np.zeros(num_actions, dtype=np.float32)
        self.depth_runtime = DepthCameraRuntime() if depth_enabled else None
        self.motion_runtime = motion_runtime
        self.delta_logger = delta_logger
        self.arm_command_adapter = DeploymentArmCommandAdapter() if arm_command_enabled else None

        # Pre-allocate observation tensor
        self.obs_tensor = torch.zeros(num_obs, dtype=torch.float32)
        self._history_layout = observation_layout in {
            "go2_apex_tracker_history",
            "isaac_history",
            "isaac_term_history",
        }
        self._history_single_obs_dim = (
            GO2_APEX_TRACKER_SINGLE_OBS_DIM
            if observation_layout == "go2_apex_tracker_history"
            else ISAAC_HISTORY_SINGLE_OBS_DIM
        )
        self._history_length = (
            GO2_APEX_TRACKER_HISTORY_LENGTH
            if observation_layout == "go2_apex_tracker_history"
            else ISAAC_HISTORY_LENGTH
        )
        self._history_initialized = False
        if self._history_layout:
            self.current_obs_tensor = torch.zeros(self._history_single_obs_dim, dtype=torch.float32)
            self.obs_history = torch.zeros(
                (self._history_length, self._history_single_obs_dim),
                dtype=torch.float32,
            )
            self._history_term_slices = (
                self._apex_tracker_history_term_slices()
                if observation_layout == "go2_apex_tracker_history"
                else self._isaac_history_term_slices()
            )
        else:
            self.current_obs_tensor = self.obs_tensor
            self.obs_history = None
            self._history_term_slices = []
        
        # Pre-allocate temporary arrays for calculations
        self.temp_dof_pos = np.zeros(num_joint_obs, dtype=np.float32)
        self.temp_dof_vel = np.zeros(num_joint_obs, dtype=np.float32)
        self.temp_commands = np.zeros(3, dtype=np.float32)
        self.temp_velocity_posture_commands = np.zeros(5, dtype=np.float32)
        self.temp_apex_base_command = np.zeros(4, dtype=np.float32)
        self.temp_tracker_command = np.zeros(4, dtype=np.float32)
        self.temp_phase = np.zeros(1, dtype=np.float32)
        
        # Velocity limits
        self.dof_vel_limits = np.full(num_actions, 30.0, dtype=np.float32)
        
        # Keyboard control
        self.command_lock = threading.Lock()
        self.keyboard_thread = None
        self.running = True
        self.skill_number = 0.0
        self.skill_values = [0.0, 0.25, 0.5, 0.75]
        self.cmd_lin_x_min = cmd_lin_x_min
        self.cmd_lin_x_max = cmd_lin_x_max
        self.cmd_lin_y_min = cmd_lin_y_min
        self.cmd_lin_y_max = cmd_lin_y_max
        self.cmd_ang_z_min = cmd_ang_z_min
        self.cmd_ang_z_max = cmd_ang_z_max
        self.target_command = cmd_init.copy()
        self.cmd_key_step = cmd_key_step.copy()
        self.cmd_accel = cmd_accel.copy()
        self.cmd_decel = cmd_decel.copy()
        self.cmd_key_timeout = cmd_key_timeout
        self.last_key_event_time = 0.0
        self.joystick_command = np.zeros(3, dtype=np.float32)
        self.joystick_keys = [0] * 16
        self.last_joystick_time = 0.0
        self.key_index = {
            "R1": 0,
            "L1": 1,
            "start": 2,
            "select": 3,
            "R2": 4,
            "L2": 5,
            "F1": 6,
            "F2": 7,
            "A": 8,
            "B": 9,
            "X": 10,
            "Y": 11,
            "up": 12,
            "right": 13,
            "down": 14,
            "left": 15,
        }

    def wait_for_low_state(self, timeout_s: float) -> bool:
        deadline = time.time() + max(0.0, float(timeout_s))
        while time.time() < deadline:
            if self.low_state is not None:
                return True
            # The home keyframe is passive. Poll at the physics rate so the
            # measured-pose stand-up controller takes ownership promptly.
            time.sleep(min(0.005, simulation_dt))
        return False

    def wait_for_depth_sensor(self, timeout_s: float) -> bool:
        if self.depth_runtime is None:
            return True
        return self.depth_runtime.wait_until_ready(timeout_s)

    def _clip_command_inplace(self, command_array):
        """Clamp command array to configured command limits."""
        command_array[0] = np.clip(command_array[0], self.cmd_lin_x_min, self.cmd_lin_x_max)
        command_array[1] = np.clip(command_array[1], self.cmd_lin_y_min, self.cmd_lin_y_max)
        command_array[2] = np.clip(command_array[2], self.cmd_ang_z_min, self.cmd_ang_z_max)

    def _nudge_target_command(self, axis_idx, delta):
        """Increment target command on one axis and refresh keyboard activity timestamp."""
        self.target_command[axis_idx] += delta
        self._clip_command_inplace(self.target_command)
        self.last_key_event_time = time.time()

    def _format_command_status(self, label: str) -> str:
        return (
            f"\r{label:<7} "
            f"target[x={self.target_command[0]:>5.2f}, y={self.target_command[1]:>5.2f}, yaw={self.target_command[2]:>5.2f}]  "
            f"smooth[x={self.command[0]:>5.2f}, y={self.command[1]:>5.2f}, yaw={self.command[2]:>5.2f}]"
            "        \r\n"
        )

    def _stop_all_commands(self):
        """Immediate full stop for both target and smoothed commands."""
        self.target_command[:] = 0.0
        self.command[:] = 0.0
        self.last_key_event_time = 0.0

    def _scale_joystick_axis(self, value, lower, upper):
        value = float(value)
        if abs(value) < 0.05:
            return 0.0
        return value * (float(upper) if value >= 0.0 else -float(lower))

    def _apply_joystick_command_locked(self):
        """Use the latest Unitree wireless-controller message as the command target."""
        if (time.time() - self.last_joystick_time) > 0.25:
            return False

        target = self.joystick_command.copy()
        if self.joystick_keys[self.key_index["up"]]:
            target[0] = self.cmd_lin_x_max
        if self.joystick_keys[self.key_index["down"]]:
            target[0] = self.cmd_lin_x_min
        if self.joystick_keys[self.key_index["left"]]:
            target[1] = self.cmd_lin_y_max
        if self.joystick_keys[self.key_index["right"]]:
            target[1] = self.cmd_lin_y_min
        if self.joystick_keys[self.key_index["L2"]]:
            target[2] = self.cmd_ang_z_max
        if self.joystick_keys[self.key_index["R2"]]:
            target[2] = self.cmd_ang_z_min
        b_is_motion_switch = self.motion_runtime is not None and self.motion_runtime.enabled
        if self.joystick_keys[self.key_index["A"]] or (
            self.joystick_keys[self.key_index["B"]] and not b_is_motion_switch
        ):
            target[:] = 0.0

        np.copyto(self.target_command, target)
        self._clip_command_inplace(self.target_command)
        return True

    def update_command_smoothing(self, dt):
        """Update smooth command to track keyboard target like a joystick."""
        with self.command_lock:
            if self.motion_runtime is not None and self.motion_runtime.enabled:
                command_out = self.command if self.motion_runtime.drive_command else None
                self.motion_runtime.update(dt, command_out)
                if self.motion_runtime.drive_command:
                    np.copyto(self.target_command, self.command)
                    return

            if self.last_key_event_time > 0.0:
                if (time.time() - self.last_key_event_time) > self.cmd_key_timeout:
                    # Keep translational targets latched; only yaw auto-centers.
                    self.target_command[2] = 0.0

            self._apply_joystick_command_locked()

            delta = self.target_command - self.command
            rates = np.where(
                np.abs(self.target_command) < np.abs(self.command),
                self.cmd_decel,
                self.cmd_accel,
            )
            max_step = rates * dt
            self.command += np.clip(delta, -max_step, max_step)
            self._clip_command_inplace(self.command)
            self.command[np.abs(self.command) < 1e-4] = 0.0

    def _cycle_discrete_skill(self):
        """Cycle to next discrete skill value."""
        current_idx = min(
            range(len(self.skill_values)),
            key=lambda idx: abs(self.skill_values[idx] - self.skill_number),
        )
        next_idx = (current_idx + 1) % len(self.skill_values)
        self.skill_number = self.skill_values[next_idx]

    def _cycle_continuous_skill(self):
        """Cycle skill value in 0.1 increments from 0.0 to 1.0."""
        current_step = int(round(self.skill_number * 10.0))
        next_step = (current_step + 1) % 11
        self.skill_number = next_step / 10.0

    def _onnx_feed_dict_from_obs(self, obs_1d):
        """Build ONNX Runtime feed dict from a 1-D observation vector (length num_obs)."""
        inputs = self.policy_session.get_inputs()
        if len(inputs) != 1:
            raise ValueError(
                f"ONNX policy expects 1 input, got {len(inputs)} ({[i.name for i in inputs]}). "
                "This trimmed repo only keeps single-input feedforward policies."
            )
        meta = inputs[0]
        x = np.asarray(obs_1d, dtype=np.float32).reshape(-1)
        shape = meta.shape
        _validate_policy_observation_width(shape, x.size)
        if shape is None or len(shape) <= 1:
            arr = x
        elif len(shape) == 2:
            arr = x.reshape(1, -1)
        elif len(shape) == 3:
            arr = x.reshape(1, 1, -1)
        else:
            raise ValueError(
                f"Unsupported ONNX input rank {len(shape)} for input {meta.name!r} (shape={shape})"
            )
        return {meta.name: arr}

    def load_policy(self, model_path):
        """Load policy: TorchScript (.pt) via torch.jit, or feedforward ONNX (.onnx) via ONNX Runtime."""
        path = Path(model_path)
        suffix = path.suffix.lower()
        self.policy = None
        self.policy_session = None
        self._onnx_output_names = None

        if suffix == ".onnx":
            try:
                import onnxruntime as ort
            except ImportError as err:
                raise ImportError(
                    "Loading .onnx requires onnxruntime. Install with: pip install onnxruntime "
                    "(or onnxruntime-gpu for CUDA)."
                ) from err
            sess_options = ort.SessionOptions()
            sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
            available = ort.get_available_providers()
            providers = []
            if "CUDAExecutionProvider" in available:
                providers.append("CUDAExecutionProvider")
            providers.append("CPUExecutionProvider")
            self.policy_session = ort.InferenceSession(
                str(path), sess_options=sess_options, providers=providers
            )
            self._onnx_output_names = [o.name for o in self.policy_session.get_outputs()]
            self._policy_backend = "onnx"
            print(f"{_ansi('Loaded ONNX policy', '1;32')}: {_ansi(path, '1;32')}")
            print(f"  execution providers: {self.policy_session.get_providers()}")
            print(f"  inputs: {[i.name for i in self.policy_session.get_inputs()]}")
            print(f"  outputs: {self._onnx_output_names}")
        else:
            self.policy = torch.jit.load(str(path))
            self.policy.eval()
            if hasattr(torch.jit, "optimize_for_inference"):
                self.policy = torch.jit.optimize_for_inference(self.policy)
            self._policy_backend = "torch"

    def _compute_torques(self, actions):
        """Optimized torque computation with minimal memory allocation"""
        if control_type == "torque":
            # Direct numpy operations - faster than torch
            np.multiply(actions, action_scales, out=self.torques)
            np.clip(self.torques, -torque_limit, torque_limit, out=self.torques)
        elif control_type == "position":
            # Direct scaling for position control
            np.multiply(actions, action_scales, out=self.torques)
            if reference_residual_action_indices.size:
                if self.motion_runtime is None:
                    raise RuntimeError(
                        "reference_residual_action_indices requires motion_visualization.enabled=true."
                    )
                reference_q = self.motion_runtime.current_joint_position_reference()
                residual_indices = reference_residual_action_indices
                if np.any(residual_indices >= reference_q.size):
                    raise RuntimeError(
                        "Reference motion has fewer joints than the configured residual action indices."
                    )
                self.torques[residual_indices] += (
                    reference_q[residual_indices] - default_angles_policy[residual_indices]
                )

        return self.torques

    def apply_arm_command_source(self, cmd):
        """Populate non-policy arm targets before deployment shaping."""
        if self.arm_command_adapter is None:
            return
        if arm_command_source == "motion_reference":
            if self.motion_runtime is None or not self.motion_runtime.enabled:
                raise RuntimeError(
                    "motion_reference arm commands require an active motion runtime."
                )
            reference_q = self.motion_runtime.current_joint_position_reference()
            for local_idx, motor_idx in enumerate(arm_command_indices):
                reference_idx = int(arm_reference_indices[local_idx])
                cmd.motor_cmd[int(motor_idx)].q = float(reference_q[reference_idx])
        self.arm_command_adapter.set_desired_from_cmd(cmd)

    def _apex_tracker_history_term_slices(self):
        """Return IsaacLab term slices for flattened APEX tracker history.

        IsaacLab applies group-level observation history per term before concatenating
        terms. The flattened order is therefore term-major, not full-frame-major.
        """
        reference_dim = (
            GO2_APEX_TRACKER_SINGLE_OBS_DIM
            - GO2_APEX_TRACKER_BASE_OBS_DIM
            - GO2_APEX_TRACKER_SKILL_DIM
        )
        term_sizes = (
            3,  # base_ang_vel
            3,  # projected_gravity
            1,  # command_vel_x
            1,  # command_vel_y
            1,  # command_vel_z
            1,  # command_yaw
            num_joint_obs,
            actor_joint_velocity_indices.size,
            num_actions,
            GO2_APEX_TRACKER_SKILL_DIM,
            reference_dim,
        )
        slices = []
        start = 0
        for size in term_sizes:
            end = start + int(size)
            slices.append((start, end))
            start = end
        if start != GO2_APEX_TRACKER_SINGLE_OBS_DIM:
            raise ValueError(
                "APEX tracker history slices do not cover one observation frame: "
                f"{start} != {GO2_APEX_TRACKER_SINGLE_OBS_DIM}."
            )
        return slices

    def _isaac_history_term_slices(self):
        term_sizes = (3, 3, 3, num_joint_obs, num_joint_obs, num_actions)
        slices = []
        start = 0
        for size in term_sizes:
            end = start + int(size)
            slices.append((start, end))
            start = end
        if start != ISAAC_HISTORY_SINGLE_OBS_DIM:
            raise ValueError(
                "Isaac history slices do not cover one observation frame: "
                f"{start} != {ISAAC_HISTORY_SINGLE_OBS_DIM}."
            )
        return slices

    def pd_control(self, target_q, q, kp, target_dq, dq, kd):
        """Vectorized PD control - much faster than loop"""
        np.subtract(target_q, q, out=self.torques)
        self.torques *= kp
        
        temp = target_dq - dq
        temp *= kd
        self.torques += temp
        
        return self.torques
    
    def get_action(self):
        """Get action from policy with minimal tensor operations"""
        obs = self.get_obs()
        if obs is None:
            return self.action_zero.copy(), self.action_zero.copy()

        if self._policy_backend == "onnx":
            obs_np = obs.detach().cpu().numpy()
            ort_inputs = self._onnx_feed_dict_from_obs(obs_np)
            ort_outputs = self.policy_session.run(self._onnx_output_names, ort_inputs)
            action = np.asarray(ort_outputs[0], dtype=np.float32).reshape(-1)
        else:
            with torch.no_grad():
                action_tensor = self.policy(obs)
                action = action_tensor.numpy()
        if action_clip is not None:
            np.clip(action, -action_clip, action_clip, out=action)

        if self.delta_logger is not None:
            self.delta_logger.record(self.step_counter, obs.detach().cpu().numpy(), action)
        return action, self._compute_torques(action)

    def get_obs(self):
        """Optimized observation generation with pre-allocated arrays"""
        if self.low_state is None:
            return None
        output_obs_tensor = self.obs_tensor
        if self._history_layout:
            self.obs_tensor = self.current_obs_tensor
        
        # Extract angular velocity directly to pre-allocated array
        gyro = self.low_state.imu_state.gyroscope
        self.ang_vel[0] = gyro[0]
        self.ang_vel[1] = gyro[1] 
        self.ang_vel[2] = gyro[2]

        # Compute projected gravity efficiently
        quat = self.low_state.imu_state.quaternion
        rot_matrix = quat_to_rot(quat)
        np.dot(rot_matrix.T, gravity_vector, out=self.projected_gravity)

        # Extract joint states efficiently
        motor_states = self.low_state.motor_state[:num_joint_obs]
        for i in range(num_joint_obs):
            self.temp_dof_pos[i] = motor_states[i].q
            self.temp_dof_vel[i] = motor_states[i].dq
        
        # Apply mapping and default angle offset in one operation
        for i in range(num_joint_obs):
            mapped_idx = joint_obs_mapping[i]
            self.dof_pos[mapped_idx] = self.temp_dof_pos[i] - default_angles_joint_obs[mapped_idx]
            self.dof_vel[mapped_idx] = self.temp_dof_vel[i]

        self.temp_commands[0] = self.command[0] * cmd_scale[0]
        self.temp_commands[1] = self.command[1] * cmd_scale[1]
        self.temp_commands[2] = self.command[2] * cmd_scale[2]
        self.temp_velocity_posture_commands[0:3] = self.temp_commands
        self.temp_velocity_posture_commands[3] = 0.0
        self.temp_velocity_posture_commands[4] = 0.0
        if observation_layout == "go2_apex_base":
            if self.motion_runtime is None or not self.motion_runtime.enabled:
                raise RuntimeError("go2_apex_base requires an active APEX motion runtime.")
            self.temp_apex_base_command[0] = self.command[0]
            self.temp_apex_base_command[1] = self.command[1]
            self.temp_apex_base_command[2] = self.motion_runtime.current_base_lin_vel_z()
            self.temp_apex_base_command[3] = self.command[2]
        if observation_layout in {"go2_apex_tracker", "go2_apex_tracker_history"}:
            tracker_cmd_scale = np.ones(4, dtype=np.float32)
            tracker_cmd_scale[: min(4, cmd_scale.size)] = cmd_scale[: min(4, cmd_scale.size)]
            self.temp_tracker_command[0] = self.command[0] * tracker_cmd_scale[0]
            self.temp_tracker_command[1] = self.command[1] * tracker_cmd_scale[1]
            self.temp_tracker_command[2] = (
                0.0 if self.motion_runtime is None else self.motion_runtime.current_base_lin_vel_z()
            ) * tracker_cmd_scale[2]
            self.temp_tracker_command[3] = self.command[2] * tracker_cmd_scale[3]

        skill_number = np.array([self.skill_number], dtype=np.float32)

        if self.high_state is not None:
            velocity = np.array(self.high_state.velocity, dtype=np.float32)
            np.dot(velocity, rot_matrix, out=self.lin_vel)

        # Every 100 steps print current velocity
        # if self.step_counter % 100 == 0:
        #     print(f"Step {self.step_counter}: Current Velocity - Linear: {self.lin_vel}, Angular: {self.ang_vel}")

        # Build observation tensor efficiently based on num_obs
        idx = 0
        active_num_obs = (
            self._history_single_obs_dim
            if self._history_layout
            else (depth_base_obs_dim if depth_enabled else num_obs)
        )

        if observation_layout == "go2_apex_base":
            idx = _write_go2_apex_base_observation(
                self.obs_tensor,
                base_ang_vel=self.ang_vel,
                projected_gravity=self.projected_gravity,
                motion_command=self.temp_apex_base_command,
                joint_pos=self.dof_pos,
                joint_vel=self.dof_vel,
                previous_action=self.prev_action,
                ang_vel_scale=ang_vel_scale,
                gravity_scale=gravity_scale,
                cmd_scale=cmd_scale,
                dof_pos_scale=dof_pos_scale,
                dof_vel_scale=dof_vel_scale,
            )

        elif active_num_obs == 39:
            self.obs_tensor[idx:idx+3] = torch.from_numpy(self.projected_gravity * gravity_scale)
            idx += 3
            self.obs_tensor[idx:idx+12] = torch.from_numpy(self.dof_pos * dof_pos_scale)
            idx += 12
            self.obs_tensor[idx:idx+12] = torch.from_numpy(self.dof_vel * dof_vel_scale)
            idx += 12
            self.obs_tensor[idx:idx+12] = torch.from_numpy(self.prev_action)
            
        elif active_num_obs == 42:
            self.obs_tensor[idx:idx+3] = torch.from_numpy(self.projected_gravity * gravity_scale)
            idx += 3
            self.obs_tensor[idx:idx+3] = torch.from_numpy(self.temp_commands)
            idx += 3
            self.obs_tensor[idx:idx+12] = torch.from_numpy(self.dof_pos * dof_pos_scale)
            idx += 12
            self.obs_tensor[idx:idx+12] = torch.from_numpy(self.dof_vel * dof_vel_scale)
            idx += 12
            self.obs_tensor[idx:idx+12] = torch.from_numpy(self.prev_action)
            
        elif active_num_obs == 48:
            self.obs_tensor[idx:idx+3] = torch.from_numpy(self.lin_vel * lin_vel_scale)
            idx += 3
            self.obs_tensor[idx:idx+3] = torch.from_numpy(self.ang_vel * ang_vel_scale)
            idx += 3
            self.obs_tensor[idx:idx+3] = torch.from_numpy(self.projected_gravity * gravity_scale)
            idx += 3
            self.obs_tensor[idx:idx+3] = torch.from_numpy(self.temp_commands)
            idx += 3
            self.obs_tensor[idx:idx+12] = torch.from_numpy(self.dof_pos * dof_pos_scale)
            idx += 12
            self.obs_tensor[idx:idx+12] = torch.from_numpy(self.dof_vel * dof_vel_scale)
            idx += 12
            self.obs_tensor[idx:idx+12] = torch.from_numpy(self.prev_action)
            
        elif active_num_obs == 43:
            self.obs_tensor[idx:idx+3] = torch.from_numpy(self.projected_gravity * gravity_scale)
            idx += 3
            self.obs_tensor[idx:idx+3] = torch.from_numpy(self.temp_commands)
            idx += 3
            self.obs_tensor[idx:idx+12] = torch.from_numpy(self.dof_pos * dof_pos_scale)
            idx += 12
            self.obs_tensor[idx:idx+12] = torch.from_numpy(self.dof_vel * dof_vel_scale)
            idx += 12
            self.obs_tensor[idx:idx+12] = torch.from_numpy(self.prev_action)
            idx += 12
            self.obs_tensor[idx:idx+1] = torch.from_numpy(self.temp_phase)
        
        elif active_num_obs == 11 + 2 * num_joint_obs + num_actions:
            self.obs_tensor[idx : idx + 3] = torch.from_numpy(self.ang_vel * ang_vel_scale)
            idx += 3
            self.obs_tensor[idx : idx + 3] = torch.from_numpy(self.projected_gravity * gravity_scale)
            idx += 3
            self.obs_tensor[idx : idx + 5] = torch.from_numpy(self.temp_velocity_posture_commands)
            idx += 5
            self.obs_tensor[idx : idx + num_joint_obs] = torch.from_numpy(self.dof_pos * dof_pos_scale)
            idx += num_joint_obs
            self.obs_tensor[idx : idx + num_joint_obs] = torch.from_numpy(self.dof_vel * dof_vel_scale)
            idx += num_joint_obs
            self.obs_tensor[idx : idx + num_actions] = torch.from_numpy(self.prev_action)
            idx += num_actions

        elif (
            active_num_obs in (
                45,
                GO2_APEX_TRACKER_BASE_OBS_DIM
                + GO2_APEX_TRACKER_SKILL_DIM
                + GO2_APEX_TRACKER_REFERENCE_FRAME_DIM * len(GO2_APEX_TRACKER_REFERENCE_OFFSETS),
            )
            or active_num_obs == 9 + 3 * num_actions
        ):
            # Isaac / common sim2sim layout: ω, g, cmd, q, dq, a_prev
            # MJLab Go2 flat (after base_lin_vel + height_scan removed): ω, g, q, dq, a_prev, cmd
            if observation_layout in {"go2_apex_tracker", "go2_apex_tracker_history"} and active_num_obs == 45:
                raise ValueError(
                    "go2_apex_tracker layout requires reference-motion observations after its tracker base."
                )
            if observation_layout == "mjlab_go2_velocity":
                self.obs_tensor[idx : idx + 3] = torch.from_numpy(self.ang_vel * ang_vel_scale)
                idx += 3
                self.obs_tensor[idx : idx + 3] = torch.from_numpy(self.projected_gravity * gravity_scale)
                idx += 3
                self.obs_tensor[idx : idx + num_actions] = torch.from_numpy(self.dof_pos * dof_pos_scale)
                idx += num_actions
                self.obs_tensor[idx : idx + num_actions] = torch.from_numpy(self.dof_vel * dof_vel_scale)
                idx += num_actions
                self.obs_tensor[idx : idx + num_actions] = torch.from_numpy(self.prev_action)
                idx += num_actions
                self.obs_tensor[idx : idx + 3] = torch.from_numpy(self.temp_commands)
                idx += 3
            else:
                self.obs_tensor[idx : idx + 3] = torch.from_numpy(self.ang_vel * ang_vel_scale)
                idx += 3
                self.obs_tensor[idx : idx + 3] = torch.from_numpy(self.projected_gravity * gravity_scale)
                idx += 3
                if observation_layout in {"go2_apex_tracker", "go2_apex_tracker_history"}:
                    self.obs_tensor[idx : idx + 4] = torch.from_numpy(self.temp_tracker_command)
                    idx += 4
                else:
                    self.obs_tensor[idx : idx + 3] = torch.from_numpy(self.temp_commands)
                    idx += 3
                if observation_layout in {"go2_apex_tracker", "go2_apex_tracker_history"}:
                    self.obs_tensor[idx : idx + num_joint_obs] = torch.from_numpy(self.dof_pos * dof_pos_scale)
                    idx += num_joint_obs
                    actor_joint_vel = self.dof_vel[actor_joint_velocity_indices] * dof_vel_scale
                    self.obs_tensor[idx : idx + actor_joint_velocity_indices.size] = torch.from_numpy(actor_joint_vel)
                    idx += actor_joint_velocity_indices.size
                else:
                    self.obs_tensor[idx : idx + num_actions] = torch.from_numpy(self.dof_pos * dof_pos_scale)
                    idx += num_actions
                    self.obs_tensor[idx : idx + num_actions] = torch.from_numpy(self.dof_vel * dof_vel_scale)
                    idx += num_actions
                self.obs_tensor[idx : idx + num_actions] = torch.from_numpy(self.prev_action)
                idx += num_actions
                if GO2_APEX_TRACKER_SKILL_DIM:
                    if self.motion_runtime is None:
                        raise ValueError("APEX skill observation requires an active motion runtime.")
                    self.obs_tensor[idx] = self.motion_runtime.current_skill()
                    idx += GO2_APEX_TRACKER_SKILL_DIM

            if (
                active_num_obs > GO2_APEX_TRACKER_BASE_OBS_DIM
                and observation_layout in {"go2_apex_tracker", "go2_apex_tracker_history"}
            ):
                if self.motion_runtime is None:
                    raise ValueError(
                        f"{active_num_obs}-D APEX tracker observations require motion_visualization.enabled=true."
                    )
                reference_features = self.motion_runtime.reference_motion_features()
                expected_reference_size = active_num_obs - idx
                if reference_features.size != expected_reference_size:
                    raise ValueError(
                        "APEX tracker reference feature size mismatch: "
                        f"got {reference_features.size}, expected {expected_reference_size}."
                    )
                self.obs_tensor[idx : idx + reference_features.size] = torch.from_numpy(reference_features)
                idx += reference_features.size
        
        elif active_num_obs == 46:
            self.obs_tensor[idx:idx+3] = torch.from_numpy(self.ang_vel * ang_vel_scale)
            idx += 3
            self.obs_tensor[idx:idx+3] = torch.from_numpy(self.projected_gravity * gravity_scale)
            idx += 3
            self.obs_tensor[idx:idx+3] = torch.from_numpy(self.temp_commands)
            idx += 3
            self.obs_tensor[idx:idx+12] = torch.from_numpy(self.dof_pos * dof_pos_scale)
            idx += 12
            self.obs_tensor[idx:idx+12] = torch.from_numpy(self.dof_vel * dof_vel_scale)
            idx += 12
            self.obs_tensor[idx:idx+12] = torch.from_numpy(self.prev_action)
            idx += 12
            self.obs_tensor[idx:idx+1] = torch.from_numpy(skill_number)

        else:
            raise ValueError(f"Unsupported base observation size {active_num_obs} for this runner.")

        if depth_enabled:
            depth_flat = self.depth_runtime.get_flattened()
            if depth_flat is None:
                return None
            depth_end = idx + depth_image_dim
            self.obs_tensor[idx:depth_end] = torch.from_numpy(depth_flat)
            idx = depth_end

        if self._history_layout:
            if not self._history_initialized:
                self.obs_history[:] = self.current_obs_tensor
                self._history_initialized = True
            else:
                self.obs_history[:-1] = self.obs_history[1:].clone()
                self.obs_history[-1] = self.current_obs_tensor
            output_idx = 0
            for term_index, (start, end) in enumerate(self._history_term_slices):
                if observation_layout == "isaac_term_history" and term_index == 2:
                    # velocity_commands has history_length=0 in the Isaac task.
                    term_history = self.obs_history[-1, start:end]
                else:
                    term_history = self.obs_history[:, start:end].reshape(-1)
                next_idx = output_idx + term_history.numel()
                output_obs_tensor[output_idx:next_idx] = term_history
                output_idx = next_idx
            if output_idx != output_obs_tensor.numel():
                raise ValueError(
                    f"History observation wrote {output_idx} values, expected {output_obs_tensor.numel()}."
                )
            self.obs_tensor = output_obs_tensor

        return self.obs_tensor

    def LowStateHandler(self, msg: LowState_):
        """Handle low state messages"""
        self.low_state = msg

    def HighStateHandler(self, msg: SportModeState_):
        """Handle high state messages"""
        self.high_state = msg

    def WirelessControllerHandler(self, msg: WirelessController_):
        with self.command_lock:
            old_x = self.joystick_keys[self.key_index["X"]]
            old_b = self.joystick_keys[self.key_index["B"]]
            self.joystick_command[0] = self._scale_joystick_axis(msg.ly, self.cmd_lin_x_min, self.cmd_lin_x_max)
            self.joystick_command[1] = self._scale_joystick_axis(msg.lx, self.cmd_lin_y_min, self.cmd_lin_y_max)
            self.joystick_command[2] = self._scale_joystick_axis(-msg.rx, self.cmd_ang_z_min, self.cmd_ang_z_max)
            for i in range(16):
                self.joystick_keys[i] = (msg.keys & (1 << i)) >> i
            if self.motion_runtime is not None and self.motion_runtime.enabled:
                if old_x == 0 and self.joystick_keys[self.key_index["X"]]:
                    self.motion_runtime.switch_motion(delta=1)
                if old_b == 0 and self.joystick_keys[self.key_index["B"]]:
                    self.motion_runtime.switch_motion(delta=-1)
            self.last_joystick_time = time.time()
    
    def keyboard_control(self):
        """Handle keyboard input for robot control"""
        import termios
        import tty
        
        # Save terminal settings
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        
        try:
            tty.setraw(fd)
            sys.stdout.write("\r\n=== Keyboard Control ===\r\n")
            sys.stdout.write("W/S: Forward/Backward\r\n")
            sys.stdout.write("A/D: Left/Right\r\n")
            sys.stdout.write("Q/E: Turn Left/Right\r\n")
            sys.stdout.write("X: Stop\r\n")
            if self.motion_runtime is not None and self.motion_runtime.enabled:
                sys.stdout.write("N/P: Next/Previous APEX reference motion\r\n")
                if self.motion_runtime.drive_command:
                    sys.stdout.write("APEX motion command drive is active; W/A/S/D/Q/E are ignored while enabled.\r\n")
            if num_obs == 46 and observation_layout != "go2_apex_base":
                sys.stdout.write("T: Toggle Skill (0.0/0.25/0.5/0.75)\r\n")
                sys.stdout.write("G: Skill +0.1 (0.0 -> 1.0, wrap)\r\n")
            sys.stdout.write("ESC/Ctrl+C: Quit\r\n")
            sys.stdout.write("========================\r\n\r\n")
            sys.stdout.flush()
            
            while self.running:
                if select.select([sys.stdin], [], [], 0.1)[0]:
                    keys = [sys.stdin.read(1)]
                    while select.select([sys.stdin], [], [], 0.0)[0]:
                        keys.append(sys.stdin.read(1))

                    with self.command_lock:
                        status_label = None
                        for key in keys:
                            if key == 'w' or key == 'W':
                                self._nudge_target_command(0, self.cmd_key_step[0])
                                status_label = "Target"
                            elif key == 's' or key == 'S':
                                self._nudge_target_command(0, -self.cmd_key_step[0])
                                status_label = "Target"
                            elif key == 'a' or key == 'A':
                                self._nudge_target_command(1, self.cmd_key_step[1])
                                status_label = "Target"
                            elif key == 'd' or key == 'D':
                                self._nudge_target_command(1, -self.cmd_key_step[1])
                                status_label = "Target"
                            elif key == 'q' or key == 'Q':
                                self._nudge_target_command(2, self.cmd_key_step[2])
                                status_label = "Target"
                            elif key == 'e' or key == 'E':
                                self._nudge_target_command(2, -self.cmd_key_step[2])
                                status_label = "Target"
                            elif key == 'x' or key == 'X':
                                self._stop_all_commands()
                                status_label = "Stopped"
                            elif (
                                (key == 't' or key == 'T')
                                and num_obs == 46
                                and observation_layout != "go2_apex_base"
                            ):
                                self._cycle_discrete_skill()
                                sys.stdout.write(f"\rSkill Number: {self.skill_number:.2f}          \r\n")
                            elif (
                                (key == 'g' or key == 'G')
                                and num_obs == 46
                                and observation_layout != "go2_apex_base"
                            ):
                                self._cycle_continuous_skill()
                                sys.stdout.write(f"\rSkill Number: {self.skill_number:.1f}          \r\n")
                            elif (key == 'n' or key == 'N') and self.motion_runtime is not None:
                                self.motion_runtime.switch_motion(delta=1)
                            elif (key == 'p' or key == 'P') and self.motion_runtime is not None:
                                self.motion_runtime.switch_motion(delta=-1)
                            elif key == '\x1b':  # ESC key
                                sys.stdout.write("\r\nExiting keyboard control...\r\n")
                                self.running = False
                                break
                            elif key == '\x03':  # Ctrl+C in raw mode
                                sys.stdout.write("\r\nCtrl+C detected. Exiting keyboard control...\r\n")
                                self.running = False
                                break
                        if status_label is not None:
                            sys.stdout.write(self._format_command_status(status_label))
                        sys.stdout.flush()
        finally:
            # Restore terminal settings
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    
    def start_keyboard_control(self):
        """Start keyboard control in a separate thread"""
        if not sys.stdin.isatty():
            return
        self.keyboard_thread = threading.Thread(target=self.keyboard_control, daemon=True)
        self.keyboard_thread.start()
    
    def stop_keyboard_control(self):
        """Stop keyboard control thread"""
        self.running = False
        if self.keyboard_thread:
            self.keyboard_thread.join(timeout=1.0)
        if self.depth_runtime is not None:
            self.depth_runtime.close()
        if self.motion_runtime is not None:
            self.motion_runtime.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RL policy in simulation.")
    parser.add_argument(
        "-c",
        "--config",
        type=str,
        default=None,
        help=(
            "YAML config path (relative to repo root or absolute). "
            "If omitted, uses --task when provided, otherwise RL_POLICY_CONFIG or configs/go2.yaml."
        ),
    )
    parser.add_argument(
        "--task",
        type=str,
        default=None,
        help=(
            "Gurukul Unitree task selector. Supported keys: "
            + ", ".join(TASK_SPECS.keys())
            + ". Also accepts the full Gurukul task ID."
        ),
    )
    parser.add_argument(
        "--list-tasks",
        action="store_true",
        help="Print the supported Gurukul Unitree tasks and exported Unitree runs under ../logs/rsl_rl.",
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help=(
            "Exported run folder under the selected task family. "
            "Example: 2026-03-12_19-03-13. Use 'latest' for the newest exported run."
        ),
    )
    parser.add_argument("-p", "--policy-path", "-policy-path", type=str, default=None,
                        help="Override policy path: TorchScript .pt or feedforward .onnx (onnxruntime).")
    parser.add_argument("-n", "--num-obs", type=int, default=None,
                        help="Override number of observations for the policy.")
    parser.add_argument(
        "--motion-file",
        type=str,
        default=None,
        help="Override APEX motion visualization NPZ file, directory, or glob.",
    )
    parser.add_argument(
        "--apex-delta-log",
        type=str,
        default=None,
        help="Optional NPZ path for APEX tracker actor-observation/action delta diagnostics.",
    )
    parser.add_argument(
        "--apex-delta-log-steps",
        type=int,
        default=1000,
        help="Maximum number of APEX tracker diagnostic samples to write when --apex-delta-log is set.",
    )
    parser.add_argument(
        "--arm-trace-log",
        type=str,
        default=None,
        help=(
            "Optional NPZ path for arm desired, filtered-command, measured-state, "
            "gain, and estimated-torque traces."
        ),
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=None,
        help="Override the configured simulation duration in seconds.",
    )
    parser.add_argument(
        "--launch-mujoco",
        action="store_true",
        help="Launch unitree_mujoco/simulate_python/unitree_mujoco.py in a subprocess before starting the runner.",
    )
    parser.add_argument(
        "--mujoco-viewer",
        choices=("native", "viser", "none"),
        default="native",
        help="Viewer backend for --launch-mujoco. Use 'viser' for the browser-based viser server.",
    )
    parser.add_argument(
        "--viser",
        action="store_true",
        help="Shortcut for --launch-mujoco --mujoco-viewer viser.",
    )
    parser.add_argument(
        "--viser-host",
        type=str,
        default=None,
        help="Host for the launched viser server.",
    )
    parser.add_argument(
        "--viser-port",
        type=int,
        default=None,
        help="Port for the launched viser server.",
    )
    parser.add_argument("interface", nargs="?", default=None,
                        help="Network interface (default: lo for simulation).")
    args = parser.parse_args()

    if args.viser:
        args.launch_mujoco = True
        args.mujoco_viewer = "viser"

    if args.list_tasks:
        print(_format_task_listing())
        sys.exit(0)

    if args.task and args.config:
        raise ValueError("Use either --task or --config, not both.")
    if args.run and not args.task:
        raise ValueError("--run requires --task.")
    if args.run and args.policy_path:
        raise ValueError("Use either --run or --policy-path, not both.")

    task_key = _resolve_task_key(args.task) if args.task else None
    if task_key is not None:
        active_config_path = _resolve_task_config_path(task_key)
    elif args.config:
        active_config_path = _resolve_repo_path(args.config)
    else:
        active_config_path = _resolve_repo_path(DEFAULT_CONFIG_ARG)

    runtime_cfg = dict(_load_yaml_config(active_config_path))
    if args.run:
        run_policy_path = _resolve_run_policy(task_key, args.run)
        runtime_cfg["policy_path"] = str(run_policy_path)
        if args.motion_file is None:
            training_motion_file = _training_motion_file_for_export(run_policy_path)
            if training_motion_file is not None and runtime_cfg.get("motion_visualization") is not None:
                motion_cfg = dict(runtime_cfg["motion_visualization"])
                motion_cfg["enabled"] = True
                motion_cfg["motion_file"] = str(training_motion_file)
                runtime_cfg["motion_visualization"] = motion_cfg
                print(f"[Run] Using training motion recorded with export: {training_motion_file}")
    elif args.policy_path:
        runtime_cfg["policy_path"] = str(_resolve_repo_path(args.policy_path))

    if args.num_obs is not None:
        runtime_cfg["num_obs"] = int(args.num_obs)
    if args.duration is not None:
        if not np.isfinite(args.duration) or args.duration <= 0.0:
            raise ValueError(f"--duration must be positive, got {args.duration}")
        runtime_cfg["simulation_duration"] = float(args.duration)
    if args.motion_file is not None:
        motion_cfg = dict(runtime_cfg.get("motion_visualization") or {})
        motion_cfg["enabled"] = True
        motion_cfg["motion_file"] = args.motion_file
        runtime_cfg["motion_visualization"] = motion_cfg
    if args.mujoco_viewer == "viser":
        depth_cfg = dict(runtime_cfg.get("depth_camera") or {})
        if depth_cfg.get("enabled", False):
            depth_cfg["visualize"] = False
            runtime_cfg["depth_camera"] = depth_cfg

    _ensure_policy_file_exists(str(runtime_cfg["policy_path"]), task_key)
    _apply_runtime_config(runtime_cfg)

    _print_runtime_summary(task_key, active_config_path, policy_path)
    _print_policy_interface_summary()

    motion_runtime = ApexMotionRuntime(apex_motion_cfg)
    motion_runtime.update(0.0)
    apex_delta_logger = ApexTrackerDeltaLogger(args.apex_delta_log, args.apex_delta_log_steps)
    if apex_delta_logger.enabled:
        print(f"[APEX delta] logging actor slices to {Path(args.apex_delta_log).expanduser().resolve()}")
    arm_tracking_logger = ArmTrackingLogger(args.arm_trace_log)
    if arm_tracking_logger.enabled:
        print(f"[arm trace] logging arm command/state to {Path(args.arm_trace_log).expanduser().resolve()}")
    rl_policy = RLPolicy(motion_runtime=motion_runtime, delta_logger=apex_delta_logger)
    mujoco_proc = None

    try:
        # Initialization
        if args.interface is None:
            ChannelFactoryInitialize(1, "lo")
        else:
            ChannelFactoryInitialize(0, args.interface)

        # Create subscribers and publishers
        RuntimeLowCmd, RuntimeLowState, RuntimeLowCmd_default = _runtime_lowcmd_types()
        hight_state_suber = ChannelSubscriber("rt/sportmodestate", SportModeState_)
        low_state_suber = ChannelSubscriber("rt/lowstate", RuntimeLowState)
        wireless_controller_suber = ChannelSubscriber("rt/wirelesscontroller", WirelessController_)

        hight_state_suber.Init(rl_policy.HighStateHandler, 10)
        low_state_suber.Init(rl_policy.LowStateHandler, 10)
        wireless_controller_suber.Init(rl_policy.WirelessControllerHandler, 10)

        low_cmd_puber = ChannelPublisher("rt/lowcmd", RuntimeLowCmd)
        low_cmd_puber.Init()
        crc = CRC()

        cmd = RuntimeLowCmd_default()
        if hasattr(cmd, "head"):
            cmd.head[0] = 0xFE
            cmd.head[1] = 0xEF
        if hasattr(cmd, "level_flag"):
            cmd.level_flag = 0xFF
        if hasattr(cmd, "gpio"):
            cmd.gpio = 0

        # Load policy before launching MuJoCo so a freshly launched robot does not spend
        # startup time falling without commands while the ONNX/Torch model is loaded.
        rl_policy.load_policy(policy_path)

        if args.launch_mujoco:
            mujoco_proc = _launch_mujoco_subprocess(
                args.interface,
                viewer=args.mujoco_viewer,
                viser_host=args.viser_host,
                viser_port=args.viser_port,
            )

        if not rl_policy.wait_for_low_state(5.0):
            raise RuntimeError(
                "No rt/lowstate messages received from MuJoCo. "
                "Start the simulator first with: "
                f"{_manual_mujoco_command(args.mujoco_viewer)}"
            )
        if depth_enabled:
            print(
                "Depth task selected. "
                f"Waiting up to {depth_wait_timeout:.1f}s for MuJoCo frames from {depth_buffer_path}."
            )
            if not rl_policy.wait_for_depth_sensor(depth_wait_timeout):
                if rl_policy.low_state is None:
                    raise RuntimeError(
                        "No simulator data received from MuJoCo. "
                        "Start the simulator first with: "
                        f"{_manual_mujoco_command(args.mujoco_viewer)}"
                    )
                raise RuntimeError(
                    "rt/lowstate is arriving, but no depth frames were published. "
                    "Make sure MuJoCo is running the updated "
                    "unitree_mujoco/simulate_python/unitree_mujoco.py and that the depth task uses the Go2 scene "
                    f"with camera '{os.environ.get('GURUKUL_DEPTH_CAMERA_NAME', 'Gurukul_depth')}'. "
                    f"Expected shared buffer: {depth_buffer_path}"
                )
        
        # Start keyboard control
        rl_policy.start_keyboard_control()

        # Signal handler for interactive exits and job-manager termination.
        def signal_handler(sig, frame):
            sys.stdout.write("\r\nShutdown requested. Exiting policy loop...\r\n")
            sys.stdout.flush()
            rl_policy.running = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        if standup_duration_s > 0.0:
            print(_status("policy", "standing up", "1;34"))
        else:
            print(_status("policy", "direct handover (stand-up disabled)", "1;34"))
        if rl_policy.low_state is not None:
            stand_down_joint_pos = np.array(
                [rl_policy.low_state.motor_state[i].q for i in range(num_motors)], dtype=np.float32
            )
        else:
            stand_down_joint_pos = default_angles.copy()
        stand_up_joint_pos = standup_angles.astype(np.float32).copy()

        running_time = 0.0
        step_start = time.perf_counter()
        while running_time < standup_duration_s and rl_policy.running:
            running_time += 0.002
            phase = np.tanh(running_time / max(0.24 * standup_duration_s, 1.0e-6))
            stand_q = phase * stand_up_joint_pos + (1.0 - phase) * stand_down_joint_pos
            stand_kp = phase * standup_kps + (1.0 - phase) * standup_start_kps
            for i in range(num_motors):
                cmd.motor_cmd[i].q = float(stand_q[i])
                cmd.motor_cmd[i].kp = float(stand_kp[i])
                cmd.motor_cmd[i].dq = 0.0
                cmd.motor_cmd[i].kd = float(standup_kds[i])
                cmd.motor_cmd[i].tau = 0.0

            _write_crc_if_present(cmd, crc)
            low_cmd_puber.Write(cmd)
            time.sleep(max(0.0, 0.002 - (time.perf_counter() - step_start)))
            step_start = time.perf_counter()
        
        if not rl_policy.running:
            print("Exited during standup phase")
            sys.exit(0)

        # Main control loop with optimizations
        print(_status("policy", "running", "1;32"))
        start_time = time.time()
        loop_counter = 0

        # Seed frame-zero motion commands before the first observation. Keep
        # this first action for a full control-decimation interval, matching
        # Isaac Lab instead of evaluating the policy again on loop zero.
        rl_policy.update_command_smoothing(0.0)
        raw_action, calculated_action = rl_policy.get_action()
        np.copyto(rl_policy.prev_action, raw_action)
        rl_policy.step_counter += 1
        
        while time.time() - start_time < simulation_duration and rl_policy.running:
            loop_start = time.time()
            rl_policy.update_command_smoothing(simulation_dt)

            if loop_counter > 0 and loop_counter % control_decimation == 0:
                raw_action, calculated_action = rl_policy.get_action()
                np.copyto(rl_policy.prev_action, raw_action)
                rl_policy.step_counter += 1

            if control_type == 'position':
                # Position control: policy-controlled motors receive actor targets; the rest hold default pose.
                for i in range(num_motors):
                    cmd.motor_cmd[i].q = float(default_angles[i])
                    cmd.motor_cmd[i].kp = float(kps[i])
                    cmd.motor_cmd[i].dq = 0.0
                    cmd.motor_cmd[i].kd = float(kds[i])
                    cmd.motor_cmd[i].tau = 0.0
                for action_idx, motor_idx in enumerate(controlled_motor_indices):
                    mapped_idx = mapping[action_idx]
                    target_q = calculated_action[mapped_idx] + default_angles[motor_idx]
                    cmd.motor_cmd[motor_idx].q = float(
                        np.clip(
                            target_q,
                            position_target_lower_limits[motor_idx],
                            position_target_upper_limits[motor_idx],
                        )
                    )
                if rl_policy.arm_command_adapter is not None:
                    rl_policy.apply_arm_command_source(cmd)
                    rl_policy.arm_command_adapter.apply(cmd, rl_policy.low_state, simulation_dt)
                    arm_tracking_logger.record(
                        time.time() - start_time,
                        cmd,
                        rl_policy.low_state,
                        rl_policy.arm_command_adapter,
                    )

            else:
                # Torque control: non-policy motors are passive.
                for i in range(num_motors):
                    cmd.motor_cmd[i].q = 0.0
                    cmd.motor_cmd[i].kp = 0.0
                    cmd.motor_cmd[i].dq = 0.0
                    cmd.motor_cmd[i].kd = 0.0
                    cmd.motor_cmd[i].tau = 0.0
                for action_idx, motor_idx in enumerate(controlled_motor_indices):
                    mapped_idx = mapping[action_idx]
                    cmd.motor_cmd[motor_idx].tau = float(calculated_action[mapped_idx])

            _write_crc_if_present(cmd, crc)
            low_cmd_puber.Write(cmd)

            # Optimized timing control
            loop_end = time.time()
            elapsed = loop_end - loop_start
            sleep_time = simulation_dt - elapsed
            
            if sleep_time > 0:
                time.sleep(sleep_time)
            
            loop_counter += 1
            
            # Optional: Print performance statistics every 1000 loops
            # if loop_counter % 1000 == 0:
            #     avg_loop_time = elapsed
            #     print(f"Loop {loop_counter}: avg time {avg_loop_time:.4f}s, target: {simulation_dt:.4f}s")

        print(_status("policy", f"completed: {loop_counter} loops", "1;32"))
    except Exception as exc:
        sys.stdout.write(f"\r\nError: {exc}\r\n")
        sys.stdout.flush()
    finally:
        # Ensure terminal state is restored even on errors.
        apex_delta_logger.close()
        arm_tracking_logger.close(rl_policy.arm_command_adapter)
        motion_runtime.close()
        rl_policy.stop_keyboard_control()
        _terminate_process_tree(mujoco_proc)
