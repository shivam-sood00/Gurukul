#!/usr/bin/env python3

"""Convert APEX Go2 imitation CSV data into Go2 APEX motion-command NPZ format."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

CANONICAL_LEGS = ("FL", "FR", "RL", "RR")
GO2_JOINT_NAMES = (
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
)
GO2_D1_ARM_JOINT_NAMES = (
    "arm_1_joint",
    "arm_2_joint",
    "arm_3_joint",
    "arm_4_joint",
    "arm_5_joint",
    "arm_6_joint",
)
GO2_HIP_ORIGINS = {
    "FL": (0.1934, 0.0465, 0.0),
    "FR": (0.1934, -0.0465, 0.0),
    "RL": (-0.1934, 0.0465, 0.0),
    "RR": (-0.1934, -0.0465, 0.0),
}
GO2_THIGH_OFFSETS = {
    "FL": (0.0, 0.0955, 0.0),
    "FR": (0.0, -0.0955, 0.0),
    "RL": (0.0, 0.0955, 0.0),
    "RR": (0.0, -0.0955, 0.0),
}
GO2_THIGH_LENGTH = 0.213
GO2_CALF_LENGTH = 0.213


def finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    if values.shape[0] < 2:
        return out
    out[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    out[0] = (values[1] - values[0]) / dt
    out[-1] = (values[-1] - values[-2]) / dt
    return out


def quat_rotate(quat_wxyz: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate vector(s) by quaternion(s).

    quat_wxyz: (N, 4), vec: (N, K, 3) -> output: (N, K, 3)
    """
    q_xyz = quat_wxyz[:, None, 1:4]
    q_w = quat_wxyz[:, None, 0:1]
    t = 2.0 * np.cross(q_xyz, vec)
    return vec + q_w * t + np.cross(q_xyz, t)


def quat_multiply(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Multiply batches of WXYZ quaternions."""
    lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
    rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def d1_link6_quat_w(base_quat_wxyz: np.ndarray, arm_joint_pos: np.ndarray) -> np.ndarray:
    """Compute the D1 Link6 orientation using the hardware joint-axis convention."""
    axes = np.asarray(
        (
            (0.0, 0.0, -1.0),
            (0.0, 1.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (1.0, 0.0, 0.0),
        ),
        dtype=np.float32,
    )
    link_quat = np.zeros((arm_joint_pos.shape[0], 4), dtype=np.float32)
    link_quat[:, 0] = 1.0
    for joint_index, axis in enumerate(axes):
        half_angle = 0.5 * arm_joint_pos[:, joint_index]
        joint_quat = np.empty_like(link_quat)
        joint_quat[:, 0] = np.cos(half_angle)
        joint_quat[:, 1:] = np.sin(half_angle)[:, None] * axis[None, :]
        link_quat = quat_multiply(link_quat, joint_quat)
    result = quat_multiply(base_quat_wxyz, link_quat)
    return (result / np.linalg.norm(result, axis=-1, keepdims=True).clip(min=1.0e-8)).astype(np.float32)


def rotate_x(vec: np.ndarray, angle: np.ndarray) -> np.ndarray:
    out = np.empty_like(vec, dtype=np.float32)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    out[:, 0] = vec[:, 0]
    out[:, 1] = cos_angle * vec[:, 1] - sin_angle * vec[:, 2]
    out[:, 2] = sin_angle * vec[:, 1] + cos_angle * vec[:, 2]
    return out


def rotate_y(vec: np.ndarray, angle: np.ndarray) -> np.ndarray:
    out = np.empty_like(vec, dtype=np.float32)
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    out[:, 0] = cos_angle * vec[:, 0] + sin_angle * vec[:, 2]
    out[:, 1] = vec[:, 1]
    out[:, 2] = -sin_angle * vec[:, 0] + cos_angle * vec[:, 2]
    return out


def estimate_go2_feet_pos_b_from_joints(joint_pos: np.ndarray) -> np.ndarray:
    """Estimate Go2 foot positions in the base frame from FL,FR,RL,RR hip/thigh/calf joints."""
    feet_pos_b = np.zeros((joint_pos.shape[0], 4, 3), dtype=np.float32)
    thigh_link = np.tile(np.array([0.0, 0.0, -GO2_THIGH_LENGTH], dtype=np.float32), (joint_pos.shape[0], 1))
    calf_link = np.tile(np.array([0.0, 0.0, -GO2_CALF_LENGTH], dtype=np.float32), (joint_pos.shape[0], 1))

    for leg_idx, leg in enumerate(CANONICAL_LEGS):
        hip = joint_pos[:, 3 * leg_idx]
        thigh = joint_pos[:, 3 * leg_idx + 1]
        calf = joint_pos[:, 3 * leg_idx + 2]

        hip_origin = np.asarray(GO2_HIP_ORIGINS[leg], dtype=np.float32)[None, :]
        thigh_offset = np.tile(np.asarray(GO2_THIGH_OFFSETS[leg], dtype=np.float32), (joint_pos.shape[0], 1))

        thigh_joint_pos = hip_origin + rotate_x(thigh_offset, hip)
        calf_joint_offset = rotate_x(rotate_y(thigh_link, thigh), hip)
        foot_offset = rotate_x(rotate_y(rotate_y(calf_link, calf), thigh), hip)
        feet_pos_b[:, leg_idx, :] = thigh_joint_pos + calf_joint_offset + foot_offset

    return feet_pos_b


def _is_numeric(token: str) -> bool:
    try:
        float(token)
    except ValueError:
        return False
    return True


def load_csv_matrix(path: Path) -> np.ndarray:
    """Read every numeric frame; never silently drop rows and change motion timing."""
    rows = []
    width = None
    for line_number, line in enumerate(path.read_text().splitlines(), 1):
        tokens = [token.strip() for token in line.split(",")]
        if not line.strip():
            continue
        try:
            row = [float(token) for token in tokens]
        except ValueError as exc:
            if not rows and width is None and all(token and not _is_numeric(token) for token in tokens):
                width = -1  # Allow one header, independently of its claimed width.
                continue
            raise ValueError(f"{path}:{line_number}: nonnumeric motion row") from exc
        if width is None or width == -1:
            width = len(row)
        if len(row) != width:
            raise ValueError(f"{path}:{line_number}: expected {width} columns, got {len(row)}")
        rows.append(row)
    if len(rows) < 2:
        raise ValueError(f"{path}: at least two numeric motion frames are required")
    return np.asarray(rows, dtype=np.float32)


def load_csv_header(path: Path) -> list[str] | None:
    first = next((line for line in path.read_text().splitlines() if line.strip()), "")
    if not first:
        raise ValueError(f"CSV file is empty: {path}")
    tokens = [token.strip() for token in first.split(",")]
    try:
        [float(token) for token in tokens]
    except ValueError:
        return tokens
    return None


def normalize_quaternion(quaternion):
    norm = np.linalg.norm(quaternion, axis=-1, keepdims=True)
    if not np.isfinite(quaternion).all() or np.any(norm < 1e-8):
        raise ValueError("Motion contains a nonfinite or zero quaternion")
    return quaternion / norm


def feet_to_world(feet, base_pos, base_quat, frame):
    if frame == "world":
        return feet.copy()
    rotation = base_quat
    if frame == "yaw":
        w, x, y, z = np.moveaxis(base_quat, -1, 0)
        yaw = np.arctan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        rotation = np.zeros_like(base_quat)
        rotation[:, 0], rotation[:, 3] = np.cos(yaw / 2), np.sin(yaw / 2)
    return quat_rotate(rotation, feet) + base_pos[:, None, :]


def validate_go2_feet(base_pos, feet_pos_w, max_distance, source, frame_start):
    # A conservative root-to-foot bound: hip origin + hip offset + both leg
    # segments + retargeting margin. Override explicitly for another morphology.
    distance = np.linalg.norm(feet_pos_w - base_pos[:, None, :], axis=-1)
    if not np.isfinite(distance).all() or np.any(distance > max_distance):
        frame, foot = np.unravel_index(np.argmax(distance), distance.shape)
        raise ValueError(
            f"{source}: frame {frame_start + frame}, {CANONICAL_LEGS[foot]} foot is "
            f"{distance[frame, foot]:.3f} m from the root (limit {max_distance:.3f} m). "
            "Check coordinate frames or select a continuous --frame-range START STOP; reset artifacts must be excluded."
        )


def parse_csv_leg_order(order_str: str) -> list[str]:
    order = [token.strip().upper() for token in order_str.split(",") if token.strip()]
    if len(order) != 4:
        raise ValueError(f"Expected 4 leg labels in --csv-leg-order, got {len(order)} from '{order_str}'.")
    if set(order) != set(CANONICAL_LEGS):
        raise ValueError(f"--csv-leg-order must be a permutation of {CANONICAL_LEGS}, got {order}.")
    return order


def convert(
    input_csv: Path,
    output_npz: Path,
    fps: float,
    csv_leg_order: list[str],
    ground_align_foot_height: bool = False,
    *,
    velocity_frame: str = "auto",
    feet_frame: str = "auto",
    frame_range: tuple[int, int] | None = None,
    max_foot_distance: float = 0.9,
) -> None:
    if not np.isfinite(fps) or fps <= 0 or not np.isfinite(max_foot_distance) or max_foot_distance <= 0:
        raise ValueError("FPS and maximum foot distance must be positive and finite")
    if velocity_frame not in ("auto", "body", "world") or feet_frame not in ("auto", "body", "yaw", "world"):
        raise ValueError("Invalid velocity or foot coordinate frame")
    parse_csv_leg_order(",".join(csv_leg_order))
    data = load_csv_matrix(input_csv)
    frame_start, frame_stop = frame_range if frame_range is not None else (0, len(data))
    if not 0 <= frame_start < frame_stop <= len(data) or frame_stop - frame_start < 2:
        raise ValueError(f"Invalid half-open frame range {(frame_start, frame_stop)} for {len(data)} frames")
    data = data[frame_start:frame_stop]
    if not np.isfinite(data).all():
        raise ValueError("Selected motion frames contain nonfinite values")
    if data.shape[1] < 40:
        raise ValueError(f"Expected at least 40 columns in APEX CSV, got {data.shape[1]} from: {input_csv}")

    dt = 1.0 / fps
    header = load_csv_header(input_csv)
    is_d1 = header is not None and all(name in header for name in GO2_D1_ARM_JOINT_NAMES)
    if velocity_frame == "auto":
        velocity_frame = "world" if is_d1 or "_STMR_" in input_csv.name else "body"
    if feet_frame == "auto":
        feet_frame = "body" if is_d1 else "yaw"
    metadata = {
        "motion_conversion_version": np.int32(2),
        "source_frame_range": np.asarray([frame_start, frame_stop]),
        "source_velocity_frame": np.asarray(velocity_frame),
        "source_feet_frame": np.asarray(feet_frame),
    }
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    if is_d1:
        convert_header_go2_d1(
            data,
            header,
            input_csv,
            output_npz,
            fps,
            dt,
            csv_leg_order,
            ground_align_foot_height=ground_align_foot_height,
            velocity_frame=velocity_frame,
            feet_frame=feet_frame,
            max_foot_distance=max_foot_distance,
            metadata=metadata,
        )
        return

    leg_to_csv_idx = {leg: i for i, leg in enumerate(csv_leg_order)}

    joint_pos_csv = data[:, 6:18].reshape(-1, 4, 3)
    joint_pos = np.zeros((data.shape[0], 12), dtype=np.float32)
    for canonical_leg_idx, leg in enumerate(CANONICAL_LEGS):
        src_leg_idx = leg_to_csv_idx[leg]
        joint_pos[:, 3 * canonical_leg_idx : 3 * canonical_leg_idx + 3] = joint_pos_csv[:, src_leg_idx, :]

    if data.shape[1] >= 64:
        joint_vel_csv = data[:, 52:64].reshape(-1, 4, 3)
        joint_vel = np.zeros_like(joint_pos, dtype=np.float32)
        for canonical_leg_idx, leg in enumerate(CANONICAL_LEGS):
            src_leg_idx = leg_to_csv_idx[leg]
            joint_vel[:, 3 * canonical_leg_idx : 3 * canonical_leg_idx + 3] = joint_vel_csv[:, src_leg_idx, :]
    else:
        joint_vel = finite_difference(joint_pos, dt)

    base_lin_vel = data[:, 0:3]
    base_ang_vel = data[:, 3:6]
    base_pos = np.concatenate([data[:, 34:36], data[:, 21:22]], axis=1)
    base_quat_xyzw = data[:, 36:40]
    # Isaac Lab quaternions are wxyz.
    base_quat = np.concatenate([base_quat_xyzw[:, 3:4], base_quat_xyzw[:, 0:3]], axis=1)
    base_quat = normalize_quaternion(base_quat)
    if velocity_frame == "body":
        base_lin_vel = quat_rotate(base_quat, base_lin_vel[:, None, :])[:, 0]
        base_ang_vel = quat_rotate(base_quat, base_ang_vel[:, None, :])[:, 0]
    command_lin_vel_xy = data[:, 18:20]
    command_ang_vel_z = data[:, 20:21]

    feet_already_canonical = False
    feet_pos_w = _header_feet_world_columns(data, header) if header is not None else None
    if feet_pos_w is not None:
        feet_frame = "world"
    if feet_pos_w is None and data.shape[1] >= 52:
        # Legacy width-based APEX files with explicit foot world-frame coordinates.
        feet_pos_w = data[:, 40:52].reshape(-1, 4, 3)
        feet_frame = "world"
    elif feet_pos_w is None:
        # Convert source foot offsets using their explicit coordinate convention.
        feet_pos_b = data[:, 22:34].reshape(-1, 4, 3)
        if np.allclose(feet_pos_b, 0.0, atol=1.0e-7):
            print(f"[WARN] {input_csv.name}: foot position columns are all zero; estimating Go2 feet with FK.")
            feet_pos_b = estimate_go2_feet_pos_b_from_joints(joint_pos)
            feet_frame = "body"  # Kinematic fallback is explicitly body-relative.
            feet_already_canonical = True
        feet_pos_w = feet_to_world(feet_pos_b, base_pos, base_quat, feet_frame)

    if not feet_already_canonical:
        feet_pos_w = feet_pos_w[:, [leg_to_csv_idx[leg] for leg in CANONICAL_LEGS]]
    metadata["source_feet_frame"] = np.asarray(feet_frame)
    validate_go2_feet(base_pos, feet_pos_w, max_foot_distance, input_csv, int(metadata["source_frame_range"][0]))

    if ground_align_foot_height:
        foot_height_offset = feet_pos_w[:, :, 2].min(axis=1)
        base_pos = base_pos.copy()
        feet_pos_w = feet_pos_w.copy()
        base_lin_vel = base_lin_vel.copy()
        base_pos[:, 2] -= foot_height_offset
        feet_pos_w[:, :, 2] -= foot_height_offset[:, None]
        base_lin_vel[:, 2] = finite_difference(base_pos[:, 2:3], dt)[:, 0]

    feet_lin_vel_w = finite_difference(feet_pos_w, dt)
    feet_quat_w = np.zeros((data.shape[0], 4, 4), dtype=np.float32)
    feet_quat_w[..., 0] = 1.0
    feet_ang_vel_w = np.zeros((data.shape[0], 4, 3), dtype=np.float32)

    body_pos_w = np.concatenate([base_pos[:, None, :], feet_pos_w], axis=1)
    body_quat_w = np.concatenate([base_quat[:, None, :], feet_quat_w], axis=1)
    body_lin_vel_w = np.concatenate([base_lin_vel[:, None, :], feet_lin_vel_w], axis=1)
    body_ang_vel_w = np.concatenate([base_ang_vel[:, None, :], feet_ang_vel_w], axis=1)

    np.savez(
        output_npz,
        **metadata,
        fps=np.float32(fps),
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        command_lin_vel_xy=command_lin_vel_xy.astype(np.float32),
        command_ang_vel_z=command_ang_vel_z.astype(np.float32),
        body_pos_w=body_pos_w.astype(np.float32),
        body_quat_w=body_quat_w.astype(np.float32),
        body_lin_vel_w=body_lin_vel_w.astype(np.float32),
        body_ang_vel_w=body_ang_vel_w.astype(np.float32),
        body_names=np.array(["base", "FL_foot", "FR_foot", "RL_foot", "RR_foot"], dtype=np.str_),
        joint_names=np.array(
            GO2_JOINT_NAMES,
            dtype=np.str_,
        ),
    )


def _header_columns(data: np.ndarray, header: list[str], names: list[str]) -> np.ndarray:
    name_to_index = {name: index for index, name in enumerate(header)}
    missing = [name for name in names if name not in name_to_index]
    if missing:
        raise ValueError(f"CSV header is missing required columns: {missing}")
    return data[:, [name_to_index[name] for name in names]]


def _has_header_columns(header: list[str], names: list[str]) -> bool:
    return all(name in header for name in names)


def _header_feet_world_columns(data: np.ndarray, header: list[str]) -> np.ndarray | None:
    world_names = [
        coord for foot_id in range(1, 5) for coord in (f"e{foot_id}x_wf", f"e{foot_id}y_wf", f"e{foot_id}z_wf")
    ]
    if not _has_header_columns(header, world_names):
        return None
    return np.stack(
        [
            _header_columns(data, header, [f"e{foot_id}x_wf", f"e{foot_id}y_wf", f"e{foot_id}z_wf"])
            for foot_id in range(1, 5)
        ],
        axis=1,
    )


def convert_header_go2_d1(
    data: np.ndarray,
    header: list[str],
    input_csv: Path,
    output_npz: Path,
    fps: float,
    dt: float,
    csv_leg_order: list[str],
    ground_align_foot_height: bool = False,
    *,
    velocity_frame="world",
    feet_frame="body",
    max_foot_distance=0.9,
    metadata=None,
) -> None:
    """Convert header-based Go2+D1 CSVs with leg and arm joint columns."""
    metadata = metadata or {"source_frame_range": np.asarray([0, len(data)])}
    leg_to_csv_idx = {leg: i for i, leg in enumerate(csv_leg_order)}
    csv_leg_joint_groups = [
        ("base1", "shoulder1", "elbow1"),
        ("base2", "shoulder2", "elbow2"),
        ("base3", "shoulder3", "elbow3"),
        ("base4", "shoulder4", "elbow4"),
    ]
    joint_pos = np.zeros((data.shape[0], len(GO2_JOINT_NAMES) + len(GO2_D1_ARM_JOINT_NAMES)), dtype=np.float32)
    for canonical_leg_idx, leg in enumerate(CANONICAL_LEGS):
        src_leg_idx = leg_to_csv_idx[leg]
        joint_pos[:, 3 * canonical_leg_idx : 3 * canonical_leg_idx + 3] = _header_columns(
            data, header, list(csv_leg_joint_groups[src_leg_idx])
        )
    joint_pos[:, len(GO2_JOINT_NAMES) :] = _header_columns(data, header, list(GO2_D1_ARM_JOINT_NAMES))
    joint_vel = finite_difference(joint_pos, dt)

    base_lin_vel = _header_columns(data, header, ["vx", "vy", "vz"])
    base_ang_vel = _header_columns(data, header, ["wx", "wy", "wz"])
    base_pos = np.concatenate(
        [_header_columns(data, header, ["com_x", "com_y"]), _header_columns(data, header, ["height"])],
        axis=1,
    )
    base_quat_xyzw = _header_columns(data, header, ["quat_x", "quat_y", "quat_z", "quat_w"])
    base_quat = np.concatenate([base_quat_xyzw[:, 3:4], base_quat_xyzw[:, 0:3]], axis=1)
    base_quat = normalize_quaternion(base_quat)
    if velocity_frame == "body":
        base_lin_vel = quat_rotate(base_quat, base_lin_vel[:, None, :])[:, 0]
        base_ang_vel = quat_rotate(base_quat, base_ang_vel[:, None, :])[:, 0]
    command_lin_vel_xy = _header_columns(data, header, ["com_vx", "com_vy"])
    command_ang_vel_z = _header_columns(data, header, ["com_wz"])

    feet_already_canonical = False
    feet_pos_w = _header_feet_world_columns(data, header)
    if feet_pos_w is not None:
        feet_frame = "world"
    if feet_pos_w is None:
        feet_pos_b = np.stack(
            [
                _header_columns(data, header, [f"e{foot_id}x", f"e{foot_id}y", f"e{foot_id}z"])
                for foot_id in range(1, 5)
            ],
            axis=1,
        )
        if np.allclose(feet_pos_b, 0.0, atol=1.0e-7):
            print(f"[WARN] {input_csv.name}: foot position columns are all zero; estimating Go2 feet with FK.")
            feet_pos_b = estimate_go2_feet_pos_b_from_joints(joint_pos[:, : len(GO2_JOINT_NAMES)])
            feet_frame = "body"  # Kinematic fallback is explicitly body-relative.
            feet_already_canonical = True
        feet_pos_w = feet_to_world(feet_pos_b, base_pos, base_quat, feet_frame)
    arm_ee_pos_b = _header_columns(data, header, ["arm_eex", "arm_eey", "arm_eez"])
    arm_ee_pos_w = quat_rotate(base_quat, arm_ee_pos_b[:, None, :])[:, 0, :] + base_pos
    arm_ee_quat_w = d1_link6_quat_w(
        base_quat,
        joint_pos[:, len(GO2_JOINT_NAMES) :],
    )

    if not feet_already_canonical:
        feet_pos_w = feet_pos_w[:, [leg_to_csv_idx[leg] for leg in CANONICAL_LEGS]]
    metadata["source_feet_frame"] = np.asarray(feet_frame)
    validate_go2_feet(base_pos, feet_pos_w, max_foot_distance, input_csv, int(metadata["source_frame_range"][0]))

    if ground_align_foot_height:
        foot_height_offset = feet_pos_w[:, :, 2].min(axis=1)
        base_pos = base_pos.copy()
        feet_pos_w = feet_pos_w.copy()
        arm_ee_pos_w = arm_ee_pos_w.copy()
        base_lin_vel = base_lin_vel.copy()
        base_pos[:, 2] -= foot_height_offset
        feet_pos_w[:, :, 2] -= foot_height_offset[:, None]
        arm_ee_pos_w[:, 2] -= foot_height_offset
        base_lin_vel[:, 2] = finite_difference(base_pos[:, 2:3], dt)[:, 0]

    feet_lin_vel_w = finite_difference(feet_pos_w, dt)
    feet_quat_w = np.zeros((data.shape[0], 4, 4), dtype=np.float32)
    feet_quat_w[..., 0] = 1.0
    feet_ang_vel_w = np.zeros((data.shape[0], 4, 3), dtype=np.float32)

    body_pos_w = np.concatenate([base_pos[:, None, :], feet_pos_w], axis=1)
    body_quat_w = np.concatenate([base_quat[:, None, :], feet_quat_w], axis=1)
    body_lin_vel_w = np.concatenate([base_lin_vel[:, None, :], feet_lin_vel_w], axis=1)
    body_ang_vel_w = np.concatenate([base_ang_vel[:, None, :], feet_ang_vel_w], axis=1)

    np.savez(
        output_npz,
        **metadata,
        fps=np.float32(fps),
        joint_pos=joint_pos.astype(np.float32),
        joint_vel=joint_vel.astype(np.float32),
        command_lin_vel_xy=command_lin_vel_xy.astype(np.float32),
        command_ang_vel_z=command_ang_vel_z.astype(np.float32),
        body_pos_w=body_pos_w.astype(np.float32),
        body_quat_w=body_quat_w.astype(np.float32),
        body_lin_vel_w=body_lin_vel_w.astype(np.float32),
        body_ang_vel_w=body_ang_vel_w.astype(np.float32),
        arm_ee_pos_w=arm_ee_pos_w.astype(np.float32),
        arm_ee_quat_w=arm_ee_quat_w,
        arm_ee_orientation_frame=np.asarray("Link6", dtype=np.str_),
        body_names=np.array(["base", "FL_foot", "FR_foot", "RL_foot", "RR_foot"], dtype=np.str_),
        joint_names=np.array(GO2_JOINT_NAMES + GO2_D1_ARM_JOINT_NAMES, dtype=np.str_),
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="Path to APEX Go2 CSV file.")
    parser.add_argument("--output", type=Path, required=True, help="Output NPZ path.")
    parser.add_argument("--fps", type=float, default=50.0, help="CSV sampling frequency in Hz.")
    parser.add_argument(
        "--csv-leg-order",
        type=str,
        default="FL,FR,RL,RR",
        help=("Leg labels for CSV groups base1..base4, comma-separated. Example: FL,FR,RL,RR or FR,FL,RR,RL."),
    )
    parser.add_argument(
        "--ground-align-foot-height",
        action="store_true",
        help="Subtract the per-frame minimum foot world height from base and foot positions.",
    )
    parser.add_argument(
        "--velocity-frame",
        choices=("auto", "body", "world"),
        default="auto",
        help="auto: body for Go2 CSVs; world for STMR and named Go2+D1 CSVs.",
    )
    parser.add_argument(
        "--feet-frame",
        choices=("auto", "yaw", "body", "world"),
        default="auto",
        help="Frame of relative foot columns; explicit world-foot columns take precedence.",
    )
    parser.add_argument(
        "--frame-range",
        type=int,
        nargs=2,
        metavar=("START", "STOP"),
        help="Half-open range of numeric frames, selected before differentiation.",
    )
    parser.add_argument("--max-foot-distance", type=float, default=0.9, help="Maximum root-to-foot distance in metres.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert(
        args.input,
        args.output,
        args.fps,
        parse_csv_leg_order(args.csv_leg_order),
        ground_align_foot_height=args.ground_align_foot_height,
        velocity_frame=args.velocity_frame,
        feet_frame=args.feet_frame,
        frame_range=args.frame_range,
        max_foot_distance=args.max_foot_distance,
    )
    print(f"Wrote motion file: {args.output}")


if __name__ == "__main__":
    main()
