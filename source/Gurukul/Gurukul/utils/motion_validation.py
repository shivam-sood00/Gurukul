"""Simulator-independent validation of the NPZ motion arrays shared by APEX trackers."""

import numpy as np


def validate_motion_arrays(data, source):
    """Reject corrupt frames and inconsistent shapes before constructing motion tensors."""
    required = (
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise ValueError(f"Motion file '{source}' is missing required arrays: {missing}")
    fps = np.asarray(data["fps"])
    if fps.size != 1 or not np.isfinite(fps).all() or float(fps.reshape(-1)[0]) <= 0:
        raise ValueError(f"Motion file '{source}' must have positive finite fps")
    for key in (
        *required[1:],
        "command_lin_vel_xy",
        "command_ang_vel_z",
        "arm_ee_pos_w",
        "arm_ee_quat_w",
        "gripper_joint_pos",
        "gripper_joint_vel",
        "object_pos_w",
        "object_quat_w",
        "object_attached",
        "reference_foot_contact",
        "reference_airborne",
    ):
        if key in data and not np.isfinite(data[key]).all():
            raise ValueError(f"Motion file '{source}' has nonfinite {key}")
    joints = data["joint_pos"]
    if joints.ndim != 2 or joints.shape[0] < 2 or joints.shape[1] == 0:
        raise ValueError(f"Motion file '{source}' requires at least two joint-position frames")
    if data["joint_vel"].shape != joints.shape:
        raise ValueError(f"Motion file '{source}' joint velocity shape differs from joint positions")
    shape = data["body_pos_w"].shape
    if len(shape) != 3 or shape[0] != len(joints) or shape[1] == 0 or shape[-1] != 3:
        raise ValueError(f"Motion file '{source}' has invalid body position shape {shape}")
    for key in ("body_lin_vel_w", "body_ang_vel_w"):
        if data[key].shape != shape:
            raise ValueError(f"Motion file '{source}' has invalid {key} shape")
    if data["body_quat_w"].shape != (*shape[:2], 4) or np.any(np.linalg.norm(data["body_quat_w"], axis=-1) < 1e-8):
        raise ValueError(f"Motion file '{source}' has invalid body quaternions")
