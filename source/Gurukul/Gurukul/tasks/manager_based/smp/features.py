# SPDX-License-Identifier: Apache-2.0

"""Pure-Torch kinematic feature transforms for SMP motion windows.

Quaternions use scalar-first ``(w, x, y, z)`` order.  The SMP rotation
representation stores matrix columns zero and two (the local x- and z-axes),
matching MimicKit's tangent/normal representation.  All helpers accept
arbitrary broadcastable leading dimensions and have no Isaac Lab dependency.
"""

from __future__ import annotations

import torch

from .profiles import SmpRobotProfile, get_profile


def _require_tensor_last_dim(tensor: torch.Tensor, size: int, name: str) -> None:
    if not isinstance(tensor, torch.Tensor):
        raise TypeError(f"{name} must be a torch.Tensor, got {type(tensor).__name__}.")
    if tensor.ndim < 1 or tensor.shape[-1] != size:
        raise ValueError(f"{name} must end in dimension {size}; got shape {tuple(tensor.shape)}.")
    if not tensor.is_floating_point():
        raise TypeError(f"{name} must have a floating-point dtype, got {tensor.dtype}.")


def _identity_quaternion_like(quaternion: torch.Tensor) -> torch.Tensor:
    identity = torch.zeros_like(quaternion)
    identity[..., 0] = 1.0
    return identity


def _squared_norm_floor(tensor: torch.Tensor, eps: float) -> float:
    return max(eps * eps, torch.finfo(tensor.dtype).tiny)


def _require_compatible(lhs: torch.Tensor, rhs: torch.Tensor, lhs_name: str, rhs_name: str) -> None:
    if lhs.device != rhs.device:
        raise ValueError(f"{lhs_name} and {rhs_name} must be on the same device.")
    if lhs.dtype != rhs.dtype:
        raise ValueError(f"{lhs_name} and {rhs_name} must use the same dtype.")


def normalize_quaternion(quaternion: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Return unit wxyz quaternions, mapping a zero quaternion to identity."""
    _require_tensor_last_dim(quaternion, 4, "quaternion")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}.")
    norm_squared = torch.sum(quaternion.square(), dim=-1, keepdim=True)
    normalized = quaternion * torch.rsqrt(norm_squared.clamp_min(_squared_norm_floor(quaternion, eps)))
    return torch.where(norm_squared <= eps * eps, _identity_quaternion_like(quaternion), normalized)


# Both word orders are useful at call sites; keep one implementation.
quaternion_normalize = normalize_quaternion


def quaternion_conjugate(quaternion: torch.Tensor) -> torch.Tensor:
    """Return the conjugate of wxyz quaternions."""
    _require_tensor_last_dim(quaternion, 4, "quaternion")
    return torch.cat((quaternion[..., :1], -quaternion[..., 1:]), dim=-1)


def quaternion_inverse(quaternion: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Return the multiplicative inverse, mapping a zero quaternion to identity."""
    _require_tensor_last_dim(quaternion, 4, "quaternion")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}.")
    norm_squared = torch.sum(quaternion.square(), dim=-1, keepdim=True)
    inverse = quaternion_conjugate(quaternion) / norm_squared.clamp_min(_squared_norm_floor(quaternion, eps))
    return torch.where(norm_squared <= eps * eps, _identity_quaternion_like(quaternion), inverse)


def quaternion_multiply(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Hamilton product of broadcastable wxyz quaternion tensors."""
    _require_tensor_last_dim(lhs, 4, "lhs")
    _require_tensor_last_dim(rhs, 4, "rhs")
    _require_compatible(lhs, rhs, "lhs", "rhs")
    try:
        leading_shape = torch.broadcast_shapes(lhs.shape[:-1], rhs.shape[:-1])
    except RuntimeError as exc:
        raise ValueError(
            f"Quaternion leading shapes are not broadcastable: {tuple(lhs.shape)} and {tuple(rhs.shape)}."
        ) from exc
    lhs = lhs.expand(*leading_shape, 4)
    rhs = rhs.expand(*leading_shape, 4)
    lhs_w, lhs_xyz = lhs[..., :1], lhs[..., 1:]
    rhs_w, rhs_xyz = rhs[..., :1], rhs[..., 1:]
    scalar = lhs_w * rhs_w - torch.sum(lhs_xyz * rhs_xyz, dim=-1, keepdim=True)
    vector = lhs_w * rhs_xyz + rhs_w * lhs_xyz + torch.cross(lhs_xyz, rhs_xyz, dim=-1)
    return torch.cat((scalar, vector), dim=-1)


quaternion_mul = quaternion_multiply


def quaternion_apply(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate 3-D vectors by broadcastable wxyz quaternions."""
    _require_tensor_last_dim(quaternion, 4, "quaternion")
    _require_tensor_last_dim(vector, 3, "vector")
    _require_compatible(quaternion, vector, "quaternion", "vector")
    try:
        leading_shape = torch.broadcast_shapes(quaternion.shape[:-1], vector.shape[:-1])
    except RuntimeError as exc:
        raise ValueError(
            f"Quaternion/vector leading shapes are not broadcastable: "
            f"{tuple(quaternion.shape)} and {tuple(vector.shape)}."
        ) from exc
    quaternion = normalize_quaternion(quaternion).expand(*leading_shape, 4)
    vector = vector.expand(*leading_shape, 3)
    xyz = quaternion[..., 1:]
    first_cross = torch.cross(xyz, vector, dim=-1)
    return vector + 2.0 * (quaternion[..., :1] * first_cross + torch.cross(xyz, first_cross, dim=-1))


def quaternion_apply_inverse(quaternion: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate 3-D vectors by the inverse of broadcastable wxyz quaternions."""
    return quaternion_apply(quaternion_inverse(quaternion), vector)


def quaternion_to_yaw(quaternion: torch.Tensor) -> torch.Tensor:
    """Return the world-z yaw angle of wxyz quaternions in radians."""
    quaternion = normalize_quaternion(quaternion)
    w, x, y, z = quaternion.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))


def yaw_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """Project wxyz quaternions onto their world-z yaw component."""
    yaw = quaternion_to_yaw(quaternion)
    half_yaw = 0.5 * yaw
    zeros = torch.zeros_like(half_yaw)
    return torch.stack((torch.cos(half_yaw), zeros, zeros, torch.sin(half_yaw)), dim=-1)


def inverse_yaw_quaternion(quaternion: torch.Tensor) -> torch.Tensor:
    """Return the inverse world-z yaw component of wxyz quaternions."""
    return quaternion_conjugate(yaw_quaternion(quaternion))


yaw_quat = yaw_quaternion


def quaternion_to_rotation_6d(quaternion: torch.Tensor) -> torch.Tensor:
    """Encode rotations as concatenated matrix columns zero and two."""
    quaternion = normalize_quaternion(quaternion)
    x_axis = torch.zeros(*quaternion.shape[:-1], 3, dtype=quaternion.dtype, device=quaternion.device)
    z_axis = torch.zeros_like(x_axis)
    x_axis[..., 0] = 1.0
    z_axis[..., 2] = 1.0
    return torch.cat((quaternion_apply(quaternion, x_axis), quaternion_apply(quaternion, z_axis)), dim=-1)


def _normalize_vector(vector: torch.Tensor, fallback: torch.Tensor, eps: float) -> torch.Tensor:
    norm_squared = torch.sum(vector.square(), dim=-1, keepdim=True)
    normalized = vector * torch.rsqrt(norm_squared.clamp_min(_squared_norm_floor(vector, eps)))
    return torch.where(norm_squared <= eps * eps, fallback, normalized)


def rotation_6d_to_quaternion(rotation_6d: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
    """Decode matrix-column ``[x_axis, z_axis]`` rotations to wxyz quaternions.

    Gram--Schmidt projection makes noisy diffusion outputs valid rotations.
    Degenerate axes receive deterministic orthogonal fallbacks.
    """
    _require_tensor_last_dim(rotation_6d, 6, "rotation_6d")
    if eps <= 0.0:
        raise ValueError(f"eps must be positive, got {eps}.")

    output_dtype = rotation_6d.dtype
    if rotation_6d.dtype in (torch.float16, torch.bfloat16):
        rotation_6d = rotation_6d.float()

    x_raw, z_raw = rotation_6d[..., :3], rotation_6d[..., 3:]
    default_x = torch.zeros_like(x_raw)
    default_x[..., 0] = 1.0
    x_axis = _normalize_vector(x_raw, default_x, eps)

    z_orthogonal = z_raw - torch.sum(z_raw * x_axis, dim=-1, keepdim=True) * x_axis
    # Pick the coordinate axis least aligned with x, then project it into the
    # plane normal to x. This is a stable fallback when the supplied z axis is
    # zero or collinear with x.
    fallback_seed = torch.zeros_like(x_axis)
    use_y = x_axis[..., 2].abs() > 0.9
    fallback_seed[..., 1] = use_y.to(fallback_seed.dtype)
    fallback_seed[..., 2] = (~use_y).to(fallback_seed.dtype)
    fallback_z = fallback_seed - torch.sum(fallback_seed * x_axis, dim=-1, keepdim=True) * x_axis
    fallback_z = _normalize_vector(fallback_z, torch.zeros_like(fallback_z), eps)
    z_axis = _normalize_vector(z_orthogonal, fallback_z, eps)
    y_axis = torch.cross(z_axis, x_axis, dim=-1)

    # Matrix columns are x, y, z. Form all four quaternion candidates and
    # select the one with the best-conditioned denominator.
    m00, m10, m20 = x_axis.unbind(dim=-1)
    m01, m11, m21 = y_axis.unbind(dim=-1)
    m02, m12, m22 = z_axis.unbind(dim=-1)
    one = torch.ones_like(m00)
    quaternion_magnitudes = torch.sqrt(
        torch.clamp(
            torch.stack(
                (
                    one + m00 + m11 + m22,
                    one + m00 - m11 - m22,
                    one - m00 + m11 - m22,
                    one - m00 - m11 + m22,
                ),
                dim=-1,
            ),
            min=0.0,
        )
    )
    candidates = torch.stack(
        (
            torch.stack((quaternion_magnitudes[..., 0].square(), m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, quaternion_magnitudes[..., 1].square(), m10 + m01, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m10 + m01, quaternion_magnitudes[..., 2].square(), m12 + m21), dim=-1),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, quaternion_magnitudes[..., 3].square()), dim=-1),
        ),
        dim=-2,
    )
    denominator = 2.0 * quaternion_magnitudes.clamp_min(0.1).unsqueeze(-1)
    candidates = candidates / denominator
    best = torch.argmax(quaternion_magnitudes, dim=-1)
    gather_index = best[..., None, None].expand(*best.shape, 1, 4)
    quaternion = torch.gather(candidates, dim=-2, index=gather_index).squeeze(-2)
    return normalize_quaternion(quaternion, eps=eps).to(dtype=output_dtype)


def _validate_motion_inputs(
    root_pos: torch.Tensor,
    root_quat_wxyz: torch.Tensor,
    joint_pos: torch.Tensor,
    key_body_pos: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
) -> tuple[int, int]:
    _require_tensor_last_dim(root_pos, 3, "root_pos")
    _require_tensor_last_dim(root_quat_wxyz, 4, "root_quat_wxyz")
    _require_tensor_last_dim(root_lin_vel, 3, "root_lin_vel")
    _require_tensor_last_dim(root_ang_vel, 3, "root_ang_vel")
    if root_pos.ndim < 2:
        raise ValueError("Motion tensors require a time dimension before their channel dimension.")
    if not isinstance(joint_pos, torch.Tensor):
        raise TypeError(f"joint_pos must be a torch.Tensor, got {type(joint_pos).__name__}.")
    if joint_pos.ndim != root_pos.ndim or not joint_pos.is_floating_point():
        raise ValueError(
            f"joint_pos must be floating-point with shape [..., time, joints]; got {tuple(joint_pos.shape)}."
        )
    if not isinstance(key_body_pos, torch.Tensor):
        raise TypeError(f"key_body_pos must be a torch.Tensor, got {type(key_body_pos).__name__}.")
    if key_body_pos.ndim != root_pos.ndim + 1 or key_body_pos.shape[-1] != 3:
        raise ValueError(f"key_body_pos must have shape [..., time, key_bodies, 3]; got {tuple(key_body_pos.shape)}.")
    if not key_body_pos.is_floating_point():
        raise TypeError(f"key_body_pos must have a floating-point dtype, got {key_body_pos.dtype}.")

    time_shape = root_pos.shape[:-1]
    expected_shapes = {
        "root_quat_wxyz": (*time_shape, 4),
        "joint_pos": (*time_shape, joint_pos.shape[-1]),
        "key_body_pos": (*time_shape, key_body_pos.shape[-2], 3),
        "root_lin_vel": (*time_shape, 3),
        "root_ang_vel": (*time_shape, 3),
    }
    actual = {
        "root_quat_wxyz": root_quat_wxyz.shape,
        "joint_pos": joint_pos.shape,
        "key_body_pos": key_body_pos.shape,
        "root_lin_vel": root_lin_vel.shape,
        "root_ang_vel": root_ang_vel.shape,
    }
    for name, expected in expected_shapes.items():
        if tuple(actual[name]) != tuple(expected):
            raise ValueError(f"{name} must have shape {tuple(expected)}; got {tuple(actual[name])}.")
    if root_pos.shape[-2] < 1:
        raise ValueError("Motion windows must contain at least one frame.")
    if joint_pos.shape[-1] < 1:
        raise ValueError("Motion windows must contain at least one joint channel.")
    if key_body_pos.shape[-2] < 1:
        raise ValueError("Motion windows must contain at least one key body.")

    tensors = (root_quat_wxyz, joint_pos, key_body_pos, root_lin_vel, root_ang_vel)
    for tensor in tensors:
        if tensor.device != root_pos.device:
            raise ValueError("All motion tensors must be on the same device.")
        if tensor.dtype != root_pos.dtype:
            raise ValueError("All motion tensors must use the same dtype.")
    return int(joint_pos.shape[-1]), int(key_body_pos.shape[-2])


def canonicalize_motion_features(
    root_pos: torch.Tensor,
    root_quat_wxyz: torch.Tensor,
    joint_pos: torch.Tensor,
    key_body_pos: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
) -> torch.Tensor:
    """Build canonical oldest-to-newest SMP windows.

    Horizontal positions are relative to the last root frame and all vectors
    are expressed in that frame's inverse yaw. Root height stays absolute.
    Key-body positions remain relative to their root at each frame.
    """
    _validate_motion_inputs(
        root_pos,
        root_quat_wxyz,
        joint_pos,
        key_body_pos,
        root_lin_vel,
        root_ang_vel,
    )
    root_quat_wxyz = normalize_quaternion(root_quat_wxyz)
    anchor_inverse_yaw = inverse_yaw_quaternion(root_quat_wxyz[..., -1, :])

    frame_yaw = anchor_inverse_yaw.unsqueeze(-2)
    root_offset = root_pos - root_pos[..., -1:, :]
    canonical_root_pos = quaternion_apply(frame_yaw, root_offset)
    canonical_root_pos = canonical_root_pos.clone()
    canonical_root_pos[..., 2] = root_pos[..., 2]

    canonical_root_quat = quaternion_multiply(frame_yaw, root_quat_wxyz)
    canonical_root_rotation = quaternion_to_rotation_6d(canonical_root_quat)

    key_offset = key_body_pos - root_pos.unsqueeze(-2)
    canonical_key_pos = quaternion_apply(frame_yaw.unsqueeze(-2), key_offset)
    canonical_lin_vel = quaternion_apply(frame_yaw, root_lin_vel)
    canonical_ang_vel = quaternion_apply(frame_yaw, root_ang_vel)

    return torch.cat(
        (
            canonical_root_pos,
            canonical_root_rotation,
            joint_pos,
            canonical_key_pos.flatten(start_dim=-2),
            canonical_lin_vel,
            canonical_ang_vel,
        ),
        dim=-1,
    )


def split_motion_features(
    features: torch.Tensor,
    profile: str | SmpRobotProfile,
) -> dict[str, torch.Tensor]:
    """Split canonical feature vectors into named, shaped views."""
    if not isinstance(features, torch.Tensor):
        raise TypeError(f"features must be a torch.Tensor, got {type(features).__name__}.")
    if features.ndim < 1:
        raise ValueError("features must have at least one dimension.")
    if not features.is_floating_point():
        raise TypeError(f"features must have a floating-point dtype, got {features.dtype}.")
    profile = get_profile(profile)
    if features.shape[-1] != profile.feature_dim:
        raise ValueError(
            f"SMP profile '{profile.name}' requires feature dimension {profile.feature_dim}; got {features.shape[-1]}."
        )

    joint_start = 9
    key_start = joint_start + profile.num_joints
    velocity_start = key_start + 3 * profile.num_key_bodies
    return {
        "root_pos": features[..., 0:3],
        "root_rot_6d": features[..., 3:9],
        "joint_pos": features[..., joint_start:key_start],
        "key_body_pos": features[..., key_start:velocity_start].reshape(
            *features.shape[:-1], profile.num_key_bodies, 3
        ),
        "root_lin_vel": features[..., velocity_start : velocity_start + 3],
        "root_ang_vel": features[..., velocity_start + 3 : velocity_start + 6],
    }


__all__ = [
    "canonicalize_motion_features",
    "inverse_yaw_quaternion",
    "normalize_quaternion",
    "quaternion_apply",
    "quaternion_apply_inverse",
    "quaternion_conjugate",
    "quaternion_inverse",
    "quaternion_mul",
    "quaternion_multiply",
    "quaternion_normalize",
    "quaternion_to_rotation_6d",
    "quaternion_to_yaw",
    "rotation_6d_to_quaternion",
    "split_motion_features",
    "yaw_quat",
    "yaw_quaternion",
]
