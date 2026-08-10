"""Pure Torch feature construction shared by manager-based AMP environments."""

from __future__ import annotations

import torch


def amp_history_frame_indices(
    latest_frame_indices: torch.Tensor,
    history_length: int,
    total_frames: int,
    frame_stride: int = 1,
) -> torch.Tensor:
    """Return newest-first motion-frame indices for an AMP history."""
    if history_length < 1:
        raise ValueError("AMP history length must be positive.")
    if total_frames < 1:
        raise ValueError("AMP reference motion must contain at least one frame.")
    if frame_stride < 1:
        raise ValueError("AMP history frame stride must be positive.")

    latest_frame_indices = latest_frame_indices.to(dtype=torch.long)
    history_offsets = torch.arange(history_length, device=latest_frame_indices.device) * frame_stride
    return torch.clamp(
        latest_frame_indices[..., None] - history_offsets,
        min=0,
        max=total_frames - 1,
    )


def initialize_amp_history_from_reference(
    current_observation: torch.Tensor,
    reference_history: torch.Tensor,
) -> torch.Tensor:
    """Combine the actual reset state with preceding reference-motion states."""
    if reference_history.ndim != current_observation.ndim + 1:
        raise ValueError("Reference AMP history must add exactly one history dimension.")
    if reference_history.shape[0] != current_observation.shape[0]:
        raise ValueError("Current and reference AMP observations must have the same batch size.")
    if reference_history.shape[-1] != current_observation.shape[-1]:
        raise ValueError("Current and reference AMP observations must have the same feature size.")

    initialized_history = reference_history.clone()
    initialized_history[:, 0] = current_observation
    return initialized_history


def _quat_multiply(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    lhs_w, lhs_xyz = lhs[..., :1], lhs[..., 1:]
    rhs_w, rhs_xyz = rhs[..., :1], rhs[..., 1:]
    scalar = lhs_w * rhs_w - torch.sum(lhs_xyz * rhs_xyz, dim=-1, keepdim=True)
    vector = lhs_w * rhs_xyz + rhs_w * lhs_xyz + torch.cross(lhs_xyz, rhs_xyz, dim=-1)
    return torch.cat((scalar, vector), dim=-1)


def _quat_apply(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Apply scalar-first unit quaternions without requiring an Isaac Sim import."""
    xyz = quat[..., 1:]
    scalar = quat[..., :1]
    cross = torch.cross(xyz, vector, dim=-1)
    return vector + 2.0 * (scalar * cross + torch.cross(xyz, cross, dim=-1))


def _heading_inverse(root_quat: torch.Tensor) -> torch.Tensor:
    """Return the inverse yaw quaternion for scalar-first root rotations."""
    w, x, y, z = root_quat.unbind(dim=-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
    half_inverse_yaw = -0.5 * yaw
    zeros = torch.zeros_like(half_inverse_yaw)
    return torch.stack((torch.cos(half_inverse_yaw), zeros, zeros, torch.sin(half_inverse_yaw)), dim=-1)


def _orientation_tangent_normal(quat_w: torch.Tensor) -> torch.Tensor:
    tangent = torch.zeros_like(quat_w[..., :3])
    normal = torch.zeros_like(quat_w[..., :3])
    tangent[..., 0] = 1.0
    normal[..., 2] = 1.0
    return torch.cat((_quat_apply(quat_w, tangent), _quat_apply(quat_w, normal)), dim=-1)


def build_amp_observation(
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    root_lin_vel: torch.Tensor,
    root_ang_vel: torch.Tensor,
    key_body_pos: torch.Tensor,
) -> torch.Tensor:
    """Build one phase-free, velocity-bearing AMP state observation."""
    batch_size, num_key_bodies = key_body_pos.shape[:2]
    heading_inverse = _heading_inverse(root_quat)
    heading_inverse_expanded = heading_inverse[:, None, :].expand(-1, num_key_bodies, -1).reshape(-1, 4)
    key_body_pos_heading = _quat_apply(
        heading_inverse_expanded,
        (key_body_pos - root_pos[:, None, :]).reshape(-1, 3),
    ).reshape(batch_size, -1)
    root_quat_heading = _quat_multiply(heading_inverse, root_quat)
    root_lin_vel_heading = _quat_apply(heading_inverse, root_lin_vel)
    root_ang_vel_heading = _quat_apply(heading_inverse, root_ang_vel)
    return torch.cat(
        (
            joint_pos,
            joint_vel,
            root_pos[:, 2:3],
            _orientation_tangent_normal(root_quat_heading),
            root_lin_vel_heading,
            root_ang_vel_heading,
            key_body_pos_heading,
        ),
        dim=-1,
    )
