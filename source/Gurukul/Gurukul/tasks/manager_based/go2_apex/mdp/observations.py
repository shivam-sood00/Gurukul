from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

import isaaclab.utils.math as math_utils
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from .commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _safe_normalize_quat(quat: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    quat = torch.nan_to_num(quat, nan=0.0, posinf=0.0, neginf=0.0)
    norm = torch.linalg.norm(quat, dim=-1, keepdim=True)
    identity = torch.zeros_like(quat)
    identity[..., 0] = 1.0
    quat = torch.where(norm > eps, quat / norm.clamp_min(eps), identity)
    return quat


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    robot_anchor_pos_w = command.robot_anchor_pos_w[:, None, :].expand(-1, num_bodies, -1)
    robot_anchor_quat_w = _safe_normalize_quat(command.robot_anchor_quat_w)[:, None, :].expand(-1, num_bodies, -1)
    pos_b, _ = subtract_frame_transforms(
        robot_anchor_pos_w,
        robot_anchor_quat_w,
        command.robot_body_pos_w,
        _safe_normalize_quat(command.robot_body_quat_w),
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    robot_anchor_pos_w = command.robot_anchor_pos_w[:, None, :].expand(-1, num_bodies, -1)
    robot_anchor_quat_w = _safe_normalize_quat(command.robot_anchor_quat_w)[:, None, :].expand(-1, num_bodies, -1)
    _, ori_b = subtract_frame_transforms(
        robot_anchor_pos_w,
        robot_anchor_quat_w,
        command.robot_body_pos_w,
        _safe_normalize_quat(command.robot_body_quat_w),
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def reference_base_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        _safe_normalize_quat(command.robot_anchor_quat_w),
        command.anchor_pos_w,
        _safe_normalize_quat(command.anchor_quat_w),
    )

    return pos.view(env.num_envs, -1)


def reference_base_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        _safe_normalize_quat(command.robot_anchor_quat_w),
        command.anchor_pos_w,
        _safe_normalize_quat(command.anchor_quat_w),
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def reference_base_height(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return the reference base height from the imitation clip."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.anchor_pos_w[:, 2:3]


def reference_base_height_error(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return robot base height relative to the reference base height."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.robot_anchor_pos_w[:, 2:3] - command.anchor_pos_w[:, 2:3]


def motion_command_vel(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return a compact 4D motion command: vx, vy, vz, yaw_rate."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.cat([command.command_lin_vel_xy, command.command_lin_vel_z, command.command_ang_vel_z], dim=-1)


def motion_command_vel_xy(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return the imitation-sourced planar velocity command: vx, vy."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.command_lin_vel_xy


def motion_command_vel_x(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return the imitation-sourced forward velocity command: vx."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.command_lin_vel_xy[:, 0:1]


def motion_command_vel_y(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return the imitation-sourced lateral velocity command: vy."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.command_lin_vel_xy[:, 1:2]


def motion_command_vel_z(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return the reference vertical velocity command: vz."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.command_lin_vel_z


def motion_command_yaw(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return the imitation-sourced yaw command after task-level noise/clipping."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.command_ang_vel_z


def joint_position_command(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return a sampled joint-position command."""
    command = env.command_manager.get_term(command_name)
    return command.command


def motion_phase(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return the normalized imitation phase for the current motion timestep."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.time_steps.unsqueeze(-1).float() / float(max(command.motion.time_step_total, 1))


def motion_skill(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return the frame-aligned skill scalar supplied by the active motion clip."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.skill


def object_position_b(
    env: ManagerBasedEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the simulated object's position in the robot base frame."""
    object_asset = env.scene[object_cfg.name]
    robot = env.scene[robot_cfg.name]
    return math_utils.quat_apply_inverse(
        robot.data.root_quat_w,
        object_asset.data.root_pos_w - robot.data.root_pos_w,
    )


def object_linear_velocity_b(
    env: ManagerBasedEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the physical object's world velocity expressed in the robot base frame."""
    object_asset = env.scene[object_cfg.name]
    robot = env.scene[robot_cfg.name]
    return math_utils.quat_apply_inverse(robot.data.root_quat_w, object_asset.data.root_lin_vel_w)


def object_up_axis_b(
    env: ManagerBasedEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the physical object's local +Z axis in the robot base frame."""
    object_asset = env.scene[object_cfg.name]
    robot = env.scene[robot_cfg.name]
    local_up = torch.zeros((env.num_envs, 3), device=env.device, dtype=object_asset.data.root_quat_w.dtype)
    local_up[:, 2] = 1.0
    up_w = math_utils.quat_apply(object_asset.data.root_quat_w, local_up)
    return math_utils.quat_apply_inverse(robot.data.root_quat_w, up_w)


def object_orientation_b(
    env: ManagerBasedEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the physical object's full 6D orientation in the robot base frame."""
    object_asset = env.scene[object_cfg.name]
    robot = env.scene[robot_cfg.name]
    _, quat_b = subtract_frame_transforms(
        robot.data.root_pos_w,
        _safe_normalize_quat(robot.data.root_quat_w),
        object_asset.data.root_pos_w,
        _safe_normalize_quat(object_asset.data.root_quat_w),
    )
    rotation_b = matrix_from_quat(_safe_normalize_quat(quat_b))
    return rotation_b[..., :2].reshape(env.num_envs, -1)


def object_angular_velocity_b(
    env: ManagerBasedEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return the physical object's angular velocity in the robot base frame."""
    object_asset = env.scene[object_cfg.name]
    robot = env.scene[robot_cfg.name]
    return math_utils.quat_apply_inverse(robot.data.root_quat_w, object_asset.data.root_ang_vel_w)


def body_pose_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """Return selected body positions and 6D orientations in the asset root frame."""
    asset = env.scene[asset_cfg.name]
    body_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    body_quat_w = _safe_normalize_quat(asset.data.body_quat_w[:, asset_cfg.body_ids, :])
    root_pos_w = asset.data.root_pos_w[:, None, :].expand_as(body_pos_w)
    root_quat_w = _safe_normalize_quat(asset.data.root_quat_w)[:, None, :].expand_as(body_quat_w)
    pos_b, quat_b = subtract_frame_transforms(root_pos_w, root_quat_w, body_pos_w, body_quat_w)
    orientation_b = matrix_from_quat(_safe_normalize_quat(quat_b))[..., :2]
    return torch.cat([pos_b, orientation_b.reshape(env.num_envs, -1, 6)], dim=-1).reshape(env.num_envs, -1)


def gripper_object_geometry_b(
    env: ManagerBasedEnv,
    gripper_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return per-finger and gripper-center vectors/distances to the object in the base frame."""
    robot = env.scene[robot_cfg.name]
    object_asset = env.scene[object_cfg.name]
    finger_pos_w = robot.data.body_pos_w[:, gripper_cfg.body_ids, :]
    object_delta_w = object_asset.data.root_pos_w[:, None, :] - finger_pos_w
    root_quat_w = _safe_normalize_quat(robot.data.root_quat_w)[:, None, :].expand(
        -1, object_delta_w.shape[1], -1
    )
    finger_delta_b = math_utils.quat_apply_inverse(
        root_quat_w.reshape(-1, 4), object_delta_w.reshape(-1, 3)
    ).reshape_as(object_delta_w)
    finger_distance = torch.linalg.vector_norm(finger_delta_b, dim=-1)

    center_delta_w = object_asset.data.root_pos_w - finger_pos_w.mean(dim=1)
    center_delta_b = math_utils.quat_apply_inverse(robot.data.root_quat_w, center_delta_w)
    center_distance = torch.linalg.vector_norm(center_delta_b, dim=-1, keepdim=True)
    return torch.cat(
        [finger_delta_b.reshape(env.num_envs, -1), finger_distance, center_delta_b, center_distance], dim=-1
    )


def filtered_contact_force_b(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    normalize: float = 20.0,
) -> torch.Tensor:
    """Return a filtered sensor's continuous force vector and norm in the robot base frame."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    force_matrix_w = sensor.data.force_matrix_w
    if force_matrix_w is None:
        return torch.zeros(env.num_envs, 4, device=env.device)
    force_w = force_matrix_w.reshape(env.num_envs, -1, 3).sum(dim=1)
    force_b = math_utils.quat_apply_inverse(asset.data.root_quat_w, force_w)
    scale = max(float(normalize), 1.0e-6)
    force_b = force_b / scale
    return torch.cat([force_b, torch.linalg.vector_norm(force_b, dim=-1, keepdim=True)], dim=-1)


def rigid_body_mass(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg,
    normalize: float = 1.0,
) -> torch.Tensor:
    """Return selected simulator body masses for a privileged policy."""
    asset = env.scene[asset_cfg.name]
    masses = asset.root_physx_view.get_masses().to(device=env.device, dtype=torch.float32)[:, asset_cfg.body_ids]
    return masses.reshape(env.num_envs, -1) / max(float(normalize), 1.0e-6)


def environment_scalar_attribute(
    env: ManagerBasedEnv,
    attribute_name: str,
    default: float = 0.0,
) -> torch.Tensor:
    """Expose a per-environment scalar attribute such as a randomized object scale."""
    value = getattr(env, attribute_name, None)
    if value is None:
        return torch.full((env.num_envs, 1), float(default), device=env.device)
    if not isinstance(value, torch.Tensor):
        return torch.full((env.num_envs, 1), float(value), device=env.device)
    value = value.to(device=env.device, dtype=torch.float32)
    if value.numel() == 1:
        return value.reshape(1, 1).expand(env.num_envs, 1)
    return value.reshape(env.num_envs, -1)[:, :1]


def reference_object_position_b(env: ManagerBasedEnv, command_name: str, object_index: int = 0) -> torch.Tensor:
    """Return the active motion object's target position in the robot base frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    object_pos_w = command.object_target_pos_w
    if object_pos_w is None:
        return torch.zeros(env.num_envs, 3, device=env.device)
    return math_utils.quat_apply_inverse(
        command.robot_anchor_quat_w,
        object_pos_w[:, object_index] - command.robot_anchor_pos_w,
    )


def reference_object_position_trajectory_b(
    env: ManagerBasedEnv,
    command_name: str,
    time_offsets: Sequence[int] | int = (0, 1, 2, 5, 10),
    object_index: int = 0,
) -> torch.Tensor:
    """Return current/future reference object positions in the simulated robot frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    if command.motion.object_pos_w is None:
        count = 1 if isinstance(time_offsets, int) else len(time_offsets)
        return torch.zeros(env.num_envs, 3 * count, device=env.device)

    time_steps = _motion_time_steps(command, time_offsets)
    ref_pos_w = command.motion.object_pos_w[time_steps, object_index] + env.scene.env_origins[:, None, :]
    delta_pos_w = torch.cat([command.robot_anchor_pos_w[:, :2], command.anchor_pos_w[:, 2:3]], dim=-1)
    delta_quat_w = math_utils.yaw_quat(
        math_utils.quat_mul(command.robot_anchor_quat_w, math_utils.quat_inv(command.anchor_quat_w))
    )
    relative_ref_w = ref_pos_w - command.anchor_pos_w[:, None, :]
    aligned_ref_w = delta_pos_w[:, None, :] + _quat_apply_broadcast(delta_quat_w[:, None, :], relative_ref_w)
    ref_quat_w = command.motion.object_quat_w[time_steps, object_index]
    aligned_ref_quat_w = math_utils.quat_mul(
        delta_quat_w[:, None, :].expand_as(ref_quat_w).reshape(-1, 4),
        ref_quat_w.reshape(-1, 4),
    ).reshape_as(ref_quat_w)
    size_center_offset_w = command.object_size_center_offset_w(aligned_ref_quat_w)
    if size_center_offset_w is not None:
        aligned_ref_w += size_center_offset_w
    ref_pos_b = _quat_apply_inverse_broadcast(
        command.robot_anchor_quat_w[:, None, :],
        aligned_ref_w - command.robot_anchor_pos_w[:, None, :],
    )
    return ref_pos_b.reshape(env.num_envs, -1)


def reference_object_up_axis_b(
    env: ManagerBasedEnv,
    command_name: str,
    object_index: int = 0,
) -> torch.Tensor:
    """Return the reference object's local +Z axis in the simulated robot frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    object_quat_w = command.object_target_quat_w
    if object_quat_w is None:
        return torch.zeros(env.num_envs, 3, device=env.device)
    local_up = torch.zeros((env.num_envs, 3), device=env.device, dtype=object_quat_w.dtype)
    local_up[:, 2] = 1.0
    up_w = math_utils.quat_apply(object_quat_w[:, object_index], local_up)
    return math_utils.quat_apply_inverse(command.robot_anchor_quat_w, up_w)


def reference_object_orientation_trajectory_b(
    env: ManagerBasedEnv,
    command_name: str,
    time_offsets: Sequence[int] | int = (0, 1, 2, 5, 10),
    object_index: int = 0,
) -> torch.Tensor:
    """Return past/current/future reference object 6D orientations in the robot base frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    count = 1 if isinstance(time_offsets, int) else len(time_offsets)
    if command.motion.object_quat_w is None:
        return torch.zeros(env.num_envs, 6 * count, device=env.device)
    time_steps = _motion_time_steps(command, time_offsets)
    ref_quat_w = _safe_normalize_quat(command.motion.object_quat_w[time_steps, object_index])
    delta_quat_w = math_utils.yaw_quat(
        math_utils.quat_mul(command.robot_anchor_quat_w, math_utils.quat_inv(command.anchor_quat_w))
    )
    delta_quat_w = delta_quat_w[:, None, :].expand_as(ref_quat_w)
    aligned_quat_w = math_utils.quat_mul(
        delta_quat_w.reshape(-1, 4), ref_quat_w.reshape(-1, 4)
    ).reshape_as(ref_quat_w)
    robot_quat_inv = math_utils.quat_inv(_safe_normalize_quat(command.robot_anchor_quat_w))
    robot_quat_inv = robot_quat_inv[:, None, :].expand_as(aligned_quat_w)
    quat_b = math_utils.quat_mul(
        robot_quat_inv.reshape(-1, 4), aligned_quat_w.reshape(-1, 4)
    ).reshape_as(aligned_quat_w)
    orientation_b = matrix_from_quat(_safe_normalize_quat(quat_b))[..., :2]
    return orientation_b.reshape(env.num_envs, -1)


def reference_arm_ee_pose_trajectory_b(
    env: ManagerBasedEnv,
    command_name: str,
    time_offsets: Sequence[int] | int = (0, 1, 2, 5, 10),
) -> torch.Tensor:
    """Return past/current/future reference gripper-center poses in the simulated base frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    count = 1 if isinstance(time_offsets, int) else len(time_offsets)
    if command.motion.arm_ee_pos_w is None or command.motion.arm_ee_quat_w is None:
        return torch.zeros(env.num_envs, 9 * count, device=env.device)

    time_steps = _motion_time_steps(command, time_offsets)
    ref_pos_w = command.motion.arm_ee_pos_w[time_steps] + env.scene.env_origins[:, None, :]
    ref_quat_w = _safe_normalize_quat(command.motion.arm_ee_quat_w[time_steps])
    delta_pos_w = torch.cat([command.robot_anchor_pos_w[:, :2], command.anchor_pos_w[:, 2:3]], dim=-1)
    delta_quat_w = math_utils.yaw_quat(
        math_utils.quat_mul(command.robot_anchor_quat_w, math_utils.quat_inv(command.anchor_quat_w))
    )
    relative_ref_w = ref_pos_w - command.anchor_pos_w[:, None, :]
    aligned_ref_w = delta_pos_w[:, None, :] + _quat_apply_broadcast(delta_quat_w[:, None, :], relative_ref_w)
    ref_pos_b = _quat_apply_inverse_broadcast(
        command.robot_anchor_quat_w[:, None, :], aligned_ref_w - command.robot_anchor_pos_w[:, None, :]
    )

    expanded_delta_quat_w = delta_quat_w[:, None, :].expand_as(ref_quat_w)
    aligned_quat_w = math_utils.quat_mul(
        expanded_delta_quat_w.reshape(-1, 4), ref_quat_w.reshape(-1, 4)
    ).reshape_as(ref_quat_w)
    robot_quat_inv = math_utils.quat_inv(_safe_normalize_quat(command.robot_anchor_quat_w))
    robot_quat_inv = robot_quat_inv[:, None, :].expand_as(aligned_quat_w)
    quat_b = math_utils.quat_mul(
        robot_quat_inv.reshape(-1, 4), aligned_quat_w.reshape(-1, 4)
    ).reshape_as(aligned_quat_w)
    orientation_b = matrix_from_quat(_safe_normalize_quat(quat_b))[..., :2].reshape(env.num_envs, count, 6)
    return torch.cat([ref_pos_b, orientation_b], dim=-1).reshape(env.num_envs, -1)


def reference_object_attachment_phase(
    env: ManagerBasedEnv,
    command_name: str,
    time_offsets: Sequence[int] | int = (0, 1, 2, 5, 10),
    object_index: int = 0,
) -> torch.Tensor:
    """Return the known current/future grasp phase from the imitation clip."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    count = 1 if isinstance(time_offsets, int) else len(time_offsets)
    if command.motion.object_attached is None:
        return torch.zeros(env.num_envs, count, device=env.device)
    time_steps = _motion_time_steps(command, time_offsets)
    return command.motion.object_attached[time_steps, object_index].float()


def filtered_contact_state(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 0.35,
) -> torch.Tensor:
    """Return whether a filtered contact sensor saw a recent force above threshold."""
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_history = sensor.data.force_matrix_w_history
    if force_history is None:
        raise ValueError(f"Contact sensor '{sensor_cfg.name}' must define filter_prim_paths_expr.")
    hits = force_history.norm(dim=-1) > float(force_threshold)
    return hits.reshape(env.num_envs, -1).any(dim=1, keepdim=True).float()


def reference_joint_pos(
    env: ManagerBasedEnv, command_name: str, joint_names: list[str] | None = None
) -> torch.Tensor:
    """Return reference joint positions from the imitation clip."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref_joint_pos = command.joint_pos
    if joint_names is None:
        return ref_joint_pos

    motion_joint_names = getattr(command, "robot_motion_joint_names", None)
    if motion_joint_names is None:
        return ref_joint_pos

    idx_tensor = _cached_motion_joint_indices(command, joint_names, ref_joint_pos.device)
    return ref_joint_pos[:, idx_tensor]


def _cache_dict(command: MotionCommand, name: str) -> dict:
    cache = getattr(command, name, None)
    if cache is None:
        cache = {}
        setattr(command, name, cache)
    return cache


def _motion_time_steps(command: MotionCommand, time_offsets: Sequence[int] | int) -> torch.Tensor:
    if isinstance(time_offsets, int):
        time_offsets = (time_offsets,)
    if len(time_offsets) == 0:
        raise ValueError("time_offsets must contain at least one motion-frame offset.")

    offset_key = tuple(int(offset) for offset in time_offsets)
    offset_cache = _cache_dict(command, "_apex_reference_offset_cache")
    offsets = offset_cache.get(offset_key)
    if offsets is None or offsets.device != command.time_steps.device:
        offsets = torch.as_tensor(offset_key, dtype=torch.long, device=command.time_steps.device)
        offset_cache[offset_key] = offsets
    time_steps = command.time_steps[:, None] + offsets[None, :]
    start_steps = getattr(command, "time_step_starts", None)
    end_steps = getattr(command, "time_step_ends", None)
    if start_steps is None or end_steps is None:
        return torch.clamp(time_steps, min=0, max=max(int(command.motion.time_step_total) - 1, 0))
    return torch.maximum(torch.minimum(time_steps, end_steps[:, None]), start_steps[:, None])


def _select_motion_joints(command: MotionCommand, values: torch.Tensor) -> torch.Tensor:
    values = values[..., command._motion_joint_indices]
    return values


def _select_robot_ordered_joints(
    command: MotionCommand,
    values: torch.Tensor,
    joint_names: Sequence[str] | None,
) -> torch.Tensor:
    if joint_names is None:
        return values

    motion_joint_names = getattr(command, "robot_motion_joint_names", None)
    if motion_joint_names is None:
        return values

    idx_tensor = _cached_motion_joint_indices(command, joint_names, values.device)
    return values.index_select(values.ndim - 1, idx_tensor)


def _cached_motion_joint_indices(
    command: MotionCommand,
    joint_names: Sequence[str],
    device: torch.device | str,
) -> torch.Tensor:
    cache_key = tuple(joint_names)
    cache = _cache_dict(command, "_apex_motion_joint_index_cache")
    idx_tensor = cache.get(cache_key)
    if idx_tensor is not None and idx_tensor.device == torch.device(device):
        return idx_tensor

    motion_joint_names = getattr(command, "robot_motion_joint_names", None)
    if motion_joint_names is None:
        raise ValueError("Motion command does not expose robot_motion_joint_names for joint selection.")
    name_to_idx = {name: i for i, name in enumerate(motion_joint_names)}
    indices = []
    for joint_name in joint_names:
        if joint_name not in name_to_idx:
            raise ValueError(
                f"Requested joint '{joint_name}' is missing in motion-command joint names: {motion_joint_names}"
            )
        indices.append(name_to_idx[joint_name])
    idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)
    cache[cache_key] = idx_tensor
    return idx_tensor


def _motion_body_indices(command: MotionCommand, body_names: Sequence[str] | None) -> torch.Tensor:
    device = command.time_steps.device
    cache_key = None if body_names is None else tuple(body_names)
    cache = _cache_dict(command, "_apex_motion_body_index_cache")
    idx_tensor = cache.get(cache_key)
    if idx_tensor is not None and idx_tensor.device == device:
        return idx_tensor

    if body_names is None:
        indices = list(range(len(command.cfg.body_names)))
    else:
        name_to_idx = {name: i for i, name in enumerate(command.cfg.body_names)}
        indices = []
        for body_name in body_names:
            if body_name not in name_to_idx:
                raise ValueError(
                    f"Requested body '{body_name}' is missing in motion-command body names: {command.cfg.body_names}"
                )
            indices.append(name_to_idx[body_name])
    idx_tensor = torch.tensor(indices, dtype=torch.long, device=device)
    cache[cache_key] = idx_tensor
    return idx_tensor


def _quat_apply_inverse_broadcast(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply inverse quaternion rotation with explicit broadcasting for Isaac Lab's flat quaternion helper."""
    quat = quat.expand(vec.shape[:-1] + (4,))
    return math_utils.quat_apply_inverse(quat.reshape(-1, 4), vec.reshape(-1, 3)).reshape(vec.shape)


def _quat_apply_broadcast(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Apply quaternion rotation with explicit broadcasting for Isaac Lab's flat quaternion helper."""
    quat = quat.expand(vec.shape[:-1] + (4,))
    return math_utils.quat_apply(quat.reshape(-1, 4), vec.reshape(-1, 3)).reshape(vec.shape)


def reference_body_state_yaw_b(
    env: ManagerBasedEnv,
    command_name: str,
    time_offsets: Sequence[int] | int = (0,),
    body_names: Sequence[str] | None = None,
    include_pos: bool = True,
    include_lin_vel: bool = False,
    include_ang_vel: bool = False,
) -> torch.Tensor:
    """Return reference body features in the reference yaw frame.

    Positions are relative to the reference anchor body. Linear and angular velocities are rotated into the same
    yaw-only local frame. This keeps style cues local and avoids making the teacher depend on brittle global pose.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    time_steps = _motion_time_steps(command, time_offsets)
    body_idx = _motion_body_indices(command, body_names)

    anchor_pos_w = command.motion.body_pos_w[time_steps][..., command.motion_anchor_body_index, :]
    anchor_quat_w = command.motion.body_quat_w[time_steps][..., command.motion_anchor_body_index, :]
    anchor_yaw_quat = math_utils.yaw_quat(_safe_normalize_quat(anchor_quat_w))

    features = []
    if include_pos:
        body_pos_w = command.motion.body_pos_w[time_steps].index_select(-2, body_idx)
        rel_pos_w = body_pos_w - anchor_pos_w[..., None, :]
        features.append(_quat_apply_inverse_broadcast(anchor_yaw_quat[..., None, :], rel_pos_w))
    if include_lin_vel:
        body_lin_vel_w = command.motion.body_lin_vel_w[time_steps].index_select(-2, body_idx)
        features.append(_quat_apply_inverse_broadcast(anchor_yaw_quat[..., None, :], body_lin_vel_w))
    if include_ang_vel:
        body_ang_vel_w = command.motion.body_ang_vel_w[time_steps].index_select(-2, body_idx)
        features.append(_quat_apply_inverse_broadcast(anchor_yaw_quat[..., None, :], body_ang_vel_w))

    if len(features) == 0:
        raise ValueError("reference_body_state_yaw_b needs at least one included feature.")
    return torch.cat(features, dim=-1).reshape(env.num_envs, -1)


def reference_motion_state(
    env: ManagerBasedEnv,
    command_name: str,
    time_offsets: Sequence[int] | int = (0,),
    joint_names: Sequence[str] | None = None,
    include_joint_pos: bool = True,
    include_joint_vel: bool = False,
    include_base_lin_vel: bool = False,
    include_base_ang_vel: bool = False,
    include_base_quat: bool = False,
    include_command_vel: bool = False,
    add_noise: bool = False,
    noise_std: float = 0.01,
) -> torch.Tensor:
    """Return configurable reference motion features at current/future motion-frame offsets.

    Features are concatenated per offset, then flattened. For example, offsets ``(0, 2)`` return all requested
    features for the current reference frame first, followed by the same features two reference frames ahead.
    Optional noise is applied only to this observation tensor; tracking targets remain the clean motion data.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    time_steps = _motion_time_steps(command, time_offsets)
    features = []

    if include_joint_pos:
        joint_pos = command.motion_joint_pos_at(time_steps)
        features.append(_select_robot_ordered_joints(command, joint_pos, joint_names))
    if include_joint_vel:
        joint_vel = command.motion_joint_vel_at(time_steps)
        features.append(_select_robot_ordered_joints(command, joint_vel, joint_names))

    if include_base_lin_vel:
        features.append(command.motion.body_lin_vel_w[time_steps][..., command.motion_anchor_body_index, :])
    if include_base_ang_vel:
        features.append(command.motion.body_ang_vel_w[time_steps][..., command.motion_anchor_body_index, :])
    if include_base_quat:
        base_quat = command.motion.body_quat_w[time_steps][..., command.motion_anchor_body_index, :]
        features.append(_safe_normalize_quat(base_quat))
    if include_command_vel:
        if command.motion.command_lin_vel_xy is None:
            command_lin_vel_xy = command.motion.body_lin_vel_w[time_steps][..., command.motion_anchor_body_index, :2]
        else:
            command_lin_vel_xy = command.motion.command_lin_vel_xy[time_steps]
        command_lin_vel_z = command.motion.body_lin_vel_w[time_steps][..., command.motion_anchor_body_index, 2:3]
        if command.motion.command_ang_vel_z is None:
            command_ang_vel_z = command.motion.body_ang_vel_w[time_steps][..., command.motion_anchor_body_index, 2:3]
        else:
            command_ang_vel_z = command.motion.command_ang_vel_z[time_steps]
        features.append(torch.cat([command_lin_vel_xy, command_lin_vel_z, command_ang_vel_z], dim=-1))

    if len(features) == 0:
        raise ValueError("reference_motion_state needs at least one included feature.")
    reference_state = torch.cat(features, dim=-1).reshape(env.num_envs, -1)
    if add_noise:
        reference_state = reference_state + torch.randn_like(reference_state) * float(noise_std)
    return reference_state


def reference_foot_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return reference foot positions in the anchor/body frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    cache = _cache_dict(command, "_apex_reference_foot_index_cache")
    foot_idx = cache.get("foot")
    if foot_idx is None or foot_idx.device != command.time_steps.device:
        foot_indices = [i for i, body_name in enumerate(command.cfg.body_names) if "foot" in body_name.lower()]
        if not foot_indices:
            foot_indices = [
                i for i, body_name in enumerate(command.cfg.body_names) if body_name != command.cfg.anchor_body_name
            ]
        foot_idx = torch.tensor(foot_indices, dtype=torch.long, device=command.time_steps.device)
        cache["foot"] = foot_idx
    ref_yaw_quat = math_utils.yaw_quat(command.anchor_quat_w)

    ref_foot_pos = command.body_pos_w.index_select(1, foot_idx) - command.anchor_pos_w[:, None, :]
    ref_foot_pos_b = _quat_apply_inverse_broadcast(ref_yaw_quat[:, None, :], ref_foot_pos)

    return ref_foot_pos_b.reshape(env.num_envs, -1)


def robot_root_lin_vel_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return robot root linear velocity in world frame."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_lin_vel_w


def robot_root_ang_vel_w(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return robot root angular velocity in world frame."""
    asset = env.scene[asset_cfg.name]
    return asset.data.root_ang_vel_w


def joint_torques(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return applied joint torques for the selected joints."""
    asset = env.scene[asset_cfg.name]
    return asset.data.applied_torque[:, asset_cfg.joint_ids]


def feet_contact_state(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Return binary contact state for selected feet."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    return (contacts > float(force_threshold)).float()


def contact_forces_b(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*_foot"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    normalize: float = 100.0,
) -> torch.Tensor:
    """Return selected contact forces in the robot base frame for a privileged critic."""
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    asset = env.scene[asset_cfg.name]
    forces_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, forces_w.shape[1], -1)
    forces_b = math_utils.quat_apply_inverse(
        root_quat_w.reshape(-1, 4),
        forces_w.reshape(-1, 3),
    ).reshape(forces_w.shape)
    return (forces_b / float(normalize)).reshape(env.num_envs, -1)


def motion_id_one_hot(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return one-hot identity of the currently sampled reference clip."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    num_motions = max(len(command.motion.motion_names), 1)
    motion_ids = torch.clamp(command.current_motion_ids.long(), min=0, max=num_motions - 1)
    return F.one_hot(motion_ids, num_classes=num_motions).float()


def reference_base_quat(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Return the reference base quaternion from the imitation clip."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return _safe_normalize_quat(command.anchor_quat_w)


def depth_image_features(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg,
    data_type: str = "distance_to_camera",
    crop_top: int = 0,
    crop_bottom: int = 0,
    crop_left: int = 0,
    crop_right: int = 0,
    resize: tuple[int, int] | None = None,
    normalize: bool = True,
) -> torch.Tensor:
    """Return flattened depth image features from a camera sensor."""
    camera_sensor = env.scene[sensor_cfg.name]
    depth = camera_sensor.data.output[data_type]
    if depth.ndim == 4 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    depth = torch.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)

    height, width = depth.shape[-2], depth.shape[-1]
    top = max(0, int(crop_top))
    bottom = max(0, int(crop_bottom))
    left = max(0, int(crop_left))
    right = max(0, int(crop_right))

    if top + bottom >= height:
        top, bottom = 0, 0
    if left + right >= width:
        left, right = 0, 0

    row_end = height - bottom if bottom > 0 else height
    col_end = width - right if right > 0 else width
    depth = depth[:, top:row_end, left:col_end]

    if resize is not None:
        target_h, target_w = int(resize[0]), int(resize[1])
        if target_h > 0 and target_w > 0 and (depth.shape[-2] != target_h or depth.shape[-1] != target_w):
            depth = F.interpolate(
                depth.unsqueeze(1),
                size=(target_h, target_w),
                mode="bilinear",
                align_corners=False,
            ).squeeze(1)

    if normalize:
        max_distance = float(getattr(camera_sensor.cfg, "max_distance", 0.0))
        if max_distance > 0.0:
            depth = depth / max_distance - 0.5

    return depth.reshape(depth.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Backward-compatible alias for `reference_base_pos_b`."""
    return reference_base_pos_b(env, command_name)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    """Backward-compatible alias for `reference_base_ori_b`."""
    return reference_base_ori_b(env, command_name)


def decap_factor(env: ManagerBasedEnv) -> torch.Tensor:
    action_term = env.action_manager.get_term("joint_pos")
    if hasattr(action_term, "decap_lambda"):
        return action_term.decap_lambda
    return torch.zeros(env.num_envs, 1, device=env.device)
