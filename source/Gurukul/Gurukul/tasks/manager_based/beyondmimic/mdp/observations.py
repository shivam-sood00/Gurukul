from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.utils.math import matrix_from_quat, subtract_frame_transforms

from Gurukul.tasks.manager_based.beyondmimic.mdp.commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def robot_anchor_ori_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    mat = matrix_from_quat(command.robot_anchor_quat_w)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_anchor_lin_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, :3].view(env.num_envs, -1)


def robot_anchor_ang_vel_w(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    return command.robot_anchor_vel_w[:, 3:6].view(env.num_envs, -1)


def robot_body_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    pos_b, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    num_bodies = len(command.cfg.body_names)
    _, ori_b = subtract_frame_transforms(
        command.robot_anchor_pos_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_anchor_quat_w[:, None, :].repeat(1, num_bodies, 1),
        command.robot_body_pos_w,
        command.robot_body_quat_w,
    )
    mat = matrix_from_quat(ori_b)
    return mat[..., :2].reshape(mat.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    pos, _ = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )

    return pos.view(env.num_envs, -1)


def motion_anchor_ori_b(env: ManagerBasedEnv, command_name: str) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    _, ori = subtract_frame_transforms(
        command.robot_anchor_pos_w,
        command.robot_anchor_quat_w,
        command.anchor_pos_w,
        command.anchor_quat_w,
    )
    mat = matrix_from_quat(ori)
    return mat[..., :2].reshape(mat.shape[0], -1)


def _motion_time_steps(command: MotionCommand, time_offsets: Sequence[int] | int) -> torch.Tensor:
    if isinstance(time_offsets, int):
        time_offsets = (time_offsets,)
    if len(time_offsets) == 0:
        raise ValueError("time_offsets must contain at least one motion-frame offset.")

    offsets = torch.as_tensor(time_offsets, dtype=torch.long, device=command.time_steps.device)
    time_steps = command.time_steps[:, None] + offsets[None, :]
    return torch.clamp(time_steps, min=0, max=max(int(command.motion.time_step_total) - 1, 0))


def reference_motion_state(
    env: ManagerBasedEnv,
    command_name: str,
    time_offsets: Sequence[int] | int = (0,),
    include_joint_pos: bool = True,
    include_joint_vel: bool = False,
    include_base_lin_vel: bool = False,
    include_base_ang_vel: bool = False,
    include_base_quat: bool = False,
    include_base_rotmat: bool = False,
) -> torch.Tensor:
    """Return reference motion features at current/future frame offsets for APEX-style tracking."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    time_steps = _motion_time_steps(command, time_offsets)
    features = []

    if include_joint_pos:
        features.append(command.motion.joint_pos[time_steps])
    if include_joint_vel:
        features.append(command.motion.joint_vel[time_steps])
    if include_base_lin_vel:
        features.append(command.motion.body_lin_vel_w[time_steps][..., command.motion_anchor_body_index, :])
    if include_base_ang_vel:
        features.append(command.motion.body_ang_vel_w[time_steps][..., command.motion_anchor_body_index, :])
    if include_base_quat:
        features.append(command.motion.body_quat_w[time_steps][..., command.motion_anchor_body_index, :])
    if include_base_rotmat:
        base_quat = command.motion.body_quat_w[time_steps][..., command.motion_anchor_body_index, :]
        features.append(matrix_from_quat(base_quat)[..., :2].reshape(*base_quat.shape[:-1], 6))

    if len(features) == 0:
        raise ValueError("reference_motion_state needs at least one included feature.")
    return torch.cat(features, dim=-1).reshape(env.num_envs, -1)
