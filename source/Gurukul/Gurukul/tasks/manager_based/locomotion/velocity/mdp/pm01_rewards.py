# SPDX-License-Identifier: Apache-2.0
"""PM01 reward terms aligned with EngineAI-Lab (engineai_amp) velocity task — not AMP."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv, mdp as isaac_mdp
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import (
    euler_xyz_from_quat,
    quat_apply_inverse,
    quat_from_euler_xyz,
    wrap_to_pi,
    yaw_quat,
)

def action_smoothness(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize action second-order differences."""
    action_manager = env.action_manager
    prev_prev_action = getattr(action_manager, "_prev_prev_action", None)
    if prev_prev_action is None:
        prev_prev_action = torch.zeros_like(action_manager.action)
        action_manager._prev_prev_action = prev_prev_action

    second_diff = action_manager.action + prev_prev_action - 2.0 * action_manager.prev_action
    reward = torch.sum(torch.square(second_diff), dim=1)

    prev_prev_action.copy_(action_manager.prev_action)
    reset_buf = getattr(env, "reset_buf", None)
    if reset_buf is not None:
        reset_env_ids = reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if reset_env_ids.numel() > 0:
            prev_prev_action[reset_env_ids] = 0.0

    return reward


def _epoch_curriculum_scale(env: ManagerBasedRLEnv, start_scale: float, power: float, interval_steps: int) -> float:
    num_step = env.common_step_counter
    interval = max(int(interval_steps), 1)
    updates = num_step // interval
    return float(start_scale) ** (float(power) ** updates)


def action_smoothness_with_curriculum(
    env: ManagerBasedRLEnv, start_scale: float, power: float, interval_epochs: int
) -> torch.Tensor:
    reward = action_smoothness(env)
    return reward * _epoch_curriculum_scale(env, start_scale, power, interval_epochs)


def action_rate_with_curriculum(
    env: ManagerBasedRLEnv, start_scale: float, power: float, interval_epochs: int
) -> torch.Tensor:
    reward = isaac_mdp.action_rate_l2(env)
    return reward * _epoch_curriculum_scale(env, start_scale, power, interval_epochs)


def energy_cost(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    joint_torques = asset.data.applied_torque[:, :]
    joint_vel = asset.data.joint_vel[:, :]
    power = joint_torques * joint_vel
    return torch.sum(torch.abs(power), dim=1)


def energy_cost_with_curriculum(
    env: ManagerBasedRLEnv,
    start_scale: float,
    power: float,
    interval_epochs: int,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    reward = energy_cost(env, asset_cfg=asset_cfg)
    return reward * _epoch_curriculum_scale(env, start_scale, power, interval_epochs)


def track_lin_vel_xy_yaw_frame_exp_sigma(
    env: ManagerBasedRLEnv,
    sigma: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    stand_threshold: float = 0.06,
) -> torch.Tensor:
    """Hybrid stand/walk velocity tracking (EngineAI-Lab style)."""
    commands = env.command_manager.get_command(command_name)
    stand_command = (torch.norm(commands[:, :2], dim=1) < stand_threshold) & (
        torch.abs(commands[:, 2]) < stand_threshold
    )
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error_square = torch.sum(torch.square(commands[:, :2] - vel_yaw[:, :2]), dim=1)
    lin_vel_error_abs = torch.sum(torch.abs(commands[:, :2] - vel_yaw[:, :2]), dim=1)
    rew_square = torch.exp(-lin_vel_error_square * sigma)
    rew_abs = torch.exp(-lin_vel_error_abs * sigma)
    return torch.where(stand_command, rew_abs, rew_square)


def track_ang_vel_z_world_exp_sigma(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    stand_threshold: float = 0.06,
) -> torch.Tensor:
    """Hybrid stand/walk yaw-rate tracking (EngineAI-Lab style)."""
    commands = env.command_manager.get_command(command_name)
    stand_command = (torch.norm(commands[:, :2], dim=1) < stand_threshold) & (
        torch.abs(commands[:, 2]) < stand_threshold
    )
    asset = env.scene[asset_cfg.name]
    ang_vel_error_square = torch.square(commands[:, 2] - asset.data.root_ang_vel_w[:, 2])
    ang_vel_error_abs = torch.abs(commands[:, 2] - asset.data.root_ang_vel_w[:, 2])
    rew_square = torch.exp(-ang_vel_error_square * sigma)
    rew_abs = torch.exp(-ang_vel_error_abs * sigma)
    return torch.where(stand_command, rew_abs, rew_square)


def feet_contact_fixed(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    stand_threshold: float = 0.06,
    force_threshold: float = 5.0,
) -> torch.Tensor:
    commands = env.command_manager.get_command(command_name)
    stand_command = (torch.norm(commands[:, :2], dim=1) < stand_threshold) & (
        torch.abs(commands[:, 2]) < stand_threshold
    )

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact_history = contact_sensor.data.net_forces_w_history
    if contact_history is None:
        contact_history = contact_sensor.data.net_forces_w.unsqueeze(1)

    contacts = contact_history[:, :, sensor_cfg.body_ids, 2] > force_threshold
    contact_num_buf = torch.sum(contacts, dim=-1)

    stand_contact = contact_num_buf[:, -1] == 2
    reward = (stand_command & stand_contact).float()
    contact_mask = (~stand_command) & torch.any(contact_num_buf == 1, dim=1)
    reward[contact_mask] = 1.0

    return reward


def feet_position(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    stand_threshold: float = 0.06,
    ankle_distance: float = 0.22,
    base_height_target: float = 0.82,
) -> torch.Tensor:
    commands = env.command_manager.get_command(command_name)
    stand_command = (torch.norm(commands[:, :2], dim=1) < stand_threshold) & (
        torch.abs(commands[:, 2]) < stand_threshold
    )
    asset = env.scene[asset_cfg.name]

    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    base_pos_w = asset.data.root_pos_w
    base_quat_w = asset.data.root_quat_w

    r, p, y = euler_xyz_from_quat(base_quat_w)
    heading_quat = quat_from_euler_xyz(torch.zeros_like(r), torch.zeros_like(p), y)
    feet_pos_rel = feet_pos_w - base_pos_w.unsqueeze(1)
    num_envs, num_feet, _ = feet_pos_rel.shape
    heading_quat_per_foot = heading_quat.unsqueeze(1).expand(-1, num_feet, -1).reshape(-1, 4)
    feet_pos_rel_flat = feet_pos_rel.reshape(-1, 3)
    feet_pos_heading = quat_apply_inverse(heading_quat_per_foot, feet_pos_rel_flat).reshape(num_envs, num_feet, 3)

    desired_x = torch.zeros((num_envs, num_feet), device=feet_pos_heading.device)
    desired_y = torch.cat(
        (
            (ankle_distance * 0.5) * torch.ones((num_envs, num_feet // 2), device=feet_pos_heading.device),
            (-ankle_distance * 0.5) * torch.ones((num_envs, num_feet - num_feet // 2), device=feet_pos_heading.device),
        ),
        dim=1,
    )
    desired_z = -(base_height_target - 0.045) * torch.ones((num_envs, num_feet), device=feet_pos_heading.device)
    desired = torch.stack((desired_x, desired_y, desired_z), dim=-1)

    position_error = torch.sum(torch.abs(feet_pos_heading - desired), dim=(1, 2))
    reward_stand = torch.exp(-position_error * 3.0)
    return torch.where(stand_command, reward_stand, torch.ones_like(reward_stand))


def feet_orientation(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    command_name: str,
    stand_threshold: float = 0.06,
) -> torch.Tensor:
    commands = env.command_manager.get_command(command_name)
    yaw_command = torch.abs(commands[:, 2]) > stand_threshold

    asset = env.scene[asset_cfg.name]
    feet_quat = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    base_quat = asset.data.root_quat_w

    num_envs, num_feet, _ = feet_quat.shape
    feet_flat = feet_quat.reshape(-1, 4)
    roll, pitch, yaw = euler_xyz_from_quat(feet_flat)
    roll = roll.reshape(num_envs, num_feet)
    pitch = pitch.reshape(num_envs, num_feet)
    yaw = yaw.reshape(num_envs, num_feet)

    _, _, base_yaw = euler_xyz_from_quat(base_quat)

    feet_roll_pitch_error = torch.sum(torch.abs(torch.stack((roll, pitch), dim=-1)), dim=-1)
    feet_yaw_error = torch.abs(wrap_to_pi(yaw - base_yaw.unsqueeze(1)))

    rew = torch.sum(feet_roll_pitch_error + feet_yaw_error, dim=1)
    rew[yaw_command] = torch.sum(feet_roll_pitch_error[yaw_command], dim=1)
    return torch.exp(-rew * 2.0)


def base_height_tracking(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"), target_height: float = 0.82
) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    height_error = torch.abs(asset.data.root_pos_w[:, 2] - target_height)
    return torch.exp(-height_error * 30.0)


def base_orientation(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    asset = env.scene[asset_cfg.name]
    roll, pitch, yaw = euler_xyz_from_quat(asset.data.root_quat_w)
    base_euler = torch.stack((roll, pitch, yaw), dim=-1)
    return torch.exp(-torch.sum(torch.abs(base_euler[:, :2]), dim=-1) * 10.0)


def joint_deviation_exp(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    tolerance: float = 0.1,
    scale: float = 3.0,
    max_err: float = 50.0,
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids if asset_cfg.joint_ids is not None else slice(None)
    joint_pos = asset.data.joint_pos[:, joint_ids]
    default_pos = getattr(asset.data, "default_joint_pos", None)
    if default_pos is not None:
        default_pos = default_pos[:, joint_ids]
    else:
        default_pos = torch.zeros_like(joint_pos)

    joint_error = torch.norm(joint_pos - default_pos, dim=1)
    joint_error = torch.clamp(joint_error - tolerance, min=0.0, max=max_err)
    return torch.exp(-joint_error * scale)


def feet_stumble_pm01(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    tangential_threshold: float = 2.0,
    normal_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize feet hitting vertical surfaces (EngineAI-Lab style)."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    tangential = torch.norm(forces[..., :2], dim=-1) > tangential_threshold
    small_normal = torch.abs(forces[..., 2]) < normal_threshold
    stumble = tangential & small_normal
    return stumble.sum(dim=1).float()
