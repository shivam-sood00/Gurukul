# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase, SceneEntityCfg
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _base_motion_command_norm(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Return the norm of the locomotion part ``[vx, vy, wz]`` of a command."""
    command = env.command_manager.get_command(command_name)
    return torch.linalg.vector_norm(command[:, : min(3, command.shape[1])], dim=1)


def _motion_posture_command_norm(
    env: ManagerBasedRLEnv,
    command_name: str,
    posture_nominal_height: float | None = None,
) -> torch.Tensor:
    """Measure motion/posture activity without treating absolute body height as perpetual motion."""
    command = env.command_manager.get_command(command_name)
    if command.shape[1] <= 3:
        return torch.linalg.vector_norm(command, dim=1)
    activity = command.clone()
    if posture_nominal_height is not None and activity.shape[1] >= 6:
        activity[:, 5] -= float(posture_nominal_height)
    return torch.linalg.vector_norm(activity, dim=1)


def _infer_grid_shape_from_sensor(sensor: RayCaster, num_rays: int) -> tuple[int, int]:
    """Infer 2D grid shape from ray-caster pattern metadata and ray count."""
    pattern_cfg = getattr(sensor.cfg, "pattern_cfg", None)
    if pattern_cfg is not None and hasattr(pattern_cfg, "size") and hasattr(pattern_cfg, "resolution"):
        size = getattr(pattern_cfg, "size")
        resolution = float(getattr(pattern_cfg, "resolution"))
        if isinstance(size, (list, tuple)) and len(size) == 2 and resolution > 0.0:
            sx, sy = float(size[0]), float(size[1])
            candidates: list[tuple[int, int]] = []
            for add_x in (0, 1):
                for add_y in (0, 1):
                    nx = max(1, int(round(sx / resolution)) + add_x)
                    ny = max(1, int(round(sy / resolution)) + add_y)
                    candidates.append((nx, ny))
                    candidates.append((ny, nx))
            for nx, ny in candidates:
                if nx * ny == num_rays:
                    return nx, ny

    # Fallback: factorization closest to square (robust to unknown pattern metadata).
    best_rows, best_cols = num_rays, 1
    best_score = float("inf")
    max_divisor = int(math.sqrt(num_rays)) + 1
    for divisor in range(1, max_divisor):
        if num_rays % divisor != 0:
            continue
        rows = num_rays // divisor
        cols = divisor
        score = abs(rows - cols)
        if score < best_score:
            best_rows, best_cols = rows, cols
            best_score = score
    return best_rows, best_cols


def track_lin_vel_xy_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - asset.data.root_lin_vel_b[:, :2]),
        dim=1,
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    # compute the error
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_b[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_linear_velocity_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Mjlab-style base linear velocity tracking, including zero target z velocity."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    xy_error = torch.sum(torch.square(command[:, :2] - asset.data.root_lin_vel_b[:, :2]), dim=1)
    z_error = torch.square(asset.data.root_lin_vel_b[:, 2])
    return torch.exp(-(xy_error + z_error) / std**2)


def track_angular_velocity_exp(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Mjlab-style base angular velocity tracking, including zero target roll/pitch rates."""
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    z_error = torch.square(command[:, 2] - asset.data.root_ang_vel_b[:, 2])
    xy_error = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    return torch.exp(-(z_error + xy_error) / std**2)


def track_base_roll_pitch_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward tracking commanded base roll and pitch stored after velocity command dimensions."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    if command.shape[1] < 5:
        return torch.ones(env.num_envs, device=env.device)

    gravity = asset.data.projected_gravity_b
    roll = torch.atan2(gravity[:, 1], -gravity[:, 2])
    pitch = torch.atan2(-gravity[:, 0], -gravity[:, 2])
    error = torch.sum(torch.square(command[:, 3:5] - torch.stack((roll, pitch), dim=-1)), dim=1)
    return torch.exp(-error / float(std) ** 2)


def track_base_height_command_exp(
    env: ManagerBasedRLEnv,
    std: float,
    command_name: str,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward tracking commanded base height stored after velocity and roll/pitch command dimensions."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    if command.shape[1] < 6:
        return torch.ones(env.num_envs, device=env.device)

    base_height = asset.data.root_pos_w[:, 2] - _ground_height_from_sensor(env, sensor_cfg)
    error = torch.square(command[:, 5] - base_height)
    return torch.exp(-error / float(std) ** 2)


def track_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of linear velocity commands (xy axes) in the gravity aligned robot frame.

    Uses exponential kernel for reward computation.
    """
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    vel_yaw = quat_apply_inverse(yaw_quat(asset.data.root_quat_w), asset.data.root_lin_vel_w[:, :3])
    lin_vel_error = torch.sum(
        torch.square(env.command_manager.get_command(command_name)[:, :2] - vel_yaw[:, :2]), dim=1
    )
    reward = torch.exp(-lin_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def track_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking of angular velocity commands (yaw) in world frame using exponential kernel."""
    # extract the used quantities (to enable type-hinting)
    asset = env.scene[asset_cfg.name]
    ang_vel_error = torch.square(env.command_manager.get_command(command_name)[:, 2] - asset.data.root_ang_vel_w[:, 2])
    reward = torch.exp(-ang_vel_error / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_power(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Reward joint_power"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute the reward
    reward = torch.sum(
        torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids] * asset.data.applied_torque[:, asset_cfg.joint_ids]),
        dim=1,
    )
    return reward


def stand_still(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.06,
    posture_nominal_height: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    # Penalize motion when command is nearly zero.
    reward = mdp.joint_deviation_l1(env, asset_cfg)
    command_norm = _motion_posture_command_norm(env, command_name, posture_nominal_height)
    reward *= command_norm < command_threshold
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_pos_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    stand_still_scale: float,
    velocity_threshold: float,
    command_threshold: float,
    posture_nominal_height: float | None = None,
) -> torch.Tensor:
    """Penalize joint position error from default on the articulation."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = _motion_posture_command_norm(env, command_name, posture_nominal_height)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    running_reward = torch.linalg.norm(
        (asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]), dim=1
    )
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        stand_still_scale * running_reward,
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def joint_error_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint-position deviation from default with an L2 kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.square(
            asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        ),
        dim=1,
    )


def arm_joint_target_error_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_attr: str = "_arm_joint_target_pos",
) -> torch.Tensor:
    """Penalize policy-controlled arm error to a generated joint-space command."""
    asset: Articulation = env.scene[asset_cfg.name]
    target = getattr(env, target_attr, None)
    if target is None:
        return torch.zeros(env.num_envs, device=env.device)
    current = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.sum(torch.square(target - current), dim=1)


def arm_joint_target_tracking_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_attr: str = "_arm_joint_target_pos",
) -> torch.Tensor:
    """Reward tracking generated arm joint commands with an exponential kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    target = getattr(env, target_attr, None)
    if target is None:
        return torch.ones(env.num_envs, device=env.device)
    current = asset.data.joint_pos[:, asset_cfg.joint_ids]
    error = torch.mean(torch.square(target - current), dim=1)
    return torch.exp(-error / float(std) ** 2)


def arm_ee_target_error_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="Link6"),
    target_attr: str = "_arm_ee_target_pos",
) -> torch.Tensor:
    """Penalize policy-controlled arm error to a generated end-effector position command."""
    asset: Articulation = env.scene[asset_cfg.name]
    target = getattr(env, target_attr, None)
    if target is None:
        return torch.zeros(env.num_envs, device=env.device)
    ee_body_id = int(asset_cfg.body_ids[0])
    ee_pos_w = asset.data.body_pos_w[:, ee_body_id, :]
    ee_pos_b = quat_apply_inverse(asset.data.root_quat_w, ee_pos_w - asset.data.root_pos_w)
    return torch.sum(torch.square(target - ee_pos_b), dim=1)


def arm_ee_target_tracking_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="Link6"),
    target_attr: str = "_arm_ee_target_pos",
) -> torch.Tensor:
    """Reward tracking a generated end-effector position command with an exponential kernel."""
    error = arm_ee_target_error_l2(env, asset_cfg, target_attr)
    return torch.exp(-error / float(std) ** 2)


def hip_to_default_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=".*_hip_joint"),
) -> torch.Tensor:
    """Penalize hip joint deviation from the default pose, matching go2_rl_gym's hip_to_default term."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.abs(asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]),
        dim=1,
    )


def _selected_names(all_names: list[str] | tuple[str, ...], ids) -> list[str]:
    if isinstance(ids, slice):
        return list(all_names[ids])
    return [all_names[i] for i in ids]


def _regex_std_vector(std_by_pattern: dict[str, float], names: list[str], device: str, fallback: float = 1.0):
    std = torch.full((len(names),), float(fallback), device=device, dtype=torch.float32)
    for pattern, value in std_by_pattern.items():
        compiled = re.compile(pattern)
        for idx, name in enumerate(names):
            if compiled.fullmatch(name) or compiled.match(name):
                std[idx] = float(value)
    return torch.clamp(std, min=1.0e-6)


def _command_activity(env: ManagerBasedRLEnv, command_name: str | None, command_threshold: float) -> torch.Tensor:
    if command_name is None:
        return torch.ones(env.num_envs, device=env.device)
    command = env.command_manager.get_command(command_name)
    return (torch.linalg.norm(command[:, :2], dim=1) + torch.abs(command[:, 2]) > command_threshold).float()


def _terrain_reference_height(env: ManagerBasedRLEnv, height_sensor_name: str | None) -> torch.Tensor | None:
    if height_sensor_name is None:
        return None
    try:
        sensor = env.scene[height_sensor_name]
    except (KeyError, AttributeError):
        return None
    if sensor is None or not hasattr(sensor, "data") or not hasattr(sensor.data, "ray_hits_w"):
        return None
    ray_hits = sensor.data.ray_hits_w[..., 2]
    ray_hits = torch.nan_to_num(ray_hits, nan=0.0, posinf=0.0, neginf=0.0)
    return torch.mean(ray_hits, dim=1, keepdim=True)


def _terrain_normal_from_raycaster(
    env: ManagerBasedRLEnv,
    terrain_sensor_cfg: SceneEntityCfg | None,
) -> torch.Tensor | None:
    if terrain_sensor_cfg is None:
        return None
    try:
        sensor: RayCaster = env.scene[terrain_sensor_cfg.name]
    except (KeyError, AttributeError):
        return None
    if sensor is None or not hasattr(sensor, "data") or not hasattr(sensor.data, "ray_hits_w"):
        return None

    points = sensor.data.ray_hits_w
    valid = torch.isfinite(points).all(dim=-1)
    safe_points = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
    valid_count = valid.sum(dim=1).clamp(min=1).to(dtype=safe_points.dtype)
    centroid = (safe_points * valid.unsqueeze(-1)).sum(dim=1) / valid_count.unsqueeze(-1)
    centered = (safe_points - centroid.unsqueeze(1)) * valid.unsqueeze(-1)
    covariance = torch.bmm(centered.transpose(1, 2), centered) / valid_count.view(-1, 1, 1)
    _, eigvecs = torch.linalg.eigh(covariance)
    normal = eigvecs[:, :, 0]
    normal = torch.where(normal[:, 2:3] < 0.0, -normal, normal)
    normal = F.normalize(normal, dim=-1)

    fallback = torch.zeros((env.num_envs, 3), device=env.device, dtype=safe_points.dtype)
    fallback[:, 2] = 1.0
    enough_points = valid.sum(dim=1, keepdim=True) >= 3
    finite_normal = torch.isfinite(normal).all(dim=1, keepdim=True)
    return torch.where(enough_points & finite_normal, normal, fallback)


def _body_frame_xy_error(asset: RigidObject, asset_cfg: SceneEntityCfg, target_w: torch.Tensor) -> torch.Tensor:
    if asset_cfg.body_ids == slice(None):
        target_b = quat_apply_inverse(asset.data.root_quat_w, target_w)
        return torch.sum(torch.square(target_b[:, :2]), dim=1)

    body_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids, :]
    if body_quat_w.ndim == 2:
        target_b = quat_apply_inverse(body_quat_w, target_w)
        return torch.sum(torch.square(target_b[:, :2]), dim=1)

    num_bodies = body_quat_w.shape[1]
    target_w = target_w.unsqueeze(1).expand(-1, num_bodies, -1)
    target_b = quat_apply_inverse(body_quat_w.reshape(-1, 4), target_w.reshape(-1, 3))
    target_b = target_b.reshape(body_quat_w.shape[0], num_bodies, 3)
    return torch.mean(torch.sum(torch.square(target_b[..., :2]), dim=-1), dim=1)


class variable_posture(ManagerTermBase):
    """Mjlab-style speed-dependent posture reward.

    This is the Isaac/RL articulation equivalent of mjlab's variable_posture term:
    it rewards staying near the default pose, with per-joint tolerances selected
    from standing/walking/running command regimes.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        all_joint_names = getattr(asset, "joint_names", None) or getattr(asset.data, "joint_names")
        self.joint_names = _selected_names(all_joint_names, asset_cfg.joint_ids)
        self.std_standing = _regex_std_vector(cfg.params["std_standing"], self.joint_names, env.device)
        self.std_walking = _regex_std_vector(cfg.params["std_walking"], self.joint_names, env.device)
        self.std_running = _regex_std_vector(cfg.params["std_running"], self.joint_names, env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg,
        command_name: str,
        std_standing: dict[str, float],
        std_walking: dict[str, float],
        std_running: dict[str, float],
        walking_threshold: float = 0.05,
        running_threshold: float = 1.5,
    ) -> torch.Tensor:
        del std_standing, std_walking, std_running
        asset: Articulation = env.scene[asset_cfg.name]
        command = env.command_manager.get_command(command_name)
        command_speed = torch.linalg.norm(command[:, :2], dim=1) + torch.abs(command[:, 2])
        standing = command_speed < walking_threshold
        running = command_speed >= running_threshold
        walking = ~(standing | running)
        std = (
            self.std_standing.unsqueeze(0) * standing.unsqueeze(1)
            + self.std_walking.unsqueeze(0) * walking.unsqueeze(1)
            + self.std_running.unsqueeze(0) * running.unsqueeze(1)
        )
        joint_error = (
            asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
        )
        return torch.exp(-torch.mean(torch.square(joint_error) / torch.square(std), dim=1))


def upright_exp(
    env: ManagerBasedRLEnv,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    terrain_sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Reward keeping the selected body upright relative to world up or local terrain normal."""
    asset: RigidObject = env.scene[asset_cfg.name]
    terrain_normal = _terrain_normal_from_raycaster(env, terrain_sensor_cfg)
    if terrain_normal is not None:
        tilt_error = _body_frame_xy_error(asset, asset_cfg, terrain_normal)
    else:
        gravity_w = asset.data.GRAVITY_VEC_W
        if gravity_w.ndim == 1:
            gravity_w = gravity_w.unsqueeze(0).expand(env.num_envs, -1)
        elif gravity_w.ndim > 2:
            gravity_w = gravity_w.reshape(-1, gravity_w.shape[-1])
        if gravity_w.ndim == 2 and gravity_w.shape[0] == 1:
            gravity_w = gravity_w.expand(env.num_envs, -1)
        elif gravity_w.ndim == 2 and gravity_w.shape[0] != env.num_envs:
            gravity_w = gravity_w[-env.num_envs :]
        tilt_error = _body_frame_xy_error(asset, asset_cfg, gravity_w)
    return torch.exp(-tilt_error / std**2)


def foot_clearance_reward(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    std: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward swing feet for clearing a target height on flat terrain."""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_height_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2))
    reward = torch.exp(-torch.sum(foot_height_error * foot_velocity_tanh, dim=1) / std)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_clearance(
    env: ManagerBasedRLEnv,
    target_height: float,
    command_name: str | None,
    command_threshold: float,
    asset_cfg: SceneEntityCfg,
    height_sensor_name: str | None = None,
) -> torch.Tensor:
    """Penalize foot clearance error while feet are moving, approximating mjlab's foot-height sensor term."""
    asset: RigidObject = env.scene[asset_cfg.name]
    terrain_z = _terrain_reference_height(env, height_sensor_name)
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    if terrain_z is not None:
        foot_height = foot_height - terrain_z
    foot_vel_xy = torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=-1)
    reward = torch.sum(torch.abs(foot_height - target_height) * foot_vel_xy, dim=1)
    reward *= _command_activity(env, command_name, command_threshold)
    return reward


def feet_gait_pattern(
    env: ManagerBasedRLEnv,
    period: float,
    offset: list[float],
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.5,
    command_name: str | None = None,
) -> torch.Tensor:
    """Reward a commanded biped contact schedule using per-foot phase offsets."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    is_contact = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids] > 0

    global_phase = ((env.episode_length_buf * env.step_dt) % period / period).unsqueeze(1)
    leg_phase = torch.cat([((global_phase + offset_) % 1.0) for offset_ in offset], dim=-1)

    reward = torch.zeros(env.num_envs, dtype=torch.float, device=env.device)
    for index in range(len(sensor_cfg.body_ids)):
        is_stance = leg_phase[:, index] < threshold
        reward += (~(is_stance ^ is_contact[:, index])).float()

    if command_name is not None:
        reward *= _base_motion_command_norm(env, command_name) > 0.1

    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


class feet_swing_height(ManagerTermBase):
    """Penalize deviation from target swing height at landing."""

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        asset_cfg: SceneEntityCfg = cfg.params["asset_cfg"]
        asset: RigidObject = env.scene[asset_cfg.name]
        if isinstance(asset_cfg.body_ids, slice):
            num_feet = asset.num_bodies
        else:
            num_feet = len(asset_cfg.body_ids)
        self.peak_heights = torch.zeros((env.num_envs, num_feet), device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg,
        asset_cfg: SceneEntityCfg,
        target_height: float,
        command_name: str,
        command_threshold: float,
        height_sensor_name: str | None = None,
    ) -> torch.Tensor:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        asset: RigidObject = env.scene[asset_cfg.name]
        terrain_z = _terrain_reference_height(env, height_sensor_name)
        foot_heights = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
        if terrain_z is not None:
            foot_heights = foot_heights - terrain_z

        contacts = (
            contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
            .norm(dim=-1)
            .max(dim=1)[0]
            > 1.0
        )
        in_air = ~contacts
        self.peak_heights = torch.where(in_air, torch.maximum(self.peak_heights, foot_heights), self.peak_heights)
        first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
        error = self.peak_heights / max(float(target_height), 1.0e-6) - 1.0
        reward = torch.sum(torch.square(error) * first_contact.float(), dim=1)
        reward *= _command_activity(env, command_name, command_threshold)
        self.peak_heights = torch.where(first_contact, torch.zeros_like(self.peak_heights), self.peak_heights)
        return reward

    def reset(self, env_ids: torch.Tensor):
        self.peak_heights[env_ids] = 0.0


def soft_landing(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str | None = None,
    command_threshold: float = 0.05,
) -> torch.Tensor:
    """Penalize high normal force on touchdown."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1)
    reward = torch.sum(forces * first_contact.float(), dim=1)
    reward *= _command_activity(env, command_name, command_threshold)
    return reward


def angular_momentum_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Approximate mjlab's whole-body angular momentum penalty with body angular-velocity energy."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_ang_vel = asset.data.body_ang_vel_w[:, asset_cfg.body_ids, :]
    return torch.sum(torch.square(body_ang_vel), dim=(1, 2))


def body_ang_vel_xy_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Mjlab-style body angular velocity penalty for roll/pitch rates only."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_ang_vel = asset.data.body_ang_vel_w[:, asset_cfg.body_ids, :]
    return torch.sum(torch.square(body_ang_vel[..., :2]), dim=tuple(range(1, body_ang_vel.ndim)))


def self_collision_cost(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 10.0,
) -> torch.Tensor:
    """Count recent filtered self-contact force violations."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    data = contact_sensor.data
    if data.force_matrix_w_history is not None:
        forces = data.force_matrix_w_history[:, :, sensor_cfg.body_ids, :, :]
        hits = forces.norm(dim=-1) > force_threshold
        return hits.any(dim=-1).any(dim=-1).sum(dim=1).float()
    forces = data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    hits = forces.norm(dim=-1) > force_threshold
    return hits.any(dim=-1).sum(dim=1).float()


def joint_velocity_soft_limits_l2(
    env: ManagerBasedRLEnv,
    velocity_limits: tuple[float, ...],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_excess_ratio: float = 2.0,
) -> torch.Tensor:
    """Penalize normalized joint-speed excess without imposing a solver velocity clamp."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_vel = asset.data.joint_vel[:, asset_cfg.joint_ids]
    if len(velocity_limits) != joint_vel.shape[1]:
        raise ValueError(
            f"Expected one velocity limit per selected joint ({joint_vel.shape[1]}), got {len(velocity_limits)}."
        )
    limits = torch.as_tensor(velocity_limits, device=joint_vel.device, dtype=joint_vel.dtype)
    if torch.any(limits <= 0.0):
        raise ValueError(f"Velocity limits must be positive, got {velocity_limits}.")
    excess_ratio = (torch.abs(joint_vel) / limits - 1.0).clamp_(min=0.0, max=max_excess_ratio)
    return torch.sum(torch.square(excess_ratio), dim=1)


def joint_pos_limits_with_soft_factor(
    env: ManagerBasedRLEnv,
    soft_factor: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize positions outside a selected fraction of each joint's hard range.

    This task-level margin is intentionally independent of the articulation's
    global ``soft_joint_pos_limit_factor``. Composite robots can therefore keep
    the D1 gripper's measured endpoints while applying the standard Go2 10%
    safety margin only to the selected leg joints.
    """
    if not 0.0 < float(soft_factor) <= 1.0:
        raise ValueError(f"soft_factor must be in (0, 1], got {soft_factor}.")
    asset: Articulation = env.scene[asset_cfg.name]
    hard_limits = asset.data.default_joint_pos_limits[:, asset_cfg.joint_ids]
    center = 0.5 * (hard_limits[..., 0] + hard_limits[..., 1])
    half_range = 0.5 * (hard_limits[..., 1] - hard_limits[..., 0]) * float(soft_factor)
    lower = center - half_range
    upper = center + half_range
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    below = (lower - joint_pos).clamp_min(0.0)
    above = (joint_pos - upper).clamp_min(0.0)
    return torch.sum(below + above, dim=1)


def feet_slip(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    command_threshold: float = 0.01,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Mjlab-style foot slip penalty: squared horizontal foot speed while in contact."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]
    contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1) > 1.0
    foot_vel_xy = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    cost = torch.sum(torch.sum(torch.square(foot_vel_xy), dim=-1) * contacts.float(), dim=1)
    cost *= _command_activity(env, command_name, command_threshold)
    return cost


def wheel_vel_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    command_name: str,
    velocity_threshold: float,
    command_threshold: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: Articulation = env.scene[asset_cfg.name]
    cmd = _base_motion_command_norm(env, command_name)
    body_vel = torch.linalg.norm(asset.data.root_lin_vel_b[:, :2], dim=1)
    joint_vel = torch.abs(asset.data.joint_vel[:, asset_cfg.joint_ids])
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    in_air = contact_sensor.compute_first_air(env.step_dt)[:, sensor_cfg.body_ids]
    running_reward = torch.sum(in_air * joint_vel, dim=1)
    standing_reward = torch.sum(joint_vel, dim=1)
    reward = torch.where(
        torch.logical_or(cmd > command_threshold, body_vel > velocity_threshold),
        running_reward,
        standing_reward,
    )
    return reward


class GaitReward(ManagerTermBase):
    """Gait enforcing reward term for quadrupeds.

    This reward penalizes contact timing differences between selected foot pairs
    defined in :attr:`synced_feet_pair_names` to bias the policy towards a desired gait,
    i.e trotting, bounding, or pacing. Note that this reward is only for quadrupedal gaits
    with two pairs of synchronized feet.
    """

    def __init__(self, cfg: RewTerm, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.std: float = cfg.params["std"]
        self.command_name: str = cfg.params["command_name"]
        self.max_err: float = cfg.params["max_err"]
        self.velocity_threshold: float = cfg.params["velocity_threshold"]
        self.command_threshold: float = cfg.params["command_threshold"]
        self.contact_sensor: ContactSensor = env.scene.sensors[cfg.params["sensor_cfg"].name]
        self.asset: Articulation = env.scene[cfg.params["asset_cfg"].name]
        # match foot body names with corresponding foot body ids
        synced_feet_pair_names = cfg.params["synced_feet_pair_names"]
        if (
            len(synced_feet_pair_names) != 2
            or len(synced_feet_pair_names[0]) != 2
            or len(synced_feet_pair_names[1]) != 2
        ):
            raise ValueError("This reward only supports gaits with two pairs of synchronized feet, like trotting.")
        synced_feet_pair_0 = self.contact_sensor.find_bodies(synced_feet_pair_names[0])[0]
        synced_feet_pair_1 = self.contact_sensor.find_bodies(synced_feet_pair_names[1])[0]
        self.synced_feet_pairs = [synced_feet_pair_0, synced_feet_pair_1]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        std: float,
        command_name: str,
        max_err: float,
        velocity_threshold: float,
        command_threshold: float,
        synced_feet_pair_names,
        asset_cfg: SceneEntityCfg,
        sensor_cfg: SceneEntityCfg,
    ) -> torch.Tensor:
        """Compute the reward.

        This reward is defined as a multiplication between six terms where two of them enforce pair feet
        being in sync and the other four rewards if all the other remaining pairs are out of sync

        Args:
            env: The RL environment instance.
        Returns:
            The reward value.
        """
        # for synchronous feet, the contact (air) times of two feet should match
        sync_reward_0 = self._sync_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[0][1])
        sync_reward_1 = self._sync_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[1][1])
        sync_reward = sync_reward_0 * sync_reward_1
        # for asynchronous feet, the contact time of one foot should match the air time of the other one
        async_reward_0 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][0])
        async_reward_1 = self._async_reward_func(self.synced_feet_pairs[0][1], self.synced_feet_pairs[1][1])
        async_reward_2 = self._async_reward_func(self.synced_feet_pairs[0][0], self.synced_feet_pairs[1][1])
        async_reward_3 = self._async_reward_func(self.synced_feet_pairs[1][0], self.synced_feet_pairs[0][1])
        async_reward = async_reward_0 * async_reward_1 * async_reward_2 * async_reward_3
        # only enforce gait if cmd > 0
        cmd = _base_motion_command_norm(env, self.command_name)
        body_vel = torch.linalg.norm(self.asset.data.root_com_lin_vel_b[:, :2], dim=1)
        reward = torch.where(
            torch.logical_or(cmd > self.command_threshold, body_vel > self.velocity_threshold),
            sync_reward * async_reward,
            0.0,
        )
        reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
        return reward

    """
    Helper functions.
    """

    def _sync_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between the most recent air time and contact time of synced feet pairs.
        se_air = torch.clip(torch.square(air_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        se_contact = torch.clip(torch.square(contact_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_air + se_contact) / self.std)

    def _async_reward_func(self, foot_0: int, foot_1: int) -> torch.Tensor:
        """Reward anti-synchronization of two feet."""
        air_time = self.contact_sensor.data.current_air_time
        contact_time = self.contact_sensor.data.current_contact_time
        # penalize the difference between opposing contact modes air time of feet 1 to contact time of feet 2
        # and contact time of feet 1 to air time of feet 2) of feet pairs that are not in sync with each other.
        se_act_0 = torch.clip(torch.square(air_time[:, foot_0] - contact_time[:, foot_1]), max=self.max_err**2)
        se_act_1 = torch.clip(torch.square(contact_time[:, foot_0] - air_time[:, foot_1]), max=self.max_err**2)
        return torch.exp(-(se_act_0 + se_act_1) / self.std)


def joint_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "joint_mirror_joints_cache") or env.joint_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.joint_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.joint_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(asset.data.joint_pos[:, joint_pair[0][0]] - asset.data.joint_pos[:, joint_pair[1][0]]),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_mirror(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, mirror_joints: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    if not hasattr(env, "action_mirror_joints_cache") or env.action_mirror_joints_cache is None:
        # Cache joint positions for all pairs
        env.action_mirror_joints_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_pair] for joint_pair in mirror_joints
        ]
    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over all joint pairs
    for joint_pair in env.action_mirror_joints_cache:
        # Calculate the difference for each pair and add to the total reward
        diff = torch.sum(
            torch.square(
                torch.abs(env.action_manager.action[:, joint_pair[0][0]])
                - torch.abs(env.action_manager.action[:, joint_pair[1][0]])
            ),
            dim=-1,
        )
        reward += diff
    reward *= 1 / len(mirror_joints) if len(mirror_joints) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_sync(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, joint_groups: list[list[str]]) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # Cache joint indices if not already done
    if not hasattr(env, "action_sync_joint_cache") or env.action_sync_joint_cache is None:
        env.action_sync_joint_cache = [
            [asset.find_joints(joint_name) for joint_name in joint_group] for joint_group in joint_groups
        ]

    reward = torch.zeros(env.num_envs, device=env.device)
    # Iterate over each joint group
    for joint_group in env.action_sync_joint_cache:
        if len(joint_group) < 2:
            continue  # need at least 2 joints to compare

        # Get absolute actions for all joints in this group
        actions = torch.stack(
            [torch.abs(env.action_manager.action[:, joint[0]]) for joint in joint_group], dim=1
        )  # shape: (num_envs, num_joints_in_group)

        # Calculate mean action for each environment
        mean_actions = torch.mean(actions, dim=1, keepdim=True)

        # Calculate variance from mean for each joint
        variance = torch.mean(torch.square(actions - mean_actions), dim=1)

        # Add to reward (we want to minimize this variance)
        reward += variance.squeeze()
    reward *= 1 / len(joint_groups) if len(joint_groups) > 0 else 0
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= _base_motion_command_norm(env, command_name) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= _base_motion_command_norm(env, command_name) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_air_time_variance_penalty(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize variance in the amount of time each foot spends in the air/on the ground relative to each other"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    reward = torch.var(torch.clip(last_air_time, max=0.5), dim=1) + torch.var(
        torch.clip(last_contact_time, max=0.5), dim=1
    )
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact(
    env: ManagerBasedRLEnv, command_name: str, expect_contact_num: int, sensor_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    contact_num = torch.sum(contact, dim=1)
    reward = (contact_num != expect_contact_num).float()
    # no reward for zero command
    reward *= _base_motion_command_norm(env, command_name) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_contact_without_cmd(env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward feet contact"""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    reward = torch.sum(contact, dim=-1).float()
    reward *= _base_motion_command_norm(env, command_name) < 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_stumble(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces_z = torch.abs(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, 2])
    forces_xy = torch.linalg.norm(contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :2], dim=2)
    # Penalize feet hitting vertical surfaces
    reward = torch.any(forces_xy > 4 * forces_z, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_edge_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    asset_cfg: SceneEntityCfg,
    height_sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    edge_distances_cm: tuple[float, ...] = (2.5, 5.0),
    edge_weights: tuple[float, ...] = (1.0, 0.5),
    edge_height_threshold: float = 0.04,
    contact_force_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize contacts near terrain edges using local height-scan gradients."""
    if len(edge_distances_cm) != len(edge_weights):
        raise ValueError("edge_distances_cm and edge_weights must have the same length.")

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    height_sensor: RayCaster = env.scene[height_sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]

    # Contact state for feet.
    contacts = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
        > contact_force_threshold
    )  # (num_envs, n_feet)
    n_feet = contacts.shape[1]

    # Reconstruct local height grid from scanner rays.
    ray_hits_z = torch.nan_to_num(height_sensor.data.ray_hits_w[..., 2], nan=0.0, posinf=0.0, neginf=0.0)
    num_envs, num_rays = ray_hits_z.shape
    cache_key = "_feet_edge_grid_shape_cache"
    grid_shape = getattr(env, cache_key, None)
    if grid_shape is None or grid_shape[0] * grid_shape[1] != num_rays:
        grid_shape = _infer_grid_shape_from_sensor(height_sensor, num_rays)
        setattr(env, cache_key, grid_shape)
    rows, cols = grid_shape
    height_map = ray_hits_z.view(num_envs, rows, cols)

    # Edge mask from local height discontinuities.
    dx = torch.abs(height_map[:, 1:, :] - height_map[:, :-1, :])
    dy = torch.abs(height_map[:, :, 1:] - height_map[:, :, :-1])
    edge_strength = torch.zeros_like(height_map)
    edge_strength[:, 1:, :] = torch.maximum(edge_strength[:, 1:, :], dx)
    edge_strength[:, :-1, :] = torch.maximum(edge_strength[:, :-1, :], dx)
    edge_strength[:, :, 1:] = torch.maximum(edge_strength[:, :, 1:], dy)
    edge_strength[:, :, :-1] = torch.maximum(edge_strength[:, :, :-1], dy)
    edge_mask = edge_strength > float(edge_height_threshold)
    edge_mask_f = edge_mask.float().unsqueeze(1)

    # Foot positions in yaw-aligned base frame to index scanner grid.
    feet_pos_w_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    feet_pos_w_xy = feet_pos_w_xy[:, :n_feet, :]
    root_pos_w_xy = asset.data.root_pos_w[:, :2]
    feet_rel_xy = feet_pos_w_xy - root_pos_w_xy.unsqueeze(1)
    feet_rel = torch.zeros((num_envs, n_feet, 3), device=env.device)
    feet_rel[..., :2] = feet_rel_xy
    yaw_quat_w = yaw_quat(asset.data.root_quat_w)
    feet_rel_local = quat_apply_inverse(
        yaw_quat_w.unsqueeze(1).expand(-1, n_feet, -1).reshape(-1, 4),
        feet_rel.reshape(-1, 3),
    ).view(num_envs, n_feet, 3)

    pattern_cfg = getattr(height_sensor.cfg, "pattern_cfg", None)
    resolution = float(getattr(pattern_cfg, "resolution", 0.1)) if pattern_cfg is not None else 0.1
    size = getattr(pattern_cfg, "size", (float(rows) * resolution, float(cols) * resolution))
    size_x = float(size[0]) if isinstance(size, (list, tuple)) and len(size) == 2 else float(rows) * resolution
    size_y = float(size[1]) if isinstance(size, (list, tuple)) and len(size) == 2 else float(cols) * resolution
    resolution = max(resolution, 1.0e-6)
    idx_x = torch.clamp(torch.round((feet_rel_local[..., 0] + 0.5 * size_x) / resolution).long(), 0, rows - 1)
    idx_y = torch.clamp(torch.round((feet_rel_local[..., 1] + 0.5 * size_y) / resolution).long(), 0, cols - 1)
    env_idx = torch.arange(num_envs, device=env.device).unsqueeze(1).expand(-1, n_feet)

    penalty = torch.zeros(num_envs, device=env.device)
    for distance_cm, weight in zip(edge_distances_cm, edge_weights):
        if weight == 0.0:
            continue
        radius_cells = max(1, int(math.ceil((float(distance_cm) * 0.01) / resolution)))
        near_edge = F.max_pool2d(
            edge_mask_f, kernel_size=2 * radius_cells + 1, stride=1, padding=radius_cells
        ).squeeze(1)
        near_edge_at_foot = near_edge[env_idx, idx_x, idx_y]
        penalty += float(weight) * torch.sum(contacts.float() * near_edge_at_foot, dim=1)

    # Keep consistency with other locomotion rewards by down-weighting fallen states.
    penalty *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return penalty


def feet_distance_y_exp(
    env: ManagerBasedRLEnv, stance_width: float, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)
    n_feet = len(asset_cfg.body_ids)
    footsteps_in_body_frame = torch.zeros(env.num_envs, n_feet, 3, device=env.device)
    for i in range(n_feet):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )
    side_sign = torch.tensor(
        [1.0 if i % 2 == 0 else -1.0 for i in range(n_feet)],
        device=env.device,
    )
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    desired_ys = stance_width_tensor / 2 * side_sign.unsqueeze(0)
    stance_diff = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / (std**2))
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_distance_xy_exp(
    env: ManagerBasedRLEnv,
    stance_width: float,
    stance_length: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    asset: RigidObject = env.scene[asset_cfg.name]

    # Compute the current footstep positions relative to the root
    cur_footsteps_translated = asset.data.body_link_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_link_pos_w[
        :, :
    ].unsqueeze(1)

    footsteps_in_body_frame = torch.zeros(env.num_envs, 4, 3, device=env.device)
    for i in range(4):
        footsteps_in_body_frame[:, i, :] = math_utils.quat_apply(
            math_utils.quat_conjugate(asset.data.root_link_quat_w), cur_footsteps_translated[:, i, :]
        )

    # Desired x and y positions for each foot
    stance_width_tensor = stance_width * torch.ones([env.num_envs, 1], device=env.device)
    stance_length_tensor = stance_length * torch.ones([env.num_envs, 1], device=env.device)

    desired_xs = torch.cat(
        [stance_length_tensor / 2, stance_length_tensor / 2, -stance_length_tensor / 2, -stance_length_tensor / 2],
        dim=1,
    )
    desired_ys = torch.cat(
        [stance_width_tensor / 2, -stance_width_tensor / 2, stance_width_tensor / 2, -stance_width_tensor / 2], dim=1
    )

    # Compute differences in x and y
    stance_diff_x = torch.square(desired_xs - footsteps_in_body_frame[:, :, 0])
    stance_diff_y = torch.square(desired_ys - footsteps_in_body_frame[:, :, 1])

    # Combine x and y differences and compute the exponential penalty
    stance_diff = stance_diff_x + stance_diff_y
    reward = torch.exp(-torch.sum(stance_diff, dim=1) / std**2)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def legs_distance(
    env: ManagerBasedRLEnv,
    min_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg(
        "robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    ),
) -> torch.Tensor:
    """Penalize front/rear left-right feet being closer than a minimum lateral distance."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    foot_pos_rel_w = foot_pos_w - asset.data.root_pos_w.unsqueeze(1)
    foot_pos_b = quat_apply_inverse(
        asset.data.root_quat_w.repeat_interleave(4, dim=0),
        foot_pos_rel_w.reshape(-1, 3),
    ).reshape(env.num_envs, 4, 3)

    front_distance = torch.abs(foot_pos_b[:, 0, 1] - foot_pos_b[:, 1, 1])
    rear_distance = torch.abs(foot_pos_b[:, 2, 1] - foot_pos_b[:, 3, 1])
    front_penalty = torch.square(torch.clamp(float(min_distance) - front_distance, min=0.0))
    rear_penalty = torch.square(torch.clamp(float(min_distance) - rear_distance, min=0.0))
    return front_penalty + rear_penalty


def feet_height(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_z_target_error = torch.square(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - target_height)
    foot_velocity_tanh = torch.tanh(
        tanh_mult * torch.linalg.norm(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2], dim=2)
    )
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    # no reward for zero command
    reward *= _base_motion_command_norm(env, command_name) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_height_body(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    target_height: float,
    tanh_mult: float,
) -> torch.Tensor:
    """Reward the swinging feet for clearing a specified height off the ground"""
    asset: RigidObject = env.scene[asset_cfg.name]
    cur_footpos_translated = asset.data.body_pos_w[:, asset_cfg.body_ids, :] - asset.data.root_pos_w[:, :].unsqueeze(1)
    footpos_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footpos_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footpos_translated[:, i, :]
        )
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_z_target_error = torch.square(footpos_in_body_frame[:, :, 2] - target_height).view(env.num_envs, -1)
    foot_velocity_tanh = torch.tanh(tanh_mult * torch.norm(footvel_in_body_frame[:, :, :2], dim=2))
    reward = torch.sum(foot_z_target_error * foot_velocity_tanh, dim=1)
    reward *= _base_motion_command_norm(env, command_name) > 0.1
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def feet_slide(
    env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize feet sliding.

    This function penalizes the agent for sliding its feet on the ground. The reward is computed as the
    norm of the linear velocity of the feet multiplied by a binary contact sensor. This ensures that the
    agent is penalized only when the feet are in contact with the ground.
    """
    # Penalize feet sliding
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > 1.0
    asset: RigidObject = env.scene[asset_cfg.name]

    # feet_vel = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]
    # reward = torch.sum(feet_vel.norm(dim=-1) * contacts, dim=1)

    cur_footvel_translated = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :] - asset.data.root_lin_vel_w[
        :, :
    ].unsqueeze(1)
    footvel_in_body_frame = torch.zeros(env.num_envs, len(asset_cfg.body_ids), 3, device=env.device)
    for i in range(len(asset_cfg.body_ids)):
        footvel_in_body_frame[:, i, :] = math_utils.quat_apply_inverse(
            asset.data.root_quat_w, cur_footvel_translated[:, i, :]
        )
    foot_leteral_vel = torch.sqrt(torch.sum(torch.square(footvel_in_body_frame[:, :, :2]), dim=2)).view(
        env.num_envs, -1
    )
    reward = torch.sum(foot_leteral_vel * contacts, dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def action_smoothness_l2(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize second-order finite differences of the action sequence."""
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action

    cache_key = "_action_smoothness_prev_prev_action"
    prev_prev_action = getattr(env, cache_key, None)
    if prev_prev_action is None or prev_prev_action.shape != prev_action.shape:
        prev_prev_action = torch.zeros_like(prev_action)

    diff = action - 2.0 * prev_action + prev_prev_action

    # Ignore the first two steps of each episode.
    valid_prev = torch.any(prev_action != 0.0, dim=1, keepdim=True)
    valid_prev_prev = torch.any(prev_prev_action != 0.0, dim=1, keepdim=True)
    diff = diff * valid_prev * valid_prev_prev
    reward = torch.sum(torch.square(diff), dim=1)

    # Update history for the next step and clear fresh episodes.
    updated_prev_prev = prev_action.detach().clone()
    if hasattr(env, "episode_length_buf"):
        reset_mask = env.episode_length_buf <= 1
        if torch.any(reset_mask):
            updated_prev_prev[reset_mask] = 0.0
    setattr(env, cache_key, updated_prev_prev)
    return reward


def action_rate_l2_after_reset(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Penalize first-order action changes after one valid action exists."""
    diff = env.action_manager.action - env.action_manager.prev_action
    reward = torch.sum(torch.square(diff), dim=1)
    if hasattr(env, "episode_length_buf"):
        reward = reward * (env.episode_length_buf > 1).to(dtype=reward.dtype)
    return reward


def action_rate_l2_selected(env: ManagerBasedRLEnv, action_indices: tuple[int, ...]) -> torch.Tensor:
    """Penalize changes in selected policy outputs without imposing a motion limit."""
    action = env.action_manager.action
    valid_indices = tuple(index for index in action_indices if 0 <= index < action.shape[1])
    if not valid_indices:
        return torch.zeros(env.num_envs, device=env.device)
    indices = torch.as_tensor(valid_indices, device=env.device, dtype=torch.long)
    diff = (action - env.action_manager.prev_action).index_select(1, indices)
    reward = torch.sum(torch.square(diff), dim=1)
    if hasattr(env, "episode_length_buf"):
        reward = reward * (env.episode_length_buf > 1).to(dtype=reward.dtype)
    return reward


def action_l2_selected(env: ManagerBasedRLEnv, action_indices: tuple[int, ...]) -> torch.Tensor:
    """Penalize the magnitude of selected normalized policy outputs."""
    action = env.action_manager.action
    valid_indices = tuple(index for index in action_indices if 0 <= index < action.shape[1])
    if not valid_indices:
        return torch.zeros(env.num_envs, device=env.device)
    indices = torch.as_tensor(valid_indices, device=env.device, dtype=torch.long)
    return torch.sum(torch.square(action.index_select(1, indices)), dim=1)


def action_smoothness_l2_selected(env: ManagerBasedRLEnv, action_indices: tuple[int, ...]) -> torch.Tensor:
    """Penalize selected action curvature while leaving fast motion available when useful."""
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    valid_indices = tuple(index for index in action_indices if 0 <= index < action.shape[1])
    if not valid_indices:
        return torch.zeros(env.num_envs, device=env.device)
    cache_key = "_action_smoothness_prev_prev_selected_" + "_".join(str(index) for index in action_indices)
    prev_prev_action = getattr(env, cache_key, None)
    if prev_prev_action is None or prev_prev_action.shape != prev_action.shape:
        prev_prev_action = torch.zeros_like(prev_action)

    indices = torch.as_tensor(valid_indices, device=env.device, dtype=torch.long)
    diff = (action - 2.0 * prev_action + prev_prev_action).index_select(1, indices)
    valid_prev = torch.any(prev_action != 0.0, dim=1)
    valid_prev_prev = torch.any(prev_prev_action != 0.0, dim=1)
    reward = torch.sum(torch.square(diff), dim=1) * valid_prev * valid_prev_prev

    updated_prev_prev = prev_action.detach().clone()
    if hasattr(env, "episode_length_buf"):
        reset_mask = env.episode_length_buf <= 1
        if torch.any(reset_mask):
            updated_prev_prev[reset_mask] = 0.0
    setattr(env, cache_key, updated_prev_prev)
    return reward


def _ground_height_from_sensor(env: ManagerBasedRLEnv, sensor_cfg: SceneEntityCfg | None) -> torch.Tensor:
    if sensor_cfg is None:
        return torch.zeros(env.num_envs, device=env.device)
    sensor: RayCaster = env.scene[sensor_cfg.name]
    ray_hits = sensor.data.ray_hits_w[..., 2]
    valid = torch.isfinite(ray_hits) & (torch.abs(ray_hits) < 1.0e6)
    safe_hits = torch.where(valid, ray_hits, torch.zeros_like(ray_hits))
    counts = valid.sum(dim=1).clamp(min=1)
    return safe_hits.sum(dim=1) / counts


def correct_base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize base height error using the terrain estimate, matching go2_rl_gym's correct_base_height."""
    asset: RigidObject = env.scene[asset_cfg.name]
    base_height = asset.data.root_pos_w[:, 2] - _ground_height_from_sensor(env, sensor_cfg)
    return torch.square(base_height - target_height)


def feet_regulation(
    env: ManagerBasedRLEnv,
    target_base_height: float,
    asset_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize fast low swing feet; higher foot clearance reduces the penalty as in go2_rl_gym."""
    asset: RigidObject = env.scene[asset_cfg.name]
    ground_height = _ground_height_from_sensor(env, sensor_cfg).unsqueeze(1)
    feet_height = torch.clamp(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] - ground_height, min=0.0)
    feet_xy_vel_sq = torch.sum(torch.square(asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :2]), dim=-1)
    height_scale = max(1.0e-6, 0.025 * float(target_base_height))
    return torch.sum(feet_xy_vel_sq * torch.exp(-feet_height / height_scale), dim=1)


def upward(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(1 - asset.data.projected_gravity_b[:, 2])
    return reward


def base_height_l2(
    env: ManagerBasedRLEnv,
    target_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        ray_hits = sensor.data.ray_hits_w[..., 2]
        if torch.isnan(ray_hits).any() or torch.isinf(ray_hits).any() or torch.max(torch.abs(ray_hits)) > 1e6:
            adjusted_target_height = asset.data.root_link_pos_w[:, 2]
        else:
            adjusted_target_height = target_height + torch.mean(ray_hits, dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    reward = torch.square(asset.data.root_pos_w[:, 2] - adjusted_target_height)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def lin_vel_z_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize z-axis base linear velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.square(asset.data.root_lin_vel_b[:, 2])
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def undesired_contacts(env: ManagerBasedRLEnv, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize undesired contacts as the number of violations that are above a threshold."""
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # check if contact force is above threshold
    net_contact_forces = contact_sensor.data.net_forces_w_history
    is_contact = torch.max(torch.norm(net_contact_forces[:, :, sensor_cfg.body_ids], dim=-1), dim=1)[0] > threshold
    # sum over contacts for each environment
    reward = torch.sum(is_contact, dim=1).float()
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward


def flat_orientation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize non-flat base orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    reward = torch.sum(torch.square(asset.data.projected_gravity_b[:, :2]), dim=1)
    reward *= torch.clamp(-env.scene["robot"].data.projected_gravity_b[:, 2], 0, 0.7) / 0.7
    return reward
