from __future__ import annotations

from typing import TYPE_CHECKING

import torch
import torch.nn.functional as F

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def cts_teacher_role(
    env: ManagerBasedEnv,
    num_teacher_envs: int | None = None,
    teacher_fraction: float = 0.75,
) -> torch.Tensor:
    """Return 1 for rollout slots driven by the CTS teacher and 0 for student slots."""
    if num_teacher_envs is None:
        num_teacher_envs = int(env.num_envs * float(teacher_fraction))
    num_teacher_envs = max(0, min(int(num_teacher_envs), env.num_envs))
    role = torch.zeros((env.num_envs, 1), device=env.device)
    if num_teacher_envs > 0:
        role[:num_teacher_envs, 0] = 1.0
    return role


def joint_pos_rel_without_wheel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    wheel_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """The joint positions of the asset w.r.t. the default joint positions.(Without the wheel joints)"""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos_rel = asset.data.joint_pos[:, asset_cfg.joint_ids] - asset.data.default_joint_pos[:, asset_cfg.joint_ids]
    joint_pos_rel[:, wheel_asset_cfg.joint_ids] = 0
    return joint_pos_rel


def arm_joint_target_rel(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_attr: str = "_arm_joint_target_pos",
) -> torch.Tensor:
    """Return scripted arm joint targets relative to the asset default joint positions."""
    asset: Articulation = env.scene[asset_cfg.name]
    if getattr(asset_cfg, "joint_ids", None) == slice(None):
        joint_ids = torch.arange(asset.num_joints, device=asset.device, dtype=torch.long)
    elif getattr(asset_cfg, "joint_ids", None) is None:
        joint_result = asset.find_joints(asset_cfg.joint_names)
        joint_ids = torch.as_tensor(
            joint_result[0] if isinstance(joint_result, tuple) else joint_result,
            device=asset.device,
            dtype=torch.long,
        )
    else:
        joint_ids = torch.as_tensor(asset_cfg.joint_ids, device=asset.device, dtype=torch.long)
    if joint_ids.numel() == 0:
        return torch.zeros((env.num_envs, 0), device=env.device)

    target = getattr(env, target_attr, None)
    if target is None:
        return torch.zeros((env.num_envs, joint_ids.numel()), device=env.device)
    return target - asset.data.default_joint_pos[:, joint_ids]


def arm_joint_target_error(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_attr: str = "_arm_joint_target_pos",
) -> torch.Tensor:
    """Return scripted arm target error in joint space."""
    asset: Articulation = env.scene[asset_cfg.name]
    if getattr(asset_cfg, "joint_ids", None) == slice(None):
        joint_ids = torch.arange(asset.num_joints, device=asset.device, dtype=torch.long)
    elif getattr(asset_cfg, "joint_ids", None) is None:
        joint_result = asset.find_joints(asset_cfg.joint_names)
        joint_ids = torch.as_tensor(
            joint_result[0] if isinstance(joint_result, tuple) else joint_result,
            device=asset.device,
            dtype=torch.long,
        )
    else:
        joint_ids = torch.as_tensor(asset_cfg.joint_ids, device=asset.device, dtype=torch.long)
    if joint_ids.numel() == 0:
        return torch.zeros((env.num_envs, 0), device=env.device)

    target = getattr(env, target_attr, None)
    if target is None:
        return torch.zeros((env.num_envs, joint_ids.numel()), device=env.device)
    current = asset.data.joint_pos[:, joint_ids]
    return target - current


def arm_ee_target_pos_b(
    env: ManagerBasedEnv,
    target_attr: str = "_arm_ee_target_pos",
) -> torch.Tensor:
    """Return scripted arm end-effector target position in the robot base frame."""
    target = getattr(env, target_attr, None)
    if target is None:
        return torch.zeros((env.num_envs, 3), device=env.device)
    return target


def frozen_wbc_low_level_actions(env: ManagerBasedEnv, action_name: str = "wbc_command") -> torch.Tensor:
    """Return the previous low-level action emitted by a frozen WBC action term."""
    return env.action_manager.get_term(action_name).low_level_actions


def frozen_wbc_processed_actions(env: ManagerBasedEnv, action_name: str = "wbc_command") -> torch.Tensor:
    """Return the high-level commands actually applied after command shaping."""
    return env.action_manager.get_term(action_name).processed_actions


def _ground_height_from_sensor(env: ManagerBasedEnv, sensor_cfg: SceneEntityCfg | None) -> torch.Tensor:
    if sensor_cfg is None:
        return torch.zeros(env.num_envs, device=env.device)
    sensor: RayCaster = env.scene[sensor_cfg.name]
    ray_hits = sensor.data.ray_hits_w[..., 2]
    valid = torch.isfinite(ray_hits) & (torch.abs(ray_hits) < 1.0e6)
    safe_hits = torch.where(valid, ray_hits, torch.zeros_like(ray_hits))
    counts = valid.sum(dim=1).clamp(min=1)
    return safe_hits.sum(dim=1) / counts


def base_height(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Return base height above the estimated terrain height."""
    asset: Articulation | RigidObject = env.scene[asset_cfg.name]
    height = asset.data.root_pos_w[:, 2] - _ground_height_from_sensor(env, sensor_cfg)
    return height.unsqueeze(-1)


def _resolved_body_ids(body_ids, width: int, device: str) -> torch.Tensor:
    if body_ids == slice(None):
        return torch.arange(width, device=device, dtype=torch.long)
    return torch.as_tensor(body_ids, device=device, dtype=torch.long)


def feet_air_stance_time(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
) -> torch.Tensor:
    """Return current air and stance timers for selected contact bodies."""
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    air_time = getattr(sensor.data, "current_air_time", None)
    stance_time = getattr(sensor.data, "current_contact_time", None)
    if air_time is None:
        width = sensor.data.net_forces_w.shape[1]
        air_time = torch.zeros((env.num_envs, width), device=env.device)
    if stance_time is None:
        stance_time = torch.zeros_like(air_time)
    body_ids = _resolved_body_ids(sensor_cfg.body_ids, air_time.shape[1], env.device)
    if body_ids.numel() == 0:
        return torch.zeros((env.num_envs, 0), device=env.device)
    return torch.cat((air_time[:, body_ids], stance_time[:, body_ids]), dim=-1)


def contact_forces_b(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    normalize: float = 100.0,
) -> torch.Tensor:
    """Return selected contact forces in the base frame, flattened for privileged critic use."""
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    asset: Articulation | RigidObject = env.scene[asset_cfg.name]
    body_ids = _resolved_body_ids(sensor_cfg.body_ids, sensor.data.net_forces_w.shape[1], env.device)
    if body_ids.numel() == 0:
        return torch.zeros((env.num_envs, 0), device=env.device)

    forces_w = sensor.data.net_forces_w[:, body_ids, :]
    root_quat_w = asset.data.root_quat_w.unsqueeze(1).expand(-1, forces_w.shape[1], -1)
    forces_b = quat_apply_inverse(root_quat_w.reshape(-1, 4), forces_w.reshape(-1, 3)).reshape(forces_w.shape)
    return (forces_b / float(normalize)).reshape(env.num_envs, -1)


def arm_payload_mass(env: ManagerBasedEnv) -> torch.Tensor:
    """Return randomized end-effector payload mass for privileged diagnostics / critics."""
    payload_mass = getattr(env, "_arm_payload_mass", None)
    if payload_mass is None:
        return torch.zeros((env.num_envs, 1), device=env.device)
    return payload_mass.unsqueeze(-1)


def arm_ee_target_error_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names="link06"),
    target_attr: str = "_arm_ee_target_pos",
) -> torch.Tensor:
    """Return scripted arm end-effector target error in the robot base frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    if getattr(asset_cfg, "body_ids", None) is None:
        body_result = asset.find_bodies(asset_cfg.body_names)
        body_ids = torch.as_tensor(
            body_result[0] if isinstance(body_result, tuple) else body_result,
            device=asset.device,
            dtype=torch.long,
        )
    else:
        body_ids = torch.as_tensor(asset_cfg.body_ids, device=asset.device, dtype=torch.long)
    if body_ids.numel() == 0:
        return torch.zeros((env.num_envs, 3), device=env.device)

    target = getattr(env, target_attr, None)
    if target is None:
        return torch.zeros((env.num_envs, 3), device=env.device)

    ee_body_id = int(body_ids[0].item())
    ee_pose_w = asset.data.body_pose_w[:, ee_body_id]
    root_pose_w = asset.data.root_pose_w
    ee_pos_b = quat_apply_inverse(root_pose_w[:, 3:7], ee_pose_w[:, 0:3] - root_pose_w[:, 0:3])
    return target - ee_pos_b


def arm_motion_state(env: ManagerBasedEnv) -> torch.Tensor:
    """Return privileged scripted-arm curriculum state: enabled, difficulty, and stage."""
    def _state_column(attr_name: str, default: float) -> torch.Tensor:
        value = getattr(env, attr_name, default)
        if isinstance(value, torch.Tensor):
            return value.to(device=env.device, dtype=torch.float32).reshape(-1)[: env.num_envs]
        return torch.full((env.num_envs,), float(value), device=env.device, dtype=torch.float32)

    enabled = _state_column("_loco_manip_arm_motion_enabled", 0.0)
    difficulty = _state_column("_loco_manip_arm_motion_difficulty", 0.0)
    stage = _state_column("_loco_manip_training_stage", 0.0)
    return torch.stack((enabled, difficulty, stage), dim=-1)


def velocity_posture_commands(env: ManagerBasedRLEnv, command_name: str = "base_velocity") -> torch.Tensor:
    """Return velocity plus any posture command dimensions exposed by the command term."""
    command_term = env.command_manager.get_term(command_name)
    velocity = getattr(command_term, "vel_command_b", None)
    if velocity is None:
        velocity = env.command_manager.get_command(command_name)
    posture = getattr(command_term, "posture_command", None)
    if posture is None:
        posture = torch.zeros((env.num_envs, 2), device=env.device)
    return torch.cat((velocity[..., :3], posture), dim=-1)


def arm_joint_trajectory_state(env: ManagerBasedEnv) -> torch.Tensor:
    """Return normalized scripted-arm trajectory progress and duration."""
    progress = getattr(env, "_arm_joint_trajectory_progress", None)
    duration = getattr(env, "_arm_joint_trajectory_duration", None)
    if progress is None or duration is None:
        return torch.zeros((env.num_envs, 2), device=env.device)

    duration_safe = duration.clamp_min(1.0e-6)
    progress_fraction = (progress / duration_safe).clamp(0.0, 1.0)
    return torch.stack((progress_fraction, duration), dim=-1)


def arm_ee_trajectory_state(env: ManagerBasedEnv) -> torch.Tensor:
    """Return normalized task-space arm trajectory progress and duration."""
    progress = getattr(env, "_arm_trajectory_progress", None)
    duration = getattr(env, "_arm_trajectory_duration", None)
    if progress is None or duration is None:
        return torch.zeros((env.num_envs, 2), device=env.device)

    duration_safe = duration.clamp_min(1.0e-6)
    progress_fraction = (progress / duration_safe).clamp(0.0, 1.0)
    return torch.stack((progress_fraction, duration), dim=-1)


def arm_apex_reference_state(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    target_attr: str = "_arm_ik_joint_target_pos",
    ee_target_attr: str = "_arm_ee_target_pos",
    ee_goal_attr: str = "_arm_ee_goal_pos",
    include_joint_pos: bool = True,
    include_joint_error: bool = True,
    include_ee_target: bool = True,
    include_ee_error: bool = True,
    include_ee_goal: bool = True,
    include_trajectory: bool = True,
    add_noise: bool = False,
    noise_std: float = 0.01,
) -> torch.Tensor:
    """Return an online APEX-style arm reference bundle from the IK target stream.

    This is the WBC equivalent of a reference-motion observation, but the reference
    is generated online by the current task-space target and differential IK hint.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if getattr(asset_cfg, "joint_ids", None) == slice(None):
        joint_ids = torch.arange(asset.num_joints, device=asset.device, dtype=torch.long)
    elif getattr(asset_cfg, "joint_ids", None) is None:
        joint_result = asset.find_joints(asset_cfg.joint_names)
        joint_ids = torch.as_tensor(
            joint_result[0] if isinstance(joint_result, tuple) else joint_result,
            device=asset.device,
            dtype=torch.long,
        )
    else:
        joint_ids = torch.as_tensor(asset_cfg.joint_ids, device=asset.device, dtype=torch.long)

    target = getattr(env, target_attr, None)
    if target is None:
        target = torch.zeros((env.num_envs, joint_ids.numel()), device=env.device)
    current = asset.data.joint_pos[:, joint_ids] if joint_ids.numel() > 0 else target

    features: list[torch.Tensor] = []
    if include_joint_pos:
        default = asset.data.default_joint_pos[:, joint_ids] if joint_ids.numel() > 0 else target
        features.append(target - default)
    if include_joint_error:
        features.append(target - current)

    ee_target = getattr(env, ee_target_attr, None)
    if ee_target is None:
        ee_target = torch.zeros((env.num_envs, 3), device=env.device)
    if include_ee_target:
        features.append(ee_target)
    if include_ee_error:
        body_ids = getattr(asset_cfg, "body_ids", None)
        if body_ids is None:
            body_result = asset.find_bodies(asset_cfg.body_names)
            body_ids = body_result[0] if isinstance(body_result, tuple) else body_result
        body_ids = torch.as_tensor(body_ids, device=asset.device, dtype=torch.long)
        if body_ids.numel() == 0:
            features.append(torch.zeros((env.num_envs, 3), device=env.device))
        else:
            ee_body_id = int(body_ids[0].item())
            ee_pose_w = asset.data.body_pose_w[:, ee_body_id]
            root_pose_w = asset.data.root_pose_w
            ee_pos_b = quat_apply_inverse(root_pose_w[:, 3:7], ee_pose_w[:, 0:3] - root_pose_w[:, 0:3])
            features.append(ee_target - ee_pos_b)
    if include_ee_goal:
        ee_goal = getattr(env, ee_goal_attr, None)
        if ee_goal is None:
            ee_goal = torch.zeros((env.num_envs, 3), device=env.device)
        features.append(ee_goal)
    if include_trajectory:
        features.append(arm_ee_trajectory_state(env))

    if len(features) == 0:
        raise ValueError("arm_apex_reference_state needs at least one included feature.")

    reference_state = torch.cat(features, dim=-1)
    if add_noise:
        reference_state = reference_state + torch.randn_like(reference_state) * float(noise_std)
    return reference_state


def body_contact_state(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=""),
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Binary contact state for selected bodies."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    return (contacts > float(force_threshold)).float()


def body_contact_force_norms(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=""),
    scale: float = 100.0,
) -> torch.Tensor:
    """Current contact force magnitudes for selected bodies."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1)
    return forces / float(scale)


def phase(env: ManagerBasedRLEnv, cycle_time: float) -> torch.Tensor:
    if not hasattr(env, "episode_length_buf") or env.episode_length_buf is None:
        env.episode_length_buf = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    phase = env.episode_length_buf[:, None] * env.step_dt / cycle_time
    phase_tensor = torch.cat([torch.sin(2 * torch.pi * phase), torch.cos(2 * torch.pi * phase)], dim=-1)
    return phase_tensor


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
    """Return flattened depth image features from a configured camera sensor."""
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


def lidar_scan_distances(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    max_distance: float | None = None,
    normalize: bool = True,
    sample_count: int | None = None,
) -> torch.Tensor:
    """Return flattened lidar hit distances from a RayCaster-style lidar sensor."""

    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    distances = getattr(sensor.data, "distances", None)
    if distances is None:
        distances = getattr(sensor.data, "ray_hits_distance", None)
    if distances is None:
        relative_hits = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
        distances = torch.linalg.norm(relative_hits, dim=-1)

    if max_distance is None:
        max_distance = float(getattr(sensor.cfg, "max_distance", 0.0))
    max_distance = float(max_distance) if max_distance is not None else 0.0
    fill_value = max_distance if max_distance > 0.0 else 0.0
    distances = torch.nan_to_num(distances, nan=fill_value, posinf=fill_value, neginf=0.0)
    if max_distance > 0.0:
        distances = distances.clamp(0.0, max_distance)

    if sample_count is not None and int(sample_count) > 0 and distances.shape[1] > int(sample_count):
        indices = torch.linspace(
            0,
            distances.shape[1] - 1,
            steps=int(sample_count),
            device=distances.device,
        ).long()
        distances = distances.index_select(dim=1, index=indices)

    if normalize and max_distance > 0.0:
        distances = distances / max_distance - 0.5

    return distances


def lidar_point_cloud_features(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("lidar"),
    max_distance: float | None = None,
    sample_count: int | None = 512,
    include_distance: bool = True,
) -> torch.Tensor:
    """Return sampled lidar hit points in the sensor frame for a PointNet-style encoder."""

    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    pointcloud = getattr(sensor.data, "pointcloud", None)
    if pointcloud is not None and not getattr(sensor.cfg, "pointcloud_in_world_frame", False):
        num_envs, num_rays, _ = pointcloud.shape
        if max_distance is None:
            max_distance = float(getattr(sensor.cfg, "max_distance", 0.0))
        max_distance = max(float(max_distance) if max_distance is not None else 0.0, 1.0e-6)
        distances = getattr(sensor.data, "distances", None)
        if distances is None:
            distances = torch.linalg.norm(pointcloud, dim=-1)
        distances = distances.unsqueeze(-1).clamp(0.0, max_distance)
        valid = torch.isfinite(pointcloud).all(dim=-1, keepdim=True) & torch.isfinite(distances)
        pointcloud = torch.where(valid, pointcloud, torch.zeros_like(pointcloud))
        distances = torch.where(valid, distances, torch.full_like(distances, max_distance))
        point_features = pointcloud / max_distance
        if include_distance:
            point_features = torch.cat((point_features, distances / max_distance), dim=-1)
        if sample_count is not None and int(sample_count) > 0 and num_rays > int(sample_count):
            indices = torch.linspace(0, num_rays - 1, steps=int(sample_count), device=point_features.device).long()
            point_features = point_features.index_select(dim=1, index=indices)
        return point_features.reshape(point_features.shape[0], -1)

    ray_hits_w = sensor.data.ray_hits_w
    sensor_pos_w = sensor.data.pos_w
    sensor_quat_w = sensor.data.quat_w
    num_envs, num_rays, _ = ray_hits_w.shape

    relative_hits_w = ray_hits_w - sensor_pos_w.unsqueeze(1)
    sensor_quat = sensor_quat_w
    if getattr(sensor.cfg, "ray_alignment", "base") == "yaw":
        sensor_quat = yaw_quat(sensor_quat)
    relative_hits_s = quat_apply_inverse(
        sensor_quat.unsqueeze(1).expand(num_envs, num_rays, 4).reshape(num_envs * num_rays, 4),
        relative_hits_w.reshape(num_envs * num_rays, 3),
    ).reshape(num_envs, num_rays, 3)

    if max_distance is None:
        max_distance = float(getattr(sensor.cfg, "max_distance", 0.0))
    max_distance = max(float(max_distance) if max_distance is not None else 0.0, 1.0e-6)

    distances = torch.linalg.norm(relative_hits_s, dim=-1, keepdim=True)
    valid = torch.isfinite(relative_hits_s).all(dim=-1, keepdim=True) & torch.isfinite(distances)
    relative_hits_s = torch.where(valid, relative_hits_s, torch.zeros_like(relative_hits_s))
    distances = torch.where(valid, distances.clamp(0.0, max_distance), torch.full_like(distances, max_distance))

    point_features = relative_hits_s / max_distance
    if include_distance:
        point_features = torch.cat((point_features, distances / max_distance), dim=-1)

    if sample_count is not None and int(sample_count) > 0 and num_rays > int(sample_count):
        indices = torch.linspace(0, num_rays - 1, steps=int(sample_count), device=point_features.device).long()
        point_features = point_features.index_select(dim=1, index=indices)

    return point_features.reshape(point_features.shape[0], -1)


def _infer_scan_grid_shape(sensor: RayCaster, num_rays: int) -> tuple[int, int]:
    """Infer the scan grid shape from pattern metadata and ray count."""
    pattern_cfg = getattr(sensor.cfg, "pattern_cfg", None)
    if pattern_cfg is not None and hasattr(pattern_cfg, "size") and hasattr(pattern_cfg, "resolution"):
        size = getattr(pattern_cfg, "size")
        resolution = float(getattr(pattern_cfg, "resolution"))
        if isinstance(size, (list, tuple)) and len(size) == 2 and resolution > 0.0:
            nx = max(1, int(round(float(size[0]) / resolution)))
            ny = max(1, int(round(float(size[1]) / resolution)))
            if nx * ny == num_rays:
                return nx, ny
            if ny * nx == num_rays:
                return ny, nx
            if (nx + 1) * ny == num_rays:
                return nx + 1, ny
            if nx * (ny + 1) == num_rays:
                return nx, ny + 1
    # fallback to near-square factorization
    best_rows, best_cols = num_rays, 1
    best_gap = num_rays
    for divisor in range(1, int(num_rays**0.5) + 1):
        if num_rays % divisor != 0:
            continue
        rows = num_rays // divisor
        cols = divisor
        gap = abs(rows - cols)
        if gap < best_gap:
            best_rows, best_cols = rows, cols
            best_gap = gap
    return best_rows, best_cols


def _get_scan_grid_shape(sensor: RayCaster, num_rays: int) -> tuple[int, int]:
    cache_attr = "_Gurukul_scan_grid_shape_cache"
    grid_shape = getattr(sensor, cache_attr, None)
    if grid_shape is None or grid_shape[0] * grid_shape[1] != num_rays:
        grid_shape = _infer_scan_grid_shape(sensor, num_rays)
        setattr(sensor, cache_attr, grid_shape)
    return grid_shape


def _get_height_scan_map(sensor: RayCaster, offset: float = 0.5) -> tuple[torch.Tensor, int, int]:
    ray_hits_z = torch.nan_to_num(sensor.data.ray_hits_w[..., 2], nan=0.0, posinf=0.0, neginf=0.0)
    sensor_z = sensor.data.pos_w[:, 2].unsqueeze(1)
    height_flat = sensor_z - ray_hits_z - float(offset)
    num_envs, num_rays = height_flat.shape
    rows, cols = _get_scan_grid_shape(sensor, num_rays)
    return height_flat.view(num_envs, rows, cols), rows, cols


def _get_scan_resolution(sensor: RayCaster) -> float:
    pattern_cfg = getattr(sensor.cfg, "pattern_cfg", None)
    resolution = float(getattr(pattern_cfg, "resolution", 0.05)) if pattern_cfg is not None else 0.05
    return max(resolution, 1.0e-6)


def _compute_terrain_normals(
    height_map: torch.Tensor, step_x: float, step_y: float
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    padded = F.pad(height_map.unsqueeze(1), (1, 1, 1, 1), mode="replicate").squeeze(1)
    dz_dx = (padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1]) / (2.0 * float(step_x))
    dz_dy = (padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2]) / (2.0 * float(step_y))
    normal = torch.stack((-dz_dx, -dz_dy, torch.ones_like(height_map)), dim=-1)
    return F.normalize(normal, dim=-1), dz_dx, dz_dy


def terrain_normal_scan(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    offset: float = 0.5,
) -> torch.Tensor:
    """Estimate local terrain normals from the ray-cast height grid."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    height_map, _, _ = _get_height_scan_map(sensor, offset=offset)
    resolution = _get_scan_resolution(sensor)
    normals, _, _ = _compute_terrain_normals(height_map, resolution, resolution)
    return normals.reshape(normals.shape[0], -1)


def elevation_map(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    noise: bool = False,
    height_noise_std: float = 0.03,
    height_offset_range: tuple[float, float] = (-0.05, 0.05),
    min_height: float = -1.2,
    max_height: float = 0.0,
) -> torch.Tensor:
    """Return flattened local XYZ terrain ray-hit coordinates for AME."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    relative_pos_w = sensor.data.ray_hits_w - sensor.data.pos_w.unsqueeze(1)
    sensor_quat = sensor.data.quat_w
    num_envs, num_rays, _ = relative_pos_w.shape

    if getattr(sensor.cfg, "ray_alignment", "base") == "yaw":
        sensor_quat = yaw_quat(sensor_quat)

    sensor_quat = sensor_quat.unsqueeze(1).expand(num_envs, num_rays, 4).reshape(num_envs * num_rays, 4)
    sensor_coords = quat_apply_inverse(sensor_quat, relative_pos_w.reshape(num_envs * num_rays, 3))
    sensor_coords = torch.nan_to_num(sensor_coords.reshape(num_envs, num_rays, 3))

    if noise:
        offset_attr = "_Gurukul_elevation_map_offset"
        offset = getattr(env, offset_attr, None)
        if offset is None or offset.shape != (num_envs, 1):
            offset = torch.zeros((num_envs, 1), device=env.device)
            setattr(env, offset_attr, offset)
        if hasattr(env, "reset_buf"):
            reset_env_ids = env.reset_buf.nonzero(as_tuple=False).squeeze(-1)
            if len(reset_env_ids) > 0:
                low, high = height_offset_range
                offset[reset_env_ids] = torch.rand((len(reset_env_ids), 1), device=env.device) * (high - low) + low
        sensor_coords[..., 2] += torch.randn_like(sensor_coords[..., 2]) * float(height_noise_std)
        sensor_coords[..., 2] += offset

    sensor_coords[..., 2] = torch.clamp(sensor_coords[..., 2], min=float(min_height), max=float(max_height))
    return sensor_coords.reshape(num_envs, num_rays * 3)


def terrain_traversability_scan(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    offset: float = 0.5,
    slope_weight: float = 2.0,
    roughness_weight: float = 18.0,
    edge_weight: float = 24.0,
    smoothing_kernel_size: int = 3,
    normalize_distribution: bool = True,
    min_probability: float = 1.0e-6,
) -> torch.Tensor:
    """Build a traversability distribution from local slope, roughness, and edge discontinuities."""
    sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
    height_map, _, _ = _get_height_scan_map(sensor, offset=offset)
    resolution = _get_scan_resolution(sensor)
    normals, dz_dx, dz_dy = _compute_terrain_normals(height_map, resolution, resolution)

    slope = torch.sqrt(dz_dx.square() + dz_dy.square())
    up_alignment = normals[..., 2].clamp(0.0, 1.0)

    kernel = max(1, int(smoothing_kernel_size))
    if kernel % 2 == 0:
        kernel += 1
    height_map_4d = height_map.unsqueeze(1)
    local_mean = F.avg_pool2d(height_map_4d, kernel_size=kernel, stride=1, padding=kernel // 2)
    local_var = F.avg_pool2d((height_map_4d - local_mean).square(), kernel_size=kernel, stride=1, padding=kernel // 2)
    roughness = torch.sqrt(local_var.squeeze(1) + 1.0e-8)

    padded = F.pad(height_map_4d, (1, 1, 1, 1), mode="replicate").squeeze(1)
    center = padded[:, 1:-1, 1:-1]
    edge_strength = torch.maximum(
        torch.maximum((center - padded[:, 2:, 1:-1]).abs(), (center - padded[:, :-2, 1:-1]).abs()),
        torch.maximum((center - padded[:, 1:-1, 2:]).abs(), (center - padded[:, 1:-1, :-2]).abs()),
    )

    traversability = up_alignment * torch.exp(
        -(float(slope_weight) * slope + float(roughness_weight) * roughness + float(edge_weight) * edge_strength)
    )
    traversability = traversability.clamp_min(float(min_probability))
    traversability = traversability.reshape(traversability.shape[0], -1)

    if normalize_distribution:
        traversability = traversability / traversability.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)

    return traversability


def feet_contact_state(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=""),
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Binary foot contact state used as privileged supervision target."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    return (contacts > float(force_threshold)).float()


def _terrain_reference_height(env: ManagerBasedEnv, height_sensor_name: str | None) -> torch.Tensor | None:
    if height_sensor_name is None:
        return None
    try:
        sensor = env.scene.sensors[height_sensor_name]
    except (KeyError, AttributeError):
        try:
            sensor = env.scene[height_sensor_name]
        except (KeyError, AttributeError):
            return None
    if sensor is None or not hasattr(sensor, "data") or not hasattr(sensor.data, "ray_hits_w"):
        return None
    ray_hits = torch.nan_to_num(sensor.data.ray_hits_w[..., 2], nan=0.0, posinf=0.0, neginf=0.0)
    return torch.mean(ray_hits, dim=1, keepdim=True)


def feet_height_observation(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=""),
    height_sensor_name: str | None = None,
) -> torch.Tensor:
    """Foot heights relative to the local terrain reference when available."""
    asset: RigidObject = env.scene[asset_cfg.name]
    foot_height = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    terrain_z = _terrain_reference_height(env, height_sensor_name)
    if terrain_z is not None:
        foot_height = foot_height - terrain_z
    return foot_height


def feet_air_time_observation(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=""),
) -> torch.Tensor:
    """Current foot air times from the contact sensor."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    return contact_sensor.data.current_air_time[:, sensor_cfg.body_ids]


def feet_contact_observation(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=""),
    force_threshold: float = 1.0,
) -> torch.Tensor:
    """Current binary foot contact state."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contacts = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :].norm(dim=-1)
    return (contacts > float(force_threshold)).float()


def feet_contact_force_observation(
    env: ManagerBasedEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=""),
) -> torch.Tensor:
    """Current foot contact force vectors flattened per environment."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    return forces.reshape(forces.shape[0], -1)


def feet_heightmap_scan(
    env: ManagerBasedEnv,
    height_sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=""),
    patch_radius: float = 0.1,
    offset: float = 0.5,
) -> torch.Tensor:
    """Sample local body-frame heightmap patches around each foot from the global scanner grid."""
    height_sensor: RayCaster = env.scene.sensors[height_sensor_cfg.name]
    asset: RigidObject = env.scene[asset_cfg.name]

    height_map, rows, cols = _get_height_scan_map(height_sensor, offset=offset)
    num_envs = height_map.shape[0]

    n_feet = len(asset_cfg.body_ids)
    feet_pos_w_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
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
    resolution = float(getattr(pattern_cfg, "resolution", 0.05)) if pattern_cfg is not None else 0.05
    size = getattr(pattern_cfg, "size", (float(rows) * resolution, float(cols) * resolution))
    if isinstance(size, (list, tuple)) and len(size) == 2:
        size_x, size_y = float(size[0]), float(size[1])
    else:
        size_x, size_y = float(rows) * resolution, float(cols) * resolution
    resolution = max(resolution, 1.0e-6)

    idx_x = torch.clamp(torch.round((feet_rel_local[..., 0] + 0.5 * size_x) / resolution).long(), 0, rows - 1)
    idx_y = torch.clamp(torch.round((feet_rel_local[..., 1] + 0.5 * size_y) / resolution).long(), 0, cols - 1)

    radius_cells = max(1, int(round(float(patch_radius) / resolution)))
    height_pad = F.pad(height_map, (radius_cells, radius_cells, radius_cells, radius_cells), mode="replicate")

    env_idx = torch.arange(num_envs, device=env.device).unsqueeze(1).expand(-1, n_feet)
    idx_x = idx_x + radius_cells
    idx_y = idx_y + radius_cells

    patch_terms: list[torch.Tensor] = []
    for dx in range(-radius_cells, radius_cells + 1):
        for dy in range(-radius_cells, radius_cells + 1):
            patch_terms.append(height_pad[env_idx, idx_x + dx, idx_y + dy])
    patches = torch.stack(patch_terms, dim=-1)
    return patches.reshape(num_envs, -1)


def foot_pos_b(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=""),
) -> torch.Tensor:
    """Foot positions in yaw-aligned base frame, flattened as [num_envs, num_feet * 3]."""
    asset: RigidObject = env.scene[asset_cfg.name]
    feet_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    root_pos_w = asset.data.root_pos_w
    feet_rel = feet_pos_w - root_pos_w.unsqueeze(1)
    yaw_quat_w = yaw_quat(asset.data.root_quat_w)
    num_envs, n_feet = feet_rel.shape[:2]
    feet_rel_local = quat_apply_inverse(
        yaw_quat_w.unsqueeze(1).expand(-1, n_feet, -1).reshape(-1, 4),
        feet_rel.reshape(-1, 3),
    ).view(num_envs, n_feet, 3)
    return feet_rel_local.reshape(num_envs, -1)


def contact_trail_base_pose(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Base pose for contact trail warping: [pos_w(3), quat_w(4)]."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.cat((asset.data.root_pos_w, asset.data.root_quat_w), dim=-1)


def contact_trail_foot_features(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=""),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=""),
    contact_force_threshold: float = 1.0,
    force_scale: float = 200.0,
    velocity_scale: float = 2.0,
    action_scale: float = 1.0,
    ang_vel_scale: float = 4.0,
    num_actions_per_foot: int = 3,
) -> torch.Tensor:
    """Per-foot contact event features for policy-side contact trail writes.

    Feature layout per foot (11 dims):
        contact_flag, normal_force, tangential_force,
        foot_vel_x/y/z_local, foot_speed_tangent,
        previous_action_leg_norm, base_ang_vel_x/y/z
    """
    asset: RigidObject = env.scene[asset_cfg.name]
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]

    forces_w = contact_sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    force_norm = forces_w.norm(dim=-1)
    contact_flag = (force_norm > float(contact_force_threshold)).float()

    normal_force = forces_w[..., 2].abs().clamp(max=force_scale) / force_scale
    tangential_force = forces_w[..., :2].norm(dim=-1).clamp(max=force_scale) / force_scale

    foot_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids, :]
    yaw_quat_w = yaw_quat(asset.data.root_quat_w)
    num_envs, n_feet = foot_vel_w.shape[:2]
    foot_vel_b = quat_apply_inverse(
        yaw_quat_w.unsqueeze(1).expand(-1, n_feet, -1).reshape(-1, 4),
        foot_vel_w.reshape(-1, 3),
    ).view(num_envs, n_feet, 3)
    foot_vel_b = foot_vel_b / velocity_scale
    foot_speed_tangent = foot_vel_b[..., :2].norm(dim=-1, keepdim=True)

    actions = env.action_manager.action
    actions_per_foot = actions.shape[-1] // max(n_feet, 1)
    if num_actions_per_foot > 0 and actions.shape[-1] >= n_feet * num_actions_per_foot:
        leg_actions = actions[:, : n_feet * num_actions_per_foot].view(num_envs, n_feet, num_actions_per_foot)
        action_leg_norm = leg_actions.norm(dim=-1, keepdim=True) / action_scale
    else:
        action_leg_norm = torch.zeros((num_envs, n_feet, 1), device=env.device)

    base_ang_vel = asset.data.root_ang_vel_b / ang_vel_scale
    base_ang_vel = base_ang_vel.unsqueeze(1).expand(-1, n_feet, -1)

    features = torch.cat(
        (
            contact_flag.unsqueeze(-1),
            normal_force.unsqueeze(-1),
            tangential_force.unsqueeze(-1),
            foot_vel_b,
            foot_speed_tangent,
            action_leg_norm,
            base_ang_vel,
        ),
        dim=-1,
    )
    return features.reshape(num_envs, -1)


def hidden_friction_at_feet(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=""),
) -> torch.Tensor:
    """Privileged friction lookup at each foot xy from hidden friction grid."""
    grid = getattr(env, "_hidden_friction_grid", None)
    if grid is None:
        n_feet = len(asset_cfg.body_ids) if asset_cfg.body_ids != slice(None) else 4
        return torch.ones((env.num_envs, n_feet), device=env.device)

    asset: RigidObject = env.scene[asset_cfg.name]
    feet_xy = asset.data.body_pos_w[:, asset_cfg.body_ids, :2]
    env_origins = env.scene.env_origins[:, :2].unsqueeze(1)
    local_xy = feet_xy - env_origins

    grid_size = grid.shape[-1]
    tile_size = float(getattr(env, "_hidden_friction_tile_size", 0.5))
    half_extent = 0.5 * grid_size * tile_size
    idx = torch.floor((local_xy + half_extent) / tile_size).long()
    idx = idx.clamp(0, grid_size - 1)

    env_ids = torch.arange(env.num_envs, device=env.device).unsqueeze(1).expand_as(idx[..., 0])
    friction = grid[env_ids, idx[..., 0], idx[..., 1]]
    return friction
