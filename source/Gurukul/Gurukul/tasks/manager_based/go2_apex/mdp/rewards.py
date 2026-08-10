from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_error_magnitude

from .commands import MotionCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_body_indexes(command: MotionCommand, body_names: list[str] | None) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if (body_names is None) or (name in body_names)]


def _get_foot_indexes(command: MotionCommand) -> list[int]:
    return [i for i, name in enumerate(command.cfg.body_names) if "foot" in name.lower()]


def _standing_still_mask(
    command: MotionCommand,
    lin_vel_threshold: float | None,
    ang_vel_threshold: float | None = None,
    lin_vel_z_threshold: float | None = None,
) -> torch.Tensor | None:
    """True when command is effectively stationary, optionally including vertical/yaw-rate gating."""
    if lin_vel_threshold is None or lin_vel_threshold < 0.0:
        return None

    standing_still = torch.norm(command.command_lin_vel_xy, dim=1) <= lin_vel_threshold
    if lin_vel_z_threshold is not None and lin_vel_z_threshold >= 0.0:
        standing_still = standing_still & (torch.abs(command.command_lin_vel_z[:, 0]) <= lin_vel_z_threshold)
    if ang_vel_threshold is None or ang_vel_threshold < 0.0:
        return standing_still
    return standing_still & (torch.abs(command.command_ang_vel_z[:, 0]) <= ang_vel_threshold)


def _update_default_foot_yaw_b_cache(
    env: ManagerBasedRLEnv,
    foot_indexes: list[int],
    robot_foot_pos_b: torch.Tensor,
) -> torch.Tensor:
    """Yaw-frame foot positions at default stance; one snapshot per episode (APEX reference)."""
    n_feet = len(foot_indexes)
    cache = getattr(env, "_go2_apex_default_foot_yaw_b", None)
    if cache is None or cache.shape != (env.num_envs, n_feet, 3):
        env._go2_apex_default_foot_yaw_b = torch.zeros(
            env.num_envs, n_feet, 3, device=env.device, dtype=robot_foot_pos_b.dtype
        )
    cached = getattr(env, "_go2_apex_default_foot_yaw_b_cached", None)
    if cached is None or cached.shape[0] != env.num_envs:
        env._go2_apex_default_foot_yaw_b_cached = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    ep_buf = getattr(env, "episode_length_buf", None)
    if ep_buf is not None:
        # Fresh episode: allow a new snapshot (Isaac Gym captures after first post-reset step).
        reset_like = ep_buf == 0
        if torch.any(reset_like):
            env._go2_apex_default_foot_yaw_b_cached[reset_like] = False
        snap = (ep_buf >= 1) & (~env._go2_apex_default_foot_yaw_b_cached)
        if torch.any(snap):
            env._go2_apex_default_foot_yaw_b[snap] = robot_foot_pos_b[snap].detach().clone()
            env._go2_apex_default_foot_yaw_b_cached[snap] = True
    return env._go2_apex_default_foot_yaw_b


def motion_base_position_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = torch.sum(torch.square(command.anchor_pos_w - command.robot_anchor_pos_w), dim=-1)
    return torch.exp(-error / std**2)


def motion_base_orientation_error_exp(env: ManagerBasedRLEnv, command_name: str, std: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = quat_error_magnitude(command.anchor_quat_w, command.robot_anchor_quat_w) ** 2
    return torch.exp(-error / std**2)


def motion_projected_gravity_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track reference roll/pitch by matching gravity projected into the reference and robot base frames."""
    asset: Articulation = env.scene[asset_cfg.name]
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref_projected_gravity_b = math_utils.quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)
    robot_projected_gravity_b = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)
    error = torch.sum(torch.square(ref_projected_gravity_b - robot_projected_gravity_b), dim=-1)
    return torch.exp(-error / std**2)


def motion_joint_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    stand_still_vel_threshold: float | None = 0.1,
    stand_still_ang_vel_threshold: float | None = None,
    stand_still_lin_vel_z_threshold: float | None = None,
    joint_names: list[str] | None = None,
) -> torch.Tensor:
    """Exponential joint tracking; stationary commands target the default pose."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    q_ref = command.joint_pos
    q_robot = command.robot_joint_pos
    if q_ref.shape[1] != q_robot.shape[1]:
        q_robot = q_robot[:, : q_ref.shape[1]]

    mask = _standing_still_mask(
        command,
        stand_still_vel_threshold,
        stand_still_ang_vel_threshold,
        stand_still_lin_vel_z_threshold,
    )
    if mask is None:
        q_target = q_ref
    else:
        q_default = command.robot.data.default_joint_pos.to(device=q_ref.device, dtype=q_ref.dtype)
        if q_default.shape[1] != q_ref.shape[1]:
            q_default = q_default[:, : q_ref.shape[1]]
        q_target = torch.where(mask.unsqueeze(-1), q_default, q_ref)

    if joint_names is not None:
        motion_joint_names = getattr(command, "robot_motion_joint_names", None)
        if motion_joint_names is None:
            raise ValueError("joint_names filtering requires named motion-command robot joints.")
        name_to_idx = {name: idx for idx, name in enumerate(motion_joint_names)}
        missing = [name for name in joint_names if name not in name_to_idx]
        if missing:
            raise ValueError(f"Requested reward joints are missing in motion-command joints: {missing}")
        joint_indices = torch.tensor([name_to_idx[name] for name in joint_names], dtype=torch.long, device=q_ref.device)
        q_target = q_target[:, joint_indices]
        q_robot = q_robot[:, joint_indices]

    error = torch.mean(torch.square(q_target - q_robot), dim=-1)
    return torch.exp(-error / std**2)


def motion_joint_position_error_exp_per_joint(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    joint_names: list[str],
) -> torch.Tensor:
    """Average independent joint kernels so one bad joint cannot erase every arm-tracking gradient."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_joint_names = getattr(command, "robot_motion_joint_names", None)
    if motion_joint_names is None:
        raise ValueError("Per-joint motion tracking requires named motion-command robot joints.")
    name_to_idx = {name: idx for idx, name in enumerate(motion_joint_names)}
    missing = [name for name in joint_names if name not in name_to_idx]
    if missing:
        raise ValueError(f"Requested per-joint reward joints are missing in motion-command joints: {missing}")

    joint_indices = torch.tensor(
        [name_to_idx[name] for name in joint_names],
        dtype=torch.long,
        device=env.device,
    )
    error_sq = torch.square(command.joint_pos[:, joint_indices] - command.robot_joint_pos[:, joint_indices])
    return torch.mean(torch.exp(-error_sq / std**2), dim=-1)


def motion_body_position_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_pos_relative_w[:, body_indexes] - command.robot_body_pos_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_body_orientation_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = (
        quat_error_magnitude(command.body_quat_relative_w[:, body_indexes], command.robot_body_quat_w[:, body_indexes])
        ** 2
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_body_linear_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_lin_vel_w[:, body_indexes] - command.robot_body_lin_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


def motion_body_angular_velocity_error_exp(
    env: ManagerBasedRLEnv, command_name: str, std: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    body_indexes = _get_body_indexes(command, body_names)
    error = torch.sum(
        torch.square(command.body_ang_vel_w[:, body_indexes] - command.robot_body_ang_vel_w[:, body_indexes]), dim=-1
    )
    return torch.exp(-error.mean(-1) / std**2)


# Backward-compatible aliases for older Go2 APEX configs.
motion_global_anchor_position_error_exp = motion_base_position_error_exp
motion_global_anchor_orientation_error_exp = motion_base_orientation_error_exp
motion_relative_body_position_error_exp = motion_body_position_error_exp
motion_relative_body_orientation_error_exp = motion_body_orientation_error_exp
motion_global_body_linear_velocity_error_exp = motion_body_linear_velocity_error_exp
motion_global_body_angular_velocity_error_exp = motion_body_angular_velocity_error_exp


def motion_command_tracking_lin_vel_exp(
    env: ManagerBasedRLEnv, command_name: str, sigma: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Track imitation-command planar velocity using the Isaac Gym exponential form."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    lin_vel_error = torch.sum(torch.square(command.command_lin_vel_xy - command.robot_command_lin_vel_xy), dim=1)
    return torch.exp(-lin_vel_error / sigma)


def motion_command_tracking_lin_vel_z_exp(
    env: ManagerBasedRLEnv, command_name: str, sigma: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Track reference vertical velocity using the Isaac Gym exponential form."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    lin_vel_z_error = torch.square(command.command_lin_vel_z[:, 0] - command.robot_command_lin_vel_z[:, 0])
    return torch.exp(-lin_vel_z_error / sigma)


def motion_command_tracking_ang_vel_exp(
    env: ManagerBasedRLEnv, command_name: str, sigma: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Track imitation-command yaw rate using the Isaac Gym exponential form."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    ang_vel_error = torch.square(command.command_ang_vel_z[:, 0] - command.robot_command_ang_vel_z[:, 0])
    return torch.exp(-ang_vel_error / sigma)


def joint_position_command_tracking_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Track a sampled joint-position command."""
    command = env.command_manager.get_term(command_name)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    error = torch.mean(torch.square(joint_pos - command.command), dim=-1)
    return torch.exp(-error / sigma)


def joint_velocity_soft_limits_l2(
    env: ManagerBasedRLEnv,
    velocity_limits: tuple[float, ...],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    max_excess_ratio: float = 2.0,
) -> torch.Tensor:
    """Penalize normalized joint-speed excess without enforcing a physics velocity cap."""
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


def motion_foot_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma: float,
    stand_still_vel_threshold: float | None = 0.1,
    stand_still_ang_vel_threshold: float | None = None,
    stand_still_lin_vel_z_threshold: float | None = None,
) -> torch.Tensor:
    """Yaw-aligned foot tracking; stationary commands target default-stance feet."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    foot_indexes = _get_foot_indexes(command)
    if not foot_indexes:
        return torch.zeros(env.num_envs, device=env.device)

    ref_foot_pos = command.body_pos_w[:, foot_indexes] - command.anchor_pos_w[:, None, :]
    robot_foot_pos = command.robot_body_pos_w[:, foot_indexes] - command.robot_anchor_pos_w[:, None, :]
    ref_yaw_quat = math_utils.yaw_quat(command.anchor_quat_w)
    robot_yaw_quat = math_utils.yaw_quat(command.robot_anchor_quat_w)

    ref_foot_pos_b = torch.zeros_like(ref_foot_pos)
    robot_foot_pos_b = torch.zeros_like(robot_foot_pos)
    for foot_id in range(len(foot_indexes)):
        ref_foot_pos_b[:, foot_id, :] = math_utils.quat_apply_inverse(ref_yaw_quat, ref_foot_pos[:, foot_id, :])
        robot_foot_pos_b[:, foot_id, :] = math_utils.quat_apply_inverse(robot_yaw_quat, robot_foot_pos[:, foot_id, :])

    default_foot_b = _update_default_foot_yaw_b_cache(env, foot_indexes, robot_foot_pos_b)
    mask = _standing_still_mask(
        command,
        stand_still_vel_threshold,
        stand_still_ang_vel_threshold,
        stand_still_lin_vel_z_threshold,
    )
    if mask is None:
        ref_target_b = ref_foot_pos_b
    else:
        ref_target_b = torch.where(mask[:, None, None], default_foot_b, ref_foot_pos_b)

    end_effector_error = torch.sum(torch.square(ref_target_b - robot_foot_pos_b), dim=(1, 2))
    return torch.exp(-end_effector_error / sigma)


def motion_arm_ee_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["link06"]),
    align_to_robot_yaw: bool = True,
) -> torch.Tensor:
    """Track the reference arm end-effector position stored in the motion NPZ."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref_ee_pos_w = command.arm_ee_pos_w
    if ref_ee_pos_w is None:
        return torch.zeros(env.num_envs, device=env.device)

    asset: Articulation = env.scene[asset_cfg.name]
    # A physical gripper endpoint is naturally represented by the midpoint of
    # its two finger bodies. A single-body configuration remains unchanged.
    robot_ee_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :].mean(dim=1)

    if align_to_robot_yaw:
        ref_ee_pos_w = command.arm_ee_target_pos_w

    error = torch.sum(torch.square(ref_ee_pos_w - robot_ee_pos_w), dim=-1)
    return torch.exp(-error / sigma)


def motion_arm_ee_orientation_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["Link6"]),
) -> torch.Tensor:
    """Track the yaw-aligned D1 Link6 reference orientation."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    ref_ee_quat_w = command.arm_ee_target_quat_w
    if ref_ee_quat_w is None:
        return torch.zeros(env.num_envs, device=env.device)

    asset: Articulation = env.scene[asset_cfg.name]
    if len(asset_cfg.body_ids) != 1:
        raise ValueError("Arm end-effector orientation tracking requires exactly one body.")
    robot_ee_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    error = quat_error_magnitude(ref_ee_quat_w, robot_ee_quat_w)
    return torch.exp(-torch.square(error) / std**2)


def filtered_contact_pair_violations(
    env: ManagerBasedRLEnv,
    sensor_names: tuple[str, ...],
    force_threshold: float = 5.0,
) -> torch.Tensor:
    """Count non-adjacent filtered self-contact pairs that exceeded the force threshold."""
    violations = torch.zeros(env.num_envs, device=env.device)
    for sensor_name in sensor_names:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_name]
        force_history = contact_sensor.data.force_matrix_w_history
        if force_history is None:
            raise ValueError(f"Contact sensor '{sensor_name}' must define filter_prim_paths_expr.")
        hits = force_history.norm(dim=-1) > force_threshold
        # Each configured sensor owns one source body. Count each filtered target
        # at most once over the short history window so persistent contacts do not
        # receive an accidental history-length multiplier.
        violations += hits.any(dim=1).any(dim=1).sum(dim=-1).float()
    return violations


def motion_object_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object_index: int = 0,
    detached_scale: float = 1.0,
    attached_scale: float = 1.0,
) -> torch.Tensor:
    """Track a physical object's world position against the motion reference."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    reference_object_pos_w = command.object_target_pos_w
    if reference_object_pos_w is None:
        return torch.zeros(env.num_envs, device=env.device)
    object_asset: RigidObject = env.scene[object_cfg.name]
    error = torch.sum(
        torch.square(reference_object_pos_w[:, object_index] - object_asset.data.root_pos_w),
        dim=-1,
    )
    reward = torch.exp(-error / sigma**2)
    return reward * _motion_object_phase_scale(command, object_index, detached_scale, attached_scale)


def _motion_object_phase_scale(
    command: MotionCommand,
    object_index: int,
    detached_scale: float,
    attached_scale: float,
) -> torch.Tensor:
    """Weight object tracking by the demonstrated grasp phase."""
    if detached_scale < 0.0 or attached_scale < 0.0:
        raise ValueError("Object phase scales must be non-negative.")
    attached = command.object_attached
    if attached is None:
        return torch.full(
            (command.num_envs,),
            float(detached_scale),
            device=command.device,
        )
    return torch.where(
        attached[:, object_index],
        float(attached_scale),
        float(detached_scale),
    )


def motion_object_position_error_huber(
    env: ManagerBasedRLEnv,
    command_name: str,
    delta: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object_index: int = 0,
    detached_scale: float = 1.0,
    attached_scale: float = 1.0,
    max_cost: float | None = None,
) -> torch.Tensor:
    """Robust object-position cost that keeps a gradient after large tracking failures."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    reference_object_pos_w = command.object_target_pos_w
    if reference_object_pos_w is None:
        return torch.zeros(env.num_envs, device=env.device)
    object_asset: RigidObject = env.scene[object_cfg.name]
    distance = torch.linalg.vector_norm(
        reference_object_pos_w[:, object_index] - object_asset.data.root_pos_w,
        dim=-1,
    )
    delta = max(float(delta), 1.0e-6)
    cost = torch.where(distance <= delta, 0.5 * torch.square(distance) / delta, distance - 0.5 * delta)
    if max_cost is not None:
        if max_cost < 0.0:
            raise ValueError("max_cost must be non-negative.")
        cost = cost.clamp_max(float(max_cost))
    return cost * _motion_object_phase_scale(command, object_index, detached_scale, attached_scale)


def motion_object_up_axis_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object_index: int = 0,
    detached_scale: float = 1.0,
    attached_scale: float = 1.0,
) -> torch.Tensor:
    """Track can tilt while ignoring the yaw of a rotationally symmetric object."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    reference_object_quat_w = command.object_target_quat_w
    if reference_object_quat_w is None:
        return torch.zeros(env.num_envs, device=env.device)
    object_asset: RigidObject = env.scene[object_cfg.name]
    local_up = torch.zeros((env.num_envs, 3), device=env.device, dtype=object_asset.data.root_quat_w.dtype)
    local_up[:, 2] = 1.0
    reference_up_w = math_utils.quat_apply(reference_object_quat_w[:, object_index], local_up)
    object_up_w = math_utils.quat_apply(object_asset.data.root_quat_w, local_up)
    cosine = torch.sum(reference_up_w * object_up_w, dim=-1).clamp(-1.0, 1.0)
    tilt_error = torch.acos(cosine)
    reward = torch.exp(-torch.square(tilt_error) / max(float(std), 1.0e-6) ** 2)
    return reward * _motion_object_phase_scale(command, object_index, detached_scale, attached_scale)


def motion_object_linear_velocity_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object_index: int = 0,
    detached_scale: float = 1.0,
    attached_scale: float = 1.0,
) -> torch.Tensor:
    """Track finite-difference object velocity from the imitation trajectory."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    if command.motion.object_pos_w is None:
        return torch.zeros(env.num_envs, device=env.device)

    starts, ends = command.motion.clip_bounds_for(command.time_steps)
    prev_steps = torch.maximum(command.time_steps - 1, starts)
    next_steps = torch.minimum(command.time_steps + 1, ends)
    prev_pos_w = command.motion.object_pos_w[prev_steps, object_index]
    next_pos_w = command.motion.object_pos_w[next_steps, object_index]
    motion_ids = command.motion.motion_ids_for(command.time_steps)
    fps = command.motion.fps_values[motion_ids].to(device=env.device, dtype=next_pos_w.dtype)
    dt = (next_steps - prev_steps).to(dtype=next_pos_w.dtype) / fps.clamp_min(1.0e-6)
    reference_velocity_w = torch.where(
        dt[:, None] > 0.0,
        (next_pos_w - prev_pos_w) / dt.clamp_min(1.0e-6)[:, None],
        torch.zeros_like(next_pos_w),
    )
    delta_ori_w = math_utils.yaw_quat(
        math_utils.quat_mul(command.robot_anchor_quat_w, math_utils.quat_inv(command.anchor_quat_w))
    )
    reference_velocity_w = math_utils.quat_apply(delta_ori_w, reference_velocity_w)
    object_asset: RigidObject = env.scene[object_cfg.name]
    error = torch.sum(torch.square(reference_velocity_w - object_asset.data.root_lin_vel_w), dim=-1)
    reward = torch.exp(-error / max(float(sigma), 1.0e-6) ** 2)
    return reward * _motion_object_phase_scale(command, object_index, detached_scale, attached_scale)


def motion_object_attachment_offset_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma: float,
    ee_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    object_index: int = 0,
) -> torch.Tensor:
    """Track the demonstrated gripper-to-object offset only during the attached phase."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    attached = command.object_attached
    reference_object_pos_w = command.object_target_pos_w
    reference_ee_pos_w = command.arm_ee_target_pos_w
    if attached is None or reference_object_pos_w is None or reference_ee_pos_w is None:
        return torch.zeros(env.num_envs, device=env.device)
    robot: Articulation = env.scene[ee_cfg.name]
    object_asset: RigidObject = env.scene[object_cfg.name]
    actual_ee_pos_w = robot.data.body_pos_w[:, ee_cfg.body_ids].mean(dim=1)
    actual_offset = object_asset.data.root_pos_w - actual_ee_pos_w
    reference_offset = reference_object_pos_w[:, object_index] - reference_ee_pos_w
    error = torch.sum(torch.square(actual_offset - reference_offset), dim=-1)
    return attached[:, object_index].float() * torch.exp(-error / max(float(sigma), 1.0e-6) ** 2)


def _filtered_contact_hit(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float,
) -> torch.Tensor:
    sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    force_history = sensor.data.force_matrix_w_history
    if force_history is None:
        raise ValueError(f"Contact sensor '{sensor_cfg.name}' must define filter_prim_paths_expr.")
    hits = force_history.norm(dim=-1) > float(force_threshold)
    return hits.reshape(env.num_envs, -1).any(dim=1)


def motion_object_attached_bilateral_contact(
    env: ManagerBasedRLEnv,
    command_name: str,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
    force_threshold: float = 0.35,
    object_index: int = 0,
) -> torch.Tensor:
    """Reward bilateral finger contact while the demonstration marks the can attached."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    attached = command.object_attached
    if attached is None:
        return torch.zeros(env.num_envs, device=env.device)
    bilateral = _filtered_contact_hit(env, left_sensor_cfg, force_threshold) & _filtered_contact_hit(
        env, right_sensor_cfg, force_threshold
    )
    return attached[:, object_index].float() * bilateral.float()


def motion_object_attached_without_bilateral_contact(
    env: ManagerBasedRLEnv,
    command_name: str,
    left_sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg,
    force_threshold: float = 0.35,
    object_index: int = 0,
) -> torch.Tensor:
    """Penalize loss of the demonstrated grasp during pick, stow, and carry."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    attached = command.object_attached
    if attached is None:
        return torch.zeros(env.num_envs, device=env.device)
    bilateral = _filtered_contact_hit(env, left_sensor_cfg, force_threshold) & _filtered_contact_hit(
        env, right_sensor_cfg, force_threshold
    )
    return attached[:, object_index].float() * (~bilateral).float()


def motion_arm_ee_linear_velocity_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["link06"]),
    align_to_robot_yaw: bool = True,
) -> torch.Tensor:
    """Track finite-difference reference arm end-effector velocity from the motion NPZ."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    if command.motion.arm_ee_pos_w is None:
        return torch.zeros(env.num_envs, device=env.device)

    starts, ends = command.motion.clip_bounds_for(command.time_steps)
    prev_steps = torch.maximum(command.time_steps - 1, starts)
    next_steps = torch.minimum(command.time_steps + 1, ends)

    ref_prev_pos_w = command.motion.arm_ee_pos_w[prev_steps]
    ref_next_pos_w = command.motion.arm_ee_pos_w[next_steps]
    motion_ids = command.motion.motion_ids_for(command.time_steps)
    fps = command.motion.fps_values[motion_ids].to(device=env.device, dtype=ref_next_pos_w.dtype)
    dt = (next_steps - prev_steps).to(dtype=ref_next_pos_w.dtype) / fps.clamp_min(1.0e-6)
    ref_ee_lin_vel_w = torch.where(
        dt[:, None] > 0.0,
        (ref_next_pos_w - ref_prev_pos_w) / dt.clamp_min(1.0e-6)[:, None],
        torch.zeros_like(ref_next_pos_w),
    )

    asset: Articulation = env.scene[asset_cfg.name]
    robot_ee_lin_vel_w = asset.data.body_lin_vel_w[:, asset_cfg.body_ids[0], :]

    if align_to_robot_yaw:
        delta_ori_w = math_utils.yaw_quat(
            math_utils.quat_mul(command.robot_anchor_quat_w, math_utils.quat_inv(command.anchor_quat_w))
        )
        ref_ee_lin_vel_w = math_utils.quat_apply(delta_ori_w, ref_ee_lin_vel_w)

    error = torch.sum(torch.square(ref_ee_lin_vel_w - robot_ee_lin_vel_w), dim=-1)
    return torch.exp(-error / sigma)


def _selected_action_indices(
    env: ManagerBasedRLEnv,
    action_name: str,
    joint_names: list[str] | None,
) -> torch.Tensor | slice:
    if joint_names is None:
        return slice(None)

    cache_key = f"_go2_apex_action_indices_{action_name}_{'_'.join(joint_names)}"
    cached = getattr(env, cache_key, None)
    if cached is not None:
        return cached

    action_term = env.action_manager.get_term(action_name)
    action_joint_names = getattr(action_term, "_joint_names", None)
    if action_joint_names is None:
        raise ValueError(f"Action term '{action_name}' does not expose resolved joint names.")

    name_to_idx = {name: idx for idx, name in enumerate(action_joint_names)}
    missing = [name for name in joint_names if name not in name_to_idx]
    if missing:
        raise ValueError(f"Requested action reward joints are missing from action term '{action_name}': {missing}")

    indices = torch.tensor([name_to_idx[name] for name in joint_names], dtype=torch.long, device=env.device)
    setattr(env, cache_key, indices)
    return indices


def action_rate_l2_selected(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    joint_names: list[str] | None = None,
) -> torch.Tensor:
    """Penalize first-order action changes for a named subset of action joints."""
    indices = _selected_action_indices(env, action_name, joint_names)
    diff = env.action_manager.action[:, indices] - env.action_manager.prev_action[:, indices]
    return torch.sum(torch.square(diff), dim=1)


def motion_joint_action_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    std: float,
    action_name: str = "joint_pos",
    joint_names: list[str] | None = None,
) -> torch.Tensor:
    """Track motion joint targets using the policy output before the DecAP prior."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    action_term = env.action_manager.get_term(action_name)
    if not hasattr(action_term, "policy_processed_actions"):
        raise ValueError(f"Action term '{action_name}' does not expose policy_processed_actions.")

    action_indices = _selected_action_indices(env, action_name, joint_names)
    policy_targets = action_term.policy_processed_actions[:, action_indices]
    reference_targets = command.joint_pos
    if joint_names is not None:
        motion_name_to_idx = {name: idx for idx, name in enumerate(command.robot_motion_joint_names)}
        missing = [name for name in joint_names if name not in motion_name_to_idx]
        if missing:
            raise ValueError(f"Requested action-reference joints are missing from the motion command: {missing}")
        motion_indices = torch.tensor(
            [motion_name_to_idx[name] for name in joint_names],
            dtype=torch.long,
            device=env.device,
        )
        reference_targets = reference_targets[:, motion_indices]

    error = torch.sum(torch.square(reference_targets - policy_targets), dim=-1)
    return torch.exp(-error / std**2)


def motion_binary_action_logit_error_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    action_name: str = "joint_pos",
    joint_names: list[str] | None = None,
    open_logit: float = -1.0,
    close_logit: float = 1.0,
    physical_threshold: float = 0.0,
) -> torch.Tensor:
    """Supervise a binary actuator's raw logit and keep Gaussian variance identifiable."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    action_term = env.action_manager.get_term(action_name)
    action_indices = _selected_action_indices(env, action_name, joint_names)
    raw_actions = action_term.raw_actions[:, action_indices]

    reference_targets = command.joint_pos
    if joint_names is not None:
        motion_name_to_idx = {name: idx for idx, name in enumerate(command.robot_motion_joint_names)}
        missing = [name for name in joint_names if name not in motion_name_to_idx]
        if missing:
            raise ValueError(f"Requested binary action-reference joints are missing from the motion command: {missing}")
        motion_indices = torch.tensor(
            [motion_name_to_idx[name] for name in joint_names],
            dtype=torch.long,
            device=env.device,
        )
        reference_targets = reference_targets[:, motion_indices]

    desired_logits = torch.where(
        reference_targets > float(physical_threshold),
        float(close_logit),
        float(open_logit),
    )
    return torch.sum(torch.square(raw_actions - desired_logits), dim=-1)


def motion_binary_action_state_match(
    env: ManagerBasedRLEnv,
    command_name: str,
    action_name: str = "joint_pos",
    joint_names: list[str] | None = None,
    action_threshold: float = 0.0,
    physical_threshold: float = 0.0,
) -> torch.Tensor:
    """Reward only the correct open/close decision, independent of logit magnitude."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    action_term = env.action_manager.get_term(action_name)
    action_indices = _selected_action_indices(env, action_name, joint_names)
    action_closed = action_term.raw_actions[:, action_indices] > float(action_threshold)

    reference_targets = command.joint_pos
    if joint_names is not None:
        motion_name_to_idx = {name: idx for idx, name in enumerate(command.robot_motion_joint_names)}
        missing = [name for name in joint_names if name not in motion_name_to_idx]
        if missing:
            raise ValueError(f"Requested binary action-reference joints are missing from the motion command: {missing}")
        motion_indices = torch.tensor(
            [motion_name_to_idx[name] for name in joint_names],
            dtype=torch.long,
            device=env.device,
        )
        reference_targets = reference_targets[:, motion_indices]

    reference_closed = reference_targets > float(physical_threshold)
    return torch.mean((action_closed == reference_closed).float(), dim=-1)


def motion_joint_action_error_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    action_name: str = "joint_pos",
    joint_names: list[str] | None = None,
    normalize_by_action_scale: bool = False,
) -> torch.Tensor:
    """Penalize policy-target error before the DecAP prior is added.

    Unlike the bounded exponential tracking reward, this term keeps a useful
    gradient when the policy target is far from the reference. This prevents
    the action prior from carrying motion tracking while the actor learns an
    unrelated target that fails as soon as the prior reaches zero.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    action_term = env.action_manager.get_term(action_name)
    if not hasattr(action_term, "policy_processed_actions"):
        raise ValueError(f"Action term '{action_name}' does not expose policy_processed_actions.")

    action_indices = _selected_action_indices(env, action_name, joint_names)
    policy_targets = action_term.policy_processed_actions[:, action_indices]
    reference_targets = command.joint_pos
    if joint_names is not None:
        motion_name_to_idx = {name: idx for idx, name in enumerate(command.robot_motion_joint_names)}
        missing = [name for name in joint_names if name not in motion_name_to_idx]
        if missing:
            raise ValueError(f"Requested action-reference joints are missing from the motion command: {missing}")
        motion_indices = torch.tensor(
            [motion_name_to_idx[name] for name in joint_names],
            dtype=torch.long,
            device=env.device,
        )
        reference_targets = reference_targets[:, motion_indices]

    error = reference_targets - policy_targets
    if normalize_by_action_scale:
        action_scale = action_term._scale
        if isinstance(action_scale, torch.Tensor):
            action_scale = action_scale[:, action_indices]
        else:
            action_scale = torch.full_like(error, float(action_scale))
        error = error / torch.clamp(torch.abs(action_scale), min=1.0e-6)
    return torch.mean(torch.square(error), dim=-1)


def action_smoothness_l2_selected(
    env: ManagerBasedRLEnv,
    action_name: str = "joint_pos",
    joint_names: list[str] | None = None,
) -> torch.Tensor:
    """Penalize second-order action changes for a named subset of action joints."""
    cache_key = f"_go2_apex_action_smoothness_{action_name}"
    state = getattr(env, cache_key, None)
    action = env.action_manager.action
    prev_action = env.action_manager.prev_action
    step = int(getattr(env, "common_step_counter", 0))
    if state is None or state["prev_prev_action"].shape != prev_action.shape or state.get("step") != step:
        prev_prev_action = (
            torch.zeros_like(prev_action)
            if state is None or state["prev_prev_action"].shape != prev_action.shape
            else state["prev_prev_action"]
        )
        diff = action - 2.0 * prev_action + prev_prev_action

        valid_prev = torch.any(prev_action != 0.0, dim=1, keepdim=True)
        valid_prev_prev = torch.any(prev_prev_action != 0.0, dim=1, keepdim=True)
        diff = diff * valid_prev * valid_prev_prev

        updated_prev_prev = prev_action.detach().clone()
        if hasattr(env, "episode_length_buf"):
            reset_mask = env.episode_length_buf <= 1
            if torch.any(reset_mask):
                updated_prev_prev[reset_mask] = 0.0
        state = {"step": step, "diff": diff, "prev_prev_action": updated_prev_prev}
        setattr(env, cache_key, state)

    indices = _selected_action_indices(env, action_name, joint_names)
    return torch.sum(torch.square(state["diff"][:, indices]), dim=1)


def motion_world_foot_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma: float,
    height_scale: float = 1.0,
    xy_scale: float = 1.0,
) -> torch.Tensor:
    """Track reference foot positions in the world frame.

    This complements the yaw-aligned foot reward. The world-frame term preserves absolute vertical foot trajectories
    from jumping clips, while the existing yaw-aligned reward remains responsible for local gait shape.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    foot_indexes = _get_foot_indexes(command)
    if not foot_indexes:
        return torch.zeros(env.num_envs, device=env.device)

    error = command.body_pos_w[:, foot_indexes] - command.robot_body_pos_w[:, foot_indexes]
    scale = torch.tensor([xy_scale, xy_scale, height_scale], dtype=error.dtype, device=error.device)
    error = error * scale
    return torch.exp(-torch.sum(torch.square(error), dim=(1, 2)) / sigma)


def motion_world_base_position_error_exp(
    env: ManagerBasedRLEnv,
    command_name: str,
    sigma: float,
    xy_scale: float = 1.0,
    height_scale: float = 1.0,
) -> torch.Tensor:
    """Track the reference base position in the world frame."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = command.anchor_pos_w - command.robot_anchor_pos_w
    scale = torch.tensor([xy_scale, xy_scale, height_scale], dtype=error.dtype, device=error.device)
    return torch.exp(-torch.sum(torch.square(error * scale), dim=-1) / sigma)


def motion_world_base_position_error_huber(
    env: ManagerBasedRLEnv,
    command_name: str,
    delta: float,
    xy_scale: float = 1.0,
    height_scale: float = 0.0,
    max_cost: float | None = None,
) -> torch.Tensor:
    """Robust world-base tracking cost with a useful gradient outside the exponential kernel."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    error = command.anchor_pos_w - command.robot_anchor_pos_w
    scale = torch.tensor([xy_scale, xy_scale, height_scale], dtype=error.dtype, device=error.device)
    distance = torch.linalg.vector_norm(error * scale, dim=-1)
    delta = max(float(delta), 1.0e-6)
    cost = torch.where(distance <= delta, 0.5 * torch.square(distance) / delta, distance - 0.5 * delta)
    if max_cost is not None:
        if max_cost < 0.0:
            raise ValueError("max_cost must be non-negative.")
        cost = cost.clamp_max(float(max_cost))
    return cost


def motion_base_height_error_l2(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize deviation from the reference base height."""
    asset: Articulation = env.scene[asset_cfg.name]
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.square(asset.data.root_pos_w[:, 2] - command.anchor_pos_w[:, 2])


def motion_ang_vel_xy_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize xy angular velocity using the Isaac Gym formulation."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_ang_vel_b[:, :2]), dim=1)


def motion_joint_pos_limits_l1(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize selected joints outside their soft position limits."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    limits = asset.data.soft_joint_pos_limits[:, asset_cfg.joint_ids, :]
    lower_violation = torch.clamp(limits[..., 0] - joint_pos, min=0.0)
    upper_violation = torch.clamp(joint_pos - limits[..., 1], min=0.0)
    return torch.sum(lower_violation + upper_violation, dim=1)


def motion_feet_air_time(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    threshold: float = 0.35,
    command_threshold: float = 0.1,
) -> torch.Tensor:
    """Reward foot air time on moving reference commands."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - float(threshold)) * first_contact, dim=1)

    command_speed = (
        torch.linalg.norm(command.command_lin_vel_xy, dim=1)
        + torch.abs(command.command_lin_vel_z[:, 0])
        + torch.abs(command.command_ang_vel_z[:, 0])
    )
    reward *= command_speed > float(command_threshold)
    return reward


def motion_feet_air_time_variance_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_time: float = 0.5,
) -> torch.Tensor:
    """Penalize uneven foot air/contact timing across feet."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    last_air_time = contact_sensor.data.last_air_time[:, sensor_cfg.body_ids]
    last_contact_time = contact_sensor.data.last_contact_time[:, sensor_cfg.body_ids]
    return torch.var(torch.clamp(last_air_time, max=float(max_time)), dim=1) + torch.var(
        torch.clamp(last_contact_time, max=float(max_time)), dim=1
    )


def motion_legs_distance(
    env: ManagerBasedRLEnv,
    min_distance: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=["FL_foot", "FR_foot", "RL_foot", "RR_foot"]),
) -> torch.Tensor:
    """Penalize front/rear left-right feet being closer than a minimum lateral distance."""
    asset: Articulation = env.scene[asset_cfg.name]
    foot_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids, :]
    foot_pos_rel_w = foot_pos_w - asset.data.root_pos_w.unsqueeze(1)
    foot_pos_b = math_utils.quat_apply_inverse(
        asset.data.root_quat_w.repeat_interleave(4, dim=0),
        foot_pos_rel_w.reshape(-1, 3),
    ).reshape(env.num_envs, 4, 3)

    front_distance = torch.abs(foot_pos_b[:, 0, 1] - foot_pos_b[:, 1, 1])
    rear_distance = torch.abs(foot_pos_b[:, 2, 1] - foot_pos_b[:, 3, 1])
    front_penalty = torch.square(torch.clamp(float(min_distance) - front_distance, min=0.0))
    rear_penalty = torch.square(torch.clamp(float(min_distance) - rear_distance, min=0.0))
    return front_penalty + rear_penalty


def motion_feet_contact_forces_penalty(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    max_contact_force: float = 200.0,
) -> torch.Tensor:
    """Penalize foot contact forces above a threshold."""
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    forces = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0]
    return torch.sum(torch.clamp(forces - float(max_contact_force), min=0.0), dim=1)


def motion_feet_slip_penalty(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float = 1.0
) -> torch.Tensor:
    """Penalize foot slip while feet are in contact, following the Isaac Gym logic."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    foot_indexes = _get_foot_indexes(command)
    if not foot_indexes:
        return torch.zeros(env.num_envs, device=env.device)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = (
        contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > threshold
    )
    last_contacts = getattr(env, "_go2_apex_last_contacts", None)
    if last_contacts is None or last_contacts.shape != contact.shape:
        last_contacts = torch.zeros_like(contact)

    episode_length_buf = getattr(env, "episode_length_buf", None)
    if episode_length_buf is not None:
        reset_envs = episode_length_buf <= 1
        last_contacts[reset_envs] = False

    contact_filt = torch.logical_or(contact, last_contacts)
    env._go2_apex_last_contacts = contact.clone()

    foot_vel_xy_sq = torch.sum(torch.square(command.robot_body_lin_vel_w[:, foot_indexes, :2]), dim=-1)
    return torch.sum(contact_filt * foot_vel_xy_sq, dim=1)


def motion_impact_penalty(env: ManagerBasedRLEnv, command_name: str, delta_v_max_squared: float = 2.0) -> torch.Tensor:
    """Penalize changes in vertical foot velocity to reduce impacts."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    foot_indexes = _get_foot_indexes(command)
    if not foot_indexes:
        return torch.zeros(env.num_envs, device=env.device)

    foot_vel_z = command.robot_body_lin_vel_w[:, foot_indexes, 2]
    last_foot_vel_z = getattr(env, "_go2_apex_last_foot_vel_z", None)
    if last_foot_vel_z is None or last_foot_vel_z.shape != foot_vel_z.shape:
        env._go2_apex_last_foot_vel_z = foot_vel_z.clone()
        return torch.zeros(env.num_envs, device=env.device)

    delta_v_z_squared = torch.square(foot_vel_z - last_foot_vel_z)
    reward = torch.sum(torch.clamp(delta_v_z_squared, max=delta_v_max_squared), dim=1)

    episode_length_buf = getattr(env, "episode_length_buf", None)
    if episode_length_buf is not None:
        reset_envs = episode_length_buf <= 1
        reward[reset_envs] = 0.0

    env._go2_apex_last_foot_vel_z = foot_vel_z.clone()
    return reward


def motion_airborne_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    height_threshold: float = 0.05,
    vertical_velocity_threshold: float = 0.45,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize foot contact during reference airborne phases.

    Infer dynamic airborne phases from clip-relative base height and reference vertical velocity, then
    require the real robot to unload its feet during those frames.
    """
    command: MotionCommand = env.command_manager.get_term(command_name)
    foot_indexes = _get_foot_indexes(command)
    if not foot_indexes:
        return torch.zeros(env.num_envs, device=env.device)

    reference_airborne = command.reference_airborne
    if reference_airborne is None:
        motion_count = int(command.motion.motion_start_steps.shape[0])
        motion_ids = command.current_motion_ids

        base_height_min = getattr(env, "_go2_apex_motion_base_height_min", None)
        if base_height_min is None or base_height_min.shape[0] != motion_count:
            base_height_min = torch.empty(motion_count, dtype=torch.float32, device=env.device)
            base_height = command.motion._body_pos_w[:, command.motion_anchor_body_index, 2]
            for motion_id in range(motion_count):
                start = int(command.motion.motion_start_steps[motion_id].item())
                end = int(command.motion.motion_end_steps[motion_id].item()) + 1
                base_height_min[motion_id] = torch.min(base_height[start:end])
            env._go2_apex_motion_base_height_min = base_height_min

        reference_height_delta = command.anchor_pos_w[:, 2] - base_height_min[motion_ids]
        reference_airborne = reference_height_delta > float(height_threshold)
        if vertical_velocity_threshold is not None and vertical_velocity_threshold >= 0.0:
            reference_airborne = reference_airborne & (
                torch.abs(command.command_lin_vel_z[:, 0]) > float(vertical_velocity_threshold)
            )

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[0] > float(
        contact_threshold
    )
    contact_fraction = contact.float().mean(dim=1)
    return reference_airborne.float() * contact_fraction


def motion_reference_foot_contact_penalty(
    env: ManagerBasedRLEnv,
    command_name: str,
    sensor_cfg: SceneEntityCfg,
    contact_threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize mismatch against exported reference foot-contact phase labels."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    reference_contact = command.reference_foot_contact
    if reference_contact is None:
        return torch.zeros(env.num_envs, device=env.device)

    foot_indexes = _get_foot_indexes(command)
    if not foot_indexes:
        return torch.zeros(env.num_envs, device=env.device)

    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    robot_contact = contact_sensor.data.net_forces_w_history[:, :, sensor_cfg.body_ids, :].norm(dim=-1).max(dim=1)[
        0
    ] > float(contact_threshold)
    if reference_contact.shape != robot_contact.shape:
        raise RuntimeError(
            f"reference_foot_contact shape {tuple(reference_contact.shape)} does not match contact sensor shape "
            f"{tuple(robot_contact.shape)}."
        )
    return torch.mean((reference_contact != robot_contact).float(), dim=1)
