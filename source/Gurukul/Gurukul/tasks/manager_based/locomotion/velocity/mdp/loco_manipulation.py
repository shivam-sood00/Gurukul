from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import (
    combine_frame_transforms,
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _fixed_goal_w(
    env: ManagerBasedRLEnv,
    goal_offset: tuple[float, float, float],
    dtype: torch.dtype,
) -> torch.Tensor:
    if hasattr(env, "_loco_manip_goal_w"):
        return env._loco_manip_goal_w.to(device=env.device, dtype=dtype)
    if hasattr(env, "_pick_throw_bin_center_w"):
        return env._pick_throw_bin_center_w.to(device=env.device, dtype=dtype)
    return env.scene.env_origins + torch.tensor(goal_offset, device=env.device, dtype=dtype)


def object_position_b(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Object position in the robot root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w, obj.data.root_pos_w - robot.data.root_pos_w)


def object_pose_6d_b(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Object SE(3) pose in the robot frame with a continuous rotation encoding.

    The output contains root-frame position (3) followed by the first two
    columns of the root-frame rotation matrix (6). The 9-scalar encoding avoids
    quaternion sign ambiguity and Euler-angle discontinuities while still
    representing the object's full six-degree-of-freedom pose.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    position_b = quat_apply_inverse(robot.data.root_quat_w, obj.data.root_pos_w - robot.data.root_pos_w)
    object_quat_b = quat_mul(quat_inv(robot.data.root_quat_w), obj.data.root_quat_w)
    rotation_b = matrix_from_quat(object_quat_b)
    orientation_6d_b = rotation_b[:, :, 0:2].transpose(1, 2).reshape(env.num_envs, 6)
    return torch.cat((position_b, orientation_6d_b), dim=-1)


def object_goal_position_b(
    env: ManagerBasedRLEnv,
    goal_offset: tuple[float, float, float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Fixed per-environment object goal position in the robot root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    goal = _fixed_goal_w(env, goal_offset, robot.data.root_pos_w.dtype)
    return quat_apply_inverse(robot.data.root_quat_w, goal - robot.data.root_pos_w)


def object_to_goal_b(
    env: ManagerBasedRLEnv,
    goal_offset: tuple[float, float, float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Vector from object to its fixed per-environment goal in world axes."""
    obj: RigidObject = env.scene[object_cfg.name]
    goal = _fixed_goal_w(env, goal_offset, obj.data.root_pos_w.dtype)
    return goal - obj.data.root_pos_w


def object_velocity_b(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Object linear velocity in the robot root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    return quat_apply_inverse(robot.data.root_quat_w, obj.data.root_lin_vel_w)


def body_position_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Mean selected-body position in the robot root frame.

    Averaging multiple selected bodies is useful for virtual frames such as the
    center between two gripper fingers. A single selected body is unchanged.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    body_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids, :].mean(dim=1)
    return quat_apply_inverse(robot.data.root_quat_w, body_pos_w - robot.data.root_pos_w)


def body_orientation_6d_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Selected body orientation in the robot frame using continuous 6D rotation."""
    robot: Articulation = env.scene[robot_cfg.name]
    if len(asset_cfg.body_ids) != 1:
        raise ValueError("body_orientation_6d_b requires exactly one selected body.")
    body_quat_w = robot.data.body_quat_w[:, asset_cfg.body_ids[0], :]
    body_quat_b = quat_mul(quat_inv(robot.data.root_quat_w), body_quat_w)
    rotation_b = matrix_from_quat(body_quat_b)
    return rotation_b[:, :, 0:2].transpose(1, 2).reshape(env.num_envs, 6)


def body_to_object_b(
    env: ManagerBasedRLEnv,
    body_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Vector from the mean selected-body position to the object in the robot root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    body_pos_w = robot.data.body_pos_w[:, body_cfg.body_ids, :].mean(dim=1)
    return quat_apply_inverse(robot.data.root_quat_w, obj.data.root_pos_w - body_pos_w)


def body_to_object_distance(
    env: ManagerBasedRLEnv,
    body_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Distance from the mean selected-body position to the object."""
    robot: Articulation = env.scene[body_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    body_pos_w = robot.data.body_pos_w[:, body_cfg.body_ids, :].mean(dim=1)
    return torch.linalg.norm(obj.data.root_pos_w - body_pos_w, dim=1, keepdim=True)


def _selected_body_position_w(robot: Articulation, body_cfg: SceneEntityCfg) -> torch.Tensor:
    """Return a real body position or the virtual midpoint of several selected bodies."""
    return robot.data.body_pos_w[:, body_cfg.body_ids, :].mean(dim=1)


def loco_manipulation_skill(
    env: ManagerBasedRLEnv,
    manipulation_reach: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """One-hot skill request ordered as [approach, manipulate]."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    planar_distance = torch.linalg.norm(obj.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=1)
    manipulate = (planar_distance <= float(manipulation_reach)).to(dtype=obj.data.root_pos_w.dtype)
    return torch.stack((1.0 - manipulate, manipulate), dim=-1)


def _manipulation_reach_gate(
    env: ManagerBasedRLEnv,
    manipulation_reach: float,
    object_cfg: SceneEntityCfg,
    robot_cfg: SceneEntityCfg,
) -> torch.Tensor:
    return loco_manipulation_skill(env, manipulation_reach, object_cfg, robot_cfg)[:, 1]


def contact_sensor_force_norms(
    env: ManagerBasedRLEnv,
    sensor_names: tuple[str, ...],
    normalize: float = 100.0,
    use_history: bool = True,
) -> torch.Tensor:
    """Return per-sensor contact-force norms without resolving body names through a shared sensor."""
    values = []
    for sensor_name in sensor_names:
        sensor = env.scene.sensors[sensor_name]
        if use_history:
            forces = sensor.data.net_forces_w_history
            force_norm = torch.linalg.norm(forces, dim=-1).amax(dim=(1, 2))
        else:
            forces = sensor.data.net_forces_w
            force_norm = torch.linalg.norm(forces, dim=-1).amax(dim=1)
        values.append(force_norm / float(normalize))
    return torch.stack(values, dim=-1)


def base_to_object_standoff_b(
    env: ManagerBasedRLEnv,
    standoff_distance: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Vector from the robot base to the desired manipulation standoff pose in the robot frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    standoff_w = obj.data.root_pos_w.clone()
    standoff_w[:, 0] -= float(standoff_distance)
    standoff_w[:, 2] = robot.data.root_pos_w[:, 2]
    return quat_apply_inverse(robot.data.root_quat_w, standoff_w - robot.data.root_pos_w)


def object_height(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Object height relative to the per-environment origin."""
    obj: RigidObject = env.scene[object_cfg.name]
    return (obj.data.root_pos_w[:, 2:3] - env.scene.env_origins[:, 2:3])


def object_up_axis_b(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Object local z-axis expressed in the robot root frame."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    up_o = torch.zeros((env.num_envs, 3), device=env.device, dtype=obj.data.root_quat_w.dtype)
    up_o[:, 2] = 1.0
    up_w = quat_apply(obj.data.root_quat_w, up_o)
    return quat_apply_inverse(robot.data.root_quat_w, up_w)


def object_cuboid_points_b(
    env: ManagerBasedRLEnv,
    half_size: tuple[float, float, float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """SAFE-lite object geometry points in the robot root frame.

    The points are deterministic cuboid corners, which gives the policy a compact shape and orientation signal without
    adding a high-dimensional point cloud.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    half = torch.tensor(half_size, device=env.device, dtype=obj.data.root_pos_w.dtype)
    signs = torch.tensor(
        (
            (-1.0, -1.0, -1.0),
            (-1.0, -1.0, 1.0),
            (-1.0, 1.0, -1.0),
            (-1.0, 1.0, 1.0),
            (1.0, -1.0, -1.0),
            (1.0, -1.0, 1.0),
            (1.0, 1.0, -1.0),
            (1.0, 1.0, 1.0),
        ),
        device=env.device,
        dtype=obj.data.root_pos_w.dtype,
    )
    points_o = signs * half
    num_points = points_o.shape[0]
    points_w = obj.data.root_pos_w.unsqueeze(1) + quat_apply(
        obj.data.root_quat_w.unsqueeze(1).expand(-1, num_points, -1).reshape(-1, 4),
        points_o.unsqueeze(0).expand(env.num_envs, -1, -1).reshape(-1, 3),
    ).reshape(env.num_envs, num_points, 3)
    points_b = quat_apply_inverse(
        robot.data.root_quat_w.unsqueeze(1).expand(-1, num_points, -1).reshape(-1, 4),
        (points_w - robot.data.root_pos_w.unsqueeze(1)).reshape(-1, 3),
    ).reshape(env.num_envs, num_points, 3)
    return points_b.reshape(env.num_envs, num_points * 3)


def ee_to_object_exp(
    env: ManagerBasedRLEnv,
    std: float,
    ee_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward placing the selected end-effector near the object."""
    robot: Articulation = env.scene[ee_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    ee_pos_w = _selected_body_position_w(robot, ee_cfg)
    distance = torch.linalg.norm(ee_pos_w - obj.data.root_pos_w, dim=1)
    return torch.exp(-torch.square(distance) / float(std) ** 2)


def ee_to_object_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    ee_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Dense reach shaping with a longer useful gradient than a narrow exponential."""
    robot: Articulation = env.scene[ee_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    ee_pos_w = _selected_body_position_w(robot, ee_cfg)
    distance = torch.linalg.norm(ee_pos_w - obj.data.root_pos_w, dim=1)
    return 1.0 - torch.tanh(distance / float(std))


def ee_to_object_in_manipulation_reach_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    manipulation_reach: float,
    ee_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    active_pick_stages: tuple[int, ...] | None = None,
    pick_stage_params: dict | None = None,
) -> torch.Tensor:
    """Reward arm reaching only after the object enters the manipulation workspace."""
    gate = _manipulation_reach_gate(env, manipulation_reach, object_cfg, robot_cfg)
    reward = gate * ee_to_object_tanh(env, std, ee_cfg, object_cfg)
    return _apply_pick_stage_gate(env, reward, active_pick_stages, pick_stage_params)


def object_to_goal_exp(
    env: ManagerBasedRLEnv,
    std: float,
    goal_offset: tuple[float, float, float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward object position reaching a fixed per-environment goal."""
    obj: RigidObject = env.scene[object_cfg.name]
    goal = _fixed_goal_w(env, goal_offset, obj.data.root_pos_w.dtype)
    distance = torch.linalg.norm((goal - obj.data.root_pos_w)[:, :2], dim=1)
    return torch.exp(-torch.square(distance) / float(std) ** 2)


def object_to_goal_3d_exp(
    env: ManagerBasedRLEnv,
    std: float,
    goal_offset: tuple[float, float, float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward object position reaching a fixed 3D goal."""
    obj: RigidObject = env.scene[object_cfg.name]
    goal = _fixed_goal_w(env, goal_offset, obj.data.root_pos_w.dtype)
    distance = torch.linalg.norm(goal - obj.data.root_pos_w, dim=1)
    return torch.exp(-torch.square(distance) / float(std) ** 2)


def object_velocity_xy_exp(
    env: ManagerBasedRLEnv,
    std: float,
    target_velocity: tuple[float, float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward planar object velocity tracking in world axes."""
    obj: RigidObject = env.scene[object_cfg.name]
    target = torch.tensor(target_velocity, device=env.device, dtype=obj.data.root_lin_vel_w.dtype)
    error = torch.sum(torch.square(obj.data.root_lin_vel_w[:, :2] - target), dim=1)
    return torch.exp(-error / float(std) ** 2)


def object_velocity_towards_goal_exp(
    env: ManagerBasedRLEnv,
    std: float,
    goal_offset: tuple[float, float, float],
    target_speed: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward object velocity projected toward a fixed goal."""
    obj: RigidObject = env.scene[object_cfg.name]
    goal = _fixed_goal_w(env, goal_offset, obj.data.root_pos_w.dtype)
    to_goal = goal - obj.data.root_pos_w
    direction = to_goal / torch.clamp(torch.linalg.norm(to_goal, dim=1, keepdim=True), min=1e-6)
    speed_error = torch.sum(obj.data.root_lin_vel_w[:, :3] * direction, dim=1) - float(target_speed)
    return torch.exp(-torch.square(speed_error) / float(std) ** 2)


def _object_lift_gate(
    env: ManagerBasedRLEnv,
    min_height: float,
    lift_margin: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    obj: RigidObject = env.scene[object_cfg.name]
    lift_progress = (obj.data.root_pos_w[:, 2] - float(min_height)) / max(float(lift_margin), 1.0e-6)
    return torch.clamp(lift_progress, min=0.0, max=1.0)


def object_lift_progress(
    env: ManagerBasedRLEnv,
    min_height: float,
    lift_margin: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Normalized object lift progress above its reset support height."""
    return _object_lift_gate(env, min_height, lift_margin, object_cfg).unsqueeze(-1)


def _pick_grasp_contact_gate(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg | None,
    right_sensor_cfg: SceneEntityCfg | None,
    force_threshold: float,
    required_contacts: int,
) -> torch.Tensor:
    """Return whether the requested number of gripper fingers contact the object."""
    if sensor_cfg is None:
        return torch.ones(env.num_envs, device=env.device, dtype=torch.bool)
    sensor_cfgs = (sensor_cfg,) if right_sensor_cfg is None else (sensor_cfg, right_sensor_cfg)
    hits = torch.stack(
        [_filtered_contact_gate(env, cfg, force_threshold) for cfg in sensor_cfgs],
        dim=-1,
    )
    return hits.sum(dim=-1) >= max(int(required_contacts), 1)


def _advance_pick_stage(
    stage: torch.Tensor,
    base_object_distance: torch.Tensor,
    ee_object_distance: torch.Tensor,
    lift_progress: torch.Tensor,
    manipulation_reach: float,
    pregrasp_distance: float,
    lift_enter_fraction: float = 0.65,
    reach_exit_margin: float = 0.08,
    pregrasp_exit_margin: float = 0.05,
    lift_exit_fraction: float = 0.25,
    grasp_contact: torch.Tensor | None = None,
) -> torch.Tensor:
    """Advance or recover the four-stage pick graph with hysteresis.

    Stages are 0=approach, 1=reach, 2=grasp/lift, and 3=hold. Backward
    transitions let the policy recover when the object moves away or is dropped.
    """
    next_stage = stage.clone()
    reach_exit = float(manipulation_reach) + float(reach_exit_margin)
    pregrasp_exit = float(pregrasp_distance) + float(pregrasp_exit_margin)
    if grasp_contact is None:
        grasp_contact = torch.ones_like(next_stage, dtype=torch.bool)

    # Recover from a dropped object before evaluating forward transitions.
    dropped = (next_stage == 3) & (lift_progress < float(lift_exit_fraction))
    dropped_fallback = torch.where(
        base_object_distance > reach_exit,
        torch.zeros_like(next_stage),
        torch.where(ee_object_distance > pregrasp_exit, torch.ones_like(next_stage), 2 * torch.ones_like(next_stage)),
    )
    next_stage = torch.where(dropped, dropped_fallback, next_stage)

    # Earlier stages may also move backward when their geometric prerequisite is lost.
    next_stage = torch.where(
        (next_stage == 2) & (base_object_distance > reach_exit), torch.zeros_like(next_stage), next_stage
    )
    next_stage = torch.where(
        (next_stage == 2) & (ee_object_distance > pregrasp_exit), torch.ones_like(next_stage), next_stage
    )
    next_stage = torch.where(
        (next_stage == 1) & (base_object_distance > reach_exit), torch.zeros_like(next_stage), next_stage
    )

    # Forward transitions are sequential and can cross more than one boundary when
    # an episode is initialized close to a later stage.
    next_stage = torch.where(
        (next_stage == 0) & (base_object_distance <= float(manipulation_reach)),
        torch.ones_like(next_stage),
        next_stage,
    )
    next_stage = torch.where(
        (next_stage == 1) & (ee_object_distance <= float(pregrasp_distance)),
        2 * torch.ones_like(next_stage),
        next_stage,
    )
    next_stage = torch.where(
        (next_stage == 2) & (lift_progress >= float(lift_enter_fraction)) & grasp_contact,
        3 * torch.ones_like(next_stage),
        next_stage,
    )
    return next_stage


def pick_stage_index(
    env: ManagerBasedRLEnv,
    manipulation_reach: float,
    pregrasp_distance: float,
    min_height: float,
    lift_margin: float,
    lift_enter_fraction: float = 0.65,
    reach_exit_margin: float = 0.08,
    pregrasp_exit_margin: float = 0.05,
    lift_exit_fraction: float = 0.25,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    right_sensor_cfg: SceneEntityCfg | None = None,
    force_threshold: float = 0.35,
    required_contacts: int = 2,
) -> torch.Tensor:
    """Update and return the recoverable per-environment pick-stage index."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    ee_pos_w = _selected_body_position_w(robot, ee_cfg)
    base_object_distance = torch.linalg.norm(obj.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=1)
    ee_object_distance = torch.linalg.norm(obj.data.root_pos_w - ee_pos_w, dim=1)
    lift_progress = _object_lift_gate(env, min_height, lift_margin, object_cfg)
    grasp_contact = _pick_grasp_contact_gate(
        env,
        sensor_cfg,
        right_sensor_cfg,
        force_threshold,
        required_contacts,
    )

    if not hasattr(env, "_loco_manip_pick_stage"):
        env._loco_manip_pick_stage = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
    env._loco_manip_pick_stage.copy_(
        _advance_pick_stage(
            env._loco_manip_pick_stage,
            base_object_distance,
            ee_object_distance,
            lift_progress,
            manipulation_reach,
            pregrasp_distance,
            lift_enter_fraction,
            reach_exit_margin,
            pregrasp_exit_margin,
            lift_exit_fraction,
            grasp_contact,
        )
    )
    return env._loco_manip_pick_stage


def pick_stage_one_hot(
    env: ManagerBasedRLEnv,
    manipulation_reach: float,
    pregrasp_distance: float,
    min_height: float,
    lift_margin: float,
    ee_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    lift_enter_fraction: float = 0.65,
    reach_exit_margin: float = 0.08,
    pregrasp_exit_margin: float = 0.05,
    lift_exit_fraction: float = 0.25,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    right_sensor_cfg: SceneEntityCfg | None = None,
    force_threshold: float = 0.35,
    required_contacts: int = 2,
) -> torch.Tensor:
    """One-hot stage observation ordered as [approach, reach, grasp/lift, hold]."""
    stage = pick_stage_index(
        env,
        manipulation_reach,
        pregrasp_distance,
        min_height,
        lift_margin,
        lift_enter_fraction,
        reach_exit_margin,
        pregrasp_exit_margin,
        lift_exit_fraction,
        ee_cfg,
        object_cfg,
        robot_cfg,
        sensor_cfg,
        right_sensor_cfg,
        force_threshold,
        required_contacts,
    )
    dtype = env.scene[object_cfg.name].data.root_pos_w.dtype
    return torch.nn.functional.one_hot(stage, num_classes=4).to(dtype=dtype)


def _bounded_scalar_progress(
    env: ManagerBasedRLEnv,
    current_value: torch.Tensor,
    memory_attr: str,
    progress_scale: float,
    *,
    increasing_is_progress: bool,
) -> torch.Tensor:
    """Return bounded one-step scalar progress and update per-environment memory."""
    previous = getattr(env, memory_attr, None)
    if previous is None or previous.shape != current_value.shape:
        previous = current_value.detach().clone()
    fresh = ~torch.isfinite(previous)
    if hasattr(env, "episode_length_buf"):
        fresh |= env.episode_length_buf <= 1
    previous = torch.where(fresh, current_value, previous)
    delta = current_value - previous if increasing_is_progress else previous - current_value
    progress = delta / max(float(progress_scale), 1.0e-6)
    setattr(env, memory_attr, current_value.detach().clone())
    return torch.clamp(progress, min=-1.0, max=1.0)


def _distance_progress(
    env: ManagerBasedRLEnv,
    current_distance: torch.Tensor,
    memory_attr: str,
    progress_scale: float,
) -> torch.Tensor:
    """Return bounded one-step distance reduction and update its per-environment memory."""
    return _bounded_scalar_progress(
        env,
        current_distance,
        memory_attr,
        progress_scale,
        increasing_is_progress=False,
    )


def base_object_approach_progress(
    env: ManagerBasedRLEnv,
    progress_scale: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    active_pick_stages: tuple[int, ...] | None = None,
    pick_stage_params: dict | None = None,
) -> torch.Tensor:
    """Reward reducing planar base-object distance; holding position earns zero."""
    robot: Articulation = env.scene[robot_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    distance = torch.linalg.norm(obj.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=1)
    reward = _distance_progress(env, distance, "_loco_manip_prev_base_object_distance", progress_scale)
    return _apply_pick_stage_gate(env, reward, active_pick_stages, pick_stage_params)


def base_velocity_towards_object_standoff(
    env: ManagerBasedRLEnv,
    standoff_distance: float,
    speed_scale: float,
    distance_scale: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    active_pick_stages: tuple[int, ...] | None = None,
    pick_stage_params: dict | None = None,
) -> torch.Tensor:
    """Reward signed base velocity toward the object's manipulation standoff.

    Motion away from the standoff is penalized and holding still earns zero.
    The distance gate fades the velocity incentive near the target so it does
    not encourage overshoot or perpetual motion around the object.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    standoff_error_b = base_to_object_standoff_b(env, standoff_distance, object_cfg, robot_cfg)
    planar_error_b = standoff_error_b[:, :2]
    planar_distance = torch.linalg.norm(planar_error_b, dim=1)
    direction_b = planar_error_b / planar_distance.unsqueeze(1).clamp_min(1.0e-6)
    speed_towards_standoff = torch.sum(robot.data.root_lin_vel_b[:, :2] * direction_b, dim=1)
    distance_gate = torch.tanh(planar_distance / max(float(distance_scale), 1.0e-6))
    reward = torch.clamp(speed_towards_standoff / max(float(speed_scale), 1.0e-6), min=-1.0, max=1.0)
    reward *= distance_gate
    return _apply_pick_stage_gate(env, reward, active_pick_stages, pick_stage_params)


def base_object_standoff_error_l2(
    env: ManagerBasedRLEnv,
    standoff_distance: float,
    scale: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    active_pick_stages: tuple[int, ...] | None = None,
    pick_stage_params: dict | None = None,
) -> torch.Tensor:
    """Zero-centered planar error from the manipulation standoff pose."""
    error_b = base_to_object_standoff_b(env, standoff_distance, object_cfg, robot_cfg)
    error = torch.sum(torch.square(error_b[:, :2] / max(float(scale), 1.0e-6)), dim=1)
    return _apply_pick_stage_gate(env, error, active_pick_stages, pick_stage_params)


def base_object_yaw_error_l2(
    env: ManagerBasedRLEnv,
    scale: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    active_pick_stages: tuple[int, ...] | None = None,
    pick_stage_params: dict | None = None,
) -> torch.Tensor:
    """Zero-centered yaw error for facing the tabletop object."""
    object_pos_b = object_position_b(env, object_cfg, robot_cfg)
    yaw_error = torch.atan2(object_pos_b[:, 1], object_pos_b[:, 0])
    error = torch.square(yaw_error / max(float(scale), 1.0e-6))
    return _apply_pick_stage_gate(env, error, active_pick_stages, pick_stage_params)


def ee_object_reach_progress(
    env: ManagerBasedRLEnv,
    progress_scale: float,
    ee_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    active_pick_stages: tuple[int, ...] | None = None,
    pick_stage_params: dict | None = None,
) -> torch.Tensor:
    """Reward reducing EE-object distance; holding position earns zero."""
    robot: Articulation = env.scene[ee_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    distance = torch.linalg.norm(_selected_body_position_w(robot, ee_cfg) - obj.data.root_pos_w, dim=1)
    reward = _distance_progress(env, distance, "_loco_manip_prev_ee_object_distance", progress_scale)
    return _apply_pick_stage_gate(env, reward, active_pick_stages, pick_stage_params)


def ee_object_alignment_excess_l2(
    env: ManagerBasedRLEnv,
    tolerance: float,
    scale: float,
    ee_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    active_pick_stages: tuple[int, ...] | None = None,
    pick_stage_params: dict | None = None,
) -> torch.Tensor:
    """Zero-centered grasp-alignment error beyond a tolerated EE distance."""
    robot: Articulation = env.scene[ee_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    distance = torch.linalg.norm(_selected_body_position_w(robot, ee_cfg) - obj.data.root_pos_w, dim=1)
    excess = torch.clamp(distance - float(tolerance), min=0.0) / max(float(scale), 1.0e-6)
    return _apply_pick_stage_gate(env, torch.square(excess), active_pick_stages, pick_stage_params)


def grasp_frame_horizontal_error(
    env: ManagerBasedRLEnv,
    wrist_cfg: SceneEntityCfg,
    finger_cfg: SceneEntityCfg,
    active_pick_stages: tuple[int, ...] | None = None,
    pick_stage_params: dict | None = None,
) -> torch.Tensor:
    """Penalize tilt of the finger-closing and grasp-approach axes.

    The definition uses physical body positions rather than assumed asset axes,
    so it remains consistent with the virtual finger midpoint used by rewards.
    """
    robot: Articulation = env.scene[wrist_cfg.name]
    if len(wrist_cfg.body_ids) != 1 or len(finger_cfg.body_ids) != 2:
        raise ValueError("grasp_frame_horizontal_error requires one wrist body and two finger bodies.")
    wrist_pos_w = robot.data.body_pos_w[:, wrist_cfg.body_ids[0], :]
    finger_pos_w = robot.data.body_pos_w[:, finger_cfg.body_ids, :]
    midpoint_w = finger_pos_w.mean(dim=1)
    approach_axis_w = torch.nn.functional.normalize(midpoint_w - wrist_pos_w, dim=1)
    closing_axis_w = torch.nn.functional.normalize(finger_pos_w[:, 1] - finger_pos_w[:, 0], dim=1)
    error = torch.square(approach_axis_w[:, 2]) + torch.square(closing_axis_w[:, 2])
    return _apply_pick_stage_gate(env, error, active_pick_stages, pick_stage_params)


def pick_stage_transition_bonus(
    env: ManagerBasedRLEnv,
    target_stage: int,
    pick_stage_params: dict,
) -> torch.Tensor:
    """Emit once whenever an environment moves forward into the requested stage."""
    stage = pick_stage_index(env, **pick_stage_params)
    memory_attr = f"_loco_manip_transition_prev_stage_{int(target_stage)}"
    previous = getattr(env, memory_attr, None)
    if previous is None or previous.shape != stage.shape:
        previous = -torch.ones_like(stage)
    fresh = previous < 0
    # A single stage update may cross several satisfied boundaries (for example,
    # Reach -> Grasp/Lift -> Hold). Pay every crossed boundary exactly once.
    entered = (~fresh) & (previous < int(target_stage)) & (stage >= int(target_stage))
    setattr(env, memory_attr, stage.detach().clone())
    return entered.to(dtype=env.scene[pick_stage_params["object_cfg"].name].data.root_pos_w.dtype)


def pick_stage_regression_penalty(
    env: ManagerBasedRLEnv,
    pick_stage_params: dict,
) -> torch.Tensor:
    """Penalize each backward stage boundary crossed without charging for holding a stage."""
    stage = pick_stage_index(env, **pick_stage_params)
    memory_attr = "_loco_manip_regression_prev_stage"
    previous = getattr(env, memory_attr, None)
    if previous is None or previous.shape != stage.shape:
        previous = -torch.ones_like(stage)
    fresh = previous < 0
    regression = torch.clamp(previous - stage, min=0)
    regression = torch.where(fresh, torch.zeros_like(regression), regression)
    setattr(env, memory_attr, stage.detach().clone())
    return regression.to(dtype=env.scene[pick_stage_params["object_cfg"].name].data.root_pos_w.dtype)


def _pick_success_state(
    env: ManagerBasedRLEnv,
    hold_steps: int,
    pick_stage_params: dict,
    max_object_speed: float,
) -> torch.Tensor:
    """Update a cached stable, contact-verified hold counter at most once per environment step."""
    stage = pick_stage_index(env, **pick_stage_params)
    contact = _pick_grasp_contact_gate(
        env,
        pick_stage_params.get("sensor_cfg"),
        pick_stage_params.get("right_sensor_cfg"),
        float(pick_stage_params.get("force_threshold", 0.35)),
        int(pick_stage_params.get("required_contacts", 2)),
    )
    counter = getattr(env, "_loco_manip_pick_hold_counter", None)
    if counter is None or counter.shape != stage.shape:
        counter = torch.zeros_like(stage)
    step = int(env.common_step_counter)
    if getattr(env, "_loco_manip_pick_hold_counter_step", -1) != step:
        obj: RigidObject = env.scene[pick_stage_params["object_cfg"].name]
        stable = torch.linalg.norm(obj.data.root_lin_vel_w, dim=1) <= float(max_object_speed)
        active = (stage == 3) & contact & stable
        counter = torch.where(active, counter + 1, torch.zeros_like(counter))
        env._loco_manip_pick_hold_counter = counter
        env._loco_manip_pick_hold_counter_step = step
    return counter >= max(int(hold_steps), 1)


def pick_success(
    env: ManagerBasedRLEnv,
    hold_steps: int,
    pick_stage_params: dict,
    max_object_speed: float = 0.25,
) -> torch.Tensor:
    """Terminate after a stable, contact-verified hold."""
    return _pick_success_state(env, hold_steps, pick_stage_params, max_object_speed)


def pick_success_bonus(
    env: ManagerBasedRLEnv,
    hold_steps: int,
    pick_stage_params: dict,
    max_object_speed: float = 0.25,
) -> torch.Tensor:
    """Emit a single terminal success bonus per episode."""
    success = _pick_success_state(env, hold_steps, pick_stage_params, max_object_speed)
    emitted = getattr(env, "_loco_manip_pick_success_emitted", None)
    if emitted is None or emitted.shape != success.shape:
        emitted = torch.zeros_like(success)
    bonus = success & ~emitted
    env._loco_manip_pick_success_emitted = emitted | success
    success_count = getattr(env, "_loco_manip_pick_success_count", None)
    if not isinstance(success_count, torch.Tensor):
        success_count = torch.zeros((), device=env.device, dtype=torch.long)
    env._loco_manip_pick_success_count = success_count + bonus.sum()
    return bonus.to(dtype=env.scene[pick_stage_params["object_cfg"].name].data.root_pos_w.dtype)


def _apply_pick_stage_gate(
    env: ManagerBasedRLEnv,
    value: torch.Tensor,
    active_pick_stages: tuple[int, ...] | None,
    pick_stage_params: dict | None,
) -> torch.Tensor:
    """Apply an optional stage mask while leaving non-pick callers unchanged."""
    if active_pick_stages is None:
        return value
    if pick_stage_params is None:
        raise ValueError("pick_stage_params is required when active_pick_stages is set.")
    stage = pick_stage_index(env, **pick_stage_params)
    active = torch.zeros_like(stage, dtype=torch.bool)
    for stage_index in active_pick_stages:
        active |= stage == int(stage_index)
    return value * active.to(dtype=value.dtype)


def _filtered_contact_gate(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float,
) -> torch.Tensor:
    contact_sensor = env.scene.sensors[sensor_cfg.name]
    data = contact_sensor.data
    if data.force_matrix_w_history is not None:
        forces = data.force_matrix_w_history[:, :, sensor_cfg.body_ids, :, :]
        hits = forces.norm(dim=-1) > float(force_threshold)
        return hits.any(dim=1).any(dim=-1).any(dim=-1).float()

    forces = data.net_forces_w_history[:, :, sensor_cfg.body_ids, :]
    hits = forces.norm(dim=-1) > float(force_threshold)
    return hits.any(dim=1).any(dim=-1).float()


def _any_filtered_contact_gate(
    env: ManagerBasedRLEnv,
    sensor_cfgs: tuple[SceneEntityCfg, ...],
    force_threshold: float,
) -> torch.Tensor:
    gates = [_filtered_contact_gate(env, sensor_cfg, force_threshold) for sensor_cfg in sensor_cfgs]
    return torch.stack(gates, dim=-1).amax(dim=-1)


def gripper_object_contact(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg | None = None,
    force_threshold: float = 0.5,
    required_contacts: int = 2,
    active_pick_stages: tuple[int, ...] | None = None,
    pick_stage_params: dict | None = None,
) -> torch.Tensor:
    """Reward recent filtered contact between gripper finger links and the object."""
    sensor_cfgs = (sensor_cfg,) if right_sensor_cfg is None else (sensor_cfg, right_sensor_cfg)
    hits = torch.stack([_filtered_contact_gate(env, cfg, force_threshold) for cfg in sensor_cfgs], dim=-1)
    reward = torch.clamp(hits.float().sum(dim=-1) / max(int(required_contacts), 1), min=0.0, max=1.0)
    return _apply_pick_stage_gate(env, reward, active_pick_stages, pick_stage_params)


def gripper_object_contact_progress(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg | None = None,
    force_threshold: float = 0.5,
    required_contacts: int = 2,
    active_pick_stages: tuple[int, ...] | None = None,
    pick_stage_params: dict | None = None,
) -> torch.Tensor:
    """Reward acquiring filtered contacts once, and penalize losing them."""
    sensor_cfgs = (sensor_cfg,) if right_sensor_cfg is None else (sensor_cfg, right_sensor_cfg)
    hits = torch.stack([_filtered_contact_gate(env, cfg, force_threshold) for cfg in sensor_cfgs], dim=-1)
    contact_fraction = torch.clamp(
        hits.float().sum(dim=-1) / max(int(required_contacts), 1),
        min=0.0,
        max=1.0,
    )
    progress = _bounded_scalar_progress(
        env,
        contact_fraction,
        "_loco_manip_prev_gripper_contact_fraction",
        1.0,
        increasing_is_progress=True,
    )
    return _apply_pick_stage_gate(env, progress, active_pick_stages, pick_stage_params)


def gripper_object_contact_state(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg,
    force_threshold: float = 0.5,
    required_contacts: int = 2,
) -> torch.Tensor:
    """Observation form of recent filtered gripper-object contact."""
    return gripper_object_contact(
        env,
        sensor_cfg,
        force_threshold=force_threshold,
        required_contacts=required_contacts,
    ).unsqueeze(-1)


def gripper_close_near_object(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    gripper_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    near_std: float = 0.12,
    open_joint_pos: float | tuple[float, ...] = 0.0,
    closed_joint_pos: float | tuple[float, ...] = 0.03,
    active_pick_stages: tuple[int, ...] | None = None,
    pick_stage_params: dict | None = None,
) -> torch.Tensor:
    """Reward closing the fingers only when the selected end-effector is near the object."""
    robot: Articulation = env.scene[ee_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    ee_pos_w = _selected_body_position_w(robot, ee_cfg)
    distance = torch.linalg.norm(ee_pos_w - obj.data.root_pos_w, dim=1)
    near_gate = 1.0 - torch.tanh(distance / float(near_std))

    joint_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids]
    open_pos = torch.as_tensor(open_joint_pos, device=joint_pos.device, dtype=joint_pos.dtype)
    closed_pos = torch.as_tensor(closed_joint_pos, device=joint_pos.device, dtype=joint_pos.dtype)
    close_travel = closed_pos - open_pos
    close_travel = torch.where(torch.abs(close_travel) < 1.0e-6, torch.ones_like(close_travel), close_travel)
    close_fraction = (joint_pos - open_pos) / close_travel
    close_fraction = torch.clamp(close_fraction, min=0.0, max=1.0).mean(dim=-1)
    reward = near_gate * close_fraction
    return _apply_pick_stage_gate(env, reward, active_pick_stages, pick_stage_params)


def gripper_close_fraction(
    env: ManagerBasedRLEnv,
    gripper_cfg: SceneEntityCfg,
    open_joint_pos: float | tuple[float, ...],
    closed_joint_pos: float | tuple[float, ...],
) -> torch.Tensor:
    """Return per-finger closure in ``[0, 1]`` from measured gripper positions."""
    robot: Articulation = env.scene[gripper_cfg.name]
    joint_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids]
    open_pos = torch.as_tensor(open_joint_pos, device=joint_pos.device, dtype=joint_pos.dtype)
    closed_pos = torch.as_tensor(closed_joint_pos, device=joint_pos.device, dtype=joint_pos.dtype)
    close_travel = closed_pos - open_pos
    close_travel = torch.where(torch.abs(close_travel) < 1.0e-6, torch.ones_like(close_travel), close_travel)
    return torch.clamp((joint_pos - open_pos) / close_travel, min=0.0, max=1.0)


def gripper_close_progress_near_object(
    env: ManagerBasedRLEnv,
    ee_cfg: SceneEntityCfg,
    gripper_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    near_std: float = 0.12,
    progress_scale: float = 0.25,
    open_joint_pos: float | tuple[float, ...] = 0.0,
    closed_joint_pos: float | tuple[float, ...] = 0.03,
    active_pick_stages: tuple[int, ...] | None = None,
    pick_stage_params: dict | None = None,
) -> torch.Tensor:
    """Reward measured closing progress near the object, not remaining closed."""
    robot: Articulation = env.scene[ee_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    distance = torch.linalg.norm(_selected_body_position_w(robot, ee_cfg) - obj.data.root_pos_w, dim=1)
    near_gate = 1.0 - torch.tanh(distance / float(near_std))
    closure = gripper_close_fraction(
        env,
        gripper_cfg,
        open_joint_pos,
        closed_joint_pos,
    ).mean(dim=-1)
    progress = _bounded_scalar_progress(
        env,
        closure,
        "_loco_manip_prev_gripper_close_fraction",
        progress_scale,
        increasing_is_progress=True,
    )
    return _apply_pick_stage_gate(env, near_gate * progress, active_pick_stages, pick_stage_params)


def gripper_closed_fraction(
    env: ManagerBasedRLEnv,
    gripper_cfg: SceneEntityCfg,
    open_joint_pos: float | tuple[float, ...],
    closed_joint_pos: float | tuple[float, ...],
    active_pick_stages: tuple[int, ...] | None = None,
    pick_stage_params: dict | None = None,
) -> torch.Tensor:
    """Return mean closure for penalizing premature closing before pregrasp."""
    closure = gripper_close_fraction(
        env,
        gripper_cfg,
        open_joint_pos,
        closed_joint_pos,
    ).mean(dim=-1)
    return _apply_pick_stage_gate(env, closure, active_pick_stages, pick_stage_params)


def object_lifted_with_gripper_contact_exp(
    env: ManagerBasedRLEnv,
    std: float,
    min_height: float,
    target_height: float,
    sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg | None = None,
    force_threshold: float = 0.5,
    required_contacts: int = 1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    active_pick_stages: tuple[int, ...] | None = None,
    pick_stage_params: dict | None = None,
) -> torch.Tensor:
    """Reward lift progress only when the gripper recently contacted the object."""
    contact_gate = _pick_grasp_contact_gate(
        env,
        sensor_cfg,
        right_sensor_cfg,
        force_threshold,
        required_contacts,
    ).float()
    lift = object_lifted_above_height_exp(env, std, min_height, target_height, object_cfg)
    reward = contact_gate * lift
    return _apply_pick_stage_gate(env, reward, active_pick_stages, pick_stage_params)


def object_hold_lifted_with_gripper_contact(
    env: ManagerBasedRLEnv,
    min_height: float,
    lift_margin: float,
    sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg | None = None,
    force_threshold: float = 0.5,
    required_contacts: int = 1,
    velocity_std: float = 0.35,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    active_pick_stages: tuple[int, ...] | None = None,
    pick_stage_params: dict | None = None,
) -> torch.Tensor:
    """Reward a stable held object after pickup instead of a brief height spike."""
    obj: RigidObject = env.scene[object_cfg.name]
    contact_gate = _pick_grasp_contact_gate(
        env,
        sensor_cfg,
        right_sensor_cfg,
        force_threshold,
        required_contacts,
    ).float()
    lift_gate = _object_lift_gate(env, min_height, lift_margin, object_cfg)
    speed = torch.linalg.norm(obj.data.root_lin_vel_w, dim=1)
    still = torch.exp(-torch.square(speed) / float(velocity_std) ** 2)
    reward = contact_gate * lift_gate * still
    return _apply_pick_stage_gate(env, reward, active_pick_stages, pick_stage_params)


def object_lifted_without_gripper_contact(
    env: ManagerBasedRLEnv,
    min_height: float,
    lift_margin: float,
    sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg | None = None,
    force_threshold: float = 0.5,
    required_contacts: int = 1,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    active_pick_stages: tuple[int, ...] | None = None,
    pick_stage_params: dict | None = None,
) -> torch.Tensor:
    """Penalize object lift that is not explained by gripper-object contact."""
    contact_gate = _pick_grasp_contact_gate(
        env,
        sensor_cfg,
        right_sensor_cfg,
        force_threshold,
        required_contacts,
    ).float()
    lift_gate = _object_lift_gate(env, min_height, lift_margin, object_cfg)
    penalty = lift_gate * (1.0 - contact_gate)
    return _apply_pick_stage_gate(env, penalty, active_pick_stages, pick_stage_params)


def object_xy_motion_before_lift(
    env: ManagerBasedRLEnv,
    min_height: float,
    lift_margin: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    active_pick_stages: tuple[int, ...] | None = None,
    pick_stage_params: dict | None = None,
) -> torch.Tensor:
    """Penalize pushing the object around before a real lift occurs."""
    obj: RigidObject = env.scene[object_cfg.name]
    lift_gate = _object_lift_gate(env, min_height, lift_margin, object_cfg)
    penalty = (1.0 - lift_gate) * torch.sum(torch.square(obj.data.root_lin_vel_w[:, :2]), dim=1)
    return _apply_pick_stage_gate(env, penalty, active_pick_stages, pick_stage_params)


def _object_lift_memory_gate(
    env: ManagerBasedRLEnv,
    min_height: float,
    lift_margin: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    sensor_cfg: SceneEntityCfg | None = None,
    right_sensor_cfg: SceneEntityCfg | None = None,
    force_threshold: float = 0.5,
    required_contacts: int = 2,
) -> torch.Tensor:
    """Latch a lift, optionally requiring bilateral gripper contact at the lift moment."""
    lifted = _object_lift_gate(env, min_height, lift_margin, object_cfg) > 0.65
    memory_attr = "_loco_manip_object_lifted_once"
    if sensor_cfg is not None:
        contact = gripper_object_contact(
            env,
            sensor_cfg=sensor_cfg,
            right_sensor_cfg=right_sensor_cfg,
            force_threshold=force_threshold,
            required_contacts=required_contacts,
        ) >= 1.0
        lifted &= contact
        memory_attr = "_loco_manip_object_grasped_once"
    if not hasattr(env, memory_attr):
        setattr(env, memory_attr, torch.zeros(env.num_envs, device=env.device, dtype=torch.bool))
    memory = getattr(env, memory_attr)
    memory |= lifted
    return memory.to(dtype=env.scene[object_cfg.name].data.root_pos_w.dtype)


def object_to_goal_after_lift_3d_exp(
    env: ManagerBasedRLEnv,
    std: float,
    goal_offset: tuple[float, float, float],
    min_height: float,
    lift_margin: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    sensor_cfg: SceneEntityCfg | None = None,
    right_sensor_cfg: SceneEntityCfg | None = None,
    force_threshold: float = 0.5,
    required_contacts: int = 2,
) -> torch.Tensor:
    """Reward transport after a lift or, when sensors are supplied, a real grasped lift."""
    gate = _object_lift_memory_gate(
        env,
        min_height,
        lift_margin,
        object_cfg,
        sensor_cfg,
        right_sensor_cfg,
        force_threshold,
        required_contacts,
    )
    return gate * object_to_goal_3d_exp(env, std, goal_offset, object_cfg)


def object_velocity_towards_goal_after_lift_exp(
    env: ManagerBasedRLEnv,
    std: float,
    goal_offset: tuple[float, float, float],
    target_speed: float,
    min_height: float,
    lift_margin: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    sensor_cfg: SceneEntityCfg | None = None,
    right_sensor_cfg: SceneEntityCfg | None = None,
    force_threshold: float = 0.5,
    required_contacts: int = 2,
) -> torch.Tensor:
    """Reward goal-directed velocity after a lift or sensor-confirmed grasped lift."""
    gate = _object_lift_memory_gate(
        env,
        min_height,
        lift_margin,
        object_cfg,
        sensor_cfg,
        right_sensor_cfg,
        force_threshold,
        required_contacts,
    )
    return gate * object_velocity_towards_goal_exp(env, std, goal_offset, target_speed, object_cfg)


def object_lifted_exp(
    env: ManagerBasedRLEnv,
    std: float,
    target_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward lifting the object to at least the requested world-frame height."""
    obj: RigidObject = env.scene[object_cfg.name]
    height_error = torch.clamp(float(target_height) - obj.data.root_pos_w[:, 2], min=0.0)
    return torch.exp(-torch.square(height_error) / float(std) ** 2)


def object_lifted_above_height_exp(
    env: ManagerBasedRLEnv,
    std: float,
    min_height: float,
    target_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    active_pick_stages: tuple[int, ...] | None = None,
    pick_stage_params: dict | None = None,
) -> torch.Tensor:
    """Reward lift progress, normalized to be zero at the initial support height."""
    obj: RigidObject = env.scene[object_cfg.name]
    height_error = torch.clamp(float(target_height) - obj.data.root_pos_w[:, 2], min=0.0)
    reward = torch.exp(-torch.square(height_error) / float(std) ** 2)
    baseline_error = max(float(target_height) - float(min_height), 0.0)
    baseline = torch.exp(torch.tensor(-(baseline_error**2) / float(std) ** 2, device=env.device))
    reward = torch.clamp((reward - baseline) / torch.clamp(1.0 - baseline, min=1.0e-6), min=0.0, max=1.0)
    return _apply_pick_stage_gate(env, reward, active_pick_stages, pick_stage_params)


def object_lifted_near_ee_exp(
    env: ManagerBasedRLEnv,
    std: float,
    target_height: float,
    ee_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward lifting the object while keeping the selected end-effector close to it."""
    robot: Articulation = env.scene[ee_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    ee_pos_w = _selected_body_position_w(robot, ee_cfg)
    height_error = torch.clamp(float(target_height) - obj.data.root_pos_w[:, 2], min=0.0)
    ee_error = torch.linalg.norm(ee_pos_w - obj.data.root_pos_w, dim=1)
    return torch.exp(-(torch.square(height_error) + torch.square(ee_error)) / float(std) ** 2)


def object_lifted_near_ee_above_height_exp(
    env: ManagerBasedRLEnv,
    std: float,
    min_height: float,
    target_height: float,
    ee_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    active_pick_stages: tuple[int, ...] | None = None,
    pick_stage_params: dict | None = None,
) -> torch.Tensor:
    """Reward lifting above the table while keeping the end-effector close to the object."""
    robot: Articulation = env.scene[ee_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    ee_pos_w = _selected_body_position_w(robot, ee_cfg)
    height_error = torch.clamp(float(target_height) - obj.data.root_pos_w[:, 2], min=0.0)
    ee_error = torch.linalg.norm(ee_pos_w - obj.data.root_pos_w, dim=1)
    reward = torch.exp(-(torch.square(height_error) + torch.square(ee_error)) / float(std) ** 2)
    baseline_error = max(float(target_height) - float(min_height), 0.0)
    baseline = torch.exp(torch.tensor(-(baseline_error**2) / float(std) ** 2, device=env.device))
    reward = torch.clamp((reward - baseline) / torch.clamp(1.0 - baseline, min=1.0e-6), min=0.0, max=1.0)
    return _apply_pick_stage_gate(env, reward, active_pick_stages, pick_stage_params)


def object_vertical_exp(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    active_pick_stages: tuple[int, ...] | None = None,
    pick_stage_params: dict | None = None,
) -> torch.Tensor:
    """Reward keeping a cylinder/can object's local z-axis aligned with world up."""
    obj: RigidObject = env.scene[object_cfg.name]
    up_o = torch.zeros((env.num_envs, 3), device=env.device, dtype=obj.data.root_quat_w.dtype)
    up_o[:, 2] = 1.0
    up_w = quat_apply(obj.data.root_quat_w, up_o)
    tilt_error = torch.sum(torch.square(up_w[:, :2]), dim=1)
    reward = torch.exp(-tilt_error / float(std) ** 2)
    return _apply_pick_stage_gate(env, reward, active_pick_stages, pick_stage_params)


def _base_standoff_gate(
    env: ManagerBasedRLEnv,
    standoff_distance: float,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    standoff_error = base_to_object_standoff_b(env, standoff_distance, object_cfg, robot_cfg)
    return torch.exp(-torch.sum(torch.square(standoff_error[:, :2]), dim=1) / float(std) ** 2)


def ee_to_object_after_standoff_exp(
    env: ManagerBasedRLEnv,
    std: float,
    standoff_distance: float,
    standoff_std: float,
    ee_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward reaching to the object only once the base is in a useful manipulation pose."""
    gate = _base_standoff_gate(env, standoff_distance, standoff_std, object_cfg, robot_cfg)
    return gate * ee_to_object_exp(env, std, ee_cfg, object_cfg)


def ee_to_object_after_standoff_tanh(
    env: ManagerBasedRLEnv,
    std: float,
    standoff_distance: float,
    standoff_std: float,
    ee_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Dense reach shaping that turns on after the base reaches the standoff pose."""
    gate = _base_standoff_gate(env, standoff_distance, standoff_std, object_cfg, robot_cfg)
    return gate * ee_to_object_tanh(env, std, ee_cfg, object_cfg)


def object_lifted_after_standoff_exp(
    env: ManagerBasedRLEnv,
    std: float,
    min_height: float,
    target_height: float,
    standoff_distance: float,
    standoff_std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward lift progress only after the base has reached its manipulation standoff."""
    gate = _base_standoff_gate(env, standoff_distance, standoff_std, object_cfg, robot_cfg)
    return gate * object_lifted_above_height_exp(env, std, min_height, target_height, object_cfg)


def object_lifted_near_ee_after_standoff_exp(
    env: ManagerBasedRLEnv,
    std: float,
    min_height: float,
    target_height: float,
    standoff_distance: float,
    standoff_std: float,
    ee_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward grasped lifting only after the base has reached its manipulation standoff."""
    gate = _base_standoff_gate(env, standoff_distance, standoff_std, object_cfg, robot_cfg)
    lifted = object_lifted_near_ee_above_height_exp(env, std, min_height, target_height, ee_cfg, object_cfg)
    return gate * lifted


def arm_joint_deviation_before_standoff_l2(
    env: ManagerBasedRLEnv,
    standoff_distance: float,
    standoff_std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Keep the arm near its reset pose while the base is still approaching the table."""
    robot: Articulation = env.scene[asset_cfg.name]
    gate = _base_standoff_gate(env, standoff_distance, standoff_std, object_cfg, robot_cfg)
    joint_error = robot.data.joint_pos[:, asset_cfg.joint_ids] - robot.data.default_joint_pos[:, asset_cfg.joint_ids]
    return (1.0 - gate) * torch.sum(torch.square(joint_error), dim=1)


def base_xy_velocity_l2(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize planar base drift for stationary manipulation."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.root_lin_vel_b[:, :2]), dim=1)


def base_xy_velocity_in_manipulation_reach_l2(
    env: ManagerBasedRLEnv,
    manipulation_reach: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base motion only while the arm should own the reaching motion."""
    robot: Articulation = env.scene[robot_cfg.name]
    gate = _manipulation_reach_gate(env, manipulation_reach, object_cfg, robot_cfg)
    return gate * torch.sum(torch.square(robot.data.root_lin_vel_b[:, :2]), dim=1)


def base_xy_drift_in_manipulation_reach_l2(
    env: ManagerBasedRLEnv,
    manipulation_reach: float,
    drift_std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize drift from the pose where the manipulation skill became active."""
    robot: Articulation = env.scene[robot_cfg.name]
    gate = _manipulation_reach_gate(env, manipulation_reach, object_cfg, robot_cfg)
    active = gate > 0.5

    if not hasattr(env, "_loco_manip_parked_base_xy"):
        env._loco_manip_parked_base_xy = robot.data.root_pos_w[:, :2].clone()
        env._loco_manip_was_in_reach = active.clone()
        env._loco_manip_last_episode_step = env.episode_length_buf.clone()

    episode_restarted = env.episode_length_buf <= env._loco_manip_last_episode_step
    entered_reach = active & (~env._loco_manip_was_in_reach | episode_restarted)
    env._loco_manip_parked_base_xy[entered_reach] = robot.data.root_pos_w[entered_reach, :2]
    env._loco_manip_was_in_reach.copy_(active)
    env._loco_manip_last_episode_step.copy_(env.episode_length_buf)

    drift = robot.data.root_pos_w[:, :2] - env._loco_manip_parked_base_xy
    return gate * torch.sum(torch.square(drift / float(drift_std)), dim=1)


def base_to_object_standoff_outside_manipulation_reach_exp(
    env: ManagerBasedRLEnv,
    standoff_distance: float,
    manipulation_reach: float,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    active_pick_stages: tuple[int, ...] | None = None,
    pick_stage_params: dict | None = None,
) -> torch.Tensor:
    """Reward base approach only while the object is outside arm reach."""
    manipulate_gate = _manipulation_reach_gate(env, manipulation_reach, object_cfg, robot_cfg)
    approach_reward = base_to_object_standoff_exp(env, standoff_distance, std, object_cfg, robot_cfg)
    reward = (1.0 - manipulate_gate) * approach_reward
    return _apply_pick_stage_gate(env, reward, active_pick_stages, pick_stage_params)


def base_xy_velocity_near_standoff_l2(
    env: ManagerBasedRLEnv,
    standoff_distance: float,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize planar base drift mostly after the robot has reached its standoff pose."""
    robot: Articulation = env.scene[robot_cfg.name]
    standoff_error = base_to_object_standoff_b(env, standoff_distance, object_cfg, robot_cfg)
    near_standoff = torch.exp(-torch.sum(torch.square(standoff_error[:, :2]), dim=1) / float(std) ** 2)
    return near_standoff * torch.sum(torch.square(robot.data.root_lin_vel_b[:, :2]), dim=1)


def base_to_object_standoff_exp(
    env: ManagerBasedRLEnv,
    standoff_distance: float,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward walking the base to a stable manipulation standoff pose behind the object."""
    standoff_error = base_to_object_standoff_b(env, standoff_distance, object_cfg, robot_cfg)
    return torch.exp(-torch.sum(torch.square(standoff_error[:, :2]), dim=1) / float(std) ** 2)


def base_faces_object_exp(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Reward yaw alignment toward the object."""
    object_pos_b = object_position_b(env, object_cfg, robot_cfg)
    yaw_error = torch.atan2(object_pos_b[:, 1], object_pos_b[:, 0])
    return torch.exp(-torch.square(yaw_error) / float(std) ** 2)


def object_clearance_exp(
    env: ManagerBasedRLEnv,
    std: float,
    obstacle_x: float,
    target_height: float,
    x_window: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward object height while it is near a fixed obstacle x location."""
    obj: RigidObject = env.scene[object_cfg.name]
    obstacle_x_w = env.scene.env_origins[:, 0] + float(obstacle_x)
    near_obstacle = torch.exp(-torch.square(obj.data.root_pos_w[:, 0] - obstacle_x_w) / float(x_window) ** 2)
    height_error = torch.clamp(float(target_height) - obj.data.root_pos_w[:, 2], min=0.0)
    return near_obstacle * torch.exp(-torch.square(height_error) / float(std) ** 2)


def object_in_bin(
    env: ManagerBasedRLEnv,
    bin_center: tuple[float, float, float],
    bin_half_size: tuple[float, float, float],
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Binary success when the object is inside an axis-aligned bin region."""
    obj: RigidObject = env.scene[object_cfg.name]
    center = _fixed_goal_w(env, bin_center, obj.data.root_pos_w.dtype)
    half_size = torch.tensor(bin_half_size, device=env.device, dtype=obj.data.root_pos_w.dtype)
    error = torch.abs(obj.data.root_pos_w - center)
    return torch.all(error <= half_size, dim=1).to(dtype=obj.data.root_pos_w.dtype)


def object_in_bin_after_lift(
    env: ManagerBasedRLEnv,
    bin_center: tuple[float, float, float],
    bin_half_size: tuple[float, float, float],
    min_height: float,
    lift_margin: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    sensor_cfg: SceneEntityCfg | None = None,
    right_sensor_cfg: SceneEntityCfg | None = None,
    force_threshold: float = 0.5,
    required_contacts: int = 2,
) -> torch.Tensor:
    """Reward an in-bin object after a lift or sensor-confirmed grasped lift."""
    gate = _object_lift_memory_gate(
        env,
        min_height,
        lift_margin,
        object_cfg,
        sensor_cfg,
        right_sensor_cfg,
        force_threshold,
        required_contacts,
    )
    return gate * object_in_bin(env, bin_center, bin_half_size, object_cfg)


def object_released_in_bin_after_grasp(
    env: ManagerBasedRLEnv,
    bin_center: tuple[float, float, float],
    bin_half_size: tuple[float, float, float],
    min_height: float,
    lift_margin: float,
    sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg | None = None,
    force_threshold: float = 0.5,
    required_contacts: int = 2,
    max_speed: float = 0.20,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Binary placed-object success: grasped once, in the bin, released, and settled."""
    obj: RigidObject = env.scene[object_cfg.name]
    grasped_once = _object_lift_memory_gate(
        env,
        min_height,
        lift_margin,
        object_cfg,
        sensor_cfg,
        right_sensor_cfg,
        force_threshold,
        required_contacts,
    )
    sensor_cfgs = (sensor_cfg,) if right_sensor_cfg is None else (sensor_cfg, right_sensor_cfg)
    touching = _any_filtered_contact_gate(env, sensor_cfgs, force_threshold) > 0.5
    settled = torch.linalg.norm(obj.data.root_lin_vel_w, dim=1) <= float(max_speed)
    released = (~touching) & settled
    in_bin = object_in_bin(env, bin_center, bin_half_size, object_cfg) > 0.5
    return (grasped_once > 0.5) & in_bin & released


def gripper_open_after_grasp_near_goal(
    env: ManagerBasedRLEnv,
    bin_center: tuple[float, float, float],
    bin_half_size: tuple[float, float, float],
    min_height: float,
    lift_margin: float,
    gripper_cfg: SceneEntityCfg,
    sensor_cfg: SceneEntityCfg,
    right_sensor_cfg: SceneEntityCfg | None = None,
    force_threshold: float = 0.5,
    required_contacts: int = 2,
    open_joint_pos: float | tuple[float, ...] = 0.0,
    closed_joint_pos: float | tuple[float, ...] = 0.03,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward opening the gripper only after a grasped object reaches the placement region."""
    robot: Articulation = env.scene[gripper_cfg.name]
    joint_pos = robot.data.joint_pos[:, gripper_cfg.joint_ids]
    open_pos = torch.as_tensor(open_joint_pos, device=joint_pos.device, dtype=joint_pos.dtype)
    closed_pos = torch.as_tensor(closed_joint_pos, device=joint_pos.device, dtype=joint_pos.dtype)
    travel = closed_pos - open_pos
    travel = torch.where(torch.abs(travel) < 1.0e-6, torch.ones_like(travel), travel)
    close_fraction = torch.clamp((joint_pos - open_pos) / travel, min=0.0, max=1.0).mean(dim=-1)
    grasped_once = _object_lift_memory_gate(
        env,
        min_height,
        lift_margin,
        object_cfg,
        sensor_cfg,
        right_sensor_cfg,
        force_threshold,
        required_contacts,
    )
    return grasped_once * object_in_bin(env, bin_center, bin_half_size, object_cfg) * (1.0 - close_fraction)


def pick_place_stage_progress(
    env: ManagerBasedRLEnv,
    goal_offset: tuple[float, float, float],
    bin_half_size: tuple[float, float, float],
    min_height: float,
    lift_margin: float,
    standoff_distance: float | None = None,
    standoff_std: float = 0.28,
    reach_std: float = 0.25,
    goal_std: float = 0.20,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    right_sensor_cfg: SceneEntityCfg | None = None,
    force_threshold: float = 0.5,
    required_contacts: int = 2,
) -> torch.Tensor:
    """Normalized stage progress for approach, reach, lift, transport, and place."""
    if standoff_distance is None:
        at_standoff = torch.ones(env.num_envs, device=env.device)
    else:
        at_standoff = _base_standoff_gate(env, standoff_distance, standoff_std, object_cfg, robot_cfg)

    robot: Articulation = env.scene[ee_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    ee_pos_w = _selected_body_position_w(robot, ee_cfg)
    ee_to_object = torch.linalg.norm(ee_pos_w - obj.data.root_pos_w, dim=1)
    reached = 1.0 - torch.tanh(ee_to_object / float(reach_std))
    lifted = _object_lift_gate(env, min_height, lift_margin, object_cfg)
    at_goal = object_to_goal_3d_exp(env, goal_std, goal_offset, object_cfg)
    lifted_once = _object_lift_memory_gate(
        env,
        min_height,
        lift_margin,
        object_cfg,
        sensor_cfg,
        right_sensor_cfg,
        force_threshold,
        required_contacts,
    )
    in_bin = lifted_once * object_in_bin(env, goal_offset, bin_half_size, object_cfg)

    stage = (
        (at_standoff > 0.55).to(dtype=obj.data.root_pos_w.dtype)
        + (reached > 0.45).to(dtype=obj.data.root_pos_w.dtype)
        + (lifted > 0.65).to(dtype=obj.data.root_pos_w.dtype)
        + (at_goal > 0.55).to(dtype=obj.data.root_pos_w.dtype)
        + (in_bin > 0.5).to(dtype=obj.data.root_pos_w.dtype)
    )
    return stage / 5.0


def pick_place_stage_state(
    env: ManagerBasedRLEnv,
    goal_offset: tuple[float, float, float],
    bin_half_size: tuple[float, float, float],
    min_height: float,
    lift_margin: float,
    standoff_distance: float | None = None,
    standoff_std: float = 0.28,
    reach_std: float = 0.25,
    goal_std: float = 0.20,
    ee_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
    right_sensor_cfg: SceneEntityCfg | None = None,
    force_threshold: float = 0.5,
    required_contacts: int = 2,
) -> torch.Tensor:
    """Compact stage cues for long-horizon pick-place learning."""
    progress = pick_place_stage_progress(
        env,
        goal_offset=goal_offset,
        bin_half_size=bin_half_size,
        min_height=min_height,
        lift_margin=lift_margin,
        standoff_distance=standoff_distance,
        standoff_std=standoff_std,
        reach_std=reach_std,
        goal_std=goal_std,
        ee_cfg=ee_cfg,
        object_cfg=object_cfg,
        robot_cfg=robot_cfg,
        sensor_cfg=sensor_cfg,
        right_sensor_cfg=right_sensor_cfg,
        force_threshold=force_threshold,
        required_contacts=required_contacts,
    )
    lifted = _object_lift_gate(env, min_height, lift_margin, object_cfg)
    in_bin = object_in_bin(env, goal_offset, bin_half_size, object_cfg)
    return torch.stack((progress, lifted, in_bin), dim=1)


def object_upright_exp(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward keeping the object upright."""
    obj: RigidObject = env.scene[object_cfg.name]
    up_w = torch.zeros((env.num_envs, 3), device=env.device, dtype=obj.data.root_quat_w.dtype)
    up_w[:, 2] = 1.0
    up_b = quat_apply_inverse(obj.data.root_quat_w, up_w)
    tilt_error = torch.sum(torch.square(up_b[:, :2]), dim=1)
    return torch.exp(-tilt_error / float(std) ** 2)


def object_out_of_bounds(
    env: ManagerBasedRLEnv,
    max_distance: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate when the object drifts too far from its environment origin."""
    obj: RigidObject = env.scene[object_cfg.name]
    distance = torch.linalg.norm(obj.data.root_pos_w[:, :2] - env.scene.env_origins[:, :2], dim=1)
    return distance > float(max_distance)


def object_below_minimum_height(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Terminate when an object falls below a height relative to its environment origin."""
    obj: RigidObject = env.scene[object_cfg.name]
    height = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]
    return height < float(minimum_height)


def _reset_pick_episode_memory(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """Clear stage, progress, and success state for only the environments being reset."""
    episode_count = getattr(env, "_loco_manip_pick_episode_count", None)
    if not isinstance(episode_count, torch.Tensor):
        episode_count = torch.zeros((), device=env.device, dtype=torch.long)
    env._loco_manip_pick_episode_count = episode_count + env_ids.numel()
    for memory_attr in (
        "_loco_manip_object_lifted_once",
        "_loco_manip_object_grasped_once",
        "_loco_manip_pick_stage",
        "_loco_manip_pick_hold_counter",
        "_loco_manip_pick_success_emitted",
    ):
        memory = getattr(env, memory_attr, None)
        if isinstance(memory, torch.Tensor):
            memory[env_ids] = 0

    for memory_attr in (
        "_loco_manip_prev_base_object_distance",
        "_loco_manip_prev_ee_object_distance",
        "_loco_manip_prev_gripper_close_fraction",
        "_loco_manip_prev_gripper_contact_fraction",
    ):
        memory = getattr(env, memory_attr, None)
        if isinstance(memory, torch.Tensor):
            memory[env_ids] = torch.nan

    for target_stage in (1, 2, 3):
        memory = getattr(env, f"_loco_manip_transition_prev_stage_{target_stage}", None)
        if isinstance(memory, torch.Tensor):
            memory[env_ids] = -1
    regression_memory = getattr(env, "_loco_manip_regression_prev_stage", None)
    if isinstance(regression_memory, torch.Tensor):
        regression_memory[env_ids] = -1


def randomize_pick_throw_scene(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    object_cfg: SceneEntityCfg,
    bin_asset_names: tuple[str, ...],
    nominal_bin_center: tuple[float, float, float],
    object_pose_range: dict[str, tuple[float, float]],
    bin_pose_range: dict[str, tuple[float, float]],
    object_velocity_range: dict[str, tuple[float, float]],
) -> None:
    """Randomize the throwable object and move all bin parts as one coherent target."""
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).flatten()
    if env_ids.numel() == 0:
        return
    _reset_pick_episode_memory(env, env_ids)

    def sample_ranges(ranges_dict: dict[str, tuple[float, float]], keys: tuple[str, ...]) -> torch.Tensor:
        ranges = torch.tensor([ranges_dict.get(key, (0.0, 0.0)) for key in keys], device=env.device)
        return ranges[:, 0] + (ranges[:, 1] - ranges[:, 0]) * torch.rand(
            (env_ids.numel(), len(keys)), device=env.device
        )

    object_asset: RigidObject = env.scene[object_cfg.name]
    object_root = object_asset.data.default_root_state[env_ids].clone()
    object_pose_sample = sample_ranges(object_pose_range, ("x", "y", "z", "roll", "pitch", "yaw"))
    object_pos = object_root[:, :3] + env.scene.env_origins[env_ids] + object_pose_sample[:, :3]
    object_quat_delta = quat_from_euler_xyz(
        object_pose_sample[:, 3], object_pose_sample[:, 4], object_pose_sample[:, 5]
    )
    object_quat = quat_mul(object_root[:, 3:7], object_quat_delta)
    object_vel_sample = sample_ranges(object_velocity_range, ("x", "y", "z", "roll", "pitch", "yaw"))
    object_asset.write_root_pose_to_sim(torch.cat((object_pos, object_quat), dim=-1), env_ids=env_ids)
    object_asset.write_root_velocity_to_sim(object_root[:, 7:13] + object_vel_sample, env_ids=env_ids)

    bin_offset = sample_ranges(bin_pose_range, ("x", "y", "z", "roll", "pitch", "yaw"))
    bin_delta_pos = bin_offset[:, :3]
    bin_quat_delta = quat_from_euler_xyz(bin_offset[:, 3], bin_offset[:, 4], bin_offset[:, 5])
    for asset_name in bin_asset_names:
        bin_asset: RigidObject = env.scene[asset_name]
        bin_root = bin_asset.data.default_root_state[env_ids].clone()
        bin_pos = bin_root[:, :3] + env.scene.env_origins[env_ids] + bin_delta_pos
        bin_quat = quat_mul(bin_root[:, 3:7], bin_quat_delta)
        bin_asset.write_root_pose_to_sim(torch.cat((bin_pos, bin_quat), dim=-1), env_ids=env_ids)
        bin_asset.write_root_velocity_to_sim(torch.zeros_like(bin_root[:, 7:13]), env_ids=env_ids)

    if not hasattr(env, "_loco_manip_goal_w"):
        env._loco_manip_goal_w = env.scene.env_origins + torch.tensor(
            nominal_bin_center, device=env.device, dtype=object_asset.data.root_pos_w.dtype
        )
    nominal = torch.tensor(nominal_bin_center, device=env.device, dtype=object_asset.data.root_pos_w.dtype)
    env._loco_manip_goal_w[env_ids] = env.scene.env_origins[env_ids] + nominal + bin_delta_pos


def reset_pick_approach_scene(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    object_cfg: SceneEntityCfg,
    table_asset_name: str,
    table_center: tuple[float, float, float],
    object_center_z: float,
    object_table_offset: tuple[float, float],
    object_distance_range: tuple[float, float],
    object_pose_range: dict[str, tuple[float, float]],
    object_velocity_range: dict[str, tuple[float, float]],
    sample_distance_within_curriculum: bool = False,
    easy_replay_fraction: float = 0.0,
    frontier_fraction: float = 0.0,
) -> None:
    """Reset a movable tabletop/can pair using the current approach-distance curriculum."""
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).flatten()
    if env_ids.numel() == 0:
        return
    _reset_pick_episode_memory(env, env_ids)

    def sample_ranges(ranges_dict: dict[str, tuple[float, float]], keys: tuple[str, ...]) -> torch.Tensor:
        ranges = torch.tensor([ranges_dict.get(key, (0.0, 0.0)) for key in keys], device=env.device)
        return ranges[:, 0] + (ranges[:, 1] - ranges[:, 0]) * torch.rand(
            (env_ids.numel(), len(keys)), device=env.device
        )

    object_asset: RigidObject = env.scene[object_cfg.name]
    dtype = object_asset.data.root_pos_w.dtype
    frontier_distance = getattr(env, "_go2_d1_pick_object_distance", float(object_distance_range[0]))
    if isinstance(frontier_distance, torch.Tensor):
        frontier_distance = float(frontier_distance.detach().mean().item())
    min_distance = float(object_distance_range[0])
    max_distance = float(object_distance_range[1])
    frontier_distance = max(min_distance, min(max_distance, float(frontier_distance)))
    if easy_replay_fraction < 0.0 or frontier_fraction < 0.0 or easy_replay_fraction + frontier_fraction > 1.0:
        raise ValueError("easy_replay_fraction and frontier_fraction must be non-negative and sum to at most one.")
    if sample_distance_within_curriculum:
        distance = min_distance + (frontier_distance - min_distance) * torch.rand(
            env_ids.numel(), device=env.device, dtype=dtype
        )
        mixture = torch.rand(env_ids.numel(), device=env.device)
        easy = mixture < float(easy_replay_fraction)
        frontier = (mixture >= float(easy_replay_fraction)) & (
            mixture < float(easy_replay_fraction) + float(frontier_fraction)
        )
        distance[easy] = min_distance
        distance[frontier] = frontier_distance
    else:
        distance = torch.full((env_ids.numel(),), frontier_distance, device=env.device, dtype=dtype)

    table_asset: RigidObject = env.scene[table_asset_name]
    table_root = table_asset.data.default_root_state[env_ids].clone()
    table_pos = env.scene.env_origins[env_ids] + torch.tensor(table_center, device=env.device, dtype=dtype)
    table_pos[:, 0] = env.scene.env_origins[env_ids, 0] + distance
    table_asset.write_root_pose_to_sim(torch.cat((table_pos, table_root[:, 3:7]), dim=-1), env_ids=env_ids)

    object_root = object_asset.data.default_root_state[env_ids].clone()
    object_pose_sample = sample_ranges(object_pose_range, ("x", "y", "z", "roll", "pitch", "yaw"))
    nominal_object_pos = torch.empty((env_ids.numel(), 3), device=env.device, dtype=dtype)
    nominal_object_pos[:, 0] = distance + float(object_table_offset[0])
    nominal_object_pos[:, 1] = float(table_center[1]) + float(object_table_offset[1])
    nominal_object_pos[:, 2] = float(object_center_z)
    object_pos = env.scene.env_origins[env_ids] + nominal_object_pos + object_pose_sample[:, :3]
    object_quat_delta = quat_from_euler_xyz(
        object_pose_sample[:, 3], object_pose_sample[:, 4], object_pose_sample[:, 5]
    )
    object_quat = quat_mul(object_root[:, 3:7], object_quat_delta)
    object_vel_sample = sample_ranges(object_velocity_range, ("x", "y", "z", "roll", "pitch", "yaw"))
    object_asset.write_root_pose_to_sim(torch.cat((object_pos, object_quat), dim=-1), env_ids=env_ids)
    object_asset.write_root_velocity_to_sim(object_root[:, 7:13] + object_vel_sample, env_ids=env_ids)


def reset_pick_place_approach_scene(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    object_cfg: SceneEntityCfg,
    table_asset_name: str,
    table_center: tuple[float, float, float],
    object_center_z: float,
    object_table_offset: tuple[float, float],
    object_distance_range: tuple[float, float],
    bin_asset_names: tuple[str, ...],
    nominal_bin_center: tuple[float, float, float],
    object_pose_range: dict[str, tuple[float, float]],
    bin_pose_range: dict[str, tuple[float, float]],
    object_velocity_range: dict[str, tuple[float, float]],
) -> None:
    """Reset a table/can/tray scene while moving the full task farther from the robot over training."""
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).flatten()
    if env_ids.numel() == 0:
        return
    _reset_pick_episode_memory(env, env_ids)

    def sample_ranges(ranges_dict: dict[str, tuple[float, float]], keys: tuple[str, ...]) -> torch.Tensor:
        ranges = torch.tensor([ranges_dict.get(key, (0.0, 0.0)) for key in keys], device=env.device)
        return ranges[:, 0] + (ranges[:, 1] - ranges[:, 0]) * torch.rand(
            (env_ids.numel(), len(keys)), device=env.device
        )

    object_asset: RigidObject = env.scene[object_cfg.name]
    dtype = object_asset.data.root_pos_w.dtype
    distance = getattr(env, "_go2_d1_pick_object_distance", float(object_distance_range[0]))
    if isinstance(distance, torch.Tensor):
        distance = float(distance.detach().mean().item())
    distance = max(float(object_distance_range[0]), min(float(object_distance_range[1]), float(distance)))

    table_asset: RigidObject = env.scene[table_asset_name]
    table_root = table_asset.data.default_root_state[env_ids].clone()
    table_pos = env.scene.env_origins[env_ids] + torch.tensor(table_center, device=env.device, dtype=dtype)
    table_pos[:, 0] = env.scene.env_origins[env_ids, 0] + distance
    table_asset.write_root_pose_to_sim(torch.cat((table_pos, table_root[:, 3:7]), dim=-1), env_ids=env_ids)

    object_root = object_asset.data.default_root_state[env_ids].clone()
    object_pose_sample = sample_ranges(object_pose_range, ("x", "y", "z", "roll", "pitch", "yaw"))
    nominal_object_pos = torch.tensor(
        (distance + float(object_table_offset[0]), table_center[1] + float(object_table_offset[1]), object_center_z),
        device=env.device,
        dtype=dtype,
    ).repeat(env_ids.numel(), 1)
    object_pos = env.scene.env_origins[env_ids] + nominal_object_pos + object_pose_sample[:, :3]
    object_quat_delta = quat_from_euler_xyz(
        object_pose_sample[:, 3], object_pose_sample[:, 4], object_pose_sample[:, 5]
    )
    object_quat = quat_mul(object_root[:, 3:7], object_quat_delta)
    object_vel_sample = sample_ranges(object_velocity_range, ("x", "y", "z", "roll", "pitch", "yaw"))
    object_asset.write_root_pose_to_sim(torch.cat((object_pos, object_quat), dim=-1), env_ids=env_ids)
    object_asset.write_root_velocity_to_sim(object_root[:, 7:13] + object_vel_sample, env_ids=env_ids)

    bin_offset = sample_ranges(bin_pose_range, ("x", "y", "z", "roll", "pitch", "yaw"))
    scene_delta = bin_offset[:, :3]
    scene_delta[:, 0] += distance - float(table_center[0])
    bin_quat_delta = quat_from_euler_xyz(bin_offset[:, 3], bin_offset[:, 4], bin_offset[:, 5])
    for asset_name in bin_asset_names:
        bin_asset: RigidObject = env.scene[asset_name]
        bin_root = bin_asset.data.default_root_state[env_ids].clone()
        bin_pos = bin_root[:, :3] + env.scene.env_origins[env_ids] + scene_delta
        bin_quat = quat_mul(bin_root[:, 3:7], bin_quat_delta)
        bin_asset.write_root_pose_to_sim(torch.cat((bin_pos, bin_quat), dim=-1), env_ids=env_ids)
        bin_asset.write_root_velocity_to_sim(torch.zeros_like(bin_root[:, 7:13]), env_ids=env_ids)

    if not hasattr(env, "_loco_manip_goal_w"):
        env._loco_manip_goal_w = env.scene.env_origins + torch.tensor(
            nominal_bin_center, device=env.device, dtype=dtype
        )
    nominal = torch.tensor(nominal_bin_center, device=env.device, dtype=dtype)
    env._loco_manip_goal_w[env_ids] = env.scene.env_origins[env_ids] + nominal + scene_delta


def sync_kinematic_object_to_body(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    object_cfg: SceneEntityCfg,
    body_cfg: SceneEntityCfg,
    pos_offset: tuple[float, float, float],
    rot_offset_rpy: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> None:
    """Attach a kinematic rigid object to a robot body with a fixed local offset."""
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device, dtype=torch.long)
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).flatten()
    if env_ids.numel() == 0:
        return

    robot: Articulation = env.scene[body_cfg.name]
    obj: RigidObject = env.scene[object_cfg.name]
    body_id = body_cfg.body_ids[0]
    body_pos_w = robot.data.body_pos_w[env_ids, body_id, :]
    body_quat_w = robot.data.body_quat_w[env_ids, body_id, :]
    offset_pos_b = torch.tensor(pos_offset, device=env.device, dtype=body_pos_w.dtype).repeat(env_ids.numel(), 1)
    roll = torch.full((env_ids.numel(),), float(rot_offset_rpy[0]), device=env.device, dtype=body_pos_w.dtype)
    pitch = torch.full((env_ids.numel(),), float(rot_offset_rpy[1]), device=env.device, dtype=body_pos_w.dtype)
    yaw = torch.full((env_ids.numel(),), float(rot_offset_rpy[2]), device=env.device, dtype=body_pos_w.dtype)
    offset_quat_b = quat_from_euler_xyz(roll, pitch, yaw)
    obj_pos_w, obj_quat_w = combine_frame_transforms(body_pos_w, body_quat_w, offset_pos_b, offset_quat_b)
    obj.write_root_pose_to_sim(torch.cat((obj_pos_w, obj_quat_w), dim=-1), env_ids=env_ids)

    velocity = torch.zeros((env_ids.numel(), 6), device=env.device, dtype=body_pos_w.dtype)
    if hasattr(robot.data, "body_lin_vel_w"):
        velocity[:, :3] = robot.data.body_lin_vel_w[env_ids, body_id, :]
    if hasattr(robot.data, "body_ang_vel_w"):
        velocity[:, 3:] = robot.data.body_ang_vel_w[env_ids, body_id, :]
    obj.write_root_velocity_to_sim(velocity, env_ids=env_ids)
