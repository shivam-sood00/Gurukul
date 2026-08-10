"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import torch

from isaaclab.assets import Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.terrains import TerrainImporter
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _episode_boundary(env: ManagerBasedRLEnv) -> bool:
    """Return true at the coarse update cadence used by reward-sum curricula."""
    episode_length = max(1, int(env.max_episode_length))
    return int(env.common_step_counter) % episode_length == 0


def _coarse_update_due(env: ManagerBasedRLEnv, step: int, last_update_attr: str) -> bool:
    """Run at most once per episode horizon without requiring exact global-step alignment."""
    interval = max(1, int(env.max_episode_length))
    last_update = int(getattr(env, last_update_attr, -interval))
    if int(step) - last_update < interval:
        return False
    setattr(env, last_update_attr, int(step))
    return True


def _positive_reward_score(env: ManagerBasedRLEnv, env_ids: Sequence[int], reward_term_names: Sequence[str]) -> float:
    """Return the weakest normalized positive reward score across selected reward terms."""
    scores: list[float] = []
    for reward_term_name in reward_term_names:
        if reward_term_name not in env.reward_manager._episode_sums:
            continue
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        weight = float(reward_term_cfg.weight)
        if weight <= 0.0:
            continue
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_rate = torch.mean(episode_sums[env_ids]) / max(float(env.max_episode_length_s), 1.0e-6)
        scores.append(float(reward_rate / max(weight, 1.0e-6)))
    return min(scores) if scores else 0.0


def env_attr_curriculum_metric(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    attr_name: str,
    default: float = 0.0,
) -> torch.Tensor:
    """Log a scalar environment attribute through the curriculum manager."""
    value = getattr(env, attr_name, default)
    if isinstance(value, torch.Tensor):
        if isinstance(env_ids, torch.Tensor):
            env_ids_t = env_ids.to(device=value.device, dtype=torch.long)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=value.device, dtype=torch.long)
        selected = value.reshape(-1).index_select(0, env_ids_t) if value.numel() >= env_ids_t.numel() else value
        return torch.mean(selected.to(device=env.device, dtype=torch.float32))
    return torch.tensor(float(value), device=env.device)


def pick_stage_fraction(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    stage_index: int,
    pick_stage_params: dict,
) -> torch.Tensor:
    """Log current full-batch pick-stage occupancy.

    Curriculum terms receive the environments resetting on the current step.
    Restricting this diagnostic to those ids made it look like later stages had
    zero occupancy even while other rollout environments were grasping.
    """
    del env_ids
    from .loco_manipulation import pick_stage_index as compute_pick_stage

    stage = compute_pick_stage(env, **pick_stage_params)
    return torch.mean((stage == int(stage_index)).float())


def normalized_action_saturation_fraction(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    action_indices: Sequence[int],
    threshold: float = 0.98,
) -> torch.Tensor:
    """Log how often selected normalized high-level actions sit near their bounds."""
    actions = env.action_manager.action
    selected_ids = torch.as_tensor(env_ids, device=actions.device, dtype=torch.long)
    selected_actions = actions.index_select(0, selected_ids)
    valid_indices = tuple(index for index in action_indices if 0 <= int(index) < actions.shape[1])
    if selected_actions.numel() == 0 or not valid_indices:
        return torch.tensor(0.0, device=env.device)
    indices = torch.as_tensor(valid_indices, device=actions.device, dtype=torch.long)
    return torch.mean((selected_actions.index_select(1, indices).abs() >= float(threshold)).float())


def normalized_action_diagnostics(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    action_name: str,
    saturation_threshold: float = 1.0,
) -> dict[str, torch.Tensor]:
    """Log raw normalized-action magnitude and saturation diagnostics."""
    action_term = env.action_manager.get_term(action_name)
    actions = action_term.raw_actions
    selected_ids = torch.as_tensor(env_ids, device=actions.device, dtype=torch.long).flatten()
    selected = actions.index_select(0, selected_ids)
    if selected.numel() == 0:
        zero = torch.tensor(0.0, device=env.device)
        return {"abs_mean": zero, "abs_max": zero, "outside_fraction": zero}
    absolute = selected.abs()
    return {
        "abs_mean": absolute.mean(),
        "abs_max": absolute.max(),
        "outside_fraction": (absolute > float(saturation_threshold)).float().mean(),
    }


def velocity_command_diagnostics(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    command_name: str = "base_velocity",
    planar_threshold: float = 0.2,
    yaw_threshold: float = 0.05,
) -> dict[str, torch.Tensor]:
    """Log commanded/actual motion and rotation-only sampling fractions."""
    command_term = env.command_manager.get_term(command_name)
    commands = command_term.vel_command_b
    selected_ids = torch.as_tensor(env_ids, device=commands.device, dtype=torch.long).flatten()
    if selected_ids.numel() == 0:
        zero = torch.tensor(0.0, device=env.device)
        return {
            "target_planar_speed": zero,
            "actual_planar_speed": zero,
            "target_yaw_speed": zero,
            "actual_yaw_speed": zero,
            "translation_fraction": zero,
            "rotation_only_fraction": zero,
            "standing_fraction": zero,
        }

    selected_commands = commands.index_select(0, selected_ids)
    target_planar = torch.linalg.vector_norm(selected_commands[:, :2], dim=1)
    target_yaw = selected_commands[:, 2].abs()
    actual_planar = torch.linalg.vector_norm(
        command_term.robot.data.root_lin_vel_b.index_select(0, selected_ids)[:, :2], dim=1
    )
    actual_yaw = command_term.robot.data.root_ang_vel_b.index_select(0, selected_ids)[:, 2].abs()
    translation = target_planar > float(planar_threshold)
    rotation = target_yaw > float(yaw_threshold)
    return {
        "target_planar_speed": target_planar.mean(),
        "actual_planar_speed": actual_planar.mean(),
        "target_yaw_speed": target_yaw.mean(),
        "actual_yaw_speed": actual_yaw.mean(),
        "translation_fraction": translation.float().mean(),
        "rotation_only_fraction": ((~translation) & rotation).float().mean(),
        "standing_fraction": ((~translation) & (~rotation)).float().mean(),
    }


def async_arm_motion_diagnostics(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    near_limit_threshold: float = 0.90,
) -> dict[str, torch.Tensor]:
    """Log full-batch planned arm speed and active-trajectory fractions."""
    del env_ids
    required = (
        "_arm_async_joint_initialized",
        "_arm_async_joint_peak_velocity_fraction",
        "_arm_async_joint_progress",
        "_arm_async_joint_duration",
        "_arm_async_joint_start_pos",
        "_arm_async_joint_goal_pos",
    )
    if any(not hasattr(env, attr_name) for attr_name in required):
        zero = torch.tensor(0.0, device=env.device)
        return {
            "planned_peak_fraction_mean": zero,
            "planned_peak_fraction_max": zero,
            "near_limit_fraction": zero,
            "active_fraction": zero,
        }

    initialized = env._arm_async_joint_initialized
    if not torch.any(initialized):
        zero = torch.tensor(0.0, device=env.device)
        return {
            "planned_peak_fraction_mean": zero,
            "planned_peak_fraction_max": zero,
            "near_limit_fraction": zero,
            "active_fraction": zero,
        }

    peak_fraction = env._arm_async_joint_peak_velocity_fraction[initialized]
    moving = (
        torch.abs(
            env._arm_async_joint_goal_pos[initialized] - env._arm_async_joint_start_pos[initialized]
        )
        > 1.0e-5
    )
    active = moving & (
        env._arm_async_joint_progress[initialized] < env._arm_async_joint_duration[initialized]
    )
    moving_peak_fraction = peak_fraction[moving]
    if moving_peak_fraction.numel() == 0:
        mean_peak = torch.tensor(0.0, device=env.device)
        max_peak = mean_peak
        near_limit = mean_peak
    else:
        mean_peak = moving_peak_fraction.mean()
        max_peak = moving_peak_fraction.max()
        near_limit = (moving_peak_fraction >= float(near_limit_threshold)).float().mean()
    return {
        "planned_peak_fraction_mean": mean_peak,
        "planned_peak_fraction_max": max_peak,
        "near_limit_fraction": near_limit,
        "active_fraction": active.float().mean(),
    }


def joint_target_limit_diagnostics(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    action_name: str,
    soft_factor: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> dict[str, torch.Tensor]:
    """Log target and measured utilization of task-specific soft joint limits."""
    if not 0.0 < float(soft_factor) <= 1.0:
        raise ValueError(f"soft_factor must be in (0, 1], got {soft_factor}.")
    asset: Articulation = env.scene[asset_cfg.name]
    action_term = env.action_manager.get_term(action_name)
    targets = action_term.processed_actions
    selected_ids = torch.as_tensor(env_ids, device=targets.device, dtype=torch.long).flatten()
    if selected_ids.numel() == 0:
        zero = torch.tensor(0.0, device=env.device)
        return {
            "target_utilization_mean": zero,
            "target_utilization_max": zero,
            "target_violation_fraction": zero,
            "measured_utilization_max": zero,
        }

    joint_ids = getattr(action_term, "_joint_ids", asset_cfg.joint_ids)
    if isinstance(joint_ids, slice):
        joint_ids = torch.arange(asset.num_joints, device=targets.device, dtype=torch.long)
    else:
        joint_ids = torch.as_tensor(joint_ids, device=targets.device, dtype=torch.long)
    if targets.shape[1] != joint_ids.numel():
        raise ValueError(
            f"Action term '{action_name}' exposes {targets.shape[1]} targets for {joint_ids.numel()} resolved joints."
        )

    hard_limits = asset.data.default_joint_pos_limits.index_select(0, selected_ids).index_select(1, joint_ids)
    center = 0.5 * (hard_limits[..., 0] + hard_limits[..., 1])
    half_range = 0.5 * (hard_limits[..., 1] - hard_limits[..., 0]) * float(soft_factor)
    half_range = half_range.clamp_min(torch.finfo(half_range.dtype).eps)
    selected_targets = targets.index_select(0, selected_ids)
    target_utilization = (selected_targets - center).abs() / half_range
    measured = asset.data.joint_pos.index_select(0, selected_ids).index_select(1, joint_ids)
    measured_utilization = (measured - center).abs() / half_range
    return {
        "target_utilization_mean": target_utilization.mean(),
        "target_utilization_max": target_utilization.max(),
        "target_violation_fraction": (target_utilization > 1.0).float().mean(),
        "measured_utilization_max": measured_utilization.max(),
    }


def cumulative_pick_success_fraction(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
) -> torch.Tensor:
    """Log cumulative successful stable holds divided by completed/reset episodes."""
    del env_ids
    successes = getattr(env, "_loco_manip_pick_success_count", None)
    episodes = getattr(env, "_loco_manip_pick_episode_count", None)
    if not isinstance(successes, torch.Tensor) or not isinstance(episodes, torch.Tensor):
        return torch.tensor(0.0, device=env.device)
    return successes.float() / torch.clamp(episodes.float(), min=1.0)


def bilateral_gripper_contact_fraction(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    sensor_cfg,
    right_sensor_cfg,
    force_threshold: float = 0.35,
) -> torch.Tensor:
    """Log full-batch simultaneous left/right object contact."""
    del env_ids
    from .loco_manipulation import gripper_object_contact

    contact = gripper_object_contact(
        env,
        sensor_cfg,
        right_sensor_cfg=right_sensor_cfg,
        force_threshold=force_threshold,
        required_contacts=2,
    )
    return contact.mean()


def pick_geometry_diagnostics(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    ee_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    action_name: str = "wbc_command",
    pregrasp_distance: float = 0.10,
) -> dict[str, torch.Tensor]:
    """Log distances and command shaping needed to diagnose pick learning."""
    del env_ids
    robot = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    ee_pos_w = robot.data.body_pos_w[:, ee_cfg.body_ids].mean(dim=1)
    base_distance = torch.linalg.vector_norm(obj.data.root_pos_w[:, :2] - robot.data.root_pos_w[:, :2], dim=1)
    ee_distance = torch.linalg.vector_norm(obj.data.root_pos_w - ee_pos_w, dim=1)
    action_term = env.action_manager.get_term(action_name)
    raw = action_term.raw_actions
    applied = action_term.processed_actions
    return {
        "base_object_distance_mean": base_distance.mean(),
        "ee_object_distance_mean": ee_distance.mean(),
        "ee_object_distance_min": ee_distance.min(),
        "pregrasp_fraction": (ee_distance <= float(pregrasp_distance)).float().mean(),
        "raw_applied_delta_mean": torch.abs(raw - applied).mean(),
        "applied_base_command_abs_mean": torch.abs(applied[:, :3]).mean(),
    }


def pick_disturbance_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    score_attr: str = "_go2_d1_pick_approach_score",
    performance_threshold: float = 0.20,
    performance_lower_threshold: float = 0.10,
    progress_step: float = 0.02,
    regress_step: float = 0.01,
    reset_event_name: str = "randomize_reset_base",
    push_event_name: str = "randomize_push_robot",
    reset_velocity_start: float = 0.05,
    reset_velocity_end: float = 0.50,
    push_velocity_end: float = 0.50,
) -> torch.Tensor:
    """Expose reset disturbances and pushes only after stable picks."""
    del env_ids
    progress_attr = "_go2_d1_pick_disturbance_progress"
    progress = float(getattr(env, progress_attr, 0.0))
    step = int(env.common_step_counter)
    if _episode_boundary(env) and getattr(env, "_go2_d1_pick_disturbance_update_step", -1) != step:
        score = float(getattr(env, score_attr, 0.0))
        if score >= float(performance_threshold):
            progress += float(progress_step)
        elif score < float(performance_lower_threshold):
            progress -= float(regress_step)
        progress = float(max(0.0, min(1.0, progress)))
        env._go2_d1_pick_disturbance_update_step = step
        setattr(env, progress_attr, progress)

    reset_magnitude = float(reset_velocity_start) + progress * (
        float(reset_velocity_end) - float(reset_velocity_start)
    )
    # Playback deliberately removes random push events. Keep the curriculum
    # usable in that reduced event graph instead of failing during env.reset().
    # Training still resolves and updates both terms normally.
    try:
        reset_cfg = env.event_manager.get_term_cfg(reset_event_name)
    except ValueError:
        reset_cfg = None
    if reset_cfg is not None:
        for axis in tuple(reset_cfg.params["velocity_range"]):
            reset_cfg.params["velocity_range"][axis] = (-reset_magnitude, reset_magnitude)
        env.event_manager.set_term_cfg(reset_event_name, reset_cfg)

    try:
        push_cfg = env.event_manager.get_term_cfg(push_event_name)
    except ValueError:
        push_cfg = None
    if push_cfg is not None:
        push_magnitude = progress * float(push_velocity_end)
        for axis in tuple(push_cfg.params["velocity_range"]):
            push_cfg.params["velocity_range"][axis] = (-push_magnitude, push_magnitude)
        env.event_manager.set_term_cfg(push_event_name, push_cfg)

    return torch.tensor(progress, device=env.device)


def cts_role_terrain_level(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    role: str,
    teacher_fraction: float = 0.75,
) -> torch.Tensor:
    """Log mean terrain level for the CTS teacher or student rollout partition."""
    del env_ids
    terrain: TerrainImporter = env.scene.terrain
    num_teacher_envs = int(env.num_envs * float(teacher_fraction))
    num_teacher_envs = max(0, min(num_teacher_envs, env.num_envs))

    if role == "teacher":
        levels = terrain.terrain_levels[:num_teacher_envs]
    elif role == "student":
        levels = terrain.terrain_levels[num_teacher_envs:]
    else:
        raise ValueError(f"Unsupported CTS terrain role '{role}'. Expected 'teacher' or 'student'.")

    if levels.numel() == 0:
        return torch.tensor(0.0, device=env.device)
    return torch.mean(levels.float())


def linear_reward_weight_by_iteration(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    term_name: str,
    start_iteration: float,
    end_iteration: float,
    start_weight: float,
    end_weight: float,
    steps_per_iteration: int = 1,
) -> torch.Tensor:
    """Linearly tune a reward weight using the RSL-RL iteration scale."""
    del env_ids
    steps_per_iteration = max(1, int(steps_per_iteration))
    current_iteration = float(env.common_step_counter) / float(steps_per_iteration)
    if end_iteration <= start_iteration:
        ratio = 1.0 if current_iteration >= start_iteration else 0.0
    else:
        ratio = (current_iteration - float(start_iteration)) / (float(end_iteration) - float(start_iteration))
    ratio = max(0.0, min(1.0, ratio))
    weight = float(start_weight) + ratio * (float(end_weight) - float(start_weight))

    term_cfg = env.reward_manager.get_term_cfg(term_name)
    term_cfg.weight = weight
    env.reward_manager.set_term_cfg(term_name, term_cfg)
    return torch.tensor(weight, device=env.device)


def command_levels_lin_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_lin_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device)
        env._original_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device)
        env._initial_vel_x = env._original_vel_x * range_multiplier[0]
        env._final_vel_x = env._original_vel_x * range_multiplier[1]
        env._initial_vel_y = env._original_vel_y * range_multiplier[0]
        env._final_vel_y = env._original_vel_y * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.lin_vel_x = env._initial_vel_x.tolist()
        base_velocity_ranges.lin_vel_y = env._initial_vel_y.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_vel_x = torch.tensor(base_velocity_ranges.lin_vel_x, device=env.device) + delta_command
            new_vel_y = torch.tensor(base_velocity_ranges.lin_vel_y, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_vel_x = torch.clamp(new_vel_x, min=env._final_vel_x[0], max=env._final_vel_x[1])
            new_vel_y = torch.clamp(new_vel_y, min=env._final_vel_y[0], max=env._final_vel_y[1])

            # Update ranges
            base_velocity_ranges.lin_vel_x = new_vel_x.tolist()
            base_velocity_ranges.lin_vel_y = new_vel_y.tolist()

    return torch.tensor(base_velocity_ranges.lin_vel_x[1], device=env.device)


def command_levels_ang_vel(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    reward_term_name: str,
    range_multiplier: Sequence[float] = (0.1, 1.0),
) -> None:
    """command_levels_ang_vel"""
    base_velocity_ranges = env.command_manager.get_term("base_velocity").cfg.ranges
    # Get original angular velocity ranges (ONLY ON FIRST EPISODE)
    if env.common_step_counter == 0:
        env._original_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device)
        env._initial_ang_vel_z = env._original_ang_vel_z * range_multiplier[0]
        env._final_ang_vel_z = env._original_ang_vel_z * range_multiplier[1]

        # Initialize command ranges to initial values
        base_velocity_ranges.ang_vel_z = env._initial_ang_vel_z.tolist()

    # avoid updating command curriculum at each step since the maximum command is common to all envs
    if env.common_step_counter % env.max_episode_length == 0:
        episode_sums = env.reward_manager._episode_sums[reward_term_name]
        reward_term_cfg = env.reward_manager.get_term_cfg(reward_term_name)
        delta_command = torch.tensor([-0.1, 0.1], device=env.device)

        # If the tracking reward is above 80% of the maximum, increase the range of commands
        if torch.mean(episode_sums[env_ids]) / env.max_episode_length_s > 0.8 * reward_term_cfg.weight:
            new_ang_vel_z = torch.tensor(base_velocity_ranges.ang_vel_z, device=env.device) + delta_command

            # Clamp to ensure we don't exceed final ranges
            new_ang_vel_z = torch.clamp(new_ang_vel_z, min=env._final_ang_vel_z[0], max=env._final_ang_vel_z[1])

            # Update ranges
            base_velocity_ranges.ang_vel_z = new_ang_vel_z.tolist()

    return torch.tensor(base_velocity_ranges.ang_vel_z[1], device=env.device)


def go2_d1_pick_approach_curriculum(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    distance_steps: Sequence[int] = (80_000, 300_000),
    distance_iteration_bins: Sequence[float] | None = None,
    max_iterations: int | None = None,
    steps_per_iteration: int | None = None,
    resume_iteration: int = 0,
    object_distance_start: float = 0.68,
    object_distance_end: float = 1.20,
    standoff_distance: float = 0.68,
    stop_radius: float = 0.08,
    max_lin_speed: float = 0.55,
    max_lat_speed: float = 0.35,
    max_yaw_rate: float = 0.8,
    lin_gain: float = 1.3,
    lat_gain: float = 1.4,
    yaw_gain: float = 1.8,
    command_name: str = "base_velocity",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    performance_based: bool = False,
    performance_reward_terms: Sequence[str] = ("base_to_object_standoff", "base_faces_object", "ee_to_object"),
    performance_threshold: float = 0.72,
    performance_lower_threshold: float = 0.45,
    performance_progress_step: float = 0.035,
    performance_regress_step: float = 0.01,
    min_steps_before_performance_update: int = 0,
    performance_success_count_attr: str | None = None,
    performance_episode_count_attr: str = "_loco_manip_pick_episode_count",
    inject_velocity_commands: bool = True,
) -> torch.Tensor:
    """Move the can farther over training and command the base toward a manipulation standoff pose."""

    step = int(env.common_step_counter) + int(resume_iteration) * int(steps_per_iteration or 0)
    if performance_based:
        progress_attr = "_go2_d1_pick_approach_progress"
        score_attr = "_go2_d1_pick_approach_score"
        progress = float(getattr(env, progress_attr, 0.0))
        score = float(getattr(env, score_attr, 0.0))
        if _episode_boundary(env) and step >= int(min_steps_before_performance_update):
            if performance_success_count_attr is None:
                score = _positive_reward_score(env, env_ids, performance_reward_terms)
            else:
                successes = getattr(env, performance_success_count_attr, 0)
                episodes = getattr(env, performance_episode_count_attr, 0)
                if isinstance(successes, torch.Tensor):
                    successes = int(successes.item())
                if isinstance(episodes, torch.Tensor):
                    episodes = int(episodes.item())
                previous_successes = int(getattr(env, "_go2_d1_pick_last_success_count", 0))
                previous_episodes = int(getattr(env, "_go2_d1_pick_last_episode_count", 0))
                completed = max(int(episodes) - previous_episodes, 0)
                if completed > 0:
                    score = max(int(successes) - previous_successes, 0) / float(completed)
                env._go2_d1_pick_last_success_count = int(successes)
                env._go2_d1_pick_last_episode_count = int(episodes)
            if score >= float(performance_threshold):
                progress += float(performance_progress_step)
            elif score < float(performance_lower_threshold):
                progress -= float(performance_regress_step)
            progress = float(max(0.0, min(1.0, progress)))
            setattr(env, progress_attr, progress)
            setattr(env, score_attr, score)
    else:
        if len(distance_steps) != 2:
            raise ValueError("distance_steps must contain exactly two step boundaries.")
        if distance_iteration_bins is not None and len(distance_iteration_bins) != 2:
            raise ValueError("distance_iteration_bins must contain exactly two iteration fractions.")

        if distance_iteration_bins is None or max_iterations is None or steps_per_iteration is None:
            start_step, end_step = int(distance_steps[0]), int(distance_steps[1])
        else:
            rollout_steps = max(1, int(steps_per_iteration))
            max_iters = max(1, int(max_iterations))
            start_step = int(round(float(distance_iteration_bins[0]) * max_iters * rollout_steps))
            end_step = int(round(float(distance_iteration_bins[1]) * max_iters * rollout_steps))

        if end_step <= start_step:
            progress = 1.0 if step >= start_step else 0.0
        else:
            progress = (step - start_step) / float(end_step - start_step)
            progress = float(max(0.0, min(1.0, progress)))

    object_distance = float(object_distance_start) + progress * (
        float(object_distance_end) - float(object_distance_start)
    )
    env._go2_d1_pick_object_distance = object_distance
    env._go2_d1_pick_approach_progress = progress

    if not inject_velocity_commands:
        return torch.tensor(object_distance, device=env.device)

    robot: Articulation = env.scene[robot_cfg.name]
    obj = env.scene[object_cfg.name]
    command_term = env.command_manager.get_term(command_name)
    ranges = command_term.cfg.ranges
    ranges.lin_vel_x = (-float(max_lin_speed), float(max_lin_speed))
    ranges.lin_vel_y = (-float(max_lat_speed), float(max_lat_speed))
    ranges.ang_vel_z = (-float(max_yaw_rate), float(max_yaw_rate))
    ranges.heading = (0.0, 0.0)
    command_term.cfg.heading_command = False

    standoff_w = obj.data.root_pos_w.clone()
    standoff_w[:, 0] -= float(standoff_distance)
    standoff_w[:, 2] = robot.data.root_pos_w[:, 2]
    standoff_error_b = quat_apply_inverse(robot.data.root_quat_w, standoff_w - robot.data.root_pos_w)
    object_pos_b = quat_apply_inverse(robot.data.root_quat_w, obj.data.root_pos_w - robot.data.root_pos_w)
    yaw_error = torch.atan2(object_pos_b[:, 1], object_pos_b[:, 0])

    planar_error = torch.linalg.norm(standoff_error_b[:, :2], dim=1)
    moving = planar_error > float(stop_radius)
    command = torch.zeros_like(command_term.vel_command_b)
    command[:, 0] = torch.clamp(float(lin_gain) * standoff_error_b[:, 0], -float(max_lin_speed), float(max_lin_speed))
    command[:, 1] = torch.clamp(float(lat_gain) * standoff_error_b[:, 1], -float(max_lat_speed), float(max_lat_speed))
    command[:, 2] = torch.clamp(float(yaw_gain) * yaw_error, -float(max_yaw_rate), float(max_yaw_rate))
    command *= moving.unsqueeze(1)
    command_term.vel_command_b[:, :] = command
    if hasattr(command_term, "heading_target"):
        command_term.heading_target[:] = 0.0

    env._go2_d1_pick_at_standoff = ~moving
    return torch.tensor(object_distance, device=env.device)


def _sample_loco_manipulation_replay(
    env_ids: torch.Tensor,
    stage: int,
    frontier_difficulty: float,
    replay_fraction: float,
    hold_fraction: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample per-environment arm difficulty with explicit easy-task replay."""
    if replay_fraction < 0.0 or hold_fraction < 0.0 or replay_fraction + hold_fraction > 1.0:
        raise ValueError("replay_fraction and hold_fraction must be non-negative and sum to at most 1.")

    count = int(env_ids.numel())
    device = env_ids.device
    difficulty = torch.zeros(count, device=device, dtype=torch.float32)
    enabled = torch.zeros(count, device=device, dtype=torch.bool)
    if stage < 1 or count == 0:
        return enabled, difficulty

    frontier = float(max(0.0, min(1.0, frontier_difficulty)))
    difficulty.fill_(frontier)
    enabled.fill_(True)
    draw = torch.rand(count, device=device)
    hold_mask = draw < float(hold_fraction)
    replay_mask = (draw >= float(hold_fraction)) & (
        draw < float(hold_fraction) + float(replay_fraction)
    )
    if torch.any(replay_mask):
        difficulty[replay_mask] = torch.rand(int(replay_mask.sum().item()), device=device) * frontier
    difficulty[hold_mask] = 0.0
    enabled[hold_mask] = False
    return enabled, difficulty


def loco_manipulation_training_stages(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    stage_steps: Sequence[int] = (100_000, 200_000),
    arm_difficulty_steps: Sequence[int] | None = None,
    stage_iteration_bins: Sequence[float] | None = None,
    arm_difficulty_iteration_bins: Sequence[float] | None = None,
    max_iterations: int | None = None,
    steps_per_iteration: int | None = None,
    resume_iteration: int = 0,
    walking_lin_vel_x: Sequence[float] = (-1.0, 1.0),
    walking_lin_vel_y: Sequence[float] = (-0.5, 0.5),
    walking_ang_vel_z: Sequence[float] = (-1.0, 1.0),
    walking_heading: Sequence[float] = (-3.14159, 3.14159),
    walking_roll: Sequence[float] = (-0.16, 0.16),
    walking_pitch: Sequence[float] = (-0.30, 0.18),
    walking_height: Sequence[float] = (0.33, 0.33),
    nominal_height: float = 0.33,
    command_name: str = "base_velocity",
    performance_based: bool = False,
    stage0_reward_terms: Sequence[str] = ("track_lin_vel_xy_exp", "track_ang_vel_z_exp"),
    stage1_reward_terms: Sequence[str] = ("arm_joint_target_tracking",),
    stage2_reward_terms: Sequence[str] | None = None,
    stage0_threshold: float = 0.72,
    stage1_threshold: float = 0.72,
    stage2_threshold: float | None = None,
    stage_lower_threshold: float = 0.45,
    stage2_lower_threshold: float | None = None,
    arm_difficulty_reward_terms: Sequence[str] = ("arm_joint_target_tracking",),
    arm_difficulty_threshold: float = 0.70,
    arm_difficulty_lower_threshold: float = 0.45,
    arm_difficulty_step: float = 0.04,
    min_arm_difficulty_for_stage2: float = 0.50,
    min_steps_before_performance_update: int = 0,
    alternate_arm_difficulty_stages: bool = False,
    enable_combined_stage: bool = True,
    arm_difficulty_replay_fraction: float = 0.0,
    held_arm_replay_fraction: float = 0.0,
) -> torch.Tensor:
    """Three-stage loco-manipulation curriculum for mounted-arm tasks.

    Stage 0: walking commands with the arm held at its ready pose.
    Stage 1: zero base-velocity command with the arm moving.
    Stage 2: walking commands with the arm moving.

    Set ``enable_combined_stage`` false for two-stage training that stops at
    stationary-base arm motion. ``arm_difficulty_replay_fraction`` samples
    easier arm/posture difficulty for a random subset of resetting environments;
    ``held_arm_replay_fraction`` retains exact carry-pose episodes.
    """
    if len(stage_steps) != 2:
        raise ValueError("stage_steps must contain exactly two step boundaries.")
    if arm_difficulty_steps is not None and len(arm_difficulty_steps) != 2:
        raise ValueError("arm_difficulty_steps must contain exactly two step boundaries.")
    if stage_iteration_bins is not None and len(stage_iteration_bins) != 2:
        raise ValueError("stage_iteration_bins must contain exactly two iteration fractions.")
    if arm_difficulty_iteration_bins is not None and len(arm_difficulty_iteration_bins) != 2:
        raise ValueError("arm_difficulty_iteration_bins must contain exactly two iteration fractions.")

    def _iteration_bins_to_steps(
        iteration_bins: Sequence[float] | None, fallback_steps: Sequence[int]
    ) -> tuple[int, int]:
        if iteration_bins is None or max_iterations is None or steps_per_iteration is None:
            return int(fallback_steps[0]), int(fallback_steps[1])
        max_iters = max(1, int(max_iterations))
        rollout_steps = max(1, int(steps_per_iteration))
        first = int(round(float(iteration_bins[0]) * max_iters * rollout_steps))
        second = int(round(float(iteration_bins[1]) * max_iters * rollout_steps))
        return first, second

    step = int(env.common_step_counter) + int(resume_iteration) * int(steps_per_iteration or 0)
    if performance_based:
        stage = int(getattr(env, "_loco_manip_frontier_stage", 0))
        arm_difficulty = float(getattr(env, "_loco_manip_arm_motion_frontier", 0.0))
        if step >= int(min_steps_before_performance_update) and _coarse_update_due(
            env,
            step,
            "_loco_manip_training_stage_update_step",
        ):
            stage2_terms = stage1_reward_terms if stage2_reward_terms is None else stage2_reward_terms
            stage2_threshold_value = float(stage1_threshold if stage2_threshold is None else stage2_threshold)
            stage2_lower_value = float(
                arm_difficulty_lower_threshold if stage2_lower_threshold is None else stage2_lower_threshold
            )
            score_env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).flatten()
            sampled_difficulty = getattr(env, "_loco_manip_arm_motion_difficulty", None)
            sampled_enabled = getattr(env, "_loco_manip_arm_motion_enabled", None)
            if isinstance(sampled_difficulty, torch.Tensor) and sampled_difficulty.numel() == env.num_envs:
                selected_difficulty = sampled_difficulty.to(env.device).index_select(0, score_env_ids)
                frontier_mask = selected_difficulty >= float(arm_difficulty) - 1.0e-6
                if isinstance(sampled_enabled, torch.Tensor) and sampled_enabled.numel() == env.num_envs:
                    frontier_mask &= sampled_enabled.to(env.device).index_select(0, score_env_ids).bool()
                if torch.any(frontier_mask):
                    score_env_ids = score_env_ids[frontier_mask]
            base_score = _positive_reward_score(env, env_ids, stage0_reward_terms)
            arm_score = _positive_reward_score(env, score_env_ids, stage1_reward_terms)
            combined_arm_score = _positive_reward_score(env, score_env_ids, stage2_terms)
            difficulty_score = _positive_reward_score(env, score_env_ids, arm_difficulty_reward_terms)
            env._loco_manip_stage0_score = base_score
            env._loco_manip_stage1_score = arm_score
            env._loco_manip_stage2_score = combined_arm_score
            env._loco_manip_arm_difficulty_score = difficulty_score

            if alternate_arm_difficulty_stages:
                if stage == 0:
                    arm_difficulty = 0.0
                    if base_score >= float(stage0_threshold):
                        stage = 1
                elif stage == 1:
                    if base_score < float(stage_lower_threshold):
                        stage = 0
                        arm_difficulty = 0.0
                    elif not enable_combined_stage:
                        if difficulty_score >= float(arm_difficulty_threshold):
                            arm_difficulty += float(arm_difficulty_step)
                        elif difficulty_score < float(arm_difficulty_lower_threshold):
                            arm_difficulty -= float(arm_difficulty_step)
                        stage = 1
                    elif arm_score >= float(stage1_threshold):
                        stage = 2
                    elif difficulty_score < float(arm_difficulty_lower_threshold):
                        arm_difficulty -= float(arm_difficulty_step)
                else:
                    if base_score < float(stage_lower_threshold) or combined_arm_score < stage2_lower_value:
                        stage = 1
                        if difficulty_score < float(arm_difficulty_lower_threshold):
                            arm_difficulty -= float(arm_difficulty_step)
                    elif base_score >= float(stage0_threshold) and combined_arm_score >= stage2_threshold_value:
                        if arm_difficulty < 1.0:
                            arm_difficulty += float(arm_difficulty_step)
                            stage = 1
                        else:
                            stage = 2
                arm_difficulty = float(max(0.0, min(1.0, arm_difficulty)))
            else:
                if stage == 0 and base_score >= float(stage0_threshold):
                    stage = 1
                elif stage > 0 and base_score < float(stage_lower_threshold):
                    stage = max(0, stage - 1)

                if stage >= 1:
                    if difficulty_score >= float(arm_difficulty_threshold):
                        arm_difficulty += float(arm_difficulty_step)
                    elif difficulty_score < float(arm_difficulty_lower_threshold):
                        arm_difficulty -= float(arm_difficulty_step)
                    arm_difficulty = float(max(0.0, min(1.0, arm_difficulty)))
                    if (
                        stage == 1
                        and enable_combined_stage
                        and arm_difficulty >= float(min_arm_difficulty_for_stage2)
                        and arm_score >= float(stage1_threshold)
                    ):
                        stage = 2
                else:
                    arm_difficulty = 0.0
        max_stage = 2 if enable_combined_stage else 1
        stage = int(max(0, min(max_stage, stage)))
        arm_difficulty = float(max(0.0, min(1.0, arm_difficulty)))
    else:
        stage_1_step, stage_2_step = _iteration_bins_to_steps(stage_iteration_bins, stage_steps)
        if enable_combined_stage:
            stage = 0 if step < stage_1_step else 1 if step < stage_2_step else 2
        else:
            stage = 0 if step < stage_1_step else 1

    command_term = env.command_manager.get_term(command_name)
    ranges = command_term.cfg.ranges
    if not hasattr(env, "_loco_manip_original_heading_command"):
        env._loco_manip_original_heading_command = bool(getattr(command_term.cfg, "heading_command", False))

    if stage == 1:
        ranges.lin_vel_x = (0.0, 0.0)
        ranges.lin_vel_y = (0.0, 0.0)
        ranges.ang_vel_z = (0.0, 0.0)
        ranges.heading = (0.0, 0.0)
        command_term.cfg.heading_command = False
        command_term.vel_command_b[:, :] = 0.0
        if hasattr(command_term, "heading_target"):
            command_term.heading_target[:] = 0.0
    else:
        ranges.lin_vel_x = tuple(float(v) for v in walking_lin_vel_x)
        ranges.lin_vel_y = tuple(float(v) for v in walking_lin_vel_y)
        ranges.ang_vel_z = tuple(float(v) for v in walking_ang_vel_z)
        ranges.heading = tuple(float(v) for v in walking_heading)
        command_term.cfg.heading_command = env._loco_manip_original_heading_command

    env._loco_manip_frontier_stage = stage
    env._loco_manip_training_stage = stage

    if not performance_based:
        difficulty_fallback_steps = (
            (stage_1_step, stage_2_step) if arm_difficulty_steps is None else arm_difficulty_steps
        )
        difficulty_start, difficulty_end = _iteration_bins_to_steps(
            arm_difficulty_iteration_bins, difficulty_fallback_steps
        )
        if stage < 1:
            arm_difficulty = 0.0
        elif difficulty_end <= difficulty_start:
            arm_difficulty = 1.0
        else:
            arm_difficulty = (step - difficulty_start) / float(difficulty_end - difficulty_start)
            arm_difficulty = float(max(0.0, min(1.0, arm_difficulty)))
    env._loco_manip_arm_motion_frontier = float(arm_difficulty)

    if hasattr(command_term.cfg, "roll_range") and hasattr(command_term.cfg, "pitch_range"):
        if stage == 0:
            command_term.cfg.roll_range = (0.0, 0.0)
            command_term.cfg.pitch_range = (0.0, 0.0)
            command_term.posture_command[:, :2] = 0.0
            if command_term.posture_command.shape[1] > 2:
                command_term.posture_command[:, 2] = float(nominal_height)
        else:
            roll = tuple(float(v) * arm_difficulty for v in walking_roll)
            pitch = tuple(float(v) * arm_difficulty for v in walking_pitch)
            command_term.cfg.roll_range = roll
            command_term.cfg.pitch_range = pitch
    if hasattr(command_term.cfg, "height_range") and hasattr(command_term, "posture_command"):
        if stage == 0:
            command_term.cfg.height_range = (float(nominal_height), float(nominal_height))
        else:
            low = float(nominal_height) + (float(walking_height[0]) - float(nominal_height)) * arm_difficulty
            high = float(nominal_height) + (float(walking_height[1]) - float(nominal_height)) * arm_difficulty
            command_term.cfg.height_range = (low, high)

    env_ids_t = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).flatten()
    num_envs = int(env.num_envs)
    current_enabled = getattr(env, "_loco_manip_arm_motion_enabled", None)
    if not isinstance(current_enabled, torch.Tensor) or current_enabled.numel() != num_envs:
        current_enabled = torch.zeros(num_envs, device=env.device, dtype=torch.bool)
    else:
        current_enabled = current_enabled.to(device=env.device, dtype=torch.bool).reshape(-1).clone()
    current_difficulty = getattr(env, "_loco_manip_arm_motion_difficulty", None)
    if not isinstance(current_difficulty, torch.Tensor) or current_difficulty.numel() != num_envs:
        current_difficulty = torch.zeros(num_envs, device=env.device, dtype=torch.float32)
    else:
        current_difficulty = current_difficulty.to(device=env.device, dtype=torch.float32).reshape(-1).clone()

    sampled_enabled, sampled_difficulty = _sample_loco_manipulation_replay(
        env_ids_t,
        stage,
        arm_difficulty,
        arm_difficulty_replay_fraction,
        held_arm_replay_fraction,
    )
    current_enabled[env_ids_t] = sampled_enabled
    current_difficulty[env_ids_t] = sampled_difficulty
    env._loco_manip_arm_motion_enabled = current_enabled
    env._loco_manip_arm_motion_difficulty = current_difficulty
    return torch.tensor(float(stage), device=env.device)


def _terrain_name_to_columns(terrain: TerrainImporter) -> dict[str, list[int]]:
    """Resolve terrain names to curriculum column indices for generator terrains."""
    terrain_gen_cfg = terrain.cfg.terrain_generator
    if terrain_gen_cfg is None:
        return {}

    proportions = np.array([sub_cfg.proportion for sub_cfg in terrain_gen_cfg.sub_terrains.values()], dtype=np.float64)
    if proportions.sum() <= 0:
        proportions = np.ones_like(proportions) / max(len(proportions), 1)
    else:
        proportions /= proportions.sum()
    names = list(terrain_gen_cfg.sub_terrains.keys())
    cum = np.cumsum(proportions)

    name_to_cols: dict[str, list[int]] = {name: [] for name in names}
    for col in range(int(terrain_gen_cfg.num_cols)):
        threshold = col / float(terrain_gen_cfg.num_cols) + 1.0e-3
        sub_index = int(np.min(np.where(threshold < cum)[0]))
        name_to_cols[names[sub_index]].append(col)
    return name_to_cols


def start_terrain_levels_progression(
    env: ManagerBasedRLEnv,
    env_ids: Sequence[int],
    easy_terrain_names: Sequence[str],
    advanced_terrain_names: Sequence[str],
    gap_terrain_names: Sequence[str] = ("gaps",),
    p_max: float = 1.0,
    t_start: float = 2.0e4,
    t_end: float = 1.8e5,
    gap_probability: float | None = None,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """START terrain progression with linear `p_advance` schedule plus terrain-level curriculum."""
    # extract the used quantities
    asset: Articulation = env.scene[asset_cfg.name]
    terrain: TerrainImporter = env.scene.terrain
    command = env.command_manager.get_command("base_velocity")

    if isinstance(env_ids, torch.Tensor):
        env_ids_t = env_ids.to(device=env.device, dtype=torch.long)
    else:
        env_ids_t = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if env_ids_t.numel() == 0:
        return torch.mean(terrain.terrain_levels.float())

    # per-episode terrain-level update (same logic as terrain_levels_vel)
    distance = torch.norm(asset.data.root_pos_w[env_ids_t, :2] - env.scene.env_origins[env_ids_t, :2], dim=1)
    move_up = distance > terrain.cfg.terrain_generator.size[0] / 2
    move_down = distance < torch.norm(command[env_ids_t, :2], dim=1) * env.max_episode_length_s * 0.5
    move_down *= ~move_up

    # lazily cache terrain column mapping
    cache_key = "_start_terrain_name_to_cols_cache"
    if not hasattr(env, cache_key):
        setattr(env, cache_key, _terrain_name_to_columns(terrain))
    name_to_cols: dict[str, list[int]] = getattr(env, cache_key)

    def _concat_columns(names: Sequence[str]) -> list[int]:
        cols: list[int] = []
        for name in names:
            cols.extend(name_to_cols.get(name, []))
        return cols

    easy_cols = _concat_columns(easy_terrain_names)
    advanced_cols = _concat_columns(advanced_terrain_names)
    gap_cols = _concat_columns(gap_terrain_names)

    if gap_probability is None:
        total_cols = max(1, int(terrain.cfg.terrain_generator.num_cols))
        gap_probability = float(len(gap_cols) / total_cols)
    gap_probability = float(max(0.0, min(1.0, gap_probability)))

    # linear progression schedule from paper: p_advance = clamp(p_max * (T - T_start)/(T_end - T_start), [0, p_max])
    step_counter = float(env.common_step_counter)
    if t_end <= t_start:
        p_advance = p_max if step_counter >= t_start else 0.0
    else:
        p_advance = p_max * (step_counter - t_start) / (t_end - t_start)
    p_advance = float(max(0.0, min(float(p_max), p_advance)))

    def _sample_cols(col_list: list[int], count: int) -> torch.Tensor:
        if count <= 0:
            return torch.empty((0,), device=env.device, dtype=torch.long)
        if len(col_list) == 0:
            return torch.zeros((count,), device=env.device, dtype=torch.long)
        col_tensor = torch.tensor(col_list, device=env.device, dtype=torch.long)
        sample_idx = torch.randint(0, len(col_list), (count,), device=env.device)
        return col_tensor[sample_idx]

    sampled_types = terrain.terrain_types[env_ids_t].clone()

    # keep a constant share of gap terrains (excluded from the progression schedule).
    gap_draw = torch.rand(env_ids_t.shape[0], device=env.device)
    gap_mask = gap_draw < gap_probability
    if torch.any(gap_mask):
        sampled_types[gap_mask] = _sample_cols(gap_cols, int(gap_mask.sum().item()))

    non_gap_mask = ~gap_mask
    if torch.any(non_gap_mask):
        non_gap_count = int(non_gap_mask.sum().item())
        advance_mask_local = torch.rand(non_gap_count, device=env.device) < p_advance
        non_gap_indices = torch.nonzero(non_gap_mask, as_tuple=False).squeeze(-1)

        if torch.any(advance_mask_local):
            adv_indices = non_gap_indices[advance_mask_local]
            sampled_types[adv_indices] = _sample_cols(advanced_cols, int(advance_mask_local.sum().item()))
        if torch.any(~advance_mask_local):
            easy_indices = non_gap_indices[~advance_mask_local]
            sampled_types[easy_indices] = _sample_cols(easy_cols, int((~advance_mask_local).sum().item()))

    terrain.terrain_types[env_ids_t] = sampled_types
    terrain.update_env_origins(env_ids_t, move_up, move_down)
    return torch.mean(terrain.terrain_levels.float())
