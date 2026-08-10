from __future__ import annotations

from typing import TYPE_CHECKING

import torch

import isaaclab.utils.math as math_utils

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, SceneEntityCfg, TerminationTermCfg

from .commands import MotionCommand
from .rewards import _filtered_contact_hit, _get_body_indexes


class object_grasp_contact_timeout(ManagerTermBase):
    """Terminate sustained missing bilateral finger contact during the demonstrated attached phase."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._ungripped_time_s = torch.zeros(env.num_envs, dtype=torch.float32, device=env.device)

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._ungripped_time_s[env_ids] = 0.0

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        command_name: str,
        left_sensor_cfg: SceneEntityCfg,
        right_sensor_cfg: SceneEntityCfg,
        force_threshold: float = 0.35,
        object_index: int = 0,
        warmup_iterations: int = 250,
        ramp_end_iteration: int = 1000,
        timeout_start_s: float = 2.0,
        timeout_end_s: float = 0.5,
        steps_per_iteration: int = 24,
        resume_iteration: int = 0,
    ) -> torch.Tensor:
        """Apply a resume-aware timeout curriculum without treating contact flicker as grasp loss."""
        if warmup_iterations < 0 or ramp_end_iteration < warmup_iterations:
            raise ValueError(
                "Grasp-contact curriculum requires 0 <= warmup_iterations <= ramp_end_iteration."
            )
        if timeout_start_s <= 0.0 or timeout_end_s <= 0.0:
            raise ValueError("Grasp-contact timeout values must be positive.")
        if steps_per_iteration <= 0 or resume_iteration < 0:
            raise ValueError("steps_per_iteration must be positive and resume_iteration non-negative.")

        command: MotionCommand = env.command_manager.get_term(command_name)
        attached = command.object_attached
        if attached is None:
            self._ungripped_time_s.zero_()
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        current_iteration = float(resume_iteration) + (
            float(max(int(getattr(env, "common_step_counter", 0)), 0)) / float(steps_per_iteration)
        )
        if current_iteration < float(warmup_iterations):
            self._ungripped_time_s.zero_()
            return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

        if ramp_end_iteration == warmup_iterations:
            curriculum_progress = 1.0
        else:
            curriculum_progress = min(
                max(
                    (current_iteration - float(warmup_iterations))
                    / float(ramp_end_iteration - warmup_iterations),
                    0.0,
                ),
                1.0,
            )
        timeout_s = float(timeout_start_s) + curriculum_progress * (
            float(timeout_end_s) - float(timeout_start_s)
        )

        bilateral_contact = _filtered_contact_hit(
            env, left_sensor_cfg, force_threshold
        ) & _filtered_contact_hit(env, right_sensor_cfg, force_threshold)
        should_be_gripped = attached[:, object_index].bool()
        missing_grasp = should_be_gripped & (~bilateral_contact)
        self._ungripped_time_s = torch.where(
            missing_grasp,
            self._ungripped_time_s + float(env.step_dt),
            torch.zeros_like(self._ungripped_time_s),
        )

        episode_length_buf = getattr(env, "episode_length_buf", None)
        if episode_length_buf is not None:
            reset_like = episode_length_buf <= 1
            self._ungripped_time_s[reset_like] = 0.0

        return missing_grasp & (self._ungripped_time_s >= timeout_s)


def bad_anchor_pos_z_only(env: ManagerBasedRLEnv, command_name: str, threshold: float) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)
    return torch.abs(command.anchor_pos_w[:, -1] - command.robot_anchor_pos_w[:, -1]) > threshold


def bad_anchor_ori(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, command_name: str, threshold: float
) -> torch.Tensor:
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    command: MotionCommand = env.command_manager.get_term(command_name)
    motion_projected_gravity_b = math_utils.quat_apply_inverse(command.anchor_quat_w, asset.data.GRAVITY_VEC_W)

    robot_projected_gravity_b = math_utils.quat_apply_inverse(command.robot_anchor_quat_w, asset.data.GRAVITY_VEC_W)

    return (motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]).abs() > threshold


def bad_motion_body_pos_z_only(
    env: ManagerBasedRLEnv, command_name: str, threshold: float, body_names: list[str] | None = None
) -> torch.Tensor:
    command: MotionCommand = env.command_manager.get_term(command_name)

    body_indexes = _get_body_indexes(command, body_names)
    error = torch.abs(command.body_pos_relative_w[:, body_indexes, -1] - command.robot_body_pos_w[:, body_indexes, -1])
    return torch.any(error > threshold, dim=-1)


def motion_clip_end(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """End an episode at the selected clip boundary without wrapping its reference."""
    command: MotionCommand = env.command_manager.get_term(command_name)
    return command.time_steps >= command.time_step_ends
