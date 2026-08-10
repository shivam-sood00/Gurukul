"""Manager-based AMP adapter for BeyondMimic motion archives.

The discriminator observes task-agnostic PM01 state transitions, following the
state-transition formulation in Peng et al., "AMP" (arXiv:2104.02180). Object
state remains in the task policy observation and is intentionally excluded here.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

import gymnasium as gym
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv

from Gurukul.tasks.manager_based.beyondmimic.amp_features import (
    amp_history_frame_indices,
    build_amp_observation,
    initialize_amp_history_from_reference,
)
from Gurukul.tasks.manager_based.beyondmimic.mdp.commands import MotionCommand


class BeyondMimicAmpEnv(ManagerBasedRLEnv):
    """Expose manager-based BeyondMimic state transitions to SKRL AMP."""

    def __init__(self, cfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg=cfg, render_mode=render_mode, **kwargs)

        self._amp_motion: MotionCommand = self.command_manager.get_term(cfg.amp_motion_command_name)
        self._amp_history_length = int(cfg.amp_num_observations)
        if self._amp_history_length < 2:
            raise ValueError("AMP requires at least two observations to represent a state transition.")
        motion_fps = float(np.asarray(self._amp_motion.motion.fps).reshape(-1)[0])
        frames_per_policy_step = motion_fps * float(self.step_dt)
        self._amp_reference_frame_stride = int(round(frames_per_policy_step))
        if self._amp_reference_frame_stride < 1 or not math.isclose(
            frames_per_policy_step,
            self._amp_reference_frame_stride,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise ValueError(
                "AMP motion FPS multiplied by environment step_dt must be a positive integer; "
                f"got {motion_fps} * {self.step_dt} = {frames_per_policy_step}."
            )

        body_name_to_index = {name: index for index, name in enumerate(self._amp_motion.cfg.body_names)}
        missing = [name for name in cfg.amp_key_body_names if name not in body_name_to_index]
        if missing:
            raise ValueError(f"AMP key bodies are not present in the motion command: {missing}")
        self._amp_motion_key_body_indices = [body_name_to_index[name] for name in cfg.amp_key_body_names]
        self._amp_robot_key_body_indices = self._amp_motion.robot.find_bodies(
            cfg.amp_key_body_names, preserve_order=True
        )[0]

        one_frame = self._compute_current_amp_observation()
        self._amp_frame_size = one_frame.shape[-1]
        self._amp_observation_size = self._amp_history_length * self._amp_frame_size
        self.amp_observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(self._amp_observation_size,), dtype=np.float32
        )
        self._amp_observation_buffer = one_frame[:, None, :].repeat(1, self._amp_history_length, 1)
        self._amp_reset_reference_steps = self._amp_motion.time_steps.clone()
        self.extras["amp_obs"] = self._amp_observation_buffer.reshape(self.num_envs, -1)

    def _compute_current_amp_observation(self) -> torch.Tensor:
        motion = self._amp_motion
        root_pos = motion.robot_anchor_pos_w - self.scene.env_origins
        return build_amp_observation(
            motion.robot_joint_pos,
            motion.robot_joint_vel,
            root_pos,
            motion.robot_anchor_quat_w,
            motion.robot_anchor_lin_vel_w,
            motion.robot_anchor_ang_vel_w,
            motion.robot.data.body_pos_w[:, self._amp_robot_key_body_indices]
            - self.scene.env_origins[:, None, :],
        )

    def _compute_reference_amp_observations(self, frame_indices: torch.Tensor) -> torch.Tensor:
        motion = self._amp_motion.motion
        frame_shape = frame_indices.shape
        flat_frame_indices = frame_indices.reshape(-1)
        root_index = self._amp_motion.motion_anchor_body_index
        root_pos = motion.body_pos_w[flat_frame_indices, root_index]
        reference = build_amp_observation(
            motion.joint_pos[flat_frame_indices],
            motion.joint_vel[flat_frame_indices],
            root_pos,
            motion.body_quat_w[flat_frame_indices, root_index],
            motion.body_lin_vel_w[flat_frame_indices, root_index],
            motion.body_ang_vel_w[flat_frame_indices, root_index],
            motion.body_pos_w[flat_frame_indices][:, self._amp_motion_key_body_indices],
        )
        return reference.reshape(*frame_shape, self._amp_frame_size)

    def _reset_amp_history(self, env_ids: torch.Tensor, current: torch.Tensor) -> None:
        frame_indices = amp_history_frame_indices(
            self._amp_reset_reference_steps[env_ids],
            self._amp_history_length,
            self._amp_motion.motion.time_step_total,
            self._amp_reference_frame_stride,
        )
        reference_history = self._compute_reference_amp_observations(frame_indices)
        self._amp_observation_buffer[env_ids] = initialize_amp_history_from_reference(
            current[env_ids], reference_history
        )

    def _update_amp_history(self, reset_mask: torch.Tensor | None = None) -> None:
        current = self._compute_current_amp_observation()
        self._amp_observation_buffer[:, 1:] = self._amp_observation_buffer[:, :-1].clone()
        self._amp_observation_buffer[:, 0] = current
        if reset_mask is not None and torch.any(reset_mask):
            self._reset_amp_history(torch.where(reset_mask)[0], current)
        self.extras["amp_obs"] = self._amp_observation_buffer.reshape(self.num_envs, -1)

    def _reset_idx(self, env_ids: Sequence[int]):
        super()._reset_idx(env_ids)
        # ManagerBasedRLEnv.step advances commands after resetting environments.
        # Capture the reference-state initialization frame before that update so
        # AMP history never spans two episodes or repeats the reset state.
        if hasattr(self, "_amp_reset_reference_steps"):
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
            self._amp_reset_reference_steps[env_ids] = self._amp_motion.time_steps[env_ids]

    def step(self, action: torch.Tensor):
        observations, rewards, terminated, truncated, extras = super().step(action)
        self._update_amp_history(terminated | truncated)
        return observations, rewards, terminated, truncated, extras

    def reset(
        self,
        seed: int | None = None,
        env_ids: Sequence[int] | None = None,
        options: dict[str, Any] | None = None,
    ):
        observations, extras = super().reset(seed=seed, env_ids=env_ids, options=options)
        current = self._compute_current_amp_observation()
        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        self._reset_amp_history(env_ids, current)
        self.extras["amp_obs"] = self._amp_observation_buffer.reshape(self.num_envs, -1)
        return observations, extras

    def collect_reference_motions(
        self, num_samples: int, current_times: np.ndarray | torch.Tensor | None = None
    ) -> torch.Tensor:
        """Sample phase-free reference transitions for the AMP motion dataset."""
        motion = self._amp_motion.motion
        total_frames = motion.time_step_total
        minimum_frames = 1 + (self._amp_history_length - 1) * self._amp_reference_frame_stride
        if total_frames < minimum_frames:
            raise ValueError(
                f"Motion contains {total_frames} frames; AMP history requires at least {minimum_frames}."
            )

        if current_times is None:
            latest = torch.randint(
                minimum_frames - 1,
                total_frames,
                (num_samples,),
                device=self.device,
            )
        else:
            fps = float(np.asarray(motion.fps).reshape(-1)[0])
            latest = torch.as_tensor(current_times, device=self.device, dtype=torch.float32)
            latest = torch.floor(latest * fps).to(torch.long)
            latest = torch.clamp(latest, minimum_frames - 1, total_frames - 1)
            if latest.numel() != num_samples:
                raise ValueError(f"Expected {num_samples} current times, received {latest.numel()}.")

        frame_indices = amp_history_frame_indices(
            latest,
            self._amp_history_length,
            total_frames,
            self._amp_reference_frame_stride,
        )
        return self._compute_reference_amp_observations(frame_indices).reshape(num_samples, self._amp_observation_size)
