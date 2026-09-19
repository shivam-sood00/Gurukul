# Copyright (c) 2025- Shenzhen Zhongqing Robot Technology Co., Ltd. ("EngineAI")
# Copyright (c) 2026, Gurukul contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""Validated loader for EngineAI's PM01 AMP locomotion reference."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import torch

PM01_AMP_REFERENCE_SHA256 = "945fa07b32c66410d2904e9f9147f4d433a69f9d688965c366092e71a311f6b4"
PM01_AMP_REFERENCE_JOINT_COUNT = 23
PM01_AMP_FRAME_DIM = PM01_AMP_REFERENCE_JOINT_COUNT + 3


def _quat_apply_inverse(quat_wxyz: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Rotate a world-frame vector by the inverse of a scalar-first unit quaternion."""
    quat_xyz = quat_wxyz[..., 1:]
    first_cross = torch.cross(quat_xyz, vector, dim=-1)
    return vector - 2.0 * quat_wxyz[..., :1] * first_cross + 2.0 * torch.cross(quat_xyz, first_cross, dim=-1)


class Pm01AmpMotionDataset:
    """EngineAI AMP features sampled at the policy's temporal resolution.

    EngineAI's archive stores 23 joint positions in articulation order and a
    100 Hz full-body trajectory. The upstream discriminator uses joint
    positions scaled by 9 and base-frame linear velocity scaled by 7.
    """

    required_fields = {
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    }

    def __init__(
        self,
        path: str | Path,
        *,
        history_length: int,
        policy_dt: float,
        device: str | torch.device = "cpu",
        expected_sha256: str = PM01_AMP_REFERENCE_SHA256,
        min_planar_speed: float = 0.0,
    ) -> None:
        if history_length < 1:
            raise ValueError("AMP history_length must be positive.")
        if policy_dt <= 0.0:
            raise ValueError("AMP policy_dt must be positive.")
        if min_planar_speed < 0.0:
            raise ValueError("AMP min_planar_speed must be non-negative.")

        self.path = Path(path).expanduser().resolve()
        if not self.path.is_file():
            raise FileNotFoundError(f"PM01 AMP reference does not exist: {self.path}")
        digest = hashlib.sha256(self.path.read_bytes()).hexdigest()
        if expected_sha256 and digest != expected_sha256:
            raise ValueError(
                f"Unexpected PM01 AMP reference SHA-256 for {self.path}: expected {expected_sha256}, got {digest}."
            )

        with np.load(self.path, allow_pickle=False) as archive:
            missing = self.required_fields.difference(archive.files)
            if missing:
                raise ValueError(f"PM01 AMP reference is missing fields: {sorted(missing)}")

            fps_values = np.asarray(archive["fps"], dtype=np.float64).reshape(-1)
            if fps_values.size != 1 or not np.isfinite(fps_values[0]) or fps_values[0] <= 0.0:
                raise ValueError(f"PM01 AMP reference has invalid fps: {fps_values!r}")
            self.fps = float(fps_values[0])

            joint_pos = torch.as_tensor(np.asarray(archive["joint_pos"], dtype=np.float32), device=device)
            body_quat_w = torch.as_tensor(np.asarray(archive["body_quat_w"], dtype=np.float32), device=device)
            body_lin_vel_w = torch.as_tensor(np.asarray(archive["body_lin_vel_w"], dtype=np.float32), device=device)

        if joint_pos.ndim != 2 or joint_pos.shape[1] != PM01_AMP_REFERENCE_JOINT_COUNT:
            raise ValueError(f"PM01 AMP joint_pos must have shape (frames, 23); received {tuple(joint_pos.shape)}.")
        if body_quat_w.ndim != 3 or body_quat_w.shape[0] != joint_pos.shape[0] or body_quat_w.shape[2] != 4:
            raise ValueError(
                f"PM01 AMP body_quat_w must have shape (frames, bodies, 4); received {tuple(body_quat_w.shape)}."
            )
        if body_lin_vel_w.shape != (*body_quat_w.shape[:2], 3):
            raise ValueError(
                f"PM01 AMP body_lin_vel_w must align with body_quat_w; received {tuple(body_lin_vel_w.shape)}."
            )
        if not all(torch.isfinite(value).all() for value in (joint_pos, body_quat_w, body_lin_vel_w)):
            raise ValueError("PM01 AMP reference contains non-finite discriminator features.")

        # In EngineAI's articulation order, column 2 is J12_WAIST_YAW. It is
        # neutral in the released clip and is also neutralized in policy data.
        if not torch.allclose(joint_pos[:, 2], torch.zeros_like(joint_pos[:, 2]), atol=1.0e-7, rtol=0.0):
            raise ValueError("PM01 AMP reference column 2 must be the neutral J12_WAIST_YAW channel.")

        frames_per_policy_step = self.fps * float(policy_dt)
        self.frame_stride = int(round(frames_per_policy_step))
        if self.frame_stride < 1 or not np.isclose(frames_per_policy_step, self.frame_stride, rtol=0.0, atol=1.0e-6):
            raise ValueError(
                "PM01 AMP reference FPS must be an integer multiple of policy frequency; "
                f"got {self.fps} Hz with policy_dt={policy_dt}."
            )

        base_lin_vel_b = _quat_apply_inverse(body_quat_w[:, 0], body_lin_vel_w[:, 0])
        self.features = torch.cat((joint_pos * 9.0, base_lin_vel_b * 7.0), dim=-1)
        self.history_length = int(history_length)
        self.minimum_latest_frame = (self.history_length - 1) * self.frame_stride
        if self.features.shape[0] <= self.minimum_latest_frame:
            raise ValueError(
                f"PM01 AMP reference has {self.features.shape[0]} frames, but a "
                f"{self.history_length}-frame history at stride {self.frame_stride} is required."
            )

        # EngineAI's released clip is ~23 s: a short stand, ~13 s of walking at 0.25--0.94 m/s,
        # then ~8.5 s of standing still. Sampling the static tail teaches the discriminator that
        # "hold the nominal pose" is expert behavior, which rewards freezing under a walk command.
        # Keep only windows whose frames are all above ``min_planar_speed``.
        self.min_planar_speed = float(min_planar_speed)
        planar_speed = torch.linalg.vector_norm(base_lin_vel_b[:, :2], dim=-1)
        latest = torch.arange(
            self.minimum_latest_frame, self.features.shape[0], device=self.features.device, dtype=torch.long
        )
        if self.min_planar_speed > 0.0:
            keep = (planar_speed[self.window_indices(latest)] >= self.min_planar_speed).all(dim=1)
            latest = latest[keep]
            if latest.numel() == 0:
                raise ValueError(
                    f"PM01 AMP reference has no {self.history_length}-frame window above "
                    f"{self.min_planar_speed} m/s; lower ``amp_reference_min_speed``."
                )
        self.sampled_latest_frames = latest

    @property
    def frame_dim(self) -> int:
        return int(self.features.shape[1])

    @property
    def observation_dim(self) -> int:
        return self.history_length * self.frame_dim

    def window_indices(self, latest: torch.Tensor) -> torch.Tensor:
        """Return oldest-to-newest reference indices ending at ``latest``."""
        latest = latest.to(device=self.features.device, dtype=torch.long).reshape(-1)
        if torch.any(latest < self.minimum_latest_frame) or torch.any(latest >= self.features.shape[0]):
            raise ValueError("PM01 AMP latest-frame indices are outside the valid history range.")
        offsets = torch.arange(
            self.history_length - 1,
            -1,
            -1,
            device=self.features.device,
            dtype=torch.long,
        )
        return latest[:, None] - offsets[None, :] * self.frame_stride

    @property
    def num_sampled_windows(self) -> int:
        """Number of reference windows the sampler draws from after speed filtering."""
        return int(self.sampled_latest_frames.numel())

    def sample(self, batch_size: int) -> torch.Tensor:
        """Sample flattened oldest-to-newest AMP histories."""
        if batch_size < 1:
            raise ValueError("PM01 AMP batch_size must be positive.")
        picks = torch.randint(0, self.num_sampled_windows, (batch_size,), device=self.features.device)
        latest = self.sampled_latest_frames[picks]
        windows = self.features[self.window_indices(latest)]
        return windows.reshape(batch_size, self.observation_dim)
