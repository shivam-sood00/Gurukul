# SPDX-License-Identifier: Apache-2.0

"""Portable motion loading and window preparation for SMP priors.

The functions here intentionally depend only on NumPy and PyTorch.  Source
clips are name-mapped to a morphology profile, resampled to the policy rate,
and split *within each clip* before canonicalization.  Consequently, no
training window can cross a source-clip boundary.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import Dataset

from .features import canonicalize_motion_features
from .profiles import (
    DEFAULT_WINDOW_SIZE,
    SMP_SCHEMA_VERSION,
    SmpRobotProfile,
    get_profile,
    profile_from_metadata,
    validate_profile_metadata,
)

DATASET_FORMAT = "gurukul.smp.windows"
DATASET_FORMAT_VERSION = 3
DEFAULT_MINIMUM_FEATURE_RANGE = 0.4
"""Minimum q01/q99 span, equivalent to a 0.2 normalization-scale floor."""

_FIELD_ALIASES = {
    "joint_names": ("joint_names", "dof_names"),
    "joint_pos": ("joint_pos", "dof_positions", "dof_pos"),
    "body_names": ("body_names",),
    "body_pos": ("body_pos_w", "body_positions", "body_pos"),
    "body_quat": ("body_quat_w", "body_rotations", "body_quat"),
    "fps": ("fps",),
}


@dataclass(frozen=True)
class MotionSequence:
    """One chronological motion clip mapped to a canonical robot profile."""

    profile: SmpRobotProfile
    fps: float
    root_pos: np.ndarray
    root_quat_wxyz: np.ndarray
    joint_pos: np.ndarray
    key_body_pos: np.ndarray
    root_lin_vel: np.ndarray
    root_ang_vel: np.ndarray
    source_path: Path

    @property
    def num_frames(self) -> int:
        return int(self.root_pos.shape[0])


class SmpWindowDataset(Dataset[torch.Tensor]):
    """In-memory canonical SMP windows loaded from a safe NPZ archive."""

    def __init__(
        self,
        windows: torch.Tensor,
        profile: SmpRobotProfile,
        *,
        control_fps: float,
        q_low: torch.Tensor,
        q_high: torch.Tensor,
        source_files: Sequence[str] = (),
        normalization_source_files: Sequence[str] = (),
        path: Path | None = None,
    ) -> None:
        if windows.ndim != 3:
            raise ValueError(f"SMP windows must have shape [N, T, F], got {tuple(windows.shape)}.")
        if windows.shape[2] != profile.feature_dim:
            raise ValueError(
                f"SMP windows have feature dimension {windows.shape[2]}, but profile "
                f"{profile.name!r} requires {profile.feature_dim}."
            )
        if q_low.shape != (profile.feature_dim,) or q_high.shape != (profile.feature_dim,):
            raise ValueError("SMP feature bounds must each have shape [feature_dim].")
        if not torch.isfinite(windows).all():
            raise ValueError("SMP windows contain NaN or infinite values.")
        if not torch.isfinite(q_low).all() or not torch.isfinite(q_high).all():
            raise ValueError("SMP feature bounds contain NaN or infinite values.")
        if not torch.all(q_high > q_low):
            raise ValueError("Every SMP upper feature bound must be greater than its lower bound.")

        self.windows = windows.detach().to(device="cpu", dtype=torch.float32).contiguous()
        self.profile = profile
        self.control_fps = float(control_fps)
        self.window_size = int(windows.shape[1])
        self.q_low = q_low.detach().to(device="cpu", dtype=torch.float32).contiguous()
        self.q_high = q_high.detach().to(device="cpu", dtype=torch.float32).contiguous()
        self.source_files = tuple(str(item) for item in source_files)
        self.normalization_source_files = tuple(str(item) for item in normalization_source_files)
        self.path = path

    def __len__(self) -> int:
        return int(self.windows.shape[0])

    def __getitem__(self, index: int) -> torch.Tensor:
        return self.windows[index]


def _select_field(archive: np.lib.npyio.NpzFile, logical_name: str, path: Path) -> np.ndarray:
    aliases = _FIELD_ALIASES[logical_name]
    matches = [name for name in aliases if name in archive.files]
    if not matches:
        raise ValueError(f"Motion archive {path} is missing {logical_name!r}; accepted fields are {aliases}.")
    if len(matches) > 1:
        first = np.asarray(archive[matches[0]])
        for duplicate in matches[1:]:
            if not np.array_equal(first, np.asarray(archive[duplicate])):
                raise ValueError(f"Motion archive {path} contains conflicting aliases for {logical_name!r}: {matches}.")
    return np.asarray(archive[matches[0]])


def _decode_names(values: np.ndarray, field_name: str, path: Path) -> tuple[str, ...]:
    if values.ndim != 1:
        raise ValueError(f"{field_name} in {path} must be a one-dimensional string array.")
    names = tuple(
        item.decode("utf-8") if isinstance(item, (bytes, np.bytes_)) else str(item) for item in values.tolist()
    )
    if any(not name for name in names):
        raise ValueError(f"{field_name} in {path} contains an empty name.")
    if len(set(names)) != len(names):
        raise ValueError(f"{field_name} in {path} contains duplicate names.")
    return names


def _name_indices(source_names: Sequence[str], requested_names: Sequence[str], label: str, path: Path) -> list[int]:
    index_by_name = {name: index for index, name in enumerate(source_names)}
    missing = [name for name in requested_names if name not in index_by_name]
    if missing:
        raise ValueError(
            f"Motion archive {path} is missing {label} required by the SMP profile: {missing}. "
            "Re-export the clip with named robot bodies/joints; positional guessing is unsafe."
        )
    return [index_by_name[name] for name in requested_names]


def _as_float_array(values: np.ndarray, expected_tail: tuple[int, ...], field_name: str, path: Path) -> np.ndarray:
    if values.ndim != len(expected_tail) + 1 or tuple(values.shape[1:]) != expected_tail:
        raise ValueError(
            f"{field_name} in {path} must have shape [frames, {', '.join(map(str, expected_tail))}], "
            f"got {values.shape}."
        )
    result = np.asarray(values, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{field_name} in {path} contains NaN or infinite values.")
    return result


def _interpolate_vectors(values: np.ndarray, source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    flattened = values.reshape(values.shape[0], -1)
    output = np.empty((target_times.size, flattened.shape[1]), dtype=np.float64)
    for channel in range(flattened.shape[1]):
        output[:, channel] = np.interp(target_times, source_times, flattened[:, channel])
    return output.reshape((target_times.size, *values.shape[1:]))


def _interpolate_angles(values: np.ndarray, source_times: np.ndarray, target_times: np.ndarray) -> np.ndarray:
    """Interpolate scalar joint coordinates after removing ±pi wrap jumps."""
    return _interpolate_vectors(np.unwrap(values, axis=0), source_times, target_times)


def _interpolate_quaternions(
    quaternions_wxyz: np.ndarray, source_times: np.ndarray, target_times: np.ndarray
) -> np.ndarray:
    """Shortest-arc SLERP for a uniformly or non-uniformly sampled sequence."""

    norms = np.linalg.norm(quaternions_wxyz, axis=-1, keepdims=True)
    if np.any(norms < 1.0e-8):
        raise ValueError("Motion archive contains a zero-length root quaternion.")
    quaternions = quaternions_wxyz / norms

    right = np.searchsorted(source_times, target_times, side="right")
    right = np.clip(right, 1, len(source_times) - 1)
    left = right - 1
    interval = source_times[right] - source_times[left]
    fraction = np.divide(
        target_times - source_times[left],
        interval,
        out=np.zeros_like(target_times),
        where=interval > 0.0,
    )[:, None]

    q0 = quaternions[left]
    q1 = quaternions[right].copy()
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.clip(np.abs(dot), 0.0, 1.0)

    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    slerp = (
        np.sin((1.0 - fraction) * theta) / np.maximum(sin_theta, 1.0e-8) * q0
        + np.sin(fraction * theta) / np.maximum(sin_theta, 1.0e-8) * q1
    )
    linear = (1.0 - fraction) * q0 + fraction * q1
    output = np.where(dot > 0.9995, linear, slerp)
    return output / np.maximum(np.linalg.norm(output, axis=-1, keepdims=True), 1.0e-8)


def _finite_difference_root_motion(
    root_pos: np.ndarray,
    root_quat_wxyz: np.ndarray,
    fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Derive centered world-frame link velocities from a pose trajectory."""
    quaternions = np.asarray(root_quat_wxyz, dtype=np.float64)
    norms = np.linalg.norm(quaternions, axis=-1, keepdims=True)
    if np.any(norms < 1.0e-8):
        raise ValueError("Motion archive contains a zero-length root quaternion.")
    quaternions = quaternions / norms

    frame_count = root_pos.shape[0]
    left = np.maximum(np.arange(frame_count) - 1, 0)
    right = np.minimum(np.arange(frame_count) + 1, frame_count - 1)
    elapsed = ((right - left) / fps)[:, None]
    linear = (np.asarray(root_pos, dtype=np.float64)[right] - root_pos[left]) / elapsed

    previous_conjugate = quaternions[left].copy()
    previous_conjugate[:, 1:] *= -1.0
    current = quaternions[right]
    w1, x1, y1, z1 = np.moveaxis(current, -1, 0)
    w2, x2, y2, z2 = np.moveaxis(previous_conjugate, -1, 0)
    delta = np.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        axis=-1,
    )
    delta /= np.maximum(np.linalg.norm(delta, axis=-1, keepdims=True), 1.0e-12)
    delta = np.where(delta[:, :1] < 0.0, -delta, delta)
    sin_half = np.linalg.norm(delta[:, 1:], axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(sin_half, np.maximum(delta[:, :1], 0.0))
    scale = np.divide(angle, sin_half, out=np.full_like(angle, 2.0), where=sin_half > 1.0e-7)
    angular = delta[:, 1:] * scale / elapsed
    return linear, angular


def _resample_sequence(sequence: MotionSequence, target_fps: float) -> MotionSequence:
    if target_fps <= 0.0 or not np.isfinite(target_fps):
        raise ValueError(f"target_fps must be finite and positive, got {target_fps!r}.")
    if np.isclose(sequence.fps, target_fps, rtol=0.0, atol=1.0e-6):
        return sequence

    duration = (sequence.num_frames - 1) / sequence.fps
    target_count = int(np.floor(duration * target_fps + 1.0e-8)) + 1
    if target_count < 2:
        raise ValueError(f"Motion clip {sequence.source_path} is too short to resample to {target_fps:g} Hz.")
    source_times = np.arange(sequence.num_frames, dtype=np.float64) / sequence.fps
    target_times = np.arange(target_count, dtype=np.float64) / target_fps

    root_pos = _interpolate_vectors(sequence.root_pos, source_times, target_times)
    root_quat_wxyz = _interpolate_quaternions(sequence.root_quat_wxyz, source_times, target_times)
    root_lin_vel, root_ang_vel = _finite_difference_root_motion(root_pos, root_quat_wxyz, target_fps)
    return MotionSequence(
        profile=sequence.profile,
        fps=float(target_fps),
        root_pos=root_pos.astype(np.float32),
        root_quat_wxyz=root_quat_wxyz.astype(np.float32),
        joint_pos=_interpolate_angles(sequence.joint_pos, source_times, target_times).astype(np.float32),
        key_body_pos=_interpolate_vectors(sequence.key_body_pos, source_times, target_times).astype(np.float32),
        root_lin_vel=root_lin_vel.astype(np.float32),
        root_ang_vel=root_ang_vel.astype(np.float32),
        source_path=sequence.source_path,
    )


def load_motion_npz(
    path: str | Path,
    profile: str | SmpRobotProfile,
    *,
    target_fps: float | None = None,
) -> MotionSequence:
    """Load, name-map, and optionally resample one standard motion NPZ.

    Quaternion inputs are expected in Isaac Lab's scalar-first ``wxyz`` order.
    The returned sequence is chronological (oldest frame first).
    """

    source_path = Path(path).expanduser().resolve()
    if not source_path.is_file():
        raise FileNotFoundError(f"Motion archive does not exist: {source_path}")
    robot_profile = get_profile(profile)

    with np.load(source_path, allow_pickle=False) as archive:
        joint_names = _decode_names(_select_field(archive, "joint_names", source_path), "joint_names", source_path)
        body_names = _decode_names(_select_field(archive, "body_names", source_path), "body_names", source_path)
        joint_values = np.asarray(_select_field(archive, "joint_pos", source_path))
        body_positions = np.asarray(_select_field(archive, "body_pos", source_path))
        body_quaternions = np.asarray(_select_field(archive, "body_quat", source_path))
        fps_values = np.asarray(_select_field(archive, "fps", source_path))

    if fps_values.size != 1:
        raise ValueError(f"fps in {source_path} must be a numeric scalar.")
    fps = float(fps_values.reshape(-1)[0])
    if fps <= 0.0 or not np.isfinite(fps):
        raise ValueError(f"fps in {source_path} must be finite and positive, got {fps!r}.")

    num_frames = int(joint_values.shape[0]) if joint_values.ndim >= 1 else 0
    if num_frames < 2:
        raise ValueError(f"Motion archive {source_path} must contain at least two frames.")
    joint_values = _as_float_array(joint_values, (len(joint_names),), "joint_pos", source_path)
    body_positions = _as_float_array(body_positions, (len(body_names), 3), "body_pos", source_path)
    body_quaternions = _as_float_array(body_quaternions, (len(body_names), 4), "body_quat", source_path)
    for field_name, values in (
        ("body_pos", body_positions),
        ("body_quat", body_quaternions),
    ):
        if values.shape[0] != num_frames:
            raise ValueError(f"{field_name} in {source_path} has {values.shape[0]} frames; joint_pos has {num_frames}.")

    joint_indices = _name_indices(joint_names, robot_profile.joint_names, "joints", source_path)
    body_indices = _name_indices(
        body_names,
        (robot_profile.root_body_name, *robot_profile.key_body_names),
        "bodies",
        source_path,
    )
    root_index = body_indices[0]
    key_body_indices = body_indices[1:]
    root_pos = body_positions[:, root_index]
    root_quat_wxyz = body_quaternions[:, root_index]
    root_lin_vel, root_ang_vel = _finite_difference_root_motion(root_pos, root_quat_wxyz, fps)
    sequence = MotionSequence(
        profile=robot_profile,
        fps=fps,
        root_pos=root_pos.astype(np.float32),
        root_quat_wxyz=root_quat_wxyz.astype(np.float32),
        joint_pos=joint_values[:, joint_indices].astype(np.float32),
        key_body_pos=body_positions[:, key_body_indices].astype(np.float32),
        root_lin_vel=root_lin_vel.astype(np.float32),
        root_ang_vel=root_ang_vel.astype(np.float32),
        source_path=source_path,
    )
    return _resample_sequence(sequence, robot_profile.control_fps if target_fps is None else target_fps)


def motion_to_windows(
    sequence: MotionSequence,
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = 1,
    canonicalization_batch_size: int = 4096,
) -> torch.Tensor:
    """Convert one clip into chronological canonical windows ``[N, T, F]``."""

    if window_size < 2:
        raise ValueError("SMP window_size must be at least 2.")
    if stride < 1:
        raise ValueError("SMP window stride must be at least 1.")
    if canonicalization_batch_size < 1:
        raise ValueError("canonicalization_batch_size must be at least 1.")
    if sequence.num_frames < window_size:
        raise ValueError(
            f"Motion clip {sequence.source_path} has {sequence.num_frames} frames, fewer than "
            f"the required window size {window_size}."
        )

    starts = np.arange(0, sequence.num_frames - window_size + 1, stride, dtype=np.int64)
    offsets = np.arange(window_size, dtype=np.int64)
    chunks: list[torch.Tensor] = []
    for batch_start in range(0, starts.size, canonicalization_batch_size):
        batch_starts = starts[batch_start : batch_start + canonicalization_batch_size]
        frame_indices = batch_starts[:, None] + offsets[None, :]
        features = canonicalize_motion_features(
            torch.from_numpy(sequence.root_pos[frame_indices]),
            torch.from_numpy(sequence.root_quat_wxyz[frame_indices]),
            torch.from_numpy(sequence.joint_pos[frame_indices]),
            torch.from_numpy(sequence.key_body_pos[frame_indices]),
            torch.from_numpy(sequence.root_lin_vel[frame_indices]),
            torch.from_numpy(sequence.root_ang_vel[frame_indices]),
        )
        chunks.append(features.detach().to(device="cpu", dtype=torch.float32))
    windows = torch.cat(chunks, dim=0).contiguous()
    expected_shape = (starts.size, window_size, sequence.profile.feature_dim)
    if tuple(windows.shape) != expected_shape:
        raise RuntimeError(f"Canonical feature builder returned {tuple(windows.shape)}, expected {expected_shape}.")
    if not torch.isfinite(windows).all():
        raise ValueError(f"Canonical windows from {sequence.source_path} contain non-finite values.")
    return windows


def robust_feature_bounds(
    windows: torch.Tensor | np.ndarray,
    *,
    lower_quantile: float = 0.01,
    upper_quantile: float = 0.99,
    minimum_range: float = DEFAULT_MINIMUM_FEATURE_RANGE,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute robust bounds with a floor that avoids amplifying constant channels."""

    values = torch.as_tensor(windows, dtype=torch.float32, device="cpu")
    if values.ndim < 2:
        raise ValueError("SMP features must have at least a sample and feature dimension.")
    if values.numel() == 0:
        raise ValueError("Cannot compute SMP feature bounds from an empty tensor.")
    if not 0.0 <= lower_quantile < upper_quantile <= 1.0:
        raise ValueError("Feature-bound quantiles must satisfy 0 <= lower < upper <= 1.")
    if minimum_range <= 0.0:
        raise ValueError("minimum_range must be positive.")
    flattened = values.reshape(-1, values.shape[-1])
    if not torch.isfinite(flattened).all():
        raise ValueError("Cannot compute SMP feature bounds from non-finite values.")
    bounds = torch.quantile(flattened, torch.tensor([lower_quantile, upper_quantile]), dim=0)
    q_low, q_high = bounds[0], bounds[1]
    center = 0.5 * (q_low + q_high)
    half_range = torch.clamp(0.5 * (q_high - q_low), min=0.5 * minimum_range)
    return (center - half_range).contiguous(), (center + half_range).contiguous()


def _dataset_payload(
    windows: torch.Tensor,
    profile: SmpRobotProfile,
    q_low: torch.Tensor,
    q_high: torch.Tensor,
    source_files: Sequence[str],
    normalization_source_files: Sequence[str],
) -> dict[str, np.ndarray]:
    metadata = profile.to_metadata()
    return {
        "format": np.asarray(DATASET_FORMAT),
        "format_version": np.asarray(DATASET_FORMAT_VERSION, dtype=np.int64),
        "schema_version": np.asarray(SMP_SCHEMA_VERSION, dtype=np.int64),
        "profile_name": np.asarray(profile.name),
        "root_body_name": np.asarray(profile.root_body_name),
        "joint_names": np.asarray(profile.joint_names, dtype=np.str_),
        "key_body_names": np.asarray(profile.key_body_names, dtype=np.str_),
        "feature_dim": np.asarray(metadata["feature_dim"], dtype=np.int64),
        "control_fps": np.asarray(profile.control_fps, dtype=np.float64),
        "history_order": np.asarray(metadata["history_order"]),
        "quaternion_order": np.asarray(metadata["quaternion_order"]),
        "rotation_6d_columns": np.asarray(metadata["rotation_6d_columns"]),
        "up_axis": np.asarray(metadata["up_axis"]),
        "canonical_anchor": np.asarray(metadata["canonical_anchor"]),
        "root_height": np.asarray(metadata["root_height"]),
        "key_body_frame": np.asarray(metadata["key_body_frame"]),
        "root_pose_source_frame": np.asarray(metadata["root_pose_source_frame"]),
        "root_velocity_source_frame": np.asarray(metadata["root_velocity_source_frame"]),
        "window_size": np.asarray(windows.shape[1], dtype=np.int64),
        "source_files": np.asarray(tuple(source_files), dtype=np.str_),
        "normalization_source_files": np.asarray(tuple(normalization_source_files), dtype=np.str_),
        "windows": windows.numpy().astype(np.float32, copy=False),
        "q_low": q_low.numpy().astype(np.float32, copy=False),
        "q_high": q_high.numpy().astype(np.float32, copy=False),
    }


def build_smp_dataset(
    input_paths: Sequence[str | Path],
    output_path: str | Path,
    profile: str | SmpRobotProfile,
    *,
    window_size: int = DEFAULT_WINDOW_SIZE,
    stride: int = 1,
    target_fps: float | None = None,
    normalization_paths: Sequence[str | Path] | None = None,
) -> dict[str, Any]:
    """Build one SMP dataset, optionally fitting bounds on a broader corpus."""

    robot_profile = get_profile(profile)
    paths = tuple(Path(path).expanduser().resolve() for path in input_paths)
    if not paths:
        raise ValueError("At least one source motion archive is required.")
    if len(paths) != len(set(paths)):
        raise ValueError("Source motion paths must be unique; duplicate clips would change dataset weighting.")
    requested_fps = robot_profile.control_fps if target_fps is None else float(target_fps)
    if not np.isclose(requested_fps, robot_profile.control_fps, rtol=0.0, atol=1.0e-6):
        raise ValueError(
            f"Profile {robot_profile.name!r} requires {robot_profile.control_fps:g} Hz windows; "
            f"got target_fps={requested_fps:g}."
        )

    clip_windows: list[torch.Tensor] = []
    source_frame_counts: list[int] = []
    source_window_counts: list[int] = []
    for path in paths:
        sequence = load_motion_npz(path, robot_profile, target_fps=requested_fps)
        windows = motion_to_windows(sequence, window_size=window_size, stride=stride)
        clip_windows.append(windows)
        source_frame_counts.append(sequence.num_frames)
        source_window_counts.append(int(windows.shape[0]))

    all_windows = torch.cat(clip_windows, dim=0).contiguous()
    # Retain the resolved source identity so same-named clips from different
    # directories do not become ambiguous in an experiment artifact.
    source_files = tuple(str(path) for path in paths)

    if normalization_paths is None:
        normalization_files = source_files
        normalization_windows = all_windows
        normalization_window_count = int(all_windows.shape[0])
    else:
        resolved_normalization_paths = tuple(Path(path).expanduser().resolve() for path in normalization_paths)
        if not resolved_normalization_paths:
            raise ValueError("normalization_paths must contain at least one motion archive when provided.")
        if len(resolved_normalization_paths) != len(set(resolved_normalization_paths)):
            raise ValueError("Normalization motion paths must be unique.")
        normalization_chunks: list[torch.Tensor] = []
        for path in resolved_normalization_paths:
            sequence = load_motion_npz(path, robot_profile, target_fps=requested_fps)
            normalization_chunks.append(motion_to_windows(sequence, window_size=window_size, stride=1))
        normalization_windows = torch.cat(normalization_chunks, dim=0).contiguous()
        normalization_files = tuple(str(path) for path in resolved_normalization_paths)
        normalization_window_count = int(normalization_windows.shape[0])

    q_low, q_high = robust_feature_bounds(normalization_windows)
    payload = _dataset_payload(
        all_windows,
        robot_profile,
        q_low,
        q_high,
        source_files,
        normalization_files,
    )

    destination = Path(output_path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb", suffix=".npz", prefix=f".{destination.name}.", dir=destination.parent, delete=False
        ) as temporary:
            temporary_path = Path(temporary.name)
            np.savez_compressed(temporary, **payload)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()

    return {
        "output": str(destination),
        "profile": robot_profile.name,
        "control_fps": robot_profile.control_fps,
        "window_size": window_size,
        "feature_dim": robot_profile.feature_dim,
        "num_source_clips": len(paths),
        "source_frames": source_frame_counts,
        "source_windows": source_window_counts,
        "num_windows": int(all_windows.shape[0]),
        "normalization_source_files": list(normalization_files),
        "normalization_num_windows": normalization_window_count,
    }


_DATASET_KEYS = {
    "format",
    "format_version",
    "schema_version",
    "profile_name",
    "root_body_name",
    "joint_names",
    "key_body_names",
    "feature_dim",
    "control_fps",
    "history_order",
    "quaternion_order",
    "rotation_6d_columns",
    "up_axis",
    "canonical_anchor",
    "root_height",
    "key_body_frame",
    "root_pose_source_frame",
    "root_velocity_source_frame",
    "window_size",
    "source_files",
    "normalization_source_files",
    "windows",
    "q_low",
    "q_high",
}


def _scalar(archive: np.lib.npyio.NpzFile, name: str) -> Any:
    values = np.asarray(archive[name])
    if values.size != 1:
        raise ValueError(f"SMP dataset field {name!r} must be scalar, got shape {values.shape}.")
    return values.reshape(-1)[0].item()


def load_window_dataset(
    path: str | Path,
    expected_profile: str | SmpRobotProfile | None = None,
) -> SmpWindowDataset:
    """Load and strictly validate a canonical SMP dataset without pickle."""

    dataset_path = Path(path).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"SMP dataset does not exist: {dataset_path}")
    with np.load(dataset_path, allow_pickle=False) as archive:
        missing = sorted(_DATASET_KEYS.difference(archive.files))
        unknown = sorted(set(archive.files).difference(_DATASET_KEYS))
        if missing or unknown:
            raise ValueError(f"Invalid SMP dataset fields in {dataset_path}: missing={missing}, unknown={unknown}.")
        if str(_scalar(archive, "format")) != DATASET_FORMAT:
            raise ValueError(f"Unsupported SMP dataset format in {dataset_path}.")
        if int(_scalar(archive, "format_version")) != DATASET_FORMAT_VERSION:
            raise ValueError(f"Unsupported SMP dataset format version in {dataset_path}.")

        profile_metadata = {
            "schema_version": int(_scalar(archive, "schema_version")),
            "name": str(_scalar(archive, "profile_name")),
            "root_body_name": str(_scalar(archive, "root_body_name")),
            "joint_names": list(_decode_names(np.asarray(archive["joint_names"]), "joint_names", dataset_path)),
            "key_body_names": list(
                _decode_names(np.asarray(archive["key_body_names"]), "key_body_names", dataset_path)
            ),
            "feature_dim": int(_scalar(archive, "feature_dim")),
            "control_fps": float(_scalar(archive, "control_fps")),
            "history_order": str(_scalar(archive, "history_order")),
            "quaternion_order": str(_scalar(archive, "quaternion_order")),
            "rotation_6d_columns": str(_scalar(archive, "rotation_6d_columns")),
            "up_axis": str(_scalar(archive, "up_axis")),
            "canonical_anchor": str(_scalar(archive, "canonical_anchor")),
            "root_height": str(_scalar(archive, "root_height")),
            "key_body_frame": str(_scalar(archive, "key_body_frame")),
            "root_pose_source_frame": str(_scalar(archive, "root_pose_source_frame")),
            "root_velocity_source_frame": str(_scalar(archive, "root_velocity_source_frame")),
        }
        profile = profile_from_metadata(profile_metadata)
        if expected_profile is not None:
            validate_profile_metadata(profile_metadata, expected_profile)

        windows = torch.from_numpy(np.asarray(archive["windows"], dtype=np.float32).copy())
        q_low = torch.from_numpy(np.asarray(archive["q_low"], dtype=np.float32).copy())
        q_high = torch.from_numpy(np.asarray(archive["q_high"], dtype=np.float32).copy())
        source_files = _decode_names(np.asarray(archive["source_files"]), "source_files", dataset_path)
        normalization_source_files = _decode_names(
            np.asarray(archive["normalization_source_files"]),
            "normalization_source_files",
            dataset_path,
        )
        window_size = int(_scalar(archive, "window_size"))
        control_fps = float(_scalar(archive, "control_fps"))

    if windows.ndim != 3 or windows.shape[1] != window_size:
        raise ValueError(
            f"SMP dataset window metadata says T={window_size}, but tensor shape is {tuple(windows.shape)}."
        )
    if not np.isclose(control_fps, profile.control_fps, rtol=0.0, atol=1.0e-6):
        raise ValueError(f"SMP dataset rate {control_fps:g} Hz does not match profile rate {profile.control_fps:g} Hz.")
    return SmpWindowDataset(
        windows,
        profile,
        control_fps=control_fps,
        q_low=q_low,
        q_high=q_high,
        source_files=source_files,
        normalization_source_files=normalization_source_files,
        path=dataset_path,
    )


__all__ = [
    "DATASET_FORMAT",
    "DATASET_FORMAT_VERSION",
    "DEFAULT_MINIMUM_FEATURE_RANGE",
    "MotionSequence",
    "SmpWindowDataset",
    "build_smp_dataset",
    "load_motion_npz",
    "load_window_dataset",
    "motion_to_windows",
    "robust_feature_bounds",
]
