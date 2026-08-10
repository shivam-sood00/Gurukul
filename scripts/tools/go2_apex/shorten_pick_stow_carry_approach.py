#!/usr/bin/env python3
"""Shorten the pick-stow-carry approach without starting the clip mid-gait."""

from __future__ import annotations

import argparse
import io
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("current", type=Path, help="Current shortened/easier manipulation NPZ.")
    parser.add_argument("--output", type=Path, required=True, help="Output NPZ.")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--original", type=Path, help="Original unshortened NPZ.")
    source.add_argument(
        "--git-revision",
        type=str,
        help="Read the original NPZ at this Git revision (for example, HEAD).",
    )
    parser.add_argument("--frames", type=int, default=125, help="Frames retained from the original approach.")
    return parser.parse_args()


def _load_archive(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _load_original(args: argparse.Namespace) -> dict[str, np.ndarray]:
    if args.original is not None:
        return _load_archive(args.original)
    repo_root = Path(__file__).resolve().parents[3]
    relative_path = args.current.resolve().relative_to(repo_root)
    payload = subprocess.check_output(
        ["git", "show", f"{args.git_revision}:{relative_path.as_posix()}"],
        cwd=repo_root,
    )
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        return {key: archive[key] for key in archive.files}


def _smoothstep(t: np.ndarray) -> np.ndarray:
    return t * t * (3.0 - 2.0 * t)


def _linear_blend(start: np.ndarray, end: np.ndarray, t: np.ndarray) -> np.ndarray:
    shaped_t = t.reshape((t.shape[0],) + (1,) * start.ndim)
    return start[None] + shaped_t * (end - start)[None]


def _smooth_blend(start: np.ndarray, end: np.ndarray, t: np.ndarray) -> np.ndarray:
    return _linear_blend(start, end, _smoothstep(t))


def _hermite(
    start: np.ndarray,
    end: np.ndarray,
    start_velocity: np.ndarray,
    end_velocity: np.ndarray,
    t: np.ndarray,
    duration: float,
) -> tuple[np.ndarray, np.ndarray]:
    shaped_t = t.reshape((t.shape[0],) + (1,) * start.ndim)
    t2 = shaped_t * shaped_t
    t3 = t2 * shaped_t
    h00 = 2.0 * t3 - 3.0 * t2 + 1.0
    h10 = t3 - 2.0 * t2 + shaped_t
    h01 = -2.0 * t3 + 3.0 * t2
    h11 = t3 - t2
    position = (
        h00 * start[None]
        + h10 * duration * start_velocity[None]
        + h01 * end[None]
        + h11 * duration * end_velocity[None]
    )
    dh00 = 6.0 * t2 - 6.0 * shaped_t
    dh10 = 3.0 * t2 - 4.0 * shaped_t + 1.0
    dh01 = -6.0 * t2 + 6.0 * shaped_t
    dh11 = 3.0 * t2 - 2.0 * shaped_t
    velocity = (
        dh00 * start[None]
        + dh10 * duration * start_velocity[None]
        + dh01 * end[None]
        + dh11 * duration * end_velocity[None]
    ) / duration
    return position, velocity


def _slerp(start: np.ndarray, end: np.ndarray, t: np.ndarray) -> np.ndarray:
    start_flat = start.reshape(-1, 4).astype(np.float64)
    end_flat = end.reshape(-1, 4).astype(np.float64)
    dot = np.sum(start_flat * end_flat, axis=-1)
    end_flat = np.where((dot < 0.0)[:, None], -end_flat, end_flat)
    dot = np.abs(dot).clip(-1.0, 1.0)
    result = np.empty((t.shape[0], start_flat.shape[0], 4), dtype=np.float64)
    for index, alpha in enumerate(t):
        close = dot > 0.9995
        blended = (1.0 - alpha) * start_flat + alpha * end_flat
        theta = np.arccos(dot)
        sin_theta = np.sin(theta)
        safe_denominator = np.where(sin_theta > 1.0e-8, sin_theta, 1.0)
        spherical = (
            np.sin((1.0 - alpha) * theta)[:, None] * start_flat
            + np.sin(alpha * theta)[:, None] * end_flat
        ) / safe_denominator[:, None]
        blended = np.where(close[:, None], blended, spherical)
        result[index] = blended / np.linalg.norm(blended, axis=-1, keepdims=True).clip(1.0e-12)
    return result.reshape((t.shape[0],) + start.shape)


def shorten_motion(
    current: dict[str, np.ndarray],
    original: dict[str, np.ndarray],
    frames: int,
) -> dict[str, np.ndarray]:
    current_count = int(current["joint_pos"].shape[0])
    original_count = int(original["joint_pos"].shape[0])
    if original_count - current_count != frames:
        raise ValueError(
            f"Expected original-current length difference {frames}, got {original_count - current_count}."
        )
    original_walk_end = int(original["approach_walk_end_frame"])
    original_settle_end = int(original["approach_settle_end_frame"])
    if frames * 2 != original_walk_end:
        raise ValueError(f"--frames must be half the original {original_walk_end}-frame approach.")

    new_settle_end = original_settle_end - frames
    blend_count = new_settle_end - frames
    if blend_count <= 0:
        raise ValueError("The shortened approach leaves no settle interval.")
    fps = float(current["fps"])
    duration = blend_count / fps
    t = np.arange(1, blend_count + 1, dtype=np.float64) / blend_count
    target_index = new_settle_end

    if "approach_xy_offset_w" in current:
        xy_offset = np.asarray(current["approach_xy_offset_w"], dtype=np.float32)
    elif "trim_xy_offset_w" in current:
        xy_offset = np.asarray(current["trim_xy_offset_w"], dtype=np.float32)
    else:
        raise KeyError("Current motion must contain approach_xy_offset_w or trim_xy_offset_w.")
    object_z_offset = float(current["object_pos_w"][0, 0, 2] - original["object_pos_w"][frames, 0, 2])
    prefix: dict[str, np.ndarray] = {}
    for key, value in current.items():
        original_value = original.get(key)
        if value.ndim > 0 and value.shape[0] == current_count:
            if original_value is None or original_value.shape[0] != original_count:
                raise ValueError(f"Missing original frame-aligned channel: {key}.")
            prefix[key] = original_value[: frames + 1].copy()

    prefix["object_pos_w"][..., :2] -= xy_offset
    prefix["object_pos_w"][..., 2] += object_z_offset

    blend: dict[str, np.ndarray] = {}
    blend["joint_pos"], blend["joint_vel"] = _hermite(
        prefix["joint_pos"][-1],
        current["joint_pos"][target_index],
        prefix["joint_vel"][-1],
        current["joint_vel"][target_index],
        t,
        duration,
    )
    blend["body_pos_w"], blend["body_lin_vel_w"] = _hermite(
        prefix["body_pos_w"][-1],
        current["body_pos_w"][target_index],
        prefix["body_lin_vel_w"][-1],
        current["body_lin_vel_w"][target_index],
        t,
        duration,
    )
    blend["body_quat_w"] = _slerp(prefix["body_quat_w"][-1], current["body_quat_w"][target_index], t)
    blend["body_ang_vel_w"] = _linear_blend(
        prefix["body_ang_vel_w"][-1], current["body_ang_vel_w"][target_index], t
    )
    blend["arm_ee_pos_w"] = _smooth_blend(
        prefix["arm_ee_pos_w"][-1], current["arm_ee_pos_w"][target_index], t
    )
    blend["arm_ee_quat_w"] = _slerp(
        prefix["arm_ee_quat_w"][-1], current["arm_ee_quat_w"][target_index], t
    )
    blend["object_pos_w"] = _smooth_blend(
        prefix["object_pos_w"][-1], current["object_pos_w"][target_index], t
    )
    blend["object_quat_w"] = _slerp(
        prefix["object_quat_w"][-1], current["object_quat_w"][target_index], t
    )
    blend["gripper_joint_pos"] = _smooth_blend(
        prefix["gripper_joint_pos"][-1], current["gripper_joint_pos"][target_index], t
    )
    blend["gripper_joint_vel"] = _linear_blend(
        prefix["gripper_joint_vel"][-1], current["gripper_joint_vel"][target_index], t
    )
    blend["command_lin_vel_xy"] = blend["body_lin_vel_w"][:, 0, :2]
    blend["command_ang_vel_z"] = blend["body_ang_vel_w"][:, 0, 2:3]
    blend["skill"] = np.zeros((blend_count, 1), dtype=current["skill"].dtype)
    blend["reference_foot_contact"] = np.ones(
        (blend_count, current["reference_foot_contact"].shape[1]), dtype=bool
    )
    blend["reference_airborne"] = np.zeros(blend_count, dtype=bool)
    blend["object_attached"] = np.zeros(
        (blend_count, current["object_attached"].shape[1]), dtype=bool
    )

    result: dict[str, np.ndarray] = {}
    for key, value in current.items():
        if value.ndim > 0 and value.shape[0] == current_count:
            if key not in blend:
                raise ValueError(f"No settle reconstruction rule for frame-aligned channel: {key}.")
            combined = np.concatenate([prefix[key], blend[key], value[new_settle_end + 1 :]], axis=0)
            if combined.shape[0] != current_count:
                raise RuntimeError(f"Reconstructed {key} has {combined.shape[0]} frames, expected {current_count}.")
            result[key] = combined.astype(value.dtype, copy=False)
        elif key not in {"trim_xy_offset_w", "trimmed_prefix_frames"}:
            result[key] = value.copy()

    result["approach_shortened_frames"] = np.asarray(frames, dtype=np.int64)
    result["approach_xy_offset_w"] = xy_offset
    result["approach_reconstruction"] = np.asarray("original_prefix+cubic_settle")
    return result


def save_motion(path: Path, data: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix=f".{path.stem}.", suffix=".npz", dir=path.parent, delete=False) as handle:
        temporary_path = Path(handle.name)
    try:
        np.savez_compressed(temporary_path, **data)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    args = parse_args()
    current = _load_archive(args.current)
    original = _load_original(args)
    shortened = shorten_motion(current, original, args.frames)
    save_motion(args.output, shortened)
    print(
        f"Kept original frames 0..{args.frames}, rebuilt the settle through frame "
        f"{int(shortened['approach_settle_end_frame'])}, and wrote "
        f"{shortened['joint_pos'].shape[0]} frames to {args.output}."
    )


if __name__ == "__main__":
    main()
