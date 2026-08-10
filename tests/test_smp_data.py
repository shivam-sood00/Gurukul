# SPDX-License-Identifier: Apache-2.0

"""CPU-only contracts for the portable SMP motion-data pipeline."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SMP_ROOT = REPO_ROOT / "source/Gurukul/Gurukul/tasks/manager_based/smp"


@pytest.fixture(scope="module")
def smp_modules():
    """Load pure SMP modules without importing task registration or Isaac Lab."""

    package_name = "_smp_data_contract"
    package = types.ModuleType(package_name)
    package.__path__ = [str(SMP_ROOT)]
    sys.modules[package_name] = package
    loaded = {}
    try:
        for module_name in ("profiles", "features", "data"):
            qualified_name = f"{package_name}.{module_name}"
            spec = importlib.util.spec_from_file_location(qualified_name, SMP_ROOT / f"{module_name}.py")
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[qualified_name] = module
            spec.loader.exec_module(module)
            loaded[module_name] = module
        yield SimpleNamespace(**loaded)
    finally:
        for module_name in ("data", "features", "profiles"):
            sys.modules.pop(f"{package_name}.{module_name}", None)
        sys.modules.pop(package_name, None)


def _write_go2_clip(
    path: Path,
    smp_modules: SimpleNamespace,
    *,
    marker: float,
    translation: tuple[float, float],
    yaw_offset: float,
) -> dict[str, np.ndarray]:
    """Write a named 25 Hz clip whose storage order is deliberately scrambled."""

    profile = smp_modules.profiles.get_profile("go2")
    num_frames = 8
    fps = 25.0
    times = np.arange(num_frames, dtype=np.float32) / fps
    yaw_rate = 0.75
    yaw = yaw_offset + yaw_rate * times

    root_pos = np.stack(
        (
            translation[0] + 0.8 * times,
            translation[1] - 0.3 * times,
            np.full_like(times, 0.42),
        ),
        axis=-1,
    )
    root_quat = np.stack(
        (
            np.cos(0.5 * yaw),
            np.zeros_like(yaw),
            np.zeros_like(yaw),
            np.sin(0.5 * yaw),
        ),
        axis=-1,
    )
    canonical_joint_pos = (
        marker
        + np.arange(profile.num_joints, dtype=np.float32)[None, :]
        + 0.25 * np.arange(num_frames, dtype=np.float32)[:, None]
    )

    local_offsets = np.asarray(
        (
            (0.25, 0.15, -0.30),
            (0.25, -0.15, -0.30),
            (-0.25, 0.15, -0.30),
            (-0.25, -0.15, -0.30),
        ),
        dtype=np.float32,
    )
    cos_yaw = np.cos(yaw)[:, None]
    sin_yaw = np.sin(yaw)[:, None]
    key_body_pos = np.empty((num_frames, profile.num_key_bodies, 3), dtype=np.float32)
    key_body_pos[..., 0] = (
        root_pos[:, None, 0] + cos_yaw * local_offsets[None, :, 0] - sin_yaw * local_offsets[None, :, 1]
    )
    key_body_pos[..., 1] = (
        root_pos[:, None, 1] + sin_yaw * local_offsets[None, :, 0] + cos_yaw * local_offsets[None, :, 1]
    )
    key_body_pos[..., 2] = root_pos[:, None, 2] + local_offsets[None, :, 2]

    # Reverse joints and interleave the root among reordered feet. The values
    # are gathered by name so an implementation that trusts array order fails.
    source_joint_names = tuple(reversed(profile.joint_names))
    canonical_joint_index = {name: index for index, name in enumerate(profile.joint_names)}
    source_joint_pos = canonical_joint_pos[:, [canonical_joint_index[name] for name in source_joint_names]]
    source_body_names = (
        profile.key_body_names[3],
        profile.root_body_name,
        profile.key_body_names[1],
        profile.key_body_names[0],
        profile.key_body_names[2],
    )
    canonical_body_names = (profile.root_body_name, *profile.key_body_names)
    canonical_body_pos = np.concatenate((root_pos[:, None, :], key_body_pos), axis=1)
    canonical_body_quat = np.repeat(root_quat[:, None, :], len(canonical_body_names), axis=1)
    # Legacy Isaac NPZ velocities describe COM motion and are deliberately
    # incompatible here. The SMP loader must derive link velocity from pose.
    canonical_body_lin_vel = np.full((num_frames, len(canonical_body_names), 3), 77.0, dtype=np.float32)
    canonical_body_ang_vel = np.full_like(canonical_body_lin_vel, -55.0)
    canonical_body_index = {name: index for index, name in enumerate(canonical_body_names)}
    source_body_indices = [canonical_body_index[name] for name in source_body_names]

    np.savez_compressed(
        path,
        fps=np.asarray(fps, dtype=np.float32),
        joint_names=np.asarray(source_joint_names, dtype=np.str_),
        body_names=np.asarray(source_body_names, dtype=np.str_),
        joint_pos=source_joint_pos,
        body_pos_w=canonical_body_pos[:, source_body_indices],
        body_quat_w=canonical_body_quat[:, source_body_indices],
        body_lin_vel_w=canonical_body_lin_vel[:, source_body_indices],
        body_ang_vel_w=canonical_body_ang_vel[:, source_body_indices],
    )
    return {
        "joint_pos": canonical_joint_pos,
        "key_body_pos": key_body_pos,
        "root_pos": root_pos,
        "yaw": yaw,
    }


def _write_profile_clip(path: Path, smp_modules: SimpleNamespace, profile_name: str) -> None:
    """Write a small, deliberately reordered archive for any supported profile."""
    profile = smp_modules.profiles.get_profile(profile_name)
    num_frames = 12
    root_pos = np.zeros((num_frames, 3), dtype=np.float32)
    root_pos[:, 0] = np.arange(num_frames, dtype=np.float32) / 50.0
    root_pos[:, 2] = 0.75
    root_quat = np.zeros((num_frames, 4), dtype=np.float32)
    root_quat[:, 0] = 1.0
    joint_pos = np.arange(
        num_frames * profile.num_joints,
        dtype=np.float32,
    ).reshape(num_frames, profile.num_joints)
    offsets = np.zeros((profile.num_key_bodies, 3), dtype=np.float32)
    offsets[:, 0] = np.linspace(0.1, 0.3, profile.num_key_bodies)
    offsets[:, 1] = np.linspace(-0.2, 0.2, profile.num_key_bodies)
    key_body_pos = root_pos[:, None, :] + offsets[None, :, :]

    canonical_body_names = (profile.root_body_name, *profile.key_body_names)
    canonical_body_pos = np.concatenate((root_pos[:, None, :], key_body_pos), axis=1)
    canonical_body_quat = np.repeat(root_quat[:, None, :], len(canonical_body_names), axis=1)
    source_joint_names = tuple(reversed(profile.joint_names))
    source_body_names = tuple(reversed(canonical_body_names))
    joint_index = {name: index for index, name in enumerate(profile.joint_names)}
    body_index = {name: index for index, name in enumerate(canonical_body_names)}
    np.savez_compressed(
        path,
        fps=np.asarray(50.0, dtype=np.float32),
        joint_names=np.asarray(source_joint_names, dtype=np.str_),
        body_names=np.asarray(source_body_names, dtype=np.str_),
        joint_pos=joint_pos[:, [joint_index[name] for name in source_joint_names]],
        body_pos_w=canonical_body_pos[:, [body_index[name] for name in source_body_names]],
        body_quat_w=canonical_body_quat[:, [body_index[name] for name in source_body_names]],
    )


@pytest.mark.parametrize("profile_name", ("g1", "pm01", "go2"))
def test_motion_loader_covers_every_robot_profile(
    tmp_path: Path,
    smp_modules: SimpleNamespace,
    profile_name: str,
):
    source_path = tmp_path / f"{profile_name}_reordered_50hz.npz"
    _write_profile_clip(source_path, smp_modules, profile_name)
    profile = smp_modules.profiles.get_profile(profile_name)

    sequence = smp_modules.data.load_motion_npz(source_path, profile)
    windows = smp_modules.data.motion_to_windows(sequence)

    assert sequence.num_frames == 12
    assert sequence.joint_pos.shape == (12, profile.num_joints)
    assert sequence.key_body_pos.shape == (12, profile.num_key_bodies, 3)
    assert windows.shape == (3, 10, profile.feature_dim)
    assert torch.isfinite(windows).all()


def test_motion_loader_resamples_and_maps_names(tmp_path: Path, smp_modules: SimpleNamespace):
    source_path = tmp_path / "go2_reordered_25hz.npz"
    expected = _write_go2_clip(
        source_path,
        smp_modules,
        marker=10.0,
        translation=(4.0, -3.0),
        yaw_offset=0.35,
    )
    with np.load(source_path, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]).copy() for name in archive.files}
    wrapped_joint = np.asarray((3.00, 3.10, -3.08, -2.98, -2.88, -2.78, -2.68, -2.58), dtype=np.float32)
    payload["joint_pos"][:, 0] = wrapped_joint
    expected["joint_pos"][:, -1] = wrapped_joint
    np.savez_compressed(source_path, **payload)

    sequence = smp_modules.data.load_motion_npz(source_path, "unitree-go2")

    # Eight 25 Hz frames span 0.28 s, yielding fifteen frames at 50 Hz.
    assert sequence.fps == pytest.approx(50.0)
    assert sequence.num_frames == 15
    assert sequence.joint_pos.shape == (15, 12)
    assert sequence.key_body_pos.shape == (15, 4, 3)
    np.testing.assert_allclose(sequence.joint_pos[0], expected["joint_pos"][0], atol=1.0e-6)
    # Target frame two lands exactly on source frame one.
    np.testing.assert_allclose(sequence.joint_pos[2], expected["joint_pos"][1], atol=1.0e-6)
    np.testing.assert_allclose(sequence.root_pos[2], expected["root_pos"][1], atol=1.0e-6)
    np.testing.assert_allclose(sequence.key_body_pos[0], expected["key_body_pos"][0], atol=1.0e-6)
    # The midpoint across a +pi/-pi storage wrap follows the short physical
    # arc rather than linearly jumping through zero.
    assert sequence.joint_pos[5, -1] > 3.1
    np.testing.assert_allclose(sequence.root_lin_vel[1], (0.8, -0.3, 0.0), atol=1.0e-5)
    np.testing.assert_allclose(sequence.root_ang_vel[1], (0.0, 0.0, 0.75), atol=1.0e-5)

    resampled_yaw = smp_modules.features.quaternion_to_yaw(torch.from_numpy(sequence.root_quat_wxyz))
    assert float(resampled_yaw[1]) == pytest.approx(0.35 + 0.75 / 50.0, abs=1.0e-6)

    windows = smp_modules.data.motion_to_windows(sequence)
    assert windows.shape == (6, 10, 39)
    assert torch.isfinite(windows).all()
    q_low, q_high = smp_modules.data.robust_feature_bounds(windows)
    assert q_low.shape == q_high.shape == (39,)
    assert torch.isfinite(q_low).all() and torch.isfinite(q_high).all()
    assert torch.all(q_high > q_low)
    assert torch.all(q_high - q_low >= smp_modules.data.DEFAULT_MINIMUM_FEATURE_RANGE - 1.0e-6)


def test_pose_derived_root_velocity_uses_centered_differences(smp_modules: SimpleNamespace):
    fps = 50.0
    times = np.arange(7, dtype=np.float64) / fps
    root_pos = np.zeros((times.size, 3), dtype=np.float64)
    root_pos[:, 0] = times**2
    yaw = 0.5 * times**2
    root_quat = np.zeros((times.size, 4), dtype=np.float64)
    root_quat[:, 0] = np.cos(0.5 * yaw)
    root_quat[:, 3] = np.sin(0.5 * yaw)

    linear, angular = smp_modules.data._finite_difference_root_motion(root_pos, root_quat, fps)

    np.testing.assert_allclose(linear[1:-1, 0], 2.0 * times[1:-1], atol=1.0e-12)
    np.testing.assert_allclose(angular[1:-1, 2], times[1:-1], atol=1.0e-12)
    np.testing.assert_allclose(linear[:, 1:], 0.0, atol=1.0e-12)
    np.testing.assert_allclose(angular[:, :2], 0.0, atol=1.0e-12)


def test_dataset_roundtrip_keeps_clips_separate_and_metadata_safe(tmp_path: Path, smp_modules: SimpleNamespace):
    first_path = tmp_path / "go2_first.npz"
    second_path = tmp_path / "go2_second.npz"
    normalization_path = tmp_path / "go2_normalization.npz"
    _write_go2_clip(
        first_path,
        smp_modules,
        marker=10.0,
        translation=(0.0, 0.0),
        yaw_offset=0.1,
    )
    _write_go2_clip(
        second_path,
        smp_modules,
        marker=1000.0,
        translation=(50.0, -20.0),
        yaw_offset=-1.2,
    )
    _write_go2_clip(
        normalization_path,
        smp_modules,
        marker=5000.0,
        translation=(-30.0, 10.0),
        yaw_offset=0.8,
    )
    dataset_path = tmp_path / "go2_windows.npz"

    summary = smp_modules.data.build_smp_dataset(
        (first_path, second_path),
        dataset_path,
        "go2",
        normalization_paths=(normalization_path,),
    )
    dataset = smp_modules.data.load_window_dataset(dataset_path, expected_profile="unitree go2")

    # Each resampled clip has 15 frames and independently contributes six
    # windows. Concatenating frames before windowing would incorrectly give 21.
    assert summary["source_windows"] == [6, 6]
    assert summary["num_windows"] == 12
    assert summary["normalization_num_windows"] == 6
    assert dataset.windows.shape == (12, 10, 39)
    joint_windows = smp_modules.features.split_motion_features(dataset.windows, dataset.profile)["joint_pos"]
    assert torch.all(joint_windows[:6] < 100.0)
    assert torch.all(joint_windows[6:] > 900.0)
    assert dataset.source_files == (str(first_path.resolve()), str(second_path.resolve()))
    assert dataset.normalization_source_files == (str(normalization_path.resolve()),)
    assert torch.isfinite(dataset.q_low).all() and torch.isfinite(dataset.q_high).all()
    assert torch.all(dataset.q_high > dataset.q_low)
    assert float(dataset.q_high[9]) > 4000.0

    # Every array remains readable with pickle disabled and metadata is made
    # only from numeric or Unicode arrays (never object arrays).
    with np.load(dataset_path, allow_pickle=False) as archive:
        payload = {name: np.asarray(archive[name]).copy() for name in archive.files}
        assert all(values.dtype.kind != "O" for values in payload.values())
        assert payload["format"].item() == smp_modules.data.DATASET_FORMAT
        assert payload["profile_name"].item() == "go2"
        assert payload["windows"].shape == (12, 10, 39)
        assert payload["normalization_source_files"].tolist() == [str(normalization_path.resolve())]

    with pytest.raises(ValueError, match="mismatch"):
        smp_modules.data.load_window_dataset(dataset_path, expected_profile="g1")

    # A shape-compatible joint-order mutation must also fail exact profile
    # validation instead of silently reinterpreting channels.
    payload["joint_names"][[0, 1]] = payload["joint_names"][[1, 0]]
    mismatched_path = tmp_path / "go2_mismatched_order.npz"
    np.savez_compressed(mismatched_path, **payload)
    with pytest.raises(ValueError, match="joint_names"):
        smp_modules.data.load_window_dataset(mismatched_path, expected_profile="go2")

    with pytest.raises(ValueError, match="unique"):
        smp_modules.data.build_smp_dataset((first_path, first_path), tmp_path / "duplicate.npz", "go2")
    with pytest.raises(ValueError, match="unique"):
        smp_modules.data.build_smp_dataset(
            (first_path,),
            tmp_path / "duplicate_normalizer.npz",
            "go2",
            normalization_paths=(normalization_path, normalization_path),
        )
