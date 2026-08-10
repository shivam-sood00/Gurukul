# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import pytest
import torch

SMP_ROOT = Path(__file__).resolve().parents[1] / "source/Gurukul/Gurukul/tasks/manager_based/smp"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def smp_modules():
    """Load pure SMP modules without importing Gurukul.tasks."""
    package_name = "_smp_features_contract_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(SMP_ROOT)]
    sys.modules[package_name] = package
    profiles = _load_module(f"{package_name}.profiles", SMP_ROOT / "profiles.py")
    features = _load_module(f"{package_name}.features", SMP_ROOT / "features.py")
    yield types.SimpleNamespace(features=features, profiles=profiles)
    for name in (f"{package_name}.features", f"{package_name}.profiles", package_name):
        sys.modules.pop(name, None)


def test_xz_rotation_6d_preserves_random_rotations(smp_modules):
    features = smp_modules.features
    generator = torch.Generator().manual_seed(1907)
    quaternion = features.normalize_quaternion(torch.randn(10_000, 4, generator=generator))
    probe = torch.randn(10_000, 3, generator=generator)

    rotation_6d = features.quaternion_to_rotation_6d(quaternion)
    reconstructed = features.rotation_6d_to_quaternion(rotation_6d)

    assert rotation_6d.shape == (10_000, 6)
    torch.testing.assert_close(
        features.quaternion_apply(reconstructed, probe),
        features.quaternion_apply(quaternion, probe),
        atol=2.0e-5,
        rtol=2.0e-5,
    )
    assert torch.min(torch.abs(torch.sum(quaternion * reconstructed, dim=-1))) > 1.0 - 2.0e-6


def test_canonicalization_supports_arbitrary_leading_dimensions(smp_modules):
    features = smp_modules.features
    generator = torch.Generator().manual_seed(29)
    leading_shape = (2, 3)
    num_frames, num_joints, num_key_bodies = 7, 2, 3
    root_pos = torch.randn(*leading_shape, num_frames, 3, generator=generator)
    root_pos[..., 2] += 1.0
    root_quat = features.normalize_quaternion(torch.randn(*leading_shape, num_frames, 4, generator=generator))
    joint_pos = torch.randn(*leading_shape, num_frames, num_joints, generator=generator)
    key_body_pos = root_pos.unsqueeze(-2) + torch.randn(
        *leading_shape, num_frames, num_key_bodies, 3, generator=generator
    )
    root_lin_vel = torch.randn(*leading_shape, num_frames, 3, generator=generator)
    root_ang_vel = torch.randn(*leading_shape, num_frames, 3, generator=generator)

    canonical = features.canonicalize_motion_features(
        root_pos,
        root_quat,
        joint_pos,
        key_body_pos,
        root_lin_vel,
        root_ang_vel,
    )

    feature_dim = 3 + 6 + num_joints + 3 * num_key_bodies + 3 + 3
    assert canonical.shape == (*leading_shape, num_frames, feature_dim)


def test_canonicalization_is_global_xy_translation_and_yaw_invariant(smp_modules):
    features = smp_modules.features
    generator = torch.Generator().manual_seed(103)
    batch_size, num_frames, num_joints, num_key_bodies = 8, 10, 5, 4
    root_pos = torch.randn(batch_size, num_frames, 3, generator=generator)
    root_pos[..., 2] += 1.0
    root_quat = features.normalize_quaternion(torch.randn(batch_size, num_frames, 4, generator=generator))
    joint_pos = torch.randn(batch_size, num_frames, num_joints, generator=generator)
    key_body_pos = root_pos.unsqueeze(-2) + torch.randn(batch_size, num_frames, num_key_bodies, 3, generator=generator)
    root_lin_vel = torch.randn(batch_size, num_frames, 3, generator=generator)
    root_ang_vel = torch.randn(batch_size, num_frames, 3, generator=generator)
    canonical = features.canonicalize_motion_features(
        root_pos,
        root_quat,
        joint_pos,
        key_body_pos,
        root_lin_vel,
        root_ang_vel,
    )

    yaw = torch.randn(batch_size, generator=generator)
    zeros = torch.zeros_like(yaw)
    global_yaw = torch.stack((torch.cos(0.5 * yaw), zeros, zeros, torch.sin(0.5 * yaw)), dim=-1)
    global_yaw_frames = global_yaw[:, None, :]
    translation = torch.randn(batch_size, 1, 3, generator=generator)
    translation[..., 2] = 0.0
    transformed_root_pos = features.quaternion_apply(global_yaw_frames, root_pos) + translation
    transformed_root_quat = features.quaternion_multiply(global_yaw_frames, root_quat)
    transformed_key_body_pos = (
        features.quaternion_apply(global_yaw_frames[:, :, None, :], key_body_pos) + translation[:, :, None, :]
    )
    transformed_lin_vel = features.quaternion_apply(global_yaw_frames, root_lin_vel)
    transformed_ang_vel = features.quaternion_apply(global_yaw_frames, root_ang_vel)
    transformed = features.canonicalize_motion_features(
        transformed_root_pos,
        transformed_root_quat,
        joint_pos,
        transformed_key_body_pos,
        transformed_lin_vel,
        transformed_ang_vel,
    )

    torch.testing.assert_close(transformed, canonical, atol=2.0e-5, rtol=2.0e-5)


def test_split_motion_features_matches_every_profile(smp_modules):
    features = smp_modules.features
    for profile in smp_modules.profiles.profiles():
        window = torch.zeros(2, 10, profile.feature_dim)
        parts = features.split_motion_features(window, profile)
        assert set(parts) == {
            "root_pos",
            "root_rot_6d",
            "joint_pos",
            "key_body_pos",
            "root_lin_vel",
            "root_ang_vel",
        }
        assert parts["root_pos"].shape == (2, 10, 3)
        assert parts["root_rot_6d"].shape == (2, 10, 6)
        assert parts["joint_pos"].shape == (2, 10, profile.num_joints)
        assert parts["key_body_pos"].shape == (2, 10, profile.num_key_bodies, 3)
        assert parts["root_lin_vel"].shape == (2, 10, 3)
        assert parts["root_ang_vel"].shape == (2, 10, 3)


def test_oldest_to_newest_window_is_anchored_at_last_frame(smp_modules):
    features = smp_modules.features
    root_pos = torch.tensor([[0.0, 0.0, 0.8], [1.0, 0.0, 0.9], [3.0, 0.0, 1.0]])
    root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).expand(3, -1)
    joint_pos = torch.tensor([[10.0], [20.0], [30.0]])
    key_body_pos = root_pos[:, None, :] + torch.tensor([[[0.5, 0.0, 0.0]]])
    root_lin_vel = torch.tensor([[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    root_ang_vel = torch.zeros_like(root_lin_vel)

    canonical = features.canonicalize_motion_features(
        root_pos,
        root_quat,
        joint_pos,
        key_body_pos,
        root_lin_vel,
        root_ang_vel,
    )

    torch.testing.assert_close(canonical[:, :3], torch.tensor([[-3.0, 0.0, 0.8], [-2.0, 0.0, 0.9], [0.0, 0.0, 1.0]]))
    torch.testing.assert_close(canonical[:, 9], torch.tensor([10.0, 20.0, 30.0]))
    torch.testing.assert_close(canonical[:, -6:-3], root_lin_vel)


def test_degenerate_rotation_6d_decodes_to_finite_rotations(smp_modules):
    features = smp_modules.features
    degenerate = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 2.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0, 3.0],
            [1.0e-30, 0.0, 0.0, 0.0, 0.0, 1.0e-30],
        ]
    )

    quaternion = features.rotation_6d_to_quaternion(degenerate)
    projected = features.quaternion_to_rotation_6d(quaternion)
    x_axis, z_axis = projected[:, :3], projected[:, 3:]

    assert torch.isfinite(quaternion).all()
    torch.testing.assert_close(torch.linalg.vector_norm(quaternion, dim=-1), torch.ones(4))
    torch.testing.assert_close(torch.linalg.vector_norm(x_axis, dim=-1), torch.ones(4))
    torch.testing.assert_close(torch.linalg.vector_norm(z_axis, dim=-1), torch.ones(4))
    torch.testing.assert_close(torch.sum(x_axis * z_axis, dim=-1), torch.zeros(4), atol=1.0e-6, rtol=0.0)


def test_feature_shape_and_dtype_validation(smp_modules):
    features = smp_modules.features
    profiles = smp_modules.profiles
    root_pos = torch.zeros(2, 4, 3)
    root_quat = torch.zeros(2, 4, 4)
    root_quat[..., 0] = 1.0
    joint_pos = torch.zeros(2, 4, 2)
    key_body_pos = torch.zeros(2, 4, 1, 3)
    velocity = torch.zeros(2, 4, 3)

    with pytest.raises(ValueError, match="root_quat_wxyz"):
        features.canonicalize_motion_features(
            root_pos,
            root_quat[:, :-1],
            joint_pos,
            key_body_pos,
            velocity,
            velocity,
        )
    with pytest.raises(TypeError, match="floating-point"):
        features.canonicalize_motion_features(
            root_pos.to(torch.int64),
            root_quat,
            joint_pos,
            key_body_pos,
            velocity,
            velocity,
        )
    with pytest.raises(ValueError, match="dimension 6"):
        features.rotation_6d_to_quaternion(torch.zeros(3, 5))
    with pytest.raises(ValueError, match="not broadcastable"):
        features.quaternion_multiply(torch.zeros(2, 4), torch.zeros(3, 4))
    with pytest.raises(ValueError, match="feature dimension"):
        features.split_motion_features(torch.zeros(2, profiles.G1_PROFILE.feature_dim - 1), "g1")
    with pytest.raises(TypeError, match="floating-point"):
        features.split_motion_features(
            torch.zeros(profiles.GO2_PROFILE.feature_dim, dtype=torch.int64),
            "go2",
        )
