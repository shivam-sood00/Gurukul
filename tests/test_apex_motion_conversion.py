"""Frame conversion and bad-data rejection for the standalone APEX importer."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
CONVERTER = ROOT / "source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/apex_csv_to_motion_npz.py"
SPEC = importlib.util.spec_from_file_location("apex_converter_test", CONVERTER)
c = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(c)


def motion_csv(tmp_path, *, name="canter.csv", corrupt=False, zero_feet=False):
    data = np.zeros((4, 40), dtype=np.float32)
    data[:, 0] = 1
    data[:, 5] = 1
    data[:, 21] = 0.4
    data[:, 22:34] = np.tile([0, 0, -0.3], 4)
    data[:, 6:18] = np.tile([0.1, 0.8, -1.5], 4)
    # A 90 degree yaw followed by 30 degree roll: independent expected rotations.
    yaw = np.array([[np.sqrt(0.5), 0, 0, np.sqrt(0.5)]], dtype=np.float32)
    roll = np.array([[np.cos(np.pi / 12), np.sin(np.pi / 12), 0, 0]], dtype=np.float32)
    q = c.quat_multiply(yaw, roll)[0]
    data[:, 36:40] = q[[1, 2, 3, 0]]
    if corrupt:
        data[-1, 22] = 35
        data[-1, 6] = 100  # Must not influence selected-range finite differences.
    if zero_feet:
        data[:, 22:34] = 0
    path = tmp_path / name
    np.savetxt(path, data, delimiter=",")
    return path, data


def convert(tmp_path, path, **kwargs):
    out = tmp_path / "converted.npz"
    c.convert(path, out, 50, list(c.CANONICAL_LEGS), **kwargs)
    with np.load(out) as data:
        return dict(data)


def test_body_velocity_and_yaw_feet_are_converted_to_world(tmp_path):
    path, _ = motion_csv(tmp_path)
    data = convert(tmp_path, path)
    np.testing.assert_allclose(data["body_lin_vel_w"][:, 0], np.tile([0, 1, 0], (4, 1)), atol=1e-6)
    np.testing.assert_allclose(data["body_ang_vel_w"][:, 0], np.tile([0.5, 0, np.sqrt(0.75)], (4, 1)), atol=1e-6)
    np.testing.assert_allclose(data["body_pos_w"][:, 1:, 2], 0.1, atol=1e-6)
    np.testing.assert_allclose(data["body_pos_w"][:, 1:, :2], 0, atol=1e-6)
    assert data["source_velocity_frame"].item() == "body"
    assert data["motion_conversion_version"].item() == 2


def test_stmr_world_velocity_and_explicit_override(tmp_path):
    path, _ = motion_csv(tmp_path, name="walk_STMR_resampled.csv")
    data = convert(tmp_path, path)
    np.testing.assert_allclose(data["body_lin_vel_w"][:, 0], np.tile([1, 0, 0], (4, 1)))
    explicit = convert(tmp_path, path, velocity_frame="body", feet_frame="body")
    np.testing.assert_allclose(explicit["body_lin_vel_w"][:, 0], np.tile([0, 1, 0], (4, 1)), atol=1e-6)
    assert not np.allclose(explicit["body_pos_w"][:, 1:], data["body_pos_w"][:, 1:])


def test_corrupt_frame_is_rejected_and_selection_precedes_differentiation(tmp_path):
    path, _ = motion_csv(tmp_path, corrupt=True)
    with pytest.raises(ValueError, match="frame 3.*35."):
        convert(tmp_path, path)
    data = convert(tmp_path, path, frame_range=(0, 3))
    assert data["joint_pos"].shape[0] == 3
    np.testing.assert_allclose(data["joint_vel"], 0)
    assert data["source_frame_range"].tolist() == [0, 3]


@pytest.mark.parametrize("problem", ["nan", "zero_quaternion", "short_row", "nonnumeric", "bad_first_row"])
def test_bad_motion_data_is_not_silently_used_or_dropped(tmp_path, problem):
    path, data = motion_csv(tmp_path)
    if problem == "nan":
        data[2, 0] = np.nan
    elif problem == "zero_quaternion":
        data[2, 36:40] = 0
    np.savetxt(path, data, delimiter=",")
    if problem in ("short_row", "nonnumeric"):
        lines = path.read_text().splitlines()
        lines[2] = "1,2" if problem == "short_row" else "invalid," * 39 + "invalid"
        path.write_text("\n".join(lines))
    if problem == "bad_first_row":
        lines = path.read_text().splitlines()
        lines[0] = "invalid," + lines[0].split(",", 1)[1]
        path.write_text("\n".join(lines))
    with pytest.raises(ValueError):
        convert(tmp_path, path)


@pytest.mark.parametrize("selection", [(-1, 3), (0, 5), (2, 2), (2, 3)])
def test_frame_selection_bounds(tmp_path, selection):
    path, _ = motion_csv(tmp_path)
    with pytest.raises(ValueError, match="frame range"):
        convert(tmp_path, path, frame_range=selection)


def test_fk_fallback_keeps_canonical_foot_order(tmp_path):
    path, data = motion_csv(tmp_path, zero_feet=True)
    canonical = convert(tmp_path, path)
    order = ["FR", "FL", "RR", "RL"]
    indices = [c.CANONICAL_LEGS.index(leg) for leg in order]
    data[:, 6:18] = data[:, 6:18].reshape(4, 4, 3)[:, indices].reshape(4, 12)
    np.savetxt(path, data, delimiter=",")
    out = tmp_path / "permuted.npz"
    c.convert(path, out, 50, order)
    with np.load(out) as permuted:
        np.testing.assert_allclose(permuted["body_pos_w"], canonical["body_pos_w"])
        assert permuted["source_feet_frame"].item() == "body"


def load_training_motion_class(family="go2_apex"):
    """Exercise the production loader without importing simulator-dependent commands."""
    import ast
    import glob
    import os

    import torch

    source = ROOT / f"source/Gurukul/Gurukul/tasks/manager_based/{family}/mdp/commands.py"
    tree = ast.parse(source.read_text())
    node = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "MotionLoader")
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), node],
        type_ignores=[],
    )
    validator_spec = importlib.util.spec_from_file_location(
        "apex_motion_validation_test", ROOT / "source/Gurukul/Gurukul/utils/motion_validation.py"
    )
    validator = importlib.util.module_from_spec(validator_spec)
    validator_spec.loader.exec_module(validator)
    ns = {
        "np": np,
        "torch": torch,
        "glob": glob,
        "os": os,
        "Path": Path,
        "validate_motion_arrays": validator.validate_motion_arrays,
    }
    exec(compile(ast.fix_missing_locations(module), str(source), "exec"), ns)
    return ns["MotionLoader"]


def test_training_loader_rejects_corrupt_existing_npz(tmp_path):
    motion_loader = load_training_motion_class()
    path, _ = motion_csv(tmp_path)
    data = convert(tmp_path, path)
    good = tmp_path / "converted.npz"
    loader = motion_loader(str(good), max_foot_distance=0.9)
    assert loader.time_step_total == 4
    names = data.pop("body_names").tolist()
    unnamed = tmp_path / "unnamed.npz"
    np.savez(unnamed, **data)
    legacy = motion_loader(str(unnamed), max_foot_distance=0.9, fallback_body_names=names)
    assert legacy.time_step_total == 4
    data["body_names"] = np.asarray(names)
    data["body_pos_w"][-1, 1, 0] = 35
    bad = tmp_path / "bad.npz"
    np.savez(bad, **data)
    with pytest.raises(ValueError, match="frame 3.*FL_foot"):
        motion_loader(str(bad), max_foot_distance=0.9)
    data["body_pos_w"][-1, 1, 0] = np.nan
    np.savez(bad, **data)
    with pytest.raises(ValueError, match="nonfinite body_pos_w"):
        motion_loader(str(bad), max_foot_distance=0.9)


@pytest.mark.parametrize("family", ["go2_apex", "beyondmimic"])
@pytest.mark.parametrize("problem", ["missing_velocity", "nan_joint", "zero_quaternion", "wrong_shape"])
def test_both_tracker_loaders_reject_corrupt_motion_arrays(tmp_path, family, problem):
    motion_loader = load_training_motion_class(family)
    path, _ = motion_csv(tmp_path)
    data = convert(tmp_path, path)
    good = motion_loader(str(tmp_path / "converted.npz"), body_indexes=[0, 1])
    assert good.joint_pos.shape == (4, 12)
    if problem == "missing_velocity":
        data.pop("joint_vel")
    elif problem == "nan_joint":
        data["joint_pos"][1, 0] = np.nan
    elif problem == "zero_quaternion":
        data["body_quat_w"][1, 0] = 0
    else:
        data["body_lin_vel_w"] = data["body_lin_vel_w"][:-1]
    bad = tmp_path / "bad.npz"
    np.savez(bad, **data)
    with pytest.raises(ValueError, match="Motion file"):
        motion_loader(str(bad), body_indexes=[0, 1])


@pytest.mark.parametrize(
    "problem", ["arm_position", "arm_nan", "gripper_velocity", "object_shape", "object_quaternion"]
)
def test_arm_loader_rejects_corrupt_manipulation_channels(tmp_path, problem):
    motion_loader = load_training_motion_class()
    path, _ = motion_csv(tmp_path)
    data = convert(tmp_path, path)
    data.update(
        arm_ee_pos_w=np.zeros((4, 3)),
        gripper_joint_pos=np.zeros((4, 2)),
        gripper_joint_vel=np.zeros((4, 2)),
        object_names=np.array(["can"]),
        object_pos_w=np.zeros((4, 1, 3)),
        object_quat_w=np.tile([1.0, 0, 0, 0], (4, 1, 1)),
    )
    good = tmp_path / "good_arm.npz"
    np.savez(good, **data)
    assert motion_loader(str(good)).time_step_total == 4
    if problem == "arm_position":
        data["arm_ee_pos_w"] = data["arm_ee_pos_w"][:-1]
    elif problem == "arm_nan":
        data["arm_ee_pos_w"][1, 0] = np.nan
    elif problem == "gripper_velocity":
        data["gripper_joint_vel"] = data["gripper_joint_vel"][:, :1]
    elif problem == "object_shape":
        data["object_pos_w"] = data["object_pos_w"][:, 0]
    else:
        data["object_quat_w"][1, 0] = 0
    bad = tmp_path / "bad_arm.npz"
    np.savez(bad, **data)
    with pytest.raises(ValueError, match="Motion file"):
        motion_loader(str(bad))
