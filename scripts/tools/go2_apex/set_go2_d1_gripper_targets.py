#!/usr/bin/env python3
"""Replace Go2+D1 wrist targets with frame-aligned gripper-midpoint targets."""

from __future__ import annotations

import argparse
import os
import tempfile
from pathlib import Path

import numpy as np
from yourdfpy import URDF


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_URDF = REPO_ROOT / "source/Gurukul/data/Robots/unitree/go2_with_d1/urdf/go2_d1_vis.urdf"
GRIPPER_LINK_NAMES = ("Empty_Link_L", "Empty_Link_R")
GRIPPER_JOINT_NAMES = ("arm_7_1_joint", "arm_7_2_joint")
GRIPPER_FRAME_METADATA = "gripper_midpoint:Link7_1,Link7_2"


def _quat_rotate_wxyz(quaternion: np.ndarray, vector: np.ndarray) -> np.ndarray:
    xyz = quaternion[1:]
    twice_cross = 2.0 * np.cross(xyz, vector)
    return vector + quaternion[0] * twice_cross + np.cross(xyz, twice_cross)


def rebuild_gripper_targets(input_path: Path, urdf_path: Path) -> None:
    with np.load(input_path) as archive:
        arrays = {key: np.asarray(archive[key]) for key in archive.files}

    required = ("joint_names", "joint_pos", "body_pos_w", "body_quat_w", "arm_ee_pos_w")
    missing = [key for key in required if key not in arrays]
    if missing:
        raise ValueError(f"{input_path}: missing required arrays: {missing}")

    joint_names = [str(name) for name in arrays["joint_names"].tolist()]
    joint_pos = np.asarray(arrays["joint_pos"], dtype=np.float64)
    base_pos_w = np.asarray(arrays["body_pos_w"][:, 0], dtype=np.float64)
    base_quat_w = np.asarray(arrays["body_quat_w"][:, 0], dtype=np.float64)
    if joint_pos.shape[0] != base_pos_w.shape[0]:
        raise ValueError(f"{input_path}: joint and body frame counts do not match")

    robot = URDF.load(urdf_path, build_scene_graph=True, load_meshes=False)
    gripper_pos_w = np.empty((joint_pos.shape[0], 3), dtype=np.float32)
    for frame, positions in enumerate(joint_pos):
        configuration = dict(zip(joint_names, positions, strict=True))
        configuration.update({name: 0.0 for name in GRIPPER_JOINT_NAMES})
        robot.update_cfg(configuration)
        gripper_pos_b = np.mean(
            [robot.get_transform(link_name)[:3, 3] for link_name in GRIPPER_LINK_NAMES],
            axis=0,
        )
        gripper_pos_w[frame] = base_pos_w[frame] + _quat_rotate_wxyz(base_quat_w[frame], gripper_pos_b)

    arrays["arm_ee_pos_w"] = gripper_pos_w
    arrays["arm_ee_frame"] = np.asarray(GRIPPER_FRAME_METADATA, dtype=np.str_)

    with tempfile.NamedTemporaryFile(
        prefix=f".{input_path.stem}.",
        suffix=".npz",
        dir=input_path.parent,
        delete=False,
    ) as temporary:
        temporary_path = Path(temporary.name)
    try:
        np.savez_compressed(temporary_path, **arrays)
        os.replace(temporary_path, input_path)
    finally:
        temporary_path.unlink(missing_ok=True)

    print(f"Updated {input_path}: {GRIPPER_FRAME_METADATA}, frames={joint_pos.shape[0]}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, nargs="+", required=True, help="Motion NPZ files to update in place.")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="Go2+D1 URDF used for reference FK.")
    args = parser.parse_args()

    for input_path in args.input:
        rebuild_gripper_targets(input_path.resolve(), args.urdf.resolve())


if __name__ == "__main__":
    main()
