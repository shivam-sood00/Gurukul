#!/usr/bin/env python3
"""Retarget the 23-joint PM01 walking archive to EngineAI's official 24-DoF model."""

from __future__ import annotations

import argparse
from pathlib import Path

import mujoco
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_INPUT = (
    REPO_ROOT
    / "source/Gurukul/Gurukul/tasks/manager_based/beyondmimic/config"
    / "engineai_pm01/motion/locomotion.npz"
)
DEFAULT_OUTPUT = (
    REPO_ROOT
    / "source/Gurukul/Gurukul/tasks/manager_based/beyondmimic/config"
    / "engineai_pm01_24dof/motion/walking_24dof.npz"
)
DEFAULT_MODEL = (
    REPO_ROOT / "engineai-sim2real/engineai_mujoco/engineai_robots/pm01/scene.xml"
)

TARGET_JOINT_NAMES = [
    "J00_HIP_PITCH_L",
    "J06_HIP_PITCH_R",
    "J12_WAIST_YAW",
    "J01_HIP_ROLL_L",
    "J07_HIP_ROLL_R",
    "J13_SHOULDER_PITCH_L",
    "J18_SHOULDER_PITCH_R",
    "J23_HEAD_YAW",
    "J02_HIP_YAW_L",
    "J08_HIP_YAW_R",
    "J14_SHOULDER_ROLL_L",
    "J19_SHOULDER_ROLL_R",
    "J03_KNEE_PITCH_L",
    "J09_KNEE_PITCH_R",
    "J15_SHOULDER_YAW_L",
    "J20_SHOULDER_YAW_R",
    "J04_ANKLE_PITCH_L",
    "J10_ANKLE_PITCH_R",
    "J16_ELBOW_PITCH_L",
    "J21_ELBOW_PITCH_R",
    "J05_ANKLE_ROLL_L",
    "J11_ANKLE_ROLL_R",
    "J17_ELBOW_YAW_L",
    "J22_ELBOW_YAW_R",
]
SOURCE_BODY_NAMES = [
    "LINK_BASE",
    "LINK_HIP_PITCH_L",
    "LINK_HIP_PITCH_R",
    "LINK_TORSO_YAW",
    "LINK_HIP_ROLL_L",
    "LINK_HIP_ROLL_R",
    "LINK_SHOULDER_PITCH_L",
    "LINK_SHOULDER_PITCH_R",
    "LINK_HIP_YAW_L",
    "LINK_HIP_YAW_R",
    "LINK_SHOULDER_ROLL_L",
    "LINK_SHOULDER_ROLL_R",
    "LINK_KNEE_PITCH_L",
    "LINK_KNEE_PITCH_R",
    "LINK_SHOULDER_YAW_L",
    "LINK_SHOULDER_YAW_R",
    "LINK_ANKLE_PITCH_L",
    "LINK_ANKLE_PITCH_R",
    "LINK_ELBOW_PITCH_L",
    "LINK_ELBOW_PITCH_R",
    "LINK_ANKLE_ROLL_L",
    "LINK_ANKLE_ROLL_R",
    "LINK_ELBOW_YAW_L",
    "LINK_ELBOW_YAW_R",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument(
        "--output-fps",
        type=int,
        default=50,
        help="Target motion frequency; must evenly divide the source frequency.",
    )
    parser.add_argument(
        "--head-yaw",
        type=float,
        default=0.0,
        help="Constant J23_HEAD_YAW reference in radians.",
    )
    return parser.parse_args()


def quaternion_conjugate(quaternion: np.ndarray) -> np.ndarray:
    result = quaternion.copy()
    result[..., 1:] *= -1.0
    return result


def quaternion_multiply(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    lw, lx, ly, lz = np.moveaxis(left, -1, 0)
    rw, rx, ry, rz = np.moveaxis(right, -1, 0)
    return np.stack(
        (
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ),
        axis=-1,
    )


def angular_velocity_world(quaternions: np.ndarray, dt: float) -> np.ndarray:
    quaternions = np.asarray(quaternions, dtype=np.float64).copy()
    for frame in range(1, quaternions.shape[0]):
        flip = np.sum(quaternions[frame - 1] * quaternions[frame], axis=-1) < 0.0
        quaternions[frame, flip] *= -1.0

    relative = quaternion_multiply(
        quaternions[2:], quaternion_conjugate(quaternions[:-2])
    )
    negative = relative[..., 0] < 0.0
    relative[negative] *= -1.0
    vector = relative[..., 1:]
    vector_norm = np.linalg.norm(vector, axis=-1)
    angle = 2.0 * np.arctan2(vector_norm, np.clip(relative[..., 0], 0.0, 1.0))
    axis = np.divide(
        vector,
        vector_norm[..., None],
        out=np.zeros_like(vector),
        where=vector_norm[..., None] > 1.0e-12,
    )
    middle = axis * (angle / (2.0 * dt))[..., None]
    return np.concatenate((middle[:1], middle, middle[-1:]), axis=0).astype(
        np.float32
    )


def main() -> int:
    args = parse_args()
    input_path = args.input.expanduser().resolve()
    output_path = args.output.expanduser().resolve()
    model_path = args.model.expanduser().resolve()
    if not input_path.is_file() or not model_path.is_file():
        raise FileNotFoundError(f"Missing input/model: {input_path}, {model_path}")

    with np.load(input_path, allow_pickle=False) as source:
        source_fps = int(np.asarray(source["fps"]).reshape(-1)[0])
        source_joint_pos = np.asarray(source["joint_pos"], dtype=np.float64)
        source_joint_vel = np.asarray(source["joint_vel"], dtype=np.float32)
        source_body_pos = np.asarray(source["body_pos_w"], dtype=np.float64)
        source_body_quat = np.asarray(source["body_quat_w"], dtype=np.float64)
    if args.output_fps <= 0 or source_fps % args.output_fps:
        raise ValueError(
            f"--output-fps must evenly divide source fps {source_fps}, got {args.output_fps}."
        )
    stride = source_fps // args.output_fps
    source_joint_pos = source_joint_pos[::stride]
    source_joint_vel = source_joint_vel[::stride]
    source_body_pos = source_body_pos[::stride]
    source_body_quat = source_body_quat[::stride]
    fps = args.output_fps
    frames = source_joint_pos.shape[0]
    if source_joint_pos.shape != (frames, 23):
        raise ValueError(f"Expected source joint_pos shape ({frames}, 23).")
    if source_body_pos.shape != (frames, 24, 3):
        raise ValueError(f"Expected source body_pos_w shape ({frames}, 24, 3).")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        for name in TARGET_JOINT_NAMES
    ]
    if any(joint_id < 0 for joint_id in joint_ids):
        raise KeyError("The official PM01 model does not contain all 24 joints.")
    qpos_addresses = model.jnt_qposadr[np.asarray(joint_ids)]
    body_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id)
        for body_id in range(1, model.nbody)
    ]
    body_ids = np.arange(1, model.nbody)

    head_index = TARGET_JOINT_NAMES.index("J23_HEAD_YAW")
    joint_pos = np.insert(
        source_joint_pos.astype(np.float32),
        head_index,
        np.float32(args.head_yaw),
        axis=1,
    )
    joint_vel = np.insert(
        source_joint_vel,
        head_index,
        np.float32(0.0),
        axis=1,
    )

    body_pos = np.empty((frames, len(body_ids), 3), dtype=np.float32)
    body_quat = np.empty((frames, len(body_ids), 4), dtype=np.float32)
    for frame in range(frames):
        data.qpos[:] = model.qpos0
        data.qpos[:3] = source_body_pos[frame, 0]
        data.qpos[3:7] = source_body_quat[frame, 0]
        data.qpos[qpos_addresses] = joint_pos[frame]
        mujoco.mj_forward(model, data)
        body_pos[frame] = data.xpos[body_ids]
        body_quat[frame] = data.xquat[body_ids]

    dt = 1.0 / fps
    body_lin_vel = np.gradient(body_pos, dt, axis=0).astype(np.float32)
    body_ang_vel = angular_velocity_world(body_quat, dt)

    source_indexes = {name: index for index, name in enumerate(SOURCE_BODY_NAMES)}
    target_indexes = {name: index for index, name in enumerate(body_names)}
    common_position_errors = np.stack(
        [
            np.linalg.norm(
                body_pos[:, target_indexes[name]] - source_body_pos[:, source_index],
                axis=-1,
            )
            for name, source_index in source_indexes.items()
        ],
        axis=-1,
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        fps=np.asarray([fps], dtype=np.int64),
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        body_pos_w=body_pos,
        body_quat_w=body_quat,
        body_lin_vel_w=body_lin_vel,
        body_ang_vel_w=body_ang_vel,
        joint_names=np.asarray(TARGET_JOINT_NAMES),
        body_names=np.asarray(body_names),
    )
    print(f"saved: {output_path}")
    print(f"frames/fps/duration: {frames}/{fps}/{frames / fps:.3f}s")
    print(f"joint/body shapes: {joint_pos.shape}/{body_pos.shape}")
    print(
        f"head yaw index/range: {head_index}/"
        f"{joint_pos[:, head_index].min():.6f}..{joint_pos[:, head_index].max():.6f}"
    )
    print(
        "official-FK common-body displacement from legacy archive: "
        f"mean={common_position_errors.mean():.6f}m, max={common_position_errors.max():.6f}m"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
