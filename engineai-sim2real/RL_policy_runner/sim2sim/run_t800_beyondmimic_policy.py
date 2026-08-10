#!/usr/bin/env python3
"""Run the official EngineAI T800 BeyondMimic export in MuJoCo.

The observation and control contract follows EngineAI's BSD-3-Clause licensed
``engineai_robotics_native_sdk`` RL dance runner.  The MuJoCo robot model is
loaded from a local checkout of that SDK so its meshes and simulator model stay
on the official update path.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import yaml

ROOT = Path(__file__).resolve().parents[3]
T800_TASK = ROOT / "source/Gurukul/Gurukul/tasks/manager_based/beyondmimic/config/engineai_t800"
DEFAULT_POLICY = T800_TASK / "pretrained/policy.onnx"
DEFAULT_MOTION = T800_TASK / "motion/dance_t800.npz"
DEFAULT_DEPLOY_CONFIG = T800_TASK / "pretrained/deploy_config.yaml"
NATIVE_MODEL_RELATIVE = Path("assets/resource/t800.xml")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Headless or viewed MuJoCo smoke test for the official T800 dance policy."
    )
    parser.add_argument(
        "--native-sdk-root",
        type=Path,
        default=os.environ.get("ENGINEAI_NATIVE_SDK_ROOT"),
        help="Checkout of engineai-robotics/engineai_robotics_native_sdk.",
    )
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--deploy-config", type=Path, default=DEFAULT_DEPLOY_CONFIG)
    parser.add_argument(
        "--action-scale-source",
        choices=("native-sdk", "export"),
        default="native-sdk",
        help="Use the native SDK's safer zero elbow-yaw residuals or the RL export's 0.05 values.",
    )
    parser.add_argument(
        "--max-policy-steps",
        type=int,
        default=None,
        help="Stop early; the default runs all motion frames.",
    )
    parser.add_argument("--viewer", action="store_true", help="Open the passive MuJoCo viewer.")
    parser.add_argument("--real-time", action="store_true", help="Pace a headless run in wall time.")
    parser.add_argument(
        "--initialize-from-motion",
        dest="initialize_from_motion",
        action="store_true",
        default=True,
        help="Start at the first reference root pose and joints (recommended for policy-only smoke tests).",
    )
    parser.add_argument(
        "--no-initialize-from-motion",
        dest="initialize_from_motion",
        action="store_false",
        help="Use the native model's default floating-base pose instead.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    path = path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"{label} not found: {path}")
    return path


def quat_to_matrix(quat_wxyz: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat_wxyz, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm < 1.0e-12:
        raise ValueError("Cannot convert a zero quaternion.")
    w, x, y, z = quat / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def yaw_matrix(rotation: np.ndarray) -> np.ndarray:
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    c, s = np.cos(yaw), np.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def sensor(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
        raise KeyError(f"MuJoCo sensor is missing: {name}")
    address = model.sensor_adr[sensor_id]
    dimension = model.sensor_dim[sensor_id]
    return data.sensordata[address : address + dimension].copy()


def joint_addresses(model: mujoco.MjModel, names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    qpos_addresses: list[int] = []
    dof_addresses: list[int] = []
    for name in names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise KeyError(f"MuJoCo joint is missing: {name}")
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        dof_addresses.append(int(model.jnt_dofadr[joint_id]))
    return np.asarray(qpos_addresses), np.asarray(dof_addresses)


def actuator_addresses(model: mujoco.MjModel, names: list[str]) -> np.ndarray:
    addresses: list[int] = []
    for name in names:
        actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{name}")
        if actuator_id < 0:
            raise KeyError(f"MuJoCo actuator is missing: motor_{name}")
        addresses.append(actuator_id)
    return np.asarray(addresses)


def load_contract(config_path: Path, action_scale_source: str) -> dict[str, object]:
    with config_path.open(encoding="utf-8") as stream:
        contract = yaml.safe_load(stream)
    required = (
        "default_joint_pos",
        "joint_names",
        "joint_stiffness",
        "joint_damping",
        "observation_names",
        "observation_history_lengths",
        "action_scale",
    )
    missing = [key for key in required if key not in contract]
    if missing:
        raise KeyError(f"Deployment config is missing: {', '.join(missing)}")
    if contract["observation_names"] != [
        "command",
        "motion_anchor_ori_b",
        "base_ang_vel",
        "joint_pos",
        "joint_vel",
        "actions",
    ]:
        raise ValueError("The T800 policy observation order no longer matches this runner.")
    if contract["observation_history_lengths"] != [1, 1, 1, 1, 1, 1]:
        raise ValueError("This runner expects the exported one-frame observation history.")
    if action_scale_source == "native-sdk":
        contract["action_scale"][-2:] = [0.0, 0.0]
    return contract


def validate_motion(motion: np.lib.npyio.NpzFile, joint_names: list[str]) -> None:
    required = ("fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w")
    missing = [key for key in required if key not in motion.files]
    if missing:
        raise KeyError(f"Motion archive is missing: {', '.join(missing)}")
    if "joint_names" in motion.files:
        motion_names = [str(name) for name in motion["joint_names"]]
        if motion_names != joint_names:
            raise ValueError("Motion joint order does not match the deployment contract.")
    frames, joints = motion["joint_pos"].shape
    if joints != len(joint_names) or motion["joint_vel"].shape != (frames, joints):
        raise ValueError("Motion joint arrays do not match the T800 action count.")
    if motion["body_quat_w"].shape[0] != frames:
        raise ValueError("Motion body orientation length does not match joint data.")


def initialize_state(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    motion: np.lib.npyio.NpzFile,
    qpos_addresses: np.ndarray,
) -> None:
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    data.qpos[:3] = motion["body_pos_w"][0, 0]
    data.qpos[3:7] = motion["body_quat_w"][0, 0]
    data.qpos[qpos_addresses] = motion["joint_pos"][0]
    mujoco.mj_forward(model, data)


def run(args: argparse.Namespace) -> int:
    if args.native_sdk_root is None:
        raise ValueError("Pass --native-sdk-root or set ENGINEAI_NATIVE_SDK_ROOT to an official native SDK checkout.")
    model_path = require_file(Path(args.native_sdk_root) / NATIVE_MODEL_RELATIVE, "Official T800 MuJoCo model")
    policy_path = require_file(args.policy, "ONNX policy")
    motion_path = require_file(args.motion, "Motion archive")
    deploy_config_path = require_file(args.deploy_config, "Deployment config")

    contract = load_contract(deploy_config_path, args.action_scale_source)
    names = [str(name) for name in contract["joint_names"]]
    default_joint_pos = np.asarray(contract["default_joint_pos"], dtype=np.float64)
    stiffness = np.asarray(contract["joint_stiffness"], dtype=np.float64)
    damping = np.asarray(contract["joint_damping"], dtype=np.float64)
    action_scale = np.asarray(contract["action_scale"], dtype=np.float64)

    with np.load(motion_path) as motion:
        validate_motion(motion, names)
        joint_pos_ref = np.asarray(motion["joint_pos"], dtype=np.float32)
        joint_vel_ref = np.asarray(motion["joint_vel"], dtype=np.float32)
        body_pos_ref = np.asarray(motion["body_pos_w"], dtype=np.float64)
        body_quat_ref = np.asarray(motion["body_quat_w"], dtype=np.float64)
        fps = float(np.asarray(motion["fps"]).reshape(-1)[0])

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    qpos_addresses, dof_addresses = joint_addresses(model, names)
    control_addresses = actuator_addresses(model, names)
    if args.initialize_from_motion:
        state = {
            "body_pos_w": body_pos_ref,
            "body_quat_w": body_quat_ref,
            "joint_pos": joint_pos_ref,
        }
        initialize_state(model, data, state, qpos_addresses)
    else:
        data.qpos[qpos_addresses] = default_joint_pos
        mujoco.mj_forward(model, data)

    policy_dt = 1.0 / fps
    sim_steps_per_policy = round(policy_dt / model.opt.timestep)
    if not np.isclose(sim_steps_per_policy * model.opt.timestep, policy_dt):
        raise ValueError(f"Motion dt {policy_dt} is not divisible by MuJoCo dt {model.opt.timestep}.")

    session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    if input_meta.shape != [1, 134] or output_meta.shape != [1, 25]:
        raise ValueError(f"Unexpected ONNX shapes: input={input_meta.shape}, output={output_meta.shape}.")

    motion_frames = joint_pos_ref.shape[0]
    max_steps = motion_frames
    if args.max_policy_steps is not None:
        if args.max_policy_steps <= 0:
            raise ValueError("--max-policy-steps must be positive.")
        max_steps = min(max_steps, args.max_policy_steps)

    previous_action = np.zeros(len(names), dtype=np.float32)
    mujoco.mj_forward(model, data)
    body_initial_rotation = yaw_matrix(quat_to_matrix(sensor(model, data, "imu_quaternion")))
    ref_initial_rotation = yaw_matrix(quat_to_matrix(body_quat_ref[0, 0]))

    min_base_height = float("inf")
    max_abs_action = 0.0
    nonfinite = False
    start_time = time.monotonic()

    def policy_step(step: int) -> None:
        nonlocal previous_action, min_base_height, max_abs_action, nonfinite
        imu_rotation = quat_to_matrix(sensor(model, data, "imu_quaternion"))
        ref_rotation = quat_to_matrix(body_quat_ref[step, 0])
        ref_aligned = ref_initial_rotation.T @ ref_rotation
        body_aligned = body_initial_rotation.T @ imu_rotation
        orientation_error = body_aligned.T @ ref_aligned
        orientation_observation = orientation_error[:, :2].reshape(-1)

        observation = np.concatenate(
            (
                joint_pos_ref[step],
                joint_vel_ref[step],
                orientation_observation.astype(np.float32),
                sensor(model, data, "imu_angular_velocity").astype(np.float32),
                (data.qpos[qpos_addresses] - default_joint_pos).astype(np.float32),
                data.qvel[dof_addresses].astype(np.float32),
                previous_action,
            )
        ).astype(np.float32)
        if observation.shape != (134,):
            raise RuntimeError(f"Observation has shape {observation.shape}, expected (134,).")

        action = session.run([output_meta.name], {input_meta.name: observation[None]})[0][0]
        target = default_joint_pos + np.asarray(action, dtype=np.float64) * action_scale
        for _ in range(sim_steps_per_policy):
            position_error = target - data.qpos[qpos_addresses]
            torque = stiffness * position_error - damping * data.qvel[dof_addresses]
            data.ctrl[control_addresses] = torque
            mujoco.mj_step(model, data)

        min_base_height = min(min_base_height, float(data.qpos[2]))
        max_abs_action = max(max_abs_action, float(np.max(np.abs(action))))
        nonfinite |= not (
            np.all(np.isfinite(observation))
            and np.all(np.isfinite(action))
            and np.all(np.isfinite(data.qpos))
            and np.all(np.isfinite(data.qvel))
        )
        previous_action = np.asarray(action, dtype=np.float32)

    if args.viewer:
        from mujoco import viewer as mj_viewer

        with mj_viewer.launch_passive(model, data) as viewer:
            for step in range(max_steps):
                if not viewer.is_running():
                    break
                tick = time.monotonic()
                policy_step(step)
                viewer.sync()
                if args.real_time:
                    time.sleep(max(0.0, policy_dt - (time.monotonic() - tick)))
    else:
        for step in range(max_steps):
            tick = time.monotonic()
            policy_step(step)
            if args.real_time:
                time.sleep(max(0.0, policy_dt - (time.monotonic() - tick)))

    elapsed = time.monotonic() - start_time
    print(f"policy_steps: {max_steps}/{motion_frames}")
    print(f"simulated_seconds: {max_steps * policy_dt:.3f}")
    print(f"wall_seconds: {elapsed:.3f}")
    print(f"final_base_height_m: {float(data.qpos[2]):.4f}")
    print(f"minimum_base_height_m: {min_base_height:.4f}")
    print(f"maximum_absolute_action: {max_abs_action:.4f}")
    print(f"action_scale_source: {args.action_scale_source}")
    print(f"finite: {not nonfinite}")
    if nonfinite:
        return 2
    if min_base_height < 0.45:
        print("warning: base height fell below 0.45 m", file=sys.stderr)
        return 3
    return 0


def main() -> int:
    try:
        return run(parse_args())
    except (FileNotFoundError, KeyError, RuntimeError, ValueError) as error:
        print(f"error: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
