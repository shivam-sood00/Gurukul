#!/usr/bin/env python3
"""Run a 24-action PM01 BeyondMimic policy in MuJoCo."""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
import onnxruntime as ort
import yaml

ROOT = Path(__file__).resolve().parents[3]
TASK = (
    ROOT
    / "source/Gurukul/Gurukul/tasks/manager_based/beyondmimic/config"
    / "engineai_pm01_24dof"
)
DEFAULT_MODEL = (
    ROOT / "engineai-sim2real/engineai_mujoco/engineai_robots/pm01/scene.xml"
)
DEFAULT_POLICY = TASK / "pretrained/policy.onnx"
DEFAULT_MOTION = TASK / "motion/dance.npz"
DEFAULT_DEPLOY_CONFIG = TASK / "pretrained/deploy_config.yaml"
EXPECTED_OBSERVATIONS = [
    "command",
    "motion_anchor_ori_b",
    "base_ang_vel",
    "joint_pos",
    "joint_vel",
    "actions",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MuJoCo runner for 24-DoF PM01 BeyondMimic policies."
    )
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL)
    parser.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    parser.add_argument("--motion", type=Path, default=DEFAULT_MOTION)
    parser.add_argument("--deploy-config", type=Path, default=DEFAULT_DEPLOY_CONFIG)
    parser.add_argument(
        "--max-policy-steps",
        type=int,
        default=None,
        help="Stop early; by default all reference-motion frames are simulated.",
    )
    parser.add_argument("--viewer", action="store_true", help="Open the passive MuJoCo viewer.")
    parser.add_argument("--real-time", action="store_true", help="Pace simulation in wall time.")
    parser.add_argument(
        "--initialize-from-motion",
        dest="initialize_from_motion",
        action="store_true",
        default=True,
        help="Initialize the floating base and joints from the first reference frame.",
    )
    parser.add_argument(
        "--no-initialize-from-motion",
        dest="initialize_from_motion",
        action="store_false",
        help="Initialize from the MuJoCo home keyframe and deployment defaults.",
    )
    return parser.parse_args()


def require_file(path: Path, label: str) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved


def quaternion_matrix(quaternion_wxyz: np.ndarray) -> np.ndarray:
    quaternion = np.asarray(quaternion_wxyz, dtype=np.float64)
    norm = np.linalg.norm(quaternion)
    if norm < 1.0e-12:
        raise ValueError("Cannot convert a zero quaternion.")
    w, x, y, z = quaternion / norm
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def yaw_matrix(rotation: np.ndarray) -> np.ndarray:
    yaw = np.arctan2(rotation[1, 0], rotation[0, 0])
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return np.array([[cosine, -sine, 0.0], [sine, cosine, 0.0], [0.0, 0.0, 1.0]])


def sensor(model: mujoco.MjModel, data: mujoco.MjData, name: str) -> np.ndarray:
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    if sensor_id < 0:
        raise KeyError(f"MuJoCo sensor is missing: {name}")
    address = int(model.sensor_adr[sensor_id])
    dimension = int(model.sensor_dim[sensor_id])
    return data.sensordata[address : address + dimension].copy()


def resolve_joint_addresses(
    model: mujoco.MjModel, joint_names: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    qpos_addresses: list[int] = []
    dof_addresses: list[int] = []
    actuator_addresses: list[int] = []
    joint_lower_limits: list[float] = []
    joint_upper_limits: list[float] = []
    for name in joint_names:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        actuator_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"motor_{name}"
        )
        if joint_id < 0 or actuator_id < 0:
            raise KeyError(f"Official PM01 model is missing joint/actuator: {name}")
        qpos_addresses.append(int(model.jnt_qposadr[joint_id]))
        dof_addresses.append(int(model.jnt_dofadr[joint_id]))
        actuator_addresses.append(actuator_id)
        joint_lower_limits.append(float(model.jnt_range[joint_id, 0]))
        joint_upper_limits.append(float(model.jnt_range[joint_id, 1]))
    return (
        np.asarray(qpos_addresses),
        np.asarray(dof_addresses),
        np.asarray(actuator_addresses),
        np.asarray(joint_lower_limits),
        np.asarray(joint_upper_limits),
    )


def load_contract(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as stream:
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
    if contract["observation_names"] != EXPECTED_OBSERVATIONS:
        raise ValueError("PM01 observation order does not match the 24-DoF tracker.")
    if contract["observation_history_lengths"] != [1, 1, 1, 1, 1, 1]:
        raise ValueError("PM01 24-DoF export must use one-frame observation histories.")
    lengths = {
        len(contract[key])
        for key in (
            "default_joint_pos",
            "joint_names",
            "joint_stiffness",
            "joint_damping",
            "action_scale",
        )
    }
    if lengths != {24}:
        raise ValueError(f"PM01 deployment vectors must all have 24 entries, got {lengths}.")
    return contract


def load_motion(
    path: Path, joint_names: list[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
    with np.load(path, allow_pickle=False) as motion:
        required = ("fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w")
        missing = [key for key in required if key not in motion.files]
        if missing:
            raise KeyError(f"Motion archive is missing: {', '.join(missing)}")
        if "joint_names" in motion.files:
            names = [str(name) for name in motion["joint_names"]]
            if names != joint_names:
                raise ValueError("Motion and deployment joint orders differ.")
        joint_pos = np.asarray(motion["joint_pos"], dtype=np.float32)
        joint_vel = np.asarray(motion["joint_vel"], dtype=np.float32)
        body_pos = np.asarray(motion["body_pos_w"], dtype=np.float64)
        body_quat = np.asarray(motion["body_quat_w"], dtype=np.float64)
        fps = float(np.asarray(motion["fps"]).reshape(-1)[0])
    frames = joint_pos.shape[0]
    if joint_pos.shape != (frames, 24) or joint_vel.shape != (frames, 24):
        raise ValueError("PM01 motion must contain 24 joint positions and velocities.")
    if body_pos.shape[0] != frames or body_quat.shape[0] != frames:
        raise ValueError("PM01 body and joint motion lengths differ.")
    return joint_pos, joint_vel, body_pos, body_quat, fps


def run(args: argparse.Namespace) -> int:
    model_path = require_file(args.model, "Official PM01 MuJoCo scene")
    policy_path = require_file(args.policy, "PM01 ONNX policy")
    motion_path = require_file(args.motion, "PM01 motion")
    config_path = require_file(args.deploy_config, "PM01 deployment config")

    contract = load_contract(config_path)
    names = [str(name) for name in contract["joint_names"]]
    default_joint_pos = np.asarray(contract["default_joint_pos"], dtype=np.float64)
    stiffness = np.asarray(contract["joint_stiffness"], dtype=np.float64)
    damping = np.asarray(contract["joint_damping"], dtype=np.float64)
    action_scale = np.asarray(contract["action_scale"], dtype=np.float64)
    joint_pos_ref, joint_vel_ref, body_pos_ref, body_quat_ref, fps = load_motion(
        motion_path, names
    )

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    (
        qpos_addresses,
        dof_addresses,
        control_addresses,
        joint_lower_limits,
        joint_upper_limits,
    ) = resolve_joint_addresses(model, names)
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0.0
    if args.initialize_from_motion:
        data.qpos[:3] = body_pos_ref[0, 0]
        data.qpos[3:7] = body_quat_ref[0, 0]
        data.qpos[qpos_addresses] = joint_pos_ref[0]
    else:
        data.qpos[qpos_addresses] = default_joint_pos
    mujoco.mj_forward(model, data)

    policy_dt = 1.0 / fps
    sim_steps_per_policy = round(policy_dt / model.opt.timestep)
    if not np.isclose(sim_steps_per_policy * model.opt.timestep, policy_dt):
        raise ValueError("Motion frequency is not divisible by the PM01 MuJoCo timestep.")

    session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    output_meta = session.get_outputs()[0]
    if input_meta.shape != [1, 129] or output_meta.shape != [1, 24]:
        raise ValueError(
            f"Unexpected PM01 ONNX shapes: input={input_meta.shape}, output={output_meta.shape}."
        )

    frames = joint_pos_ref.shape[0]
    policy_steps = frames
    if args.max_policy_steps is not None:
        if args.max_policy_steps <= 0:
            raise ValueError("--max-policy-steps must be positive.")
        policy_steps = min(policy_steps, args.max_policy_steps)

    previous_action = np.zeros(24, dtype=np.float32)
    body_initial_yaw = yaw_matrix(
        quaternion_matrix(sensor(model, data, "imu_quaternion"))
    )
    reference_initial_yaw = yaw_matrix(quaternion_matrix(body_quat_ref[0, 0]))
    minimum_base_height = float("inf")
    maximum_absolute_action = 0.0
    nonfinite = False
    start = time.monotonic()

    def policy_step(step: int) -> None:
        nonlocal previous_action, minimum_base_height
        nonlocal maximum_absolute_action, nonfinite
        body_rotation = quaternion_matrix(sensor(model, data, "imu_quaternion"))
        reference_rotation = quaternion_matrix(body_quat_ref[step, 0])
        reference_aligned = reference_initial_yaw.T @ reference_rotation
        body_aligned = body_initial_yaw.T @ body_rotation
        orientation = (body_aligned.T @ reference_aligned)[:, :2].reshape(-1)

        observation = np.concatenate(
            (
                joint_pos_ref[step],
                joint_vel_ref[step],
                orientation.astype(np.float32),
                sensor(model, data, "imu_angular_velocity").astype(np.float32),
                (
                    np.clip(
                        data.qpos[qpos_addresses],
                        joint_lower_limits,
                        joint_upper_limits,
                    )
                    - default_joint_pos
                ).astype(np.float32),
                data.qvel[dof_addresses].astype(np.float32),
                previous_action,
            )
        ).astype(np.float32)
        if observation.shape != (129,):
            raise RuntimeError(f"Observation has shape {observation.shape}, expected (129,).")

        action = session.run(
            [output_meta.name], {input_meta.name: observation[None]}
        )[0][0]
        # EngineAI's native PM01 rl_dance_example sets resident_control: false.
        target = default_joint_pos + np.asarray(action, dtype=np.float64) * action_scale
        for _ in range(sim_steps_per_policy):
            torque = (
                stiffness * (target - data.qpos[qpos_addresses])
                - damping * data.qvel[dof_addresses]
            )
            data.ctrl[control_addresses] = torque
            mujoco.mj_step(model, data)

        minimum_base_height = min(minimum_base_height, float(data.qpos[2]))
        maximum_absolute_action = max(
            maximum_absolute_action, float(np.max(np.abs(action)))
        )
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
            for step in range(policy_steps):
                if not viewer.is_running():
                    break
                tick = time.monotonic()
                policy_step(step)
                viewer.sync()
                if args.real_time:
                    time.sleep(max(0.0, policy_dt - (time.monotonic() - tick)))
    else:
        for step in range(policy_steps):
            tick = time.monotonic()
            policy_step(step)
            if args.real_time:
                time.sleep(max(0.0, policy_dt - (time.monotonic() - tick)))

    print(f"policy_steps: {policy_steps}/{frames}")
    print(f"simulated_seconds: {policy_steps * policy_dt:.3f}")
    print(f"wall_seconds: {time.monotonic() - start:.3f}")
    print(f"final_base_height_m: {float(data.qpos[2]):.4f}")
    print(f"minimum_base_height_m: {minimum_base_height:.4f}")
    print(f"maximum_absolute_action: {maximum_absolute_action:.4f}")
    print(f"finite: {not nonfinite}")
    if nonfinite:
        return 2
    if minimum_base_height < 0.45:
        print("warning: PM01 base height fell below 0.45 m", file=sys.stderr)
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
