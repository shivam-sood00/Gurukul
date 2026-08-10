"""Go2+D1 direct APEX flat tracker on Unitree hardware.

Observation / action contract matches
`go2_d1_arm_apex_flat_tracker_v0.yaml` and MuJoCo
`RL_policy_runner/sim2sim/run_rl_policy.py`:

  206-D actor obs =
    ang_vel(3)*0.25 + projected_gravity(3) + command(4)
    + joint_pos_rel(19) + go2_joint_vel(12)*0.05 + prev_action(19)
    + skill(1) + reference_features(5 * 29)

  19 position targets at 50 Hz (no DecAP, no reference-residual):
    q_target = default_angle + clip(action, ±6) * per_joint_scale

Hardware backends (--arm-backend):
  d1-dds   Real robot: 12 Go2 legs on unitree_sdk2py rt/lowcmd.
           Six D1 rotary joints + gripper on rt/arm_Command (degrees).
  unified  MuJoCo / combined lowstate: all 19 motors on rt/lowcmd
           (gripper target is the simulated jaw travel in metres).
"""

from __future__ import annotations

import argparse
import glob
import json
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import onnxruntime as ort

SCRIPT_DIR = Path(__file__).resolve().parent


def _discover_gurukul_root() -> Path:
    env = os.environ.get("GURUKUL_ROOT")
    if env:
        return Path(env).expanduser().resolve()
    for parent in SCRIPT_DIR.parents:
        if (parent / "pyproject.toml").is_file() and (parent / "source/Gurukul").is_dir():
            return parent
    raise RuntimeError("Could not locate the Gurukul repository. Set GURUKUL_ROOT to its path.")


def _discover_d1_python_dir() -> Path:
    env = os.environ.get("D1_PYTHON_DIR")
    if env:
        return Path(env).expanduser().resolve()
    candidates = (
        SCRIPT_DIR.parent / "python",
        SCRIPT_DIR.parents[1] / "UnitreeGo2_D1_Deployment" / "python",
        SCRIPT_DIR.parents[2] / "UnitreeGo2_D1_Deployment" / "python",
    )
    for candidate in candidates:
        if (candidate / "d1_idl.py").is_file():
            return candidate.resolve()
    return candidates[0].resolve()


GURUKUL_ROOT = _discover_gurukul_root()
D1_PYTHON_DIR = _discover_d1_python_dir()
PHYSICAL_ROOT = D1_PYTHON_DIR.parents[1] if D1_PYTHON_DIR.name == "python" else SCRIPT_DIR.parents[1]

# Bundled next to this script (override with --policy-path / --motion-file).
DEFAULT_POLICY_PATH = SCRIPT_DIR / "policy.onnx"
DEFAULT_MOTION_PATH = SCRIPT_DIR / "walk_then_wave_hello.npz"
DEFAULT_CONFIG_PATH = (
    GURUKUL_ROOT
    / "unitree-sim2real/RL_policy_runner/configs/Gurukul"
    / "go2_d1_arm_apex_flat_tracker_v0.yaml"
)

DEFAULT_NETWORK_INTERFACE = os.environ.get("GO2D1_NETWORK_INTERFACE", "enp4s0")
SIM_NETWORK_INTERFACE = "lo"

NUM_LEG_ACTIONS = 12
NUM_ARM_JOINTS = 6
NUM_ACTIONS = 19
NUM_MOTORS = 19
ARM_MOTOR_START = NUM_LEG_ACTIONS
GRIPPER_INDEX = NUM_ACTIONS - 1

REFERENCE_OFFSETS = (0, 1, 2, 5, 10)
ACTOR_JOINT_VELOCITY_INDICES = tuple(range(NUM_LEG_ACTIONS))
TRACKER_BASE_OBS_DIM = 10 + NUM_ACTIONS + len(ACTOR_JOINT_VELOCITY_INDICES) + NUM_ACTIONS  # 60
SKILL_DIM = 1
REFERENCE_FRAME_DIM = NUM_ACTIONS + 10  # 29
REFERENCE_DIM = REFERENCE_FRAME_DIM * len(REFERENCE_OFFSETS)
EXPECTED_OBS_DIM = TRACKER_BASE_OBS_DIM + SKILL_DIM + REFERENCE_DIM  # 206

POLICY_WARMUP_RUNS = 5
POLICY_DT = 0.02  # 50 Hz outer policy rate
D1_ARM_COMMAND_DT = 0.1  # D1 seven-angle position packet: 10 Hz
DEFAULT_SITDOWN_DURATION_S = 5.0

# Standard Go2 lying / sit-down pose used by Unitree low-level examples.
GO2_SIT_DOWN_LEG_POS = np.array(
    [
        -0.0473455, 1.22187, -2.44375,
         0.0473455, 1.22187, -2.44375,
        -0.0473455, 1.22187, -2.44375,
         0.0473455, 1.22187, -2.44375,
    ],
    dtype=np.float32,
)

GO2_D1_TRACKER_JOINT_NAMES = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "arm_1_joint",
    "arm_2_joint",
    "arm_3_joint",
    "arm_4_joint",
    "arm_5_joint",
    "arm_6_joint",
    "arm_7_1_joint",
)

# Measured D1 servo endpoints used by Isaac / MuJoCo APEX (metres <-> degrees).
GO2_D1_GRIPPER_OPEN_HARDWARE_ANGLE_DEG = 68.6
GO2_D1_GRIPPER_CLOSED_HARDWARE_ANGLE_DEG = -28.7
GO2_D1_GRIPPER_MAX_JAW_TRAVEL_M = 0.033

# URDF joint limits (rad) + gripper jaw travel (m).
ARM_LOWER_LIMITS = np.array(
    [-2.36, -1.57, -1.57, -2.36, -1.57, -2.36, 0.0], dtype=np.float32
)
ARM_UPPER_LIMITS = np.array(
    [2.36, 1.57, 1.57, 2.36, 1.57, 2.36, GO2_D1_GRIPPER_MAX_JAW_TRAVEL_M],
    dtype=np.float32,
)
ARM_VELOCITY_LIMITS = np.array(
    [1.05, 1.05, 1.05, 1.73, 1.73, 1.73, 0.05], dtype=np.float32
)

_SHUTDOWN_REQUESTED = False

ChannelFactoryInitialize = None
ChannelPublisher = None
ChannelSubscriber = None
MotionSwitcherClient = None
SportClient = None
unitree_go_msg_dds__LowCmd_ = None
LowCmd_ = None
LowState_ = None
WirelessController_ = None
CRC = None


def _sigint_handler(_signum, _frame) -> None:
    global _SHUTDOWN_REQUESTED
    if _SHUTDOWN_REQUESTED:
        print("\n[FAIL] Second Ctrl+C - force exit.", flush=True)
        os._exit(130)
    _SHUTDOWN_REQUESTED = True
    print("\n[INFO] Stop requested (Ctrl+C). Shutting down...", flush=True)


def gripper_hardware_angle_to_joint_position(angle_deg: float) -> float:
    closure = (
        GO2_D1_GRIPPER_OPEN_HARDWARE_ANGLE_DEG - float(angle_deg)
    ) / (
        GO2_D1_GRIPPER_OPEN_HARDWARE_ANGLE_DEG
        - GO2_D1_GRIPPER_CLOSED_HARDWARE_ANGLE_DEG
    )
    return float(min(max(closure, 0.0), 1.0) * GO2_D1_GRIPPER_MAX_JAW_TRAVEL_M)


def gripper_joint_position_to_hardware_angle(joint_position_m: float) -> float:
    closure = min(
        max(float(joint_position_m) / GO2_D1_GRIPPER_MAX_JAW_TRAVEL_M, 0.0),
        1.0,
    )
    return float(
        GO2_D1_GRIPPER_OPEN_HARDWARE_ANGLE_DEG
        - closure
        * (
            GO2_D1_GRIPPER_OPEN_HARDWARE_ANGLE_DEG
            - GO2_D1_GRIPPER_CLOSED_HARDWARE_ANGLE_DEG
        )
    )


def quat_to_rot(quat) -> np.ndarray:
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _import_unitree_sdk() -> None:
    global ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    global MotionSwitcherClient, SportClient, unitree_go_msg_dds__LowCmd_
    global LowCmd_, LowState_, WirelessController_, CRC

    if ChannelFactoryInitialize is not None:
        return
    try:
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import (
            MotionSwitcherClient as _MotionSwitcherClient,
        )
        from unitree_sdk2py.core.channel import ChannelFactoryInitialize as _ChannelFactoryInitialize
        from unitree_sdk2py.core.channel import ChannelPublisher as _ChannelPublisher
        from unitree_sdk2py.core.channel import ChannelSubscriber as _ChannelSubscriber
        from unitree_sdk2py.go2.sport.sport_client import SportClient as _SportClient
        from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_ as _LowCmdDefault
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_ as _LowCmd
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_ as _LowState
        from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_ as _WirelessController
        from unitree_sdk2py.utils.crc import CRC as _CRC
    except ImportError as exc:
        raise ImportError(
            "unitree_sdk2py is required for Go2+D1 hardware control. "
            "Install/source the Unitree SDK environment, then rerun this script."
        ) from exc

    ChannelFactoryInitialize = _ChannelFactoryInitialize
    ChannelPublisher = _ChannelPublisher
    ChannelSubscriber = _ChannelSubscriber
    MotionSwitcherClient = _MotionSwitcherClient
    SportClient = _SportClient
    unitree_go_msg_dds__LowCmd_ = _LowCmdDefault
    LowCmd_ = _LowCmd
    LowState_ = _LowState
    WirelessController_ = _WirelessController
    CRC = _CRC


def _resolve_network_interface(interface: str | None, *, arm_backend: str) -> str:
    if interface is not None:
        return interface
    return SIM_NETWORK_INTERFACE if arm_backend == "unified" else DEFAULT_NETWORK_INTERFACE


def init_unitree_channel(interface: str | None = None, *, arm_backend: str = "d1-dds") -> str:
    _import_unitree_sdk()
    iface = _resolve_network_interface(interface, arm_backend=arm_backend)
    if iface == SIM_NETWORK_INTERFACE:
        ChannelFactoryInitialize(1, SIM_NETWORK_INTERFACE)
    else:
        ChannelFactoryInitialize(0, iface)
    return iface


def _resolve_path(path_str: str | Path) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = (SCRIPT_DIR / path).resolve()
    return path


def _resolve_motion_files(patterns) -> list[Path]:
    if isinstance(patterns, (str, os.PathLike)):
        patterns = [patterns]
    paths: list[Path] = []
    for pattern in patterns:
        pattern_path = Path(pattern).expanduser()
        if not pattern_path.is_absolute():
            for root in (SCRIPT_DIR, GURUKUL_ROOT, PHYSICAL_ROOT):
                candidate = root / pattern_path
                matches = sorted(glob.glob(str(candidate), recursive=True))
                if matches:
                    pattern_path = candidate
                    break
            else:
                pattern_path = SCRIPT_DIR / pattern_path
        matches = sorted(glob.glob(str(pattern_path), recursive=True))
        for match in matches:
            path = Path(match)
            if path.is_dir():
                paths.extend(sorted(path.rglob("*.npz")))
            elif path.is_file():
                paths.append(path)
    return list(dict.fromkeys(path.resolve() for path in paths))


def _onnx_input_width(session: "ort.InferenceSession") -> int | None:
    shape = session.get_inputs()[0].shape
    dims = [dim for dim in shape if isinstance(dim, int) and dim > 0]
    return int(dims[-1]) if dims else None


def _feed_dict(session: "ort.InferenceSession", obs: np.ndarray) -> dict[str, np.ndarray]:
    meta = session.get_inputs()[0]
    values = np.asarray(obs, dtype=np.float32).reshape(-1)
    if len(meta.shape) == 2:
        values = values.reshape(1, -1)
    elif len(meta.shape) == 3:
        values = values.reshape(1, 1, -1)
    return {meta.name: values}


def _import_d1_dds_types():
    d1_dir = _discover_d1_python_dir()
    if str(d1_dir) not in sys.path:
        sys.path.insert(0, str(d1_dir))
    try:
        from d1_idl import ArmString_
    except ImportError as exc:
        raise RuntimeError(
            "D1 DDS types unavailable. Set D1_PYTHON_DIR to "
            "UnitreeGo2_D1_Deployment/python and run in the Unitree SDK environment."
        ) from exc
    return ArmString_


class D1ArmDDSClient:
    """Non-owning D1 DDS adapter (assumes ChannelFactory already initialized)."""

    CMD_TOPIC = "rt/arm_Command"
    FEEDBACK_TOPICS = ("rt/arm_Feedback", "arm_Feedback")
    STREAM_MODE = 0

    def __init__(self) -> None:
        ArmString_ = _import_d1_dds_types()
        self._ArmString = ArmString_
        self._publisher = ChannelPublisher(self.CMD_TOPIC, ArmString_)
        self._publisher.Init()
        self._subscribers = []
        self._seq = 1000
        self.last_angles_deg: np.ndarray | None = None
        self.enable_status: int | None = None
        self.power_status: int | None = None
        self.error_status: int | None = None
        for topic in self.FEEDBACK_TOPICS:
            subscriber = ChannelSubscriber(topic, ArmString_)
            subscriber.Init(self._on_feedback, 10)
            self._subscribers.append(subscriber)
        print("[D1] command + feedback channels ready")

    def _on_feedback(self, message) -> None:
        try:
            payload = json.loads(message.data)
        except (AttributeError, TypeError, ValueError, json.JSONDecodeError):
            return
        data = payload.get("data")
        if not isinstance(data, dict):
            return
        if all(f"angle{i}" in data for i in range(7)):
            self.last_angles_deg = np.array(
                [float(data[f"angle{i}"]) for i in range(7)],
                dtype=np.float64,
            )
        if "enable_status" in data:
            self.enable_status = int(data["enable_status"])
        if "power_status" in data:
            self.power_status = int(data["power_status"])
        if "error_status" in data:
            self.error_status = int(data["error_status"])

    def _publish(self, funcode: int, data: dict, *, quiet: bool = True) -> None:
        self._seq += 1
        payload = {
            "seq": self._seq,
            "address": 1,
            "funcode": int(funcode),
            "data": data,
        }
        message = self._ArmString(data=json.dumps(payload, separators=(",", ":")))
        ok = self._publisher.Write(message, 0.05)
        if not ok and not quiet:
            print(f"[D1] WARNING: publish timed out: {message.data}")

    def enable(self, on: bool = True) -> None:
        self._publish(5, {"mode": 0 if on else 1}, quiet=False)

    def command_policy_targets(self, arm_rad: np.ndarray, gripper_m: float) -> None:
        arm = np.asarray(arm_rad, dtype=np.float64).reshape(-1)
        if arm.size != NUM_ARM_JOINTS or not np.isfinite(arm).all():
            raise ValueError("D1 arm command must contain six finite radian targets.")
        gripper_deg = gripper_joint_position_to_hardware_angle(gripper_m)
        values_deg = np.concatenate((np.degrees(arm), [gripper_deg]))
        data = {"mode": int(self.STREAM_MODE)}
        for index, angle in enumerate(values_deg):
            data[f"angle{index}"] = round(float(angle), 1)
        self._publish(2, data)

    def measured_policy_state(self) -> tuple[np.ndarray, float] | None:
        if self.last_angles_deg is None:
            return None
        arm_rad = np.radians(self.last_angles_deg[:NUM_ARM_JOINTS]).astype(np.float32)
        gripper_m = np.float32(gripper_hardware_angle_to_joint_position(self.last_angles_deg[6]))
        return arm_rad, float(gripper_m)

    def wait_for_feedback(self, timeout_s: float = 5.0) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.last_angles_deg is not None:
                return True
            time.sleep(0.05)
        return False


class ApexMotionRuntime:
    def __init__(
        self,
        motion_files: list[Path],
        reference_offsets: tuple[int, ...] = REFERENCE_OFFSETS,
        drive_command: bool = True,
    ):
        self.reference_offsets = tuple(int(offset) for offset in reference_offsets)
        self.drive_command = bool(drive_command)
        self.motions = [self._load_motion(path) for path in motion_files]
        if not self.motions:
            raise FileNotFoundError("No Go2+D1 APEX tracker motion NPZ files found.")
        self.motion_id = 0
        self.local_frame = 0
        self.frame_accum = 0.0
        print(f"Loaded {len(self.motions)} Go2+D1 APEX motion(s).")
        self.print_status()

    @staticmethod
    def _load_motion(path: Path) -> dict:
        data = np.load(path)
        body_names = data["body_names"].tolist() if "body_names" in data else []
        joint_names = data["joint_names"].tolist()
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
        joint_vel = (
            np.asarray(data["joint_vel"], dtype=np.float32)
            if "joint_vel" in data
            else np.zeros_like(joint_pos)
        )
        if "gripper_joint_pos" in data:
            gripper_names = (
                data["gripper_joint_names"].tolist()
                if "gripper_joint_names" in data
                else [f"gripper_{idx}" for idx in range(np.asarray(data["gripper_joint_pos"]).shape[1])]
            )
            gripper_pos = np.asarray(data["gripper_joint_pos"], dtype=np.float32)
            gripper_vel = (
                np.asarray(data["gripper_joint_vel"], dtype=np.float32)
                if "gripper_joint_vel" in data
                else np.zeros_like(gripper_pos)
            )
            joint_names = [*joint_names, *gripper_names]
            joint_pos = np.concatenate((joint_pos, gripper_pos), axis=1)
            joint_vel = np.concatenate((joint_vel, gripper_vel), axis=1)

        joint_name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
        missing = [name for name in GO2_D1_TRACKER_JOINT_NAMES if name not in joint_name_to_idx]
        if missing:
            raise ValueError(f"Motion file {path} is missing tracker joints: {missing}")
        tracker_joint_indices = [joint_name_to_idx[name] for name in GO2_D1_TRACKER_JOINT_NAMES]
        base_idx = body_names.index("base") if "base" in body_names else 0
        frame_count = joint_pos.shape[0]
        if "skill" in data:
            skill = np.asarray(data["skill"], dtype=np.float32).reshape(frame_count, -1)
            if skill.shape[1] != 1:
                raise ValueError(f"Motion file {path} skill must be (N, 1); got {skill.shape}.")
        else:
            skill = np.zeros((frame_count, 1), dtype=np.float32)
        return {
            "path": path,
            "fps": float(np.asarray(data["fps"]).item()),
            "base_idx": base_idx,
            "tracker_joint_indices": tracker_joint_indices,
            "joint_pos": joint_pos,
            "body_quat_w": np.asarray(data["body_quat_w"], dtype=np.float32),
            "body_lin_vel_w": np.asarray(data["body_lin_vel_w"], dtype=np.float32),
            "body_ang_vel_w": np.asarray(data["body_ang_vel_w"], dtype=np.float32),
            "skill": skill,
            "command_lin_vel_xy": (
                np.asarray(data["command_lin_vel_xy"], dtype=np.float32)
                if "command_lin_vel_xy" in data
                else None
            ),
            "command_ang_vel_z": (
                np.asarray(data["command_ang_vel_z"], dtype=np.float32).reshape(-1)
                if "command_ang_vel_z" in data
                else None
            ),
        }

    @property
    def active_motion(self) -> dict:
        return self.motions[self.motion_id]

    def print_status(self) -> None:
        motion = self.active_motion
        frame_count = motion["joint_pos"].shape[0]
        print(
            f"Active reference {self.motion_id + 1}/{len(self.motions)}: "
            f"{motion['path'].name}, frame={self.local_frame}/{max(frame_count - 1, 0)}"
        )

    def switch_motion(self, delta: int) -> None:
        if len(self.motions) <= 1:
            return
        self.motion_id = (self.motion_id + int(delta)) % len(self.motions)
        self.local_frame = 0
        self.frame_accum = 0.0
        self.print_status()

    def update(self, dt: float, command_out: np.ndarray | None = None) -> None:
        motion = self.active_motion
        frame_count = motion["joint_pos"].shape[0]
        self.frame_accum += float(dt) * motion["fps"]
        frame_step = int(self.frame_accum)
        if frame_step > 0:
            self.frame_accum -= frame_step
            self.local_frame = (self.local_frame + frame_step) % frame_count

        if not self.drive_command or command_out is None:
            return
        if motion["command_lin_vel_xy"] is not None:
            command_out[0:2] = motion["command_lin_vel_xy"][self.local_frame]
        else:
            command_out[0:2] = motion["body_lin_vel_w"][self.local_frame, motion["base_idx"], :2]
        command_out[2] = (
            float(motion["command_ang_vel_z"][self.local_frame])
            if motion["command_ang_vel_z"] is not None
            else 0.0
        )

    def current_base_lin_vel_z(self) -> float:
        motion = self.active_motion
        return float(motion["body_lin_vel_w"][self.local_frame, motion["base_idx"], 2])

    def current_skill(self) -> float:
        motion = self.active_motion
        frame = min(self.local_frame, motion["skill"].shape[0] - 1)
        return float(motion["skill"][frame, 0])

    def reference_features(self) -> np.ndarray:
        motion = self.active_motion
        frame_count = motion["joint_pos"].shape[0]
        base_idx = motion["base_idx"]
        features = []
        for offset in self.reference_offsets:
            frame = min(self.local_frame + int(offset), frame_count - 1)
            quat = motion["body_quat_w"][frame, base_idx].copy()
            norm = np.linalg.norm(quat)
            if norm > 1.0e-6:
                quat /= norm
            features.extend(
                [
                    motion["joint_pos"][frame, motion["tracker_joint_indices"]],
                    motion["body_lin_vel_w"][frame, base_idx],
                    motion["body_ang_vel_w"][frame, base_idx],
                    quat,
                ]
            )
        return np.concatenate(features).astype(np.float32)


class RLController:
    """50 Hz Go2+D1 APEX tracker with 200 Hz leg command hold."""

    default_joint_angles = np.array(
        [
            -0.1, 0.8, -1.5,
            0.1, 0.8, -1.5,
            -0.1, 1.0, -1.5,
            0.1, 1.0, -1.5,
            0.0, -math.pi / 2.0, math.pi / 2.0, 0.0, 0.0, 0.0,
            0.0,
        ],
        dtype=np.float32,
    )
    action_scales = np.array(
        [
            0.125, 0.25, 0.25,
            0.125, 0.25, 0.25,
            0.125, 0.25, 0.25,
            0.125, 0.25, 0.25,
            0.25, 0.25, 0.25, 0.25, 0.25, 0.25,
            0.005,
        ],
        dtype=np.float32,
    )
    action_clip = 6.0
    frequency = 200.0
    simulation_dt = 1.0 / frequency
    control_decimation = 4
    arm_command_decimation = int(round(D1_ARM_COMMAND_DT / simulation_dt))
    kps = np.array(
        [
            20.0, 20.0, 20.0, 20.0, 20.0, 20.0,
            20.0, 20.0, 20.0, 20.0, 20.0, 20.0,
            200.0, 200.0, 200.0, 50.0, 50.0, 50.0,
            200.0,
        ],
        dtype=np.float32,
    )
    kds = np.array(
        [
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
            0.5, 0.5, 0.5, 0.5, 0.5, 0.5,
            5.0, 5.0, 4.0, 0.25, 0.25, 0.25,
            3.0,
        ],
        dtype=np.float32,
    )
    stand_kps = np.array(
        [
            50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
            50.0, 50.0, 50.0, 50.0, 50.0, 50.0,
            80.0, 80.0, 80.0, 40.0, 40.0, 40.0,
            40.0,
        ],
        dtype=np.float32,
    )
    gravity_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    cmd_limits = np.array([[-1.0, 1.5], [-1.0e9, 1.0e9], [-1.0e9, 1.0e9]], dtype=np.float32)
    cmd_scale = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    key_state = [
        ["R1", 0], ["L1", 0], ["start", 0], ["select", 0],
        ["R2", 0], ["L2", 0], ["F1", 0], ["F2", 0],
        ["A", 0], ["B", 0], ["X", 0], ["Y", 0],
        ["up", 0], ["right", 0], ["down", 0], ["left", 0],
    ]
    key_index = {name: idx for idx, (name, _) in enumerate(key_state)}

    def __init__(
        self,
        policy_path: Path,
        motion_runtime: ApexMotionRuntime,
        arm_backend: str = "d1-dds",
        arm_observation_mode: str = "measured",
        standup_duration_s: float = 5.0,
        sitdown_duration_s: float = DEFAULT_SITDOWN_DURATION_S,
    ):
        if arm_backend not in {"d1-dds", "unified"}:
            raise ValueError("arm_backend must be 'd1-dds' or 'unified'.")
        if arm_observation_mode not in {"measured", "commanded"}:
            raise ValueError("arm_observation_mode must be 'measured' or 'commanded'.")

        self.arm_backend = arm_backend
        self.arm_observation_mode = arm_observation_mode
        self.motion_runtime = motion_runtime
        self.standup_duration_s = float(standup_duration_s)
        self.sitdown_duration_s = float(sitdown_duration_s)
        self._sitdown_requested = False
        self._sitdown_active = False
        self.reference_dim = REFERENCE_FRAME_DIM * len(motion_runtime.reference_offsets)
        self.expected_obs_dim = TRACKER_BASE_OBS_DIM + SKILL_DIM + self.reference_dim
        if self.expected_obs_dim != EXPECTED_OBS_DIM and len(motion_runtime.reference_offsets) == len(REFERENCE_OFFSETS):
            raise ValueError(
                f"Observation layout mismatch: expected {EXPECTED_OBS_DIM}, got {self.expected_obs_dim}."
            )

        self.d1_arm: D1ArmDDSClient | None = None
        if self.arm_backend == "d1-dds":
            self.d1_arm = D1ArmDDSClient()

        self.low_state = None
        self.low_state_suber = ChannelSubscriber("rt/lowstate", LowState_)
        self.low_state_suber.Init(self.LowStateHandler, 10)
        self.low_cmd_puber = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.low_cmd_puber.Init()
        self.wireless_controller_suber = ChannelSubscriber("rt/wirelesscontroller", WirelessController_)
        self.wireless_controller_suber.Init(self.WirelessControllerHandler, 10)
        print("Created lowstate/lowcmd/wirelesscontroller channels.")

        self.interval = 1.0 / self.frequency
        self.actor = self._load_policy(policy_path)
        self.actor_output_names = [output.name for output in self.actor.get_outputs()]

        self.command = np.zeros(3, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.projected_gravity = np.zeros(3, dtype=np.float32)
        self.dof_pos = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.dof_vel = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.prev_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.current_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.current_targets = self.default_joint_angles.copy()
        self.arm_targets = self.default_joint_angles[ARM_MOTOR_START:].copy()
        self.obs = np.zeros(self.expected_obs_dim, dtype=np.float32)
        self.loop_counter = 0
        self.j_lx = self.j_ly = self.j_rx = self.j_ry = 0.0
        self._max_arm_delta = ARM_VELOCITY_LIMITS * POLICY_DT

        self.crc = CRC()
        self.cmd = unitree_go_msg_dds__LowCmd_()
        self.cmd.head[0] = 0xFE
        self.cmd.head[1] = 0xEF
        self.cmd.level_flag = 0xFF
        self.cmd.gpio = 0
        for i in range(20):
            self.cmd.motor_cmd[i].mode = 0x01
            self.cmd.motor_cmd[i].q = 0.0
            self.cmd.motor_cmd[i].kp = 0.0
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kd = 0.0
            self.cmd.motor_cmd[i].tau = 0.0

        self.sc = SportClient()
        self.sc.SetTimeout(5.0)
        self.sc.Init()
        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()
        self._release_builtin_motion_modes()
        if not self.wait_for_low_state(5.0):
            raise RuntimeError("No rt/lowstate messages received. Check robot/sim connection.")
        if self.d1_arm is not None:
            if not self.d1_arm.wait_for_feedback(5.0):
                raise RuntimeError("No D1 rt/arm_Feedback received. Check arm power/network.")
            for _ in range(3):
                self.d1_arm.enable(True)
                time.sleep(0.1)
            measured = self.d1_arm.measured_policy_state()
            if measured is None:
                raise RuntimeError("D1 feedback disappeared before stand-up.")
            arm_rad, gripper_m = measured
            self.arm_targets[:NUM_ARM_JOINTS] = arm_rad
            self.arm_targets[NUM_ARM_JOINTS] = gripper_m
            self.current_targets[ARM_MOTOR_START:] = self.arm_targets
            print(
                "[D1] enabled; measured arm (rad)="
                f"{np.array2string(arm_rad, precision=3)}, gripper_m={gripper_m:.4f}"
            )
        self.stand_up_and_arm_ready()

    def check_stop(self) -> None:
        if _SHUTDOWN_REQUESTED:
            raise KeyboardInterrupt

    def _interruptible_sleep(self, duration_s: float) -> None:
        deadline = time.perf_counter() + duration_s
        while time.perf_counter() < deadline:
            self.check_stop()
            time.sleep(min(0.02, max(0.0, deadline - time.perf_counter())))

    def _release_builtin_motion_modes(self) -> None:
        status, result = self.msc.CheckMode()
        mode_name = result.get("name", "") if isinstance(result, dict) else ""
        if not isinstance(result, dict):
            print(
                f"[WARN] MotionSwitcher CheckMode returned no mode info (status={status}, result={result}). "
                f"On hardware, pass the NIC as the first CLI argument (default: {DEFAULT_NETWORK_INTERFACE})."
            )
        while mode_name:
            self.sc.StandDown()
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            mode_name = result.get("name", "") if isinstance(result, dict) else ""
            if not isinstance(result, dict):
                print(f"[WARN] MotionSwitcher CheckMode failed during release (status={status}, result={result}).")
                break
            self._interruptible_sleep(1.0)

    def _load_policy(self, policy_path: Path) -> "ort.InferenceSession":
        import onnxruntime as ort

        if not policy_path.is_file():
            raise FileNotFoundError(
                f"Go2+D1 APEX policy not found: {policy_path}. "
                "Pass --policy-path to an exported policy.onnx under "
                "logs/rsl_rl/unitree_go2_d1_arm_apex_flat_tracker/<run>/exported/."
            )
        session = ort.InferenceSession(str(policy_path), providers=["CPUExecutionProvider"])
        input_width = _onnx_input_width(session)
        if input_width not in (None, -1, self.expected_obs_dim):
            raise ValueError(
                f"Policy input width is {input_width}, expected {self.expected_obs_dim} "
                f"(offsets={self.motion_runtime.reference_offsets})."
            )
        output_names = [output.name for output in session.get_outputs()]
        dummy = np.zeros(self.expected_obs_dim, dtype=np.float32)
        for _ in range(POLICY_WARMUP_RUNS):
            actions = session.run(output_names, _feed_dict(session, dummy))[0].reshape(-1)
            if actions.size != NUM_ACTIONS:
                raise ValueError(f"Policy output has {actions.size} actions, expected {NUM_ACTIONS}.")
        print(
            f"Loaded policy from {policy_path} "
            f"(obs={input_width}, actions={NUM_ACTIONS}, warmup={POLICY_WARMUP_RUNS})."
        )
        return session

    def _apply_joystick_command(self) -> None:
        self.command[0] = self.j_ly
        self.command[1] = self.j_lx
        self.command[2] = -self.j_rx

    def update_motion_and_command(self, dt: float) -> None:
        if self.motion_runtime.drive_command:
            self.motion_runtime.update(dt, self.command)
        else:
            self.motion_runtime.update(dt, None)
            self._apply_joystick_command()
        if self.key_state[self.key_index["A"]][1] == 1 or self.key_state[self.key_index["B"]][1] == 1:
            self.command[:] = 0.0
        self.command[:] = np.clip(self.command, self.cmd_limits[:, 0], self.cmd_limits[:, 1])

    def _read_arm_state(self) -> tuple[np.ndarray, np.ndarray]:
        """Return absolute arm q (7) and dq (7; dq unused by actor, kept for completeness)."""
        arm_q = self.arm_targets.copy()
        arm_dq = np.zeros(NUM_ARM_JOINTS + 1, dtype=np.float32)
        if self.d1_arm is not None:
            measured = self.d1_arm.measured_policy_state()
            if measured is not None:
                arm_rad, gripper_m = measured
                if self.arm_observation_mode == "measured":
                    arm_q[:NUM_ARM_JOINTS] = arm_rad
                    arm_q[NUM_ARM_JOINTS] = gripper_m
        elif self.low_state is not None:
            for j in range(NUM_ARM_JOINTS + 1):
                motor = self.low_state.motor_state[ARM_MOTOR_START + j]
                if self.arm_observation_mode == "measured":
                    arm_q[j] = motor.q
                arm_dq[j] = motor.dq
        return arm_q, arm_dq

    def get_obs(self) -> np.ndarray:
        self.ang_vel[:] = self.low_state.imu_state.gyroscope[:3]
        rot_matrix = quat_to_rot(self.low_state.imu_state.quaternion)
        self.projected_gravity[:] = rot_matrix.T @ self.gravity_vec

        for i in range(NUM_LEG_ACTIONS):
            motor = self.low_state.motor_state[i]
            self.dof_pos[i] = motor.q - self.default_joint_angles[i]
            self.dof_vel[i] = motor.dq

        arm_q, _arm_dq = self._read_arm_state()
        for j in range(NUM_ARM_JOINTS + 1):
            motor_i = ARM_MOTOR_START + j
            self.dof_pos[motor_i] = arm_q[j] - self.default_joint_angles[motor_i]
            # Deployable actor intentionally omits D1 velocities.
            self.dof_vel[motor_i] = 0.0

        command_obs = np.array(
            [
                self.command[0] * self.cmd_scale[0],
                self.command[1] * self.cmd_scale[1],
                self.motion_runtime.current_base_lin_vel_z() * self.cmd_scale[2],
                self.command[2] * self.cmd_scale[3],
            ],
            dtype=np.float32,
        )
        actor_joint_vel = self.dof_vel[list(ACTOR_JOINT_VELOCITY_INDICES)] * 0.05
        reference_features = self.motion_runtime.reference_features()
        if reference_features.size != self.reference_dim:
            raise ValueError(
                f"Reference feature size is {reference_features.size}, expected {self.reference_dim}."
            )
        skill = np.array([self.motion_runtime.current_skill()], dtype=np.float32)

        self.obs[:] = np.concatenate(
            (
                self.ang_vel * 0.25,
                self.projected_gravity,
                command_obs,
                self.dof_pos,
                actor_joint_vel,
                self.prev_action,
                skill,
                reference_features,
            )
        ).astype(np.float32, copy=False)
        if not np.all(np.isfinite(self.obs)):
            raise ValueError("APEX observation contains NaN or Inf; refusing to run policy.")
        return self.obs

    def _clip_arm_targets(self, targets: np.ndarray) -> np.ndarray:
        clipped = np.clip(targets, ARM_LOWER_LIMITS, ARM_UPPER_LIMITS)
        delta = np.clip(clipped - self.arm_targets, -self._max_arm_delta, self._max_arm_delta)
        return self.arm_targets + delta

    def get_action(self) -> tuple[np.ndarray, np.ndarray]:
        obs = self.get_obs()
        actions = (
            self.actor.run(self.actor_output_names, _feed_dict(self.actor, obs))[0]
            .reshape(-1)
            .astype(np.float32)
        )
        if actions.size != NUM_ACTIONS:
            raise ValueError(f"Policy output has {actions.size} actions, expected {NUM_ACTIONS}.")
        if not np.all(np.isfinite(actions)):
            raise ValueError("APEX policy output contains NaN or Inf; refusing to command robot.")
        np.clip(actions, -self.action_clip, self.action_clip, out=actions)
        # Direct absolute targets: no DecAP / reference residual.
        targets = self.default_joint_angles + actions * self.action_scales
        targets[ARM_MOTOR_START:] = self._clip_arm_targets(targets[ARM_MOTOR_START:])
        return actions, targets

    def _send_arm_targets(self) -> None:
        if self.d1_arm is None:
            return
        self.d1_arm.command_policy_targets(
            self.arm_targets[:NUM_ARM_JOINTS],
            float(self.arm_targets[NUM_ARM_JOINTS]),
        )

    def _write_leg_cmd(self, q: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> None:
        for i in range(NUM_LEG_ACTIONS):
            self.cmd.motor_cmd[i].q = float(q[i])
            self.cmd.motor_cmd[i].kp = float(kp[i])
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kd = float(kd[i])
            self.cmd.motor_cmd[i].tau = 0.0

    def _write_unified_arm_cmd(self, q: np.ndarray, kp: np.ndarray, kd: np.ndarray) -> None:
        for i in range(ARM_MOTOR_START, NUM_MOTORS):
            self.cmd.motor_cmd[i].q = float(q[i])
            self.cmd.motor_cmd[i].kp = float(kp[i])
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kd = float(kd[i])
            self.cmd.motor_cmd[i].tau = 0.0

    def run(self) -> None:
        self.check_stop()
        if self.estop_requested():
            self.e_stop()
            raise SystemExit(0)
        if self._sitdown_requested and not self._sitdown_active:
            self.sit_down_and_exit()
            raise SystemExit(0)

        self.update_motion_and_command(self.interval)
        if self.loop_counter % self.control_decimation == 0:
            raw_action, targets = self.get_action()
            self.current_action[:] = raw_action
            self.prev_action[:] = raw_action
            self.current_targets[:] = targets
            self.arm_targets[:] = targets[ARM_MOTOR_START:]
        if self.loop_counter % self.arm_command_decimation == 0:
            self._send_arm_targets()

        self._write_leg_cmd(self.current_targets, self.kps, self.kds)
        if self.d1_arm is None:
            self._write_unified_arm_cmd(self.current_targets, self.kps, self.kds)

        self.safe_write_cmd()
        self.loop_counter += 1

    def estop_requested(self) -> bool:
        return (
            self.key_state[self.key_index["R1"]][1] == 1
            and self.key_state[self.key_index["L1"]][1] == 1
        )

    def safe_write_cmd(self) -> None:
        if self.estop_requested():
            self.e_stop()
            raise SystemExit(0)
        self.cmd.crc = self.crc.Crc(self.cmd)
        self.low_cmd_puber.Write(self.cmd)

    def _passive_leg_cmd(self) -> None:
        motor_count = NUM_LEG_ACTIONS if self.d1_arm is not None else NUM_MOTORS
        for i in range(motor_count):
            self.cmd.motor_cmd[i].q = 0.0
            self.cmd.motor_cmd[i].kp = 0.0
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kd = 0.0
            self.cmd.motor_cmd[i].tau = 0.0
        self.cmd.crc = self.crc.Crc(self.cmd)
        self.low_cmd_puber.Write(self.cmd)

    def shutdown(self, *, discharge_arm: bool = True) -> None:
        if discharge_arm and self.d1_arm is not None:
            try:
                self.d1_arm.enable(False)
            except Exception as exc:  # noqa: BLE001 - best-effort estop path
                print(f"[D1] WARNING: discharge failed: {exc}")

    def e_stop(self) -> None:
        self._passive_leg_cmd()
        self.shutdown(discharge_arm=True)
        print("Emergency stop: R1+L1 - Go2 legs passive, D1 discharged.")

    def stand_up_and_arm_ready(self) -> None:
        """Interpolate measured Go2 + D1 pose to the APEX default stand / arm-ready pose."""
        duration = max(0.0, self.standup_duration_s)
        if duration <= 0.0:
            self.current_targets[:] = self.default_joint_angles
            self.arm_targets[:] = self.default_joint_angles[ARM_MOTOR_START:]
            self._send_arm_targets()
            self._write_leg_cmd(self.current_targets, self.kps, self.kds)
            if self.d1_arm is None:
                self._write_unified_arm_cmd(self.current_targets, self.kps, self.kds)
            self.safe_write_cmd()
            return

        start_q = np.zeros(NUM_MOTORS, dtype=np.float32)
        for i in range(NUM_LEG_ACTIONS):
            start_q[i] = float(self.low_state.motor_state[i].q)
        arm_q, _ = self._read_arm_state()
        start_q[ARM_MOTOR_START:] = arm_q
        goal_q = self.default_joint_angles.copy()

        print(
            f"[INFO] Stand-up / arm-ready for {duration:.1f}s "
            f"(legs+arm from measured state -> defaults)."
        )
        running_time = 0.0
        step_start = time.perf_counter()
        while running_time < duration:
            self.check_stop()
            if self.estop_requested():
                self.e_stop()
                raise SystemExit(0)
            running_time += 0.002
            phase = float(np.tanh(running_time / max(0.24 * duration, 1.0e-6)))
            q = phase * goal_q + (1.0 - phase) * start_q
            kp = phase * self.stand_kps + (1.0 - phase) * 20.0
            self.current_targets[:] = q
            self.arm_targets[:] = q[ARM_MOTOR_START:]
            # Match the D1's 10 Hz position-command interface during stand-up.
            if int(running_time / 0.002) % 50 == 0:
                self._send_arm_targets()
            self._write_leg_cmd(q, kp, self.kds)
            if self.d1_arm is None:
                self._write_unified_arm_cmd(q, kp, self.kds)
            self.safe_write_cmd()
            self._interruptible_sleep(max(0.0, 0.002 - (time.perf_counter() - step_start)))
            step_start = time.perf_counter()

        self.current_targets[:] = goal_q
        self.arm_targets[:] = goal_q[ARM_MOTOR_START:]
        self._send_arm_targets()
        print("[INFO] Stand-up / arm-ready complete.")

    def request_sit_down(self) -> None:
        if self._sitdown_requested or self._sitdown_active:
            return
        self._sitdown_requested = True
        print(
            f"[INFO] Select pressed: stopping policy and sitting down "
            f"over {self.sitdown_duration_s:.1f}s.",
            flush=True,
        )

    def sit_down_and_exit(self) -> None:
        """Slowly move legs to the Go2 sit pose and fold the arm, then exit."""
        self._sitdown_active = True
        duration = max(0.5, float(self.sitdown_duration_s))

        start_q = np.zeros(NUM_MOTORS, dtype=np.float32)
        for i in range(NUM_LEG_ACTIONS):
            start_q[i] = float(self.low_state.motor_state[i].q)
        arm_q, _ = self._read_arm_state()
        start_q[ARM_MOTOR_START:] = arm_q

        goal_q = np.zeros(NUM_MOTORS, dtype=np.float32)
        goal_q[:NUM_LEG_ACTIONS] = GO2_SIT_DOWN_LEG_POS
        # Fold the arm to the default rest pose while the dog sits.
        goal_q[ARM_MOTOR_START:] = self.default_joint_angles[ARM_MOTOR_START:]

        print(
            f"[INFO] Sit-down for {duration:.1f}s "
            "(current pose -> Go2 sit + arm fold).",
            flush=True,
        )
        running_time = 0.0
        step_start = time.perf_counter()
        while running_time < duration:
            self.check_stop()
            if self.estop_requested():
                self.e_stop()
                raise SystemExit(0)
            running_time += 0.002
            phase = float(np.tanh(running_time / max(0.24 * duration, 1.0e-6)))
            q = phase * goal_q + (1.0 - phase) * start_q
            kp = phase * self.stand_kps + (1.0 - phase) * np.maximum(self.kps, 20.0)
            self.current_targets[:] = q
            self.arm_targets[:] = q[ARM_MOTOR_START:]
            if int(running_time / 0.002) % 50 == 0:
                self._send_arm_targets()
            self._write_leg_cmd(q, kp, self.kds)
            if self.d1_arm is None:
                self._write_unified_arm_cmd(q, kp, self.kds)
            self.safe_write_cmd()
            self._interruptible_sleep(max(0.0, 0.002 - (time.perf_counter() - step_start)))
            step_start = time.perf_counter()

        self.current_targets[:] = goal_q
        self.arm_targets[:] = goal_q[ARM_MOTOR_START:]
        self._send_arm_targets()
        # Hold the sit pose briefly, then release so the script can exit cleanly.
        hold_deadline = time.perf_counter() + 1.0
        while time.perf_counter() < hold_deadline:
            self.check_stop()
            if self.estop_requested():
                self.e_stop()
                raise SystemExit(0)
            self._write_leg_cmd(goal_q, self.stand_kps * 0.5, self.kds)
            if self.d1_arm is None:
                self._write_unified_arm_cmd(goal_q, self.stand_kps * 0.5, self.kds)
            self.safe_write_cmd()
            self._interruptible_sleep(0.002)

        self._passive_leg_cmd()
        self.shutdown(discharge_arm=True)
        print("[INFO] Sit-down complete. Policy stopped; legs passive, D1 discharged.")

    def wait_for_low_state(self, timeout_s: float) -> bool:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            self.check_stop()
            if self.low_state is not None:
                return True
            if self.estop_requested():
                self.e_stop()
                raise SystemExit(0)
            self._interruptible_sleep(0.05)
        return False

    def WirelessControllerHandler(self, msg: "WirelessController_") -> None:
        old_x = self.key_state[self.key_index["X"]][1]
        old_y = self.key_state[self.key_index["Y"]][1]
        old_select = self.key_state[self.key_index["select"]][1]
        self.j_lx = msg.lx
        self.j_ly = msg.ly
        self.j_rx = msg.rx
        self.j_ry = msg.ry
        for i in range(16):
            self.key_state[i][1] = (msg.keys & (1 << i)) >> i
        if old_y == 0 and self.key_state[self.key_index["Y"]][1] == 1:
            self.motion_runtime.switch_motion(delta=1)
        if old_x == 0 and self.key_state[self.key_index["X"]][1] == 1:
            self.motion_runtime.switch_motion(delta=-1)
        if old_select == 0 and self.key_state[self.key_index["select"]][1] == 1:
            self.request_sit_down()

    def LowStateHandler(self, msg: "LowState_") -> None:
        self.low_state = msg


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run Go2+D1 direct APEX flat-tracker ONNX policy on Unitree hardware."
    )
    parser.add_argument(
        "interface",
        nargs="?",
        default=None,
        help=f"Network interface (default hardware: {DEFAULT_NETWORK_INTERFACE}, unified: lo).",
    )
    parser.add_argument(
        "--policy-path",
        default=os.environ.get("GURUKUL_POLICY_PATH", str(DEFAULT_POLICY_PATH)),
        help="Exported policy.onnx under unitree_go2_d1_arm_apex_flat_tracker/<run>/exported/.",
    )
    parser.add_argument(
        "--motion-file",
        action="append",
        default=None,
        help="Reference NPZ file, directory, or glob. Can be passed multiple times.",
    )
    parser.add_argument(
        "--arm-backend",
        choices=("d1-dds", "unified"),
        default="d1-dds",
        help="d1-dds: real hardware split transports; unified: all 19 joints on rt/lowcmd.",
    )
    parser.add_argument(
        "--arm-observation-mode",
        choices=("measured", "commanded"),
        default="measured",
        help="Use measured D1 feedback or the latest commanded arm targets in the actor obs.",
    )
    parser.add_argument(
        "--joystick-command",
        action="store_true",
        help="Use remote sticks instead of motion command channels.",
    )
    parser.add_argument(
        "--standup-duration",
        type=float,
        default=5.0,
        help="Measured-to-default stand-up / arm-ready duration in seconds.",
    )
    parser.add_argument(
        "--sitdown-duration",
        type=float,
        default=DEFAULT_SITDOWN_DURATION_S,
        help="Select-button slow sit-down duration in seconds.",
    )
    parser.add_argument(
        "--config-path",
        default=str(DEFAULT_CONFIG_PATH),
        help="Optional yaml path (documented contract; values are embedded for safety).",
    )
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _sigint_handler)

    policy_path = _resolve_path(args.policy_path)
    motion_patterns = args.motion_file or [str(DEFAULT_MOTION_PATH)]
    motion_files = _resolve_motion_files(motion_patterns)
    if not motion_files:
        raise FileNotFoundError(f"No motion files matched: {motion_patterns}")

    motion_runtime = ApexMotionRuntime(
        motion_files,
        reference_offsets=REFERENCE_OFFSETS,
        drive_command=not args.joystick_command,
    )

    print("WARNING: Go2+D1 APEX hardware runner. Clear the workspace and keep R1+L1 for estop.")
    print("Remote: Select = slow sit-down and stop policy; R1+L1 = estop.")
    print(f"Policy: {policy_path}")
    print(f"Config contract: {args.config_path}")
    print(f"Arm backend: {args.arm_backend}; obs mode: {args.arm_observation_mode}")
    print(
        "Contract: 19 policy targets @ 50 Hz; D1 packet commands @ 10 Hz "
        "(no DecAP or reference-residual path)."
    )
    input("Press Enter to continue...")

    iface = init_unitree_channel(args.interface, arm_backend=args.arm_backend)
    print(f"DDS interface: {iface}")

    controller = RLController(
        policy_path,
        motion_runtime,
        arm_backend=args.arm_backend,
        arm_observation_mode=args.arm_observation_mode,
        standup_duration_s=args.standup_duration,
        sitdown_duration_s=args.sitdown_duration,
    )
    try:
        while True:
            start = time.perf_counter()
            controller.run()
            elapsed = time.perf_counter() - start
            controller._interruptible_sleep(max(0.0, controller.interval - elapsed))
    except (KeyboardInterrupt, SystemExit):
        pass
    finally:
        controller.shutdown(discharge_arm=True)
        controller._passive_leg_cmd()
        print("[INFO] Shutdown complete.")


if __name__ == "__main__":
    main()
