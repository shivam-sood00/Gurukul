"""B2+Z1 APEX one-step history tracker on Unitree hardware.

Observation layout matches
Gurukul-Isaac-B2-Z1-Arm-APEX-Flat-Tracker-One-Step-Future-History-v0:

  5-frame term-major history over a 120-D tracker observation.

Single-frame terms:
  base_ang_vel(3), projected_gravity(3), command_x/y/z/yaw(4),
  joint_pos(18), joint_vel(18), previous_action(18), reference_features(56)

The ONNX actor outputs 18 position deltas: 12 B2 leg joints plus Z1 joint1..joint6.
The gripper is not in the APEX action/reference subset and is held at 0.0 by default.

Hardware backends (--arm-backend):
  z1-sdk  Real robot: B2 legs on unitree_sdk2py rt/lowcmd.
          Z1 joint1..joint6 and gripper hold via unitree_arm_interface UDP.
  unified MuJoCo / combined lowstate: all 19 motors on rt/lowcmd.
"""

from __future__ import annotations

import argparse
import glob
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
UNITREE_ROOT = SCRIPT_DIR.parents[1]
REPO_ROOT = SCRIPT_DIR.parents[2]
DEPLOY_SIM2REAL_DIR = UNITREE_ROOT / "UnitreeB2_Z1_Deployment" / "sim2real"
if str(DEPLOY_SIM2REAL_DIR) not in sys.path:
    sys.path.insert(0, str(DEPLOY_SIM2REAL_DIR))

from z1_arm_client import Z1ArmClient, z1_controller_is_running  # noqa: E402

# Edit this path for the exported 600-D history policy you want to run.
DEFAULT_POLICY_PATH = (
    REPO_ROOT
    / "logs/rsl_rl/unitree_b2_z1_arm_apex_flat_tracker_one_step_future_history_distill"
    / "<run>/exported/policy.onnx"
)

DEFAULT_MOTION_GLOB = (
    REPO_ROOT
    / "source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/npz"
    / "b2_z1_motions/**/*.npz"
)

DEFAULT_NETWORK_INTERFACE = os.environ.get("B2Z1_NETWORK_INTERFACE", "enp4s0")
SIM_NETWORK_INTERFACE = "lo"

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
            "unitree_sdk2py is required for B2+Z1 hardware control. "
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


def _resolve_network_interface(interface: str | None, *, arm_backend: str = "z1-sdk") -> str:
    if interface is not None:
        return interface
    return SIM_NETWORK_INTERFACE if arm_backend == "unified" else DEFAULT_NETWORK_INTERFACE


def init_unitree_channel(interface: str | None = None, *, arm_backend: str = "z1-sdk") -> str:
    _import_unitree_sdk()
    iface = _resolve_network_interface(interface, arm_backend=arm_backend)
    if iface == SIM_NETWORK_INTERFACE:
        ChannelFactoryInitialize(1, SIM_NETWORK_INTERFACE)
    else:
        ChannelFactoryInitialize(0, iface)
    return iface


NUM_LEG_ACTIONS = 12
NUM_ACTIONS = 18
NUM_MOTORS = 19
ARM_MOTOR_START = NUM_LEG_ACTIONS
NUM_TRACKED_ARM_JOINTS = 6
NUM_ARM_MOTORS = 7

HISTORY_LENGTH = 5
TRACKER_BASE_OBS_DIM = 10 + 2 * NUM_ACTIONS + NUM_ACTIONS
REFERENCE_OFFSETS = (0, 1)
REFERENCE_FRAME_DIM = NUM_ACTIONS + 10
REFERENCE_DIM = REFERENCE_FRAME_DIM * len(REFERENCE_OFFSETS)
SINGLE_OBS_DIM = TRACKER_BASE_OBS_DIM + REFERENCE_DIM
EXPECTED_OBS_DIM = HISTORY_LENGTH * SINGLE_OBS_DIM

POLICY_WARMUP_RUNS = 5
GRIPPER_DEFAULT = 0.0

B2_Z1_TRACKER_JOINT_NAMES = (
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
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
)

_SHUTDOWN_REQUESTED = False


def _sigint_handler(_signum, _frame) -> None:
    global _SHUTDOWN_REQUESTED
    if _SHUTDOWN_REQUESTED:
        print("\n[FAIL] Second Ctrl+C - force exit.", flush=True)
        os._exit(130)
    _SHUTDOWN_REQUESTED = True
    print("\n[INFO] Stop requested (Ctrl+C). Shutting down...", flush=True)


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


def _resolve_motion_files(patterns) -> list[Path]:
    if isinstance(patterns, (str, os.PathLike)):
        patterns = [patterns]
    paths: list[Path] = []
    for pattern in patterns:
        matches = sorted(glob.glob(str(Path(pattern).expanduser()), recursive=True))
        for match in matches:
            path = Path(match)
            if path.is_dir():
                paths.extend(sorted(path.rglob("*.npz")))
            elif path.is_file():
                paths.append(path)
    return list(dict.fromkeys(path.resolve() for path in paths))


def _parse_reference_offsets(value: str | None) -> tuple[int, ...]:
    if value is None:
        return REFERENCE_OFFSETS
    offsets = tuple(int(part.strip()) for part in value.split(",") if part.strip())
    if not offsets:
        raise ValueError("--reference-offsets cannot be empty.")
    return offsets


def _parse_float_list(text: str, expected_len: int, label: str) -> np.ndarray:
    values = [float(part.strip()) for part in text.split(",") if part.strip()]
    if len(values) != expected_len:
        raise ValueError(f"{label} expects {expected_len} comma-separated values, got {len(values)}.")
    return np.asarray(values, dtype=np.float32)


def _onnx_input_width(session: ort.InferenceSession) -> int | None:
    shape = session.get_inputs()[0].shape
    dims = [dim for dim in shape if isinstance(dim, int) and dim > 0]
    return int(dims[-1]) if dims else None


def _feed_dict(session: ort.InferenceSession, obs: np.ndarray) -> dict[str, np.ndarray]:
    meta = session.get_inputs()[0]
    values = np.asarray(obs, dtype=np.float32).reshape(-1)
    if len(meta.shape) == 2:
        values = values.reshape(1, -1)
    elif len(meta.shape) == 3:
        values = values.reshape(1, 1, -1)
    return {meta.name: values}


def _history_term_slices(reference_dim: int) -> list[tuple[int, int]]:
    term_sizes = (
        3,
        3,
        1,
        1,
        1,
        1,
        NUM_ACTIONS,
        NUM_ACTIONS,
        NUM_ACTIONS,
        reference_dim,
    )
    slices: list[tuple[int, int]] = []
    start = 0
    for size in term_sizes:
        end = start + int(size)
        slices.append((start, end))
        start = end
    if start != SINGLE_OBS_DIM:
        raise ValueError(f"APEX history terms cover {start} dims, expected {SINGLE_OBS_DIM}.")
    return slices


class ApexMotionRuntime:
    def __init__(self, motion_files: list[Path], reference_offsets: tuple[int, ...], drive_command: bool = True):
        self.reference_offsets = tuple(int(offset) for offset in reference_offsets)
        self.drive_command = bool(drive_command)
        self.motions = [self._load_motion(path) for path in motion_files]
        if not self.motions:
            raise FileNotFoundError("No B2+Z1 APEX tracker motion NPZ files found.")
        self.motion_id = 0
        self.local_frame = 0
        self.frame_accum = 0.0
        print(f"Loaded {len(self.motions)} B2+Z1 APEX motion(s).")
        self.print_status()

    @staticmethod
    def _load_motion(path: Path) -> dict:
        data = np.load(path)
        body_names = data["body_names"].tolist() if "body_names" in data else []
        joint_names = data["joint_names"].tolist()
        joint_name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
        missing = [name for name in B2_Z1_TRACKER_JOINT_NAMES if name not in joint_name_to_idx]
        if missing:
            raise ValueError(f"Motion file {path} is missing tracker joints: {missing}")
        tracker_joint_indices = [joint_name_to_idx[name] for name in B2_Z1_TRACKER_JOINT_NAMES]
        base_idx = body_names.index("base") if "base" in body_names else 0
        return {
            "path": path,
            "fps": float(np.asarray(data["fps"]).item()),
            "base_idx": base_idx,
            "tracker_joint_indices": tracker_joint_indices,
            "joint_pos": np.asarray(data["joint_pos"], dtype=np.float32),
            "body_quat_w": np.asarray(data["body_quat_w"], dtype=np.float32),
            "body_lin_vel_w": np.asarray(data["body_lin_vel_w"], dtype=np.float32),
            "body_ang_vel_w": np.asarray(data["body_ang_vel_w"], dtype=np.float32),
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
    default_joint_angles = np.array(
        [
            -0.1, 0.8, -1.5,
            0.1, 0.8, -1.5,
            -0.1, 0.8, -1.5,
            0.1, 0.8, -1.5,
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, GRIPPER_DEFAULT,
        ],
        dtype=np.float32,
    )
    frequency = 200.0
    simulation_dt = 1.0 / frequency
    control_decimation = 4
    kps = np.array(
        [
            360.0, 360.0, 360.0, 360.0, 360.0, 360.0,
            360.0, 360.0, 360.0, 360.0, 360.0, 360.0,
            512.0, 768.0, 768.0, 512.0, 384.0, 256.0, 512.0,
        ],
        dtype=np.float32,
    )
    kds = np.array(
        [
            5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
            5.0, 5.0, 5.0, 5.0, 5.0, 5.0,
            25.6, 25.6, 25.6, 25.6, 25.6, 25.6, 25.6,
        ],
        dtype=np.float32,
    )
    stand_kps = np.array(
        [
            350.0, 350.0, 350.0, 350.0, 350.0, 350.0,
            350.0, 350.0, 350.0, 350.0, 350.0, 350.0,
            80.0, 80.0, 80.0, 60.0, 40.0, 30.0, 20.0,
        ],
        dtype=np.float32,
    )
    action_scales = np.array(
        [
            0.125, 0.25, 0.25,
            0.125, 0.25, 0.25,
            0.125, 0.25, 0.25,
            0.125, 0.25, 0.25,
            0.05, 0.05, 0.05, 0.05, 0.05, 0.05,
        ],
        dtype=np.float32,
    )
    gravity_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    cmd_limits = np.array([[-1.0, 1.5], [-1.0e9, 1.0e9], [-1.0e9, 1.0e9]], dtype=np.float32)

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
        arm_observation_mode: str = "measured",
        arm_backend: str = "z1-sdk",
        z1_lib_dir: Path | None = None,
    ):
        if arm_observation_mode not in {"measured", "commanded"}:
            raise ValueError("arm_observation_mode must be 'measured' or 'commanded'.")
        if arm_backend not in {"z1-sdk", "unified"}:
            raise ValueError("arm_backend must be 'z1-sdk' or 'unified'.")
        self.arm_observation_mode = arm_observation_mode
        self.arm_backend = arm_backend
        self.motion_runtime = motion_runtime
        self.reference_dim = REFERENCE_FRAME_DIM * len(motion_runtime.reference_offsets)
        self.expected_obs_dim = HISTORY_LENGTH * (TRACKER_BASE_OBS_DIM + self.reference_dim)
        self.history_term_slices = _history_term_slices(self.reference_dim)
        self.obs_history = np.zeros((HISTORY_LENGTH, SINGLE_OBS_DIM), dtype=np.float32)
        self.history_initialized = False

        self.arm_joint_positions = self.default_joint_angles[ARM_MOTOR_START:].copy()
        self.z1_arm: Z1ArmClient | None = None
        if self.arm_backend == "z1-sdk":
            self.z1_arm = Z1ArmClient(self.arm_joint_positions, z1_lib_dir=z1_lib_dir)
            self.z1_arm.start()
            print("Z1 arm: unitree_arm_interface (UDP). B2 lowcmd: legs only.")

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
        self.actor_input_name = self.actor.get_inputs()[0].name
        self.actor_output_names = [output.name for output in self.actor.get_outputs()]

        self.command = np.zeros(3, dtype=np.float32)
        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.projected_gravity = np.zeros(3, dtype=np.float32)
        self.dof_pos = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.dof_vel = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.prev_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.current_action = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.current_target_delta = np.zeros(NUM_ACTIONS, dtype=np.float32)
        self.current_obs = np.zeros(SINGLE_OBS_DIM, dtype=np.float32)
        self.flattened_obs = np.zeros(self.expected_obs_dim, dtype=np.float32)
        self.loop_counter = 0
        self.j_lx = self.j_ly = self.j_rx = self.j_ry = 0.0

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
        self.stand_up()

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

    def _load_policy(self, policy_path: Path) -> ort.InferenceSession:
        import onnxruntime as ort

        if not policy_path.is_file():
            raise FileNotFoundError(
                f"B2+Z1 APEX history policy not found: {policy_path}. "
                "Pass --policy-path or edit DEFAULT_POLICY_PATH in this file."
            )
        session = ort.InferenceSession(str(policy_path))
        input_width = _onnx_input_width(session)
        if input_width not in (None, -1, self.expected_obs_dim):
            raise ValueError(
                f"Policy input width is {input_width}, expected {self.expected_obs_dim}. "
                "Use the B2+Z1 APEX one-step history exported actor."
            )
        dummy_obs = np.zeros(self.expected_obs_dim, dtype=np.float32)
        output_names = [output.name for output in session.get_outputs()]
        for _ in range(POLICY_WARMUP_RUNS):
            session.run(output_names, _feed_dict(session, dummy_obs))
        print(f"Loaded policy from {policy_path} (input dim={input_width}, warmup={POLICY_WARMUP_RUNS}).")
        return session

    def _apply_joystick_command(self) -> None:
        self.command[0] = self.j_ly
        self.command[1] = self.j_lx
        self.command[2] = -self.j_rx
        if self.key_state[self.key_index["up"]][1] == 1:
            self.command[0] = self.cmd_limits[0, 1]
        if self.key_state[self.key_index["down"]][1] == 1:
            self.command[0] = self.cmd_limits[0, 0]
        if self.key_state[self.key_index["left"]][1] == 1:
            self.command[1] = self.cmd_limits[1, 1]
        if self.key_state[self.key_index["right"]][1] == 1:
            self.command[1] = self.cmd_limits[1, 0]
        if self.key_state[self.key_index["L2"]][1] == 1:
            self.command[2] = self.cmd_limits[2, 1]
        if self.key_state[self.key_index["R2"]][1] == 1:
            self.command[2] = self.cmd_limits[2, 0]

    def update_motion_and_command(self, dt: float) -> None:
        if self.motion_runtime.drive_command:
            self.motion_runtime.update(dt, self.command)
        else:
            self.motion_runtime.update(dt, None)
            self._apply_joystick_command()
        if self.key_state[self.key_index["A"]][1] == 1 or self.key_state[self.key_index["B"]][1] == 1:
            self.command[:] = 0.0
        self.command[:] = np.clip(self.command, self.cmd_limits[:, 0], self.cmd_limits[:, 1])

    def get_obs(self) -> np.ndarray:
        self.ang_vel[:] = self.low_state.imu_state.gyroscope[:3]
        rot_matrix = quat_to_rot(self.low_state.imu_state.quaternion)
        self.projected_gravity[:] = rot_matrix.T @ self.gravity_vec

        dof_info = self.low_state.motor_state
        for i in range(NUM_LEG_ACTIONS):
            motor = dof_info[i]
            self.dof_pos[i] = motor.q - self.default_joint_angles[i]
            self.dof_vel[i] = motor.dq

        if self.z1_arm is not None:
            arm_q = self.z1_arm.get_joint_positions()
            arm_dq = self.z1_arm.get_joint_velocities()
            for j in range(NUM_TRACKED_ARM_JOINTS):
                motor_i = ARM_MOTOR_START + j
                if self.arm_observation_mode == "measured":
                    self.dof_pos[motor_i] = arm_q[j] - self.default_joint_angles[motor_i]
                    self.dof_vel[motor_i] = arm_dq[j]
                else:
                    self.dof_pos[motor_i] = self.arm_joint_positions[j] - self.default_joint_angles[motor_i]
                    self.dof_vel[motor_i] = arm_dq[j]
        else:
            for i in range(ARM_MOTOR_START, ARM_MOTOR_START + NUM_TRACKED_ARM_JOINTS):
                motor = dof_info[i]
                if self.arm_observation_mode == "measured":
                    self.dof_pos[i] = motor.q - self.default_joint_angles[i]
                else:
                    self.dof_pos[i] = self.arm_joint_positions[i - ARM_MOTOR_START] - self.default_joint_angles[i]
                self.dof_vel[i] = motor.dq

        command_obs = np.array(
            [
                self.command[0],
                self.command[1],
                self.motion_runtime.current_base_lin_vel_z(),
                self.command[2],
            ],
            dtype=np.float32,
        )
        reference_features = self.motion_runtime.reference_features()
        if reference_features.size != self.reference_dim:
            raise ValueError(f"Reference feature size is {reference_features.size}, expected {self.reference_dim}.")

        self.current_obs[:] = np.concatenate(
            (
                self.ang_vel * 0.25,
                self.projected_gravity,
                command_obs,
                self.dof_pos,
                self.dof_vel * 0.05,
                self.prev_action,
                reference_features,
            )
        ).astype(np.float32, copy=False)
        if not self.history_initialized:
            self.obs_history[:] = self.current_obs
            self.history_initialized = True
        else:
            self.obs_history[:-1] = self.obs_history[1:]
            self.obs_history[-1] = self.current_obs

        output_idx = 0
        for start, end in self.history_term_slices:
            term_history = self.obs_history[:, start:end].reshape(-1)
            next_idx = output_idx + term_history.size
            self.flattened_obs[output_idx:next_idx] = term_history
            output_idx = next_idx
        if not np.all(np.isfinite(self.flattened_obs)):
            raise ValueError("APEX observation contains NaN or Inf; refusing to run policy.")
        return self.flattened_obs

    def get_action(self) -> tuple[np.ndarray, np.ndarray]:
        obs = self.get_obs()
        actions = self.actor.run(self.actor_output_names, _feed_dict(self.actor, obs))[0].reshape(-1).astype(np.float32)
        if actions.size != NUM_ACTIONS:
            raise ValueError(f"Policy output has {actions.size} actions, expected {NUM_ACTIONS}.")
        if not np.all(np.isfinite(actions)):
            raise ValueError("APEX policy output contains NaN or Inf; refusing to command robot.")
        return actions, actions * self.action_scales

    def run(self) -> None:
        self.check_stop()
        if self.estop_requested():
            self.e_stop()
            raise SystemExit(0)

        self.update_motion_and_command(self.interval)
        if self.loop_counter % self.control_decimation == 0:
            raw_action, target_delta = self.get_action()
            self.current_action[:] = raw_action
            self.current_target_delta[:] = target_delta
            self.prev_action[:] = raw_action
            self.arm_joint_positions[:NUM_TRACKED_ARM_JOINTS] = (
                self.default_joint_angles[ARM_MOTOR_START:ARM_MOTOR_START + NUM_TRACKED_ARM_JOINTS]
                + target_delta[ARM_MOTOR_START:ARM_MOTOR_START + NUM_TRACKED_ARM_JOINTS]
            )
            self.arm_joint_positions[NUM_TRACKED_ARM_JOINTS] = GRIPPER_DEFAULT
            if self.z1_arm is not None:
                self.z1_arm.set_target_positions(self.arm_joint_positions)

        for i in range(NUM_LEG_ACTIONS):
            self.cmd.motor_cmd[i].q = float(self.current_target_delta[i] + self.default_joint_angles[i])
            self.cmd.motor_cmd[i].kp = float(self.kps[i])
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kd = float(self.kds[i])
            self.cmd.motor_cmd[i].tau = 0.0

        if self.z1_arm is None:
            for i in range(ARM_MOTOR_START, ARM_MOTOR_START + NUM_TRACKED_ARM_JOINTS):
                self.cmd.motor_cmd[i].q = float(self.current_target_delta[i] + self.default_joint_angles[i])
                self.cmd.motor_cmd[i].kp = float(self.kps[i])
                self.cmd.motor_cmd[i].dq = 0.0
                self.cmd.motor_cmd[i].kd = float(self.kds[i])
                self.cmd.motor_cmd[i].tau = 0.0
            self.cmd.motor_cmd[NUM_MOTORS - 1].q = float(GRIPPER_DEFAULT)
            self.cmd.motor_cmd[NUM_MOTORS - 1].kp = float(self.kps[NUM_MOTORS - 1])
            self.cmd.motor_cmd[NUM_MOTORS - 1].dq = 0.0
            self.cmd.motor_cmd[NUM_MOTORS - 1].kd = float(self.kds[NUM_MOTORS - 1])
            self.cmd.motor_cmd[NUM_MOTORS - 1].tau = 0.0
        # z1-sdk: arm is driven by Z1ArmClient; do not command arm slots on B2 lowcmd.

        self.safe_write_cmd()
        self.loop_counter += 1

    def estop_requested(self) -> bool:
        return self.key_state[self.key_index["R1"]][1] == 1 and self.key_state[self.key_index["L1"]][1] == 1

    def safe_write_cmd(self) -> None:
        if self.estop_requested():
            self.e_stop()
            raise SystemExit(0)
        self.cmd.crc = self.crc.Crc(self.cmd)
        self.low_cmd_puber.Write(self.cmd)

    def _passive_leg_cmd(self) -> None:
        motor_count = NUM_LEG_ACTIONS if self.z1_arm is not None else NUM_MOTORS
        for i in range(motor_count):
            self.cmd.motor_cmd[i].q = 0.0
            self.cmd.motor_cmd[i].kp = 0.0
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kd = 0.0
            self.cmd.motor_cmd[i].tau = 0.0
        self.cmd.crc = self.crc.Crc(self.cmd)
        self.low_cmd_puber.Write(self.cmd)

    def shutdown(self, *, passive_legs: bool = False) -> None:
        if passive_legs:
            self._passive_leg_cmd()
        if self.z1_arm is not None:
            self.z1_arm.stop()
            self.z1_arm = None

    def e_stop(self) -> None:
        self._passive_leg_cmd()
        self.shutdown()
        print("Emergency stop: R1+L1 pressed - B2 legs passive, Z1 released.")

    def stand_up(self) -> None:
        if self.z1_arm is not None:
            self.z1_arm.set_target_positions(self.arm_joint_positions)

        motor_count = NUM_LEG_ACTIONS if self.z1_arm is not None else NUM_MOTORS
        running_time = 0.0
        stand_up_joint_pos = self.default_joint_angles[:motor_count].astype(float).copy()
        stand_down_joint_pos = np.array(
            [self.low_state.motor_state[i].q for i in range(motor_count)], dtype=float
        )
        step_start = time.perf_counter()
        while running_time < 5.0:
            self.check_stop()
            if self.estop_requested():
                self.e_stop()
                raise SystemExit(0)
            running_time += 0.002
            phase = np.tanh(running_time / 1.2)
            for i in range(motor_count):
                self.cmd.motor_cmd[i].q = phase * stand_up_joint_pos[i] + (1.0 - phase) * stand_down_joint_pos[i]
                self.cmd.motor_cmd[i].kp = phase * float(self.stand_kps[i]) + (1.0 - phase) * 60.0
                self.cmd.motor_cmd[i].dq = 0.0
                self.cmd.motor_cmd[i].kd = float(self.kds[i])
                self.cmd.motor_cmd[i].tau = 0.0
            self.safe_write_cmd()
            self._interruptible_sleep(max(0.0, 0.002 - (time.perf_counter() - step_start)))
            step_start = time.perf_counter()

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

    def WirelessControllerHandler(self, msg: WirelessController_) -> None:
        old_x = self.key_state[self.key_index["X"]][1]
        old_y = self.key_state[self.key_index["Y"]][1]
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

    def LowStateHandler(self, msg: LowState_) -> None:
        self.low_state = msg


def _format_array(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{float(v):.4g}" for v in values.reshape(-1)) + "]"


def main() -> None:
    parser = argparse.ArgumentParser(description="Run B2+Z1 APEX one-step history policy on Unitree hardware.")
    parser.add_argument(
        "interface",
        nargs="?",
        default=None,
        help=(
            f"B2 DDS LAN interface (default: {DEFAULT_NETWORK_INTERFACE} for z1-sdk, "
            f"{SIM_NETWORK_INTERFACE} for unified/MuJoCo). Override: B2Z1_NETWORK_INTERFACE."
        ),
    )
    parser.add_argument(
        "--policy-path",
        default=str(DEFAULT_POLICY_PATH),
        help="B2+Z1 APEX one-step history ONNX policy path. Defaults to DEFAULT_POLICY_PATH in this file.",
    )
    parser.add_argument(
        "--motion-file",
        action="append",
        default=None,
        help="Reference NPZ file, directory, or glob. Can be passed multiple times.",
    )
    parser.add_argument(
        "--reference-offsets",
        default=os.environ.get("GURUKUL_APEX_REFERENCE_OFFSETS"),
        help="Comma-separated reference frame offsets. Default: 0,1.",
    )
    parser.add_argument("--joystick-command", action="store_true", help="Use remote sticks instead of motion commands.")
    parser.add_argument(
        "--arm-observation-mode",
        choices=["measured", "commanded"],
        default=os.environ.get("B2Z1_ARM_OBSERVATION_MODE", "measured"),
        help="measured: arm joint_pos obs from Z1 SDK or lowstate. commanded: from last policy target.",
    )
    parser.add_argument(
        "--arm-backend",
        choices=["z1-sdk", "unified"],
        default=os.environ.get("B2Z1_ARM_BACKEND", "z1-sdk"),
        help=(
            "z1-sdk: real hardware - B2 rt/lowcmd for 12 legs, Z1 via unitree_arm_interface. "
            "unified: MuJoCo sim - all 19 DOF on rt/lowcmd."
        ),
    )
    parser.add_argument(
        "--z1-lib-dir",
        default=os.environ.get("B2Z1_Z1_LIB_DIR"),
        help="Path to z1_sdk/lib (unitree_arm_interface.so). Default: ../z1_sdk/lib.",
    )
    args = parser.parse_args()
    signal.signal(signal.SIGINT, _sigint_handler)

    policy_path = Path(args.policy_path).expanduser().resolve()
    motion_patterns = args.motion_file or [str(DEFAULT_MOTION_GLOB)]
    motion_files = _resolve_motion_files(motion_patterns)
    reference_offsets = _parse_reference_offsets(args.reference_offsets)
    if reference_offsets != REFERENCE_OFFSETS:
        raise ValueError(
            f"This runner is for the one-step history student and expects offsets {REFERENCE_OFFSETS}; "
            f"got {reference_offsets}."
        )

    if not policy_path.is_file():
        print(f"[FAIL] Policy ONNX not found: {policy_path}")
        sys.exit(1)
    if not motion_files:
        print(f"[FAIL] No motion files matched: {motion_patterns}")
        sys.exit(1)
    if args.arm_backend == "z1-sdk" and not z1_controller_is_running():
        print("[FAIL] z1_ctrl is not running. Start the z1_controller build for your deployment checkout.")
        sys.exit(3)

    print("WARNING: B2+Z1 APEX hardware runner. Clear the workspace and keep R1+L1 for estop.")
    print(f"Policy: {policy_path}")
    print(f"Motion files: {len(motion_files)}")
    print(f"Reference offsets: {reference_offsets}")
    print(f"Arm backend: {args.arm_backend}")
    print(f"Arm observation mode: {args.arm_observation_mode}")
    print(f"Initial arm/gripper hold: {_format_array(RLController.default_joint_angles[ARM_MOTOR_START:])}")
    input("Press Enter to continue...")

    dds_iface = init_unitree_channel(args.interface, arm_backend=args.arm_backend)
    print(f"B2 DDS interface: {dds_iface}")

    motion_runtime = ApexMotionRuntime(
        motion_files,
        reference_offsets=reference_offsets,
        drive_command=not args.joystick_command,
    )
    z1_lib = Path(args.z1_lib_dir).expanduser().resolve() if args.z1_lib_dir else None
    controller: RLController | None = None
    controller = RLController(
        policy_path=policy_path,
        motion_runtime=motion_runtime,
        arm_observation_mode=args.arm_observation_mode,
        arm_backend=args.arm_backend,
        z1_lib_dir=z1_lib,
    )
    try:
        next_time = time.perf_counter() + controller.interval
        while True:
            controller.check_stop()
            controller.run()
            sleep_time = next_time - time.perf_counter()
            if sleep_time > 0.0:
                controller._interruptible_sleep(sleep_time)
            next_time += controller.interval
    except (KeyboardInterrupt, SystemExit):
        print("\nStopped by user (Ctrl+C or estop).", flush=True)
    finally:
        if controller is not None:
            controller.shutdown(passive_legs=True)
    print("[DONE] Exiting.", flush=True)
    os._exit(0)


if __name__ == "__main__":
    main()
