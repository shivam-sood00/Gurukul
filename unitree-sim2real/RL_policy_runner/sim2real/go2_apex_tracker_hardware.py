import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_POLICY_PATH = SCRIPT_DIR / "go2_apex_tracker_policy.onnx"
DEFAULT_MOTION_GLOB = SCRIPT_DIR / "apex_tracker_motions" / "*.npz"

GO2_TRACKER_JOINT_NAMES = (
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
)
DEFAULT_REFERENCE_OFFSETS = (0, 1)
LEGACY_REFERENCE_OFFSETS = (0, 1, 2, 5, 10)
REFERENCE_FRAME_DIM = 22

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


def _import_unitree_sdk():
    global ChannelFactoryInitialize, ChannelPublisher, ChannelSubscriber
    global MotionSwitcherClient, SportClient, unitree_go_msg_dds__LowCmd_
    global LowCmd_, LowState_, WirelessController_, CRC

    if ChannelFactoryInitialize is not None:
        return
    try:
        from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient as _MotionSwitcherClient
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
            "unitree_sdk2py is required to run the Go2 APEX tracker hardware controller. "
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


def quat_to_rot(quat):
    w, x, y, z = quat
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ],
        dtype=np.float32,
    )


def _resolve_motion_files(patterns):
    if isinstance(patterns, (str, os.PathLike)):
        patterns = [patterns]
    paths = []
    for pattern in patterns:
        matches = sorted(glob.glob(str(Path(pattern).expanduser()), recursive=True))
        for match in matches:
            path = Path(match)
            if path.is_dir():
                paths.extend(sorted(path.rglob("*.npz")))
            elif path.is_file():
                paths.append(path)
    return list(dict.fromkeys(path.resolve() for path in paths))


def _onnx_input_width(session):
    shape = session.get_inputs()[0].shape
    dims = [dim for dim in shape if isinstance(dim, int) and dim > 0]
    return dims[-1] if dims else None


def _feed_dict(session, obs):
    meta = session.get_inputs()[0]
    obs = np.asarray(obs, dtype=np.float32).reshape(-1)
    if len(meta.shape) == 2:
        obs = obs.reshape(1, -1)
    elif len(meta.shape) == 3:
        obs = obs.reshape(1, 1, -1)
    return {meta.name: obs}


class ApexMotionRuntime:
    def __init__(self, motion_files, reference_offsets, drive_command=True):
        self.reference_offsets = tuple(int(offset) for offset in reference_offsets)
        self.drive_command = bool(drive_command)
        self.motions = [self._load_motion(path) for path in motion_files]
        if not self.motions:
            raise FileNotFoundError("No APEX tracker motion NPZ files found.")
        self.motion_id = 0
        self.local_frame = 0
        self.frame_accum = 0.0
        self._last_switch_key = None
        print(f"Loaded {len(self.motions)} APEX tracker motion(s).")
        self.print_status()

    @staticmethod
    def _load_motion(path):
        data = np.load(path)
        body_names = data["body_names"].tolist()
        joint_names = data["joint_names"].tolist()
        joint_name_to_idx = {name: idx for idx, name in enumerate(joint_names)}
        tracker_joint_indices = [joint_name_to_idx[name] for name in GO2_TRACKER_JOINT_NAMES]
        base_idx = body_names.index("base") if "base" in body_names else 0
        return {
            "path": Path(path),
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
    def active_motion(self):
        return self.motions[self.motion_id]

    def print_status(self):
        motion = self.active_motion
        frame_count = motion["joint_pos"].shape[0]
        print(
            f"Active reference {self.motion_id + 1}/{len(self.motions)}: "
            f"{motion['path'].name}, frame={self.local_frame}/{max(frame_count - 1, 0)}"
        )

    def switch_motion(self, delta):
        if len(self.motions) <= 1:
            return
        self.motion_id = (self.motion_id + int(delta)) % len(self.motions)
        self.local_frame = 0
        self.frame_accum = 0.0
        self.print_status()

    def update(self, dt, command_out):
        motion = self.active_motion
        frame_count = motion["joint_pos"].shape[0]
        self.frame_accum += float(dt) * motion["fps"]
        frame_step = int(self.frame_accum)
        if frame_step > 0:
            self.frame_accum -= frame_step
            self.local_frame = (self.local_frame + frame_step) % frame_count

        if not self.drive_command:
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

    def current_base_lin_vel_z(self):
        motion = self.active_motion
        return float(motion["body_lin_vel_w"][self.local_frame, motion["base_idx"], 2])

    def reference_features(self):
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
            -0.1,
            0.8,
            -1.5,
            0.1,
            0.8,
            -1.5,
            -0.1,
            1.0,
            -1.5,
            0.1,
            1.0,
            -1.5,
        ],
        dtype=np.float32,
    )
    mapping = np.arange(12, dtype=np.int32)
    frequency = 200.0
    simulation_dt = 1.0 / frequency
    control_decimation = 4
    action_clip = 4.0
    action_scale = 0.25
    kps = np.full(12, 20.0, dtype=np.float32)
    kds = np.full(12, 0.5, dtype=np.float32)
    gravity_vec = np.array([0.0, 0.0, -1.0], dtype=np.float32)
    cmd_limits = np.array([[0.0, 1.5], [-1.0e9, 1.0e9], [-1.0e9, 1.0e9]], dtype=np.float32)
    cmd_scale = np.array([1.0, 1.0, 1.0, 1.0], dtype=np.float32)

    key_state = [
        ["R1", 0],
        ["L1", 0],
        ["start", 0],
        ["select", 0],
        ["R2", 0],
        ["L2", 0],
        ["F1", 0],
        ["F2", 0],
        ["A", 0],
        ["B", 0],
        ["X", 0],
        ["Y", 0],
        ["up", 0],
        ["right", 0],
        ["down", 0],
        ["left", 0],
    ]
    key_index = {name: idx for idx, (name, _) in enumerate(key_state)}

    def __init__(self, policy_path, motion_runtime, reference_base_dim):
        self.low_state = None
        self.motion_runtime = motion_runtime
        self.reference_base_dim = int(reference_base_dim)
        self.interval = 1.0 / self.frequency
        self.actor = ort.InferenceSession(str(policy_path))
        self.actor_output_names = [output.name for output in self.actor.get_outputs()]
        self.num_obs = _onnx_input_width(self.actor)
        if self.num_obs is None:
            raise RuntimeError(f"Could not infer ONNX input width from {policy_path}.")
        expected_obs = self.reference_base_dim + REFERENCE_FRAME_DIM * len(self.motion_runtime.reference_offsets)
        if self.num_obs != expected_obs:
            raise ValueError(
                f"Policy expects {self.num_obs} observations, but base_dim={self.reference_base_dim} "
                f"and reference_offsets={self.motion_runtime.reference_offsets} imply {expected_obs}."
            )
        print(f"Loaded actor from {policy_path}")
        print(f"Observation layout: {self.num_obs}D, base_dim={self.reference_base_dim}, offsets={self.motion_runtime.reference_offsets}")

        self.low_state_suber = ChannelSubscriber("rt/lowstate", LowState_)
        self.low_state_suber.Init(self.LowStateHandler, 10)
        self.low_cmd_puber = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.low_cmd_puber.Init()
        self.wireless_controller_suber = ChannelSubscriber("rt/wirelesscontroller", WirelessController_)
        self.wireless_controller_suber.Init(self.WirelessControllerHandler, 10)

        self.ang_vel = np.zeros(3, dtype=np.float32)
        self.projected_gravity = np.zeros(3, dtype=np.float32)
        self.command = np.zeros(3, dtype=np.float32)
        self.dof_pos = np.zeros(12, dtype=np.float32)
        self.dof_vel = np.zeros(12, dtype=np.float32)
        self.prev_action = np.zeros(12, dtype=np.float32)
        self.current_action = np.zeros(12, dtype=np.float32)
        self.current_target_delta = np.zeros(12, dtype=np.float32)
        self.loop_counter = 0
        self.j_lx = 0.0
        self.j_ly = 0.0
        self.j_rx = 0.0
        self.j_ry = 0.0

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

        status, result = self.msc.CheckMode()
        mode_name = result.get("name", "") if isinstance(result, dict) else ""
        if not isinstance(result, dict):
            print(f"[WARN] MotionSwitcher CheckMode returned {result!r} (status={status}).")
        while mode_name:
            self.sc.StandDown()
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            mode_name = result.get("name", "") if isinstance(result, dict) else ""
            time.sleep(1.0)

        if not self.wait_for_low_state(5.0):
            raise RuntimeError("No rt/lowstate messages received. Check the robot/simulator connection.")
        self.stand_up()

    def get_obs(self):
        self.ang_vel[:] = self.low_state.imu_state.gyroscope[:3]
        rot_matrix = quat_to_rot(self.low_state.imu_state.quaternion)
        self.projected_gravity[:] = rot_matrix.T @ self.gravity_vec

        for i in range(12):
            mapped_idx = self.mapping[i]
            motor = self.low_state.motor_state[i]
            self.dof_pos[mapped_idx] = motor.q - self.default_joint_angles[mapped_idx]
            self.dof_vel[mapped_idx] = motor.dq

        if self.motion_runtime.drive_command:
            self.motion_runtime.update(self.simulation_dt, self.command)
        else:
            self.command[0] = self.j_ly
            self.command[1] = self.j_lx
            self.command[2] = -self.j_rx

        if self.key_state[self.key_index["A"]][1] == 1 or self.key_state[self.key_index["B"]][1] == 1:
            self.command[:] = 0.0
        self.command[:] = np.clip(self.command, self.cmd_limits[:, 0], self.cmd_limits[:, 1])

        if self.reference_base_dim == 45:
            command_obs = self.command * self.cmd_scale[:3]
        else:
            command_obs = np.array(
                [
                    self.command[0] * self.cmd_scale[0],
                    self.command[1] * self.cmd_scale[1],
                    self.motion_runtime.current_base_lin_vel_z() * self.cmd_scale[2],
                    self.command[2] * self.cmd_scale[3],
                ],
                dtype=np.float32,
            )

        return np.concatenate(
            (
                self.ang_vel * 0.25,
                self.projected_gravity,
                command_obs,
                self.dof_pos,
                self.dof_vel * 0.05,
                self.prev_action,
                self.motion_runtime.reference_features(),
            ),
            dtype=np.float32,
        )

    def get_action(self):
        obs = self.get_obs()
        actions = self.actor.run(self.actor_output_names, _feed_dict(self.actor, obs))[0].reshape(-1).astype(np.float32)
        np.clip(actions, -self.action_clip, self.action_clip, out=actions)
        return actions, actions * self.action_scale

    def run(self):
        if self.estop_requested():
            self.e_stop()
            sys.exit(0)

        if self.loop_counter % self.control_decimation == 0:
            raw_action, target_delta = self.get_action()
            self.current_action[:] = raw_action
            self.current_target_delta[:] = target_delta
            self.prev_action[:] = raw_action

        for i, policy_idx in enumerate(self.mapping):
            self.cmd.motor_cmd[i].q = float(self.current_target_delta[policy_idx] + self.default_joint_angles[i])
            self.cmd.motor_cmd[i].kp = float(self.kps[i])
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kd = float(self.kds[i])
            self.cmd.motor_cmd[i].tau = 0.0

        self.safe_write_cmd()
        self.loop_counter += 1

    def estop_requested(self):
        return self.key_state[self.key_index["R1"]][1] == 1 and self.key_state[self.key_index["L1"]][1] == 1

    def safe_write_cmd(self):
        if self.estop_requested():
            self.e_stop()
            sys.exit(0)
        self.cmd.crc = self.crc.Crc(self.cmd)
        self.low_cmd_puber.Write(self.cmd)

    def e_stop(self):
        for i in range(12):
            self.cmd.motor_cmd[i].q = 0.0
            self.cmd.motor_cmd[i].kp = 0.0
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kd = 0.0
            self.cmd.motor_cmd[i].tau = 0.0
        self.cmd.crc = CRC().Crc(self.cmd)
        self.low_cmd_puber.Write(self.cmd)
        print("Emergency stop: R1+L1 pressed, motors commanded passive.")

    def stand_up(self):
        stand_up_joint_pos = np.array(
            [
                -0.00571868,
                0.608813,
                -1.21763,
                0.00571868,
                0.608813,
                -1.21763,
                -0.00571868,
                0.608813,
                -1.21763,
                0.00571868,
                0.608813,
                -1.21763,
            ],
            dtype=np.float32,
        )
        stand_down_joint_pos = np.array(
            [
                -0.0473455,
                1.22187,
                -2.44375,
                0.0473455,
                1.22187,
                -2.44375,
                -0.0473455,
                1.22187,
                -2.44375,
                0.0473455,
                1.22187,
                -2.44375,
            ],
            dtype=np.float32,
        )
        running_time = 0.0
        step_start = time.perf_counter()
        while running_time < 5.0:
            if self.estop_requested():
                self.e_stop()
                sys.exit(0)
            running_time += 0.002
            phase = np.tanh(running_time / 1.2)
            for i in range(12):
                self.cmd.motor_cmd[i].q = float(phase * stand_up_joint_pos[i] + (1.0 - phase) * stand_down_joint_pos[i])
                self.cmd.motor_cmd[i].kp = float(phase * 50.0 + (1.0 - phase) * 20.0)
                self.cmd.motor_cmd[i].dq = 0.0
                self.cmd.motor_cmd[i].kd = 3.5
                self.cmd.motor_cmd[i].tau = 0.0
            self.safe_write_cmd()
            sleep_time = 0.002 - (time.perf_counter() - step_start)
            if sleep_time > 0.0:
                time.sleep(sleep_time)
            step_start = time.perf_counter()

    def wait_for_low_state(self, timeout_s):
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            if self.low_state is not None:
                return True
            if self.estop_requested():
                self.e_stop()
                sys.exit(0)
            time.sleep(0.05)
        return False

    def WirelessControllerHandler(self, msg):
        self.j_lx = msg.lx
        self.j_ly = msg.ly
        self.j_rx = msg.rx
        self.j_ry = msg.ry
        old_x = self.key_state[self.key_index["X"]][1]
        old_y = self.key_state[self.key_index["Y"]][1]
        for i in range(16):
            self.key_state[i][1] = (msg.keys & (1 << i)) >> i
        if old_y == 0 and self.key_state[self.key_index["Y"]][1] == 1:
            self.motion_runtime.switch_motion(delta=1)
        if old_x == 0 and self.key_state[self.key_index["X"]][1] == 1:
            self.motion_runtime.switch_motion(delta=-1)

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg


def _parse_reference_offsets(value):
    if value is None:
        return None
    return tuple(int(part.strip()) for part in value.split(",") if part.strip())


def main():
    parser = argparse.ArgumentParser(description="Run a Go2 APEX tracker ONNX policy on Unitree hardware.")
    parser.add_argument("interface", nargs="?", default=None, help="Network interface, for example eth0. Defaults to lo.")
    parser.add_argument(
        "--policy-path",
        default=os.environ.get("GURUKUL_POLICY_PATH", str(DEFAULT_POLICY_PATH)),
        help="APEX tracker ONNX policy path.",
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
        help="Comma-separated tracker reference frame offsets. Defaults are inferred from policy input width.",
    )
    parser.add_argument("--joystick-command", action="store_true", help="Use remote sticks instead of motion commands.")
    args = parser.parse_args()

    policy_path = Path(args.policy_path).expanduser().resolve()
    if not policy_path.is_file():
        raise FileNotFoundError(f"Policy file not found: {policy_path}")

    probe_session = ort.InferenceSession(str(policy_path))
    input_width = _onnx_input_width(probe_session)
    reference_offsets = _parse_reference_offsets(args.reference_offsets)
    if reference_offsets is None:
        if input_width == 90:
            reference_offsets = DEFAULT_REFERENCE_OFFSETS
            reference_base_dim = 46
        elif input_width == 155:
            reference_offsets = LEGACY_REFERENCE_OFFSETS
            reference_base_dim = 45
        elif input_width == 156:
            reference_offsets = LEGACY_REFERENCE_OFFSETS
            reference_base_dim = 46
        else:
            raise ValueError(
                f"Cannot infer APEX tracker reference offsets for ONNX input width {input_width}; "
                "pass --reference-offsets."
            )
    else:
        base_dim = input_width - REFERENCE_FRAME_DIM * len(reference_offsets)
        if base_dim not in (45, 46):
            raise ValueError(
                f"Policy width {input_width} and offsets {reference_offsets} imply unsupported base dim {base_dim}."
            )
        reference_base_dim = base_dim

    motion_patterns = args.motion_file or [str(DEFAULT_MOTION_GLOB)]
    motion_files = _resolve_motion_files(motion_patterns)
    motion_runtime = ApexMotionRuntime(
        motion_files,
        reference_offsets=reference_offsets,
        drive_command=not args.joystick_command,
    )

    print("WARNING: Check robot clearance and keep R1+L1 available for emergency stop.")
    input("Press Enter to continue...")
    _import_unitree_sdk()
    if args.interface is None:
        ChannelFactoryInitialize(1, "lo")
    else:
        ChannelFactoryInitialize(0, args.interface)

    controller = RLController(policy_path, motion_runtime, reference_base_dim)
    while True:
        start = time.time()
        controller.run()
        end = time.time()
        time.sleep(max(0.0, controller.interval + start - end))


if __name__ == "__main__":
    main()
