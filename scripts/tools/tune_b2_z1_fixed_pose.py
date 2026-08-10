# SPDX-License-Identifier: Apache-2.0

"""Interactively tune the fixed B2 + Z1 arm start pose.

Type commands in the terminal while the Isaac Sim viewer is open:

* ``2=0`` sets ``joint2`` to 0 degrees.
* ``j2 0`` and ``joint2 0`` are equivalent.
* ``pose 0 0 0 0 0 0`` sets the six arm angles in degrees.
* ``select 3`` selects ``joint3`` for incremental edits.
* ``+`` or ``-`` nudges the selected joint by ``--step-deg``.
* ``show`` prints a copy/pasteable pose.
* ``reset`` restores the starting pose.
* ``quit`` exits.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import queue
import sys
import threading
import time

from isaaclab.app import AppLauncher


DEFAULT_ARM_POSE_DEG = {
    "joint1": 0.0,
    "joint2": 0.0,
    "joint3": 0.0,
    "joint4": 0.0,
    "joint5": 0.0,
    "joint6": 0.0,
    "jointGripper": -1.2,
}


parser = argparse.ArgumentParser(description="Tune the fixed B2 + Z1 arm start pose.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--base-height", type=float, default=0.75, help="Fixed B2 base height in meters.")
parser.add_argument("--step-deg", type=float, default=2.5, help="Increment used by + and - commands.")
parser.add_argument("--real-time", action="store_true", default=False, help="Throttle stepping to real time.")
parser.add_argument("--max-steps", type=int, default=None, help="Optional maximum number of sim steps before exit.")
parser.add_argument(
    "--pose",
    "--pose-deg",
    dest="pose",
    type=float,
    nargs="+",
    metavar="VALUE",
    help="Initial pose: 6 arm angles in degrees, optionally followed by gripper angle in radians.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg

from Gurukul.assets.unitree import UNITREE_B2_Z1_ARM_CFG


ARM_JOINTS = tuple(DEFAULT_ARM_POSE_DEG)
REVOLUTE_ARM_JOINTS = ARM_JOINTS[:6]
GRIPPER_JOINTS = ARM_JOINTS[6:]
LEG_POSE = {
    "FL_hip_joint": 0.0,
    "FL_thigh_joint": 0.8,
    "FL_calf_joint": -1.5,
    "FR_hip_joint": 0.0,
    "FR_thigh_joint": 0.8,
    "FR_calf_joint": -1.5,
    "RL_hip_joint": 0.0,
    "RL_thigh_joint": 0.8,
    "RL_calf_joint": -1.5,
    "RR_hip_joint": 0.0,
    "RR_thigh_joint": 0.8,
    "RR_calf_joint": -1.5,
}


def _make_scene_cfg() -> InteractiveSceneCfg:
    robot_cfg = UNITREE_B2_Z1_ARM_CFG.replace(prim_path="/World/Robot")
    robot_cfg.init_state.pos = (0.0, 0.0, args_cli.base_height)
    if robot_cfg.spawn.articulation_props is not None:
        robot_cfg.spawn.articulation_props.fix_root_link = True
    robot_cfg.init_state.joint_pos.update(LEG_POSE)
    if args_cli.pose is not None:
        if len(args_cli.pose) not in (6, 7):
            raise ValueError("--pose expects 6 arm angles in degrees, optionally followed by gripper angle in radians.")
        robot_cfg.init_state.joint_pos.update(
            {name: torch.deg2rad(torch.tensor(deg)).item() for name, deg in zip(REVOLUTE_ARM_JOINTS, args_cli.pose[:6])}
        )
        if len(args_cli.pose) == 7:
            robot_cfg.init_state.joint_pos.update({name: value for name, value in zip(GRIPPER_JOINTS, args_cli.pose[6:])})

    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    scene_cfg.robot = robot_cfg
    return scene_cfg


def _print_pose(pose_rad: dict[str, float]) -> None:
    print("\nCurrent Z1 pose:")
    for name in REVOLUTE_ARM_JOINTS:
        print(f"  {name}: {torch.rad2deg(torch.tensor(pose_rad[name])).item():8.3f} deg")
    for name in GRIPPER_JOINTS:
        print(f"  {name}: {pose_rad[name]:8.4f} rad")
    print("\nPython radians:")
    for name in REVOLUTE_ARM_JOINTS:
        print(f'    "{name}": {pose_rad[name]:.6f},')
    for name in GRIPPER_JOINTS:
        print(f'    "{name}": {pose_rad[name]:.6f},')
    print("")


def _print_help() -> None:
    print(
        """
Commands:
  2=-75                 set joint2 to -75 deg
  j2 -75                same as above
  joint2 -75            same as above
  arm2=-75              same as above
  gripper -1.2          set jointGripper to -1.2 rad
  pose 0 0 0 0 0 0
                         set joint1..joint6 in degrees
  pose 0 0 0 0 0 0 -1.2
                         set arm angles in degrees, gripper in radians
  select 3              select joint3
  +                     increase selected joint by --step-deg
  -                     decrease selected joint by --step-deg
  show                  print current pose
  reset                 restore the initial pose
  help                  show this help
  quit                  exit
"""
    )


def _start_input_thread() -> queue.Queue[str]:
    command_queue: queue.Queue[str] = queue.Queue()

    def _reader() -> None:
        while True:
            line = sys.stdin.readline()
            if line == "":
                break
            command_queue.put(line)

    thread = threading.Thread(target=_reader, daemon=True)
    thread.start()
    return command_queue


def _apply_pose(robot, joint_ids: torch.Tensor, pose_rad: dict[str, float], device: str) -> None:
    joint_pos = robot.data.default_joint_pos.clone()
    for local_idx, name in enumerate(ARM_JOINTS):
        joint_pos[0, joint_ids[local_idx]] = pose_rad[name]
    joint_vel = torch.zeros_like(joint_pos)
    env_ids = torch.tensor([0], device=device)
    robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)
    robot.set_joint_position_target(joint_pos)


def _parse_joint_token(token: str) -> str | None:
    token = token.lower().strip()
    if token.endswith("_joint") and token in ARM_JOINTS:
        return token
    if token in ARM_JOINTS:
        return token
    if token.startswith("arm") and not token.startswith("arm_"):
        token = token[3:]
    if token.startswith("arm_") and not token.endswith("_joint"):
        token = token[4:]
    if token.startswith("arm_") and token.endswith("_joint") and token in ARM_JOINTS:
        return token
    if token.startswith("j"):
        token = token[1:]
    if token in {"1", "2", "3", "4", "5", "6"}:
        return f"joint{token}"
    if token in {"7", "gripper", "g", "g1"}:
        return "jointGripper"
    return None


def _set_joint_value(pose_rad: dict[str, float], joint_name: str, value: float) -> None:
    if joint_name in GRIPPER_JOINTS:
        pose_rad[joint_name] = value
    else:
        pose_rad[joint_name] = torch.deg2rad(torch.tensor(value)).item()


def _format_joint_value(pose_rad: dict[str, float], joint_name: str) -> str:
    if joint_name in GRIPPER_JOINTS:
        return f"{pose_rad[joint_name]:.4f} rad"
    return f"{torch.rad2deg(torch.tensor(pose_rad[joint_name])).item():.3f} deg"


def _handle_command(line: str, pose_rad: dict[str, float], selected: str) -> tuple[str, bool]:
    line = line.strip()
    if "=" in line and not line.lower().startswith("pose"):
        left, right = line.split("=", 1)
        line = f"{left.strip()} {right.strip()}"
    parts = line.strip().split()
    if not parts:
        return selected, False

    cmd = parts[0].lower()
    if cmd in {"quit", "q", "exit"}:
        return selected, True
    if cmd in {"help", "h", "?"}:
        _print_help()
        return selected, False
    if cmd in {"print", "p", "show"}:
        _print_pose(pose_rad)
        return selected, False
    if cmd == "pose":
        if len(parts) not in (len(REVOLUTE_ARM_JOINTS) + 1, len(ARM_JOINTS) + 1):
            print("[WARN] pose expects 6 arm angles in degrees, optionally followed by gripper angle in radians.")
            return selected, False
        try:
            values = [float(value) for value in parts[1:]]
        except ValueError:
            print("[WARN] pose values must be numbers.")
            return selected, False
        for name, deg in zip(REVOLUTE_ARM_JOINTS, values[:6]):
            pose_rad[name] = torch.deg2rad(torch.tensor(deg)).item()
        if len(values) == 7:
            for name, value in zip(GRIPPER_JOINTS, values[6:]):
                pose_rad[name] = value
        _print_pose(pose_rad)
        return selected, False
    if cmd in {"reset", "r"}:
        for name, deg in list(DEFAULT_ARM_POSE_DEG.items())[:6]:
            pose_rad[name] = torch.deg2rad(torch.tensor(deg)).item()
        for name, value in list(DEFAULT_ARM_POSE_DEG.items())[6:]:
            pose_rad[name] = value
        print("[INFO] Reset pose.")
        return selected, False
    if cmd in {"select", "s"} and len(parts) >= 2:
        joint_name = _parse_joint_token(parts[1])
        if joint_name is not None:
            print(f"[INFO] Selected {joint_name}.")
            return joint_name, False
        print(f"[WARN] Unknown joint: {parts[1]}")
        return selected, False
    if cmd in {"+", "-"}:
        delta = args_cli.step_deg if cmd == "+" else -args_cli.step_deg
        if selected in GRIPPER_JOINTS:
            pose_rad[selected] += 0.001 if cmd == "+" else -0.001
        else:
            pose_rad[selected] += torch.deg2rad(torch.tensor(delta)).item()
        print(f"[INFO] {selected} = {_format_joint_value(pose_rad, selected)}")
        return selected, False

    joint_name = _parse_joint_token(cmd)
    if joint_name is not None:
        if len(parts) == 1:
            print(f"[INFO] Selected {joint_name}.")
            return joint_name, False
        try:
            value = float(parts[1])
        except ValueError:
            print(f"[WARN] Expected numeric value after {parts[0]!r}.")
            return selected, False
        _set_joint_value(pose_rad, joint_name, value)
        print(f"[INFO] {joint_name} = {_format_joint_value(pose_rad, joint_name)}")
        return joint_name, False

    print(f"[WARN] Unknown command: {line.strip()}")
    return selected, False


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, use_fabric=not args_cli.disable_fabric)
    sim = sim_utils.SimulationContext(sim_cfg)
    sim.set_camera_view([1.6, -2.2, 1.4], [0.0, 0.0, 0.45])

    ground_cfg = sim_utils.GroundPlaneCfg()
    ground_cfg.func("/World/defaultGroundPlane", ground_cfg)
    light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)

    scene = InteractiveScene(_make_scene_cfg())
    sim.reset()
    scene.reset()

    robot = scene["robot"]
    joint_ids = torch.tensor(robot.find_joints(list(ARM_JOINTS), preserve_order=True)[0], device=args_cli.device)
    pose_rad = {name: float(robot.data.default_joint_pos[0, joint_ids[idx]].item()) for idx, name in enumerate(ARM_JOINTS)}
    selected = "joint2"

    print("[INFO] B2+Z1 fixed-pose tuner", flush=True)
    print("[INFO] Commands: 2=-75 | pose 0 0 0 0 0 0 | + | - | show | help | quit", flush=True)
    _print_pose(pose_rad)
    command_queue = _start_input_thread()

    step_count = 0
    wall_t0 = time.time()
    while True:
        if args_cli.max_steps is not None and step_count >= args_cli.max_steps:
            print(f"[INFO] Reached --max-steps={args_cli.max_steps}.", flush=True)
            return
        if args_cli.max_steps is None and args_cli.headless and not simulation_app.is_running():
            print("[INFO] Headless app stopped.", flush=True)
            return

        while not command_queue.empty():
            selected, should_quit = _handle_command(command_queue.get_nowait(), pose_rad, selected)
            if should_quit:
                return
            _apply_pose(robot, joint_ids, pose_rad, args_cli.device)

        _apply_pose(robot, joint_ids, pose_rad, args_cli.device)
        scene.write_data_to_sim()
        sim.step(render=not bool(args_cli.headless))
        scene.update(dt=sim.get_physics_dt())

        step_count += 1
        if args_cli.real_time:
            sleep_time = step_count * sim.get_physics_dt() - (time.time() - wall_t0)
            if sleep_time > 0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
