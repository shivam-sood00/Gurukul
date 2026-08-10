# SPDX-License-Identifier: Apache-2.0

"""Visualize mounted-arm motion without training or loading a policy."""

"""Launch Isaac Sim Simulator first."""

import argparse
import inspect
import time

from isaaclab.app import AppLauncher

DEFAULT_TASK = "Gurukul-Isaac-Velocity-Flat-Unitree-Go2-Airbot-Arm-ArmMoving-v0"


parser = argparse.ArgumentParser(description="Visualize mounted-arm deployment-like motion.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="ArmMoving mounted-arm task to simulate.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument("--max_steps", type=int, default=None, help="Optional maximum number of env steps before exit.")
parser.add_argument("--real-time", action="store_true", default=False, help="Throttle stepping to real time.")
parser.add_argument(
    "--free-base",
    action="store_true",
    default=False,
    help="Leave the robot base free. By default the viewer fixes the robot in the air to isolate arm motion.",
)
parser.add_argument(
    "--base-height",
    type=float,
    default=0.75,
    help="Base height used when the robot is fixed in the air.",
)
parser.add_argument(
    "--interval",
    type=float,
    default=0.75,
    help="Seconds between sampled arm waypoints while visualizing.",
)
parser.add_argument(
    "--max-joint-change",
    type=float,
    default=0.35,
    help="Maximum arm joint target change per sampled waypoint.",
)
parser.add_argument(
    "--motion-speed",
    type=float,
    default=0.45,
    help="Nominal arm joint interpolation speed in rad/s.",
)
parser.add_argument(
    "--no-interpolation",
    action="store_true",
    default=False,
    help="Jump arm commands directly between sampled waypoints instead of interpolating targets.",
)
parser.add_argument(
    "--full-workspace",
    action="store_true",
    default=False,
    help="Use the configured full task-space arm box instead of curriculum-scaled workspace sampling.",
)
parser.add_argument(
    "--free-ee-orientation",
    action="store_true",
    default=False,
    help="Do not preserve the current end-effector orientation for task-space IK targets.",
)
parser.add_argument(
    "--primitive",
    action="append",
    choices=[
        "move",
        "pick_forward",
        "pick_low",
        "pick_high",
        "place",
        "pick_place",
        "reach",
        "sweep",
        "stow",
        "workspace",
    ],
    help="Restrict visualization to one or more primitives. Repeat the flag to include several.",
)
parser.add_argument(
    "--hold-pose-deg",
    type=float,
    nargs=6,
    metavar=("J1", "J2", "J3", "J4", "J5", "J6"),
    help="Hold a custom six-joint arm pose in degrees instead of the configured motion primitives.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import Gurukul.tasks  # noqa: F401
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def _configure_arm_visualization(env_cfg) -> None:
    """Force quicker arm waypoint sampling for inspection."""
    if not args_cli.free_base:
        if hasattr(env_cfg.scene.robot.spawn, "fix_base"):
            env_cfg.scene.robot.spawn.fix_base = True
        if getattr(env_cfg.scene.robot.spawn, "articulation_props", None) is not None:
            env_cfg.scene.robot.spawn.articulation_props.fix_root_link = True
        env_cfg.scene.robot.init_state.pos = (0.0, 0.0, args_cli.base_height)
        if hasattr(env_cfg.events, "randomize_reset_base"):
            env_cfg.events.randomize_reset_base = None
    for curriculum_name in ("loco_manipulation_training_stages",):
        if hasattr(env_cfg.curriculum, curriculum_name):
            setattr(env_cfg.curriculum, curriculum_name, None)

    arm_event = getattr(env_cfg.events, "randomize_push_robot", None)
    if arm_event is None:
        raise RuntimeError(f"Task '{args_cli.task}' does not define a mounted-arm motion event.")

    event_params = inspect.signature(arm_event.func).parameters
    arm_event.interval_range_s = (args_cli.interval, args_cli.interval)
    if "visualize" in event_params:
        arm_event.params["visualize"] = True
    if "start_motion_enabled" in event_params:
        arm_event.params["start_motion_enabled"] = True
    if "max_joint_change" in event_params:
        arm_event.params["max_joint_change"] = args_cli.max_joint_change
    if "max_pos_change" in event_params:
        arm_event.params["max_pos_change"] = args_cli.max_joint_change
    if "motion_speed" in event_params:
        arm_event.params["motion_speed"] = args_cli.motion_speed
    if "interpolate" in event_params:
        arm_event.params["interpolate"] = not args_cli.no_interpolation
    if args_cli.full_workspace and "min_workspace_fraction" in event_params:
        arm_event.params["min_workspace_fraction"] = 1.0
    if args_cli.free_ee_orientation and "preserve_current_orientation" in event_params:
        arm_event.params["preserve_current_orientation"] = False
    if args_cli.primitive and "motion_primitives" in event_params:
        arm_event.params["motion_primitives"] = tuple(args_cli.primitive)
    if args_cli.hold_pose_deg is not None:
        pose = tuple(torch.deg2rad(torch.tensor(args_cli.hold_pose_deg, dtype=torch.float32)).tolist())
        arm_event.params["motion_primitives"] = ("stow",)
        if "joint_motion_library" in arm_event.params:
            arm_event.params["joint_motion_library"] = {"stow": (pose,)}
        joint_names = tuple(getattr(arm_event.params.get("asset_cfg"), "joint_names", ()))
        for joint_name, value in zip(joint_names, pose):
            env_cfg.scene.robot.init_state.joint_pos[joint_name] = float(value)


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    _configure_arm_visualization(env_cfg)

    env = gym.make(args_cli.task, cfg=env_cfg)
    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Gym observation space: {env.observation_space}")
    print(f"[INFO] Gym action space: {env.action_space}")
    print("[INFO] Running zero leg actions; mounted arm is driven by configured motion primitives.")
    if not args_cli.free_base:
        print(f"[INFO] Robot base is fixed in the air at z={args_cli.base_height:.2f} m for visualization.")

    env.reset()
    step_count = 0
    wall_t0 = time.time()
    sim_t0 = 0.0

    while args_cli.max_steps is not None or simulation_app.is_running():
        if args_cli.max_steps is not None and step_count >= args_cli.max_steps:
            break

        with torch.inference_mode():
            actions = torch.zeros(env.action_space.shape, device=env.unwrapped.device)
            env.step(actions)

        step_count += 1
        if args_cli.real_time:
            sim_time = sim_t0 + step_count * env.unwrapped.step_dt
            sleep_time = sim_time - (time.time() - wall_t0)
            if sleep_time > 0:
                time.sleep(sleep_time)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
