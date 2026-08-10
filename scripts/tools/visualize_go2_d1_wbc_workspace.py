# SPDX-License-Identifier: Apache-2.0

"""Visualize the full Go2+D1 WBC end-effector command workspace.

The Go2 base is fixed and zero policy actions are sent. The D1 arm is moved by
the same differential-IK target machinery used by the WBC command generator.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import itertools
import time

from isaaclab.app import AppLauncher


DEFAULT_TASK = "Gurukul-Isaac-WBC-Flat-Unitree-Go2-D1-Arm-v0"


parser = argparse.ArgumentParser(description="Sweep the Go2+D1 WBC arm command workspace.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="WBC task to instantiate.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to visualize.")
parser.add_argument("--base-height", type=float, default=0.75, help="Fixed Go2 base height in meters.")
parser.add_argument("--grid-size", type=int, default=3, help="Number of samples per axis for the workspace grid.")
parser.add_argument("--hold-s", type=float, default=1.25, help="Seconds to hold each waypoint after reaching it.")
parser.add_argument("--speed", type=float, default=0.08, help="Nominal end-effector interpolation speed in m/s.")
parser.add_argument(
    "--workspace-fraction",
    type=float,
    default=1.0,
    help="Workspace fraction to visualize. Use 1.0 for the full WBC command box.",
)
parser.add_argument(
    "--corners-only",
    action="store_true",
    default=False,
    help="Only visit center, face centers, and corners instead of the full grid.",
)
parser.add_argument(
    "--no-clamp",
    action="store_true",
    default=False,
    help="Do not apply the WBC body-exclusion clamp to requested waypoints.",
)
parser.add_argument(
    "--max-cycles",
    type=int,
    default=None,
    help="Optional number of full workspace cycles before exit.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Throttle stepping to real time.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import Gurukul.tasks  # noqa: F401
import torch

import isaaclab_tasks  # noqa: F401
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import FRAME_MARKER_CFG
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import combine_frame_transforms, subtract_frame_transforms
from isaaclab_tasks.utils import parse_env_cfg

from Gurukul.tasks.manager_based.locomotion.velocity.mdp.events import (
    _clamp_arm_ee_target,
    _scale_arm_ee_range,
)


def _configure_env(env_cfg) -> None:
    """Fix the base and make the WBC IK event drive the arm directly."""
    if hasattr(env_cfg.scene.robot.spawn, "fix_base"):
        env_cfg.scene.robot.spawn.fix_base = True
    if getattr(env_cfg.scene.robot.spawn, "articulation_props", None) is not None:
        env_cfg.scene.robot.spawn.articulation_props.fix_root_link = True
    env_cfg.scene.robot.init_state.pos = (0.0, 0.0, args_cli.base_height)

    for event_name in (
        "randomize_reset_base",
        "randomize_push_robot",
        "base_external_force_torque",
        "push_robot",
    ):
        if hasattr(env_cfg.events, event_name):
            setattr(env_cfg.events, event_name, None)

    if hasattr(env_cfg.events, "randomize_arm_command"):
        env_cfg.events.randomize_arm_command = None
    if hasattr(env_cfg.events, "advance_arm_command"):
        env_cfg.events.advance_arm_command.params["apply_target"] = True

    # The normal WBC policy action owns legs, arm, and gripper. For this
    # workspace sweep, leave only legs in the action term so the IK event can
    # be the sole writer for D1 arm joint targets.
    if hasattr(env_cfg.actions, "joint_pos") and hasattr(env_cfg, "leg_joint_names"):
        env_cfg.actions.joint_pos.joint_names = env_cfg.leg_joint_names
        env_cfg.actions.joint_pos.scale = {
            r"^(FL|FR|RL|RR)_hip_joint$": 0.125,
            r"^(FL|FR|RL|RR)_(thigh|calf)_joint$": 0.25,
        }
        # The WBC task normally clips arm and gripper targets too. Once this
        # diagnostic narrows the action term to legs, those regexes no longer
        # have matching joints and Isaac Lab rejects the action configuration.
        env_cfg.actions.joint_pos.clip = {".*": (-100.0, 100.0)}

    for curriculum_name in ("loco_manipulation_training_stages", "command_levels_lin_vel", "command_levels_ang_vel"):
        if hasattr(env_cfg.curriculum, curriculum_name):
            setattr(env_cfg.curriculum, curriculum_name, None)

    if hasattr(env_cfg.commands, "base_velocity"):
        env_cfg.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        env_cfg.commands.base_velocity.ranges.heading = (0.0, 0.0)
        env_cfg.commands.base_velocity.heading_command = False
        env_cfg.commands.base_velocity.rel_standing_envs = 1.0


def _linspace_pair(low: float, high: float, count: int) -> list[float]:
    if count <= 1:
        return [(low + high) * 0.5]
    return torch.linspace(low, high, count).tolist()


def _workspace_waypoints(
    ee_pos_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> list[tuple[float, float, float]]:
    center = tuple((low + high) * 0.5 for low, high in ee_pos_range)
    corners = list(itertools.product(*[(low, high) for low, high in ee_pos_range]))
    face_centers = []
    for axis in range(3):
        for side in range(2):
            point = list(center)
            point[axis] = ee_pos_range[axis][side]
            face_centers.append(tuple(point))

    if args_cli.corners_only:
        requested = [center, *face_centers, *corners, center]
    else:
        samples = [_linspace_pair(low, high, max(2, args_cli.grid_size)) for low, high in ee_pos_range]
        grid = []
        for ix, x in enumerate(samples[0]):
            ys = samples[1] if ix % 2 == 0 else list(reversed(samples[1]))
            for iy, y in enumerate(ys):
                zs = samples[2] if (ix + iy) % 2 == 0 else list(reversed(samples[2]))
                grid.extend((x, y, z) for z in zs)
        requested = [center, *face_centers, *corners, *grid, center]

    unique = []
    seen = set()
    for point in requested:
        key = tuple(round(float(value), 4) for value in point)
        if key in seen:
            continue
        seen.add(key)
        unique.append(tuple(float(value) for value in point))
    return unique


def _make_target_marker(path: str, scale: tuple[float, float, float]) -> VisualizationMarkers:
    cfg = FRAME_MARKER_CFG.replace()
    cfg.markers["frame"].scale = scale
    return VisualizationMarkers(cfg.replace(prim_path=path))


def _visualize_frames(env, robot, ee_body_id: int, target_pos_b: torch.Tensor, target_quat_b: torch.Tensor) -> None:
    if not hasattr(env, "_wbc_workspace_target_marker"):
        env._wbc_workspace_target_marker = _make_target_marker("/Visuals/wbc_workspace_target", (0.16, 0.16, 0.16))
        env._wbc_workspace_current_marker = _make_target_marker("/Visuals/wbc_workspace_current", (0.10, 0.10, 0.10))

    root_pose_w = robot.data.root_pose_w
    target_pos_w, target_quat_w = combine_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], target_pos_b, target_quat_b
    )
    ee_pose_w = robot.data.body_pose_w[:, ee_body_id]
    env._wbc_workspace_target_marker.visualize(translations=target_pos_w, orientations=target_quat_w)
    env._wbc_workspace_current_marker.visualize(translations=ee_pose_w[:, 0:3], orientations=ee_pose_w[:, 3:7])


def _set_waypoint(env, target_pos_b: torch.Tensor, speed: float) -> None:
    initialized = env._arm_ee_target_initialized
    current_target = env._arm_ee_target_pos.clone()
    env._arm_ee_start_pos[initialized] = current_target[initialized]
    env._arm_ee_goal_pos[initialized] = target_pos_b[initialized]
    distance = torch.linalg.norm(env._arm_ee_goal_pos - env._arm_ee_start_pos, dim=1)
    env._arm_trajectory_duration[:] = torch.clamp(distance / max(float(speed), 1.0e-6), min=1.0)
    env._arm_trajectory_progress[:] = 0.0


def _ee_error_b(robot, ee_body_id: int, target_pos_b: torch.Tensor) -> torch.Tensor:
    ee_pose_w = robot.data.body_pose_w[:, ee_body_id]
    root_pose_w = robot.data.root_pose_w
    ee_pos_b, _ = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    return torch.linalg.norm(target_pos_b - ee_pos_b, dim=1)


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    _configure_env(env_cfg)

    reset_arm_event = getattr(env_cfg.events, "reset_arm_command", None)
    if reset_arm_event is None:
        raise RuntimeError(f"Task '{args_cli.task}' does not define reset_arm_command.")
    arm_params = reset_arm_event.params
    raw_range = arm_params["ee_pos_range"]
    ee_pos_range = _scale_arm_ee_range(
        raw_range,
        difficulty=float(args_cli.workspace_fraction),
        neutral_pos=arm_params.get("neutral_pos", (0.42, 0.0, 0.34)),
        min_fraction=0.0,
    )

    waypoints = _workspace_waypoints(ee_pos_range)
    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Full WBC raw workspace: {raw_range}")
    print(f"[INFO] Visualized workspace fraction: {args_cli.workspace_fraction:.2f} -> {ee_pos_range}")
    print(f"[INFO] Waypoint count: {len(waypoints)}")
    print("[INFO] Target frame is large; current EE frame is small.")

    env = gym.make(args_cli.task, cfg=env_cfg)
    env_unwrapped = env.unwrapped
    robot = env_unwrapped.scene["robot"]
    arm_cfg: SceneEntityCfg = arm_params["asset_cfg"]
    ee_body_id = int(robot.find_bodies(arm_cfg.body_names)[0][0])

    env.reset()
    env_unwrapped._loco_manip_arm_motion_enabled = True
    env_unwrapped._loco_manip_arm_motion_difficulty = float(args_cli.workspace_fraction)
    env_unwrapped._arm_ee_target_initialized[:] = True

    target_quat_b = env_unwrapped._arm_ee_target_quat.clone()
    actions = torch.zeros(env.action_space.shape, device=env_unwrapped.device)
    hold_steps = max(1, int(round(float(args_cli.hold_s) / env_unwrapped.step_dt)))
    cycle = 0
    waypoint_index = 0
    wall_t0 = time.time()
    step_count = 0

    try:
        deployment_waypoints = tuple(arm_params.get("deployment_ee_waypoints", ()))
        if deployment_waypoints:
            print(f"[INFO] Executing {len(deployment_waypoints)} carry-to-workspace deployment segments.")
        for deployment_index, waypoint in enumerate(deployment_waypoints):
            deployment_target = torch.tensor(
                waypoint,
                device=env_unwrapped.device,
                dtype=torch.float32,
            ).repeat(env_unwrapped.num_envs, 1)
            _set_waypoint(env_unwrapped, deployment_target, args_cli.speed)
            deployment_steps = max(
                1,
                int(
                    round(
                        (float(torch.max(env_unwrapped._arm_trajectory_duration).item()) + 0.25)
                        / env_unwrapped.step_dt
                    )
                ),
            )
            for _ in range(deployment_steps):
                with torch.inference_mode():
                    env.step(actions)
                    _visualize_frames(
                        env_unwrapped,
                        robot,
                        ee_body_id,
                        env_unwrapped._arm_ee_target_pos,
                        target_quat_b,
                    )
                step_count += 1
            error = _ee_error_b(robot, ee_body_id, deployment_target)
            print(
                f"[DEPLOY] {deployment_index + 1}/{len(deployment_waypoints)} "
                f"error_mean={torch.mean(error).item():.4f}m error_max={torch.max(error).item():.4f}m"
            )

        while simulation_app.is_running():
            if args_cli.max_cycles is not None and cycle >= args_cli.max_cycles:
                break

            requested = torch.tensor(waypoints[waypoint_index], device=env_unwrapped.device, dtype=torch.float32)
            commanded = requested.clone()
            if not args_cli.no_clamp:
                commanded = _clamp_arm_ee_target(
                    commanded,
                    ee_pos_range,
                    arm_params.get("body_exclusion_box", ((-0.30, 0.34), (-0.20, 0.20), (-0.02, 0.30))),
                    float(arm_params.get("body_clearance", 0.07)),
                    arm_params.get("extra_exclusion_boxes", ()),
                    arm_params.get("workspace_origin"),
                    arm_params.get("reach_range"),
                )
            target_pos_b = commanded.repeat(env_unwrapped.num_envs, 1)
            _set_waypoint(env_unwrapped, target_pos_b, args_cli.speed)

            for _ in range(hold_steps):
                if not simulation_app.is_running():
                    break
                with torch.inference_mode():
                    env.step(actions)
                    _visualize_frames(env_unwrapped, robot, ee_body_id, env_unwrapped._arm_ee_target_pos, target_quat_b)
                step_count += 1
                if args_cli.real_time:
                    sim_time = step_count * env_unwrapped.step_dt
                    sleep_time = sim_time - (time.time() - wall_t0)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

            error = _ee_error_b(robot, ee_body_id, target_pos_b)
            print(
                "[INFO] "
                f"cycle={cycle + 1} waypoint={waypoint_index + 1:03d}/{len(waypoints):03d} "
                f"requested=({requested[0]: .3f}, {requested[1]: .3f}, {requested[2]: .3f}) "
                f"commanded=({commanded[0]: .3f}, {commanded[1]: .3f}, {commanded[2]: .3f}) "
                f"mean_ee_error={torch.mean(error).item():.4f} m"
            )

            waypoint_index += 1
            if waypoint_index >= len(waypoints):
                waypoint_index = 0
                cycle += 1

    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
