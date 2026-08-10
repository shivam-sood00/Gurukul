"""Evaluate Go2+D1 WBC random end-effector commands and IK trackability."""

"""Launch Isaac Sim Simulator first."""

import argparse
import copy
import csv
import math
import time
from pathlib import Path

from isaaclab.app import AppLauncher

DEFAULT_TASK = "Gurukul-Isaac-WBC-Flat-Unitree-Go2-D1-Arm-v0"


parser = argparse.ArgumentParser(description="Evaluate random Go2+D1 WBC arm commands.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="WBC task to instantiate.")
parser.add_argument("--num_envs", type=int, default=32, help="Number of environments to sample in parallel.")
parser.add_argument("--samples", type=int, default=64, help="Number of random command batches to evaluate.")
parser.add_argument("--seed", type=int, default=42, help="Deterministic environment and command-sampling seed.")
parser.add_argument(
    "--warmup-samples",
    type=int,
    default=1,
    help="Initial sampled batches to report but exclude from post-warmup summary.",
)
parser.add_argument("--difficulty", type=float, default=1.0, help="Arm workspace/curriculum difficulty in [0, 1].")
parser.add_argument("--base-height", type=float, default=0.75, help="Fixed-base height used for isolated arm testing.")
parser.add_argument(
    "--base-rpy",
    type=float,
    nargs=3,
    default=(0.0, 0.0, 0.0),
    metavar=("ROLL", "PITCH", "YAW"),
    help="Fixed-base roll, pitch, and yaw in radians for Jacobian frame checks.",
)
parser.add_argument("--free-base", action="store_true", default=False, help="Leave the Go2 base floating/free.")
parser.add_argument("--settle-s", type=float, default=0.75, help="Extra settling time after each target trajectory.")
parser.add_argument("--max-hold-s", type=float, default=4.0, help="Maximum hold time per sampled target.")
parser.add_argument("--good-threshold", type=float, default=0.05, help="Good final EE error threshold in meters.")
parser.add_argument("--bad-threshold", type=float, default=0.10, help="Bad final EE error threshold in meters.")
parser.add_argument("--csv", type=str, default=None, help="Optional CSV path for per-sample metrics.")
parser.add_argument("--real-time", action="store_true", default=False, help="Throttle stepping to real time.")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

import gymnasium as gym
import Gurukul.tasks  # noqa: F401
import torch

from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

import isaaclab_tasks  # noqa: F401
from isaaclab_tasks.utils import parse_env_cfg


def _quat_wxyz_from_rpy(roll: float, pitch: float, yaw: float) -> tuple[float, float, float, float]:
    """Convert fixed-axis roll, pitch, yaw angles to a scalar-first quaternion."""
    half_roll = 0.5 * float(roll)
    half_pitch = 0.5 * float(pitch)
    half_yaw = 0.5 * float(yaw)
    cr, sr = math.cos(half_roll), math.sin(half_roll)
    cp, sp = math.cos(half_pitch), math.sin(half_pitch)
    cy, sy = math.cos(half_yaw), math.sin(half_yaw)
    return (
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    )


def _configure_env(env_cfg) -> None:
    """Isolate the arm command generator and let IK be the only D1 target writer."""
    env_cfg.seed = int(args_cli.seed)
    if not args_cli.free_base:
        if hasattr(env_cfg.scene.robot.spawn, "fix_base"):
            env_cfg.scene.robot.spawn.fix_base = True
        if getattr(env_cfg.scene.robot.spawn, "articulation_props", None) is not None:
            env_cfg.scene.robot.spawn.articulation_props.fix_root_link = True
        env_cfg.scene.robot.init_state.pos = (0.0, 0.0, float(args_cli.base_height))
        env_cfg.scene.robot.init_state.rot = _quat_wxyz_from_rpy(*args_cli.base_rpy)

    for event_name in (
        "randomize_rigid_body_material",
        "randomize_rigid_body_mass_base",
        "randomize_rigid_body_mass_others",
        "randomize_com_positions",
        "randomize_actuator_gains",
        "randomize_reset_base",
        "randomize_push_robot",
        "base_external_force_torque",
        "push_robot",
        "randomize_apply_external_force_torque",
    ):
        if hasattr(env_cfg.events, event_name):
            setattr(env_cfg.events, event_name, None)

    # A timeout during a target hold resets the robot and command buffers, then
    # makes the old sampled goal look like an IK failure. The isolated audit
    # owns its reset schedule, so automatic episode terminations must be off.
    for termination_name in ("time_out", "terrain_out_of_bounds"):
        if hasattr(env_cfg.terminations, termination_name):
            setattr(env_cfg.terminations, termination_name, None)

    if hasattr(env_cfg.events, "randomize_arm_command"):
        env_cfg.events.randomize_arm_command = None
    if hasattr(env_cfg.events, "advance_arm_command"):
        env_cfg.events.advance_arm_command.params["apply_target"] = True

    if hasattr(env_cfg.actions, "joint_pos") and hasattr(env_cfg, "leg_joint_names"):
        env_cfg.actions.joint_pos.joint_names = env_cfg.leg_joint_names
        env_cfg.actions.joint_pos.scale = {
            r"^(FL|FR|RL|RR)_hip_joint$": 0.125,
            r"^(FL|FR|RL|RR)_(thigh|calf)_joint$": 0.25,
        }
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


def _ee_pos_b(robot, ee_body_id: int) -> torch.Tensor:
    ee_pose_w = robot.data.body_pose_w[:, ee_body_id]
    root_pose_w = robot.data.root_pose_w
    ee_pos_b, _ = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    return ee_pos_b


def _target_error(robot, ee_body_id: int, target_pos_b: torch.Tensor) -> torch.Tensor:
    return torch.linalg.norm(target_pos_b - _ee_pos_b(robot, ee_body_id), dim=1)


def _inside_box(
    pos: torch.Tensor,
    box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
) -> torch.Tensor:
    mask = torch.ones(pos.shape[0], device=pos.device, dtype=torch.bool)
    for axis, (low, high) in enumerate(box):
        mask &= (pos[:, axis] >= float(low)) & (pos[:, axis] <= float(high))
    return mask


def _step(env, actions: torch.Tensor, steps: int, real_time_state: dict[str, float]) -> None:
    for _ in range(max(0, int(steps))):
        with torch.inference_mode():
            env.step(actions)
        real_time_state["steps"] += 1
        if args_cli.real_time:
            sim_time = real_time_state["steps"] * env.unwrapped.step_dt
            sleep_time = sim_time - (time.time() - real_time_state["wall_t0"])
            if sleep_time > 0.0:
                time.sleep(sleep_time)


def _write_csv(path: str, rows: list[dict[str, float]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(rows[0].keys()) if rows else []
    with output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    _configure_env(env_cfg)

    reset_arm_event = getattr(env_cfg.events, "reset_arm_command", None)
    advance_arm_event = getattr(env_cfg.events, "advance_arm_command", None)
    if reset_arm_event is None or advance_arm_event is None:
        raise RuntimeError(f"Task '{args_cli.task}' must define reset_arm_command and advance_arm_command.")
    reset_arm_func = reset_arm_event.func
    reset_arm_params = copy.deepcopy(reset_arm_event.params)
    raw_arm_cfg = reset_arm_params["asset_cfg"]
    reset_arm_params["asset_cfg"] = SceneEntityCfg(
        raw_arm_cfg.name,
        joint_names=list(getattr(env_cfg, "arm_joint_names", raw_arm_cfg.joint_names)),
        body_names=list(raw_arm_cfg.body_names or ["Link6"]),
        preserve_order=True,
    )
    reset_arm_params["asset_cfg"].joint_ids = None
    reset_arm_params["asset_cfg"].body_ids = None

    env = gym.make(args_cli.task, cfg=env_cfg)
    env_unwrapped = env.unwrapped
    robot = env_unwrapped.scene["robot"]
    arm_cfg: SceneEntityCfg = reset_arm_params["asset_cfg"]
    ee_body_id = int(robot.find_bodies(arm_cfg.body_names)[0][0])
    arm_joint_ids = torch.as_tensor(
        robot.find_joints(arm_cfg.joint_names, preserve_order=True)[0],
        device=env_unwrapped.device,
        dtype=torch.long,
    )
    env_ids = torch.arange(env_unwrapped.num_envs, device=env_unwrapped.device, dtype=torch.long)
    actions = torch.zeros(env.action_space.shape, device=env_unwrapped.device)

    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Envs: {env_unwrapped.num_envs}")
    print(f"[INFO] Samples: {args_cli.samples}")
    print(f"[INFO] Seed: {args_cli.seed}")
    print(f"[INFO] Difficulty: {args_cli.difficulty:.3f}")
    if not args_cli.free_base:
        print(f"[INFO] Fixed base RPY: {tuple(float(angle) for angle in args_cli.base_rpy)} rad")
    print(f"[INFO] EE range: {reset_arm_params['ee_pos_range']}")
    print(
        f"[INFO] Reach shell: origin={reset_arm_params.get('workspace_origin')} "
        f"range={reset_arm_params.get('reach_range')}"
    )
    print(f"[INFO] Body exclusion box: {reset_arm_params.get('body_exclusion_box')}")

    env.reset()
    env_unwrapped._loco_manip_arm_motion_enabled = True
    env_unwrapped._loco_manip_arm_motion_difficulty = float(args_cli.difficulty)
    real_time_state = {"steps": 0, "wall_t0": time.time()}
    deployment_waypoints = tuple(reset_arm_params.get("deployment_ee_waypoints", ()))
    reset_arm_params["reset_deployment"] = False
    if deployment_waypoints:
        print(f"[INFO] Executing {len(deployment_waypoints)} carry-to-workspace deployment segments before sampling.")
        for deployment_index in range(len(deployment_waypoints)):
            reset_arm_func(env_unwrapped, env_ids, **reset_arm_params)
            duration = float(torch.max(env_unwrapped._arm_trajectory_duration).item())
            steps = int(round((duration + float(args_cli.settle_s)) / env_unwrapped.step_dt))
            _step(env, actions, steps, real_time_state)
            target = env_unwrapped._arm_ee_goal_pos.clone()
            error = _target_error(robot, ee_body_id, target)
            print(
                f"[DEPLOY] {deployment_index + 1}/{len(deployment_waypoints)} "
                f"error_mean={torch.mean(error).item():.4f}m error_max={torch.max(error).item():.4f}m"
            )

    rows: list[dict[str, float]] = []
    all_final_errors = []
    all_interp_errors = []
    all_goal_steps = []

    try:
        for sample_idx in range(int(args_cli.samples)):
            reset_arm_func(env_unwrapped, env_ids, **reset_arm_params)
            duration = float(torch.max(env_unwrapped._arm_trajectory_duration).item())
            hold_time = min(duration + float(args_cli.settle_s), float(args_cli.max_hold_s))
            steps = int(round(hold_time / env_unwrapped.step_dt))
            _step(env, actions, steps, real_time_state)

            goal_pos = env_unwrapped._arm_ee_goal_pos.clone()
            target_pos = env_unwrapped._arm_ee_target_pos.clone()
            ee_pos = _ee_pos_b(robot, ee_body_id)
            final_error = torch.linalg.norm(goal_pos - ee_pos, dim=1)
            interp_error = torch.linalg.norm(target_pos - ee_pos, dim=1)
            goal_step = torch.linalg.norm(goal_pos - env_unwrapped._arm_ee_start_pos, dim=1)
            body_box = reset_arm_params.get("body_exclusion_box")
            goal_in_body = (
                _inside_box(goal_pos, body_box) if body_box is not None else torch.zeros_like(final_error).bool()
            )

            all_final_errors.append(final_error.detach().cpu())
            all_interp_errors.append(interp_error.detach().cpu())
            all_goal_steps.append(goal_step.detach().cpu())
            row = {
                "sample": float(sample_idx),
                "warmup": float(sample_idx < int(args_cli.warmup_samples)),
                "hold_steps": float(steps),
                "duration_s": duration,
                "goal_step_mean_m": float(torch.mean(goal_step).item()),
                "goal_step_max_m": float(torch.max(goal_step).item()),
                "final_error_mean_m": float(torch.mean(final_error).item()),
                "final_error_p90_m": float(torch.quantile(final_error, 0.90).item()),
                "final_error_max_m": float(torch.max(final_error).item()),
                "interp_error_mean_m": float(torch.mean(interp_error).item()),
                "goal_inside_body_frac": float(goal_in_body.float().mean().item()),
            }
            arm_joint_pos = robot.data.joint_pos.index_select(1, arm_joint_ids)
            for env_index in range(env_unwrapped.num_envs):
                command_row = {
                    "sample": float(sample_idx),
                    "env": float(env_index),
                    "warmup": row["warmup"],
                    "hold_steps": row["hold_steps"],
                    "duration_s": row["duration_s"],
                    "goal_x": float(goal_pos[env_index, 0].item()),
                    "goal_y": float(goal_pos[env_index, 1].item()),
                    "goal_z": float(goal_pos[env_index, 2].item()),
                    "ee_x": float(ee_pos[env_index, 0].item()),
                    "ee_y": float(ee_pos[env_index, 1].item()),
                    "ee_z": float(ee_pos[env_index, 2].item()),
                    "goal_step_m": float(goal_step[env_index].item()),
                    "final_error_m": float(final_error[env_index].item()),
                    "interp_error_m": float(interp_error[env_index].item()),
                    "goal_inside_body": float(goal_in_body[env_index].item()),
                }
                for joint_index, joint_name in enumerate(arm_cfg.joint_names):
                    command_row[joint_name] = float(arm_joint_pos[env_index, joint_index].item())
                rows.append(command_row)
            print(
                "[SAMPLE] "
                f"{sample_idx + 1:04d}/{int(args_cli.samples):04d} "
                f"step_mean={row['goal_step_mean_m']:.3f}m "
                f"err_mean={row['final_error_mean_m']:.4f}m "
                f"err_p90={row['final_error_p90_m']:.4f}m "
                f"err_max={row['final_error_max_m']:.4f}m "
                f"body_goal_frac={row['goal_inside_body_frac']:.3f}"
            )
    finally:
        env.close()

    def _print_summary(label: str, final_chunks: list[torch.Tensor], interp_chunks: list[torch.Tensor], step_chunks):
        errors = torch.cat(final_chunks) if final_chunks else torch.empty(0)
        interp_errors = torch.cat(interp_chunks) if interp_chunks else torch.empty(0)
        goal_steps = torch.cat(step_chunks) if step_chunks else torch.empty(0)
        if errors.numel() == 0:
            return
        good = (errors <= float(args_cli.good_threshold)).float().mean().item()
        bad = (errors > float(args_cli.bad_threshold)).float().mean().item()
        print(f"[SUMMARY] {label}", flush=True)
        print(f"  commands evaluated:       {errors.numel()}", flush=True)
        print(
            f"  goal step mean/max:       {torch.mean(goal_steps).item():.4f} / {torch.max(goal_steps).item():.4f} m",
            flush=True,
        )
        print(f"  final-goal error mean:    {torch.mean(errors).item():.4f} m", flush=True)
        final_p50 = torch.quantile(errors, 0.50).item()
        final_p90 = torch.quantile(errors, 0.90).item()
        final_p99 = torch.quantile(errors, 0.99).item()
        interp_p90 = torch.quantile(interp_errors, 0.90).item()
        print(f"  final-goal error p50/p90: {final_p50:.4f} / {final_p90:.4f} m", flush=True)
        print(f"  final-goal error p99/max: {final_p99:.4f} / {torch.max(errors).item():.4f} m", flush=True)
        print(f"  interp-target err mean:   {torch.mean(interp_errors).item():.4f} m", flush=True)
        print(f"  interp-target err p90/max:{interp_p90:.4f} / {torch.max(interp_errors).item():.4f} m", flush=True)
        print(f"  <= good threshold:        {100.0 * good:.1f}%", flush=True)
        print(f"  > bad threshold:          {100.0 * bad:.1f}%", flush=True)

    _print_summary("all samples", all_final_errors, all_interp_errors, all_goal_steps)
    warmup = max(0, int(args_cli.warmup_samples))
    if warmup > 0:
        _print_summary(
            f"post-warmup samples (skipped {warmup})",
            all_final_errors[warmup:],
            all_interp_errors[warmup:],
            all_goal_steps[warmup:],
        )

    if args_cli.csv is not None:
        _write_csv(args_cli.csv, rows)
        print(f"[INFO] Wrote CSV: {args_cli.csv}")


if __name__ == "__main__":
    main()
    simulation_app.close()
