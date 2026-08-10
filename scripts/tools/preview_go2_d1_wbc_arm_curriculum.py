"""Preview the Go2+D1 WBC arm curriculum on a fixed Go2 in Isaac Lab.

The task's real curriculum manager and random-arm event remain active.  Only
the training timeline is compressed, the Go2 root is fixed, and the arm IK
reference is applied directly so a trained WBC checkpoint is not required.
"""

"""Launch Isaac Sim Simulator first."""

import argparse
import time

from isaaclab.app import AppLauncher


DEFAULT_TASK = "Gurukul-Isaac-WBC-Flat-Unitree-Go2-D1-Arm-v0"


parser = argparse.ArgumentParser(description="Preview the exact Go2+D1 WBC arm curriculum on a fixed robot.")
parser.add_argument("--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O.")
parser.add_argument("--task", type=str, default=DEFAULT_TASK, help="WBC task to instantiate.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of synchronized preview environments.")
parser.add_argument("--seed", type=int, default=42, help="Environment and arm-command sampling seed.")
parser.add_argument(
    "--duration-s",
    type=float,
    default=120.0,
    help="Wall-clock curriculum preview duration. Training progress is compressed into this interval.",
)
parser.add_argument(
    "--base-height",
    type=float,
    default=None,
    help="Optional fixed root height. By default the task's normal 0.4 m spawn height is retained.",
)
parser.add_argument(
    "--status-interval-s", type=float, default=1.0, help="Simulation-time interval between status lines."
)
parser.add_argument(
    "--real-time",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Throttle the GUI preview to simulation time; use --no-real-time for a fast headless check.",
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
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab_tasks.utils import parse_env_cfg


def _configure_fixed_curriculum_preview(env_cfg) -> tuple[int, tuple[int, int], tuple[int, int], object]:
    """Keep the real curriculum/event path while making it safe to inspect without a policy."""
    env_cfg.seed = int(args_cli.seed)
    if hasattr(env_cfg.scene.robot.spawn, "fix_base"):
        env_cfg.scene.robot.spawn.fix_base = True
    articulation_props = getattr(env_cfg.scene.robot.spawn, "articulation_props", None)
    if articulation_props is not None:
        articulation_props.fix_root_link = True
    if args_cli.base_height is not None:
        env_cfg.scene.robot.init_state.pos = (0.0, 0.0, float(args_cli.base_height))

    if hasattr(env_cfg, "viewer"):
        env_cfg.viewer.eye = (2.2, 2.2, 1.4)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.55)

    # Exact curriculum inspection should not be obscured by training-only
    # pushes, gain/mass variation, random root resets, or automatic resets.
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
    for termination_name in ("time_out", "terrain_out_of_bounds", "illegal_contact"):
        if hasattr(env_cfg.terminations, termination_name):
            setattr(env_cfg.terminations, termination_name, None)

    advance_arm = getattr(env_cfg.events, "advance_arm_command", None)
    randomize_arm = getattr(env_cfg.events, "randomize_arm_command", None)
    curriculum = getattr(env_cfg.curriculum, "loco_manipulation_training_stages", None)
    if advance_arm is None or randomize_arm is None or curriculum is None:
        raise RuntimeError(f"Task '{args_cli.task}' is not a Go2+D1 WBC curriculum task.")
    advance_arm.params["apply_target"] = True
    advance_arm.params["apply_gripper_target"] = True
    advance_arm.params["gripper_joint_names"] = tuple(env_cfg.gripper_joint_names)

    # In training the policy owns legs, arm, and gripper. Here the action term
    # owns only the legs, leaving the real differential-IK event as the sole
    # writer of arm joint targets. No checkpoint is therefore needed.
    if not hasattr(env_cfg.actions, "joint_pos") or not hasattr(env_cfg, "leg_joint_names"):
        raise RuntimeError(f"Task '{args_cli.task}' does not expose the expected joint_pos WBC action.")
    env_cfg.actions.joint_pos.joint_names = list(env_cfg.leg_joint_names)
    env_cfg.actions.joint_pos.scale = {
        r"^(FL|FR|RL|RR)_hip_joint$": 0.125,
        r"^(FL|FR|RL|RR)_(thigh|calf)_joint$": 0.25,
    }
    env_cfg.actions.joint_pos.clip = {".*": (-100.0, 100.0)}

    # Read the iteration-normalized boundaries from the task itself, then feed
    # the same real curriculum function equivalent preview-step bounds. This
    # deliberately fails if a different task has no normalized schedule rather
    # than silently showing a curriculum that no longer matches training.
    stage_bins = curriculum.params.get("stage_iteration_bins")
    difficulty_bins = curriculum.params.get("arm_difficulty_iteration_bins")
    if stage_bins is None or len(stage_bins) != 2 or difficulty_bins is None or len(difficulty_bins) != 2:
        raise RuntimeError(
            f"Task '{args_cli.task}' must define two stage_iteration_bins and two "
            "arm_difficulty_iteration_bins for an exact normalized preview."
        )
    step_dt = float(env_cfg.sim.dt) * int(env_cfg.decimation)
    total_steps = max(4, int(round(float(args_cli.duration_s) / step_dt)))
    stage_1_step = max(1, int(round(float(stage_bins[0]) * total_steps)))
    stage_2_step = max(stage_1_step + 1, int(round(float(stage_bins[1]) * total_steps)))
    difficulty_start_step = max(1, int(round(float(difficulty_bins[0]) * total_steps)))
    difficulty_end_step = max(difficulty_start_step + 1, int(round(float(difficulty_bins[1]) * total_steps)))
    curriculum.params.update(
        {
            "stage_steps": (stage_1_step, stage_2_step),
            "arm_difficulty_steps": (difficulty_start_step, difficulty_end_step),
            "stage_iteration_bins": None,
            "arm_difficulty_iteration_bins": None,
            "max_iterations": None,
            "steps_per_iteration": None,
            "resume_iteration": 0,
        }
    )
    for curriculum_name in ("command_levels_lin_vel", "command_levels_ang_vel"):
        if hasattr(env_cfg.curriculum, curriculum_name):
            setattr(env_cfg.curriculum, curriculum_name, None)
    return total_steps, (stage_1_step, stage_2_step), (difficulty_start_step, difficulty_end_step), curriculum


def _ee_position_in_root(robot, ee_body_id: int) -> torch.Tensor:
    ee_pose_w = robot.data.body_pose_w[:, ee_body_id]
    root_pose_w = robot.data.root_pose_w
    ee_pos_b, _ = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    return ee_pos_b


def _stage_description(stage: int) -> str:
    return {
        0: "carry pose (training: locomotion warmup)",
        1: "arm-only training",
        2: "combined arm+walking training (root fixed only for this preview)",
    }.get(stage, "unknown")


def main() -> None:
    if float(args_cli.duration_s) <= 0.0:
        raise ValueError("--duration-s must be positive.")

    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    total_steps, stage_steps, difficulty_steps, curriculum = _configure_fixed_curriculum_preview(env_cfg)
    env = gym.make(args_cli.task, cfg=env_cfg)
    base_env = env.unwrapped
    robot = base_env.scene["robot"]
    ee_body_ids = robot.find_bodies("Link6")[0]
    if len(ee_body_ids) != 1:
        raise RuntimeError(f"Expected exactly one Link6 body, found {ee_body_ids}.")
    ee_body_id = int(ee_body_ids[0])
    gripper_joint_names = tuple(env_cfg.gripper_joint_names)
    gripper_joint_ids = torch.as_tensor(
        robot.find_joints(gripper_joint_names, preserve_order=True)[0], device=base_env.device, dtype=torch.long
    )
    actions = torch.zeros(env.action_space.shape, device=base_env.device)
    env_ids = torch.arange(base_env.num_envs, device=base_env.device, dtype=torch.long)
    step_dt = float(base_env.step_dt)
    status_steps = max(1, int(round(float(args_cli.status_interval_s) / step_dt)))

    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Fixed Go2 root; direct IK arm reference; envs={base_env.num_envs}; seed={args_cli.seed}")
    print(
        f"[INFO] Preview: {total_steps} steps / {total_steps * step_dt:.1f}s | "
        f"stage 1 at {stage_steps[0] * step_dt:.1f}s (10%) | "
        f"stage 2 at {stage_steps[1] * step_dt:.1f}s (25%)"
    )
    print(
        f"[INFO] Arm difficulty is 0 through {100.0 * difficulty_steps[0] / total_steps:.1f}%, then increases "
        f"linearly to 1.0 at {100.0 * difficulty_steps[1] / total_steps:.1f}%."
    )
    print("[INFO] The task's original 3-5 s random-command interval and carry deployment waypoints are unchanged.")

    env.reset()
    wall_start = time.time()
    previous_stage = None
    previous_goal = None
    try:
        for step in range(total_steps):
            if not simulation_app.is_running():
                break
            with torch.inference_mode():
                # Isaac Lab evaluates curriculum terms on reset boundaries. A
                # large training batch has staggered resets, but this fixed
                # one-env preview intentionally has none. Re-evaluate the same
                # term against its real common_step_counter so progression is
                # visible without repeatedly snapping the arm back to reset.
                curriculum.func(base_env, env_ids, **curriculum.params)
                env.step(actions)

            stage = int(getattr(base_env, "_loco_manip_training_stage", 0))
            difficulty = float(getattr(base_env, "_loco_manip_arm_motion_difficulty", 0.0))
            goal = getattr(base_env, "_arm_ee_goal_pos", None)
            gripper_target = getattr(base_env, "_gripper_target_pos", None)
            goal_changed = False
            if isinstance(goal, torch.Tensor):
                goal_changed = previous_goal is None or bool(torch.max(torch.abs(goal - previous_goal)).item() > 1.0e-5)
                if goal_changed:
                    previous_goal = goal.clone()

            stage_changed = previous_stage != stage
            if stage_changed or goal_changed or step % status_steps == 0 or step == total_steps - 1:
                progress = min(1.0, (step + 1) / float(total_steps))
                ee_pos = _ee_position_in_root(robot, ee_body_id)
                if isinstance(goal, torch.Tensor):
                    error = torch.linalg.vector_norm(goal - ee_pos, dim=1)
                    goal_text = tuple(round(float(value), 3) for value in goal[0].tolist())
                    error_text = f" | goal={goal_text} | error={float(error[0]):.3f}m"
                else:
                    error_text = ""
                if isinstance(gripper_target, torch.Tensor):
                    target_aperture = float(torch.abs(gripper_target[0, 0] - gripper_target[0, 1]).item())
                    gripper_pos = robot.data.joint_pos.index_select(1, gripper_joint_ids)
                    actual_aperture = float(torch.abs(gripper_pos[0, 0] - gripper_pos[0, 1]).item())
                    gripper_text = (
                        f" | gripper={'open' if target_aperture > 0.02 else 'closed'} "
                        f"(target={target_aperture:.3f}m actual={actual_aperture:.3f}m)"
                    )
                else:
                    gripper_text = ""
                print(
                    f"[CURRICULUM] {100.0 * progress:5.1f}% | stage={stage} "
                    f"({_stage_description(stage)}) | difficulty={difficulty:.3f}{error_text}{gripper_text}"
                )
                previous_stage = stage

            if args_cli.real_time:
                sleep_time = (step + 1) * step_dt - (time.time() - wall_start)
                if sleep_time > 0.0:
                    time.sleep(sleep_time)
    finally:
        env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
