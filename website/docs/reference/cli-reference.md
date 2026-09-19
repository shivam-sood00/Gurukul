---
title: CLI Reference
description: Shared RSL-RL scripts and frequently used Gurukul command-line flags.
---

# CLI Reference

## Shared RSL-RL scripts

| File | Purpose |
| --- | --- |
| `scripts/reinforcement_learning/rsl_rl/train.py` | Main training entry point for PPO, distillation, DecAP, and distributed mode. |
| `scripts/reinforcement_learning/rsl_rl/play.py` | Checkpoint playback, visualization, and supported policy export paths. |
| `scripts/reinforcement_learning/rsl_rl/play_with_depth.py` | Playback helper for depth-observation pipelines. |
| `scripts/reinforcement_learning/rsl_rl/eval_student.py` | Quantitative teacher/student evaluation for supported baselines. |
| `scripts/reinforcement_learning/rsl_rl/decap.py` | DecAP/action-prior runner implementation. |
| `scripts/reinforcement_learning/rsl_rl/multi_critic.py` | Multi-critic runner implementation. |
| `scripts/reinforcement_learning/rsl_rl/cli_args.py` | Shared CLI argument definitions. |

## Frequently used flags

| Flag | Notes |
| --- | --- |
| `--task` | Registered task ID. |
| `--agent` | Agent config entry point, such as `rsl_rl_cfg_entry_point`. |
| `--headless` | Use for training without viewport. |
| `--enable_cameras` | Required for depth-camera pipelines. |
| `--load_run` | Run folder name for checkpoint discovery or teacher loading. |
| `--checkpoint` | Explicit checkpoint path. |
| `--run_name` | Name for a new training run. |
| `--teacher_load_run` | Teacher override for distillation or action-prior workflows. |

## Logging flags

- `--log_project_name` sets both `wandb_project` and `neptune_project`.
- `--wandb_project_name` overrides only the W&B project.

## Playback camera

The RSL-RL, CusRL, and skrl playback scripts keep the existing starting camera angle and tracking behavior, with
mouse steering enabled. Orbit, pan, and zoom using the Isaac Sim viewport controls. After an adjustment, a tracking
camera continues moving with its target while preserving your chosen angle, pan offset, and zoom. Keyboard robot
control does not lock the camera.

In RSL-RL `play.py` and `play_with_depth.py`, the default `--camera_follow_mode auto` preserves the previous starting
angle and tracking behavior. Use `--camera_follow_mode mouse` (or `none`) for a free camera starting from the task's
configured view. The `follow`, `isometric`, and `topdown` modes select a tracking view that also supports mouse steering.

## Xbox controller recognition

On Linux, the RSL-RL, CusRL, and skrl training/playback scripts and RSL-RL evaluation automatically register the
USB Xbox Series S|X controller mapping before Isaac Sim starts. This addresses the `Joystick with unknown remapping`
warning for device ID `030000005e040000120b000009050000`. Restart the script with the controller connected; no extra
flag or driver change is needed. The mapping comes from [SDL_GameControllerDB](https://github.com/mdqinc/SDL_GameControllerDB).
It enables recognition by Isaac Sim; robot commands still require the task's controller option, such as
`--go2-d1-live-control`. The startup fix is skipped in headless runs and applies only to this Linux USB device ID.

The mapping is loaded into both the kernel and extension-cache GLFW copies before startup, since Isaac Sim 5.1
can select the windowing extension's copy. Xbox Series USB recognition was checked in Isaac Sim 5.1 with the
controller connected.

RSL-RL `play.py` accepts `--gamepad` for tasks with `commands.base_velocity` and `velocity_commands` observations.
Use it in place of `--keyboard` to command motion with the left stick and turning with the right stick. It requires
the GUI and the first controller recognized by Isaac Sim; connect the controller before launching. The default
stick deadzone is `0.08`, adjustable with `--gamepad-deadzone`. See [PM01 Xbox playback](../tasks/velocity-locomotion/pm01.md#xbox-playback)
for a complete command.

## Mounted-arm playback flags

| Flag | Notes |
| --- | --- |
| `--loco-manip-stage` | Select `fixed`, `arm`, `combined`, `cycle`, `curriculum`, `grid`, or `off` playback behavior. |
| `--loco-manip-arm-difficulty` | Set scripted arm/posture difficulty outside grid mode. |
| `--loco-manip-grid-probe-steps` | Set the phase duration for the eight grid extrema probes (six individual axes and two combined corners); default `250`. |
| `--go2-d1-play-domain-randomization` | Retain Go2+D1 training-time domain randomization during playback. |

## Distillation template

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=<task_id> \
  --agent=<distill_agent_entry_point> \
  --load_run <teacher_run_folder_name> \
  --headless \
  --run_name <run_name>
```

Task pages may add flags such as `--motion-file`, `--keyboard`, `--camera_follow_mode`, or task-specific playback
options.
