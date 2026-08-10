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
