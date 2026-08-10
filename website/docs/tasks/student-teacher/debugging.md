---
title: Debugging
---

# Debugging

Student-teacher failures usually come from mismatched checkpoints, observation groups, camera flags, or hidden assumptions about teacher and student interfaces.

## Checkpoint loading

Distillation and DecAP/action-prior paths can load teacher checkpoints from a different experiment than the student run. Prefer explicit teacher flags when reproducing a run:

```bash
--teacher_experiment_name <teacher_experiment>
--teacher_load_run <teacher_run_folder>
--teacher_load_checkpoint <checkpoint_pattern>
```

Use `--teacher_checkpoint <path>` when you want an exact checkpoint file.

## Observation groups

Check the agent config before debugging policy outputs:

- proprioceptive distillation commonly uses student `policy` observations and teacher `critic` observations.
- depth students use `policy + depth_camera` for the student path.
- action-prior PPO commonly keeps the critic and teacher on `critic`.

If a depth backbone fails at startup, confirm the `depth_camera` group is present in the active policy observation groups.

## Camera flags

Depth pipelines need camera support:

```bash
--enable_cameras
```

Playback with depth visualization can use:

```bash
python scripts/reinforcement_learning/rsl_rl/play_with_depth.py ...
```

Useful flags include `--no_depth_vis`, `--depth_env_index`, `--depth_window_scale`, `--depth_colormap`, and `--depth_print_interval`.

## Evaluation discipline

Run teacher, proprioceptive student, and depth/action-prior student evaluations with comparable episode counts and explicit checkpoint paths. Use `--keep_randomization` for robustness measurements under training-time randomization or corruption.

## Sim2sim mismatch checks

When Isaac Lab playback works but MuJoCo sim2sim diverges, use the [Sim2Sim](sim2sim) student-teacher checklist before changing policy code. Most failures come from actor observation layout, depth image preprocessing, action scale, joint order, or PD gain mismatches.
