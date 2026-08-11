---
title: Contact Trails
description: Experimental policy-side egocentric contact-memory locomotion for Go2.
---

# Contact Trails

## What It Is

Contact Trails add a short-term egocentric spatial memory map around the robot. The map is updated online from foot
contact events, warped with base motion, decayed over time, encoded by a CNN, and fused into the RL policy.

This is implemented for Unitree Go2 rough velocity locomotion first. Baseline Go2 training is unchanged when you use the
standard task ID.

## Validation Stage

| Surface | Training evidence | Artifact | Transfer |
| --- | --- | --- | --- |
| Task and two agents registered | Algorithm/map unit checks; convergence not evaluated | No bundled checkpoint | No deployment runner; policy memory must be reimplemented |

| Field | Value |
| --- | --- |
| Task ID | `Gurukul-Isaac-Velocity-Rough-Unitree-Go2-Contact-Trails-v0` |
| Learned-write agent | `rsl_rl_cfg_entry_point` |
| Engineered-write agent | `rsl_rl_contact_trails_engineered_cfg_entry_point` |
| Learned-write run | `unitree_go2_rough_contact_trails` |
| Engineered-write run | `unitree_go2_rough_contact_trails_engineered` |

## Main Commands

Train learned writes:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Velocity-Rough-Unitree-Go2-Contact-Trails-v0 \
  --num_envs 16384 \
  --logger wandb \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --run_name go2_contact_trails
```

Train engineered writes for warping/write sanity:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Velocity-Rough-Unitree-Go2-Contact-Trails-v0 \
  --num_envs 16384 \
  --logger wandb \
  --headless \
  --agent=rsl_rl_contact_trails_engineered_cfg_entry_point \
  --run_name go2_contact_trails_engineered
```

Run the unchanged rough baseline:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Velocity-Rough-Unitree-Go2-v0 \
  --num_envs 16384 \
  --logger wandb \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --run_name go2_rough_baseline
```

Debug map warping without Isaac Sim:

```bash
python scripts/debug_contact_trails.py \
  --output-dir logs/debug_contact_trails \
  --steps 8
```

Visualize learned policy memory during playback:

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=Gurukul-Isaac-Velocity-Rough-Unitree-Go2-Contact-Trails-v0 \
  --load_run <run_folder_name> \
  --checkpoint model_<iter>.pt \
  --num_envs 1 \
  --contact-trail-vis
```

Headless frame dumps are also supported:

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=Gurukul-Isaac-Velocity-Rough-Unitree-Go2-Contact-Trails-v0 \
  --load_run <run_folder_name> \
  --checkpoint model_<iter>.pt \
  --headless \
  --contact-trail-vis-output-dir logs/contact_trail_play
```

## Credits and Scope

Contact Trails is a repository-authored experimental policy module built on the local Go2/Isaac Lab locomotion
baseline. It does not claim implementation of an external paper; the inherited task structure is credited in
[Acknowledgements](../../reference/credits).
