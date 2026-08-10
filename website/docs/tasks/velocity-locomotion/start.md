---
title: START
description: Experimental START-inspired single-stage depth locomotion for sparse footholds.
---

# START

This is an experimental Go2/Isaac Lab adaptation inspired by
[START: Traversing Sparse Footholds with Terrain Reconstruction](https://arxiv.org/abs/2512.13153). It uses depth input
and terrain reconstruction losses, but it does not reproduce the paper's Lite3/Isaac Gym results or hardware transfer.
Its shared recurrent actor-critic and auxiliary losses use the RSL-RL 5.3 storage and named mini-batch interface.

## Validation Stage

| Surface | Training evidence | Artifact | Transfer |
| --- | --- | --- | --- |
| Task registered; depth agent configured | Algorithm-level CPU rollout/update test; convergence not evaluated | No bundled checkpoint | No sim2sim or hardware validation |

Ground-truth heightmaps are training-only AdaSmpl inputs; playback uses reconstructed maps. The PPO rollout stores one
whole-map AdaSmpl choice per transition, and the behavior-policy mean uses the VAE posterior mean so likelihoods are
recomputed under the same context. The local map encoder is an MLP with three global fusion tokens, whereas the paper
uses a CNN that produces spatial terrain tokens. Treat that as a material architecture simplification.

## Task

```text
Gurukul-Isaac-Velocity-Rough-Unitree-Go2-Start-v0
```

## Train

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Velocity-Rough-Unitree-Go2-Start-v0 \
  --headless \
  --enable_cameras \
  --agent=rsl_rl_cfg_entry_point \
  --run_name go2_rough_start
```

## Play with depth viewer

```bash
python scripts/reinforcement_learning/rsl_rl/play_with_depth.py \
  --task=Gurukul-Isaac-Velocity-Rough-Unitree-Go2-Start-v0 \
  --agent=rsl_rl_cfg_entry_point \
  --load_run <run_folder_name> \
  --num_envs 1 \
  --real-time
```

## Related code

- `source/Gurukul/Gurukul/tasks/manager_based/locomotion/velocity/start_actor_critic.py`
- `source/Gurukul/Gurukul/tasks/manager_based/locomotion/velocity/depth_student_teacher.py`
