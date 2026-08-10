---
title: G1 + BrainCo Revo2
description: G1 velocity-task metadata for the BrainCo Revo2 hardware command interface.
---

# G1 + BrainCo Revo2

## What It Is

This is a platform-specific [Velocity Locomotion](velocity-locomotion/overview) note, not an articulated-hand
manipulation task. Gurukul includes Unitree G1 velocity tasks tagged with the BrainCo Revo2 hand command interface from
[`BrainCoTech/unitree-g1-brainco-hand`](https://github.com/BrainCoTech/unitree-g1-brainco-hand).

| Surface | Training evidence | Artifact | Transfer |
| --- | --- | --- | --- |
| Two registered G1 velocity configs with hand-interface metadata | No articulated-hand training/runtime result | No bundled checkpoint | Metadata only; not a validated sim2real path |

| Task ID | Notes |
| --- | --- |
| `Gurukul-Isaac-Velocity-Flat-Unitree-G1-BrainCo-Revo2-v0` | Flat G1 velocity task with Revo2 command metadata. |
| `Gurukul-Isaac-Velocity-Rough-Unitree-G1-BrainCo-Revo2-v0` | Rough G1 velocity task with Revo2 command metadata. |

## Main Commands

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Velocity-Flat-Unitree-G1-BrainCo-Revo2-v0 \
  --headless
```
