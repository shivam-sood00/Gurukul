---
title: B2+Z1 Pick and Throw
---

# B2+Z1 Pick and Throw

## What It Is

This task trains the Unitree B2 with Z1 arm to pick up a small rigid object and throw it into a bin.

| Field | Value |
| --- | --- |
| Task ID | `Gurukul-Isaac-LocoManip-Flat-Unitree-B2-Z1-PickThrow-v0` |
| Terrain | Flat |
| Robot action space | 12 B2 leg joints + 6 Z1 arm joints + Z1 gripper |
| Runner experiment | `unitree_b2_z1_pick_throw_flat` |
| Main config | `source/Gurukul/Gurukul/tasks/manager_based/locomotion/velocity/config/quadruped_with_arm/unitree_b2_z1_arm/loco_manipulation_env_cfg.py` |

## Main Commands

For a smoke test, use a smaller environment count such as `--num_envs 256`.

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-LocoManip-Flat-Unitree-B2-Z1-PickThrow-v0 \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --run_name b2_z1_pick_throw_flat
```

## Credits

This repository-authored object task extends Gurukul's B2+Z1 task/config base. It does not claim to reproduce an
external pick-and-throw paper result.
