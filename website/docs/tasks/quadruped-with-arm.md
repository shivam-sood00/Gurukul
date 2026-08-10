---
title: Quadruped + Arm
description: Choose a Go2 or B2 mounted-arm task family and its user guide.
---

import YouTubeEmbed from '@site/src/components/YouTubeEmbed';

# Quadruped + Arm

## What It Is

Quadruped + Arm covers mounted-manipulator tasks for Go2 and B2 platforms. Use this index to select the robot and task
family, then follow its prerequisites, commands, and limitations.

## Video

<YouTubeEmbed
  videoId="0FOvqjk3mTo"
  title="Go2 Airbot locomotion with random arm movements"
  caption="Airbot example only; use each robot page for task-specific evidence."
/>

## Main Commands

Run a small smoke test first:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-WBC-Flat-Unitree-Go2-D1-Arm-v0 \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --run_name unitree_go2_d1_wbc_flat \
  --num_envs 256
```

Train an object-manipulation task:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-Pick-v0 \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --run_name go2_d1_pick_flat \
  --num_envs 256
```

Play an ArmMoving policy:

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=Gurukul-Isaac-Velocity-Rough-Unitree-Go2-D1-Arm-ArmMoving-v0 \
  --agent=rsl_rl_cfg_entry_point \
  --load_run <run_folder_name> \
  --loco-manip-stage combined \
  --real-time \
  --num_envs 1
```

## Task Pages

| Page | Main task IDs | Notes |
| --- | --- | --- |
| [Go2 + Airbot](quadruped-with-arm/go2-airbot) | `Go2-Airbot-Arm-Fixed-v0`, `Go2-Airbot-Arm-ArmMoving-v0` | Fixed-arm and scripted Airbot motion primitives. |
| [Go2 + D1 WBC](quadruped-with-arm/go2-d1-wbc) | `Gurukul-Isaac-WBC-Flat-Unitree-Go2-D1-Arm-v0`, `Gurukul-Isaac-WBC-Flat-Unitree-Go2-D1-Arm-v0-MJLabActionScale`, `Gurukul-Isaac-WBC-Rough-Unitree-Go2-D1-Arm-v0`, `Gurukul-Isaac-LegWBC-AsyncArm-Flat-Unitree-Go2-D1-Arm-v0`, `Gurukul-Isaac-LegWBC-AsyncArm-Rough-Unitree-Go2-D1-Arm-v0`, `Gurukul-Isaac-WBC-ApexArm-Flat-Unitree-Go2-D1-Arm-v0`, `Gurukul-Isaac-WBC-ApexArm-Rough-Unitree-Go2-D1-Arm-v0` | Stage-1 pretraining and D1 whole-body control details. |
| [LLM High-Level Planning](llm-high-level-planning) | `Gurukul-Isaac-LLM-Go2-D1-Press-Switch-v0`, `Gurukul-Isaac-LLM-Go2-D1-Open-Door-v0`, `Gurukul-Isaac-LLM-Go2-D1-Pick-Place-v0` | Robot-agnostic planner family; Go2+D1 is the initial velocity, IK, posture, and gripper adapter. |
| [Go2 + D1 Pick / Pick-Place](quadruped-with-arm/go2-d1-pick-place) | `Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-Pick-v0`, `Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-PickPlace-v0`, `Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-Pick-Wbc-Hierarchical-v0`, `Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-Pick-LegWbcArm-Hierarchical-v0` | Object pickup and tray placement. |
| [B2 + Z1 Whole-Body Control](quadruped-with-arm/b2-z1-wbc) | `B2-Z1-Arm-v0`, `B2-Z1-Arm-ArmMoving-v0`, `B2-Z1-Arm-WideArmMoving-v0` | B2+Z1 locomotion, scripted arm motion, teachers, and distillation. |
| [B2 + Z1 Pick and Throw](quadruped-with-arm/pick-throw) | `Gurukul-Isaac-LocoManip-Flat-Unitree-B2-Z1-PickThrow-v0` | Pickup object, bin target, and throw rewards. |
| [B2 + Z1 Badminton](quadruped-with-arm/badminton) | `Gurukul-Isaac-LocoManip-Flat-Unitree-B2-Z1-Badminton-v0` | Shuttle proxy, racket proxy, net, and target rewards. |
