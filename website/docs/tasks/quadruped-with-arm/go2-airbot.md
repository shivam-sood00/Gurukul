---
title: Go2+Airbot
---

# Go2+Airbot

## What It Is

Go2+Airbot tasks train Go2 locomotion with a mounted Airbot arm. Fixed-arm variants keep the arm at the default pose;
ArmMoving variants use scripted safe motion primitives while the policy controls the 12 Go2 leg joints.

## Main Commands

Train the flat fixed-arm baseline:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Velocity-Flat-Unitree-Go2-Airbot-Arm-Fixed-v0 \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --run_name go2_airbot_fixed_flat
```

Train the rough fixed-arm baseline:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Velocity-Rough-Unitree-Go2-Airbot-Arm-Fixed-v0 \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --run_name go2_airbot_fixed_rough
```

Train flat ArmMoving:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Velocity-Flat-Unitree-Go2-Airbot-Arm-ArmMoving-v0 \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --run_name go2_airbot_ik_flat
```

Visualize Airbot arm motion before training:

```bash
python scripts/tools/visualize_go2_airbot_arm_motion.py \
  --task=Gurukul-Isaac-Velocity-Flat-Unitree-Go2-Airbot-Arm-ArmMoving-v0 \
  --num_envs 1 \
  --real-time
```

Play ArmMoving with a chosen stage:

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=Gurukul-Isaac-Velocity-Rough-Unitree-Go2-Airbot-Arm-ArmMoving-v0 \
  --agent=rsl_rl_cfg_entry_point \
  --load_run <run_folder_name> \
  --loco-manip-stage combined \
  --num_envs 1
```

## Task IDs

| Variant | Task ID | Run name |
| --- | --- | --- |
| Flat fixed arm | `Gurukul-Isaac-Velocity-Flat-Unitree-Go2-Airbot-Arm-Fixed-v0` | `go2_airbot_fixed_flat` |
| Rough fixed arm | `Gurukul-Isaac-Velocity-Rough-Unitree-Go2-Airbot-Arm-Fixed-v0` | `go2_airbot_fixed_rough` |
| Flat ArmMoving | `Gurukul-Isaac-Velocity-Flat-Unitree-Go2-Airbot-Arm-ArmMoving-v0` | `go2_airbot_ik_flat` |
| Rough ArmMoving | `Gurukul-Isaac-Velocity-Rough-Unitree-Go2-Airbot-Arm-ArmMoving-v0` | `go2_airbot_ik_rough` |

## Credits

The Go2+Airbot task/config surface was inspired by and adapted from
[`fan-ziqi/robot_lab`](https://github.com/fan-ziqi/robot_lab). The local ArmMoving curriculum and tooling are Gurukul
extensions to that baseline.
