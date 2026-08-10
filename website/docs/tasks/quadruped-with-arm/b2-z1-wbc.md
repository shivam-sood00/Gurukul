---
title: B2+Z1 Whole-Body Control
description: B2+Z1 locomotion, arm-motion teachers, object-task prototypes, and transfer surfaces.
---

# B2+Z1 Whole-Body Control

## What It Is

This page covers velocity-locomotion B2+Z1 tasks where the policy controls the B2 body while the mounted Z1 arm is fixed,
policy-controlled, or driven by scripted motion. Object-centric reach, push, pick-and-throw, and badminton tasks are
also registered; pick-and-throw and badminton have dedicated guides.

| Surface | Training evidence | Artifact | Transfer |
| --- | --- | --- | --- |
| Locomotion, teacher, and experimental object-task registrations | Agent configs and static contracts; no exact-task convergence result | Selected runner exports exist without asserted training provenance | Selected MuJoCo/hardware runners only; no recorded rollout or hardware exercise |

## Main Commands

Train the flat WideArmMoving student:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Velocity-Flat-Unitree-B2-Z1-Arm-WideArmMoving-v0 \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --run_name unitree_b2_z1_wide_arm_moving_flat
```

Train the flat WideArmMoving teacher:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Velocity-Flat-Unitree-B2-Z1-Arm-WideArmMoving-Teacher-v0 \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --run_name unitree_b2_z1_wide_arm_moving_flat_teacher
```

Distill the teacher into the deployable observation/action interface:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Velocity-Flat-Unitree-B2-Z1-Arm-WideArmMoving-Teacher-v0 \
  --headless \
  --agent=rsl_rl_distillation_cfg_entry_point \
  --run_name unitree_b2_z1_wide_arm_moving_flat_distill
```

Play a checkpoint with a forced combined locomotion-and-arm curriculum stage:

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=Gurukul-Isaac-Velocity-Flat-Unitree-B2-Z1-Arm-WideArmMoving-v0 \
  --agent=rsl_rl_cfg_entry_point \
  --load_run <run_folder_name> \
  --loco-manip-stage combined \
  --num_envs 1 \
  --real-time
```

Inspect the normal ArmMoving primitives:

```bash
python scripts/tools/visualize_b2_z1_arm_motion.py \
  --task=Gurukul-Isaac-Velocity-Flat-Unitree-B2-Z1-Arm-ArmMoving-v0 \
  --num_envs 1 \
  --real-time
```

Inspect the raw WideArmMoving workspace:

```bash
python scripts/tools/visualize_b2_z1_arm_motion.py --wide-workspace-check
```

## Task IDs

| Mode | Flat | Rough |
| --- | --- | --- |
| Fixed or policy-controlled arm | `Gurukul-Isaac-Velocity-Flat-Unitree-B2-Z1-Arm-v0` | `Gurukul-Isaac-Velocity-Rough-Unitree-B2-Z1-Arm-v0` |
| Joint-space ArmMoving | `Gurukul-Isaac-Velocity-Flat-Unitree-B2-Z1-Arm-ArmMoving-v0` | `Gurukul-Isaac-Velocity-Rough-Unitree-B2-Z1-Arm-ArmMoving-v0` |
| Task-space WideArmMoving | `Gurukul-Isaac-Velocity-Flat-Unitree-B2-Z1-Arm-WideArmMoving-v0` | `Gurukul-Isaac-Velocity-Rough-Unitree-B2-Z1-Arm-WideArmMoving-v0` |
| ArmMoving teacher | `Gurukul-Isaac-Velocity-Flat-Unitree-B2-Z1-Arm-ArmMoving-Teacher-v0` | `Gurukul-Isaac-Velocity-Rough-Unitree-B2-Z1-Arm-ArmMoving-Teacher-v0` |
| WideArmMoving teacher | `Gurukul-Isaac-Velocity-Flat-Unitree-B2-Z1-Arm-WideArmMoving-Teacher-v0` | `Gurukul-Isaac-Velocity-Rough-Unitree-B2-Z1-Arm-WideArmMoving-Teacher-v0` |
| Object reach | `Gurukul-Isaac-LocoManip-Flat-Unitree-B2-Z1-Reach-v0` | — |
| Object push | `Gurukul-Isaac-LocoManip-Flat-Unitree-B2-Z1-Push-v0` | — |
| Object rearrange | `Gurukul-Isaac-LocoManip-Flat-Unitree-B2-Z1-Rearrange-v0` | — |

The object-centric rows share the B2+Z1 flat scene and runner surface. Use [Pick and Throw](pick-throw) and
[Badminton](badminton) for the separately documented object tasks.

The rearrange row is inspired only at the task-concept level by
[ALORE](https://arxiv.org/abs/2602.04214) ([project page](https://zhihaibi.github.io/Alore/)). The local environment is
a single-policy, shaped-reward cuboid task; it does not implement ALORE's hierarchical object-velocity policy,
estimator, planning stack, or multi-object system, and it does not inherit the paper's results.

## Credits

The B2+Z1 task family, WBC curricula, teacher/student variants, and workspace safety logic are Gurukul work built on
Isaac Lab. The papers that informed individual methods are cited on their relevant pages; shared foundations are
listed in [Acknowledgements](../../reference/credits).
