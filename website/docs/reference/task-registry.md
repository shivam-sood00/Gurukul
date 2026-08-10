---
title: Find a Task
description: Search the runtime Gurukul task registry and choose the matching task guide.
---

# Find a Task

The runtime registry is the source of truth for task IDs. The helper launches Isaac Sim headlessly and prints each
Gurukul task with its environment config:

```bash
python scripts/tools/list_envs.py
```

Filter the output when you know the robot or method family:

```bash
python scripts/tools/list_envs.py --keyword Go2
python scripts/tools/list_envs.py --keyword APEX
python scripts/tools/list_envs.py --keyword BeyondMimic
python scripts/tools/list_envs.py --keyword LocoManip
python scripts/tools/list_envs.py --keyword PM01
python scripts/tools/list_envs.py --keyword Revo3
python scripts/tools/list_envs.py --keyword SMP
```

The keyword match is case-sensitive. Run the helper from the repository root in the same Isaac Lab environment used
for training.

Registration is not evidence that a policy learns or transfers. After locating an ID, check its independent training,
artifact, and transfer evidence in the [Task Status catalog](task-status).

Some task IDs appear only after installing optional local integrations, and some registered tasks require external
robot or motion assets before launch. Those prerequisites are documented directly on the matching task page; common
examples are [LiDAR Locomotion](../tasks/lidar-based), [Go2+D1 WBC](../tasks/quadruped-with-arm/go2-d1-wbc),
[APEX Motion Data](../tasks/apex/motion-data), and [G1 BeyondMimic](../tasks/beyondmimic/g1).

## Choose the Matching Guide

| Task name contains | Start with |
| --- | --- |
| `Velocity` | [Velocity Locomotion](../tasks/velocity-locomotion/overview) |
| `SMP` | [Score-Matching Motion Priors](../training-methods/score-matching-motion-priors) |
| `APEX` | [APEX](../tasks/go2-apex) and [APEX Training](../tasks/apex/training) |
| `BeyondMimic` | [BeyondMimic](../tasks/beyondmimic/overview) |
| `Depth`, `Distill`, `Teacher`, or `CTS` | [Student-Teacher](../tasks/student-teacher/overview) |
| `LocoManip`, `WBC`, or an arm name | [Loco-Manipulation](../tasks/loco-manipulation) |
| `LLM` | [LLM High-Level Planning](../tasks/llm-high-level-planning) |
| `Lidar` | [LiDAR Locomotion](../tasks/lidar-based) |
| `Collaboration` | [Multi-Agent](../tasks/multi-agent) |
| `Revo3` | [Revo3 DexHand](../tasks/revo3-dexhand) |

Family pages document the primary runnable tasks and important variants. The runtime output may also contain
compatibility aliases, ablations, teacher-only tasks, and secondary robot variants that do not need separate public
pages.

## Before Adding Another Task ID

Check the filtered registry and the matching family page first. Reuse an existing task or configuration when it already
expresses the required behavior; add a new ID only for a distinct runnable configuration or an explicit compatibility
surface.
