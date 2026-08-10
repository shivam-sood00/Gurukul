---
title: Repository Tour
description: Find Gurukul task code, assets, runners, deployment paths, and tests.
---

# Repository Tour

Gurukul is organized as a workspace rather than a single narrow package.

| Path | Purpose |
| --- | --- |
| `source/Gurukul/Gurukul/assets/` | Python asset configs, robot metadata, lidar integration notes, and actuator helpers. |
| `source/Gurukul/data/Robots/` | Robot model data such as URDF, USD, and mesh directories. |
| `source/Gurukul/Gurukul/tasks/manager_based/locomotion/velocity/` | Manager-based velocity locomotion tasks. |
| `source/Gurukul/Gurukul/tasks/manager_based/go2_apex/` | Go2 APEX motion-imitation tasks and motion data tooling. |
| `source/Gurukul/Gurukul/tasks/direct/` | Direct-task examples, including G1 AMP and multi-robot collaboration areas. |
| `scripts/reinforcement_learning/rsl_rl/` | Shared RSL-RL training, playback, distillation, DecAP, and multi-critic entry points. |
| `scripts/tools/` | Task discovery, motion conversion, visualization, contract checks, and debugging tools. |
| `tests/` | Targeted regression and interface-contract tests. |
| `deploy/` | G1 ONNX export and controller build notes. |
| `unitree-sim2real/` | Trimmed Unitree Go2 MuJoCo sim2sim and hardware-runner bundle. |
| `engineai-sim2real/` | EngineAI MuJoCo contracts and policy runners. |
| `third_party/` | Vendored upstream integrations and their retained licenses. |

Use [Find a Task](../reference/task-registry) to map a registered task ID to the corresponding family guide.
