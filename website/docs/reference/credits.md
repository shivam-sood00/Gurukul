---
title: Acknowledgements
description: Major projects and research that helped shape Gurukul.
---

# Acknowledgements

Gurukul was made possible by open-source robotics software and research shared by the community. This page provides a
high-level acknowledgement; each task page cites the papers and repositories that directly informed that task.

## Foundations

- [Isaac Lab](https://github.com/isaac-sim/IsaacLab) provides the simulation framework, task APIs, and extension model.
- [`fan-ziqi/robot_lab`](https://github.com/fan-ziqi/robot_lab) inspired Gurukul's task structure and Isaac Lab
  workspace layout. Selected unchanged Go2+Airbot baseline files retain their upstream headers; the Go2+D1 and B2+Z1
  task families are Gurukul work.
- [RSL-RL](https://github.com/leggedrobotics/rsl_rl) and
  [Legged Gym](https://github.com/leggedrobotics/legged_gym) underpin several reinforcement-learning and locomotion
  conventions used by the project.
- [MuJoCo](https://github.com/google-deepmind/mujoco),
  [Unitree's open-source projects](https://github.com/unitreerobotics), and
  [EngineAI RL Lab](https://github.com/engineai-robotics/engineai_rl_lab) support the robot descriptions, simulation,
  and transfer workflows.
- [BrainCo RevoLab](https://github.com/BrainCoTech/RevoLab) provided the licensed starting point for the adapted Revo3
  hand integration.

## Research Methods

- Motion tracking and action priors draw from [APEX](https://arxiv.org/abs/2511.09091),
  [DecAP](https://arxiv.org/abs/2310.05714), [BeyondMimic](https://arxiv.org/abs/2508.08241),
  [AMP](https://arxiv.org/abs/2104.02180), and [DeepMimic](https://arxiv.org/abs/1804.02717).
- Locomotion and student learning experiments reference [AME](https://arxiv.org/abs/2506.09588),
  [START](https://arxiv.org/abs/2512.13153), [REAL](https://arxiv.org/abs/2603.17653), and
  [Concurrent Teacher-Student RL](https://arxiv.org/abs/2405.10830).
- Loco-manipulation tasks were informed by work on
  [visual whole-body control](https://arxiv.org/abs/2403.16967),
  [arm-constrained curricula](https://arxiv.org/abs/2403.16535),
  [SLIM](https://arxiv.org/abs/2509.03859), and the additional references listed on the relevant task pages.
- Dexterous-hand task pages acknowledge the manipulation papers used for their reward, observation, and training
  designs.

These references describe inspiration and implementation lineage; they do not imply that Gurukul reproduces every
method or reported result.

## Assets, Data, and Licenses

Gurukul's Go2+D1 APEX demonstrations are project-authored. Third-party motion collections, D1 geometry, and
OmniPerception-derived files with unresolved redistribution records are not included in the public snapshot. The
relevant [LiDAR Locomotion](../tasks/lidar-based), [Go2+D1](../tasks/quadruped-with-arm/go2-d1-wbc),
[APEX Motion Data](../tasks/apex/motion-data), and [G1 BeyondMimic](../tasks/beyondmimic/g1) guides link their
authoritative sources and document the ignored local installation paths.

Formal license texts, retained copyright notices, and adapted or vendored component details are recorded in the
repository's
[`THIRD_PARTY_NOTICES.md`](https://github.com/shivam-sood00/Gurukul/blob/main/THIRD_PARTY_NOTICES.md) and beside the
relevant components.
