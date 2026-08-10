---
title: Task Overview
description: Choose a Gurukul research area and find its runnable tasks.
---

# Tasks

Start with a research goal, then use the linked task page for prerequisites, exact task IDs, agent entry points, and
commands. A listed simulation or transfer path applies only to the tasks documented on that page.

## Choose By Goal

- **Make a robot move through terrain:** start with [Locomotion](locomotion), then choose velocity- or
  LiDAR-conditioned locomotion.
- **Train a locomotion baseline:** start with [Velocity Locomotion](velocity-locomotion/overview), then use
  [Common Commands](../getting-started/common-commands).
- **Train a teacher or student policy:** start with [Student-Teacher Learning](student-teacher/overview), then choose a
  [training recipe](student-teacher/training-recipes).
- **Track reference motion:** start with [Motion Imitation](motion-imitation), then choose a method under
  [Motion Tracking Methods](../training-methods/motion-tracking).
- **Interact with objects:** start with [Manipulation](manipulation), then choose mobile or dexterous manipulation.
- **Use LiDAR observations for locomotion:** start with [LiDAR Locomotion](lidar-based).
- **Run cooperative robots:** start with [Multi-Agent](multi-agent).
- **Evaluate language-model robot planning:** start with [Planning & Autonomy](planning-autonomy).
- **Choose a learning architecture or research method:** use [Training & Methods](../training-methods/overview).
- **Move toward MuJoCo or hardware:** use the transfer page for the selected task family and review
  [Deployment Artifacts](../reference/deployment-artifacts).
- **Search all registered task IDs:** use [Find a Task](../reference/task-registry).
- **Check implementation and validation stage:** use the per-ID [Task Status catalog](../reference/task-status).

## Task Families

### Locomotion

[Locomotion](locomotion) covers velocity tracking on flat and rough terrain plus the experimental Go2 LiDAR policy.
Robot platforms and baseline environments remain under the task; START, REAL, AME, Contact Trails, and
student-teacher workflows are organized by training method.

### Motion Imitation

[Motion Imitation](motion-imitation) covers reference-motion and motion-style objectives. The APEX and BeyondMimic
implementations live under Motion Tracking Methods with their own transfer instructions.

### Manipulation

[Manipulation](manipulation) covers quadruped-mounted arms, PM01 whole-body object interaction, and Revo3 dexterous-hand
tasks.

### Multi-Agent

[Multi-Agent](multi-agent) covers cooperative Go2+B2 DirectMARL and hierarchical velocity control.

### Planning and Autonomy

[Planning & Autonomy](planning-autonomy) covers high-level decision tasks over trained lower-level controllers. Go2+D1
is the first adapter for the current LLM planning evaluation family.

## Research Status

Use the [Task Status catalog](../reference/task-status) for every exact registered ID. It separates registration,
training evidence, bundled artifacts, transfer adapters, and actual validation so a configured runner is not mistaken
for a converged policy or exercised hardware path.

## Acknowledgements

Gurukul builds on Isaac Lab, and its workspace structure and selected baseline configurations were inspired by or
adapted from [`fan-ziqi/robot_lab`](https://github.com/fan-ziqi/robot_lab). Each method page identifies the papers and
repositories that directly informed it. See [Acknowledgements](../reference/credits) for the shared foundations and the
repository's third-party notices for formal redistribution details.
