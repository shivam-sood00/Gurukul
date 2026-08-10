---
title: Locomotion
description: Choose a locomotion objective before selecting a robot platform or training method.
---

# Locomotion

Locomotion tasks define how the robot should move through its environment. Choose the task by the commanded behavior
and input contract; choose a training method separately under [Training & Methods](../training-methods/overview).

| Task family | Use it for |
| --- | --- |
| [Velocity Locomotion](velocity-locomotion/overview) | Track commanded forward, lateral, and yaw velocities on flat or rough terrain across supported robot platforms. |
| [LiDAR Locomotion](lidar-based) | Train the experimental Go2 rough-terrain policy whose actor consumes Livox-style point observations. |

START, REAL, AME, and Contact Trails all modify how a locomotion policy is trained or represented; they are collected
under [Locomotion Methods](../training-methods/locomotion). Teacher, student, and depth-distillation workflows live
under [Student-Teacher Learning](student-teacher/overview).
