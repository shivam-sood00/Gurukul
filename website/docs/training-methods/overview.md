---
title: Training & Methods
description: Choose how a task policy is trained, distilled, represented, and transferred.
---

# Training & Methods

This section explains how policies and models are trained. Start in [Tasks](../tasks/overview) when choosing what the
robot should do; return here for a learning method, architecture, or method-specific transfer workflow.

| Method family | What belongs here |
| --- | --- |
| [Student-Teacher Learning](../tasks/student-teacher/overview) | Privileged teachers, deployable students, depth backbones, distillation losses, recipes, Sim2Sim, and debugging. |
| [Locomotion Methods](locomotion) | START, REAL, AME, and Contact Trails adaptations for velocity locomotion. |
| [Score-Matching Motion Priors](score-matching-motion-priors) | Morphology-specific diffusion priors trained from motion data and used as frozen guidance for velocity-policy learning. |
| [Motion Tracking Methods](motion-tracking) | APEX and BeyondMimic training, data preparation, robot variants, and their specific transfer paths. |

Transfer pages remain with the method or task whose observation, action, policy, and runtime contract they require.
The method-independent export-file contract remains in [Deployment Artifacts](../reference/deployment-artifacts).
