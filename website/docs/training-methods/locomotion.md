---
title: Locomotion Methods
description: Research methods and policy architectures applied to velocity locomotion tasks.
---

# Locomotion Methods

These pages modify how a velocity-locomotion policy learns or represents terrain information. They are methods applied
to the [Velocity Locomotion](../tasks/velocity-locomotion/overview) objective, not separate top-level robot goals.

| Method | Role |
| --- | --- |
| [START](../tasks/velocity-locomotion/start) | Single-stage recurrent depth locomotion with terrain reconstruction. |
| [REAL](../tasks/velocity-locomotion/real) | Privileged terrain-attention teacher and offline attention pretraining. |
| [AME](../tasks/velocity-locomotion/ame) | Attention-based elevation-map encoding for Go2 and G1. |
| [Contact Trails](../tasks/velocity-locomotion/contact-trails) | Policy-side egocentric memory built from foot-contact events. |

Depth distillation and concurrent teacher/student training are grouped under
[Student-Teacher Learning](../tasks/student-teacher/overview) because their primary organizing principle is the
teacher-to-student training workflow.
