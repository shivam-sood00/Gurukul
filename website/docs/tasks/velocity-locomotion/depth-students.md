---
title: Velocity Student Variants
description: Go2 velocity-locomotion depth distillation and recurrent student workflows.
---

# Velocity Student Variants

Velocity locomotion includes depth-student and student-teacher variants for Go2 rough terrain.
The combined depth student/teacher policies use an RSL-RL 5.3 storage adapter; the recurrent variant keeps terrain
reconstruction and AdaSmpl losses in the same distillation update.

## Task

```text
Gurukul-Isaac-Velocity-Rough-Unitree-Go2-Depth-Distill-v0
```

## Agents

| Entry point | Use |
| --- | --- |
| `rsl_rl_distillation_cfg_entry_point` | Student-teacher distillation. |
| `rsl_rl_depth_recurrent_distillation_cfg_entry_point` | START-style recurrent depth student with terrain reconstruction. |
| `rsl_rl_depth_action_prior_cfg_entry_point` | Depth PPO student with teacher action priors. |

## Depth student command

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Velocity-Rough-Unitree-Go2-Depth-Distill-v0 \
  --headless \
  --enable_cameras \
  --agent=rsl_rl_distillation_cfg_entry_point \
  --load_run <teacher_run_folder_name> \
  --run_name go2_rough_depth_student_teacher
```

For detailed student-teacher notes, see [Student-Teacher](../student-teacher/overview).
