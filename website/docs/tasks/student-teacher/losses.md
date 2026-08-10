---
title: Losses
---

# Losses

Student-teacher losses are split across distillation and action-prior PPO paths.

## Distillation behavior loss

The baseline distillation path trains the student to match privileged teacher actions. The runbook uses `rsl_rl_distillation_cfg_entry_point` for proprioceptive student-teacher distillation and depth student variants.

For the START-style recurrent depth student, `StartDistillation` defaults to `loss_type="mse"`, so the behavior term is an MSE-style action matching loss between student actions and privileged teacher actions.

Relevant code:

```text
source/Gurukul/Gurukul/tasks/manager_based/locomotion/velocity/depth_student_teacher.py
```

## Terrain reconstruction losses

`StudentTeacherDepthBackboneRecurrentSTART` adds auxiliary terrain reconstruction losses when a `terrain_heightmap` observation is available.

The current auxiliary terms are:

| Term | Implementation |
| --- | --- |
| Rough terrain reconstruction | `F.mse_loss(rough_map, terrain_heightmap_gt)` |
| Refined terrain reconstruction | `F.l1_loss(refined_map, terrain_heightmap_gt)` |
| Total reconstruction | weighted rough loss + weighted refined loss |

The recurrent Go2 config sets:

- `terrain_recon_rough_loss_weight = 1.0`
- `terrain_recon_refined_loss_weight = 1.0`
- `reconstruction_loss_weight = 0.5` at the algorithm level

## Action-prior reward terms

The DecAP/action-prior PPO path uses teacher actions during rollout and can add teacher-action matching reward terms. The current reward mode options in `scripts/reinforcement_learning/rsl_rl/decap.py` are:

- `exp_mse`
- `negative_mse`

The logged metrics include `decap_action_prior_mse`, `decap_action_prior_reward`, and decay-related fields.

## Scope

The documented implementation covers action matching, terrain reconstruction, and DecAP/action-prior reward terms.
Feature losses, contrastive losses, and representation-level distillation are not implemented here.
