---
title: APEX Distillation
---

# APEX Distillation

Go2 APEX has teacher, distillation, depth distillation, and depth action-prior paths documented on
[APEX](../go2-apex) and this page.

## APEX task variants

| Task ID | Purpose |
| --- | --- |
| `Gurukul-Isaac-Go2-APEX-Flat-v0` | Base motion-imitation PPO task. |
| `Gurukul-Isaac-Go2-APEX-Flat-Tracker-v0` | Reference-conditioned tracker. |
| `Gurukul-Isaac-Go2-APEX-Flat-Depth-Distill-v0` | Depth-camera student distillation. |
| `Gurukul-Isaac-Go2-APEX-Flat-Depth-Action-Prior-v0` | PPO student with teacher action priors. |

## APEX agent entry points

| Entry point | Use |
| --- | --- |
| `rsl_rl_teacher_cfg_entry_point` | Privileged teacher. |
| `rsl_rl_distillation_cfg_entry_point` | Student distillation. |
| `rsl_rl_decap_cfg_entry_point` | DecAP-style policy. |

For `Gurukul-Isaac-Go2-APEX-Flat-Depth-Action-Prior-v0`, the task default `rsl_rl_cfg_entry_point` is wired to the depth action-prior DecAP runner.

## Depth action-prior details

The APEX depth action-prior config uses:

- actor observations: `policy + depth_camera`
- critic observations: `critic`
- teacher observations: `critic`
- teacher experiment family: `unitree_go2_apex_flat_teacher`
- student experiment family: `unitree_go2_apex_flat_depth_action_prior`

The depth variants use Isaac Lab training and playback paths. Task-specific MuJoCo YAMLs are not provided.

## References

The action-prior path adapts ideas from [APEX](https://arxiv.org/abs/2511.09091) and
[DecAP](https://arxiv.org/abs/2310.05714) to the local position-target teacher/student interface. It is not a
torque-control reproduction of DecAP. See the [official APEX implementation](https://github.com/marmotlab/APEX).
