---
title: Motion Tracking Methods
description: Training and transfer methods for reference-motion policies.
---

# Motion Tracking Methods

The task is to follow a reference motion; these method families define how the policy, reference interface, data, and
transfer contract implement that task.

| Method | Scope |
| --- | --- |
| [APEX](../tasks/go2-apex) | Go2, Go2+D1, G1, and B2+Z1 reference-conditioned trackers, motion tooling, distillation variants, and exact APEX Sim2Sim/Sim2Real contracts. |
| [BeyondMimic](../tasks/beyondmimic/overview) | G1, EngineAI PM01, and EngineAI T800 reference-motion PPO, with robot-specific artifacts and transfer instructions. |

APEX transfer pages stay beneath APEX because they depend on its reference frames, actor observations, action order,
motion files, and policy variants. PM01 and T800 transfer commands remain on their BeyondMimic robot pages for the
same reason; those instructions do not apply to unrelated motion-tracking policies.
