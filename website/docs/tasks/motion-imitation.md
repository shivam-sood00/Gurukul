---
title: Motion Imitation
description: Find tasks that make a robot follow reference motion.
---

# Motion Imitation

Motion-imitation tasks ask a robot to reproduce a reference trajectory or motion style. Start with
[Motion Tracking](motion-tracking/overview) for the implemented task contracts and supported robot families.

APEX and BeyondMimic describe how those policies are trained and transferred, so their training, motion-data,
Sim2Sim, and Sim2Real guides live together under
[Motion Tracking Methods](../training-methods/motion-tracking).

[Score-Matching Motion Priors](../training-methods/score-matching-motion-priors) use motion data differently: they
pretrain a frozen, morphology-specific prior whose scores guide a velocity policy instead of asking the policy to
track a reference trajectory at every step. SMP therefore lives under Training & Methods while remaining closely
related to the Motion Imitation family.
