---
title: Revo3 Dexterous Hand
description: Adapted BrainCo RevoLab Revo3 hand tasks and compatibility aliases.
---

# Revo3 Dexterous Hand

Gurukul includes the public BrainCo RevoLab Isaac Lab task suite as native task registrations.

| Surface | Training evidence | Artifact | Transfer |
| --- | --- | --- | --- |
| Seven adapted registrations; APEX IDs are naming aliases | Static/config tests only; no local convergence result | No bundled local checkpoint | No locally validated sim2real or hardware path |

## Tasks

| Task ID | Workflow | Notes |
| --- | --- | --- |
| `Gurukul-Direct-Revo3-Repose-Cube-v0` | Direct | Revo3 in-hand cube repose. |
| `Gurukul-Direct-Revo3-Reorient-Cylinder-v0` | Direct | Revo3 in-hand cylinder reorientation. |
| `Gurukul-Direct-Revo3-APEX-Repose-Cube-v0` | Direct | APEX-named alias for Revo3 in-hand cube repose. |
| `Gurukul-Direct-Revo3-APEX-Reorient-Cylinder-v0` | Direct | APEX-named alias for Revo3 in-hand cylinder reorientation. |
| `Gurukul-Dexsuite-Revo3-Right-Lift-v0` | Manager-based | Tianji arm with Revo3 right-hand lift task. |
| `Gurukul-Dexsuite-Revo3-Right-Lift-Play-v0` | Manager-based | Play-sized lift config. |
| `Gurukul-Dexsuite-Revo3-APEX-Right-Lift-v0` | Manager-based | APEX-named alias for Tianji/Revo3 lift. |

Direct-task joint reset noise samples the full configured joint range symmetrically, blends it with the default pose,
and clamps the result to the articulation limits. Full policy observations contain 152 values; the optional reduced
observation contains 43, and asymmetric full critic state contains 182.

## Assets

Revo3 USD assets are stored in:

```text
source/Gurukul/data/Robots/brainco/revo3
```

The Isaac Lab asset configs are:

```text
source/Gurukul/Gurukul/assets/brainco_revo3.py
source/Gurukul/Gurukul/assets/brainco_tianji_revo3_right.py
```

## Train

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Direct-Revo3-Repose-Cube-v0 \
  --num_envs=8192 \
  --headless
```

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Dexsuite-Revo3-Right-Lift-v0 \
  --num_envs=4096 \
  --headless
```
