---
title: REAL
description: Experimental Stage-1 REAL-inspired privileged terrain-attention teachers for Go2.
---

# REAL

This page covers an experimental Stage-1 teacher adaptation inspired by
[REAL](https://arxiv.org/abs/2603.17653) and its
[official implementation](https://github.com/JIAlonglong/REAL). It is not the paper's complete deployable pipeline.
The shared REAL actor-critic trains through the RSL-RL 5.3 storage interface, and its attention metrics are forwarded
through the current logger API.

## Validation Stage

| Surface | Training evidence | Artifact | Transfer |
| --- | --- | --- | --- |
| Sparse and beam teacher tasks registered; agent configured | Targeted CPU/model tests; no Isaac convergence result | Offline prior is generated locally; no bundled trained policy | No Stage-2 student, sim2sim, or hardware validation |

The paper's Stage 2 depth student, FiLM/Mamba backbone, consistency-gated distillation, state estimator/EKF, deployable
student, paper metrics, and hardware results are not implemented here. The kinematic offline prior below is a
repository-specific utility rather than a paper reproduction.

## Tasks

| Task ID | Notes |
| --- | --- |
| `Gurukul-Isaac-Velocity-Rough-Unitree-Go2-Real-Sparse-v0` | Sparse-terrain REAL teacher. |
| `Gurukul-Isaac-Velocity-Rough-Unitree-Go2-Real-Beam-v0` | Beam-gap REAL smoke-test variant. |

## Agents

| Entry point | Use |
| --- | --- |
| `rsl_rl_real_teacher_cfg_entry_point` | Train REAL teacher from scratch. |
| `rsl_rl_real_teacher_pretrained_cfg_entry_point` | Initialize the terrain-token encoder and supervised key projection from a checkpoint. |
| `rsl_rl_real_teacher_frozen_cfg_entry_point` | Initialize and freeze only the supervised terrain-token encoder. |

## Sparse REAL teacher

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Velocity-Rough-Unitree-Go2-Real-Sparse-v0 \
  --headless \
  --agent=rsl_rl_real_teacher_cfg_entry_point \
  --run_name go2_rough_real_teacher_sparse
```

## Offline terrain-token pretraining

Pretrain the terrain-token/key prior with a kinematic probe sampler:

```bash
python scripts/tools/pretrain_real_attention.py \
  --headless \
  --terrain-set both \
  --base-height 0.33 \
  --base-height-jitter 0.04 \
  --output artifacts/real_attention/pretrained_attention.pt
```

Useful smoke-test options:

```bash
python scripts/tools/pretrain_real_attention.py \
  --headless \
  --terrain-set start_sparse \
  --samples-per-terrain 256 \
  --num-probes 64 \
  --epochs 5 \
  --batch-size 64 \
  --output /tmp/pretrained_attention_smoke.pt
```

The probe uses the same `height_scanner` footprint as the Go2 rough-terrain task. Its target is a soft traversability
distribution: flat and locally supported areas are rewarded, while slopes, roughness, holes, and sharp steps are
penalized.

The terrain-token encoder and the supervised attention-key projection are exported. Only the encoder is optionally
frozen; the key projection remains trainable because the runtime proprioceptive query differs from the offline constant
query. The runtime query/Q path, value and output projections, and context normalization are initialized and learned by
PPO because the offline attention-probability loss does not supervise them. This repository-specific prior remains
experimental; its benefit under the runtime query has not been evaluated.

`train.py` interprets `--checkpoint` as an offline REAL attention checkpoint when the selected agent is a REAL teacher
config and `resume=False`.

## Playback

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=Gurukul-Isaac-Velocity-Rough-Unitree-Go2-Real-Sparse-v0 \
  --agent=rsl_rl_real_teacher_cfg_entry_point \
  --load_run <run_folder_name>
```

During non-headless playback, `play.py` opens the `Rough Elevation Attention` overlay window and uses `num_envs=1` by
default for clearer attention visualization.

## Targeted Checks

```bash
python -m pytest tests/test_real_teacher.py -q
```
