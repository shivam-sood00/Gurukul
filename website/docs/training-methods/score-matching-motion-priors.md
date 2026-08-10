---
title: Score-Matching Motion Priors
description: Prepare morphology-specific motion priors and train SMP-guided velocity policies for G1, PM01, and Go2.
---

# Score-Matching Motion Priors

SMP is a training method applied through separate manager-based robot configurations for G1, PM01, and Go2. It adds a
frozen, pretrained score-matching motion prior to the flat-ground velocity reward. The prior scores a
ten-frame, oldest-to-newest window of task-independent robot kinematics at 50 Hz. Policy observations and actions stay
with the existing velocity task. The registered configs scale the existing task rewards by `0.5` and add the SMP prior
reward with weight `0.5` during PPO training.

:::caution[Experimental research path]
The repository provides task registrations, offline prior tooling, tensor/static tests, and a two-environment,
one-iteration headless training smoke for each registered robot. It does not bundle an SMP prior or policy checkpoint,
and no convergence, longer-horizon rollout, Sim2Sim, hardware, or cross-morphology result is claimed.
:::

## Tasks

| Robot | Exact task ID | Prior profile | Features/frame |
| --- | --- | --- | ---: |
| Unitree G1 | `Gurukul-Isaac-SMP-Velocity-Flat-Unitree-G1-v0` | `g1` | 59 |
| EngineAI PM01 | `Gurukul-Isaac-SMP-Velocity-Flat-EngineAI-PM01-v0` | `pm01` | 54 |
| Unitree Go2 | `Gurukul-Isaac-SMP-Velocity-Flat-Unitree-Go2-v0` | `go2` | 39 |

Each profile fixes the joint order, root body, key-body order, feature dimension, 50 Hz control rate, and dataset schema.
A prior trained for one profile cannot be used by either of the other robots. Shape-compatible files are also rejected
when their named joints or bodies differ.

Each frame contains root-link position, root-link rotation in a six-dimensional x/z-column representation, ordered
joint positions, root-relative key-body positions, and root-link linear/angular velocity. Window translation and
heading are canonicalized against the newest frame while root height remains absolute.

## Prerequisites

Supply one or more motion archives for the selected robot in the repository's named NPZ motion format. Each archive
must contain `fps`, `body_names`, and `joint_names` or `dof_names`. Body quaternions must use scalar-first `wxyz`
ordering. The state arrays may use these repository formats:

| State | Accepted NPZ keys |
| --- | --- |
| Joint positions | `joint_pos`, `dof_positions`, or `dof_pos` |
| World body positions | `body_pos_w`, `body_positions`, or `body_pos` |
| World body quaternions | `body_quat_w`, `body_rotations`, or `body_quat` |

The offline builder validates names, shapes, source rate, and finite values before producing windows. It reorders named
channels to the profile contract and resamples compatible source clips to the profile's 50 Hz control rate. Joint
angles are unwrapped before interpolation. Root link velocities are always derived with centered differences from the
root link pose, avoiding the ambiguous COM velocity convention used by older Isaac Lab NPZ fields.
Pass
`--target-fps 50` to make that output rate explicit; another rate is incompatible with the registered tasks.

Motion files and generated checkpoints are intentionally not distributed with this feature. Use only data that you are
licensed to use, and keep private, restricted, or non-redistributable motion outside the repository. Resampling changes
the sample rate; it does not retarget motion between robot morphologies or repair missing named channels. Key-body
positions are interpolated from the supplied archive, so a source exported with simulator FK at 50 Hz provides the
strongest offline/online kinematic parity.

The [G1 BeyondMimic](../tasks/beyondmimic/g1), [PM01 BeyondMimic](../tasks/beyondmimic/pm01), and
[Go2 APEX motion-data](../tasks/apex/motion-data) pages describe related source-motion workflows. Their outputs are direct SMP
inputs only when they include every named field listed above; the SMP loader does not guess missing names.

## Prepare A Prior Dataset

Build ten-frame canonical windows from one or more compatible clips:

```bash
python scripts/tools/smp/build_dataset.py \
  --profile g1 \
  --motion /absolute/path/to/g1_motion.npz \
  --normalization-motion /absolute/path/to/diverse_g1_corpus.npz \
  --target-fps 50 \
  --output /absolute/path/to/g1_smp_windows.npz
```

Use `--profile pm01` or `--profile go2` with motion from that morphology. Repeat `--motion` to combine clips. Windows
never cross a clip boundary, and the output compressed NPZ records the complete robot profile and resolved source-path
list. `--normalization-motion` is optional and repeatable: it fits the checkpoint's q01/q99 bounds on a broader
same-robot corpus without adding those clips to the denoiser's training windows. This is recommended for a narrow
task prior. Without it, bounds come from the training clips. Every channel uses a minimum `0.4` q01/q99 span to avoid
amplifying simulator noise in nearly constant features. The q01/q99 anchors map to `-1`/`+1`; tail samples are left
unclipped, matching the denoiser training contract.

## Pretrain The Frozen Prior

```bash
python scripts/reinforcement_learning/smp/pretrain.py \
  --dataset /absolute/path/to/g1_smp_windows.npz \
  --output /absolute/path/to/g1_smp_prior.pt \
  --device cuda
```

The default prior uses 50 diffusion steps, a cosine schedule, epsilon prediction, an L1 denoising loss, and the
two-layer, four-head TinyMDM denoiser structure used by MimicKit and the SUZ reproduction: shared AdaLN timestep
conditioning, SwiGLU feed-forward layers, and residual 1×1 input/output projections. The checkpoint contains the
feature normalizer and exact morphology metadata used by the runtime compatibility check. Keep the dataset beside the
experiment record; it is not embedded in the checkpoint.

## Train And Play A Policy

Pass the matching frozen prior to the standard RSL-RL entry point:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Gurukul-Isaac-SMP-Velocity-Flat-Unitree-G1-v0 \
  --smp-prior /absolute/path/to/g1_smp_prior.pt \
  --headless
```

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task Gurukul-Isaac-SMP-Velocity-Flat-Unitree-G1-v0 \
  --smp-prior /absolute/path/to/g1_smp_prior.pt \
  --checkpoint /absolute/path/to/policy_model.pt
```

Substitute the PM01 or Go2 task ID and its matching prior. Startup fails if the prior is missing, uses another profile,
was prepared at another control rate, or describes a different feature schema.

Policy checkpoints also store the adaptive score-normalizer state and the prior file's content digest. Resume and
playback therefore require the exact prior file used by that policy run, not merely another prior with the same shape.
Training accumulates score statistics over a complete PPO rollout and commits them once at the rollout boundary;
playback keeps the restored statistics frozen so evaluation rewards do not drift.

Generative state initialization (GSI) is opt-in. Pass a positive `--smp-gsi-pool-size` to generate a fixed startup
pool, or leave it at `0` to use the inherited robot resets while retaining the frozen SMP reward. Generated histories
have finite-value and joint-limit safeguards, but their key-body channels are not yet recomputed through simulator FK
after decoding; validate reset feasibility for the selected corpus before enabling GSI in a long run.

## Method And Limitations

The implementation follows the offline-prior structure in
[SMP: Reusable Score-Matching Motion Priors for Physics-Based Character Control](https://yxmu.foo/smp-page/)
([arXiv:2512.03028](https://arxiv.org/abs/2512.03028)):
the denoiser is pretrained on motion windows, frozen during PPO, and evaluated at fixed diffusion steps to form an
exponentiated score-distillation guidance reward. This repository-authored implementation is informed by the
[Apache-2.0-licensed MimicKit SMP code at commit `2ed1e6c`](https://github.com/xbpeng/MimicKit/tree/2ed1e6c093bb0829f55d33cb4f7a1731cfe6cb69). The
[SUZ-tsinghua/smp repository](https://github.com/SUZ-tsinghua/smp) was used only as a secondary implementation
comparison; it is not vendored here.

The cited SMP work evaluates simulated human characters and reports a real G1 experiment. This repository's G1 and
PM01 profiles are robot-specific ports of the method; Go2 is a quadruped extrapolation of the score-prior mechanism,
not an outcome established by that paper. All three require separate motion data, separate prior training, and separate
policy validation. The current public surface does not claim that a prior transfers between robots or that SMP improves
the corresponding PPO baseline without a measured experiment.
