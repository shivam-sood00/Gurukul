---
title: Depth Backbones
---

# Depth Backbones

Depth student-teacher support currently lives in `source/Gurukul/Gurukul/tasks/manager_based/locomotion/velocity/depth_student_teacher.py` with Go2 config entry points under `config/quadruped/unitree_go2/agents/`.

## Feed-forward CNN depth student

`StudentTeacherDepthBackbone` uses:

- student observations: `policy + depth_camera`
- teacher observations: `critic`
- depth image shape: `(58, 87)`
- depth CNN channels: `[32, 64]`
- depth CNN kernels: `[5, 3]`
- first-layer max-pool kernel: `2`
- image projection dimension: `128`
- depth latent dimension: `32`
- student hidden dimensions in the Go2 config: `[512, 256, 128]`

The actor encodes the flattened depth image with a CNN, projects it into a depth latent, concatenates that latent with proprioceptive observations, then sends the fused vector through the actor MLP.

Config:

```text
source/Gurukul/Gurukul/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_go2/agents/rsl_rl_depth_distillation_cfg.py
```

## START-style recurrent depth student

`StudentTeacherDepthBackboneRecurrentSTART` extends the depth student with recurrent memory, terrain reconstruction heads, and AdaSmpl probability scheduling.

The Go2 recurrent config uses:

- depth image shape: `(58, 87)`
- depth CNN channels: `[32, 64]`
- depth latent dimension: `32`
- recurrent fusion dimension: `128`
- RNN type: `gru`
- RNN hidden dimension: `256`
- RNN layers: `1`
- terrain heightmap observation group: `terrain_heightmap`
- terrain reconstruction hidden dimension: `256`
- terrain reconstruction latent dimension: `64`

Config:

```text
source/Gurukul/Gurukul/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_go2/agents/rsl_rl_depth_recurrent_distillation_cfg.py
```
