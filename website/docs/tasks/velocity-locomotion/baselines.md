---
title: Platforms and Baselines
---

# Platforms and Baselines

The velocity task family uses manager-based Isaac Lab environments with RSL-RL runners. CTS has a CPU tensor-level
rollout/update smoke test with `rsl-rl-lib==5.3.0`; that test does not launch an Isaac environment or establish
convergence.

## Core Go2 tasks

| Task ID | Notes |
| --- | --- |
| `Gurukul-Isaac-Velocity-Flat-Unitree-Go2-v0` | Flat Go2 PPO baseline. |
| `Gurukul-Isaac-Velocity-Flat-Unitree-Go2-v0-MJLabActionScale` | Flat v0 reward/reset surface with only the mjlab Go2 action scale. |
| `Gurukul-Isaac-Velocity-Rough-Unitree-Go2-v0` | Rough Go2 PPO, privileged teachers, DecAP, and distillation. |
| `Gurukul-Isaac-Velocity-Rough-Unitree-Go2-CTS-v0` | Rough Go2 concurrent teacher-student starter task. |
| `Gurukul-Isaac-Velocity-Flat-Unitree-Go2-v1` | mjlab-aligned flat config. |
| `Gurukul-Isaac-Velocity-Rough-Unitree-Go2-v1` | mjlab-aligned rough config. |

## Other maintained families

- Unitree G1 flat/rough, `v1`, and BrainCo Revo2 interface configs.
- Unitree B2, H1, B2W, and Go2W.
- Booster T1.
- DeepRobotics Lite3 and M20.
- EngineAI PM01.

B2 locomotion keeps a `legs_distance` penalty active: front and rear left-right foot pairs are penalized if their
body-frame lateral spacing drops below `0.18 m`.

## Common agents

| Entry point | Use |
| --- | --- |
| `rsl_rl_cfg_entry_point` | PPO baseline or task default. |
| `rsl_rl_teacher_cfg_entry_point` | Privileged teacher. |
| `rsl_rl_full_teacher_cfg_entry_point` | Full-state rough teacher. |
| `rsl_rl_cts_cfg_entry_point` | Concurrent teacher-student runner for rough Go2. |

## Rough Go2 PPO

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Velocity-Rough-Unitree-Go2-v0 \
  --num_envs 16384 \
  --logger wandb \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --run_name go2_rough_ppo
```

## Flat Go2 v1 PPO

The flat v1 task keeps the mjlab-style reward surface and uses the mjlab-style 10k PPO iteration budget, but removes
rough-terrain heightmap observations. Foot-height rewards use the flat-ground fallback instead of a terrain ray grid.

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Velocity-Flat-Unitree-Go2-v1 \
  --num_envs 16384 \
  --logger wandb \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --run_name unitree_go2_flat_v1_check
```

## Flat Go2 v0 With Mjlab Action Scale

This keeps the v0 flat rewards, resets, and default Go2 actuator config, but replaces the action scale with the mjlab
parity values computed as `0.25 * tau_max / Kp`.

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Velocity-Flat-Unitree-Go2-v0-MJLabActionScale \
  --num_envs 16384 \
  --logger wandb \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --run_name unitree_go2_flat_v0_mjlab_action_scale
```

Export and run the matching MuJoCo sim2sim config:

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=Gurukul-Isaac-Velocity-Flat-Unitree-Go2-v0-MJLabActionScale \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --load_run <run_folder> \
  --checkpoint model_.*.pt \
  --num_envs 1 \
  --max_steps 1

cd unitree-sim2real
python RL_policy_runner/sim2sim/run_rl_policy.py \
  --task go2_velocity_flat_v0_mjlab_action_scale \
  --run <run_folder> \
  --launch-mujoco
```

## Plain Go2 Policy With Moving D1

Use the plain flat Go2 actor on the Go2+D1 MuJoCo model to measure robustness to an arm that was absent from training:

```bash
cd unitree-sim2real
python RL_policy_runner/sim2sim/run_rl_policy.py \
  --task go2_velocity_flat_v0_with_d1_motion \
  --launch-mujoco
```

The actor contract remains the original 45 Go2 observations and 12 leg actions. D1 state is not added to the actor,
and motors 12–18 are driven independently from `wave_hello.npz` through the existing D1 command model. The motion
does not drive the base velocity command, so use the normal keyboard or joystick controls for locomotion. Override the
D1 trajectory without changing the locomotion policy:

```bash
python RL_policy_runner/sim2sim/run_rl_policy.py \
  --task go2_velocity_flat_v0_with_d1_motion \
  --motion-file ../source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/npz/go2_d1/pick_stow_carry.npz \
  --launch-mujoco
```

## Rough Go2 privileged teacher

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Velocity-Rough-Unitree-Go2-v0 \
  --headless \
  --agent=rsl_rl_teacher_cfg_entry_point \
  --num_envs 16384 \
  --logger wandb \
  --run_name go2_rough_teacher
```

## Rough Go2 CTS

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Velocity-Rough-Unitree-Go2-CTS-v0 \
  --num_envs 16384 \
  --logger wandb \
  --agent=rsl_rl_cts_cfg_entry_point \
  --headless \
  --run_name go2_rough_cts
```

## EngineAI PM01

| Task ID | Terrain | Notes |
| --- | --- | --- |
| `Gurukul-Isaac-Velocity-Flat-EngineAI-PM01-v0` | Plane | No height scan. |
| `Gurukul-Isaac-Velocity-Rough-EngineAI-PM01-v0` | Procedural rough terrain | Terrain curriculum enabled. |

EngineAI PM01 uses the retained official 24-DoF URDF and controls all joints, including head yaw. The actuator gains,
armatures, limits, and action scales match the official BeyondMimic PM01 asset. The rough run writes under
`logs/rsl_rl/engineai_pm01_official_urdf_rough/`; the flat run writes under
`logs/rsl_rl/engineai_pm01_official_urdf_flat/`.

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Velocity-Rough-EngineAI-PM01-v0 \
  --num_envs 16384 \
  --logger wandb \
  --headless \
  --run_name engineai_pm01_official_urdf_rough_ppo
```

The PM01 asset config is `Gurukul.assets.engineai_pm01.ENGINEAI_PM01_URDF_CFG`. If meshes fail to load,
check:

```text
source/Gurukul/data/Robots/engineai/pm01_description/urdf/pm01.urdf
```

The base link for commands and height raycasts is `link_base`; the feet are `link_ankle_roll_l` and
`link_ankle_roll_r`. PM01 rough also filters base self-collision contacts against the explicit official URDF names.

## Credits

The baseline task/config organization and several robot families were inspired by or adapted from
[`fan-ziqi/robot_lab`](https://github.com/fan-ziqi/robot_lab) and Isaac Lab. EngineAI PM01 assets come from
[`engineai_rl_lab`](https://github.com/engineai-robotics/engineai_rl_lab). See
[Acknowledgements](../../reference/credits) for shared foundations; retained license details are in the repository's
third-party notices.
