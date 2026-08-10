---
title: AME
description: Experimental AME-inspired map-attention locomotion tasks for Go2 and G1.
---

# AME

These are experimental, single-stage adaptations inspired by
[Attention-Based Map Encoding for Learning Generalized Legged Locomotion](https://arxiv.org/abs/2506.09588).
They are not a reproduction of the paper's training pipeline or reported hardware results.

## Validation Stage

| Surface | Training evidence | Artifact | Transfer |
| --- | --- | --- | --- |
| Go2 and G1 tasks registered; agents configured | Model/unit checks only; convergence not evaluated | No bundled checkpoint | No AME-specific sim2sim or hardware validation |

During training, AME does not materialize per-head attention weights; they are computed only in eval/play mode for visualization. This keeps the attention encoder path lighter without changing the policy output.

This repository uses four minibatches and 2,048 environments to keep the attention update practical on a typical
24 GB GPU. That is a repository-specific setting; the paper's supplementary configuration implies three minibatches
from its rollout and minibatch sizes.

The local model also differs structurally: its stride-2 CNN consumes XYZ directly, actor and critic have separate
encoders, and training begins on noisy maps in one stage. The paper uses a stride-1 z-only CNN followed by the original
XYZ coordinates, a shared encoder, and a two-stage clean-to-noisy curriculum. Base linear velocity is also absent from
the local actor observation. Treat these differences as experimental adaptations when comparing results.

## Tasks

| Task ID | Notes |
| --- | --- |
| `Gurukul-Isaac-Velocity-Rough-Unitree-Go2-AME-v0` | Go2 rough AME task. |
| `Gurukul-Isaac-Velocity-Rough-Unitree-G1-AME-v0` | G1 rough AME task. |

## Key Files

| Purpose | File |
| --- | --- |
| AME RSL-RL model | `scripts/reinforcement_learning/rsl_rl/ame.py` |
| XYZ elevation-map observation | `source/Gurukul/Gurukul/tasks/manager_based/locomotion/velocity/mdp/observations.py` |
| Go2 AME environment | `source/Gurukul/Gurukul/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_go2/ame_env_cfg.py` |
| Go2 AME RSL-RL config | `source/Gurukul/Gurukul/tasks/manager_based/locomotion/velocity/config/quadruped/unitree_go2/agents/rsl_rl_ame_cfg.py` |
| G1 AME environment | `source/Gurukul/Gurukul/tasks/manager_based/locomotion/velocity/config/humanoid/unitree_g1/ame_env_cfg.py` |
| G1 AME RSL-RL config | `source/Gurukul/Gurukul/tasks/manager_based/locomotion/velocity/config/humanoid/unitree_g1/agents/rsl_rl_ame_cfg.py` |

## Train

Go2:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Velocity-Rough-Unitree-Go2-AME-v0 \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --run_name go2_rough_ame
```

G1:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Velocity-Rough-Unitree-G1-AME-v0 \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --run_name g1_rough_ame
```

## Play

AME play enables the height-scanner point cloud and terrain-attention marker overlay by default. In non-headless runs,
`play.py` opens a rough-elevation attention window if OpenCV is installed.

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=Gurukul-Isaac-Velocity-Rough-Unitree-Go2-AME-v0 \
  --agent=rsl_rl_cfg_entry_point \
  --load_run <run_folder_name> \
  --num_envs 1
```

Use `--checkpoint logs/rsl_rl/<experiment_name>/<run_folder_name>/model_<iter>.pt` to load an exact checkpoint.

## Credits and Scope

This experimental adaptation is informed by the [AME paper](https://arxiv.org/abs/2506.09588) and its
[reference implementation](https://github.com/SII-FUSC/AME_Locomotion). The local encoder and curriculum differ; the
task page does not claim reproduction of the paper's reported results.
