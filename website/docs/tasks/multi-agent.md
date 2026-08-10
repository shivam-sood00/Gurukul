---
title: Multi-Agent
---

# Multi-Agent

Gurukul includes a heterogeneous Go2 + B2 collaboration task built on Isaac Lab's `DirectMARLEnv` API.

## Tasks

| Gym ID | Notes |
| --- | --- |
| `Gurukul-Go2-B2-Collaboration-Direct-v0` | Direct multi-agent Go2 + B2 rendezvous and formation task. |
| `Gurukul-Go2-B2-Hierarchical-Collaboration-Direct-v0` | High-level velocity-command collaboration over frozen locomotion policies. |

## Agents

| Agent ID | Robot | Action interface |
| --- | --- | --- |
| `go2` | Unitree Go2 | 12 joint-position actions in the direct task. |
| `b2` | Unitree B2 | 12 joint-position actions in the direct task. |

The hierarchical variant changes each agent action to a high-level velocity command:

```text
[vx, vy, wz]
```

Those commands are passed to frozen exported locomotion policies, which produce the low-level joint actions.

## Objective

The current task is cooperative rendezvous and formation:

- Move the team centroid toward a fixed target.
- Keep Go2 and B2 near the desired separation.
- Penalize large actions and high joint velocities.
- Reset when either robot falls or the pair separates too far.

Both agents receive the same team reward.

## Train the direct task

The repository RSL-RL script converts the `DirectMARLEnv` dictionary API into a centralized single-agent wrapper for training.

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Gurukul-Go2-B2-Collaboration-Direct-v0 \
  --num_envs 1024 \
  --headless
```

## Train the hierarchical task

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Gurukul-Go2-B2-Hierarchical-Collaboration-Direct-v0 \
  --num_envs 1024 \
  --headless \
  go2_policy_path=/path/to/go2/exported/policy.pt \
  b2_policy_path=/path/to/b2/exported/policy.pt
```

The frozen low-level policies use the standard 45D quadruped locomotion policy observation:

- scaled base angular velocity
- projected gravity
- commanded base velocity
- relative joint position
- scaled joint velocity
- previous low-level action

## Related paths

- `source/Gurukul/Gurukul/tasks/direct/multi_robot_collaboration/go2_b2_collaboration_env.py`
- `source/Gurukul/Gurukul/tasks/direct/multi_robot_collaboration/agents/rsl_rl_ppo_cfg.py`

## Credits

The cooperative task logic is repository-authored on Isaac Lab's `DirectMARLEnv` API and builds on Gurukul's local
Go2/B2 robot and locomotion configurations. See [Acknowledgements](../reference/credits) for shared foundations; no
external multi-agent paper result is claimed.
