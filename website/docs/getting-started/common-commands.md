---
title: Common Commands
description: Common training, playback, checkpoint, and task-discovery commands.
---

# Common Commands

These are the common launcher shapes. Pick the task ID and agent entry point from the relevant task page.

## Before Running

1. Install the workspace with [Installation](installation).
2. List available environments with `scripts/tools/list_envs.py`.
3. Choose a task family from [Task Overview](../tasks/overview).

## Train

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=<task_id> \
  --headless \
  --agent=<agent_entry_point> \
  --run_name <run_name>
```

## Play

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=<task_id> \
  --agent=<agent_entry_point> \
  --load_run <run_folder_name> \
  --num_envs 16
```

## Load an explicit checkpoint

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=<task_id> \
  --agent=<agent_entry_point> \
  --checkpoint logs/rsl_rl/<experiment>/<run_folder>/model_<iter>.pt
```

## List registered environments

```bash
python scripts/tools/list_envs.py
```

## Where To Go Next

- For task families, videos, and implementation notes, start with [Task Overview](../tasks/overview).
- For exact task IDs, agent entry points, checkpoint flags, and runner-specific options, use the matching task page.
- For exported policy layouts and transfer contracts, use [Deployment Artifacts](../reference/deployment-artifacts).
