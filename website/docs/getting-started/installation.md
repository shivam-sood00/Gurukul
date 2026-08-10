---
title: Installation
description: Install Gurukul in an Isaac Lab environment and verify its registered tasks.
---

# Installation

Gurukul is a robotics research workspace built around Isaac Lab and Python entry points. Install Isaac Lab
first, then install this repository as an editable extension from the same Python environment.

## Install Isaac Lab

Follow the official Isaac Lab installation guide:

- [Isaac Lab local installation](https://isaac-sim.github.io/IsaacLab/main/source/setup/installation/index.html)

The Isaac Lab guide recommends the Isaac Sim pip-package workflow for most new users. Use the Isaac Lab Python or conda
environment for the remaining commands on this page. Isaac Sim 5.x uses Python 3.11; Isaac Sim 4.x uses Python 3.10.

## Clone and enter the workspace

Clone this repository separately from the Isaac Lab repository, outside the `IsaacLab/` directory:

```bash
git clone https://github.com/shivam-sood00/Gurukul.git
cd Gurukul
```

## Install the local extension

The extension package lives in `source/Gurukul/`. A typical editable install is:

```bash
python -m pip install -e source/Gurukul
```

Install the RSL-RL version expected by the training scripts:

```bash
python -m pip install "rsl-rl-lib==5.3.0"
```

## Version compatibility

This project is known to work with Isaac Sim 5.1 and a matching Isaac Lab installation. Use one Isaac Lab Python
environment for installing the extension, training, playback, and export so package versions stay consistent.

Recommended baseline:

| Component | Recommended version |
| --- | --- |
| Isaac Sim | 5.1.x |
| Isaac Lab | Version compatible with Isaac Sim 5.1.x |
| Python | 3.11 for Isaac Sim 5.x |

One known-good development environment uses:

| Package | Version | Notes |
| --- | --- | --- |
| Isaac Sim | 5.1.0.0 | All installed `isaacsim-*` packages are 5.1.0.0. |
| Isaac Lab | 0.54.3 | Editable source checkout. |
| `isaaclab_assets` | 0.2.4 | Editable source checkout. |
| `isaaclab_rl` | 0.5.0 | Editable source checkout. |
| `isaaclab_tasks` | 0.11.14 | Editable source checkout. |
| `rsl-rl-lib` | 5.3.0 | Required for `rsl_rl.models` and `rsl_rl.utils.logger`. |
| Gurukul | 0.1 | Editable install from `source/Gurukul`. |

Check your local versions with:

```bash
python -m pip list | grep -Ei "isaac|gurukul|rsl-rl"
python -m pip show isaacsim isaaclab
```

## Verify tasks

Use the environment registry helper:

```bash
python scripts/tools/list_envs.py
```

Look for task IDs beginning with `Gurukul-Isaac-`.
