#!/usr/bin/env python3
"""Run an exported official-URDF PM01 velocity policy in MuJoCo."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CONFIG = ROOT / "engineai-sim2real/RL_policy_runner/configs/Gurukul/engineai_pm01_flat_v0.yaml"
SCENE = ROOT / "engineai-sim2real/engineai_mujoco/engineai_robots/pm01/scene.xml"
RUNNER = ROOT / "unitree-sim2real/RL_policy_runner/sim2sim/run_rl_policy.py"


if not SCENE.is_file():
    raise FileNotFoundError(f"Official EngineAI PM01 MuJoCo scene is missing: {SCENE}")
if not any(argument in {"--policy-path", "-p", "-policy-path", "--run"} for argument in sys.argv[1:]):
    raise SystemExit(
        "A newly trained policy is required. Pass --policy-path <exported/policy.onnx> or --run <run-name>."
    )

sys.argv = [str(RUNNER), "--config", str(CONFIG), *sys.argv[1:]]
runpy.run_path(str(RUNNER), run_name="__main__")
