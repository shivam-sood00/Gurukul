#!/usr/bin/env python3
from __future__ import annotations

import os
import runpy
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[3]
SCENE = ROOT / "engineai-sim2real/engineai_mujoco/engineai_robots/pm01/scene.xml"
SIM = ROOT / "unitree-sim2real/unitree_mujoco/simulate_python/unitree_mujoco.py"


if not SCENE.is_file():
    raise FileNotFoundError(f"Official EngineAI PM01 MuJoCo scene is missing: {SCENE}")

os.environ.setdefault("GURUKUL_MUJOCO_ROBOT", "pm01")
os.environ.setdefault("GURUKUL_MUJOCO_SCENE", str(SCENE))
sys.argv = [str(SIM), *sys.argv[1:]]
runpy.run_path(str(SIM), run_name="__main__")
