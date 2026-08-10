#!/usr/bin/env python3
"""Replay converted B2+Z1 APEX motions in Isaac Lab.

This is a convenience wrapper around ``scripts/tools/go2_apex/replay_npz.py``.
It defaults to the B2+Z1 motion NPZ folder and keeps all replay_npz.py flags
available.

Examples:
    python scripts/tools/go2_apex/visualize_b2_z1_motions.py

    python scripts/tools/go2_apex/visualize_b2_z1_motions.py --once

    python scripts/tools/go2_apex/visualize_b2_z1_motions.py \
        --input source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/npz/b2_z1_motions/salut.npz
"""

from __future__ import annotations

import os
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
REPLAY_SCRIPT = REPO_ROOT / "scripts" / "tools" / "go2_apex" / "replay_npz.py"
DEFAULT_MOTION_DIR = (
    REPO_ROOT
    / "source"
    / "Gurukul"
    / "Gurukul"
    / "tasks"
    / "manager_based"
    / "go2_apex"
    / "config"
    / "go2"
    / "motion"
    / "npz"
    / "b2_z1_motions"
)


def _has_arg(args: list[str], *names: str) -> bool:
    return any(arg in names or any(arg.startswith(f"{name}=") for name in names) for arg in args)


def main() -> None:
    args = sys.argv[1:]
    if not _has_arg(args, "--input", "-i"):
        args = ["--input", str(DEFAULT_MOTION_DIR), *args]
    if not _has_arg(args, "--robot"):
        args = ["--robot", "b2_z1", *args]
    if "--zero-xy-origin" not in args:
        args = ["--zero-xy-origin", *args]

    os.execv(sys.executable, [sys.executable, str(REPLAY_SCRIPT), *args])


if __name__ == "__main__":
    main()
