#!/usr/bin/env python3
"""Replay the converted ``corss_morpho_new`` Go2 APEX motions in Isaac Lab.

This is a convenience wrapper around ``scripts/tools/go2_apex/replay_npz.py``.
It defaults to the converted NPZ folder and keeps all replay_npz.py flags
available.

Examples:
    python scripts/tools/go2_apex/visualize_corss_morpho_new.py

    python scripts/tools/go2_apex/visualize_corss_morpho_new.py --once

    python scripts/tools/go2_apex/visualize_corss_morpho_new.py \
        --input source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/npz/corss_morpho_new/walk1_subject1_go2.npz
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
    / "cross_morpho_replay_added"
)


def _has_input_arg(args: list[str]) -> bool:
    return any(arg in ("--input", "-i") or arg.startswith("--input=") for arg in args)


def main() -> None:
    args = sys.argv[1:]
    if not _has_input_arg(args):
        args = ["--input", str(DEFAULT_MOTION_DIR), "--zero-xy-origin", *args]

    os.execv(sys.executable, [sys.executable, str(REPLAY_SCRIPT), *args])


if __name__ == "__main__":
    main()
