# SPDX-License-Identifier: Apache-2.0

"""Visualize B2 + Z1 ArmMoving primitives with the B2 base fixed."""

import argparse
import runpy
import sys
from pathlib import Path

DEFAULT_B2_Z1_TASK = "Gurukul-Isaac-Velocity-Flat-Unitree-B2-Z1-Arm-ArmMoving-v0"
DEFAULT_B2_Z1_WIDE_TASK = "Gurukul-Isaac-Velocity-Flat-Unitree-B2-Z1-Arm-WideArmMoving-v0"


def _has_arg(name: str, args: list[str]) -> bool:
    return name in args or any(arg.startswith(f"{name}=") for arg in args)


def _apply_b2_presets() -> None:
    if "-h" in sys.argv[1:] or "--help" in sys.argv[1:]:
        print("B2+Z1 wrapper presets:")
        print("  --wide-workspace-check   Raw WideArmMoving workspace visualization preset.")
        print()

    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--wide-workspace-check",
        action="store_true",
        help="B2+Z1 preset for raw WideArmMoving task-space workspace visualization.",
    )
    args, remaining = parser.parse_known_args(sys.argv[1:])
    sys.argv = [sys.argv[0], *remaining]

    if not args.wide_workspace_check:
        return

    if not _has_arg("--task", remaining):
        sys.argv.extend(["--task", DEFAULT_B2_Z1_WIDE_TASK])
    if not _has_arg("--primitive", remaining):
        sys.argv.extend(["--primitive", "workspace"])
    for flag in ("--no-interpolation", "--full-workspace", "--free-ee-orientation"):
        if not _has_arg(flag, remaining):
            sys.argv.append(flag)


def _main() -> None:
    _apply_b2_presets()
    if not _has_arg("--task", sys.argv[1:]):
        sys.argv.extend(["--task", DEFAULT_B2_Z1_TASK])
    if "--base-height" not in sys.argv and "--free-base" not in sys.argv:
        sys.argv.extend(["--base-height", "0.75"])

    visualizer = Path(__file__).with_name("visualize_go2_airbot_arm_motion.py")
    runpy.run_path(str(visualizer), run_name="__main__")


if __name__ == "__main__":
    _main()
