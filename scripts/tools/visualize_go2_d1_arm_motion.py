# SPDX-License-Identifier: Apache-2.0

"""Visualize Go2 + D1 moving-arm primitives without training or loading a policy."""

import runpy
import sys
from pathlib import Path

DEFAULT_D1_TASK = "Gurukul-Isaac-Velocity-Flat-Unitree-Go2-D1-Arm-ArmMoving-v0"


def _main() -> None:
    if "--task" not in sys.argv:
        sys.argv.extend(["--task", DEFAULT_D1_TASK])

    visualizer = Path(__file__).with_name("visualize_go2_airbot_arm_motion.py")
    runpy.run_path(str(visualizer), run_name="__main__")


if __name__ == "__main__":
    _main()
