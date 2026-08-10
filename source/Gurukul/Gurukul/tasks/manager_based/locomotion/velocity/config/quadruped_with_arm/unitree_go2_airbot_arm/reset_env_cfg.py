# Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

import Gurukul.tasks.manager_based.locomotion.velocity.mdp as mdp

GO2_BASE_RESET_PARAMS = {
    "pose_range": {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (0.0, 0.2),
        "roll": (-0.25, 0.25),
        "pitch": (-0.25, 0.25),
        "yaw": (-1.57, 1.57),
    },
    "velocity_range": {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (-0.5, 0.5),
        "roll": (-0.5, 0.5),
        "pitch": (-0.5, 0.5),
        "yaw": (-0.5, 0.5),
    },
}

GO2_JOINT_RESET_PARAMS = {
    "position_range": (-0.1, 0.1),
    "velocity_range": (-0.1, 0.1),
}


def configure_go2_like_resets(env_cfg) -> None:
    """Use the same reset randomization as the standard Go2 flat/rough tasks."""
    env_cfg.events.randomize_reset_base.params = GO2_BASE_RESET_PARAMS
    env_cfg.events.randomize_reset_joints.func = mdp.reset_joints_by_offset
    env_cfg.events.randomize_reset_joints.params = GO2_JOINT_RESET_PARAMS
