# SPDX-License-Identifier: Apache-2.0

import Gurukul.tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

from .rough_env_cfg import EngineAiPm01RoughEnvCfg


@configclass
class EngineAiPm01FlatEnvCfg(EngineAiPm01RoughEnvCfg):
    """Flat plane PM01 — same MDP stack as rough without terrain curriculum / height scan."""

    def __post_init__(self) -> None:
        super().__post_init__()

        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.scene.height_scanner_base = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None

        self.rewards.lin_vel_z_l2.weight = -0.2

        # The flat task has no terrain failures to terminate a fallen rollout.
        # Stop collecting low-quality data once the robot is clearly down.
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": 0.8},
        )
        self.terminations.base_height = DoneTerm(
            func=mdp.root_height_below_minimum,
            params={"minimum_height": 0.45},
        )

        # First establish an upright tracking policy on the plane; pushes can
        # be re-enabled once the policy is stable.
        self.events.randomize_push_robot = None

        if self.__class__.__name__ == "EngineAiPm01FlatEnvCfg":
            self.disable_zero_weight_rewards()
