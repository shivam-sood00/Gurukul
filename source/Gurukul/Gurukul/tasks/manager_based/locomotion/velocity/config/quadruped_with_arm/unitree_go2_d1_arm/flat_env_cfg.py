# SPDX-License-Identifier: Apache-2.0

import math

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import Gurukul.tasks.manager_based.locomotion.velocity.mdp as mdp

from .rough_env_cfg import UnitreeGo2D1ArmRoughEnvCfg


@configclass
class UnitreeGo2D1ArmFlatEnvCfg(UnitreeGo2D1ArmRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None
        # A plane has no meaningful terrain boundary. Terminate actual falls so a
        # tumbling robot cannot contaminate the remainder of a 20-second rollout.
        self.terminations.terrain_out_of_bounds = None
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(70.0)},
        )

        if self.__class__.__name__ == "UnitreeGo2D1ArmFlatEnvCfg":
            self.disable_zero_weight_rewards()
