"""Go2 flat velocity env: mjlab v1 rough settings + flat terrain (mirrors ``flat_env_cfg``)."""

import math

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import Gurukul.tasks.manager_based.locomotion.velocity.mdp as mdp

from .rough_env_cfg_v1 import UnitreeGo2RoughEnvCfgV1


@configclass
class UnitreeGo2FlatEnvCfgV1(UnitreeGo2RoughEnvCfgV1):
    def __post_init__(self):
        super().__post_init__()

        if self.rewards.base_height_l2 is not None:
            self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.observations.teacher_full.height_scan = None
        self.observations.teacher_full.feet_heightmap = None
        self.observations.real_teacher_terrain = None
        self.observations.real_teacher_privileged.feet_heightmap = None
        self.curriculum.terrain_levels = None
        self.rewards.upright.params["terrain_sensor_cfg"] = None
        self.rewards.foot_clearance.params["height_sensor_name"] = None
        self.rewards.foot_swing_height.params["height_sensor_name"] = None
        self.terminations.illegal_contact = None
        self.terminations.terrain_out_of_bounds = None
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(70.0)},
        )
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-1.0, 2.0)

        if self.__class__.__name__ == "UnitreeGo2FlatEnvCfgV1":
            self.disable_zero_weight_rewards()
