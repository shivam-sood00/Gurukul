import math

from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import Gurukul.tasks.manager_based.locomotion.velocity.mdp as mdp

from .mjlab_parity_constants import GO2_MJLAB_PARITY_ACTION_SCALE
from .rough_env_cfg import UnitreeGo2RoughEnvCfg


@configclass
class UnitreeGo2FlatEnvCfg(UnitreeGo2RoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # override rewards
        if self.rewards.base_height_l2 is not None:
            self.rewards.base_height_l2.params["sensor_cfg"] = None
        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.observations.teacher_full.height_scan = None
        self.observations.teacher_full.feet_heightmap = None
        self.observations.real_teacher_terrain = None
        self.observations.real_teacher_privileged.feet_heightmap = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None
        self.terminations.terrain_out_of_bounds = None
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": math.radians(70.0)},
        )
        # flat-only DR override for base mass randomization
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-1.0, 2.0)

        # If the weight of rewards is 0, set rewards to None
        if self.__class__.__name__ == "UnitreeGo2FlatEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class UnitreeGo2FlatMjlabActionScaleEnvCfg(UnitreeGo2FlatEnvCfg):
    """Flat v0 reward/reset surface with only the mjlab Go2 action scale."""

    def __post_init__(self):
        super().__post_init__()

        self.actions.joint_pos.scale = GO2_MJLAB_PARITY_ACTION_SCALE

        if self.__class__.__name__ == "UnitreeGo2FlatMjlabActionScaleEnvCfg":
            self.disable_zero_weight_rewards()
