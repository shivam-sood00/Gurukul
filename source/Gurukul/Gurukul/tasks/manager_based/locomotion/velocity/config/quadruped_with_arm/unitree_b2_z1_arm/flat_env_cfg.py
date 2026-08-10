# SPDX-License-Identifier: Apache-2.0

from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeB2Z1ArmRoughEnvCfg


@configclass
class UnitreeB2Z1ArmFlatEnvCfg(UnitreeB2Z1ArmRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.rewards.base_height_l2.params["sensor_cfg"] = None
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.scene.height_scanner = None
        self.observations.policy.height_scan = None
        self.observations.critic.height_scan = None
        self.curriculum.terrain_levels = None
        self.events.randomize_rigid_body_mass_base.params["mass_distribution_params"] = (-1.0, 2.0)
        self.rewards.feet_air_time.weight = 0.5

        if self.__class__.__name__ == "UnitreeB2Z1ArmFlatEnvCfg":
            self.disable_zero_weight_rewards()
