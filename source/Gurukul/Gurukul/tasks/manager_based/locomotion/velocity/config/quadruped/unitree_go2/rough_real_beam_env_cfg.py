from __future__ import annotations

from isaaclab.utils import configclass

from .rough_env_cfg import UnitreeGo2RoughEnvCfg
from .start_terrains_cfg import REAL_BEAM_GAP_TERRAINS_CFG


@configclass
class UnitreeGo2RoughRealBeamEnvCfg(UnitreeGo2RoughEnvCfg):
    """Go2 REAL teacher rough-velocity task on a forward-crossing beam-and-gap terrain mix."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 4096
        self.scene.terrain.terrain_generator = REAL_BEAM_GAP_TERRAINS_CFG.replace(curriculum=True)
        self.scene.terrain.max_init_terrain_level = 0
        self.scene.terrain.terrain_generator.curriculum = True


        # Keep commands aligned with forward beam-crossing: positive forward velocity, no lateral drift, small yaw.
        self.commands.base_velocity.heading_command = False
        self.commands.base_velocity.rel_heading_envs = 0.0
        self.commands.base_velocity.rel_standing_envs = 0.0
        self.commands.base_velocity.ranges.lin_vel_x = (0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.4, 0.4)


        # Disable generic rough-locomotion priors that can fight sparse foothold selection.
        self.rewards.feet_height_body.weight = 0.0
        self.rewards.feet_gait.weight = 0.0
        self.rewards.feet_air_time_variance.weight = 0.0
        self.rewards.joint_mirror.weight = 0.0

        self.disable_zero_weight_rewards()
