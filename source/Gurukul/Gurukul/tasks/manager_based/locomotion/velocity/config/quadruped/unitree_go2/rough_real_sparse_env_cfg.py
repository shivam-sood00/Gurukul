from __future__ import annotations

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import Gurukul.tasks.manager_based.locomotion.velocity.mdp as mdp

from .rough_env_cfg import UnitreeGo2RoughEnvCfg
from .start_terrains_cfg import START_SPARSE_TERRAINS_CFG


@configclass
class UnitreeGo2RoughRealSparseEnvCfg(UnitreeGo2RoughEnvCfg):
    """Go2 REAL teacher rough-velocity task on sparse terrains with terrain curriculum."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.terrain.terrain_generator = START_SPARSE_TERRAINS_CFG.replace(curriculum=True)
        self.scene.terrain.max_init_terrain_level = 2
        self.scene.terrain.terrain_generator.curriculum = True

        self.curriculum.terrain_levels = CurrTerm(
            func=mdp.start_terrain_levels_progression,
            params={
                "easy_terrain_names": ("flat", "stepping_stones_low_random"),
                "advanced_terrain_names": (
                    "stepping_stones_high_random",
                    "balance_beams",
                    "stepping_beams",
                ),
                "gap_terrain_names": ("gaps",),
                "p_max": 1.0,
                "t_start": 2.0e4,
                "t_end": 1.8e5,
                "gap_probability": 0.08,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )


        # Disable generic rough-locomotion priors that can fight sparse foothold selection.
        self.rewards.feet_height_body.weight = 0.0
        self.rewards.feet_gait.weight = 0.0
        self.rewards.feet_air_time_variance.weight = 0.0
        self.rewards.joint_mirror.weight = 0.0

        self.disable_zero_weight_rewards()
