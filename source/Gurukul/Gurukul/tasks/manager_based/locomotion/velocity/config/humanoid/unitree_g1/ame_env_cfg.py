from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import patterns
from isaaclab.utils import configclass

import Gurukul.tasks.manager_based.locomotion.velocity.mdp as mdp

from .rough_env_cfg import UnitreeG1RoughEnvCfg


@configclass
class UnitreeG1AMERoughEnvCfg(UnitreeG1RoughEnvCfg):
    """Unitree G1 rough velocity task with AME XYZ terrain-map observations."""

    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 2048
        self.scene.height_scanner.pattern_cfg = patterns.GridPatternCfg(resolution=0.05, size=[1.6, 1.0])
        self.observations.policy.height_scan = ObsTerm(
            func=mdp.elevation_map,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "noise": True},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        self.observations.critic.height_scan = ObsTerm(
            func=mdp.elevation_map,
            params={"sensor_cfg": SceneEntityCfg("height_scanner"), "noise": False},
            clip=(-100.0, 100.0),
            scale=1.0,
        )
        self.disable_zero_weight_rewards()
