from isaaclab.utils import configclass

from Gurukul.assets.unitree import UNITREE_G1_BRAINCO_REVO2_CFG

from .flat_env_cfg import UnitreeG1FlatEnvCfg
from .rough_env_cfg import UnitreeG1RoughEnvCfg


@configclass
class UnitreeG1BrainCoRevoRoughEnvCfg(UnitreeG1RoughEnvCfg):
    """G1 rough velocity task tagged with the BrainCo Revo2 hand-control interface."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_G1_BRAINCO_REVO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.disable_zero_weight_rewards()


@configclass
class UnitreeG1BrainCoRevoFlatEnvCfg(UnitreeG1FlatEnvCfg):
    """G1 flat velocity task tagged with the BrainCo Revo2 hand-control interface."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = UNITREE_G1_BRAINCO_REVO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.disable_zero_weight_rewards()
