from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from isaaclab.assets import RigidObject
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def foot_height_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=""),
) -> torch.Tensor:
    """Terminate when any selected foot drops below a world-frame height threshold."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.any(asset.data.body_pos_w[:, asset_cfg.body_ids, 2] < float(minimum_height), dim=1)
