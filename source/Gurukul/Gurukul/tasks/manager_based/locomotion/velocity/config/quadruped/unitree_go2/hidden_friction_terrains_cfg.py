"""Flat terrain config for hidden-friction contact trail experiments."""

from __future__ import annotations

import numpy as np
import trimesh

from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
from isaaclab.utils import configclass


def hidden_friction_flat_terrain(difficulty: float, cfg: "HiddenFrictionFlatTerrainCfg") -> tuple[list, np.ndarray]:
    """Generate a nearly flat terrain mesh (height variation only from difficulty noise)."""
    del difficulty
    size_x, size_y = cfg.size
    thickness = 0.05
    z_top = cfg.base_height + cfg.height_noise * 0.0
    z_center = z_top - 0.5 * thickness
    transform = trimesh.transformations.translation_matrix((0.5 * size_x, 0.5 * size_y, z_center))
    mesh = trimesh.creation.box((size_x, size_y, thickness), transform)
    return [mesh], np.array([0.5 * size_x, 0.5 * size_y, z_top])


@configclass
class HiddenFrictionFlatTerrainCfg(SubTerrainBaseCfg):
    function = hidden_friction_flat_terrain
    proportion = 1.0
    size = (8.0, 8.0)
    base_height = 0.0
    height_noise = 0.0


HIDDEN_FRICTION_FLAT_TERRAINS_CFG = TerrainGeneratorCfg(
    curriculum=False,
    size=(8.0, 8.0),
    border_width=20.0,
    num_rows=4,
    num_cols=4,
    horizontal_scale=0.1,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": HiddenFrictionFlatTerrainCfg(proportion=1.0),
    },
)
