"""START sparse-foothold terrain configuration for Go2."""

from __future__ import annotations

from dataclasses import MISSING, dataclass, replace as dataclass_replace

import numpy as np
import trimesh

try:
    from isaaclab.terrains.sub_terrain_cfg import SubTerrainBaseCfg
    from isaaclab.terrains.terrain_generator_cfg import TerrainGeneratorCfg
    from isaaclab.utils import configclass
except Exception:
    def configclass(cls=None, **kwargs):
        def wrap(klass):
            annotations = dict(getattr(klass, "__annotations__", {}))
            for name, value in list(klass.__dict__.items()):
                if name.startswith("__") or name in {"replace", "copy"}:
                    continue
                if isinstance(value, (staticmethod, classmethod, property)) or callable(value):
                    continue
                if name not in annotations:
                    annotations[name] = object if value is MISSING else type(value)
            klass.__annotations__ = annotations
            wrapped = dataclass(klass, kw_only=True, **kwargs)
            setattr(wrapped, "replace", lambda self, **updates: dataclass_replace(self, **updates))
            setattr(wrapped, "copy", lambda self: dataclass_replace(self))
            return wrapped

        return wrap(cls) if cls is not None else wrap

    @configclass
    class SubTerrainBaseCfg:
        function: object | None = None
        proportion: float = 1.0
        size: tuple[float, float] = (10.0, 10.0)
        flat_patch_sampling: dict | None = None

    @configclass
    class TerrainGeneratorCfg:
        class_type: type | None = None
        seed: int | None = None
        curriculum: bool = False
        size: tuple[float, float] = MISSING
        border_width: float = 0.0
        border_height: float = 1.0
        num_rows: int = 1
        num_cols: int = 1
        color_scheme: str = "none"
        horizontal_scale: float = 0.1
        vertical_scale: float = 0.005
        slope_threshold: float | None = 0.75
        sub_terrains: dict[str, SubTerrainBaseCfg] = MISSING
        difficulty_range: tuple[float, float] = (0.0, 1.0)
        use_cache: bool = False
        cache_dir: str = "/tmp/isaaclab/terrains"


def _lin_interp(value_range: tuple[float, float], difficulty: float) -> float:
    return float(value_range[0] + difficulty * (value_range[1] - value_range[0]))


def _descending_interp(value_range: tuple[float, float], difficulty: float) -> float:
    return float(value_range[1] - difficulty * (value_range[1] - value_range[0]))


def _make_box(size: tuple[float, float, float], center: tuple[float, float, float]) -> trimesh.Trimesh:
    transform = trimesh.transformations.translation_matrix(center)
    return trimesh.creation.box(size, transform)


def _make_base_floor(size: tuple[float, float], depth: float, thickness: float = 0.04) -> trimesh.Trimesh:
    # Top of the floor is at -depth, creating negative obstacles around footholds.
    z_center = -depth - 0.5 * thickness
    return _make_box((size[0], size[1], depth + thickness), (0.5 * size[0], 0.5 * size[1], z_center))


def start_flat_terrain(difficulty: float, cfg: "StartFlatTerrainCfg") -> tuple[list[trimesh.Trimesh], np.ndarray]:
    size_x, size_y = cfg.size
    depth = _lin_interp(cfg.depth_range, difficulty)
    top_height = np.random.uniform(-cfg.height_variation, cfg.height_variation)
    bottom_height = -depth
    platform_height = max(0.02, top_height - bottom_height)
    platform_center_z = 0.5 * (top_height + bottom_height)

    meshes = [
        _make_base_floor(cfg.size, depth),
        _make_box((size_x, size_y, platform_height), (0.5 * size_x, 0.5 * size_y, platform_center_z)),
    ]
    origin = np.array([0.5 * size_x, 0.5 * size_y, 0.0], dtype=np.float32)
    return meshes, origin


def start_stepping_stones_terrain(
    difficulty: float, cfg: "StartSteppingStonesTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    size_x, size_y = cfg.size
    depth = _lin_interp(cfg.depth_range, difficulty)
    stone_width = _descending_interp(cfg.stone_width_range, difficulty)
    gap_x = _lin_interp(cfg.gap_x_range, difficulty)
    gap_y = _lin_interp(cfg.gap_y_range, difficulty)

    step_x = stone_width + gap_x
    step_y = stone_width + gap_y
    x_centers = np.arange(0.5 * stone_width, size_x - 0.5 * stone_width + 1.0e-6, step_x)
    y_centers = np.arange(0.5 * stone_width, size_y - 0.5 * stone_width + 1.0e-6, step_y)

    meshes: list[trimesh.Trimesh] = [_make_base_floor(cfg.size, depth)]

    per_col_x_shift = np.zeros(len(x_centers), dtype=np.float32)
    if cfg.randomize_stones:
        col_shift_mag = cfg.column_shift_ratio * gap_x
        per_col_x_shift = np.random.uniform(-col_shift_mag, col_shift_mag, size=len(x_centers)).astype(np.float32)

    stone_shift_x = cfg.stone_shift_ratio * gap_x if cfg.randomize_stones else 0.0
    stone_shift_y = cfg.stone_shift_ratio * gap_y if cfg.randomize_stones else 0.0

    for col_idx, x_center in enumerate(x_centers):
        for y_center in y_centers:
            jitter_x = np.random.uniform(-stone_shift_x, stone_shift_x) if stone_shift_x > 0.0 else 0.0
            jitter_y = np.random.uniform(-stone_shift_y, stone_shift_y) if stone_shift_y > 0.0 else 0.0
            cx = np.clip(x_center + per_col_x_shift[col_idx] + jitter_x, 0.5 * stone_width, size_x - 0.5 * stone_width)
            cy = np.clip(y_center + jitter_y, 0.5 * stone_width, size_y - 0.5 * stone_width)

            top_height = np.random.uniform(-cfg.height_variation, cfg.height_variation)
            bottom_height = -depth
            height = max(0.02, top_height - bottom_height)
            center_z = 0.5 * (top_height + bottom_height)
            meshes.append(_make_box((stone_width, stone_width, height), (cx, cy, center_z)))

    origin = np.array([0.5 * size_x, 0.5 * size_y, 0.0], dtype=np.float32)
    return meshes, origin


def start_balance_beam_terrain(
    difficulty: float, cfg: "StartBalanceBeamTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    size_x, size_y = cfg.size
    depth = _lin_interp(cfg.depth_range, difficulty)
    beam_width = _descending_interp(cfg.beam_width_range, difficulty)

    top_height = np.random.uniform(-cfg.height_variation, cfg.height_variation)
    bottom_height = -depth
    beam_height = max(0.02, top_height - bottom_height)
    center_z = 0.5 * (top_height + bottom_height)

    meshes = [
        _make_base_floor(cfg.size, depth),
        _make_box((size_x, beam_width, beam_height), (0.5 * size_x, 0.5 * size_y, center_z)),
    ]
    origin = np.array([0.5 * size_x, 0.5 * size_y, 0.0], dtype=np.float32)
    return meshes, origin


def start_stepping_beam_terrain(
    difficulty: float, cfg: "StartSteppingBeamTerrainCfg"
) -> tuple[list[trimesh.Trimesh], np.ndarray]:
    size_x, size_y = cfg.size
    depth = _lin_interp(cfg.depth_range, difficulty)
    beam_width = _descending_interp(cfg.beam_width_range, difficulty)
    gap_x = _lin_interp(cfg.gap_x_range, difficulty)

    top_height = np.random.uniform(-cfg.height_variation, cfg.height_variation)
    bottom_height = -depth
    beam_height = max(0.02, top_height - bottom_height)
    center_z = 0.5 * (top_height + bottom_height)

    meshes: list[trimesh.Trimesh] = [_make_base_floor(cfg.size, depth)]

    stride_x = beam_width + gap_x
    x_start = 0.0
    while x_start < size_x:
        seg_width = min(beam_width, size_x - x_start)
        if seg_width <= 0.0:
            break
        y_center = 0.5 * size_y
        if cfg.randomize_lateral:
            lateral_room = max(0.0, 0.5 * (size_y - cfg.segment_length_y))
            y_center += np.random.uniform(-lateral_room, lateral_room)
            seg_length_y = min(size_y, cfg.segment_length_y)
        else:
            seg_length_y = size_y
        meshes.append(
            _make_box(
                (seg_width, seg_length_y, beam_height),
                (x_start + 0.5 * seg_width, y_center, center_z),
            )
        )
        x_start += stride_x

    origin = np.array([0.5 * size_x, 0.5 * size_y, 0.0], dtype=np.float32)
    return meshes, origin


def start_gap_terrain(difficulty: float, cfg: "StartGapTerrainCfg") -> tuple[list[trimesh.Trimesh], np.ndarray]:
    size_x, size_y = cfg.size
    depth = _lin_interp(cfg.depth_range, difficulty)
    gap_width = _lin_interp(cfg.gap_width_range, difficulty)
    gap_width = float(np.clip(gap_width, 0.1, size_x - 1.0))

    rear_length = max(0.4, 0.5 * size_x - 0.5 * gap_width)
    front_length = max(0.4, size_x - rear_length - gap_width)

    rear_top = np.random.uniform(-cfg.height_variation, cfg.height_variation)
    front_top = np.random.uniform(-cfg.height_variation, cfg.height_variation)
    bottom = -depth

    rear_height = max(0.02, rear_top - bottom)
    front_height = max(0.02, front_top - bottom)
    rear_center_z = 0.5 * (rear_top + bottom)
    front_center_z = 0.5 * (front_top + bottom)

    rear_center_x = 0.5 * rear_length
    front_center_x = rear_length + gap_width + 0.5 * front_length

    meshes = [
        _make_base_floor(cfg.size, depth),
        _make_box((rear_length, size_y, rear_height), (rear_center_x, 0.5 * size_y, rear_center_z)),
        _make_box((front_length, size_y, front_height), (front_center_x, 0.5 * size_y, front_center_z)),
    ]
    origin = np.array([0.5 * size_x, 0.5 * size_y, 0.0], dtype=np.float32)
    return meshes, origin


@configclass
class StartFlatTerrainCfg(SubTerrainBaseCfg):
    function = start_flat_terrain

    depth_range: tuple[float, float] = (0.2, 0.2)
    height_variation: float = 0.0


@configclass
class StartSteppingStonesTerrainCfg(SubTerrainBaseCfg):
    function = start_stepping_stones_terrain

    stone_width_range: tuple[float, float] = MISSING
    gap_x_range: tuple[float, float] = MISSING
    gap_y_range: tuple[float, float] = MISSING
    depth_range: tuple[float, float] = (0.2, 0.7)
    height_variation: float = 0.05
    randomize_stones: bool = False
    stone_shift_ratio: float = 0.5
    column_shift_ratio: float = 0.45


@configclass
class StartBalanceBeamTerrainCfg(SubTerrainBaseCfg):
    function = start_balance_beam_terrain

    beam_width_range: tuple[float, float] = MISSING
    depth_range: tuple[float, float] = (0.2, 0.7)
    height_variation: float = 0.05


@configclass
class StartSteppingBeamTerrainCfg(SubTerrainBaseCfg):
    function = start_stepping_beam_terrain

    beam_width_range: tuple[float, float] = MISSING
    gap_x_range: tuple[float, float] = MISSING
    depth_range: tuple[float, float] = (0.2, 0.7)
    height_variation: float = 0.05
    randomize_lateral: bool = False
    segment_length_y: float = 4.0


@configclass
class StartGapTerrainCfg(SubTerrainBaseCfg):
    function = start_gap_terrain

    gap_width_range: tuple[float, float] = MISSING
    depth_range: tuple[float, float] = (0.2, 0.7)
    height_variation: float = 0.0


START_SPARSE_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 4.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "flat": StartFlatTerrainCfg(proportion=0.18, depth_range=(0.2, 0.2), height_variation=0.0),
        "stepping_stones_low_random": StartSteppingStonesTerrainCfg(
            proportion=0.28,
            stone_width_range=(0.2, 0.2),
            gap_x_range=(0.25, 0.25),
            gap_y_range=(0.175, 0.175),
            randomize_stones=False,
            depth_range=(0.2, 0.7),
            height_variation=0.05,
        ),
        "stepping_stones_high_random": StartSteppingStonesTerrainCfg(
            proportion=0.18,
            stone_width_range=(0.2, 0.2),
            gap_x_range=(0.25, 0.25),
            gap_y_range=(0.175, 0.175),
            randomize_stones=True,
            stone_shift_ratio=0.5,
            column_shift_ratio=0.45,
            depth_range=(0.2, 0.7),
            height_variation=0.05,
        ),
        "balance_beams": StartBalanceBeamTerrainCfg(
            proportion=0.14,
            beam_width_range=(0.175, 0.35),
            depth_range=(0.2, 0.7),
            height_variation=0.05,
        ),
        "stepping_beams": StartSteppingBeamTerrainCfg(
            proportion=0.14,
            beam_width_range=(0.15, 0.25),
            gap_x_range=(0.15, 0.275),
            depth_range=(0.2, 0.7),
            height_variation=0.05,
            randomize_lateral=False,
            segment_length_y=4.0,
        ),
        "gaps": StartGapTerrainCfg(
            proportion=0.08,
            gap_width_range=(0.2, 0.7),
            depth_range=(0.2, 0.7),
            height_variation=0.0,
        ),
    },
)



REAL_BEAM_GAP_TERRAINS_CFG = TerrainGeneratorCfg(
    size=(8.0, 4.0),
    border_width=20.0,
    num_rows=10,
    num_cols=20,
    horizontal_scale=0.05,
    vertical_scale=0.005,
    slope_threshold=0.75,
    use_cache=False,
    sub_terrains={
        "stepping_beams": StartSteppingBeamTerrainCfg(
            proportion=0.6,
            beam_width_range=(0.18, 0.3),
            gap_x_range=(0.35, 0.8),
            depth_range=(0.3, 0.8),
            height_variation=0.0,
            randomize_lateral=False,
            segment_length_y=4.0,
        ),
        "gaps": StartGapTerrainCfg(
            proportion=0.4,
            gap_width_range=(0.35, 0.85),
            depth_range=(0.3, 0.8),
            height_variation=0.0,
        ),
    },
)
