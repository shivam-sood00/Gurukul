from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

pytest.importorskip("isaaclab")
pytest.importorskip("pxr")

REPO_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = REPO_ROOT / "source" / "Gurukul"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

pytest.importorskip(
    "Gurukul.assets.realistic_lidar",
    reason="The optional OmniPerception-derived lidar adapter is not distributed with the public repository.",
)

from Gurukul.assets.realistic_lidar import (  # noqa: E402
    RealisticLidar,
    RealisticLidarData,
    livox_realistic_pattern,
    load_livox_pattern_data,
    make_lidar_grid_pattern_cfg,
    make_livox_mid360_pattern_cfg,
    make_realistic_lidar_cfg,
)


def test_mid360_pattern_loading():
    pattern_data = load_livox_pattern_data("mid360")
    assert pattern_data.ndim == 2
    assert pattern_data.shape[1] >= 2

    cfg = make_livox_mid360_pattern_cfg(samples=256, rolling=False)
    starts, directions = livox_realistic_pattern(cfg, "cpu")
    assert starts.shape == (256, 3)
    assert directions.shape == (256, 3)
    assert torch.allclose(torch.linalg.norm(directions, dim=-1), torch.ones(256), atol=1.0e-5)


def test_grid_pattern_fallback():
    cfg = make_lidar_grid_pattern_cfg(channels=2, horizontal_fov=90.0, horizontal_res=45.0)
    starts, directions = cfg.func(cfg, "cpu")
    assert starts.shape[-1] == 3
    assert directions.shape[-1] == 3
    assert starts.shape[0] == directions.shape[0]
    assert directions.shape[0] > 0


def test_realistic_lidar_cfg_creation():
    cfg = make_realistic_lidar_cfg(prim_path="/World/Lidar", mesh_prim_paths=["/World/ground"])
    assert cfg.class_type is RealisticLidar
    assert cfg.pattern_cfg.samples == 6000
    assert cfg.min_range == pytest.approx(0.1)
    assert cfg.max_distance == pytest.approx(200.0)


def _make_synthetic_sensor(*, min_range: float = 0.1, max_distance: float = 10.0, dropout: float = 0.0):
    cfg = make_realistic_lidar_cfg(
        prim_path="/World/Lidar",
        mesh_prim_paths=["/World/ground"],
        min_range=min_range,
        max_distance=max_distance,
        ray_dropout_prob=dropout,
        enable_sensor_noise=False,
    )
    sensor = object.__new__(RealisticLidar)
    sensor.cfg = cfg
    sensor._device = "cpu"
    sensor._update_outdated_buffers = lambda: None

    ray_dirs = torch.tensor(
        [[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [-1.0, 0.0, 0.0]]],
        dtype=torch.float32,
    )
    ray_starts_w = torch.zeros((1, 4, 3), dtype=torch.float32)
    hit_distances = torch.tensor([[0.05, 1.0, 5.0, 300.0]], dtype=torch.float32)
    ray_hits_w = ray_starts_w + ray_dirs * hit_distances.unsqueeze(-1)

    sensor.ray_directions = ray_dirs
    sensor._ray_directions_w = ray_dirs
    sensor._ray_starts_w = ray_starts_w
    sensor._data = RealisticLidarData(
        pos_w=torch.zeros((1, 3), dtype=torch.float32),
        quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]], dtype=torch.float32),
        ray_hits_w=ray_hits_w,
        distances=torch.zeros((1, 4), dtype=torch.float32),
        pointcloud_sensor=torch.zeros((1, 4, 3), dtype=torch.float32),
        pointcloud_world=torch.zeros((1, 4, 3), dtype=torch.float32),
        ray_hits_sensor=torch.zeros((1, 4, 3), dtype=torch.float32),
        ray_hits_world=torch.zeros((1, 4, 3), dtype=torch.float32),
        valid_mask=torch.zeros((1, 4), dtype=torch.bool),
    )
    return sensor


def test_distance_pointcloud_and_valid_mask_shapes():
    sensor = _make_synthetic_sensor()
    sensor._update_lidar_outputs(torch.tensor([0]))

    assert sensor.get_distances().shape == (1, 4)
    assert sensor.get_pointcloud(frame="sensor").shape == (1, 4, 3)
    assert sensor.get_pointcloud(frame="world").shape == (1, 4, 3)
    assert sensor.get_valid_mask().shape == (1, 4)
    assert sensor.get_valid_mask().tolist() == [[False, True, True, False]]


def test_range_clipping_and_invalid_hits():
    sensor = _make_synthetic_sensor(min_range=0.1, max_distance=10.0)
    sensor._update_lidar_outputs(torch.tensor([0]))

    distances = sensor.get_distances()[0]
    assert distances.tolist() == pytest.approx([10.0, 1.0, 5.0, 10.0])
    assert torch.all(sensor.get_pointcloud(frame="sensor")[0, [0, 3]] == 0.0)


def test_frame_conversion_correctness_identity_pose():
    sensor = _make_synthetic_sensor()
    sensor._update_lidar_outputs(torch.tensor([0]))

    valid = sensor.get_valid_mask()
    converted = sensor.sensor_to_world(sensor.get_pointcloud(frame="sensor"))
    assert torch.allclose(converted[valid], sensor.get_pointcloud(frame="world")[valid])
    assert torch.allclose(sensor.world_to_sensor(sensor.get_pointcloud(frame="world"))[valid], sensor.get_pointcloud()[valid])


def test_dropout_behavior():
    torch.manual_seed(7)
    sensor = _make_synthetic_sensor(dropout=1.0)
    sensor._update_lidar_outputs(torch.tensor([0]))

    assert not torch.any(sensor.get_valid_mask())
    assert torch.all(sensor.get_distances() == sensor.cfg.max_distance)
