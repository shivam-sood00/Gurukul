"""Shared helpers for the RealisticLidar visual demo scripts."""

from __future__ import annotations

import sys

import torch
from pxr import Gf, UsdGeom

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg


def define_xform(path: str, translation: tuple[float, float, float] = (0.0, 0.0, 0.0)):
    """Create a USD Xform with Isaac Lab's canonical translate/orient ops."""

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    xform = UsdGeom.Xform.Define(stage, path)
    xformable = UsdGeom.Xformable(xform.GetPrim())
    xformable.ClearXformOpOrder()
    translate_op = xformable.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble)
    orient_op = xformable.AddOrientOp()
    scale_op = xformable.AddScaleOp(UsdGeom.XformOp.PrecisionDouble)
    translate_op.Set(Gf.Vec3d(*translation))
    orient_op.Set(Gf.Quatf(1.0, 0.0, 0.0, 0.0))
    scale_op.Set(Gf.Vec3d(1.0, 1.0, 1.0))
    return translate_op, orient_op


def spawn_static_obstacle_scene(prefix: str = "/World") -> list[str]:
    """Spawn ground, boxes, wall, and cylinder for LiDAR verification."""

    ground_path = f"{prefix}/ground"
    ground = sim_utils.GroundPlaneCfg()
    ground.func(ground_path, ground)

    mesh_paths: list[str] = [ground_path]
    box_cfg = sim_utils.MeshCuboidCfg(
        size=(0.6, 0.6, 1.2),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.15, 0.35, 0.9)),
    )
    for index, pos in enumerate(((2.0, 0.0, 0.6), (3.0, 1.25, 0.6), (3.0, -1.25, 0.6))):
        path = f"{prefix}/Box_{index}"
        box_cfg.func(path, box_cfg, translation=pos)
        mesh_paths.append(path)

    wall_cfg = sim_utils.MeshCuboidCfg(
        size=(0.25, 3.0, 1.4),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.85, 0.35, 0.15)),
    )
    wall_path = f"{prefix}/Wall"
    wall_cfg.func(wall_path, wall_cfg, translation=(4.0, 0.0, 0.7))
    mesh_paths.append(wall_path)

    cyl_cfg = sim_utils.MeshCylinderCfg(
        radius=0.35,
        height=1.2,
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.7, 0.35)),
    )
    cyl_path = f"{prefix}/Cylinder"
    cyl_cfg.func(cyl_path, cyl_cfg, translation=(1.4, -1.7, 0.6))
    mesh_paths.append(cyl_path)
    return mesh_paths


def add_light(path: str = "/World/Light", intensity: float = 2500.0) -> None:
    """Add a simple dome light."""

    light_cfg = sim_utils.DomeLightCfg(intensity=float(intensity))
    light_cfg.func(path, light_cfg)


def make_pointcloud_visualizer(
    prim_path: str = "/Visuals/RealisticLidar/PointCloud",
    color: tuple[float, float, float] = (0.0, 1.0, 0.35),
    radius: float = 0.012,
) -> VisualizationMarkers:
    """Create a marker set for explicit point-cloud visualization."""

    marker_cfg = VisualizationMarkersCfg(
        prim_path=prim_path,
        markers={
            "point": sim_utils.SphereCfg(
                radius=float(radius),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
        },
    )
    return VisualizationMarkers(marker_cfg)


def visualize_lidar_pointcloud(
    lidar,
    visualizer: VisualizationMarkers | None,
    *,
    frame: str = "world",
    max_points: int = 5000,
) -> int:
    """Draw point cloud returned by ``RealisticLidar.get_pointcloud`` and return visible point count."""

    if visualizer is None:
        return 0
    valid = lidar.get_valid_mask().reshape(-1)
    points = lidar.get_pointcloud(frame=frame).reshape(-1, 3)
    points = points[valid]
    if frame == "sensor":
        if points.shape[0] == 0:
            return 0
        points = lidar.sensor_to_world(points.reshape(1, -1, 3)).reshape(-1, 3)
    if int(max_points) > 0 and points.shape[0] > int(max_points):
        indices = torch.linspace(0, points.shape[0] - 1, steps=int(max_points), device=points.device).long()
        points = points.index_select(0, indices)
    if points.shape[0] == 0:
        return 0
    visualizer.visualize(translations=points)
    return int(points.shape[0])


def print_lidar_stats(label: str, lidar, step: int | None = None) -> None:
    """Print min/max/mean distance and valid-hit count."""

    distances = lidar.get_distances()
    valid = lidar.get_valid_mask()
    valid_distances = distances[valid]
    prefix = f"[INFO] {label}"
    if step is not None:
        prefix += f" step={step}"
    if valid_distances.numel() == 0:
        print(f"{prefix}: no valid lidar hits", file=sys.__stdout__, flush=True)
        return
    print(
        f"{prefix}: shape={tuple(distances.shape)} min={float(valid_distances.min()):.3f} "
        f"max={float(valid_distances.max()):.3f} mean={float(valid_distances.mean()):.3f} "
        f"valid={int(valid.sum())}/{valid.numel()}",
        file=sys.__stdout__,
        flush=True,
    )
