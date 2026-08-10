"""Visualize robot lidar mounts with OmniPerception-style hit point markers."""

from __future__ import annotations

import argparse
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description=(
        "Spawn one robot with a configured lidar mount, use repo-local OmniPerception scan patterns, and draw "
        "RayCaster hit points as red marker spheres."
    )
)
parser.add_argument(
    "--robot",
    default="go2",
    choices=("go2", "go2_airbot", "go2_d1", "g1"),
    help="Robot asset to spawn.",
)
parser.add_argument(
    "--mount",
    default=None,
    help="Mount name to visualize. Defaults to the first mount declared on the selected robot.",
)
parser.add_argument(
    "--terrain",
    default="flat",
    choices=("flat", "uneven"),
    help="Terrain to raycast against.",
)
parser.add_argument(
    "--duration-seconds",
    type=float,
    default=0.0,
    help="Viewer duration. Use 0 to run until the Isaac Sim window is closed.",
)
parser.add_argument(
    "--real-time",
    action="store_true",
    default=False,
    help="Throttle stepping to real time.",
)
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument(
    "--free-root",
    action="store_true",
    default=False,
    help="Let the robot root move under physics. By default, the root is fixed for lidar inspection.",
)
parser.add_argument(
    "--print-every",
    type=int,
    default=60,
    help="Print lidar hit tensor status every N sim steps. Use 0 to disable.",
)
parser.add_argument(
    "--point-radius",
    type=float,
    default=0.015,
    help="Radius of each lidar hit marker sphere in meters.",
)
parser.add_argument(
    "--max-points",
    type=int,
    default=30000,
    help="Maximum finite lidar hit points to draw each frame. Use 0 to draw all points.",
)
parser.add_argument(
    "--dynamic-obstacles",
    action="store_true",
    default=False,
    help="Add moving analytic sphere obstacles to the OmniPerception-style lidar ray hits.",
)
parser.add_argument(
    "--dynamic-mesh-obstacles",
    action="store_true",
    default=False,
    help="Add moving USD mesh obstacles to the OmniPerception-style lidar Warp raycast path.",
)
parser.add_argument(
    "--dynamic-obstacle-count",
    type=int,
    default=2,
    help="Number of moving sphere obstacles when --dynamic-obstacles is set.",
)
parser.add_argument(
    "--dynamic-obstacle-radius",
    type=float,
    default=0.22,
    help="Radius of each moving sphere obstacle, or half-width of each moving mesh pillar, in meters.",
)
parser.add_argument(
    "--dynamic-mesh-obstacle-height",
    type=float,
    default=0.9,
    help="Height of each moving mesh pillar in meters.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import RayCaster
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG

from Gurukul.assets.lidar import make_omni_perception_lidar_ray_caster_cfg
from Gurukul.assets.unitree import (
    UNITREE_G1_29DOF_CFG,
    UNITREE_GO2_AIRBOT_ARM_CFG,
    UNITREE_GO2_CFG,
    UNITREE_GO2_D1_ARM_CFG,
)


ROBOT_CFGS = {
    "go2": UNITREE_GO2_CFG,
    "go2_airbot": UNITREE_GO2_AIRBOT_ARM_CFG,
    "go2_d1": UNITREE_GO2_D1_ARM_CFG,
    "g1": UNITREE_G1_29DOF_CFG,
}

CAMERA_VIEWS = {
    "go2": ([1.8, -2.1, 1.2], [0.0, 0.0, 0.35]),
    "go2_airbot": ([1.8, -2.1, 1.2], [0.0, 0.0, 0.4]),
    "go2_d1": ([1.9, -2.2, 1.35], [0.0, 0.0, 0.45]),
    "g1": ([2.4, -3.0, 1.8], [0.0, 0.0, 0.8]),
}


def _select_mount(robot_cfg, mount_name: str | None):
    mounts = getattr(robot_cfg, "lidar_mounts", None)
    if not mounts:
        raise RuntimeError(f"{args_cli.robot}: no lidar_mounts declared.")
    if mount_name is None:
        return next(iter(mounts.items()))
    if mount_name not in mounts:
        available = ", ".join(mounts)
        raise RuntimeError(f"{args_cli.robot}: unknown lidar mount {mount_name!r}. Available mounts: {available}")
    return mount_name, mounts[mount_name]


def _make_scene_cfg(robot_cfg, mount) -> InteractiveSceneCfg:
    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)

    robot_spawn_cfg = robot_cfg.replace(prim_path="/World/Robot")
    if not args_cli.free_root and getattr(robot_spawn_cfg.spawn, "articulation_props", None) is not None:
        robot_spawn_cfg.spawn.articulation_props.fix_root_link = True
    scene_cfg.robot = robot_spawn_cfg

    return scene_cfg


def _make_lidar_cfg(mount, dynamic_mesh_prim_paths: tuple[str, ...] = ()):
    lidar_cfg = make_omni_perception_lidar_ray_caster_cfg(mount, debug_vis=False)
    lidar_cfg.prim_path = f"/World/Robot/{mount.parent_link}"
    lidar_cfg.mesh_prim_paths = [_terrain_mesh_prim_path()]
    lidar_cfg.dynamic_mesh_prim_paths = dynamic_mesh_prim_paths
    lidar_cfg.visualizer_cfg.prim_path = "/Visuals/RobotLidarRays"
    return lidar_cfg


def _terrain_mesh_prim_path() -> str:
    if args_cli.terrain == "flat":
        return "/World/ground"
    return "/World/roughTerrain"


def _spawn_terrain():
    if args_cli.terrain == "flat":
        ground_cfg = sim_utils.GroundPlaneCfg()
        ground_cfg.func(_terrain_mesh_prim_path(), ground_cfg)
        return None

    terrain_cfg = TerrainImporterCfg(
        prim_path=_terrain_mesh_prim_path(),
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=5,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=1.0,
        ),
        debug_vis=False,
    )
    return TerrainImporter(terrain_cfg)


def _make_lidar_point_visualizer() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/RobotLidarPointCloud",
        markers={
            "hit": sim_utils.SphereCfg(
                radius=args_cli.point_radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
        },
    )
    visualizer = VisualizationMarkers(marker_cfg)
    visualizer.set_visibility(not bool(args_cli.headless))
    return visualizer


def _make_dynamic_obstacle_visualizer() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/RobotLidarDynamicObstacles",
        markers={
            "obstacle": sim_utils.SphereCfg(
                radius=args_cli.dynamic_obstacle_radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.35, 1.0)),
            ),
        },
    )
    visualizer = VisualizationMarkers(marker_cfg)
    visualizer.set_visibility(not bool(args_cli.headless))
    return visualizer


def _create_dynamic_mesh_obstacles() -> tuple[tuple[str, ...], tuple[object, ...]]:
    if not args_cli.dynamic_mesh_obstacles:
        return (), ()

    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    mesh_paths: list[str] = []
    translate_ops: list[object] = []
    half_extent = float(args_cli.dynamic_obstacle_radius)
    half_height = 0.5 * float(args_cli.dynamic_mesh_obstacle_height)
    points = [
        (-half_extent, -half_extent, -half_height),
        (half_extent, -half_extent, -half_height),
        (half_extent, half_extent, -half_height),
        (-half_extent, half_extent, -half_height),
        (-half_extent, -half_extent, half_height),
        (half_extent, -half_extent, half_height),
        (half_extent, half_extent, half_height),
        (-half_extent, half_extent, half_height),
    ]
    face_counts = [4, 4, 4, 4, 4, 4]
    face_indices = [
        0,
        1,
        2,
        3,
        4,
        7,
        6,
        5,
        0,
        4,
        5,
        1,
        1,
        5,
        6,
        2,
        2,
        6,
        7,
        3,
        3,
        7,
        4,
        0,
    ]

    for index in range(max(0, int(args_cli.dynamic_obstacle_count))):
        mesh_path = f"/World/DynamicMeshObstacles/obstacle_{index}"
        mesh = UsdGeom.Mesh.Define(stage, mesh_path)
        mesh.GetPointsAttr().Set(points)
        mesh.GetFaceVertexCountsAttr().Set(face_counts)
        mesh.GetFaceVertexIndicesAttr().Set(face_indices)
        xformable = UsdGeom.Xformable(mesh.GetPrim())
        translate_op = xformable.AddTranslateOp()
        translate_op.Set(Gf.Vec3d(*_dynamic_obstacle_position(index, 0.0)))
        mesh_paths.append(mesh_path)
        translate_ops.append(translate_op)

    return tuple(mesh_paths), tuple(translate_ops)


def _dynamic_obstacles(sim_time: float) -> tuple[tuple[float, float, float, float], ...]:
    if not args_cli.dynamic_obstacles:
        return ()

    obstacles: list[tuple[float, float, float, float]] = []
    count = max(0, int(args_cli.dynamic_obstacle_count))
    radius = float(args_cli.dynamic_obstacle_radius)
    for index in range(count):
        x_pos, y_pos, z_pos = _dynamic_obstacle_position(index, sim_time)
        obstacles.append((x_pos, y_pos, z_pos, radius))
    return tuple(obstacles)


def _dynamic_obstacle_position(index: int, sim_time: float) -> tuple[float, float, float]:
    phase = 1.2 * index
    x_pos = 1.75 + 0.75 * index
    y_center = -0.55 if index % 2 == 0 else 0.55
    y_pos = y_center + 0.35 * torch.sin(torch.tensor(0.55 * sim_time + phase)).item()
    if args_cli.dynamic_mesh_obstacles and not args_cli.dynamic_obstacles:
        z_pos = 0.5 * float(args_cli.dynamic_mesh_obstacle_height)
    else:
        z_pos = float(args_cli.dynamic_obstacle_radius) + 0.25
    return x_pos, y_pos, z_pos


def _render_dynamic_obstacles(
    obstacles: tuple[tuple[float, float, float, float], ...],
    visualizer: VisualizationMarkers | None,
) -> None:
    if visualizer is None:
        return
    if not obstacles:
        visualizer.visualize(translations=torch.empty((0, 3), device=args_cli.device))
        return
    translations = torch.tensor([obstacle[:3] for obstacle in obstacles], device=args_cli.device, dtype=torch.float32)
    visualizer.visualize(translations=translations)


def _update_dynamic_mesh_obstacles(translate_ops: tuple[object, ...], sim_time: float) -> None:
    if not translate_ops:
        return
    from pxr import Gf

    for index, translate_op in enumerate(translate_ops):
        translate_op.Set(Gf.Vec3d(*_dynamic_obstacle_position(index, sim_time)))


def _render_lidar_points(ray_caster: RayCaster, visualizer: VisualizationMarkers | None) -> int:
    if visualizer is None:
        return 0
    data = getattr(ray_caster, "data", None)
    ray_hits_w = getattr(data, "ray_hits_w", None)
    if ray_hits_w is None:
        return 0

    points = ray_hits_w.reshape(-1, 3).detach()
    finite_mask = torch.isfinite(points).all(dim=1)
    if not torch.any(finite_mask):
        visualizer.visualize(translations=torch.empty((0, 3), device=points.device, dtype=points.dtype))
        return 0

    points = points[finite_mask]
    if args_cli.max_points > 0 and points.shape[0] > args_cli.max_points:
        stride = max(1, points.shape[0] // args_cli.max_points)
        points = points[::stride][: args_cli.max_points]
    visualizer.visualize(translations=points)
    return int(points.shape[0])


def _print_lidar_status(ray_caster: RayCaster, step_count: int) -> None:
    data = getattr(ray_caster, "data", None)
    ray_hits_w = getattr(data, "ray_hits_w", None)
    if ray_hits_w is None:
        print(f"[INFO] step={step_count}: lidar data available, ray_hits_w not exposed by this Isaac Lab version.")
        return
    finite_hits = torch.isfinite(ray_hits_w).all(dim=-1)
    hit_count = int(finite_hits.sum().item())
    total_count = int(finite_hits.numel())
    distances = getattr(data, "distances", None)
    distance_msg = ""
    if distances is not None:
        finite_distances = distances[torch.isfinite(distances)]
        if finite_distances.numel() > 0:
            distance_msg = (
                f" distance_range=({float(finite_distances.min()):.3f}, {float(finite_distances.max()):.3f})"
            )
    print(
        f"[INFO] step={step_count}: ray_hits_w shape={tuple(ray_hits_w.shape)} "
        f"finite_hits={hit_count}/{total_count}{distance_msg}"
    )


def main() -> None:
    if args_cli.duration_seconds < 0.0:
        raise ValueError("--duration-seconds must be >= 0.")
    if args_cli.print_every < 0:
        raise ValueError("--print-every must be >= 0.")
    if args_cli.point_radius <= 0.0:
        raise ValueError("--point-radius must be > 0.")
    if args_cli.max_points < 0:
        raise ValueError("--max-points must be >= 0.")
    if args_cli.dynamic_obstacle_count < 0:
        raise ValueError("--dynamic-obstacle-count must be >= 0.")
    if args_cli.dynamic_obstacle_radius <= 0.0:
        raise ValueError("--dynamic-obstacle-radius must be > 0.")

    robot_cfg = ROBOT_CFGS[args_cli.robot]
    mount_name, mount = _select_mount(robot_cfg, args_cli.mount)

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, use_fabric=not args_cli.disable_fabric)
    sim = sim_utils.SimulationContext(sim_cfg)
    camera_eye, camera_target = CAMERA_VIEWS[args_cli.robot]
    sim.set_camera_view(camera_eye, camera_target)

    terrain_importer = _spawn_terrain()
    light_cfg = sim_utils.DomeLightCfg(intensity=2500.0, color=(0.75, 0.75, 0.75))
    light_cfg.func("/World/Light", light_cfg)
    dynamic_mesh_prim_paths, dynamic_mesh_translate_ops = _create_dynamic_mesh_obstacles()
    _update_dynamic_mesh_obstacles(dynamic_mesh_translate_ops, sim_time=0.0)

    scene = InteractiveScene(_make_scene_cfg(robot_cfg, mount))
    lidar_cfg = _make_lidar_cfg(mount, dynamic_mesh_prim_paths=dynamic_mesh_prim_paths)
    ray_caster = lidar_cfg.class_type(lidar_cfg)
    point_visualizer = None if args_cli.headless else _make_lidar_point_visualizer()
    obstacle_visualizer = None
    if args_cli.dynamic_obstacles and not args_cli.headless:
        obstacle_visualizer = _make_dynamic_obstacle_visualizer()
    sim.reset()
    scene.reset()
    ray_caster.reset()

    print("[INFO] Robot lidar 3D visualization")
    print(
        f"[INFO] robot={args_cli.robot} mount={mount_name} terrain={args_cli.terrain} "
        f"mesh={_terrain_mesh_prim_path()} parent={mount.parent_link} pos={mount.pos} rot={mount.rot}"
    )
    print(
        f"[INFO] lidar_sensor={ray_caster.__class__.__name__} "
        f"pattern={lidar_cfg.pattern_cfg.__class__.__name__} "
        f"samples={getattr(lidar_cfg.pattern_cfg, 'samples', 'grid')}"
    )
    print("[INFO] Lidar hits are drawn as red point markers from ray_hits_w, following OmniPerception's viewer style.")
    if args_cli.dynamic_obstacles:
        print(
            f"[INFO] dynamic_obstacles={args_cli.dynamic_obstacle_count} "
            f"radius={args_cli.dynamic_obstacle_radius:g}m included in lidar ray hits"
        )
    if args_cli.dynamic_mesh_obstacles:
        print(
            f"[INFO] dynamic_mesh_obstacles={len(dynamic_mesh_prim_paths)} "
            f"half_width={args_cli.dynamic_obstacle_radius:g}m "
            f"height={args_cli.dynamic_mesh_obstacle_height:g}m included in Warp mesh ray hits"
        )
    print("[INFO] Close the Isaac Sim window to exit, or use --duration-seconds for a fixed run.")
    if args_cli.headless:
        print("[WARN] Running headless: point markers are hidden. Run without --headless for 3D visualization.")

    step_count = 0
    wall_t0 = time.time()
    drawn_points = 0
    while simulation_app.is_running():
        scene.write_data_to_sim()
        sim.step(render=not bool(args_cli.headless))
        scene.update(dt=sim.get_physics_dt())
        sim_time = step_count * sim.get_physics_dt()
        _update_dynamic_mesh_obstacles(dynamic_mesh_translate_ops, sim_time)
        obstacles = _dynamic_obstacles(sim_time)
        ray_caster.cfg.dynamic_sphere_obstacles = obstacles
        _render_dynamic_obstacles(obstacles, obstacle_visualizer)
        ray_caster.update(dt=sim.get_physics_dt(), force_recompute=True)
        drawn_points = _render_lidar_points(ray_caster, point_visualizer)

        step_count += 1
        if step_count == 1 or (args_cli.print_every and step_count % args_cli.print_every == 0):
            _print_lidar_status(ray_caster, step_count)
            print(f"[INFO] step={step_count}: red point markers drawn={drawn_points}")
        if args_cli.duration_seconds > 0.0 and step_count * sim.get_physics_dt() >= args_cli.duration_seconds:
            break
        if args_cli.real_time:
            sleep_time = step_count * sim.get_physics_dt() - (time.time() - wall_t0)
            if sleep_time > 0.0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
