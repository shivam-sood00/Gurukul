"""Visual demo: fixed realistic LiDAR ray-casting against static primitives."""

from __future__ import annotations

import argparse
import math
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Demo RealisticLidar in a static scene.")
parser.add_argument("--duration-seconds", type=float, default=0.0, help="0 runs until the app closes.")
parser.add_argument("--print-every", type=int, default=30, help="Print stats every N sim steps.")
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--pattern", choices=("mid360", "grid"), default="mid360")
parser.add_argument("--rays", action="store_true", default=False, help="Visualize sampled ray lines.")
parser.add_argument("--frame", action="store_true", default=False, help="Visualize the LiDAR frame.")
parser.add_argument("--pointcloud", action="store_true", default=False, help="Visualize get_pointcloud() output.")
parser.add_argument("--pointcloud-frame", choices=("sensor", "world"), default="world")
parser.add_argument("--max-pointcloud-points", type=int, default=5000)
parser.add_argument("--max-debug-rays", type=int, default=128, help="Maximum rays sampled for --rays visualization.")
parser.add_argument("--max-debug-hits", type=int, default=5000, help="Maximum hit points visualized per debug update.")
parser.add_argument("--debug-every", type=int, default=8, help="Refresh debug markers every N frames.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils

from Gurukul.assets.realistic_lidar import make_realistic_lidar_cfg
from realistic_lidar_demo_utils import (
    add_light,
    define_xform,
    make_pointcloud_visualizer,
    print_lidar_stats,
    spawn_static_obstacle_scene,
    visualize_lidar_pointcloud,
)


def _should_continue(step: int, sim: sim_utils.SimulationContext) -> bool:
    if args_cli.duration_seconds > 0.0:
        max_steps = max(1, math.ceil(args_cli.duration_seconds / sim.get_physics_dt()))
        return step < max_steps
    return simulation_app.is_running()


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, use_fabric=False))
    sim.set_camera_view([5.0, -5.0, 3.0], [2.0, 0.0, 0.5])
    mesh_paths = spawn_static_obstacle_scene()
    define_xform("/World/Lidar", (0.0, 0.0, 0.8))
    add_light()

    lidar_cfg = make_realistic_lidar_cfg(
        prim_path="/World/Lidar",
        mesh_prim_paths=mesh_paths,
        pattern=args_cli.pattern,
        max_distance=20.0,
        debug_vis=not bool(args_cli.headless),
        max_debug_rays=args_cli.max_debug_rays,
        max_debug_hits=args_cli.max_debug_hits,
        debug_vis_interval=args_cli.debug_every,
    )
    lidar_cfg.debug_vis_hits = True
    lidar_cfg.debug_vis_rays = bool(args_cli.rays)
    lidar_cfg.debug_vis_frame = bool(args_cli.frame)
    lidar = lidar_cfg.class_type(lidar_cfg)

    sim.reset()
    lidar.reset()
    pointcloud_visualizer = None
    if bool(args_cli.pointcloud) and not bool(args_cli.headless):
        pointcloud_visualizer = make_pointcloud_visualizer()
    wall_t0 = time.time()
    step = 0
    while _should_continue(step, sim):
        sim.step(render=not bool(args_cli.headless))
        lidar.update(dt=sim.get_physics_dt(), force_recompute=True)
        visualize_lidar_pointcloud(
            lidar,
            pointcloud_visualizer,
            frame=args_cli.pointcloud_frame,
            max_points=args_cli.max_pointcloud_points,
        )
        if step == 0 or (args_cli.print_every > 0 and step % args_cli.print_every == 0):
            print_lidar_stats("static", lidar, step)
        step += 1
        if args_cli.real_time:
            sleep_time = step * sim.get_physics_dt() - (time.time() - wall_t0)
            if sleep_time > 0.0:
                time.sleep(sleep_time)


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
