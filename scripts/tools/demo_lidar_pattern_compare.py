"""Visual demo: compare grid and Livox Mid-360 ray direction distributions."""

from __future__ import annotations

import argparse
import math
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Visualize grid vs Livox Mid-360 ray directions.")
parser.add_argument("--duration-seconds", type=float, default=0.0, help="0 runs until the app closes.")
parser.add_argument("--mid360-samples", type=int, default=6000)
parser.add_argument("--grid-channels", type=int, default=32)
parser.add_argument("--grid-horizontal-res", type=float, default=1.0)
parser.add_argument("--real-time", action="store_true", default=False)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg

from Gurukul.assets.realistic_lidar import make_lidar_grid_pattern_cfg, make_livox_mid360_pattern_cfg
from realistic_lidar_demo_utils import add_light


def _should_continue(step: int, sim: sim_utils.SimulationContext) -> bool:
    if args_cli.duration_seconds > 0.0:
        max_steps = max(1, math.ceil(args_cli.duration_seconds / sim.get_physics_dt()))
        return step < max_steps
    return simulation_app.is_running()


def _marker_cfg(path: str, color: tuple[float, float, float]) -> VisualizationMarkersCfg:
    return VisualizationMarkersCfg(
        prim_path=path,
        markers={
            "point": sim_utils.SphereCfg(
                radius=0.018,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=color),
            )
        },
    )


def _print_distribution(label: str, directions: torch.Tensor) -> None:
    xy = directions[:, :2]
    yaw = torch.rad2deg(torch.atan2(xy[:, 1], xy[:, 0]))
    pitch = torch.rad2deg(torch.atan2(directions[:, 2], torch.linalg.norm(xy, dim=-1).clamp_min(1.0e-8)))
    print(
        f"[INFO] {label}: rays={directions.shape[0]} yaw=[{float(yaw.min()):.2f}, {float(yaw.max()):.2f}] deg "
        f"pitch=[{float(pitch.min()):.2f}, {float(pitch.max()):.2f}] deg"
    )


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
    sim.set_camera_view([4.0, -5.0, 3.0], [0.0, 0.0, 1.0])
    add_light()

    device = str(args_cli.device)
    mid_cfg = make_livox_mid360_pattern_cfg(samples=args_cli.mid360_samples, rolling=False)
    _, mid_dirs = mid_cfg.func(mid_cfg, device)
    grid_cfg = make_lidar_grid_pattern_cfg(channels=args_cli.grid_channels, horizontal_res=args_cli.grid_horizontal_res)
    _, grid_dirs = grid_cfg.func(grid_cfg, device)

    mid_origin = torch.tensor([-1.5, 0.0, 1.0], device=device)
    grid_origin = torch.tensor([1.5, 0.0, 1.0], device=device)
    radius = 1.25
    mid_points = mid_origin.unsqueeze(0) + radius * mid_dirs
    grid_points = grid_origin.unsqueeze(0) + radius * grid_dirs

    mid_markers = VisualizationMarkers(_marker_cfg("/Visuals/RealisticLidar/PatternCompare/Mid360", (0.1, 0.45, 1.0)))
    grid_markers = VisualizationMarkers(_marker_cfg("/Visuals/RealisticLidar/PatternCompare/Grid", (1.0, 0.35, 0.05)))
    _print_distribution("mid360", mid_dirs)
    _print_distribution("grid", grid_dirs)

    sim.reset()
    wall_t0 = time.time()
    step = 0
    while _should_continue(step, sim):
        sim.step(render=not bool(args_cli.headless))
        mid_markers.visualize(translations=mid_points)
        grid_markers.visualize(translations=grid_points)
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
