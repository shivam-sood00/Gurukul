"""Visual demo: compare clean, noisy, dropout, and noisy-dropout LiDAR outputs."""

from __future__ import annotations

import argparse
import math
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Compare RealisticLidar noise/dropout modes.")
parser.add_argument("--duration-seconds", type=float, default=0.0, help="0 runs until the app closes.")
parser.add_argument("--print-every", type=int, default=30, help="Print stats every N sim steps.")
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--dropout", type=float, default=0.25, help="Ray dropout probability for dropout demos.")
parser.add_argument("--noise-std", type=float, default=0.05, help="Gaussian distance noise standard deviation.")
parser.add_argument("--max-debug-hits", type=int, default=4000, help="Maximum hit points visualized per debug update.")
parser.add_argument("--debug-every", type=int, default=8, help="Refresh debug markers every N frames.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils

from Gurukul.assets.realistic_lidar import make_realistic_lidar_cfg
from realistic_lidar_demo_utils import add_light, define_xform, print_lidar_stats, spawn_static_obstacle_scene


def _should_continue(step: int, sim: sim_utils.SimulationContext) -> bool:
    if args_cli.duration_seconds > 0.0:
        max_steps = max(1, math.ceil(args_cli.duration_seconds / sim.get_physics_dt()))
        return step < max_steps
    return simulation_app.is_running()


def _make_lidar(name: str, mesh_paths: list[str], *, noise: bool, dropout: bool, debug_vis: bool):
    prim_path = f"/World/Lidar_{name}"
    define_xform(prim_path, (0.0, 0.0, 0.8))
    cfg = make_realistic_lidar_cfg(
        prim_path=prim_path,
        mesh_prim_paths=mesh_paths,
        pattern="mid360",
        max_distance=20.0,
        debug_vis=debug_vis,
        enable_sensor_noise=noise,
        random_distance_noise=float(args_cli.noise_std),
        ray_dropout_prob=float(args_cli.dropout) if dropout else 0.0,
        max_debug_hits=args_cli.max_debug_hits,
        debug_vis_interval=args_cli.debug_every,
    )
    cfg.hit_visualizer_cfg = cfg.hit_visualizer_cfg.replace(prim_path=f"/Visuals/RealisticLidar/{name}/Hits")
    cfg.debug_vis_hits = debug_vis
    cfg.debug_vis_rays = False
    cfg.debug_vis_frame = name == "clean"
    return cfg.class_type(cfg)


def _print_dropout_stats(label: str, lidar, clean_valid: torch.Tensor | None, step: int) -> None:
    print_lidar_stats(label, lidar, step)
    valid = lidar.get_valid_mask()
    valid_percent = 100.0 * float(valid.sum()) / max(1, valid.numel())
    print(f"[INFO] {label} step={step}: valid_percent={valid_percent:.2f}%")
    if clean_valid is not None:
        dropped_from_clean = clean_valid & ~valid
        print(
            f"[INFO] {label} step={step}: dropped_from_clean="
            f"{100.0 * float(dropped_from_clean.sum()) / max(1, int(clean_valid.sum())):.2f}%"
        )


def main() -> None:
    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device, use_fabric=False))
    sim.set_camera_view([5.0, -5.0, 3.0], [2.0, 0.0, 0.5])
    mesh_paths = spawn_static_obstacle_scene()
    add_light()

    debug_vis = not bool(args_cli.headless)
    lidars = {
        "clean": _make_lidar("clean", mesh_paths, noise=False, dropout=False, debug_vis=debug_vis),
        "noise": _make_lidar("noise", mesh_paths, noise=True, dropout=False, debug_vis=False),
        "dropout": _make_lidar("dropout", mesh_paths, noise=False, dropout=True, debug_vis=debug_vis),
        "noise_dropout": _make_lidar("noise_dropout", mesh_paths, noise=True, dropout=True, debug_vis=False),
    }

    sim.reset()
    for lidar in lidars.values():
        lidar.reset()
    wall_t0 = time.time()
    step = 0
    while _should_continue(step, sim):
        sim.step(render=not bool(args_cli.headless))
        for lidar in lidars.values():
            lidar.update(dt=sim.get_physics_dt(), force_recompute=True)

        if step == 0 or (args_cli.print_every > 0 and step % args_cli.print_every == 0):
            clean_valid = lidars["clean"].get_valid_mask().clone()
            for label, lidar in lidars.items():
                _print_dropout_stats(label, lidar, clean_valid if label != "clean" else None, step)

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
