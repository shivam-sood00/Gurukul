"""Visual demo: mount RealisticLidar on a robot link."""

from __future__ import annotations

import argparse
import math
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Mount RealisticLidar on Go2 if available.")
parser.add_argument("--robot", default="go2", choices=("go2", "go2_airbot", "go2_d1", "g1"))
parser.add_argument("--mount", default=None, help="Mount name. Defaults to first mount declared on the robot.")
parser.add_argument("--duration-seconds", type=float, default=0.0, help="0 runs until the app closes.")
parser.add_argument("--print-every", type=int, default=60)
parser.add_argument("--real-time", action="store_true", default=False)
parser.add_argument("--terrain", choices=("flat", "rough"), default="flat")
parser.add_argument("--scripted-motion", action="store_true", default=False, help="Move the robot root for LiDAR transform checks.")
parser.add_argument("--motion-speed", type=float, default=0.25, help="Scripted root speed in m/s.")
parser.add_argument("--base-height", type=float, default=0.55, help="Scripted root height above world origin/terrain patch.")
parser.add_argument("--rays", action="store_true", default=False, help="Visualize sampled ray lines.")
parser.add_argument("--pointcloud", action="store_true", default=False, help="Visualize get_pointcloud() output.")
parser.add_argument("--pointcloud-frame", choices=("sensor", "world"), default="world")
parser.add_argument("--max-pointcloud-points", type=int, default=5000)
parser.add_argument("--free-root", action="store_true", default=False, help="Let robot root move under physics.")
parser.add_argument("--max-debug-rays", type=int, default=128, help="Maximum rays sampled for --rays visualization.")
parser.add_argument("--max-debug-hits", type=int, default=5000, help="Maximum hit points visualized per debug update.")
parser.add_argument("--debug-every", type=int, default=8, help="Refresh debug markers every N frames.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.terrains import TerrainImporter, TerrainImporterCfg
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG
from isaaclab.utils.math import quat_from_euler_xyz

from Gurukul.assets.realistic_lidar import make_realistic_lidar_cfg
from Gurukul.assets.unitree import (
    UNITREE_G1_29DOF_CFG,
    UNITREE_GO2_AIRBOT_ARM_CFG,
    UNITREE_GO2_CFG,
    UNITREE_GO2_D1_ARM_CFG,
)
from realistic_lidar_demo_utils import add_light, make_pointcloud_visualizer, print_lidar_stats, visualize_lidar_pointcloud


def _should_continue(step: int, sim: sim_utils.SimulationContext) -> bool:
    if args_cli.duration_seconds > 0.0:
        max_steps = max(1, math.ceil(args_cli.duration_seconds / sim.get_physics_dt()))
        return step < max_steps
    return simulation_app.is_running()


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
        raise RuntimeError(f"{args_cli.robot}: unknown mount {mount_name!r}. Available mounts: {', '.join(mounts)}")
    return mount_name, mounts[mount_name]


def _make_scene_cfg(robot_cfg) -> InteractiveSceneCfg:
    scene_cfg = InteractiveSceneCfg(num_envs=1, env_spacing=2.0)
    robot_spawn_cfg = robot_cfg.replace(prim_path="/World/Robot")
    if not args_cli.free_root and getattr(robot_spawn_cfg.spawn, "articulation_props", None) is not None:
        robot_spawn_cfg.spawn.articulation_props.fix_root_link = True
    scene_cfg.robot = robot_spawn_cfg
    return scene_cfg


def _terrain_mesh_prim_path() -> str:
    return "/World/ground" if args_cli.terrain == "flat" else "/World/roughTerrain"


def _spawn_terrain():
    if args_cli.terrain == "flat":
        ground = sim_utils.GroundPlaneCfg()
        ground.func(_terrain_mesh_prim_path(), ground)
        return None

    terrain_cfg = TerrainImporterCfg(
        prim_path=_terrain_mesh_prim_path(),
        terrain_type="generator",
        terrain_generator=ROUGH_TERRAINS_CFG,
        max_init_terrain_level=3,
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )
    return TerrainImporter(terrain_cfg)


def _apply_scripted_root_motion(robot, sim_time: float) -> None:
    if not bool(args_cli.scripted_motion):
        return
    root_state = robot.data.default_root_state.clone()
    root_state[:, 0] = -0.6 + float(args_cli.motion_speed) * sim_time
    root_state[:, 1] = 0.35 * torch.sin(torch.tensor(0.6 * sim_time, device=root_state.device))
    root_state[:, 2] = float(args_cli.base_height)
    yaw = 0.35 * torch.sin(torch.tensor(0.4 * sim_time, device=root_state.device)).reshape(1)
    zero = torch.zeros_like(yaw)
    root_state[:, 3:7] = quat_from_euler_xyz(zero, zero, yaw)
    root_state[:, 7:] = 0.0
    robot.write_root_state_to_sim(root_state)


def main() -> None:
    robot_cfg = ROBOT_CFGS[args_cli.robot]
    mount_name, mount = _select_mount(robot_cfg, args_cli.mount)

    sim = sim_utils.SimulationContext(sim_utils.SimulationCfg(device=args_cli.device))
    sim.set_camera_view(*CAMERA_VIEWS[args_cli.robot])

    _spawn_terrain()
    add_light()
    scene = InteractiveScene(_make_scene_cfg(robot_cfg))
    robot = scene["robot"]

    parent_link_path = f"/World/Robot/{mount.parent_link}"
    lidar_cfg = make_realistic_lidar_cfg(
        prim_path=parent_link_path,
        mesh_prim_paths=[_terrain_mesh_prim_path()],
        pattern="mid360",
        offset_pos=tuple(mount.pos),
        offset_rot=tuple(mount.rot),
        max_distance=float(mount.sensor.max_range),
        min_range=0.1,
        update_period=float(mount.sensor.update_period),
        enable_sensor_noise=bool(mount.sensor.enable_sensor_noise),
        debug_vis=not bool(args_cli.headless),
        max_debug_rays=args_cli.max_debug_rays,
        max_debug_hits=args_cli.max_debug_hits,
        debug_vis_interval=args_cli.debug_every,
    )
    lidar_cfg.debug_vis_hits = True
    lidar_cfg.debug_vis_rays = bool(args_cli.rays)
    lidar_cfg.debug_vis_frame = True
    lidar = lidar_cfg.class_type(lidar_cfg)
    print(f"[INFO] Mounted RealisticLidar on {args_cli.robot}:{mount_name} at {parent_link_path}")

    sim.reset()
    scene.reset()
    lidar.reset()
    pointcloud_visualizer = None
    if bool(args_cli.pointcloud) and not bool(args_cli.headless):
        pointcloud_visualizer = make_pointcloud_visualizer("/Visuals/RealisticLidar/RobotMountPointCloud")
    wall_t0 = time.time()
    step = 0
    while _should_continue(step, sim):
        sim_time = step * sim.get_physics_dt()
        _apply_scripted_root_motion(robot, sim_time)
        scene.write_data_to_sim()
        sim.step(render=not bool(args_cli.headless))
        scene.update(sim.get_physics_dt())
        lidar.update(dt=sim.get_physics_dt(), force_recompute=True)
        visualize_lidar_pointcloud(
            lidar,
            pointcloud_visualizer,
            frame=args_cli.pointcloud_frame,
            max_points=args_cli.max_pointcloud_points,
        )
        if step == 0 or (args_cli.print_every > 0 and step % args_cli.print_every == 0):
            print_lidar_stats("robot_mount", lidar, step)

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
