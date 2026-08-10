"""Smoke-test robot lidar metadata and OmniPerception-pattern RayCaster configs."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from typing import Any

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description=(
        "Validate robot lidar mount metadata and build native Isaac Lab RayCaster configs using repo-local "
        "OmniPerception scan patterns for each mount against both flat and uneven terrain targets. This does not "
        "spawn a simulation scene."
    )
)
parser.add_argument(
    "--robots",
    nargs="+",
    default=("go2", "go2_airbot", "go2_d1", "g1"),
    choices=("go2", "go2_airbot", "go2_d1", "g1"),
    help="Robot lidar configs to validate.",
)
parser.add_argument(
    "--json",
    action="store_true",
    help="Print machine-readable JSON instead of the text report.",
)
parser.add_argument(
    "--terrains",
    nargs="+",
    default=("flat", "uneven"),
    choices=("flat", "uneven"),
    help="Terrain raycast targets to validate for each robot mount.",
)
parser.add_argument(
    "--debug-vis",
    action="store_true",
    help="Enable debug_vis in the generated RayCasterCfg objects.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

app_launcher = AppLauncher(headless=True)
simulation_app = app_launcher.app

from Gurukul.assets.lidar import make_omni_perception_lidar_ray_caster_cfg  # noqa: E402
from Gurukul.assets.unitree import (  # noqa: E402
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

TERRAIN_MESH_PRIMS = {
    "flat": "/World/ground",
    "uneven": "/World/roughTerrain",
}


@dataclass(frozen=True)
class LidarMountReport:
    robot: str
    mount_name: str
    terrain: str
    terrain_mesh_prim_path: str
    parent_link: str
    prim_path: str
    pos: tuple[float, float, float]
    rot: tuple[float, float, float, float]
    sensor_type: str
    omni_perception_type: str | None
    max_range: float
    update_period: float
    horizontal_fov: float
    vertical_fov: float
    channels: int
    horizontal_res: float
    raycaster_prim_path: str
    raycaster_max_distance: float
    sensor_cfg_type: str
    sensor_class_type: str
    pattern_cfg_type: str
    pattern_samples: int | None


def _fail(message: str) -> None:
    raise RuntimeError(message)


def _validate_mount(
    robot_name: str,
    mount_name: str,
    mount: Any,
    terrain_name: str,
    debug_vis: bool,
) -> LidarMountReport:
    sensor = getattr(mount, "sensor", None)
    if sensor is None:
        _fail(f"{robot_name}.{mount_name}: mount has no sensor config.")
    if not getattr(mount, "parent_link", None):
        _fail(f"{robot_name}.{mount_name}: parent_link is empty.")
    if not getattr(mount, "prim_suffix", None):
        _fail(f"{robot_name}.{mount_name}: prim_suffix is empty.")

    pos = tuple(float(v) for v in getattr(mount, "pos", ()))
    rot = tuple(float(v) for v in getattr(mount, "rot", ()))
    if len(pos) != 3:
        _fail(f"{robot_name}.{mount_name}: expected 3D position, got {pos!r}.")
    if len(rot) != 4:
        _fail(f"{robot_name}.{mount_name}: expected quaternion rotation, got {rot!r}.")

    max_range = float(getattr(sensor, "max_range", 0.0))
    update_period = float(getattr(sensor, "update_period", 0.0))
    horizontal_fov = float(getattr(sensor, "horizontal_fov", 0.0))
    vertical_fov = float(getattr(sensor, "vertical_fov", 0.0))
    channels = int(getattr(sensor, "channels", 0))
    horizontal_res = float(getattr(sensor, "horizontal_res", 0.0))
    if max_range <= 0.0:
        _fail(f"{robot_name}.{mount_name}: max_range must be positive.")
    if update_period <= 0.0:
        _fail(f"{robot_name}.{mount_name}: update_period must be positive.")
    if not (0.0 < horizontal_fov <= 360.0):
        _fail(f"{robot_name}.{mount_name}: horizontal_fov must be in (0, 360].")
    if vertical_fov <= 0.0:
        _fail(f"{robot_name}.{mount_name}: vertical_fov must be positive.")
    if channels <= 0:
        _fail(f"{robot_name}.{mount_name}: channels must be positive.")
    if horizontal_res <= 0.0:
        _fail(f"{robot_name}.{mount_name}: horizontal_res must be positive.")

    terrain_mesh_prim_path = TERRAIN_MESH_PRIMS[terrain_name]
    raycaster_cfg = make_omni_perception_lidar_ray_caster_cfg(
        mount,
        mesh_prim_paths=[terrain_mesh_prim_path],
        debug_vis=debug_vis,
    )
    raycaster_prim_path = getattr(raycaster_cfg, "prim_path", "")
    raycaster_max_distance = float(getattr(raycaster_cfg, "max_distance", 0.0))
    if raycaster_prim_path != mount.parent_prim_path:
        _fail(
            f"{robot_name}.{mount_name}: RayCaster prim_path {raycaster_prim_path!r} does not match "
            f"mount parent path {mount.parent_prim_path!r}."
        )
    if raycaster_max_distance != max_range:
        _fail(
            f"{robot_name}.{mount_name}: RayCaster max_distance {raycaster_max_distance} does not match "
            f"sensor max_range {max_range}."
        )
    raycaster_mesh_prim_paths = list(getattr(raycaster_cfg, "mesh_prim_paths", ()))
    if raycaster_mesh_prim_paths != [terrain_mesh_prim_path]:
        _fail(
            f"{robot_name}.{mount_name}.{terrain_name}: RayCaster mesh_prim_paths {raycaster_mesh_prim_paths!r} "
            f"does not match expected terrain mesh {terrain_mesh_prim_path!r}."
        )
    pattern_cfg = getattr(raycaster_cfg, "pattern_cfg", None)
    class_type = getattr(raycaster_cfg, "class_type", None)

    return LidarMountReport(
        robot=robot_name,
        mount_name=mount_name,
        terrain=terrain_name,
        terrain_mesh_prim_path=terrain_mesh_prim_path,
        parent_link=mount.parent_link,
        prim_path=mount.prim_path,
        pos=pos,
        rot=rot,
        sensor_type=sensor.sensor_type,
        omni_perception_type=sensor.omni_perception_type,
        max_range=max_range,
        update_period=update_period,
        horizontal_fov=horizontal_fov,
        vertical_fov=vertical_fov,
        channels=channels,
        horizontal_res=horizontal_res,
        raycaster_prim_path=raycaster_prim_path,
        raycaster_max_distance=raycaster_max_distance,
        sensor_cfg_type=raycaster_cfg.__class__.__name__,
        sensor_class_type=getattr(class_type, "__name__", str(class_type)),
        pattern_cfg_type=pattern_cfg.__class__.__name__ if pattern_cfg is not None else "",
        pattern_samples=getattr(pattern_cfg, "samples", None),
    )


def validate_robot_lidars(robot_names: list[str], terrain_names: list[str], debug_vis: bool) -> list[LidarMountReport]:
    reports: list[LidarMountReport] = []
    for robot_name in robot_names:
        robot_cfg = ROBOT_CFGS[robot_name]
        mounts = getattr(robot_cfg, "lidar_mounts", None)
        if not mounts:
            _fail(f"{robot_name}: no lidar_mounts declared.")
        for mount_name, mount in mounts.items():
            for terrain_name in terrain_names:
                reports.append(
                    _validate_mount(
                        robot_name,
                        mount_name,
                        mount,
                        terrain_name=terrain_name,
                        debug_vis=debug_vis,
                    )
                )
    return reports


def _print_text_report(reports: list[LidarMountReport]) -> None:
    print("Robot lidar mount smoke test", flush=True)
    print("=" * 31, flush=True)
    for report in reports:
        print(
            f"[OK] {report.robot}.{report.mount_name}: "
            f"terrain={report.terrain} mesh={report.terrain_mesh_prim_path} "
            f"{report.sensor_type}({report.omni_perception_type}) on {report.parent_link} "
            f"pos={report.pos} range={report.max_range:g}m "
            f"sensor={report.sensor_class_type} pattern={report.pattern_cfg_type}"
            f"({report.pattern_samples or 'grid'}) raycaster={report.raycaster_prim_path}",
            flush=True,
        )


def main() -> int:
    reports = validate_robot_lidars(
        list(args_cli.robots),
        terrain_names=list(args_cli.terrains),
        debug_vis=args_cli.debug_vis,
    )
    if args_cli.json:
        print(json.dumps([asdict(report) for report in reports], indent=2, sort_keys=True), flush=True)
    else:
        _print_text_report(reports)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
    finally:
        simulation_app.close()
