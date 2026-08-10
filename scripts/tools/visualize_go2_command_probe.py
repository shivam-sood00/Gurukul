"""Visualize a command-driven kinematic probe with one-to-one Go2 terrain scanner settings."""

from __future__ import annotations

import argparse
import copy
import math
import sys
from dataclasses import dataclass

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description=(
        "Kinematic sphere probe viewer using exact Go2 IsaacLab task terrain, height-scanner footprint, "
        "and command ranges. No Go2 articulation is spawned."
    )
)
parser.add_argument(
    "--go2-env-profile",
    type=str,
    default="start_sparse",
    choices=("rough", "start_sparse", "real_sparse", "real_beam"),
    help="Go2 env config profile to clone exactly for terrain/scanner/command settings.",
)
parser.add_argument(
    "--duration-seconds",
    type=float,
    default=45.0,
    help="Total visualization duration.",
)
parser.add_argument(
    "--command-resample-seconds",
    type=float,
    default=None,
    help="How often to resample velocity commands. Defaults to the selected Go2 command config.",
)
parser.add_argument(
    "--base-height",
    type=float,
    default=None,
    help="Probe base height above terrain. Defaults to the selected Go2 config base-height target.",
)
parser.add_argument(
    "--height-scan-offset",
    type=float,
    default=0.5,
    help="Height-scan offset used by Go2 height observations.",
)
parser.add_argument(
    "--sphere-radius",
    type=float,
    default=0.06,
    help="Radius of the probe sphere marker.",
)
parser.add_argument("--seed", type=int, default=7, help="Random seed.")
parser.add_argument(
    "--hide-heightmap-window",
    action="store_true",
    default=False,
    help="Disable the 2D OpenCV heightmap window.",
)
parser.add_argument(
    "--disable-ray-debug-vis",
    action="store_true",
    default=False,
    help="Disable 3D ray-hit visualization in the IsaacLab viewport.",
)
parser.add_argument(
    "--random-patch",
    action="store_true",
    default=False,
    help="Sample a random terrain patch instead of using a central patch.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.duration_seconds <= 0.0:
    parser.error("--duration-seconds must be > 0.")
if args_cli.command_resample_seconds is not None and args_cli.command_resample_seconds <= 0.0:
    parser.error("--command-resample-seconds must be > 0 when provided.")
if args_cli.sphere_radius <= 0.0:
    parser.error("--sphere-radius must be > 0.")
if args_cli.base_height is not None and args_cli.base_height < 0.0:
    parser.error("--base-height must be >= 0.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.api.simulation_context import SimulationContext
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.viewports import set_camera_view

import isaaclab.sim as sim_utils
from isaaclab.sensors import RayCaster
from isaaclab.terrains import TerrainImporter

import Gurukul.tasks  # noqa: F401
from Gurukul.tasks.manager_based.locomotion.velocity.config.quadruped.unitree_go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
)
from Gurukul.tasks.manager_based.locomotion.velocity.config.quadruped.unitree_go2.rough_real_beam_env_cfg import (
    UnitreeGo2RoughRealBeamEnvCfg,
)
from Gurukul.tasks.manager_based.locomotion.velocity.config.quadruped.unitree_go2.rough_real_sparse_env_cfg import (
    UnitreeGo2RoughRealSparseEnvCfg,
)
from Gurukul.tasks.manager_based.locomotion.velocity.config.quadruped.unitree_go2.rough_start_env_cfg import (
    UnitreeGo2RoughStartEnvCfg,
)


@dataclass
class CommandRuntimeState:
    sampled_vel_command_b: torch.Tensor
    heading_target: float
    is_heading_env: bool
    is_standing_env: bool


@dataclass
class ProbeSession:
    terrain_importer: TerrainImporter
    ray_caster: RayCaster
    probe_view: XFormPrim
    terrain_size: tuple[float, float]
    scan_size: tuple[float, float]
    scan_resolution: float
    scan_shape: tuple[int, int]
    center_ray_index: int


def _make_simulation_context(device: str, headless: bool) -> SimulationContext:
    sim_params = {
        "use_gpu": device.startswith("cuda"),
        "use_gpu_pipeline": device.startswith("cuda"),
        "use_flatcache": headless,
        "use_fabric": True,
        "enable_scene_query_support": True,
    }
    return SimulationContext(
        physics_dt=1.0 / 60.0,
        rendering_dt=1.0 / 60.0,
        sim_params=sim_params,
        backend="torch",
        device=device,
    )


def _build_reference_env_cfg(profile: str):
    profile_to_cfg_cls = {
        "rough": UnitreeGo2RoughEnvCfg,
        "start_sparse": UnitreeGo2RoughStartEnvCfg,
        "real_sparse": UnitreeGo2RoughRealSparseEnvCfg,
        "real_beam": UnitreeGo2RoughRealBeamEnvCfg,
    }
    return profile_to_cfg_cls[profile]()


def _resolve_default_base_height(env_cfg) -> float:
    rewards_cfg = getattr(env_cfg, "rewards", None)
    base_height_term = getattr(rewards_cfg, "base_height_l2", None)
    params = getattr(base_height_term, "params", None)
    if isinstance(params, dict):
        target = params.get("target_height", None)
        if target is not None:
            return float(target)
    return 0.33


def _as_range(value, fallback: tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, (tuple, list)) and len(value) == 2:
        return float(value[0]), float(value[1])
    return float(fallback[0]), float(fallback[1])


def _sample_uniform(low: float, high: float, device: torch.device) -> float:
    return float(torch.empty((), device=device).uniform_(float(low), float(high)).item())


def _wrap_to_pi(angle: float) -> float:
    return (float(angle) + math.pi) % (2.0 * math.pi) - math.pi


def _is_start_sparse_command(command_cfg) -> bool:
    if hasattr(command_cfg, "turn_trigger_threshold"):
        return True
    class_type = getattr(command_cfg, "class_type", None)
    return getattr(class_type, "__name__", "") == "StartSparseVelocityCommand"


def _is_uniform_threshold_command(command_cfg) -> bool:
    class_type = getattr(command_cfg, "class_type", None)
    return getattr(class_type, "__name__", "") == "UniformThresholdVelocityCommand"


def _resample_command_state(command_cfg, device: torch.device) -> CommandRuntimeState:
    ranges_cfg = getattr(command_cfg, "ranges", None)
    lin_x_range = _as_range(getattr(ranges_cfg, "lin_vel_x", (-1.0, 1.0)), (-1.0, 1.0))
    lin_y_range = _as_range(getattr(ranges_cfg, "lin_vel_y", (-1.0, 1.0)), (-1.0, 1.0))
    ang_z_range = _as_range(getattr(ranges_cfg, "ang_vel_z", (-1.0, 1.0)), (-1.0, 1.0))

    vx = _sample_uniform(*lin_x_range, device=device)
    vy = _sample_uniform(*lin_y_range, device=device)
    wz = _sample_uniform(*ang_z_range, device=device)

    heading_command = bool(getattr(command_cfg, "heading_command", False))
    rel_heading_envs = float(getattr(command_cfg, "rel_heading_envs", 1.0))
    rel_standing_envs = float(getattr(command_cfg, "rel_standing_envs", 0.0))
    heading_range = _as_range(getattr(ranges_cfg, "heading", (-math.pi, math.pi)), (-math.pi, math.pi))

    heading_target = _sample_uniform(*heading_range, device=device) if heading_command else 0.0
    is_heading_env = bool(heading_command and (_sample_uniform(0.0, 1.0, device=device) <= rel_heading_envs))
    is_standing_env = bool(_sample_uniform(0.0, 1.0, device=device) <= rel_standing_envs)

    if _is_start_sparse_command(command_cfg):
        # Mirrors StartSparseVelocityCommand._resample_command.
        vy = 0.0
        turn_threshold = float(getattr(command_cfg, "turn_trigger_threshold", 0.3))
        if vx < turn_threshold:
            vx = 0.0
            wz = _sample_uniform(*ang_z_range, device=device)
        else:
            wz = 0.0

    if _is_uniform_threshold_command(command_cfg):
        # Mirrors UniformThresholdVelocityCommand._resample_command thresholding.
        if math.hypot(vx, vy) <= 0.2:
            vx, vy = 0.0, 0.0

    sampled_command = torch.tensor([vx, vy, wz], device=device, dtype=torch.float32)
    return CommandRuntimeState(
        sampled_vel_command_b=sampled_command,
        heading_target=float(heading_target),
        is_heading_env=is_heading_env,
        is_standing_env=is_standing_env,
    )


def _resolve_command_for_step(command_cfg, state: CommandRuntimeState, probe_heading: float) -> torch.Tensor:
    command = state.sampled_vel_command_b.clone()
    ranges_cfg = getattr(command_cfg, "ranges", None)
    ang_z_range = _as_range(getattr(ranges_cfg, "ang_vel_z", (-1.0, 1.0)), (-1.0, 1.0))

    if bool(getattr(command_cfg, "heading_command", False)) and state.is_heading_env:
        heading_control_stiffness = float(getattr(command_cfg, "heading_control_stiffness", 1.0))
        heading_error = _wrap_to_pi(state.heading_target - float(probe_heading))
        command[2] = torch.tensor(
            float(np.clip(heading_control_stiffness * heading_error, ang_z_range[0], ang_z_range[1])),
            dtype=torch.float32,
            device=command.device,
        )

    if state.is_standing_env:
        command[:] = 0.0

    return command


def _yaw_to_quat(yaw: float, device: torch.device) -> torch.Tensor:
    half_yaw = 0.5 * float(yaw)
    return torch.tensor([math.cos(half_yaw), 0.0, 0.0, math.sin(half_yaw)], device=device, dtype=torch.float32)


def _integrate_body_command_xy(rel_xy: torch.Tensor, yaw: float, command_b: torch.Tensor, dt: float) -> torch.Tensor:
    cos_yaw = math.cos(float(yaw))
    sin_yaw = math.sin(float(yaw))
    vx = float(command_b[0].item())
    vy = float(command_b[1].item())
    rel_xy = rel_xy.clone()
    rel_xy[0] += (vx * cos_yaw - vy * sin_yaw) * dt
    rel_xy[1] += (vx * sin_yaw + vy * cos_yaw) * dt
    return rel_xy


def _nominal_scan_shape(scan_size: tuple[float, float], scan_resolution: float) -> tuple[int, int]:
    if scan_resolution <= 0.0:
        return 1, 1
    rows = max(1, int(round(float(scan_size[0]) / float(scan_resolution))))
    cols = max(1, int(round(float(scan_size[1]) / float(scan_resolution))))
    return int(rows), int(cols)


def _infer_scan_shape(num_rays: int, scan_size: tuple[float, float], scan_resolution: float) -> tuple[int, int]:
    nominal_rows, nominal_cols = _nominal_scan_shape(scan_size=scan_size, scan_resolution=scan_resolution)

    for rows, cols in (
        (nominal_rows, nominal_cols),
        (nominal_cols, nominal_rows),
        (nominal_rows + 1, nominal_cols),
        (nominal_rows, nominal_cols + 1),
        (nominal_rows + 1, nominal_cols + 1),
        (max(1, nominal_rows - 1), nominal_cols),
        (nominal_rows, max(1, nominal_cols - 1)),
    ):
        if rows * cols == int(num_rays):
            return int(rows), int(cols)

    best_rows, best_cols = int(num_rays), 1
    best_cost = abs(best_rows - nominal_rows) + abs(best_cols - nominal_cols)
    for divisor in range(1, int(num_rays**0.5) + 1):
        if num_rays % divisor != 0:
            continue
        for rows, cols in ((num_rays // divisor, divisor), (divisor, num_rays // divisor)):
            cost = abs(rows - nominal_rows) + abs(cols - nominal_cols)
            if cost < best_cost:
                best_rows, best_cols, best_cost = rows, cols, cost
    return int(best_rows), int(best_cols)


def _make_terrain_cfg(prefix: str, env_cfg):
    terrain_cfg = copy.deepcopy(env_cfg.scene.terrain)
    terrain_cfg.prim_path = f"{prefix}/ground"
    terrain_cfg.max_init_terrain_level = None
    terrain_cfg.debug_vis = False
    if hasattr(terrain_cfg, "num_envs"):
        terrain_cfg.num_envs = 1
    return terrain_cfg


def _make_probe_ray_caster_cfg(prefix: str, mesh_prim_path: str, env_cfg):
    ray_caster_cfg = copy.deepcopy(env_cfg.scene.height_scanner)
    if ray_caster_cfg is None:
        raise RuntimeError("Selected Go2 env config has no height_scanner.")
    ray_caster_cfg.prim_path = f"{prefix}/envs/env_.*/probe"
    ray_caster_cfg.mesh_prim_paths = [mesh_prim_path]
    ray_caster_cfg.debug_vis = not bool(args_cli.disable_ray_debug_vis)
    return ray_caster_cfg


def _build_probe_session(prefix: str, env_cfg) -> ProbeSession:
    terrain_cfg = _make_terrain_cfg(prefix, env_cfg)
    terrain_importer = TerrainImporter(terrain_cfg)

    prim_utils.create_prim(f"{prefix}/envs", "Xform", translation=(0.0, 0.0, 0.0))
    prim_utils.create_prim(f"{prefix}/envs/env_0", "Xform", translation=(0.0, 0.0, 0.0))
    prim_utils.create_prim(f"{prefix}/envs/env_0/probe", "Xform", translation=(0.0, 0.0, 0.0))

    ray_caster_cfg = _make_probe_ray_caster_cfg(prefix, terrain_cfg.prim_path, env_cfg)
    ray_caster = RayCaster(ray_caster_cfg)
    probe_view = XFormPrim(ray_caster_cfg.prim_path, reset_xform_properties=False)

    pattern_cfg = ray_caster_cfg.pattern_cfg
    scan_resolution = float(getattr(pattern_cfg, "resolution", 0.1))
    scan_size = getattr(pattern_cfg, "size", (1.6, 1.0))
    scan_size = (float(scan_size[0]), float(scan_size[1]))

    scan_shape = _nominal_scan_shape(scan_size=scan_size, scan_resolution=scan_resolution)
    center_row = int(scan_shape[0] // 2)
    center_col = int(scan_shape[1] // 2)
    center_ray_index = int(center_row * scan_shape[1] + center_col)

    terrain_generator = terrain_cfg.terrain_generator
    terrain_size = (float(terrain_generator.size[0]), float(terrain_generator.size[1]))

    return ProbeSession(
        terrain_importer=terrain_importer,
        ray_caster=ray_caster,
        probe_view=probe_view,
        terrain_size=terrain_size,
        scan_size=scan_size,
        scan_resolution=scan_resolution,
        scan_shape=scan_shape,
        center_ray_index=center_ray_index,
    )


def _compute_motion_limits(session: ProbeSession) -> tuple[float, float]:
    margin_x = 0.5 * session.scan_size[0] + session.scan_resolution
    margin_y = 0.5 * session.scan_size[1] + session.scan_resolution
    limit_x = max(0.05, 0.5 * session.terrain_size[0] - margin_x)
    limit_y = max(0.05, 0.5 * session.terrain_size[1] - margin_y)
    return float(limit_x), float(limit_y)


def _pick_patch_origin(session: ProbeSession, device: torch.device) -> torch.Tensor:
    if session.terrain_importer.terrain_origins is None:
        raise RuntimeError("Terrain importer does not expose terrain origins.")
    terrain_origins = session.terrain_importer.terrain_origins.reshape(-1, 3)
    if bool(getattr(args_cli, "random_patch", False)):
        patch_index = int(torch.randint(0, terrain_origins.shape[0], (1,), device=device).item())
    else:
        # Default to a central patch so the initial camera view stays near the action.
        center_xy = terrain_origins[:, :2].mean(dim=0, keepdim=True)
        distances = torch.linalg.norm(terrain_origins[:, :2] - center_xy, dim=1)
        patch_index = int(torch.argmin(distances).item())
    return terrain_origins[patch_index].to(device=device, dtype=torch.float32)


def _build_probe_sphere_view(radius: float) -> XFormPrim:
    sphere_prim_path = "/World/Go2CommandProbe/visual/probe_sphere"
    prim_utils.create_prim(
        sphere_prim_path,
        "Sphere",
        translation=(0.0, 0.0, 0.0),
        attributes={
            "radius": float(radius),
            "primvars:displayColor": [(0.10, 0.72, 0.98)],
        },
    )
    return XFormPrim(sphere_prim_path, reset_xform_properties=False)


def _render_heightmap_window(height_map: torch.Tensor, command_b: torch.Tensor, elapsed_seconds: float) -> None:
    if cv2 is None:
        return

    map_np = height_map.detach().cpu().numpy()
    finite_mask = np.isfinite(map_np)
    if np.any(finite_mask):
        finite_values = map_np[finite_mask]
        lo = float(np.percentile(finite_values, 5.0))
        hi = float(np.percentile(finite_values, 95.0))
        if hi - lo < 1.0e-6:
            hi = lo + 1.0e-6
        normalized = np.clip((np.nan_to_num(map_np, nan=lo) - lo) / (hi - lo), 0.0, 1.0)
        min_val = float(np.min(finite_values))
        max_val = float(np.max(finite_values))
    else:
        normalized = np.zeros_like(map_np, dtype=np.float32)
        min_val = 0.0
        max_val = 0.0

    image_u8 = np.uint8(255.0 * normalized)
    color = cv2.applyColorMap(image_u8, cv2.COLORMAP_TURBO)
    scale = max(5, int(round(320 / max(1, color.shape[1]))))
    color = cv2.resize(color, (color.shape[1] * scale, color.shape[0] * scale), interpolation=cv2.INTER_NEAREST)

    cmd_x = float(command_b[0].item())
    cmd_y = float(command_b[1].item())
    cmd_yaw = float(command_b[2].item())
    cv2.putText(
        color,
        f"t={elapsed_seconds:05.1f}s cmd=[{cmd_x:+.2f}, {cmd_y:+.2f}, {cmd_yaw:+.2f}]",
        (8, 22),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        color,
        f"height range [{min_val:+.3f}, {max_val:+.3f}]",
        (8, 44),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (235, 235, 235),
        1,
        cv2.LINE_AA,
    )
    cv2.namedWindow("Go2 Command Probe Heightmap", cv2.WINDOW_NORMAL)
    cv2.imshow("Go2 Command Probe Heightmap", color)
    cv2.waitKey(1)


def main() -> None:
    torch.manual_seed(int(args_cli.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args_cli.seed))

    sim_device = args_cli.device if args_cli.device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
    sim = _make_simulation_context(device=sim_device, headless=bool(args_cli.headless))

    light_cfg = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 0.9))
    light_cfg.func("/World/Go2ProbeLight", light_cfg)

    if not args_cli.headless:
        set_camera_view([36.0, 0.0, 24.0], [0.0, 0.0, 0.0])

    env_cfg = _build_reference_env_cfg(args_cli.go2_env_profile)
    command_cfg = copy.deepcopy(env_cfg.commands.base_velocity)
    base_height = (
        float(args_cli.base_height)
        if args_cli.base_height is not None
        else float(_resolve_default_base_height(env_cfg))
    )

    resample_seconds = (
        float(args_cli.command_resample_seconds)
        if args_cli.command_resample_seconds is not None
        else float(getattr(command_cfg, "resampling_time_range", (10.0, 10.0))[1])
    )
    session = _build_probe_session(prefix="/World/Go2CommandProbe", env_cfg=env_cfg)
    probe_sphere_view = _build_probe_sphere_view(radius=float(args_cli.sphere_radius))
    sim.reset()
    session.probe_view.initialize()
    session.ray_caster.reset()
    probe_sphere_view.initialize()
    session.scan_shape = _infer_scan_shape(
        int(session.ray_caster.num_rays),
        scan_size=session.scan_size,
        scan_resolution=session.scan_resolution,
    )
    session.center_ray_index = int(
        (session.scan_shape[0] // 2) * session.scan_shape[1] + (session.scan_shape[1] // 2)
    )
    for _ in range(2):
        sim.step(render=not bool(args_cli.headless))

    device = torch.device(sim.device)
    dt = 1.0 / 60.0
    total_steps = max(1, int(round(float(args_cli.duration_seconds) * 60.0)))
    resample_steps = max(1, int(round(float(resample_seconds) * 60.0)))

    patch_origin = _pick_patch_origin(session, device=device)
    if not bool(args_cli.headless):
        set_camera_view(
            [
                float(patch_origin[0].item()) + 6.0,
                float(patch_origin[1].item()) - 4.0,
                float(patch_origin[2].item()) + 4.0,
            ],
            [
                float(patch_origin[0].item()),
                float(patch_origin[1].item()),
                float(patch_origin[2].item()),
            ],
        )
    probe_pos_w = patch_origin.clone()
    probe_pos_w[2] += float(base_height)
    probe_yaw = float((2.0 * torch.rand(1, device=device) - 1.0).item() * math.pi)
    command_state = _resample_command_state(command_cfg, device=device)

    half_limit_x, half_limit_y = _compute_motion_limits(session)

    ranges_cfg = getattr(command_cfg, "ranges", None)
    lin_x_range = _as_range(getattr(ranges_cfg, "lin_vel_x", (-1.0, 1.0)), (-1.0, 1.0))
    lin_y_range = _as_range(getattr(ranges_cfg, "lin_vel_y", (-1.0, 1.0)), (-1.0, 1.0))
    ang_z_range = _as_range(getattr(ranges_cfg, "ang_vel_z", (-1.0, 1.0)), (-1.0, 1.0))

    print(
        f"[INFO] Go2 profile={args_cli.go2_env_profile} "
        f"terrain_size={session.terrain_size} scan_size={session.scan_size} "
        f"scan_resolution={session.scan_resolution:.3f} scan_shape={session.scan_shape}",
        flush=True,
    )
    print(
        f"[INFO] Command ranges lin_x={lin_x_range} lin_y={lin_y_range} ang_z={ang_z_range} "
        f"resample={resample_seconds:.2f}s "
        f"command_type={getattr(getattr(command_cfg, 'class_type', None), '__name__', type(command_cfg).__name__)}",
        flush=True,
    )
    print(
        f"[INFO] Base height={base_height:.3f} height_scan_offset={float(args_cli.height_scan_offset):.3f} "
        f"duration={float(args_cli.duration_seconds):.1f}s",
        flush=True,
    )
    print(
        f"[INFO] Motion bounds inside patch: x=[{-half_limit_x:.2f}, {half_limit_x:.2f}] "
        f"y=[{-half_limit_y:.2f}, {half_limit_y:.2f}]",
        flush=True,
    )
    print(
        f"[INFO] Patch origin: x={float(patch_origin[0].item()):.3f} "
        f"y={float(patch_origin[1].item()):.3f} z={float(patch_origin[2].item()):.3f}",
        flush=True,
    )
    if bool(args_cli.headless):
        print(
            "[WARN] Running in headless mode: viewport sphere/ray visuals are not shown. "
            "Run without --headless for IsaacLab on-screen visualization.",
            flush=True,
        )

    enable_heightmap_window = (not bool(args_cli.headless)) and (not bool(args_cli.hide_heightmap_window))
    if enable_heightmap_window and cv2 is None:
        enable_heightmap_window = False
        print("[WARN] OpenCV is not installed; disabling heightmap window.", flush=True)
    for step_idx in range(total_steps):
        if step_idx % resample_steps == 0:
            command_state = _resample_command_state(command_cfg, device=device)
            print(
                f"[INFO] step={step_idx:05d} sampled=[{float(command_state.sampled_vel_command_b[0]):+.3f}, "
                f"{float(command_state.sampled_vel_command_b[1]):+.3f}, "
                f"{float(command_state.sampled_vel_command_b[2]):+.3f}] "
                f"heading_env={int(command_state.is_heading_env)} standing_env={int(command_state.is_standing_env)}",
                flush=True,
            )

        command_b = _resolve_command_for_step(command_cfg, command_state, probe_heading=probe_yaw)

        probe_yaw += float(command_b[2].item()) * dt
        rel_xy = probe_pos_w[:2] - patch_origin[:2]
        proposed_rel_xy = _integrate_body_command_xy(rel_xy, probe_yaw, command_b, dt)
        clamped_rel_xy = proposed_rel_xy.clone()
        clamped_rel_xy[0] = torch.clamp(clamped_rel_xy[0], min=-half_limit_x, max=half_limit_x)
        clamped_rel_xy[1] = torch.clamp(clamped_rel_xy[1], min=-half_limit_y, max=half_limit_y)
        if not torch.allclose(clamped_rel_xy, proposed_rel_xy, atol=1.0e-6):
            command_state = _resample_command_state(command_cfg, device=device)
        probe_pos_w[:2] = patch_origin[:2] + clamped_rel_xy

        probe_quat_w = _yaw_to_quat(probe_yaw, device=device)
        session.probe_view.set_world_poses(probe_pos_w.unsqueeze(0), probe_quat_w.unsqueeze(0))

        sim.step(render=not bool(args_cli.headless))
        session.ray_caster.update(dt=dt, force_recompute=True)

        ray_hits_z_raw = session.ray_caster.data.ray_hits_w[0, :, 2]
        finite_hit_mask = torch.isfinite(ray_hits_z_raw)
        center_hit_z = ray_hits_z_raw[session.center_ray_index]
        if torch.isfinite(center_hit_z):
            probe_pos_w[2] = center_hit_z + float(base_height)
        elif torch.any(finite_hit_mask):
            probe_pos_w[2] = torch.median(ray_hits_z_raw[finite_hit_mask]) + float(base_height)

        session.probe_view.set_world_poses(probe_pos_w.unsqueeze(0), probe_quat_w.unsqueeze(0))
        probe_sphere_view.set_world_poses(probe_pos_w.unsqueeze(0), probe_quat_w.unsqueeze(0))

        ray_hits_z = torch.nan_to_num(ray_hits_z_raw, nan=0.0, posinf=0.0, neginf=0.0)
        sensor_z = session.ray_caster.data.pos_w[0, 2]
        height_scan = sensor_z - ray_hits_z - float(args_cli.height_scan_offset)
        height_map = height_scan.view(session.scan_shape[0], session.scan_shape[1])
        if enable_heightmap_window:
            _render_heightmap_window(
                height_map=height_map,
                command_b=command_b,
                elapsed_seconds=(step_idx + 1) * dt,
            )
    if cv2 is not None:
        try:
            cv2.destroyWindow("Go2 Command Probe Heightmap")
        except cv2.error:
            pass


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
