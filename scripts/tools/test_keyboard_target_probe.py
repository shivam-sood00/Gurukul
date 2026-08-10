"""Interactive probe test: drive a sphere and inspect live traversability targets."""

from __future__ import annotations

import argparse
import copy
import math
import sys
import weakref
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
    help="When using --opencv-heightmap, do not open the 2D OpenCV panel.",
)
parser.add_argument(
    "--opencv-heightmap",
    action="store_true",
    default=False,
    help="Show the 2D OpenCV heatmap panel (optional). Default is viewport-only visualization in Isaac Sim.",
)
parser.add_argument(
    "--disable-isaac-target-overlay",
    action="store_true",
    default=False,
    help="Disable 3D target-heightmap spheres in the Isaac Sim viewport.",
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
parser.add_argument(
    "--keyboard-control",
    action=argparse.BooleanOptionalAction,
    default=True,
    help="Enable/disable keyboard control for the probe (enabled by default).",
)
parser.add_argument(
    "--manual-command-step",
    type=float,
    default=0.25,
    help="Keyboard remote: scales linear/yaw acceleration (default 0.25 ≈ nominal). Use [ / ] to change gain live.",
)
parser.add_argument(
    "--simple-normal-exponent",
    type=float,
    default=2.0,
    help="Exponent for normal-z traversability term.",
)
parser.add_argument(
    "--simple-hole-weight",
    type=float,
    default=28.0,
    help="Penalty for local downward drops in target distribution.",
)
parser.add_argument(
    "--simple-step-weight",
    type=float,
    default=10.0,
    help="Penalty for local upward steps in target distribution.",
)
parser.add_argument(
    "--command-prior-strength",
    type=float,
    default=0.65,
    help="Blend strength for command-conditioned target prior (0 to 1).",
)
parser.add_argument(
    "--traversability-mode",
    type=str,
    default="next_steps",
    choices=("next_steps", "surface", "simple", "go2", "full"),
    help=(
        "Pretrain target: next_steps (default) = surface × velocity half-plane prior "
        "(next_step_topk=0); surface = dense surface map; go2 = dense handcrafted; simple = legacy; "
        "full = full defaults."
    ),
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
if args_cli.manual_command_step <= 0.0:
    parser.error("--manual-command-step must be > 0.")
if not 0.0 <= float(args_cli.command_prior_strength) <= 1.0:
    parser.error("--command-prior-strength must be in [0, 1].")
if args_cli.simple_normal_exponent <= 0.0:
    parser.error("--simple-normal-exponent must be > 0.")
if args_cli.simple_hole_weight < 0.0:
    parser.error("--simple-hole-weight must be >= 0.")
if args_cli.simple_step_weight < 0.0:
    parser.error("--simple-step-weight must be >= 0.")
if bool(args_cli.keyboard_control) and bool(args_cli.headless):
    parser.error("--keyboard-control requires non-headless mode.")

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

import carb
import omni

import isaaclab.sim as sim_utils
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
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
from Gurukul.tasks.manager_based.locomotion.velocity.pretrain_attention import (
    build_kinematic_traversability_targets,
    go2_default_traversability_kwargs,
    next_steps_traversability_defaults,
    surface_traversability_defaults,
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


def _integrate_world_velocity_xy(xy_w: torch.Tensor, yaw: float, command_b: torch.Tensor, dt: float) -> torch.Tensor:
    """Integrate body-frame planar velocity into world XY (same kinematics as training)."""
    cos_yaw = math.cos(float(yaw))
    sin_yaw = math.sin(float(yaw))
    vx = float(command_b[0].item())
    vy = float(command_b[1].item())
    out = xy_w.clone()
    out[0] += (vx * cos_yaw - vy * sin_yaw) * dt
    out[1] += (vx * sin_yaw + vy * cos_yaw) * dt
    return out


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


def _compute_world_motion_bounds(session: ProbeSession) -> tuple[float, float, float, float]:
    """World-frame XY limits over the full terrain grid (all sub-patch centers ± half cell minus scan margin)."""
    margin_x = 0.5 * session.scan_size[0] + session.scan_resolution
    margin_y = 0.5 * session.scan_size[1] + session.scan_resolution
    half_wx = max(0.05, 0.5 * session.terrain_size[0] - margin_x)
    half_wy = max(0.05, 0.5 * session.terrain_size[1] - margin_y)
    if session.terrain_importer.terrain_origins is None:
        raise RuntimeError("Terrain importer does not expose terrain origins.")
    origins = session.terrain_importer.terrain_origins.reshape(-1, 3)
    min_x = float(origins[:, 0].min().item()) - half_wx
    max_x = float(origins[:, 0].max().item()) + half_wx
    min_y = float(origins[:, 1].min().item()) - half_wy
    max_y = float(origins[:, 1].max().item()) + half_wy
    return min_x, max_x, min_y, max_y


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


WINDOW_NAME = "Go2 Keyboard Probe Target — traversability pretrain"

# namedWindow() must run once: calling it every frame can spawn duplicate windows / Qt issues on some systems.
_PROBE_OPENCV_WINDOW_READY = False


def _ensure_probe_opencv_window() -> None:
    global _PROBE_OPENCV_WINDOW_READY
    if cv2 is None or _PROBE_OPENCV_WINDOW_READY:
        return
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    _PROBE_OPENCV_WINDOW_READY = True


def _teardown_probe_opencv_window() -> None:
    global _PROBE_OPENCV_WINDOW_READY
    if cv2 is None:
        return
    try:
        cv2.destroyWindow(WINDOW_NAME)
    except cv2.error:
        pass
    _PROBE_OPENCV_WINDOW_READY = False


# OpenCV waitKeyEx: GTK-style 65361–65364 plus legacy values seen on some builds.
_KEY_ARROW_UP = {65362, 2490368}
_KEY_ARROW_DOWN = {65364, 2621440}
_KEY_ARROW_LEFT = {65361, 2424832}
_KEY_ARROW_RIGHT = {65363, 2555904}

# RC-remote style: hold keys to accelerate toward max body velocity; release coasts (decay after key-repeat grace).
_REMOTE_MOVE_GRACE_FRAMES = 4
_REMOTE_LIN_ACCEL_BASE = 5.0
_REMOTE_YAW_ACCEL_BASE = 12.0
_REMOTE_IDLE_DECAY = 7.0
_REMOTE_ACCEL_MULT_MIN = 0.25
_REMOTE_ACCEL_MULT_MAX = 4.0

_REMOTE_FORWARD = {ord("w"), ord("W")} | _KEY_ARROW_UP
_REMOTE_BACK = {ord("s"), ord("S")} | _KEY_ARROW_DOWN
_REMOTE_LEFT = {ord("a"), ord("A")} | _KEY_ARROW_LEFT
_REMOTE_RIGHT = {ord("d"), ord("D")} | _KEY_ARROW_RIGHT

# Isaac Sim carb.input key names (viewport-focused; same drive semantics as OpenCV remote).
_ISAAC_FORWARD = {"W", "UP", "NUMPAD_8"}
_ISAAC_BACK = {"S", "DOWN", "NUMPAD_2"}
_ISAAC_LEFT = {"A", "LEFT", "NUMPAD_4"}
_ISAAC_RIGHT = {"D", "RIGHT", "NUMPAD_6"}
_ISAAC_YAW_POS = {"Q", "NUMPAD_7"}
_ISAAC_YAW_NEG = {"E", "NUMPAD_9"}
_ISAAC_GAIN_DOWN = {"LEFT_BRACKET", "OPEN_BRACKET", "MINUS", "UNDERSCORE"}
_ISAAC_GAIN_UP = {"RIGHT_BRACKET", "CLOSE_BRACKET", "EQUAL", "PLUS"}
_ISAAC_RESET = {"X", "SPACE", "0", "KEY_0", "DIGIT0", "NUMPAD_0"}
_ISAAC_HELP = {"H"}
_ISAAC_QUIT = {"ESCAPE"}


def _target_overlay_bin_colors(num_bins: int) -> list[tuple[float, float, float]]:
    """Discrete HOT-style ramp for USD spheres: dark → red → orange → yellow (RGB 0–1)."""
    stops = [
        (0.06, 0.06, 0.06),
        (0.55, 0.02, 0.02),
        (0.92, 0.42, 0.0),
        (1.0, 0.95, 0.28),
    ]
    if num_bins <= 1:
        return [stops[0]]
    out: list[tuple[float, float, float]] = []
    for i in range(num_bins):
        t = float(i) / float(num_bins - 1)
        x = t * (len(stops) - 1)
        j = int(math.floor(x))
        j = min(j, len(stops) - 2)
        u = x - j
        c0, c1 = stops[j], stops[j + 1]
        out.append(
            (
                float(c0[0] * (1.0 - u) + c1[0] * u),
                float(c0[1] * (1.0 - u) + c1[1] * u),
                float(c0[2] * (1.0 - u) + c1[2] * u),
            )
        )
    return out


def _build_target_heightmap_visualizer(num_bins: int = 4) -> VisualizationMarkers:
    colors = _target_overlay_bin_colors(num_bins)
    markers = {
        f"bin_{idx}": sim_utils.SphereCfg(
            radius=0.014,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=colors[idx]),
        )
        for idx in range(num_bins)
    }
    cfg = VisualizationMarkersCfg(prim_path="/World/Go2CommandProbe/visual/target_heightmap", markers=markers)
    vis = VisualizationMarkers(cfg)
    vis.set_visibility(True)
    return vis


def _render_isaac_target_heightmap_overlay(
    visualizer: VisualizationMarkers,
    ray_hits_w: torch.Tensor,
    target_flat: torch.Tensor,
) -> None:
    """Draw colored spheres at ray hit sites; color bin ~ local percentile of target (HOT-style bins)."""
    hits = ray_hits_w[0].detach()
    tgt = target_flat.detach().reshape(-1).float()
    if hits.ndim != 2 or hits.shape[1] != 3 or hits.shape[0] != tgt.shape[0]:
        return
    valid = torch.isfinite(hits).all(dim=1) & torch.isfinite(tgt)
    if not torch.any(valid):
        return

    hits_v = hits[valid]
    vals = tgt[valid].detach().cpu().numpy()
    lo = float(np.percentile(vals, 5.0))
    hi = float(np.percentile(vals, 95.0))
    span = hi - lo
    if span < _TARGET_DISPLAY_FLOOR_SPAN:
        mid = 0.5 * (float(lo) + float(hi))
        half = 0.5 * float(_TARGET_DISPLAY_FLOOR_SPAN)
        lo = mid - half
        hi = mid + half
    elif span < 1.0e-8:
        hi = lo + 1.0e-8
    norm = np.clip((vals - lo) / (hi - lo), 0.0, 1.0).astype(np.float32)

    num_bins = int(visualizer.num_prototypes)
    bin_idx = np.minimum((norm * float(num_bins)).astype(np.int64), num_bins - 1)
    strength = norm
    scales_np = (0.55 + 0.85 * strength).astype(np.float32).reshape(-1, 1)
    scales = np.repeat(scales_np, 3, axis=1)

    z_boost = torch.as_tensor(0.018 + 0.022 * strength, device=hits_v.device, dtype=hits_v.dtype)
    pts = hits_v.clone()
    pts[:, 2] = pts[:, 2] + z_boost

    visualizer.visualize(
        translations=pts,
        scales=torch.as_tensor(scales, device=pts.device, dtype=pts.dtype),
        marker_indices=torch.as_tensor(bin_idx, device=pts.device, dtype=torch.int64),
    )


class IsaacProbeRemoteKeyboard:
    """RC-style probe control via carb keyboard while the Isaac viewport has focus (no OpenCV)."""

    def __init__(self) -> None:
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._keys_down: set[str] = set()
        self._pending: list[str] = []
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

    def close(self) -> None:
        if self._keyboard_sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def __del__(self) -> None:
        self.close()

    def _on_keyboard_event(self, event, *args, **kwargs) -> bool:
        name = getattr(getattr(event, "input", None), "name", None)
        if not isinstance(name, str):
            return True
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            self._keys_down.add(name)
            if name in _ISAAC_QUIT:
                self._pending.append("quit")
            elif name in _ISAAC_HELP:
                self._pending.append("help")
            elif name in _ISAAC_GAIN_DOWN:
                self._pending.append("gain_down")
            elif name in _ISAAC_GAIN_UP:
                self._pending.append("gain_up")
            elif name in _ISAAC_RESET:
                self._pending.append("reset")
        elif event.type == carb.input.KeyboardEventType.KEY_RELEASE:
            self._keys_down.discard(name)
        return True

    def advance(
        self,
        *,
        manual_command_b: torch.Tensor,
        dt: float,
        step_idx: int,
        last_movement_step: list[int],
        lin_x_range: tuple[float, float],
        lin_y_range: tuple[float, float],
        ang_z_range: tuple[float, float],
        accel_mult: float,
        remote_sensitivity: float,
    ) -> ManualKeyResult:
        out = ManualKeyResult()
        for ev in self._pending:
            if ev == "quit":
                out.quit_requested = True
            elif ev == "help":
                out.help_requested = True
            elif ev == "gain_down":
                lo, hi = float(_REMOTE_ACCEL_MULT_MIN), float(_REMOTE_ACCEL_MULT_MAX)
                new_m = max(lo, min(hi, float(accel_mult) * 0.8))
                out.new_accel_multiplier = float(new_m)
                out.accel_mult_changed = abs(new_m - float(accel_mult)) > 1.0e-9
            elif ev == "gain_up":
                lo, hi = float(_REMOTE_ACCEL_MULT_MIN), float(_REMOTE_ACCEL_MULT_MAX)
                new_m = max(lo, min(hi, float(accel_mult) * 1.25))
                out.new_accel_multiplier = float(new_m)
                out.accel_mult_changed = abs(new_m - float(accel_mult)) > 1.0e-9
            elif ev == "reset":
                manual_command_b[:] = 0.0
                out.velocity_reset = True
        self._pending.clear()

        sens = max(0.05, float(remote_sensitivity)) / 0.25
        lin_a = _REMOTE_LIN_ACCEL_BASE * float(accel_mult) * sens
        yaw_a = _REMOTE_YAW_ACCEL_BASE * float(accel_mult) * sens
        decay = math.exp(-float(_REMOTE_IDLE_DECAY) * float(dt))

        move_keys = (
            _ISAAC_FORWARD | _ISAAC_BACK | _ISAAC_LEFT | _ISAAC_RIGHT | _ISAAC_YAW_POS | _ISAAC_YAW_NEG
        )
        any_move = bool(self._keys_down & move_keys)

        if not any_move:
            if step_idx - last_movement_step[0] > _REMOTE_MOVE_GRACE_FRAMES:
                manual_command_b[0] *= decay
                manual_command_b[1] *= decay
                manual_command_b[2] *= decay
        else:
            last_movement_step[0] = step_idx
            if self._keys_down & _ISAAC_FORWARD:
                manual_command_b[0] += float(lin_a * dt)
            if self._keys_down & _ISAAC_BACK:
                manual_command_b[0] -= float(lin_a * dt)
            if self._keys_down & _ISAAC_LEFT:
                manual_command_b[1] += float(lin_a * dt)
            if self._keys_down & _ISAAC_RIGHT:
                manual_command_b[1] -= float(lin_a * dt)
            if self._keys_down & _ISAAC_YAW_POS:
                manual_command_b[2] += float(yaw_a * dt)
            if self._keys_down & _ISAAC_YAW_NEG:
                manual_command_b[2] -= float(yaw_a * dt)

        manual_command_b[0] = torch.clamp(manual_command_b[0], min=float(lin_x_range[0]), max=float(lin_x_range[1]))
        manual_command_b[1] = torch.clamp(manual_command_b[1], min=float(lin_y_range[0]), max=float(lin_y_range[1]))
        manual_command_b[2] = torch.clamp(manual_command_b[2], min=float(ang_z_range[0]), max=float(ang_z_range[1]))
        return out


# Minimum display span so flat terrain / near-uniform maps do not stretch numerical noise across the full colormap.
_HEIGHT_DISPLAY_FLOOR_SPAN_M = 0.05
_TARGET_DISPLAY_FLOOR_SPAN = 0.02


def _normalize_heatmap(
    values: np.ndarray,
    *,
    display_floor_span: float | None = None,
) -> tuple[np.ndarray, float, float]:
    finite_mask = np.isfinite(values)
    if not np.any(finite_mask):
        return np.zeros_like(values, dtype=np.float32), 0.0, 0.0

    finite_values = values[finite_mask]
    lo = float(np.percentile(finite_values, 5.0))
    hi = float(np.percentile(finite_values, 95.0))
    span = hi - lo
    if display_floor_span is not None and span < float(display_floor_span):
        mid = 0.5 * (float(lo) + float(hi))
        half = 0.5 * float(display_floor_span)
        lo = mid - half
        hi = mid + half
    elif span < 1.0e-6:
        hi = lo + 1.0e-6
    normalized = np.clip((np.nan_to_num(values, nan=lo) - lo) / (hi - lo), 0.0, 1.0)
    return normalized.astype(np.float32), float(np.min(finite_values)), float(np.max(finite_values))


def _estimate_normal_z_and_valid_mask(
    ray_hits_w: torch.Tensor,
    scan_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = int(ray_hits_w.shape[0])
    rows, cols = int(scan_shape[0]), int(scan_shape[1])
    hits = ray_hits_w.view(batch_size, rows, cols, 3)

    valid = torch.isfinite(hits).all(dim=-1)
    safe_hits = torch.nan_to_num(hits, nan=0.0, posinf=0.0, neginf=0.0)
    padded_hits = torch.nn.functional.pad(safe_hits.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="replicate").permute(0, 2, 3, 1)

    tangent_x = padded_hits[:, 2:, 1:-1, :] - padded_hits[:, :-2, 1:-1, :]
    tangent_y = padded_hits[:, 1:-1, 2:, :] - padded_hits[:, 1:-1, :-2, :]
    normals = torch.cross(tangent_y, tangent_x, dim=-1)
    normals = torch.nn.functional.normalize(normals, dim=-1, eps=1.0e-6)
    normal_z = normals[..., 2].abs()

    valid_pad = torch.nn.functional.pad(valid.unsqueeze(1).float(), (1, 1, 1, 1), mode="replicate").squeeze(1) > 0.5
    neighborhood_valid = (
        valid
        & valid_pad[:, 2:, 1:-1]
        & valid_pad[:, :-2, 1:-1]
        & valid_pad[:, 1:-1, 2:]
        & valid_pad[:, 1:-1, :-2]
    )
    normal_z = torch.where(neighborhood_valid, normal_z, torch.zeros_like(normal_z))
    return normal_z.reshape(batch_size, -1), valid.reshape(batch_size, -1).float()


@dataclass
class ManualKeyResult:
    """Result of handling one OpenCV key in RC-remote drive mode."""

    quit_requested: bool = False
    help_requested: bool = False
    accel_mult_changed: bool = False
    new_accel_multiplier: float | None = None
    velocity_reset: bool = False


def _apply_remote_keyboard(
    key_code: int,
    manual_command_b: torch.Tensor,
    dt: float,
    step_idx: int,
    last_movement_step: list[int],
    lin_x_range: tuple[float, float],
    lin_y_range: tuple[float, float],
    ang_z_range: tuple[float, float],
    accel_mult: float,
    remote_sensitivity: float,
) -> ManualKeyResult:
    """Hold WASD/arrows/QE to accelerate body-frame velocity; release to coast. [ / ] adjusts gain."""
    out = ManualKeyResult()
    sens = max(0.05, float(remote_sensitivity)) / 0.25
    lin_a = _REMOTE_LIN_ACCEL_BASE * float(accel_mult) * sens
    yaw_a = _REMOTE_YAW_ACCEL_BASE * float(accel_mult) * sens
    decay = math.exp(-float(_REMOTE_IDLE_DECAY) * float(dt))

    if key_code < 0:
        if step_idx - last_movement_step[0] > _REMOTE_MOVE_GRACE_FRAMES:
            manual_command_b[0] *= decay
            manual_command_b[1] *= decay
            manual_command_b[2] *= decay
        manual_command_b[0] = torch.clamp(manual_command_b[0], min=float(lin_x_range[0]), max=float(lin_x_range[1]))
        manual_command_b[1] = torch.clamp(manual_command_b[1], min=float(lin_y_range[0]), max=float(lin_y_range[1]))
        manual_command_b[2] = torch.clamp(manual_command_b[2], min=float(ang_z_range[0]), max=float(ang_z_range[1]))
        return out

    if key_code == 27:
        out.quit_requested = True
        return out

    lo, hi = float(_REMOTE_ACCEL_MULT_MIN), float(_REMOTE_ACCEL_MULT_MAX)
    if key_code in (ord("["), ord("{")):
        new_m = max(lo, min(hi, float(accel_mult) * 0.8))
        out.new_accel_multiplier = float(new_m)
        out.accel_mult_changed = abs(new_m - float(accel_mult)) > 1.0e-9
        return out
    if key_code in (ord("]"), ord("}")):
        new_m = max(lo, min(hi, float(accel_mult) * 1.25))
        out.new_accel_multiplier = float(new_m)
        out.accel_mult_changed = abs(new_m - float(accel_mult)) > 1.0e-9
        return out
    if key_code in (ord("h"), ord("H")):
        out.help_requested = True
        return out

    if key_code in (ord("x"), ord("X"), ord(" "), ord("0")):
        manual_command_b[:] = 0.0
        out.velocity_reset = True
        return out

    moved = False
    if key_code in _REMOTE_FORWARD:
        last_movement_step[0] = step_idx
        manual_command_b[0] += float(lin_a * dt)
        moved = True
    elif key_code in _REMOTE_BACK:
        last_movement_step[0] = step_idx
        manual_command_b[0] -= float(lin_a * dt)
        moved = True
    elif key_code in _REMOTE_LEFT:
        last_movement_step[0] = step_idx
        manual_command_b[1] += float(lin_a * dt)
        moved = True
    elif key_code in _REMOTE_RIGHT:
        last_movement_step[0] = step_idx
        manual_command_b[1] -= float(lin_a * dt)
        moved = True
    elif key_code in (ord("q"), ord("Q")):
        last_movement_step[0] = step_idx
        manual_command_b[2] += float(yaw_a * dt)
        moved = True
    elif key_code in (ord("e"), ord("E")):
        last_movement_step[0] = step_idx
        manual_command_b[2] -= float(yaw_a * dt)
        moved = True

    if moved:
        manual_command_b[0] = torch.clamp(manual_command_b[0], min=float(lin_x_range[0]), max=float(lin_x_range[1]))
        manual_command_b[1] = torch.clamp(manual_command_b[1], min=float(lin_y_range[0]), max=float(lin_y_range[1]))
        manual_command_b[2] = torch.clamp(manual_command_b[2], min=float(ang_z_range[0]), max=float(ang_z_range[1]))
    return out


def _draw_center_crosshair(img: np.ndarray, color: tuple[int, int, int] = (255, 255, 255)) -> None:
    w = int(img.shape[1])
    cx, cy = w // 2, int(img.shape[0]) // 2
    cv2.line(img, (cx - 10, cy), (cx + 10, cy), color, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - 10), (cx, cy + 10), color, 1, cv2.LINE_AA)


def _print_keyboard_help(*, viewport_focus: bool) -> None:
    focus = "Focus the Isaac Sim viewport" if viewport_focus else "Focus the OpenCV window"
    print(
        "[INFO] RC remote: hold WASD / arrows to accelerate (body vx, vy); hold Q/E for yaw rate; "
        "release to coast. X / Space / 0 — stop. [ / ] or - / = — gain. H — help. ESC — quit. "
        f"{focus} for keys.",
        flush=True,
    )


def _draw_body_velocity_arrow(
    img: np.ndarray,
    vx: float,
    vy: float,
    *,
    origin_xy: tuple[int, int],
    max_speed: float,
    radius: int = 36,
) -> None:
    """Draw body-frame velocity (vx forward, vy left) as an arrow in a small HUD box."""
    if max_speed <= 1.0e-6:
        max_speed = 1.0
    ox, oy = int(origin_xy[0]), int(origin_xy[1])
    scale = (radius - 6) / max_speed
    # Image: +y is down, body +vx forward -> up on screen (-dy).
    ex = ox + float(vy) * scale
    ey = oy - float(vx) * scale
    cv2.circle(img, (ox, oy), radius, (40, 40, 40), 1, cv2.LINE_AA)
    cv2.line(
        img,
        (ox, oy),
        (int(round(ex)), int(round(ey))),
        (120, 220, 255),
        2,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "+vx",
        (ox - 14, oy - radius - 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        img,
        "+vy L",
        (ox + radius + 2, oy + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )


def _add_title_bar(
    img: np.ndarray,
    lines: list[str],
    *,
    bar_bgr: tuple[int, int, int] = (28, 28, 32),
    text_bgr: tuple[int, int, int] = (235, 235, 235),
) -> np.ndarray:
    bar_h = 22 + 18 * max(0, len(lines) - 1)
    bar = np.full((bar_h, img.shape[1], 3), bar_bgr, dtype=np.uint8)
    for i, line in enumerate(lines):
        y = 18 + i * 18
        cv2.putText(bar, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_bgr, 1, cv2.LINE_AA)
    return np.vstack((bar, img))


def _render_probe_window(
    height_map: torch.Tensor,
    target_map: torch.Tensor,
    command_b: torch.Tensor,
    elapsed_seconds: float,
    keyboard_mode: bool,
    *,
    profile_name: str,
    remote_accel_mult: float,
    remote_sensitivity: float,
    command_prior_strength: float,
) -> int:
    if cv2 is None:
        return -1

    height_np = height_map.detach().cpu().numpy()
    target_np = target_map.detach().cpu().numpy()

    height_norm, height_min, height_max = _normalize_heatmap(
        height_np, display_floor_span=_HEIGHT_DISPLAY_FLOOR_SPAN_M
    )
    target_norm, target_min, target_max = _normalize_heatmap(
        target_np, display_floor_span=_TARGET_DISPLAY_FLOOR_SPAN
    )

    height_u8 = np.uint8(255.0 * height_norm)
    height_color = cv2.cvtColor(height_u8, cv2.COLOR_GRAY2BGR)
    target_color = cv2.applyColorMap(np.uint8(255.0 * target_norm), cv2.COLORMAP_HOT)
    blend = cv2.addWeighted(height_color, 0.45, target_color, 0.55, 0.0)

    scale = max(5, int(round(280 / max(1, height_color.shape[1]))))
    new_wh = (height_color.shape[1] * scale, height_color.shape[0] * scale)
    height_color = cv2.resize(height_color, new_wh, interpolation=cv2.INTER_NEAREST)
    target_color = cv2.resize(target_color, new_wh, interpolation=cv2.INTER_NEAREST)
    blend = cv2.resize(blend, new_wh, interpolation=cv2.INTER_NEAREST)

    sep_w = 8
    sep = np.full((height_color.shape[0], sep_w, 3), 18, dtype=np.uint8)
    row = np.hstack((height_color, sep, target_color, sep, blend))

    cmd_x = float(command_b[0].item())
    cmd_y = float(command_b[1].item())
    cmd_yaw = float(command_b[2].item())

    for sub in (0, 1, 2):
        x0 = sub * (height_color.shape[1] + sep_w)
        tile = row[:, x0 : x0 + height_color.shape[1], :]
        _draw_center_crosshair(tile, (255, 255, 255))

    labels = [
        "Height (gray): distance to hit in display range",
        "Target (HOT): black→red→yellow = low→high mass",
        "Overlay: gray height + HOT target",
    ]
    label_h = 22
    for sub, label in enumerate(labels):
        x0 = sub * (height_color.shape[1] + sep_w) + 6
        cv2.putText(
            row,
            label,
            (x0, label_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )

    hud_x = row.shape[1] - 120
    hud_y = height_color.shape[0] - 12
    max_lin = max(abs(cmd_x), abs(cmd_y), 0.5)
    _draw_body_velocity_arrow(row, cmd_x, cmd_y, origin_xy=(hud_x, hud_y), max_speed=float(max_lin))

    mode = "keyboard" if keyboard_mode else "auto command"
    title_lines = [
        f"Attention pretrain preview  |  profile={profile_name}  |  {mode}  |  t={elapsed_seconds:05.1f}s",
        (
            f"cmd_b=[{cmd_x:+.2f}, {cmd_y:+.2f}, {cmd_yaw:+.2f}]  |  prior={command_prior_strength:.2f}  |  "
            f"height∈[{height_min:+.3f},{height_max:+.3f}]  target∈[{target_min:+.4f},{target_max:+.4f}]"
        ),
    ]
    if keyboard_mode:
        title_lines.append(
            f"RC remote  gain×{remote_accel_mult:.2f}  sens={remote_sensitivity:.3f}  |  [ / ] gain  |  H help"
        )

    panel = _add_title_bar(row, title_lines)

    footer_h = 68
    footer = np.full((footer_h, panel.shape[1], 3), (22, 22, 26), dtype=np.uint8)
    help_lines = [
        "Goal: learn attention on heightmap tokens that match kinematic traversability (flat support, no holes/steps) and command direction.",
        "Height is grayscale (darker = smaller value in the 5–95% window); on flat ground a 5 cm display floor keeps the tile uniform gray (no fake contrast from noise).",
        (
            "Hold WASD/arrows: accel vx, vy  ·  hold Q/E: yaw  ·  X Space 0: stop  ·  ESC: quit"
            + ("  ·  [ ]: gain" if keyboard_mode else "")
        ),
    ]
    for i, line in enumerate(help_lines):
        cv2.putText(
            footer,
            line,
            (10, 18 + i * 18),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.42,
            (200, 200, 205),
            1,
            cv2.LINE_AA,
        )
    panel = np.vstack((panel, footer))

    max_w = 1680
    if panel.shape[1] > max_w:
        r = max_w / float(panel.shape[1])
        panel = cv2.resize(panel, (int(panel.shape[1] * r), int(panel.shape[0] * r)), interpolation=cv2.INTER_AREA)

    _ensure_probe_opencv_window()
    cv2.imshow(WINDOW_NAME, panel)
    key_code = cv2.waitKeyEx(1)
    if key_code < 0:
        return -1
    return int(key_code)


def _resolve_probe_traversability_kwargs(resolution_m: float) -> dict[str, float | int | None]:
    mode = str(args_cli.traversability_mode).lower()
    if mode == "full":
        return {}
    if mode == "next_steps":
        return next_steps_traversability_defaults()
    if mode == "surface":
        return surface_traversability_defaults()
    if mode == "go2":
        return go2_default_traversability_kwargs(resolution_m=float(resolution_m))
    return {
        "slope_weight": 0.0,
        "roughness_weight": 0.0,
        "support_exponent": 0.0,
        "hole_weight": float(args_cli.simple_hole_weight),
        "step_weight": float(args_cli.simple_step_weight),
        "normal_exponent": float(args_cli.simple_normal_exponent),
    }


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

    world_min_x, world_max_x, world_min_y, world_max_y = _compute_world_motion_bounds(session)

    ranges_cfg = getattr(command_cfg, "ranges", None)
    lin_x_range = _as_range(getattr(ranges_cfg, "lin_vel_x", (-1.0, 1.0)), (-1.0, 1.0))
    lin_y_range = _as_range(getattr(ranges_cfg, "lin_vel_y", (-1.0, 1.0)), (-1.0, 1.0))
    ang_z_range = _as_range(getattr(ranges_cfg, "ang_vel_z", (-1.0, 1.0)), (-1.0, 1.0))

    manual_lin_x_max = max(abs(float(lin_x_range[0])), abs(float(lin_x_range[1])), 1.0)
    manual_lin_y_max = max(abs(float(lin_y_range[0])), abs(float(lin_y_range[1])), 0.6)
    manual_ang_z_max = max(abs(float(ang_z_range[0])), abs(float(ang_z_range[1])), 1.0)
    manual_lin_x_range = (-manual_lin_x_max, manual_lin_x_max)
    manual_lin_y_range = (-manual_lin_y_max, manual_lin_y_max)
    manual_ang_z_range = (-manual_ang_z_max, manual_ang_z_max)

    print(
        f"[INFO] Go2 profile={args_cli.go2_env_profile} "
        f"terrain_size={session.terrain_size} scan_size={session.scan_size} "
        f"scan_resolution={session.scan_resolution:.3f} scan_shape={session.scan_shape} "
        f"traversability_mode={args_cli.traversability_mode}",
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
        f"[INFO] World motion bounds (full terrain grid): "
        f"x=[{world_min_x:.2f}, {world_max_x:.2f}] y=[{world_min_y:.2f}, {world_max_y:.2f}]",
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

    enable_heightmap_window = (
        (not bool(args_cli.headless))
        and bool(args_cli.opencv_heightmap)
        and (not bool(args_cli.hide_heightmap_window))
    )
    if enable_heightmap_window and cv2 is None:
        enable_heightmap_window = False
        print("[WARN] OpenCV is not installed; disabling OpenCV heightmap panel.", flush=True)

    isaac_target_overlay = (not bool(args_cli.headless)) and (not bool(args_cli.disable_isaac_target_overlay))
    target_vis: VisualizationMarkers | None = None
    if isaac_target_overlay:
        target_vis = _build_target_heightmap_visualizer()

    if not bool(args_cli.headless):
        print(
            "[INFO] Visualization: "
            f"3D target overlay={'on' if isaac_target_overlay else 'off'}; "
            f"OpenCV 2D panel={'on' if enable_heightmap_window else 'off'} "
            "(use --opencv-heightmap for the legacy 2D heatmap window).",
            flush=True,
        )

    keyboard_mode = bool(args_cli.keyboard_control)
    manual_command_b = torch.zeros(3, device=device, dtype=torch.float32)
    remote_sensitivity = float(args_cli.manual_command_step)
    current_accel_mult = 1.0
    last_movement_step = [-1_000_000_000]

    isaac_kb: IsaacProbeRemoteKeyboard | None = None
    use_isaac_keyboard = keyboard_mode and (not enable_heightmap_window)
    if use_isaac_keyboard:
        isaac_kb = IsaacProbeRemoteKeyboard()

    if keyboard_mode:
        _print_keyboard_help(viewport_focus=use_isaac_keyboard)
        if use_isaac_keyboard:
            print("[INFO] Keyboard input is read from the Isaac Sim viewport (click the viewport first).", flush=True)
        else:
            print("[INFO] Click the OpenCV window to focus keyboard input.", flush=True)
        print(
            f"[INFO] Remote limits lin_x={manual_lin_x_range} lin_y={manual_lin_y_range} yaw={manual_ang_z_range} "
            f"gain∈[{_REMOTE_ACCEL_MULT_MIN:.2f}, {_REMOTE_ACCEL_MULT_MAX:.2f}]  "
            f"--manual-command-step={remote_sensitivity:.3f} (scales acceleration)",
            flush=True,
        )

    step_idx = 0
    print("[INFO] Entering interactive probe loop.", flush=True)
    while True:
        if (not keyboard_mode) and (step_idx > 0) and (not bool(args_cli.headless)) and (not simulation_app.is_running()):
            break
        if keyboard_mode and (not bool(args_cli.headless)) and (not simulation_app.is_running()):
            break
        if (not keyboard_mode) and (step_idx >= total_steps):
            break
        if keyboard_mode:
            command_b = manual_command_b.clone()
        else:
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
        proposed_xy = _integrate_world_velocity_xy(probe_pos_w[:2], probe_yaw, command_b, dt)
        clamped_xy = proposed_xy.clone()
        clamped_xy[0] = torch.clamp(clamped_xy[0], min=world_min_x, max=world_max_x)
        clamped_xy[1] = torch.clamp(clamped_xy[1], min=world_min_y, max=world_max_y)
        if (not keyboard_mode) and (not torch.allclose(clamped_xy, proposed_xy, atol=1.0e-6)):
            command_state = _resample_command_state(command_cfg, device=device)
        probe_pos_w[:2] = clamped_xy

        probe_quat_w = _yaw_to_quat(probe_yaw, device=device)
        session.probe_view.set_world_poses(probe_pos_w.unsqueeze(0), probe_quat_w.unsqueeze(0))

        sim.step(render=not bool(args_cli.headless))
        session.ray_caster.update(dt=dt, force_recompute=True)

        ray_hits_w = session.ray_caster.data.ray_hits_w
        ray_hits_z_raw = ray_hits_w[0, :, 2]
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

        normal_z, valid_mask = _estimate_normal_z_and_valid_mask(ray_hits_w, scan_shape=session.scan_shape)
        trav_kw = _resolve_probe_traversability_kwargs(float(session.scan_resolution))
        target = build_kinematic_traversability_targets(
            height_scans=height_scan.unsqueeze(0),
            commands_b=command_b.unsqueeze(0),
            scan_shape=session.scan_shape,
            raster_resolution=float(session.scan_resolution),
            raster_resolution_y=float(session.scan_resolution),
            surface_normal_z=normal_z[0:1],
            valid_mask=valid_mask[0:1],
            apply_forward_prior=False,
            apply_command_prior=True,
            command_prior_strength=float(args_cli.command_prior_strength),
            **trav_kw,
        )
        target_map = target[0].view(session.scan_shape[0], session.scan_shape[1])

        key_code = -1
        if enable_heightmap_window:
            key_code = _render_probe_window(
                height_map=height_map,
                target_map=target_map,
                command_b=command_b,
                elapsed_seconds=(step_idx + 1) * dt,
                keyboard_mode=keyboard_mode,
                profile_name=str(args_cli.go2_env_profile),
                remote_accel_mult=current_accel_mult if keyboard_mode else 1.0,
                remote_sensitivity=remote_sensitivity,
                command_prior_strength=float(args_cli.command_prior_strength),
            )

        if isaac_target_overlay and target_vis is not None:
            _render_isaac_target_heightmap_overlay(
                target_vis,
                ray_hits_w,
                target[0],
            )

        if keyboard_mode:
            if use_isaac_keyboard and isaac_kb is not None:
                kr = isaac_kb.advance(
                    manual_command_b=manual_command_b,
                    dt=dt,
                    step_idx=step_idx,
                    last_movement_step=last_movement_step,
                    lin_x_range=manual_lin_x_range,
                    lin_y_range=manual_lin_y_range,
                    ang_z_range=manual_ang_z_range,
                    accel_mult=current_accel_mult,
                    remote_sensitivity=remote_sensitivity,
                )
            else:
                kr = _apply_remote_keyboard(
                    key_code=key_code,
                    manual_command_b=manual_command_b,
                    dt=dt,
                    step_idx=step_idx,
                    last_movement_step=last_movement_step,
                    lin_x_range=manual_lin_x_range,
                    lin_y_range=manual_lin_y_range,
                    ang_z_range=manual_ang_z_range,
                    accel_mult=current_accel_mult,
                    remote_sensitivity=remote_sensitivity,
                )
            if kr.accel_mult_changed and kr.new_accel_multiplier is not None:
                current_accel_mult = float(kr.new_accel_multiplier)
                print(f"[INFO] remote gain×{current_accel_mult:.3f}", flush=True)
            if kr.velocity_reset:
                print("[INFO] remote velocity zeroed", flush=True)
            if kr.help_requested:
                _print_keyboard_help(viewport_focus=use_isaac_keyboard)
            if kr.quit_requested:
                break
        elif key_code == 27:
            break

        step_idx += 1

    _teardown_probe_opencv_window()
    if isaac_kb is not None:
        isaac_kb.close()


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
