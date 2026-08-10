"""Visualize pretrained REAL terrain attention with a command-driven Go2 probe scanner."""

from __future__ import annotations

import argparse
import copy
import math
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description=(
        "Pretrained REAL attention probe viewer using exact Go2 IsaacLab terrain/scanner settings. "
        "No Go2 articulation is spawned."
    )
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default="artifacts/real_attention/pretrained_attention.pt",
    help="Path to pretrained REAL terrain-attention checkpoint.",
)
parser.add_argument(
    "--go2-env-profile",
    type=str,
    default="start_sparse",
    choices=("rough", "start_sparse", "real_sparse", "real_beam"),
    help="Go2 env config profile to clone for terrain/scanner/command settings.",
)
parser.add_argument(
    "--duration-seconds",
    type=float,
    default=0.0,
    help=(
        "Total visualization duration in simulated seconds. "
        "Set <= 0 to run until quit (default)."
    ),
)
parser.add_argument(
    "--command-resample-seconds",
    type=float,
    default=None,
    help="How often to resample velocity commands. Defaults to selected Go2 command config.",
)
parser.add_argument(
    "--base-height",
    type=float,
    default=None,
    help="Probe base height above terrain. Defaults to selected Go2 base-height target.",
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
parser.add_argument(
    "--attention-device",
    type=str,
    default=None,
    help="Device for pretrained attention inference. Defaults to simulation device.",
)
parser.add_argument(
    "--attention-head-index",
    type=int,
    default=-1,
    help="Head index to visualize (-1 means mean across heads).",
)
parser.add_argument("--seed", type=int, default=7, help="Random seed.")
parser.add_argument(
    "--hide-attention-window",
    action="store_true",
    default=False,
    help="Disable the 2D OpenCV height/attention window.",
)
parser.add_argument(
    "--hide-heightmap-window",
    action="store_true",
    default=False,
    help=argparse.SUPPRESS,
)
parser.add_argument(
    "--disable-ray-debug-vis",
    action="store_true",
    default=False,
    help="Disable 3D ray-hit visualization in IsaacLab viewport.",
)
parser.add_argument(
    "--disable-attention-overlay",
    action="store_true",
    default=False,
    help="Disable 3D attention spheres at ray hits in the Isaac viewport (same scheme as keyboard target overlay).",
)
parser.add_argument(
    "--disable-teacher-target",
    action="store_true",
    default=False,
    help="Hide the kinematic teacher-target column (next_steps + velocity half-plane prior; same as offline pretrain).",
)
parser.add_argument(
    "--command-prior-strength",
    type=float,
    default=0.65,
    help="Teacher column: blend strength for command prior (matches pretrain_real_attention default).",
)
parser.add_argument(
    "--random-patch",
    action="store_true",
    default=False,
    help="Sample a random terrain patch instead of using a central patch.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.hide_attention_window = bool(args_cli.hide_attention_window or args_cli.hide_heightmap_window)

if args_cli.command_resample_seconds is not None and args_cli.command_resample_seconds <= 0.0:
    parser.error("--command-resample-seconds must be > 0 when provided.")
if args_cli.sphere_radius <= 0.0:
    parser.error("--sphere-radius must be > 0.")
if args_cli.base_height is not None and args_cli.base_height < 0.0:
    parser.error("--base-height must be >= 0.")
if args_cli.attention_head_index < -1:
    parser.error("--attention-head-index must be >= -1.")
if not Path(args_cli.checkpoint).is_file():
    parser.error(f"Checkpoint not found: {args_cli.checkpoint}")
if not 0.0 <= float(args_cli.command_prior_strength) <= 1.0:
    parser.error("--command-prior-strength must be in [0, 1].")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import os

# Quieter Qt/OpenCV HighGUI when conda OpenCV has no bundled fonts (harmless but noisy).
os.environ.setdefault("QT_LOGGING_RULES", "qt.qpa.fonts=false")

import numpy as np
import torch
import torch.nn.functional as F

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.api.simulation_context import SimulationContext
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.viewports import set_camera_view

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
    load_attention_pretrainer_model,
    next_steps_traversability_defaults,
)


WINDOW_NAME = "Go2 Pretrained Attention Probe"


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
        vy = 0.0
        turn_threshold = float(getattr(command_cfg, "turn_trigger_threshold", 0.3))
        if vx < turn_threshold:
            vx = 0.0
            wz = _sample_uniform(*ang_z_range, device=device)
        else:
            wz = 0.0

    if _is_uniform_threshold_command(command_cfg):
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
    return int(num_rays), 1


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
        center_xy = terrain_origins[:, :2].mean(dim=0, keepdim=True)
        distances = torch.linalg.norm(terrain_origins[:, :2] - center_xy, dim=1)
        patch_index = int(torch.argmin(distances).item())
    return terrain_origins[patch_index].to(device=device, dtype=torch.float32)


def _build_probe_sphere_view(radius: float) -> XFormPrim:
    sphere_prim_path = "/World/Go2PretrainedAttentionProbe/visual/probe_sphere"
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


_HEIGHT_DISPLAY_FLOOR_SPAN_M = 0.05
_ATTENTION_DISPLAY_FLOOR_SPAN = 0.02


def _estimate_normal_z_and_valid_mask(
    ray_hits_w: torch.Tensor,
    scan_shape: tuple[int, int],
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = int(ray_hits_w.shape[0])
    rows, cols = int(scan_shape[0]), int(scan_shape[1])
    hits = ray_hits_w.view(batch_size, rows, cols, 3)

    valid = torch.isfinite(hits).all(dim=-1)
    safe_hits = torch.nan_to_num(hits, nan=0.0, posinf=0.0, neginf=0.0)
    padded_hits = F.pad(safe_hits.permute(0, 3, 1, 2), (1, 1, 1, 1), mode="replicate").permute(0, 2, 3, 1)

    tangent_x = padded_hits[:, 2:, 1:-1, :] - padded_hits[:, :-2, 1:-1, :]
    tangent_y = padded_hits[:, 1:-1, 2:, :] - padded_hits[:, 1:-1, :-2, :]
    normals = torch.cross(tangent_y, tangent_x, dim=-1)
    normals = F.normalize(normals, dim=-1, eps=1.0e-6)
    normal_z = normals[..., 2].abs()

    valid_pad = F.pad(valid.unsqueeze(1).float(), (1, 1, 1, 1), mode="replicate").squeeze(1) > 0.5
    neighborhood_valid = (
        valid
        & valid_pad[:, 2:, 1:-1]
        & valid_pad[:, :-2, 1:-1]
        & valid_pad[:, 1:-1, 2:]
        & valid_pad[:, 1:-1, :-2]
    )
    normal_z = torch.where(neighborhood_valid, normal_z, torch.zeros_like(normal_z))
    return normal_z.reshape(batch_size, -1), valid.reshape(batch_size, -1).float()


def _compute_teacher_target_map(
    height_scan: torch.Tensor,
    command_b: torch.Tensor,
    ray_hits_w: torch.Tensor,
    scan_shape: tuple[int, int],
    scan_resolution: float,
    *,
    command_prior_strength: float,
) -> torch.Tensor:
    """Kinematic next_steps target (surface × velocity half-plane prior), matching offline pretrain."""
    normal_z, valid_mask = _estimate_normal_z_and_valid_mask(ray_hits_w, scan_shape)
    cmds = command_b.unsqueeze(0) if command_b.ndim == 1 else command_b
    hs = height_scan.flatten().unsqueeze(0)
    dev = hs.device
    dtype = hs.dtype
    trav_kw = next_steps_traversability_defaults()
    with torch.inference_mode():
        teacher = build_kinematic_traversability_targets(
            height_scans=hs,
            commands_b=cmds.to(device=dev, dtype=dtype),
            scan_shape=scan_shape,
            raster_resolution=float(scan_resolution),
            raster_resolution_y=float(scan_resolution),
            surface_normal_z=normal_z.to(device=dev, dtype=dtype),
            valid_mask=valid_mask.to(device=dev, dtype=dtype),
            apply_forward_prior=False,
            apply_command_prior=True,
            command_prior_strength=float(command_prior_strength),
            **trav_kw,
        )
    return teacher.view(scan_shape[0], scan_shape[1])


def _attention_overlay_bin_colors(num_bins: int) -> list[tuple[float, float, float]]:
    """Discrete HOT-style ramp for USD spheres (dark → red → orange → yellow), RGB 0–1."""
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


def _build_attention_heightmap_visualizer(num_bins: int = 4) -> VisualizationMarkers:
    colors = _attention_overlay_bin_colors(num_bins)
    markers = {
        f"bin_{idx}": sim_utils.SphereCfg(
            radius=0.014,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=colors[idx]),
        )
        for idx in range(num_bins)
    }
    cfg = VisualizationMarkersCfg(
        prim_path="/World/Go2PretrainedAttentionProbe/visual/attention_heightmap",
        markers=markers,
    )
    vis = VisualizationMarkers(cfg)
    vis.set_visibility(True)
    return vis


def _render_isaac_attention_heightmap_overlay(
    visualizer: VisualizationMarkers,
    ray_hits_w: torch.Tensor,
    attention_flat: torch.Tensor,
) -> None:
    """Colored spheres at ray hits; color/size from attention (percentile + floor, same as 2D window)."""
    hits = ray_hits_w[0].detach()
    attn = attention_flat.detach().reshape(-1).float()
    if hits.ndim != 2 or hits.shape[1] != 3 or hits.shape[0] != attn.shape[0]:
        return
    valid = torch.isfinite(hits).all(dim=1) & torch.isfinite(attn)
    if not torch.any(valid):
        return

    hits_v = hits[valid]
    vals = attn[valid].detach().cpu().numpy()
    lo = float(np.percentile(vals, 5.0))
    hi = float(np.percentile(vals, 95.0))
    span = hi - lo
    if span < _ATTENTION_DISPLAY_FLOOR_SPAN:
        mid = 0.5 * (float(lo) + float(hi))
        half = 0.5 * float(_ATTENTION_DISPLAY_FLOOR_SPAN)
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


def _draw_center_crosshair(img: np.ndarray, color: tuple[int, int, int] = (255, 255, 255)) -> None:
    w = int(img.shape[1])
    cx, cy = w // 2, int(img.shape[0]) // 2
    cv2.line(img, (cx - 10, cy), (cx + 10, cy), color, 1, cv2.LINE_AA)
    cv2.line(img, (cx, cy - 10), (cx, cy + 10), color, 1, cv2.LINE_AA)


def _normalize_map(
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


def _colorize_map(
    values: np.ndarray,
    target_width: int = 300,
    *,
    display_floor_span: float | None = None,
    grayscale: bool = False,
    colormap: int | None = None,
) -> tuple[np.ndarray, float, float]:
    if cv2 is None:
        raise RuntimeError("OpenCV is required for _colorize_map.")
    cmap = cv2.COLORMAP_HOT if colormap is None else colormap
    norm, vmin, vmax = _normalize_map(values, display_floor_span=display_floor_span)
    u8 = np.uint8(255.0 * norm)
    if grayscale:
        img = cv2.cvtColor(u8, cv2.COLOR_GRAY2BGR)
    else:
        img = cv2.applyColorMap(u8, cmap)
    scale = max(4, int(round(float(target_width) / max(1, img.shape[1]))))
    img = cv2.resize(img, (img.shape[1] * scale, img.shape[0] * scale), interpolation=cv2.INTER_NEAREST)
    return img, vmin, vmax


def _render_attention_window(
    height_map: torch.Tensor,
    attention_mean_map: torch.Tensor,
    attention_selected_map: torch.Tensor,
    command_b: torch.Tensor,
    elapsed_seconds: float,
    selected_head: int,
    entropy: float,
    topk5_mass: float,
    *,
    teacher_map: torch.Tensor | None = None,
) -> bool:
    if cv2 is None:
        return False

    tw = 210 if teacher_map is not None else 280

    height_np = height_map.detach().cpu().numpy()
    mean_np = attention_mean_map.detach().cpu().numpy()
    selected_np = attention_selected_map.detach().cpu().numpy()

    height_img, hmin, hmax = _colorize_map(
        height_np,
        target_width=tw,
        display_floor_span=_HEIGHT_DISPLAY_FLOOR_SPAN_M,
        grayscale=True,
    )
    mean_img, mmin, mmax = _colorize_map(
        mean_np, target_width=tw, display_floor_span=_ATTENTION_DISPLAY_FLOOR_SPAN
    )
    selected_img, smin, smax = _colorize_map(
        selected_np, target_width=tw, display_floor_span=_ATTENTION_DISPLAY_FLOOR_SPAN
    )

    tmin: float | None = None
    tmax: float | None = None
    if teacher_map is not None:
        teacher_np = teacher_map.detach().cpu().numpy()
        teacher_img, tmin, tmax = _colorize_map(
            teacher_np, target_width=tw, display_floor_span=_ATTENTION_DISPLAY_FLOOR_SPAN
        )
        separator = np.full((height_img.shape[0], 6, 3), 24, dtype=np.uint8)
        panel = np.hstack((height_img, separator, teacher_img, separator, mean_img, separator, selected_img))
        col_labels = (
            "Height (gray)",
            "Teacher target (HOT) next_steps",
            "Attn mean (HOT)",
            "Attn selected (HOT)",
        )
    else:
        separator = np.full((height_img.shape[0], 6, 3), 24, dtype=np.uint8)
        panel = np.hstack((height_img, separator, mean_img, separator, selected_img))
        col_labels = (
            "Height (gray): scan distance",
            "Attention mean (HOT)",
            "Attention selected (HOT)",
        )

    sep_w = 6
    label_h = 22
    for sub, label in enumerate(col_labels):
        x0 = sub * (height_img.shape[1] + sep_w) + 6
        cv2.putText(
            panel,
            label,
            (x0, label_h),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.40,
            (240, 240, 240),
            1,
            cv2.LINE_AA,
        )
    for sub in range(len(col_labels)):
        x0 = sub * (height_img.shape[1] + sep_w)
        tile = panel[:, x0 : x0 + height_img.shape[1], :]
        _draw_center_crosshair(tile, (255, 255, 255))

    cmd_x = float(command_b[0].item())
    cmd_y = float(command_b[1].item())
    cmd_w = float(command_b[2].item())
    selected_label = "mean" if selected_head < 0 else f"head{selected_head}"

    cv2.putText(
        panel,
        f"t={elapsed_seconds:05.1f}s cmd=[{cmd_x:+.2f}, {cmd_y:+.2f}, {cmd_w:+.2f}] selected={selected_label}",
        (8, 46),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.52,
        (238, 238, 238),
        1,
        cv2.LINE_AA,
    )
    if teacher_map is not None and tmin is not None and tmax is not None:
        cv2.putText(
            panel,
            f"h[{hmin:+.3f},{hmax:+.3f}]m  teacher[{tmin:+.5f},{tmax:+.5f}]  mean[{mmin:+.5f},{mmax:+.5f}]  sel[{smin:+.5f},{smax:+.5f}]",
            (8, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (228, 228, 228),
            1,
            cv2.LINE_AA,
        )
    else:
        cv2.putText(
            panel,
            f"height[{hmin:+.3f},{hmax:+.3f}]m  mean[{mmin:+.5f},{mmax:+.5f}]  sel[{smin:+.5f},{smax:+.5f}]",
            (8, 70),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.48,
            (228, 228, 228),
            1,
            cv2.LINE_AA,
        )
    cv2.putText(
        panel,
        f"entropy={entropy:.4f} top5_mass={topk5_mass:.4f}",
        (8, 94),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.48,
        (220, 220, 220),
        1,
        cv2.LINE_AA,
    )
    cv2.putText(
        panel,
        "HOT: black→red→yellow; flat height: 5 cm display floor; teacher = offline pretrain target",
        (8, 118),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.45,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )

    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
    cv2.imshow(WINDOW_NAME, panel)
    key = int(cv2.waitKey(1)) & 0xFF
    if key in (27, ord("q"), ord("Q")):
        return True
    return False


def main() -> None:
    torch.manual_seed(int(args_cli.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args_cli.seed))

    sim_device = args_cli.device if args_cli.device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
    attention_device = args_cli.attention_device if args_cli.attention_device is not None else str(sim_device)
    sim = _make_simulation_context(device=str(sim_device), headless=bool(args_cli.headless))

    model, metadata = load_attention_pretrainer_model(args_cli.checkpoint, map_location=attention_device)
    model = model.to(attention_device)
    model.eval()

    light_cfg = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 0.9))
    light_cfg.func("/World/Go2PretrainedAttentionProbeLight", light_cfg)

    if not args_cli.headless:
        set_camera_view([36.0, 0.0, 24.0], [0.0, 0.0, 0.0])

    env_cfg = _build_reference_env_cfg(args_cli.go2_env_profile)
    command_cfg = copy.deepcopy(env_cfg.commands.base_velocity)
    base_height = float(args_cli.base_height) if args_cli.base_height is not None else float(_resolve_default_base_height(env_cfg))

    resample_seconds = (
        float(args_cli.command_resample_seconds)
        if args_cli.command_resample_seconds is not None
        else float(getattr(command_cfg, "resampling_time_range", (10.0, 10.0))[1])
    )

    session = _build_probe_session(prefix="/World/Go2PretrainedAttentionProbe", env_cfg=env_cfg)
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
    session.center_ray_index = int((session.scan_shape[0] // 2) * session.scan_shape[1] + (session.scan_shape[1] // 2))

    model_scan_shape = tuple(int(v) for v in getattr(model, "scan_shape", session.scan_shape))
    scan_shape_matches_checkpoint = tuple(model_scan_shape) == tuple(session.scan_shape)

    num_heads = int(getattr(model, "attention_num_heads", 0))
    if args_cli.attention_head_index >= num_heads and args_cli.attention_head_index >= 0:
        raise ValueError(
            f"--attention-head-index {args_cli.attention_head_index} is invalid for checkpoint with {num_heads} heads."
        )

    for _ in range(2):
        sim.step(render=not bool(args_cli.headless))

    device = torch.device(sim.device)
    dt = 1.0 / 60.0
    run_until_quit = float(args_cli.duration_seconds) <= 0.0
    total_steps = None if run_until_quit else max(1, int(round(float(args_cli.duration_seconds) * 60.0)))
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
        f"[INFO] Go2 profile={args_cli.go2_env_profile} terrain_size={session.terrain_size} "
        f"scan_size={session.scan_size} scan_resolution={session.scan_resolution:.3f} scan_shape={session.scan_shape}",
        flush=True,
    )
    print(
        f"[INFO] Loaded pretrained checkpoint={args_cli.checkpoint} "
        f"attention_heads={num_heads} attention_embed_dim={getattr(model, 'attention_embed_dim', '?')} "
        f"scan_shape={model_scan_shape} metadata_terrain_set={metadata.get('terrain_set', 'unknown')}",
        flush=True,
    )
    if not scan_shape_matches_checkpoint:
        print(
            f"[WARN] Scan-shape mismatch: probe={session.scan_shape} checkpoint={model_scan_shape}. "
            "Resampling height maps for inference and resampling attention maps back for display.",
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
        f"duration={'until quit' if run_until_quit else f'{float(args_cli.duration_seconds):.1f}s'}",
        flush=True,
    )
    print(
        f"[INFO] Motion bounds inside patch: x=[{-half_limit_x:.2f}, {half_limit_x:.2f}] "
        f"y=[{-half_limit_y:.2f}, {half_limit_y:.2f}]",
        flush=True,
    )

    enable_attention_window = (not bool(args_cli.headless)) and (not bool(args_cli.hide_attention_window))
    if enable_attention_window and cv2 is None:
        enable_attention_window = False
        print("[WARN] OpenCV is not installed; disabling attention window.", flush=True)
    if bool(args_cli.headless):
        print("[WARN] Running headless: viewport/window visuals are disabled.", flush=True)

    isaac_attention_overlay = (not bool(args_cli.headless)) and (not bool(args_cli.disable_attention_overlay))
    attention_vis: VisualizationMarkers | None = None
    if isaac_attention_overlay:
        attention_vis = _build_attention_heightmap_visualizer()
        print(
            "[INFO] Isaac 3D attention overlay on (4-bin HOT spheres at ray hits; matches selected-head panel; "
            "--disable-attention-overlay to hide).",
            flush=True,
        )

    show_teacher_column = not bool(args_cli.disable_teacher_target)
    if enable_attention_window and show_teacher_column:
        print(
            "[INFO] OpenCV: height | teacher target (next_steps, half-plane prior) | mean attention | selected; "
            "--disable-teacher-target for three columns only.",
            flush=True,
        )

    step_idx = 0
    while True:
        if (total_steps is not None) and (step_idx >= total_steps):
            break
        if step_idx > 0 and (step_idx % 600 == 0):
            print(f"[INFO] progress step={step_idx:05d}", flush=True)
        try:
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

            model_scan = height_scan.to(attention_device)
            if not scan_shape_matches_checkpoint:
                model_height_map = height_map.to(attention_device).unsqueeze(0).unsqueeze(0)
                model_height_map = F.interpolate(
                    model_height_map,
                    size=model_scan_shape,
                    mode="bilinear",
                    align_corners=True,
                ).squeeze(0).squeeze(0)
                model_scan = model_height_map.reshape(-1)

            with torch.inference_mode():
                _, attention_weights = model(model_scan.unsqueeze(0))
            attention = attention_weights[0].detach().cpu()

            attention_mean_model = attention.mean(dim=0).view(model_scan_shape[0], model_scan_shape[1])
            selected_head = int(args_cli.attention_head_index)
            if selected_head >= 0:
                attention_selected_model = attention[selected_head].view(model_scan_shape[0], model_scan_shape[1])
            else:
                attention_selected_model = attention_mean_model

            if scan_shape_matches_checkpoint:
                attention_mean = attention_mean_model
                attention_selected = attention_selected_model
            else:
                attention_mean = F.interpolate(
                    attention_mean_model.unsqueeze(0).unsqueeze(0),
                    size=session.scan_shape,
                    mode="bilinear",
                    align_corners=True,
                ).squeeze(0).squeeze(0)
                attention_selected = F.interpolate(
                    attention_selected_model.unsqueeze(0).unsqueeze(0),
                    size=session.scan_shape,
                    mode="bilinear",
                    align_corners=True,
                ).squeeze(0).squeeze(0)

            entropy = float(-(attention * attention.clamp_min(1.0e-8).log()).sum(dim=-1).mean().item())
            topk5_mass = float(torch.topk(attention_mean_model.reshape(-1), k=min(5, attention_mean_model.numel())).values.sum().item())

            teacher_map: torch.Tensor | None = None
            if enable_attention_window and show_teacher_column:
                teacher_map = _compute_teacher_target_map(
                    height_scan,
                    command_b,
                    session.ray_caster.data.ray_hits_w,
                    session.scan_shape,
                    float(session.scan_resolution),
                    command_prior_strength=float(args_cli.command_prior_strength),
                )

            if enable_attention_window:
                try:
                    quit_requested = _render_attention_window(
                        height_map=height_map,
                        attention_mean_map=attention_mean,
                        attention_selected_map=attention_selected,
                        command_b=command_b,
                        elapsed_seconds=(step_idx + 1) * dt,
                        selected_head=selected_head,
                        entropy=entropy,
                        topk5_mass=topk5_mass,
                        teacher_map=teacher_map,
                    )
                except SystemExit as exc:
                    # Some OpenCV/Qt stacks can raise SystemExit(0) spuriously; keep running unless user quits explicitly.
                    print(f"[WARN] Ignoring transient SystemExit from attention window: {exc!r}", flush=True)
                    quit_requested = False
                if quit_requested:
                    print("[INFO] Quit requested from OpenCV window.", flush=True)
                    break

            if attention_vis is not None:
                _render_isaac_attention_heightmap_overlay(
                    attention_vis,
                    session.ray_caster.data.ray_hits_w,
                    attention_selected.reshape(-1),
                )

            step_idx += 1
        except SystemExit as exc:
            # Guard against silent toolkit exits (Qt/Isaac integration); user-triggered quits are handled explicitly above.
            print(f"[WARN] Ignoring transient SystemExit in main loop: {exc!r}", flush=True)
            continue

    if cv2 is not None:
        try:
            cv2.destroyWindow(WINDOW_NAME)
        except cv2.error:
            pass


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
