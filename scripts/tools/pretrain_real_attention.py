
"""Offline terrain-only REAL attention pretraining with a kinematic probe."""

from __future__ import annotations

import argparse
import copy
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

try:
    sys.stdout.reconfigure(line_buffering=True)
except Exception:
    pass

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Pretrain REAL terrain attention with a simple scanner probe.")
parser.add_argument(
    "--terrain-set",
    type=str,
    default="both",
    choices=("start_sparse", "real_beam_gap", "both"),
    help="Terrain family to sample.",
)
parser.add_argument(
    "--samples-per-terrain",
    type=int,
    default=4096,
    help="Number of terrain scans to collect per terrain family.",
)
parser.add_argument("--num-probes", type=int, default=256, help="Number of batched probe anchors.")
parser.add_argument(
    "--base-height",
    type=float,
    default=0.33,
    help="Nominal scanner base height above the sampled terrain origin.",
)
parser.add_argument(
    "--base-height-jitter",
    type=float,
    default=0.04,
    help="Uniform base-height jitter applied per sample.",
)
parser.add_argument(
    "--height-scan-offset",
    type=float,
    default=0.5,
    help="Offset used by the existing Go2 height_scan observation.",
)
parser.add_argument("--epochs", type=int, default=50, help="Number of offline optimization epochs.")
parser.add_argument("--batch-size", type=int, default=256, help="Offline optimization batch size.")
parser.add_argument("--learning-rate", type=float, default=1.0e-3, help="Offline optimizer learning rate.")
parser.add_argument(
    "--train-device",
    type=str,
    default=None,
    help="Device for offline optimization. Defaults to the simulation device.",
)
parser.add_argument("--seed", type=int, default=7, help="Random seed.")
parser.add_argument(
    "--output",
    type=str,
    default="artifacts/real_attention/pretrained_attention.pt",
    help="Output checkpoint path.",
)
parser.add_argument(
    "--dataset-output",
    type=str,
    default=None,
    help="Optional output path for the cached offline dataset.",
)
parser.add_argument(
    "--attention-embed-dim",
    type=int,
    default=128,
    help="Embedding dimension used by the terrain attention prior.",
)
parser.add_argument(
    "--attention-num-heads",
    type=int,
    default=4,
    help="Number of attention heads used by the terrain attention prior.",
)
parser.add_argument(
    "--terrain-encoder-hidden-dim",
    type=int,
    default=128,
    help="Hidden width used by the terrain token encoder.",
)
parser.add_argument("--activation", type=str, default="elu", help="Activation used by the pretrainer MLPs.")
parser.add_argument(
    "--disable-positional-encoding",
    action="store_true",
    default=False,
    help="Disable 2D scan-coordinate inputs in the terrain token encoder.",
)
parser.add_argument(
    "--forward-prior",
    action="store_true",
    default=False,
    help="Apply an optional near-forward bias to the traversability target distribution.",
)
parser.add_argument(
    "--log-every-epochs",
    type=int,
    default=5,
    help="How often to print offline optimization progress.",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.samples_per_terrain < 1:
    parser.error("--samples-per-terrain must be at least 1.")
if args_cli.num_probes < 1:
    parser.error("--num-probes must be at least 1.")
if args_cli.base_height < 0.0:
    parser.error("--base-height must be non-negative.")
if args_cli.base_height_jitter < 0.0:
    parser.error("--base-height-jitter must be non-negative.")

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaacsim.core.utils.prims as prim_utils
from isaacsim.core.api.simulation_context import SimulationContext
from isaacsim.core.prims import XFormPrim
from isaacsim.core.utils.viewports import set_camera_view

import isaaclab.sim as sim_utils
from isaaclab.sensors import RayCaster
from isaaclab.terrains import TerrainImporter

import Gurukul.tasks  # noqa: F401
from Gurukul.tasks.manager_based.locomotion.velocity.config.quadruped.unitree_go2.start_terrains_cfg import (
    REAL_BEAM_GAP_TERRAINS_CFG,
    START_SPARSE_TERRAINS_CFG,
)
from Gurukul.tasks.manager_based.locomotion.velocity.pretrain_attention import (
    OfflineTerrainDataset,
    TerrainAttentionPretrainer,
    _resolve_scan_shape,
    build_traversability_targets,
    save_attention_checkpoint,
    save_offline_dataset,
    train_offline_attention,
)
from Gurukul.tasks.manager_based.locomotion.velocity.velocity_env_cfg import MySceneCfg


@dataclass
class TerrainSamplingSession:
    name: str
    terrain_importer: TerrainImporter
    ray_caster: RayCaster
    probe_view: XFormPrim
    terrain_size: tuple[float, float]
    scan_size: tuple[float, float]
    scan_resolution: float
    scan_shape: tuple[int, int] | None = None


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


def _terrain_specs(terrain_set: str) -> list[tuple[str, object]]:
    if terrain_set == "start_sparse":
        return [("start_sparse", START_SPARSE_TERRAINS_CFG)]
    if terrain_set == "real_beam_gap":
        return [("real_beam_gap", REAL_BEAM_GAP_TERRAINS_CFG)]
    return [
        ("start_sparse", START_SPARSE_TERRAINS_CFG),
        ("real_beam_gap", REAL_BEAM_GAP_TERRAINS_CFG),
    ]


def _yaw_to_quat(yaw: torch.Tensor) -> torch.Tensor:
    half_yaw = 0.5 * yaw
    zeros = torch.zeros_like(half_yaw)
    return torch.stack((torch.cos(half_yaw), zeros, zeros, torch.sin(half_yaw)), dim=-1)


def _infer_scan_shape(num_rays: int, scan_size: tuple[float, float], scan_resolution: float) -> tuple[int, int]:
    nominal_rows = max(1, int(round(float(scan_size[0]) / float(scan_resolution))))
    nominal_cols = max(1, int(round(float(scan_size[1]) / float(scan_resolution))))
    for rows, cols in (
        (nominal_rows, nominal_cols),
        (nominal_rows + 1, nominal_cols),
        (nominal_rows, nominal_cols + 1),
        (nominal_rows + 1, nominal_cols + 1),
    ):
        if rows * cols == num_rays:
            return int(rows), int(cols)
    return _resolve_scan_shape(num_rays, (nominal_rows, nominal_cols))


def _make_terrain_cfg(prefix: str, terrain_generator_cfg) -> object:
    terrain_cfg = copy.deepcopy(MySceneCfg.terrain)
    terrain_cfg.prim_path = f"{prefix}/ground"
    terrain_cfg.terrain_generator = copy.deepcopy(terrain_generator_cfg)
    terrain_cfg.max_init_terrain_level = None
    terrain_cfg.debug_vis = False
    if hasattr(terrain_cfg, "num_envs"):
        terrain_cfg.num_envs = 1
    return terrain_cfg


def _make_probe_ray_caster_cfg(prefix: str, mesh_prim_path: str):
    ray_caster_cfg = copy.deepcopy(MySceneCfg.height_scanner)
    ray_caster_cfg.prim_path = f"{prefix}/envs/env_.*/probe"
    ray_caster_cfg.mesh_prim_paths = [mesh_prim_path]
    ray_caster_cfg.debug_vis = False
    return ray_caster_cfg


def _build_sampling_session(
    prefix: str,
    name: str,
    terrain_generator_cfg,
    num_probes: int,
) -> TerrainSamplingSession:
    terrain_cfg = _make_terrain_cfg(prefix, terrain_generator_cfg)
    terrain_importer = TerrainImporter(terrain_cfg)

    prim_utils.create_prim(f"{prefix}/envs", "Xform", translation=(0.0, 0.0, 0.0))
    for probe_idx in range(int(num_probes)):
        env_path = f"{prefix}/envs/env_{probe_idx}"
        prim_utils.create_prim(env_path, "Xform", translation=(0.0, 0.0, 0.0))
        prim_utils.create_prim(f"{env_path}/probe", "Xform", translation=(0.0, 0.0, 0.0))

    ray_caster_cfg = _make_probe_ray_caster_cfg(prefix, terrain_cfg.prim_path)
    ray_caster = RayCaster(ray_caster_cfg)
    probe_view = XFormPrim(ray_caster_cfg.prim_path, reset_xform_properties=False)

    pattern_cfg = ray_caster_cfg.pattern_cfg
    scan_resolution = float(getattr(pattern_cfg, "resolution", 0.1))
    scan_size = getattr(pattern_cfg, "size", (1.6, 1.0))
    scan_size = (float(scan_size[0]), float(scan_size[1]))

    return TerrainSamplingSession(
        name=name,
        terrain_importer=terrain_importer,
        ray_caster=ray_caster,
        probe_view=probe_view,
        terrain_size=(float(terrain_generator_cfg.size[0]), float(terrain_generator_cfg.size[1])),
        scan_size=scan_size,
        scan_resolution=scan_resolution,
    )


def _initialize_session(session: TerrainSamplingSession) -> None:
    session.probe_view.initialize()
    session.ray_caster.reset()
    session.scan_shape = _infer_scan_shape(
        session.ray_caster.num_rays,
        scan_size=session.scan_size,
        scan_resolution=session.scan_resolution,
    )


def _sample_probe_batch(
    session: TerrainSamplingSession,
    batch_size: int,
    base_height: float,
    base_height_jitter: float,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if session.terrain_importer.terrain_origins is None:
        raise RuntimeError(f"Terrain session '{session.name}' does not expose terrain origins.")

    terrain_origins = session.terrain_importer.terrain_origins.reshape(-1, 3)
    patch_indices = torch.randint(0, terrain_origins.shape[0], (batch_size,), device=device)
    patch_origins = terrain_origins[patch_indices].clone()

    margin_x = 0.5 * session.scan_size[0] + session.scan_resolution
    margin_y = 0.5 * session.scan_size[1] + session.scan_resolution
    sample_half_x = max(0.05, 0.5 * session.terrain_size[0] - margin_x)
    sample_half_y = max(0.05, 0.5 * session.terrain_size[1] - margin_y)

    dx = (2.0 * torch.rand(batch_size, device=device) - 1.0) * sample_half_x
    dy = (2.0 * torch.rand(batch_size, device=device) - 1.0) * sample_half_y
    yaw = (2.0 * torch.rand(batch_size, device=device) - 1.0) * math.pi
    base_heights = float(base_height) + (2.0 * torch.rand(batch_size, device=device) - 1.0) * float(base_height_jitter)

    patch_origins[:, 0] += dx
    patch_origins[:, 1] += dy
    patch_origins[:, 2] += base_heights
    return patch_origins, _yaw_to_quat(yaw), base_heights


def _collect_terrain_scans(
    sim: SimulationContext,
    session: TerrainSamplingSession,
    samples_per_terrain: int,
    base_height: float,
    base_height_jitter: float,
    height_scan_offset: float,
    headless: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    batch_size = int(session.probe_view.count)
    total_batches = int(math.ceil(float(samples_per_terrain) / float(batch_size)))
    collected_scans: list[torch.Tensor] = []
    collected_heights: list[torch.Tensor] = []

    print(
        f"[INFO] Collecting terrain-only scans for {session.name}: "
        f"samples={samples_per_terrain} probes={batch_size} scan_shape={session.scan_shape}",
        flush=True,
    )

    for batch_idx in range(total_batches):
        positions, quats, base_heights = _sample_probe_batch(
            session,
            batch_size=batch_size,
            base_height=base_height,
            base_height_jitter=base_height_jitter,
            device=torch.device(sim.device),
        )
        session.probe_view.set_world_poses(positions, quats)
        sim.step(render=not headless)
        session.ray_caster.update(dt=sim.get_physics_dt(), force_recompute=True)

        ray_hits_z = torch.nan_to_num(
            session.ray_caster.data.ray_hits_w[..., 2],
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        )
        sensor_z = session.ray_caster.data.pos_w[:, 2].unsqueeze(1)
        height_scans = sensor_z - ray_hits_z - float(height_scan_offset)
        collected_scans.append(height_scans.detach().cpu())
        collected_heights.append(base_heights.detach().cpu())

        done = min((batch_idx + 1) * batch_size, samples_per_terrain)
        if (batch_idx + 1) == total_batches or (batch_idx + 1) % max(1, total_batches // 4) == 0:
            print(
                f"[INFO] [{session.name}] collected {done}/{samples_per_terrain} scans "
                f"({batch_idx + 1}/{total_batches} batches)",
                flush=True,
            )

    terrain_scans = torch.cat(collected_scans, dim=0)[:samples_per_terrain]
    base_heights = torch.cat(collected_heights, dim=0)[:samples_per_terrain]
    print(
        f"[INFO] Finished {session.name}: scan_shape={session.scan_shape} "
        f"base_height_range=[{float(base_heights.min()):.6f}, {float(base_heights.max()):.6f}]",
        flush=True,
    )
    return terrain_scans, base_heights


def main() -> None:
    torch.manual_seed(int(args_cli.seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(args_cli.seed))

    sim_device = args_cli.device if args_cli.device is not None else ("cuda:0" if torch.cuda.is_available() else "cpu")
    train_device = args_cli.train_device or sim_device
    sim = _make_simulation_context(device=sim_device, headless=bool(args_cli.headless))

    light_cfg = sim_utils.DomeLightCfg(intensity=1500.0, color=(0.9, 0.9, 0.9))
    light_cfg.func("/World/REALAttentionLight", light_cfg)

    if not args_cli.headless:
        set_camera_view([35.0, 0.0, 25.0], [0.0, 0.0, 0.0])

    terrain_specs = _terrain_specs(args_cli.terrain_set)
    sessions = [
        _build_sampling_session(
            prefix=f"/World/REALAttention/{terrain_name}",
            name=terrain_name,
            terrain_generator_cfg=terrain_cfg,
            num_probes=int(args_cli.num_probes),
        )
        for terrain_name, terrain_cfg in terrain_specs
    ]

    sim.reset()
    for session in sessions:
        _initialize_session(session)
    for _ in range(3):
        sim.step(render=not args_cli.headless)

    all_height_scans: list[torch.Tensor] = []
    all_base_heights: list[torch.Tensor] = []
    scan_shape: tuple[int, int] | None = None
    scan_resolution: float | None = None

    for session in sessions:
        terrain_scans, base_heights = _collect_terrain_scans(
            sim=sim,
            session=session,
            samples_per_terrain=int(args_cli.samples_per_terrain),
            base_height=float(args_cli.base_height),
            base_height_jitter=float(args_cli.base_height_jitter),
            height_scan_offset=float(args_cli.height_scan_offset),
            headless=bool(args_cli.headless),
        )
        all_height_scans.append(terrain_scans)
        all_base_heights.append(base_heights)
        if scan_shape is None:
            scan_shape = session.scan_shape
            scan_resolution = session.scan_resolution
        elif scan_shape != session.scan_shape:
            raise RuntimeError(
                f"Mixed scan shapes are not supported: {scan_shape} vs {session.scan_shape}."
            )

    if scan_shape is None or scan_resolution is None:
        raise RuntimeError("No terrain sessions were initialized.")

    height_scans = torch.cat(all_height_scans, dim=0)
    base_heights = torch.cat(all_base_heights, dim=0)
    traversability_targets = build_traversability_targets(
        height_scans,
        scan_shape=scan_shape,
        raster_resolution=float(scan_resolution),
        raster_resolution_y=float(scan_resolution),
        apply_forward_prior=bool(args_cli.forward_prior),
    )

    dataset = OfflineTerrainDataset(
        height_scans=height_scans,
        traversability_targets=traversability_targets,
        scan_shape=scan_shape,
        terrain_set=str(args_cli.terrain_set),
        raster_resolution=float(scan_resolution),
        base_heights=base_heights,
        metadata={
            "terrain_names": [session.name for session in sessions],
            "samples_per_terrain": int(args_cli.samples_per_terrain),
            "num_probes": int(args_cli.num_probes),
            "base_height": float(args_cli.base_height),
            "base_height_jitter": float(args_cli.base_height_jitter),
            "height_scan_offset": float(args_cli.height_scan_offset),
            "scan_resolution_y": float(scan_resolution),
            "scan_size": sessions[0].scan_size,
            "forward_prior": bool(args_cli.forward_prior),
            "sampler": "kinematic_probe",
        },
    )

    if args_cli.dataset_output:
        dataset_path = save_offline_dataset(dataset, args_cli.dataset_output)
        print(f"[INFO] Saved terrain-only dataset to {dataset_path}", flush=True)
    else:
        dataset_path = None

    model = TerrainAttentionPretrainer(
        scan_shape=scan_shape,
        attention_embed_dim=int(args_cli.attention_embed_dim),
        attention_num_heads=int(args_cli.attention_num_heads),
        terrain_encoder_hidden_dim=int(args_cli.terrain_encoder_hidden_dim),
        activation=str(args_cli.activation),
        use_scan_positional_encoding=not bool(args_cli.disable_positional_encoding),
    )
    print(
        f"[INFO] Training REAL terrain attention: samples={int(height_scans.shape[0])} "
        f"epochs={int(args_cli.epochs)} batch_size={int(args_cli.batch_size)}",
        flush=True,
    )
    history = train_offline_attention(
        model=model,
        dataset=dataset,
        epochs=int(args_cli.epochs),
        batch_size=int(args_cli.batch_size),
        learning_rate=float(args_cli.learning_rate),
        device=str(train_device),
        log_prefix="REAL attention pretrain",
        log_every_epochs=int(args_cli.log_every_epochs),
    )

    print("[INFO] Saving pretrained REAL attention checkpoint...", flush=True)
    checkpoint_path = save_attention_checkpoint(
        model=model,
        output_path=args_cli.output,
        dataset=dataset,
        history=history,
        extra_metadata={"train_device": str(train_device)},
    )

    final_stats = history[-1] if history else {"loss": float("nan"), "alignment": float("nan"), "entropy": float("nan")}
    print(
        json.dumps(
            {
                "checkpoint_path": str(checkpoint_path),
                "dataset_path": None if dataset_path is None else str(dataset_path),
                "num_samples": int(height_scans.shape[0]),
                "scan_shape": list(scan_shape),
                "terrain_set": str(args_cli.terrain_set),
                "terrain_names": [session.name for session in sessions],
                "base_height_range": [float(base_heights.min()), float(base_heights.max())],
                "base_height_mean": float(base_heights.mean()),
                "base_height_std": float(base_heights.std(unbiased=False)),
                "final_loss": float(final_stats["loss"]),
                "final_alignment": float(final_stats["alignment"]),
                "final_entropy": float(final_stats["entropy"]),
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    try:
        main()
    finally:
        simulation_app.close()
