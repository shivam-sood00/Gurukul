"""Validate realistic Mid360 lidar tensor shapes and throughput across environment counts."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
MID360_PATTERN_PATH = (
    REPO_ROOT
    / "source"
    / "Gurukul"
    / "Gurukul"
    / "assets"
    / "lidar_patterns"
    / "livox"
    / "mid360.npy"
)


parser = argparse.ArgumentParser(
    description=(
        "Validate the tensor path used by the OmniPerception-style Mid360 lidar integration. "
        "This uses the repo-local realistic Livox Mid360 scan pattern, applies the configured "
        "distance noise/dropout model, materializes distances and pointcloud tensors, and reports "
        "shape stability and throughput for cloned environment counts."
    )
)
parser.add_argument(
    "--num-envs",
    nargs="+",
    type=int,
    default=(1, 16, 256, 4096),
    help="Environment counts to validate.",
)
parser.add_argument("--num-rays", type=int, default=20000, help="Number of Mid360 rays per environment.")
parser.add_argument("--warmup-steps", type=int, default=1, help="Warmup iterations before timing.")
parser.add_argument("--measure-steps", type=int, default=5, help="Measured iterations for each environment count.")
parser.add_argument("--max-range", type=float, default=30.0, help="Maximum lidar range in meters.")
parser.add_argument("--min-range", type=float, default=0.2, help="Minimum lidar range in meters.")
parser.add_argument("--noise-std", type=float, default=0.03, help="Distance noise standard deviation in meters.")
parser.add_argument("--dropout-prob", type=float, default=0.01, help="Per-ray dropout probability.")
parser.add_argument(
    "--device",
    default="cuda:0" if torch.cuda.is_available() else "cpu",
    help="Torch device for the validation.",
)
parser.add_argument("--json", action="store_true", help="Print machine-readable JSON.")
args_cli = parser.parse_args()


@dataclass(frozen=True)
class LidarTensorValidationReport:
    num_envs: int
    num_rays: int
    distances_shape: tuple[int, ...]
    pointcloud_shape: tuple[int, ...]
    finite_distance_ratio: float
    finite_pointcloud_ratio: float
    min_distance: float
    max_distance: float
    warmup_steps: int
    measure_steps: int
    average_step_ms: float
    steps_per_second: float
    million_rays_per_second: float
    sensor_noise_enabled: bool
    device: str
    pattern_path: str


def _sync_if_cuda(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _finite_ratio(tensor: torch.Tensor) -> float:
    return float(torch.isfinite(tensor).float().mean().item())


def _load_mid360_directions(device: torch.device, num_rays: int) -> torch.Tensor:
    if not MID360_PATTERN_PATH.exists():
        raise FileNotFoundError(f"Mid360 scan pattern not found: {MID360_PATTERN_PATH}")
    pattern = np.load(MID360_PATTERN_PATH)
    if pattern.shape[0] < num_rays:
        raise ValueError(f"Requested {num_rays} rays, but pattern only has {pattern.shape[0]}.")

    theta = torch.as_tensor(pattern[:num_rays, 0], dtype=torch.float32, device=device)
    phi = torch.as_tensor(pattern[:num_rays, 1], dtype=torch.float32, device=device)
    cos_theta = torch.cos(theta)
    sin_theta = torch.sin(theta)
    cos_phi = torch.cos(phi)
    sin_phi = torch.sin(phi)
    directions = torch.stack((cos_theta * cos_phi, sin_theta * cos_phi, sin_phi), dim=-1)
    return directions / torch.norm(directions, dim=-1, keepdim=True).clamp_min(1.0e-8)


def _materialize_lidar_tensors(
    directions: torch.Tensor,
    num_envs: int,
    *,
    min_range: float,
    max_range: float,
    noise_std: float,
    dropout_prob: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    num_rays = directions.shape[0]
    base_distances = torch.linspace(min_range, max_range, steps=num_rays, device=directions.device)
    distances = base_distances.unsqueeze(0).repeat(num_envs, 1)
    if noise_std > 0.0:
        distances = distances + torch.randn_like(distances) * float(noise_std)
    if dropout_prob > 0.0:
        dropout = torch.rand_like(distances) < float(dropout_prob)
        distances = torch.where(dropout, torch.full_like(distances, float(max_range)), distances)
    distances = distances.clamp(float(min_range), float(max_range))
    pointcloud = directions.unsqueeze(0) * distances.unsqueeze(-1)
    return distances, pointcloud


def _validate_shapes(num_envs: int, num_rays: int, distances: torch.Tensor, pointcloud: torch.Tensor) -> None:
    expected_distances = (num_envs, num_rays)
    expected_pointcloud = (num_envs, num_rays, 3)
    if tuple(distances.shape) != expected_distances:
        raise RuntimeError(f"Expected distances shape {expected_distances}, got {tuple(distances.shape)}.")
    if tuple(pointcloud.shape) != expected_pointcloud:
        raise RuntimeError(f"Expected pointcloud shape {expected_pointcloud}, got {tuple(pointcloud.shape)}.")


def _run_one(num_envs: int, directions: torch.Tensor, device: torch.device) -> LidarTensorValidationReport:
    num_rays = int(directions.shape[0])
    warmup_steps = max(0, int(args_cli.warmup_steps))
    measure_steps = max(1, int(args_cli.measure_steps))

    print(f"[INFO] Validating Mid360 tensors: num_envs={num_envs} num_rays={num_rays} device={device}", flush=True)
    for _ in range(warmup_steps):
        distances, pointcloud = _materialize_lidar_tensors(
            directions,
            num_envs,
            min_range=args_cli.min_range,
            max_range=args_cli.max_range,
            noise_std=args_cli.noise_std,
            dropout_prob=args_cli.dropout_prob,
        )
        _validate_shapes(num_envs, num_rays, distances, pointcloud)
        del distances, pointcloud

    _sync_if_cuda(device)
    start_time = time.perf_counter()
    for _ in range(measure_steps):
        distances, pointcloud = _materialize_lidar_tensors(
            directions,
            num_envs,
            min_range=args_cli.min_range,
            max_range=args_cli.max_range,
            noise_std=args_cli.noise_std,
            dropout_prob=args_cli.dropout_prob,
        )
        _validate_shapes(num_envs, num_rays, distances, pointcloud)
    _sync_if_cuda(device)
    elapsed = time.perf_counter() - start_time

    finite_distances = distances[torch.isfinite(distances)]
    average_step_ms = 1000.0 * elapsed / measure_steps
    report = LidarTensorValidationReport(
        num_envs=num_envs,
        num_rays=num_rays,
        distances_shape=tuple(int(dim) for dim in distances.shape),
        pointcloud_shape=tuple(int(dim) for dim in pointcloud.shape),
        finite_distance_ratio=_finite_ratio(distances),
        finite_pointcloud_ratio=_finite_ratio(pointcloud),
        min_distance=float(finite_distances.min().item()) if finite_distances.numel() else float("nan"),
        max_distance=float(finite_distances.max().item()) if finite_distances.numel() else float("nan"),
        warmup_steps=warmup_steps,
        measure_steps=measure_steps,
        average_step_ms=average_step_ms,
        steps_per_second=measure_steps / elapsed if elapsed > 0.0 else float("inf"),
        million_rays_per_second=(measure_steps * num_envs * num_rays) / max(elapsed, 1.0e-12) / 1.0e6,
        sensor_noise_enabled=args_cli.noise_std > 0.0 or args_cli.dropout_prob > 0.0,
        device=str(device),
        pattern_path=str(MID360_PATTERN_PATH),
    )
    del distances, pointcloud
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return report


def _print_text(reports: list[LidarTensorValidationReport]) -> None:
    print("OmniPerception-style Mid360 tensor validation", flush=True)
    print("=" * 48, flush=True)
    for report in reports:
        print(
            f"[OK] num_envs={report.num_envs} rays={report.num_rays} "
            f"distances={report.distances_shape} pointcloud={report.pointcloud_shape} "
            f"noise={report.sensor_noise_enabled} avg_step={report.average_step_ms:.2f}ms "
            f"throughput={report.million_rays_per_second:.2f} Mrays/s "
            f"finite_dist={report.finite_distance_ratio:.3f} "
            f"range=({report.min_distance:.3f}, {report.max_distance:.3f})",
            flush=True,
        )


def main() -> int:
    if any(num_envs <= 0 for num_envs in args_cli.num_envs):
        raise ValueError("--num-envs values must be positive.")
    if args_cli.num_rays <= 0:
        raise ValueError("--num-rays must be positive.")
    if args_cli.warmup_steps < 0:
        raise ValueError("--warmup-steps must be >= 0.")
    if args_cli.measure_steps <= 0:
        raise ValueError("--measure-steps must be > 0.")
    if args_cli.min_range <= 0.0 or args_cli.max_range <= args_cli.min_range:
        raise ValueError("--max-range must be greater than --min-range, and both must be positive.")
    if args_cli.noise_std < 0.0:
        raise ValueError("--noise-std must be >= 0.")
    if not 0.0 <= args_cli.dropout_prob <= 1.0:
        raise ValueError("--dropout-prob must be in [0, 1].")

    device = torch.device(args_cli.device)
    directions = _load_mid360_directions(device, int(args_cli.num_rays))
    reports = [_run_one(int(num_envs), directions, device) for num_envs in args_cli.num_envs]
    if args_cli.json:
        print(json.dumps([asdict(report) for report in reports], indent=2), flush=True)
    else:
        _print_text(reports)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
