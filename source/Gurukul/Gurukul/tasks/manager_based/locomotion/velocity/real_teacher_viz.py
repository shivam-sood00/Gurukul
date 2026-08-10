"""Visualization helpers for the REAL teacher attention module."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None


DEFAULT_PROPRIO_DIM = 48
DEFAULT_PRIVILEGED_DIM = 50
DEFAULT_TERRAIN_SHAPE = (16, 10)
DEFAULT_HEAD_COLORS_BGR: tuple[tuple[int, int, int], ...] = (
    (235, 115, 55),
    (215, 200, 60),
    (115, 210, 95),
    (60, 210, 245),
    (120, 135, 255),
    (205, 110, 245),
)
DEFAULT_BASE_POINT_COLOR_BGR = (225, 105, 45)


def _normalize_image_01(image: torch.Tensor) -> torch.Tensor:
    image = torch.nan_to_num(image.float(), nan=0.0, posinf=0.0, neginf=0.0)
    image_min = image.min()
    image_max = image.max()
    return (image - image_min) / torch.clamp(image_max - image_min, min=1.0e-6)


def _default_scan_coordinates(rows: int, cols: int) -> torch.Tensor:
    x_coords = torch.linspace(-1.0, 1.0, rows)
    y_coords = torch.linspace(-1.0, 1.0, cols)
    grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing="ij")
    return torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)


def _draw_filled_circle(image: np.ndarray, center: tuple[int, int], radius: int, color: tuple[int, int, int]) -> None:
    if cv2 is not None:
        cv2.circle(image, center, radius, color, thickness=-1, lineType=cv2.LINE_AA)
        return
    center_x, center_y = center
    height, width = image.shape[:2]
    x0 = max(center_x - radius, 0)
    x1 = min(center_x + radius + 1, width)
    y0 = max(center_y - radius, 0)
    y1 = min(center_y + radius + 1, height)
    yy, xx = np.ogrid[y0:y1, x0:x1]
    mask = (xx - center_x) ** 2 + (yy - center_y) ** 2 <= radius**2
    image[y0:y1, x0:x1][mask] = np.asarray(color, dtype=np.uint8)


def resolve_scan_shape(num_scan_values: int, scan_shape: tuple[int, int] | None = None) -> tuple[int, int]:
    """Infer a 2D terrain scan shape from a flat scan vector."""
    if scan_shape is not None:
        rows, cols = int(scan_shape[0]), int(scan_shape[1])
        if rows > 0 and cols > 0 and rows * cols == num_scan_values:
            return rows, cols
    best_rows, best_cols = num_scan_values, 1
    best_gap = num_scan_values
    for divisor in range(1, int(num_scan_values**0.5) + 1):
        if num_scan_values % divisor != 0:
            continue
        rows = num_scan_values // divisor
        cols = divisor
        gap = abs(rows - cols)
        if gap < best_gap:
            best_rows, best_cols = rows, cols
            best_gap = gap
    return int(best_rows), int(best_cols)



def reshape_attention_map(
    attention_map: torch.Tensor,
    scan_shape: tuple[int, int] | None = None,
) -> torch.Tensor:
    """Convert flat per-head attention vectors into a 2D head-grid representation."""
    attention_map = torch.as_tensor(attention_map, dtype=torch.float32).detach().cpu()
    if attention_map.ndim == 3:
        return attention_map
    if attention_map.ndim != 2:
        raise ValueError(
            f"Expected attention_map with shape [heads, rows, cols] or [heads, tokens], got {tuple(attention_map.shape)}."
        )
    rows, cols = resolve_scan_shape(int(attention_map.shape[-1]), scan_shape=scan_shape)
    return attention_map.reshape(int(attention_map.shape[0]), rows, cols)



def _flatten_attention_heads(
    attention_map: torch.Tensor,
    scan_shape: tuple[int, int] | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    attention_map = reshape_attention_map(attention_map, scan_shape=scan_shape)
    num_heads = int(attention_map.shape[0])
    flat_attention = attention_map.reshape(num_heads, -1)
    flat_attention = torch.nan_to_num(flat_attention, nan=0.0, posinf=0.0, neginf=0.0)
    head_norm = flat_attention / torch.clamp(flat_attention.amax(dim=1, keepdim=True), min=1.0e-6)
    point_strength = torch.clamp(head_norm.amax(dim=0), 0.0, 1.0)
    dominant_head = torch.argmax(head_norm, dim=0)
    return attention_map, head_norm, point_strength, dominant_head


def compute_attention_marker_overlay(
    attention_map: torch.Tensor,
    scan_shape: tuple[int, int] | None = None,
    min_scale: float = 0.55,
    max_scale: float = 1.45,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert attention heads into dominant-head marker indices and per-point scales."""
    _, _, point_strength, dominant_head = _flatten_attention_heads(attention_map, scan_shape=scan_shape)
    point_scale = float(min_scale) + (float(max_scale) - float(min_scale)) * point_strength.pow(0.7)
    marker_scales = point_scale.unsqueeze(-1).repeat(1, 3)
    return dominant_head.to(torch.int64), marker_scales, point_strength


def render_attention_point_overlay(
    terrain_map: torch.Tensor,
    attention_map: torch.Tensor,
    scan_coordinates: torch.Tensor | None = None,
    cell_size: int = 28,
    side_padding: int = 48,
    top_padding: int = 52,
    point_radius: int = 4,
) -> np.ndarray:
    """Render a point-style top-down terrain overlay with per-head attention colors."""
    terrain_map = torch.as_tensor(terrain_map, dtype=torch.float32).detach().cpu()
    if terrain_map.ndim != 2:
        raise ValueError(f"Expected terrain_map with shape [rows, cols], got {tuple(terrain_map.shape)}.")

    rows, cols = int(terrain_map.shape[0]), int(terrain_map.shape[1])
    attention_map, head_norm, point_strength, _ = _flatten_attention_heads(attention_map, scan_shape=(rows, cols))
    if tuple(attention_map.shape[1:]) != (rows, cols):
        raise ValueError(
            f"Attention shape {tuple(attention_map.shape[1:])} does not match terrain shape {(rows, cols)}."
        )

    content_height = max(220, int((rows - 1) * cell_size + 1))
    content_width = max(160, int((cols - 1) * cell_size + 1))
    canvas_height = content_height + top_padding + side_padding
    canvas_width = content_width + 2 * side_padding

    terrain_image = F.interpolate(
        terrain_map.unsqueeze(0).unsqueeze(0),
        size=(content_height, content_width),
        mode="bilinear",
        align_corners=True,
    ).squeeze(0).squeeze(0)
    terrain_image = _normalize_image_01(terrain_image)
    terrain_gray = (55.0 + 175.0 * terrain_image).to(torch.uint8).numpy()
    terrain_bgr = np.repeat(terrain_gray[..., None], 3, axis=2)

    canvas = np.full((canvas_height, canvas_width, 3), 28, dtype=np.uint8)
    y0 = top_padding
    x0 = side_padding
    canvas[y0 : y0 + content_height, x0 : x0 + content_width] = terrain_bgr

    if cv2 is not None:
        cv2.rectangle(canvas, (x0 - 2, y0 - 2), (x0 + content_width + 1, y0 + content_height + 1), (65, 65, 65), 1)
        cv2.putText(
            canvas,
            "Rough Elevation + Attention",
            (14, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )

    if scan_coordinates is None:
        scan_coordinates = _default_scan_coordinates(rows, cols)
    scan_coordinates = torch.as_tensor(scan_coordinates, dtype=torch.float32).detach().cpu().reshape(rows * cols, 2)

    num_heads = int(attention_map.shape[0])
    head_colors = torch.tensor(
        [DEFAULT_HEAD_COLORS_BGR[idx % len(DEFAULT_HEAD_COLORS_BGR)] for idx in range(num_heads)],
        dtype=torch.float32,
    )
    base_color = torch.tensor(DEFAULT_BASE_POINT_COLOR_BGR, dtype=torch.float32).view(1, 3)

    point_strength = point_strength.pow(1.75)
    point_mix = torch.matmul(head_norm.transpose(0, 1), head_colors)
    point_mix = point_mix / torch.clamp(head_norm.sum(dim=0, keepdim=True).transpose(0, 1), min=1.0e-6)
    point_colors = base_color * (1.0 - point_strength.unsqueeze(-1)) + point_mix * point_strength.unsqueeze(-1)
    point_colors = torch.clamp(point_colors + 22.0 * point_strength.unsqueeze(-1), 0.0, 255.0).to(torch.uint8).numpy()
    point_radii = (point_radius + torch.round(2.5 * point_strength)).to(torch.int64).numpy()
    point_strength_np = point_strength.numpy()

    for point_idx in range(rows * cols):
        scan_x = float(scan_coordinates[point_idx, 0].item())
        scan_y = float(scan_coordinates[point_idx, 1].item())
        center_x = int(round(x0 + ((scan_y + 1.0) * 0.5) * (content_width - 1)))
        center_y = int(round(y0 + (1.0 - (scan_x + 1.0) * 0.5) * (content_height - 1)))
        radius = int(max(2, point_radii[point_idx]))
        outline_radius = radius + 1
        _draw_filled_circle(canvas, (center_x, center_y), outline_radius, (22, 22, 22))
        _draw_filled_circle(canvas, (center_x, center_y), radius, tuple(int(channel) for channel in point_colors[point_idx]))
        if point_strength_np[point_idx] > 0.72:
            _draw_filled_circle(canvas, (center_x, center_y), max(1, radius - 2), (245, 245, 245))

    if cv2 is not None:
        legend_x = 14
        legend_y = 38
        for head_idx in range(num_heads):
            color = tuple(int(channel) for channel in head_colors[head_idx].tolist())
            cv2.circle(canvas, (legend_x, legend_y), 5, color, thickness=-1, lineType=cv2.LINE_AA)
            cv2.putText(
                canvas,
                f"H{head_idx}",
                (legend_x + 10, legend_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.42,
                (225, 225, 225),
                1,
                cv2.LINE_AA,
            )
            legend_x += 46

    return canvas


def generate_synthetic_terrain_patterns(rows: int, cols: int) -> dict[str, torch.Tensor]:
    """Create a small set of interpretable terrain patterns for attention inspection."""
    flat = torch.zeros(rows, cols)

    step = torch.zeros(rows, cols)
    step[rows // 2 :, :] = 0.25

    gap = torch.zeros(rows, cols)
    gap[max(0, rows // 2 - 1) : min(rows, rows // 2 + 1), :] = -0.35

    ridge = torch.zeros(rows, cols)
    ridge[:, cols // 2] = 0.3
    if cols > 2:
        ridge[:, max(0, cols // 2 - 1)] = 0.15
        ridge[:, min(cols - 1, cols // 2 + 1)] = 0.15

    checker = torch.zeros(rows, cols)
    checker[::2, ::2] = 0.2
    checker[1::2, 1::2] = -0.15

    slope = torch.linspace(-0.25, 0.25, rows).unsqueeze(1).repeat(1, cols)

    return {
        "flat": flat,
        "step_up": step,
        "gap": gap,
        "ridge": ridge,
        "checker": checker,
        "slope": slope,
    }


def build_demo_observations(
    patterns: dict[str, torch.Tensor],
    proprio_dim: int = DEFAULT_PROPRIO_DIM,
    privileged_dim: int = DEFAULT_PRIVILEGED_DIM,
    device: str | torch.device = "cpu",
) -> dict[str, torch.Tensor]:
    """Pack synthetic terrain patterns into the observation groups expected by the REAL teacher."""
    names = list(patterns.keys())
    terrain = torch.stack([patterns[name].reshape(-1) for name in names], dim=0).to(device=device, dtype=torch.float32)
    proprio = torch.zeros((len(names), int(proprio_dim)), device=device, dtype=torch.float32)
    privileged = torch.zeros((len(names), int(privileged_dim)), device=device, dtype=torch.float32)

    if proprio.shape[-1] >= 6:
        proprio[:, 0] = torch.linspace(0.2, 0.8, len(names), device=device)
        proprio[:, 1] = torch.linspace(-0.1, 0.1, len(names), device=device)
        proprio[:, 5] = torch.linspace(-0.4, 0.4, len(names), device=device)
    if proprio.shape[-1] >= 21:
        proprio[:, 18] = torch.linspace(0.4, 0.9, len(names), device=device)
        proprio[:, 20] = torch.linspace(-0.6, 0.6, len(names), device=device)
    if privileged.shape[-1] >= 4:
        privileged[:, 0] = torch.linspace(0.0, 0.3, len(names), device=device)
        privileged[:, 3] = 1.0

    return {
        "real_teacher_proprio": proprio,
        "real_teacher_terrain": terrain,
        "real_teacher_privileged": privileged,
    }


def save_attention_visualizations(
    policy: Any,
    obs: dict[str, torch.Tensor],
    output_dir: str | Path,
    sample_names: list[str] | None = None,
) -> list[Path]:
    """Render one attention figure per sample and return the saved file paths."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    with torch.no_grad():
        debug_result = policy.forward_actor_debug(obs)

    rows, cols = debug_result.terrain_scan_shape
    terrain_maps = debug_result.terrain_scan.detach().cpu().reshape(-1, rows, cols)
    attention_maps = debug_result.attention_weights.detach().cpu().reshape(debug_result.attention_weights.shape[0], -1, rows, cols)
    action_means = debug_result.action_mean.detach().cpu()

    if sample_names is None:
        sample_names = [f"sample_{idx:02d}" for idx in range(terrain_maps.shape[0])]

    saved_paths: list[Path] = []
    for sample_idx, sample_name in enumerate(sample_names):
        terrain_map = terrain_maps[sample_idx]
        per_head_attention = attention_maps[sample_idx]
        avg_attention = per_head_attention.mean(dim=0)
        point_overlay = render_attention_point_overlay(terrain_map, per_head_attention)
        num_heads = per_head_attention.shape[0]
        figure_cols = 3 + num_heads
        fig, axes = plt.subplots(1, figure_cols, figsize=(4.0 * figure_cols, 4.0), squeeze=False)
        axes = axes[0]

        terrain_im = axes[0].imshow(terrain_map, cmap="terrain", origin="lower")
        axes[0].set_title(f"Terrain: {sample_name}")
        axes[0].set_xlabel("y")
        axes[0].set_ylabel("x")
        fig.colorbar(terrain_im, ax=axes[0], fraction=0.046, pad=0.04)

        axes[1].imshow(point_overlay[..., ::-1], origin="upper")
        axes[1].set_title("Point Overlay")
        axes[1].axis("off")

        avg_im = axes[2].imshow(avg_attention, cmap="magma", origin="lower")
        axes[2].set_title("Attention Mean")
        axes[2].set_xlabel("y")
        axes[2].set_ylabel("x")
        fig.colorbar(avg_im, ax=axes[2], fraction=0.046, pad=0.04)

        for head_idx in range(num_heads):
            head_im = axes[3 + head_idx].imshow(per_head_attention[head_idx], cmap="magma", origin="lower")
            axes[3 + head_idx].set_title(f"Head {head_idx}")
            axes[3 + head_idx].set_xlabel("y")
            axes[3 + head_idx].set_ylabel("x")
            fig.colorbar(head_im, ax=axes[3 + head_idx], fraction=0.046, pad=0.04)

        fig.suptitle(
            f"REAL Teacher Attention | action_mean_norm={float(action_means[sample_idx].norm()):.3f}",
            fontsize=12,
        )
        fig.tight_layout()
        file_path = output_path / f"{sample_name}_attention.png"
        fig.savefig(file_path, dpi=160, bbox_inches="tight")
        plt.close(fig)
        saved_paths.append(file_path)

    return saved_paths
