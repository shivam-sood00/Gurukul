#!/usr/bin/env python3
"""Compare APEX tracker actor-observation/action delta logs from Isaac and sim2sim."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


SLICE_NAMES = (
    "base_ang_vel",
    "projected_gravity",
    "command",
    "joint_pos",
    "joint_vel",
    "prev_action",
    "reference_joint_pos",
    "reference_base_lin_vel",
    "reference_base_ang_vel",
    "reference_quat",
    "action",
)


def _format_float(value: float) -> str:
    return f"{value:.4g}"


def _load(path: Path) -> dict[str, np.ndarray]:
    if not path.is_file():
        raise FileNotFoundError(path)
    data = np.load(path)
    return {key: data[key] for key in data.files}


def _max_delta(data: dict[str, np.ndarray], name: str, count: int | None) -> float:
    values = data[f"{name}_max_abs_delta"]
    if count is not None:
        values = values[:count]
    if values.size <= 1:
        return 0.0
    return float(np.max(values[1:]))


def _p99_delta(data: dict[str, np.ndarray], name: str, count: int | None) -> float:
    values = data[f"{name}_max_abs_delta"]
    if count is not None:
        values = values[:count]
    if values.size <= 1:
        return 0.0
    return float(np.percentile(values[1:], 99.0))


def _shape(data: dict[str, np.ndarray], name: str) -> str:
    return "x".join(str(dim) for dim in data[f"{name}_values"].shape)


def _write_svg_plot(path: Path, isaac: dict[str, np.ndarray], sim2sim: dict[str, np.ndarray], count: int | None) -> None:
    rows = []
    for name in SLICE_NAMES:
        required_keys = {f"{name}_values", f"{name}_max_abs_delta"}
        if not required_keys.issubset(isaac) or not required_keys.issubset(sim2sim):
            continue
        rows.append((name, _max_delta(isaac, name, count), _max_delta(sim2sim, name, count)))
    if not rows:
        raise ValueError("No matching delta slices were found to plot.")

    width = 1200
    row_h = 34
    margin_l = 190
    margin_r = 80
    margin_t = 42
    margin_b = 48
    height = margin_t + margin_b + row_h * len(rows)
    plot_w = width - margin_l - margin_r
    max_value = max(max(isaac_max, sim2sim_max) for _, isaac_max, sim2sim_max in rows)
    max_value = max(max_value, 1.0e-6)

    def x_pos(value: float) -> float:
        return margin_l + (value / max_value) * plot_w

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<text x="24" y="28" font-family="monospace" font-size="18" fill="#111">APEX tracker max per-step deltas</text>',
        f'<line x1="{margin_l}" y1="{margin_t - 8}" x2="{margin_l}" y2="{height - margin_b + 8}" stroke="#999"/>',
        f'<line x1="{margin_l}" y1="{height - margin_b + 8}" x2="{width - margin_r}" y2="{height - margin_b + 8}" stroke="#999"/>',
        f'<text x="{margin_l}" y="{height - 16}" font-family="monospace" font-size="12" fill="#555">0</text>',
        f'<text x="{width - margin_r - 60}" y="{height - 16}" font-family="monospace" font-size="12" fill="#555">{_format_float(max_value)}</text>',
        f'<rect x="{width - 235}" y="14" width="14" height="14" fill="#2563eb"/>',
        f'<text x="{width - 215}" y="26" font-family="monospace" font-size="13" fill="#222">Isaac</text>',
        f'<rect x="{width - 150}" y="14" width="14" height="14" fill="#dc2626"/>',
        f'<text x="{width - 130}" y="26" font-family="monospace" font-size="13" fill="#222">sim2sim</text>',
    ]
    for idx, (name, isaac_max, sim2sim_max) in enumerate(rows):
        y = margin_t + idx * row_h + row_h * 0.5
        parts.append(f'<text x="16" y="{y + 4:.1f}" font-family="monospace" font-size="13" fill="#222">{name}</text>')
        parts.append(
            f'<line x1="{margin_l}" y1="{y - 5:.1f}" x2="{x_pos(isaac_max):.1f}" y2="{y - 5:.1f}" '
            'stroke="#2563eb" stroke-width="9" stroke-linecap="round"/>'
        )
        parts.append(
            f'<line x1="{margin_l}" y1="{y + 7:.1f}" x2="{x_pos(sim2sim_max):.1f}" y2="{y + 7:.1f}" '
            'stroke="#dc2626" stroke-width="9" stroke-linecap="round"/>'
        )
        parts.append(
            f'<text x="{width - margin_r + 8}" y="{y - 1:.1f}" font-family="monospace" font-size="11" fill="#2563eb">'
            f'{_format_float(isaac_max)}</text>'
        )
        parts.append(
            f'<text x="{width - margin_r + 8}" y="{y + 13:.1f}" font-family="monospace" font-size="11" fill="#dc2626">'
            f'{_format_float(sim2sim_max)}</text>'
        )
    parts.append("</svg>")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--isaac", required=True, type=Path, help="Isaac playback NPZ from --apex-delta-log.")
    parser.add_argument("--sim2sim", required=True, type=Path, help="MuJoCo sim2sim NPZ from --apex-delta-log.")
    parser.add_argument(
        "--all-samples",
        action="store_true",
        help="Compare each log over its full length instead of truncating to the overlapping sample count.",
    )
    parser.add_argument("--plot", type=Path, default=None, help="Optional SVG path for a max-delta comparison plot.")
    args = parser.parse_args()

    isaac = _load(args.isaac)
    sim2sim = _load(args.sim2sim)
    isaac_count = int(isaac["steps"].shape[0])
    sim2sim_count = int(sim2sim["steps"].shape[0])
    count = None if args.all_samples else min(isaac_count, sim2sim_count)

    compared = "all samples" if count is None else f"first {count} overlapping samples"
    print(f"Compared {compared} (isaac={isaac_count}, sim2sim={sim2sim_count})")
    print("slice, isaac_max, sim2sim_max, ratio, isaac_p99, sim2sim_p99, isaac_shape, sim2sim_shape")
    for name in SLICE_NAMES:
        required_keys = {f"{name}_values", f"{name}_max_abs_delta"}
        if not required_keys.issubset(isaac) or not required_keys.issubset(sim2sim):
            print(f"{name}, missing, missing, missing, missing, missing, regenerate logs, regenerate logs")
            continue
        isaac_max = _max_delta(isaac, name, count)
        sim2sim_max = _max_delta(sim2sim, name, count)
        ratio = float("inf") if isaac_max == 0.0 and sim2sim_max > 0.0 else sim2sim_max / max(isaac_max, 1.0e-12)
        isaac_p99 = _p99_delta(isaac, name, count)
        sim2sim_p99 = _p99_delta(sim2sim, name, count)
        print(
            f"{name}, {isaac_max:.8g}, {sim2sim_max:.8g}, {ratio:.4g}, "
            f"{isaac_p99:.8g}, {sim2sim_p99:.8g}, {_shape(isaac, name)}, {_shape(sim2sim, name)}"
        )
    if args.plot is not None:
        _write_svg_plot(args.plot, isaac, sim2sim, count)
        print(f"Wrote plot: {args.plot}")


if __name__ == "__main__":
    main()
