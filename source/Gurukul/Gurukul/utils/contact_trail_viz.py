"""Visualization helpers for contact trail maps."""

from __future__ import annotations

from pathlib import Path

import torch


def save_contact_trail_images(
    trail_map: torch.Tensor,
    output_dir: str | Path,
    prefix: str = "contact_trail",
    env_idx: int = 0,
    foot_pos_b: torch.Tensor | None = None,
    grid_size: tuple[int, int] = (40, 40),
    resolution: float = 0.05,
) -> None:
    """Save per-channel contact trail heatmaps for one environment."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError as exc:
        raise ImportError("matplotlib is required for contact trail visualization.") from exc

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    env_map = trail_map[env_idx].detach().cpu().numpy()
    num_channels, grid_h, grid_w = env_map.shape
    assert (grid_h, grid_w) == tuple(grid_size)

    for channel_idx in range(num_channels):
        fig, ax = plt.subplots(figsize=(5, 5))
        im = ax.imshow(env_map[channel_idx], origin="lower", cmap="viridis")
        ax.set_title(f"{prefix} channel {channel_idx}")
        ax.set_xlabel("col (+x forward)")
        ax.set_ylabel("row (+y left)")
        fig.colorbar(im, ax=ax, fraction=0.046)
        if foot_pos_b is not None:
            feet = foot_pos_b[env_idx].detach().cpu().numpy()
            for foot in feet:
                col = foot[0] / resolution + grid_w / 2.0
                row = foot[1] / resolution + grid_h / 2.0
                ax.plot(col, row, "r+", markersize=8)
        fig.tight_layout()
        fig.savefig(output_dir / f"{prefix}_channel_{channel_idx}.png", dpi=120)
        plt.close(fig)

    # Combined grid image for quick inspection.
    cols = min(4, num_channels)
    rows = int(np.ceil(num_channels / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.atleast_1d(axes).reshape(rows, cols)
    for channel_idx in range(rows * cols):
        r, c = divmod(channel_idx, cols)
        ax = axes[r, c]
        if channel_idx < num_channels:
            ax.imshow(env_map[channel_idx], origin="lower", cmap="viridis")
            ax.set_title(f"ch {channel_idx}")
        ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_dir / f"{prefix}_grid.png", dpi=120)
    plt.close(fig)
