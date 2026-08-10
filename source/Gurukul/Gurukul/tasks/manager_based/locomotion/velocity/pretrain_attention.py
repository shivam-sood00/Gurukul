
"""Terrain-only attention pretraining utilities for REAL."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

from rsl_rl.utils import resolve_nn_activation


def _resolve_scan_shape(num_scan_values: int, scan_shape: tuple[int, int] | None) -> tuple[int, int]:
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


def _build_scan_coordinates(rows: int, cols: int) -> torch.Tensor:
    x_coords = torch.linspace(-1.0, 1.0, rows)
    y_coords = torch.linspace(-1.0, 1.0, cols)
    grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing="ij")
    return torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)


def _build_metric_scan_coordinates(
    rows: int,
    cols: int,
    resolution_x: float,
    resolution_y: float,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor]:
    x_coords = (torch.arange(rows, device=device, dtype=dtype) - 0.5 * (rows - 1)) * float(resolution_x)
    y_coords = (torch.arange(cols, device=device, dtype=dtype) - 0.5 * (cols - 1)) * float(resolution_y)
    return torch.meshgrid(x_coords, y_coords, indexing="ij")


def _make_activation(activation: str) -> nn.Module:
    return copy.deepcopy(resolve_nn_activation(activation))


@dataclass(frozen=True)
class OfflineTerrainDataset:
    """Cached terrain-only dataset for attention pretraining."""

    height_scans: torch.Tensor
    traversability_targets: torch.Tensor
    scan_shape: tuple[int, int]
    terrain_set: str
    raster_resolution: float
    base_heights: torch.Tensor | None = None
    metadata: dict[str, object] | None = None


def reshape_scan_to_map(height_scan: torch.Tensor, scan_shape: tuple[int, int] | None = None) -> torch.Tensor:
    squeeze_batch = height_scan.ndim == 1
    if squeeze_batch:
        height_scan = height_scan.unsqueeze(0)
    if height_scan.ndim != 2:
        raise ValueError(f"Expected height_scan with shape [batch, num_scan_values], got {tuple(height_scan.shape)}.")
    rows, cols = _resolve_scan_shape(int(height_scan.shape[-1]), scan_shape)
    height_map = height_scan.view(height_scan.shape[0], rows, cols)
    return height_map[0] if squeeze_batch else height_map


class TerrainTokenEncoder(nn.Module):
    """Encode flat terrain scan cells into per-cell tokens with 2D coordinates."""

    def __init__(
        self,
        num_scan_values: int,
        embedding_dim: int,
        activation: str,
        scan_shape: tuple[int, int] | None = None,
        hidden_dim: int = 128,
        use_scan_positional_encoding: bool = True,
    ):
        super().__init__()
        self.num_scan_values = int(num_scan_values)
        self.embedding_dim = int(embedding_dim)
        self.hidden_dim = int(hidden_dim)
        self.activation_name = str(activation)
        self.scan_rows, self.scan_cols = _resolve_scan_shape(self.num_scan_values, scan_shape)
        self.use_scan_positional_encoding = bool(use_scan_positional_encoding)
        input_dim = 1 + (2 if self.use_scan_positional_encoding else 0)
        self.register_buffer(
            "scan_coordinates",
            _build_scan_coordinates(self.scan_rows, self.scan_cols),
            persistent=False,
        )
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, self.hidden_dim),
            _make_activation(self.activation_name),
            nn.Linear(self.hidden_dim, self.embedding_dim),
            _make_activation(self.activation_name),
        )

    def forward(self, terrain_scan: torch.Tensor) -> torch.Tensor:
        if terrain_scan.shape[-1] != self.num_scan_values:
            raise ValueError(f"Expected terrain scan dim {self.num_scan_values}, got {terrain_scan.shape[-1]}.")
        tokens = terrain_scan.unsqueeze(-1)
        if self.use_scan_positional_encoding:
            coords = self.scan_coordinates.unsqueeze(0).expand(terrain_scan.shape[0], -1, -1)
            tokens = torch.cat((tokens, coords), dim=-1)
        return self.encoder(tokens)


class TerrainAttentionPretrainer(nn.Module):
    """Terrain-only attention model whose weights are aligned to traversability targets."""

    def __init__(
        self,
        scan_shape: tuple[int, int] = (16, 10),
        attention_embed_dim: int = 128,
        attention_num_heads: int = 4,
        terrain_encoder_hidden_dim: int = 128,
        activation: str = "elu",
        use_scan_positional_encoding: bool = True,
    ):
        super().__init__()
        self.scan_shape = (int(scan_shape[0]), int(scan_shape[1]))
        self.attention_embed_dim = int(attention_embed_dim)
        self.attention_num_heads = int(attention_num_heads)
        self.terrain_encoder_hidden_dim = int(terrain_encoder_hidden_dim)
        self.activation_name = str(activation)
        self.use_scan_positional_encoding = bool(use_scan_positional_encoding)
        num_scan_values = int(self.scan_shape[0]) * int(self.scan_shape[1])
        if self.attention_embed_dim % self.attention_num_heads != 0:
            raise ValueError(
                f"attention_embed_dim={self.attention_embed_dim} must be divisible by attention_num_heads={self.attention_num_heads}."
            )
        self.terrain_encoder = TerrainTokenEncoder(
            num_scan_values=num_scan_values,
            embedding_dim=self.attention_embed_dim,
            activation=self.activation_name,
            scan_shape=self.scan_shape,
            hidden_dim=self.terrain_encoder_hidden_dim,
            use_scan_positional_encoding=self.use_scan_positional_encoding,
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=self.attention_embed_dim,
            num_heads=self.attention_num_heads,
            batch_first=True,
        )
        self.context_norm = nn.LayerNorm(self.attention_embed_dim)
        self.query_token = nn.Parameter(torch.zeros(1, 1, self.attention_embed_dim))
        nn.init.normal_(self.query_token, mean=0.0, std=0.02)

    @classmethod
    def from_metadata(cls, metadata: dict[str, object]) -> "TerrainAttentionPretrainer":
        if not isinstance(metadata, dict):
            raise ValueError("metadata must be a dict to reconstruct a TerrainAttentionPretrainer.")
        scan_shape = metadata.get("scan_shape")
        if not isinstance(scan_shape, (list, tuple)) or len(scan_shape) != 2:
            raise ValueError("Checkpoint metadata does not contain a valid scan_shape entry.")
        attention_embed_dim = int(metadata.get("attention_embed_dim", 128))
        attention_num_heads = int(metadata.get("attention_num_heads", 4))
        terrain_encoder_hidden_dim = int(metadata.get("terrain_encoder_hidden_dim", attention_embed_dim))
        activation = str(metadata.get("activation", "elu"))
        use_scan_positional_encoding = bool(metadata.get("use_scan_positional_encoding", True))
        return cls(
            scan_shape=(int(scan_shape[0]), int(scan_shape[1])),
            attention_embed_dim=attention_embed_dim,
            attention_num_heads=attention_num_heads,
            terrain_encoder_hidden_dim=terrain_encoder_hidden_dim,
            activation=activation,
            use_scan_positional_encoding=use_scan_positional_encoding,
        )

    def export_model_metadata(self) -> dict[str, object]:
        return {
            "scan_shape": self.scan_shape,
            "attention_embed_dim": self.attention_embed_dim,
            "attention_num_heads": self.attention_num_heads,
            "terrain_encoder_hidden_dim": self.terrain_encoder_hidden_dim,
            "activation": self.activation_name,
            "use_scan_positional_encoding": self.use_scan_positional_encoding,
        }

    def forward(self, height_scan: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        terrain_tokens = self.terrain_encoder(height_scan)
        query = self.query_token.expand(height_scan.shape[0], -1, -1)
        attended, attention_weights = self.cross_attention(
            query,
            terrain_tokens,
            terrain_tokens,
            need_weights=True,
            average_attn_weights=False,
        )
        context = self.context_norm((attended + query).squeeze(1))
        return context, attention_weights.squeeze(2)

    def compute_loss(
        self,
        height_scan: torch.Tensor,
        traversability_target: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        _, attention_weights = self.forward(height_scan)
        target = traversability_target.clamp_min(0.0)
        target = target / target.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
        target = target.unsqueeze(1).expand(-1, attention_weights.shape[1], -1)
        loss = -(target * attention_weights.clamp_min(1.0e-8).log()).sum(dim=-1).mean()
        alignment = (attention_weights * target).sum(dim=-1).mean()
        entropy = -(attention_weights * attention_weights.clamp_min(1.0e-8).log()).sum(dim=-1).mean()
        return loss, {
            "alignment": alignment.detach(),
            "entropy": entropy.detach(),
        }

    def export_attention_state_dict(self) -> dict[str, torch.Tensor]:
        """Export terrain-token and key-projection weights supervised offline.

        The traversability loss supervises the terrain encoder and attention
        key projection through the attention logits. Value, output-projection,
        and context-normalization weights receive no gradient and are omitted.
        """
        exported: dict[str, torch.Tensor] = {}
        for key, value in self.terrain_encoder.state_dict().items():
            exported[f"terrain_encoder.{key}"] = value.detach().cpu()
        embed_dim = self.attention_embed_dim
        exported["cross_attention.k_in_proj_weight"] = self.cross_attention.in_proj_weight.detach().cpu()[
            embed_dim : 2 * embed_dim
        ]
        return exported


def _compute_local_support(
    height_maps: torch.Tensor,
    kernel_size: int,
    tolerance: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    pad = kernel_size // 2
    padded = F.pad(height_maps.unsqueeze(1), (pad, pad, pad, pad), mode="replicate")
    patches = F.unfold(padded, kernel_size=kernel_size).transpose(1, 2)
    patches = patches.view(height_maps.shape[0], height_maps.shape[1], height_maps.shape[2], kernel_size * kernel_size)
    center = height_maps.unsqueeze(-1)
    support = torch.exp(-torch.abs(patches - center) / max(float(tolerance), 1.0e-6)).mean(dim=-1)
    roughness = torch.sqrt(torch.mean((patches - center).square(), dim=-1) + 1.0e-8)
    downward_drop = (center - patches).clamp_min(0.0).amax(dim=-1)
    upward_step = (patches - center).clamp_min(0.0).amax(dim=-1)
    return support, roughness, downward_drop, upward_step


def compute_traversability_from_heightmap(
    height_maps: torch.Tensor,
    resolution_x: float = 1.0,
    resolution_y: float | None = None,
    slope_weight: float = 2.0,
    roughness_weight: float = 18.0,
    hole_weight: float = 28.0,
    step_weight: float = 10.0,
    support_exponent: float = 2.0,
    support_kernel_size: int = 3,
    support_height_tolerance: float = 0.04,
    normalize_distribution: bool = True,
    min_probability: float = 1.0e-6,
) -> torch.Tensor:
    """Compute a traversability distribution from height maps.

    The score rewards flat, well-supported regions and penalizes roughness, holes, and sharp steps.
    """
    squeeze_batch = height_maps.ndim == 2
    if squeeze_batch:
        height_maps = height_maps.unsqueeze(0)
    if height_maps.ndim != 3:
        raise ValueError(f"Expected height_maps with shape [batch, rows, cols], got {tuple(height_maps.shape)}.")

    step_x = max(float(resolution_x), 1.0e-6)
    step_y = step_x if resolution_y is None else max(float(resolution_y), 1.0e-6)
    kernel = max(1, int(support_kernel_size))
    if kernel % 2 == 0:
        kernel += 1

    padded = F.pad(height_maps.unsqueeze(1), (1, 1, 1, 1), mode="replicate").squeeze(1)
    dz_dx = (padded[:, 2:, 1:-1] - padded[:, :-2, 1:-1]) / (2.0 * step_x)
    dz_dy = (padded[:, 1:-1, 2:] - padded[:, 1:-1, :-2]) / (2.0 * step_y)
    slope = torch.sqrt(dz_dx.square() + dz_dy.square())
    normals = F.normalize(torch.stack((-dz_dx, -dz_dy, torch.ones_like(height_maps)), dim=-1), dim=-1)
    flatness = normals[..., 2].clamp(0.0, 1.0)

    support, roughness, downward_drop, upward_step = _compute_local_support(
        height_maps,
        kernel_size=kernel,
        tolerance=float(support_height_tolerance),
    )

    traversability = flatness.square() * support.clamp_min(1.0e-6).pow(float(support_exponent))
    traversability = traversability * torch.exp(
        -(
            float(slope_weight) * slope
            + float(roughness_weight) * roughness
            + float(hole_weight) * downward_drop
            + float(step_weight) * upward_step
        )
    )
    traversability = traversability.clamp_min(float(min_probability))
    traversability = traversability.reshape(traversability.shape[0], -1)
    if normalize_distribution:
        traversability = traversability / traversability.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    return traversability[0] if squeeze_batch else traversability


def apply_forward_attention_prior(
    attention_scores: torch.Tensor,
    scan_shape: tuple[int, int],
    resolution_x: float,
    resolution_y: float | None = None,
    lookahead_center: float = 0.35,
    forward_sigma: float = 0.45,
    lateral_sigma: float = 0.30,
    min_forward: float = -0.05,
    normalize_distribution: bool = True,
) -> torch.Tensor:
    """Bias a per-cell score map toward the near-forward band the robot is likely to step into next."""
    squeeze_batch = attention_scores.ndim == 1
    if squeeze_batch:
        attention_scores = attention_scores.unsqueeze(0)
    if attention_scores.ndim != 2:
        raise ValueError(
            f"Expected attention_scores with shape [batch, num_scan_values], got {tuple(attention_scores.shape)}."
        )
    rows, cols = int(scan_shape[0]), int(scan_shape[1])
    if rows * cols != attention_scores.shape[-1]:
        raise ValueError(
            f"scan_shape {scan_shape} is incompatible with attention_scores dim {attention_scores.shape[-1]}."
        )
    step_y = float(resolution_x) if resolution_y is None else float(resolution_y)
    grid_x, grid_y = _build_metric_scan_coordinates(
        rows,
        cols,
        resolution_x=float(resolution_x),
        resolution_y=step_y,
        device=attention_scores.device,
        dtype=attention_scores.dtype,
    )
    forward_gate = torch.sigmoid((grid_x - float(min_forward)) / max(0.05, float(forward_sigma) * 0.25))
    lookahead_band = torch.exp(
        -0.5 * ((grid_x - float(lookahead_center)) / max(float(forward_sigma), 1.0e-6)).square()
        -0.5 * (grid_y / max(float(lateral_sigma), 1.0e-6)).square()
    )
    prior = (forward_gate * lookahead_band).reshape(1, -1)
    weighted = attention_scores.clamp_min(1.0e-8) * prior
    weighted = weighted.clamp_min(1.0e-8)
    if normalize_distribution:
        weighted = weighted / weighted.sum(dim=-1, keepdim=True).clamp_min(1.0e-8)
    return weighted[0] if squeeze_batch else weighted


def build_traversability_targets(
    height_scans: torch.Tensor,
    scan_shape: tuple[int, int],
    raster_resolution: float,
    raster_resolution_y: float | None = None,
    apply_forward_prior: bool = True,
    **traversability_kwargs,
) -> torch.Tensor:
    height_maps = reshape_scan_to_map(height_scans, scan_shape=scan_shape)
    targets = compute_traversability_from_heightmap(
        height_maps,
        resolution_x=float(raster_resolution),
        resolution_y=raster_resolution_y,
        **traversability_kwargs,
    )
    if apply_forward_prior:
        targets = apply_forward_attention_prior(
            targets,
            scan_shape=scan_shape,
            resolution_x=float(raster_resolution),
            resolution_y=raster_resolution_y,
        )
    return targets


def train_offline_attention(
    model: TerrainAttentionPretrainer,
    dataset: OfflineTerrainDataset,
    epochs: int = 50,
    batch_size: int = 256,
    learning_rate: float = 1.0e-3,
    device: str = "cpu",
    log_prefix: str | None = None,
    log_every_epochs: int = 0,
) -> list[dict[str, float]]:
    model = model.to(device)
    tensor_dataset = TensorDataset(dataset.height_scans, dataset.traversability_targets)
    loader = DataLoader(tensor_dataset, batch_size=int(batch_size), shuffle=True)
    optimizer = torch.optim.Adam(model.parameters(), lr=float(learning_rate))
    history: list[dict[str, float]] = []
    total_epochs = max(int(epochs), 1)
    log_interval = max(int(log_every_epochs), 1) if log_prefix is not None else 0

    for epoch in range(total_epochs):
        running_loss = 0.0
        running_alignment = 0.0
        running_entropy = 0.0
        num_batches = 0
        model.train()
        for height_scans, targets in loader:
            height_scans = height_scans.to(device)
            targets = targets.to(device)
            loss, metrics = model.compute_loss(height_scans, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
            running_alignment += float(metrics["alignment"].item())
            running_entropy += float(metrics["entropy"].item())
            num_batches += 1

        epoch_stats = {
            "epoch": float(epoch + 1),
            "loss": running_loss / max(num_batches, 1),
            "alignment": running_alignment / max(num_batches, 1),
            "entropy": running_entropy / max(num_batches, 1),
        }
        history.append(epoch_stats)

        if log_interval > 0 and ((epoch + 1) % log_interval == 0 or epoch == 0 or (epoch + 1) == total_epochs):
            print(
                f"[INFO] [{log_prefix}] epoch {epoch + 1}/{total_epochs} "
                f"loss={epoch_stats['loss']:.4f} "
                f"alignment={epoch_stats['alignment']:.4f} "
                f"entropy={epoch_stats['entropy']:.4f}",
                flush=True,
            )

    return history


def save_offline_dataset(dataset: OfflineTerrainDataset, output_path: str | Path) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "height_scans": dataset.height_scans.detach().cpu(),
        "traversability_targets": dataset.traversability_targets.detach().cpu(),
        "scan_shape": tuple(dataset.scan_shape),
        "terrain_set": dataset.terrain_set,
        "raster_resolution": float(dataset.raster_resolution),
        "base_heights": None if dataset.base_heights is None else dataset.base_heights.detach().cpu(),
        "metadata": dict(dataset.metadata or {}),
    }
    torch.save(payload, output_path)
    return output_path


def save_attention_checkpoint(
    model: TerrainAttentionPretrainer,
    output_path: str | Path,
    dataset: OfflineTerrainDataset,
    history: list[dict[str, float]] | None = None,
    extra_metadata: dict[str, object] | None = None,
) -> Path:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "attention_state_dict": model.export_attention_state_dict(),
        "metadata": {
            "terrain_set": dataset.terrain_set,
            "scan_shape": dataset.scan_shape,
            "raster_resolution": float(dataset.raster_resolution),
            "num_samples": int(dataset.height_scans.shape[0]),
            **model.export_model_metadata(),
            **(dataset.metadata or {}),
            **(extra_metadata or {}),
        },
        "history": history or [],
    }
    torch.save(payload, output_path)
    return output_path


def _load_checkpoint_payload(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[dict[str, object], dict[str, object]]:
    payload = torch.load(Path(checkpoint_path), map_location=map_location)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected checkpoint payload to be a dict, got {type(payload)!r}.")
    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    return payload, metadata


def load_attention_checkpoint(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    payload, metadata = _load_checkpoint_payload(checkpoint_path, map_location=map_location)
    attention_state_dict = payload.get("attention_state_dict")
    if not isinstance(attention_state_dict, dict):
        raise ValueError("Checkpoint does not contain an attention_state_dict payload.")
    return attention_state_dict, metadata


def load_attention_pretrainer_model(
    checkpoint_path: str | Path,
    map_location: str | torch.device = "cpu",
    eval_mode: bool = True,
) -> tuple[TerrainAttentionPretrainer, dict[str, object]]:
    payload, metadata = _load_checkpoint_payload(checkpoint_path, map_location=map_location)
    model_state_dict = payload.get("model_state_dict")
    if not isinstance(model_state_dict, dict):
        raise ValueError("Checkpoint does not contain a model_state_dict payload.")
    model = TerrainAttentionPretrainer.from_metadata(metadata)
    model.load_state_dict(model_state_dict)
    if eval_mode:
        model.eval()
    return model, metadata
