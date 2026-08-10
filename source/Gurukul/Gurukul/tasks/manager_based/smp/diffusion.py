# SPDX-License-Identifier: Apache-2.0

"""Dependency-free diffusion prior used by SMP pretraining and RL scoring.

The module implements a compact DiT-style epsilon predictor, a cosine DDPM
scheduler, EMA inference, SDS scoring, ancestral sampling, and a strict
weights-only checkpoint contract. Physical motion features are affinely
normalized so their per-channel robust dataset bounds map to ``[-1, 1]``;
values outside those bounds are intentionally not clipped.
"""

from __future__ import annotations

import copy
import math
import os
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .profiles import SmpRobotProfile, get_profile, profile_from_metadata, validate_profile_metadata

SMP_CHECKPOINT_FORMAT = "gurukul.smp.prior"
SMP_CHECKPOINT_FORMAT_VERSION = 1


@dataclass(frozen=True)
class MotionDenoiserConfig:
    """Architecture of the motion-window epsilon predictor."""

    window_size: int
    feature_dim: int
    hidden_dim: int = 256
    num_layers: int = 2
    num_heads: int = 4
    mlp_ratio: float = 4.0
    dropout: float = 0.0
    norm_eps: float = 1.0e-5

    def __post_init__(self) -> None:
        for name in ("window_size", "feature_dim", "hidden_dim", "num_layers", "num_heads"):
            if type(getattr(self, name)) is not int:
                raise TypeError(f"{name} must be an integer.")
        for name in ("mlp_ratio", "dropout", "norm_eps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")
        if self.window_size < 1:
            raise ValueError("window_size must be positive.")
        if self.feature_dim < 1:
            raise ValueError("feature_dim must be positive.")
        if self.hidden_dim < 1:
            raise ValueError("hidden_dim must be positive.")
        if self.num_layers < 1:
            raise ValueError("num_layers must be positive.")
        if self.num_heads < 1:
            raise ValueError("num_heads must be positive.")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads.")
        if self.hidden_dim % 2 != 0:
            raise ValueError("hidden_dim must be even for sinusoidal position embeddings.")
        if self.mlp_ratio <= 0.0:
            raise ValueError("mlp_ratio must be positive.")
        if not 0.0 <= self.dropout < 1.0:
            raise ValueError("dropout must be in [0, 1).")
        if self.norm_eps <= 0.0:
            raise ValueError("norm_eps must be positive.")

    def to_dict(self) -> dict[str, int | float]:
        return asdict(self)

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> MotionDenoiserConfig:
        if not isinstance(values, dict):
            raise TypeError("Motion denoiser configuration must be a dictionary.")
        if not all(isinstance(key, str) for key in values):
            raise TypeError("Motion denoiser configuration keys must be strings.")
        allowed = set(cls.__dataclass_fields__)
        unknown = set(values).difference(allowed)
        if unknown:
            raise ValueError(f"Unknown motion denoiser configuration keys: {sorted(unknown)}.")
        required = {"window_size", "feature_dim"}
        missing = required.difference(values)
        if missing:
            raise ValueError(f"Missing motion denoiser configuration keys: {sorted(missing)}.")
        return cls(**values)


@dataclass(frozen=True, init=False)
class DiffusionConfig:
    """Cosine DDPM configuration.

    ``num_timesteps`` is accepted as an alias for ``T`` for ergonomic use in
    callers that follow PyTorch scheduler naming.
    """

    T: int
    schedule: str
    cosine_s: float
    max_beta: float

    def __init__(
        self,
        T: int = 50,
        schedule: str = "cosine",
        cosine_s: float = 0.008,
        max_beta: float = 0.999,
        *,
        num_timesteps: int | None = None,
    ) -> None:
        if type(T) is not int:
            raise TypeError("Diffusion T must be an integer.")
        if num_timesteps is not None:
            if type(num_timesteps) is not int:
                raise TypeError("num_timesteps must be an integer.")
            if T != 50 and num_timesteps != T:
                raise ValueError(f"Conflicting diffusion lengths: T={T}, num_timesteps={num_timesteps}.")
            T = num_timesteps
        if not isinstance(schedule, str):
            raise TypeError("Diffusion schedule must be a string.")
        for name, value in (("cosine_s", cosine_s), ("max_beta", max_beta)):
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise TypeError(f"{name} must be a real number.")
            if not math.isfinite(float(value)):
                raise ValueError(f"{name} must be finite.")
        object.__setattr__(self, "T", T)
        object.__setattr__(self, "schedule", schedule)
        object.__setattr__(self, "cosine_s", float(cosine_s))
        object.__setattr__(self, "max_beta", float(max_beta))
        self._validate()

    def _validate(self) -> None:
        if self.T < 2:
            raise ValueError("Diffusion T must be at least 2.")
        if self.schedule != "cosine":
            raise ValueError(f"Only the cosine diffusion schedule is supported, got {self.schedule!r}.")
        if self.cosine_s < 0.0:
            raise ValueError("cosine_s must be non-negative.")
        if not 0.0 < self.max_beta < 1.0:
            raise ValueError("max_beta must be in (0, 1).")

    @property
    def num_timesteps(self) -> int:
        return self.T

    def to_dict(self) -> dict[str, int | float | str]:
        return {
            "T": self.T,
            "schedule": self.schedule,
            "cosine_s": self.cosine_s,
            "max_beta": self.max_beta,
        }

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> DiffusionConfig:
        if not isinstance(values, dict):
            raise TypeError("Diffusion configuration must be a dictionary.")
        if not all(isinstance(key, str) for key in values):
            raise TypeError("Diffusion configuration keys must be strings.")
        allowed = {"T", "num_timesteps", "schedule", "cosine_s", "max_beta"}
        unknown = set(values).difference(allowed)
        if unknown:
            raise ValueError(f"Unknown diffusion configuration keys: {sorted(unknown)}.")
        if "T" in values and "num_timesteps" in values:
            raise ValueError("Diffusion configuration must not contain both T and num_timesteps.")
        return cls(**values)


def _timestep_embedding(values: torch.Tensor, dimension: int = 256) -> torch.Tensor:
    """Return the cos-first diffusion timestep encoding used by TinyMDM."""
    half = dimension // 2
    if half == 0:
        return values.float().unsqueeze(-1)
    exponent = -math.log(10_000.0) * torch.arange(half, device=values.device, dtype=torch.float32)
    exponent = exponent / half
    phase = values.float().unsqueeze(-1) * torch.exp(exponent)
    embedding = torch.cat((torch.cos(phase), torch.sin(phase)), dim=-1)
    if embedding.shape[-1] < dimension:
        embedding = F.pad(embedding, (0, dimension - embedding.shape[-1]))
    return embedding


class _SinusoidalPositionEmbedding(nn.Module):
    """Interleaved sine/cosine sequence positions from the reference TinyMDM."""

    def __init__(self, dimension: int, max_sequence_length: int) -> None:
        super().__init__()
        if dimension % 2 != 0:
            raise ValueError("Sinusoidal position embeddings require an even hidden dimension.")
        position = torch.arange(max_sequence_length, dtype=torch.float32).unsqueeze(1)
        frequencies = torch.exp(torch.arange(0, dimension, 2, dtype=torch.float32) * (-math.log(10_000.0) / dimension))
        embedding = torch.zeros(1, max_sequence_length, dimension)
        embedding[0, :, 0::2] = torch.sin(position * frequencies)
        embedding[0, :, 1::2] = torch.cos(position * frequencies)
        self.register_buffer("embedding", embedding, persistent=False)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        return hidden + self.embedding[:, : hidden.shape[1]].to(dtype=hidden.dtype)


class _SwiGluFeedForward(nn.Module):
    """Four-times-expanded SwiGLU feed-forward layer used by TinyMDM."""

    def __init__(self, config: MotionDenoiserConfig) -> None:
        super().__init__()
        inner_dim = max(1, int(round(config.hidden_dim * config.mlp_ratio)))
        self.input_projection = nn.Linear(config.hidden_dim, 2 * inner_dim, bias=True)
        self.dropout = nn.Dropout(config.dropout)
        self.output_projection = nn.Linear(inner_dim, config.hidden_dim, bias=True)

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        activation, gate = self.input_projection(hidden).chunk(2, dim=-1)
        hidden = F.silu(activation) * gate
        return self.output_projection(self.dropout(hidden))


class _AdaptiveTransformerBlock(nn.Module):
    def __init__(self, config: MotionDenoiserConfig) -> None:
        super().__init__()
        hidden_dim = config.hidden_dim
        self.norm_attention = nn.LayerNorm(hidden_dim, eps=config.norm_eps, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(
            hidden_dim,
            config.num_heads,
            dropout=0.0,
            bias=False,
            batch_first=True,
        )
        self.attention_dropout = nn.Dropout(config.dropout)
        self.norm_mlp = nn.LayerNorm(hidden_dim, eps=config.norm_eps, elementwise_affine=False)
        self.mlp = _SwiGluFeedForward(config)
        self.scale_shift_table = nn.Parameter(torch.randn(1, 1, 6, hidden_dim) / math.sqrt(hidden_dim))

    def forward(self, hidden: torch.Tensor, modulation: torch.Tensor) -> torch.Tensor:
        batch_size = hidden.shape[0]
        shift_attn, scale_attn, gate_attn, shift_mlp, scale_mlp, gate_mlp = (
            self.scale_shift_table + modulation.reshape(batch_size, 1, 6, -1)
        ).unbind(dim=2)
        normalized = self.norm_attention(hidden)
        normalized = normalized * (1.0 + scale_attn) + shift_attn
        attended = self.attention(normalized, normalized, normalized, need_weights=False)[0]
        hidden = hidden + gate_attn * self.attention_dropout(attended)
        normalized = self.norm_mlp(hidden)
        normalized = normalized * (1.0 + scale_mlp) + shift_mlp
        return hidden + gate_mlp * self.mlp(normalized)


class MotionDenoiser(nn.Module):
    """Small DiT-style epsilon predictor for fixed-size motion windows."""

    def __init__(self, config: MotionDenoiserConfig) -> None:
        super().__init__()
        self.config = config
        self.preprocess_projection = nn.Conv1d(config.feature_dim, config.feature_dim, 1, bias=False)
        self.input_projection = nn.Linear(config.feature_dim, config.hidden_dim, bias=False)
        self.time_mlp = nn.Sequential(
            nn.Linear(256, config.hidden_dim),
            nn.SiLU(),
            nn.Linear(config.hidden_dim, config.hidden_dim),
        )
        self.time_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(config.hidden_dim, 6 * config.hidden_dim),
        )
        self.position_embedding = _SinusoidalPositionEmbedding(
            config.hidden_dim,
            max(config.window_size, 32),
        )
        self.blocks = nn.ModuleList(_AdaptiveTransformerBlock(config) for _ in range(config.num_layers))
        self.output_projection = nn.Linear(config.hidden_dim, config.feature_dim, bias=False)
        self.postprocess_projection = nn.Conv1d(config.feature_dim, config.feature_dim, 1, bias=False)

    @property
    def window_size(self) -> int:
        return self.config.window_size

    @property
    def feature_dim(self) -> int:
        return self.config.feature_dim

    def forward(self, noised_windows: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if noised_windows.ndim != 3:
            raise ValueError(f"MotionDenoiser expects [batch, window, feature], got {tuple(noised_windows.shape)}.")
        expected = (self.window_size, self.feature_dim)
        if tuple(noised_windows.shape[1:]) != expected:
            raise ValueError(
                f"MotionDenoiser expects trailing shape {expected}, got {tuple(noised_windows.shape[1:])}."
            )
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(noised_windows.shape[0])
        if timesteps.ndim != 1 or timesteps.shape[0] != noised_windows.shape[0]:
            raise ValueError(f"timesteps must have shape ({noised_windows.shape[0]},), got {tuple(timesteps.shape)}.")

        if timesteps.dtype != torch.long:
            raise TypeError("timesteps must use torch.long dtype.")
        if timesteps.device != noised_windows.device:
            raise ValueError("timesteps and noised_windows must be on the same device.")
        time_embedding = _timestep_embedding(timesteps)
        time_embedding = time_embedding.to(dtype=self.time_mlp[0].weight.dtype)
        modulation = self.time_modulation(self.time_mlp(time_embedding))

        hidden = noised_windows.transpose(1, 2)
        hidden = self.preprocess_projection(hidden) + hidden
        hidden = self.input_projection(hidden.transpose(1, 2))
        hidden = self.position_embedding(hidden)
        for block in self.blocks:
            hidden = block(hidden, modulation.to(hidden.dtype))
        output = self.output_projection(hidden).transpose(1, 2)
        output = self.postprocess_projection(output) + output
        return output.transpose(1, 2)


def _cosine_betas(config: DiffusionConfig) -> torch.Tensor:
    steps = torch.arange(config.T + 1, dtype=torch.float64) / config.T
    alpha_bar = torch.cos(((steps + config.cosine_s) / (1.0 + config.cosine_s)) * math.pi * 0.5).square()
    alpha_bar = alpha_bar / alpha_bar[0]
    betas = 1.0 - alpha_bar[1:] / alpha_bar[:-1]
    return betas.clamp(min=1.0e-8, max=config.max_beta).to(torch.float32)


def _random_normal_like(tensor: torch.Tensor, generator: torch.Generator | None = None) -> torch.Tensor:
    return torch.randn(tensor.shape, dtype=tensor.dtype, device=tensor.device, generator=generator)


class CosineDDPMScheduler(nn.Module):
    """Minimal ancestral DDPM scheduler with the Nichol--Dhariwal cosine schedule."""

    def __init__(self, config: DiffusionConfig | int | None = None) -> None:
        super().__init__()
        if config is None:
            config = DiffusionConfig()
        if isinstance(config, int):
            config = DiffusionConfig(T=config)
        if not isinstance(config, DiffusionConfig):
            raise TypeError("CosineDDPMScheduler requires a DiffusionConfig or integer T.")
        self.config = config
        betas = _cosine_betas(config)
        alphas = 1.0 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        alpha_bars_previous = torch.cat((torch.ones(1), alpha_bars[:-1]))
        posterior_variance = betas * (1.0 - alpha_bars_previous) / (1.0 - alpha_bars)
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_bars", alpha_bars)
        self.register_buffer("alpha_bars_previous", alpha_bars_previous)
        self.register_buffer("sqrt_alpha_bars", torch.sqrt(alpha_bars))
        self.register_buffer("sqrt_one_minus_alpha_bars", torch.sqrt(1.0 - alpha_bars))
        self.register_buffer("sqrt_reciprocal_alpha_bars", torch.sqrt(1.0 / alpha_bars))
        self.register_buffer("sqrt_reciprocal_minus_one_alpha_bars", torch.sqrt(1.0 / alpha_bars - 1.0))
        self.register_buffer("posterior_variance", posterior_variance.clamp_min(0.0))
        self.register_buffer(
            "posterior_mean_coefficient_x0",
            betas * torch.sqrt(alpha_bars_previous) / (1.0 - alpha_bars),
        )
        self.register_buffer(
            "posterior_mean_coefficient_xt",
            (1.0 - alpha_bars_previous) * torch.sqrt(alphas) / (1.0 - alpha_bars),
        )

    @property
    def T(self) -> int:
        return self.config.T

    @property
    def num_timesteps(self) -> int:
        return self.config.T

    @staticmethod
    def _extract(values: torch.Tensor, timesteps: torch.Tensor, sample: torch.Tensor) -> torch.Tensor:
        extracted = values[timesteps]
        return extracted.reshape(*timesteps.shape, *([1] * (sample.ndim - timesteps.ndim)))

    def sample_timesteps(
        self,
        batch_size: int,
        device: torch.device | str,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        if type(batch_size) is not int:
            raise TypeError("batch_size must be an integer.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        return torch.randint(0, self.T, (batch_size,), device=device, dtype=torch.long, generator=generator)

    def add_noise(self, clean: torch.Tensor, noise: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if clean.shape != noise.shape:
            raise ValueError(f"clean and noise shapes differ: {tuple(clean.shape)} vs {tuple(noise.shape)}.")
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(clean.shape[0])
        if timesteps.ndim != 1 or timesteps.shape[0] != clean.shape[0]:
            raise ValueError(f"timesteps must have shape ({clean.shape[0]},), got {tuple(timesteps.shape)}.")
        self._validate_timesteps(timesteps)
        return (
            self._extract(self.sqrt_alpha_bars, timesteps, clean) * clean
            + self._extract(self.sqrt_one_minus_alpha_bars, timesteps, clean) * noise
        )

    def predict_clean(self, epsilon: torch.Tensor, noised: torch.Tensor, timesteps: torch.Tensor) -> torch.Tensor:
        if epsilon.shape != noised.shape:
            raise ValueError("epsilon and noised samples must have identical shapes.")
        if timesteps.ndim == 0:
            timesteps = timesteps.expand(noised.shape[0])
        if timesteps.ndim != 1 or timesteps.shape[0] != noised.shape[0]:
            raise ValueError(f"timesteps must have shape ({noised.shape[0]},), got {tuple(timesteps.shape)}.")
        self._validate_timesteps(timesteps)
        return (
            self._extract(self.sqrt_reciprocal_alpha_bars, timesteps, noised) * noised
            - self._extract(self.sqrt_reciprocal_minus_one_alpha_bars, timesteps, noised) * epsilon
        )

    def step(
        self,
        epsilon: torch.Tensor,
        noised: torch.Tensor,
        timestep: int | torch.Tensor,
        *,
        generator: torch.Generator | None = None,
        noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if epsilon.shape != noised.shape:
            raise ValueError("epsilon and noised samples must have identical shapes.")
        if isinstance(timestep, bool):
            raise TypeError("timestep must be an integer, not bool.")
        if isinstance(timestep, int):
            timesteps = torch.full((noised.shape[0],), timestep, dtype=torch.long, device=noised.device)
        else:
            if not isinstance(timestep, torch.Tensor):
                raise TypeError("timestep must be an integer or integer tensor.")
            if timestep.dtype != torch.long:
                raise TypeError("timestep tensors must use torch.long dtype.")
            timesteps = timestep.to(device=noised.device, dtype=torch.long)
            if timesteps.ndim == 0:
                timesteps = timesteps.expand(noised.shape[0])
        if timesteps.shape != (noised.shape[0],):
            raise ValueError(f"timestep tensor must have shape ({noised.shape[0]},), got {tuple(timesteps.shape)}.")
        self._validate_timesteps(timesteps)
        clean_prediction = self.predict_clean(epsilon, noised, timesteps)
        posterior_mean = self._extract(self.posterior_mean_coefficient_x0, timesteps, noised) * clean_prediction
        posterior_mean = posterior_mean + self._extract(self.posterior_mean_coefficient_xt, timesteps, noised) * noised
        if noise is not None and noise.shape != noised.shape:
            raise ValueError("step noise must have the same shape as the noised sample.")
        nonzero = (timesteps > 0).to(noised.dtype).reshape(-1, *([1] * (noised.ndim - 1)))
        if not torch.any(timesteps > 0):
            return posterior_mean
        if noise is None:
            noise = _random_normal_like(noised, generator)
        return posterior_mean + nonzero * torch.sqrt(self._extract(self.posterior_variance, timesteps, noised)) * noise

    def _validate_timesteps(self, timesteps: torch.Tensor) -> None:
        if timesteps.dtype != torch.long:
            raise TypeError("timesteps must use torch.long dtype.")
        if torch.any((timesteps < 0) | (timesteps >= self.T)):
            raise ValueError(f"Diffusion timesteps must be in [0, {self.T}).")


class SmpPrior(nn.Module):
    """Trainable motion prior with a frozen EMA inference model and SDS statistics."""

    def __init__(
        self,
        model_config: MotionDenoiserConfig,
        diffusion_config: DiffusionConfig,
        q_low: torch.Tensor,
        q_high: torch.Tensor,
        *,
        ema_decay: float = 0.9999,
        normalization_eps: float = 1.0e-6,
    ) -> None:
        super().__init__()
        if not isinstance(model_config, MotionDenoiserConfig):
            raise TypeError("model_config must be a MotionDenoiserConfig.")
        if not isinstance(diffusion_config, DiffusionConfig):
            raise TypeError("diffusion_config must be a DiffusionConfig.")
        if isinstance(ema_decay, bool) or not isinstance(ema_decay, (int, float)):
            raise TypeError("ema_decay must be a real number.")
        if isinstance(normalization_eps, bool) or not isinstance(normalization_eps, (int, float)):
            raise TypeError("normalization_eps must be a real number.")
        if not math.isfinite(float(ema_decay)):
            raise ValueError("ema_decay must be finite.")
        if not math.isfinite(float(normalization_eps)):
            raise ValueError("normalization_eps must be finite.")
        if not 0.0 <= ema_decay < 1.0:
            raise ValueError("ema_decay must be in [0, 1).")
        if normalization_eps <= 0.0:
            raise ValueError("normalization_eps must be positive.")
        self.model_config = model_config
        self.diffusion_config = diffusion_config
        self.ema_decay = float(ema_decay)
        self.normalization_eps = float(normalization_eps)
        self._validate_bounds(q_low, q_high, model_config.feature_dim)
        self.register_buffer("q_low", q_low.detach().clone())
        self.register_buffer("q_high", q_high.detach().clone())
        self.model = MotionDenoiser(model_config)
        self.ema_model = copy.deepcopy(self.model)
        self.ema_model.requires_grad_(False)
        self.ema_model.eval()
        self.scheduler = CosineDDPMScheduler(diffusion_config)
        self.register_buffer("score_running_mean", torch.ones(diffusion_config.T))
        self.register_buffer("score_running_count", torch.zeros(diffusion_config.T, dtype=torch.long))
        self.register_buffer("score_pending_sum", torch.zeros(diffusion_config.T))
        self.register_buffer("score_pending_count", torch.zeros(diffusion_config.T, dtype=torch.long))
        self.register_buffer("ema_updates", torch.zeros((), dtype=torch.long))
        self.to(device=q_low.device, dtype=q_low.dtype)

    @staticmethod
    def _validate_bounds(q_low: torch.Tensor, q_high: torch.Tensor, feature_dim: int) -> None:
        if not isinstance(q_low, torch.Tensor) or not isinstance(q_high, torch.Tensor):
            raise TypeError("q_low and q_high must be torch tensors.")
        if q_low.shape != (feature_dim,) or q_high.shape != (feature_dim,):
            raise ValueError(f"q_low and q_high must both have shape ({feature_dim},).")
        if not q_low.is_floating_point() or not q_high.is_floating_point():
            raise TypeError("q_low and q_high must have floating-point dtypes.")
        if q_low.dtype != q_high.dtype:
            raise TypeError("q_low and q_high must have the same dtype.")
        if q_low.device != q_high.device:
            raise ValueError("q_low and q_high must be on the same device.")
        if not torch.isfinite(q_low).all() or not torch.isfinite(q_high).all():
            raise ValueError("SMP feature bounds must be finite.")
        if not torch.all(q_high > q_low):
            raise ValueError("Every q_high value must be greater than q_low.")

    @property
    def window_size(self) -> int:
        return self.model_config.window_size

    @property
    def feature_dim(self) -> int:
        return self.model_config.feature_dim

    @property
    def num_timesteps(self) -> int:
        return self.diffusion_config.T

    def train(self, mode: bool = True) -> SmpPrior:
        super().train(mode)
        self.ema_model.eval()
        return self

    def normalize(self, physical_windows: torch.Tensor) -> torch.Tensor:
        self._validate_windows(physical_windows)
        return 2.0 * (physical_windows - self.q_low) / (self.q_high - self.q_low) - 1.0

    def unnormalize(self, normalized_windows: torch.Tensor) -> torch.Tensor:
        self._validate_windows(normalized_windows)
        return 0.5 * (normalized_windows + 1.0) * (self.q_high - self.q_low) + self.q_low

    def _validate_windows(self, windows: torch.Tensor) -> None:
        if not isinstance(windows, torch.Tensor):
            raise TypeError("Motion windows must be a torch tensor.")
        if windows.ndim != 3:
            raise ValueError(f"Motion windows must have shape [batch, window, feature], got {tuple(windows.shape)}.")
        expected = (self.window_size, self.feature_dim)
        if tuple(windows.shape[1:]) != expected:
            raise ValueError(f"Motion windows must have trailing shape {expected}, got {tuple(windows.shape[1:])}.")
        if windows.shape[0] < 1:
            raise ValueError("Motion-window batches must be non-empty.")
        if not windows.is_floating_point():
            raise TypeError("Motion windows must have a floating-point dtype.")
        if windows.dtype != self.q_low.dtype:
            raise TypeError(f"Motion windows use {windows.dtype}, but the prior and bounds use {self.q_low.dtype}.")
        if windows.device != self.q_low.device:
            raise ValueError(
                f"Motion windows are on {windows.device}, but the prior and bounds are on {self.q_low.device}."
            )
        if not torch.isfinite(windows).all():
            raise ValueError("Motion windows must be finite.")

    @staticmethod
    def _validate_noise(noise: torch.Tensor, reference: torch.Tensor, name: str) -> None:
        if not isinstance(noise, torch.Tensor):
            raise TypeError(f"{name} must be a torch tensor.")
        if noise.shape != reference.shape:
            raise ValueError(f"{name} must have shape {tuple(reference.shape)}, got {tuple(noise.shape)}.")
        if noise.dtype != reference.dtype or noise.device != reference.device:
            raise TypeError(f"{name} must have the same dtype and device as the motion windows.")
        if not torch.isfinite(noise).all():
            raise ValueError(f"{name} must be finite.")

    def forward(self, noised_windows: torch.Tensor, timesteps: torch.Tensor, *, use_ema: bool = False) -> torch.Tensor:
        return (self.ema_model if use_ema else self.model)(noised_windows, timesteps)

    def training_loss(
        self,
        physical_windows: torch.Tensor,
        *,
        timesteps: torch.Tensor | None = None,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        normalized = self.normalize(physical_windows)
        if timesteps is None:
            timesteps = self.scheduler.sample_timesteps(normalized.shape[0], normalized.device, generator)
        else:
            if not isinstance(timesteps, torch.Tensor):
                raise TypeError("training timesteps must be a torch tensor.")
            if timesteps.dtype != torch.long:
                raise TypeError("training timesteps must use torch.long dtype.")
            timesteps = timesteps.to(device=normalized.device)
        if noise is None:
            noise = _random_normal_like(normalized, generator)
        else:
            self._validate_noise(noise, normalized, "training noise")
        noised = self.scheduler.add_noise(normalized, noise, timesteps)
        prediction = self.model(noised, timesteps)
        return F.l1_loss(prediction, noise)

    @torch.no_grad()
    def update_ema(self, decay: float | None = None) -> None:
        if decay is None:
            decay = self.ema_decay
        else:
            if isinstance(decay, bool) or not isinstance(decay, (int, float)):
                raise TypeError("EMA decay must be a real number.")
            decay = float(decay)
        if not math.isfinite(decay):
            raise ValueError("EMA decay must be finite.")
        if not 0.0 <= decay < 1.0:
            raise ValueError("EMA decay must be in [0, 1).")
        for ema_parameter, parameter in zip(self.ema_model.parameters(), self.model.parameters(), strict=True):
            ema_parameter.mul_(decay).add_(parameter, alpha=1.0 - decay)
        for ema_buffer, buffer in zip(self.ema_model.buffers(), self.model.buffers(), strict=True):
            ema_buffer.copy_(buffer)
        self.ema_updates.add_(1)

    @staticmethod
    def _coerce_timesteps(timesteps: int | tuple[int, ...] | list[int] | torch.Tensor) -> tuple[int, ...]:
        if isinstance(timesteps, bool):
            raise TypeError("SDS timesteps must be integers, not bool values.")
        if isinstance(timesteps, int):
            values = (timesteps,)
        elif isinstance(timesteps, torch.Tensor):
            if timesteps.dtype == torch.bool or timesteps.is_floating_point() or timesteps.is_complex():
                raise TypeError("SDS timestep tensors must have an integer dtype.")
            values = tuple(int(value) for value in timesteps.detach().cpu().flatten().tolist())
        elif isinstance(timesteps, (tuple, list)):
            if not all(type(value) is int for value in timesteps):
                raise TypeError("SDS timesteps must contain only integers.")
            values = tuple(timesteps)
        else:
            raise TypeError("SDS timesteps must be an integer, sequence of integers, or integer tensor.")
        if not values:
            raise ValueError("At least one SDS timestep is required.")
        if len(values) != len(set(values)):
            raise ValueError("SDS timesteps must be unique.")
        return values

    @torch.no_grad()
    def sds_losses(
        self,
        physical_windows: torch.Tensor,
        timesteps: int | tuple[int, ...] | list[int] | torch.Tensor = (8, 15, 22),
        *,
        noise: torch.Tensor | None = None,
        generator: torch.Generator | None = None,
        use_ema: bool = True,
    ) -> torch.Tensor:
        normalized = self.normalize(physical_windows)
        timestep_values = self._coerce_timesteps(timesteps)
        if any(value < 0 or value >= self.num_timesteps for value in timestep_values):
            raise ValueError(f"SDS timesteps must be in [0, {self.num_timesteps}).")
        batch_size, window_size, feature_dim = normalized.shape
        count = len(timestep_values)
        expanded = normalized[:, None].expand(batch_size, count, window_size, feature_dim)
        flat_windows = expanded.reshape(batch_size * count, window_size, feature_dim)
        flat_timesteps = (
            torch.tensor(timestep_values, device=normalized.device, dtype=torch.long)
            .unsqueeze(0)
            .expand(batch_size, count)
            .reshape(-1)
        )
        if noise is None:
            flat_noise = _random_normal_like(flat_windows, generator)
        elif not isinstance(noise, torch.Tensor):
            raise TypeError("SDS noise must be a torch tensor.")
        elif noise.shape == normalized.shape:
            self._validate_noise(noise, normalized, "SDS noise")
            flat_noise = noise[:, None].expand_as(expanded).reshape_as(flat_windows)
        elif noise.shape == expanded.shape:
            if noise.dtype != normalized.dtype or noise.device != normalized.device:
                raise TypeError("SDS noise must have the same dtype and device as the motion windows.")
            if not torch.isfinite(noise).all():
                raise ValueError("SDS noise must be finite.")
            flat_noise = noise.reshape_as(flat_windows)
        else:
            raise ValueError(
                f"SDS noise must have shape {tuple(normalized.shape)} or {tuple(expanded.shape)}, "
                f"got {tuple(noise.shape)}."
            )
        noised = self.scheduler.add_noise(flat_windows, flat_noise, flat_timesteps)
        prediction = self.forward(noised, flat_timesteps, use_ema=use_ema)
        losses = (prediction - flat_noise).square().mean(dim=(-2, -1))
        return losses.reshape(batch_size, count)

    @torch.no_grad()
    def score_from_losses(
        self,
        losses: torch.Tensor,
        timesteps: int | tuple[int, ...] | list[int] | torch.Tensor = (8, 15, 22),
        loss_scale: float = 6.0,
        update_normalizer: bool = True,
        *,
        defer_normalizer_update: bool = False,
        synchronize_distributed: bool = True,
    ) -> torch.Tensor:
        """Convert precomputed SDS losses to rewards using one statistics snapshot.

        Callers may evaluate large environment batches through multiple
        :meth:`sds_losses` calls, concatenate the resulting ``[B, K]`` tensors,
        and invoke this method once. Every reward then uses the same previous
        running statistic, independent of inference chunking or environment
        order. When distributed training is initialized, the subsequent
        normalizer update aggregates sums and counts across ranks.
        """
        if isinstance(loss_scale, bool) or not isinstance(loss_scale, (int, float)):
            raise TypeError("loss_scale must be a real number.")
        if not math.isfinite(float(loss_scale)):
            raise ValueError("loss_scale must be finite.")
        if loss_scale < 0.0:
            raise ValueError("loss_scale must be non-negative.")
        timestep_values = self._coerce_timesteps(timesteps)
        if any(value < 0 or value >= self.num_timesteps for value in timestep_values):
            raise ValueError(f"SDS timesteps must be in [0, {self.num_timesteps}).")
        if not isinstance(losses, torch.Tensor):
            raise TypeError("SDS losses must be a torch tensor.")
        expected_shape = (len(timestep_values),)
        if losses.ndim != 2 or tuple(losses.shape[1:]) != expected_shape or losses.shape[0] < 1:
            raise ValueError(f"SDS losses must have shape [batch, {len(timestep_values)}], got {tuple(losses.shape)}.")
        if not losses.is_floating_point():
            raise TypeError("SDS losses must have a floating-point dtype.")
        if losses.dtype != self.score_running_mean.dtype or losses.device != self.score_running_mean.device:
            raise TypeError("SDS losses must have the same dtype and device as the prior.")
        if not torch.isfinite(losses).all() or torch.any(losses < 0.0):
            raise ValueError("SDS losses must be finite and non-negative.")
        timestep_tensor = torch.tensor(timestep_values, device=losses.device, dtype=torch.long)
        previous_mean = self.score_running_mean[timestep_tensor].clamp_min(self.normalization_eps)
        normalized_losses = losses / previous_mean.unsqueeze(0)
        reward = torch.exp(-float(loss_scale) * normalized_losses.mean(dim=-1))
        if update_normalizer:
            if defer_normalizer_update:
                self._record_score_statistics(timestep_tensor, losses)
            else:
                self._update_score_statistics(
                    timestep_tensor,
                    losses,
                    synchronize_distributed=synchronize_distributed,
                )
        return reward

    @torch.no_grad()
    def score(
        self,
        physical_windows: torch.Tensor,
        timesteps: int | tuple[int, ...] | list[int] | torch.Tensor = (8, 15, 22),
        loss_scale: float = 6.0,
        update_normalizer: bool = True,
        *,
        generator: torch.Generator | None = None,
        synchronize_distributed: bool = True,
    ) -> torch.Tensor:
        """Evaluate SDS losses and convert them to morphology rewards."""
        timestep_values = self._coerce_timesteps(timesteps)
        losses = self.sds_losses(physical_windows, timestep_values, generator=generator)
        return self.score_from_losses(
            losses,
            timestep_values,
            loss_scale,
            update_normalizer,
            synchronize_distributed=synchronize_distributed,
        )

    @torch.no_grad()
    def _update_score_statistics(
        self,
        timesteps: torch.Tensor,
        losses: torch.Tensor,
        *,
        synchronize_distributed: bool,
    ) -> None:
        batch_sum = torch.zeros_like(self.score_running_mean)
        batch_count = torch.zeros_like(self.score_running_count)
        batch_sum[timesteps] = losses.sum(dim=0)
        batch_count[timesteps] = losses.shape[0]
        if synchronize_distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(batch_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(batch_count, op=torch.distributed.ReduceOp.SUM)
        self._apply_score_statistics(batch_sum, batch_count)

    @torch.no_grad()
    def _record_score_statistics(self, timesteps: torch.Tensor, losses: torch.Tensor) -> None:
        """Accumulate one environment step without changing the active scale."""
        self.score_pending_sum[timesteps] += losses.sum(dim=0)
        self.score_pending_count[timesteps] += losses.shape[0]

    @torch.no_grad()
    def _apply_score_statistics(self, batch_sum: torch.Tensor, batch_count: torch.Tensor) -> None:
        active = batch_count > 0
        if not torch.any(active):
            return
        old_count = self.score_running_count[active]
        new_count = old_count + batch_count[active]
        total = self.score_running_mean[active] * old_count.to(batch_sum.dtype) + batch_sum[active]
        self.score_running_mean[active] = total / new_count.to(batch_sum.dtype)
        self.score_running_count[active] = new_count

    @torch.no_grad()
    def flush_score_normalizer(self, *, synchronize_distributed: bool = True) -> None:
        """Commit deferred rollout statistics once, synchronizing across ranks."""
        batch_sum = self.score_pending_sum.clone()
        batch_count = self.score_pending_count.clone()
        if synchronize_distributed and torch.distributed.is_available() and torch.distributed.is_initialized():
            torch.distributed.all_reduce(batch_sum, op=torch.distributed.ReduceOp.SUM)
            torch.distributed.all_reduce(batch_count, op=torch.distributed.ReduceOp.SUM)
        self._apply_score_statistics(batch_sum, batch_count)
        self.score_pending_sum.zero_()
        self.score_pending_count.zero_()

    @torch.no_grad()
    def score_normalizer_state(self) -> dict[str, torch.Tensor]:
        """Return the adaptive reward state without denoiser or EMA weights."""
        return {
            "score_running_mean": self.score_running_mean.detach().cpu().clone(),
            "score_running_count": self.score_running_count.detach().cpu().clone(),
        }

    @torch.no_grad()
    def load_score_normalizer_state(self, state: dict[str, torch.Tensor]) -> None:
        """Strictly restore adaptive reward state from an RL checkpoint."""
        if not isinstance(state, dict) or not all(isinstance(key, str) for key in state):
            raise TypeError("SMP score-normalizer state must be a string-to-tensor dictionary.")
        expected_keys = {"score_running_mean", "score_running_count"}
        missing = expected_keys.difference(state)
        unknown = set(state).difference(expected_keys)
        if missing or unknown:
            raise ValueError(
                f"Invalid SMP score-normalizer keys; missing={sorted(missing)}, unknown={sorted(unknown)}."
            )
        mean = state["score_running_mean"]
        count = state["score_running_count"]
        if not isinstance(mean, torch.Tensor) or not isinstance(count, torch.Tensor):
            raise TypeError("SMP score-normalizer values must be torch tensors.")
        expected_shape = (self.num_timesteps,)
        if mean.shape != expected_shape or count.shape != expected_shape:
            raise ValueError(f"SMP score-normalizer tensors must have shape {expected_shape}.")
        if mean.dtype != self.score_running_mean.dtype:
            raise TypeError(f"score_running_mean must use dtype {self.score_running_mean.dtype}.")
        if count.dtype != torch.long:
            raise TypeError("score_running_count must use torch.long dtype.")
        if not torch.isfinite(mean).all() or torch.any(mean < 0.0):
            raise ValueError("score_running_mean must be finite and non-negative.")
        if torch.any(count < 0):
            raise ValueError("score_running_count must be non-negative.")
        self.score_running_mean.copy_(mean.to(device=self.score_running_mean.device))
        self.score_running_count.copy_(count.to(device=self.score_running_count.device))
        self.score_pending_sum.zero_()
        self.score_pending_count.zero_()

    @torch.no_grad()
    def sample(
        self,
        batch_size: int,
        *,
        device: torch.device | str | None = None,
        generator: torch.Generator | None = None,
        use_ema: bool = True,
    ) -> torch.Tensor:
        if type(batch_size) is not int:
            raise TypeError("batch_size must be an integer.")
        if batch_size < 1:
            raise ValueError("batch_size must be positive.")
        model = self.ema_model if use_ema else self.model
        model_device = next(model.parameters()).device
        requested_device = model_device if device is None else torch.device(device)
        if requested_device != model_device:
            raise ValueError(
                f"Requested sampling device {requested_device}, but the prior is on {model_device}; "
                "move the prior with `.to(device)` first."
            )
        sample = torch.randn(
            batch_size,
            self.window_size,
            self.feature_dim,
            device=model_device,
            dtype=self.q_low.dtype,
            generator=generator,
        )
        for timestep in reversed(range(self.num_timesteps)):
            timestep_batch = torch.full((batch_size,), timestep, dtype=torch.long, device=model_device)
            epsilon = model(sample, timestep_batch)
            sample = self.scheduler.step(epsilon, sample, timestep, generator=generator)
        return self.unnormalize(sample)


@dataclass(frozen=True)
class LoadedSmpCheckpoint:
    """Validated frozen SMP checkpoint returned by :func:`load_smp_checkpoint`."""

    prior: SmpPrior
    profile: SmpRobotProfile
    model_config: MotionDenoiserConfig
    diffusion_config: DiffusionConfig
    training_metadata: dict[str, Any]
    metadata: dict[str, Any]
    path: Path

    @property
    def window_size(self) -> int:
        return self.model_config.window_size

    @property
    def feature_dim(self) -> int:
        return self.model_config.feature_dim

    @property
    def control_fps(self) -> float:
        return self.profile.control_fps

    @property
    def q_low(self) -> torch.Tensor:
        return self.prior.q_low

    @property
    def q_high(self) -> torch.Tensor:
        return self.prior.q_high


def _safe_metadata(value: Any, location: str = "training_metadata") -> Any:
    if value is None or isinstance(value, (bool, int, str)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"Checkpoint metadata value at {location} must be finite.")
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu()
    if isinstance(value, (list, tuple)):
        return [_safe_metadata(item, f"{location}[]") for item in value]
    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise TypeError(f"{location} dictionary keys must be strings.")
        return {key: _safe_metadata(item, f"{location}.{key}") for key, item in value.items()}
    raise TypeError(f"Unsupported checkpoint metadata value at {location}: {type(value).__name__}.")


def _cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone() for name, tensor in module.state_dict().items()}


def _validate_state_dict(value: Any, name: str) -> dict[str, torch.Tensor]:
    if not isinstance(value, dict) or not all(isinstance(key, str) for key in value):
        raise TypeError(f"{name} must be a string-to-tensor dictionary.")
    if not all(isinstance(tensor, torch.Tensor) for tensor in value.values()):
        raise TypeError(f"Every {name} value must be a tensor.")
    return value


def _load_validated_state_dict(module: nn.Module, value: Any, name: str) -> None:
    state = _validate_state_dict(value, name)
    expected = module.state_dict()
    missing = set(expected).difference(state)
    unknown = set(state).difference(expected)
    if missing or unknown:
        raise ValueError(f"Invalid {name} keys; missing={sorted(missing)}, unknown={sorted(unknown)}.")
    for key, tensor in state.items():
        expected_tensor = expected[key]
        if tensor.shape != expected_tensor.shape:
            raise ValueError(
                f"{name}[{key!r}] has shape {tuple(tensor.shape)}, expected {tuple(expected_tensor.shape)}."
            )
        if tensor.dtype != expected_tensor.dtype:
            raise TypeError(f"{name}[{key!r}] uses dtype {tensor.dtype}, expected {expected_tensor.dtype}.")
        if tensor.layout != expected_tensor.layout:
            raise TypeError(f"{name}[{key!r}] uses layout {tensor.layout}, expected {expected_tensor.layout}.")
        if (tensor.is_floating_point() or tensor.is_complex()) and not torch.isfinite(tensor).all():
            raise ValueError(f"{name}[{key!r}] contains non-finite values.")
    module.load_state_dict(state, strict=True)


def save_smp_checkpoint(
    path: str | Path,
    prior: SmpPrior,
    profile: str | SmpRobotProfile,
    q_low: torch.Tensor | None = None,
    q_high: torch.Tensor | None = None,
    *,
    training_metadata: dict[str, Any] | None = None,
) -> Path:
    """Atomically save a primitive/tensor-only SMP checkpoint."""
    if not isinstance(prior, SmpPrior):
        raise TypeError("prior must be an SmpPrior.")
    robot_profile = get_profile(profile)
    robot_profile = validate_profile_metadata(robot_profile.to_metadata(), robot_profile.name)
    if prior.feature_dim != robot_profile.feature_dim:
        raise ValueError(
            f"Prior feature dimension {prior.feature_dim} does not match profile "
            f"{robot_profile.name!r} ({robot_profile.feature_dim})."
        )
    for override, current, name in ((q_low, prior.q_low, "q_low"), (q_high, prior.q_high, "q_high")):
        if override is None:
            continue
        if not isinstance(override, torch.Tensor):
            raise TypeError(f"Explicit {name} must be a torch tensor.")
        if override.shape != current.shape or not torch.equal(
            override.detach().cpu(),
            current.detach().cpu(),
        ):
            raise ValueError(f"Explicit {name} does not match the prior's registered bounds.")
    if training_metadata is not None and not isinstance(training_metadata, dict):
        raise TypeError("training_metadata must be a dictionary or None.")
    safe_training_metadata = _safe_metadata(training_metadata or {})
    payload = {
        "format": SMP_CHECKPOINT_FORMAT,
        "format_version": SMP_CHECKPOINT_FORMAT_VERSION,
        "profile": robot_profile.to_metadata(),
        "model_config": prior.model_config.to_dict(),
        "diffusion_config": prior.diffusion_config.to_dict(),
        "ema_decay": prior.ema_decay,
        "normalization_eps": prior.normalization_eps,
        "q_low": prior.q_low.detach().cpu(),
        "q_high": prior.q_high.detach().cpu(),
        "model_state_dict": _cpu_state_dict(prior.model),
        "ema_model_state_dict": _cpu_state_dict(prior.ema_model),
        "score_running_mean": prior.score_running_mean.detach().cpu(),
        "score_running_count": prior.score_running_count.detach().cpu(),
        "ema_updates": int(prior.ema_updates.item()),
        "training_metadata": safe_training_metadata,
    }
    destination = Path(path).expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".pt",
            prefix=f".{destination.name}.",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)
            torch.save(payload, temporary)
        os.replace(temporary_path, destination)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return destination


_CHECKPOINT_KEYS = {
    "format",
    "format_version",
    "profile",
    "model_config",
    "diffusion_config",
    "ema_decay",
    "normalization_eps",
    "q_low",
    "q_high",
    "model_state_dict",
    "ema_model_state_dict",
    "score_running_mean",
    "score_running_count",
    "ema_updates",
    "training_metadata",
}


def load_smp_checkpoint(
    path: str | Path,
    expected_profile: str | SmpRobotProfile | None = None,
    expected_control_fps: float | None = None,
    device: torch.device | str = "cpu",
) -> LoadedSmpCheckpoint:
    """Load, strictly validate, and freeze an SMP checkpoint."""
    checkpoint_path = Path(path).expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"SMP checkpoint does not exist or is not a file: {checkpoint_path}")
    try:
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("This PyTorch version lacks safe weights-only checkpoint loading.") from exc
    if not isinstance(payload, dict):
        raise TypeError("SMP checkpoint root must be a dictionary.")
    if not all(isinstance(key, str) for key in payload):
        raise TypeError("SMP checkpoint root keys must be strings.")
    missing = _CHECKPOINT_KEYS.difference(payload)
    unknown = set(payload).difference(_CHECKPOINT_KEYS)
    if missing or unknown:
        raise ValueError(f"Invalid SMP checkpoint keys; missing={sorted(missing)}, unknown={sorted(unknown)}.")
    if not isinstance(payload["format"], str) or payload["format"] != SMP_CHECKPOINT_FORMAT:
        raise ValueError(f"Unsupported SMP checkpoint format: {payload['format']!r}.")
    if type(payload["format_version"]) is not int or payload["format_version"] != SMP_CHECKPOINT_FORMAT_VERSION:
        raise ValueError(f"Unsupported SMP checkpoint version: {payload['format_version']!r}.")

    if not isinstance(payload["profile"], dict) or not all(isinstance(key, str) for key in payload["profile"]):
        raise TypeError("SMP checkpoint profile must be a string-keyed dictionary.")
    profile = profile_from_metadata(payload["profile"])
    if expected_profile is not None:
        profile = validate_profile_metadata(payload["profile"], expected_profile)
    if expected_control_fps is not None:
        if isinstance(expected_control_fps, bool) or not isinstance(expected_control_fps, (int, float)):
            raise TypeError("expected_control_fps must be a real number or None.")
        if not math.isfinite(float(expected_control_fps)) or expected_control_fps <= 0.0:
            raise ValueError("expected_control_fps must be finite and positive.")
        if profile.control_fps != float(expected_control_fps):
            raise ValueError(
                f"SMP control-rate mismatch: checkpoint requires {profile.control_fps:g} Hz, "
                f"environment provides {float(expected_control_fps):g} Hz."
            )
    model_config = MotionDenoiserConfig.from_dict(payload["model_config"])
    diffusion_config = DiffusionConfig.from_dict(payload["diffusion_config"])
    if model_config.feature_dim != profile.feature_dim:
        raise ValueError(
            f"Checkpoint feature dimension {model_config.feature_dim} does not match profile "
            f"{profile.name!r} ({profile.feature_dim})."
        )
    q_low, q_high = payload["q_low"], payload["q_high"]
    SmpPrior._validate_bounds(q_low, q_high, model_config.feature_dim)
    if (
        isinstance(payload["ema_decay"], bool)
        or not isinstance(payload["ema_decay"], (int, float))
        or isinstance(payload["normalization_eps"], bool)
        or not isinstance(payload["normalization_eps"], (int, float))
    ):
        raise TypeError("Checkpoint EMA decay and normalization epsilon must be numeric.")
    prior = SmpPrior(
        model_config,
        diffusion_config,
        q_low,
        q_high,
        ema_decay=float(payload["ema_decay"]),
        normalization_eps=float(payload["normalization_eps"]),
    )
    _load_validated_state_dict(prior.model, payload["model_state_dict"], "model_state_dict")
    _load_validated_state_dict(prior.ema_model, payload["ema_model_state_dict"], "ema_model_state_dict")
    running_mean = payload["score_running_mean"]
    running_count = payload["score_running_count"]
    if not isinstance(running_mean, torch.Tensor) or running_mean.shape != (diffusion_config.T,):
        raise ValueError(f"score_running_mean must have shape ({diffusion_config.T},).")
    if not isinstance(running_count, torch.Tensor) or running_count.shape != (diffusion_config.T,):
        raise ValueError(f"score_running_count must have shape ({diffusion_config.T},).")
    if running_mean.dtype != prior.score_running_mean.dtype:
        raise TypeError(f"score_running_mean must use dtype {prior.score_running_mean.dtype}.")
    if not torch.isfinite(running_mean).all() or torch.any(running_mean < 0.0):
        raise ValueError("score_running_mean must be finite and non-negative.")
    if running_count.dtype != torch.long or torch.any(running_count < 0):
        raise ValueError("score_running_count must be a non-negative torch.long tensor.")
    if type(payload["ema_updates"]) is not int or payload["ema_updates"] < 0:
        raise ValueError("ema_updates must be a non-negative integer.")
    prior.score_running_mean.copy_(running_mean)
    prior.score_running_count.copy_(running_count.to(torch.long))
    prior.ema_updates.fill_(payload["ema_updates"])
    training_metadata = _safe_metadata(payload["training_metadata"])
    if not isinstance(training_metadata, dict):
        raise TypeError("training_metadata must be a dictionary.")
    prior.to(device)
    prior.eval()
    prior.requires_grad_(False)
    metadata = {
        "format": SMP_CHECKPOINT_FORMAT,
        "format_version": SMP_CHECKPOINT_FORMAT_VERSION,
        "profile": profile.to_metadata(),
        "model_config": model_config.to_dict(),
        "diffusion_config": diffusion_config.to_dict(),
        "training_metadata": training_metadata,
    }
    return LoadedSmpCheckpoint(
        prior=prior,
        profile=profile,
        model_config=model_config,
        diffusion_config=diffusion_config,
        training_metadata=training_metadata,
        metadata=metadata,
        path=checkpoint_path,
    )


__all__ = [
    "CosineDDPMScheduler",
    "DiffusionConfig",
    "LoadedSmpCheckpoint",
    "MotionDenoiser",
    "MotionDenoiserConfig",
    "SMP_CHECKPOINT_FORMAT",
    "SMP_CHECKPOINT_FORMAT_VERSION",
    "SmpPrior",
    "load_smp_checkpoint",
    "save_smp_checkpoint",
]
