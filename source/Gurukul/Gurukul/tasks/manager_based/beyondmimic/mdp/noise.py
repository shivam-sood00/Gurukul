from __future__ import annotations

import inspect
import re

import torch

from isaaclab.utils import configclass
from isaaclab.utils.noise import UniformNoiseCfg


def jointwise_uniform_noise(data: torch.Tensor, cfg: JointwiseUniformNoiseCfg) -> torch.Tensor:
    """Apply uniform observation noise with optional per-joint ranges."""
    if cfg.joint_noise_scales:
        _resolve_joint_noise_ranges(data, cfg)

    if isinstance(cfg.n_max, torch.Tensor):
        cfg.n_max = cfg.n_max.to(data.device)
    if isinstance(cfg.n_min, torch.Tensor):
        cfg.n_min = cfg.n_min.to(data.device)

    noise = torch.rand_like(data) * (cfg.n_max - cfg.n_min) + cfg.n_min
    if cfg.operation == "add":
        return data + noise
    if cfg.operation == "scale":
        return data * noise
    if cfg.operation == "abs":
        return noise
    raise ValueError(f"Unknown noise operation: {cfg.operation}")


def _resolve_joint_noise_ranges(data: torch.Tensor, cfg: JointwiseUniformNoiseCfg) -> None:
    if isinstance(cfg.n_min, torch.Tensor) and isinstance(cfg.n_max, torch.Tensor):
        return

    env = _find_env_from_call_stack()
    if env is None:
        raise RuntimeError(
            "JointwiseUniformNoiseCfg requires an active Isaac Lab observation manager to resolve joint names."
        )

    joint_names = list(env.scene[cfg.asset_name].joint_names)
    if data.shape[-1] != len(joint_names):
        raise ValueError(
            f"Joint-wise noise expected {len(joint_names)} values for {cfg.asset_name}, got {data.shape[-1]}."
        )

    n_min = torch.full((len(joint_names),), float(cfg.default_n_min), dtype=data.dtype, device=data.device)
    n_max = torch.full((len(joint_names),), float(cfg.default_n_max), dtype=data.dtype, device=data.device)
    for index, joint_name in enumerate(joint_names):
        for joint_pattern, scale in cfg.joint_noise_scales.items():
            if re.fullmatch(joint_pattern, joint_name):
                n_min[index] = -float(scale)
                n_max[index] = float(scale)
                break

    cfg.n_min = n_min
    cfg.n_max = n_max


def _find_env_from_call_stack():
    frame = inspect.currentframe()
    while frame is not None:
        maybe_self = frame.f_locals.get("self")
        maybe_env = getattr(maybe_self, "_env", None)
        if maybe_env is not None and hasattr(maybe_env, "scene"):
            return maybe_env
        frame = frame.f_back
    return None


@configclass
class JointwiseUniformNoiseCfg(UniformNoiseCfg):
    """Uniform noise with regex-selected joint ranges."""

    func = jointwise_uniform_noise

    asset_name: str = "robot"
    default_n_min: float = -0.5
    default_n_max: float = 0.5
    joint_noise_scales: dict[str, float] | None = None
