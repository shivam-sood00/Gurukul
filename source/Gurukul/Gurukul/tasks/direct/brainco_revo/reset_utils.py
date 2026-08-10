from __future__ import annotations

import torch


def sample_joint_reset_positions(
    default: torch.Tensor,
    lower: torch.Tensor,
    upper: torch.Tensor,
    noise_scale: float,
    unit_samples: torch.Tensor,
) -> torch.Tensor:
    """Sample limit-safe reset positions and blend them with the default pose.

    ``unit_samples`` is expected in ``[-1, 1]``. A scale of zero returns the
    default pose, while a scale of one spans the full joint range without the
    lower-limit bias caused by treating signed samples as ``[0, 1]`` samples.
    """
    noise_scale = float(noise_scale)
    if not 0.0 <= noise_scale <= 1.0:
        raise ValueError(f"noise_scale must be within [0, 1], got {noise_scale}.")
    interpolation = 0.5 * (unit_samples.clamp(-1.0, 1.0) + 1.0)
    sampled = lower + (upper - lower) * interpolation
    positions = default + noise_scale * (sampled - default)
    return torch.minimum(torch.maximum(positions, lower), upper)
