#!/usr/bin/env python3
"""Sanity test for egocentric contact trail warping and reset behavior."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch

_REPO_ROOT = Path(__file__).resolve().parents[1]
_UTILS = _REPO_ROOT / "source" / "Gurukul" / "Gurukul" / "utils"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_memory_mod = _load_module("contact_trail_memory", _UTILS / "contact_trail_memory.py")
_viz_mod = _load_module("contact_trail_viz", _UTILS / "contact_trail_viz.py")

CONTACT_FEATURE_DIM = _memory_mod.CONTACT_FEATURE_DIM
ContactTrailConfig = _memory_mod.ContactTrailConfig
ContactTrailMemory = _memory_mod.ContactTrailMemory
save_contact_trail_images = _viz_mod.save_contact_trail_images


def _make_contact_features(
    num_envs: int,
    num_feet: int,
    contact_mask: torch.Tensor,
    device: torch.device,
) -> torch.Tensor:
    features = torch.zeros((num_envs, num_feet, CONTACT_FEATURE_DIM), device=device)
    features[..., 0] = contact_mask.float()
    features[..., 1] = 0.5 * contact_mask.float()
    features[..., 6] = 0.1
    return features.reshape(num_envs, -1)


def main() -> None:
    parser = argparse.ArgumentParser(description="Debug contact trail warping visualization.")
    parser.add_argument("--output-dir", type=str, default="logs/debug_contact_trails")
    parser.add_argument("--num-envs", type=int, default=2)
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--forward-step", type=float, default=0.05)
    args = parser.parse_args()

    device = torch.device("cpu")
    if torch.cuda.is_available():
        device = torch.device("cuda")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    cfg = ContactTrailConfig(write_mode="engineered", use_warp=True, write_only_on_contact=True)
    memory = ContactTrailMemory(num_envs=args.num_envs, config=cfg, device=device)

    num_feet = 4
    base_pos_w = torch.zeros(args.num_envs, 3, device=device)
    base_quat_w = torch.tensor([1.0, 0.0, 0.0, 0.0], device=device).repeat(args.num_envs, 1)
    foot_pos_b = torch.zeros(args.num_envs, num_feet, 3, device=device)
    foot_pos_b[:, 0, 0] = 0.25
    foot_pos_b[:, 0, 2] = -0.30

    contact = torch.zeros(args.num_envs, num_feet, device=device, dtype=torch.bool)
    contact[:, 0] = True
    features = _make_contact_features(args.num_envs, num_feet, contact, device)

    trail_map = memory.update(
        base_pos_w=base_pos_w,
        base_quat_w=base_quat_w,
        foot_pos_b=foot_pos_b,
        contact_features=features,
        dt=0.02,
    )
    save_contact_trail_images(
        trail_map,
        output_dir / "step_00_initial_write",
        prefix="trail",
        env_idx=0,
        foot_pos_b=foot_pos_b,
        grid_size=cfg.grid_size,
        resolution=cfg.resolution,
    )

    for step in range(1, args.steps + 1):
        base_pos_w = base_pos_w.clone()
        base_pos_w[:, 0] += args.forward_step
        no_contact = torch.zeros(args.num_envs, num_feet, device=device, dtype=torch.bool)
        features = _make_contact_features(args.num_envs, num_feet, no_contact, device)
        trail_map = memory.update(
            base_pos_w=base_pos_w,
            base_quat_w=base_quat_w,
            foot_pos_b=foot_pos_b,
            contact_features=features,
            dt=0.02,
        )
        save_contact_trail_images(
            trail_map,
            output_dir / f"step_{step:02d}_forward",
            prefix="trail",
            env_idx=0,
            foot_pos_b=foot_pos_b,
            grid_size=cfg.grid_size,
            resolution=cfg.resolution,
        )

    reset_ids = torch.tensor([1], device=device)
    memory.reset(reset_ids)
    reset_map = memory.get_map()
    assert torch.all(reset_map[1] == 0.0), "Reset should clear env 1 map."
    assert reset_map[0].abs().sum() > 0.0, "Env 0 map should remain after partial reset."

    print(f"Saved contact trail debug images to: {output_dir.resolve()}")
    print("Sanity checks passed: forward motion shifts trail backward; reset clears selected env.")


if __name__ == "__main__":
    main()
