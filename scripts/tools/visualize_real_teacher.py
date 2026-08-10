#!/usr/bin/env python3
"""Render offline attention maps for the REAL privileged teacher before training."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import torch


REPO_ROOT = Path(__file__).resolve().parents[2]
REAL_TEACHER_PATH = REPO_ROOT / "source/Gurukul/Gurukul/tasks/manager_based/locomotion/velocity/real_teacher.py"
REAL_TEACHER_VIZ_PATH = REPO_ROOT / "source/Gurukul/Gurukul/tasks/manager_based/locomotion/velocity/real_teacher_viz.py"


def _load_module(module_name: str, path: Path):
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def parse_args() -> argparse.Namespace:
    viz_module = _load_module("real_teacher_viz_cli_defaults", REAL_TEACHER_VIZ_PATH)
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/real_teacher_attention"))
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--num-actions", type=int, default=12)
    parser.add_argument("--proprio-dim", type=int, default=viz_module.DEFAULT_PROPRIO_DIM)
    parser.add_argument("--privileged-dim", type=int, default=viz_module.DEFAULT_PRIVILEGED_DIM)
    parser.add_argument("--terrain-rows", type=int, default=viz_module.DEFAULT_TERRAIN_SHAPE[0])
    parser.add_argument("--terrain-cols", type=int, default=viz_module.DEFAULT_TERRAIN_SHAPE[1])
    parser.add_argument("--attention-embed-dim", type=int, default=128)
    parser.add_argument("--attention-num-heads", type=int, default=4)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)

    real_teacher_module = _load_module("real_teacher_cli", REAL_TEACHER_PATH)
    viz_module = _load_module("real_teacher_viz_cli", REAL_TEACHER_VIZ_PATH)

    terrain_shape = (int(args.terrain_rows), int(args.terrain_cols))
    patterns = viz_module.generate_synthetic_terrain_patterns(*terrain_shape)
    obs = viz_module.build_demo_observations(
        patterns=patterns,
        proprio_dim=int(args.proprio_dim),
        privileged_dim=int(args.privileged_dim),
        device=args.device,
    )
    obs_groups = {
        "policy": ["real_teacher_proprio", "real_teacher_terrain", "real_teacher_privileged"],
        "critic": ["real_teacher_proprio", "real_teacher_terrain", "real_teacher_privileged"],
    }
    policy = real_teacher_module.RealTeacherActorCritic(
        obs=obs,
        obs_groups=obs_groups,
        num_actions=int(args.num_actions),
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        terrain_scan_shape=terrain_shape,
        attention_embed_dim=int(args.attention_embed_dim),
        attention_num_heads=int(args.attention_num_heads),
    ).to(args.device)
    policy.eval()

    output_paths = viz_module.save_attention_visualizations(
        policy=policy,
        obs=obs,
        output_dir=args.output_dir,
        sample_names=list(patterns.keys()),
    )
    print(f"Saved {len(output_paths)} REAL teacher attention figure(s) to {args.output_dir}")
    for path in output_paths:
        print(path)


if __name__ == "__main__":
    main()
