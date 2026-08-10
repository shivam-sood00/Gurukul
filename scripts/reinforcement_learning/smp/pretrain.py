#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Pretrain a morphology-specific SMP diffusion prior from canonical windows."""

from __future__ import annotations

import argparse
import importlib
import random
import sys
import time
import types
from collections.abc import Sequence
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SMP_ROOT = REPO_ROOT / "source/Gurukul/Gurukul/tasks/manager_based/smp"
_OFFLINE_PACKAGE = "_gurukul_smp_offline"


def _load_smp_module(module_name: str):
    """Load a pure SMP module without executing Gurukul.__init__."""

    if _OFFLINE_PACKAGE not in sys.modules:
        package = types.ModuleType(_OFFLINE_PACKAGE)
        package.__package__ = _OFFLINE_PACKAGE
        package.__path__ = [str(SMP_ROOT)]
        sys.modules[_OFFLINE_PACKAGE] = package
    return importlib.import_module(f"{_OFFLINE_PACKAGE}.{module_name}")


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, type=Path, help="Prepared SMP window NPZ.")
    parser.add_argument("--output", required=True, type=Path, help="Destination SMP prior checkpoint.")
    parser.add_argument(
        "--profile",
        choices=("g1", "pm01", "go2"),
        default=None,
        help="Optional expected dataset profile; mismatches fail before training.",
    )
    parser.add_argument("--device", default="cuda", help="Torch training device (default: cuda).")
    parser.add_argument("--steps", type=int, default=100_000, help="Optimizer steps (default: 100000).")
    parser.add_argument("--batch-size", type=int, default=256, help="Windows per optimizer step.")
    parser.add_argument("--learning-rate", type=float, default=1.0e-4, help="AdamW learning rate.")
    parser.add_argument("--weight-decay", type=float, default=0.0, help="AdamW weight decay.")
    parser.add_argument("--gradient-clip", type=float, default=1.0, help="Maximum gradient norm.")
    parser.add_argument("--num-workers", type=int, default=0, help="DataLoader worker processes.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--log-every", type=int, default=100, help="Progress interval in steps.")
    parser.add_argument("--force", action="store_true", help="Replace an existing checkpoint.")
    return parser


def _validate_args(args: argparse.Namespace, parser: argparse.ArgumentParser) -> tuple[Path, Path]:
    if args.steps < 1:
        parser.error("--steps must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch-size must be at least 1")
    if args.learning_rate <= 0.0:
        parser.error("--learning-rate must be positive")
    if args.weight_decay < 0.0:
        parser.error("--weight-decay must be non-negative")
    if args.gradient_clip <= 0.0:
        parser.error("--gradient-clip must be positive")
    if args.num_workers < 0:
        parser.error("--num-workers must be non-negative")
    if args.log_every < 1:
        parser.error("--log-every must be at least 1")

    dataset_path = args.dataset.expanduser().resolve()
    if not dataset_path.is_file():
        parser.error(f"dataset does not exist or is not a file: {dataset_path}")
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.force:
        parser.error(f"output already exists: {output_path} (pass --force to replace it)")
    return dataset_path, output_path


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    dataset_path, output_path = _validate_args(args, parser)

    # Keep Torch and SMP out of module import and --help paths. Resolve only the
    # pure SMP files here; importing Gurukul would require Isaac Sim.
    import torch
    from torch.utils.data import DataLoader

    data_module = _load_smp_module("data")
    diffusion_module = _load_smp_module("diffusion")

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dataset = data_module.load_window_dataset(dataset_path, expected_profile=args.profile)
    windows = dataset.windows
    if windows.ndim != 3:
        raise ValueError(f"Expected dataset windows [N, T, F], got {tuple(windows.shape)}.")
    if len(dataset) == 0:
        raise ValueError("The SMP dataset contains no windows.")

    device = torch.device(args.device)
    # Keep preprocessing and checkpoint normalization coupled exactly: these
    # are the bounds validated from the prepared dataset, not recomputed here.
    q_low, q_high = dataset.q_low, dataset.q_high
    model_config = diffusion_module.MotionDenoiserConfig(
        window_size=int(windows.shape[1]),
        feature_dim=int(windows.shape[2]),
    )
    diffusion_config = diffusion_module.DiffusionConfig()
    prior = diffusion_module.SmpPrior(model_config, diffusion_config, q_low, q_high).to(device)
    trainable_parameters = [parameter for parameter in prior.parameters() if parameter.requires_grad]
    optimizer = torch.optim.AdamW(
        trainable_parameters,
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    effective_batch_size = min(args.batch_size, len(dataset))
    loader = DataLoader(
        dataset,
        batch_size=effective_batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )

    prior.train()
    data_iterator = iter(loader)
    running_loss = 0.0
    interval_started = time.monotonic()
    for step in range(1, args.steps + 1):
        try:
            batch = next(data_iterator)
        except StopIteration:
            data_iterator = iter(loader)
            batch = next(data_iterator)
        batch = batch.to(device=device, dtype=torch.float32, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        loss = prior.training_loss(batch)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Non-finite denoising loss at step {step}: {loss.item()!r}")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(trainable_parameters, args.gradient_clip)
        if not torch.isfinite(gradient_norm):
            raise FloatingPointError(f"Non-finite gradient norm at step {step}.")
        optimizer.step()
        prior.update_ema()

        running_loss += float(loss.detach())
        if step % args.log_every == 0 or step == args.steps:
            interval_steps = args.log_every if step % args.log_every == 0 else step % args.log_every
            elapsed = time.monotonic() - interval_started
            print(
                f"step={step}/{args.steps} loss={running_loss / interval_steps:.6f} "
                f"steps_per_second={interval_steps / max(elapsed, 1.0e-9):.2f}",
                flush=True,
            )
            running_loss = 0.0
            interval_started = time.monotonic()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    diffusion_module.save_smp_checkpoint(
        output_path,
        prior,
        dataset.profile,
        training_metadata={
            "dataset_path": str(dataset_path),
            "dataset_source_files": list(dataset.source_files),
            "normalization_source_files": list(dataset.normalization_source_files),
            "optimizer": "AdamW",
            "steps": args.steps,
            "requested_batch_size": args.batch_size,
            "effective_batch_size": effective_batch_size,
            "learning_rate": args.learning_rate,
            "weight_decay": args.weight_decay,
            "gradient_clip": args.gradient_clip,
            "seed": args.seed,
        },
    )
    print(f"saved SMP prior: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
