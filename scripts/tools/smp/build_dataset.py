#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0

"""Build canonical SMP motion windows from named robot-motion NPZ archives."""

from __future__ import annotations

import argparse
import importlib
import json
import sys
import types
from collections.abc import Sequence
from pathlib import Path

PROFILE_NAMES = ("g1", "pm01", "go2")
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
    parser.add_argument(
        "--profile",
        required=True,
        choices=PROFILE_NAMES,
        help="Morphology contract used to validate and encode every input clip.",
    )
    parser.add_argument(
        "--motion",
        action="append",
        required=True,
        type=Path,
        metavar="PATH",
        help="Named NPZ motion archive. Repeat this option to add another clip.",
    )
    parser.add_argument(
        "--normalization-motion",
        action="append",
        type=Path,
        metavar="PATH",
        help=(
            "Optional broader same-robot corpus used only for q01/q99 bounds. Repeat this option for multiple clips."
        ),
    )
    parser.add_argument(
        "--output",
        required=True,
        type=Path,
        help="Destination compressed NPZ containing chronological motion windows.",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Start-frame stride within each source clip (default: 1).",
    )
    parser.add_argument(
        "--target-fps",
        type=float,
        default=None,
        help="Resampling rate. Defaults to the profile control rate (50 Hz).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace an existing output file.",
    )
    return parser


def _resolved_file(path: Path, parser: argparse.ArgumentParser) -> Path:
    resolved = path.expanduser().resolve()
    if not resolved.is_file():
        parser.error(f"motion archive does not exist or is not a file: {resolved}")
    return resolved


def main(argv: Sequence[str] | None = None) -> int:
    parser = _parser()
    args = parser.parse_args(argv)
    if args.stride < 1:
        parser.error("--stride must be at least 1")
    if args.target_fps is not None and args.target_fps <= 0.0:
        parser.error("--target-fps must be positive")

    motion_paths = [_resolved_file(path, parser) for path in args.motion]
    normalization_paths = (
        [_resolved_file(path, parser) for path in args.normalization_motion]
        if args.normalization_motion is not None
        else None
    )
    output_path = args.output.expanduser().resolve()
    if output_path.exists() and not args.force:
        parser.error(f"output already exists: {output_path} (pass --force to replace it)")

    # Resolve only the pure SMP files. Importing Gurukul would run
    # task registration and require Isaac Sim even for this offline converter.
    data_module = _load_smp_module("data")
    profiles_module = _load_smp_module("profiles")

    profile = profiles_module.get_profile(args.profile)
    summary = data_module.build_smp_dataset(
        motion_paths,
        output_path,
        profile,
        stride=args.stride,
        target_fps=args.target_fps,
        normalization_paths=normalization_paths,
    )
    print(json.dumps(summary, indent=2, sort_keys=True, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
