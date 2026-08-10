#!/usr/bin/env python3

"""Batch convert Go2 APEX CSV motion files into NPZ files."""

from __future__ import annotations

import argparse
from pathlib import Path

from apex_csv_to_motion_npz import convert, parse_csv_leg_order

DEFAULT_GO2_PATTERNS = [
    "animal_mocap/go2_*.csv",
    "animal_mocap/hopturn_go2_*.csv",
    "corss_morpho_new/*.csv",
    "gait_switch/*.csv",
    "walk_these_ways/*.csv",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input-root",
        type=Path,
        default=Path(__file__).resolve().parent / "imitation_data",
        help="Root directory that contains CSV motion files.",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path(__file__).resolve().parent / "npz",
        help="Root directory to write NPZ motion files.",
    )
    parser.add_argument("--fps", type=float, default=50.0, help="CSV sampling frequency in Hz.")
    parser.add_argument(
        "--csv-leg-order",
        type=str,
        default="FL,FR,RL,RR",
        help="Leg labels for CSV groups base1..base4, comma-separated.",
    )
    parser.add_argument(
        "--include",
        action="append",
        default=None,
        help=(
            "Relative glob pattern(s) under --input-root. "
            "Can be used multiple times. If omitted, uses finalized Go2 patterns."
        ),
    )
    parser.add_argument(
        "--all-csv",
        action="store_true",
        help="Convert every CSV under --input-root (recursive).",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip files when output NPZ already exists.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help="Return non-zero exit status if any file fails conversion.",
    )
    parser.add_argument(
        "--ground-align-foot-height",
        action="store_true",
        help="Subtract the per-frame minimum foot world height from base and foot positions.",
    )
    return parser.parse_args()


def collect_csv_files(input_root: Path, patterns: list[str], all_csv: bool) -> list[Path]:
    if all_csv:
        files = sorted(path for path in input_root.rglob("*.csv") if path.is_file())
        return files

    matched: set[Path] = set()
    for pattern in patterns:
        matched.update(path for path in input_root.glob(pattern) if path.is_file())
    return sorted(matched)


def main() -> int:
    args = parse_args()
    leg_order = parse_csv_leg_order(args.csv_leg_order)

    if not args.input_root.is_dir():
        raise FileNotFoundError(f"Input root does not exist: {args.input_root}")
    args.output_root.mkdir(parents=True, exist_ok=True)

    patterns = args.include if args.include is not None else DEFAULT_GO2_PATTERNS
    csv_files = collect_csv_files(args.input_root, patterns, args.all_csv)
    if len(csv_files) == 0:
        print("No CSV files found for conversion.")
        return 0

    converted = 0
    skipped = 0
    failed = 0

    print(f"Input root : {args.input_root}")
    print(f"Output root: {args.output_root}")
    print(f"CSV count  : {len(csv_files)}")

    for input_csv in csv_files:
        relative = input_csv.relative_to(args.input_root)
        output_npz = args.output_root / relative.with_suffix(".npz")
        output_npz.parent.mkdir(parents=True, exist_ok=True)

        if args.skip_existing and output_npz.is_file():
            skipped += 1
            print(f"[SKIP] {relative}")
            continue

        try:
            convert(
                input_csv,
                output_npz,
                args.fps,
                leg_order,
                ground_align_foot_height=args.ground_align_foot_height,
            )
            converted += 1
            print(f"[OK]   {relative} -> {output_npz.relative_to(args.output_root)}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"[FAIL] {relative}: {exc}")

    print("")
    print(f"Converted: {converted}")
    print(f"Skipped  : {skipped}")
    print(f"Failed   : {failed}")

    if failed > 0 and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
