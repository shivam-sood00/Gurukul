#!/usr/bin/env python3
"""Create a robot-only motion NPZ by removing every object_* channel."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Source motion NPZ containing robot and object channels.")
    parser.add_argument("output", type=Path, help="Output robot-only NPZ path.")
    args = parser.parse_args()

    input_path = args.input.resolve()
    output_path = args.output.resolve()
    if input_path == output_path:
        raise ValueError("Input and output paths must differ.")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)

    with np.load(input_path, allow_pickle=False) as motion:
        removed_keys = tuple(key for key in motion.files if key.startswith("object_"))
        if not removed_keys:
            raise ValueError(f"Motion has no object_* channels to remove: {input_path}")
        output = {key: np.asarray(motion[key]) for key in motion.files if key not in removed_keys}

    output["robot_only_source"] = np.asarray(input_path.name)
    output["removed_object_channels"] = np.asarray(removed_keys)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **output)
    print(f"Wrote {output_path}")
    print(f"Removed {len(removed_keys)} channels: {', '.join(removed_keys)}")


if __name__ == "__main__":
    main()
