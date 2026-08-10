#!/usr/bin/env python3
"""Replay Go2 APEX motion NPZ data in Isaac Lab for joint-order verification.

Example:
    python scripts/tools/go2_apex/replay_npz.py \
        --input <path-to-motion.npz> \
        --zero-xy-origin
"""

from __future__ import annotations

import argparse
import glob
import os
import sys
import weakref
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher

CANONICAL_JOINT_NAMES = [
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        nargs="+",
        required=True,
        help="APEX motion NPZ file(s), directories, or glob patterns.",
    )
    parser.add_argument(
        "--disable-root-replay",
        action="store_true",
        help="Disable root replay (by default root state is replayed from the motion file).",
    )
    parser.add_argument(
        "--zero-xy-origin",
        action="store_true",
        help="Subtract first-frame XY to keep the trajectory near the scene origin.",
    )
    parser.add_argument("--start-frame", type=int, default=0, help="First frame index.")
    parser.add_argument("--end-frame", type=int, default=-1, help="Last frame index (inclusive), -1 means full clip.")
    parser.add_argument(
        "--frame-stride",
        type=int,
        default=1,
        help="Advance this many source frames per render. Use a larger value for quick kinematic checks.",
    )
    parser.add_argument("--once", action="store_true", help="Play one pass and exit.")
    parser.add_argument(
        "--disable-keyboard-motion-switch",
        action="store_true",
        help="Disable N/P keyboard shortcuts for switching between loaded motion files.",
    )
    parser.add_argument(
        "--oracle-rewards",
        action="store_true",
        help="Print an oracle-style APEX reward report from the replayed motion clip.",
    )
    parser.add_argument(
        "--oracle-print-interval",
        type=int,
        default=0,
        help="If > 0, print weighted oracle reward terms every N replayed frames.",
    )
    parser.add_argument(
        "--disable-foot-pos-vis",
        action="store_true",
        help="Disable reference foot-position sphere markers.",
    )
    parser.add_argument(
        "--disable-arm-ee-pos-vis",
        action="store_true",
        help="Disable reference arm end-effector position marker when present in the NPZ.",
    )
    parser.add_argument(
        "--disable-object-vis",
        action="store_true",
        help="Disable reference object geometry when object pose/shape channels are present in the NPZ.",
    )
    parser.add_argument(
        "--object-size-scale-range",
        "--object-height-scale-range",
        dest="object_size_scale_range",
        type=float,
        nargs=2,
        metavar=("MIN", "MAX"),
        default=(0.70, 0.90),
        help=(
            "Uniform XYZ object scale range. With multiple environments, scales are ordered from MIN to MAX. "
            "The old --object-height-scale-range spelling remains as an alias."
        ),
    )
    parser.add_argument(
        "--object-size-seed",
        "--object-height-seed",
        dest="object_size_seed",
        type=int,
        default=None,
        help="Optional seed for repeatable single-environment object-size sampling.",
    )
    parser.add_argument(
        "--object-mass-range",
        type=float,
        nargs=2,
        metavar=("MIN_KG", "MAX_KG"),
        default=(0.015, 0.040),
        help="Object mass range paired from lightest to heaviest across multiple environments.",
    )
    parser.add_argument(
        "--num-envs",
        "--num_envs",
        dest="num_envs",
        type=int,
        default=1,
        help="Number of side-by-side replay environments. Use 2 or more to show an ordered size/mass sweep.",
    )
    parser.add_argument(
        "--env-spacing",
        type=float,
        default=2.0,
        help="Spacing in metres between replay environments.",
    )
    parser.add_argument(
        "--gripper-position-scale",
        type=float,
        default=1.0,
        help="Scale stored gripper position/velocity channels during replay.",
    )
    parser.add_argument(
        "--gripper-closed-position",
        type=float,
        default=None,
        help=(
            "Optionally remap the largest stored gripper position to this physical joint target in metres."
        ),
    )
    parser.add_argument(
        "--binary-gripper",
        action="store_true",
        help=(
            "Replay the source gripper phase as open/fully-close targets only. "
            "Requires --gripper-closed-position."
        ),
    )
    parser.add_argument(
        "--camera-focus",
        choices=("auto", "base", "gripper"),
        default="base",
        help=(
            "Replay camera target. Base is the default wide full-robot view; gripper is a close arm view; "
            "auto selects gripper when gripper channels are present."
        ),
    )
    parser.add_argument(
        "--robot",
        choices=("go2", "go2_d1", "b2_z1"),
        default="go2",
        help="Robot asset to use for replay.",
    )
    parser.add_argument(
        "--drive-arm-with-pd",
        action="store_true",
        help=(
            "Go2+D1 only: keep the root and non-arm joints on the kinematic reference, but send J1-J6 and the "
            "physical gripper joint through the current actuator Kp/Kd as 50 Hz position targets."
        ),
    )
    parser.add_argument(
        "--export-replayed-motion-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory for simulator-derived NPZs. The replay pass writes body state read back from "
            "the Isaac articulation and derived reference foot-contact masks."
        ),
    )
    parser.add_argument(
        "--export-corrected-arm-ee-dir",
        type=Path,
        default=None,
        help=(
            "Optional output directory for copies whose arm_ee_pos_w is regenerated from the selected Isaac robot "
            "articulation. Other motion channels are preserved."
        ),
    )
    parser.add_argument(
        "--export-contact-height-threshold",
        type=float,
        default=0.035,
        help="Foot height above per-clip minimum used to derive reference_foot_contact in exported NPZs.",
    )
    parser.add_argument(
        "--export-contact-vertical-speed-threshold",
        type=float,
        default=0.25,
        help="Maximum absolute foot vertical speed used to derive reference_foot_contact in exported NPZs.",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import omni

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# Allow running this script directly from the repo root without an editable install.
_GURUKUL_SOURCE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..", "source", "Gurukul")
)
if os.path.isdir(_GURUKUL_SOURCE_PATH) and _GURUKUL_SOURCE_PATH not in sys.path:
    sys.path.insert(0, _GURUKUL_SOURCE_PATH)

from Gurukul.assets.unitree import (  # isort: skip
    UNITREE_B2_Z1_ARM_CFG,
    UNITREE_GO2_CFG,
    UNITREE_GO2_D1_ARM_APEX_CFG,
)
from Gurukul.tasks.manager_based.go2_apex.config.go2.flat_env_cfg import UnitreeGo2ApexFlatEnvCfg  # isort: skip


REPLAY_ROBOT_CFG = {
    "go2": UNITREE_GO2_CFG,
    "go2_d1": UNITREE_GO2_D1_ARM_APEX_CFG,
    "b2_z1": UNITREE_B2_Z1_ARM_CFG,
}[args_cli.robot]

# Each Viser/Isaac finger mesh extends 0.25 mm inward from its link origin.
GO2_D1_TOTAL_INNER_PAD_INSET_M = 0.0005
GO2_D1_PD_REPLAY_SIM_DT = 0.005
GO2_D1_PD_REPLAY_CONTROL_DECIMATION = 4
GO2_D1_PD_REPLAY_FREQUENCY_HZ = 1.0 / (
    GO2_D1_PD_REPLAY_SIM_DT * GO2_D1_PD_REPLAY_CONTROL_DECIMATION
)


def make_foot_position_visualizer() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/APEXReplay/foot_positions",
        markers={
            "foot": sim_utils.SphereCfg(
                radius=0.03,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.1)),
            )
        },
    )
    visualizer = VisualizationMarkers(marker_cfg)
    visualizer.set_visibility(not bool(getattr(args_cli, "headless", False)))
    return visualizer


def make_arm_ee_position_visualizer() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/APEXReplay/arm_ee_position",
        markers={
            "arm_ee": sim_utils.SphereCfg(
                radius=0.015,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.2, 0.0)),
            )
        },
    )
    visualizer = VisualizationMarkers(marker_cfg)
    visualizer.set_visibility(not bool(getattr(args_cli, "headless", False)))
    return visualizer


def make_current_arm_ee_position_visualizer() -> VisualizationMarkers:
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/APEXReplay/arm_ee_position_current",
        markers={
            "arm_ee": sim_utils.SphereCfg(
                radius=0.012,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.4, 1.0)),
            )
        },
    )
    visualizer = VisualizationMarkers(marker_cfg)
    visualizer.set_visibility(not bool(getattr(args_cli, "headless", False)))
    return visualizer


def make_reference_object_visualizer() -> VisualizationMarkers:
    """Create unit object markers whose per-frame scale comes from NPZ object_size."""
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/APEXReplay/reference_objects",
        markers={
            "cylinder": sim_utils.CylinderCfg(
                radius=0.5,
                height=1.0,
                axis="Z",
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.85, 0.06, 0.04),
                    metallic=0.25,
                    roughness=0.35,
                ),
            ),
            "box": sim_utils.CuboidCfg(
                size=(1.0, 1.0, 1.0),
                visual_material=sim_utils.PreviewSurfaceCfg(
                    diffuse_color=(0.85, 0.06, 0.04),
                    metallic=0.25,
                    roughness=0.35,
                ),
            ),
        },
    )
    visualizer = VisualizationMarkers(marker_cfg)
    visualizer.set_visibility(not bool(getattr(args_cli, "headless", False)))
    return visualizer


def parse_arm_ee_body_names(arm_ee_frame: str | None) -> tuple[str, ...]:
    """Resolve the physical bodies represented by the stored end-effector point."""
    if arm_ee_frame is None:
        return ()
    prefix = "gripper_midpoint:"
    if arm_ee_frame.startswith(prefix):
        return tuple(name.strip() for name in arm_ee_frame[len(prefix) :].split(",") if name.strip())
    return (arm_ee_frame,)


def print_arm_ee_error_summary(error_vectors: list[np.ndarray], *, suffix: str = "") -> None:
    """Print reference-minus-simulator end-effector error statistics."""
    if not error_vectors:
        return
    errors = np.asarray(error_vectors, dtype=np.float64)
    norms = np.linalg.norm(errors, axis=-1)
    mean_vector = errors.mean(axis=0)
    print(
        f"[Replay] Arm EE reference-vs-simulator error{suffix}: "
        f"mean={norms.mean():.6f} m, max={norms.max():.6f} m, "
        f"mean reference-minus-simulator xyz="
        f"[{mean_vector[0]:+.6f}, {mean_vector[1]:+.6f}, {mean_vector[2]:+.6f}] m"
    )


def print_arm_joint_error_summary(
    joint_names: list[str],
    error_vectors: list[np.ndarray],
    *,
    suffix: str = "",
) -> None:
    """Print per-joint reference-minus-measured PD tracking errors."""
    if not error_vectors:
        return
    errors = np.asarray(error_vectors, dtype=np.float64)
    rmse = np.sqrt(np.mean(np.square(errors), axis=0))
    max_abs = np.max(np.abs(errors), axis=0)
    print(f"[Replay] PD arm joint tracking error{suffix} (reference-minus-measured):")
    for index, joint_name in enumerate(joint_names):
        unit = "m" if joint_name.startswith("arm_7_") else "rad"
        print(
            f"[Replay]   {joint_name}: rmse={rmse[index]:.6f} {unit}, "
            f"max_abs={max_abs[index]:.6f} {unit}"
        )


def resolve_motion_inputs(inputs: list[str]) -> list[Path]:
    motion_paths: list[Path] = []
    for source in inputs:
        source_path = Path(source).expanduser()
        if source_path.is_file():
            matches = [source_path]
        elif source_path.is_dir():
            matches = sorted(source_path.rglob("*.npz"))
        else:
            matches = [Path(path) for path in sorted(glob.glob(str(source_path), recursive=True))]
        motion_paths.extend(path.resolve() for path in matches if path.is_file())

    deduped_paths = list(dict.fromkeys(motion_paths))
    if not deduped_paths:
        raise FileNotFoundError(f"No motion NPZ files found from input(s): {inputs}")
    return deduped_paths


def display_names_from_paths(motion_paths: list[Path]) -> list[str]:
    if len(motion_paths) == 1:
        return [motion_paths[0].name]

    common_root = Path(os.path.commonpath([str(path) for path in motion_paths]))
    if common_root.is_file():
        common_root = common_root.parent

    display_names: list[str] = []
    for path in motion_paths:
        try:
            display_names.append(str(path.relative_to(common_root)))
        except ValueError:
            display_names.append(path.name)
    return display_names


def load_motion_npz(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Motion file not found: {path}")

    with np.load(path, allow_pickle=False) as data:
        required_keys = [
            "joint_pos",
            "joint_vel",
            "body_pos_w",
            "body_quat_w",
            "body_lin_vel_w",
            "body_ang_vel_w",
        ]
        missing_keys = [key for key in required_keys if key not in data]
        if missing_keys:
            raise ValueError(f"Motion file is missing required keys: {missing_keys}")

        joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
        joint_vel = np.asarray(data["joint_vel"], dtype=np.float32)
        command_lin_vel_xy = (
            np.asarray(data["command_lin_vel_xy"], dtype=np.float32) if "command_lin_vel_xy" in data else None
        )
        command_ang_vel_z = (
            np.asarray(data["command_ang_vel_z"], dtype=np.float32) if "command_ang_vel_z" in data else None
        )
        body_pos_w = np.asarray(data["body_pos_w"], dtype=np.float32)
        body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float32)
        body_lin_vel_w = np.asarray(data["body_lin_vel_w"], dtype=np.float32)
        body_ang_vel_w = np.asarray(data["body_ang_vel_w"], dtype=np.float32)
        arm_ee_pos_w = np.asarray(data["arm_ee_pos_w"], dtype=np.float32) if "arm_ee_pos_w" in data else None
        arm_ee_frame = (
            str(np.asarray(data["arm_ee_frame"]).reshape(-1)[0]) if "arm_ee_frame" in data else None
        )
        object_names = [str(name) for name in data["object_names"].tolist()] if "object_names" in data else []
        object_shapes = [str(shape) for shape in data["object_shapes"].tolist()] if "object_shapes" in data else []
        object_pos_w = np.asarray(data["object_pos_w"], dtype=np.float32) if "object_pos_w" in data else None
        object_quat_w = np.asarray(data["object_quat_w"], dtype=np.float32) if "object_quat_w" in data else None
        object_size = np.asarray(data["object_size"], dtype=np.float32) if "object_size" in data else None
        object_attached = (
            np.asarray(data["object_attached"], dtype=np.bool_) if "object_attached" in data else None
        )
        gripper_joint_pos = (
            np.asarray(data["gripper_joint_pos"], dtype=np.float32) if "gripper_joint_pos" in data else None
        )
        gripper_joint_names = []
        gripper_joint_vel = None
        if gripper_joint_pos is not None:
            gripper_joint_names = (
                [str(name) for name in data["gripper_joint_names"].tolist()]
                if "gripper_joint_names" in data
                else ["arm_7_1_joint", "arm_7_2_joint"]
            )
            if "gripper_joint_vel" in data:
                gripper_joint_vel = np.asarray(data["gripper_joint_vel"], dtype=np.float32)
            else:
                gripper_fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else 50.0
                gripper_joint_vel = np.gradient(
                    gripper_joint_pos,
                    1.0 / gripper_fps,
                    axis=0,
                ).astype(np.float32)
        fps = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else 50.0
        joint_names = data["joint_names"].tolist() if "joint_names" in data else list(CANONICAL_JOINT_NAMES)
        body_names = data["body_names"].tolist() if "body_names" in data else ["base"]

    joint_names = [str(name) for name in joint_names]
    body_names = [str(name) for name in body_names]

    if joint_pos.ndim != 2 or joint_vel.ndim != 2:
        raise ValueError(
            f"Expected `joint_pos` and `joint_vel` to be rank-2, got {joint_pos.shape} and {joint_vel.shape}."
        )
    if joint_pos.shape != joint_vel.shape:
        raise ValueError(f"`joint_pos` and `joint_vel` shape mismatch: {joint_pos.shape} vs {joint_vel.shape}.")
    if body_pos_w.ndim != 3 or body_quat_w.ndim != 3 or body_lin_vel_w.ndim != 3 or body_ang_vel_w.ndim != 3:
        raise ValueError("Expected body state arrays to be rank-3 tensors.")
    if len(joint_names) != joint_pos.shape[1]:
        raise ValueError(
            f"`joint_names` length {len(joint_names)} does not match joint dimension {joint_pos.shape[1]}."
        )
    if len(body_names) != body_pos_w.shape[1]:
        raise ValueError(f"`body_names` length {len(body_names)} does not match body dimension {body_pos_w.shape[1]}.")
    if arm_ee_pos_w is not None and (arm_ee_pos_w.ndim != 2 or arm_ee_pos_w.shape != (joint_pos.shape[0], 3)):
        raise ValueError(f"Expected `arm_ee_pos_w` shape {(joint_pos.shape[0], 3)}, got {arm_ee_pos_w.shape}.")
    if gripper_joint_pos is not None:
        expected_gripper_shape = (joint_pos.shape[0], len(gripper_joint_names))
        if gripper_joint_pos.shape != expected_gripper_shape:
            raise ValueError(
                f"Expected gripper_joint_pos shape {expected_gripper_shape}, got {gripper_joint_pos.shape}."
            )
        if gripper_joint_vel.shape != expected_gripper_shape:
            raise ValueError(
                f"Expected gripper_joint_vel shape {expected_gripper_shape}, got {gripper_joint_vel.shape}."
            )
    object_channels = (object_pos_w, object_quat_w, object_size)
    if any(value is not None for value in object_channels):
        if not all(value is not None for value in object_channels):
            raise ValueError("Object replay requires object_pos_w, object_quat_w, and object_size together.")
        num_objects = int(object_pos_w.shape[1])
        if object_pos_w.shape != (joint_pos.shape[0], num_objects, 3):
            raise ValueError(
                f"Expected object_pos_w shape ({joint_pos.shape[0]}, N, 3), got {object_pos_w.shape}."
            )
        if object_quat_w.shape != (joint_pos.shape[0], num_objects, 4):
            raise ValueError(
                f"Expected object_quat_w shape ({joint_pos.shape[0]}, {num_objects}, 4), got {object_quat_w.shape}."
            )
        if object_size.shape != (num_objects, 3):
            raise ValueError(f"Expected object_size shape ({num_objects}, 3), got {object_size.shape}.")
        if object_names and len(object_names) != num_objects:
            raise ValueError(f"object_names has {len(object_names)} entries for {num_objects} objects.")
        if object_shapes and len(object_shapes) != num_objects:
            raise ValueError(f"object_shapes has {len(object_shapes)} entries for {num_objects} objects.")
        unsupported_shapes = sorted(set(object_shapes) - {"box", "cylinder"})
        if unsupported_shapes:
            raise ValueError(
                f"Unsupported replay object shapes: {unsupported_shapes}; supported: box, cylinder."
            )
        if object_attached is not None and object_attached.shape != (joint_pos.shape[0], num_objects):
            raise ValueError(
                f"Expected object_attached shape ({joint_pos.shape[0]}, {num_objects}), got {object_attached.shape}."
            )

    return {
        "path": path,
        "fps": fps,
        "joint_names": joint_names,
        "body_names": body_names,
        "joint_pos": joint_pos,
        "joint_vel": joint_vel,
        "command_lin_vel_xy": command_lin_vel_xy,
        "command_ang_vel_z": command_ang_vel_z,
        "arm_ee_pos_w": arm_ee_pos_w,
        "arm_ee_frame": arm_ee_frame,
        "object_names": object_names,
        "object_shapes": object_shapes,
        "object_pos_w": object_pos_w,
        "object_quat_w": object_quat_w,
        "object_size": object_size,
        "object_attached": object_attached,
        "gripper_joint_names": gripper_joint_names,
        "gripper_joint_pos": gripper_joint_pos,
        "gripper_joint_vel": gripper_joint_vel,
        "body_pos_w": body_pos_w,
        "body_quat_w": body_quat_w,
        "body_lin_vel_w": body_lin_vel_w,
        "body_ang_vel_w": body_ang_vel_w,
    }


def load_replay_motions(inputs: list[str]) -> list[dict[str, object]]:
    motion_paths = resolve_motion_inputs(inputs)
    motion_names = display_names_from_paths(motion_paths)
    motions = []
    for path, name in zip(motion_paths, motion_names, strict=True):
        motion = load_motion_npz(path)
        motion["name"] = name
        motions.append(motion)

    first_motion = motions[0]
    for motion in motions[1:]:
        for key in ("joint_names", "body_names"):
            if motion[key] != first_motion[key]:
                raise ValueError(f"{motion['path']} has different {key} metadata than {first_motion['path']}.")
        if (motion["arm_ee_pos_w"] is None) != (first_motion["arm_ee_pos_w"] is None):
            raise ValueError(f"{motion['path']} has inconsistent arm_ee_pos_w metadata.")
        if motion["arm_ee_frame"] != first_motion["arm_ee_frame"]:
            raise ValueError(f"{motion['path']} has different arm_ee_frame metadata than {first_motion['path']}.")
        if motion["arm_ee_pos_w"] is not None and motion["arm_ee_pos_w"].shape[1:] != first_motion[
            "arm_ee_pos_w"
        ].shape[1:]:
            raise ValueError(
                f"{motion['path']} has incompatible arm_ee_pos_w shape {motion['arm_ee_pos_w'].shape}; "
                f"expected trailing shape {first_motion['arm_ee_pos_w'].shape[1:]}."
            )
        for key in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
            if motion[key].shape[1:] != first_motion[key].shape[1:]:
                raise ValueError(
                    f"{motion['path']} has incompatible {key} shape {motion[key].shape}; "
                    f"expected trailing shape {first_motion[key].shape[1:]}."
                )
    return motions


def copy_motion_metadata_npz(motion: dict[str, object]) -> dict[str, np.ndarray]:
    """Return all array payloads from the source NPZ so enrichment preserves optional channels."""
    path = motion["path"]
    assert isinstance(path, Path)
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def gripper_replay_scale(gripper_joint_pos: torch.Tensor) -> torch.Tensor:
    """Return the per-joint scale for hardware-positive gripper replay."""
    scale = torch.full(
        (gripper_joint_pos.shape[1],),
        float(args_cli.gripper_position_scale),
        dtype=gripper_joint_pos.dtype,
        device=gripper_joint_pos.device,
    )
    if args_cli.gripper_closed_position is None:
        return scale

    source_closed = torch.amax(torch.abs(gripper_joint_pos), dim=0)
    if torch.any(source_closed <= 0.0):
        raise ValueError(
            "--gripper-closed-position requires every source gripper channel "
            "to contain a non-zero close phase."
        )
    return float(args_cli.gripper_closed_position) / source_closed


def gripper_replay_target_at(
    gripper_joint_pos: torch.Tensor,
    gripper_joint_vel: torch.Tensor,
    gripper_scale: torch.Tensor,
    frame: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the configured physical jaw target for one replay frame."""
    source_position = gripper_joint_pos[frame].unsqueeze(0)
    if args_cli.binary_gripper:
        closed = torch.full_like(source_position, float(args_cli.gripper_closed_position))
        target_position = torch.where(source_position > 0.0, closed, torch.zeros_like(source_position))
        return target_position, torch.zeros_like(target_position)
    return (
        gripper_scale * source_position,
        gripper_scale * gripper_joint_vel[frame].unsqueeze(0),
    )


class KeyboardMotionSwitcher:
    def __init__(self, on_next, on_previous):
        self._input = carb.input.acquire_input_interface()
        self._keyboard = omni.appwindow.get_default_app_window().get_keyboard()
        self._on_next = on_next
        self._on_previous = on_previous
        self._keyboard_sub = self._input.subscribe_to_keyboard_events(
            self._keyboard,
            lambda event, *args, obj=weakref.proxy(self): obj._on_keyboard_event(event, *args),
        )

    def close(self) -> None:
        if self._keyboard_sub is not None:
            self._input.unsubscribe_to_keyboard_events(self._keyboard, self._keyboard_sub)
            self._keyboard_sub = None

    def _on_keyboard_event(self, event, *args, **kwargs) -> bool:
        if event.type == carb.input.KeyboardEventType.KEY_PRESS:
            if event.input.name == "N":
                self._on_next()
            elif event.input.name == "P":
                self._on_previous()
        return True


def compute_oracle_reward_report(
    motion: dict[str, object],
    frame_start: int,
    frame_end: int,
) -> dict[str, object]:
    """Compute a deterministic reward upper-bound from replayed motion states.

    The report mirrors the current Go2 APEX reward formulas for terms that can be
    evaluated directly from the motion clip. Terms that depend on control actions,
    applied torques, or contact sensor signals are reported as unavailable.
    """

    cfg = UnitreeGo2ApexFlatEnvCfg()
    reward_cfg = cfg.rewards
    anchor_body_name = cfg.commands.motion.anchor_body_name

    body_names = motion["body_names"]
    assert isinstance(body_names, list)
    if anchor_body_name not in body_names:
        raise ValueError(f"Anchor body '{anchor_body_name}' is missing from motion body names: {body_names}")

    device = "cpu"
    frame_slice = slice(frame_start, frame_end + 1)
    joint_pos = torch.tensor(motion["joint_pos"][frame_slice], dtype=torch.float32, device=device)
    joint_vel = torch.tensor(motion["joint_vel"][frame_slice], dtype=torch.float32, device=device)
    body_pos_w = torch.tensor(motion["body_pos_w"][frame_slice], dtype=torch.float32, device=device)
    body_quat_w = torch.tensor(motion["body_quat_w"][frame_slice], dtype=torch.float32, device=device)
    body_lin_vel_w = torch.tensor(motion["body_lin_vel_w"][frame_slice], dtype=torch.float32, device=device)
    body_ang_vel_w = torch.tensor(motion["body_ang_vel_w"][frame_slice], dtype=torch.float32, device=device)

    command_lin_vel_xy_np = motion.get("command_lin_vel_xy")
    if command_lin_vel_xy_np is None:
        command_lin_vel_xy = body_lin_vel_w[:, body_names.index(anchor_body_name), :2]
    else:
        command_lin_vel_xy = torch.tensor(command_lin_vel_xy_np[frame_slice], dtype=torch.float32, device=device)

    command_ang_vel_z_np = motion.get("command_ang_vel_z")
    if command_ang_vel_z_np is None:
        command_ang_vel_z = body_ang_vel_w[:, body_names.index(anchor_body_name), 2:3]
    else:
        command_ang_vel_z = torch.tensor(command_ang_vel_z_np[frame_slice], dtype=torch.float32, device=device)
        command_ang_vel_z = command_ang_vel_z.view(command_ang_vel_z.shape[0], -1)
    command_ang_vel_z = torch.clamp(
        command_ang_vel_z,
        min=float(cfg.commands.motion.command_ang_vel_z_clip[0]),
        max=float(cfg.commands.motion.command_ang_vel_z_clip[1]),
    )

    anchor_idx = body_names.index(anchor_body_name)
    foot_indices = [i for i, name in enumerate(body_names) if "foot" in name.lower()]

    root_quat_w = body_quat_w[:, anchor_idx]
    root_lin_vel_b = math_utils.quat_apply_inverse(root_quat_w, body_lin_vel_w[:, anchor_idx])
    root_ang_vel_b = math_utils.quat_apply_inverse(root_quat_w, body_ang_vel_w[:, anchor_idx])

    dt = 1.0 / float(motion["fps"])
    joint_acc = torch.zeros_like(joint_vel)
    if joint_vel.shape[0] > 1:
        joint_acc[1:] = (joint_vel[1:] - joint_vel[:-1]) / dt

    oracle_terms: dict[str, torch.Tensor] = {}
    oracle_terms["imitate_joint_pos"] = torch.full(
        (joint_pos.shape[0],), float(reward_cfg.imitate_joint_pos.weight), dtype=torch.float32, device=device
    )
    oracle_terms["imitate_base_orientation"] = torch.full(
        (joint_pos.shape[0],), float(reward_cfg.imitate_base_orientation.weight), dtype=torch.float32, device=device
    )
    oracle_terms["imitate_projected_gravity"] = torch.full(
        (joint_pos.shape[0],), float(reward_cfg.imitate_projected_gravity.weight), dtype=torch.float32, device=device
    )
    oracle_terms["imitate_foot_pos"] = torch.full(
        (joint_pos.shape[0],), float(reward_cfg.imitate_foot_pos.weight), dtype=torch.float32, device=device
    )

    lin_vel_error = torch.sum(torch.square(command_lin_vel_xy - root_lin_vel_b[:, :2]), dim=1)
    oracle_terms["track_command_lin_vel_xy"] = float(reward_cfg.track_command_lin_vel_xy.weight) * torch.exp(
        -lin_vel_error / float(reward_cfg.track_command_lin_vel_xy.params["sigma"])
    )

    ang_vel_error = torch.square(command_ang_vel_z[:, 0] - root_ang_vel_b[:, 2])
    oracle_terms["track_command_ang_vel_z"] = float(reward_cfg.track_command_ang_vel_z.weight) * torch.exp(
        -ang_vel_error / float(reward_cfg.track_command_ang_vel_z.params["sigma"])
    )

    base_height_error = torch.square(body_pos_w[:, anchor_idx, 2] - body_pos_w[:, anchor_idx, 2])
    oracle_terms["imitate_base_height"] = float(reward_cfg.imitate_base_height.weight) * base_height_error

    ang_vel_xy_sq = torch.sum(torch.square(root_ang_vel_b[:, :2]), dim=1)
    oracle_terms["ang_vel_xy_l2"] = float(reward_cfg.ang_vel_xy_l2.weight) * ang_vel_xy_sq

    joint_acc_sq = torch.sum(torch.square(joint_acc), dim=1)
    oracle_terms["joint_acc_l2"] = float(reward_cfg.joint_acc_l2.weight) * joint_acc_sq

    if foot_indices:
        foot_vel_z = body_lin_vel_w[:, foot_indices, 2]
        delta_v_sq = torch.zeros_like(foot_vel_z)
        if foot_vel_z.shape[0] > 1:
            delta_v_sq[1:] = torch.square(foot_vel_z[1:] - foot_vel_z[:-1])
        impact_raw = torch.sum(torch.clamp(delta_v_sq, max=2.0), dim=1)
        oracle_terms["impact_reduction"] = float(reward_cfg.impact_reduction.weight) * impact_raw
    else:
        oracle_terms["impact_reduction"] = torch.zeros(joint_pos.shape[0], dtype=torch.float32, device=device)

    total = torch.zeros(joint_pos.shape[0], dtype=torch.float32, device=device)
    for value in oracle_terms.values():
        total += value

    omitted_terms = [
        "joint_torques_l2",
        "action_rate_l2",
        "feet_slip",
        "undesired_contacts",
    ]

    return {
        "terms": oracle_terms,
        "total": total,
        "omitted_terms": omitted_terms,
        "frame_start": frame_start,
        "frame_end": frame_end,
    }


def print_oracle_reward_summary(report: dict[str, object]) -> None:
    total = report["total"]
    assert isinstance(total, torch.Tensor)
    print(
        "[Oracle] Deterministic replay reward summary "
        f"for frames [{report['frame_start']}, {report['frame_end']}] "
        "(direct replay upper bound; unavailable penalties assumed zero)."
    )
    terms = report["terms"]
    assert isinstance(terms, dict)
    for name, values in terms.items():
        assert isinstance(values, torch.Tensor)
        print(
            f"[Oracle] {name}: "
            f"mean={values.mean().item():.4f}, min={values.min().item():.4f}, max={values.max().item():.4f}"
        )
    print(
        f"[Oracle] total_upper_bound: mean={total.mean().item():.4f}, "
        f"min={total.min().item():.4f}, max={total.max().item():.4f}"
    )
    omitted_terms = report["omitted_terms"]
    assert isinstance(omitted_terms, list)
    print(f"[Oracle] omitted_terms: {', '.join(omitted_terms)}")


def derive_reference_contact_channels(
    body_pos_w: np.ndarray,
    body_lin_vel_w: np.ndarray,
    body_names: list[str],
    height_threshold: float,
    vertical_speed_threshold: float,
) -> dict[str, np.ndarray]:
    """Derive simple reference contact/airborne labels from replayed foot height and vertical speed."""
    foot_indices = [idx for idx, name in enumerate(body_names) if "foot" in name.lower()]
    if not foot_indices:
        return {
            "reference_foot_contact": np.zeros((body_pos_w.shape[0], 0), dtype=np.bool_),
            "reference_airborne": np.zeros((body_pos_w.shape[0],), dtype=np.bool_),
            "reference_foot_height_above_min": np.zeros((body_pos_w.shape[0], 0), dtype=np.float32),
        }

    foot_z = body_pos_w[:, foot_indices, 2]
    foot_height_above_min = foot_z - np.min(foot_z, axis=0, keepdims=True)
    foot_vz = body_lin_vel_w[:, foot_indices, 2]
    contact = (foot_height_above_min <= float(height_threshold)) & (
        np.abs(foot_vz) <= float(vertical_speed_threshold)
    )
    return {
        "reference_foot_contact": contact.astype(np.bool_),
        "reference_airborne": (~np.any(contact, axis=1)).astype(np.bool_),
        "reference_foot_height_above_min": foot_height_above_min.astype(np.float32),
    }


def export_replayed_motion_npzs(
    sim: SimulationContext,
    scene: InteractiveScene,
    robot: Articulation,
    motions: list[dict[str, object]],
    mapped_joint_indices: torch.Tensor,
    default_joint_pos: torch.Tensor,
    default_joint_vel: torch.Tensor,
    default_root_state: torch.Tensor,
) -> None:
    """Replay each clip once and export simulator-derived state requested by the CLI."""
    output_dir = args_cli.export_replayed_motion_dir or args_cli.export_corrected_arm_ee_dir
    if output_dir is None:
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    export_body_state = args_cli.export_replayed_motion_dir is not None

    robot_body_name_to_index = {name: index for index, name in enumerate(robot.body_names)}

    export_kind = "full body state + arm EE" if export_body_state else "corrected arm EE only"
    print(f"[ReplayExport] Output dir: {output_dir} ({export_kind})")
    for motion_index, motion in enumerate(motions, start=1):
        motion_name = motion["name"]
        assert isinstance(motion_name, str)
        motion_body_names = motion["body_names"]
        assert isinstance(motion_body_names, list)
        missing_body_names = [name for name in motion_body_names if name not in robot_body_name_to_index]
        if missing_body_names:
            raise RuntimeError(
                f"Robot is missing bodies required by {motion['path']}: {missing_body_names}\n"
                f"Robot body names: {robot.body_names}"
            )
        mapped_body_indices = torch.tensor(
            [robot_body_name_to_index[name] for name in motion_body_names], dtype=torch.long, device=sim.device
        )
        arm_ee_body_names = (
            parse_arm_ee_body_names(motion.get("arm_ee_frame"))
            if motion.get("arm_ee_pos_w") is not None
            else ()
        )
        missing_arm_ee_body_names = [
            name for name in arm_ee_body_names if name not in robot_body_name_to_index
        ]
        if missing_arm_ee_body_names:
            raise RuntimeError(
                f"Robot is missing arm EE bodies required by {motion['path']}: {missing_arm_ee_body_names}\n"
                f"Robot body names: {robot.body_names}"
            )
        arm_ee_body_indices = torch.tensor(
            [robot_body_name_to_index[name] for name in arm_ee_body_names],
            dtype=torch.long,
            device=sim.device,
        )
        gripper_joint_names = motion.get("gripper_joint_names") or []
        missing_gripper_joint_names = [name for name in gripper_joint_names if name not in robot.joint_names]
        if missing_gripper_joint_names:
            raise RuntimeError(
                f"Robot is missing gripper joints required by {motion['path']}: {missing_gripper_joint_names}"
            )
        gripper_joint_indices = torch.tensor(
            [robot.joint_names.index(name) for name in gripper_joint_names],
            dtype=torch.long,
            device=sim.device,
        )
        gripper_joint_pos = (
            torch.tensor(motion["gripper_joint_pos"], dtype=torch.float32, device=sim.device)
            if motion.get("gripper_joint_pos") is not None
            else None
        )
        gripper_joint_vel = (
            torch.tensor(motion["gripper_joint_vel"], dtype=torch.float32, device=sim.device)
            if motion.get("gripper_joint_vel") is not None
            else None
        )
        gripper_scale = (
            gripper_replay_scale(gripper_joint_pos)
            if gripper_joint_pos is not None
            else None
        )
        base_body_index = motion_body_names.index("base") if "base" in motion_body_names else 0

        joint_pos = torch.tensor(motion["joint_pos"], dtype=torch.float32, device=sim.device)
        joint_vel = torch.tensor(motion["joint_vel"], dtype=torch.float32, device=sim.device)
        body_pos_w = torch.tensor(motion["body_pos_w"], dtype=torch.float32, device=sim.device)
        body_quat_w = torch.tensor(motion["body_quat_w"], dtype=torch.float32, device=sim.device)
        body_lin_vel_w = torch.tensor(motion["body_lin_vel_w"], dtype=torch.float32, device=sim.device)
        body_ang_vel_w = torch.tensor(motion["body_ang_vel_w"], dtype=torch.float32, device=sim.device)
        num_frames = int(joint_pos.shape[0])

        replayed_body_pos_w = np.zeros((num_frames, len(motion_body_names), 3), dtype=np.float32)
        replayed_body_quat_w = np.zeros((num_frames, len(motion_body_names), 4), dtype=np.float32)
        replayed_body_lin_vel_w = np.zeros((num_frames, len(motion_body_names), 3), dtype=np.float32)
        replayed_body_ang_vel_w = np.zeros((num_frames, len(motion_body_names), 3), dtype=np.float32)
        replayed_arm_ee_pos_w = (
            np.zeros((num_frames, 3), dtype=np.float32) if arm_ee_body_names else None
        )

        for frame in range(num_frames):
            replay_joint_pos = default_joint_pos.clone()
            replay_joint_vel = default_joint_vel.clone()
            replay_joint_pos[:, mapped_joint_indices] = joint_pos[frame].unsqueeze(0)
            replay_joint_vel[:, mapped_joint_indices] = joint_vel[frame].unsqueeze(0)
            if gripper_joint_pos is not None:
                gripper_target_pos, gripper_target_vel = gripper_replay_target_at(
                    gripper_joint_pos,
                    gripper_joint_vel,
                    gripper_scale,
                    frame,
                )
                replay_joint_pos[:, gripper_joint_indices] = gripper_target_pos
                replay_joint_vel[:, gripper_joint_indices] = gripper_target_vel
            robot.write_joint_state_to_sim(replay_joint_pos, replay_joint_vel)

            if not args_cli.disable_root_replay:
                root_state = default_root_state.clone()
                root_state[:, :3] = body_pos_w[frame, base_body_index].unsqueeze(0) + scene.env_origins
                root_state[:, 3:7] = body_quat_w[frame, base_body_index].unsqueeze(0)
                root_state[:, 7:10] = body_lin_vel_w[frame, base_body_index].unsqueeze(0)
                root_state[:, 10:13] = body_ang_vel_w[frame, base_body_index].unsqueeze(0)
                robot.write_root_state_to_sim(root_state)

            scene.write_data_to_sim()
            sim.forward()
            scene.update(sim.get_physics_dt())
            if not getattr(args_cli, "headless", False):
                sim.render()

            origin = scene.env_origins[0]
            replayed_body_pos_w[frame] = (
                robot.data.body_pos_w[0, mapped_body_indices] - origin.unsqueeze(0)
            ).detach().cpu().numpy()
            replayed_body_quat_w[frame] = robot.data.body_quat_w[0, mapped_body_indices].detach().cpu().numpy()
            replayed_body_lin_vel_w[frame] = robot.data.body_lin_vel_w[0, mapped_body_indices].detach().cpu().numpy()
            replayed_body_ang_vel_w[frame] = robot.data.body_ang_vel_w[0, mapped_body_indices].detach().cpu().numpy()
            if replayed_arm_ee_pos_w is not None:
                replayed_arm_ee_pos_w[frame] = (
                    robot.data.body_pos_w[0, arm_ee_body_indices].mean(dim=0) - origin
                ).detach().cpu().numpy()

        payload = copy_motion_metadata_npz(motion)
        if export_body_state:
            payload["body_pos_w"] = replayed_body_pos_w
            payload["body_quat_w"] = replayed_body_quat_w
            payload["body_lin_vel_w"] = replayed_body_lin_vel_w
            payload["body_ang_vel_w"] = replayed_body_ang_vel_w
            payload["body_names"] = np.asarray(motion_body_names, dtype=np.str_)
            payload["sim_replay_body_state_source"] = np.asarray(
                "scripts/tools/go2_apex/replay_npz.py --export-replayed-motion-dir", dtype=np.str_
            )
            payload.update(
                derive_reference_contact_channels(
                    replayed_body_pos_w,
                    replayed_body_lin_vel_w,
                    motion_body_names,
                    height_threshold=args_cli.export_contact_height_threshold,
                    vertical_speed_threshold=args_cli.export_contact_vertical_speed_threshold,
                )
            )
        if replayed_arm_ee_pos_w is not None:
            payload["arm_ee_pos_w_embodik"] = np.asarray(motion["arm_ee_pos_w"], dtype=np.float32)
            payload["arm_ee_pos_w"] = replayed_arm_ee_pos_w
            payload["arm_ee_position_source"] = np.asarray(
                "Isaac articulation FK from arm_ee_frame bodies", dtype=np.str_
            )

        rel_output = Path(motion_name)
        output_path = output_dir / rel_output
        output_path.parent.mkdir(parents=True, exist_ok=True)
        input_path = motion["path"]
        assert isinstance(input_path, Path)
        if output_path.resolve() == input_path.resolve():
            raise RuntimeError(f"Refusing to overwrite input motion during replay export: {output_path}")
        np.savez(output_path, **payload)
        print(f"[ReplayExport] {motion_index}/{len(motions)} wrote {output_path}")


@configclass
class ReplayGo2NpzSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = REPLAY_ROBOT_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: SimulationContext, scene: InteractiveScene, motions: list[dict[str, object]]) -> None:
    robot: Articulation = scene["robot"]

    first_motion = motions[0]
    motion_joint_names = first_motion["joint_names"]
    assert isinstance(motion_joint_names, list)
    robot_joint_name_to_index = {name: i for i, name in enumerate(robot.joint_names)}
    missing_joint_names = [name for name in motion_joint_names if name not in robot_joint_name_to_index]
    if missing_joint_names:
        raise RuntimeError(
            f"Robot is missing joints required by motion file: {missing_joint_names}\n"
            f"Robot joint names: {robot.joint_names}"
        )
    mapped_joint_indices = torch.tensor(
        [robot_joint_name_to_index[name] for name in motion_joint_names], dtype=torch.long, device=sim.device
    )
    pd_arm_joint_names: list[str] = []
    pd_arm_joint_indices = torch.empty(0, dtype=torch.long, device=sim.device)
    kinematic_joint_indices = torch.arange(len(robot.joint_names), dtype=torch.long, device=sim.device)
    if args_cli.drive_arm_with_pd:
        rotary_arm_joint_names = [f"arm_{index}_joint" for index in range(1, 7)]
        missing_motion_arm_joint_names = [
            joint_name for joint_name in rotary_arm_joint_names if joint_name not in motion_joint_names
        ]
        if missing_motion_arm_joint_names:
            raise RuntimeError(
                "--drive-arm-with-pd requires a 19-channel Go2+D1 motion; "
                f"motion joint_pos is missing {missing_motion_arm_joint_names}."
            )
        gripper_joint_names = list(first_motion.get("gripper_joint_names") or [])
        if gripper_joint_names != ["arm_7_1_joint"]:
            raise RuntimeError(
                "--drive-arm-with-pd requires the physical gripper channel "
                f"['arm_7_1_joint']; got {gripper_joint_names}."
            )
        pd_arm_joint_names = rotary_arm_joint_names + gripper_joint_names
        missing_pd_arm_joint_names = [
            joint_name for joint_name in pd_arm_joint_names if joint_name not in robot_joint_name_to_index
        ]
        if missing_pd_arm_joint_names:
            raise RuntimeError(
                "The PD arm replay requires D1 J1-J6 and the physical gripper joint; "
                f"missing {missing_pd_arm_joint_names}."
            )
        pd_arm_joint_indices = torch.tensor(
            [robot_joint_name_to_index[joint_name] for joint_name in pd_arm_joint_names],
            dtype=torch.long,
            device=sim.device,
        )
        # The second gripper jaw is driven by the articulation's mimic
        # constraint. Do not overwrite it or include it in the commanded set.
        dynamic_joint_names = set(pd_arm_joint_names) | {"arm_7_2_joint"}
        kinematic_joint_indices = torch.tensor(
            [
                joint_index
                for joint_index, joint_name in enumerate(robot.joint_names)
                if joint_name not in dynamic_joint_names
            ],
            dtype=torch.long,
            device=sim.device,
        )

    motion_body_names = first_motion["body_names"]
    assert isinstance(motion_body_names, list)
    base_body_index = motion_body_names.index("base") if "base" in motion_body_names else 0
    foot_body_indices = [idx for idx, name in enumerate(motion_body_names) if "foot" in name.lower()]
    foot_pos_visualizer = None
    if foot_body_indices and not args_cli.disable_foot_pos_vis:
        foot_pos_visualizer = make_foot_position_visualizer()
    arm_ee_pos_visualizer = None
    current_arm_ee_pos_visualizer = None
    arm_ee_body_indices = torch.empty(0, dtype=torch.long, device=sim.device)
    if first_motion.get("arm_ee_pos_w") is not None and not args_cli.disable_arm_ee_pos_vis:
        arm_ee_pos_visualizer = make_arm_ee_position_visualizer()
        arm_ee_body_names = parse_arm_ee_body_names(first_motion.get("arm_ee_frame"))
        if arm_ee_body_names:
            missing_arm_ee_bodies = [name for name in arm_ee_body_names if name not in robot.body_names]
            if missing_arm_ee_bodies:
                raise RuntimeError(
                    f"Robot is missing arm end-effector bodies {missing_arm_ee_bodies}. "
                    f"Available bodies: {robot.body_names}"
                )
            arm_ee_body_indices = torch.tensor(
                [robot.body_names.index(name) for name in arm_ee_body_names],
                dtype=torch.long,
                device=sim.device,
            )
            current_arm_ee_pos_visualizer = make_current_arm_ee_position_visualizer()
    gripper_body_indices = torch.empty(0, dtype=torch.long, device=sim.device)
    if args_cli.robot == "go2_d1" and all(name in robot.body_names for name in ("Link7_1", "Link7_2")):
        gripper_body_indices = torch.tensor(
            [robot.body_names.index("Link7_1"), robot.body_names.index("Link7_2")],
            dtype=torch.long,
            device=sim.device,
        )
    object_visualizer = None
    if any(motion.get("object_pos_w") is not None for motion in motions) and not args_cli.disable_object_vis:
        object_visualizer = make_reference_object_visualizer()

    print(f"[Replay] Loaded motions: {len(motions)}")
    print(f"[Replay] Robot asset: {args_cli.robot}")
    print(f"[Replay] Environments: {scene.num_envs}, spacing={args_cli.env_spacing:.2f} m")
    motion_fps_values = sorted({float(motion["fps"]) for motion in motions})
    if len(motion_fps_values) == 1:
        print(f"[Replay] FPS: {motion_fps_values[0]}")
    else:
        print(
            "[Replay] FPS values: "
            f"{motion_fps_values[0]}..{motion_fps_values[-1]} across {len(motion_fps_values)} values; "
            f"simulation dt uses first motion fps={first_motion['fps']}"
        )
    print(f"[Replay] Motion joint order -> {motion_joint_names}")
    print(f"[Replay] Robot joint order -> {robot.joint_names}")
    print(f"[Replay] Motion body order -> {motion_body_names}")
    print(f"[Replay] Root replay enabled: {not args_cli.disable_root_replay}")
    if args_cli.drive_arm_with_pd:
        print(
            "[Replay] Control mode: kinematic Go2/root + actuator-driven D1 arm; "
            "targets=50 Hz, physics/PD=200 Hz, target hold=4 physics steps"
        )
        print(f"[Replay] PD-driven joints -> {pd_arm_joint_names}")
        for actuator_name in ("arm_j1_j2", "arm_j3", "arm_j4", "arm_j5_j6", "gripper"):
            actuator_cfg = REPLAY_ROBOT_CFG.actuators[actuator_name]
            print(
                f"[Replay]   {actuator_name}: Kp={actuator_cfg.stiffness}, "
                f"Kd={actuator_cfg.damping}, effort_limit={actuator_cfg.effort_limit}"
            )
    else:
        print("[Replay] Control mode: direct kinematic joint-state replay")
    print(
        "[Replay] Foot position markers: "
        f"{'enabled' if foot_pos_visualizer is not None else 'disabled'}"
    )
    print(
        "[Replay] Arm EE markers: "
        f"{'reference=orange, simulator midpoint=blue' if current_arm_ee_pos_visualizer is not None else 'disabled'}"
    )
    print(
        "[Replay] Reference objects: "
        f"{'enabled (red NPZ geometry)' if object_visualizer is not None else 'disabled'}"
    )

    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    default_root_state = robot.data.default_root_state.clone()

    export_replayed_motion_npzs(
        sim,
        scene,
        robot,
        motions,
        mapped_joint_indices,
        default_joint_pos,
        default_joint_vel,
        default_root_state,
    )
    export_requested = (
        args_cli.export_replayed_motion_dir is not None
        or args_cli.export_corrected_arm_ee_dir is not None
    )
    if export_requested and args_cli.once:
        return

    motion_index = 0
    object_size_rng = np.random.default_rng(args_cli.object_size_seed)
    frame = 0
    current: dict[str, object] = {}
    oracle_report = None
    arm_ee_error_vectors: list[np.ndarray] = []
    arm_joint_error_vectors: list[np.ndarray] = []
    reported_open_gap = False
    reset_pd_arm_state = True

    def _activate_motion(new_motion_index: int) -> None:
        nonlocal motion_index, frame, current, oracle_report
        nonlocal arm_ee_error_vectors, arm_joint_error_vectors, reported_open_gap, reset_pd_arm_state
        motion_index = new_motion_index % len(motions)
        motion = motions[motion_index]

        joint_pos = torch.tensor(motion["joint_pos"], dtype=torch.float32, device=sim.device)
        joint_vel = torch.tensor(motion["joint_vel"], dtype=torch.float32, device=sim.device)
        body_pos_w = torch.tensor(motion["body_pos_w"], dtype=torch.float32, device=sim.device)
        body_quat_w = torch.tensor(motion["body_quat_w"], dtype=torch.float32, device=sim.device)
        body_lin_vel_w = torch.tensor(motion["body_lin_vel_w"], dtype=torch.float32, device=sim.device)
        body_ang_vel_w = torch.tensor(motion["body_ang_vel_w"], dtype=torch.float32, device=sim.device)
        arm_ee_pos_w = (
            torch.tensor(motion["arm_ee_pos_w"], dtype=torch.float32, device=sim.device)
            if motion["arm_ee_pos_w"] is not None
            else None
        )
        object_pos_w = (
            torch.tensor(motion["object_pos_w"], dtype=torch.float32, device=sim.device)
            if motion["object_pos_w"] is not None
            else None
        )
        object_quat_w = (
            torch.tensor(motion["object_quat_w"], dtype=torch.float32, device=sim.device)
            if motion["object_quat_w"] is not None
            else None
        )
        object_size = (
            torch.tensor(motion["object_size"], dtype=torch.float32, device=sim.device)
            if motion["object_size"] is not None
            else None
        )
        object_marker_indices = None
        if object_size is not None:
            shape_to_marker_index = {"cylinder": 0, "box": 1}
            object_shapes = motion["object_shapes"] or ["cylinder"] * object_size.shape[0]
            object_marker_indices = torch.tensor(
                [shape_to_marker_index[shape] for shape in object_shapes],
                dtype=torch.int64,
                device=sim.device,
            )
        object_size_scale = None
        object_mass = None
        if object_pos_w is not None and object_size is not None:
            size_low, size_high = args_cli.object_size_scale_range
            mass_low, mass_high = args_cli.object_mass_range
            if scene.num_envs > 1:
                size_scale_values = np.linspace(size_low, size_high, scene.num_envs)
                mass_values = np.linspace(mass_low, mass_high, scene.num_envs)
            else:
                size_scale_values = object_size_rng.uniform(size_low, size_high, size=1)
                mass_values = object_size_rng.uniform(mass_low, mass_high, size=1)
            object_size_scale = torch.tensor(
                size_scale_values,
                dtype=object_size.dtype,
                device=object_size.device,
            )
            object_mass = torch.tensor(mass_values, dtype=object_size.dtype, device=object_size.device)
            nominal_object_size = object_size
            object_size = nominal_object_size.unsqueeze(0) * object_size_scale[:, None, None]
            object_pos_w = object_pos_w[:, None, :, :].expand(-1, scene.num_envs, -1, -1).clone()
            local_grasp_face_offset = torch.zeros(
                (scene.num_envs, nominal_object_size.shape[0], 3),
                dtype=object_size.dtype,
                device=object_size.device,
            )
            local_grasp_face_offset[..., 0] = (
                0.5
                * nominal_object_size[None, :, 0]
                * (object_size_scale[:, None] - 1.0)
            )
            expanded_object_quat_w = object_quat_w[:, None, :, :].expand(
                -1, scene.num_envs, -1, -1
            )
            grasp_face_offset_w = math_utils.quat_apply(
                expanded_object_quat_w.reshape(-1, 4),
                local_grasp_face_offset.unsqueeze(0)
                .expand(object_quat_w.shape[0], -1, -1, -1)
                .reshape(-1, 3),
            ).reshape_as(object_pos_w)
            object_pos_w += grasp_face_offset_w
            grounded_height_offset = (
                0.5
                * nominal_object_size[None, :, 2]
                * (object_size_scale[:, None] - 1.0)
            )
            object_pos_w[..., 2] += grounded_height_offset.unsqueeze(0)
            object_marker_indices = object_marker_indices.unsqueeze(0).expand(scene.num_envs, -1).reshape(-1)
        gripper_joint_pos = (
            torch.tensor(motion["gripper_joint_pos"], dtype=torch.float32, device=sim.device)
            if motion["gripper_joint_pos"] is not None
            else None
        )
        gripper_joint_vel = (
            torch.tensor(motion["gripper_joint_vel"], dtype=torch.float32, device=sim.device)
            if motion["gripper_joint_vel"] is not None
            else None
        )
        gripper_scale = (
            gripper_replay_scale(gripper_joint_pos)
            if gripper_joint_pos is not None
            else None
        )
        gripper_joint_names = motion["gripper_joint_names"]
        gripper_joint_indices = torch.tensor(
            [robot_joint_name_to_index[name] for name in gripper_joint_names],
            dtype=torch.long,
            device=sim.device,
        )

        num_frames = int(joint_pos.shape[0])
        frame_start = max(0, min(args_cli.start_frame, num_frames - 1))
        frame_end = (
            num_frames - 1 if args_cli.end_frame < 0 else max(frame_start, min(args_cli.end_frame, num_frames - 1))
        )
        frame = frame_start
        arm_ee_error_vectors = []
        arm_joint_error_vectors = []
        reported_open_gap = False
        reset_pd_arm_state = True
        current = {
            "motion": motion,
            "joint_pos": joint_pos,
            "joint_vel": joint_vel,
            "body_pos_w": body_pos_w,
            "body_quat_w": body_quat_w,
            "body_lin_vel_w": body_lin_vel_w,
            "body_ang_vel_w": body_ang_vel_w,
            "arm_ee_pos_w": arm_ee_pos_w,
            "object_pos_w": object_pos_w,
            "object_quat_w": object_quat_w,
            "object_size": object_size,
            "object_marker_indices": object_marker_indices,
            "object_size_scale": object_size_scale,
            "object_mass": object_mass,
            "gripper_joint_pos": gripper_joint_pos,
            "gripper_joint_vel": gripper_joint_vel,
            "gripper_scale": gripper_scale,
            "gripper_joint_indices": gripper_joint_indices,
            "num_frames": num_frames,
            "frame_start": frame_start,
            "frame_end": frame_end,
            "origin_xy": body_pos_w[frame_start, base_body_index, 0:2].clone(),
        }

        oracle_report = None
        if args_cli.oracle_rewards:
            oracle_report = compute_oracle_reward_report(motion, frame_start=frame_start, frame_end=frame_end)
            print_oracle_reward_summary(oracle_report)

        print(
            f"[Replay] Active motion {motion_index + 1}/{len(motions)}: {motion['name']} "
            f"fps={motion['fps']}, frames={num_frames}, range=[{frame_start}, {frame_end}], file={motion['path']}"
        )
        if object_pos_w is not None:
            variant_summary = [
                {
                    "env": env_index,
                    "scale": round(float(object_size_scale[env_index]), 4),
                    "mass_g": round(1000.0 * float(object_mass[env_index]), 2),
                    "size_mm": [round(1000.0 * float(value), 2) for value in object_size[env_index, 0]],
                }
                for env_index in range(scene.num_envs)
            ]
            print(
                f"[Replay] Objects: {motion['object_names'] or ['object']} "
                f"shapes={motion['object_shapes'] or ['cylinder']} "
                f"variants_smallest_to_largest={variant_summary}"
            )
        if gripper_joint_pos is not None:
            if args_cli.binary_gripper:
                applied_max = float(args_cli.gripper_closed_position)
            else:
                applied_max = float(torch.amax(torch.abs(gripper_joint_pos * gripper_scale)))
            print(
                f"[Replay] Gripper trajectory: joints={gripper_joint_names}, "
                f"raw_range=[{float(gripper_joint_pos.min()):.7f}, {float(gripper_joint_pos.max()):.7f}] m, "
                f"starts_fully_open={bool(torch.all(gripper_joint_pos[0] == 0.0))}, "
                f"binary={args_cli.binary_gripper}, applied_max={applied_max:.7f} m"
            )

    def _next_motion() -> None:
        _activate_motion(motion_index + 1)

    def _previous_motion() -> None:
        _activate_motion(motion_index - 1)

    keyboard_switcher = None
    if len(motions) > 1 and not args_cli.disable_keyboard_motion_switch and not getattr(args_cli, "headless", False):
        try:
            keyboard_switcher = KeyboardMotionSwitcher(_next_motion, _previous_motion)
            print("[INFO] Keyboard motion switching enabled: N=next motion, P=previous motion.")
        except Exception as exc:
            print(f"[WARN] Could not enable keyboard motion switching: {exc}")

    _activate_motion(0)

    while simulation_app.is_running():
        joint_pos = current["joint_pos"]
        joint_vel = current["joint_vel"]
        body_pos_w = current["body_pos_w"]
        body_quat_w = current["body_quat_w"]
        body_lin_vel_w = current["body_lin_vel_w"]
        body_ang_vel_w = current["body_ang_vel_w"]
        arm_ee_pos_w = current["arm_ee_pos_w"]
        object_pos_w = current["object_pos_w"]
        object_quat_w = current["object_quat_w"]
        object_size = current["object_size"]
        object_marker_indices = current["object_marker_indices"]
        gripper_joint_pos = current["gripper_joint_pos"]
        gripper_joint_vel = current["gripper_joint_vel"]
        gripper_scale = current["gripper_scale"]
        gripper_joint_indices = current["gripper_joint_indices"]
        frame_start = int(current["frame_start"])
        frame_end = int(current["frame_end"])
        origin_xy = current["origin_xy"]

        replay_joint_pos = default_joint_pos.clone()
        replay_joint_vel = default_joint_vel.clone()
        replay_joint_pos[:, mapped_joint_indices] = joint_pos[frame].unsqueeze(0)
        replay_joint_vel[:, mapped_joint_indices] = joint_vel[frame].unsqueeze(0)
        if gripper_joint_pos is not None:
            gripper_target_pos, gripper_target_vel = gripper_replay_target_at(
                gripper_joint_pos,
                gripper_joint_vel,
                gripper_scale,
                frame,
            )
            replay_joint_pos[:, gripper_joint_indices] = gripper_target_pos
            replay_joint_vel[:, gripper_joint_indices] = gripper_target_vel
        root_state = None
        if not args_cli.disable_root_replay:
            root_state = default_root_state.clone()
            root_state[:, :3] = body_pos_w[frame, base_body_index].unsqueeze(0) + scene.env_origins
            if args_cli.zero_xy_origin:
                root_state[:, 0:2] -= origin_xy
            root_state[:, 3:7] = body_quat_w[frame, base_body_index].unsqueeze(0)
            root_state[:, 7:10] = body_lin_vel_w[frame, base_body_index].unsqueeze(0)
            root_state[:, 10:13] = body_ang_vel_w[frame, base_body_index].unsqueeze(0)

        if args_cli.drive_arm_with_pd:
            pd_arm_target = replay_joint_pos[:, pd_arm_joint_indices]
            # Match every actuator's desired pose to the replay state so the
            # kinematically pinned dog does not generate irrelevant leg effort.
            # Only the D1 arm/gripper states are allowed to evolve dynamically.
            robot.set_joint_position_target(replay_joint_pos)
            for _ in range(GO2_D1_PD_REPLAY_CONTROL_DECIMATION):
                if reset_pd_arm_state:
                    robot.write_joint_state_to_sim(replay_joint_pos, replay_joint_vel)
                    reset_pd_arm_state = False
                else:
                    robot.write_joint_state_to_sim(
                        replay_joint_pos[:, kinematic_joint_indices],
                        replay_joint_vel[:, kinematic_joint_indices],
                        joint_ids=kinematic_joint_indices,
                    )
                if root_state is not None:
                    robot.write_root_state_to_sim(root_state)
                scene.write_data_to_sim()
                sim.step(render=False)
                scene.update(sim.get_physics_dt())
            measured_pd_arm_pos = robot.data.joint_pos[0, pd_arm_joint_indices]
            arm_joint_error_vectors.append(
                (pd_arm_target[0] - measured_pd_arm_pos).detach().cpu().numpy()
            )
        else:
            robot.write_joint_state_to_sim(replay_joint_pos, replay_joint_vel)
            if root_state is not None:
                robot.write_root_state_to_sim(root_state)
            scene.write_data_to_sim()
            sim.forward()
            scene.update(sim.get_physics_dt())

        focus_gripper = args_cli.camera_focus == "gripper" or (
            args_cli.camera_focus == "auto"
            and gripper_joint_pos is not None
            and gripper_body_indices.numel() == 2
        )
        if focus_gripper:
            look_at = robot.data.body_pos_w[:, gripper_body_indices].mean(dim=(0, 1)).detach().cpu().numpy()
            if scene.num_envs == 1:
                camera_offset = np.array([0.75, 0.75, 0.25])
            else:
                camera_distance = max(3.0, args_cli.env_spacing * (np.sqrt(scene.num_envs) + 1.0))
                camera_offset = np.array([camera_distance, camera_distance, 0.5 * camera_distance])
            sim.set_camera_view(look_at + camera_offset, look_at)
        elif not args_cli.disable_root_replay:
            look_at = robot.data.root_pos_w.mean(dim=0).detach().cpu().numpy()
            camera_distance = max(3.0, args_cli.env_spacing * (np.sqrt(scene.num_envs) + 1.0))
            sim.set_camera_view(
                look_at + np.array([camera_distance, camera_distance, 0.5 * camera_distance]),
                look_at,
            )

        if (
            not reported_open_gap
            and gripper_joint_pos is not None
            and gripper_body_indices.numel() == 2
            and bool(torch.all(gripper_joint_pos[frame] == 0.0))
        ):
            gripper_body_pos = robot.data.body_pos_w[0, gripper_body_indices]
            center_separation = torch.linalg.vector_norm(
                gripper_body_pos[1] - gripper_body_pos[0]
            ).item()
            inner_pad_opening = max(
                center_separation - GO2_D1_TOTAL_INNER_PAD_INSET_M,
                0.0,
            )
            print(
                f"[Replay] Fully-open frame {frame}: q=[0, 0] m, "
                f"finger-center separation={1000.0 * center_separation:.3f} mm, "
                f"visual inner-pad opening={1000.0 * inner_pad_opening:.3f} mm"
            )
            reported_open_gap = True

        if foot_pos_visualizer is not None:
            foot_pos_w = (
                body_pos_w[frame, foot_body_indices].unsqueeze(0) + scene.env_origins[:, None, :]
            ).reshape(-1, 3)
            if args_cli.zero_xy_origin:
                foot_pos_w[:, 0:2] -= origin_xy
            foot_pos_visualizer.visualize(foot_pos_w)

        if arm_ee_pos_visualizer is not None and arm_ee_pos_w is not None:
            arm_ee_pos = arm_ee_pos_w[frame].unsqueeze(0) + scene.env_origins
            if args_cli.zero_xy_origin:
                arm_ee_pos[:, 0:2] -= origin_xy
            arm_ee_pos_visualizer.visualize(arm_ee_pos)
            if current_arm_ee_pos_visualizer is not None:
                current_arm_ee_pos = robot.data.body_pos_w[:, arm_ee_body_indices, :].mean(dim=1)
                current_arm_ee_pos_visualizer.visualize(current_arm_ee_pos)
                arm_ee_error_vectors.append(
                    (arm_ee_pos[0] - current_arm_ee_pos[0]).detach().cpu().numpy()
                )

        if object_visualizer is not None:
            if object_pos_w is None or object_quat_w is None or object_size is None:
                object_visualizer.set_visibility(False)
            else:
                object_visualizer.set_visibility(not bool(getattr(args_cli, "headless", False)))
                object_pos = object_pos_w[frame] + scene.env_origins[:, None, :]
                if args_cli.zero_xy_origin:
                    object_pos[..., 0:2] -= origin_xy
                object_orientations = (
                    object_quat_w[frame]
                    .unsqueeze(0)
                    .expand(scene.num_envs, -1, -1)
                    .reshape(-1, 4)
                )
                object_visualizer.visualize(
                    translations=object_pos.reshape(-1, 3),
                    orientations=object_orientations,
                    scales=object_size.reshape(-1, 3),
                    marker_indices=object_marker_indices,
                )

        if not getattr(args_cli, "headless", False):
            sim.render()

        if oracle_report is not None and args_cli.oracle_print_interval > 0:
            oracle_frame = frame - frame_start
            if oracle_frame == 0 or oracle_frame % args_cli.oracle_print_interval == 0:
                oracle_total = oracle_report["total"][oracle_frame].item()
                oracle_terms = oracle_report["terms"]
                assert isinstance(oracle_terms, dict)
                terms_str = ", ".join(
                    f"{name}={values[oracle_frame].item():.4f}" for name, values in oracle_terms.items()
                )
                print(f"[Oracle][frame={frame}] total_upper_bound={oracle_total:.4f} | {terms_str}")

        if frame >= frame_end:
            if args_cli.once:
                break
            print_arm_ee_error_summary(arm_ee_error_vectors, suffix=" over pass")
            print_arm_joint_error_summary(pd_arm_joint_names, arm_joint_error_vectors, suffix=" over pass")
            frame = frame_start
            arm_ee_error_vectors = []
            arm_joint_error_vectors = []
            reset_pd_arm_state = True
        else:
            frame = min(frame + args_cli.frame_stride, frame_end)

    if keyboard_switcher is not None:
        keyboard_switcher.close()
    if foot_pos_visualizer is not None:
        foot_pos_visualizer.set_visibility(False)
    if arm_ee_pos_visualizer is not None:
        arm_ee_pos_visualizer.set_visibility(False)
    if current_arm_ee_pos_visualizer is not None:
        current_arm_ee_pos_visualizer.set_visibility(False)
    if object_visualizer is not None:
        object_visualizer.set_visibility(False)
    print_arm_ee_error_summary(arm_ee_error_vectors)
    print_arm_joint_error_summary(pd_arm_joint_names, arm_joint_error_vectors)


def main() -> None:
    if args_cli.frame_stride < 1:
        raise ValueError(f"--frame-stride must be at least 1, got {args_cli.frame_stride}.")
    if args_cli.num_envs < 1:
        raise ValueError(f"--num-envs must be at least 1, got {args_cli.num_envs}.")
    if args_cli.env_spacing <= 0.0:
        raise ValueError(f"--env-spacing must be positive, got {args_cli.env_spacing}.")
    if args_cli.drive_arm_with_pd and args_cli.robot != "go2_d1":
        raise ValueError("--drive-arm-with-pd requires --robot go2_d1.")
    if args_cli.drive_arm_with_pd and args_cli.frame_stride != 1:
        raise ValueError("--drive-arm-with-pd requires --frame-stride 1 to preserve the 50 Hz target stream.")
    if args_cli.drive_arm_with_pd and (
        args_cli.export_replayed_motion_dir is not None or args_cli.export_corrected_arm_ee_dir is not None
    ):
        raise ValueError("--drive-arm-with-pd cannot be combined with replay export flags.")
    if args_cli.gripper_position_scale <= 0.0:
        raise ValueError(
            f"--gripper-position-scale must be positive, got {args_cli.gripper_position_scale}."
        )
    object_size_scale_low, object_size_scale_high = args_cli.object_size_scale_range
    if object_size_scale_low <= 0.0 or object_size_scale_high < object_size_scale_low:
        raise ValueError(
            "--object-size-scale-range must satisfy 0 < MIN <= MAX, "
            f"got {args_cli.object_size_scale_range}."
        )
    object_mass_low, object_mass_high = args_cli.object_mass_range
    if object_mass_low <= 0.0 or object_mass_high < object_mass_low:
        raise ValueError(
            "--object-mass-range must satisfy 0 < MIN_KG <= MAX_KG, "
            f"got {args_cli.object_mass_range}."
        )
    if args_cli.gripper_closed_position is not None and not (
        0.0 < args_cli.gripper_closed_position <= 0.033
    ):
        raise ValueError(
            "--gripper-closed-position must be in the D1 joint range (0, 0.033] m, "
            f"got {args_cli.gripper_closed_position}."
        )
    if args_cli.binary_gripper and args_cli.gripper_closed_position is None:
        raise ValueError("--binary-gripper requires --gripper-closed-position.")
    if args_cli.export_replayed_motion_dir is not None and args_cli.export_corrected_arm_ee_dir is not None:
        raise ValueError("Choose only one of --export-replayed-motion-dir and --export-corrected-arm-ee-dir.")
    motions = load_replay_motions(args_cli.input)
    if args_cli.drive_arm_with_pd:
        motion_fps_values = {float(motion["fps"]) for motion in motions}
        if motion_fps_values != {GO2_D1_PD_REPLAY_FREQUENCY_HZ}:
            raise ValueError(
                "--drive-arm-with-pd requires 50 Hz motion clips; "
                f"got fps values {sorted(motion_fps_values)}."
            )
        sim_dt = GO2_D1_PD_REPLAY_SIM_DT
    else:
        sim_dt = 1.0 / float(motions[0]["fps"])
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=sim_dt)
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(
        ReplayGo2NpzSceneCfg(num_envs=args_cli.num_envs, env_spacing=args_cli.env_spacing)
    )
    sim.reset()
    run_simulator(sim, scene, motions)


if __name__ == "__main__":
    main()
    simulation_app.close()
