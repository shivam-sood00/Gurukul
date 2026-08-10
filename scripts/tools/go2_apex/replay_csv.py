"""Replay APEX-style Go2 CSV motion directly in Isaac Lab for joint-order verification.

Example:
    python scripts/tools/go2_apex/replay_csv.py \
        --input source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/imitation_data/animal_mocap/go2_retarget_trot.csv \
        --csv-leg-order FL,FR,RL,RR \
        --fps 50

Notes:
    Root state replay (height + base position + orientation + base velocity) is enabled by default.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from isaaclab.app import AppLauncher


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", "-i", type=Path, required=True, help="Path to APEX CSV file.")
    parser.add_argument(
        "--csv-leg-order",
        type=str,
        default="FL,FR,RL,RR",
        help=(
            "Leg labels for CSV groups base1..base4, comma-separated. "
            "Example: FL,FR,RL,RR or FR,FL,RR,RL."
        ),
    )
    parser.add_argument("--fps", type=float, default=50.0, help="Replay frequency of CSV frames.")
    parser.add_argument(
        "--disable-root-replay",
        action="store_true",
        help="Disable root replay (by default root state is replayed from CSV).",
    )
    parser.add_argument(
        "--zero-xy-origin",
        action="store_true",
        help="When replaying root, subtract first-frame XY to keep the trajectory near scene origin.",
    )
    parser.add_argument("--start-frame", type=int, default=0, help="First frame index.")
    parser.add_argument("--end-frame", type=int, default=-1, help="Last frame index (inclusive), -1 means full clip.")
    parser.add_argument("--once", action="store_true", help="Play one pass and exit.")
    AppLauncher.add_app_launcher_args(parser)
    return parser.parse_args()


args_cli = parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from Gurukul.assets.unitree import UNITREE_GO2_CFG

CANONICAL_LEGS = ("FL", "FR", "RL", "RR")
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


def finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    if values.shape[0] < 2:
        return out
    out[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    out[0] = (values[1] - values[0]) / dt
    out[-1] = (values[-1] - values[-2]) / dt
    return out


def _load_csv_matrix(path: Path) -> tuple[np.ndarray, list[str] | None]:
    if not path.is_file():
        raise FileNotFoundError(f"CSV not found: {path}")

    rows: list[list[float]] = []
    skipped_bad_width = 0
    skipped_bad_parse = 0
    expected_cols: int | None = None
    header: list[str] | None = None

    with path.open("r", encoding="utf-8") as file:
        first_line = file.readline()
        if first_line == "":
            raise ValueError(f"CSV file is empty: {path}")
        first_tokens = [token.strip() for token in first_line.strip().split(",")]

        has_header = False
        if first_tokens and first_tokens[0] != "":
            try:
                float(first_tokens[0])
            except ValueError:
                has_header = True

        if has_header:
            header = first_tokens
            expected_cols = len(first_tokens)
        else:
            expected_cols = len(first_tokens)
            try:
                rows.append([float(token) for token in first_tokens])
            except ValueError:
                skipped_bad_parse += 1

        for line in file:
            stripped = line.strip()
            if not stripped:
                continue
            tokens = [token.strip() for token in stripped.split(",")]
            if len(tokens) != expected_cols:
                skipped_bad_width += 1
                continue
            try:
                rows.append([float(token) for token in tokens])
            except ValueError:
                skipped_bad_parse += 1

    if len(rows) == 0:
        raise ValueError(f"No valid numeric rows found in CSV: {path}")

    if skipped_bad_width or skipped_bad_parse:
        print(
            f"[WARN] Skipped malformed rows in {path.name}: "
            f"bad_width={skipped_bad_width}, bad_parse={skipped_bad_parse}"
        )

    return np.asarray(rows, dtype=np.float32), header


def _parse_csv_leg_order(order_str: str) -> list[str]:
    order = [token.strip().upper() for token in order_str.split(",") if token.strip()]
    if len(order) != 4:
        raise ValueError(
            f"Expected 4 leg labels in --csv-leg-order, got {len(order)} from '{order_str}'."
        )
    if set(order) != set(CANONICAL_LEGS):
        raise ValueError(
            f"--csv-leg-order must be a permutation of {CANONICAL_LEGS}, got {order}."
        )
    return order


def load_go2_motion_from_apex_csv(path: Path, fps: float, csv_leg_order: list[str]) -> dict[str, np.ndarray]:
    data, header = _load_csv_matrix(path)
    if data.shape[1] < 40:
        raise ValueError(
            f"Expected at least 40 columns for Go2 APEX-style CSV, got {data.shape[1]} from: {path}"
        )

    dt = 1.0 / fps
    leg_to_csv_idx = {leg: i for i, leg in enumerate(csv_leg_order)}

    joint_pos_csv = data[:, 6:18].reshape(-1, 4, 3)
    joint_pos = np.zeros((data.shape[0], 12), dtype=np.float32)
    for canonical_leg_idx, leg in enumerate(CANONICAL_LEGS):
        src_leg_idx = leg_to_csv_idx[leg]
        joint_pos[:, 3 * canonical_leg_idx : 3 * canonical_leg_idx + 3] = joint_pos_csv[:, src_leg_idx, :]

    if data.shape[1] >= 64:
        joint_vel_csv = data[:, 52:64].reshape(-1, 4, 3)
        joint_vel = np.zeros_like(joint_pos, dtype=np.float32)
        for canonical_leg_idx, leg in enumerate(CANONICAL_LEGS):
            src_leg_idx = leg_to_csv_idx[leg]
            joint_vel[:, 3 * canonical_leg_idx : 3 * canonical_leg_idx + 3] = joint_vel_csv[:, src_leg_idx, :]
    else:
        joint_vel = finite_difference(joint_pos, dt)

    base_pos = np.concatenate([data[:, 34:36], data[:, 21:22]], axis=1).astype(np.float32)
    base_quat_xyzw = data[:, 36:40].astype(np.float32)
    # Isaac Lab expects quaternions in wxyz.
    base_quat_wxyz = np.concatenate([base_quat_xyzw[:, 3:4], base_quat_xyzw[:, 0:3]], axis=1)
    base_quat_wxyz /= np.linalg.norm(base_quat_wxyz, axis=1, keepdims=True).clip(min=1.0e-8)

    return {
        "joint_pos": joint_pos.astype(np.float32),
        "joint_vel": joint_vel.astype(np.float32),
        "base_pos": base_pos.astype(np.float32),
        "base_quat_wxyz": base_quat_wxyz.astype(np.float32),
        "base_lin_vel": data[:, 0:3].astype(np.float32),
        "base_ang_vel": data[:, 3:6].astype(np.float32),
        "num_frames": np.array([data.shape[0]], dtype=np.int64),
        "num_cols": np.array([data.shape[1]], dtype=np.int64),
        "has_header": np.array([1 if header is not None else 0], dtype=np.int64),
    }


@configclass
class ReplayGo2CsvSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())
    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=1000.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )
    robot: ArticulationCfg = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


def run_simulator(sim: SimulationContext, scene: InteractiveScene, motion: dict[str, np.ndarray]):
    robot: Articulation = scene["robot"]

    joint_name_to_index = {name: i for i, name in enumerate(robot.joint_names)}
    missing_joint_names = [name for name in CANONICAL_JOINT_NAMES if name not in joint_name_to_index]
    if missing_joint_names:
        raise RuntimeError(
            f"Robot is missing expected leg joints: {missing_joint_names}\n"
            f"Robot joint names: {robot.joint_names}"
        )
    mapped_joint_indices = torch.tensor(
        [joint_name_to_index[name] for name in CANONICAL_JOINT_NAMES], dtype=torch.long, device=sim.device
    )

    joint_pos = torch.tensor(motion["joint_pos"], dtype=torch.float32, device=sim.device)
    joint_vel = torch.tensor(motion["joint_vel"], dtype=torch.float32, device=sim.device)
    base_pos = torch.tensor(motion["base_pos"], dtype=torch.float32, device=sim.device)
    base_quat = torch.tensor(motion["base_quat_wxyz"], dtype=torch.float32, device=sim.device)
    base_lin_vel = torch.tensor(motion["base_lin_vel"], dtype=torch.float32, device=sim.device)
    base_ang_vel = torch.tensor(motion["base_ang_vel"], dtype=torch.float32, device=sim.device)

    num_frames = int(motion["num_frames"][0])
    frame_start = max(0, min(args_cli.start_frame, num_frames - 1))
    frame_end = num_frames - 1 if args_cli.end_frame < 0 else max(frame_start, min(args_cli.end_frame, num_frames - 1))

    print(f"[Replay] File: {args_cli.input}")
    print(f"[Replay] CSV columns: {int(motion['num_cols'][0])}, header: {bool(motion['has_header'][0])}")
    print(f"[Replay] Frames: {num_frames}, fps: {args_cli.fps}, range: [{frame_start}, {frame_end}]")
    print(f"[Replay] CSV base1..base4 -> {args_cli.csv_leg_order}")
    print(f"[Replay] Canonical leg joint order -> {CANONICAL_JOINT_NAMES}")
    print(f"[Replay] Robot joint order -> {robot.joint_names}")
    print(
        "[Replay] Base state columns: "
        "lin_vel=(0:3), ang_vel=(3:6), height=(21), xy_pos=(34:36), quat_xyzw=(36:40->wxyz)"
    )
    print(f"[Replay] Root replay enabled: {not args_cli.disable_root_replay}")

    default_joint_pos = robot.data.default_joint_pos.clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    default_root_state = robot.data.default_root_state.clone()
    origin_xy = base_pos[frame_start, 0:2].clone()

    frame = frame_start
    while simulation_app.is_running():
        replay_joint_pos = default_joint_pos.clone()
        replay_joint_vel = default_joint_vel.clone()
        replay_joint_pos[:, mapped_joint_indices] = joint_pos[frame].unsqueeze(0)
        replay_joint_vel[:, mapped_joint_indices] = joint_vel[frame].unsqueeze(0)
        robot.write_joint_state_to_sim(replay_joint_pos, replay_joint_vel)

        if not args_cli.disable_root_replay:
            root_state = default_root_state.clone()
            root_state[:, :3] = base_pos[frame].unsqueeze(0) + scene.env_origins
            if args_cli.zero_xy_origin:
                root_state[:, 0:2] -= origin_xy
            root_state[:, 3:7] = base_quat[frame].unsqueeze(0)
            root_state[:, 7:10] = base_lin_vel[frame].unsqueeze(0)
            root_state[:, 10:13] = base_ang_vel[frame].unsqueeze(0)
            robot.write_root_state_to_sim(root_state)

            look_at = root_state[0, :3].detach().cpu().numpy()
            sim.set_camera_view(look_at + np.array([2.0, 2.0, 0.6]), look_at)

        scene.write_data_to_sim()
        sim.render()
        scene.update(sim.get_physics_dt())

        if frame >= frame_end:
            if args_cli.once:
                break
            frame = frame_start
        else:
            frame += 1


def main():
    csv_leg_order = _parse_csv_leg_order(args_cli.csv_leg_order)
    motion = load_go2_motion_from_apex_csv(args_cli.input, args_cli.fps, csv_leg_order)

    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device, dt=1.0 / args_cli.fps)
    sim = SimulationContext(sim_cfg)
    scene = InteractiveScene(ReplayGo2CsvSceneCfg(num_envs=1, env_spacing=2.0))
    sim.reset()
    run_simulator(sim, scene, motion)


if __name__ == "__main__":
    main()
    simulation_app.close()
