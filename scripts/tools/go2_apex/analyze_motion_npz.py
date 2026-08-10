#!/usr/bin/env python3
"""Analyze Go2 APEX motion NPZ files for visual and numerical data issues.

The tool scans one or more NPZ files, computes continuity diagnostics, and
writes per-clip plots plus CSV/JSON summaries. It is intended for offline
motion-data review before training or Isaac replay.

Example:
    python scripts/tools/go2_apex/analyze_motion_npz.py \
        --input source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/npz/corss_morpho_new \
        --output-dir /tmp/go2_motion_audit
"""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


DEFAULT_INPUT = (
    Path(__file__).resolve().parents[3]
    / "source"
    / "Gurukul"
    / "Gurukul"
    / "tasks"
    / "manager_based"
    / "go2_apex"
    / "config"
    / "go2"
    / "motion"
    / "npz"
    / "corss_morpho_new"
)


@dataclass(frozen=True)
class Issue:
    severity: str
    metric: str
    value: float
    threshold: float
    frame: int
    detail: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        nargs="+",
        default=[str(DEFAULT_INPUT)],
        help="Motion NPZ file(s), directories, or glob patterns. Defaults to corss_morpho_new.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("logs/motion_audit/corss_morpho_new"),
        help="Directory for plots and CSV/JSON reports.",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=12,
        help="Number of worst frames to write per metric in each per-clip event CSV.",
    )
    parser.add_argument(
        "--plot-format",
        choices=("png", "svg"),
        default="png",
        help="Per-clip plot format.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=150,
        help="DPI for PNG plots.",
    )
    parser.add_argument(
        "--ground-tolerance",
        type=float,
        default=0.03,
        help="Warn when minimum foot height is below -ground-tolerance.",
    )
    parser.add_argument(
        "--base-speed-warn",
        type=float,
        default=5.0,
        help="Warn when base horizontal speed exceeds this value in m/s.",
    )
    parser.add_argument(
        "--base-acc-warn",
        type=float,
        default=30.0,
        help="Warn when base horizontal acceleration exceeds this value in m/s^2.",
    )
    parser.add_argument(
        "--joint-vel-warn",
        type=float,
        default=30.0,
        help="Warn when max absolute joint velocity exceeds this value in rad/s.",
    )
    parser.add_argument(
        "--joint-acc-warn",
        type=float,
        default=300.0,
        help="Warn when max absolute joint acceleration exceeds this value in rad/s^2.",
    )
    parser.add_argument(
        "--velocity-mismatch-warn",
        type=float,
        default=1.0,
        help="Warn when finite-difference base velocity and stored base velocity differ by this value in m/s.",
    )
    return parser.parse_args()


def resolve_inputs(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for source in inputs:
        source_path = Path(source).expanduser()
        if source_path.is_file():
            matches = [source_path]
        elif source_path.is_dir():
            matches = sorted(source_path.rglob("*.npz"))
        else:
            matches = [Path(match) for match in sorted(glob.glob(str(source_path), recursive=True))]
        paths.extend(path.resolve() for path in matches if path.is_file() and path.suffix == ".npz")
    deduped = list(dict.fromkeys(paths))
    if not deduped:
        raise FileNotFoundError(f"No NPZ motion files found from input(s): {inputs}")
    return deduped


def finite_difference(values: np.ndarray, dt: float) -> np.ndarray:
    out = np.zeros_like(values, dtype=np.float32)
    if values.shape[0] < 2:
        return out
    out[1:-1] = (values[2:] - values[:-2]) / (2.0 * dt)
    out[0] = (values[1] - values[0]) / dt
    out[-1] = (values[-1] - values[-2]) / dt
    return out


def robust_threshold(values: np.ndarray, floor: float, sigma: float = 8.0) -> float:
    flat = np.asarray(values, dtype=np.float64).reshape(-1)
    if flat.size == 0:
        return floor
    median = float(np.median(flat))
    mad = float(np.median(np.abs(flat - median)))
    robust_sigma = 1.4826 * mad
    if robust_sigma <= 1.0e-9:
        return floor
    return max(floor, median + sigma * robust_sigma)


def load_motion(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        required = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
        missing = [key for key in required if key not in data]
        if missing:
            raise ValueError(f"{path} is missing required keys: {missing}")
        motion = {key: np.asarray(data[key], dtype=np.float32) for key in required}
        motion["fps"] = float(np.asarray(data["fps"]).reshape(-1)[0]) if "fps" in data else 50.0
        motion["joint_names"] = [str(name) for name in data["joint_names"].tolist()] if "joint_names" in data else []
        motion["body_names"] = [str(name) for name in data["body_names"].tolist()] if "body_names" in data else []
        if "command_lin_vel_xy" in data:
            motion["command_lin_vel_xy"] = np.asarray(data["command_lin_vel_xy"], dtype=np.float32)
        if "command_ang_vel_z" in data:
            motion["command_ang_vel_z"] = np.asarray(data["command_ang_vel_z"], dtype=np.float32)
    return motion


def top_frames(values: np.ndarray, top_k: int) -> list[tuple[int, float]]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return []
    count = min(top_k, values.size)
    indices = np.argpartition(values, -count)[-count:]
    return [(int(index), float(values[index])) for index in indices[np.argsort(values[indices])[::-1]]]


def add_issue(
    issues: list[Issue],
    severity: str,
    metric: str,
    values: np.ndarray,
    threshold: float,
    detail: str,
) -> None:
    if values.size == 0:
        return
    frame, value = top_frames(values, 1)[0]
    if value > threshold:
        issues.append(Issue(severity, metric, value, threshold, frame, detail))


def analyze_motion(path: Path, motion: dict[str, Any], args: argparse.Namespace) -> tuple[dict[str, Any], list[Issue], dict[str, np.ndarray]]:
    fps = float(motion["fps"])
    dt = 1.0 / fps
    joint_pos = motion["joint_pos"]
    joint_vel = motion["joint_vel"]
    body_pos_w = motion["body_pos_w"]
    body_lin_vel_w = motion["body_lin_vel_w"]
    body_ang_vel_w = motion["body_ang_vel_w"]
    body_names = motion["body_names"]

    if joint_pos.ndim != 2 or joint_vel.shape != joint_pos.shape:
        raise ValueError(f"{path} has invalid joint arrays: {joint_pos.shape}, {joint_vel.shape}")
    if body_pos_w.ndim != 3 or body_lin_vel_w.shape[:2] != body_pos_w.shape[:2]:
        raise ValueError(f"{path} has invalid body arrays.")

    base_idx = body_names.index("base") if "base" in body_names else 0
    foot_idx = [idx for idx, name in enumerate(body_names) if "foot" in name.lower()]

    base_pos = body_pos_w[:, base_idx]
    base_lin_vel = body_lin_vel_w[:, base_idx]
    base_ang_vel = body_ang_vel_w[:, base_idx]
    fd_base_lin_vel = finite_difference(base_pos, dt)
    base_speed_xy = np.linalg.norm(base_lin_vel[:, :2], axis=1)
    fd_base_speed_xy = np.linalg.norm(fd_base_lin_vel[:, :2], axis=1)
    base_acc_xy = np.linalg.norm(finite_difference(base_lin_vel[:, :2], dt), axis=1)
    base_vel_mismatch = np.linalg.norm(fd_base_lin_vel - base_lin_vel, axis=1)
    joint_delta = np.max(np.abs(np.diff(joint_pos, axis=0)), axis=1) if joint_pos.shape[0] > 1 else np.zeros(1)
    joint_acc = finite_difference(joint_vel, dt)
    max_abs_joint_vel = np.max(np.abs(joint_vel), axis=1)
    max_abs_joint_acc = np.max(np.abs(joint_acc), axis=1)
    quat_norm_error = np.abs(np.linalg.norm(motion["body_quat_w"][:, base_idx], axis=1) - 1.0)

    if foot_idx:
        foot_z = body_pos_w[:, foot_idx, 2]
        min_foot_z = np.min(foot_z, axis=1)
        foot_speed = np.linalg.norm(body_lin_vel_w[:, foot_idx], axis=2)
        max_foot_speed = np.max(foot_speed, axis=1)
    else:
        min_foot_z = np.full(joint_pos.shape[0], np.nan, dtype=np.float32)
        max_foot_speed = np.zeros(joint_pos.shape[0], dtype=np.float32)

    command_lin = motion.get("command_lin_vel_xy")
    if command_lin is not None:
        command_speed_xy = np.linalg.norm(command_lin, axis=1)
        command_error_xy = np.abs(command_speed_xy - base_speed_xy)
    else:
        command_speed_xy = np.zeros(joint_pos.shape[0], dtype=np.float32)
        command_error_xy = np.zeros(joint_pos.shape[0], dtype=np.float32)

    metrics = {
        "base_speed_xy": base_speed_xy,
        "fd_base_speed_xy": fd_base_speed_xy,
        "base_acc_xy": base_acc_xy,
        "base_vel_mismatch": base_vel_mismatch,
        "joint_delta": np.pad(joint_delta, (1, 0), mode="constant")[: joint_pos.shape[0]],
        "max_abs_joint_vel": max_abs_joint_vel,
        "max_abs_joint_acc": max_abs_joint_acc,
        "min_foot_z": min_foot_z,
        "max_foot_speed": max_foot_speed,
        "quat_norm_error": quat_norm_error,
        "command_speed_xy": command_speed_xy,
        "command_error_xy": command_error_xy,
        "base_height": base_pos[:, 2],
        "base_yaw_rate": base_ang_vel[:, 2],
    }

    issues: list[Issue] = []
    add_issue(issues, "warn", "base_speed_xy", base_speed_xy, args.base_speed_warn, "high base horizontal speed")
    add_issue(issues, "warn", "base_acc_xy", base_acc_xy, args.base_acc_warn, "high base horizontal acceleration")
    add_issue(
        issues,
        "warn",
        "max_abs_joint_vel",
        max_abs_joint_vel,
        args.joint_vel_warn,
        "high joint velocity",
    )
    add_issue(
        issues,
        "warn",
        "max_abs_joint_acc",
        max_abs_joint_acc,
        args.joint_acc_warn,
        "high joint acceleration",
    )
    add_issue(
        issues,
        "warn",
        "base_vel_mismatch",
        base_vel_mismatch,
        args.velocity_mismatch_warn,
        "stored base velocity disagrees with finite difference of base position",
    )
    add_issue(issues, "warn", "quat_norm_error", quat_norm_error, 1.0e-3, "base quaternion norm is not close to one")

    finite_min_foot = min_foot_z[np.isfinite(min_foot_z)]
    if finite_min_foot.size > 0 and float(np.min(finite_min_foot)) < -args.ground_tolerance:
        frame = int(np.argmin(min_foot_z))
        value = float(min_foot_z[frame])
        issues.append(
            Issue(
                "warn",
                "min_foot_z",
                value,
                -args.ground_tolerance,
                frame,
                "foot position below ground; consider reconverting with --ground-align-foot-height",
            )
        )

    for metric_name, floor, detail in (
        ("joint_delta", 0.35, "robust outlier in per-frame joint-position delta"),
        ("base_acc_xy", args.base_acc_warn, "robust outlier in base acceleration"),
        ("max_abs_joint_acc", args.joint_acc_warn, "robust outlier in joint acceleration"),
        ("max_foot_speed", 8.0, "robust outlier in foot speed"),
    ):
        threshold = robust_threshold(metrics[metric_name], floor=floor)
        add_issue(issues, "outlier", metric_name, metrics[metric_name], threshold, detail)

    issues.sort(key=lambda issue: (0 if issue.severity == "warn" else 1, -abs(issue.value)))
    summary = {
        "file": str(path),
        "name": path.name,
        "frames": int(joint_pos.shape[0]),
        "fps": fps,
        "duration_s": float((joint_pos.shape[0] - 1) / fps) if joint_pos.shape[0] > 0 else 0.0,
        "joint_count": int(joint_pos.shape[1]),
        "body_count": int(body_pos_w.shape[1]),
        "base_speed_xy_max": float(np.max(base_speed_xy)),
        "base_speed_xy_p99": float(np.percentile(base_speed_xy, 99.0)),
        "base_acc_xy_max": float(np.max(base_acc_xy)),
        "joint_vel_abs_max": float(np.max(max_abs_joint_vel)),
        "joint_acc_abs_max": float(np.max(max_abs_joint_acc)),
        "joint_delta_abs_max": float(np.max(metrics["joint_delta"])),
        "foot_z_min": float(np.nanmin(min_foot_z)) if finite_min_foot.size > 0 else math.nan,
        "foot_z_max": float(np.nanmax(min_foot_z)) if finite_min_foot.size > 0 else math.nan,
        "base_height_min": float(np.min(base_pos[:, 2])),
        "base_height_max": float(np.max(base_pos[:, 2])),
        "base_vel_mismatch_max": float(np.max(base_vel_mismatch)),
        "quat_norm_error_max": float(np.max(quat_norm_error)),
        "command_error_xy_p95": float(np.percentile(command_error_xy, 95.0)) if command_lin is not None else math.nan,
        "issue_count": len(issues),
        "worst_issue": issues[0].metric if issues else "",
    }
    return summary, issues, metrics


def write_event_csv(path: Path, metrics: dict[str, np.ndarray], issues: list[Issue], top_k: int) -> None:
    rows = []
    interesting = ["base_acc_xy", "base_vel_mismatch", "max_abs_joint_acc", "joint_delta", "max_foot_speed", "min_foot_z"]
    for metric_name in interesting:
        values = metrics[metric_name]
        ranked_values = -values if metric_name == "min_foot_z" else values
        for rank, (frame, _) in enumerate(top_frames(ranked_values, top_k), start=1):
            rows.append(
                {
                    "metric": metric_name,
                    "rank": rank,
                    "frame": frame,
                    "value": float(values[frame]),
                }
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["metric", "rank", "frame", "value"])
        writer.writeheader()
        writer.writerows(rows)

    issue_path = path.with_name(path.stem + "_issues.csv")
    with issue_path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=["severity", "metric", "frame", "value", "threshold", "detail"])
        writer.writeheader()
        for issue in issues:
            writer.writerow(issue.__dict__)


def plot_motion(path: Path, motion: dict[str, Any], metrics: dict[str, np.ndarray], output_path: Path, dpi: int) -> None:
    fps = float(motion["fps"])
    frame_count = int(motion["joint_pos"].shape[0])
    time = np.arange(frame_count, dtype=np.float32) / fps
    body_names = motion["body_names"]
    base_idx = body_names.index("base") if "base" in body_names else 0
    foot_idx = [idx for idx, name in enumerate(body_names) if "foot" in name.lower()]
    base_pos = motion["body_pos_w"][:, base_idx]
    joint_pos = motion["joint_pos"]
    joint_vel = motion["joint_vel"]

    fig, axes = plt.subplots(3, 2, figsize=(15, 12), constrained_layout=True)
    fig.suptitle(path.name, fontsize=14)

    scatter = axes[0, 0].scatter(base_pos[:, 0], base_pos[:, 1], c=metrics["base_speed_xy"], s=8, cmap="viridis")
    axes[0, 0].set_title("Base XY trajectory colored by speed")
    axes[0, 0].set_xlabel("x [m]")
    axes[0, 0].set_ylabel("y [m]")
    axes[0, 0].axis("equal")
    fig.colorbar(scatter, ax=axes[0, 0], label="speed [m/s]")

    axes[0, 1].plot(time, metrics["base_height"], label="base z")
    if foot_idx:
        axes[0, 1].plot(time, metrics["min_foot_z"], label="min foot z")
        axes[0, 1].axhline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[0, 1].set_title("Height and ground check")
    axes[0, 1].set_xlabel("time [s]")
    axes[0, 1].set_ylabel("z [m]")
    axes[0, 1].legend(loc="best")

    axes[1, 0].plot(time, metrics["base_speed_xy"], label="stored base speed")
    axes[1, 0].plot(time, metrics["fd_base_speed_xy"], label="finite-diff base speed", alpha=0.75)
    axes[1, 0].plot(time, metrics["command_speed_xy"], label="command speed", alpha=0.75)
    axes[1, 0].set_title("Base velocity consistency")
    axes[1, 0].set_xlabel("time [s]")
    axes[1, 0].set_ylabel("speed [m/s]")
    axes[1, 0].legend(loc="best")

    axes[1, 1].plot(time, metrics["base_acc_xy"], label="base acceleration xy")
    axes[1, 1].plot(time, metrics["base_vel_mismatch"], label="stored vs finite-diff velocity mismatch")
    axes[1, 1].set_title("Velocity spikes and mismatch")
    axes[1, 1].set_xlabel("time [s]")
    axes[1, 1].set_ylabel("metric")
    axes[1, 1].legend(loc="best")

    axes[2, 0].plot(time, np.max(joint_pos, axis=1), label="max joint pos")
    axes[2, 0].plot(time, np.min(joint_pos, axis=1), label="min joint pos")
    axes[2, 0].plot(time, metrics["joint_delta"], label="max per-frame joint delta")
    axes[2, 0].set_title("Joint-position envelope")
    axes[2, 0].set_xlabel("time [s]")
    axes[2, 0].set_ylabel("rad")
    axes[2, 0].legend(loc="best")

    axes[2, 1].plot(time, np.max(np.abs(joint_vel), axis=1), label="max abs joint velocity")
    axes[2, 1].plot(time, metrics["max_abs_joint_acc"], label="max abs joint acceleration")
    axes[2, 1].set_title("Joint velocity and acceleration")
    axes[2, 1].set_xlabel("time [s]")
    axes[2, 1].set_ylabel("rad/s, rad/s^2")
    axes[2, 1].legend(loc="best")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    plt.close(fig)


def write_summary_csv(path: Path, summaries: list[dict[str, Any]]) -> None:
    fieldnames = [
        "name",
        "frames",
        "fps",
        "duration_s",
        "base_speed_xy_max",
        "base_acc_xy_max",
        "joint_vel_abs_max",
        "joint_acc_abs_max",
        "joint_delta_abs_max",
        "foot_z_min",
        "base_height_min",
        "base_height_max",
        "base_vel_mismatch_max",
        "quat_norm_error_max",
        "command_error_xy_p95",
        "issue_count",
        "worst_issue",
        "file",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for summary in summaries:
            writer.writerow({key: summary.get(key, "") for key in fieldnames})


def plot_summary(path: Path, summaries: list[dict[str, Any]], dpi: int) -> None:
    if not summaries:
        return
    rows = sorted(summaries, key=lambda item: (-int(item["issue_count"]), -float(item["base_vel_mismatch_max"])))
    names = [str(row["name"]).replace("_go2.npz", "") for row in rows]
    y = np.arange(len(rows))

    fig, axes = plt.subplots(1, 4, figsize=(18, max(8, 0.34 * len(rows))), sharey=True, constrained_layout=True)
    fig.suptitle("Go2 APEX motion audit overview", fontsize=14)

    columns = [
        ("issue_count", "Issues", "#", "#ef4444"),
        ("base_vel_mismatch_max", "Velocity mismatch", "m/s", "#f97316"),
        ("joint_acc_abs_max", "Max joint accel", "rad/s^2", "#2563eb"),
        ("foot_z_min", "Min foot z", "m", "#16a34a"),
    ]
    for ax, (key, title, xlabel, color) in zip(axes, columns, strict=True):
        values = [float(row[key]) for row in rows]
        ax.barh(y, values, color=color, alpha=0.82)
        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.grid(axis="x", alpha=0.25)
        if key == "foot_z_min":
            ax.axvline(0.0, color="black", linewidth=0.8, linestyle="--")
    axes[0].set_yticks(y, labels=names, fontsize=8)
    axes[0].invert_yaxis()

    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    motion_paths = resolve_inputs(args.input)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_dir = args.output_dir / "plots"
    event_dir = args.output_dir / "events"

    summaries: list[dict[str, Any]] = []
    all_issues: dict[str, list[dict[str, Any]]] = {}

    print(f"Input motions: {len(motion_paths)}")
    print(f"Output dir   : {args.output_dir}")
    for index, path in enumerate(motion_paths, start=1):
        print(f"[{index}/{len(motion_paths)}] {path.name}")
        motion = load_motion(path)
        summary, issues, metrics = analyze_motion(path, motion, args)
        summaries.append(summary)
        all_issues[path.name] = [issue.__dict__ for issue in issues]

        stem = path.stem.replace(" ", "_")
        plot_motion(path, motion, metrics, plot_dir / f"{stem}.{args.plot_format}", dpi=args.dpi)
        write_event_csv(event_dir / f"{stem}_events.csv", metrics, issues, top_k=args.top_k)

        if issues:
            worst = issues[0]
            print(
                f"  issues={len(issues)} worst={worst.metric} "
                f"value={worst.value:.4g} frame={worst.frame} detail={worst.detail}"
            )
        else:
            print("  issues=0")

    summaries.sort(key=lambda item: (-int(item["issue_count"]), -float(item["base_acc_xy_max"])))
    write_summary_csv(args.output_dir / "summary.csv", summaries)
    plot_summary(args.output_dir / f"overview.{args.plot_format}", summaries, dpi=args.dpi)
    with (args.output_dir / "summary.json").open("w", encoding="utf-8") as file:
        json.dump({"motions": summaries, "issues": all_issues}, file, indent=2)

    print("")
    print(f"Wrote summary: {args.output_dir / 'summary.csv'}")
    print(f"Wrote details: {args.output_dir / 'summary.json'}")
    print(f"Wrote overview: {args.output_dir / f'overview.{args.plot_format}'}")
    print(f"Wrote plots  : {plot_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
