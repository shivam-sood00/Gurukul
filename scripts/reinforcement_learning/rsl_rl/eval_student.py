# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Evaluate RSL-RL teacher/student checkpoints with closed-loop locomotion metrics."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import tempfile
from dataclasses import MISSING
from datetime import datetime
from types import SimpleNamespace
from typing import Any

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip


parser = argparse.ArgumentParser(description="Evaluate an RSL-RL checkpoint with task-level metrics.")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to evaluate in parallel.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment.")
parser.add_argument("--episodes", type=int, default=1024, help="Number of completed episodes to aggregate.")
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Safety cap on environment steps. Defaults to enough steps to collect --episodes timeouts.",
)
parser.add_argument(
    "--output",
    type=str,
    default=None,
    help="Path to write the JSON metrics file. Defaults to <checkpoint run>/eval/<timestamp>.json.",
)
parser.add_argument(
    "--csv",
    type=str,
    default=None,
    help="Optional CSV file to append a flattened one-row summary.",
)
parser.add_argument("--run_label", type=str, default=None, help="Label stored in the metrics output.")
parser.add_argument(
    "--reference_return",
    type=float,
    default=None,
    help="Optional teacher/oracle return for normalized_return = return / reference_return.",
)
parser.add_argument(
    "--baseline_return",
    type=float,
    default=None,
    help="Optional baseline return for gap_closed = (return - baseline) / (reference - baseline).",
)
parser.add_argument(
    "--max_init_terrain_level",
    type=int,
    default=None,
    help="Optional maximum initial terrain level. Leave unset to sample all configured levels.",
)
parser.add_argument(
    "--terrain_rows",
    type=int,
    default=5,
    help="Terrain generator rows for eval. Set <=0 to keep the environment config.",
)
parser.add_argument(
    "--terrain_cols",
    type=int,
    default=5,
    help="Terrain generator columns for eval. Set <=0 to keep the environment config.",
)
parser.add_argument(
    "--keep_randomization",
    action="store_true",
    default=False,
    help="Keep env randomization and corruption enabled. Default eval disables them for clean comparisons.",
)
parser.add_argument(
    "--wandb_run",
    type=str,
    default=None,
    help="Weights & Biases run path in the form <entity>/<project>/<run_id>.",
)
parser.add_argument(
    "--wandb_checkpoint",
    type=str,
    default=None,
    help=(
        "Checkpoint file name or regex to fetch from --wandb_run. "
        "Defaults to the configured load_checkpoint."
    ),
)
cli_args.add_rsl_rl_args(parser)
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# Evaluation should not require camera rendering unless the user explicitly enables it through AppLauncher args.
sys.argv = [sys.argv[0]] + hydra_args

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Everything below runs after Isaac Sim has launched."""

import importlib.metadata as metadata

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from rsl_rl.utils import resolve_callable

from isaaclab.envs import DirectMARLEnv, DirectRLEnvCfg, ManagerBasedRLEnvCfg, multi_agent_to_single_agent
from isaaclab.utils.assets import retrieve_file_path

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
)
from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl_config_compat import migrate_custom_policy_cfg

import Gurukul.tasks  # noqa: F401  # isort: skip

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from legacy_checkpoint import load_checkpoint_for_play

installed_version = metadata.version("rsl-rl-lib")


class ScalarAccumulator:
    """Streaming scalar statistics for tensors or floats."""

    def __init__(self):
        self.count = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.min_value = None
        self.max_value = None

    def update(self, value: torch.Tensor | float | int | None) -> None:
        if value is None:
            return
        if not isinstance(value, torch.Tensor):
            value = torch.as_tensor(value, dtype=torch.float32)
        value = value.detach().float()
        value = value[torch.isfinite(value)]
        if value.numel() == 0:
            return
        self.count += int(value.numel())
        self.total += float(value.sum().item())
        self.total_sq += float(value.square().sum().item())
        current_min = float(value.min().item())
        current_max = float(value.max().item())
        self.min_value = current_min if self.min_value is None else min(self.min_value, current_min)
        self.max_value = current_max if self.max_value is None else max(self.max_value, current_max)

    @property
    def mean(self) -> float | None:
        if self.count == 0:
            return None
        return self.total / self.count

    @property
    def std(self) -> float | None:
        if self.count == 0:
            return None
        mean = self.total / self.count
        var = max(0.0, self.total_sq / self.count - mean * mean)
        return var**0.5

    def as_dict(self) -> dict[str, float | int | None]:
        return {
            "count": self.count,
            "mean": self.mean,
            "std": self.std,
            "min": self.min_value,
            "max": self.max_value,
        }


class EvalMetrics:
    """Accumulates step-level and episode-level evaluation metrics."""

    def __init__(self):
        self.step_scalars: dict[str, ScalarAccumulator] = {}
        self.reward_terms: dict[str, ScalarAccumulator] = {}
        self.episodes: list[dict[str, float | int | bool | None]] = []

    def add_step(self, name: str, value: torch.Tensor | float | int | None) -> None:
        self.step_scalars.setdefault(name, ScalarAccumulator()).update(value)

    def add_reward_terms(self, env) -> None:
        reward_manager = getattr(env.unwrapped, "reward_manager", None)
        if reward_manager is None or not hasattr(reward_manager, "_step_reward"):
            return
        names = list(getattr(reward_manager, "active_terms", ()))
        step_reward = getattr(reward_manager, "_step_reward", None)
        if step_reward is None:
            return
        for idx, name in enumerate(names):
            if idx < step_reward.shape[1]:
                self.reward_terms.setdefault(name, ScalarAccumulator()).update(step_reward[:, idx])

    def add_episode(self, row: dict[str, float | int | bool | None]) -> None:
        self.episodes.append(row)

    def summarize(self) -> dict[str, Any]:
        summary: dict[str, Any] = {
            "num_episodes": len(self.episodes),
            "step_metrics": {name: acc.as_dict() for name, acc in sorted(self.step_scalars.items())},
            "reward_terms": {name: acc.as_dict() for name, acc in sorted(self.reward_terms.items())},
        }
        if not self.episodes:
            return summary

        def _tensor_from_episode(key: str) -> torch.Tensor:
            values = [row.get(key) for row in self.episodes if row.get(key) is not None]
            return torch.tensor(values, dtype=torch.float32) if values else torch.empty(0)

        returns = _tensor_from_episode("return")
        lengths = _tensor_from_episode("length_steps")
        length_seconds = _tensor_from_episode("length_seconds")
        distances = _tensor_from_episode("distance_xy")
        timeouts = _tensor_from_episode("timeout")
        terminated = _tensor_from_episode("terminated")
        terrain_levels = _tensor_from_episode("terrain_level")
        terrain_types = _tensor_from_episode("terrain_type")

        episode_summary: dict[str, Any] = {
            "return": _tensor_summary(returns),
            "length_steps": _tensor_summary(lengths),
            "length_seconds": _tensor_summary(length_seconds),
            "distance_xy": _tensor_summary(distances),
            "timeout_rate": _rate(timeouts),
            "terminated_rate": _rate(terminated),
            "terrain_level": _tensor_summary(terrain_levels),
            "terrain_type": _tensor_summary(terrain_types),
        }
        if terrain_levels.numel() > 0:
            by_level: dict[str, dict[str, float | int | None]] = {}
            for level in sorted({int(v) for v in terrain_levels.tolist()}):
                mask = terrain_levels == level
                level_returns = returns[mask] if returns.numel() == terrain_levels.numel() else torch.empty(0)
                level_timeouts = timeouts[mask] if timeouts.numel() == terrain_levels.numel() else torch.empty(0)
                by_level[str(level)] = {
                    "episodes": int(mask.sum().item()),
                    "return_mean": _mean_or_none(level_returns),
                    "success_rate": _rate(level_timeouts),
                }
            episode_summary["by_terrain_level"] = by_level
        summary["episode_metrics"] = episode_summary
        return summary


def _tensor_summary(values: torch.Tensor) -> dict[str, float | int | None]:
    if values.numel() == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None}
    values = values.float()
    return {
        "count": int(values.numel()),
        "mean": float(values.mean().item()),
        "std": float(values.std(unbiased=False).item()),
        "min": float(values.min().item()),
        "max": float(values.max().item()),
    }


def _mean_or_none(values: torch.Tensor) -> float | None:
    return float(values.float().mean().item()) if values.numel() > 0 else None


def _rate(values: torch.Tensor) -> float | None:
    return _mean_or_none(values)


def _is_missing_cfg(value) -> bool:
    return isinstance(value, type(MISSING))


def _uses_legacy_distillation_policy(agent_cfg: RslRlBaseRunnerCfg) -> bool:
    """Detect old combined student-teacher distillation policies that RSL-RL 5 no longer constructs."""
    if getattr(agent_cfg, "class_name", "") != "DistillationRunner":
        return False
    policy_cfg = getattr(agent_cfg, "policy", MISSING)
    if _is_missing_cfg(policy_cfg):
        return False
    student_cfg = getattr(agent_cfg, "student", MISSING)
    if not _is_missing_cfg(student_cfg) and getattr(student_cfg, "class_name", None):
        return False
    policy_class_name = getattr(policy_cfg, "class_name", "")
    return policy_class_name not in {"", "StudentTeacher", "StudentTeacherRecurrent"}


class _LegacyDistillationEvalRunner:
    """Minimal inference runner for pre-RSL-RL-5 combined StudentTeacher policies."""

    is_legacy_distillation_play_runner = True

    def __init__(self, env, train_cfg: dict, device: str):
        self.env = env
        self.cfg = train_cfg
        self.device = device
        self.current_learning_iteration = 0

        policy_cfg = dict(train_cfg["policy"])
        policy_class_name = policy_cfg.pop("class_name")
        policy_class = resolve_callable(policy_class_name)
        obs_groups = dict(train_cfg["obs_groups"])
        obs = env.get_observations()
        policy = policy_class(obs, obs_groups, env.num_actions, **policy_cfg).to(device)
        self.alg = SimpleNamespace(policy=policy)

    def load(self, checkpoint_path: str, strict: bool = True):
        loaded_dict = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        state_dict = loaded_dict.get("model_state_dict")
        if state_dict is None:
            raise KeyError(
                "Legacy distillation eval expects a checkpoint with `model_state_dict`; "
                f"available keys: {sorted(loaded_dict.keys())}"
            )
        self.alg.policy.load_state_dict(state_dict, strict=strict)
        if "iter" in loaded_dict:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos")

    def get_inference_policy(self, device=None):
        if device is not None:
            self.alg.policy.to(device)
        self.alg.policy.eval()
        return self.alg.policy.act_inference


def _resolve_wandb_checkpoint(run_path: str, checkpoint_selector: str) -> str:
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError("wandb is required for --wandb_run checkpoint loading.") from exc

    api = wandb.Api()
    run = api.run(run_path)
    files = list(run.files())
    exact_matches = [f for f in files if f.name == checkpoint_selector]
    if exact_matches:
        selected = exact_matches[0]
    else:
        pattern = re.compile(checkpoint_selector)
        matched = [f for f in files if pattern.fullmatch(f.name)]
        if not matched:
            raise FileNotFoundError(f"No W&B checkpoint matched selector '{checkpoint_selector}' in run '{run_path}'.")

        def _sort_key(wb_file):
            match = re.search(r"model_(\d+)\.pt$", wb_file.name)
            if match:
                return (2, int(match.group(1)))
            if wb_file.name == "model.pt":
                return (1, 0)
            return (0, wb_file.name)

        selected = sorted(matched, key=_sort_key)[-1]

    download_root = tempfile.mkdtemp(prefix="Gurukul_wandb_eval_ckpt_")
    downloaded = selected.download(root=download_root, replace=True)
    local_path = getattr(downloaded, "name", "")
    if not local_path or not os.path.exists(local_path):
        local_path = os.path.join(download_root, selected.name)
    return os.path.abspath(local_path)


def _set_attr_if_exists(obj, attr_name: str, value) -> bool:
    if hasattr(obj, attr_name):
        setattr(obj, attr_name, value)
        return True
    return False


def _disable_observation_corruption(env_cfg) -> None:
    observations = getattr(env_cfg, "observations", None)
    if observations is None:
        return
    for group_cfg in vars(observations).values():
        if hasattr(group_cfg, "enable_corruption"):
            group_cfg.enable_corruption = False


def _disable_eval_randomization(env_cfg) -> None:
    _disable_observation_corruption(env_cfg)
    events = getattr(env_cfg, "events", None)
    if events is not None:
        for name in list(vars(events).keys()):
            if "randomize" in name or "push" in name:
                _set_attr_if_exists(events, name, None)
    curriculum = getattr(env_cfg, "curriculum", None)
    if curriculum is not None:
        for name in list(vars(curriculum).keys()):
            _set_attr_if_exists(curriculum, name, None)


def _configure_eval_terrain(env_cfg) -> None:
    terrain_cfg = getattr(getattr(env_cfg, "scene", None), "terrain", None)
    if terrain_cfg is None:
        return
    if hasattr(terrain_cfg, "max_init_terrain_level"):
        terrain_cfg.max_init_terrain_level = args_cli.max_init_terrain_level
    terrain_generator = getattr(terrain_cfg, "terrain_generator", None)
    if terrain_generator is not None:
        if args_cli.terrain_rows is not None and args_cli.terrain_rows > 0:
            terrain_generator.num_rows = int(args_cli.terrain_rows)
        if args_cli.terrain_cols is not None and args_cli.terrain_cols > 0:
            terrain_generator.num_cols = int(args_cli.terrain_cols)
        if hasattr(terrain_generator, "curriculum"):
            terrain_generator.curriculum = False


def _get_robot(env):
    scene = getattr(env.unwrapped, "scene", None)
    if scene is None:
        return None
    try:
        return scene["robot"]
    except Exception:
        return getattr(scene, "robot", None)


def _get_command(env, name: str = "base_velocity") -> torch.Tensor | None:
    command_manager = getattr(env.unwrapped, "command_manager", None)
    if command_manager is None:
        return None
    try:
        return command_manager.get_command(name)
    except Exception:
        return None


def _get_terrain_levels(env) -> tuple[torch.Tensor | None, torch.Tensor | None]:
    terrain = getattr(getattr(env.unwrapped, "scene", None), "terrain", None)
    if terrain is None:
        return None, None
    levels = getattr(terrain, "terrain_levels", None)
    types = getattr(terrain, "terrain_types", None)
    return levels, types


def _get_root_pos_xy(env) -> torch.Tensor | None:
    robot = _get_robot(env)
    root_pos_w = getattr(getattr(robot, "data", None), "root_pos_w", None)
    if root_pos_w is None:
        return None
    return root_pos_w[:, :2].detach().clone()


def _as_done_mask(tensor: torch.Tensor | None, num_envs: int, device: str | torch.device) -> torch.Tensor:
    if tensor is None:
        return torch.zeros(num_envs, dtype=torch.bool, device=device)
    return tensor.to(device=device).view(-1).bool()


def _extract_depth_tensor(env, obs) -> torch.Tensor | None:
    scene = getattr(env.unwrapped, "scene", None)
    if scene is not None:
        try:
            depth_sensor = scene["depth_camera"]
            output = getattr(getattr(depth_sensor, "data", None), "output", {})
            depth = output.get("distance_to_camera") if isinstance(output, dict) else None
            if depth is not None:
                if depth.ndim == 4 and depth.shape[-1] == 1:
                    depth = depth[..., 0]
                return depth.detach().float()
        except Exception:
            pass
    if isinstance(obs, dict) and "depth_camera" in obs:
        return obs["depth_camera"].detach().float()
    try:
        if "depth_camera" in obs:
            return obs["depth_camera"].detach().float()
    except Exception:
        return None
    return None


def _update_step_metrics(metrics: EvalMetrics, env, obs, actions: torch.Tensor, prev_actions: torch.Tensor | None) -> None:
    robot = _get_robot(env)
    robot_data = getattr(robot, "data", None)
    command = _get_command(env, "base_velocity")
    if robot_data is not None and command is not None:
        root_lin_vel_b = getattr(robot_data, "root_lin_vel_b", None)
        root_ang_vel_b = getattr(robot_data, "root_ang_vel_b", None)
        if root_lin_vel_b is not None:
            lin_error = command[:, :2] - root_lin_vel_b[:, :2]
            metrics.add_step("lin_vel_xy_error_sq", lin_error.square().sum(dim=-1))
            metrics.add_step("lin_vel_xy_error_norm", lin_error.norm(dim=-1))
            metrics.add_step("command_speed_xy", command[:, :2].norm(dim=-1))
            metrics.add_step("actual_speed_xy", root_lin_vel_b[:, :2].norm(dim=-1))
            metrics.add_step("root_lin_vel_z_abs", root_lin_vel_b[:, 2].abs())
        if root_ang_vel_b is not None:
            yaw_error = command[:, 2] - root_ang_vel_b[:, 2]
            metrics.add_step("yaw_rate_error_sq", yaw_error.square())
            metrics.add_step("yaw_rate_error_abs", yaw_error.abs())
            metrics.add_step("base_ang_vel_xy_sq", root_ang_vel_b[:, :2].square().sum(dim=-1))

    if robot_data is not None:
        root_pos_w = getattr(robot_data, "root_pos_w", None)
        projected_gravity_b = getattr(robot_data, "projected_gravity_b", None)
        if root_pos_w is not None:
            metrics.add_step("base_height", root_pos_w[:, 2])
        if projected_gravity_b is not None:
            metrics.add_step("projected_gravity_xy_norm", projected_gravity_b[:, :2].norm(dim=-1))

    if actions is not None:
        metrics.add_step("action_l2", actions.square().sum(dim=-1))
        if prev_actions is not None:
            metrics.add_step("action_rate_l2", (actions - prev_actions).square().sum(dim=-1))

    depth = _extract_depth_tensor(env, obs)
    if depth is not None:
        finite = torch.isfinite(depth)
        metrics.add_step("depth_valid_ratio", finite.float().mean(dim=tuple(range(1, depth.ndim))))
        finite_depth = torch.where(finite, depth, torch.zeros_like(depth))
        metrics.add_step("depth_mean", finite_depth.mean(dim=tuple(range(1, finite_depth.ndim))))
        scene = getattr(env.unwrapped, "scene", None)
        max_distance = None
        if scene is not None:
            try:
                max_distance = float(getattr(scene["depth_camera"].cfg, "max_distance", 0.0))
            except Exception:
                max_distance = None
        if max_distance is not None and max_distance > 0.0:
            saturated = finite & (depth >= 0.999 * max_distance)
            metrics.add_step("depth_saturated_ratio", saturated.float().mean(dim=tuple(range(1, depth.ndim))))

    metrics.add_reward_terms(env)


def _update_teacher_action_metrics(metrics: EvalMetrics, policy_nn, obs, student_actions: torch.Tensor) -> None:
    if not hasattr(policy_nn, "evaluate"):
        return
    try:
        teacher_actions = policy_nn.evaluate(obs)
    except Exception:
        return
    if teacher_actions is None or teacher_actions.shape != student_actions.shape:
        return
    delta = student_actions - teacher_actions
    metrics.add_step("teacher_action_mse", delta.square().mean(dim=-1))
    metrics.add_step("teacher_action_mae", delta.abs().mean(dim=-1))
    metrics.add_step("teacher_action_linf", delta.abs().amax(dim=-1))


def _compute_derived_metrics(summary: dict[str, Any]) -> None:
    step_metrics = summary.get("step_metrics", {})
    if "lin_vel_xy_error_sq" in step_metrics and step_metrics["lin_vel_xy_error_sq"]["mean"] is not None:
        step_metrics["lin_vel_xy_rmse"] = {
            "value": float(step_metrics["lin_vel_xy_error_sq"]["mean"]) ** 0.5
        }
    if "yaw_rate_error_sq" in step_metrics and step_metrics["yaw_rate_error_sq"]["mean"] is not None:
        step_metrics["yaw_rate_rmse"] = {"value": float(step_metrics["yaw_rate_error_sq"]["mean"]) ** 0.5}
    if "base_ang_vel_xy_sq" in step_metrics and step_metrics["base_ang_vel_xy_sq"]["mean"] is not None:
        step_metrics["base_ang_vel_xy_rms"] = {
            "value": float(step_metrics["base_ang_vel_xy_sq"]["mean"]) ** 0.5
        }
    if "action_rate_l2" in step_metrics and step_metrics["action_rate_l2"]["mean"] is not None:
        step_metrics["action_rate_rms"] = {"value": float(step_metrics["action_rate_l2"]["mean"]) ** 0.5}

    episode_metrics = summary.get("episode_metrics", {})
    return_mean = episode_metrics.get("return", {}).get("mean") if episode_metrics else None
    if return_mean is not None and args_cli.reference_return is not None:
        episode_metrics["normalized_return"] = float(return_mean) / float(args_cli.reference_return)
    if (
        return_mean is not None
        and args_cli.reference_return is not None
        and args_cli.baseline_return is not None
        and abs(float(args_cli.reference_return) - float(args_cli.baseline_return)) > 1.0e-8
    ):
        episode_metrics["gap_closed"] = (
            float(return_mean) - float(args_cli.baseline_return)
        ) / (float(args_cli.reference_return) - float(args_cli.baseline_return))


def _flatten(prefix: str, value: Any, out: dict[str, Any]) -> None:
    if isinstance(value, dict):
        for key, item in value.items():
            _flatten(f"{prefix}.{key}" if prefix else str(key), item, out)
    elif isinstance(value, (str, int, float, bool)) or value is None:
        out[prefix] = value


def _write_outputs(payload: dict[str, Any], output_path: str, csv_path: str | None) -> None:
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
    print(f"[INFO] Wrote eval metrics JSON: {output_path}")

    if csv_path is None:
        return
    flat: dict[str, Any] = {}
    _flatten("", payload, flat)
    os.makedirs(os.path.dirname(os.path.abspath(csv_path)), exist_ok=True)
    rows: list[dict[str, Any]] = []
    existing_fieldnames: list[str] = []
    if os.path.exists(csv_path):
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            existing_fieldnames = list(reader.fieldnames or [])
            rows = list(reader)
    fieldnames = sorted(set(existing_fieldnames) | set(flat.keys()))
    rows.append(flat)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"[INFO] Appended eval metrics CSV: {csv_path}")


def _resolve_checkpoint(agent_cfg: RslRlBaseRunnerCfg) -> str:
    log_root_path = os.path.abspath(os.path.join("logs", "rsl_rl", agent_cfg.experiment_name))
    if args_cli.checkpoint:
        return retrieve_file_path(args_cli.checkpoint)
    if args_cli.wandb_run:
        selector = args_cli.wandb_checkpoint if args_cli.wandb_checkpoint is not None else agent_cfg.load_checkpoint
        return _resolve_wandb_checkpoint(args_cli.wandb_run, selector)
    return get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)


def _make_runner(env, agent_cfg: RslRlBaseRunnerCfg, use_legacy_distillation_policy: bool):
    if agent_cfg.class_name == "OnPolicyRunner":
        return OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    if agent_cfg.class_name == "DistillationRunner":
        if use_legacy_distillation_policy:
            return _LegacyDistillationEvalRunner(env, agent_cfg.to_dict(), device=agent_cfg.device)
        return DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    if agent_cfg.class_name == "DecAPRunner":
        from decap import DecAPRunner

        return DecAPRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    if agent_cfg.class_name == "MultiCriticRunner":
        from multi_critic import MultiCriticRunner

        return MultiCriticRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    use_legacy_distillation_policy = _uses_legacy_distillation_policy(agent_cfg)
    if not use_legacy_distillation_policy:
        agent_cfg = migrate_custom_policy_cfg(agent_cfg, installed_version)

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if not args_cli.keep_randomization:
        _disable_eval_randomization(env_cfg)
    _configure_eval_terrain(env_cfg)

    resume_path = _resolve_checkpoint(agent_cfg)
    log_dir = os.path.dirname(resume_path)
    env_cfg.log_dir = log_dir
    print(f"[INFO] Evaluating checkpoint: {resume_path}")
    print(f"[INFO] Task: {args_cli.task}")
    print(f"[INFO] Episodes target: {args_cli.episodes}")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode=None)
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    runner = _make_runner(env, agent_cfg, use_legacy_distillation_policy)
    if getattr(runner, "is_legacy_distillation_play_runner", False):
        runner.load(resume_path)
    else:
        load_checkpoint_for_play(runner, resume_path)
    policy = runner.get_inference_policy(device=env.unwrapped.device)
    policy_nn = getattr(runner.alg, "policy", getattr(runner.alg, "actor_critic", getattr(runner.alg, "actor", None)))
    if policy_nn is None:
        raise RuntimeError("Could not resolve policy module from runner.")

    obs = env.get_observations()
    num_envs = env.num_envs
    device = env.unwrapped.device
    dt = float(getattr(env.unwrapped, "step_dt", 1.0))
    max_steps = args_cli.max_steps
    if max_steps is None:
        max_steps = max(1, int(args_cli.episodes / max(1, num_envs)) + 2) * int(env.max_episode_length)

    metrics = EvalMetrics()
    current_returns = torch.zeros(num_envs, dtype=torch.float32, device=device)
    current_lengths = torch.zeros(num_envs, dtype=torch.long, device=device)
    root_start_xy = _get_root_pos_xy(env)
    if root_start_xy is None:
        root_start_xy = torch.zeros((num_envs, 2), dtype=torch.float32, device=device)
    prev_actions = None
    completed = 0
    step = 0

    while simulation_app.is_running() and completed < args_cli.episodes and step < max_steps:
        root_xy_before_step = _get_root_pos_xy(env)
        terrain_levels, terrain_types = _get_terrain_levels(env)
        episode_terrain_levels = terrain_levels.detach().clone() if terrain_levels is not None else None
        episode_terrain_types = terrain_types.detach().clone() if terrain_types is not None else None

        with torch.inference_mode():
            actions = policy(obs)
            _update_teacher_action_metrics(metrics, policy_nn, obs, actions)
            obs, rewards, dones, extras = env.step(actions)
            policy_nn.reset(dones)

        rewards = rewards.view(-1).to(device=device)
        dones_bool = dones.view(-1).bool().to(device=device)
        timeout_src = extras.get("time_outs") if isinstance(extras, dict) else None
        if timeout_src is None:
            timeout_src = getattr(env.unwrapped, "reset_time_outs", None)
        terminated_src = getattr(env.unwrapped, "reset_terminated", None)
        timeouts = _as_done_mask(timeout_src, num_envs, device)
        terminated = _as_done_mask(terminated_src, num_envs, device) if terminated_src is not None else dones_bool & ~timeouts

        current_returns += rewards
        current_lengths += 1
        _update_step_metrics(metrics, env, obs, actions, prev_actions)
        prev_actions = actions.detach().clone()
        step += 1

        if torch.any(dones_bool):
            root_xy = root_xy_before_step
            if root_xy is None:
                root_xy = root_start_xy
            done_ids = torch.nonzero(dones_bool, as_tuple=False).flatten()
            for env_id_t in done_ids:
                if completed >= args_cli.episodes:
                    break
                env_id = int(env_id_t.item())
                distance_xy = float(torch.norm(root_xy[env_id] - root_start_xy[env_id]).item())
                terrain_level = (
                    int(episode_terrain_levels[env_id].item()) if episode_terrain_levels is not None else None
                )
                terrain_type = int(episode_terrain_types[env_id].item()) if episode_terrain_types is not None else None
                metrics.add_episode(
                    {
                        "return": float(current_returns[env_id].item()),
                        "length_steps": int(current_lengths[env_id].item()),
                        "length_seconds": float(current_lengths[env_id].item()) * dt,
                        "timeout": bool(timeouts[env_id].item()),
                        "terminated": bool(terminated[env_id].item()),
                        "distance_xy": distance_xy,
                        "terrain_level": terrain_level,
                        "terrain_type": terrain_type,
                    }
                )
                completed += 1
            current_returns[dones_bool] = 0.0
            current_lengths[dones_bool] = 0
            root_after_reset = _get_root_pos_xy(env)
            if root_after_reset is not None:
                root_start_xy[dones_bool] = root_after_reset[dones_bool]
            if prev_actions is not None:
                prev_actions[dones_bool] = 0.0

        if step == 1 or step % 100 == 0:
            print(f"[INFO] Eval progress: episodes={completed}/{args_cli.episodes}, steps={step}/{max_steps}")

    summary = metrics.summarize()
    _compute_derived_metrics(summary)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = args_cli.output
    if output_path is None:
        output_path = os.path.join(log_dir, "eval", f"eval_student_{timestamp}.json")
    csv_path = args_cli.csv
    if csv_path is None:
        csv_path = os.path.join(log_dir, "eval", "eval_student_summary.csv")

    payload = {
        "run_label": args_cli.run_label,
        "task": args_cli.task,
        "agent": args_cli.agent,
        "checkpoint": os.path.abspath(resume_path),
        "checkpoint_dir": os.path.abspath(log_dir),
        "num_envs": num_envs,
        "target_episodes": int(args_cli.episodes),
        "completed_episodes": int(completed),
        "steps": int(step),
        "max_steps": int(max_steps),
        "seed": int(agent_cfg.seed) if agent_cfg.seed is not None else None,
        "keep_randomization": bool(args_cli.keep_randomization),
        "max_init_terrain_level": args_cli.max_init_terrain_level,
        "reference_return": args_cli.reference_return,
        "baseline_return": args_cli.baseline_return,
        "summary": summary,
    }
    _write_outputs(payload, os.path.abspath(output_path), os.path.abspath(csv_path) if csv_path else None)

    episode_metrics = summary.get("episode_metrics", {})
    step_metrics = summary.get("step_metrics", {})
    return_mean = episode_metrics.get("return", {}).get("mean")
    timeout_rate = episode_metrics.get("timeout_rate")
    lin_rmse = step_metrics.get("lin_vel_xy_rmse", {}).get("value")
    yaw_rmse = step_metrics.get("yaw_rate_rmse", {}).get("value")
    print(
        "[RESULT] "
        f"return_mean={return_mean}, timeout_rate={timeout_rate}, "
        f"lin_vel_xy_rmse={lin_rmse}, yaw_rate_rmse={yaw_rmse}"
    )

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
