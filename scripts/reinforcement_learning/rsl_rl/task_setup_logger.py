"""Collect the resolved task and motion-dataset schema for experiment logging."""

from __future__ import annotations

import hashlib
import json
import math
import os
from collections.abc import Mapping, Sequence
from dataclasses import asdict, is_dataclass
from typing import Any


def _callable_name(value: Any) -> str:
    module = getattr(value, "__module__", "")
    name = getattr(value, "__qualname__", getattr(value, "__name__", type(value).__name__))
    return f"{module}.{name}" if module else str(name)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [_json_safe(item) for item in value]
    if callable(value):
        return _callable_name(value)
    if hasattr(value, "detach") and hasattr(value, "cpu") and hasattr(value, "tolist"):
        return value.detach().cpu().tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    if hasattr(value, "to_dict"):
        try:
            return _json_safe(value.to_dict())
        except Exception:
            pass
    if is_dataclass(value):
        try:
            return _json_safe(asdict(value))
        except Exception:
            pass
    return repr(value)


def _shape(value: Any) -> list[int]:
    if isinstance(value, int):
        return [int(value)]
    return [int(dimension) for dimension in value]


def _term_size(shape: Sequence[int]) -> int:
    return math.prod(shape) if shape else 1


def _observation_schema(observation_manager: Any) -> dict[str, Any]:
    if observation_manager is None:
        return {"available": False, "groups": {}}

    names_by_group = getattr(
        observation_manager,
        "active_terms",
        getattr(observation_manager, "_group_obs_term_names", {}),
    )
    dims_by_group = getattr(
        observation_manager,
        "group_obs_term_dim",
        getattr(observation_manager, "_group_obs_term_dim", {}),
    )
    group_dims = getattr(
        observation_manager,
        "group_obs_dim",
        getattr(observation_manager, "_group_obs_dim", {}),
    )
    cfgs_by_group = getattr(observation_manager, "_group_obs_term_cfgs", {})
    concatenate_by_group = getattr(
        observation_manager,
        "group_obs_concatenate",
        getattr(observation_manager, "_group_obs_concatenate", {}),
    )

    groups: dict[str, Any] = {}
    for group_name, term_names in names_by_group.items():
        term_dims = list(dims_by_group.get(group_name, ()))
        term_cfgs = list(cfgs_by_group.get(group_name, ()))
        terms = []
        for index, term_name in enumerate(term_names):
            shape = _shape(term_dims[index]) if index < len(term_dims) else []
            term_cfg = term_cfgs[index] if index < len(term_cfgs) else None
            terms.append(
                {
                    "index": index,
                    "name": str(term_name),
                    "shape": shape,
                    "size": _term_size(shape),
                    "function": _callable_name(getattr(term_cfg, "func", None)) if term_cfg is not None else None,
                    "scale": _json_safe(getattr(term_cfg, "scale", None)),
                    "clip": _json_safe(getattr(term_cfg, "clip", None)),
                    "history_length": int(getattr(term_cfg, "history_length", 0) or 0),
                    "params": _json_safe(getattr(term_cfg, "params", {})),
                }
            )

        groups[str(group_name)] = {
            "shape": _shape(group_dims.get(group_name, ())),
            "size": sum(term["size"] for term in terms),
            "concatenate_terms": bool(concatenate_by_group.get(group_name, True)),
            "terms": terms,
        }
    return {"available": True, "groups": groups}


def _reward_schema(reward_manager: Any) -> dict[str, Any]:
    if reward_manager is None:
        return {"available": False, "active_rewards": [], "terms": []}

    term_names = list(getattr(reward_manager, "active_terms", getattr(reward_manager, "_term_names", ())))
    term_cfgs = list(getattr(reward_manager, "_term_cfgs", ()))
    terms = []
    for index, term_name in enumerate(term_names):
        term_cfg = term_cfgs[index] if index < len(term_cfgs) else None
        weight = float(getattr(term_cfg, "weight", 0.0)) if term_cfg is not None else None
        terms.append(
            {
                "index": index,
                "name": str(term_name),
                "weight": weight,
                "enabled": weight is None or weight != 0.0,
                "function": _callable_name(getattr(term_cfg, "func", None)) if term_cfg is not None else None,
                "params": _json_safe(getattr(term_cfg, "params", {})),
            }
        )
    return {
        "available": True,
        "active_rewards": [term["name"] for term in terms if term["enabled"]],
        "terms": terms,
    }


def _sha256_file(path: str) -> str | None:
    try:
        digest = hashlib.sha256()
        with open(path, "rb") as input_file:
            for chunk in iter(lambda: input_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _motion_dataset(env: Any) -> dict[str, Any]:
    """Return the exact resolved motion library used by a manager-based task."""
    command_manager = getattr(env, "command_manager", None)
    if command_manager is None or not hasattr(command_manager, "get_term"):
        return {"available": False}

    try:
        command = command_manager.get_term("motion")
    except (KeyError, ValueError):
        return {"available": False}

    motion = getattr(command, "motion", None)
    if motion is None:
        return {"available": False}

    motion_files = [str(path) for path in getattr(motion, "motion_files", ())]
    motion_names = [str(name) for name in getattr(motion, "motion_names", ())]
    motion_lengths = _json_safe(getattr(motion, "motion_lengths", ()))
    fps_values = _json_safe(getattr(motion, "fps_values", ()))
    if not isinstance(motion_lengths, list):
        motion_lengths = []
    if not isinstance(fps_values, list):
        fps_values = []

    cfg = getattr(command, "cfg", None)
    configured_sources = list(getattr(cfg, "motion_files", ()) or ()) if cfg is not None else []
    if len(configured_sources) == 0 and cfg is not None:
        configured_source = getattr(cfg, "motion_file", None)
        if configured_source is not None:
            configured_sources = [configured_source]

    files = []
    manifest_digest = hashlib.sha256()
    for index, path in enumerate(motion_files):
        name = motion_names[index] if index < len(motion_names) else os.path.basename(path)
        frames = int(motion_lengths[index]) if index < len(motion_lengths) else None
        fps = float(fps_values[index]) if index < len(fps_values) else None
        file_sha256 = _sha256_file(path)
        size_bytes = None
        try:
            size_bytes = os.path.getsize(path)
        except OSError:
            pass
        files.append(
            {
                "name": name,
                "path": path,
                "frames": frames,
                "fps": fps,
                "duration_seconds": frames / fps if frames is not None and fps not in {None, 0.0} else None,
                "size_bytes": size_bytes,
                "sha256": file_sha256,
            }
        )
        if file_sha256 is not None:
            manifest_digest.update(name.encode("utf-8"))
            manifest_digest.update(b"\0")
            manifest_digest.update(file_sha256.encode("ascii"))
            manifest_digest.update(b"\n")

    total_frames = sum(item["frames"] for item in files if item["frames"] is not None)
    total_duration_seconds = sum(
        item["duration_seconds"] for item in files if item["duration_seconds"] is not None
    )
    return {
        "available": True,
        "configured_sources": [str(source) for source in configured_sources],
        "resolved_file_count": len(files),
        "total_frames": total_frames,
        "total_duration_seconds": total_duration_seconds,
        "manifest_sha256": manifest_digest.hexdigest() if files and all(item["sha256"] for item in files) else None,
        "joint_names": _json_safe(getattr(motion, "joint_names", None)),
        "body_names": _json_safe(getattr(motion, "body_names", None)),
        "files": files,
    }


def collect_task_setup(env: Any, runner_cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return the resolved actor/critic observation groups and reward graph."""
    observation_manager = getattr(env, "observation_manager", None)
    reward_manager = getattr(env, "reward_manager", None)
    runner_obs_groups = _json_safe(runner_cfg.get("obs_groups", {}))
    if not isinstance(runner_obs_groups, dict):
        runner_obs_groups = {}

    return {
        "schema_version": 2,
        "runner_observation_groups": runner_obs_groups,
        "actor_observation_groups": runner_obs_groups.get(
            "actor", runner_obs_groups.get("policy", ["policy"])
        ),
        "critic_observation_groups": runner_obs_groups.get("critic", ["critic"]),
        "observations": _observation_schema(observation_manager),
        "rewards": _reward_schema(reward_manager),
        "motion_dataset": _motion_dataset(env),
    }


def write_task_setup(log_dir: str, schema: Mapping[str, Any]) -> str:
    """Write the resolved schema beside the environment and agent configuration."""
    params_dir = os.path.join(log_dir, "params")
    os.makedirs(params_dir, exist_ok=True)
    output_path = os.path.join(params_dir, "task_setup.json")
    with open(output_path, "w", encoding="utf-8") as output_file:
        json.dump(schema, output_file, indent=2, sort_keys=True)
        output_file.write("\n")
    return output_path
