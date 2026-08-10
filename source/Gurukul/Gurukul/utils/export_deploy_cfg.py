from __future__ import annotations

import logging
import os
from collections.abc import Sequence

import numpy as np
import yaml

from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils import class_to_dict
from isaaclab.utils.string import resolve_matching_names

logger = logging.getLogger(__name__)


def format_value(value):
    if isinstance(value, float):
        return float(f"{value:.6g}")
    if isinstance(value, list):
        return [format_value(item) for item in value]
    if isinstance(value, dict):
        return {key: format_value(item) for key, item in value.items()}
    return value


def _as_python_list(value) -> list[float] | list[int]:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return list(value)
    return [value]


def _expand_scale(scale, dim: int) -> list[float]:
    if scale is None:
        return [1.0] * dim
    if hasattr(scale, "detach"):
        return _as_python_list(scale)
    if isinstance(scale, (float, int)):
        return [float(scale)] * dim
    if isinstance(scale, Sequence) and not isinstance(scale, (str, bytes)):
        values = list(scale)
        if len(values) == 1 and dim != 1:
            return [float(values[0])] * dim
        return [float(item) for item in values]
    return [float(scale)] * dim


def _action_joint_ids(joint_ids) -> list[int] | None:
    if isinstance(joint_ids, slice):
        return None
    if hasattr(joint_ids, "tolist"):
        return [int(item) for item in joint_ids.tolist()]
    return [int(item) for item in joint_ids]


def export_deploy_cfg(env: ManagerBasedRLEnv, log_dir: str) -> str | None:
    """Export a deploy-side YAML description for Unitree sim2sim/real controllers."""
    if not isinstance(env, ManagerBasedRLEnv):
        return None

    robot_cfg = getattr(getattr(env.cfg, "scene", None), "robot", None)
    joint_sdk_names = getattr(robot_cfg, "joint_sdk_names", None)
    if not joint_sdk_names:
        return None

    asset: Articulation = env.scene["robot"]
    joint_ids_map, _ = resolve_matching_names(asset.data.joint_names, joint_sdk_names, preserve_order=True)

    cfg = {}
    cfg["joint_ids_map"] = joint_ids_map
    cfg["step_dt"] = env.cfg.sim.dt * env.cfg.decimation

    stiffness = np.zeros(len(joint_sdk_names), dtype=np.float32)
    stiffness[joint_ids_map] = asset.data.default_joint_stiffness[0].detach().cpu().numpy()
    cfg["stiffness"] = stiffness.tolist()

    damping = np.zeros(len(joint_sdk_names), dtype=np.float32)
    damping[joint_ids_map] = asset.data.default_joint_damping[0].detach().cpu().numpy()
    cfg["damping"] = damping.tolist()
    cfg["default_joint_pos"] = asset.data.default_joint_pos[0].detach().cpu().numpy().tolist()

    cfg["commands"] = {}
    base_velocity_cfg = getattr(getattr(env.cfg, "commands", None), "base_velocity", None)
    if base_velocity_cfg is not None:
        ranges = class_to_dict(base_velocity_cfg.ranges)
        cfg["commands"]["base_velocity"] = {"ranges": {}}
        for item_name, item_value in ranges.items():
            if isinstance(item_value, (list, tuple)):
                cfg["commands"]["base_velocity"]["ranges"][item_name] = list(item_value)

    cfg["actions"] = {}
    for action_name, action_term in zip(env.action_manager.active_terms, env.action_manager._terms.values(), strict=False):
        term_cfg = action_term.cfg.copy()
        if isinstance(term_cfg.scale, (float, int)):
            term_cfg.scale = [float(term_cfg.scale)] * action_term.action_dim
        else:
            term_cfg.scale = _as_python_list(action_term._scale[0])

        if term_cfg.clip is not None:
            term_cfg.clip = _as_python_list(action_term._clip[0])

        if action_name in {"joint_pos", "JointPositionAction", "JointVelocityAction"}:
            if getattr(term_cfg, "use_default_offset", False):
                term_cfg.offset = _as_python_list(action_term._offset[0])
            else:
                term_cfg.offset = [0.0] * action_term.action_dim

        term_dict = term_cfg.to_dict()
        for key in ["class_type", "asset_name", "debug_vis", "preserve_order", "use_default_offset"]:
            term_dict.pop(key, None)
        term_dict["joint_ids"] = _action_joint_ids(action_term._joint_ids)
        cfg["actions"][action_name] = term_dict

    cfg["observations"] = {}
    obs_names = env.observation_manager.active_terms["policy"]
    obs_cfgs = env.observation_manager._group_obs_term_cfgs["policy"]
    for obs_name, obs_cfg in zip(obs_names, obs_cfgs, strict=False):
        obs_sample = obs_cfg.func(env, **obs_cfg.params)
        obs_dim = int(obs_sample.reshape(obs_sample.shape[0], -1).shape[1])

        term_cfg = obs_cfg.copy()
        term_cfg.scale = _expand_scale(term_cfg.scale, obs_dim)
        if term_cfg.clip is not None:
            term_cfg.clip = list(term_cfg.clip)
        if getattr(term_cfg, "history_length", 0) == 0:
            term_cfg.history_length = 1

        term_dict = term_cfg.to_dict()
        for key in ["func", "modifiers", "noise", "flatten_history_dim"]:
            term_dict.pop(key, None)
        cfg["observations"][obs_name] = term_dict

    filename = os.path.join(log_dir, "params", "deploy.yaml")
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    cfg = format_value(cfg)

    contact_trails = getattr(getattr(env, "cfg", None), "contact_trails", None)
    if contact_trails is not None and getattr(contact_trails, "use_contact_trails", False):
        cfg["policy_class"] = "ContactTrailActorCritic"
        cfg["contact_trails"] = format_value(
            {
                "use_contact_trails": bool(contact_trails.use_contact_trails),
                "num_channels": int(contact_trails.num_channels),
                "grid_size": list(contact_trails.grid_size),
                "resolution": float(contact_trails.resolution),
                "decay": float(contact_trails.decay),
                "write_mode": str(contact_trails.write_mode),
                "use_warp": bool(contact_trails.use_warp),
                "cnn_latent_dim": int(contact_trails.cnn_latent_dim),
                "stateful_policy_memory": True,
                "deploy_obs_groups": ["policy"],
                "notes": (
                    "Contact trail memory runs policy-side. Hardware/sim2real runners must "
                    "reimplement ContactTrailMemory + encoder or load exported TorchScript/ONNX."
                ),
            }
        )

    with open(filename, "w", encoding="utf-8") as file:
        yaml.dump(cfg, file, default_flow_style=None, sort_keys=False)
    return filename


def maybe_export_deploy_cfg(env: ManagerBasedRLEnv, log_dir: str) -> str | None:
    """Best-effort deploy-config export that quietly skips unsupported tasks."""
    try:
        return export_deploy_cfg(env, log_dir)
    except Exception as exc:  # pragma: no cover - best effort export for local deployment tooling.
        logger.warning("Failed to export deploy config to %s: %s", log_dir, exc)
        return None
