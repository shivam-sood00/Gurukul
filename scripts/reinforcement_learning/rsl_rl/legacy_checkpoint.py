"""Checkpoint loading helpers for RSL-RL version transitions."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

_SHARED_POLICY_NAMES = {
    "ContactTrailActorCritic",
    "CTSActorCritic",
    "MultiCriticActorCritic",
    "RealTeacherActorCritic",
    "StartActorCritic",
}

_GO2_D1_LEG_WBC_LEGACY_INPUT_DIM = 64
_GO2_D1_LEG_WBC_DEPLOYABLE_INPUT_DIM = 56
_GO2_D1_LEG_WBC_ACTION_DIM = 12
_GO2_D1_LEG_WBC_ARM_VELOCITY_SLICE = slice(44, 52)


def _split_legacy_actor_critic_state_dict(model_state_dict: Mapping[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    """Convert pre-RSL-RL-5 ActorCritic keys into split actor and critic model keys."""
    actor_state_dict: dict[str, Any] = {}
    critic_state_dict: dict[str, Any] = {}

    for key, value in model_state_dict.items():
        if key.startswith("actor."):
            actor_state_dict[f"mlp.{key.removeprefix('actor.')}"] = value
        elif key.startswith("critic."):
            critic_state_dict[f"mlp.{key.removeprefix('critic.')}"] = value
        elif key == "std":
            actor_state_dict["distribution.std_param"] = value
        elif key == "log_std":
            actor_state_dict["distribution.log_std_param"] = value
        elif key.startswith("actor_obs_normalizer."):
            actor_state_dict[f"obs_normalizer.{key.removeprefix('actor_obs_normalizer.')}"] = value
        elif key.startswith("critic_obs_normalizer."):
            critic_state_dict[f"obs_normalizer.{key.removeprefix('critic_obs_normalizer.')}"] = value

    return actor_state_dict, critic_state_dict


def _convert_legacy_checkpoint(loaded_dict: Mapping[str, Any]) -> dict[str, Any] | None:
    """Return an RSL-RL-5-style checkpoint dict when the payload uses the old model_state_dict format."""
    if "actor_state_dict" in loaded_dict or "critic_state_dict" in loaded_dict:
        return None

    model_state_dict = loaded_dict.get("model_state_dict")
    if not isinstance(model_state_dict, Mapping):
        return None

    actor_state_dict, critic_state_dict = _split_legacy_actor_critic_state_dict(model_state_dict)
    if not actor_state_dict or not critic_state_dict:
        return None

    converted_dict = dict(loaded_dict)
    converted_dict["actor_state_dict"] = actor_state_dict
    converted_dict["critic_state_dict"] = critic_state_dict
    return converted_dict


def _raise_unsupported_checkpoint_payload(checkpoint_path: Path, payload: Any) -> None:
    payload_type = type(payload).__name__
    if isinstance(payload, torch.jit.ScriptModule):
        raise TypeError(
            f"Checkpoint {checkpoint_path} is a TorchScript archive ({payload_type}), "
            "not an RSL-RL training checkpoint. "
            "Use `logs/rsl_rl/<experiment>/<run>/model_<iter>.pt` with `play.py`/sim2sim. "
            "Use the exported bundle (`policy.onnx` + `deploy.yaml`, or the run directory) with the separate "
            "`unitree-sim2real` repo."
        )
    raise TypeError(
        f"Checkpoint {checkpoint_path} produced unsupported payload type {payload_type}. "
        "Expected an RSL-RL checkpoint dictionary such as `model_<iter>.pt`."
    )


def _adapt_go2_d1_leg_wbc_actor_state(
    actor_state_dict: Mapping[str, Any], target_input_dim: int
) -> dict[str, Any] | None:
    """Drop the eight unavailable D1/gripper velocity columns from a legacy leg-WBC actor.

    The historical 64D layout was ``ang_vel[3], gravity[3], command[6], joint_pos[20],
    joint_vel[20], action[12]``. The deployable 56D layout keeps only the 12 Go2 velocities,
    so columns 44:52 are removed and every downstream actor parameter remains unchanged.
    """
    first_weight = actor_state_dict.get("mlp.0.weight")
    output_weight = actor_state_dict.get("mlp.6.weight")
    if not isinstance(first_weight, torch.Tensor) or first_weight.ndim != 2:
        return None
    if first_weight.shape[1] != _GO2_D1_LEG_WBC_LEGACY_INPUT_DIM:
        return None
    if target_input_dim != _GO2_D1_LEG_WBC_DEPLOYABLE_INPUT_DIM:
        return None
    if (
        not isinstance(output_weight, torch.Tensor)
        or output_weight.ndim != 2
        or output_weight.shape[0] != _GO2_D1_LEG_WBC_ACTION_DIM
    ):
        raise ValueError("The 64D checkpoint does not have the expected 12-action Go2+D1 leg-WBC actor.")

    velocity_slice = _GO2_D1_LEG_WBC_ARM_VELOCITY_SLICE
    adapted_state = dict(actor_state_dict)
    adapted_state["mlp.0.weight"] = torch.cat(
        (first_weight[:, : velocity_slice.start], first_weight[:, velocity_slice.stop :]),
        dim=1,
    )
    return adapted_state


def load_checkpoint_for_train(runner, checkpoint_path: str, task_name: str | None) -> dict[str, Any] | None:
    """Load a training checkpoint, warm-starting historical Go2+D1 leg WBC when required."""
    normalized_task_name = (task_name or "").lower().replace("-", "").replace("_", "")
    if "legwbcasyncarm" not in normalized_task_name:
        return runner.load(checkpoint_path)

    checkpoint_file = Path(checkpoint_path)
    loaded_dict = torch.load(checkpoint_file, weights_only=False, map_location=getattr(runner, "device", None))
    if not isinstance(loaded_dict, Mapping):
        _raise_unsupported_checkpoint_payload(checkpoint_file, loaded_dict)
    actor_state_dict = loaded_dict.get("actor_state_dict")
    if not isinstance(actor_state_dict, Mapping):
        return runner.load(checkpoint_path)

    target_actor_state = runner.alg.get_policy().state_dict()
    target_first_weight = target_actor_state.get("mlp.0.weight")
    if not isinstance(target_first_weight, torch.Tensor) or target_first_weight.ndim != 2:
        return runner.load(checkpoint_path)
    adapted_actor_state = _adapt_go2_d1_leg_wbc_actor_state(actor_state_dict, int(target_first_weight.shape[1]))
    if adapted_actor_state is None:
        return runner.load(checkpoint_path)

    converted_dict = dict(loaded_dict)
    converted_dict["actor_state_dict"] = adapted_actor_state
    load_cfg = {"actor": True, "critic": True, "optimizer": False, "iteration": True, "rnd": False}
    load_iteration = runner.alg.load(converted_dict, load_cfg, strict=True)
    if load_iteration and "iter" in converted_dict:
        runner.current_learning_iteration = converted_dict["iter"]
    print(
        "[INFO]: Warm-started the 56D deployable Go2+D1 leg WBC from a legacy 64D actor; "
        "removed eight arm/gripper velocity inputs and initialized a fresh optimizer."
    )
    return converted_dict.get("infos")


def load_checkpoint_for_play(runner, checkpoint_path: str) -> dict[str, Any] | None:
    """Load a checkpoint for inference, including old single-model RSL-RL checkpoints."""
    checkpoint_file = Path(checkpoint_path)
    if checkpoint_file.name in {"policy.pt", "policy.onnx"} or (
        checkpoint_file.parent.name == "exported" and checkpoint_file.suffix in {".pt", ".onnx"}
    ):
        raise TypeError(
            f"Checkpoint {checkpoint_file} is an exported policy artifact, not an RSL-RL training checkpoint. "
            "Use `logs/rsl_rl/<experiment>/<run>/model_<iter>.pt` with `play.py`/sim2sim. "
            "Use the exported bundle (`policy.onnx` + `deploy.yaml`, or the run directory) with the separate "
            "`unitree-sim2real` repo."
        )

    loaded_dict = torch.load(checkpoint_file, weights_only=False, map_location=getattr(runner, "device", None))
    if not isinstance(loaded_dict, Mapping):
        _raise_unsupported_checkpoint_payload(checkpoint_file, loaded_dict)
    policy = getattr(getattr(runner, "alg", None), "policy", None)
    is_shared_policy = policy is not None and policy.__class__.__name__ in _SHARED_POLICY_NAMES
    if is_shared_policy and "model_state_dict" in loaded_dict and "actor_state_dict" not in loaded_dict:
        converted_dict = dict(loaded_dict)
        converted_dict["actor_state_dict"] = loaded_dict["model_state_dict"]
        converted_dict["critic_state_dict"] = loaded_dict["model_state_dict"]
    else:
        converted_dict = _convert_legacy_checkpoint(loaded_dict)

    if converted_dict is not None:
        print("[INFO]: Adapted legacy checkpoint keys for RSL-RL 5 playback.")
        loaded_dict = converted_dict

    runner_name = runner.__class__.__name__
    if runner_name == "DistillationRunner":
        load_cfg = {"student": True, "teacher": True, "optimizer": False, "iteration": True}
    else:
        load_cfg = {"actor": True, "critic": True, "optimizer": False, "iteration": True, "rnd": False}

    load_iteration = runner.alg.load(loaded_dict, load_cfg, strict=True)
    if load_iteration and "iter" in loaded_dict:
        runner.current_learning_iteration = loaded_dict["iter"]
    return loaded_dict.get("infos")
