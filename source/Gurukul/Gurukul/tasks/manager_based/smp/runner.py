# SPDX-License-Identifier: Apache-2.0

"""RSL-RL runner that checkpoints SMP's adaptive reward normalizer."""

from __future__ import annotations

import warnings
from collections.abc import Mapping
from typing import Any

from rsl_rl.runners import OnPolicyRunner

SMP_RUNNER_STATE_KEY = "gurukul_smp_state"
SMP_RUNNER_STATE_VERSION = 1


def _find_runtime_state(env: Any):
    """Find the manager environment beneath common vector/wrapper layers."""
    current = env
    visited: set[int] = set()
    while current is not None and id(current) not in visited:
        visited.add(id(current))
        state = getattr(current, "_smp_runtime_state", None)
        if state is not None:
            return state
        unwrapped = getattr(current, "unwrapped", None)
        if unwrapped is not None and unwrapped is not current:
            current = unwrapped
            continue
        nested = getattr(current, "env", None)
        if nested is not None and nested is not current:
            current = nested
            continue
        break
    raise RuntimeError("SmpOnPolicyRunner could not find initialized SMP state beneath its environment wrapper.")


def _export_smp_state(env: Any) -> dict[str, Any]:
    state = _find_runtime_state(env)
    return {
        "format_version": SMP_RUNNER_STATE_VERSION,
        "prior_sha256": state.checkpoint_sha256,
        "profile": state.profile.to_metadata(),
        "score_normalizer": state.checkpoint.prior.score_normalizer_state(),
    }


def _flush_smp_normalizer(env: Any) -> None:
    """Commit one rollout's deferred SDS statistics before PPO updates."""
    state = _find_runtime_state(env)
    state.checkpoint.prior.flush_score_normalizer(synchronize_distributed=True)


def _restore_smp_state(env: Any, payload: Mapping[str, Any]) -> None:
    state = _find_runtime_state(env)
    required = {"format_version", "prior_sha256", "profile", "score_normalizer"}
    if set(payload) != required:
        raise ValueError(
            "Invalid SMP runner checkpoint state fields: "
            f"missing={sorted(required.difference(payload))}, "
            f"unknown={sorted(set(payload).difference(required))}."
        )
    if payload["format_version"] != SMP_RUNNER_STATE_VERSION:
        raise ValueError(f"Unsupported SMP runner checkpoint state version: {payload['format_version']!r}.")
    if payload["prior_sha256"] != state.checkpoint_sha256:
        raise ValueError(
            "The policy checkpoint's SMP normalizer belongs to a different prior file. "
            "Pass the exact prior checkpoint used for policy training."
        )
    if payload["profile"] != state.profile.to_metadata():
        raise ValueError("The policy checkpoint's SMP profile does not match the active environment profile.")
    normalizer = payload["score_normalizer"]
    if not isinstance(normalizer, Mapping):
        raise TypeError("The policy checkpoint's SMP score normalizer must be a mapping.")
    state.checkpoint.prior.load_score_normalizer_state(normalizer)


class SmpOnPolicyRunner(OnPolicyRunner):
    """Stock PPO runner plus strict SMP normalizer save/restore."""

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        """Run stock PPO while committing score statistics once per rollout."""
        original_update = self.alg.update

        def update_after_normalizer_flush(*args, **kwargs):
            _flush_smp_normalizer(self.env)
            return original_update(*args, **kwargs)

        self.alg.update = update_after_normalizer_flush
        try:
            return super().learn(num_learning_iterations, init_at_random_ep_len)
        finally:
            self.alg.update = original_update

    def save(self, path: str, infos: dict | None = None) -> None:
        checkpoint_infos = dict(infos or {})
        if SMP_RUNNER_STATE_KEY in checkpoint_infos:
            raise KeyError(f"Checkpoint infos already contain reserved key {SMP_RUNNER_STATE_KEY!r}.")
        checkpoint_infos[SMP_RUNNER_STATE_KEY] = _export_smp_state(self.env)
        super().save(path, checkpoint_infos)

    def load(
        self,
        path: str,
        load_cfg: dict | None = None,
        strict: bool = True,
        map_location: str | None = None,
    ) -> dict | None:
        infos = super().load(path, load_cfg=load_cfg, strict=strict, map_location=map_location)
        self.restore_smp_infos(infos, require_state=True)
        return infos

    def restore_smp_infos(self, infos: Mapping[str, Any] | None, *, require_state: bool = False) -> None:
        """Restore SMP state when a compatibility loader handled policy weights."""
        if not isinstance(infos, Mapping) or SMP_RUNNER_STATE_KEY not in infos:
            if require_state:
                raise ValueError(
                    "An SMP training resume requires score-normalizer state in the policy checkpoint. "
                    "Use a checkpoint written by SmpOnPolicyRunner."
                )
            warnings.warn(
                "Policy checkpoint has no SMP normalizer state; using the prior checkpoint's initial statistics.",
                RuntimeWarning,
                stacklevel=2,
            )
            return
        payload = infos[SMP_RUNNER_STATE_KEY]
        if not isinstance(payload, Mapping):
            raise TypeError("Policy checkpoint SMP state must be a mapping.")
        _restore_smp_state(self.env, payload)


__all__ = ["SMP_RUNNER_STATE_KEY", "SmpOnPolicyRunner"]
