# SPDX-License-Identifier: Apache-2.0

"""Shared reward and event installation for SMP velocity tasks."""

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm

from . import runtime

DEFAULT_TASK_REWARD_WEIGHT = 0.5
DEFAULT_PRIOR_REWARD_WEIGHT = 0.5


def install_smp_terms(
    env_cfg,
    profile_name: str,
    task_reward_weight: float = DEFAULT_TASK_REWARD_WEIGHT,
    prior_reward_weight: float = DEFAULT_PRIOR_REWARD_WEIGHT,
) -> None:
    """Install SMP events and form ``task_weight * task + prior_weight * SMP``."""
    if task_reward_weight < 0.0 or prior_reward_weight < 0.0:
        raise ValueError("SMP task and prior reward weights must be non-negative.")
    for reward_name in dir(env_cfg.rewards):
        if reward_name.startswith("__"):
            continue
        reward_term = getattr(env_cfg.rewards, reward_name)
        if reward_term is not None and not callable(reward_term) and hasattr(reward_term, "weight"):
            reward_term.weight *= task_reward_weight

    env_cfg.events.initialize_smp = EventTerm(
        func=runtime.initialize_smp,
        mode="startup",
        params={
            "checkpoint_path": "",
            "profile_name": profile_name,
            # Generated resets remain opt-in until robot-specific feasibility
            # filtering and simulator-FK parity have been validated.
            "gsi_pool_size": 0,
            "gsi_batch_size": 256,
            "fixed_timesteps": (8, 15, 22),
            "sds_loss_scale": 6.0,
            "score_batch_size": 1024,
            "update_score_normalizer": True,
        },
    )
    # This term is appended after the inherited reset events so GSI owns the
    # final root/joint reset when enabled.
    env_cfg.events.reset_smp_state = EventTerm(func=runtime.reset_smp_state, mode="reset")
    env_cfg.rewards.smp_guidance = RewTerm(
        func=runtime.smp_guidance_reward,
        weight=prior_reward_weight,
    )


__all__ = [
    "DEFAULT_PRIOR_REWARD_WEIGHT",
    "DEFAULT_TASK_REWARD_WEIGHT",
    "install_smp_terms",
]
