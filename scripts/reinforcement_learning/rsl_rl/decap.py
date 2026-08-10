# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import copy
import math
import os
import re
from typing import Any

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from tensordict import TensorDict


def _migrate_legacy_distribution_cfg(model_cfg: dict[str, Any]) -> dict[str, Any]:
    """Accept Isaac Lab legacy stochastic model keys for configs not handled by its compatibility shim."""
    model_cfg = copy.deepcopy(model_cfg)
    if "distribution_cfg" not in model_cfg and model_cfg.pop("stochastic", False):
        init_std = model_cfg.pop("init_noise_std", 1.0)
        std_type = model_cfg.pop("noise_std_type", "scalar")
        state_dependent_std = model_cfg.pop("state_dependent_std", False)
        model_cfg["distribution_cfg"] = {
            "class_name": "HeteroscedasticGaussianDistribution" if state_dependent_std else "GaussianDistribution",
            "init_std": init_std,
            "std_type": std_type,
        }
    else:
        model_cfg.pop("stochastic", None)
        model_cfg.pop("init_noise_std", None)
        model_cfg.pop("noise_std_type", None)
        model_cfg.pop("state_dependent_std", None)
    return model_cfg


def _normalize_teacher_state_dict_keys(state_dict: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Translate legacy actor checkpoint keys into the current MLPModel layout."""
    normalized_state_dict: dict[str, torch.Tensor] = {}

    for key, value in state_dict.items():
        new_key = key

        if new_key.startswith("actor."):
            new_key = new_key.removeprefix("actor.")
        elif new_key.startswith("actor_obs_normalizer."):
            new_key = new_key.replace("actor_obs_normalizer.", "obs_normalizer.", 1)

        if new_key == "std":
            new_key = "distribution.std_param"
        elif new_key == "log_std":
            new_key = "distribution.log_std_param"
        elif new_key.split(".", 1)[0].isdigit():
            # Older actor checkpoints store the MLP directly as a Sequential, e.g. `0.weight`.
            new_key = f"mlp.{new_key}"

        normalized_state_dict[new_key] = value

    return normalized_state_dict


class DecAPPPO(PPO):
    """PPO with a teacher action prior and optional teacher-action imitation reward."""

    def __init__(
        self,
        actor,
        critic,
        teacher_actor,
        storage: RolloutStorage,
        decap_lambda_start: float = 1.0,
        decap_lambda_end: float = 0.0,
        decap_decay_type: str = "linear",
        decap_decay_start_iteration: int = 0,
        decap_decay_end_iteration: int | None = None,
        decap_decay_iterations: int | None = None,
        decap_warmup_iterations: int = 0,
        decap_steps_per_iteration: int = 1,
        decap_decay_steps: int | None = None,
        decap_exp_gamma: float = 0.99,
        decap_exp_k: float = 100.0,
        decap_adaptive_decay: bool = False,
        decap_adaptive_start_iteration: int = 0,
        decap_adaptive_metric_ema_alpha: float = 0.1,
        decap_adaptive_pause_drop_ratio: float = 0.1,
        decap_adaptive_resume_ratio: float = 0.97,
        decap_adaptive_pause_patience: int = 2,
        decap_adaptive_resume_patience: int = 2,
        decap_teacher_deterministic: bool = True,
        decap_action_reward_weight: float = 0.0,
        decap_action_reward_sigma: float = 0.25,
        decap_action_reward_mode: str = "exp_mse",
        decap_action_reward_use_mean: bool = True,
        decap_action_reward_scale_with_lambda: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(actor=actor, critic=critic, storage=storage, **kwargs)

        self.teacher_actor = teacher_actor.to(self.device)
        self.teacher_actor.eval()
        for param in self.teacher_actor.parameters():
            param.requires_grad_(False)

        self.decap_lambda_start = float(decap_lambda_start)
        self.decap_lambda_end = float(decap_lambda_end)
        self.decap_decay_type = str(decap_decay_type).lower()
        if self.decap_decay_type not in {"linear", "cosine", "exp"}:
            raise ValueError(f"Unsupported DecAP decay type: {decap_decay_type}. Use 'linear', 'cosine', or 'exp'.")
        self.decap_steps_per_iteration = max(1, int(decap_steps_per_iteration))
        self.decap_warmup_iterations = max(0, int(decap_warmup_iterations))
        self.decap_decay_start_iteration = max(0, int(decap_decay_start_iteration))
        self.decap_decay_start_iteration = max(self.decap_decay_start_iteration, self.decap_warmup_iterations)
        self.decap_exp_gamma = float(decap_exp_gamma)
        self.decap_exp_k = max(float(decap_exp_k), 1.0e-6)
        if decap_decay_iterations is not None:
            self.decap_decay_end_iteration = self.decap_decay_start_iteration + max(1, int(decap_decay_iterations))
        elif decap_decay_end_iteration is None:
            if decap_decay_steps is not None:
                decay_iters = max(1, math.ceil(float(decap_decay_steps) / float(self.decap_steps_per_iteration)))
                self.decap_decay_end_iteration = self.decap_decay_start_iteration + decay_iters
            else:
                self.decap_decay_end_iteration = self.decap_decay_start_iteration + 1
        else:
            self.decap_decay_end_iteration = int(decap_decay_end_iteration)
        if self.decap_decay_end_iteration <= self.decap_decay_start_iteration:
            self.decap_decay_end_iteration = self.decap_decay_start_iteration + 1

        self.decap_teacher_deterministic = bool(decap_teacher_deterministic)
        self.decap_action_reward_weight = float(decap_action_reward_weight)
        self.decap_action_reward_sigma = max(float(decap_action_reward_sigma), 1.0e-8)
        self.decap_action_reward_mode = str(decap_action_reward_mode).lower()
        if self.decap_action_reward_mode not in {"exp_mse", "negative_mse"}:
            raise ValueError("decap_action_reward_mode must be 'exp_mse' or 'negative_mse'.")
        self.decap_action_reward_use_mean = bool(decap_action_reward_use_mean)
        self.decap_action_reward_scale_with_lambda = bool(decap_action_reward_scale_with_lambda)

        self.decap_step = 0
        self.current_lambda = self.decap_lambda_start
        self.current_decay_progress = 0.0
        self.learning_iteration = 0

        self.decap_adaptive_decay = bool(decap_adaptive_decay)
        self.decap_adaptive_start_iteration = max(0, int(decap_adaptive_start_iteration))
        self.decap_adaptive_metric_ema_alpha = float(max(0.0, min(1.0, decap_adaptive_metric_ema_alpha)))
        self.decap_adaptive_pause_drop_ratio = float(max(0.0, decap_adaptive_pause_drop_ratio))
        self.decap_adaptive_resume_ratio = float(max(0.0, min(1.0, decap_adaptive_resume_ratio)))
        self.decap_adaptive_pause_patience = max(1, int(decap_adaptive_pause_patience))
        self.decap_adaptive_resume_patience = max(1, int(decap_adaptive_resume_patience))

        self.decap_decay_paused = False
        self.decap_pause_events = 0
        self.decap_resume_events = 0
        self._decap_pause_candidate_count = 0
        self._decap_resume_candidate_count = 0
        self._pause_reference_perf_ema: float | None = None
        self._perf_rollout_last = 0.0
        self._perf_ema: float | None = None
        self._best_perf_ema: float | None = None
        self._episode_reward_sums: torch.Tensor | None = None
        self._last_student_actions: torch.Tensor | None = None
        self._last_student_action_mean: torch.Tensor | None = None
        self._last_teacher_actions: torch.Tensor | None = None

        self.teacher_loaded = False
        self._reset_rollout_stats()

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> DecAPPPO:
        """Construct DecAP PPO using RSL-RL 5 model configs plus a frozen teacher actor."""
        alg_cfg = copy.deepcopy(cfg["algorithm"])
        alg_class = resolve_callable(alg_cfg.pop("class_name"))

        obs_groups = copy.deepcopy(cfg.get("obs_groups", {}))
        default_sets = ["actor", "critic", "teacher"]
        if alg_cfg.get("rnd_cfg") is not None:
            default_sets.append("rnd_state")
        obs_groups = resolve_obs_groups(obs, obs_groups, default_sets)

        alg_cfg = resolve_rnd_config(alg_cfg, obs, obs_groups, env)
        alg_cfg = resolve_symmetry_config(alg_cfg, env)

        actor_cfg = _migrate_legacy_distribution_cfg(cfg["actor"])
        actor_class = resolve_callable(actor_cfg.pop("class_name"))
        actor = actor_class(obs, obs_groups, "actor", env.num_actions, **actor_cfg).to(device)
        print(f"Actor Model: {actor}")

        critic_cfg = _migrate_legacy_distribution_cfg(cfg["critic"])
        if alg_cfg.pop("share_cnn_encoders", None):
            critic_cfg["cnns"] = actor.cnns
        critic_class = resolve_callable(critic_cfg.pop("class_name"))
        critic = critic_class(obs, obs_groups, "critic", 1, **critic_cfg).to(device)
        print(f"Critic Model: {critic}")

        teacher_actor_cfg = _migrate_legacy_distribution_cfg(cfg.get("teacher_actor", cfg["actor"]))
        teacher_actor_class = resolve_callable(teacher_actor_cfg.pop("class_name"))
        teacher_actor = teacher_actor_class(obs, obs_groups, "teacher", env.num_actions, **teacher_actor_cfg).to(device)
        print(f"Teacher Actor Model: {teacher_actor}")

        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)

        return alg_class(
            actor=actor,
            critic=critic,
            teacher_actor=teacher_actor,
            storage=storage,
            decap_lambda_start=cfg.get("decap_lambda_start", 1.0),
            decap_lambda_end=cfg.get("decap_lambda_end", 0.0),
            decap_decay_type=cfg.get("decap_decay_type", "linear"),
            decap_decay_start_iteration=cfg.get("decap_decay_start_iteration", 0),
            decap_decay_end_iteration=cfg.get("decap_decay_end_iteration"),
            decap_decay_iterations=cfg.get("decap_decay_iterations"),
            decap_warmup_iterations=cfg.get("decap_warmup_iterations", 0),
            decap_steps_per_iteration=cfg["num_steps_per_env"],
            decap_decay_steps=cfg.get("decap_decay_steps", 100_000),
            decap_exp_gamma=cfg.get("decap_exp_gamma", 0.99),
            decap_exp_k=cfg.get("decap_exp_k", 100.0),
            decap_adaptive_decay=cfg.get("decap_adaptive_decay", False),
            decap_adaptive_start_iteration=cfg.get("decap_adaptive_start_iteration", 0),
            decap_adaptive_metric_ema_alpha=cfg.get("decap_adaptive_metric_ema_alpha", 0.1),
            decap_adaptive_pause_drop_ratio=cfg.get("decap_adaptive_pause_drop_ratio", 0.1),
            decap_adaptive_resume_ratio=cfg.get("decap_adaptive_resume_ratio", 0.97),
            decap_adaptive_pause_patience=cfg.get("decap_adaptive_pause_patience", 2),
            decap_adaptive_resume_patience=cfg.get("decap_adaptive_resume_patience", 2),
            decap_teacher_deterministic=cfg.get("decap_teacher_deterministic", True),
            decap_action_reward_weight=cfg.get("decap_action_reward_weight", 0.0),
            decap_action_reward_sigma=cfg.get("decap_action_reward_sigma", 0.25),
            decap_action_reward_mode=cfg.get("decap_action_reward_mode", "exp_mse"),
            decap_action_reward_use_mean=cfg.get("decap_action_reward_use_mean", True),
            decap_action_reward_scale_with_lambda=cfg.get("decap_action_reward_scale_with_lambda", False),
            device=device,
            **alg_cfg,
            multi_gpu_cfg=cfg["multi_gpu"],
        )

    def train_mode(self) -> None:
        super().train_mode()
        self.teacher_actor.eval()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.teacher_actor.eval()

    def _reset_rollout_stats(self) -> None:
        self._rollout_steps = 0
        self._lambda_sum = torch.tensor(0.0, device=self.device)
        self._student_action_norm_sum = torch.tensor(0.0, device=self.device)
        self._prior_action_norm_sum = torch.tensor(0.0, device=self.device)
        self._completed_episode_return_sum = torch.tensor(0.0, device=self.device)
        self._completed_episode_count = 0
        self._action_prior_reward_sum = torch.tensor(0.0, device=self.device)
        self._action_prior_mse_sum = torch.tensor(0.0, device=self.device)
        self._action_prior_steps = 0

    def _accumulate_reward_stats(self, rewards: torch.Tensor) -> None:
        reward_vec = rewards.reshape(rewards.shape[0], -1).mean(dim=1).detach()
        if self._episode_reward_sums is None or self._episode_reward_sums.shape != reward_vec.shape:
            self._episode_reward_sums = torch.zeros_like(reward_vec, device=self.device)
        self._episode_reward_sums += reward_vec

    def _accumulate_completed_episodes(self, dones: torch.Tensor) -> None:
        if self._episode_reward_sums is None:
            return
        done_mask = dones.reshape(dones.shape[0], -1).any(dim=1)
        if not bool(done_mask.any()):
            return
        completed_returns = self._episode_reward_sums[done_mask]
        self._completed_episode_return_sum += completed_returns.sum()
        self._completed_episode_count += int(completed_returns.numel())
        self._episode_reward_sums[done_mask] = 0.0

    def _accumulate_rollout_stats(
        self,
        student_actions: torch.Tensor,
        teacher_actions: torch.Tensor,
        current_lambda: float,
    ) -> None:
        lambda_tensor = torch.as_tensor(current_lambda, device=self.device)
        self._rollout_steps += 1
        self._lambda_sum += lambda_tensor
        self._student_action_norm_sum += student_actions.norm(dim=-1).mean().detach()
        self._prior_action_norm_sum += (current_lambda * teacher_actions).norm(dim=-1).mean().detach()

    def _compute_current_iteration(self) -> float:
        return float(self.decap_step) / float(self.decap_steps_per_iteration)

    def _compute_decay_progress(self) -> float:
        current_iteration = self._compute_current_iteration()
        if current_iteration <= self.decap_decay_start_iteration:
            return 0.0
        if current_iteration >= self.decap_decay_end_iteration:
            return 1.0
        return (current_iteration - self.decap_decay_start_iteration) / (
            self.decap_decay_end_iteration - self.decap_decay_start_iteration
        )

    def _compute_decay_counter(self) -> float:
        current_iteration = self._compute_current_iteration()
        if current_iteration <= self.decap_decay_start_iteration:
            return 0.0
        start_step = self.decap_decay_start_iteration * self.decap_steps_per_iteration
        return max(0.0, float(self.decap_step) - float(start_step))

    def _compute_lambda(self) -> float:
        if self.decap_decay_type == "exp":
            if self.decap_exp_gamma <= 0.0:
                raise ValueError(f"decap_exp_gamma must be > 0, got {self.decap_exp_gamma}.")
            self.current_decay_progress = 0.0
            decay_counter = self._compute_decay_counter()
            value = self.decap_lambda_start * (self.decap_exp_gamma ** (decay_counter / self.decap_exp_k))
            low = min(self.decap_lambda_start, self.decap_lambda_end)
            high = max(self.decap_lambda_start, self.decap_lambda_end)
            return float(max(low, min(high, value)))

        progress = self._compute_decay_progress()
        self.current_decay_progress = progress
        if self.decap_decay_type == "linear":
            return self.decap_lambda_start + (self.decap_lambda_end - self.decap_lambda_start) * progress

        cosine_weight = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.decap_lambda_end + (self.decap_lambda_start - self.decap_lambda_end) * cosine_weight

    def restore_decap_schedule_from_iteration(self, iteration: int | float) -> None:
        """Resume DecAP decay from an RSL-RL checkpoint iteration."""
        resumed_iteration = max(0, int(iteration))
        self.decap_step = resumed_iteration * self.decap_steps_per_iteration
        self.learning_iteration = resumed_iteration
        self.current_lambda = self._compute_lambda()

    def _update_performance_state(self, rollout_perf: float) -> None:
        self._perf_rollout_last = rollout_perf
        if self._perf_ema is None:
            self._perf_ema = rollout_perf
        else:
            alpha = self.decap_adaptive_metric_ema_alpha
            self._perf_ema = (1.0 - alpha) * self._perf_ema + alpha * rollout_perf
        if self._best_perf_ema is None or self._perf_ema > self._best_perf_ema:
            self._best_perf_ema = self._perf_ema

    @staticmethod
    def _relative_drop_threshold(reference: float, drop_ratio: float) -> float:
        return reference - drop_ratio * max(abs(reference), 1.0)

    @staticmethod
    def _relative_recovery_threshold(reference: float, resume_ratio: float) -> float:
        return reference - (1.0 - resume_ratio) * max(abs(reference), 1.0)

    def _update_adaptive_decay_state(self) -> None:
        if not self.decap_adaptive_decay:
            return
        if self.learning_iteration < self.decap_adaptive_start_iteration:
            return
        if self._perf_ema is None or self._best_perf_ema is None:
            return

        if self.decap_decay_paused:
            reference = self._pause_reference_perf_ema
            if reference is None:
                reference = self._best_perf_ema
            resume_threshold = self._relative_recovery_threshold(reference, self.decap_adaptive_resume_ratio)
            if self._perf_ema >= resume_threshold:
                self._decap_resume_candidate_count += 1
            else:
                self._decap_resume_candidate_count = 0

            if self._decap_resume_candidate_count >= self.decap_adaptive_resume_patience:
                self.decap_decay_paused = False
                self.decap_resume_events += 1
                self._decap_resume_candidate_count = 0
                self._decap_pause_candidate_count = 0
                self._pause_reference_perf_ema = None
            return

        pause_threshold = self._relative_drop_threshold(self._best_perf_ema, self.decap_adaptive_pause_drop_ratio)
        if self._perf_ema < pause_threshold:
            self._decap_pause_candidate_count += 1
        else:
            self._decap_pause_candidate_count = 0

        if self._decap_pause_candidate_count >= self.decap_adaptive_pause_patience:
            self.decap_decay_paused = True
            self.decap_pause_events += 1
            self._pause_reference_perf_ema = self._best_perf_ema
            self._decap_pause_candidate_count = 0
            self._decap_resume_candidate_count = 0

    def _teacher_action(self, obs: TensorDict) -> torch.Tensor:
        if self.decap_teacher_deterministic:
            return self.teacher_actor(obs).detach()
        return self.teacher_actor(obs, stochastic_output=True).detach()

    def act(self, obs: TensorDict) -> torch.Tensor:
        self.transition.hidden_states = (self.actor.get_hidden_state(), self.critic.get_hidden_state())
        student_actions = self.actor(obs, stochastic_output=True).detach()
        self.transition.actions = student_actions
        self.transition.values = self.critic(obs).detach()
        self.transition.actions_log_prob = self.actor.get_output_log_prob(student_actions).detach()
        self.transition.distribution_params = tuple(p.detach() for p in self.actor.output_distribution_params)
        self.transition.observations = obs

        self.current_lambda = self._compute_lambda()
        if not self.decap_decay_paused:
            self.decap_step += 1

        need_teacher_actions = self.current_lambda != 0.0 or self.decap_action_reward_weight != 0.0
        if need_teacher_actions:
            with torch.no_grad():
                teacher_actions = self._teacher_action(obs)
        else:
            teacher_actions = torch.zeros_like(student_actions)

        self._last_student_actions = student_actions.detach()
        self._last_student_action_mean = self.actor.output_mean.detach()
        self._last_teacher_actions = teacher_actions.detach()

        self._accumulate_rollout_stats(student_actions, teacher_actions, self.current_lambda)
        return student_actions + self.current_lambda * teacher_actions

    def _compute_action_prior_reward(self, rewards: torch.Tensor) -> torch.Tensor:
        if self.decap_action_reward_weight == 0.0 or self._last_teacher_actions is None:
            return rewards

        if self.decap_action_reward_use_mean and self._last_student_action_mean is not None:
            student_reference = self._last_student_action_mean
        else:
            student_reference = self._last_student_actions
        if student_reference is None:
            return rewards

        mse = torch.square(student_reference - self._last_teacher_actions).mean(dim=-1)
        if self.decap_action_reward_mode == "exp_mse":
            shaped_reward = self.decap_action_reward_weight * torch.exp(-mse / self.decap_action_reward_sigma)
        else:
            shaped_reward = -self.decap_action_reward_weight * mse
        if self.decap_action_reward_scale_with_lambda:
            shaped_reward = shaped_reward * float(self.current_lambda)

        self._action_prior_reward_sum += shaped_reward.mean().detach()
        self._action_prior_mse_sum += mse.mean().detach()
        self._action_prior_steps += 1

        if rewards.ndim == 2 and rewards.shape[-1] == 1:
            shaped_reward = shaped_reward.unsqueeze(-1)
        return rewards + shaped_reward

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        self._accumulate_reward_stats(rewards)
        self._accumulate_completed_episodes(dones)
        ppo_rewards = self._compute_action_prior_reward(rewards)
        super().process_env_step(obs, ppo_rewards, dones, extras)
        self.teacher_actor.reset(dones)

    def update(self) -> dict[str, float]:
        loss_dict = super().update()
        rollout_steps = max(self._rollout_steps, 1)
        mean_lambda = float((self._lambda_sum / rollout_steps).item())
        mean_student_norm = float((self._student_action_norm_sum / rollout_steps).item())
        mean_prior_norm = float((self._prior_action_norm_sum / rollout_steps).item())
        prior_to_student_ratio = mean_prior_norm / max(mean_student_norm, 1e-8)
        if self._completed_episode_count > 0:
            rollout_perf = float((self._completed_episode_return_sum / self._completed_episode_count).item())
        else:
            rollout_perf = float(self._perf_ema if self._perf_ema is not None else 0.0)

        self._update_performance_state(rollout_perf)
        self._update_adaptive_decay_state()

        action_prior_steps = max(self._action_prior_steps, 1)
        loss_dict["decap_lambda"] = mean_lambda
        loss_dict["decap_decay_progress"] = self.current_decay_progress
        loss_dict["decap_prior_to_student_norm_ratio"] = prior_to_student_ratio
        loss_dict["decap_decay_paused"] = float(self.decap_decay_paused)
        loss_dict["decap_in_warmup"] = float(self._compute_current_iteration() <= self.decap_warmup_iterations)
        loss_dict["decap_perf_rollout"] = rollout_perf
        loss_dict["decap_perf_episode_count"] = float(self._completed_episode_count)
        loss_dict["decap_perf_ema"] = float(self._perf_ema if self._perf_ema is not None else rollout_perf)
        loss_dict["decap_perf_best_ema"] = float(
            self._best_perf_ema if self._best_perf_ema is not None else rollout_perf
        )
        loss_dict["decap_pause_events"] = float(self.decap_pause_events)
        loss_dict["decap_resume_events"] = float(self.decap_resume_events)
        loss_dict["decap_action_prior_reward"] = float((self._action_prior_reward_sum / action_prior_steps).item())
        loss_dict["decap_action_prior_mse"] = float((self._action_prior_mse_sum / action_prior_steps).item())
        loss_dict["decap_action_reward_weight"] = self.decap_action_reward_weight

        self._reset_rollout_stats()
        self.learning_iteration += 1
        return loss_dict

    def load_teacher_state_dict(self, state_dict: dict[str, torch.Tensor], strict: bool = True) -> None:
        if "actor_state_dict" in state_dict:
            teacher_state_dict = state_dict["actor_state_dict"]
        elif "model_state_dict" in state_dict:
            model_state_dict = state_dict["model_state_dict"]
            if any(key.startswith("actor.") for key in model_state_dict):
                teacher_state_dict = {}
                for key, value in model_state_dict.items():
                    if key.startswith("actor."):
                        teacher_state_dict[key.removeprefix("actor.")] = value
                    elif key.startswith("actor_obs_normalizer."):
                        teacher_state_dict[key.replace("actor_obs_normalizer.", "obs_normalizer.")] = value
                    elif key in {"std", "log_std"}:
                        teacher_state_dict[key] = value
            else:
                teacher_state_dict = model_state_dict
        else:
            teacher_state_dict = state_dict

        teacher_state_dict = _normalize_teacher_state_dict_keys(teacher_state_dict)
        self.teacher_actor.load_state_dict(teacher_state_dict, strict=strict)
        self.teacher_actor.eval()
        for param in self.teacher_actor.parameters():
            param.requires_grad_(False)
        self.teacher_loaded = True


class DecAPRunner(OnPolicyRunner):
    """Runner for PPO with decaying teacher action priors."""

    alg: DecAPPPO

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu") -> None:
        train_cfg = copy.deepcopy(train_cfg)
        train_cfg["algorithm"]["class_name"] = "decap:DecAPPPO"
        super().__init__(env, train_cfg, log_dir=log_dir, device=device)

    def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
        if not self.alg.teacher_loaded:
            raise ValueError("Teacher model parameters are not loaded. Call `load_teacher()` before training.")
        super().learn(num_learning_iterations, init_at_random_ep_len)

    @staticmethod
    def _iteration_from_checkpoint_name(path: str) -> int:
        match = re.search(r"model_(\d+)\.pt$", os.path.basename(path))
        return int(match.group(1)) if match else 0

    def load(self, path: str, *args, **kwargs):
        infos = super().load(path, *args, **kwargs)
        resume_iteration = int(getattr(self, "current_learning_iteration", 0) or 0)
        if resume_iteration <= 0:
            resume_iteration = self._iteration_from_checkpoint_name(path)
        self.alg.restore_decap_schedule_from_iteration(resume_iteration)
        print(
            "[INFO]: Restored DecAP schedule from checkpoint iteration "
            f"{resume_iteration} with lambda={self.alg.current_lambda:.6f}."
        )
        return infos

    def load_teacher(self, path: str, map_location: str | None = None, strict: bool = True) -> None:
        loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
        self.alg.load_teacher_state_dict(loaded_dict, strict=strict)
