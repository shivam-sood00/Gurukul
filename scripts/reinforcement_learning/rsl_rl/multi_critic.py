# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION.
# Copyright (c) 2022-2026, The Isaac Lab Project Developers.
# SPDX-License-Identifier: BSD-3-Clause
# See LICENSE.rsl_rl.txt for the RSL-RL license notice.
"""APEX grouped PPO adapted to Gurukul's RSL-RL 5.3 and manager-based tasks.

Method reference: https://github.com/marmotlab/APEX
"""

from __future__ import annotations

import copy
import os
import random
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from rsl_rl.algorithms import PPO
from rsl_rl.models import MLPModel, RNNModel
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.storage import RolloutStorage
from rsl_rl.utils import check_nan, resolve_obs_groups
from rsl_rl.utils.logger import Logger
from torch import nn


class MultiCriticModel(nn.Module):
    """Independent native value networks, including independent recurrent states."""

    def __init__(self, obs, critic_groups, *, recurrent=False, **kwargs):
        super().__init__()
        if not critic_groups or any(not group for group in critic_groups):
            raise ValueError("Each critic must have a nonempty observation group list")
        self.is_recurrent = recurrent
        model = RNNModel if recurrent else MLPModel
        self.heads = nn.ModuleList(
            [model(obs, {"critic": group}, "critic", 1, **copy.deepcopy(kwargs)) for group in critic_groups]
        )
        self.num_critics = len(self.heads)

    def _split_state(self, state):
        if state is None:
            return [None] * self.num_critics
        if isinstance(state, (tuple, list)):
            return list(zip(*(tensor.chunk(self.num_critics, dim=0) for tensor in state), strict=True))
        return state.chunk(self.num_critics, dim=0)

    def forward(self, obs, masks=None, hidden_state=None):
        return torch.cat(
            [
                head(obs, masks=masks, hidden_state=state)
                for head, state in zip(self.heads, self._split_state(hidden_state), strict=True)
            ],
            dim=-1,
        )

    def get_hidden_state(self):
        states = [head.get_hidden_state() for head in self.heads]
        if states[0] is None:
            return None
        if isinstance(states[0], tuple):
            return tuple(torch.cat(parts, dim=0) for parts in zip(*states, strict=True))
        return torch.cat(states, dim=0)

    def reset(self, dones=None, hidden_state=None):
        for head, state in zip(self.heads, self._split_state(hidden_state), strict=True):
            head.reset(dones, hidden_state=state)

    def update_normalization(self, obs):
        for head in self.heads:
            head.update_normalization(obs)


class MultiCriticRolloutStorage(RolloutStorage):
    """Native TensorDict/recurrent storage with one reward/value column per head."""

    def __init__(self, num_envs, num_steps, obs, actions_shape, num_critics, device="cpu"):
        if min(num_envs, num_steps, num_critics) < 1:
            raise ValueError("Environment, rollout and critic counts must be positive")
        super().__init__("rl", num_envs, num_steps, obs, actions_shape, device)
        self.num_critics = num_critics
        for name in ("rewards", "values", "returns"):
            setattr(self, name, torch.zeros(num_steps, num_envs, num_critics, device=device))

    def add_transition(self, transition):
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow")
        expected = (self.num_envs, self.num_critics)
        if transition.rewards.shape != expected or transition.values.shape != expected:
            raise ValueError(f"Expected reward/value shape {expected}")
        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards)
        self.dones[self.step].copy_(transition.dones.view(-1, 1))
        self.values[self.step].copy_(transition.values)
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(-1, 1))
        if self.distribution_params is None:
            self.distribution_params = tuple(
                torch.zeros(self.num_transitions_per_env, *p.shape, device=self.device)
                for p in transition.distribution_params
            )
        for dest, value in zip(self.distribution_params, transition.distribution_params, strict=True):
            dest[self.step].copy_(value)
        self._save_hidden_states(transition.hidden_states)
        self.step += 1

    def mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if num_mini_batches < 1 or self.num_envs * self.num_transitions_per_env % num_mini_batches:
            raise ValueError("num_envs * num_steps_per_env must divide evenly into minibatches")
        yield from super().mini_batch_generator(num_mini_batches, num_epochs)

    def recurrent_mini_batch_generator(self, num_mini_batches, num_epochs=8):
        if num_mini_batches < 1 or self.num_envs % num_mini_batches:
            raise ValueError("Recurrent PPO num_envs must divide evenly into minibatches")
        yield from super().recurrent_mini_batch_generator(num_mini_batches, num_epochs)


class MultiCriticPPO(PPO):
    """Native PPO with explicit reward partitions and terminal-state bootstrapping."""

    def __init__(self, actor, critic, storage, *, advantage_weights=None, reward_term_groups=None, **kwargs):
        if kwargs.get("rnd_cfg") or kwargs.get("symmetry_cfg"):
            raise ValueError("Grouped PPO does not yet support RND or symmetry")
        super().__init__(actor, critic, storage, **kwargs)
        self.num_critics = storage.num_critics
        weights = advantage_weights if advantage_weights is not None else [1.0] * self.num_critics
        self.advantage_weights = torch.as_tensor(weights, dtype=torch.float32, device=self.device)
        if (
            self.advantage_weights.shape != (self.num_critics,)
            or not torch.isfinite(self.advantage_weights).all()
            or (self.advantage_weights < 0).any()
            or self.advantage_weights.sum() <= 0
        ):
            raise ValueError("advantage_weights must be finite, nonnegative and match the heads, with positive sum")
        self.advantage_weights /= self.advantage_weights.sum()
        if reward_term_groups is not None and len(reward_term_groups) != self.num_critics:
            raise ValueError("Reward term groups must match the number of critics")
        self.reward_term_groups = reward_term_groups
        self._vec_env = None
        self._reward_term_indices = None
        self._warned_reward_term_group_config = False
        if min(self.num_learning_epochs, self.num_mini_batches) < 1:
            raise ValueError("Learning epochs and minibatch count must be positive")
        size = (
            storage.num_envs
            if actor.is_recurrent or critic.is_recurrent
            else (storage.num_envs * storage.num_transitions_per_env)
        )
        if size % self.num_mini_batches:
            raise ValueError("Rollout size must divide evenly into minibatches (num_envs for recurrent PPO)")

    def bind_env(self, env):
        self._vec_env = env
        self._reward_term_indices = None

    def _extract_reward_groups_from_env_terms(self) -> torch.Tensor | None:
        """Build grouped rewards from Isaac Lab reward terms, if configured."""
        if self.reward_term_groups is None or self._vec_env is None:
            return None

        unwrapped_env = getattr(self._vec_env, "unwrapped", None)
        reward_manager = getattr(unwrapped_env, "reward_manager", None)
        if reward_manager is None or not hasattr(reward_manager, "_step_reward"):
            return None

        if self._reward_term_indices is None:
            term_names = list(reward_manager.active_terms)
            term_cfgs = list(getattr(reward_manager, "_term_cfgs", [None] * len(term_names)))
            group_indices: list[list[int]] = []
            missing_term_names: list[str] = []
            for group in self.reward_term_groups:
                curr_indices: list[int] = []
                for term_name in group:
                    if term_name in term_names:
                        curr_indices.append(term_names.index(term_name))
                    else:
                        missing_term_names.append(term_name)
                group_indices.append(curr_indices)
            configured_names = [name for group in self.reward_term_groups for name in group]
            duplicate_names = sorted({name for name in configured_names if configured_names.count(name) > 1})
            nonzero_names = {
                name
                for name, term_cfg in zip(term_names, term_cfgs)
                if term_cfg is None or float(getattr(term_cfg, "weight", 0.0)) != 0.0
            }
            unassigned_names = sorted(nonzero_names - set(configured_names))
            if duplicate_names or unassigned_names:
                details = []
                if duplicate_names:
                    details.append(f"assigned more than once: {duplicate_names}")
                if unassigned_names:
                    details.append(f"active nonzero terms not assigned: {unassigned_names}")
                raise ValueError("Invalid multi-critic reward grouping; " + "; ".join(details))
            if missing_term_names and not self._warned_reward_term_group_config:
                warnings.warn(
                    "MultiCriticPPO could not match some configured reward_term_groups names in "
                    f"the environment reward manager: {sorted(set(missing_term_names))}. "
                    "Missing terms contribute 0 to grouped rewards.",
                    RuntimeWarning,
                )
                self._warned_reward_term_group_config = True
            self._reward_term_indices = group_indices

        step_reward = reward_manager._step_reward.to(self.device)
        dt = float(getattr(unwrapped_env, "step_dt", 1.0))
        grouped_rewards: list[torch.Tensor] = []
        for idxs in self._reward_term_indices:
            if len(idxs) == 0:
                grouped_rewards.append(
                    torch.zeros((step_reward.shape[0], 1), dtype=step_reward.dtype, device=self.device)
                )
            else:
                grouped_rewards.append(step_reward[:, idxs].sum(dim=1, keepdim=True) * dt)
        return torch.cat(grouped_rewards, dim=1)

    def _extract_reward_groups(self, rewards, extras):
        candidate = (
            rewards if rewards.ndim == 2 and rewards.shape[-1] == self.num_critics else extras.get("reward_groups")
        )
        if candidate is None:
            candidate = self._extract_reward_groups_from_env_terms()
        if candidate is None:
            raise ValueError("Multi-critic PPO requires explicit reward_groups or a complete reward-term partition")
        if candidate.shape != (self.storage.num_envs, self.num_critics):
            raise ValueError(f"Invalid grouped reward shape: {tuple(candidate.shape)}")
        return candidate.to(self.device)

    def process_env_step(self, obs, rewards, dones, extras):
        self.transition.rewards = self._extract_reward_groups(rewards, extras).clone()
        self.transition.dones = dones
        timeouts = extras.get("terminal_time_outs", extras.get("time_outs"))
        if timeouts is not None and torch.any(timeouts):
            terminal = extras.get("terminal_observations")
            if terminal is None:
                raise ValueError("Timeouts require pre-reset terminal_observations; enable the APEX environment hook")
            hidden = self.critic.get_hidden_state()
            values = self.critic(terminal.to(self.device)).detach()
            self.critic.reset(hidden_state=hidden)
            mask = timeouts.to(self.device).reshape(-1, 1)
            self.transition.rewards += self.gamma * torch.where(mask.bool(), values, 0.0)
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.actor.reset(dones)
        self.critic.reset(dones)

    def compute_returns(self, obs):
        st = self.storage
        hidden = self.critic.get_hidden_state()
        last_values = self.critic(obs).detach()
        self.critic.reset(hidden_state=hidden)
        advantage = torch.zeros_like(last_values)
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            live = 1.0 - st.dones[step].float()
            delta = st.rewards[step] + live * self.gamma * next_values - st.values[step]
            advantage = delta + live * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]
        heads = st.returns - st.values
        # Head normalization always precedes mixing, even with additional minibatch normalization.
        heads = (heads - heads.mean(dim=(0, 1), keepdim=True)) / heads.std(
            dim=(0, 1), unbiased=False, keepdim=True
        ).clamp_min(1e-8)
        st.advantages = (heads * self.advantage_weights).sum(-1, keepdim=True)

    def update(self):
        losses = torch.zeros(3, device=self.device)
        recurrent = self.actor.is_recurrent or self.critic.is_recurrent
        generator = self.storage.recurrent_mini_batch_generator if recurrent else self.storage.mini_batch_generator
        for batch in generator(self.num_mini_batches, self.num_learning_epochs):
            advantage = batch.advantages
            if self.normalize_advantage_per_mini_batch:
                advantage = (advantage - advantage.mean()) / advantage.std(unbiased=False).clamp_min(1e-8)
            self.actor(
                batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[0], stochastic_output=True
            )
            log_prob = self.actor.get_output_log_prob(batch.actions)
            values = self.critic(batch.observations, masks=batch.masks, hidden_state=batch.hidden_states[1])
            entropy = self.actor.output_entropy.mean()
            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.no_grad():
                    kl = self.actor.get_kl_divergence(
                        batch.old_distribution_params, self.actor.output_distribution_params
                    ).mean()
                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl)
                        kl /= self.gpu_world_size
                    if self.gpu_global_rank == 0:
                        if kl > 2 * self.desired_kl:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif 0 < kl < self.desired_kl / 2:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)
                    if self.is_multi_gpu:
                        lr = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr, src=0)
                        self.learning_rate = lr.item()
                    for group in self.optimizer.param_groups:
                        group["lr"] = self.learning_rate
            ratio = torch.exp(log_prob - batch.old_actions_log_prob.squeeze(-1))
            surrogate = torch.maximum(
                -advantage.squeeze(-1) * ratio,
                -advantage.squeeze(-1) * ratio.clamp(1 - self.clip_param, 1 + self.clip_param),
            ).mean()
            value_loss = (values - batch.returns).square()
            if self.use_clipped_value_loss:
                clipped = batch.values + (values - batch.values).clamp(-self.clip_param, self.clip_param)
                value_loss = torch.maximum(value_loss, (clipped - batch.returns).square())
            value_loss = value_loss.mean()
            loss = surrogate + self.value_loss_coef * value_loss - self.entropy_coef * entropy
            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
            self.optimizer.step()
            losses += torch.stack((value_loss.detach(), surrogate.detach(), entropy.detach()))
        observations = self.storage.observations.flatten(0, 1)
        self.actor.update_normalization(observations)
        self.critic.update_normalization(observations)
        self.storage.clear()
        return dict(
            zip(
                ("value_function", "surrogate", "entropy"),
                (losses / (self.num_learning_epochs * self.num_mini_batches)).cpu().tolist(),
                strict=True,
            )
        )

    def save(self):
        state = super().save()
        state["learning_rate"] = self.learning_rate
        state["multi_critic_format_version"] = 2
        return state

    def load(self, loaded_dict, load_cfg=None, strict=True):
        if loaded_dict.get("multi_critic_format_version") != 2:
            raise ValueError(
                "Legacy multi-critic checkpoints use a different model/distribution contract; start a fresh run"
            )
        resume = super().load(loaded_dict, load_cfg, strict)
        if load_cfg is None or load_cfg.get("optimizer", False):
            self.learning_rate = float(self.optimizer.param_groups[0]["lr"])
        return resume


class MultiCriticRunner(OnPolicyRunner):
    """Grouped PPO with completed-iteration checkpoints and native policy export."""

    def __init__(self, env, train_cfg, log_dir=None, device="cpu"):
        self.env, self.cfg, self.device = env, copy.deepcopy(train_cfg), device
        self._configure_multi_gpu()
        obs = env.get_observations().to(device)
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg.get("obs_groups", {}), ["policy", "critic"])
        groups = self.cfg.get("multi_critic_groups") or [[name] for name in self.cfg["obs_groups"]["critic"]]
        recurrent = self.cfg.get("multi_critic_recurrent", False)
        model = RNNModel if recurrent else MLPModel
        policy = self.cfg["policy"]
        rnn = (
            {
                "rnn_type": self.cfg.get("multi_critic_rnn_type", "lstm"),
                "rnn_hidden_dim": self.cfg.get("multi_critic_rnn_hidden_dim", 256),
                "rnn_num_layers": self.cfg.get("multi_critic_rnn_num_layers", 1),
            }
            if recurrent
            else {}
        )
        actor = model(
            obs,
            self.cfg["obs_groups"],
            "policy",
            env.num_actions,
            hidden_dims=policy["actor_hidden_dims"],
            activation=policy.get("activation", "elu"),
            obs_normalization=policy.get("actor_obs_normalization", False),
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": policy.get("init_noise_std", 1.0),
                "std_type": "log",
                "std_range": self.cfg.get("multi_critic_std_range", [0.01, 2.0]),
            },
            **rnn,
        )
        critic = MultiCriticModel(
            obs,
            groups,
            recurrent=recurrent,
            hidden_dims=policy["critic_hidden_dims"],
            activation=policy.get("activation", "elu"),
            obs_normalization=policy.get("critic_obs_normalization", False),
            **rnn,
        )
        storage = MultiCriticRolloutStorage(
            env.num_envs, self.cfg["num_steps_per_env"], obs, [env.num_actions], len(groups), device
        )
        algorithm = dict(self.cfg["algorithm"])
        algorithm.pop("class_name", None)
        algorithm.pop("share_cnn_encoders", None)
        self.cfg["algorithm"].setdefault("rnd_cfg", None)
        self.alg = MultiCriticPPO(
            actor,
            critic,
            storage,
            advantage_weights=self.cfg.get("multi_critic_advantage_weights"),
            reward_term_groups=self.cfg.get("multi_critic_reward_term_groups"),
            device=device,
            multi_gpu_cfg=self.cfg["multi_gpu"],
            **algorithm,
        )
        self.alg.bind_env(env)
        unwrapped = getattr(env, "unwrapped", env)
        if hasattr(unwrapped, "enable_terminal_observations"):
            unwrapped.enable_terminal_observations(list(dict.fromkeys(name for group in groups for name in group)))
        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=env.cfg,
            num_envs=env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=device,
        )
        self.current_learning_iteration = 0

    def learn(self, num_learning_iterations, init_at_random_ep_len=False):
        if num_learning_iterations < 0:
            raise ValueError("Number of learning iterations must be nonnegative")
        if init_at_random_ep_len:
            self.env.episode_length_buf[:] = torch.randint_like(
                self.env.episode_length_buf, high=self.env.max_episode_length
            )
        obs = self.env.get_observations().to(self.device)
        self.alg.train_mode()
        if self.is_distributed:
            self.alg.broadcast_parameters()
        self.logger.init_logging_writer()
        start_it = self.current_learning_iteration
        total_it = start_it + num_learning_iterations
        try:
            for it in range(start_it, total_it):
                start = time.perf_counter()
                with torch.inference_mode():
                    for _ in range(self.cfg["num_steps_per_env"]):
                        actions = self.alg.act(obs)
                        obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
                        if self.cfg.get("check_for_nan", True):
                            check_nan(obs, rewards, dones)
                        obs, rewards, dones = obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                        self.alg.process_env_step(obs, rewards, dones, extras)
                        self.logger.process_env_step(rewards, dones, extras, None)
                    self.alg.compute_returns(obs)
                collected = time.perf_counter()
                losses = self.alg.update()
                self.current_learning_iteration = it + 1
                self.logger.log(
                    it=it,
                    start_it=start_it,
                    total_it=total_it,
                    collect_time=collected - start,
                    learn_time=time.perf_counter() - collected,
                    loss_dict=losses,
                    learning_rate=self.alg.learning_rate,
                    action_std=self.alg.get_policy().output_std,
                    rnd_weight=None,
                )
                if self.logger.log_dir and (it + 1) % self.cfg["save_interval"] == 0:
                    self.save(str(Path(self.logger.log_dir) / f"model_{it + 1}.pt"))
            if self.logger.log_dir:
                self.save(str(Path(self.logger.log_dir) / f"model_{self.current_learning_iteration}.pt"))
        finally:
            self.logger.stop_logging_writer()

    def _checkpoint_contract(self):
        contract = {
            key: self.cfg.get(key)
            for key in (
                "task_name",
                "obs_groups",
                "multi_critic_groups",
                "multi_critic_reward_term_groups",
                "multi_critic_advantage_weights",
                "multi_critic_recurrent",
                "multi_critic_rnn_type",
                "multi_critic_rnn_hidden_dim",
                "multi_critic_rnn_num_layers",
                "multi_critic_std_range",
                "num_steps_per_env",
                "policy",
            )
        }
        setup = self.cfg.get("task_setup", {})
        contract["motion_manifest"] = (setup.get("motion_dataset") or {}).get("manifest_sha256")
        contract["world_size"] = self.gpu_world_size
        return contract

    def save(self, path, infos=None):
        env = getattr(self.env, "unwrapped", self.env)
        local_state = {
            "environment": env.training_state_dict() if hasattr(env, "training_state_dict") else None,
            "rng": {
                "python": random.getstate(),
                "numpy": np.random.get_state(),
                "torch": torch.get_rng_state(),
                "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            },
        }
        rank_states = None
        if self.is_distributed:
            # Every rank participates; retain its own sampler and random streams.
            # Object collectives serialize tensors, so stage curriculum tensors on CPU.
            if local_state["environment"] is not None:
                local_state["environment"] = {
                    key: value.cpu() if isinstance(value, torch.Tensor) else value
                    for key, value in local_state["environment"].items()
                }
            rank_states = [None] * self.gpu_world_size if self.gpu_global_rank == 0 else None
            torch.distributed.gather_object(local_state, rank_states, dst=0)
            if self.gpu_global_rank != 0:
                return
        state = self.alg.save()
        state.update(
            iter=self.current_learning_iteration,
            infos=infos,
            contract=self._checkpoint_contract(),
            **local_state,
            rank_states=rank_states,
            logger={"tot_timesteps": self.logger.tot_timesteps, "tot_time": self.logger.tot_time},
        )
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(state, temporary)
        os.replace(temporary, path)
        self.logger.save_model(str(path), self.current_learning_iteration)

    def load(self, path, load_cfg=None, strict=True, map_location=None):
        state = torch.load(path, map_location=map_location or self.device, weights_only=False)
        resume = load_cfg is None or load_cfg.get("iteration", False)
        if resume and state.get("contract") != self._checkpoint_contract():
            raise ValueError(
                "Resume requires the same task, models, rollout length, motion dataset, world size, "
                "observation groups and ordered reward partition"
            )
        if self.alg.load(state, load_cfg, strict):
            self.current_learning_iteration = state["iter"]
            env = getattr(self.env, "unwrapped", self.env)
            local_state = state["rank_states"][self.gpu_global_rank] if state.get("rank_states") else state
            if local_state.get("environment") is not None:
                env.load_training_state_dict(local_state["environment"])
            rng = local_state["rng"]
            random.setstate(rng["python"])
            np.random.set_state(rng["numpy"])
            torch.set_rng_state(rng["torch"].cpu())
            if rng["cuda"] is not None and torch.cuda.is_available():
                torch.cuda.set_rng_state_all([value.cpu() for value in rng["cuda"]])
            for key, value in state.get("logger", {}).items():
                setattr(self.logger, key, value)
        return state.get("infos")
