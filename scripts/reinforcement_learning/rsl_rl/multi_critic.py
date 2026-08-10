# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import warnings
from typing import Any

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.algorithms import PPO
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.modules import EmpiricalNormalization, MLP
from rsl_rl.runners import OnPolicyRunner
from rsl_rl.utils import resolve_obs_groups, resolve_optimizer
from rsl_rl.utils.logger import Logger


class MultiCriticRolloutStorage:
    """Rollout storage with multi-head critic targets."""

    class Transition:
        def __init__(self):
            self.observations = None
            self.actions = None
            self.rewards = None
            self.dones = None
            self.values = None
            self.actions_log_prob = None
            self.action_mean = None
            self.action_sigma = None
            self.hidden_states = None

        def clear(self):
            self.__init__()

    def __init__(
        self,
        num_envs: int,
        num_transitions_per_env: int,
        obs: TensorDict,
        actions_shape: list[int],
        num_critics: int,
        device: str = "cpu",
    ):
        self.device = device
        self.num_envs = num_envs
        self.num_transitions_per_env = num_transitions_per_env
        self.actions_shape = actions_shape
        self.num_critics = num_critics

        self.observations = TensorDict(
            {key: torch.zeros(num_transitions_per_env, *value.shape, device=device) for key, value in obs.items()},
            batch_size=[num_transitions_per_env, num_envs],
            device=device,
        )
        self.rewards = torch.zeros(num_transitions_per_env, num_envs, num_critics, device=device)
        self.actions = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
        self.dones = torch.zeros(num_transitions_per_env, num_envs, 1, device=device).byte()
        self.values = torch.zeros(num_transitions_per_env, num_envs, num_critics, device=device)
        self.returns = torch.zeros(num_transitions_per_env, num_envs, num_critics, device=device)
        self.advantages = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.actions_log_prob = torch.zeros(num_transitions_per_env, num_envs, 1, device=device)
        self.mu = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
        self.sigma = torch.zeros(num_transitions_per_env, num_envs, *actions_shape, device=device)
        self.step = 0

    def add_transitions(self, transition: Transition):
        if self.step >= self.num_transitions_per_env:
            raise OverflowError("Rollout buffer overflow! Call clear() before adding new transitions.")

        self.observations[self.step].copy_(transition.observations)
        self.actions[self.step].copy_(transition.actions)
        self.rewards[self.step].copy_(transition.rewards.view(self.num_envs, self.num_critics))
        self.dones[self.step].copy_(transition.dones.view(self.num_envs, 1))
        self.values[self.step].copy_(transition.values.view(self.num_envs, self.num_critics))
        self.actions_log_prob[self.step].copy_(transition.actions_log_prob.view(self.num_envs, 1))
        self.mu[self.step].copy_(transition.action_mean)
        self.sigma[self.step].copy_(transition.action_sigma)
        self.step += 1

    def clear(self):
        self.step = 0

    def compute_returns(
        self,
        last_values: torch.Tensor,
        gamma: float,
        lam: float,
        advantage_weights: torch.Tensor | None = None,
        normalize_advantage: bool = True,
    ):
        advantage = torch.zeros_like(last_values)
        for step in reversed(range(self.num_transitions_per_env)):
            if step == self.num_transitions_per_env - 1:
                next_values = last_values
            else:
                next_values = self.values[step + 1]

            next_not_terminal = 1.0 - self.dones[step].float()
            delta = self.rewards[step] + next_not_terminal * gamma * next_values - self.values[step]
            advantage = delta + next_not_terminal * gamma * lam * advantage
            self.returns[step] = advantage + self.values[step]

        head_advantages = self.returns - self.values
        if normalize_advantage:
            mean = head_advantages.mean(dim=(0, 1), keepdim=True)
            std = head_advantages.std(dim=(0, 1), keepdim=True).clamp_min(1e-8)
            head_advantages = (head_advantages - mean) / std

        if advantage_weights is None:
            advantage_weights = torch.ones(self.num_critics, device=self.device) / float(self.num_critics)
        else:
            advantage_weights = advantage_weights.to(self.device)
            advantage_weights = advantage_weights / advantage_weights.sum().clamp_min(1e-8)

        self.advantages = (head_advantages * advantage_weights.view(1, 1, -1)).sum(dim=-1, keepdim=True)

    def mini_batch_generator(self, num_mini_batches: int, num_epochs: int = 8):
        batch_size = self.num_envs * self.num_transitions_per_env
        mini_batch_size = batch_size // num_mini_batches
        indices = torch.randperm(num_mini_batches * mini_batch_size, requires_grad=False, device=self.device)

        observations = self.observations.flatten(0, 1)
        actions = self.actions.flatten(0, 1)
        values = self.values.flatten(0, 1)
        returns = self.returns.flatten(0, 1)
        advantages = self.advantages.flatten(0, 1)
        old_actions_log_prob = self.actions_log_prob.flatten(0, 1)
        old_mu = self.mu.flatten(0, 1)
        old_sigma = self.sigma.flatten(0, 1)

        for _ in range(num_epochs):
            for i in range(num_mini_batches):
                start = i * mini_batch_size
                end = (i + 1) * mini_batch_size
                batch_idx = indices[start:end]

                yield (
                    observations[batch_idx],
                    actions[batch_idx],
                    values[batch_idx],
                    advantages[batch_idx],
                    returns[batch_idx],
                    old_actions_log_prob[batch_idx],
                    old_mu[batch_idx],
                    old_sigma[batch_idx],
                    (None, None),
                    None,
                )

    def recurrent_mini_batch_generator(self, *args, **kwargs):
        raise NotImplementedError("MultiCriticPPO currently supports feed-forward policies only.")


class MultiCriticActorCritic(nn.Module):
    """Actor with multiple critic heads."""

    is_recurrent = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] = [256, 256, 256],
        critic_hidden_dims: list[int] = [256, 256, 256],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        num_critics: int | None = None,
        critic_obs_groups: list[list[str]] | None = None,
        **kwargs,
    ):
        if kwargs:
            print(
                "MultiCriticActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.obs_groups = obs_groups

        self._critic_obs_groups = critic_obs_groups if critic_obs_groups is not None else self._infer_critic_groups()
        if len(self._critic_obs_groups) == 0:
            raise ValueError("MultiCriticActorCritic requires at least one critic head.")

        if num_critics is not None and int(num_critics) != len(self._critic_obs_groups):
            raise ValueError(
                f"num_critics ({num_critics}) does not match critic group count ({len(self._critic_obs_groups)})."
            )
        self.num_critics = len(self._critic_obs_groups)

        num_actor_obs = 0
        for group in obs_groups["policy"]:
            assert len(obs[group].shape) == 2, "Only 1D observations are supported."
            num_actor_obs += obs[group].shape[-1]

        self.actor = MLP(num_actor_obs, num_actions, actor_hidden_dims, activation)
        self.actor_obs_normalization = actor_obs_normalization
        self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs) if actor_obs_normalization else nn.Identity()

        critic_input_dims = []
        for critic_groups in self._critic_obs_groups:
            num_critic_obs = 0
            for group in critic_groups:
                assert len(obs[group].shape) == 2, "Only 1D observations are supported."
                num_critic_obs += obs[group].shape[-1]
            critic_input_dims.append(num_critic_obs)

        self.critics = nn.ModuleList(
            [MLP(input_dim, 1, critic_hidden_dims, activation) for input_dim in critic_input_dims]
        )
        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizers = nn.ModuleList(
                [EmpiricalNormalization(input_dim) for input_dim in critic_input_dims]
            )
        else:
            self.critic_obs_normalizers = nn.ModuleList([nn.Identity() for _ in critic_input_dims])

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(
                f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'."
            )

        self.distribution = None
        Normal.set_default_validate_args(False)

        print(f"Actor MLP: {self.actor}")
        for i, critic in enumerate(self.critics):
            print(f"Critic[{i}] MLP: {critic}")

    def _infer_critic_groups(self) -> list[list[str]]:
        critic_groups = self.obs_groups.get("critic", [])
        if len(critic_groups) == 0:
            raise ValueError("obs_groups must include a non-empty 'critic' set for multi-critic policy.")
        return [[group] for group in critic_groups]

    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev

    @property
    def output_std(self):
        if self.noise_std_type == "scalar":
            return self.std
        return torch.exp(self.log_std)

    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def reset(self, dones=None):
        pass

    def forward(self, obs):
        return self.act_inference(obs)

    def update_distribution(self, obs: torch.Tensor):
        mean = self.actor(obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        return torch.cat([obs[group] for group in self.obs_groups["policy"]], dim=-1)

    def get_critic_obs(self, obs: TensorDict, critic_idx: int) -> torch.Tensor:
        return torch.cat([obs[group] for group in self._critic_obs_groups[critic_idx]], dim=-1)

    def act(self, obs: TensorDict, **kwargs):
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        self.update_distribution(actor_obs)
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict):
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        return self.actor(actor_obs)

    def evaluate_heads(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        values = []
        for i, critic in enumerate(self.critics):
            critic_obs = self.get_critic_obs(obs, i)
            critic_obs = self.critic_obs_normalizers[i](critic_obs)
            values.append(critic(critic_obs))
        return torch.cat(values, dim=-1)

    def evaluate(self, obs: TensorDict, **kwargs) -> torch.Tensor:
        # Compatibility path for callers expecting a single value.
        return self.evaluate_heads(obs, **kwargs).mean(dim=-1, keepdim=True)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict):
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            for i, normalizer in enumerate(self.critic_obs_normalizers):
                normalizer.update(self.get_critic_obs(obs, i))

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True


class MultiCriticPPO(PPO):
    """PPO variant with multi-head critic values and grouped rewards."""

    policy: MultiCriticActorCritic

    def __init__(
        self,
        policy: MultiCriticActorCritic,
        storage: MultiCriticRolloutStorage,
        num_critics: int | None = None,
        reward_group_weights: list[float] | None = None,
        advantage_weights: list[float] | None = None,
        reward_group_extras_keys: list[str] | None = None,
        reward_term_groups: list[list[str]] | None = None,
        multi_gpu_cfg: dict | None = None,
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        **kwargs,
    ):
        if rnd_cfg:
            raise ValueError("MultiCriticPPO currently does not support RND.")
        if symmetry_cfg and (symmetry_cfg.get("use_data_augmentation") or symmetry_cfg.get("use_mirror_loss")):
            raise ValueError("MultiCriticPPO currently does not support symmetry augmentation/loss.")

        self.device = kwargs.pop("device", "cpu")
        self.is_multi_gpu = multi_gpu_cfg is not None
        self.gpu_global_rank = multi_gpu_cfg["global_rank"] if multi_gpu_cfg is not None else 0
        self.gpu_world_size = multi_gpu_cfg["world_size"] if multi_gpu_cfg is not None else 1
        self.rnd = None
        self.symmetry = symmetry_cfg
        self.policy = policy.to(self.device)
        self.actor = self.policy
        self.critic = self.policy
        self._raw_actor = self.policy
        self._raw_critic = self.policy
        self.storage = storage
        self.transition = MultiCriticRolloutStorage.Transition()

        self.clip_param = kwargs.pop("clip_param", 0.2)
        self.num_learning_epochs = kwargs.pop("num_learning_epochs", 5)
        self.num_mini_batches = kwargs.pop("num_mini_batches", 4)
        self.value_loss_coef = kwargs.pop("value_loss_coef", 1.0)
        self.entropy_coef = kwargs.pop("entropy_coef", 0.01)
        self.gamma = kwargs.pop("gamma", 0.99)
        self.lam = kwargs.pop("lam", 0.95)
        self.max_grad_norm = kwargs.pop("max_grad_norm", 1.0)
        self.use_clipped_value_loss = kwargs.pop("use_clipped_value_loss", True)
        self.desired_kl = kwargs.pop("desired_kl", None)
        self.schedule = kwargs.pop("schedule", "fixed")
        self.learning_rate = kwargs.pop("learning_rate", 1.0e-3)
        self.normalize_advantage_per_mini_batch = kwargs.pop("normalize_advantage_per_mini_batch", False)
        kwargs.pop("share_cnn_encoders", None)
        optimizer_name = kwargs.pop("optimizer", "adam")
        if kwargs:
            raise TypeError(f"Unexpected MultiCriticPPO arguments: {sorted(kwargs)}")
        self.optimizer = resolve_optimizer(optimizer_name)(self.policy.parameters(), lr=self.learning_rate)

        self.num_critics = int(num_critics) if num_critics is not None else int(policy.num_critics)
        if self.num_critics <= 0:
            raise ValueError("MultiCriticPPO requires num_critics > 0.")
        if self.num_critics != policy.num_critics:
            raise ValueError(
                f"Configured num_critics ({self.num_critics}) does not match policy critics ({policy.num_critics})."
            )

        self.reward_group_weights = self._normalize_weights(reward_group_weights)
        self.advantage_weights = self._normalize_weights(advantage_weights)
        self.reward_group_extras_keys = (
            reward_group_extras_keys
            if reward_group_extras_keys is not None
            else ["reward_groups", "reward_grouped", "reward_components", "reward_terms"]
        )
        self.reward_term_groups = reward_term_groups
        self._vec_env = None
        self._reward_term_indices: list[list[int]] | None = None
        self._warned_reward_term_group_config = False
        self._warned_reward_fallback = False

    def bind_env(self, vec_env):
        """Attach the vectorized environment for optional reward-term grouping."""
        self._vec_env = vec_env
        self._reward_term_indices = None

    def _normalize_weights(self, weights: list[float] | None) -> torch.Tensor:
        if weights is None:
            return torch.ones(self.num_critics, device=self.device) / float(self.num_critics)
        if len(weights) != self.num_critics:
            raise ValueError(f"Weight count ({len(weights)}) must match num_critics ({self.num_critics}).")
        weight_tensor = torch.tensor(weights, dtype=torch.float32, device=self.device)
        if not torch.isfinite(weight_tensor).all() or torch.any(weight_tensor < 0.0):
            raise ValueError(f"Multi-critic weights must be finite and non-negative, got {weights}.")
        weight_sum = weight_tensor.sum()
        if weight_sum <= 0.0:
            raise ValueError("Multi-critic weights must contain at least one positive value.")
        return weight_tensor / weight_sum

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
                grouped_rewards.append(torch.zeros((step_reward.shape[0], 1), dtype=step_reward.dtype, device=self.device))
            else:
                grouped_rewards.append(step_reward[:, idxs].sum(dim=1, keepdim=True) * dt)
        return torch.cat(grouped_rewards, dim=1)

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        if training_type != "rl":
            raise ValueError("MultiCriticPPO supports RL training type only.")
        self.storage = MultiCriticRolloutStorage(
            num_envs=num_envs,
            num_transitions_per_env=num_transitions_per_env,
            obs=obs,
            actions_shape=actions_shape,
            num_critics=self.num_critics,
            device=self.device,
        )

    def _extract_reward_groups(self, rewards: torch.Tensor, extras: dict[str, Any]) -> torch.Tensor:
        if rewards.dim() == 2 and rewards.shape[-1] == self.num_critics:
            return rewards.to(self.device)

        for key in self.reward_group_extras_keys:
            candidate = extras.get(key, None)
            if candidate is None and isinstance(extras.get("log", None), dict):
                candidate = extras["log"].get(key, None)
            if isinstance(candidate, torch.Tensor):
                candidate = candidate.to(self.device)
                if candidate.dim() == 1:
                    candidate = candidate.unsqueeze(-1)
                if candidate.dim() == 2 and candidate.shape[1] == self.num_critics:
                    return candidate

        grouped_from_terms = self._extract_reward_groups_from_env_terms()
        if grouped_from_terms is not None:
            return grouped_from_terms

        scalar_rewards = rewards.to(self.device)
        if scalar_rewards.dim() == 1:
            scalar_rewards = scalar_rewards.unsqueeze(-1)
        elif scalar_rewards.dim() != 2 or scalar_rewards.shape[1] != 1:
            raise ValueError(
                f"Unsupported reward shape {tuple(rewards.shape)} for MultiCriticPPO with {self.num_critics} heads."
            )

        if not self._warned_reward_fallback:
            warnings.warn(
                "MultiCriticPPO did not find grouped rewards in environment outputs; "
                "falling back to weighted scalar reward split.",
                RuntimeWarning,
            )
            self._warned_reward_fallback = True

        return scalar_rewards * self.reward_group_weights.view(1, -1)

    def act(self, obs):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()

        self.transition.actions = self.policy.act(obs).detach()
        self.transition.values = self.policy.evaluate_heads(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(self.transition.actions).detach()
        self.transition.action_mean = self.policy.action_mean.detach()
        self.transition.action_sigma = self.policy.action_std.detach()
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(self, obs, rewards, dones, extras):
        self.policy.update_normalization(obs)

        reward_groups = self._extract_reward_groups(rewards, extras)
        self.transition.rewards = reward_groups.clone()
        self.transition.dones = dones

        if "time_outs" in extras:
            time_outs = extras["time_outs"].to(self.device).reshape(-1, 1)
            self.transition.rewards += self.gamma * self.transition.values * time_outs

        self.storage.add_transitions(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs):
        last_values = self.policy.evaluate_heads(obs).detach()
        self.storage.compute_returns(
            last_values,
            self.gamma,
            self.lam,
            advantage_weights=self.advantage_weights,
            normalize_advantage=not self.normalize_advantage_per_mini_batch,
        )

    def update(self):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        head_value_losses = torch.zeros(self.num_critics, device=self.device)

        if self.policy.is_recurrent:
            raise NotImplementedError("MultiCriticPPO currently supports feed-forward policies only.")
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for (
            obs_batch,
            actions_batch,
            target_values_batch,
            advantages_batch,
            returns_batch,
            old_actions_log_prob_batch,
            old_mu_batch,
            old_sigma_batch,
            hid_states_batch,
            masks_batch,
        ) in generator:
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch_all = self.policy.evaluate_heads(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

            if self.desired_kl is not None and self.schedule == "adaptive":
                with torch.inference_mode():
                    kl = torch.sum(
                        torch.log(sigma_batch / old_sigma_batch + 1.0e-5)
                        + (torch.square(old_sigma_batch) + torch.square(old_mu_batch - mu_batch))
                        / (2.0 * torch.square(sigma_batch))
                        - 0.5,
                        axis=-1,
                    )
                    kl_mean = torch.mean(kl)

                    if self.is_multi_gpu:
                        torch.distributed.all_reduce(kl_mean, op=torch.distributed.ReduceOp.SUM)
                        kl_mean /= self.gpu_world_size

                    if self.gpu_global_rank == 0:
                        if kl_mean > self.desired_kl * 2.0:
                            self.learning_rate = max(1e-5, self.learning_rate / 1.5)
                        elif kl_mean < self.desired_kl / 2.0 and kl_mean > 0.0:
                            self.learning_rate = min(1e-2, self.learning_rate * 1.5)

                    if self.is_multi_gpu:
                        lr_tensor = torch.tensor(self.learning_rate, device=self.device)
                        torch.distributed.broadcast(lr_tensor, src=0)
                        self.learning_rate = lr_tensor.item()

                    for param_group in self.optimizer.param_groups:
                        param_group["lr"] = self.learning_rate

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch, dim=-1))
            advantages = torch.squeeze(advantages_batch, dim=-1)
            surrogate = -advantages * ratio
            surrogate_clipped = -advantages * torch.clamp(ratio, 1.0 - self.clip_param, 1.0 + self.clip_param)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            per_head_losses = []
            for i in range(self.num_critics):
                value_batch = value_batch_all[:, i : i + 1]
                target_values = target_values_batch[:, i : i + 1]
                returns = returns_batch[:, i : i + 1]

                if self.use_clipped_value_loss:
                    value_clipped = target_values + (value_batch - target_values).clamp(-self.clip_param, self.clip_param)
                    value_losses = (value_batch - returns).pow(2)
                    value_losses_clipped = (value_clipped - returns).pow(2)
                    value_loss_i = torch.max(value_losses, value_losses_clipped).mean()
                else:
                    value_loss_i = (returns - value_batch).pow(2).mean()
                per_head_losses.append(value_loss_i)

            value_loss = torch.stack(per_head_losses).mean()
            loss = surrogate_loss + self.value_loss_coef * value_loss - self.entropy_coef * entropy_batch.mean()

            self.optimizer.zero_grad()
            loss.backward()

            if self.is_multi_gpu:
                self.reduce_parameters()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            head_value_losses += torch.stack([loss_i.detach() for loss_i in per_head_losses])

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        head_value_losses /= float(num_updates)
        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
        }
        for i in range(self.num_critics):
            loss_dict[f"value_function_head_{i}"] = float(head_value_losses[i].item())
        return loss_dict

    def reduce_parameters(self) -> None:
        """Synchronize the shared policy once instead of reducing actor/critic aliases twice."""
        parameters = [parameter for parameter in self.policy.parameters() if parameter.grad is not None]
        if not parameters:
            return
        flattened = torch.cat([parameter.grad.reshape(-1) for parameter in parameters])
        torch.distributed.all_reduce(flattened, op=torch.distributed.ReduceOp.SUM)
        flattened /= self.gpu_world_size
        offset = 0
        for parameter in parameters:
            numel = parameter.numel()
            parameter.grad.copy_(flattened[offset : offset + numel].view_as(parameter.grad))
            offset += numel


class MultiCriticRunner(OnPolicyRunner):
    """On-policy runner that wires MultiCriticActorCritic + MultiCriticPPO."""

    alg: MultiCriticPPO

    def __init__(self, env, train_cfg: dict, log_dir: str | None = None, device: str = "cpu"):
        """Construct the custom algorithm using the RSL-RL 5 runner contract."""
        self.env = env
        self.cfg = train_cfg
        self.device = device
        self._configure_multi_gpu()

        obs = self.env.get_observations()
        self.alg = self._construct_algorithm(obs)
        self.logger = Logger(
            log_dir=log_dir,
            cfg=self.cfg,
            env_cfg=self.env.cfg,
            num_envs=self.env.num_envs,
            is_distributed=self.is_distributed,
            gpu_world_size=self.gpu_world_size,
            gpu_global_rank=self.gpu_global_rank,
            device=self.device,
        )
        self.current_learning_iteration = 0

    def _construct_algorithm(self, obs) -> MultiCriticPPO:
        self.cfg["obs_groups"] = resolve_obs_groups(obs, self.cfg["obs_groups"], ["policy", "critic"])
        alg_cfg = resolve_rnd_config(dict(self.cfg["algorithm"]), obs, self.cfg["obs_groups"], self.env)
        alg_cfg = resolve_symmetry_config(alg_cfg, self.env)
        policy_cfg = dict(self.cfg["policy"])

        if self.cfg.get("empirical_normalization") is not None:
            warnings.warn(
                "The `empirical_normalization` parameter is deprecated. Please set `actor_obs_normalization` and "
                "`critic_obs_normalization` as part of the `policy` configuration instead.",
                DeprecationWarning,
            )
            if policy_cfg.get("actor_obs_normalization") is None:
                policy_cfg["actor_obs_normalization"] = self.cfg["empirical_normalization"]
            if policy_cfg.get("critic_obs_normalization") is None:
                policy_cfg["critic_obs_normalization"] = self.cfg["empirical_normalization"]

        critic_groups = self.cfg.get("multi_critic_groups", None)
        if critic_groups is None:
            critic_groups = [[group] for group in self.cfg["obs_groups"].get("critic", [])]
        if len(critic_groups) == 0:
            raise ValueError("MultiCriticRunner requires a non-empty `multi_critic_groups` (or `obs_groups['critic']`).")

        policy_class_name = policy_cfg.pop("class_name", "MultiCriticActorCritic")
        if policy_class_name != "MultiCriticActorCritic":
            raise ValueError(
                "MultiCriticRunner expects policy.class_name='MultiCriticActorCritic', "
                f"got '{policy_class_name}'."
            )
        policy = MultiCriticActorCritic(
            obs,
            self.cfg["obs_groups"],
            self.env.num_actions,
            num_critics=len(critic_groups),
            critic_obs_groups=critic_groups,
            **policy_cfg,
        ).to(self.device)
        print(f"Multi-Critic Actor-Critic Model: {policy}")

        alg_class_name = alg_cfg.pop("class_name", "MultiCriticPPO")
        if alg_class_name not in {"PPO", "MultiCriticPPO"}:
            raise ValueError(
                "MultiCriticRunner expects algorithm.class_name to be 'PPO' or 'MultiCriticPPO', "
                f"got '{alg_class_name}'."
            )

        storage = MultiCriticRolloutStorage(
            num_envs=self.env.num_envs,
            num_transitions_per_env=self.cfg["num_steps_per_env"],
            obs=obs,
            actions_shape=[self.env.num_actions],
            num_critics=len(critic_groups),
            device=self.device,
        )
        alg = MultiCriticPPO(
            policy=policy,
            storage=storage,
            num_critics=len(critic_groups),
            reward_group_weights=self.cfg.get("multi_critic_reward_weights", None),
            advantage_weights=self.cfg.get("multi_critic_advantage_weights", None),
            reward_group_extras_keys=self.cfg.get("multi_critic_reward_extras_keys", None),
            reward_term_groups=self.cfg.get("multi_critic_reward_term_groups", None),
            device=self.device,
            **alg_cfg,
            multi_gpu_cfg=self.cfg["multi_gpu"],
        )
        alg.bind_env(self.env)
        return alg
