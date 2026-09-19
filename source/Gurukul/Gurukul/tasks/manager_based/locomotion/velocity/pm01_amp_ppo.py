# Copyright (c) 2025- Shenzhen Zhongqing Robot Technology Co., Ltd. ("EngineAI")
# Copyright (c) 2026, Gurukul contributors.
# SPDX-License-Identifier: BSD-3-Clause
"""RSL-RL AMP extension for PM01 velocity following.

Adapted from ``engineai-robotics/engineai_amp`` commit
``83ba64bbb58a02e14483e52adce5f893f3f31cdf``.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.optim as optim
from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv
from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.storage import RolloutStorage
from tensordict import TensorDict

from .pm01_amp_data import PM01_AMP_FRAME_DIM, Pm01AmpMotionDataset


class Pm01AmpDiscriminator(nn.Module):
    """Least-squares AMP discriminator with per-frame normalization."""

    def __init__(
        self,
        *,
        frame_dim: int,
        history_length: int,
        hidden_dims: list[int],
        feature_normalization: bool,
        device: str,
    ) -> None:
        super().__init__()
        if frame_dim < 1 or history_length < 1 or not hidden_dims:
            raise ValueError("AMP discriminator dimensions must be positive and non-empty.")
        self.frame_dim = int(frame_dim)
        self.history_length = int(history_length)
        self.feature_normalization = bool(feature_normalization)
        self.feature_norm = EmpiricalNormalization(shape=(self.frame_dim,)).to(device)

        layers: list[nn.Module] = []
        input_dim = self.frame_dim * self.history_length
        for hidden_dim in hidden_dims:
            layers.extend((nn.Linear(input_dim, int(hidden_dim)), nn.ReLU()))
            input_dim = int(hidden_dim)
        layers.append(nn.Linear(input_dim, 1))
        self.model = nn.Sequential(*layers).to(device)

    def _normalized(self, values: torch.Tensor) -> torch.Tensor:
        if values.shape[-1] != self.frame_dim * self.history_length:
            raise ValueError(
                f"AMP discriminator expected {self.frame_dim * self.history_length} features, "
                f"received {values.shape[-1]}."
            )
        if not self.feature_normalization:
            return values
        frames = values.reshape(-1, self.frame_dim)
        return self.feature_norm(frames).reshape(values.shape)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.model(self._normalized(values)).squeeze(-1)

    @torch.no_grad()
    def update_normalization(self, values: torch.Tensor) -> None:
        if self.feature_normalization:
            self.feature_norm.update(values.reshape(-1, self.frame_dim))

    @torch.no_grad()
    def style_reward(self, values: torch.Tensor) -> torch.Tensor:
        score = self(values)
        return torch.clamp(1.0 - 0.25 * (score - 1.0).square(), min=0.0)

    def gradient_penalty(self, expert_values: torch.Tensor, scale: float) -> torch.Tensor:
        expert_values = expert_values.detach().requires_grad_(True)
        score = self(expert_values)
        gradient = torch.autograd.grad(
            outputs=score,
            inputs=expert_values,
            grad_outputs=torch.ones_like(score),
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return float(scale) * gradient.norm(2, dim=1).square().mean()


class Pm01AmpPPO(PPO):
    """PPO with EngineAI's walking-style discriminator reward."""

    def __init__(
        self,
        actor: MLPModel,
        critic: MLPModel,
        storage: RolloutStorage,
        *,
        amp_dataset_path: str,
        amp_dataset_sha256: str,
        amp_history_length: int = 5,
        amp_frame_dim: int = PM01_AMP_FRAME_DIM,
        amp_policy_dt: float,
        amp_style_reward_weight: float = 0.5,
        amp_style_reward_warmup_iterations: int = 1_500,
        amp_style_reward_ramp_iterations: int = 1_500,
        amp_discriminator_hidden_dims: tuple[int, ...] = (512, 256, 128),
        amp_feature_normalization: bool = True,
        amp_discriminator_learning_rate: float = 1.0e-4,
        amp_discriminator_weight_decay: float = 1.0e-4,
        amp_discriminator_update_interval: int = 4,
        amp_batch_size: int = 16_384,
        amp_gradient_penalty_scale: float = 10.0,
        amp_reference_min_speed: float = 0.0,
        device: str = "cpu",
        **kwargs,
    ) -> None:
        if amp_style_reward_weight < 0.0:
            raise ValueError("amp_style_reward_weight must be non-negative.")
        if amp_style_reward_warmup_iterations < 0 or amp_style_reward_ramp_iterations < 0:
            raise ValueError("AMP style-reward warm-up and ramp iterations must be non-negative.")
        if amp_discriminator_update_interval < 1 or amp_batch_size < 1:
            raise ValueError("AMP update interval and batch size must be positive.")

        self.amp_policy_dt = float(amp_policy_dt)
        self.amp_style_reward_weight = float(amp_style_reward_weight)
        self.amp_style_reward_warmup_iterations = int(amp_style_reward_warmup_iterations)
        self.amp_style_reward_ramp_iterations = int(amp_style_reward_ramp_iterations)
        self.amp_discriminator_update_interval = int(amp_discriminator_update_interval)
        self.amp_batch_size = int(amp_batch_size)
        self.amp_gradient_penalty_scale = float(amp_gradient_penalty_scale)
        self.amp_update_counter = 0

        self.amp_dataset = Pm01AmpMotionDataset(
            amp_dataset_path,
            history_length=amp_history_length,
            policy_dt=self.amp_policy_dt,
            device=device,
            expected_sha256=amp_dataset_sha256,
            min_planar_speed=amp_reference_min_speed,
        )
        if self.amp_dataset.frame_dim != int(amp_frame_dim):
            raise ValueError(
                f"AMP dataset frame dimension {self.amp_dataset.frame_dim} does not match "
                f"configured dimension {amp_frame_dim}."
            )
        self.amp_discriminator = Pm01AmpDiscriminator(
            frame_dim=amp_frame_dim,
            history_length=amp_history_length,
            hidden_dims=amp_discriminator_hidden_dims,
            feature_normalization=amp_feature_normalization,
            device=device,
        )
        # Weight decay is AMP's standard guard against a discriminator that separates the two
        # distributions so cleanly that the style reward saturates at zero and stops informing PPO.
        self.amp_discriminator_optimizer = optim.Adam(
            self.amp_discriminator.parameters(),
            lr=float(amp_discriminator_learning_rate),
            weight_decay=float(amp_discriminator_weight_decay),
        )

        super().__init__(actor=actor, critic=critic, storage=storage, device=device, **kwargs)
        if self.is_multi_gpu:
            raise NotImplementedError("PM01 AMP discriminator training currently supports one GPU process.")

    def _effective_style_reward_weight(self) -> float:
        """Style-prior weight for this update.

        With both schedule lengths at zero (the default) the prior is live from the first
        iteration, matching upstream. A non-zero warm-up holds the prior out until PPO has
        learned the velocity task, then a non-zero ramp fades it in.
        """
        if self.amp_update_counter < self.amp_style_reward_warmup_iterations:
            return 0.0
        iterations_after_warmup = self.amp_update_counter - self.amp_style_reward_warmup_iterations
        if self.amp_style_reward_ramp_iterations == 0:
            return self.amp_style_reward_weight
        progress = min(iterations_after_warmup / self.amp_style_reward_ramp_iterations, 1.0)
        return self.amp_style_reward_weight * progress

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> Pm01AmpPPO:
        """Inject the runtime policy period, validate AMP observations, then construct PPO."""
        history_length = int(cfg["algorithm"]["amp_history_length"])
        frame_dim = int(cfg["algorithm"]["amp_frame_dim"])
        expected_dim = history_length * frame_dim
        if "amp" not in obs.keys() or obs["amp"].shape[-1] != expected_dim:
            actual = None if "amp" not in obs.keys() else obs["amp"].shape[-1]
            raise ValueError(f"PM01 AMP observation must have width {expected_dim}; received {actual}.")
        cfg["algorithm"]["amp_policy_dt"] = float(env.unwrapped.step_dt)
        return PPO.construct_algorithm(obs, env, cfg, device)  # type: ignore[return-value]

    def process_env_step(
        self,
        obs: TensorDict,
        rewards: torch.Tensor,
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> None:
        with torch.no_grad():
            raw_style_reward = self.amp_discriminator.style_reward(obs["amp"])
            effective_style_weight = self._effective_style_reward_weight()
            style_reward = self.amp_policy_dt * effective_style_weight * raw_style_reward
        task_reward = rewards.clone()
        log_values = extras.setdefault("log", {})
        log_values["Step_Reward/amp_style"] = style_reward
        log_values["Step_Reward/task"] = task_reward
        # Unweighted discriminator reward in [0, 1]. Near 0 means the discriminator has
        # saturated and the prior carries no signal, whatever the configured weight.
        log_values["Metrics/amp_style_raw"] = raw_style_reward
        log_values["Metrics/amp_style_weight"] = torch.full_like(task_reward, effective_style_weight)
        super().process_env_step(obs, task_reward + style_reward, dones, extras)

    def _update_amp_discriminator(self) -> dict[str, float]:
        mean_loss = 0.0
        mean_gradient_penalty = 0.0
        mean_expert_score = 0.0
        mean_policy_score = 0.0
        num_updates = 0

        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        for batch in generator:
            policy_values = batch.observations["amp"].detach()
            if policy_values.shape[0] > self.amp_batch_size:
                indices = torch.randperm(policy_values.shape[0], device=policy_values.device)[: self.amp_batch_size]
                policy_values = policy_values[indices]
            expert_values = self.amp_dataset.sample(policy_values.shape[0])

            expert_score = self.amp_discriminator(expert_values)
            policy_score = self.amp_discriminator(policy_values)
            expert_loss = torch.nn.functional.mse_loss(expert_score, torch.ones_like(expert_score))
            policy_loss = torch.nn.functional.mse_loss(policy_score, -torch.ones_like(policy_score))
            gradient_penalty = self.amp_discriminator.gradient_penalty(expert_values, self.amp_gradient_penalty_scale)
            loss = 0.5 * (expert_loss + policy_loss) + gradient_penalty

            self.amp_discriminator_optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.amp_discriminator.parameters(), self.max_grad_norm)
            self.amp_discriminator_optimizer.step()
            self.amp_discriminator.update_normalization(torch.cat((expert_values, policy_values), dim=0))

            mean_loss += float(loss.item())
            mean_gradient_penalty += float(gradient_penalty.item())
            mean_expert_score += float(expert_score.mean().item())
            mean_policy_score += float(policy_score.mean().item())
            num_updates += 1

        divisor = max(num_updates, 1)
        return {
            "amp_discriminator": mean_loss / divisor,
            "amp_gradient_penalty": mean_gradient_penalty / divisor,
            "amp_expert_score": mean_expert_score / divisor,
            "amp_policy_score": mean_policy_score / divisor,
        }

    def update(self) -> dict[str, float]:
        amp_metrics: dict[str, float] = {}
        if self.amp_update_counter % self.amp_discriminator_update_interval == 0:
            amp_metrics = self._update_amp_discriminator()
        ppo_metrics = super().update()
        self.amp_update_counter += 1
        return {**amp_metrics, **ppo_metrics}

    def save(self) -> dict:
        saved = super().save()
        saved.update(
            {
                "amp_discriminator_state_dict": self.amp_discriminator.state_dict(),
                "amp_discriminator_optimizer_state_dict": self.amp_discriminator_optimizer.state_dict(),
                "amp_update_counter": self.amp_update_counter,
            }
        )
        return saved

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        should_load_amp = load_cfg is None or load_cfg.get("amp", True)
        if should_load_amp:
            self.amp_discriminator.load_state_dict(loaded_dict["amp_discriminator_state_dict"], strict=strict)
            self.amp_discriminator_optimizer.load_state_dict(loaded_dict["amp_discriminator_optimizer_state_dict"])
            self.amp_update_counter = int(loaded_dict.get("amp_update_counter", 0))
        return load_iteration
