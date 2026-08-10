"""Policy-side contact trail memory actor-critic for RSL-RL PPO."""

from __future__ import annotations

import copy
import os
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rsl_rl.algorithms import PPO
from rsl_rl.utils import resolve_nn_activation, resolve_optimizer

from Gurukul.utils.contact_trail_memory import (
    CONTACT_FEATURE_DIM,
    ContactQualityHead,
    ContactTrailConfig,
    ContactTrailEncoder,
    ContactTrailMemory,
    ContactTrailWriteNet,
)

from .rsl_rl_compat import EmpiricalNormalization, MLP, Memory


class ContactTrailActorCritic(nn.Module):
    """Actor-critic with policy-side egocentric contact trail memory."""

    is_recurrent = True

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] | None = None,
        critic_hidden_dims: list[int] | None = None,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        rnn_type: str = "gru",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        use_gru: bool = True,
        contact_trail_events_group: str = "contact_trail_events",
        contact_trail_pose_group: str = "contact_trail_pose",
        foot_pos_b_group: str = "foot_pos_b",
        num_envs: int = 4096,
        contact_trail_cfg: dict[str, Any] | None = None,
        cnn_latent_dim: int = 128,
        proprio_hidden_dim: int = 128,
        use_contact_quality_aux_loss: bool = True,
        debug_save_maps: bool = False,
        debug_save_interval: int = 500,
        log_stats_interval: int = 24,
        **kwargs,
    ):
        if kwargs:
            print(
                "ContactTrailActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str(list(kwargs.keys()))
            )
        super().__init__()

        actor_hidden_dims = actor_hidden_dims or [512, 256, 128]
        critic_hidden_dims = critic_hidden_dims or [512, 256, 128]

        self.obs_groups = obs_groups
        self.contact_trail_events_group = contact_trail_events_group
        self.contact_trail_pose_group = contact_trail_pose_group
        self.foot_pos_b_group = foot_pos_b_group
        self.use_gru = bool(use_gru)
        self.use_contact_quality_aux_loss = bool(use_contact_quality_aux_loss)
        self.debug_save_maps = bool(debug_save_maps)
        self.debug_save_interval = max(1, int(debug_save_interval))
        self.log_stats_interval = max(1, int(log_stats_interval))
        self._log_dir: str | None = None
        self._debug_iteration = 0
        self._log_stats_counter = 0
        self._pending_log_stats: dict[str, torch.Tensor] = {}
        self._last_foot_pos_b: torch.Tensor | None = None

        cfg_dict = contact_trail_cfg or {}
        self.trail_cfg = ContactTrailConfig(
            num_channels=int(cfg_dict.get("num_channels", 8)),
            grid_size=tuple(cfg_dict.get("grid_size", (40, 40))),
            resolution=float(cfg_dict.get("resolution", 0.05)),
            decay=float(cfg_dict.get("decay", 0.985)),
            write_radius=int(cfg_dict.get("write_radius", 1)),
            write_mode=str(cfg_dict.get("write_mode", "learned")),
            use_warp=bool(cfg_dict.get("use_warp", True)),
            write_only_on_contact=bool(cfg_dict.get("write_only_on_contact", True)),
            contact_force_threshold=float(cfg_dict.get("contact_force_threshold", 1.0)),
            slip_velocity_scale=float(cfg_dict.get("slip_velocity_scale", 0.5)),
            slip_k=float(cfg_dict.get("slip_k", 5.0)),
            impact_force_threshold=float(cfg_dict.get("impact_force_threshold", 100.0)),
            impact_force_scale=float(cfg_dict.get("impact_force_scale", 200.0)),
            map_clamp=float(cfg_dict.get("map_clamp", 5.0)),
            debug=bool(cfg_dict.get("debug", False)),
        )
        self.trail_cfg.learned_write = self.trail_cfg.write_mode == "learned"

        offset = 0
        proprio_slices: list[tuple[int, int]] = []
        events_slice: tuple[int, int] | None = None
        pose_slice: tuple[int, int] | None = None
        foot_slice: tuple[int, int] | None = None
        for group_name in obs_groups["policy"]:
            dim = int(obs[group_name].shape[-1])
            start, end = offset, offset + dim
            if group_name == contact_trail_events_group:
                events_slice = (start, end)
            elif group_name == contact_trail_pose_group:
                pose_slice = (start, end)
            elif group_name == foot_pos_b_group:
                foot_slice = (start, end)
            else:
                proprio_slices.append((start, end))
            offset = end

        if events_slice is None or pose_slice is None or foot_slice is None:
            raise ValueError(
                "ContactTrailActorCritic requires contact_trail_events, contact_trail_pose, and foot_pos_b "
                "observation groups in policy obs_groups."
            )

        self.proprio_slices = proprio_slices
        self.events_slice = events_slice
        self.pose_slice = pose_slice
        self.foot_slice = foot_slice
        self.total_actor_obs_dim = offset
        self.proprio_dim = sum(end - start for start, end in proprio_slices)

        self.events_dim = events_slice[1] - events_slice[0]
        self.foot_dim = foot_slice[1] - foot_slice[0]
        if self.events_dim % CONTACT_FEATURE_DIM != 0:
            raise ValueError(
                f"contact_trail_events dim {self.events_dim} is not divisible by feature dim {CONTACT_FEATURE_DIM}."
            )
        self.num_feet = self.events_dim // CONTACT_FEATURE_DIM
        if self.foot_dim != self.num_feet * 3:
            raise ValueError(f"foot_pos_b dim {self.foot_dim} != num_feet * 3 ({self.num_feet * 3}).")

        self.total_critic_obs_dim = sum(int(obs[group_name].shape[-1]) for group_name in obs_groups["critic"])

        def _act() -> nn.Module:
            return copy.deepcopy(resolve_nn_activation(activation))

        self.write_net = ContactTrailWriteNet(CONTACT_FEATURE_DIM, self.trail_cfg.num_channels, activation=activation)
        self.contact_trail_memory = ContactTrailMemory(
            num_envs=int(num_envs),
            config=self.trail_cfg,
            contact_feature_dim=CONTACT_FEATURE_DIM,
            write_net=self.write_net if self.trail_cfg.write_mode == "learned" else None,
            device="cpu",
        )
        self.trail_encoder = ContactTrailEncoder(
            num_channels=self.trail_cfg.num_channels,
            grid_size=self.trail_cfg.grid_size,
            latent_dim=int(cnn_latent_dim),
            activation=activation,
        )
        self.quality_head = ContactQualityHead(int(cnn_latent_dim), self.trail_cfg.grid_size)

        self.proprio_encoder = nn.Sequential(
            nn.Linear(self.proprio_dim, int(proprio_hidden_dim)),
            _act(),
        )
        fusion_dim = int(proprio_hidden_dim) + int(cnn_latent_dim)
        self.memory = None
        if self.use_gru:
            self.memory = Memory(
                input_size=fusion_dim,
                type=rnn_type,
                num_layers=int(rnn_num_layers),
                hidden_size=int(rnn_hidden_dim),
            )
            actor_input_dim = int(rnn_hidden_dim)
            critic_input_dim = int(rnn_hidden_dim)
        else:
            actor_input_dim = fusion_dim
            critic_input_dim = fusion_dim

        self.actor = MLP(
            input_dim=actor_input_dim,
            output_dim=int(num_actions),
            hidden_dims=list(actor_hidden_dims),
            activation=activation,
        )

        if self.use_gru:
            self.memory_c = Memory(
                input_size=self.total_critic_obs_dim,
                type=rnn_type,
                num_layers=int(rnn_num_layers),
                hidden_size=int(rnn_hidden_dim),
            )
        else:
            self.critic_encoder = nn.Sequential(
                nn.Linear(self.total_critic_obs_dim, int(rnn_hidden_dim)),
                _act(),
            )
            critic_input_dim = int(rnn_hidden_dim)

        self.critic = MLP(
            input_dim=critic_input_dim,
            output_dim=1,
            hidden_dims=list(critic_hidden_dims),
            activation=activation,
        )

        self.actor_obs_normalization = bool(actor_obs_normalization)
        self.actor_obs_normalizer = (
            EmpiricalNormalization(self.total_actor_obs_dim) if self.actor_obs_normalization else nn.Identity()
        )
        self.critic_obs_normalization = bool(critic_obs_normalization)
        self.critic_obs_normalizer = (
            EmpiricalNormalization(self.total_critic_obs_dim) if self.critic_obs_normalization else nn.Identity()
        )

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}.")

        self.distribution = None
        Normal.set_default_validate_args(False)
        self._aux_cache: dict[str, torch.Tensor | None] = {}

    def _to_device(self, device: torch.device) -> None:
        self.contact_trail_memory.to(device)

    def _split_actor_obs(self, actor_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        proprio_parts = [actor_obs[..., start:end] for start, end in self.proprio_slices]
        proprio = torch.cat(proprio_parts, dim=-1) if proprio_parts else actor_obs.new_zeros((*actor_obs.shape[:-1], 0))
        events = actor_obs[..., self.events_slice[0] : self.events_slice[1]]
        pose = actor_obs[..., self.pose_slice[0] : self.pose_slice[1]]
        foot_pos_b = actor_obs[..., self.foot_slice[0] : self.foot_slice[1]]
        return proprio, events, pose, foot_pos_b

    def _update_trail_memory(
        self,
        events: torch.Tensor,
        pose: torch.Tensor,
        foot_pos_b: torch.Tensor,
        masks: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = events.device
        self._to_device(device)
        self._last_foot_pos_b = foot_pos_b

        if events.ndim == 3 and masks is not None:
            time_steps, batch_size, _ = events.shape
            base_pos_w = pose[..., :3]
            base_quat_w = pose[..., 3:7]
            foot_pos = foot_pos_b.view(time_steps, batch_size, self.num_feet, 3)
            features = events.view(time_steps, batch_size, self.num_feet, CONTACT_FEATURE_DIM)
            trail_map = self.contact_trail_memory.update_sequence(
                base_pos_w=base_pos_w,
                base_quat_w=base_quat_w,
                foot_pos_b=foot_pos,
                contact_features=features.reshape(time_steps, batch_size, -1),
                masks=masks,
                dt=0.0,
            )
            self._collect_logging_stats(trail_map[-1], features[-1])
            return trail_map

        flat_events = events.reshape(-1, self.events_dim)
        flat_pose = pose.reshape(-1, 7)
        flat_foot = foot_pos_b.reshape(-1, self.foot_dim)

        base_pos_w = flat_pose[:, :3]
        base_quat_w = flat_pose[:, 3:7]
        foot_pos = flat_foot.view(-1, self.num_feet, 3)

        trail_map = self.contact_trail_memory.update(
            base_pos_w=base_pos_w,
            base_quat_w=base_quat_w,
            foot_pos_b=foot_pos,
            contact_features=flat_events,
            dt=0.0,
        )
        features = flat_events.view(-1, self.num_feet, CONTACT_FEATURE_DIM)
        self._collect_logging_stats(trail_map, features)
        return trail_map

    def _collect_logging_stats(self, trail_map: torch.Tensor, features: torch.Tensor) -> None:
        self._log_stats_counter += 1
        if self._log_stats_counter % self.log_stats_interval != 0:
            return

        with torch.no_grad():
            quality = self.contact_trail_memory.compute_quality_targets(features)
            stats = {
                "ContactTrail/map_abs_mean": trail_map.abs().mean().detach(),
                "ContactTrail/map_nonzero_frac": (trail_map.abs() > 1.0e-5).float().mean().detach(),
                "ContactTrail/quality_mean": quality.mean().detach(),
                "ContactTrail/contact_rate": features[..., 0].mean().detach(),
            }
            if self.trail_cfg.write_mode == "learned" and self.write_net is not None:
                norm_features = ContactTrailMemory.normalize_contact_features(features, self.trail_cfg)
                _, alphas = self.write_net(norm_features)
                stats["ContactTrail/write_alpha_mean"] = alphas.mean().detach()
        self._pending_log_stats = stats

    def pop_logging_stats(self) -> dict[str, float]:
        stats = {tag: float(value.cpu()) for tag, value in self._pending_log_stats.items()}
        self._pending_log_stats.clear()
        return stats

    def set_debug_context(self, log_dir: str | None, iteration: int = 0) -> None:
        self._log_dir = log_dir
        self._debug_iteration = int(iteration)

    def maybe_save_debug_maps(self) -> None:
        if not self.debug_save_maps or self._log_dir is None:
            return
        if self._debug_iteration <= 0 or self._debug_iteration % self.debug_save_interval != 0:
            return
        try:
            from Gurukul.utils.contact_trail_viz import save_contact_trail_images
        except ImportError:
            return

        trail_map = self.contact_trail_memory.get_map()
        foot_pos = self._last_foot_pos_b
        if foot_pos is not None and foot_pos.ndim == 3:
            foot_pos = foot_pos[-1]
        output_dir = os.path.join(
            self._log_dir,
            "contact_trail_debug",
            f"iter_{self._debug_iteration:06d}",
        )
        save_contact_trail_images(
            trail_map,
            output_dir,
            prefix="contact_trail",
            env_idx=0,
            foot_pos_b=foot_pos,
            grid_size=self.trail_cfg.grid_size,
            resolution=self.trail_cfg.resolution,
        )

    def _forward_actor(
        self,
        obs,
        masks: torch.Tensor | None = None,
        hidden_states=None,
        cache_aux: bool = True,
        detach_memory: bool = False,
    ) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)
        proprio, events, pose, foot_pos_b = self._split_actor_obs(actor_obs)

        freeze_memory = detach_memory and self.trail_cfg.write_mode != "learned"
        memory_ctx = torch.no_grad() if freeze_memory else torch.enable_grad()
        with memory_ctx:
            trail_map = self._update_trail_memory(events, pose, foot_pos_b, masks=masks)
        if trail_map.ndim == 5:
            time_steps, batch_size = trail_map.shape[:2]
            trail_latent = self.trail_encoder(trail_map.reshape(time_steps * batch_size, *trail_map.shape[2:])).reshape(
                time_steps, batch_size, -1
            )
            proprio_latent = self.proprio_encoder(proprio.reshape(-1, self.proprio_dim)).reshape(
                time_steps, batch_size, -1
            )
            fused = torch.cat((proprio_latent, trail_latent), dim=-1)
        else:
            trail_latent = self.trail_encoder(trail_map)
            proprio_latent = self.proprio_encoder(proprio.reshape(-1, self.proprio_dim))
            fused = torch.cat((proprio_latent, trail_latent), dim=-1)

        if self.use_gru:
            rnn_out = self.memory(fused, masks, hidden_states)
            actor_features = rnn_out.squeeze(0) if rnn_out.ndim == 3 else rnn_out
        else:
            actor_features = fused if fused.ndim == 2 else fused.reshape(-1, fused.shape[-1])

        action_mean = self.actor(actor_features.reshape(-1, actor_features.shape[-1]))

        if cache_aux and self.use_contact_quality_aux_loss:
            trail_latent_flat = trail_latent.reshape(-1, trail_latent.shape[-1])
            quality_pred = self.quality_head(trail_latent_flat)
            features = events.reshape(-1, self.num_feet, CONTACT_FEATURE_DIM)
            quality_target = self.contact_trail_memory.compute_quality_targets(features)
            self._aux_cache = {
                "quality_pred": quality_pred,
                "quality_target": quality_target,
                "foot_pos_b": foot_pos_b.reshape(-1, self.num_feet, 3),
                "trail_map": trail_map[-1] if trail_map.ndim == 5 else trail_map,
            }
        return action_mean

    def compute_auxiliary_losses(self) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        zero = torch.zeros((), device=device)
        if not self._aux_cache or not self.use_contact_quality_aux_loss:
            return {"contact_quality": zero, "aux_total": zero}

        quality_pred = self._aux_cache.get("quality_pred")
        quality_target = self._aux_cache.get("quality_target")
        foot_pos_b = self._aux_cache.get("foot_pos_b")
        if quality_pred is None or quality_target is None or foot_pos_b is None:
            return {"contact_quality": zero, "aux_total": zero}

        grid_h, grid_w = self.trail_cfg.grid_size
        resolution = self.trail_cfg.resolution
        rows = torch.floor(foot_pos_b[..., 1] / resolution + grid_h / 2.0).long().clamp(0, grid_h - 1)
        cols = torch.floor(foot_pos_b[..., 0] / resolution + grid_w / 2.0).long().clamp(0, grid_w - 1)
        contact_mask = quality_target.abs() > 1.0e-5
        if not torch.any(contact_mask):
            return {"contact_quality": zero, "aux_total": zero}

        batch_idx = torch.arange(quality_pred.shape[0], device=device).unsqueeze(1).expand_as(rows)
        pred_vals = quality_pred[batch_idx, 0, rows, cols]
        loss = F.mse_loss(pred_vals[contact_mask], quality_target[contact_mask])
        return {"contact_quality": loss, "aux_total": loss}

    def reset(self, dones=None, hidden_states=None):
        if dones is not None:
            done_ids = (dones > 0).nonzero(as_tuple=False).flatten()
            if len(done_ids) > 0:
                self.contact_trail_memory.reset(done_ids)
        if self.use_gru:
            if hidden_states is None:
                self.memory.reset(dones=dones)
                self.memory_c.reset(dones=dones)
            else:
                actor_hidden = hidden_states[0] if isinstance(hidden_states, tuple) else hidden_states
                critic_hidden = hidden_states[1] if isinstance(hidden_states, tuple) and len(hidden_states) > 1 else None
                self.memory.reset(dones=dones, hidden_states=actor_hidden)
                self.memory_c.reset(dones=dones, hidden_states=critic_hidden)

    def forward(self, obs, *args, **kwargs):
        del args, kwargs
        return self.act_inference(obs)

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

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def get_actor_obs(self, obs):
        return torch.cat([obs[group_name] for group_name in self.obs_groups["policy"]], dim=-1)

    def get_critic_obs(self, obs):
        return torch.cat([obs[group_name] for group_name in self.obs_groups["critic"]], dim=-1)

    def _update_distribution_from_mean(self, mean: torch.Tensor) -> None:
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        else:
            std = torch.exp(self.log_std).expand_as(mean)
        self.distribution = Normal(mean, std)

    def act(self, obs, masks=None, hidden_states=None, cache_aux: bool = True):
        mean = self._forward_actor(
            obs,
            masks=masks,
            hidden_states=hidden_states,
            cache_aux=cache_aux,
            detach_memory=masks is not None,
        )
        self._update_distribution_from_mean(mean)
        return self.distribution.sample()

    def act_inference(self, obs):
        with torch.no_grad():
            mean = self._forward_actor(obs, masks=None, hidden_states=None, cache_aux=False)
        return mean

    def evaluate(self, obs, masks=None, hidden_states=None):
        critic_obs = self.critic_obs_normalizer(self.get_critic_obs(obs))
        if self.use_gru:
            critic_hidden = hidden_states[1] if isinstance(hidden_states, tuple) and len(hidden_states) > 1 else hidden_states
            critic_state = self.memory_c(critic_obs, masks, critic_hidden).squeeze(0)
        else:
            critic_state = self.critic_encoder(critic_obs)
        return self.critic(critic_state)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_hidden_states(self):
        if not self.use_gru:
            return None
        return self.memory.hidden_states, self.memory_c.hidden_states

    def detach_hidden_states(self, dones=None):
        if self.use_gru:
            self.memory.detach_hidden_states(dones)
            self.memory_c.detach_hidden_states(dones)

    def reset_rollout_memory(self, num_envs: int) -> None:
        """Start a new PPO rollout with an empty, correctly-sized trail map."""
        device = next(self.parameters()).device
        self._to_device(device)
        self.contact_trail_memory._ensure_num_envs(int(num_envs), device)
        self.contact_trail_memory.reset()


class ContactTrailPPO(PPO):
    """PPO with optional contact-quality auxiliary loss."""

    policy: ContactTrailActorCritic

    _POLICY_CFG_KEYS = (
        "actor_hidden_dims",
        "critic_hidden_dims",
        "activation",
        "init_noise_std",
        "noise_std_type",
        "actor_obs_normalization",
        "critic_obs_normalization",
        "rnn_type",
        "rnn_hidden_dim",
        "rnn_num_layers",
        "use_gru",
        "cnn_latent_dim",
        "proprio_hidden_dim",
        "use_contact_quality_aux_loss",
        "debug_save_maps",
        "debug_save_interval",
        "log_stats_interval",
        "contact_trail_events_group",
        "contact_trail_pose_group",
        "foot_pos_b_group",
        "contact_trail_cfg",
    )

    def __init__(
        self,
        policy,
        storage=None,
        contact_quality_loss_coef: float = 0.01,
        multi_gpu_cfg: dict | None = None,
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        **kwargs,
    ):
        if storage is None:
            raise ValueError("ContactTrailPPO requires an initialized RolloutStorage instance.")
        if rnd_cfg is not None or symmetry_cfg is not None:
            raise NotImplementedError("ContactTrailPPO does not support RND or symmetry extensions.")

        self.device = kwargs.pop("device", "cpu")
        self.is_multi_gpu = multi_gpu_cfg is not None
        self.gpu_global_rank = multi_gpu_cfg["global_rank"] if multi_gpu_cfg is not None else 0
        self.gpu_world_size = multi_gpu_cfg["world_size"] if multi_gpu_cfg is not None else 1
        self.rnd = None
        self.intrinsic_rewards = None
        self.symmetry = None
        self.policy = policy.to(self.device)
        self.actor = self.policy
        self.critic = self.policy
        self._raw_actor = self.policy
        self._raw_critic = self.policy
        self.storage = storage
        from rsl_rl.storage import RolloutStorage

        self.transition = RolloutStorage.Transition()
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
            raise TypeError(f"Unexpected ContactTrailPPO arguments: {sorted(kwargs)}")
        self.contact_quality_loss_coef = float(contact_quality_loss_coef)
        self.optimizer = resolve_optimizer(optimizer_name)(self.policy.parameters(), lr=self.learning_rate)

    @staticmethod
    def construct_algorithm(obs, env, cfg: dict, device: str):
        """Construct contact-trail PPO from a shared actor-critic policy config."""
        from rsl_rl.storage import RolloutStorage
        from rsl_rl.utils import resolve_callable, resolve_obs_groups

        alg_class = ContactTrailPPO
        policy_class = ContactTrailActorCritic
        algorithm_class_name = cfg["algorithm"].pop("class_name", None)
        if algorithm_class_name and algorithm_class_name not in {"ContactTrailPPO", "PPO"}:
            if ":" in algorithm_class_name or "." in algorithm_class_name:
                alg_class = resolve_callable(algorithm_class_name)
            elif algorithm_class_name != "ContactTrailPPO":
                raise ValueError(f"Unsupported contact-trail algorithm class: {algorithm_class_name}")

        policy_cfg = dict(cfg.get("policy") or {})
        policy_class_name = policy_cfg.pop("class_name", None)
        if policy_class_name and (":" in policy_class_name or "." in policy_class_name):
            policy_class = resolve_callable(policy_class_name)

        actor_cfg = cfg.get("actor", {}) or {}
        critic_cfg = cfg.get("critic", {}) or {}
        if actor_cfg:
            policy_cfg.setdefault("actor_hidden_dims", actor_cfg.get("hidden_dims"))
            policy_cfg.setdefault("activation", actor_cfg.get("activation"))
            policy_cfg.setdefault("actor_obs_normalization", actor_cfg.get("obs_normalization", False))
            policy_cfg.setdefault("init_noise_std", actor_cfg.get("init_noise_std", 1.0))
            policy_cfg.setdefault("noise_std_type", actor_cfg.get("noise_std_type", "scalar"))
            policy_cfg.setdefault("rnn_type", actor_cfg.get("rnn_type"))
            policy_cfg.setdefault("rnn_hidden_dim", actor_cfg.get("rnn_hidden_dim"))
            policy_cfg.setdefault("rnn_num_layers", actor_cfg.get("rnn_num_layers"))
        if critic_cfg:
            policy_cfg.setdefault("critic_hidden_dims", critic_cfg.get("hidden_dims"))
            policy_cfg.setdefault("critic_obs_normalization", critic_cfg.get("obs_normalization", False))

        default_sets = ["policy", "critic"]
        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
        for key in ContactTrailPPO._POLICY_CFG_KEYS:
            if key in cfg["algorithm"]:
                policy_cfg[key] = cfg["algorithm"].pop(key)
        policy_cfg = {key: value for key, value in policy_cfg.items() if value is not None}
        policy_cfg["num_envs"] = env.num_envs
        policy = policy_class(obs, cfg["obs_groups"], env.num_actions, **policy_cfg).to(device)
        print(f"Contact Trail Actor-Critic Model: {policy}")
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        return alg_class(policy, storage=storage, device=device, **cfg["algorithm"], multi_gpu_cfg=cfg["multi_gpu"])

    def act(self, obs):
        if self.policy.is_recurrent:
            self.transition.hidden_states = self.policy.get_hidden_states()

        actions = self.policy.act(obs, cache_aux=False).detach()
        self.transition.actions = actions
        self.transition.values = self.policy.evaluate(obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(actions).detach()
        self.transition.distribution_params = (
            self.policy.action_mean.detach(),
            self.policy.action_std.detach(),
        )
        self.transition.observations = obs
        return actions

    def process_env_step(self, obs, rewards, dones, extras):
        self.policy.update_normalization(obs)
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in extras:
            time_outs = extras["time_outs"].reshape(-1, 1).to(self.device)
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * time_outs,
                1,
            )
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs):
        st = self.storage
        hidden_states = self.policy.get_hidden_states() if self.policy.is_recurrent else None
        last_values = self.policy.evaluate(obs).detach()
        if self.policy.is_recurrent:
            self.policy.reset(dones=None, hidden_states=hidden_states)

        advantage = 0
        for step in reversed(range(st.num_transitions_per_env)):
            next_values = last_values if step == st.num_transitions_per_env - 1 else st.values[step + 1]
            next_is_not_terminal = 1.0 - st.dones[step].float()
            delta = st.rewards[step] + next_is_not_terminal * self.gamma * next_values - st.values[step]
            advantage = delta + next_is_not_terminal * self.gamma * self.lam * advantage
            st.returns[step] = advantage + st.values[step]
        st.advantages = st.returns - st.values
        if not self.normalize_advantage_per_mini_batch:
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1e-8)

    def get_policy(self):
        return self.policy

    def train_mode(self):
        self.policy.train()

    def eval_mode(self):
        self.policy.eval()

    def save(self) -> dict:
        state_dict = self.policy.state_dict()
        return {
            "actor_state_dict": state_dict,
            "critic_state_dict": state_dict,
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        if load_cfg is None:
            load_cfg = {"actor": True, "critic": True, "optimizer": True, "iteration": True}
        if load_cfg.get("actor") or load_cfg.get("critic"):
            model_state = loaded_dict.get("actor_state_dict", loaded_dict.get("model_state_dict"))
            if model_state is not None:
                self.policy.load_state_dict(model_state, strict=strict)
        if load_cfg.get("optimizer"):
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        return load_cfg.get("iteration", False)

    def broadcast_parameters(self) -> None:
        model_params = [self.policy.state_dict()]
        torch.distributed.broadcast_object_list(model_params, src=0)
        self.policy.load_state_dict(model_params[0])

    def reduce_parameters(self) -> None:
        all_params = list(self.policy.parameters())
        grads = [param.grad.view(-1) for param in all_params if param.grad is not None]
        if not grads:
            return
        all_grads = torch.cat(grads)
        torch.distributed.all_reduce(all_grads, op=torch.distributed.ReduceOp.SUM)
        all_grads /= self.gpu_world_size
        offset = 0
        for param in all_params:
            if param.grad is not None:
                numel = param.numel()
                param.grad.data.copy_(all_grads[offset : offset + numel].view_as(param.grad.data))
                offset += numel

    @staticmethod
    def _unpack_batch(batch):
        if hasattr(batch, "observations"):
            old_mu_batch, old_sigma_batch = batch.old_distribution_params
            return (
                batch.observations,
                batch.actions,
                batch.values,
                batch.advantages,
                batch.returns,
                batch.old_actions_log_prob,
                old_mu_batch,
                old_sigma_batch,
                batch.hidden_states,
                batch.masks,
            )
        return batch

    @staticmethod
    def _flatten_time_env_batch(tensor):
        if tensor is None or tensor.ndim < 3:
            return tensor
        return tensor.reshape(-1, tensor.shape[-1]) if tensor.shape[-1] != 1 else tensor.reshape(-1)

    def update(self):
        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0
        mean_contact_quality = 0.0

        if self.policy.is_recurrent:
            generator = self.storage.recurrent_mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        else:
            generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)

        for batch in generator:
            (
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
            ) = self._unpack_batch(batch)
            if masks_batch is not None:
                actions_batch = self._flatten_time_env_batch(actions_batch)
                target_values_batch = self._flatten_time_env_batch(target_values_batch)
                advantages_batch = self._flatten_time_env_batch(advantages_batch)
                returns_batch = self._flatten_time_env_batch(returns_batch)
                old_actions_log_prob_batch = self._flatten_time_env_batch(old_actions_log_prob_batch)
                old_mu_batch = self._flatten_time_env_batch(old_mu_batch)
                old_sigma_batch = self._flatten_time_env_batch(old_sigma_batch)
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            actor_hidden = hid_states_batch[0] if hid_states_batch is not None else None
            self.policy.act(obs_batch, masks=masks_batch, hidden_states=actor_hidden)
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_states=hid_states_batch)
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy
            if masks_batch is not None:
                value_batch = self._flatten_time_env_batch(value_batch)
                entropy_batch = self._flatten_time_env_batch(entropy_batch)

            if value_batch.shape != returns_batch.shape:
                value_batch = value_batch.reshape_as(returns_batch)

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

            if getattr(self.policy, "use_contact_quality_aux_loss", False):
                contact_quality_loss = self.policy.compute_auxiliary_losses()["contact_quality"]
            else:
                contact_quality_loss = torch.zeros((), device=self.device)

            ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch))
            surrogate = -torch.squeeze(advantages_batch) * ratio
            surrogate_clipped = -torch.squeeze(advantages_batch) * torch.clamp(
                ratio, 1.0 - self.clip_param, 1.0 + self.clip_param
            )
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            if self.use_clipped_value_loss:
                value_clipped = target_values_batch + (value_batch - target_values_batch).clamp(
                    -self.clip_param, self.clip_param
                )
                value_losses = (value_batch - returns_batch).pow(2)
                value_losses_clipped = (value_clipped - returns_batch).pow(2)
                value_loss = torch.max(value_losses, value_losses_clipped).mean()
            else:
                value_loss = (returns_batch - value_batch).pow(2).mean()

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + self.contact_quality_loss_coef * contact_quality_loss
            )

            self.optimizer.zero_grad()
            loss.backward()
            if self.is_multi_gpu:
                self.reduce_parameters()
            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            self.optimizer.step()

            mean_value_loss += value_loss.item()
            mean_surrogate_loss += surrogate_loss.item()
            mean_entropy += entropy_batch.mean().item()
            mean_contact_quality += contact_quality_loss.item()

        num_updates = max(self.num_learning_epochs * self.num_mini_batches, 1)
        self.storage.clear()
        # Replays start from an empty trail map, so rollout collection uses the
        # same boundary condition.  Episode-local memory spans one rollout.
        if hasattr(self.policy, "reset_rollout_memory"):
            self.policy.reset_rollout_memory(self.storage.num_envs)

        return {
            "value": mean_value_loss / num_updates,
            "surrogate": mean_surrogate_loss / num_updates,
            "entropy": mean_entropy / num_updates,
            "contact_quality": mean_contact_quality / num_updates,
        }
