"""START single-stage policy and PPO extensions."""

from __future__ import annotations

import copy
import math
from collections import deque
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal

from rsl_rl.algorithms import PPO
from rsl_rl.utils import resolve_nn_activation, resolve_optimizer, unpad_trajectories

from .rsl_rl_compat import EmpiricalNormalization, MLP, Memory


_START_ADASMPL_SELECTOR = "_start_adasmpl_selector"


class _StartRefineUNet(nn.Module):
    """Lightweight U-Net-style refinement decoder for local heightmaps."""

    def __init__(self, map_h: int, map_w: int, base_channels: int = 16, activation: str = "elu"):
        super().__init__()
        self.map_h = int(map_h)
        self.map_w = int(map_w)

        def _act() -> nn.Module:
            return copy.deepcopy(resolve_nn_activation(activation))

        c = int(base_channels)
        self.enc1 = nn.Sequential(
            nn.Conv2d(1, c, kernel_size=3, padding=1),
            _act(),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            _act(),
        )
        self.enc2 = nn.Sequential(
            nn.Conv2d(c, 2 * c, kernel_size=3, padding=1),
            _act(),
            nn.Conv2d(2 * c, 2 * c, kernel_size=3, padding=1),
            _act(),
        )
        self.bottleneck = nn.Sequential(
            nn.Conv2d(2 * c, 4 * c, kernel_size=3, padding=1),
            _act(),
            nn.Conv2d(4 * c, 4 * c, kernel_size=3, padding=1),
            _act(),
        )
        self.dec2 = nn.Sequential(
            nn.Conv2d(4 * c + 2 * c, 2 * c, kernel_size=3, padding=1),
            _act(),
            nn.Conv2d(2 * c, 2 * c, kernel_size=3, padding=1),
            _act(),
        )
        self.dec1 = nn.Sequential(
            nn.Conv2d(2 * c + c, c, kernel_size=3, padding=1),
            _act(),
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            _act(),
        )
        self.out = nn.Conv2d(c, 1, kernel_size=1)

    def forward(self, rough_heightmap: torch.Tensor) -> torch.Tensor:
        shape_prefix = rough_heightmap.shape[:-1]
        flat = rough_heightmap.reshape(-1, rough_heightmap.shape[-1])
        x = flat.view(-1, 1, self.map_h, self.map_w)

        e1 = self.enc1(x)
        p1 = F.max_pool2d(e1, kernel_size=2, stride=2, ceil_mode=True)

        e2 = self.enc2(p1)
        p2 = F.max_pool2d(e2, kernel_size=2, stride=2, ceil_mode=True)

        b = self.bottleneck(p2)

        u2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([u2, e2], dim=1))

        u1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([u1, e1], dim=1))

        delta = self.out(d1).view(-1, self.map_h * self.map_w)
        refined = flat + delta
        return refined.view(*shape_prefix, self.map_h * self.map_w)


class StartActorCritic(nn.Module):
    """START actor-critic with TR-Net, I-E estimator, and AdaSmpl support."""

    is_recurrent = True

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: list[int] = [512, 256, 128],
        critic_hidden_dims: list[int] = [512, 256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        rnn_type: str = "gru",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        depth_obs_group: str = "depth_camera",
        depth_image_shape: tuple[int, int] = (60, 60),
        depth_backbone_channels: list[int] = [32, 64],
        depth_backbone_kernels: list[int] = [5, 3],
        depth_backbone_pool_kernel: int = 2,
        depth_backbone_image_fc_dim: int = 128,
        depth_backbone_latent_dim: int = 64,
        terrain_heightmap_obs_group: str = "terrain_heightmap",
        feet_heightmap_obs_group: str | None = "feet_heightmap",
        base_velocity_obs_group: str | None = "base_velocity_gt",
        terrain_map_shape: tuple[int, int] | None = None,
        tr_proprio_hidden_dim: int = 128,
        tr_rnn_hidden_dim: int | None = None,
        tr_rnn_num_layers: int | None = None,
        tr_refine_base_channels: int = 16,
        ie_prop_hidden_dim: int = 128,
        ie_rnn_hidden_dim: int | None = None,
        ie_rnn_num_layers: int | None = None,
        ie_map_latent_dim: int = 128,
        ie_transformer_dim: int = 256,
        ie_transformer_heads: int = 8,
        ie_transformer_layers: int = 2,
        ie_transformer_ff_dim: int = 512,
        explicit_latent_dim: int = 128,
        implicit_latent_dim: int = 64,
        critic_rnn_type: str | None = None,
        critic_rnn_hidden_dim: int | None = None,
        critic_rnn_num_layers: int | None = None,
        tr_rough_loss_weight: float = 1.0,
        tr_refined_loss_weight: float = 1.0,
        ie_kld_loss_weight: float = 1.0,
        ie_base_vel_loss_weight: float = 1.0,
        ie_body_map_loss_weight: float = 1.0,
        ie_feet_map_loss_weight: float = 1.0,
        ie_proprio_loss_weight: float = 1.0,
        adasmpl_enabled: bool = True,
        adasmpl_initial_probability: float = 1.0,
        adasmpl_min_probability: float = 0.0,
        adasmpl_max_probability: float = 1.0,
        **kwargs,
    ):
        if kwargs:
            print(
                "StartActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.obs_groups = obs_groups
        self.depth_obs_group = depth_obs_group
        self.terrain_heightmap_obs_group = terrain_heightmap_obs_group
        self.feet_heightmap_obs_group = feet_heightmap_obs_group
        self.base_velocity_obs_group = base_velocity_obs_group

        if depth_obs_group not in obs_groups["policy"]:
            raise ValueError(f"Depth observation group '{depth_obs_group}' not found in policy obs groups.")

        # Build actor observation slices.
        offset = 0
        proprio_slices: list[tuple[int, int]] = []
        depth_slice: tuple[int, int] | None = None
        num_actor_obs = 0
        for group_name in obs_groups["policy"]:
            dim = int(obs[group_name].shape[-1])
            start, end = offset, offset + dim
            if group_name == depth_obs_group:
                depth_slice = (start, end)
            else:
                proprio_slices.append((start, end))
            offset = end
            num_actor_obs += dim
        if depth_slice is None:
            raise ValueError(f"Failed to build depth slice for group '{depth_obs_group}'.")

        self.proprio_slices = proprio_slices
        self.depth_slice = depth_slice
        self.total_actor_obs_dim = num_actor_obs

        # Critic observation dimension.
        num_critic_obs = 0
        for group_name in obs_groups["critic"]:
            num_critic_obs += int(obs[group_name].shape[-1])
        self.total_critic_obs_dim = num_critic_obs

        # Optional supervision targets.
        if terrain_heightmap_obs_group not in obs:
            raise ValueError(f"Terrain heightmap group '{terrain_heightmap_obs_group}' not found in observations.")
        self.terrain_map_dim = int(obs[terrain_heightmap_obs_group].shape[-1])
        self.feet_map_dim = int(obs[feet_heightmap_obs_group].shape[-1]) if feet_heightmap_obs_group in obs else 0
        self.base_velocity_dim = int(obs[base_velocity_obs_group].shape[-1]) if base_velocity_obs_group in obs else 0

        self.map_h, self.map_w = self._resolve_map_shape(self.terrain_map_dim, terrain_map_shape)

        self.depth_h = int(depth_image_shape[0])
        self.depth_w = int(depth_image_shape[1])
        self.depth_dim = self.depth_slice[1] - self.depth_slice[0]
        if self.depth_h * self.depth_w != self.depth_dim:
            raise ValueError(
                f"Depth image shape {depth_image_shape} is incompatible with depth dimension {self.depth_dim}."
            )
        if len(depth_backbone_channels) != len(depth_backbone_kernels):
            raise ValueError("depth_backbone_channels and depth_backbone_kernels must have the same length.")

        def _act() -> nn.Module:
            return copy.deepcopy(resolve_nn_activation(activation))

        tr_rnn_hidden_dim = int(rnn_hidden_dim if tr_rnn_hidden_dim is None else tr_rnn_hidden_dim)
        tr_rnn_num_layers = int(rnn_num_layers if tr_rnn_num_layers is None else tr_rnn_num_layers)
        ie_rnn_hidden_dim = int(rnn_hidden_dim if ie_rnn_hidden_dim is None else ie_rnn_hidden_dim)
        ie_rnn_num_layers = int(rnn_num_layers if ie_rnn_num_layers is None else ie_rnn_num_layers)
        critic_rnn_type = rnn_type if critic_rnn_type is None else critic_rnn_type
        critic_rnn_hidden_dim = int(rnn_hidden_dim if critic_rnn_hidden_dim is None else critic_rnn_hidden_dim)
        critic_rnn_num_layers = int(rnn_num_layers if critic_rnn_num_layers is None else critic_rnn_num_layers)
        self._actor_rnn_type = str(rnn_type).lower()
        self._critic_rnn_type = str(critic_rnn_type).lower()

        if self._actor_rnn_type != "gru":
            raise ValueError(
                f"StartActorCritic currently supports GRU actor memories only, got rnn_type='{rnn_type}'."
            )

        # Depth encoder for TR-Net.
        conv_layers: list[nn.Module] = []
        in_channels = 1
        cur_h, cur_w = self.depth_h, self.depth_w
        for layer_idx, (out_channels, kernel) in enumerate(zip(depth_backbone_channels, depth_backbone_kernels)):
            k = int(kernel)
            conv_layers.append(nn.Conv2d(in_channels=in_channels, out_channels=int(out_channels), kernel_size=k))
            cur_h = cur_h - k + 1
            cur_w = cur_w - k + 1
            if cur_h <= 0 or cur_w <= 0:
                raise ValueError("Depth backbone convolution stack produced non-positive spatial size.")
            if layer_idx == 0 and int(depth_backbone_pool_kernel) > 1:
                pool_k = int(depth_backbone_pool_kernel)
                conv_layers.append(nn.MaxPool2d(kernel_size=pool_k, stride=pool_k))
                cur_h = cur_h // pool_k
                cur_w = cur_w // pool_k
                if cur_h <= 0 or cur_w <= 0:
                    raise ValueError("Depth backbone pooling produced non-positive spatial size.")
            conv_layers.append(_act())
            in_channels = int(out_channels)

        self.depth_encoder = nn.Sequential(*conv_layers)
        self.depth_flatten = nn.Flatten()
        self.depth_projector = nn.Sequential(
            nn.Linear(in_channels * cur_h * cur_w, int(depth_backbone_image_fc_dim)),
            _act(),
            nn.Linear(int(depth_backbone_image_fc_dim), int(depth_backbone_latent_dim)),
            _act(),
        )

        # Proprio branch.
        self.proprio_dim = sum(end - start for start, end in self.proprio_slices)
        self.tr_proprio_encoder = nn.Sequential(
            nn.Linear(self.proprio_dim, int(tr_proprio_hidden_dim)),
            _act(),
        )

        # TR-Net memory and decoders.
        self.memory_tr = Memory(
            input_size=int(tr_proprio_hidden_dim) + int(depth_backbone_latent_dim),
            type=rnn_type,
            num_layers=int(tr_rnn_num_layers),
            hidden_size=int(tr_rnn_hidden_dim),
        )
        self.tr_rough_decoder = MLP(
            input_dim=int(tr_rnn_hidden_dim),
            output_dim=self.terrain_map_dim,
            hidden_dims=[256, 256],
            activation=activation,
        )
        self.tr_refine_decoder = _StartRefineUNet(
            map_h=self.map_h,
            map_w=self.map_w,
            base_channels=int(tr_refine_base_channels),
            activation=activation,
        )

        # I-E estimator.
        self.ie_prop_encoder = nn.Sequential(
            nn.Linear(self.proprio_dim, int(ie_prop_hidden_dim)),
            _act(),
        )
        self.memory_ie = Memory(
            input_size=int(ie_prop_hidden_dim),
            type=rnn_type,
            num_layers=int(ie_rnn_num_layers),
            hidden_size=int(ie_rnn_hidden_dim),
        )
        self.ie_map_encoder = MLP(
            input_dim=self.terrain_map_dim,
            output_dim=int(ie_map_latent_dim),
            hidden_dims=[256],
            activation=activation,
        )

        d_model = int(ie_transformer_dim)
        self.tr_token_proj = nn.Linear(int(tr_rnn_hidden_dim), d_model)
        self.ie_token_proj = nn.Linear(int(ie_rnn_hidden_dim), d_model)
        self.map_token_proj = nn.Linear(int(ie_map_latent_dim), d_model)
        self.ie_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=d_model,
                nhead=int(ie_transformer_heads),
                dim_feedforward=int(ie_transformer_ff_dim),
                dropout=0.0,
                activation="gelu",
                batch_first=True,
            ),
            num_layers=int(ie_transformer_layers),
        )

        self.explicit_latent_head = nn.Sequential(
            nn.Linear(d_model, int(explicit_latent_dim)),
            _act(),
        )
        self.implicit_mu_head = nn.Linear(d_model, int(implicit_latent_dim))
        self.implicit_logvar_head = nn.Linear(d_model, int(implicit_latent_dim))

        self.base_velocity_decoder = (
            nn.Linear(int(explicit_latent_dim), self.base_velocity_dim) if self.base_velocity_dim > 0 else None
        )
        self.body_heightmap_decoder = nn.Linear(int(explicit_latent_dim), self.terrain_map_dim)
        self.feet_heightmap_decoder = (
            nn.Linear(int(explicit_latent_dim), self.feet_map_dim) if self.feet_map_dim > 0 else None
        )
        self.proprio_reconstruction_head = nn.Linear(
            int(explicit_latent_dim) + int(implicit_latent_dim),
            self.proprio_dim,
        )

        self.actor = MLP(
            input_dim=self.proprio_dim + int(explicit_latent_dim) + int(implicit_latent_dim),
            output_dim=int(num_actions),
            hidden_dims=list(actor_hidden_dims),
            activation=activation,
        )

        # Critic branch.
        self.memory_c = Memory(
            input_size=self.total_critic_obs_dim,
            type=critic_rnn_type,
            num_layers=int(critic_rnn_num_layers),
            hidden_size=int(critic_rnn_hidden_dim),
        )
        self.critic = MLP(
            input_dim=int(critic_rnn_hidden_dim),
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

        # Action noise.
        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

        # Loss weights.
        self.tr_rough_loss_weight = float(tr_rough_loss_weight)
        self.tr_refined_loss_weight = float(tr_refined_loss_weight)
        self.ie_kld_loss_weight = float(ie_kld_loss_weight)
        self.ie_base_vel_loss_weight = float(ie_base_vel_loss_weight)
        self.ie_body_map_loss_weight = float(ie_body_map_loss_weight)
        self.ie_feet_map_loss_weight = float(ie_feet_map_loss_weight)
        self.ie_proprio_loss_weight = float(ie_proprio_loss_weight)

        # AdaSmpl.
        self.adasmpl_enabled = bool(adasmpl_enabled)
        self.adasmpl_min_probability = float(min(adasmpl_min_probability, adasmpl_max_probability))
        self.adasmpl_max_probability = float(max(adasmpl_min_probability, adasmpl_max_probability))
        self.adasmpl_probability = float(adasmpl_initial_probability)
        self.set_adasmpl_probability(self.adasmpl_probability)

        # Cache from latest actor forward pass (used by StartPPO.update).
        self._aux_cache: dict[str, torch.Tensor | None] = {}

    @staticmethod
    def _resolve_map_shape(map_dim: int, terrain_map_shape: tuple[int, int] | None) -> tuple[int, int]:
        if terrain_map_shape is not None:
            h, w = int(terrain_map_shape[0]), int(terrain_map_shape[1])
            if h * w != map_dim:
                raise ValueError(f"terrain_map_shape {terrain_map_shape} is incompatible with map dim {map_dim}.")
            return h, w

        # Fallback: choose the factorization closest to aspect ratio 2:1 (x:y in the paper).
        best_h, best_w = map_dim, 1
        best_score = float("inf")
        for w in range(1, int(math.sqrt(map_dim)) + 1):
            if map_dim % w != 0:
                continue
            h = map_dim // w
            score = abs((h / max(w, 1)) - 2.0)
            if score < best_score:
                best_h, best_w = h, w
                best_score = score
        return best_h, best_w

    def _get_group(self, obs, group_name: str | None) -> torch.Tensor | None:
        if group_name is None:
            return None
        if group_name in obs:
            return obs[group_name]
        return None

    def _split_actor_obs(self, actor_obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        proprio_parts = [actor_obs[..., start:end] for start, end in self.proprio_slices]
        proprio = torch.cat(proprio_parts, dim=-1) if proprio_parts else actor_obs.new_zeros((*actor_obs.shape[:-1], 0))
        depth = actor_obs[..., self.depth_slice[0] : self.depth_slice[1]]
        return proprio, depth

    def _build_next_proprio_target(
        self, proprio_seq: torch.Tensor, masks: torch.Tensor | None
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        """Build next-step proprio targets aligned with recurrent PPO minibatches."""
        if masks is None or proprio_seq.ndim < 3:
            return None, None

        next_proprio = torch.zeros_like(proprio_seq)
        next_valid = torch.zeros((*proprio_seq.shape[:-1], 1), device=proprio_seq.device, dtype=proprio_seq.dtype)

        if proprio_seq.shape[0] > 1:
            next_proprio[:-1] = proprio_seq[1:]
            next_valid[:-1, :, 0] = (masks[:-1] & masks[1:]).to(proprio_seq.dtype)

        next_proprio = unpad_trajectories(next_proprio, masks)
        next_valid = unpad_trajectories(next_valid, masks).squeeze(-1) > 0.5
        return next_proprio, next_valid

    def _encode_depth(self, depth_flat: torch.Tensor) -> torch.Tensor:
        flat = depth_flat.reshape(-1, self.depth_dim)
        depth_image = flat.view(-1, 1, self.depth_h, self.depth_w)
        feat = self.depth_encoder(depth_image)
        feat = self.depth_flatten(feat)
        feat = self.depth_projector(feat)
        return feat.view(*depth_flat.shape[:-1], feat.shape[-1])

    @staticmethod
    def _parse_actor_hidden(hidden_states) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if hidden_states is None:
            return None, None
        if isinstance(hidden_states, (list, tuple)):
            tr_hidden = hidden_states[0] if len(hidden_states) > 0 else None
            ie_hidden = hidden_states[1] if len(hidden_states) > 1 else None
            return tr_hidden, ie_hidden
        return hidden_states, None

    def _parse_critic_hidden(self, hidden_states):
        if hidden_states is None:
            return None
        if self._critic_rnn_type == "lstm":
            if isinstance(hidden_states, (list, tuple)) and len(hidden_states) >= 2:
                return hidden_states[0], hidden_states[1]
            return hidden_states
        # GRU path: storage may pass duplicated tuple/list to match actor hidden-state arity.
        if isinstance(hidden_states, (list, tuple)):
            return hidden_states[0] if len(hidden_states) > 0 else None
        return hidden_states

    @staticmethod
    def _sample_implicit(
        mu: torch.Tensor,
        logvar: torch.Tensor,
        training: bool,
    ) -> torch.Tensor:
        if not training:
            return mu
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def _update_distribution_from_mean(self, mean: torch.Tensor) -> None:
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def _forward_actor(
        self,
        obs,
        masks=None,
        hidden_states=None,
        use_adasmpl: bool = False,
        sample_implicit: bool | None = None,
    ) -> torch.Tensor:
        actor_obs = self.get_actor_obs(obs)
        actor_obs = self.actor_obs_normalizer(actor_obs)

        proprio_seq, depth = self._split_actor_obs(actor_obs)
        depth_feat = self._encode_depth(depth)
        tr_prop_feat = self.tr_proprio_encoder(proprio_seq)

        tr_input = torch.cat((tr_prop_feat, depth_feat), dim=-1)
        tr_hidden_state, ie_hidden_state = self._parse_actor_hidden(hidden_states)
        tr_state = self.memory_tr(tr_input, masks, tr_hidden_state).squeeze(0)

        rough_heightmap = self.tr_rough_decoder(tr_state)
        refined_heightmap = self.tr_refine_decoder(rough_heightmap)

        ie_prop_feat = self.ie_prop_encoder(proprio_seq)
        ie_state = self.memory_ie(ie_prop_feat, masks, ie_hidden_state).squeeze(0)
        action_batch_shape = tr_state.shape[:-1]

        terrain_heightmap_gt = self._get_group(obs, self.terrain_heightmap_obs_group)
        feet_heightmap_gt = self._get_group(obs, self.feet_heightmap_obs_group)
        base_velocity_gt = self._get_group(obs, self.base_velocity_obs_group)
        adasmpl_selector = self._get_group(obs, _START_ADASMPL_SELECTOR)
        next_proprio_target, next_proprio_valid = self._build_next_proprio_target(proprio_seq, masks)

        # For recurrent updates, align supervision/conditioning tensors with unpadded memory outputs.
        proprio = proprio_seq
        if masks is not None:
            proprio = unpad_trajectories(proprio, masks)
            if terrain_heightmap_gt is not None and terrain_heightmap_gt.ndim >= 3:
                terrain_heightmap_gt = unpad_trajectories(terrain_heightmap_gt, masks)
            if feet_heightmap_gt is not None and feet_heightmap_gt.ndim >= 3:
                feet_heightmap_gt = unpad_trajectories(feet_heightmap_gt, masks)
            if base_velocity_gt is not None and base_velocity_gt.ndim >= 3:
                base_velocity_gt = unpad_trajectories(base_velocity_gt, masks)
            if adasmpl_selector is not None and adasmpl_selector.ndim >= 3:
                adasmpl_selector = unpad_trajectories(adasmpl_selector, masks)

        # Collapse potential [time, batch, dim] into [N, dim] consistently.
        proprio = proprio.reshape(-1, proprio.shape[-1])
        tr_state = tr_state.reshape(-1, tr_state.shape[-1])
        ie_state = ie_state.reshape(-1, ie_state.shape[-1])
        refined_heightmap = refined_heightmap.reshape(-1, refined_heightmap.shape[-1])
        rough_heightmap = rough_heightmap.reshape(-1, rough_heightmap.shape[-1])
        if terrain_heightmap_gt is not None:
            terrain_heightmap_gt = terrain_heightmap_gt.reshape(-1, terrain_heightmap_gt.shape[-1])
        if feet_heightmap_gt is not None:
            feet_heightmap_gt = feet_heightmap_gt.reshape(-1, feet_heightmap_gt.shape[-1])
        if base_velocity_gt is not None:
            base_velocity_gt = base_velocity_gt.reshape(-1, base_velocity_gt.shape[-1])
        if adasmpl_selector is not None:
            adasmpl_selector = adasmpl_selector.reshape(-1, adasmpl_selector.shape[-1])
        if next_proprio_target is not None:
            next_proprio_target = next_proprio_target.reshape(-1, next_proprio_target.shape[-1])
        if next_proprio_valid is not None:
            next_proprio_valid = next_proprio_valid.reshape(-1)

        # Recurrent minibatches can have branch-specific packing quirks; trim all branches to a shared prefix.
        num_samples = min(
            proprio.shape[0],
            tr_state.shape[0],
            ie_state.shape[0],
            rough_heightmap.shape[0],
            refined_heightmap.shape[0],
        )
        if terrain_heightmap_gt is not None:
            num_samples = min(num_samples, terrain_heightmap_gt.shape[0])
        if feet_heightmap_gt is not None:
            num_samples = min(num_samples, feet_heightmap_gt.shape[0])
        if base_velocity_gt is not None:
            num_samples = min(num_samples, base_velocity_gt.shape[0])
        if adasmpl_selector is not None:
            num_samples = min(num_samples, adasmpl_selector.shape[0])
        if next_proprio_target is not None:
            num_samples = min(num_samples, next_proprio_target.shape[0])
        if next_proprio_valid is not None:
            num_samples = min(num_samples, next_proprio_valid.shape[0])
        if num_samples <= 0:
            raise RuntimeError("START actor forward produced no valid samples after recurrent alignment.")

        proprio = proprio[:num_samples]
        tr_state = tr_state[:num_samples]
        ie_state = ie_state[:num_samples]
        rough_heightmap = rough_heightmap[:num_samples]
        refined_heightmap = refined_heightmap[:num_samples]
        if terrain_heightmap_gt is not None:
            terrain_heightmap_gt = terrain_heightmap_gt[:num_samples]
        if feet_heightmap_gt is not None:
            feet_heightmap_gt = feet_heightmap_gt[:num_samples]
        if base_velocity_gt is not None:
            base_velocity_gt = base_velocity_gt[:num_samples]
        if adasmpl_selector is not None:
            adasmpl_selector = adasmpl_selector[:num_samples]
        if next_proprio_target is not None:
            next_proprio_target = next_proprio_target[:num_samples]
        if next_proprio_valid is not None:
            next_proprio_valid = next_proprio_valid[:num_samples]

        map_for_actor = refined_heightmap
        if use_adasmpl and terrain_heightmap_gt is not None:
            if adasmpl_selector is None:
                raise RuntimeError(
                    "START training requires a rollout-stored AdaSmpl selector; use StartPPO.act to collect samples."
                )
            adasmpl_selector = adasmpl_selector > 0.5
            map_for_actor = torch.where(adasmpl_selector, terrain_heightmap_gt, refined_heightmap)

        map_latent = self.ie_map_encoder(map_for_actor)
        tr_token = self.tr_token_proj(tr_state)
        ie_token = self.ie_token_proj(ie_state)
        map_token = self.map_token_proj(map_latent.reshape(-1, map_latent.shape[-1]))
        tokens = torch.stack([tr_token, ie_token, map_token], dim=1)
        fused = self.ie_transformer(tokens).mean(dim=1)

        explicit_latent = self.explicit_latent_head(fused)
        implicit_mu = self.implicit_mu_head(fused)
        implicit_logvar = torch.clamp(self.implicit_logvar_head(fused), -8.0, 8.0)
        if sample_implicit is None:
            sample_implicit = self.training
        implicit_z = self._sample_implicit(
            implicit_mu,
            implicit_logvar,
            training=sample_implicit,
        )

        # PPO needs a deterministic action mean for a fixed observation.  The
        # sampled VAE latent remains available to the supervised reconstruction
        # branch, while the behavior policy consumes its posterior mean.
        actor_input = torch.cat((proprio, explicit_latent, implicit_mu), dim=-1)
        action_mean = self.actor(actor_input)
        if len(action_batch_shape) > 0:
            expected_samples = math.prod(action_batch_shape)
            if expected_samples == action_mean.shape[0]:
                action_mean = action_mean.view(*action_batch_shape, action_mean.shape[-1])

        base_vel_pred = self.base_velocity_decoder(explicit_latent) if self.base_velocity_decoder is not None else None
        body_heightmap_pred = self.body_heightmap_decoder(explicit_latent)
        feet_heightmap_pred = self.feet_heightmap_decoder(explicit_latent) if self.feet_heightmap_decoder else None
        proprio_recon = self.proprio_reconstruction_head(torch.cat((explicit_latent, implicit_z), dim=-1))

        self._aux_cache = {
            "terrain_gt": terrain_heightmap_gt,
            "feet_gt": feet_heightmap_gt,
            "base_vel_gt": base_velocity_gt,
            "next_proprio_target": next_proprio_target,
            "next_proprio_valid": next_proprio_valid,
            "rough_map": rough_heightmap,
            "refined_map": refined_heightmap,
            "base_vel_pred": base_vel_pred,
            "body_map_pred": body_heightmap_pred,
            "feet_map_pred": feet_heightmap_pred,
            "proprio_recon": proprio_recon,
            "implicit_mu": implicit_mu,
            "implicit_logvar": implicit_logvar,
        }
        return action_mean

    def compute_auxiliary_losses(self) -> dict[str, torch.Tensor]:
        device = next(self.parameters()).device
        zero = torch.zeros((), device=device)
        if not self._aux_cache:
            return {
                "terrain_reconstruction": zero,
                "terrain_reconstruction_rough": zero,
                "terrain_reconstruction_refined": zero,
                "ie_total": zero,
                "ie_kld": zero,
                "ie_base_vel": zero,
                "ie_body_map": zero,
                "ie_feet_map": zero,
                "ie_proprio": zero,
                "aux_total": zero,
            }

        terrain_gt = self._aux_cache["terrain_gt"]
        rough_map = self._aux_cache["rough_map"]
        refined_map = self._aux_cache["refined_map"]

        if terrain_gt is None:
            tr_rough_loss = zero
            tr_refined_loss = zero
        else:
            tr_rough_loss = F.mse_loss(rough_map, terrain_gt)
            tr_refined_loss = F.l1_loss(refined_map, terrain_gt)

        tr_total = self.tr_rough_loss_weight * tr_rough_loss + self.tr_refined_loss_weight * tr_refined_loss

        implicit_mu = self._aux_cache["implicit_mu"]
        implicit_logvar = self._aux_cache["implicit_logvar"]
        ie_kld = -0.5 * torch.mean(torch.sum(1.0 + implicit_logvar - implicit_mu.pow(2) - implicit_logvar.exp(), dim=-1))

        base_vel_pred = self._aux_cache["base_vel_pred"]
        base_vel_gt = self._aux_cache["base_vel_gt"]
        if base_vel_pred is not None and base_vel_gt is not None:
            ie_base_vel = F.mse_loss(base_vel_pred, base_vel_gt)
        else:
            ie_base_vel = zero

        body_map_pred = self._aux_cache["body_map_pred"]
        if terrain_gt is not None:
            ie_body_map = F.mse_loss(body_map_pred, terrain_gt)
        else:
            ie_body_map = zero

        feet_map_pred = self._aux_cache["feet_map_pred"]
        feet_gt = self._aux_cache["feet_gt"]
        if feet_map_pred is not None and feet_gt is not None:
            ie_feet_map = F.mse_loss(feet_map_pred, feet_gt)
        else:
            ie_feet_map = zero

        proprio_recon = self._aux_cache["proprio_recon"]
        next_proprio_target = self._aux_cache["next_proprio_target"]
        next_proprio_valid = self._aux_cache["next_proprio_valid"]
        if next_proprio_target is not None and next_proprio_valid is not None and torch.any(next_proprio_valid):
            ie_proprio = F.mse_loss(proprio_recon[next_proprio_valid], next_proprio_target[next_proprio_valid])
        else:
            ie_proprio = zero

        ie_total = (
            self.ie_kld_loss_weight * ie_kld
            + self.ie_base_vel_loss_weight * ie_base_vel
            + self.ie_body_map_loss_weight * ie_body_map
            + self.ie_feet_map_loss_weight * ie_feet_map
            + self.ie_proprio_loss_weight * ie_proprio
        )

        return {
            "terrain_reconstruction": tr_total,
            "terrain_reconstruction_rough": tr_rough_loss,
            "terrain_reconstruction_refined": tr_refined_loss,
            "ie_total": ie_total,
            "ie_kld": ie_kld,
            "ie_base_vel": ie_base_vel,
            "ie_body_map": ie_body_map,
            "ie_feet_map": ie_feet_map,
            "ie_proprio": ie_proprio,
            "aux_total": tr_total + ie_total,
        }

    def set_adasmpl_probability(self, probability: float) -> None:
        probability = float(probability)
        probability = max(self.adasmpl_min_probability, min(self.adasmpl_max_probability, probability))
        self.adasmpl_probability = probability

    def get_adasmpl_probability(self) -> float:
        return float(self.adasmpl_probability)

    def reset(self, dones=None, hidden_states=None):
        actor_hidden = None
        critic_hidden = None
        if isinstance(hidden_states, tuple):
            actor_hidden = hidden_states[0]
            critic_hidden = hidden_states[1] if len(hidden_states) > 1 else None
        elif hidden_states is not None:
            actor_hidden = hidden_states

        tr_hidden, ie_hidden = self._parse_actor_hidden(actor_hidden)
        critic_hidden = self._parse_critic_hidden(critic_hidden)
        self.memory_tr.reset(dones=dones, hidden_states=tr_hidden)
        self.memory_ie.reset(dones=dones, hidden_states=ie_hidden)
        self.memory_c.reset(dones=dones, hidden_states=critic_hidden)

    def forward(self, obs):
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

    def get_actor_obs(self, obs):
        return torch.cat([obs[group_name] for group_name in self.obs_groups["policy"]], dim=-1)

    def get_critic_obs(self, obs):
        return torch.cat([obs[group_name] for group_name in self.obs_groups["critic"]], dim=-1)

    def act(self, obs, masks=None, hidden_states=None):
        use_adasmpl = self.training and self.adasmpl_enabled
        mean = self._forward_actor(obs, masks=masks, hidden_states=hidden_states, use_adasmpl=use_adasmpl)
        self._update_distribution_from_mean(mean)
        return self.distribution.sample()

    def act_inference(self, obs):
        with torch.no_grad():
            mean = self._forward_actor(
                obs,
                masks=None,
                hidden_states=None,
                use_adasmpl=False,
                sample_implicit=False,
            )
        return mean

    def evaluate(self, obs, masks=None, hidden_states=None):
        critic_obs = self.get_critic_obs(obs)
        critic_obs = self.critic_obs_normalizer(critic_obs)
        hidden_states = self._parse_critic_hidden(hidden_states)
        critic_state = self.memory_c(critic_obs, masks, hidden_states).squeeze(0)
        return self.critic(critic_state)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def get_hidden_states(self):
        tr_hidden = self.memory_tr.hidden_states
        ie_hidden = self.memory_ie.hidden_states
        critic_hidden = self.memory_c.hidden_states
        if tr_hidden is None and ie_hidden is None and critic_hidden is None:
            return (None, None)
        # RolloutStorage expects actor and critic hidden-state tuples with the same length.
        # START has two actor memories (TR, I-E) but one critic memory; duplicate critic GRU hidden.
        if self._critic_rnn_type == "gru":
            critic_hidden_for_storage = (critic_hidden, critic_hidden)
        else:
            critic_hidden_for_storage = critic_hidden
        return (tr_hidden, ie_hidden), critic_hidden_for_storage

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True


class StartPPO(PPO):
    """START PPO variant with TR-Net/I-E auxiliary losses and AdaSmpl updates."""

    policy: StartActorCritic

    def __init__(
        self,
        policy,
        storage=None,
        tr_loss_coef: float = 1.0,
        ie_loss_coef: float = 1.0,
        adasmpl_reward_window: int = 256,
        adasmpl_min_episodes: int = 8,
        adasmpl_cv_scale: float = 1.0,
        adasmpl_probability_ema_alpha: float = 1.0,
        adasmpl_probability_min: float = 0.0,
        adasmpl_probability_max: float = 1.0,
        multi_gpu_cfg: dict | None = None,
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        **kwargs,
    ):
        if storage is None:
            raise ValueError("StartPPO requires an initialized RolloutStorage instance.")
        if rnd_cfg:
            raise ValueError("StartPPO does not support RND.")
        if symmetry_cfg and (symmetry_cfg.get("use_data_augmentation") or symmetry_cfg.get("use_mirror_loss")):
            raise ValueError("StartPPO does not support symmetry augmentation or mirror losses.")

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
        from rsl_rl.storage import RolloutStorage

        self.transition = RolloutStorage.Transition()
        self._add_policy_context_to_storage(self.storage)
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
            raise TypeError(f"Unexpected StartPPO arguments: {sorted(kwargs)}")
        self.optimizer = resolve_optimizer(optimizer_name)(self.policy.parameters(), lr=self.learning_rate)
        self.tr_loss_coef = float(tr_loss_coef)
        self.ie_loss_coef = float(ie_loss_coef)

        self._episode_return_buffer = deque(maxlen=max(8, int(adasmpl_reward_window)))
        self._episode_returns: torch.Tensor | None = None
        self._adasmpl_min_episodes = max(1, int(adasmpl_min_episodes))
        self._adasmpl_cv_scale = float(adasmpl_cv_scale)
        self._adasmpl_prob_ema_alpha = float(max(0.0, min(1.0, adasmpl_probability_ema_alpha)))
        self._adasmpl_prob_min = float(min(adasmpl_probability_min, adasmpl_probability_max))
        self._adasmpl_prob_max = float(max(adasmpl_probability_min, adasmpl_probability_max))
        self._adasmpl_prob_ema = self._adasmpl_prob_min

        if hasattr(self.policy, "get_adasmpl_probability"):
            self._adasmpl_prob_ema = float(self.policy.get_adasmpl_probability())

        self._episode_returns = torch.zeros(storage.num_envs, device=self.device)

    def _add_policy_context_to_storage(self, storage) -> None:
        """Allocate rollout-only randomness so PPO reuses the behavior-policy context."""
        reference = next(iter(storage.observations.values()))
        if _START_ADASMPL_SELECTOR not in storage.observations:
            storage.observations.set(
                _START_ADASMPL_SELECTOR,
                reference.new_zeros((*reference.shape[:-1], 1)),
            )

    def _add_policy_context_to_observations(self, obs):
        """Sample one whole-map source selector per environment transition."""
        rollout_obs = obs.clone(recurse=False) if hasattr(obs, "clone") else dict(obs)
        reference = next(iter(obs.values()))
        batch_shape = reference.shape[:-1]
        probability = self.policy.get_adasmpl_probability() if self.policy.adasmpl_enabled else 0.0
        selector = torch.rand((*batch_shape, 1), device=reference.device) < probability
        rollout_obs[_START_ADASMPL_SELECTOR] = selector.to(dtype=reference.dtype)
        return rollout_obs

    @staticmethod
    def construct_algorithm(obs, env, cfg: dict, device: str):
        """Construct START around its shared recurrent actor-critic model."""
        from rsl_rl.storage import RolloutStorage
        from rsl_rl.utils import resolve_callable, resolve_obs_groups

        alg_class = StartPPO
        policy_class = StartActorCritic
        algorithm_class_name = cfg["algorithm"].pop("class_name", None)
        if algorithm_class_name and (":" in algorithm_class_name or "." in algorithm_class_name):
            alg_class = resolve_callable(algorithm_class_name)
        policy_cfg = dict(cfg["policy"])
        policy_class_name = policy_cfg.pop("class_name", None)
        if policy_class_name and (":" in policy_class_name or "." in policy_class_name):
            policy_class = resolve_callable(policy_class_name)

        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["policy", "critic"])
        policy = policy_class(obs, cfg["obs_groups"], env.num_actions, **policy_cfg).to(device)
        print(f"START Actor-Critic Model: {policy}")
        storage = RolloutStorage("rl", env.num_envs, cfg["num_steps_per_env"], obs, [env.num_actions], device)
        return alg_class(
            policy,
            storage=storage,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        if training_type != "rl":
            raise ValueError("StartPPO supports RL training type only.")
        from rsl_rl.storage import RolloutStorage

        self.storage = RolloutStorage("rl", num_envs, num_transitions_per_env, obs, actions_shape, self.device)
        self._add_policy_context_to_storage(self.storage)
        self.transition = RolloutStorage.Transition()
        self._episode_returns = torch.zeros(num_envs, device=self.device)

    def act(self, obs):
        rollout_obs = self._add_policy_context_to_observations(obs)
        self.transition.hidden_states = self.policy.get_hidden_states()
        actions = self.policy.act(rollout_obs).detach()
        self.transition.actions = actions
        self.transition.values = self.policy.evaluate(rollout_obs).detach()
        self.transition.actions_log_prob = self.policy.get_actions_log_prob(actions).detach()
        self.transition.distribution_params = (
            self.policy.action_mean.detach(),
            self.policy.action_std.detach(),
        )
        self.transition.observations = rollout_obs
        return actions

    def _update_adasmpl_probability(self):
        if not hasattr(self.policy, "set_adasmpl_probability"):
            return
        if len(self._episode_return_buffer) < self._adasmpl_min_episodes:
            return

        returns = torch.tensor(list(self._episode_return_buffer), device=self.device, dtype=torch.float32)
        mean_abs = torch.abs(torch.mean(returns)).clamp(min=1.0e-6)
        std = torch.std(returns, unbiased=False)
        cv = (std / mean_abs).item()

        target_prob = math.tanh(self._adasmpl_cv_scale * cv)
        target_prob = max(self._adasmpl_prob_min, min(self._adasmpl_prob_max, target_prob))
        self._adasmpl_prob_ema = (
            self._adasmpl_prob_ema_alpha * target_prob + (1.0 - self._adasmpl_prob_ema_alpha) * self._adasmpl_prob_ema
        )
        self.policy.set_adasmpl_probability(self._adasmpl_prob_ema)

    def process_env_step(self, obs, rewards, dones, extras):
        if self._episode_returns is not None:
            rewards_flat = rewards.view(-1)
            done_mask = dones.view(-1) > 0
            self._episode_returns += rewards_flat
            if torch.any(done_mask):
                finished_returns = self._episode_returns[done_mask].detach().cpu().tolist()
                self._episode_return_buffer.extend(finished_returns)
                self._episode_returns[done_mask] = 0.0
                self._update_adasmpl_probability()
        self.policy.update_normalization(obs)
        self.transition.rewards = rewards.clone()
        self.transition.dones = dones
        if "time_outs" in extras:
            self.transition.rewards += self.gamma * torch.squeeze(
                self.transition.values * extras["time_outs"].reshape(-1, 1).to(self.device),
                1,
            )
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs):
        st = self.storage
        hidden_states = self.policy.get_hidden_states()
        last_values = self.policy.evaluate(obs).detach()
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
            st.advantages = (st.advantages - st.advantages.mean()) / (st.advantages.std() + 1.0e-8)

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

    def update(self):  # noqa: C901
        if self.rnd:
            raise NotImplementedError("StartPPO does not support RND.")
        if self.symmetry and (self.symmetry["use_data_augmentation"] or self.symmetry["use_mirror_loss"]):
            raise NotImplementedError("StartPPO does not support symmetry augmentation/mirror losses.")

        mean_value_loss = 0.0
        mean_surrogate_loss = 0.0
        mean_entropy = 0.0

        mean_tr_loss = 0.0
        mean_tr_rough_loss = 0.0
        mean_tr_refined_loss = 0.0
        mean_ie_loss = 0.0
        mean_ie_kld = 0.0
        mean_ie_base_vel = 0.0
        mean_ie_body_map = 0.0
        mean_ie_feet_map = 0.0
        mean_ie_proprio = 0.0

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
            hid_states_batch = hid_states_batch or (None, None)
            if self.normalize_advantage_per_mini_batch:
                with torch.no_grad():
                    advantages_batch = (advantages_batch - advantages_batch.mean()) / (advantages_batch.std() + 1e-8)

            self.policy.act(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[0])
            actions_log_prob_batch = self.policy.get_actions_log_prob(actions_batch)
            value_batch = self.policy.evaluate(obs_batch, masks=masks_batch, hidden_states=hid_states_batch[1])
            mu_batch = self.policy.action_mean
            sigma_batch = self.policy.action_std
            entropy_batch = self.policy.entropy

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

            aux_losses = self.policy.compute_auxiliary_losses()
            tr_loss = aux_losses["terrain_reconstruction"]
            ie_loss = aux_losses["ie_total"]

            loss = (
                surrogate_loss
                + self.value_loss_coef * value_loss
                - self.entropy_coef * entropy_batch.mean()
                + self.tr_loss_coef * tr_loss
                + self.ie_loss_coef * ie_loss
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

            mean_tr_loss += float(tr_loss.detach().item())
            mean_tr_rough_loss += float(aux_losses["terrain_reconstruction_rough"].detach().item())
            mean_tr_refined_loss += float(aux_losses["terrain_reconstruction_refined"].detach().item())
            mean_ie_loss += float(ie_loss.detach().item())
            mean_ie_kld += float(aux_losses["ie_kld"].detach().item())
            mean_ie_base_vel += float(aux_losses["ie_base_vel"].detach().item())
            mean_ie_body_map += float(aux_losses["ie_body_map"].detach().item())
            mean_ie_feet_map += float(aux_losses["ie_feet_map"].detach().item())
            mean_ie_proprio += float(aux_losses["ie_proprio"].detach().item())

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates
        mean_entropy /= num_updates
        mean_tr_loss /= num_updates
        mean_tr_rough_loss /= num_updates
        mean_tr_refined_loss /= num_updates
        mean_ie_loss /= num_updates
        mean_ie_kld /= num_updates
        mean_ie_base_vel /= num_updates
        mean_ie_body_map /= num_updates
        mean_ie_feet_map /= num_updates
        mean_ie_proprio /= num_updates

        self.storage.clear()

        loss_dict = {
            "value_function": mean_value_loss,
            "surrogate": mean_surrogate_loss,
            "entropy": mean_entropy,
            "terrain_reconstruction": mean_tr_loss,
            "terrain_reconstruction_rough": mean_tr_rough_loss,
            "terrain_reconstruction_refined": mean_tr_refined_loss,
            "ie_total": mean_ie_loss,
            "ie_kld": mean_ie_kld,
            "ie_base_vel": mean_ie_base_vel,
            "ie_body_map": mean_ie_body_map,
            "ie_feet_map": mean_ie_feet_map,
            "ie_proprio": mean_ie_proprio,
        }
        if hasattr(self.policy, "get_adasmpl_probability"):
            loss_dict["adasmpl_probability"] = float(self.policy.get_adasmpl_probability())
        return loss_dict

    def reduce_parameters(self) -> None:
        """Synchronize the shared recurrent policy once across distributed workers."""
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
