"""REAL-style privileged teacher with proprio-terrain cross-modal attention."""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn as nn
from torch.distributions import Normal

from rsl_rl.utils import resolve_nn_activation

from .rsl_rl_compat import EmpiricalNormalization, MLP


def _resolve_scan_shape(num_scan_values: int, scan_shape: tuple[int, int] | None) -> tuple[int, int]:
    """Infer a 2D terrain scan shape from a flat scan vector."""
    if scan_shape is not None:
        rows, cols = int(scan_shape[0]), int(scan_shape[1])
        if rows > 0 and cols > 0 and rows * cols == num_scan_values:
            return rows, cols
    best_rows, best_cols = num_scan_values, 1
    best_gap = num_scan_values
    for divisor in range(1, int(num_scan_values**0.5) + 1):
        if num_scan_values % divisor != 0:
            continue
        rows = num_scan_values // divisor
        cols = divisor
        gap = abs(rows - cols)
        if gap < best_gap:
            best_rows, best_cols = rows, cols
            best_gap = gap
    return int(best_rows), int(best_cols)


def _build_scan_coordinates(rows: int, cols: int) -> torch.Tensor:
    """Create normalized terrain-grid coordinates in row-major order."""
    x_coords = torch.linspace(-1.0, 1.0, rows)
    y_coords = torch.linspace(-1.0, 1.0, cols)
    grid_x, grid_y = torch.meshgrid(x_coords, y_coords, indexing="ij")
    return torch.stack((grid_x.reshape(-1), grid_y.reshape(-1)), dim=-1)


def _make_activation(activation: str) -> nn.Module:
    """Create a fresh activation module instance."""
    return copy.deepcopy(resolve_nn_activation(activation))


_ATTENTION_TRANSFER_PREFIXES = ("terrain_encoder.",)
_ATTENTION_K_WEIGHT_KEY = "cross_attention.k_in_proj_weight"
_LEGACY_ATTENTION_KV_WEIGHT_KEY = "cross_attention.kv_in_proj_weight"
_ATTENTION_TRANSFER_KEYS = {_ATTENTION_K_WEIGHT_KEY}


def _add_legacy_key_projection(
    extracted: dict[str, torch.Tensor],
    candidate: dict[str, object],
    prefix: str = "",
) -> None:
    """Recover the supervised K slice from older K+V checkpoint payloads."""
    legacy_weight = candidate.get(f"{prefix}{_LEGACY_ATTENTION_KV_WEIGHT_KEY}")
    if legacy_weight is None or _ATTENTION_K_WEIGHT_KEY in extracted:
        return
    if not isinstance(legacy_weight, torch.Tensor) or legacy_weight.ndim != 2:
        raise ValueError("Legacy REAL KV projection weight must be a rank-2 tensor.")
    embed_dim = int(legacy_weight.shape[1])
    if tuple(legacy_weight.shape) != (2 * embed_dim, embed_dim):
        raise ValueError(
            "Legacy REAL KV projection weight must contain stacked K and V rows; "
            f"got {tuple(legacy_weight.shape)}."
        )
    extracted[_ATTENTION_K_WEIGHT_KEY] = legacy_weight[:embed_dim]


def _extract_attention_state_dict(
    checkpoint: str | Path | dict[str, torch.Tensor] | dict[str, object],
) -> tuple[dict[str, torch.Tensor], dict[str, object]]:
    if isinstance(checkpoint, (str, Path)):
        payload = torch.load(Path(checkpoint), map_location="cpu")
    else:
        payload = checkpoint

    if not isinstance(payload, dict):
        raise ValueError(f"Expected checkpoint payload to be a dict, got {type(payload)!r}.")

    metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
    candidate = payload.get("attention_state_dict")
    if not isinstance(candidate, dict):
        candidate = payload.get("model_state_dict")
    if not isinstance(candidate, dict):
        candidate = payload
    if not isinstance(candidate, dict):
        raise ValueError("Checkpoint does not contain a usable attention state_dict payload.")

    direct = {
        key: value
        for key, value in candidate.items()
        if isinstance(key, str)
        and (key.startswith(_ATTENTION_TRANSFER_PREFIXES) or key in _ATTENTION_TRANSFER_KEYS)
    }
    _add_legacy_key_projection(direct, candidate)
    if direct:
        return direct, metadata

    prefixed_roots = (
        "actor.",
        "critic.",
        "module.actor.",
        "module.critic.",
        "actor_critic.actor.",
        "actor_critic.critic.",
        "alg.actor_critic.actor.",
        "alg.actor_critic.critic.",
    )
    for prefix in prefixed_roots:
        extracted: dict[str, torch.Tensor] = {}
        for key, value in candidate.items():
            if not isinstance(key, str) or not key.startswith(prefix):
                continue
            stripped_key = key[len(prefix) :]
            if stripped_key.startswith(_ATTENTION_TRANSFER_PREFIXES) or stripped_key in _ATTENTION_TRANSFER_KEYS:
                extracted[stripped_key] = value
        _add_legacy_key_projection(extracted, candidate, prefix=prefix)
        if extracted:
            return extracted, metadata

    raise ValueError("Unable to find compatible terrain-token/key weights in the provided checkpoint payload.")


class TerrainTokenEncoder(nn.Module):
    """Encode flat terrain scan dots into tokens with fixed 2D coordinates."""

    def __init__(
        self,
        num_scan_values: int,
        embedding_dim: int,
        activation: str,
        scan_shape: tuple[int, int] | None = None,
        hidden_dim: int = 128,
        use_scan_positional_encoding: bool = True,
    ):
        super().__init__()
        self.num_scan_values = int(num_scan_values)
        self.scan_rows, self.scan_cols = _resolve_scan_shape(self.num_scan_values, scan_shape)
        self.use_scan_positional_encoding = bool(use_scan_positional_encoding)
        input_dim = 1 + (2 if self.use_scan_positional_encoding else 0)
        self.register_buffer(
            "scan_coordinates",
            _build_scan_coordinates(self.scan_rows, self.scan_cols),
            persistent=False,
        )
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, int(hidden_dim)),
            _make_activation(activation),
            nn.Linear(int(hidden_dim), int(embedding_dim)),
            _make_activation(activation),
        )

    def forward(self, terrain_scan: torch.Tensor) -> torch.Tensor:
        if terrain_scan.shape[-1] != self.num_scan_values:
            raise ValueError(f"Expected terrain scan dim {self.num_scan_values}, got {terrain_scan.shape[-1]}.")
        tokens = terrain_scan.unsqueeze(-1)
        if self.use_scan_positional_encoding:
            coords = self.scan_coordinates.unsqueeze(0).expand(terrain_scan.shape[0], -1, -1)
            tokens = torch.cat((tokens, coords), dim=-1)
        return self.encoder(tokens)


class PrivilegedEncoder(nn.Module):
    """Compact encoder for teacher-only privileged state."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: list[int], activation: str):
        super().__init__()
        if not hidden_dims:
            hidden_dims = [output_dim]
        self.network = MLP(int(input_dim), int(output_dim), list(hidden_dims), activation)

    def forward(self, privileged_state: torch.Tensor) -> torch.Tensor:
        return self.network(privileged_state)


class CrossModalTeacherTower(nn.Module):
    """Cross-modal tower used by the actor or the critic."""

    def __init__(
        self,
        proprio_dim: int,
        terrain_dim: int,
        privileged_dim: int,
        output_dim: int,
        hidden_dims: list[int],
        activation: str,
        attention_embed_dim: int,
        attention_num_heads: int,
        terrain_encoder_hidden_dim: int,
        privileged_latent_dim: int,
        privileged_hidden_dims: list[int],
        terrain_scan_shape: tuple[int, int] | None,
        use_scan_positional_encoding: bool,
    ):
        super().__init__()
        if attention_embed_dim % attention_num_heads != 0:
            raise ValueError(
                f"attention_embed_dim={attention_embed_dim} must be divisible by attention_num_heads={attention_num_heads}."
            )
        self.proprio_encoder = nn.Sequential(
            nn.Linear(int(proprio_dim), int(attention_embed_dim)),
            _make_activation(activation),
            nn.Linear(int(attention_embed_dim), int(attention_embed_dim)),
            _make_activation(activation),
        )
        self.terrain_encoder = TerrainTokenEncoder(
            num_scan_values=int(terrain_dim),
            embedding_dim=int(attention_embed_dim),
            activation=activation,
            scan_shape=terrain_scan_shape,
            hidden_dim=int(terrain_encoder_hidden_dim),
            use_scan_positional_encoding=use_scan_positional_encoding,
        )
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=int(attention_embed_dim),
            num_heads=int(attention_num_heads),
            batch_first=True,
        )
        self.context_norm = nn.LayerNorm(int(attention_embed_dim))
        self.privileged_encoder = PrivilegedEncoder(
            input_dim=int(privileged_dim),
            output_dim=int(privileged_latent_dim),
            hidden_dims=list(privileged_hidden_dims),
            activation=activation,
        )
        self.head = MLP(
            input_dim=int(attention_embed_dim) * 2 + int(privileged_latent_dim),
            output_dim=int(output_dim),
            hidden_dims=list(hidden_dims),
            activation=activation,
        )

    @property
    def terrain_scan_shape(self) -> tuple[int, int]:
        return self.terrain_encoder.scan_rows, self.terrain_encoder.scan_cols

    def forward(
        self,
        proprio: torch.Tensor,
        terrain_scan: torch.Tensor,
        privileged_state: torch.Tensor,
        need_attention: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        proprio_latent = self.proprio_encoder(proprio)
        query = proprio_latent.unsqueeze(1)
        terrain_tokens = self.terrain_encoder(terrain_scan)
        attended, attention_weights = self.cross_attention(
            query,
            terrain_tokens,
            terrain_tokens,
            need_weights=need_attention,
            average_attn_weights=False,
        )
        terrain_context = self.context_norm((attended + query).squeeze(1))
        privileged_latent = self.privileged_encoder(privileged_state)
        fused = torch.cat((proprio_latent, terrain_context, privileged_latent), dim=-1)
        output = self.head(fused)
        if attention_weights is None:
            return output, None
        return output, attention_weights.squeeze(2)

    def load_pretrained_attention(self, attention_state_dict: dict[str, torch.Tensor]) -> list[str]:
        """Load encoder and key-projection parameters supervised offline."""
        direct_state = {
            key: value
            for key, value in attention_state_dict.items()
            if key.startswith("terrain_encoder.")
        }
        if not direct_state:
            raise ValueError("Pretrained checkpoint did not contain supervised terrain-token encoder weights.")
        incompatible = self.load_state_dict(direct_state, strict=False)
        if incompatible.unexpected_keys:
            raise ValueError(f"Unexpected pretrained terrain-token encoder keys: {incompatible.unexpected_keys}")
        loaded_keys = sorted(direct_state.keys())

        embed_dim = int(self.cross_attention.embed_dim)
        key_weight = attention_state_dict.get(_ATTENTION_K_WEIGHT_KEY)
        if key_weight is not None:
            if tuple(key_weight.shape) != (embed_dim, embed_dim):
                raise ValueError(
                    f"Expected {_ATTENTION_K_WEIGHT_KEY} shape {(embed_dim, embed_dim)}, "
                    f"got {tuple(key_weight.shape)}."
                )
            with torch.no_grad():
                self.cross_attention.in_proj_weight[embed_dim : 2 * embed_dim].copy_(
                    key_weight.to(self.cross_attention.in_proj_weight)
                )
            loaded_keys.append(_ATTENTION_K_WEIGHT_KEY)
        return loaded_keys

    def freeze_pretrained_attention(self) -> None:
        # Offline traversability pretraining directly supervises only the
        # terrain-token encoder.  The runtime proprioceptive query, K/V path,
        # output projection, and normalization remain trainable under PPO.
        for parameter in self.terrain_encoder.parameters():
            parameter.requires_grad = False


@dataclass(frozen=True)
class AttentionDebugResult:
    """Debug bundle for attention visualization."""

    action_mean: torch.Tensor
    attention_weights: torch.Tensor
    terrain_scan: torch.Tensor
    terrain_scan_shape: tuple[int, int]


class RealTeacherActorCritic(nn.Module):
    """Paper-inspired privileged teacher using proprio-terrain associated reasoning."""

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        actor_obs_normalization=False,
        critic_obs_normalization=False,
        actor_hidden_dims=[512, 256, 128],
        critic_hidden_dims=[512, 256, 128],
        activation="elu",
        init_noise_std=1.0,
        noise_std_type: str = "scalar",
        proprio_obs_group: str = "real_teacher_proprio",
        terrain_obs_group: str = "real_teacher_terrain",
        privileged_obs_group: str = "real_teacher_privileged",
        terrain_scan_shape: tuple[int, int] | None = (16, 10),
        attention_embed_dim: int = 128,
        attention_num_heads: int = 4,
        terrain_encoder_hidden_dim: int = 128,
        privileged_latent_dim: int = 128,
        privileged_hidden_dims: list[int] = [128],
        use_scan_positional_encoding: bool = True,
        pretrained_attention_checkpoint: str | None = None,
        freeze_pretrained_attention: bool = False,
        load_pretrained_attention_into_actor: bool = True,
        load_pretrained_attention_into_critic: bool = True,
        **kwargs,
    ):
        if kwargs:
            print(
                "RealTeacherActorCritic.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.obs_groups = obs_groups
        self.proprio_obs_group = proprio_obs_group
        self.terrain_obs_group = terrain_obs_group
        self.privileged_obs_group = privileged_obs_group
        self.pretrained_attention_metadata: dict[str, object] = {}

        for group_name in {proprio_obs_group, terrain_obs_group, privileged_obs_group}:
            if group_name not in obs:
                raise ValueError(f"Observation group '{group_name}' not found in observations: {list(obs.keys())}")
            if len(obs[group_name].shape) != 2:
                raise ValueError(f"Observation group '{group_name}' must be 1D per environment.")

        actor_group_name = "policy" if "policy" in obs_groups else "actor"
        if actor_group_name not in obs_groups or "critic" not in obs_groups:
            raise ValueError("REAL teacher observation groups must define 'policy' (or 'actor') and 'critic'.")
        self._actor_group_names = list(obs_groups[actor_group_name])
        self._critic_group_names = list(obs_groups["critic"])
        self._actor_group_dims = {name: int(obs[name].shape[-1]) for name in self._actor_group_names}
        self._critic_group_dims = {name: int(obs[name].shape[-1]) for name in self._critic_group_names}
        actor_total_obs = sum(self._actor_group_dims.values())
        critic_total_obs = sum(self._critic_group_dims.values())

        proprio_dim = int(obs[proprio_obs_group].shape[-1])
        terrain_dim = int(obs[terrain_obs_group].shape[-1])
        privileged_dim = int(obs[privileged_obs_group].shape[-1])

        self.actor = CrossModalTeacherTower(
            proprio_dim=proprio_dim,
            terrain_dim=terrain_dim,
            privileged_dim=privileged_dim,
            output_dim=int(num_actions),
            hidden_dims=list(actor_hidden_dims),
            activation=activation,
            attention_embed_dim=int(attention_embed_dim),
            attention_num_heads=int(attention_num_heads),
            terrain_encoder_hidden_dim=int(terrain_encoder_hidden_dim),
            privileged_latent_dim=int(privileged_latent_dim),
            privileged_hidden_dims=list(privileged_hidden_dims),
            terrain_scan_shape=terrain_scan_shape,
            use_scan_positional_encoding=bool(use_scan_positional_encoding),
        )
        self.critic = CrossModalTeacherTower(
            proprio_dim=proprio_dim,
            terrain_dim=terrain_dim,
            privileged_dim=privileged_dim,
            output_dim=1,
            hidden_dims=list(critic_hidden_dims),
            activation=activation,
            attention_embed_dim=int(attention_embed_dim),
            attention_num_heads=int(attention_num_heads),
            terrain_encoder_hidden_dim=int(terrain_encoder_hidden_dim),
            privileged_latent_dim=int(privileged_latent_dim),
            privileged_hidden_dims=list(privileged_hidden_dims),
            terrain_scan_shape=terrain_scan_shape,
            use_scan_positional_encoding=bool(use_scan_positional_encoding),
        )

        self.actor_obs_normalization = bool(actor_obs_normalization)
        if self.actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(actor_total_obs)
        else:
            self.actor_obs_normalizer = nn.Identity()
        self.critic_obs_normalization = bool(critic_obs_normalization)
        if self.critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(critic_total_obs)
        else:
            self.critic_obs_normalizer = nn.Identity()

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        self.last_actor_attention_weights = None
        self._reset_logging_stats()
        Normal.set_default_validate_args(False)

        if pretrained_attention_checkpoint:
            self.load_pretrained_attention(
                pretrained_attention_checkpoint,
                load_actor=bool(load_pretrained_attention_into_actor),
                load_critic=bool(load_pretrained_attention_into_critic),
            )
        if freeze_pretrained_attention:
            self.freeze_pretrained_attention(
                freeze_actor=bool(load_pretrained_attention_into_actor),
                freeze_critic=bool(load_pretrained_attention_into_critic),
            )

    @property
    def terrain_scan_shape(self) -> tuple[int, int]:
        return self.actor.terrain_scan_shape

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

    def get_actor_obs(self, obs):
        obs_list = [obs[group_name] for group_name in self._actor_group_names]
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs):
        obs_list = [obs[group_name] for group_name in self._critic_group_names]
        return torch.cat(obs_list, dim=-1)

    def _split_flat_obs(
        self,
        flat_obs: torch.Tensor,
        group_names: list[str],
        group_dims: dict[str, int],
    ) -> dict[str, torch.Tensor]:
        split_obs: dict[str, torch.Tensor] = {}
        offset = 0
        for group_name in group_names:
            dim = int(group_dims[group_name])
            split_obs[group_name] = flat_obs[:, offset : offset + dim]
            offset += dim
        return split_obs

    def _prepare_actor_groups(self, obs) -> dict[str, torch.Tensor]:
        if self.actor_obs_normalization:
            flat_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
            return self._split_flat_obs(flat_obs, self._actor_group_names, self._actor_group_dims)
        return {group_name: obs[group_name] for group_name in self._actor_group_names}

    def _prepare_critic_groups(self, obs) -> dict[str, torch.Tensor]:
        if self.critic_obs_normalization:
            flat_obs = self.critic_obs_normalizer(self.get_critic_obs(obs))
            return self._split_flat_obs(flat_obs, self._critic_group_names, self._critic_group_dims)
        return {group_name: obs[group_name] for group_name in self._critic_group_names}

    def _forward_actor(self, obs, need_attention: bool = False) -> tuple[torch.Tensor, torch.Tensor | None]:
        obs_groups = self._prepare_actor_groups(obs)
        return self.actor(
            proprio=obs_groups[self.proprio_obs_group],
            terrain_scan=obs_groups[self.terrain_obs_group],
            privileged_state=obs_groups[self.privileged_obs_group],
            need_attention=need_attention,
        )

    def _forward_critic(self, obs) -> torch.Tensor:
        obs_groups = self._prepare_critic_groups(obs)
        value, _ = self.critic(
            proprio=obs_groups[self.proprio_obs_group],
            terrain_scan=obs_groups[self.terrain_obs_group],
            privileged_state=obs_groups[self.privileged_obs_group],
            need_attention=False,
        )
        return value

    def load_pretrained_attention(
        self,
        checkpoint: str | Path | dict[str, torch.Tensor] | dict[str, object],
        load_actor: bool = True,
        load_critic: bool = True,
    ) -> dict[str, object]:
        attention_state_dict, metadata = _extract_attention_state_dict(checkpoint)
        if load_actor:
            self.actor.load_pretrained_attention(attention_state_dict)
        if load_critic:
            self.critic.load_pretrained_attention(attention_state_dict)
        self.pretrained_attention_metadata = dict(metadata)
        return self.pretrained_attention_metadata

    def freeze_pretrained_attention(self, freeze_actor: bool = True, freeze_critic: bool = True) -> None:
        if freeze_actor:
            self.actor.freeze_pretrained_attention()
        if freeze_critic:
            self.critic.freeze_pretrained_attention()

    def _reset_logging_stats(self) -> None:
        metric_names = (
            "Policy/real_teacher_actor_attention_entropy",
            "Policy/real_teacher_actor_attention_max",
            "Policy/real_teacher_actor_attention_top5_mass",
            "Policy/real_teacher_terrain_scan_std",
            "Policy/real_teacher_terrain_scan_abs_mean",
        )
        self._logging_metric_sums = {name: 0.0 for name in metric_names}
        self._logging_metric_counts = {name: 0.0 for name in metric_names}

    def _record_logging_stats(self, attention: torch.Tensor | None, terrain_scan: torch.Tensor) -> None:
        terrain_scan = terrain_scan.detach()
        terrain_std = terrain_scan.std(dim=-1)
        terrain_abs_mean = terrain_scan.abs().mean(dim=-1)

        self._logging_metric_sums["Policy/real_teacher_terrain_scan_std"] += float(terrain_std.sum().item())
        self._logging_metric_counts["Policy/real_teacher_terrain_scan_std"] += float(terrain_std.numel())
        self._logging_metric_sums["Policy/real_teacher_terrain_scan_abs_mean"] += float(terrain_abs_mean.sum().item())
        self._logging_metric_counts["Policy/real_teacher_terrain_scan_abs_mean"] += float(terrain_abs_mean.numel())

        if attention is None:
            return

        attention = attention.detach()
        num_scan_points = max(int(attention.shape[-1]), 1)
        log_norm = max(math.log(float(num_scan_points)), 1.0e-6)
        attention_entropy = -(attention * attention.clamp_min(1.0e-8).log()).sum(dim=-1) / log_norm
        attention_max = attention.amax(dim=-1)
        topk = min(5, num_scan_points)
        attention_topk_mass = attention.topk(k=topk, dim=-1).values.sum(dim=-1)

        self._logging_metric_sums["Policy/real_teacher_actor_attention_entropy"] += float(attention_entropy.sum().item())
        self._logging_metric_counts["Policy/real_teacher_actor_attention_entropy"] += float(attention_entropy.numel())
        self._logging_metric_sums["Policy/real_teacher_actor_attention_max"] += float(attention_max.sum().item())
        self._logging_metric_counts["Policy/real_teacher_actor_attention_max"] += float(attention_max.numel())
        self._logging_metric_sums["Policy/real_teacher_actor_attention_top5_mass"] += float(attention_topk_mass.sum().item())
        self._logging_metric_counts["Policy/real_teacher_actor_attention_top5_mass"] += float(attention_topk_mass.numel())

    def pop_logging_stats(self) -> dict[str, float]:
        metrics: dict[str, float] = {}
        for key, total in self._logging_metric_sums.items():
            count = self._logging_metric_counts[key]
            if count > 0.0:
                metrics[key] = total / count
        self._reset_logging_stats()
        return metrics

    def update_distribution(self, obs):
        obs_groups = self._prepare_actor_groups(obs)
        mean, attention = self.actor(
            proprio=obs_groups[self.proprio_obs_group],
            terrain_scan=obs_groups[self.terrain_obs_group],
            privileged_state=obs_groups[self.privileged_obs_group],
            need_attention=True,
        )
        self.last_actor_attention_weights = attention
        self._record_logging_stats(attention, obs_groups[self.terrain_obs_group])
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, obs, **kwargs):
        self.update_distribution(obs)
        return self.distribution.sample()

    def act_inference(self, obs):
        obs_groups = self._prepare_actor_groups(obs)
        mean, attention = self.actor(
            proprio=obs_groups[self.proprio_obs_group],
            terrain_scan=obs_groups[self.terrain_obs_group],
            privileged_state=obs_groups[self.privileged_obs_group],
            need_attention=True,
        )
        self.last_actor_attention_weights = attention
        self._record_logging_stats(attention, obs_groups[self.terrain_obs_group])
        return mean

    def evaluate(self, obs, **kwargs):
        return self._forward_critic(obs)

    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs):
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def forward_actor_debug(self, obs) -> AttentionDebugResult:
        action_mean, attention = self._forward_actor(obs, need_attention=True)
        if attention is None:
            raise RuntimeError("Attention weights were not produced by the actor.")
        return AttentionDebugResult(
            action_mean=action_mean,
            attention_weights=attention,
            terrain_scan=obs[self.terrain_obs_group],
            terrain_scan_shape=self.terrain_scan_shape,
        )

    def get_actor_attention_map(self, obs) -> torch.Tensor:
        debug_result = self.forward_actor_debug(obs)
        rows, cols = debug_result.terrain_scan_shape
        return debug_result.attention_weights.reshape(debug_result.attention_weights.shape[0], -1, rows, cols)

    def load_state_dict(self, state_dict, strict=True):
        super().load_state_dict(state_dict, strict=strict)
        return True
