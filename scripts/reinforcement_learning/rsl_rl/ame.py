from __future__ import annotations

import torch
import torch.nn as nn

from rsl_rl.models.mlp_model import MLP, MLPModel


class AMETerrainEncoderModel(MLPModel):
    """RSL-RL 3.x model with an attention-based terrain-map encoder."""

    def __init__(
        self,
        obs,
        obs_groups,
        obs_set: str,
        output_dim: int,
        hidden_dims=(512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        map_scan_dim: tuple[int, int, int] = (33, 21, 3),
        mha_dim: int = 64,
        num_heads: int = 16,
        cnn_downsample: bool = True,
        attach_global: bool = False,
    ) -> None:
        super().__init__(
            obs=obs,
            obs_groups=obs_groups,
            obs_set=obs_set,
            output_dim=output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=obs_normalization,
            distribution_cfg=distribution_cfg,
        )
        self.map_scan_dim = tuple(map_scan_dim)
        self.map_l, self.map_w, self.coord_dim = self.map_scan_dim
        self.map_scan_size = self.map_l * self.map_w * self.coord_dim
        self.mha_dim = int(mha_dim)
        self.num_heads = int(num_heads)
        self.cnn_downsample = bool(cnn_downsample)
        self.attach_global = bool(attach_global)

        self.proprio_dim = self.obs_dim - self.map_scan_size
        if self.proprio_dim <= 0:
            raise ValueError(
                f"AME expects observations to end with a flattened map of size {self.map_scan_size}, "
                f"but {obs_set} observation dim is {self.obs_dim}."
            )

        self.map_cnn = self._make_map_cnn()
        self.proprio_embedding = nn.Linear(self.proprio_dim, self.mha_dim)
        self.mha = nn.MultiheadAttention(embed_dim=self.mha_dim, num_heads=self.num_heads, batch_first=True)
        if self.attach_global:
            self.global_encoder = MLP(self.mha_dim, self.mha_dim, [256, 128], "elu")
            self.query_projector = nn.Linear(self.mha_dim * 2, self.mha_dim)
        else:
            self.global_encoder = None
            self.query_projector = None

        encoded_dim = self.mha_dim + self.proprio_dim + (self.mha_dim if self.attach_global else 0)
        mlp_output_dim = self.distribution.input_dim if self.distribution is not None else output_dim
        self.mlp = MLP(encoded_dim, mlp_output_dim, hidden_dims, activation)
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.mlp)

    def _make_map_cnn(self) -> nn.Sequential:
        if not self.cnn_downsample:
            return nn.Sequential(
                nn.Conv2d(self.coord_dim, 16, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.BatchNorm2d(16),
                nn.Conv2d(16, self.mha_dim, kernel_size=5, padding=2),
                nn.ReLU(),
                nn.BatchNorm2d(self.mha_dim),
            )
        return nn.Sequential(
            nn.Conv2d(self.coord_dim, 16, kernel_size=5, padding=2, stride=2),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, self.mha_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(self.mha_dim),
        )

    def get_latent(self, obs, masks: torch.Tensor | None = None, hidden_state=None) -> torch.Tensor:
        latent = torch.cat([obs[obs_group] for obs_group in self.obs_groups], dim=-1)
        latent = self.obs_normalizer(latent)

        map_scan = latent[:, -self.map_scan_size :].reshape(-1, self.map_w, self.map_l, self.coord_dim)
        proprio_obs = latent[:, : -self.map_scan_size]

        map_features = self.map_cnn(map_scan.permute(0, 3, 1, 2))
        map_features = map_features.permute(0, 2, 3, 1).reshape(map_features.shape[0], -1, self.mha_dim)

        proprio_query = self.proprio_embedding(proprio_obs)
        if self.attach_global:
            global_features = self.global_encoder(map_features)
            global_features_max = torch.max(global_features, dim=1)[0]
            proprio_query = self.query_projector(torch.cat([global_features_max, proprio_query], dim=-1))
        else:
            global_features_max = None

        need_attention_weights = not self.training
        attention_output, attention_weights = self.mha(
            query=proprio_query.unsqueeze(1),
            key=map_features,
            value=map_features,
            need_weights=need_attention_weights,
            average_attn_weights=False,
        )
        self.last_attention_weights = attention_weights.squeeze(2).detach() if attention_weights is not None else None
        encoded = torch.cat([attention_output.squeeze(1), proprio_obs], dim=-1)
        if global_features_max is not None:
            encoded = torch.cat([global_features_max, encoded], dim=-1)
        return encoded
