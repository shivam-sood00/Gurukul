from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules import MLP


class LidarPointNetModel(MLPModel):
    """RSL-RL model that encodes a flat lidar point cloud with a PointNet-style branch."""

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (512, 256, 128),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        stochastic: bool = False,
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        lidar_obs_group: str = "lidar",
        point_feature_dim: int = 4,
        pointnet_feature_dim: int = 64,
        pointnet_hidden_dims: tuple[int, ...] | list[int] = (64, 128),
    ) -> None:
        self.lidar_obs_group = str(lidar_obs_group)
        self.point_feature_dim = int(point_feature_dim)
        self.pointnet_feature_dim = int(pointnet_feature_dim)
        self.pointnet_hidden_dims = list(pointnet_hidden_dims)
        self._obs_group_dims: dict[str, int] = {}
        if distribution_cfg is None and stochastic:
            distribution_cfg = {
                "class_name": "HeteroscedasticGaussianDistribution" if state_dependent_std else "GaussianDistribution",
                "init_std": float(init_noise_std),
                "std_type": str(noise_std_type),
            }
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
        if self.lidar_obs_group not in self.obs_groups:
            raise ValueError(
                f"LidarPointNetModel expected observation group {self.lidar_obs_group!r} in {obs_set!r}, "
                f"got {self.obs_groups}."
            )
        lidar_dim = self._obs_group_dims[self.lidar_obs_group]
        if lidar_dim % self.point_feature_dim != 0:
            raise ValueError(
                f"Lidar observation dimension {lidar_dim} is not divisible by point_feature_dim={self.point_feature_dim}."
            )
        self.num_lidar_points = lidar_dim // self.point_feature_dim
        self.point_encoder = MLP(
            self.point_feature_dim,
            self.pointnet_feature_dim,
            self.pointnet_hidden_dims,
            activation=activation,
        )
        self.point_pool_projection = MLP(
            2 * self.pointnet_feature_dim,
            self.pointnet_feature_dim,
            [self.pointnet_feature_dim],
            activation=activation,
        )

    def get_latent(self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state=None) -> torch.Tensor:
        obs_list = []
        for obs_group in self.obs_groups:
            if obs_group == self.lidar_obs_group:
                obs_list.append(self._encode_lidar(obs[obs_group]))
            else:
                obs_list.append(obs[obs_group])
        latent = torch.cat(obs_list, dim=-1)
        return self.obs_normalizer(latent)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.obs_normalization:
            obs_list = []
            for obs_group in self.obs_groups:
                if obs_group == self.lidar_obs_group:
                    obs_list.append(self._encode_lidar(obs[obs_group]))
                else:
                    obs_list.append(obs[obs_group])
            self.obs_normalizer.update(torch.cat(obs_list, dim=-1))

    def _encode_lidar(self, lidar_obs: torch.Tensor) -> torch.Tensor:
        points = lidar_obs.reshape(lidar_obs.shape[0], self.num_lidar_points, self.point_feature_dim)
        point_features = self.point_encoder(points.reshape(-1, self.point_feature_dim)).reshape(
            points.shape[0],
            points.shape[1],
            self.pointnet_feature_dim,
        )
        max_pool = point_features.max(dim=1).values
        mean_pool = point_features.mean(dim=1)
        return self.point_pool_projection(torch.cat((max_pool, mean_pool), dim=-1))

    def _get_obs_dim(self, obs: TensorDict, obs_groups: dict[str, list[str]], obs_set: str) -> tuple[list[str], int]:
        active_obs_groups = obs_groups[obs_set]
        obs_dim = 0
        for obs_group in active_obs_groups:
            if len(obs[obs_group].shape) != 2:
                raise ValueError(
                    f"LidarPointNetModel only supports flat observation groups, got shape {obs[obs_group].shape} "
                    f"for {obs_group!r}."
                )
            dim = int(obs[obs_group].shape[-1])
            self._obs_group_dims[obs_group] = dim
            obs_dim += self.pointnet_feature_dim if obs_group == self.lidar_obs_group else dim
        return active_obs_groups, obs_dim

    def _get_latent_dim(self) -> int:
        return self.obs_dim
