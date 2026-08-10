"""Depth-backbone student-teacher policy for distillation."""

from __future__ import annotations

import copy
import math
from collections import deque

import torch
import torch.nn as nn
import torch.nn.functional as F
from rsl_rl.algorithms.distillation import Distillation
from rsl_rl.utils import resolve_callable, resolve_nn_activation, resolve_obs_groups, resolve_optimizer, unpad_trajectories
from torch.distributions import Normal

from .rsl_rl_compat import MLP, EmpiricalNormalization, Memory


class DepthBackboneActor(nn.Module):
    """Actor that encodes depth with a CNN backbone and fuses with proprioception."""

    def __init__(
        self,
        total_obs_dim: int,
        proprio_slices: list[tuple[int, int]],
        depth_slice: tuple[int, int],
        depth_image_shape: tuple[int, int],
        num_actions: int,
        actor_hidden_dims: list[int],
        activation: str,
        depth_backbone_channels: list[int],
        depth_backbone_kernels: list[int],
        depth_backbone_pool_kernel: int,
        depth_backbone_image_fc_dim: int,
        depth_backbone_latent_dim: int,
    ):
        super().__init__()

        self.total_obs_dim = int(total_obs_dim)
        self.proprio_slices = list(proprio_slices)
        self.depth_slice = (int(depth_slice[0]), int(depth_slice[1]))
        self.depth_h = int(depth_image_shape[0])
        self.depth_w = int(depth_image_shape[1])
        self.depth_dim = self.depth_slice[1] - self.depth_slice[0]

        if self.depth_h * self.depth_w != self.depth_dim:
            raise ValueError(
                f"Depth image shape {depth_image_shape} is incompatible with depth dimension {self.depth_dim}."
            )

        if len(depth_backbone_channels) != len(depth_backbone_kernels):
            raise ValueError(
                "depth_backbone_channels and depth_backbone_kernels must have the same length."
            )

        if len(depth_backbone_channels) < 1:
            raise ValueError("Depth backbone must contain at least one convolution layer.")

        def _act() -> nn.Module:
            # Resolve a fresh activation module instance each time.
            return copy.deepcopy(resolve_nn_activation(activation))

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

        self.depth_cnn = nn.Sequential(*conv_layers)
        self.depth_flatten = nn.Flatten()
        flattened_dim = in_channels * cur_h * cur_w
        self.depth_projector = nn.Sequential(
            nn.Linear(flattened_dim, int(depth_backbone_image_fc_dim)),
            _act(),
            nn.Linear(int(depth_backbone_image_fc_dim), int(depth_backbone_latent_dim)),
            _act(),
        )

        proprio_dim = sum(end - start for start, end in self.proprio_slices)
        self.actor_mlp = MLP(
            input_dim=proprio_dim + int(depth_backbone_latent_dim),
            output_dim=int(num_actions),
            hidden_dims=list(actor_hidden_dims),
            activation=activation,
        )

    def _split(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.shape[-1] != self.total_obs_dim:
            raise ValueError(f"Expected obs dim {self.total_obs_dim}, got {obs.shape[-1]}")

        proprio_parts = [obs[:, start:end] for start, end in self.proprio_slices]
        proprio = torch.cat(proprio_parts, dim=-1) if proprio_parts else obs.new_zeros((obs.shape[0], 0))
        depth = obs[:, self.depth_slice[0] : self.depth_slice[1]]
        return proprio, depth

    def encode_depth(self, depth_flat: torch.Tensor) -> torch.Tensor:
        depth_image = depth_flat.view(-1, 1, self.depth_h, self.depth_w)
        depth_feat = self.depth_cnn(depth_image)
        depth_feat = self.depth_flatten(depth_feat)
        return self.depth_projector(depth_feat)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        proprio, depth_flat = self._split(obs)
        depth_latent = self.encode_depth(depth_flat)
        fused = torch.cat((proprio, depth_latent), dim=-1)
        return self.actor_mlp(fused)


class StudentTeacherDepthBackbone(nn.Module):
    """Student-teacher model with depth CNN backbone for the student path."""

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[256, 256, 256],
        teacher_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=0.1,
        noise_std_type: str = "scalar",
        depth_obs_group: str = "depth_camera",
        depth_image_shape: tuple[int, int] = (58, 87),
        depth_backbone_channels: list[int] = [32, 64],
        depth_backbone_kernels: list[int] = [5, 3],
        depth_backbone_pool_kernel: int = 2,
        depth_backbone_image_fc_dim: int = 128,
        depth_backbone_latent_dim: int = 32,
        **kwargs,
    ):
        if kwargs:
            print(
                "StudentTeacherDepthBackbone.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.loaded_teacher = False
        self.obs_groups = obs_groups
        self.depth_obs_group = depth_obs_group

        policy_groups = list(obs_groups["policy"])
        if depth_obs_group not in policy_groups:
            raise ValueError(
                f"Depth observation group '{depth_obs_group}' not found in policy obs groups: {policy_groups}"
            )

        # Build student-group index map on the concatenated student observation vector.
        offset = 0
        proprio_slices: list[tuple[int, int]] = []
        depth_slice: tuple[int, int] | None = None
        num_student_obs = 0
        for group_name in policy_groups:
            assert len(obs[group_name].shape) == 2, "StudentTeacherDepthBackbone only supports 1D observation groups."
            dim = int(obs[group_name].shape[-1])
            start, end = offset, offset + dim
            if group_name == depth_obs_group:
                depth_slice = (start, end)
            else:
                proprio_slices.append((start, end))
            offset = end
            num_student_obs += dim

        if depth_slice is None:
            raise ValueError(f"Depth group '{depth_obs_group}' slice was not built.")

        # Teacher observation dimension.
        num_teacher_obs = 0
        for obs_group in obs_groups["teacher"]:
            assert len(obs[obs_group].shape) == 2, "StudentTeacherDepthBackbone only supports 1D teacher groups."
            num_teacher_obs += int(obs[obs_group].shape[-1])

        self.student = DepthBackboneActor(
            total_obs_dim=num_student_obs,
            proprio_slices=proprio_slices,
            depth_slice=depth_slice,
            depth_image_shape=depth_image_shape,
            num_actions=num_actions,
            actor_hidden_dims=student_hidden_dims,
            activation=activation,
            depth_backbone_channels=depth_backbone_channels,
            depth_backbone_kernels=depth_backbone_kernels,
            depth_backbone_pool_kernel=depth_backbone_pool_kernel,
            depth_backbone_image_fc_dim=depth_backbone_image_fc_dim,
            depth_backbone_latent_dim=depth_backbone_latent_dim,
        )

        self.student_obs_normalization = student_obs_normalization
        if student_obs_normalization:
            self.student_obs_normalizer = EmpiricalNormalization(num_student_obs)
        else:
            self.student_obs_normalizer = torch.nn.Identity()

        self.teacher = MLP(num_teacher_obs, num_actions, teacher_hidden_dims, activation)
        self.teacher.eval()

        self.teacher_obs_normalization = teacher_obs_normalization
        if teacher_obs_normalization:
            self.teacher_obs_normalizer = EmpiricalNormalization(num_teacher_obs)
        else:
            self.teacher_obs_normalizer = torch.nn.Identity()

        print(f"Student Depth-Backbone Actor: {self.student}")
        print(f"Teacher MLP: {self.teacher}")

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    def reset(self, dones=None, hidden_states=None):
        pass

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

    def update_distribution(self, obs):
        mean = self.student(obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, obs):
        obs = self.get_student_obs(obs)
        obs = self.student_obs_normalizer(obs)
        # Rollout collection in distillation does not require gradients.
        # Keep gradients only in act_inference() during the update phase.
        with torch.no_grad():
            self.update_distribution(obs)
            return self.distribution.sample()

    def act_inference(self, obs):
        obs = self.get_student_obs(obs)
        obs = self.student_obs_normalizer(obs)
        return self.student(obs)

    def evaluate(self, obs):
        obs = self.get_teacher_obs(obs)
        obs = self.teacher_obs_normalizer(obs)
        with torch.no_grad():
            return self.teacher(obs)

    def get_student_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_teacher_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["teacher"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_hidden_states(self):
        return None

    def detach_hidden_states(self, dones=None):
        pass

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        self.teacher_obs_normalizer.eval()

    def update_normalization(self, obs):
        if self.student_obs_normalization:
            student_obs = self.get_student_obs(obs)
            self.student_obs_normalizer.update(student_obs)

    def load_state_dict(self, state_dict, strict=True):
        """Load parameters from teacher RL training or distillation training."""
        if any(key.startswith("actor.") or key.startswith("actor_obs_normalizer.") for key in state_dict.keys()):
            teacher_state_dict = {}
            teacher_obs_normalizer_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    teacher_state_dict[key.replace("actor.", "")] = value
                if "actor_obs_normalizer." in key:
                    teacher_obs_normalizer_state_dict[key.replace("actor_obs_normalizer.", "")] = value
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            self.teacher_obs_normalizer.load_state_dict(teacher_obs_normalizer_state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return False
        elif any(key.startswith("student.") for key in state_dict.keys()):
            super().load_state_dict(state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return True
        else:
            raise ValueError("state_dict does not contain student or teacher parameters")


class DepthBackboneActorModel(nn.Module):
    """RSL-RL model wrapper for a stochastic depth-CNN actor."""

    is_recurrent = False

    def __init__(
        self,
        obs,
        obs_groups,
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        distribution_cfg: dict | None = None,
        depth_obs_group: str = "depth_camera",
        depth_image_shape: tuple[int, int] = (58, 87),
        depth_backbone_channels: list[int] = [32, 64],
        depth_backbone_kernels: list[int] = [5, 3],
        depth_backbone_pool_kernel: int = 2,
        depth_backbone_image_fc_dim: int = 128,
        depth_backbone_latent_dim: int = 32,
        **kwargs,
    ) -> None:
        if kwargs:
            print(
                "DepthBackboneActorModel.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.obs_groups, obs_dim, proprio_slices, depth_slice = self._get_obs_layout(
            obs, obs_groups, obs_set, depth_obs_group
        )
        self.obs_dim = obs_dim
        self.obs_normalization = obs_normalization
        if obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(obs_dim)
        else:
            self.obs_normalizer = torch.nn.Identity()

        if distribution_cfg is not None:
            distribution_cfg = copy.deepcopy(distribution_cfg)
            dist_class = resolve_callable(distribution_cfg.pop("class_name"))
            self.distribution = dist_class(output_dim, **distribution_cfg)
            actor_output_dim = self.distribution.input_dim
        else:
            self.distribution = None
            actor_output_dim = output_dim

        self.actor = DepthBackboneActor(
            total_obs_dim=obs_dim,
            proprio_slices=proprio_slices,
            depth_slice=depth_slice,
            depth_image_shape=depth_image_shape,
            num_actions=actor_output_dim,
            actor_hidden_dims=list(hidden_dims),
            activation=activation,
            depth_backbone_channels=depth_backbone_channels,
            depth_backbone_kernels=depth_backbone_kernels,
            depth_backbone_pool_kernel=depth_backbone_pool_kernel,
            depth_backbone_image_fc_dim=depth_backbone_image_fc_dim,
            depth_backbone_latent_dim=depth_backbone_latent_dim,
        )
        if self.distribution is not None:
            self.distribution.init_mlp_weights(self.actor.actor_mlp)

    def forward(self, obs, masks=None, hidden_state=None, stochastic_output: bool = False):
        obs = unpad_trajectories(obs, masks) if masks is not None else obs
        latent = self.get_latent(obs)
        actor_output = self.actor(latent)
        if self.distribution is not None:
            if stochastic_output:
                self.distribution.update(actor_output)
                return self.distribution.sample()
            return self.distribution.deterministic_output(actor_output)
        return actor_output

    def get_latent(self, obs, masks=None, hidden_state=None):
        obs_list = [obs[obs_group] for obs_group in self.obs_groups]
        latent = torch.cat(obs_list, dim=-1)
        return self.obs_normalizer(latent)

    def reset(self, dones=None, hidden_state=None):
        pass

    def get_hidden_state(self):
        return None

    def detach_hidden_state(self, dones=None):
        pass

    @property
    def output_mean(self):
        return self.distribution.mean

    @property
    def output_std(self):
        return self.distribution.std

    @property
    def output_entropy(self):
        return self.distribution.entropy

    @property
    def output_distribution_params(self):
        return self.distribution.params

    def get_output_log_prob(self, outputs):
        return self.distribution.log_prob(outputs)

    def get_kl_divergence(self, old_params, new_params):
        return self.distribution.kl_divergence(old_params, new_params)

    def update_normalization(self, obs) -> None:
        if self.obs_normalization:
            obs_list = [obs[obs_group] for obs_group in self.obs_groups]
            self.obs_normalizer.update(torch.cat(obs_list, dim=-1))

    def _get_obs_layout(self, obs, obs_groups, obs_set: str, depth_obs_group: str):
        active_obs_groups = obs_groups[obs_set]
        if depth_obs_group not in active_obs_groups:
            raise ValueError(
                f"Depth observation group '{depth_obs_group}' not found in {obs_set} obs groups: {active_obs_groups}"
            )

        offset = 0
        obs_dim = 0
        proprio_slices: list[tuple[int, int]] = []
        depth_slice: tuple[int, int] | None = None
        for group_name in active_obs_groups:
            if len(obs[group_name].shape) != 2:
                raise ValueError(
                    f"DepthBackboneActorModel only supports 1D observations, got {obs[group_name].shape} "
                    f"for '{group_name}'."
                )
            dim = int(obs[group_name].shape[-1])
            start, end = offset, offset + dim
            if group_name == depth_obs_group:
                depth_slice = (start, end)
            else:
                proprio_slices.append((start, end))
            offset = end
            obs_dim += dim

        if depth_slice is None:
            raise ValueError(f"Depth group '{depth_obs_group}' slice was not built.")
        return list(active_obs_groups), obs_dim, proprio_slices, depth_slice


class StartDepthRecurrentActor(nn.Module):
    """START-style student actor with depth encoding, recurrent memory, and terrain reconstruction heads."""

    def __init__(
        self,
        total_obs_dim: int,
        proprio_slices: list[tuple[int, int]],
        depth_slice: tuple[int, int],
        depth_image_shape: tuple[int, int],
        num_actions: int,
        actor_hidden_dims: list[int],
        activation: str,
        depth_backbone_channels: list[int],
        depth_backbone_kernels: list[int],
        depth_backbone_pool_kernel: int,
        depth_backbone_image_fc_dim: int,
        depth_backbone_latent_dim: int,
        depth_recurrent_fusion_dim: int,
        rnn_type: str,
        rnn_hidden_dim: int,
        rnn_num_layers: int,
        terrain_map_dim: int,
        terrain_reconstruction_hidden_dim: int,
        terrain_reconstruction_latent_dim: int,
    ):
        super().__init__()

        self.total_obs_dim = int(total_obs_dim)
        self.proprio_slices = list(proprio_slices)
        self.depth_slice = (int(depth_slice[0]), int(depth_slice[1]))
        self.depth_h = int(depth_image_shape[0])
        self.depth_w = int(depth_image_shape[1])
        self.depth_dim = self.depth_slice[1] - self.depth_slice[0]
        self.terrain_map_dim = int(terrain_map_dim)

        if self.depth_h * self.depth_w != self.depth_dim:
            raise ValueError(
                f"Depth image shape {depth_image_shape} is incompatible with depth dimension {self.depth_dim}."
            )
        if len(depth_backbone_channels) != len(depth_backbone_kernels):
            raise ValueError("depth_backbone_channels and depth_backbone_kernels must have the same length.")
        if len(depth_backbone_channels) < 1:
            raise ValueError("Depth backbone must contain at least one convolution layer.")
        if self.terrain_map_dim <= 0:
            raise ValueError("terrain_map_dim must be positive for START terrain reconstruction.")

        def _act() -> nn.Module:
            return copy.deepcopy(resolve_nn_activation(activation))

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

        self.depth_cnn = nn.Sequential(*conv_layers)
        self.depth_flatten = nn.Flatten()
        flattened_dim = in_channels * cur_h * cur_w
        self.depth_projector = nn.Sequential(
            nn.Linear(flattened_dim, int(depth_backbone_image_fc_dim)),
            _act(),
            nn.Linear(int(depth_backbone_image_fc_dim), int(depth_backbone_latent_dim)),
            _act(),
        )

        proprio_dim = sum(end - start for start, end in self.proprio_slices)
        self.fusion = nn.Sequential(
            nn.Linear(proprio_dim + int(depth_backbone_latent_dim), int(depth_recurrent_fusion_dim)),
            _act(),
        )
        self.memory = Memory(
            input_size=int(depth_recurrent_fusion_dim),
            type=rnn_type,
            num_layers=int(rnn_num_layers),
            hidden_size=int(rnn_hidden_dim),
        )

        recon_hidden = int(terrain_reconstruction_hidden_dim)
        self.terrain_rough_decoder = nn.Sequential(
            nn.Linear(int(rnn_hidden_dim), recon_hidden),
            _act(),
            nn.Linear(recon_hidden, self.terrain_map_dim),
        )
        self.terrain_refine_decoder = nn.Sequential(
            nn.Linear(int(rnn_hidden_dim) + self.terrain_map_dim, recon_hidden),
            _act(),
            nn.Linear(recon_hidden, self.terrain_map_dim),
        )
        self.terrain_map_encoder = nn.Sequential(
            nn.Linear(self.terrain_map_dim, recon_hidden),
            _act(),
            nn.Linear(recon_hidden, int(terrain_reconstruction_latent_dim)),
            _act(),
        )

        self.actor_mlp = MLP(
            input_dim=proprio_dim + int(rnn_hidden_dim) + int(terrain_reconstruction_latent_dim),
            output_dim=int(num_actions),
            hidden_dims=list(actor_hidden_dims),
            activation=activation,
        )

    def _split(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.shape[-1] != self.total_obs_dim:
            raise ValueError(f"Expected obs dim {self.total_obs_dim}, got {obs.shape[-1]}")
        proprio_parts = [obs[:, start:end] for start, end in self.proprio_slices]
        proprio = torch.cat(proprio_parts, dim=-1) if proprio_parts else obs.new_zeros((obs.shape[0], 0))
        depth = obs[:, self.depth_slice[0] : self.depth_slice[1]]
        return proprio, depth

    def encode_depth(self, depth_flat: torch.Tensor) -> torch.Tensor:
        depth_image = depth_flat.view(-1, 1, self.depth_h, self.depth_w)
        depth_feat = self.depth_cnn(depth_image)
        depth_feat = self.depth_flatten(depth_feat)
        return self.depth_projector(depth_feat)

    def forward(
        self,
        obs: torch.Tensor,
        terrain_heightmap_gt: torch.Tensor | None = None,
        adasmpl_prob: float = 0.0,
        use_adasmpl: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        proprio, depth_flat = self._split(obs)
        depth_latent = self.encode_depth(depth_flat)
        fused = self.fusion(torch.cat((proprio, depth_latent), dim=-1))
        recurrent_out = self.memory(fused).squeeze(0)

        rough_map = self.terrain_rough_decoder(recurrent_out)
        refine_delta = self.terrain_refine_decoder(torch.cat((recurrent_out, rough_map), dim=-1))
        refined_map = rough_map + refine_delta
        map_for_actor = refined_map

        if use_adasmpl and terrain_heightmap_gt is not None and adasmpl_prob > 0.0:
            p = float(max(0.0, min(1.0, adasmpl_prob)))
            sample_mask = (torch.rand((obs.shape[0], 1), device=obs.device) < p).expand_as(refined_map)
            map_for_actor = torch.where(sample_mask, terrain_heightmap_gt, refined_map)

        map_latent = self.terrain_map_encoder(map_for_actor)
        actor_input = torch.cat((proprio, recurrent_out, map_latent), dim=-1)
        actions = self.actor_mlp(actor_input)
        return actions, rough_map, refined_map, map_for_actor

    def reset(self, dones=None, hidden_states=None):
        self.memory.reset(dones=dones, hidden_states=hidden_states)

    def detach_hidden_states(self, dones=None):
        self.memory.detach_hidden_states(dones=dones)

    def get_hidden_states(self):
        return self.memory.hidden_states


class StudentTeacherDepthBackboneRecurrentSTART(nn.Module):
    """START-style recurrent depth student-teacher model with AdaSmpl and terrain reconstruction losses."""

    is_recurrent = True

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[256, 256, 256],
        teacher_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=0.1,
        noise_std_type: str = "scalar",
        depth_obs_group: str = "depth_camera",
        depth_image_shape: tuple[int, int] = (58, 87),
        depth_backbone_channels: list[int] = [32, 64],
        depth_backbone_kernels: list[int] = [5, 3],
        depth_backbone_pool_kernel: int = 2,
        depth_backbone_image_fc_dim: int = 128,
        depth_backbone_latent_dim: int = 32,
        depth_recurrent_fusion_dim: int = 128,
        rnn_type: str = "gru",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        terrain_heightmap_obs_group: str | None = "terrain_heightmap",
        terrain_reconstruction_dim: int | None = None,
        terrain_reconstruction_hidden_dim: int = 256,
        terrain_reconstruction_latent_dim: int = 64,
        terrain_recon_rough_loss_weight: float = 1.0,
        terrain_recon_refined_loss_weight: float = 1.0,
        terrain_adasmpl_enabled: bool = True,
        terrain_adasmpl_initial_prob: float = 1.0,
        terrain_adasmpl_min_prob: float = 0.0,
        terrain_adasmpl_max_prob: float = 1.0,
        terrain_adasmpl_use_in_update: bool = True,
        **kwargs,
    ):
        if kwargs:
            print(
                "StudentTeacherDepthBackboneRecurrentSTART.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.loaded_teacher = False
        self.obs_groups = obs_groups
        self.depth_obs_group = depth_obs_group
        self.terrain_heightmap_obs_group = terrain_heightmap_obs_group
        self.terrain_recon_rough_loss_weight = float(terrain_recon_rough_loss_weight)
        self.terrain_recon_refined_loss_weight = float(terrain_recon_refined_loss_weight)

        self.terrain_adasmpl_enabled = bool(terrain_adasmpl_enabled)
        self.terrain_adasmpl_use_in_update = bool(terrain_adasmpl_use_in_update)
        self.terrain_adasmpl_min_prob = float(terrain_adasmpl_min_prob)
        self.terrain_adasmpl_max_prob = float(terrain_adasmpl_max_prob)
        self.terrain_adasmpl_probability = float(terrain_adasmpl_initial_prob)
        self.set_adasmpl_probability(self.terrain_adasmpl_probability)
        self._latest_reconstruction: tuple[torch.Tensor, torch.Tensor, torch.Tensor | None] | None = None

        policy_groups = list(obs_groups["policy"])
        if depth_obs_group not in policy_groups:
            raise ValueError(
                f"Depth observation group '{depth_obs_group}' not found in policy obs groups: {policy_groups}"
            )

        offset = 0
        proprio_slices: list[tuple[int, int]] = []
        depth_slice: tuple[int, int] | None = None
        num_student_obs = 0
        for group_name in policy_groups:
            assert len(obs[group_name].shape) == 2, "StudentTeacherDepthBackboneRecurrentSTART expects 1D groups."
            dim = int(obs[group_name].shape[-1])
            start, end = offset, offset + dim
            if group_name == depth_obs_group:
                depth_slice = (start, end)
            else:
                proprio_slices.append((start, end))
            offset = end
            num_student_obs += dim
        if depth_slice is None:
            raise ValueError(f"Depth group '{depth_obs_group}' slice was not built.")

        num_teacher_obs = 0
        for obs_group in obs_groups["teacher"]:
            assert len(obs[obs_group].shape) == 2, "StudentTeacherDepthBackboneRecurrentSTART expects 1D groups."
            num_teacher_obs += int(obs[obs_group].shape[-1])

        terrain_map_dim: int | None = None
        if terrain_heightmap_obs_group is not None and terrain_heightmap_obs_group in obs:
            assert len(obs[terrain_heightmap_obs_group].shape) == 2, "Terrain heightmap observation group must be 1D."
            terrain_map_dim = int(obs[terrain_heightmap_obs_group].shape[-1])
        elif terrain_heightmap_obs_group is not None:
            raise ValueError(
                f"Terrain heightmap group '{terrain_heightmap_obs_group}' not found in observations: {list(obs.keys())}"
            )

        if terrain_reconstruction_dim is not None:
            terrain_reconstruction_dim = int(terrain_reconstruction_dim)
            if terrain_map_dim is not None and terrain_reconstruction_dim != terrain_map_dim:
                raise ValueError(
                    f"terrain_reconstruction_dim ({terrain_reconstruction_dim}) does not match "
                    f"ground-truth terrain dim ({terrain_map_dim})."
                )
            terrain_map_dim = terrain_reconstruction_dim
        if terrain_map_dim is None:
            raise ValueError("START student requires a valid terrain heightmap dimension.")

        self.student = StartDepthRecurrentActor(
            total_obs_dim=num_student_obs,
            proprio_slices=proprio_slices,
            depth_slice=depth_slice,
            depth_image_shape=depth_image_shape,
            num_actions=num_actions,
            actor_hidden_dims=student_hidden_dims,
            activation=activation,
            depth_backbone_channels=depth_backbone_channels,
            depth_backbone_kernels=depth_backbone_kernels,
            depth_backbone_pool_kernel=depth_backbone_pool_kernel,
            depth_backbone_image_fc_dim=depth_backbone_image_fc_dim,
            depth_backbone_latent_dim=depth_backbone_latent_dim,
            depth_recurrent_fusion_dim=depth_recurrent_fusion_dim,
            rnn_type=rnn_type,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
            terrain_map_dim=terrain_map_dim,
            terrain_reconstruction_hidden_dim=terrain_reconstruction_hidden_dim,
            terrain_reconstruction_latent_dim=terrain_reconstruction_latent_dim,
        )

        self.student_obs_normalization = student_obs_normalization
        if student_obs_normalization:
            self.student_obs_normalizer = EmpiricalNormalization(num_student_obs)
        else:
            self.student_obs_normalizer = torch.nn.Identity()

        self.teacher = MLP(num_teacher_obs, num_actions, teacher_hidden_dims, activation)
        self.teacher.eval()

        self.teacher_obs_normalization = teacher_obs_normalization
        if teacher_obs_normalization:
            self.teacher_obs_normalizer = EmpiricalNormalization(num_teacher_obs)
        else:
            self.teacher_obs_normalizer = torch.nn.Identity()

        print(f"Student START Recurrent Actor: {self.student}")
        print(f"Teacher MLP: {self.teacher}")

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    def reset(self, dones=None, hidden_states=None):
        if hidden_states is None:
            student_hidden_states = None
        elif isinstance(hidden_states, tuple):
            student_hidden_states = hidden_states[0]
        else:
            student_hidden_states = hidden_states
        self.student.reset(dones=dones, hidden_states=student_hidden_states)

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

    def set_adasmpl_probability(self, probability: float) -> None:
        probability = float(probability)
        probability = max(self.terrain_adasmpl_min_prob, min(self.terrain_adasmpl_max_prob, probability))
        self.terrain_adasmpl_probability = probability

    def get_adasmpl_probability(self) -> float:
        return float(self.terrain_adasmpl_probability)

    def _extract_terrain_heightmap(self, obs) -> torch.Tensor | None:
        if self.terrain_heightmap_obs_group is None:
            return None
        if self.terrain_heightmap_obs_group not in obs:
            return None
        return obs[self.terrain_heightmap_obs_group]

    def _prepare_student_inputs(self, obs) -> tuple[torch.Tensor, torch.Tensor | None]:
        student_obs = self.get_student_obs(obs)
        student_obs = self.student_obs_normalizer(student_obs)
        terrain_heightmap_gt = self._extract_terrain_heightmap(obs)
        return student_obs, terrain_heightmap_gt

    def _compute_student_mean(
        self,
        obs,
        use_adasmpl: bool,
        requires_grad: bool,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        student_obs, terrain_heightmap_gt = self._prepare_student_inputs(obs)
        adasmpl_prob = self.terrain_adasmpl_probability if use_adasmpl else 0.0
        if requires_grad:
            mean, rough_map, refined_map, _ = self.student(
                student_obs,
                terrain_heightmap_gt=terrain_heightmap_gt,
                adasmpl_prob=adasmpl_prob,
                use_adasmpl=use_adasmpl and terrain_heightmap_gt is not None,
            )
        else:
            with torch.no_grad():
                mean, rough_map, refined_map, _ = self.student(
                    student_obs,
                    terrain_heightmap_gt=terrain_heightmap_gt,
                    adasmpl_prob=adasmpl_prob,
                    use_adasmpl=use_adasmpl and terrain_heightmap_gt is not None,
                )
        self._latest_reconstruction = (rough_map, refined_map, terrain_heightmap_gt)
        return mean, rough_map, refined_map

    def update_distribution(self, mean: torch.Tensor):
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, obs):
        use_adasmpl = self.terrain_adasmpl_enabled and self.training
        mean, _, _ = self._compute_student_mean(obs, use_adasmpl=use_adasmpl, requires_grad=False)
        self.update_distribution(mean)
        return self.distribution.sample()

    def act_inference(self, obs):
        use_adasmpl = self.terrain_adasmpl_enabled and self.training and self.terrain_adasmpl_use_in_update
        mean, _, _ = self._compute_student_mean(obs, use_adasmpl=use_adasmpl, requires_grad=True)
        return mean

    def evaluate(self, obs):
        obs = self.get_teacher_obs(obs)
        obs = self.teacher_obs_normalizer(obs)
        with torch.no_grad():
            return self.teacher(obs)

    def compute_auxiliary_losses(self, obs) -> dict[str, torch.Tensor]:
        terrain_heightmap_gt = self._extract_terrain_heightmap(obs)
        if terrain_heightmap_gt is None:
            zero = torch.zeros((), device=next(self.parameters()).device)
            return {
                "terrain_reconstruction": zero,
                "terrain_reconstruction_rough": zero,
                "terrain_reconstruction_refined": zero,
            }
        student_obs = self.get_student_obs(obs)
        student_obs = self.student_obs_normalizer(student_obs)
        _, rough_map, refined_map, _ = self.student(
            student_obs,
            terrain_heightmap_gt=terrain_heightmap_gt,
            adasmpl_prob=0.0,
            use_adasmpl=False,
        )
        rough_loss = F.mse_loss(rough_map, terrain_heightmap_gt)
        refined_loss = F.l1_loss(refined_map, terrain_heightmap_gt)
        total = (
            self.terrain_recon_rough_loss_weight * rough_loss
            + self.terrain_recon_refined_loss_weight * refined_loss
        )
        return {
            "terrain_reconstruction": total,
            "terrain_reconstruction_rough": rough_loss,
            "terrain_reconstruction_refined": refined_loss,
        }

    def compute_latest_auxiliary_losses(self) -> dict[str, torch.Tensor]:
        """Compute reconstruction losses from the actor pass without advancing its RNN again."""
        if self._latest_reconstruction is None:
            raise RuntimeError("No START student forward pass is available for auxiliary loss computation.")
        rough_map, refined_map, terrain_heightmap_gt = self._latest_reconstruction
        if terrain_heightmap_gt is None:
            zero = rough_map.new_zeros(())
            return {
                "terrain_reconstruction": zero,
                "terrain_reconstruction_rough": zero,
                "terrain_reconstruction_refined": zero,
            }
        rough_loss = F.mse_loss(rough_map, terrain_heightmap_gt)
        refined_loss = F.l1_loss(refined_map, terrain_heightmap_gt)
        total = (
            self.terrain_recon_rough_loss_weight * rough_loss
            + self.terrain_recon_refined_loss_weight * refined_loss
        )
        return {
            "terrain_reconstruction": total,
            "terrain_reconstruction_rough": rough_loss,
            "terrain_reconstruction_refined": refined_loss,
        }

    def get_student_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_teacher_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["teacher"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_hidden_states(self):
        return self.student.get_hidden_states(), None

    def detach_hidden_states(self, dones=None):
        self.student.detach_hidden_states(dones=dones)

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        self.teacher_obs_normalizer.eval()

    def update_normalization(self, obs):
        if self.student_obs_normalization:
            student_obs = self.get_student_obs(obs)
            self.student_obs_normalizer.update(student_obs)

    def load_state_dict(self, state_dict, strict=True):
        """Load parameters from teacher RL training or distillation training."""
        if any(key.startswith("actor.") or key.startswith("actor_obs_normalizer.") for key in state_dict.keys()):
            teacher_state_dict = {}
            teacher_obs_normalizer_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    teacher_state_dict[key.replace("actor.", "")] = value
                if "actor_obs_normalizer." in key:
                    teacher_obs_normalizer_state_dict[key.replace("actor_obs_normalizer.", "")] = value
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            self.teacher_obs_normalizer.load_state_dict(teacher_obs_normalizer_state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return False
        elif any(key.startswith("student.") for key in state_dict.keys()):
            super().load_state_dict(state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return True
        else:
            raise ValueError("state_dict does not contain student or teacher parameters")


class CombinedDistillation(Distillation):
    """RSL-RL 5 adapter for legacy combined student-teacher policy modules."""

    def __init__(
        self,
        policy,
        storage=None,
        num_learning_epochs=1,
        gradient_length=15,
        learning_rate=1.0e-3,
        max_grad_norm=None,
        loss_type="mse",
        optimizer="adam",
        device="cpu",
        multi_gpu_cfg: dict | None = None,
        rnd_cfg: dict | None = None,
        symmetry_cfg: dict | None = None,
        **kwargs,
    ):
        if storage is None:
            raise ValueError("CombinedDistillation requires an initialized distillation RolloutStorage.")
        if rnd_cfg is not None or symmetry_cfg is not None:
            raise ValueError("CombinedDistillation does not support RND or symmetry extensions.")
        if kwargs:
            raise TypeError(f"Unexpected CombinedDistillation arguments: {sorted(kwargs)}")

        self.device = device
        self.is_multi_gpu = multi_gpu_cfg is not None
        self.gpu_global_rank = multi_gpu_cfg["global_rank"] if multi_gpu_cfg is not None else 0
        self.gpu_world_size = multi_gpu_cfg["world_size"] if multi_gpu_cfg is not None else 1
        self.policy = policy.to(device)
        self.student = self.policy
        self.teacher = self.policy
        self._raw_student = self.policy
        self._raw_teacher = self.policy
        self.optimizer = resolve_optimizer(optimizer)(self.policy.student.parameters(), lr=learning_rate)
        self.storage = storage
        from rsl_rl.storage import RolloutStorage

        self.transition = RolloutStorage.Transition()
        self.last_hidden_states = None
        self.num_learning_epochs = int(num_learning_epochs)
        self.gradient_length = max(1, int(gradient_length))
        self.learning_rate = float(learning_rate)
        self.max_grad_norm = max_grad_norm
        loss_functions = {"mse": F.mse_loss, "huber": F.huber_loss}
        if loss_type not in loss_functions:
            raise ValueError(f"Unknown loss type: {loss_type}. Supported values: {sorted(loss_functions)}")
        self.loss_fn = loss_functions[loss_type]
        self.num_updates = 0
        self.teacher_loaded = bool(getattr(self.policy, "loaded_teacher", False))

    @staticmethod
    def construct_algorithm(obs, env, cfg: dict, device: str):
        from rsl_rl.storage import RolloutStorage

        algorithm_name = cfg["algorithm"].pop("class_name", "CombinedDistillation")
        algorithm_classes = {
            "CombinedDistillation": CombinedDistillation,
            "Distillation": CombinedDistillation,
            "StartDistillation": StartDistillation,
        }
        if algorithm_name in algorithm_classes:
            algorithm_class = algorithm_classes[algorithm_name]
        elif ":" in algorithm_name or "." in algorithm_name:
            algorithm_class = resolve_callable(algorithm_name)
        else:
            raise ValueError(f"Unsupported combined distillation algorithm: {algorithm_name}")

        policy_cfg = dict(cfg["policy"])
        policy_name = policy_cfg.pop("class_name", None)
        policy_classes = {
            "StudentTeacherDepthBackbone": StudentTeacherDepthBackbone,
            "StudentTeacherDepthBackboneRecurrent": StudentTeacherDepthBackboneRecurrent,
            "StudentTeacherDepthBackboneRecurrentSTART": StudentTeacherDepthBackboneRecurrentSTART,
        }
        if policy_name in policy_classes:
            policy_class = policy_classes[policy_name]
        elif policy_name and (":" in policy_name or "." in policy_name):
            policy_class = resolve_callable(policy_name)
        else:
            raise ValueError(f"Unsupported combined distillation policy: {policy_name}")

        cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], ["policy", "teacher"])
        policy = policy_class(obs, cfg["obs_groups"], env.num_actions, **policy_cfg).to(device)
        print(f"Combined Student-Teacher Model: {policy}")
        storage = RolloutStorage(
            "distillation",
            env.num_envs,
            cfg["num_steps_per_env"],
            obs,
            [env.num_actions],
            device,
        )
        algorithm = algorithm_class(
            policy,
            storage=storage,
            device=device,
            **cfg["algorithm"],
            multi_gpu_cfg=cfg["multi_gpu"],
        )
        algorithm.compile(cfg.get("torch_compile_mode"))
        return algorithm

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        if training_type != "distillation":
            raise ValueError("CombinedDistillation supports distillation storage only.")
        from rsl_rl.storage import RolloutStorage

        self.storage = RolloutStorage(
            "distillation", num_envs, num_transitions_per_env, obs, actions_shape, self.device
        )
        self.transition = RolloutStorage.Transition()

    def act(self, obs):
        self.transition.actions = self.policy.act(obs).detach()
        self.transition.privileged_actions = self.policy.evaluate(obs).detach()
        self.transition.observations = obs
        return self.transition.actions

    def process_env_step(self, obs, rewards, dones, extras):
        del extras
        self.policy.update_normalization(obs)
        self.transition.rewards = rewards
        self.transition.dones = dones
        self.storage.add_transition(self.transition)
        self.transition.clear()
        self.policy.reset(dones)

    def compute_returns(self, obs):
        del obs

    @staticmethod
    def _distillation_batch(batch):
        if hasattr(batch, "observations"):
            return batch.observations, batch.privileged_actions, batch.dones
        obs, _, privileged_actions, dones = batch
        return obs, privileged_actions, dones

    def _step_accumulated_loss(self, loss):
        self.optimizer.zero_grad()
        loss.backward()
        if self.is_multi_gpu:
            self.reduce_parameters()
        if self.max_grad_norm:
            nn.utils.clip_grad_norm_(self.policy.student.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.policy.detach_hidden_states()

    def update(self):
        self.num_updates += 1
        mean_behavior_loss = 0.0
        count = 0

        for _ in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            accumulated_loss = None
            accumulated_count = 0
            for batch in self.storage.generator():
                obs, privileged_actions, dones = self._distillation_batch(batch)
                actions = self.policy.act_inference(obs)
                behavior_loss = self.loss_fn(actions, privileged_actions)
                accumulated_loss = behavior_loss if accumulated_loss is None else accumulated_loss + behavior_loss
                accumulated_count += 1
                mean_behavior_loss += float(behavior_loss.detach().item())
                count += 1

                if accumulated_count == self.gradient_length:
                    self._step_accumulated_loss(accumulated_loss)
                    accumulated_loss = None
                    accumulated_count = 0

                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

            if accumulated_loss is not None:
                self._step_accumulated_loss(accumulated_loss)

        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()
        return {"behavior": mean_behavior_loss / max(count, 1)}

    def train_mode(self):
        self.policy.train()

    def eval_mode(self):
        self.policy.eval()

    def save(self):
        return {
            "model_state_dict": self.policy.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
        }

    def _load_teacher_checkpoint_state(self, actor_state: dict, strict: bool) -> None:
        """Accept both legacy shared-policy and RSL-RL 5 split-actor checkpoints."""
        if any(key.startswith("actor.") or key.startswith("actor_obs_normalizer.") for key in actor_state):
            self.policy.load_state_dict(actor_state, strict=strict)
        else:
            teacher_state = {
                key.removeprefix("mlp."): value for key, value in actor_state.items() if key.startswith("mlp.")
            }
            if not teacher_state:
                teacher_keys = set(self.policy.teacher.state_dict())
                teacher_state = {key: value for key, value in actor_state.items() if key in teacher_keys}
            if not teacher_state:
                raise ValueError("Teacher checkpoint does not contain compatible actor MLP parameters.")
            self.policy.teacher.load_state_dict(teacher_state, strict=strict)

            normalizer_state = {
                key.removeprefix("obs_normalizer."): value
                for key, value in actor_state.items()
                if key.startswith("obs_normalizer.")
            }
            if normalizer_state:
                self.policy.teacher_obs_normalizer.load_state_dict(normalizer_state, strict=strict)
            self.policy.loaded_teacher = True
            self.policy.teacher.eval()
            self.policy.teacher_obs_normalizer.eval()

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        is_teacher_checkpoint = "actor_state_dict" in loaded_dict and "model_state_dict" not in loaded_dict
        if load_cfg is None:
            load_cfg = (
                {"teacher": True, "iteration": False}
                if is_teacher_checkpoint
                else {"student": True, "teacher": True, "optimizer": True, "iteration": True}
            )
        if load_cfg.get("teacher") and is_teacher_checkpoint:
            self._load_teacher_checkpoint_state(loaded_dict["actor_state_dict"], strict=strict)
            self.teacher_loaded = True
        elif load_cfg.get("student") or load_cfg.get("teacher"):
            model_state = loaded_dict.get("model_state_dict")
            if model_state is None:
                raise KeyError("Combined distillation checkpoint is missing 'model_state_dict'.")
            self.policy.load_state_dict(model_state, strict=strict)
            self.teacher_loaded = True
        if load_cfg.get("optimizer") and "optimizer_state_dict" in loaded_dict:
            self.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
        return bool(load_cfg.get("iteration", False))

    def get_policy(self):
        return self.policy

    def compile(self, mode: str | None = None):
        if mode:
            raise ValueError("torch.compile is not supported for combined distillation policies.")

    def broadcast_parameters(self):
        states = [self.policy.state_dict()]
        torch.distributed.broadcast_object_list(states, src=0)
        self.policy.load_state_dict(states[0])

    def reduce_parameters(self):
        parameters = [parameter for parameter in self.policy.student.parameters() if parameter.grad is not None]
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


class StartDistillation(CombinedDistillation):
    """Distillation variant with START-style auxiliary terrain reconstruction and AdaSmpl updates."""

    def __init__(
        self,
        policy,
        storage=None,
        num_learning_epochs=1,
        gradient_length=15,
        learning_rate=1e-3,
        max_grad_norm=None,
        loss_type="mse",
        optimizer="adam",
        device="cpu",
        multi_gpu_cfg: dict | None = None,
        reconstruction_loss_weight: float = 1.0,
        adasmpl_reward_window: int = 256,
        adasmpl_min_episodes: int = 8,
        adasmpl_cv_scale: float = 1.0,
        adasmpl_probability_ema_alpha: float = 0.2,
        adasmpl_probability_min: float = 0.0,
        adasmpl_probability_max: float = 1.0,
    ):
        super().__init__(
            policy=policy,
            storage=storage,
            num_learning_epochs=num_learning_epochs,
            gradient_length=gradient_length,
            learning_rate=learning_rate,
            max_grad_norm=max_grad_norm,
            loss_type=loss_type,
            optimizer=optimizer,
            device=device,
            multi_gpu_cfg=multi_gpu_cfg,
        )
        self.reconstruction_loss_weight = float(reconstruction_loss_weight)

        self._episode_return_buffer = deque(maxlen=max(8, int(adasmpl_reward_window)))
        self._episode_returns = torch.zeros(storage.num_envs, device=self.device)
        self._adasmpl_min_episodes = max(1, int(adasmpl_min_episodes))
        self._adasmpl_cv_scale = float(adasmpl_cv_scale)
        self._adasmpl_prob_ema_alpha = float(max(0.0, min(1.0, adasmpl_probability_ema_alpha)))
        self._adasmpl_prob_min = float(min(adasmpl_probability_min, adasmpl_probability_max))
        self._adasmpl_prob_max = float(max(adasmpl_probability_min, adasmpl_probability_max))
        self._adasmpl_prob_ema = self._adasmpl_prob_min

        if hasattr(self.policy, "get_adasmpl_probability"):
            self._adasmpl_prob_ema = float(self.policy.get_adasmpl_probability())

    def init_storage(self, training_type, num_envs, num_transitions_per_env, obs, actions_shape):
        super().init_storage(training_type, num_envs, num_transitions_per_env, obs, actions_shape)
        self._episode_returns = torch.zeros(num_envs, device=self.device)

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
            rew = rewards.view(-1)
            done = dones.view(-1) > 0
            self._episode_returns += rew
            if torch.any(done):
                finished_returns = self._episode_returns[done].detach().cpu().tolist()
                self._episode_return_buffer.extend(finished_returns)
                self._episode_returns[done] = 0.0
                self._update_adasmpl_probability()
        super().process_env_step(obs, rewards, dones, extras)

    def update(self):
        self.num_updates += 1
        mean_behavior_loss = 0.0
        mean_recon_loss = 0.0
        mean_recon_rough_loss = 0.0
        mean_recon_refined_loss = 0.0
        cnt = 0

        for _ in range(self.num_learning_epochs):
            self.policy.reset(hidden_states=self.last_hidden_states)
            self.policy.detach_hidden_states()
            accumulated_loss = None
            accumulated_count = 0
            for batch in self.storage.generator():
                obs, privileged_actions, dones = self._distillation_batch(batch)
                actions = self.policy.act_inference(obs)
                behavior_loss = self.loss_fn(actions, privileged_actions)
                total_loss = behavior_loss

                if hasattr(self.policy, "compute_latest_auxiliary_losses"):
                    aux_losses = self.policy.compute_latest_auxiliary_losses()
                elif hasattr(self.policy, "compute_auxiliary_losses"):
                    aux_losses = self.policy.compute_auxiliary_losses(obs)
                else:
                    aux_losses = {}
                recon_loss = aux_losses.get("terrain_reconstruction", None)
                if recon_loss is not None:
                    total_loss = total_loss + self.reconstruction_loss_weight * recon_loss
                    mean_recon_loss += float(recon_loss.detach().item())
                rough_loss = aux_losses.get("terrain_reconstruction_rough", None)
                if rough_loss is not None:
                    mean_recon_rough_loss += float(rough_loss.detach().item())
                refined_loss = aux_losses.get("terrain_reconstruction_refined", None)
                if refined_loss is not None:
                    mean_recon_refined_loss += float(refined_loss.detach().item())

                accumulated_loss = total_loss if accumulated_loss is None else accumulated_loss + total_loss
                accumulated_count += 1
                mean_behavior_loss += float(behavior_loss.detach().item())
                cnt += 1

                if accumulated_count == self.gradient_length:
                    self._step_accumulated_loss(accumulated_loss)
                    accumulated_loss = None
                    accumulated_count = 0

                self.policy.reset(dones.view(-1))
                self.policy.detach_hidden_states(dones.view(-1))

            if accumulated_loss is not None:
                self._step_accumulated_loss(accumulated_loss)

        mean_behavior_loss /= max(cnt, 1)
        mean_recon_loss /= max(cnt, 1)
        mean_recon_rough_loss /= max(cnt, 1)
        mean_recon_refined_loss /= max(cnt, 1)

        self.storage.clear()
        self.last_hidden_states = self.policy.get_hidden_states()
        self.policy.detach_hidden_states()

        loss_dict = {
            "behavior": mean_behavior_loss,
            "terrain_reconstruction": mean_recon_loss,
            "terrain_reconstruction_rough": mean_recon_rough_loss,
            "terrain_reconstruction_refined": mean_recon_refined_loss,
        }
        if hasattr(self.policy, "get_adasmpl_probability"):
            loss_dict["adasmpl_probability"] = float(self.policy.get_adasmpl_probability())
        return loss_dict


class DepthBackboneRecurrentActor(nn.Module):
    """Actor that encodes depth with a CNN, fuses with proprioception, then applies a recurrent memory."""

    def __init__(
        self,
        total_obs_dim: int,
        proprio_slices: list[tuple[int, int]],
        depth_slice: tuple[int, int],
        depth_image_shape: tuple[int, int],
        num_actions: int,
        actor_hidden_dims: list[int],
        activation: str,
        depth_backbone_channels: list[int],
        depth_backbone_kernels: list[int],
        depth_backbone_pool_kernel: int,
        depth_backbone_image_fc_dim: int,
        depth_backbone_latent_dim: int,
        depth_recurrent_fusion_dim: int,
        rnn_type: str,
        rnn_hidden_dim: int,
        rnn_num_layers: int,
    ):
        super().__init__()

        self.total_obs_dim = int(total_obs_dim)
        self.proprio_slices = list(proprio_slices)
        self.depth_slice = (int(depth_slice[0]), int(depth_slice[1]))
        self.depth_h = int(depth_image_shape[0])
        self.depth_w = int(depth_image_shape[1])
        self.depth_dim = self.depth_slice[1] - self.depth_slice[0]

        if self.depth_h * self.depth_w != self.depth_dim:
            raise ValueError(
                f"Depth image shape {depth_image_shape} is incompatible with depth dimension {self.depth_dim}."
            )

        if len(depth_backbone_channels) != len(depth_backbone_kernels):
            raise ValueError(
                "depth_backbone_channels and depth_backbone_kernels must have the same length."
            )
        if len(depth_backbone_channels) < 1:
            raise ValueError("Depth backbone must contain at least one convolution layer.")

        def _act() -> nn.Module:
            return copy.deepcopy(resolve_nn_activation(activation))

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

        self.depth_cnn = nn.Sequential(*conv_layers)
        self.depth_flatten = nn.Flatten()
        flattened_dim = in_channels * cur_h * cur_w
        self.depth_projector = nn.Sequential(
            nn.Linear(flattened_dim, int(depth_backbone_image_fc_dim)),
            _act(),
            nn.Linear(int(depth_backbone_image_fc_dim), int(depth_backbone_latent_dim)),
            _act(),
        )

        proprio_dim = sum(end - start for start, end in self.proprio_slices)
        self.fusion = nn.Sequential(
            nn.Linear(proprio_dim + int(depth_backbone_latent_dim), int(depth_recurrent_fusion_dim)),
            _act(),
        )
        self.memory = Memory(
            input_size=int(depth_recurrent_fusion_dim),
            type=rnn_type,
            num_layers=int(rnn_num_layers),
            hidden_size=int(rnn_hidden_dim),
        )
        self.actor_mlp = MLP(
            input_dim=int(rnn_hidden_dim),
            output_dim=int(num_actions),
            hidden_dims=list(actor_hidden_dims),
            activation=activation,
        )

    def _split(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if obs.shape[-1] != self.total_obs_dim:
            raise ValueError(f"Expected obs dim {self.total_obs_dim}, got {obs.shape[-1]}")
        proprio_parts = [obs[:, start:end] for start, end in self.proprio_slices]
        proprio = torch.cat(proprio_parts, dim=-1) if proprio_parts else obs.new_zeros((obs.shape[0], 0))
        depth = obs[:, self.depth_slice[0] : self.depth_slice[1]]
        return proprio, depth

    def encode_depth(self, depth_flat: torch.Tensor) -> torch.Tensor:
        depth_image = depth_flat.view(-1, 1, self.depth_h, self.depth_w)
        depth_feat = self.depth_cnn(depth_image)
        depth_feat = self.depth_flatten(depth_feat)
        return self.depth_projector(depth_feat)

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        proprio, depth_flat = self._split(obs)
        depth_latent = self.encode_depth(depth_flat)
        fused = self.fusion(torch.cat((proprio, depth_latent), dim=-1))
        recurrent_out = self.memory(fused).squeeze(0)
        return self.actor_mlp(recurrent_out)

    def reset(self, dones=None, hidden_states=None):
        self.memory.reset(dones=dones, hidden_states=hidden_states)

    def detach_hidden_states(self, dones=None):
        self.memory.detach_hidden_states(dones=dones)

    def get_hidden_states(self):
        return self.memory.hidden_states


class StudentTeacherDepthBackboneRecurrent(nn.Module):
    """Student-teacher model with depth CNN backbone and recurrent student memory."""

    is_recurrent = True

    def __init__(
        self,
        obs,
        obs_groups,
        num_actions,
        student_obs_normalization=False,
        teacher_obs_normalization=False,
        student_hidden_dims=[256, 256, 256],
        teacher_hidden_dims=[256, 256, 256],
        activation="elu",
        init_noise_std=0.1,
        noise_std_type: str = "scalar",
        depth_obs_group: str = "depth_camera",
        depth_image_shape: tuple[int, int] = (58, 87),
        depth_backbone_channels: list[int] = [32, 64],
        depth_backbone_kernels: list[int] = [5, 3],
        depth_backbone_pool_kernel: int = 2,
        depth_backbone_image_fc_dim: int = 128,
        depth_backbone_latent_dim: int = 32,
        depth_recurrent_fusion_dim: int = 128,
        rnn_type: str = "gru",
        rnn_hidden_dim: int = 256,
        rnn_num_layers: int = 1,
        **kwargs,
    ):
        if kwargs:
            print(
                "StudentTeacherDepthBackboneRecurrent.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs.keys()])
            )
        super().__init__()

        self.loaded_teacher = False
        self.obs_groups = obs_groups
        self.depth_obs_group = depth_obs_group

        policy_groups = list(obs_groups["policy"])
        if depth_obs_group not in policy_groups:
            raise ValueError(
                f"Depth observation group '{depth_obs_group}' not found in policy obs groups: {policy_groups}"
            )

        offset = 0
        proprio_slices: list[tuple[int, int]] = []
        depth_slice: tuple[int, int] | None = None
        num_student_obs = 0
        for group_name in policy_groups:
            assert len(obs[group_name].shape) == 2, "StudentTeacherDepthBackboneRecurrent only supports 1D groups."
            dim = int(obs[group_name].shape[-1])
            start, end = offset, offset + dim
            if group_name == depth_obs_group:
                depth_slice = (start, end)
            else:
                proprio_slices.append((start, end))
            offset = end
            num_student_obs += dim
        if depth_slice is None:
            raise ValueError(f"Depth group '{depth_obs_group}' slice was not built.")

        num_teacher_obs = 0
        for obs_group in obs_groups["teacher"]:
            assert len(obs[obs_group].shape) == 2, "StudentTeacherDepthBackboneRecurrent only supports 1D groups."
            num_teacher_obs += int(obs[obs_group].shape[-1])

        self.student = DepthBackboneRecurrentActor(
            total_obs_dim=num_student_obs,
            proprio_slices=proprio_slices,
            depth_slice=depth_slice,
            depth_image_shape=depth_image_shape,
            num_actions=num_actions,
            actor_hidden_dims=student_hidden_dims,
            activation=activation,
            depth_backbone_channels=depth_backbone_channels,
            depth_backbone_kernels=depth_backbone_kernels,
            depth_backbone_pool_kernel=depth_backbone_pool_kernel,
            depth_backbone_image_fc_dim=depth_backbone_image_fc_dim,
            depth_backbone_latent_dim=depth_backbone_latent_dim,
            depth_recurrent_fusion_dim=depth_recurrent_fusion_dim,
            rnn_type=rnn_type,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
        )

        self.student_obs_normalization = student_obs_normalization
        if student_obs_normalization:
            self.student_obs_normalizer = EmpiricalNormalization(num_student_obs)
        else:
            self.student_obs_normalizer = torch.nn.Identity()

        self.teacher = MLP(num_teacher_obs, num_actions, teacher_hidden_dims, activation)
        self.teacher.eval()

        self.teacher_obs_normalization = teacher_obs_normalization
        if teacher_obs_normalization:
            self.teacher_obs_normalizer = EmpiricalNormalization(num_teacher_obs)
        else:
            self.teacher_obs_normalizer = torch.nn.Identity()

        print(f"Student Depth-Backbone Recurrent Actor: {self.student}")
        print(f"Teacher MLP: {self.teacher}")

        self.noise_std_type = noise_std_type
        if self.noise_std_type == "scalar":
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        elif self.noise_std_type == "log":
            self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

        self.distribution = None
        Normal.set_default_validate_args(False)

    def reset(self, dones=None, hidden_states=None):
        if hidden_states is None:
            student_hidden_states = None
        elif isinstance(hidden_states, tuple):
            student_hidden_states = hidden_states[0]
        else:
            student_hidden_states = hidden_states
        self.student.reset(dones=dones, hidden_states=student_hidden_states)

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

    def update_distribution(self, obs):
        mean = self.student(obs)
        if self.noise_std_type == "scalar":
            std = self.std.expand_as(mean)
        elif self.noise_std_type == "log":
            std = torch.exp(self.log_std).expand_as(mean)
        else:
            raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, obs):
        obs = self.get_student_obs(obs)
        obs = self.student_obs_normalizer(obs)
        with torch.no_grad():
            self.update_distribution(obs)
            return self.distribution.sample()

    def act_inference(self, obs):
        obs = self.get_student_obs(obs)
        obs = self.student_obs_normalizer(obs)
        return self.student(obs)

    def evaluate(self, obs):
        obs = self.get_teacher_obs(obs)
        obs = self.teacher_obs_normalizer(obs)
        with torch.no_grad():
            return self.teacher(obs)

    def get_student_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["policy"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_teacher_obs(self, obs):
        obs_list = []
        for obs_group in self.obs_groups["teacher"]:
            obs_list.append(obs[obs_group])
        return torch.cat(obs_list, dim=-1)

    def get_hidden_states(self):
        return self.student.get_hidden_states(), None

    def detach_hidden_states(self, dones=None):
        self.student.detach_hidden_states(dones=dones)

    def train(self, mode=True):
        super().train(mode)
        self.teacher.eval()
        self.teacher_obs_normalizer.eval()

    def update_normalization(self, obs):
        if self.student_obs_normalization:
            student_obs = self.get_student_obs(obs)
            self.student_obs_normalizer.update(student_obs)

    def load_state_dict(self, state_dict, strict=True):
        """Load parameters from teacher RL training or distillation training."""
        if any(key.startswith("actor.") or key.startswith("actor_obs_normalizer.") for key in state_dict.keys()):
            teacher_state_dict = {}
            teacher_obs_normalizer_state_dict = {}
            for key, value in state_dict.items():
                if "actor." in key:
                    teacher_state_dict[key.replace("actor.", "")] = value
                if "actor_obs_normalizer." in key:
                    teacher_obs_normalizer_state_dict[key.replace("actor_obs_normalizer.", "")] = value
            self.teacher.load_state_dict(teacher_state_dict, strict=strict)
            self.teacher_obs_normalizer.load_state_dict(teacher_obs_normalizer_state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return False
        elif any(key.startswith("student.") for key in state_dict.keys()):
            super().load_state_dict(state_dict, strict=strict)
            self.loaded_teacher = True
            self.teacher.eval()
            self.teacher_obs_normalizer.eval()
            return True
        else:
            raise ValueError("state_dict does not contain student or teacher parameters")
