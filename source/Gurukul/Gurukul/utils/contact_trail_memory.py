"""Egocentric contact-trail spatial memory for locomotion policies.

Coordinate convention (robot base frame, yaw-only map):
    - Origin at base xy projection; map is centered on the robot.
    - +x forward, +y left, +z up (Isaac Lab base convention).
    - Grid index (row i, col j) with cell center:
        x = (j - W/2 + 0.5) * resolution
        y = (i - H/2 + 0.5) * resolution

Warping sign convention:
    If the robot moves forward (+x), previously written contact content shifts
    backward (-x) in the egocentric map (content stays fixed in world frame).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

# Per-foot contact feature layout (11 dims).
CONTACT_FEATURE_NAMES = (
    "contact_flag",
    "normal_force",
    "tangential_force",
    "foot_vel_x_local",
    "foot_vel_y_local",
    "foot_vel_z_local",
    "foot_speed_tangent",
    "previous_action_leg_norm",
    "base_ang_vel_x",
    "base_ang_vel_y",
    "base_ang_vel_z",
)
CONTACT_FEATURE_DIM = len(CONTACT_FEATURE_NAMES)


@dataclass
class ContactTrailConfig:
    """Configuration for contact trail memory."""

    num_channels: int = 8
    grid_size: tuple[int, int] = (40, 40)
    resolution: float = 0.05
    decay: float = 0.985
    write_radius: int = 1
    learned_write: bool = True
    write_mode: str = "learned"  # "learned" | "engineered"
    use_warp: bool = True
    write_only_on_contact: bool = True
    contact_force_threshold: float = 1.0
    slip_velocity_scale: float = 0.5
    slip_k: float = 5.0
    impact_force_threshold: float = 100.0
    impact_force_scale: float = 200.0
    map_clamp: float = 5.0
    debug: bool = False
    # Normalization scales for WriteNet inputs.
    force_scale: float = 200.0
    velocity_scale: float = 2.0
    action_scale: float = 1.0
    ang_vel_scale: float = 4.0


def yaw_from_quat(quat_w: torch.Tensor) -> torch.Tensor:
    """Extract yaw (rotation about +z) from (w, x, y, z) quaternions."""
    w, x, y, z = quat_w.unbind(dim=-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def rotate_xy(points_xy: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Rotate 2D points by yaw (counter-clockwise about +z)."""
    c = torch.cos(yaw)
    s = torch.sin(yaw)
    target_ndim = points_xy.ndim - 1
    while c.ndim < target_ndim:
        c = c.unsqueeze(-1)
        s = s.unsqueeze(-1)
    x = points_xy[..., 0]
    y = points_xy[..., 1]
    return torch.stack((c * x - s * y, s * x + c * y), dim=-1)


class ContactTrailWriteNet(nn.Module):
    """Learned write embedding and gate from per-foot contact features."""

    def __init__(self, contact_feature_dim: int, num_channels: int, activation: str = "elu"):
        super().__init__()
        act = nn.ELU() if activation == "elu" else nn.ReLU()
        hidden = 64
        self.embed_mlp = nn.Sequential(
            nn.Linear(contact_feature_dim, hidden),
            act,
            nn.Linear(hidden, hidden),
            act,
            nn.Linear(hidden, num_channels),
        )
        self.gate_mlp = nn.Sequential(
            nn.Linear(contact_feature_dim, 32),
            act,
            nn.Linear(32, 1),
        )

    def forward(self, features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return event embedding [B, F, C] and gate alpha [B, F, 1]."""
        embedding = self.embed_mlp(features)
        alpha = torch.sigmoid(self.gate_mlp(features))
        return embedding, alpha


class ContactTrailEncoder(nn.Module):
    """CNN encoder for contact trail maps."""

    def __init__(
        self,
        num_channels: int,
        grid_size: tuple[int, int],
        latent_dim: int = 128,
        activation: str = "elu",
    ):
        super().__init__()
        act = nn.ELU() if activation == "elu" else nn.ReLU()
        self.grid_h, self.grid_w = int(grid_size[0]), int(grid_size[1])
        self.cnn = nn.Sequential(
            nn.Conv2d(num_channels, 16, kernel_size=3, stride=2, padding=1),
            act,
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            act,
            nn.Conv2d(32, 32, kernel_size=3, stride=2, padding=1),
            act,
        )
        with torch.no_grad():
            dummy = torch.zeros(1, num_channels, self.grid_h, self.grid_w)
            flat_dim = self.cnn(dummy).reshape(1, -1).shape[-1]
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(flat_dim, latent_dim),
            act,
        )
        self.latent_dim = int(latent_dim)

    def forward(self, trail_map: torch.Tensor) -> torch.Tensor:
        return self.head(self.cnn(trail_map))


class ContactQualityHead(nn.Module):
    """Optional auxiliary head predicting local contact quality."""

    def __init__(self, trail_latent_dim: int, grid_size: tuple[int, int]):
        super().__init__()
        self.grid_h, self.grid_w = int(grid_size[0]), int(grid_size[1])
        self.decoder = nn.Sequential(
            nn.Linear(trail_latent_dim, 256),
            nn.ELU(),
            nn.Linear(256, self.grid_h * self.grid_w),
        )

    def forward(self, trail_latent: torch.Tensor) -> torch.Tensor:
        quality = self.decoder(trail_latent)
        return quality.view(-1, 1, self.grid_h, self.grid_w)


class ContactTrailMemory(nn.Module):
    """Short-term egocentric contact trail map maintained online."""

    is_recurrent = False

    def __init__(
        self,
        num_envs: int,
        num_channels: int = 8,
        grid_size: tuple[int, int] = (40, 40),
        resolution: float = 0.05,
        decay: float = 0.985,
        write_radius: int = 1,
        device: torch.device | str = "cuda",
        learned_write: bool = True,
        contact_feature_dim: int = CONTACT_FEATURE_DIM,
        config: ContactTrailConfig | None = None,
        write_net: ContactTrailWriteNet | None = None,
    ):
        super().__init__()
        self.cfg = config or ContactTrailConfig(
            num_channels=num_channels,
            grid_size=grid_size,
            resolution=resolution,
            decay=decay,
            write_radius=write_radius,
            learned_write=learned_write,
        )
        self.num_envs = int(num_envs)
        self.num_channels = int(self.cfg.num_channels)
        self.grid_h, self.grid_w = int(self.cfg.grid_size[0]), int(self.cfg.grid_size[1])
        self.resolution = float(self.cfg.resolution)
        self.decay = float(self.cfg.decay)
        self.write_radius = int(self.cfg.write_radius)
        init_device = torch.device(device)
        self.contact_feature_dim = int(contact_feature_dim)

        self.write_net = write_net
        if self.cfg.write_mode == "learned" and self.write_net is None:
            self.write_net = ContactTrailWriteNet(self.contact_feature_dim, self.num_channels)

        self.register_buffer(
            "map",
            torch.zeros(self.num_envs, self.num_channels, self.grid_h, self.grid_w, device=init_device),
            persistent=False,
        )
        self.register_buffer("prev_base_pos_w", torch.zeros(self.num_envs, 3, device=init_device), persistent=False)
        self.register_buffer("prev_base_yaw", torch.zeros(self.num_envs, device=init_device), persistent=False)
        self.register_buffer("_initialized", torch.zeros(self.num_envs, dtype=torch.bool, device=init_device), persistent=False)

        # Precompute cell centers in base frame (+x forward, +y left).
        rows = torch.arange(self.grid_h, device=init_device, dtype=torch.float32)
        cols = torch.arange(self.grid_w, device=init_device, dtype=torch.float32)
        yy, xx = torch.meshgrid(rows, cols, indexing="ij")
        cell_x = (xx - self.grid_w / 2.0 + 0.5) * self.resolution
        cell_y = (yy - self.grid_h / 2.0 + 0.5) * self.resolution
        self.register_buffer("cell_centers_xy", torch.stack((cell_x, cell_y), dim=-1), persistent=False)

        self._build_write_kernel()

        # Debug stats (updated each step when debug=True).
        self.last_stats: dict[str, float] = {}

    def _build_write_kernel(self) -> None:
        radius = max(0, self.write_radius)
        size = 2 * radius + 1
        coords = torch.arange(-radius, radius + 1, device=self.map.device, dtype=torch.float32)
        yy, xx = torch.meshgrid(coords, coords, indexing="ij")
        dist2 = xx.square() + yy.square()
        if radius > 0:
            kernel = torch.exp(-dist2 / max(radius * radius, 1.0))
        else:
            kernel = torch.ones(1, 1, device=self.map.device)
        kernel = kernel / kernel.sum().clamp_min(1.0e-8)
        self.register_buffer("write_kernel", kernel.view(1, 1, size, size), persistent=False)
        self._write_kernel_radius = radius

    @property
    def device(self) -> torch.device:
        return self.map.device

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if env_ids is None:
            self.map.zero_()
            self.prev_base_pos_w.zero_()
            self.prev_base_yaw.zero_()
            self._initialized.zero_()
            return
        if len(env_ids) == 0:
            return
        self.map[env_ids] = 0.0
        self.prev_base_pos_w[env_ids] = 0.0
        self.prev_base_yaw[env_ids] = 0.0
        self._initialized[env_ids] = False

    def _ensure_num_envs(self, batch_size: int, device: torch.device) -> None:
        if batch_size == self.num_envs and self.map.device == device:
            return
        self.num_envs = int(batch_size)
        self.map = torch.zeros(
            self.num_envs, self.num_channels, self.grid_h, self.grid_w, device=device, dtype=self.map.dtype
        )
        self.prev_base_pos_w = torch.zeros(self.num_envs, 3, device=device, dtype=self.prev_base_pos_w.dtype)
        self.prev_base_yaw = torch.zeros(self.num_envs, device=device, dtype=self.prev_base_yaw.dtype)
        self._initialized = torch.zeros(self.num_envs, dtype=torch.bool, device=device)

    def get_map(self) -> torch.Tensor:
        return self.map

    @staticmethod
    def normalize_contact_features(features: torch.Tensor, cfg: ContactTrailConfig) -> torch.Tensor:
        """Normalize raw contact features before WriteNet."""
        norm = features.clone()
        norm[..., 1] = norm[..., 1] / cfg.force_scale
        norm[..., 2] = norm[..., 2] / cfg.force_scale
        norm[..., 3:7] = norm[..., 3:7] / cfg.velocity_scale
        norm[..., 7] = norm[..., 7] / cfg.action_scale
        norm[..., 8:11] = norm[..., 8:11] / cfg.ang_vel_scale
        return norm.clamp(-5.0, 5.0)

    def _warp_map(
        self,
        prev_map: torch.Tensor,
        base_pos_w: torch.Tensor,
        base_yaw: torch.Tensor,
        env_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self.cfg.use_warp:
            return prev_map

        delta_pos_w = base_pos_w - self.prev_base_pos_w
        rel_yaw = base_yaw - self.prev_base_yaw
        cos_rel = torch.cos(rel_yaw)
        sin_rel = torch.sin(rel_yaw)
        cos_prev = torch.cos(self.prev_base_yaw)
        sin_prev = torch.sin(self.prev_base_yaw)

        half_w_m = self.grid_w * self.resolution / 2.0
        half_h_m = self.grid_h * self.resolution / 2.0
        sx_over_sy = half_w_m / half_h_m
        sy_over_sx = half_h_m / half_w_m

        theta = prev_map.new_zeros((self.num_envs, 2, 3))
        theta[:, 0, 0] = cos_rel
        theta[:, 0, 1] = -sin_rel * sy_over_sx
        theta[:, 1, 0] = sin_rel * sx_over_sy
        theta[:, 1, 1] = cos_rel
        theta[:, 0, 2] = (cos_prev * delta_pos_w[:, 0] + sin_prev * delta_pos_w[:, 1]) / half_w_m
        theta[:, 1, 2] = (-sin_prev * delta_pos_w[:, 0] + cos_prev * delta_pos_w[:, 1]) / half_h_m

        grid = F.affine_grid(theta, prev_map.shape, align_corners=False)

        warped = F.grid_sample(
            prev_map,
            grid,
            mode="bilinear",
            padding_mode="zeros",
            align_corners=False,
        )
        if env_mask is not None:
            warped = torch.where(env_mask.view(-1, 1, 1, 1), warped, prev_map)
        return warped

    def _local_xy_to_grid(self, foot_pos_b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert foot xy in base frame to integer grid row/col indices."""
        x = foot_pos_b[..., 0]
        y = foot_pos_b[..., 1]
        col = torch.floor(x / self.resolution + self.grid_w / 2.0).long()
        row = torch.floor(y / self.resolution + self.grid_h / 2.0).long()
        return row, col

    def _engineered_embedding(self, features: torch.Tensor) -> torch.Tensor:
        """Engineered 5-channel write vector for sanity debugging."""
        contact = features[..., 0:1]
        tangent_speed = features[..., 6:7]
        normal_force = features[..., 1:2] * self.cfg.force_scale

        stable = contact * torch.exp(-self.cfg.slip_k * tangent_speed * self.cfg.velocity_scale)
        slip = contact * torch.clamp(
            tangent_speed * self.cfg.velocity_scale / max(self.cfg.slip_velocity_scale, 1.0e-6), 0.0, 1.0
        )
        impact = contact * torch.clamp(
            (normal_force - self.cfg.impact_force_threshold) / max(self.cfg.impact_force_scale, 1.0e-6), 0.0, 1.0
        )
        no_contact = 1.0 - contact

        emb = torch.zeros(*features.shape[:-1], self.num_channels, device=features.device, dtype=features.dtype)
        emb[..., 0] = contact.squeeze(-1)
        emb[..., 1] = stable.squeeze(-1)
        emb[..., 2] = slip.squeeze(-1)
        emb[..., 3] = impact.squeeze(-1)
        emb[..., 4] = no_contact.squeeze(-1)
        return emb

    def _write_foot_contacts(
        self,
        trail_map: torch.Tensor,
        foot_pos_b: torch.Tensor,
        features: torch.Tensor,
        embeddings: torch.Tensor,
        alphas: torch.Tensor,
    ) -> torch.Tensor:
        num_feet = foot_pos_b.shape[1]
        out_map = trail_map.clone()
        radius = self._write_kernel_radius

        for foot_idx in range(num_feet):
            row, col = self._local_xy_to_grid(foot_pos_b[:, foot_idx, :2])
            valid = (row >= 0) & (row < self.grid_h) & (col >= 0) & (col < self.grid_w)
            if self.cfg.write_only_on_contact:
                valid = valid & (features[:, foot_idx, 0] > 0.5)

            emb = embeddings[:, foot_idx]
            alpha = alphas[:, foot_idx].view(-1, 1, 1, 1)

            for env_idx in range(self.num_envs):
                if not valid[env_idx]:
                    continue
                r = int(row[env_idx].item())
                c = int(col[env_idx].item())
                r0 = max(0, r - radius)
                r1 = min(self.grid_h, r + radius + 1)
                c0 = max(0, c - radius)
                c1 = min(self.grid_w, c + radius + 1)

                kr0 = r0 - (r - radius)
                kr1 = kr0 + (r1 - r0)
                kc0 = c0 - (c - radius)
                kc1 = kc0 + (c1 - c0)
                kernel = self.write_kernel[:, :, kr0:kr1, kc0:kc1]

                patch = out_map[env_idx : env_idx + 1, :, r0:r1, c0:c1]
                write_val = emb[env_idx].view(1, -1, 1, 1)
                blended = (1.0 - alpha[env_idx]) * patch + alpha[env_idx] * write_val
                out_map[env_idx : env_idx + 1, :, r0:r1, c0:c1] = (
                    patch * (1.0 - kernel * alpha[env_idx]) + blended * (kernel * alpha[env_idx])
                )
        return out_map

    def _write_foot_contacts_vectorized(
        self,
        trail_map: torch.Tensor,
        foot_pos_b: torch.Tensor,
        features: torch.Tensor,
        embeddings: torch.Tensor,
        alphas: torch.Tensor,
    ) -> torch.Tensor:
        """Vectorized write over envs; loops only over feet and kernel offsets."""
        out_map = trail_map.clone()
        num_feet = foot_pos_b.shape[1]
        radius = self._write_kernel_radius

        for foot_idx in range(num_feet):
            row, col = self._local_xy_to_grid(foot_pos_b[:, foot_idx, :2])
            valid = (row >= 0) & (row < self.grid_h) & (col >= 0) & (col < self.grid_w)
            if self.cfg.write_only_on_contact:
                valid = valid & (features[:, foot_idx, 0] > 0.5)

            env_ids = valid.nonzero(as_tuple=False).flatten()
            if env_ids.numel() == 0:
                continue
            rows = row[env_ids]
            cols = col[env_ids]
            emb = embeddings[env_ids, foot_idx]
            alpha = alphas[env_ids, foot_idx]

            for dr in range(-radius, radius + 1):
                rr = rows + dr
                row_valid = (rr >= 0) & (rr < self.grid_h)
                kr = dr + radius
                for dc in range(-radius, radius + 1):
                    cc = cols + dc
                    offset_valid = row_valid & (cc >= 0) & (cc < self.grid_w)
                    kc = dc + radius
                    local_ids = offset_valid.nonzero(as_tuple=False).flatten()
                    if local_ids.numel() == 0:
                        continue
                    selected_envs = env_ids[local_ids]
                    selected_rows = rr[local_ids]
                    selected_cols = cc[local_ids]
                    weight = self.write_kernel[0, 0, kr, kc] * alpha[local_ids]
                    patch = out_map[selected_envs, :, selected_rows, selected_cols]
                    write_val = emb[local_ids]
                    out_map[selected_envs, :, selected_rows, selected_cols] = (
                        patch * (1.0 - weight) + write_val * weight
                    )
        return out_map

    def compute_quality_targets(self, features: torch.Tensor) -> torch.Tensor:
        """Compute scalar contact quality labels per foot in [-1, 1]."""
        contact = features[..., 0]
        tangent_speed = features[..., 6] * self.cfg.velocity_scale
        normal_force = features[..., 1] * self.cfg.force_scale

        stable = contact * torch.exp(-self.cfg.slip_k * tangent_speed)
        slip = contact * torch.clamp(tangent_speed / max(self.cfg.slip_velocity_scale, 1.0e-6), 0.0, 1.0)
        impact = torch.clamp(
            (normal_force - self.cfg.impact_force_threshold) / max(self.cfg.impact_force_scale, 1.0e-6), 0.0, 1.0
        )
        quality = stable - slip - 0.5 * impact
        return quality.clamp(-1.0, 1.0)

    def update(
        self,
        base_pos_w: torch.Tensor,
        base_quat_w: torch.Tensor,
        foot_pos_b: torch.Tensor,
        contact_features: torch.Tensor,
        dt: float,
        env_ids: torch.Tensor | None = None,
        foot_pos_w: torch.Tensor | None = None,
        foot_vel_w: torch.Tensor | None = None,
        contact_forces_w: torch.Tensor | None = None,
        joint_pos: torch.Tensor | None = None,
        joint_vel: torch.Tensor | None = None,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Update map and return current trail tensor [num_envs, C, H, W]."""
        del dt, foot_pos_w, foot_vel_w, contact_forces_w, joint_pos, joint_vel, actions  # API compatibility.

        device = base_pos_w.device
        self._ensure_num_envs(base_pos_w.shape[0], device)

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=device)

        base_yaw = yaw_from_quat(base_quat_w)
        init_mask = ~self._initialized[env_ids]
        if torch.any(init_mask):
            init_envs = env_ids[init_mask]
            self.prev_base_pos_w[init_envs] = base_pos_w[init_envs].detach()
            self.prev_base_yaw[init_envs] = base_yaw[init_envs].detach()
            self._initialized[init_envs] = True

        prev_map = self.map.detach()
        warped = self._warp_map(prev_map, base_pos_w, base_yaw)
        trail_map = warped * self.decay

        num_feet = foot_pos_b.shape[1]
        features = contact_features.view(self.num_envs, num_feet, self.contact_feature_dim)
        norm_features = self.normalize_contact_features(features, self.cfg)

        if self.cfg.write_mode == "engineered":
            embeddings = self._engineered_embedding(norm_features)
            alphas = features[..., 0:1].clamp(0.0, 1.0)
        else:
            if self.write_net is None:
                raise RuntimeError("write_mode='learned' requires a ContactTrailWriteNet instance.")
            embeddings, alphas = self.write_net(norm_features)

        trail_map = self._write_foot_contacts_vectorized(trail_map, foot_pos_b, features, embeddings, alphas)

        if self.cfg.map_clamp > 0.0:
            trail_map = torch.clamp(trail_map, -self.cfg.map_clamp, self.cfg.map_clamp)

        if self.cfg.debug:
            if torch.isnan(trail_map).any():
                raise RuntimeError("ContactTrailMemory produced NaN map values.")
            self.last_stats = {
                "map_abs_mean": trail_map.abs().mean().item(),
                "map_nonzero_frac": (trail_map.abs() > 1.0e-5).float().mean().item(),
                "write_alpha_mean": alphas.mean().item(),
                "quality_mean": self.compute_quality_targets(features).mean().item(),
            }

        self.map = trail_map.detach()
        self.prev_base_pos_w = base_pos_w.detach()
        self.prev_base_yaw = base_yaw.detach()
        return trail_map

    def update_sequence(
        self,
        base_pos_w: torch.Tensor,
        base_quat_w: torch.Tensor,
        foot_pos_b: torch.Tensor,
        contact_features: torch.Tensor,
        masks: torch.Tensor,
        dt: float = 0.0,
    ) -> torch.Tensor:
        """Replay a padded trajectory without changing the live rollout memory."""
        if base_pos_w.ndim != 3:
            return self.update(
                base_pos_w=base_pos_w,
                base_quat_w=base_quat_w,
                foot_pos_b=foot_pos_b,
                contact_features=contact_features,
                dt=dt,
            )

        time_steps, batch_size = base_pos_w.shape[:2]
        device = base_pos_w.device

        # PPO can replay several differently-sized trajectory batches while the
        # simulator's rollout state must survive untouched.  Swap in scratch
        # buffers rather than cloning the (potentially very large) live map.
        live_state = (
            self.num_envs,
            self.map,
            self.prev_base_pos_w,
            self.prev_base_yaw,
            self._initialized,
        )
        self.num_envs = -1  # Force scratch allocation even when B == num_envs.
        try:
            self._ensure_num_envs(batch_size, device)
            self.reset()

            if masks.ndim == 3:
                mask_b = masks.squeeze(-1)
            else:
                mask_b = masks

            last_map = self.get_map()
            map_history: list[torch.Tensor] = []
            for step_idx in range(time_steps):
                active = mask_b[step_idx].bool()
                if step_idx == 0:
                    episode_start = active
                else:
                    episode_start = active & (~mask_b[step_idx - 1].bool())
                if torch.any(episode_start):
                    self.reset(episode_start.nonzero(as_tuple=False).flatten())

                if not torch.any(active):
                    map_history.append(last_map)
                    continue

                inactive_ids = (~active).nonzero(as_tuple=False).flatten()
                if inactive_ids.numel() > 0:
                    inactive_map = self.map[inactive_ids].clone()
                    inactive_prev_pos = self.prev_base_pos_w[inactive_ids].clone()
                    inactive_prev_yaw = self.prev_base_yaw[inactive_ids].clone()
                    inactive_initialized = self._initialized[inactive_ids].clone()

                last_map = self.update(
                    base_pos_w=base_pos_w[step_idx],
                    base_quat_w=base_quat_w[step_idx],
                    foot_pos_b=foot_pos_b[step_idx],
                    contact_features=contact_features[step_idx],
                    dt=dt,
                )
                if inactive_ids.numel() > 0:
                    last_map = last_map.clone()
                    last_map[inactive_ids] = inactive_map
                    self.map[inactive_ids] = inactive_map
                    self.prev_base_pos_w[inactive_ids] = inactive_prev_pos
                    self.prev_base_yaw[inactive_ids] = inactive_prev_yaw
                    self._initialized[inactive_ids] = inactive_initialized
                map_history.append(last_map)
            return torch.stack(map_history, dim=0)
        finally:
            (
                self.num_envs,
                self.map,
                self.prev_base_pos_w,
                self.prev_base_yaw,
                self._initialized,
            ) = live_state
