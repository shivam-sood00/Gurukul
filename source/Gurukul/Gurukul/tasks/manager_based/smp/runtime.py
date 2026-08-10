# SPDX-License-Identifier: Apache-2.0

"""Isaac Lab runtime adapter for Score-Matching Motion Priors (SMP).

The diffusion model is deliberately kept outside the PPO optimization. A
startup event installs a frozen, morphology-checked prior on the environment,
a reset event optionally performs generative state initialization (GSI), and a
regular manager reward term evaluates the rolling kinematic window. A thin
RSL-RL runner subclass persists only the adaptive score normalizer.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import torch

from .diffusion import LoadedSmpCheckpoint, load_smp_checkpoint
from .features import canonicalize_motion_features, rotation_6d_to_quaternion
from .profiles import DEFAULT_CONTROL_FPS, DEFAULT_WINDOW_SIZE, SmpRobotProfile, get_profile

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_STATE_ATTRIBUTE = "_smp_runtime_state"


@dataclass
class _KinematicHistory:
    """Rolling world-frame kinematics used to construct canonical SMP features."""

    root_pos: torch.Tensor
    root_quat: torch.Tensor
    joint_pos: torch.Tensor
    key_body_pos: torch.Tensor
    root_lin_vel: torch.Tensor
    root_ang_vel: torch.Tensor

    @property
    def window_size(self) -> int:
        return int(self.root_pos.shape[1])

    def append(
        self,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        joint_pos: torch.Tensor,
        key_body_pos: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
    ) -> None:
        for history, current in (
            (self.root_pos, root_pos),
            (self.root_quat, root_quat),
            (self.joint_pos, joint_pos),
            (self.key_body_pos, key_body_pos),
            (self.root_lin_vel, root_lin_vel),
            (self.root_ang_vel, root_ang_vel),
        ):
            history[:, :-1].copy_(history[:, 1:].clone())
            history[:, -1].copy_(current)

    def fill(
        self,
        env_ids: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        joint_pos: torch.Tensor,
        key_body_pos: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
    ) -> None:
        """Fill selected histories from either one frame or a complete window."""
        if env_ids.numel() == 0:
            return
        for history, value in (
            (self.root_pos, root_pos),
            (self.root_quat, root_quat),
            (self.joint_pos, joint_pos),
            (self.key_body_pos, key_body_pos),
            (self.root_lin_vel, root_lin_vel),
            (self.root_ang_vel, root_ang_vel),
        ):
            if value.ndim == history.ndim - 1:
                value = value[:, None].expand(-1, self.window_size, *value.shape[1:])
            history[env_ids] = value

    def features(self) -> torch.Tensor:
        return canonicalize_motion_features(
            self.root_pos,
            self.root_quat,
            self.joint_pos,
            self.key_body_pos,
            self.root_lin_vel,
            self.root_ang_vel,
        )


@dataclass
class _SmpRuntimeState:
    checkpoint: LoadedSmpCheckpoint
    checkpoint_sha256: str
    profile: SmpRobotProfile
    joint_ids: torch.Tensor
    key_body_ids: torch.Tensor
    root_body_id: int
    history: _KinematicHistory
    fixed_timesteps: tuple[int, ...]
    loss_scale: float
    score_batch_size: int
    update_score_normalizer: bool
    gsi_pool: torch.Tensor | None
    needs_history_prime: torch.Tensor


def _as_index_tensor(ids: Any, device: torch.device | str) -> torch.Tensor:
    return torch.as_tensor(ids, dtype=torch.long, device=device)


def _resolve_named_indices(robot, profile: SmpRobotProfile, device: torch.device | str):
    joint_ids, joint_names = robot.find_joints(list(profile.joint_names), preserve_order=True)
    if tuple(joint_names) != tuple(profile.joint_names):
        raise RuntimeError(
            f"SMP profile '{profile.name}' expects joints {profile.joint_names}, but the robot resolved "
            f"{tuple(joint_names)}. The prior and simulated morphology must use the same ordered joints."
        )

    root_ids, root_names = robot.find_bodies([profile.root_body_name], preserve_order=True)
    if tuple(root_names) != (profile.root_body_name,):
        raise RuntimeError(
            f"SMP profile '{profile.name}' expects root body '{profile.root_body_name}', but the robot resolved "
            f"{tuple(root_names)}."
        )
    root_body_id = int(root_ids[0])
    if root_body_id != 0:
        # Some assets retain a massless dummy articulation root. G1's `base`
        # and `pelvis`, for example, are joined by a fixed identity transform.
        # Scoring can use the named motion root, and GSI can write the
        # articulation root, only when those frames are coincident.
        position_error = torch.max(torch.abs(robot.data.body_link_pos_w[:, root_body_id] - robot.data.root_link_pos_w))
        orientation_alignment = torch.abs(
            torch.sum(robot.data.body_link_quat_w[:, root_body_id] * robot.data.root_link_quat_w, dim=-1)
        )
        if position_error > 1.0e-5 or torch.any(orientation_alignment < 1.0 - 1.0e-5):
            raise RuntimeError(
                f"SMP motion root '{profile.root_body_name}' is not coincident with the articulation root. "
                "This adapter cannot safely translate generated root states for that asset."
            )

    key_body_ids, key_body_names = robot.find_bodies(list(profile.key_body_names), preserve_order=True)
    if tuple(key_body_names) != tuple(profile.key_body_names):
        raise RuntimeError(
            f"SMP profile '{profile.name}' expects key bodies {profile.key_body_names}, but the robot resolved "
            f"{tuple(key_body_names)}."
        )
    return (
        _as_index_tensor(joint_ids, device),
        _as_index_tensor(key_body_ids, device),
        root_body_id,
    )


def _allocate_history(
    *,
    num_envs: int,
    window_size: int,
    profile: SmpRobotProfile,
    device: torch.device | str,
) -> _KinematicHistory:
    zeros = torch.zeros
    root_quat = zeros(num_envs, window_size, 4, device=device)
    root_quat[..., 0] = 1.0
    return _KinematicHistory(
        root_pos=zeros(num_envs, window_size, 3, device=device),
        root_quat=root_quat,
        joint_pos=zeros(num_envs, window_size, len(profile.joint_names), device=device),
        key_body_pos=zeros(num_envs, window_size, len(profile.key_body_names), 3, device=device),
        root_lin_vel=zeros(num_envs, window_size, 3, device=device),
        root_ang_vel=zeros(num_envs, window_size, 3, device=device),
    )


def _read_kinematics(env: ManagerBasedRLEnv, state: _SmpRuntimeState, env_ids: torch.Tensor | None = None):
    robot = env.scene["robot"]
    select = slice(None) if env_ids is None else env_ids
    origins = env.scene.env_origins[select]
    return (
        robot.data.body_link_pos_w[select, state.root_body_id] - origins,
        robot.data.body_link_quat_w[select, state.root_body_id],
        robot.data.joint_pos[select][:, state.joint_ids],
        robot.data.body_link_pos_w[select][:, state.key_body_ids] - origins[:, None, :],
        robot.data.body_link_lin_vel_w[select, state.root_body_id],
        robot.data.body_link_ang_vel_w[select, state.root_body_id],
    )


def _fill_history_from_sim(
    env: ManagerBasedRLEnv,
    state: _SmpRuntimeState,
    env_ids: torch.Tensor,
) -> None:
    state.history.fill(env_ids, *_read_kinematics(env, state, env_ids))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


@torch.no_grad()
def initialize_smp(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor | None,
    checkpoint_path: str = "",
    profile_name: str = "",
    gsi_pool_size: int = 0,
    gsi_batch_size: int = 256,
    fixed_timesteps: tuple[int, ...] = (8, 15, 22),
    sds_loss_scale: float = 6.0,
    score_batch_size: int = 1024,
    update_score_normalizer: bool = True,
) -> None:
    """Load a frozen SMP prior and initialize per-environment feature history."""
    del env_ids
    if not checkpoint_path:
        raise RuntimeError(
            "This SMP task has no prior checkpoint configured. Pretrain a morphology-specific prior, then pass "
            "its path to RSL-RL with `--smp-prior /path/to/smp_prior.pt`."
        )
    checkpoint_file = Path(checkpoint_path).expanduser()
    if not checkpoint_file.is_file():
        raise FileNotFoundError(f"SMP prior checkpoint does not exist or is not a file: {checkpoint_file}")
    if gsi_pool_size < 0:
        raise ValueError(f"gsi_pool_size must be non-negative; got {gsi_pool_size}.")
    if gsi_pool_size > 0 and gsi_batch_size <= 0:
        raise ValueError(f"gsi_batch_size must be positive when GSI is enabled; got {gsi_batch_size}.")
    if score_batch_size <= 0:
        raise ValueError(f"score_batch_size must be positive; got {score_batch_size}.")
    if type(update_score_normalizer) is not bool:
        raise TypeError("update_score_normalizer must be a boolean.")
    if tuple(fixed_timesteps) != (8, 15, 22):
        raise ValueError("Canonical SMP runtime scoring requires fixed_timesteps=(8, 15, 22).")
    if isinstance(sds_loss_scale, bool) or not isinstance(sds_loss_scale, (int, float)):
        raise TypeError("sds_loss_scale must be a real number.")
    if not math.isfinite(float(sds_loss_scale)) or sds_loss_scale < 0.0:
        raise ValueError("sds_loss_scale must be finite and non-negative.")

    profile = get_profile(profile_name)
    step_dt = float(env.step_dt)
    control_fps = 1.0 / step_dt
    if abs(control_fps - DEFAULT_CONTROL_FPS) > 1.0e-3:
        raise RuntimeError(
            f"SMP profile '{profile.name}' requires {DEFAULT_CONTROL_FPS:g} Hz policy steps, but this "
            f"environment runs at {control_fps:g} Hz (step_dt={step_dt:g})."
        )
    checkpoint = load_smp_checkpoint(
        checkpoint_file,
        expected_profile=profile,
        expected_control_fps=control_fps,
        device=env.device,
    )
    checkpoint.prior.eval()
    checkpoint.prior.requires_grad_(False)
    if checkpoint.window_size != DEFAULT_WINDOW_SIZE:
        raise RuntimeError(
            f"Canonical SMP tasks require H={DEFAULT_WINDOW_SIZE}, but the checkpoint uses H={checkpoint.window_size}."
        )
    if checkpoint.diffusion_config.T != 50:
        raise RuntimeError(
            f"Canonical SMP tasks require 50 diffusion steps, but the checkpoint uses "
            f"T={checkpoint.diffusion_config.T}."
        )

    joint_ids, key_body_ids, root_body_id = _resolve_named_indices(env.scene["robot"], profile, env.device)
    history = _allocate_history(
        num_envs=env.num_envs,
        window_size=checkpoint.window_size,
        profile=profile,
        device=env.device,
    )
    state = _SmpRuntimeState(
        checkpoint=checkpoint,
        checkpoint_sha256=_sha256_file(checkpoint.path),
        profile=profile,
        joint_ids=joint_ids,
        key_body_ids=key_body_ids,
        root_body_id=root_body_id,
        history=history,
        fixed_timesteps=tuple(fixed_timesteps),
        loss_scale=float(sds_loss_scale),
        score_batch_size=int(score_batch_size),
        update_score_normalizer=update_score_normalizer,
        gsi_pool=None,
        needs_history_prime=torch.zeros(env.num_envs, dtype=torch.bool, device=env.device),
    )
    setattr(env, _STATE_ATTRIBUTE, state)

    all_env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    _fill_history_from_sim(env, state, all_env_ids)
    if gsi_pool_size > 0:
        chunks = []
        for start in range(0, gsi_pool_size, gsi_batch_size):
            chunks.append(
                checkpoint.prior.sample(
                    batch_size=min(gsi_batch_size, gsi_pool_size - start),
                    device=env.device,
                )
            )
        state.gsi_pool = torch.cat(chunks, dim=0)


def _state(env: ManagerBasedRLEnv) -> _SmpRuntimeState:
    state = getattr(env, _STATE_ATTRIBUTE, None)
    if state is None:
        raise RuntimeError(
            "SMP runtime state is unavailable. Ensure the `initialize_smp` startup event is enabled and its "
            "checkpoint loaded successfully."
        )
    return state


def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _quat_apply(quat: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    xyz = quat[..., 1:]
    uv = torch.cross(xyz, vector, dim=-1)
    return vector + 2.0 * (quat[..., :1] * uv + torch.cross(xyz, uv, dim=-1))


def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    result = quat.clone()
    result[..., 1:] *= -1.0
    return result


def _yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    w, x, y, z = quat.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y.square() + z.square()))
    half = 0.5 * yaw
    result = torch.zeros_like(quat)
    result[..., 0] = torch.cos(half)
    result[..., 3] = torch.sin(half)
    return result


def _normalize_quat(quat: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
    normalized = quat / norm.clamp_min(torch.finfo(quat.dtype).eps)
    identity = torch.zeros_like(quat)
    identity[..., 0] = 1.0
    return torch.where(norm > 1.0e-8, normalized, identity)


def _rotate_window(yaw: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    expanded = yaw
    while expanded.ndim < values.ndim:
        expanded = expanded.unsqueeze(1)
    expanded = expanded.expand(*values.shape[:-1], 4)
    return _quat_apply(expanded.reshape(-1, 4), values.reshape(-1, 3)).reshape_as(values)


def _finite_difference_root_motion(
    root_pos: torch.Tensor,
    root_quat: torch.Tensor,
    step_dt: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Derive centered world-frame link velocities from decoded poses."""
    frame_count = root_pos.shape[1]
    frame_ids = torch.arange(frame_count, device=root_pos.device)
    left = (frame_ids - 1).clamp_min(0)
    right = (frame_ids + 1).clamp_max(frame_count - 1)
    elapsed = (right - left).to(root_pos.dtype) * step_dt
    linear = (root_pos[:, right] - root_pos[:, left]) / elapsed[None, :, None]

    delta = _normalize_quat(
        _quat_mul(
            root_quat[:, right].reshape(-1, 4),
            _quat_conjugate(root_quat[:, left].reshape(-1, 4)),
        )
    ).reshape(root_quat.shape[0], frame_count, 4)
    delta = torch.where(delta[..., :1] < 0.0, -delta, delta)
    sin_half = torch.linalg.vector_norm(delta[..., 1:], dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(sin_half, delta[..., :1].clamp_min(0.0))
    scale = torch.where(sin_half > 1.0e-7, angle / sin_half, torch.full_like(sin_half, 2.0))
    angular = delta[..., 1:] * scale / elapsed[None, :, None]
    return linear, angular


def _decode_generated_window(
    env: ManagerBasedRLEnv,
    state: _SmpRuntimeState,
    env_ids: torch.Tensor,
    windows: torch.Tensor,
) -> None:
    """Write the final generated frame to simulation and prime its full history."""
    if not torch.isfinite(windows).all():
        raise RuntimeError("The SMP prior generated non-finite GSI features; refusing to write them to simulation.")
    num_joints = len(state.profile.joint_names)
    num_key_bodies = len(state.profile.key_body_names)
    root_pos_local = windows[..., 0:3]
    root_rot_6d = windows[..., 3:9]
    joint_pos = windows[..., 9 : 9 + num_joints]
    key_start = 9 + num_joints
    key_body_offset = windows[..., key_start : key_start + 3 * num_key_bodies].reshape(
        windows.shape[0], windows.shape[1], num_key_bodies, 3
    )
    robot = env.scene["robot"]
    origins = env.scene.env_origins[env_ids]
    current_root_pos = robot.data.body_link_pos_w[env_ids, state.root_body_id] - origins
    current_root_quat = _normalize_quat(robot.data.body_link_quat_w[env_ids, state.root_body_id])
    base_xy = current_root_pos[:, None, 0:2]
    anchor_yaw = _yaw_quat(current_root_quat)

    local_quat = _normalize_quat(rotation_6d_to_quaternion(root_rot_6d.reshape(-1, 6))).reshape(
        windows.shape[0], windows.shape[1], 4
    )
    # Canonical windows should end at zero xy and zero yaw. Re-anchor those
    # quantities explicitly so small diffusion reconstruction error cannot
    # move or turn the post-reset base pose unexpectedly.
    generated_final_yaw_inv = _quat_conjugate(_yaw_quat(local_quat[:, -1]))
    frame_yaw = _normalize_quat(_quat_mul(anchor_yaw, generated_final_yaw_inv))
    root_xy_offset = root_pos_local.clone()
    root_xy_offset[..., 2] = 0.0
    root_xy_offset[..., 0:2] -= root_xy_offset[:, -1:, 0:2]
    root_pos = _rotate_window(frame_yaw, root_xy_offset)
    root_pos[..., 0:2] += base_xy
    root_pos[..., 2] = root_pos_local[..., 2]

    yaw_window = frame_yaw[:, None, :].expand_as(local_quat)
    root_quat = _normalize_quat(_quat_mul(yaw_window.reshape(-1, 4), local_quat.reshape(-1, 4))).reshape_as(local_quat)
    root_lin_vel, root_ang_vel = _finite_difference_root_motion(
        root_pos,
        root_quat,
        float(env.step_dt),
    )
    key_body_pos = root_pos[:, :, None, :] + _rotate_window(frame_yaw, key_body_offset)

    limits = robot.data.soft_joint_pos_limits[env_ids][:, state.joint_ids]
    joint_pos = joint_pos.clamp(limits[:, None, :, 0], limits[:, None, :, 1])
    selected_joint_pos = joint_pos[:, -1]
    if windows.shape[1] > 1:
        selected_joint_vel = (joint_pos[:, -1] - joint_pos[:, -2]) / float(env.step_dt)
    else:
        selected_joint_vel = torch.zeros_like(selected_joint_pos)
    velocity_limits = getattr(robot.data, "soft_joint_vel_limits", None)
    if velocity_limits is None:
        velocity_limits = getattr(robot.data, "joint_vel_limits", None)
    if velocity_limits is not None:
        if velocity_limits.ndim == 1:
            selected_velocity_limits = velocity_limits[state.joint_ids]
        else:
            selected_velocity_limits = velocity_limits[env_ids][:, state.joint_ids]
        selected_joint_vel = selected_joint_vel.clamp(
            -selected_velocity_limits.abs(),
            selected_velocity_limits.abs(),
        )

    full_joint_pos = robot.data.joint_pos[env_ids].clone()
    full_joint_vel = robot.data.joint_vel[env_ids].clone()
    full_joint_pos[:, state.joint_ids] = selected_joint_pos
    full_joint_vel[:, state.joint_ids] = selected_joint_vel
    root_state = torch.cat(
        (
            root_pos[:, -1] + origins,
            root_quat[:, -1],
            root_lin_vel[:, -1],
            root_ang_vel[:, -1],
        ),
        dim=-1,
    )
    if not torch.isfinite(root_state).all() or not torch.isfinite(selected_joint_vel).all():
        raise RuntimeError("Decoded SMP GSI state is non-finite; refusing to write it to simulation.")
    robot.write_root_link_state_to_sim(root_state, env_ids=env_ids)
    robot.write_joint_state_to_sim(full_joint_pos, full_joint_vel, env_ids=env_ids)

    # Preserve the generated lead-in for the first post-reset score.  Use the
    # clamped last pose in both the simulator and history.
    state.history.fill(
        env_ids,
        root_pos,
        root_quat,
        joint_pos,
        key_body_pos,
        root_lin_vel,
        root_ang_vel,
    )


@torch.no_grad()
def reset_smp_state(env: ManagerBasedRLEnv, env_ids: torch.Tensor | slice | None) -> None:
    """Reset rolling history, optionally using a generated prior window (GSI)."""
    state = _state(env)
    all_env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)
    if env_ids is None:
        env_ids = all_env_ids
    elif isinstance(env_ids, slice):
        env_ids = all_env_ids[env_ids]
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device)
        if env_ids.dtype == torch.bool:
            raise TypeError("SMP reset env_ids must contain integer indices, not booleans.")
        if env_ids.dtype.is_floating_point or env_ids.dtype.is_complex:
            raise TypeError("SMP reset env_ids must contain integer indices.")
        env_ids = env_ids.to(dtype=torch.long).flatten()
    if env_ids.numel() == 0:
        return
    if torch.any((env_ids < 0) | (env_ids >= env.num_envs)):
        raise IndexError(f"SMP reset env_ids must be in [0, {env.num_envs}).")
    if torch.unique(env_ids).numel() != env_ids.numel():
        raise ValueError("SMP reset env_ids must not contain duplicates.")
    if state.gsi_pool is None:
        # Articulation link transforms are not guaranteed to be refreshed until
        # the environment's post-reset forward pass. Prime at the next reward
        # evaluation so no pre-reset key-body poses leak into the new episode.
        state.needs_history_prime[env_ids] = True
        return
    sample_ids = torch.randint(state.gsi_pool.shape[0], (env_ids.numel(),), device=env.device)
    _decode_generated_window(env, state, env_ids, state.gsi_pool[sample_ids])
    state.needs_history_prime[env_ids] = False


def _score_prior(state: _SmpRuntimeState, windows: torch.Tensor) -> torch.Tensor:
    loss_chunks = []
    for start in range(0, windows.shape[0], state.score_batch_size):
        batch = windows[start : start + state.score_batch_size]
        loss_chunks.append(
            state.checkpoint.prior.sds_losses(
                batch,
                timesteps=state.fixed_timesteps,
            )
        )
    losses = torch.cat(loss_chunks, dim=0)
    return state.checkpoint.prior.score_from_losses(
        losses,
        timesteps=state.fixed_timesteps,
        loss_scale=state.loss_scale,
        update_normalizer=state.update_score_normalizer,
        defer_normalizer_update=state.update_score_normalizer,
    )


@torch.no_grad()
def smp_guidance_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the frozen prior's score for each environment's recent motion."""
    state = _state(env)
    kinematics = _read_kinematics(env, state)
    env_ids = state.needs_history_prime.nonzero().flatten()
    if env_ids.numel() > 0:
        state.history.fill(env_ids, *(value[env_ids] for value in kinematics))
        state.needs_history_prime[env_ids] = False
    state.history.append(*kinematics)
    reward = _score_prior(state, state.history.features())
    if reward.shape != (env.num_envs,):
        raise RuntimeError(f"SMP prior returned reward shape {tuple(reward.shape)}; expected ({env.num_envs},).")
    if not torch.isfinite(reward).all():
        raise RuntimeError("SMP prior returned a non-finite reward; refusing to pass it to PPO.")
    if torch.any((reward < 0.0) | (reward > 1.0)):
        raise RuntimeError("SMP prior reward must remain in [0, 1].")
    return reward


__all__ = ["initialize_smp", "reset_smp_state", "smp_guidance_reward"]
