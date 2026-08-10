"""MDP terms for language-conditioned interaction tasks."""

from __future__ import annotations

import torch

from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers import SceneEntityCfg


def switch_state(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("switch", joint_names=["button_joint"]),
    travel: float = 0.025,
    pressed_threshold: float = 0.020,
) -> torch.Tensor:
    """Return normalized button travel and a binary pressed flag."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    normalized_depth = torch.clamp(joint_pos / float(travel), min=0.0, max=1.0)
    pressed = torch.any(joint_pos >= float(pressed_threshold), dim=1, keepdim=True)
    return torch.cat((normalized_depth, pressed.to(joint_pos.dtype)), dim=1)


def switch_press_progress(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("switch", joint_names=["button_joint"]),
    pressed_threshold: float = 0.020,
) -> torch.Tensor:
    """Measure button travel toward the pressed threshold in ``[0, 1]``."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.clamp(joint_pos / float(pressed_threshold), min=0.0, max=1.0).amax(dim=1)


def switch_pressed(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("switch", joint_names=["button_joint"]),
    pressed_threshold: float = 0.020,
) -> torch.Tensor:
    """Terminate successfully once the button crosses the pressed threshold."""
    asset: Articulation = env.scene[asset_cfg.name]
    joint_pos = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.any(joint_pos >= float(pressed_threshold), dim=1)


def hinged_door_state(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("door", joint_names=["door_hinge"]),
    open_threshold_rad: float = 1.0471975512,
) -> torch.Tensor:
    """Return door angle, normalized opening progress, and an opened flag."""
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids]
    progress = torch.clamp(angle / float(open_threshold_rad), min=0.0, max=1.0)
    opened = torch.any(angle >= float(open_threshold_rad), dim=1, keepdim=True)
    return torch.cat((angle, progress, opened.to(angle.dtype)), dim=1)


def hinged_door_open_progress(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("door", joint_names=["door_hinge"]),
    open_threshold_rad: float = 1.0471975512,
) -> torch.Tensor:
    """Measure revolute-door progress toward the opening threshold."""
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.clamp(angle / float(open_threshold_rad), min=0.0, max=1.0).amax(dim=1)


def hinged_door_opened(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("door", joint_names=["door_hinge"]),
    open_threshold_rad: float = 1.0471975512,
) -> torch.Tensor:
    """Return true once the physical hinge reaches the requested angle."""
    asset: Articulation = env.scene[asset_cfg.name]
    angle = asset.data.joint_pos[:, asset_cfg.joint_ids]
    return torch.any(angle >= float(open_threshold_rad), dim=1)


def rigid_object_service_state(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Return world pose and velocity for a service-task rigid object."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.cat(
        (
            asset.data.root_pos_w,
            asset.data.root_quat_w,
            asset.data.root_lin_vel_w,
            asset.data.root_ang_vel_w,
        ),
        dim=1,
    )


def object_lift_progress(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    initial_height_m: float = 0.625,
    required_lift_m: float = 0.10,
) -> torch.Tensor:
    """Measure object-center lift progress in ``[0, 1]``."""
    asset: RigidObject = env.scene[asset_cfg.name]
    return torch.clamp(
        (asset.data.root_pos_w[:, 2] - float(initial_height_m)) / float(required_lift_m),
        min=0.0,
        max=1.0,
    )


def object_settled_in_region(
    env: ManagerBasedEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    center_m: tuple[float, float, float] = (0.53, 0.15, 0.64),
    half_size_m: tuple[float, float, float] = (0.08, 0.08, 0.08),
    max_speed_mps: float = 0.20,
) -> torch.Tensor:
    """Return true for a low-speed object whose center lies inside a goal region."""
    asset: RigidObject = env.scene[asset_cfg.name]
    center = torch.as_tensor(center_m, device=asset.device, dtype=asset.data.root_pos_w.dtype)
    half_size = torch.as_tensor(half_size_m, device=asset.device, dtype=asset.data.root_pos_w.dtype)
    inside = torch.all(torch.abs(asset.data.root_pos_w - center) <= half_size, dim=1)
    settled = torch.linalg.norm(asset.data.root_lin_vel_w, dim=1) <= float(max_speed_mps)
    return inside & settled
