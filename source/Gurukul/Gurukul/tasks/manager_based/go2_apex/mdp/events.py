from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Literal

import torch
from pxr import Usd, UsdPhysics

import isaaclab.sim as sim_utils
import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs.mdp.events import (
    _randomize_prop_by_op,
    randomize_rigid_body_scale as _isaac_randomize_rigid_body_scale,
    reset_root_state_uniform as _isaac_reset_root_state_uniform,
)
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


def _get_filtered_pairs_relationship(prim: Usd.Prim) -> Usd.Relationship:
    """Return the USD relationship used to disable a precise collision pair."""
    if hasattr(UsdPhysics, "FilteredPairsAPI"):
        api = UsdPhysics.FilteredPairsAPI.Apply(prim)
        if hasattr(api, "GetFilteredPairsRel"):
            relationship = api.GetFilteredPairsRel()
            if not relationship:
                relationship = api.CreateFilteredPairsRel()
            return relationship
    relationship = prim.GetRelationship("filteredPairs")
    if not relationship:
        relationship = prim.CreateRelationship("filteredPairs")
    return relationship


def _collect_collision_prims(prim: Usd.Prim) -> list[Usd.Prim]:
    """Resolve authored colliders below a rigid body, or retain the body itself."""
    collision_prims = [
        candidate for candidate in Usd.PrimRange(prim) if candidate.HasAPI(UsdPhysics.CollisionAPI)
    ]
    return collision_prims or [prim]


def disable_collision_pairs(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | Sequence[int] | slice | None,
    prim_path_pairs: Sequence[tuple[str, str]],
) -> None:
    """Disable selected same-environment collision pairs before simulation starts.

    Paths are relative to each ``env_<index>`` prim, for example
    ``("Robot/d1/Link6", "Object")``. Authoring the relationship on both
    bodies keeps the result consistent across USD/PhysX versions.
    """
    stage = sim_utils.get_current_stage()
    if stage is None or not prim_path_pairs:
        return

    if env_ids is None or isinstance(env_ids, slice):
        resolved_env_ids = range(env.scene.num_envs)
    elif isinstance(env_ids, torch.Tensor):
        resolved_env_ids = env_ids.detach().cpu().flatten().tolist()
    else:
        resolved_env_ids = list(env_ids)

    for env_id in resolved_env_ids:
        env_path = f"{env.scene.env_ns}/env_{int(env_id)}"
        for relative_path_a, relative_path_b in prim_path_pairs:
            prim_a = stage.GetPrimAtPath(f"{env_path}/{relative_path_a}")
            prim_b = stage.GetPrimAtPath(f"{env_path}/{relative_path_b}")
            if not prim_a.IsValid() or not prim_b.IsValid():
                raise RuntimeError(
                    "Cannot filter missing collision prim pair: "
                    f"{prim_a.GetPath()} ({prim_a.IsValid()}) and "
                    f"{prim_b.GetPath()} ({prim_b.IsValid()})."
                )
            for collider_a in _collect_collision_prims(prim_a):
                for collider_b in _collect_collision_prims(prim_b):
                    _get_filtered_pairs_relationship(collider_a).AddTarget(collider_b.GetPath())
                    _get_filtered_pairs_relationship(collider_b).AddTarget(collider_a.GetPath())


def randomize_rigid_body_size_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    size_scale_range: tuple[float, float],
    asset_cfg: SceneEntityCfg,
    scale_attr_name: str = "_object_size_scale",
) -> None:
    """Randomize object XYZ size isotropically and retain one scale per environment."""
    low, high = size_scale_range
    if low <= 0.0 or high < low:
        raise ValueError(f"Invalid size_scale_range: {size_scale_range}")

    env_ids_cpu = (
        torch.arange(env.scene.num_envs, device="cpu", dtype=torch.long)
        if env_ids is None
        else torch.as_tensor(env_ids, device="cpu", dtype=torch.long).flatten()
    )
    # Isaac Lab samples an (N, 1) CPU tensor for an isotropic tuple range.
    # Replaying that exact RNG call lets us retain the sampled values while the
    # upstream function remains responsible for authoring the USD scales.
    rng_state = torch.random.get_rng_state()
    sampled_scale = math_utils.sample_uniform(
        low,
        high,
        (env_ids_cpu.numel(), 1),
        device="cpu",
    )
    torch.random.set_rng_state(rng_state)
    _isaac_randomize_rigid_body_scale(
        env,
        env_ids,
        scale_range=size_scale_range,
        asset_cfg=asset_cfg,
    )

    retained_scale = getattr(env, scale_attr_name, None)
    if retained_scale is None:
        retained_scale = torch.ones(env.scene.num_envs, dtype=torch.float32, device=env.device)
        setattr(env, scale_attr_name, retained_scale)
    retained_scale[env_ids_cpu.to(env.device)] = sampled_scale[:, 0].to(env.device)


def reset_root_state_uniform_with_size_scale(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    nominal_size: Sequence[float],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    scale_attr_name: str = "_object_size_scale",
    preserve_grasp_face: bool = True,
) -> None:
    """Reset a scaled object while fixing its ground plane and optional grasp-facing -X face."""
    if len(nominal_size) != 3:
        raise ValueError(f"nominal_size must contain XYZ dimensions, got {nominal_size}.")
    _isaac_reset_root_state_uniform(
        env,
        env_ids,
        pose_range=pose_range,
        velocity_range=velocity_range,
        asset_cfg=asset_cfg,
    )
    asset: RigidObject = env.scene[asset_cfg.name]
    resolved_env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=asset.device).flatten()
    if resolved_env_ids.numel() == 0:
        return
    size_scale = getattr(env, scale_attr_name, None)
    if size_scale is None:
        return
    selected_scale = size_scale[resolved_env_ids]
    root_pose_w = asset.data.root_pose_w[resolved_env_ids].clone()
    local_grasp_face_offset = torch.zeros_like(root_pose_w[:, :3])
    if preserve_grasp_face:
        local_grasp_face_offset[:, 0] = 0.5 * float(nominal_size[0]) * (selected_scale - 1.0)
    root_pose_w[:, :3] += math_utils.quat_apply(root_pose_w[:, 3:7], local_grasp_face_offset)
    root_pose_w[:, 2] += 0.5 * float(nominal_size[2]) * (selected_scale - 1.0)
    asset.write_root_pose_to_sim(root_pose_w, env_ids=resolved_env_ids)


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "abs",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """
    Randomize the joint default positions which may be different from URDF due to calibration errors.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]

    # save nominal value for export
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    # resolve joint indices
    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)  # for optimization purposes
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids = env_ids[:, None]
        asset.data.default_joint_pos[env_ids, joint_ids] = pos
        # update the offset in action since it is not updated automatically
        env.action_manager.get_term("joint_pos")._offset[env_ids, joint_ids] = pos


def randomize_fixed_joint_defaults(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    position_ranges: dict[str, tuple[float, float]],
    velocity_range: tuple[float, float] = (0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Randomize non-action joints as fixed per-episode morphology offsets.

    This updates both the articulation default joint position and the current sim
    joint state. It intentionally does not touch action offsets because these
    joints are not controlled by the policy action term.
    """
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device, dtype=torch.long)
    else:
        env_ids = torch.as_tensor(env_ids, device=asset.device, dtype=torch.long).flatten()
    if env_ids.numel() == 0:
        return

    joint_pos = asset.data.joint_pos[env_ids].clone()
    joint_vel = asset.data.joint_vel[env_ids].clone()
    default_joint_pos = asset.data.default_joint_pos.clone()
    joint_pos_limits = asset.data.default_joint_pos_limits.clone()
    velocities = torch.tensor(velocity_range, device=asset.device, dtype=joint_vel.dtype)
    limit_eps = torch.tensor(1.0e-6, device=asset.device, dtype=joint_pos_limits.dtype)

    for joint_name, position_range in position_ranges.items():
        joint_result = asset.find_joints(joint_name)
        joint_ids = torch.as_tensor(
            joint_result[0] if isinstance(joint_result, tuple) else joint_result,
            device=asset.device,
            dtype=torch.long,
        )
        if joint_ids.numel() == 0:
            continue

        ranges = torch.tensor(position_range, device=asset.device, dtype=joint_pos.dtype)
        sampled_pos = math_utils.sample_uniform(
            ranges[0], ranges[1], (env_ids.numel(), joint_ids.numel()), device=asset.device
        )
        sampled_vel = math_utils.sample_uniform(
            velocities[0], velocities[1], (env_ids.numel(), joint_ids.numel()), device=asset.device
        )
        joint_pos[:, joint_ids] = sampled_pos
        joint_vel[:, joint_ids] = sampled_vel
        default_joint_pos[env_ids[:, None], joint_ids] = sampled_pos
        # Lock the randomized non-action joint around the sampled morphology offset.
        joint_pos_limits[env_ids[:, None], joint_ids, 0] = sampled_pos - limit_eps
        joint_pos_limits[env_ids[:, None], joint_ids, 1] = sampled_pos + limit_eps

    asset.data.default_joint_pos[env_ids] = default_joint_pos[env_ids]
    asset.write_joint_position_limit_to_sim(
        joint_pos_limits[env_ids][:, asset_cfg.joint_ids],
        joint_ids=asset_cfg.joint_ids,
        env_ids=env_ids,
        warn_limit_violation=False,
    )
    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)
