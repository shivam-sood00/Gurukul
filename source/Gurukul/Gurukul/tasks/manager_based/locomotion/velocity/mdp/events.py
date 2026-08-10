from __future__ import annotations

import math
from typing import Literal

import torch

import isaaclab.utils.math as math_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp.events import _validate_scale_range
from isaaclab.managers import SceneEntityCfg

from .utils import is_env_assigned_to_terrain


def randomize_rigid_body_inertia(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    inertia_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the inertia tensors of the bodies by adding, scaling, or setting random values.

    This function allows randomizing only the diagonal inertia tensor components (xx, yy, zz) of the bodies.
    The function samples random values from the given distribution parameters and adds, scales, or sets the values
    into the physics simulation based on the operation.

    .. tip::
        This function uses CPU tensors to assign the body inertias. It is recommended to use this function
        only during the initialization of the environment.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # resolve environment ids
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # get the current inertia tensors of the bodies (num_assets, num_bodies, 9 for articulations or 9 for rigid objects)
    inertias = asset.root_physx_view.get_inertias()

    # apply randomization on default values
    inertias[env_ids[:, None], body_ids, :] = asset.data.default_inertia[env_ids[:, None], body_ids, :].clone()

    # randomize each diagonal element (xx, yy, zz -> indices 0, 4, 8)
    for idx in [0, 4, 8]:
        # Extract and randomize the specific diagonal element
        randomized_inertias = _randomize_prop_by_op(
            inertias[:, :, idx],
            inertia_distribution_params,
            env_ids,
            body_ids,
            operation,
            distribution,
        )
        # Assign the randomized values back to the inertia tensor
        inertias[env_ids[:, None], body_ids, idx] = randomized_inertias

    # set the inertia tensors into the physics simulation
    asset.root_physx_view.set_inertias(inertias, env_ids)


def randomize_com_positions(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    com_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize the center of mass (COM) positions for the rigid bodies.

    This function allows randomizing the COM positions of the bodies in the physics simulation. The positions can be
    randomized by adding, scaling, or setting random values sampled from the specified distribution.

    .. tip::
        This function is intended for initialization or offline adjustments, as it modifies physics properties directly.

    Args:
        env (ManagerBasedEnv): The simulation environment.
        env_ids (torch.Tensor | None): Specific environment indices to apply randomization,
            or None for all environments.
        asset_cfg (SceneEntityCfg): The configuration for the target asset whose COM will be randomized.
        com_distribution_params (tuple[float, float]): Parameters of the distribution (e.g., min and max for uniform).
        operation (Literal["add", "scale", "abs"]): The operation to apply for randomization.
        distribution (Literal["uniform", "log_uniform", "gaussian"]): The distribution to sample random values from.
    """
    # Extract the asset (Articulation or RigidObject)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # Resolve environment indices
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    # Resolve body indices
    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    # Get the current COM offsets (num_assets, num_bodies, 3)
    com_offsets = asset.root_physx_view.get_coms()

    for dim_idx in range(3):  # Randomize x, y, z independently
        randomized_offset = _randomize_prop_by_op(
            com_offsets[:, :, dim_idx],
            com_distribution_params,
            env_ids,
            body_ids,
            operation,
            distribution,
        )
        com_offsets[env_ids[:, None], body_ids, dim_idx] = randomized_offset[env_ids[:, None], body_ids]

    # Set the randomized COM offsets into the simulation
    asset.root_physx_view.set_coms(com_offsets, env_ids)


"""
Internal helper functions.
"""


def _randomize_prop_by_op(
    data: torch.Tensor,
    distribution_parameters: tuple[float | torch.Tensor, float | torch.Tensor],
    dim_0_ids: torch.Tensor | None,
    dim_1_ids: torch.Tensor | slice,
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"],
) -> torch.Tensor:
    """Perform data randomization based on the given operation and distribution.

    Args:
        data: The data tensor to be randomized. Shape is (dim_0, dim_1).
        distribution_parameters: The parameters for the distribution to sample values from.
        dim_0_ids: The indices of the first dimension to randomize.
        dim_1_ids: The indices of the second dimension to randomize.
        operation: The operation to perform on the data. Options: 'add', 'scale', 'abs'.
        distribution: The distribution to sample the random values from. Options: 'uniform', 'log_uniform'.

    Returns:
        The data tensor after randomization. Shape is (dim_0, dim_1).

    Raises:
        NotImplementedError: If the operation or distribution is not supported.
    """
    # resolve shape
    # -- dim 0
    if dim_0_ids is None:
        n_dim_0 = data.shape[0]
        dim_0_ids = slice(None)
    else:
        n_dim_0 = len(dim_0_ids)
        if not isinstance(dim_1_ids, slice):
            dim_0_ids = dim_0_ids[:, None]
    # -- dim 1
    if isinstance(dim_1_ids, slice):
        n_dim_1 = data.shape[1]
    else:
        n_dim_1 = len(dim_1_ids)

    # resolve the distribution
    if distribution == "uniform":
        dist_fn = math_utils.sample_uniform
    elif distribution == "log_uniform":
        dist_fn = math_utils.sample_log_uniform
    elif distribution == "gaussian":
        dist_fn = math_utils.sample_gaussian
    else:
        raise NotImplementedError(
            f"Unknown distribution: '{distribution}' for joint properties randomization."
            " Please use 'uniform', 'log_uniform', 'gaussian'."
        )
    # perform the operation
    if operation == "add":
        data[dim_0_ids, dim_1_ids] += dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "scale":
        data[dim_0_ids, dim_1_ids] *= dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    elif operation == "abs":
        data[dim_0_ids, dim_1_ids] = dist_fn(*distribution_parameters, (n_dim_0, n_dim_1), device=data.device)
    else:
        raise NotImplementedError(
            f"Unknown operation: '{operation}' for property randomization. Please use 'add', 'scale', or 'abs'."
        )
    return data


def reset_root_state_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    pose_range: dict[str, tuple[float, float]],
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
):
    """Reset the asset root state to a random position and velocity uniformly within the given ranges.

    This function randomizes the root position and velocity of the asset.

    * It samples the root position from the given ranges and adds them to the default root position, before setting
      them into the physics simulation.
    * It samples the root orientation from the given ranges and sets them into the physics simulation.
    * It samples the root velocity from the given ranges and sets them into the physics simulation.

    The function takes a dictionary of pose and velocity ranges for each axis and rotation. The keys of the
    dictionary are ``x``, ``y``, ``z``, ``roll``, ``pitch``, and ``yaw``. The values are tuples of the form
    ``(min, max)``. If the dictionary does not contain a key, the position or velocity is set to zero for that axis.

    Note: If "pits" terrain exists, environments on pit terrain will be reset to default state without random
    perturbations to avoid the robot falling into the pit.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject | Articulation = env.scene[asset_cfg.name]

    # Separate pit and non-pit environments
    # Check which environments are assigned to pit terrain (not random reset)
    assigned_to_pits = is_env_assigned_to_terrain(env, "pits")
    pit_env_ids = env_ids[assigned_to_pits[env_ids]]
    non_pit_env_ids = env_ids[~assigned_to_pits[env_ids]]

    # Reset pit environments to default state (no random perturbations)
    if len(pit_env_ids) > 0:
        root_states = asset.data.default_root_state[pit_env_ids].clone()
        positions = root_states[:, 0:3] + env.scene.env_origins[pit_env_ids]
        orientations = root_states[:, 3:7]
        velocities = torch.zeros_like(root_states[:, 7:13])
        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=pit_env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=pit_env_ids)

    # Reset non-pit environments with random perturbations
    if len(non_pit_env_ids) > 0:
        root_states = asset.data.default_root_state[non_pit_env_ids].clone()

        # poses
        range_list = [pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(non_pit_env_ids), 6), device=asset.device
        )

        positions = root_states[:, 0:3] + env.scene.env_origins[non_pit_env_ids] + rand_samples[:, 0:3]
        orientations_delta = math_utils.quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        orientations = math_utils.quat_mul(root_states[:, 3:7], orientations_delta)
        # velocities
        range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=asset.device)
        rand_samples = math_utils.sample_uniform(
            ranges[:, 0], ranges[:, 1], (len(non_pit_env_ids), 6), device=asset.device
        )

        velocities = root_states[:, 7:13] + rand_samples

        # set into the physics simulation
        asset.write_root_pose_to_sim(torch.cat([positions, orientations], dim=-1), env_ids=non_pit_env_ids)
        asset.write_root_velocity_to_sim(velocities, env_ids=non_pit_env_ids)


def randomize_joint_default_pos(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    pos_distribution_params: tuple[float, float] | None = None,
    operation: Literal["add", "scale", "abs"] = "add",
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
):
    """Randomize default joint positions (EngineAI-style calibration offsets)."""
    asset: Articulation = env.scene[asset_cfg.name]
    asset.data.default_joint_pos_nominal = torch.clone(asset.data.default_joint_pos[0])

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device)

    if asset_cfg.joint_ids == slice(None):
        joint_ids = slice(None)
    else:
        joint_ids = torch.tensor(asset_cfg.joint_ids, dtype=torch.int, device=asset.device)

    if pos_distribution_params is not None:
        pos = asset.data.default_joint_pos.to(asset.device).clone()
        pos = _randomize_prop_by_op(
            pos, pos_distribution_params, env_ids, joint_ids, operation=operation, distribution=distribution
        )[env_ids][:, joint_ids]

        # Grid indices for articulation tensors (global joint/body ids).
        if env_ids != slice(None) and joint_ids != slice(None):
            env_ids_bc = env_ids[:, None]
        else:
            env_ids_bc = env_ids
        asset.data.default_joint_pos[env_ids_bc, joint_ids] = pos

        # JointPositionAction._offset columns are local action indices 0..action_dim-1, not articulation joint indices.
        joint_action = env.action_manager.get_term("joint_pos")
        local_joint_ids = torch.arange(pos.shape[1], dtype=torch.long, device=joint_action.device)
        joint_action._offset[env_ids_bc, local_joint_ids] = pos  # noqa: SLF001


def reset_joints_by_absolute_uniform(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor,
    position_ranges: dict[str, tuple[float, float]],
    velocity_range: tuple[float, float] = (0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Reset selected joints to absolute uniformly sampled positions."""
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device, dtype=torch.long)
    else:
        env_ids = torch.as_tensor(env_ids, device=asset.device, dtype=torch.long).flatten()
    if env_ids.numel() == 0:
        return

    joint_pos = asset.data.joint_pos[env_ids].clone()
    joint_vel = asset.data.joint_vel[env_ids].clone()
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
        velocities = torch.tensor(velocity_range, device=asset.device, dtype=joint_vel.dtype)
        joint_pos[:, joint_ids] = math_utils.sample_uniform(
            ranges[0], ranges[1], (env_ids.numel(), joint_ids.numel()), device=asset.device
        )
        joint_vel[:, joint_ids] = math_utils.sample_uniform(
            velocities[0], velocities[1], (env_ids.numel(), joint_ids.numel()), device=asset.device
        )

    asset.write_joint_state_to_sim(joint_pos, joint_vel, env_ids=env_ids)


def randomize_fixed_joint_defaults(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    position_ranges: dict[str, tuple[float, float]],
    velocity_range: tuple[float, float] = (0.0, 0.0),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Sample non-action morphology joints and lock their limits to the sampled offset."""
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


def randomize_rigid_body_mass_fn(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    mass_distribution_params: tuple[float, float],
    operation: Literal["add", "scale", "abs"],
    distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
    recompute_inertia: bool = True,
    min_mass: float = 1e-6,
) -> None:
    """Randomize rigid-body masses (same behavior as ``isaaclab...randomize_rigid_body_mass`` ``__call__``).

    Isaac Lab registers mass randomization as a :class:`ManagerTermBase` class, which is only constructed
    after the timeline PLAY callback. ``apply(startup)`` can run before that, so the class is invoked
    incorrectly. Use this **function** in ``EventTermCfg`` to randomize at startup safely.
    """
    if operation == "scale":
        _validate_scale_range(mass_distribution_params, "mass_distribution_params", allow_zero=False)
    elif operation not in ("abs", "add"):
        raise ValueError(
            f"randomize_rigid_body_mass_fn does not support operation: '{operation}'. Use 'add', 'scale', or 'abs'."
        )
    if min_mass < 1e-6:
        raise ValueError("min_mass must be at least 1e-6 to avoid physics errors.")

    # Skip a second resolve: ManagerBase may already have resolved this cfg (timeline PLAY). Calling
    # resolve() again hits SceneEntityCfg._resolve_body_names with both body_names and body_ids set and
    # can raise "not consistent" for patterns like body_names='.*'.
    if asset_cfg.body_ids == slice(None):
        asset_cfg.resolve(env.scene)
    asset: Articulation | RigidObject = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")

    masses = asset.root_physx_view.get_masses()
    masses[env_ids[:, None], body_ids] = asset.data.default_mass[env_ids[:, None], body_ids].clone()

    masses = _randomize_prop_by_op(
        masses, mass_distribution_params, env_ids, body_ids, operation=operation, distribution=distribution
    )
    masses = torch.clamp(masses, min=min_mass)

    asset.root_physx_view.set_masses(masses, env_ids)

    if recompute_inertia:
        ratios = masses[env_ids[:, None], body_ids] / asset.data.default_mass[env_ids[:, None], body_ids]
        inertias = asset.root_physx_view.get_inertias()
        if isinstance(asset, Articulation):
            inertias[env_ids[:, None], body_ids] = (
                asset.data.default_inertia[env_ids[:, None], body_ids] * ratios[..., None]
            )
        else:
            inertias[env_ids] = asset.data.default_inertia[env_ids] * ratios
        asset.root_physx_view.set_inertias(inertias, env_ids)


def randomize_end_effector_payload_mass(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    payload_mass_range: tuple[float, float] = (0.0, 0.5),
    recompute_inertia: bool = True,
    difficulty_attr: str | None = None,
) -> None:
    """Approximate a grasped payload by adding random mass to the selected end-effector body.

    When ``difficulty_attr`` is provided, the sampled payload is multiplied by
    that per-environment curriculum value. A difficulty of zero therefore
    restores the nominal end-effector mass, while a difficulty of one samples
    the complete configured payload range.
    """
    if asset_cfg.body_ids is None:
        asset_cfg.resolve(env.scene)
    asset: Articulation = env.scene[asset_cfg.name]

    if env_ids is None:
        env_ids_device = torch.arange(env.scene.num_envs, device=env.device, dtype=torch.long)
        env_ids_cpu = torch.arange(env.scene.num_envs, device="cpu", dtype=torch.long)
    else:
        env_ids_device = torch.as_tensor(env_ids, device=env.device, dtype=torch.long).flatten()
        env_ids_cpu = env_ids_device.cpu()
    if env_ids_cpu.numel() == 0:
        return

    if asset_cfg.body_ids == slice(None):
        body_ids = torch.arange(asset.num_bodies, dtype=torch.int, device="cpu")
    else:
        body_ids = torch.tensor(asset_cfg.body_ids, dtype=torch.int, device="cpu")
    if body_ids.numel() == 0:
        return

    masses = asset.root_physx_view.get_masses()
    low, high = float(payload_mass_range[0]), float(payload_mass_range[1])
    if low < 0.0 or high < low:
        raise ValueError(f"Invalid payload mass range {payload_mass_range}; expected 0 <= low <= high.")
    payload = torch.empty((env_ids_cpu.numel(), body_ids.numel()), device=masses.device).uniform_(low, high)
    if difficulty_attr is not None:
        difficulty = getattr(env, difficulty_attr, 0.0)
        difficulty = torch.as_tensor(difficulty, device=env.device, dtype=torch.float32)
        if difficulty.numel() == 1:
            difficulty = difficulty.expand(env.scene.num_envs)
        elif difficulty.numel() != env.scene.num_envs:
            raise ValueError(
                f"Environment attribute '{difficulty_attr}' must be scalar or have {env.scene.num_envs} values, "
                f"got shape {tuple(difficulty.shape)}."
            )
        selected_difficulty = difficulty.flatten().index_select(0, env_ids_device).clamp(0.0, 1.0)
        payload *= selected_difficulty.to(device=payload.device, dtype=payload.dtype).unsqueeze(1)
    masses[env_ids_cpu[:, None], body_ids] = asset.data.default_mass[env_ids_cpu[:, None], body_ids].clone() + payload
    asset.root_physx_view.set_masses(masses, env_ids_cpu)

    if recompute_inertia:
        ratios = masses[env_ids_cpu[:, None], body_ids] / asset.data.default_mass[env_ids_cpu[:, None], body_ids]
        inertias = asset.root_physx_view.get_inertias()
        inertias[env_ids_cpu[:, None], body_ids] = asset.data.default_inertia[env_ids_cpu[:, None], body_ids] * ratios[
            ..., None
        ]
        asset.root_physx_view.set_inertias(inertias, env_ids_cpu)

    if not hasattr(env, "_arm_payload_mass"):
        env._arm_payload_mass = torch.zeros(env.scene.num_envs, device=env.device, dtype=torch.float32)
    env._arm_payload_mass[env_ids_device] = payload.sum(dim=1).to(env.device, dtype=torch.float32)


def _clamp_arm_ee_target(
    pos: torch.Tensor,
    ee_pos_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    body_exclusion_box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    body_clearance: float,
    extra_exclusion_boxes: tuple[tuple[tuple[float, float], tuple[float, float], tuple[float, float]], ...] = (),
    workspace_origin: tuple[float, float, float] | None = None,
    reach_range: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Project a command into the radial workspace and out of expanded keep-out volumes."""
    for axis, (low, high) in enumerate(ee_pos_range):
        pos[axis] = torch.clamp(pos[axis], low, high)

    if workspace_origin is not None and reach_range is not None:
        origin = torch.tensor(workspace_origin, device=pos.device, dtype=pos.dtype)
        radial = pos - origin
        distance = torch.linalg.vector_norm(radial)
        min_reach, max_reach = float(reach_range[0]), float(reach_range[1])
        if distance < 1.0e-6:
            radial = torch.tensor((max(min_reach, 1.0e-3), 0.0, 0.0), device=pos.device, dtype=pos.dtype)
        elif distance < min_reach or distance > max_reach:
            radial = radial * (torch.clamp(distance, min=min_reach, max=max_reach) / distance)
        pos[:] = origin + radial

    epsilon = 1.0e-4
    for exclusion_box in (body_exclusion_box, *extra_exclusion_boxes):
        expanded = tuple(
            (float(low) - float(body_clearance), float(high) + float(body_clearance))
            for low, high in exclusion_box
        )
        inside = all(low <= pos[axis].item() <= high for axis, (low, high) in enumerate(expanded))
        if not inside:
            continue

        # The command box lies in front of and above the robot, so project to
        # the nearest reachable face: front, either side, or top.
        distances = torch.stack(
            (
                torch.as_tensor(expanded[0][1], device=pos.device, dtype=pos.dtype) - pos[0],
                pos[1] - torch.as_tensor(expanded[1][0], device=pos.device, dtype=pos.dtype),
                torch.as_tensor(expanded[1][1], device=pos.device, dtype=pos.dtype) - pos[1],
                torch.as_tensor(expanded[2][1], device=pos.device, dtype=pos.dtype) - pos[2],
            )
        )
        face = int(torch.argmin(distances).item())
        if face == 0:
            pos[0] = expanded[0][1] + epsilon
        elif face == 1:
            pos[1] = expanded[1][0] - epsilon
        elif face == 2:
            pos[1] = expanded[1][1] + epsilon
        else:
            pos[2] = expanded[2][1] + epsilon

    for axis, (low, high) in enumerate(ee_pos_range):
        pos[axis] = torch.clamp(pos[axis], low, high)
    return pos


def _arm_ee_target_is_feasible(
    pos: torch.Tensor,
    ee_pos_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    body_exclusion_box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    body_clearance: float,
    extra_exclusion_boxes: tuple[tuple[tuple[float, float], tuple[float, float], tuple[float, float]], ...] = (),
    workspace_origin: tuple[float, float, float] | None = None,
    reach_range: tuple[float, float] | None = None,
) -> bool:
    """Return whether a sampled target belongs to the exact WBC training volume."""
    for axis, (low, high) in enumerate(ee_pos_range):
        value = float(pos[axis].item())
        if value < float(low) or value > float(high):
            return False

    if workspace_origin is not None and reach_range is not None:
        origin = torch.tensor(workspace_origin, device=pos.device, dtype=pos.dtype)
        distance = float(torch.linalg.vector_norm(pos - origin).item())
        if distance < float(reach_range[0]) or distance > float(reach_range[1]):
            return False

    for exclusion_box in (body_exclusion_box, *extra_exclusion_boxes):
        expanded = tuple(
            (float(low) - float(body_clearance), float(high) + float(body_clearance))
            for low, high in exclusion_box
        )
        if all(low <= float(pos[axis].item()) <= high for axis, (low, high) in enumerate(expanded)):
            return False
    return True


def _sample_arm_ee_point(
    device: torch.device,
    ee_pos_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    body_exclusion_box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    body_clearance: float,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    z_range: tuple[float, float],
    extra_exclusion_boxes: tuple[tuple[tuple[float, float], tuple[float, float], tuple[float, float]], ...] = (),
    workspace_origin: tuple[float, float, float] | None = None,
    reach_range: tuple[float, float] | None = None,
) -> torch.Tensor:
    pos = torch.empty(3, device=device, dtype=torch.float32)
    clipped_ranges = []
    for axis, value_range in enumerate((x_range, y_range, z_range)):
        low = max(float(value_range[0]), float(ee_pos_range[axis][0]))
        high = min(float(value_range[1]), float(ee_pos_range[axis][1]))
        if high < low:
            nearest = min(
                max(0.5 * (float(value_range[0]) + float(value_range[1])), float(ee_pos_range[axis][0])),
                float(ee_pos_range[axis][1]),
            )
            low = high = nearest
        clipped_ranges.append((low, high))

    def sample_until_feasible(ranges, attempts: int) -> bool:
        for _ in range(attempts):
            for axis, (low, high) in enumerate(ranges):
                pos[axis] = (high - low) * torch.rand((), device=device) + low
            if _arm_ee_target_is_feasible(
                pos,
                ee_pos_range,
                body_exclusion_box,
                body_clearance,
                extra_exclusion_boxes,
                workspace_origin,
                reach_range,
            ):
                return True
        return False

    if sample_until_feasible(clipped_ranges, 48):
        return pos
    # Some motion-primitive subranges can become empty after the reach-shell
    # and keep-out cuts. Fall back to another legal training sample instead of
    # projecting many commands onto the same boundary.
    if sample_until_feasible(ee_pos_range, 96):
        return pos
    return _clamp_arm_ee_target(
        pos,
        ee_pos_range,
        body_exclusion_box,
        body_clearance,
        extra_exclusion_boxes,
        workspace_origin,
        reach_range,
    )


def _sample_stratified_arm_ee_point(
    device: torch.device,
    env_index: int,
    phase: int,
    ee_pos_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    body_exclusion_box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    body_clearance: float,
    bins: tuple[int, int, int] = (6, 6, 5),
    extra_exclusion_boxes: tuple[tuple[tuple[float, float], tuple[float, float], tuple[float, float]], ...] = (),
    workspace_origin: tuple[float, float, float] | None = None,
    reach_range: tuple[float, float] | None = None,
) -> torch.Tensor:
    """Sample a legal workspace point from independent axis bins with random jitter."""
    bin_count_total = math.prod(bins)
    start_bin = (int(env_index) + 53 * int(phase)) % bin_count_total
    for attempt in range(min(48, bin_count_total)):
        flat_bin = (start_bin + 53 * attempt) % bin_count_total
        bin_ids = (
            flat_bin // (bins[1] * bins[2]),
            (flat_bin // bins[2]) % bins[1],
            flat_bin % bins[2],
        )
        pos = torch.empty(3, device=device, dtype=torch.float32)
        for axis, (axis_range, bin_id, bin_count) in enumerate(zip(ee_pos_range, bin_ids, bins, strict=True)):
            low, high = float(axis_range[0]), float(axis_range[1])
            width = (high - low) / float(bin_count)
            bin_low = low + float(bin_id) * width
            pos[axis] = bin_low + width * torch.rand((), device=device)
        if _arm_ee_target_is_feasible(
            pos,
            ee_pos_range,
            body_exclusion_box,
            body_clearance,
            extra_exclusion_boxes,
            workspace_origin,
            reach_range,
        ):
            return pos
    return _sample_arm_ee_point(
        device,
        ee_pos_range,
        body_exclusion_box,
        body_clearance,
        ee_pos_range[0],
        ee_pos_range[1],
        ee_pos_range[2],
        extra_exclusion_boxes,
        workspace_origin,
        reach_range,
    )


def _scale_arm_ee_range(
    ee_pos_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    difficulty: float,
    neutral_pos: tuple[float, float, float] = (0.42, 0.0, 0.34),
    min_fraction: float = 0.35,
) -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
    """Interpolate from a compact center workspace to the full configured workspace."""
    fraction = min_fraction + (1.0 - min_fraction) * max(0.0, min(1.0, float(difficulty)))
    scaled = []
    for axis, axis_range in enumerate(ee_pos_range):
        low, high = float(axis_range[0]), float(axis_range[1])
        center = min(max(float(neutral_pos[axis]), low), high)
        scaled.append((center + (low - center) * fraction, center + (high - center) * fraction))
    return tuple(scaled)  # type: ignore[return-value]


def _start_arm_motion_primitive(
    env: ManagerBasedEnv,
    env_id: torch.Tensor,
    primitive_id_to_name: dict[int, str],
    ee_pos_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    body_exclusion_box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    body_clearance: float,
    extra_exclusion_boxes: tuple[tuple[tuple[float, float], tuple[float, float], tuple[float, float]], ...],
    workspace_origin: tuple[float, float, float] | None,
    reach_range: tuple[float, float] | None,
) -> None:
    device = env.device
    choices = torch.tensor(list(primitive_id_to_name.keys()), device=device, dtype=torch.long)
    kind = choices[torch.randint(0, choices.numel(), (), device=device)]
    env._arm_motion_kind[env_id] = kind
    env._arm_motion_phase[env_id] = 0

    if primitive_id_to_name[int(kind.item())] == "pick_place":
        pick = _sample_arm_ee_point(
            device,
            ee_pos_range,
            body_exclusion_box,
            body_clearance,
            (0.48, 0.56),
            (-0.18, 0.18),
            (0.32, 0.44),
            extra_exclusion_boxes,
            workspace_origin,
            reach_range,
        )
        place_y = torch.clamp(
            pick[1] + math_utils.sample_uniform(-0.20, 0.20, (1,), device=device)[0], -0.22, 0.22
        )
        place = _sample_arm_ee_point(
            device,
            ee_pos_range,
            body_exclusion_box,
            body_clearance,
            (0.46, 0.60),
            (place_y.item(), place_y.item()),
            (0.32, 0.48),
            extra_exclusion_boxes,
            workspace_origin,
            reach_range,
        )
        env._arm_motion_pick_pos[env_id] = pick
        env._arm_motion_place_pos[env_id] = place


def _arm_motion_gripper_command(kind: str, phase: int) -> bool | None:
    """Return a phase-aware close command, or ``None`` for generic random practice."""
    if kind != "pick_place":
        return None
    # approach-above, descend-open, grasp-at-pick-closed, lift-closed,
    # transfer-closed, place-closed, release-at-place-open, retreat-open.
    return int(phase) in (2, 3, 4, 5)


def _next_arm_motion_target(
    env: ManagerBasedEnv,
    env_id: torch.Tensor,
    primitive_id_to_name: dict[int, str],
    ee_pos_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    body_exclusion_box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    body_clearance: float,
    extra_exclusion_boxes: tuple[tuple[tuple[float, float], tuple[float, float], tuple[float, float]], ...],
    workspace_origin: tuple[float, float, float] | None,
    reach_range: tuple[float, float] | None,
) -> torch.Tensor:
    device = env.device
    if env._arm_motion_kind[env_id].item() < 0:
        _start_arm_motion_primitive(
            env,
            env_id,
            primitive_id_to_name,
            ee_pos_range,
            body_exclusion_box,
            body_clearance,
            extra_exclusion_boxes,
            workspace_origin,
            reach_range,
        )

    kind = primitive_id_to_name[int(env._arm_motion_kind[env_id].item())]
    phase = int(env._arm_motion_phase[env_id].item())
    if hasattr(env, "_arm_motion_gripper_command"):
        gripper_command = _arm_motion_gripper_command(kind, phase)
        env._arm_motion_gripper_command[env_id] = -1 if gripper_command is None else int(gripper_command)

    def sample_point(
        x_range: tuple[float, float],
        y_range: tuple[float, float],
        z_range: tuple[float, float],
    ) -> torch.Tensor:
        return _sample_arm_ee_point(
            device,
            ee_pos_range,
            body_exclusion_box,
            body_clearance,
            x_range,
            y_range,
            z_range,
            extra_exclusion_boxes,
            workspace_origin,
            reach_range,
        )

    if kind == "pick_place":
        pick = env._arm_motion_pick_pos[env_id]
        place = env._arm_motion_place_pos[env_id]
        lift_pick = _clamp_arm_ee_target(
            pick + torch.tensor((0.0, 0.0, 0.12), device=device),
            ee_pos_range,
            body_exclusion_box,
            body_clearance,
            extra_exclusion_boxes,
            workspace_origin,
            reach_range,
        )
        lift_place = _clamp_arm_ee_target(
            place + torch.tensor((0.0, 0.0, 0.12), device=device),
            ee_pos_range,
            body_exclusion_box,
            body_clearance,
            extra_exclusion_boxes,
            workspace_origin,
            reach_range,
        )
        # Repeated pick/place poses provide explicit close/open dwell phases,
        # avoiding a grasp command that starts only after the lift begins.
        waypoints = (lift_pick, pick, pick, lift_pick, lift_place, place, place, lift_place)
    elif kind == "reach":
        waypoints = (
            sample_point((0.50, 0.56), (-0.22, 0.22), (0.38, 0.58)),
            sample_point((0.44, 0.52), (-0.14, 0.14), (0.34, 0.54)),
        )
    elif kind == "sweep":
        x = math_utils.sample_uniform(0.46, 0.58, (1,), device=device)[0].item()
        z = math_utils.sample_uniform(0.34, 0.52, (1,), device=device)[0].item()
        waypoints = (
            sample_point((x, x), (-0.22, -0.10), (z, z)),
            sample_point((x, x), (0.10, 0.22), (z, z)),
            sample_point((0.46, 0.52), (-0.08, 0.08), (0.34, 0.52)),
        )
    elif kind == "workspace":
        env_index = int(env_id.item())
        workspace_phase = int(env._arm_workspace_sample_phase[env_id].item())
        waypoints = (
            _sample_stratified_arm_ee_point(
                device,
                env_index,
                workspace_phase,
                ee_pos_range,
                body_exclusion_box,
                body_clearance,
                extra_exclusion_boxes=extra_exclusion_boxes,
                workspace_origin=workspace_origin,
                reach_range=reach_range,
            ),
        )
        env._arm_workspace_sample_phase[env_id] += 1
    else:
        waypoints = (sample_point((0.46, 0.52), (-0.12, 0.12), (0.34, 0.52)),)

    target = waypoints[phase % len(waypoints)].clone()
    env._arm_motion_phase[env_id] += 1
    if phase + 1 >= len(waypoints):
        env._arm_motion_kind[env_id] = -1
    return _clamp_arm_ee_target(
        target,
        ee_pos_range,
        body_exclusion_box,
        body_clearance,
        extra_exclusion_boxes,
        workspace_origin,
        reach_range,
    )


def _as_per_env_tensor(
    value: bool | float | torch.Tensor,
    num_envs: int,
    device: torch.device,
    dtype: torch.dtype,
    attr_name: str,
) -> torch.Tensor:
    """Normalize scalar or per-environment curriculum state to a flat tensor."""
    values = torch.as_tensor(value, device=device, dtype=dtype).reshape(-1)
    if values.numel() == 1:
        return values.expand(num_envs)
    if values.numel() != num_envs:
        raise ValueError(f"{attr_name} must contain either 1 or {num_envs} values, got {values.numel()}.")
    return values


def _rotate_jacobian_to_root_frame(
    jacobian_w: torch.Tensor,
    root_rotation_w_to_b: torch.Tensor,
) -> torch.Tensor:
    """Express a world-frame geometric Jacobian in the articulation root frame."""
    jacobian_b = jacobian_w.clone()
    jacobian_b[:, 0:3, :] = torch.bmm(root_rotation_w_to_b, jacobian_w[:, 0:3, :])
    jacobian_b[:, 3:6, :] = torch.bmm(root_rotation_w_to_b, jacobian_w[:, 3:6, :])
    return jacobian_b


def _arm_deployment_waypoint(
    deployment_ee_waypoints: tuple[tuple[float, float, float], ...],
    phase: int,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> torch.Tensor | None:
    """Return one explicit carry-to-workspace waypoint, or ``None`` after deployment."""
    if phase < 0 or phase >= len(deployment_ee_waypoints):
        return None
    return torch.tensor(deployment_ee_waypoints[phase], device=device, dtype=dtype)


def random_arm_joint_motion(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    joint_motion_library: dict[str, tuple[tuple[float, ...], ...]],
    motion_primitives: tuple[str, ...] | None = None,
    difficulty_motion_primitives: tuple[tuple[str, ...], ...] | None = None,
    max_joint_change: float = 0.35,
    max_joint_change_range: tuple[float, float] | None = None,
    motion_speed: float = 0.45,
    motion_speed_range: tuple[float, float] | None = None,
    interpolate: bool = True,
    start_motion_enabled: bool = True,
    apply_target_when_disabled: bool = True,
    visualize: bool = False,
) -> None:
    """Sample collision-checked Airbot joint-space waypoints.

    Cartesian IK can jump to an elbow-up branch that intersects the Go2. These
    waypoints keep shoulder/elbow/wrist on the FK-checked forward branch.
    """
    del visualize  # kept for config compatibility with the IK visualizer script.

    device = env.device
    num_envs = env.scene.num_envs

    if env_ids is None:
        env_ids = torch.arange(num_envs, device=device, dtype=torch.long)
    else:
        env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.long).flatten()
    if env_ids.numel() == 0:
        return

    art = env.scene[asset_cfg.name]
    if getattr(asset_cfg, "joint_ids", None):
        joint_ids = torch.as_tensor(list(asset_cfg.joint_ids), device=device, dtype=torch.long)
    else:
        joint_result = art.find_joints(asset_cfg.joint_names)
        joint_ids = torch.as_tensor(
            joint_result[0] if isinstance(joint_result, tuple) else joint_result, device=device, dtype=torch.long
        )
    if joint_ids.numel() == 0:
        return

    hold_pose = joint_motion_library.get("stow", ((0.0,) * int(joint_ids.numel()),))[0]
    hold_goal = torch.tensor(hold_pose, device=device, dtype=torch.float32)

    if not hasattr(env, "_arm_joint_target_pos"):
        env._arm_joint_target_pos = art.data.joint_pos.index_select(dim=1, index=joint_ids).clone()
        env._arm_joint_start_pos = env._arm_joint_target_pos.clone()
        env._arm_joint_goal_pos = env._arm_joint_target_pos.clone()
        env._arm_joint_trajectory_progress = torch.zeros(num_envs, device=device, dtype=torch.float32)
        env._arm_joint_trajectory_duration = torch.zeros(num_envs, device=device, dtype=torch.float32)
        env._arm_joint_target_initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        env._arm_joint_motion_kind = [""] * num_envs
        env._arm_joint_motion_phase = torch.zeros(num_envs, device=device, dtype=torch.long)
    env._arm_joint_interpolate = bool(interpolate)

    motion_enabled = _as_per_env_tensor(
        getattr(env, "_loco_manip_arm_motion_enabled", start_motion_enabled),
        num_envs,
        device,
        torch.bool,
        "_loco_manip_arm_motion_enabled",
    )
    enabled_by_env = motion_enabled.index_select(0, env_ids)

    disabled_env_ids = env_ids[~enabled_by_env]
    if disabled_env_ids.numel() > 0:
        for env_id in disabled_env_ids:
            env._arm_joint_start_pos[env_id] = art.data.joint_pos[env_id].index_select(0, joint_ids).clone()
            env._arm_joint_goal_pos[env_id] = hold_goal
            env._arm_joint_target_pos[env_id] = hold_goal
            env._arm_joint_trajectory_duration[env_id] = 1.0
            env._arm_joint_trajectory_progress[env_id] = 0.0
            env._arm_joint_target_initialized[env_id] = True
            env._arm_joint_motion_kind[int(env_id.item())] = ""
            env._arm_joint_motion_phase[env_id] = 0
        if apply_target_when_disabled:
            art.set_joint_position_target(
                hold_goal.repeat(disabled_env_ids.numel(), 1),
                joint_ids=joint_ids,
                env_ids=disabled_env_ids,
            )

    env_ids = env_ids[enabled_by_env]
    if env_ids.numel() == 0:
        return

    arm_difficulties = _as_per_env_tensor(
        getattr(env, "_loco_manip_arm_motion_difficulty", 1.0),
        num_envs,
        device,
        torch.float32,
        "_loco_manip_arm_motion_difficulty",
    ).clamp(0.0, 1.0)

    def _primitive_names_for_difficulty(arm_difficulty: float) -> tuple[str, ...]:
        if difficulty_motion_primitives:
            level_count = len(difficulty_motion_primitives)
            level = min(level_count - 1, int(arm_difficulty * level_count))
            primitive_source = difficulty_motion_primitives[level]
        else:
            primitive_source = motion_primitives or tuple(joint_motion_library)
        primitive_names = tuple(name for name in primitive_source if name in joint_motion_library)
        return primitive_names if primitive_names else tuple(joint_motion_library)

    def _start_sequence(env_index: int, primitive_names: tuple[str, ...]) -> None:
        choice = primitive_names[int(torch.randint(0, len(primitive_names), (), device=device).item())]
        env._arm_joint_motion_kind[env_index] = choice
        env._arm_joint_motion_phase[env_index] = 0

    for env_id in env_ids:
        env_index = int(env_id.item())
        arm_difficulty = float(arm_difficulties[env_id].item())
        primitive_names = _primitive_names_for_difficulty(arm_difficulty)
        if not env._arm_joint_target_initialized[env_id] or not env._arm_joint_motion_kind[env_index]:
            _start_sequence(env_index, primitive_names)

        env_max_joint_change = max_joint_change
        env_motion_speed = motion_speed
        if max_joint_change_range is not None:
            low, high = float(max_joint_change_range[0]), float(max_joint_change_range[1])
            env_max_joint_change = low + (high - low) * arm_difficulty
        if motion_speed_range is not None:
            low, high = float(motion_speed_range[0]), float(motion_speed_range[1])
            env_motion_speed = low + (high - low) * arm_difficulty

        sequence = joint_motion_library[env._arm_joint_motion_kind[env_index]]
        phase = int(env._arm_joint_motion_phase[env_id].item())
        raw_goal = torch.tensor(sequence[phase % len(sequence)], device=device, dtype=torch.float32)

        current_target = env._arm_joint_target_pos[env_id]
        delta = raw_goal - current_target
        max_delta = torch.max(torch.abs(delta))
        if interpolate and max_delta > env_max_joint_change:
            raw_goal = current_target + delta / max_delta * env_max_joint_change

        if not interpolate:
            env._arm_joint_start_pos[env_id] = raw_goal
        elif env._arm_joint_target_initialized[env_id]:
            env._arm_joint_start_pos[env_id] = current_target.clone()
        else:
            env._arm_joint_start_pos[env_id] = art.data.joint_pos[env_id].index_select(0, joint_ids).clone()

        env._arm_joint_goal_pos[env_id] = raw_goal
        distance = torch.max(torch.abs(env._arm_joint_goal_pos[env_id] - env._arm_joint_start_pos[env_id]))
        env._arm_joint_trajectory_duration[env_id] = (
            torch.clamp(distance / max(float(env_motion_speed), 1.0e-6), min=1.0) if interpolate else 1.0
        )
        env._arm_joint_trajectory_progress[env_id] = 0.0 if interpolate else 1.0
        if not interpolate:
            env._arm_joint_target_pos[env_id] = raw_goal
        env._arm_joint_target_initialized[env_id] = True

        env._arm_joint_motion_phase[env_id] += 1
        if phase + 1 >= len(sequence):
            env._arm_joint_motion_kind[env_index] = ""


def continuous_arm_joint_tracking(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    apply_target: bool = True,
) -> None:
    """Track the stored Airbot joint-space target while leaving Go2 leg targets unchanged."""
    del env_ids

    device = env.device
    if not hasattr(env, "_arm_joint_target_initialized"):
        return

    art = env.scene[asset_cfg.name]
    if getattr(asset_cfg, "joint_ids", None):
        joint_ids = torch.as_tensor(list(asset_cfg.joint_ids), device=device, dtype=torch.long)
    else:
        joint_result = art.find_joints(asset_cfg.joint_names)
        joint_ids = torch.as_tensor(
            joint_result[0] if isinstance(joint_result, tuple) else joint_result, device=device, dtype=torch.long
        )
    if joint_ids.numel() == 0:
        return

    initialized_mask = env._arm_joint_target_initialized
    interpolate = bool(getattr(env, "_arm_joint_interpolate", True))
    if initialized_mask.any() and interpolate:
        env._arm_joint_trajectory_progress[initialized_mask] += env.step_dt
        t = torch.clamp(
            env._arm_joint_trajectory_progress[initialized_mask]
            / env._arm_joint_trajectory_duration[initialized_mask],
            0.0,
            1.0,
        )
        t_smooth = 3 * t**2 - 2 * t**3
        env._arm_joint_target_pos[initialized_mask] = (
            env._arm_joint_start_pos[initialized_mask] * (1 - t_smooth.unsqueeze(-1))
            + env._arm_joint_goal_pos[initialized_mask] * t_smooth.unsqueeze(-1)
        )
    elif initialized_mask.any():
        env._arm_joint_target_pos[initialized_mask] = env._arm_joint_goal_pos[initialized_mask]

    if not apply_target:
        return

    full_target = art.data.joint_pos.clone()
    if initialized_mask.any():
        for local_idx, joint_id in enumerate(joint_ids):
            full_target[initialized_mask, joint_id] = env._arm_joint_target_pos[initialized_mask, local_idx]
    art.set_joint_position_target(full_target)


def random_async_arm_joint_motion(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    joint_position_ranges: dict[str, tuple[float, float]],
    nominal_joint_pos: tuple[float, ...],
    safe_waypoints: tuple[tuple[float, ...], ...] = (),
    waypoint_probability: float = 0.65,
    waypoint_jitter_fraction: float = 0.08,
    max_joint_change_range: tuple[float, float] = (0.10, 0.45),
    trajectory_duration_range: tuple[float, float] = (0.35, 1.40),
    joint_velocity_limits: tuple[float, ...] = (),
    max_velocity_fraction: float = 0.85,
    velocity_fraction_range: tuple[float, float] | None = None,
    start_motion_enabled: bool = True,
    apply_target_when_disabled: bool = True,
    reset_to_nominal: bool = False,
    gripper_joint_names: tuple[str, ...] | None = None,
    gripper_open_pos: tuple[float, ...] = (),
    gripper_closed_pos: tuple[float, ...] = (),
    gripper_close_probability: float = 0.20,
) -> None:
    """Sample asynchronous D1 joint-space targets for low-level Go2 stabilization training.

    The sampler follows the dynamic-pick low-level idea: each arm joint receives an independent target, duration, and
    motion profile. Collision risk is controlled by conservative per-joint ranges plus optional known-good waypoints.
    """
    device = env.device
    num_envs = env.scene.num_envs

    if env_ids is None:
        env_ids = torch.arange(num_envs, device=device, dtype=torch.long)
    else:
        env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.long).flatten()
    if env_ids.numel() == 0:
        return

    art = env.scene[asset_cfg.name]
    if getattr(asset_cfg, "joint_ids", None):
        joint_ids = torch.as_tensor(list(asset_cfg.joint_ids), device=device, dtype=torch.long)
        joint_names = list(asset_cfg.joint_names)
    else:
        joint_result = art.find_joints(asset_cfg.joint_names, preserve_order=True)
        joint_ids = torch.as_tensor(joint_result[0], device=device, dtype=torch.long)
        joint_names = list(joint_result[1])
    if joint_ids.numel() == 0:
        return

    num_joints = int(joint_ids.numel())
    nominal = torch.tensor(nominal_joint_pos, device=device, dtype=torch.float32)
    if nominal.numel() != num_joints:
        raise ValueError(f"nominal_joint_pos must have {num_joints} values for joints {joint_names}.")
    if joint_velocity_limits and len(joint_velocity_limits) != num_joints:
        raise ValueError(
            f"joint_velocity_limits must have {num_joints} values for joints {joint_names}, "
            f"got {len(joint_velocity_limits)}."
        )
    velocity_fraction_low = float(max_velocity_fraction)
    velocity_fraction_high = float(max_velocity_fraction)
    if velocity_fraction_range is not None:
        if not joint_velocity_limits:
            raise ValueError("velocity_fraction_range requires joint_velocity_limits.")
        if len(velocity_fraction_range) != 2:
            raise ValueError("velocity_fraction_range must contain exactly two values.")
        velocity_fraction_low, velocity_fraction_high = (float(value) for value in velocity_fraction_range)
    if joint_velocity_limits and not 0.0 < velocity_fraction_low <= velocity_fraction_high <= 1.0:
        raise ValueError(
            "Arm velocity fractions must satisfy 0 < low <= high <= 1, got "
            f"({velocity_fraction_low}, {velocity_fraction_high})."
        )

    low = torch.empty(num_joints, device=device, dtype=torch.float32)
    high = torch.empty(num_joints, device=device, dtype=torch.float32)
    for index, joint_name in enumerate(joint_names):
        if joint_name not in joint_position_ranges:
            raise ValueError(f"Missing joint range for async arm joint '{joint_name}'.")
        low[index], high[index] = joint_position_ranges[joint_name]

    if not hasattr(env, "_arm_async_joint_target_pos"):
        current = art.data.joint_pos.index_select(dim=1, index=joint_ids).clone()
        env._arm_async_joint_target_pos = current
        env._arm_async_joint_start_pos = current.clone()
        env._arm_async_joint_goal_pos = current.clone()
        env._arm_async_joint_progress = torch.zeros((num_envs, num_joints), device=device, dtype=torch.float32)
        env._arm_async_joint_duration = torch.ones((num_envs, num_joints), device=device, dtype=torch.float32)
        env._arm_async_joint_mode = torch.zeros((num_envs, num_joints), device=device, dtype=torch.long)
        env._arm_async_joint_initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
    if not hasattr(env, "_arm_async_joint_velocity_cap_fraction"):
        env._arm_async_joint_velocity_cap_fraction = torch.zeros(
            (num_envs, num_joints), device=device, dtype=torch.float32
        )
        env._arm_async_joint_peak_velocity_fraction = torch.zeros_like(
            env._arm_async_joint_velocity_cap_fraction
        )

    gripper_enabled = gripper_joint_names is not None and len(gripper_joint_names) > 0
    if gripper_enabled and not hasattr(env, "_gripper_target_pos"):
        open_pos = gripper_open_pos or tuple(0.0 for _ in gripper_joint_names)
        env._gripper_target_pos = torch.tensor(open_pos, device=device, dtype=torch.float32).repeat(num_envs, 1)

    # Reset events run before PhysX refreshes articulation kinematics. Never seed a
    # new trajectory from art.data here: it can still describe the previous
    # episode and produces a visible carry-pose jump even at difficulty zero.
    if reset_to_nominal:
        env._arm_async_joint_start_pos[env_ids] = nominal
        env._arm_async_joint_goal_pos[env_ids] = nominal
        env._arm_async_joint_target_pos[env_ids] = nominal
        env._arm_async_joint_progress[env_ids] = 1.0
        env._arm_async_joint_duration[env_ids] = 1.0
        env._arm_async_joint_mode[env_ids] = 0
        env._arm_async_joint_velocity_cap_fraction[env_ids] = 0.0
        env._arm_async_joint_peak_velocity_fraction[env_ids] = 0.0
        env._arm_async_joint_initialized[env_ids] = True
        art.set_joint_position_target(
            nominal.repeat(env_ids.numel(), 1),
            joint_ids=joint_ids,
            env_ids=env_ids,
        )
        if gripper_enabled:
            open_pos = gripper_open_pos or tuple(0.0 for _ in gripper_joint_names)
            env._gripper_target_pos[env_ids] = torch.tensor(open_pos, device=device, dtype=torch.float32)

    motion_enabled = _as_per_env_tensor(
        getattr(env, "_loco_manip_arm_motion_enabled", start_motion_enabled),
        num_envs,
        device,
        torch.bool,
        "_loco_manip_arm_motion_enabled",
    )
    enabled_by_env = motion_enabled.index_select(0, env_ids)
    disabled_env_ids = env_ids[~enabled_by_env]
    if disabled_env_ids.numel() > 0:
        env._arm_async_joint_start_pos[disabled_env_ids] = art.data.joint_pos[disabled_env_ids].index_select(
            dim=1, index=joint_ids
        )
        env._arm_async_joint_goal_pos[disabled_env_ids] = nominal
        env._arm_async_joint_target_pos[disabled_env_ids] = nominal
        env._arm_async_joint_progress[disabled_env_ids] = 1.0
        env._arm_async_joint_duration[disabled_env_ids] = 1.0
        env._arm_async_joint_mode[disabled_env_ids] = 0
        env._arm_async_joint_velocity_cap_fraction[disabled_env_ids] = 0.0
        env._arm_async_joint_peak_velocity_fraction[disabled_env_ids] = 0.0
        env._arm_async_joint_initialized[disabled_env_ids] = True
        if apply_target_when_disabled:
            art.set_joint_position_target(
                nominal.repeat(disabled_env_ids.numel(), 1),
                joint_ids=joint_ids,
                env_ids=disabled_env_ids,
            )
        if gripper_enabled:
            open_pos = gripper_open_pos or tuple(0.0 for _ in gripper_joint_names)
            env._gripper_target_pos[disabled_env_ids] = torch.tensor(open_pos, device=device, dtype=torch.float32)

    env_ids = env_ids[enabled_by_env]
    if env_ids.numel() == 0:
        return

    arm_difficulties = _as_per_env_tensor(
        getattr(env, "_loco_manip_arm_motion_difficulty", 1.0),
        num_envs,
        device,
        torch.float32,
        "_loco_manip_arm_motion_difficulty",
    ).clamp(0.0, 1.0)
    difficulties = arm_difficulties.index_select(0, env_ids).unsqueeze(1)
    sample_low = nominal.unsqueeze(0) + (low - nominal).unsqueeze(0) * difficulties
    sample_high = nominal.unsqueeze(0) + (high - nominal).unsqueeze(0) * difficulties
    span = torch.clamp(sample_high - sample_low, min=1.0e-6)
    max_delta = float(max_joint_change_range[0]) + difficulties * (
        float(max_joint_change_range[1]) - float(max_joint_change_range[0])
    )

    n = env_ids.numel()
    use_waypoint = safe_waypoints and torch.rand(n, device=device) < float(waypoint_probability)
    goals = sample_low + torch.rand((n, num_joints), device=device) * span
    if safe_waypoints:
        waypoint_tensor = torch.tensor(safe_waypoints, device=device, dtype=torch.float32)
        waypoint_indices = torch.randint(0, waypoint_tensor.shape[0], (n,), device=device)
        waypoint_goals = waypoint_tensor[waypoint_indices]
        jitter = (torch.rand((n, num_joints), device=device) * 2.0 - 1.0) * span * float(waypoint_jitter_fraction)
        waypoint_goals = torch.clamp(waypoint_goals + jitter * difficulties, min=sample_low, max=sample_high)
        goals = torch.where(use_waypoint.unsqueeze(1), waypoint_goals, goals)

    current_target = env._arm_async_joint_target_pos[env_ids]
    delta = torch.clamp(goals - current_target, min=-max_delta, max=max_delta)
    goals = torch.clamp(current_target + delta, min=sample_low, max=sample_high)

    motion_mode = torch.randint(0, 2, (n, num_joints), device=device)
    duration_low, duration_high = trajectory_duration_range
    duration = float(duration_low) + torch.rand((n, num_joints), device=device) * (
        float(duration_high) - float(duration_low)
    )
    velocity_cap_fraction = torch.zeros_like(duration)
    planned_peak_fraction = torch.zeros_like(duration)
    if joint_velocity_limits:
        velocity_limits = torch.tensor(joint_velocity_limits, device=device, dtype=torch.float32).unsqueeze(0)
        if torch.any(velocity_limits <= 0.0):
            raise ValueError("joint_velocity_limits must contain only positive values.")
        velocity_cap_fraction = velocity_fraction_low + torch.rand((n, num_joints), device=device) * (
            velocity_fraction_high - velocity_fraction_low
        )
        # Smooth-step peaks at 1.5 times average trajectory velocity; linear
        # profiles peak at the average. Each joint samples its own hardware-limit
        # fraction, so a batch contains slow disturbances and near-limit motions.
        profile_factor = torch.where(motion_mode == 0, 1.0, 1.5)
        minimum_duration = profile_factor * torch.abs(delta) / (
            velocity_limits * velocity_cap_fraction
        )
        duration = torch.maximum(duration, minimum_duration)
        planned_peak_fraction = profile_factor * torch.abs(delta) / (
            velocity_limits * duration.clamp_min(torch.finfo(duration.dtype).eps)
        )

    env._arm_async_joint_start_pos[env_ids] = current_target
    env._arm_async_joint_goal_pos[env_ids] = goals
    env._arm_async_joint_progress[env_ids] = 0.0
    env._arm_async_joint_duration[env_ids] = duration
    env._arm_async_joint_mode[env_ids] = motion_mode
    env._arm_async_joint_velocity_cap_fraction[env_ids] = velocity_cap_fraction
    env._arm_async_joint_peak_velocity_fraction[env_ids] = planned_peak_fraction
    env._arm_async_joint_initialized[env_ids] = True

    if gripper_enabled:
        open_pos = torch.tensor(gripper_open_pos or tuple(0.0 for _ in gripper_joint_names), device=device)
        closed_pos = torch.tensor(gripper_closed_pos or tuple(0.0 for _ in gripper_joint_names), device=device)
        close_mask = torch.rand((n, 1), device=device) < float(gripper_close_probability)
        env._gripper_target_pos[env_ids] = torch.where(close_mask, closed_pos, open_pos)


def continuous_async_arm_joint_tracking(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    apply_target: bool = True,
    apply_gripper_target: bool = False,
    gripper_joint_names: tuple[str, ...] | list[str] = (),
) -> None:
    """Advance independent async arm joint trajectories and optionally apply them as joint targets."""
    del env_ids
    if not hasattr(env, "_arm_async_joint_initialized"):
        return

    device = env.device
    art = env.scene[asset_cfg.name]
    if getattr(asset_cfg, "joint_ids", None):
        joint_ids = torch.as_tensor(list(asset_cfg.joint_ids), device=device, dtype=torch.long)
    else:
        joint_result = art.find_joints(asset_cfg.joint_names, preserve_order=True)
        joint_ids = torch.as_tensor(joint_result[0], device=device, dtype=torch.long)
    if joint_ids.numel() == 0:
        return

    initialized = env._arm_async_joint_initialized
    if torch.any(initialized):
        env._arm_async_joint_progress[initialized] += env.step_dt
        t = torch.clamp(
            env._arm_async_joint_progress[initialized] / env._arm_async_joint_duration[initialized],
            0.0,
            1.0,
        )
        smooth_t = 3.0 * t**2 - 2.0 * t**3
        profile_t = torch.where(env._arm_async_joint_mode[initialized] == 0, t, smooth_t)
        env._arm_async_joint_target_pos[initialized] = (
            env._arm_async_joint_start_pos[initialized] * (1.0 - profile_t)
            + env._arm_async_joint_goal_pos[initialized] * profile_t
        )

    if not apply_target:
        return

    initialized_env_ids = torch.nonzero(initialized, as_tuple=False).flatten()
    if initialized_env_ids.numel() > 0:
        art.set_joint_position_target(
            env._arm_async_joint_target_pos[initialized_env_ids],
            joint_ids=joint_ids,
            env_ids=initialized_env_ids,
        )

    if apply_gripper_target and gripper_joint_names and hasattr(env, "_gripper_target_pos"):
        gripper_result = art.find_joints(gripper_joint_names, preserve_order=True)
        gripper_joint_ids = gripper_result[0] if isinstance(gripper_result, tuple) else gripper_result
        if len(gripper_joint_ids) != env._gripper_target_pos.shape[1]:
            raise ValueError(
                "gripper_joint_names and _gripper_target_pos must have matching widths: "
                f"{len(gripper_joint_ids)} != {env._gripper_target_pos.shape[1]}."
            )
        if initialized_env_ids.numel() > 0:
            art.set_joint_position_target(
                env._gripper_target_pos[initialized_env_ids],
                joint_ids=gripper_joint_ids,
                env_ids=initialized_env_ids,
            )


def random_arm_ik_motion(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    ee_pos_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    ik_controller_cfg: dict | None = None,
    visualize: bool = True,
    smooth_motion: bool = True,
    max_pos_change: float | None = 0.1,
    max_pos_change_range: tuple[float, float] | None = None,
    motion_speed: float = 0.10,
    motion_speed_range: tuple[float, float] | None = None,
    motion_primitives: tuple[str, ...] | None = None,
    global_workspace_probability: float = 0.0,
    interpolate: bool = True,
    start_motion_enabled: bool = True,
    min_arm_motion_difficulty: float = 0.0,
    preserve_current_orientation: bool = True,
    body_exclusion_box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (-0.32, 0.34),
        (-0.22, 0.22),
        (-0.02, 0.34),
    ),
    extra_exclusion_boxes: tuple[tuple[tuple[float, float], tuple[float, float], tuple[float, float]], ...] = (),
    body_clearance: float = 0.06,
    workspace_origin: tuple[float, float, float] | None = None,
    reach_range: tuple[float, float] | None = None,
    neutral_pos: tuple[float, float, float] = (0.42, 0.0, 0.34),
    deployment_ee_waypoints: tuple[tuple[float, float, float], ...] = (),
    reset_deployment: bool = False,
    min_workspace_fraction: float = 0.35,
    joint_position_ranges: dict[str, tuple[float, float]] | None = None,
    gripper_joint_names: tuple[str, ...] | None = None,
    gripper_open_pos: tuple[float, ...] = (),
    gripper_closed_pos: tuple[float, ...] = (),
    gripper_close_probability: float = 0.35,
):
    """Sample deployment-like, body-clear end-effector targets for an arm driven by differential IK."""
    device = env.device
    num_envs = env.scene.num_envs

    if env_ids is None:
        env_ids = torch.arange(num_envs, device=device, dtype=torch.long)
    else:
        env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.long).flatten()
    if env_ids.numel() == 0:
        return

    art = env.scene[asset_cfg.name]
    if getattr(asset_cfg, "joint_ids", None):
        joint_ids = torch.as_tensor(list(asset_cfg.joint_ids), device=device, dtype=torch.long)
    else:
        joint_result = art.find_joints(asset_cfg.joint_names)
        joint_ids = torch.as_tensor(
            joint_result[0] if isinstance(joint_result, tuple) else joint_result, device=device, dtype=torch.long
        )
    if getattr(asset_cfg, "body_ids", None):
        body_ids = torch.as_tensor(list(asset_cfg.body_ids), device=device, dtype=torch.long)
    else:
        body_result = art.find_bodies(asset_cfg.body_names)
        body_ids = torch.as_tensor(
            body_result[0] if isinstance(body_result, tuple) else body_result, device=device, dtype=torch.long
        )
    if joint_ids.numel() == 0 or body_ids.numel() == 0:
        return

    ee_body_id = int(body_ids[0].item())

    if not hasattr(env, "_arm_ik_controller"):
        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        ik_cfg_params = ik_controller_cfg or {}
        command_type = ik_cfg_params.get("command_type", "position")
        ik_method = ik_cfg_params.get("ik_method", "dls")
        ik_params = None
        if ik_method == "dls" and "dls_damping" in ik_cfg_params:
            ik_params = {"lambda_val": ik_cfg_params["dls_damping"]}
        ik_cfg = DifferentialIKControllerCfg(
            command_type=command_type,
            use_relative_mode=False,
            ik_method=ik_method,
            ik_params=ik_params,
        )
        env._arm_ik_controller = DifferentialIKController(ik_cfg, num_envs=num_envs, device=device)
        env._arm_ik_command_type = command_type
    if not hasattr(env, "_arm_ik_joint_target_pos"):
        env._arm_ik_joint_target_pos = art.data.joint_pos.index_select(dim=1, index=joint_ids).clone()
    if joint_position_ranges is not None:
        resolved_joint_names = list(asset_cfg.joint_names)
        if len(resolved_joint_names) != joint_ids.numel() or any(
            name not in joint_position_ranges for name in resolved_joint_names
        ):
            _, resolved_joint_names = art.find_joints(asset_cfg.joint_names, preserve_order=True)
        env._arm_ik_joint_lower = torch.tensor(
            [joint_position_ranges[name][0] for name in resolved_joint_names],
            device=device,
            dtype=torch.float32,
        )
        env._arm_ik_joint_upper = torch.tensor(
            [joint_position_ranges[name][1] for name in resolved_joint_names],
            device=device,
            dtype=torch.float32,
        )

    if not hasattr(env, "_arm_ee_target_pos"):
        env._arm_ee_target_pos = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
        env._arm_ee_target_quat = torch.zeros((num_envs, 4), device=device, dtype=torch.float32)
        env._arm_ee_target_quat[:, 0] = 1.0
        env._arm_ee_target_initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        env._arm_ee_start_pos = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
        env._arm_ee_goal_pos = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
        env._arm_trajectory_progress = torch.zeros(num_envs, device=device, dtype=torch.float32)
        env._arm_trajectory_duration = torch.zeros(num_envs, device=device, dtype=torch.float32)
        env._arm_motion_kind = torch.full((num_envs,), -1, device=device, dtype=torch.long)
        env._arm_motion_phase = torch.zeros(num_envs, device=device, dtype=torch.long)
        env._arm_workspace_sample_phase = torch.zeros(num_envs, device=device, dtype=torch.long)
        env._arm_motion_pick_pos = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
        env._arm_motion_place_pos = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
        env._arm_motion_gripper_command = torch.full((num_envs,), -1, device=device, dtype=torch.int8)
    if not hasattr(env, "_arm_motion_gripper_command"):
        env._arm_motion_gripper_command = torch.full((num_envs,), -1, device=device, dtype=torch.int8)
    if not hasattr(env, "_arm_deployment_phase"):
        env._arm_deployment_phase = torch.zeros(num_envs, device=device, dtype=torch.long)
    if reset_deployment:
        env._arm_deployment_phase[env_ids] = 0
    env._arm_ik_interpolate = bool(interpolate)

    gripper_enabled = gripper_joint_names is not None and len(gripper_joint_names) > 0
    if gripper_enabled and not hasattr(env, "_gripper_target_pos"):
        open_pos = gripper_open_pos or tuple(0.0 for _ in gripper_joint_names)
        env._gripper_target_pos = torch.tensor(open_pos, device=device, dtype=torch.float32).repeat(num_envs, 1)

    from isaaclab.utils.math import subtract_frame_transforms

    ee_pose_w = art.data.body_pose_w[:, ee_body_id]
    root_pose_w = art.data.root_pose_w
    ee_pos_b_current, ee_quat_b_current = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )

    arm_difficulties = _as_per_env_tensor(
        getattr(env, "_loco_manip_arm_motion_difficulty", 1.0),
        num_envs,
        device,
        torch.float32,
        "_loco_manip_arm_motion_difficulty",
    ).clamp(0.0, 1.0)

    motion_enabled = _as_per_env_tensor(
        getattr(env, "_loco_manip_arm_motion_enabled", start_motion_enabled),
        num_envs,
        device,
        torch.bool,
        "_loco_manip_arm_motion_enabled",
    )
    enabled_by_env = motion_enabled.index_select(0, env_ids)
    if float(min_arm_motion_difficulty) > 0.0:
        difficulty_enabled = arm_difficulties.index_select(0, env_ids) > float(min_arm_motion_difficulty)
        enabled_by_env = enabled_by_env & difficulty_enabled

    disabled_env_ids = env_ids[~enabled_by_env]
    if disabled_env_ids.numel() > 0:
        env._arm_ee_target_pos[disabled_env_ids] = ee_pos_b_current.index_select(0, disabled_env_ids)
        env._arm_ee_target_quat[disabled_env_ids] = ee_quat_b_current.index_select(0, disabled_env_ids)
        env._arm_ee_start_pos[disabled_env_ids] = env._arm_ee_target_pos[disabled_env_ids]
        env._arm_ee_goal_pos[disabled_env_ids] = env._arm_ee_target_pos[disabled_env_ids]
        env._arm_trajectory_progress[disabled_env_ids] = 0.0
        env._arm_trajectory_duration[disabled_env_ids] = 1.0
        env._arm_ee_target_initialized[disabled_env_ids] = True
        env._arm_motion_kind[disabled_env_ids] = -1
        env._arm_motion_phase[disabled_env_ids] = 0
        env._arm_deployment_phase[disabled_env_ids] = 0
        env._arm_motion_gripper_command[disabled_env_ids] = 0
        if gripper_enabled:
            open_pos = gripper_open_pos or tuple(0.0 for _ in gripper_joint_names)
            env._gripper_target_pos[disabled_env_ids] = torch.tensor(open_pos, device=device, dtype=torch.float32)

    env_ids = env_ids[enabled_by_env]
    if env_ids.numel() == 0:
        return

    if not interpolate:
        smooth_motion = False

    primitive_names = motion_primitives or ("workspace", "pick_place", "reach", "sweep", "stow")
    primitive_ids = {
        name: idx
        for idx, name in enumerate(("pick_place", "reach", "sweep", "stow", "workspace"))
        if name in primitive_names
    }
    if not primitive_ids:
        primitive_ids = {"workspace": 4}
    id_to_name = {idx: name for name, idx in primitive_ids.items()}

    def sample_training_goal(env_id: torch.Tensor, env_ee_pos_range) -> torch.Tensor:
        """Mix broad stratified coverage with task-shaped multi-waypoint motions."""
        motion_active = int(env._arm_motion_kind[env_id].item()) >= 0
        if not motion_active and torch.rand((), device=device) < float(global_workspace_probability):
            env._arm_motion_kind[env_id] = -1
            env._arm_motion_phase[env_id] = 0
            env._arm_motion_gripper_command[env_id] = -1
            env_index = int(env_id.item())
            workspace_phase = int(env._arm_workspace_sample_phase[env_id].item())
            target = _sample_stratified_arm_ee_point(
                device,
                env_index,
                workspace_phase,
                env_ee_pos_range,
                body_exclusion_box,
                body_clearance,
                extra_exclusion_boxes=extra_exclusion_boxes,
                workspace_origin=workspace_origin,
                reach_range=reach_range,
            )
            env._arm_workspace_sample_phase[env_id] += 1
            return target
        return _next_arm_motion_target(
            env,
            env_id,
            id_to_name,
            env_ee_pos_range,
            body_exclusion_box,
            body_clearance,
            extra_exclusion_boxes,
            workspace_origin,
            reach_range,
        )

    n = env_ids.numel()
    ee_pos_sampled = torch.empty((n, 3), device=device, dtype=torch.float32)
    deployment_mask = torch.zeros(n, device=device, dtype=torch.bool)
    for i, env_id in enumerate(env_ids):
        phase = int(env._arm_deployment_phase[env_id].item())
        target = _arm_deployment_waypoint(deployment_ee_waypoints, phase, device)
        if target is None:
            continue
        ee_pos_sampled[i] = target
        deployment_mask[i] = True
        env._arm_deployment_phase[env_id] += 1
        env._arm_motion_kind[env_id] = -1
        env._arm_motion_phase[env_id] = 0
        env._arm_motion_gripper_command[env_id] = 0

    if smooth_motion:
        for i, env_id in enumerate(env_ids):
            if deployment_mask[i]:
                continue
            arm_difficulty = float(arm_difficulties[env_id].item())
            env_max_pos_change = max_pos_change
            if max_pos_change_range is not None:
                low, high = float(max_pos_change_range[0]), float(max_pos_change_range[1])
                env_max_pos_change = low + (high - low) * arm_difficulty
            env_ee_pos_range = _scale_arm_ee_range(
                ee_pos_range, arm_difficulty, neutral_pos, min_workspace_fraction
            )
            if env._arm_ee_target_initialized[env_id]:
                target = sample_training_goal(env_id, env_ee_pos_range)
                delta = target - env._arm_ee_target_pos[env_id]
                distance = torch.linalg.vector_norm(delta)
                if env_max_pos_change is not None and distance > env_max_pos_change:
                    target = env._arm_ee_target_pos[env_id] + delta / distance * env_max_pos_change
                ee_pos_sampled[i] = _clamp_arm_ee_target(
                    target,
                    env_ee_pos_range,
                    body_exclusion_box,
                    body_clearance,
                    extra_exclusion_boxes,
                    workspace_origin,
                    reach_range,
                )
            else:
                current = ee_pos_b_current[env_id].clone()
                ee_pos_sampled[i] = _clamp_arm_ee_target(
                    current,
                    env_ee_pos_range,
                    body_exclusion_box,
                    body_clearance,
                    extra_exclusion_boxes,
                    workspace_origin,
                    reach_range,
                )
    else:
        for i, env_id in enumerate(env_ids):
            if deployment_mask[i]:
                continue
            arm_difficulty = float(arm_difficulties[env_id].item())
            env_ee_pos_range = _scale_arm_ee_range(
                ee_pos_range, arm_difficulty, neutral_pos, min_workspace_fraction
            )
            ee_pos_sampled[i] = sample_training_goal(env_id, env_ee_pos_range)

    if preserve_current_orientation:
        ee_quat_sampled = ee_quat_b_current.index_select(0, env_ids).clone()
    else:
        ee_quat_sampled = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=device).repeat(n, 1)

    for i, env_id in enumerate(env_ids):
        arm_difficulty = float(arm_difficulties[env_id].item())
        env_motion_speed = motion_speed
        if motion_speed_range is not None:
            low, high = float(motion_speed_range[0]), float(motion_speed_range[1])
            env_motion_speed = low + (high - low) * arm_difficulty
        if not interpolate:
            env._arm_ee_start_pos[env_id] = ee_pos_sampled[i]
        elif env._arm_ee_target_initialized[env_id]:
            env._arm_ee_start_pos[env_id] = env._arm_ee_target_pos[env_id].clone()
        else:
            env._arm_ee_start_pos[env_id] = ee_pos_b_current[env_id].clone()
            env._arm_ee_target_quat[env_id] = ee_quat_b_current[env_id].clone()
        env._arm_ee_goal_pos[env_id] = ee_pos_sampled[i]
        distance = torch.norm(env._arm_ee_goal_pos[env_id] - env._arm_ee_start_pos[env_id])
        env._arm_trajectory_duration[env_id] = (
            torch.clamp(distance / max(float(env_motion_speed), 1.0e-6), min=1.0) if interpolate else 1.0
        )
        env._arm_trajectory_progress[env_id] = 0.0 if interpolate else 1.0

    env._arm_ee_target_pos[env_ids] = ee_pos_sampled
    env._arm_ee_target_quat[env_ids] = ee_quat_sampled
    env._arm_ee_target_initialized[env_ids] = True

    if gripper_enabled:
        open_pos = gripper_open_pos or tuple(0.0 for _ in gripper_joint_names)
        closed_pos = gripper_closed_pos or tuple(0.0 for _ in gripper_joint_names)
        open_target = torch.tensor(open_pos, device=device, dtype=torch.float32)
        closed_target = torch.tensor(closed_pos, device=device, dtype=torch.float32)
        close_mask = torch.rand((env_ids.numel(), 1), device=device) < float(gripper_close_probability)
        phase_command = env._arm_motion_gripper_command.index_select(0, env_ids).unsqueeze(1)
        close_mask = torch.where(phase_command >= 0, phase_command.bool(), close_mask)
        env._gripper_target_pos[env_ids] = torch.where(close_mask, closed_target, open_target)

    if visualize:
        if not hasattr(env, "_arm_ee_visualizer_current"):
            from isaaclab.markers import VisualizationMarkers
            from isaaclab.markers.config import FRAME_MARKER_CFG

            cfg_cur = FRAME_MARKER_CFG.replace()
            cfg_cur.markers["frame"].scale = (0.10, 0.10, 0.10)
            env._arm_ee_visualizer_current = VisualizationMarkers(
                cfg_cur.replace(prim_path="/Visuals/arm_ee_current")
            )

            cfg_des = FRAME_MARKER_CFG.replace()
            cfg_des.markers["frame"].scale = (0.12, 0.12, 0.12)
            env._arm_ee_visualizer_desired = VisualizationMarkers(
                cfg_des.replace(prim_path="/Visuals/arm_ee_desired")
            )
            env._arm_ee_initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)

        from isaaclab.utils.math import combine_frame_transforms

        env._arm_ee_initialized[env_ids] = True
        root_pose_w = art.data.root_pose_w
        ee_pos_des_w, ee_quat_des_w = combine_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            env._arm_ee_target_pos,
            env._arm_ee_target_quat,
        )
        ee_state_w = art.data.body_state_w[:, ee_body_id, 0:7]
        valid_mask = env._arm_ee_initialized
        ee_pos_cur_vis = ee_state_w[:, 0:3].clone()
        ee_quat_cur_vis = ee_state_w[:, 3:7].clone()
        ee_pos_des_vis = ee_pos_des_w.clone()
        ee_quat_des_vis = ee_quat_des_w.clone()
        ee_pos_cur_vis[~valid_mask, 2] = -1000.0
        ee_pos_des_vis[~valid_mask, 2] = -1000.0

        env._arm_ee_visualizer_current.visualize(
            translations=ee_pos_cur_vis,
            orientations=ee_quat_cur_vis,
        )
        env._arm_ee_visualizer_desired.visualize(
            translations=ee_pos_des_vis,
            orientations=ee_quat_des_vis,
        )


def hold_arm_ee_command_at_current_pose(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    gripper_joint_names: tuple[str, ...] | list[str] | None = None,
    gripper_open_pos: tuple[float, ...] | None = (0.0, 0.0),
) -> None:
    """Initialize task-space arm command buffers to the current Link6 pose."""
    device = env.device
    num_envs = env.scene.num_envs

    if env_ids is None:
        env_ids = torch.arange(num_envs, device=device, dtype=torch.long)
    else:
        env_ids = torch.as_tensor(env_ids, device=device, dtype=torch.long).flatten()
    if env_ids.numel() == 0:
        return

    art = env.scene[asset_cfg.name]
    if getattr(asset_cfg, "body_ids", None):
        body_ids = torch.as_tensor(list(asset_cfg.body_ids), device=device, dtype=torch.long)
    else:
        body_result = art.find_bodies(asset_cfg.body_names)
        body_ids = torch.as_tensor(
            body_result[0] if isinstance(body_result, tuple) else body_result, device=device, dtype=torch.long
        )
    if body_ids.numel() == 0:
        return

    ee_body_id = int(body_ids[0].item())
    if not hasattr(env, "_arm_ee_target_pos"):
        env._arm_ee_target_pos = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
        env._arm_ee_target_quat = torch.zeros((num_envs, 4), device=device, dtype=torch.float32)
        env._arm_ee_target_quat[:, 0] = 1.0
        env._arm_ee_target_initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        env._arm_ee_start_pos = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
        env._arm_ee_goal_pos = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
        env._arm_trajectory_progress = torch.zeros(num_envs, device=device, dtype=torch.float32)
        env._arm_trajectory_duration = torch.ones(num_envs, device=device, dtype=torch.float32)

    from isaaclab.utils.math import subtract_frame_transforms

    ee_pose_w = art.data.body_pose_w[:, ee_body_id]
    root_pose_w = art.data.root_pose_w
    ee_pos_b_current, ee_quat_b_current = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    env._arm_ee_target_pos[env_ids] = ee_pos_b_current.index_select(0, env_ids)
    env._arm_ee_target_quat[env_ids] = ee_quat_b_current.index_select(0, env_ids)
    env._arm_ee_start_pos[env_ids] = env._arm_ee_target_pos[env_ids]
    env._arm_ee_goal_pos[env_ids] = env._arm_ee_target_pos[env_ids]
    env._arm_trajectory_progress[env_ids] = 1.0
    env._arm_trajectory_duration[env_ids] = 1.0
    env._arm_ee_target_initialized[env_ids] = True

    if gripper_joint_names:
        if not hasattr(env, "_gripper_target_pos"):
            open_pos = gripper_open_pos or tuple(0.0 for _ in gripper_joint_names)
            env._gripper_target_pos = torch.tensor(open_pos, device=device, dtype=torch.float32).repeat(num_envs, 1)


def continuous_arm_ik_tracking(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    asset_cfg: SceneEntityCfg,
    apply_target: bool = True,
    apply_gripper_target: bool = False,
    gripper_joint_names: tuple[str, ...] | list[str] = (),
):
    """Track the stored arm end-effector target with differential IK.

    When ``apply_target`` is false, the computed joint target is still stored on
    the environment so a learned policy can use it as an IK hint.
    """
    device = env.device
    num_envs = env.scene.num_envs

    if not hasattr(env, "_arm_ee_target_initialized"):
        return

    art = env.scene[asset_cfg.name]
    if getattr(asset_cfg, "joint_ids", None):
        joint_ids = torch.as_tensor(list(asset_cfg.joint_ids), device=device, dtype=torch.long)
    else:
        joint_result = art.find_joints(asset_cfg.joint_names)
        joint_ids = torch.as_tensor(
            joint_result[0] if isinstance(joint_result, tuple) else joint_result, device=device, dtype=torch.long
        )
    if getattr(asset_cfg, "body_ids", None):
        body_ids = torch.as_tensor(list(asset_cfg.body_ids), device=device, dtype=torch.long)
    else:
        body_result = art.find_bodies(asset_cfg.body_names)
        body_ids = torch.as_tensor(
            body_result[0] if isinstance(body_result, tuple) else body_result, device=device, dtype=torch.long
        )
    if joint_ids.numel() == 0 or body_ids.numel() == 0:
        return

    ee_body_id = int(body_ids[0].item())
    ee_jacobi_idx = max(ee_body_id - 1, 0) if getattr(art, "is_fixed_base", True) else ee_body_id

    if not hasattr(env, "_arm_ik_controller"):
        return
    ik_controller = env._arm_ik_controller
    if not hasattr(env, "_arm_ik_joint_target_pos"):
        env._arm_ik_joint_target_pos = art.data.joint_pos.index_select(dim=1, index=joint_ids).clone()

    initialized_mask = env._arm_ee_target_initialized
    interpolate = bool(getattr(env, "_arm_ik_interpolate", True))
    if initialized_mask.any() and interpolate:
        env._arm_trajectory_progress[initialized_mask] += env.step_dt
        t = torch.clamp(
            env._arm_trajectory_progress[initialized_mask] / env._arm_trajectory_duration[initialized_mask],
            0.0,
            1.0,
        )
        t_smooth = 3 * t**2 - 2 * t**3
        interpolated_pos = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
        interpolated_pos[initialized_mask] = (
            env._arm_ee_start_pos[initialized_mask] * (1 - t_smooth.unsqueeze(-1))
            + env._arm_ee_goal_pos[initialized_mask] * t_smooth.unsqueeze(-1)
        )
        env._arm_ee_target_pos[initialized_mask] = interpolated_pos[initialized_mask]
    elif initialized_mask.any():
        env._arm_ee_target_pos[initialized_mask] = env._arm_ee_goal_pos[initialized_mask]

    if getattr(env, "_arm_ik_command_type", "pose") == "position":
        # Isaac Lab 5.1 requires a quaternion for position commands even though
        # it is used only for the controller's desired-frame visualization.
        # The position-mode compute path still uses only the translational
        # Jacobian and does not constrain wrist orientation.
        ik_controller.set_command(env._arm_ee_target_pos, ee_quat=env._arm_ee_target_quat)
    else:
        ik_controller.set_command(torch.cat([env._arm_ee_target_pos, env._arm_ee_target_quat], dim=1))

    full_jac = art.root_physx_view.get_jacobians()
    jacobian_w = full_jac[:, ee_jacobi_idx, :, :].index_select(dim=2, index=joint_ids)
    ee_pose_w = art.data.body_pose_w[:, ee_body_id]
    root_pose_w = art.data.root_pose_w
    joint_pos_all = art.data.joint_pos
    joint_pos_arm = joint_pos_all.index_select(dim=1, index=joint_ids)

    from isaaclab.utils.math import matrix_from_quat, quat_inv, subtract_frame_transforms

    ee_pos_b, ee_quat_b = subtract_frame_transforms(
        root_pose_w[:, 0:3], root_pose_w[:, 3:7], ee_pose_w[:, 0:3], ee_pose_w[:, 3:7]
    )
    root_rotation_w_to_b = matrix_from_quat(quat_inv(root_pose_w[:, 3:7]))
    jacobian = _rotate_jacobian_to_root_frame(jacobian_w, root_rotation_w_to_b)
    joint_pos_des_arm = ik_controller.compute(ee_pos_b, ee_quat_b, jacobian, joint_pos_arm)
    if hasattr(env, "_arm_ik_joint_lower") and hasattr(env, "_arm_ik_joint_upper"):
        joint_pos_des_arm = torch.clamp(
            joint_pos_des_arm,
            min=env._arm_ik_joint_lower.unsqueeze(0),
            max=env._arm_ik_joint_upper.unsqueeze(0),
        )
    if initialized_mask.any():
        env._arm_ik_joint_target_pos[initialized_mask] = joint_pos_des_arm[initialized_mask]

    if not apply_target:
        return

    full_target = joint_pos_all.clone()
    if initialized_mask.any():
        for i, joint_id in enumerate(joint_ids):
            full_target[initialized_mask, joint_id] = joint_pos_des_arm[initialized_mask, i]

    if apply_gripper_target and gripper_joint_names and hasattr(env, "_gripper_target_pos"):
        gripper_result = art.find_joints(gripper_joint_names, preserve_order=True)
        gripper_joint_ids = gripper_result[0] if isinstance(gripper_result, tuple) else gripper_result
        if len(gripper_joint_ids) != env._gripper_target_pos.shape[1]:
            raise ValueError(
                "gripper_joint_names and _gripper_target_pos must have matching widths: "
                f"{len(gripper_joint_ids)} != {env._gripper_target_pos.shape[1]}."
            )
        for i, joint_id in enumerate(gripper_joint_ids):
            full_target[:, joint_id] = env._gripper_target_pos[:, i]

    art.set_joint_position_target(full_target)


def initialize_hidden_friction_grid(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    grid_size: int = 8,
    tile_size: float = 0.5,
    friction_range: tuple[float, float] = (0.15, 1.2),
) -> None:
    """Sample a per-env hidden friction tile grid stored on the environment object.

    The grid is used for privileged critic observations and quadrant PhysX patch materials.
    """
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=env.device)
    grid_size = int(grid_size)
    low, high = float(friction_range[0]), float(friction_range[1])
    grid = torch.empty((env.scene.num_envs, grid_size, grid_size), device=env.device)
    grid[env_ids] = torch.rand((len(env_ids), grid_size, grid_size), device=env.device) * (high - low) + low
    if not hasattr(env, "_hidden_friction_grid") or env._hidden_friction_grid is None:
        env._hidden_friction_grid = grid
    else:
        env._hidden_friction_grid[env_ids] = grid[env_ids]
    env._hidden_friction_tile_size = float(tile_size)


def apply_hidden_friction_patch_materials(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    patch_asset_names: tuple[str, ...] | list[str],
    dynamic_friction_scale: float = 0.8,
) -> None:
    """Bind hidden friction grid quadrants to invisible PhysX patch colliders."""
    grid = getattr(env, "_hidden_friction_grid", None)
    if grid is None:
        return

    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device="cpu")
    else:
        env_ids = env_ids.cpu()

    grid_size = int(grid.shape[-1])
    half = max(1, grid_size // 2)
    quadrant_slices = (
        (slice(None), slice(0, half), slice(0, half)),
        (slice(None), slice(0, half), slice(half, grid_size)),
        (slice(None), slice(half, grid_size), slice(0, half)),
        (slice(None), slice(half, grid_size), slice(half, grid_size)),
    )

    for patch_name, grid_slice in zip(patch_asset_names, quadrant_slices):
        asset: RigidObject = env.scene[patch_name]
        static = grid[grid_slice].mean(dim=(1, 2))[env_ids].cpu()
        dynamic = (static * float(dynamic_friction_scale)).clamp(min=0.0)
        restitution = torch.zeros_like(static)

        materials = asset.root_physx_view.get_material_properties()
        patch_materials = torch.stack((static, dynamic, restitution), dim=-1).unsqueeze(1)
        patch_materials = patch_materials.expand(-1, materials.shape[1], -1)
        materials[env_ids] = patch_materials
        asset.root_physx_view.set_material_properties(materials, env_ids)
