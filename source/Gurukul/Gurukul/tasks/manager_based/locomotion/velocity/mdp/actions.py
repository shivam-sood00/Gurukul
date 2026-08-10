from __future__ import annotations

import math
import os
import re
from collections.abc import Sequence
from typing import Literal

import torch

from isaaclab.assets.articulation import Articulation
from isaaclab.envs.mdp import (
    BinaryJointPositionAction,
    BinaryJointPositionActionCfg,
    JointPositionAction,
    JointPositionActionCfg,
)
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass


def _slew_limit_normalized_actions(
    previous: torch.Tensor,
    target: torch.Tensor,
    rate_limits: torch.Tensor,
    step_dt: float,
) -> torch.Tensor:
    """Move normalized commands toward a target without exceeding per-second rates."""
    max_delta = rate_limits.unsqueeze(0) * float(step_dt)
    delta = torch.clamp(target - previous, min=-max_delta, max=max_delta)
    return previous + delta


class PositiveBinaryJointPositionAction(BinaryJointPositionAction):
    """Binary joint target where a positive policy logit means close."""

    cfg: PositiveBinaryJointPositionActionCfg

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions
        close_mask = actions > float(self.cfg.threshold)
        self._processed_actions = torch.where(close_mask, self._close_command, self._open_command)
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions,
                min=self._clip[:, :, 0],
                max=self._clip[:, :, 1],
            )


@configclass
class PositiveBinaryJointPositionActionCfg(BinaryJointPositionActionCfg):
    """Configuration for a 0=open, 1=close manipulation gripper command."""

    class_type: type = PositiveBinaryJointPositionAction
    threshold: float = 0.0


class OnlineArmDecapJointPositionAction(JointPositionAction):
    """Joint position action with an online decaying arm prior."""

    cfg: OnlineArmDecapJointPositionActionCfg

    def __init__(self, cfg: OnlineArmDecapJointPositionActionCfg, env):
        super().__init__(cfg, env)
        self._decap_lambda = torch.full((self.num_envs, 1), float(self.cfg.decap_lambda_start), device=self.device)
        self._decap_steps_per_iteration = max(1, int(self.cfg.decap_steps_per_iteration))
        self._decap_resume_step_offset = self._iterations_to_steps(self.cfg.decap_resume_iteration)
        self._decap_warmup_steps, self._decap_decay_start_step, self._decap_decay_end_step = (
            self._resolve_decap_schedule()
        )

        decap_joint_ids, decap_joint_names = self._asset.find_joints(
            self.cfg.decap_joint_names,
            preserve_order=self.cfg.decap_preserve_order,
        )
        action_name_to_index = {joint_name: index for index, joint_name in enumerate(self._joint_names)}
        missing = [joint_name for joint_name in decap_joint_names if joint_name not in action_name_to_index]
        if missing:
            raise ValueError(f"DecAP joints {missing} are not included in action joints: {self._joint_names}")

        self._decap_action_indices = torch.tensor(
            [action_name_to_index[joint_name] for joint_name in decap_joint_names],
            dtype=torch.long,
            device=self.device,
        )
        self._decap_robot_joint_ids = torch.tensor(decap_joint_ids, dtype=torch.long, device=self.device)

    @property
    def decap_lambda(self) -> torch.Tensor:
        return self._decap_lambda

    def _iterations_to_steps(self, iterations: int | float) -> int:
        return max(0, int(iterations)) * self._decap_steps_per_iteration

    def _current_decap_step(self) -> int:
        return self._decap_resume_step_offset + max(int(self._env.common_step_counter), 0)

    def _resolve_decap_schedule(self) -> tuple[int, int, int]:
        if self.cfg.decap_warmup_iterations is None:
            warmup_steps = max(0, int(self.cfg.decap_warmup_steps))
        else:
            warmup_steps = self._iterations_to_steps(self.cfg.decap_warmup_iterations)

        if self.cfg.decap_decay_start_iteration is None:
            decay_start_step = warmup_steps + max(0, int(self.cfg.decap_decay_start_step))
        else:
            decay_start_step = max(warmup_steps, self._iterations_to_steps(self.cfg.decap_decay_start_iteration))

        if self.cfg.decap_decay_iterations is not None:
            decay_end_step = decay_start_step + max(1, int(self.cfg.decap_decay_iterations)) * (
                self._decap_steps_per_iteration
            )
        elif self.cfg.decap_decay_end_iteration is not None:
            decay_end_step = self._iterations_to_steps(self.cfg.decap_decay_end_iteration)
        elif self.cfg.decap_decay_end_step is not None:
            decay_end_step = warmup_steps + max(0, int(self.cfg.decap_decay_end_step))
        else:
            decay_end_step = decay_start_step + 1

        if decay_end_step <= decay_start_step:
            decay_end_step = decay_start_step + 1

        return warmup_steps, decay_start_step, decay_end_step

    def _compute_decap_decay_counter(self) -> float:
        step = self._current_decap_step()
        if step <= self._decap_decay_start_step:
            return 0.0
        return float(step - self._decap_decay_start_step)

    def _compute_decap_progress(self) -> float:
        step = self._current_decap_step()
        if step <= self._decap_decay_start_step:
            return 0.0
        if step >= self._decap_decay_end_step:
            return 1.0
        return (step - self._decap_decay_start_step) / float(
            self._decap_decay_end_step - self._decap_decay_start_step
        )

    def _compute_decap_lambda(self, progress: float | None = None) -> float:
        start = float(self.cfg.decap_lambda_start)
        end = float(self.cfg.decap_lambda_end)
        decay_type = self.cfg.decap_decay_type

        if decay_type == "constant":
            return start

        if decay_type == "exp":
            gamma = float(self.cfg.decap_exp_gamma)
            k = max(float(self.cfg.decap_exp_k), 1.0e-6)
            if gamma <= 0.0:
                raise ValueError(f"decap_exp_gamma must be > 0, got {gamma}.")
            value = start * (gamma ** (self._compute_decap_decay_counter() / k))
            low = min(start, end)
            high = max(start, end)
            return float(max(low, min(high, value)))

        if progress is None:
            progress = self._compute_decap_progress()

        if decay_type == "linear":
            value = start + progress * (end - start)
        elif decay_type == "cosine":
            value = end + 0.5 * (start - end) * (1.0 + math.cos(math.pi * progress))
        else:
            raise ValueError(f"Unsupported decap decay type: {decay_type}")

        low = min(start, end)
        high = max(start, end)
        return float(max(low, min(high, value)))

    def process_actions(self, actions: torch.Tensor):
        super().process_actions(actions)

        decap_decay_counter = 0.0
        if self.cfg.decap_decay_type == "exp":
            decap_lambda = self._compute_decap_lambda(0.0)
            decap_decay_counter = self._compute_decap_decay_counter()
        else:
            decap_lambda = self._compute_decap_lambda(self._compute_decap_progress())
        self._decap_lambda.fill_(decap_lambda)

        arm_target = getattr(self._env, self.cfg.decap_target_attr, None)
        if arm_target is not None:
            if arm_target.shape[1] != self._decap_action_indices.numel():
                raise RuntimeError(
                    f"Online DecAP target '{self.cfg.decap_target_attr}' has width {arm_target.shape[1]}, expected "
                    f"{self._decap_action_indices.numel()} for joints {self.cfg.decap_joint_names}."
                )
            arm_current = self._asset.data.joint_pos[:, self._decap_robot_joint_ids]
            prior_delta = decap_lambda * (arm_target - arm_current)

            if self.cfg.decap_prior_only:
                self._processed_actions[:, self._decap_action_indices] = arm_current + prior_delta
            else:
                self._processed_actions[:, self._decap_action_indices] = (
                    self._processed_actions[:, self._decap_action_indices] + prior_delta
                )

            if self.cfg.clip is not None:
                self._processed_actions = torch.clamp(
                    self._processed_actions,
                    min=self._clip[:, :, 0],
                    max=self._clip[:, :, 1],
                )

        if isinstance(getattr(self._env, "extras", None), dict):
            self._env.extras["decap_factor"] = decap_lambda
            log_dict = self._env.extras.setdefault("log", {})
            if isinstance(log_dict, dict):
                log_dict["Metrics/decap/factor"] = decap_lambda
                log_dict["Metrics/decap/arm_factor"] = decap_lambda
                if self.cfg.decap_decay_type == "exp":
                    log_dict["Metrics/decap/exp_decay_counter"] = decap_decay_counter


@configclass
class OnlineArmDecapJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for online APEX-style decaying arm action prior."""

    class_type: type = OnlineArmDecapJointPositionAction

    decap_joint_names: tuple[str, ...] = ()
    decap_preserve_order: bool = True
    decap_target_attr: str = "_arm_ik_joint_target_pos"
    decap_lambda_start: float = 1.0
    decap_lambda_end: float = 0.0
    decap_decay_type: Literal["constant", "linear", "cosine", "exp"] = "cosine"

    decap_steps_per_iteration: int = 24
    decap_warmup_iterations: int | None = None
    decap_decay_start_iteration: int | None = None
    decap_decay_end_iteration: int | None = None
    decap_decay_iterations: int | None = None
    decap_resume_iteration: int = 0

    decap_warmup_steps: int = 0
    decap_decay_start_step: int = 0
    decap_decay_end_step: int | None = None

    decap_exp_gamma: float = 0.99
    decap_exp_k: float = 100.0
    decap_prior_only: bool = False


def _resolve_joint_action_scales(
    scale_by_pattern: dict[str, float],
    joint_names: list[str],
    device: str,
    fallback: float = 0.25,
) -> torch.Tensor:
    scales = torch.full((len(joint_names),), float(fallback), device=device, dtype=torch.float32)
    for pattern, value in scale_by_pattern.items():
        compiled = re.compile(pattern)
        for idx, name in enumerate(joint_names):
            if compiled.fullmatch(name) or compiled.match(name):
                scales[idx] = float(value)
    return scales


def _resolve_policy_output(output) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, dict):
        for key in ("actions", "action", "mean_actions", "policy"):
            if key in output:
                return output[key]
    if isinstance(output, tuple | list) and output:
        return _resolve_policy_output(output[0])
    raise TypeError(f"Unsupported frozen WBC policy output type: {type(output)!r}")


def _resolve_policy_path(policy_path: str) -> str:
    expanded = os.path.abspath(os.path.expanduser(policy_path))
    if os.path.isfile(expanded):
        return expanded
    if os.path.isdir(expanded):
        candidates = (
            os.path.join(expanded, "policy.pt"),
            os.path.join(expanded, "exported", "policy.pt"),
        )
        for candidate in candidates:
            if os.path.isfile(candidate):
                return candidate
    raise FileNotFoundError(
        f"Frozen Go2+D1 WBC policy path does not exist or is not a supported run/export directory: {policy_path}"
    )


def _clamp_ee_targets(
    pos: torch.Tensor,
    ee_pos_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    body_exclusion_box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    body_clearance: float,
    extra_exclusion_boxes: tuple[tuple[tuple[float, float], tuple[float, float], tuple[float, float]], ...],
    workspace_origin: tuple[float, float, float] | None = None,
    reach_range: tuple[float, float] | None = None,
) -> torch.Tensor:
    clamped = pos.clone()
    for axis, (low, high) in enumerate(ee_pos_range):
        clamped[:, axis] = torch.clamp(clamped[:, axis], float(low), float(high))

    if workspace_origin is not None and reach_range is not None:
        origin = torch.tensor(workspace_origin, device=clamped.device, dtype=clamped.dtype)
        radial = clamped - origin.unsqueeze(0)
        distance = torch.linalg.vector_norm(radial, dim=1, keepdim=True)
        min_reach, max_reach = float(reach_range[0]), float(reach_range[1])
        safe_distance = distance.clamp_min(1.0e-6)
        projected_distance = distance.clamp(min=min_reach, max=max_reach)
        radial = radial * (projected_distance / safe_distance)
        zero_distance = distance.squeeze(1) < 1.0e-6
        if torch.any(zero_distance):
            radial[zero_distance] = torch.tensor(
                (max(min_reach, 1.0e-3), 0.0, 0.0),
                device=clamped.device,
                dtype=clamped.dtype,
            )
        clamped = origin.unsqueeze(0) + radial

    epsilon = 1.0e-4
    for exclusion_box in (body_exclusion_box, *extra_exclusion_boxes):
        expanded = tuple(
            (float(low) - float(body_clearance), float(high) + float(body_clearance))
            for low, high in exclusion_box
        )
        inside = torch.ones(clamped.shape[0], dtype=torch.bool, device=clamped.device)
        for axis, (low, high) in enumerate(expanded):
            inside &= (clamped[:, axis] >= low) & (clamped[:, axis] <= high)
        if not torch.any(inside):
            continue

        inside_ids = torch.nonzero(inside, as_tuple=False).squeeze(1)
        points = clamped.index_select(0, inside_ids)
        distances = torch.stack(
            (
                expanded[0][1] - points[:, 0],
                points[:, 1] - expanded[1][0],
                expanded[1][1] - points[:, 1],
                expanded[2][1] - points[:, 2],
            ),
            dim=1,
        )
        nearest_face = torch.argmin(distances, dim=1)
        for face in range(4):
            face_ids = inside_ids[nearest_face == face]
            if face_ids.numel() == 0:
                continue
            if face == 0:
                clamped[face_ids, 0] = expanded[0][1] + epsilon
            elif face == 1:
                clamped[face_ids, 1] = expanded[1][0] - epsilon
            elif face == 2:
                clamped[face_ids, 1] = expanded[1][1] + epsilon
            else:
                clamped[face_ids, 2] = expanded[2][1] + epsilon

    for axis, (low, high) in enumerate(ee_pos_range):
        clamped[:, axis] = torch.clamp(clamped[:, axis], float(low), float(high))
    return clamped


def _link_position_from_grasp_target(
    grasp_target_b: torch.Tensor,
    link_rotation_b: torch.Tensor,
    grasp_offset_link: torch.Tensor,
) -> torch.Tensor:
    """Convert a grasp-center target into the controlled-link position.

    ``grasp_offset_link`` is the vector from the controlled link origin to the
    midpoint of the fingers, expressed in the controlled link frame. Keeping
    this offset in the link frame is essential: subtracting it directly in the
    robot frame becomes wrong as soon as the wrist rotates.
    """
    offset_b = torch.bmm(link_rotation_b, grasp_offset_link.unsqueeze(-1)).squeeze(-1)
    return grasp_target_b - offset_b


def _advance_ee_deployment_targets(
    requested_targets: torch.Tensor,
    waypoint_targets: torch.Tensor,
    current_ee_pos: torch.Tensor,
    phase: torch.Tensor,
    progress: torch.Tensor,
    start_pos: torch.Tensor,
    initialized: torch.Tensor,
    handoff_goal: torch.Tensor,
    handoff_captured: torch.Tensor,
    step_dt: float,
    motion_speed: float,
    min_segment_duration_s: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Advance a vectorized carry-to-workspace route plus a smooth command handoff."""
    waypoint_count = int(waypoint_targets.shape[0])
    if waypoint_count == 0:
        return requested_targets.clone(), torch.zeros_like(phase, dtype=torch.bool)
    active = phase <= waypoint_count
    output = requested_targets.clone()
    if not torch.any(active):
        return output, active

    newly_initialized = active & ~initialized
    start_pos[newly_initialized] = current_ee_pos[newly_initialized]
    progress[newly_initialized] = 0.0
    initialized[newly_initialized] = True

    segment_goal = requested_targets.clone()
    waypoint_active = active & (phase < waypoint_count)
    waypoint_ids = torch.nonzero(waypoint_active, as_tuple=False).squeeze(1)
    if waypoint_ids.numel() > 0:
        segment_goal[waypoint_ids] = waypoint_targets.index_select(0, phase[waypoint_ids])

    handoff_active = active & (phase == waypoint_count)
    new_handoff = handoff_active & ~handoff_captured
    handoff_goal[new_handoff] = requested_targets[new_handoff]
    handoff_captured[new_handoff] = True
    segment_goal[handoff_active] = handoff_goal[handoff_active]

    distance = torch.linalg.vector_norm(segment_goal - start_pos, dim=1)
    duration = torch.clamp(
        distance / max(float(motion_speed), 1.0e-6),
        min=max(float(min_segment_duration_s), 1.0e-6),
    )
    next_progress = progress + float(step_dt) / duration
    t = torch.clamp(next_progress, 0.0, 1.0)
    t_smooth = 3.0 * t**2 - 2.0 * t**3
    output[active] = (
        start_pos[active] * (1.0 - t_smooth[active].unsqueeze(1))
        + segment_goal[active] * t_smooth[active].unsqueeze(1)
    )

    completed = active & (next_progress >= 1.0)
    continuing = active & ~completed
    progress[continuing] = next_progress[continuing]
    progress[completed] = 0.0
    start_pos[completed] = segment_goal[completed]
    phase[completed] += 1
    handoff_captured[completed & (phase == waypoint_count)] = False
    return output, active


class FrozenGo2D1WbcCommandAction(ActionTerm):
    """Execute a frozen Go2+D1 WBC policy from high-level velocity and EE commands."""

    cfg: FrozenGo2D1WbcCommandActionCfg

    def __init__(self, cfg: FrozenGo2D1WbcCommandActionCfg, env):
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.asset_name]
        self._raw_actions = torch.zeros(env.num_envs, self.action_dim, device=env.device)
        self._low_level_actions = torch.zeros(env.num_envs, len(cfg.joint_names), device=env.device)
        self._policy = None

        joint_ids, joint_names = self._asset.find_joints(cfg.joint_names, preserve_order=cfg.preserve_order)
        self._joint_ids = torch.tensor(joint_ids, dtype=torch.long, device=env.device)
        self._joint_names = list(joint_names)
        self._action_scales = _resolve_joint_action_scales(
            cfg.joint_action_scales,
            self._joint_names,
            env.device,
        )
        hard_limits = self._asset.data.default_joint_pos_limits[:, self._joint_ids]
        center = 0.5 * (hard_limits[..., 0] + hard_limits[..., 1])
        half_range = 0.5 * (hard_limits[..., 1] - hard_limits[..., 0]) * float(cfg.soft_joint_pos_limit_factor)
        self._joint_lower = center - half_range
        self._joint_upper = center + half_range
        for joint_index, joint_name in enumerate(self._joint_names):
            if joint_name in cfg.joint_position_ranges:
                self._joint_lower[:, joint_index] = float(cfg.joint_position_ranges[joint_name][0])
                self._joint_upper[:, joint_index] = float(cfg.joint_position_ranges[joint_name][1])
        if cfg.use_default_offset:
            self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
        else:
            self._offset = torch.zeros(env.num_envs, len(self._joint_names), device=env.device)
        self._joint_targets = self._offset.clone()

        self._deployment_waypoints = torch.tensor(
            cfg.deployment_ee_waypoints,
            device=env.device,
            dtype=torch.float32,
        ).reshape(-1, 3)
        self._deployment_phase = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self._deployment_progress = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        self._deployment_start_pos = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)
        self._deployment_initialized = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self._deployment_handoff_goal = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)
        self._deployment_handoff_captured = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self._deployment_ee_body_id = None
        if self._deployment_waypoints.shape[0] > 0:
            body_ids = self._asset.find_bodies(cfg.ee_body_name)[0]
            if len(body_ids) == 0:
                raise ValueError(f"Deployment EE body '{cfg.ee_body_name}' was not found in asset '{cfg.asset_name}'.")
            self._deployment_ee_body_id = int(body_ids[0])

        if cfg.clip is not None:
            clip = torch.tensor(
                [[cfg.clip[".*"][0], cfg.clip[".*"][1]]] * len(self._joint_names),
                device=env.device,
            )
            self._clip = clip
        else:
            self._clip = None

    @property
    def action_dim(self) -> int:
        return 9

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """High-level commands after applying the action bounds."""
        return self._raw_actions

    @property
    def low_level_actions(self) -> torch.Tensor:
        return self._low_level_actions

    def _load_policy(self):
        policy_path = self.cfg.policy_path or getattr(self._env.cfg, "wbc_policy_path", "")
        if not policy_path:
            raise RuntimeError(
                "Frozen Go2+D1 WBC requires `wbc_policy_path` on the env cfg or `policy_path` on the action cfg."
            )
        policy_path = _resolve_policy_path(policy_path)
        policy = torch.jit.load(policy_path, map_location=self._env.device)
        policy.eval()
        return policy

    def _inject_commands(self, actions: torch.Tensor) -> None:
        command_term = self._env.command_manager.get_term("base_velocity")
        ranges = command_term.cfg.ranges
        velocity = torch.stack(
            (
                actions[:, 0] * float(self.cfg.velocity_scale[0]),
                actions[:, 1] * float(self.cfg.velocity_scale[1]),
                actions[:, 2] * float(self.cfg.velocity_scale[2]),
            ),
            dim=1,
        )
        command_term.vel_command_b[:, 0] = torch.clamp(
            velocity[:, 0],
            min=ranges.lin_vel_x[0],
            max=ranges.lin_vel_x[1],
        )
        command_term.vel_command_b[:, 1] = torch.clamp(
            velocity[:, 1],
            min=ranges.lin_vel_y[0],
            max=ranges.lin_vel_y[1],
        )
        command_term.vel_command_b[:, 2] = torch.clamp(
            velocity[:, 2],
            min=ranges.ang_vel_z[0],
            max=ranges.ang_vel_z[1],
        )
        posture_command = getattr(command_term, "posture_command", None)
        if posture_command is not None:
            posture_command[:, 0] = 0.0
            pitch_low, pitch_high = self.cfg.body_pitch_range
            posture_command[:, 1] = float(pitch_low) + 0.5 * (actions[:, 3] + 1.0) * (
                float(pitch_high) - float(pitch_low)
            )
            if posture_command.shape[1] > 2:
                height_low, height_high = self.cfg.body_height_range
                posture_command[:, 2] = float(height_low) + 0.5 * (actions[:, 4] + 1.0) * (
                    float(height_high) - float(height_low)
                )
        if hasattr(command_term.cfg, "roll_range"):
            command_term.cfg.roll_range = (0.0, 0.0)
        if hasattr(command_term.cfg, "pitch_range"):
            command_term.cfg.pitch_range = self.cfg.body_pitch_range
        if hasattr(command_term.cfg, "height_range"):
            command_term.cfg.height_range = self.cfg.body_height_range
        if hasattr(command_term.cfg, "nominal_height"):
            command_term.cfg.nominal_height = float(self.cfg.body_nominal_height)

        ee_pos = torch.empty((self._env.num_envs, 3), device=self._env.device, dtype=actions.dtype)
        for axis, (low, high) in enumerate(self.cfg.ee_pos_range):
            scale = float(self.cfg.ee_pos_scale[axis])
            command = actions[:, 5 + axis] * scale
            ee_pos[:, axis] = float(low) + 0.5 * (command + 1.0) * (float(high) - float(low))
        ee_pos = _clamp_ee_targets(
            ee_pos,
            self.cfg.ee_pos_range,
            self.cfg.body_exclusion_box,
            self.cfg.body_clearance,
            self.cfg.extra_exclusion_boxes,
            self.cfg.workspace_origin,
            self.cfg.reach_range,
        )
        deploying = torch.zeros(self._env.num_envs, device=self._env.device, dtype=torch.bool)
        if self._deployment_ee_body_id is not None:
            from isaaclab.utils.math import subtract_frame_transforms

            ee_pose_w = self._asset.data.body_pose_w[:, self._deployment_ee_body_id]
            root_pose_w = self._asset.data.root_pose_w
            current_ee_pos, _ = subtract_frame_transforms(
                root_pose_w[:, 0:3],
                root_pose_w[:, 3:7],
                ee_pose_w[:, 0:3],
                ee_pose_w[:, 3:7],
            )
            ee_pos, deploying = _advance_ee_deployment_targets(
                ee_pos,
                self._deployment_waypoints,
                current_ee_pos,
                self._deployment_phase,
                self._deployment_progress,
                self._deployment_start_pos,
                self._deployment_initialized,
                self._deployment_handoff_goal,
                self._deployment_handoff_captured,
                self._env.step_dt,
                self.cfg.deployment_motion_speed,
                self.cfg.deployment_min_segment_duration_s,
            )

        if not hasattr(self._env, "_arm_ee_goal_pos"):
            self._env._arm_ee_goal_pos = ee_pos.clone()
            self._env._arm_ee_target_pos = ee_pos.clone()
            self._env._arm_ee_start_pos = ee_pos.clone()
            self._env._arm_ee_target_initialized = torch.ones(
                self._env.num_envs,
                dtype=torch.bool,
                device=self._env.device,
            )
        else:
            self._env._arm_ee_goal_pos[:] = ee_pos
            self._env._arm_ee_target_pos[:] = ee_pos
            self._env._arm_ee_start_pos[:] = ee_pos
            self._env._arm_ee_target_initialized[:] = True

        if hasattr(self._env, "_gripper_target_pos"):
            gripper = torch.where(
                actions[:, 8] > 0.0,
                float(self.cfg.gripper_scale),
                0.0,
            )
            gripper[deploying] = 0.0
            self._env._gripper_target_pos[:, 0] = gripper
            if self._env._gripper_target_pos.shape[1] > 1:
                self._env._gripper_target_pos[:, 1] = gripper

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions.clamp(-1.0, 1.0)
        self._inject_commands(self._raw_actions)
        if self._policy is None:
            self._policy = self._load_policy()
        wbc_obs = self._env.observation_manager.compute_group(self.cfg.wbc_obs_group)
        with torch.no_grad():
            low_level_actions = _resolve_policy_output(self._policy(wbc_obs))
            if low_level_actions.shape != self._low_level_actions.shape:
                raise RuntimeError(
                    f"Frozen Go2+D1 WBC policy output shape {tuple(low_level_actions.shape)} does not match "
                    f"expected low-level action shape {tuple(self._low_level_actions.shape)} for joints "
                    f"{self._joint_names}."
                )
            self._low_level_actions[:] = low_level_actions

        desired_targets = self._offset + self._action_scales.unsqueeze(0) * self._low_level_actions
        desired_targets = torch.clamp(
            desired_targets,
            min=self._joint_lower,
            max=self._joint_upper,
        )
        if self._clip is not None:
            desired_targets = torch.clamp(desired_targets, min=self._clip[:, 0], max=self._clip[:, 1])
        self._joint_targets[:] = desired_targets

    def apply_actions(self):
        self._asset.set_joint_position_target(self._joint_targets, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._low_level_actions[env_ids] = 0.0
        self._joint_targets[env_ids] = self._offset[env_ids]
        self._deployment_phase[env_ids] = 0
        self._deployment_progress[env_ids] = 0.0
        self._deployment_initialized[env_ids] = False
        self._deployment_handoff_captured[env_ids] = False


@configclass
class FrozenGo2D1WbcCommandActionCfg(ActionTermCfg):
    """High-level WBC command action executed by a frozen exported policy."""

    class_type: type = FrozenGo2D1WbcCommandAction

    asset_name: str = "robot"
    joint_names: list[str] = []
    preserve_order: bool = True
    wbc_obs_group: str = "wbc_policy"
    policy_path: str = ""
    velocity_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ee_pos_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ee_pos_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (0.10, 0.56),
        (-0.40, 0.40),
        (0.18, 0.65),
    )
    body_pitch_range: tuple[float, float] = (-0.16, 0.12)
    body_height_range: tuple[float, float] = (0.28, 0.39)
    body_nominal_height: float = 0.33
    body_exclusion_box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (-0.30, 0.34),
        (-0.20, 0.20),
        (-0.02, 0.30),
    )
    body_clearance: float = 0.07
    extra_exclusion_boxes: tuple[tuple[tuple[float, float], tuple[float, float], tuple[float, float]], ...] = ()
    workspace_origin: tuple[float, float, float] | None = (0.0, 0.0, 0.08)
    reach_range: tuple[float, float] | None = (0.12, 0.58)
    ee_body_name: str = "Link6"
    deployment_ee_waypoints: tuple[tuple[float, float, float], ...] = ()
    deployment_motion_speed: float = 0.18
    deployment_min_segment_duration_s: float = 1.0
    gripper_scale: float = 0.033
    joint_action_scales: dict[str, float] = {}
    joint_position_ranges: dict[str, tuple[float, float]] = {}
    soft_joint_pos_limit_factor: float = 0.9
    use_default_offset: bool = True
    clip: dict[str, tuple[float, float]] | None = None


class FrozenGo2D1LegWbcArmCommandAction(ActionTerm):
    """Execute a frozen leg-only Go2 WBC while high-level actions command D1 joints directly."""

    cfg: FrozenGo2D1LegWbcArmCommandActionCfg

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._asset: Articulation = env.scene[cfg.asset_name]
        self._raw_actions = torch.zeros(env.num_envs, self.action_dim, device=env.device)
        self._low_level_actions = torch.zeros(env.num_envs, len(cfg.leg_joint_names), device=env.device)
        self._policy = None

        leg_joint_ids, leg_joint_names = self._asset.find_joints(cfg.leg_joint_names, preserve_order=cfg.preserve_order)
        arm_joint_ids, arm_joint_names = self._asset.find_joints(cfg.arm_joint_names, preserve_order=cfg.preserve_order)
        gripper_joint_ids, gripper_joint_names = self._asset.find_joints(
            cfg.gripper_joint_names, preserve_order=cfg.preserve_order
        )
        self._leg_joint_ids = torch.tensor(leg_joint_ids, dtype=torch.long, device=env.device)
        self._arm_joint_ids = torch.tensor(arm_joint_ids, dtype=torch.long, device=env.device)
        self._gripper_joint_ids = torch.tensor(gripper_joint_ids, dtype=torch.long, device=env.device)
        self._leg_joint_names = list(leg_joint_names)
        self._arm_joint_names = list(arm_joint_names)
        self._gripper_joint_names = list(gripper_joint_names)
        self._leg_action_scales = _resolve_joint_action_scales(
            cfg.leg_joint_action_scales,
            self._leg_joint_names,
            env.device,
        )
        self._leg_offset = self._asset.data.default_joint_pos[:, self._leg_joint_ids].clone()
        leg_hard_limits = self._asset.data.default_joint_pos_limits[:, self._leg_joint_ids]
        leg_center = 0.5 * (leg_hard_limits[..., 0] + leg_hard_limits[..., 1])
        leg_half_range = (
            0.5
            * (leg_hard_limits[..., 1] - leg_hard_limits[..., 0])
            * float(cfg.leg_soft_joint_pos_limit_factor)
        )
        self._leg_target_lower = leg_center - leg_half_range
        self._leg_target_upper = leg_center + leg_half_range

        if len(self._arm_joint_names) != len(cfg.arm_joint_ranges):
            raise ValueError(
                f"arm_joint_ranges has {len(cfg.arm_joint_ranges)} entries, expected {len(self._arm_joint_names)} "
                f"for joints {self._arm_joint_names}."
            )
        self._arm_joint_low = torch.tensor(
            [cfg.arm_joint_ranges[name][0] for name in self._arm_joint_names],
            device=env.device,
            dtype=torch.float32,
        )
        self._arm_joint_high = torch.tensor(
            [cfg.arm_joint_ranges[name][1] for name in self._arm_joint_names],
            device=env.device,
            dtype=torch.float32,
        )
        self._arm_joint_targets = self._asset.data.default_joint_pos[:, self._arm_joint_ids].clone()
        self._gripper_targets = self._asset.data.default_joint_pos[:, self._gripper_joint_ids].clone()

    @property
    def action_dim(self) -> int:
        return 5 + len(self.cfg.arm_joint_names) + 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        """High-level commands after applying the action bounds."""
        return self._raw_actions

    @property
    def low_level_actions(self) -> torch.Tensor:
        return self._low_level_actions

    def _load_policy(self):
        policy_path = self.cfg.policy_path or getattr(self._env.cfg, "wbc_policy_path", "")
        if not policy_path:
            raise RuntimeError(
                "Frozen Go2+D1 leg WBC requires `wbc_policy_path` on the env cfg or `policy_path` on the action cfg."
            )
        policy_path = _resolve_policy_path(policy_path)
        policy = torch.jit.load(policy_path, map_location=self._env.device)
        policy.eval()
        return policy

    def _inject_commands(self, actions: torch.Tensor) -> None:
        command_term = self._env.command_manager.get_term("base_velocity")
        ranges = command_term.cfg.ranges
        command_term.vel_command_b[:, 0] = torch.clamp(
            actions[:, 0] * float(self.cfg.velocity_scale[0]),
            min=ranges.lin_vel_x[0],
            max=ranges.lin_vel_x[1],
        )
        command_term.vel_command_b[:, 1] = torch.clamp(
            actions[:, 1] * float(self.cfg.velocity_scale[1]),
            min=ranges.lin_vel_y[0],
            max=ranges.lin_vel_y[1],
        )
        command_term.vel_command_b[:, 2] = torch.clamp(
            actions[:, 2] * float(self.cfg.velocity_scale[2]),
            min=ranges.ang_vel_z[0],
            max=ranges.ang_vel_z[1],
        )

        posture_command = getattr(command_term, "posture_command", None)
        if posture_command is not None:
            posture_command[:, 0] = 0.0
            pitch_low, pitch_high = self.cfg.body_pitch_range
            posture_command[:, 1] = float(pitch_low) + 0.5 * (actions[:, 3] + 1.0) * (
                float(pitch_high) - float(pitch_low)
            )
            if posture_command.shape[1] > 2:
                height_low, height_high = self.cfg.body_height_range
                posture_command[:, 2] = float(height_low) + 0.5 * (actions[:, 4] + 1.0) * (
                    float(height_high) - float(height_low)
                )

        arm_raw = actions[:, 5 : 5 + len(self._arm_joint_names)]
        requested_arm_targets = self._arm_joint_low + 0.5 * (arm_raw + 1.0) * (
            self._arm_joint_high - self._arm_joint_low
        )
        self._arm_joint_targets[:] = requested_arm_targets
        if hasattr(self._env, "_arm_joint_target_pos"):
            self._env._arm_joint_target_pos = self._arm_joint_targets.clone()
        if hasattr(self._env, "_arm_async_joint_target_pos"):
            self._env._arm_async_joint_target_pos = self._arm_joint_targets.clone()

        gripper_action = actions[:, 5 + len(self._arm_joint_names)]
        gripper = torch.where(
            gripper_action > 0.0,
            float(self.cfg.gripper_scale),
            0.0,
        )
        if self._gripper_targets.shape[1] > 0:
            self._gripper_targets[:, 0] = gripper
        if self._gripper_targets.shape[1] > 1:
            self._gripper_targets[:, 1] = gripper
        if hasattr(self._env, "_gripper_target_pos"):
            self._env._gripper_target_pos = self._gripper_targets.clone()

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions.clamp(-1.0, 1.0)
        self._inject_commands(self._raw_actions)
        if self._policy is None:
            self._policy = self._load_policy()
        wbc_obs = self._env.observation_manager.compute_group(self.cfg.wbc_obs_group)
        with torch.no_grad():
            low_level_actions = _resolve_policy_output(self._policy(wbc_obs))
            if low_level_actions.shape != self._low_level_actions.shape:
                raise RuntimeError(
                    f"Frozen Go2+D1 leg WBC policy output shape {tuple(low_level_actions.shape)} does not match "
                    f"expected leg action shape {tuple(self._low_level_actions.shape)} for joints "
                    f"{self._leg_joint_names}."
                )
            self._low_level_actions[:] = low_level_actions

    def apply_actions(self):
        leg_targets = self._leg_offset + self._leg_action_scales.unsqueeze(0) * self._low_level_actions
        leg_targets = torch.clamp(leg_targets, min=self._leg_target_lower, max=self._leg_target_upper)
        self._asset.set_joint_position_target(leg_targets, joint_ids=self._leg_joint_ids)
        self._asset.set_joint_position_target(self._arm_joint_targets, joint_ids=self._arm_joint_ids)
        if self._gripper_joint_ids.numel() > 0:
            self._asset.set_joint_position_target(self._gripper_targets, joint_ids=self._gripper_joint_ids)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 0.0
        self._low_level_actions[env_ids] = 0.0
        self._arm_joint_targets[env_ids] = self._asset.data.joint_pos[env_ids][:, self._arm_joint_ids]
        self._gripper_targets[env_ids] = self._asset.data.joint_pos[env_ids][:, self._gripper_joint_ids]


@configclass
class FrozenGo2D1LegWbcArmCommandActionCfg(ActionTermCfg):
    """High-level action that commands Go2 body targets and D1 joint targets over a frozen leg WBC."""

    class_type: type = FrozenGo2D1LegWbcArmCommandAction

    asset_name: str = "robot"
    leg_joint_names: list[str] = []
    arm_joint_names: list[str] = []
    gripper_joint_names: list[str] = []
    preserve_order: bool = True
    wbc_obs_group: str = "wbc_policy"
    policy_path: str = ""
    velocity_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    body_pitch_range: tuple[float, float] = (-0.16, 0.12)
    body_height_range: tuple[float, float] = (0.28, 0.39)
    arm_joint_ranges: dict[str, tuple[float, float]] = {}
    gripper_scale: float = 0.033
    leg_joint_action_scales: dict[str, float] = {}
    leg_soft_joint_pos_limit_factor: float = 0.9


class FrozenGo2D1LegWbcEeCommandAction(FrozenGo2D1LegWbcArmCommandAction):
    """Run a frozen leg WBC while root-frame pose IK controls the D1 grasp frame."""

    cfg: FrozenGo2D1LegWbcEeCommandActionCfg

    def __init__(self, cfg: FrozenGo2D1LegWbcEeCommandActionCfg, env):
        super().__init__(cfg, env)

        expected_rate_limits = self.action_dim - 1
        if len(cfg.normalized_action_rate_limits) != expected_rate_limits:
            raise ValueError(
                "normalized_action_rate_limits must contain one entry for each continuous "
                f"high-level action ({expected_rate_limits}), got {len(cfg.normalized_action_rate_limits)}."
            )
        if cfg.gripper_open_threshold >= cfg.gripper_close_threshold:
            raise ValueError(
                "gripper_open_threshold must be smaller than gripper_close_threshold to provide hysteresis."
            )
        self._applied_actions = torch.zeros_like(self._raw_actions)
        self._normalized_action_rate_limits = torch.as_tensor(
            cfg.normalized_action_rate_limits,
            device=env.device,
            dtype=self._raw_actions.dtype,
        )
        self._gripper_closed_command = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

        from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg

        body_ids = self._asset.find_bodies(cfg.ee_body_name)[0]
        if len(body_ids) == 0:
            raise ValueError(f"EE body '{cfg.ee_body_name}' was not found in asset '{cfg.asset_name}'.")
        self._ee_body_id = int(body_ids[0])
        grasp_body_ids = self._asset.find_bodies(cfg.grasp_body_names, preserve_order=True)[0]
        if len(grasp_body_ids) != 2:
            raise ValueError(
                f"Expected exactly two grasp bodies {cfg.grasp_body_names}, found body ids {grasp_body_ids}."
            )
        self._grasp_body_ids = torch.as_tensor(grasp_body_ids, device=env.device, dtype=torch.long)
        if getattr(self._asset, "is_fixed_base", True):
            self._ee_jacobi_idx = max(self._ee_body_id - 1, 0)
            self._ee_jacobi_joint_ids = self._arm_joint_ids
        else:
            self._ee_jacobi_idx = self._ee_body_id
            # PhysX prepends the floating base's six degrees of freedom to the
            # articulation Jacobian columns. Articulation joint IDs do not
            # include that prefix, so offset them before selecting the arm.
            self._ee_jacobi_joint_ids = self._arm_joint_ids + 6
        ik_cfg = DifferentialIKControllerCfg(
            command_type="pose",
            use_relative_mode=False,
            ik_method="dls",
            ik_params={"lambda_val": float(cfg.dls_damping)},
        )
        self._ik_controller = DifferentialIKController(
            ik_cfg,
            num_envs=env.num_envs,
            device=env.device,
        )

        self._deployment_waypoints = torch.tensor(
            cfg.deployment_ee_waypoints,
            device=env.device,
            dtype=torch.float32,
        ).reshape(-1, 3)
        self._deployment_phase = torch.zeros(env.num_envs, device=env.device, dtype=torch.long)
        self._deployment_progress = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        self._deployment_start_pos = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)
        self._deployment_initialized = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)
        self._deployment_handoff_goal = torch.zeros(env.num_envs, 3, device=env.device, dtype=torch.float32)
        self._deployment_handoff_captured = torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)

    @property
    def action_dim(self) -> int:
        # vx, vy, wz, body_pitch, body_height,
        # grasp_x, grasp_y, grasp_z, wrist_roll, gripper
        return 10

    @property
    def processed_actions(self) -> torch.Tensor:
        """Commands after continuous slew limiting and binary-gripper hysteresis."""
        return self._applied_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions[:] = actions.clamp(-1.0, 1.0)
        continuous_target = self._raw_actions[:, :-1]
        self._applied_actions[:, :-1] = _slew_limit_normalized_actions(
            self._applied_actions[:, :-1],
            continuous_target,
            self._normalized_action_rate_limits,
            self._env.step_dt,
        )

        gripper_action = self._raw_actions[:, -1]
        self._gripper_closed_command |= gripper_action >= float(self.cfg.gripper_close_threshold)
        self._gripper_closed_command &= gripper_action > float(self.cfg.gripper_open_threshold)
        self._applied_actions[:, -1] = torch.where(
            self._gripper_closed_command,
            torch.ones_like(gripper_action),
            -torch.ones_like(gripper_action),
        )

        self._inject_commands(self._applied_actions)
        if self._policy is None:
            self._policy = self._load_policy()
        wbc_obs = self._env.observation_manager.compute_group(self.cfg.wbc_obs_group)
        with torch.no_grad():
            low_level_actions = _resolve_policy_output(self._policy(wbc_obs))
            if low_level_actions.shape != self._low_level_actions.shape:
                raise RuntimeError(
                    f"Frozen Go2+D1 leg WBC policy output shape {tuple(low_level_actions.shape)} does not match "
                    f"expected leg action shape {tuple(self._low_level_actions.shape)} for joints "
                    f"{self._leg_joint_names}."
                )
            self._low_level_actions[:] = low_level_actions

    def _inject_commands(self, actions: torch.Tensor) -> None:
        command_term = self._env.command_manager.get_term("base_velocity")
        velocity_limits = tuple(abs(float(scale)) for scale in self.cfg.velocity_scale)
        command_term.vel_command_b[:, 0] = torch.clamp(
            actions[:, 0] * float(self.cfg.velocity_scale[0]),
            min=-velocity_limits[0],
            max=velocity_limits[0],
        )
        command_term.vel_command_b[:, 1] = torch.clamp(
            actions[:, 1] * float(self.cfg.velocity_scale[1]),
            min=-velocity_limits[1],
            max=velocity_limits[1],
        )
        command_term.vel_command_b[:, 2] = torch.clamp(
            actions[:, 2] * float(self.cfg.velocity_scale[2]),
            min=-velocity_limits[2],
            max=velocity_limits[2],
        )
        posture_command = getattr(command_term, "posture_command", None)
        if posture_command is not None:
            # Roll is intentionally fixed: the trained leg WBC exposes pitch
            # and height as useful manipulation degrees of freedom, while roll
            # adds little for a forward tabletop grasp.
            posture_command[:, 0] = 0.0
            pitch_low, pitch_high = self.cfg.body_pitch_range
            posture_command[:, 1] = float(pitch_low) + 0.5 * (actions[:, 3] + 1.0) * (
                float(pitch_high) - float(pitch_low)
            )
            if posture_command.shape[1] > 2:
                height_low, height_high = self.cfg.body_height_range
                posture_command[:, 2] = float(height_low) + 0.5 * (actions[:, 4] + 1.0) * (
                    float(height_high) - float(height_low)
                )

        grasp_pos_b = torch.empty((self._env.num_envs, 3), device=self._env.device, dtype=actions.dtype)
        for axis, (low, high) in enumerate(self.cfg.ee_pos_range):
            scaled = actions[:, 5 + axis] * float(self.cfg.ee_pos_scale[axis])
            grasp_pos_b[:, axis] = float(low) + 0.5 * (scaled + 1.0) * (float(high) - float(low))
        grasp_pos_b = _clamp_ee_targets(
            grasp_pos_b,
            self.cfg.ee_pos_range,
            self.cfg.body_exclusion_box,
            self.cfg.body_clearance,
            self.cfg.extra_exclusion_boxes,
            self.cfg.workspace_origin,
            self.cfg.reach_range,
        )

        from isaaclab.utils.math import (
            matrix_from_quat,
            quat_from_euler_xyz,
            quat_inv,
            quat_mul,
            subtract_frame_transforms,
        )

        ee_pose_w = self._asset.data.body_pose_w[:, self._ee_body_id]
        root_pose_w = self._asset.data.root_pose_w
        ee_pos_b, ee_quat_b = subtract_frame_transforms(
            root_pose_w[:, 0:3],
            root_pose_w[:, 3:7],
            ee_pose_w[:, 0:3],
            ee_pose_w[:, 3:7],
        )

        # The policy commands the same virtual grasp center used by the task
        # observations and rewards. Pose IK controls Link6, so resolve the
        # finger-midpoint offset in Link6 coordinates before rotating it into
        # the desired root-frame wrist orientation.
        grasp_midpoint_w = self._asset.data.body_pos_w.index_select(1, self._grasp_body_ids).mean(dim=1)
        root_rotation_w_to_b = matrix_from_quat(quat_inv(root_pose_w[:, 3:7]))
        grasp_midpoint_b = torch.bmm(
            root_rotation_w_to_b,
            (grasp_midpoint_w - root_pose_w[:, 0:3]).unsqueeze(-1),
        ).squeeze(-1)
        ee_rotation_w_to_link = matrix_from_quat(quat_inv(ee_pose_w[:, 3:7]))
        grasp_offset_link = torch.bmm(
            ee_rotation_w_to_link,
            (grasp_midpoint_w - ee_pose_w[:, 0:3]).unsqueeze(-1),
        ).squeeze(-1)

        wrist_low, wrist_high = self.cfg.wrist_roll_range
        wrist_roll = float(wrist_low) + 0.5 * (actions[:, 8] + 1.0) * (
            float(wrist_high) - float(wrist_low)
        )
        zeros = torch.zeros_like(wrist_roll)
        local_wrist_quat = quat_from_euler_xyz(zeros, zeros, wrist_roll)
        nominal_quat = torch.as_tensor(
            self.cfg.ee_nominal_quat,
            device=self._env.device,
            dtype=actions.dtype,
        )
        nominal_quat = nominal_quat / torch.linalg.vector_norm(nominal_quat).clamp_min(1.0e-8)
        ee_quat_des_b = quat_mul(nominal_quat.unsqueeze(0).expand(self._env.num_envs, -1), local_wrist_quat)

        deploying = torch.zeros(self._env.num_envs, device=self._env.device, dtype=torch.bool)
        if self._deployment_waypoints.shape[0] > 0:
            grasp_pos_b, deploying = _advance_ee_deployment_targets(
                grasp_pos_b,
                self._deployment_waypoints,
                grasp_midpoint_b,
                self._deployment_phase,
                self._deployment_progress,
                self._deployment_start_pos,
                self._deployment_initialized,
                self._deployment_handoff_goal,
                self._deployment_handoff_captured,
                self._env.step_dt,
                self.cfg.deployment_motion_speed,
                self.cfg.deployment_min_segment_duration_s,
            )

        ee_rotation_des_b = matrix_from_quat(ee_quat_des_b)
        ee_pos_des_b = _link_position_from_grasp_target(
            grasp_pos_b,
            ee_rotation_des_b,
            grasp_offset_link,
        )
        # The grasp center can be collision-safe while Link6 sits inside the
        # body keep-out. Project the actual controlled link as a final safety
        # constraint, then report the resulting effective grasp-center target.
        ee_pos_des_b = _clamp_ee_targets(
            ee_pos_des_b,
            self.cfg.ee_pos_range,
            self.cfg.body_exclusion_box,
            self.cfg.body_clearance,
            self.cfg.extra_exclusion_boxes,
            self.cfg.workspace_origin,
            self.cfg.reach_range,
        )
        rotated_grasp_offset_b = torch.bmm(
            ee_rotation_des_b,
            grasp_offset_link.unsqueeze(-1),
        ).squeeze(-1)
        grasp_pos_b = ee_pos_des_b + rotated_grasp_offset_b
        self._ik_controller.set_command(torch.cat((ee_pos_des_b, ee_quat_des_b), dim=1))
        full_jacobian = self._asset.root_physx_view.get_jacobians()
        jacobian_w = full_jacobian[:, self._ee_jacobi_idx, :, :].index_select(
            dim=2,
            index=self._ee_jacobi_joint_ids,
        )
        jacobian_b = jacobian_w.clone()
        jacobian_b[:, 0:3, :] = torch.bmm(root_rotation_w_to_b, jacobian_w[:, 0:3, :])
        jacobian_b[:, 3:6, :] = torch.bmm(root_rotation_w_to_b, jacobian_w[:, 3:6, :])
        joint_pos_arm = self._asset.data.joint_pos.index_select(dim=1, index=self._arm_joint_ids)
        self._arm_joint_targets[:] = torch.clamp(
            self._ik_controller.compute(ee_pos_b, ee_quat_b, jacobian_b, joint_pos_arm),
            min=self._arm_joint_low.unsqueeze(0),
            max=self._arm_joint_high.unsqueeze(0),
        )

        self._env._arm_grasp_goal_pos = grasp_pos_b.clone()
        self._env._arm_ee_goal_pos = ee_pos_des_b.clone()
        self._env._arm_ee_target_pos = ee_pos_des_b.clone()
        self._env._arm_ee_target_quat = ee_quat_des_b.clone()
        self._env._arm_ik_joint_target_pos = self._arm_joint_targets.clone()
        self._env._arm_ee_target_initialized = torch.ones(
            self._env.num_envs,
            dtype=torch.bool,
            device=self._env.device,
        )

        gripper = torch.where(
            actions[:, 9] > 0.0,
            float(self.cfg.gripper_scale),
            0.0,
        )
        gripper[deploying] = 0.0
        if self._gripper_targets.shape[1] > 0:
            self._gripper_targets[:, 0] = gripper
        if self._gripper_targets.shape[1] > 1:
            self._gripper_targets[:, 1] = gripper
        self._env._gripper_target_pos = self._gripper_targets.clone()

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        super().reset(env_ids)
        if env_ids is None:
            env_ids_t = torch.arange(self._env.num_envs, device=self._env.device, dtype=torch.long)
        else:
            env_ids_t = torch.as_tensor(env_ids, device=self._env.device, dtype=torch.long).flatten()

        # Seed the shaped command from the measured reset pose. Starting at a
        # normalized zero would command the workspace midpoint on the first
        # step and immediately pull the verified ready pose away from the can.
        from isaaclab.utils.math import quat_apply_inverse

        root_pos_w = self._asset.data.root_pos_w.index_select(0, env_ids_t)
        root_quat_w = self._asset.data.root_quat_w.index_select(0, env_ids_t)
        grasp_midpoint_w = self._asset.data.body_pos_w.index_select(0, env_ids_t).index_select(
            1, self._grasp_body_ids
        ).mean(dim=1)
        grasp_midpoint_b = quat_apply_inverse(root_quat_w, grasp_midpoint_w - root_pos_w)

        self._applied_actions[env_ids_t] = 0.0
        pitch_low, pitch_high = self.cfg.body_pitch_range
        height_low, height_high = self.cfg.body_height_range
        self._applied_actions[env_ids_t, 3] = 2.0 * (0.0 - float(pitch_low)) / (
            float(pitch_high) - float(pitch_low)
        ) - 1.0
        self._applied_actions[env_ids_t, 4] = 2.0 * (float(self.cfg.body_nominal_height) - float(height_low)) / (
            float(height_high) - float(height_low)
        ) - 1.0
        for axis, (low, high) in enumerate(self.cfg.ee_pos_range):
            self._applied_actions[env_ids_t, 5 + axis] = torch.clamp(
                2.0 * (grasp_midpoint_b[:, axis] - float(low)) / (float(high) - float(low)) - 1.0,
                min=-1.0,
                max=1.0,
            )
        self._applied_actions[env_ids_t, 9] = -1.0
        self._gripper_closed_command[env_ids_t] = False
        self._deployment_phase[env_ids_t] = 0
        self._deployment_progress[env_ids_t] = 0.0
        self._deployment_initialized[env_ids_t] = False
        self._deployment_handoff_captured[env_ids_t] = False
        self._ik_controller.reset(env_ids_t)


@configclass
class FrozenGo2D1LegWbcEeCommandActionCfg(FrozenGo2D1LegWbcArmCommandActionCfg):
    """Grasp pose plus Go2 velocity, pitch, and height over a frozen leg WBC."""

    class_type: type = FrozenGo2D1LegWbcEeCommandAction

    ee_pos_scale: tuple[float, float, float] = (1.0, 1.0, 1.0)
    ee_pos_range: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (0.10, 0.56),
        (-0.40, 0.40),
        (0.18, 0.65),
    )
    body_nominal_height: float = 0.33
    body_exclusion_box: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = (
        (-0.30, 0.34),
        (-0.20, 0.20),
        (-0.02, 0.30),
    )
    body_clearance: float = 0.07
    extra_exclusion_boxes: tuple[tuple[tuple[float, float], tuple[float, float], tuple[float, float]], ...] = ()
    workspace_origin: tuple[float, float, float] | None = (0.0, 0.0, 0.08)
    reach_range: tuple[float, float] | None = (0.12, 0.58)
    ee_body_name: str = "Link6"
    grasp_body_names: tuple[str, str] = ("Link7_1", "Link7_2")
    # Link6's local +Z axis is the gripper approach direction in the imported
    # asset. A +90 degree root-frame pitch points it forward; wrist_roll then
    # rotates the fingers about that approach axis.
    ee_nominal_quat: tuple[float, float, float, float] = (
        0.7071067811865476,
        0.0,
        0.7071067811865475,
        0.0,
    )
    wrist_roll_range: tuple[float, float] = (-1.5707963267948966, 1.5707963267948966)
    deployment_ee_waypoints: tuple[tuple[float, float, float], ...] = ()
    deployment_motion_speed: float = 0.18
    deployment_min_segment_duration_s: float = 1.0
    dls_damping: float = 0.06
    # Limits are in normalized action units per second for
    # [vx, vy, wz, pitch, height, grasp_x, grasp_y, grasp_z, wrist_roll].
    normalized_action_rate_limits: tuple[float, ...] = (4.0, 4.0, 4.0, 1.5, 1.5, 1.3, 0.8, 1.3, 1.5)
    gripper_open_threshold: float = -0.25
    gripper_close_threshold: float = 0.25
