from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Literal

import torch

from isaaclab.envs.mdp import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass


class DecapJointPositionAction(JointPositionAction):
    """Joint position action with the original additive decaying motion prior.

    Selected joints can optionally pass through a deployment-style angle-command
    model. The policy target can be sampled at a slower command rate, quantized,
    delayed, and optionally evolved through a second-order setpoint filter at
    physics rate. The articulation actuator then converts the held or filtered
    position target into torque. This preserves an absolute-angle policy
    interface without making the simulated joint kinematic.

    The DecAP correction is
    ``q_policy + lambda * (q_reference - q_robot)``.
    """

    cfg: DecapJointPositionActionCfg

    def __init__(self, cfg: DecapJointPositionActionCfg, env):
        super().__init__(cfg, env)
        self._decap_lambda = torch.full((self.num_envs, 1), float(self.cfg.decap_lambda_start), device=self.device)
        self._decap_steps_per_iteration = max(1, int(self.cfg.decap_steps_per_iteration))
        self._decap_resume_step_offset = self._iterations_to_steps(self.cfg.decap_resume_iteration)
        self._decap_warmup_steps, self._decap_decay_start_step, self._decap_decay_end_step = (
            self._resolve_decap_schedule()
        )
        self._motion_command = None
        self._motion_joint_indices = None
        self._robot_joint_indices = None
        self._policy_processed_actions = self._processed_actions.clone()
        self._configure_reference_action_groups()
        self._configure_binary_action_group()
        self._configure_servo_command_model()

    @property
    def decap_lambda(self) -> torch.Tensor:
        return self._decap_lambda

    @property
    def policy_processed_actions(self) -> torch.Tensor:
        """Final policy targets before applying the optional decaying motion prior."""
        return self._policy_processed_actions

    def _action_indices_for_joint_names(
        self,
        joint_names: tuple[str, ...] | None,
        *,
        default_to_all: bool,
        field_name: str,
    ) -> torch.Tensor:
        if joint_names is None:
            resolved_names = tuple(self._joint_names) if default_to_all else ()
        else:
            resolved_names = tuple(joint_names)
        missing = [name for name in resolved_names if name not in self._joint_names]
        if missing:
            raise ValueError(
                f"{field_name} contains joints not controlled by this action term: {missing}. "
                f"Available joints: {self._joint_names}"
            )
        return torch.tensor(
            [self._joint_names.index(name) for name in resolved_names],
            dtype=torch.long,
            device=self.device,
        )

    def _configure_reference_action_groups(self) -> None:
        self._decap_action_indices = self._action_indices_for_joint_names(
            self.cfg.decap_joint_names,
            default_to_all=True,
            field_name="decap_joint_names",
        )
        self._reference_residual_action_indices = self._action_indices_for_joint_names(
            self.cfg.reference_residual_joint_names,
            default_to_all=False,
            field_name="reference_residual_joint_names",
        )
        overlap = set(self._decap_action_indices.tolist()).intersection(
            self._reference_residual_action_indices.tolist()
        )
        if overlap:
            names = [self._joint_names[index] for index in sorted(overlap)]
            raise ValueError(
                "decap_joint_names and reference_residual_joint_names must be disjoint; "
                f"overlap: {names}"
            )
        if self.cfg.decap_prior_only and self._reference_residual_action_indices.numel() > 0:
            raise ValueError("decap_prior_only cannot be combined with reference_residual_joint_names.")

    def _configure_binary_action_group(self) -> None:
        """Resolve actions whose continuous policy output selects one of two physical targets."""
        self._binary_action_indices = self._action_indices_for_joint_names(
            self.cfg.binary_joint_names,
            default_to_all=False,
            field_name="binary_joint_names",
        )
        if self._binary_action_indices.numel() == 0:
            return

        binary_indices = set(self._binary_action_indices.tolist())
        decap_overlap = binary_indices.intersection(self._decap_action_indices.tolist())
        residual_overlap = binary_indices.intersection(self._reference_residual_action_indices.tolist())
        if decap_overlap or residual_overlap:
            overlap = sorted(decap_overlap.union(residual_overlap))
            names = [self._joint_names[index] for index in overlap]
            raise ValueError(
                "Binary joints cannot also use DecAP or reference-relative residual actions; "
                f"overlap: {names}"
            )
        if math.isclose(float(self.cfg.binary_open_position), float(self.cfg.binary_closed_position)):
            raise ValueError("binary_open_position and binary_closed_position must be different.")

    def _apply_binary_policy_targets(self) -> None:
        if self._binary_action_indices.numel() == 0:
            return
        binary_state = (
            self._raw_actions[:, self._binary_action_indices] > float(self.cfg.binary_action_threshold)
        ).to(self._processed_actions.dtype)
        open_position = float(self.cfg.binary_open_position)
        target = open_position + binary_state * (float(self.cfg.binary_closed_position) - open_position)
        self._policy_processed_actions[:, self._binary_action_indices] = target
        self._processed_actions[:, self._binary_action_indices] = target

    def _get_motion_command(self):
        if self._motion_command is None:
            self._motion_command = self._env.command_manager.get_term(self.cfg.decap_command_name)
        return self._motion_command

    def _resolve_motion_joint_indices(self, motion_joint_pos: torch.Tensor) -> torch.Tensor | slice:
        if self._motion_joint_indices is not None:
            return self._motion_joint_indices

        motion_joint_names = getattr(self._get_motion_command(), "robot_motion_joint_names", None)
        if motion_joint_names is not None:
            motion_name_to_idx = {name: i for i, name in enumerate(motion_joint_names)}
            joint_indices = []
            for joint_name in self._joint_names:
                if joint_name not in motion_name_to_idx:
                    raise ValueError(
                        f"Joint '{joint_name}' used by action term is missing in motion data joints: "
                        f"{motion_joint_names}"
                    )
                joint_indices.append(motion_name_to_idx[joint_name])
            self._motion_joint_indices = torch.tensor(joint_indices, dtype=torch.long, device=self.device)
            return self._motion_joint_indices

        if motion_joint_pos.shape[1] == self.action_dim:
            self._motion_joint_indices = slice(None)
            return self._motion_joint_indices

        if motion_joint_pos.shape[1] < self.action_dim:
            raise ValueError(
                f"Motion reference has fewer joints ({motion_joint_pos.shape[1]}) than action term ({self.action_dim})."
            )

        self._motion_joint_indices = torch.arange(self.action_dim, dtype=torch.long, device=self.device)
        return self._motion_joint_indices

    def _resolve_robot_joint_indices(self, robot_joint_pos: torch.Tensor) -> torch.Tensor | slice:
        if self._robot_joint_indices is not None:
            return self._robot_joint_indices

        motion_command = self._get_motion_command()
        robot_joint_names = getattr(motion_command, "robot_motion_joint_names", None)
        if robot_joint_names is not None:
            robot_name_to_idx = {name: i for i, name in enumerate(robot_joint_names)}
            joint_indices = []
            for joint_name in self._joint_names:
                if joint_name not in robot_name_to_idx:
                    raise ValueError(
                        f"Joint '{joint_name}' used by action term is missing in robot joint names: {robot_joint_names}"
                    )
                joint_indices.append(robot_name_to_idx[joint_name])
            self._robot_joint_indices = torch.tensor(joint_indices, dtype=torch.long, device=self.device)
            return self._robot_joint_indices

        if robot_joint_pos.shape[1] == self.action_dim:
            self._robot_joint_indices = slice(None)
            return self._robot_joint_indices

        if robot_joint_pos.shape[1] < self.action_dim:
            raise ValueError(
                f"Robot state has fewer joints ({robot_joint_pos.shape[1]}) than action term ({self.action_dim})."
            )

        self._robot_joint_indices = torch.arange(self.action_dim, dtype=torch.long, device=self.device)
        return self._robot_joint_indices

    def _get_robot_joint_pos(self) -> torch.Tensor:
        motion_command = self._get_motion_command()
        robot_joint_pos = getattr(motion_command, "robot_joint_pos", None)
        if robot_joint_pos is None:
            raise RuntimeError("Motion command does not expose robot_joint_pos required for DecAP prior computation.")
        robot_joint_indices = self._resolve_robot_joint_indices(robot_joint_pos)
        return robot_joint_pos[:, robot_joint_indices]

    def _iterations_to_steps(self, iterations: int | float) -> int:
        return max(0, int(iterations)) * self._decap_steps_per_iteration

    def _current_decap_step(self) -> int:
        return self._decap_resume_step_offset + max(int(self._env.common_step_counter), 0)

    def _resolve_decap_schedule(self) -> tuple[int, int, int]:
        """Resolve user-facing iteration schedule fields into env-step counters."""
        if self.cfg.decap_warmup_iterations is None:
            warmup_steps = max(0, int(self.cfg.decap_warmup_steps))
        else:
            warmup_steps = self._iterations_to_steps(self.cfg.decap_warmup_iterations)

        if self.cfg.decap_decay_start_iteration is None:
            # Preserve legacy step semantics: decay_start_step was measured after warmup.
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
            # Preserve legacy step semantics: decay_end_step was measured after warmup.
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
            decay_counter = self._compute_decap_decay_counter()
            value = start * (gamma ** (decay_counter / k))
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

    def _compute_decap_progress(self) -> float:
        step = self._current_decap_step()
        if step <= self._decap_decay_start_step:
            return 0.0
        if step >= self._decap_decay_end_step:
            return 1.0
        return (step - self._decap_decay_start_step) / float(
            self._decap_decay_end_step - self._decap_decay_start_step
        )

    def process_actions(self, actions: torch.Tensor):
        motion_command = self._get_motion_command()
        # The command exposes robot-ordered references and appends optional
        # mapped channels such as the D1's single physical gripper servo.
        motion_joint_pos = motion_command.joint_pos
        motion_indices = self._resolve_motion_joint_indices(motion_joint_pos)
        motion_reference = motion_joint_pos[:, motion_indices]
        robot_joint_pos = self._get_robot_joint_pos()

        decap_decay_counter = 0.0
        if self.cfg.decap_decay_type == "exp":
            decap_progress = 0.0
            decap_decay_counter = self._compute_decap_decay_counter()
        else:
            decap_progress = self._compute_decap_progress()
        decap_lambda = self._compute_decap_lambda(decap_progress)
        self._decap_lambda.fill_(decap_lambda)
        prior_delta = torch.zeros_like(motion_reference)
        decap_indices = self._decap_action_indices
        if decap_indices.numel() > 0:
            prior_delta[:, decap_indices] = decap_lambda * (
                motion_reference[:, decap_indices] - robot_joint_pos[:, decap_indices]
            )
        else:
            # Log the effective scheduled-prior strength. A configured schedule
            # which applies to no joints has zero effect.
            decap_lambda = 0.0
            self._decap_lambda.zero_()

        if self.cfg.decap_prior_only:
            if hasattr(self, "_raw_actions"):
                self._raw_actions[:] = actions
            self._policy_processed_actions[:] = robot_joint_pos
            self._processed_actions = robot_joint_pos + prior_delta
        else:
            # Base processing: preserve the original policy action path untouched.
            super().process_actions(actions)
            self._policy_processed_actions[:] = self._processed_actions
            # Selected arm channels are permanent reference-relative residuals:
            # q_target = q_reference + scale * actor_action. Other joints keep
            # the standard default-pose-relative position action.
            residual_indices = self._reference_residual_action_indices
            if residual_indices.numel() > 0:
                residual_scale = (
                    self._scale[:, residual_indices]
                    if isinstance(self._scale, torch.Tensor)
                    else float(self._scale)
                )
                residual = self._raw_actions[:, residual_indices] * residual_scale
                residual_target = motion_reference[:, residual_indices] + residual
                self._policy_processed_actions[:, residual_indices] = residual_target
                self._processed_actions[:, residual_indices] = residual_target
            self._apply_binary_policy_targets()
            self._processed_actions = self._processed_actions + prior_delta

        # Re-apply clip after adding the prior term.
        if self.cfg.clip is not None:
            self._processed_actions = torch.clamp(
                self._processed_actions, min=self._clip[:, :, 0], max=self._clip[:, :, 1]
            )

        self._process_servo_command()

        if isinstance(getattr(self._env, "extras", None), dict):
            self._env.extras["decap_factor"] = decap_lambda
            log_dict = self._env.extras.setdefault("log", {})
            if isinstance(log_dict, dict):
                log_dict["Metrics/decap/factor"] = decap_lambda
                if self.cfg.decap_decay_type == "exp":
                    log_dict["Metrics/decap/exp_decay_counter"] = decap_decay_counter

    def _configure_servo_command_model(self) -> None:
        self._servo_enabled = bool(self.cfg.servo_joint_names)
        if not self._servo_enabled:
            return

        missing = [name for name in self.cfg.servo_joint_names if name not in self._joint_names]
        if missing:
            raise ValueError(
                f"Servo command joints {missing} are not controlled by this action term. "
                f"Available joints: {self._joint_names}"
            )
        self._servo_action_indices = torch.tensor(
            [self._joint_names.index(name) for name in self.cfg.servo_joint_names],
            dtype=torch.long,
            device=self.device,
        )
        self._servo_num_joints = len(self.cfg.servo_joint_names)
        if isinstance(self.cfg.servo_command_quantization, (float, int)):
            self._servo_command_quantization = torch.full(
                (1, self._servo_num_joints),
                float(self.cfg.servo_command_quantization),
                dtype=torch.float32,
                device=self.device,
            )
        else:
            if len(self.cfg.servo_command_quantization) != self._servo_num_joints:
                raise ValueError(
                    "servo_command_quantization must be a scalar or have one value per servo joint; "
                    f"got {len(self.cfg.servo_command_quantization)}/{self._servo_num_joints}."
                )
            self._servo_command_quantization = torch.tensor(
                self.cfg.servo_command_quantization,
                dtype=torch.float32,
                device=self.device,
            ).unsqueeze(0)

        step_dt = float(self._env.step_dt)
        physics_dt = float(self._env.physics_dt)
        self._servo_command_period_steps = max(1, int(round(self.cfg.servo_command_period_s / step_dt)))
        resolved_period_s = self._servo_command_period_steps * step_dt
        if not math.isclose(resolved_period_s, float(self.cfg.servo_command_period_s), abs_tol=1.0e-8):
            raise ValueError(
                "servo_command_period_s must be an integer multiple of the environment step: "
                f"requested {self.cfg.servo_command_period_s}, resolved {resolved_period_s}."
            )
        self._servo_physics_dt = physics_dt

        min_latency_s, max_latency_s = self.cfg.servo_latency_s_range
        if min_latency_s < 0.0 or max_latency_s < min_latency_s:
            raise ValueError(f"Invalid servo_latency_s_range: {self.cfg.servo_latency_s_range}")
        self._servo_min_latency_steps = max(0, int(round(min_latency_s / step_dt)))
        self._servo_max_latency_steps = max(0, int(round(max_latency_s / step_dt)))
        self._servo_history_length = self._servo_max_latency_steps + 1
        self._servo_history_index = 0

        initial_q = self._processed_actions[:, self._servo_action_indices].clone()
        self._servo_target_history = initial_q.unsqueeze(0).repeat(self._servo_history_length, 1, 1)
        self._servo_latched_target = initial_q.clone()
        self._servo_filtered_target = initial_q.clone()
        self._servo_filtered_velocity = torch.zeros_like(initial_q)
        self._servo_command_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._servo_latency_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._servo_natural_frequency = torch.zeros_like(initial_q)
        self._servo_damping_ratio = torch.zeros_like(initial_q)
        self._servo_applied_actions = self._processed_actions.clone()
        self._reset_servo_command_model(None)

    def _env_id_tensor(self, env_ids: Sequence[int] | torch.Tensor | slice | None) -> torch.Tensor:
        all_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
        if env_ids is None:
            return all_ids
        if isinstance(env_ids, slice):
            return all_ids[env_ids]
        return torch.as_tensor(env_ids, dtype=torch.long, device=self.device)

    def _sample_servo_uniform(
        self,
        value_range: tuple[float, float],
        shape: tuple[int, ...],
    ) -> torch.Tensor:
        low, high = value_range
        if high < low:
            raise ValueError(f"Invalid servo randomization range: {value_range}")
        if math.isclose(low, high):
            return torch.full(shape, float(low), dtype=torch.float32, device=self.device)
        return torch.empty(shape, dtype=torch.float32, device=self.device).uniform_(float(low), float(high))

    def _reset_servo_command_model(
        self,
        env_ids: Sequence[int] | torch.Tensor | slice | None,
    ) -> None:
        if not self._servo_enabled:
            return
        ids = self._env_id_tensor(env_ids)
        if ids.numel() == 0:
            return

        current_action_q = self._asset.data.joint_pos[:, self._joint_ids]
        initial_q = current_action_q[ids][:, self._servo_action_indices]
        self._servo_latched_target[ids] = initial_q
        self._servo_filtered_target[ids] = initial_q
        self._servo_filtered_velocity[ids] = 0.0
        self._servo_command_counter[ids] = 0
        self._servo_target_history[:, ids, :] = initial_q.unsqueeze(0)

        sample_shape = (ids.numel(), self._servo_num_joints)
        frequency_hz = self._sample_servo_uniform(self.cfg.servo_natural_frequency_hz_range, sample_shape)
        self._servo_natural_frequency[ids] = 2.0 * math.pi * frequency_hz
        self._servo_damping_ratio[ids] = self._sample_servo_uniform(
            self.cfg.servo_damping_ratio_range,
            sample_shape,
        )
        if self._servo_max_latency_steps == self._servo_min_latency_steps:
            self._servo_latency_steps[ids] = self._servo_min_latency_steps
        else:
            self._servo_latency_steps[ids] = torch.randint(
                self._servo_min_latency_steps,
                self._servo_max_latency_steps + 1,
                (ids.numel(),),
                device=self.device,
            )

    def _process_servo_command(self) -> None:
        if not self._servo_enabled:
            return

        self._servo_history_index = (self._servo_history_index + 1) % self._servo_history_length
        desired_target = self._processed_actions[:, self._servo_action_indices]
        self._servo_target_history[self._servo_history_index] = desired_target

        latch_mask = self._servo_command_counter == 0
        if torch.any(latch_mask):
            env_ids = torch.arange(self.num_envs, dtype=torch.long, device=self.device)
            history_ids = torch.remainder(
                self._servo_history_index - self._servo_latency_steps,
                self._servo_history_length,
            )
            delayed_target = self._servo_target_history[history_ids, env_ids]
            quantum = self._servo_command_quantization
            delayed_target = torch.where(
                quantum > 0.0,
                torch.round(delayed_target / torch.clamp_min(quantum, 1.0e-12)) * quantum,
                delayed_target,
            )
            self._servo_latched_target[latch_mask] = delayed_target[latch_mask]

        self._servo_command_counter.add_(1)
        self._servo_command_counter.remainder_(self._servo_command_period_steps)

    def apply_actions(self) -> None:
        if not self._servo_enabled:
            super().apply_actions()
            return

        if not self.cfg.servo_filter_enabled:
            self._servo_applied_actions[:] = self._processed_actions
            self._servo_applied_actions[:, self._servo_action_indices] = self._servo_latched_target
            self._asset.set_joint_position_target(self._servo_applied_actions, joint_ids=self._joint_ids)
            return

        omega = self._servo_natural_frequency
        acceleration = (
            omega.square() * (self._servo_latched_target - self._servo_filtered_target)
            - 2.0 * self._servo_damping_ratio * omega * self._servo_filtered_velocity
        )
        self._servo_filtered_velocity.add_(acceleration * self._servo_physics_dt)
        self._servo_filtered_target.add_(self._servo_filtered_velocity * self._servo_physics_dt)

        self._servo_applied_actions[:] = self._processed_actions
        self._servo_applied_actions[:, self._servo_action_indices] = self._servo_filtered_target
        self._asset.set_joint_position_target(self._servo_applied_actions, joint_ids=self._joint_ids)

    def reset(self, env_ids: Sequence[int] | torch.Tensor | slice | None = None) -> None:
        super().reset(env_ids)
        self._reset_servo_command_model(env_ids)


@configclass
class DecapJointPositionActionCfg(JointPositionActionCfg):
    """Configuration for APEX-style decaying joint-position action prior."""

    class_type: type = DecapJointPositionAction

    decap_command_name: str = "motion"
    decap_lambda_start: float = 1.0
    decap_lambda_end: float = 0.0
    decap_decay_type: Literal["constant", "linear", "cosine", "exp"] = "exp"

    # Iteration-based schedule. One iteration is `decap_steps_per_iteration` env steps.
    decap_steps_per_iteration: int = 24
    decap_warmup_iterations: int | None = None
    decap_decay_start_iteration: int | None = None
    decap_decay_end_iteration: int | None = None
    decap_decay_iterations: int | None = None
    decap_resume_iteration: int = 0

    # Step-based fallback for old configs. These are ignored when the matching iteration field is set.
    decap_warmup_steps: int = 0
    decap_decay_start_step: int = 0
    decap_decay_end_step: int | None = None

    # Exponential schedule parameters. Used only when decap_decay_type == "exp".
    decap_exp_gamma: float = 0.99
    decap_exp_k: float = 100.0
    decap_prior_only: bool = False
    # None preserves legacy behavior (the scheduled prior applies to every
    # action joint). An empty tuple disables the scheduled prior.
    decap_joint_names: tuple[str, ...] | None = None
    # These joints always use q_ref + scale * action. This is independent of
    # the decaying prior and is intended for imitation residual learning.
    reference_residual_joint_names: tuple[str, ...] = ()

    # Optional binary actuator command. The actor output remains a Gaussian
    # logit, but the physical target is exactly open or closed. Keeping the
    # threshold at zero makes the initial policy unbiased between both states.
    binary_joint_names: tuple[str, ...] = ()
    binary_action_threshold: float = 0.0
    binary_open_position: float = 0.0
    binary_closed_position: float = 1.0

    # Optional deployment-style absolute-angle command model. Empty joint names
    # preserve the original direct target behavior.
    servo_joint_names: tuple[str, ...] = ()
    servo_command_period_s: float = 0.1
    servo_command_quantization: float | tuple[float, ...] = math.radians(0.1)
    # False models only the command-rate boundary as a zero-order hold. True
    # additionally models an identified/assumed second-order firmware response.
    servo_filter_enabled: bool = True
    servo_natural_frequency_hz_range: tuple[float, float] = (6.0, 6.0)
    servo_damping_ratio_range: tuple[float, float] = (1.0, 1.0)
    servo_latency_s_range: tuple[float, float] = (0.0, 0.0)
