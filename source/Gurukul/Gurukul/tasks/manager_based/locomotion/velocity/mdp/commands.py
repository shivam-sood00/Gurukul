from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

import Gurukul.tasks.manager_based.locomotion.velocity.mdp as mdp

from .utils import is_robot_on_terrain

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv


class UniformThresholdVelocityCommand(mdp.UniformVelocityCommand):
    """Command generator that generates a velocity command in SE(2) from uniform distribution with threshold.

    This command generator automatically detects "pits" terrain and applies restrictions:
    - For pit terrains: only allow forward movement (no lateral or rotational movement)
    """

    cfg: mdp.UniformThresholdVelocityCommandCfg  # type: ignore
    """The configuration of the command generator."""

    def __init__(self, cfg: mdp.UniformThresholdVelocityCommandCfg, env: ManagerBasedEnv):
        """Initialize the command generator.

        Args:
            cfg: The configuration of the command generator.
            env: The environment.
        """
        super().__init__(cfg, env)
        # Track which robots were on pit terrain in the previous step
        self.was_on_pit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample velocity commands with threshold."""
        super()._resample_command(env_ids)
        # set small commands to zero
        self.vel_command_b[env_ids, :2] *= (torch.norm(self.vel_command_b[env_ids, :2], dim=1) > 0.2).unsqueeze(1)

    def _update_command(self):
        """Update commands and apply terrain-aware restrictions in real-time.

        This function:
        1. Calls parent's update to handle heading and standing envs
        2. Checks which robots are currently on pit terrain
        3. For robots leaving pits: resamples their commands
        4. For robots on pits: restricts to forward-only movement and sets heading to 0
        """
        # First, call parent's update command
        super()._update_command()

        # Check which robots are currently on pit terrain (real-time check every step)
        on_pits = is_robot_on_terrain(self._env, "pits")

        # Find robots that just left pit terrain (need to resample)
        left_pit_mask = self.was_on_pit & ~on_pits
        if left_pit_mask.any():
            left_pit_env_ids = torch.where(left_pit_mask)[0]
            # Resample commands for robots that left pits
            self._resample_command(left_pit_env_ids)

        # For robots currently on pits: restrict to forward-only movement with min/max speed
        if on_pits.any():
            pit_env_ids = torch.where(on_pits)[0]
            # Force forward-only movement with min and max speed limits
            self.vel_command_b[pit_env_ids, 0] = torch.clamp(
                torch.abs(self.vel_command_b[pit_env_ids, 0]), min=0.3, max=0.6
            )
            self.vel_command_b[pit_env_ids, 1] = 0.0  # no lateral movement
            self.vel_command_b[pit_env_ids, 2] = 0.0  # no yaw rotation
            # Set heading to 0 for pit robots
            if self.cfg.heading_command:
                self.heading_target[pit_env_ids] = 0.0

        # Update tracking state
        self.was_on_pit = on_pits

    def _debug_vis_callback(self, event):
        """Draw velocity arrows at the configured height above the robot root."""
        if not self.robot.is_initialized:
            return

        base_pos_w = self.robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += float(self.cfg.marker_height_offset)
        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2])
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(
            self.robot.data.root_lin_vel_b[:, :2]
        )
        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)


@configclass
class UniformThresholdVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for the uniform threshold velocity command generator."""

    class_type: type = UniformThresholdVelocityCommand
    marker_height_offset: float = 0.5
    """World-frame height above the robot root used for velocity-arrow markers."""


class UniformVelocityPostureCommand(UniformThresholdVelocityCommand):
    """Velocity command with additional commanded base roll and pitch targets."""

    cfg: mdp.UniformVelocityPostureCommandCfg  # type: ignore

    def __init__(self, cfg: mdp.UniformVelocityPostureCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.posture_command = torch.zeros(self.num_envs, 2, device=self.device)
        self.metrics["error_roll"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_pitch"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Desired base velocity and posture command: ``[vx, vy, wz, roll, pitch]``."""
        return torch.cat((self.vel_command_b, self.posture_command), dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        super()._resample_command(env_ids)
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids_t.numel() == 0:
            return
        r = torch.empty(env_ids_t.numel(), device=self.device)
        self.posture_command[env_ids_t, 0] = r.uniform_(*self.cfg.roll_range)
        self.posture_command[env_ids_t, 1] = r.uniform_(*self.cfg.pitch_range)
        if self.cfg.zero_posture_probability > 0.0:
            zero_mask = r.uniform_(0.0, 1.0) <= float(self.cfg.zero_posture_probability)
            self.posture_command[env_ids_t[zero_mask], :] = 0.0

    def _update_metrics(self):
        super()._update_metrics()
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        gravity = self.robot.data.projected_gravity_b
        self.metrics["error_roll"] += (
            torch.abs(self.posture_command[:, 0] - torch.atan2(gravity[:, 1], -gravity[:, 2])) / max_command_step
        )
        self.metrics["error_pitch"] += torch.abs(
            self.posture_command[:, 1] - torch.atan2(-gravity[:, 0], -gravity[:, 2])
        ) / max_command_step


@configclass
class UniformVelocityPostureCommandCfg(UniformThresholdVelocityCommandCfg):
    """Configuration for velocity commands augmented with base roll and pitch targets."""

    class_type: type = UniformVelocityPostureCommand
    roll_range: tuple[float, float] = (0.0, 0.0)
    pitch_range: tuple[float, float] = (0.0, 0.0)
    zero_posture_probability: float = 0.15


class UniformVelocityBodyPostureCommand(UniformThresholdVelocityCommand):
    """Velocity command augmented with base roll, pitch, and height targets."""

    cfg: mdp.UniformVelocityBodyPostureCommandCfg  # type: ignore

    def __init__(self, cfg: mdp.UniformVelocityBodyPostureCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self.posture_command = torch.zeros(self.num_envs, 3, device=self.device)
        self.metrics["error_roll"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_pitch"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_height"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        """Desired base command: ``[vx, vy, wz, roll, pitch, height]``."""
        return torch.cat((self.vel_command_b, self.posture_command), dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        UniformThresholdVelocityCommand._resample_command(self, env_ids)
        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids_t.numel() == 0:
            return
        r = torch.empty(env_ids_t.numel(), device=self.device)
        self.posture_command[env_ids_t, 0] = r.uniform_(*self.cfg.roll_range)
        self.posture_command[env_ids_t, 1] = r.uniform_(*self.cfg.pitch_range)
        self.posture_command[env_ids_t, 2] = r.uniform_(*self.cfg.height_range)
        if self.cfg.zero_posture_probability > 0.0:
            zero_mask = r.uniform_(0.0, 1.0) <= float(self.cfg.zero_posture_probability)
            self.posture_command[env_ids_t[zero_mask], 0] = 0.0
            self.posture_command[env_ids_t[zero_mask], 1] = 0.0
            self.posture_command[env_ids_t[zero_mask], 2] = float(self.cfg.nominal_height)

        # A global curriculum frontier controls the configured pitch/height
        # envelope, while a replay subset may use an easier per-environment
        # difficulty. Scale newly sampled posture commands back toward nominal
        # for those replay environments. Roll is deliberately not commanded.
        difficulty_attr = self.cfg.curriculum_difficulty_attr
        if difficulty_attr:
            difficulty = getattr(self._env, difficulty_attr, 0.0)
            if isinstance(difficulty, torch.Tensor):
                difficulty = difficulty.to(device=self.device, dtype=torch.float32).reshape(-1)
                if difficulty.numel() == 1:
                    difficulty = difficulty.repeat(self.num_envs)
                elif difficulty.numel() != self.num_envs:
                    raise ValueError(
                        f"{difficulty_attr} must contain either 1 or {self.num_envs} values, "
                        f"got {difficulty.numel()}."
                    )
            else:
                difficulty = torch.full(
                    (self.num_envs,), float(difficulty), device=self.device, dtype=torch.float32
                )

            frontier_attr = self.cfg.curriculum_frontier_difficulty_attr
            frontier = getattr(self._env, frontier_attr, None) if frontier_attr else None
            if frontier is None:
                frontier_value = float(torch.max(difficulty).item())
            elif isinstance(frontier, torch.Tensor):
                frontier_value = float(torch.max(frontier).item())
            else:
                frontier_value = float(frontier)
            if frontier_value <= 1.0e-6:
                replay_scale = torch.zeros(env_ids_t.numel(), device=self.device)
            else:
                replay_scale = (difficulty[env_ids_t] / frontier_value).clamp(0.0, 1.0)

            self.posture_command[env_ids_t, 0] = 0.0
            self.posture_command[env_ids_t, 1] *= replay_scale
            nominal_height = float(self.cfg.nominal_height)
            self.posture_command[env_ids_t, 2] = nominal_height + (
                self.posture_command[env_ids_t, 2] - nominal_height
            ) * replay_scale

    def _update_metrics(self):
        super()._update_metrics()
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max_command_time / self._env.step_dt
        gravity = self.robot.data.projected_gravity_b
        self.metrics["error_roll"] += (
            torch.abs(self.posture_command[:, 0] - torch.atan2(gravity[:, 1], -gravity[:, 2])) / max_command_step
        )
        self.metrics["error_pitch"] += torch.abs(
            self.posture_command[:, 1] - torch.atan2(-gravity[:, 0], -gravity[:, 2])
        ) / max_command_step
        self.metrics["error_height"] += torch.abs(
            self.posture_command[:, 2] - self.robot.data.root_pos_w[:, 2]
        ) / max_command_step


@configclass
class UniformVelocityBodyPostureCommandCfg(UniformVelocityPostureCommandCfg):
    """Configuration for velocity commands augmented with base roll, pitch, and height targets."""

    class_type: type = UniformVelocityBodyPostureCommand
    height_range: tuple[float, float] = (0.33, 0.33)
    nominal_height: float = 0.33
    curriculum_difficulty_attr: str | None = None
    curriculum_frontier_difficulty_attr: str | None = None


class StartSparseVelocityCommand(mdp.UniformVelocityCommand):
    """START command sampler: forward-only walking or in-place turning."""

    cfg: mdp.StartSparseVelocityCommandCfg  # type: ignore

    def _resample_command(self, env_ids: Sequence[int]):
        # Sample from configured ranges first.
        super()._resample_command(env_ids)

        env_ids_t = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        if env_ids_t.numel() == 0:
            return

        # Paper command setup: no lateral velocity.
        self.vel_command_b[env_ids_t, 1] = 0.0

        # If forward command is small, switch to turning mode by setting vx=0 and sampling yaw rate.
        forward_cmd = self.vel_command_b[env_ids_t, 0]
        turn_mask = forward_cmd < float(self.cfg.turn_trigger_threshold)
        walk_mask = ~turn_mask

        if torch.any(turn_mask):
            turn_ids = env_ids_t[turn_mask]
            self.vel_command_b[turn_ids, 0] = 0.0
            yaw_low, yaw_high = self.cfg.ranges.ang_vel_z
            self.vel_command_b[turn_ids, 2] = torch.empty(len(turn_ids), device=self.device).uniform_(
                float(yaw_low), float(yaw_high)
            )

        if torch.any(walk_mask):
            walk_ids = env_ids_t[walk_mask]
            self.vel_command_b[walk_ids, 2] = 0.0


@configclass
class StartSparseVelocityCommandCfg(mdp.UniformVelocityCommandCfg):
    """Configuration for START sparse-foothold command sampling."""

    class_type: type = StartSparseVelocityCommand
    turn_trigger_threshold: float = 0.3


class DiscreteCommandController(CommandTerm):
    """
    Command generator that assigns discrete commands to environments.

    Commands are stored as a list of predefined integers.
    The controller maps these commands by their indices (e.g., index 0 -> 10, index 1 -> 20).
    """

    cfg: DiscreteCommandControllerCfg
    """Configuration for the command controller."""

    def __init__(self, cfg: DiscreteCommandControllerCfg, env: ManagerBasedEnv):
        """
        Initialize the command controller.

        Args:
            cfg: The configuration of the command controller.
            env: The environment object.
        """
        # Initialize the base class
        super().__init__(cfg, env)

        # Validate that available_commands is non-empty
        if not self.cfg.available_commands:
            raise ValueError("The available_commands list cannot be empty.")

        # Ensure all elements are integers
        if not all(isinstance(cmd, int) for cmd in self.cfg.available_commands):
            raise ValueError("All elements in available_commands must be integers.")

        # Store the available commands
        self.available_commands = self.cfg.available_commands

        # Create buffers to store the command
        # -- command buffer: stores discrete action indices for each environment
        self.command_buffer = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)

        # -- current_commands: stores a snapshot of the current commands (as integers)
        self.current_commands = [self.available_commands[0]] * self.num_envs  # Default to the first command

    def __str__(self) -> str:
        """Return a string representation of the command controller."""
        return (
            "DiscreteCommandController:\n"
            f"\tNumber of environments: {self.num_envs}\n"
            f"\tAvailable commands: {self.available_commands}\n"
        )

    """
    Properties
    """

    @property
    def command(self) -> torch.Tensor:
        """Return the current command buffer. Shape is (num_envs, 1)."""
        return self.command_buffer

    """
    Implementation specific functions.
    """

    def _update_metrics(self):
        """Update metrics for the command controller."""
        pass

    def _resample_command(self, env_ids: Sequence[int]):
        """Resample commands for the given environments."""
        sampled_indices = torch.randint(
            len(self.available_commands), (len(env_ids),), dtype=torch.int32, device=self.device
        )
        sampled_commands = torch.tensor(
            [self.available_commands[idx.item()] for idx in sampled_indices], dtype=torch.int32, device=self.device
        )
        self.command_buffer[env_ids] = sampled_commands

    def _update_command(self):
        """Update and store the current commands."""
        self.current_commands = self.command_buffer.tolist()


@configclass
class DiscreteCommandControllerCfg(CommandTermCfg):
    """Configuration for the discrete command controller."""

    class_type: type = DiscreteCommandController

    available_commands: list[int] = []
    """
    List of available discrete commands, where each element is an integer.
    Example: [10, 20, 30, 40, 50]
    """
