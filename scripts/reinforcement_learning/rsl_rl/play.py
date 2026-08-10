# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint if an RL agent from RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import re
import sys
import tempfile
from collections.abc import Mapping, Sequence

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument(
    "--use_pretrained_checkpoint",
    action="store_true",
    help="Use the pre-trained checkpoint from Nucleus.",
)
parser.add_argument("--real-time", action="store_true", default=False, help="Run in real-time, if possible.")
parser.add_argument("--keyboard", action="store_true", default=False, help="Whether to use keyboard.")
parser.add_argument(
    "--velocity-demo",
    "--velocity_demo",
    dest="velocity_demo",
    action="store_true",
    default=False,
    help=(
        "Drive commands through a video-friendly sequence: fast forward, forward turns, in-place rotations, "
        "fast backward, and stop. Commands stay within the task's configured training ranges."
    ),
)
parser.add_argument(
    "--go2-d1-live-control",
    action="store_true",
    default=False,
    help="Use Xbox live base velocity and D1 end-effector commands for Go2+D1 WBC play.",
)
parser.add_argument(
    "--go2-d1-xbox-device-id",
    type=int,
    default=0,
    help="pygame joystick index for Go2+D1 live control.",
)
parser.add_argument("--go2-d1-xbox-deadzone", type=float, default=0.08, help="Xbox stick deadzone.")
parser.add_argument("--go2-d1-ee-speed", type=float, default=0.15, help="Xbox EE jog speed in m/s.")
parser.add_argument("--go2-d1-ee-step", type=float, default=0.02, help="Keyboard EE jog step in meters.")
parser.add_argument(
    "--camera_follow_mode",
    "--camera-follow-mode",
    dest="camera_follow_mode",
    type=str,
    default="auto",
    choices=["auto", "mouse", "none", "follow", "isometric", "topdown"],
    help=(
        "Viewport camera mode. 'auto' preserves the old behavior (follow when --keyboard is used), "
        "'mouse'/'none' leave the IsaacLab viewport camera under mouse control, and the remaining modes "
        "programmatically follow env 0."
    ),
)
parser.add_argument(
    "--camera_smooth_window",
    "--camera-smooth-window",
    dest="camera_smooth_window",
    type=int,
    default=50,
    help="Smoothing window size for programmatic camera follow modes.",
)
parser.add_argument(
    "--camera_follow_distance_scale",
    "--camera-follow-distance-scale",
    dest="camera_follow_distance_scale",
    type=float,
    default=1.0,
    help="Multiply the follow-camera offset to zoom out (>1) or in (<1) without changing its angle.",
)
parser.add_argument(
    "--motion-debug-vis",
    action="store_true",
    default=False,
    help="Enable commands.motion debug visualization when available. APEX tasks enable it by default during play.",
)
parser.add_argument(
    "--motion-velocity-vis-only",
    action="store_true",
    default=False,
    help="With motion debug visualization, show only velocity arrows (hide pose/frame overlays) when supported.",
)
parser.add_argument(
    "--motion-foot-ref-vis",
    action="store_true",
    default=False,
    help="With motion debug visualization, enable simplified reference-foot sphere visualization when supported.",
)
parser.add_argument(
    "--heightmap-debug-vis",
    action="store_true",
    default=False,
    help="Enable height-scanner point visualization for tasks using heightmap observations.",
)
parser.add_argument(
    "--lidar-debug-vis",
    action="store_true",
    default=False,
    help="Enable lidar ray/hit visualization for tasks with a scene.lidar sensor.",
)
parser.add_argument(
    "--contact-trail-vis",
    action="store_true",
    default=False,
    help="Show the contact-trail policy memory map in an OpenCV window during play.",
)
parser.add_argument(
    "--contact-trail-vis-interval",
    type=int,
    default=5,
    help="Render contact-trail map every N play steps when --contact-trail-vis is enabled.",
)
parser.add_argument(
    "--contact-trail-vis-env-idx",
    type=int,
    default=0,
    help="Environment index to visualize for --contact-trail-vis.",
)
parser.add_argument(
    "--contact-trail-vis-output-dir",
    type=str,
    default=None,
    help="Optional directory for contact-trail PNG frame dumps during play.",
)
parser.add_argument(
    "--contact-trail-vis-save-interval",
    type=int,
    default=50,
    help="Save contact-trail PNG frames every N play steps when --contact-trail-vis-output-dir is set.",
)
parser.add_argument(
    "--motion-file",
    type=str,
    default=None,
    help="Optional NPZ file to override env commands.motion.motion_file.",
)
parser.add_argument(
    "--smp-prior",
    type=str,
    default=None,
    help="Frozen, morphology-specific SMP prior checkpoint required by SMP tasks.",
)
parser.add_argument(
    "--smp-gsi-pool-size",
    type=int,
    default=None,
    help="Override the number of generated SMP reset windows; use 0 to disable GSI.",
)
parser.add_argument(
    "--motion-reset-curriculum",
    action="store_true",
    default=False,
    help="Preserve a motion task's randomized training reset phases during play.",
)
parser.add_argument(
    "--prior-only",
    action="store_true",
    default=False,
    help="For APEX action-prior tasks, ignore the policy action and execute only the imitation prior during play.",
)
parser.add_argument(
    "--print-rewards",
    action="store_true",
    default=False,
    help="Print the env-0 reward term breakdown during play.",
)
parser.add_argument(
    "--reward-print-interval",
    type=int,
    default=10,
    help="Print reward terms every N environment steps when --print-rewards is enabled.",
)
parser.add_argument(
    "--max_steps",
    type=int,
    default=None,
    help="Optional maximum number of play-loop environment steps before exiting.",
)
parser.add_argument(
    "--apex-delta-log",
    type=str,
    default=None,
    help="Optional NPZ path for APEX tracker actor-observation/action delta diagnostics.",
)
parser.add_argument(
    "--apex-delta-log-steps",
    type=int,
    default=1000,
    help="Maximum number of APEX tracker diagnostic samples to write when --apex-delta-log is set.",
)
parser.add_argument(
    "--loco-manip-stage",
    "--go2-airbot-stage",
    dest="loco_manip_stage",
    type=str,
    default="combined",
    choices=["fixed", "arm", "combined", "cycle", "curriculum", "grid", "off"],
    help=(
        "Play-time stage override for loco-manipulation tasks: fixed=walking with stowed arm, "
        "arm=zero base command with moving arm, combined=walking with moving arm, "
        "cycle=repeat those three stages, curriculum=use training curriculum timing, "
        "grid=spread envs from stage-0/difficulty-0 to stage-2/difficulty-1, off=legacy play behavior."
    ),
)
parser.add_argument(
    "--loco-manip-stage-cycle-steps",
    "--go2-airbot-stage-cycle-steps",
    dest="loco_manip_stage_cycle_steps",
    type=int,
    default=500,
    help="Number of play steps per loco-manipulation stage when --loco-manip-stage=cycle.",
)
parser.add_argument(
    "--loco-manip-grid-probe-steps",
    type=int,
    default=250,
    help=(
        "Steps per configured command-extremum probe phase in --loco-manip-stage=grid. The sequence tests each "
        "velocity/yaw axis independently, then both combined corners."
    ),
)
parser.add_argument(
    "--loco-manip-arm-difficulty",
    "--go2-airbot-arm-difficulty",
    dest="loco_manip_arm_difficulty",
    type=float,
    default=0.35,
    help=(
        "Arm motion difficulty used by mounted-arm play-stage overrides. This scales the task's arm and posture "
        "envelopes; use 1.0 to inspect the full training range."
    ),
)
parser.add_argument(
    "--go2-d1-play-domain-randomization",
    action="store_true",
    default=False,
    help="Retain Go2+D1 WBC training-time mass, gain, reset, and disturbance randomization during playback.",
)
parser.add_argument(
    "--loco-manip-curriculum-speedup",
    type=float,
    default=1.0,
    help=(
        "Play-only multiplier for mounted-arm curriculum timing when --loco-manip-stage=curriculum. "
        "For example, 80 turns the Go2+D1 ArmMoving 40k/120k step schedule into roughly 500/1500 steps."
    ),
)
parser.add_argument(
    "--loco-manip-curriculum-status-interval",
    type=int,
    default=250,
    help="Print mounted-arm curriculum stage/difficulty every N play steps when using curriculum mode.",
)
parser.add_argument(
    "--b2-z1-legacy-ik-arm-moving",
    action="store_true",
    default=False,
    help=(
        "Restore the pre-joint-primitive B2-Z1 ArmMoving play config for old checkpoints. This uses the legacy "
        "task-space IK arm event, posture command observations, and disables B2/Z1 self-collisions."
    ),
)
parser.add_argument(
    "--wandb_run",
    type=str,
    default=None,
    help="Weights & Biases run path in the form <entity>/<project>/<run_id>.",
)
parser.add_argument(
    "--wandb_checkpoint",
    type=str,
    default=None,
    help=(
        "Checkpoint file name or regex to fetch from --wandb_run (defaults to agent_cfg.load_checkpoint). "
        "Examples: model.pt, model_5000.pt, model_.*\\.pt"
    ),
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli, hydra_args = parser.parse_known_args()
# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

import importlib.metadata as metadata

from packaging import version

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import numpy as np
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

import isaaclab.sim as sim_utils

try:
    import cv2
except ModuleNotFoundError:
    cv2 = None

from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.managers import ObservationTermCfg as ObsTerm, SceneEntityCfg
from isaaclab.managers import EventTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import quat_apply_inverse

from isaaclab_rl.rsl_rl import (
    RslRlBaseRunnerCfg,
    RslRlVecEnvWrapper,
    export_policy_as_jit,
    export_policy_as_onnx,
)

try:
    # Isaac Lab >= 2.3
    from isaaclab.utils.pretrained_checkpoint import get_published_pretrained_checkpoint
except ModuleNotFoundError:
    # Backward compatibility with older Isaac Lab layouts
    from isaaclab_rl.utils.pretrained_checkpoint import get_published_pretrained_checkpoint

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config
from rsl_rl_config_compat import migrate_custom_policy_cfg

import Gurukul.tasks  # noqa: F401  # isort: skip
import Gurukul.tasks.manager_based.locomotion.velocity.mdp as mdp
from Gurukul.assets.unitree import GO2_D1_ARM_JOINT_DEFAULTS
from Gurukul.tasks.manager_based.locomotion.velocity.config.quadruped_with_arm.unitree_go2_d1_arm.whole_body_controller_env_cfg import (
    GO2_D1_WBC_BODY_CLEARANCE,
    GO2_D1_WBC_BODY_EXCLUSION_BOX,
    GO2_D1_WBC_EE_POS_RANGE,
    GO2_D1_WBC_NEUTRAL_EE_POS,
    GO2_D1_WBC_REACH_RANGE,
    GO2_D1_WBC_WORKSPACE_ORIGIN,
)
from Gurukul.tasks.manager_based.locomotion.velocity.real_teacher_viz import (
    DEFAULT_HEAD_COLORS_BGR,
    compute_attention_marker_overlay,
    render_attention_point_overlay,
    resolve_scan_shape,
)
from Gurukul.utils.export_deploy_cfg import maybe_export_deploy_cfg

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from legacy_checkpoint import load_checkpoint_for_play
from rl_utils import camera_follow

# PLACEHOLDER: Extension template (do not remove this comment)


class ApexTrackerDeltaLogger:
    """Record APEX tracker actor slices and per-step max deltas for env 0."""

    def __init__(self, path: str | None, max_steps: int = 1000):
        self.path = path
        self.max_steps = max(0, int(max_steps))
        self.enabled = bool(path) and self.max_steps > 0
        self.steps: list[int] = []
        self.values: dict[str, list[np.ndarray]] = {
            "base_ang_vel": [],
            "projected_gravity": [],
            "command": [],
            "joint_pos": [],
            "joint_vel": [],
            "prev_action": [],
            "skill": [],
            "reference_joint_pos": [],
            "reference_base_lin_vel": [],
            "reference_base_ang_vel": [],
            "reference_quat": [],
            "action": [],
        }
        self.deltas: dict[str, list[float]] = {name: [] for name in self.values}
        self._prev: dict[str, np.ndarray] | None = None

    @staticmethod
    def _actor_obs_array(obs) -> np.ndarray:
        policy_obs = obs["policy"] if isinstance(obs, Mapping) and "policy" in obs else obs
        return ApexTrackerDeltaLogger._flatten_obs_value(policy_obs)

    @staticmethod
    def _flatten_obs_value(value) -> np.ndarray:
        if isinstance(value, torch.Tensor):
            actor_value = value if value.ndim == 1 else value[0]
            return actor_value.detach().cpu().numpy().astype(np.float32, copy=True).reshape(-1)
        if isinstance(value, Mapping):
            parts = [ApexTrackerDeltaLogger._flatten_obs_value(term) for term in value.values()]
            return np.concatenate(parts).astype(np.float32, copy=False) if parts else np.zeros(0, dtype=np.float32)
        if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
            parts = [ApexTrackerDeltaLogger._flatten_obs_value(term) for term in value]
            return np.concatenate(parts).astype(np.float32, copy=False) if parts else np.zeros(0, dtype=np.float32)
        if hasattr(value, "values"):
            parts = [ApexTrackerDeltaLogger._flatten_obs_value(term) for term in value.values()]
            return np.concatenate(parts).astype(np.float32, copy=False) if parts else np.zeros(0, dtype=np.float32)
        if hasattr(value, "detach"):
            tensor = value.detach()
            actor_value = tensor if tensor.ndim == 1 else tensor[0]
            return actor_value.cpu().numpy().astype(np.float32, copy=True).reshape(-1)
        if isinstance(value, np.ndarray):
            array = value.astype(np.float32, copy=True)
            if array.ndim > 1:
                array = array[0]
            return array.reshape(-1)
        try:
            array = np.asarray(value, dtype=np.float32)
        except Exception as exc:
            raise TypeError(
                "Unsupported APEX delta logger observation value type "
                f"{type(value).__module__}.{type(value).__qualname__}"
            ) from exc
        if array.ndim > 1:
            array = array[0]
        return array.reshape(-1).copy()

    @staticmethod
    def _action_array(actions) -> np.ndarray:
        action = actions if actions.ndim == 1 else actions[0]
        return action.detach().cpu().numpy().astype(np.float32, copy=True).reshape(-1)

    @staticmethod
    def _slices(actor_obs: np.ndarray, action: np.ndarray) -> dict[str, np.ndarray]:
        joint_count = int(action.size)
        reference_frame_dim = joint_count + 10
        layout = None
        # Go2 uses a 3-D command with no skill; Go2+D1 uses a 4-D tracker
        # command and one skill scalar. Infer the layout so this diagnostic
        # remains valid for both actor contracts.
        for joint_position_count in (joint_count, joint_count + 1):
            for joint_velocity_count in (joint_count, 12):
                for command_dim, skill_dim in ((4, 1), (3, 0), (4, 0), (3, 1)):
                    current_dim = (
                        6
                        + command_dim
                        + joint_position_count
                        + joint_velocity_count
                        + joint_count
                        + skill_dim
                    )
                    reference_size = actor_obs.size - current_dim
                    if reference_size > 0 and reference_size % reference_frame_dim == 0:
                        layout = (
                            command_dim,
                            skill_dim,
                            joint_position_count,
                            joint_velocity_count,
                            current_dim,
                        )
                        break
                if layout is not None:
                    break
            if layout is not None:
                break
        if layout is None:
            raise ValueError(
                f"Cannot infer APEX observation layout from obs={actor_obs.size}, action={joint_count}."
            )

        command_dim, skill_dim, joint_position_count, joint_velocity_count, current_dim = layout
        joint_pos_start = 6 + command_dim
        joint_vel_start = joint_pos_start + joint_position_count
        prev_action_start = joint_vel_start + joint_velocity_count
        skill_start = prev_action_start + joint_count
        reference = actor_obs[current_dim:]
        reference_frames = reference.reshape(-1, reference_frame_dim)
        return {
            "base_ang_vel": actor_obs[0:3].copy(),
            "projected_gravity": actor_obs[3:6].copy(),
            "command": actor_obs[6:joint_pos_start].copy(),
            "joint_pos": actor_obs[joint_pos_start:joint_vel_start].copy(),
            "joint_vel": actor_obs[joint_vel_start:prev_action_start].copy(),
            "prev_action": actor_obs[prev_action_start:skill_start].copy(),
            "skill": actor_obs[skill_start:current_dim].copy(),
            "reference_joint_pos": reference_frames[:, 0:joint_count].reshape(-1).copy(),
            "reference_base_lin_vel": reference_frames[:, joint_count : joint_count + 3].reshape(-1).copy(),
            "reference_base_ang_vel": reference_frames[
                :, joint_count + 3 : joint_count + 6
            ].reshape(-1).copy(),
            "reference_quat": reference_frames[:, joint_count + 6 : joint_count + 10].reshape(-1).copy(),
            "action": action.copy(),
        }

    def record(self, step: int, obs, actions) -> None:
        if not self.enabled or len(self.steps) >= self.max_steps:
            return
        current = self._slices(self._actor_obs_array(obs), self._action_array(actions))
        self.steps.append(int(step))
        for name, value in current.items():
            self.values[name].append(value)
            if self._prev is None or self._prev[name].shape != value.shape:
                self.deltas[name].append(0.0)
            else:
                self.deltas[name].append(float(np.max(np.abs(value - self._prev[name]))))
        self._prev = {name: value.copy() for name, value in current.items()}

    def close(self) -> None:
        if not self.enabled:
            return
        output_path = os.path.abspath(os.path.expanduser(self.path))
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        payload = {"steps": np.asarray(self.steps, dtype=np.int64)}
        for name, values in self.values.items():
            payload[f"{name}_values"] = np.stack(values) if values else np.zeros((0, 0), dtype=np.float32)
            payload[f"{name}_max_abs_delta"] = np.asarray(self.deltas[name], dtype=np.float32)
        np.savez(output_path, **payload)
        summary_parts = []
        for name in self.values:
            delta_values = payload[f"{name}_max_abs_delta"]
            max_delta = float(np.max(delta_values)) if delta_values.size else 0.0
            summary_parts.append(f"{name}={max_delta:.5g}")
        summary = ", ".join(summary_parts)
        print(f"[APEX delta] wrote {len(self.steps)} samples to {output_path}; max deltas: {summary}")


def _resolve_motion_source_path(path: str) -> str:
    """Resolve local motion files/directories while preserving Isaac asset-path support."""
    expanded_path = os.path.expanduser(path)
    if os.path.exists(expanded_path):
        return expanded_path
    return retrieve_file_path(path)


def _describe_motion_source(motion_cfg) -> str:
    motion_files = tuple(getattr(motion_cfg, "motion_files", ()))
    if len(motion_files) > 0:
        return ", ".join(str(path) for path in motion_files)
    return str(getattr(motion_cfg, "motion_file", ""))


def _configure_smp_prior(env_cfg) -> None:
    """Apply SMP-only CLI overrides to the startup event configuration."""
    events_cfg = getattr(env_cfg, "events", None)
    initialize_cfg = getattr(events_cfg, "initialize_smp", None)
    requested = args_cli.smp_prior is not None or args_cli.smp_gsi_pool_size is not None
    if initialize_cfg is None:
        if requested:
            raise ValueError("--smp-prior/--smp-gsi-pool-size require an SMP task.")
        return

    configured_path = args_cli.smp_prior or initialize_cfg.params.get("checkpoint_path")
    if not configured_path:
        raise ValueError("SMP tasks require a morphology-matched prior via --smp-prior.")
    checkpoint_path = _resolve_motion_source_path(configured_path)
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(f"SMP prior checkpoint is not a file: {checkpoint_path}")
    initialize_cfg.params["checkpoint_path"] = checkpoint_path
    # Evaluation must use the score statistics restored from the policy
    # checkpoint without changing the reward scale as the rollout proceeds.
    initialize_cfg.params["update_score_normalizer"] = False
    print(f"[INFO] Using frozen SMP prior: {checkpoint_path}")
    if args_cli.smp_gsi_pool_size is not None:
        if args_cli.smp_gsi_pool_size < 0:
            raise ValueError("--smp-gsi-pool-size must be non-negative.")
        initialize_cfg.params["gsi_pool_size"] = args_cli.smp_gsi_pool_size
        print(f"[INFO] SMP GSI pool size: {args_cli.smp_gsi_pool_size}")


def _resolve_wandb_checkpoint(run_path: str, checkpoint_selector: str) -> str:
    """Download and return a checkpoint from a W&B run."""
    try:
        import wandb
    except ModuleNotFoundError as exc:
        raise ModuleNotFoundError(
            "wandb is required for --wandb_run checkpoint loading. Please install wandb in this environment."
        ) from exc

    api = wandb.Api()
    run = api.run(run_path)
    files = list(run.files())
    file_names = [f.name for f in files]

    # 1) Exact file-name match takes priority.
    exact_matches = [f for f in files if f.name == checkpoint_selector]
    if exact_matches:
        selected = exact_matches[0]
    else:
        # 2) Regex match.
        pattern = re.compile(checkpoint_selector)
        matched = [f for f in files if pattern.fullmatch(f.name)]
        if not matched:
            raise FileNotFoundError(
                f"No W&B checkpoint matched selector '{checkpoint_selector}' in run '{run_path}'. "
                f"Available files: {file_names}"
            )

        # Prefer highest model step if filenames follow model_<step>.pt.
        def _sort_key(wb_file):
            name = wb_file.name
            match = re.search(r"model_(\d+)\.pt$", name)
            if match:
                return (2, int(match.group(1)))
            if name == "model.pt":
                return (1, 0)
            return (0, name)

        selected = sorted(matched, key=_sort_key)[-1]

    download_root = tempfile.mkdtemp(prefix="Gurukul_wandb_ckpt_")
    downloaded = selected.download(root=download_root, replace=True)
    local_path = getattr(downloaded, "name", "")
    if not local_path or not os.path.exists(local_path):
        local_path = os.path.join(download_root, selected.name)
    return os.path.abspath(local_path)


def _set_attr_if_exists(obj, attr_name: str, value) -> bool:
    """Set an attribute only if it exists."""
    if hasattr(obj, attr_name):
        setattr(obj, attr_name, value)
        return True
    return False


def _configure_play_terrain(env_cfg) -> None:
    """Reduce generated terrain for play when the environment defines one."""
    terrain_cfg = getattr(getattr(env_cfg, "scene", None), "terrain", None)
    if terrain_cfg is None:
        return

    _set_attr_if_exists(terrain_cfg, "max_init_terrain_level", None)
    terrain_generator = getattr(terrain_cfg, "terrain_generator", None)
    if terrain_generator is not None:
        _set_attr_if_exists(terrain_generator, "num_rows", 5)
        _set_attr_if_exists(terrain_generator, "num_cols", 5)
        _set_attr_if_exists(terrain_generator, "curriculum", False)


def _disable_play_observation_corruption(env_cfg) -> None:
    """Disable policy observation corruption when using manager-based observations."""
    policy_observations = getattr(getattr(env_cfg, "observations", None), "policy", None)
    _set_attr_if_exists(policy_observations, "enable_corruption", False)


def _is_go2_d1_wbc_task() -> bool:
    task_name = (args_cli.task or "").lower()
    return "wbc" in task_name and "d1" in task_name and "locomanip" not in task_name


def _disable_go2_d1_play_randomization(env_cfg) -> tuple[str, ...]:
    """Use a nominal robot for visual policy assessment while retaining command sampling."""
    disabled = []
    for event_name in (
        "randomize_rigid_body_material",
        "randomize_rigid_body_mass_base",
        "randomize_rigid_body_mass_others",
        "randomize_com_positions",
        "randomize_actuator_gains",
        "randomize_reset_base",
        "randomize_reset_joints",
        "randomize_push_robot",
        "base_external_force_torque",
        "push_robot",
        "randomize_apply_external_force_torque",
    ):
        if hasattr(env_cfg.events, event_name) and getattr(env_cfg.events, event_name) is not None:
            setattr(env_cfg.events, event_name, None)
            disabled.append(event_name)
    return tuple(disabled)


def _is_loco_manip_play_task(env_cfg) -> bool:
    # Learned hierarchical actions own the base, EE, and gripper commands. The
    # mounted-arm stage override is only for inspecting low-level/scripted tasks.
    actions_cfg = getattr(env_cfg, "actions", None)
    if getattr(actions_cfg, "wbc_command", None) is not None:
        return False

    commands_cfg = getattr(env_cfg, "commands", None)
    if getattr(commands_cfg, "base_velocity", None) is None:
        return False

    curriculum_cfg = getattr(env_cfg, "curriculum", None)
    if curriculum_cfg is not None and hasattr(curriculum_cfg, "loco_manipulation_training_stages"):
        return True
    task_name = (args_cli.task or "").lower()
    return ("go2" in task_name and ("airbot" in task_name or "d1" in task_name)) or (
        "b2" in task_name and "z1" in task_name
    )


def _is_b2_z1_arm_moving_task() -> bool:
    task_name = (args_cli.task or "").lower()
    return "b2" in task_name and "z1" in task_name and "armmoving" in task_name.replace("-", "")


def _checkpoint_actor_input_dim(checkpoint_path: str) -> int | None:
    try:
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    except Exception as exc:
        print(f"[WARN] Could not inspect checkpoint actor input size before play config selection: {exc}")
        return None
    if not isinstance(checkpoint, Mapping):
        return None

    actor_state_dict = checkpoint.get("actor_state_dict")
    if isinstance(actor_state_dict, Mapping):
        weight = actor_state_dict.get("mlp.0.weight")
        if hasattr(weight, "shape") and len(weight.shape) == 2:
            return int(weight.shape[1])

    model_state_dict = checkpoint.get("model_state_dict")
    if isinstance(model_state_dict, Mapping):
        for key in ("actor.0.weight", "actor.0.linear.weight", "mlp.0.weight"):
            weight = model_state_dict.get(key)
            if hasattr(weight, "shape") and len(weight.shape) == 2:
                return int(weight.shape[1])
    return None


def _configure_legacy_b2_z1_arm_moving_play(env_cfg, *, force: bool = False) -> None:
    if not (force or args_cli.b2_z1_legacy_ik_arm_moving):
        return
    if not _is_b2_z1_arm_moving_task():
        print("[WARN] --b2-z1-legacy-ik-arm-moving is only used for B2-Z1 ArmMoving tasks. Ignoring.")
        return
    if getattr(env_cfg, "_b2_z1_legacy_ik_arm_moving_configured", False):
        return

    from Gurukul.tasks.manager_based.locomotion.velocity.config.quadruped_with_arm.unitree_b2_z1_arm.arm_motion_env_cfg import (
        configure_z1_arm_task_space_motion,
    )
    import Gurukul.tasks.manager_based.locomotion.velocity.mdp as mdp

    configure_z1_arm_task_space_motion(env_cfg)
    env_cfg.observations.policy.velocity_commands = ObsTerm(
        func=mdp.velocity_posture_commands,
        params={"command_name": "base_velocity"},
        clip=(-100.0, 100.0),
        scale=1.0,
    )
    articulation_props = getattr(getattr(env_cfg.scene, "robot", None).spawn, "articulation_props", None)
    if articulation_props is not None:
        articulation_props.enabled_self_collisions = False
    env_cfg._b2_z1_legacy_ik_arm_moving_configured = True
    print("[B2-Z1] Restored legacy IK ArmMoving play config for old checkpoints.")


def _build_keyboard_velocity_observation(controller: Se2Keyboard, command_name: str = "base_velocity"):
    command_cache: dict[str, object] = {"step": -1, "command": None}

    def _keyboard_velocity_commands(env):
        step_counter = getattr(env, "common_step_counter", 0)
        if isinstance(step_counter, torch.Tensor):
            step_counter = int(step_counter.item())
        else:
            step_counter = int(step_counter)

        if command_cache["step"] != step_counter:
            command_cache["step"] = step_counter
            velocity = torch.as_tensor(controller.advance(), dtype=torch.float32, device=env.device).reshape(1, -1)
            command_cache["command"] = velocity[..., :3]

        command = command_cache["command"]
        if not isinstance(command, torch.Tensor):
            command = torch.zeros((1, 3), dtype=torch.float32, device=env.device)
            command_cache["command"] = command
        if getattr(env, "num_envs", 1) > 1:
            command = command.expand(env.num_envs, -1)

        command_manager = getattr(env, "command_manager", None)
        if command_manager is None:
            return command
        try:
            command_term = command_manager.get_term(command_name)
        except Exception:
            return command

        posture = getattr(command_term, "posture_command", None)
        if posture is None:
            return command
        return torch.cat((command, posture), dim=-1)

    return _keyboard_velocity_commands


class Go2D1LiveControl:
    neutral_ee_pos = GO2_D1_WBC_NEUTRAL_EE_POS
    ee_range = GO2_D1_WBC_EE_POS_RANGE
    keepout = GO2_D1_WBC_BODY_EXCLUSION_BOX
    body_clearance = GO2_D1_WBC_BODY_CLEARANCE
    workspace_origin = GO2_D1_WBC_WORKSPACE_ORIGIN
    reach_range = GO2_D1_WBC_REACH_RANGE

    def __init__(
        self,
        env_cfg,
        keyboard_controller: Se2Keyboard | None = None,
        arm_command_params: dict | None = None,
    ):
        arm_command_params = arm_command_params or {}
        self.ee_range = tuple(arm_command_params.get("ee_pos_range", self.ee_range))
        self.keepout = tuple(arm_command_params.get("body_exclusion_box", self.keepout))
        self.body_clearance = float(arm_command_params.get("body_clearance", self.body_clearance))
        self.workspace_origin = tuple(arm_command_params.get("workspace_origin", self.workspace_origin))
        self.reach_range = tuple(arm_command_params.get("reach_range", self.reach_range))
        self.neutral_ee_pos = tuple(arm_command_params.get("neutral_pos", self.neutral_ee_pos))
        ranges = env_cfg.commands.base_velocity.ranges
        self.vel = torch.zeros(3, dtype=torch.float32)
        self.ee_pos = torch.zeros(3, dtype=torch.float32)
        self.gripper = 0.0
        self.vel_limits = (tuple(ranges.lin_vel_x), tuple(ranges.lin_vel_y), tuple(ranges.ang_vel_z))
        self.ee_offset_limits = tuple(
            (low - self.neutral_ee_pos[idx], high - self.neutral_ee_pos[idx])
            for idx, (low, high) in enumerate(self.ee_range)
        )
        self.ee_anchor_pos: torch.Tensor | None = None
        self.ee_command_pos: torch.Tensor | None = None
        self.ee_command_active = False
        self.deployment_waypoints = tuple(arm_command_params.get("deployment_ee_waypoints", ()))
        self.deployment_phase = 0
        self.keyboard_controller = keyboard_controller

    @staticmethod
    def _clip(value: float, limits: tuple[float, float]) -> float:
        return min(max(float(value), float(limits[0])), float(limits[1]))

    def set_vel(self, vx: float, vy: float, wz: float) -> None:
        self.vel[:] = torch.tensor(
            [
                self._clip(vx, self.vel_limits[0]),
                self._clip(vy, self.vel_limits[1]),
                self._clip(wz, self.vel_limits[2]),
            ],
            dtype=torch.float32,
        )

    def nudge_ee(self, axis: int, delta: float) -> None:
        self.ee_pos[axis] = self._clip(float(self.ee_pos[axis]) + delta, self.ee_offset_limits[axis])
        self.ee_command_active = True

    def reset_ee(self, reset_deployment: bool = False) -> None:
        self.ee_pos[:] = 0.0
        self.ee_command_active = False
        self.ee_anchor_pos = None
        self.ee_command_pos = None
        if reset_deployment:
            self.deployment_phase = 0


def _start_go2_d1_xbox():
    try:
        import pygame
    except ModuleNotFoundError:
        print("[WARN] pygame is not installed; Go2+D1 Xbox control disabled.")
        return None

    pygame.init()
    pygame.joystick.init()
    if pygame.joystick.get_count() <= args_cli.go2_d1_xbox_device_id:
        print("[WARN] No Xbox controller detected for Go2+D1 live control.")
        return None
    joystick = pygame.joystick.Joystick(args_cli.go2_d1_xbox_device_id)
    joystick.init()
    print(f"[INFO] Go2+D1 Xbox control enabled: {joystick.get_name()}.")
    print("[INFO] Xbox EE: LB/RB=x, D-pad=y/z, right-stick Y=z, X=reset EE, B/Y=close/open gripper.")
    return pygame, joystick


def _go2_d1_axis(joystick, axis_id: int) -> float:
    if joystick.get_numaxes() <= axis_id:
        return 0.0
    value = float(joystick.get_axis(axis_id))
    return 0.0 if abs(value) < float(args_cli.go2_d1_xbox_deadzone) else value


def _go2_d1_button(joystick, button_id: int) -> bool:
    return joystick.get_numbuttons() > button_id and bool(joystick.get_button(button_id))


def _go2_d1_hat(joystick) -> tuple[int, int]:
    if joystick.get_numhats() > 0:
        return joystick.get_hat(0)
    if joystick.get_numaxes() <= 7:
        return 0, 0
    hat_x = int(round(_go2_d1_axis(joystick, 6)))
    hat_y = -int(round(_go2_d1_axis(joystick, 7)))
    return hat_x, hat_y


def _poll_go2_d1_live_control(state: Go2D1LiveControl, xbox, dt: float) -> None:
    if state.keyboard_controller is not None:
        state.set_vel(*torch.as_tensor(state.keyboard_controller.advance(), dtype=torch.float32)[:3].tolist())
    if xbox is None:
        return
    pygame, joystick = xbox
    pygame.event.pump()
    state.set_vel(
        -_go2_d1_axis(joystick, 1) * state.vel_limits[0][1],
        -_go2_d1_axis(joystick, 0) * state.vel_limits[1][1],
        -_go2_d1_axis(joystick, 3) * state.vel_limits[2][1],
    )
    ee_delta = float(args_cli.go2_d1_ee_speed) * float(dt)
    if _go2_d1_button(joystick, 5):
        state.nudge_ee(0, ee_delta)
    if _go2_d1_button(joystick, 4):
        state.nudge_ee(0, -ee_delta)
    hat_x, hat_y = _go2_d1_hat(joystick)
    if hat_x:
        state.nudge_ee(1, -hat_x * ee_delta)
    if hat_y:
        state.nudge_ee(2, hat_y * ee_delta)
    right_y = _go2_d1_axis(joystick, 4)
    if right_y:
        state.nudge_ee(2, -right_y * ee_delta)
    if _go2_d1_button(joystick, 0):
        state.set_vel(0.0, 0.0, 0.0)
    if _go2_d1_button(joystick, 2):
        state.reset_ee()
    if _go2_d1_button(joystick, 1) or _go2_d1_button(joystick, 3):
        state.gripper = 0.025 if _go2_d1_button(joystick, 1) else 0.0


def _bind_go2_d1_keyboard(state: Go2D1LiveControl) -> None:
    controller = state.keyboard_controller
    if controller is None:
        return
    step = float(args_cli.go2_d1_ee_step)
    controller.add_callback("I", lambda: state.nudge_ee(0, step))
    controller.add_callback("K", lambda: state.nudge_ee(0, -step))
    controller.add_callback("J", lambda: state.nudge_ee(1, step))
    controller.add_callback("L", lambda: state.nudge_ee(1, -step))
    controller.add_callback("U", lambda: state.nudge_ee(2, step))
    controller.add_callback("O", lambda: state.nudge_ee(2, -step))
    controller.add_callback("R", state.reset_ee)
    controller.add_callback("G", lambda: setattr(state, "gripper", 0.0 if state.gripper > 0.0 else 0.025))
    print("[INFO] Go2+D1 keyboard EE jog enabled: I/K x, J/L y, U/O z, R reset, G gripper.")


def _build_go2_d1_velocity_observation(state: Go2D1LiveControl, command_name: str = "base_velocity"):
    def _live_velocity_commands(env):
        command = state.vel.to(env.device).reshape(1, 3).repeat(env.num_envs, 1)
        try:
            command_term = env.command_manager.get_term(command_name)
        except Exception:
            return command
        if hasattr(command_term, "vel_command_b"):
            command_term.vel_command_b[:, :3] = command
        if hasattr(command_term, "heading_target"):
            command_term.heading_target[:] = 0.0
        posture = getattr(command_term, "posture_command", None)
        return command if posture is None else torch.cat((command, posture), dim=-1)

    return _live_velocity_commands


def _visualize_go2_d1_ee_command(env) -> None:
    """Draw current vs commanded D1 end-effector frames during live play."""
    if not hasattr(env, "_arm_ee_goal_pos"):
        return

    from isaaclab.markers import VisualizationMarkers
    from isaaclab.markers.config import FRAME_MARKER_CFG
    from isaaclab.utils.math import combine_frame_transforms

    robot = env.scene["robot"]
    ee_body_id = int(robot.find_bodies("Link6")[0][0])
    device = env.device
    num_envs = env.num_envs

    if not hasattr(env, "_arm_ee_target_quat"):
        env._arm_ee_target_quat = torch.zeros((num_envs, 4), device=device, dtype=torch.float32)
        env._arm_ee_target_quat[:, 0] = 1.0

    if not hasattr(env, "_arm_ee_visualizer_current"):
        cfg_cur = FRAME_MARKER_CFG.replace()
        cfg_cur.markers["frame"].scale = (0.10, 0.10, 0.10)
        env._arm_ee_visualizer_current = VisualizationMarkers(cfg_cur.replace(prim_path="/Visuals/arm_ee_current"))
        cfg_des = FRAME_MARKER_CFG.replace()
        cfg_des.markers["frame"].scale = (0.14, 0.14, 0.14)
        env._arm_ee_visualizer_desired = VisualizationMarkers(cfg_des.replace(prim_path="/Visuals/arm_ee_desired"))
        print("[INFO] Go2+D1 EE markers: small frame = current Link6, large frame = commanded target.")

    root_pose_w = robot.data.root_pose_w
    ee_state_w = robot.data.body_state_w[:, ee_body_id, 0:7]
    des_pos_w, des_quat_w = combine_frame_transforms(
        root_pose_w[:, 0:3],
        root_pose_w[:, 3:7],
        env._arm_ee_goal_pos,
        env._arm_ee_target_quat,
    )
    env._arm_ee_visualizer_current.visualize(
        translations=ee_state_w[:, 0:3],
        orientations=ee_state_w[:, 3:7],
    )
    env._arm_ee_visualizer_desired.visualize(
        translations=des_pos_w,
        orientations=des_quat_w,
    )


def _apply_go2_d1_live_command(env, state: Go2D1LiveControl) -> None:
    from Gurukul.tasks.manager_based.locomotion.velocity.mdp.events import _clamp_arm_ee_target

    if state.ee_anchor_pos is None:
        robot = env.scene["robot"]
        body_ids = robot.find_bodies("Link6")[0]
        ee_pos_w = robot.data.body_pos_w[:, body_ids[0], :]
        state.ee_anchor_pos = quat_apply_inverse(robot.data.root_quat_w, ee_pos_w - robot.data.root_pos_w)[0].detach()
        state.ee_command_pos = state.ee_anchor_pos.clone()
        state.ee_offset_limits = tuple(
            (low - float(state.ee_anchor_pos[idx]), high - float(state.ee_anchor_pos[idx]))
            for idx, (low, high) in enumerate(state.ee_range)
        )
        print(f"[INFO] Go2+D1 EE zero anchored at current pose: {state.ee_anchor_pos.cpu().tolist()}")

    if state.ee_command_pos is None:
        state.ee_command_pos = state.ee_anchor_pos.clone()
    if not state.ee_command_active:
        target = state.ee_command_pos.to(env.device)
    else:
        deploying = state.deployment_phase < len(state.deployment_waypoints)
        if deploying:
            desired = torch.tensor(
                state.deployment_waypoints[state.deployment_phase],
                device=env.device,
                dtype=torch.float32,
            )
        else:
            desired = state.ee_anchor_pos.to(env.device) + state.ee_pos.to(env.device)
            desired = _clamp_arm_ee_target(
                desired.clone(),
                state.ee_range,
                state.keepout,
                state.body_clearance,
                workspace_origin=state.workspace_origin,
                reach_range=state.reach_range,
            )
        current_command = state.ee_command_pos.to(env.device)
        delta = desired - current_command
        distance = torch.linalg.vector_norm(delta)
        max_change = max(float(args_cli.go2_d1_ee_speed) * float(env.step_dt), 1.0e-6)
        if distance > max_change:
            target = current_command + delta / distance * max_change
        else:
            target = desired
            if deploying:
                state.deployment_phase += 1
        state.ee_command_pos = target.detach().clone()
    num_envs = env.num_envs
    device = env.device
    if not hasattr(env, "_arm_ee_goal_pos"):
        env._arm_ee_target_pos = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
        env._arm_ee_target_quat = torch.zeros((num_envs, 4), device=device, dtype=torch.float32)
        env._arm_ee_target_quat[:, 0] = 1.0
        env._arm_ee_target_initialized = torch.zeros(num_envs, dtype=torch.bool, device=device)
        env._arm_ee_start_pos = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
        env._arm_ee_goal_pos = torch.zeros((num_envs, 3), device=device, dtype=torch.float32)
        env._arm_trajectory_progress = torch.ones(num_envs, device=device, dtype=torch.float32)
        env._arm_trajectory_duration = torch.ones(num_envs, device=device, dtype=torch.float32)
    env._arm_ee_target_initialized[:] = True
    env._arm_ee_target_pos[:] = target
    env._arm_ee_start_pos[:] = target
    env._arm_ee_goal_pos[:] = target
    env._arm_trajectory_progress[:] = 1.0
    env._arm_trajectory_duration[:] = 1.0
    if not hasattr(env, "_gripper_target_pos"):
        env._gripper_target_pos = torch.zeros((num_envs, 2), device=device, dtype=torch.float32)
    env._gripper_target_pos[:, 0] = float(state.gripper)
    if env._gripper_target_pos.shape[1] > 1:
        env._gripper_target_pos[:, 1] = float(state.gripper)
    _visualize_go2_d1_ee_command(env)


def _resolve_go2_d1_spawn_joint_pos(env_cfg) -> dict[str, float]:
    """Match play spawn/reset arm joints to the task config or Go2+D1 asset defaults."""
    joint_names = list(env_cfg.arm_joint_names) + list(getattr(env_cfg, "gripper_joint_names", ()))
    init_joint_pos = dict(env_cfg.scene.robot.init_state.joint_pos)
    resolved: dict[str, float] = {}
    for joint_name in joint_names:
        if joint_name in init_joint_pos:
            resolved[joint_name] = float(init_joint_pos[joint_name])
        elif joint_name in GO2_D1_ARM_JOINT_DEFAULTS:
            resolved[joint_name] = float(GO2_D1_ARM_JOINT_DEFAULTS[joint_name])
        else:
            resolved[joint_name] = 0.0
    return resolved


def _configure_go2_d1_live_control(env_cfg, keyboard_controller: Se2Keyboard | None) -> Go2D1LiveControl:
    arm_command_params = {}
    reset_arm_command = getattr(env_cfg.events, "reset_arm_command", None)
    if reset_arm_command is not None:
        arm_command_params = dict(reset_arm_command.params)
    state = Go2D1LiveControl(env_cfg, keyboard_controller, arm_command_params)
    arm_joint_pos = _resolve_go2_d1_spawn_joint_pos(env_cfg)
    env_cfg.scene.robot.init_state.joint_pos.update(arm_joint_pos)
    reset_arm_joint_state = getattr(env_cfg.events, "reset_arm_joint_state", None)
    if reset_arm_joint_state is not None:
        reset_arm_joint_state.params["position_ranges"].update(
            {joint_name: (value, value) for joint_name, value in arm_joint_pos.items()}
        )
    reset_joints_event = getattr(env_cfg.events, "randomize_reset_joints", None)
    if reset_joints_event is not None:
        reset_joints_event.params = dict(reset_joints_event.params)
        reset_joints_event.params["asset_cfg"] = SceneEntityCfg("robot", joint_names=env_cfg.leg_joint_names)
    env_cfg.scene.num_envs = 1
    if hasattr(env_cfg.terminations, "time_out"):
        env_cfg.terminations.time_out = None
    _set_attr_if_exists(env_cfg.events, "randomize_arm_command", None)
    _set_attr_if_exists(env_cfg.events, "reset_arm_command", None)
    _set_attr_if_exists(env_cfg.events, "advance_arm_command", None)
    _set_attr_if_exists(env_cfg.events, "randomize_arm_async_motion", None)
    _set_attr_if_exists(env_cfg.events, "reset_arm_async_motion", None)
    _set_attr_if_exists(env_cfg.events, "advance_arm_async_motion", None)
    _set_attr_if_exists(env_cfg.curriculum, "loco_manipulation_training_stages", None)
    arm_cfg = SceneEntityCfg("robot", joint_names=env_cfg.arm_joint_names, body_names=["Link6"], preserve_order=True)
    env_cfg.events.hold_arm_ee_command = EventTermCfg(
        func=mdp.hold_arm_ee_command_at_current_pose,
        mode="reset",
        params={
            "asset_cfg": arm_cfg,
            "gripper_joint_names": tuple(env_cfg.gripper_joint_names),
            "gripper_open_pos": (0.0, 0.0),
        },
    )
    if hasattr(env_cfg.commands, "base_velocity"):
        env_cfg.commands.base_velocity.heading_command = False
        env_cfg.commands.base_velocity.rel_standing_envs = 0.0
        env_cfg.commands.base_velocity.resampling_time_range = (1.0e6, 1.0e6)
        env_cfg.commands.base_velocity.debug_vis = True
    env_cfg.observations.policy.velocity_commands = ObsTerm(func=_build_go2_d1_velocity_observation(state))
    _bind_go2_d1_keyboard(state)
    print(
        "[INFO] Go2+D1 live control enabled: "
        f"ee_range={state.ee_range}, reach={state.reach_range}, keepout={state.keepout}, "
        f"clearance={state.body_clearance:.2f} m"
    )
    print(
        "[INFO] Go2+D1 spawn arm joints (rad, asset default): "
        f"{arm_joint_pos}"
    )
    print(
        "[INFO] Base velocity: keyboard arrows (if --keyboard) or Xbox left stick + right-stick X for yaw. "
        "EE jog: I/K x, J/L y, U/O z, R reset EE offset, G toggle gripper. "
        "Xbox: LB/RB=x, D-pad=y/z, right-stick Y=z, X=reset EE, A=stop base, B/Y=gripper."
    )
    return state


def _mounted_arm_task_label() -> str:
    task_name = (args_cli.task or "").lower()
    if "b2" in task_name and "z1" in task_name:
        return "B2-Z1"
    return "Go2-D1" if "d1" in task_name else "Go2-Airbot"


def _play_arm_difficulty() -> float:
    return max(0.0, min(1.0, float(args_cli.loco_manip_arm_difficulty)))


def _set_base_velocity_cfg(command_cfg, stage: str, arm_difficulty: float = 1.0) -> None:
    if command_cfg is None or not hasattr(command_cfg, "ranges"):
        return

    ranges = command_cfg.ranges
    if not hasattr(command_cfg, "_loco_manip_play_full_ranges"):
        command_cfg._loco_manip_play_full_ranges = {
            "lin_vel_x": tuple(ranges.lin_vel_x),
            "lin_vel_y": tuple(ranges.lin_vel_y),
            "ang_vel_z": tuple(ranges.ang_vel_z),
            "heading": tuple(ranges.heading) if hasattr(ranges, "heading") else None,
            "roll": tuple(command_cfg.roll_range) if hasattr(command_cfg, "roll_range") else None,
            "pitch": tuple(command_cfg.pitch_range) if hasattr(command_cfg, "pitch_range") else None,
            "height": tuple(command_cfg.height_range) if hasattr(command_cfg, "height_range") else None,
            "nominal_height": float(getattr(command_cfg, "nominal_height", 0.0)),
        }
    full_ranges = command_cfg._loco_manip_play_full_ranges

    if hasattr(command_cfg, "roll_range") and hasattr(command_cfg, "pitch_range"):
        if stage == "fixed":
            command_cfg.roll_range = (0.0, 0.0)
            command_cfg.pitch_range = (0.0, 0.0)
        else:
            full_roll = full_ranges["roll"] or (0.0, 0.0)
            full_pitch = full_ranges["pitch"] or (0.0, 0.0)
            command_cfg.roll_range = tuple(float(v) * arm_difficulty for v in full_roll)
            command_cfg.pitch_range = tuple(float(v) * arm_difficulty for v in full_pitch)
    if hasattr(command_cfg, "height_range"):
        nominal_height = float(full_ranges.get("nominal_height", getattr(command_cfg, "nominal_height", 0.0)))
        full_height = full_ranges.get("height") or (nominal_height, nominal_height)
        if stage == "fixed":
            command_cfg.height_range = (nominal_height, nominal_height)
        else:
            command_cfg.height_range = tuple(
                nominal_height + (float(value) - nominal_height) * arm_difficulty for value in full_height
            )

    if stage == "arm":
        ranges.lin_vel_x = (0.0, 0.0)
        ranges.lin_vel_y = (0.0, 0.0)
        ranges.ang_vel_z = (0.0, 0.0)
        if hasattr(ranges, "heading"):
            ranges.heading = (0.0, 0.0)
        if hasattr(command_cfg, "heading_command"):
            command_cfg.heading_command = False
        return

    ranges.lin_vel_x = full_ranges["lin_vel_x"]
    ranges.lin_vel_y = full_ranges["lin_vel_y"]
    ranges.ang_vel_z = full_ranges["ang_vel_z"]
    if hasattr(ranges, "heading") and full_ranges["heading"] is not None:
        ranges.heading = full_ranges["heading"]


def _configure_loco_manip_play_cfg(env_cfg, stage: str) -> bool:
    """Prepare mounted-arm events/curriculum for deterministic play-stage inspection."""
    if not _is_loco_manip_play_task(env_cfg) or stage == "off":
        return False

    curriculum_term = getattr(env_cfg.curriculum, "loco_manipulation_training_stages", None)
    base_velocity_cfg = getattr(getattr(env_cfg, "commands", None), "base_velocity", None)
    if curriculum_term is not None and base_velocity_cfg is not None:
        params = curriculum_term.params
        ranges = base_velocity_cfg.ranges
        base_velocity_cfg._loco_manip_play_full_ranges = {
            "lin_vel_x": tuple(params.get("walking_lin_vel_x", ranges.lin_vel_x)),
            "lin_vel_y": tuple(params.get("walking_lin_vel_y", ranges.lin_vel_y)),
            "ang_vel_z": tuple(params.get("walking_ang_vel_z", ranges.ang_vel_z)),
            "heading": tuple(params.get("walking_heading", ranges.heading)) if hasattr(ranges, "heading") else None,
            "roll": tuple(params.get("walking_roll", getattr(base_velocity_cfg, "roll_range", (0.0, 0.0)))),
            "pitch": tuple(params.get("walking_pitch", getattr(base_velocity_cfg, "pitch_range", (0.0, 0.0)))),
            "height": tuple(params.get("walking_height", getattr(base_velocity_cfg, "height_range", (0.0, 0.0)))),
            "nominal_height": float(params.get("nominal_height", getattr(base_velocity_cfg, "nominal_height", 0.0))),
        }

    if stage != "curriculum":
        _set_attr_if_exists(env_cfg.curriculum, "loco_manipulation_training_stages", None)

    stage_for_cfg = "fixed" if stage in ("cycle", "curriculum") else "combined" if stage == "grid" else stage
    play_difficulty = 1.0 if stage == "grid" else _play_arm_difficulty()
    _set_base_velocity_cfg(base_velocity_cfg, stage_for_cfg, play_difficulty)
    if stage == "grid" and hasattr(env_cfg, "viewer"):
        grid_width = max(1.0, float(env_cfg.scene.env_spacing) * max(1, int(env_cfg.scene.num_envs) ** 0.5))
        camera_distance = max(4.0, 1.35 * grid_width)
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.asset_name = None
        env_cfg.viewer.body_name = None
        env_cfg.viewer.eye = (camera_distance, camera_distance, 0.65 * camera_distance)
        env_cfg.viewer.lookat = (0.0, 0.0, 0.4)
    return True


def _speed_up_loco_manip_curriculum_for_play(env_cfg) -> None:
    if args_cli.loco_manip_stage != "curriculum":
        return
    curriculum = getattr(getattr(env_cfg, "curriculum", None), "loco_manipulation_training_stages", None)
    if curriculum is None:
        return
    speedup = max(1.0e-6, float(args_cli.loco_manip_curriculum_speedup))
    if speedup == 1.0:
        return

    params = curriculum.params

    def _scale_steps(name: str) -> None:
        steps = params.get(name)
        if steps is None:
            return
        params[name] = tuple(max(1, int(round(float(step) / speedup))) for step in steps)

    _scale_steps("stage_steps")
    _scale_steps("arm_difficulty_steps")
    params["stage_iteration_bins"] = None
    params["arm_difficulty_iteration_bins"] = None
    print(
        f"[{_mounted_arm_task_label()}] Curriculum play speedup={speedup:g}: "
        f"stage_steps={params.get('stage_steps')}, arm_difficulty_steps={params.get('arm_difficulty_steps')}."
    )


def _get_loco_manip_stage_name(stage: str, timestep: int) -> str:
    if stage != "cycle":
        return stage

    cycle_steps = max(1, int(args_cli.loco_manip_stage_cycle_steps))
    return ("fixed", "arm", "combined")[(timestep // cycle_steps) % 3]


def _velocity_demo_command(ranges, timestep: int, total_steps: int) -> tuple[str, tuple[float, float, float]]:
    """Return the video-showcase phase and SE(2) command for one play step."""
    total_steps = max(1, int(total_steps))
    progress = (int(timestep) % total_steps) / total_steps
    x_min, x_max = (float(value) for value in ranges.lin_vel_x)
    yaw_min, yaw_max = (float(value) for value in ranges.ang_vel_z)
    forward_arc = 0.7 * max(0.0, x_max)

    if progress < 0.05:
        return "ready", (0.0, 0.0, 0.0)
    if progress < 0.33:
        return "fast_forward", (x_max, 0.0, 0.0)
    if progress < 0.455:
        return "forward_left_arc", (forward_arc, 0.0, yaw_max)
    if progress < 0.58:
        return "forward_right_arc", (forward_arc, 0.0, yaw_min)
    if progress < 0.655:
        return "rotate_left", (0.0, 0.0, yaw_max)
    if progress < 0.73:
        return "rotate_right", (0.0, 0.0, yaw_min)
    if progress < 0.95:
        return "fast_backward", (x_min, 0.0, 0.0)
    return "stop", (0.0, 0.0, 0.0)


def _configure_velocity_demo(env_cfg) -> bool:
    """Disable autonomous command resampling when scripted playback is requested."""
    if not args_cli.velocity_demo:
        return False
    command_cfg = getattr(getattr(env_cfg, "commands", None), "base_velocity", None)
    if command_cfg is None:
        print("[WARN] --velocity-demo requested, but this task has no `commands.base_velocity` term.")
        return False
    command_cfg.heading_command = False
    command_cfg.rel_standing_envs = 0.0
    command_cfg.rel_heading_envs = 0.0
    command_cfg.resampling_time_range = (1.0e6, 1.0e6)
    print("[INFO] Scripted velocity demo enabled; commands will span the configured training limits.")
    return True


def _apply_velocity_demo(env, timestep: int, total_steps: int, state: dict[str, str]) -> None:
    """Write the active showcase command directly to the base-velocity term."""
    command_manager = getattr(env.unwrapped, "command_manager", None)
    if command_manager is None:
        return
    try:
        command_term = command_manager.get_term("base_velocity")
    except Exception:
        return
    if not hasattr(command_term, "vel_command_b"):
        return

    label, command = _velocity_demo_command(command_term.cfg.ranges, timestep, total_steps)
    command_term.vel_command_b[:] = torch.tensor(
        command,
        device=command_term.vel_command_b.device,
        dtype=command_term.vel_command_b.dtype,
    )
    if hasattr(command_term, "heading_target"):
        command_term.heading_target[:] = 0.0
    if state.get("phase") != label:
        print(f"[Velocity demo] {label}: vx={command[0]:.2f}, vy={command[1]:.2f}, yaw={command[2]:.2f}")
        state["phase"] = label


def _loco_manip_play_grid_slices(num_envs: int, device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return easy-to-hard stage and difficulty assignments for a play grid."""
    if num_envs <= 0:
        return (
            torch.empty(0, device=device, dtype=torch.long),
            torch.empty(0, device=device, dtype=torch.float32),
        )
    progress = torch.linspace(0.0, 1.0, num_envs, device=device, dtype=torch.float32)
    stages = torch.round(2.0 * progress).to(dtype=torch.long)
    difficulties = progress.clone()
    difficulties[stages == 0] = 0.0
    return stages, difficulties


def _loco_manip_extreme_probe(ranges, phase_index: int) -> tuple[str, tuple[float, float, float]]:
    """Return an axis-isolated or combined command at the configured range extrema."""
    x_min, x_max = (float(value) for value in ranges.lin_vel_x)
    y_min, y_max = (float(value) for value in ranges.lin_vel_y)
    yaw_min, yaw_max = (float(value) for value in ranges.ang_vel_z)
    probes = (
        ("forward", (x_max, 0.0, 0.0)),
        ("backward", (x_min, 0.0, 0.0)),
        ("left", (0.0, y_max, 0.0)),
        ("right", (0.0, y_min, 0.0)),
        ("left_yaw", (0.0, 0.0, yaw_max)),
        ("right_yaw", (0.0, 0.0, yaw_min)),
        ("positive_corner", (x_max, y_max, yaw_max)),
        ("negative_corner", (x_min, y_min, yaw_min)),
    )
    return probes[int(phase_index) % len(probes)]


def _configure_loco_manip_play_grid(env, state: dict[str, object], force: bool = False) -> None:
    if args_cli.loco_manip_stage != "grid":
        return

    base_env = env.unwrapped
    num_envs = int(base_env.num_envs)
    if num_envs <= 0:
        return
    if state.get("configured") and not force:
        return

    device = base_env.device
    stages, difficulties = _loco_manip_play_grid_slices(num_envs, device)
    enabled = stages >= 1

    base_env._loco_manip_training_stage = stages
    base_env._loco_manip_arm_motion_difficulty = difficulties
    base_env._loco_manip_arm_motion_enabled = enabled
    state["configured"] = True

    command_manager = getattr(base_env, "command_manager", None)
    if command_manager is not None:
        try:
            command_term = command_manager.get_term("base_velocity")
        except Exception:
            command_term = None
        if command_term is not None:
            _set_base_velocity_cfg(command_term.cfg, "combined", 1.0)
            if hasattr(command_term.cfg, "heading_command"):
                command_term.cfg.heading_command = False
            if hasattr(command_term, "_resample_command"):
                command_term._resample_command(torch.arange(num_envs, device=device))
            if hasattr(command_term, "posture_command"):
                state["raw_posture_command"] = command_term.posture_command.clone()
            arm_only_envs = stages == 1
            if hasattr(command_term, "vel_command_b"):
                command_term.vel_command_b[arm_only_envs, :] = 0.0
            if hasattr(command_term, "heading_target"):
                command_term.heading_target[arm_only_envs] = 0.0

    counts = {stage_id: int((stages == stage_id).sum().item()) for stage_id in (0, 1, 2)}
    assignments = ", ".join(
        f"env{env_id}=stage{int(stages[env_id])}/difficulty{float(difficulties[env_id]):.2f}"
        for env_id in range(num_envs)
    )
    print(
        f"[{_mounted_arm_task_label()}] Play grid across {num_envs} envs: "
        f"stage0={counts[0]}, stage1={counts[1]}, stage2={counts[2]}, "
        f"{assignments}."
    )
    print(
        f"[{_mounted_arm_task_label()}] Walking grid robots cycle configured extrema for forward/backward, "
        f"left/right, yaw, and both combined corners; the hardest robot alternates low/high pitch and height every "
        f"{max(1, int(args_cli.loco_manip_grid_probe_steps))} steps."
    )


def _apply_loco_manip_play_grid_commands(env, state: dict[str, object], timestep: int = 0) -> None:
    if args_cli.loco_manip_stage != "grid":
        return
    base_env = env.unwrapped
    stages = getattr(base_env, "_loco_manip_training_stage", None)
    if not isinstance(stages, torch.Tensor):
        return

    command_manager = getattr(base_env, "command_manager", None)
    if command_manager is None:
        return
    try:
        command_term = command_manager.get_term("base_velocity")
    except Exception:
        return

    arm_only_envs = stages.to(device=base_env.device) == 1
    walking_envs = ~arm_only_envs
    difficulties = getattr(base_env, "_loco_manip_arm_motion_difficulty", None)
    probe_env = None
    probe_phase = 0
    if isinstance(difficulties, torch.Tensor) and difficulties.numel() > 0:
        probe_env = int(torch.argmax(difficulties).item())
        probe_phase = (int(timestep) // max(1, int(args_cli.loco_manip_grid_probe_steps))) % 8
    probe_label = "unknown"
    if hasattr(command_term, "vel_command_b"):
        command_term.vel_command_b[arm_only_envs, :] = 0.0
        if torch.any(walking_envs):
            probe_label, probe_velocity = _loco_manip_extreme_probe(command_term.cfg.ranges, probe_phase)
            command_term.vel_command_b[walking_envs, :] = torch.tensor(
                probe_velocity,
                device=command_term.vel_command_b.device,
                dtype=command_term.vel_command_b.dtype,
            )
    if hasattr(command_term, "heading_target"):
        command_term.heading_target[arm_only_envs] = 0.0
        if probe_env is not None:
            command_term.heading_target[probe_env] = 0.0
    posture_command = getattr(command_term, "posture_command", None)
    if isinstance(posture_command, torch.Tensor) and isinstance(difficulties, torch.Tensor):
        last_applied = state.get("applied_posture_command")
        if not isinstance(last_applied, torch.Tensor) or not torch.allclose(posture_command, last_applied):
            state["raw_posture_command"] = posture_command.clone()
        raw_posture = state.get("raw_posture_command")
        if isinstance(raw_posture, torch.Tensor):
            scale = difficulties.to(device=posture_command.device, dtype=posture_command.dtype).unsqueeze(1)
            scaled_posture = raw_posture.clone()
            scaled_posture[:, 0] = 0.0
            scaled_posture[:, 1] *= scale[:, 0]
            if scaled_posture.shape[1] > 2:
                nominal_height = float(getattr(command_term.cfg, "nominal_height", 0.0))
                scaled_posture[:, 2] = nominal_height + (raw_posture[:, 2] - nominal_height) * scale[:, 0]
            if probe_env is not None and int(stages[probe_env].item()) == 2:
                full_ranges = getattr(command_term.cfg, "_loco_manip_play_full_ranges", {})
                pitch_range = full_ranges.get("pitch") or tuple(command_term.cfg.pitch_range)
                posture_phase = probe_phase % 2
                scaled_posture[probe_env, 1] = float(pitch_range[posture_phase])
                if scaled_posture.shape[1] > 2:
                    height_range = full_ranges.get("height") or tuple(command_term.cfg.height_range)
                    scaled_posture[probe_env, 2] = float(height_range[posture_phase])
            posture_command.copy_(scaled_posture)
            state["applied_posture_command"] = scaled_posture.clone()

    if probe_env is not None and state.get("grid_probe_phase") != probe_phase:
        velocity = command_term.vel_command_b[probe_env].detach().cpu().tolist()
        posture = command_term.posture_command[probe_env].detach().cpu().tolist()
        height_status = f", height={posture[2]:.3f}" if len(posture) > 2 else ""
        print(
            f"[{_mounted_arm_task_label()}][grid step={timestep}] env{probe_env} "
            f"probe={probe_label}, velocity={velocity}, "
            f"posture=[roll={posture[0]:.3f}, pitch={posture[1]:.3f}{height_status}]."
        )
        state["grid_probe_phase"] = probe_phase


def _apply_loco_manip_play_stage(env, stage: str, state: dict[str, str | None], force: bool = False) -> None:
    if stage in ("off", "curriculum", "grid"):
        return

    stage_changed = force or state.get("stage") != stage
    base_env = env.unwrapped
    stage_id = {"fixed": 0, "arm": 1, "combined": 2}[stage]
    arm_difficulty = 0.0 if stage_id == 0 else _play_arm_difficulty()
    base_env._loco_manip_arm_motion_enabled = stage_id >= 1
    base_env._loco_manip_arm_motion_difficulty = arm_difficulty
    base_env._loco_manip_training_stage = stage_id

    command_manager = getattr(base_env, "command_manager", None)
    command_term = None
    if command_manager is not None:
        try:
            command_term = command_manager.get_term("base_velocity")
        except Exception:
            command_term = None

    if command_term is not None:
        if not hasattr(base_env, "_loco_manip_play_original_heading_command"):
            base_env._loco_manip_play_original_heading_command = bool(
                getattr(command_term.cfg, "heading_command", False)
            )
        _set_base_velocity_cfg(command_term.cfg, stage, arm_difficulty)
        if hasattr(command_term.cfg, "heading_command") and stage != "arm":
            command_term.cfg.heading_command = base_env._loco_manip_play_original_heading_command
        if stage == "arm":
            command_term.vel_command_b[:, :] = 0.0
            if hasattr(command_term, "heading_target"):
                command_term.heading_target[:] = 0.0
        elif stage_changed:
            env_ids = torch.arange(base_env.num_envs, device=base_env.device)
            if hasattr(command_term, "_resample_command"):
                command_term._resample_command(env_ids)

    if stage_changed:
        descriptions = {
            "fixed": "stage 0: walking command, arm held at ready pose",
            "arm": f"stage 1: zero base command, arm primitives enabled at difficulty {arm_difficulty:.2f}",
            "combined": f"stage 2: walking command and arm primitives enabled at difficulty {arm_difficulty:.2f}",
        }
        print(f"[{_mounted_arm_task_label()}] Play {descriptions[stage]}.")
        state["stage"] = stage


def _print_loco_manip_curriculum_status(env, state: dict[str, object], timestep: int, force: bool = False) -> None:
    if args_cli.loco_manip_stage != "curriculum":
        return
    base_env = env.unwrapped
    stage_id = int(getattr(base_env, "_loco_manip_training_stage", 0))
    sampled_difficulty = getattr(base_env, "_loco_manip_arm_motion_difficulty", 0.0)
    frontier = getattr(base_env, "_loco_manip_arm_motion_frontier", sampled_difficulty)
    if isinstance(frontier, torch.Tensor):
        frontier_difficulty = float(torch.max(frontier).item())
    else:
        frontier_difficulty = float(frontier)
    if isinstance(sampled_difficulty, torch.Tensor):
        mean_difficulty = float(sampled_difficulty.float().mean().item())
        replay_fraction = float((sampled_difficulty < frontier_difficulty - 1.0e-6).float().mean().item())
    else:
        mean_difficulty = float(sampled_difficulty)
        replay_fraction = 0.0
    interval = max(1, int(args_cli.loco_manip_curriculum_status_interval))
    previous = state.get("stage_id")
    changed = previous is None or int(previous) != stage_id
    if not (force or changed or timestep % interval == 0):
        return
    descriptions = {
        0: "walking command, arm held at ready pose",
        1: "zero base command, arm primitives enabled",
        2: "walking command and arm primitives enabled",
    }
    print(
        f"[{_mounted_arm_task_label()}][step={timestep}] "
        f"curriculum stage {stage_id}: {descriptions.get(stage_id, 'unknown')} "
        f"| arm_difficulty_frontier={frontier_difficulty:.2f} "
        f"| batch_mean={mean_difficulty:.2f} | easier_replay={replay_fraction:.1%}"
    )
    state["stage_id"] = stage_id
    state["difficulty"] = frontier_difficulty


def _format_reward_terms(env, env_idx: int = 0) -> str:
    reward_manager = getattr(env.unwrapped, "reward_manager", None)
    if reward_manager is None:
        return "reward_manager=unavailable"
    terms = reward_manager.get_active_iterable_terms(env_idx)
    return ", ".join(f"{name}={values[0]:.4f}" for name, values in terms)


def _actor_uses_height_scan(env_cfg, agent_cfg) -> bool:
    observations_cfg = getattr(env_cfg, "observations", None)
    if observations_cfg is None:
        return False

    actor_groups = ["policy"]
    obs_groups_cfg = getattr(agent_cfg, "obs_groups", None)
    if isinstance(obs_groups_cfg, dict):
        actor_groups = obs_groups_cfg.get("actor", obs_groups_cfg.get("policy", ["policy"]))
    if isinstance(actor_groups, str):
        actor_groups = [actor_groups]

    for group_name in actor_groups:
        group_cfg = getattr(observations_cfg, group_name, None)
        if group_cfg is None:
            continue
        if getattr(group_cfg, "height_scan", None) is not None:
            return True
    return False


def _is_ame_policy(policy_nn) -> bool:
    return hasattr(policy_nn, "map_scan_dim") and hasattr(policy_nn, "last_attention_weights")


def _get_policy_attention(policy_nn):
    attention = getattr(policy_nn, "last_actor_attention_weights", None)
    if attention is None:
        attention = getattr(policy_nn, "last_attention_weights", None)
    if attention is not None and attention.ndim == 4 and attention.shape[2] == 1:
        attention = attention.squeeze(2)
    return attention


def _get_policy_attention_scan_shape(policy_nn) -> tuple[int, int] | None:
    scan_shape = getattr(policy_nn, "terrain_scan_shape", None)
    if scan_shape is not None:
        return tuple(int(value) for value in scan_shape)
    map_scan_dim = getattr(policy_nn, "map_scan_dim", None)
    if map_scan_dim is None or len(map_scan_dim) < 2:
        return None
    map_l, map_w = int(map_scan_dim[0]), int(map_scan_dim[1])
    if getattr(policy_nn, "cnn_downsample", False):
        return (map_w // 2 + 1, map_l // 2 + 1)
    return (map_w, map_l)


def _get_ame_map_points(policy_nn, points: torch.Tensor) -> torch.Tensor | None:
    map_scan_dim = getattr(policy_nn, "map_scan_dim", None)
    if map_scan_dim is None or len(map_scan_dim) < 2:
        return None
    map_l, map_w = int(map_scan_dim[0]), int(map_scan_dim[1])
    if points.shape[0] != map_l * map_w:
        return None
    points_grid = points.reshape(map_w, map_l, 3)
    if getattr(policy_nn, "cnn_downsample", False):
        points_grid = points_grid[::2, ::2]
    return points_grid.reshape(-1, 3)


def _policy_supports_attention_overlay(policy_nn) -> bool:
    if _is_ame_policy(policy_nn):
        return True
    return all(
        hasattr(policy_nn, attr_name)
        for attr_name in ("terrain_scan_shape", "terrain_obs_group", "last_actor_attention_weights")
    )


def _style_heightmap_visualizer_for_attention(height_scanner_cfg) -> None:
    height_scanner_cfg.debug_vis = True
    visualizer_cfg = getattr(height_scanner_cfg, "visualizer_cfg", None)
    markers = getattr(visualizer_cfg, "markers", None) if visualizer_cfg is not None else None
    hit_cfg = markers.get("hit") if isinstance(markers, dict) else None
    if hit_cfg is None:
        return
    if hasattr(hit_cfg, "radius"):
        hit_cfg.radius = 0.012
    visual_material = getattr(hit_cfg, "visual_material", None)
    if visual_material is not None and hasattr(visual_material, "diffuse_color"):
        visual_material.diffuse_color = (0.62, 0.62, 0.62)


def _build_attention_heightmap_visualizer(policy_nn) -> VisualizationMarkers:
    actor = getattr(policy_nn, "actor", None)
    cross_attention = getattr(actor, "cross_attention", None)
    num_heads = int(getattr(policy_nn, "num_heads", getattr(cross_attention, "num_heads", 0)))
    if num_heads <= 0:
        attention = _get_policy_attention(policy_nn)
        if attention is not None and attention.ndim == 3:
            num_heads = int(attention.shape[1])
    if num_heads <= 0:
        raise ValueError("Policy does not expose a valid number of attention heads.")
    markers = {}
    for head_idx in range(num_heads):
        bgr = DEFAULT_HEAD_COLORS_BGR[head_idx % len(DEFAULT_HEAD_COLORS_BGR)]
        rgb = tuple(float(channel) / 255.0 for channel in reversed(bgr))
        markers[f"head_{head_idx}"] = sim_utils.SphereCfg(
            radius=0.022,
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=rgb),
        )
    marker_cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/Policy/HeightmapAttention",
        markers=markers,
    )
    visualizer = VisualizationMarkers(marker_cfg)
    visualizer.set_visibility(True)
    return visualizer


def _render_attention_heightmap_overlay(policy_nn, env, visualizer: VisualizationMarkers) -> bool:
    attention = _get_policy_attention(policy_nn)
    scan_shape = _get_policy_attention_scan_shape(policy_nn)
    if attention is None or attention.ndim != 3 or attention.shape[0] < 1:
        return False

    scene_sensors = getattr(env.unwrapped.scene, "sensors", None)
    if not isinstance(scene_sensors, dict) or "height_scanner" not in scene_sensors:
        return False
    height_sensor = scene_sensors["height_scanner"]
    ray_hits_w = getattr(getattr(height_sensor, "data", None), "ray_hits_w", None)
    if ray_hits_w is None or ray_hits_w.shape[0] < 1:
        return False

    world_points = ray_hits_w[0].detach()
    if world_points.ndim != 2 or world_points.shape[1] != 3:
        return False
    if _is_ame_policy(policy_nn):
        world_points = _get_ame_map_points(policy_nn, world_points)
        if world_points is None:
            return False
    if attention.shape[-1] != world_points.shape[0]:
        return False

    marker_indices, marker_scales, point_strength = compute_attention_marker_overlay(
        attention[0],
        scan_shape=scan_shape,
    )
    valid_mask = torch.isfinite(world_points).all(dim=1)
    if not torch.any(valid_mask):
        return False

    world_points = world_points[valid_mask].clone()
    marker_indices = marker_indices[valid_mask]
    marker_scales = marker_scales[valid_mask].clone()
    point_strength = point_strength[valid_mask]
    world_points[:, 2] += 0.015 + 0.015 * point_strength
    marker_scales *= (0.7 + 0.4 * point_strength.unsqueeze(-1))
    visualizer.visualize(
        translations=world_points,
        scales=marker_scales,
        marker_indices=marker_indices,
    )
    return True


def _normalize_image_u8(image: torch.Tensor) -> torch.Tensor:
    image = torch.nan_to_num(image.float(), nan=0.0, posinf=0.0, neginf=0.0)
    image_min = image.min()
    image_max = image.max()
    denom = torch.clamp(image_max - image_min, min=1.0e-6)
    normalized = (image - image_min) / denom
    return (normalized * 255.0).to(torch.uint8)


def _contact_trail_policy(policy_nn):
    if hasattr(policy_nn, "contact_trail_memory"):
        return policy_nn
    policy = getattr(policy_nn, "module", None)
    if policy is not None and hasattr(policy, "contact_trail_memory"):
        return policy
    return None


def _contact_trail_foot_positions(policy_nn, env_idx: int) -> np.ndarray | None:
    foot_pos_b = getattr(policy_nn, "_last_foot_pos_b", None)
    if foot_pos_b is None:
        return None
    num_feet = int(getattr(policy_nn, "num_feet", 4))
    foot_pos_b = foot_pos_b.detach()
    if foot_pos_b.ndim == 3:
        foot_pos_b = foot_pos_b[-1]
    if foot_pos_b.ndim == 2 and foot_pos_b.shape[-1] == num_feet * 3:
        foot_pos_b = foot_pos_b.reshape(foot_pos_b.shape[0], num_feet, 3)
    if foot_pos_b.ndim != 3 or env_idx >= foot_pos_b.shape[0]:
        return None
    return foot_pos_b[env_idx].cpu().numpy()


def _render_contact_trail_map(
    policy_nn,
    step: int,
    env_idx: int = 0,
    output_dir: str | None = None,
    show_window: bool = True,
) -> bool:
    contact_policy = _contact_trail_policy(policy_nn)
    if cv2 is None or contact_policy is None:
        return False

    memory = getattr(contact_policy, "contact_trail_memory", None)
    if memory is None or not hasattr(memory, "get_map"):
        return False
    trail_map = memory.get_map()
    if trail_map is None or trail_map.ndim != 4 or env_idx >= trail_map.shape[0]:
        return False

    env_map = trail_map[env_idx].detach().float().cpu()
    num_channels, grid_h, grid_w = env_map.shape
    cols = min(4, num_channels)
    rows = int(np.ceil(num_channels / cols))
    cell_px = 128
    tiles = []
    for channel_idx in range(rows * cols):
        if channel_idx < num_channels:
            channel = _normalize_image_u8(env_map[channel_idx]).numpy()
            channel = cv2.applyColorMap(channel, cv2.COLORMAP_VIRIDIS)
            channel = cv2.resize(channel, (cell_px, cell_px), interpolation=cv2.INTER_NEAREST)
            cv2.putText(
                channel,
                f"ch {channel_idx}",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (245, 245, 245),
                1,
                cv2.LINE_AA,
            )
        else:
            channel = np.zeros((cell_px, cell_px, 3), dtype=np.uint8)
        tiles.append(channel)

    grid_rows = [np.concatenate(tiles[row * cols : (row + 1) * cols], axis=1) for row in range(rows)]
    image = np.concatenate(grid_rows, axis=0)
    foot_positions = _contact_trail_foot_positions(contact_policy, env_idx)
    resolution = float(getattr(getattr(contact_policy, "trail_cfg", None), "resolution", 0.05))
    if foot_positions is not None:
        scale_x = cell_px / float(grid_w)
        scale_y = cell_px / float(grid_h)
        for foot in foot_positions:
            col = int(round((foot[0] / resolution + grid_w / 2.0) * scale_x))
            row = int(round((grid_h - (foot[1] / resolution + grid_h / 2.0)) * scale_y))
            for tile_row in range(rows):
                for tile_col in range(cols):
                    x = tile_col * cell_px + col
                    y = tile_row * cell_px + row
                    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
                        cv2.drawMarker(
                            image,
                            (x, y),
                            (0, 0, 255),
                            markerType=cv2.MARKER_CROSS,
                            markerSize=10,
                            thickness=1,
                        )

    cv2.putText(
        image,
        f"step={step} env={env_idx}",
        (image.shape[1] - 150, image.shape[0] - 10),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (245, 245, 245),
        1,
        cv2.LINE_AA,
    )
    if show_window:
        cv2.namedWindow("Contact Trail Map", cv2.WINDOW_NORMAL)
        cv2.imshow("Contact Trail Map", image)
        cv2.waitKey(1)
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        cv2.imwrite(os.path.join(output_dir, f"contact_trail_step_{step:06d}.png"), image)
    return True


def _render_attention_overlay(policy_nn, obs, step: int) -> bool:
    if cv2 is None:
        return False

    terrain_group = getattr(policy_nn, "terrain_obs_group", None)
    attention = _get_policy_attention(policy_nn)
    scan_shape = _get_policy_attention_scan_shape(policy_nn)
    if attention is None or scan_shape is None:
        return False

    if _is_ame_policy(policy_nn):
        obs_groups = getattr(policy_nn, "obs_groups", ["policy"])
        terrain_group = obs_groups[0] if obs_groups else "policy"
        try:
            actor_obs = obs[terrain_group][0]
        except Exception:
            return False
        map_scan_dim = getattr(policy_nn, "map_scan_dim", None)
        if map_scan_dim is None or len(map_scan_dim) < 3:
            return False
        map_l, map_w, coord_dim = (int(map_scan_dim[0]), int(map_scan_dim[1]), int(map_scan_dim[2]))
        map_scan_size = map_l * map_w * coord_dim
        if actor_obs.numel() < map_scan_size:
            return False
        map_scan = actor_obs[-map_scan_size:].reshape(map_w, map_l, coord_dim)
        if getattr(policy_nn, "cnn_downsample", False):
            map_scan = map_scan[::2, ::2]
        terrain_scan = map_scan[..., 2].reshape(-1)
    else:
        if terrain_group is None:
            return False
        try:
            terrain_scan = obs[terrain_group][0]
        except Exception:
            return False

    if attention.ndim != 3 or attention.shape[0] < 1:
        return False

    rows, cols = resolve_scan_shape(int(terrain_scan.numel()), scan_shape=scan_shape)
    if attention.shape[-1] != rows * cols:
        return False

    terrain_map = terrain_scan.reshape(rows, cols).detach().cpu()
    attention_map = attention[0].reshape(attention.shape[1], rows, cols).detach().cpu()
    scan_coordinates = None
    actor = getattr(policy_nn, "actor", None)
    terrain_encoder = getattr(actor, "terrain_encoder", None)
    if terrain_encoder is not None:
        scan_coordinates = getattr(terrain_encoder, "scan_coordinates", None)

    overlay = render_attention_point_overlay(
        terrain_map,
        attention_map,
        scan_coordinates=scan_coordinates,
    )
    if cv2 is not None:
        cv2.putText(
            overlay,
            f"step={step}",
            (overlay.shape[1] - 96, 22),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (235, 235, 235),
            1,
            cv2.LINE_AA,
        )
        cv2.namedWindow("Rough Elevation Attention", cv2.WINDOW_NORMAL)
        cv2.imshow("Rough Elevation Attention", overlay)
        cv2.waitKey(1)
    return True


def _get_motion_command_term(env):
    command_manager = getattr(env.unwrapped, "command_manager", None)
    if command_manager is None:
        return None

    try:
        return command_manager.get_term("motion")
    except Exception:
        return None


def _print_motion_status_if_changed(command, state: dict[str, str | None], force: bool = False) -> None:
    if command is None or not hasattr(command, "motion_status") or not hasattr(command, "current_motion_names"):
        return

    try:
        motion_name = command.current_motion_names([0])[0]
        status = command.motion_status(env_id=0)
    except Exception as exc:
        if not state.get("warned"):
            print(f"[WARN] Could not resolve active motion status: {exc}")
            state["warned"] = "1"
        return

    if force or motion_name != state.get("motion_name"):
        print(f"[Motion] Active reference: {status}")
        state["motion_name"] = motion_name


def _bind_motion_switch_keyboard(controller: Se2Keyboard, command, state: dict[str, str | None]) -> bool:
    motion = getattr(command, "motion", None)
    motion_names = getattr(motion, "motion_names", ())
    if not hasattr(command, "switch_motion") or len(motion_names) <= 1:
        return False

    def _next_motion():
        command.switch_motion(delta=1)
        _print_motion_status_if_changed(command, state, force=True)

    def _previous_motion():
        command.switch_motion(delta=-1)
        _print_motion_status_if_changed(command, state, force=True)

    controller.add_callback("N", _next_motion)
    controller.add_callback("P", _previous_motion)
    print("[INFO] Keyboard motion switching enabled: N=next motion, P=previous motion.")
    return True


def _resolve_camera_mode() -> tuple[str, bool]:
    """Return the active scripted camera mode and whether the user explicitly requested a camera mode."""
    requested_mode = str(args_cli.camera_follow_mode).lower()
    if requested_mode == "auto":
        task_name = str(args_cli.task or "")
        if "APEX-Flat-Privileged-Tracker" in task_name:
            return "none", False
        return ("follow" if args_cli.keyboard else "none"), False
    if requested_mode == "mouse":
        return "none", True
    return requested_mode, True


def _configure_viewer_for_camera_mode(env_cfg, camera_mode: str, camera_mode_requested: bool) -> None:
    """Disable IsaacLab asset-root recentering when the script or user owns the viewport camera."""
    if camera_mode != "none" or camera_mode_requested:
        env_cfg.viewer.origin_type = "world"
        env_cfg.viewer.asset_name = None
        env_cfg.viewer.body_name = None

    if camera_mode == "none" and camera_mode_requested:
        print("[INFO] Camera follow disabled; use the IsaacLab viewport mouse controls to move the camera.")
    elif camera_mode != "none":
        print(
            f"[INFO] Camera follow enabled: mode={camera_mode}, "
            f"smooth_window={max(1, args_cli.camera_smooth_window)}, "
            f"distance_scale={args_cli.camera_follow_distance_scale:g}"
        )


def _configure_play_resume_aware_terminations(env_cfg, checkpoint_path: str) -> None:
    """Align iteration-scheduled termination terms with the checkpoint being evaluated."""
    match = re.search(r"model_(\d+)\.pt$", os.path.basename(checkpoint_path))
    if match is None:
        return
    checkpoint_iteration = int(match.group(1))
    termination_cfg = getattr(env_cfg, "terminations", None)
    if termination_cfg is None:
        return
    for termination_name, term_cfg in vars(termination_cfg).items():
        params = getattr(term_cfg, "params", None)
        if not isinstance(params, dict) or "resume_iteration" not in params:
            continue
        params["resume_iteration"] = checkpoint_iteration
        print(
            f"[INFO] Play termination schedule '{termination_name}' starts at "
            f"checkpoint iteration {checkpoint_iteration}."
        )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    is_apex_task = "APEX" in task_name.upper()
    is_ame_task = "-AME-" in task_name.upper()
    is_rough_task = "ROUGH" in task_name.upper()
    attention_vis_requested = is_ame_task or (
        is_rough_task and getattr(getattr(agent_cfg, "policy", None), "class_name", "") == "RealTeacherActorCritic"
    )
    contact_trail_vis_requested = args_cli.contact_trail_vis or args_cli.contact_trail_vis_output_dir is not None

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 64
    if attention_vis_requested and args_cli.num_envs is None:
        env_cfg.scene.num_envs = 1
        print("[INFO] No --num_envs provided, set num_envs=1 for clearer terrain-attention visualization.")

    # handle deprecated configurations
    agent_cfg = migrate_custom_policy_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    if args_cli.motion_file is not None:
        if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion"):
            env_cfg.commands.motion.motion_file = _resolve_motion_source_path(args_cli.motion_file)
            env_cfg.commands.motion.motion_files = ()
            env_cfg.commands.motion.start_at_motion_beginning_on_reset = True
            print(f"[INFO] Using motion file override: {env_cfg.commands.motion.motion_file}")
            print("[INFO] Explicit motion playback starts at frame 0.")
        else:
            print("[WARN] --motion-file requested, but env has no `commands.motion` term.")
    elif hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion"):
        print(f"[INFO] Using default env motion source: {_describe_motion_source(env_cfg.commands.motion)}")
    _configure_smp_prior(env_cfg)
    if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion"):
        motion_cfg = env_cfg.commands.motion
        starts_at_beginning = bool(getattr(motion_cfg, "start_at_motion_beginning_on_reset", False))
        if starts_at_beginning and not args_cli.motion_reset_curriculum:
            if hasattr(motion_cfg, "reset_random_frame_probability"):
                motion_cfg.reset_random_frame_probability = 0.0
            if hasattr(motion_cfg, "reset_attached_frame_probability"):
                motion_cfg.reset_attached_frame_probability = 0.0
            print("[INFO] Motion playback resets at frame 0; training phase curriculum is disabled.")
        elif starts_at_beginning:
            print("[INFO] Preserving the task's randomized motion reset curriculum during play.")
    _configure_legacy_b2_z1_arm_moving_play(env_cfg)
    if hasattr(env_cfg, "actions") and hasattr(env_cfg.actions, "joint_pos"):
        action_cfg = env_cfg.actions.joint_pos
        if hasattr(action_cfg, "decap_lambda_start"):
            action_cfg.decap_prior_only = False
            action_cfg.decap_lambda_start = 0.0
            action_cfg.decap_lambda_end = 0.0
            action_cfg.decap_decay_type = "constant"
            action_cfg.decap_warmup_steps = 0
            action_cfg.decap_decay_start_step = 0
            action_cfg.decap_decay_end_step = 1
            print("[INFO] Disabled DecAP action prior for standard play; policy actions are sent without env-side prior.")
    if args_cli.prior_only:
        if hasattr(env_cfg, "actions") and hasattr(env_cfg.actions, "joint_pos"):
            action_cfg = env_cfg.actions.joint_pos
            if hasattr(action_cfg, "decap_prior_only"):
                action_cfg.decap_prior_only = True
                action_cfg.decap_lambda_start = 1.0
                action_cfg.decap_lambda_end = 1.0
                action_cfg.decap_decay_type = "constant"
                action_cfg.decap_warmup_steps = 0
                action_cfg.decap_decay_start_step = 0
                action_cfg.decap_decay_end_step = 1
                print("[INFO] Enabled prior-only play mode (policy action bypassed, prior scale fixed at 1.0).")
            else:
                print("[WARN] --prior-only requested, but env action term has no DecAP prior-only support.")
        else:
            print("[WARN] --prior-only requested, but env has no `actions.joint_pos` term.")

    # Spawn robots randomly in the grid and reduce generated terrain memory when
    # the task has a terrain importer. Direct environments may use a bare scene.
    _configure_play_terrain(env_cfg)

    # disable randomization for play
    _disable_play_observation_corruption(env_cfg)
    if _is_go2_d1_wbc_task() and not args_cli.go2_d1_play_domain_randomization:
        disabled_randomization = _disable_go2_d1_play_randomization(env_cfg)
        print(
            "[INFO] Go2+D1 nominal playback: disabled training domain randomization "
            f"({', '.join(disabled_randomization) or 'none configured'}). "
            "Pass --go2-d1-play-domain-randomization to retain it."
        )
    if args_cli.go2_d1_live_control:
        args_cli.loco_manip_stage = "off"
        print("[INFO] Go2+D1 live control overrides --loco-manip-stage to off.")
    elif _is_go2_d1_wbc_task() and args_cli.loco_manip_stage == "combined":
        print(
            "[INFO] Go2+D1 WBC play with default --loco-manip-stage=combined keeps training arm-command "
            "samplers active (the leg WBC uses asynchronous joint targets; the full WBC uses Cartesian EE targets). "
            "Use --go2-d1-live-control for manual EE commands, or --loco-manip-stage=curriculum "
            "to mirror training timing."
        )
    # remove random pushing
    _speed_up_loco_manip_curriculum_for_play(env_cfg)
    loco_manip_play_enabled = _configure_loco_manip_play_cfg(env_cfg, args_cli.loco_manip_stage)
    events_cfg = getattr(env_cfg, "events", None)
    curriculum_cfg = getattr(env_cfg, "curriculum", None)
    _set_attr_if_exists(events_cfg, "randomize_apply_external_force_torque", None)
    _set_attr_if_exists(events_cfg, "push_robot", None)
    if not loco_manip_play_enabled:
        _set_attr_if_exists(events_cfg, "randomize_push_robot", None)
    _set_attr_if_exists(curriculum_cfg, "command_levels_lin_vel", None)
    _set_attr_if_exists(curriculum_cfg, "command_levels_ang_vel", None)
    if loco_manip_play_enabled:
        if args_cli.loco_manip_stage == "cycle":
            print(
                f"[{_mounted_arm_task_label()}] Cycling play stages: fixed -> arm -> combined "
                f"every {max(1, args_cli.loco_manip_stage_cycle_steps)} steps."
            )
        elif args_cli.loco_manip_stage == "curriculum":
            print(f"[{_mounted_arm_task_label()}] Using training curriculum timing during play.")

    velocity_demo_enabled = _configure_velocity_demo(env_cfg)

    enable_motion_debug_vis = args_cli.motion_debug_vis or is_apex_task
    if enable_motion_debug_vis:
        commands_cfg = getattr(env_cfg, "commands", None)
        if hasattr(commands_cfg, "motion"):
            commands_cfg.motion.debug_vis = True
            print("[INFO] Enabled motion debug visualization overlays (commands.motion.debug_vis=True).")
            if hasattr(commands_cfg.motion, "debug_vis_show_pose_frames"):
                # APEX default in play: show velocity arrows + simplified foot-reference spheres.
                if is_apex_task and not args_cli.motion_velocity_vis_only and not args_cli.motion_foot_ref_vis:
                    commands_cfg.motion.debug_vis_show_pose_frames = False
                    commands_cfg.motion.debug_vis_show_velocity = True
                    commands_cfg.motion.debug_vis_simplified_foot_reference = True
                    print("[INFO] APEX play default debug mode: velocity arrows + simplified foot-reference spheres.")
                if args_cli.motion_velocity_vis_only:
                    commands_cfg.motion.debug_vis_show_pose_frames = False
                    commands_cfg.motion.debug_vis_show_velocity = True
                    commands_cfg.motion.debug_vis_simplified_foot_reference = False
                    print("[INFO] Motion debug visualization mode: velocity-only arrows.")
                if args_cli.motion_foot_ref_vis:
                    commands_cfg.motion.debug_vis_show_pose_frames = False
                    commands_cfg.motion.debug_vis_show_velocity = True
                    commands_cfg.motion.debug_vis_simplified_foot_reference = True
                    print("[INFO] Motion debug visualization mode: simplified reference-foot spheres.")
            elif args_cli.motion_velocity_vis_only or args_cli.motion_foot_ref_vis:
                print(
                    "[WARN] This task's motion command does not expose debug-vis mode toggles; "
                    "ignoring --motion-velocity-vis-only/--motion-foot-ref-vis."
                )
            if args_cli.num_envs is None:
                env_cfg.scene.num_envs = 1
                print("[INFO] No --num_envs provided, set num_envs=1 for a clearer overlay view.")
        else:
            print("[WARN] Motion debug visualization requested/enabled, but env has no `commands.motion` term.")

    keyboard_controller = None
    keyboard_velocity_active = False
    go2_d1_xbox_controller = _start_go2_d1_xbox() if args_cli.go2_d1_live_control else None
    if args_cli.go2_d1_live_control and go2_d1_xbox_controller is None:
        args_cli.keyboard = True
        print("[INFO] Go2+D1 live control falling back to keyboard input.")
    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        _set_attr_if_exists(getattr(env_cfg, "terminations", None), "time_out", None)
        commands_cfg = getattr(env_cfg, "commands", None)
        if hasattr(commands_cfg, "base_velocity"):
            commands_cfg.base_velocity.debug_vis = False
            config = Se2KeyboardCfg(
                v_x_sensitivity=commands_cfg.base_velocity.ranges.lin_vel_x[1],
                v_y_sensitivity=commands_cfg.base_velocity.ranges.lin_vel_y[1],
                omega_z_sensitivity=commands_cfg.base_velocity.ranges.ang_vel_z[1],
            )
            keyboard_controller = Se2Keyboard(config)
            keyboard_velocity_active = True
            env_cfg.observations.policy.velocity_commands = ObsTerm(
                func=_build_keyboard_velocity_observation(keyboard_controller),
            )
    go2_d1_live_control = (
        _configure_go2_d1_live_control(env_cfg, keyboard_controller) if args_cli.go2_d1_live_control else None
    )

    enable_heightmap_debug_vis = args_cli.heightmap_debug_vis or attention_vis_requested
    if enable_heightmap_debug_vis:
        height_scanner_cfg = getattr(env_cfg.scene, "height_scanner", None)
        if height_scanner_cfg is None:
            if args_cli.heightmap_debug_vis:
                print("[WARN] --heightmap-debug-vis requested, but env has no `scene.height_scanner` sensor.")
            elif attention_vis_requested:
                print("[WARN] Terrain-attention overlay requested, but env has no `scene.height_scanner` sensor.")
        else:
            if attention_vis_requested:
                _style_heightmap_visualizer_for_attention(height_scanner_cfg)
            else:
                height_scanner_cfg.debug_vis = True

            if attention_vis_requested and not args_cli.heightmap_debug_vis:
                print("[INFO] Enabled heightmap debug visualization automatically for terrain-attention overlay.")
            elif _actor_uses_height_scan(env_cfg, agent_cfg):
                print("[INFO] Enabled heightmap debug visualization (scene.height_scanner.debug_vis=True).")
            else:
                print(
                    "[INFO] Enabled height-scanner debug visualization, but active actor observations "
                    "do not include `height_scan`."
                )

            if args_cli.num_envs is None and not attention_vis_requested:
                env_cfg.scene.num_envs = 1
                print("[INFO] No --num_envs provided, set num_envs=1 for clearer heightmap visualization.")

    if args_cli.lidar_debug_vis:
        lidar_cfg = getattr(env_cfg.scene, "lidar", None)
        if lidar_cfg is None:
            print("[WARN] --lidar-debug-vis requested, but env has no `scene.lidar` sensor.")
        else:
            lidar_cfg.debug_vis = True
            print("[INFO] Enabled lidar debug visualization (scene.lidar.debug_vis=True).")
            if args_cli.num_envs is None:
                env_cfg.scene.num_envs = 1
                print("[INFO] No --num_envs provided, set num_envs=1 for clearer lidar visualization.")

    if contact_trail_vis_requested:
        if args_cli.num_envs is None:
            env_cfg.scene.num_envs = 1
            print("[INFO] No --num_envs provided, set num_envs=1 for clearer contact-trail visualization.")
        if cv2 is None:
            print("[WARN] Contact-trail visualization requested, but OpenCV (cv2) is not installed.")
        elif getattr(args_cli, "headless", False) and args_cli.contact_trail_vis_output_dir is None:
            print("[WARN] Contact-trail live window requested in headless mode without an output dir; disabling it.")
        elif getattr(args_cli, "headless", False):
            print("[WARN] --contact-trail-vis requested in headless mode; live window disabled, frame dumps still work.")

    camera_follow_mode, camera_mode_requested = _resolve_camera_mode()
    camera_follow_offset = None
    if camera_follow_mode == "follow":
        distance_scale = float(args_cli.camera_follow_distance_scale)
        if distance_scale <= 0.0:
            raise ValueError("--camera-follow-distance-scale must be greater than zero.")
        # Preserve task-authored framing while rotating the offset with the robot.
        base_offset = env_cfg.viewer.eye if env_cfg.viewer.origin_type == "asset_root" else (-3.0, 0.0, 0.5)
        camera_follow_offset = tuple(distance_scale * float(value) for value in base_offset)
    _configure_viewer_for_camera_mode(env_cfg, camera_follow_mode, camera_mode_requested)

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    print(f"[INFO] Loading experiment from directory: {log_root_path}")
    if args_cli.use_pretrained_checkpoint:
        resume_path = get_published_pretrained_checkpoint("rsl_rl", task_name)
        if not resume_path:
            print("[INFO] Unfortunately a pre-trained checkpoint is currently unavailable for this task.")
            return
    elif args_cli.checkpoint:
        resume_path = retrieve_file_path(args_cli.checkpoint)
    elif args_cli.wandb_run:
        wandb_selector = args_cli.wandb_checkpoint if args_cli.wandb_checkpoint is not None else agent_cfg.load_checkpoint
        print(f"[INFO] Resolving W&B checkpoint from run: {args_cli.wandb_run}")
        print(f"[INFO] W&B checkpoint selector: {wandb_selector}")
        resume_path = _resolve_wandb_checkpoint(args_cli.wandb_run, wandb_selector)
    else:
        resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

    _configure_play_resume_aware_terminations(env_cfg, resume_path)
    checkpoint_actor_input_dim = _checkpoint_actor_input_dim(resume_path)
    if _is_b2_z1_arm_moving_task() and checkpoint_actor_input_dim == 61:
        _configure_legacy_b2_z1_arm_moving_play(env_cfg, force=True)
    elif _is_b2_z1_arm_moving_task() and checkpoint_actor_input_dim not in (None, 59):
        print(
            "[WARN] B2-Z1 ArmMoving checkpoint actor input size is "
            f"{checkpoint_actor_input_dim}, but play auto-compatibility only knows 59D and 61D layouts."
        )

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

    deploy_cfg_path = maybe_export_deploy_cfg(env.unwrapped, log_dir)
    if deploy_cfg_path is not None:
        print(f"[INFO] Exported deploy config to: {deploy_cfg_path}")

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "play"),
            "step_trigger": lambda step: step == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        print("[INFO] Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    # wrap around environment for rsl-rl
    env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    loco_manip_play_state: dict[str, object] = {}
    if loco_manip_play_enabled and args_cli.loco_manip_stage == "grid":
        _configure_loco_manip_play_grid(env, loco_manip_play_state, force=True)
    elif loco_manip_play_enabled and args_cli.loco_manip_stage not in ("off", "curriculum"):
        initial_loco_manip_stage = _get_loco_manip_stage_name(args_cli.loco_manip_stage, timestep=0)
        _apply_loco_manip_play_stage(env, initial_loco_manip_stage, loco_manip_play_state, force=True)
    elif loco_manip_play_enabled and args_cli.loco_manip_stage == "curriculum":
        _print_loco_manip_curriculum_status(env, loco_manip_play_state, timestep=0, force=True)

    print(f"\033[1;36m[CHECKPOINT]\033[0m Loading policy from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "SmpOnPolicyRunner":
        from Gurukul.tasks.manager_based.smp.runner import SmpOnPolicyRunner

        runner = SmpOnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DecAPRunner":
        from decap import DecAPRunner

        runner = DecAPRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "MultiCriticRunner":
        from multi_critic import MultiCriticRunner

        runner = MultiCriticRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    checkpoint_infos = load_checkpoint_for_play(runner, resume_path)
    if agent_cfg.class_name == "SmpOnPolicyRunner":
        runner.restore_smp_infos(checkpoint_infos)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    motion_command_for_display = _get_motion_command_term(env)
    motion_display_state: dict[str, str | None] = {}
    if motion_command_for_display is not None:
        _print_motion_status_if_changed(motion_command_for_display, motion_display_state, force=True)

    if args_cli.keyboard and motion_command_for_display is not None:
        if keyboard_controller is None:
            keyboard_controller = Se2Keyboard(
                Se2KeyboardCfg(v_x_sensitivity=0.0, v_y_sensitivity=0.0, omega_z_sensitivity=0.0)
            )
        _bind_motion_switch_keyboard(
            keyboard_controller,
            motion_command_for_display,
            motion_display_state,
        )
    elif args_cli.keyboard and not keyboard_velocity_active:
        print("[WARN] --keyboard requested, but this task has neither `commands.base_velocity` nor `commands.motion`.")

    rsl_rl_version = version.parse(installed_version)
    attention_vis_enabled = False
    attention_vis_warned = False
    attention_heightmap_vis_enabled = False
    attention_heightmap_vis_warned = False
    attention_heightmap_visualizer = None
    contact_trail_vis_enabled = False
    contact_trail_vis_warned = False

    # Extract the neural network module when the active RSL-RL version still exposes it.
    runner_alg = getattr(runner, "alg", None)
    policy_nn = getattr(runner_alg, "policy", None)
    if policy_nn is None:
        policy_nn = getattr(runner_alg, "actor_critic", None)
    if policy_nn is None:
        policy_nn = getattr(runner_alg, "actor", None)

    if attention_vis_requested and policy_nn is not None and _policy_supports_attention_overlay(policy_nn):
        if cv2 is None:
            print("[WARN] OpenCV (cv2) is not installed; disabling terrain-attention window.")
        elif getattr(args_cli, "headless", False):
            print("[WARN] Headless mode requested; disabling terrain-attention window.")
        else:
            attention_vis_enabled = True
            print("[INFO] Enabled terrain-attention window for the rough elevation map.")

        try:
            attention_heightmap_visualizer = _build_attention_heightmap_visualizer(policy_nn)
            attention_heightmap_vis_enabled = True
            print("[INFO] Enabled terrain-attention markers on the heightmap debug visualization.")
        except Exception as exc:
            print(f"[WARN] Failed to create terrain-attention heightmap overlay: {exc}")
    elif attention_vis_requested:
        if policy_nn is None:
            print(
                "[WARN] Requested terrain-attention visualization, but the loaded runner does not expose "
                "a policy module for attention introspection."
            )
        else:
            print("[WARN] Requested terrain-attention visualization, but the loaded policy does not expose attention maps.")

    if contact_trail_vis_requested and policy_nn is not None:
        if _contact_trail_policy(policy_nn) is None:
            print("[WARN] Contact-trail visualization requested, but the loaded policy does not expose contact trail memory.")
        elif cv2 is not None and (not getattr(args_cli, "headless", False) or args_cli.contact_trail_vis_output_dir):
            contact_trail_vis_enabled = True
            if getattr(args_cli, "headless", False):
                print("[INFO] Enabled contact-trail map frame dumps during play.")
            else:
                print("[INFO] Enabled contact-trail map window during play.")
    elif contact_trail_vis_requested:
        print("[WARN] Contact-trail visualization requested, but the loaded runner does not expose a policy module.")

    # export policy to onnx/jit
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    if rsl_rl_version >= version.parse("4.0.0"):
        # use the new export functions for rsl-rl >= 4.0.0
        try:
            runner.export_policy_to_jit(path=export_model_dir, filename="policy.pt")
        except Exception as exc:
            print(f"[WARN] Skipping JIT export via runner for RSL-RL {installed_version}: {exc}")
        try:
            runner.export_policy_to_onnx(path=export_model_dir, filename="policy.onnx")
        except Exception as exc:
            print(f"[WARN] Skipping ONNX export via runner for RSL-RL {installed_version}: {exc}")
    else:
        if policy_nn is None:
            print("[WARN] Skipping policy export; loaded runner does not expose a policy module.")
        else:
            # extract the normalizer
            if hasattr(policy_nn, "actor_obs_normalizer"):
                normalizer = policy_nn.actor_obs_normalizer
            elif hasattr(policy_nn, "student_obs_normalizer"):
                normalizer = policy_nn.student_obs_normalizer
            else:
                normalizer = None

            try:
                export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
            except Exception as exc:
                print(f"[WARN] Skipping JIT export for policy type {type(policy_nn).__name__}: {exc}")
            try:
                export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
            except Exception as exc:
                print(f"[WARN] Skipping ONNX export for policy type {type(policy_nn).__name__}: {exc}")

    dt = env.unwrapped.step_dt

    # reset environment so spawn pose / reset events match training before the first policy step
    obs, _ = env.reset()
    if loco_manip_play_enabled and args_cli.loco_manip_stage == "grid":
        _apply_loco_manip_play_grid_commands(env, loco_manip_play_state)
        obs = env.get_observations()
    if go2_d1_live_control is not None:
        _apply_go2_d1_live_command(env.unwrapped, go2_d1_live_control)
        obs = env.get_observations()
    velocity_demo_steps = args_cli.video_length if args_cli.video else 800
    velocity_demo_state: dict[str, str] = {}
    if velocity_demo_enabled:
        _apply_velocity_demo(env, timestep=0, total_steps=velocity_demo_steps, state=velocity_demo_state)
        obs = env.get_observations()
    apex_delta_logger = ApexTrackerDeltaLogger(args_cli.apex_delta_log, args_cli.apex_delta_log_steps)
    if apex_delta_logger.enabled:
        print(f"[APEX delta] logging actor slices to {os.path.abspath(os.path.expanduser(args_cli.apex_delta_log))}")
    timestep = 0
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        if loco_manip_play_enabled and args_cli.loco_manip_stage == "cycle":
            current_loco_manip_stage = _get_loco_manip_stage_name(args_cli.loco_manip_stage, timestep)
            _apply_loco_manip_play_stage(env, current_loco_manip_stage, loco_manip_play_state)
        if loco_manip_play_enabled and args_cli.loco_manip_stage == "grid":
            _apply_loco_manip_play_grid_commands(env, loco_manip_play_state, timestep=timestep)
            obs = env.get_observations()
        if go2_d1_live_control is not None:
            _poll_go2_d1_live_control(go2_d1_live_control, go2_d1_xbox_controller, dt)
            _apply_go2_d1_live_command(env.unwrapped, go2_d1_live_control)
            obs = env.get_observations()
        if velocity_demo_enabled:
            _apply_velocity_demo(env, timestep, velocity_demo_steps, velocity_demo_state)
            obs = env.get_observations()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            apex_delta_logger.record(timestep, obs, actions)
            if contact_trail_vis_enabled and timestep % max(1, args_cli.contact_trail_vis_interval) == 0:
                try:
                    output_dir = None
                    if args_cli.contact_trail_vis_output_dir is not None and (
                        timestep % max(1, args_cli.contact_trail_vis_save_interval) == 0
                    ):
                        output_dir = os.path.abspath(os.path.expanduser(args_cli.contact_trail_vis_output_dir))
                    rendered = _render_contact_trail_map(
                        policy_nn,
                        timestep + 1,
                        env_idx=max(0, int(args_cli.contact_trail_vis_env_idx)),
                        output_dir=output_dir,
                        show_window=not getattr(args_cli, "headless", False),
                    )
                    if not rendered and not contact_trail_vis_warned:
                        print("[WARN] Contact-trail visualization is enabled, but no map frame was produced.")
                        contact_trail_vis_warned = True
                except cv2.error as exc:
                    print(f"[WARN] Failed to render contact-trail map window: {exc}")
                    contact_trail_vis_enabled = False
            if attention_vis_enabled:
                try:
                    overlay_rendered = _render_attention_overlay(policy_nn, obs, timestep + 1)
                    if not overlay_rendered and not attention_vis_warned:
                        print("[WARN] Terrain-attention window is enabled, but no attention frame was produced.")
                        attention_vis_warned = True
                except cv2.error as exc:
                    print(f"[WARN] Failed to render terrain-attention window: {exc}")
                    attention_vis_enabled = False
            if attention_heightmap_vis_enabled:
                try:
                    heightmap_overlay_rendered = _render_attention_heightmap_overlay(
                        policy_nn,
                        env,
                        attention_heightmap_visualizer,
                    )
                    if not heightmap_overlay_rendered and not attention_heightmap_vis_warned:
                        print("[WARN] Heightmap attention overlay is enabled, but no marker frame was produced.")
                        attention_heightmap_vis_warned = True
                except Exception as exc:
                    print(f"[WARN] Failed to render terrain-attention heightmap overlay: {exc}")
                    attention_heightmap_vis_enabled = False
            # env stepping
            obs, rewards, dones, extras = env.step(actions)
            if go2_d1_live_control is not None:
                _apply_go2_d1_live_command(env.unwrapped, go2_d1_live_control)
                if torch.any(dones):
                    go2_d1_live_control.reset_ee(reset_deployment=True)
            if motion_command_for_display is not None:
                _print_motion_status_if_changed(motion_command_for_display, motion_display_state)
            if loco_manip_play_enabled and args_cli.loco_manip_stage == "curriculum":
                _print_loco_manip_curriculum_status(env, loco_manip_play_state, timestep + 1)
            # reset recurrent states for episodes that have terminated
            if rsl_rl_version >= version.parse("4.0.0"):
                policy_reset = getattr(policy, "reset", None)
                if policy_reset is not None:
                    policy_reset(dones)
            else:
                policy_nn_reset = getattr(policy_nn, "reset", None)
                if policy_nn_reset is not None:
                    policy_nn_reset(dones)
        timestep += 1

        if args_cli.print_rewards and (timestep == 1 or timestep % max(1, args_cli.reward_print_interval) == 0):
            reward_terms = _format_reward_terms(env, env_idx=0)
            total_reward = rewards[0].item() if rewards.numel() > 0 else 0.0
            terminated = int(dones[0].item()) if dones.numel() > 0 else 0
            print(f"[Rewards][step={timestep}] total={total_reward:.4f} done={terminated} | {reward_terms}")
        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        if args_cli.max_steps is not None and timestep >= args_cli.max_steps:
            break

        if camera_follow_mode != "none":
            camera_follow(
                env,
                mode=camera_follow_mode,
                window_size=args_cli.camera_smooth_window,
                env_index=0,
                follow_offset=camera_follow_offset,
            )

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    if attention_vis_enabled and cv2 is not None:
        try:
            cv2.destroyWindow("Rough Elevation Attention")
        except cv2.error:
            pass
    if contact_trail_vis_enabled and cv2 is not None:
        try:
            cv2.destroyWindow("Contact Trail Map")
        except cv2.error:
            pass
    if attention_heightmap_visualizer is not None:
        attention_heightmap_visualizer.set_visibility(False)
    apex_delta_logger.close()
    if go2_d1_xbox_controller is not None:
        go2_d1_xbox_controller[0].quit()

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
