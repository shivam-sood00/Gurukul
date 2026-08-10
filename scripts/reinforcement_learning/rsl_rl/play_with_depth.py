# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to play a checkpoint and visualize depth observations in real-time."""

"""Launch Isaac Sim Simulator first."""

import argparse
import importlib.metadata as metadata
import math
import re
import sys
import tempfile
from dataclasses import MISSING
from types import SimpleNamespace

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Play an RL checkpoint with optional live depth visualization.")
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
    "--camera_follow_mode",
    type=str,
    default="follow",
    choices=["none", "follow", "isometric", "topdown"],
    help="Viewport camera follow mode.",
)
parser.add_argument(
    "--camera_smooth_window",
    type=int,
    default=50,
    help="Smoothing window size for camera follow.",
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
    "--motion-file",
    type=str,
    default=None,
    help="Optional NPZ file to override env commands.motion.motion_file.",
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
parser.add_argument(
    "--no_depth_vis",
    action="store_true",
    default=False,
    help="Disable live depth window visualization.",
)
parser.add_argument(
    "--depth_env_index",
    type=int,
    default=0,
    help="Environment index whose depth image is shown in the depth window.",
)
parser.add_argument(
    "--depth_window_scale",
    type=int,
    default=6,
    help="Integer up-scaling factor applied to the depth image window.",
)
parser.add_argument(
    "--depth_colormap",
    type=str,
    default="turbo",
    choices=["gray", "turbo", "jet", "inferno", "magma"],
    help="Color map used in the depth image window.",
)
parser.add_argument(
    "--depth_print_interval",
    type=int,
    default=100,
    help="Print depth min/max stats every N env steps (0 disables printing).",
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
# depth tasks require camera support enabled
if not args_cli.no_depth_vis:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for installed RSL-RL version."""

installed_version = metadata.version("rsl-rl-lib")

"""Rest everything follows."""

import os
import time

import gymnasium as gym
import torch
from rsl_rl.runners import DistillationRunner, OnPolicyRunner
from rsl_rl.utils import resolve_callable

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
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.markers import VisualizationMarkers
from isaaclab.markers.config import GREEN_ARROW_X_MARKER_CFG
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.math import quat_from_euler_xyz, quat_mul

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

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from legacy_checkpoint import load_checkpoint_for_play
from rl_utils import camera_follow

# PLACEHOLDER: Extension template (do not remove this comment)


def _resolve_motion_source_path(path: str) -> str:
    """Resolve local motion files/directories while preserving Isaac asset-path support."""
    expanded_path = os.path.expanduser(path)
    if os.path.exists(expanded_path):
        return expanded_path
    return retrieve_file_path(path)


def _is_missing_cfg(value) -> bool:
    return isinstance(value, type(MISSING))


def _uses_legacy_distillation_policy(agent_cfg: RslRlBaseRunnerCfg) -> bool:
    """Detect old combined student-teacher distillation policies that RSL-RL 5 no longer constructs."""
    if getattr(agent_cfg, "class_name", "") != "DistillationRunner":
        return False

    policy_cfg = getattr(agent_cfg, "policy", MISSING)
    if _is_missing_cfg(policy_cfg):
        return False

    student_cfg = getattr(agent_cfg, "student", MISSING)
    if not _is_missing_cfg(student_cfg) and getattr(student_cfg, "class_name", None):
        return False

    policy_class_name = getattr(policy_cfg, "class_name", "")
    return policy_class_name not in {"", "StudentTeacher", "StudentTeacherRecurrent"}


class _LegacyDistillationPlayRunner:
    """Minimal playback runner for pre-RSL-RL-5 combined StudentTeacher policies."""

    is_legacy_distillation_play_runner = True

    def __init__(self, env, train_cfg: dict, device: str):
        self.env = env
        self.cfg = train_cfg
        self.device = device
        self.current_learning_iteration = 0

        policy_cfg = dict(train_cfg["policy"])
        policy_class_name = policy_cfg.pop("class_name")
        policy_class = resolve_callable(policy_class_name)
        obs_groups = dict(train_cfg["obs_groups"])
        obs = env.get_observations()

        policy = policy_class(obs, obs_groups, env.num_actions, **policy_cfg).to(device)
        self.alg = SimpleNamespace(policy=policy)

    def load(self, checkpoint_path: str, strict: bool = True):
        loaded_dict = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
        state_dict = loaded_dict.get("model_state_dict")
        if state_dict is None:
            raise KeyError(
                "Legacy distillation playback expects a checkpoint with `model_state_dict`; "
                f"available keys: {sorted(loaded_dict.keys())}"
            )
        self.alg.policy.load_state_dict(state_dict, strict=strict)
        if "iter" in loaded_dict:
            self.current_learning_iteration = loaded_dict["iter"]
        return loaded_dict.get("infos")

    def get_inference_policy(self, device=None):
        if device is not None:
            self.alg.policy.to(device)
        self.alg.policy.eval()
        return self.alg.policy.act_inference


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
        actor_groups = obs_groups_cfg.get("policy", ["policy"])
    if isinstance(actor_groups, str):
        actor_groups = [actor_groups]

    for group_name in actor_groups:
        group_cfg = getattr(observations_cfg, group_name, None)
        if group_cfg is None:
            continue
        if getattr(group_cfg, "height_scan", None) is not None:
            return True
    return False


def _build_keyboard_velocity_observation(controller: Se2Keyboard, command_cache: dict | None = None):
    # Cache one command per environment step to keep all observation groups consistent.
    cache = command_cache if command_cache is not None else {"step": -1, "command": torch.zeros((1, 3), dtype=torch.float32)}

    def _keyboard_velocity_commands(env):
        step_counter = getattr(env, "common_step_counter", 0)
        if isinstance(step_counter, torch.Tensor):
            step_counter = int(step_counter.item())
        else:
            step_counter = int(step_counter)

        if cache["step"] != step_counter:
            cache["step"] = step_counter
            cache["command"] = torch.tensor(controller.advance(), dtype=torch.float32, device=env.device).unsqueeze(0)

        if getattr(env, "num_envs", 1) <= 1:
            return cache["command"]
        return cache["command"].expand(env.num_envs, -1)

    return _keyboard_velocity_commands


def _apply_keyboard_velocity_to_observation_groups(
    env_cfg,
    controller: Se2Keyboard,
    command_cache: dict | None = None,
) -> list[str]:
    applied_groups: list[str] = []
    observations_cfg = getattr(env_cfg, "observations", None)
    if observations_cfg is None:
        return applied_groups

    keyboard_velocity_fn = _build_keyboard_velocity_observation(controller, command_cache)
    for group_name, group_cfg in vars(observations_cfg).items():
        if group_name.startswith("_") or group_cfg is None:
            continue
        if hasattr(group_cfg, "velocity_commands"):
            # Create one term per group: sharing a single ObsTerm instance across groups can
            # cause history settings to bleed between groups during ObservationManager setup.
            setattr(group_cfg, "velocity_commands", ObsTerm(func=keyboard_velocity_fn))
            applied_groups.append(group_name)

    return applied_groups


def _create_keyboard_command_visualizer() -> VisualizationMarkers:
    marker_cfg = GREEN_ARROW_X_MARKER_CFG.replace(prim_path="/Visuals/Command/keyboard_velocity_goal")
    marker_cfg.markers["arrow"].scale = (0.6, 0.6, 0.6)
    visualizer = VisualizationMarkers(marker_cfg)
    visualizer.set_visibility(True)
    return visualizer


def _resolve_xy_velocity_to_arrow(
    xy_velocity: torch.Tensor,
    base_quat_w: torch.Tensor,
    default_scale: tuple[float, float, float],
) -> tuple[torch.Tensor, torch.Tensor]:
    arrow_scale = torch.tensor(default_scale, device=xy_velocity.device).repeat(xy_velocity.shape[0], 1)
    arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0
    heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
    zeros = torch.zeros_like(heading_angle)
    arrow_quat = quat_from_euler_xyz(zeros, zeros, heading_angle)
    return arrow_scale, quat_mul(base_quat_w, arrow_quat)


def _visualize_keyboard_command_velocity(
    env,
    visualizer: VisualizationMarkers,
    command_cache: dict | None,
):
    if command_cache is None:
        return

    scene = getattr(env.unwrapped, "scene", None)
    if scene is None:
        return

    try:
        robot = scene["robot"]
    except Exception:
        return

    if not robot.is_initialized:
        return

    command = command_cache.get("command")
    if not isinstance(command, torch.Tensor) or command.ndim != 2 or command.shape[1] < 2:
        return

    base_pos_w = robot.data.root_pos_w
    base_quat_w = robot.data.root_quat_w
    num_envs = base_pos_w.shape[0]

    if command.shape[0] == 1 and num_envs > 1:
        command = command.expand(num_envs, -1)
    elif command.shape[0] != num_envs:
        command = command[:num_envs]
    command = command.to(device=base_pos_w.device)

    marker_pos = base_pos_w.clone()
    marker_pos[:, 2] += 0.5
    marker_scale, marker_quat = _resolve_xy_velocity_to_arrow(
        command[:, :2],
        base_quat_w,
        visualizer.cfg.markers["arrow"].scale,
    )
    visualizer.visualize(marker_pos, marker_quat, marker_scale)


def _infer_depth_obs_resize(env_cfg) -> tuple[int, int] | None:
    depth_group = getattr(getattr(env_cfg, "observations", None), "depth_camera", None)
    if depth_group is None:
        return None
    depth_term = getattr(depth_group, "depth_image", None)
    if depth_term is None:
        return None
    params = getattr(depth_term, "params", None)
    if not isinstance(params, dict):
        return None
    resize = params.get("resize")
    if isinstance(resize, (list, tuple)) and len(resize) == 2:
        height, width = int(resize[0]), int(resize[1])
        if height > 0 and width > 0:
            return (height, width)
    return None


def _guess_2d_shape(numel: int) -> tuple[int, int] | None:
    if numel <= 0:
        return None
    if numel == 58 * 87:
        return (58, 87)
    side = int(math.sqrt(numel))
    if side * side == numel:
        return (side, side)
    return None


def _extract_depth_frame(
    env,
    obs,
    extras,
    env_index: int,
    expected_obs_shape: tuple[int, int] | None,
) -> tuple[torch.Tensor | None, str]:
    # 1) Prefer raw sensor output for true rendered depth.
    scene = getattr(env.unwrapped, "scene", None)
    if scene is not None:
        try:
            depth_sensor = scene["depth_camera"]
        except Exception:
            depth_sensor = None
        if depth_sensor is not None:
            sensor_data = getattr(getattr(depth_sensor, "data", None), "output", None)
            if isinstance(sensor_data, dict) and "distance_to_camera" in sensor_data:
                depth_tensor = sensor_data["distance_to_camera"]
                if depth_tensor.ndim >= 3 and 0 <= env_index < depth_tensor.shape[0]:
                    depth_frame = depth_tensor[env_index]
                    if depth_frame.ndim == 3 and depth_frame.shape[-1] == 1:
                        depth_frame = depth_frame[..., 0]
                    if depth_frame.ndim == 2:
                        return depth_frame, "sensor_raw"

    # 2) Fall back to depth observation tensor from obs/extras.
    depth_obs = None
    if isinstance(obs, dict) and "depth_camera" in obs:
        depth_obs = obs["depth_camera"]
    elif isinstance(extras, dict):
        obs_dict = extras.get("observations")
        if isinstance(obs_dict, dict) and "depth_camera" in obs_dict:
            depth_obs = obs_dict["depth_camera"]

    if depth_obs is None or depth_obs.ndim < 2 or not (0 <= env_index < depth_obs.shape[0]):
        return None, ""

    depth_frame = depth_obs[env_index]
    if depth_frame.ndim == 1:
        shape = expected_obs_shape if expected_obs_shape is not None else _guess_2d_shape(depth_frame.numel())
        if shape is None:
            return None, ""
        depth_frame = depth_frame.view(shape[0], shape[1])
    elif depth_frame.ndim == 3 and depth_frame.shape[-1] == 1:
        depth_frame = depth_frame[..., 0]

    if depth_frame.ndim != 2:
        return None, ""
    return depth_frame, "observation"


def _depth_to_u8(
    depth_frame: torch.Tensor,
    source: str,
    max_distance: float,
) -> torch.Tensor:
    depth = torch.nan_to_num(depth_frame.float(), nan=0.0, posinf=0.0, neginf=0.0)
    if source == "sensor_raw" and max_distance > 0.0:
        normalized = torch.clamp(depth / max_distance, 0.0, 1.0)
    elif source == "observation":
        normalized = torch.clamp(depth + 0.5, 0.0, 1.0)
    else:
        d_min = depth.min()
        d_max = depth.max()
        denom = torch.clamp(d_max - d_min, min=1e-6)
        normalized = (depth - d_min) / denom
    return (normalized * 255.0).to(torch.uint8)


def _apply_colormap(depth_u8, colormap: str):
    if colormap == "gray":
        return depth_u8
    cmap_map = {
        "turbo": cv2.COLORMAP_TURBO,
        "jet": cv2.COLORMAP_JET,
        "inferno": cv2.COLORMAP_INFERNO,
        "magma": cv2.COLORMAP_MAGMA,
    }
    return cv2.applyColorMap(depth_u8, cmap_map[colormap])


@hydra_task_config(args_cli.task, args_cli.agent)
def main(env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg):
    """Play with RSL-RL agent."""
    # grab task name for checkpoint path
    task_name = args_cli.task.split(":")[-1]
    is_apex_task = "APEX" in task_name.upper()

    # override configurations with non-hydra CLI arguments
    agent_cfg: RslRlBaseRunnerCfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else 64
    use_legacy_distillation_policy = _uses_legacy_distillation_policy(agent_cfg)
    if use_legacy_distillation_policy:
        print(
            "[INFO] Using legacy combined student-teacher distillation playback for "
            f"{agent_cfg.policy.class_name}."
        )
    else:
        agent_cfg = migrate_custom_policy_cfg(agent_cfg, installed_version)

    # set the environment seed
    # note: certain randomizations occur in the environment initialization so we set the seed here
    env_cfg.seed = agent_cfg.seed
    env_cfg.sim.device = args_cli.device if args_cli.device is not None else env_cfg.sim.device
    depth_obs_shape = _infer_depth_obs_resize(env_cfg)
    depth_max_distance = float(getattr(getattr(env_cfg.scene, "depth_camera", None), "max_distance", 0.0))
    depth_vis_enabled = not args_cli.no_depth_vis
    if depth_vis_enabled:
        if cv2 is None:
            print("[WARN] OpenCV (cv2) is not installed; disabling depth visualization.")
            depth_vis_enabled = False
        elif getattr(args_cli, "headless", False):
            print("[WARN] Headless mode requested; disabling depth window visualization.")
            depth_vis_enabled = False
    if args_cli.motion_file is not None:
        if hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion"):
            env_cfg.commands.motion.motion_file = _resolve_motion_source_path(args_cli.motion_file)
            env_cfg.commands.motion.motion_files = ()
            print(f"[INFO] Using motion file override: {env_cfg.commands.motion.motion_file}")
        else:
            print("[WARN] --motion-file requested, but env has no `commands.motion` term.")
    elif hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion"):
        print(f"[INFO] Using default env motion file: {env_cfg.commands.motion.motion_file}")
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

    # start play from the easiest terrain level
    env_cfg.scene.terrain.max_init_terrain_level = 0
    _set_attr_if_exists(env_cfg.curriculum, "terrain_levels", None)
    # reduce the number of terrains to save memory
    if env_cfg.scene.terrain.terrain_generator is not None:
        env_cfg.scene.terrain.terrain_generator.num_rows = 5
        env_cfg.scene.terrain.terrain_generator.num_cols = 5
        env_cfg.scene.terrain.terrain_generator.curriculum = False

    # disable randomization for play
    env_cfg.observations.policy.enable_corruption = False
    # remove random pushing
    _set_attr_if_exists(env_cfg.events, "randomize_apply_external_force_torque", None)
    _set_attr_if_exists(env_cfg.events, "push_robot", None)
    _set_attr_if_exists(env_cfg.events, "randomize_push_robot", None)
    _set_attr_if_exists(env_cfg.curriculum, "command_levels_lin_vel", None)
    _set_attr_if_exists(env_cfg.curriculum, "command_levels_ang_vel", None)

    enable_motion_debug_vis = args_cli.motion_debug_vis or is_apex_task
    if enable_motion_debug_vis:
        if hasattr(env_cfg.commands, "motion"):
            env_cfg.commands.motion.debug_vis = True
            print("[INFO] Enabled motion debug visualization overlays (commands.motion.debug_vis=True).")
            if hasattr(env_cfg.commands.motion, "debug_vis_show_pose_frames"):
                # APEX default in play: show velocity arrows + simplified foot-reference spheres.
                if is_apex_task and not args_cli.motion_velocity_vis_only and not args_cli.motion_foot_ref_vis:
                    env_cfg.commands.motion.debug_vis_show_pose_frames = False
                    env_cfg.commands.motion.debug_vis_show_velocity = True
                    env_cfg.commands.motion.debug_vis_simplified_foot_reference = True
                    print("[INFO] APEX play default debug mode: velocity arrows + simplified foot-reference spheres.")
                if args_cli.motion_velocity_vis_only:
                    env_cfg.commands.motion.debug_vis_show_pose_frames = False
                    env_cfg.commands.motion.debug_vis_show_velocity = True
                    env_cfg.commands.motion.debug_vis_simplified_foot_reference = False
                    print("[INFO] Motion debug visualization mode: velocity-only arrows.")
                if args_cli.motion_foot_ref_vis:
                    env_cfg.commands.motion.debug_vis_show_pose_frames = False
                    env_cfg.commands.motion.debug_vis_show_velocity = True
                    env_cfg.commands.motion.debug_vis_simplified_foot_reference = True
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

    keyboard_command_cache = None
    if args_cli.keyboard:
        env_cfg.scene.num_envs = 1
        env_cfg.terminations.time_out = None
        if hasattr(env_cfg.commands, "base_velocity"):
            env_cfg.commands.base_velocity.debug_vis = False
            config = Se2KeyboardCfg(
                v_x_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_x[1],
                v_y_sensitivity=env_cfg.commands.base_velocity.ranges.lin_vel_y[1],
                omega_z_sensitivity=env_cfg.commands.base_velocity.ranges.ang_vel_z[1],
            )
            controller = Se2Keyboard(config)
            keyboard_command_cache = {"step": -1, "command": torch.zeros((1, 3), dtype=torch.float32)}
            applied_obs_groups = _apply_keyboard_velocity_to_observation_groups(
                env_cfg,
                controller,
                keyboard_command_cache,
            )
            if applied_obs_groups:
                print(
                    "[INFO] Enabled keyboard velocity control for observation groups: "
                    + ", ".join(applied_obs_groups)
                )
            else:
                print("[WARN] Keyboard requested but no observation groups expose `velocity_commands`.")
            print(
                "[INFO] Keyboard controls: UP/DOWN for +/-x, LEFT/RIGHT for +/-y, "
                "Z/X for +/-yaw, L to reset commands."
            )
        else:
            print("[WARN] --keyboard is only supported for tasks with `commands.base_velocity`.")

    if args_cli.heightmap_debug_vis:
        height_scanner_cfg = getattr(env_cfg.scene, "height_scanner", None)
        if height_scanner_cfg is None:
            print("[WARN] --heightmap-debug-vis requested, but env has no `scene.height_scanner` sensor.")
        else:
            height_scanner_cfg.debug_vis = True
            if _actor_uses_height_scan(env_cfg, agent_cfg):
                print("[INFO] Enabled heightmap debug visualization (scene.height_scanner.debug_vis=True).")
            else:
                print(
                    "[INFO] Enabled height-scanner debug visualization, but active actor observations "
                    "do not include `height_scan`."
                )
            if args_cli.num_envs is None:
                env_cfg.scene.num_envs = 1
                print("[INFO] No --num_envs provided, set num_envs=1 for clearer heightmap visualization.")

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

    log_dir = os.path.dirname(resume_path)

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)

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

    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    # load previously trained model
    if agent_cfg.class_name == "OnPolicyRunner":
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        if use_legacy_distillation_policy:
            runner = _LegacyDistillationPlayRunner(env, agent_cfg.to_dict(), device=agent_cfg.device)
        else:
            runner = DistillationRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "DecAPRunner":
        from decap import DecAPRunner

        runner = DecAPRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    elif agent_cfg.class_name == "MultiCriticRunner":
        from multi_critic import MultiCriticRunner

        runner = MultiCriticRunner(env, agent_cfg.to_dict(), log_dir=None, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    if getattr(runner, "is_legacy_distillation_play_runner", False):
        runner.load(resume_path)
    else:
        load_checkpoint_for_play(runner, resume_path)

    # obtain the trained policy for inference
    policy = runner.get_inference_policy(device=env.unwrapped.device)

    # extract the neural network module
    # we do this in a try-except to maintain backwards compatibility.
    try:
        # version 2.3 onwards
        policy_nn = runner.alg.policy
    except AttributeError:
        # version 2.2 and below
        policy_nn = runner.alg.actor_critic

    # extract the normalizer
    if hasattr(policy_nn, "actor_obs_normalizer"):
        normalizer = policy_nn.actor_obs_normalizer
    elif hasattr(policy_nn, "student_obs_normalizer"):
        normalizer = policy_nn.student_obs_normalizer
    else:
        normalizer = None

    # export policy to onnx/jit (best-effort; some custom policies are not compatible with default exporters)
    export_model_dir = os.path.join(os.path.dirname(resume_path), "exported")
    try:
        export_policy_as_jit(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.pt")
    except Exception as exc:
        print(f"[WARN] Skipping JIT export for policy type {type(policy_nn).__name__}: {exc}")
    try:
        export_policy_as_onnx(policy_nn, normalizer=normalizer, path=export_model_dir, filename="policy.onnx")
    except Exception as exc:
        print(f"[WARN] Skipping ONNX export for policy type {type(policy_nn).__name__}: {exc}")

    dt = env.unwrapped.step_dt

    # reset environment
    obs = env.get_observations()
    timestep = 0
    if args_cli.camera_follow_mode != "none":
        print(
            f"[INFO] Camera follow enabled: mode={args_cli.camera_follow_mode}, "
            f"smooth_window={max(1, args_cli.camera_smooth_window)}"
        )
    depth_window_name = "Depth Camera (env index {})".format(args_cli.depth_env_index)
    depth_gui_error_reported = False
    keyboard_velocity_visualizer = None
    if args_cli.keyboard and keyboard_command_cache is not None:
        keyboard_velocity_visualizer = _create_keyboard_command_visualizer()
        print("[INFO] Enabled keyboard commanded-velocity arrow visualization above the robot.")
    # simulate environment
    while simulation_app.is_running():
        start_time = time.time()
        # run everything in inference mode
        with torch.inference_mode():
            # agent stepping
            actions = policy(obs)
            # env stepping
            obs, rewards, dones, extras = env.step(actions)
            # reset recurrent states for episodes that have terminated
            policy_nn.reset(dones)
        timestep += 1

        if args_cli.print_rewards and (timestep == 1 or timestep % max(1, args_cli.reward_print_interval) == 0):
            reward_terms = _format_reward_terms(env, env_idx=0)
            total_reward = rewards[0].item() if rewards.numel() > 0 else 0.0
            terminated = int(dones[0].item()) if dones.numel() > 0 else 0
            print(f"[Rewards][step={timestep}] total={total_reward:.4f} done={terminated} | {reward_terms}")

        if keyboard_velocity_visualizer is not None:
            _visualize_keyboard_command_velocity(env, keyboard_velocity_visualizer, keyboard_command_cache)

        if depth_vis_enabled:
            depth_frame, source = _extract_depth_frame(
                env=env,
                obs=obs,
                extras=extras,
                env_index=args_cli.depth_env_index,
                expected_obs_shape=depth_obs_shape,
            )
            if depth_frame is not None:
                depth_u8 = _depth_to_u8(depth_frame, source=source, max_distance=depth_max_distance)
                depth_np = depth_u8.detach().cpu().numpy()
                if args_cli.depth_window_scale > 1:
                    depth_np = cv2.resize(
                        depth_np,
                        None,
                        fx=args_cli.depth_window_scale,
                        fy=args_cli.depth_window_scale,
                        interpolation=cv2.INTER_NEAREST,
                    )
                depth_img = _apply_colormap(depth_np, args_cli.depth_colormap)
                try:
                    cv2.imshow(depth_window_name, depth_img)
                    key = cv2.waitKey(1) & 0xFF
                except cv2.error as exc:
                    if not depth_gui_error_reported:
                        print(
                            "[WARN] OpenCV HighGUI backend unavailable; disabling live depth window. "
                            f"Details: {exc}"
                        )
                        print(
                            "[WARN] Continue play without depth window, or install GUI-enabled OpenCV. "
                            "You can also pass --no_depth_vis explicitly."
                        )
                        depth_gui_error_reported = True
                    depth_vis_enabled = False
                    key = 255
                if key in (27, ord("q")):
                    print("[INFO] Closing play loop because depth window requested exit (ESC/q).")
                    break

                if args_cli.depth_print_interval > 0 and (
                    timestep == 1 or timestep % args_cli.depth_print_interval == 0
                ):
                    print(
                        f"[Depth][step={timestep}] source={source} "
                        f"shape={tuple(depth_frame.shape)} "
                        f"min={depth_frame.min().item():.4f} "
                        f"max={depth_frame.max().item():.4f}"
                    )

        if args_cli.video:
            # Exit the play loop after recording one video
            if timestep == args_cli.video_length:
                break
        if args_cli.max_steps is not None and timestep >= args_cli.max_steps:
            break

        if args_cli.camera_follow_mode != "none":
            camera_follow(
                env,
                mode=args_cli.camera_follow_mode,
                window_size=args_cli.camera_smooth_window,
                env_index=0,
            )

        # time delay for real-time evaluation
        sleep_time = dt - (time.time() - start_time)
        if args_cli.real_time and sleep_time > 0:
            time.sleep(sleep_time)

    # close the simulator
    if keyboard_velocity_visualizer is not None:
        keyboard_velocity_visualizer.set_visibility(False)
    env.close()
    if depth_vis_enabled:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
