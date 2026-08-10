# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Script to train RL agent with RSL-RL."""

"""Launch Isaac Sim Simulator first."""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

# local imports
import cli_args  # isort: skip

# add argparse arguments
parser = argparse.ArgumentParser(description="Train an RL agent with RSL-RL.")
parser.add_argument("--video", action="store_true", default=False, help="Record videos during training.")
parser.add_argument("--video_length", type=int, default=200, help="Length of the recorded video (in steps).")
parser.add_argument("--video_interval", type=int, default=2000, help="Interval between video recordings (in steps).")
parser.add_argument("--num_envs", type=int, default=None, help="Number of environments to simulate.")
parser.add_argument("--task", type=str, default=None, help="Name of the task.")
parser.add_argument(
    "--agent", type=str, default="rsl_rl_cfg_entry_point", help="Name of the RL agent configuration entry point."
)
parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
parser.add_argument("--max_iterations", type=int, default=None, help="RL Policy training iterations.")
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
    "--distributed", action="store_true", default=False, help="Run training with multiple GPUs or nodes."
)
parser.add_argument("--export_io_descriptors", action="store_true", default=False, help="Export IO descriptors.")
parser.add_argument(
    "--ray-proc-id", "-rid", type=int, default=None, help="Automatically configured by Ray integration, otherwise None."
)
parser.add_argument(
    "--live-viser",
    action="store_true",
    default=False,
    help="Stream one env to a Viser browser viewer while training (works with --headless).",
)
parser.add_argument(
    "--viser-host",
    type=str,
    default="0.0.0.0",
    help="Host for the live Viser server started by --live-viser.",
)
parser.add_argument(
    "--viser-port",
    type=int,
    default=8080,
    help="Port for the live Viser server started by --live-viser.",
)
parser.add_argument(
    "--viser-env-id",
    type=int,
    default=0,
    help="Initial env index shown in the live Viser viewer.",
)
parser.add_argument(
    "--viser-update-every",
    type=int,
    default=20,
    help="Publish a Viser frame every N simulation steps.",
)
parser.add_argument(
    "--viser-robot",
    type=str,
    default="robot",
    help="Scene asset name to visualize in the live Viser viewer.",
)
parser.add_argument(
    "--viser-urdf",
    type=str,
    default=None,
    help="Optional URDF override for the live Viser mesh viewer.",
)
# append RSL-RL cli arguments
cli_args.add_rsl_rl_args(parser)
# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
args_cli, hydra_args = parser.parse_known_args()

# always enable cameras to record video
if args_cli.video:
    args_cli.enable_cameras = True

# clear out sys.argv for Hydra
sys.argv = [sys.argv[0]] + hydra_args

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Check for the supported RSL-RL version."""

import importlib.metadata as metadata
import platform

from packaging import version

# Custom models and loggers in this workspace target the 5.3 API.  Do not accept
# newer releases implicitly: RSL-RL uses minor releases for breaking API changes.
RSL_RL_VERSION = "5.3.0"
installed_version = metadata.version("rsl-rl-lib")
if version.parse(installed_version) != version.parse(RSL_RL_VERSION):
    if platform.system() == "Windows":
        cmd = [r".\isaaclab.bat", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    else:
        cmd = ["./isaaclab.sh", "-p", "-m", "pip", "install", f"rsl-rl-lib=={RSL_RL_VERSION}"]
    print(
        f"Please install the correct version of RSL-RL.\nExisting version is: '{installed_version}'"
        f" and required version is: '{RSL_RL_VERSION}'.\nTo install the correct version, run:"
        f"\n\n\t{' '.join(cmd)}\n"
    )
    exit(1)

"""Rest everything follows."""

import logging
import re
import time
from datetime import datetime

import gymnasium as gym
import torch
from contact_trail_runner import ContactTrailOnPolicyRunner
from real_teacher_runner import RealTeacherOnPolicyRunner
from rsl_rl.runners import DistillationRunner, OnPolicyRunner

from isaaclab.envs import (
    DirectMARLEnv,
    DirectMARLEnvCfg,
    DirectRLEnvCfg,
    ManagerBasedRLEnvCfg,
    multi_agent_to_single_agent,
)
from isaaclab.utils.assets import retrieve_file_path
from isaaclab.utils.dict import print_dict
from isaaclab.utils.io import dump_yaml

from isaaclab_rl.rsl_rl import RslRlBaseRunnerCfg, RslRlVecEnvWrapper

from isaaclab_tasks.utils import get_checkpoint_path
from isaaclab_tasks.utils.hydra import hydra_task_config

# Allow running this script directly from the repo root without an editable install.
_GURUKUL_SOURCE_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "../../..", "source", "Gurukul")
)
if os.path.isdir(_GURUKUL_SOURCE_PATH) and _GURUKUL_SOURCE_PATH not in sys.path:
    sys.path.insert(0, _GURUKUL_SOURCE_PATH)

import Gurukul.tasks  # noqa: F401  # isort: skip
from grouped_logger import GroupedConsoleLogger  # isort: skip
from legacy_checkpoint import load_checkpoint_for_train  # isort: skip
from rsl_rl_config_compat import migrate_custom_policy_cfg  # isort: skip
from task_setup_logger import collect_task_setup, write_task_setup  # isort: skip
from Gurukul.utils.export_deploy_cfg import maybe_export_deploy_cfg  # isort: skip
from Gurukul.utils.live_env_viewer import LiveEnvViewer  # isort: skip
from Gurukul.utils.live_env_viewer_wrapper import LiveViserRslRlVecEnvWrapper  # isort: skip

# import logger
logger = logging.getLogger(__name__)

# PLACEHOLDER: Extension template (do not remove this comment)

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False


def _train_log(category: str, message: str, level: str = "INFO") -> None:
    """Print a consistent, scannable training setup message."""
    print(f"[{level}][{category:<11}] {message}")


def _install_grouped_console_logger() -> None:
    """Use deterministic grouped console metrics for RSL-RL runners."""
    import rsl_rl.runners.on_policy_runner as on_policy_runner_module

    on_policy_runner_module.Logger = GroupedConsoleLogger


def _resolve_motion_source_path(path: str) -> str:
    """Resolve local motion files/directories while preserving Isaac asset-path support."""
    expanded_path = os.path.expanduser(path)
    if os.path.exists(expanded_path):
        return expanded_path
    return retrieve_file_path(path)


def _resolve_checkpoint_file_path(path: str, checkpoint_pattern: str = r"model_.*\.pt") -> str:
    """Resolve a checkpoint file, or select the latest matching checkpoint from a run directory."""
    expanded_path = os.path.expanduser(path)
    resolved_path = expanded_path if os.path.exists(expanded_path) else retrieve_file_path(path)
    if not os.path.isdir(resolved_path):
        return resolved_path

    checkpoint_re = re.compile(checkpoint_pattern)
    candidates = []
    for root, _, files in os.walk(resolved_path):
        for filename in files:
            if checkpoint_re.fullmatch(filename):
                checkpoint_path = os.path.join(root, filename)
                candidates.append(
                    (_checkpoint_iteration(checkpoint_path), os.path.getmtime(checkpoint_path), checkpoint_path)
                )

    if len(candidates) == 0:
        raise FileNotFoundError(f"No checkpoint matching '{checkpoint_pattern}' found under directory: {resolved_path}")

    selected_path = max(candidates, key=lambda item: (item[0], item[1]))[2]
    _train_log("Checkpoints", f"Resolved checkpoint directory {resolved_path} to {selected_path}")
    return selected_path


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
    _train_log("SMP", f"Using frozen prior: {checkpoint_path}")
    if args_cli.smp_gsi_pool_size is not None:
        if args_cli.smp_gsi_pool_size < 0:
            raise ValueError("--smp-gsi-pool-size must be non-negative.")
        initialize_cfg.params["gsi_pool_size"] = args_cli.smp_gsi_pool_size
        _train_log("SMP", f"GSI pool size: {args_cli.smp_gsi_pool_size}")


def _resolve_teacher_checkpoint_path(args_cli, agent_cfg: RslRlBaseRunnerCfg) -> str:
    """Resolve the teacher checkpoint path from CLI overrides or agent defaults."""
    if args_cli.teacher_checkpoint is not None:
        return _resolve_checkpoint_file_path(args_cli.teacher_checkpoint)

    teacher_experiment_name = (
        args_cli.teacher_experiment_name
        if args_cli.teacher_experiment_name is not None
        else getattr(agent_cfg, "teacher_experiment_name", agent_cfg.experiment_name)
    )
    teacher_load_run = (
        args_cli.teacher_load_run
        if args_cli.teacher_load_run is not None
        else getattr(agent_cfg, "teacher_load_run", agent_cfg.load_run)
    )
    teacher_load_checkpoint = (
        args_cli.teacher_load_checkpoint
        if args_cli.teacher_load_checkpoint is not None
        else getattr(agent_cfg, "teacher_load_checkpoint", agent_cfg.load_checkpoint)
    )
    teacher_log_root_path = os.path.join("logs", "rsl_rl", teacher_experiment_name)
    teacher_log_root_path = os.path.abspath(teacher_log_root_path)
    return get_checkpoint_path(teacher_log_root_path, teacher_load_run, teacher_load_checkpoint)


def _print_decap_training_config(env_cfg, agent_cfg) -> None:
    """Print active DecAP settings when either the runner or env action uses DecAP."""
    action_cfg = getattr(getattr(env_cfg, "actions", None), "joint_pos", None)
    has_env_decap = hasattr(action_cfg, "decap_decay_type")
    has_runner_decap = getattr(agent_cfg, "class_name", None) == "DecAPRunner" or hasattr(agent_cfg, "decap_decay_type")
    if not has_env_decap and not has_runner_decap:
        return

    parts = []
    if has_runner_decap:
        parts.append(
            "runner="
            f"{getattr(agent_cfg, 'class_name', 'unknown')}"
            f", type={getattr(agent_cfg, 'decap_decay_type', 'n/a')}"
            f", lambda={getattr(agent_cfg, 'decap_lambda_start', 'n/a')}"
            f"->{getattr(agent_cfg, 'decap_lambda_end', 'n/a')}"
        )
    if has_env_decap:
        parts.append(
            "env_action="
            f"{type(action_cfg).__name__}"
            f", type={action_cfg.decap_decay_type}"
            f", lambda={action_cfg.decap_lambda_start}->{action_cfg.decap_lambda_end}"
            ", application=additive_error"
            f", prior_only={getattr(action_cfg, 'decap_prior_only', False)}"
        )
    _train_log("DecAP", f"Training config: {' | '.join(parts)}")


def _force_env_decap_action_to_zero(env_cfg) -> bool:
    """Force env-side DecAP action prior off when the task exposes it."""
    if hasattr(env_cfg, "force_zero_decap"):
        env_cfg.force_zero_decap()
        return True

    action_cfg = getattr(getattr(env_cfg, "actions", None), "joint_pos", None)
    if not hasattr(action_cfg, "decap_lambda_start"):
        return False

    action_cfg.decap_prior_only = False
    action_cfg.decap_lambda_start = 0.0
    action_cfg.decap_lambda_end = 0.0
    action_cfg.decap_decay_type = "constant"
    if hasattr(action_cfg, "decap_resume_iteration"):
        action_cfg.decap_resume_iteration = 0
    action_cfg.decap_warmup_steps = 0
    action_cfg.decap_decay_start_step = 0
    action_cfg.decap_decay_end_step = 1
    return True


def _configure_env_decap_resume(env_cfg, resume_iteration: int) -> None:
    """Resume env-side DecAP action-prior schedules from the student checkpoint iteration."""
    if resume_iteration <= 0:
        return
    action_cfg = getattr(getattr(env_cfg, "actions", None), "joint_pos", None)
    if not hasattr(action_cfg, "decap_resume_iteration"):
        return

    action_cfg.decap_resume_iteration = int(resume_iteration)
    steps_per_iteration = getattr(action_cfg, "decap_steps_per_iteration", "n/a")
    _train_log(
        "DecAP",
        f"Env action schedule resume_iteration={resume_iteration}, steps_per_iteration={steps_per_iteration}.",
    )


def _configure_env_decap_iteration_scale(env_cfg, agent_cfg) -> None:
    """Keep env-side DecAP schedules aligned with the active RSL-RL rollout length."""
    action_cfg = getattr(getattr(env_cfg, "actions", None), "joint_pos", None)
    if not hasattr(action_cfg, "decap_steps_per_iteration"):
        return

    steps_per_iteration = int(getattr(agent_cfg, "num_steps_per_env", 0) or 0)
    if steps_per_iteration <= 0:
        return

    previous = int(getattr(action_cfg, "decap_steps_per_iteration", steps_per_iteration))
    action_cfg.decap_steps_per_iteration = steps_per_iteration
    if previous != steps_per_iteration:
        _train_log(
            "DecAP",
            f"Env action schedule num_steps_per_env={steps_per_iteration} (was decap_steps_per_iteration={previous}).",
        )


def _force_runner_decap_to_zero(agent_cfg) -> bool:
    """Force runner-side DecAP off when a DecAP runner config is selected."""
    if not hasattr(agent_cfg, "decap_lambda_start"):
        return False

    agent_cfg.decap_lambda_start = 0.0
    agent_cfg.decap_lambda_end = 0.0
    if hasattr(agent_cfg, "decap_adaptive_decay"):
        agent_cfg.decap_adaptive_decay = False
    return True


def _force_decap_to_zero(env_cfg, agent_cfg, reason: str) -> None:
    if hasattr(env_cfg, "train_with_zero_decap"):
        env_cfg.train_with_zero_decap = True

    changed_env = _force_env_decap_action_to_zero(env_cfg)
    changed_runner = _force_runner_decap_to_zero(agent_cfg)
    if changed_env or changed_runner:
        targets = []
        if changed_env:
            targets.append("env action")
        if changed_runner:
            targets.append("runner")
        _train_log("DecAP", f"{reason}; forced {' and '.join(targets)} DecAP lambda to 0.0.")
    else:
        _train_log("DecAP", f"{reason}; no DecAP config was present to modify.", level="WARN")


def _add_marl_single_agent_observation_shim(env) -> None:
    """Add hooks expected by RslRlVecEnvWrapper to Isaac Lab's MARL converter."""

    def _get_observations():
        obs = env.env._get_observations()
        return {
            "policy": torch.cat(
                [obs[agent].reshape(env.num_envs, -1) for agent in env.env.possible_agents],
                dim=-1,
            )
        }

    env._get_observations = _get_observations
    type(env).episode_length_buf = property(
        lambda self: self.env.episode_length_buf,
        lambda self, value: setattr(self.env, "episode_length_buf", value),
    )


def _checkpoint_iteration(checkpoint_path: str | None) -> int:
    """Best-effort extraction of the RSL-RL iteration from model_<iter>.pt checkpoint names."""
    if not checkpoint_path:
        return 0
    match = re.search(r"model_(\d+)\.pt$", os.path.basename(checkpoint_path))
    return int(match.group(1)) if match else 0


def _configure_iteration_scaled_curricula(env_cfg, agent_cfg, resume_iteration: int = 0) -> None:
    """Inject resolved RSL-RL iteration counts into curricula that use fractional iteration bins."""
    curriculum_cfg = getattr(env_cfg, "curriculum", None)
    if curriculum_cfg is None:
        return

    max_iterations = int(getattr(agent_cfg, "max_iterations", 0) or 0)
    steps_per_iteration = int(getattr(agent_cfg, "num_steps_per_env", 0) or 0)
    if max_iterations <= 0 or steps_per_iteration <= 0:
        return
    total_iterations = max_iterations + max(0, int(resume_iteration))

    for curriculum_name, term_cfg in vars(curriculum_cfg).items():
        params = getattr(term_cfg, "params", None)
        if not isinstance(params, dict):
            continue
        iteration_bin_keys = (
            "stage_iteration_bins",
            "arm_difficulty_iteration_bins",
            "distance_iteration_bins",
        )
        if not any(key in params for key in iteration_bin_keys):
            continue

        params["max_iterations"] = total_iterations
        params["steps_per_iteration"] = steps_per_iteration
        params["resume_iteration"] = max(0, int(resume_iteration))
        stage_bins = params.get("stage_iteration_bins")
        difficulty_bins = params.get("arm_difficulty_iteration_bins")
        distance_bins = params.get("distance_iteration_bins")
        total_steps = total_iterations * steps_per_iteration
        _train_log(
            "Curriculum",
            f"{curriculum_name}: scaled bins with "
            f"max_iterations={total_iterations}, resume_iteration={resume_iteration}, "
            f"num_steps_per_env={steps_per_iteration}, "
            f"total_env_steps={total_steps}, stage_bins={stage_bins}, arm_difficulty_bins={difficulty_bins}, "
            f"distance_bins={distance_bins}.",
        )

    for curriculum_name, term_cfg in vars(curriculum_cfg).items():
        params = getattr(term_cfg, "params", None)
        if not isinstance(params, dict):
            continue
        if "start_iteration" in params and "end_iteration" in params:
            params["steps_per_iteration"] = steps_per_iteration
            _train_log(
                "Curriculum",
                f"{curriculum_name}: scaled iteration curriculum with "
                f"num_steps_per_env={steps_per_iteration}, start_iteration={params['start_iteration']}, "
                f"end_iteration={params['end_iteration']}.",
            )

    termination_cfg = getattr(env_cfg, "terminations", None)
    if termination_cfg is None:
        return
    for termination_name, term_cfg in vars(termination_cfg).items():
        params = getattr(term_cfg, "params", None)
        if not isinstance(params, dict):
            continue
        if "warmup_iterations" not in params or "ramp_end_iteration" not in params:
            continue
        params["steps_per_iteration"] = steps_per_iteration
        params["resume_iteration"] = max(0, int(resume_iteration))
        _train_log(
            "Curriculum",
            f"{termination_name}: resume-aware termination schedule with "
            f"num_steps_per_env={steps_per_iteration}, resume_iteration={resume_iteration}, "
            f"warmup_iterations={params['warmup_iterations']}, "
            f"ramp_end_iteration={params['ramp_end_iteration']}.",
        )


@hydra_task_config(args_cli.task, args_cli.agent)
def main(  # noqa: C901
    env_cfg: ManagerBasedRLEnvCfg | DirectRLEnvCfg | DirectMARLEnvCfg, agent_cfg: RslRlBaseRunnerCfg
):
    """Train with RSL-RL agent."""
    # override configurations with non-hydra CLI arguments
    agent_cfg = cli_args.update_rsl_rl_cfg(agent_cfg, args_cli)
    env_cfg.scene.num_envs = args_cli.num_envs if args_cli.num_envs is not None else env_cfg.scene.num_envs
    agent_cfg.max_iterations = (
        args_cli.max_iterations if args_cli.max_iterations is not None else agent_cfg.max_iterations
    )

    policy_class_name = getattr(getattr(agent_cfg, "policy", None), "class_name", "")

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
            _train_log("Motion", f"Using motion file override: {env_cfg.commands.motion.motion_file}")
        else:
            logger.warning("--motion-file provided but this task has no commands.motion term. Ignoring override.")
    elif hasattr(env_cfg, "commands") and hasattr(env_cfg.commands, "motion"):
        _train_log("Motion", f"Using default env motion source: {_describe_motion_source(env_cfg.commands.motion)}")
    _configure_smp_prior(env_cfg)
    if getattr(args_cli, "zero_decap", False):
        _force_decap_to_zero(env_cfg, agent_cfg, "--zero_decap set")
    elif getattr(env_cfg, "train_with_zero_decap", False):
        _force_decap_to_zero(env_cfg, agent_cfg, "env_cfg.train_with_zero_decap enabled")
    elif agent_cfg.class_name == "DistillationRunner":
        _force_decap_to_zero(env_cfg, agent_cfg, "RSL-RL supervised distillation selected")
    # Prevent double prior injection when using policy-side DecAPRunner with env-side DecAP action prior.
    if agent_cfg.class_name == "DecAPRunner":
        if hasattr(env_cfg, "actions") and hasattr(env_cfg.actions, "joint_pos"):
            action_cfg = env_cfg.actions.joint_pos
            if hasattr(action_cfg, "decap_lambda_start"):
                _force_env_decap_action_to_zero(env_cfg)
                _train_log(
                    "DecAP",
                    "DecAPRunner selected; disabled env-side DecAP prior to avoid double prior injection.",
                )
    _configure_env_decap_iteration_scale(env_cfg, agent_cfg)
    _print_decap_training_config(env_cfg, agent_cfg)
    # check for invalid combination of CPU device with distributed training
    if args_cli.distributed and args_cli.device is not None and "cpu" in args_cli.device:
        raise ValueError(
            "Distributed training is not supported when using CPU device. "
            "Please use GPU device (e.g., --device cuda) for distributed training."
        )

    # multi-gpu training configuration
    if args_cli.distributed:
        env_cfg.sim.device = f"cuda:{app_launcher.local_rank}"
        agent_cfg.device = f"cuda:{app_launcher.local_rank}"

        # set seed to have diversity in different threads
        seed = agent_cfg.seed + app_launcher.local_rank
        env_cfg.seed = seed
        agent_cfg.seed = seed

    # specify directory for logging experiments
    log_root_path = os.path.join("logs", "rsl_rl", agent_cfg.experiment_name)
    log_root_path = os.path.abspath(log_root_path)
    _train_log("Logging", f"Experiment root: {log_root_path}")
    # specify directory for logging runs: {time-stamp}_{run_name}
    log_dir = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # The Ray Tune workflow extracts experiment name using the logging line below, hence, do not
    # change it (see PR #2346, comment-2819298849)
    print(f"Exact experiment name requested from command line: {log_dir}")
    if agent_cfg.run_name:
        log_dir += f"_{agent_cfg.run_name}"
    log_dir = os.path.join(log_root_path, log_dir)

    # set the IO descriptors export flag if requested
    if isinstance(env_cfg, ManagerBasedRLEnvCfg):
        env_cfg.export_io_descriptors = args_cli.export_io_descriptors
    else:
        logger.warning(
            "IO descriptors are only supported for manager based RL environments. No IO descriptors will be exported."
        )

    # set the log directory for the environment (works for all environment types)
    env_cfg.log_dir = log_dir

    policy_cfg = getattr(agent_cfg, "policy", None)
    policy_class_name = getattr(policy_cfg, "class_name", "")

    # Save checkpoint paths before creating the environment so resume-aware curricula can be configured.
    resume_path = None
    teacher_resume_path = None
    if agent_cfg.class_name == "DecAPRunner":
        if agent_cfg.resume:
            if args_cli.checkpoint:
                resume_path = _resolve_checkpoint_file_path(args_cli.checkpoint)
            else:
                resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)

        if args_cli.teacher_checkpoint is not None:
            teacher_resume_path = _resolve_checkpoint_file_path(args_cli.teacher_checkpoint)
        elif args_cli.checkpoint and not agent_cfg.resume:
            teacher_resume_path = _resolve_checkpoint_file_path(args_cli.checkpoint)
        else:
            teacher_resume_path = _resolve_teacher_checkpoint_path(args_cli, agent_cfg)
    elif agent_cfg.resume:
        if args_cli.checkpoint:
            resume_path = _resolve_checkpoint_file_path(args_cli.checkpoint)
        else:
            resume_path = get_checkpoint_path(log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint)
    elif agent_cfg.class_name == "DistillationRunner":
        # Distillation commonly loads a teacher policy from a different experiment folder.
        if args_cli.teacher_checkpoint is not None:
            resume_path = _resolve_checkpoint_file_path(args_cli.teacher_checkpoint)
        elif args_cli.checkpoint:
            # Backward-compatible teacher checkpoint override.
            resume_path = _resolve_checkpoint_file_path(args_cli.checkpoint)
        else:
            resume_path = _resolve_teacher_checkpoint_path(args_cli, agent_cfg)

    resume_iteration = _checkpoint_iteration(resume_path if agent_cfg.resume else None)
    _configure_iteration_scaled_curricula(env_cfg, agent_cfg, resume_iteration=resume_iteration)
    _configure_env_decap_resume(env_cfg, resume_iteration=resume_iteration)

    # create isaac environment
    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array" if args_cli.video else None)
    live_viewer = None
    if args_cli.live_viser:
        if args_cli.distributed:
            raise ValueError("--live-viser is only supported for single-process training.")
        live_viewer = LiveEnvViewer(
            host=args_cli.viser_host,
            port=args_cli.viser_port,
            robot_name=args_cli.viser_robot,
            urdf_path=args_cli.viser_urdf,
            initial_env_id=args_cli.viser_env_id,
            update_every_steps=args_cli.viser_update_every,
        )
        _train_log(
            "LiveViser",
            (
                f"Browser viewer on {args_cli.viser_host}:{args_cli.viser_port} "
                f"(initial env_id={args_cli.viser_env_id}, update_every={args_cli.viser_update_every})"
            ),
        )

    # convert to single-agent instance if required by the RL algorithm
    if isinstance(env.unwrapped, DirectMARLEnv):
        env = multi_agent_to_single_agent(env)
        _add_marl_single_agent_observation_shim(env)

    # Resolve the actual manager terms after environment construction. Store the
    # schema in the runner configuration so external loggers (including W&B)
    # receive the actor/critic observations and active reward graph at startup.
    runner_cfg = agent_cfg.to_dict()
    runner_cfg["task_name"] = args_cli.task
    task_setup = collect_task_setup(env.unwrapped, runner_cfg)
    task_setup_path = write_task_setup(log_dir, task_setup)
    runner_cfg["task_setup"] = task_setup
    runner_cfg["task_setup_path"] = task_setup_path
    observation_groups = task_setup["observations"]["groups"]
    active_rewards = task_setup["rewards"]["active_rewards"]
    _train_log(
        "Task setup",
        (
            f"Logged {len(observation_groups)} observation groups and "
            f"{len(active_rewards)} active rewards to {task_setup_path}."
        ),
    )

    if (
        args_cli.checkpoint
        and not agent_cfg.resume
        and policy_class_name == "RealTeacherActorCritic"
        and getattr(policy_cfg, "pretrained_attention_checkpoint", None) in {None, ""}
    ):
        policy_cfg.pretrained_attention_checkpoint = retrieve_file_path(args_cli.checkpoint)
        _train_log(
            "Checkpoints",
            f"Using pretrained REAL attention checkpoint: {policy_cfg.pretrained_attention_checkpoint}",
        )

    # wrap for video recording
    if args_cli.video:
        video_kwargs = {
            "video_folder": os.path.join(log_dir, "videos", "train"),
            "step_trigger": lambda step: step % args_cli.video_interval == 0,
            "video_length": args_cli.video_length,
            "disable_logger": True,
        }
        _train_log("Video", "Recording videos during training.")
        print_dict(video_kwargs, nesting=4)
        env = gym.wrappers.RecordVideo(env, **video_kwargs)

    deploy_cfg_path = maybe_export_deploy_cfg(env.unwrapped, log_dir)
    if deploy_cfg_path is not None:
        _train_log("Artifacts", f"Exported deploy config to: {deploy_cfg_path}")

    start_time = time.time()

    # wrap around environment for rsl-rl
    if live_viewer is not None:
        env = LiveViserRslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions, viewer=live_viewer)
    else:
        env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

    # create runner from rsl-rl
    _install_grouped_console_logger()
    if agent_cfg.class_name == "OnPolicyRunner":
        policy_class_name = getattr(getattr(agent_cfg, "policy", None), "class_name", "")
        if policy_class_name == "RealTeacherActorCritic":
            runner_cls = RealTeacherOnPolicyRunner
        elif policy_class_name == "ContactTrailActorCritic":
            runner_cls = ContactTrailOnPolicyRunner
        else:
            runner_cls = OnPolicyRunner
        if runner_cls is RealTeacherOnPolicyRunner:
            _train_log("Runner", "Using RealTeacherOnPolicyRunner for REAL policy metric logging.")
        if runner_cls is ContactTrailOnPolicyRunner:
            _train_log("Runner", "Using ContactTrailOnPolicyRunner for contact trail metric logging.")
        runner = runner_cls(env, runner_cfg, log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "SmpOnPolicyRunner":
        from Gurukul.tasks.manager_based.smp.runner import SmpOnPolicyRunner

        runner = SmpOnPolicyRunner(env, runner_cfg, log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DistillationRunner":
        runner = DistillationRunner(env, runner_cfg, log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "DecAPRunner":
        from decap import DecAPRunner

        runner = DecAPRunner(env, runner_cfg, log_dir=log_dir, device=agent_cfg.device)
    elif agent_cfg.class_name == "MultiCriticRunner":
        from multi_critic import MultiCriticRunner

        runner = MultiCriticRunner(env, runner_cfg, log_dir=log_dir, device=agent_cfg.device)
    else:
        raise ValueError(f"Unsupported runner class: {agent_cfg.class_name}")
    # write git state to logs
    runner.add_git_repo_to_log(__file__)
    if agent_cfg.class_name == "DecAPRunner":
        _train_log("Checkpoints", f"Loading DecAP teacher checkpoint: {teacher_resume_path}")
        runner.load_teacher(teacher_resume_path)
    # load the checkpoint for student/resume policies
    if resume_path is not None:
        _train_log("Checkpoints", f"Loading model checkpoint: {resume_path}")
        # Load the previous model, including task-scoped observation-contract warm starts.
        load_checkpoint_for_train(runner, resume_path, args_cli.task)

    # dump the configuration into log-directory
    dump_yaml(os.path.join(log_dir, "params", "env.yaml"), env_cfg)
    dump_yaml(os.path.join(log_dir, "params", "agent.yaml"), agent_cfg)

    # run training
    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

    _train_log("Timing", f"Training time: {round(time.time() - start_time, 2)} seconds")

    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
