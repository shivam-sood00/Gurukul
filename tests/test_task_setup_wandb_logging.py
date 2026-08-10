from __future__ import annotations

import ast
import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

REPO_ROOT = Path(__file__).resolve().parents[1]
LOGGER_PATH = REPO_ROOT / "scripts/reinforcement_learning/rsl_rl/task_setup_logger.py"
TRAIN_PATH = REPO_ROOT / "scripts/reinforcement_learning/rsl_rl/train.py"


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Assignment {name!r} not found in {path}")


def _load_logger_module():
    spec = importlib.util.spec_from_file_location("task_setup_logger_test", LOGGER_PATH)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _observation(env):
    return env


def _reward(env):
    return env


def test_task_setup_records_actor_critic_terms_dimensions_and_active_rewards(tmp_path):
    module = _load_logger_module()
    policy_cfgs = [
        SimpleNamespace(
            func=_observation,
            scale=0.25,
            clip=(-100.0, 100.0),
            history_length=0,
            params={"source": "imu"},
        ),
        SimpleNamespace(
            func=_observation,
            scale=1.0,
            clip=(-100.0, 100.0),
            history_length=0,
            params={"joint_names": ["arm_1_joint"]},
        ),
    ]
    critic_cfgs = [
        SimpleNamespace(
            func=_observation,
            scale=1.0,
            clip=None,
            history_length=0,
            params={"privileged": True},
        )
    ]
    observation_manager = SimpleNamespace(
        active_terms={"policy": ["base_ang_vel", "joint_pos"], "critic": ["privileged_state"]},
        group_obs_term_dim={"policy": [(3,), (19,)], "critic": [(64,)]},
        group_obs_dim={"policy": (22,), "critic": (64,)},
        group_obs_concatenate={"policy": True, "critic": True},
        _group_obs_term_cfgs={"policy": policy_cfgs, "critic": critic_cfgs},
    )
    reward_manager = SimpleNamespace(
        active_terms=["tracking", "disabled_penalty"],
        _term_cfgs=[
            SimpleNamespace(func=_reward, weight=3.0, params={"std": 0.25}),
            SimpleNamespace(func=_reward, weight=0.0, params={}),
        ],
    )
    env = SimpleNamespace(observation_manager=observation_manager, reward_manager=reward_manager)

    schema = module.collect_task_setup(
        env,
        {"obs_groups": {"policy": ["policy"], "critic": ["critic"]}},
    )

    assert schema["actor_observation_groups"] == ["policy"]
    assert schema["critic_observation_groups"] == ["critic"]
    assert schema["observations"]["groups"]["policy"]["size"] == 22
    assert schema["observations"]["groups"]["policy"]["terms"][1]["name"] == "joint_pos"
    assert schema["observations"]["groups"]["critic"]["shape"] == [64]
    assert schema["rewards"]["active_rewards"] == ["tracking"]
    assert schema["rewards"]["terms"][1]["enabled"] is False

    output_path = Path(module.write_task_setup(str(tmp_path), schema))
    assert output_path == tmp_path / "params/task_setup.json"
    assert json.loads(output_path.read_text()) == schema


def test_task_setup_prefers_explicit_actor_group_for_privileged_teachers():
    module = _load_logger_module()
    env = SimpleNamespace(observation_manager=None, reward_manager=None)

    schema = module.collect_task_setup(
        env,
        {"obs_groups": {"actor": ["teacher"], "critic": ["teacher"]}},
    )

    assert schema["actor_observation_groups"] == ["teacher"]
    assert schema["critic_observation_groups"] == ["teacher"]


def test_task_setup_records_resolved_motion_dataset_manifest(tmp_path):
    module = _load_logger_module()
    motion_a = tmp_path / "walk.npz"
    motion_b = tmp_path / "jump.npz"
    motion_a.write_bytes(b"walk-motion")
    motion_b.write_bytes(b"jump-motion")
    motion = SimpleNamespace(
        motion_files=[str(motion_a), str(motion_b)],
        motion_names=["walk.npz", "jump.npz"],
        motion_lengths=[300, 150],
        fps_values=[30.0, 30.0],
        joint_names=["FL_hip_joint", "FR_hip_joint"],
        body_names=["base"],
    )
    command = SimpleNamespace(
        cfg=SimpleNamespace(motion_files=(), motion_file=str(tmp_path)),
        motion=motion,
    )
    env = SimpleNamespace(
        observation_manager=None,
        reward_manager=None,
        command_manager=SimpleNamespace(get_term=lambda name: command if name == "motion" else None),
    )

    dataset = module.collect_task_setup(env, {})["motion_dataset"]

    assert dataset["configured_sources"] == [str(tmp_path)]
    assert dataset["resolved_file_count"] == 2
    assert dataset["total_frames"] == 450
    assert dataset["total_duration_seconds"] == 15.0
    assert dataset["files"][0]["sha256"] == hashlib.sha256(b"walk-motion").hexdigest()
    assert dataset["manifest_sha256"] is not None


def test_every_training_runner_receives_task_setup_in_wandb_train_config():
    train_source = TRAIN_PATH.read_text()

    assert "task_setup = collect_task_setup(env.unwrapped, runner_cfg)" in train_source
    assert 'runner_cfg["task_name"] = args_cli.task' in train_source
    assert 'runner_cfg["task_setup"] = task_setup' in train_source
    assert 'runner_cfg["task_setup_path"] = task_setup_path' in train_source
    assert "agent_cfg.to_dict(), log_dir=log_dir" not in train_source
    assert train_source.count("runner_cfg, log_dir=log_dir") == 5


def test_training_entrypoint_enforces_the_documented_rsl_rl_version():
    train_source = TRAIN_PATH.read_text()

    assert _literal_assignment(TRAIN_PATH, "RSL_RL_VERSION") == "5.3.0"
    assert "version.parse(installed_version) != version.parse(RSL_RL_VERSION)" in train_source
