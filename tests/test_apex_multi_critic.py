"""Grouped PPO contracts exercised without launching Isaac Sim."""

from __future__ import annotations

import ast
import copy
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "apex_multi_critic_tests", ROOT / "scripts/reinforcement_learning/rsl_rl/multi_critic.py"
)
mc = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(mc)


class ToyEnv:
    cfg = {}
    device = "cpu"
    max_episode_length = 3
    num_actions = 2

    def __init__(self, num_envs=4, groups=2, num_actions=2):
        self.num_envs, self.groups = num_envs, groups
        self.num_actions = num_actions
        self.episode_length_buf = torch.zeros(num_envs, dtype=torch.long)
        self.state = torch.zeros(num_envs, 3)
        self.steps = 0

    def get_observations(self):
        return TensorDict({"policy": self.state.clone(), "critic": self.state.clone()}, [self.num_envs])

    def step(self, actions):
        self.steps += 1
        self.state[:, :2] += actions[:, :2] * 0.05
        self.episode_length_buf += 1
        terminal = self.get_observations()
        done = self.episode_length_buf >= self.max_episode_length
        rewards = torch.stack([-(self.state[:, 0] - i).square() for i in range(self.groups)], -1)
        self.state[done] = 0
        self.episode_length_buf[done] = 0
        return (
            self.get_observations(),
            rewards.sum(-1),
            done,
            {
                "reward_groups": rewards,
                "terminal_observations": terminal,
                "time_outs": done,
            },
        )

    def training_state_dict(self):
        return {"steps": self.steps}

    def load_training_state_dict(self, state):
        self.steps = state["steps"]


def runner(
    tmp_path=None,
    *,
    groups=2,
    num_envs=4,
    num_steps=4,
    recurrent=False,
    rnn_type="lstm",
    minibatch_normalization=False,
    num_actions=2,
):
    env = ToyEnv(num_envs, groups, num_actions)
    cfg = {
        "num_steps_per_env": num_steps,
        "save_interval": 1,
        "logger": "tensorboard",
        "obs_groups": {"policy": ["policy"], "critic": ["critic"]},
        "policy": {"actor_hidden_dims": [8], "critic_hidden_dims": [8]},
        "algorithm": {
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "schedule": "fixed",
            "normalize_advantage_per_mini_batch": minibatch_normalization,
        },
        "multi_critic_groups": [["critic"]] * groups,
        "multi_critic_recurrent": recurrent,
        "multi_critic_rnn_type": rnn_type,
        "multi_critic_rnn_hidden_dim": 8,
    }
    return mc.MultiCriticRunner(env, cfg, str(tmp_path) if tmp_path else None, "cpu")


@pytest.mark.parametrize("groups", [1, 2, 3])
@pytest.mark.parametrize("model", ["mlp", "lstm", "gru"])
def test_complete_rollout_updates_all_heads_and_actor(groups, model):
    torch.manual_seed(21)
    r = runner(groups=groups, recurrent=model != "mlp", rnn_type=model if model != "mlp" else "lstm")
    before_actor = copy.deepcopy(r.alg.actor.state_dict())
    before_critic = copy.deepcopy(r.alg.critic.state_dict())
    r.learn(2)
    assert r.current_learning_iteration == 2
    assert any(not torch.equal(value, before_actor[key]) for key, value in r.alg.actor.state_dict().items())
    for index in range(groups):
        assert any(
            not torch.equal(value, before_critic[key])
            for key, value in r.alg.critic.state_dict().items()
            if key.startswith(f"heads.{index}.")
        )
    assert all(torch.isfinite(p).all() for p in r.alg.actor.parameters())
    assert all(torch.isfinite(p).all() for p in r.alg.critic.parameters())


@pytest.mark.parametrize("recurrent", [False, True])
def test_timeout_uses_terminal_value_for_every_head_and_preserves_hidden_state(recurrent):
    r = runner(groups=3, recurrent=recurrent)
    alg = r.alg
    obs = r.env.get_observations()
    terminal = obs.clone()
    terminal["critic"].fill_(25)
    terminal["critic"][1:] = float("nan")
    with torch.inference_mode():
        alg.act(obs)
        hidden = copy.deepcopy(alg.critic.get_hidden_state())
        value = alg.critic(terminal)
        alg.critic.reset(hidden_state=hidden)
        rewards = torch.arange(12.0).reshape(4, 3)
        mask = torch.tensor([True, False, False, False])
        alg.process_env_step(obs, rewards, mask, {"terminal_observations": terminal, "time_outs": mask})
    torch.testing.assert_close(alg.storage.rewards[0], rewards + alg.gamma * torch.where(mask[:, None], value, 0.0))
    if recurrent:
        for actual, old in zip(alg.critic.get_hidden_state(), hidden, strict=True):
            torch.testing.assert_close(actual[:, 1:], old[:, 1:])
            assert torch.count_nonzero(actual[:, :1]) == 0


def test_missing_terminal_observations_and_scalar_reward_fallback_fail():
    r = runner()
    alg = r.alg
    obs = r.env.get_observations()
    with torch.inference_mode():
        alg.act(obs)
        with pytest.raises(ValueError, match="terminal_observations"):
            alg.process_env_step(obs, torch.ones(4, 2), torch.ones(4, dtype=torch.bool), {"time_outs": torch.ones(4)})
    with pytest.raises(ValueError, match="explicit reward_groups"):
        alg._extract_reward_groups(torch.ones(4), {})


@pytest.mark.parametrize("normalize", [False, True])
def test_singleton_rollout_is_finite(normalize):
    r = runner(num_envs=1, num_steps=1, minibatch_normalization=normalize)
    r.learn(1)
    assert all(torch.isfinite(p).all() for p in r.alg.actor.parameters())


@pytest.mark.parametrize("normalize", [False, True])
def test_heads_normalize_before_mixing_even_with_minibatch_option(normalize):
    r = runner(num_envs=2, num_steps=1, minibatch_normalization=normalize)
    alg = r.alg
    alg.storage.rewards[0] = torch.tensor([[100.0, -1.0], [-100.0, 1.0]])
    alg.storage.dones.fill_(1)
    with torch.no_grad():
        alg.compute_returns(r.env.get_observations())
    torch.testing.assert_close(alg.storage.advantages, torch.zeros(1, 2, 1))


def test_nondivisible_batches_are_rejected():
    r = runner(num_envs=5, num_steps=1)
    with pytest.raises(ValueError, match="divide evenly"):
        next(r.alg.storage.mini_batch_generator(2))
    with pytest.raises(ValueError, match="divide evenly"):
        next(r.alg.storage.recurrent_mini_batch_generator(2))


def test_resume_completed_updates_learning_rate_curriculum_and_rng(tmp_path):
    r = runner(tmp_path)
    r.alg.learning_rate = 1e-4
    r.alg.optimizer.param_groups[0]["lr"] = 1e-4
    r.learn(2)
    expected_random = torch.rand(5)
    state = torch.load(tmp_path / "model_2.pt", weights_only=False)
    assert state["iter"] == 2 and state["environment"]["steps"] == 8
    restored = runner(tmp_path / "resume")
    restored.load(tmp_path / "model_2.pt")
    assert restored.alg.learning_rate == restored.alg.optimizer.param_groups[0]["lr"] == 1e-4
    assert restored.env.steps == 8
    torch.testing.assert_close(torch.rand(5), expected_random)
    restored.learn(1)
    assert (tmp_path / "resume/model_3.pt").exists()
    assert restored.env.steps == 12
    assert not list(tmp_path.rglob("*.tmp"))
    restored.cfg["multi_critic_advantage_weights"] = [0.1, 0.9]
    with pytest.raises(ValueError, match="ordered reward partition"):
        restored.load(tmp_path / "model_2.pt")


@pytest.mark.parametrize("model", ["mlp", "lstm", "gru"])
def test_native_jit_export_matches_actor(tmp_path, model):
    r = runner(recurrent=model != "mlp", rnn_type=model if model != "mlp" else "lstm")
    r.alg.eval_mode()
    r.export_policy_to_jit(str(tmp_path))
    policy = torch.jit.load(str(tmp_path / "policy.pt"))
    obs = r.env.get_observations()[:1]
    with torch.no_grad():
        torch.testing.assert_close(policy(obs["policy"]), r.alg.actor(obs))


@pytest.mark.parametrize("model", ["mlp", "lstm", "gru"])
def test_native_onnx_export_matches_actor(tmp_path, model):
    ort = pytest.importorskip("onnxruntime")
    r = runner(recurrent=model != "mlp", rnn_type=model if model != "mlp" else "lstm")
    r.export_policy_to_onnx(str(tmp_path))
    session = ort.InferenceSession(str(tmp_path / "policy.onnx"), providers=["CPUExecutionProvider"])
    x = np.array([[0.2, -0.1, 0.3]], dtype=np.float32)
    feed = {i.name: np.zeros(i.shape, dtype=np.float32) for i in session.get_inputs()}
    feed[session.get_inputs()[0].name] = x
    actual = session.run(None, feed)[0]
    with torch.no_grad():
        expected = r.alg.actor(TensorDict({"policy": torch.from_numpy(x)}, [1])).numpy()
    np.testing.assert_allclose(actual, expected, atol=1e-6)


def test_checkpoint_keeps_distinct_rank_curriculum_and_rng(tmp_path, monkeypatch):
    # Exercise serialization/selection on CPU; GPU collective transport needs a GPU smoke run.
    pending = {}

    def gather(local, received, dst):
        if received is None:
            pending[1] = local
        else:
            received[:] = [local, pending[1]]

    monkeypatch.setattr(torch.distributed, "gather_object", gather)
    workers = [runner(), runner()]
    expected = {}
    for rank in (1, 0):
        worker = workers[rank]
        worker.is_distributed, worker.gpu_world_size, worker.gpu_global_rank = True, 2, rank
        worker.env.steps = rank + 10
        worker.logger.init_logging_writer()
        torch.manual_seed(100 + rank)
        worker.save(tmp_path / "distributed.pt")
        expected[rank] = torch.rand(5)
    for rank, worker in enumerate(workers):
        worker.env.steps = 0
        worker.load(tmp_path / "distributed.pt")
        assert worker.env.steps == rank + 10
        torch.testing.assert_close(torch.rand(5), expected[rank])


def test_registered_go2_trackers_expose_multi_critic_without_new_task_ids():
    path = ROOT / "source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/__init__.py"
    calls = [
        n.value for n in ast.parse(path.read_text()).body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)
    ]
    registrations = {next(k.value.value for k in c.keywords if k.arg == "id"): c for c in calls}
    for task in (
        "Flat",
        "Flat-Tracker",
        "Flat-Privileged-Tracker",
        "Flat-Tracker-One-Step-Future",
        "Flat-Tracker-One-Step-Future-History",
    ):
        call = registrations[f"Gurukul-Isaac-Go2-APEX-{task}-v0"]
        kwargs = next(k.value for k in call.keywords if k.arg == "kwargs")
        assert "rsl_rl_multi_critic_cfg_entry_point" in [k.value for k in kwargs.keys]
        assert "ApexManagerBasedRLEnv" in next(k.value.value for k in call.keywords if k.arg == "entry_point")


@pytest.mark.parametrize("motion_kind", ["quadruped", "g1"])
def test_environment_captures_before_reset_without_mutating_surviving_histories(motion_kind):
    path = ROOT / "source/Gurukul/Gurukul/tasks/manager_based/go2_apex/env.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))

    class Base:
        def _reset_idx(self, ids):
            self.state[ids] = 999

    ns = {"ManagerBasedRLEnv": Base, "torch": torch, "copy": copy, "TensorDict": TensorDict}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    env = ns["ApexManagerBasedRLEnv"]()
    env.num_envs, env.step_dt, env._in_policy_step = 3, 0.02, True
    env.state, env.extras = torch.tensor([[3.0], [4.0], [5.0]]), {}
    env.reset_time_outs = torch.tensor([True, True, False])
    env.reset_terminated = torch.tensor([False, True, False])
    env.enable_terminal_observations(["critic"])
    motion = SimpleNamespace(
        time_steps=torch.ones(3, dtype=torch.long),
        time_step_ends=torch.full((3,), 9),
        _motion_frame_accumulator=torch.zeros(3),
        motion=SimpleNamespace(fps_values=torch.tensor([50.0])),
        current_motion_ids=torch.zeros(3, dtype=torch.long),
        _refresh_relative_motion_state=lambda: None,
    )
    if motion_kind == "g1":
        del motion._motion_frame_accumulator
        motion.motion.time_step_total = 10
    env.command_manager = SimpleNamespace(get_term=lambda _: motion)
    manager = SimpleNamespace(_group_obs_term_history_buffer={"critic": [0]})

    def observe(name, update_history):
        manager._group_obs_term_history_buffer[name].append(1)
        return env.state + motion.time_steps[:, None]

    manager.compute_group = observe
    env.observation_manager = manager
    env._reset_idx(torch.tensor([0, 1]))
    torch.testing.assert_close(env.extras["terminal_observations"]["critic"], torch.tensor([[5.0], [6.0], [7.0]]))
    assert env.extras["terminal_time_outs"].tolist() == [True, False, False]
    assert env.state[0].item() == 999
    assert manager._group_obs_term_history_buffer["critic"] == [0]
    assert motion.time_steps.tolist() == [1, 1, 1]


@pytest.mark.parametrize("use_decap", [False, True])
def test_environment_resume_restores_curriculum_before_reset_without_doubling_decap(use_decap):
    path = ROOT / "source/Gurukul/Gurukul/tasks/manager_based/go2_apex/env.py"
    node = next(n for n in ast.parse(path.read_text()).body if isinstance(n, ast.ClassDef))

    class Base:
        def reset(self):
            self.bins_seen_at_reset = motion.bin_failed_count.clone()

    ns = {"ManagerBasedRLEnv": Base, "torch": torch, "copy": copy, "TensorDict": TensorDict}
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), ns)
    env = ns["ApexManagerBasedRLEnv"]()
    env.device, env.common_step_counter = "cpu", 0
    motion = SimpleNamespace(bin_failed_count=torch.zeros(3), motion=SimpleNamespace(motion_files=["clip.npz"]))
    action = SimpleNamespace()
    if use_decap:
        action._decap_resume_step_offset = 240
        action._current_decap_step = lambda: env.common_step_counter + action._decap_resume_step_offset
    env.command_manager = SimpleNamespace(get_term=lambda _: motion)
    env.action_manager = SimpleNamespace(get_term=lambda _: action)
    grasp_cfg = SimpleNamespace(params={"resume_iteration": 10, "steps_per_iteration": 24})
    env.termination_manager = SimpleNamespace(active_terms=["grasp"], get_term_cfg=lambda _: grasp_cfg)
    state = {
        "common_step_counter": 240,
        "decap_step": 264 if use_decap else None,
        "motion_files": ["clip.npz"],
        "bin_failed_count": torch.tensor([1.0, 5.0, 2.0]),
        "termination_resume_offsets": {"grasp": 0},
    }
    env.load_training_state_dict(state)
    if use_decap:
        assert action._current_decap_step() == 264
    assert grasp_cfg.params["resume_iteration"] + env.common_step_counter / 24 == 10
    torch.testing.assert_close(env.bins_seen_at_reset, state["bin_failed_count"])
    saved = env.training_state_dict()
    assert saved["decap_step"] == (264 if use_decap else None) and saved["common_step_counter"] == 240
    assert saved["termination_resume_offsets"] == {"grasp": 0}
    state["motion_files"] = ["another.npz"]
    with pytest.raises(ValueError, match="ordered motion files"):
        env.load_training_state_dict(state)


@pytest.mark.parametrize("num_actions", [18, 19])
@pytest.mark.parametrize("recurrent", [False, True])
def test_arm_action_dimensions_rollout_update_and_resume(tmp_path, num_actions, recurrent):
    r = runner(tmp_path, num_actions=num_actions, recurrent=recurrent)
    r.learn(1)
    assert r.alg.storage.actions.shape == (4, 4, num_actions)
    restored = runner(num_actions=num_actions, recurrent=recurrent)
    restored.load(tmp_path / "model_1.pt")
    restored.learn(1)
    assert restored.current_learning_iteration == 2
    with torch.inference_mode():
        action = restored.alg.actor(restored.env.get_observations())
    assert action.shape == (4, num_actions) and torch.isfinite(action).all()
