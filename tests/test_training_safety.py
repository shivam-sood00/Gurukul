from __future__ import annotations

import ast
import importlib.util
import types
from pathlib import Path

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
MOTION_LOADER_PATH = (
    REPO_ROOT / "source/Gurukul/Gurukul/tasks/direct/g1_amp/motions/motion_loader.py"
)
RESET_UTILS_PATH = REPO_ROOT / "source/Gurukul/Gurukul/tasks/direct/brainco_revo/reset_utils.py"
LEGACY_CHECKPOINT_PATH = REPO_ROOT / "scripts/reinforcement_learning/rsl_rl/legacy_checkpoint.py"
G1_AMP_ENV_PATH = REPO_ROOT / "source/Gurukul/Gurukul/tasks/direct/g1_amp/g1_amp_env.py"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_motion_frame_blend_clips_history_times_and_uses_adjacent_frames():
    motion_loader = _load_module("motion_loader_safety_test", MOTION_LOADER_PATH)
    loader = motion_loader.MotionLoader.__new__(motion_loader.MotionLoader)
    loader.dt = 0.5
    loader.num_frames = 5
    loader.duration = 2.0

    start, end, blend = loader._compute_frame_blend(np.array([-0.5, 0.375, 2.5]))

    assert start.tolist() == [0, 0, 4]
    assert end.tolist() == [1, 1, 4]
    assert blend.tolist() == [0.0, 0.75, 0.0]

    history = loader.history_times(np.array([0.0, 1.0]), history_length=3, step_dt=0.25)
    assert history.tolist() == [[0.0, 0.0, 0.0], [1.0, 0.75, 0.5]]

    loader._dof_names = ["joint_a"]
    loader._body_names = ["body_a"]
    with pytest.raises(ValueError, match="missing_joint"):
        loader.get_dof_index(["missing_joint"])
    with pytest.raises(ValueError, match="missing_body"):
        loader.get_body_index(["missing_body"])


def _load_g1_amp_reference_helpers():
    helper_names = {
        "_reference_motion_time",
        "_reference_motion_progress",
        "_reference_motion_horizon_reached",
    }
    tree = ast.parse(G1_AMP_ENV_PATH.read_text())
    helpers = [
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in helper_names
    ]
    assert {node.name for node in helpers} == helper_names
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            *helpers,
        ],
        type_ignores=[],
    )
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(module), str(G1_AMP_ENV_PATH), "exec"), namespace)
    return tuple(namespace[name] for name in sorted(helper_names))


def _load_g1_amp_default_reset_class():
    tree = ast.parse(G1_AMP_ENV_PATH.read_text())
    env_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "G1AmpEnv")
    method_names = {"_reset_strategy_default", "_set_reference_start_times"}
    methods = [
        node for node in env_class.body if isinstance(node, ast.FunctionDef) and node.name in method_names
    ]
    assert {node.name for node in methods} == method_names
    dummy_class = ast.ClassDef(
        name="DummyG1AmpEnv",
        bases=[],
        keywords=[],
        body=methods,
        decorator_list=[],
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            dummy_class,
        ],
        type_ignores=[],
    )
    namespace = {"np": np, "torch": torch}
    exec(compile(ast.fix_missing_locations(module), str(G1_AMP_ENV_PATH), "exec"), namespace)
    return namespace["DummyG1AmpEnv"]


def _load_g1_amp_history_update_helper():
    tree = ast.parse(G1_AMP_ENV_PATH.read_text())
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_update_amp_history_after_observation"
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0),
            helper,
        ],
        type_ignores=[],
    )
    namespace = {"torch": torch}
    exec(compile(ast.fix_missing_locations(module), str(G1_AMP_ENV_PATH), "exec"), namespace)
    return namespace["_update_amp_history_after_observation"]


def test_g1_amp_progress_tracks_sampled_reference_time_and_stops_before_clip_end():
    horizon_reached, progress, reference_time = _load_g1_amp_reference_helpers()
    start_times = torch.tensor([0.0, 4.0])
    episode_steps = torch.tensor([3, 3])

    times = reference_time(start_times, episode_steps, 1.0)
    phases = progress(start_times, episode_steps, 1.0, 10.0)

    torch.testing.assert_close(times, torch.tensor([3.0, 7.0]))
    torch.testing.assert_close(phases, torch.tensor([[0.3], [0.7]]))
    assert phases.shape == (2, 1)
    assert horizon_reached(start_times, torch.tensor([8, 5]), 1.0, 10.0).tolist() == [False, True]
    final_valid_start = torch.tensor([10.0 - 1.0 / 60.0])
    final_sample_time = reference_time(final_valid_start, torch.tensor([1]), 1.0 / 60.0)
    assert final_sample_time.item() == pytest.approx(10.0, abs=1.0e-6)
    assert horizon_reached(final_valid_start, torch.tensor([1]), 1.0 / 60.0, 10.0).item()
    assert "duration=latest_start" in G1_AMP_ENV_PATH.read_text()


def test_g1_amp_default_reset_reseeds_every_amp_history_frame():
    dummy_cls = _load_g1_amp_default_reset_class()
    update_history = _load_g1_amp_history_update_helper()
    env = dummy_cls()
    env._reference_start_times = torch.full((3,), 7.0)
    env.cfg = types.SimpleNamespace(num_amp_observations=3)
    env.amp_observation_buffer = torch.full((3, 3, 2), 99.0)
    env._repeat_amp_history_on_next_observation = torch.zeros(3, dtype=torch.bool)
    env._reference_amp_history_on_next_observation = torch.zeros(3, dtype=torch.bool)
    env.robot = types.SimpleNamespace(
        data=types.SimpleNamespace(
            default_root_state=torch.arange(39, dtype=torch.float32).reshape(3, 13),
            default_joint_pos=torch.arange(6, dtype=torch.float32).reshape(3, 2),
            default_joint_vel=torch.arange(6, 12, dtype=torch.float32).reshape(3, 2),
        )
    )
    env.scene = types.SimpleNamespace(env_origins=torch.zeros(3, 3))

    root_state, joint_pos, joint_vel = env._reset_strategy_default(torch.tensor([1]))

    assert env._reference_start_times.tolist() == [7.0, 0.0, 7.0]
    torch.testing.assert_close(env.amp_observation_buffer[1], torch.zeros(3, 2))
    torch.testing.assert_close(env.amp_observation_buffer[0], torch.full((3, 2), 99.0))
    assert env._repeat_amp_history_on_next_observation.tolist() == [False, True, False]
    assert env._reference_amp_history_on_next_observation.tolist() == [False, False, False]

    actual_post_reset_observation = torch.tensor([[1.0, 2.0], [7.0, 8.0], [3.0, 4.0]])
    update_history(
        env.amp_observation_buffer,
        actual_post_reset_observation,
        env._repeat_amp_history_on_next_observation,
        env._reference_amp_history_on_next_observation,
    )
    torch.testing.assert_close(
        env.amp_observation_buffer[1],
        actual_post_reset_observation[1].repeat(3, 1),
    )
    assert env._repeat_amp_history_on_next_observation.tolist() == [False, False, False]
    torch.testing.assert_close(root_state, env.robot.data.default_root_state[1:2])
    torch.testing.assert_close(joint_pos, env.robot.data.default_joint_pos[1:2])
    torch.testing.assert_close(joint_vel, env.robot.data.default_joint_vel[1:2])


def test_g1_amp_random_reset_replaces_current_reference_without_shifting_history():
    update_history = _load_g1_amp_history_update_helper()
    history = torch.tensor(
        [
            [[3.0], [2.0], [1.0]],
            [[10.0], [9.0], [8.0]],
        ]
    )
    current = torch.tensor([[4.0], [10.5]])
    repeat_current = torch.tensor([False, False])
    reference_seeded = torch.tensor([False, True])

    update_history(history, current, repeat_current, reference_seeded)

    # Normal rollout history shifts newest-first.
    torch.testing.assert_close(history[0], torch.tensor([[4.0], [3.0], [2.0]]))
    # A fresh random reset starts with [ref(t), ref(t-dt), ref(t-2dt)].
    # Replacing slot zero yields [actual(t), ref(t-dt), ref(t-2dt)] without duplicating t.
    torch.testing.assert_close(history[1], torch.tensor([[10.5], [9.0], [8.0]]))
    assert reference_seeded.tolist() == [False, False]


def test_g1_amp_observation_update_preserves_reward_diagnostics():
    source = G1_AMP_ENV_PATH.read_text()

    assert 'self.extras["amp_obs"] =' in source
    assert 'self.extras = {"amp_obs":' not in source


def test_joint_reset_sampling_is_unbiased_and_limit_safe():
    reset_utils = _load_module("reset_utils_safety_test", RESET_UTILS_PATH)
    default = torch.tensor([[0.0, 0.5]])
    lower = torch.tensor([[-1.0, 0.0]])
    upper = torch.tensor([[1.0, 1.0]])

    samples = torch.tensor([[-1.0, 1.0]])
    full_noise = reset_utils.sample_joint_reset_positions(default, lower, upper, 1.0, samples)
    no_noise = reset_utils.sample_joint_reset_positions(default, lower, upper, 0.0, samples)

    assert torch.equal(full_noise, torch.tensor([[-1.0, 1.0]]))
    assert torch.equal(no_noise, default)
    assert torch.all(full_noise >= lower)
    assert torch.all(full_noise <= upper)
    with pytest.raises(ValueError, match="noise_scale"):
        reset_utils.sample_joint_reset_positions(default, lower, upper, 1.1, samples)


def test_legacy_shared_policy_checkpoint_keeps_full_model_state(tmp_path):
    legacy_checkpoint = _load_module("legacy_checkpoint_safety_test", LEGACY_CHECKPOINT_PATH)
    model_state = {"actor.0.weight": torch.randn(2, 2), "critic.0.weight": torch.randn(1, 2)}
    checkpoint = tmp_path / "model_7.pt"
    torch.save({"model_state_dict": model_state, "iter": 7, "infos": {}}, checkpoint)

    class CTSActorCritic:
        pass

    class FakeAlgorithm:
        policy = CTSActorCritic()

        def load(self, loaded_dict, load_cfg, strict):
            assert strict
            assert load_cfg["actor"] and load_cfg["critic"]
            assert loaded_dict["actor_state_dict"] is loaded_dict["model_state_dict"]
            assert loaded_dict["critic_state_dict"] is loaded_dict["model_state_dict"]
            return True

    runner = types.SimpleNamespace(alg=FakeAlgorithm(), device="cpu", current_learning_iteration=0)
    legacy_checkpoint.load_checkpoint_for_play(runner, str(checkpoint))
    assert runner.current_learning_iteration == 7


def test_legacy_go2_d1_leg_wbc_warm_start_removes_only_arm_velocity_inputs(tmp_path):
    legacy_checkpoint = _load_module("legacy_checkpoint_go2_d1_test", LEGACY_CHECKPOINT_PATH)
    first_weight = torch.arange(2 * 64, dtype=torch.float32).reshape(2, 64)
    actor_state = {
        "mlp.0.weight": first_weight,
        "mlp.6.weight": torch.zeros((12, 2)),
    }
    checkpoint = tmp_path / "model_6999.pt"
    torch.save(
        {
            "actor_state_dict": actor_state,
            "critic_state_dict": {"mlp.0.weight": torch.zeros((1, 88))},
            "optimizer_state_dict": {"state": "must not load"},
            "iter": 6999,
            "infos": {"source": "legacy"},
        },
        checkpoint,
    )

    class FakePolicy:
        def state_dict(self):
            return {"mlp.0.weight": torch.zeros((2, 56))}

    class FakeAlgorithm:
        def __init__(self):
            self.loaded = None

        def get_policy(self):
            return FakePolicy()

        def load(self, loaded_dict, load_cfg, strict):
            self.loaded = (loaded_dict, load_cfg, strict)
            return True

    class FakeRunner:
        def __init__(self):
            self.alg = FakeAlgorithm()
            self.device = "cpu"
            self.current_learning_iteration = 0

        def load(self, path):
            raise AssertionError(f"The unadapted checkpoint loader must not receive {path}")

    runner = FakeRunner()
    infos = legacy_checkpoint.load_checkpoint_for_train(
        runner,
        str(checkpoint),
        "Gurukul-Isaac-LegWBC-AsyncArm-Flat-Unitree-Go2-D1-Arm-v0",
    )

    loaded_dict, load_cfg, strict = runner.alg.loaded
    expected = torch.cat((first_weight[:, :44], first_weight[:, 52:]), dim=1)
    assert torch.equal(loaded_dict["actor_state_dict"]["mlp.0.weight"], expected)
    assert torch.equal(actor_state["mlp.0.weight"], first_weight)
    assert load_cfg == {"actor": True, "critic": True, "optimizer": False, "iteration": True, "rnd": False}
    assert strict
    assert runner.current_learning_iteration == 6999
    assert infos == {"source": "legacy"}
