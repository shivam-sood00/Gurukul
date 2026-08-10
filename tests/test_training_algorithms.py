from __future__ import annotations

import importlib.util
import sys
import types
from copy import deepcopy
from pathlib import Path

import pytest
import torch

pytest.importorskip("rsl_rl")
TensorDict = pytest.importorskip("tensordict").TensorDict

from rsl_rl.storage import RolloutStorage

REPO_ROOT = Path(__file__).resolve().parents[1]
CTS_PATH = REPO_ROOT / "source/Gurukul/Gurukul/tasks/manager_based/concurrent_teacher_student.py"
CTS_CFG_PATH = (
    REPO_ROOT
    / "source/Gurukul/Gurukul/tasks/manager_based/locomotion/velocity/config/quadruped"
    / "unitree_go2/agents/rsl_rl_cts_cfg.py"
)
GO2_REGISTRY_PATH = CTS_CFG_PATH.parents[1] / "__init__.py"
MULTI_CRITIC_PATH = REPO_ROOT / "scripts/reinforcement_learning/rsl_rl/multi_critic.py"
APEX_MULTI_CRITIC_CFG_PATH = (
    REPO_ROOT
    / "source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/agents"
    / "rsl_rl_multi_critic_cfg.py"
)
CONTACT_MEMORY_PATH = REPO_ROOT / "source/Gurukul/Gurukul/utils/contact_trail_memory.py"
VELOCITY_PATH = REPO_ROOT / "source/Gurukul/Gurukul/tasks/manager_based/locomotion/velocity"


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Unable to load module from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_cts_rollout_and_update_smoke():
    torch.manual_seed(1)
    cts = _load_module("cts_training_smoke_test", CTS_PATH)
    num_envs, num_steps, num_actions = 8, 4, 2
    obs_groups = {
        "policy": ["policy"],
        "teacher": ["teacher"],
        "student_history": ["history"],
        "critic": ["teacher"],
        "cts_role": ["cts_role"],
    }

    def make_obs():
        return TensorDict(
            {
                "policy": torch.randn(num_envs, 5),
                "teacher": torch.randn(num_envs, 7),
                "history": torch.randn(num_envs, 15),
                "cts_role": torch.cat((torch.ones(6, 1), torch.zeros(2, 1))),
            },
            batch_size=[num_envs],
        )

    obs = make_obs()
    policy = cts.CTSActorCritic(
        obs,
        obs_groups,
        num_actions,
        actor_hidden_dims=[16],
        critic_hidden_dims=[16],
        privileged_encoder_hidden_dims=[8],
        history_encoder_hidden_dims=[8],
        latent_dim=4,
    )
    storage = RolloutStorage("rl", num_envs, num_steps, obs, [num_actions], "cpu")
    algorithm = cts.CTSPPO(
        policy,
        storage=storage,
        device="cpu",
        optimizer="sgd",
        share_cnn_encoders=False,
        num_learning_epochs=1,
        num_mini_batches=2,
    )
    assert isinstance(algorithm.optimizer, torch.optim.SGD)
    for _ in range(num_steps):
        algorithm.act(obs)
        obs = make_obs()
        algorithm.process_env_step(obs, torch.randn(num_envs), torch.zeros(num_envs, dtype=torch.bool), {})
    algorithm.compute_returns(obs)

    metrics = algorithm.update()
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert policy.output_std.shape == (num_actions,)
    assert policy(obs).shape == (num_envs, num_actions)
    assert callable(torch.distributions.Normal.set_default_validate_args)
    assert "history_encoder_optimizer_state_dict" in algorithm.save()


def test_cts_reconstruction_uses_student_samples_and_squared_l2_norm():
    cts = _load_module("cts_reconstruction_test", CTS_PATH)
    num_envs = 4
    obs_groups = {
        "policy": ["policy"],
        "teacher": ["teacher"],
        "student_history": ["history"],
        "critic": ["teacher"],
        "cts_role": ["cts_role"],
    }
    obs = TensorDict(
        {
            "policy": torch.zeros(num_envs, 1),
            "teacher": torch.tensor([[100.0, 100.0], [200.0, 200.0], [1.0, 2.0], [3.0, 4.0]]),
            "history": torch.zeros(num_envs, 2),
            "cts_role": torch.tensor([[1.0], [1.0], [0.0], [0.0]]),
        },
        batch_size=[num_envs],
    )
    policy = cts.CTSActorCritic(
        obs,
        obs_groups,
        num_actions=1,
        actor_hidden_dims=[4],
        critic_hidden_dims=[4],
        privileged_encoder_hidden_dims=[],
        history_encoder_hidden_dims=[],
        latent_dim=2,
        latent_norm=None,
    )
    with torch.no_grad():
        for encoder in (policy.privileged_encoder, policy.history_encoder):
            encoder[0].weight.copy_(torch.eye(2))
            encoder[0].bias.zero_()
    storage = RolloutStorage("rl", num_envs, 1, obs, [1], "cpu")
    algorithm = cts.CTSPPO(policy, storage=storage, device="cpu", num_encoder_epochs=1)

    # Eq. 8: ((1^2 + 2^2) + (3^2 + 4^2)) / 2 student samples = 15.
    assert algorithm._encoder_loss(obs).item() == pytest.approx(15.0)


def test_cts_adaptive_kl_schedule_changes_the_ppo_learning_rate():
    cts = _load_module("cts_adaptive_kl_test", CTS_PATH)
    num_envs = 2
    obs_groups = {
        "policy": ["policy"],
        "teacher": ["teacher"],
        "student_history": ["history"],
        "critic": ["teacher"],
        "cts_role": ["cts_role"],
    }
    obs = TensorDict(
        {
            "policy": torch.zeros(num_envs, 1),
            "teacher": torch.zeros(num_envs, 2),
            "history": torch.zeros(num_envs, 2),
            "cts_role": torch.tensor([[1.0], [0.0]]),
        },
        batch_size=[num_envs],
    )
    policy = cts.CTSActorCritic(
        obs,
        obs_groups,
        num_actions=1,
        actor_hidden_dims=[4],
        critic_hidden_dims=[4],
        privileged_encoder_hidden_dims=[4],
        history_encoder_hidden_dims=[4],
        latent_dim=2,
    )
    storage = RolloutStorage("rl", num_envs, 1, obs, [1], "cpu")
    algorithm = cts.CTSPPO(
        policy,
        storage=storage,
        device="cpu",
        learning_rate=1.0e-3,
        schedule="adaptive",
        desired_kl=0.01,
    )
    old_params = (torch.zeros(num_envs, 1), torch.ones(num_envs, 1))
    new_params = (torch.full((num_envs, 1), 10.0), torch.ones(num_envs, 1))

    algorithm._adapt_learning_rate(old_params, new_params)

    assert algorithm.learning_rate == pytest.approx(1.0e-3 / 1.5)
    assert algorithm.optimizer.param_groups[0]["lr"] == pytest.approx(algorithm.learning_rate)


def test_cts_encoder_gradients_follow_algorithm_one_update_partition():
    cts = _load_module("cts_gradient_partition_test", CTS_PATH)
    num_envs = 4
    obs_groups = {
        "policy": ["policy"],
        "teacher": ["teacher"],
        "student_history": ["history"],
        "critic": ["teacher"],
        "cts_role": ["cts_role"],
    }
    obs = TensorDict(
        {
            "policy": torch.randn(num_envs, 3),
            "teacher": torch.randn(num_envs, 5),
            "history": torch.randn(num_envs, 7),
            "cts_role": torch.tensor([[1.0], [1.0], [0.0], [0.0]]),
        },
        batch_size=[num_envs],
    )
    policy = cts.CTSActorCritic(
        obs,
        obs_groups,
        num_actions=2,
        actor_hidden_dims=[8],
        critic_hidden_dims=[8],
        privileged_encoder_hidden_dims=[8],
        history_encoder_hidden_dims=[8],
        latent_dim=4,
    )

    teacher_obs = obs[:2]
    policy.forward_teacher(teacher_obs).sum().backward()
    assert any(parameter.grad is not None for parameter in policy.privileged_encoder.parameters())

    policy.zero_grad(set_to_none=True)
    student_obs = obs[2:]
    policy.forward_student(student_obs).sum().backward()
    assert all(parameter.grad is None for parameter in policy.history_encoder.parameters())

    policy.zero_grad(set_to_none=True)
    policy.evaluate(obs).sum().backward()
    assert all(parameter.grad is None for parameter in policy.privileged_encoder.parameters())
    assert all(parameter.grad is None for parameter in policy.history_encoder.parameters())
    assert any(parameter.grad is not None for parameter in policy.critic.parameters())


def test_cts_paper_defaults_and_named_agent_entry_point_are_registered():
    cfg_source = CTS_CFG_PATH.read_text()
    registry_source = GO2_REGISTRY_PATH.read_text()
    cts_registration = registry_source.split(
        'id="Gurukul-Isaac-Velocity-Rough-Unitree-Go2-CTS-v0"', maxsplit=1
    )[1].split("gym.register(", maxsplit=1)[0]

    assert "latent_dim: int = 32" in cfg_source
    assert 'latent_norm: str | None = "l2"' in cfg_source
    assert "privileged_encoder_hidden_dims: list[int] = [512, 256]" in cfg_source
    assert "critic_hidden_dims: list[int] = [512, 256, 128]" in cfg_source
    assert "num_encoder_epochs: int = 5" in cfg_source
    assert 'schedule="adaptive"' in cfg_source
    assert '"rsl_rl_cts_cfg_entry_point"' in cts_registration


def test_multi_critic_rollout_and_update_smoke():
    torch.manual_seed(2)
    multi_critic = _load_module("multi_critic_training_smoke_test", MULTI_CRITIC_PATH)
    num_envs, num_steps, num_actions = 8, 4, 3
    obs_groups = {"policy": ["policy"], "critic": ["critic_a", "critic_b"]}

    def make_obs():
        return TensorDict(
            {
                "policy": torch.randn(num_envs, 6),
                "critic_a": torch.randn(num_envs, 7),
                "critic_b": torch.randn(num_envs, 5),
            },
            batch_size=[num_envs],
        )

    obs = make_obs()
    policy = multi_critic.MultiCriticActorCritic(
        obs,
        obs_groups,
        num_actions,
        actor_hidden_dims=[16],
        critic_hidden_dims=[16],
        critic_obs_groups=[["critic_a"], ["critic_b"]],
    )
    storage = multi_critic.MultiCriticRolloutStorage(num_envs, num_steps, obs, [num_actions], 2, "cpu")
    algorithm = multi_critic.MultiCriticPPO(
        policy,
        storage,
        num_critics=2,
        device="cpu",
        num_learning_epochs=1,
        num_mini_batches=2,
    )
    for _ in range(num_steps):
        algorithm.act(obs)
        obs = make_obs()
        algorithm.process_env_step(
            obs,
            torch.randn(num_envs, 2),
            torch.zeros(num_envs, dtype=torch.bool),
            {},
        )
    algorithm.compute_returns(obs)

    metrics = algorithm.update()
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert policy(obs).shape == (num_envs, num_actions)

    class FakeEnv:
        num_envs = 2
        num_actions = 3
        cfg = {}

        def get_observations(self):
            return TensorDict(
                {
                    "policy": torch.zeros(self.num_envs, 6),
                    "critic_a": torch.zeros(self.num_envs, 7),
                    "critic_b": torch.zeros(self.num_envs, 5),
                },
                batch_size=[self.num_envs],
            )

    runner_cfg = {
        "num_steps_per_env": 2,
        "obs_groups": deepcopy(obs_groups),
        "policy": {
            "class_name": "MultiCriticActorCritic",
            "actor_hidden_dims": [8],
            "critic_hidden_dims": [8],
        },
        "algorithm": {
            "class_name": "MultiCriticPPO",
            "num_learning_epochs": 1,
            "num_mini_batches": 1,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
        "multi_critic_groups": [["critic_a"], ["critic_b"]],
        "multi_critic_reward_weights": [0.5, 0.5],
        "multi_critic_advantage_weights": [0.5, 0.5],
    }
    runner = multi_critic.MultiCriticRunner(FakeEnv(), runner_cfg, log_dir=None, device="cpu")
    assert isinstance(runner.alg, multi_critic.MultiCriticPPO)
    assert runner.alg.get_policy().output_std.shape == (3,)

    reward_manager = types.SimpleNamespace(
        active_terms=["style", "task"],
        _term_cfgs=[types.SimpleNamespace(weight=1.0), types.SimpleNamespace(weight=-0.5)],
        _step_reward=torch.tensor([[2.0, -0.5], [1.0, -0.25]]),
    )
    vec_env = types.SimpleNamespace(
        unwrapped=types.SimpleNamespace(reward_manager=reward_manager, step_dt=0.02)
    )
    algorithm.bind_env(vec_env)
    algorithm.reward_term_groups = [["style"], []]
    with pytest.raises(ValueError, match="active nonzero terms not assigned"):
        algorithm._extract_reward_groups_from_env_terms()

    algorithm.reward_term_groups = [["style"], ["task"]]
    algorithm._reward_term_indices = None
    grouped = algorithm._extract_reward_groups_from_env_terms()
    torch.testing.assert_close(grouped.sum(dim=1), reward_manager._step_reward.sum(dim=1) * 0.02)


def test_apex_multi_critic_assigns_every_active_reward_term():
    cfg_source = APEX_MULTI_CRITIC_CFG_PATH.read_text()
    for term_name in (
        "imitate_world_foot_pos",
        "imitate_world_base_pos",
        "airborne_contact",
        "track_command_lin_vel_z",
    ):
        assert f'"{term_name}"' in cfg_source


def test_start_rollout_and_update_smoke():
    torch.manual_seed(3)
    package_name = "_start_training_smoke_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(VELOCITY_PATH)]
    sys.modules[package_name] = package
    start = _load_module(f"{package_name}.start_actor_critic", VELOCITY_PATH / "start_actor_critic.py")
    sys.modules[f"{package_name}.start_actor_critic"] = start

    num_envs, num_steps, num_actions = 4, 2, 2
    obs_groups = {"policy": ["policy", "depth_camera"], "critic": ["critic"]}

    def make_obs():
        return TensorDict(
            {
                "policy": torch.randn(num_envs, 10),
                "depth_camera": torch.randn(num_envs, 64),
                "critic": torch.randn(num_envs, 12),
                "terrain_heightmap": torch.randn(num_envs, 16),
                "feet_heightmap": torch.randn(num_envs, 8),
                "base_velocity_gt": torch.randn(num_envs, 3),
            },
            batch_size=[num_envs],
        )

    obs = make_obs()
    policy = start.StartActorCritic(
        obs,
        obs_groups,
        num_actions,
        actor_hidden_dims=[16],
        critic_hidden_dims=[16],
        depth_image_shape=(8, 8),
        depth_backbone_channels=[4],
        depth_backbone_kernels=[3],
        depth_backbone_image_fc_dim=8,
        depth_backbone_latent_dim=4,
        terrain_map_shape=(4, 4),
        tr_proprio_hidden_dim=8,
        tr_rnn_hidden_dim=8,
        ie_prop_hidden_dim=8,
        ie_rnn_hidden_dim=8,
        ie_map_latent_dim=4,
        ie_transformer_dim=8,
        ie_transformer_heads=2,
        ie_transformer_layers=1,
        ie_transformer_ff_dim=16,
        explicit_latent_dim=4,
        implicit_latent_dim=4,
        critic_rnn_hidden_dim=8,
        tr_refine_base_channels=2,
    )
    storage = RolloutStorage("rl", num_envs, num_steps, obs, [num_actions], "cpu")
    algorithm = start.StartPPO(
        policy,
        storage=storage,
        device="cpu",
        num_learning_epochs=1,
        num_mini_batches=1,
    )
    fixed_rollout_obs = None
    for step in range(num_steps):
        algorithm.act(obs)
        rollout_obs = algorithm.transition.observations
        assert rollout_obs["_start_adasmpl_selector"].shape == (num_envs, 1)
        if step == 0:
            fixed_rollout_obs = rollout_obs.clone()
        obs = make_obs()
        algorithm.process_env_step(obs, torch.randn(num_envs), torch.zeros(num_envs, dtype=torch.bool), {})
    algorithm.compute_returns(obs)

    # The curriculum probability can change before PPO replays this rollout.
    # Replay must use the realized selector stored with each transition rather
    # than consulting the policy's current sampling probability.
    policy.set_adasmpl_probability(0.0)
    batch = next(storage.recurrent_mini_batch_generator(num_mini_batches=1, num_epochs=1))
    policy.act(batch.observations, masks=batch.masks, hidden_states=batch.hidden_states[0])
    recomputed_log_prob = policy.get_actions_log_prob(batch.actions)
    torch.testing.assert_close(recomputed_log_prob.reshape(-1), batch.old_actions_log_prob.reshape(-1))

    metrics = algorithm.update()
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert policy.output_std.shape == (num_actions,)
    assert fixed_rollout_obs is not None
    policy.reset()
    policy.act(fixed_rollout_obs)
    first_mean = policy.action_mean.detach().clone()
    policy.reset()
    policy.act(fixed_rollout_obs)
    torch.testing.assert_close(policy.action_mean, first_mean)
    policy.reset(torch.ones(num_envs, dtype=torch.bool))
    assert policy(obs).shape == (num_envs, num_actions)


def test_contact_trail_replay_matches_rollout_and_preserves_live_memory():
    torch.manual_seed(4)
    package_name = "_contact_trail_training_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(VELOCITY_PATH)]
    sys.modules[package_name] = package

    stub_names = ("Gurukul", "Gurukul.utils", "Gurukul.utils.contact_trail_memory")
    previous_modules = {name: sys.modules.get(name) for name in stub_names}
    try:
        mega_package = types.ModuleType("Gurukul")
        mega_package.__path__ = []
        utils_package = types.ModuleType("Gurukul.utils")
        utils_package.__path__ = []
        sys.modules["Gurukul"] = mega_package
        sys.modules["Gurukul.utils"] = utils_package
        contact_memory = _load_module("Gurukul.utils.contact_trail_memory", CONTACT_MEMORY_PATH)
        sys.modules["Gurukul.utils.contact_trail_memory"] = contact_memory
        contact_trail = _load_module(
            f"{package_name}.contact_trail_actor_critic",
            VELOCITY_PATH / "contact_trail_actor_critic.py",
        )
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

    num_envs, num_steps, num_actions = 4, 2, 2
    obs_groups = {
        "policy": ["proprio", "contact_trail_events", "contact_trail_pose", "foot_pos_b"],
        "critic": ["critic"],
    }

    def make_obs(step: int):
        events = torch.zeros(num_envs, 4, contact_memory.CONTACT_FEATURE_DIM)
        events[..., 0] = 1.0
        events[..., 1] = 50.0
        pose = torch.zeros(num_envs, 7)
        pose[:, 0] = 0.01 * step
        pose[:, 3] = 1.0
        feet = torch.tensor(
            [[0.10, 0.10, -0.25], [0.10, -0.10, -0.25], [-0.10, 0.10, -0.25], [-0.10, -0.10, -0.25]]
        ).expand(num_envs, -1, -1)
        return TensorDict(
            {
                "proprio": torch.randn(num_envs, 6),
                "contact_trail_events": events.flatten(1),
                "contact_trail_pose": pose,
                "foot_pos_b": feet.flatten(1),
                "critic": torch.randn(num_envs, 8),
            },
            batch_size=[num_envs],
        )

    obs = make_obs(0)
    policy = contact_trail.ContactTrailActorCritic(
        obs,
        obs_groups,
        num_actions,
        actor_hidden_dims=[16],
        critic_hidden_dims=[16],
        rnn_hidden_dim=8,
        cnn_latent_dim=4,
        proprio_hidden_dim=4,
        num_envs=num_envs,
        use_contact_quality_aux_loss=False,
        contact_trail_cfg={"num_channels": 4, "grid_size": (8, 8), "write_radius": 0, "use_warp": True},
    )
    storage = RolloutStorage("rl", num_envs, num_steps, obs, [num_actions], "cpu")
    algorithm = contact_trail.ContactTrailPPO(
        policy,
        storage=storage,
        device="cpu",
        learning_rate=0.0,
        num_learning_epochs=1,
        num_mini_batches=1,
        contact_quality_loss_coef=0.0,
    )
    for step in range(num_steps):
        algorithm.act(obs)
        obs = make_obs(step + 1)
        algorithm.process_env_step(obs, torch.randn(num_envs), torch.zeros(num_envs, dtype=torch.bool), {})
    algorithm.compute_returns(obs)

    memory = policy.contact_trail_memory
    live_state = (memory.num_envs, memory.map.clone(), memory.prev_base_pos_w.clone(), memory.prev_base_yaw.clone())
    batch = next(storage.recurrent_mini_batch_generator(num_mini_batches=1, num_epochs=1))
    policy.act(batch.observations, masks=batch.masks, hidden_states=batch.hidden_states[0])
    torch.testing.assert_close(
        policy.get_actions_log_prob(batch.actions.reshape(-1, num_actions)).reshape(-1),
        batch.old_actions_log_prob.reshape(-1),
    )
    assert memory.num_envs == live_state[0]
    torch.testing.assert_close(memory.map, live_state[1])
    torch.testing.assert_close(memory.prev_base_pos_w, live_state[2])
    torch.testing.assert_close(memory.prev_base_yaw, live_state[3])

    metrics = algorithm.update()
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert memory.num_envs == num_envs
    assert torch.count_nonzero(memory.map) == 0


def test_real_teacher_rollout_and_update_smoke():
    torch.manual_seed(4)
    package_name = "_real_teacher_training_smoke_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(VELOCITY_PATH)]
    sys.modules[package_name] = package

    stub_names = ("Gurukul", "Gurukul.utils", "Gurukul.utils.contact_trail_memory")
    previous_modules = {name: sys.modules.get(name) for name in stub_names}
    try:
        mega_package = types.ModuleType("Gurukul")
        mega_package.__path__ = []
        utils_package = types.ModuleType("Gurukul.utils")
        utils_package.__path__ = []
        sys.modules["Gurukul"] = mega_package
        sys.modules["Gurukul.utils"] = utils_package
        contact_memory = _load_module("Gurukul.utils.contact_trail_memory", CONTACT_MEMORY_PATH)
        sys.modules["Gurukul.utils.contact_trail_memory"] = contact_memory

        real_teacher = _load_module(f"{package_name}.real_teacher", VELOCITY_PATH / "real_teacher.py")
        sys.modules[f"{package_name}.real_teacher"] = real_teacher
        real_teacher_ppo = _load_module(
            f"{package_name}.real_teacher_ppo",
            VELOCITY_PATH / "real_teacher_ppo.py",
        )
    finally:
        for name, previous in previous_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous

    num_envs, num_steps, num_actions = 4, 2, 2
    obs_groups = {
        "actor": ["real_teacher_proprio", "real_teacher_terrain", "real_teacher_privileged"],
        "critic": ["real_teacher_proprio", "real_teacher_terrain", "real_teacher_privileged"],
    }

    def make_obs():
        return TensorDict(
            {
                "real_teacher_proprio": torch.randn(num_envs, 8),
                "real_teacher_terrain": torch.randn(num_envs, 16),
                "real_teacher_privileged": torch.randn(num_envs, 6),
            },
            batch_size=[num_envs],
        )

    obs = make_obs()
    policy = real_teacher.RealTeacherActorCritic(
        obs,
        obs_groups,
        num_actions,
        actor_hidden_dims=[16],
        critic_hidden_dims=[16],
        terrain_scan_shape=(4, 4),
        attention_embed_dim=8,
        attention_num_heads=2,
        terrain_encoder_hidden_dim=8,
        privileged_latent_dim=8,
        privileged_hidden_dims=[8],
    )
    storage = RolloutStorage("rl", num_envs, num_steps, obs, [num_actions], "cpu")
    algorithm = real_teacher_ppo.RealTeacherPPO(
        policy,
        storage=storage,
        device="cpu",
        num_learning_epochs=1,
        num_mini_batches=1,
    )
    for _ in range(num_steps):
        algorithm.act(obs)
        obs = make_obs()
        algorithm.process_env_step(obs, torch.randn(num_envs), torch.zeros(num_envs, dtype=torch.bool), {})
    algorithm.compute_returns(obs)

    metrics = algorithm.update()
    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert policy.output_std.shape == (num_actions,)
    assert policy(obs).shape == (num_envs, num_actions)


def test_combined_depth_distillation_rollout_and_update_smoke():
    torch.manual_seed(5)
    package_name = "_depth_distillation_training_smoke_test"
    package = types.ModuleType(package_name)
    package.__path__ = [str(VELOCITY_PATH)]
    sys.modules[package_name] = package
    depth = _load_module(
        f"{package_name}.depth_student_teacher",
        VELOCITY_PATH / "depth_student_teacher.py",
    )

    num_envs, num_steps, num_actions = 4, 3, 2
    obs_groups = {"policy": ["policy", "depth_camera"], "teacher": ["teacher"]}

    def make_obs(include_terrain: bool = False):
        values = {
            "policy": torch.randn(num_envs, 10),
            "depth_camera": torch.randn(num_envs, 64),
            "teacher": torch.randn(num_envs, 12),
        }
        if include_terrain:
            values["terrain_heightmap"] = torch.randn(num_envs, 16)
        return TensorDict(values, batch_size=[num_envs])

    obs = make_obs()
    policy = depth.StudentTeacherDepthBackbone(
        obs,
        obs_groups,
        num_actions,
        student_hidden_dims=[16],
        teacher_hidden_dims=[16],
        depth_image_shape=(8, 8),
        depth_backbone_channels=[4],
        depth_backbone_kernels=[3],
        depth_backbone_image_fc_dim=8,
        depth_backbone_latent_dim=4,
    )
    storage = RolloutStorage("distillation", num_envs, num_steps, obs, [num_actions], "cpu")
    algorithm = depth.CombinedDistillation(
        policy,
        storage=storage,
        device="cpu",
        num_learning_epochs=1,
        gradient_length=2,
    )
    first_parameter = next(policy.student.parameters())
    parameter_before = first_parameter.detach().clone()
    for _ in range(num_steps):
        algorithm.act(obs)
        obs = make_obs()
        algorithm.process_env_step(obs, torch.zeros(num_envs), torch.zeros(num_envs, dtype=torch.bool), {})
    algorithm.compute_returns(obs)
    metrics = algorithm.update()

    assert all(torch.isfinite(torch.tensor(value)) for value in metrics.values())
    assert not torch.equal(parameter_before, first_parameter.detach())
    assert policy.output_std.shape == (num_actions,)
    assert policy(obs).shape == (num_envs, num_actions)

    source_teacher_state = {
        f"mlp.{key}": value.detach().clone() + 0.25 for key, value in policy.teacher.state_dict().items()
    }
    algorithm.load(
        {"actor_state_dict": source_teacher_state},
        load_cfg={"teacher": True, "iteration": False},
        strict=True,
    )
    assert algorithm.teacher_loaded
    assert torch.equal(
        next(policy.teacher.parameters()).detach(),
        next(iter(source_teacher_state.values())),
    )

    obs = make_obs(include_terrain=True)
    start_policy = depth.StudentTeacherDepthBackboneRecurrentSTART(
        obs,
        obs_groups,
        num_actions,
        student_hidden_dims=[16],
        teacher_hidden_dims=[16],
        depth_image_shape=(8, 8),
        depth_backbone_channels=[4],
        depth_backbone_kernels=[3],
        depth_backbone_image_fc_dim=8,
        depth_backbone_latent_dim=4,
        depth_recurrent_fusion_dim=8,
        rnn_hidden_dim=8,
        terrain_reconstruction_hidden_dim=8,
        terrain_reconstruction_latent_dim=4,
    )
    start_storage = RolloutStorage("distillation", num_envs, 2, obs, [num_actions], "cpu")
    start_algorithm = depth.StartDistillation(
        start_policy,
        storage=start_storage,
        device="cpu",
        num_learning_epochs=1,
        gradient_length=2,
    )
    for _ in range(2):
        start_algorithm.act(obs)
        obs = make_obs(include_terrain=True)
        start_algorithm.process_env_step(
            obs,
            torch.zeros(num_envs),
            torch.zeros(num_envs, dtype=torch.bool),
            {},
        )
    student_forward_calls = 0

    def count_student_forward(_module, _inputs, _output):
        nonlocal student_forward_calls
        student_forward_calls += 1

    hook = start_policy.student.register_forward_hook(count_student_forward)
    start_metrics = start_algorithm.update()
    hook.remove()
    assert all(torch.isfinite(torch.tensor(value)) for value in start_metrics.values())
    assert student_forward_calls == 2
    assert start_metrics["terrain_reconstruction"] > 0.0
