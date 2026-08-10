from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DECAP_PATH = REPO_ROOT / "scripts/reinforcement_learning/rsl_rl/decap.py"
ACTION_PATH = (
    REPO_ROOT / "source/Gurukul/Gurukul/tasks/manager_based/go2_apex/mdp/actions.py"
)


def _install_rsl_rl_stubs():
    rsl_rl = types.ModuleType("rsl_rl")
    algorithms = types.ModuleType("rsl_rl.algorithms")
    env = types.ModuleType("rsl_rl.env")
    extensions = types.ModuleType("rsl_rl.extensions")
    runners = types.ModuleType("rsl_rl.runners")
    storage = types.ModuleType("rsl_rl.storage")
    utils = types.ModuleType("rsl_rl.utils")
    tensordict = types.ModuleType("tensordict")
    torch = types.ModuleType("torch")

    class PPO:
        pass

    class VecEnv:
        pass

    class OnPolicyRunner:
        def load(self, path, *args, **kwargs):
            self.current_learning_iteration = 123
            return {"path": path}

    class RolloutStorage:
        def __init__(self, *args, **kwargs):
            pass

    class TensorDict(dict):
        pass

    algorithms.PPO = PPO
    env.VecEnv = VecEnv
    extensions.resolve_rnd_config = lambda alg_cfg, obs, obs_groups, env: alg_cfg
    extensions.resolve_symmetry_config = lambda alg_cfg, env: alg_cfg
    runners.OnPolicyRunner = OnPolicyRunner
    storage.RolloutStorage = RolloutStorage
    torch.load = lambda *args, **kwargs: {}
    utils.resolve_callable = lambda value: value
    utils.resolve_obs_groups = lambda obs, obs_groups, default_sets: obs_groups
    tensordict.TensorDict = TensorDict

    modules = {
        "rsl_rl": rsl_rl,
        "rsl_rl.algorithms": algorithms,
        "rsl_rl.env": env,
        "rsl_rl.extensions": extensions,
        "rsl_rl.runners": runners,
        "rsl_rl.storage": storage,
        "rsl_rl.utils": utils,
        "tensordict": tensordict,
        "torch": torch,
    }
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    return previous


def _restore_modules(previous):
    for name, module in previous.items():
        if module is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = module


def _load_decap_module():
    previous = _install_rsl_rl_stubs()
    try:
        spec = importlib.util.spec_from_file_location("decap_resume_test_module", DECAP_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load module from {DECAP_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        _restore_modules(previous)


def _install_action_stubs():
    isaaclab = types.ModuleType("isaaclab")
    envs = types.ModuleType("isaaclab.envs")
    mdp = types.ModuleType("isaaclab.envs.mdp")
    utils = types.ModuleType("isaaclab.utils")
    torch = types.ModuleType("torch")

    class JointPositionAction:
        pass

    class JointPositionActionCfg:
        pass

    mdp.JointPositionAction = JointPositionAction
    mdp.JointPositionActionCfg = JointPositionActionCfg
    utils.configclass = lambda cls: cls
    torch.Tensor = object
    torch.full = lambda *args, **kwargs: None
    torch.tensor = lambda *args, **kwargs: None
    torch.arange = lambda *args, **kwargs: None
    torch.long = object()

    modules = {
        "isaaclab": isaaclab,
        "isaaclab.envs": envs,
        "isaaclab.envs.mdp": mdp,
        "isaaclab.utils": utils,
        "torch": torch,
    }
    previous = {name: sys.modules.get(name) for name in modules}
    sys.modules.update(modules)
    return previous


def _load_action_module():
    previous = _install_action_stubs()
    try:
        spec = importlib.util.spec_from_file_location("decap_action_test_module", ACTION_PATH)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"Unable to load module from {ACTION_PATH}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    finally:
        _restore_modules(previous)


def test_decap_schedule_restores_from_iteration():
    decap = _load_decap_module()
    alg = decap.DecAPPPO.__new__(decap.DecAPPPO)
    alg.decap_steps_per_iteration = 24
    alg.decap_lambda_start = 1.0
    alg.decap_lambda_end = 0.0
    alg.decap_decay_type = "linear"
    alg.decap_decay_start_iteration = 50
    alg.decap_decay_end_iteration = 150
    alg.current_decay_progress = 0.0

    alg.restore_decap_schedule_from_iteration(100)

    assert alg.decap_step == 2400
    assert alg.learning_iteration == 100
    assert alg.current_lambda == 0.5


def test_decap_runner_load_restores_loaded_iteration():
    decap = _load_decap_module()

    class Alg:
        current_lambda = 0.0

        def restore_decap_schedule_from_iteration(self, iteration):
            self.restored_iteration = iteration
            self.current_lambda = 0.25

    runner = decap.DecAPRunner.__new__(decap.DecAPRunner)
    runner.alg = Alg()

    infos = runner.load("/tmp/model_999.pt")

    assert infos == {"path": "/tmp/model_999.pt"}
    assert runner.alg.restored_iteration == 123


def test_env_decap_action_uses_resume_iteration_offset():
    actions = _load_action_module()
    action = actions.DecapJointPositionAction.__new__(actions.DecapJointPositionAction)
    action._decap_steps_per_iteration = 24
    action._decap_resume_step_offset = 1000 * 24
    action._decap_decay_start_step = 100 * 24
    action._decap_decay_end_step = 1000 * 24
    action._env = types.SimpleNamespace(common_step_counter=0)

    assert action._current_decap_step() == 24000
    assert action._compute_decap_progress() == 1.0
