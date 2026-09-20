"""APEX registrations, reward partitions and robot-specific policy settings without Isaac Sim."""

from __future__ import annotations

import ast
import importlib
import importlib.util
import sys
import types
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
APEX = ROOT / "source/Gurukul/Gurukul/tasks/manager_based/go2_apex"


@pytest.fixture
def agent_configs(monkeypatch):
    """Use Isaac Lab's real configclass and runner configs, bypassing simulator package initializers."""
    isaac = importlib.util.find_spec("isaaclab")
    rl = importlib.util.find_spec("isaaclab_rl")
    if isaac is None or rl is None:
        pytest.skip("Isaac Lab configuration sources are required")

    def package(name, path):
        module = types.ModuleType(name)
        module.__path__ = [str(path)]
        monkeypatch.setitem(sys.modules, name, module)
        return module

    package("_apex_test_utils", Path(next(iter(isaac.submodule_search_locations))) / "utils")
    configclass = importlib.import_module("_apex_test_utils.configclass").configclass
    utils = types.ModuleType("isaaclab.utils")
    utils.configclass = configclass
    monkeypatch.setitem(sys.modules, "isaaclab.utils", utils)
    package("_apex_test_rl", Path(next(iter(rl.submodule_search_locations))) / "rsl_rl")
    cfg = importlib.import_module("_apex_test_rl.rl_cfg")
    monkeypatch.setitem(sys.modules, "isaaclab_rl.rsl_rl", cfg)
    package("_apex_test_agents", APEX / "config/go2/agents")
    return importlib.import_module("_apex_test_agents.rsl_rl_multi_critic_cfg")


def registrations():
    tree = ast.parse((APEX / "config/go2/__init__.py").read_text())
    ns = {
        "gym": types.SimpleNamespace(register=lambda **kw: result.append(kw)),
        "__name__": "apex",
        "agents": types.SimpleNamespace(__name__="agents"),
    }
    result = []
    calls = [n for n in tree.body if isinstance(n, ast.Expr) and isinstance(n.value, ast.Call)]
    exec(compile(ast.Module(body=calls, type_ignores=[]), "registrations", "exec"), ns)
    return result


def declared_arm_rewards(family):
    files = [APEX / "tracking_env_cfg.py", APEX / "config/go2" / f"flat_{family}_arm_tracker_env_cfg.py"]
    names = set()
    for path in files:
        tree = ast.parse(path.read_text())
        for node in ast.walk(tree):
            if not isinstance(node, ast.Assign) or not isinstance(node.value, ast.Call):
                continue
            if not isinstance(node.value.func, ast.Name) or node.value.func.id != "RewTerm":
                continue
            for target in node.targets:
                if isinstance(target, ast.Name):
                    names.add(target.id)
                elif isinstance(target, ast.Attribute) and ast.unparse(target.value) == "self.rewards":
                    names.add(target.attr)
    # These terms are deliberately absent in this family.
    if family == "d1":
        names.discard("imitate_joint_pos_arms")
    return names


@pytest.mark.parametrize("family", ["Go2-D1", "B2-Z1"])
def test_every_arm_ppo_task_has_a_complete_partition_and_preserves_baseline_settings(agent_configs, family):
    matched = 0
    for registration in registrations():
        if family not in registration["id"] or "rsl_rl_cfg_entry_point" not in registration["kwargs"]:
            continue
        matched += 1
        assert registration["entry_point"].endswith(":ApexManagerBasedRLEnv")
        entries = registration["kwargs"]
        cls = getattr(agent_configs, entries["rsl_rl_multi_critic_cfg_entry_point"].split(":")[-1])
        cfg = cls()
        baseline = cls.__bases__[0]()
        assert cfg.clip_actions == baseline.clip_actions
        assert cfg.policy.actor_hidden_dims == baseline.policy.actor_hidden_dims
        assert cfg.policy.critic_hidden_dims == baseline.policy.critic_hidden_dims
        assert cfg.policy.actor_obs_normalization == baseline.policy.actor_obs_normalization
        assert cfg.algorithm.learning_rate == baseline.algorithm.learning_rate
        assert cfg.num_steps_per_env == baseline.num_steps_per_env
        assert cfg.experiment_name == baseline.experiment_name + "_multi_critic"
        assert cfg.class_name == "MultiCriticRunner"
        assert cfg.to_dict()["multi_critic_recurrent"] is False
        expected_group = "privileged" if "Privileged" in cls.__name__ or "OriginalDecap" in cls.__name__ else "policy"
        assert cfg.obs_groups["policy"] == [expected_group]
        assert cfg.multi_critic_groups == [cfg.obs_groups["critic"], cfg.obs_groups["critic"]]
        flat = sum(cfg.multi_critic_reward_term_groups, [])
        assert len(flat) == len(set(flat))
        expected = declared_arm_rewards("d1" if family == "Go2-D1" else "b2_z1")
        assert expected <= set(flat), (registration["id"], expected - set(flat))
    assert matched == 6


def test_b2_missing_dataset_cannot_fall_back_to_go2_reference():
    tree = ast.parse((APEX / "config/go2/flat_b2_z1_arm_tracker_env_cfg.py").read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "UnitreeB2Z1ArmApexFlatTrackerEnvCfg")
    assignments = {ast.unparse(t): n.value for n in ast.walk(cls) if isinstance(n, ast.Assign) for t in n.targets}
    assert ast.unparse(assignments["self.commands.motion.motion_file"]) == "b2_z1_motion_glob"


def test_g1_apex_preserves_ppo_settings_and_has_complete_reward_groups(agent_configs, monkeypatch):
    module_name = "Gurukul.tasks.manager_based.go2_apex.config.go2.agents.rsl_rl_multi_critic_cfg"
    monkeypatch.setitem(sys.modules, module_name, agent_configs)
    path = APEX.parent / "beyondmimic/config/g1/agents/rsl_rl_ppo_cfg.py"
    spec = importlib.util.spec_from_file_location("_apex_test_g1_agents", path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    cfg = module.UnitreeG1ApexFlatMultiCriticRunnerCfg()
    baseline = module.UnitreeG1BeyondMimicFlatPPORunnerCfg()
    assert cfg.algorithm.entropy_coef == baseline.algorithm.entropy_coef == 0.005
    assert cfg.max_iterations == baseline.max_iterations == 30000
    assert cfg.policy.actor_hidden_dims == baseline.policy.actor_hidden_dims
    assert cfg.obs_groups == {"policy": ["policy"], "critic": ["critic"]}
    rewards = ast.parse((APEX.parent / "beyondmimic/tracking_env_cfg.py").read_text())
    cls = next(n for n in rewards.body if isinstance(n, ast.ClassDef) and n.name == "RewardsCfg")
    names = {n.targets[0].id for n in cls.body if isinstance(n, ast.Assign) and isinstance(n.value, ast.Call)}
    grouped = sum(cfg.multi_critic_reward_term_groups, [])
    assert names == set(grouped) and len(grouped) == len(set(grouped))
    registry = (APEX.parent / "beyondmimic/config/g1/__init__.py").read_text()
    assert registry.count('"rsl_rl_multi_critic_cfg_entry_point"') == 2
    assert registry.count('entry_point="Gurukul.tasks.manager_based.go2_apex.env:ApexManagerBasedRLEnv"') == 2
