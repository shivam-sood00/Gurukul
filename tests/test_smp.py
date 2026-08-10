# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import ast
import importlib.util
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SMP_ROOT = REPO_ROOT / "source/Gurukul/Gurukul/tasks/manager_based/smp"
PROFILES = SMP_ROOT / "profiles.py"
FAMILY_INIT = SMP_ROOT / "__init__.py"
BUILD_DATASET = REPO_ROOT / "scripts/tools/smp/build_dataset.py"
PRETRAIN = REPO_ROOT / "scripts/reinforcement_learning/smp/pretrain.py"
TRAIN = REPO_ROOT / "scripts/reinforcement_learning/rsl_rl/train.py"
PLAY = REPO_ROOT / "scripts/reinforcement_learning/rsl_rl/play.py"
G1_CONVERTER = REPO_ROOT / "scripts/tools/beyondmimic/csv_to_npz.py"
PACKAGE_SETUP = REPO_ROOT / "source/Gurukul/setup.py"
DOCS = REPO_ROOT / "website/docs/training-methods/score-matching-motion-priors.md"
TASK_STATUS = REPO_ROOT / "website/docs/reference/task-status.md"

TASK_IDS = (
    "Gurukul-Isaac-SMP-Velocity-Flat-Unitree-G1-v0",
    "Gurukul-Isaac-SMP-Velocity-Flat-EngineAI-PM01-v0",
    "Gurukul-Isaac-SMP-Velocity-Flat-Unitree-Go2-v0",
)

TASK_REGISTRIES = {
    TASK_IDS[0]: SMP_ROOT / "config/unitree_g1/__init__.py",
    TASK_IDS[1]: SMP_ROOT / "config/engineai_pm01/__init__.py",
    TASK_IDS[2]: SMP_ROOT / "config/unitree_go2/__init__.py",
}

TASK_ENTRY_POINTS = {
    TASK_IDS[0]: (
        ".flat_env_cfg:UnitreeG1SmpFlatEnvCfg",
        ".rsl_rl_ppo_cfg:UnitreeG1SmpFlatPPORunnerCfg",
    ),
    TASK_IDS[1]: (
        ".flat_env_cfg:EngineAiPm01SmpFlatEnvCfg",
        ".rsl_rl_ppo_cfg:EngineAiPm01SmpFlatPPORunnerCfg",
    ),
    TASK_IDS[2]: (
        ".flat_env_cfg:UnitreeGo2SmpFlatEnvCfg",
        ".rsl_rl_ppo_cfg:UnitreeGo2SmpFlatPPORunnerCfg",
    ),
}


def _load_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def test_profiles_are_morphology_specific_and_have_canonical_dimensions():
    profiles = _load_module("smp_profiles_contract_test", PROFILES)

    expected = {
        "g1": (29, 5, 59),
        "pm01": (24, 5, 54),
        "go2": (12, 4, 39),
    }
    assert tuple(profile.name for profile in profiles.profiles()) == tuple(expected)
    for name, (num_joints, num_key_bodies, feature_dim) in expected.items():
        profile = profiles.get_profile(name)
        assert profile.num_joints == num_joints
        assert profile.num_key_bodies == num_key_bodies
        assert profile.feature_dim == feature_dim
        assert profile.control_fps == pytest.approx(50.0)
        assert len(profile.joint_names) == len(set(profile.joint_names))
        assert len(profile.key_body_names) == len(set(profile.key_body_names))
        metadata = profile.to_metadata()
        assert metadata["history_order"] == "oldest_to_newest"
        assert metadata["quaternion_order"] == "wxyz"
        assert metadata["rotation_6d_columns"] == "x_z"
        assert metadata["up_axis"] == "z"
        assert metadata["canonical_anchor"] == "newest_root_xy_and_yaw"
        assert metadata["root_height"] == "absolute"
        assert metadata["key_body_frame"] == "root_relative_newest_heading"
        assert metadata["root_pose_source_frame"] == "link_actor"
        assert metadata["root_velocity_source_frame"] == "link_actor"

    assert profiles.get_profile("unitree-g1").name == "g1"
    assert profiles.get_profile("engineai_pm01").name == "pm01"
    assert profiles.get_profile("unitree go2").name == "go2"


def test_profile_metadata_validation_rejects_shape_compatible_schema_changes():
    profiles = _load_module("smp_profiles_metadata_test", PROFILES)
    metadata = profiles.get_profile("g1").to_metadata()
    metadata["joint_names"][0], metadata["joint_names"][1] = (
        metadata["joint_names"][1],
        metadata["joint_names"][0],
    )
    with pytest.raises(ValueError, match="joint_names"):
        profiles.validate_profile_metadata(metadata, "g1")

    metadata = profiles.get_profile("g1").to_metadata()
    metadata["quaternion_order"] = "xyzw"
    with pytest.raises(ValueError, match="quaternion_order"):
        profiles.validate_profile_metadata(metadata, "g1")


def test_smp_task_ids_are_canonical_and_publicly_catalogued():
    docs = DOCS.read_text()
    task_status = TASK_STATUS.read_text()
    for task_id in TASK_IDS:
        assert TASK_REGISTRIES[task_id].read_text().count(task_id) == 1
        assert docs.count(task_id) >= 1
        assert task_status.count(task_id) == 1


def _joined_string_suffix(node: ast.AST) -> str:
    assert isinstance(node, ast.JoinedStr)
    return "".join(value.value for value in node.values if isinstance(value, ast.Constant))


def test_smp_task_ids_map_to_the_expected_robot_configs_and_runners():
    registrations = {}
    for registry_path in TASK_REGISTRIES.values():
        register_calls = []
        for node in ast.walk(ast.parse(registry_path.read_text())):
            if not (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and isinstance(node.func.value, ast.Name)
                and node.func.value.id == "gym"
                and node.func.attr == "register"
            ):
                continue
            register_calls.append(node)
            keywords = {keyword.arg: keyword.value for keyword in node.keywords}
            task_id = ast.literal_eval(keywords["id"])
            kwargs = {
                ast.literal_eval(key): value
                for key, value in zip(keywords["kwargs"].keys, keywords["kwargs"].values, strict=True)
            }
            registrations[task_id] = (
                _joined_string_suffix(kwargs["env_cfg_entry_point"]),
                _joined_string_suffix(kwargs["rsl_rl_cfg_entry_point"]),
            )
        assert len(register_calls) == 1

    assert registrations == TASK_ENTRY_POINTS


def test_smp_is_a_standalone_robot_config_family():
    family_tree = ast.parse(FAMILY_INIT.read_text())
    assert not any(
        isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and isinstance(node.func.value, ast.Name)
        and node.func.value.id == "gym"
        and node.func.attr == "register"
        for node in ast.walk(family_tree)
    )
    assert "paper2code" not in str(SMP_ROOT).lower()
    for registry_path in TASK_REGISTRIES.values():
        assert registry_path.is_file()
        assert (registry_path.parent / "flat_env_cfg.py").is_file()
        assert (registry_path.parent / "agents/rsl_rl_ppo_cfg.py").is_file()
    assert "packages=find_packages()" in "".join(PACKAGE_SETUP.read_text().split())


def test_documented_g1_exporter_emits_named_channels():
    source = G1_CONVERTER.read_text()
    assert '"joint_names": np.asarray(robot.joint_names' in source
    assert '"body_names": np.asarray(robot.body_names' in source


def test_rsl_rl_entry_points_apply_smp_cli_overrides():
    for script in (TRAIN, PLAY):
        source = script.read_text()
        assert source.count('"--smp-prior"') == 1
        assert source.count('"--smp-gsi-pool-size"') == 1
        assert source.count("def _configure_smp_prior(env_cfg)") == 1
        assert source.count("_configure_smp_prior(env_cfg)") == 2
        assert 'initialize_cfg.params["checkpoint_path"]' in source
        assert 'initialize_cfg.params["gsi_pool_size"]' in source
    assert 'initialize_cfg.params["update_score_normalizer"] = False' in PLAY.read_text()


def test_offline_scripts_parse_and_help_without_importing_isaac_lab():
    for script in (BUILD_DATASET, PRETRAIN):
        tree = ast.parse(script.read_text())
        top_level_imports = {alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names}
        top_level_from_imports = {
            node.module for node in tree.body if isinstance(node, ast.ImportFrom) and node.module is not None
        }
        assert not any(name.startswith("isaaclab") for name in top_level_imports)
        assert not any(name.startswith("Gurukul") for name in top_level_from_imports)

        completed = subprocess.run(
            [sys.executable, str(script), "--help"],
            check=False,
            capture_output=True,
            text=True,
        )
        assert completed.returncode == 0, completed.stderr
        assert "usage:" in completed.stdout.lower()


def test_public_docs_state_data_provenance_and_evidence_limits():
    docs = DOCS.read_text()
    prose = " ".join(docs.split())
    assert "licensed to use" in prose
    assert "Apache-2.0-licensed MimicKit" in prose
    assert "https://yxmu.foo/smp-page/" in docs
    assert "https://arxiv.org/abs/2512.03028" in docs
    assert "SUZ-tsinghua/smp" in docs
    assert "secondary implementation" in prose
    assert "quadruped extrapolation" in prose
    assert "does not bundle an SMP prior" in prose
    assert "oldest-to-newest" in prose
