from __future__ import annotations

import ast
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
BEYONDMIMIC = (
    REPO_ROOT
    / "source/Gurukul/Gurukul/tasks/manager_based/beyondmimic"
)
PM01_TASK = BEYONDMIMIC / "config/engineai_pm01_24dof"
REGISTRY = PM01_TASK / "__init__.py"
ENV_CFG = PM01_TASK / "loco_manipulation_env_cfg.py"
AMP_ENV = BEYONDMIMIC / "amp_env.py"
AMP_FEATURES = BEYONDMIMIC / "amp_features.py"
AMP_CFG = PM01_TASK / "agents/skrl_pm01_loco_manip_amp_cfg.yaml"
MOTION_COMMAND = BEYONDMIMIC / "mdp/commands.py"
NOTEBOOK = REPO_ROOT / "docs/pm01_loco_manipulation_walkthrough.ipynb"
SKRL_TRAIN = REPO_ROOT / "scripts/reinforcement_learning/skrl/train.py"
PM01_DOCS = REPO_ROOT / "website/docs/tasks/beyondmimic/pm01.md"


def _load_amp_features():
    spec = importlib.util.spec_from_file_location("pm01_amp_features", AMP_FEATURES)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pm01_loco_manipulation_python_and_notebook_are_well_formed():
    for path in (ENV_CFG, AMP_ENV, AMP_FEATURES):
        ast.parse(path.read_text())
    notebook = json.loads(NOTEBOOK.read_text())
    assert notebook["nbformat"] == 4
    assert any(cell["cell_type"] == "code" for cell in notebook["cells"])


def test_pm01_registers_distinct_tabletop_and_heavy_deepmimic_and_amp_tasks():
    registry = REGISTRY.read_text()
    expected_ids = (
        "Gurukul-Isaac-LocoManip-Tabletop-EngineAI-PM01-DeepMimic-v0",
        "Gurukul-Isaac-LocoManip-HeavyPush-EngineAI-PM01-DeepMimic-v0",
        "Gurukul-Isaac-LocoManip-Tabletop-EngineAI-PM01-AMP-v0",
        "Gurukul-Isaac-LocoManip-HeavyPush-EngineAI-PM01-AMP-v0",
    )
    for task_id in expected_ids:
        assert registry.count(task_id) == 1
    assert registry.count("beyondmimic.amp_env:BeyondMimicAmpEnv") == 2
    assert registry.count("skrl_amp_cfg_entry_point") == 2


def test_amp_configuration_keeps_core_amp_contract():
    cfg = yaml.safe_load(AMP_CFG.read_text())
    assert cfg["agent"]["class"] == "AMP"
    assert cfg["agent"]["task_reward_weight"] == 0.5
    assert cfg["agent"]["style_reward_weight"] == 0.5
    assert cfg["agent"]["discount_factor"] == 0.99
    assert cfg["agent"]["ratio_clip"] == 0.2
    assert cfg["agent"]["discriminator_gradient_penalty_scale"] == 10.0
    assert cfg["motion_dataset"]["memory_size"] >= 100_000
    assert cfg["reply_buffer"]["memory_size"] == 100_000


def test_skrl_amp_training_exposes_the_documented_wandb_cli():
    train_source = SKRL_TRAIN.read_text()
    docs_source = PM01_DOCS.read_text()

    assert '"--logger"' in train_source
    assert '"--wandb_project_name"' in train_source
    assert 'experiment_cfg["wandb"] = args_cli.logger == "wandb"' in train_source
    assert 'wandb_kwargs["project"] = args_cli.wandb_project_name' in train_source
    assert docs_source.count("--logger wandb") == 4
    assert docs_source.count("--wandb_project_name Gurukul-EngineAI-PM01-LocoManip") == 4


def test_amp_features_are_phase_free_velocity_bearing_two_frame_states():
    features = _load_amp_features()
    batch_size, num_joints, num_bodies = 3, 24, 13
    joint_pos = torch.zeros(batch_size, num_joints)
    joint_vel = torch.ones(batch_size, num_joints)
    root_pos = torch.tensor([[0.0, 0.0, 0.82]]).repeat(batch_size, 1)
    root_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]]).repeat(batch_size, 1)
    root_lin_vel = torch.tensor([[0.2, -0.1, 0.0]]).repeat(batch_size, 1)
    root_ang_vel = torch.tensor([[0.0, 0.0, 0.3]]).repeat(batch_size, 1)
    key_body_pos = torch.zeros(batch_size, num_bodies, 3)

    observation = features.build_amp_observation(
        joint_pos,
        joint_vel,
        root_pos,
        root_quat,
        root_lin_vel,
        root_ang_vel,
        key_body_pos,
    )
    one_frame_size = 2 * num_joints + 1 + 6 + 3 + 3 + 3 * num_bodies
    assert observation.shape == (batch_size, one_frame_size)
    assert torch.all(torch.isfinite(observation))

    env_source = AMP_ENV.read_text()
    cfg_source = ENV_CFG.read_text()
    assert "amp_num_observations = 2" in cfg_source
    assert "amp_history_frame_indices" in env_source
    assert "root_lin_vel" in AMP_FEATURES.read_text()
    assert "root_ang_vel" in AMP_FEATURES.read_text()
    assert "progress" not in AMP_FEATURES.read_text()
    assert "object" not in AMP_FEATURES.read_text().lower()


def test_amp_features_are_invariant_to_global_heading():
    features = _load_amp_features()
    joint_state = torch.zeros(1, 24)
    root_pos = torch.tensor([[0.0, 0.0, 0.82]])
    identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    yaw_90 = torch.tensor([[2.0**-0.5, 0.0, 0.0, 2.0**-0.5]])
    zero_ang_vel = torch.zeros(1, 3)
    key_body_x = torch.tensor([[[1.0, 0.0, 0.82]]])
    key_body_y = torch.tensor([[[0.0, 1.0, 0.82]]])

    facing_x = features.build_amp_observation(
        joint_state,
        joint_state,
        root_pos,
        identity,
        torch.tensor([[1.0, 0.0, 0.0]]),
        zero_ang_vel,
        key_body_x,
    )
    facing_y = features.build_amp_observation(
        joint_state,
        joint_state,
        root_pos,
        yaw_90,
        torch.tensor([[0.0, 1.0, 0.0]]),
        zero_ang_vel,
        key_body_y,
    )
    torch.testing.assert_close(facing_x, facing_y, atol=1.0e-5, rtol=1.0e-5)


def test_amp_reset_history_uses_preceding_reference_frames():
    features = _load_amp_features()
    latest = torch.tensor([5, 0])
    frame_indices = features.amp_history_frame_indices(
        latest, history_length=2, total_frames=8, frame_stride=1
    )
    torch.testing.assert_close(frame_indices, torch.tensor([[5, 4], [0, 0]]))
    strided_indices = features.amp_history_frame_indices(
        torch.tensor([5]), history_length=3, total_frames=8, frame_stride=2
    )
    torch.testing.assert_close(strided_indices, torch.tensor([[5, 3, 1]]))

    current = torch.tensor([[50.0, 51.0], [60.0, 61.0]])
    reference = torch.tensor(
        [
            [[5.0, 5.1], [4.0, 4.1]],
            [[0.0, 0.1], [0.0, 0.1]],
        ]
    )
    initialized = features.initialize_amp_history_from_reference(current, reference)
    torch.testing.assert_close(initialized[:, 0], current)
    torch.testing.assert_close(initialized[:, 1], reference[:, 1])
    assert not torch.equal(initialized[0, 0], initialized[0, 1])

    env_source = AMP_ENV.read_text()
    assert "self._amp_reset_reference_steps[env_ids] = self._amp_motion.time_steps[env_ids]" in env_source
    assert "initialize_amp_history_from_reference" in env_source
    assert "motion_fps * float(self.step_dt)" in env_source


@pytest.mark.skipif(
    not (PM01_TASK / "motion/dance.npz").exists(),
    reason="Optional EngineAI motion archives are not distributed.",
)
def test_pm01_motion_archives_cover_amp_joint_and_body_contracts():
    expected_key_bodies = {
        "LINK_BASE",
        "LINK_ANKLE_ROLL_L",
        "LINK_ANKLE_ROLL_R",
        "LINK_ELBOW_YAW_L",
        "LINK_ELBOW_YAW_R",
        "LINK_HEAD_YAW",
    }
    for motion_name in ("dance.npz", "walking_24dof.npz"):
        with np.load(PM01_TASK / "motion" / motion_name, allow_pickle=False) as motion:
            assert motion["joint_pos"].shape[1] == 24
            assert motion["joint_vel"].shape == motion["joint_pos"].shape
            assert motion["body_pos_w"].shape[1] == 29
            if "body_names" in motion.files:
                assert expected_key_bodies <= set(motion["body_names"].tolist())
            assert np.all(np.isfinite(motion["body_lin_vel_w"]))
            assert np.all(np.isfinite(motion["body_ang_vel_w"]))


def test_heavy_task_randomizes_supported_box_properties_without_claiming_grasping():
    source = ENV_CFG.read_text()
    assert "object_mass = 10.0" in source
    assert "object_mass_scale_range = (0.60, 1.00)" in source
    assert "object_scale_range = (0.85, 1.15)" in source
    assert "randomize_object_material" in source
    assert "randomize_object_mass" in source
    assert "randomize_object_scale" in source
    assert "self.scene.replicate_physics = False" in source
    assert "object_velocity_towards_goal_exp" in source
    assert "base_to_object_standoff_exp" in source
    assert "self.commands.motion.make_in_place = True" in source
    assert "make_in_place: bool = False" in MOTION_COMMAND.read_text()
    assert "grasp" not in source.lower()
    assert "carry" not in source.lower()
