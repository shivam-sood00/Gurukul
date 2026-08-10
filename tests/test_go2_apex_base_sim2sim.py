from __future__ import annotations

import ast
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = REPO_ROOT / "unitree-sim2real/RL_policy_runner/sim2sim/run_rl_policy.py"
CURRENT_CONFIG_PATH = (
    REPO_ROOT
    / "unitree-sim2real/RL_policy_runner/configs/Gurukul/go2_apex_flat_v0.yaml"
)
LEGACY_CONFIG_PATH = (
    REPO_ROOT
    / "unitree-sim2real/RL_policy_runner/configs/Gurukul/go2_apex_flat_legacy_45.yaml"
)
APEX_ENV_CFG_PATH = (
    REPO_ROOT
    / "source/Gurukul/Gurukul/tasks/manager_based/go2_apex/tracking_env_cfg.py"
)


def _load_runner_contract_helpers():
    tree = ast.parse(RUNNER_PATH.read_text())
    helper_names = {
        "_write_go2_apex_base_observation",
        "_validate_policy_observation_width",
    }
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
    namespace = {"np": np, "torch": torch, "GO2_APEX_BASE_OBS_DIM": 46}
    exec(compile(ast.fix_missing_locations(module), str(RUNNER_PATH), "exec"), namespace)
    return tuple(namespace[name] for name in sorted(helper_names))


def _policy_observation_term_names() -> list[str]:
    tree = ast.parse(APEX_ENV_CFG_PATH.read_text())
    observations = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ObservationsCfg"
    )
    policy = next(
        node for node in observations.body if isinstance(node, ast.ClassDef) and node.name == "PolicyCfg"
    )
    return [
        target.id
        for node in policy.body
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    ]


def test_current_and_legacy_go2_apex_configs_are_explicitly_separate():
    current = yaml.safe_load(CURRENT_CONFIG_PATH.read_text())
    legacy = yaml.safe_load(LEGACY_CONFIG_PATH.read_text())

    assert current["observation_layout"] == "go2_apex_base"
    assert current["num_obs"] == 46
    assert current["ang_vel_scale"] == 0.25
    assert current["gravity_scale"] == 1.0
    assert current["dof_pos_scale"] == 1.0
    assert current["dof_vel_scale"] == 0.05
    assert current["cmd_scale"] == [1.0, 1.0, 1.0, 1.0]
    assert "<exported_run>" in current["policy_path"]

    assert legacy["observation_layout"] == "isaac_45"
    assert legacy["num_obs"] == 45
    assert legacy["gravity_scale"] == 0.0
    assert legacy["cmd_scale"] == [1.0, 1.0, 1.0]
    assert legacy["policy_path"].endswith(
        "2026-04-16_17-20-34_go2_apex_flat_ppo/exported/policy.onnx"
    )


def test_go2_apex_base_writer_matches_the_isaac_actor_order_and_scales():
    assert _policy_observation_term_names() == [
        "base_ang_vel",
        "projected_gravity",
        "command_vel_x",
        "command_vel_y",
        "command_vel_z",
        "command_yaw",
        "joint_pos",
        "joint_vel",
        "actions",
    ]
    validate_width, write_observation = _load_runner_contract_helpers()
    del validate_width

    base_ang_vel = np.arange(1, 4, dtype=np.float32)
    projected_gravity = np.arange(4, 7, dtype=np.float32)
    motion_command = np.arange(7, 11, dtype=np.float32)
    joint_pos = np.arange(11, 23, dtype=np.float32)
    joint_vel = np.arange(23, 35, dtype=np.float32)
    previous_action = np.arange(35, 47, dtype=np.float32)
    command_scale = np.array([1.5, 2.0, 2.5, 3.0], dtype=np.float32)
    output = torch.empty(46, dtype=torch.float32)

    written = write_observation(
        output,
        base_ang_vel=base_ang_vel,
        projected_gravity=projected_gravity,
        motion_command=motion_command,
        joint_pos=joint_pos,
        joint_vel=joint_vel,
        previous_action=previous_action,
        ang_vel_scale=0.25,
        gravity_scale=0.5,
        cmd_scale=command_scale,
        dof_pos_scale=0.1,
        dof_vel_scale=0.05,
    )
    expected = torch.from_numpy(
        np.concatenate(
            (
                base_ang_vel * 0.25,
                projected_gravity * 0.5,
                motion_command * command_scale,
                joint_pos * 0.1,
                joint_vel * 0.05,
                previous_action,
            )
        )
    )

    assert written == 46
    torch.testing.assert_close(output, expected)
    torch.testing.assert_close(output[6:10], torch.tensor([10.5, 16.0, 22.5, 30.0]))


def test_exported_policy_width_mismatch_fails_before_inference():
    validate_width, write_observation = _load_runner_contract_helpers()
    del write_observation

    validate_width([1, 46], 46)
    validate_width(["batch", "obs"], 46)
    with pytest.raises(ValueError, match="export expects 45.*config builds 46"):
        validate_width([1, 45], 46)
