import ast
from pathlib import Path
from types import SimpleNamespace

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
PLAY_SCRIPT = REPO_ROOT / "scripts/reinforcement_learning/rsl_rl/play.py"


def _load_play_config_helpers():
    tree = ast.parse(PLAY_SCRIPT.read_text())
    helper_names = {
        "_set_attr_if_exists",
        "_configure_play_terrain",
        "_disable_play_observation_corruption",
        "_velocity_demo_command",
    }
    helper_nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in helper_names]
    namespace = {}
    exec(compile(ast.Module(body=helper_nodes, type_ignores=[]), PLAY_SCRIPT, "exec"), namespace)
    return namespace


def test_play_configuration_accepts_direct_environment_without_manager_terms():
    helpers = _load_play_config_helpers()
    direct_env_cfg = SimpleNamespace(scene=SimpleNamespace())

    helpers["_configure_play_terrain"](direct_env_cfg)
    helpers["_disable_play_observation_corruption"](direct_env_cfg)


def test_play_configuration_preserves_manager_based_terrain_overrides():
    helpers = _load_play_config_helpers()
    terrain_generator = SimpleNamespace(num_rows=10, num_cols=20, curriculum=True)
    terrain = SimpleNamespace(max_init_terrain_level=3, terrain_generator=terrain_generator)
    policy_observations = SimpleNamespace(enable_corruption=True)
    manager_env_cfg = SimpleNamespace(
        scene=SimpleNamespace(terrain=terrain),
        observations=SimpleNamespace(policy=policy_observations),
    )

    helpers["_configure_play_terrain"](manager_env_cfg)
    helpers["_disable_play_observation_corruption"](manager_env_cfg)

    assert terrain.max_init_terrain_level is None
    assert terrain_generator.num_rows == 5
    assert terrain_generator.num_cols == 5
    assert terrain_generator.curriculum is False
    assert policy_observations.enable_corruption is False


def test_velocity_demo_uses_configured_extrema_in_video_order():
    command = _load_play_config_helpers()["_velocity_demo_command"]
    ranges = SimpleNamespace(lin_vel_x=(-1.0, 1.5), ang_vel_z=(-1.0, 1.0))

    assert command(ranges, 0, 800) == ("ready", (0.0, 0.0, 0.0))
    assert command(ranges, 40, 800) == ("fast_forward", (1.5, 0.0, 0.0))
    assert command(ranges, 264, 800) == ("forward_left_arc", pytest.approx((1.05, 0.0, 1.0)))
    assert command(ranges, 364, 800) == ("forward_right_arc", pytest.approx((1.05, 0.0, -1.0)))
    assert command(ranges, 464, 800) == ("rotate_left", (0.0, 0.0, 1.0))
    assert command(ranges, 524, 800) == ("rotate_right", (0.0, 0.0, -1.0))
    assert command(ranges, 584, 800) == ("fast_backward", (-1.0, 0.0, 0.0))
    assert command(ranges, 760, 800) == ("stop", (0.0, 0.0, 0.0))
