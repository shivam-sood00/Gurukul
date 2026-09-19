"""Test playback input routing without launching Isaac Sim."""

import argparse
import ast
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

PLAY_SCRIPT = Path(__file__).resolve().parents[1] / "scripts/reinforcement_learning/rsl_rl/play.py"


@pytest.fixture
def helpers(monkeypatch):
    tree = ast.parse(PLAY_SCRIPT.read_text())
    names = {
        "_set_attr_if_exists",
        "_scale_gamepad_velocity",
        "_build_velocity_observation",
        "_configure_gamepad_velocity_control",
    }
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    controller = Mock()
    controller.advance.return_value = torch.zeros(3)
    namespace = {
        "torch": torch,
        "Se2Keyboard": Mock,
        "Se2Gamepad": Mock,
        "Se2GamepadCfg": SimpleNamespace,
        "args_cli": SimpleNamespace(gamepad_deadzone=0.08),
    }
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PLAY_SCRIPT), "exec"), namespace)
    namespace["Se2Gamepad"] = Mock(return_value=controller)
    omni = ModuleType("omni")
    omni.appwindow = ModuleType("omni.appwindow")
    window = SimpleNamespace(get_gamepad=Mock(return_value=object()))
    omni.appwindow.get_default_app_window = lambda: window
    monkeypatch.setitem(sys.modules, "omni", omni)
    monkeypatch.setitem(sys.modules, "omni.appwindow", omni.appwindow)
    return namespace, controller, window


@pytest.fixture
def env_cfg():
    def term():
        return SimpleNamespace(
            func=object(), params={"command_name": "base_velocity"}, scale=2.0, clip=(-100.0, 100.0), history_length=15
        )

    return SimpleNamespace(
        scene=SimpleNamespace(num_envs=64),
        terminations=SimpleNamespace(time_out=object(), fall=object()),
        commands=SimpleNamespace(
            base_velocity=SimpleNamespace(
                ranges=SimpleNamespace(lin_vel_x=(-1.0, 1.5), lin_vel_y=(-0.5, 0.5), ang_vel_z=(-1.0, 1.0)),
                heading_command=True,
                rel_heading_envs=1.0,
                rel_standing_envs=0.1,
                resampling_time_range=(7.5, 7.5),
                debug_vis=False,
            )
        ),
        observations=SimpleNamespace(
            policy=SimpleNamespace(velocity_commands=term()),
            critic=SimpleNamespace(velocity_commands=term()),
            amp=SimpleNamespace(),
            disabled=SimpleNamespace(velocity_commands=None),
        ),
    )


@pytest.mark.parametrize(
    "sticks,expected",
    [
        ([0.0, 0.0, 0.0], [0.0, 0.0, 0.0]),
        ([1.0, 1.0, 1.0], [1.5, -0.5, -1.0]),
        ([-1.0, -1.0, -1.0], [-1.0, 0.5, 1.0]),
        ([0.5, -0.5, 0.5], [0.75, 0.25, -0.5]),
        ([-0.5, 0.5, -0.5], [-0.5, -0.25, 0.5]),
        ([2.0, -2.0, 2.0], [1.5, 0.5, -1.0]),
    ],
)
def test_stick_direction_and_asymmetric_training_limits(helpers, env_cfg, sticks, expected):
    namespace, _, _ = helpers
    command = namespace["_scale_gamepad_velocity"](torch.tensor([sticks]), env_cfg.commands.base_velocity.ranges)
    torch.testing.assert_close(command, torch.tensor([expected]))


def test_gamepad_reaches_actor_critic_and_arrow_and_stops_on_release(helpers, env_cfg):
    namespace, controller, window = helpers
    fall = env_cfg.terminations.fall
    namespace["_configure_gamepad_velocity_control"](env_cfg)

    window.get_gamepad.assert_called_once_with(0)
    controller.reset.assert_called_once()
    config = namespace["Se2Gamepad"].call_args.args[0]
    assert config.dead_zone == 0.08
    assert (config.v_x_sensitivity, config.v_y_sensitivity, config.omega_z_sensitivity) == (1.0, 1.0, 1.0)
    assert env_cfg.scene.num_envs == 1
    assert env_cfg.terminations.time_out is None
    assert env_cfg.terminations.fall is fall
    command_cfg = env_cfg.commands.base_velocity
    assert not command_cfg.heading_command
    assert command_cfg.rel_heading_envs == command_cfg.rel_standing_envs == 0.0
    assert command_cfg.resampling_time_range == (1.0e6, 1.0e6)
    assert command_cfg.debug_vis

    policy = env_cfg.observations.policy.velocity_commands
    critic = env_cfg.observations.critic.velocity_commands
    for term in (policy, critic):
        assert term.scale == 2.0 and term.clip == (-100.0, 100.0) and term.history_length == 15
        assert term.params == {}
    assert policy.func is critic.func
    command_term = SimpleNamespace(vel_command_b=torch.full((1, 3), 9.0))
    env = SimpleNamespace(
        common_step_counter=0,
        device="cpu",
        num_envs=1,
        command_manager=SimpleNamespace(get_term=lambda _: command_term),
    )

    controller.advance.return_value = torch.tensor([1.0, -0.5, -1.0])
    expected = torch.tensor([[1.5, 0.25, 1.0]])
    torch.testing.assert_close(policy.func(env), expected)
    torch.testing.assert_close(critic.func(env), expected)
    torch.testing.assert_close(command_term.vel_command_b, expected)
    controller.advance.assert_called_once()

    # Centering the sticks must replace the prior command on the very next step.
    env.common_step_counter = 1
    controller.advance.return_value = torch.zeros(3)
    torch.testing.assert_close(policy.func(env), torch.zeros(1, 3))
    torch.testing.assert_close(command_term.vel_command_b, torch.zeros(1, 3))

    # Even a same-step reset of the manager buffer must not leave a sampled command visible.
    command_term.vel_command_b.fill_(9.0)
    torch.testing.assert_close(critic.func(env), torch.zeros(1, 3))
    torch.testing.assert_close(command_term.vel_command_b, torch.zeros(1, 3))
    assert controller.advance.call_count == 2


def test_keyboard_velocity_and_optional_posture_are_preserved(helpers):
    namespace, controller, _ = helpers
    controller.advance.return_value = torch.tensor([0.5, -0.25, 0.75])
    term = SimpleNamespace(vel_command_b=torch.zeros(2, 3), posture_command=torch.ones(2, 2))
    env = SimpleNamespace(
        common_step_counter=torch.tensor(0),
        device="cpu",
        num_envs=2,
        command_manager=SimpleNamespace(get_term=lambda _: term),
    )
    command = namespace["_build_velocity_observation"](controller)(env)
    torch.testing.assert_close(command, torch.tensor([[0.5, -0.25, 0.75, 1.0, 1.0]]).expand(2, -1))
    torch.testing.assert_close(term.vel_command_b, command[:, :3])


@pytest.mark.parametrize("missing", ["command", "observations", "gamepad", "window"])
def test_unsupported_task_or_missing_device_fails_clearly(helpers, env_cfg, missing, monkeypatch):
    namespace, _, window = helpers
    if missing == "command":
        env_cfg.commands.base_velocity = None
    elif missing == "observations":
        env_cfg.observations = SimpleNamespace()
    elif missing == "gamepad":
        window.get_gamepad.return_value = None
    else:
        monkeypatch.setattr(sys.modules["omni.appwindow"], "get_default_app_window", lambda: None)
    with pytest.raises((ValueError, RuntimeError), match="gamepad"):
        namespace["_configure_gamepad_velocity_control"](env_cfg)
    namespace["Se2Gamepad"].assert_not_called()


@pytest.mark.parametrize(
    "flags,error",
    [
        (["--gamepad"], None),
        (["--keyboard"], None),
        (["--keyboard", "--gamepad"], "not allowed"),
        (["--gamepad", "--headless"], "requires the Isaac Sim GUI"),
        (["--gamepad", "--velocity-demo"], "cannot be combined"),
        (["--gamepad", "--go2-d1-live-control"], "cannot be combined"),
        (["--gamepad", "--gamepad-deadzone", "1"], "must be in"),
        (["--gamepad", "--gamepad-deadzone", "-0.1"], "must be in"),
    ],
)
def test_cli_input_selection_and_validation(flags, error, monkeypatch, capsys):
    tree = ast.parse(PLAY_SCRIPT.read_text())
    nodes = []
    for node in tree.body:
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) in {"parser", "input_group"}:
            nodes.append(node)
        elif isinstance(node, ast.Expr) and isinstance(node.value, ast.Call):
            if ast.unparse(node.value.func) in {"parser.add_argument", "input_group.add_argument"}:
                nodes.append(node)
    namespace = {"argparse": argparse, "os": os}
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(PLAY_SCRIPT), "exec"), namespace)
    parser = namespace["parser"]
    parser.add_argument("--headless", action="store_true")
    monkeypatch.delenv("HEADLESS", raising=False)
    validation = next(
        node for node in tree.body if isinstance(node, ast.If) and ast.unparse(node.test) == "args_cli.gamepad"
    )

    def parse():
        namespace["args_cli"] = parser.parse_args(flags)
        exec(compile(ast.Module(body=[validation], type_ignores=[]), str(PLAY_SCRIPT), "exec"), namespace)

    if error:
        with pytest.raises(SystemExit) as exc:
            parse()
        assert exc.value.code == 2
        assert error in capsys.readouterr().err
    else:
        parse()
        assert namespace["args_cli"].gamepad == ("--gamepad" in flags)
