"""Check pre-launch gamepad mapping setup without starting Isaac Sim."""

import ast
import ctypes
import importlib.util
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

RL_SCRIPTS = Path(__file__).resolve().parents[1] / "scripts/reinforcement_learning"


@pytest.fixture
def compat(monkeypatch, tmp_path):
    path = RL_SCRIPTS / "gamepad_compat.py"
    spec = importlib.util.spec_from_file_location("gamepad_compat", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module.sys, "platform", "linux")
    monkeypatch.setenv("DISPLAY", ":0")
    monkeypatch.setenv("CARB_APP_PATH", str(tmp_path))
    monkeypatch.delenv("HEADLESS", raising=False)
    library = SimpleNamespace(glfwInit=Mock(return_value=1), glfwUpdateGamepadMappings=Mock(return_value=1))
    loader = Mock(return_value=library)
    monkeypatch.setattr(module.ctypes, "CDLL", loader)
    return module, library, loader


def test_mapping_is_loaded_into_kits_library_before_enumeration(compat, tmp_path):
    module, library, loader = compat
    library.glfwUpdateGamepadMappings.side_effect = lambda mapping: int(library.glfwInit.called)

    assert module.prepare_gamepad_mappings()

    loader.assert_called_once_with(str(tmp_path / "kernel/plugins/libcarb.windowing-glfw.plugin.so"))
    library.glfwInit.assert_called_once_with()
    library.glfwUpdateGamepadMappings.assert_called_once_with(module.XBOX_SERIES_USB_MAPPING.encode("ascii"))
    assert library.glfwUpdateGamepadMappings.argtypes == [ctypes.c_char_p]
    assert library.glfwUpdateGamepadMappings.restype is ctypes.c_int
    assert module._glfw_libraries == {tmp_path / "kernel/plugins/libcarb.windowing-glfw.plugin.so": library}


@pytest.mark.parametrize("extension_root", ["extscache", "exts", "kit/exts"])
def test_extension_and_kernel_copies_both_receive_mapping(compat, monkeypatch, tmp_path, extension_root):
    module, kernel, loader = compat
    kit_path = tmp_path / "kit"
    monkeypatch.setenv("CARB_APP_PATH", str(kit_path))
    extension_path = (
        tmp_path / extension_root / "carb.windowing.plugins-1.0.0+build/bin/libcarb.windowing-glfw.plugin.so"
    )
    extension_path.parent.mkdir(parents=True)
    extension_path.touch()
    extension = SimpleNamespace(glfwInit=Mock(return_value=1), glfwUpdateGamepadMappings=Mock(return_value=1))
    loader.side_effect = [kernel, extension]

    assert module.prepare_gamepad_mappings()

    assert [call.args[0] for call in loader.call_args_list] == [
        str(kit_path / "kernel/plugins/libcarb.windowing-glfw.plugin.so"),
        str(extension_path),
    ]
    for library in (kernel, extension):
        library.glfwInit.assert_called_once_with()
        library.glfwUpdateGamepadMappings.assert_called_once_with(module.XBOX_SERIES_USB_MAPPING.encode("ascii"))
    assert module._glfw_libraries[extension_path] is extension


def test_missing_kernel_copy_does_not_skip_extension_plugin(compat, monkeypatch, tmp_path):
    module, extension, loader = compat
    monkeypatch.setenv("CARB_APP_PATH", str(tmp_path / "kit"))
    plugin = tmp_path / "extscache/carb.windowing.plugins-1.0.0/bin/libcarb.windowing-glfw.plugin.so"
    plugin.parent.mkdir(parents=True)
    plugin.touch()
    loader.side_effect = [OSError("kernel plugin not installed"), extension]

    assert module.prepare_gamepad_mappings()
    extension.glfwUpdateGamepadMappings.assert_called_once_with(module.XBOX_SERIES_USB_MAPPING.encode("ascii"))


@pytest.mark.parametrize("skip_reason", ["headless_arg", "headless_env", "windows", "no_display", "no_kit_path"])
def test_non_gui_runs_do_not_load_or_initialize_glfw(compat, monkeypatch, skip_reason):
    module, _, loader = compat
    if skip_reason == "headless_env":
        monkeypatch.setenv("HEADLESS", "1")
    elif skip_reason == "windows":
        monkeypatch.setattr(module.sys, "platform", "win32")
    elif skip_reason == "no_display":
        monkeypatch.delenv("DISPLAY")
    elif skip_reason == "no_kit_path":
        monkeypatch.delenv("CARB_APP_PATH")

    assert not module.prepare_gamepad_mappings(headless=skip_reason == "headless_arg")
    loader.assert_not_called()


@pytest.mark.parametrize("failure", ["missing_library", "missing_symbol", "no_display_connection", "rejected_mapping"])
def test_optional_compatibility_failure_does_not_abort_simulator_startup(compat, capsys, failure):
    module, library, loader = compat
    if failure == "missing_library":
        loader.side_effect = OSError("missing plugin")
    elif failure == "missing_symbol":
        del library.glfwUpdateGamepadMappings
    elif failure == "no_display_connection":
        library.glfwInit.return_value = 0
    else:
        library.glfwUpdateGamepadMappings.return_value = 0

    assert not module.prepare_gamepad_mappings()
    assert "[WARN]" in capsys.readouterr().out
    if failure == "no_display_connection":
        library.glfwUpdateGamepadMappings.assert_not_called()


@pytest.mark.parametrize(
    "script",
    [
        "rsl_rl/play.py",
        "rsl_rl/play_cs.py",
        "rsl_rl/play_with_depth.py",
        "rsl_rl/train.py",
        "rsl_rl/eval_student.py",
        "cusrl/play.py",
        "cusrl/train.py",
        "skrl/play.py",
        "skrl/train.py",
    ],
)
def test_rl_entry_points_register_mapping_before_app_launcher(script):
    tree = ast.parse((RL_SCRIPTS / script).read_text())
    calls = [node for node in ast.walk(tree) if isinstance(node, ast.Call)]
    prepare = next(
        node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "prepare_gamepad_mappings"
    )
    launch = next(node for node in calls if isinstance(node.func, ast.Name) and node.func.id == "AppLauncher")
    parse = next(
        node for node in calls if isinstance(node.func, ast.Attribute) and node.func.attr == "parse_known_args"
    )
    assert parse.lineno < prepare.lineno < launch.lineno
    assert ast.unparse(prepare.keywords[0].value) == "args_cli.headless"
