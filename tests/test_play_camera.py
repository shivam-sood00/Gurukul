"""Playback camera checks that do not launch Isaac Sim."""

import ast
import sys
import weakref
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
RL_SCRIPTS = REPO_ROOT / "scripts/reinforcement_learning"


def _load_functions(path, names, **namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), "exec"), namespace)
    return namespace


@pytest.fixture
def camera_helpers():
    # These tests use an identity robot orientation.
    math_utils = SimpleNamespace(transform_points=lambda points, pos, quat: points + pos)
    return _load_functions(
        RL_SCRIPTS / "rl_utils.py",
        {"_MouseSteerableCamera", "enable_mouse_camera", "enable_free_camera", "camera_follow"},
        torch=torch,
        math_utils=math_utils,
        np=np,
        weakref=weakref,
    )


class ViewportController:
    """Small viewport double with Isaac Lab's asset-relative framing and render callback behavior."""

    def __init__(self, origin_type):
        self.cfg = SimpleNamespace(origin_type=origin_type, asset_name="robot", body_name="torso", env_index=0)
        self.eye_offset = np.array([1.5, -2.0, 1.5])
        self.target_offset = np.array([0.2, 0.0, 0.4])
        self.origin = np.zeros(3)
        self.asset_position = np.array([10.0, 20.0, 0.6])
        if origin_type == "env":
            self.origin = np.array([10.0, 20.0, 0.0])
        self.update_view_location()

    @property
    def default_cam_eye(self):
        return self.eye_offset

    @property
    def default_cam_lookat(self):
        return self.target_offset

    @property
    def viewer_origin(self):
        return torch.from_numpy(self.origin)

    def update_view_location(self, eye=None, lookat=None):
        if eye is not None:
            self.eye_offset = np.asarray(eye)
        if lookat is not None:
            self.target_offset = np.asarray(lookat)
        self.eye = self.origin + self.eye_offset
        self.target = self.origin + self.target_offset

    def update_view_to_world(self):
        self.cfg.origin_type = "world"
        self.origin = np.zeros(3)
        self.update_view_location()

    def update_view_to_asset_root(self, asset_name):
        assert asset_name == "robot"
        self.origin = self.asset_position.copy()
        self.update_view_location()

    def update_view_to_asset_body(self, asset_name, body_name):
        assert asset_name == "robot"
        assert body_name == "torso"
        self.origin = self.asset_position.copy()
        self.origin[2] += 0.5
        self.update_view_location()

    def set_view_env_index(self, env_index):
        self.cfg.env_index = env_index

    def render(self):
        if self.cfg.origin_type == "asset_root":
            self.update_view_to_asset_root(self.cfg.asset_name)
        elif self.cfg.origin_type == "asset_body":
            self.update_view_to_asset_body(self.cfg.asset_name, self.cfg.body_name)


class CameraState:
    def __init__(self, controller):
        self.controller = weakref.proxy(controller)

    @property
    def position_world(self):
        return self.controller.eye

    @property
    def target_world(self):
        return self.controller.target


@pytest.mark.parametrize("origin_type", ["asset_root", "asset_body"])
def test_tracking_preserves_default_view_then_follows_with_mouse_edits(camera_helpers, origin_type):
    controller = ViewportController(origin_type)
    controller.render()
    initial_eye, initial_target = controller.eye.copy(), controller.target.copy()
    controller.update_view_location = camera_helpers["_MouseSteerableCamera"](controller, CameraState(controller))

    # Tracking without mouse input retains the authored offsets.
    displacement = np.array([2.0, -1.0, 0.25])
    controller.asset_position += displacement
    controller.render()
    np.testing.assert_allclose(controller.eye, initial_eye + displacement)
    np.testing.assert_allclose(controller.target, initial_target + displacement)

    # An orbit/pan/zoom changes the live camera; moving and resetting must preserve that framing.
    controller.eye += np.array([-3.0, 2.0, 1.0])
    controller.target += np.array([0.5, 0.0, 0.2])
    for displacement in (np.zeros(3), np.array([1.0, 2.0, 0.0]), np.array([-3.0, -1.0, -0.25])):
        expected_eye, expected_target = controller.eye + displacement, controller.target + displacement
        controller.asset_position += displacement
        controller.render()
        controller.render()  # Repeated renders must not accumulate the mouse adjustment again.
        np.testing.assert_allclose(controller.eye, expected_eye)
        np.testing.assert_allclose(controller.target, expected_target)


@pytest.mark.parametrize("origin_type", ["world", "env", "asset_root", "asset_body"])
def test_free_camera_preserves_framing_and_mouse_changes_after_robot_moves(camera_helpers, origin_type):
    controller = ViewportController(origin_type)
    controller.render()
    initial_eye, initial_target = controller.eye.copy(), controller.target.copy()
    env = SimpleNamespace(unwrapped=SimpleNamespace(viewport_camera_controller=controller))

    camera_helpers["enable_free_camera"](env)
    np.testing.assert_allclose(controller.eye, initial_eye)
    np.testing.assert_allclose(controller.target, initial_target)

    # A mouse orbit/pan/zoom edits the live viewport, not the controller's configured offsets.
    controller.eye = np.array([7.0, -3.0, 4.0])
    controller.target = np.array([1.0, 2.0, 0.5])
    for position in ([12.0, 22.0, 0.6], [10.0, 20.0, 0.6]):
        controller.asset_position = np.array(position)
        controller.render()
        np.testing.assert_allclose(controller.eye, [7.0, -3.0, 4.0])
        np.testing.assert_allclose(controller.target, [1.0, 2.0, 0.5])


@pytest.mark.parametrize("offset", [None, (1.5, 1.5, 1.5), (3.0, 3.0, 3.0)])
@pytest.mark.parametrize("mode", ["follow", "isometric", "topdown"])
def test_scripted_follow_preserves_starting_angle_and_mouse_framing(camera_helpers, offset, mode):
    controller = ViewportController("world")
    controller.update_view_location = camera_helpers["_MouseSteerableCamera"](controller, CameraState(controller))
    robot_data = SimpleNamespace(
        root_pos_w=torch.tensor([[10.0, 20.0, 0.6]]), root_quat_w=torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    )
    env = SimpleNamespace(
        device="cpu",
        unwrapped=SimpleNamespace(
            viewport_camera_controller=controller, scene={"robot": SimpleNamespace(data=robot_data)}
        ),
    )

    follow = camera_helpers["camera_follow"]
    follow(env, mode=mode, window_size=1, follow_offset=offset)

    initial_offset = {"follow": offset or (-3.0, 0.0, 0.5), "isometric": (-3.5, 3.5, 2.8), "topdown": (0.0, 0.0, 6.0)}[
        mode
    ]
    np.testing.assert_allclose(controller.eye, robot_data.root_pos_w[0].numpy() + initial_offset)
    np.testing.assert_allclose(controller.target, robot_data.root_pos_w[0].numpy())
    assert controller.cfg.origin_type == "world"

    controller.eye += np.array([2.0, -1.0, 3.0])
    controller.target += np.array([0.5, -0.5, 0.0])
    expected_eye, expected_target = controller.eye + [1.0, 2.0, 0.0], controller.target + [1.0, 2.0, 0.0]
    robot_data.root_pos_w += torch.tensor([1.0, 2.0, 0.0])
    follow(env, mode=mode, window_size=1, follow_offset=offset)
    follow(env, mode=mode, window_size=1, follow_offset=offset)
    np.testing.assert_allclose(controller.eye, expected_eye)
    np.testing.assert_allclose(controller.target, expected_target)


def test_camera_helpers_allow_headless_environment(camera_helpers):
    env = SimpleNamespace(unwrapped=SimpleNamespace(viewport_camera_controller=None))
    camera_helpers["enable_free_camera"](env)
    camera_helpers["enable_mouse_camera"](env)
    camera_helpers["camera_follow"](env)


def test_enable_mouse_camera_keeps_asset_tracking_and_is_idempotent(camera_helpers, monkeypatch):
    controller = ViewportController("asset_root")
    viewport = object()
    utility = ModuleType("omni.kit.viewport.utility")
    utility.get_active_viewport = lambda: viewport
    state_module = ModuleType("omni.kit.viewport.utility.camera_state")
    state_module.ViewportCameraState = Mock(return_value=CameraState(controller))
    monkeypatch.setitem(sys.modules, utility.__name__, utility)
    monkeypatch.setitem(sys.modules, state_module.__name__, state_module)
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(viewport_camera_controller=controller, sim=SimpleNamespace(has_gui=lambda: True))
    )

    camera_helpers["enable_mouse_camera"](env)
    adapter = controller.update_view_location
    assert isinstance(adapter, camera_helpers["_MouseSteerableCamera"])
    assert controller.cfg.origin_type == "asset_root"
    np.testing.assert_allclose(controller.eye, controller.asset_position + [1.5, -2.0, 1.5])
    state_module.ViewportCameraState.assert_called_once_with("/OmniverseKit_Persp", viewport)

    controller.eye += np.array([2.0, 1.0, 0.0])
    expected_eye = controller.eye.copy()
    camera_helpers["enable_mouse_camera"](env)
    assert controller.update_view_location is adapter
    np.testing.assert_allclose(controller.eye, expected_eye)


def test_headless_video_keeps_original_camera_controller(camera_helpers):
    controller = ViewportController("asset_root")
    original_update = controller.update_view_location
    env = SimpleNamespace(
        unwrapped=SimpleNamespace(viewport_camera_controller=controller, sim=SimpleNamespace(has_gui=lambda: False))
    )
    camera_helpers["enable_mouse_camera"](env)
    assert controller.update_view_location == original_update
    assert controller.cfg.origin_type == "asset_root"


def test_mouse_adapter_does_not_keep_controller_alive_after_close(camera_helpers):
    controller = ViewportController("world")
    reference = weakref.ref(controller)
    controller.update_view_location = camera_helpers["_MouseSteerableCamera"](controller, CameraState(controller))
    del controller
    assert reference() is None


@pytest.mark.parametrize("requested_mode", ["auto", "mouse", "none", "follow", "isometric", "topdown"])
@pytest.mark.parametrize("keyboard,gamepad", [(False, False), (True, False), (False, True)])
def test_main_play_keeps_default_tracking_and_allows_explicit_free_camera(requested_mode, keyboard, gamepad):
    path = RL_SCRIPTS / "rsl_rl/play.py"
    namespace = _load_functions(
        path,
        {"_resolve_camera_mode", "_configure_viewer_for_camera_mode"},
        args_cli=SimpleNamespace(
            camera_follow_mode=requested_mode,
            keyboard=keyboard,
            gamepad=gamepad,
            task="Gurukul-Test-Play-v0",
            camera_smooth_window=50,
            camera_follow_distance_scale=1.0,
        ),
    )
    initial_mode, explicit = namespace["_resolve_camera_mode"]()
    viewer = SimpleNamespace(origin_type="asset_root", asset_name="robot", body_name=None, eye=(1.5, 1.5, 1.5))
    namespace["_configure_viewer_for_camera_mode"](SimpleNamespace(viewer=viewer), initial_mode, explicit)
    assert viewer.eye == (1.5, 1.5, 1.5)
    if requested_mode in ("mouse", "none") or (requested_mode == "auto" and not (keyboard or gamepad)):
        assert viewer.origin_type == "asset_root"
    if requested_mode == "auto":
        assert initial_mode == ("follow" if keyboard or gamepad else "none")

    # Execute the actual post-reset camera setup from main, without importing the simulator or a policy.
    main = next(
        node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    setup = next(node for node in main.body if isinstance(node, ast.If) and "enable_free_camera" in ast.unparse(node))
    enable_free_camera = Mock()
    enable_mouse_camera = Mock()
    namespace.update(
        camera_follow_mode=initial_mode,
        camera_mode_requested=explicit,
        camera_follow_offset=None,
        enable_free_camera=enable_free_camera,
        enable_mouse_camera=enable_mouse_camera,
        env=object(),
    )
    exec(compile(ast.Module(body=[setup], type_ignores=[]), str(path), "exec"), namespace)
    if requested_mode in ("mouse", "none"):
        assert namespace["camera_follow_mode"] == "none"
        enable_free_camera.assert_called_once()
        enable_mouse_camera.assert_not_called()
    else:
        assert namespace["camera_follow_mode"] == initial_mode
        enable_free_camera.assert_not_called()
        enable_mouse_camera.assert_called_once()


@pytest.mark.parametrize("requested_mode", ["auto", "mouse", "none", "follow", "isometric", "topdown"])
def test_depth_play_keeps_following_by_default(requested_mode):
    path = RL_SCRIPTS / "rsl_rl/play_with_depth.py"
    main = next(
        node for node in ast.parse(path.read_text()).body if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    assignment = next(
        node
        for node in main.body
        if isinstance(node, ast.Assign) and ast.unparse(node.targets[0]) == "camera_follow_mode"
    )
    setup = next(node for node in main.body if isinstance(node, ast.If) and "enable_free_camera" in ast.unparse(node))
    free_camera, mouse_camera = Mock(), Mock()
    namespace = {
        "args_cli": SimpleNamespace(camera_follow_mode=requested_mode, camera_smooth_window=50),
        "enable_free_camera": free_camera,
        "enable_mouse_camera": mouse_camera,
        "env": object(),
    }
    exec(compile(ast.Module(body=[assignment, setup], type_ignores=[]), str(path), "exec"), namespace)
    if requested_mode in ("mouse", "none"):
        assert namespace["camera_follow_mode"] == "none"
        free_camera.assert_called_once()
        mouse_camera.assert_not_called()
    else:
        assert namespace["camera_follow_mode"] == ("follow" if requested_mode == "auto" else requested_mode)
        free_camera.assert_not_called()
        mouse_camera.assert_called_once()


@pytest.mark.parametrize("keyboard", [False, True])
def test_cusrl_initializes_mouse_steering_once_and_keeps_keyboard_follow(camera_helpers, keyboard):
    mouse_camera, follow = Mock(), Mock()
    namespace = _load_functions(
        RL_SCRIPTS / "cusrl/play.py",
        {"CameraPlayerHook"},
        cusrl=SimpleNamespace(Player=SimpleNamespace(Hook=object)),
        args_cli=SimpleNamespace(keyboard=keyboard),
        enable_mouse_camera=mouse_camera,
        camera_follow=follow,
    )
    hook = namespace["CameraPlayerHook"]()
    env = object()
    hook.player = SimpleNamespace(environment=env)
    for step in range(3):
        hook.step(step, {}, {})
    mouse_camera.assert_called_once_with(env)
    assert follow.call_count == (3 if keyboard else 0)
