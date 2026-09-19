import weakref

import numpy as np
import torch

import isaaclab.utils.math as math_utils


class _MouseSteerableCamera:
    """Adapt a viewport controller so tracking retains mouse edits to the live camera.

    The original trajectory is used until the user adjusts the view. Afterwards, translate the
    live eye and target by the tracking target's displacement, preserving the user's framing.
    Both Isaac Lab's asset tracking and scripted follow call update_view_location.
    """

    def __init__(self, controller, camera_state):
        self._controller = weakref.proxy(controller)
        self._camera_state = camera_state
        # Keep the controller's normal destruction/unsubscription behavior when env.close() deletes it.
        self._update_view_location = weakref.WeakMethod(controller.update_view_location)
        self._eye = controller.default_cam_eye.copy()
        self._lookat = controller.default_cam_lookat.copy()
        self._last_tracking_target = self._origin() + self._lookat
        self._last_eye, self._last_target = self._read_view()
        self._steered = False

    def _origin(self):
        return self._controller.viewer_origin.detach().cpu().numpy()

    def _read_view(self):
        return np.array(self._camera_state.position_world), np.array(self._camera_state.target_world)

    def __call__(self, eye=None, lookat=None):
        if eye is not None:
            self._eye = np.asarray(eye, dtype=float).copy()
        if lookat is not None:
            self._lookat = np.asarray(lookat, dtype=float).copy()
        origin = self._origin()
        tracking_target = origin + self._lookat
        live_eye, live_target = self._read_view()
        self._steered |= not (
            np.allclose(live_eye, self._last_eye, rtol=0.0, atol=1e-5)
            and np.allclose(live_target, self._last_target, rtol=0.0, atol=1e-5)
        )
        if self._steered:
            displacement = tracking_target - self._last_tracking_target
            eye = live_eye + displacement - origin
            lookat = live_target + displacement - origin
        else:
            eye, lookat = self._eye, self._lookat
        self._update_view_location()(eye=eye, lookat=lookat)
        # Sample the authored result so USD precision/normalization is not mistaken for mouse input.
        self._last_eye, self._last_target = self._read_view()
        self._last_tracking_target = tracking_target.copy()


def enable_mouse_camera(env):
    """Retain the existing starting view and follow behavior while allowing mouse steering."""
    controller = getattr(env.unwrapped, "viewport_camera_controller", None)
    if controller is None or not env.unwrapped.sim.has_gui():
        return
    if isinstance(controller.update_view_location, _MouseSteerableCamera):
        return

    from omni.kit.viewport.utility import get_active_viewport
    from omni.kit.viewport.utility.camera_state import ViewportCameraState

    viewport = get_active_viewport()
    if viewport is None:
        return
    # Asset tracking may not have rendered yet after reset; initialize its authored view once.
    if controller.cfg.origin_type == "asset_root":
        controller.update_view_to_asset_root(controller.cfg.asset_name)
    elif controller.cfg.origin_type == "asset_body":
        controller.update_view_to_asset_body(controller.cfg.asset_name, controller.cfg.body_name)
    camera_state = ViewportCameraState("/OmniverseKit_Persp", viewport)
    controller.update_view_location = _MouseSteerableCamera(controller, camera_state)


def enable_free_camera(env):
    """Apply the existing starting view once, then leave the viewport under mouse control.

    Call after the initial environment reset so asset-relative views use the actual spawn pose.
    Changing the live controller's origin mode stops its render callback without moving the camera.
    """
    controller = getattr(env.unwrapped, "viewport_camera_controller", None)
    if controller is None:
        return

    if controller.cfg.origin_type == "asset_root":
        controller.update_view_to_asset_root(controller.cfg.asset_name)
    elif controller.cfg.origin_type == "asset_body":
        controller.update_view_to_asset_body(controller.cfg.asset_name, controller.cfg.body_name)

    # Do not call update_view_to_world(): that would discard the asset/env origin and shift the view.
    controller.cfg.origin_type = "world"


def camera_follow(
    env,
    mode: str = "follow",
    window_size: int = 50,
    env_index: int = 0,
    follow_offset: tuple[float, float, float] | None = None,
):
    controller = getattr(env.unwrapped, "viewport_camera_controller", None)
    if controller is None:
        return

    if not hasattr(camera_follow, "history"):
        camera_follow.history = {}

    window_size = max(1, int(window_size))
    history_key = (int(env_index), str(mode))
    smooth_camera_positions = camera_follow.history.setdefault(history_key, [])

    robot_pos = env.unwrapped.scene["robot"].data.root_pos_w[0]
    robot_quat = env.unwrapped.scene["robot"].data.root_quat_w[0]

    if mode == "follow":
        offset = follow_offset if follow_offset is not None else (-3.0, 0.0, 0.5)
        camera_offset = torch.tensor(offset, dtype=torch.float32, device=env.device)
        camera_pos = math_utils.transform_points(
            camera_offset.unsqueeze(0), pos=robot_pos.unsqueeze(0), quat=robot_quat.unsqueeze(0)
        ).squeeze(0)
    elif mode == "isometric":
        # Keep camera in world frame for a stable diagonal overview.
        camera_offset = torch.tensor([-3.5, 3.5, 2.8], dtype=torch.float32, device=env.device)
        camera_pos = robot_pos + camera_offset
    elif mode == "topdown":
        camera_offset = torch.tensor([0.0, 0.0, 6.0], dtype=torch.float32, device=env.device)
        camera_pos = robot_pos + camera_offset
    else:
        raise ValueError(f"Unsupported camera follow mode: {mode}")

    smooth_camera_positions.append(camera_pos)
    if len(smooth_camera_positions) > window_size:
        smooth_camera_positions.pop(0)
    smooth_camera_pos = torch.mean(torch.stack(smooth_camera_positions), dim=0)

    # Scripted poses are world coordinates; also stop Isaac Lab's competing asset-tracking callback.
    if controller.cfg.origin_type != "world":
        controller.update_view_to_world()
    controller.set_view_env_index(env_index=env_index)
    controller.update_view_location(eye=smooth_camera_pos.cpu().numpy(), lookat=robot_pos.cpu().numpy())
