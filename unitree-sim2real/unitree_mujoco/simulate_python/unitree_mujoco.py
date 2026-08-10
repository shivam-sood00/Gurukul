from __future__ import annotations

import os
import sys
import time
import argparse
from pathlib import Path
from threading import Thread
import threading
import numpy as np


COLOR_ENABLED = sys.stdout.isatty() and os.environ.get("NO_COLOR") is None


def _ansi(text: object, code: str) -> str:
    if not COLOR_ENABLED:
        return str(text)
    return f"\033[{code}m{text}\033[0m"


def _status(label: str, value: object, code: str = "1;36") -> str:
    return f"{_ansi(f'[{label}]', code)} {value}"


def _label(text: str) -> str:
    return _ansi(f"{text:<8}", "1;37")


def _parse_args():
    parser = argparse.ArgumentParser(description="Run the Unitree MuJoCo simulation.")
    parser.add_argument(
        "--viewer",
        choices=("native", "viser", "none"),
        default=None,
        help="Viewer backend. Defaults to GURUKUL_VIEWER or native.",
    )
    parser.add_argument(
        "--robot",
        default=None,
        help=(
            "Robot asset folder or alias. Examples: go2, b2, b2_with_z1, b2_z1, b2_z1_arm. "
            "Defaults to GURUKUL_MUJOCO_ROBOT."
        ),
    )
    parser.add_argument(
        "--scene",
        default=None,
        help=(
            "Scene XML name/path or alias. Examples: scene_flat.xml, scene_terrain.xml, flat, rough. "
            "Defaults to GURUKUL_MUJOCO_SCENE."
        ),
    )
    parser.add_argument(
        "--viser",
        action="store_true",
        help="Shortcut for --viewer viser.",
    )
    parser.add_argument(
        "--viser-host",
        default=None,
        help="Host for the viser web server.",
    )
    parser.add_argument(
        "--viser-port",
        type=int,
        default=None,
        help="Port for the viser web server.",
    )
    return parser.parse_args()


_ARGS = _parse_args()
if _ARGS.robot is not None:
    os.environ["GURUKUL_MUJOCO_ROBOT"] = _ARGS.robot
if _ARGS.scene is not None:
    os.environ["GURUKUL_MUJOCO_SCENE"] = _ARGS.scene
if _ARGS.viser:
    os.environ["GURUKUL_VIEWER"] = "viser"
elif _ARGS.viewer is not None:
    os.environ["GURUKUL_VIEWER"] = _ARGS.viewer
if _ARGS.viser_host is not None:
    os.environ["GURUKUL_VISER_HOST"] = _ARGS.viser_host
if _ARGS.viser_port is not None:
    os.environ["GURUKUL_VISER_PORT"] = str(_ARGS.viser_port)

# This simulator uses an interactive MuJoCo viewer. In this path, GLFW is the
# reliable backend for both the viewer and the offscreen depth renderer. Many
# IsaacLab shells export MUJOCO_GL=egl globally, which breaks the added depth
# Renderer in this interactive script even when DISPLAY is available.
_VIEWER_SETTING = os.environ.get("GURUKUL_VIEWER", "native").strip().lower()
if "GURUKUL_MUJOCO_GL" in os.environ:
    os.environ["MUJOCO_GL"] = os.environ["GURUKUL_MUJOCO_GL"]
elif _VIEWER_SETTING in ("viser", "none"):
    os.environ.setdefault("MUJOCO_GL", "egl")
elif os.environ.get("DISPLAY"):
    os.environ["MUJOCO_GL"] = "glfw"

import mujoco
import mujoco.viewer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared_depth import (
    DepthFrameWriter,
    PinholeCameraIntrinsics,
    build_distance_to_camera_scale_map,
    convert_image_plane_depth_to_camera_distance,
    depth_preview_uint8,
)
from unitree_sdk2py.core.channel import ChannelFactoryInitialize
from unitree_sdk2py_bridge import UnitreeSdk2Bridge, ElasticBand

import config


locker = threading.Lock()
running = True
depth_publisher_runtime = None

mj_model = mujoco.MjModel.from_xml_path(config.ROBOT_SCENE)
mj_data = mujoco.MjData(mj_model)

home_key_id = -1
for home_key_name in ("home", "floating_base_homing"):
    home_key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, home_key_name)
    if home_key_id >= 0:
        mujoco.mj_resetDataKeyframe(mj_model, mj_data, home_key_id)
        break


viewer = None
if config.VIEWER == "native" and config.ENABLE_ELASTIC_BAND:
    elastic_band = ElasticBand()
    if config.ROBOT == "pm01":
        band_attached_link = mj_model.body("LINK_TORSO_YAW").id
    elif config.ROBOT == "h1" or config.ROBOT == "g1":
        band_attached_link = mj_model.body("torso_link").id
    else:
        band_attached_link = mj_model.body("base_link").id
    viewer = mujoco.viewer.launch_passive(
        mj_model, mj_data, key_callback=elastic_band.MujuocoKeyCallback
    )
elif config.VIEWER == "native":
    viewer = mujoco.viewer.launch_passive(mj_model, mj_data)
elif config.ENABLE_ELASTIC_BAND:
    print("Elastic band is only available with the native MuJoCo viewer; disabling it for this viewer mode.")

mj_model.opt.timestep = config.SIMULATE_DT
mujoco.mj_forward(mj_model, mj_data)

time.sleep(0.2)


def _simulation_running():
    if not running:
        return False
    if viewer is None:
        return True
    return viewer.is_running()


def _mat3_to_wxyz(rot: np.ndarray) -> tuple[float, float, float, float]:
    """Convert a 3x3 rotation matrix to a normalized wxyz quaternion."""
    matrix = np.asarray(rot, dtype=np.float64).reshape(3, 3)
    trace = float(np.trace(matrix))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        quat = np.array(
            [
                0.25 * s,
                (matrix[2, 1] - matrix[1, 2]) / s,
                (matrix[0, 2] - matrix[2, 0]) / s,
                (matrix[1, 0] - matrix[0, 1]) / s,
            ],
            dtype=np.float64,
        )
    else:
        axis = int(np.argmax(np.diag(matrix)))
        if axis == 0:
            s = np.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[2, 1] - matrix[1, 2]) / s,
                    0.25 * s,
                    (matrix[0, 1] + matrix[1, 0]) / s,
                    (matrix[0, 2] + matrix[2, 0]) / s,
                ],
                dtype=np.float64,
            )
        elif axis == 1:
            s = np.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quat = np.array(
                [
                    (matrix[0, 2] - matrix[2, 0]) / s,
                    (matrix[0, 1] + matrix[1, 0]) / s,
                    0.25 * s,
                    (matrix[1, 2] + matrix[2, 1]) / s,
                ],
                dtype=np.float64,
            )
        else:
            s = np.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quat = np.array(
                [
                    (matrix[1, 0] - matrix[0, 1]) / s,
                    (matrix[0, 2] + matrix[2, 0]) / s,
                    (matrix[1, 2] + matrix[2, 1]) / s,
                    0.25 * s,
                ],
                dtype=np.float64,
            )
    norm = np.linalg.norm(quat)
    if norm > 1.0e-8:
        quat /= norm
    return tuple(float(v) for v in quat)


def _mujoco_color_to_rgb(rgba: np.ndarray, fallback=(145, 145, 145)) -> tuple[int, int, int]:
    color = np.asarray(rgba, dtype=np.float64).reshape(-1)
    if color.size < 3 or float(np.max(color[:3])) <= 0.0:
        return fallback
    return tuple(int(np.clip(v, 0.0, 1.0) * 255.0) for v in color[:3])


class DepthCameraPublisher:
    def __init__(self, mj_model, mj_data):
        self.mj_model = mj_model
        self.mj_data = mj_data
        self.enabled = False
        self._step_counter = 0
        self.camera_id = -1
        self.latest_depth = None
        self.latest_seq = 0

        if not config.ENABLE_DEPTH_CAMERA:
            return

        self.camera_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, config.DEPTH_CAMERA_NAME)
        if self.camera_id < 0:
            print(f"Depth camera '{config.DEPTH_CAMERA_NAME}' not found in {config.ROBOT_SCENE}.")
            return

        try:
            self.renderer = mujoco.Renderer(
                mj_model,
                height=config.DEPTH_HEIGHT,
                width=config.DEPTH_WIDTH,
            )
            self.renderer.enable_depth_rendering()
            self.writer = DepthFrameWriter(
                config.DEPTH_BUFFER_PATH,
                height=config.DEPTH_HEIGHT,
                width=config.DEPTH_WIDTH,
            )
            self.intrinsics = PinholeCameraIntrinsics(
                height=config.DEPTH_HEIGHT,
                width=config.DEPTH_WIDTH,
                fx=config.DEPTH_FX,
                fy=config.DEPTH_FY,
                cx=config.DEPTH_CX,
                cy=config.DEPTH_CY,
            )
            self.distance_scale_map = build_distance_to_camera_scale_map(self.intrinsics)
        except Exception as exc:
            print(f"Depth camera publishing disabled: {exc}")
            return

        self.enabled = True
        print(
            "Depth camera publishing enabled: "
            f"camera={config.DEPTH_CAMERA_NAME} "
            f"shape=({config.DEPTH_HEIGHT}, {config.DEPTH_WIDTH}) "
            f"buffer={config.DEPTH_BUFFER_PATH} "
            f"fx={config.DEPTH_FX:.4f} fy={config.DEPTH_FY:.4f} "
            f"cx={config.DEPTH_CX:.2f} cy={config.DEPTH_CY:.2f} "
            f"flip_vertical={config.DEPTH_FLIP_VERTICAL} "
            f"max_distance={config.DEPTH_MAX_DISTANCE:.2f}"
        )

    def maybe_publish(self):
        if not self.enabled:
            return
        self._step_counter += 1
        if self._step_counter % max(1, config.DEPTH_RENDER_DECIMATION) != 0:
            return
        self.renderer.update_scene(self.mj_data, camera=config.DEPTH_CAMERA_NAME)
        depth = self.renderer.render()
        depth = convert_image_plane_depth_to_camera_distance(
            depth,
            scale_map=self.distance_scale_map,
            flip_vertical=config.DEPTH_FLIP_VERTICAL,
            max_distance=config.DEPTH_MAX_DISTANCE,
        )
        self.latest_depth = depth
        self.latest_seq += 1
        self.writer.write(depth)

    def close(self):
        if not self.enabled:
            return
        self.writer.close()
        if hasattr(self.renderer, "close"):
            self.renderer.close()


class ViserSimulationViewer:
    def __init__(self, mj_model):
        self.mj_model = mj_model
        self.enabled = False
        self.server = None
        self.root = None
        self.robot = None
        self.robot_geom_handles = []
        self.body_ids = []
        self.body_colors = None
        self.apex_marker_handles = []
        self.depth_image_handle = None
        self.depth_gui_handle = None
        self.depth_points_handle = None
        self.depth_seq_seen = -1
        self.depth_update_time = 0.0
        self.depth_point_stride = 3
        self.depth_camera_available = (
            config.ENABLE_DEPTH_CAMERA
            and mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, config.DEPTH_CAMERA_NAME) >= 0
        )

        try:
            import viser

            self._viser = viser
            try:
                from viser.extras import ViserUrdf
            except Exception:
                ViserUrdf = None
            self._viser_urdf_cls = ViserUrdf
        except Exception as exc:
            raise RuntimeError(
                "Viser viewer requested but the 'viser' package is not available. "
                "Install it in this environment or run with --viewer native."
            ) from exc

        self.server = self._viser.ViserServer(host=config.VISER_HOST, port=config.VISER_PORT)
        self.server.scene.set_up_direction("+z")
        self.server.scene.configure_default_lights(True, True)
        self.server.initial_camera.position = (3.0, -5.0, 2.0)
        self.server.initial_camera.look_at = (0.5, 0.0, 0.3)
        self.server.scene.add_grid(
            name="/grid_floor",
            width=30.0,
            height=30.0,
            width_segments=30,
            height_segments=30,
            plane="xy",
            cell_size=1.0,
            section_size=1.0,
            position=(0.0, 0.0, 0.0),
        )
        self.root = self.server.scene.add_frame("/robot", axes_length=0.25, axes_radius=0.01)
        self._add_environment_geometries()
        self._try_add_urdf()
        self._add_robot_geometries()
        self._setup_body_points()
        self._setup_apex_markers()
        self._setup_depth_gui()
        self.enabled = True
        print(
            "Viser viewer enabled: "
            f"http://{config.VISER_HOST}:{config.VISER_PORT} "
            f"(urdf_mesh={'yes' if self.robot is not None else 'no'}, "
            f"mujoco_robot_geoms={len(self.robot_geom_handles)})"
        )

    @staticmethod
    def _default_go2_urdf_path():
        return (
            REPO_ROOT.parent
            / "third_party"
            / "OmniPerception"
            / "LidarSensor"
            / "LidarSensor"
            / "resources"
            / "robots"
            / "go2"
            / "urdf"
            / "go2.urdf"
        )

    def _try_add_urdf(self):
        if config.ROBOT != "go2" and not config.VISER_ROBOT_URDF:
            return
        if self._viser_urdf_cls is None:
            return
        urdf_path = Path(config.VISER_ROBOT_URDF).expanduser() if config.VISER_ROBOT_URDF else self._default_go2_urdf_path()
        if not urdf_path.is_file():
            return
        try:
            self.robot = self._viser_urdf_cls(
                self.server,
                urdf_or_path=str(urdf_path),
                root_node_name="/robot",
                load_meshes=True,
                load_collision_meshes=False,
            )
        except Exception as exc:
            print(f"Viser URDF mesh disabled: {exc}")
            self.robot = None

    def _geom_pose(self, geom_id: int):
        pos = np.asarray(self.mj_model.geom_pos[geom_id], dtype=np.float64)
        quat = np.asarray(self.mj_model.geom_quat[geom_id], dtype=np.float64)
        if np.linalg.norm(quat) > 1.0e-8:
            quat = quat / np.linalg.norm(quat)
            return tuple(float(v) for v in pos), tuple(float(v) for v in quat)
        return tuple(float(v) for v in pos), (1.0, 0.0, 0.0, 0.0)

    @staticmethod
    def _runtime_geom_pose(mj_data, geom_id: int):
        pos = np.asarray(mj_data.geom_xpos[geom_id], dtype=np.float64)
        mat = np.asarray(mj_data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3)
        return tuple(float(v) for v in pos), _mat3_to_wxyz(mat)

    def _add_environment_geometries(self):
        env_count = 0
        for geom_id in range(self.mj_model.ngeom):
            if int(self.mj_model.geom_bodyid[geom_id]) != 0:
                continue
            geom_type = int(self.mj_model.geom_type[geom_id])
            geom_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            geom_name = geom_name or f"geom_{geom_id}"
            safe_name = geom_name.replace("/", "_")
            size = np.asarray(self.mj_model.geom_size[geom_id], dtype=np.float64)
            color = _mujoco_color_to_rgb(self.mj_model.geom_rgba[geom_id])
            pos, quat = self._geom_pose(geom_id)

            try:
                if geom_type == int(mujoco.mjtGeom.mjGEOM_PLANE):
                    self.server.scene.add_box(
                        f"/environment/{safe_name}",
                        dimensions=(30.0, 30.0, 0.01),
                        color=(85, 90, 95),
                        position=(0.0, 0.0, -0.005),
                    )
                elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
                    self.server.scene.add_box(
                        f"/environment/{safe_name}",
                        dimensions=tuple(float(v) for v in 2.0 * size[:3]),
                        color=color,
                        position=pos,
                        wxyz=quat,
                    )
                elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                    self.server.scene.add_icosphere(
                        f"/environment/{safe_name}",
                        radius=float(size[0]),
                        color=color,
                        position=pos,
                        wxyz=quat,
                    )
                elif geom_type == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
                    self.server.scene.add_cylinder(
                        f"/environment/{safe_name}",
                        radius=float(size[0]),
                        height=float(2.0 * size[1]),
                        color=color,
                        position=pos,
                        wxyz=quat,
                    )
                elif geom_type == int(mujoco.mjtGeom.mjGEOM_HFIELD):
                    self.server.scene.add_box(
                        f"/environment/{safe_name}",
                        dimensions=(float(2.0 * size[0]), float(2.0 * size[1]), float(max(size[2], 0.02))),
                        color=(105, 125, 95),
                        position=pos,
                        wxyz=quat,
                    )
                else:
                    self.server.scene.add_icosphere(
                        f"/environment/{safe_name}",
                        radius=0.035,
                        color=color,
                        position=pos,
                    )
                env_count += 1
            except Exception as exc:
                print(f"Viser skipped environment geom '{geom_name}': {exc}")
        print(f"Viser environment geometry loaded: {env_count} world geom(s)")

    def _mesh_vertices_faces(self, mesh_id: int):
        vert_adr = int(self.mj_model.mesh_vertadr[mesh_id])
        vert_num = int(self.mj_model.mesh_vertnum[mesh_id])
        face_adr = int(self.mj_model.mesh_faceadr[mesh_id])
        face_num = int(self.mj_model.mesh_facenum[mesh_id])
        vertices = np.asarray(self.mj_model.mesh_vert[vert_adr : vert_adr + vert_num], dtype=np.float32)
        faces = np.asarray(self.mj_model.mesh_face[face_adr : face_adr + face_num], dtype=np.int32)
        if faces.ndim == 1:
            faces = faces.reshape(-1, 3)
        return vertices, faces

    def _add_robot_geometries(self):
        robot_count = 0
        for geom_id in range(self.mj_model.ngeom):
            body_id = int(self.mj_model.geom_bodyid[geom_id])
            if body_id == 0:
                continue
            body_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
            geom_name = mujoco.mj_id2name(self.mj_model, mujoco.mjtObj.mjOBJ_GEOM, geom_id)
            geom_name = geom_name or f"robot_geom_{geom_id}"
            if body_name.startswith("apex_ref_foot") or geom_name.startswith("apex_ref_foot"):
                continue

            safe_name = geom_name.replace("/", "_")
            geom_type = int(self.mj_model.geom_type[geom_id])
            size = np.asarray(self.mj_model.geom_size[geom_id], dtype=np.float64)
            color = _mujoco_color_to_rgb(self.mj_model.geom_rgba[geom_id], fallback=(175, 175, 190))
            pos, quat = self._runtime_geom_pose(mj_data, geom_id)

            try:
                handle = None
                if geom_type == int(mujoco.mjtGeom.mjGEOM_MESH):
                    mesh_id = int(self.mj_model.geom_dataid[geom_id])
                    if mesh_id >= 0:
                        vertices, faces = self._mesh_vertices_faces(mesh_id)
                        if vertices.size > 0 and faces.size > 0:
                            handle = self.server.scene.add_mesh_simple(
                                f"/mujoco_robot/{safe_name}",
                                vertices=vertices,
                                faces=faces,
                                color=color,
                                position=pos,
                                wxyz=quat,
                            )
                elif geom_type == int(mujoco.mjtGeom.mjGEOM_BOX):
                    handle = self.server.scene.add_box(
                        f"/mujoco_robot/{safe_name}",
                        dimensions=tuple(float(v) for v in 2.0 * size[:3]),
                        color=color,
                        position=pos,
                        wxyz=quat,
                    )
                elif geom_type == int(mujoco.mjtGeom.mjGEOM_SPHERE):
                    handle = self.server.scene.add_icosphere(
                        f"/mujoco_robot/{safe_name}",
                        radius=float(size[0]),
                        color=color,
                        position=pos,
                        wxyz=quat,
                    )
                elif geom_type in (int(mujoco.mjtGeom.mjGEOM_CYLINDER), int(mujoco.mjtGeom.mjGEOM_CAPSULE)):
                    handle = self.server.scene.add_cylinder(
                        f"/mujoco_robot/{safe_name}",
                        radius=float(size[0]),
                        height=float(2.0 * size[1]),
                        color=color,
                        position=pos,
                        wxyz=quat,
                    )

                if handle is not None:
                    self.robot_geom_handles.append((geom_id, handle))
                    robot_count += 1
            except Exception as exc:
                print(f"Viser skipped robot geom '{geom_name}': {exc}")
        print(f"Viser MuJoCo robot geometry loaded: {robot_count} geom(s)")

    def _setup_body_points(self):
        names = ("base", "FL_foot", "FR_foot", "RL_foot", "RR_foot")
        for name in names:
            body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id >= 0:
                self.body_ids.append(body_id)
        self.body_colors = np.array(
            [[230, 230, 230], [220, 70, 70], [70, 190, 110], [80, 130, 240], [245, 190, 70]],
            dtype=np.uint8,
        )[: len(self.body_ids)]

    def _setup_apex_markers(self):
        for idx in range(4):
            handle = self.server.scene.add_icosphere(
                f"/apex_reference/foot_{idx}",
                radius=0.045,
                color=(40, 240, 80),
                position=(0.0, 0.0, -10.0),
                visible=True,
            )
            self.apex_marker_handles.append(handle)

    def _setup_depth_gui(self):
        if not self.depth_camera_available:
            return
        with self.server.gui.add_folder("Depth Camera"):
            self.depth_gui_handle = self.server.gui.add_image(
                np.zeros((config.DEPTH_HEIGHT, config.DEPTH_WIDTH, 3), dtype=np.uint8),
                label=config.DEPTH_CAMERA_NAME,
                format="jpeg",
                jpeg_quality=70,
            )
            stride = self.server.gui.add_slider(
                "Point stride",
                min=1,
                max=8,
                step=1,
                initial_value=self.depth_point_stride,
            )

            @stride.on_update
            def _(_event):
                self.depth_point_stride = int(stride.value)

        self.depth_image_handle = self.server.scene.add_image(
            "/depth_camera/preview",
            np.zeros((config.DEPTH_HEIGHT, config.DEPTH_WIDTH, 3), dtype=np.uint8),
            render_width=0.8,
            render_height=0.45,
            position=(0.0, -1.3, 0.7),
            wxyz=(0.7071068, 0.7071068, 0.0, 0.0),
            format="jpeg",
            jpeg_quality=70,
            visible=True,
        )

    def _update_apex_markers(self, mj_data):
        for idx, handle in enumerate(self.apex_marker_handles):
            body_name = f"apex_ref_foot_{idx}"
            body_id = mujoco.mj_name2id(self.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                handle.visible = False
                continue
            pos = np.asarray(mj_data.xpos[body_id], dtype=np.float64)
            handle.position = tuple(float(v) for v in pos)
            handle.visible = bool(pos[2] > -5.0)

    def _depth_point_cloud(self, depth: np.ndarray, depth_publisher: DepthCameraPublisher, mj_data):
        stride = max(1, int(self.depth_point_stride))
        depth_s = np.asarray(depth[::stride, ::stride], dtype=np.float32)
        rows = np.arange(0, depth.shape[0], stride, dtype=np.float32)
        cols = np.arange(0, depth.shape[1], stride, dtype=np.float32)
        uu, vv = np.meshgrid(cols, rows)
        xn = ((uu + 0.5) - np.float32(config.DEPTH_CX)) / np.float32(config.DEPTH_FX)
        yn = ((vv + 0.5) - np.float32(config.DEPTH_CY)) / np.float32(config.DEPTH_FY)
        scale = np.sqrt(1.0 + xn * xn + yn * yn)
        z = depth_s / np.maximum(scale, 1.0e-6)
        valid = np.isfinite(z) & (z > 0.02)
        if config.DEPTH_MAX_DISTANCE > 0.0:
            valid &= z < config.DEPTH_MAX_DISTANCE
        if not np.any(valid):
            return np.zeros((0, 3), dtype=np.float32), np.zeros((0, 3), dtype=np.uint8)

        local = np.stack([xn[valid] * z[valid], -yn[valid] * z[valid], -z[valid]], axis=1)
        camera_pos = np.asarray(mj_data.cam_xpos[depth_publisher.camera_id], dtype=np.float64)
        camera_rot = np.asarray(mj_data.cam_xmat[depth_publisher.camera_id], dtype=np.float64).reshape(3, 3)
        points = camera_pos[None, :] + local @ camera_rot.T

        scaled = np.clip(depth_s[valid] / max(config.DEPTH_MAX_DISTANCE, 1.0e-6), 0.0, 1.0)
        colors = np.stack(
            [
                (255.0 * (1.0 - scaled)).astype(np.uint8),
                (120.0 + 100.0 * (1.0 - scaled)).astype(np.uint8),
                (255.0 * scaled).astype(np.uint8),
            ],
            axis=1,
        )
        return points.astype(np.float32, copy=False), colors

    def update_depth(self, depth_publisher: DepthCameraPublisher, mj_data):
        if self.depth_gui_handle is None or not depth_publisher.enabled or depth_publisher.latest_depth is None:
            return
        if depth_publisher.latest_seq == self.depth_seq_seen:
            return
        now = time.monotonic()
        if (now - self.depth_update_time) < 0.10:
            return
        self.depth_update_time = now
        self.depth_seq_seen = depth_publisher.latest_seq

        depth = depth_publisher.latest_depth
        preview = depth_preview_uint8(depth, normalized=False, max_distance=config.DEPTH_MAX_DISTANCE)
        preview_rgb = np.repeat(preview[:, :, None], 3, axis=2)

        self.depth_gui_handle.image = preview_rgb
        if self.depth_image_handle is not None:
            self.depth_image_handle.image = preview_rgb

        points, colors = self._depth_point_cloud(depth, depth_publisher, mj_data)
        if points.size > 0:
            self.depth_points_handle = self.server.scene.add_point_cloud(
                "/depth_camera/point_cloud",
                points=points,
                colors=colors,
                point_size=0.012,
            )

    def update(self, mj_data):
        if not self.enabled:
            return
        base_pos = np.asarray(mj_data.qpos[:3], dtype=np.float64)
        base_quat_wxyz = np.asarray(mj_data.qpos[3:7], dtype=np.float64)
        self.root.position = tuple(base_pos)
        self.root.wxyz = tuple(base_quat_wxyz)

        if self.robot is not None:
            try:
                self.robot.update_cfg(np.asarray(mj_data.qpos[7 : 7 + 12], dtype=np.float64))
            except Exception as exc:
                print(f"Viser URDF updates disabled: {exc}")
                self.robot = None

        for geom_id, handle in self.robot_geom_handles:
            pos, quat = self._runtime_geom_pose(mj_data, geom_id)
            handle.position = pos
            handle.wxyz = quat

        # if self.body_ids:
        #     points = np.asarray(mj_data.xpos[self.body_ids], dtype=np.float64)
        #     self.server.scene.add_point_cloud(
        #         "/mujoco_body_points",
        #         points=points,
        #         colors=self.body_colors,
        #         point_size=0.045,
        #     )
        self._update_apex_markers(mj_data)


class ApexMotionOverlay:
    def __init__(self, mj_model):
        self.mj_model = mj_model
        self.buffer = None
        self._warned = False
        self._mocap_ids = []
        for idx in range(4):
            body_name = f"apex_ref_foot_{idx}"
            body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id < 0:
                continue
            mocap_id = int(mj_model.body_mocapid[body_id])
            if mocap_id >= 0:
                self._mocap_ids.append(mocap_id)

    def _try_open(self):
        if not config.ENABLE_APEX_VIS_OVERLAY:
            return False
        if not self._mocap_ids:
            return False
        if self.buffer is not None:
            return True
        path = Path(config.APEX_VIS_BUFFER_PATH)
        if not path.is_file():
            return False
        try:
            self.buffer = np.memmap(path, dtype=np.float32, mode="r", shape=(config.APEX_VIS_FLOAT_COUNT,))
            print(_status("apex", f"motion overlay enabled: {path}", "1;35"))
            return True
        except Exception as exc:
            if not self._warned:
                print(_status("apex", f"motion overlay disabled: {exc}", "1;33"))
                self._warned = True
            return False

    @staticmethod
    def _yaw_from_quat_wxyz(quat):
        w, x, y, z = quat
        return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

    @staticmethod
    def _rot_z(yaw):
        c = np.cos(yaw)
        s = np.sin(yaw)
        return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64)

    @staticmethod
    def _hide_pos():
        return np.array([0.0, 0.0, -10.0], dtype=np.float64)

    def _hide_markers(self, mj_data):
        for mocap_id in self._mocap_ids:
            mj_data.mocap_pos[mocap_id] = self._hide_pos()
            mj_data.mocap_quat[mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)

    def update(self, mj_data):
        if not self._try_open():
            return
        data = np.asarray(self.buffer)
        if abs(float(data[0]) - config.APEX_VIS_MAGIC) > 1.0e-3 or data[3] < 0.5:
            self._hide_markers(mj_data)
            return

        foot_count = min(int(np.clip(data[4], 0, 4)), len(self._mocap_ids))
        foot_rel = data[16 : 16 + foot_count * 3].reshape(foot_count, 3).astype(np.float64)

        base_pos = np.asarray(mj_data.qpos[:3], dtype=np.float64)
        yaw = self._yaw_from_quat_wxyz(np.asarray(mj_data.qpos[3:7], dtype=np.float64))
        rot = self._rot_z(yaw)

        for idx, mocap_id in enumerate(self._mocap_ids):
            if idx < foot_count:
                mj_data.mocap_pos[mocap_id] = base_pos + rot @ foot_rel[idx]
            else:
                mj_data.mocap_pos[mocap_id] = self._hide_pos()
            mj_data.mocap_quat[mocap_id] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)


apex_motion_overlay = ApexMotionOverlay(mj_model)
viser_viewer = ViserSimulationViewer(mj_model) if config.VIEWER == "viser" else None


def SimulationThread():
    global mj_data, mj_model, depth_publisher_runtime

    ChannelFactoryInitialize(config.DOMAIN_ID, config.INTERFACE)
    unitree = UnitreeSdk2Bridge(mj_model, mj_data)
    depth_publisher = DepthCameraPublisher(mj_model, mj_data)
    depth_publisher_runtime = depth_publisher

    if config.USE_JOYSTICK:
        unitree.SetupJoystick(
            device_id=config.JOYSTICK_DEVICE,
            js_type=config.JOYSTICK_TYPE,
            required=config.REQUIRE_JOYSTICK,
        )
    if config.PRINT_SCENE_INFORMATION:
        unitree.PrintSceneInformation()

    try:
        while _simulation_running():
            step_start = time.perf_counter()

            locker.acquire()

            if config.VIEWER == "native" and config.ENABLE_ELASTIC_BAND and elastic_band.enable:
                mj_data.xfrc_applied[band_attached_link, :3] = elastic_band.Advance(
                    mj_data.qpos[:3], mj_data.qvel[:3]
                )

            if unitree.low_cmd_received:
                mujoco.mj_step(mj_model, mj_data)
            else:
                # Publish the exact home state without letting the uncommanded
                # torque-actuated robot collapse during native-viewer startup.
                mujoco.mj_forward(mj_model, mj_data)
            apex_motion_overlay.update(mj_data)
            depth_publisher.maybe_publish()

            locker.release()

            time_until_next_step = mj_model.opt.timestep - (time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)
    finally:
        depth_publisher.close()


def PhysicsViewerThread():
    while _simulation_running():
        locker.acquire()
        if viewer is not None:
            viewer.sync()
        if viser_viewer is not None:
            viser_viewer.update(mj_data)
            if depth_publisher_runtime is not None:
                viser_viewer.update_depth(depth_publisher_runtime, mj_data)
        locker.release()
        time.sleep(config.VIEWER_DT)


if __name__ == "__main__":
    title = _ansi("Unitree MuJoCo", "1;36")
    print(f"\n{title}")
    print("-" * 56)
    print(f"{_label('Scene')}: {config.ROBOT_SCENE}")
    print(f"{_label('Viewer')}: {config.VIEWER}")
    print("-" * 56)
    viewer_thread = Thread(target=PhysicsViewerThread)
    sim_thread = Thread(target=SimulationThread)

    viewer_thread.start()
    sim_thread.start()
    try:
        while viewer_thread.is_alive() and sim_thread.is_alive():
            time.sleep(0.2)
    except KeyboardInterrupt:
        print("Stopping MuJoCo simulation...")
        running = False
    finally:
        viewer_thread.join(timeout=1.0)
        sim_thread.join(timeout=1.0)
