"""Lightweight Viser side viewer for a single Isaac Lab env during headless training."""

from __future__ import annotations

import math
import os
import re
import tempfile
from pathlib import Path
from typing import Any, Literal

import numpy as np

from Gurukul.assets import ISAACLAB_ASSETS_DATA_DIR

_FOOT_KEYWORDS = ("foot", "ankle", "toe")
_ARM_KEYWORDS = ("arm", "link", "gripper", "eef", "hand")
_BASE_KEYWORDS = ("base", "trunk", "pelvis", "torso")

_GO2_D1_VIS_URDF = os.path.join(
    ISAACLAB_ASSETS_DATA_DIR,
    "Robots/unitree/go2_with_d1/urdf/go2_d1_vis.urdf",
)
_WEBSOCKETS_SHIM_VERSION = "full-websockets-finder-v2"

RootOrientationMode = Literal["yaw_only", "full"]


def _find_full_websockets_site_packages() -> str | None:
    """Return site-packages dir for a pip-installed websockets with asyncio support."""
    import importlib.metadata
    import site
    import sys

    best_root: str | None = None
    best_version: tuple[int, ...] = ()
    for dist in importlib.metadata.distributions():
        if dist.metadata.get("Name", "").lower() != "websockets":
            continue
        version_parts = tuple(int(part) for part in dist.version.split(".") if part.isdigit())
        if not version_parts or version_parts[0] < 12:
            continue
        asyncio_init = dist.locate_file("websockets/asyncio/__init__.py")
        if not asyncio_init.is_file():
            continue
        site_packages = str(asyncio_init.parent.parent.parent)
        if version_parts > best_version:
            best_version = version_parts
            best_root = site_packages
    if best_root is not None:
        return best_root

    search_roots: list[str] = []
    for entry in site.getsitepackages():
        if entry and entry not in search_roots:
            search_roots.append(entry)
    user_site = site.getusersitepackages()
    if user_site and user_site not in search_roots:
        search_roots.append(user_site)
    for entry in sys.path:
        if entry and entry not in search_roots:
            search_roots.append(entry)

    for entry in search_roots:
        asyncio_init = Path(entry) / "websockets" / "asyncio" / "__init__.py"
        if asyncio_init.is_file():
            return entry
    return None


def _purge_websockets_modules() -> None:
    import sys

    for name in list(sys.modules):
        if name == "websockets" or name.startswith("websockets."):
            del sys.modules[name]


def _is_isaac_pip_prebundle_path(path: str) -> bool:
    normalized = path.replace("\\", "/")
    return "omni.kit.pip_archive" in normalized or "pip_prebundle" in normalized


def _module_file_root(module: Any) -> str | None:
    module_file = getattr(module, "__file__", None)
    if module_file is None:
        return None
    try:
        path = Path(module_file).resolve()
        for parent in path.parents:
            if parent.name == "websockets":
                return str(parent.parent)
        return str(path.parents[1])
    except (IndexError, OSError):
        return None


def _load_module_from_file(module_name: str, module_file: Path, package_dir: Path | None = None) -> Any:
    """Load a module from an exact file path and attach it to its parent package."""
    import importlib.util
    import sys

    kwargs = {}
    if package_dir is not None:
        kwargs["submodule_search_locations"] = [str(package_dir)]
    spec = importlib.util.spec_from_file_location(module_name, module_file, **kwargs)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to create module spec for {module_name} from: {module_file}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)

    parent_name, _, attr_name = module_name.rpartition(".")
    parent = sys.modules.get(parent_name)
    if parent is not None:
        setattr(parent, attr_name, module)
    return module


class _FullWebsocketsFinder:
    """Import hook that pins every ``websockets`` submodule to one package root."""

    def __init__(self, site_packages: str):
        self.site_packages = Path(site_packages).resolve()
        self.ws_pkg = self.site_packages / "websockets"

    def find_spec(self, fullname: str, path: Any = None, target: Any = None) -> Any:
        del path, target
        if fullname != "websockets" and not fullname.startswith("websockets."):
            return None

        parts = fullname.split(".")[1:]
        if not parts:
            init_file = self.ws_pkg / "__init__.py"
            return self._spec(fullname, init_file, self.ws_pkg)

        module_path = self.ws_pkg.joinpath(*parts)
        init_file = module_path / "__init__.py"
        if init_file.is_file():
            return self._spec(fullname, init_file, module_path)

        py_file = module_path.with_suffix(".py")
        if py_file.is_file():
            return self._spec(fullname, py_file, None)
        for suffix in self._extension_suffixes():
            extension_file = module_path.with_suffix(suffix)
            if extension_file.is_file():
                return self._spec(fullname, extension_file, None)
        return None

    @staticmethod
    def _spec(fullname: str, module_file: Path, package_dir: Path | None) -> Any:
        import importlib.util

        kwargs = {}
        if package_dir is not None:
            kwargs["submodule_search_locations"] = [str(package_dir)]
        return importlib.util.spec_from_file_location(fullname, module_file, **kwargs)

    @staticmethod
    def _extension_suffixes() -> tuple[str, ...]:
        import importlib.machinery

        return tuple(importlib.machinery.EXTENSION_SUFFIXES)


def _install_full_websockets_finder(site_packages: str) -> None:
    import sys

    expected = str(Path(site_packages).resolve())
    sys.meta_path[:] = [
        finder
        for finder in sys.meta_path
        if not isinstance(finder, _FullWebsocketsFinder) or str(finder.site_packages) != expected
    ]
    sys.meta_path.insert(0, _FullWebsocketsFinder(site_packages))


def _websockets_asyncio_importable() -> bool:
    import importlib

    try:
        websockets = importlib.import_module("websockets")
        http11 = importlib.import_module("websockets.http11")
        server = importlib.import_module("websockets.asyncio.server")
        root = _module_file_root(websockets)
        return (
            root is not None
            and _module_file_root(http11) == root
            and _module_file_root(server) == root
            and not _is_isaac_pip_prebundle_path(root)
        )
    except (ImportError, ModuleNotFoundError):
        return False


def _force_load_full_websockets(site_packages: str) -> None:
    """Pin ``websockets`` to the pip install Isaac Sim may have shadowed."""
    import importlib
    import sys

    ws_pkg = Path(site_packages) / "websockets"
    init = ws_pkg / "__init__.py"
    asyncio_init = ws_pkg / "asyncio" / "__init__.py"
    if not init.is_file() or not asyncio_init.is_file():
        raise RuntimeError(f"websockets asyncio support not found under: {site_packages}")

    _purge_websockets_modules()
    importlib.invalidate_caches()

    original_sys_path = list(sys.path)
    _install_full_websockets_finder(site_packages)
    sanitized_sys_path = [str(site_packages)] + [
        entry
        for entry in original_sys_path
        if entry != str(site_packages) and not _is_isaac_pip_prebundle_path(entry)
    ]
    sys.path[:] = sanitized_sys_path
    sys.path_importer_cache.clear()

    try:
        importlib.invalidate_caches()

        # Isaac / Kit may keep resolving submodules from pip_prebundle even after the package is repinned.
        # The meta-path finder above pins the whole websockets import graph to this package root.
        module = importlib.import_module("websockets")
        module.__path__ = [str(ws_pkg)]
        if module.__spec__ is not None:
            module.__spec__.submodule_search_locations = [str(ws_pkg)]
        http11 = importlib.import_module("websockets.http11")
        server = importlib.import_module("websockets.asyncio.server")
        expected_root = str(Path(site_packages).resolve())
        bad_modules = []
        for name, loaded_module in sys.modules.items():
            if name != "websockets" and not name.startswith("websockets."):
                continue
            root = _module_file_root(loaded_module)
            if root is not None and root != expected_root:
                bad_modules.append(f"{name}={getattr(loaded_module, '__file__', None)}")
        if (
            _module_file_root(http11) != expected_root
            or _module_file_root(server) != expected_root
            or bad_modules
        ):
            raise RuntimeError(
                "Failed to pin websockets to the full install for viser. "
                f"Expected {expected_root}; mismatched modules: {', '.join(bad_modules)}"
            )
        print(
            f"[INFO][LiveViser] Websockets shim {_WEBSOCKETS_SHIM_VERSION}: pinned to {expected_root}",
            flush=True,
        )
    finally:
        # Keep the full install first, but leave Isaac / Kit paths available for the rest of the process.
        sys.path[:] = [str(site_packages)] + [entry for entry in original_sys_path if entry != str(site_packages)]
        sys.path_importer_cache.clear()
        importlib.invalidate_caches()


def _ensure_viser_websockets() -> None:
    """Use a full pip-installed websockets even when Isaac Sim shadows it.

    Isaac Sim / Omniverse Kit prepends extension ``pip_prebundle/`` directories to
    ``sys.path``. Those bundles ship a trimmed ``websockets`` build for Kit's own
    services and omit ``websockets.asyncio``, which viser requires. This is standard
    Kit behavior on all platforms, not a machine-specific path issue.
    """
    if _websockets_asyncio_importable():
        return

    full_ws_root = _find_full_websockets_site_packages()
    if full_ws_root is None:
        raise RuntimeError(
            "viser requires websockets>=12 with the asyncio submodule in this Python env. "
            "Isaac Sim may shadow an existing install with its bundled copy — run: "
            "pip install 'websockets>=12'"
        )

    _force_load_full_websockets(full_ws_root)
    if not _websockets_asyncio_importable():
        raise RuntimeError(
            "Failed to load a full websockets install for viser after Isaac Sim startup."
        )


def _quat_apply_wxyz(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    vec = np.asarray(vec, dtype=np.float64).reshape(3)
    w, xyz = float(quat[0]), quat[1:]
    t = 2.0 * np.cross(xyz, vec)
    return vec + w * t + np.cross(xyz, t)


def _normalize_quat_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64).reshape(4)
    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-8:
        return np.array((1.0, 0.0, 0.0, 0.0), dtype=np.float64)
    return quat / norm


def _yaw_quat_from_wxyz(quat: np.ndarray) -> np.ndarray:
    quat = _normalize_quat_wxyz(quat)
    w, x, y, z = quat
    yaw = math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    half = 0.5 * yaw
    return np.array((math.cos(half), 0.0, 0.0, math.sin(half)), dtype=np.float64)


def _visual_root_quat(sim_quat: np.ndarray, mode: RootOrientationMode) -> np.ndarray:
    if mode == "full":
        return _normalize_quat_wxyz(sim_quat)
    return _yaw_quat_from_wxyz(sim_quat)

# Approximate visual-only URDFs for common Unitree assets spawned from USD.
_USD_KEYWORD_URDF_FALLBACKS: tuple[tuple[str, str], ...] = (
    ("go2_with_d1", _GO2_D1_VIS_URDF),
    ("go2_description", "Robots/unitree/go2_description/urdf/go2_description.urdf"),
    ("go2_with_airbot", "Robots/unitree/go2_with_airbot/urdf/go2_with_airbot_vis_flip.urdf"),
    ("g1_description", "Robots/unitree/g1_description/urdf/g1_29dof_rev_1_0.urdf"),
    ("b2_description", "Robots/unitree/b2_description/urdf/b2_description.urdf"),
)


def _rewrite_ros_package_mesh_paths(urdf_path: Path) -> Path:
    """Return a Viser/yourdfpy-friendly URDF path with relative mesh filenames."""
    text = urdf_path.read_text(encoding="utf-8")
    if "package://" not in text:
        return urdf_path

    def _replace(match: re.Match[str]) -> str:
        rel = match.group(1).split("/", 1)[-1]
        return f'filename="../{rel}"'

    rewritten = re.sub(r'filename="package://([^"]+)"', _replace, text)
    temp_file = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        suffix=".urdf",
        prefix=f"{urdf_path.stem}_vis_",
        delete=False,
    )
    temp_file.write(rewritten)
    temp_file.close()
    return Path(temp_file.name)


def _resolve_urdf_path(robot: Any, urdf_override: str | None = None) -> str | None:
    if urdf_override:
        override = os.path.expanduser(urdf_override)
        return override if os.path.isfile(override) else None

    spawn = getattr(getattr(robot, "cfg", None), "spawn", None)
    for path_attr in ("asset_path", "usd_path"):
        asset_path = getattr(spawn, path_attr, None)
        if not asset_path:
            continue
        asset_path = os.path.expanduser(str(asset_path))
        if asset_path.lower().endswith(".urdf"):
            if os.path.isfile(asset_path):
                return str(_rewrite_ros_package_mesh_paths(Path(asset_path)))
            try:
                from isaaclab.utils.assets import retrieve_file_path

                resolved = retrieve_file_path(asset_path)
                if resolved and os.path.isfile(resolved):
                    return str(_rewrite_ros_package_mesh_paths(Path(resolved)))
            except Exception:
                pass

        lowered = asset_path.lower()
        for keyword, rel_path in _USD_KEYWORD_URDF_FALLBACKS:
            if keyword in lowered:
                if os.path.isabs(rel_path):
                    candidate = rel_path
                else:
                    candidate = os.path.join(ISAACLAB_ASSETS_DATA_DIR, rel_path)
                if os.path.isfile(candidate):
                    return str(_rewrite_ros_package_mesh_paths(Path(candidate)))
    return None


def resolve_robot_asset(env: Any, robot_name: str = "robot") -> Any | None:
    """Return the articulation asset from manager-based or direct RL envs."""
    scene = getattr(env, "scene", None)
    if scene is not None:
        try:
            return scene[robot_name]
        except (KeyError, TypeError):
            articulations = getattr(scene, "articulations", None)
            if isinstance(articulations, dict) and robot_name in articulations:
                return articulations[robot_name]

    robot = getattr(env, robot_name, None)
    if robot is not None:
        return robot
    return None


def _robot_joint_map(robot: Any) -> dict[str, int]:
    joint_names = list(getattr(robot, "joint_names", []) or [])
    if not joint_names:
        data_names = getattr(getattr(robot, "data", None), "joint_names", None)
        if data_names:
            joint_names = list(data_names)
    if not joint_names:
        return {}
    return dict(zip(joint_names, range(len(joint_names)), strict=False))


def _joint_cfg_for_viser(robot: Any, env_id: int, actuated_joint_names: tuple[str, ...] | list[str]) -> np.ndarray:
    joint_pos = robot.data.joint_pos[env_id].detach().cpu().numpy().reshape(-1)
    name_to_index = _robot_joint_map(robot)
    if not name_to_index:
        return np.asarray(joint_pos, dtype=np.float64)

    values = []
    for joint_name in actuated_joint_names:
        index = name_to_index.get(joint_name)
        if index is None:
            raise KeyError(f"Isaac robot is missing joint '{joint_name}' required by the Viser URDF.")
        values.append(float(joint_pos[index]))
    return np.asarray(values, dtype=np.float64)


def _body_color(name: str) -> tuple[int, int, int]:
    lower = name.lower()
    if any(keyword in lower for keyword in _FOOT_KEYWORDS):
        return (220, 70, 70)
    if any(keyword in lower for keyword in _ARM_KEYWORDS):
        return (70, 140, 240)
    if any(keyword in lower for keyword in _BASE_KEYWORDS):
        return (240, 210, 60)
    return (170, 170, 170)


def _body_radius(name: str) -> float:
    lower = name.lower()
    if any(keyword in lower for keyword in _BASE_KEYWORDS):
        return 0.08
    if any(keyword in lower for keyword in _FOOT_KEYWORDS):
        return 0.045
    if any(keyword in lower for keyword in _ARM_KEYWORDS):
        return 0.03
    return 0.02


class LiveEnvViewer:
    """Stream one env's robot pose to a Viser web UI while training runs headless."""

    def __init__(
        self,
        *,
        host: str = "0.0.0.0",
        port: int = 8080,
        robot_name: str = "robot",
        urdf_path: str | None = None,
        initial_env_id: int = 0,
        update_every_steps: int = 20,
        max_body_markers: int = 64,
        follow_camera: bool = True,
        show_body_markers: bool = False,
        root_orientation_mode: RootOrientationMode = "yaw_only",
    ) -> None:
        _ensure_viser_websockets()
        try:
            import viser
            from viser.extras import ViserUrdf
        except ImportError as exc:
            if "websockets.asyncio" in str(exc):
                raise RuntimeError(
                    "viser could not import websockets.asyncio after Isaac Sim startup. "
                    "Run: pip install 'websockets>=12' in this env."
                ) from exc
            raise RuntimeError(
                "viser is required for --live-viser. Run: pip install viser 'websockets>=12'"
            ) from exc

        self._viser = viser
        self._viser_urdf_cls = ViserUrdf
        self.robot_name = robot_name
        self.max_body_markers = max_body_markers
        self._urdf_path = urdf_path
        self._viser_robot = None
        self._urdf_actuated_joint_names: tuple[str, ...] = ()
        self._reward_text = None
        self._info_text = None
        self._num_envs: int | None = None
        self._did_log_first_update = False
        self._follow_camera = follow_camera
        self._show_body_markers = show_body_markers
        self._root_orientation_mode: RootOrientationMode = root_orientation_mode
        self._default_camera_back = 2.8
        self._default_camera_side = -2.0
        self._default_camera_height = 1.3

        self.server = viser.ViserServer(host=host, port=port)
        self.server.scene.set_up_direction("+z")
        self.server.scene.configure_default_lights(True, True)
        self.server.initial_camera.position = (2.5, -3.5, 1.5)
        self.server.initial_camera.look_at = (0.0, 0.0, 0.35)
        self.server.scene.add_grid(
            name="/grid",
            width=20.0,
            height=20.0,
            width_segments=20,
            height_segments=20,
            plane="xy",
            cell_size=1.0,
            section_size=5.0,
            position=(0.0, 0.0, 0.0),
        )

        with self.server.gui.add_folder("Live training view"):
            self._env_slider = self.server.gui.add_number(
                "env_id",
                initial_value=initial_env_id,
                min=0,
                max=max(0, initial_env_id),
                step=1,
            )
            self._period_slider = self.server.gui.add_number(
                "update_every_steps",
                initial_value=update_every_steps,
                min=1,
                step=1,
            )
            self._follow_checkbox = self.server.gui.add_checkbox("follow_camera", initial_value=follow_camera)
            self._markers_checkbox = self.server.gui.add_checkbox(
                "show_body_markers",
                initial_value=show_body_markers,
            )
            self._orientation_dropdown = self.server.gui.add_dropdown(
                "root_orientation",
                options=("yaw_only", "full"),
                initial_value=root_orientation_mode,
            )
            self._camera_back_slider = self.server.gui.add_slider(
                "camera_back",
                min=1.0,
                max=8.0,
                step=0.1,
                initial_value=self._default_camera_back,
            )
            self._camera_side_slider = self.server.gui.add_slider(
                "camera_side",
                min=-6.0,
                max=6.0,
                step=0.1,
                initial_value=self._default_camera_side,
            )
            self._camera_height_slider = self.server.gui.add_slider(
                "camera_height",
                min=0.2,
                max=4.0,
                step=0.1,
                initial_value=self._default_camera_height,
            )
            self._reward_text = self.server.gui.add_text("reward", initial_value="—", disabled=True)
            self._info_text = self.server.gui.add_text("info", initial_value="waiting for first step", disabled=True)

        self._root = self.server.scene.add_frame("/robot", axes_length=0.2, axes_radius=0.008)
        display_host = "localhost" if host == "0.0.0.0" else host
        print(f"[INFO][LiveViser] Open http://{display_host}:{port} to inspect env {initial_env_id}.")

    @property
    def env_id(self) -> int:
        return max(0, int(self._env_slider.value))

    @property
    def period(self) -> int:
        return max(1, int(self._period_slider.value))

    @property
    def follow_camera(self) -> bool:
        return bool(self._follow_checkbox.value)

    @property
    def show_body_markers(self) -> bool:
        return bool(self._markers_checkbox.value)

    def set_num_envs(self, num_envs: int) -> None:
        self._num_envs = max(1, int(num_envs))
        self._env_slider.max = self._num_envs - 1

    def _ensure_urdf(self, robot: Any) -> None:
        if self._viser_robot is not None:
            return
        urdf_path = _resolve_urdf_path(robot, self._urdf_path)
        if not urdf_path:
            if not self._did_log_first_update:
                print(
                    "[INFO][LiveViser] No URDF available for this asset. "
                    "For Go2+D1 run: python scripts/tools/build_go2_d1_viser_urdf.py"
                )
            return
        try:
            self._viser_robot = self._viser_urdf_cls(
                self.server,
                urdf_or_path=Path(urdf_path),
                root_node_name="/robot",
                load_meshes=True,
                load_collision_meshes=False,
            )
            self._urdf_actuated_joint_names = self._viser_robot.get_actuated_joint_names()
            print(f"[INFO][LiveViser] Loaded URDF mesh viewer from: {urdf_path}")
            print(f"[INFO][LiveViser] URDF actuated joints: {', '.join(self._urdf_actuated_joint_names)}")
        except Exception as exc:
            print(f"[WARN][LiveViser] URDF mesh viewer disabled ({type(exc).__name__}): {exc}")
            self._viser_robot = None

    def _update_body_markers(self, robot: Any, env_id: int) -> int:
        if not self.show_body_markers:
            return 0

        body_names = list(getattr(robot.data, "body_names", []) or [])
        if not body_names:
            return 0
        body_pos = robot.data.body_pos_w[env_id].detach().cpu().numpy()

        shown = 0
        for name, pos in zip(body_names, body_pos):
            safe_name = name.replace("/", "_")
            self.server.scene.add_icosphere(
                f"/markers/{safe_name}",
                radius=_body_radius(name),
                position=tuple(float(v) for v in pos),
                color=_body_color(name),
            )
            shown += 1
            if shown >= self.max_body_markers:
                break
        return shown

    @property
    def root_orientation_mode(self) -> RootOrientationMode:
        value = str(self._orientation_dropdown.value)
        return "full" if value == "full" else "yaw_only"

    def _update_camera(self, root_pos: np.ndarray, root_quat: np.ndarray) -> None:
        if not self.follow_camera:
            return
        target = root_pos.astype(np.float64, copy=False)
        look_at = target + np.array((0.0, 0.0, 0.25), dtype=np.float64)
        camera_offset_body = np.array(
            (
                -float(self._camera_back_slider.value),
                float(self._camera_side_slider.value),
                float(self._camera_height_slider.value),
            ),
            dtype=np.float64,
        )
        eye = target + _quat_apply_wxyz(root_quat, camera_offset_body)
        for client in self.server.get_clients().values():
            client.camera.position = tuple(float(v) for v in eye)
            client.camera.look_at = tuple(float(v) for v in look_at)

    def update(
        self,
        env: Any,
        step: int,
        *,
        reward: float | None = None,
        done: bool | None = None,
    ) -> None:
        if step % self.period != 0:
            return

        robot = resolve_robot_asset(env, self.robot_name)
        if robot is None:
            if self._info_text is not None:
                self._info_text.value = f"asset '{self.robot_name}' not found on env"
            return

        num_envs = int(getattr(env, "num_envs", self._num_envs or 1))
        self.set_num_envs(num_envs)
        env_id = min(self.env_id, max(0, num_envs - 1))
        if self._env_slider.value != env_id:
            self._env_slider.value = env_id

        root_pos = robot.data.root_pos_w[env_id].detach().cpu().numpy()
        sim_quat = robot.data.root_quat_w[env_id].detach().cpu().numpy()
        vis_quat = _visual_root_quat(sim_quat, self.root_orientation_mode)
        self._root.position = tuple(float(v) for v in root_pos)
        self._root.wxyz = tuple(float(v) for v in vis_quat)

        self._ensure_urdf(robot)
        if self._viser_robot is not None and self._urdf_actuated_joint_names:
            try:
                joint_cfg = _joint_cfg_for_viser(robot, env_id, self._urdf_actuated_joint_names)
                self._viser_robot.update_cfg(joint_cfg)
            except Exception as exc:
                print(f"[WARN][LiveViser] URDF pose updates disabled ({type(exc).__name__}): {exc}")
                self._viser_robot = None

        marker_count = self._update_body_markers(robot, env_id)
        self._update_camera(root_pos, vis_quat)

        if not self._did_log_first_update:
            self._did_log_first_update = True
            print(
                "[INFO][LiveViser] First frame published: "
                f"env_id={env_id} root_pos=({root_pos[0]:.2f}, {root_pos[1]:.2f}, {root_pos[2]:.2f}) "
                f"markers={marker_count} urdf={'yes' if self._viser_robot is not None else 'no'}"
            )

        if self._reward_text is not None and reward is not None:
            self._reward_text.value = f"{reward:.4f}"
        if self._info_text is not None:
            done_text = "done" if done else "running"
            self._info_text.value = (
                f"step={step} env={env_id}/{num_envs - 1} {done_text} "
                f"pos=({root_pos[0]:.2f}, {root_pos[1]:.2f}, {root_pos[2]:.2f})"
            )

    def close(self) -> None:
        if hasattr(self.server, "stop"):
            self.server.stop()
