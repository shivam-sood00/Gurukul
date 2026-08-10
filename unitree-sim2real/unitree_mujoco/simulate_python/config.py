import os
from pathlib import Path


_ROBOT_ALIASES = {
    "b2_z1": "b2_with_z1",
    "b2_z1_arm": "b2_with_z1",
    "b2+z1": "b2_with_z1",
}
_ROBOT_REQUEST = os.environ.get("GURUKUL_MUJOCO_ROBOT", "b2_with_z1")
ROBOT = _ROBOT_ALIASES.get(_ROBOT_REQUEST, _ROBOT_REQUEST)
# Supported values include "go2", "go2_with_d1", "b2", "b2_with_z1",
# "b2w", "h1", "go2w", "g1", and "pm01".
_BASE_DIR = Path(__file__).resolve().parent
_ROBOT_DIR = (_BASE_DIR / ".." / "unitree_robots" / ROBOT).resolve()

_SCENE_ALIASES = {
    "flat": "scene.xml",
    "default": "scene.xml",
    "rough": "scene_terrain.xml",
    "terrain": "scene_terrain.xml",
    "stairs": "scene_stairs.xml",
}
_DEFAULT_SCENE = "flat" if ROBOT == "go2" else "scene_flat.xml"
if not (_ROBOT_DIR / _DEFAULT_SCENE).is_file() and (_ROBOT_DIR / "scene.xml").is_file():
    _DEFAULT_SCENE = "scene.xml"


def _resolve_scene_path(scene_value: str) -> str:
    candidate = _SCENE_ALIASES.get(scene_value, scene_value)
    path = Path(candidate)
    if path.is_absolute():
        return str(path)
    return str((_ROBOT_DIR / path).resolve())


ROBOT_SCENE = _resolve_scene_path(os.environ.get("GURUKUL_MUJOCO_SCENE", _DEFAULT_SCENE))
DOMAIN_ID = int(os.environ.get("GURUKUL_DOMAIN_ID", "1"))
INTERFACE = os.environ.get("GURUKUL_INTERFACE", "lo")

_JOYSTICK_MODE = os.environ.get("GURUKUL_USE_JOYSTICK", "auto").strip().lower()
USE_JOYSTICK = _JOYSTICK_MODE not in ("0", "false", "off", "no", "none")
REQUIRE_JOYSTICK = _JOYSTICK_MODE in ("1", "true", "on", "yes", "require", "required")
JOYSTICK_TYPE = os.environ.get("GURUKUL_JOYSTICK_TYPE", "xbox")
JOYSTICK_DEVICE = int(os.environ.get("GURUKUL_JOYSTICK_DEVICE", "0"))

PRINT_SCENE_INFORMATION = os.environ.get("GURUKUL_PRINT_SCENE_INFORMATION", "0") == "1"
ENABLE_ELASTIC_BAND = os.environ.get("GURUKUL_ENABLE_ELASTIC_BAND", "0") == "1"

SIMULATE_DT = float(os.environ.get("GURUKUL_SIMULATE_DT", "0.005"))
VIEWER_DT = float(os.environ.get("GURUKUL_VIEWER_DT", "0.02"))
VIEWER = os.environ.get("GURUKUL_VIEWER", "native").strip().lower()
VISER_HOST = os.environ.get("GURUKUL_VISER_HOST", "0.0.0.0")
VISER_PORT = int(os.environ.get("GURUKUL_VISER_PORT", "8080"))
VISER_ROBOT_URDF = os.environ.get("GURUKUL_VISER_ROBOT_URDF", "")

ENABLE_DEPTH_CAMERA = os.environ.get("GURUKUL_ENABLE_DEPTH_CAMERA", "1") != "0"
DEPTH_CAMERA_NAME = os.environ.get("GURUKUL_DEPTH_CAMERA_NAME", "Gurukul_depth")
DEPTH_BUFFER_PATH = os.environ.get("GURUKUL_DEPTH_BUFFER_PATH", "/tmp/Gurukul_go2_depth.mmap")
DEPTH_HEIGHT = int(os.environ.get("GURUKUL_DEPTH_HEIGHT", "60"))
DEPTH_WIDTH = int(os.environ.get("GURUKUL_DEPTH_WIDTH", "106"))
DEPTH_RENDER_DECIMATION = int(os.environ.get("GURUKUL_DEPTH_RENDER_DECIMATION", "4"))
DEPTH_MAX_DISTANCE = float(os.environ.get("GURUKUL_DEPTH_MAX_DISTANCE", "2.0"))
# mujoco.Renderer.render() in this setup already arrives in the image orientation
# expected by the kept Gurukul preview/flatten path. Flipping again inverts the
# scene vertically and makes the depth task unstable.
DEPTH_FLIP_VERTICAL = os.environ.get("GURUKUL_DEPTH_FLIP_VERTICAL", "0") != "0"

# Match Gurukul / IsaacLab GO2_DEPTH_CAMERA_CFG intrinsics.
DEPTH_FOCAL_LENGTH = float(os.environ.get("GURUKUL_DEPTH_FOCAL_LENGTH_M", "0.11041"))
DEPTH_SENSOR_WIDTH = float(os.environ.get("GURUKUL_DEPTH_SENSOR_WIDTH_M", "0.20955"))
DEPTH_SENSOR_HEIGHT = float(os.environ.get("GURUKUL_DEPTH_SENSOR_HEIGHT_M", "0.12240"))
DEPTH_PRINCIPAL_OFFSET_X = float(os.environ.get("GURUKUL_DEPTH_PRINCIPAL_OFFSET_X_PX", "0.0"))
DEPTH_PRINCIPAL_OFFSET_Y = float(os.environ.get("GURUKUL_DEPTH_PRINCIPAL_OFFSET_Y_PX", "0.0"))
DEPTH_FX = DEPTH_WIDTH * DEPTH_FOCAL_LENGTH / DEPTH_SENSOR_WIDTH
DEPTH_FY = DEPTH_HEIGHT * DEPTH_FOCAL_LENGTH / DEPTH_SENSOR_HEIGHT
DEPTH_CX = DEPTH_WIDTH / 2.0 + DEPTH_PRINCIPAL_OFFSET_X
DEPTH_CY = DEPTH_HEIGHT / 2.0 + DEPTH_PRINCIPAL_OFFSET_Y

APEX_VIS_BUFFER_PATH = os.environ.get("GURUKUL_APEX_VIS_BUFFER_PATH", "/tmp/Gurukul_go2_apex_motion_vis.mmap")
APEX_VIS_FLOAT_COUNT = int(os.environ.get("GURUKUL_APEX_VIS_FLOAT_COUNT", "32"))
APEX_VIS_MAGIC = float(os.environ.get("GURUKUL_APEX_VIS_MAGIC", "2048.042"))
ENABLE_APEX_VIS_OVERLAY = os.environ.get("GURUKUL_ENABLE_APEX_VIS_OVERLAY", "1") != "0"
