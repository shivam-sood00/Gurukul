"""Load missing mappings into Isaac Sim's own GLFW before it enumerates gamepads."""

import ctypes
import os
import sys
from pathlib import Path

# Unmodified Linux USB mapping from SDL_GameControllerDB:
# https://github.com/mdqinc/SDL_GameControllerDB/blob/master/gamecontrollerdb.txt
# See LICENSE.SDL_GameControllerDB.txt for the upstream license.
XBOX_SERIES_USB_MAPPING = (
    "030000005e040000120b000009050000,Xbox Series Controller,"
    "a:b0,b:b1,back:b6,dpdown:h0.4,dpleft:h0.8,dpright:h0.2,dpup:h0.1,guide:b8,"
    "leftshoulder:b4,leftstick:b9,lefttrigger:a2,leftx:a0,lefty:a1,rightshoulder:b5,"
    "rightstick:b10,righttrigger:a5,rightx:a3,righty:a4,start:b7,x:b2,y:b3,platform:Linux,"
)

# Retain every library handle for the lifetime of the process. Kit owns GLFW shutdown.
_glfw_libraries = {}


def prepare_gamepad_mappings(*, headless: bool = False) -> bool:
    """Register the Xbox Series USB mapping before AppLauncher creates the simulator.

    Import isaaclab.app first: its Isaac Sim bootstrap sets CARB_APP_PATH. The regular
    Python glfw package uses a separate library, so updating that would not fix Kit's input.
    Kit may load its windowing plugin from the extension cache instead of kernel/plugins;
    prepare both copies before startup chooses one. Calling glfwInit again during Kit
    startup preserves mappings in the selected library.
    """
    if sys.platform != "linux" or headless or os.environ.get("HEADLESS") == "1" or not os.environ.get("DISPLAY"):
        return False
    kit_path = os.environ.get("CARB_APP_PATH")
    if not kit_path:
        return False

    kit_path = Path(kit_path)
    plugin_name = "libcarb.windowing-glfw.plugin.so"
    plugin_paths = [kit_path / "kernel/plugins" / plugin_name]
    for root in (kit_path.parent / "extscache", kit_path.parent / "exts", kit_path / "exts"):
        plugin_paths.extend(sorted(root.glob(f"carb.windowing.plugins*/bin/{plugin_name}")))

    registered = False
    for plugin_path in dict.fromkeys(path.resolve() for path in plugin_paths):
        try:
            library = ctypes.CDLL(str(plugin_path))
            library.glfwInit.argtypes = []
            library.glfwInit.restype = ctypes.c_int
            library.glfwUpdateGamepadMappings.argtypes = [ctypes.c_char_p]
            library.glfwUpdateGamepadMappings.restype = ctypes.c_int
            if not library.glfwInit():
                print(f"[WARN] Xbox mapping setup skipped for {plugin_path}: GLFW could not initialize the display.")
                continue
            _glfw_libraries[plugin_path] = library
            if not library.glfwUpdateGamepadMappings(XBOX_SERIES_USB_MAPPING.encode("ascii")):
                print(f"[WARN] Isaac Sim's GLFW rejected the Xbox Series USB mapping: {plugin_path}")
                continue
            registered = True
        except (OSError, AttributeError) as exc:
            print(f"[WARN] Could not register the Xbox Series USB mapping in {plugin_path}: {exc}")
    return registered
