"""License-independent lidar mount metadata used by robot asset configs.

The runtime lidar integrations and device scan-pattern data are optional and
are intentionally not imported by this module.
"""

from dataclasses import MISSING

from isaaclab.utils import configclass


@configclass
class LidarSensorSpecCfg:
    """Basic sensor profile metadata, independent of a runtime implementation."""

    sensor_type: str = "generic"
    max_range: float = 30.0
    update_period: float = 0.05
    horizontal_fov: float = 360.0
    vertical_fov: float = 59.0
    channels: int = 32
    horizontal_res: float = 1.0
    num_rays: int = 20000
    enable_sensor_noise: bool = False
    optional_runtime_profile: str | None = None


@configclass
class LidarMountCfg:
    """Pose and profile metadata for a lidar mounted on a robot link."""

    parent_link: str = MISSING
    sensor: LidarSensorSpecCfg = MISSING
    prim_suffix: str = "lidar"
    pos: tuple[float, float, float] = (0.0, 0.0, 0.0)
    rot: tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    self_occlusion_mesh: str | None = None

    @property
    def prim_path(self) -> str:
        return f"{{ENV_REGEX_NS}}/Robot/{self.parent_link}/{self.prim_suffix}"

    @property
    def parent_prim_path(self) -> str:
        return f"{{ENV_REGEX_NS}}/Robot/{self.parent_link}"


LIVOX_MID360_METADATA = LidarSensorSpecCfg(
    sensor_type="livox_mid360",
    max_range=30.0,
    update_period=0.05,
    horizontal_fov=360.0,
    vertical_fov=59.0,
    channels=32,
    horizontal_res=1.0,
    num_rays=20000,
    enable_sensor_noise=True,
    optional_runtime_profile="MID360",
)


def make_mid360_mount(
    parent_link: str = "base",
    pos: tuple[float, float, float] = (0.22, 0.0, 0.13),
    prim_suffix: str = "lidar_mid360",
) -> LidarMountCfg:
    """Return mount metadata for an optional Livox Mid-360 integration."""

    return LidarMountCfg(
        parent_link=parent_link,
        sensor=LIVOX_MID360_METADATA,
        prim_suffix=prim_suffix,
        pos=pos,
    )
