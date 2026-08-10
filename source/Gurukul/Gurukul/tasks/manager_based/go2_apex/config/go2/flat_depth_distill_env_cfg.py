import math

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCameraCfg
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.utils import configclass

import Gurukul.tasks.manager_based.go2_apex.mdp as mdp

from .flat_env_cfg import UnitreeGo2ApexFlatEnvCfg


def _quat_from_euler_xyz_deg(roll_deg: float, pitch_deg: float, yaw_deg: float) -> tuple[float, float, float, float]:
    """Compute a ROS-camera-compatible quaternion tuple from XYZ Euler angles in degrees."""
    roll = math.radians(roll_deg)
    pitch = math.radians(pitch_deg)
    yaw = math.radians(yaw_deg)

    cy = math.cos(yaw * 0.5)
    sy = math.sin(yaw * 0.5)
    cr = math.cos(roll * 0.5)
    sr = math.sin(roll * 0.5)
    cp = math.cos(pitch * 0.5)
    sp = math.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    # Keep parity with the Isaaclab_Parkour camera convention used for Go2 depth distillation.
    return (qw, qx, qy, -qz)


GO2_APEX_DEPTH_CAMERA_CFG = RayCasterCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    data_types=["distance_to_camera"],
    offset=RayCasterCameraCfg.OffsetCfg(
        pos=(0.33, 0.0, 0.08),
        rot=_quat_from_euler_xyz_deg(180.0, 70.0, -90.0),
        convention="ros",
    ),
    depth_clipping_behavior="max",
    pattern_cfg=PinholeCameraPatternCfg(
        focal_length=11.041,
        horizontal_aperture=20.955,
        vertical_aperture=12.240,
        height=60,
        width=106,
    ),
    mesh_prim_paths=["/World/ground"],
    max_distance=2.0,
)


@configclass
class DepthCameraObservationsCfg(ObsGroup):
    """Depth camera observations for student distillation."""

    depth_image = ObsTerm(
        func=mdp.depth_image_features,
        params={
            "sensor_cfg": SceneEntityCfg("depth_camera"),
            "data_type": "distance_to_camera",
            "crop_top": 0,
            "crop_bottom": 2,
            "crop_left": 4,
            "crop_right": 4,
            "resize": (58, 87),
            "normalize": True,
        },
        clip=(-1.0, 1.0),
        scale=1.0,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class UnitreeGo2ApexFlatDepthDistillEnvCfg(UnitreeGo2ApexFlatEnvCfg):
    """Go2 APEX flat environment variant with depth camera observations for student distillation."""

    def __post_init__(self):
        super().__post_init__()
        # Depth rendering has a higher memory footprint than proprio-only training defaults.
        self.scene.num_envs = 512
        self.scene.depth_camera = GO2_APEX_DEPTH_CAMERA_CFG
        self.scene.depth_camera.update_period = self.sim.dt * self.decimation
        self.observations.depth_camera = DepthCameraObservationsCfg()
