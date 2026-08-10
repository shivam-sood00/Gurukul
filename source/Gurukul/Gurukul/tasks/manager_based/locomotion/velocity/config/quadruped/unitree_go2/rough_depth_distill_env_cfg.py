import math

from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import RayCasterCameraCfg
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.utils import configclass

import Gurukul.tasks.manager_based.locomotion.velocity.mdp as mdp

from .rough_env_cfg import UnitreeGo2RoughEnvCfg


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

    # Keep parity with the Parkour Go2 depth-camera convention.
    return (qw, qx, qy, -qz)


GO2_DEPTH_CAMERA_CFG = RayCasterCameraCfg(
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
class TerrainHeightmapObservationsCfg(ObsGroup):
    """Privileged local heightmap target used for START terrain reconstruction training."""

    height_scan = ObsTerm(
        func=mdp.height_scan,
        params={"sensor_cfg": SceneEntityCfg("height_scanner")},
        clip=(-1.0, 1.0),
        scale=1.0,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class UnitreeGo2RoughDepthDistillEnvCfg(UnitreeGo2RoughEnvCfg):
    """Go2 rough-terrain locomotion with depth camera observations for student distillation."""

    def __post_init__(self):
        super().__post_init__()
        # Depth rendering has a higher memory footprint than proprioceptive-only settings.
        self.scene.num_envs = 512
        self.scene.depth_camera = GO2_DEPTH_CAMERA_CFG
        self.scene.depth_camera.update_period = self.sim.dt * self.decimation
        self.observations.depth_camera = DepthCameraObservationsCfg()
        self.observations.terrain_heightmap = TerrainHeightmapObservationsCfg()

        # START-style feet-edge penalty for safer foothold selection.
        self.rewards.feet_edge.weight = -1.0
        self.rewards.feet_edge.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_edge.params["asset_cfg"].body_names = [self.foot_link_name]

        # Match Go2 rough behavior by removing zero-weight reward terms in this subclass.
        self.disable_zero_weight_rewards()
