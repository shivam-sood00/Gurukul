from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import Gurukul.tasks.manager_based.locomotion.velocity.mdp as mdp
from Gurukul.assets.lidar import make_omni_perception_lidar_ray_caster_cfg

from .rough_env_cfg import UnitreeGo2RoughEnvCfg


LIDAR_POINT_SAMPLE_COUNT = 512


@configclass
class Go2LidarPointCloudObservationsCfg(ObsGroup):
    """Sampled lidar point cloud for the learned PointNet branch."""

    point_cloud = ObsTerm(
        func=mdp.lidar_point_cloud_features,
        params={
            "sensor_cfg": SceneEntityCfg("lidar"),
            "sample_count": LIDAR_POINT_SAMPLE_COUNT,
            "include_distance": True,
        },
        clip=(-1.0, 1.0),
        scale=1.0,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class UnitreeGo2RoughLidarEnvCfg(UnitreeGo2RoughEnvCfg):
    """Go2 rough locomotion with a forward Livox Mid360 lidar mount."""

    def __post_init__(self):
        super().__post_init__()

        mount = self.scene.robot.lidar_mounts["front_mid360"]
        self.scene.lidar = make_omni_perception_lidar_ray_caster_cfg(
            mount,
            mesh_prim_paths=["/World/ground"],
            debug_vis=False,
        )
        # The policy consumes 512 points, so avoid ray-casting the full 20k-point device profile only to discard it.
        # Uniform sampling preserves the Mid360 field of view; taking the first 512 pattern rows does not.
        self.scene.lidar.pattern_cfg.samples = LIDAR_POINT_SAMPLE_COUNT
        self.scene.lidar.pattern_cfg.sampling_mode = "uniform"
        self.scene.lidar.update_period = self.decimation * self.sim.dt

        self.observations.lidar = Go2LidarPointCloudObservationsCfg()

        # PointNet expands every point before pooling, so use a practical default rollout batch for 24 GB GPUs.
        self.scene.num_envs = 512

        self.disable_zero_weight_rewards()
