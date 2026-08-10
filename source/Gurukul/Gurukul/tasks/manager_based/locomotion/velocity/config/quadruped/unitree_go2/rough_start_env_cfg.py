from __future__ import annotations

import math

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import RayCasterCameraCfg
from isaaclab.sensors.ray_caster.patterns import PinholeCameraPatternCfg
from isaaclab.utils import configclass

import Gurukul.tasks.manager_based.locomotion.velocity.mdp as mdp

from .rough_env_cfg import UnitreeGo2RoughEnvCfg
from .start_terrains_cfg import START_SPARSE_TERRAINS_CFG


GO2_START_DEPTH_CAMERA_CFG = RayCasterCameraCfg(
    prim_path="{ENV_REGEX_NS}/Robot/base",
    data_types=["distance_to_camera"],
    offset=RayCasterCameraCfg.OffsetCfg(
        pos=(0.33, 0.0, 0.08),
        rot=(0.36326292, 0.36190961, 0.60922265, 0.60550254),  # quaternion from (roll=180, pitch=70, yaw=-90)
        convention="ros",
    ),
    depth_clipping_behavior="max",
    pattern_cfg=PinholeCameraPatternCfg(
        focal_length=11.041,
        horizontal_aperture=20.955,
        vertical_aperture=12.240,
        height=60,
        width=60,
    ),
    mesh_prim_paths=["/World/ground"],
    max_distance=2.0,
)


@configclass
class StartDepthCameraObservationsCfg(ObsGroup):
    depth_image = ObsTerm(
        func=mdp.depth_image_features,
        params={
            "sensor_cfg": SceneEntityCfg("depth_camera"),
            "data_type": "distance_to_camera",
            "crop_top": 0,
            "crop_bottom": 0,
            "crop_left": 0,
            "crop_right": 0,
            "resize": (60, 60),
            "normalize": True,
        },
        clip=(-1.0, 1.0),
        scale=1.0,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class StartTerrainHeightmapObservationsCfg(ObsGroup):
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
class StartFeetHeightmapObservationsCfg(ObsGroup):
    foot_heightmap = ObsTerm(
        func=mdp.feet_heightmap_scan,
        params={
            "height_sensor_cfg": SceneEntityCfg("height_scanner"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "patch_radius": 0.1,
            "offset": 0.5,
        },
        clip=(-1.0, 1.0),
        scale=1.0,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class StartFeetContactObservationsCfg(ObsGroup):
    contact = ObsTerm(
        func=mdp.feet_contact_state,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "force_threshold": 1.0,
        },
        clip=(0.0, 1.0),
        scale=1.0,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class StartBaseVelocityObservationsCfg(ObsGroup):
    base_lin_vel = ObsTerm(
        func=mdp.base_lin_vel,
        params={"asset_cfg": SceneEntityCfg("robot")},
        clip=(-100.0, 100.0),
        scale=1.0,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class UnitreeGo2RoughStartEnvCfg(UnitreeGo2RoughEnvCfg):
    """Go2 START sparse-foothold environment configuration."""

    def __post_init__(self):
        super().__post_init__()

        # Paper setting: 3072 parallel envs.
        self.scene.num_envs = 3072
        self.scene.terrain.terrain_generator = START_SPARSE_TERRAINS_CFG
        self.scene.terrain.max_init_terrain_level = 2

        # Body-centric heightmap used for TR-Net supervision/policy input:
        # x-range [-0.5, 1.1], y-range [-0.4, 0.4], resolution = 0.05.
        self.scene.height_scanner.offset.pos = (0.3, 0.0, 20.0)
        self.scene.height_scanner.pattern_cfg.size = (1.6, 0.8)
        self.scene.height_scanner.pattern_cfg.resolution = 0.05
        self.scene.height_scanner.update_period = self.decimation * self.sim.dt

        self.scene.depth_camera = GO2_START_DEPTH_CAMERA_CFG
        # Real robot depth is 10 Hz in the paper.
        self.scene.depth_camera.update_period = 0.1

        self.observations.depth_camera = StartDepthCameraObservationsCfg()
        self.observations.terrain_heightmap = StartTerrainHeightmapObservationsCfg()
        self.observations.feet_heightmap = StartFeetHeightmapObservationsCfg()
        self.observations.feet_contact_state = StartFeetContactObservationsCfg()
        self.observations.base_velocity_gt = StartBaseVelocityObservationsCfg()

        # Ensure feet-specific privileged terms match the actual foot links.
        self.observations.feet_heightmap.foot_heightmap.params["asset_cfg"].body_names = [self.foot_link_name]
        self.observations.feet_contact_state.contact.params["sensor_cfg"].body_names = [self.foot_link_name]

        self.observations.critic.height_scan.params["sensor_cfg"] = SceneEntityCfg("height_scanner")
        self.observations.critic.feet_heightmap = ObsTerm(
            func=mdp.feet_heightmap_scan,
            params={
                "height_sensor_cfg": SceneEntityCfg("height_scanner"),
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.foot_link_name]),
                "patch_radius": 0.1,
                "offset": 0.5,
            },
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        self.observations.critic.feet_contact_state = ObsTerm(
            func=mdp.feet_contact_state,
            params={
                "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[self.foot_link_name]),
                "force_threshold": 1.0,
            },
            clip=(0.0, 1.0),
            scale=1.0,
        )

        # START command sampling from the paper.
        self.commands.base_velocity = mdp.StartSparseVelocityCommandCfg(
            asset_name="robot",
            resampling_time_range=(10.0, 10.0),
            rel_standing_envs=0.0,
            rel_heading_envs=0.0,
            heading_command=False,
            ranges=mdp.StartSparseVelocityCommandCfg.Ranges(
                lin_vel_x=(0.0, 1.5),
                lin_vel_y=(0.0, 0.0),
                ang_vel_z=(-1.2, 1.2),
                heading=(-math.pi, math.pi),
            ),
            turn_trigger_threshold=0.3,
        )

        # Reward terms from the START paper.
        self.rewards.track_lin_vel_xy_exp.weight = 1.5
        self.rewards.track_ang_vel_z_exp.weight = 0.5
        self.rewards.lin_vel_z_l2.weight = -2.0
        self.rewards.ang_vel_xy_l2.weight = -0.05
        self.rewards.flat_orientation_l2.weight = -1.0

        self.rewards.joint_torques_l2.weight = -1.0e-5
        self.rewards.action_rate_l2.weight = -0.01
        self.rewards.action_smoothness_l2.weight = -0.01
        self.rewards.joint_power.weight = -2.0e-5
        self.rewards.joint_acc_l2.weight = -2.5e-7
        self.rewards.joint_error.weight = -0.01

        self.rewards.undesired_contacts.weight = -10.0
        self.rewards.undesired_contacts.params["threshold"] = 0.1
        self.rewards.undesired_contacts.params["sensor_cfg"].body_names = [f"^(?!.*{self.foot_link_name}).*"]
        self.rewards.feet_stumble.weight = -1.0
        self.rewards.feet_stumble.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_edge.weight = -1.0
        self.rewards.feet_edge.params["sensor_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_edge.params["asset_cfg"].body_names = [self.foot_link_name]
        self.rewards.feet_edge.params["height_sensor_cfg"] = SceneEntityCfg("height_scanner")
        self.rewards.feet_edge.params["edge_distances_cm"] = (2.5, 5.0)
        self.rewards.feet_edge.params["edge_weights"] = (1.0, 0.5)
        self.rewards.feet_edge.params["edge_height_threshold"] = 0.04
        self.rewards.feet_edge.params["contact_force_threshold"] = 1.0

        # Disable terms not listed in the paper table.
        self.rewards.base_height_l2.weight = 0.0
        self.rewards.body_lin_acc_l2.weight = 0.0
        self.rewards.joint_vel_l2.weight = 0.0
        self.rewards.joint_pos_limits.weight = 0.0
        self.rewards.joint_vel_limits.weight = 0.0
        self.rewards.stand_still.weight = 0.0
        self.rewards.joint_pos_penalty.weight = 0.0
        self.rewards.joint_mirror.weight = 0.0
        self.rewards.applied_torque_limits.weight = 0.0
        self.rewards.contact_forces.weight = 0.0
        self.rewards.feet_air_time.weight = 0.0
        self.rewards.feet_air_time_variance.weight = 0.0
        self.rewards.feet_gait.weight = 0.0
        self.rewards.feet_contact.weight = 0.0
        self.rewards.feet_contact_without_cmd.weight = 0.0
        self.rewards.feet_slide.weight = 0.0
        self.rewards.feet_height.weight = 0.0
        self.rewards.feet_height_body.weight = 0.0
        self.rewards.feet_distance_y_exp.weight = 0.0
        self.rewards.upward.weight = 0.0

        # Terminations approximating paper reset conditions.
        self.terminations.illegal_contact = None
        self.terminations.bad_orientation = DoneTerm(
            func=mdp.bad_orientation,
            params={"limit_angle": 1.0, "asset_cfg": SceneEntityCfg("robot")},
        )
        self.terminations.root_height_min = DoneTerm(
            func=mdp.foot_height_below_minimum,
            params={
                "minimum_height": 0.0,
                "asset_cfg": SceneEntityCfg("robot", body_names=[self.foot_link_name]),
            },
        )

        # START terrain progression schedule.
        self.curriculum.terrain_levels = CurrTerm(
            func=mdp.start_terrain_levels_progression,
            params={
                "easy_terrain_names": ("flat", "stepping_stones_low_random"),
                "advanced_terrain_names": (
                    "stepping_stones_high_random",
                    "balance_beams",
                    "stepping_beams",
                ),
                "gap_terrain_names": ("gaps",),
                "p_max": 1.0,
                "t_start": 2.0e4,
                "t_end": 1.8e5,
                "gap_probability": 0.08,
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )

        self.disable_zero_weight_rewards()
