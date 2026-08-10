"""Go2 rough locomotion environment with contact trail memory observations."""

from __future__ import annotations

from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

import Gurukul.tasks.manager_based.locomotion.velocity.mdp as mdp
from Gurukul.tasks.manager_based.locomotion.velocity.config.quadruped.unitree_go2.hidden_friction_patches_cfg import (
    HIDDEN_FRICTION_PATCH_NAMES,
    hidden_friction_patch_cfg,
)
from Gurukul.tasks.manager_based.locomotion.velocity.config.quadruped.unitree_go2.hidden_friction_terrains_cfg import (
    HIDDEN_FRICTION_FLAT_TERRAINS_CFG,
)
from Gurukul.tasks.manager_based.locomotion.velocity.config.quadruped.unitree_go2.rough_env_cfg import (
    UnitreeGo2RoughEnvCfg,
)
from Gurukul.tasks.manager_based.locomotion.velocity.config.quadruped.unitree_go2.start_terrains_cfg import (
    START_SPARSE_TERRAINS_CFG,
)
from Gurukul.tasks.manager_based.locomotion.velocity.velocity_env_cfg import MySceneCfg


@configclass
class ContactTrailEnvConfig:
    """Contact trail memory settings referenced by env and agent configs."""

    use_contact_trails: bool = True
    num_channels: int = 8
    grid_size: tuple[int, int] = (40, 40)
    resolution: float = 0.05
    decay: float = 0.985
    write_radius: int = 1
    write_mode: str = "learned"
    use_warp: bool = True
    use_aux_loss: bool = True
    aux_loss_weight: float = 0.01
    cnn_latent_dim: int = 128
    write_only_on_contact: bool = True
    contact_force_threshold: float = 1.0
    slip_velocity_scale: float = 0.5
    use_gru: bool = True
    terrain_profile: str = "start_sparse"  # "start_sparse" | "rough" | "hidden_friction_flat"
    hidden_friction_tiles: bool = False
    hidden_friction_grid_size: int = 8
    hidden_friction_tile_size: float = 0.5
    hidden_friction_range: tuple[float, float] = (0.15, 1.2)


@configclass
class ContactTrailsSceneCfg(MySceneCfg):
    """Contact-trails scene.

    Hidden-friction patch colliders are added only when that terrain profile is enabled.
    """


@configclass
class Go2ContactTrailEventsObservationsCfg(ObsGroup):
    """Per-foot contact event features for policy-side writes."""

    contact_trail_events = ObsTerm(
        func=mdp.contact_trail_foot_features,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_foot"),
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*_foot"),
            "contact_force_threshold": 1.0,
        },
        clip=(-5.0, 5.0),
        scale=1.0,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class Go2ContactTrailPoseObservationsCfg(ObsGroup):
    """Base pose for egocentric map warping."""

    contact_trail_pose = ObsTerm(
        func=mdp.contact_trail_base_pose,
        params={"asset_cfg": SceneEntityCfg("robot")},
        clip=(-100.0, 100.0),
        scale=1.0,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class Go2FootPosBObservationsCfg(ObsGroup):
    """Foot positions in yaw-aligned base frame."""

    foot_pos_b = ObsTerm(
        func=mdp.foot_pos_b,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")},
        clip=(-10.0, 10.0),
        scale=1.0,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class Go2ContactTrailPrivilegedObservationsCfg(ObsGroup):
    """Privileged critic-only friction map lookups."""

    hidden_friction_at_feet = ObsTerm(
        func=mdp.hidden_friction_at_feet,
        params={"asset_cfg": SceneEntityCfg("robot", body_names=".*_foot")},
        clip=(0.0, 2.0),
        scale=1.0,
    )

    def __post_init__(self):
        self.enable_corruption = False
        self.concatenate_terms = True


@configclass
class UnitreeGo2RoughContactTrailsEnvCfg(UnitreeGo2RoughEnvCfg):
    """Go2 rough velocity with contact trail memory observations and optional hidden friction."""

    contact_trails: ContactTrailEnvConfig = ContactTrailEnvConfig()
    scene: ContactTrailsSceneCfg = ContactTrailsSceneCfg(num_envs=4096, env_spacing=2.5)

    def __post_init__(self):
        super().__post_init__()

        if not self.contact_trails.use_contact_trails:
            if self.__class__.__name__ == "UnitreeGo2RoughContactTrailsEnvCfg":
                self.disable_zero_weight_rewards()
            return

        self.observations.contact_trail_events = Go2ContactTrailEventsObservationsCfg()
        self.observations.contact_trail_events.contact_trail_events.params["contact_force_threshold"] = (
            self.contact_trails.contact_force_threshold
        )
        self.observations.contact_trail_events.contact_trail_events.params["sensor_cfg"].body_names = [
            self.foot_link_name
        ]
        self.observations.contact_trail_events.contact_trail_events.params["asset_cfg"].body_names = [
            self.foot_link_name
        ]
        self.observations.contact_trail_pose = Go2ContactTrailPoseObservationsCfg()
        self.observations.foot_pos_b = Go2FootPosBObservationsCfg()
        self.observations.foot_pos_b.foot_pos_b.params["asset_cfg"].body_names = [self.foot_link_name]
        self.observations.contact_trail_privileged = Go2ContactTrailPrivilegedObservationsCfg()
        self.observations.contact_trail_privileged.hidden_friction_at_feet.params["asset_cfg"].body_names = [
            self.foot_link_name
        ]

        terrain_profile = str(self.contact_trails.terrain_profile).lower()
        if terrain_profile == "start_sparse":
            self.scene.terrain.terrain_generator = START_SPARSE_TERRAINS_CFG
            self.scene.terrain.max_init_terrain_level = 2
        elif terrain_profile not in {"rough", "hidden_friction_flat"}:
            raise ValueError(
                "contact_trails.terrain_profile must be one of "
                "'start_sparse', 'rough', or 'hidden_friction_flat'."
            )

        if self.contact_trails.hidden_friction_tiles or terrain_profile == "hidden_friction_flat":
            self.scene.terrain.terrain_generator = HIDDEN_FRICTION_FLAT_TERRAINS_CFG
            self.scene.hidden_friction_patch_q00 = hidden_friction_patch_cfg("q00", (-1.0, -1.0), 2.0)
            self.scene.hidden_friction_patch_q01 = hidden_friction_patch_cfg("q01", (-1.0, 1.0), 2.0)
            self.scene.hidden_friction_patch_q10 = hidden_friction_patch_cfg("q10", (1.0, -1.0), 2.0)
            self.scene.hidden_friction_patch_q11 = hidden_friction_patch_cfg("q11", (1.0, 1.0), 2.0)
            self.events.initialize_hidden_friction_grid = EventTerm(
                func=mdp.initialize_hidden_friction_grid,
                mode="startup",
                params={
                    "grid_size": self.contact_trails.hidden_friction_grid_size,
                    "tile_size": self.contact_trails.hidden_friction_tile_size,
                    "friction_range": self.contact_trails.hidden_friction_range,
                },
            )
            self.events.apply_hidden_friction_patch_materials = EventTerm(
                func=mdp.apply_hidden_friction_patch_materials,
                mode="startup",
                params={"patch_asset_names": list(HIDDEN_FRICTION_PATCH_NAMES)},
            )

        if self.__class__.__name__ == "UnitreeGo2RoughContactTrailsEnvCfg":
            self.disable_zero_weight_rewards()


@configclass
class UnitreeGo2RoughContactTrailsEngineeredEnvCfg(UnitreeGo2RoughContactTrailsEnvCfg):
    """Contact trails with engineered writes for warping/write sanity checks."""

    def __post_init__(self):
        self.contact_trails.write_mode = "engineered"
        super().__post_init__()
        self.disable_zero_weight_rewards()
