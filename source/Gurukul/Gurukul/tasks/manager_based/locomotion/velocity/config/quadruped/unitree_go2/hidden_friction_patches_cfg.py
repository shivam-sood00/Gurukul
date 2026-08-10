"""Hidden friction patch scene assets for contact-trail locomotion."""

from __future__ import annotations

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg


def hidden_friction_patch_cfg(name: str, center_xy: tuple[float, float], patch_size: float) -> RigidObjectCfg:
    """Invisible kinematic patch collider with independent friction material."""
    return RigidObjectCfg(
        prim_path=f"{{ENV_REGEX_NS}}/HiddenFrictionPatch_{name}",
        spawn=sim_utils.CuboidCfg(
            size=(patch_size, patch_size, 0.02),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            physics_material=sim_utils.RigidBodyMaterialCfg(static_friction=1.0, dynamic_friction=0.8, restitution=0.0),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.2, 0.2, 0.2), opacity=0.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(center_xy[0], center_xy[1], -0.005)),
    )


HIDDEN_FRICTION_PATCH_NAMES: tuple[str, ...] = (
    "hidden_friction_patch_q00",
    "hidden_friction_patch_q01",
    "hidden_friction_patch_q10",
    "hidden_friction_patch_q11",
)
