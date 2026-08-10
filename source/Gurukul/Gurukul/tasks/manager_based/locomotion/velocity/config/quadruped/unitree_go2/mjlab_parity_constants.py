"""Numeric parity helpers for Go2 velocity tasks vs `Gurukul_mjlab_Cursor_Reference` Go2 MJCF actuators.

Values follow ``mjlab/asset_zoo/robots/unitree_go2/go2_constants.py`` (10 Hz natural
frequency, damping ratio 2.0, reflected inertia from rotor + gear).
"""

from __future__ import annotations

import math

from isaaclab.actuators import DCMotorCfg

from Gurukul.assets.unitree import UNITREE_GO2_CFG

ROTOR_INERTIA = 0.000111842
HIP_GEAR_RATIO = 6.0
KNEE_GEAR_RATIO = HIP_GEAR_RATIO * 1.5
NATURAL_FREQ = 10.0 * 2.0 * math.pi
DAMPING_RATIO = 2.0

_HIP_I = ROTOR_INERTIA * HIP_GEAR_RATIO**2
_KNEE_I = ROTOR_INERTIA * KNEE_GEAR_RATIO**2

STIFFNESS_HIP_THIGH = _HIP_I * NATURAL_FREQ**2
DAMPING_HIP_THIGH = 2.0 * DAMPING_RATIO * _HIP_I * NATURAL_FREQ
STIFFNESS_CALF = _KNEE_I * NATURAL_FREQ**2
DAMPING_CALF = 2.0 * DAMPING_RATIO * _KNEE_I * NATURAL_FREQ

EFFORT_HIP_THIGH = 23.7
VELOCITY_LIMIT_HIP_THIGH = 30.1
EFFORT_CALF = 35.55
VELOCITY_LIMIT_CALF = 20.07

# Same rule as mjlab ``GO2_ACTION_SCALE``: 0.25 * tau_max / Kp
ACTION_SCALE_HIP_THIGH = 0.25 * EFFORT_HIP_THIGH / STIFFNESS_HIP_THIGH
ACTION_SCALE_CALF = 0.25 * EFFORT_CALF / STIFFNESS_CALF

GO2_MJLAB_PARITY_ACTUATORS = {
    "hips_thighs": DCMotorCfg(
        joint_names_expr=[".*_hip_joint", ".*_thigh_joint"],
        effort_limit=EFFORT_HIP_THIGH,
        saturation_effort=EFFORT_HIP_THIGH,
        velocity_limit=VELOCITY_LIMIT_HIP_THIGH,
        stiffness=STIFFNESS_HIP_THIGH,
        damping=DAMPING_HIP_THIGH,
        friction=0.0,
    ),
    "calf": DCMotorCfg(
        joint_names_expr=[".*_calf_joint"],
        effort_limit=EFFORT_CALF,
        saturation_effort=EFFORT_CALF,
        velocity_limit=VELOCITY_LIMIT_CALF,
        stiffness=STIFFNESS_CALF,
        damping=DAMPING_CALF,
        friction=0.0,
    ),
}

UNITREE_GO2_VELOCITY_MJLAB_V1_CFG = UNITREE_GO2_CFG.replace(
    init_state=UNITREE_GO2_CFG.init_state.replace(
        pos=(0.0, 0.0, 0.31),
        joint_pos={
            ".*thigh_joint": 0.9,
            ".*calf_joint": -1.8,
            ".*R_hip_joint": 0.1,
            ".*L_hip_joint": -0.1,
        },
        joint_vel={".*": 0.0},
    ),
    actuators=GO2_MJLAB_PARITY_ACTUATORS,
)

GO2_MJLAB_PARITY_ACTION_SCALE = {
    ".*_hip_joint": ACTION_SCALE_HIP_THIGH,
    ".*_thigh_joint": ACTION_SCALE_HIP_THIGH,
    ".*_calf_joint": ACTION_SCALE_CALF,
}
