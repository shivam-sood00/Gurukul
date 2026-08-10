# SPDX-License-Identifier: Apache-2.0

"""Configuration for the EngineAI T800 humanoid."""

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from Gurukul.assets import ISAACLAB_ASSETS_DATA_DIR
from Gurukul.assets.delayed_implicit_actuator import DelayedImplicitActuatorCfg

# T800 model and actuator parameters adapted from engineai-robotics/engineai_rl_lab
# commit 14ec57be718586bd0ac45375aa1115bd896fbdbc (BSD-3-Clause).
ENGINEAI_T800_POLICY_JOINT_NAMES = [
    "J00_HIP_PITCH_L",
    "J06_HIP_PITCH_R",
    "J12_TORSO_YAW",
    "J01_HIP_ROLL_L",
    "J07_HIP_ROLL_R",
    "J13_SHOULDER_PITCH_L",
    "J18_SHOULDER_PITCH_R",
    "J23_HEAD_PITCH",
    "J02_HIP_YAW_L",
    "J08_HIP_YAW_R",
    "J14_SHOULDER_ROLL_L",
    "J19_SHOULDER_ROLL_R",
    "J24_HEAD_YAW",
    "J03_KNEE_PITCH_L",
    "J09_KNEE_PITCH_R",
    "J15_SHOULDER_YAW_L",
    "J20_SHOULDER_YAW_R",
    "J04_ANKLE_PITCH_L",
    "J10_ANKLE_PITCH_R",
    "J16_ELBOW_PITCH_L",
    "J21_ELBOW_PITCH_R",
    "J05_ANKLE_ROLL_L",
    "J11_ANKLE_ROLL_R",
    "J17_ELBOW_YAW_L",
    "J22_ELBOW_YAW_R",
]

ENGINEAI_T800_ACTION_SCALE = {
    "J00_HIP_PITCH_L": 0.5,
    "J01_HIP_ROLL_L": 0.2,
    "J02_HIP_YAW_L": 0.2,
    "J03_KNEE_PITCH_L": 0.5,
    "J04_ANKLE_PITCH_L": 0.5,
    "J05_ANKLE_ROLL_L": 0.2,
    "J06_HIP_PITCH_R": 0.5,
    "J07_HIP_ROLL_R": 0.2,
    "J08_HIP_YAW_R": 0.2,
    "J09_KNEE_PITCH_R": 0.5,
    "J10_ANKLE_PITCH_R": 0.5,
    "J11_ANKLE_ROLL_R": 0.2,
    "J12_TORSO_YAW": 0.2,
    "J13_SHOULDER_PITCH_L": 0.2,
    "J14_SHOULDER_ROLL_L": 0.2,
    "J15_SHOULDER_YAW_L": 0.05,
    "J16_ELBOW_PITCH_L": 0.2,
    "J17_ELBOW_YAW_L": 0.05,
    "J18_SHOULDER_PITCH_R": 0.2,
    "J19_SHOULDER_ROLL_R": 0.2,
    "J20_SHOULDER_YAW_R": 0.05,
    "J21_ELBOW_PITCH_R": 0.2,
    "J22_ELBOW_YAW_R": 0.05,
    "J23_HEAD_PITCH": 0.2,
    "J24_HEAD_YAW": 0.2,
}

_T800_EFFORT_Q300HL = 415.0
_T800_EFFORT_Q300H = 370.0
_T800_EFFORT_Q200H = 222.0
_T800_EFFORT_Q50H = 160.0
_T800_EFFORT_Q25H = 52.0

_T800_VELOCITY_Q300HL = 25.96
_T800_VELOCITY_Q300H = 25.31
_T800_VELOCITY_Q200H = 23.19
_T800_VELOCITY_Q50H = 33.51
_T800_VELOCITY_Q25H = 35.2

_T800_ARMATURE_Q300HL = 0.2427264
_T800_ARMATURE_Q300H = 0.14110848
_T800_ARMATURE_Q200H = 0.0448737
_T800_ARMATURE_Q50H = 0.0354625
_T800_ARMATURE_Q25H = 0.00671625

ENGINEAI_T800_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/engineai/t800/serial_t800.usd",
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 1.06),
        joint_pos={
            "J00_HIP_PITCH_L": -0.06,
            "J01_HIP_ROLL_L": 0.0,
            "J02_HIP_YAW_L": 0.0,
            "J03_KNEE_PITCH_L": 0.12,
            "J04_ANKLE_PITCH_L": -0.06,
            "J05_ANKLE_ROLL_L": 0.0,
            "J06_HIP_PITCH_R": -0.06,
            "J07_HIP_ROLL_R": 0.0,
            "J08_HIP_YAW_R": 0.0,
            "J09_KNEE_PITCH_R": 0.12,
            "J10_ANKLE_PITCH_R": -0.06,
            "J11_ANKLE_ROLL_R": 0.0,
            "J12_TORSO_YAW": 0.0,
            "J13_SHOULDER_PITCH_L": 0.0,
            "J14_SHOULDER_ROLL_L": 0.15,
            "J15_SHOULDER_YAW_L": 0.0,
            "J16_ELBOW_PITCH_L": -0.25,
            "J17_ELBOW_YAW_L": 0.0,
            "J18_SHOULDER_PITCH_R": 0.0,
            "J19_SHOULDER_ROLL_R": -0.15,
            "J20_SHOULDER_YAW_R": 0.0,
            "J21_ELBOW_PITCH_R": -0.25,
            "J22_ELBOW_YAW_R": 0.0,
            "J23_HEAD_PITCH": 0.0,
            "J24_HEAD_YAW": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip_pitch_and_knee_pitch": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                "J00_HIP_PITCH_L",
                "J06_HIP_PITCH_R",
                "J03_KNEE_PITCH_L",
                "J09_KNEE_PITCH_R",
            ],
            effort_limit_sim=_T800_EFFORT_Q300HL * 0.95,
            velocity_limit_sim=_T800_VELOCITY_Q300HL,
            stiffness=180.0,
            damping=5.0,
            armature=_T800_ARMATURE_Q300HL,
            min_delay=1,
            max_delay=3,
        ),
        "hip_roll": DelayedImplicitActuatorCfg(
            joint_names_expr=["J01_HIP_ROLL_L", "J07_HIP_ROLL_R"],
            effort_limit_sim=_T800_EFFORT_Q300H,
            velocity_limit_sim=_T800_VELOCITY_Q300H,
            stiffness=100.0,
            damping=3.0,
            armature=_T800_ARMATURE_Q300H,
            min_delay=1,
            max_delay=3,
        ),
        "hip_yaw": DelayedImplicitActuatorCfg(
            joint_names_expr=["J02_HIP_YAW_L", "J08_HIP_YAW_R"],
            effort_limit_sim=_T800_EFFORT_Q200H,
            velocity_limit_sim=_T800_VELOCITY_Q200H,
            stiffness=100.0,
            damping=3.0,
            armature=_T800_ARMATURE_Q200H,
            min_delay=1,
            max_delay=3,
        ),
        "ankles": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                "J04_ANKLE_PITCH_L",
                "J05_ANKLE_ROLL_L",
                "J10_ANKLE_PITCH_R",
                "J11_ANKLE_ROLL_R",
            ],
            effort_limit_sim=_T800_EFFORT_Q50H,
            velocity_limit_sim=_T800_VELOCITY_Q50H,
            stiffness=40.0,
            damping=2.0,
            armature=_T800_ARMATURE_Q50H,
            min_delay=1,
            max_delay=3,
        ),
        "torso_yaw": DelayedImplicitActuatorCfg(
            joint_names_expr=["J12_TORSO_YAW"],
            effort_limit_sim=_T800_EFFORT_Q200H,
            velocity_limit_sim=_T800_VELOCITY_Q200H,
            stiffness=100.0,
            damping=3.0,
            armature=_T800_ARMATURE_Q200H,
            min_delay=1,
            max_delay=3,
        ),
        "shoulder_and_elbow_pitch": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                "J13_SHOULDER_PITCH_L",
                "J14_SHOULDER_ROLL_L",
                "J15_SHOULDER_YAW_L",
                "J16_ELBOW_PITCH_L",
                "J18_SHOULDER_PITCH_R",
                "J19_SHOULDER_ROLL_R",
                "J20_SHOULDER_YAW_R",
                "J21_ELBOW_PITCH_R",
            ],
            effort_limit_sim=_T800_EFFORT_Q50H,
            velocity_limit_sim=_T800_VELOCITY_Q50H,
            stiffness=50.0,
            damping=0.3,
            armature=_T800_ARMATURE_Q50H,
            min_delay=1,
            max_delay=3,
        ),
        "heads_and_elbow_yaw": DelayedImplicitActuatorCfg(
            joint_names_expr=[
                "J17_ELBOW_YAW_L",
                "J22_ELBOW_YAW_R",
                "J23_HEAD_PITCH",
                "J24_HEAD_YAW",
            ],
            effort_limit_sim=_T800_EFFORT_Q25H,
            velocity_limit_sim=_T800_VELOCITY_Q25H,
            stiffness=50.0,
            damping=0.3,
            armature=_T800_ARMATURE_Q25H,
            min_delay=1,
            max_delay=3,
        ),
    },
)
