"""Official EngineAI PM01 24-DoF articulation used by the tracking release."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from Gurukul.assets import ISAACLAB_ASSETS_DATA_DIR

# Adapted from engineai-robotics/engineai_rl_lab commit
# 14ec57be718586bd0ac45375aa1115bd896fbdbc (BSD-3-Clause).
PM01_24DOF_POLICY_JOINT_NAMES = [
    "J00_HIP_PITCH_L",
    "J06_HIP_PITCH_R",
    "J12_WAIST_YAW",
    "J01_HIP_ROLL_L",
    "J07_HIP_ROLL_R",
    "J13_SHOULDER_PITCH_L",
    "J18_SHOULDER_PITCH_R",
    "J23_HEAD_YAW",
    "J02_HIP_YAW_L",
    "J08_HIP_YAW_R",
    "J14_SHOULDER_ROLL_L",
    "J19_SHOULDER_ROLL_R",
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

PM01_ARMATURE_Q90 = 0.0453
PM01_EFFORT_Q90 = 164.0
PM01_VELOCITY_Q90 = 26.3
PM01_ARMATURE_Q25 = 0.0067
PM01_EFFORT_Q25 = 52.0
PM01_VELOCITY_Q25 = 35.2
PM01_NATURAL_FREQUENCY = 10.0 * 2.0 * 3.1415926535
PM01_DAMPING_RATIO = 2.0
PM01_STIFFNESS_Q90 = PM01_ARMATURE_Q90 * PM01_NATURAL_FREQUENCY**2
PM01_STIFFNESS_Q25 = PM01_ARMATURE_Q25 * PM01_NATURAL_FREQUENCY**2
PM01_DAMPING_Q90 = 2.0 * PM01_DAMPING_RATIO * PM01_ARMATURE_Q90 * PM01_NATURAL_FREQUENCY
PM01_DAMPING_Q25 = 2.0 * PM01_DAMPING_RATIO * PM01_ARMATURE_Q25 * PM01_NATURAL_FREQUENCY

PM01_24DOF_ACTION_SCALE = {
    name: (
        0.25 * PM01_EFFORT_Q90 / PM01_STIFFNESS_Q90
        if any(token in name for token in ("HIP_PITCH", "HIP_ROLL", "KNEE_PITCH"))
        else 0.25 * PM01_EFFORT_Q25 / PM01_STIFFNESS_Q25
    )
    for name in PM01_24DOF_POLICY_JOINT_NAMES
}

ENGINEAI_PM01_24DOF_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/engineai/pm01_24dof/serial_pm01_edu.usd",
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
        pos=(0.0, 0.0, 0.9),
        joint_pos={
            ".*HIP_PITCH.*": -0.06,
            ".*KNEE_PITCH.*": 0.12,
            ".*ANKLE_PITCH.*": -0.06,
            ".*ELBOW_PITCH.*": -0.25,
            "J14_SHOULDER_ROLL_L": 0.15,
            "J19_SHOULDER_ROLL_R": -0.15,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*HIP.*", ".*KNEE.*"],
            effort_limit_sim={
                ".*HIP_PITCH.*": PM01_EFFORT_Q90,
                ".*HIP_ROLL.*": PM01_EFFORT_Q90,
                ".*HIP_YAW.*": PM01_EFFORT_Q25,
                ".*KNEE_PITCH.*": PM01_EFFORT_Q90,
            },
            velocity_limit_sim={
                ".*HIP_PITCH.*": PM01_VELOCITY_Q90,
                ".*HIP_ROLL.*": PM01_VELOCITY_Q90,
                ".*HIP_YAW.*": PM01_VELOCITY_Q25,
                ".*KNEE_PITCH.*": PM01_VELOCITY_Q90,
            },
            stiffness={
                ".*HIP_PITCH.*": PM01_STIFFNESS_Q90,
                ".*HIP_ROLL.*": PM01_STIFFNESS_Q90,
                ".*HIP_YAW.*": PM01_STIFFNESS_Q25,
                ".*KNEE_PITCH.*": PM01_STIFFNESS_Q90,
            },
            damping={
                ".*HIP_PITCH.*": PM01_DAMPING_Q90,
                ".*HIP_ROLL.*": PM01_DAMPING_Q90,
                ".*HIP_YAW.*": PM01_DAMPING_Q25,
                ".*KNEE_PITCH.*": PM01_DAMPING_Q90,
            },
            armature={
                ".*HIP_PITCH.*": PM01_ARMATURE_Q90,
                ".*HIP_ROLL.*": PM01_ARMATURE_Q90,
                ".*HIP_YAW.*": PM01_ARMATURE_Q25,
                ".*KNEE_PITCH.*": PM01_ARMATURE_Q90,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*ANKLE.*"],
            effort_limit_sim=PM01_EFFORT_Q25,
            velocity_limit_sim=PM01_VELOCITY_Q25,
            stiffness=PM01_STIFFNESS_Q25,
            damping=0.5,
            armature=PM01_ARMATURE_Q25,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["J12_WAIST_YAW"],
            effort_limit_sim=PM01_EFFORT_Q25,
            velocity_limit_sim=PM01_VELOCITY_Q25,
            stiffness=PM01_STIFFNESS_Q25,
            damping=PM01_DAMPING_Q25,
            armature=PM01_ARMATURE_Q25,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[".*SHOULDER.*", ".*ELBOW.*"],
            effort_limit_sim=PM01_EFFORT_Q25,
            velocity_limit_sim=PM01_VELOCITY_Q25,
            stiffness=PM01_STIFFNESS_Q25,
            damping=PM01_DAMPING_Q25,
            armature=PM01_ARMATURE_Q25,
        ),
        "head": ImplicitActuatorCfg(
            joint_names_expr=["J23_HEAD_YAW"],
            effort_limit_sim=PM01_EFFORT_Q25,
            velocity_limit_sim=PM01_VELOCITY_Q25,
            stiffness=PM01_STIFFNESS_Q25,
            damping=PM01_DAMPING_Q25,
            armature=PM01_ARMATURE_Q25,
        ),
    },
)
