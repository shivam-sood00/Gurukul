# SPDX-License-Identifier: Apache-2.0

"""EngineAI PM01 articulation built from the retained official URDF."""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

from Gurukul.assets import ISAACLAB_ASSETS_DATA_DIR
from Gurukul.assets.engineai_pm01_official import (
    PM01_ARMATURE_Q25,
    PM01_ARMATURE_Q90,
    PM01_DAMPING_Q25,
    PM01_DAMPING_Q90,
    PM01_EFFORT_Q25,
    PM01_EFFORT_Q90,
    PM01_STIFFNESS_Q25,
    PM01_STIFFNESS_Q90,
    PM01_VELOCITY_Q25,
    PM01_VELOCITY_Q90,
)

ENGINEAI_PM01_URDF_JOINT_NAMES = [
    "j00_hip_pitch_l",
    "j01_hip_roll_l",
    "j02_hip_yaw_l",
    "j03_knee_pitch_l",
    "j04_ankle_pitch_l",
    "j05_ankle_roll_l",
    "j06_hip_pitch_r",
    "j07_hip_roll_r",
    "j08_hip_yaw_r",
    "j09_knee_pitch_r",
    "j10_ankle_pitch_r",
    "j11_ankle_roll_r",
    "j12_waist_yaw",
    "j13_shoulder_pitch_l",
    "j14_shoulder_roll_l",
    "j15_shoulder_yaw_l",
    "j16_elbow_pitch_l",
    "j17_elbow_yaw_l",
    "j18_shoulder_pitch_r",
    "j19_shoulder_roll_r",
    "j20_shoulder_yaw_r",
    "j21_elbow_pitch_r",
    "j22_elbow_yaw_r",
    "j23_head_yaw",
]

ENGINEAI_PM01_URDF_ACTION_SCALE = {
    name: (
        0.25 * PM01_EFFORT_Q90 / PM01_STIFFNESS_Q90
        if any(token in name for token in ("hip_pitch", "hip_roll", "knee_pitch"))
        else 0.25 * PM01_EFFORT_Q25 / PM01_STIFFNESS_Q25
    )
    for name in ENGINEAI_PM01_URDF_JOINT_NAMES
}

ENGINEAI_PM01_URDF_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/engineai/pm01_description/urdf/pm01.urdf",
        activate_contact_sensors=True,
        self_collision=False,
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
            # The raw URDF collision hulls overlap between non-adjacent links
            # (base/hip-roll and knee/ankle-roll), so articulation self-contact
            # would create illegal-contact resets in the first policy step.
            enabled_self_collisions=False,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.9),
        joint_pos={
            "j00_hip_pitch_l": -0.06,
            "j01_hip_roll_l": 0.0,
            "j02_hip_yaw_l": 0.0,
            "j03_knee_pitch_l": 0.12,
            "j04_ankle_pitch_l": -0.06,
            "j05_ankle_roll_l": 0.0,
            "j06_hip_pitch_r": -0.06,
            "j07_hip_roll_r": 0.0,
            "j08_hip_yaw_r": 0.0,
            "j09_knee_pitch_r": 0.12,
            "j10_ankle_pitch_r": -0.06,
            "j11_ankle_roll_r": 0.0,
            "j12_waist_yaw": 0.0,
            "j13_shoulder_pitch_l": 0.0,
            "j14_shoulder_roll_l": 0.15,
            "j15_shoulder_yaw_l": 0.0,
            "j16_elbow_pitch_l": -0.25,
            "j17_elbow_yaw_l": 0.0,
            "j18_shoulder_pitch_r": 0.0,
            "j19_shoulder_roll_r": -0.15,
            "j20_shoulder_yaw_r": 0.0,
            "j21_elbow_pitch_r": -0.25,
            "j22_elbow_yaw_r": 0.0,
            "j23_head_yaw": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[".*hip.*", ".*knee.*"],
            effort_limit_sim={
                ".*hip_pitch.*": PM01_EFFORT_Q90,
                ".*hip_roll.*": PM01_EFFORT_Q90,
                ".*hip_yaw.*": PM01_EFFORT_Q25,
                ".*knee_pitch.*": PM01_EFFORT_Q90,
            },
            velocity_limit_sim={
                ".*hip_pitch.*": PM01_VELOCITY_Q90,
                ".*hip_roll.*": PM01_VELOCITY_Q90,
                ".*hip_yaw.*": PM01_VELOCITY_Q25,
                ".*knee_pitch.*": PM01_VELOCITY_Q90,
            },
            stiffness={
                ".*hip_pitch.*": PM01_STIFFNESS_Q90,
                ".*hip_roll.*": PM01_STIFFNESS_Q90,
                ".*hip_yaw.*": PM01_STIFFNESS_Q25,
                ".*knee_pitch.*": PM01_STIFFNESS_Q90,
            },
            damping={
                ".*hip_pitch.*": PM01_DAMPING_Q90,
                ".*hip_roll.*": PM01_DAMPING_Q90,
                ".*hip_yaw.*": PM01_DAMPING_Q25,
                ".*knee_pitch.*": PM01_DAMPING_Q90,
            },
            armature={
                ".*hip_pitch.*": PM01_ARMATURE_Q90,
                ".*hip_roll.*": PM01_ARMATURE_Q90,
                ".*hip_yaw.*": PM01_ARMATURE_Q25,
                ".*knee_pitch.*": PM01_ARMATURE_Q90,
            },
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*ankle.*"],
            effort_limit_sim=PM01_EFFORT_Q25,
            velocity_limit_sim=PM01_VELOCITY_Q25,
            stiffness=PM01_STIFFNESS_Q25,
            damping=0.5,
            armature=PM01_ARMATURE_Q25,
        ),
        "upper_body": ImplicitActuatorCfg(
            joint_names_expr=[".*waist.*", ".*shoulder.*", ".*elbow.*", ".*head.*"],
            effort_limit_sim=PM01_EFFORT_Q25,
            velocity_limit_sim=PM01_VELOCITY_Q25,
            stiffness=PM01_STIFFNESS_Q25,
            damping=PM01_DAMPING_Q25,
            armature=PM01_ARMATURE_Q25,
        ),
    },
)
