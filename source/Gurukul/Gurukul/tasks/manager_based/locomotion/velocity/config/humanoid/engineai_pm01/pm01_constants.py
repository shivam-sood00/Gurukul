# SPDX-License-Identifier: Apache-2.0
"""Constants for the official EngineAI PM01 velocity tasks."""

from isaaclab.managers import SceneEntityCfg

# Preserve the velocity checkpoint's logical j00--j23 order while using the
# official colored USD asset's uppercase joint names.
PM01_POLICY_JOINT_NAMES: list[str] = [
    "J00_HIP_PITCH_L",
    "J01_HIP_ROLL_L",
    "J02_HIP_YAW_L",
    "J03_KNEE_PITCH_L",
    "J04_ANKLE_PITCH_L",
    "J05_ANKLE_ROLL_L",
    "J06_HIP_PITCH_R",
    "J07_HIP_ROLL_R",
    "J08_HIP_YAW_R",
    "J09_KNEE_PITCH_R",
    "J10_ANKLE_PITCH_R",
    "J11_ANKLE_ROLL_R",
    "J12_WAIST_YAW",
    "J13_SHOULDER_PITCH_L",
    "J14_SHOULDER_ROLL_L",
    "J15_SHOULDER_YAW_L",
    "J16_ELBOW_PITCH_L",
    "J17_ELBOW_YAW_L",
    "J18_SHOULDER_PITCH_R",
    "J19_SHOULDER_ROLL_R",
    "J20_SHOULDER_YAW_R",
    "J21_ELBOW_PITCH_R",
    "J22_ELBOW_YAW_R",
    "J23_HEAD_YAW",
]

PM01_POLICY_JOINT_ASSET_CFG = SceneEntityCfg("robot", joint_names=PM01_POLICY_JOINT_NAMES, preserve_order=True)

PM01_BASE_HEIGHT_TARGET = 0.82

PM01_FEET_BODY_CFG = SceneEntityCfg("robot", body_names=["LINK_ANKLE_ROLL_L", "LINK_ANKLE_ROLL_R"])

# Push velocity ranges (rad/s and m/s) — engineai_amp PM01EventCfg.
PM01_PUSH_VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

# mjlab ``variable_posture`` std maps (G1 values, PM01 joint-name patterns).
PM01_VARIABLE_POSTURE_STD_STANDING: dict[str, float] = {".*": 0.05}
PM01_VARIABLE_POSTURE_STD_WALKING: dict[str, float] = {
    r".*HIP_PITCH.*": 0.3,
    r".*HIP_ROLL.*": 0.15,
    r".*HIP_YAW.*": 0.15,
    r".*KNEE_PITCH.*": 0.35,
    r".*ANKLE_PITCH.*": 0.25,
    r".*ANKLE_ROLL.*": 0.1,
    r".*WAIST_YAW.*": 0.2,
    r".*SHOULDER_PITCH.*": 0.15,
    r".*SHOULDER_ROLL.*": 0.15,
    r".*SHOULDER_YAW.*": 0.2,
    r".*ELBOW_PITCH.*": 0.15,
    r".*ELBOW_YAW.*": 0.3,
    r".*HEAD_YAW.*": 0.15,
}
PM01_VARIABLE_POSTURE_STD_RUNNING: dict[str, float] = {
    r".*HIP_PITCH.*": 0.5,
    r".*HIP_ROLL.*": 0.2,
    r".*HIP_YAW.*": 0.2,
    r".*KNEE_PITCH.*": 0.6,
    r".*ANKLE_PITCH.*": 0.35,
    r".*ANKLE_ROLL.*": 0.15,
    r".*WAIST_YAW.*": 0.3,
    r".*SHOULDER_PITCH.*": 0.5,
    r".*SHOULDER_ROLL.*": 0.2,
    r".*SHOULDER_YAW.*": 0.3,
    r".*ELBOW_PITCH.*": 0.35,
    r".*ELBOW_YAW.*": 0.4,
    r".*HEAD_YAW.*": 0.2,
}
