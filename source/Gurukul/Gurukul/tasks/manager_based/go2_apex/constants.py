"""Shared constants for manager-based Go2 APEX tasks."""

VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-1.5, 1.5),
}

FOOT_BODY_NAMES = ("FL_foot", "FR_foot", "RL_foot", "RR_foot")
ILLEGAL_CONTACT_BODY_NAMES = ("base", ".*_hip.*")

GO2_ACTION_JOINT_NAMES = (
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
)

GO2_D1_ARM_JOINT_NAMES = (
    "arm_1_joint",
    "arm_2_joint",
    "arm_3_joint",
    "arm_4_joint",
    "arm_5_joint",
    "arm_6_joint",
)

GO2_D1_GRIPPER_JOINT_NAMES = (
    "arm_7_1_joint",
    "arm_7_2_joint",
)

# Measured D1 servo endpoints. The simulator uses one positive jaw-travel
# coordinate: 0 m is open, 0.033 m is closed, and the second jaw is mimicked.
GO2_D1_GRIPPER_OPEN_HARDWARE_ANGLE_DEG = 68.6
GO2_D1_GRIPPER_CLOSED_HARDWARE_ANGLE_DEG = -28.7
GO2_D1_GRIPPER_MAX_JAW_TRAVEL_M = 0.033


def go2_d1_gripper_hardware_angle_to_joint_position(angle_deg: float) -> float:
    """Convert a D1 servo angle in degrees to clamped per-jaw travel in metres."""
    closure = (
        GO2_D1_GRIPPER_OPEN_HARDWARE_ANGLE_DEG - float(angle_deg)
    ) / (
        GO2_D1_GRIPPER_OPEN_HARDWARE_ANGLE_DEG
        - GO2_D1_GRIPPER_CLOSED_HARDWARE_ANGLE_DEG
    )
    return min(max(closure, 0.0), 1.0) * GO2_D1_GRIPPER_MAX_JAW_TRAVEL_M


def go2_d1_gripper_joint_position_to_hardware_angle(joint_position_m: float) -> float:
    """Convert clamped per-jaw travel in metres to the D1 servo angle in degrees."""
    closure = min(
        max(float(joint_position_m) / GO2_D1_GRIPPER_MAX_JAW_TRAVEL_M, 0.0),
        1.0,
    )
    return GO2_D1_GRIPPER_OPEN_HARDWARE_ANGLE_DEG - closure * (
        GO2_D1_GRIPPER_OPEN_HARDWARE_ANGLE_DEG
        - GO2_D1_GRIPPER_CLOSED_HARDWARE_ANGLE_DEG
    )


GO2_D1_MOTION_JOINT_NAMES = GO2_ACTION_JOINT_NAMES + GO2_D1_ARM_JOINT_NAMES
GO2_D1_ACTION_JOINT_NAMES = GO2_D1_MOTION_JOINT_NAMES + ("arm_7_1_joint",)
# Backward-compatible name used by the dedicated manipulation task.
GO2_D1_PICK_STOW_ACTION_JOINT_NAMES = GO2_D1_ACTION_JOINT_NAMES

B2_Z1_ARM_JOINT_NAMES = (
    "joint1",
    "joint2",
    "joint3",
    "joint4",
    "joint5",
    "joint6",
)

B2_Z1_GRIPPER_JOINT_NAMES = ("jointGripper",)

B2_Z1_ACTION_JOINT_NAMES = GO2_ACTION_JOINT_NAMES + B2_Z1_ARM_JOINT_NAMES

GO2_DEFAULT_JOINT_ANGLES = {
    "FL_hip_joint": 0.1,
    "RL_hip_joint": 0.1,
    "FR_hip_joint": -0.1,
    "RR_hip_joint": -0.1,
    "FL_thigh_joint": 0.8,
    "RL_thigh_joint": 1.0,
    "FR_thigh_joint": 0.8,
    "RR_thigh_joint": 1.0,
    "FL_calf_joint": -1.5,
    "RL_calf_joint": -1.5,
    "FR_calf_joint": -1.5,
    "RR_calf_joint": -1.5,
}

GO2_MOTION_BODY_NAMES = ("base", "FL_foot", "FR_foot", "RL_foot", "RR_foot")
