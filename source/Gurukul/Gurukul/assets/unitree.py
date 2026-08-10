"""Configuration for Unitree robots.
Reference: https://github.com/unitreerobotics/unitree_ros
"""

import math

import isaaclab.sim as sim_utils
from isaaclab.actuators import DCMotorCfg, IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils import configclass

from Gurukul.assets import ISAACLAB_ASSETS_DATA_DIR
from Gurukul.assets.lidar_mounts import LidarMountCfg, make_mid360_mount

B2_Z1_LIDAR_POS = (0.342, 0.0, 0.18)
B2_Z1_ARM_MOUNT_DISTANCE_FROM_LIDAR = 0.20
B2_Z1_ARM_MOUNT_POS = (
    B2_Z1_LIDAR_POS[0] - B2_Z1_ARM_MOUNT_DISTANCE_FROM_LIDAR,
    0.0,
    0.140,
)

##
# Configuration
##


@configclass
class UnitreeArticulationCfg(ArticulationCfg):
    joint_sdk_names: list[str] | None = None
    lidar_mounts: dict[str, LidarMountCfg] | None = None
    brainco_revo_hand_interface: dict[str, object] | None = None


UNITREE_GO2_CFG = UnitreeArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/unitree/go2_description/urdf/go2_description.urdf",
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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.38),
        joint_pos={
            "FL_hip_joint": 0.1,
            "FL_thigh_joint": 0.8,
            "FL_calf_joint": -1.5,
            "FR_hip_joint": -0.1,
            "FR_thigh_joint": 0.8,
            "FR_calf_joint": -1.5,
            "RL_hip_joint": 0.1,
            "RL_thigh_joint": 1.0,
            "RL_calf_joint": -1.5,
            "RR_hip_joint": -0.1,
            "RR_thigh_joint": 1.0,
            "RR_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[".*"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=20.0,
            damping=0.5,
            friction=0.0,
        ),
    },
    joint_sdk_names=[
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
    ],
    lidar_mounts={
        "front_mid360": make_mid360_mount(parent_link="base", pos=(0.249, 0.0, 0.135)),
    },
)
"""Configuration of Unitree Go2 using DC motor.
"""

UNITREE_GO2_AIRBOT_ARM_CFG = UnitreeArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/unitree/go2_with_airbot/urdf/go2_with_airbot_vis_flip.urdf",
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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.38),
        joint_pos={
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": -0.0,
            "F.*_thigh_joint": 0.8,
            "R.*_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
            "airbot_.*": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[r"^(FL|FR|RL|RR)_(hip|thigh|calf)_joint$"],
            effort_limit=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
        "arm": DCMotorCfg(
            joint_names_expr=[r"^airbot_j[1-6]$"],
            effort_limit=30.0,
            saturation_effort=30.0,
            velocity_limit=10.0,
            stiffness={
                "airbot_j1": 100.0,
                "airbot_j2": 100.0,
                "airbot_j3": 100.0,
                "airbot_j4": 20.0,
                "airbot_j5": 20.0,
                "airbot_j6": 5.0,
            },
            damping={
                "airbot_j1": 5.0,
                "airbot_j2": 5.0,
                "airbot_j3": 5.0,
                "airbot_j4": 3.0,
                "airbot_j5": 1.0,
                "airbot_j6": 0.5,
            },
            friction=0.0,
        ),
    },
    lidar_mounts={
        "front_mid360": make_mid360_mount(parent_link="base", pos=(0.249, 0.0, 0.135)),
    },
)
"""Configuration of Unitree Go2 with an Airbot arm."""

UNITREE_GO2_D1_ARM_CFG = UnitreeArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/unitree/go2_with_d1/usd/go2_d1_center_gripper.usda",
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
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            fix_root_link=False,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.4),
        joint_pos={
            "FL_hip_joint": 0.1,
            "FL_thigh_joint": 0.8,
            "FL_calf_joint": -1.5,
            "FR_hip_joint": -0.1,
            "FR_thigh_joint": 0.8,
            "FR_calf_joint": -1.5,
            "RL_hip_joint": 0.1,
            "RL_thigh_joint": 1.0,
            "RL_calf_joint": -1.5,
            "RR_hip_joint": -0.1,
            "RR_thigh_joint": 1.0,
            "RR_calf_joint": -1.5,
            "arm_1_joint": 0.0,
            "arm_2_joint": math.radians(-90.0),
            "arm_3_joint": math.radians(90.0),
            "arm_4_joint": 0.0,
            "arm_5_joint": 0.0,
            "arm_6_joint": 0.0,
            "arm_7_1_joint": 0.0,
            "arm_7_2_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    # Preserve the measured D1 gripper endpoints: q=0 open and q=0.033 closed.
    soft_joint_pos_limit_factor=1.0,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[r"^(FL|FR|RL|RR)_(hip|thigh|calf)_joint$"],
            effort_limit=23.5,
            effort_limit_sim=23.5,
            saturation_effort=23.5,
            velocity_limit=30.0,
            velocity_limit_sim=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
        "arm_j1_j2": ImplicitActuatorCfg(
            joint_names_expr=[r"^arm_[12]_joint$"],
            effort_limit=3.3,
            effort_limit_sim=3.3,
            velocity_limit=1.05,
            velocity_limit_sim=1.05,
            stiffness=200.0,
            damping=5.0,
        ),
        "arm_j3": ImplicitActuatorCfg(
            joint_names_expr=[r"^arm_3_joint$"],
            effort_limit=3.3,
            effort_limit_sim=3.3,
            velocity_limit=1.05,
            velocity_limit_sim=1.05,
            stiffness=200.0,
            damping=4.0,
        ),
        "arm_j4": ImplicitActuatorCfg(
            joint_names_expr=[r"^arm_4_joint$"],
            effort_limit=1.7,
            effort_limit_sim=1.7,
            velocity_limit=1.73,
            velocity_limit_sim=1.73,
            stiffness=200.0,
            damping=2.0,
        ),
        "arm_j5_j6": ImplicitActuatorCfg(
            joint_names_expr=[r"^arm_[56]_joint$"],
            effort_limit=1.7,
            effort_limit_sim=1.7,
            velocity_limit=1.73,
            velocity_limit_sim=1.73,
            stiffness=200.0,
            damping=1.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=[r"^arm_7_1_joint$"],
            effort_limit=15.0,
            effort_limit_sim=15.0,
            velocity_limit=0.02,
            velocity_limit_sim=0.02,
            stiffness=200.0,
            damping=3.0,
        ),
    },
    lidar_mounts={
        "front_mid360": make_mid360_mount(parent_link="base", pos=(0.249, 0.0, 0.135)),
    },
)
"""Configuration of Unitree Go2 with a D1 arm and two-finger gripper."""

UNITREE_GO2_D1_ARM_APEX_CFG = UNITREE_GO2_D1_ARM_CFG.replace()
UNITREE_GO2_D1_ARM_APEX_CFG.spawn.usd_path = (
    f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/unitree/go2_with_d1/usd/go2_d1_center_gripper_apex.usda"
)
UNITREE_GO2_D1_ARM_APEX_CFG.spawn.articulation_props.enabled_self_collisions = True
UNITREE_GO2_D1_ARM_APEX_CFG.spawn.articulation_props.solver_position_iteration_count = 8
UNITREE_GO2_D1_ARM_APEX_CFG.spawn.articulation_props.solver_velocity_iteration_count = 2
UNITREE_GO2_D1_ARM_APEX_CFG.actuators["arm_j1_j2"] = IdealPDActuatorCfg(
    joint_names_expr=[r"^arm_[12]_joint$"],
    effort_limit=3.3,
    effort_limit_sim=1.0e9,
    velocity_limit=100.0,
    velocity_limit_sim=100.0,
    stiffness=200.0,
    damping=5.0,
    armature=0.001,
    friction=0.02,
    dynamic_friction=0.02,
    viscous_friction=0.1,
)
UNITREE_GO2_D1_ARM_APEX_CFG.actuators["arm_j3"] = IdealPDActuatorCfg(
    joint_names_expr=[r"^arm_3_joint$"],
    effort_limit=3.3,
    effort_limit_sim=1.0e9,
    velocity_limit=100.0,
    velocity_limit_sim=100.0,
    stiffness=200.0,
    damping=4.0,
    armature=0.001,
    friction=0.02,
    dynamic_friction=0.02,
    viscous_friction=0.1,
)
UNITREE_GO2_D1_ARM_APEX_CFG.actuators["arm_j4"] = IdealPDActuatorCfg(
    joint_names_expr=[r"^arm_4_joint$"],
    effort_limit=1.7,
    effort_limit_sim=1.0e9,
    velocity_limit=100.0,
    velocity_limit_sim=100.0,
    stiffness=50.0,
    damping=0.25,
    armature=0.001,
    friction=0.02,
    dynamic_friction=0.02,
    viscous_friction=0.1,
)
UNITREE_GO2_D1_ARM_APEX_CFG.actuators["arm_j5_j6"] = IdealPDActuatorCfg(
    joint_names_expr=[r"^arm_[56]_joint$"],
    effort_limit=1.7,
    effort_limit_sim=1.0e9,
    velocity_limit=100.0,
    velocity_limit_sim=100.0,
    stiffness=50.0,
    damping=0.25,
    armature=0.001,
    friction=0.02,
    dynamic_friction=0.02,
    viscous_friction=0.1,
)
"""Go2+D1 APEX asset with a lockable fore-aft arm-mount calibration DOF."""

GO2_D1_ARM_JOINT_DEFAULTS = {
    joint_name: float(UNITREE_GO2_D1_ARM_CFG.init_state.joint_pos[joint_name])
    for joint_name in (
        "arm_1_joint",
        "arm_2_joint",
        "arm_3_joint",
        "arm_4_joint",
        "arm_5_joint",
        "arm_6_joint",
        "arm_7_1_joint",
        "arm_7_2_joint",
    )
}
GO2_D1_ARM_READY_POSE = tuple(GO2_D1_ARM_JOINT_DEFAULTS[f"arm_{index}_joint"] for index in range(1, 7))

UNITREE_GO2W_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/unitree/go2w_description/urdf/go2w_description.urdf",
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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=1
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.45),
        joint_pos={
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": -0.0,
            "F.*_thigh_joint": 0.8,
            "R.*_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=["^(?!.*_foot_joint).*"],
            effort_limit_sim=23.5,
            velocity_limit_sim=30.0,
            stiffness=25.0,
            damping=0.5,
            friction=0.0,
        ),
        "wheels": ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_joint"],
            effort_limit_sim=23.5,
            velocity_limit_sim=30.0,
            stiffness=0.0,
            damping=0.5,
            friction=0.0,
        ),
    },
)
"""Configuration of Unitree Go2W using DC motor.
"""

UNITREE_B2_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/unitree/b2_description/urdf/b2_description.urdf",
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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.58),
        joint_pos={
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": -0.0,
            "F.*_thigh_joint": 0.8,
            "R.*_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip": DCMotorCfg(
            joint_names_expr=[".*_hip_joint"],
            effort_limit=200.0,
            saturation_effort=200.0,
            velocity_limit=23.0,
            stiffness=160.0,
            damping=5.0,
            friction=0.0,
        ),
        "thigh": DCMotorCfg(
            joint_names_expr=[".*_thigh_joint"],
            effort_limit=200.0,
            saturation_effort=200.0,
            velocity_limit=23.0,
            stiffness=160.0,
            damping=5.0,
            friction=0.0,
        ),
        "calf": DCMotorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=320.0,
            saturation_effort=320.0,
            velocity_limit=14.0,
            stiffness=160.0,
            damping=5.0,
            friction=0.0,
        ),
    },
)
"""Configuration of Unitree B2 using DC motor.
"""


UNITREE_B2_Z1_ARM_CFG = UnitreeArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        force_usd_conversion=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/unitree/b2_with_z1/urdf/b2_plus_z1.urdf",
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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.6),
        joint_pos={
            ".*L_hip_joint": 0.1,
            ".*R_hip_joint": -0.1,
            "F[L,R]_thigh_joint": 0.8,
            "R[L,R]_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
            "joint1": 0.0,
            "joint2": 0.0,
            "joint3": 0.0,
            "joint4": 0.0,
            "joint5": 0.0,
            "joint6": 0.0,
            "jointGripper": -1.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": DCMotorCfg(
            joint_names_expr=[r"^(FL|FR|RL|RR)_(hip|thigh|calf)_joint$"],
            effort_limit=300.0,
            saturation_effort=300.0,
            velocity_limit=23.0,
            stiffness=360.0,
            damping=5.0,
            friction=0.0,
        ),
        "joint1": ImplicitActuatorCfg(
            joint_names_expr=["joint1"],
            effort_limit=30.0,
            effort_limit_sim=30.0,
            velocity_limit=3.0,
            velocity_limit_sim=3.0,
            stiffness=512.0,
            damping=25.6,
        ),
        "joint2": ImplicitActuatorCfg(
            joint_names_expr=["joint2"],
            effort_limit=60.0,
            effort_limit_sim=60.0,
            velocity_limit=3.0,
            velocity_limit_sim=3.0,
            stiffness=768.0,
            damping=25.6,
        ),
        "joint3": ImplicitActuatorCfg(
            joint_names_expr=["joint3"],
            effort_limit=30.0,
            effort_limit_sim=30.0,
            velocity_limit=3.0,
            velocity_limit_sim=3.0,
            stiffness=768.0,
            damping=25.6,
        ),
        "joint4": ImplicitActuatorCfg(
            joint_names_expr=["joint4"],
            effort_limit=30.0,
            effort_limit_sim=30.0,
            velocity_limit=3.0,
            velocity_limit_sim=3.0,
            stiffness=512.0,
            damping=25.6,
        ),
        "joint5": ImplicitActuatorCfg(
            joint_names_expr=["joint5"],
            effort_limit=30.0,
            effort_limit_sim=30.0,
            velocity_limit=3.0,
            velocity_limit_sim=3.0,
            stiffness=384.0,
            damping=25.6,
        ),
        "joint6": ImplicitActuatorCfg(
            joint_names_expr=["joint6"],
            effort_limit=30.0,
            effort_limit_sim=30.0,
            velocity_limit=3.0,
            velocity_limit_sim=3.0,
            stiffness=256.0,
            damping=25.6,
        ),
        "jointGripper": ImplicitActuatorCfg(
            joint_names_expr=["jointGripper"],
            effort_limit=30.0,
            effort_limit_sim=30.0,
            velocity_limit=3.0,
            velocity_limit_sim=3.0,
            stiffness=512.0,
            damping=25.6,
        ),
    },
    lidar_mounts={
        "front_mid360": make_mid360_mount(parent_link="base", pos=B2_Z1_LIDAR_POS),
    },
)
"""Configuration of Unitree B2 with a Unitree Z1 arm."""


UNITREE_B2W_CFG = ArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        merge_fixed_joints=True,
        replace_cylinders_with_capsules=False,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/unitree/b2w_description/urdf/b2w_description.urdf",
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
            enabled_self_collisions=False, solver_position_iteration_count=4, solver_velocity_iteration_count=0
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.65),
        joint_pos={
            ".*L_hip_joint": 0.0,
            ".*R_hip_joint": -0.0,
            "F.*_thigh_joint": 0.8,
            "R.*_thigh_joint": 0.8,
            ".*_calf_joint": -1.5,
            ".*_foot_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "hip": DCMotorCfg(
            joint_names_expr=[".*_hip_joint"],
            effort_limit=200.0,
            saturation_effort=200.0,
            velocity_limit=23.0,
            stiffness=160.0,
            damping=5.0,
            friction=0.0,
        ),
        "thigh": DCMotorCfg(
            joint_names_expr=[".*_thigh_joint"],
            effort_limit=200.0,
            saturation_effort=200.0,
            velocity_limit=23.0,
            stiffness=160.0,
            damping=5.0,
            friction=0.0,
        ),
        "calf": DCMotorCfg(
            joint_names_expr=[".*_calf_joint"],
            effort_limit=320.0,
            saturation_effort=320.0,
            velocity_limit=14.0,
            stiffness=160.0,
            damping=5.0,
            friction=0.0,
        ),
        "wheel": ImplicitActuatorCfg(
            joint_names_expr=[".*_foot_joint"],
            effort_limit_sim=20.0,
            velocity_limit_sim=50.0,
            stiffness=0.0,
            damping=1.0,
            friction=0.0,
        ),
    },
)
"""Configuration of Unitree B2W using DC motor.
"""


# UNITREE_G1_29DOF_CFG = ArticulationCfg(
#     spawn=sim_utils.UrdfFileCfg(
#         fix_base=False,
#         merge_fixed_joints=True,
#         replace_cylinders_with_capsules=False,
#         asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/unitree/g1_description/urdf/g1_29dof_rev_1_0.urdf",
#         activate_contact_sensors=True,
#         rigid_props=sim_utils.RigidBodyPropertiesCfg(
#             disable_gravity=False,
#             retain_accelerations=False,
#             linear_damping=0.0,
#             angular_damping=0.0,
#             max_linear_velocity=3.0,
#             max_angular_velocity=3.0,
#             max_depenetration_velocity=10.0,
#         ),
#         articulation_props=sim_utils.ArticulationRootPropertiesCfg(
#             enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
#         ),
#         joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
#             gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
#         ),
#     ),
#     init_state=ArticulationCfg.InitialStateCfg(
#         pos=(0.0, 0.0, 0.8),
#         joint_pos={
#             ".*_hip_pitch_joint": -0.1,
#             ".*_hip_roll_joint": 0.0,
#             ".*_hip_yaw_joint": 0.0,
#             ".*_knee_joint": 0.3,
#             ".*_ankle_pitch_joint": -0.2,
#             ".*_ankle_roll_joint": 0.0,
#             "waist_yaw_joint": 0.0,
#             "waist_roll_joint": 0.0,
#             "waist_pitch_joint": 0.0,
#             ".*_shoulder_pitch_joint": 0.0,
#             ".*_shoulder_roll_joint": 0.0,
#             ".*_shoulder_yaw_joint": 0.0,
#             ".*_elbow_joint": 0.0,
#             ".*_wrist_roll_joint": 0.0,
#             ".*_wrist_pitch_joint": 0.0,
#             ".*_wrist_yaw_joint": 0.0,
#         },
#         joint_vel={".*": 0.0},
#     ),
#     soft_joint_pos_limit_factor=0.9,
#     actuators={
#         "legs": ImplicitActuatorCfg(
#             joint_names_expr=[
#                 ".*_hip_pitch_joint",
#                 ".*_hip_roll_joint",
#                 ".*_hip_yaw_joint",
#                 ".*_knee_joint",
#             ],
#             effort_limit_sim=300,
#             velocity_limit_sim=100.0,
#             stiffness={
#                 ".*_hip_pitch_joint": 200.0,
#                 ".*_hip_roll_joint": 150.0,
#                 ".*_hip_yaw_joint": 150.0,
#                 ".*_knee_joint": 200.0,
#             },
#             damping={
#                 ".*_hip_pitch_joint": 5.0,
#                 ".*_hip_roll_joint": 5.0,
#                 ".*_hip_yaw_joint": 5.0,
#                 ".*_knee_joint": 5.0,
#             },
#             armature={
#                 ".*_hip_.*": 0.01,
#                 ".*_knee_joint": 0.01,
#             },
#         ),
#         "waist": ImplicitActuatorCfg(
#             joint_names_expr=["waist_.*_joint"],
#             effort_limit_sim=300,
#             velocity_limit_sim=100.0,
#             stiffness={
#                 "waist_yaw_joint": 200.0,
#                 "waist_roll_joint": 200.0,
#                 "waist_pitch_joint": 200.0,
#             },
#             damping={
#                 "waist_yaw_joint": 5.0,
#                 "waist_roll_joint": 5.0,
#                 "waist_pitch_joint": 5.0,
#             },
#             armature={
#                 "waist_yaw_joint": 0.01,
#                 "waist_roll_joint": 0.01,
#                 "waist_pitch_joint": 0.01,
#             },
#         ),
#         "feet": ImplicitActuatorCfg(
#             effort_limit_sim=20,
#             joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
#             stiffness=20.0,
#             damping=2.0,
#             armature=0.01,
#         ),
#         "arms": ImplicitActuatorCfg(
#             joint_names_expr=[
#                 ".*_shoulder_pitch_joint",
#                 ".*_shoulder_roll_joint",
#                 ".*_shoulder_yaw_joint",
#                 ".*_elbow_joint",
#                 ".*_wrist_.*",
#             ],
#             effort_limit_sim=300,
#             velocity_limit_sim=100.0,
#             stiffness=40.0,
#             damping=10.0,
#             armature={
#                 ".*_shoulder_.*": 0.01,
#                 ".*_elbow_.*": 0.01,
#                 ".*_wrist_.*": 0.01,
#             },
#         ),
#     },
# )


ARMATURE_5020 = 0.003609725
ARMATURE_7520_14 = 0.010177520
ARMATURE_7520_22 = 0.025101925
ARMATURE_4010 = 0.00425

NATURAL_FREQ = 10 * 2.0 * 3.1415926535  # 10Hz
DAMPING_RATIO = 2.0

STIFFNESS_5020 = ARMATURE_5020 * NATURAL_FREQ**2
STIFFNESS_7520_14 = ARMATURE_7520_14 * NATURAL_FREQ**2
STIFFNESS_7520_22 = ARMATURE_7520_22 * NATURAL_FREQ**2
STIFFNESS_4010 = ARMATURE_4010 * NATURAL_FREQ**2

DAMPING_5020 = 2.0 * DAMPING_RATIO * ARMATURE_5020 * NATURAL_FREQ
DAMPING_7520_14 = 2.0 * DAMPING_RATIO * ARMATURE_7520_14 * NATURAL_FREQ
DAMPING_7520_22 = 2.0 * DAMPING_RATIO * ARMATURE_7520_22 * NATURAL_FREQ
DAMPING_4010 = 2.0 * DAMPING_RATIO * ARMATURE_4010 * NATURAL_FREQ

UNITREE_G1_29DOF_CFG = UnitreeArticulationCfg(
    spawn=sim_utils.UrdfFileCfg(
        fix_base=False,
        replace_cylinders_with_capsules=True,
        asset_path=f"{ISAACLAB_ASSETS_DATA_DIR}/Robots/unitree/g1_description/urdf/g1_29dof_rev_1_0.urdf",
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
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.76),
        joint_pos={
            ".*_hip_pitch_joint": -0.312,
            ".*_knee_joint": 0.669,
            ".*_ankle_pitch_joint": -0.363,
            ".*_elbow_joint": 0.6,
            "left_shoulder_roll_joint": 0.2,
            "left_shoulder_pitch_joint": 0.2,
            "right_shoulder_roll_joint": -0.2,
            "right_shoulder_pitch_joint": 0.2,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit_sim={
                ".*_hip_yaw_joint": 88.0,
                ".*_hip_roll_joint": 139.0,
                ".*_hip_pitch_joint": 88.0,
                ".*_knee_joint": 139.0,
            },
            velocity_limit_sim={
                ".*_hip_yaw_joint": 32.0,
                ".*_hip_roll_joint": 20.0,
                ".*_hip_pitch_joint": 32.0,
                ".*_knee_joint": 20.0,
            },
            stiffness={
                ".*_hip_pitch_joint": STIFFNESS_7520_14,
                ".*_hip_roll_joint": STIFFNESS_7520_22,
                ".*_hip_yaw_joint": STIFFNESS_7520_14,
                ".*_knee_joint": STIFFNESS_7520_22,
            },
            damping={
                ".*_hip_pitch_joint": DAMPING_7520_14,
                ".*_hip_roll_joint": DAMPING_7520_22,
                ".*_hip_yaw_joint": DAMPING_7520_14,
                ".*_knee_joint": DAMPING_7520_22,
            },
            armature={
                ".*_hip_pitch_joint": ARMATURE_7520_14,
                ".*_hip_roll_joint": ARMATURE_7520_22,
                ".*_hip_yaw_joint": ARMATURE_7520_14,
                ".*_knee_joint": ARMATURE_7520_22,
            },
        ),
        "feet": ImplicitActuatorCfg(
            effort_limit_sim=50.0,
            velocity_limit_sim=37.0,
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        "waist": ImplicitActuatorCfg(
            effort_limit_sim=50,
            velocity_limit_sim=37.0,
            joint_names_expr=["waist_roll_joint", "waist_pitch_joint"],
            stiffness=2.0 * STIFFNESS_5020,
            damping=2.0 * DAMPING_5020,
            armature=2.0 * ARMATURE_5020,
        ),
        "waist_yaw": ImplicitActuatorCfg(
            effort_limit_sim=88,
            velocity_limit_sim=32.0,
            joint_names_expr=["waist_yaw_joint"],
            stiffness=STIFFNESS_7520_14,
            damping=DAMPING_7520_14,
            armature=ARMATURE_7520_14,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_roll_joint",
                ".*_wrist_pitch_joint",
                ".*_wrist_yaw_joint",
            ],
            effort_limit_sim={
                ".*_shoulder_pitch_joint": 25.0,
                ".*_shoulder_roll_joint": 25.0,
                ".*_shoulder_yaw_joint": 25.0,
                ".*_elbow_joint": 25.0,
                ".*_wrist_roll_joint": 25.0,
                ".*_wrist_pitch_joint": 5.0,
                ".*_wrist_yaw_joint": 5.0,
            },
            velocity_limit_sim={
                ".*_shoulder_pitch_joint": 37.0,
                ".*_shoulder_roll_joint": 37.0,
                ".*_shoulder_yaw_joint": 37.0,
                ".*_elbow_joint": 37.0,
                ".*_wrist_roll_joint": 37.0,
                ".*_wrist_pitch_joint": 22.0,
                ".*_wrist_yaw_joint": 22.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": STIFFNESS_5020,
                ".*_shoulder_roll_joint": STIFFNESS_5020,
                ".*_shoulder_yaw_joint": STIFFNESS_5020,
                ".*_elbow_joint": STIFFNESS_5020,
                ".*_wrist_roll_joint": STIFFNESS_5020,
                ".*_wrist_pitch_joint": STIFFNESS_4010,
                ".*_wrist_yaw_joint": STIFFNESS_4010,
            },
            damping={
                ".*_shoulder_pitch_joint": DAMPING_5020,
                ".*_shoulder_roll_joint": DAMPING_5020,
                ".*_shoulder_yaw_joint": DAMPING_5020,
                ".*_elbow_joint": DAMPING_5020,
                ".*_wrist_roll_joint": DAMPING_5020,
                ".*_wrist_pitch_joint": DAMPING_4010,
                ".*_wrist_yaw_joint": DAMPING_4010,
            },
            armature={
                ".*_shoulder_pitch_joint": ARMATURE_5020,
                ".*_shoulder_roll_joint": ARMATURE_5020,
                ".*_shoulder_yaw_joint": ARMATURE_5020,
                ".*_elbow_joint": ARMATURE_5020,
                ".*_wrist_roll_joint": ARMATURE_5020,
                ".*_wrist_pitch_joint": ARMATURE_4010,
                ".*_wrist_yaw_joint": ARMATURE_4010,
            },
        ),
    },
    joint_sdk_names=[
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ],
    lidar_mounts={
        "head_mid360": make_mid360_mount(parent_link="head_link", pos=(0.0, 0.0, 0.08)),
    },
)

UNITREE_G1_BRAINCO_REVO2_HAND_JOINT_NAMES = [
    "thumb",
    "thumb_aux",
    "index",
    "middle",
    "ring",
    "pinky",
]
"""Normalized command channels used by BrainCo's Unitree G1 + Revo2 ROS2 bridge."""

UNITREE_G1_BRAINCO_REVO2_GESTURES = {
    "open": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    "like": [0.0, 0.0, 0.9, 0.9, 0.9, 0.9],
    "handshake": [0.3, 0.3, 0.1, 0.1, 0.1, 0.1],
    "rock": [0.7, 0.0, 0.9, 0.9, 0.9, 0.9],
    "scissors": [0.7, 0.0, 0.0, 0.0, 0.9, 0.9],
    "paper": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
}
"""Representative normalized Revo2 gestures from BrainCoTech/unitree-g1-brainco-hand."""

UNITREE_G1_BRAINCO_REVO2_CFG = UNITREE_G1_29DOF_CFG.replace(
    brainco_revo_hand_interface={
        "source": "https://github.com/BrainCoTech/unitree-g1-brainco-hand",
        "robot_dof_options": [23, 29],
        "sim_robot_dof": 29,
        "hand_model": "BrainCo Revo2",
        "hardware_topics": {
            "both": "/joint_commands",
            "left": "/joint_commands_left",
            "right": "/joint_commands_right",
        },
        "normalized_joint_names": UNITREE_G1_BRAINCO_REVO2_HAND_JOINT_NAMES,
        "normalized_range": [0.0, 1.0],
        "gestures": UNITREE_G1_BRAINCO_REVO2_GESTURES,
        "sim_note": (
            "The upstream G1+BrainCo repository provides ROS2 hardware control, not a full Isaac Lab robot asset. "
            "This profile uses the local Unitree G1 29-DOF simulation asset and records the Revo2 command contract "
            "for policy/export/sim2real alignment."
        ),
    }
)

UNITREE_G1_29DOF_ACTION_SCALE = {}
for a in UNITREE_G1_29DOF_CFG.actuators.values():
    e = a.effort_limit_sim
    s = a.stiffness
    names = a.joint_names_expr
    if not isinstance(e, dict):
        e = {n: e for n in names}
    if not isinstance(s, dict):
        s = {n: s for n in names}
    for n in names:
        if n in e and n in s and s[n]:
            UNITREE_G1_29DOF_ACTION_SCALE[n] = 0.25 * e[n] / s[n]
