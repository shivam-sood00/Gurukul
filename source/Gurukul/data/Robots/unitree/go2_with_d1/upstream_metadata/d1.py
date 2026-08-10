# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

USD_DIR_PATH = os.path.join(os.path.dirname(os.path.realpath(__file__)), "usds")

""" D1 base configuration using TBD actuator model."""
D1_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{USD_DIR_PATH}/d1/d1.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
            fix_root_link=True,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0),
        joint_pos={".*": 0.0},
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=100.0,
            velocity_limit_sim=100.0,
            effort_limit=87.0,
            effort_limit_sim=87.0,
            stiffness=25.0,
            damping=0.5,
        )
    },
)


JOINT_ORDER_ARM = ["arm_1_joint", "arm_2_joint", "arm_3_joint", "arm_4_joint", "arm_5_joint", "arm_6_joint"]
JOINT_ORDER_GRIPPER = ["arm_7_1_joint", "arm_7_2_joint"]

D1_CONTACT_PAD_FLAT_CFG = D1_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
D1_CONTACT_PAD_FLAT_CFG.spawn.usd_path = f"{USD_DIR_PATH}/d1/d1_contact_pad_flat.usd"

D1_IDEAL_PD_ACTUATOR_CFG = D1_CFG.replace(
    actuators={
        "arm": IdealPDActuatorCfg(
            joint_names_expr=[".*"],
            velocity_limit=100.0,
            effort_limit=87.0,
            stiffness=250.0,
            damping=250.0,
        ),
    },
)
