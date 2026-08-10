import os

from isaaclab.utils import configclass

from Gurukul.assets.unitree import UNITREE_GO2_CFG
from Gurukul.tasks.manager_based.go2_apex.constants import (
    GO2_ACTION_JOINT_NAMES,
    GO2_DEFAULT_JOINT_ANGLES,
    GO2_MOTION_BODY_NAMES,
)
from Gurukul.tasks.manager_based.go2_apex.tracking_env_cfg import Go2ApexEnvCfg


@configclass
class UnitreeGo2ApexFlatEnvCfg(Go2ApexEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.robot = UNITREE_GO2_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.robot.init_state.joint_pos = GO2_DEFAULT_JOINT_ANGLES
        self.scene.contact_forces.debug_vis = False

        self.actions.joint_pos.scale = 0.25
        self.actions.joint_pos.joint_names = list(GO2_ACTION_JOINT_NAMES)
        self.actions.joint_pos.preserve_order = True

        # Match Isaac Lab Go2 velocity-flat domain randomization settings.
        self.scene.robot.actuators["legs"].stiffness = 20.0
        self.scene.robot.actuators["legs"].damping = 0.5
        self.events.randomize_actuator_gains.params["stiffness_distribution_params"] = (0.8, 1.2)
        self.events.randomize_actuator_gains.params["damping_distribution_params"] = (0.8, 1.2)

        # Default reference motion (override from CLI with --motion-file when needed).
        # Keep this aligned with the expected default train/play dataset to avoid replay mismatches.
        self.commands.motion.motion_file = (
            f"{os.path.dirname(__file__)}/motion/npz/animal_mocap/go2_retarget_canter_2ms.npz"
        )
        self.commands.motion.anchor_body_name = "base"
        self.commands.motion.body_names = list(GO2_MOTION_BODY_NAMES)
        self.commands.motion.debug_vis = False
