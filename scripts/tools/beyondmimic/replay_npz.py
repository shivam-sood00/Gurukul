"""This script demonstrates how to use the interactive scene interface to setup a scene with multiple prims.

.. code-block:: bash

    # Usage
    python replay_npz.py -f path_to_motion.npz
"""

"""Launch Isaac Sim Simulator first."""

import argparse

import numpy as np
import torch

from isaaclab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Replay converted motions.")
parser.add_argument("--file", "-f", type=str, required=True)
parser.add_argument(
    "--robot",
    choices=("g1", "pm01-24dof", "t800"),
    default="g1",
    help="Robot asset to use for replay.",
)

# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""

##
# Pre-defined configs
##
from Gurukul.assets.engineai import ENGINEAI_T800_CFG
from Gurukul.assets.engineai_pm01_official import ENGINEAI_PM01_24DOF_CFG
from Gurukul.assets.unitree import UNITREE_G1_29DOF_CFG
from Gurukul.tasks.manager_based.beyondmimic.mdp import MotionLoader

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR


def _robot_cfg() -> ArticulationCfg:
    if args_cli.robot == "pm01-24dof":
        return ENGINEAI_PM01_24DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    if args_cli.robot == "t800":
        return ENGINEAI_T800_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    return UNITREE_G1_29DOF_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")


@configclass
class ReplayMotionsSceneCfg(InteractiveSceneCfg):
    """Configuration for a replay motions scene."""

    ground = AssetBaseCfg(prim_path="/World/defaultGroundPlane", spawn=sim_utils.GroundPlaneCfg())

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=750.0,
            texture_file=f"{ISAAC_NUCLEUS_DIR}/Materials/Textures/Skies/PolyHaven/kloofendal_43d_clear_puresky_4k.hdr",
        ),
    )

    # articulation
    robot: ArticulationCfg = _robot_cfg()


def _fit_joint_state_to_robot(
    robot: Articulation, joint_pos: torch.Tensor, joint_vel: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Match replay joint columns to the articulation's simulated joints."""
    expected_joint_count = robot.data.default_joint_pos.shape[1]
    replay_joint_count = joint_pos.shape[1]
    if replay_joint_count == expected_joint_count:
        return joint_pos, joint_vel
    if replay_joint_count > expected_joint_count:
        return joint_pos[:, :expected_joint_count], joint_vel[:, :expected_joint_count]

    replay_joint_pos = robot.data.default_joint_pos.clone()
    replay_joint_vel = robot.data.default_joint_vel.clone()
    replay_joint_pos[:, :replay_joint_count] = joint_pos
    replay_joint_vel[:, :replay_joint_count] = joint_vel
    return replay_joint_pos, replay_joint_vel


def run_simulator(sim: sim_utils.SimulationContext, scene: InteractiveScene):
    # Extract scene entities
    robot: Articulation = scene["robot"]
    # Define simulation stepping
    sim_dt = sim.get_physics_dt()

    motion = MotionLoader(
        args_cli.file,
        torch.tensor([0], dtype=torch.long, device=sim.device),
        sim.device,
        joint_names=robot.joint_names,
    )
    time_steps = torch.zeros(scene.num_envs, dtype=torch.long, device=sim.device)

    # Simulation loop
    while simulation_app.is_running():
        time_steps += 1
        reset_ids = time_steps >= motion.time_step_total
        time_steps[reset_ids] = 0

        root_states = robot.data.default_root_state.clone()
        root_states[:, :3] = motion.body_pos_w[time_steps][:, 0] + scene.env_origins
        root_states[:, 3:7] = motion.body_quat_w[time_steps][:, 0]
        root_states[:, 7:10] = motion.body_lin_vel_w[time_steps][:, 0]
        root_states[:, 10:] = motion.body_ang_vel_w[time_steps][:, 0]

        robot.write_root_state_to_sim(root_states)
        joint_pos, joint_vel = _fit_joint_state_to_robot(
            robot, motion.joint_pos[time_steps], motion.joint_vel[time_steps]
        )
        robot.write_joint_state_to_sim(joint_pos, joint_vel)
        scene.write_data_to_sim()
        sim.render()  # We don't want physic (sim.step())
        scene.update(sim_dt)

        pos_lookat = root_states[0, :3].cpu().numpy()
        sim.set_camera_view(pos_lookat + np.array([2.0, 2.0, 0.5]), pos_lookat)


def main():
    sim_cfg = sim_utils.SimulationCfg(device=args_cli.device)
    sim_cfg.dt = 0.02
    sim = SimulationContext(sim_cfg)

    scene_cfg = ReplayMotionsSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)
    sim.reset()
    # Run the simulator
    run_simulator(sim, scene)


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()
