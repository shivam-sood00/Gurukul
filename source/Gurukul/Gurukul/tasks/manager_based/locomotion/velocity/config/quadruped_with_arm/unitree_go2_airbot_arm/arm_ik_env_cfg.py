# SPDX-License-Identifier: Apache-2.0

from isaaclab.managers import CurriculumTermCfg, EventTermCfg, SceneEntityCfg

import Gurukul.tasks.manager_based.locomotion.velocity.mdp as mdp


def configure_realistic_arm_ik_motion(env_cfg, *, visualize: bool = False) -> None:
    """Drive the Airbot arm through smooth safe joint-space primitives while the policy controls Go2 legs."""
    arm_entity_cfg = SceneEntityCfg(
        "robot",
        joint_names=[f"airbot_j{i}" for i in range(1, 7)],
        body_names=["eef_end_link"],
    )

    # The locomotion policy only controls the 12 Go2 leg joints. The arm is an external,
    # observable moving payload driven by IK.
    env_cfg.actions.joint_pos.scale = {
        r"^(FL|FR|RL|RR)_hip_joint$": 0.125,
        r"^(FL|FR|RL|RR)_(thigh|calf)_joint$": 0.25,
    }
    env_cfg.actions.joint_pos.clip = {".*": (-100.0, 100.0)}
    env_cfg.actions.joint_pos.joint_names = env_cfg.leg_joint_names

    # All-zero Airbot joints fold the arm over the Go2 trunk. Start ArmMoving tasks
    # from a forward ready pose so IK moves inside the deployment workspace.
    env_cfg.scene.robot.init_state.joint_pos.pop("airbot_.*", None)
    env_cfg.scene.robot.init_state.joint_pos.update(
        {
            "airbot_j1": 0.0,
            "airbot_j2": -1.0,
            "airbot_j3": 1.2,
            "airbot_j4": 0.0,
            "airbot_j5": 0.0,
            "airbot_j6": 0.0,
        }
    )

    env_cfg.observations.policy.joint_pos.params["asset_cfg"].joint_names = env_cfg.all_joint_names
    env_cfg.observations.policy.joint_vel.params["asset_cfg"].joint_names = env_cfg.all_joint_names

    env_cfg.events.randomize_push_robot.func = mdp.random_arm_joint_motion
    env_cfg.events.randomize_push_robot.interval_range_s = (2.0, 3.5)
    env_cfg.events.randomize_push_robot.params = {
        "asset_cfg": arm_entity_cfg,
        "visualize": visualize,
        "max_joint_change": 0.35,
        "motion_speed": 0.45,
        "motion_primitives": ("pick_place", "reach", "sweep", "stow"),
        # FK-checked Airbot poses: j2/j3 keep the elbow above and forward of Go2;
        # j1 sweeps laterally only after the shoulder is lifted.
        "joint_motion_library": {
            "stow": (
                (0.0, -1.0, 1.2, 0.0, 0.0, 0.0),
            ),
            "reach": (
                (0.0, -1.0, 1.2, 0.0, 0.0, 0.0),
                (0.0, -1.3, 1.7, 0.0, 0.2, 0.0),
                (0.0, -1.55, 2.05, 0.0, 0.35, 0.0),
                (0.0, -1.3, 1.7, 0.0, 0.2, 0.0),
            ),
            "pick_place": (
                (0.0, -1.0, 1.2, 0.0, 0.0, 0.0),
                (0.0, -1.1, 1.55, 0.0, 0.1, 0.0),
                (0.0, -1.45, 1.75, 0.0, 0.3, 0.0),
                (0.35, -1.25, 1.65, 0.0, 0.15, 0.0),
                (0.35, -1.1, 1.55, 0.0, 0.1, 0.0),
                (0.0, -1.0, 1.2, 0.0, 0.0, 0.0),
            ),
            "sweep": (
                (0.0, -1.3, 1.7, 0.0, 0.2, 0.0),
                (0.35, -1.25, 1.65, 0.0, 0.15, 0.0),
                (0.0, -1.3, 1.7, 0.0, 0.2, 0.0),
                (-0.35, -1.25, 1.65, 0.0, 0.15, 0.0),
                (0.0, -1.0, 1.2, 0.0, 0.0, 0.0),
            ),
        },
    }

    env_cfg.events.continuous_arm_tracking = EventTermCfg(
        func=mdp.continuous_arm_joint_tracking,
        mode="interval",
        interval_range_s=(0.0, 0.0),
        params={"asset_cfg": arm_entity_cfg},
    )

    env_cfg.curriculum.command_levels_lin_vel = None
    env_cfg.curriculum.command_levels_ang_vel = None
    env_cfg.curriculum.loco_manipulation_training_stages = CurriculumTermCfg(
        func=mdp.loco_manipulation_training_stages,
        params={
            # RSL-RL Airbot configs use num_steps_per_env=24. For 5000 iterations,
            # one-third and two-third boundaries are 40k and 80k env steps.
            "stage_steps": (40_000, 80_000),
            "walking_lin_vel_x": (-1.0, 1.0),
            "walking_lin_vel_y": (-0.5, 0.5),
            "walking_ang_vel_z": (-1.0, 1.0),
            "command_name": "base_velocity",
        },
    )
