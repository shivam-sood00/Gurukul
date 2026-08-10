# SPDX-License-Identifier: Apache-2.0

"""Hierarchical Go2+D1 pick tasks that command a frozen WBC policy."""

from __future__ import annotations

import copy

from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass

import Gurukul.tasks.manager_based.locomotion.velocity.mdp as mdp

from .loco_manipulation_env_cfg import (
    UnitreeGo2D1PickFlatEnvCfg,
    _configure_collision_safety,
    _configure_compact_collision_safety,
)
from .whole_body_controller_env_cfg import (
    GO2_D1_WBC_WORKSPACE_READY_POSE,
    UnitreeGo2D1LegWbcAsyncArmFlatEnvCfg,
    UnitreeGo2D1WholeBodyControllerFlatEnvCfg,
    configure_go2_d1_leg_wbc_ee_hierarchical_runtime,
    configure_go2_d1_leg_wbc_arm_hierarchical_runtime,
    configure_go2_d1_wbc_hierarchical_runtime,
)


@configclass
class _UnitreeGo2D1HighLevelPickSourceCfg(UnitreeGo2D1PickFlatEnvCfg):
    """Task geometry and randomization used by the high-level teacher.

    The cylinder stays comfortably inside the D1 gripper aperture at every
    scale, and absolute mass randomization enforces the 400 g payload limit.
    """

    object_radius = 0.028
    object_height = 0.10
    object_mass = 0.10
    table_center = (0.65, 0.0, 0.54)
    object_table_offset = (-0.12, 0.0)
    object_init_pos = (0.53, 0.0, 0.613)
    target_lift_height = 0.75
    approach_distance_start = 0.65
    approach_distance_end = 1.20
    standoff_distance = 0.55
    manipulation_reach = 0.60

    def __post_init__(self):
        super().__post_init__()

        self.episode_length_s = 12.0
        self.events.reset_object.params.update(
            {
                "object_pose_range": {
                    "x": (-0.020, 0.020),
                    "y": (-0.080, 0.080),
                    "z": (0.0, 0.003),
                    "roll": (-0.015, 0.015),
                    "pitch": (-0.015, 0.015),
                    "yaw": (-3.14159, 3.14159),
                },
                "object_velocity_range": {
                    "x": (-0.01, 0.01),
                    "y": (-0.01, 0.01),
                    "z": (0.0, 0.01),
                    "roll": (-0.02, 0.02),
                    "pitch": (-0.02, 0.02),
                    "yaw": (-0.02, 0.02),
                },
                "sample_distance_within_curriculum": True,
                "easy_replay_fraction": 0.15,
                "frontier_fraction": 0.25,
            }
        )
        self.events.randomize_object_scale.params["scale_range"] = (0.95, 1.05)
        self.events.randomize_object_material.params.update(
            {
                "static_friction_range": (0.70, 1.30),
                "dynamic_friction_range": (0.50, 0.95),
            }
        )
        self.events.randomize_object_mass.func = mdp.randomize_rigid_body_mass_fn
        self.events.randomize_object_mass.params.update(
            {
                "mass_distribution_params": (0.04, 0.25),
                "operation": "abs",
                "recompute_inertia": True,
            }
        )
        self.events.randomize_reset_base.params["pose_range"] = {
            "x": (-0.025, 0.025),
            "y": (-0.035, 0.035),
            "yaw": (-0.08, 0.08),
        }


def _configure_hierarchical_pick_policy_obs(env_cfg, pick_policy) -> None:
    """Keep object-centric observations on the high-level actor; WBC obs live in `wbc_policy`."""
    low_level_critic = copy.deepcopy(env_cfg.observations.critic)
    wbc_only_terms = (
        "base_lin_vel",
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
        "joint_pos",
        "joint_vel",
        "actions",
        "height_scan",
        "arm_ee_command_pos",
        "arm_ee_command_error",
        "gripper_command_pos",
        "gripper_command_error",
    )
    for term_name in wbc_only_terms:
        if hasattr(env_cfg.observations.policy, term_name):
            setattr(env_cfg.observations.policy, term_name, None)

    for term_name in dir(pick_policy):
        if term_name.startswith("_") or term_name in wbc_only_terms:
            continue
        term = getattr(pick_policy, term_name)
        if isinstance(term, ObsTerm):
            setattr(env_cfg.observations.policy, term_name, copy.deepcopy(term))

    # The high-level policy should observe its own previous command. The frozen WBC
    # receives its previous low-level action through the separate `wbc_policy` group.
    if getattr(pick_policy, "actions", None) is not None:
        env_cfg.observations.policy.actions = copy.deepcopy(pick_policy.actions)

    env_cfg.observations.critic = copy.deepcopy(env_cfg.observations.policy)
    if low_level_critic is not None:
        for term_name in dir(low_level_critic):
            if term_name.startswith("_"):
                continue
            term = getattr(low_level_critic, term_name)
            if isinstance(term, ObsTerm) and getattr(env_cfg.observations.critic, term_name, None) is None:
                setattr(env_cfg.observations.critic, term_name, copy.deepcopy(term))


def _configure_cartesian_teacher_observations(env_cfg, pick_cfg) -> None:
    """Build deployable student and privileged teacher groups with short history."""
    # Restore the high-level proprioception that was moved to the frozen WBC's
    # private observation group. The student-facing group must still know the
    # measured base motion/posture and arm state; it never consumes the frozen
    # policy's low-level action history.
    for term_name in (
        "base_lin_vel",
        "base_ang_vel",
        "projected_gravity",
        "velocity_commands",
        "joint_pos",
        "joint_vel",
        "arm_ee_command_pos",
        "arm_ee_command_error",
        "gripper_command_pos",
        "gripper_command_error",
    ):
        term = getattr(env_cfg.observations.wbc_policy, term_name, None)
        if isinstance(term, ObsTerm):
            setattr(env_cfg.observations.policy, term_name, copy.deepcopy(term))

    object_cfg = SceneEntityCfg("object")
    object_pose = ObsTerm(
        func=mdp.object_pose_6d_b,
        params={"object_cfg": object_cfg},
        clip=(-5.0, 5.0),
        scale=1.0,
    )
    standoff_error = ObsTerm(
        func=mdp.base_to_object_standoff_b,
        params={
            "standoff_distance": float(pick_cfg.standoff_distance),
            "object_cfg": object_cfg,
        },
        clip=(-5.0, 5.0),
        scale=1.0,
    )
    for group_name in ("policy", "critic"):
        group = getattr(env_cfg.observations, group_name)
        # Exact object SE(3), stage, and proprioception supersede the inherited
        # repeated point clouds/skill flag. Removing static geometry and frozen-
        # WBC foot diagnostics from every history frame cuts hundreds of inputs.
        for redundant_term in (
            "object_geometry_points",
            "table_geometry_points",
            "loco_manipulation_skill",
            "object_height",
            "feet_air_stance_time",
            "feet_contact_forces",
        ):
            if hasattr(group, redundant_term):
                setattr(group, redundant_term, None)
        # The complete pose supersedes the position + up-axis fragments.
        group.object_position = None
        group.object_up_axis = None
        group.object_pose_6d = copy.deepcopy(object_pose)
        group.base_to_object_standoff = copy.deepcopy(standoff_error)
        group.actions = ObsTerm(
            func=mdp.frozen_wbc_processed_actions,
            params={"action_name": "wbc_command"},
            clip=(-1.0, 1.0),
            scale=1.0,
        )
        group.history_length = 5
        group.flatten_history_dim = True

    # The teacher sees the complete critic state (including exact robot
    # proprioception, contacts, object state, and stage variables). Retaining
    # `policy` separately gives a clean target interface for later distillation.
    env_cfg.observations.teacher = copy.deepcopy(env_cfg.observations.critic)
    env_cfg.observations.teacher.enable_corruption = False
    env_cfg.observations.teacher.history_length = 5
    env_cfg.observations.teacher.flatten_history_dim = True


def _configure_smooth_high_level_actions(env_cfg, command_action_indices: tuple[int, ...]) -> None:
    """Teach smooth command sequences without filtering the policy's actions."""
    env_cfg.rewards.action_rate_l2.weight = -0.01
    env_cfg.rewards.action_smoothness_l2.weight = -0.005
    # Stage-1 arm action indices do not apply to the smaller high-level action.
    env_cfg.rewards.arm_action_rate_l2 = None
    env_cfg.rewards.arm_action_smoothness_l2 = None
    env_cfg.rewards.command_action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2_selected,
        weight=-0.08,
        params={"action_indices": command_action_indices},
    )
    env_cfg.rewards.command_action_smoothness_l2 = RewTerm(
        func=mdp.action_smoothness_l2_selected,
        weight=-0.04,
        params={"action_indices": command_action_indices},
    )


_CARTESIAN_HIGH_LEVEL_DISABLED_REWARDS = (
    # Velocity tracking and gait formation are Stage-1 objectives. The frozen leg
    # WBC already owns them; retaining them here rewards motion instead of picking.
    "track_lin_vel_xy_exp",
    "track_ang_vel_z_exp",
    "feet_air_time",
    "feet_air_time_variance",
    "feet_gait",
    "feet_contact_without_cmd",
    "feet_slide",
    "feet_height_body",
    # These terms regularize the frozen policy's leg targets and mechanics. They
    # are not high-level commands and therefore create an indirect, duplicated cost.
    "joint_torques_l2",
    "joint_acc_l2",
    "joint_pos_limits",
    "joint_power",
    "stand_still",
    "joint_pos_penalty",
    "joint_mirror",
    "contact_forces",
    # Replace the inherited all-action penalties with explicit base/manipulation
    # command subsets below, so their scales are intentional and non-overlapping.
    "action_rate_l2",
    "action_smoothness_l2",
    "arm_action_rate_l2",
    "arm_action_smoothness_l2",
)

_CARTESIAN_PICK_REWARD_STAGES = {
    "base_approach_outside_arm_reach": (0,),
    "base_velocity_towards_standoff": (0,),
    "base_standoff_error": (0, 1, 2, 3),
    "base_facing_error": (0, 1, 2, 3),
    "ee_reach_progress": (1,),
    "ee_reach_alignment_error": (0, 1),
    "ee_grasp_alignment_error": (2,),
    "grasp_frame_horizontal_error": (0, 1, 2, 3),
    "gripper_closed_before_pregrasp": (0, 1),
    "gripper_close_near_object": (2,),
    "gripper_object_contact": (2, 3),
    "object_lifted": (2, 3),
    "object_lifted_near_ee": (2, 3),
    "object_lifted_with_gripper_contact": (2, 3),
    "object_hold_lifted": (3,),
    "object_lifted_without_gripper_contact": (2, 3),
    "object_xy_motion_before_lift": (0, 1, 2),
    "object_vertical": (3,),
}


def _configure_cartesian_high_level_rewards(env_cfg, pick_cfg) -> None:
    """Keep Stage 2 manipulation-centric while preserving command-level safety."""
    for reward_name in _CARTESIAN_HIGH_LEVEL_DISABLED_REWARDS:
        setattr(env_cfg.rewards, reward_name, None)

    # The high level can command destabilizing base motion, so retain compact
    # state-level stability/contact costs without teaching a second foot gait.
    env_cfg.rewards.lin_vel_z_l2.weight = -0.5
    env_cfg.rewards.ang_vel_xy_l2.weight = -0.05
    # Positive posture/living rewards outweighed all task progress in the
    # failed run. The frozen WBC already tracks these commands; retain only
    # zero-centered physical stability costs at this level.
    env_cfg.rewards.upward = None
    env_cfg.rewards.track_base_roll_pitch_exp = None
    env_cfg.rewards.track_base_height_command = None
    # This term created a discontinuous escape incentive: once unavoidable
    # drift accrued, leaving manipulation reach removed a large persistent cost.
    env_cfg.rewards.base_xy_drift_in_arm_reach = None

    object_cfg = copy.deepcopy(env_cfg.rewards.ee_to_object.params["object_cfg"])
    ee_cfg = copy.deepcopy(env_cfg.rewards.ee_to_object.params["ee_cfg"])
    gripper_close_params = copy.deepcopy(env_cfg.rewards.gripper_close_near_object.params)
    gripper_contact_params = copy.deepcopy(env_cfg.rewards.gripper_object_contact.params)
    env_cfg.rewards.base_approach_outside_arm_reach = RewTerm(
        func=mdp.base_object_approach_progress,
        weight=3.0,
        params={
            "progress_scale": 0.04,
            "object_cfg": object_cfg,
        },
    )
    env_cfg.rewards.base_velocity_towards_standoff = RewTerm(
        func=mdp.base_velocity_towards_object_standoff,
        weight=1.0,
        params={
            "standoff_distance": float(pick_cfg.standoff_distance),
            "speed_scale": 0.50,
            "distance_scale": 0.25,
            "object_cfg": object_cfg,
        },
    )
    env_cfg.rewards.base_standoff_error = RewTerm(
        func=mdp.base_object_standoff_error_l2,
        weight=-0.35,
        params={
            "standoff_distance": float(pick_cfg.standoff_distance),
            "scale": 0.25,
            "object_cfg": object_cfg,
        },
    )
    env_cfg.rewards.base_facing_error = RewTerm(
        func=mdp.base_object_yaw_error_l2,
        weight=-0.20,
        params={
            "scale": 0.50,
            "object_cfg": object_cfg,
        },
    )
    # A renewable proximity reward let the policy hover just outside the next
    # transition forever. Reaching is now potential-like progress; alignment
    # after pregrasp is a zero-centered error rather than a positive living reward.
    env_cfg.rewards.ee_to_object = None
    env_cfg.rewards.ee_reach_progress = RewTerm(
        func=mdp.ee_object_reach_progress,
        weight=4.0,
        params={
            "progress_scale": 0.025,
            "ee_cfg": ee_cfg,
            "object_cfg": object_cfg,
        },
    )
    env_cfg.rewards.ee_reach_alignment_error = RewTerm(
        func=mdp.ee_object_alignment_excess_l2,
        weight=-0.75,
        params={
            "tolerance": 0.080,
            "scale": 0.30,
            "ee_cfg": ee_cfg,
            "object_cfg": object_cfg,
        },
    )
    env_cfg.rewards.ee_grasp_alignment_error = RewTerm(
        func=mdp.ee_object_alignment_excess_l2,
        weight=-4.0,
        params={
            "tolerance": 0.055,
            "scale": 0.10,
            "ee_cfg": ee_cfg,
            "object_cfg": object_cfg,
        },
    )
    env_cfg.rewards.grasp_frame_horizontal_error = RewTerm(
        func=mdp.grasp_frame_horizontal_error,
        weight=-0.25,
        params={
            "wrist_cfg": SceneEntityCfg("robot", body_names="Link6"),
            "finger_cfg": ee_cfg,
        },
    )
    env_cfg.rewards.gripper_closed_before_pregrasp = RewTerm(
        func=mdp.gripper_closed_fraction,
        weight=-0.5,
        params={
            "gripper_cfg": gripper_close_params["gripper_cfg"],
            "open_joint_pos": gripper_close_params["open_joint_pos"],
            "closed_joint_pos": gripper_close_params["closed_joint_pos"],
        },
    )
    env_cfg.rewards.gripper_close_near_object = RewTerm(
        func=mdp.gripper_close_progress_near_object,
        weight=2.0,
        params={
            "ee_cfg": ee_cfg,
            "gripper_cfg": gripper_close_params["gripper_cfg"],
            "object_cfg": object_cfg,
            "near_std": gripper_close_params["near_std"],
            "progress_scale": 0.25,
            "open_joint_pos": gripper_close_params["open_joint_pos"],
            "closed_joint_pos": gripper_close_params["closed_joint_pos"],
        },
    )
    env_cfg.rewards.gripper_object_contact = RewTerm(
        func=mdp.gripper_object_contact_progress,
        weight=5.0,
        params=gripper_contact_params,
    )
    # Finishing promptly must dominate waiting at a safe configuration, while
    # deliberately knocking the object out of bounds must not become the
    # cheapest way to avoid the time cost.
    env_cfg.rewards.pick_time_penalty = RewTerm(func=mdp.is_alive, weight=-0.05)
    env_cfg.rewards.pick_failure_termination = RewTerm(
        func=mdp.is_terminated_term,
        weight=-25.0,
        params={"term_keys": "object_out_of_bounds"},
    )

    # Base velocity/posture (0:5) and grasp pose/gripper commands (5:10) have
    # different physical scales. Penalize each exactly once instead of stacking
    # a global action penalty with another manipulation-only penalty.
    env_cfg.rewards.base_command_action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2_selected,
        weight=-0.005,
        params={"action_indices": (0, 1, 2, 3, 4)},
    )
    env_cfg.rewards.base_command_action_smoothness_l2 = RewTerm(
        func=mdp.action_smoothness_l2_selected,
        weight=-0.0025,
        params={"action_indices": (0, 1, 2, 3, 4)},
    )
    env_cfg.rewards.base_command_action_l2 = RewTerm(
        func=mdp.action_l2_selected,
        weight=-0.001,
        params={"action_indices": (0, 1, 2, 3, 4)},
    )
    env_cfg.rewards.manipulation_command_action_rate_l2 = RewTerm(
        func=mdp.action_rate_l2_selected,
        weight=-0.02,
        params={"action_indices": (5, 6, 7, 8, 9)},
    )
    env_cfg.rewards.manipulation_command_action_smoothness_l2 = RewTerm(
        func=mdp.action_smoothness_l2_selected,
        weight=-0.01,
        params={"action_indices": (5, 6, 7, 8, 9)},
    )
    env_cfg.rewards.manipulation_command_action_l2 = RewTerm(
        func=mdp.action_l2_selected,
        weight=-0.002,
        params={"action_indices": (5, 6, 7, 8, 9)},
    )

    # Unlike the frozen legs, the D1 is executed from this policy's Cartesian
    # command through IK. Its measured limits, velocity, acceleration, and torque
    # therefore remain legitimate high-level manipulation safety costs.
    arm_cfg = SceneEntityCfg(
        "robot",
        joint_names=env_cfg.arm_joint_names,
        preserve_order=True,
    )
    env_cfg.rewards.arm_joint_pos_limits = RewTerm(
        func=mdp.joint_pos_limits,
        weight=-5.0,
        params={"asset_cfg": arm_cfg},
    )
    env_cfg.rewards.arm_joint_velocity_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-5.0e-4,
        params={"asset_cfg": arm_cfg},
    )
    env_cfg.rewards.arm_joint_acceleration_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-2.5e-7,
        params={"asset_cfg": arm_cfg},
    )
    env_cfg.rewards.arm_joint_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-1.0e-5,
        params={"asset_cfg": arm_cfg},
    )


def _configure_cartesian_pick_stages(env_cfg, pick_cfg) -> None:
    """Gate task rewards with a recoverable Approach/Reach/Grasp-Lift/Hold graph."""
    # The renewable ee_to_object reward is intentionally disabled for this
    # task, so take the shared virtual-grasp entities from the source pick cfg.
    ee_cfg = copy.deepcopy(pick_cfg.rewards.ee_to_object.params["ee_cfg"])
    object_cfg = copy.deepcopy(pick_cfg.rewards.ee_to_object.params["object_cfg"])
    contact_params = env_cfg.rewards.gripper_object_contact.params
    stage_params = {
        "manipulation_reach": float(pick_cfg.manipulation_reach),
        "pregrasp_distance": 0.10,
        "min_height": float(pick_cfg.object_init_pos[2]),
        "lift_margin": float(pick_cfg.target_lift_height - pick_cfg.object_init_pos[2]),
        "lift_enter_fraction": 0.65,
        "reach_exit_margin": 0.08,
        "pregrasp_exit_margin": 0.05,
        "lift_exit_fraction": 0.25,
        "ee_cfg": ee_cfg,
        "object_cfg": object_cfg,
        "sensor_cfg": copy.deepcopy(contact_params["sensor_cfg"]),
        "right_sensor_cfg": copy.deepcopy(contact_params["right_sensor_cfg"]),
        "force_threshold": float(contact_params["force_threshold"]),
        "required_contacts": int(contact_params["required_contacts"]),
    }

    # Lift, hold, and invalid-lift terms must agree with the bilateral contact
    # semantics used by the stage graph and terminal success.
    for reward_name in (
        "object_lifted_with_gripper_contact",
        "object_hold_lifted",
        "object_lifted_without_gripper_contact",
    ):
        getattr(env_cfg.rewards, reward_name).params["required_contacts"] = 2

    for reward_name, active_stages in _CARTESIAN_PICK_REWARD_STAGES.items():
        reward = getattr(env_cfg.rewards, reward_name, None)
        if reward is None:
            raise ValueError(f"Cartesian pick stage reward {reward_name!r} is missing.")
        reward.params["active_pick_stages"] = active_stages
        reward.params["pick_stage_params"] = copy.deepcopy(stage_params)

    env_cfg.observations.policy.pick_stage = ObsTerm(
        func=mdp.pick_stage_one_hot,
        params=copy.deepcopy(stage_params),
        clip=(0.0, 1.0),
        scale=1.0,
    )
    env_cfg.observations.critic.pick_stage = copy.deepcopy(env_cfg.observations.policy.pick_stage)

    gripper_reward_params = env_cfg.rewards.gripper_close_near_object.params
    env_cfg.observations.policy.gripper_close_fraction = ObsTerm(
        func=mdp.gripper_close_fraction,
        params={
            "gripper_cfg": copy.deepcopy(gripper_reward_params["gripper_cfg"]),
            "open_joint_pos": gripper_reward_params["open_joint_pos"],
            "closed_joint_pos": gripper_reward_params["closed_joint_pos"],
        },
        clip=(0.0, 1.0),
        scale=1.0,
    )
    env_cfg.observations.policy.left_gripper_object_contact = ObsTerm(
        func=mdp.gripper_object_contact_state,
        params={
            "sensor_cfg": copy.deepcopy(stage_params["sensor_cfg"]),
            "force_threshold": stage_params["force_threshold"],
            "required_contacts": 1,
        },
        clip=(0.0, 1.0),
        scale=1.0,
    )
    env_cfg.observations.policy.right_gripper_object_contact = ObsTerm(
        func=mdp.gripper_object_contact_state,
        params={
            "sensor_cfg": copy.deepcopy(stage_params["right_sensor_cfg"]),
            "force_threshold": stage_params["force_threshold"],
            "required_contacts": 1,
        },
        clip=(0.0, 1.0),
        scale=1.0,
    )
    env_cfg.observations.policy.object_lift_progress = ObsTerm(
        func=mdp.object_lift_progress,
        params={
            "object_cfg": copy.deepcopy(object_cfg),
            "min_height": stage_params["min_height"],
            "lift_margin": stage_params["lift_margin"],
        },
        clip=(0.0, 1.0),
        scale=1.0,
    )
    env_cfg.observations.policy.wrist_orientation = ObsTerm(
        func=mdp.body_orientation_6d_b,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="Link6"),
        },
        clip=(-1.0, 1.0),
        scale=1.0,
    )
    for observation_name in (
        "gripper_close_fraction",
        "left_gripper_object_contact",
        "right_gripper_object_contact",
        "object_lift_progress",
        "wrist_orientation",
    ):
        setattr(
            env_cfg.observations.critic,
            observation_name,
            copy.deepcopy(getattr(env_cfg.observations.policy, observation_name)),
        )

    for target_stage, weight in ((1, 2.0), (2, 5.0), (3, 10.0)):
        setattr(
            env_cfg.rewards,
            f"pick_stage_{target_stage}_transition",
            RewTerm(
                func=mdp.pick_stage_transition_bonus,
                weight=weight,
                params={
                    "target_stage": target_stage,
                    "pick_stage_params": copy.deepcopy(stage_params),
                },
            ),
        )
    env_cfg.rewards.pick_stage_regression = RewTerm(
        func=mdp.pick_stage_regression_penalty,
        weight=-10.0,
        params={"pick_stage_params": copy.deepcopy(stage_params)},
    )

    success_params = {
        "hold_steps": 25,
        "max_object_speed": 0.25,
        "pick_stage_params": copy.deepcopy(stage_params),
    }
    env_cfg.rewards.pick_success_bonus = RewTerm(
        func=mdp.pick_success_bonus,
        weight=25.0,
        params=copy.deepcopy(success_params),
    )
    env_cfg.terminations.pick_success = DoneTerm(
        func=mdp.pick_success,
        params=copy.deepcopy(success_params),
    )

    for stage_index, stage_name in enumerate(("approach", "reach", "grasp_lift", "hold")):
        setattr(
            env_cfg.curriculum,
            f"pick_stage_{stage_name}_fraction",
            CurrTerm(
                func=mdp.pick_stage_fraction,
                params={
                    "stage_index": stage_index,
                    "pick_stage_params": copy.deepcopy(stage_params),
                },
            ),
        )
    env_cfg.curriculum.base_action_saturation = CurrTerm(
        func=mdp.normalized_action_saturation_fraction,
        params={"action_indices": (0, 1, 2, 3, 4), "threshold": 0.98},
    )
    env_cfg.curriculum.manipulation_action_saturation = CurrTerm(
        func=mdp.normalized_action_saturation_fraction,
        params={"action_indices": (5, 6, 7, 8, 9), "threshold": 0.98},
    )
    env_cfg.curriculum.pick_success_fraction = CurrTerm(
        func=mdp.cumulative_pick_success_fraction,
    )
    env_cfg.curriculum.pick_bilateral_contact = CurrTerm(
        func=mdp.bilateral_gripper_contact_fraction,
        params={
            "sensor_cfg": copy.deepcopy(stage_params["sensor_cfg"]),
            "right_sensor_cfg": copy.deepcopy(stage_params["right_sensor_cfg"]),
            "force_threshold": stage_params["force_threshold"],
        },
    )
    env_cfg.curriculum.pick_geometry = CurrTerm(
        func=mdp.pick_geometry_diagnostics,
        params={
            "ee_cfg": copy.deepcopy(ee_cfg),
            "object_cfg": copy.deepcopy(object_cfg),
            "action_name": "wbc_command",
            "pregrasp_distance": stage_params["pregrasp_distance"],
        },
    )


def _configure_cartesian_pick_initial_state_curriculum(env_cfg, pick_cfg) -> None:
    """Learn the near manipulation primitive, then expand locomotion distance."""
    object_distance_end = float(pick_cfg.approach_distance_end)
    env_cfg.events.reset_object.params["object_distance_range"] = (
        float(pick_cfg.approach_distance_start),
        object_distance_end,
    )
    env_cfg.curriculum.go2_d1_pick_approach = CurrTerm(
        func=mdp.go2_d1_pick_approach_curriculum,
        params={
            "object_distance_start": float(pick_cfg.approach_distance_start),
            "object_distance_end": object_distance_end,
            "standoff_distance": float(pick_cfg.standoff_distance),
            "object_cfg": SceneEntityCfg("object"),
            "performance_based": True,
            "performance_threshold": 0.20,
            "performance_lower_threshold": 0.05,
            "performance_progress_step": 0.02,
            "performance_regress_step": 0.005,
            "min_steps_before_performance_update": 20_000,
            "performance_success_count_attr": "_loco_manip_pick_success_count",
            # The high-level policy, not a scripted curriculum controller,
            # chooses the base velocity command.
            "inject_velocity_commands": False,
        },
    )
    env_cfg.curriculum.recent_pick_success = CurrTerm(
        func=mdp.env_attr_curriculum_metric,
        params={"attr_name": "_go2_d1_pick_approach_score"},
    )
    env_cfg.curriculum.pick_disturbance = CurrTerm(
        func=mdp.pick_disturbance_curriculum,
        params={
            "performance_threshold": 0.20,
            "performance_lower_threshold": 0.10,
            "progress_step": 0.02,
            "regress_step": 0.01,
            "reset_velocity_start": 0.05,
            "reset_velocity_end": 0.50,
            "push_velocity_end": 0.50,
        },
    )


def _configure_external_high_level_velocity_command(env_cfg) -> None:
    """Make the command term a passive buffer owned by the high-level action."""
    command_cfg = env_cfg.commands.base_velocity
    # The inherited locomotion command generator otherwise resamples an
    # unrelated command every 4--6 s and overwrites 20% of environments with
    # standing commands. That corrupts both actor history and playback arrows.
    command_cfg.resampling_time_range = (1.0e6, 1.0e6)
    command_cfg.rel_standing_envs = 0.0
    command_cfg.rel_heading_envs = 0.0
    command_cfg.ranges.lin_vel_x = (0.0, 0.0)
    command_cfg.ranges.lin_vel_y = (0.0, 0.0)
    command_cfg.ranges.ang_vel_z = (0.0, 0.0)
    command_cfg.roll_range = (0.0, 0.0)
    command_cfg.pitch_range = (0.0, 0.0)
    command_cfg.height_range = (float(command_cfg.nominal_height), float(command_cfg.nominal_height))
    command_cfg.zero_posture_probability = 1.0


def _configure_cartesian_pick_ready_reset(env_cfg) -> None:
    """Reset directly in the collision-safe workspace posture used by Stage 2."""
    if env_cfg.events.reset_arm_joint_state is not None:
        position_ranges = env_cfg.events.reset_arm_joint_state.params["position_ranges"]
        position_ranges.update(
            {
                joint_name: (joint_position, joint_position)
                for joint_name, joint_position in zip(
                    env_cfg.arm_joint_names,
                    GO2_D1_WBC_WORKSPACE_READY_POSE,
                )
            }
        )
        position_ranges.update(
            {"arm_7_1_joint": (0.0, 0.0), "arm_7_2_joint": (0.0, 0.0)}
        )
    if env_cfg.actions.wbc_command is not None:
        env_cfg.actions.wbc_command.deployment_ee_waypoints = ()


@configclass
class UnitreeGo2D1PickWbcHierarchicalFlatEnvCfg(UnitreeGo2D1WholeBodyControllerFlatEnvCfg):
    """Pick a tabletop can by commanding a frozen Go2+D1 WBC policy."""

    wbc_policy_path: str = ""

    def __post_init__(self):
        pick_cfg = UnitreeGo2D1PickFlatEnvCfg()

        super().__post_init__()

        self.episode_length_s = pick_cfg.episode_length_s
        self.scene.env_spacing = pick_cfg.scene.env_spacing
        self.scene.replicate_physics = pick_cfg.scene.replicate_physics
        self.scene.pick_table = pick_cfg.scene.pick_table
        self.scene.object = pick_cfg.scene.object
        self.scene.left_gripper_object_contact_forces = copy.deepcopy(
            pick_cfg.scene.left_gripper_object_contact_forces
        )
        self.scene.right_gripper_object_contact_forces = copy.deepcopy(
            pick_cfg.scene.right_gripper_object_contact_forces
        )
        _configure_collision_safety(self, support_filter_paths=("{ENV_REGEX_NS}/PickTable",))
        if self.events.reset_arm_joint_state is not None:
            self.events.reset_arm_joint_state.params["position_ranges"].update(
                {"arm_7_1_joint": (0.0, 0.0), "arm_7_2_joint": (0.0, 0.0)}
            )

        self.events.reset_object = pick_cfg.events.reset_object
        self.events.randomize_object_scale = pick_cfg.events.randomize_object_scale
        self.events.randomize_object_material = pick_cfg.events.randomize_object_material
        self.events.randomize_object_mass = pick_cfg.events.randomize_object_mass
        self.events.randomize_reset_base = pick_cfg.events.randomize_reset_base

        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None
        self.curriculum.go2_d1_pick_approach = pick_cfg.curriculum.go2_d1_pick_approach
        if self.curriculum.go2_d1_pick_approach is not None:
            self.curriculum.go2_d1_pick_approach.params["inject_velocity_commands"] = False
        self.curriculum.loco_manipulation_training_stages = None

        self.terminations.object_out_of_bounds = pick_cfg.terminations.object_out_of_bounds

        for reward_name in (
            "base_to_object_standoff",
            "base_faces_object",
            "ee_to_object",
            "base_approach_outside_arm_reach",
            "gripper_close_near_object",
            "gripper_object_contact",
            "object_lifted",
            "object_lifted_near_ee",
            "object_lifted_with_gripper_contact",
            "object_hold_lifted",
            "object_lifted_without_gripper_contact",
            "object_xy_motion_before_lift",
            "object_vertical",
            "object_still_xy",
            "arm_ready_before_standoff",
            "base_xy_velocity_near_standoff_l2",
            "base_xy_velocity_l2",
            "base_xy_drift_in_arm_reach",
        ):
            reward = getattr(pick_cfg.rewards, reward_name, None)
            if reward is not None:
                setattr(self.rewards, reward_name, copy.deepcopy(reward))

        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0

        configure_go2_d1_wbc_hierarchical_runtime(self)
        _configure_smooth_high_level_actions(self, tuple(range(3, 9)))
        if self.actions.wbc_command is not None:
            self.actions.wbc_command.policy_path = self.wbc_policy_path

        _configure_hierarchical_pick_policy_obs(self, pick_cfg.observations.policy)
        self.disable_zero_weight_rewards()


@configclass
class UnitreeGo2D1PickLegWbcArmHierarchicalFlatEnvCfg(UnitreeGo2D1LegWbcAsyncArmFlatEnvCfg):
    """Pick a tabletop can by commanding Go2 body targets and D1 arm joints over a frozen leg WBC."""

    wbc_policy_path: str = ""

    def __post_init__(self):
        pick_cfg = UnitreeGo2D1PickFlatEnvCfg()

        super().__post_init__()

        self.episode_length_s = pick_cfg.episode_length_s
        self.scene.env_spacing = pick_cfg.scene.env_spacing
        self.scene.replicate_physics = pick_cfg.scene.replicate_physics
        self.scene.pick_table = pick_cfg.scene.pick_table
        self.scene.object = pick_cfg.scene.object
        self.scene.left_gripper_object_contact_forces = copy.deepcopy(
            pick_cfg.scene.left_gripper_object_contact_forces
        )
        self.scene.right_gripper_object_contact_forces = copy.deepcopy(
            pick_cfg.scene.right_gripper_object_contact_forces
        )
        _configure_collision_safety(self, support_filter_paths=("{ENV_REGEX_NS}/PickTable",))
        if self.events.reset_arm_joint_state is not None:
            self.events.reset_arm_joint_state.params["position_ranges"].update(
                {"arm_7_1_joint": (0.0, 0.0), "arm_7_2_joint": (0.0, 0.0)}
            )

        self.events.reset_object = pick_cfg.events.reset_object
        self.events.randomize_object_scale = pick_cfg.events.randomize_object_scale
        self.events.randomize_object_material = pick_cfg.events.randomize_object_material
        self.events.randomize_object_mass = pick_cfg.events.randomize_object_mass
        self.events.randomize_reset_base = pick_cfg.events.randomize_reset_base

        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None
        self.curriculum.go2_d1_pick_approach = pick_cfg.curriculum.go2_d1_pick_approach
        if self.curriculum.go2_d1_pick_approach is not None:
            self.curriculum.go2_d1_pick_approach.params["inject_velocity_commands"] = False
        self.curriculum.loco_manipulation_training_stages = None

        self.terminations.object_out_of_bounds = pick_cfg.terminations.object_out_of_bounds

        for reward_name in (
            "base_to_object_standoff",
            "base_faces_object",
            "ee_to_object",
            "base_approach_outside_arm_reach",
            "gripper_close_near_object",
            "gripper_object_contact",
            "object_lifted",
            "object_lifted_near_ee",
            "object_lifted_with_gripper_contact",
            "object_hold_lifted",
            "object_lifted_without_gripper_contact",
            "object_xy_motion_before_lift",
            "object_vertical",
            "object_still_xy",
            "arm_ready_before_standoff",
            "base_xy_velocity_near_standoff_l2",
            "base_xy_velocity_l2",
            "base_xy_drift_in_arm_reach",
        ):
            reward = getattr(pick_cfg.rewards, reward_name, None)
            if reward is not None:
                setattr(self.rewards, reward_name, copy.deepcopy(reward))

        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0

        configure_go2_d1_leg_wbc_arm_hierarchical_runtime(self)
        _configure_smooth_high_level_actions(self, tuple(range(3, 12)))
        if self.actions.wbc_command is not None:
            self.actions.wbc_command.policy_path = self.wbc_policy_path

        _configure_hierarchical_pick_policy_obs(self, pick_cfg.observations.policy)
        self.disable_zero_weight_rewards()


@configclass
class UnitreeGo2D1PickLegWbcEeHierarchicalFlatEnvCfg(UnitreeGo2D1LegWbcAsyncArmFlatEnvCfg):
    """Privileged high-level teacher controlling Go2 posture and the D1 grasp pose."""

    wbc_policy_path: str = ""

    def __post_init__(self):
        pick_cfg = _UnitreeGo2D1HighLevelPickSourceCfg()

        super().__post_init__()

        self.episode_length_s = pick_cfg.episode_length_s
        self.scene.env_spacing = pick_cfg.scene.env_spacing
        self.scene.replicate_physics = pick_cfg.scene.replicate_physics
        self.scene.pick_table = pick_cfg.scene.pick_table
        self.scene.object = pick_cfg.scene.object
        self.scene.left_gripper_object_contact_forces = copy.deepcopy(
            pick_cfg.scene.left_gripper_object_contact_forces
        )
        self.scene.right_gripper_object_contact_forces = copy.deepcopy(
            pick_cfg.scene.right_gripper_object_contact_forces
        )
        _configure_collision_safety(self, support_filter_paths=("{ENV_REGEX_NS}/PickTable",))
        if self.events.reset_arm_joint_state is not None:
            self.events.reset_arm_joint_state.params["position_ranges"].update(
                {"arm_7_1_joint": (0.0, 0.0), "arm_7_2_joint": (0.0, 0.0)}
            )

        self.events.reset_object = pick_cfg.events.reset_object
        self.events.randomize_object_scale = pick_cfg.events.randomize_object_scale
        self.events.randomize_object_material = pick_cfg.events.randomize_object_material
        self.events.randomize_object_mass = pick_cfg.events.randomize_object_mass
        self.events.randomize_reset_base = pick_cfg.events.randomize_reset_base
        for axis in tuple(self.events.randomize_reset_base.params["velocity_range"]):
            self.events.randomize_reset_base.params["velocity_range"][axis] = (-0.05, 0.05)
        if self.events.randomize_push_robot is not None:
            for axis in tuple(self.events.randomize_push_robot.params["velocity_range"]):
                self.events.randomize_push_robot.params["velocity_range"][axis] = (0.0, 0.0)

        self.curriculum.command_levels_lin_vel = None
        self.curriculum.command_levels_ang_vel = None
        self.curriculum.go2_d1_pick_approach = pick_cfg.curriculum.go2_d1_pick_approach
        if self.curriculum.go2_d1_pick_approach is not None:
            self.curriculum.go2_d1_pick_approach.params["inject_velocity_commands"] = False
        self.curriculum.loco_manipulation_training_stages = None

        self.terminations.object_out_of_bounds = pick_cfg.terminations.object_out_of_bounds

        for reward_name in (
            "base_to_object_standoff",
            "base_faces_object",
            "ee_to_object",
            "base_approach_outside_arm_reach",
            "gripper_close_near_object",
            "gripper_object_contact",
            "object_lifted",
            "object_lifted_near_ee",
            "object_lifted_with_gripper_contact",
            "object_hold_lifted",
            "object_lifted_without_gripper_contact",
            "object_xy_motion_before_lift",
            "object_vertical",
            "object_still_xy",
            "arm_ready_before_standoff",
            "base_xy_velocity_near_standoff_l2",
            "base_xy_velocity_l2",
            "base_xy_drift_in_arm_reach",
        ):
            reward = getattr(pick_cfg.rewards, reward_name, None)
            if reward is not None:
                setattr(self.rewards, reward_name, copy.deepcopy(reward))

        self.rewards.track_lin_vel_xy_exp.weight = 0.0
        self.rewards.track_ang_vel_z_exp.weight = 0.0

        configure_go2_d1_leg_wbc_ee_hierarchical_runtime(self)
        _configure_external_high_level_velocity_command(self)
        _configure_cartesian_pick_ready_reset(self)
        _configure_cartesian_high_level_rewards(self, pick_cfg)
        if self.actions.wbc_command is not None:
            self.actions.wbc_command.policy_path = self.wbc_policy_path

        _configure_hierarchical_pick_policy_obs(self, pick_cfg.observations.policy)
        _configure_cartesian_pick_stages(self, pick_cfg)
        _configure_cartesian_pick_initial_state_curriculum(self, pick_cfg)
        _configure_cartesian_teacher_observations(self, pick_cfg)
        self.disable_zero_weight_rewards()


@configclass
class UnitreeGo2D1PickLegWbcEeHierarchicalFastFlatEnvCfg(
    UnitreeGo2D1PickLegWbcEeHierarchicalFlatEnvCfg
):
    """Throughput-oriented Cartesian pick teacher with replicated physics."""

    def __post_init__(self):
        super().__post_init__()

        # Per-environment USD scale edits are the only object randomization in
        # this task that requires independent physics parsing. Keep the nominal
        # 56 mm cylinder during fast training; pose, velocity, material, and
        # mass randomization remain active through tensor/PhysX APIs.
        self.events.randomize_object_scale = None
        self.scene.replicate_physics = True
        _configure_compact_collision_safety(self)
