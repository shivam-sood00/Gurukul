from __future__ import annotations

import ast
import math
import re
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
VELOCITY_ROOT = REPO_ROOT / "source/Gurukul/Gurukul/tasks/manager_based/locomotion/velocity"
WBC_CFG_PATH = VELOCITY_ROOT / "config/quadruped_with_arm/unitree_go2_d1_arm/whole_body_controller_env_cfg.py"
FLAT_CFG_PATH = VELOCITY_ROOT / "config/quadruped_with_arm/unitree_go2_d1_arm/flat_env_cfg.py"
ROUGH_CFG_PATH = VELOCITY_ROOT / "config/quadruped_with_arm/unitree_go2_d1_arm/rough_env_cfg.py"
ARM_MOTION_CFG_PATH = VELOCITY_ROOT / "config/quadruped_with_arm/unitree_go2_d1_arm/arm_motion_env_cfg.py"
PICK_CFG_PATH = VELOCITY_ROOT / "config/quadruped_with_arm/unitree_go2_d1_arm/loco_manipulation_env_cfg.py"
HIERARCHICAL_CFG_PATH = VELOCITY_ROOT / "config/quadruped_with_arm/unitree_go2_d1_arm/wbc_hierarchical_env_cfg.py"
ACTIONS_PATH = VELOCITY_ROOT / "mdp/actions.py"
EVENTS_PATH = VELOCITY_ROOT / "mdp/events.py"
REWARDS_PATH = VELOCITY_ROOT / "mdp/rewards.py"
CURRICULUMS_PATH = VELOCITY_ROOT / "mdp/curriculums.py"
RSL_RL_AGENT_CFG_PATH = VELOCITY_ROOT / "config/quadruped_with_arm/unitree_go2_d1_arm/agents/rsl_rl_ppo_cfg.py"
CUSRL_AGENT_CFG_PATH = VELOCITY_ROOT / "config/quadruped_with_arm/unitree_go2_d1_arm/agents/cusrl_ppo_cfg.py"
GO2_D1_REGISTRY_PATH = VELOCITY_ROOT / "config/quadruped_with_arm/unitree_go2_d1_arm/__init__.py"
RSL_RL_COMPAT_PATH = REPO_ROOT / "scripts/reinforcement_learning/rsl_rl/rsl_rl_config_compat.py"
ISAAC_WORKSPACE_VIS_PATH = REPO_ROOT / "scripts/tools/visualize_go2_d1_wbc_workspace.py"
ISAAC_WORKSPACE_EVAL_PATH = REPO_ROOT / "scripts/tools/evaluate_go2_d1_wbc_arm_commands.py"
ISAAC_CURRICULUM_PREVIEW_PATH = REPO_ROOT / "scripts/tools/preview_go2_d1_wbc_arm_curriculum.py"
RSL_RL_PLAY_PATH = REPO_ROOT / "scripts/reinforcement_learning/rsl_rl/play.py"
RSL_RL_TRAIN_PATH = REPO_ROOT / "scripts/reinforcement_learning/rsl_rl/train.py"


def _literal_assignment(path: Path, name: str):
    tree = ast.parse(path.read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(
            isinstance(target, ast.Name) and target.id == name for target in node.targets
        ):
            return ast.literal_eval(node.value)
    raise AssertionError(f"Assignment {name!r} not found in {path}")


def _load_functions(path: Path, *names: str):
    tree = ast.parse(path.read_text())
    functions = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    assert len(functions) == len(names)
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), *functions],
        type_ignores=[],
    )
    namespace = {
        "math": math,
        "torch": torch,
        "SceneEntityCfg": lambda name, **kwargs: SimpleNamespace(name=name, joint_ids=slice(None), **kwargs),
    }
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), namespace)
    return tuple(namespace[name] for name in names)


def _load_function(path: Path, name: str):
    return _load_functions(path, name)[0]


def test_events_imports_math_for_stratified_workspace_sampling():
    tree = ast.parse(EVENTS_PATH.read_text())
    imported_names = {alias.name for node in tree.body if isinstance(node, ast.Import) for alias in node.names}

    assert "math" in imported_names


def test_all_non_apex_go2_d1_tasks_inherit_calibrated_dynamics_and_deployable_feedback():
    rough_source = ROUGH_CFG_PATH.read_text()
    arm_motion_source = ARM_MOTION_CFG_PATH.read_text()
    wbc_source = WBC_CFG_PATH.read_text()
    pick_source = PICK_CFG_PATH.read_text()
    reward_source = REWARDS_PATH.read_text()

    assert "UNITREE_GO2_D1_ARM_APEX_CFG.replace" in rough_source
    assert 'self.scene.robot.actuators["legs"].stiffness = 20.0' in rough_source
    assert 'self.scene.robot.actuators["legs"].damping = 0.5' in rough_source
    assert '"stiffness_distribution_params"] = (0.9, 1.1)' in rough_source
    assert '"damping_distribution_params"] = (0.9, 1.1)' in rough_source
    assert '"position_ranges": {"base_link_joint": (-0.005, 0.005)}' in rough_source
    assert "D1_ARM_HARDWARE_VELOCITY_LIMITS = (1.05, 1.05, 1.05, 1.73, 1.73, 1.73)" in rough_source
    assert "func=mdp.joint_velocity_soft_limits_l2" in rough_source
    assert "def joint_velocity_soft_limits_l2(" in reward_source
    assert "_D1_NONADJACENT_SELF_COLLISION_FILTERS = (" in rough_source

    actor_velocity_contract = 'observations.policy.joint_vel.params["asset_cfg"].joint_names = env_cfg.leg_joint_names'
    assert actor_velocity_contract in arm_motion_source
    assert actor_velocity_contract in wbc_source
    assert 'self.observations.policy.joint_vel.params["asset_cfg"].joint_names = self.leg_joint_names' in rough_source
    assert 'self.observations.critic.joint_vel.params["asset_cfg"].joint_names = self.joint_names' in rough_source

    assert '"static_friction_range"] = (0.8, 1.0)' in rough_source
    assert '"dynamic_friction_range"] = (0.3, 0.8)' in rough_source
    assert pick_source.count('"static_friction_range": (0.90, 1.30)') == 2
    assert pick_source.count('"dynamic_friction_range": (0.60, 0.85)') == 2
    assert 'inherited_reward_name = f"d1_nonadjacent_contact_{source_body.lower()}"' in pick_source


def test_go2_d1_workspace_is_broad_and_inside_soft_hardware_limits():
    workspace = _literal_assignment(WBC_CFG_PATH, "GO2_D1_WBC_EE_POS_RANGE")
    reach_range = _literal_assignment(WBC_CFG_PATH, "GO2_D1_WBC_REACH_RANGE")
    ik_ranges = _literal_assignment(WBC_CFG_PATH, "GO2_D1_IK_ARM_JOINT_RANGES")

    assert workspace == ((0.10, 0.56), (-0.40, 0.40), (0.18, 0.65))
    assert workspace[1][1] - workspace[1][0] >= 0.8
    assert workspace[2][1] - workspace[2][0] >= 0.47
    assert reach_range == (0.12, 0.58)
    assert ik_ranges["arm_1_joint"] == (-2.12, 2.12)
    assert ik_ranges["arm_2_joint"] == (-1.41, 1.41)


def test_wbc_joint_clip_patterns_are_disjoint_and_cover_every_action_joint():
    clip = _literal_assignment(WBC_CFG_PATH, "GO2_D1_WBC_JOINT_CLIP")
    joints = [
        *(f"{leg}_{part}_joint" for leg in ("FL", "FR", "RL", "RR") for part in ("hip", "thigh", "calf")),
        *(f"arm_{index}_joint" for index in range(1, 7)),
        "arm_7_1_joint",
        "arm_7_2_joint",
    ]

    for joint in joints:
        matches = [pattern for pattern in clip if re.fullmatch(pattern, joint)]
        assert len(matches) == 1, f"{joint} matched {matches}"


def test_hierarchical_ee_commands_project_to_keepout_and_reach_shell():
    clamp_targets = _load_function(ACTIONS_PATH, "_clamp_ee_targets")
    workspace = ((0.10, 0.56), (-0.40, 0.40), (0.18, 0.65))
    keepout = ((-0.30, 0.34), (-0.20, 0.20), (-0.02, 0.30))
    origin = (0.0, 0.0, 0.08)
    reach_range = (0.12, 0.58)
    targets = torch.tensor(((0.1, 0.0, 0.1), (1.0, 1.0, 1.0)), dtype=torch.float32)

    projected = clamp_targets(targets, workspace, keepout, 0.07, (), origin, reach_range)
    distance = torch.linalg.vector_norm(projected - torch.tensor(origin), dim=1)

    assert torch.all(distance >= reach_range[0] - 1.0e-6)
    assert torch.all(distance <= reach_range[1] + 1.0e-6)
    expanded_keepout = tuple((low - 0.07, high + 0.07) for low, high in keepout)
    for point in projected:
        assert not all(low <= point[axis] <= high for axis, (low, high) in enumerate(expanded_keepout))
    for axis, (low, high) in enumerate(workspace):
        assert torch.all(projected[:, axis] >= low)
        assert torch.all(projected[:, axis] <= high)


def test_grasp_center_target_is_shifted_by_the_rotated_link_offset():
    link_target = _load_function(ACTIONS_PATH, "_link_position_from_grasp_target")
    grasp_target = torch.tensor(((1.0, 2.0, 3.0),), dtype=torch.float32)
    rotation_z_90 = torch.tensor(
        (((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),),
        dtype=torch.float32,
    )
    offset_link = torch.tensor(((0.2, 0.0, 0.1),), dtype=torch.float32)

    result = link_target(grasp_target, rotation_z_90, offset_link)

    assert torch.allclose(result, torch.tensor(((1.0, 1.8, 2.9),)), atol=1.0e-6)


def test_training_volume_predicate_rejects_shell_and_body_violations():
    is_feasible = _load_function(EVENTS_PATH, "_arm_ee_target_is_feasible")
    workspace = ((0.10, 0.56), (-0.40, 0.40), (0.18, 0.65))
    keepout = ((-0.30, 0.34), (-0.20, 0.20), (-0.02, 0.30))
    common_args = (workspace, keepout, 0.07, (), (0.0, 0.0, 0.08), (0.12, 0.58))

    assert is_feasible(torch.tensor((0.42, 0.0, 0.34)), *common_args)
    assert not is_feasible(torch.tensor((0.20, 0.0, 0.20)), *common_args)
    assert not is_feasible(torch.tensor((0.56, 0.40, 0.18)), *common_args)


def test_random_training_targets_are_rejection_sampled_inside_exact_volume():
    clamp_target, is_feasible, sample_target = _load_functions(
        EVENTS_PATH,
        "_clamp_arm_ee_target",
        "_arm_ee_target_is_feasible",
        "_sample_arm_ee_point",
    )
    del clamp_target
    workspace = ((0.10, 0.56), (-0.40, 0.40), (0.18, 0.65))
    keepout = ((-0.30, 0.34), (-0.20, 0.20), (-0.02, 0.30))
    common_args = (workspace, keepout, 0.07, (), (0.0, 0.0, 0.08), (0.12, 0.58))
    torch.manual_seed(7)
    points = torch.stack(
        [
            sample_target(
                torch.device("cpu"),
                workspace,
                keepout,
                0.07,
                workspace[0],
                workspace[1],
                workspace[2],
                (),
                (0.0, 0.0, 0.08),
                (0.12, 0.58),
            )
            for _ in range(256)
        ]
    )

    assert all(is_feasible(point, *common_args) for point in points)
    reach = torch.linalg.vector_norm(points - torch.tensor((0.0, 0.0, 0.08)), dim=1)
    assert not torch.any(torch.isclose(reach, torch.tensor(0.58), atol=1.0e-5))


def test_position_ik_supplies_isaaclab_display_quaternion():
    source = EVENTS_PATH.read_text()
    assert "ik_controller.set_command(env._arm_ee_target_pos, ee_quat=env._arm_ee_target_quat)" in source


def test_world_jacobian_is_rotated_into_root_frame_without_mutating_input():
    rotate_jacobian = _load_function(EVENTS_PATH, "_rotate_jacobian_to_root_frame")
    root_rotation_b_to_w = torch.tensor(
        (((0.0, -1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),),
        dtype=torch.float32,
    )
    root_rotation_w_to_b = root_rotation_b_to_w.transpose(1, 2)
    jacobian_b_expected = torch.tensor(
        (
            (
                (1.0, 0.0),
                (0.0, 1.0),
                (0.0, 0.0),
                (1.0, 0.0),
                (0.0, 1.0),
                (0.0, 0.0),
            ),
        ),
        dtype=torch.float32,
    )
    jacobian_w = jacobian_b_expected.clone()
    jacobian_w[:, 0:3, :] = torch.bmm(root_rotation_b_to_w, jacobian_b_expected[:, 0:3, :])
    jacobian_w[:, 3:6, :] = torch.bmm(root_rotation_b_to_w, jacobian_b_expected[:, 3:6, :])
    original = jacobian_w.clone()

    jacobian_b = rotate_jacobian(jacobian_w, root_rotation_w_to_b)

    assert torch.allclose(jacobian_b, jacobian_b_expected, atol=1.0e-6)
    assert torch.equal(jacobian_w, original)
    tracking_source = ast.get_source_segment(
        EVENTS_PATH.read_text(),
        next(
            node
            for node in ast.parse(EVENTS_PATH.read_text()).body
            if isinstance(node, ast.FunctionDef) and node.name == "continuous_arm_ik_tracking"
        ),
    )
    assert tracking_source is not None
    assert "jacobian = _rotate_jacobian_to_root_frame(jacobian_w, root_rotation_w_to_b)" in tracking_source


def test_wbc_starts_compact_and_deploys_out_of_the_body_keepout():
    carry_pose = _literal_assignment(WBC_CFG_PATH, "GO2_D1_WBC_CARRY_POSE")
    deployment = _literal_assignment(WBC_CFG_PATH, "GO2_D1_WBC_DEPLOYMENT_EE_WAYPOINTS")

    assert carry_pose == (0.0, -1.15, 1.35, 0.0, -0.30, 0.0)
    assert deployment[-1] == (0.4159, 0.0, 0.3381)

    expanded_keepout = ((-0.37, 0.41), (-0.27, 0.27), (-0.09, 0.37))
    carry_ee = torch.tensor((0.0398, 0.0, 0.3285))
    route = torch.tensor((carry_ee.tolist(), *deployment), dtype=torch.float32)

    def inside_keepout(points):
        mask = torch.ones(points.shape[0], dtype=torch.bool)
        for axis, (low, high) in enumerate(expanded_keepout):
            mask &= (points[:, axis] >= low) & (points[:, axis] <= high)
        return mask

    for segment_index in range(route.shape[0] - 1):
        t = torch.linspace(0.0, 1.0, 101).unsqueeze(1)
        segment = route[segment_index] * (1.0 - t) + route[segment_index + 1] * t
        inside = inside_keepout(segment)
        if segment_index == 0:
            # The gripper begins in the carry volume and exits upward without re-entering.
            inside_ids = torch.nonzero(inside, as_tuple=False).squeeze(1)
            assert inside_ids.numel() > 0
            assert torch.equal(inside_ids, torch.arange(inside_ids.numel()))
            assert not inside[-1]
        else:
            assert not torch.any(inside)

    config_source = WBC_CFG_PATH.read_text()
    assert '"arm_1_joint": GO2_D1_WBC_CARRY_POSE[0]' in config_source
    assert '"deployment_ee_waypoints": GO2_D1_WBC_DEPLOYMENT_EE_WAYPOINTS' in config_source
    assert 'params={**arm_command_params, "reset_deployment": True}' in config_source


def test_arm_deployment_waypoints_finish_before_random_workspace_sampling():
    next_waypoint = _load_function(EVENTS_PATH, "_arm_deployment_waypoint")
    waypoints = ((0.1, 0.0, 0.4), (0.4, 0.0, 0.4))

    assert torch.equal(next_waypoint(waypoints, 0, torch.device("cpu")), torch.tensor(waypoints[0]))
    assert torch.equal(next_waypoint(waypoints, 1, torch.device("cpu")), torch.tensor(waypoints[1]))
    assert next_waypoint(waypoints, 2, torch.device("cpu")) is None

    event_source = EVENTS_PATH.read_text()
    assert "env._arm_deployment_phase[disabled_env_ids] = 0" in event_source
    assert "if reset_deployment:" in event_source
    assert "if deployment_mask[i]:" in event_source


def test_frozen_wbc_smoothly_deploys_and_hands_off_the_first_high_level_goal():
    advance = _load_function(ACTIONS_PATH, "_advance_ee_deployment_targets")
    requested = torch.tensor(((0.50, 0.10, 0.40),), dtype=torch.float32)
    waypoints = torch.tensor(((0.10, 0.0, 0.40), (0.42, 0.0, 0.42)), dtype=torch.float32)
    current = torch.tensor(((0.04, 0.0, 0.33),), dtype=torch.float32)
    phase = torch.zeros(1, dtype=torch.long)
    progress = torch.zeros(1)
    start = torch.zeros((1, 3))
    initialized = torch.zeros(1, dtype=torch.bool)
    handoff_goal = torch.zeros((1, 3))
    handoff_captured = torch.zeros(1, dtype=torch.bool)
    outputs = []

    for _ in range(200):
        output, active = advance(
            requested,
            waypoints,
            current,
            phase,
            progress,
            start,
            initialized,
            handoff_goal,
            handoff_captured,
            0.05,
            0.20,
            0.25,
        )
        outputs.append(output.clone())
        current = output
        if not active.item():
            break

    assert phase.item() == waypoints.shape[0] + 1
    assert torch.allclose(outputs[-1], requested, atol=1.0e-6)
    assert torch.isfinite(torch.stack(outputs)).all()

    passthrough, active = advance(
        requested,
        torch.empty((0, 3)),
        current,
        phase,
        progress,
        start,
        initialized,
        handoff_goal,
        handoff_captured,
        0.05,
        0.20,
        0.25,
    )
    assert torch.equal(passthrough, requested)
    assert not active.item()


def test_workspace_evaluator_can_start_with_a_rotated_fixed_base():
    quat_from_rpy = _load_function(ISAAC_WORKSPACE_EVAL_PATH, "_quat_wxyz_from_rpy")

    identity = quat_from_rpy(0.0, 0.0, 0.0)
    yaw_90 = quat_from_rpy(0.0, 0.0, math.pi / 2.0)

    assert identity == (1.0, 0.0, 0.0, 0.0)
    assert torch.allclose(
        torch.tensor(yaw_90),
        torch.tensor((math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))),
        atol=1.0e-7,
    )
    evaluator_source = ISAAC_WORKSPACE_EVAL_PATH.read_text()
    assert '"--base-rpy"' in evaluator_source
    assert "env_cfg.scene.robot.init_state.rot = _quat_wxyz_from_rpy(*args_cli.base_rpy)" in evaluator_source
    assert 'for termination_name in ("time_out", "terrain_out_of_bounds"):' in evaluator_source
    assert "carry-to-workspace deployment segments before sampling" in evaluator_source


def test_per_env_curriculum_state_accepts_scalars_and_vectors():
    as_per_env = _load_function(EVENTS_PATH, "_as_per_env_tensor")

    scalar = as_per_env(0.25, 3, torch.device("cpu"), torch.float32, "difficulty")
    vector = as_per_env(torch.tensor((0.0, 0.5, 1.0)), 3, torch.device("cpu"), torch.float32, "difficulty")

    assert torch.equal(scalar, torch.tensor((0.25, 0.25, 0.25)))
    assert torch.equal(vector, torch.tensor((0.0, 0.5, 1.0)))
    try:
        as_per_env(torch.tensor((0.0, 1.0)), 3, torch.device("cpu"), torch.float32, "difficulty")
    except ValueError as exc:
        assert "either 1 or 3 values" in str(exc)
    else:
        raise AssertionError("A curriculum vector with the wrong environment count must be rejected.")


def test_async_arm_sampler_supports_mixed_per_env_curriculum_state():
    as_per_env, sample_async = _load_functions(EVENTS_PATH, "_as_per_env_tensor", "random_async_arm_joint_motion")
    del as_per_env

    class Articulation:
        def __init__(self):
            self.data = SimpleNamespace(joint_pos=torch.zeros((3, 2), dtype=torch.float32))
            self.target_calls = []

        def set_joint_position_target(self, target, *, joint_ids, env_ids):
            self.target_calls.append((target.clone(), joint_ids.clone(), env_ids.clone()))

    art = Articulation()

    class Scene:
        num_envs = 3

        def __getitem__(self, name):
            assert name == "robot"
            return art

    env = SimpleNamespace(
        device=torch.device("cpu"),
        scene=Scene(),
        _loco_manip_arm_motion_enabled=torch.tensor((False, True, True)),
        _loco_manip_arm_motion_difficulty=torch.tensor((0.0, 0.25, 1.0)),
    )
    asset_cfg = SimpleNamespace(name="robot", joint_ids=[0, 1], joint_names=["joint_a", "joint_b"])
    torch.manual_seed(17)

    sample_async(
        env,
        None,
        asset_cfg,
        joint_position_ranges={"joint_a": (-1.0, 1.0), "joint_b": (-1.0, 1.0)},
        nominal_joint_pos=(0.0, 0.0),
        safe_waypoints=(),
        max_joint_change_range=(0.10, 0.40),
        trajectory_duration_range=(0.50, 0.50),
        joint_velocity_limits=(0.10, 0.20),
        max_velocity_fraction=0.50,
        start_motion_enabled=False,
    )

    assert torch.equal(env._arm_async_joint_goal_pos[0], torch.zeros(2))
    assert torch.all(torch.abs(env._arm_async_joint_goal_pos[1]) <= 0.175 + 1.0e-6)
    assert torch.all(torch.abs(env._arm_async_joint_goal_pos[2]) <= 0.400 + 1.0e-6)
    assert torch.equal(env._arm_async_joint_initialized, torch.ones(3, dtype=torch.bool))
    profile_factor = torch.where(env._arm_async_joint_mode == 0, 1.0, 1.5)
    velocity_limits = torch.tensor((0.10, 0.20)).unsqueeze(0)
    minimum_duration = (
        profile_factor
        * torch.abs(env._arm_async_joint_goal_pos - env._arm_async_joint_start_pos)
        / (velocity_limits * 0.50)
    )
    assert torch.all(env._arm_async_joint_duration + 1.0e-6 >= minimum_duration)
    assert torch.allclose(env._arm_async_joint_velocity_cap_fraction[1:], torch.full((2, 2), 0.50))
    assert torch.all(
        env._arm_async_joint_peak_velocity_fraction <= env._arm_async_joint_velocity_cap_fraction + 1.0e-6
    )
    assert len(art.target_calls) == 1
    assert torch.equal(art.target_calls[0][2], torch.tensor((0,)))


def test_async_arm_sampler_includes_varied_and_near_limit_joint_speeds():
    _, sample_async = _load_functions(EVENTS_PATH, "_as_per_env_tensor", "random_async_arm_joint_motion")

    class Articulation:
        def __init__(self):
            self.data = SimpleNamespace(joint_pos=torch.zeros((64, 2), dtype=torch.float32))

        def set_joint_position_target(self, target, *, joint_ids, env_ids):
            del target, joint_ids, env_ids

    art = Articulation()

    class Scene:
        num_envs = 64

        def __getitem__(self, name):
            assert name == "robot"
            return art

    env = SimpleNamespace(
        device=torch.device("cpu"),
        scene=Scene(),
        _loco_manip_arm_motion_enabled=torch.ones(64, dtype=torch.bool),
        _loco_manip_arm_motion_difficulty=torch.ones(64),
    )
    asset_cfg = SimpleNamespace(name="robot", joint_ids=[0, 1], joint_names=["joint_a", "joint_b"])
    torch.manual_seed(23)

    sample_async(
        env,
        None,
        asset_cfg,
        joint_position_ranges={"joint_a": (-1.0, 1.0), "joint_b": (-1.0, 1.0)},
        nominal_joint_pos=(0.0, 0.0),
        safe_waypoints=(),
        max_joint_change_range=(1.0, 1.0),
        trajectory_duration_range=(0.35, 1.40),
        joint_velocity_limits=(1.0, 1.0),
        max_velocity_fraction=0.98,
        velocity_fraction_range=(0.35, 0.98),
    )

    caps = env._arm_async_joint_velocity_cap_fraction
    planned = env._arm_async_joint_peak_velocity_fraction
    assert torch.min(caps).item() >= 0.35
    assert torch.max(caps).item() <= 0.98
    assert torch.std(caps).item() > 0.10
    assert torch.mean((planned >= 0.90).float()).item() > 0.01
    assert torch.all(planned <= caps + 1.0e-6)


def test_async_arm_reset_ignores_stale_physics_state_and_holds_nominal_at_zero_difficulty():
    _, sample_async, track_async = _load_functions(
        EVENTS_PATH,
        "_as_per_env_tensor",
        "random_async_arm_joint_motion",
        "continuous_async_arm_joint_tracking",
    )

    class Articulation:
        def __init__(self):
            # Non-arm entries make it possible to verify that tracking writes
            # only the selected arm joints, not a stale full-articulation target.
            self.data = SimpleNamespace(joint_pos=torch.tensor(((9.0, 0.8, 9.0, -0.8),)))
            self.target_calls = []

        def set_joint_position_target(self, target, *, joint_ids, env_ids):
            self.target_calls.append(
                (target.clone(), torch.as_tensor(joint_ids).clone(), torch.as_tensor(env_ids).clone())
            )

    art = Articulation()

    class Scene:
        num_envs = 1

        def __getitem__(self, name):
            assert name == "robot"
            return art

    nominal = torch.tensor((0.1, -0.2))
    env = SimpleNamespace(
        device=torch.device("cpu"),
        scene=Scene(),
        step_dt=0.02,
        _loco_manip_arm_motion_enabled=torch.tensor((True,)),
        _loco_manip_arm_motion_difficulty=torch.tensor((0.0,)),
    )
    asset_cfg = SimpleNamespace(name="robot", joint_ids=[1, 3], joint_names=["joint_a", "joint_b"])

    sample_async(
        env,
        None,
        asset_cfg,
        joint_position_ranges={"joint_a": (-1.0, 1.0), "joint_b": (-1.0, 1.0)},
        nominal_joint_pos=tuple(nominal.tolist()),
        safe_waypoints=(),
        reset_to_nominal=True,
    )
    track_async(env, None, asset_cfg)

    assert torch.allclose(env._arm_async_joint_start_pos[0], nominal)
    assert torch.allclose(env._arm_async_joint_goal_pos[0], nominal)
    assert torch.allclose(env._arm_async_joint_target_pos[0], nominal)
    assert len(art.target_calls) == 2
    assert all(call[0].shape == (1, 2) for call in art.target_calls)
    assert torch.allclose(art.target_calls[-1][0][0], nominal)
    assert torch.equal(art.target_calls[-1][1], torch.tensor((1, 3)))


def test_loco_manipulation_replay_retains_held_and_easier_arm_environments():
    sample_replay = _load_function(CURRICULUMS_PATH, "_sample_loco_manipulation_replay")
    env_ids = torch.arange(10_000)
    torch.manual_seed(23)

    enabled, difficulty = sample_replay(
        env_ids,
        stage=2,
        frontier_difficulty=0.8,
        replay_fraction=0.25,
        hold_fraction=0.10,
    )

    hold_fraction = (~enabled).float().mean()
    easier_fraction = (enabled & (difficulty < 0.8)).float().mean()
    frontier_fraction = torch.isclose(difficulty, torch.tensor(0.8)).float().mean()
    assert 0.08 < hold_fraction < 0.12
    assert 0.23 < easier_fraction < 0.27
    assert 0.63 < frontier_fraction < 0.67
    assert torch.all(difficulty[~enabled] == 0.0)

    enabled, difficulty = sample_replay(
        env_ids,
        stage=0,
        frontier_difficulty=1.0,
        replay_fraction=0.25,
        hold_fraction=0.10,
    )
    assert not torch.any(enabled)
    assert torch.all(difficulty == 0.0)


def test_isaac_workspace_tools_remove_stale_arm_clip_patterns():
    expected_reset = 'env_cfg.actions.joint_pos.clip = {".*": (-100.0, 100.0)}'
    for path in (ISAAC_WORKSPACE_VIS_PATH, ISAAC_WORKSPACE_EVAL_PATH, ISAAC_CURRICULUM_PREVIEW_PATH):
        source = path.read_text()
        assert expected_reset in source


def test_fixed_arm_curriculum_preview_uses_the_training_schedule_and_events():
    source = ISAAC_CURRICULUM_PREVIEW_PATH.read_text()

    assert "articulation_props.fix_root_link = True" in source
    assert 'advance_arm.params["apply_target"] = True' in source
    assert 'advance_arm.params["apply_gripper_target"] = True' in source
    assert 'advance_arm.params["gripper_joint_names"] = tuple(env_cfg.gripper_joint_names)' in source
    assert 'curriculum.params.get("stage_iteration_bins")' in source
    assert 'curriculum.params.get("arm_difficulty_iteration_bins")' in source
    assert '"stage_iteration_bins": None' in source
    assert '"arm_difficulty_iteration_bins": None' in source
    assert "env_cfg.events.randomize_arm_command = None" not in source
    assert "curriculum.func(base_env, env_ids, **curriculum.params)" in source


def test_pick_place_gripper_commands_follow_grasp_and_release_phases():
    gripper_command = _load_function(EVENTS_PATH, "_arm_motion_gripper_command")

    assert [gripper_command("pick_place", phase) for phase in range(8)] == [
        False,
        False,
        True,
        True,
        True,
        True,
        False,
        False,
    ]
    assert gripper_command("reach", 0) is None
    assert gripper_command("workspace", 0) is None

    event_source = EVENTS_PATH.read_text()
    assert "waypoints = (lift_pick, pick, pick, lift_pick, lift_place, place, place, lift_place)" in event_source
    assert "if not motion_active and torch.rand" in event_source
    assert "if apply_gripper_target and gripper_joint_names" in event_source
    assert "full_target[:, joint_id] = env._gripper_target_pos[:, i]" in event_source


def test_pick_and_hierarchical_actors_observe_table_geometry():
    pick_source = PICK_CFG_PATH.read_text()
    hierarchical_source = HIERARCHICAL_CFG_PATH.read_text()

    assert "self.observations.policy.table_geometry_points = ObsTerm(" in pick_source
    assert '"object_cfg": table_cfg' in pick_source
    assert "self.observations.critic.table_geometry_points = copy.deepcopy(" in pick_source
    assert "for term_name in dir(pick_policy):" in hierarchical_source
    assert "setattr(env_cfg.observations.policy, term_name, copy.deepcopy(term))" in hierarchical_source


def test_frozen_wbc_runtime_disables_unused_differential_ik_event():
    source = WBC_CFG_PATH.read_text()
    function = ast.parse(source)
    runtime = next(
        node
        for node in function.body
        if isinstance(node, ast.FunctionDef) and node.name == "configure_go2_d1_wbc_hierarchical_runtime"
    )
    runtime_source = ast.get_source_segment(source, runtime)

    assert runtime_source is not None
    assert "env_cfg.events.advance_arm_command = None" in runtime_source


def test_hierarchical_wbc_play_is_not_treated_as_scripted_wbc_inspection():
    source = RSL_RL_PLAY_PATH.read_text()

    assert 'and "locomanip" not in task_name' in source
    assert 'if getattr(actions_cfg, "wbc_command", None) is not None:' in source


def test_play_stage_preserves_task_velocity_and_posture_envelopes():
    set_ranges = _load_function(RSL_RL_PLAY_PATH, "_set_base_velocity_cfg")
    ranges = SimpleNamespace(
        lin_vel_x=(-0.45, 0.55),
        lin_vel_y=(-0.20, 0.20),
        ang_vel_z=(-0.45, 0.45),
        heading=(0.0, 0.0),
    )
    command_cfg = SimpleNamespace(
        ranges=ranges,
        roll_range=(0.0, 0.0),
        pitch_range=(-0.16, 0.12),
        heading_command=False,
    )

    set_ranges(command_cfg, "combined", 0.5)
    assert ranges.lin_vel_x == (-0.45, 0.55)
    assert ranges.lin_vel_y == (-0.20, 0.20)
    assert ranges.ang_vel_z == (-0.45, 0.45)
    assert command_cfg.roll_range == (0.0, 0.0)
    assert command_cfg.pitch_range == (-0.08, 0.06)

    set_ranges(command_cfg, "arm", 0.5)
    assert ranges.lin_vel_x == (0.0, 0.0)
    assert ranges.lin_vel_y == (0.0, 0.0)
    assert ranges.ang_vel_z == (0.0, 0.0)

    set_ranges(command_cfg, "fixed", 0.0)
    assert ranges.lin_vel_x == (-0.45, 0.55)
    assert ranges.lin_vel_y == (-0.20, 0.20)
    assert ranges.ang_vel_z == (-0.45, 0.45)
    assert command_cfg.pitch_range == (0.0, 0.0)

    play_source = RSL_RL_PLAY_PATH.read_text()
    assert "ranges.lin_vel_x = (-1.0, 1.0)" not in play_source
    assert 'params.get("walking_lin_vel_x", ranges.lin_vel_x)' in play_source


def test_go2_d1_play_is_nominal_by_default_but_keeps_arm_commands():
    disable_randomization = _load_function(RSL_RL_PLAY_PATH, "_disable_go2_d1_play_randomization")
    events = SimpleNamespace(
        randomize_rigid_body_mass_base=object(),
        randomize_actuator_gains=object(),
        randomize_reset_base=object(),
        randomize_reset_joints=object(),
        randomize_push_robot=object(),
        randomize_arm_command=object(),
        advance_arm_command=object(),
    )
    disabled = disable_randomization(SimpleNamespace(events=events))

    assert set(disabled) == {
        "randomize_rigid_body_mass_base",
        "randomize_actuator_gains",
        "randomize_reset_base",
        "randomize_reset_joints",
        "randomize_push_robot",
    }
    assert events.randomize_rigid_body_mass_base is None
    assert events.randomize_actuator_gains is None
    assert events.randomize_reset_base is None
    assert events.randomize_reset_joints is None
    assert events.randomize_push_robot is None
    assert events.randomize_arm_command is not None
    assert events.advance_arm_command is not None

    play_source = RSL_RL_PLAY_PATH.read_text()
    assert '"--go2-d1-play-domain-randomization"' in play_source


def test_go2_d1_play_grid_spans_easiest_to_full_curriculum():
    grid_slices = _load_function(RSL_RL_PLAY_PATH, "_loco_manip_play_grid_slices")
    extreme_probe = _load_function(RSL_RL_PLAY_PATH, "_loco_manip_extreme_probe")

    stages, difficulties = grid_slices(2, torch.device("cpu"))
    assert torch.equal(stages, torch.tensor((0, 2)))
    assert torch.allclose(difficulties, torch.tensor((0.0, 1.0)))

    stages, difficulties = grid_slices(5, torch.device("cpu"))
    assert torch.equal(stages, torch.tensor((0, 0, 1, 2, 2)))
    assert torch.allclose(difficulties, torch.tensor((0.0, 0.0, 0.5, 0.75, 1.0)))

    ranges = SimpleNamespace(
        lin_vel_x=(-1.5, 1.5),
        lin_vel_y=(-0.5, 0.5),
        ang_vel_z=(-1.5, 1.5),
    )
    expected_probes = (
        ("forward", (1.5, 0.0, 0.0)),
        ("backward", (-1.5, 0.0, 0.0)),
        ("left", (0.0, 0.5, 0.0)),
        ("right", (0.0, -0.5, 0.0)),
        ("left_yaw", (0.0, 0.0, 1.5)),
        ("right_yaw", (0.0, 0.0, -1.5)),
        ("positive_corner", (1.5, 0.5, 1.5)),
        ("negative_corner", (-1.5, -0.5, -1.5)),
    )
    assert tuple(extreme_probe(ranges, phase) for phase in range(8)) == expected_probes

    play_source = RSL_RL_PLAY_PATH.read_text()
    assert 'play_difficulty = 1.0 if stage == "grid"' in play_source
    assert 'state["raw_posture_command"] = command_term.posture_command.clone()' in play_source
    assert "scaled_posture[:, 0] = 0.0" in play_source
    assert "scaled_posture[:, 1] *= scale[:, 0]" in play_source
    assert "_loco_manip_extreme_probe(command_term.cfg.ranges, probe_phase)" in play_source
    assert "command_term.vel_command_b[walking_envs, :] = torch.tensor(" in play_source
    assert 'scaled_posture[probe_env, 1] = float(pitch_range[posture_phase])' in play_source
    assert '"--loco-manip-grid-probe-steps"' in play_source
    assert "env_cfg.viewer.origin_type = \"world\"" in play_source


def test_wbc_learns_smoothness_and_cartesian_hierarchy_has_physical_command_shaping():
    wbc_source = WBC_CFG_PATH.read_text()
    actions_source = ACTIONS_PATH.read_text()
    hierarchical_source = HIERARCHICAL_CFG_PATH.read_text()

    assert "RateLimitedJointPositionAction" not in actions_source
    assert "arm_joint_velocity_limit" not in actions_source
    assert "def _slew_limit_normalized_actions(" in actions_source
    assert "normalized_action_rate_limits" in actions_source
    assert "gripper_open_threshold" in actions_source
    assert "gripper_close_threshold" in actions_source
    assert "func=mdp.action_rate_l2_selected" in wbc_source
    assert "func=mdp.action_smoothness_l2_selected" in wbc_source
    assert "func=mdp.action_rate_l2_selected" in hierarchical_source
    assert "func=mdp.action_smoothness_l2_selected" in hierarchical_source
    assert "env_cfg.rewards.action_rate_l2.func = mdp.action_rate_l2_after_reset" in wbc_source


def test_cartesian_high_level_slew_limit_preserves_direction_and_rate():
    slew_limit = _load_function(ACTIONS_PATH, "_slew_limit_normalized_actions")
    previous = torch.tensor(((0.0, 0.4, -0.8),), dtype=torch.float32)
    target = torch.tensor(((1.0, -1.0, -0.82),), dtype=torch.float32)
    rates = torch.tensor((4.0, 1.0, 0.5), dtype=torch.float32)

    applied = slew_limit(previous, target, rates, 0.02)

    assert torch.allclose(applied, torch.tensor(((0.08, 0.38, -0.81),)), atol=1.0e-6)


def test_cartesian_high_level_reward_graph_does_not_retrain_the_frozen_leg_wbc():
    disabled = set(_literal_assignment(HIERARCHICAL_CFG_PATH, "_CARTESIAN_HIGH_LEVEL_DISABLED_REWARDS"))
    expected_low_level_terms = {
        "track_lin_vel_xy_exp",
        "track_ang_vel_z_exp",
        "feet_air_time",
        "feet_air_time_variance",
        "feet_gait",
        "feet_contact_without_cmd",
        "feet_slide",
        "feet_height_body",
        "joint_torques_l2",
        "joint_acc_l2",
        "joint_pos_limits",
        "joint_power",
        "stand_still",
        "joint_pos_penalty",
        "joint_mirror",
        "contact_forces",
    }
    assert expected_low_level_terms <= disabled

    source = HIERARCHICAL_CFG_PATH.read_text()
    tree = ast.parse(source)
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_configure_cartesian_high_level_rewards"
    )
    helper_source = ast.get_source_segment(source, helper)
    cartesian_cfg = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "UnitreeGo2D1PickLegWbcEeHierarchicalFlatEnvCfg"
    )
    cartesian_source = ast.get_source_segment(source, cartesian_cfg)

    assert helper_source is not None
    assert cartesian_source is not None
    assert "_configure_cartesian_high_level_rewards(self, pick_cfg)" in cartesian_source
    assert 'params={"action_indices": (0, 1, 2, 3, 4)}' in helper_source
    assert 'params={"action_indices": (5, 6, 7, 8, 9)}' in helper_source
    assert "func=mdp.base_velocity_towards_object_standoff" in helper_source
    assert "arm_joint_velocity_l2" in helper_source
    assert "arm_joint_acceleration_l2" in helper_source
    assert "arm_joint_torques_l2" in helper_source
    assert "env_cfg.rewards.undesired_contacts" not in helper_source


def test_pick_stage_graph_uses_hysteresis_and_allows_recovery():
    advance_stage = _load_function(REWARDS_PATH.parent / "loco_manipulation.py", "_advance_pick_stage")

    stage = advance_stage(
        torch.zeros(4, dtype=torch.long),
        torch.tensor((0.80, 0.55, 0.55, 0.55)),
        torch.tensor((0.50, 0.50, 0.10, 0.10)),
        torch.tensor((0.0, 0.0, 0.0, 0.80)),
        0.60,
        0.13,
    )
    assert torch.equal(stage, torch.tensor((0, 1, 2, 3)))

    # Values between enter and exit thresholds must not chatter between stages.
    stage = advance_stage(
        torch.tensor((1, 2, 3)),
        torch.tensor((0.64, 0.64, 0.64)),
        torch.tensor((0.20, 0.16, 0.10)),
        torch.tensor((0.0, 0.0, 0.50)),
        0.60,
        0.13,
    )
    assert torch.equal(stage, torch.tensor((1, 2, 3)))

    # Losing reach, pre-grasp alignment, or a completed lift re-enables earlier rewards.
    stage = advance_stage(
        torch.tensor((1, 2, 3, 3)),
        torch.tensor((0.70, 0.60, 0.60, 0.75)),
        torch.tensor((0.50, 0.19, 0.20, 0.10)),
        torch.tensor((0.0, 0.0, 0.10, 0.10)),
        0.60,
        0.13,
    )
    assert torch.equal(stage, torch.tensor((0, 1, 1, 0)))

    # A height spike without a real bilateral grasp must not enter the hold stage.
    stage = advance_stage(
        torch.tensor((2, 2)),
        torch.tensor((0.55, 0.55)),
        torch.tensor((0.10, 0.10)),
        torch.tensor((0.80, 0.80)),
        0.60,
        0.13,
        grasp_contact=torch.tensor((False, True)),
    )
    assert torch.equal(stage, torch.tensor((2, 3)))


def test_cartesian_pick_rewards_are_conditioned_on_their_relevant_stage():
    stage_map = _literal_assignment(HIERARCHICAL_CFG_PATH, "_CARTESIAN_PICK_REWARD_STAGES")

    assert stage_map["base_approach_outside_arm_reach"] == (0,)
    assert stage_map["base_velocity_towards_standoff"] == (0,)
    assert "ee_to_object" not in stage_map
    assert stage_map["ee_reach_progress"] == (1,)
    assert stage_map["ee_reach_alignment_error"] == (0, 1)
    assert stage_map["ee_grasp_alignment_error"] == (2,)
    assert stage_map["grasp_frame_horizontal_error"] == (0, 1, 2, 3)
    assert stage_map["gripper_closed_before_pregrasp"] == (0, 1)
    assert stage_map["gripper_close_near_object"] == (2,)
    assert stage_map["object_lifted"] == (2, 3)
    assert stage_map["object_hold_lifted"] == (3,)
    assert stage_map["object_vertical"] == (3,)

    source = HIERARCHICAL_CFG_PATH.read_text()
    tree = ast.parse(source)
    cartesian_cfg = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "UnitreeGo2D1PickLegWbcEeHierarchicalFlatEnvCfg"
    )
    cartesian_source = ast.get_source_segment(source, cartesian_cfg)
    assert cartesian_source is not None
    assert "_configure_cartesian_pick_stages(self, pick_cfg)" in cartesian_source

    for class_name in ("UnitreeGo2D1PickWbcHierarchicalFlatEnvCfg", "UnitreeGo2D1PickLegWbcArmHierarchicalFlatEnvCfg"):
        legacy_cfg = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        legacy_source = ast.get_source_segment(source, legacy_cfg)
        assert legacy_source is not None
        assert "_configure_cartesian_pick_stages" not in legacy_source

    reward_source = (REWARDS_PATH.parent / "loco_manipulation.py").read_text()
    assert "def pick_stage_one_hot(" in reward_source
    assert '"_loco_manip_pick_stage"' in reward_source
    assert "def pick_stage_transition_bonus(" in reward_source
    assert "def pick_success_bonus(" in reward_source
    assert "return _apply_pick_stage_gate(env, reward, active_pick_stages, pick_stage_params)" in reward_source


def test_cartesian_pick_initial_state_curriculum_learns_near_pick_then_approach():
    source = HIERARCHICAL_CFG_PATH.read_text()
    tree = ast.parse(source)
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_configure_cartesian_pick_initial_state_curriculum"
    )
    helper_source = ast.get_source_segment(source, helper)
    assert helper_source is not None

    assert '"performance_based": True' in helper_source
    assert '"performance_threshold": 0.20' in helper_source
    assert '"performance_success_count_attr": "_loco_manip_pick_success_count"' in helper_source
    assert '"distance_iteration_bins"' not in helper_source
    assert '"inject_velocity_commands": False' in helper_source
    assert "object_distance_end = float(pick_cfg.approach_distance_end)" in helper_source
    source_cfg = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_UnitreeGo2D1HighLevelPickSourceCfg"
    )
    source_cfg_source = ast.get_source_segment(source, source_cfg)
    assert source_cfg_source is not None
    assert "approach_distance_start = 0.65" in source_cfg_source
    assert "approach_distance_end = 1.20" in source_cfg_source
    assert "object_table_offset = (-0.12, 0.0)" in source_cfg_source
    assert '"sample_distance_within_curriculum": True' in source_cfg_source
    assert '"easy_replay_fraction": 0.15' in source_cfg_source
    assert '"frontier_fraction": 0.25' in source_cfg_source
    # Difficulty zero reproduces the validated nearby pick geometry, keeping
    # every x-jittered object in Reach. Success then expands to 1.20 m.
    assert 0.65 - 0.12 + 0.02 < 0.60

    cartesian_cfg = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "UnitreeGo2D1PickLegWbcEeHierarchicalFlatEnvCfg"
    )
    cartesian_source = ast.get_source_segment(source, cartesian_cfg)
    assert cartesian_source is not None
    assert "_configure_cartesian_pick_initial_state_curriculum(self, pick_cfg)" in cartesian_source


def test_cartesian_pick_teacher_has_full_object_pose_proprioception_and_history():
    hierarchy_source = HIERARCHICAL_CFG_PATH.read_text()
    agent_source = RSL_RL_AGENT_CFG_PATH.read_text()

    assert "def _configure_cartesian_teacher_observations(" in hierarchy_source
    assert "func=mdp.object_pose_6d_b" in hierarchy_source
    assert "func=mdp.base_to_object_standoff_b" in hierarchy_source
    assert "func=mdp.frozen_wbc_processed_actions" in hierarchy_source
    assert "group.history_length = 5" in hierarchy_source
    assert "env_cfg.observations.teacher = copy.deepcopy(env_cfg.observations.critic)" in hierarchy_source
    assert "env_cfg.observations.teacher.enable_corruption = False" in hierarchy_source
    assert "low_level_critic = copy.deepcopy(env_cfg.observations.critic)" in hierarchy_source
    assert "_configure_cartesian_teacher_observations(self, pick_cfg)" in hierarchy_source
    for proprioceptive_term in (
        '"base_lin_vel"',
        '"base_ang_vel"',
        '"projected_gravity"',
        '"joint_pos"',
        '"joint_vel"',
    ):
        assert proprioceptive_term in hierarchy_source

    tree = ast.parse(agent_source)
    runner = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "UnitreeGo2D1PickLegWbcEeHierarchicalFlatPPORunnerCfg"
    )
    runner_source = ast.get_source_segment(agent_source, runner)
    assert runner_source is not None
    assert 'self.obs_groups = {"actor": ["teacher"], "critic": ["teacher"]}' in runner_source
    assert 'self.experiment_name = "unitree_go2_d1_pick_leg_wbc_ee_teacher_flat"' in runner_source

    cusrl_source = CUSRL_AGENT_CFG_PATH.read_text()
    registry_source = GO2_D1_REGISTRY_PATH.read_text()
    assert "class UnitreeGo2D1PickLegWbcEeTeacherFlatTrainerCfg" in cusrl_source
    assert 'self.obs_groups = {"actor": ["teacher"], "critic": ["teacher"]}' in cusrl_source
    assert "cusrl_ppo_cfg:UnitreeGo2D1PickLegWbcEeTeacherFlatTrainerCfg" in registry_source


def test_cartesian_pick_objects_are_graspable_and_below_payload_limit():
    source = HIERARCHICAL_CFG_PATH.read_text()
    tree = ast.parse(source)
    source_cfg = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "_UnitreeGo2D1HighLevelPickSourceCfg"
    )
    source_cfg_source = ast.get_source_segment(source, source_cfg)
    assert source_cfg_source is not None

    assert "object_radius = 0.028" in source_cfg_source
    assert "object_height = 0.10" in source_cfg_source
    assert 'self.events.randomize_object_scale.params["scale_range"] = (0.95, 1.05)' in source_cfg_source
    assert "self.events.randomize_object_mass.func = mdp.randomize_rigid_body_mass_fn" in source_cfg_source
    assert '"mass_distribution_params": (0.04, 0.25)' in source_cfg_source
    assert '"operation": "abs"' in source_cfg_source
    assert 2.0 * 0.028 * 1.05 < 0.07
    assert 0.25 < 0.4


def test_cartesian_pick_fast_variant_replicates_physics_and_compacts_contact_sensors():
    hierarchy_source = HIERARCHICAL_CFG_PATH.read_text()
    hierarchy_tree = ast.parse(hierarchy_source)
    fast_cfg = next(
        node
        for node in hierarchy_tree.body
        if isinstance(node, ast.ClassDef)
        and node.name == "UnitreeGo2D1PickLegWbcEeHierarchicalFastFlatEnvCfg"
    )
    fast_source = ast.get_source_segment(hierarchy_source, fast_cfg)
    assert fast_source is not None
    assert "self.events.randomize_object_scale = None" in fast_source
    assert "self.scene.replicate_physics = True" in fast_source
    assert "_configure_compact_collision_safety(self)" in fast_source

    pick_source = PICK_CFG_PATH.read_text()
    pick_tree = ast.parse(pick_source)
    compact_helper = next(
        node
        for node in pick_tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_configure_compact_collision_safety"
    )
    compact_source = ast.get_source_segment(pick_source, compact_helper)
    assert compact_source is not None
    assert 'setattr(cfg.scene, sensor_name, None)' in compact_source
    assert 'setattr(cfg.rewards, sensor_name, None)' in compact_source
    assert 'prim_path="{ENV_REGEX_NS}/Robot/d1/(base_link|Link[1-6])"' in compact_source
    assert "history_length=1" in compact_source
    assert "filter_prim_paths_expr" not in compact_source
    assert '"d1_compact_hard_contact"' in compact_source

    registry_source = GO2_D1_REGISTRY_PATH.read_text()
    runner_source = RSL_RL_AGENT_CFG_PATH.read_text()
    cusrl_source = CUSRL_AGENT_CFG_PATH.read_text()
    assert (
        "Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-"
        "Pick-LegWbcEe-Hierarchical-Fast-v0"
    ) in registry_source
    assert "UnitreeGo2D1PickLegWbcEeHierarchicalFastFlatPPORunnerCfg" in runner_source
    assert 'self.experiment_name = "unitree_go2_d1_pick_leg_wbc_ee_teacher_fast_flat"' in runner_source
    assert "UnitreeGo2D1PickLegWbcEeTeacherFastFlatTrainerCfg" in cusrl_source


def test_object_pose_observation_uses_continuous_rotation_and_approach_velocity_is_signed():
    source = (REWARDS_PATH.parent / "loco_manipulation.py").read_text()
    tree = ast.parse(source)
    pose_fn = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "object_pose_6d_b")
    velocity_fn = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "base_velocity_towards_object_standoff"
    )
    reset_fn = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "reset_pick_approach_scene"
    )
    pose_source = ast.get_source_segment(source, pose_fn)
    velocity_source = ast.get_source_segment(source, velocity_fn)
    reset_source = ast.get_source_segment(source, reset_fn)
    assert pose_source is not None and velocity_source is not None and reset_source is not None

    assert "orientation_6d_b" in pose_source
    assert "torch.cat((position_b, orientation_6d_b), dim=-1)" in pose_source
    assert "speed_towards_standoff" in velocity_source
    assert "torch.clamp(speed_towards_standoff" in velocity_source
    assert "distance_gate" in velocity_source
    assert "sample_distance_within_curriculum" in reset_source
    assert "distance[easy] = min_distance" in reset_source
    assert "distance[frontier] = frontier_distance" in reset_source


def test_rsl_rl_trainer_scales_pick_distance_curriculum_with_requested_run_length():
    source = RSL_RL_TRAIN_PATH.read_text()
    assert '"distance_iteration_bins"' in source


def test_pick_resets_do_not_write_velocity_to_kinematic_tables():
    source = (REWARDS_PATH.parent / "loco_manipulation.py").read_text()
    assert "table_asset.write_root_pose_to_sim" in source
    assert "table_asset.write_root_velocity_to_sim" not in source


def test_wbc_sampler_mix_covers_global_workspace_without_local_step_cap():
    source = WBC_CFG_PATH.read_text()

    assert '"global_workspace_probability": 0.55' in source
    assert '"max_pos_change": None' in source
    assert '"motion_speed_range": (0.12, 0.32)' in source
    assert "interval_range_s=(3.0, 5.0)" in source


def test_selected_smoothness_rewards_allow_diagnostic_action_subsets():
    action_rate, action_magnitude, action_smoothness = _load_functions(
        REWARDS_PATH,
        "action_rate_l2_selected",
        "action_l2_selected",
        "action_smoothness_l2_selected",
    )

    class ActionManager:
        action = torch.zeros((2, 12))
        prev_action = torch.zeros((2, 12))

    class Env:
        device = torch.device("cpu")
        num_envs = 2
        episode_length_buf = torch.ones(2, dtype=torch.long)
        action_manager = ActionManager()

    assert torch.equal(action_rate(Env(), tuple(range(12, 20))), torch.zeros(2))
    assert torch.equal(action_magnitude(Env(), tuple(range(12, 20))), torch.zeros(2))
    assert torch.equal(action_smoothness(Env(), tuple(range(12, 20))), torch.zeros(2))


def test_go2_d1_leg_soft_limits_are_independent_of_the_gripper_asset_margin():
    reward = _load_function(REWARDS_PATH, "joint_pos_limits_with_soft_factor")

    asset = SimpleNamespace(
        data=SimpleNamespace(
            default_joint_pos_limits=torch.tensor([[[-1.0, 1.0]], [[-1.0, 1.0]]]),
            joint_pos=torch.tensor([[0.95], [0.70]]),
        )
    )
    env = SimpleNamespace(scene={"robot": asset})
    asset_cfg = SimpleNamespace(name="robot", joint_ids=[0])

    assert torch.allclose(reward(env, 0.9, asset_cfg), torch.tensor([0.05, 0.0]))


def test_payload_randomization_scales_with_per_environment_curriculum_difficulty():
    randomize_payload = _load_function(EVENTS_PATH, "randomize_end_effector_payload_mass")

    class RootView:
        def __init__(self):
            self.masses = torch.ones((2, 1))
            self.inertias = torch.ones((2, 1, 9))

        def get_masses(self):
            return self.masses.clone()

        def set_masses(self, masses, _env_ids):
            self.masses = masses.clone()

        def get_inertias(self):
            return self.inertias.clone()

        def set_inertias(self, inertias, _env_ids):
            self.inertias = inertias.clone()

    root_view = RootView()
    asset = SimpleNamespace(
        root_physx_view=root_view,
        data=SimpleNamespace(
            default_mass=torch.ones((2, 1)),
            default_inertia=torch.ones((2, 1, 9)),
        ),
    )
    class Scene(dict):
        pass

    scene = Scene(robot=asset)
    scene.num_envs = 2
    env = SimpleNamespace(
        scene=scene,
        device=torch.device("cpu"),
        _loco_manip_arm_motion_difficulty=torch.tensor([0.0, 1.0]),
    )
    asset_cfg = SimpleNamespace(name="robot", body_ids=[0])

    torch.manual_seed(7)
    randomize_payload(
        env,
        torch.tensor([0, 1]),
        asset_cfg,
        payload_mass_range=(1.0, 2.0),
        difficulty_attr="_loco_manip_arm_motion_difficulty",
    )

    assert env._arm_payload_mass[0].item() == 0.0
    assert 1.0 <= env._arm_payload_mass[1].item() <= 2.0
    assert root_view.masses[0, 0].item() == 1.0
    assert root_view.masses[1, 0].item() > 2.0


def test_leg_wbc_diagnostics_report_unbounded_actions_and_soft_limit_targets():
    action_diagnostics, limit_diagnostics = _load_functions(
        CURRICULUMS_PATH,
        "normalized_action_diagnostics",
        "joint_target_limit_diagnostics",
    )

    action_term = SimpleNamespace(
        raw_actions=torch.tensor([[0.5, 1.5], [-2.0, 0.0]]),
        processed_actions=torch.tensor([[0.5, 0.95], [-0.95, 0.0]]),
        _joint_ids=torch.tensor([0, 1]),
    )

    class ActionManager:
        def get_term(self, name):
            assert name == "joint_pos"
            return action_term

    asset = SimpleNamespace(
        num_joints=2,
        data=SimpleNamespace(
            default_joint_pos_limits=torch.tensor(
                [[[-1.0, 1.0], [-1.0, 1.0]], [[-1.0, 1.0], [-1.0, 1.0]]]
            ),
            joint_pos=torch.tensor([[0.2, 0.4], [-0.1, 0.92]]),
        ),
    )
    env = SimpleNamespace(
        device=torch.device("cpu"),
        action_manager=ActionManager(),
        scene={"robot": asset},
    )
    env_ids = torch.tensor([0, 1])

    action_metrics = action_diagnostics(env, env_ids, "joint_pos", saturation_threshold=1.0)
    assert action_metrics["outside_fraction"].item() == 0.5
    assert action_metrics["abs_max"].item() == 2.0

    limit_metrics = limit_diagnostics(
        env,
        env_ids,
        "joint_pos",
        0.9,
        SimpleNamespace(name="robot", joint_ids=[0, 1]),
    )
    assert limit_metrics["target_violation_fraction"].item() == 0.5
    assert limit_metrics["target_utilization_max"].item() > 1.0
    assert limit_metrics["measured_utilization_max"].item() > 1.0


def test_async_arm_motion_diagnostics_report_near_limit_and_active_fractions():
    (diagnostics,) = _load_functions(CURRICULUMS_PATH, "async_arm_motion_diagnostics")
    env = SimpleNamespace(
        device=torch.device("cpu"),
        _arm_async_joint_initialized=torch.tensor((True, True)),
        _arm_async_joint_peak_velocity_fraction=torch.tensor(((0.95, 0.40), (0.92, 0.70))),
        _arm_async_joint_progress=torch.tensor(((0.2, 1.0), (0.4, 0.8))),
        _arm_async_joint_duration=torch.ones((2, 2)),
        _arm_async_joint_start_pos=torch.zeros((2, 2)),
        _arm_async_joint_goal_pos=torch.ones((2, 2)),
    )

    metrics = diagnostics(env, torch.tensor((0, 1)), near_limit_threshold=0.90)
    assert math.isclose(metrics["planned_peak_fraction_max"].item(), 0.95, abs_tol=1.0e-6)
    assert metrics["near_limit_fraction"].item() == 0.5
    assert metrics["active_fraction"].item() == 0.75


def test_cartesian_pick_uses_bounded_actions_progress_rewards_and_success_diagnostics():
    hierarchy_source = HIERARCHICAL_CFG_PATH.read_text()
    agent_source = RSL_RL_AGENT_CFG_PATH.read_text()
    compat_source = RSL_RL_COMPAT_PATH.read_text()
    curriculum_source = CURRICULUMS_PATH.read_text()

    assert "use_beta_action_distribution: bool = True" in agent_source
    assert "self.clip_actions = 1.0" in agent_source
    assert 'class_name="BetaDistribution"' in compat_source
    assert "func=mdp.base_object_approach_progress" in hierarchy_source
    assert "func=mdp.ee_object_reach_progress" in hierarchy_source
    assert "func=mdp.pick_stage_transition_bonus" in hierarchy_source
    assert "env_cfg.terminations.pick_success = DoneTerm(" in hierarchy_source
    assert "func=mdp.normalized_action_saturation_fraction" in hierarchy_source
    assert "env_cfg.rewards.ee_to_object = None" in hierarchy_source
    assert "func=mdp.gripper_close_progress_near_object" in hierarchy_source
    assert "func=mdp.gripper_object_contact_progress" in hierarchy_source
    assert "func=mdp.is_terminated_term" in hierarchy_source
    assert "func=mdp.bilateral_gripper_contact_fraction" in hierarchy_source
    assert "func=mdp.pick_geometry_diagnostics" in hierarchy_source
    assert "env_cfg.rewards.arm_joint_pos_limits = RewTerm(" in hierarchy_source
    assert "func=mdp.joint_pos_limits" in hierarchy_source
    assert "def cumulative_pick_success_fraction(" in curriculum_source
    assert "def pick_geometry_diagnostics(" in curriculum_source


def test_cartesian_pick_resets_ready_without_replaying_the_carry_deployment():
    source = HIERARCHICAL_CFG_PATH.read_text()
    action_source = ACTIONS_PATH.read_text()
    tree = ast.parse(source)
    helper = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "_configure_cartesian_pick_ready_reset"
    )
    helper_source = ast.get_source_segment(source, helper)

    assert helper_source is not None
    assert "GO2_D1_WBC_WORKSPACE_READY_POSE" in helper_source
    assert "deployment_ee_waypoints = ()" in helper_source
    assert "_configure_cartesian_pick_ready_reset(self)" in source
    assert "grasp_midpoint_b = quat_apply_inverse(root_quat_w, grasp_midpoint_w - root_pos_w)" in action_source
    assert "self._applied_actions[env_ids_t, 5 + axis]" in action_source


def test_cartesian_pick_prevents_reach_escape_and_observes_grasp_state():
    hierarchy_source = HIERARCHICAL_CFG_PATH.read_text()
    action_source = ACTIONS_PATH.read_text()
    reward_source = (REWARDS_PATH.parent / "loco_manipulation.py").read_text()
    agent_source = RSL_RL_AGENT_CFG_PATH.read_text()

    assert "env_cfg.rewards.base_xy_drift_in_arm_reach = None" in hierarchy_source
    assert '"pregrasp_distance": 0.10' in hierarchy_source
    assert "func=mdp.pick_stage_regression_penalty" in hierarchy_source
    assert "weight=-10.0" in hierarchy_source
    assert "park_base_pick_stages" not in action_source
    assert "manipulation_base_velocity_scale" not in action_source
    assert "def _configure_external_high_level_velocity_command(" in hierarchy_source
    assert "command_cfg.resampling_time_range = (1.0e6, 1.0e6)" in hierarchy_source
    assert '"ee_reach_alignment_error": (0, 1)' in hierarchy_source
    assert '"grasp_frame_horizontal_error": (0, 1, 2, 3)' in hierarchy_source
    assert "env_cfg.rewards.upward = None" in hierarchy_source

    for observation_name in (
        "gripper_close_fraction",
        "left_gripper_object_contact",
        "right_gripper_object_contact",
        "object_lift_progress",
        "wrist_orientation",
    ):
        assert f"env_cfg.observations.policy.{observation_name} = ObsTerm(" in hierarchy_source

    assert "(previous < int(target_stage)) & (stage >= int(target_stage))" in reward_source
    assert 'getattr(env_cfg.rewards, reward_name).params["required_contacts"] = 2' in hierarchy_source
    assert "self.algorithm.entropy_coef = 0.002" in agent_source
    assert 'self.algorithm.schedule = "fixed"' in agent_source
    assert "self.algorithm.learning_rate = 3.0e-4" in agent_source


def test_cartesian_pick_disturbances_are_success_gated():
    hierarchy_source = HIERARCHICAL_CFG_PATH.read_text()
    curriculum_source = CURRICULUMS_PATH.read_text()

    assert 'self.events.randomize_reset_base.params["velocity_range"][axis] = (-0.05, 0.05)' in hierarchy_source
    assert 'self.events.randomize_push_robot.params["velocity_range"][axis] = (0.0, 0.0)' in hierarchy_source
    assert "func=mdp.pick_disturbance_curriculum" in hierarchy_source
    assert "func=mdp.env_attr_curriculum_metric" in hierarchy_source
    assert "def pick_disturbance_curriculum(" in curriculum_source
    assert "manipulation_base_velocity_scale" not in curriculum_source


def test_pick_stage_fraction_reports_rollout_occupancy_not_only_reset_envs():
    source = CURRICULUMS_PATH.read_text()
    tree = ast.parse(source)
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "pick_stage_fraction"
    )
    function_source = ast.get_source_segment(source, function)

    assert function_source is not None
    assert "del env_ids" in function_source
    assert "stage.index_select" not in function_source
    assert "torch.mean((stage == int(stage_index)).float())" in function_source


def test_forward_stage_bonus_pays_for_a_skipped_intermediate_stage():
    transition_bonus = _load_function(
        REWARDS_PATH.parent / "loco_manipulation.py",
        "pick_stage_transition_bonus",
    )
    transition_bonus.__globals__["pick_stage_index"] = lambda env, **_params: env.stage

    class Env:
        stage = torch.tensor((1,), dtype=torch.long)
        scene = {"object": SimpleNamespace(data=SimpleNamespace(root_pos_w=torch.zeros((1, 3))))}

    env = Env()
    params = {"object_cfg": SimpleNamespace(name="object")}
    assert torch.equal(transition_bonus(env, 2, params), torch.zeros(1))

    env.stage = torch.tensor((3,), dtype=torch.long)
    assert torch.equal(transition_bonus(env, 2, params), torch.ones(1))


def test_bounded_progress_does_not_pay_for_holding_a_completed_command():
    progress = _load_function(REWARDS_PATH.parent / "loco_manipulation.py", "_bounded_scalar_progress")

    class Env:
        episode_length_buf = torch.tensor((2,), dtype=torch.long)

    env = Env()
    assert torch.equal(
        progress(env, torch.tensor((0.2,)), "memory", 0.25, increasing_is_progress=True),
        torch.zeros(1),
    )
    assert torch.allclose(
        progress(env, torch.tensor((0.45,)), "memory", 0.25, increasing_is_progress=True),
        torch.ones(1),
    )
    assert torch.equal(
        progress(env, torch.tensor((0.45,)), "memory", 0.25, increasing_is_progress=True),
        torch.zeros(1),
    )


def test_measured_gripper_closure_handles_mirrored_joint_directions():
    closure = _load_function(REWARDS_PATH.parent / "loco_manipulation.py", "gripper_close_fraction")

    class Env:
        scene = {
            "robot": SimpleNamespace(
                data=SimpleNamespace(
                    joint_pos=torch.tensor(
                        (
                            (0.025, -0.025),
                            (0.000, 0.000),
                            (0.0125, -0.0125),
                        )
                    )
                )
            )
        }

    cfg = SimpleNamespace(name="robot", joint_ids=[0, 1])
    result = closure(Env(), cfg, (0.025, -0.025), (0.0, 0.0))
    assert torch.allclose(result, torch.tensor(((0.0, 0.0), (1.0, 1.0), (0.5, 0.5))))


def test_pick_disturbance_curriculum_unlocks_all_disturbances_together():
    curriculum = _load_function(CURRICULUMS_PATH, "pick_disturbance_curriculum")
    curriculum.__globals__["_episode_boundary"] = lambda _env: True

    class EventManager:
        terms = {
            "randomize_reset_base": SimpleNamespace(
                params={"velocity_range": {"x": (-0.05, 0.05), "yaw": (-0.05, 0.05)}}
            ),
            "randomize_push_robot": SimpleNamespace(params={"velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0)}}),
        }

        def get_term_cfg(self, name):
            return self.terms[name]

        def set_term_cfg(self, name, cfg):
            self.terms[name] = cfg

    class Env:
        device = torch.device("cpu")
        common_step_counter = 500
        event_manager = EventManager()
        _go2_d1_pick_approach_score = 0.25

    env = Env()
    progress = curriculum(env, ())
    assert torch.allclose(progress, torch.tensor(0.02))
    assert env.event_manager.terms["randomize_reset_base"].params["velocity_range"]["x"] == (
        -0.059000000000000004,
        0.059000000000000004,
    )
    assert env.event_manager.terms["randomize_push_robot"].params["velocity_range"]["x"] == (-0.01, 0.01)


def test_pick_disturbance_curriculum_tolerates_push_event_removed_for_play():
    curriculum = _load_function(CURRICULUMS_PATH, "pick_disturbance_curriculum")
    curriculum.__globals__["_episode_boundary"] = lambda _env: False

    class EventManager:
        terms = {
            "randomize_reset_base": SimpleNamespace(
                params={"velocity_range": {"x": (-0.05, 0.05), "yaw": (-0.05, 0.05)}}
            )
        }

        def get_term_cfg(self, name):
            if name not in self.terms:
                raise ValueError(f"Event term {name!r} not found.")
            return self.terms[name]

        def set_term_cfg(self, name, cfg):
            self.terms[name] = cfg

    env = SimpleNamespace(
        device=torch.device("cpu"),
        common_step_counter=0,
        event_manager=EventManager(),
    )

    progress = curriculum(env, ())

    assert torch.equal(progress, torch.tensor(0.0))
    assert env.event_manager.terms["randomize_reset_base"].params["velocity_range"]["x"] == (-0.05, 0.05)


def test_augmented_height_command_does_not_force_locomotion_activity():
    base_norm, posture_norm = _load_functions(
        REWARDS_PATH,
        "_base_motion_command_norm",
        "_motion_posture_command_norm",
    )

    class CommandManager:
        command = torch.tensor(
            (
                (0.0, 0.0, 0.0, 0.0, 0.0, 0.33),
                (0.2, 0.0, 0.0, 0.0, 0.0, 0.33),
                (0.0, 0.0, 0.0, 0.0, 0.1, 0.33),
            )
        )

        def get_command(self, _name):
            return self.command

    class Env:
        command_manager = CommandManager()

    assert torch.allclose(base_norm(Env(), "base_velocity"), torch.tensor((0.0, 0.2, 0.0)))
    assert torch.allclose(
        posture_norm(Env(), "base_velocity", posture_nominal_height=0.33),
        torch.tensor((0.0, 0.2, 0.1)),
    )


def test_wbc_height_reward_has_nonvanishing_reset_gradient():
    source = WBC_CFG_PATH.read_text()

    assert source.count('"posture_nominal_height"') == 4
    assert source.count('"std": 0.12') >= 2


def test_leg_wbc_trains_posture_control_with_async_joint_arm_disturbances():
    source = WBC_CFG_PATH.read_text()
    flat_source = FLAT_CFG_PATH.read_text()
    rough_source = ROUGH_CFG_PATH.read_text()
    action_source = ACTIONS_PATH.read_text()

    leg_start = source.index("def configure_go2_d1_leg_wbc_async_arm_controller")
    leg_end = source.index("def configure_go2_d1_leg_wbc_arm_hierarchical_runtime", leg_start)
    leg_source = source[leg_start:leg_end]
    full_wbc_start = source.index("def configure_go2_d1_whole_body_controller")
    full_wbc_end = source.index("def _disable_go2_d1_wbc_training_curricula", full_wbc_start)
    full_wbc_source = source[full_wbc_start:full_wbc_end]

    assert "env_cfg.actions.joint_pos.joint_names = env_cfg.leg_joint_names" in leg_source
    assert "func=mdp.random_async_arm_joint_motion" in leg_source
    assert "func=mdp.continuous_async_arm_joint_tracking" in leg_source
    assert "func=mdp.random_arm_ik_motion" not in leg_source
    assert "func=mdp.continuous_arm_ik_tracking" not in leg_source
    assert '"joint_position_ranges": GO2_D1_SAFE_ARM_JOINT_RANGES' in leg_source
    assert '"nominal_joint_pos": ready' in leg_source
    assert '"reset_to_nominal": True' in leg_source
    assert '"apply_target": True' in leg_source
    assert '"walking_pitch": GO2_D1_WBC_BODY_PITCH_RANGE' in leg_source
    assert '"walking_height": GO2_D1_WBC_BODY_HEIGHT_RANGE' in leg_source
    assert '"walking_lin_vel_x": (-1.50, 1.50)' in leg_source
    assert '"walking_lin_vel_y": (-0.50, 0.50)' in leg_source
    assert '"walking_ang_vel_z": (-1.50, 1.50)' in leg_source
    assert "lin_vel_x=(-1.50, 1.50)" in leg_source
    assert "lin_vel_y=(-0.50, 0.50)" in leg_source
    assert "ang_vel_z=(-1.50, 1.50)" in leg_source
    assert "lin_vel_x=(-0.45, 0.55)" in full_wbc_source
    assert "lin_vel_y=(-0.20, 0.20)" in full_wbc_source
    assert "ang_vel_z=(-0.45, 0.45)" in full_wbc_source
    assert '"performance_based": True' in leg_source
    assert '"stage0_reward_terms": ("track_lin_vel_xy_exp", "track_ang_vel_z_exp")' in leg_source
    assert '"stage1_reward_terms": ("track_base_roll_pitch_exp", "track_base_height_command")' in leg_source
    assert '"alternate_arm_difficulty_stages": True' in leg_source
    assert '"arm_difficulty_replay_fraction": 0.25' in leg_source
    assert '"held_arm_replay_fraction": 0.10' in leg_source
    assert 'curriculum_difficulty_attr="_loco_manip_arm_motion_difficulty"' in leg_source
    assert 'curriculum_frontier_difficulty_attr="_loco_manip_arm_motion_frontier"' in leg_source
    assert '"walking_roll": (0.0, 0.0)' in leg_source
    assert "env_cfg.curriculum.go2_d1_leg_wbc_locomotion_score = CurriculumTermCfg(" in leg_source
    assert 'params={"attr_name": "_loco_manip_stage0_score"}' in leg_source
    assert "env_cfg.curriculum.go2_d1_leg_wbc_posture_score = CurriculumTermCfg(" in leg_source
    assert 'params={"attr_name": "_loco_manip_stage1_score"}' in leg_source
    assert '"difficulty_attr": "_loco_manip_arm_motion_difficulty"' in leg_source
    assert '"payload_mass_range": (0.0, 0.4)' in leg_source
    assert '"waypoint_probability": 0.80' in leg_source
    assert '"max_joint_change_range": (0.15, 2.60)' in leg_source
    assert '"joint_velocity_limits": D1_ARM_HARDWARE_VELOCITY_LIMITS' in leg_source
    assert '"trajectory_duration_range": (0.35, 1.40)' in leg_source
    assert '"max_velocity_fraction": 0.98' in leg_source
    assert '"velocity_fraction_range": (0.35, 0.98)' in leg_source
    assert "func=mdp.normalized_action_diagnostics" in leg_source
    assert "func=mdp.joint_target_limit_diagnostics" in leg_source
    assert "func=mdp.velocity_command_diagnostics" in leg_source
    assert "func=mdp.async_arm_motion_diagnostics" in leg_source
    assert 'params={"attr_name": "_arm_payload_mass"}' in leg_source
    assert "reward.func = mdp.joint_pos_limits_with_soft_factor" in source
    assert 'reward.params["soft_factor"] = 0.9' in source
    assert "self.rewards.feet_air_time.weight = 0.2" in rough_source
    assert "self.rewards.feet_slide.weight = -0.1" in rough_source
    assert '"arm_1_joint": (-1.30, 1.30)' in source
    assert '"arm_2_joint": (-1.25, 1.30)' in source
    assert '"arm_3_joint": (-0.65, 1.55)' in source
    assert '"arm_6_joint": (-1.00, 1.00)' in source

    assert "self.terminations.terrain_out_of_bounds = None" in flat_source
    assert "self.terminations.bad_orientation = DoneTerm(" in flat_source
    assert 'params={"limit_angle": math.radians(70.0)}' in flat_source

    assert "self._low_level_actions[:] = torch.clamp(low_level_actions, -1.0, 1.0)" not in action_source
    assert "leg_targets = torch.clamp(leg_targets, min=self._leg_target_lower, max=self._leg_target_upper)" in action_source

    disable_start = source.index("def _disable_go2_d1_wbc_training_curricula")
    disable_end = source.index("def configure_go2_d1_leg_wbc_async_arm_controller", disable_start)
    disable_source = source[disable_start:disable_end]
    assert '"go2_d1_leg_wbc_locomotion_score"' in disable_source
    assert '"go2_d1_leg_wbc_posture_score"' in disable_source

    agent_source = RSL_RL_AGENT_CFG_PATH.read_text()
    for class_name in (
        "UnitreeGo2D1LegWbcAsyncArmRoughPPORunnerCfg",
        "UnitreeGo2D1LegWbcAsyncArmFlatPPORunnerCfg",
    ):
        class_start = agent_source.index(f"class {class_name}")
        class_end = agent_source.find("\n@configclass", class_start + 1)
        class_source = agent_source[class_start:] if class_end < 0 else agent_source[class_start:class_end]
        assert "self.clip_actions" not in class_source
        assert "self.policy.init_noise_std" not in class_source
        assert "self.algorithm.entropy_coef" not in class_source
        assert "self.algorithm.learning_rate" not in class_source

    train_source = RSL_RL_TRAIN_PATH.read_text()
    assert "from legacy_checkpoint import load_checkpoint_for_train" in train_source
    assert "load_checkpoint_for_train(runner, resume_path, args_cli.task)" in train_source


def test_cartesian_high_level_owns_ee_and_frozen_policy_owns_only_legs():
    action_source = ACTIONS_PATH.read_text()
    config_source = WBC_CFG_PATH.read_text()
    hierarchy_source = HIERARCHICAL_CFG_PATH.read_text()

    ee_start = action_source.index("class FrozenGo2D1LegWbcEeCommandAction(")
    ee_end = action_source.index("class FrozenGo2D1LegWbcEeCommandActionCfg", ee_start)
    ee_source = action_source[ee_start:ee_end]

    assert "return 10" in ee_source
    assert "actions[:, 3]" in ee_source
    assert "actions[:, 4]" in ee_source
    assert "actions[:, 5 + axis]" in ee_source
    assert "actions[:, 8]" in ee_source
    assert "actions[:, 9]" in ee_source
    assert "posture_command[:, 0] = 0.0" in ee_source
    assert "body_pitch_range" in ee_source
    assert "body_height_range" in ee_source
    assert 'command_type="pose"' in ee_source
    assert "self._grasp_body_ids" in ee_source
    assert "_link_position_from_grasp_target" in ee_source
    assert "ee_quat_des_b" in ee_source
    assert ee_source.count("_clamp_ee_targets(") == 2
    assert "root_rotation_w_to_b" in ee_source
    assert "jacobian_b[:, 0:3, :]" in ee_source
    assert "jacobian_b[:, 3:6, :]" in ee_source
    assert "len(cfg.leg_joint_names)" in action_source
    assert "FrozenGo2D1LegWbcEeCommandActionCfg" in config_source
    assert "UnitreeGo2D1PickLegWbcEeHierarchicalFlatEnvCfg" in hierarchy_source
