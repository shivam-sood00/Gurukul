"""Static and dependency-free contracts for useful Go2+D1 service tasks."""

from __future__ import annotations

import ast
import importlib.util
import math
import sys
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = REPO_ROOT / "source/Gurukul/Gurukul/tasks/manager_based/llm"
SERVICE_CFG_PATH = TASK_ROOT / "go2_d1_service_env_cfg.py"
REGISTRY_PATH = TASK_ROOT / "__init__.py"
MDP_PATH = TASK_ROOT / "mdp.py"
DOOR_URDF_PATH = REPO_ROOT / "source/Gurukul/data/Objects/service_door/hinged_door.urdf"
DOOR_USD_PATH = REPO_ROOT / "source/Gurukul/data/Objects/service_door/hinged_door.usda"
RUNNER_PATH = REPO_ROOT / "scripts/llm/run_go2_d1_service.py"
POLICY_PATH = REPO_ROOT / "scripts/llm/service_policy.py"
PRIMITIVE_POLICY_PATH = REPO_ROOT / "scripts/llm/service_primitive_policy.py"

sys.path.insert(0, str(POLICY_PATH.parent))
POLICY_SPEC = importlib.util.spec_from_file_location("go2_d1_service_policy_test_module", POLICY_PATH)
assert POLICY_SPEC is not None and POLICY_SPEC.loader is not None
POLICY_MODULE = importlib.util.module_from_spec(POLICY_SPEC)
sys.modules[POLICY_SPEC.name] = POLICY_MODULE
POLICY_SPEC.loader.exec_module(POLICY_MODULE)

PRIMITIVE_SPEC = importlib.util.spec_from_file_location("go2_d1_primitive_policy_test_module", PRIMITIVE_POLICY_PATH)
assert PRIMITIVE_SPEC is not None and PRIMITIVE_SPEC.loader is not None
PRIMITIVE_MODULE = importlib.util.module_from_spec(PRIMITIVE_SPEC)
sys.modules[PRIMITIVE_SPEC.name] = PRIMITIVE_MODULE
PRIMITIVE_SPEC.loader.exec_module(PRIMITIVE_MODULE)


def test_service_sources_parse() -> None:
    for path in (SERVICE_CFG_PATH, REGISTRY_PATH, MDP_PATH, RUNNER_PATH, POLICY_PATH, PRIMITIVE_POLICY_PATH):
        ast.parse(path.read_text())


def test_service_door_asset_has_a_bounded_vertical_revolute_hinge() -> None:
    root = ET.parse(DOOR_URDF_PATH).getroot()
    joint = root.find("./joint[@name='door_hinge']")
    assert joint is not None
    assert joint.attrib["type"] == "revolute"
    assert joint.find("axis").attrib["xyz"] == "0 0 1"
    limit = joint.find("limit")
    assert float(limit.attrib["lower"]) == 0.0
    assert float(limit.attrib["upper"]) == pytest.approx(math.radians(100.0))
    usd_source = DOOR_USD_PATH.read_text()
    assert 'def PhysicsRevoluteJoint "door_hinge"' in usd_source
    assert 'uniform token physics:axis = "Z"' in usd_source
    assert "float physics:upperLimit = 100" in usd_source
    assert 'def Cylinder "handle"' in usd_source


def test_service_door_has_robot_sized_clearance_and_a_realistic_lever_handle() -> None:
    root = ET.parse(DOOR_URDF_PATH).getroot()
    frame = root.find("./link[@name='frame']")
    panel = root.find("./link[@name='panel']")
    assert frame is not None and panel is not None

    hinge_jamb = frame.find("./collision[@name='hinge_jamb']")
    far_jamb = frame.find("./collision[@name='far_jamb']")
    header = frame.find("./collision[@name='header']")
    slab = panel.find("./collision[@name='panel_slab']")
    lever = panel.find("./collision[@name='lever_grip']")
    rose = panel.find("./collision[@name='handle_rose']")
    neck = panel.find("./collision[@name='handle_neck']")
    assert all(item is not None for item in (hinge_jamb, far_jamb, header, slab, lever, rose, neck))

    hinge_y = float(hinge_jamb.find("origin").attrib["xyz"].split()[1])
    far_y = float(far_jamb.find("origin").attrib["xyz"].split()[1])
    hinge_width = float(hinge_jamb.find("geometry/box").attrib["size"].split()[1])
    far_width = float(far_jamb.find("geometry/box").attrib["size"].split()[1])
    clear_width_m = abs(far_y - hinge_y) - 0.5 * (hinge_width + far_width)
    header_z = float(header.find("origin").attrib["xyz"].split()[2])
    header_height = float(header.find("geometry/box").attrib["size"].split()[2])
    clear_height_m = header_z - 0.5 * header_height
    assert clear_width_m == pytest.approx(0.90)
    assert clear_height_m == pytest.approx(1.58)
    assert slab.find("geometry/box").attrib["size"] == "0.055 0.90 1.55"

    lever_cylinder = lever.find("geometry/cylinder")
    assert float(lever_cylinder.attrib["radius"]) == pytest.approx(0.018)
    assert float(lever_cylinder.attrib["length"]) == pytest.approx(0.24)
    assert lever.find("origin").attrib["rpy"] == "1.57079632679 0 0"

    usd_source = DOOR_USD_PATH.read_text()
    assert 'def Cylinder "handle_rose"' in usd_source
    assert 'def Cylinder "handle_neck"' in usd_source
    lever_start = usd_source.index('def Cylinder "handle"')
    assert 'uniform token axis = "Y"' in usd_source[lever_start : lever_start + 400]


def test_door_success_angle_corresponds_to_a_traversable_projected_gap() -> None:
    source = SERVICE_CFG_PATH.read_text()
    assert "DOOR_OPEN_THRESHOLD_RAD = 1.3089969390" in source
    assert 'door_clear_width_m = 0.90' in source
    assert 'door_clear_height_m = 1.58' in source
    assert 'door_required_robot_clearance_m = 0.55' in source
    assert 'hinge reaches 75 degrees' in source
    assert 0.90 * (1.0 - math.cos(math.radians(75.0))) > 0.55


def test_service_tasks_reuse_frozen_wbc_cartesian_runtime_and_existing_pick_geometry() -> None:
    source = SERVICE_CFG_PATH.read_text()
    assert "class Go2D1OpenDoorLlmEnvCfg(Go2D1PressSwitchLlmEnvCfg)" in source
    assert "class Go2D1PickPlaceLlmEnvCfg(Go2D1PressSwitchLlmEnvCfg)" in source
    assert "manipulation_cfg._can_object_cfg(" in source
    assert "manipulation_cfg._kinematic_box_cfg(" in source
    assert 'filter_prim_paths_expr=["{ENV_REGEX_NS}/Object"]' in source
    assert "self.terminations.door_opened = DoneTerm(" in source
    assert "self.terminations.object_placed = DoneTerm(" in source
    assert "cfg.scene.switch = None" in source
    registry = REGISTRY_PATH.read_text()
    assert "Gurukul-Isaac-LLM-Go2-D1-Open-Door-v0" in registry
    assert "Gurukul-Isaac-LLM-Go2-D1-Pick-Place-v0" in registry


def test_service_policy_accepts_only_named_semantic_skills() -> None:
    program = POLICY_MODULE.parse_service_program(
        """{
          "summary":"retrieve the can",
          "steps":[
            {"op":"pick_object","target":"can","timeout_s":20},
            {"op":"place_object","target":"tray","timeout_s":20},
            {"op":"stop"}
          ]
        }""",
        "pick-place",
    )
    assert [step.op for step in program.steps] == ["pick_object", "place_object", "stop"]
    with pytest.raises(ValueError, match="not allowed"):
        POLICY_MODULE.parse_service_program('{"steps":[{"op":"exec","code":"robot.move()"}]}', "pick-place")
    with pytest.raises(ValueError, match="target must be 'can'"):
        POLICY_MODULE.parse_service_program(
            '{"steps":[{"op":"pick_object","target":"door","timeout_s":20}]}',
            "pick-place",
        )


def test_deterministic_programs_isolate_physical_skills_before_llm_use() -> None:
    assert [step.op for step in POLICY_MODULE.default_service_program("open-door").steps] == ["open_door", "stop"]
    assert [step.op for step in POLICY_MODULE.default_service_program("pick").steps] == ["pick_object", "stop"]
    assert [step.op for step in POLICY_MODULE.default_service_program("pick-place").steps] == [
        "pick_object",
        "place_object",
        "stop",
    ]
    prompt = POLICY_MODULE.build_service_prompt(
        "Pick and place the can.",
        {"object": {"name": "can"}, "tray": {"name": "tray"}},
        "pick-place",
    )
    assert "no RGB pixels are included" in prompt
    assert "Never emit Python" in prompt
    assert '"pick_object"' in prompt
    assert '"place_object"' in prompt


def test_receding_horizon_keeps_initial_plan_but_executes_one_validated_skill() -> None:
    initial_plan = POLICY_MODULE.default_service_program("pick-place")
    first_skill = POLICY_MODULE.single_skill_program(initial_plan)
    assert [step.op for step in initial_plan.steps] == ["pick_object", "place_object", "stop"]
    assert [step.op for step in first_skill.steps] == ["pick_object"]

    decision = POLICY_MODULE.parse_service_decision(
        '{"summary":"place next","steps":[{"op":"place_object","target":"tray","timeout_s":20}]}',
        "pick-place",
    )
    assert [step.op for step in decision.steps] == ["place_object"]
    with pytest.raises(ValueError, match="exactly one semantic skill"):
        POLICY_MODULE.parse_service_decision(
            """{
              "steps":[
                {"op":"pick_object","target":"can","timeout_s":20},
                {"op":"place_object","target":"tray","timeout_s":20}
              ]
            }""",
            "pick-place",
        )


def test_receding_horizon_prompt_contains_plan_history_and_latest_physical_state() -> None:
    prompt = POLICY_MODULE.build_service_replan_prompt(
        "Pick and place the can.",
        {"object": {"lifted_once": True, "bilateral_contact": True}, "tray": {"name": "tray"}},
        "pick-place",
        POLICY_MODULE.default_service_program("pick-place"),
        [{"skill": "pick_object", "outcome": "completed", "physical_task_success": False}],
    )
    assert "Initial semantic horizon" in prompt
    assert "Completed/failed skill history" in prompt
    assert "Select exactly one next semantic skill" in prompt
    assert '"op": "place_object"' in prompt
    assert '"lifted_once": true' in prompt

    reactive_prompt = POLICY_MODULE.build_service_reactive_prompt(
        "Pick and place the can.",
        {"object": {"lifted_once": False}, "tray": {"name": "tray"}},
        "pick-place",
        [],
    )
    assert "There is no persistent initial plan" in reactive_prompt
    assert "Initial semantic horizon" not in reactive_prompt
    assert "Choose exactly one next named skill" in reactive_prompt

    primitive_prompt = PRIMITIVE_MODULE.build_primitive_replan_prompt(
        "Open the door.",
        {"door": {"opened": False}},
        "open-door",
        None,
        [],
    )
    assert '"steps": [' in primitive_prompt
    assert '"frame": "root"' in primitive_prompt
    assert "`base` is not valid" in primitive_prompt
    assert '"op": "drive_base_to"' in primitive_prompt


def test_primitive_llm_policy_exposes_only_real_velocity_ik_and_gripper_controls() -> None:
    program = PRIMITIVE_MODULE.parse_primitive_program(
        """{
          "summary":"walk, reach, and close",
          "steps":[
            {"op":"base_velocity","vx_mps":0.4,"vy_mps":0.0,"yaw_rps":0.0,
             "body_pitch_rad":-0.1,"body_height_m":0.33,"duration_s":0.5},
            {"op":"move_ee","frame":"root","position_m":[0.45,0.0,0.4],
             "wrist_roll_rad":0.0,"body_pitch_rad":-0.1,"body_height_m":0.33,
             "tolerance_m":0.03,"timeout_s":4.0},
            {"op":"set_gripper","closed_fraction":1,"duration_s":0.8},
            {"op":"stop"}
          ]
        }"""
    )
    assert [step.op for step in program.steps] == ["base_velocity", "move_ee", "set_gripper", "stop"]
    assert program.steps[0].body_pitch_rad == pytest.approx(-0.1)
    assert program.steps[2].gripper_closed_fraction == 1.0

    with pytest.raises(ValueError, match="real allowed control primitive"):
        PRIMITIVE_MODULE.parse_primitive_program('{"steps":[{"op":"open_door"}]}')
    with pytest.raises(ValueError, match="magnitude >= 0.4"):
        PRIMITIVE_MODULE.parse_primitive_program(
            '{"steps":[{"op":"base_velocity","vx_mps":0.1,"vy_mps":0,"yaw_rps":0,'
            '"body_pitch_rad":0,"body_height_m":0.33,"duration_s":0.5}]}'
        )
    with pytest.raises(ValueError, match=r"exactly 0 \(open\) or 1 \(closed\)"):
        PRIMITIVE_MODULE.parse_primitive_program(
            '{"steps":[{"op":"set_gripper","closed_fraction":0.5,"duration_s":0.5}]}'
        )


def test_primitive_executor_maps_physical_pitch_velocity_and_zero_to_one_gripper_contract() -> None:
    cfg = SimpleNamespace(
        velocity_scale=(1.5, 0.5, 1.5),
        body_pitch_range=(-0.16, 0.12),
        body_height_range=(0.28, 0.39),
        wrist_roll_range=(-1.57, 1.57),
        ee_pos_range=((0.10, 0.56), (-0.40, 0.40), (0.18, 0.65)),
    )
    action_term = SimpleNamespace(cfg=cfg)
    actions = torch.zeros((1, 10))
    velocity_program = PRIMITIVE_MODULE.PrimitiveProgram(
        summary="physical velocity and pitch",
        steps=(
            PRIMITIVE_MODULE.PrimitiveStep(
                op="base_velocity",
                vx_mps=0.4,
                body_pitch_rad=-0.10,
                body_height_m=0.34,
                duration_s=0.5,
            ),
        ),
    )
    executor = PRIMITIVE_MODULE.PrimitiveExecutor(velocity_program)
    executor.apply(actions, action_term, {}, 0.02)
    assert actions[0, 0].item() == pytest.approx(0.4 / 1.5)
    assert actions[0, 3].item() == pytest.approx(2.0 * (-0.10 + 0.16) / 0.28 - 1.0)

    hold_state = {
        "robot": {
            "root_pose_world": {
                "position_m": [0.0, 0.0, 0.33],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            }
        }
    }
    for fraction, expected_internal in ((0.0, -1.0), (1.0, 1.0)):
        gripper_program = PRIMITIVE_MODULE.PrimitiveProgram(
            summary="separate gripper",
            steps=(
                PRIMITIVE_MODULE.PrimitiveStep(
                    op="set_gripper",
                    gripper_closed_fraction=fraction,
                    duration_s=0.5,
                ),
            ),
        )
        PRIMITIVE_MODULE.PrimitiveExecutor(gripper_program).apply(actions, action_term, hold_state, 0.02)
        assert actions[0, 9].item() == expected_internal


def test_pending_llm_station_keep_uses_trackable_forward_velocity() -> None:
    action_term = SimpleNamespace(cfg=SimpleNamespace(velocity_scale=(1.5, 0.5, 1.5)))
    actions = torch.zeros((1, 10))
    state = {
        "robot": {
            "root_pose_world": {
                "position_m": [-0.20, 0.0, 0.33],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            }
        }
    }
    command = PRIMITIVE_MODULE.apply_base_station_keep(actions, action_term, state, (0.0, 0.0, 0.33))
    assert command == pytest.approx((0.40, 0.0, 0.0))
    assert actions[0, 0].item() == pytest.approx(0.40 / 1.5)


def test_drive_base_to_closes_velocity_loop_over_live_world_to_root_distance() -> None:
    program = PRIMITIVE_MODULE.parse_primitive_program(
        """{
          "steps":[{
            "op":"drive_base_to","frame":"world","position_m":[0.85,0.0,0.65],
            "standoff_m":0.42,"max_vx_mps":0.75,"max_vy_mps":0.25,"max_yaw_rps":0.5,
            "body_pitch_rad":0.0,"body_height_m":0.33,"tolerance_m":0.05,"timeout_s":8.0
          }]
        }"""
    )
    action_term = SimpleNamespace(
        cfg=SimpleNamespace(
            velocity_scale=(1.5, 0.5, 1.5),
            body_pitch_range=(-0.16, 0.12),
            body_height_range=(0.28, 0.39),
        )
    )
    state = {
        "robot": {
            "root_pose_world": {
                "position_m": [0.0, 0.0, 0.33],
                "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
            }
        }
    }
    actions = torch.zeros((1, 10))
    PRIMITIVE_MODULE.PrimitiveExecutor(program).apply(actions, action_term, state, 0.02)
    assert actions[0, 0].item() == pytest.approx(0.645 / 1.5)
    assert actions[0, 1].item() == 0.0


def test_invalid_visible_primitive_response_is_retained_for_one_repair(monkeypatch: pytest.MonkeyPatch) -> None:
    invalid = '{"summary":"walk","steps":[{"op":"stop"}'
    monkeypatch.setattr(PRIMITIVE_MODULE, "request_text_completion", lambda *args: (invalid, {"total_tokens": 7}))
    with pytest.raises(PRIMITIVE_MODULE.PrimitiveResponseValidationError) as error:
        PRIMITIVE_MODULE.request_primitive_program("prompt", "open-door", "model")
    assert error.value.raw_response == invalid
    assert error.value.usage == {"total_tokens": 7}
    repair = PRIMITIVE_MODULE.build_primitive_repair_prompt(invalid, str(error.value), single_step=False)
    assert "Return only one corrected JSON object" in repair
    assert invalid in repair


def test_primitive_prompt_does_not_pretend_door_and_pick_are_available_skills() -> None:
    prompt = PRIMITIVE_MODULE.build_primitive_prompt(
        "Open the door.",
        {"door": {"grasp_to_handle_distance_m": 0.4}},
        "open-door",
    )
    assert "There is NO `open_door`, `pick_object`, or `place_object` skill" in prompt
    assert "magnitude >= 0.40 m/s" in prompt
    assert "clamp" in prompt
    assert "`closed_fraction=0` is fully open and `1` is fully closed" in prompt
    assert "body pitch" in prompt
    assert "safe travel pose" in prompt
    assert "[0.34, 0.0, 0.60]" in prompt


def test_primitive_planning_context_advises_the_llm_without_overriding_its_choice() -> None:
    state = {
        "controller": {
            "planning_context": {
                "recommended_op": "drive_base_to",
                "advisory_only": True,
                "reason": "target is outside the live IK workspace",
            }
        }
    }
    prompt = PRIMITIVE_MODULE.build_primitive_replan_prompt("Open the door.", state, "open-door", None, [])
    assert "you own the final choice" in prompt
    assert "advisory, not an action override" in prompt
    assert '"recommended_op": "drive_base_to"' in prompt
    assert "hard geometric/safety constraint" not in prompt

    repair = PRIMITIVE_MODULE.build_primitive_repair_prompt(
        '{"summary":"open","steps":[{"op":"set_gripper"}]}',
        "set_gripper is missing required fields",
        single_step=True,
        planning_context=state["controller"]["planning_context"],
    )
    assert "does not require a particular `op`" in repair
    assert "must equal `required_op`" not in repair


def test_place_fails_safe_without_a_previously_verified_lift() -> None:
    program = POLICY_MODULE.ServiceProgram(
        summary="invalid standalone placement",
        steps=(POLICY_MODULE.ServiceStep(op="place_object", target="tray", timeout_s=10.0),),
    )
    executor = POLICY_MODULE.ServiceSkillExecutor(program)
    actions = torch.zeros((1, 10))
    transition = executor.apply(
        actions,
        action_term=None,
        state={
            "object": {"lifted_once": False},
            "tray": {
                "center_world_m": [0.53, 0.15, 0.64],
                "placement_center_world_m": [0.53, 0.15, 0.66],
            },
        },
        step_dt=0.02,
    )
    assert transition == "[SERVICE][SAFE STOP] place requires a previously verified lifted grasp"
    assert executor.failed
    assert torch.all(actions[0, :3] == 0.0)


def test_runner_uses_live_frames_contacts_and_strong_success_contracts() -> None:
    source = RUNNER_PATH.read_text()
    assert "world_point_to_root(handle_center_w" in source
    assert "rotate_vector_wxyz(panel_quaternion_w, cfg.door_handle_local_position_m)" in source
    assert "rotate_vector_wxyz(panel_quaternion_w, cfg.door_push_axis_local)" in source
    assert 'env.unwrapped.scene.sensors["left_gripper_object_contact_forces"]' in source
    assert 'env.unwrapped.scene.sensors["right_gripper_object_contact_forces"]' in source
    assert 'memory["lifted_once"] |= bilateral_contact and lifted' in source
    assert 'obj["lifted_once"] and obj["in_tray"] and not obj["any_finger_contact"]' in source
    assert "env_cfg.terminations.door_opened = None" in source
    assert "env_cfg.terminations.object_placed = None" in source
    assert "[SUCCESS]" in source
    assert 'choices=("one-shot", "reactive", "receding-horizon")' in source
    assert 'request_kind = "initial_horizon"' in source
    assert 'request_kind = "replan"' in source
    assert 'request_kind = "reactive_decision"' in source
    assert "build_service_reactive_prompt(" in source
    assert "build_service_replan_prompt(" in source
    assert "execution_history.append(history_entry)" in source
    assert 'request_budget = 1 if args_cli.llm_control_mode == "one-shot"' in source
    assert 'choices=("primitive", "semantic")' in source
    assert 'default="primitive"' in source
    assert "default=12" in source
    assert "build_primitive_prompt(" in source
    assert "PrimitiveExecutor(program)" in source
    assert "apply_base_station_keep(actions, action_term, state, waiting_anchor_world_m)" in source
    assert "build_primitive_repair_prompt(" in source
    assert '"planning_context": {' in source
    assert '"advisory_only": True' in source
    assert "control gate requires op" not in source
    assert "selected_op != required_op" not in source
    assert '"gripper_command_closed_fraction"' in source
    assert "Initial grasp-to-target clearance" in source
    assert "Trace HTML:" in source
    assert "Event log:" in source
    assert "if args_cli.dashboard:" in source


def test_service_tasks_remove_base_velocity_slew_below_the_minimum_command() -> None:
    source = SERVICE_CFG_PATH.read_text()
    assert "def _configure_service_velocity_response(" in source
    assert "rate_limits[:3] = [20.0, 20.0, 20.0]" in source
