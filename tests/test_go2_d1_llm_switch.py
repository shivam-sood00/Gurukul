"""Static contracts for the first Go2+D1 language-conditioned task."""

from __future__ import annotations

import ast
import importlib.util
import json
import math
import sys
import threading
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
TASK_ROOT = REPO_ROOT / "source/Gurukul/Gurukul/tasks/manager_based/llm"
TASK_CFG_PATH = TASK_ROOT / "go2_d1_switch_env_cfg.py"
REGISTRY_PATH = TASK_ROOT / "__init__.py"
MDP_PATH = TASK_ROOT / "mdp.py"
SWITCH_URDF_PATH = REPO_ROOT / "source/Gurukul/data/Objects/llm_switch/spring_button.urdf"
SWITCH_USD_PATH = REPO_ROOT / "source/Gurukul/data/Objects/llm_switch/spring_button.usda"
RUNNER_PATH = REPO_ROOT / "scripts/llm/run_go2_d1_switch.py"
CODE_POLICY_PATH = REPO_ROOT / "scripts/llm/code_policy.py"
DASHBOARD_PATH = REPO_ROOT / "scripts/llm/live_dashboard.py"
ACTIONS_PATH = (
    REPO_ROOT / "source/Gurukul/Gurukul/tasks/manager_based/locomotion/velocity/mdp/actions.py"
)

CODE_POLICY_SPEC = importlib.util.spec_from_file_location("go2_d1_code_policy_test_module", CODE_POLICY_PATH)
assert CODE_POLICY_SPEC is not None and CODE_POLICY_SPEC.loader is not None
CODE_POLICY_MODULE = importlib.util.module_from_spec(CODE_POLICY_SPEC)
sys.modules[CODE_POLICY_SPEC.name] = CODE_POLICY_MODULE
CODE_POLICY_SPEC.loader.exec_module(CODE_POLICY_MODULE)

DASHBOARD_SPEC = importlib.util.spec_from_file_location("go2_d1_dashboard_test_module", DASHBOARD_PATH)
assert DASHBOARD_SPEC is not None and DASHBOARD_SPEC.loader is not None
DASHBOARD_MODULE = importlib.util.module_from_spec(DASHBOARD_SPEC)
DASHBOARD_SPEC.loader.exec_module(DASHBOARD_MODULE)


def test_llm_switch_sources_parse() -> None:
    for path in (TASK_CFG_PATH, REGISTRY_PATH, MDP_PATH, RUNNER_PATH, CODE_POLICY_PATH, DASHBOARD_PATH, ACTIONS_PATH):
        ast.parse(path.read_text())


def test_switch_asset_has_a_spring_return_prismatic_joint() -> None:
    root = ET.parse(SWITCH_URDF_PATH).getroot()
    joint = root.find("./joint[@name='button_joint']")
    assert joint is not None
    assert joint.attrib["type"] == "prismatic"
    assert joint.find("axis").attrib["xyz"] == "1 0 0"
    limit = joint.find("limit")
    assert float(limit.attrib["lower"]) == 0.0
    assert float(limit.attrib["upper"]) == 0.025
    usd_source = SWITCH_USD_PATH.read_text()
    assert 'def PhysicsPrismaticJoint "button_joint"' in usd_source
    assert "float physics:upperLimit = 0.025" in usd_source


def test_task_reuses_the_arm_disturbance_leg_policy_contract() -> None:
    source = TASK_CFG_PATH.read_text()
    assert "class Go2D1PressSwitchLlmEnvCfg(controller_cfg.UnitreeGo2D1LegWbcAsyncArmFlatEnvCfg)" in source
    assert "controller_cfg.configure_go2_d1_leg_wbc_ee_hierarchical_runtime(self)" in source
    assert "self.actions.wbc_command.policy_path = self.wbc_policy_path" in source
    assert '"vx",' in source
    assert '"grasp_x",' in source
    assert '"wrist_roll",' in source
    assert '"gripper",' in source


def test_task_exposes_robot_rgb_without_changing_frozen_policy_observations() -> None:
    source = TASK_CFG_PATH.read_text()
    assert 'prim_path="{ENV_REGEX_NS}/Robot/base/front_rgb_camera"' in source
    assert 'data_types=["rgb"]' in source
    assert '"normalize": False' in source
    assert "self.observations.llm = Go2D1LlmObservationsCfg()" in source
    assert 'llm_observation_group = "llm"' in source
    assert "self.concatenate_terms = False" in source
    assert "self.sim.use_fabric = True" in source


def test_task_registers_distinct_runnable_environment_and_success_condition() -> None:
    registry = REGISTRY_PATH.read_text()
    source = TASK_CFG_PATH.read_text()
    assert "Gurukul-Isaac-LLM-Go2-D1-Press-Switch-v0" in registry
    assert "Go2D1PressSwitchLlmEnvCfg" in registry
    assert "self.terminations.switch_pressed = DoneTerm(" in source
    assert "self.rewards.switch_press_progress = RewTerm(" in source


def test_interactive_task_starts_ready_without_training_auto_resets() -> None:
    source = TASK_CFG_PATH.read_text()
    assert "hierarchical_cfg._configure_cartesian_pick_ready_reset(self)" in source
    assert "controller_cfg.GO2_D1_WBC_WORKSPACE_READY_POSE" in source
    assert "self.commands.base_velocity.debug_vis = False" in source
    assert "self.terminations.time_out = None" in source
    assert "self.terminations.bad_orientation = None" in source
    assert 'self.events.randomize_reset_joints.params["position_range"] = (0.0, 0.0)' in source
    assert 'self.events.randomize_reset_joints.params["velocity_range"] = (0.0, 0.0)' in source


def test_runner_requires_the_exported_base_policy_and_enables_cameras() -> None:
    source = RUNNER_PATH.read_text()
    assert '"--base-policy"' in source
    assert "args_cli.enable_cameras = True" in source
    assert "env_cfg.actions.wbc_command.policy_path = args_cli.base_policy" in source
    assert 'observations["llm"]["switch"]' in source
    assert 'observations["llm"]["rgb"]' in source
    assert "cv2.imshow(RGB_WINDOW_NAME, frame)" in source
    assert "action_term.processed_actions.detach().clone()" in source
    assert "persistent_base_commands = actions[dones, 0:3].clone()" in source
    assert "actions[dones, 0:3] = persistent_base_commands" in source
    assert "action_term.low_level_actions" in source
    assert 'env.unwrapped.scene["robot"]' in source
    assert "base_z=" in source
    assert '"--base-command"' in source
    assert '"--reset-on-success"' in source
    assert '"--no-fabric"' in source
    assert "use_fabric=True" in source
    assert "env_cfg.terminations.switch_pressed = None" in source
    assert "actions[new_success, 0:3] = 0.0" in source
    assert "def _apply_rgb_base_key" in source
    assert "policy: RUN | arm: HOLD" in source
    assert "actor_cmd=" in source
    assert "body_vel=" in source
    assert "travel=" in source
    assert "rgb_delta=" in source
    assert "rgb_updates=" in source
    assert "leg_qd_rms=" in source


def test_code_policy_frame_math_uses_wxyz_and_round_trips_points() -> None:
    yaw_90_wxyz = (math.sqrt(0.5), 0.0, 0.0, math.sqrt(0.5))
    root_position_w = (1.0, 2.0, 0.5)
    point_b = (0.4, -0.2, 0.1)
    point_w = CODE_POLICY_MODULE.root_point_to_world(point_b, root_position_w, yaw_90_wxyz)
    assert point_w == pytest.approx((1.2, 2.4, 0.6))
    assert CODE_POLICY_MODULE.world_point_to_root(point_w, root_position_w, yaw_90_wxyz) == pytest.approx(point_b)
    assert CODE_POLICY_MODULE.rotate_vector_wxyz(yaw_90_wxyz, (1.0, 0.0, 0.0)) == pytest.approx((0.0, 1.0, 0.0))
    assert CODE_POLICY_MODULE.inverse_rotate_vector_wxyz(yaw_90_wxyz, (0.0, 1.0, 0.0)) == pytest.approx((1.0, 0.0, 0.0))


def test_code_policy_accepts_only_bounded_primitives_and_explicit_target_frames() -> None:
    program = CODE_POLICY_MODULE.parse_policy_program(
        """{
          "summary": "approach and press",
          "steps": [
            {"op":"approach_button", "standoff_m":0.50, "timeout_s":4.0},
            {"op":"move_grasp", "target":{"frame":"world", "position_m":[0.77,0.0,0.62]},
             "tolerance_m":0.025, "timeout_s":5.0, "wrist_roll_rad":0.0, "gripper":"open"},
            {"op":"press_button", "overtravel_m":0.006, "timeout_s":4.0},
            {"op":"stop"}
          ]
        }"""
    )
    assert [step.op for step in program.steps] == ["approach_button", "move_grasp", "press_button", "stop"]
    assert program.steps[1].target_frame == "world"
    with pytest.raises(ValueError, match="op must be one of"):
        CODE_POLICY_MODULE.parse_policy_program('{"steps":[{"op":"exec", "code":"robot.move()"}]}')
    with pytest.raises(ValueError, match="frame must be 'world' or 'root'"):
        CODE_POLICY_MODULE.parse_policy_program(
            '{"steps":[{"op":"move_grasp", "target":{"frame":"camera", "position_m":[0,0,0]}, '
            '"tolerance_m":0.02, "timeout_s":2.0}]}'
        )


def test_text_policy_prompt_is_symbolic_and_runner_transforms_world_targets_each_tick() -> None:
    prompt = CODE_POLICY_MODULE.build_policy_prompt(
        "Press the button.",
        {"robot": {"grasp_midpoint": {"root_position_m": [0.4, 0.0, 0.3]}}},
    )
    assert "grasp_midpoint" in prompt
    assert "rgb" not in prompt.lower()
    runner_source = RUNNER_PATH.read_text()
    assert "world_point_to_root(target_w, root_position_w, root_quaternion_w)" in runner_source
    assert "robot.data.body_pos_w[env_index].index_select(0, action_term._grasp_body_ids).mean(dim=0)" in runner_source
    assert "press_axis_w = rotate_vector_wxyz(button_quaternion_w, cfg.switch_press_axis_local)" in runner_source
    assert "Only symbolic state is sent; RGB pixels are not included." in runner_source


def test_floating_base_cartesian_ik_offsets_physx_jacobian_joint_columns() -> None:
    source = ACTIONS_PATH.read_text()
    assert "self._ee_jacobi_joint_ids = self._arm_joint_ids" in source
    assert "self._ee_jacobi_joint_ids = self._arm_joint_ids + 6" in source
    assert "index=self._ee_jacobi_joint_ids" in source


def test_provider_token_usage_is_normalized_when_available() -> None:
    assert CODE_POLICY_MODULE._response_usage(
        {"usage": {"prompt_tokens": 1200, "completion_tokens": 180, "total_tokens": 1380}}
    ) == {"prompt_tokens": 1200, "completion_tokens": 180, "total_tokens": 1380}
    assert CODE_POLICY_MODULE._response_usage({}) == {}


def test_live_dashboard_writes_refreshing_html_and_append_only_jsonl(tmp_path: Path) -> None:
    dashboard = DASHBOARD_MODULE.LiveDashboard(
        tmp_path,
        model="test/model",
        provider="test-provider",
        backend="hosted test",
        refresh_s=0.25,
        input_price_per_million=0.01,
        output_price_per_million=0.03,
    )
    dashboard.event(
        "request_submitted",
        prompt="Return <strict JSON>",
        observation={"button": {"pressed": False}},
    )
    dashboard.update(
        status="step 1/2: approach_button",
        task_id="test-switch-task",
        scenario="press-switch",
        instruction="Press the red button.",
        control_mode="one-shot",
        prompt="Return <strict JSON>",
        request_observation={"button": {"pressed": False}},
        raw_response='{"steps":[{"op":"stop"}]}',
        program={"steps": [{"op": "stop"}]},
        usage={"prompt_tokens": 1000, "completion_tokens": 100, "total_tokens": 1100},
    )
    dashboard.sample(
        observation={"robot": {"root_pose_world": [0.0, 0.0, 0.4]}},
        telemetry={"step": 10, "policy_status": "step 1/2: approach_button"},
    )
    dashboard.close("program stopped")

    html_source = dashboard.html_path.read_text()
    assert "Pause refresh" in html_source
    assert "window.setTimeout" in html_source
    assert "}, 250);" in html_source
    assert "Return &lt;strict JSON&gt;" in html_source
    assert "Visible model response" in html_source
    assert "Current live symbolic observation" in html_source
    assert "Go2+D1 Mission Console" in html_source
    assert "LLM mode" in html_source
    assert "One Shot" in html_source
    assert "Press the red button." in html_source
    assert "Validated execution plan" in html_source
    assert "Physical task progress" in html_source
    assert "Execution timeline" in html_source
    assert 'data-tab="calls"' in html_source
    assert "Robot state" in html_source
    assert "Raw event stream" in html_source
    assert 'id="event-search"' in html_source
    assert 'data-copy-target="prompt-1"' in html_source
    assert "hidden chain-of-thought" in html_source
    assert "$0.00001300" in html_source
    events = [json.loads(line) for line in dashboard.jsonl_path.read_text().splitlines()]
    assert [event["type"] for event in events] == [
        "run_started",
        "request_submitted",
        "control_sample",
        "run_finished",
    ]


def test_live_dashboard_preserves_each_llm_request_as_a_separate_trace(tmp_path: Path) -> None:
    dashboard = DASHBOARD_MODULE.LiveDashboard(
        tmp_path,
        model="test/model",
        provider="test-provider",
        backend="local test",
    )
    dashboard.event(
        "request_submitted",
        request_count=1,
        request_kind="initial_horizon",
        prompt="FIRST UNIQUE PROMPT",
        observation={"phase": "initial"},
    )
    dashboard.event(
        "request_completed",
        request_count=1,
        latency_s=0.5,
        raw_response="FIRST UNIQUE RESPONSE",
        program={"steps": [{"op": "pick_object"}]},
        usage={"prompt_tokens": 10, "completion_tokens": 4, "total_tokens": 14},
    )
    dashboard.event(
        "request_submitted",
        request_count=2,
        request_kind="replan",
        prompt="SECOND UNIQUE PROMPT",
        observation={"phase": "after-pick"},
    )
    dashboard.update(
        raw_response="SECOND UNIQUE RESPONSE",
        program={"steps": [{"op": "place_object"}]},
        latency_s=0.25,
    )
    dashboard.close()

    html_source = dashboard.html_path.read_text()
    assert "Request #1" in html_source
    assert "Request #2" in html_source
    assert "FIRST UNIQUE PROMPT" in html_source
    assert "FIRST UNIQUE RESPONSE" in html_source
    assert "SECOND UNIQUE PROMPT" in html_source
    assert "SECOND UNIQUE RESPONSE" in html_source
    assert html_source.index("Request #2") < html_source.index("Request #1")


def test_live_dashboard_publishes_on_loopback_only(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    class FakeHttpServer:
        def __init__(self, address: tuple[str, int], handler: object) -> None:
            assert address == ("127.0.0.1", 8765)
            self.server_address = address
            self.daemon_threads = False
            self.stopped = threading.Event()

        def serve_forever(self) -> None:
            self.stopped.wait(timeout=1.0)

        def shutdown(self) -> None:
            self.stopped.set()

        def server_close(self) -> None:
            return

    monkeypatch.setattr(DASHBOARD_MODULE, "ThreadingHTTPServer", FakeHttpServer)
    dashboard = DASHBOARD_MODULE.LiveDashboard(
        tmp_path,
        model="test/model",
        provider="test-provider",
        backend="hosted test",
    )
    assert dashboard.start_local_server(8765) == "http://127.0.0.1:8765/"
    dashboard.close()

    events = [json.loads(line) for line in dashboard.jsonl_path.read_text().splitlines()]
    assert events[1] == {
        "timestamp_utc": events[1]["timestamp_utc"],
        "type": "dashboard_server_started",
        "bind_host": "127.0.0.1",
        "port": 8765,
        "url": "http://127.0.0.1:8765/",
    }
    runner_source = RUNNER_PATH.read_text()
    assert "dashboard.start_local_server(args_cli.llm_dashboard_port)" in runner_source


def test_live_dashboard_serves_the_mission_console_over_real_loopback_http(tmp_path: Path) -> None:
    dashboard = DASHBOARD_MODULE.LiveDashboard(
        tmp_path,
        model="test/model",
        provider="test-provider",
        backend="local test",
    )
    try:
        url = dashboard.start_local_server(0)
        with urllib.request.urlopen(url, timeout=2.0) as response:  # noqa: S310 - loopback URL created above
            assert response.status == 200
            html_source = response.read().decode("utf-8")
        assert "Go2+D1 Mission Console" in html_source
        assert "LLM calls" in html_source
        assert "Raw event stream" in html_source
    finally:
        dashboard.close()
