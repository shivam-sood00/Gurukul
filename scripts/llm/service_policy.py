"""Validated semantic skill policy for useful Go2+D1 service tasks."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from code_policy import request_text_completion, world_point_to_root

ALLOWED_SERVICE_OPS = ("open_door", "pick_object", "place_object", "stop")
SERVICE_TARGETS = {"door", "can", "tray"}
MAX_SERVICE_STEPS = 4
TRAVEL_GRASP_ROOT_M = (0.34, 0.0, 0.60)


@dataclass(frozen=True)
class ServiceStep:
    """One validated semantic service skill."""

    op: str
    target: str | None = None
    timeout_s: float = 0.0
    standoff_m: float = 0.0
    target_angle_rad: float = 0.0


@dataclass(frozen=True)
class ServiceProgram:
    """A bounded semantic program that cannot contain executable code."""

    summary: str
    steps: tuple[ServiceStep, ...]


@dataclass(frozen=True)
class ServiceRequestResult:
    """Validated service program plus visible provider telemetry."""

    program: ServiceProgram
    raw_response: str
    usage: dict[str, int]


def service_program_payload(program: ServiceProgram) -> dict[str, Any]:
    """Convert a validated program into the exact JSON representation shown to the model."""
    return {
        "summary": program.summary,
        "steps": [
            {
                **{"op": step.op},
                **({"target": step.target, "timeout_s": step.timeout_s} if step.target else {}),
                **(
                    {
                        "standoff_m": step.standoff_m,
                        "target_angle_deg": round(math.degrees(step.target_angle_rad), 3),
                    }
                    if step.op == "open_door"
                    else {}
                ),
            }
            for step in program.steps
        ],
    }


def single_skill_program(program: ServiceProgram) -> ServiceProgram:
    """Return a program containing only the first validated semantic decision."""
    if not program.steps:
        raise ValueError("Cannot execute an empty service program.")
    return ServiceProgram(summary=program.summary, steps=(program.steps[0],))


def _finite_bounded(value: Any, name: str, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number.")
    result = float(value)
    if not math.isfinite(result) or not lower <= result <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    return result


def _require_keys(data: dict[str, Any], allowed: set[str], required: set[str], name: str) -> None:
    missing = required - data.keys()
    unknown = data.keys() - allowed
    if missing:
        raise ValueError(f"{name} is missing required fields: {sorted(missing)}.")
    if unknown:
        raise ValueError(f"{name} contains unsupported fields: {sorted(unknown)}.")


def _extract_json(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].lstrip()
    try:
        value = json.loads(stripped)
    except json.JSONDecodeError:
        start, end = stripped.find("{"), stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("The model response did not contain a JSON object.") from None
        try:
            value = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"The model response contained invalid JSON: {exc}.") from exc
    if not isinstance(value, dict):
        raise ValueError("The model response must be one JSON object.")
    return value


def parse_service_program(text: str, scenario: str) -> ServiceProgram:
    """Strictly parse door/pick/place skills and reject arbitrary commands."""
    data = _extract_json(text)
    _require_keys(data, {"summary", "steps"}, {"steps"}, "program")
    summary = data.get("summary", "")
    if not isinstance(summary, str) or len(summary) > 300:
        raise ValueError("program.summary must be a string no longer than 300 characters.")
    raw_steps = data["steps"]
    if not isinstance(raw_steps, list) or not 1 <= len(raw_steps) <= MAX_SERVICE_STEPS:
        raise ValueError(f"program.steps must contain 1 to {MAX_SERVICE_STEPS} steps.")

    scenario_ops = {
        "open-door": {"open_door", "stop"},
        "pick": {"pick_object", "stop"},
        "pick-place": {"pick_object", "place_object", "stop"},
    }
    if scenario not in scenario_ops:
        raise ValueError(f"Unsupported service scenario: {scenario!r}.")

    steps: list[ServiceStep] = []
    for index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, dict):
            raise ValueError(f"step[{index}] must be an object.")
        op = raw_step.get("op")
        if op not in ALLOWED_SERVICE_OPS or op not in scenario_ops[scenario]:
            raise ValueError(f"step[{index}].op {op!r} is not allowed for scenario {scenario!r}.")
        name = f"step[{index}]"
        if op == "open_door":
            _require_keys(
                raw_step,
                {"op", "target", "standoff_m", "target_angle_deg", "timeout_s"},
                {"op", "target", "standoff_m", "target_angle_deg", "timeout_s"},
                name,
            )
            if raw_step["target"] != "door":
                raise ValueError(f"{name}.target must be 'door'.")
            steps.append(
                ServiceStep(
                    op=op,
                    target="door",
                    standoff_m=_finite_bounded(raw_step["standoff_m"], f"{name}.standoff_m", 0.38, 0.48),
                    target_angle_rad=math.radians(
                        _finite_bounded(raw_step["target_angle_deg"], f"{name}.target_angle_deg", 35.0, 65.0)
                    ),
                    timeout_s=_finite_bounded(raw_step["timeout_s"], f"{name}.timeout_s", 5.0, 35.0),
                )
            )
            continue
        if op in ("pick_object", "place_object"):
            _require_keys(raw_step, {"op", "target", "timeout_s"}, {"op", "target", "timeout_s"}, name)
            expected_target = "can" if op == "pick_object" else "tray"
            if raw_step["target"] != expected_target or raw_step["target"] not in SERVICE_TARGETS:
                raise ValueError(f"{name}.target must be {expected_target!r}.")
            steps.append(
                ServiceStep(
                    op=op,
                    target=expected_target,
                    timeout_s=_finite_bounded(raw_step["timeout_s"], f"{name}.timeout_s", 4.0, 30.0),
                )
            )
            continue
        _require_keys(raw_step, {"op"}, {"op"}, name)
        steps.append(ServiceStep(op="stop"))
    return ServiceProgram(summary=summary.strip(), steps=tuple(steps))


def parse_service_decision(text: str, scenario: str) -> ServiceProgram:
    """Parse exactly one semantic skill for an event-triggered replan."""
    program = parse_service_program(text, scenario)
    if len(program.steps) != 1:
        raise ValueError("A receding-horizon decision must contain exactly one semantic skill.")
    return program


def default_service_program(scenario: str) -> ServiceProgram:
    """Return the deterministic mechanics-isolation program for a scenario."""
    if scenario == "open-door":
        return ServiceProgram(
            summary="Approach and push the hinged door open.",
            steps=(
                ServiceStep(
                    op="open_door",
                    target="door",
                    standoff_m=0.42,
                    target_angle_rad=math.radians(45.0),
                    timeout_s=30.0,
                ),
                ServiceStep(op="stop"),
            ),
        )
    if scenario == "pick":
        return ServiceProgram(
            summary="Acquire bilateral contact and lift the can.",
            steps=(ServiceStep(op="pick_object", target="can", timeout_s=20.0), ServiceStep(op="stop")),
        )
    if scenario == "pick-place":
        return ServiceProgram(
            summary="Pick the can and release it into the tray.",
            steps=(
                ServiceStep(op="pick_object", target="can", timeout_s=20.0),
                ServiceStep(op="place_object", target="tray", timeout_s=20.0),
                ServiceStep(op="stop"),
            ),
        )
    raise ValueError(f"Unsupported service scenario: {scenario!r}.")


def build_service_prompt(instruction: str, state: dict[str, Any], scenario: str) -> str:
    """Build a symbolic-only prompt whose output is limited to named skills."""
    example = default_service_program(scenario)
    example_json = service_program_payload(example)
    return f"""You are a high-level service planner for a Go2 quadruped with a D1 arm.
Return exactly one JSON object and no Markdown or explanatory text.

Task: {instruction}
Scenario: {scenario}

Safety and frame contract:
- The model may select only the named semantic skills shown in the example.
- The deterministic controller owns all world-to-root transforms, locomotion, IK, contact checks, and timeouts.
- Never emit Python, joint commands, raw Cartesian coordinates, or unlisted target names.
- `world` is fixed; `root` is the moving Go2 base frame (+x forward, +y left, +z up).
- Door success is measured from the physical revolute joint.
- Pick success requires simultaneous filtered contact on both fingers and a real lift.
- Place success requires a prior lifted grasp, release inside the tray, and low object speed.

Return this exact schema for the requested scenario:
{json.dumps(example_json, indent=2)}

Current symbolic state (no RGB pixels are included):
{json.dumps(state, indent=2, sort_keys=True)}
"""


def build_service_replan_prompt(
    instruction: str,
    state: dict[str, Any],
    scenario: str,
    initial_plan: ServiceProgram,
    execution_history: list[dict[str, Any]],
) -> str:
    """Build one receding-horizon decision from the initial plan and latest state."""
    example = single_skill_program(default_service_program(scenario))
    if scenario == "pick-place" and state.get("object", {}).get("lifted_once", False):
        example = ServiceProgram(
            summary="Place the verified grasped can into the tray.",
            steps=(ServiceStep(op="place_object", target="tray", timeout_s=20.0),),
        )
    return f"""You are the event-triggered replanning layer for a Go2 quadruped with a D1 arm.
Return exactly one JSON object and no Markdown or explanatory text.

Task: {instruction}
Scenario: {scenario}

This is a semantic receding-horizon update, not raw motor control:
- The initial horizon is context, not an instruction to repeat a completed skill.
- Select exactly one next semantic skill based on the latest physical state and execution history.
- Use `stop` when the physical task is already complete or no safe allowed skill remains.
- The deterministic controller owns all world-to-root transforms, 50 Hz tracking, IK, contact checks, and timeouts.
- Never emit Python, joint commands, velocities, raw Cartesian coordinates, or unlisted target names.
- Door success is measured from the physical revolute joint.
- Pick success requires simultaneous filtered contact on both fingers and a real lift.
- Place requires a prior verified lift and succeeds only after release and settling in the tray.

Return exactly one step using this schema:
{json.dumps(service_program_payload(example), indent=2)}

Initial semantic horizon:
{json.dumps(service_program_payload(initial_plan), indent=2)}

Completed/failed skill history:
{json.dumps(execution_history, indent=2, sort_keys=True)}

Latest symbolic state (no RGB pixels are included):
{json.dumps(state, indent=2, sort_keys=True)}
"""


def build_service_reactive_prompt(
    instruction: str,
    state: dict[str, Any],
    scenario: str,
    execution_history: list[dict[str, Any]],
) -> str:
    """Build a memory-light one-skill decision without an initial horizon."""
    example = single_skill_program(default_service_program(scenario))
    if scenario == "pick-place" and state.get("object", {}).get("lifted_once", False):
        example = ServiceProgram(
            summary="Place the verified grasped can into the tray.",
            steps=(ServiceStep(op="place_object", target="tray", timeout_s=20.0),),
        )
    return f"""You are a reactive semantic controller for a Go2 quadruped with a D1 arm.
Return exactly one JSON object and no Markdown or explanatory text.

Task: {instruction}
Scenario: {scenario}

Choose exactly one next named skill from the latest state and execution history. There is no persistent initial plan.
Use `stop` if the task is complete or no safe skill remains. The deterministic controller owns 50 Hz tracking,
world-to-root transforms, IK, contacts, and timeouts. Never emit Python, joints, velocities, Cartesian coordinates,
or unlisted targets. Pick requires bilateral contact and physical lift; place requires a verified prior lift and
release inside the tray; door success uses the measured hinge joint.

Return exactly one step using this schema:
{json.dumps(service_program_payload(example), indent=2)}

Completed/failed skill history:
{json.dumps(execution_history, indent=2, sort_keys=True)}

Latest symbolic state (no RGB pixels are included):
{json.dumps(state, indent=2, sort_keys=True)}
"""


def request_service_program(
    prompt: str,
    scenario: str,
    model: str,
    provider: str = "auto",
    base_url: str | None = None,
    timeout_s: float = 60.0,
) -> ServiceRequestResult:
    """Request and validate one semantic service program."""
    response, usage = request_text_completion(prompt, model, provider, base_url, timeout_s)
    return ServiceRequestResult(parse_service_program(response, scenario), response, usage)


def request_service_decision(
    prompt: str,
    scenario: str,
    model: str,
    provider: str = "auto",
    base_url: str | None = None,
    timeout_s: float = 60.0,
) -> ServiceRequestResult:
    """Request and validate one event-triggered semantic skill."""
    response, usage = request_text_completion(prompt, model, provider, base_url, timeout_s)
    return ServiceRequestResult(parse_service_decision(response, scenario), response, usage)


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(value, lower), upper)


def _walk_command(error: float, gain: float, minimum: float, maximum: float, tolerance: float) -> float:
    if abs(error) <= tolerance:
        return 0.0
    return math.copysign(_clamp(abs(error) * gain, minimum, maximum), error)


def _set_base_velocity(actions: Any, action_term: Any, env_index: int, command_b: tuple[float, float, float]) -> None:
    for axis, value in enumerate(command_b):
        actions[env_index, axis] = _clamp(float(value) / float(action_term.cfg.velocity_scale[axis]), -1.0, 1.0)


def _set_grasp_target(
    actions: Any,
    action_term: Any,
    env_index: int,
    target_b: tuple[float, float, float],
    *,
    closed: bool,
) -> None:
    for axis, (lower, upper) in enumerate(action_term.cfg.ee_pos_range):
        normalized = 2.0 * (float(target_b[axis]) - float(lower)) / (float(upper) - float(lower)) - 1.0
        actions[env_index, 5 + axis] = _clamp(normalized, -1.0, 1.0)
    wrist_lower, wrist_upper = action_term.cfg.wrist_roll_range
    actions[env_index, 8] = _clamp(
        2.0 * (0.0 - float(wrist_lower)) / (float(wrist_upper) - float(wrist_lower)) - 1.0,
        -1.0,
        1.0,
    )
    actions[env_index, 9] = 1.0 if closed else -1.0


def _distance(a: Any, b: Any) -> float:
    return math.sqrt(sum((float(a[axis]) - float(b[axis])) ** 2 for axis in range(3)))


class ServiceSkillExecutor:
    """Closed-loop executor for door, bilateral pick, and release-aware place."""

    def __init__(self, program: ServiceProgram, env_index: int = 0) -> None:
        self.program = program
        self.env_index = env_index
        self.step_index = 0
        self.step_elapsed_s = 0.0
        self.phase = "start"
        self.phase_elapsed_s = 0.0
        self.finished = False
        self.failed = False
        self.message = "program ready"
        self._waypoint_w: tuple[float, float, float] | None = None
        self._contact_dwell_s = 0.0

    @property
    def status(self) -> str:
        if self.failed or self.finished:
            return self.message
        step = self.program.steps[self.step_index]
        return f"step {self.step_index + 1}/{len(self.program.steps)}: {step.op}/{self.phase}"

    def _change_phase(self, phase: str) -> str:
        previous = self.phase
        self.phase = phase
        self.phase_elapsed_s = 0.0
        return f"[SERVICE] {self.program.steps[self.step_index].op}: {previous} -> {phase}"

    def _advance(self, detail: str) -> str:
        op = self.program.steps[self.step_index].op
        self.step_index += 1
        self.step_elapsed_s = 0.0
        self.phase_elapsed_s = 0.0
        self.phase = "start"
        self._waypoint_w = None
        self._contact_dwell_s = 0.0
        if self.step_index >= len(self.program.steps):
            self.finished = True
            self.message = "program complete"
        return f"[SERVICE] completed {op}: {detail}"

    def _fail(self, actions: Any, detail: str) -> str:
        actions[self.env_index, 0:3] = 0.0
        self.failed = True
        self.message = f"program stopped: {detail}"
        return f"[SERVICE][SAFE STOP] {detail}"

    def _target_world(
        self,
        actions: Any,
        action_term: Any,
        state: dict[str, Any],
        target_w: tuple[float, float, float],
        *,
        closed: bool,
    ) -> float:
        root = state["robot"]["root_pose_world"]
        target_b = world_point_to_root(target_w, root["position_m"], root["quaternion_wxyz"])
        _set_grasp_target(actions, action_term, self.env_index, target_b, closed=closed)
        return _distance(state["robot"]["grasp_midpoint"]["world_position_m"], target_w)

    def apply(self, actions: Any, action_term: Any, state: dict[str, Any], step_dt: float) -> str | None:
        """Apply one control tick and report phase/skill transitions."""
        if self.finished or self.failed:
            actions[self.env_index, 0:3] = 0.0
            return None
        step = self.program.steps[self.step_index]
        self.step_elapsed_s += float(step_dt)
        self.phase_elapsed_s += float(step_dt)
        if step.timeout_s and self.step_elapsed_s >= step.timeout_s:
            return self._fail(actions, f"{step.op} timed out in phase {self.phase}")
        if step.op == "stop":
            actions[self.env_index, 0:3] = 0.0
            self.finished = True
            self.message = "program stopped"
            return "[SERVICE] stop primitive reached"
        if step.op == "open_door":
            return self._open_door(actions, action_term, state, step)
        if step.op == "pick_object":
            return self._pick(actions, action_term, state, step_dt)
        return self._place(actions, action_term, state)

    def _open_door(
        self,
        actions: Any,
        action_term: Any,
        state: dict[str, Any],
        step: ServiceStep,
    ) -> str | None:
        door = state["door"]
        handle_b = door["handle_center"]["root_position_m"]
        if self.phase == "start":
            precontact_b = door["precontact"]["root_position_m"]
            inside_workspace = all(
                float(bounds[0]) + 0.01 <= float(precontact_b[axis]) <= float(bounds[1]) - 0.01
                for axis, bounds in enumerate(action_term.cfg.ee_pos_range)
            )
            self.phase = "precontact" if inside_workspace else "approach"
        if self.phase == "approach":
            # Keep the deployed arm above the interaction geometry while the
            # locomotion policy brings the base into manipulation range.
            _set_grasp_target(
                actions,
                action_term,
                self.env_index,
                TRAVEL_GRASP_ROOT_M,
                closed=False,
            )
            x_error = float(handle_b[0]) - step.standoff_m
            y_error = float(handle_b[1])
            heading_error = math.atan2(float(handle_b[1]), max(float(handle_b[0]), 1.0e-6))
            _set_base_velocity(
                actions,
                action_term,
                self.env_index,
                (
                    _walk_command(x_error, 1.7, 0.40, 0.55, 0.035),
                    _walk_command(y_error, 1.0, 0.10, 0.22, 0.035),
                    _walk_command(heading_error, 1.1, 0.16, 0.45, 0.08),
                ),
            )
            if abs(x_error) <= 0.045 and abs(y_error) <= 0.06 and abs(heading_error) <= 0.12:
                actions[self.env_index, 0:3] = 0.0
                return self._change_phase("precontact")
            return None
        if self.phase == "precontact":
            actions[self.env_index, 0:3] = 0.0
            target_w = tuple(door["precontact"]["world_position_m"])
            if float(door["hinge_angle_rad"]) >= math.radians(2.0):
                return self._change_phase("push")
            if self._target_world(actions, action_term, state, target_w, closed=False) <= 0.025:
                return self._change_phase("push")
            return None

        push_direction = door["push_direction"]["world_unit_vector"]
        handle_w = door["handle_center"]["world_position_m"]
        # At the 0.42 m base standoff, a 0.12 m follow-through remains inside
        # the 0.56 m root-frame IK boundary while crossing the full handle.
        target_w = tuple(float(handle_w[axis]) + 0.12 * float(push_direction[axis]) for axis in range(3))
        self._target_world(actions, action_term, state, target_w, closed=False)
        # Arm contact alone can push the light quadruped away from the hinge.
        # Follow the live handle with the base while the IK target supplies the
        # local push, keeping the target inside the root-frame workspace.
        push_x_error = float(handle_b[0]) - step.standoff_m
        push_y_error = float(handle_b[1])
        _set_base_velocity(
            actions,
            action_term,
            self.env_index,
            (
                _walk_command(push_x_error, 1.6, 0.40, 0.55, 0.025),
                _walk_command(push_y_error, 1.0, 0.10, 0.20, 0.04),
                0.0,
            ),
        )
        if float(door["hinge_angle_rad"]) >= step.target_angle_rad:
            actions[self.env_index, 0:3] = 0.0
            return self._advance(f"hinge angle={float(door['hinge_angle_rad']):.3f} rad")
        return None

    def _pick(self, actions: Any, action_term: Any, state: dict[str, Any], step_dt: float) -> str | None:
        obj = state["object"]
        object_w = tuple(obj["world_position_m"])
        object_b = obj["root_position_m"]
        if self.phase == "start":
            directly_reachable = (
                float(action_term.cfg.ee_pos_range[0][0]) + 0.10
                <= float(object_b[0])
                <= float(action_term.cfg.ee_pos_range[0][1])
                and abs(float(object_b[1])) <= 0.30
                and float(action_term.cfg.ee_pos_range[2][0])
                <= float(object_b[2])
                <= float(action_term.cfg.ee_pos_range[2][1])
            )
            if directly_reachable:
                self._waypoint_w = tuple(state["robot"]["grasp_midpoint"]["world_position_m"])
                self.phase = "close"
            else:
                self.phase = "approach"
        if self.phase == "approach":
            _set_grasp_target(
                actions,
                action_term,
                self.env_index,
                TRAVEL_GRASP_ROOT_M,
                closed=False,
            )
            x_error = float(object_b[0]) - 0.48
            y_error = float(object_b[1])
            heading_error = math.atan2(float(object_b[1]), max(float(object_b[0]), 1.0e-6))
            _set_base_velocity(
                actions,
                action_term,
                self.env_index,
                (
                    _walk_command(x_error, 1.5, 0.40, 0.55, 0.035),
                    _walk_command(y_error, 1.0, 0.08, 0.18, 0.035),
                    _walk_command(heading_error, 1.0, 0.12, 0.35, 0.08),
                ),
            )
            if abs(x_error) <= 0.035 and abs(y_error) <= 0.035 and abs(heading_error) <= 0.08:
                actions[self.env_index, 0:3] = 0.0
                return self._change_phase("pregrasp")
            return None
        actions[self.env_index, 0:3] = 0.0
        if self.phase == "pregrasp":
            root_w = state["robot"]["root_pose_world"]["position_m"]
            dx, dy = float(object_w[0]) - float(root_w[0]), float(object_w[1]) - float(root_w[1])
            norm = max(math.hypot(dx, dy), 1.0e-6)
            target_w = (object_w[0] - 0.09 * dx / norm, object_w[1] - 0.09 * dy / norm, object_w[2])
            if self._target_world(actions, action_term, state, target_w, closed=False) <= 0.025:
                return self._change_phase("engage")
            return None
        if self.phase == "engage":
            engage_error = self._target_world(actions, action_term, state, object_w, closed=False)
            if engage_error <= 0.045 or self.phase_elapsed_s >= 0.8:
                return self._change_phase("close")
            return None
        if self.phase == "close":
            close_target_w = self._waypoint_w or object_w
            self._target_world(actions, action_term, state, close_target_w, closed=True)
            if obj["bilateral_contact"]:
                self._contact_dwell_s += float(step_dt)
            else:
                self._contact_dwell_s = 0.0
            if self._contact_dwell_s >= 0.02:
                root_w = state["robot"]["root_pose_world"]["position_m"]
                dx = float(object_w[0]) - float(root_w[0])
                dy = float(object_w[1]) - float(root_w[1])
                planar_distance = max(math.hypot(dx, dy), 1.0e-6)
                self._waypoint_w = (
                    object_w[0] - 0.05 * dx / planar_distance,
                    object_w[1] - 0.05 * dy / planar_distance,
                    object_w[2] + 0.10,
                )
                return self._change_phase("lift")
            if self.phase_elapsed_s >= 2.5:
                return self._fail(actions, "pick did not acquire bilateral finger contact")
            return None
        assert self._waypoint_w is not None
        self._target_world(actions, action_term, state, self._waypoint_w, closed=True)
        if obj["lifted"] and obj["bilateral_contact"]:
            return self._advance("bilateral grasp and physical lift verified")
        return None

    def _place(self, actions: Any, action_term: Any, state: dict[str, Any]) -> str | None:
        obj = state["object"]
        tray = state["tray"]
        actions[self.env_index, 0:3] = 0.0
        if self.phase == "start":
            if not obj["lifted_once"]:
                return self._fail(actions, "place requires a previously verified lifted grasp")
            self._waypoint_w = (
                float(tray["center_world_m"][0]),
                float(tray["center_world_m"][1]),
                float(tray["placement_center_world_m"][2]) + 0.14,
            )
            self.phase = "above_tray"
        assert self._waypoint_w is not None
        if self.phase == "above_tray":
            if self._target_world(actions, action_term, state, self._waypoint_w, closed=True) <= 0.03:
                self._waypoint_w = tuple(tray["placement_center_world_m"])
                return self._change_phase("lower")
            return None
        if self.phase == "lower":
            if self._target_world(actions, action_term, state, self._waypoint_w, closed=True) <= 0.025:
                return self._change_phase("release")
            return None
        if self.phase == "release":
            self._target_world(actions, action_term, state, self._waypoint_w, closed=False)
            if self.phase_elapsed_s >= 0.8:
                return self._change_phase("settle")
            return None
        self._target_world(actions, action_term, state, self._waypoint_w, closed=False)
        if obj["in_tray"] and not obj["any_finger_contact"] and float(obj["speed_mps"]) <= 0.20:
            return self._advance("object released and settled in tray after verified lift")
        if self.phase_elapsed_s >= 3.0:
            return self._fail(actions, "released object did not settle inside the tray")
        return None
