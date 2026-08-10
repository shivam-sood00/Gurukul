"""Validated primitive-level LLM control for Go2+D1 service tasks."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

from code_policy import request_text_completion, world_point_to_root

PRIMITIVE_OPS = ("base_velocity", "drive_base_to", "move_ee", "set_gripper", "hold", "stop")
MAX_PRIMITIVE_STEPS = 12
MIN_FORWARD_SPEED_MPS = 0.40


@dataclass(frozen=True)
class PrimitiveStep:
    """One bounded command over a control interface that exists in the environment."""

    op: str
    duration_s: float = 0.0
    vx_mps: float = 0.0
    vy_mps: float = 0.0
    yaw_rps: float = 0.0
    max_vx_mps: float = 0.0
    max_vy_mps: float = 0.0
    max_yaw_rps: float = 0.0
    standoff_m: float = 0.0
    body_pitch_rad: float = 0.0
    body_height_m: float = 0.33
    frame: str | None = None
    position_m: tuple[float, float, float] | None = None
    wrist_roll_rad: float = 0.0
    tolerance_m: float = 0.03
    timeout_s: float = 0.0
    gripper_closed_fraction: float | None = None


@dataclass(frozen=True)
class PrimitiveProgram:
    """A bounded horizon composed only from real velocity, IK, and gripper interfaces."""

    summary: str
    steps: tuple[PrimitiveStep, ...]


@dataclass(frozen=True)
class PrimitiveRequestResult:
    program: PrimitiveProgram
    raw_response: str
    usage: dict[str, int]


class PrimitiveResponseValidationError(ValueError):
    """Retain an invalid visible response so it can be logged and repaired."""

    def __init__(self, message: str, raw_response: str, usage: dict[str, int]) -> None:
        super().__init__(message)
        self.raw_response = raw_response
        self.usage = usage


def _finite(value: Any, name: str, lower: float, upper: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number.")
    result = float(value)
    if not math.isfinite(result) or not lower <= result <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}].")
    return result


def _keys(data: dict[str, Any], allowed: set[str], required: set[str], name: str) -> None:
    missing = required - data.keys()
    unknown = data.keys() - allowed
    if missing:
        raise ValueError(f"{name} is missing required fields: {sorted(missing)}.")
    if unknown:
        raise ValueError(f"{name} contains unsupported fields: {sorted(unknown)}.")


def _json_object(text: str) -> dict[str, Any]:
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


def _velocity_with_dead_zone(value: Any, name: str, maximum: float, minimum: float) -> float:
    result = _finite(value, name, -maximum, maximum)
    if result != 0.0 and abs(result) < minimum:
        raise ValueError(f"{name} must be zero or have magnitude >= {minimum}.")
    return result


def parse_primitive_program(text: str) -> PrimitiveProgram:
    """Parse a primitive horizon and reject skills, code, and untrackable small velocities."""
    data = _json_object(text)
    _keys(data, {"summary", "steps"}, {"steps"}, "program")
    summary = data.get("summary", "")
    if not isinstance(summary, str) or len(summary) > 300:
        raise ValueError("program.summary must be a string no longer than 300 characters.")
    raw_steps = data["steps"]
    if not isinstance(raw_steps, list) or not 1 <= len(raw_steps) <= MAX_PRIMITIVE_STEPS:
        raise ValueError(f"program.steps must contain 1 to {MAX_PRIMITIVE_STEPS} steps.")

    steps: list[PrimitiveStep] = []
    for index, raw in enumerate(raw_steps):
        if not isinstance(raw, dict):
            raise ValueError(f"step[{index}] must be an object.")
        op = raw.get("op")
        if op not in PRIMITIVE_OPS:
            raise ValueError(f"step[{index}].op {op!r} is not a real allowed control primitive.")
        name = f"step[{index}]"
        if op == "base_velocity":
            fields = {"op", "vx_mps", "vy_mps", "yaw_rps", "body_pitch_rad", "body_height_m", "duration_s"}
            _keys(raw, fields, fields, name)
            steps.append(
                PrimitiveStep(
                    op=op,
                    vx_mps=_velocity_with_dead_zone(
                        raw["vx_mps"], f"{name}.vx_mps", 0.75, MIN_FORWARD_SPEED_MPS
                    ),
                    vy_mps=_velocity_with_dead_zone(raw["vy_mps"], f"{name}.vy_mps", 0.35, 0.15),
                    yaw_rps=_velocity_with_dead_zone(raw["yaw_rps"], f"{name}.yaw_rps", 0.80, 0.20),
                    body_pitch_rad=_finite(raw["body_pitch_rad"], f"{name}.body_pitch_rad", -0.16, 0.12),
                    body_height_m=_finite(raw["body_height_m"], f"{name}.body_height_m", 0.28, 0.39),
                    duration_s=_finite(raw["duration_s"], f"{name}.duration_s", 0.20, 2.0),
                )
            )
            continue
        if op == "drive_base_to":
            fields = {
                "op",
                "frame",
                "position_m",
                "standoff_m",
                "max_vx_mps",
                "max_vy_mps",
                "max_yaw_rps",
                "body_pitch_rad",
                "body_height_m",
                "tolerance_m",
                "timeout_s",
            }
            _keys(raw, fields, fields, name)
            if raw["frame"] != "world":
                raise ValueError(f"{name}.frame must be 'world' for a persistent base target.")
            position = raw["position_m"]
            if not isinstance(position, list) or len(position) != 3:
                raise ValueError(f"{name}.position_m must contain exactly three numbers.")
            steps.append(
                PrimitiveStep(
                    op=op,
                    frame="world",
                    position_m=tuple(
                        _finite(position[axis], f"{name}.position_m[{axis}]", -2.0, 2.0) for axis in range(3)
                    ),
                    standoff_m=_finite(raw["standoff_m"], f"{name}.standoff_m", 0.30, 0.65),
                    max_vx_mps=_finite(
                        raw["max_vx_mps"], f"{name}.max_vx_mps", MIN_FORWARD_SPEED_MPS, 0.75
                    ),
                    max_vy_mps=_finite(raw["max_vy_mps"], f"{name}.max_vy_mps", 0.15, 0.35),
                    max_yaw_rps=_finite(raw["max_yaw_rps"], f"{name}.max_yaw_rps", 0.20, 0.80),
                    body_pitch_rad=_finite(raw["body_pitch_rad"], f"{name}.body_pitch_rad", -0.16, 0.12),
                    body_height_m=_finite(raw["body_height_m"], f"{name}.body_height_m", 0.28, 0.39),
                    tolerance_m=_finite(raw["tolerance_m"], f"{name}.tolerance_m", 0.03, 0.10),
                    timeout_s=_finite(raw["timeout_s"], f"{name}.timeout_s", 1.0, 12.0),
                )
            )
            continue
        if op == "move_ee":
            fields = {
                "op",
                "frame",
                "position_m",
                "wrist_roll_rad",
                "body_pitch_rad",
                "body_height_m",
                "tolerance_m",
                "timeout_s",
            }
            _keys(raw, fields, fields, name)
            frame = raw["frame"]
            if frame not in ("world", "root"):
                raise ValueError(f"{name}.frame must be 'world' or 'root'.")
            position = raw["position_m"]
            if not isinstance(position, list) or len(position) != 3:
                raise ValueError(f"{name}.position_m must contain exactly three numbers.")
            if frame == "root":
                bounds = ((0.10, 0.56), (-0.40, 0.40), (0.18, 0.65))
            else:
                bounds = ((-2.0, 2.0), (-2.0, 2.0), (0.10, 1.50))
            position_m = tuple(
                _finite(position[axis], f"{name}.position_m[{axis}]", *bounds[axis]) for axis in range(3)
            )
            steps.append(
                PrimitiveStep(
                    op=op,
                    frame=frame,
                    position_m=position_m,
                    wrist_roll_rad=_finite(raw["wrist_roll_rad"], f"{name}.wrist_roll_rad", -1.57, 1.57),
                    body_pitch_rad=_finite(raw["body_pitch_rad"], f"{name}.body_pitch_rad", -0.16, 0.12),
                    body_height_m=_finite(raw["body_height_m"], f"{name}.body_height_m", 0.28, 0.39),
                    tolerance_m=_finite(raw["tolerance_m"], f"{name}.tolerance_m", 0.015, 0.08),
                    timeout_s=_finite(raw["timeout_s"], f"{name}.timeout_s", 0.50, 8.0),
                )
            )
            continue
        if op == "set_gripper":
            _keys(raw, {"op", "closed_fraction", "duration_s"}, {"op", "closed_fraction", "duration_s"}, name)
            fraction = _finite(raw["closed_fraction"], f"{name}.closed_fraction", 0.0, 1.0)
            if fraction not in (0.0, 1.0):
                raise ValueError(f"{name}.closed_fraction must be exactly 0 (open) or 1 (closed).")
            steps.append(
                PrimitiveStep(
                    op=op,
                    gripper_closed_fraction=fraction,
                    duration_s=_finite(raw["duration_s"], f"{name}.duration_s", 0.20, 3.0),
                )
            )
            continue
        if op == "hold":
            _keys(raw, {"op", "duration_s"}, {"op", "duration_s"}, name)
            steps.append(PrimitiveStep(op=op, duration_s=_finite(raw["duration_s"], f"{name}.duration_s", 0.10, 3.0)))
            continue
        _keys(raw, {"op"}, {"op"}, name)
        steps.append(PrimitiveStep(op="stop"))
    return PrimitiveProgram(summary=summary.strip(), steps=tuple(steps))


def parse_primitive_decision(text: str) -> PrimitiveProgram:
    program = parse_primitive_program(text)
    if len(program.steps) != 1:
        raise ValueError("A reactive primitive decision must contain exactly one step.")
    return program


def primitive_program_payload(program: PrimitiveProgram) -> dict[str, Any]:
    payload_steps: list[dict[str, Any]] = []
    for step in program.steps:
        if step.op == "base_velocity":
            payload_steps.append(
                {
                    "op": step.op,
                    "vx_mps": step.vx_mps,
                    "vy_mps": step.vy_mps,
                    "yaw_rps": step.yaw_rps,
                    "body_pitch_rad": step.body_pitch_rad,
                    "body_height_m": step.body_height_m,
                    "duration_s": step.duration_s,
                }
            )
        elif step.op == "drive_base_to":
            payload_steps.append(
                {
                    "op": step.op,
                    "frame": step.frame,
                    "position_m": list(step.position_m or ()),
                    "standoff_m": step.standoff_m,
                    "max_vx_mps": step.max_vx_mps,
                    "max_vy_mps": step.max_vy_mps,
                    "max_yaw_rps": step.max_yaw_rps,
                    "body_pitch_rad": step.body_pitch_rad,
                    "body_height_m": step.body_height_m,
                    "tolerance_m": step.tolerance_m,
                    "timeout_s": step.timeout_s,
                }
            )
        elif step.op == "move_ee":
            payload_steps.append(
                {
                    "op": step.op,
                    "frame": step.frame,
                    "position_m": list(step.position_m or ()),
                    "wrist_roll_rad": step.wrist_roll_rad,
                    "body_pitch_rad": step.body_pitch_rad,
                    "body_height_m": step.body_height_m,
                    "tolerance_m": step.tolerance_m,
                    "timeout_s": step.timeout_s,
                }
            )
        elif step.op == "set_gripper":
            payload_steps.append(
                {"op": step.op, "closed_fraction": step.gripper_closed_fraction, "duration_s": step.duration_s}
            )
        elif step.op == "hold":
            payload_steps.append({"op": step.op, "duration_s": step.duration_s})
        else:
            payload_steps.append({"op": "stop"})
    return {"summary": program.summary, "steps": payload_steps}


def single_primitive_program(program: PrimitiveProgram) -> PrimitiveProgram:
    if not program.steps:
        raise ValueError("Cannot execute an empty primitive program.")
    return PrimitiveProgram(summary=program.summary, steps=(program.steps[0],))


def _interface_contract() -> str:
    return f"""Real control interface:
- `base_velocity`: root-frame m/s and rad/s for 0.20-2.0 s, plus body pitch and height. Nonzero `vx_mps` must have
  magnitude >= {MIN_FORWARD_SPEED_MPS:.2f} m/s because smaller forward commands are not reliably tracked.
  Choose its magnitude from the measured forward distance error: zero inside the stopping tolerance, otherwise clamp
  a proportional command to [{MIN_FORWARD_SPEED_MPS:.2f}, 0.75] m/s rather than always using the minimum.
- `drive_base_to`: closed-loop Go2 velocity control toward a fixed world target and requested standoff. At 50 Hz it
  recomputes root-frame distance/heading, applies zero inside tolerance, and otherwise uses proportional commands with
  the {MIN_FORWARD_SPEED_MPS:.2f} m/s forward floor. This is a generic velocity controller, not a door/pick skill.
- `move_ee`: absolute finger-midpoint Cartesian IK target in `world` or current `root` frame. The executor transforms
  world targets into the live root frame at 50 Hz and rejects targets outside the current D1 workspace.
- `set_gripper`: separate binary gripper interface; `closed_fraction=0` is fully open and `1` is fully closed.
- `hold`: hold current posture, IK, and gripper targets with zero base velocity.
- `stop`: safe zero-base stop.
There is NO `open_door`, `pick_object`, or `place_object` skill. Those are success conditions that must emerge from
your primitive sequence. Never emit Python, joint positions, torques, or any operation outside this interface."""


def build_primitive_prompt(instruction: str, state: dict[str, Any], scenario: str) -> str:
    planning_context = state.get("controller", {}).get("planning_context", "none available")
    schema = {
        "summary": "short plan description",
        "steps": [
            {
                "op": "base_velocity",
                "vx_mps": 0.4,
                "vy_mps": 0.0,
                "yaw_rps": 0.0,
                "body_pitch_rad": 0.0,
                "body_height_m": 0.33,
                "duration_s": 0.5,
            },
            {
                "op": "move_ee",
                "frame": "world",
                "position_m": [0.8, 0.0, 0.65],
                "wrist_roll_rad": 0.0,
                "body_pitch_rad": 0.0,
                "body_height_m": 0.33,
                "tolerance_m": 0.03,
                "timeout_s": 4.0,
            },
            {
                "op": "drive_base_to",
                "frame": "world",
                "position_m": [0.85, 0.0, 0.65],
                "standoff_m": 0.42,
                "max_vx_mps": 0.75,
                "max_vy_mps": 0.25,
                "max_yaw_rps": 0.5,
                "body_pitch_rad": 0.0,
                "body_height_m": 0.33,
                "tolerance_m": 0.05,
                "timeout_s": 8.0,
            },
            {"op": "set_gripper", "closed_fraction": 1, "duration_s": 0.8},
            {"op": "hold", "duration_s": 0.5},
            {"op": "stop"},
        ],
    }
    return f"""You are planning bounded real control primitives for a Go2 quadruped with a D1 arm.
Return exactly one JSON object and no Markdown or explanatory text.

Task: {instruction}
Scenario: {scenario}

{_interface_contract()}

Frame and safety facts:
- `world` is fixed. `root` is the live Go2 body frame: +x forward, +y left, +z up. Quaternions are wxyz.
- The object/door starts away from the reset gripper; initial contact is not task progress.
- Before walking near scene geometry, consider putting the open gripper at the root-frame safe travel pose
  [0.34, 0.0, 0.60] m. `base_velocity` and `drive_base_to` preserve the most recent arm and gripper commands.
- Use measured distances and root-frame targets. Walk until a world IK goal lies inside the reported root workspace.
- The LLM controls pitch explicitly in both locomotion and IK primitives.
- Keep horizons short enough to inspect and replan. Success is checked only from physical hinge, contact, lift, and
  placement state.
- `controller.planning_context` is advisory geometry computed from the latest measurements. Its `recommended_op` and
  suggested parameters identify a likely feasible next action, but you remain responsible for choosing every
  primitive and value. If you deviate, briefly state the measured reason in `summary`.

Planning guidance:
- Check `inside_live_ik_workspace` before choosing `move_ee`. An out-of-workspace target will be rejected rather than
  silently changed. Prefer `drive_base_to` when base motion can make that target reachable.
- Prefer a fixed world target and closed-loop `drive_base_to` over guessing a timed velocity duration for approach.
- Use the latest execution failure and distance errors to correct the next choice; do not repeat an unreachable IK
  request without first changing the geometry.

Interaction facts (these are task geometry, not hidden actions):
- Door: keep the gripper open, move to the reported precontact point, then move past the live handle along the reported
  world-frame push direction. The hinge angle is the only opening signal.
- Pick: open before approaching, align the finger midpoint with the live can center, close only at the can, then command
  a world target at least the reported required lift above the live can while remaining closed. Bilateral finger
  contact plus measured lift is the only pick signal.

Allowed JSON shapes (use only the steps needed; at most {MAX_PRIMITIVE_STEPS}):
{json.dumps(schema, indent=2)}

Current symbolic state (no RGB pixels):
{json.dumps(state, indent=2, sort_keys=True)}

Advisory planning context (repeated for salience; it does not override your choice):
{json.dumps(planning_context, indent=2, sort_keys=True)}
"""


def build_primitive_replan_prompt(
    instruction: str,
    state: dict[str, Any],
    scenario: str,
    initial_plan: PrimitiveProgram | None,
    execution_history: list[dict[str, Any]],
) -> str:
    plan_context = primitive_program_payload(initial_plan) if initial_plan is not None else "none (reactive mode)"
    planning_context = state.get("controller", {}).get("planning_context", "none available")
    valid_examples = [
        {
            "summary": "Drive toward a fixed world target until the requested standoff.",
            "steps": [
                {
                    "op": "drive_base_to",
                    "frame": "world",
                    "position_m": [0.85, 0.0, 0.65],
                    "standoff_m": 0.42,
                    "max_vx_mps": 0.75,
                    "max_vy_mps": 0.25,
                    "max_yaw_rps": 0.5,
                    "body_pitch_rad": 0.0,
                    "body_height_m": 0.33,
                    "tolerance_m": 0.05,
                    "timeout_s": 8.0,
                }
            ],
        },
        {
            "summary": "Correct planar distance using a trackable command.",
            "steps": [
                {
                    "op": "base_velocity",
                    "vx_mps": 0.4,
                    "vy_mps": 0.0,
                    "yaw_rps": 0.0,
                    "body_pitch_rad": 0.0,
                    "body_height_m": 0.33,
                    "duration_s": 0.5,
                }
            ],
        },
        {
            "summary": "Put the open arm in its safe travel pose.",
            "steps": [
                {
                    "op": "move_ee",
                    "frame": "root",
                    "position_m": [0.34, 0.0, 0.60],
                    "wrist_roll_rad": 0.0,
                    "body_pitch_rad": 0.0,
                    "body_height_m": 0.33,
                    "tolerance_m": 0.06,
                    "timeout_s": 4.0,
                }
            ],
        },
        {"summary": "Open the gripper.", "steps": [{"op": "set_gripper", "closed_fraction": 0, "duration_s": 0.5}]},
        {"summary": "Hold the current targets.", "steps": [{"op": "hold", "duration_s": 0.5}]},
        {"summary": "Stop safely.", "steps": [{"op": "stop"}]},
    ]
    return f"""You are choosing the next real control primitive for a Go2 quadruped with a D1 arm.
Return exactly one JSON object with exactly one step and no Markdown.

Task: {instruction}
Scenario: {scenario}

{_interface_contract()}

Use the latest measured state, not assumptions from the initial horizon. If the previous primitive failed, choose a
safe corrective primitive. Use `stop` when the physical task is complete or no safe progress is possible.
The latest state's `controller.planning_context` is advisory, not an action override. Its `recommended_op`, feasibility
facts, and suggested parameters are intended to help you choose a command that will execute successfully. You still
choose the operation, velocity, pitch, IK target, and gripper command. If you deviate from the recommendation, explain
the measured reason briefly in `summary`.

Before choosing, check whether the intended world IK target is inside the reported live root workspace. If it is not,
`move_ee` will be rejected; `drive_base_to` can change the geometry using a fixed world target and distance-aware stop.
Use execution failures as feedback and do not repeat an unchanged unreachable target.

Choose one operation and copy its exact JSON envelope and field names from these syntax examples. These are format
examples, not a sequence to execute. For `move_ee`, `frame` must be exactly `world` or `root`; `base` is not valid.
{json.dumps(valid_examples, indent=2)}

Initial primitive horizon:
{json.dumps(plan_context, indent=2)}

Primitive execution history:
{json.dumps(execution_history, indent=2, sort_keys=True)}

Latest symbolic state (no RGB pixels):
{json.dumps(state, indent=2, sort_keys=True)}

Advisory planning context (repeated for salience; you own the final choice):
{json.dumps(planning_context, indent=2, sort_keys=True)}
"""


def build_primitive_repair_prompt(
    raw_response: str,
    validation_error: str,
    *,
    single_step: bool,
    planning_context: dict[str, Any] | None = None,
) -> str:
    """Ask for one syntax/schema repair without changing the prior control intent."""
    step_limit = "exactly one step" if single_step else f"1 to {MAX_PRIMITIVE_STEPS} steps"
    context_text = json.dumps(planning_context, indent=2) if planning_context else "none"
    return f"""Your previous robot-control response failed strict validation:
{validation_error}

Return only one corrected JSON object with `summary` and `steps` containing {step_limit}. Do not add Markdown or
explanations. Preserve the safe intent and numeric values where possible. Every step must use exactly one of these
schemas:
- base_velocity: op, vx_mps, vy_mps, yaw_rps, body_pitch_rad, body_height_m, duration_s
- drive_base_to: op, frame, position_m, standoff_m, max_vx_mps, max_vy_mps, max_yaw_rps, body_pitch_rad,
  body_height_m, tolerance_m, timeout_s
- move_ee: op, frame, position_m, wrist_roll_rad, body_pitch_rad, body_height_m, tolerance_m, timeout_s
- set_gripper: op, closed_fraction, duration_s
- hold: op, duration_s
- stop: op
Do not emit door, pick, or place skills. Gripper closure must be exactly 0 or 1. Forward velocity must be zero inside
the stopping tolerance; otherwise choose it from distance error with magnitude >= {MIN_FORWARD_SPEED_MPS:.2f} m/s.
For `move_ee`, `frame` is exactly `world` or `root` (never `base`), body_height_m is in [0.28, 0.39], tolerance_m is
in [0.015, 0.08], and timeout_s is in [0.5, 8.0]. A complete valid move example is:
{{"summary":"Use safe travel pose","steps":[{{"op":"move_ee","frame":"root","position_m":[0.34,0.0,0.60],
"wrist_roll_rad":0.0,"body_pitch_rad":0.0,"body_height_m":0.33,"tolerance_m":0.06,"timeout_s":4.0}}]}}

Current advisory planning context:
{context_text}
Use this context to repair unsafe numeric or frame mistakes while preserving the model's control intent. It recommends
a feasible action but does not require a particular `op`. Omit diagnostic fields such as `reason`, `feasibility`, and
`forward_error_m` because they are not action-schema fields.

Previous visible response:
{raw_response[:8000]}
"""


def request_primitive_program(
    prompt: str,
    scenario: str,
    model: str,
    provider: str = "auto",
    base_url: str | None = None,
    timeout_s: float = 60.0,
) -> PrimitiveRequestResult:
    del scenario
    response, usage = request_text_completion(prompt, model, provider, base_url, timeout_s)
    try:
        program = parse_primitive_program(response)
    except ValueError as exc:
        raise PrimitiveResponseValidationError(str(exc), response, usage) from exc
    return PrimitiveRequestResult(program, response, usage)


def request_primitive_decision(
    prompt: str,
    scenario: str,
    model: str,
    provider: str = "auto",
    base_url: str | None = None,
    timeout_s: float = 60.0,
) -> PrimitiveRequestResult:
    del scenario
    response, usage = request_text_completion(prompt, model, provider, base_url, timeout_s)
    try:
        program = parse_primitive_decision(response)
    except ValueError as exc:
        raise PrimitiveResponseValidationError(str(exc), response, usage) from exc
    return PrimitiveRequestResult(program, response, usage)


def _clamp(value: float, lower: float, upper: float) -> float:
    return min(max(float(value), lower), upper)


def _normalize(value: float, lower: float, upper: float) -> float:
    return _clamp(2.0 * (float(value) - float(lower)) / (float(upper) - float(lower)) - 1.0, -1.0, 1.0)


def apply_base_station_keep(
    actions: Any,
    action_term: Any,
    state: dict[str, Any],
    anchor_world_m: tuple[float, float, float],
    env_index: int = 0,
) -> tuple[float, float, float]:
    """Hold the request-time planar position while an asynchronous LLM call is pending."""
    root = state["robot"]["root_pose_world"]
    anchor_root = world_point_to_root(anchor_world_m, root["position_m"], root["quaternion_wxyz"])

    def correction(error: float, minimum: float, maximum: float) -> float:
        if abs(float(error)) <= 0.06:
            return 0.0
        return math.copysign(_clamp(1.5 * abs(float(error)), minimum, maximum), float(error))

    command = (
        correction(float(anchor_root[0]), MIN_FORWARD_SPEED_MPS, 0.60),
        correction(float(anchor_root[1]), 0.15, 0.25),
        0.0,
    )
    for axis, value in enumerate(command):
        actions[env_index, axis] = _clamp(value / float(action_term.cfg.velocity_scale[axis]), -1.0, 1.0)
    return command


class PrimitiveExecutor:
    """Execute validated velocity, posture, Cartesian IK, and gripper primitives at control rate."""

    def __init__(self, program: PrimitiveProgram, env_index: int = 0) -> None:
        self.program = program
        self.env_index = env_index
        self.step_index = 0
        self.step_elapsed_s = 0.0
        self.finished = False
        self.failed = False
        self.message = "primitive horizon ready"
        self._target_dwell_s = 0.0
        self._station_anchor_world_m: tuple[float, float, float] | None = None

    @property
    def status(self) -> str:
        if self.finished or self.failed:
            return self.message
        return f"step {self.step_index + 1}/{len(self.program.steps)}: {self.program.steps[self.step_index].op}"

    def _advance(self, detail: str) -> str:
        op = self.program.steps[self.step_index].op
        self.step_index += 1
        self.step_elapsed_s = 0.0
        self._target_dwell_s = 0.0
        self._station_anchor_world_m = None
        if self.step_index >= len(self.program.steps):
            self.finished = True
            self.message = "primitive horizon complete"
        return f"[PRIMITIVE] completed {op}: {detail}"

    def _fail(self, actions: Any, detail: str) -> str:
        actions[self.env_index, 0:3] = 0.0
        self.failed = True
        self.message = f"primitive horizon stopped: {detail}"
        return f"[PRIMITIVE][SAFE STOP] {detail}"

    def _set_posture(self, actions: Any, action_term: Any, step: PrimitiveStep) -> None:
        actions[self.env_index, 3] = _normalize(step.body_pitch_rad, *action_term.cfg.body_pitch_range)
        actions[self.env_index, 4] = _normalize(step.body_height_m, *action_term.cfg.body_height_range)

    def apply(self, actions: Any, action_term: Any, state: dict[str, Any], step_dt: float) -> str | None:
        if self.finished or self.failed:
            actions[self.env_index, 0:3] = 0.0
            return None
        step = self.program.steps[self.step_index]
        self.step_elapsed_s += float(step_dt)
        if step.op == "stop":
            actions[self.env_index, 0:3] = 0.0
            self.finished = True
            self.message = "primitive stop reached"
            return "[PRIMITIVE] stop reached"
        if step.op == "base_velocity":
            self._set_posture(actions, action_term, step)
            for axis, value in enumerate((step.vx_mps, step.vy_mps, step.yaw_rps)):
                actions[self.env_index, axis] = _clamp(value / float(action_term.cfg.velocity_scale[axis]), -1.0, 1.0)
            if self.step_elapsed_s >= step.duration_s:
                actions[self.env_index, 0:3] = 0.0
                return self._advance(f"held commanded body velocity for {step.duration_s:.2f}s")
            return None
        if step.op == "drive_base_to":
            assert step.position_m is not None
            self._set_posture(actions, action_term, step)
            root = state["robot"]["root_pose_world"]
            target_root = world_point_to_root(step.position_m, root["position_m"], root["quaternion_wxyz"])
            x_error = float(target_root[0]) - step.standoff_m
            y_error = float(target_root[1])
            heading_error = math.atan2(float(target_root[1]), max(float(target_root[0]), 1.0e-6))

            def distance_command(error: float, gain: float, minimum: float, maximum: float, tolerance: float) -> float:
                if abs(error) <= tolerance:
                    return 0.0
                return math.copysign(_clamp(gain * abs(error), minimum, maximum), error)

            command = (
                distance_command(x_error, 1.5, MIN_FORWARD_SPEED_MPS, step.max_vx_mps, step.tolerance_m),
                distance_command(y_error, 1.0, 0.15, step.max_vy_mps, step.tolerance_m),
                distance_command(heading_error, 1.0, 0.20, step.max_yaw_rps, 0.10),
            )
            for axis, value in enumerate(command):
                actions[self.env_index, axis] = _clamp(
                    value / float(action_term.cfg.velocity_scale[axis]), -1.0, 1.0
                )
            if abs(x_error) <= step.tolerance_m and abs(y_error) <= step.tolerance_m and abs(heading_error) <= 0.12:
                self._target_dwell_s += float(step_dt)
                actions[self.env_index, 0:3] = 0.0
            else:
                self._target_dwell_s = 0.0
            if self._target_dwell_s >= 0.10:
                return self._advance(
                    f"base reached standoff with forward_error={x_error:.3f}m lateral_error={y_error:.3f}m"
                )
            if self.step_elapsed_s >= step.timeout_s:
                return self._fail(
                    actions,
                    f"drive_base_to timed out with forward_error={x_error:.3f}m lateral_error={y_error:.3f}m",
                )
            return None
        if self._station_anchor_world_m is None:
            self._station_anchor_world_m = tuple(state["robot"]["root_pose_world"]["position_m"])
        apply_base_station_keep(actions, action_term, state, self._station_anchor_world_m, self.env_index)
        if step.op == "set_gripper":
            assert step.gripper_closed_fraction is not None
            actions[self.env_index, 9] = 2.0 * step.gripper_closed_fraction - 1.0
            if self.step_elapsed_s >= step.duration_s:
                return self._advance(f"gripper closed_fraction={step.gripper_closed_fraction:.0f}")
            return None
        if step.op == "hold":
            if self.step_elapsed_s >= step.duration_s:
                return self._advance(f"held for {step.duration_s:.2f}s")
            return None

        assert step.position_m is not None and step.frame is not None
        self._set_posture(actions, action_term, step)
        actions[self.env_index, 8] = _normalize(step.wrist_roll_rad, *action_term.cfg.wrist_roll_range)
        if step.frame == "world":
            root = state["robot"]["root_pose_world"]
            target_root = world_point_to_root(step.position_m, root["position_m"], root["quaternion_wxyz"])
            current = state["robot"]["grasp_midpoint"]["world_position_m"]
        else:
            target_root = step.position_m
            current = state["robot"]["grasp_midpoint"]["root_position_m"]
        inside = all(
            float(bounds[0]) <= float(target_root[axis]) <= float(bounds[1])
            for axis, bounds in enumerate(action_term.cfg.ee_pos_range)
        )
        if not inside:
            return self._fail(actions, f"move_ee target is outside live root workspace: {list(target_root)}")
        for axis, bounds in enumerate(action_term.cfg.ee_pos_range):
            actions[self.env_index, 5 + axis] = _normalize(float(target_root[axis]), *bounds)
        error = math.sqrt(sum((float(current[axis]) - float(step.position_m[axis])) ** 2 for axis in range(3)))
        if error <= step.tolerance_m:
            self._target_dwell_s += float(step_dt)
        else:
            self._target_dwell_s = 0.0
        if self._target_dwell_s >= 0.10:
            return self._advance(f"IK target reached with error={error:.3f}m")
        if self.step_elapsed_s >= step.timeout_s:
            return self._fail(actions, f"move_ee timed out with error={error:.3f}m")
        return None
