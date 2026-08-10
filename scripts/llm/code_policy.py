"""Text-only code-policy helpers for the Go2+D1 language task.

The language model emits a small JSON program composed from validated robot
primitives.  This module intentionally has no Isaac Sim dependency so parsing,
frame math, and HTTP behavior can be tested without launching the simulator.
"""

from __future__ import annotations

import json
import math
import urllib.parse
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

WORLD_FRAME = "world"
ROOT_FRAME = "root"
ALLOWED_OPS = ("approach_button", "move_grasp", "press_button", "hold", "stop")
MAX_PROGRAM_STEPS = 8


@dataclass(frozen=True)
class PolicyStep:
    """One validated high-level robot primitive."""

    op: str
    timeout_s: float = 0.0
    standoff_m: float = 0.0
    target_frame: str | None = None
    target_position_m: tuple[float, float, float] | None = None
    tolerance_m: float = 0.025
    overtravel_m: float = 0.006
    duration_s: float = 0.0
    wrist_roll_rad: float = 0.0
    gripper: str = "open"


@dataclass(frozen=True)
class PolicyProgram:
    """A bounded sequence returned by the language model."""

    summary: str
    steps: tuple[PolicyStep, ...]


@dataclass(frozen=True)
class PolicyRequestResult:
    """Validated program plus provider telemetry for one inference request."""

    program: PolicyProgram
    raw_response: str
    usage: dict[str, int]


def _finite_float(value: Any, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number.")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number.")
    return result


def _bounded_float(value: Any, name: str, lower: float, upper: float) -> float:
    result = _finite_float(value, name)
    if not lower <= result <= upper:
        raise ValueError(f"{name} must be in [{lower}, {upper}], got {result}.")
    return result


def _require_keys(data: dict[str, Any], allowed: set[str], required: set[str], name: str) -> None:
    missing = required - data.keys()
    if missing:
        raise ValueError(f"{name} is missing required fields: {sorted(missing)}.")
    unknown = data.keys() - allowed
    if unknown:
        raise ValueError(f"{name} contains unsupported fields: {sorted(unknown)}.")


def _extract_json_object(text: str) -> dict[str, Any]:
    stripped = text.strip()
    if stripped.startswith("```"):
        lines = stripped.splitlines()
        if len(lines) >= 3 and lines[-1].strip() == "```":
            stripped = "\n".join(lines[1:-1]).strip()
            if stripped.startswith("json"):
                stripped = stripped[4:].lstrip()
    try:
        decoded = json.loads(stripped)
    except json.JSONDecodeError:
        start = stripped.find("{")
        end = stripped.rfind("}")
        if start < 0 or end <= start:
            raise ValueError("The model response did not contain a JSON object.") from None
        try:
            decoded = json.loads(stripped[start : end + 1])
        except json.JSONDecodeError as exc:
            raise ValueError(f"The model response contained invalid JSON: {exc}.") from exc
    if not isinstance(decoded, dict):
        raise ValueError("The model response must be one JSON object.")
    return decoded


def parse_policy_program(text: str) -> PolicyProgram:
    """Parse and strictly validate a code-policy JSON response."""
    data = _extract_json_object(text)
    _require_keys(data, {"summary", "steps"}, {"steps"}, "program")
    summary = data.get("summary", "")
    if not isinstance(summary, str) or len(summary) > 300:
        raise ValueError("program.summary must be a string no longer than 300 characters.")
    raw_steps = data["steps"]
    if not isinstance(raw_steps, list) or not 1 <= len(raw_steps) <= MAX_PROGRAM_STEPS:
        raise ValueError(f"program.steps must contain 1 to {MAX_PROGRAM_STEPS} steps.")

    steps: list[PolicyStep] = []
    for index, raw_step in enumerate(raw_steps):
        if not isinstance(raw_step, dict):
            raise ValueError(f"step[{index}] must be an object.")
        op = raw_step.get("op")
        if op not in ALLOWED_OPS:
            raise ValueError(f"step[{index}].op must be one of {ALLOWED_OPS}, got {op!r}.")
        name = f"step[{index}]"

        if op == "approach_button":
            _require_keys(raw_step, {"op", "standoff_m", "timeout_s"}, {"op", "standoff_m", "timeout_s"}, name)
            steps.append(
                PolicyStep(
                    op=op,
                    standoff_m=_bounded_float(raw_step["standoff_m"], f"{name}.standoff_m", 0.42, 0.55),
                    timeout_s=_bounded_float(raw_step["timeout_s"], f"{name}.timeout_s", 0.5, 10.0),
                )
            )
            continue

        if op == "move_grasp":
            _require_keys(
                raw_step,
                {"op", "target", "tolerance_m", "timeout_s", "wrist_roll_rad", "gripper"},
                {"op", "target", "tolerance_m", "timeout_s"},
                name,
            )
            target = raw_step["target"]
            if not isinstance(target, dict):
                raise ValueError(f"{name}.target must be an object.")
            _require_keys(target, {"frame", "position_m"}, {"frame", "position_m"}, f"{name}.target")
            frame = target["frame"]
            if frame not in (WORLD_FRAME, ROOT_FRAME):
                raise ValueError(f"{name}.target.frame must be 'world' or 'root'.")
            position = target["position_m"]
            if not isinstance(position, list) or len(position) != 3:
                raise ValueError(f"{name}.target.position_m must contain exactly three values.")
            position_m = tuple(
                _bounded_float(value, f"{name}.target.position_m[{axis}]", -20.0, 20.0)
                for axis, value in enumerate(position)
            )
            gripper = raw_step.get("gripper", "open")
            if gripper not in ("open", "closed"):
                raise ValueError(f"{name}.gripper must be 'open' or 'closed'.")
            steps.append(
                PolicyStep(
                    op=op,
                    timeout_s=_bounded_float(raw_step["timeout_s"], f"{name}.timeout_s", 0.5, 10.0),
                    target_frame=frame,
                    target_position_m=position_m,
                    tolerance_m=_bounded_float(raw_step["tolerance_m"], f"{name}.tolerance_m", 0.005, 0.08),
                    wrist_roll_rad=_bounded_float(
                        raw_step.get("wrist_roll_rad", 0.0), f"{name}.wrist_roll_rad", -1.5708, 1.5708
                    ),
                    gripper=gripper,
                )
            )
            continue

        if op == "press_button":
            _require_keys(
                raw_step,
                {"op", "overtravel_m", "timeout_s"},
                {"op", "overtravel_m", "timeout_s"},
                name,
            )
            steps.append(
                PolicyStep(
                    op=op,
                    overtravel_m=_bounded_float(raw_step["overtravel_m"], f"{name}.overtravel_m", 0.002, 0.015),
                    timeout_s=_bounded_float(raw_step["timeout_s"], f"{name}.timeout_s", 0.5, 8.0),
                )
            )
            continue

        if op == "hold":
            _require_keys(raw_step, {"op", "duration_s"}, {"op", "duration_s"}, name)
            steps.append(
                PolicyStep(
                    op=op,
                    duration_s=_bounded_float(raw_step["duration_s"], f"{name}.duration_s", 0.1, 5.0),
                )
            )
            continue

        _require_keys(raw_step, {"op"}, {"op"}, name)
        steps.append(PolicyStep(op=op))

    return PolicyProgram(summary=summary.strip(), steps=tuple(steps))


def _normalized_quaternion_wxyz(quaternion: Sequence[float]) -> tuple[float, float, float, float]:
    if len(quaternion) != 4:
        raise ValueError("A quaternion must contain four values in (w, x, y, z) order.")
    values = tuple(_finite_float(value, "quaternion component") for value in quaternion)
    norm = math.sqrt(sum(value * value for value in values))
    if norm < 1.0e-8:
        raise ValueError("A quaternion must have nonzero magnitude.")
    return tuple(value / norm for value in values)


def rotate_vector_wxyz(quaternion: Sequence[float], vector: Sequence[float]) -> tuple[float, float, float]:
    """Rotate a vector by a `(w, x, y, z)` quaternion."""
    if len(vector) != 3:
        raise ValueError("A vector must contain three values.")
    w, x, y, z = _normalized_quaternion_wxyz(quaternion)
    vx, vy, vz = (_finite_float(value, "vector component") for value in vector)
    tx = 2.0 * (y * vz - z * vy)
    ty = 2.0 * (z * vx - x * vz)
    tz = 2.0 * (x * vy - y * vx)
    return (
        vx + w * tx + (y * tz - z * ty),
        vy + w * ty + (z * tx - x * tz),
        vz + w * tz + (x * ty - y * tx),
    )


def inverse_rotate_vector_wxyz(quaternion: Sequence[float], vector: Sequence[float]) -> tuple[float, float, float]:
    """Rotate a world-frame direction into the quaternion's local frame."""
    w, x, y, z = _normalized_quaternion_wxyz(quaternion)
    return rotate_vector_wxyz((w, -x, -y, -z), vector)


def world_point_to_root(
    point_w: Sequence[float], root_position_w: Sequence[float], root_quaternion_w: Sequence[float]
) -> tuple[float, float, float]:
    """Transform a world-frame point into the floating robot-root frame."""
    if len(point_w) != 3 or len(root_position_w) != 3:
        raise ValueError("World and root positions must each contain three values.")
    delta_w = tuple(
        _finite_float(point_w[axis], "point component")
        - _finite_float(root_position_w[axis], "root-position component")
        for axis in range(3)
    )
    return inverse_rotate_vector_wxyz(root_quaternion_w, delta_w)


def root_point_to_world(
    point_b: Sequence[float], root_position_w: Sequence[float], root_quaternion_w: Sequence[float]
) -> tuple[float, float, float]:
    """Transform a floating robot-root-frame point into the world frame."""
    if len(root_position_w) != 3:
        raise ValueError("The root position must contain three values.")
    rotated = rotate_vector_wxyz(root_quaternion_w, point_b)
    return tuple(rotated[axis] + _finite_float(root_position_w[axis], "root-position component") for axis in range(3))


def build_policy_prompt(instruction: str, state: dict[str, Any]) -> str:
    """Build the text-only prompt with an explicit frame and primitive contract."""
    state_json = json.dumps(state, indent=2, sort_keys=True)
    prepress_w = state.get("button", {}).get("prepress", {}).get("world_position_m", [0.0, 0.0, 0.0])
    suggested_program = {
        "summary": "Approach the button, move the grasp midpoint to prepress, press, and stop.",
        "steps": [
            {"op": "approach_button", "standoff_m": 0.50, "timeout_s": 6.0},
            {
                "op": "move_grasp",
                "target": {"frame": "world", "position_m": prepress_w},
                "tolerance_m": 0.025,
                "timeout_s": 6.0,
                "wrist_roll_rad": 0.0,
                "gripper": "open",
            },
            {"op": "press_button", "overtravel_m": 0.006, "timeout_s": 5.0},
            {"op": "stop"},
        ],
    }
    suggested_json = json.dumps(suggested_program, indent=2)
    return f"""You are a high-level code policy for a Go2 quadruped with a D1 arm.
Return exactly one JSON object and no Markdown or explanatory text.

Task: {instruction}

Frame contract:
- `world` is the fixed Isaac simulation world frame.
- `root` is the current floating Go2 base frame: +x forward, +y left, +z up.
- Quaternion arrays use `(w, x, y, z)` order.
- All positions are metres. Base velocities are resolved in `root`/body coordinates.
- The Cartesian arm command controls `grasp_midpoint`, the midpoint of the two finger bodies.
  It does not command the Link6 origin.
- A target attached to the stationary button should use `world`, because a stale root-frame target
  changes meaning while the base moves.
- The button moves along `button.press_axis`; `button.prepress` is outside the face and
  `press_button` follows the live axis safely.

Allowed operations and numeric limits:
- approach_button: standoff_m [0.42, 0.55], timeout_s [0.5, 10]
- move_grasp: target frame world/root, position_m [x,y,z], tolerance_m [0.005, 0.08],
  timeout_s [0.5, 10], wrist_roll_rad [-1.5708, 1.5708], gripper open/closed
- press_button: overtravel_m [0.002, 0.015], timeout_s [0.5, 8]
- hold: duration_s [0.1, 5]
- stop: no additional fields

Use no more than {MAX_PROGRAM_STEPS} steps. For this task, first approach until the button is inside the root-frame
arm workspace, then move the open grasp midpoint to the provided world-frame prepress point, then press and stop.
Do not invent coordinates: copy the relevant numeric position from the state.

A valid program specialized to the current prepress coordinate is shown below. Return this shape, changing it only
when the current state requires a safer plan:
{suggested_json}

Current symbolic state:
{state_json}
"""


def _response_usage(response: Any) -> dict[str, int]:
    """Normalize OpenAI-compatible token usage without requiring it."""
    usage = response.get("usage") if isinstance(response, dict) else getattr(response, "usage", None)
    if usage is None:
        return {}
    normalized: dict[str, int] = {}
    for name in ("prompt_tokens", "completion_tokens", "total_tokens"):
        value = usage.get(name) if isinstance(usage, dict) else getattr(usage, name, None)
        if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
            normalized[name] = value
    return normalized


def _local_chat_completion(base_url: str, model: str, prompt: str, timeout_s: float) -> tuple[str, dict[str, int]]:
    parsed_url = urllib.parse.urlparse(base_url)
    if parsed_url.scheme not in ("http", "https") or parsed_url.hostname not in ("127.0.0.1", "localhost", "::1"):
        raise ValueError("--llm-base-url must be an HTTP(S) loopback URL on this PC.")
    endpoint = base_url.rstrip("/")
    if endpoint.endswith("/v1"):
        endpoint = f"{endpoint}/chat/completions"
    else:
        endpoint = f"{endpoint}/v1/chat/completions"
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.1,
            "max_tokens": 1400,
            "response_format": {"type": "json_object"},
        }
    ).encode("utf-8")
    request = urllib.request.Request(
        endpoint,
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_s) as response:  # noqa: S310 - explicit local endpoint
        decoded = json.loads(response.read().decode("utf-8"))
    try:
        content = decoded["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError) as exc:
        raise RuntimeError("The local model server returned an unexpected chat-completion response.") from exc
    if not isinstance(content, str):
        raise RuntimeError("The local model server returned non-text message content.")
    return content, _response_usage(decoded)


def request_policy_program(
    prompt: str,
    model: str,
    provider: str = "auto",
    base_url: str | None = None,
    timeout_s: float = 60.0,
) -> PolicyRequestResult:
    """Request and validate one code-policy program from hosted HF or a local server."""
    response_text, usage = request_text_completion(prompt, model, provider, base_url, timeout_s)
    return PolicyRequestResult(parse_policy_program(response_text), response_text, usage)


def request_text_completion(
    prompt: str,
    model: str,
    provider: str = "auto",
    base_url: str | None = None,
    timeout_s: float = 60.0,
) -> tuple[str, dict[str, int]]:
    """Request one visible text completion and normalize optional token usage."""
    if base_url:
        response_text, usage = _local_chat_completion(base_url, model, prompt, timeout_s)
    else:
        try:
            from huggingface_hub import InferenceClient
        except ModuleNotFoundError as exc:
            raise RuntimeError(
                "huggingface_hub is required for hosted inference; install a compatible version or use --llm-base-url."
            ) from exc
        client = InferenceClient(provider=provider, timeout=timeout_s)
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1400,
            response_format={"type": "json_object"},
        )
        response_text = response.choices[0].message.content
        if not isinstance(response_text, str):
            raise RuntimeError("Hugging Face returned non-text message content.")
        usage = _response_usage(response)
    return response_text, usage
