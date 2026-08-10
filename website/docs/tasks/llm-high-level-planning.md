---
title: LLM High-Level Planning
description: Language-model planning over bounded robot-control primitives with live physical feedback.
---

# LLM High-Level Planning

## What It Is

LLM High-Level Planning is the robot-agnostic task family for language-conditioned control over measured simulator
state. A language model selects bounded high-level actions; deterministic robot adapters execute those actions through
the platform's locomotion, posture, arm, and gripper controllers. Physics remains the source of task progress and
success.

The first implementation uses Go2+D1, but the task-family boundary is intentionally broader. Adding a humanoid or
another mobile manipulator requires a robot-specific symbolic-state adapter, action schema, executor, and physical
success contract without changing the planner evaluation concepts on this page.

```text
instruction + symbolic state + execution history
                    ↓
            LLM JSON decision
                    ↓
      schema and physical safety checks
                    ↓
 robot executor → simulator → measured feedback
                    └───────────────↑
```

The LLM remains responsible for choosing each action and its parameters. Geometric planning context may recommend a
feasible operation, but it does not replace or override a valid model decision.

## Initial Supported Tasks

| Task ID | Runner scenario | Physical success contract |
| --- | --- | --- |
| `Gurukul-Isaac-LLM-Go2-D1-Press-Switch-v0` | `run_go2_d1_switch.py` | The spring-return button reaches its measured press threshold. |
| `Gurukul-Isaac-LLM-Go2-D1-Open-Door-v0` | `--scenario open-door` | The physical door hinge reaches 75 degrees. |
| `Gurukul-Isaac-LLM-Go2-D1-Pick-Place-v0` | `--scenario pick` | Bilateral finger contact and at least 8 cm measured lift. |
| `Gurukul-Isaac-LLM-Go2-D1-Pick-Place-v0` | `--scenario pick-place` | A previously lifted can is released inside the tray and settles below the speed threshold. |

The door has a `0.90 × 1.58 m` collision opening and a lever handle. At the 75-degree success angle, its projected
lateral gap is approximately `0.67 m`, leaving room for a later traversal phase. Door traversal is not yet part of the
success contract.

## Prerequisite Controller

The current Go2+D1 adapter uses an exported `LegWBC-AsyncArm` policy for the 12 Go2 legs. The frozen policy stabilizes
base velocity, body pitch, and body height while the high-level executor controls D1 and its gripper separately. See
[Go2+D1 WBC](quadruped-with-arm/go2-d1-wbc) for training and export commands.

Use an export with the current 56-input low-level observation contract:

```text
/path/to/unitree_go2_d1_leg_wbc_async_arm_flat/exported/policy.pt
```

## Run the Planner

Start with the door task in event-triggered receding-horizon mode:

```bash
python scripts/llm/run_go2_d1_service.py \
  --base-policy /path/to/unitree_go2_d1_leg_wbc_async_arm_flat/exported/policy.pt \
  --scenario open-door \
  --controller llm \
  --llm-action-space primitive \
  --llm-control-mode receding-horizon \
  --llm-max-requests 12 \
  --llm-model Qwen/Qwen3-4B-Instruct-2507 \
  --llm-provider nscale \
  --dashboard
```

Use `--scenario pick` to isolate grasp-and-lift mechanics before testing `--scenario pick-place`. Add
`--exit-on-success` for a bounded automated evaluation. The mounted RGB sensor is rendered by default but is not sent
to this text-only planner; use `--no-camera` for a faster mechanics run.

Before attributing a failure to the LLM, run the deterministic mechanics oracle with the same scene and low-level
checkpoint:

```bash
python scripts/llm/run_go2_d1_service.py \
  --base-policy /path/to/unitree_go2_d1_leg_wbc_async_arm_flat/exported/policy.pt \
  --scenario open-door \
  --controller demo \
  --dashboard
```

The oracle is a mechanics-isolation ceiling, not the default LLM action space and not a learned policy.

## Primitive Action Contract

The default `primitive` action space exposes only controls that exist on the robot adapter:

| Operation | Model-controlled fields | Executor behavior |
| --- | --- | --- |
| `base_velocity` | Root-frame `vx`, `vy`, yaw rate, pitch, height, duration | Applies a bounded timed locomotion command. |
| `drive_base_to` | World target, standoff, maximum speeds, pitch, height, tolerance, timeout | Recomputes target distance and heading at 50 Hz. |
| `move_ee` | World/root finger-midpoint target, wrist roll, pitch, height, tolerance, timeout | Transforms the target into the live root frame and runs Cartesian IK at 50 Hz. |
| `set_gripper` | `closed_fraction`, duration | Uses `0` for fully open and `1` for fully closed. |
| `hold` | Duration | Holds posture, arm, and gripper targets with zero nominal base velocity. |
| `stop` | None | Requests a safe zero-base stop. |

There is no callable `open_door`, `pick_object`, or `place_object` primitive in this action space. Those names describe
physical outcomes that must emerge from velocity, IK, and gripper decisions. For a deliberately easier oracle-style
ablation, use `--llm-action-space semantic`; report it separately because the action spaces are not equivalent.

Small forward commands are not reliably tracked by the current low-level checkpoint. A nonzero forward request is
therefore at least `0.40 m/s`, while `drive_base_to` remains exactly zero inside its stopping tolerance and scales the
requested speed with measured distance outside it.

## Planner Modes and Frequency

| Mode | Initial model output | Later calls | Use |
| --- | --- | --- | --- |
| `one-shot` | Up to 12 primitives | None | Bounded open-loop language-plan baseline. |
| `reactive` | One primitive | After every outcome | Latest-state closed-loop baseline without persistent plan context. |
| `receding-horizon` | Up to 12 primitives | One primitive after every outcome | Retains the initial horizon while replanning from live state. |

This is language-level receding-horizon control, not numerical torque MPC. Physics runs at 200 Hz and robot control at
50 Hz. LLM requests are asynchronous and event-triggered, so there is no fixed LLM frequency: a new decision is made
after the prior primitive completes or fails. While a request or non-base primitive is active, a bounded planar station
keeper corrects excessive base drift.

## Observations, Frames, and LLM Authority

The text model receives unit-labelled symbolic state rather than pixels. The current adapter reports:

- the Go2 root pose and measured body velocity;
- body pitch/height and commanded/measured gripper closure;
- the D1 finger-midpoint pose;
- live target poses in both fixed `world` and moving `root` frames;
- IK workspace bounds, contact state, hinge angle, object lift, placement, and task progress;
- the previous primitive outcome and any timeout or reachability failure; and
- advisory feasibility context, including distance-scaled approach values and whether a target is inside the live IK
  workspace.

`root` uses `+x` forward, `+y` left, and `+z` up. Quaternions use `(w, x, y, z)`. Stationary targets stay in world
coordinates and are transformed through the current root pose every control step.

The advisory context contains `recommended_op`, a measured reason, and candidate parameters. The LLM may choose a
different valid operation and that choice is executed. Hard checks are limited to JSON/schema validity, command bounds,
supported frames, Cartesian workspace, and timeouts. An unreachable target is rejected and returned as execution
feedback; it is never silently replaced with a deterministic operation.

## Live Dashboard and Saved Traces

Every service run creates a timestamped directory under `logs/llm/go2_d1_service/` containing:

- `dashboard.html`, a self-contained trace viewer; and
- `events.jsonl`, the append-only structured event record.

Add `--dashboard` to publish the trace at `http://127.0.0.1:8765/` while the runner is active. The page shows the exact
visible prompt and response, request-time observation, advisory planning context, validated model program, primitive
transitions, physical telemetry, failures, latency, token usage, and optional cost estimate. It cannot display private
chain-of-thought that the model did not return.

Malformed model output gets at most one counted JSON-repair request. The raw invalid response and provider usage are
saved before repair. A second validation failure produces the normal safe stop.

## Hosted and Local Models

For hosted Hugging Face inference, authenticate with `hf auth login` and verify current provider pricing before setting
a large request budget. Provider pricing changes; use the
[Hugging Face Inference Providers pricing guide](https://huggingface.co/docs/inference-providers/en/pricing) as the
current reference.

For local evaluation, keep the model server in a separate environment so its dependencies do not alter Isaac Sim:

```bash
vllm serve Qwen/Qwen3-4B-Instruct-2507 \
  --host 127.0.0.1 \
  --port 8000 \
  --max-model-len 4096 \
  --gpu-memory-utilization 0.50
```

Point the robot runner at that server with:

```bash
--llm-base-url http://127.0.0.1:8000
```

## Button-Press Starter

The earlier switch runner remains useful for validating RGB updates, low-level policy motion, frames, and a bounded
single-request language program:

```bash
python scripts/llm/run_go2_d1_switch.py \
  --base-policy /path/to/unitree_go2_d1_leg_wbc_async_arm_flat/exported/policy.pt \
  --llm-model Qwen/Qwen3-4B-Instruct-2507 \
  --llm-provider nscale \
  --llm-max-requests 1 \
  --llm-dashboard
```

Its logs are stored separately under `logs/llm/go2_d1_switch/`. Unlike the service runner's robot-agnostic primitive
evaluation surface, this starter uses button-specific bounded operations.

## Current Limitations

- The maintained high-level planner adapter currently supports Go2+D1 only; humanoid and other robot adapters are
  future additions to this task family.
- The service planner is text-only even when the mounted RGB camera is enabled.
- Door, pick, and pick-place are physical promotion gates, not guaranteed demonstrations. Low-level locomotion, IK,
  contact geometry, and gripper mechanics can fail independently of the language plan.
- Door traversal geometry is present, but traversal is not yet included in the instruction or success condition.
- Primitive and semantic action-space results must not be compared as if they expose equal controller authority.

## Credits and Scope

The high-level planning/evaluation surface is repository-authored and does not claim implementation of a specific
robot-planning paper. Its first adapter builds on the Go2+D1 task and robot sources documented in
[Acknowledgements](../reference/credits). Model-provider and model-card terms remain separate from Gurukul's
license.
