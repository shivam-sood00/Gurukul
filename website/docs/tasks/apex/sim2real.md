---
title: Sim2Real
---

# Sim2Real

APEX sim2real is limited to the hardware runners listed here. Treat every runner as a specific policy-interface
contract, not a generic APEX deployment path.

## Hardware Runners

| Runner | Policy shape | Notes |
| --- | --- | --- |
| `unitree-sim2real/RL_policy_runner/sim2real/go2_apex_tracker_hardware.py` | Go2 APEX tracker ONNX actor. | Loads reference NPZ files, infers tracker offsets from the ONNX width when possible, and supports `--policy-path`, `--motion-file`, `--reference-offsets`, and `--joystick-command`. |
| `unitree-sim2real/RL_policy_runner/sim2real/b2_z1_apex_history_hardware.py` | B2+Z1 one-step history ONNX actor. | Expects 600-D term-major history input and 18 actions for 12 B2 leg joints plus Z1 `joint1` through `joint6`; the gripper is held at `0.0`. |

## Go2 Tracker

```bash
cd unitree-sim2real
python RL_policy_runner/sim2real/go2_apex_tracker_hardware.py eth0 \
  --policy-path RL_policy_runner/sim2real/go2_apex_tracker_policy.onnx
```

The runner defaults to packaged tracker motions under:

```text
unitree-sim2real/RL_policy_runner/sim2real/apex_tracker_motions/*.npz
```

Use `--motion-file <npz_or_glob>` for a different motion set. Use `--joystick-command` when the remote should drive
`vx`, `vy`, and yaw instead of the active reference clip.

## B2+Z1 History Tracker

```bash
cd unitree-sim2real
python RL_policy_runner/sim2real/b2_z1_apex_history_hardware.py eth0 \
  --policy-path ../logs/rsl_rl/unitree_b2_z1_arm_apex_flat_tracker_one_step_future_history_distill/<run>/exported/policy.onnx
```

The default motion glob points at B2+Z1 APEX NPZ clips:

```text
source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/npz/b2_z1_motions/**/*.npz
```

The B2 legs are sent through DDS lowcmd. Z1 `joint1` through `joint6` targets are sent through the Z1 SDK with the same
deployment-style arm adapter used by the B2+Z1 MuJoCo sim2sim path.

## Before Hardware

- Match the exported actor observation width to the runner's expected layout.
- Match reference offsets, reference frame features, and history flattening order.
- Match action order, action scale, PD gains, default joint pose, and torque limits.
- Confirm the motion NPZ joint names match the runner's tracked joint subset.
- Validate the exported policy in Isaac Lab playback and APEX sim2sim first.
- Start with low-speed motions or joystick commands before running dynamic clips.
- Keep estop, fall detection, and operator handoff behavior checked before policy control.

## Remote Notes

- `R1+L1` is the always-checked estop.
- APEX motion switching uses `X` and `B` to cycle reference clips where supported.
- `A` remains the controller stop button while APEX switching is active.
