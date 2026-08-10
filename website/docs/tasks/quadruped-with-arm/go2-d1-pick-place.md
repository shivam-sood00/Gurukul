---
title: Go2+D1 Pick And Place
description: Experimental Go2+D1 pick, place, teacher, and frozen-controller training tasks.
---

# Go2+D1 Pick And Place

## What It Is

These flat-terrain tasks train Unitree Go2 with the D1 arm to pick up a can-like cylindrical object, then extend the
behavior to placement in a shallow tray.

The primary starting point is `Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-Pick-v0`. Use the variants below
for privileged training, placement, or a frozen low-level controller.

## Validation Stage

| Surface | Training evidence | Artifact | Transfer |
| --- | --- | --- | --- |
| Eight tasks registered and configured | Static reward/curriculum/interface tests; promotion gates have not passed | Direct tasks have no bundled checkpoint; hierarchical tasks require a user WBC export | No object-scene sim2sim or hardware validation |

These are training environments, not demonstrated pick/place skills. Do not promote a run until bilateral contact,
lift/hold, release, and task-success metrics pass together.

<details>
<summary>Show task IDs, horizons, and the action contract</summary>

| Field | Value |
| --- | --- |
| Pick task ID | `Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-Pick-v0` |
| Pick teacher task ID | `Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-Pick-Teacher-v0` |
| Pick-place task ID | `Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-PickPlace-v0` |
| Stationary pick-place task ID | `Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-StationaryPickPlace-v0` |
| Frozen-WBC pick task ID | `Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-Pick-Wbc-Hierarchical-v0` |
| Frozen-leg-WBC pick task ID | `Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-Pick-LegWbcArm-Hierarchical-v0` |
| Frozen-leg-WBC Cartesian pick task ID | `Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-Pick-LegWbcEe-Hierarchical-v0` |
| Fast replicated Cartesian pick task ID | `Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-Pick-LegWbcEe-Hierarchical-Fast-v0` |
| Robot action space | 12 Go2 leg joints + 6 D1 arm joints + 1 binary D1 gripper state |
| Pick episode length | `10 s` |
| Pick-place episode length | `22 s` |
| Stationary pick-place episode length | `14 s` |
| Main config | `source/Gurukul/Gurukul/tasks/manager_based/locomotion/velocity/config/quadruped_with_arm/unitree_go2_d1_arm/loco_manipulation_env_cfg.py` |

</details>

## Main Commands

Use `--num_envs 256` for an initial smoke run. The recommended training order is WBC, pick, stationary pick-place, full
pick-place, then a hierarchical pick variant. Move past WBC only after base tracking is stable and
`arm_ee_target_tracking` improves while `arm_ee_target_error` falls.

Train the stationary pick primitive first:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-Pick-v0 \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --run_name go2_d1_pick_flat \
  --num_envs 256
```

Train the privileged pick teacher only when preparing a teacher/student distillation run:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-Pick-Teacher-v0 \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --run_name go2_d1_pick_teacher_flat \
  --num_envs 256
```

After grasp and held-lift metrics are reliable, train stationary pick-place:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-StationaryPickPlace-v0 \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --run_name go2_d1_stationary_pick_place_flat
```

Then train full pick-place with the approach curriculum:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-PickPlace-v0 \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --run_name go2_d1_pick_place_flat
```

For hierarchical pick over a frozen full WBC policy, export the WBC actor and run:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-Pick-Wbc-Hierarchical-v0 \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --run_name unitree_go2_d1_pick_wbc_hierarchical_flat \
  env.wbc_policy_path=/path/to/exported_wbc_policy.pt
```

Play a trained hierarchical pick checkpoint with the same frozen WBC export:

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-Pick-Wbc-Hierarchical-v0 \
  --agent=rsl_rl_cfg_entry_point \
  --load_run <high_level_run_folder_name> \
  --num_envs 1 \
  --real-time \
  env.wbc_policy_path=/absolute/path/to/exported_wbc_policy.pt
```

Hierarchical playback leaves the learned policy in control of all nine high-level commands; the scripted mounted-arm
play-stage override is not applied. Its frozen-WBC runtime also disables the stage-1 differential-IK reference event,
which is not consumed by the deployed WBC and can otherwise trigger an unnecessary CUDA cuSolver initialization.

For direct high-level D1 arm commands over a frozen leg WBC, export the leg-only actor and run:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-Pick-LegWbcArm-Hierarchical-v0 \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --run_name unitree_go2_d1_pick_leg_wbc_arm_hierarchical_flat \
  env.wbc_policy_path=/path/to/exported_leg_wbc_policy.pt
```

The recommended Stage-2 task is now a privileged high-level teacher. It commands base velocity, pitch, height,
Cartesian grasp-center position, wrist roll, and gripper state while the exported Stage-1 policy controls only the
legs:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-Pick-LegWbcEe-Hierarchical-v0 \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --run_name unitree_go2_d1_pick_leg_wbc_ee_teacher_flat \
  --num_envs 4096 \
  env.wbc_policy_path=/absolute/path/to/exported_leg_wbc_policy.pt
```

For the first throughput-focused run, use the otherwise identical fast variant:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-Pick-LegWbcEe-Hierarchical-Fast-v0 \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --run_name unitree_go2_d1_pick_leg_wbc_ee_teacher_fast_flat \
  --num_envs 4096 \
  env.wbc_policy_path=/absolute/path/to/exported_leg_wbc_policy.pt
```

This variant fixes the object at its nominal `56 mm` diameter, retains pose, velocity, material, and mass
randomization, and enables replicated physics. It also replaces the seven filtered, four-frame non-gripper D1 contact
sensors with one unfiltered history-1 sensor. Consequently, any contact on `base_link` or `Link1`--`Link6` is treated
as undesired, while the two filtered finger--object sensors and bilateral grasp semantics are unchanged. Use the
standard hierarchical task for final object-scale robustness fine-tuning.

Start this as a fresh run. The ten normalized actions are
`[vx, vy, yaw_rate, body_pitch, body_height, grasp_x, grasp_y, grasp_z, wrist_roll, gripper]`; roll remains fixed at
zero. The actor and critic train from the privileged `teacher` observation group. It contains five frames of exact
proprioception, object/contact/stage state, and the object's full SE(3) pose. The pose is represented by three
root-frame position values plus a continuous six-value rotation representation. A separate five-frame `policy` group
is retained as the future distillation interface. The frozen leg policy still consumes one unstacked 56-value frame
and produces 12 leg actions.

The teacher starts with the table at `0.65 m`, placing the object near `0.53 m` before pose jitter. This is the verified
nearby-pick geometry and keeps the initial curriculum inside the `0.60 m` Reach transition, so it learns the
arm/gripper primitive before requiring a long approach. The shaped Cartesian action is initialized from the measured
finger midpoint at reset, so the first policy step does not pull the ready pose toward the workspace center. Distance
expands to `1.20 m` only after successful picks. Resets
mix 15% easiest-distance replay, 25% current-frontier examples, and uniformly sampled intermediate distances, so the
near grasp is retained as locomotion distance grows. The can-like object is
`56 mm` in nominal diameter and `100 mm` tall, is scaled by `0.95–1.05`, and has randomized friction, lateral pose,
orientation, and absolute mass from `0.04–0.25 kg`. It therefore remains below the `0.40 kg` payload limit and inside
the gripper aperture throughout training.

Approach shaping combines signed distance progress with measured velocity toward the desired object standoff. Moving
away is penalized, standing still earns no approach reward, and the velocity incentive fades near the standoff.
Zero-centered standoff, facing, and end-effector distance costs remain continuous across stage boundaries, so avoiding
Reach cannot remove a large manipulation cost. The stage graph then transfers control to bilateral grasp, lift, and
hold rewards. Older eight-action Cartesian checkpoints are incompatible with this task.

The frozen full-WBC variant commands D1 in Cartesian space over the same body-clear `0.12–0.58 m` reach shell used for
low-level pretraining. The Cartesian frozen-leg variant slew-limits base, posture, grasp-position, and wrist commands
before execution. Its binary gripper uses `-0.25/+0.25` open/close hysteresis, preventing a stochastic action near zero
from opening and closing the jaws every control step. The actor observes these applied commands, while first- and
second-order raw-action costs still teach a stable policy. Measured D1 velocity and acceleration remain regularized.

For this hierarchy, the inherited random locomotion command sampler is converted to a passive zero-initialized buffer.
Only the high-level action writes the velocity, pitch, and height consumed by the frozen WBC; the old 4–6 second
resampling and standing-environment mask no longer corrupt actor history or playback command arrows. Small learned
velocity commands are preserved rather than thresholded to zero.

Retrain the WBC first after the command-sampling and smoothness-reward changes, export it, and then train the
hierarchical pick policy from scratch. Older checkpoints saw a materially different command distribution.

Treat the following as promotion gates, not just diagnostic rewards:

- Pick is working only when `Curriculum/pick_bilateral_contact`, `object_lifted_with_gripper_contact`, and
  `object_hold_lifted` become nonzero together. `gripper_object_contact` now reports contact acquisition or loss, so
  it should not remain positive while a grasp is merely being held. Object height without bilateral contact is not a
  valid grasp.
- Pick-place is working only when `object_released_in_tray` becomes nonzero and the `object_placed` termination fires.
  `object_in_tray` alone can still describe an unreleased or transient object.
- Retrain instead of resuming direct manipulation checkpoints created before the reachable-table, virtual-gripper,
  grasp-memory, and action-scale changes. Retrain and re-export WBC policies before hierarchical training because the
  arm command-to-joint scale changed even though the exported actor tensor shapes did not.
