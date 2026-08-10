---
title: Go2+D1 WBC
description: Train and export Go2+D1 low-level whole-body and leg-only controllers.
---

# Go2+D1 WBC

:::caution[Optional robot asset]
The public repository includes Gurukul's Go2+D1 tasks and authored model overlays, but not Unitree's downloaded D1
meshes or generated USD files that embed them. Install the source geometry and validate the local composite paths
below before running these tasks.
:::

## Install the D1 Geometry

- [Unitree official open-source portal](https://www.unitree.com/opensource/)
- [Unitree download center](https://www.unitree.com/download/)
- [Official D1-550 description archive](https://oss-global-cdn.unitree.com/static/9b20252a26374d50aa369532657d0143.zip)
- Isaac Lab destination: `source/Gurukul/data/Robots/unitree/go2_with_d1/`
- MuJoCo destination: `unitree-sim2real/unitree_mujoco/unitree_robots/go2_with_d1/`

The official archive's `package.xml` declares `BSD`, but the archive does not include the complete license text, BSD
variant, or copyright notice needed for an unambiguous redistribution record. Gurukul therefore downloads the geometry
locally instead of redistributing it. The pinned archive SHA-256 is
`71abcc4cd6359bf09765b1fa9e87ed5369b6a688bf8c9432d5ccc2dd8e42faa5`.

Build the local composite URDF and install the official D1 meshes for Isaac tools with:

```bash
python scripts/tools/build_go2_d1_viser_urdf.py
```

The builder downloads the archive directly from Unitree, verifies its checksum, and writes into the ignored asset
directory. It creates the Viser URDF but does not reconstruct the exact Isaac training USD. The Isaac tasks also expect
a locally imported or composed `usd/go2_d1.usd` with the `go2_description/d1` prim structure used by the checked-in
overlays. Generate that base USD from the official geometry in Isaac Sim, verify its prim and joint paths against the
overlays, and keep it outside Git. The checked-in `go2_d1_center_gripper.usda` and
`go2_d1_center_gripper_apex.usda` layers apply Gurukul's task-specific corrections on top of it.

For MuJoCo, copy the nine downloaded STL files into the ignored `go2_with_d1/assets/` directory. The checked-in
`go2_with_d1.xml` and `scene_flat.xml` files then provide Gurukul's composed model and scene definitions.

## What It Is

Go2+D1 WBC tasks are the recommended stage-1 pretraining tasks for Go2+D1 loco-manipulation. The policy controls Go2
legs, D1 arm joints, and gripper joints while tracking bounded base-velocity commands, D1 end-effector goal positions,
base pitch/height commands, and gripper targets.

Policy-controlled arm tasks expose Go2 leg and D1 arm joints to the policy. ArmMoving tasks use leg-only policy actions
while the D1 arm follows scripted joint-space primitives.

## Validation Stage

| Surface | Training evidence | Artifact | Transfer |
| --- | --- | --- | --- |
| Registered training configurations | Static contract and targeted CPU tests | Train and export locally; no bundled checkpoint | No validated sim2sim or hardware result |

These tasks are development surfaces. A successful registry or contract test does not establish policy convergence;
use the promotion metrics below before reusing a checkpoint in a hierarchical task.

## Main Commands

Train the flat WBC pretraining task first:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-WBC-Flat-Unitree-Go2-D1-Arm-v0 \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --run_name unitree_go2_d1_wbc_flat
```

Export a finished WBC checkpoint. Playback exports `policy.pt` before taking the single requested simulation step:

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=Gurukul-Isaac-WBC-Flat-Unitree-Go2-D1-Arm-v0 \
  --agent=rsl_rl_cfg_entry_point \
  --load_run <wbc_run_folder_name> \
  --num_envs 1 \
  --headless \
  --max_steps 1
```

Omitting `--checkpoint` selects the latest checkpoint in `--load_run`. To select an exact checkpoint, pass its full
path with `--checkpoint /absolute/path/to/model_<iteration>.pt`; a bare checkpoint filename is not resolved relative
to `--load_run`.

The frozen high-level task consumes
`logs/rsl_rl/unitree_go2_d1_wbc_flat/<wbc_run_folder_name>/exported/policy.pt`.

Smoothness is learned through action-rate, action-curvature, D1 velocity, and D1 acceleration penalties; the action
path does not impose velocity or acceleration filters. Train a new WBC from scratch after changing these rewards and
the command distribution, then train a new hierarchical policy against its export.

Try the mjlab-derived action-scale variant when comparing Go2 leg-scale parity while keeping the task-capable D1
workspace:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-WBC-Flat-Unitree-Go2-D1-Arm-v0-MJLabActionScale \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --run_name unitree_go2_d1_wbc_flat_mjlab_action_scale
```

Train the rough WBC pretraining task after the flat WBC is stable:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-WBC-Rough-Unitree-Go2-D1-Arm-v0 \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --run_name unitree_go2_d1_wbc_rough
```

Train the leg-only WBC directly on the Go2+D1 assembly while D1 follows externally generated asynchronous joint-space
trajectories. The learned action contains only the 12 Go2 leg targets and tracks velocity, body pitch, and body height.
The performance-gated curriculum alternates stationary and walking trials while expanding the safe D1 joint-motion and
posture ranges; it regresses instead of increasing difficulty when locomotion or posture tracking falls below its lower
threshold:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-LegWBC-AsyncArm-Flat-Unitree-Go2-D1-Arm-v0 \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --run_name unitree_go2_d1_leg_wbc_async_arm_flat
```

The current deployable actor has 56 inputs: base angular velocity, gravity, six velocity/posture fields, all 20
Go2+D1 joint positions, the 12 Go2 velocities, and the previous 12 leg actions. Its command field is
`[vx, vy, wz, 0, pitch, height]`: roll is never sampled or commanded; the zero placeholder preserves compatibility with
existing 56-input checkpoints and exported low-level interfaces. Historical `LegWBC-AsyncArm`
checkpoints used 64 inputs because they also included eight unavailable D1/gripper velocities. The training loader can
warm-start a historical 64-input checkpoint: it removes only those eight first-layer columns, retains all downstream
actor and critic weights, and initializes a fresh optimizer. This avoids restarting from bare-Go2 locomotion:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-LegWBC-AsyncArm-Flat-Unitree-Go2-D1-Arm-v0 \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --resume \
  --load_run <strong_leg_wbc_run_folder> \
  --max_iterations 4000 \
  --run_name posture_arm_finetune
```

Omitting `--checkpoint` loads the latest checkpoint from `--load_run`. To select a different iteration, pass its full
absolute path with `--checkpoint /absolute/path/to/model_<iteration>.pt`. A normal 56-input checkpoint resumes its
optimizer; the 64-to-56 warm start intentionally resets optimizer moments. For a new controller, omit `--resume` and
`--load_run`: the leg WBC uses the same unclipped Gaussian action contract and PPO defaults as the proven Go2
locomotion baseline (`1.0` initial standard deviation, `1e-3` learning rate, and `0.01` entropy coefficient). Frozen
hierarchical inference preserves those low-level outputs and clamps the resulting joint targets to each leg joint's
90% safety range; it does not truncate policy outputs to `[-1, 1]`.

Use a Go2-only checkpoint only as a last resort: its actor input and action shapes generally differ, and it was not
trained against the mounted D1 mass, moving-arm reactions, or payload randomization.

The language-conditioned switch, door, pick, and pick-place runners now have a dedicated public guide:
[LLM High-Level Planning](../llm-high-level-planning). That page is the maintained reference for planner modes, action
schemas, prompts, live traces, hosted/local models, and physical success contracts. This page remains the reference for
training and exporting the Go2+D1 low-level controller used by the current planner adapter.

Train the online-IK APEX arm variant when testing a decaying arm action prior without offline motion data:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-WBC-ApexArm-Flat-Unitree-Go2-D1-Arm-v0 \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --run_name unitree_go2_d1_wbc_apex_arm_flat
```

After exporting a flat WBC policy, train hierarchical pick with the frozen low-level controller:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-Pick-Wbc-Hierarchical-v0 \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --run_name unitree_go2_d1_pick_wbc_hierarchical_flat \
  env.wbc_policy_path=/path/to/exported_wbc_policy.pt
```

After exporting a flat leg-only WBC policy, train high-level pick with direct D1 arm-joint commands:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-Pick-LegWbcArm-Hierarchical-v0 \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --run_name unitree_go2_d1_pick_leg_wbc_arm_hierarchical_flat \
  env.wbc_policy_path=/path/to/exported_leg_wbc_policy.pt
```

For the velocity-plus-grasp-pose hierarchy, train the privileged Cartesian high-level teacher instead. It uses one
policy for base velocity, body pitch/height, grasp-center XYZ, wrist roll, and gripper commands:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-Pick-LegWbcEe-Hierarchical-v0 \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --run_name unitree_go2_d1_pick_leg_wbc_ee_teacher_flat \
  env.wbc_policy_path=/path/to/exported_leg_wbc_policy.pt
```

Train the flat policy-controlled arm task:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Velocity-Flat-Unitree-Go2-D1-Arm-v0 \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --run_name go2_d1_arm_flat
```

Visualize D1 moving-arm primitives:

```bash
python scripts/tools/visualize_go2_d1_arm_motion.py \
  --num_envs 1 \
  --real-time
```

Check the WBC end-effector workspace in a browser before training:

```bash
python scripts/tools/visualize_go2_d1_workspace_viser.py \
  --grid 9 \
  --max-samples 300 \
  --max-ghosts 12 \
  --ik-starts 5 \
  --save-results /tmp/go2_d1_workspace_ik.csv
```

The pale-blue points are the actual geometrically sampled training volume. Green/red points are the subset audited by
IK. The full reach spheres and ghost arms are hidden by default because neither one alone represents the training set;
use the GUI toggles to inspect them when needed.

Sweep the same commands through the Isaac Lab differential-IK path:

```bash
python scripts/tools/visualize_go2_d1_wbc_workspace.py \
  --task=Gurukul-Isaac-WBC-Flat-Unitree-Go2-D1-Arm-v0 \
  --num_envs 1 \
  --grid-size 4 \
  --max-cycles 1 \
  --real-time
```

Audit randomized targets with a deliberately rotated fixed base to exercise the Jacobian frame conversion:

```bash
python scripts/tools/evaluate_go2_d1_wbc_arm_commands.py \
  --task=Gurukul-Isaac-WBC-Flat-Unitree-Go2-D1-Arm-v0 \
  --num_envs 16 \
  --samples 32 \
  --base-rpy 0.15 -0.10 0.35 \
  --headless
```

This isolated nominal-model audit disables domain randomization and automatic episode timeouts, executes the compact
carry-to-workspace deployment route first, and then records every environment's target, achieved pose, error, and arm
joint configuration. This keeps gain/mass robustness and reset artifacts from being misreported as IK failures.

Preview the complete arm curriculum before training, with the Go2 root fixed and no checkpoint required:

```bash
python scripts/tools/preview_go2_d1_wbc_arm_curriculum.py \
  --task=Gurukul-Isaac-WBC-Flat-Unitree-Go2-D1-Arm-v0 \
  --num_envs 1 \
  --duration-s 120
```

The preview reads the task's real normalized curriculum boundaries, keeps the original random-arm command event and
3–5 second refresh interval, and applies its differential-IK and gripper references directly. Pick-place primitives
approach and descend with an open gripper, dwell at the pick pose to close before lifting, stay closed through transfer,
dwell at the place pose to release, then retreat open; other primitives
retain randomized open/close practice. The preview compresses training into two
minutes: the compact carry pose through 10%, stationary arm training from 10–25%, and combined-training arm commands
from 25–100%, with arm difficulty increasing linearly from 0 to 1 between 10% and 100%. The Go2 root remains fixed in
all three preview stages; in real stage 0 and stage 2 training, the policy also receives walking commands.

Inspect a trained WBC with the base held still and the full-difficulty scripted arm targets enabled:

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=Gurukul-Isaac-WBC-Flat-Unitree-Go2-D1-Arm-v0 \
  --agent=rsl_rl_cfg_entry_point \
  --load_run <wbc_run_folder_name> \
  --num_envs 1 \
  --loco-manip-stage arm \
  --loco-manip-arm-difficulty 1.0 \
  --print-rewards \
  --reward-print-interval 50 \
  --real-time
```

Compare the easiest and hardest curriculum settings side by side with two robots:

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=Gurukul-Isaac-LegWBC-AsyncArm-Flat-Unitree-Go2-D1-Arm-v0 \
  --agent=rsl_rl_cfg_entry_point \
  --load_run <leg_wbc_run_folder_name> \
  --num_envs 2 \
  --loco-manip-stage grid \
  --real-time
```

Grid mode orders robots from easiest to hardest and frames the complete environment grid in the viewer. With two
robots, environment 0 is pinned to stage 0 at difficulty 0: nominal posture, held arm, and zero added payload.
Environment 1 is pinned to stage 2 at difficulty 1: full pitch/height command range, full safe asynchronous arm-joint
envelope, and a payload sampled from 0 to 0.4 kg. With more than two robots, stage and difficulty slices are distributed
monotonically between those endpoints. The console prints every environment's assignment at startup.
To make the comparison unambiguous, every stage-0/stage-2 grid robot receives the same deterministic command. Eight
phases test maximum forward, backward, left, right, left yaw, and right yaw separately, followed by the positive and
negative combined corners of the configured command box. The hardest robot also alternates low/high pitch and height;
the easiest keeps nominal pitch/height, and roll remains fixed at zero. Each phase lasts 250 steps by default; override
it with `--loco-manip-grid-probe-steps`. Other play modes retain normal random command sampling.

Go2+D1 playback uses nominal mass, COM, actuator gains, joint/root reset, and disturbance settings by default so a
single visual rollout measures the policy rather than one random domain draw. The task's training arm-motion generator
remains active. Add `--go2-d1-play-domain-randomization` when explicitly testing robustness to the original
training randomization. Play-stage overrides preserve the task's trained base velocity and posture envelopes rather
than substituting generic locomotion ranges. For an unfinished checkpoint, set `--loco-manip-arm-difficulty` near the
difficulty reached by its training iteration; reserve `1.0` for a checkpoint trained through the end of the schedule.

For `LegWBC-AsyncArm`, the generator changes arm-joint goals every `4–6 s`. Every D1 joint gets its own linear or
smooth-step profile and a separately sampled `0.35–1.4 s` base duration. Every new joint trajectory also samples a
hardware-limit fraction from 35% through 98%; longer moves automatically stretch their duration to respect that
sampled limit. This deliberately mixes slow arm disturbances with motions near the D1 limit. Eighty percent of goals come from position-IK
solutions distributed over the boundary and interior of the configured Cartesian workspace; the remainder sample the
covered joint envelope uniformly. Difficulty scales that complete envelope around the folded carry pose, while
difficulty 0 commands the exact carry pose every step. The full WBC task continues to use its Cartesian workspace
generator. Let the inspection run for several samples; this checks disturbance rejection, not autonomous grasping.

Play D1 ArmMoving:

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=Gurukul-Isaac-Velocity-Rough-Unitree-Go2-D1-Arm-ArmMoving-v0 \
  --agent=rsl_rl_cfg_entry_point \
  --load_run <run_folder_name> \
  --loco-manip-stage combined \
  --real-time \
  --num_envs 1
```

Play one Go2+D1 WBC robot with live velocity and end-effector commands:

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=Gurukul-Isaac-WBC-Flat-Unitree-Go2-D1-Arm-v0 \
  --agent=rsl_rl_cfg_entry_point \
  --load_run <run_folder_name> \
  --go2-d1-live-control \
  --real-time
```

The flag uses a plugged-in Xbox controller when available and falls back to keyboard control when no controller is
detected. Xbox mapping: left stick drives base `x/y`, right stick `x` drives yaw, right stick `y` and D-pad up/down jog
EE `z`, D-pad left/right jogs EE `y`, `LB/RB` jog EE `x`, `A` stops the base, `X` resets EE, and `B/Y` close/open the
gripper. The live EE offset command starts and resets at `(0, 0, 0)`, anchored to the current reset EE pose. Keyboard
fallback uses normal base velocity keys plus `I/K`, `J/L`, and `U/O` for EE jogging. Live-control play resets the D1 arm
to the Go2+D1 asset default joint pose before anchoring the EE offset. The arm intentionally remains at that pose until
one of the EE jog controls is pressed; remove `--go2-d1-live-control` to use scripted targets instead.

## Task IDs

| Variant | Task ID | Action space | Run name |
| --- | --- | --- | --- |
| Flat WBC pretraining | `Gurukul-Isaac-WBC-Flat-Unitree-Go2-D1-Arm-v0` | 12 Go2 + 6 D1 + 2 gripper | `unitree_go2_d1_wbc_flat` |
| Flat WBC mjlab action scale | `Gurukul-Isaac-WBC-Flat-Unitree-Go2-D1-Arm-v0-MJLabActionScale` | 12 Go2 + 6 D1 + 2 gripper | `unitree_go2_d1_wbc_flat_mjlab_action_scale` |
| Rough WBC pretraining | `Gurukul-Isaac-WBC-Rough-Unitree-Go2-D1-Arm-v0` | 12 Go2 + 6 D1 + 2 gripper | `unitree_go2_d1_wbc_rough` |
| Flat leg WBC async arm | `Gurukul-Isaac-LegWBC-AsyncArm-Flat-Unitree-Go2-D1-Arm-v0` | 12 Go2 legs | `unitree_go2_d1_leg_wbc_async_arm_flat` |
| Rough leg WBC async arm | `Gurukul-Isaac-LegWBC-AsyncArm-Rough-Unitree-Go2-D1-Arm-v0` | 12 Go2 legs | `unitree_go2_d1_leg_wbc_async_arm_rough` |
| Flat WBC online APEX arm | `Gurukul-Isaac-WBC-ApexArm-Flat-Unitree-Go2-D1-Arm-v0` | 12 Go2 + 6 D1 + 2 gripper | `unitree_go2_d1_wbc_apex_arm_flat` |
| Rough WBC online APEX arm | `Gurukul-Isaac-WBC-ApexArm-Rough-Unitree-Go2-D1-Arm-v0` | 12 Go2 + 6 D1 + 2 gripper | `unitree_go2_d1_wbc_apex_arm_rough` |
| Hierarchical pick over frozen WBC | `Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-Pick-Wbc-Hierarchical-v0` | `vx, vy, wz, body_pitch, body_height, ee_x, ee_y, ee_z, gripper` | `unitree_go2_d1_pick_wbc_hierarchical_flat` |
| Hierarchical pick over frozen leg WBC | `Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-Pick-LegWbcArm-Hierarchical-v0` | `vx, vy, wz, body_pitch, body_height, arm_1..6, gripper` | `unitree_go2_d1_pick_leg_wbc_arm_hierarchical_flat` |
| Cartesian teacher pick over frozen leg WBC | `Gurukul-Isaac-LocoManip-Flat-Unitree-Go2-D1-Pick-LegWbcEe-Hierarchical-v0` | `vx, vy, wz, body_pitch, body_height, grasp_x, grasp_y, grasp_z, wrist_roll, gripper` | `unitree_go2_d1_pick_leg_wbc_ee_teacher_flat` |
| Flat policy-controlled arm | `Gurukul-Isaac-Velocity-Flat-Unitree-Go2-D1-Arm-v0` | 12 Go2 + 6 D1 | `go2_d1_arm_flat` |
| Rough policy-controlled arm | `Gurukul-Isaac-Velocity-Rough-Unitree-Go2-D1-Arm-v0` | 12 Go2 + 6 D1 | `go2_d1_arm_rough` |
| Flat ArmMoving | `Gurukul-Isaac-Velocity-Flat-Unitree-Go2-D1-Arm-ArmMoving-v0` | 12 Go2 | `go2_d1_arm_moving_flat` |
| Rough ArmMoving | `Gurukul-Isaac-Velocity-Rough-Unitree-Go2-D1-Arm-ArmMoving-v0` | 12 Go2 | `go2_d1_arm_moving_rough` |
