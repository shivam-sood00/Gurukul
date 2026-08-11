---
title: Training
---

# Training

Use this page for APEX task IDs, agent entry points, training commands, playback commands, and training-side notes.
Motion conversion lives in [Motion Data](motion-data). MuJoCo transfer lives in [Sim2Sim](sim2sim).

## Registered Environments

<details>
<summary>Show the 23 registered APEX task IDs</summary>

| Task ID | Purpose |
| --- | --- |
| `Gurukul-Isaac-Go2-APEX-Flat-v0` | Base motion-imitation PPO task. |
| `Gurukul-Isaac-Go2-APEX-Flat-Tracker-v0` | Reference-conditioned tracking with future-motion observations. |
| `Gurukul-Isaac-Go2-APEX-Flat-Privileged-Tracker-v0` | Full-state tracker teacher for distilling deployable trackers. |
| `Gurukul-Isaac-Go2-APEX-Flat-Tracker-One-Step-Future-v0` | Tracker ablation with current plus one future reference frame. |
| `Gurukul-Isaac-Go2-APEX-Flat-Tracker-One-Step-Future-History-v0` | Deployable one-step tracker student with 5-frame observation/reference history. |
| `Gurukul-Isaac-Go2-D1-Arm-APEX-Flat-Tracker-v0` | Normal direct APEX PPO task: deployable actor observations, asymmetric critic, and original DecAP schedule. |
| `Gurukul-Isaac-Go2-D1-Arm-APEX-Distillation-Student-v0` | Separate zero-DecAP deployable student, trained only by supervised teacher distillation. |
| `Gurukul-Isaac-Go2-D1-Arm-APEX-Original-DecAP-Teacher-v0` | Privileged 19-output Go2+D1 teacher with original additive DecAP, a 50 Hz policy, and a 10 Hz held D1 command packet. |
| `Gurukul-Isaac-Go2-D1-Arm-APEX-Pick-Stow-Carry-Flat-Tracker-v0` | Contact-aware 19-action Go2+D1 walk-pick-stow-carry-place tracker. |
| `Gurukul-Isaac-Go2-D1-Arm-APEX-Can-Pick-Carry-Drop-Flat-Tracker-v0` | Top-down floor-can pickup, walking carry, and gravity-driven drop. |
| `Gurukul-Isaac-Go2-D1-Arm-APEX-Pick-Stow-Carry-Robot-Only-Flat-Tracker-v0` | Object-free APEX imitation of the same Go2+D1 robot trajectory. |
| `Gurukul-Isaac-Go2-D1-Arm-APEX-Pick-Stow-Carry-Privileged-Teacher-v0` | Separate full-state manipulation teacher with contact forces and extended temporal references. |
| `Gurukul-Isaac-Go2-D1-Arm-APEX-Pick-Stow-Carry-Distillation-Student-v0` | Zero-DecAP deployable manipulation student paired with the separate teacher. |
| `Gurukul-Isaac-B2-Z1-Arm-APEX-Flat-Tracker-v0` | B2+Z1 tracker for B2 leg plus Z1 arm reference joints and end-effector tracking. |
| `Gurukul-Isaac-B2-Z1-Arm-APEX-Flat-Privileged-Tracker-v0` | Full-state B2+Z1 tracker teacher for distillation. |
| `Gurukul-Isaac-B2-Z1-Arm-APEX-Flat-Tracker-One-Step-Future-v0` | B2+Z1 one-step-future tracker student. |
| `Gurukul-Isaac-B2-Z1-Arm-APEX-Flat-Tracker-One-Step-Future-History-v0` | Deployable B2+Z1 one-step tracker student with 5-frame observation/reference history. |
| `Gurukul-Isaac-B2-Z1-Arm-APEX-Flat-Fixed-Wrist-Gripper-Tracker-v0` | B2+Z1 tracker with the fixed-wrist gripper contract and deployable actor inputs. |
| `Gurukul-Isaac-B2-Z1-Arm-APEX-Flat-Fixed-Wrist-Gripper-Privileged-Tracker-v0` | Privileged teacher paired with the fixed-wrist gripper tracker. |
| `Gurukul-Isaac-Go2-APEX-Flat-Depth-Distill-v0` | Depth-camera student distillation. |
| `Gurukul-Isaac-Go2-APEX-Flat-Depth-Action-Prior-v0` | PPO student with teacher action priors. |
| `Gurukul-Isaac-BeyondMimic-APEX-Flat-Unitree-G1-v0` | G1 BeyondMimic tracker with APEX-style future reference observations. |
| `Gurukul-Isaac-G1-APEX-Flat-Tracker-v0` | Short G1 APEX tracker alias over the same BeyondMimic config. |

</details>

## Agents

| Entry point | Use |
| --- | --- |
| `rsl_rl_cfg_entry_point` | PPO baseline or task default. |
| `rsl_rl_teacher_cfg_entry_point` | Privileged teacher. |
| `rsl_rl_distillation_cfg_entry_point` | Student distillation. |
| `rsl_rl_decap_cfg_entry_point` | DecAP-style policy. |
| `rsl_rl_multi_critic_cfg_entry_point` | Grouped-critic variant. |

For `Gurukul-Isaac-Go2-APEX-Flat-Depth-Action-Prior-v0`, `rsl_rl_cfg_entry_point` launches the depth
action-prior DecAP runner by design.

Run the grouped two-head critic with:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Go2-APEX-Flat-v0 \
  --num_envs 16384 \
  --logger wandb \
  --agent=rsl_rl_multi_critic_cfg_entry_point \
  --headless
```

The custom runner uses the RSL-RL 5.3 actor/critic/storage interface and optimizes the shared multi-head policy once.

## Base PPO

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Go2-APEX-Flat-v0 \
  --num_envs 16384 \
  --logger wandb \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --run_name go2_apex_flat_ppo
```

## Tracker PPO

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Go2-APEX-Flat-Tracker-v0 \
  --num_envs 16384 \
  --logger wandb \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --run_name go2_apex_flat_tracker
```

Train the normal Go2+D1 policy directly with APEX PPO and deployable actor observations:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Go2-D1-Arm-APEX-Flat-Tracker-v0 \
  --motion-file source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/npz/go2_d1/wave_hello.npz \
  --headless \
  --num_envs 16384 \
  --max_iterations 5000 \
  --logger wandb \
  --wandb_project_name Gurukul-go2-d1 \
  --agent=rsl_rl_cfg_entry_point \
  --run_name go2_d1_wave_hi_direct_apex
```

This is the normal experiment. It is direct PPO, not behavior cloning: the actor receives the deployable `policy`
group, the critic receives the simulator-only `critic` group, and original additive DecAP decays from `1.0` to `0.0`.

Teacher-to-student distillation is a separate two-task pipeline. First train the privileged teacher:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Go2-D1-Arm-APEX-Original-DecAP-Teacher-v0 \
  --motion-file source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/npz/go2_d1/wave_hello.npz \
  --headless \
  --num_envs 16384 \
  --max_iterations 5000 \
  --logger wandb \
  --wandb_project_name Gurukul-go2-d1 \
  --agent=rsl_rl_cfg_entry_point \
  --run_name go2_d1_wave_hi_original_decap_teacher
```

Then supervise the separate deployable student from that teacher:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Go2-D1-Arm-APEX-Distillation-Student-v0 \
  --motion-file source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/npz/go2_d1/wave_hello.npz \
  --headless \
  --num_envs 16384 \
  --max_iterations 5000 \
  --logger wandb \
  --wandb_project_name Gurukul-go2-d1 \
  --agent=rsl_rl_distillation_cfg_entry_point \
  --teacher_load_run <original_decap_teacher_run_folder_name> \
  --run_name go2_d1_wave_hi_distilled
```

At startup, every RSL-RL training task writes the resolved manager schema to
`<run>/params/task_setup.json`. When `--logger wandb` is selected, the same `task_setup` object is stored in the W&B
run configuration. It lists the runner's actor and critic group mappings, every observation term with its resolved
shape, and every configured reward with its weight and enabled state. Motion-based tasks also record the configured
source, resolved file list, per-file frame count, frame rate, duration, size, SHA-256 digest, and an aggregate dataset
digest so data ablations remain distinguishable if a local dataset directory is later replaced. The W&B training
configuration also records the exact registered task ID.

Train on the walk-then-wave motion with its frame-aligned `skill` observation (`0=walk/settle`, `1=hello`):

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Go2-D1-Arm-APEX-Flat-Tracker-v0 \
  --motion-file source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/npz/go2_d1/walk_then_wave_hello.npz \
  --num_envs 16384 \
  --logger wandb \
  --headless \
  --agent=rsl_rl_cfg_entry_point \
  --run_name go2_d1_apex_walk_then_wave_hello
```

Train the deployable direct-PPO policy on the student showcase motion:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Go2-D1-Arm-APEX-Flat-Tracker-v0 \
  --motion-file source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/npz/go2_d1/student_showcase.npz \
  --headless \
  --num_envs 16384 \
  --max_iterations 5000 \
  --logger wandb \
  --wandb_project_name Gurukul-go2-d1 \
  --agent=rsl_rl_cfg_entry_point \
  --run_name go2_d1_student_showcase_direct_apex
```

All three task surfaces produce 19 absolute position targets at `50 Hz`: 12 Go2 legs, six D1 rotary joints, and one
physical gripper coordinate. The seven D1 targets cross a `10 Hz` zero-order-hold boundary before reaching the D1
plant; only the Go2 targets remain directly applied at `50 Hz`. The adapter adds no assumed quantization, latency, or
firmware response. The normal direct-PPO task
and privileged teacher apply the original DecAP equation
`q_target = q_policy + lambda * (q_reference - q_robot)` to only the original 18 leg-and-rotary-arm channels. The
gripper is the independently supervised 19th output and never receives the prior. Lambda remains `1.0` for the first
100 iterations and decays with a cosine schedule to `0.0` at iteration 1,000.

The actor observation, reference phase, and policy inference retain the `20 ms` boundary. Isaac evaluates physics and
the held PD targets at `200 Hz` (`dt=0.005`, decimation `4`); J1--J6 and the physical gripper latch the newest policy
target every fifth policy step, matching the D1's `100 ms` seven-angle command period.

The direct policy, teacher, and student share the same explicit, torque-limited, MuJoCo-parity D1 plant, gain and mount
randomization, filtered self-collisions, and 19-output ordering. The teacher's actor and critic both use the clean
`privileged` observation group. The normal PPO actor and supervised student use deployable position-only arm
observations. The student task explicitly configures DecAP to zero; choosing the distillation runner is not what defines
its action semantics.

Let the teacher finish all `5,000` iterations; iteration 1,000 is only the end of the DecAP transition. Start both stages
fresh. Checkpoints trained with the former permanent `reference + residual` or target-blend action contracts have the
same tensor dimensions but incompatible action semantics.

The rotary-arm posture reward is split into independent per-joint kernels: J1--J3 use a `0.12 rad` kernel and J4--J6
use a tighter `0.08 rad` kernel with higher weight. A bad wrist joint therefore cannot zero the learning signal for the
other five joints. Pre-prior policy-target supervision is weighted `-0.25` for J1--J3 and `-0.75` for J4--J6, and a
`Link6` orientation term complements the gripper-center position target.

APEX enables physical D1 self-collisions. Parent-child pairs are filtered in the USD, while six per-link contact
sensors cover each unique non-adjacent D1 pair and apply a penalty above `5 N`. The two fingers retain their normal
closing motion and object contacts. Link2--Link4 and Link2--Link5 are also filtered because the coarse collision
proxies overlap in the valid ready pose and hello reference.

After replaying and visually checking the clip, train the dedicated pick-stow-carry-place tracker:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Go2-D1-Arm-APEX-Pick-Stow-Carry-Flat-Tracker-v0 \
  --headless \
  --zero_decap \
  --num_envs 16384 \
  --max_iterations 5000 \
  --logger wandb \
  --wandb_project_name Gurukul-go2-d1 \
  --agent=rsl_rl_cfg_entry_point \
  --run_name go2_d1_apex_pick_stow_carry_small_box_zero_decap_v1
```

This command fixes the environment-side DecAP contribution at zero. Always start a fresh run after changing object
geometry or base-tracking rewards; old checkpoints have already optimized a different physical task.

Train the shorter top-down can pickup, carry walk, and drop clip with:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Go2-D1-Arm-APEX-Can-Pick-Carry-Drop-Flat-Tracker-v0 \
  --headless \
  --zero_decap \
  --num_envs 16384 \
  --max_iterations 5000 \
  --logger wandb \
  --wandb_project_name Gurukul-go2-d1 \
  --agent=rsl_rl_cfg_entry_point \
  --run_name go2_d1_can_pick_carry_drop_zero_decap_v1
```

This task uses `top_down_can_pick_and_move.npz`: the robot picks a floor can from above, walks `1.04 m`, settles,
opens the gripper, and tracks the final `0.2 s` gravity-driven drop. Training randomizes can diameter and height by
`0.92–1.08×`, mass over `0.18–0.32 kg`, contact friction and restitution, and the initial XY position by `±10 mm`.
The cylinder stays grounded when its sampled scale changes; unlike the side-grasped box, it is not shifted toward a
lateral grasp face.

To learn only the demonstrated Go2+D1 motion, with no object learning surface, use the separate robot-only task:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Go2-D1-Arm-APEX-Pick-Stow-Carry-Robot-Only-Flat-Tracker-v0 \
  --headless \
  --num_envs 16384 \
  --max_iterations 5000 \
  --logger wandb \
  --wandb_project_name Gurukul-go2-d1 \
  --agent=rsl_rl_cfg_entry_point \
  --run_name go2_d1_pick_stow_carry_robot_only_v1
```

This task loads `pick_stow_carry_robot_only.npz`, which contains the same base/body, leg, arm, end-effector, continuous
gripper, command, skill, and phase data as the manipulation clip but has every `object_*` channel removed. Isaac spawns
no object and registers no object sensors, resets, observations, rewards, or terminations. The actor has 206 inputs,
the asymmetric critic has 842, and the action remains the same 19-coordinate Go2+D1 interface. Half the environments
start at frame zero and half sample a uniform motion frame. The task retains the original APEX DecAP schedule and the
dense world-base tracking fix; add `--zero_decap` only for a separate no-prior experiment.

Start with `--num_envs 64` for a smoke run before scaling the environment count. The 46.82-second reference contains
2,341 frames and a nominal `135 × 55 × 255 mm` rectangular box. The physical training scene samples smaller boxes and
lower masses as described below. The robot starts at the environment origin, walks `1.5297 m` to the box, and reaches
`4.0526 m` during carry. The scene uses 12 m environment spacing.

Half of the episodes begin at frame zero, `20%` begin at a uniform trajectory frame, and `30%` begin during an attached
phase. Random-phase resets initialize the complete robot reference state and the corresponding box pose, linear
velocity, and angular velocity. The object transform is anchored to the root pose just written to simulation instead
of a possibly stale articulation buffer. Root reference velocities are rotated with sampled pose perturbations.
Attached resets use the demonstrated `0.00565 m` jaw coordinate, then the binary endpoint drive maintains contact.
Each episode ends at its selected clip boundary.

Bilateral finger contact becomes a task contract during demonstrated attached frames (`640–2039`). It is disabled
through PPO iteration `250`, then its continuous no-contact allowance tightens linearly from `2.0 s` to `0.5 s` by
iteration `1000`. Exceeding that allowance terminates the episode with the normal `-200` failure cost. Contact at
either an approach frame or after the demonstrated release is not required, and any restored bilateral contact clears
the timer. The schedule uses the active rollout length and checkpoint iteration, so resumed training and playback do
not restart the warmup.

The task controls 18 leg-and-arm joints plus the D1's single physical gripper servo. Object position uses exponential
and capped Huber terms; box tilt and linear velocity are tracked separately. Attachment gates bilateral contact and
gripper-to-box offset supervision. Object terms have full weight while attached and `0.1×` weight while detached.
Hip contact is penalized but not terminal; base collision or excessive anchor height/tilt ends a genuine fall. A
robust world-base XY cost prevents the policy from stopping early and exploiting the long stationary phase.

The box uses independent `15–40 g` mass randomization, friction randomization, `±2.5 mm` horizontal reset jitter,
`±0.05 rad` yaw jitter, and per-environment isotropic `0.70–0.90×` size scaling. This yields physical boxes from
`94.5 × 38.5 × 178.5 mm` to `121.5 × 49.5 × 229.5 mm`. The matching grounded-center correction is applied to reset,
current reference, and future-reference trajectories. The grasp-facing `-X` plane is fixed as depth changes, avoiding
a reach mismatch with the demonstrated fingers. The actor's 19th output is a binary logit (`> 0` closes); the physical
endpoint is `0.033 m`, while attached-state initialization uses the recorded contact coordinate. J2 and J3 action
scales are `0.50 rad` and `0.35 rad`; the other arm joints use `0.25 rad`, keeping the demonstration within the
runner's normalized `±6.6` bound.

The new ground-grasp reference was swept against the exact D1 collision meshes and Go2 collision proxies. Link4,
Link5, and Link6 cross the conservative `Head_upper` cylinder during pickup/place, and Link6 crosses the placed box
during retraction. These four pairs are filtered only for this task. All other self-collisions, ground contacts, and
both finger-object contacts remain active. In particular, object/head collision is not filtered. The bilateral-contact
contract prevents the policy from collecting robot-imitation return while simply leaving the physical box behind. The
normal task, privileged teacher, and distillation student share this scene configuration.

The second jaw follows through the USD mimic constraint and does not consume another policy action. Isaac uses the
combined Unitree hardware URDF jaw anchors and exact Viser finger render/collision meshes. Both jaw coordinates
increase in the closing direction. The configured reset arm pose is `[0, -90, 90, 0, 0, 0] deg`.

The measured D1 servo endpoints are `68.6 deg` fully open and `-28.7 deg` fully closed. The shared Go2 APEX constants
provide a clamped affine conversion between those hardware angles and per-jaw simulation travel (`0--0.033 m`).
The Go2+D1 asset uses a `1.0` soft joint-limit factor so neither measured gripper endpoint is contracted away.
Two endpoint measurements determine only an affine calibration; collect intermediate servo-angle/jaw-gap pairs if a
more accurate linkage model is needed.

### Privileged manipulation teacher

Train the separate manipulation teacher after validating the corrected data and base-tracking rewards:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Go2-D1-Arm-APEX-Pick-Stow-Carry-Privileged-Teacher-v0 \
  --headless \
  --num_envs 16384 \
  --max_iterations 5000 \
  --logger wandb \
  --wandb_project_name Gurukul-go2-d1 \
  --agent=rsl_rl_cfg_entry_point \
  --run_name go2_d1_pick_stow_carry_tall_box_privileged_teacher_v1
```

This is a distinct task: both teacher actor and critic consume the `privileged` group, while the normal pick task's
actor remains on the 238-value deployable `policy` group. The teacher receives five flattened 50 Hz state snapshots
(an 80 ms history span), clean robot pose/velocity/torque and foot-force state, full object position, 6D orientation,
linear/angular velocity, randomized mass and isotropic size scale, continuous left/right object-contact force vectors, the
poses of Link6 and both fingers, and explicit per-finger/gripper-center vectors and distances to the box. Its reference
bundle covers frames `(-20, -10, -5, -2, 0, 1, 2, 5, 10, 20, 40)`, or `-0.4 s` through `+0.8 s`, for robot motion,
feet/body state, object position/orientation/attachment, and gripper-center pose. The teacher retains the original
DecAP schedule (`1 → 0` through iteration 1,000) to make the long manipulation sequence learnable.

After the teacher succeeds, distill its 19 direct actions into the separate zero-DecAP student:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Go2-D1-Arm-APEX-Pick-Stow-Carry-Distillation-Student-v0 \
  --headless \
  --num_envs 16384 \
  --max_iterations 5000 \
  --logger wandb \
  --agent=rsl_rl_distillation_cfg_entry_point \
  --teacher_load_run <privileged_teacher_run_folder_name> \
  --run_name go2_d1_pick_stow_carry_distilled_student_v1
```

The distillation environment recreates the teacher's privileged 7,435-value input exactly, but the student sees only
the 238-value deployable policy input and executes with DecAP fixed at zero.

The Go2+D1 tracker uses the shared APEX exploration settings (`1.0` initial policy noise and `0.01` entropy weight) so
the legs continue exploring locomotion. Its normalized action clip is `6.0` because the motion reaches `+5.21` on D1
joint 2 and `-4.68` on joint 3 with the configured `0.25 rad` action scale. Checkpoints trained with a `4.0` action clip
or `0.35` global initial noise use incompatible exploration settings and should not be resumed for this configuration.

## History Distillation

Distill the Go2 deployable one-step history tracker from a privileged tracker teacher:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Go2-APEX-Flat-Tracker-One-Step-Future-History-v0 \
  --num_envs 16384 \
  --logger wandb \
  --headless \
  --agent=rsl_rl_distillation_cfg_entry_point \
  --teacher_load_run <privileged_tracker_teacher_run_folder_name> \
  --run_name go2_apex_flat_tracker_one_step_future_history_distill
```

Train the B2+Z1 privileged tracker teacher and history student:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-B2-Z1-Arm-APEX-Flat-Privileged-Tracker-v0 \
  --num_envs 16384 \
  --logger wandb \
  --headless \
  --run_name b2_z1_arm_apex_flat_privileged_tracker
```

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-B2-Z1-Arm-APEX-Flat-Tracker-One-Step-Future-History-v0 \
  --headless \
  --num_envs 16384 \
  --logger wandb \
  --agent=rsl_rl_distillation_cfg_entry_point \
  --teacher_load_run <b2_z1_privileged_tracker_teacher_run_folder_name> \
  --run_name b2_z1_arm_apex_flat_tracker_one_step_future_history_distill
```

Use `--teacher_checkpoint logs/rsl_rl/<experiment>/<run>/model_<iter>.pt` when a student should use a pinned teacher
checkpoint.

## Playback

APEX playback enables imitation-data visualization by default. Use `--motion-velocity-vis-only` to hide reference-foot
spheres.

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task=Gurukul-Isaac-Go2-APEX-Flat-Tracker-v0 \
  --agent=rsl_rl_cfg_entry_point \
  --load_run <run_folder_name> \
  --num_envs 1 \
  --keyboard
```

In tracker playback, press `N` for the next reference motion and `P` for the previous one. Use
`--camera_follow_mode mouse` to keep normal viewport mouse control while keeping motion switching active.
Tasks configured to begin at the start of a motion now reset to frame zero during playback, including the dedicated
pick-stow-carry task. Passing an explicit `--motion-file` also enables this behavior. Add
`--motion-reset-curriculum` only when you intentionally want to inspect the randomized training reset phases; training
itself retains the configured randomized/adaptive resets.

## Training Notes

<details>
<summary>Open task-specific implementation notes</summary>

- Go2 tracker actor command observations are `[vx, vy, vz, yaw_rate]`; `vz` comes from the reference base linear
  velocity and prevents vertical motions from being treated as stand-still default-stance frames.
- Go2 tracker training currently keeps `reference_state_init=False`, so resets start from the deterministic Go2 reset
  state while adaptive sampling focuses hard reference phases through the command timeline.
- The shared APEX reward set includes yaw-aligned local foot imitation plus world-frame base and foot-position rewards.
  The world-frame terms are weighted toward vertical tracking so jump takeoff, flight, and landing height are supervised
  directly from the motion NPZ.
- Enriched motion NPZs can include simulator-derived `reference_foot_contact` and `reference_airborne` channels from
  `scripts/tools/go2_apex/replay_npz.py --export-replayed-motion-dir`; the reward set consumes those labels when
  present.
- `tracker_reference_params(..., add_noise=True, noise_std=0.01)` injects small Gaussian noise into the reference
  observation while keeping rewards, resets, and tracking metrics tied to the clean imitation clip.
- The privileged tracker teacher uses the `privileged` observation group for both actor and critic. It includes clean
  proprioception, base velocity, phase, clean reference motion with joint velocities and command velocity, reference foot
  and base pose, robot body pose/orientation features, applied joint torques, foot contact state, and a one-hot motion
  ID.
- Normal Go2+D1 PPO is asymmetric around the hardware interface. Its actor observes and commands 19 positions (including the
  single physical gripper-servo position), retains 19 previous position targets and the 12 Go2 leg velocities, but no D1 arm
  velocity, torque, or contact-force channels.
  Its critic uses the simulator-only full-state group: all Go2+D1 joint velocities and applied torques, base/body and
  reference state, binary foot contacts, and normalized foot contact-force vectors in the robot base frame.
- The intended Go2 distillation student is `Gurukul-Isaac-Go2-APEX-Flat-Tracker-One-Step-Future-History-v0`.
  RSL-RL supervised distillation trains its deployable 5-frame history `policy` group with MSE behavior cloning on
  teacher actions. Env-side DecAP is disabled automatically for supervised distillation, so `Metrics/decap/factor`
  should stay at `0.0`.
- The Go2+D1 distillation student is the separate
  `Gurukul-Isaac-Go2-D1-Arm-APEX-Distillation-Student-v0` task. It uses the same deployable actor
  observations as normal direct PPO, but its actions are supervised from the privileged teacher and its environment
  explicitly fixes DecAP at zero.
- The tracker uses reduced domain randomization for imitation quality: mild reset force/torque, narrower
  mass/friction/COM randomization, and smaller pushes every `12-18 s`. All controlled Go2+D1 actuator gains are scaled
  independently in `(0.9, 1.1)` around their nominal values during Go2+D1 hello-motion training. The tuned D1 gains
  therefore remain the center of the sim2real randomization range.
- Go2 and B2+Z1 tracker configs allow backward x commands down to `-1.0 m/s`; their RSL-RL tracker runners and MuJoCo
  tracker YAMLs leave policy action clipping disabled.
- B2+Z1 APEX splits joint acceleration, torque, action-rate, and action-smoothness penalties into leg and arm reward
  terms. It also tracks finite-difference Z1 `link06` end-effector linear velocity when `arm_ee_pos_w` is available.
- B2+Z1 APEX uses a smaller Z1 joint-position action scale of `0.05 rad` for `joint1` through `joint6` and stronger
  arm-only acceleration, torque, action-rate, and action-smoothness penalties to reduce jerky arm targets.
- Go2+D1 APEX splits stationary leg and arm tracking so zero-velocity manipulation clips continue to follow their D1
  joint and gripper-center references. Normal direct PPO and the privileged teacher apply original additive DecAP to
  the 12 legs and six rotary arm joints, exclude the independently supervised physical gripper action, and decay the
  prior completely. The separate supervised student uses no DecAP. The D1's two simulated prismatic jaws are coupled as one physical gripper DOF;
  clips without a gripper-servo channel hold its fully-open default jaw position. The Isaac and MuJoCo jaw constraints
  use the same hardware-positive convention, so positive travel closes both fingers. Rotary actuator limits follow the
  D1-550 joint groups:
  J1-J3 use `3.3 Nm, 1.05 rad/s`, and J4-J6 use `1.7 Nm, 1.73 rad/s`. APEX models the D1's internal
  joint-position servo with `Kp=200`, `Kd=5` for J1-J2 and `Kp=200`, `Kd=4` for J3. J4-J6 use `Kp=50`, `Kd=0.25`.
  The APEX Isaac plant also matches MuJoCo's solver-side D1 dynamics: `0.001 kg m^2` joint armature, `0.1` viscous
  damping, and `0.02 Nm` static/dynamic Coulomb friction. These passive terms are distinct from controller Kd and are necessary to
  prevent the low-inertia distal joints from becoming a numerically stiff explicit-PD system. The rotary PD computation is explicit and
  effort-limited; the gripper uses `Kp=200`, `Kd=3`.
  These torque-limited simulator gains replace the old
  `4000/400` APEX override, which saturated at sub-degree errors. They are a stable deployment baseline, not identified
  firmware gains; manipulation transfer should re-identify them from measured hardware step responses and
  payload/contact tests.
  The APEX overlay restores the D1-550's published `3.152 kg` assembly mass: per-link masses use a SolidWorks
  component export totaling `2.941 kg`, the remaining `0.211 kg` is assigned to the fixed arm base, and the original
  Link1--Link3 inertia tensors are scaled by their mass ratios. Link4--Link6 use the complete tensors from the
  [tactile-aware loco-manipulation D1 asset](https://github.com/PokuangZhou/tactile-aware-quadrupedal-loco-manipulation/tree/285b3a50c0538bdfb15fddd12f37589e81fc5564),
  transformed into the imported USD body frames; see also the associated
  [paper](https://arxiv.org/html/2604.27224). That asset models its larger UMI fingers and tactile pads as separate
  child bodies, so the wrist-chain tensors remain applicable to this gripper; this asset retains its own finger masses
  and inertias.
  Its APEX-only asset exposes a non-action fore-aft mount calibration joint; every reset locks that joint to a sampled
  offset in `[-5, 5] mm`. Go2+D1 uses each NPZ's command channels directly rather than adding the generic Go2 tracker's
  random forward-velocity offset, so stationary manipulation clips remain stationary. Action-rate and
  second-difference reward penalties remain regularizers for all controlled joints. Teacher, student, sim2sim, and
  hardware keep `50 Hz` policy inference while latching the seven D1 packet targets at `10 Hz`. The APEX USD reverses
  the source USD's J1 and J6 joint frames so their positive directions
  match the Unitree/combined hardware URDF (`J1: -Z`, `J6: +X`). The bundled end-effector paths are regenerated from this
  corrected articulation, avoiding a conflicting FK target during training. Playback caches the reference used by the
  completed physics step before advancing the command phase, keeping the end-effector target marker synchronized with
  the robot. Velocity debug markers are fully hidden below `0.001 m/s` instead of collapsing into colored discs.
- B2+Z1 APEX overrides the shared Go2 tracker domain randomization for the larger platform: base mass is scaled by
  `(0.9, 1.15)`, non-base bodies by `(0.85, 1.15)`, base COM by `x=[-0.04, 0.04] m`,
  `y,z=[-0.03, 0.03] m`, reset force by `[-40, 40] N`, reset torque by `[-20, 20] Nm`, and actuator
  stiffness/damping by `(0.85, 1.15)`.
- All B2+Z1 APEX variants include locomotion stabilizers on top of motion imitation: leg joint-limit penalties,
  feet-air-time reward and variance penalty, `legs_distance` below `0.18 m`, and high foot-contact-force penalty.
- Depth variants use Isaac Lab training and playback paths; task-specific MuJoCo YAMLs are not provided.
- G1 APEX uses a locally generated G1 BVH-derived NPZ from the BeyondMimic task tree, not the Go2 APEX motion dataset;
  follow the [G1 BeyondMimic motion setup](../beyondmimic/g1) first.

</details>
