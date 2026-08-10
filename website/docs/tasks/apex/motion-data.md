---
title: Motion Data
---

# Motion Data

APEX motion assets live under:

```text
source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/
```

The public repository keeps the project-authored Go2+D1 demonstrations but omits uncleared third-party motion data.
Install only source clips whose terms cover your use.

Training and playback consume converted NPZ clips. The default flat-task clip is:

```text
source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/npz/animal_mocap/go2_retarget_trot.npz
```

The motion loader accepts a single NPZ file, a directory, a glob pattern, or an explicit
`commands.motion.motion_files` tuple.

## External Motion Sources

| Motion family | Authoritative source | Expected local area |
| --- | --- | --- |
| APEX example clips | [Official APEX repository](https://github.com/marmotlab/APEX) | `imitation_data/` |
| Cross-morphology motions | [X-Morph project](https://maker-rat.github.io/morph/) | `imitation_data/cross_morpho/` or a user-selected input directory |
| Walk These Ways | [Official repository](https://github.com/Improbable-AI/walk-these-ways) | A user-selected input directory |
| Unitree-retargeted LAFAN1 | [Dataset page](https://huggingface.co/datasets/lvhaidong/LAFAN1_Retargeting_Dataset) | A user-selected input directory |

Paths in the table are relative to the `go2_apex/config/go2/motion/` directory shown above. Convert permitted CSVs
into the corresponding local `npz/` directories with the commands below. These third-party data paths are ignored by
Git; only the Gurukul-authored `imitation_data/go2_d1_motions_csv/` and `npz/go2_d1/` branches are published.

The LAFAN1 dataset page identifies CC BY-NC-ND terms for the underlying motion data. Confirm that your use and any
derived-file handling are permitted before downloading or converting it.

:::caution[Provenance before redistribution]
Motion data is licensed separately from the loader code. The archive currently mixes official APEX examples,
retargeted clips, and locally generated demonstrations. The project maintainer has confirmed authorship of the
Go2+D1 directories listed in `motion/GO2_D1_DATA_PROVENANCE.md`; those files are included. For every other group,
record its source, author, source-data license, generation or retargeting tool, and redistribution permission. The
`cross_morpho`/`corss_morpho_new`, `learn_diverse_quad_loco`, and inherited G1 AMP groups are not cleared by this
documentation. Exclude any group whose rights cannot be confirmed. See
[Acknowledgements](../../reference/credits) and the repository's third-party notices.
:::

## Layout

<details>
<summary>Open the archive layout and motion-field reference</summary>

| Data | Path | Notes |
| --- | --- | --- |
| Raw CSVs | `imitation_data/` | Source APEX-style motion files. |
| Converted NPZs | `npz/` | Gurukul motion format used by training and playback. |
| Go2+D1 tracker NPZs | `npz/go2_d1/` | Clips for the Go2+D1 trackers, including hello-wave and pick-stow-carry-place sequences. |
| B2+Z1 tracker NPZs | `npz/b2_z1_motions/` | Clips for B2+Z1 tracker tasks. |
| CrossMorpho replay clips | `npz/cross_morpho_replay_added/` | Replay-enriched Go2 tracker clips. |
| LearnDiverseQuadLoco NPZs | `npz/learn_diverse_quad_loco/{train,test,combined}` | Converted quadruped clips. |

NPZ files contain the keys used by `go2_apex/mdp/commands.py`: `fps`, `joint_pos`, `joint_vel`,
`command_lin_vel_xy`, `command_ang_vel_z`, `body_pos_w`, `body_quat_w`, `body_lin_vel_w`, and `body_ang_vel_w`.
The canonical body order is `base`, `FL_foot`, `FR_foot`, `RL_foot`, `RR_foot`.

Go2+D1 tracker clips must include the 12 Go2 leg joints plus `arm_1_joint` through `arm_6_joint`. B2+Z1 clips use the
12 B2 leg joints plus Z1 `joint1` through `joint6`. When `arm_ee_pos_w` is present, replay tools draw the arm
end-effector target.

Go2+D1 orientation tracking uses the `Link6` wrist frame. New CSV conversions store its WXYZ world quaternion as
`arm_ee_quat_w`. Existing D1 clips remain compatible: the motion loader derives the same quaternion from the six
hardware-convention arm angles and the reference base orientation.

The D1 hardware exposes one gripper-servo command. Its simulation geometry uses two prismatic jaw joints, with
`arm_7_2_joint` coupled to the driven `arm_7_1_joint`. The bundled hello clips do not contain a gripper-servo channel,
so they hold the gripper at its default opening. Their gripper-center path is derived by forward kinematics from the six
arm joints and stored with `arm_ee_frame=gripper_midpoint:Link7_1,Link7_2`.

`wave_hello.npz` is a 10-second, 50 Hz stationary-base hello wave generated with EmbodiK. It includes the 18 tracked
joint channels, the D1 end-effector path, stationary foot contacts, and the canonical base-plus-feet body metadata.

`walk_then_wave_hello.npz` is a 19-second, 50 Hz walk-settle-wave sequence. Its frame-aligned `skill` channel has shape
`(T, 1)`: `0` through walking and settling, then `1` for the hello-wave segment. Go2+D1 APEX exposes this scalar to the
actor and mirrors it into the critic observation groups. Older APEX clips without `skill` receive a backward-compatible
value of `0`.

`student_showcase.npz` is an 18.62-second, 50 Hz walk-settle-showcase sequence with bow, dance, flourish, and finale
phase metadata. It uses the deployable Go2+D1 tracker schema and holds the physical gripper fully open throughout.

`pick_stow_carry.npz` is a 46.82-second, 50 Hz walk-pick-stow-carry-place sequence for the dedicated
`Gurukul-Isaac-Go2-D1-Arm-APEX-Pick-Stow-Carry-Flat-Tracker-v0` task. It contains the 18 tracked robot joints,
frame-aligned locomotion/manipulation labels (`0` and `2`), the D1 gripper-center path, and the rectangular
`tall_grasp_box` trajectory with grasp and release flags. The box is `135 × 55 × 255 mm` and begins at
`[2.0321982, 0, 0.1275] m`. The replay tool applies both raw jaw channels and renders the reference geometry as a red
cuboid. The dedicated contact-aware task controls 19 physical actuators: 12 legs, 6 arm joints, and
`arm_7_1_joint`; the USD mimic constraint drives `arm_7_2_joint`. Training consumes the attachment flag directly:
bilateral finger contact and the demonstrated gripper-to-box offset are supervised only during attached frames. It
also tracks the box position, up axis, and finite-difference linear velocity and exposes current/future reference box
positions and attachment phases to the actor. Object tracking is phase weighted: attached frames receive full credit,
while detached approach and placement frames receive only `0.1×`, so an object left on the ground cannot dominate return.

`top_down_can_pick_and_move.npz` is a 22.82-second, 50 Hz top-down floor-can pickup, carry walk, and drop sequence for
`Gurukul-Isaac-Go2-D1-Arm-APEX-Can-Pick-Carry-Drop-Flat-Tracker-v0`. The dynamic object is a vertical
`53 mm × 135 mm` cylinder starting at `[0.48, 0, 0.0675] m`. Attachment begins at frame `390`, the robot walks
`1.04 m` while carrying it, the gripper opens over frames `1090–1130`, and the last ten frames record `0.2 s` of
gravity-driven free fall.

`pick_stow_carry_robot_only.npz` is a mechanically derived copy for the object-free APEX tracker. All robot trajectory,
end-effector, continuous gripper, command, skill, contact-label, and phase-marker arrays are
byte-for-byte equal to `pick_stow_carry.npz`; every `object_*` array is absent. Rebuild it with:

```bash
python scripts/tools/go2_apex/strip_object_channels.py \
  source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/npz/go2_d1/pick_stow_carry.npz \
  source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/npz/go2_d1/pick_stow_carry_robot_only.npz
```

The robot starts at the environment origin, walks `1.5297 m` during the 3-second approach, and reaches `4.0526 m`
during carry. The phase markers are approach `150`, settle `250`, pre-grasp `525`, grasp `600–640`, stow `940`,
departure `1040`, carry `1590`, final settle `1690`, place `2000`, release `2040`, and retract `2340`.

Training samples an isotropic `0.70–0.90×` box scale per environment, producing boxes from
`94.5 × 38.5 × 178.5 mm` to `121.5 × 49.5 × 229.5 mm`. Mass is sampled independently from `15–40 g`. The matching
center correction is applied to reset state and every current/future reference position, keeping each sampled box
grounded. A scale-dependent center shift also fixes the grasp-facing `-X` surface in place, so shrinking box depth
does not move the contact surface away from the demonstrated fingers.

The source gripper channel is already in the D1's physical joint convention: both jaw coordinates are positive,
`0 m` is fully open, and each jaw moves to `0.00565 m` at box contact. Replay preserves this continuous
recording for inspection. The manipulation training task converts any positive source position into a binary close
phase, so its reference target is only `0` (open) or `0.033 m` (fully close). Its 19th policy output is likewise a
binary logit: values above zero command close and all other values command open. The USD mimic constraint applies the
same physical target to the second jaw. The gripper reward checks only whether this open/close state matches the
demonstration; it does not reward jaw position or logit magnitude. The asset default remains fully open.

Manipulation resets form a phase curriculum: `50%` begin at frame zero, `20%` begin at a uniformly sampled trajectory
frame, and `30%` begin in a sampled attached-object frame. Random-phase resets initialize the complete robot reference
state and the matching physical box pose, linear velocity, and angular velocity. The object transform uses the robot
anchor pose just written to simulation, so reset initialization never depends on stale articulation buffers. Root
linear and angular velocities are rotated with pose perturbations. For attached resets, the jaws use the recorded
`5.65 mm` contact coordinate rather than the fully closed command endpoint, placing the box in the hand without initial
penetration. The binary close drive then supplies normal grasp force. Episodes end at the selected clip boundary.

An exact-mesh sweep of all 2,341 reference frames found that the pickup/place reach crosses the Go2 `Head_upper`
collision cylinder with D1 Link4, Link5, and Link6, and that Link6 crosses the placed box during retraction. The pick
task filters only those four reference-conflicting pairs. Other robot self-collisions, ground contacts, and both
finger-to-object contacts remain enabled. The normal policy, privileged teacher, and distillation student inherit the
same scene and reset behavior.

Isaac replay uses the same `Empty_Link_L.STL` and `Empty_Link_R.STL` finger surfaces as the source Viser URDF for
both rendering and collision. The imported D1 USD link frames use Z as their prismatic axis, so the asset applies the
required `+/-90 deg` mesh-frame conversion from the hardware URDF's Y-axis jaw convention. This keeps the visible pads,
collision surfaces, and recorded joint coordinates aligned.

</details>

## Convert CSVs

Convert the locally installed CSV folders:

```bash
python source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/apex_batch_csv_to_motion_npz.py
```

Convert Go2+D1 CSVs:

```bash
python source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/apex_batch_csv_to_motion_npz.py \
  --input-root source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/imitation_data/go2_d1_motions_csv \
  --output-root source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/npz/go2_d1 \
  --all-csv \
  --strict
```

Convert one CSV:

```bash
python source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/apex_csv_to_motion_npz.py \
  --input /path/to/go2_retarget_canter_2ms.csv \
  --output go2_apex_reference.npz \
  --csv-leg-order FL,FR,RL,RR
```

## Replay And Audit

Replay a CSV:

```bash
python scripts/tools/go2_apex/replay_csv.py \
  --input source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/imitation_data/animal_mocap/hopturn_go2_STMR_resampled_50Hz.csv \
  --csv-leg-order FL,FR,RL,RR \
  --fps 50 \
  --zero-xy-origin
```

Replay an NPZ:

```bash
python scripts/tools/go2_apex/replay_npz.py \
  --input source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/npz/animal_mocap/hopturn_go2_STMR_resampled_50Hz.npz \
  --zero-xy-origin
```

Replay a folder or glob and switch clips with `N` and `P`:

```bash
python scripts/tools/go2_apex/replay_npz.py \
  --input source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/npz/learn_diverse_quad_loco/combined \
  --zero-xy-origin
```

Replay Go2+D1 tracker clips:

```bash
python scripts/tools/go2_apex/replay_npz.py \
  --robot go2_d1 \
  --input source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/npz/go2_d1 \
  --zero-xy-origin
```

Replay the hello-wave clip by itself:

```bash
python scripts/tools/go2_apex/replay_npz.py \
  --robot go2_d1 \
  --input source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/npz/go2_d1/wave_hello.npz \
  --zero-xy-origin
```

To keep the dog and root exactly on the recorded motion while testing the six arm joints and physical gripper through
the current actuator Kp/Kd, add `--drive-arm-with-pd`:

```bash
python scripts/tools/go2_apex/replay_npz.py \
  --robot go2_d1 \
  --input source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/npz/go2_d1/wave_hello.npz \
  --drive-arm-with-pd \
  --zero-xy-origin
```

This diagnostic mode requires a 50 Hz, 19-channel Go2+D1 motion. It holds each arm target for four `5 ms` physics
steps, matching training, and prints per-joint RMSE and maximum tracking error after each pass. The ordinary replay
without this flag remains fully kinematic.

Replay the walk-then-wave clip:

```bash
python scripts/tools/go2_apex/replay_npz.py \
  --robot go2_d1 \
  --input source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/npz/go2_d1/walk_then_wave_hello.npz \
  --zero-xy-origin
```

Replay the pick-stow-carry-place clip with its box trajectory:

```bash
python scripts/tools/go2_apex/replay_npz.py \
  --robot go2_d1 \
  --input source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/npz/go2_d1/pick_stow_carry.npz \
  --binary-gripper \
  --gripper-closed-position 0.033 \
  --zero-xy-origin
```

Replay the exact nominal box and overlay the Isaac/PhysX collision proxies:

```bash
python scripts/tools/go2_apex/replay_npz.py \
  --robot go2_d1 \
  --input source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/npz/go2_d1/pick_stow_carry.npz \
  --binary-gripper \
  --gripper-closed-position 0.033 \
  --object-size-scale-range 1.0 1.0 \
  --zero-xy-origin \
  --camera-focus gripper \
  --kit_args="--/persistent/physics/visualizationDisplayColliders=2"
```

The collision overlay shows the physical proxies rather than only sphere markers. If the Kit setting is not retained,
select **Viewport eye → Show By Type → Physics → Colliders**. Add `--start-frame 500 --end-frame 700` for the pickup
and head-proxy intersection, or `--start-frame 1950 --end-frame 2150` for placement and retraction.

To inspect only the open-to-close transition, add `--start-frame 575 --end-frame 675`. The kinematic replay applies
the same binary open/fully-close targets as training when `--binary-gripper` is present; omit that flag and
`--gripper-closed-position` to inspect the original continuous jaw recording. The training task adds physical
finger/object contact. For clips with gripper
channels, the default `--camera-focus base` uses a wide full-robot view for both kinematic and PD-driven arm replay.
Use `--camera-focus gripper` for a close view of the fingers, or `--camera-focus auto` to select that close view
automatically when gripper channels are present. Replay also prints the measured finger-center separation and visual
inner-pad opening at the first fully-open frame. To hold a known fully-open pose continuously for visual comparison
with Viser, use `--start-frame 575 --end-frame 575`; frame 575 has `q=[0, 0] m` and a `66.3 mm` inner-pad opening.
The close begins after frame 600 and reaches the recorded box-contact coordinate at frame 640.

To compare the training variants side by side, replay two or more environments. Environments are ordered from the
smallest/lightest variant to the largest/heaviest variant; size is visible and the paired mass is printed in the
console:

```bash
python scripts/tools/go2_apex/replay_npz.py \
  --robot go2_d1 \
  --input source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/npz/go2_d1/pick_stow_carry.npz \
  --binary-gripper \
  --gripper-closed-position 0.033 \
  --num-envs 5 \
  --object-size-scale-range 0.70 0.90 \
  --object-mass-range 0.015 0.040 \
  --zero-xy-origin
```

With one environment, replay samples one variant; add `--object-size-seed 0` for a repeatable sample. The legacy
`--object-height-scale-range` and `--object-height-seed` spellings remain accepted as aliases, but scaling now applies
uniformly to X, Y, and Z.

For Go2+D1, the replay uses the same APEX USD as training. The orange sphere is
the stored `arm_ee_pos_w` target and the smaller blue sphere is the midpoint of
the simulator's `Link7_1` and `Link7_2` bodies at the replayed joints. They
should overlap. The tool also prints their mean and maximum distance when a pass
finishes; add `--once --headless --frame-stride 10` for a quick numerical check.

If a motion was generated from another kinematic model, create a copy with only
its end-effector channel regenerated from the selected Isaac articulation:

```bash
python scripts/tools/go2_apex/replay_npz.py \
  --robot go2_d1 \
  --input path/to/motion.npz \
  --export-corrected-arm-ee-dir /tmp/go2_d1_corrected \
  --once \
  --headless
```

The exporter preserves all other motion channels and stores the original path
as `arm_ee_pos_w_embodik`. The bundled hello clips have already been corrected
this way.

Replay B2+Z1 tracker clips:

```bash
python scripts/tools/go2_apex/visualize_b2_z1_motions.py \
  --input source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/npz/b2_z1_motions
```

Audit converted clips before training:

```bash
python scripts/tools/go2_apex/analyze_motion_npz.py \
  --input source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/npz/cross_morpho_replay_added \
  --output-dir logs/motion_audit/cross_morpho_replay_added
```

The audit writes per-clip summaries, overview plots, detailed plots, and event CSVs for velocity mismatch, joint
acceleration, frame jumps, foot height, and quaternion normalization.

## Tracker Motion Notes

- Go2 tracker reference observations default to frames `(0, 1, 2, 5, 10)`; one-step variants use `(0, 1)`.
- Go2+D1 replay and APEX motion debug draw the reference arm end-effector target when `arm_ee_pos_w` is present.
- B2+Z1 tracker, privileged tracker, and one-step/history student configs currently run at `sim.dt=0.005` with
  decimation `4`. Reference phase advances by clip `fps`, so 30 Hz clips hold frames across some 50 Hz policy updates
  instead of playing fast.
- The configured B2+Z1 tracking ablation excludes `walk4_subject1.npz`, removes the arm end-effector velocity reward,
  doubles arm joint-position and end-effector position reward weights, and leaves B2+Z1 RSL-RL action clipping disabled.

## Train Or Play With One Clip

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Gurukul-Isaac-Go2-APEX-Flat-v0 \
  --agent rsl_rl_cfg_entry_point \
  --motion-file source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/npz/animal_mocap/hopturn_go2_STMR_resampled_50Hz.npz \
  --headless
```

```bash
python scripts/reinforcement_learning/rsl_rl/play.py \
  --task Gurukul-Isaac-Go2-APEX-Flat-v0 \
  --agent rsl_rl_cfg_entry_point \
  --motion-file source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/npz/animal_mocap/hopturn_go2_STMR_resampled_50Hz.npz \
  --num_envs 1
```
