# EngineAI RL Lab audit

Audit date: 2026-07-27

## Snapshots

- `engineai-robotics/engineai_rl_lab`: `14ec57be718586bd0ac45375aa1115bd896fbdbc`
- `engineai-robotics/engineai_robotics_native_sdk`: `83204a459e0e786f855235a8507197496a79acc7`
- Gurukul comparison base: `d96ca005a05154e9279cf49a1b53e54ceb4cc7eb`

Both EngineAI repositories were cloned under a temporary `/tmp/engineai_rl_lab_audit.*` directory. The native SDK was
also inspected because `engineai_rl_lab` delegates MuJoCo sim2sim and hardware deployment to that repository rather
than containing the simulator/controller itself.

## Scope comparison

| Area | Official EngineAI repositories | Gurukul before this import |
| --- | --- | --- |
| Robots | PM01 Edu and T800 whole-body tracking | Many robot/task families; PM01 tracking and velocity, but no T800 |
| Training | A focused manager-based tracking environment and RSL-RL PPO | Shared BeyondMimic tracking plus APEX extensions, locomotion, manipulation, teacher/student, and other task families |
| Motion input | CSV conversion and NPZ replay for PM01/T800 | G1/PM01 BeyondMimic converters and replay |
| Policy export | Checkpoint plus ONNX, MNN, and deployment YAML | RSL-RL checkpoint/ONNX flows; no EngineAI T800 MNN package |
| Sim2sim | Separate native SDK: MuJoCo, MNN C++ runner, LCM, state machine, virtual gamepad | PM01 velocity Python path; no T800 tracking sim2sim |
| Sim2real | Same native SDK and EngineAI robot install/run scripts | PM01 deployment area intentionally separated, with no T800 hardware route |

## Tracking implementation differences

- The official project is a narrow EngineAI-specific fork of the same reference-tracking design used by this
  repository's BeyondMimic task.
- EngineAI's policy joint-velocity observation uses per-joint noise: ankle joints use `[-3, 3]`, while other joints
  use `[-0.5, 0.5]`. The previous shared Gurukul task used a uniform `[-0.5, 0.5]`.
- EngineAI removed the shared `joint_acc_l2` and `joint_torques_l2` reward terms. Gurukul retains those terms
  for its existing task families.
- EngineAI's base tracking episode is 10 seconds; the previous Gurukul base is 20 seconds.
- EngineAI uses `10 * 2^15` GPU contact-pair capacity; Gurukul uses `16 * 2^15`.
- Gurukul adds APEX-style current/future reference observations, body-name-aware NPZ mapping, compact
  reference visualization, and additional multi-robot integrations that are absent upstream.
- The imported motion loader now also honors NPZ `joint_names` metadata and reorders joint columns to the articulation
  order. EngineAI stores the names but its loader does not perform this reorder.

## T800 contract

- 25 policy joints, including head pitch/yaw.
- Actor observation: 134 values.
  - reference joint position and velocity: 50
  - reference-to-current anchor orientation, first two rotation-matrix columns: 6
  - local IMU angular velocity: 3
  - joint position relative to default: 25
  - joint velocity: 25
  - previous action: 25
- Actor output: 25 joint residuals.
- Policy rate: 50 Hz. Official MuJoCo step: 500 Hz.
- The native SDK has `resident_control: false`; targets are default joint positions plus scaled actions, not reference
  joint positions plus residuals.
- Dance motion: 1,054 frames, 50 Hz, 25 joints, 30 bodies.
- Imported checkpoint iteration: 44,000. The checkpoint's saved run configuration allows 50,000 iterations, while the
  current upstream source config declares 20,000; this is an upstream snapshot/run inconsistency.
- The exported deployment YAML assigns `0.05` action scale to both elbow-yaw joints. The native SDK T800 dance YAML
  assigns `0.0`. The new sim2sim runner defaults to the native SDK values and requires an explicit option for the raw
  export values.

## Imported and adapted components

- T800 USD and configuration layers under `source/Gurukul/data/Robots/engineai/t800/`.
- T800 asset/actuator configuration, including the upstream 1–3 step actuator delay.
- Three registered T800 BeyondMimic task variants.
- Official dance CSV/NPZ, RSL-RL checkpoint, ONNX export, MNN export, and deployment YAML.
- Per-joint observation-noise configuration.
- T800 support in the BeyondMimic NPZ replay tool.
- A standalone ONNX/MuJoCo contract runner that consumes the official native SDK model without copying its 113 MB
  MuJoCo mesh tree into this repository.
- Public task, deployment-artifact, task-registry, and navigation documentation.
- The upstream BSD-3-Clause notice is retained with the imported task artifacts.

## Verification

- Artifact/contract tests: 4 passed.
- Python compilation and targeted Ruff checks: passed.
- Docusaurus production build: passed.
- Official T800 MuJoCo model loaded as `nq=32`, `nv=31`, `nu=25`, `dt=0.002`.
- Full 1,054-step policy run with native SDK scales:
  - finite observations, actions, positions, and velocities
  - minimum base height: 0.8880 m
  - final base height: 0.9964 m
- Full run with export scales also remained finite and upright.
- Native SDK C++ configure was attempted on the host and stopped because CMake could not find LCM. EngineAI's
  documented container/native dependency environment is still required for the full state-machine build.
- Isaac task registration reached Isaac Sim startup, but this execution environment has no available CUDA/Vulkan
  device. Full Isaac playback was therefore not run here.

## Before sim2real

1. Run the standalone T800 contract test and inspect the motion with the viewer.
2. Build and run EngineAI's complete native SDK in its documented container.
3. Test the PD-stand, reset, dance, dance-to-walk, and passive/emergency-stop transitions in MuJoCo.
4. Resolve and record the elbow-yaw scale choice.
5. Confirm controller/robot firmware compatibility, joint order, gains, torque limits, and remote-stop behavior.
6. Start hardware validation suspended, at reduced authority, with a clear exclusion zone and an operator on the
   emergency stop.
