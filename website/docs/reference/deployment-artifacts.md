---
title: Deployment Artifacts
description: Policy exports, parameter contracts, and supported simulation and hardware consumers.
---

# Deployment Artifacts

Supported Unitree task paths can export ONNX policy and YAML metadata through playback.

```text
logs/rsl_rl/<experiment>/<run_name>/
  exported/policy.onnx
  params/deploy.yaml
  params/env.yaml
  params/agent.yaml
```

## Consumers

- `unitree-sim2real/` consumes exported Go2 ONNX policies for MuJoCo sim2sim and the provided hardware runner.
- `deploy/` documents G1 controller build and policy consumption details.
- `engineai-sim2real/` contains the exact official PM01 MuJoCo model, the 1,173-observation velocity runner, the
  official 129-observation/24-action PM01 dance runner, and the T800 BeyondMimic contract runner. Imported tracking
  policies, motions, and deployment metadata are stored with their tasks.

An ONNX file is only one deployment artifact. The observation/action interface, timing, gains, and safety behavior also need to match the target runtime.

## Hardware Runners

Current hardware entry points:

- `unitree-sim2real/RL_policy_runner/sim2real/go2_hardware_pos_obs45.py`
- `unitree-sim2real/RL_policy_runner/sim2real/b2_hardware_pos_obs45.py`
- `unitree-sim2real/RL_policy_runner/sim2real/b2_z1_arm_hardware_pos_obs61.py`
- `unitree-sim2real/RL_policy_runner/sim2real/go2_apex_tracker_hardware.py`
- `unitree-sim2real/RL_policy_runner/sim2real/b2_z1_apex_history_hardware.py`
- `deploy/robots/g1_29dof/`

Before hardware, check observation order, action order, control rate, PD gains, torque limits, exported deploy config,
emergency stop behavior, and low-speed validation.

## EngineAI PM01

The PM01 MuJoCo scene is stored under:

```text
engineai-sim2real/engineai_mujoco/engineai_robots/pm01/
  scene.xml
  official/
```

The official-URDF velocity actor has 1,173 observations and 24 actions. Five actor terms have 15-frame histories;
the velocity command is current-only, and all joints including head yaw are policy-controlled. The separate official
dance actor has 129 observations and 24 actions. Its artifacts are under
`source/Gurukul/Gurukul/tasks/manager_based/beyondmimic/config/engineai_pm01_24dof/`.
That directory also contains the 24-DoF walking reference and
`hardware/pm01_24dof_contract.yaml`, which records both joint-order mappings, `kp`, `kd`, defaults, action scales,
effort limits, control timing, and native motor signs.

## EngineAI T800

The T800 trained artifacts are under:

```text
source/Gurukul/Gurukul/tasks/manager_based/beyondmimic/config/engineai_t800/
  motion/dance_t800.npz
  pretrained/dance.pt
  pretrained/policy.onnx
  pretrained/policy.mnn
  pretrained/deploy_config.yaml
```

Use the [T800 BeyondMimic page](../tasks/beyondmimic/t800) to run Isaac Lab playback or the official EngineAI MuJoCo
model. The native SDK configuration intentionally zeros the final two elbow-yaw action scales, while the RL export
metadata sets them to `0.05`; the sim2sim runner defaults to the native SDK safety values.

For Go2 velocity policies:

```bash
cd unitree-sim2real
python RL_policy_runner/sim2real/go2_hardware_pos_obs45.py eth0
```

Use `GURUKUL_POLICY_PATH` to point a runner at a specific exported policy.

## G1 Controller Build

Install the native dependencies used by the reference deploy stack, then build the G1 controller:

```bash
sudo apt install -y libyaml-cpp-dev libboost-all-dev libeigen3-dev libspdlog-dev libfmt-dev
git clone git@github.com:unitreerobotics/unitree_sdk2.git
cd unitree_sdk2
mkdir build && cd build
cmake .. -DBUILD_EXAMPLES=OFF
sudo make install
```

```bash
cd deploy/robots/g1_29dof
mkdir -p build
cd build
cmake ..
make
```

## G1 Sim2Sim Flow

1. Start the G1 MuJoCo simulation.
2. Run `deploy/robots/g1_29dof/build/g1_ctrl`.
3. Press `L2 + Up` to enter `FixStand`.
4. Press `R1 + X` to switch into the exported RL policy.

The G1 port is velocity-only. The mimic-state path from the reference repository is intentionally omitted.
