# EngineAI Deployment

EngineAI deployment files live here instead of under `unitree-sim2real/`.

## PM01 Velocity Sim2Sim

The PM01 velocity task uses the retained official 24-DoF URDF for Isaac training and EngineAI's official native-SDK
MuJoCo model for sim2sim. The actor has 1,173 observations and 24 actions, including head yaw, and runs at 50 Hz over
a 500 Hz low-level simulation.

Train a flat policy first:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Velocity-Flat-EngineAI-PM01-v0 \
  --num_envs 4096 \
  --headless \
  --run_name engineai_pm01_official_urdf_flat_ppo
```

Then run its exported ONNX in MuJoCo:

```bash
python engineai-sim2real/RL_policy_runner/sim2sim/run_pm01_policy.py \
  --policy-path logs/rsl_rl/engineai_pm01_official_urdf_flat/<run>/exported/policy.onnx \
  --launch-mujoco \
  --mujoco-viewer native
```

The checked-in PM01 scene is the official EngineAI native-SDK model. Its provenance is recorded in
`engineai_mujoco/engineai_robots/pm01/official/UPSTREAM.md`.

## PM01 BeyondMimic Sim2Sim

Run EngineAI's official 129-observation/24-action PM01 dance tracker against the same official model:

```bash
MUJOCO_GL=disable python engineai-sim2real/RL_policy_runner/sim2sim/run_pm01_beyondmimic_policy.py
```

Add `--viewer --real-time` for interactive playback.

## T800 BeyondMimic Sim2Sim

```bash
python engineai-sim2real/RL_policy_runner/sim2sim/run_t800_beyondmimic_policy.py \
  --native-sdk-root ../engineai_robotics_native_sdk
```

## Sim2Real

EngineAI real-robot deployment should use the EngineAI humanoid/native SDK stack, not the Unitree SDK2 bridge. The
local DDS path is a sim2sim adapter only.
