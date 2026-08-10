# EngineAI PM01 24-DoF tracking provenance

The PM01 USD, dance motion, trained checkpoint, deployment config, ONNX export, and MNN export were imported from
`engineai-robotics/engineai_rl_lab` commit `14ec57be718586bd0ac45375aa1115bd896fbdbc` (BSD-3-Clause).

The exported MNN and motion are byte-identical to the PM01 `rl_dance_example` files in
`engineai-robotics/engineai_robotics_native_sdk` commit `83204a459e0e786f855235a8507197496a79acc7`.

This tracking release remains separate from Gurukul's official-URDF 24-action velocity tasks. The official USD
keeps `J23_HEAD_YAW` actuated and the imported dance policy has a 129-observation, 24-action interface.
