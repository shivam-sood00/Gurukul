# PM01 24-DoF hardware contract

`pm01_24dof_contract.yaml` freezes the joint-name mapping, policy-to-hardware indices, control rates, gains, default
positions, action scales, effort limits, and native motor signs needed to consume a newly trained 24-action walking
export.

Keep inference tensors in `policy_joint_names` order. Map targets into `hardware_joint_names` with
`policy_to_hardware_index` before sending commands. The policy target is a residual around `default_joint_pos`; it is
not a residual around the current walking reference because EngineAI's native configuration sets
`resident_control: false`.

This file is integration metadata, not authorization to run on hardware. Preserve the native SDK's PD stand,
transition state machine, joint-limit checks, torque limits, communications watchdog, and emergency stop.
