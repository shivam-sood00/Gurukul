# EngineAI Sim2Real

This folder is reserved for EngineAI PM01 and T800 hardware deployment.

EngineAI's public stack is `engineai_robotics_native_sdk` and its LCM tools. Do not route EngineAI hardware through
the Unitree SDK2 scripts. Validate the T800 policy first with
`../sim2sim/run_t800_beyondmimic_policy.py`; the hardware state machine, motor enable, emergency stop, PD-stand
transition, and remote installation remain the responsibility of the official native SDK.
