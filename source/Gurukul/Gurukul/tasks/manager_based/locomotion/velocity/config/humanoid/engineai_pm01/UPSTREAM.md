# EngineAI PM01 AMP provenance

The PM01 walking reference at `motion/locomotion.npz` and the AMP task design were adapted from
[`engineai-robotics/engineai_amp`](https://github.com/engineai-robotics/engineai_amp) commit
`83ba64bbb58a02e14483e52adce5f893f3f31cdf` (BSD-3-Clause).

The reference is byte-identical to upstream `dataset/data/locomotion.npz` and has SHA-256
`945fa07b32c66410d2904e9f9147f4d433a69f9d688965c366092e71a311f6b4`. It contains 2,303 frames at 100 Hz for
EngineAI's legacy 23-movable-joint PM01. The maintained Gurukul velocity actor still controls all 24 joints, including
`J23_HEAD_YAW`; the AMP discriminator excludes head yaw and consumes the upstream five-frame state made from scaled
joint positions and base-frame linear velocity.

Upstream trains its actor at 100 Hz. Gurukul retains the maintained PM01 deployment contract of a 500 Hz simulation
and 50 Hz actor, so the reference loader takes every second upstream frame to align discriminator and policy time.
