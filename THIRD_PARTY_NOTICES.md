# Third-Party Notices and Provenance

Gurukul combines repository-authored work with adapted and vendored material. This catalog is an index, not a
replacement for file-level notices or the license files shipped beside individual components.

## Foundation

- **Isaac Lab** — https://github.com/isaac-sim/IsaacLab — BSD-3-Clause. Gurukul is an Isaac Lab extension and retains
  Isaac Lab headers in adapted templates and task infrastructure.
- **robot_lab** — https://github.com/fan-ziqi/robot_lab — Apache-2.0. Gurukul's task structure and Isaac Lab extension
  layout were inspired by `robot_lab`, and some baseline task, robot-asset, runner, and template files were adapted
  from upstream commit `09f7fb4aa93fdeac3eb91d8c155f67cab7a10e46`.
- **RSL-RL** — https://github.com/leggedrobotics/rsl_rl — runtime/training dependency. Local runner extensions build on
  its public APIs; the installed package remains governed by its own license.
- **Legged Gym** — https://github.com/leggedrobotics/legged_gym — BSD-3-Clause. The velocity-locomotion task lineage
  follows the environment structure introduced by Legged Gym, through Isaac Lab and `robot_lab` adaptations.
- **Unitree ROS** — https://github.com/unitreerobotics/unitree_ros — BSD-3-Clause reference source for Unitree robot
  descriptions and joint metadata used by the inherited asset/configuration surface.
- **MuJoCo and Unitree communication stacks** — https://github.com/google-deepmind/mujoco,
  https://github.com/unitreerobotics/unitree_sdk2, https://github.com/unitreerobotics/unitree_sdk2_python, and
  https://github.com/unitreerobotics/unitree_ros2 are external dependencies referenced by the sim2sim and guarded
  hardware paths. They remain governed by their respective upstream terms.

## Vendored Code and Assets

- **Unitree MuJoCo** — `unitree-sim2real/unitree_mujoco/` — BSD-3-Clause. The retained license is
  `unitree-sim2real/unitree_mujoco/LICENSE`.
- **ONNX Runtime 1.22.0** — `deploy/thirdparty/onnxruntime-linux-x64-1.22.0/` — MIT. Its license and upstream third-party
  notices are retained in that directory.
- **EngineAI RL Lab and native SDK** — PM01/T800 assets, checkpoints, exports, and MuJoCo material are pinned and
  documented beside the files in `UPSTREAM.md` and `LICENSE*.txt` files under the relevant EngineAI directories.
- **BrainCo RevoLab** — the Revo3 task and USD integration are adapted from
  https://github.com/BrainCoTech/RevoLab under the MIT License. The notice and full license text are retained in
  `source/Gurukul/Gurukul/tasks/direct/brainco_revo/`.
- **OmniPerception** — `third_party/OmniPerception/` is pinned from
  https://github.com/aCodeDog/OmniPerception at commit `a1059ae3ffb91ebea2854f8633a28027a0477d1c`. 

## Research Methods

The website's [Acknowledgements](website/docs/reference/credits.md) page highlights the major projects and research
that shaped Gurukul. Individual task pages record task-specific references and scope differences.

- **MimicKit** — https://github.com/xbpeng/MimicKit/tree/2ed1e6c093bb0829f55d33cb4f7a1731cfe6cb69 — Apache-2.0. The score-matching motion-prior implementation is
  informed by MimicKit's frozen diffusion reward and generative state-initialization design. Gurukul does not bundle
  MimicKit motion datasets or pretrained prior checkpoints.
