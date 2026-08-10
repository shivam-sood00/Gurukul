---
title: Concurrent Teacher-Student
description: Experimental concurrent privileged-teacher and deployable-student PPO adaptation for Go2.
---

# Concurrent Teacher-Student

Concurrent teacher-student (CTS) trains privileged teacher and deployable student action sources in one PPO run. The current starter task is a rough Go2 velocity-locomotion task, not an APEX motion-imitation task.

| Surface | Training evidence | Artifact | Transfer |
| --- | --- | --- | --- |
| Registered; teacher/student agent configured | Algorithm-level CPU smoke only; no Isaac convergence result | No bundled checkpoint | No task-specific sim2sim or hardware validation |

## Task

```text
Gurukul-Isaac-Velocity-Rough-Unitree-Go2-CTS-v0
```

## Train

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-Velocity-Rough-Unitree-Go2-CTS-v0 \
  --agent=rsl_rl_cts_cfg_entry_point \
  --headless \
  --run_name go2_rough_cts
```

## Interfaces

- Teacher path: a repository-defined privileged-state subset, not every full-state signal used by the paper.
- Student path: deployable `policy` observations plus a flattened five-observation history.
- Shared extension code: `source/Gurukul/Gurukul/tasks/manager_based/concurrent_teacher_student.py`.
- Velocity task registration and configs: `source/Gurukul/Gurukul/tasks/manager_based/locomotion/velocity/`.

The paper-aligned defaults use a 32-dimensional L2-normalized latent, `[512, 256]` privileged and history encoders,
and `[512, 256, 128]` shared actor and critic MLPs. The critic consumes the privileged state and the latent selected for
the rollout role. Teacher and student PPO-Clip objectives are averaged separately and summed; latent reconstruction is
computed only from student trajectories as the squared L2 loss in Equation 8.

## Logs

CTS logs separate `teacher_surrogate`, `student_surrogate`, and `history_encoder` losses. The rough Go2 CTS task also logs `Curriculum/cts_teacher_terrain_level` and `Curriculum/cts_student_terrain_level` so teacher and student terrain progression can be checked independently.

Distributed CTS runs synchronize both PPO parameters and the separately optimized student history encoder. RND and
symmetry augmentation are rejected at startup because this custom PPO path does not implement them.

CTS uses five PPO epochs followed by five student-encoder epochs. PPO uses the desired-KL adaptive learning-rate
schedule (`desired_kl=0.01`); the encoder uses Adam at `1e-3`. CTS checkpoints include both optimizer states so resumed
training preserves the history encoder's momentum.

CTS checkpoints created before the paper-alignment update used a different latent size and network shape and are not
strict-load compatible with the current 32-dimensional configuration.

## Reward and curriculum notes

The rough Go2 CTS reward set follows
[`wty-yy/go2_rl_gym`](https://github.com/wty-yy/go2_rl_gym): velocity tracking, vertical/angular velocity penalties,
torque/acceleration/power penalties, action-rate and action-smoothness penalties, calf/thigh collision penalty,
joint-position-limit and hip-default penalties, base-height correction, and feet-regulation.

The CTS task schedules `lin_vel_z_l2` from `-2.0` to `0.0` by iteration 1500 and `correct_base_height` from `-1.0` to `-10.0` by iteration 5000.

## Related docs

- [Velocity locomotion baselines](../velocity-locomotion/baselines)
- [Training recipes](training-recipes)

## References And Credits

- CTS follows [Concurrent Teacher-Student RL](https://arxiv.org/abs/2405.10830) for the concurrent update, network
  widths, latent contract, rollout split, and selected network/update defaults. This remains a Go2/Isaac Lab adaptation:
  the local environment uses 4,096 environments rather than the paper's 8,192, and the repository privileged-state
  teacher subset omits paper inputs such as joint torque, joint acceleration, and full contact-force values. The paper
  used Isaac Gym with several quadruped and humanoid platforms, and no official source repository was published. An algorithm smoke run
  establishes software integration, not reproduction of the paper's 5,000-iteration multi-robot results.
