---
title: Training Recipes
---

# Training Recipes

These recipes are the maintained source for teacher/student command templates.

## Go2 rough oracle teacher

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Gurukul-Isaac-Velocity-Rough-Unitree-Go2-v0 \
  --agent rsl_rl_teacher_cfg_entry_point \
  --num_envs 4096 \
  --headless \
  --logger wandb \
  --wandb_project_name Go2_Student_Teacher_Baselines \
  --run_name oracle_teacher
```

## Go2 rough student-teacher distillation

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Gurukul-Isaac-Velocity-Rough-Unitree-Go2-v0 \
  --agent rsl_rl_distillation_cfg_entry_point \
  --num_envs 512 \
  --headless \
  --logger wandb \
  --wandb_project_name Go2_Student_Teacher_Baselines \
  --experiment_name unitree_go2_rough_student_teacher \
  --teacher_load_run <teacher_run_folder_name> \
  --run_name student_teacher_distillation
```

## Go2 rough depth action-prior PPO

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Gurukul-Isaac-Velocity-Rough-Unitree-Go2-Depth-Distill-v0 \
  --agent rsl_rl_depth_action_prior_cfg_entry_point \
  --num_envs 512 \
  --headless \
  --enable_cameras \
  --logger wandb \
  --wandb_project_name Go2_Student_Teacher_Baselines \
  --teacher_load_run <teacher_run_folder_name> \
  --run_name depth_action_prior
```

## Go2 rough concurrent teacher-student PPO

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task Gurukul-Isaac-Velocity-Rough-Unitree-Go2-CTS-v0 \
  --agent rsl_rl_cts_cfg_entry_point \
  --num_envs 4096 \
  --headless \
  --logger wandb \
  --wandb_project_name Go2_Student_Teacher_Baselines \
  --run_name go2_rough_cts
```

See [Concurrent Teacher-Student](concurrent-teacher-student) for CTS-specific observation paths, logs, and reward notes.

## B2+Z1 APEX one-step history distillation

Distill a privileged B2+Z1 APEX teacher into the deployable one-step-future history student:

```bash
python scripts/reinforcement_learning/rsl_rl/train.py \
  --task=Gurukul-Isaac-B2-Z1-Arm-APEX-Flat-Tracker-One-Step-Future-History-v0 \
  --headless \
  --num_envs 16384 \
  --logger wandb \
  --agent=rsl_rl_distillation_cfg_entry_point \
  --teacher_load_run <b2_z1_privileged_tracker_teacher_run_folder_name> \
  --run_name b2_z1_arm_apex_flat_tracker_one_step_future_history_distill
```

Use `--teacher_checkpoint logs/rsl_rl/unitree_b2_z1_arm_apex_flat_privileged_tracker/<run>/model_<iter>.pt` to pin a
specific privileged checkpoint. `--teacher_checkpoint` can also point at a run directory; training resolves it to the
latest `model_<iter>.pt` in that directory.

## Teacher checkpoint overrides

Distillation and action-prior training resolve teacher checkpoints separately from student resume checkpoints. Use:

- `--teacher_experiment_name`
- `--teacher_load_run`
- `--teacher_load_checkpoint`
- `--teacher_checkpoint`

When resuming a DecAP/action-prior run with `--resume`, runner-side DecAP and env-side action priors restore the decay
schedule from the student checkpoint iteration, so the first resumed rollout uses the factor that corresponds to the
checkpoint progress.

## Evaluation

Use `scripts/reinforcement_learning/rsl_rl/eval_student.py` for quantitative teacher and student comparisons. It reports closed-loop episode metrics, reward-term metrics, velocity tracking RMSE, depth validity/saturation metrics, and teacher-action mismatch metrics when the policy exposes a teacher path.

Evaluate the oracle teacher:

```bash
python scripts/reinforcement_learning/rsl_rl/eval_student.py \
  --task Gurukul-Isaac-Velocity-Rough-Unitree-Go2-v0 \
  --agent rsl_rl_teacher_cfg_entry_point \
  --episodes 1024 \
  --num_envs 256 \
  --run_label oracle_teacher \
  --csv logs/rsl_rl/go2_student_eval_summary.csv
```

Evaluate a student-teacher distillation baseline:

```bash
python scripts/reinforcement_learning/rsl_rl/eval_student.py \
  --task Gurukul-Isaac-Velocity-Rough-Unitree-Go2-v0 \
  --agent rsl_rl_distillation_cfg_entry_point \
  --experiment_name unitree_go2_rough_student_teacher \
  --episodes 1024 \
  --num_envs 256 \
  --run_label student_teacher_distillation \
  --reference_return <oracle_return_mean> \
  --csv logs/rsl_rl/go2_student_eval_summary.csv
```

Use `--checkpoint logs/rsl_rl/<experiment>/<run_folder>/model_<iter>.pt` for exact checkpoint comparisons. Use
`--keep_randomization` only when robustness under training-time randomization or corruption is intentional.

## Sim2sim handoff

After the student actor is stable in Isaac Lab playback, export the actor and check [Sim2Sim](sim2sim) for the matching MuJoCo runner task. Teacher checkpoint flags are for training and evaluation; the sim2sim runner consumes the exported student actor.
