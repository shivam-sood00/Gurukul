# Gurukul Agent Guide

Gurukul is an Isaac Lab research workspace for robot-learning experiments, baselines, tutorials, and
deployment work. Optimize for correct, reproducible, understandable changes rather than turning the repository into a
generic software product.

## Default Workflow

1. Read this file and any nearer `AGENTS.md` that applies to the files being changed.
2. Inspect the relevant source, tests, task registrations, and public documentation before implementing.
3. Search for an existing implementation before adding a task, config, runner, asset, or documentation page.
4. Make the smallest coherent change that satisfies the request. Preserve public behavior unless a change is
   intentional and documented.
5. Run the smallest relevant validation first, then broader or simulator-dependent checks when justified.
6. Update the closest public docs page when user-visible behavior changes.
7. Report what was validated and what could not be run.

Preserve unrelated user changes in the worktree. Do not rewrite adjacent experimental code or perform broad cleanup
unless it is necessary for the requested change.

## Preserve Author-Written Prose

- Treat prose in `README.md` and `website/` as author-written text. Do not rewrite, rephrase, summarize, polish, or
  change its tone unless the user explicitly asks for that specific text to be edited.
- For visual, layout, navigation, or structural requests, preserve the existing wording exactly. Moving or wrapping
  text is acceptable only when its words and meaning remain unchanged.
- If wording appears inaccurate or unclear but the user did not request copy editing, point it out and propose a change
  instead of editing it directly.

## Repository Map

| Path | Purpose |
| --- | --- |
| `source/Gurukul/Gurukul/assets/` | Robot, actuator, sensor, and asset configuration |
| `source/Gurukul/Gurukul/tasks/` | Task implementations, configs, agents, and Gym registrations |
| `scripts/reinforcement_learning/` | Training, playback, distillation, and evaluation entry points |
| `scripts/tools/` | Inspection, conversion, visualization, and debugging tools |
| `tests/` | Targeted regression and contract tests |
| `unitree-sim2real/`, `engineai-sim2real/`, `deploy/` | Sim2sim, hardware, export, and deployment paths |
| `website/docs/` | Maintained public, user-facing documentation |
| `docs/` | Public-safe development notes, audits, and migration records |
| `knowledge/` | Optional gitignored local workspace for literature and implementation planning |
| `third_party/` | Vendored or adapted upstream projects |

## Reuse Before Creation

Before adding a task, config, runner, asset, or documentation page:

- Search the repository, including registrations with `rg "gym.register\(" source/Gurukul/Gurukul/tasks`.
- Check the closest page under `website/docs/tasks/` and `website/docs/reference/task-registry.md`.
- When Isaac Sim is available, use `scripts/tools/list_envs.py` as the runtime source of truth for registered tasks.
- Prefer extending, inheriting from, or composing an existing implementation over copying it.
- Do not add a new task ID merely to rename or duplicate existing behavior. Use an existing config or a documented CLI
  override when it expresses the variation clearly.
- Compatibility aliases are acceptable only when their purpose is explicit and they resolve to a maintained
  implementation.

A genuinely new task should provide distinct runnable behavior and include the relevant environment/config code, Gym
registration, agent entry point when training is supported, targeted validation, and an update to the nearest existing
public task page. Do not create a new documentation page when an existing task-family page is the clearer home.

## Public Documentation

The website is for users, not for agents or maintainers.

- Explain what the feature or task does, its prerequisites, exact commands, expected artifacts, and important
  limitations.
- Keep the public website free of prompts, agent instructions, private paths, investigation logs, and maintainer-only
  reasoning. Authoring rules belong in `AGENTS.md`, not in `website/`.
- Do not claim that a paper or method is implemented without corresponding code/configs and a runnable training,
  evaluation, or replay path.
- Describe support and validation at the exact task or runner level. Do not turn one checked path into a family-wide
  sim, sim2sim, sim2real, or hardware claim.
- Keep pages concise and scan-first: what the task is, then a video only when a real video exists, then main commands,
  then implementation details and field notes.
- Do not add placeholder sections such as "add notes here" or "video not recorded yet". Omit empty sections.
- When a task page has a Notes section, wrap it in `:::note[Field Notes]{.notebook-notes}`.
- Field Notes may include experimental observations, recommended workflows, known limitations, troubleshooting notes,
  and current interface behavior.
- Update the closest existing page rather than repeating the same instructions across multiple pages.
- If setup, training, robot assets, sim2sim, sim2real, task IDs, CLI flags, or artifact contracts change, update the
  corresponding public page in the same change.

Build documentation changes with:

```bash
cd website
npm ci
npm run build
```

## Working Notes and Private Maintainer Context

This is a public repository. Treat every tracked file, including files under `docs/`, as publishable.

- Use `docs/` for public-safe development notes, audits, and migration records that are useful to the repository but
  are not polished user documentation.
- Promote stable user-facing guidance to `website/docs/`; do not make users depend on working notes.
- Keep credentials, tokens, private integration details, collaborator-specific information, sensitive hardware
  information, local machine paths, and internal debugging or investigation notes outside this repository.
- Do not create a tracked "private" notes file. Gitignore is not a security boundary.
- If private context is needed for a public change, ask the maintainer for the minimum public-safe excerpt and record
  only the reusable behavior, test, or interface contract.

## Papers, Upstream Code, and Third-Party Material

- The `knowledge/` directory is local-only and should remain gitignored. Do not commit papers, unpublished manuscripts,
  review material, private notes, or copyrighted source material stored there.
- Before implementing a paper-inspired method, check whether a local `knowledge/AGENTS.md` exists and follow it.
- Add concise citation and credit metadata to the relevant public docs page when a method is paper-inspired.
- Preserve applicable upstream attribution and license files. Clearly distinguish vendored code, adapted code, and
  repository-authored code.
- Avoid modifying `third_party/` when a local adapter, subclass, or compatibility layer is sufficient.
- Do not add large datasets, motions, meshes, checkpoints, or generated artifacts without confirming provenance,
  redistribution rights, and whether they belong in Git/LFS or external storage.

## Validation

Choose checks according to the change and the available environment:

- Python logic: run the most targeted `python -m pytest <test-file-or-node> -q` first.
- Python style: run `pre-commit run --files <changed-files>` or targeted Ruff checks.
- Task registration: run `python scripts/tools/list_envs.py` when Isaac Sim can launch.
- Public docs: run `cd website && npm run build` after dependencies are installed.
- Training or simulation: prefer a short smoke test before a longer run.
- Sim2real or hardware: validate observation/action shapes, joint order, scales, gains, timing, artifact paths, and stop
  behavior statically before any device run.

Do not launch long training jobs, download large artifacts, or execute hardware commands unless the task requires it.
Do not imply simulator, GPU, checkpoint, or hardware validation happened when it did not; state the limitation and the
checks that did run.

## Definition of Done

A change is complete when:

- the requested behavior exists without unnecessary duplicate task/config surfaces;
- registrations, configs, scripts, artifacts, and docs agree;
- relevant targeted checks pass;
- user-facing behavior and limitations are documented;
- paper/upstream credit is recorded when applicable; and
- unperformed simulator, training, or hardware validation is reported explicitly.
