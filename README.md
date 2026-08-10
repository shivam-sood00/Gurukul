<div align="center">

<h1>
  <a href="https://shivamsood.org/Gurukul/"><img src="website/static/img/logo.svg" width="72" alt="Gurukul logo" align="middle"></a>&nbsp;Gurukul
</h1>

<p><strong>An open research workspace for robot learning</strong></p>

<p>
  Locomotion · Motion Tracking · Loco-Manipulation · Distillation · Multi-Robot Learning
</p>

<p>
  <a href="https://shivamsood.org/Gurukul/"><strong>Website &amp; Documentation →</strong></a>
</p>

<p>
  <a href="https://github.com/isaac-sim/IsaacLab"><img src="https://img.shields.io/badge/Built_on-Isaac_Lab-20c4b8?style=flat-square" alt="Built on Isaac Lab"></a>
  <a href="source/Gurukul/setup.py"><img src="https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&amp;logo=python&amp;logoColor=white" alt="Python 3.10 or newer"></a>
  <a href="LICENSE"><img src="https://img.shields.io/badge/License-Apache_2.0-78c56e?style=flat-square" alt="Apache License 2.0"></a>
</p>

</div>

---

Gurukul contains implementations of my own research work, along with a bunch of other cool research work. It spans
legged locomotion, motion imitation, manipulation, perception, and multi-robot learning.

The goal is to have everything in one place, making it easier to build on existing research work while also creating a
kind of knowledge graph for LLM agents to combine and build on existing, working research. The project is still in its
early stages and will be updated with more tasks and research work over time.
The implemented tasks have tags based on their current stage, showing whether they are completed baselines or still
under development. See the [Task Status](https://shivamsood.org/Gurukul/docs/reference/task-status) catalog
for details.

## Demos

<table>
<tr>
<td width="50%" align="center">

### [Locomotion](https://shivamsood.org/Gurukul/docs/tasks/velocity-locomotion/overview)

<a href="https://shivamsood.org/Gurukul/docs/tasks/velocity-locomotion/overview">
  <img src="https://img.youtube.com/vi/YSVoAav-n2I/hqdefault.jpg" width="100%" alt="Velocity locomotion demo">
</a>

</td>
<td width="50%" align="center">

### [Motion Tracking](https://shivamsood.org/Gurukul/docs/tasks/go2-apex)

<a href="https://shivamsood.org/Gurukul/docs/tasks/go2-apex">
  <img src="https://img.youtube.com/vi/B-rWg1W3ttk/hqdefault.jpg" width="100%" alt="Go2 APEX motion-tracking demo">
</a>

</td>
</tr>
<tr>
<td width="50%" align="center">

### [Loco-Manipulation](https://shivamsood.org/Gurukul/docs/tasks/loco-manipulation)

<a href="https://shivamsood.org/Gurukul/docs/tasks/loco-manipulation">
  <img src="https://img.youtube.com/vi/0FOvqjk3mTo/hqdefault.jpg" width="100%" alt="Go2 Airbot loco-manipulation demo">
</a>

</td>
<td width="50%" align="center">

### [Sim2Sim Transfer](https://shivamsood.org/Gurukul/docs/tasks/student-teacher/sim2sim)

<a href="https://shivamsood.org/Gurukul/docs/tasks/student-teacher/sim2sim">
  <img src="https://img.youtube.com/vi/WkIet5w7lNI/hqdefault.jpg" width="100%" alt="Student-teacher MuJoCo sim2sim demo">
</a>

</td>
</tr>
<tr>
<td colspan="2" align="center">

### [EngineAI PM01 Velocity](https://shivamsood.org/Gurukul/docs/tasks/velocity-locomotion/pm01)

<a href="https://shivamsood.org/Gurukul/docs/tasks/velocity-locomotion/pm01">
  <img src="https://img.youtube.com/vi/hJl8ZTJkCC0/hqdefault.jpg" width="50%" alt="EngineAI PM01 velocity locomotion demo">
</a>

</td>
</tr>
</table>

## Research Areas

| | |
| --- | --- |
| **Locomotion** | Velocity control, terrain adaptation, and LiDAR-aware policies |
| **Motion Tracking** | APEX, BeyondMimic, AMP, and reference-conditioned control |
| **Learning Methods** | Teacher-student learning, distillation, depth students, and action priors |
| **Manipulation** | Go2+D1, Go2+Airbot, B2+Z1, and Revo3 DexHand tasks |
| **Autonomy** | Language-model planning, perception-driven control, and multi-robot tasks |
| **Transfer** | Task-specific MuJoCo sim2sim and guarded hardware handoff paths |

Explore the implementations in the [Task Overview](https://shivamsood.org/Gurukul/docs/tasks/overview) and
check the [Task Status](https://shivamsood.org/Gurukul/docs/reference/task-status) catalog for exact support
and validation levels.

## Quick Start

Install [Isaac Lab](https://github.com/isaac-sim/IsaacLab), then:

```bash
git lfs install
git clone https://github.com/shivam-sood00/Gurukul.git
cd Gurukul

python -m pip install -e source/Gurukul
python -m pip install "rsl-rl-lib==5.3.0"
python scripts/tools/list_envs.py
```

Continue with the [Getting Started guide](https://shivamsood.org/Gurukul/docs/getting-started/installation) or the page
for the task you want to run.


## Acknowledgements & License

Gurukul builds on [Isaac Lab](https://github.com/isaac-sim/IsaacLab) and with the task structure originally based on
[fan-ziqi/robot_lab](https://github.com/fan-ziqi/robot_lab). Individual task pages document the papers, repositories,
datasets, and assets used by each implementation.

See the project [acknowledgements](https://shivamsood.org/Gurukul/docs/reference/credits),
[third-party notices](THIRD_PARTY_NOTICES.md), and [license](LICENSE) for details. Repository-authored code is provided
under the Apache License 2.0 except where individual files or bundled components specify otherwise.
