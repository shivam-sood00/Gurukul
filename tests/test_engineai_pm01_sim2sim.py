from __future__ import annotations

import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PM01_ROOT = REPO_ROOT / "engineai-sim2real/engineai_mujoco/engineai_robots/pm01"
SCENE = PM01_ROOT / "scene.xml"
UPSTREAM = PM01_ROOT / "official/UPSTREAM.md"
URDF = REPO_ROOT / "source/Gurukul/data/Robots/engineai/pm01_description/urdf/pm01.urdf"
ASSET_CFG = REPO_ROOT / "source/Gurukul/Gurukul/assets/engineai_pm01.py"
OFFICIAL_ASSET_CFG = REPO_ROOT / "source/Gurukul/Gurukul/assets/engineai_pm01_official.py"
OFFICIAL_VISUAL_ASSET = (
    REPO_ROOT
    / "source/Gurukul/data/Robots/engineai/pm01_24dof/configuration/serial_pm01_edu_base.usd"
)
ROUGH_CFG = (
    REPO_ROOT
    / "source/Gurukul/Gurukul/tasks/manager_based/locomotion/velocity/config/humanoid"
    / "engineai_pm01/rough_env_cfg.py"
)
FLAT_CFG = ROUGH_CFG.with_name("flat_env_cfg.py")
COMMANDS_CFG = ROUGH_CFG.with_name("engineai_pm01_commands_actions_cfg.py")
CONSTANTS_CFG = ROUGH_CFG.with_name("pm01_constants.py")
VELOCITY_COMMANDS = (
    REPO_ROOT
    / "source/Gurukul/Gurukul/tasks/manager_based/locomotion/velocity/mdp/commands.py"
)
RL_UTILS = REPO_ROOT / "scripts/reinforcement_learning/rl_utils.py"
PLAY_SCRIPT = REPO_ROOT / "scripts/reinforcement_learning/rsl_rl/play.py"
DEPLOY_CONFIG = (
    REPO_ROOT
    / "source/Gurukul/Gurukul/tasks/manager_based/beyondmimic/config"
    / "engineai_pm01_24dof/pretrained/deploy_config.yaml"
)
CONFIG = (
    REPO_ROOT
    / "engineai-sim2real/RL_policy_runner/configs/Gurukul/engineai_pm01_flat_v0.yaml"
)
RUNNER = REPO_ROOT / "unitree-sim2real/RL_policy_runner/sim2sim/run_rl_policy.py"


def test_pm01_official_urdf_keeps_all_24_revolute_joints():
    root = ET.parse(URDF).getroot()
    joints = [joint for joint in root.findall("joint") if joint.attrib["type"] != "fixed"]
    names = [joint.attrib["name"] for joint in joints]
    mesh_paths = [URDF.parent / mesh.attrib["filename"] for mesh in root.findall(".//mesh")]

    assert root.attrib["name"] == "pm01"
    assert names == [f"j{index:02d}_{suffix}" for index, suffix in enumerate(
        (
            "hip_pitch_l", "hip_roll_l", "hip_yaw_l", "knee_pitch_l", "ankle_pitch_l", "ankle_roll_l",
            "hip_pitch_r", "hip_roll_r", "hip_yaw_r", "knee_pitch_r", "ankle_pitch_r", "ankle_roll_r",
            "waist_yaw", "shoulder_pitch_l", "shoulder_roll_l", "shoulder_yaw_l", "elbow_pitch_l",
            "elbow_yaw_l", "shoulder_pitch_r", "shoulder_roll_r", "shoulder_yaw_r", "elbow_pitch_r",
            "elbow_yaw_r", "head_yaw",
        )
    )]
    assert all(joint.attrib["type"] == "revolute" for joint in joints)
    assert mesh_paths
    assert all(path.is_file() for path in mesh_paths)
    assert {path.suffix for path in mesh_paths} == {".stl"}


def test_pm01_velocity_uses_official_24dof_gains_and_interface():
    cfg = yaml.safe_load(CONFIG.read_text())
    deploy = yaml.safe_load(DEPLOY_CONFIG.read_text())

    assert cfg["observation_layout"] == "isaac_term_history"
    assert cfg["history_length"] == 15
    assert cfg["num_obs"] == 1173
    assert cfg["num_actions"] == 24
    assert cfg["num_joint_obs"] == 24
    assert cfg["num_motors"] == 24
    assert cfg["controlled_motor_indices"] == list(range(24))
    assert cfg["joint_mapping"] == list(range(24))
    assert cfg["joint_obs_mapping"] == list(range(24))
    assert cfg["control_decimation"] == 10
    assert cfg["simulation_dt"] == pytest.approx(0.002)

    deploy_by_joint = {
        name: (kp, kd, scale)
        for name, kp, kd, scale in zip(
            deploy["joint_names"],
            deploy["joint_stiffness"],
            deploy["joint_damping"],
            deploy["action_scale"],
        )
    }
    expected = [deploy_by_joint[name] for name in cfg["motor_names"]]
    np.testing.assert_allclose(cfg["kps"], [values[0] for values in expected], rtol=0, atol=1.0e-6)
    np.testing.assert_allclose(cfg["kds"], [values[1] for values in expected], rtol=0, atol=1.0e-6)
    np.testing.assert_allclose(
        cfg["action_scales_per_joint"], [values[2] for values in expected], rtol=0, atol=1.0e-6
    )


def test_pm01_velocity_source_uses_official_materials_and_preserves_the_24dof_contract():
    asset_source = ASSET_CFG.read_text()
    official_asset_source = OFFICIAL_ASSET_CFG.read_text()
    env_source = ROUGH_CFG.read_text()
    flat_env_source = FLAT_CFG.read_text()
    commands_source = COMMANDS_CFG.read_text()
    constants_source = CONSTANTS_CFG.read_text()
    velocity_commands_source = VELOCITY_COMMANDS.read_text()
    rl_utils_source = RL_UTILS.read_text()
    play_source = PLAY_SCRIPT.read_text()

    assert "pm01_description/urdf/pm01.urdf" in asset_source
    assert '"j23_head_yaw"' in asset_source
    assert '".*head.*"' in asset_source
    assert "DelayedImplicitActuatorCfg" not in asset_source
    assert "self_collision=False" in asset_source
    assert "enabled_self_collisions=False" in asset_source
    assert "PM01_STIFFNESS_Q90" in asset_source
    assert "PM01_STIFFNESS_Q25" in asset_source
    assert "visual_material=" not in asset_source
    assert "serial_pm01_edu.usd" in official_asset_source
    assert "visual_material=" not in official_asset_source
    assert OFFICIAL_VISUAL_ASSET.stat().st_size > 100_000_000
    assert "ENGINEAI_PM01_24DOF_CFG" in env_source
    assert "self_collision = ContactSensorCfg" not in env_source
    assert 'base_link_name = "LINK_BASE"' in env_source
    assert 'foot_link_name = "LINK_ANKLE_ROLL_[LR]"' in env_source
    assert "enabled_self_collisions = False" in env_source
    assert "self.viewer.eye = (1.5, 1.5, 1.5)" in env_source
    assert 'self.viewer.origin_type = "asset_root"' in env_source
    assert 'self.viewer.asset_name = "robot"' in env_source
    assert "self.decimation = 10" in env_source
    assert "self.observations.critic.base_lin_vel.history_length = _HIST_LEN" in env_source
    assert "EngineAiPm01SceneCfg" not in env_source
    assert "visual_material" not in flat_env_source
    assert '"J00_HIP_PITCH_L"' in constants_source
    assert '"J23_HEAD_YAW"' in constants_source
    assert '"j00_hip_pitch_l"' not in constants_source
    assert "PM01_24DOF_ACTION_SCALE" in commands_source
    assert "marker_height_offset=0.65" in commands_source
    assert "base_pos_w[:, 2] += float(self.cfg.marker_height_offset)" in velocity_commands_source
    assert "follow_offset: tuple[float, float, float] | None = None" in rl_utils_source
    assert "offset = follow_offset if follow_offset is not None else (-3.0, 0.0, 0.5)" in rl_utils_source
    assert '"--camera-follow-distance-scale"' in play_source
    assert "base_offset = env_cfg.viewer.eye if env_cfg.viewer.origin_type == \"asset_root\"" in play_source
    assert "camera_follow_offset = tuple(distance_scale * float(value) for value in base_offset)" in play_source
    assert "follow_offset=camera_follow_offset" in play_source


def test_pm01_official_mujoco_model_contract():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(SCENE.resolve()))

    assert (model.nq, model.nv, model.nu) == (31, 30, 24)
    assert (model.nbody, model.ngeom, model.nsensor) == (30, 58, 15)
    assert model.opt.timestep == pytest.approx(0.002)
    assert int(model.opt.integrator) == 3  # implicitfast
    assert float(model.body_mass.sum()) == pytest.approx(40.92281135)
    assert sum(int(value != 0) for value in model.geom_contype) == 33

    actuator_names = [
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, index)
        for index in range(model.nu)
    ]
    assert actuator_names[0] == "motor_J00_HIP_PITCH_L"
    assert actuator_names[-1] == "motor_J23_HEAD_YAW"


def test_pm01_runner_preserves_term_level_history_and_provenance():
    source = RUNNER.read_text()
    provenance = UPSTREAM.read_text()

    assert 'observation_layout == "isaac_term_history" and term_index == 2' in source
    assert "(ISAAC_HISTORY_SINGLE_OBS_DIM - 3) * ISAAC_HISTORY_LENGTH + 3" in source
    assert "83204a459e0e786f855235a8507197496a79acc7" in provenance
    assert "BSD-3-Clause" in provenance
