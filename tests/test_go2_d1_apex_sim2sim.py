from __future__ import annotations

import ast
import re
import types
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
SIM2SIM_CFG = (
    REPO_ROOT / "unitree-sim2real/RL_policy_runner/configs/Gurukul/go2_d1_arm_apex_flat_tracker_v0.yaml"
)
MJCF = REPO_ROOT / "unitree-sim2real/unitree_mujoco/unitree_robots/go2_with_d1/go2_with_d1.xml"
APEX_CFG = (
    REPO_ROOT / "source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/"
    "flat_d1_arm_tracker_env_cfg.py"
)
APEX_REWARDS = REPO_ROOT / "source/Gurukul/Gurukul/tasks/manager_based/go2_apex/mdp/rewards.py"
APEX_ACTIONS = REPO_ROOT / "source/Gurukul/Gurukul/tasks/manager_based/go2_apex/mdp/actions.py"
APEX_COMMANDS = REPO_ROOT / "source/Gurukul/Gurukul/tasks/manager_based/go2_apex/mdp/commands.py"
APEX_RUNNER_CFG = (
    REPO_ROOT / "source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/agents/"
    "rsl_rl_ppo_cfg.py"
)
UNITREE_ASSET = REPO_ROOT / "source/Gurukul/Gurukul/assets/unitree.py"
APEX_USD = REPO_ROOT / "source/Gurukul/data/Robots/unitree/go2_with_d1/usd/go2_d1_center_gripper_apex.usda"
PLAY_SCRIPT = REPO_ROOT / "scripts/reinforcement_learning/rsl_rl/play.py"
SIM2SIM_RUNNER = REPO_ROOT / "unitree-sim2real/RL_policy_runner/sim2sim/run_rl_policy.py"
MUJOCO_SIMULATOR = REPO_ROOT / "unitree-sim2real/unitree_mujoco/simulate_python/unitree_mujoco.py"
MUJOCO_BRIDGE = REPO_ROOT / "unitree-sim2real/unitree_mujoco/simulate_python/unitree_sdk2py_bridge.py"
APEX_HARDWARE = REPO_ROOT / "unitree-sim2real/RL_policy_runner/sim2real/go2_d1_arm_apex_hardware.py"


def test_go2_d1_sim2sim_matches_actor_and_motor_contracts():
    cfg = yaml.safe_load(SIM2SIM_CFG.read_text())
    runner_source = SIM2SIM_RUNNER.read_text()

    assert cfg["num_actions"] == 19
    assert cfg["num_joint_obs"] == 19
    assert cfg["num_motors"] == 19
    assert cfg["actor_joint_velocity_indices"] == list(range(12))
    # 10 base/command + 19 joint positions + 12 Go2 velocities + 19 actions.
    # D1 arm velocity channels are intentionally absent from the actor.
    assert cfg["num_obs"] == 60 + 1 + 5 * 29
    assert len(cfg["tracker_joint_names"]) == 19
    assert cfg["tracker_joint_names"][-1] == "arm_7_1_joint"
    assert cfg["motion_visualization"]["include_skill_observation"] is True
    assert cfg["motion_visualization"]["reference_offsets"] == [0, 1, 2, 5, 10]
    assert cfg["controlled_motor_indices"] == list(range(19))
    assert cfg["action_clip"] == 6.0
    np.testing.assert_allclose(
        cfg["default_angles"][12:18],
        np.deg2rad([0.0, -90.0, 90.0, 0.0, 0.0, 0.0]),
        atol=1.0e-10,
    )
    assert cfg["default_angles"][18] == 0.0
    assert cfg["action_scales_per_joint"][12:18] == [0.25] * 6
    assert cfg["action_scales_per_joint"][18] == 0.005
    assert cfg["position_target_lower_limits"][12:] == [-2.35, -1.57, -1.57, -2.35, -1.57, -2.35, 0.0]
    assert cfg["position_target_upper_limits"][12:] == [2.35, 1.57, 1.57, 2.35, 1.57, 2.35, 0.033]
    assert cfg["reference_residual_action_indices"] == []
    assert cfg["arm_command"]["enabled"] is True
    assert cfg["arm_command"]["model"] == "sample_and_hold"
    assert cfg["arm_command"]["indices"] == list(range(12, 19))
    assert cfg["arm_command"]["period_s"] == 0.1
    assert cfg["arm_command"]["quantization_rad"] == 0.0
    assert cfg["simulation_dt"] == 0.005
    assert cfg["control_decimation"] == 4
    assert cfg["standup_duration_s"] == 0.0
    assert "standup_start_kp" not in cfg
    assert cfg["standup_kp"] == "policy"
    assert cfg["standup_kd"] == "policy"
    assert 1.0 / (cfg["simulation_dt"] * cfg["control_decimation"]) == 50.0
    assert "np.multiply(actions, action_scales, out=self.torques)" in runner_source
    assert "position_target_lower_limits[motor_idx]" in runner_source
    assert "position_target_upper_limits[motor_idx]" in runner_source
    assert "_training_motion_file_for_export(run_policy_path)" in runner_source


@pytest.mark.skipif(not MJCF.exists(), reason="Optional Go2+D1 MuJoCo assets are not distributed.")
def test_go2_d1_mjcf_has_one_physical_gripper_command_and_ordered_sensors():
    root = ET.parse(MJCF).getroot()
    actuators = root.findall("./actuator/motor")
    position_sensors = root.findall("./sensor/jointpos")
    velocity_sensors = root.findall("./sensor/jointvel")
    torque_sensors = root.findall("./sensor/jointactuatorfrc")
    mimic = root.find("./equality/joint[@name='d1_gripper_mimic']")
    home_key = root.find("./keyframe/key[@name='home']")

    assert len(actuators) == 19
    assert len(position_sensors) == len(velocity_sensors) == len(torque_sensors) == 19
    assert home_key is not None
    np.testing.assert_array_equal(np.fromstring(home_key.attrib["ctrl"], sep=" "), np.zeros(19))
    assert [sensor.attrib["joint"] for sensor in position_sensors] == [
        sensor.attrib["joint"] for sensor in velocity_sensors
    ]
    assert [sensor.attrib["joint"] for sensor in position_sensors] == [
        sensor.attrib["joint"] for sensor in torque_sensors
    ]
    assert actuators[-1].attrib["joint"] == "arm_7_1_joint"
    assert mimic is not None
    assert mimic.attrib["joint1"] == "arm_7_2_joint"
    assert mimic.attrib["joint2"] == "arm_7_1_joint"
    joints = {joint.attrib["name"]: joint for joint in root.findall(".//joint[@name]")}
    assert joints["arm_2_joint"].attrib["range"] == "-1.570796327 1.570796327"
    assert joints["arm_3_joint"].attrib["range"] == "-1.570796327 1.570796327"


def test_mujoco_waits_for_first_low_command_before_advancing_physics():
    simulator_source = MUJOCO_SIMULATOR.read_text()
    bridge_source = MUJOCO_BRIDGE.read_text()

    assert "self.low_cmd_received = False" in bridge_source
    assert "self.low_cmd_received = True" in bridge_source
    assert "if unitree.low_cmd_received:" in simulator_source
    assert "mujoco.mj_step(mj_model, mj_data)" in simulator_source
    assert "mujoco.mj_forward(mj_model, mj_data)" in simulator_source


@pytest.mark.skipif(
    not MJCF.exists() or not APEX_USD.exists(),
    reason="Optional Go2+D1 Isaac/MuJoCo assets are not distributed.",
)
def test_go2_d1_apex_enables_filtered_nonadjacent_self_collisions_in_both_sims():
    apex_source = APEX_CFG.read_text()
    asset_source = UNITREE_ASSET.read_text()
    usd_source = APEX_USD.read_text()
    root = ET.parse(MJCF).getroot()

    assert "UNITREE_GO2_D1_ARM_APEX_CFG.spawn.articulation_props.enabled_self_collisions = True" in asset_source
    assert "_D1_NONADJACENT_SELF_COLLISION_FILTERS = (" in apex_source
    assert "ContactSensorCfg(" in apex_source
    assert "func=mdp.filtered_contact_pair_violations" in apex_source
    assert "weight=-2.0" in apex_source
    assert '"force_threshold": 5.0' in apex_source
    assert usd_source.count('prepend apiSchemas = ["PhysicsFilteredPairsAPI"]') == 7
    assert usd_source.count("prepend rel physics:filteredPairs") == 7

    d1_collision = root.find(".//default[@class='d1_collision']/geom")
    assert d1_collision is not None
    assert d1_collision.attrib["conaffinity"] == "1"
    excluded_pairs = {
        (exclude.attrib["body1"], exclude.attrib["body2"]) for exclude in root.findall("./contact/exclude")
    }
    assert excluded_pairs == {
        ("d1_base_link", "d1_link1"),
        ("d1_link1", "d1_link2"),
        ("d1_link2", "d1_link3"),
        ("d1_link2", "d1_link4"),
        ("d1_link2", "d1_link5"),
        ("d1_link3", "d1_link4"),
        ("d1_link4", "d1_link5"),
        ("d1_link5", "d1_link6"),
        ("d1_link6", "d1_finger_left"),
        ("d1_link6", "d1_finger_right"),
    }


def test_go2_d1_apex_uses_explicit_torque_servo_without_saturated_override():
    apex_source = APEX_CFG.read_text()
    asset_source = UNITREE_ASSET.read_text()
    cfg = yaml.safe_load(SIM2SIM_CFG.read_text())

    assert "4000.0" not in apex_source
    assert 'UNITREE_GO2_D1_ARM_APEX_CFG.actuators["arm_j1_j2"] = IdealPDActuatorCfg(' in asset_source
    assert 'UNITREE_GO2_D1_ARM_APEX_CFG.actuators["arm_j5_j6"] = IdealPDActuatorCfg(' in asset_source
    assert "effort_limit_sim=1.0e9" in asset_source
    assert "damping=5.0" in asset_source
    assert "damping=4.0" in asset_source
    assert asset_source.count("stiffness=50.0") >= 2
    assert asset_source.count("damping=0.25") >= 2
    assert asset_source.count("armature=0.001") >= 4
    assert asset_source.count("friction=0.02") >= 4
    assert asset_source.count("dynamic_friction=0.02") >= 4
    assert asset_source.count("viscous_friction=0.1") >= 4
    assert cfg["kps"][12:18] == [200.0, 200.0, 200.0, 50.0, 50.0, 50.0]
    assert cfg["kds"][12:18] == [5.0, 5.0, 4.0, 0.25, 0.25, 0.25]
    assert cfg["torque_limit"][12:18] == [3.3, 3.3, 3.3, 1.7, 1.7, 1.7]
    assert (
        'self.events.randomize_actuator_gains.params["asset_cfg"].joint_names = list(GO2_D1_ACTION_JOINT_NAMES)'
    ) in apex_source


@pytest.mark.skipif(
    not MJCF.exists() or not APEX_USD.exists(),
    reason="Optional Go2+D1 Isaac/MuJoCo assets are not distributed.",
)
def test_go2_d1_apex_uses_gripper_independent_full_wrist_inertias():
    source = APEX_USD.read_text()
    expected = {
        "Link4": {
            "com": (-0.00044332823, 0.000054163142, 0.088537998),
            "mjcf_com": (0.088538, 0.0000532, 0.000443),
            "moments": (0.000049973879, 0.0007160241, 0.00074700203),
        },
        "Link5": {
            "com": (0.04423572, 0.005275813, -0.025833514),
            "mjcf_com": (0.044236, 0.025833, 0.005276),
            "moments": (0.00003826195, 0.00006693525, 0.0000727028),
        },
        "Link6": {
            "com": (-0.0058941394, -0.000036683425, 0.03834998),
            "mjcf_com": (0.03835, -0.000037, 0.005894),
            "moments": (0.000114995535, 0.00012441425, 0.00018659022),
        },
    }
    mjcf_root = ET.parse(MJCF).getroot()

    for link_name, properties in expected.items():
        start = source.index(f'over "{link_name}" (')
        end = source.find('\n        over "Link', start + 1)
        block = source[start:] if end < 0 else source[start:end]

        def tuple_property(name: str) -> np.ndarray:
            match = re.search(rf"physics:{name} = \(([^)]+)\)", block)
            assert match is not None
            return np.fromstring(match.group(1), sep=",")

        np.testing.assert_allclose(tuple_property("centerOfMass"), properties["com"], rtol=0.0, atol=1.0e-9)
        np.testing.assert_allclose(tuple_property("diagonalInertia"), properties["moments"], rtol=1.0e-7)
        assert np.linalg.norm(tuple_property("principalAxes")) == pytest.approx(1.0, abs=1.0e-6)

        mjcf_inertial = mjcf_root.find(f".//body[@name='d1_{link_name.lower()}']/inertial")
        assert mjcf_inertial is not None
        np.testing.assert_allclose(np.fromstring(mjcf_inertial.attrib["pos"], sep=" "), properties["mjcf_com"])
        np.testing.assert_allclose(
            np.fromstring(mjcf_inertial.attrib["diaginertia"], sep=" "), properties["moments"], rtol=1.0e-7
        )
        assert np.linalg.norm(np.fromstring(mjcf_inertial.attrib["quat"], sep=" ")) == pytest.approx(
            1.0, abs=1.0e-6
        )

    assert "custom fingers and elastomer pads are separate child bodies" in source


def test_go2_d1_runner_can_record_arm_command_and_tracking_traces():
    runner_source = SIM2SIM_RUNNER.read_text()

    assert "class ArmTrackingLogger:" in runner_source
    assert '"--arm-trace-log"' in runner_source
    assert "desired_q=_stack(self.desired_q)" in runner_source
    assert "command_q=_stack(self.command_q)" in runner_source
    assert "measured_q=_stack(self.measured_q)" in runner_source
    assert "tau_est=_stack(self.tau_est)" in runner_source


def test_go2_d1_apex_uses_hardware_velocity_limits_as_soft_penalties():
    apex_source = APEX_CFG.read_text()
    reward_source = APEX_REWARDS.read_text()
    asset_source = UNITREE_ASSET.read_text()

    # The shared asset retains the published hardware limits as metadata for
    # non-APEX tasks, while APEX training raises only its PhysX solver limits.
    assert "D1_ARM_HARDWARE_VELOCITY_LIMITS = (1.05, 1.05, 1.05, 1.73, 1.73, 1.73)" in apex_source
    assert "D1_ARM_NONBINDING_SIM_VELOCITY_LIMIT = 100.0" in apex_source
    assert 'for actuator_name in ("arm_j1_j2", "arm_j3", "arm_j4", "arm_j5_j6")' in apex_source
    assert "actuator_cfg.velocity_limit_sim = D1_ARM_NONBINDING_SIM_VELOCITY_LIMIT" in apex_source
    assert "d1_arm_joint_velocity_limits" in apex_source
    assert "func=mdp.joint_velocity_soft_limits_l2" in apex_source
    assert "def joint_velocity_soft_limits_l2(" in reward_source
    assert "velocity_limit_sim=1.05" in asset_source
    assert "velocity_limit_sim=1.73" in asset_source


def test_go2_d1_separates_direct_rl_teacher_and_supervised_student_tasks():
    apex_source = APEX_CFG.read_text()
    action_source = APEX_ACTIONS.read_text()
    registry_source = (APEX_CFG.parent / "__init__.py").read_text()
    teacher_cfg_source = (APEX_CFG.parent / "agents/rsl_rl_teacher_cfg.py").read_text()
    distill_cfg_source = (APEX_CFG.parent / "agents/rsl_rl_distillation_cfg.py").read_text()

    normal_block, student_and_teacher_blocks = apex_source.split(
        "class UnitreeGo2D1ArmApexDistillationStudentEnvCfg", maxsplit=1
    )
    student_block, teacher_block = student_and_teacher_blocks.split(
        "class UnitreeGo2D1ArmApexOriginalDecapTeacherEnvCfg", maxsplit=1
    )
    normal_registry, remaining_registry = registry_source.split(
        'id="Gurukul-Isaac-Go2-D1-Arm-APEX-Distillation-Student-v0"', maxsplit=1
    )
    student_registry, _ = remaining_registry.split(
        'id="Gurukul-Isaac-Go2-D1-Arm-APEX-Original-DecAP-Teacher-v0"', maxsplit=1
    )

    assert "GO2_D1_SIM_DT = 0.005" in apex_source
    assert "GO2_D1_CONTROL_DECIMATION = 4" in apex_source
    assert "GO2_D1_CONTROL_FREQUENCY_HZ = 1.0 / (GO2_D1_SIM_DT * GO2_D1_CONTROL_DECIMATION)" in apex_source
    assert "self.sim.dt = GO2_D1_SIM_DT" in apex_source
    assert "self.decimation = GO2_D1_CONTROL_DECIMATION" in apex_source
    assert "UNITREE_GO2_D1_ARM_CFG" not in apex_source
    assert "self.events.randomize_d1_mount_x = None" not in apex_source
    assert "self.actions.joint_pos.decap_joint_names = tuple(GO2_D1_MOTION_JOINT_NAMES)" in normal_block
    assert "self.actions.joint_pos.decap_lambda_start = 1.0" in normal_block
    assert "self.actions.joint_pos.decap_lambda_end = 0.0" in normal_block
    assert 'self.actions.joint_pos.decap_decay_type = "cosine"' in normal_block
    assert "self.actions.joint_pos.decap_warmup_iterations = 100" in normal_block
    assert "self.actions.joint_pos.decap_decay_end_iteration = 1000" in normal_block
    assert "self.actions.joint_pos.reference_residual_joint_names = ()" in apex_source
    assert "self.rewards.d1_nonadjacent_self_collision = None" not in apex_source
    assert "self.actions.joint_pos.decap_joint_names = ()" not in normal_block
    assert "train_with_zero_decap: bool = True" in student_block
    assert "self.actions.joint_pos.decap_joint_names = ()" in student_block
    assert "self.actions.joint_pos.decap_lambda_start = 0.0" in student_block
    assert 'self.actions.joint_pos.decap_decay_type = "constant"' in student_block
    assert "self.actions.joint_pos.decap_joint_names = tuple(GO2_D1_MOTION_JOINT_NAMES)" in teacher_block
    assert 'D1_COMMAND_JOINT_NAMES = GO2_D1_ARM_JOINT_NAMES + ("arm_7_1_joint",)' in apex_source
    assert apex_source.count("self.actions.joint_pos.servo_joint_names = D1_COMMAND_JOINT_NAMES") >= 2
    assert "self.actions.joint_pos.servo_command_period_s = D1_ARM_COMMAND_PERIOD_S" in apex_source
    assert "self.actions.joint_pos.servo_command_quantization = 0.0" in apex_source
    assert "self.actions.joint_pos.servo_filter_enabled = False" in apex_source
    assert "if not self.cfg.servo_filter_enabled:" in action_source
    assert "target_blend" not in action_source
    assert "torch.lerp(" not in action_source
    assert "prior_delta[:, decap_indices] = decap_lambda * (" in action_source
    assert "self._processed_actions = self._processed_actions + prior_delta" in action_source
    assert "rsl_rl_distillation_cfg_entry_point" not in normal_registry.rsplit("gym.register(", maxsplit=1)[-1]
    assert "rsl_rl_teacher_cfg_entry_point" not in normal_registry.rsplit("gym.register(", maxsplit=1)[-1]
    assert "UnitreeGo2D1ArmApexDistillationStudentEnvCfg" in student_registry
    assert "rsl_rl_distillation_cfg_entry_point" in student_registry
    assert "UnitreeGo2D1ArmApexOriginalDecapTeacherPPORunnerCfg" in teacher_cfg_source
    assert 'self.experiment_name = "unitree_go2_d1_arm_apex_original_decap_teacher"' in teacher_cfg_source
    assert 'self.obs_groups = {"actor": ["privileged"], "critic": ["privileged"]}' in teacher_cfg_source
    assert "UnitreeGo2D1ArmApexDistillationStudentRunnerCfg" in distill_cfg_source
    assert 'teacher_experiment_name = "unitree_go2_d1_arm_apex_original_decap_teacher"' in distill_cfg_source
    assert 'obs_groups = {"student": ["policy"], "teacher": ["privileged"]}' in distill_cfg_source
    assert '"go2_d1_arm_apex_distillation_student_v0"' in SIM2SIM_RUNNER.read_text()
    assert '"experiment": "unitree_go2_d1_arm_apex_flat_tracker_distill"' in SIM2SIM_RUNNER.read_text()


@pytest.mark.skipif(not MJCF.exists(), reason="Optional Go2+D1 MuJoCo assets and motions are not distributed.")
def test_go2_d1_wave_motions_expose_the_physical_gripper_position_channel():
    apex_source = APEX_CFG.read_text()
    runner_source = SIM2SIM_RUNNER.read_text()
    assert "self.commands.motion.joint_names = list(GO2_D1_MOTION_JOINT_NAMES)" in apex_source
    assert 'self.commands.motion.gripper_joint_names = ("arm_7_1_joint",)' in apex_source
    assert "joint_pos = np.concatenate((joint_pos, gripper_joint_pos), axis=1)" in runner_source

    motion_root = (
        REPO_ROOT
        / "source/Gurukul/Gurukul/tasks/manager_based/go2_apex/config/go2/motion/npz/go2_d1"
    )
    for motion_name in ("student_showcase.npz", "walk_then_wave_hello.npz", "wave_hello.npz"):
        with np.load(motion_root / motion_name) as motion:
            assert motion["gripper_joint_names"].tolist() == ["arm_7_1_joint"]
            assert motion["gripper_joint_pos"].shape == (motion["joint_pos"].shape[0], 1)
            assert motion["gripper_joint_vel"].shape == motion["gripper_joint_pos"].shape
            np.testing.assert_array_equal(motion["gripper_joint_pos"], 0.0)


def _load_deployment_arm_adapter(include_gripper: bool = False, model: str = "second_order_angle"):
    tree = ast.parse(SIM2SIM_RUNNER.read_text())
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "DeploymentArmCommandAdapter"
    )
    module = ast.Module(body=[class_node], type_ignores=[])
    joint_count = 7 if include_gripper else 6
    quantization = np.full(joint_count, np.deg2rad(0.1), dtype=np.float32)
    if include_gripper:
        quantization[-1] = 0.00003391572456
    namespace = {
        "np": np,
        "arm_command_indices": np.arange(joint_count, dtype=np.int32),
        "arm_gripper_index": 6 if include_gripper else None,
        "arm_lower_limits": np.concatenate((np.full(6, -2.35, dtype=np.float32), np.array([0.0], dtype=np.float32)))
        if include_gripper
        else np.full(6, -2.35, dtype=np.float32),
        "arm_upper_limits": np.concatenate((np.full(6, 2.35, dtype=np.float32), np.array([0.033], dtype=np.float32)))
        if include_gripper
        else np.full(6, 2.35, dtype=np.float32),
        "arm_command_period_s": np.float32(0.1),
        "arm_max_joint_delta": np.float32(0.05),
        "arm_command_velocity": False,
        "arm_command_model": model,
        "arm_command_quantization": quantization,
        "arm_command_natural_frequency_hz": np.float32(6.0),
        "arm_command_damping_ratio": np.float32(1.0),
        "arm_command_latency_s": np.float32(0.02),
        "default_angles": np.zeros(joint_count, dtype=np.float32),
        "simulation_dt": 0.005,
    }
    exec(compile(ast.fix_missing_locations(module), str(SIM2SIM_RUNNER), "exec"), namespace)
    return namespace["DeploymentArmCommandAdapter"]


def test_sample_and_hold_d1_adapter_updates_rotary_targets_at_10_hz():
    adapter = _load_deployment_arm_adapter(model="sample_and_hold")()
    adapter.quantization[:] = 0.0
    cmd = types.SimpleNamespace(motor_cmd=[types.SimpleNamespace(q=0.1, dq=99.0) for _ in range(6)])
    low_state = types.SimpleNamespace(motor_state=[types.SimpleNamespace(q=0.0) for _ in range(6)])

    adapter.set_desired_from_cmd(cmd)
    adapter.apply(cmd, low_state, 0.005)
    np.testing.assert_allclose([motor.q for motor in cmd.motor_cmd], 0.1, atol=1.0e-7)

    held_samples = []
    for _ in range(10):
        for motor in cmd.motor_cmd:
            motor.q = 0.2
        adapter.set_desired_from_cmd(cmd)
        adapter.apply(cmd, low_state, 0.005)
        held_samples.append(float(cmd.motor_cmd[0].q))
    np.testing.assert_allclose(held_samples, 0.1, atol=1.0e-7)

    for _ in range(10):
        for motor in cmd.motor_cmd:
            motor.q = 0.2
        adapter.set_desired_from_cmd(cmd)
        adapter.apply(cmd, low_state, 0.005)
    assert cmd.motor_cmd[0].q == pytest.approx(0.2, abs=1.0e-7)
    assert all(motor.dq == 0.0 for motor in cmd.motor_cmd)


def test_apex_hardware_keeps_50_hz_policy_and_sends_d1_at_10_hz():
    source = APEX_HARDWARE.read_text()

    assert "POLICY_DT = 0.02  # 50 Hz outer policy rate" in source
    assert "D1_ARM_COMMAND_DT = 0.1  # D1 seven-angle position packet: 10 Hz" in source
    assert "arm_command_decimation = int(round(D1_ARM_COMMAND_DT / simulation_dt))" in source
    assert "if self.loop_counter % self.control_decimation == 0:" in source
    assert "if self.loop_counter % self.arm_command_decimation == 0:" in source
    assert "if int(running_time / 0.002) % 50 == 0:" in source


def test_second_order_d1_adapter_latches_quantized_targets_without_velocity_cap():
    adapter_cls = _load_deployment_arm_adapter()
    adapter = adapter_cls()
    cmd = types.SimpleNamespace(motor_cmd=[types.SimpleNamespace(q=0.1234, dq=99.0) for _ in range(6)])
    low_state = types.SimpleNamespace(motor_state=[types.SimpleNamespace(q=0.0) for _ in range(6)])

    samples = []
    for _ in range(120):
        for motor in cmd.motor_cmd:
            motor.q = 0.1234
        adapter.set_desired_from_cmd(cmd)
        adapter.apply(cmd, low_state, 0.005)
        samples.append(float(cmd.motor_cmd[0].q))

    quantum = np.deg2rad(0.1)
    assert adapter.segment_target_q[0] == pytest.approx(round(0.1234 / quantum) * quantum, abs=1.0e-7)
    assert samples[5] == pytest.approx(0.0, abs=1.0e-7)
    assert samples[-1] == pytest.approx(adapter.segment_target_q[0], abs=2.0e-4)
    assert all(motor.dq == 0.0 for motor in cmd.motor_cmd)


def test_second_order_d1_adapter_filters_the_physical_gripper_coordinate():
    adapter = _load_deployment_arm_adapter(include_gripper=True)()
    cmd = types.SimpleNamespace(motor_cmd=[types.SimpleNamespace(q=0.0, dq=99.0) for _ in range(7)])
    low_state = types.SimpleNamespace(motor_state=[types.SimpleNamespace(q=0.0) for _ in range(7)])

    samples = []
    for _ in range(120):
        cmd.motor_cmd[6].q = 0.01234
        adapter.set_desired_from_cmd(cmd)
        adapter.apply(cmd, low_state, 0.005)
        samples.append(float(cmd.motor_cmd[6].q))

    quantum = 0.00003391572456
    expected = round(0.01234 / quantum) * quantum
    assert adapter.segment_target_q[6] == pytest.approx(expected, abs=1.0e-7)
    assert samples[5] == pytest.approx(0.0, abs=1.0e-7)
    assert samples[-1] == pytest.approx(expected, abs=2.0e-4)
    assert cmd.motor_cmd[6].dq == 0.0


def test_go2_d1_apex_keeps_leg_tracking_gradient_and_regularizes_arm_targets():
    apex_source = APEX_CFG.read_text()
    reward_source = APEX_REWARDS.read_text()

    assert "self.rewards.imitate_joint_pos_legs" in apex_source
    assert '"std": 0.12' in apex_source
    assert "self.rewards.track_command_lin_vel_xy.weight = 3.0" in apex_source
    assert "self.rewards.imitate_leg_policy_targets" in apex_source
    assert "self.rewards.imitate_joint_pos_arms = None" in apex_source
    assert "self.rewards.imitate_arm_joint_pos_proximal" in apex_source
    assert "self.rewards.imitate_arm_joint_pos_wrist" in apex_source
    assert "func=mdp.motion_joint_position_error_exp_per_joint" in apex_source
    assert "weight=2.5" in apex_source
    assert '"std": 0.08' in apex_source
    assert "self.rewards.imitate_arm_policy_targets = None" in apex_source
    assert "self.rewards.imitate_arm_policy_targets_proximal" in apex_source
    assert "self.rewards.imitate_arm_policy_targets_wrist" in apex_source
    assert "weight=-0.25" in apex_source
    assert "weight=-0.75" in apex_source
    assert "self.rewards.imitate_gripper_policy_targets" in apex_source
    assert '"normalize_by_action_scale": True' in apex_source
    assert "func=mdp.motion_joint_action_error_l2" in apex_source
    assert "self.rewards.d1_arm_action_rate" in apex_source
    assert "self.rewards.d1_arm_action_smoothness" in apex_source
    assert "self.rewards.d1_arm_joint_velocity" in apex_source
    assert "def motion_joint_action_error_l2(" in reward_source
    assert "def motion_joint_position_error_exp_per_joint(" in reward_source
    assert "error = error / torch.clamp(torch.abs(action_scale), min=1.0e-6)" in reward_source
    assert "return torch.mean(torch.square(error), dim=-1)" in reward_source


def test_go2_d1_tracks_link6_orientation_from_the_reference_joints():
    apex_source = APEX_CFG.read_text()
    reward_source = APEX_REWARDS.read_text()
    command_source = APEX_COMMANDS.read_text()

    assert 'self.commands.motion.arm_ee_orientation_body_name = "Link6"' in apex_source
    assert "self.rewards.imitate_arm_ee_orientation = RewTerm(" in apex_source
    assert "func=mdp.motion_arm_ee_orientation_error_exp" in apex_source
    assert '"asset_cfg": SceneEntityCfg("robot", body_names=["Link6"])' in apex_source
    assert "def motion_arm_ee_orientation_error_exp(" in reward_source
    assert "def _derive_d1_link6_quat_w(" in command_source
    assert "def arm_ee_target_quat_w(self)" in command_source
    assert 'terms["arm_ee_orientation"] = quat_error_magnitude(' in command_source


def test_go2_d1_actor_uses_arm_positions_only_and_critic_keeps_simulator_state():
    apex_source = APEX_CFG.read_text()
    runner_source = APEX_RUNNER_CFG.read_text()
    observation_source = (
        REPO_ROOT / "source/Gurukul/Gurukul/tasks/manager_based/go2_apex/mdp/observations.py"
    ).read_text()

    assert (
        'self.observations.policy.joint_vel.params["asset_cfg"].joint_names = list(GO2_ACTION_JOINT_NAMES)'
    ) in apex_source
    assert (
        'self.events.randomize_actuator_gains.params["asset_cfg"].joint_names = list(GO2_D1_ACTION_JOINT_NAMES)'
    ) in apex_source
    assert ("class Go2D1ArmApexTrackerCriticObservationsCfg(Go2ApexPrivilegedTrackerObservationsCfg)") in apex_source
    assert (
        'self.observations.critic.joint_vel.params["asset_cfg"].joint_names = list(GO2_D1_ACTION_JOINT_NAMES)'
    ) in apex_source
    assert (
        'self.observations.critic.joint_torques.params["asset_cfg"].joint_names = list(GO2_D1_ACTION_JOINT_NAMES)'
    ) in apex_source
    assert "feet_contact_forces_b = ObsTerm(" in apex_source
    assert "def contact_forces_b(" in observation_source
    assert 'self.obs_groups = {"policy": ["policy"], "critic": ["critic"]}' in runner_source


def _load_apex_delta_logger_class():
    tree = ast.parse(PLAY_SCRIPT.read_text())
    class_node = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "ApexTrackerDeltaLogger"
    )
    module = ast.Module(
        body=[ast.ImportFrom(module="__future__", names=[ast.alias(name="annotations")], level=0), class_node],
        type_ignores=[],
    )
    namespace = {"np": np, "Mapping": Mapping, "Sequence": Sequence}
    exec(compile(ast.fix_missing_locations(module), str(PLAY_SCRIPT), "exec"), namespace)
    return namespace["ApexTrackerDeltaLogger"]


def test_play_delta_logger_supports_go2_and_go2_d1_actor_layouts():
    logger_cls = _load_apex_delta_logger_class()

    go2 = logger_cls._slices(np.arange(155, dtype=np.float32), np.zeros(12, dtype=np.float32))
    assert go2["command"].shape == (3,)
    assert go2["skill"].shape == (0,)
    assert go2["joint_pos"].shape == (12,)
    assert go2["reference_joint_pos"].shape == (5 * 12,)

    go2_d1 = logger_cls._slices(np.arange(206, dtype=np.float32), np.zeros(19, dtype=np.float32))
    assert go2_d1["command"].shape == (4,)
    assert go2_d1["skill"].shape == (1,)
    assert go2_d1["joint_pos"].shape == (19,)
    assert go2_d1["joint_vel"].shape == (12,)
    assert go2_d1["prev_action"].shape == (19,)
    assert go2_d1["reference_joint_pos"].shape == (5 * 19,)


def test_sim2sim_honors_configured_standup_duration():
    runner_source = SIM2SIM_RUNNER.read_text()

    assert "while running_time < standup_duration_s and rl_policy.running:" in runner_source
    assert '"disabled; direct policy handover"' in runner_source
    assert '"direct handover (stand-up disabled)"' in runner_source
    assert "while running_time < 5.0 and rl_policy.running:" not in runner_source
    assert "rl_policy.update_command_smoothing(0.0)" in runner_source
    assert "if loop_counter > 0 and loop_counter % control_decimation == 0:" in runner_source
