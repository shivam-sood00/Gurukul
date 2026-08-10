import mujoco
import numpy as np
import pygame
import shutil
import sys
import struct

from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelPublisher

from unitree_sdk2py.idl.unitree_go.msg.dds_ import SportModeState_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import WirelessController_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__SportModeState_
from unitree_sdk2py.idl.default import unitree_go_msg_dds__WirelessController_
from unitree_sdk2py.utils.thread import RecurrentThread

import config
if config.ROBOT in {"g1", "pm01"}:
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowCmd_
    from unitree_sdk2py.idl.unitree_hg.msg.dds_ import LowState_
    from unitree_sdk2py.idl.default import unitree_hg_msg_dds__LowState_ as LowState_default
else:
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_
    from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowState_
    from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowState_ as LowState_default

TOPIC_LOWCMD = "rt/lowcmd"
TOPIC_LOWSTATE = "rt/lowstate"
TOPIC_HIGHSTATE = "rt/sportmodestate"
TOPIC_WIRELESS_CONTROLLER = "rt/wirelesscontroller"

MOTOR_SENSOR_NUM = 3
NUM_MOTOR_IDL_GO = 20
NUM_MOTOR_IDL_HG = 35

class UnitreeSdk2Bridge:

    def __init__(self, mj_model, mj_data):
        self.mj_model = mj_model
        self.mj_data = mj_data

        self.num_motor = self.mj_model.nu
        self.dim_motor_sensor = MOTOR_SENSOR_NUM * self.num_motor
        self.dt = self.mj_model.opt.timestep
        self.idl_type = (self.num_motor > NUM_MOTOR_IDL_GO) # 0: unitree_go, 1: unitree_hg
        # Physics must not advance from a torque-motor keyframe before a
        # controller owns it. The low-state publishers still expose the home
        # pose so the policy can construct and publish its first command.
        self.low_cmd_received = False

        self.joystick = None

        # Resolve motor state from actuator joints instead of assuming joint
        # sensors occupy the first 3 * nu scalar slots. EngineAI's official
        # MJCF deliberately puts base and IMU sensors first.
        actuator_joint_ids = np.asarray(self.mj_model.actuator_trnid[: self.num_motor, 0], dtype=np.int32)
        if np.any(actuator_joint_ids < 0):
            raise ValueError("Every MuJoCo actuator must target a joint.")
        self.motor_qpos_adr = np.asarray(
            self.mj_model.jnt_qposadr[actuator_joint_ids], dtype=np.int32
        )
        self.motor_dof_adr = np.asarray(
            self.mj_model.jnt_dofadr[actuator_joint_ids], dtype=np.int32
        )

        self.imu_quaternion_sensor = self._first_sensor("imu_quaternion", "imu_quat")
        self.imu_gyro_sensor = self._first_sensor("imu_angular_velocity", "imu_gyro")
        self.imu_acc_sensor = self._first_sensor("imu_linear_acceleration", "imu_acc")
        self.frame_position_sensor = self._first_sensor("base_link_position", "frame_pos")
        self.frame_velocity_sensor = self._first_sensor("base_link_linear_velocity", "frame_vel")

        # Unitree sdk2 message
        self.low_state = LowState_default()
        self.low_state_puber = ChannelPublisher(TOPIC_LOWSTATE, LowState_)
        self.low_state_puber.Init()
        self.lowStateThread = RecurrentThread(
            interval=self.dt, target=self.PublishLowState, name="sim_lowstate"
        )
        self.lowStateThread.Start()

        self.high_state = unitree_go_msg_dds__SportModeState_()
        self.high_state_puber = ChannelPublisher(TOPIC_HIGHSTATE, SportModeState_)
        self.high_state_puber.Init()
        self.HighStateThread = RecurrentThread(
            interval=self.dt, target=self.PublishHighState, name="sim_highstate"
        )
        self.HighStateThread.Start()

        self.wireless_controller = unitree_go_msg_dds__WirelessController_()
        self.wireless_controller_puber = ChannelPublisher(
            TOPIC_WIRELESS_CONTROLLER, WirelessController_
        )
        self.wireless_controller_puber.Init()
        self.WirelessControllerThread = RecurrentThread(
            interval=0.01,
            target=self.PublishWirelessController,
            name="sim_wireless_controller",
        )
        self.WirelessControllerThread.Start()

        self.low_cmd_suber = ChannelSubscriber(TOPIC_LOWCMD, LowCmd_)
        self.low_cmd_suber.Init(self.LowCmdHandler, 10)

        # joystick
        self.key_map = {
            "R1": 0,
            "L1": 1,
            "start": 2,
            "select": 3,
            "R2": 4,
            "L2": 5,
            "F1": 6,
            "F2": 7,
            "A": 8,
            "B": 9,
            "X": 10,
            "Y": 11,
            "up": 12,
            "right": 13,
            "down": 14,
            "left": 15,
        }

    def LowCmdHandler(self, msg: LowCmd_):
        if self.mj_data != None:
            for i in range(self.num_motor):
                self.mj_data.ctrl[i] = (
                    msg.motor_cmd[i].tau
                    + msg.motor_cmd[i].kp
                    * (msg.motor_cmd[i].q - self.mj_data.qpos[self.motor_qpos_adr[i]])
                    + msg.motor_cmd[i].kd
                    * (
                        msg.motor_cmd[i].dq
                        - self.mj_data.qvel[self.motor_dof_adr[i]]
                    )
                )
            self.low_cmd_received = True

    def PublishLowState(self):
        if self.mj_data != None:
            for i in range(self.num_motor):
                self.low_state.motor_state[i].q = self.mj_data.qpos[self.motor_qpos_adr[i]]
                self.low_state.motor_state[i].dq = self.mj_data.qvel[self.motor_dof_adr[i]]
                self.low_state.motor_state[i].tau_est = self.mj_data.actuator_force[i]

            if self.imu_quaternion_sensor is not None:
                self.low_state.imu_state.quaternion[:] = self._sensor_values(
                    self.imu_quaternion_sensor
                )
            if self.imu_gyro_sensor is not None:
                self.low_state.imu_state.gyroscope[:] = self._sensor_values(self.imu_gyro_sensor)
            if self.imu_acc_sensor is not None:
                self.low_state.imu_state.accelerometer[:] = self._sensor_values(self.imu_acc_sensor)

            if self.joystick != None:
                pygame.event.get()
                # Buttons
                self.low_state.wireless_remote[2] = int(
                    "".join(
                        [
                            f"{key}"
                            for key in [
                                0,
                                0,
                                int(self.joystick.get_axis(self.axis_id["LT"]) > 0),
                                int(self.joystick.get_axis(self.axis_id["RT"]) > 0),
                                int(self.joystick.get_button(self.button_id["SELECT"])),
                                int(self.joystick.get_button(self.button_id["START"])),
                                int(self.joystick.get_button(self.button_id["LB"])),
                                int(self.joystick.get_button(self.button_id["RB"])),
                            ]
                        ]
                    ),
                    2,
                )
                self.low_state.wireless_remote[3] = int(
                    "".join(
                        [
                            f"{key}"
                            for key in [
                                int(self.joystick.get_hat(0)[0] < 0),  # left
                                int(self.joystick.get_hat(0)[1] < 0),  # down
                                int(self.joystick.get_hat(0)[0] > 0), # right
                                int(self.joystick.get_hat(0)[1] > 0),    # up
                                int(self.joystick.get_button(self.button_id["Y"])),     # Y
                                int(self.joystick.get_button(self.button_id["X"])),     # X
                                int(self.joystick.get_button(self.button_id["B"])),     # B
                                int(self.joystick.get_button(self.button_id["A"])),     # A
                            ]
                        ]
                    ),
                    2,
                )
                # Axes
                sticks = [
                    self.joystick.get_axis(self.axis_id["LX"]),
                    self.joystick.get_axis(self.axis_id["RX"]),
                    -self.joystick.get_axis(self.axis_id["RY"]),
                    -self.joystick.get_axis(self.axis_id["LY"]),
                ]
                packs = list(map(lambda x: struct.pack("f", x), sticks))
                self.low_state.wireless_remote[4:8] = packs[0]
                self.low_state.wireless_remote[8:12] = packs[1]
                self.low_state.wireless_remote[12:16] = packs[2]
                self.low_state.wireless_remote[20:24] = packs[3]

            self.low_state_puber.Write(self.low_state)

    def PublishHighState(self):

        if self.mj_data != None:
            if self.frame_position_sensor is not None:
                self.high_state.position[:] = self._sensor_values(self.frame_position_sensor)
            elif self.mj_model.nq >= 7:
                self.high_state.position[:] = self.mj_data.qpos[:3]

            if self.frame_velocity_sensor is not None:
                self.high_state.velocity[:] = self._sensor_values(self.frame_velocity_sensor)
            elif self.mj_model.nv >= 6:
                self.high_state.velocity[:] = self.mj_data.qvel[:3]

        self.high_state_puber.Write(self.high_state)

    def _first_sensor(self, *names):
        for name in names:
            sensor_id = mujoco.mj_name2id(
                self.mj_model, mujoco.mjtObj.mjOBJ_SENSOR, name
            )
            if sensor_id >= 0:
                return sensor_id
        return None

    def _sensor_values(self, sensor_id):
        address = int(self.mj_model.sensor_adr[sensor_id])
        dimension = int(self.mj_model.sensor_dim[sensor_id])
        return self.mj_data.sensordata[address : address + dimension]

    def PublishWirelessController(self):
        if self.joystick != None:
            pygame.event.get()
            key_state = [0] * 16
            key_state[self.key_map["R1"]] = self.joystick.get_button(
                self.button_id["RB"]
            )
            key_state[self.key_map["L1"]] = self.joystick.get_button(
                self.button_id["LB"]
            )
            key_state[self.key_map["start"]] = self.joystick.get_button(
                self.button_id["START"]
            )
            key_state[self.key_map["select"]] = self.joystick.get_button(
                self.button_id["SELECT"]
            )
            key_state[self.key_map["R2"]] = (
                self.joystick.get_axis(self.axis_id["RT"]) > 0
            )
            key_state[self.key_map["L2"]] = (
                self.joystick.get_axis(self.axis_id["LT"]) > 0
            )
            key_state[self.key_map["F1"]] = 0
            key_state[self.key_map["F2"]] = 0
            key_state[self.key_map["A"]] = self.joystick.get_button(self.button_id["A"])
            key_state[self.key_map["B"]] = self.joystick.get_button(self.button_id["B"])
            key_state[self.key_map["X"]] = self.joystick.get_button(self.button_id["X"])
            key_state[self.key_map["Y"]] = self.joystick.get_button(self.button_id["Y"])
            key_state[self.key_map["up"]] = self.joystick.get_hat(0)[1] > 0
            key_state[self.key_map["right"]] = self.joystick.get_hat(0)[0] > 0
            key_state[self.key_map["down"]] = self.joystick.get_hat(0)[1] < 0
            key_state[self.key_map["left"]] = self.joystick.get_hat(0)[0] < 0

            key_value = 0
            for i in range(16):
                key_value += key_state[i] << i

            self.wireless_controller.keys = key_value
            self.wireless_controller.lx = self.joystick.get_axis(self.axis_id["LX"])
            self.wireless_controller.ly = -self.joystick.get_axis(self.axis_id["LY"])
            self.wireless_controller.rx = self.joystick.get_axis(self.axis_id["RX"])
            self.wireless_controller.ry = -self.joystick.get_axis(self.axis_id["RY"])

            self.wireless_controller_puber.Write(self.wireless_controller)

    def SetupJoystick(self, device_id=0, js_type="xbox", required=True):
        pygame.init()
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count > device_id:
            self.joystick = pygame.joystick.Joystick(device_id)
            self.joystick.init()
            print(f"Gamepad detected: {self.joystick.get_name()} (device {device_id}, layout={js_type})")
        else:
            print("No gamepad detected; continuing without wireless controller input.")
            if required:
                sys.exit()
            return False

        if js_type == "xbox":
            self.axis_id = {
                "LX": 0,  # Left stick axis x
                "LY": 1,  # Left stick axis y
                "RX": 3,  # Right stick axis x
                "RY": 4,  # Right stick axis y
                "LT": 2,  # Left trigger
                "RT": 5,  # Right trigger
                "DX": 6,  # Directional pad x
                "DY": 7,  # Directional pad y
            }

            self.button_id = {
                "X": 2,
                "Y": 3,
                "B": 1,
                "A": 0,
                "LB": 4,
                "RB": 5,
                "SELECT": 6,
                "START": 7,
            }

        elif js_type == "switch":
            self.axis_id = {
                "LX": 0,  # Left stick axis x
                "LY": 1,  # Left stick axis y
                "RX": 2,  # Right stick axis x
                "RY": 3,  # Right stick axis y
                "LT": 5,  # Left trigger
                "RT": 4,  # Right trigger
                "DX": 6,  # Directional pad x
                "DY": 7,  # Directional pad y
            }

            self.button_id = {
                "X": 3,
                "Y": 4,
                "B": 1,
                "A": 0,
                "LB": 6,
                "RB": 7,
                "SELECT": 10,
                "START": 11,
            }
        else:
            print("Unsupported gamepad. ")
            self.joystick = None
            return False
        return True

    def PrintSceneInformation(self):
        width = min(shutil.get_terminal_size((100, 24)).columns, 120)
        rule = "-" * width

        def names(obj_type, count):
            rows = []
            for index in range(count):
                name = mujoco.mj_id2name(self.mj_model, obj_type, index)
                if name:
                    rows.append((index, name))
            return rows

        def trim(value, max_len):
            value = str(value)
            if len(value) <= max_len:
                return value
            return value[: max(1, max_len - 3)] + "..."

        def print_name_table(title, rows, index_label):
            print(f"\n{title} ({len(rows)})")
            print(rule)
            print(f"{index_label:>5}  name")
            name_width = max(16, width - 8)
            for index, name in rows:
                print(f"{index:>5}  {trim(name, name_width)}")

        links = names(mujoco._enums.mjtObj.mjOBJ_BODY, self.mj_model.nbody)
        joints = names(mujoco._enums.mjtObj.mjOBJ_JOINT, self.mj_model.njnt)
        actuators = names(mujoco._enums.mjtObj.mjOBJ_ACTUATOR, self.mj_model.nu)

        sensors = []
        data_index = 0
        for sensor_index in range(self.mj_model.nsensor):
            name = mujoco.mj_id2name(
                self.mj_model, mujoco._enums.mjtObj.mjOBJ_SENSOR, sensor_index
            )
            dim = int(self.mj_model.sensor_dim[sensor_index])
            if name:
                sensors.append((sensor_index, data_index, dim, name))
            data_index += dim

        print("\nMuJoCo model")
        print(rule)
        print(
            f"bodies={len(links)}  joints={len(joints)}  actuators={len(actuators)}  "
            f"sensors={len(sensors)}  sensor_values={data_index}"
        )
        print_name_table("Bodies", links, "body")
        print_name_table("Joints", joints, "joint")
        print_name_table("Actuators", actuators, "act")

        print(f"\nSensors ({len(sensors)})")
        print(rule)
        print(f"{'id':>5}  {'data':>5}  {'dim':>3}  name")
        name_width = max(16, width - 23)
        for sensor_index, data_index, dim, name in sensors:
            print(f"{sensor_index:>5}  {data_index:>5}  {dim:>3}  {trim(name, name_width)}")
        print()


class ElasticBand:

    def __init__(self):
        self.stiffness = 200
        self.damping = 100
        self.point = np.array([0, 0, 3])
        self.length = 0
        self.enable = True

    def Advance(self, x, dx):
        """
        Args:
          δx: desired position - current position
          dx: current velocity
        """
        δx = self.point - x
        distance = np.linalg.norm(δx)
        direction = δx / distance
        v = np.dot(dx, direction)
        f = (self.stiffness * (distance - self.length) - self.damping * v) * direction
        return f

    def MujuocoKeyCallback(self, key):
        glfw = mujoco.glfw.glfw
        if key == glfw.KEY_7:
            self.length -= 0.1
        if key == glfw.KEY_8:
            self.length += 0.1
        if key == glfw.KEY_9:
            self.enable = not self.enable
