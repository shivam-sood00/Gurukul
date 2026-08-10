import os
import time
import sys

import numpy as np
import onnxruntime as ort
from unitree_sdk2py.core.channel import ChannelPublisher
from unitree_sdk2py.core.channel import ChannelSubscriber, ChannelFactoryInitialize
from unitree_sdk2py.idl.default import unitree_go_msg_dds__LowCmd_
from unitree_sdk2py.idl.unitree_go.msg.dds_ import LowCmd_, LowState_, WirelessController_
from unitree_sdk2py.utils.crc import CRC
from unitree_sdk2py.comm.motion_switcher.motion_switcher_client import MotionSwitcherClient
from unitree_sdk2py.go2.sport.sport_client import SportClient


def quat_to_rot(quat):
    w, x, y, z = quat
    R = np.array([
        [1 - 2 * (y ** 2 + z ** 2), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x ** 2 + z ** 2), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x ** 2 + y ** 2)]
    ])
    return R


class RLController(object):
    default_joint_angles = {
        'FL_hip_joint': 0.1, 'FL_thigh_joint': 0.8, 'FL_calf_joint': -1.5,
        'FR_hip_joint': -0.1, 'FR_thigh_joint': 0.8, 'FR_calf_joint': -1.5,
        'RL_hip_joint': 0.1, 'RL_thigh_joint': 1.0, 'RL_calf_joint': -1.5,
        'RR_hip_joint': -0.1, 'RR_thigh_joint': 1.0, 'RR_calf_joint': -1.5,
    }
    default_joint_angles = np.array(list(default_joint_angles.values()), dtype=np.float32)
    mapping = [3, 4, 5, 0, 1, 2, 9, 10, 11, 6, 7, 8]
    mapping_contact = [1, 0, 3, 2]
    frequency = 200.0
    dof_vel_limits = np.array([30, 30, 20, 30, 30, 20, 30, 30, 20, 30, 30, 20], dtype=np.float32)
    action_scale = 0.25
    torque_limit = 23.5
    key_state = [
        ["R1", 0], ["L1", 0], ["start", 0], ["select", 0], ["R2", 0], ["L2", 0], ["F1", 0], ["F2", 0], ["A", 0],
        ["B", 0], ["X", 0], ["Y", 0], ["up", 0], ["right", 0], ["down", 0], ["left", 0],
    ]

    # actor_path = 'all_gait_16Jan.onnx'
    actor_path = 'all_gait_23Dec2025.onnx'

    class obs_scales:
        lin_vel = 2.0
        ang_vel = 0.25
        dof_pos = 1.0
        dof_vel = 0.05
        height_measurements = 5.0

    def __init__(self):
        self.low_state = None
        # create subscriber and publisher
        self.low_state_suber = ChannelSubscriber("rt/lowstate", LowState_)
        self.low_state_suber.Init(self.LowStateHandler, 10)
        self.low_cmd_puber = ChannelPublisher("rt/lowcmd", LowCmd_)
        self.low_cmd_puber.Init()
        self.wireless_controller_suber = ChannelSubscriber("rt/wirelesscontroller", WirelessController_)
        self.wireless_controller_suber.Init(self.WirelessControllerHandler, 10)
        print('Creat subscriber and publisher')
        # load policy
        self.interval = 1.0 / self.frequency
        self.actor = self.load_policy()
        self.actor_input_name = self.actor.get_inputs()[0].name
        self.actor_output_name = self.actor.get_outputs()[0].name
        self.actor_input_shape = self.actor.get_inputs()[0].shape
        # prepare obs and actions
        self.step_count = 0
        self.command = np.array([[0.0, 0, 0]], dtype=np.float32)
        self.lin_vel = np.zeros((1, 3), dtype=np.float32)
        self.ang_vel = np.zeros((1, 3), dtype=np.float32)
        self.projected_gravity = np.zeros((1, 3), dtype=np.float32)
        self.dof_pos = np.zeros((1, 12), dtype=np.float32)
        self.dof_vel = np.zeros((1, 12), dtype=np.float32)
        self.torques = np.zeros((1, 12), dtype=np.float32)
        self.torques_est = np.zeros((1, 12), dtype=np.float32)
        self.activation_sign = np.zeros((1, 12), dtype=np.float32)
        self.motor_fatigue = np.zeros((1, 12), dtype=np.float32)
        self.motor_fatigue_est = np.zeros((1, 12), dtype=np.float32)
        # self.estimator_obs = np.zeros((1, 330), dtype=np.float32)
        self.prev_action = np.zeros((1, 12), dtype=np.float32)
        self.control_decimation = 4
        
        # Pre-allocate buffers for get_obs to avoid repeated allocations
        self.skill_number_array = np.zeros((1, 1), dtype=np.float32)
        self.commands_scaled = np.zeros((1, 3), dtype=np.float32)
        self.scaled_ang_vel = np.zeros((1, 3), dtype=np.float32)
        self.gravity_vec = np.array([0, 0, -1], dtype=np.float32)
        
        # Skill number control
        self.skill_number = 0.0
        # self.skill_values = [0.0, 0.25, 0.5, 0.75]
        self.skill_values = [0.0, 0.25, 0.5]
        self.skill_index = 0
        
        # Command clips for each skill (lin_vel_x, lin_vel_y, ang_vel_z)
        self.skill_command_clips = {
            0.0: {'lin_vel_x': [0.0, 1.0], 'lin_vel_y': [-0.1, 0.1], 'ang_vel_z': [-1.0, 1.0]},
            0.25: {'lin_vel_x': [0.0, 1.5], 'lin_vel_y': [-0.1, 0.1], 'ang_vel_z': [-1.5, 1.5]},
            0.5: {'lin_vel_x': [0.0, 2.0], 'lin_vel_y': [-0.1, 0.1], 'ang_vel_z': [-1.5, 1.5]},
            # 0.75: {'lin_vel_x': [0.0, 2.5], 'lin_vel_y': [-0.1, 0.1], 'ang_vel_z': [-1.5, 1.5]}
        }

        self.crc = CRC()
        self.cmd = unitree_go_msg_dds__LowCmd_()
        self.cmd.head[0] = 0xFE
        self.cmd.head[1] = 0xEF
        self.cmd.level_flag = 0xFF
        self.cmd.gpio = 0

        self.j_lx, self.j_ly, self.j_rx, self.j_ry = 0, 0, 0, 0

        for i in range(20):
            self.cmd.motor_cmd[i].mode = 0x01  # (PMSM) mode
            self.cmd.motor_cmd[i].q = 0.0
            self.cmd.motor_cmd[i].kp = 0.0
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kd = 0.0
            self.cmd.motor_cmd[i].tau = 0.0
        
        self.sc = SportClient()  
        self.sc.SetTimeout(5.0)
        self.sc.Init()

        self.msc = MotionSwitcherClient()
        self.msc.SetTimeout(5.0)
        self.msc.Init()

        status, result = self.msc.CheckMode()
        while result['name']:
            self.sc.StandDown()
            self.msc.ReleaseMode()
            status, result = self.msc.CheckMode()
            time.sleep(1)

        self.stand_up()

    def get_obs(self):
        self.ang_vel[0, :] = [
            self.low_state.imu_state.gyroscope[0],
            self.low_state.imu_state.gyroscope[1],
            self.low_state.imu_state.gyroscope[2]
        ]
        rot_matrix = quat_to_rot(self.low_state.imu_state.quaternion)
        self.projected_gravity[0] = rot_matrix.T @ self.gravity_vec

        dof_info = np.array(list(map(lambda x: x, self.low_state.motor_state))[:12])
        for i in range(12):
            self.dof_pos[0, self.mapping[i]] = dof_info[i].q - self.default_joint_angles[self.mapping[i]]
            self.dof_vel[0, self.mapping[i]] = dof_info[i].dq

        # Get command clips for current skill
        clips = self.skill_command_clips[self.skill_number]
        
        # Apply joystick inputs with clipping based on skill number
        lin_vel_x = self.j_ly * 2.0
        lin_vel_y = 0.0
        ang_vel_z = self.j_rx * (-1.5)
        
        self.command[0, 0] = np.clip(lin_vel_x, clips['lin_vel_x'][0], clips['lin_vel_x'][1])
        self.command[0, 1] = np.clip(lin_vel_y, clips['lin_vel_y'][0], clips['lin_vel_y'][1])
        self.command[0, 2] = np.clip(ang_vel_z, clips['ang_vel_z'][0], clips['ang_vel_z'][1])

        
        # Use skill_number instead of phase - use pre-allocated buffer
        self.skill_number_array[0, 0] = self.skill_number
        
        # Scale commands in-place
        self.commands_scaled[0, 0] = self.command[0, 0] * self.obs_scales.lin_vel
        self.commands_scaled[0, 1] = self.command[0, 1] * self.obs_scales.lin_vel
        self.commands_scaled[0, 2] = self.command[0, 2] * self.obs_scales.ang_vel

        np.multiply(self.ang_vel, self.obs_scales.ang_vel, out=self.scaled_ang_vel)

        return np.concatenate((self.scaled_ang_vel,
                               self.projected_gravity,
                               self.commands_scaled,
                               self.dof_pos * self.obs_scales.dof_pos,
                               self.dof_vel * self.obs_scales.dof_vel,
                               self.prev_action,
                               self.skill_number_array
                               ), axis=-1)

    def get_action(self):
        obs = self.get_obs()
        actions = self.actor.run([self.actor_output_name], {self.actor_input_name: obs})[0]
        return actions, self.compute_torque(actions)

    def compute_torque(self, actions):
        actions_scaled = actions * self.action_scale
        self.torques = actions_scaled

        return self.torques

    def run(self):
        raw_action, calculated_action = self.get_action()
        calculated_action = calculated_action[0]
        
        # Apply action control_decimation times (4x) for stability
        for j in range(self.control_decimation):
            for i, a in enumerate(self.mapping):
                self.cmd.motor_cmd[i].q = calculated_action[a] + self.default_joint_angles[i]
                self.cmd.motor_cmd[i].kp = 20.0
                self.cmd.motor_cmd[i].dq = 0.0
                self.cmd.motor_cmd[i].kd = 0.5
                self.cmd.motor_cmd[i].tau = 0.0
            
            self.cmd.crc = self.crc.Crc(self.cmd)
            self.low_cmd_puber.Write(self.cmd)

        # Store raw action directly (already has correct shape)
        np.copyto(self.prev_action, raw_action)

        if self.key_state[0][1] == 1 and self.key_state[1][1] == 1:
            self.e_stop()
            exit(0)

    def e_stop(self):
        for i in range(12):
            self.cmd.motor_cmd[i].q = 0.
            self.cmd.motor_cmd[i].kp = 0.0
            self.cmd.motor_cmd[i].dq = 0.0
            self.cmd.motor_cmd[i].kd = 0.
            self.cmd.motor_cmd[i].tau = 0.
        self.cmd.crc = CRC().Crc(self.cmd)
        self.low_cmd_puber.Write(self.cmd)

    def stand_up(self):
        runing_time = 0.0
        stand_up_joint_pos = np.array([
            0.00571868, 0.608813, -1.21763, -0.00571868, 0.608813, -1.21763,
            0.00571868, 0.608813, -1.21763, -0.00571868, 0.608813, -1.21763
        ],
            dtype=float)
        stand_down_joint_pos = np.array([
            0.0473455, 1.22187, -2.44375, -0.0473455, 1.22187, -2.44375, 0.0473455,
            1.22187, -2.44375, -0.0473455, 1.22187, -2.44375
        ],
            dtype=float)
        step_start = time.perf_counter()
        while True:
            runing_time += 0.002

            if (runing_time < 5.0):
                # Stand up in first 3 second

                # Total time for standing up or standing down is about 1.2s
                phase = np.tanh(runing_time / 1.2)
                for i in range(12):
                    self.cmd.motor_cmd[i].q = phase * stand_up_joint_pos[i] + (
                            1 - phase) * stand_down_joint_pos[i]
                    self.cmd.motor_cmd[i].kp = phase * 50.0 + (1 - phase) * 20.0
                    self.cmd.motor_cmd[i].dq = 0.0
                    self.cmd.motor_cmd[i].kd = 3.5
                    self.cmd.motor_cmd[i].tau = 0.0

            else:
                break

            self.cmd.crc = self.crc.Crc(self.cmd)
            self.low_cmd_puber.Write(self.cmd)

            time_until_next_step = 0.002 - (time.perf_counter() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)

    def load_policy(self):
        actor = ort.InferenceSession(self.actor_path)
        print('load actor from {}'.format(self.actor_path))
        # estimator = ort.InferenceSession(self.estimator_path)
        # print('load estimator from {}'.format(self.estimator_path))
        return actor

    def WirelessControllerHandler(self, msg: WirelessController_):
        self.j_lx = msg.lx
        self.j_ly = msg.ly
        self.j_rx = msg.rx
        self.j_ry = msg.ry
        
        # Check if X button is pressed (toggle skill number)
        prev_x_state = self.key_state[10][1]
        
        # Update key state
        for i in range(16):
            self.key_state[i][1] = (msg.keys & (1 << i)) >> i
        
        # Toggle skill number on X button press (rising edge)
        if self.key_state[10][1] == 1 and prev_x_state == 0:
            self.skill_index = (self.skill_index + 1) % len(self.skill_values)
            self.skill_number = self.skill_values[self.skill_index]
            print(f"Skill number changed to: {self.skill_number}")

    def LowStateHandler(self, msg: LowState_):
        self.low_state = msg


if __name__ == '__main__':
    print("WARNING: Please ensure there are no obstacles around the robot while running this example.")
    input("Press Enter to continue...")

    if len(sys.argv) > 1:
        ChannelFactoryInitialize(0, sys.argv[1])
    else:
        ChannelFactoryInitialize(1, "lo")

    rlcontroller = RLController()
    while True:
        start = time.time()
        rlcontroller.run()
        end = time.time()
        time.sleep(max(0., rlcontroller.interval + start - end))
