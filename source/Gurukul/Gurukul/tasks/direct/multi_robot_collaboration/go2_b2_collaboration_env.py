from __future__ import annotations

import os
from collections.abc import Sequence

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectMARLEnv, DirectMARLEnvCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import PhysxCfg, SimulationCfg
from isaaclab.sim.spawners.from_files import GroundPlaneCfg, spawn_ground_plane
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply_inverse, quat_from_euler_xyz, sample_uniform

from Gurukul.assets.unitree import UNITREE_B2_CFG, UNITREE_GO2_CFG

GO2_AGENT = "go2"
B2_AGENT = "b2"
JOINT_ORDER = [
    "FR_hip_joint",
    "FR_thigh_joint",
    "FR_calf_joint",
    "FL_hip_joint",
    "FL_thigh_joint",
    "FL_calf_joint",
    "RR_hip_joint",
    "RR_thigh_joint",
    "RR_calf_joint",
    "RL_hip_joint",
    "RL_thigh_joint",
    "RL_calf_joint",
]


@configclass
class Go2B2CollaborationEnvCfg(DirectMARLEnvCfg):
    """Direct MARL task with one Unitree Go2 and one Unitree B2."""

    # env
    decimation = 4
    episode_length_s = 20.0
    possible_agents = [GO2_AGENT, B2_AGENT]
    action_spaces = {GO2_AGENT: 12, B2_AGENT: 12}
    observation_spaces = {GO2_AGENT: 51, B2_AGENT: 51}
    state_space = -1

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 200,
        render_interval=decimation,
        physx=PhysxCfg(
            gpu_found_lost_pairs_capacity=2**23,
            gpu_total_aggregate_pairs_capacity=2**23,
        ),
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=6.0, replicate_physics=True)

    # robots
    go2_cfg: ArticulationCfg = UNITREE_GO2_CFG.replace(
        prim_path="/World/envs/env_.*/Go2",
        init_state=UNITREE_GO2_CFG.init_state.replace(pos=(-0.8, 0.0, 0.38)),
    )
    b2_cfg: ArticulationCfg = UNITREE_B2_CFG.replace(
        prim_path="/World/envs/env_.*/B2",
        init_state=UNITREE_B2_CFG.init_state.replace(pos=(0.8, 0.0, 0.58)),
    )

    # task
    target_xy = (0.0, 2.5)
    desired_separation = 1.6
    max_separation = 4.0
    go2_min_height = 0.18
    b2_min_height = 0.32

    # reset
    reset_xy_noise = 0.35
    reset_yaw_noise = 0.35
    reset_joint_pos_noise = 0.05
    reset_joint_vel_noise = 0.05

    # actions
    go2_action_scale = 0.25
    b2_action_scale = 0.20
    use_low_level_policies = False
    go2_policy_path = ""
    b2_policy_path = ""
    low_level_decimation = 4
    high_level_velocity_scale = (1.5, 1.0, 1.0)

    # rewards
    rew_alive = 0.5
    rew_centroid_to_target = 3.0
    rew_formation = 1.5
    rew_individual_target = 0.5
    rew_action_l2 = -0.02
    rew_joint_vel_l2 = -0.0005
    rew_termination = -2.0


@configclass
class Go2B2HierarchicalCollaborationEnvCfg(Go2B2CollaborationEnvCfg):
    """Go2/B2 collaboration config with frozen low-level locomotion policies."""

    action_spaces = {GO2_AGENT: 3, B2_AGENT: 3}
    observation_spaces = {GO2_AGENT: 54, B2_AGENT: 54}

    use_low_level_policies = True
    go2_policy_path = ""
    b2_policy_path = ""
    low_level_decimation = 4
    high_level_velocity_scale = (1.5, 1.0, 1.0)


class Go2B2CollaborationEnv(DirectMARLEnv):
    """Two-robot cooperative formation task.

    Go2 and B2 are independent agents. They receive proprioception, last action,
    partner relative position, and a shared target vector. Both agents receive the
    same cooperative reward, which asks the pair centroid to reach the target while
    maintaining a preferred separation.
    """

    cfg: Go2B2CollaborationEnvCfg

    def __init__(self, cfg: Go2B2CollaborationEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        self.go2_joint_ids = self._joint_ids(self.go2)
        self.b2_joint_ids = self._joint_ids(self.b2)

        self.go2_default_joint_pos = self.go2.data.default_joint_pos[:, self.go2_joint_ids].clone()
        self.b2_default_joint_pos = self.b2.data.default_joint_pos[:, self.b2_joint_ids].clone()
        self.target_pos_w = torch.tensor((*self.cfg.target_xy, 0.0), device=self.device).repeat(self.num_envs, 1)
        self.high_level_velocity_scale = torch.tensor(self.cfg.high_level_velocity_scale, device=self.device)
        self.go2_low_level_actions = torch.zeros((self.num_envs, 12), device=self.device)
        self.b2_low_level_actions = torch.zeros((self.num_envs, 12), device=self.device)
        self._low_level_step_counter = 0

        self.go2_low_level_policy = None
        self.b2_low_level_policy = None
        if self.cfg.use_low_level_policies:
            self.go2_low_level_policy = self._load_low_level_policy(self.cfg.go2_policy_path, GO2_AGENT)
            self.b2_low_level_policy = self._load_low_level_policy(self.cfg.b2_policy_path, B2_AGENT)

    def _setup_scene(self):
        self.go2 = Articulation(self.cfg.go2_cfg)
        self.b2 = Articulation(self.cfg.b2_cfg)

        spawn_ground_plane(
            prim_path="/World/ground",
            cfg=GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                ),
            ),
        )
        self.scene.clone_environments(copy_from_source=False)
        if self.device == "cpu":
            self.scene.filter_collisions(global_prim_paths=["/World/ground"])

        self.scene.articulations[GO2_AGENT] = self.go2
        self.scene.articulations[B2_AGENT] = self.b2

        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: dict[str, torch.Tensor]) -> None:
        self.actions = {
            GO2_AGENT: torch.clamp(actions[GO2_AGENT], -1.0, 1.0),
            B2_AGENT: torch.clamp(actions[B2_AGENT], -1.0, 1.0),
        }

    def _apply_action(self) -> None:
        if self.cfg.use_low_level_policies:
            self._apply_low_level_policy_actions()
            return

        go2_target = self.go2_default_joint_pos + self.cfg.go2_action_scale * self.actions[GO2_AGENT]
        b2_target = self.b2_default_joint_pos + self.cfg.b2_action_scale * self.actions[B2_AGENT]
        self.go2.set_joint_position_target(go2_target, joint_ids=self.go2_joint_ids)
        self.b2.set_joint_position_target(b2_target, joint_ids=self.b2_joint_ids)

    def _get_observations(self) -> dict[str, torch.Tensor]:
        go2_obs = self._robot_obs(
            robot=self.go2,
            partner=self.b2,
            joint_ids=self.go2_joint_ids,
            default_joint_pos=self.go2_default_joint_pos,
            actions=self.actions[GO2_AGENT],
        )
        b2_obs = self._robot_obs(
            robot=self.b2,
            partner=self.go2,
            joint_ids=self.b2_joint_ids,
            default_joint_pos=self.b2_default_joint_pos,
            actions=self.actions[B2_AGENT],
        )
        return {GO2_AGENT: go2_obs, B2_AGENT: b2_obs}

    def _get_states(self) -> torch.Tensor:
        return torch.cat([self.obs_dict[agent] for agent in self.cfg.possible_agents], dim=-1)

    def _get_rewards(self) -> dict[str, torch.Tensor]:
        go2_pos = self.go2.data.root_pos_w
        b2_pos = self.b2.data.root_pos_w
        target_pos = self.target_pos_w + self.scene.env_origins

        centroid_xy = 0.5 * (go2_pos[:, :2] + b2_pos[:, :2])
        target_xy = target_pos[:, :2]
        centroid_dist = torch.linalg.norm(centroid_xy - target_xy, dim=-1)
        go2_target_dist = torch.linalg.norm(go2_pos[:, :2] - target_xy, dim=-1)
        b2_target_dist = torch.linalg.norm(b2_pos[:, :2] - target_xy, dim=-1)
        separation = torch.linalg.norm(go2_pos[:, :2] - b2_pos[:, :2], dim=-1)

        rew_target = self.cfg.rew_centroid_to_target * torch.exp(-centroid_dist)
        rew_individual = self.cfg.rew_individual_target * (
            torch.exp(-go2_target_dist) + torch.exp(-b2_target_dist)
        )
        rew_formation = self.cfg.rew_formation * torch.exp(
            -torch.square(separation - self.cfg.desired_separation)
        )
        action_penalty = self.cfg.rew_action_l2 * self._action_l2()
        joint_vel_penalty = self.cfg.rew_joint_vel_l2 * (
            torch.sum(torch.square(self.go2.data.joint_vel[:, self.go2_joint_ids]), dim=-1)
            + torch.sum(torch.square(self.b2.data.joint_vel[:, self.b2_joint_ids]), dim=-1)
        )
        termination_penalty = self.cfg.rew_termination * math_prod_bool(self.terminated_dict.values()).float()

        reward = (
            self.cfg.rew_alive
            + rew_target
            + rew_individual
            + rew_formation
            + action_penalty
            + joint_vel_penalty
            + termination_penalty
        )

        log = self.extras.setdefault("log", {})
        log["collab/centroid_dist"] = centroid_dist.mean()
        log["collab/separation"] = separation.mean()
        log["collab/rew_target"] = rew_target.mean()
        log["collab/rew_formation"] = rew_formation.mean()
        return {GO2_AGENT: reward, B2_AGENT: reward}

    def _get_dones(self) -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        separation = torch.linalg.norm(self.go2.data.root_pos_w[:, :2] - self.b2.data.root_pos_w[:, :2], dim=-1)
        fell = (
            (self.go2.data.root_pos_w[:, 2] < self.cfg.go2_min_height)
            | (self.b2.data.root_pos_w[:, 2] < self.cfg.b2_min_height)
            | (separation > self.cfg.max_separation)
        )
        terminated = {agent: fell for agent in self.cfg.possible_agents}
        time_outs = {agent: time_out for agent in self.cfg.possible_agents}
        return terminated, time_outs

    def _reset_idx(self, env_ids: Sequence[int] | torch.Tensor | None):
        if env_ids is None:
            env_ids = self.go2._ALL_INDICES
        super()._reset_idx(env_ids)

        self._reset_robot(self.go2, self.go2_joint_ids, env_ids)
        self._reset_robot(self.b2, self.b2_joint_ids, env_ids)
        if hasattr(self, "go2_low_level_actions"):
            self.go2_low_level_actions[env_ids] = 0.0
            self.b2_low_level_actions[env_ids] = 0.0

    def _joint_ids(self, robot: Articulation) -> list[int]:
        return [robot.joint_names.index(name) for name in JOINT_ORDER]

    def _robot_obs(
        self,
        robot: Articulation,
        partner: Articulation,
        joint_ids: list[int],
        default_joint_pos: torch.Tensor,
        actions: torch.Tensor,
    ) -> torch.Tensor:
        partner_rel_w = partner.data.root_pos_w - robot.data.root_pos_w
        partner_rel_b = quat_apply_inverse(robot.data.root_quat_w, partner_rel_w)
        target_rel_w = self.target_pos_w + self.scene.env_origins - robot.data.root_pos_w
        target_rel_b = quat_apply_inverse(robot.data.root_quat_w, target_rel_w)
        partner_dist = torch.linalg.norm(partner_rel_w[:, :2], dim=-1, keepdim=True)
        action_obs = self._action_observation(robot, actions)

        return torch.cat(
            (
                robot.data.root_lin_vel_b,
                robot.data.root_ang_vel_b,
                robot.data.projected_gravity_b,
                robot.data.joint_pos[:, joint_ids] - default_joint_pos,
                0.05 * robot.data.joint_vel[:, joint_ids],
                action_obs,
                partner_rel_b,
                target_rel_b[:, :2],
                partner_dist,
            ),
            dim=-1,
        )

    def _reset_robot(self, robot: Articulation, joint_ids: list[int], env_ids: Sequence[int] | torch.Tensor):
        joint_pos = robot.data.default_joint_pos[env_ids].clone()
        joint_vel = robot.data.default_joint_vel[env_ids].clone()
        joint_pos[:, joint_ids] += sample_uniform(
            -self.cfg.reset_joint_pos_noise,
            self.cfg.reset_joint_pos_noise,
            (len(env_ids), len(joint_ids)),
            self.device,
        )
        joint_vel[:, joint_ids] += sample_uniform(
            -self.cfg.reset_joint_vel_noise,
            self.cfg.reset_joint_vel_noise,
            (len(env_ids), len(joint_ids)),
            self.device,
        )

        root_state = robot.data.default_root_state[env_ids].clone()
        root_state[:, :3] += self.scene.env_origins[env_ids]
        root_state[:, 0:2] += sample_uniform(
            -self.cfg.reset_xy_noise,
            self.cfg.reset_xy_noise,
            (len(env_ids), 2),
            self.device,
        )
        yaw = sample_uniform(
            -self.cfg.reset_yaw_noise,
            self.cfg.reset_yaw_noise,
            (len(env_ids),),
            self.device,
        )
        root_state[:, 3:7] = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        root_state[:, 7:] = 0.0

        robot.write_root_pose_to_sim(root_state[:, :7], env_ids)
        robot.write_root_velocity_to_sim(root_state[:, 7:], env_ids)
        robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _action_observation(self, robot: Articulation, actions: torch.Tensor) -> torch.Tensor:
        if not self.cfg.use_low_level_policies:
            return actions
        low_level_actions = self.go2_low_level_actions if robot is self.go2 else self.b2_low_level_actions
        command = actions * self.high_level_velocity_scale
        return torch.cat((command, low_level_actions), dim=-1)

    def _action_l2(self) -> torch.Tensor:
        if self.cfg.use_low_level_policies:
            return torch.sum(torch.square(self.actions[GO2_AGENT]), dim=-1) + torch.sum(
                torch.square(self.actions[B2_AGENT]), dim=-1
            )
        return torch.sum(torch.square(self.actions[GO2_AGENT]), dim=-1) + torch.sum(
            torch.square(self.actions[B2_AGENT]), dim=-1
        )

    def _load_low_level_policy(self, policy_path: str, agent: str):
        if not policy_path:
            raise ValueError(
                f"{agent} low-level policy path is required when use_low_level_policies=True. "
                f"Set `{agent}_policy_path=/path/to/exported/policy.pt`."
            )
        if not os.path.isfile(policy_path):
            raise FileNotFoundError(f"{agent} low-level policy file does not exist: {policy_path}")
        policy = torch.jit.load(policy_path, map_location=self.device).eval()
        print(f"[INFO] Loaded frozen {agent} low-level policy from: {policy_path}")
        return policy

    def _apply_low_level_policy_actions(self) -> None:
        if self._low_level_step_counter % self.cfg.low_level_decimation == 0:
            with torch.no_grad():
                go2_obs = self._low_level_policy_obs(
                    self.go2,
                    self.go2_joint_ids,
                    self.go2_default_joint_pos,
                    self.actions[GO2_AGENT] * self.high_level_velocity_scale,
                    self.go2_low_level_actions,
                )
                b2_obs = self._low_level_policy_obs(
                    self.b2,
                    self.b2_joint_ids,
                    self.b2_default_joint_pos,
                    self.actions[B2_AGENT] * self.high_level_velocity_scale,
                    self.b2_low_level_actions,
                )
                self.go2_low_level_actions[:] = torch.clamp(
                    self._resolve_policy_output(self.go2_low_level_policy(go2_obs)),
                    -1.0,
                    1.0,
                )
                self.b2_low_level_actions[:] = torch.clamp(
                    self._resolve_policy_output(self.b2_low_level_policy(b2_obs)),
                    -1.0,
                    1.0,
                )
            self._low_level_step_counter = 0

        self.go2.set_joint_position_target(
            self.go2_default_joint_pos + self._joint_action_scale() * self.go2_low_level_actions,
            joint_ids=self.go2_joint_ids,
        )
        self.b2.set_joint_position_target(
            self.b2_default_joint_pos + self._joint_action_scale() * self.b2_low_level_actions,
            joint_ids=self.b2_joint_ids,
        )
        self._low_level_step_counter += 1

    def _low_level_policy_obs(
        self,
        robot: Articulation,
        joint_ids: list[int],
        default_joint_pos: torch.Tensor,
        velocity_command: torch.Tensor,
        low_level_actions: torch.Tensor,
    ) -> torch.Tensor:
        return torch.cat(
            (
                0.25 * robot.data.root_ang_vel_b,
                robot.data.projected_gravity_b,
                velocity_command,
                robot.data.joint_pos[:, joint_ids] - default_joint_pos,
                0.05 * robot.data.joint_vel[:, joint_ids],
                low_level_actions,
            ),
            dim=-1,
        )

    def _joint_action_scale(self) -> torch.Tensor:
        scale = torch.full((12,), 0.25, device=self.device)
        scale[0::3] = 0.125
        return scale

    def _resolve_policy_output(self, output) -> torch.Tensor:
        if isinstance(output, torch.Tensor):
            return output
        if isinstance(output, dict):
            for key in ("actions", "action", "mean_actions", "policy"):
                if key in output:
                    return output[key]
        if isinstance(output, tuple | list) and output:
            return self._resolve_policy_output(output[0])
        raise TypeError(f"Unsupported low-level policy output type: {type(output)!r}")


def math_prod_bool(values) -> torch.Tensor:
    out = None
    for value in values:
        out = value if out is None else out & value
    return out
