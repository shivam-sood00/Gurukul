from __future__ import annotations

import glob
import math
import os
from collections.abc import Sequence
from dataclasses import MISSING
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import (
    BLUE_ARROW_X_MARKER_CFG,
    FRAME_MARKER_CFG,
    GREEN_ARROW_X_MARKER_CFG,
    SPHERE_MARKER_CFG,
)
from isaaclab.utils import configclass
from isaaclab.utils.math import (
    axis_angle_from_quat,
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inv,
    quat_mul,
    sample_uniform,
    yaw_quat,
)

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class BinaryJointPositionCommand(CommandTerm):
    """Sample a binary joint-position command such as open/close gripper targets."""

    cfg: BinaryJointPositionCommandCfg

    def __init__(self, cfg: BinaryJointPositionCommandCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.robot: Articulation = env.scene[cfg.asset_name]
        self._joint_ids = self.robot.find_joints(cfg.joint_names, preserve_order=True)[0]
        self._command = torch.zeros(self.num_envs, len(cfg.joint_names), device=self.device)
        self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)

    @property
    def command(self) -> torch.Tensor:
        return self._command

    def _update_metrics(self):
        joint_pos = self.robot.data.joint_pos[:, self._joint_ids]
        self.metrics["error_joint_pos"] = torch.mean(torch.square(joint_pos - self._command), dim=-1)

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        low = torch.tensor(self.cfg.low_position, device=self.device, dtype=self._command.dtype)
        high = torch.tensor(self.cfg.high_position, device=self.device, dtype=self._command.dtype)
        choose_high = torch.rand((len(env_ids), len(self.cfg.joint_names)), device=self.device) < self.cfg.high_prob
        self._command[env_ids] = torch.where(choose_high, high, low)

    def _update_command(self):
        pass


class MotionLoader:
    def __init__(
        self,
        motion_file: str | Sequence[str],
        body_indexes: Sequence[int] | None = None,
        device: str = "cpu",
    ):
        motion_files = self._resolve_motion_files(motion_file)
        loaded_motions = [self._load_motion_file(path) for path in motion_files]
        self._validate_motion_metadata(loaded_motions)

        first_motion = loaded_motions[0]
        self.motion_files = motion_files
        self.motion_names = self._display_names_from_paths(motion_files)
        self.fps = first_motion["fps"]
        self.fps_values = torch.tensor(
            [float(motion["fps"]) for motion in loaded_motions],
            dtype=torch.float32,
            device=device,
        )
        self.body_names = first_motion["body_names"]
        self.joint_names = first_motion["joint_names"]
        self.joint_pos = torch.tensor(
            np.concatenate([motion["joint_pos"] for motion in loaded_motions], axis=0),
            dtype=torch.float32,
            device=device,
        )
        self.joint_vel = torch.tensor(
            np.concatenate([motion["joint_vel"] for motion in loaded_motions], axis=0),
            dtype=torch.float32,
            device=device,
        )
        # Skill is optional for backward compatibility. Clips without it receive the
        # neutral scalar 0 while clips that provide it retain their frame-aligned value.
        self.skill = torch.tensor(
            np.concatenate(
                [
                    motion["skill"]
                    if motion["skill"] is not None
                    else np.zeros((motion["joint_pos"].shape[0], 1), dtype=np.float32)
                    for motion in loaded_motions
                ],
                axis=0,
            ),
            dtype=torch.float32,
            device=device,
        )

        gripper_joint_pos = [motion["gripper_joint_pos"] for motion in loaded_motions]
        if all(value is not None for value in gripper_joint_pos):
            self.gripper_joint_names = first_motion["gripper_joint_names"]
            self.gripper_joint_pos = torch.tensor(
                np.concatenate(gripper_joint_pos, axis=0),
                dtype=torch.float32,
                device=device,
            )
            self.gripper_joint_vel = torch.tensor(
                np.concatenate([motion["gripper_joint_vel"] for motion in loaded_motions], axis=0),
                dtype=torch.float32,
                device=device,
            )
        else:
            # Mixed motion libraries may contain locomotion clips without a gripper
            # channel. Tasks that command a gripper must select a compatible clip.
            self.gripper_joint_names = None
            self.gripper_joint_pos = None
            self.gripper_joint_vel = None

        object_pos_w = [motion["object_pos_w"] for motion in loaded_motions]
        if all(value is not None for value in object_pos_w):
            self.object_names = first_motion["object_names"]
            self.object_pos_w = torch.tensor(
                np.concatenate(object_pos_w, axis=0),
                dtype=torch.float32,
                device=device,
            )
            self.object_quat_w = torch.tensor(
                np.concatenate([motion["object_quat_w"] for motion in loaded_motions], axis=0),
                dtype=torch.float32,
                device=device,
            )
        else:
            self.object_names = None
            self.object_pos_w = None
            self.object_quat_w = None

        object_attached = [motion["object_attached"] for motion in loaded_motions]
        if all(value is None for value in object_attached):
            self.object_attached = None
        elif any(value is None for value in object_attached):
            raise ValueError("All loaded motion files must either include or omit 'object_attached'.")
        else:
            self.object_attached = torch.tensor(
                np.concatenate(object_attached, axis=0),
                dtype=torch.bool,
                device=device,
            )

        command_lin_vel_xy = [motion["command_lin_vel_xy"] for motion in loaded_motions]
        if all(value is None for value in command_lin_vel_xy):
            self.command_lin_vel_xy = None
        elif any(value is None for value in command_lin_vel_xy):
            raise ValueError("All loaded motion files must either include or omit 'command_lin_vel_xy'.")
        else:
            self.command_lin_vel_xy = torch.tensor(
                np.concatenate(command_lin_vel_xy, axis=0),
                dtype=torch.float32,
                device=device,
            )

        command_ang_vel_z = [motion["command_ang_vel_z"] for motion in loaded_motions]
        if all(value is None for value in command_ang_vel_z):
            self.command_ang_vel_z = None
        elif any(value is None for value in command_ang_vel_z):
            raise ValueError("All loaded motion files must either include or omit 'command_ang_vel_z'.")
        else:
            command_ang_vel_z_tensor = torch.tensor(
                np.concatenate(command_ang_vel_z, axis=0),
                dtype=torch.float32,
                device=device,
            )
            self.command_ang_vel_z = command_ang_vel_z_tensor.view(command_ang_vel_z_tensor.shape[0], -1)

        arm_ee_pos_w = [motion["arm_ee_pos_w"] for motion in loaded_motions]
        if all(value is None for value in arm_ee_pos_w):
            self.arm_ee_pos_w = None
        elif any(value is None for value in arm_ee_pos_w):
            raise ValueError("All loaded motion files must either include or omit 'arm_ee_pos_w'.")
        else:
            self.arm_ee_pos_w = torch.tensor(
                np.concatenate(arm_ee_pos_w, axis=0),
                dtype=torch.float32,
                device=device,
            )

        arm_ee_quat_w = [motion["arm_ee_quat_w"] for motion in loaded_motions]
        if all(value is None for value in arm_ee_quat_w):
            self.arm_ee_quat_w = None
        elif any(value is None for value in arm_ee_quat_w):
            raise ValueError("All loaded motion files must either include or omit 'arm_ee_quat_w'.")
        else:
            self.arm_ee_quat_w = torch.tensor(
                np.concatenate(arm_ee_quat_w, axis=0),
                dtype=torch.float32,
                device=device,
            )

        reference_foot_contact = [motion["reference_foot_contact"] for motion in loaded_motions]
        if all(value is None for value in reference_foot_contact):
            self.reference_foot_contact = None
        elif any(value is None for value in reference_foot_contact):
            raise ValueError("All loaded motion files must either include or omit 'reference_foot_contact'.")
        else:
            self.reference_foot_contact = torch.tensor(
                np.concatenate(reference_foot_contact, axis=0),
                dtype=torch.bool,
                device=device,
            )

        reference_airborne = [motion["reference_airborne"] for motion in loaded_motions]
        if all(value is None for value in reference_airborne):
            self.reference_airborne = None
        elif any(value is None for value in reference_airborne):
            raise ValueError("All loaded motion files must either include or omit 'reference_airborne'.")
        else:
            self.reference_airborne = torch.tensor(
                np.concatenate(reference_airborne, axis=0).reshape(-1),
                dtype=torch.bool,
                device=device,
            )

        self._body_pos_w = torch.tensor(
            np.concatenate([motion["body_pos_w"] for motion in loaded_motions], axis=0),
            dtype=torch.float32,
            device=device,
        )
        self._body_quat_w = torch.tensor(
            np.concatenate([motion["body_quat_w"] for motion in loaded_motions], axis=0),
            dtype=torch.float32,
            device=device,
        )
        self._body_lin_vel_w = torch.tensor(
            np.concatenate([motion["body_lin_vel_w"] for motion in loaded_motions], axis=0),
            dtype=torch.float32,
            device=device,
        )
        self._body_ang_vel_w = torch.tensor(
            np.concatenate([motion["body_ang_vel_w"] for motion in loaded_motions], axis=0),
            dtype=torch.float32,
            device=device,
        )
        if body_indexes is None:
            self._body_indexes = torch.arange(self._body_pos_w.shape[1], dtype=torch.long, device=device)
        else:
            self._body_indexes = torch.as_tensor(body_indexes, dtype=torch.long, device=device)
        self.time_step_total = self.joint_pos.shape[0]
        motion_lengths = [motion["joint_pos"].shape[0] for motion in loaded_motions]
        motion_starts = np.cumsum([0, *motion_lengths[:-1]], dtype=np.int64)
        motion_ends = motion_starts + np.asarray(motion_lengths, dtype=np.int64) - 1
        self.motion_lengths = torch.tensor(motion_lengths, dtype=torch.long, device=device)
        self.motion_start_steps = torch.tensor(motion_starts, dtype=torch.long, device=device)
        self.motion_end_steps = torch.tensor(motion_ends, dtype=torch.long, device=device)

    @staticmethod
    def _resolve_motion_files(motion_file: str | Sequence[str]) -> list[str]:
        if isinstance(motion_file, str | os.PathLike):
            motion_sources = [motion_file]
        else:
            motion_sources = list(motion_file)

        motion_files: list[str] = []
        for source in motion_sources:
            source_path = Path(source).expanduser()
            if source_path.is_file():
                matches = [source_path]
            elif source_path.is_dir():
                matches = sorted(source_path.rglob("*.npz"))
            else:
                matches = [Path(path) for path in sorted(glob.glob(str(source_path), recursive=True))]
            motion_files.extend(str(path) for path in matches if path.is_file())

        deduped_motion_files = list(dict.fromkeys(motion_files))
        if len(deduped_motion_files) == 0:
            raise FileNotFoundError(f"No motion NPZ files found from: {motion_file}")
        return deduped_motion_files

    @staticmethod
    def _display_names_from_paths(motion_files: Sequence[str]) -> list[str]:
        if len(motion_files) == 1:
            return [Path(motion_files[0]).name]

        common_root = Path(os.path.commonpath(motion_files))
        if common_root.is_file():
            common_root = common_root.parent

        display_names: list[str] = []
        for motion_file in motion_files:
            path = Path(motion_file)
            try:
                display_names.append(str(path.relative_to(common_root)))
            except ValueError:
                display_names.append(path.name)
        return display_names

    @staticmethod
    def _quat_mul_np(lhs: np.ndarray, rhs: np.ndarray) -> np.ndarray:
        """Multiply WXYZ quaternion batches."""
        lw, lx, ly, lz = np.moveaxis(lhs, -1, 0)
        rw, rx, ry, rz = np.moveaxis(rhs, -1, 0)
        return np.stack(
            (
                lw * rw - lx * rx - ly * ry - lz * rz,
                lw * rx + lx * rw + ly * rz - lz * ry,
                lw * ry - lx * rz + ly * rw + lz * rx,
                lw * rz + lx * ry - ly * rx + lz * rw,
            ),
            axis=-1,
        )

    @classmethod
    def _derive_d1_link6_quat_w(
        cls,
        joint_pos: np.ndarray,
        joint_names: list[str] | None,
        body_quat_w: np.ndarray,
        body_names: list[str] | None,
    ) -> np.ndarray | None:
        """Derive the D1 Link6 world orientation from its hardware joint convention."""
        d1_names = [f"arm_{index}_joint" for index in range(1, 7)]
        if joint_names is None or any(name not in joint_names for name in d1_names):
            return None
        base_index = body_names.index("base") if body_names is not None and "base" in body_names else 0
        q_arm = joint_pos[:, [joint_names.index(name) for name in d1_names]]
        axes = np.asarray(
            (
                (0.0, 0.0, -1.0),
                (0.0, 1.0, 0.0),
                (0.0, 1.0, 0.0),
                (1.0, 0.0, 0.0),
                (0.0, 1.0, 0.0),
                (1.0, 0.0, 0.0),
            ),
            dtype=np.float32,
        )
        link_quat = np.zeros((joint_pos.shape[0], 4), dtype=np.float32)
        link_quat[:, 0] = 1.0
        for joint_index, axis in enumerate(axes):
            half_angle = 0.5 * q_arm[:, joint_index]
            joint_quat = np.empty_like(link_quat)
            joint_quat[:, 0] = np.cos(half_angle)
            joint_quat[:, 1:] = np.sin(half_angle)[:, None] * axis[None, :]
            link_quat = cls._quat_mul_np(link_quat, joint_quat)
        result = cls._quat_mul_np(body_quat_w[:, base_index], link_quat)
        return (result / np.linalg.norm(result, axis=-1, keepdims=True).clip(min=1.0e-8)).astype(np.float32)

    @staticmethod
    def _load_motion_file(motion_file: str) -> dict:
        if not os.path.isfile(motion_file):
            raise FileNotFoundError(f"Invalid file path: {motion_file}")

        data = np.load(motion_file)
        joint_pos = np.asarray(data["joint_pos"], dtype=np.float32)
        command_ang_vel_z = None
        if "command_ang_vel_z" in data:
            command_ang_vel_z = data["command_ang_vel_z"].reshape(data["command_ang_vel_z"].shape[0], -1)
        skill = None
        if "skill" in data:
            skill = np.asarray(data["skill"], dtype=np.float32)
            if skill.ndim == 1:
                skill = skill[:, None]
            if skill.shape != (joint_pos.shape[0], 1):
                raise ValueError(
                    f"Motion file '{motion_file}' has skill shape {skill.shape}; expected ({joint_pos.shape[0]}, 1)."
                )
            if not np.isfinite(skill).all():
                raise ValueError(f"Motion file '{motion_file}' has non-finite skill values.")

        gripper_joint_pos = (
            np.asarray(data["gripper_joint_pos"], dtype=np.float32) if "gripper_joint_pos" in data else None
        )
        gripper_joint_names = None
        gripper_joint_vel = None
        if gripper_joint_pos is not None:
            gripper_joint_names = (
                [str(name) for name in data["gripper_joint_names"].tolist()]
                if "gripper_joint_names" in data
                else ["arm_7_1_joint", "arm_7_2_joint"]
            )
            if gripper_joint_pos.shape != (joint_pos.shape[0], len(gripper_joint_names)):
                raise ValueError(
                    f"Motion file '{motion_file}' has gripper_joint_pos shape {gripper_joint_pos.shape}; "
                    f"expected ({joint_pos.shape[0]}, {len(gripper_joint_names)})."
                )
            if "gripper_joint_vel" in data:
                gripper_joint_vel = np.asarray(data["gripper_joint_vel"], dtype=np.float32)
            else:
                gripper_joint_vel = np.gradient(
                    gripper_joint_pos,
                    1.0 / float(np.asarray(data["fps"]).reshape(-1)[0]),
                    axis=0,
                ).astype(np.float32)

        body_names = data["body_names"].tolist() if "body_names" in data else None
        joint_names = data["joint_names"].tolist() if "joint_names" in data else None
        body_quat_w = np.asarray(data["body_quat_w"], dtype=np.float32)
        arm_ee_quat_w = (
            np.asarray(data["arm_ee_quat_w"], dtype=np.float32)
            if "arm_ee_quat_w" in data
            else MotionLoader._derive_d1_link6_quat_w(
                joint_pos,
                joint_names,
                body_quat_w,
                body_names,
            )
        )
        if arm_ee_quat_w is not None:
            expected_quat_shape = (joint_pos.shape[0], 4)
            if arm_ee_quat_w.shape != expected_quat_shape:
                raise ValueError(
                    f"Motion file '{motion_file}' has arm_ee_quat_w shape {arm_ee_quat_w.shape}; "
                    f"expected {expected_quat_shape}."
                )
            quat_norm = np.linalg.norm(arm_ee_quat_w, axis=-1, keepdims=True)
            if not np.isfinite(arm_ee_quat_w).all() or np.any(quat_norm < 1.0e-8):
                raise ValueError(f"Motion file '{motion_file}' has invalid arm_ee_quat_w values.")
            arm_ee_quat_w = (arm_ee_quat_w / quat_norm).astype(np.float32)

        object_attached = None
        if "object_attached" in data:
            object_attached = np.asarray(data["object_attached"], dtype=np.bool_)
            if object_attached.ndim == 1:
                object_attached = object_attached[:, None]
            if object_attached.shape[0] != joint_pos.shape[0]:
                raise ValueError(
                    f"Motion file '{motion_file}' has object_attached shape {object_attached.shape}; "
                    f"expected first dimension {joint_pos.shape[0]}."
                )
            if "object_names" in data and object_attached.shape[1] != len(data["object_names"]):
                raise ValueError(
                    f"Motion file '{motion_file}' has {object_attached.shape[1]} object attachment channels "
                    f"for {len(data['object_names'])} object names."
                )

        return {
            "path": motion_file,
            "fps": np.float32(data["fps"]),
            "body_names": body_names,
            "joint_names": joint_names,
            "joint_pos": joint_pos,
            "joint_vel": np.asarray(data["joint_vel"], dtype=np.float32),
            "skill": skill,
            "gripper_joint_names": gripper_joint_names,
            "gripper_joint_pos": gripper_joint_pos,
            "gripper_joint_vel": gripper_joint_vel,
            "command_lin_vel_xy": (
                np.asarray(data["command_lin_vel_xy"], dtype=np.float32) if "command_lin_vel_xy" in data else None
            ),
            "command_ang_vel_z": (
                np.asarray(command_ang_vel_z, dtype=np.float32) if command_ang_vel_z is not None else None
            ),
            "arm_ee_pos_w": np.asarray(data["arm_ee_pos_w"], dtype=np.float32) if "arm_ee_pos_w" in data else None,
            "arm_ee_quat_w": arm_ee_quat_w,
            "object_names": data["object_names"].tolist() if "object_names" in data else None,
            "object_pos_w": np.asarray(data["object_pos_w"], dtype=np.float32) if "object_pos_w" in data else None,
            "object_quat_w": np.asarray(data["object_quat_w"], dtype=np.float32) if "object_quat_w" in data else None,
            "object_attached": object_attached,
            "reference_foot_contact": (
                np.asarray(data["reference_foot_contact"], dtype=np.bool_) if "reference_foot_contact" in data else None
            ),
            "reference_airborne": (
                np.asarray(data["reference_airborne"], dtype=np.bool_) if "reference_airborne" in data else None
            ),
            "body_pos_w": np.asarray(data["body_pos_w"], dtype=np.float32),
            "body_quat_w": body_quat_w,
            "body_lin_vel_w": np.asarray(data["body_lin_vel_w"], dtype=np.float32),
            "body_ang_vel_w": np.asarray(data["body_ang_vel_w"], dtype=np.float32),
        }

    @staticmethod
    def _validate_motion_metadata(loaded_motions: list[dict]) -> None:
        first_motion = loaded_motions[0]
        for motion in loaded_motions[1:]:
            for metadata_key in ("body_names", "joint_names"):
                if motion[metadata_key] != first_motion[metadata_key]:
                    raise ValueError(
                        f"Motion file '{motion['path']}' has different {metadata_key} than '{first_motion['path']}'."
                    )
            if (motion["arm_ee_pos_w"] is None) != (first_motion["arm_ee_pos_w"] is None):
                raise ValueError(
                    f"Motion file '{motion['path']}' has inconsistent arm_ee_pos_w metadata with "
                    f"'{first_motion['path']}'."
                )
            if (motion["arm_ee_quat_w"] is None) != (first_motion["arm_ee_quat_w"] is None):
                raise ValueError(
                    f"Motion file '{motion['path']}' has inconsistent arm_ee_quat_w metadata with "
                    f"'{first_motion['path']}'."
                )
            if motion["gripper_joint_pos"] is not None and first_motion["gripper_joint_pos"] is not None:
                if motion["gripper_joint_names"] != first_motion["gripper_joint_names"]:
                    raise ValueError(
                        f"Motion file '{motion['path']}' has different gripper_joint_names than "
                        f"'{first_motion['path']}'."
                    )
                if motion["gripper_joint_pos"].shape[1:] != first_motion["gripper_joint_pos"].shape[1:]:
                    raise ValueError(
                        f"Motion file '{motion['path']}' has incompatible gripper_joint_pos shape "
                        f"{motion['gripper_joint_pos'].shape}."
                    )
            if motion["object_pos_w"] is not None and first_motion["object_pos_w"] is not None:
                if motion["object_names"] != first_motion["object_names"]:
                    raise ValueError(
                        f"Motion file '{motion['path']}' has different object_names than '{first_motion['path']}'."
                    )
                if motion["object_pos_w"].shape[1:] != first_motion["object_pos_w"].shape[1:]:
                    raise ValueError(
                        f"Motion file '{motion['path']}' has incompatible object_pos_w shape "
                        f"{motion['object_pos_w'].shape}."
                    )
            for optional_key in ("reference_foot_contact", "reference_airborne"):
                if (motion[optional_key] is None) != (first_motion[optional_key] is None):
                    raise ValueError(
                        f"Motion file '{motion['path']}' has inconsistent {optional_key} metadata with "
                        f"'{first_motion['path']}'."
                    )
            for array_key in (
                "joint_pos",
                "joint_vel",
                "body_pos_w",
                "body_quat_w",
                "body_lin_vel_w",
                "body_ang_vel_w",
            ):
                if motion[array_key].shape[1:] != first_motion[array_key].shape[1:]:
                    raise ValueError(
                        f"Motion file '{motion['path']}' has incompatible {array_key} shape "
                        f"{motion[array_key].shape}; expected trailing shape {first_motion[array_key].shape[1:]}."
                    )
            if (
                motion["arm_ee_pos_w"] is not None
                and motion["arm_ee_pos_w"].shape[1:] != first_motion["arm_ee_pos_w"].shape[1:]
            ):
                raise ValueError(
                    f"Motion file '{motion['path']}' has incompatible arm_ee_pos_w shape "
                    f"{motion['arm_ee_pos_w'].shape}; expected trailing shape "
                    f"{first_motion['arm_ee_pos_w'].shape[1:]}."
                )
            if (
                motion["arm_ee_quat_w"] is not None
                and motion["arm_ee_quat_w"].shape[1:] != first_motion["arm_ee_quat_w"].shape[1:]
            ):
                raise ValueError(
                    f"Motion file '{motion['path']}' has incompatible arm_ee_quat_w shape "
                    f"{motion['arm_ee_quat_w'].shape}; expected trailing shape "
                    f"{first_motion['arm_ee_quat_w'].shape[1:]}."
                )
            if motion["reference_foot_contact"] is not None:
                expected_shape = first_motion["reference_foot_contact"].shape[1:]
                actual_shape = motion["reference_foot_contact"].shape[1:]
                if actual_shape != expected_shape:
                    raise ValueError(
                        f"Motion file '{motion['path']}' has incompatible reference_foot_contact shape "
                        f"{motion['reference_foot_contact'].shape}; expected trailing shape {expected_shape}."
                    )

    def clip_bounds_for(self, time_steps: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        motion_ids = self.motion_ids_for(time_steps)
        starts = self.motion_start_steps[motion_ids].reshape(time_steps.shape)
        ends = self.motion_end_steps[motion_ids].reshape(time_steps.shape)
        return starts, ends

    def motion_ids_for(self, time_steps: torch.Tensor) -> torch.Tensor:
        motion_ids = torch.bucketize(time_steps.reshape(-1), self.motion_end_steps)
        motion_ids = torch.clamp(motion_ids, max=self.motion_end_steps.shape[0] - 1)
        return motion_ids.reshape(time_steps.shape)

    def sample_time_steps_from_motion_ids(self, motion_ids: torch.Tensor, randomize: bool = False) -> torch.Tensor:
        motion_ids = torch.clamp(motion_ids.long(), min=0, max=self.motion_start_steps.shape[0] - 1)
        starts = self.motion_start_steps[motion_ids]
        if not randomize:
            return starts

        lengths = self.motion_lengths[motion_ids]
        local_time_steps = (torch.rand(motion_ids.shape, device=motion_ids.device) * lengths.float()).long()
        return starts + torch.clamp(local_time_steps, max=lengths - 1)

    def sample_time_steps(self, count: int, device: str, sample_motions_uniformly: bool = False) -> torch.Tensor:
        """Sample global frame indices without preserving file order.

        With ``sample_motions_uniformly=True``, each environment first samples a motion id uniformly, then samples a
        random start frame inside that motion. This avoids long clips dominating the tracker purely because they have
        more frames in the concatenated motion table.
        """
        if count <= 0:
            return torch.empty(0, dtype=torch.long, device=device)

        if sample_motions_uniformly and self.motion_start_steps.shape[0] > 1:
            motion_ids = torch.randint(self.motion_start_steps.shape[0], (count,), device=device)
            return self.sample_time_steps_from_motion_ids(motion_ids, randomize=True)

        return sample_uniform(
            0.0,
            float(max(self.time_step_total - 1, 0)),
            (count,),
            device=device,
        ).long()

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w[:, self._body_indexes]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w[:, self._body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w[:, self._body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w[:, self._body_indexes]


class MotionCommand(CommandTerm):
    cfg: MotionCommandCfg

    def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRLEnv):
        # CommandTerm.__init__ may call set_debug_vis(...) immediately.
        # Initialize debug-vis state first to avoid attribute-order races.
        self._debug_vis_foot_indices: list[int] = []
        super().__init__(cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        self.robot_anchor_body_index = self.robot.body_names.index(self.cfg.anchor_body_name)
        self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
        self.body_indexes = torch.tensor(
            self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0], dtype=torch.long, device=self.device
        )
        self.arm_ee_body_indexes = (
            torch.tensor(
                self.robot.find_bodies(self.cfg.arm_ee_body_names, preserve_order=True)[0],
                dtype=torch.long,
                device=self.device,
            )
            if self.cfg.arm_ee_body_names
            else torch.empty(0, dtype=torch.long, device=self.device)
        )
        self.arm_ee_orientation_body_index = (
            self.robot.body_names.index(self.cfg.arm_ee_orientation_body_name)
            if self.cfg.arm_ee_orientation_body_name is not None
            else None
        )
        if self.cfg.joint_names is None:
            base_joint_indexes = torch.arange(self.robot.data.joint_pos.shape[1], dtype=torch.long, device=self.device)
            self.base_motion_joint_names = list(self.robot.joint_names)
        else:
            base_joint_indexes = torch.tensor(
                self.robot.find_joints(self.cfg.joint_names, preserve_order=True)[0],
                dtype=torch.long,
                device=self.device,
            )
            self.base_motion_joint_names = list(self.cfg.joint_names)
        gripper_joint_indexes = (
            torch.tensor(
                self.robot.find_joints(self.cfg.gripper_joint_names, preserve_order=True)[0],
                dtype=torch.long,
                device=self.device,
            )
            if self.cfg.gripper_joint_names
            else torch.empty(0, dtype=torch.long, device=self.device)
        )
        self.joint_indexes = torch.cat([base_joint_indexes, gripper_joint_indexes])
        self.robot_motion_joint_names = [*self.base_motion_joint_names, *self.cfg.gripper_joint_names]
        motion_file = self.cfg.motion_files if len(self.cfg.motion_files) > 0 else self.cfg.motion_file
        self.motion = MotionLoader(motion_file, body_indexes=None, device=self.device)
        self.motion._body_indexes = self._resolve_motion_body_indexes()
        self._motion_joint_indices = self._resolve_motion_joint_indices()
        self._motion_gripper_joint_indices = self._resolve_motion_gripper_joint_indices()
        self._motion_gripper_position_scale = self._resolve_motion_gripper_position_scale()
        self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._time_step_starts = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._time_step_ends = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._motion_frame_accumulator = torch.zeros(self.num_envs, dtype=torch.float32, device=self.device)
        self._update_time_step_bounds(torch.arange(self.num_envs, device=self.device))
        self._is_reset_resample = False
        self._command_lin_vel_xy_offset = torch.zeros(self.num_envs, 2, device=self.device)
        self._command_ang_vel_z_offset = torch.zeros(self.num_envs, 1, device=self.device)
        self.body_pos_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 3, device=self.device)
        self.body_quat_relative_w = torch.zeros(self.num_envs, len(cfg.body_names), 4, device=self.device)
        self.body_quat_relative_w[:, :, 0] = 1.0
        # Command state advances after reward computation. Debug drawing happens after
        # that advance, so retain the completed-step reference instead of rendering the
        # next reference frame one policy step ahead of the simulated robot.
        self._debug_vis_anchor_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        self._debug_vis_anchor_quat_w = torch.zeros(self.num_envs, 4, device=self.device)
        self._debug_vis_anchor_quat_w[:, 0] = 1.0
        self._debug_vis_body_pos_relative_w = self.body_pos_relative_w.clone()
        self._debug_vis_body_quat_relative_w = self.body_quat_relative_w.clone()
        self._debug_vis_command_lin_vel_xy = torch.zeros(self.num_envs, 2, device=self.device)
        self._debug_vis_arm_ee_pos_w = (
            torch.zeros(self.num_envs, 3, device=self.device) if self.motion.arm_ee_pos_w is not None else None
        )
        self._debug_vis_foot_indices = self._resolve_debug_vis_foot_indices()

        self.bin_count = int(self.motion.time_step_total // (1 / (env.cfg.decimation * env.cfg.sim.dt))) + 1
        self._adaptive_sampling_strategy = str(self.cfg.adaptive_sampling_strategy).lower()
        if self._adaptive_sampling_strategy not in {"failure", "tracking_error", "mixed"}:
            raise ValueError(
                "adaptive_sampling_strategy must be one of 'failure', 'tracking_error', or 'mixed', "
                f"got {self.cfg.adaptive_sampling_strategy!r}."
            )
        self.bin_failed_count = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self._current_bin_score = torch.zeros(self.bin_count, dtype=torch.float, device=self.device)
        self.kernel = torch.tensor(
            [self.cfg.adaptive_lambda**i for i in range(max(1, int(self.cfg.adaptive_kernel_size)))],
            device=self.device,
        )
        self.kernel = self.kernel / self.kernel.sum()

        for name in (
            "base_pos",
            "base_orientation",
            "base_lin_vel",
            "base_ang_vel",
            "command_lin_vel_xy",
            "command_lin_vel_z",
            "command_ang_vel_z",
            "command_lin_vel_xy_delta",
            "command_ang_vel_z_delta",
            "body_pos",
            "body_orientation",
            "body_lin_vel",
            "body_ang_vel",
            "joint_pos",
            "joint_vel",
            "arm_ee_pos",
            "arm_ee_orientation",
        ):
            self.metrics[f"error_{name}"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["motion_id"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["motion_local_frame"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_score_mean"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["sampling_score_max"] = torch.zeros(self.num_envs, device=self.device)
        self._sample_command_offsets(torch.arange(self.num_envs, device=self.device))

        # Re-apply visibility once indices are resolved (important for simplified foot-ref mode).
        if self.cfg.debug_vis:
            self._set_debug_vis_impl(True)

    def _resolve_motion_body_indexes(self) -> torch.Tensor:
        if self.motion is not None and getattr(self.motion, "body_names", None) is not None:
            motion_name_to_idx = {name: i for i, name in enumerate(self.motion.body_names)}
            motion_body_indices = []
            for body_name in self.cfg.body_names:
                if body_name not in motion_name_to_idx:
                    raise ValueError(
                        f"Body '{body_name}' is missing in motion file body_names: {self.motion.body_names}"
                    )
                motion_body_indices.append(motion_name_to_idx[body_name])
            return torch.tensor(motion_body_indices, dtype=torch.long, device=self.device)

        # Fallback for files without body_names: assume the first K bodies follow cfg.body_names order.
        num_motion_bodies = int(self.motion._body_pos_w.shape[1])
        if num_motion_bodies < len(self.cfg.body_names):
            raise ValueError(
                f"Motion file has {num_motion_bodies} bodies, fewer than required {len(self.cfg.body_names)}"
            )
        return torch.arange(len(self.cfg.body_names), dtype=torch.long, device=self.device)

    def _resolve_motion_joint_indices(self) -> torch.Tensor | slice:
        motion_joint_names = getattr(self.motion, "joint_names", None)
        robot_joint_names = getattr(self.robot, "joint_names", None)

        # If names are unavailable, fall back to shape-based alignment.
        if motion_joint_names is None or robot_joint_names is None:
            motion_joint_count = int(self.motion.joint_pos.shape[1])
            robot_joint_count = int(self.robot.data.joint_pos.shape[1])
            if motion_joint_count == robot_joint_count:
                return slice(None)
            if motion_joint_count < robot_joint_count:
                raise ValueError(
                    "Motion file joint count is smaller than robot joint count and no joint names are available for "
                    f"mapping ({motion_joint_count} < {robot_joint_count})."
                )
            return torch.arange(robot_joint_count, dtype=torch.long, device=self.device)

        motion_name_to_idx = {name: i for i, name in enumerate(motion_joint_names)}
        joint_indices = []
        for joint_name in self.base_motion_joint_names:
            if joint_name not in motion_name_to_idx:
                raise ValueError(f"Robot joint '{joint_name}' is missing in motion file joints: {motion_joint_names}")
            joint_indices.append(motion_name_to_idx[joint_name])
        return torch.tensor(joint_indices, dtype=torch.long, device=self.device)

    def _resolve_motion_gripper_joint_indices(self) -> torch.Tensor:
        if len(self.cfg.gripper_joint_names) == 0:
            return torch.empty(0, dtype=torch.long, device=self.device)
        if self.motion.gripper_joint_pos is None or self.motion.gripper_joint_names is None:
            raise ValueError(
                "Motion command requests gripper joints, but the selected motion does not contain gripper_joint_pos."
            )
        name_to_idx = {name: i for i, name in enumerate(self.motion.gripper_joint_names)}
        missing = [name for name in self.cfg.gripper_joint_names if name not in name_to_idx]
        if missing:
            raise ValueError(
                f"Motion gripper channels are missing requested joints {missing}: {self.motion.gripper_joint_names}"
            )
        return torch.tensor(
            [name_to_idx[name] for name in self.cfg.gripper_joint_names],
            dtype=torch.long,
            device=self.device,
        )

    def _resolve_motion_gripper_position_scale(self) -> torch.Tensor:
        """Map the source gripper phase to an explicit physical closed target."""
        if self._motion_gripper_joint_indices.numel() == 0:
            return torch.empty(0, dtype=torch.float32, device=self.device)

        scale = torch.full(
            (self._motion_gripper_joint_indices.numel(),),
            float(self.cfg.gripper_position_scale),
            dtype=torch.float32,
            device=self.device,
        )
        if self.cfg.gripper_closed_position is None:
            if self.cfg.gripper_binary:
                raise ValueError("gripper_binary requires gripper_closed_position.")
            return scale

        source_pos = self.motion.gripper_joint_pos[:, self._motion_gripper_joint_indices]
        source_closed = torch.amax(torch.abs(source_pos), dim=0)
        if torch.any(source_closed <= 0.0):
            raise ValueError(
                "gripper_closed_position requires every requested source gripper channel "
                "to contain a non-zero close phase."
            )
        return float(self.cfg.gripper_closed_position) / source_closed

    def _resolve_debug_vis_foot_indices(self) -> list[int]:
        if len(self.cfg.debug_vis_foot_body_names) > 0:
            body_index_map = {name: idx for idx, name in enumerate(self.cfg.body_names)}
            resolved = [body_index_map[name] for name in self.cfg.debug_vis_foot_body_names if name in body_index_map]
            if len(resolved) > 0:
                return resolved

        auto_detected = [idx for idx, name in enumerate(self.cfg.body_names) if "foot" in name.lower()]
        if len(auto_detected) > 0:
            return auto_detected

        return [idx for idx, name in enumerate(self.cfg.body_names) if name != self.cfg.anchor_body_name]

    @property
    def command(self) -> torch.Tensor:  # TODO Consider again if this is the best observation
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    @property
    def time_step_starts(self) -> torch.Tensor:
        return self._time_step_starts

    @property
    def time_step_ends(self) -> torch.Tensor:
        return self._time_step_ends

    @property
    def current_motion_ids(self) -> torch.Tensor:
        return self.motion.motion_ids_for(self.time_steps)

    def current_motion_names(self, env_ids: Sequence[int] | torch.Tensor | None = None) -> list[str]:
        if env_ids is None:
            motion_ids = self.current_motion_ids
        else:
            motion_ids = self.current_motion_ids[env_ids]
        motion_ids_list = motion_ids.detach().cpu().reshape(-1).tolist()
        return [self.motion.motion_names[int(motion_id)] for motion_id in motion_ids_list]

    def motion_status(self, env_id: int = 0) -> str:
        motion_id = int(self.current_motion_ids[env_id].detach().cpu().item())
        motion_count = len(self.motion.motion_names)
        motion_name = self.motion.motion_names[motion_id]
        local_frame = int((self.time_steps[env_id] - self._time_step_starts[env_id]).detach().cpu().item())
        total_frames = int(self.motion.motion_lengths[motion_id].detach().cpu().item())
        return f"{motion_id + 1}/{motion_count} {motion_name} frame={local_frame}/{max(total_frames - 1, 0)}"

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        self._is_reset_resample = True
        try:
            return super().reset(env_ids)
        finally:
            self._is_reset_resample = False

    def switch_motion(
        self,
        delta: int = 1,
        env_ids: Sequence[int] | torch.Tensor | None = None,
        motion_id: int | None = None,
        randomize: bool = False,
    ) -> None:
        if len(self.motion.motion_names) <= 1:
            return

        if env_ids is None:
            env_ids = torch.arange(self.num_envs, device=self.device)
        else:
            env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
        if env_ids.numel() == 0:
            return

        current_motion_ids = self.current_motion_ids[env_ids]
        if motion_id is None:
            target_motion_ids = torch.remainder(current_motion_ids + int(delta), len(self.motion.motion_names))
        else:
            target_motion_ids = torch.full_like(current_motion_ids, int(motion_id) % len(self.motion.motion_names))

        self.time_steps[env_ids] = self.motion.sample_time_steps_from_motion_ids(target_motion_ids, randomize=randomize)
        self._motion_frame_accumulator[env_ids] = 0.0
        self._update_time_step_bounds(env_ids)
        self._sample_command_offsets(env_ids)
        self._refresh_relative_motion_state()
        if self.cfg.debug_vis:
            self._cache_completed_step_debug_reference()

    @property
    def joint_pos(self) -> torch.Tensor:
        return self.motion_joint_pos_at(self.time_steps)

    @property
    def joint_vel(self) -> torch.Tensor:
        return self.motion_joint_vel_at(self.time_steps)

    def motion_joint_pos_at(self, time_steps: torch.Tensor) -> torch.Tensor:
        joint_pos = self.motion.joint_pos[time_steps][..., self._motion_joint_indices]
        if self._motion_gripper_joint_indices.numel() > 0:
            gripper_joint_pos = self.motion.gripper_joint_pos[time_steps][..., self._motion_gripper_joint_indices]
            if self.cfg.gripper_binary:
                closed = torch.full_like(gripper_joint_pos, float(self.cfg.gripper_closed_position))
                opened = torch.full_like(gripper_joint_pos, float(self.cfg.gripper_open_position))
                gripper_joint_pos = torch.where(
                    gripper_joint_pos > float(self.cfg.gripper_binary_threshold),
                    closed,
                    opened,
                )
            else:
                gripper_joint_pos = gripper_joint_pos * self._motion_gripper_position_scale
            joint_pos = torch.cat([joint_pos, gripper_joint_pos], dim=-1)
        return joint_pos

    def motion_joint_vel_at(self, time_steps: torch.Tensor) -> torch.Tensor:
        joint_vel = self.motion.joint_vel[time_steps][..., self._motion_joint_indices]
        if self._motion_gripper_joint_indices.numel() > 0:
            gripper_joint_vel = self.motion.gripper_joint_vel[time_steps][..., self._motion_gripper_joint_indices]
            if self.cfg.gripper_binary:
                gripper_joint_vel = torch.zeros_like(gripper_joint_vel)
            else:
                gripper_joint_vel = gripper_joint_vel * self._motion_gripper_position_scale
            joint_vel = torch.cat([joint_vel, gripper_joint_vel], dim=-1)
        return joint_vel

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps]

    @property
    def arm_ee_pos_w(self) -> torch.Tensor | None:
        if self.motion.arm_ee_pos_w is None:
            return None
        return self.motion.arm_ee_pos_w[self.time_steps] + self._env.scene.env_origins

    @property
    def arm_ee_quat_w(self) -> torch.Tensor | None:
        if self.motion.arm_ee_quat_w is None:
            return None
        return self.motion.arm_ee_quat_w[self.time_steps]

    @property
    def object_pos_w(self) -> torch.Tensor | None:
        if self.motion.object_pos_w is None:
            return None
        return self.motion.object_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]

    @property
    def object_quat_w(self) -> torch.Tensor | None:
        if self.motion.object_quat_w is None:
            return None
        return self.motion.object_quat_w[self.time_steps]

    @property
    def object_attached(self) -> torch.Tensor | None:
        if self.motion.object_attached is None:
            return None
        return self.motion.object_attached[self.time_steps]

    @property
    def object_size_scale(self) -> torch.Tensor | None:
        """Return the retained isotropic object scale for every environment."""
        if self.cfg.object_size_scale_attr is None:
            return None
        return getattr(self._env, self.cfg.object_size_scale_attr, None)

    def object_size_center_offset_w(
        self,
        object_quat_w: torch.Tensor,
        env_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Return the center offset that fixes the ground and optional grasp-facing -X planes."""
        size_scale = self.object_size_scale
        if size_scale is None or self.cfg.object_nominal_size is None:
            return None
        if env_ids is not None:
            size_scale = size_scale[env_ids]
        scale_shape = (size_scale.shape[0],) + (1,) * (object_quat_w.ndim - 2)
        broadcast_scale = size_scale.reshape(scale_shape)
        local_grasp_face_offset = torch.zeros_like(object_quat_w[..., :3])
        if self.cfg.object_size_preserve_grasp_face:
            local_grasp_face_offset[..., 0] = (
                0.5 * float(self.cfg.object_nominal_size[0]) * (broadcast_scale - 1.0)
            )
        center_offset_w = quat_apply(
            object_quat_w.reshape(-1, 4),
            local_grasp_face_offset.reshape(-1, 3),
        ).reshape_as(local_grasp_face_offset)
        center_offset_w[..., 2] += (
            0.5 * float(self.cfg.object_nominal_size[2]) * (broadcast_scale - 1.0)
        )
        return center_offset_w

    @property
    def object_target_pos_w(self) -> torch.Tensor | None:
        """Return reference object positions aligned to the simulated robot anchor."""
        ref_object_pos_w = self.object_pos_w
        if ref_object_pos_w is None:
            return None
        delta_pos_w = torch.cat([self.robot_anchor_pos_w[:, :2], self.anchor_pos_w[:, 2:3]], dim=-1)
        delta_ori_w = yaw_quat(quat_mul(self.robot_anchor_quat_w, quat_inv(self.anchor_quat_w)))
        relative_pos_w = ref_object_pos_w - self.anchor_pos_w[:, None, :]
        aligned_relative_pos_w = quat_apply(
            delta_ori_w[:, None, :].expand(-1, relative_pos_w.shape[1], -1).reshape(-1, 4),
            relative_pos_w.reshape(-1, 3),
        ).reshape_as(relative_pos_w)
        target_pos_w = delta_pos_w[:, None, :] + aligned_relative_pos_w
        target_quat_w = self.object_target_quat_w
        if target_quat_w is not None:
            size_center_offset_w = self.object_size_center_offset_w(target_quat_w)
            if size_center_offset_w is not None:
                target_pos_w += size_center_offset_w
        return target_pos_w

    @property
    def object_target_quat_w(self) -> torch.Tensor | None:
        """Return reference object orientations aligned to the simulated robot yaw."""
        ref_object_quat_w = self.object_quat_w
        if ref_object_quat_w is None:
            return None
        delta_ori_w = yaw_quat(quat_mul(self.robot_anchor_quat_w, quat_inv(self.anchor_quat_w)))
        return quat_mul(
            delta_ori_w[:, None, :].expand_as(ref_object_quat_w).reshape(-1, 4),
            ref_object_quat_w.reshape(-1, 4),
        ).reshape_as(ref_object_quat_w)

    @property
    def arm_ee_target_pos_w(self) -> torch.Tensor | None:
        """Return the reference endpoint after aligning it to the simulated robot yaw."""
        ref_ee_pos_w = self.arm_ee_pos_w
        if ref_ee_pos_w is None:
            return None
        delta_pos_w = torch.cat([self.robot_anchor_pos_w[:, :2], self.anchor_pos_w[:, 2:3]], dim=-1)
        delta_ori_w = yaw_quat(quat_mul(self.robot_anchor_quat_w, quat_inv(self.anchor_quat_w)))
        return delta_pos_w + quat_apply(delta_ori_w, ref_ee_pos_w - self.anchor_pos_w)

    @property
    def arm_ee_target_quat_w(self) -> torch.Tensor | None:
        """Return the Link6 reference orientation after aligning it to the simulated robot yaw."""
        ref_ee_quat_w = self.arm_ee_quat_w
        if ref_ee_quat_w is None:
            return None
        delta_ori_w = yaw_quat(quat_mul(self.robot_anchor_quat_w, quat_inv(self.anchor_quat_w)))
        return quat_mul(delta_ori_w, ref_ee_quat_w)

    @property
    def robot_arm_ee_pos_w(self) -> torch.Tensor | None:
        """Return the mean position of the configured physical endpoint bodies."""
        if self.arm_ee_body_indexes.numel() == 0:
            return None
        return self.robot.data.body_pos_w[:, self.arm_ee_body_indexes, :].mean(dim=1)

    @property
    def skill(self) -> torch.Tensor:
        """Return the frame-aligned skill scalar from the active motion clip."""
        return self.motion.skill[self.time_steps]

    @property
    def reference_foot_contact(self) -> torch.Tensor | None:
        if self.motion.reference_foot_contact is None:
            return None
        return self.motion.reference_foot_contact[self.time_steps]

    @property
    def reference_airborne(self) -> torch.Tensor | None:
        if self.motion.reference_airborne is None:
            return None
        return self.motion.reference_airborne[self.time_steps]

    @property
    def anchor_pos_w(self) -> torch.Tensor:
        return self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index] + self._env.scene.env_origins

    @property
    def anchor_quat_w(self) -> torch.Tensor:
        return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_lin_vel_w(self) -> torch.Tensor:
        return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

    @property
    def anchor_ang_vel_w(self) -> torch.Tensor:
        return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

    def _reference_command_lin_vel_xy(self) -> torch.Tensor:
        if self.motion.command_lin_vel_xy is None:
            if self.cfg.require_command_channels:
                raise RuntimeError(
                    "Motion file is missing 'command_lin_vel_xy' but this task requires inbuilt command channels."
                )
            # Backward-compatible fallback for legacy motion files.
            return self.anchor_lin_vel_w[:, :2]
        return self.motion.command_lin_vel_xy[self.time_steps]

    def _reference_command_ang_vel_z(self) -> torch.Tensor:
        if self.motion.command_ang_vel_z is None:
            if self.cfg.require_command_channels:
                raise RuntimeError(
                    "Motion file is missing 'command_ang_vel_z' but this task requires inbuilt command channels."
                )
            # Backward-compatible fallback for legacy motion files.
            return self.anchor_ang_vel_w[:, 2:3]
        return self.motion.command_ang_vel_z[self.time_steps]

    @staticmethod
    def _preserve_nonzero_reference_sign(command: torch.Tensor, reference: torch.Tensor) -> torch.Tensor:
        positive_reference = reference > 0.0
        negative_reference = reference < 0.0
        command = torch.where(positive_reference & (command < 0.0), torch.zeros_like(command), command)
        return torch.where(negative_reference & (command > 0.0), torch.zeros_like(command), command)

    @property
    def command_lin_vel_xy(self) -> torch.Tensor:
        command_lin_vel_xy = self._reference_command_lin_vel_xy()
        if self.cfg.use_reference_command_channels_directly:
            return command_lin_vel_xy
        reference_command_lin_vel_xy = command_lin_vel_xy
        command_lin_vel_xy = self._preserve_nonzero_reference_sign(
            command_lin_vel_xy + self._command_lin_vel_xy_offset,
            reference_command_lin_vel_xy,
        )
        command_lin_vel_x = torch.clamp(
            command_lin_vel_xy[:, 0:1],
            min=float(self.cfg.command_lin_vel_x_clip[0]),
            max=float(self.cfg.command_lin_vel_x_clip[1]),
        )
        command_lin_vel_y = torch.clamp(
            command_lin_vel_xy[:, 1:2],
            min=float(self.cfg.command_lin_vel_y_clip[0]),
            max=float(self.cfg.command_lin_vel_y_clip[1]),
        )
        return torch.cat([command_lin_vel_x, command_lin_vel_y], dim=-1)

    @property
    def command_lin_vel_z(self) -> torch.Tensor:
        return self.anchor_lin_vel_w[:, 2:3]

    @property
    def command_ang_vel_z(self) -> torch.Tensor:
        command_ang_vel_z = self._reference_command_ang_vel_z()
        if self.cfg.use_reference_command_channels_directly:
            return command_ang_vel_z
        command_ang_vel_z = self._preserve_nonzero_reference_sign(
            command_ang_vel_z + self._command_ang_vel_z_offset,
            command_ang_vel_z,
        )
        return torch.clamp(
            command_ang_vel_z,
            min=float(self.cfg.command_ang_vel_z_clip[0]),
            max=float(self.cfg.command_ang_vel_z_clip[1]),
        )

    @property
    def robot_command_lin_vel_xy(self) -> torch.Tensor:
        if self.cfg.command_velocity_frame == "world":
            return self.robot.data.root_lin_vel_w[:, :2]
        return self.robot.data.root_lin_vel_b[:, :2]

    @property
    def robot_command_lin_vel_z(self) -> torch.Tensor:
        return self.robot.data.root_lin_vel_w[:, 2:3]

    @property
    def robot_command_ang_vel_z(self) -> torch.Tensor:
        if self.cfg.command_velocity_frame == "world":
            return self.robot.data.root_ang_vel_w[:, 2:3]
        return self.robot.data.root_ang_vel_b[:, 2:3]

    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self.robot.data.joint_pos[:, self.joint_indexes]

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self.robot.data.joint_vel[:, self.joint_indexes]

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.body_indexes]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.body_indexes]

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.body_indexes]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.body_indexes]

    @property
    def robot_anchor_pos_w(self) -> torch.Tensor:
        return self.robot.data.body_pos_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_quat_w(self) -> torch.Tensor:
        return self.robot.data.body_quat_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_lin_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_lin_vel_w[:, self.robot_anchor_body_index]

    @property
    def robot_anchor_ang_vel_w(self) -> torch.Tensor:
        return self.robot.data.body_ang_vel_w[:, self.robot_anchor_body_index]

    def _tracking_error_terms(self) -> dict[str, torch.Tensor]:
        command_lin_vel_xy = self.command_lin_vel_xy
        command_lin_vel_z = self.command_lin_vel_z
        command_ang_vel_z = self.command_ang_vel_z
        reference_command_lin_vel_xy = self._reference_command_lin_vel_xy()
        reference_command_ang_vel_z = self._reference_command_ang_vel_z()
        terms = {
            "base_pos": torch.norm(self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1),
            "base_orientation": quat_error_magnitude(self.anchor_quat_w, self.robot_anchor_quat_w),
            "base_lin_vel": torch.norm(self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1),
            "base_ang_vel": torch.norm(self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1),
            "command_lin_vel_xy": torch.norm(command_lin_vel_xy - self.robot_command_lin_vel_xy, dim=-1),
            "command_lin_vel_z": torch.abs(command_lin_vel_z[:, 0] - self.robot_command_lin_vel_z[:, 0]),
            "command_ang_vel_z": torch.abs(command_ang_vel_z[:, 0] - self.robot_command_ang_vel_z[:, 0]),
            "command_lin_vel_xy_delta": torch.norm(command_lin_vel_xy - reference_command_lin_vel_xy, dim=-1),
            "command_ang_vel_z_delta": torch.abs(command_ang_vel_z[:, 0] - reference_command_ang_vel_z[:, 0]),
            "body_pos": torch.norm(self.body_pos_relative_w - self.robot_body_pos_w, dim=-1).mean(dim=-1),
            "body_orientation": quat_error_magnitude(self.body_quat_relative_w, self.robot_body_quat_w).mean(dim=-1),
            "body_lin_vel": torch.norm(self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1).mean(dim=-1),
            "body_ang_vel": torch.norm(self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1).mean(dim=-1),
            "joint_pos": torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1),
            "joint_vel": torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1),
        }
        arm_ee_target_pos_w = self.arm_ee_target_pos_w
        robot_arm_ee_pos_w = self.robot_arm_ee_pos_w
        if arm_ee_target_pos_w is not None and robot_arm_ee_pos_w is not None:
            terms["arm_ee_pos"] = torch.norm(arm_ee_target_pos_w - robot_arm_ee_pos_w, dim=-1)
        arm_ee_target_quat_w = self.arm_ee_target_quat_w
        if arm_ee_target_quat_w is not None and self.arm_ee_orientation_body_index is not None:
            robot_arm_ee_quat_w = self.robot.data.body_quat_w[:, self.arm_ee_orientation_body_index]
            terms["arm_ee_orientation"] = quat_error_magnitude(
                arm_ee_target_quat_w,
                robot_arm_ee_quat_w,
            )
        return terms

    def _update_metrics(self):
        for name, error in self._tracking_error_terms().items():
            self.metrics[f"error_{name}"] = error
        self.metrics["motion_id"] = self.current_motion_ids.float()
        self.metrics["motion_local_frame"] = (self.time_steps - self._time_step_starts).float()

    def _current_bin_indices(self) -> torch.Tensor:
        return torch.clamp(
            (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1), 0, self.bin_count - 1
        )

    def _tracking_error_sampling_score(self) -> torch.Tensor:
        score = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
        terms = self._tracking_error_terms()
        aliases = {
            "anchor_pos": "base_pos",
            "anchor_rot": "base_orientation",
            "anchor_lin_vel": "base_lin_vel",
            "anchor_ang_vel": "base_ang_vel",
            "body_rot": "body_orientation",
        }
        for name, weight in self.cfg.adaptive_tracking_error_weights.items():
            weight = float(weight)
            if weight == 0.0:
                continue
            term_name = aliases.get(name, name)
            if term_name not in terms:
                raise ValueError(
                    f"Unknown adaptive tracking error term '{name}'. Available terms: {sorted(terms.keys())}."
                )
            score = score + weight * terms[term_name].detach()
        score = score * float(self.cfg.adaptive_tracking_error_scale)
        return torch.nan_to_num(torch.clamp(score, min=0.0), nan=0.0, posinf=1.0e6, neginf=0.0)

    def _uniform_bin_probabilities(self) -> torch.Tensor:
        if not self.cfg.sample_motions_uniformly or len(self.motion.motion_names) <= 1:
            return torch.full((self.bin_count,), 1.0 / float(self.bin_count), device=self.device)

        bin_time_steps = (
            (torch.arange(self.bin_count, device=self.device, dtype=torch.float) + 0.5)
            / float(self.bin_count)
            * float(max(self.motion.time_step_total - 1, 0))
        ).long()
        motion_ids = self.motion.motion_ids_for(bin_time_steps)
        motion_bin_counts = torch.bincount(motion_ids, minlength=len(self.motion.motion_names)).float()
        motion_bin_counts = torch.clamp(motion_bin_counts, min=1.0)
        probabilities = 1.0 / (float(len(self.motion.motion_names)) * motion_bin_counts[motion_ids])
        return probabilities / probabilities.sum()

    def _sample_from_adaptive_bins(self) -> torch.Tensor:
        sampling_score = torch.clamp(self.bin_failed_count, min=0.0)
        sampling_probabilities = torch.nn.functional.pad(
            sampling_score.unsqueeze(0).unsqueeze(0),
            (0, self.kernel.numel() - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)
        sampling_probabilities = torch.nan_to_num(
            torch.clamp(sampling_probabilities, min=0.0), nan=0.0, posinf=1.0e6, neginf=0.0
        )

        uniform_probabilities = self._uniform_bin_probabilities()
        score_sum = sampling_probabilities.sum()
        uniform_ratio = float(max(0.0, min(1.0, self.cfg.adaptive_uniform_ratio)))
        if not bool(torch.isfinite(score_sum).item()) or float(score_sum.item()) <= 0.0:
            sampling_probabilities = uniform_probabilities
        else:
            focused_probabilities = sampling_probabilities / score_sum
            sampling_probabilities = (
                1.0 - uniform_ratio
            ) * focused_probabilities + uniform_ratio * uniform_probabilities
            sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

        return sampling_probabilities

    def _accumulate_adaptive_bin_score(self, score_bins: torch.Tensor, score: torch.Tensor) -> None:
        """Safely accumulate adaptive-sampling scores into fixed-width bins.

        Keep the histogram update on CPU. CUDA-side bincount/masked indexing can fail with a device assert if upstream
        tensors are ever malformed; CPU validation gives us bounded behavior and clearer errors for this low-rate path.
        """
        score_bins_cpu = score_bins.detach().reshape(-1).to(device="cpu", dtype=torch.long)
        score_cpu = score.detach().reshape(-1).to(device="cpu", dtype=torch.float32)
        if score_bins_cpu.numel() == 0:
            return
        if score_cpu.shape[0] != score_bins_cpu.shape[0]:
            raise RuntimeError(
                f"Adaptive sampling score/bin shape mismatch: scores={tuple(score_cpu.shape)}, "
                f"bins={tuple(score_bins_cpu.shape)}."
            )

        score_bins_np = score_bins_cpu.numpy()
        score_np = score_cpu.numpy()
        valid = np.isfinite(score_np) & (score_bins_np >= 0) & (score_bins_np < self.bin_count)
        if not np.any(valid):
            return

        score_bins_np = np.clip(score_bins_np[valid], 0, self.bin_count - 1)
        score_np = np.nan_to_num(np.clip(score_np[valid], 0.0, None), nan=0.0, posinf=1.0e6, neginf=0.0)
        bin_score_np = np.bincount(score_bins_np, weights=score_np, minlength=self.bin_count)
        bin_score = torch.as_tensor(bin_score_np, dtype=self._current_bin_score.dtype, device=self.device)
        self._current_bin_score += bin_score[: self.bin_count]

    def _adaptive_sampling(self, env_ids: Sequence[int]):
        current_bin_index = self._current_bin_indices()
        if self._adaptive_sampling_strategy in {"failure", "mixed"}:
            episode_failed = self._env.termination_manager.terminated[env_ids]
            if torch.any(episode_failed):
                fail_bins = current_bin_index[env_ids][episode_failed]
                fail_score = torch.full(
                    (fail_bins.shape[0],),
                    float(self.cfg.adaptive_failure_weight),
                    dtype=torch.float,
                    device=self.device,
                )
                self._accumulate_adaptive_bin_score(fail_bins, fail_score)

        if self._adaptive_sampling_strategy in {"tracking_error", "mixed"}:
            tracking_score = self._tracking_error_sampling_score()[env_ids]
            score_bins = current_bin_index[env_ids]
            self._accumulate_adaptive_bin_score(score_bins, tracking_score)

        sampling_probabilities = self._sample_from_adaptive_bins()

        sampled_bins = torch.multinomial(sampling_probabilities, len(env_ids), replacement=True)

        self.time_steps[env_ids] = (
            (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
            / self.bin_count
            * (self.motion.time_step_total - 1)
        ).long()

        # Metrics
        H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
        H_norm = H / math.log(self.bin_count) if self.bin_count > 1 else torch.tensor(0.0, device=self.device)
        pmax, imax = sampling_probabilities.max(dim=0)
        self.metrics["sampling_entropy"][:] = H_norm
        self.metrics["sampling_top1_prob"][:] = pmax
        self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count
        self.metrics["sampling_score_mean"][:] = self.bin_failed_count.mean()
        self.metrics["sampling_score_max"][:] = self.bin_failed_count.max()

    def _uniform_sampling(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return

        if self.cfg.start_at_motion_beginning_on_reset and self._is_reset_resample:
            motion_ids = torch.randint(
                len(self.motion.motion_names),
                (len(env_ids),),
                device=self.device,
            )
            self.time_steps[env_ids] = self.motion.sample_time_steps_from_motion_ids(
                motion_ids,
                randomize=False,
            )
            random_probability = float(self.cfg.reset_random_frame_probability)
            attached_probability = float(self.cfg.reset_attached_frame_probability)
            if random_probability < 0.0 or attached_probability < 0.0:
                raise ValueError("Reset-frame probabilities must be non-negative.")
            if random_probability + attached_probability > 1.0:
                raise ValueError(
                    "reset_random_frame_probability + reset_attached_frame_probability "
                    "must not exceed 1."
                )

            reset_choices = torch.rand(len(env_ids), device=self.device)
            attached_mask = reset_choices < attached_probability
            random_mask = (reset_choices >= attached_probability) & (
                reset_choices < attached_probability + random_probability
            )
            env_ids_tensor = torch.as_tensor(env_ids, dtype=torch.long, device=self.device)
            if torch.any(random_mask):
                selected_env_ids = env_ids_tensor[random_mask]
                self.time_steps[selected_env_ids] = self.motion.sample_time_steps(
                    selected_env_ids.numel(),
                    device=self.device,
                    sample_motions_uniformly=self.cfg.sample_motions_uniformly,
                )
            if torch.any(attached_mask):
                if self.motion.object_attached is None:
                    raise ValueError(
                        "reset_attached_frame_probability requires object_attached in the motion data."
                    )
                attached_frames = torch.where(
                    self.motion.object_attached[:, self.cfg.reset_attached_object_index]
                )[0]
                if attached_frames.numel() == 0:
                    raise ValueError("The selected motion library has no attached-object frames to sample.")
                selected_env_ids = env_ids_tensor[attached_mask]
                sampled_indices = torch.randint(
                    attached_frames.numel(),
                    (selected_env_ids.numel(),),
                    device=self.device,
                )
                self.time_steps[selected_env_ids] = attached_frames[sampled_indices]
        else:
            self.time_steps[env_ids] = self.motion.sample_time_steps(
                len(env_ids),
                device=self.device,
                sample_motions_uniformly=self.cfg.sample_motions_uniformly,
            )

        if self.bin_count > 1:
            self.metrics["sampling_entropy"][:] = 1.0
            self.metrics["sampling_top1_prob"][:] = 1.0 / float(self.bin_count)
        else:
            self.metrics["sampling_entropy"][:] = 0.0
            self.metrics["sampling_top1_prob"][:] = 1.0
        self.metrics["sampling_top1_bin"][:] = 0.0
        self.metrics["sampling_score_mean"][:] = 0.0
        self.metrics["sampling_score_max"][:] = 0.0

    def _sample_command_offsets(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        self._command_lin_vel_xy_offset[env_ids, 0:1] = sample_uniform(
            *self.cfg.command_lin_vel_x_offset_range,
            (len(env_ids), 1),
            device=self.device,
        )
        self._command_lin_vel_xy_offset[env_ids, 1:2] = sample_uniform(
            *self.cfg.command_lin_vel_y_offset_range,
            (len(env_ids), 1),
            device=self.device,
        )
        self._command_ang_vel_z_offset[env_ids] = sample_uniform(
            *self.cfg.command_ang_vel_z_offset_range,
            (len(env_ids), 1),
            device=self.device,
        )

    def _update_time_step_bounds(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        starts, ends = self.motion.clip_bounds_for(self.time_steps[env_ids])
        self._time_step_starts[env_ids] = starts
        self._time_step_ends[env_ids] = ends

    def _resample_command(self, env_ids: Sequence[int]):
        if len(env_ids) == 0:
            return
        if self.cfg.use_adaptive_sampling:
            self._adaptive_sampling(env_ids)
        else:
            self._uniform_sampling(env_ids)
        self._update_time_step_bounds(env_ids)
        self._motion_frame_accumulator[env_ids] = 0.0
        self._sample_command_offsets(env_ids)

        # Only initialize state on environment reset. Clip-end command resampling must not teleport the robot.
        if not self.cfg.reference_state_init or not self._is_reset_resample:
            return

        root_pos = self.body_pos_w[:, 0].clone()
        root_ori = self.body_quat_w[:, 0].clone()
        root_lin_vel = self.body_lin_vel_w[:, 0].clone()
        root_ang_vel = self.body_ang_vel_w[:, 0].clone()

        range_list = [self.cfg.pose_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_pos[env_ids] += rand_samples[:, 0:3]
        orientations_delta = quat_from_euler_xyz(rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5])
        root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
        # Pose perturbations rotate the complete reference state. Leaving the
        # recorded world velocity unrotated creates an inconsistent reset when
        # yaw randomization is enabled.
        root_lin_vel[env_ids] = quat_apply(orientations_delta, root_lin_vel[env_ids])
        root_ang_vel[env_ids] = quat_apply(orientations_delta, root_ang_vel[env_ids])
        range_list = [self.cfg.velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
        ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device)
        root_lin_vel[env_ids] += rand_samples[:, :3]
        root_ang_vel[env_ids] += rand_samples[:, 3:]

        joint_pos = self.joint_pos.clone()
        joint_vel = self.joint_vel.clone()
        if (
            self.cfg.gripper_binary
            and self.cfg.reference_state_use_recorded_gripper
            and self._motion_gripper_joint_indices.numel() > 0
        ):
            # Binary actions command the servo endpoint, but teleporting the
            # physical jaw to that endpoint around an object would create deep
            # penetration. Initialize from the recorded contact position; the
            # binary close target then maintains force through the normal drive.
            gripper_start = len(self.base_motion_joint_names)
            joint_pos[:, gripper_start:] = self.motion.gripper_joint_pos[self.time_steps][
                ..., self._motion_gripper_joint_indices
            ]
            joint_vel[:, gripper_start:] = self.motion.gripper_joint_vel[self.time_steps][
                ..., self._motion_gripper_joint_indices
            ]

        joint_pos += sample_uniform(*self.cfg.joint_position_range, joint_pos.shape, joint_pos.device)
        soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids][:, self.joint_indexes]
        joint_pos[env_ids] = torch.clip(
            joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
        )
        full_joint_pos = self.robot.data.default_joint_pos[env_ids].clone()
        full_joint_vel = self.robot.data.default_joint_vel[env_ids].clone()
        full_joint_pos[:, self.joint_indexes] = joint_pos[env_ids]
        full_joint_vel[:, self.joint_indexes] = joint_vel[env_ids]
        self.robot.write_joint_state_to_sim(full_joint_pos, full_joint_vel, env_ids=env_ids)
        self.robot.write_root_state_to_sim(
            torch.cat([root_pos[env_ids], root_ori[env_ids], root_lin_vel[env_ids], root_ang_vel[env_ids]], dim=-1),
            env_ids=env_ids,
        )
        # Pass the just-authored anchor pose explicitly. Reading body buffers
        # immediately after a teleport can otherwise use a pre-reset pose (or
        # force a full articulation refresh) before the next simulation forward.
        self._initialize_reference_object_state(
            env_ids,
            robot_anchor_pos_w=root_pos[env_ids],
            robot_anchor_quat_w=root_ori[env_ids],
        )

    def _initialize_reference_object_state(
        self,
        env_ids: Sequence[int],
        *,
        robot_anchor_pos_w: torch.Tensor | None = None,
        robot_anchor_quat_w: torch.Tensor | None = None,
    ) -> None:
        """Place a manipulation object at its phase-consistent reference state."""
        if not self.cfg.reference_object_state_init:
            return
        if self.cfg.reference_object_asset_name is None:
            raise ValueError("reference_object_state_init requires reference_object_asset_name.")
        if self.motion.object_pos_w is None or self.motion.object_quat_w is None:
            raise ValueError("reference_object_state_init requires object pose channels in the motion data.")

        resolved_env_ids = torch.as_tensor(env_ids, dtype=torch.long, device=self.device).flatten()
        if resolved_env_ids.numel() == 0:
            return
        object_asset: RigidObject = self._env.scene[self.cfg.reference_object_asset_name]
        object_index = int(self.cfg.reference_object_state_index)
        selected_steps = self.time_steps[resolved_env_ids]
        starts, ends = self.motion.clip_bounds_for(selected_steps)
        prev_steps = torch.maximum(selected_steps - 1, starts)
        next_steps = torch.minimum(selected_steps + 1, ends)

        reference_anchor_pos_w = (
            self.motion.body_pos_w[selected_steps, self.motion_anchor_body_index]
            + self._env.scene.env_origins[resolved_env_ids]
        )
        reference_anchor_quat_w = self.motion.body_quat_w[selected_steps, self.motion_anchor_body_index]
        if robot_anchor_pos_w is None:
            robot_anchor_pos_w = self.robot_anchor_pos_w[resolved_env_ids]
        if robot_anchor_quat_w is None:
            robot_anchor_quat_w = self.robot_anchor_quat_w[resolved_env_ids]
        if (
            robot_anchor_pos_w.shape[0] != resolved_env_ids.numel()
            or robot_anchor_quat_w.shape[0] != resolved_env_ids.numel()
        ):
            raise ValueError("Explicit reference object anchor poses must match env_ids.")

        delta_pos_w = torch.cat(
            [robot_anchor_pos_w[:, :2], reference_anchor_pos_w[:, 2:3]],
            dim=-1,
        )
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w, quat_inv(reference_anchor_quat_w)))
        reference_object_pos_w = (
            self.motion.object_pos_w[selected_steps, object_index]
            + self._env.scene.env_origins[resolved_env_ids]
        )
        object_pos_w = delta_pos_w + quat_apply(
            delta_ori_w,
            reference_object_pos_w - reference_anchor_pos_w,
        )
        object_quat_w = quat_mul(
            delta_ori_w,
            self.motion.object_quat_w[selected_steps, object_index],
        )
        size_center_offset_w = self.object_size_center_offset_w(
            object_quat_w.unsqueeze(1),
            env_ids=resolved_env_ids,
        )
        if size_center_offset_w is not None:
            object_pos_w += size_center_offset_w[:, 0]

        prev_pos_w = self.motion.object_pos_w[prev_steps, object_index]
        next_pos_w = self.motion.object_pos_w[next_steps, object_index]
        motion_ids = self.motion.motion_ids_for(selected_steps)
        fps = self.motion.fps_values[motion_ids].to(device=self.device, dtype=next_pos_w.dtype)
        dt = (next_steps - prev_steps).to(dtype=next_pos_w.dtype) / fps.clamp_min(1.0e-6)
        object_lin_vel_w = torch.where(
            dt[:, None] > 0.0,
            (next_pos_w - prev_pos_w) / dt.clamp_min(1.0e-6)[:, None],
            torch.zeros_like(next_pos_w),
        )
        object_lin_vel_w = quat_apply(delta_ori_w, object_lin_vel_w)
        prev_quat_w = self.motion.object_quat_w[prev_steps, object_index]
        next_quat_w = self.motion.object_quat_w[next_steps, object_index]
        reference_ang_vel_w = axis_angle_from_quat(quat_mul(next_quat_w, quat_inv(prev_quat_w)))
        object_ang_vel_w = torch.where(
            dt[:, None] > 0.0,
            reference_ang_vel_w / dt.clamp_min(1.0e-6)[:, None],
            torch.zeros_like(reference_ang_vel_w),
        )
        object_ang_vel_w = quat_apply(delta_ori_w, object_ang_vel_w)

        object_asset.write_root_state_to_sim(
            torch.cat(
                [
                    object_pos_w,
                    object_quat_w,
                    object_lin_vel_w,
                    object_ang_vel_w,
                ],
                dim=-1,
            ),
            env_ids=resolved_env_ids,
        )

    def _update_command(self):
        if self.cfg.debug_vis:
            # Refresh against the post-physics robot pose while time_steps still points
            # at the reference used to produce and score the completed action.
            self._refresh_relative_motion_state()
            self._cache_completed_step_debug_reference()

        motion_ids = self.current_motion_ids
        frame_rate = self.motion.fps_values[motion_ids].to(dtype=self._motion_frame_accumulator.dtype)
        self._motion_frame_accumulator += frame_rate * float(self._env.step_dt)
        frame_steps = torch.floor(self._motion_frame_accumulator).long()
        if torch.any(frame_steps > 0):
            self._motion_frame_accumulator -= frame_steps.to(dtype=self._motion_frame_accumulator.dtype)
            self.time_steps += frame_steps
        env_ids = torch.where(self.time_steps > self._time_step_ends)[0]
        if self.cfg.terminate_episode_at_motion_end:
            self.time_steps[env_ids] = self._time_step_ends[env_ids]
        else:
            self._resample_command(env_ids)

        self._refresh_relative_motion_state()

        if self.cfg.use_adaptive_sampling:
            self.bin_failed_count = (
                self.cfg.adaptive_alpha * self._current_bin_score
                + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
            )
        else:
            self.bin_failed_count.zero_()
        self._current_bin_score.zero_()

    def _refresh_relative_motion_state(self):
        num_bodies = len(self.cfg.body_names)
        anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].expand(-1, num_bodies, -1)
        anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].expand(-1, num_bodies, -1)
        robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].expand(-1, num_bodies, -1)
        robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].expand(-1, num_bodies, -1)

        delta_pos_w = torch.cat([robot_anchor_pos_w_repeat[..., :2], anchor_pos_w_repeat[..., 2:3]], dim=-1)
        delta_ori_w = yaw_quat(quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat)))

        self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
        self.body_pos_relative_w = delta_pos_w + quat_apply(delta_ori_w, self.body_pos_w - anchor_pos_w_repeat)

    def _cache_completed_step_debug_reference(self):
        """Cache the reference frame aligned with the latest completed physics step."""
        self._debug_vis_anchor_pos_w.copy_(self.anchor_pos_w)
        self._debug_vis_anchor_quat_w.copy_(self.anchor_quat_w)
        self._debug_vis_body_pos_relative_w.copy_(self.body_pos_relative_w)
        self._debug_vis_body_quat_relative_w.copy_(self.body_quat_relative_w)
        self._debug_vis_command_lin_vel_xy.copy_(self.command_lin_vel_xy)

        if self._debug_vis_arm_ee_pos_w is not None:
            arm_ee_target_pos_w = self.arm_ee_target_pos_w
            if arm_ee_target_pos_w is not None:
                self._debug_vis_arm_ee_pos_w.copy_(arm_ee_target_pos_w)

    def _set_debug_vis_impl(self, debug_vis: bool):
        foot_indices = getattr(self, "_debug_vis_foot_indices", [])
        use_pose_frames = self.cfg.debug_vis_show_pose_frames and not self.cfg.debug_vis_simplified_foot_reference
        use_velocity = self.cfg.debug_vis_show_velocity
        use_simple_foot_ref = self.cfg.debug_vis_simplified_foot_reference and len(foot_indices) > 0
        motion = getattr(self, "motion", None)
        use_arm_ee_ref = (
            self.cfg.debug_vis_show_arm_ee_reference
            and motion is not None
            and getattr(motion, "arm_ee_pos_w", None) is not None
        )

        if debug_vis:
            if use_pose_frames:
                if not hasattr(self, "current_anchor_visualizer"):
                    self.current_anchor_visualizer = VisualizationMarkers(
                        self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/current/anchor")
                    )
                    self.goal_anchor_visualizer = VisualizationMarkers(
                        self.cfg.anchor_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/anchor")
                    )

                    self.current_body_visualizers = []
                    self.goal_body_visualizers = []
                    for name in self.cfg.body_names:
                        self.current_body_visualizers.append(
                            VisualizationMarkers(
                                self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/current/" + name)
                            )
                        )
                        self.goal_body_visualizers.append(
                            VisualizationMarkers(
                                self.cfg.body_visualizer_cfg.replace(prim_path="/Visuals/Command/goal/" + name)
                            )
                        )

                self.current_anchor_visualizer.set_visibility(True)
                self.goal_anchor_visualizer.set_visibility(True)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(True)
                    self.goal_body_visualizers[i].set_visibility(True)
            elif hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)

            if use_velocity:
                if not hasattr(self, "goal_vel_visualizer"):
                    self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                    self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
                self.goal_vel_visualizer.set_visibility(True)
                self.current_vel_visualizer.set_visibility(True)
            elif hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)

            if use_simple_foot_ref:
                if not hasattr(self, "goal_foot_ref_visualizer"):
                    self.goal_foot_ref_visualizer = VisualizationMarkers(self.cfg.foot_ref_visualizer_cfg)
                self.goal_foot_ref_visualizer.set_visibility(True)
            elif hasattr(self, "goal_foot_ref_visualizer"):
                self.goal_foot_ref_visualizer.set_visibility(False)

            if use_arm_ee_ref:
                if not hasattr(self, "goal_arm_ee_ref_visualizer"):
                    self.goal_arm_ee_ref_visualizer = VisualizationMarkers(self.cfg.arm_ee_ref_visualizer_cfg)
                    self.current_arm_ee_visualizer = VisualizationMarkers(self.cfg.arm_ee_current_visualizer_cfg)
                self.goal_arm_ee_ref_visualizer.set_visibility(True)
                self.current_arm_ee_visualizer.set_visibility(True)
            elif hasattr(self, "goal_arm_ee_ref_visualizer"):
                self.goal_arm_ee_ref_visualizer.set_visibility(False)
                self.current_arm_ee_visualizer.set_visibility(False)
        else:
            if hasattr(self, "current_anchor_visualizer"):
                self.current_anchor_visualizer.set_visibility(False)
                self.goal_anchor_visualizer.set_visibility(False)
                for i in range(len(self.cfg.body_names)):
                    self.current_body_visualizers[i].set_visibility(False)
                    self.goal_body_visualizers[i].set_visibility(False)
            if hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer.set_visibility(False)
                self.current_vel_visualizer.set_visibility(False)
            if hasattr(self, "goal_foot_ref_visualizer"):
                self.goal_foot_ref_visualizer.set_visibility(False)
            if hasattr(self, "goal_arm_ee_ref_visualizer"):
                self.goal_arm_ee_ref_visualizer.set_visibility(False)
                self.current_arm_ee_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        if self.cfg.debug_vis_show_pose_frames and not self.cfg.debug_vis_simplified_foot_reference:
            self.current_anchor_visualizer.visualize(self.robot_anchor_pos_w, self.robot_anchor_quat_w)
            self.goal_anchor_visualizer.visualize(self._debug_vis_anchor_pos_w, self._debug_vis_anchor_quat_w)

            for i in range(len(self.cfg.body_names)):
                self.current_body_visualizers[i].visualize(self.robot_body_pos_w[:, i], self.robot_body_quat_w[:, i])
                self.goal_body_visualizers[i].visualize(
                    self._debug_vis_body_pos_relative_w[:, i],
                    self._debug_vis_body_quat_relative_w[:, i],
                )

        if self.cfg.debug_vis_show_velocity:
            base_pos_w = self.robot.data.root_pos_w.clone()
            base_pos_w[:, 2] += 0.5
            vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(
                self._debug_vis_command_lin_vel_xy, self.cfg.command_velocity_frame
            )
            vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(self.robot.data.root_lin_vel_b[:, :2])
            self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
            self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

        if self.cfg.debug_vis_simplified_foot_reference and len(self._debug_vis_foot_indices) > 0:
            foot_ref_pos_w = self._debug_vis_body_pos_relative_w[:, self._debug_vis_foot_indices, :].reshape(-1, 3)
            self.goal_foot_ref_visualizer.visualize(foot_ref_pos_w)

        if self.cfg.debug_vis_show_arm_ee_reference and self._debug_vis_arm_ee_pos_w is not None:
            self.goal_arm_ee_ref_visualizer.visualize(self._debug_vis_arm_ee_pos_w)
            robot_arm_ee_pos_w = self.robot_arm_ee_pos_w
            if robot_arm_ee_pos_w is not None:
                self.current_arm_ee_visualizer.visualize(robot_arm_ee_pos_w)

    def _resolve_xy_velocity_to_arrow(
        self, xy_velocity: torch.Tensor, velocity_frame: str = "base"
    ) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        speed = torch.linalg.norm(xy_velocity, dim=1)
        arrow_scale[:, 0] *= speed * 3.0
        # Scaling only the arrow length to zero leaves its Y/Z cross-section visible as
        # a large colored disc. Hide the complete marker when the velocity is negligible.
        arrow_scale = torch.where((speed > 1.0e-3).unsqueeze(-1), arrow_scale, torch.zeros_like(arrow_scale))
        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = quat_from_euler_xyz(zeros, zeros, heading_angle)
        if velocity_frame == "base":
            arrow_quat = quat_mul(self.robot.data.root_quat_w, arrow_quat)
        return arrow_scale, arrow_quat


@configclass
class BinaryJointPositionCommandCfg(CommandTermCfg):
    """Configuration for binary joint-position commands."""

    class_type: type = BinaryJointPositionCommand

    asset_name: str = MISSING
    joint_names: list[str] = MISSING
    low_position: tuple[float, ...] = MISSING
    high_position: tuple[float, ...] = MISSING
    high_prob: float = 0.5


@configclass
class MotionCommandCfg(CommandTermCfg):
    """Configuration for the motion command."""

    class_type: type = MotionCommand

    asset_name: str = MISSING

    motion_file: str = MISSING
    motion_files: tuple[str, ...] = ()
    anchor_body_name: str = MISSING
    body_names: list[str] = MISSING
    joint_names: list[str] | None = None
    gripper_joint_names: tuple[str, ...] = ()
    gripper_position_scale: float = 1.0
    gripper_closed_position: float | None = None
    gripper_binary: bool = False
    gripper_binary_threshold: float = 0.0
    gripper_open_position: float = 0.0
    object_size_scale_attr: str | None = None
    object_nominal_size: tuple[float, float, float] | None = None
    object_size_preserve_grasp_face: bool = True
    arm_ee_body_names: tuple[str, ...] = ()
    arm_ee_orientation_body_name: str | None = None

    pose_range: dict[str, tuple[float, float]] = {}
    velocity_range: dict[str, tuple[float, float]] = {}

    joint_position_range: tuple[float, float] = (-0.52, 0.52)
    command_lin_vel_x_offset_range: tuple[float, float] = (0.0, 0.0)
    command_lin_vel_y_offset_range: tuple[float, float] = (0.0, 0.0)
    command_lin_vel_x_clip: tuple[float, float] = (-1.0e9, 1.0e9)
    command_lin_vel_y_clip: tuple[float, float] = (-1.0e9, 1.0e9)
    command_ang_vel_z_offset_range: tuple[float, float] = (0.0, 0.0)
    command_ang_vel_z_clip: tuple[float, float] = (-1.0e9, 1.0e9)

    use_adaptive_sampling: bool = False
    adaptive_sampling_strategy: str = "failure"
    adaptive_kernel_size: int = 1
    adaptive_lambda: float = 0.8
    adaptive_uniform_ratio: float = 0.1
    adaptive_alpha: float = 0.001
    adaptive_failure_weight: float = 1.0
    adaptive_tracking_error_weights: dict[str, float] = {
        "joint_pos": 1.0,
        "base_pos": 0.5,
        "base_orientation": 0.5,
        "command_lin_vel_xy": 0.25,
        "command_ang_vel_z": 0.25,
    }
    adaptive_tracking_error_scale: float = 1.0
    reference_state_init: bool = False
    reference_object_state_init: bool = False
    reference_object_asset_name: str | None = None
    reference_object_state_index: int = 0
    reference_state_use_recorded_gripper: bool = True
    require_command_channels: bool = False
    use_reference_command_channels_directly: bool = False
    command_velocity_frame: str = "base"
    sample_motions_uniformly: bool = False
    start_at_motion_beginning_on_reset: bool = False
    reset_random_frame_probability: float = 0.0
    reset_attached_frame_probability: float = 0.0
    reset_attached_object_index: int = 0
    terminate_episode_at_motion_end: bool = False

    anchor_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    anchor_visualizer_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)

    body_visualizer_cfg: VisualizationMarkersCfg = FRAME_MARKER_CFG.replace(prim_path="/Visuals/Command/pose")
    body_visualizer_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)

    debug_vis_show_pose_frames: bool = True
    debug_vis_show_velocity: bool = True
    debug_vis_simplified_foot_reference: bool = False
    debug_vis_foot_body_names: tuple[str, ...] = ()
    debug_vis_show_arm_ee_reference: bool = False

    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.1, 0.1)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.1, 0.1)

    foot_ref_visualizer_cfg: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(
        prim_path="/Visuals/Command/foot_reference"
    )
    foot_ref_visualizer_cfg.markers["sphere"].radius = 0.03

    arm_ee_ref_visualizer_cfg: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(
        prim_path="/Visuals/Command/arm_ee_reference"
    )
    # Keep both markers small enough that their centers show the actual tracking
    # error. The previous 4 cm reference sphere obscured the gripper and made a
    # correctly positioned marker appear offset from its tracked point.
    arm_ee_ref_visualizer_cfg.markers["sphere"].radius = 0.015
    arm_ee_ref_visualizer_cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(1.0, 0.2, 0.0)
    )
    arm_ee_current_visualizer_cfg: VisualizationMarkersCfg = SPHERE_MARKER_CFG.replace(
        prim_path="/Visuals/Command/arm_ee_current"
    )
    arm_ee_current_visualizer_cfg.markers["sphere"].radius = 0.012
    arm_ee_current_visualizer_cfg.markers["sphere"].visual_material = sim_utils.PreviewSurfaceCfg(
        diffuse_color=(0.0, 0.4, 1.0)
    )
