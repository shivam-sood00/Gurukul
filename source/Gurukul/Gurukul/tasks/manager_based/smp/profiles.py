# SPDX-License-Identifier: Apache-2.0

"""Robot-specific contracts for score-matching motion priors.

An SMP prior is tied to a morphology: changing a joint order or a key body
changes the meaning of every feature channel.  The profiles in this module are
therefore deliberately explicit and are embedded verbatim in every checkpoint
and prepared dataset.

This module has no Isaac Lab dependency so dataset preparation and prior
pretraining can run in a lightweight Python environment.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType
from typing import Any

SMP_SCHEMA_VERSION = 2
"""Version of the canonical motion-feature layout."""

DEFAULT_WINDOW_SIZE = 10
"""Number of 50 Hz frames scored by the motion prior."""

DEFAULT_CONTROL_FPS = 50.0
"""Required policy and motion-data frequency for the initial SMP release."""

SMP_FEATURE_CONVENTIONS = MappingProxyType(
    {
        "history_order": "oldest_to_newest",
        "quaternion_order": "wxyz",
        "rotation_6d_columns": "x_z",
        "up_axis": "z",
        "canonical_anchor": "newest_root_xy_and_yaw",
        "root_height": "absolute",
        "key_body_frame": "root_relative_newest_heading",
        "root_pose_source_frame": "link_actor",
        "root_velocity_source_frame": "link_actor",
    }
)
"""Self-describing conventions embedded in every profile artifact."""


@dataclass(frozen=True)
class SmpRobotProfile:
    """Immutable feature contract for one robot morphology.

    Attributes:
        name: Stable, checkpoint-facing profile identifier.
        aliases: Accepted user-facing names. Aliases are never saved in place
            of :attr:`name`.
        root_body_name: Named motion root used for the canonical root state. A
            simulator asset may retain a coincident, fixed dummy articulation
            root, which the runtime validates explicitly.
        joint_names: Joint-position channels in their exact policy order.
        key_body_names: World-space bodies represented relative to the root.
        control_fps: Frequency expected by both datasets and environments.
    """

    name: str
    aliases: tuple[str, ...]
    root_body_name: str
    joint_names: tuple[str, ...]
    key_body_names: tuple[str, ...]
    control_fps: float = DEFAULT_CONTROL_FPS

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("An SMP profile requires a non-empty name.")
        if not self.root_body_name:
            raise ValueError(f"SMP profile {self.name!r} has no root body.")
        if not self.joint_names:
            raise ValueError(f"SMP profile {self.name!r} has no joints.")
        if not self.key_body_names:
            raise ValueError(f"SMP profile {self.name!r} has no key bodies.")
        if len(set(self.joint_names)) != len(self.joint_names):
            raise ValueError(f"SMP profile {self.name!r} contains duplicate joint names.")
        if len(set(self.key_body_names)) != len(self.key_body_names):
            raise ValueError(f"SMP profile {self.name!r} contains duplicate key-body names.")
        if self.root_body_name in self.key_body_names:
            raise ValueError(f"SMP profile {self.name!r} must not repeat its root in key_body_names.")
        if self.control_fps <= 0.0:
            raise ValueError(f"SMP profile {self.name!r} has an invalid control frequency.")

    @property
    def num_joints(self) -> int:
        """Number of scalar joint-position features."""

        return len(self.joint_names)

    @property
    def num_key_bodies(self) -> int:
        """Number of key bodies represented by 3-D relative positions."""

        return len(self.key_body_names)

    @property
    def feature_dim(self) -> int:
        """Per-frame feature dimension for this profile.

        The schema is ``root_pos(3) + root_rot_6d(6) + joint_pos(J) +
        key_body_pos(3K) + root_lin_vel(3) + root_ang_vel(3)``.
        """

        return 3 + 6 + self.num_joints + 3 * self.num_key_bodies + 3 + 3

    def to_metadata(self) -> dict[str, Any]:
        """Return the strict, serialization-safe morphology identity."""

        return {
            "schema_version": SMP_SCHEMA_VERSION,
            "name": self.name,
            "root_body_name": self.root_body_name,
            "joint_names": list(self.joint_names),
            "key_body_names": list(self.key_body_names),
            "feature_dim": self.feature_dim,
            "control_fps": float(self.control_fps),
            **SMP_FEATURE_CONVENTIONS,
        }


# G1 order matches UNITREE_G1_29DOF_CFG.joint_sdk_names in assets/unitree.py.
G1_PROFILE = SmpRobotProfile(
    name="g1",
    aliases=("unitree_g1", "unitree-g1", "unitree g1"),
    root_body_name="pelvis",
    joint_names=(
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
    ),
    key_body_names=(
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_rubber_hand",
        "right_rubber_hand",
    ),
)


# PM01 order matches PM01_24DOF_POLICY_JOINT_NAMES in
# assets/engineai_pm01_official.py (and intentionally differs from URDF order).
PM01_PROFILE = SmpRobotProfile(
    name="pm01",
    aliases=("engineai_pm01", "engineai-pm01", "engineai pm01"),
    root_body_name="LINK_BASE",
    joint_names=(
        "J00_HIP_PITCH_L",
        "J06_HIP_PITCH_R",
        "J12_WAIST_YAW",
        "J01_HIP_ROLL_L",
        "J07_HIP_ROLL_R",
        "J13_SHOULDER_PITCH_L",
        "J18_SHOULDER_PITCH_R",
        "J23_HEAD_YAW",
        "J02_HIP_YAW_L",
        "J08_HIP_YAW_R",
        "J14_SHOULDER_ROLL_L",
        "J19_SHOULDER_ROLL_R",
        "J03_KNEE_PITCH_L",
        "J09_KNEE_PITCH_R",
        "J15_SHOULDER_YAW_L",
        "J20_SHOULDER_YAW_R",
        "J04_ANKLE_PITCH_L",
        "J10_ANKLE_PITCH_R",
        "J16_ELBOW_PITCH_L",
        "J21_ELBOW_PITCH_R",
        "J05_ANKLE_ROLL_L",
        "J11_ANKLE_ROLL_R",
        "J17_ELBOW_YAW_L",
        "J22_ELBOW_YAW_R",
    ),
    key_body_names=(
        "LINK_ELBOW_YAW_L",
        "LINK_ELBOW_YAW_R",
        "LINK_ANKLE_ROLL_L",
        "LINK_ANKLE_ROLL_R",
        "LINK_HEAD_YAW",
    ),
)


# Go2 order matches GO2_ACTION_JOINT_NAMES in go2_apex/constants.py.
GO2_PROFILE = SmpRobotProfile(
    name="go2",
    aliases=("unitree_go2", "unitree-go2", "unitree go2"),
    root_body_name="base",
    joint_names=(
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
    ),
    key_body_names=("FL_foot", "FR_foot", "RL_foot", "RR_foot"),
)


_PROFILES = (G1_PROFILE, PM01_PROFILE, GO2_PROFILE)


def _normalize_profile_name(name: str) -> str:
    return " ".join(name.strip().lower().replace("-", " ").replace("_", " ").split())


_PROFILE_LOOKUP: dict[str, SmpRobotProfile] = {}
for _profile in _PROFILES:
    for _profile_name in (_profile.name, *_profile.aliases):
        _lookup_key = _normalize_profile_name(_profile_name)
        if _lookup_key in _PROFILE_LOOKUP and _PROFILE_LOOKUP[_lookup_key] is not _profile:
            raise RuntimeError(f"Duplicate normalized SMP profile alias: {_profile_name!r}")
        _PROFILE_LOOKUP[_lookup_key] = _profile


def profiles() -> tuple[SmpRobotProfile, ...]:
    """Return all supported SMP morphology profiles in stable order."""

    return _PROFILES


def get_profile(name: str | SmpRobotProfile) -> SmpRobotProfile:
    """Resolve a canonical SMP profile from a name or pass one through."""

    if isinstance(name, SmpRobotProfile):
        return name
    if not isinstance(name, str):
        raise TypeError(f"SMP profile must be a name or SmpRobotProfile, got {type(name).__name__}.")
    try:
        return _PROFILE_LOOKUP[_normalize_profile_name(name)]
    except KeyError as exc:
        supported = ", ".join(profile.name for profile in _PROFILES)
        raise ValueError(f"Unknown SMP robot profile {name!r}; choose one of: {supported}.") from exc


def validate_profile_metadata(metadata: Mapping[str, Any], expected: str | SmpRobotProfile) -> SmpRobotProfile:
    """Validate serialized morphology metadata against a local profile.

    Validation is intentionally exact. A checkpoint trained with a different
    joint order can produce plausible tensor shapes while assigning actions and
    body features to the wrong channels, so shape-only compatibility is unsafe.
    """

    profile = get_profile(expected)
    if not isinstance(metadata, Mapping):
        raise TypeError("SMP profile metadata must be a mapping.")

    canonical = profile.to_metadata()
    required = set(canonical)
    missing = sorted(required.difference(metadata))
    unknown = sorted(set(metadata).difference(required))
    if missing or unknown:
        details = []
        if missing:
            details.append(f"missing keys {missing}")
        if unknown:
            details.append(f"unknown keys {unknown}")
        raise ValueError(f"Invalid SMP profile metadata: {', '.join(details)}.")

    for key, expected_value in canonical.items():
        actual_value = metadata[key]
        if key in ("joint_names", "key_body_names"):
            if not isinstance(actual_value, (list, tuple)):
                raise TypeError(f"SMP profile metadata field {key!r} must be a sequence.")
            actual_value = list(actual_value)
        elif key == "control_fps":
            try:
                actual_value = float(actual_value)
            except (TypeError, ValueError) as exc:
                raise TypeError("SMP profile control_fps must be numeric.") from exc

        if actual_value != expected_value:
            raise ValueError(
                f"SMP profile mismatch for {key!r}: checkpoint has {actual_value!r}, "
                f"but profile {profile.name!r} requires {expected_value!r}."
            )
    return profile


def profile_from_metadata(metadata: Mapping[str, Any]) -> SmpRobotProfile:
    """Resolve and strictly validate the registered profile in metadata."""

    if not isinstance(metadata, Mapping) or not isinstance(metadata.get("name"), str):
        raise ValueError("SMP profile metadata is missing its string 'name' field.")
    profile = get_profile(metadata["name"])
    return validate_profile_metadata(metadata, profile)


__all__ = [
    "DEFAULT_CONTROL_FPS",
    "DEFAULT_WINDOW_SIZE",
    "G1_PROFILE",
    "GO2_PROFILE",
    "PM01_PROFILE",
    "SMP_SCHEMA_VERSION",
    "SMP_FEATURE_CONVENTIONS",
    "SmpRobotProfile",
    "get_profile",
    "profile_from_metadata",
    "profiles",
    "validate_profile_metadata",
]
