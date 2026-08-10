#!/usr/bin/env python3

"""Visualize the Go2+D1 WBC arm workspace in Viser.

The default model is the repository's Go2+D1 URDF, so the mount, D1 links, and
joint limits match the Isaac asset while avoiding Isaac runtime dependencies.
"""

from __future__ import annotations

import argparse
import csv
import math
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import trimesh


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_URDF = (
    REPO_ROOT
    / "source/Gurukul/data/Robots/unitree/go2_with_d1/urdf/go2_d1_vis.urdf"
)
DEFAULT_WORKSPACE = ((0.10, 0.56), (-0.40, 0.40), (0.18, 0.65))
DEFAULT_WORKSPACE_SLIDER_LIMITS = ((0.05, 0.75), (-0.55, 0.55), (0.05, 0.75))
DEFAULT_BODY_BOX = ((-0.30, 0.34), (-0.20, 0.20), (-0.02, 0.30))
DEFAULT_BODY_CLEARANCE = 0.07
DEFAULT_KEEP_OUT = tuple(
    (low - DEFAULT_BODY_CLEARANCE, high + DEFAULT_BODY_CLEARANCE) for low, high in DEFAULT_BODY_BOX
)
DEFAULT_WORKSPACE_ORIGIN = np.asarray((0.0, 0.0, 0.08), dtype=np.float64)
DEFAULT_REACH_RANGE = (0.12, 0.58)
ARM_JOINTS = tuple(f"arm_{index}_joint" for index in range(1, 7))
ARM_LINK_CHAIN = (
    "d1_base_link",
    "Empty_Link1",
    "Empty_Link2",
    "Empty_Link3",
    "Empty_Link4",
    "Empty_Link5",
    "Empty_Link6",
)
EE_LINK = "Empty_Link6"
CARRY_Q = np.asarray((0.0, -1.15, 1.35, 0.0, -0.30, 0.0), dtype=np.float64)
WORKSPACE_READY_Q = np.asarray((0.0, 0.58, 0.02, 0.0, -0.38, 0.0), dtype=np.float64)
FOLDED_Q = np.asarray((0.0, -1.553, 1.553, 0.0, 0.0, 0.0), dtype=np.float64)
DEFAULT_GO2_STANCE = {
    "FL_hip_joint": 0.10,
    "FR_hip_joint": -0.10,
    "RL_hip_joint": 0.10,
    "RR_hip_joint": -0.10,
    "FL_thigh_joint": 0.80,
    "FR_thigh_joint": 0.80,
    "RL_thigh_joint": 1.00,
    "RR_thigh_joint": 1.00,
    "FL_calf_joint": -1.50,
    "FR_calf_joint": -1.50,
    "RL_calf_joint": -1.50,
    "RR_calf_joint": -1.50,
}


@dataclass(frozen=True)
class Visual:
    mesh_path: Path
    link_from_visual: np.ndarray
    color: tuple[int, int, int]


@dataclass(frozen=True)
class Joint:
    name: str
    parent: str
    child: str
    joint_type: str
    parent_from_joint: np.ndarray
    axis: np.ndarray
    lower: float | None = None
    upper: float | None = None


@dataclass
class RobotModel:
    root: str
    visuals_by_link: dict[str, list[Visual]]
    joints_by_parent: dict[str, list[Joint]]
    joints_by_name: dict[str, Joint]
    limits_by_joint: dict[str, tuple[float, float]]
    mesh_cache: dict[tuple[Path, int], tuple[np.ndarray, np.ndarray]] = field(default_factory=dict)

    def fk(self, joint_pos: dict[str, float]) -> dict[str, np.ndarray]:
        transforms: dict[str, np.ndarray] = {self.root: np.eye(4, dtype=np.float64)}
        stack = [self.root]
        while stack:
            parent = stack.pop()
            parent_from_root = transforms[parent]
            for joint in self.joints_by_parent.get(parent, ()):
                transforms[joint.child] = parent_from_root @ joint_transform(joint, joint_pos.get(joint.name, 0.0))
                stack.append(joint.child)
        return transforms

    def joint_limits(self, joint_names: tuple[str, ...]) -> np.ndarray:
        limits = []
        for name in joint_names:
            lower, upper = self.limits_by_joint.get(name, (-math.pi, math.pi))
            limits.append((lower, upper))
        return np.asarray(limits, dtype=np.float64)


def parse_vec(text: str | None, default: tuple[float, ...]) -> np.ndarray:
    if not text:
        return np.asarray(default, dtype=np.float64)
    return np.asarray([float(v) for v in text.split()], dtype=np.float64)


def rot_x(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(((1.0, 0.0, 0.0), (0.0, c, -s), (0.0, s, c)), dtype=np.float64)


def rot_y(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(((c, 0.0, s), (0.0, 1.0, 0.0), (-s, 0.0, c)), dtype=np.float64)


def rot_z(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.asarray(((c, -s, 0.0), (s, c, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64)


def rpy_to_matrix(rpy: np.ndarray) -> np.ndarray:
    return rot_z(float(rpy[2])) @ rot_y(float(rpy[1])) @ rot_x(float(rpy[0]))


def transform_from_xyz_rpy(xyz: np.ndarray, rpy: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rpy_to_matrix(rpy)
    transform[:3, 3] = xyz
    return transform


def axis_angle_to_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    norm = float(np.linalg.norm(axis))
    if norm < 1.0e-9:
        return np.eye(3, dtype=np.float64)
    x, y, z = axis / norm
    c, s = math.cos(angle), math.sin(angle)
    one_c = 1.0 - c
    return np.asarray(
        (
            (c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s),
            (y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s),
            (z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c),
        ),
        dtype=np.float64,
    )


def joint_transform(joint: Joint, q: float) -> np.ndarray:
    transform = np.array(joint.parent_from_joint, copy=True)
    if joint.joint_type in ("revolute", "continuous"):
        motion = np.eye(4, dtype=np.float64)
        motion[:3, :3] = axis_angle_to_matrix(joint.axis, q)
        transform = transform @ motion
    elif joint.joint_type == "prismatic":
        motion = np.eye(4, dtype=np.float64)
        motion[:3, 3] = joint.axis * q
        transform = transform @ motion
    return transform


def matrix_to_wxyz(matrix: np.ndarray) -> np.ndarray:
    rot = matrix[:3, :3]
    trace = float(np.trace(rot))
    if trace > 0.0:
        s = math.sqrt(trace + 1.0) * 2.0
        w = 0.25 * s
        x = (rot[2, 1] - rot[1, 2]) / s
        y = (rot[0, 2] - rot[2, 0]) / s
        z = (rot[1, 0] - rot[0, 1]) / s
    else:
        idx = int(np.argmax(np.diag(rot)))
        if idx == 0:
            s = math.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2.0
            w = (rot[2, 1] - rot[1, 2]) / s
            x = 0.25 * s
            y = (rot[0, 1] + rot[1, 0]) / s
            z = (rot[0, 2] + rot[2, 0]) / s
        elif idx == 1:
            s = math.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2.0
            w = (rot[0, 2] - rot[2, 0]) / s
            x = (rot[0, 1] + rot[1, 0]) / s
            y = 0.25 * s
            z = (rot[1, 2] + rot[2, 1]) / s
        else:
            s = math.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2.0
            w = (rot[1, 0] - rot[0, 1]) / s
            x = (rot[0, 2] + rot[2, 0]) / s
            y = (rot[1, 2] + rot[2, 1]) / s
            z = 0.25 * s
    quat = np.asarray((w, x, y, z), dtype=np.float64)
    return quat / max(float(np.linalg.norm(quat)), 1.0e-9)


def parse_origin(node: ET.Element | None) -> np.ndarray:
    if node is None:
        return np.eye(4, dtype=np.float64)
    return transform_from_xyz_rpy(parse_vec(node.attrib.get("xyz"), (0.0, 0.0, 0.0)), parse_vec(node.attrib.get("rpy"), (0.0, 0.0, 0.0)))


def resolve_mesh_path(filename: str, urdf_path: Path) -> Path:
    if filename.startswith("package://"):
        parts = filename.removeprefix("package://").split("/", 1)
        filename = parts[1] if len(parts) == 2 else parts[0]
    path = Path(filename)
    if not path.is_absolute():
        path = urdf_path.parent / path
    return path.resolve()


def link_color(mesh_path: Path) -> tuple[int, int, int]:
    text = str(mesh_path)
    if "/meshes/arm/" in text:
        return (225, 225, 225)
    if "/meshes/g2/" in text or "/meshes/connector/" in text:
        return (245, 206, 80)
    if "foot" in text:
        return (35, 35, 35)
    return (235, 235, 235)


def load_robot_model(urdf_path: Path) -> RobotModel:
    tree = ET.parse(urdf_path)
    root_node = tree.getroot()
    visuals_by_link: dict[str, list[Visual]] = {}
    joints_by_parent: dict[str, list[Joint]] = {}
    joints_by_name: dict[str, Joint] = {}
    limits_by_joint: dict[str, tuple[float, float]] = {}
    child_links: set[str] = set()
    all_links: set[str] = set()

    for link_node in root_node.findall("link"):
        link_name = link_node.attrib["name"]
        all_links.add(link_name)
        visuals: list[Visual] = []
        for visual_node in link_node.findall("visual"):
            mesh_node = visual_node.find("geometry/mesh")
            if mesh_node is None or "filename" not in mesh_node.attrib:
                continue
            mesh_path = resolve_mesh_path(mesh_node.attrib["filename"], urdf_path)
            visuals.append(Visual(mesh_path, parse_origin(visual_node.find("origin")), link_color(mesh_path)))
        visuals_by_link[link_name] = visuals

    for joint_node in root_node.findall("joint"):
        parent_node = joint_node.find("parent")
        child_node = joint_node.find("child")
        if parent_node is None or child_node is None:
            continue
        axis = parse_vec(joint_node.find("axis").attrib.get("xyz") if joint_node.find("axis") is not None else None, (1.0, 0.0, 0.0))
        limit_node = joint_node.find("limit")
        lower = float(limit_node.attrib["lower"]) if limit_node is not None and "lower" in limit_node.attrib else None
        upper = float(limit_node.attrib["upper"]) if limit_node is not None and "upper" in limit_node.attrib else None
        joint = Joint(
            name=joint_node.attrib["name"],
            parent=parent_node.attrib["link"],
            child=child_node.attrib["link"],
            joint_type=joint_node.attrib.get("type", "fixed"),
            parent_from_joint=parse_origin(joint_node.find("origin")),
            axis=axis,
            lower=lower,
            upper=upper,
        )
        joints_by_name[joint.name] = joint
        joints_by_parent.setdefault(joint.parent, []).append(joint)
        child_links.add(joint.child)
        if lower is not None and upper is not None:
            limits_by_joint[joint.name] = (lower, upper)

    roots = sorted(all_links - child_links)
    return RobotModel(roots[0] if roots else "base", visuals_by_link, joints_by_parent, joints_by_name, limits_by_joint)


def load_mesh_vertices_faces(model: RobotModel, mesh_path: Path, max_faces: int) -> tuple[np.ndarray, np.ndarray]:
    cache_key = (mesh_path, max_faces)
    if cache_key in model.mesh_cache:
        return model.mesh_cache[cache_key]
    mesh = trimesh.load(mesh_path, force="mesh")
    if isinstance(mesh, trimesh.Scene):
        mesh = trimesh.util.concatenate(tuple(mesh.dump()))
    vertices = np.asarray(mesh.vertices, dtype=np.float32)
    faces = np.asarray(mesh.faces, dtype=np.uint32)
    if max_faces > 0 and faces.shape[0] > max_faces:
        face_idx = np.linspace(0, faces.shape[0] - 1, max_faces, dtype=np.int64)
        faces = faces[face_idx]
        used_vertices, inverse = np.unique(faces.reshape(-1), return_inverse=True)
        vertices = vertices[used_vertices]
        faces = inverse.reshape((-1, 3)).astype(np.uint32)
    model.mesh_cache[cache_key] = (vertices, faces)
    return vertices, faces


def _parse_range(values: list[float] | None, default: tuple[tuple[float, float], ...]) -> tuple[tuple[float, float], ...]:
    if values is None:
        return default
    if len(values) != 6:
        raise ValueError("Expected six values: xmin xmax ymin ymax zmin zmax.")
    return ((values[0], values[1]), (values[2], values[3]), (values[4], values[5]))


def _box_segments(bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]) -> np.ndarray:
    xs, ys, zs = bounds
    corners = np.asarray([[x, y, z] for x in xs for y in ys for z in zs], dtype=np.float32)
    segments = []
    for i, a in enumerate(corners):
        for b in corners[i + 1 :]:
            if np.count_nonzero(np.isclose(a, b)) == 2:
                segments.append([a, b])
    return np.asarray(segments, dtype=np.float32)


def _grid_points(
    bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]],
    count: int,
    shell_only: bool,
    workspace_origin: np.ndarray = DEFAULT_WORKSPACE_ORIGIN,
    reach_range: tuple[float, float] = DEFAULT_REACH_RANGE,
) -> np.ndarray:
    axes = [np.linspace(lo, hi, count, dtype=np.float32) for lo, hi in bounds]
    points = []
    for ix, x in enumerate(axes[0]):
        for iy, y in enumerate(axes[1]):
            for iz, z in enumerate(axes[2]):
                if shell_only and not (ix in (0, count - 1) or iy in (0, count - 1) or iz in (0, count - 1)):
                    continue
                point = np.asarray((x, y, z), dtype=np.float64)
                radial = point - workspace_origin
                reach = float(np.linalg.norm(radial))
                if not reach_range[0] <= reach <= reach_range[1]:
                    continue
                if all(low <= point[axis] <= high for axis, (low, high) in enumerate(DEFAULT_KEEP_OUT)):
                    continue
                points.append(point)
    if not points:
        return np.zeros((0, 3), dtype=np.float32)
    return np.unique(np.round(np.asarray(points, dtype=np.float64), decimals=6), axis=0).astype(np.float32)


def _color_segments(count: int, rgb: tuple[int, int, int]) -> np.ndarray:
    colors = np.zeros((count, 2, 3), dtype=np.uint8)
    colors[:, :, :] = np.asarray(rgb, dtype=np.uint8)
    return colors


def _add_line_segments(server, name: str, points: np.ndarray, rgb: tuple[int, int, int], line_width: float):
    return server.scene.add_line_segments(name, points=points, colors=_color_segments(points.shape[0], rgb), line_width=line_width)


def _add_point_cloud(server, name: str, points: np.ndarray, rgb: tuple[int, int, int], point_size: float):
    colors = np.zeros((points.shape[0], 3), dtype=np.uint8)
    colors[:, :] = np.asarray(rgb, dtype=np.uint8)
    return server.scene.add_point_cloud(name, points=points, colors=colors, point_size=point_size)


def _point_colors(points: np.ndarray, rgb: tuple[int, int, int]) -> np.ndarray:
    colors = np.zeros((points.shape[0], 3), dtype=np.uint8)
    colors[:, :] = np.asarray(rgb, dtype=np.uint8)
    return colors


def _format_workspace(bounds: tuple[tuple[float, float], tuple[float, float], tuple[float, float]]) -> str:
    return (
        f"(({bounds[0][0]:.3f}, {bounds[0][1]:.3f}), "
        f"({bounds[1][0]:.3f}, {bounds[1][1]:.3f}), "
        f"({bounds[2][0]:.3f}, {bounds[2][1]:.3f}))"
    )


def default_joint_positions() -> dict[str, float]:
    joint_pos = dict(DEFAULT_GO2_STANCE)
    joint_pos.update(dict(zip(ARM_JOINTS, CARRY_Q.tolist())))
    return joint_pos


def arm_joint_dict(q: np.ndarray) -> dict[str, float]:
    return dict(zip(ARM_JOINTS, q.astype(float).tolist()))


def ee_position(model: RobotModel, base_joint_pos: dict[str, float], q: np.ndarray, ee_link: str) -> np.ndarray:
    joint_pos = dict(base_joint_pos)
    joint_pos.update(arm_joint_dict(q))
    return model.fk(joint_pos)[ee_link][:3, 3]


def solve_ik(
    model: RobotModel,
    target: np.ndarray,
    base_joint_pos: dict[str, float],
    seed_q: np.ndarray,
    limits: np.ndarray,
    ee_link: str,
    max_iters: int,
    damping: float,
    tol: float,
) -> tuple[np.ndarray, float, bool]:
    q = np.clip(seed_q.astype(np.float64, copy=True), limits[:, 0], limits[:, 1])
    eps = 1.0e-4
    for _ in range(max_iters):
        pos = ee_position(model, base_joint_pos, q, ee_link)
        err = target - pos
        err_norm = float(np.linalg.norm(err))
        if err_norm <= tol:
            return q, err_norm, True
        jac = np.zeros((3, len(q)), dtype=np.float64)
        for i in range(len(q)):
            q_eps = q.copy()
            q_eps[i] = min(q_eps[i] + eps, limits[i, 1])
            jac[:, i] = (ee_position(model, base_joint_pos, q_eps, ee_link) - pos) / max(q_eps[i] - q[i], eps)
        system = jac @ jac.T + (damping * damping) * np.eye(3, dtype=np.float64)
        dq = jac.T @ np.linalg.solve(system, err)
        dq = np.clip(dq, -0.16, 0.16)
        q = np.clip(q + dq, limits[:, 0], limits[:, 1])
    final_err = float(np.linalg.norm(target - ee_position(model, base_joint_pos, q, ee_link)))
    return q, final_err, final_err <= tol


def solve_ik_multistart(
    model: RobotModel,
    target: np.ndarray,
    base_joint_pos: dict[str, float],
    primary_seed: np.ndarray,
    limits: np.ndarray,
    ee_link: str,
    max_iters: int,
    damping: float,
    tol: float,
    max_starts: int,
) -> tuple[np.ndarray, float, bool]:
    """Solve position IK from several deterministic D1 branches and keep the best result."""
    shoulder_yaw = -math.atan2(float(target[1]), max(float(target[0]), 1.0e-6))
    seed_library = (
        primary_seed,
        WORKSPACE_READY_Q + np.asarray((shoulder_yaw, 0.0, 0.0, 0.0, 0.0, 0.0)),
        CARRY_Q + np.asarray((shoulder_yaw, 0.0, 0.0, 0.0, 0.0, 0.0)),
        FOLDED_Q + np.asarray((shoulder_yaw, 0.0, 0.0, 0.0, 0.0, 0.0)),
        np.asarray((shoulder_yaw, 0.95, -0.65, 0.0, -0.35, 0.0), dtype=np.float64),
        np.asarray((shoulder_yaw, -0.45, 0.85, 0.0, -0.35, 0.0), dtype=np.float64),
    )
    best_q = np.clip(primary_seed.astype(np.float64, copy=True), limits[:, 0], limits[:, 1])
    best_error = float("inf")
    for seed in seed_library[: max(1, int(max_starts))]:
        q, error, reachable = solve_ik(
            model,
            target,
            base_joint_pos,
            seed,
            limits,
            ee_link,
            max_iters,
            damping,
            tol,
        )
        if error < best_error:
            best_q, best_error = q, error
        if reachable:
            return q, error, True
    return best_q, best_error, best_error <= tol


def arm_chain_segments(model: RobotModel, base_joint_pos: dict[str, float], q: np.ndarray) -> np.ndarray:
    joint_pos = dict(base_joint_pos)
    joint_pos.update(arm_joint_dict(q))
    transforms = model.fk(joint_pos)
    points = [transforms[link][:3, 3] for link in ARM_LINK_CHAIN if link in transforms]
    return np.asarray([[points[i], points[i + 1]] for i in range(len(points) - 1)], dtype=np.float32)


def solve_workspace(
    model: RobotModel,
    targets: np.ndarray,
    base_joint_pos: dict[str, float],
    limits: np.ndarray,
    max_samples: int,
    max_ghosts: int,
    max_iters: int,
    damping: float,
    tol: float,
    ee_link: str,
    max_starts: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(targets.shape[0])
    if targets.shape[0] > max_samples:
        idx = np.linspace(0, targets.shape[0] - 1, max_samples, dtype=np.int64)
    sample_targets = targets[idx]
    ghost_indices: set[int] = set()
    if max_ghosts > 0 and sample_targets.shape[0] > 0:
        ghost_count = min(max_ghosts, sample_targets.shape[0])
        ghost_indices = set(np.linspace(0, sample_targets.shape[0] - 1, ghost_count, dtype=np.int64).tolist())
    solved_q = []
    errors = []
    reachable = []
    segments_ok = []
    segments_bad = []
    seed = WORKSPACE_READY_Q
    for sample_index, target in enumerate(sample_targets):
        q, err, ok = solve_ik_multistart(
            model,
            target.astype(np.float64),
            base_joint_pos,
            seed,
            limits,
            ee_link,
            max_iters,
            damping,
            tol,
            max_starts,
        )
        solved_q.append(q)
        errors.append(err)
        reachable.append(ok)
        if sample_index in ghost_indices:
            if ok:
                segments_ok.append(arm_chain_segments(model, base_joint_pos, q))
            else:
                segments_bad.append(arm_chain_segments(model, base_joint_pos, q))
    ok_lines = np.concatenate(segments_ok, axis=0) if segments_ok else np.zeros((0, 2, 3), dtype=np.float32)
    bad_lines = np.concatenate(segments_bad, axis=0) if segments_bad else np.zeros((0, 2, 3), dtype=np.float32)
    return (
        sample_targets,
        np.asarray(solved_q, dtype=np.float64).reshape((-1, len(ARM_JOINTS))),
        np.asarray(errors, dtype=np.float64),
        np.asarray(reachable, dtype=bool),
        np.concatenate((ok_lines, bad_lines), axis=0),
    )


def write_results(path: Path, sample_targets: np.ndarray, solved_q: np.ndarray, errors: np.ndarray, reachable: np.ndarray) -> None:
    with path.open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(("x", "y", "z", *ARM_JOINTS, "ik_error_m", "reachable"))
        for target, q, err, ok in zip(sample_targets, solved_q, errors, reachable):
            writer.writerow((*target.tolist(), *q.tolist(), float(err), bool(ok)))


def render_robot(server, model: RobotModel, joint_pos: dict[str, float], opacity: float, max_faces_per_mesh: int):
    transforms = model.fk(joint_pos)
    handles = []
    for link_name, visuals in model.visuals_by_link.items():
        if link_name not in transforms:
            continue
        for index, visual in enumerate(visuals):
            vertices, faces = load_mesh_vertices_faces(model, visual.mesh_path, max_faces=max_faces_per_mesh)
            world_from_visual = transforms[link_name] @ visual.link_from_visual
            handle = server.scene.add_mesh_simple(
                f"/robot/{link_name}/visual_{index}",
                vertices=vertices,
                faces=faces,
                color=visual.color,
                opacity=opacity if opacity < 1.0 else None,
                side="double",
                flat_shading=False,
                position=world_from_visual[:3, 3],
                wxyz=matrix_to_wxyz(world_from_visual),
            )
            handles.append((handle, link_name, visual.link_from_visual))
    return handles


def update_robot(handles, model: RobotModel, joint_pos: dict[str, float]) -> None:
    transforms = model.fk(joint_pos)
    for handle, link_name, link_from_visual in handles:
        if link_name not in transforms:
            continue
        world_from_visual = transforms[link_name] @ link_from_visual
        handle.position = world_from_visual[:3, 3]
        handle.wxyz = matrix_to_wxyz(world_from_visual)


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize Go2+D1 arm workspace coverage in Viser.")
    parser.add_argument("--host", default="0.0.0.0", help="Viser host.")
    parser.add_argument("--port", type=int, default=8080, help="Viser port.")
    parser.add_argument("--urdf", type=Path, default=DEFAULT_URDF, help="Full Go2+arm URDF to render and use for FK/IK.")
    parser.add_argument("--workspace", nargs=6, type=float, default=None, metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"))
    parser.add_argument(
        "--workspace-slider-limits",
        nargs=6,
        type=float,
        default=None,
        metavar=("XMIN", "XMAX", "YMIN", "YMAX", "ZMIN", "ZMAX"),
        help="Limits for the workspace min/max GUI sliders.",
    )
    parser.add_argument("--target", nargs=3, type=float, default=None, metavar=("X", "Y", "Z"), help="Initial EE target to solve for.")
    parser.add_argument("--grid", type=int, default=8, help="Samples per workspace axis.")
    parser.add_argument("--max-samples", type=int, default=300, help="Maximum workspace points to solve with IK.")
    parser.add_argument("--max-ghosts", type=int, default=12, help="Maximum IK ghost arm poses to draw.")
    parser.add_argument(
        "--show-ghosts",
        action="store_true",
        help="Show solved arm skeletons. Hidden by default to keep the workspace readable.",
    )
    parser.add_argument("--shell-only", action="store_true", help="Only draw workspace shell samples.")
    parser.add_argument("--no-keepout", action="store_true", help="Hide the Go2 body keep-out box.")
    parser.add_argument("--ik-iters", type=int, default=90, help="Damped least-squares IK iterations per target.")
    parser.add_argument("--ik-damping", type=float, default=0.06, help="Damped least-squares IK damping.")
    parser.add_argument("--ik-tol", type=float, default=0.025, help="Target is reachable if final IK error is below this many meters.")
    parser.add_argument("--ik-starts", type=int, default=5, help="Deterministic IK seeds tried per workspace point.")
    parser.add_argument("--robot-opacity", type=float, default=0.78, help="Opacity for the full robot mesh.")
    parser.add_argument(
        "--max-faces-per-mesh",
        type=int,
        default=30000,
        help="Cap faces per visual mesh for browser rendering. Use 0 to keep original mesh density.",
    )
    parser.add_argument("--save-results", type=Path, default=None, help="Optional CSV path for sampled IK results.")
    args = parser.parse_args()

    try:
        import viser
    except ModuleNotFoundError as exc:
        raise SystemExit("viser is not installed. Install it in this environment with: pip install viser") from exc

    urdf_path = args.urdf.resolve()
    if not urdf_path.exists():
        raise SystemExit(f"URDF not found: {urdf_path}")
    model = load_robot_model(urdf_path)
    base_joint_pos = default_joint_positions()
    limits = 0.9 * model.joint_limits(ARM_JOINTS)
    workspace = _parse_range(args.workspace, DEFAULT_WORKSPACE)
    workspace_slider_limits = _parse_range(args.workspace_slider_limits, DEFAULT_WORKSPACE_SLIDER_LIMITS)
    targets = _grid_points(workspace, max(2, args.grid), shell_only=args.shell_only)
    training_volume_points = _grid_points(workspace, max(18, args.grid), shell_only=False)
    sample_targets, solved_q, errors, reachable, ghost_lines = solve_workspace(
        model,
        targets,
        base_joint_pos,
        limits,
        max(1, args.max_samples),
        max(0, args.max_ghosts),
        args.ik_iters,
        args.ik_damping,
        args.ik_tol,
        EE_LINK,
        args.ik_starts,
    )

    if args.save_results is not None:
        write_results(args.save_results, sample_targets, solved_q, errors, reachable)

    initial_target = np.asarray(args.target if args.target is not None else [(lo + hi) * 0.5 for lo, hi in workspace], dtype=np.float64)
    selected_q, selected_error, selected_ok = solve_ik_multistart(
        model,
        initial_target,
        base_joint_pos,
        WORKSPACE_READY_Q,
        limits,
        EE_LINK,
        args.ik_iters,
        args.ik_damping,
        args.ik_tol,
        args.ik_starts,
    )
    robot_joint_pos = dict(base_joint_pos)
    robot_joint_pos.update(arm_joint_dict(selected_q))

    server = viser.ViserServer(host=args.host, port=args.port)
    server.scene.set_up_direction("+z")
    server.initial_camera.position = (1.20, -1.20, 0.85)
    server.initial_camera.look_at = (0.15, 0.0, 0.30)
    server.initial_camera.up_direction = (0.0, 0.0, 1.0)
    url = f"http://{args.host if args.host != '0.0.0.0' else 'localhost'}:{args.port}"
    print(f"[INFO] Viser server: {url}", flush=True)
    server.scene.world_axes.visible = True
    server.scene.add_grid("/ground", width=1.4, height=1.0, cell_size=0.05, section_size=0.25, plane="xy")
    robot_handles = render_robot(
        server,
        model,
        robot_joint_pos,
        opacity=float(np.clip(args.robot_opacity, 0.05, 1.0)),
        max_faces_per_mesh=args.max_faces_per_mesh,
    )

    reach_outer_surface = server.scene.add_icosphere(
        "/workspace/reach_shell/outer_surface",
        radius=DEFAULT_REACH_RANGE[1],
        color=(70, 190, 255),
        subdivisions=3,
        opacity=0.045,
        side="double",
        cast_shadow=False,
        receive_shadow=False,
        position=DEFAULT_WORKSPACE_ORIGIN,
        visible=False,
    )
    reach_outer_boundary = server.scene.add_icosphere(
        "/workspace/reach_shell/outer_boundary",
        radius=DEFAULT_REACH_RANGE[1],
        color=(70, 190, 255),
        subdivisions=2,
        wireframe=True,
        opacity=0.45,
        cast_shadow=False,
        receive_shadow=False,
        position=DEFAULT_WORKSPACE_ORIGIN,
        visible=False,
    )
    reach_inner_keepout = server.scene.add_icosphere(
        "/workspace/reach_shell/inner_keepout",
        radius=DEFAULT_REACH_RANGE[0],
        color=(255, 92, 92),
        subdivisions=2,
        opacity=0.18,
        side="double",
        cast_shadow=False,
        receive_shadow=False,
        position=DEFAULT_WORKSPACE_ORIGIN,
        visible=False,
    )
    server.scene.add_frame(
        "/workspace/d1_mount_origin",
        position=DEFAULT_WORKSPACE_ORIGIN,
        axes_length=0.07,
        axes_radius=0.003,
    )
    command_box = _add_line_segments(
        server,
        "/workspace/command_box",
        _box_segments(workspace),
        (255, 171, 64),
        2.0,
    )
    keepout_box = None
    if not args.no_keepout:
        keepout_box = _add_line_segments(
            server,
            "/workspace/body_keepout",
            _box_segments(DEFAULT_KEEP_OUT),
            (255, 72, 72),
            2.5,
        )
    if ghost_lines.shape[0] == 0:
        ghost_lines = np.zeros((0, 2, 3), dtype=np.float32)
    ghost_arms = _add_line_segments(server, "/d1/ik_ghost_arms", ghost_lines, (120, 170, 255), 0.8)
    ghost_arms.visible = bool(args.show_ghosts)
    training_volume = _add_point_cloud(
        server,
        "/workspace/training_volume",
        training_volume_points,
        (145, 205, 230),
        0.004,
    )
    reachable_targets = _add_point_cloud(
        server,
        "/workspace/reachable_targets",
        sample_targets[reachable],
        (52, 211, 126),
        0.012,
    )
    failed_targets = _add_point_cloud(
        server,
        "/workspace/failed_targets",
        sample_targets[~reachable],
        (255, 82, 82),
        0.018,
    )
    target_frame = server.scene.add_frame("/ik/target", position=initial_target, axes_length=0.08, axes_radius=0.004)
    ee_frame = server.scene.add_frame("/ik/solved_eef", axes_length=0.08, axes_radius=0.004)
    selected_line = _add_line_segments(server, "/ik/target_to_eef_error", np.zeros((1, 2, 3), dtype=np.float32), (255, 70, 70), 3.0)

    status = server.gui.add_text("IK status", "")
    workspace_status = server.gui.add_text("Workspace", "")
    server.gui.add_text(
        "Legend",
        "Pale blue: sampled training volume | green/red: IK pass/fail | amber: command bounds | "
        "red box: clearance-expanded Go2 keep-out | optional spheres: geometric reach",
    )
    show_training_volume = server.gui.add_checkbox("Show training volume", True)
    show_reach_shell = server.gui.add_checkbox("Show geometric reach spheres", False)
    show_command_box = server.gui.add_checkbox("Show command box", True)
    show_sample_points = server.gui.add_checkbox("Show IK samples", True)
    show_ghost_arms = server.gui.add_checkbox("Show ghost arms", bool(args.show_ghosts))
    show_keepout_box = server.gui.add_checkbox(
        "Show body keep-out",
        not args.no_keepout,
        disabled=args.no_keepout,
    )
    workspace_x_min = server.gui.add_slider(
        "Workspace x min", workspace_slider_limits[0][0], workspace_slider_limits[0][1], 0.005, float(workspace[0][0])
    )
    workspace_x_max = server.gui.add_slider(
        "Workspace x max", workspace_slider_limits[0][0], workspace_slider_limits[0][1], 0.005, float(workspace[0][1])
    )
    workspace_y_min = server.gui.add_slider(
        "Workspace y min", workspace_slider_limits[1][0], workspace_slider_limits[1][1], 0.005, float(workspace[1][0])
    )
    workspace_y_max = server.gui.add_slider(
        "Workspace y max", workspace_slider_limits[1][0], workspace_slider_limits[1][1], 0.005, float(workspace[1][1])
    )
    workspace_z_min = server.gui.add_slider(
        "Workspace z min", workspace_slider_limits[2][0], workspace_slider_limits[2][1], 0.005, float(workspace[2][0])
    )
    workspace_z_max = server.gui.add_slider(
        "Workspace z max", workspace_slider_limits[2][0], workspace_slider_limits[2][1], 0.005, float(workspace[2][1])
    )
    apply_workspace = server.gui.add_button("Apply workspace")
    target_x = server.gui.add_slider("Target x", workspace[0][0], workspace[0][1], 0.005, float(initial_target[0]))
    target_y = server.gui.add_slider("Target y", workspace[1][0], workspace[1][1], 0.005, float(initial_target[1]))
    target_z = server.gui.add_slider("Target z", workspace[2][0], workspace[2][1], 0.005, float(initial_target[2]))

    state = {"q": selected_q, "workspace": workspace}

    def read_workspace_sliders() -> tuple[tuple[float, float], tuple[float, float], tuple[float, float]]:
        bounds = (
            (float(workspace_x_min.value), float(workspace_x_max.value)),
            (float(workspace_y_min.value), float(workspace_y_max.value)),
            (float(workspace_z_min.value), float(workspace_z_max.value)),
        )
        fixed_bounds = []
        for low, high in bounds:
            if high < low + 0.005:
                high = low + 0.005
            fixed_bounds.append((low, high))
        return tuple(fixed_bounds)  # type: ignore[return-value]

    def update_workspace_status(bounds, sample_targets, errors, reachable) -> None:
        if sample_targets.shape[0] == 0:
            workspace_status.value = (
                "0 IK samples: command box does not overlap the configured reach shell outside the body keep-out | "
                f"ee_pos_range={_format_workspace(bounds)}"
            )
            return
        workspace_status.value = (
            f"{int(reachable.sum())}/{reachable.shape[0]} reachable | "
            f"median {np.median(errors) * 1000.0:.1f} mm | "
            f"p95 {np.percentile(errors, 95) * 1000.0:.1f} mm | "
            f"ee_pos_range={_format_workspace(bounds)}"
        )

    def set_target_slider_bounds(bounds) -> None:
        target_x.min, target_x.max = bounds[0]
        target_y.min, target_y.max = bounds[1]
        target_z.min, target_z.max = bounds[2]
        target_x.value = float(np.clip(target_x.value, bounds[0][0], bounds[0][1]))
        target_y.value = float(np.clip(target_y.value, bounds[1][0], bounds[1][1]))
        target_z.value = float(np.clip(target_z.value, bounds[2][0], bounds[2][1]))

    def recompute_workspace(bounds) -> None:
        targets_new = _grid_points(bounds, max(2, args.grid), shell_only=args.shell_only)
        training_volume_new = _grid_points(bounds, max(18, args.grid), shell_only=False)
        sample_targets_new, solved_q_new, errors_new, reachable_new, ghost_lines_new = solve_workspace(
            model,
            targets_new,
            base_joint_pos,
            limits,
            max(1, args.max_samples),
            max(0, args.max_ghosts),
            args.ik_iters,
            args.ik_damping,
            args.ik_tol,
            EE_LINK,
            args.ik_starts,
        )
        if args.save_results is not None:
            write_results(args.save_results, sample_targets_new, solved_q_new, errors_new, reachable_new)
        command_box.points = _box_segments(bounds)
        ghost_arms.points = ghost_lines_new.astype(np.float32)
        ghost_arms.colors = _color_segments(ghost_lines_new.shape[0], (92, 176, 255))
        reachable_points = sample_targets_new[reachable_new]
        failed_points = sample_targets_new[~reachable_new]
        reachable_targets.points = reachable_points.astype(np.float32)
        reachable_targets.colors = _point_colors(reachable_points, (68, 220, 116))
        failed_targets.points = failed_points.astype(np.float32)
        failed_targets.colors = _point_colors(failed_points, (255, 70, 70))
        training_volume.points = training_volume_new.astype(np.float32)
        training_volume.colors = _point_colors(training_volume_new, (145, 205, 230))
        state["workspace"] = bounds
        set_target_slider_bounds(bounds)
        update_workspace_status(bounds, sample_targets_new, errors_new, reachable_new)
        update_selected_pose(seed_from_current=False)

    def update_selected_pose(seed_from_current: bool = True) -> None:
        target = np.asarray((target_x.value, target_y.value, target_z.value), dtype=np.float64)
        seed = state["q"] if seed_from_current else WORKSPACE_READY_Q
        q, error, ok = solve_ik_multistart(
            model,
            target,
            base_joint_pos,
            seed,
            limits,
            EE_LINK,
            args.ik_iters,
            args.ik_damping,
            args.ik_tol,
            args.ik_starts,
        )
        state["q"] = q
        joint_pos = dict(base_joint_pos)
        joint_pos.update(arm_joint_dict(q))
        transforms = model.fk(joint_pos)
        ee_pos = transforms[EE_LINK][:3, 3]
        update_robot(robot_handles, model, joint_pos)
        target_frame.position = target
        ee_frame.position = ee_pos
        ee_frame.wxyz = matrix_to_wxyz(transforms[EE_LINK])
        selected_line.points = np.asarray([[target, ee_pos]], dtype=np.float32)
        status.value = f"{'reachable' if ok else 'failed'} | error {error * 1000.0:.1f} mm | q {[round(v, 3) for v in q]}"

    @target_x.on_update
    def _(_) -> None:
        update_selected_pose()

    @target_y.on_update
    def _(_) -> None:
        update_selected_pose()

    @target_z.on_update
    def _(_) -> None:
        update_selected_pose()

    @apply_workspace.on_click
    def _(_) -> None:
        recompute_workspace(read_workspace_sliders())

    @show_reach_shell.on_update
    def _(_) -> None:
        visible = bool(show_reach_shell.value)
        reach_outer_surface.visible = visible
        reach_outer_boundary.visible = visible
        reach_inner_keepout.visible = visible

    @show_training_volume.on_update
    def _(_) -> None:
        training_volume.visible = bool(show_training_volume.value)

    @show_command_box.on_update
    def _(_) -> None:
        command_box.visible = bool(show_command_box.value)

    @show_sample_points.on_update
    def _(_) -> None:
        visible = bool(show_sample_points.value)
        reachable_targets.visible = visible
        failed_targets.visible = visible

    @show_ghost_arms.on_update
    def _(_) -> None:
        ghost_arms.visible = bool(show_ghost_arms.value)

    @show_keepout_box.on_update
    def _(_) -> None:
        if keepout_box is not None:
            keepout_box.visible = bool(show_keepout_box.value)

    update_selected_pose(seed_from_current=False)
    update_workspace_status(workspace, sample_targets, errors, reachable)

    print(f"[INFO] URDF: {urdf_path}")
    print(f"[INFO] Workspace: x={workspace[0]}, y={workspace[1]}, z={workspace[2]}")
    print(f"[INFO] Sampled IK targets: {sample_targets.shape[0]} / {targets.shape[0]}")
    print(f"[INFO] Reachable under this URDF IK tolerance: {int(reachable.sum())}/{reachable.shape[0]}")
    print(f"[INFO] Median/Max sampled IK error: {np.median(errors) * 1000.0:.1f} mm / {np.max(errors) * 1000.0:.1f} mm")
    print(f"[INFO] Initial target solve: {'reachable' if selected_ok else 'failed'}, error={selected_error * 1000.0:.1f} mm")
    if args.save_results is not None:
        print(f"[INFO] Wrote sampled IK results: {args.save_results}")

    while True:
        time.sleep(1.0)


if __name__ == "__main__":
    main()
