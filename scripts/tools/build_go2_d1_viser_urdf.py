#!/usr/bin/env python3
"""Build a Viser-compatible Go2+D1 URDF for live training visualization.

The D1 arm meshes and kinematics come from Unitree's public D1-550 URDF package:
https://oss-global-cdn.unitree.com/static/9b20252a26374d50aa369532657d0143.zip

The Go2 leg model is taken from this repo's ``go2_description`` asset. Joint names are
renamed to match the Isaac Lab Go2+D1 USD asset (``arm_1_joint`` … ``arm_7_2_joint``).
"""

from __future__ import annotations

import argparse
import hashlib
import re
import shutil
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
ASSETS_ROOT = REPO_ROOT / "source/Gurukul/data/Robots/unitree"
GO2_URDF = ASSETS_ROOT / "go2_description/urdf/go2_description.urdf"
OUTPUT_DIR = ASSETS_ROOT / "go2_with_d1"
OUTPUT_URDF = OUTPUT_DIR / "urdf/go2_d1_vis.urdf"
D1_MESH_DIR = OUTPUT_DIR / "meshes/d1"
D1_ZIP_URL = "https://oss-global-cdn.unitree.com/static/9b20252a26374d50aa369532657d0143.zip"
D1_ZIP_SHA256 = "71abcc4cd6359bf09765b1fa9e87ed5369b6a688bf8c9432d5ccc2dd8e42faa5"

D1_JOINT_RENAMES = {
    "Joint1": "arm_1_joint",
    "Joint2": "arm_2_joint",
    "Joint3": "arm_3_joint",
    "Joint4": "arm_4_joint",
    "Joint5": "arm_5_joint",
    "Joint6": "arm_6_joint",
    "Joint_L": "arm_7_1_joint",
    "Joint_R": "arm_7_2_joint",
}

# Mount taken from go2_d1_center_gripper.usda (base -> d1/base_link).
D1_MOUNT_XYZ = "0 0 0.08"
D1_MOUNT_RPY = "0 0 0"


def _local(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def _rewrite_go2_mesh_paths(mesh_filename: str) -> str:
    match = re.search(r"go2_description/meshes/(.+)$", mesh_filename)
    if match:
        return f"../../go2_description/meshes/{match.group(1)}"
    return mesh_filename


def _load_go2_robot() -> ET.Element:
    tree = ET.parse(GO2_URDF)
    root = tree.getroot()
    assert _local(root.tag) == "robot"
    root.set("name", "go2_d1_vis")

    for mesh in root.iter():
        if _local(mesh.tag) != "mesh":
            continue
        filename = mesh.get("filename")
        if filename:
            mesh.set("filename", _rewrite_go2_mesh_paths(filename))
    return root


def _download_d1_source(tmp_dir: Path) -> Path:
    zip_path = tmp_dir / "d1_550_description.zip"
    extract_dir = tmp_dir / "d1_550_description"
    if not extract_dir.is_dir():
        print(f"[INFO] Downloading D1 URDF package from Unitree CDN...")
        urllib.request.urlretrieve(D1_ZIP_URL, zip_path)
        archive_sha256 = hashlib.sha256(zip_path.read_bytes()).hexdigest()
        if archive_sha256 != D1_ZIP_SHA256:
            zip_path.unlink(missing_ok=True)
            raise RuntimeError(
                "Unitree D1-550 archive checksum mismatch: "
                f"expected {D1_ZIP_SHA256}, received {archive_sha256}"
            )
        shutil.unpack_archive(zip_path, tmp_dir)
    return extract_dir / "urdf/d1_550_description.urdf"


def _copy_d1_meshes(source_mesh_dir: Path) -> None:
    D1_MESH_DIR.mkdir(parents=True, exist_ok=True)
    for mesh_path in sorted(source_mesh_dir.glob("*.STL")):
        target = D1_MESH_DIR / mesh_path.name
        if not target.exists() or target.stat().st_mtime < mesh_path.stat().st_mtime:
            shutil.copy2(mesh_path, target)


def _append_d1_arm(go2_root: ET.Element, d1_urdf_path: Path) -> None:
    d1_tree = ET.parse(d1_urdf_path)
    d1_root = d1_tree.getroot()

    mount = ET.SubElement(
        go2_root,
        "joint",
        {
            "name": "d1_mount_joint",
            "type": "fixed",
        },
    )
    ET.SubElement(mount, "origin", {"xyz": D1_MOUNT_XYZ, "rpy": D1_MOUNT_RPY})
    ET.SubElement(mount, "parent", {"link": "base"})
    ET.SubElement(mount, "child", {"link": "d1_base_link"})

    for element in list(d1_root):
        tag = _local(element.tag)
        if tag == "link":
            link_name = element.get("name")
            if link_name == "base_link":
                element.set("name", "d1_base_link")
            for mesh in element.iter():
                if _local(mesh.tag) == "mesh":
                    filename = mesh.get("filename")
                    if filename:
                        mesh_name = Path(filename).name
                        mesh.set("filename", f"../meshes/d1/{mesh_name}")
            go2_root.append(element)
        elif tag == "joint":
            joint_name = element.get("name")
            if joint_name in D1_JOINT_RENAMES:
                element.set("name", D1_JOINT_RENAMES[joint_name])
            for parent in element.findall("parent"):
                if parent.get("link") == "base_link":
                    parent.set("link", "d1_base_link")
            go2_root.append(element)


def build(output_urdf: Path = OUTPUT_URDF) -> Path:
    if not GO2_URDF.is_file():
        raise FileNotFoundError(f"Missing Go2 URDF: {GO2_URDF}")

    tmp_dir = REPO_ROOT / ".cache/go2_d1_viser_urdf"
    tmp_dir.mkdir(parents=True, exist_ok=True)
    d1_urdf_path = _download_d1_source(tmp_dir)
    _copy_d1_meshes(d1_urdf_path.parent)

    go2_root = _load_go2_robot()
    _append_d1_arm(go2_root, d1_urdf_path)

    output_urdf.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(go2_root, space="  ")
    ET.ElementTree(go2_root).write(output_urdf, encoding="utf-8", xml_declaration=True)
    print(f"[INFO] Wrote {output_urdf}")
    return output_urdf


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Go2+D1 Viser URDF for live training visualization.")
    parser.add_argument(
        "--output",
        type=Path,
        default=OUTPUT_URDF,
        help="Output URDF path.",
    )
    args = parser.parse_args()
    build(args.output.resolve())


if __name__ == "__main__":
    main()
