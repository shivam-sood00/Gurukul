from __future__ import annotations

import mmap
import os
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np


DEPTH_BUFFER_MAGIC = 0x524C4442  # "RLDB"
DEPTH_BUFFER_VERSION = 1

HEADER_DTYPE = np.dtype(
    [
        ("magic", np.uint32),
        ("version", np.uint32),
        ("height", np.uint32),
        ("width", np.uint32),
        ("seq", np.uint64),
        ("timestamp_ns", np.uint64),
        ("status", np.uint32),
        ("reserved", np.uint32),
    ]
)


@dataclass
class DepthFrame:
    seq: int
    timestamp_ns: int
    frame: np.ndarray


@dataclass(frozen=True)
class PinholeCameraIntrinsics:
    height: int
    width: int
    fx: float
    fy: float
    cx: float
    cy: float


def intrinsics_from_isaaclab_pinhole_cfg(
    *,
    width: int,
    height: int,
    focal_length: float,
    horizontal_aperture: float,
    vertical_aperture: float,
    horizontal_aperture_offset: float = 0.0,
    vertical_aperture_offset: float = 0.0,
) -> PinholeCameraIntrinsics:
    """Match IsaacLab RayCasterCamera intrinsic construction for a pinhole camera."""
    width_i = int(width)
    height_i = int(height)
    fx = width_i * float(focal_length) / float(horizontal_aperture)
    fy = height_i * float(focal_length) / float(vertical_aperture)
    cx = float(horizontal_aperture_offset) * fx + width_i / 2.0
    cy = float(vertical_aperture_offset) * fy + height_i / 2.0
    return PinholeCameraIntrinsics(height=height_i, width=width_i, fx=fx, fy=fy, cx=cx, cy=cy)


def build_distance_to_camera_scale_map(intrinsics: PinholeCameraIntrinsics) -> np.ndarray:
    """Scale image-plane depth to Euclidean distance using IsaacLab pixel-center conventions."""
    xs = ((np.arange(intrinsics.width, dtype=np.float32) + 0.5) - np.float32(intrinsics.cx)) / np.float32(
        intrinsics.fx
    )
    ys = ((np.arange(intrinsics.height, dtype=np.float32) + 0.5) - np.float32(intrinsics.cy)) / np.float32(
        intrinsics.fy
    )
    scale = 1.0 + np.square(ys, dtype=np.float32)[:, None] + np.square(xs, dtype=np.float32)[None, :]
    np.sqrt(scale, out=scale)
    return scale.astype(np.float32, copy=False)


def convert_image_plane_depth_to_camera_distance(
    depth_image: np.ndarray,
    *,
    scale_map: np.ndarray,
    flip_vertical: bool = False,
    max_distance: float = 0.0,
) -> np.ndarray:
    """Convert image-plane depth to IsaacLab-style distance-to-camera observations."""
    depth = np.asarray(depth_image, dtype=np.float32)
    if flip_vertical:
        depth = np.flipud(depth)
    if depth.shape != scale_map.shape:
        raise ValueError(f"Depth image shape mismatch: expected {scale_map.shape}, got {depth.shape}")

    max_distance_f = float(max_distance)
    posinf_fill = max_distance_f if max_distance_f > 0.0 else 0.0
    depth = np.nan_to_num(depth, nan=0.0, posinf=posinf_fill, neginf=0.0)
    distance = depth * scale_map
    if max_distance_f > 0.0:
        np.clip(distance, 0.0, max_distance_f, out=distance)
    else:
        np.maximum(distance, 0.0, out=distance)
    return distance.astype(np.float32, copy=False)


def resize_bilinear(image: np.ndarray, target_h: int, target_w: int) -> np.ndarray:
    """Resize a 2-D float32 image using bilinear sampling aligned with PyTorch."""
    src = np.asarray(image, dtype=np.float32)
    src_h, src_w = src.shape
    if src_h == target_h and src_w == target_w:
        return src.copy()
    if target_h <= 0 or target_w <= 0:
        raise ValueError(f"Invalid resize target {(target_h, target_w)}")

    ys = (np.arange(target_h, dtype=np.float32) + 0.5) * (src_h / target_h) - 0.5
    xs = (np.arange(target_w, dtype=np.float32) + 0.5) * (src_w / target_w) - 0.5

    y0 = np.floor(ys).astype(np.int32)
    x0 = np.floor(xs).astype(np.int32)
    y1 = y0 + 1
    x1 = x0 + 1

    wy1 = ys - y0
    wx1 = xs - x0
    wy0 = 1.0 - wy1
    wx0 = 1.0 - wx1

    y0 = np.clip(y0, 0, src_h - 1)
    x0 = np.clip(x0, 0, src_w - 1)
    y1 = np.clip(y1, 0, src_h - 1)
    x1 = np.clip(x1, 0, src_w - 1)

    top_left = src[y0[:, None], x0[None, :]]
    top_right = src[y0[:, None], x1[None, :]]
    bottom_left = src[y1[:, None], x0[None, :]]
    bottom_right = src[y1[:, None], x1[None, :]]

    out = (
        top_left * wy0[:, None] * wx0[None, :]
        + top_right * wy0[:, None] * wx1[None, :]
        + bottom_left * wy1[:, None] * wx0[None, :]
        + bottom_right * wy1[:, None] * wx1[None, :]
    )
    return out.astype(np.float32, copy=False)


def preprocess_depth_image(
    raw_depth: np.ndarray,
    *,
    crop_top: int = 0,
    crop_bottom: int = 0,
    crop_left: int = 0,
    crop_right: int = 0,
    resize: tuple[int, int] | None = None,
    normalize: bool = True,
    max_distance: float = 0.0,
    clip_to_max_distance: bool = False,
) -> np.ndarray:
    """Apply the same crop/resize/normalize contract used by Gurukul depth obs."""
    max_distance_f = float(max_distance)
    posinf_fill = max_distance_f if clip_to_max_distance and max_distance_f > 0.0 else 0.0
    depth = np.nan_to_num(np.asarray(raw_depth, dtype=np.float32), nan=0.0, posinf=posinf_fill, neginf=0.0)
    if clip_to_max_distance and max_distance_f > 0.0:
        np.clip(depth, 0.0, max_distance_f, out=depth)

    height, width = depth.shape
    top = max(0, int(crop_top))
    bottom = max(0, int(crop_bottom))
    left = max(0, int(crop_left))
    right = max(0, int(crop_right))

    if top + bottom >= height:
        top = 0
        bottom = 0
    if left + right >= width:
        left = 0
        right = 0

    row_end = height - bottom if bottom > 0 else height
    col_end = width - right if right > 0 else width
    depth = depth[top:row_end, left:col_end]

    if resize is not None:
        target_h, target_w = int(resize[0]), int(resize[1])
        if depth.shape != (target_h, target_w):
            depth = resize_bilinear(depth, target_h, target_w)

    if normalize and max_distance_f > 0.0:
        depth = depth / max_distance_f - 0.5

    return depth.astype(np.float32, copy=False)


def depth_preview_uint8(
    depth_image: np.ndarray,
    *,
    normalized: bool,
    max_distance: float,
) -> np.ndarray:
    """Convert a depth image to an 8-bit preview where closer is brighter."""
    depth = np.asarray(depth_image, dtype=np.float32)
    if normalized and max_distance > 0.0:
        scaled = np.clip(depth + 0.5, 0.0, 1.0)
    elif max_distance > 0.0:
        scaled = np.clip(depth / max_distance, 0.0, 1.0)
    else:
        finite = depth[np.isfinite(depth)]
        if finite.size == 0:
            scaled = np.zeros_like(depth, dtype=np.float32)
        else:
            lo = float(finite.min())
            hi = float(finite.max())
            denom = hi - lo if hi > lo else 1.0
            scaled = np.clip((depth - lo) / denom, 0.0, 1.0)
    preview = (1.0 - scaled) * 255.0
    return preview.astype(np.uint8, copy=False)


class DepthFrameWriter:
    def __init__(self, buffer_path: str | os.PathLike[str], height: int, width: int):
        self.buffer_path = Path(buffer_path)
        self.buffer_path.parent.mkdir(parents=True, exist_ok=True)
        self.height = int(height)
        self.width = int(width)
        self.total_size = HEADER_DTYPE.itemsize + self.height * self.width * np.dtype(np.float32).itemsize
        self._seq = 0

        fd = os.open(str(self.buffer_path), os.O_RDWR | os.O_CREAT)
        try:
            os.ftruncate(fd, self.total_size)
            self._mm = mmap.mmap(fd, self.total_size, access=mmap.ACCESS_WRITE)
        finally:
            os.close(fd)

        self.header = np.ndarray((1,), dtype=HEADER_DTYPE, buffer=self._mm, offset=0)
        self.frame = np.ndarray(
            (self.height, self.width),
            dtype=np.float32,
            buffer=self._mm,
            offset=HEADER_DTYPE.itemsize,
        )
        self.header["magic"][0] = DEPTH_BUFFER_MAGIC
        self.header["version"][0] = DEPTH_BUFFER_VERSION
        self.header["height"][0] = self.height
        self.header["width"][0] = self.width
        self.header["seq"][0] = 0
        self.header["timestamp_ns"][0] = 0
        self.header["status"][0] = 0
        self.frame.fill(0.0)

    def write(self, depth_frame: np.ndarray):
        depth = np.asarray(depth_frame, dtype=np.float32)
        if depth.shape != (self.height, self.width):
            raise ValueError(
                f"Depth frame shape mismatch: expected {(self.height, self.width)}, got {depth.shape}"
            )

        seq_start = self._seq + 1
        self.header["seq"][0] = seq_start
        np.copyto(self.frame, depth)
        self.header["timestamp_ns"][0] = time.time_ns()
        self.header["status"][0] = 1
        self._seq = seq_start + 1
        self.header["seq"][0] = self._seq

    def close(self):
        self._mm.close()


class DepthFrameReader:
    def __init__(
        self,
        buffer_path: str | os.PathLike[str],
        *,
        expected_height: int | None = None,
        expected_width: int | None = None,
    ):
        self.buffer_path = Path(buffer_path)
        self.expected_height = expected_height
        self.expected_width = expected_width
        self._mm = None
        self.header = None
        self.frame = None

    def _open_if_needed(self) -> bool:
        if self._mm is not None:
            return True
        if not self.buffer_path.is_file():
            return False

        size = self.buffer_path.stat().st_size
        if size < HEADER_DTYPE.itemsize:
            return False

        fd = os.open(str(self.buffer_path), os.O_RDONLY)
        try:
            self._mm = mmap.mmap(fd, size, access=mmap.ACCESS_READ)
        finally:
            os.close(fd)

        self.header = np.ndarray((1,), dtype=HEADER_DTYPE, buffer=self._mm, offset=0)
        if int(self.header["magic"][0]) != DEPTH_BUFFER_MAGIC:
            self.close()
            return False

        height = int(self.header["height"][0])
        width = int(self.header["width"][0])
        if self.expected_height is not None and height != int(self.expected_height):
            raise ValueError(f"Depth buffer height mismatch: expected {self.expected_height}, got {height}")
        if self.expected_width is not None and width != int(self.expected_width):
            raise ValueError(f"Depth buffer width mismatch: expected {self.expected_width}, got {width}")

        self.frame = np.ndarray((height, width), dtype=np.float32, buffer=self._mm, offset=HEADER_DTYPE.itemsize)
        return True

    def read(self, *, retries: int = 3) -> DepthFrame | None:
        if not self._open_if_needed():
            return None
        if int(self.header["status"][0]) == 0:
            return None

        for _ in range(max(1, int(retries))):
            seq1 = int(self.header["seq"][0])
            if seq1 < 2 or seq1 % 2 == 1:
                time.sleep(0.001)
                continue

            timestamp_ns = int(self.header["timestamp_ns"][0])
            frame = self.frame.copy()
            seq2 = int(self.header["seq"][0])
            if seq1 == seq2 and seq2 % 2 == 0:
                return DepthFrame(seq=seq2 // 2, timestamp_ns=timestamp_ns, frame=frame)
        return None

    def close(self):
        if self._mm is not None:
            self._mm.close()
            self._mm = None
            self.header = None
            self.frame = None
