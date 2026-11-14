"""Immutable RGB-D snapshots and optical-frame camera calibration."""
from dataclasses import dataclass
import math

import numpy as np


def camera_intrinsics(width, height, fovy_deg):
    if (width < 1 or height < 1 or not math.isfinite(fovy_deg)
            or not 0 < fovy_deg < 180):
        raise ValueError("invalid perspective camera dimensions or fovy")
    focal = height / (2 * math.tan(math.radians(fovy_deg) / 2))
    # Integer u/v denote pixel centers; OpenGL viewport boundaries are half a
    # pixel beyond the first/last centers.
    return np.array([[focal, 0., (width - 1) / 2],
                     [0., focal, (height - 1) / 2], [0., 0., 1.]])


@dataclass(frozen=True)
class CameraSnapshot:
    observation_id: str
    sim_time_seconds: float
    rgb: np.ndarray
    depth_m: np.ndarray
    K: np.ndarray
    T_world_optical: np.ndarray
    near_m: float
    far_m: float

    def __post_init__(self):
        for name, dtype in (("rgb", np.uint8), ("depth_m", float),
                            ("K", float), ("T_world_optical", float)):
            original = np.asarray(getattr(self, name))
            if name == "rgb" and original.dtype != np.uint8:
                raise ValueError("snapshot rgb must be uint8")
            value = np.array(original, dtype=dtype, copy=True)
            value.setflags(write=False)
            object.__setattr__(self, name, value)
        if (not self.observation_id or not math.isfinite(self.sim_time_seconds)
                or self.depth_m.ndim != 2 or min(self.depth_m.shape) < 1
                or self.rgb.shape != (*self.depth_m.shape, 3)):
            raise ValueError("RGB and depth must be aligned HxW arrays with valid identity/time")
        if (self.K.shape != (3, 3) or not np.isfinite(self.K).all()
                or self.K[0, 0] <= 0 or self.K[1, 1] <= 0
                or not np.allclose(self.K[2], [0., 0., 1.])
                or not np.allclose([self.K[0, 1], self.K[1, 0]], 0.)):
            raise ValueError("K must be finite pinhole intrinsics with zero skew")
        transform = self.T_world_optical
        if (transform.shape != (4, 4) or not np.isfinite(transform).all()
                or not np.allclose(transform[3], [0, 0, 0, 1])
                or not np.allclose(transform[:3, :3].T @ transform[:3, :3], np.eye(3), atol=1e-6)
                or not np.isclose(np.linalg.det(transform[:3, :3]), 1., atol=1e-6)
                or not math.isfinite(self.near_m) or not math.isfinite(self.far_m)
                or not 0 < self.near_m < self.far_m):
            raise ValueError("invalid optical-to-world pose or clipping distances")


def backproject(snapshot):
    """Return world points and original flattened indices; invalid pixels omitted."""
    depth = snapshot.depth_m
    valid = (np.isfinite(depth) & (depth > snapshot.near_m)
             & (depth < snapshot.far_m * .99))
    indices = np.flatnonzero(valid)
    v, u = np.divmod(indices, depth.shape[1])
    z = depth.ravel()[indices]
    K = snapshot.K
    optical = np.column_stack(((u - K[0, 2]) * z / K[0, 0],
                               (v - K[1, 2]) * z / K[1, 1], z))
    T = snapshot.T_world_optical
    return optical @ T[:3, :3].T + T[:3, 3], indices
