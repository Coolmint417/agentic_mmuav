"""Single-frame multi-plane RANSAC/PCA; no simulator access or plane memory."""
from dataclasses import dataclass
import math
import time

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from .camera import CameraSnapshot, backproject


@dataclass(frozen=True)
class PlaneDetectionConfig:
    max_planes: int = 5
    voxel_size_m: float = .03
    distance_threshold_m: float = .015
    mask_threshold_m: float = .02
    ransac_iterations: int = 500
    min_inliers: int = 80
    min_support_fraction: float = .01
    min_mask_pixels: int = 100
    max_ransac_points: int = 30000
    support_margin_m: float = .06
    overlay_alpha: float = .28
    random_seed: int = 0

    def __post_init__(self):
        for name, value in vars(self).items():
            if name in {"max_planes", "ransac_iterations", "min_inliers", "min_mask_pixels",
                        "max_ransac_points", "random_seed"}:
                if type(value) is not int or value < (0 if name == "random_seed" else 1):
                    raise ValueError(f"invalid {name}")
            elif isinstance(value, bool) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"invalid {name}")
        if self.min_support_fraction >= 1 or self.overlay_alpha >= 1 or self.min_inliers < 3:
            raise ValueError("invalid plane support or alpha configuration")


def _check_cancelled(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError("plane detection cancelled")


def _fit(points, camera_position):
    center = points.mean(axis=0)
    centered = points - center
    values, vectors = np.linalg.eigh(centered.T @ centered / len(points))
    if values[1] < 1e-10:
        raise ValueError("degenerate plane support")
    normal = vectors[:, 0]
    distance = float(np.dot(normal, camera_position - center))
    if abs(distance) < 1e-7:
        raise ValueError("camera lies on fitted plane; orientation is ambiguous")
    if distance < 0:
        normal = -normal
    offset = -float(np.dot(normal, center))
    axis = np.eye(3)[np.argmin(np.abs(normal))]
    tangent1 = np.cross(normal, axis)
    tangent1 /= np.linalg.norm(tangent1)
    tangent2 = np.cross(normal, tangent1)
    basis = np.column_stack((tangent1, tangent2))
    return normal, offset, center, basis


@dataclass(frozen=True)
class DetectedPlane:
    plane_id: str
    normal_world: np.ndarray
    offset_world_m: float
    centroid_world: np.ndarray
    distance_to_camera_m: float
    inclination_deg: float
    inlier_count: int
    inlier_ratio: float
    fit_rmse_m: float
    observed_area_m2: float
    extent_uv_m: tuple
    bbox_xyxy: tuple
    mask: np.ndarray
    boundary_uv: np.ndarray
    basis_world: np.ndarray

    def __post_init__(self):
        for name in ("normal_world", "centroid_world", "mask", "boundary_uv", "basis_world"):
            array = np.array(getattr(self, name), copy=True)
            array.setflags(write=False)
            object.__setattr__(self, name, array)

    def to_dict(self, observation_id):
        return {
            "plane_id": self.plane_id, "normal_world": self.normal_world.tolist(),
            "offset_world_m": self.offset_world_m, "centroid_world_m": self.centroid_world.tolist(),
            "distance_to_camera_m": self.distance_to_camera_m,
            "inclination_deg": self.inclination_deg, "inlier_count": self.inlier_count,
            "inlier_ratio": self.inlier_ratio, "fit_rmse_m": self.fit_rmse_m,
            "observed_area_m2": self.observed_area_m2, "extent_uv_m": list(self.extent_uv_m),
            "tangent_axes_world": self.basis_world.T.tolist(),
            "bbox_xyxy": list(self.bbox_xyxy), "mask_id": f"{observation_id}_{self.plane_id}",
        }


@dataclass(frozen=True)
class PlaneDetection:
    snapshot: CameraSnapshot
    planes: tuple
    annotated_rgb: np.ndarray
    timings_seconds: dict
    valid_points: int
    downsampled_points: int

    def __post_init__(self):
        image = np.array(self.annotated_rgb, copy=True)
        image.setflags(write=False)
        object.__setattr__(self, "annotated_rgb", image)

    def to_dict(self):
        s = self.snapshot
        height, width = s.depth_m.shape
        return {
            "tool": "detect_planes", "status": "success", "observation_id": s.observation_id,
            "sim_time_seconds": s.sim_time_seconds, "coordinate_frame": "world_nwu",
            "normal_convention": "unit_normal_points_toward_camera",
            "plane_equation": "dot(normal_world, world_point_m) + offset_world_m = 0",
            "distance_convention": "perpendicular distance from camera to infinite plane, not ray depth",
            "support_convention": "observed single-frame convex hull; holes/occlusions may be filled; not adhesion clearance",
            "source_image_size": [width, height], "bbox_convention": "source-image pixel centers; xyxy inclusive",
            "planes": [p.to_dict(s.observation_id) for p in self.planes],
            "quality": {"valid_points": self.valid_points, "downsampled_points": self.downsampled_points,
                        "reason": "planes_detected" if self.planes else "insufficient_planar_support"},
            "timings_seconds": self.timings_seconds,
            "visualization": {"kind": "annotated_rgb", "attachment": "annotated_rgb",
                              "image_id": f"{s.observation_id}_plane_overlay",
                              "attached_to_agent_message": True},
        }


def annotate(rgb, planes, alpha):
    from scipy.ndimage import distance_transform_edt
    colors = ((239, 83, 80), (38, 166, 154), (66, 133, 244),
              (255, 193, 7), (171, 71, 188), (255, 112, 67))
    output = rgb.astype(float).copy()
    edges = []
    for i, plane in enumerate(planes):
        mask = plane.mask
        color = np.array(colors[i % len(colors)])
        output[mask] = (1-alpha) * output[mask] + alpha * color
        interior = mask.copy()
        interior[1:] &= mask[:-1]
        interior[:-1] &= mask[1:]
        interior[:, 1:] &= mask[:, :-1]
        interior[:, :-1] &= mask[:, 1:]
        interior[[0, -1], :] = False
        interior[:, [0, -1]] = False
        edges.append((mask & ~interior, color))
    for edge, color in edges:
        output[edge] = color
    image = Image.fromarray(np.rint(output).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    font_size = max(24, min(image.size) // 12)
    font = ImageFont.load_default(size=font_size)
    for plane in planes:
        v, u = np.nonzero(plane.mask)
        # Stay in the mask interior, not at the edge of an occlusion/hole near
        # the centroid. Thin supports use their deepest available mask pixel.
        distance = distance_transform_edt(plane.mask)[v, u]
        inset = distance >= min(font_size * .8, float(distance.max()))
        score = (u-u.mean())**2 + (v-v.mean())**2
        score[~inset] = np.inf
        index = np.argmin(score)
        draw.text((int(u[index]), int(v[index])), plane.plane_id, font=font,
                  anchor="mm", fill="white", stroke_width=2, stroke_fill="black")
    return np.asarray(image)


def detect_planes(snapshot, config=None, cancel_event=None):
    """Detect every plane orientation, returning masks at native RGB-D resolution."""
    if not isinstance(snapshot, CameraSnapshot):
        raise TypeError("detect_planes requires CameraSnapshot")
    config = config or PlaneDetectionConfig()
    started = time.perf_counter()
    timings = {name: 0. for name in ("backprojection", "downsampling", "ransac", "pca", "mask", "boundary", "annotation")}
    _check_cancelled(cancel_event)
    try:
        import open3d as o3d
        from scipy.spatial import ConvexHull, QhullError
    except ImportError as exc:
        raise RuntimeError("detect_planes requires Open3D and SciPy: python -m pip install -r requirements-planes.txt") from exc
    tick = time.perf_counter()
    points, pixel_indices = backproject(snapshot)
    timings["backprojection"] = time.perf_counter() - tick
    camera = snapshot.T_world_optical[:3, 3]
    tick = time.perf_counter()
    cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(points))
    down = np.asarray(cloud.voxel_down_sample(config.voxel_size_m).points).copy() if len(points) else points
    rng = np.random.default_rng(config.random_seed)
    if len(down) > config.max_ransac_points:
        down = down[rng.choice(len(down), config.max_ransac_points, replace=False)]
    timings["downsampling"] = time.perf_counter() - tick
    o3d.utility.random.seed(config.random_seed)
    working = down
    candidates = []
    required = max(config.min_inliers, math.ceil(config.min_support_fraction * len(down)))
    for _ in range(config.max_planes):
        _check_cancelled(cancel_event)
        if len(working) < required:
            break
        tick = time.perf_counter()
        cloud = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(working))
        _, ids = cloud.segment_plane(distance_threshold=config.distance_threshold_m,
                                    ransac_n=3, num_iterations=config.ransac_iterations, probability=.999)
        timings["ransac"] += time.perf_counter() - tick
        if len(ids) < required:
            break
        support = working[ids]
        remaining = np.ones(len(working), dtype=bool)
        remaining[ids] = False
        working = working[remaining]
        tick = time.perf_counter()
        try:
            normal, offset, center, basis = _fit(support, camera)
        except ValueError:
            continue
        local = (support-center) @ basis
        z = (support - camera) @ snapshot.T_world_optical[:3, 2]
        candidates.append((normal, offset, center, basis,
                           local.min(axis=0)-config.support_margin_m,
                           local.max(axis=0)+config.support_margin_m,
                           z.min()-.1, z.max()+.1))
        timings["pca"] += time.perf_counter() - tick
    _check_cancelled(cancel_event)
    tick = time.perf_counter()
    labels = np.full(len(points), -1, dtype=np.int16)
    best = np.full(len(points), np.inf)
    depth_z = snapshot.depth_m.ravel()[pixel_indices]
    for index, (n, d, center, basis, lo, hi, zlo, zhi) in enumerate(candidates):
        _check_cancelled(cancel_event)
        residual = np.abs(points @ n + d) / config.mask_threshold_m
        local = (points-center) @ basis
        mask = ((residual < 1) & (residual < best)
                & np.all((local >= lo) & (local <= hi), axis=1)
                & (depth_z >= zlo) & (depth_z <= zhi))
        labels[mask], best[mask] = index, residual[mask]
    timings["mask"] = time.perf_counter() - tick
    planes = []
    for index, _ in enumerate(candidates):
        _check_cancelled(cancel_event)
        ids = np.flatnonzero(labels == index)
        if len(ids) < config.min_mask_pixels:
            continue
        tick = time.perf_counter()
        support = points[ids]
        try:
            n, d, center, basis = _fit(support, camera)
        except ValueError:
            continue
        timings["pca"] += time.perf_counter() - tick
        tick = time.perf_counter()
        local = (support-center) @ basis
        try:
            hull = ConvexHull(local)
        except QhullError:
            continue
        mask = np.zeros(snapshot.depth_m.size, dtype=bool)
        mask[pixel_indices[ids]] = True
        mask = mask.reshape(snapshot.depth_m.shape)
        v, u = np.nonzero(mask)
        plane = DetectedPlane(
            f"P{len(planes)+1}", n, d, center, abs(float(np.dot(n, camera)+d)),
            math.degrees(math.acos(float(np.clip(abs(n[2]), 0., 1.)))),
            len(ids), len(ids)/max(1, len(points)), float(np.sqrt(np.mean((support @ n+d)**2))),
            float(hull.volume), tuple(np.ptp(local, axis=0).tolist()),
            (int(u.min()), int(v.min()), int(u.max()), int(v.max())), mask,
            local[hull.vertices], basis)
        planes.append(plane)
        timings["boundary"] += time.perf_counter() - tick
    _check_cancelled(cancel_event)
    tick = time.perf_counter()
    annotated = annotate(snapshot.rgb, planes, config.overlay_alpha)
    _check_cancelled(cancel_event)
    timings["annotation"] = time.perf_counter() - tick
    timings["total"] = time.perf_counter() - started
    return PlaneDetection(snapshot, tuple(planes), annotated, timings, len(points), len(down))
