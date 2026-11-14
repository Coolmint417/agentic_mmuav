"""Analytic RGB-D planes, projection conventions and native mask support."""
from dataclasses import replace
import threading
import unittest
from unittest.mock import patch

import numpy as np

from observations.camera import CameraSnapshot, backproject, camera_intrinsics
from observations.planes import PlaneDetectionConfig, detect_planes


def synthetic(angles=(45,), *, width=160, height=120, noise=False):
    K = camera_intrinsics(width, height, 90.)
    T = np.eye(4)
    T[:3, :3] = [[0, 0, 1], [-1, 0, 0], [0, -1, 0]]
    T[:3, 3] = [0., 0., 2.]
    v, u = np.indices((height, width))
    rays = np.stack(((u-K[0, 2])/K[0, 0], (v-K[1, 2])/K[1, 1], np.ones_like(u)), axis=-1)
    world_rays = rays @ T[:3, :3].T
    depth = np.full((height, width), np.inf)
    truths = []
    for i, angle in enumerate(angles):
        radians = np.deg2rad(angle)
        normal = np.array([-np.sin(radians), 0., np.cos(radians)])
        center = T[:3, 3] - 2 * normal
        offset = -normal @ center
        denominator = world_rays @ normal
        valid = (denominator < -1e-6) & (u >= i*width/len(angles)) & (u < (i+1)*width/len(angles))
        depth[valid] = -2 / denominator[valid]
        truths.append((normal, offset))
    if noise:
        rng = np.random.default_rng(123)
        finite = np.isfinite(depth)
        depth[finite] += rng.normal(0, .001, finite.sum())
        outliers = rng.random(depth.shape) < .08
        depth[outliers] = rng.uniform(.5, 12., outliers.sum())
    rgb = np.stack((u % 256, v % 256, np.full_like(u, 90)), axis=-1).astype(np.uint8)
    return CameraSnapshot("synthetic", 1., rgb, depth, K, T, .002, 16.), truths


CONFIG = PlaneDetectionConfig(min_inliers=40, min_mask_pixels=40, max_ransac_points=8000)


class PlaneTests(unittest.TestCase):
    def test_all_inclinations_and_camera_facing_normal(self):
        for angle in (0, 30, 45, 60, 90):
            with self.subTest(angle=angle):
                snapshot, truths = synthetic((angle,))
                detection = detect_planes(snapshot, CONFIG)
                self.assertGreaterEqual(len(detection.planes), 1)
                plane = max(detection.planes, key=lambda p: p.inlier_count)
                self.assertAlmostEqual(plane.inclination_deg, angle, places=5)
                self.assertGreater(plane.normal_world @ truths[0][0], .999999)
                self.assertGreater(plane.normal_world @ (snapshot.T_world_optical[:3, 3]-plane.centroid_world), 0)
                self.assertLess(abs(plane.offset_world_m - truths[0][1]), 1e-6)
                self.assertGreater(plane.observed_area_m2, 0.)
                self.assertTrue(all(v > 0 for v in plane.extent_uv_m))

    def test_noise_outliers_and_multiple_planes_unique_native_masks(self):
        snapshot, truths = synthetic((30, 60, 90), noise=True)
        result = detect_planes(snapshot, CONFIG)
        for normal, offset in truths:
            matching = min(result.planes, key=lambda p: np.linalg.norm(p.normal_world-normal))
            self.assertGreater(matching.normal_world @ normal, .999)
            self.assertLess(abs(matching.offset_world_m-offset), .02)
        labels = np.zeros(snapshot.depth_m.shape, int)
        for plane in result.planes:
            labels += plane.mask
            self.assertEqual(plane.mask.shape, snapshot.depth_m.shape)
            v, u = np.nonzero(plane.mask)
            self.assertEqual(plane.bbox_xyxy, (u.min(), v.min(), u.max(), v.max()))
            self.assertFalse(plane.mask.flags.writeable)
            self.assertTrue(np.isfinite(plane.boundary_uv).all())
        self.assertLessEqual(labels.max(), 1)
        self.assertLess(result.downsampled_points, result.valid_points)
        self.assertEqual([p.plane_id for p in result.planes], [f"P{i+1}" for i in range(len(result.planes))])
        self.assertEqual(result.annotated_rgb.shape, snapshot.rgb.shape)

    def test_invalid_background_empty_cloud_and_snapshot_validation(self):
        snapshot, _ = synthetic()
        depth = np.full_like(snapshot.depth_m, snapshot.far_m)
        depth[0, :4] = [np.nan, np.inf, 0, -1]
        empty = replace(snapshot, depth_m=depth)
        result = detect_planes(empty, CONFIG)
        self.assertEqual(result.planes, ())
        self.assertEqual(result.valid_points, 0)
        np.testing.assert_array_equal(result.annotated_rgb, snapshot.rgb)
        with self.assertRaises(ValueError):
            replace(snapshot, rgb=np.zeros((10, 10, 3), np.uint8))
        with self.assertRaises(ValueError):
            replace(snapshot, K=np.zeros((3, 3)))
        self.assertFalse(snapshot.depth_m.flags.writeable)
        # Missing optional dependency is an actionable error, not fake planes.
        with patch.dict("sys.modules", {"open3d": None}), self.assertRaisesRegex(RuntimeError, "requirements-planes"):
            detect_planes(snapshot)
        cancelled = threading.Event()
        cancelled.set()
        with self.assertRaises(InterruptedError):
            detect_planes(snapshot, cancel_event=cancelled)

    def test_z_depth_backprojection_and_intrinsics(self):
        snapshot, _ = synthetic((90,))
        points, indices = backproject(snapshot)
        np.testing.assert_allclose(points[:, 0], 2., atol=1e-10)
        # Off-center range exceeds Z, although the forward wall Z is constant.
        self.assertGreater(np.linalg.norm(points[0]-snapshot.T_world_optical[:3, 3]), snapshot.depth_m.flat[0])
        K = camera_intrinsics(640, 480, 90.)
        np.testing.assert_allclose(K, [[240, 0, 319.5], [0, 240, 239.5], [0, 0, 1]])
