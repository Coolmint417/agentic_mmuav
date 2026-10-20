"""Native world-camera interaction never changes model or agent RGB cameras."""
import threading
import unittest
from unittest.mock import patch

import mujoco
import numpy as np

from env.mmuav_env import MMUAVEnv
from observations import MuJoCoObservationProvider, ObservationConfig
from runtime.preview import PreviewRenderer, WorldCamera
from tests.support import FakeJoystick
from tests.test_runner import ScriptedModel, running_system, wait_until


class CameraTests(unittest.TestCase):
    def test_world_pose_fov_reset_and_unchanged_fpv(self):
        with MMUAVEnv(joystick_reader=FakeJoystick(False)) as env:
            world = WorldCamera(env.model, env.data)
            initial = world.state()
            index = mujoco.mj_name2id(
                env.model, mujoco.mjtObj.mjOBJ_CAMERA, "world_camera"
            )
            np.testing.assert_allclose(
                initial["position"], env.data.cam_xpos[index], atol=1e-8
            )
            self.assertAlmostEqual(initial["fov"], env.model.cam_fovy[index])
            before = env.data.qpos.copy()
            camera_fov = env.model.cam_fovy.copy()
            global_fov = env.model.vis.global_.fovy
            provider = MuJoCoObservationProvider(env)
            preview = PreviewRenderer(env, world, 320, 240)
            try:
                # The first GL frame can differ slightly during renderer warmup.
                provider.collect()
                fpv = np.asarray(provider.collect().images["rgb"])
                fpv_eye = provider._renderer.scene.camera[0].pos.copy()
                fpv_forward = provider._renderer.scene.camera[0].forward.copy()
                preview.render("world_camera")
                original_image = np.asarray(preview.render("world_camera"))
                world.move("rotate", 0.08, -0.04)
                self.assertNotEqual(world.state()["azimuth"], initial["azimuth"])
                self.assertNotEqual(world.state()["elevation"], initial["elevation"])
                previous = world.state()
                world.move("move", 0.03, 0.02)
                self.assertNotEqual(world.state()["lookat"], previous["lookat"])
                world.move("zoom", 0, -0.1)
                self.assertNotEqual(world.state()["distance"], initial["distance"])
                world.set_fov(85)
                self.assertFalse(
                    np.array_equal(preview.render("world_camera"), original_image)
                )
                # Offscreen GL dithering may differ by one intensity level.
                np.testing.assert_allclose(
                    np.asarray(provider.collect().images["rgb"]), fpv, atol=1
                )
                np.testing.assert_array_equal(
                    provider._renderer.scene.camera[0].pos, fpv_eye
                )
                np.testing.assert_array_equal(
                    provider._renderer.scene.camera[0].forward, fpv_forward
                )
                np.testing.assert_array_equal(env.data.qpos, before)
                np.testing.assert_array_equal(env.model.cam_fovy, camera_fov)
                self.assertEqual(env.model.vis.global_.fovy, global_fov)
                world.reset()
                self.assertEqual(world.state(), initial)
                # Shadow rasterization can vary by a few intensity levels after a view change.
                np.testing.assert_allclose(
                    np.asarray(preview.render("world_camera")), original_image, atol=3
                )
                world.set_fov(90)
                wide = preview.render("world_camera")
                self.assertFalse(np.array_equal(wide, original_image))
                for camera in preview._renderer.scene.camera:
                    self.assertAlmostEqual(
                        camera.frustum_top / camera.frustum_near, 1, places=5
                    )
                with self.assertRaises(ValueError):
                    world.set_fov(float("nan"))
                with self.assertRaises(ValueError):
                    world.move("invalid", 0, 0)
            finally:
                preview.close()
                provider.close()

    def test_owner_thread_camera_switch_persistence_and_reset(self):
        original_move = WorldCamera.move
        owners = []

        def move(camera, *args):
            owners.append(threading.current_thread().name)
            return original_move(camera, *args)

        with patch.object(WorldCamera, "move", move):
            with running_system(ScriptedModel([]), preview=True) as (
                runtime,
                runner,
                remote,
                events,
            ):
                initial = runtime.get_status()["world_camera"]
                runtime.move_world_camera("rotate", 0.1, 0.1).result(timeout=5)
                self.assertEqual(
                    runtime.get_status()["world_camera"], initial
                )  # FPV ignores mouse moves
                runtime.set_camera("world_camera").result(timeout=5)
                wait_until(lambda: runtime.get_preview() is not None)
                runtime.move_world_camera("rotate", 0.1, 0.1).result(timeout=5)
                runtime.set_world_camera_fov(75).result(timeout=5)
                changed = runtime.get_status()["world_camera"]
                self.assertNotEqual(changed, initial)
                self.assertEqual(changed["fov"], 75)
                self.assertEqual(owners, ["environment-runtime"])
                runtime.set_camera("fpv_camera").result(timeout=5)
                runtime.set_camera("world_camera").result(timeout=5)
                self.assertEqual(runtime.get_status()["world_camera"], changed)
                runtime.reset_world_camera().result(timeout=5)
                self.assertEqual(runtime.get_status()["world_camera"], initial)
                with self.assertRaises(ValueError):
                    runtime.set_world_camera_fov(180).result(timeout=5)
                self.assertTrue(runtime.get_status()["ready"])
