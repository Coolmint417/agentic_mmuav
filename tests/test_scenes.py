"""Real scene compilation, rendering, reset and runner history isolation."""
import unittest
from unittest.mock import patch

import mujoco
import numpy as np
from env.scenes import SCENES, load_scene_model
from runtime import RuntimeConfig
from tests.test_runner import ScriptedModel, running_system, wait_until


class SceneTests(unittest.TestCase):
    def test_perch_panels_have_requested_surface_centers_and_normals(self):
        model = load_scene_model("map6")
        data = mujoco.MjData(model)
        mujoco.mj_fwdPosition(model, data)
        cases = (
            (45, 0, [3, 2, 2]),
            (60, np.pi / 2, [-2, 7, 2]),
            (90, -3 * np.pi / 4, [-6, -2, 2]),
        )
        for angle, yaw, center in cases:
            geom = model.geom(f"map6_surface{angle}")
            rotation = data.geom_xmat[geom.id].reshape(3, 3)
            normal = rotation[:, 2]
            tilt = np.deg2rad(angle)
            expected = [-np.sin(tilt) * np.cos(yaw),
                        -np.sin(tilt) * np.sin(yaw), np.cos(tilt)]
            np.testing.assert_allclose(normal, expected, atol=1e-9)
            np.testing.assert_allclose(
                data.geom_xpos[geom.id] + geom.size[2] * normal, center, atol=1e-9)
            self.assertTrue(geom.contype[0] and geom.conaffinity[0])
            # Every corner is above the floor, including the rear face.
            lowest_z = data.geom_xpos[geom.id, 2] - np.dot(
                np.abs(rotation[2]), geom.size)
            self.assertGreater(lowest_z, 0.4)

    def test_all_maps_compile_with_clear_spawn_and_distinct_layouts(self):
        layouts = set()
        for scene in SCENES:
            with self.subTest(scene=scene.id):
                model = load_scene_model(scene.id)
                data = mujoco.MjData(model)
                mujoco.mj_forward(model, data)
                self.assertGreaterEqual(model.geom("floor").id, 0)
                self.assertGreaterEqual(model.camera("fpv_camera").id, 0)
                self.assertGreaterEqual(model.camera("world_camera").id, 0)
                # Initial craft overlaps only its supporting floor, no map obstacles.
                for contact in data.contact:
                    names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, i)
                             for i in contact.geom]
                    self.assertFalse(any(name and name.startswith("map") for name in names))
                layouts.add(model.geom_pos.tobytes())
        self.assertEqual(len(layouts), len(SCENES))

    def test_reload_renders_every_map_and_invalid_map_preserves_environment(self):
        model = ScriptedModel(['{"final":"旧地图完成"}', '{"final":"新地图完成"}'])
        config = RuntimeConfig(realtime=True, preview_enabled=True,
                               preview_width=320, preview_height=240)
        with running_system(model, runtime_config=config) as (runtime, runner, _, _):
            runner.submit("旧地图任务")
            wait_until(lambda: not runner.get_status()["busy"])
            old_turn = runtime._invoke(lambda: runtime._env.agent_state_machine.turn_id)
            runtime._invoke(lambda: setattr(runtime._env.data, "time", 12.0))
            previous_observer = runtime._observer
            for scene in SCENES[1:]:
                self.assertTrue(runner.reload_environment(scene.id))
                wait_until(lambda: not runner.get_status()["busy"])
                status = runtime.get_status()
                self.assertEqual(status["scene_id"], scene.id)
                self.assertEqual(status["agent_state"], "end")
                self.assertEqual(status["flight_state"], "on_ground")
                self.assertTrue(status["authorized"])
                self.assertLess(status["sim_time"], 0.2)
                self.assertEqual(runner._conversation, [])
                self.assertIs(runner._backend, model)
                self.assertTrue(previous_observer._closed)
                previous_observer = runtime._observer
                for camera in ("fpv_camera", "world_camera", "depth_camera"):
                    runtime.set_camera(camera).result(timeout=10)
                    runtime._invoke(runtime._render_preview)
                    self.assertEqual(runtime.get_preview().size, (320, 240))
                packet = runtime._invoke(runtime._observer.collect)
                self.assertEqual(packet.data["type"], "uav_observation")
                self.assertEqual(packet.images["rgb"].size, (224, 224))
            old_model = runtime._env.model
            with patch("runtime.environment.load_scene_model", side_effect=ValueError("bad XML")):
                with self.assertRaisesRegex(ValueError, "bad XML"):
                    runtime.reload_scene("map1")
            self.assertIs(runtime._env.model, old_model)
            self.assertFalse(runtime._observer._closed)
            self.assertTrue(runtime.get_status()["ready"])
            runner.submit("新地图任务")
            wait_until(lambda: not runner.get_status()["busy"])
            new_turn = runtime._invoke(lambda: runtime._env.agent_state_machine.turn_id)
            self.assertGreater(new_turn, old_turn)
            self.assertNotIn("旧地图任务", str(model.calls[-1]["messages"]))

    def test_reload_rejected_during_active_turn(self):
        with running_system(ScriptedModel([]), realtime=True) as (runtime, _, _, _):
            turn_id, _ = runtime.begin_turn()
            with self.assertRaisesRegex(RuntimeError, "终止当前任务"):
                runtime.reload_scene("map2")
            self.assertEqual(runtime.get_status()["scene_id"], "map1")
            runtime.validate_turn(turn_id)
