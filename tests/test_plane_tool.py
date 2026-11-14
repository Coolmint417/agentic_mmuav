"""Owner-thread RGB-D capture, worker isolation and actual multimodal HTTP JSON."""
from dataclasses import replace
import base64
from io import BytesIO
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import httpx
import mujoco
import numpy as np
from PIL import Image

from agent.backends import DeepSeekConfig
from agent.deepseek import DeepSeekBackend
from agent.protocol import parse_decision
from agent.runner import AgentRunner, RunnerConfig
from env.mmuav_env import MMUAVEnv
from env.flight_state_machine import FlightState
from observations import MuJoCoObservationProvider
from observations.camera import backproject
from observations.planes import detect_planes
from runtime import EnvironmentRuntime, RuntimeConfig
from tests.support import FakeJoystick
from tests.test_aggressive_perch import surface_fixture
from tests.test_deepseek import response
from tests.test_runner import OwnedEnvironment, ScriptedModel, wait_until
from tools.detect_planes import analyze_snapshot


class PlaneToolTests(unittest.TestCase):
    def test_mujoco_depth_and_optical_extrinsics_against_known_surfaces(self):
        for angle, yaw in ((45, 0.), (90, 0.), (60, .8)):
            with self.subTest(angle=angle, yaw=yaw), MMUAVEnv(joystick_reader=FakeJoystick()) as env:
                point, normal = surface_fixture(env, angle, yaw)
                forward = np.array([np.cos(yaw), np.sin(yaw), 0.])
                env.data.qpos[:3] = point - 2 * forward
                env.data.qpos[3:7] = [np.cos(yaw/2), 0., 0., np.sin(yaw/2)]
                env.data.qvel[:] = 0
                with MuJoCoObservationProvider(env) as observer:
                    before = (env.data.qpos.copy(), env.data.ctrl.copy(), env.data.time)
                    snapshot, packet = observer.capture_rgbd(640, 480)
                    self.assertEqual(snapshot.observation_id, packet.data["observation_id"])
                    self.assertEqual(snapshot.sim_time_seconds, packet.data["sim_time_seconds"])
                    self.assertEqual(packet.images["rgb"].size, (224, 168))
                    self.assertNotIn("depth", packet.images)
                    np.testing.assert_allclose(snapshot.T_world_optical[:3, 3], env.data.qpos[:3]+.3*forward)
                    if angle == 90:
                        np.testing.assert_allclose(snapshot.depth_m[220:260, 200:440], 1.7, atol=2e-5)
                    result = detect_planes(snapshot)
                    plane = max(result.planes, key=lambda p: p.inlier_count)
                    self.assertGreater(abs(plane.normal_world @ normal), .999)
                    self.assertLess(abs(plane.normal_world @ point + plane.offset_world_m), .005)
                    points, pixels = backproject(snapshot)
                    selected = plane.mask.ravel()[pixels]
                    self.assertLess(np.quantile(np.abs((points[selected]-point) @ normal), .95), .02)
                    np.testing.assert_array_equal(env.data.qpos, before[0])
                    np.testing.assert_array_equal(env.data.ctrl, before[1])
                    self.assertEqual(env.data.time, before[2])

    def test_deepseek_receives_real_overlay_and_snapshot_identity_without_movement(self):
        requests = []

        def handler(request):
            payload = json.loads(request.content)
            requests.append(payload)
            if len(requests) == 1:
                return response('{"tool":"detect_planes"}')
            return response('{"final":"已观测 P1"}')

        backend = DeepSeekBackend(DeepSeekConfig(api_key="test-key"), transport=httpx.MockTransport(handler))
        runtime = EnvironmentRuntime(RuntimeConfig(realtime=False, preview_enabled=False),
                                     environment_factory=lambda: OwnedEnvironment(joystick_reader=FakeJoystick()))
        runtime.start()
        runner = None
        with tempfile.TemporaryDirectory() as directory:
            try:
                runtime.wait_ready()
                runtime._plane_config = replace(runtime._plane_config, output_directory=directory)
                runner = AgentRunner(runtime, config=RunnerConfig(include_depth=False), backend_factory=lambda _: backend)
                runner.load_model()
                wait_until(lambda: not runner.get_status()["busy"])
                self.assertTrue(runner.submit("检测平面，告诉我 P1 对应哪里"))
                wait_until(lambda: not runner.get_status()["busy"], timeout=30)
                self.assertEqual(len(requests), 2)
                blocks = requests[1]["messages"][-1]["content"]
                feedback = json.loads(blocks[0]["text"])
                result, observation = feedback["tool_result"], feedback["observation"]
                self.assertEqual(result["status"], "success", result)
                self.assertEqual(result["observation_id"], observation["observation_id"])
                self.assertEqual(result["sim_time_seconds"], observation["sim_time_seconds"])
                self.assertEqual(observation["image_order"], ["rgb", "annotated_rgb"])
                self.assertTrue(any("annotated_rgb" in block.get("text", "") for block in blocks))
                images = [Image.open(BytesIO(base64.b64decode(b["image_url"]["url"].split(",", 1)[1])))
                          for b in blocks if b["type"] == "image_url"]
                self.assertEqual([i.size for i in images], [(224, 168), (224, 168)])
                self.assertFalse(np.array_equal(np.asarray(images[0]), np.asarray(images[1])))
                self.assertNotIn("process", observation)
                path = Path(result["artifacts"]["annotated_rgb"])
                with Image.open(path) as saved_image:
                    self.assertEqual(saved_image.size, (640, 480))
                saved = json.loads(Path(result["artifacts"]["json"]).read_text())
                self.assertEqual(saved["observation_id"], result["observation_id"])
                self.assertEqual(runtime.get_status()["flight_state"], "on_ground")
                self.assertEqual(runtime.get_status()["sucker_states"], [False]*4)
                self.assertIsNotNone(runtime._plane_cache)
            finally:
                if runner is not None:
                    runner.close()
                runtime.close()

    def test_cancellation_does_not_publish_late_geometry_and_physics_continues(self):
        entered, release = threading.Event(), threading.Event()
        worker_names = []

        def delayed(**kwargs):
            worker_names.append(threading.current_thread().name)
            entered.set()
            release.wait(10)
            return analyze_snapshot(**kwargs)

        runtime = EnvironmentRuntime(RuntimeConfig(realtime=False, preview_enabled=False),
                                     environment_factory=lambda: OwnedEnvironment(joystick_reader=FakeJoystick()))
        runtime.start()
        try:
            runtime.wait_ready()
            turn, _ = runtime.begin_turn(save_annotated_rgb=False)
            with patch("tools.detect_planes.analyze_snapshot", delayed):
                future = runtime.execute_tool({"tool": "detect_planes"}, turn)
                self.assertTrue(entered.wait(10))
                before = runtime.get_status()["sim_time"]
                wait_until(lambda: runtime.get_status()["sim_time"] > before+.1)
                runtime.cancel_turn("test cancellation")
                self.assertEqual(future.result(timeout=5)["status"], "terminated")
                release.set()
                runtime._invoke(lambda: None)
                self.assertIsNone(runtime._plane_pending)
                self.assertIsNone(runtime._plane_cache)
                self.assertTrue(worker_names[0].startswith("trajectory-planner"))
                self.assertEqual(runtime.get_status()["agent_state"], "end")
        finally:
            release.set()
            runtime.close()

    def test_save_disabled_and_protocol_validation(self):
        self.assertEqual(parse_decision('{"tool":"detect_planes"}'), {"tool": "detect_planes"})
        with self.assertRaises(ValueError):
            parse_decision('{"tool":"detect_planes","args":{}}')
        model = ScriptedModel(['{"tool":"detect_planes"}', '{"final":"完成"}'])
        from tests.test_runner import running_system
        with running_system(model, config=RunnerConfig(save_annotated_rgb=False)) as (runtime, runner, _, _):
            runner.submit("检测")
            wait_until(lambda: not runner.get_status()["busy"], timeout=30)
            feedback = json.loads(model.calls[-1]["messages"][-1]["content"])
            self.assertNotIn("artifacts", feedback["tool_result"])
            self.assertIn("annotated_rgb", model.calls[-1]["extra_images"])
