"""Depth rendering, metric encoding, optional delivery and model image binding."""
import base64
from io import BytesIO
import json
import unittest

import httpx
import mujoco
import numpy as np
from PIL import Image

from agent.backends import DeepSeekConfig
from agent.deepseek import DeepSeekBackend
from agent.messages import attach_image
from agent.runner import RunnerConfig
from env.mmuav_env import MMUAVEnv
from observations import MuJoCoObservationProvider, ObservationConfig
from observations.depth import depth_image
from runtime.preview import PreviewRenderer, WorldCamera
from tests.support import FakeJoystick
from tests.test_runner import ScriptedModel, running_system, wait_until


class DepthTests(unittest.TestCase):
    def test_fixed_metric_encoding(self):
        depths = np.array([[0, 5, 10, 20, np.inf, np.nan]], dtype=float)
        np.testing.assert_array_equal(
            np.array(depth_image(depths))[0, :, 0], [255, 128, 0, 0, 0, 0]
        )

    def test_provider_depth_matches_mujoco_without_stepping(self):
        with MMUAVEnv(joystick_reader=FakeJoystick(False)) as env:
            env.data.qpos[:3] = [-2.0, 2.0, 1.0]
            mujoco.mj_fwdPosition(env.model, env.data)
            before = env.data.qpos.copy()
            config = ObservationConfig(include_depth=True)
            with MuJoCoObservationProvider(env, config) as provider:
                packet = provider.collect()
                self.assertEqual(list(packet.images), ["rgb", "depth"])
                self.assertEqual(packet.images["rgb"].size, (224, 224))
                self.assertEqual(packet.images["depth"].size, (56, 56))
                self.assertEqual(packet.data["depth"]["unit"], "m")
                self.assertEqual(len(packet.data["depth"]["sector_min_m"]), 3)
                self.assertGreater(packet.data["depth"]["minimum_visible_m"], 0)
                with mujoco.Renderer(env.model, height=56, width=56) as renderer:
                    renderer.enable_depth_rendering()
                    renderer.update_scene(provider._snapshot, camera="fpv_camera")
                    expected = depth_image(renderer.render())
                    np.testing.assert_array_equal(packet.images["depth"], expected)
                self.assertEqual(env.data.time, 0.0)
                np.testing.assert_array_equal(env.data.qpos, before)
                inputs = packet.to_model_inputs()
                self.assertEqual(inputs["extra_images"]["depth"].size, (56, 56))
                inputs["extra_images"]["depth"].putpixel((0, 0), (1, 2, 3))
                self.assertNotEqual(packet.images["depth"].getpixel((0, 0)), (1, 2, 3))
            with MuJoCoObservationProvider(env) as provider:
                self.assertNotIn("depth", provider.collect().images)
            with MuJoCoObservationProvider(
                env, ObservationConfig(include_depth=True, depth_size=80)
            ) as provider:
                self.assertEqual(provider.collect().images["depth"].size, (80, 80))

    def test_runtime_depth_preview_is_independent_of_model_switch(self):
        with running_system(ScriptedModel([]), preview=True) as (
            runtime,
            runner,
            remote,
            events,
        ):
            runtime.set_camera("depth_camera").result(timeout=5)
            wait_until(lambda: runtime.get_preview() is not None)
            self.assertEqual(runtime.get_preview().size, (1280, 960))
            turn, packet = runtime.begin_turn(include_depth=False)
            self.assertNotIn("depth", packet.images)
            runtime.finish_turn(turn)
            turn, packet = runtime.begin_turn(include_depth=True, depth_size=72)
            self.assertEqual(packet.images["depth"].size, (72, 72))
            runtime.finish_turn(turn)
            runtime.set_camera("fpv_camera").result(timeout=5)
            wait_until(lambda: runtime.get_preview() is not None)
            self.assertTrue(runtime.get_status()["ready"])

    def test_deepseek_rgb_depth_repair_and_turn_settings(self):
        calls = []

        def handler(request):
            calls.append(json.loads(request.content))
            text = "bad json" if len(calls) == 1 else '{"final":"完成"}'
            return httpx.Response(
                200,
                json={
                    "choices": [{"message": {"content": text}, "finish_reason": "stop"}]
                },
            )

        model = DeepSeekBackend(
            DeepSeekConfig(api_key="test-key"), transport=httpx.MockTransport(handler)
        )
        with running_system(model, config=RunnerConfig(include_depth=True)) as (
            runtime,
            runner,
            remote,
            events,
        ):
            runner.submit("观察")
            wait_until(lambda: not runner.get_status()["busy"])
            self.assertEqual(len(calls), 2)
            for call in calls:
                images = []
                for message in call["messages"]:
                    if isinstance(message["content"], list):
                        for block in message["content"]:
                            if block["type"] == "image_url":
                                images.append(
                                    Image.open(
                                        BytesIO(
                                            base64.b64decode(
                                                block["image_url"]["url"].split(",", 1)[
                                                    1
                                                ]
                                            )
                                        )
                                    ).size
                                )
                self.assertEqual(images, [(224, 224), (56, 56)])
            self.assertIsInstance(calls[1]["messages"][-1]["content"], str)
            self.assertIn("请修正", calls[1]["messages"][-1]["content"])
            self.assertTrue(
                any(
                    e.kind == "agent_log"
                    and e.data["kind"] == "model_input"
                    and e.data["data"]["extra_image_sizes"] == {"depth": [56, 56]}
                    for e in events
                )
            )
            runner.configure(RunnerConfig(include_depth=False))
            runner.submit("只用 RGB")
            wait_until(lambda: not runner.get_status()["busy"])
            image_count = sum(
                block["type"] == "image_url"
                for msg in calls[-1]["messages"]
                if isinstance(msg["content"], list)
                for block in msg["content"]
            )
            self.assertEqual(image_count, 1)

    def test_local_backend_image_markers(self):
        messages = [
            {"role": "user", "content": "observation"},
            {"role": "assistant", "content": "invalid"},
            {"role": "user", "content": "repair"},
        ]
        result = attach_image(messages, 0, ("depth",))
        self.assertEqual(result[0]["content"].count("[IMG]"), 2)
        self.assertIn("depth:", result[0]["content"])
        self.assertEqual(result[-1]["content"], "repair")
        self.assertNotIn("[IMG]", messages[0]["content"])
        for size in (0, 4, 1025, True):
            with self.assertRaises(ValueError):
                RunnerConfig(depth_size=size)
