"""Execution RGB histories: timing, multimodal transport and history isolation."""
import json
import unittest

import httpx

from agent.backends import DeepSeekConfig
from agent.deepseek import DeepSeekBackend
from agent.runner import AgentRunner, RunnerConfig
from agent.messages import attach_image
from env.mmuav_env import MMUAVEnv
from observations import MuJoCoObservationProvider, ObservationConfig
from observations.process import ProcessObservation
from tests.support import FakeJoystick
from tests.test_runner import ScriptedModel, running_system, wait_until


class ProcessObservationTests(unittest.TestCase):
    def test_history_trimming_keeps_images_on_the_correct_observation(self):
        calls = []

        def handler(request):
            calls.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{
                "message": {"content": '{"final":"已观测"}'}, "finish_reason": "stop"}]})

        backend = DeepSeekBackend(DeepSeekConfig(api_key="test-key"),
                                  transport=httpx.MockTransport(handler))
        with running_system(backend, config=RunnerConfig(history_messages=3)) as (
                runtime, runner, remote, events):
            for index in range(3):
                runner.submit(f"观察{index}")
                wait_until(lambda: not runner.get_status()["busy"])
            self.assertEqual(len(calls), 3)
            for index, call in enumerate(calls):
                groups = [m["content"] for m in call["messages"]
                          if isinstance(m["content"], list)]
                self.assertEqual(len(groups), 1 if index == 0 else 2)
                tasks = [json.loads(g[0]["text"])["task"] for g in groups]
                self.assertEqual(tasks, [f"观察{i}" for i in range(max(0, index - 1), index + 1)])
                self.assertTrue(all(sum(b["type"] == "image_url" for b in g) == 1 for g in groups))
                self.assertTrue(all(set(m) == {"role", "content"} for m in call["messages"]))

    def test_cancel_discards_process_images_before_next_task(self):
        model = ScriptedModel(['{"tool":"takeoff"}', '{"final":"新的观测"}'])
        with running_system(model, realtime=True, config=RunnerConfig(
                process_sample_interval=.1)) as (runtime, runner, remote, events):
            runner.submit("起飞")
            wait_until(lambda: runtime._invoke(lambda: runtime._active is not None
                       and len(runtime._active["process"].frames) >= 2))
            runner.cancel("test cancellation")
            wait_until(lambda: not runner.get_status()["busy"])
            self.assertIsNone(runtime._invoke(lambda: runtime._process_pending))
            runner.submit("观察当前位置")
            wait_until(lambda: not runner.get_status()["busy"])
            self.assertEqual(len(model.calls), 2)
            self.assertNotIn("extra_images", model.calls[-1])
            observation = json.loads(model.calls[-1]["messages"][-1]["content"])["observation"]
            self.assertNotIn("process", observation)

    def test_no_frame_cap_same_snapshot_and_no_process_depth(self):
        with MMUAVEnv(joystick_reader=FakeJoystick()) as env:
            with MuJoCoObservationProvider(env, ObservationConfig(include_depth=True)) as provider:
                process = ProcessObservation("test", 0.0, .1)
                for index in range(1, 13):
                    env.data.time = index / 10
                    env.data.qpos[0] = index / 100
                    env.data.qvel[:3] = [index, 2, 3]
                    env.data.qvel[3:6] = [.1, .2, .3]
                    before = env.data.qpos.copy()
                    process.sample(provider, env.data.time)
                    self.assertTrue((env.data.qpos == before).all())
                self.assertIsNone(provider._depth_renderer)
                env.data.time = 1.25
                process.end_time = 1.25
                packet = process.attach(provider.collect())
                self.assertEqual(len(packet.data["process"]["frames"]), 12)
                self.assertEqual(len(packet.images), 14)  # current RGB/depth + 12 process RGB
                for index, frame in enumerate(packet.data["process"]["frames"], start=1):
                    self.assertEqual(frame["time"], index / 10)
                    self.assertEqual(frame["position"][0], index / 100)
                    self.assertEqual(frame["velocity"], [index, 2, 3])
                    self.assertEqual(frame["angular_velocity"], [.1, .2, .3])
                history = []
                AgentRunner._append_observation(history, packet, task="test")
                runner = object.__new__(AgentRunner)
                runner.config = RunnerConfig(max_context_chars=2000)
                self.assertGreater(len(history[0]["content"]), 2000)
                self.assertEqual(runner._bounded(history, required_index=0), history)
                AgentRunner._append_observation(history, provider.collect(), task="next")
                old = json.loads(history[0]["content"])["observation"]
                self.assertNotIn("process", old)
                self.assertEqual(list(history[0]["_images"]), ["rgb", "depth"])

    def test_api_process_rgb_sensors_repairs_and_no_old_images_next_turn(self):
        calls = []
        replies = iter(['{"tool":"takeoff"}', 'bad json', '{"final":"完成"}',
                        '{"tool":"land"}', '{"final":"已降落"}'])

        def handler(request):
            calls.append(json.loads(request.content))
            return httpx.Response(200, json={"choices": [{
                "message": {"content": next(replies)}, "finish_reason": "stop"}]})

        backend = DeepSeekBackend(DeepSeekConfig(api_key="test-key"),
                                  transport=httpx.MockTransport(handler))
        with running_system(backend, config=RunnerConfig(
                process_sample_interval=.2, include_depth=True)) as (runtime, runner, remote, events):
            runner.submit("起飞")
            wait_until(lambda: not runner.get_status()["busy"])
            self.assertEqual(len(calls), 3)
            observation_messages = [m for m in calls[1]["messages"] if isinstance(m["content"], list)]
            self.assertEqual(len(observation_messages), 2)
            blocks = observation_messages[-1]["content"]
            observation = json.loads(blocks[0]["text"])["observation"]
            process = observation["process"]
            frames = process["frames"]
            self.assertGreater(len(frames), 0)
            self.assertEqual(process["sample_interval_seconds"], .2)
            self.assertEqual(process["tool"], "takeoff")
            times = [frame["time"] for frame in frames]
            self.assertEqual(times, sorted(set(times)))
            for frame in frames:
                self.assertGreater(frame["elapsed_seconds"], 0)
                self.assertLess(frame["time"], process["end_time_seconds"])
                self.assertEqual(len(frame["position"]), 3)
                self.assertEqual(len(frame["velocity"]), 3)
                self.assertEqual(len(frame["rpy"]), 3)
                self.assertEqual(len(frame["angular_velocity"]), 3)
                self.assertIn(frame["state"], {"on_ground", "hovering"})
                self.assertNotIn("depth", frame)
                self.assertEqual(observation[frame["rgb_attachment"]]["width"], 224)
            self.assertEqual(sum(b["type"] == "image_url" for b in blocks), 2 + len(frames))
            retry_blocks = [m["content"] for m in calls[2]["messages"]
                            if isinstance(m["content"], list)][-1]
            self.assertEqual(blocks, retry_blocks)
            self.assertIsInstance(calls[2]["messages"][-1]["content"], str)
            # HF marker order matches the same named image order.
            names = ["depth"] + [frame["rgb_attachment"] for frame in frames]
            rendered = attach_image([{"role": "user", "content": "observation"}], 0, names)
            self.assertEqual(rendered[0]["content"].count("[IMG]"), 2 + len(frames))

            runner.submit("降落")
            wait_until(lambda: not runner.get_status()["busy"])
            self.assertEqual(len(calls), 5)
            # The next task retains initial and post-takeoff RGB/depth pairs,
            # but none of takeoff's process images or sampled sensor data.
            groups = [m["content"] for m in calls[3]["messages"]
                      if isinstance(m["content"], list)]
            self.assertEqual(len(groups), 3)
            self.assertEqual(sum(b["type"] == "image_url" for g in groups for b in g), 6)
            old = [json.loads(g[0]["text"])["observation"] for g in groups[:-1]]
            self.assertTrue(all("process" not in o and "sample_count" not in o for o in old))
            self.assertTrue(all(not any(k.startswith("process_rgb_") for k in o) for o in old))
            retained = [b["image_url"]["url"] for b in groups[1] if b["type"] == "image_url"]
            original = [b["image_url"]["url"] for b in blocks if b["type"] == "image_url"][:2]
            self.assertEqual(retained, original)
            latest = [m for m in calls[4]["messages"] if isinstance(m["content"], list)][-1]
            data = json.loads(latest["content"][0]["text"])["observation"]
            self.assertEqual(data["process"]["tool"], "land")
            self.assertGreater(data["process"]["start_time_seconds"], process["end_time_seconds"])

    def test_default_settings_and_interval_validation(self):
        self.assertEqual(RunnerConfig().max_decisions, 32)
        self.assertEqual(RunnerConfig().history_messages, 24)
        self.assertEqual(RunnerConfig().process_sample_interval, 1.0)
        for value in (0, -1, True, float("inf"), float("nan"), "1"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                RunnerConfig(process_sample_interval=value)


if __name__ == "__main__":
    unittest.main()
