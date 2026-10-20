"""Real environment runtime + scripted model integration, without loading GPU weights."""
from contextlib import contextmanager
import json
import threading
import time
import unittest
from unittest.mock import patch

from agent.protocol import parse_decision
from agent.runner import AgentRunner, RunnerConfig
from env.mmuav_env import MMUAVEnv
from runtime import EnvironmentRuntime, RuntimeConfig
from runtime.events import EventBus
from tests.support import FakeJoystick


def wait_until(condition, timeout=15):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if condition():
            return
        time.sleep(0.01)
    raise AssertionError("Timed out waiting for background operation")


class ScriptedModel:
    def __init__(self, answers):
        self.answers = iter(answers)
        self.calls = []
        self.entered = threading.Event()
        self.wait_for_cancel = False
        self.unloaded = False

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        self.entered.set()
        if self.wait_for_cancel:
            if not kwargs["cancel_event"].wait(10):
                raise RuntimeError("Cancellation never reached the model")
        answer = next(self.answers)
        if isinstance(answer, Exception):
            raise answer
        return {"text": answer}

    def unload(self):
        self.unloaded = True


class OwnedEnvironment(MMUAVEnv):
    def step(self):
        if threading.current_thread().name != "environment-runtime":
            raise AssertionError("Physics stepped outside the owner thread")
        return super().step()


@contextmanager
def running_system(
    model, *, authorized=True, realtime=False, preview=False, config=None
):
    events, received = EventBus(), []
    events.subscribe(received.append)
    remote = FakeJoystick(authorized)
    runtime = EnvironmentRuntime(
        RuntimeConfig(realtime=realtime, preview_enabled=preview),
        events=events,
        environment_factory=lambda: OwnedEnvironment(joystick_reader=remote),
    )
    runtime.start()
    runtime.wait_ready()
    runner = AgentRunner(
        runtime, events=events, config=config, backend_factory=lambda path: model
    )
    try:
        runner.load_model()
        wait_until(lambda: not runner.get_status()["busy"])
        yield runtime, runner, remote, received
    finally:
        runner.close()
        runtime.close()


class ProtocolTests(unittest.TestCase):
    def test_valid_and_invalid_decisions(self):
        for value in (
            '{"tool":"takeoff"}',
            '{"tool":"land"}',
            '```json\n{"final":"完成"}\n```',
        ):
            self.assertIsInstance(parse_decision(value), dict)
        for value in (
            "[]",
            "{}",
            '{"tool":"other"}',
            '{"final":""}',
            '{"tool":"takeoff","height":1}',
            '{"tool":"fly_waypoint","waypoint":[]}',
            '{"final":"x","thought":"x"}',
        ):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_decision(value)


class RunnerIntegrationTests(unittest.TestCase):
    def test_json_repairs_preserve_task_observation_and_image_binding(self):
        from agent.messages import attach_image

        model = ScriptedModel(
            ['{"final":', '{"final":{"message":"完成"}}', '{"final":"完成"}']
        )
        with running_system(model, config=RunnerConfig(history_messages=1)) as (
            runtime,
            runner,
            remote,
            events,
        ):
            runner.submit("向前飞1m")
            wait_until(lambda: not runner.get_status()["busy"])
            self.assertEqual(len(model.calls), 3)
            for call in model.calls:
                index = call["image_message_index"]
                observation = json.loads(call["messages"][index]["content"])
                self.assertEqual(observation["task"], "向前飞1m")
                self.assertIn("observation", observation)
                self.assertEqual(call["image"].size, (224, 224))
                rendered = attach_image(call["messages"], index)
                self.assertTrue(rendered[index]["content"].startswith("[IMG]"))
                if call is not model.calls[0]:
                    self.assertNotIn("[IMG]", rendered[-1]["content"])
                    self.assertIn(
                        "请修正上一条回复的 JSON 格式，保持原任务和判断不变", rendered[-1]["content"]
                    )
            self.assertIn("JSONDecodeError", model.calls[1]["messages"][-1]["content"])
            self.assertIn("line 1 column", model.calls[1]["messages"][-1]["content"])
            self.assertIn(
                "final must be a nonempty string",
                model.calls[2]["messages"][-1]["content"],
            )
            self.assertEqual(runtime.get_status()["agent_state"], "end")

    def test_settings_apply_to_model_limits_and_log_snapshots(self):
        model = ScriptedModel(["bad JSON", '{"tool":"land"}', '{"final":"done"}'])
        with running_system(model) as (runtime, runner, remote, events):
            runner.configure(
                RunnerConfig(
                    system_prompt="Custom system", max_decisions=2, max_new_tokens=42
                )
            )
            runner.submit("降落")
            wait_until(lambda: not runner.get_status()["busy"])
            self.assertEqual(len(model.calls), 2)
            self.assertEqual(model.calls[0]["messages"][0]["content"], "Custom system")
            self.assertEqual(model.calls[0]["max_new_tokens"], 42)
            logs = [event.data for event in events if event.kind == "agent_log"]
            self.assertEqual(
                [r["data"]["raw"] for r in logs if r["kind"] == "model_output"],
                ["bad JSON", '{"tool":"land"}'],
            )
            self.assertTrue(any(r["kind"] == "parse_error" for r in logs))
            self.assertTrue(any(r["kind"] == "tool_result" for r in logs))
            first_input = next(
                r["data"]["messages"] for r in logs if r["kind"] == "model_input"
            )
            self.assertEqual(len(first_input), 2)  # no later history mutations
            self.assertTrue(any(r["kind"] == "error" for r in logs))

    def test_unauthorized_prompt_never_reaches_model(self):
        model = ScriptedModel(['{"final":"不应发生"}'])
        with running_system(model, authorized=False) as (
            runtime,
            runner,
            remote,
            events,
        ):
            self.assertFalse(runner.submit("起飞"))
            self.assertEqual(model.calls, [])
            self.assertTrue(
                any(
                    event.kind == "chat" and "Axis4" in event.data["text"]
                    for event in events
                )
            )

    def test_tool_observation_final_and_conversation_history(self):
        model = ScriptedModel(
            ['{"tool":"takeoff"}', '{"final":"已起飞"}', '{"final":"当前在悬停"}']
        )
        with running_system(model) as (runtime, runner, remote, events):
            self.assertTrue(runner.submit("起飞"))
            wait_until(lambda: not runner.get_status()["busy"])
            wait_until(lambda: runtime.get_status().get("agent_state") == "end")
            self.assertEqual(len(model.calls), 2)
            self.assertEqual(runtime.get_status()["flight_state"], "hovering")
            self.assertEqual(model.calls[0]["image"].size, (224, 224))
            feedback = json.loads(model.calls[1]["messages"][-1]["content"])
            self.assertEqual(feedback["tool_result"]["status"], "success")
            self.assertEqual(feedback["observation"]["flight_state"], "hovering")
            self.assertNotIn("data", feedback["observation"]["rgb"])
            runner.submit("现在什么状态？")
            wait_until(lambda: not runner.get_status()["busy"])
            self.assertTrue(
                any("已起飞" in msg["content"] for msg in model.calls[-1]["messages"])
            )
            self.assertTrue(
                any(
                    event.kind == "chat" and event.data.get("text") == "当前在悬停"
                    for event in events
                )
            )
        self.assertTrue(model.unloaded)

    def test_authoritative_axis4_check_even_with_stale_status(self):
        model = ScriptedModel(['{"final":"不应发生"}'])
        with running_system(model) as (runtime, runner, remote, events):
            actual_status = runtime.get_status
            # Simulate a UI snapshot which still says authorized after the switch falls.
            runtime.get_status = lambda: {**actual_status(), "authorized": True}
            remote.axes[4] = 0.0
            runner.submit("起飞")
            wait_until(lambda: not runner.get_status()["busy"])
            self.assertEqual(model.calls, [])

    def test_inference_oom_is_chat_error_and_ends_turn(self):
        model = ScriptedModel([MemoryError("CUDA out of memory")])
        with running_system(model) as (runtime, runner, remote, events):
            runner.submit("起飞")
            wait_until(lambda: not runner.get_status()["busy"])
            wait_until(lambda: runtime.get_status().get("agent_state") == "end")
            errors = [
                event.data["text"]
                for event in events
                if event.kind == "chat" and event.data["role"] == "error"
            ]
            self.assertTrue(any("OOM" in text for text in errors), errors)
            self.assertEqual(runner.get_status()["model_state"], "ready")

    def test_axis4_cancels_generation_drops_late_output_and_keeps_sim_running(self):
        model = ScriptedModel(['{"tool":"takeoff"}'])
        model.wait_for_cancel = True
        with running_system(model, realtime=True) as (runtime, runner, remote, events):
            runner.submit("起飞")
            self.assertTrue(model.entered.wait(10))
            with self.assertRaises(RuntimeError):
                runner.configure(RunnerConfig(max_decisions=1))
            start = runtime.get_status()["sim_time"]
            wait_until(lambda: runtime.get_status()["sim_time"] > start + 0.08)
            remote.axes[4] = 0.0
            wait_until(lambda: not runner.get_status()["busy"])
            wait_until(lambda: not runtime.get_status()["authorized"])
            self.assertEqual(runtime.get_status()["agent_state"], "end")
            self.assertFalse(
                any(
                    event.kind == "chat" and event.data["role"] == "tool"
                    for event in events
                )
            )
            self.assertFalse(runner.submit("再次起飞"))

    def test_invalid_json_retry_limit_and_tool_failure_feedback(self):
        model = ScriptedModel(["oops", "still invalid"])
        with running_system(model, config=RunnerConfig(max_parse_retries=1)) as (
            runtime,
            runner,
            remote,
            events,
        ):
            runner.submit("起飞")
            wait_until(lambda: not runner.get_status()["busy"])
            self.assertEqual(len(model.calls), 2)
            self.assertTrue(
                any(
                    event.kind == "chat" and "合法 JSON" in event.data["text"]
                    for event in events
                )
            )
        model = ScriptedModel(['{"tool":"land"}', '{"final":"尚未起飞，不能降落"}'])
        with running_system(model) as (runtime, runner, remote, events):
            runner.submit("降落")
            wait_until(lambda: not runner.get_status()["busy"])
            feedback = json.loads(model.calls[1]["messages"][-1]["content"])
            self.assertEqual(feedback["tool_result"]["status"], "error")

    def test_background_planning_does_not_block_axis4(self):
        entered, release = threading.Event(), threading.Event()

        class SlowOptimizer:
            def optimize(self, **kwargs):
                entered.set()
                release.wait(10)
                raise RuntimeError("discarded planning result")

        model = ScriptedModel(
            [
                '{"tool":"takeoff"}',
                '{"tool":"fly_waypoint","waypoint":[{"time":2,"position":[-1.8,2,1],"yaw":0}]}',
            ]
        )
        with patch("tools.fly_waypoint.PolynomialTrajectoryOptimizer", SlowOptimizer):
            try:
                with running_system(model) as (runtime, runner, remote, events):
                    runner.submit("起飞后飞行")
                    self.assertTrue(entered.wait(10))
                    start = runtime.get_status()["sim_time"]
                    wait_until(lambda: runtime.get_status()["sim_time"] > start + 0.05)
                    remote.axes[4] = 0.0
                    wait_until(lambda: not runner.get_status()["busy"])
                    self.assertEqual(runtime.get_status()["agent_state"], "end")
                    release.set()
            finally:
                release.set()

    def test_preview_camera_switch_and_model_load_error(self):
        model = ScriptedModel(['{"final":"就绪"}'])
        with running_system(model, preview=True) as (runtime, runner, remote, events):
            wait_until(lambda: runtime.get_preview() is not None)
            fpv = runtime.get_preview()
            runtime.set_camera("world_camera").result(timeout=10)
            wait_until(
                lambda: runtime.get_status().get("camera") == "world_camera"
                and runtime.get_preview() is not None
            )
            world = runtime.get_preview()
            self.assertEqual(world.size, fpv.size)
            self.assertNotEqual(world.tobytes(), fpv.tobytes())
            runner.unload_model()
            wait_until(lambda: not runner.get_status()["busy"])

            def failed_load(path):
                raise MemoryError("load OOM")

            runner._factory = failed_load
            runner.load_model()
            wait_until(lambda: not runner.get_status()["busy"])
            self.assertEqual(runner.get_status()["model_state"], "error")
            self.assertTrue(
                any(
                    event.kind == "chat" and "OOM" in event.data["text"]
                    for event in events
                )
            )


if __name__ == "__main__":
    unittest.main()
