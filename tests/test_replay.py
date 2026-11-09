"""Replay dispatch, deferred pause, cancellation and real simulation integration."""
from concurrent.futures import Future
import json
from pathlib import Path
import tempfile
import threading
import time
import unittest

from agent.replay import ReplayLog
from agent.replay_runner import ReplayRunner
from runtime.events import EventBus
from tests.test_runner import wait_until


def record(kind, data, turn=1):
    return {"kind": kind, "data": data, "turn_id": turn}


def simple_log():
    command = {"tool": "takeoff"}
    return ReplayLog(Path("test.jsonl"), (
        record("observation", {"task": "起飞", "observation": {"position_m": [0, 0, 0]}}),
        record("model_input", {"messages": [{"role": "user", "content": "起飞"}]}),
        record("model_output", {"raw": "bad json"}),
        record("parse_error", {"error": "invalid JSON"}),
        record("model_output", {"raw": json.dumps(command)}),
        record("decision", command),
        record("tool_call", command),
        record("tool_result", {"status": "success"}),
        record("observation", {"observation": {"position_m": [0, 0, 1]}}),
        record("model_output", {"raw": '{"final":"完成"}'}),
        record("decision", {"final": "完成"}),
    ))


class Packet:
    def to_dict(self, **kwargs):
        return {"position_m": [0, 0, 1], "flight_state": "hovering"}


class FakeRuntime:
    def __init__(self):
        self.calls = []
        self.entered = threading.Event()
        self.future = Future()
        self.observed = threading.Event()

    def set_cancel_callback(self, callback):
        self.callback = callback

    def wait_ready(self):
        pass

    def reload_scene(self, scene):
        self.calls.append(("reload", scene))

    def begin_turn(self, **kwargs):
        self.calls.append(("begin", kwargs))
        return 42, Packet()

    def finish_turn(self, turn):
        self.calls.append(("finish", turn))

    def execute_tool(self, command, turn):
        self.calls.append(("tool", command, turn))
        self.entered.set()
        return self.future

    def collect_observation(self, turn):
        self.calls.append(("observe", turn))
        self.observed.set()
        return Packet()

    def cancel_turn(self, reason):
        self.calls.append(("cancel", reason))
        if not self.future.done():
            self.future.set_result({"status": "terminated"})


class ReplayTests(unittest.TestCase):
    def test_reader_reports_line_and_ignores_bad_model_json(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "log.jsonl"
            path.write_text("\n".join(json.dumps(r) for r in simple_log().records))
            parsed = ReplayLog.load(path)
            self.assertEqual(parsed.summary["tools"], 1)
            self.assertEqual(parsed.summary["turns"], 1)
            path.write_text('{}\n{"kind":"tool_call"}')
            with self.assertRaisesRegex(ValueError, "第 1 行"):
                ReplayLog.load(path)
            path.write_text(json.dumps(record("tool_call", {"tool": "unknown"})))
            with self.assertRaisesRegex(ValueError, "第 1 行"):
                ReplayLog.load(path)

    def test_pause_waits_for_tool_and_observation_without_duplicate_dispatch(self):
        runtime, events, received = FakeRuntime(), EventBus(), []
        events.subscribe(received.append)
        runner = ReplayRunner(runtime, events=events)
        try:
            runner.start(simple_log(), "map1", event_interval=0)
            self.assertTrue(runtime.entered.wait(3))
            runner.pause()
            self.assertEqual(runner.get_status()["state"], "pausing")
            self.assertFalse(runtime.observed.is_set())
            runtime.future.set_result({"status": "success"})
            wait_until(lambda: runner.get_status()["state"] == "paused")
            self.assertTrue(runtime.observed.is_set())
            index = runner.get_status()["index"]
            time.sleep(.05)
            self.assertEqual(runner.get_status()["index"], index)
            self.assertFalse(any(call[0] == "finish" for call in runtime.calls))
            runner.resume()
            wait_until(lambda: not runner.get_status()["busy"])
            self.assertEqual(runner.get_status()["state"], "completed")
            self.assertEqual(sum(call[0] == "tool" for call in runtime.calls), 1)
            self.assertIn(("finish", 42), runtime.calls)
            records = [e.data["record"] for e in received if e.kind == "replay_record"]
            self.assertEqual(records, list(simple_log().records))
            result = next(e.data for e in received if e.kind == "replay_result")
            self.assertEqual(result["actual"], result["recorded"])
        finally:
            runner.close()

    def test_stop_during_tool_and_reject_concurrent_start(self):
        runtime = FakeRuntime()
        runner = ReplayRunner(runtime)
        try:
            runner.start(simple_log(), "map1", event_interval=0)
            self.assertTrue(runtime.entered.wait(3))
            with self.assertRaises(RuntimeError):
                runner.start(simple_log(), "map1")
            runner.cancel()
            wait_until(lambda: not runner.get_status()["busy"])
            self.assertEqual(runner.get_status()["state"], "stopped")
            self.assertFalse(runtime.observed.is_set())
        finally:
            runner.close()

    def test_pause_between_events_can_be_cancelled(self):
        runtime = FakeRuntime()
        runner = ReplayRunner(runtime)
        try:
            runner.start(simple_log(), "map1", event_interval=10)
            wait_until(lambda: runner.get_status()["index"] == 1)
            runner.pause()
            wait_until(lambda: runner.get_status()["state"] == "paused")
            self.assertFalse(runtime.entered.is_set())
            runner.cancel()
            wait_until(lambda: not runner.get_status()["busy"])
            self.assertEqual(runner.get_status()["state"], "stopped")
        finally:
            runner.close()

    def test_real_runtime_multiple_turns_and_paused_physics(self):
        from env.mmuav_env import MMUAVEnv
        from runtime import EnvironmentRuntime, RuntimeConfig
        from tests.support import FakeJoystick
        runtime = EnvironmentRuntime(
            RuntimeConfig(realtime=False, preview_enabled=False),
            environment_factory=lambda: MMUAVEnv(joystick_reader=FakeJoystick(True)))
        runtime.start()
        runtime.wait_ready()
        runner = ReplayRunner(runtime)
        events, received = runner.events, []
        events.subscribe(received.append)
        try:
            records = list(simple_log().records)
            records.extend((record("observation", {"task": "第二轮"}, 2),
                            record("tool_call", {"tool": "land"}, 2),
                            record("decision", {"final": "结束"}, 2)))
            runner.start(ReplayLog(Path("test"), tuple(records)), "map1", event_interval=.01)
            wait_until(lambda: len([e for e in received if e.kind == "replay_result"]) == 1)
            runner.pause()
            wait_until(lambda: runner.get_status()["state"] == "paused")
            before = runtime.get_status()["sim_time"]
            time.sleep(.05)
            self.assertGreater(runtime.get_status()["sim_time"], before)
            runner.resume()
            wait_until(lambda: not runner.get_status()["busy"], timeout=30)
            self.assertEqual(runner.get_status()["state"], "completed", runner.get_status())
            results = [e.data["actual"] for e in received if e.kind == "replay_result"]
            self.assertEqual(len(results), 2)
            self.assertTrue(all(result["status"] == "success" for result in results), results)
            self.assertEqual(runtime.get_status()["agent_state"], "end")
            self.assertEqual(runtime.get_status()["flight_state"], "on_ground")
        finally:
            runner.close()
            runtime.close()
