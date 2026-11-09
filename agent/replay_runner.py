"""Replay log events and execute tools through the existing simulation owner."""
import math
import threading
import time

from runtime.events import EventBus, describe_error
from .replay import ReplayLog


class ReplayRunner:
    def __init__(self, runtime, *, events=None):
        self.runtime = runtime
        self.events = events or EventBus()
        self._condition = threading.Condition()
        self._cancel = threading.Event()
        self._pause = False
        self._closed = False
        self._thread = None
        self._status = {"state": "idle", "busy": False, "index": 0, "total": 0,
                        "source_turn": None, "tool": None, "error": None}
        self.runtime.set_cancel_callback(self.cancel)

    def get_status(self):
        with self._condition:
            return dict(self._status)

    def _update(self, **values):
        with self._condition:
            self._status.update(values)
            snapshot = dict(self._status)
        self.events.publish("replay_status", snapshot)

    def start(self, log, scene_id, *, event_interval=0.3):
        if not isinstance(log, ReplayLog):
            raise TypeError("Expected ReplayLog")
        if (isinstance(event_interval, bool) or not isinstance(event_interval, (int, float))
                or not math.isfinite(event_interval) or event_interval < 0):
            raise ValueError("事件展示间隔必须为非负有限秒数")
        with self._condition:
            if self._closed or self._status["busy"]:
                raise RuntimeError("请先等待当前重放退出")
            self._cancel.clear()
            self._pause = False
            self._status.update(state="playing", busy=True, index=0,
                                total=len(log.records), source_turn=None, tool=None, error=None)
            self._thread = threading.Thread(target=self._run,
                                            args=(log, scene_id, event_interval),
                                            name="replay-runner", daemon=True)
            self._thread.start()

    def pause(self):
        with self._condition:
            if self._status["busy"]:
                self._pause = True
                self._status["state"] = "pausing"
                self._condition.notify_all()

    def resume(self):
        with self._condition:
            if self._status["busy"]:
                self._pause = False
                self._status["state"] = "playing"
                self._condition.notify_all()

    def cancel(self, reason="用户终止重放"):
        with self._condition:
            if not self._status["busy"]:
                return
            self._cancel.set()
            self._status["state"] = "stopping"
            self._condition.notify_all()
        self.runtime.cancel_turn(reason)

    def _check_cancelled(self):
        if self._cancel.is_set():
            raise InterruptedError("重放已终止")

    def _boundary(self, delay=0):
        """Only the replay thread waits; never the physics thread or Qt thread."""
        with self._condition:
            remaining = delay
            while True:
                self._check_cancelled()
                if self._pause:
                    self._status["state"] = "paused"
                    self._condition.wait()
                elif remaining > 0:
                    started = time.monotonic()
                    self._condition.wait(remaining)
                    remaining = max(0, remaining - (time.monotonic() - started))
                else:
                    self._status["state"] = "playing"
                    return

    def _run(self, log, scene_id, interval):
        active_turn = None
        source_turn = None
        actual_observation = None
        completed_turns = set()
        outcome = "completed"
        try:
            self.runtime.wait_ready()
            self._check_cancelled()
            self.runtime.reload_scene(scene_id)
            self.events.publish("replay_started", log.summary)
            for index, record in enumerate(log.records):
                self._boundary(interval if index else 0)
                kind, data, turn = record["kind"], record["data"], record.get("turn_id")
                if turn is not None and turn != source_turn:
                    if turn in completed_turns:
                        raise ValueError("日志轮次交错，无法安全重放")
                    if active_turn is not None:
                        self.runtime.finish_turn(active_turn)
                    if source_turn is not None:
                        completed_turns.add(source_turn)
                    first_observation = next((r["data"].get("observation", {})
                                              for r in log.records[index:]
                                              if r.get("turn_id") == turn and r["kind"] == "observation"), {})
                    depth = first_observation.get("depth", {})
                    active_turn, packet = self.runtime.begin_turn(
                        include_depth=bool(depth), depth_size=depth.get("width", 56))
                    actual_observation = packet.to_dict(embed_images=False)
                    source_turn = turn
                self._update(index=index + 1, source_turn=turn)
                self.events.publish("replay_record", {"index": index + 1, "record": record})
                if kind == "environment_reload":
                    if active_turn is not None:
                        self.runtime.finish_turn(active_turn)
                        active_turn = None
                    self.runtime.reload_scene(data["scene_id"])
                elif kind == "tool_call":
                    if active_turn is None:
                        raise ValueError("已结束的轮次又出现工具调用")
                    self._update(tool=data["tool"])
                    # No pause boundary until the tool AND observation are complete.
                    result = self.runtime.execute_tool(data, active_turn).result()
                    self._check_cancelled()
                    packet = self.runtime.collect_observation(active_turn)
                    actual_observation = packet.to_dict(embed_images=False)
                    self.events.publish("replay_result", {
                        "index": index + 1, "source_turn": turn, "command": data,
                        "recorded": log.expected_result(index), "actual": result,
                    })
                    self._update(tool=None)
                elif kind == "observation":
                    old = data.get("observation", {})
                    new = actual_observation or {}
                    old_position, new_position = old.get("position_m"), new.get("position_m")
                    self.events.publish("replay_observation", {
                        "source_turn": turn,
                        "recorded_position": old_position, "actual_position": new_position,
                        "position_error_m": math.dist(old_position, new_position)
                        if old_position is not None and new_position is not None else None,
                        "recorded_state": old.get("flight_state"),
                        "actual_state": new.get("flight_state"),
                        "recorded_sim_time": old.get("sim_time_seconds"),
                        "actual_sim_time": new.get("sim_time_seconds"),
                    })
                elif kind == "decision" and "final" in data:
                    if active_turn is not None:
                        self.runtime.finish_turn(active_turn)
                        active_turn = None
                elif kind in {"error", "cancelled"}:
                    self.runtime.cancel_turn("重放日志中的终止事件")
                    active_turn = None
            self._boundary()
            if active_turn is not None:
                self.runtime.finish_turn(active_turn)
                active_turn = None
        except InterruptedError:
            outcome = "stopped"
        except Exception as exc:
            outcome = "stopped" if self._cancel.is_set() else "error"
            if outcome == "error":
                self._update(error=describe_error(exc))
                self.events.publish("replay_error", describe_error(exc))
        finally:
            if active_turn is not None:
                self.runtime.cancel_turn("重放退出")
            self._update(state=outcome, busy=False, tool=None)

    def close(self):
        with self._condition:
            self._closed = True
        self.cancel("关闭重放")
        if self._thread is not None:
            self._thread.join(timeout=20)
            if self._thread.is_alive():
                raise TimeoutError("重放线程尚未退出")
