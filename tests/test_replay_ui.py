"""Standalone replay UI wiring without a model or an interactive display."""
import json
import os
from pathlib import Path
import tempfile
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PIL import Image
from PySide6.QtWidgets import QApplication

from agent.replay_runner import ReplayRunner
from runtime.events import EventBus
from tests.test_replay import FakeRuntime, simple_log
from ui.replay_window import ReplayWindow


class UIRuntime(FakeRuntime):
    def __init__(self):
        super().__init__()
        self.camera = "fpv_camera"
        self.closed = False

    def get_status(self):
        return {"ready": True, "camera": self.camera, "sim_time": 1,
                "authorized": True, "flight_state": "hovering", "agent_state": "think"}

    def get_preview(self):
        return Image.new("RGB", (320, 240))

    def set_camera(self, camera):
        from concurrent.futures import Future
        self.camera = camera
        future = Future()
        future.set_result(None)
        return future

    def close(self):
        self.closed = True


class ReplayUITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.application = QApplication.instance() or QApplication([])

    def pump_until(self, predicate):
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            self.application.processEvents()
            if predicate():
                return
            time.sleep(.01)
        self.fail("UI operation timed out")

    def test_standalone_window_renders_records_compares_results_and_closes(self):
        runtime, events = UIRuntime(), EventBus()
        runner = ReplayRunner(runtime, events=events)
        window = ReplayWindow(runtime, runner, events)
        try:
            window.show()
            window._refresh()
            self.assertFalse(window.preview.pixmap().isNull())
            window.camera_selector.setCurrentIndex(1)
            self.assertEqual(runtime.camera, "world_camera")
            self.assertTrue(window.preview._interactive)
            window.camera_selector.setCurrentIndex(2)
            self.assertEqual(runtime.camera, "depth_camera")
            self.assertFalse(window.preview._interactive)
            with tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "replay.jsonl"
                path.write_text("\n".join(json.dumps(r) for r in simple_log().records))
                window.log_path.setText(str(path))
                window.interval.setValue(0)
                window.start_button.click()
                self.pump_until(runtime.entered.is_set)
                window._refresh()
                window.pause_button.click()
                runtime.future.set_result({"status": "success"})
                self.pump_until(lambda: runner.get_status()["state"] == "paused")
                window._refresh()
                self.assertEqual(window.pause_button.text(), "继续")
                self.assertFalse(window.start_button.isEnabled())
                window.pause_button.click()
                self.pump_until(lambda: not runner.get_status()["busy"])
                self.pump_until(lambda: '"final"' in window.raw.toPlainText())
                self.assertIn("原模型输出", window.chat.toPlainText())
                self.assertIn("bad json", window.chat.toPlainText())
                self.assertIn("replay_result", window.comparison.toPlainText())
        finally:
            window.close()
            self.pump_until(lambda: runtime.closed)
            self.pump_until(lambda: not window.isVisible())
