"""Offscreen Qt UI checks: one preview, camera arrows and chat errors."""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from concurrent.futures import Future
import threading
import time
import unittest

from PIL import Image
from PySide6.QtCore import QPoint, QPointF, Qt
from PySide6.QtGui import QMouseEvent, QWheelEvent
from PySide6.QtWidgets import QApplication

from runtime.events import EventBus
from agent.runner import RunnerConfig
from ui.main_window import AgentWindow


class UIRuntime:
    def __init__(self):
        self.camera = "fpv_camera"
        self.closed = False
        self.camera_actions = []
        self.fov = 60

    def move_world_camera(self, *args):
        self.camera_actions.append(args)
        result = Future()
        result.set_result(None)
        return result

    def set_world_camera_fov(self, value):
        self.fov = value
        result = Future()
        result.set_result(None)
        return result

    def reset_world_camera(self):
        self.fov = 60
        result = Future()
        result.set_result(None)
        return result

    def get_status(self):
        return {
            "ready": True,
            "scene_id": "map1",
            "camera": self.camera,
            "authorized": False,
            "agent_state": "end",
            "flight_state": "hovering",
            "sim_time": 1.0,
            "position": [0.0, 0.0, 1.0],
            "yaw": 0.0,
        }

    def get_preview(self):
        return Image.new(
            "RGB", (224, 224), "red" if self.camera == "fpv_camera" else "blue"
        )

    def set_camera(self, camera):
        self.camera = camera
        result = Future()
        result.set_result(None)
        return result

    def close(self):
        self.closed = True


class UIRunner:
    def __init__(self, events):
        self.events = events
        self.closed = False
        self.config = RunnerConfig()

    def get_config(self):
        return self.config

    def configure(self, config):
        self.config = config

    def get_status(self):
        return {"model_state": "unloaded", "busy": False, "running_turn": False}

    def submit(self, text):
        self.events.publish(
            "chat", {"role": "system", "text": "Axis4 未授权，不会发送模型 prompt"}
        )

    def load_model(self, path):
        self.loaded = path

    def unload_model(self):
        pass

    def reload_environment(self, scene_id):
        self.reloaded_scene = scene_id
        return True

    def cancel(self, reason):
        pass

    def close(self):
        self.closed = True


class DesktopTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_camera_arrows_chat_and_async_close(self):
        events = EventBus()
        runtime, runner = UIRuntime(), UIRunner(events)
        window = AgentWindow(runtime, runner, events)
        window.show()
        window._refresh()
        self.assertEqual(window.scene_selector.count(), 5)
        window.scene_selector.setCurrentIndex(3)
        window.reload_scene_button.click()
        self.assertEqual(runner.reloaded_scene, "map4")
        self.assertIn("原地图", window.scene_status.text())
        idle_status = runner.get_status
        runner.get_status = lambda: {**idle_status(), "busy": True}
        window._refresh()
        self.assertFalse(window.reload_scene_button.isEnabled())
        self.assertFalse(window.scene_selector.isEnabled())
        runner.get_status = idle_status
        window._refresh()
        self.assertEqual(window.tabs.count(), 3)
        self.assertEqual(
            window.system_prompt.toPlainText(), runner.config.system_prompt
        )
        window.system_prompt.setPlainText("Custom JSON protocol")
        window.setting_fields["max_decisions"].setValue(3)
        window.apply_settings.click()
        self.assertEqual(runner.config.system_prompt, "Custom JSON protocol")
        self.assertEqual(runner.config.max_decisions, 3)
        window.include_depth.setChecked(True)
        window.setting_fields["depth_size"].setValue(80)
        window.apply_settings.click()
        self.assertTrue(runner.config.include_depth)
        self.assertEqual(runner.config.depth_size, 80)
        window.system_prompt.setPlainText("  ")
        window.apply_settings.click()
        self.assertEqual(runner.config.system_prompt, "Custom JSON protocol")
        for raw in ("invalid JSON", '{"final":"ok"}'):
            events.publish("agent_log", {"kind": "model_output", "data": {"raw": raw}})
        self.app.processEvents()
        self.assertIn("invalid JSON", window.model_output.toPlainText())
        self.assertIn("final", window.model_output.toPlainText())
        self.assertEqual(len(window._log_records), 2)
        window.backend_selector.setCurrentIndex(1)
        self.assertEqual(window.model_path.text(), "deepseek-flash")
        self.assertFalse(window.api_panel.isHidden())
        window.api_key.setText("ui-test-key")
        window.api_thinking.setChecked(False)
        window.load_button.click()
        self.assertEqual(runner.loaded.model, "deepseek-flash")
        self.assertFalse(runner.loaded.thinking)
        self.assertNotIn("ui-test-key", repr(runner.loaded))
        window.backend_selector.setCurrentIndex(0)
        self.assertTrue(window.api_panel.isHidden())
        window._refresh()
        self.assertFalse(window.preview.pixmap().isNull())
        window.next_camera.click()
        window._refresh()
        self.assertEqual(runtime.camera, "world_camera")
        self.assertEqual(window.camera_label.text(), "世界视角")
        actions = []
        window.preview.camera_move.connect(lambda *args: actions.append(args))
        center = window.preview.rect().center()
        press = QMouseEvent(
            QMouseEvent.Type.MouseButtonPress,
            QPointF(center),
            QPointF(center),
            Qt.MouseButton.LeftButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        window.preview.mousePressEvent(press)
        move = QMouseEvent(
            QMouseEvent.Type.MouseMove,
            QPointF(center + QPoint(20, 10)),
            QPointF(center + QPoint(20, 10)),
            Qt.MouseButton.NoButton,
            Qt.MouseButton.LeftButton,
            Qt.KeyboardModifier.NoModifier,
        )
        window.preview.mouseMoveEvent(move)
        self.assertEqual(actions[-1][0], "rotate")
        self.assertEqual(runtime.camera_actions[-1], actions[-1])
        wheel = QWheelEvent(
            QPointF(center),
            QPointF(center),
            QPoint(),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.NoModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        window.preview.wheelEvent(wheel)
        self.assertEqual(actions[-1], ("zoom", 0.0, -0.05))
        fov_wheel = QWheelEvent(
            QPointF(center),
            QPointF(center),
            QPoint(),
            QPoint(0, 120),
            Qt.MouseButton.NoButton,
            Qt.KeyboardModifier.ControlModifier,
            Qt.ScrollPhase.NoScrollPhase,
            False,
        )
        window.preview.wheelEvent(fov_wheel)
        self.assertEqual(runtime.fov, 57)
        window.view_tabs.setCurrentIndex(2)
        self.assertEqual(runtime.camera, "depth_camera")
        self.assertFalse(window.depth_legend.isHidden())
        self.assertTrue(window.camera_controls.isHidden())
        window.view_tabs.setCurrentIndex(1)
        window.previous_camera.click()
        self.assertEqual(runtime.camera, "fpv_camera")
        before = len(actions)
        window.preview.mousePressEvent(press)
        window.preview.mouseMoveEvent(move)
        window.preview.wheelEvent(wheel)
        self.assertEqual(len(actions), before)
        window.input.setPlainText("起飞")
        window.send_button.click()
        worker = threading.Thread(
            target=lambda: events.publish(
                "chat", {"role": "error", "text": "CUDA OOM 测试"}
            )
        )
        worker.start()
        worker.join()
        self.app.processEvents()
        self.assertIn("Axis4 未授权", window.chat.toPlainText())
        self.assertIn("CUDA OOM", window.chat.toPlainText())
        state = runtime.get_status()
        runtime.get_status = lambda: {
            **state,
            "authority_source": "no_joystick",
            "authorized": True,
        }
        window._refresh()
        self.assertIn("无手柄自动授权", window.status_labels["control"].text())
        self.assertIn("未检测到手柄", window.chat.toPlainText())
        window.close()
        deadline = time.monotonic() + 3
        while window.isVisible() and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.02)
        self.assertFalse(window.isVisible())
        self.assertTrue(runtime.closed and runner.closed)


if __name__ == "__main__":
    unittest.main()
