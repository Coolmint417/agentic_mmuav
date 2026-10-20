"""Offscreen real-runtime desktop preview; no model load or flight tool calls."""
import os
from pathlib import Path
import sys
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from PySide6.QtWidgets import QApplication
from agent.runner import AgentRunner
from env.mmuav_env import MMUAVEnv
from runtime import EnvironmentRuntime
from runtime.events import EventBus
from tests.support import FakeJoystick
from ui.main_window import AgentWindow


def main():
    app = QApplication([])
    events = EventBus()
    runtime = EnvironmentRuntime(
        events=events,
        environment_factory=lambda: MMUAVEnv(joystick_reader=FakeJoystick(False)),
    )
    runner = AgentRunner(runtime, events=events)
    window = AgentWindow(runtime, runner, events)
    runtime.start()
    window.show()
    try:
        deadline = time.monotonic() + 15
        while runtime.get_preview() is None and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.02)
        if runtime.get_preview() is None:
            raise RuntimeError("No camera preview")
        window.next_camera.click()
        deadline = time.monotonic() + 10
        while (
            runtime.get_status().get("camera") != "world_camera"
            or runtime.get_preview() is None
        ) and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.02)
        window.input.setPlainText("起飞并向北移动 0.5 米")
        window.send_button.click()
        app.processEvents()
        window._refresh()
        path = Path("artifacts/desktop_ui_smoke.png")
        path.parent.mkdir(exist_ok=True)
        if not window.grab().save(str(path)):
            raise RuntimeError("Screenshot could not be saved")
        print("Desktop screenshot:", path)
        runtime.move_world_camera("rotate", 0.08, -0.03).result(timeout=5)
        runtime.move_world_camera("move", 0.03, 0.01).result(timeout=5)
        runtime.set_world_camera_fov(75).result(timeout=5)
        deadline = time.monotonic() + 0.25
        while time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.01)
        window._refresh()
        path = Path("artifacts/desktop_camera_interactive.png")
        if not window.grab().save(str(path)):
            raise RuntimeError("Interactive camera screenshot could not be saved")
        print("Interactive camera screenshot:", path)
        window.view_tabs.setCurrentIndex(2)
        deadline = time.monotonic() + 10
        while (
            runtime.get_status().get("camera") != "depth_camera"
            or runtime.get_preview() is None
        ) and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.02)
        window._refresh()
        path = Path("artifacts/desktop_depth_preview.png")
        if not window.grab().save(str(path)):
            raise RuntimeError("Depth screenshot could not be saved")
        print("Depth screenshot:", path)
    finally:
        window.close()
        deadline = time.monotonic() + 15
        while window.isVisible() and time.monotonic() < deadline:
            app.processEvents()
            time.sleep(0.02)
        if window.isVisible():
            raise RuntimeError("Window shutdown did not finish")


if __name__ == "__main__":
    main()
