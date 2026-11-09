"""Dedicated replay desktop, independent of the model configuration UI."""
import html
import json
from pathlib import Path
import threading

from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QComboBox, QDoubleSpinBox, QFileDialog, QHBoxLayout, QLabel, QLineEdit,
    QMainWindow, QPlainTextEdit, QPushButton, QSplitter, QTabWidget,
    QTextBrowser, QVBoxLayout, QWidget,
)

from agent.replay import ReplayLog
from env.scenes import SCENES
from .preview import CameraPreview


class ReplayBridge(QObject):
    received = Signal(object)


class ReplayWindow(QMainWindow):
    cameras = (("fpv_camera", "第一视角"), ("world_camera", "世界视角"),
               ("depth_camera", "第一视角深度"))

    def __init__(self, runtime, runner, events, *, log_path="", scene_id="map1"):
        super().__init__()
        self.runtime, self.runner, self.events = runtime, runner, events
        self._closing = False
        self._shutdown_done = threading.Event()
        self._shutdown_error = None
        self._bridge = ReplayBridge(self)
        self._bridge.received.connect(self._on_event, Qt.ConnectionType.QueuedConnection)
        events.subscribe(self._bridge.received.emit)
        self.setWindowTitle("MMUAV · 日志重放")
        self.resize(1280, 850)
        root = QWidget()
        layout = QVBoxLayout(root)
        files = QHBoxLayout()
        self.log_path = QLineEdit(str(log_path))
        self.log_path.setPlaceholderText("选择 logs/*.jsonl，无需加载模型")
        self.browse = QPushButton("选择日志")
        self.browse.clicked.connect(self._browse)
        self.scene_selector = QComboBox()
        for scene in SCENES:
            self.scene_selector.addItem(scene.label, scene.id)
        self.scene_selector.setCurrentIndex(self.scene_selector.findData(scene_id))
        for widget in (self.log_path, self.browse, QLabel("原实验地图"), self.scene_selector):
            files.addWidget(widget)
        layout.addLayout(files)
        controls = QHBoxLayout()
        self.start_button = QPushButton("重置地图并开始")
        self.start_button.clicked.connect(self._start)
        self.pause_button = QPushButton("暂停")
        self.pause_button.clicked.connect(self._pause)
        self.stop_button = QPushButton("终止")
        self.stop_button.clicked.connect(lambda: self.runner.cancel())
        self.interval = QDoubleSpinBox()
        self.interval.setRange(0, 10)
        self.interval.setSingleStep(0.1)
        self.interval.setValue(0.3)
        self.interval.setSuffix(" s")
        self.progress = QLabel("等待开始")
        for widget in (self.start_button, self.pause_button, self.stop_button,
                       QLabel("事件展示间隔"), self.interval, self.progress):
            controls.addWidget(widget)
        layout.addLayout(controls)
        note = QLabel("跳过原模型与用户等待；工具正常执行。暂停只停止日志推进，当前工具完成后暂停，仿真继续。"
                      "画面是本次渲染，日志未保存原始图片；请手动选择对应地图。")
        note.setWordWrap(True)
        layout.addWidget(note)
        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        view = QVBoxLayout(left)
        camera_bar = QHBoxLayout()
        self.camera_selector = QComboBox()
        for _, title in self.cameras:
            self.camera_selector.addItem(title)
        self.camera_selector.currentIndexChanged.connect(self._select_camera)
        self.fov = QDoubleSpinBox()
        self.fov.setRange(10, 150)
        self.fov.setValue(60)
        self.fov.setSuffix("°")
        self.fov.valueChanged.connect(lambda value: self._camera_request(
            lambda: self.runtime.set_world_camera_fov(value)))
        reset = QPushButton("重置相机")
        reset.clicked.connect(lambda: self._camera_request(self.runtime.reset_world_camera))
        for widget in (self.camera_selector, QLabel("世界视角 FOV"), self.fov, reset):
            camera_bar.addWidget(widget)
        view.addLayout(camera_bar)
        self.preview = CameraPreview("正在启动仿真……")
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(320, 240)
        self.preview.camera_move.connect(lambda action, dx, dy: self._camera_request(
            lambda: self.runtime.move_world_camera(action, dx, dy)))
        self.preview.fov_steps.connect(lambda steps: self.fov.setValue(self.fov.value() - steps * 3))
        self.preview.reset_camera.connect(lambda: self._camera_request(self.runtime.reset_world_camera))
        view.addWidget(self.preview, 1)
        self.runtime_status = QLabel("仿真初始化中")
        self.runtime_status.setWordWrap(True)
        view.addWidget(self.runtime_status)
        splitter.addWidget(left)
        tabs = QTabWidget()
        self.chat = QTextBrowser()
        self.raw = QPlainTextEdit()
        self.raw.setReadOnly(True)
        self.comparison = QPlainTextEdit()
        self.comparison.setReadOnly(True)
        tabs.addTab(self.chat, "原对话与输出")
        tabs.addTab(self.raw, "原始 JSON（全部事件）")
        tabs.addTab(self.comparison, "本次执行对比")
        splitter.addWidget(tabs)
        splitter.setSizes([720, 560])
        layout.addWidget(splitter, 1)
        self.setCentralWidget(root)
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(33)
        self._refresh()

    def _browse(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择重放日志", str(Path(__file__).resolve().parents[1] / "logs"),
            "JSONL 日志 (*.jsonl)")
        if path:
            self.log_path.setText(path)

    def _append(self, title, text):
        self.chat.append(f"<p><b>{html.escape(title)}</b><br>"
                         + html.escape(str(text)).replace("\n", "<br>") + "</p>")

    def _start(self):
        try:
            log = ReplayLog.load(self.log_path.text())
            self.runner.start(log, self.scene_selector.currentData(),
                              event_interval=self.interval.value())
            self.chat.clear()
            self.raw.clear()
            self.comparison.clear()
        except Exception as exc:
            self._append("无法开始", str(exc))

    def _pause(self):
        if self.runner.get_status()["state"] in {"paused", "pausing"}:
            self.runner.resume()
        else:
            self.runner.pause()

    def _camera_request(self, operation):
        try:
            future = operation()
            def complete(future):
                try:
                    future.result()
                except Exception as exc:
                    self.events.publish("replay_error", str(exc))
            future.add_done_callback(complete)
        except Exception as exc:
            self._append("相机错误", str(exc))

    def _select_camera(self, index):
        self.preview.set_interactive(self.cameras[index][0] == "world_camera")
        self.preview.clear()
        self._camera_request(lambda: self.runtime.set_camera(self.cameras[index][0]))

    def _on_event(self, event):
        if self._closing:
            return
        if event.kind == "replay_record":
            record = event.data["record"]
            data, kind = record["data"], record["kind"]
            self.raw.appendPlainText(json.dumps(record, ensure_ascii=False, indent=2))
            if kind == "observation" and "task" in data:
                # The task repeats after every tool; show once per turn.
                if getattr(self, "_last_task_turn", None) != record.get("turn_id"):
                    self._last_task_turn = record.get("turn_id")
                    self._append(f"原用户 · 第 {record.get('turn_id')} 轮", data["task"])
            elif kind == "model_output":
                reasoning = data.get("metadata", {}).get("reasoning_content")
                if reasoning:
                    self._append("原模型思考", reasoning)
                self._append("原模型输出", data.get("raw", ""))
            elif kind in {"tool_call", "parse_error", "error", "cancelled"}:
                self._append("原记录 · " + kind, json.dumps(data, ensure_ascii=False))
        elif event.kind == "replay_started":
            self._last_task_turn = None
            self._append("重放开始", json.dumps(event.data, ensure_ascii=False))
        elif event.kind in {"replay_result", "replay_observation"}:
            self.comparison.appendPlainText(
                event.kind + "\n" + json.dumps(event.data, ensure_ascii=False, indent=2))
        elif event.kind == "replay_error":
            self._append("本次执行错误", event.data)
        elif event.kind == "chat" and event.data.get("role") == "error":
            self._append("仿真错误", event.data["text"])

    def _refresh(self):
        if self._closing:
            if self._shutdown_done.is_set():
                if self._shutdown_error:
                    self._closing = False
                    self.setEnabled(True)
                    self._append("关闭错误", self._shutdown_error)
                    self._shutdown_done.clear()
                else:
                    self.close()
            return
        replay, state = self.runner.get_status(), self.runtime.get_status()
        busy = replay["busy"]
        for widget in (self.log_path, self.browse, self.scene_selector, self.interval):
            widget.setEnabled(not busy)
        self.start_button.setEnabled(not busy and state.get("ready", False))
        self.pause_button.setEnabled(busy and replay["state"] != "stopping")
        self.stop_button.setEnabled(busy)
        self.pause_button.setText("继续" if replay["state"] in {"paused", "pausing"} else "暂停")
        labels = {"idle": "未开始", "playing": "播放中", "pausing": "等待当前操作完成后暂停",
                  "paused": "已暂停（仿真继续）", "stopping": "正在终止", "stopped": "已终止",
                  "completed": "重放完成", "error": "执行错误"}
        self.progress.setText(f"{labels.get(replay['state'], replay['state'])} · "
                              f"{replay['index']}/{replay['total']} · 工具 {replay['tool'] or '—'}")
        self.runtime_status.setText(
            f"时间 {state.get('sim_time', 0):.2f}s · 授权 {state.get('authorized', False)} · "
            f"飞机 {state.get('flight_state', '—')} · Agent {state.get('agent_state', '—')}\n"
            f"位置 {state.get('position', '—')} · 吸盘 {state.get('sucker_states', '—')}"
            + (f"\n{state['error']}" if state.get("error") else ""))
        camera = state.get("world_camera")
        if camera:
            self.fov.blockSignals(True)
            self.fov.setValue(camera["fov"])
            self.fov.blockSignals(False)
        if state.get("camera") == self.cameras[self.camera_selector.currentIndex()][0]:
            image = self.runtime.get_preview()
            if image is not None:
                image = image.convert("RGB")
                qimage = QImage(image.tobytes(), image.width, image.height,
                                image.width * 3, QImage.Format.Format_RGB888).copy()
                ratio = self.preview.devicePixelRatioF()
                pixmap = QPixmap.fromImage(qimage).scaled(
                    self.preview.size() * ratio, Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation)
                pixmap.setDevicePixelRatio(ratio)
                self.preview.setPixmap(pixmap)

    def _shutdown(self):
        self._shutdown_error = None
        try:
            self.runner.close()
            self.runtime.close()
        except Exception as exc:
            self._shutdown_error = str(exc)
        finally:
            self._shutdown_done.set()

    def closeEvent(self, event):
        if self._shutdown_done.is_set() and not self._shutdown_error:
            self._timer.stop()
            event.accept()
        else:
            event.ignore()
            if not self._closing:
                self._closing = True
                self.setEnabled(False)
                threading.Thread(target=self._shutdown, name="replay-shutdown", daemon=True).start()
