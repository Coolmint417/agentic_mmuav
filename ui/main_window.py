"""Single-camera desktop UI; all widget updates run on the Qt main thread."""
import html
import json
import os
import threading
from dataclasses import replace

from PySide6.QtCore import QObject, Qt, QTimer, Signal, Slot
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QSpinBox,
    QTabWidget,
    QScrollArea,
    QComboBox,
    QCheckBox,
    QTabBar,
    QTextBrowser,
    QVBoxLayout,
    QWidget,
)

from agent.runner import RunnerConfig
from agent.backends import DeepSeekConfig
from .preview import CameraPreview


class EventBridge(QObject):
    received = Signal(object)


class AgentWindow(QMainWindow):
    cameras = (
        ("fpv_camera", "第一视角"),
        ("world_camera", "世界视角"),
        ("depth_camera", "第一视角深度"),
    )

    def __init__(self, runtime, runner, events):
        super().__init__()
        self.runtime, self.runner = runtime, runner
        self._camera_index = 0
        self._closing = False
        self._shutdown_done = threading.Event()
        self._shutdown_error = None
        self._log_records = []
        self._authority_source = None
        self.setWindowTitle("MMUAV · 本地无人机 Agent")
        self.resize(1180, 780)
        self._bridge = EventBridge(self)
        self._bridge.received.connect(
            self._on_event, Qt.ConnectionType.QueuedConnection
        )
        events.subscribe(self._bridge.received.emit)
        self._build()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(33)
        self._append_chat(
            "system",
            "先加载本地模型或配置 DeepSeek API。有手柄时将 Axis4 切为 1；启动时未检测到手柄则自动授权 Agent。模型观测始终使用第一视角。",
        )

    def _build(self):
        root = QWidget()
        layout = QVBoxLayout(root)
        model_bar = QHBoxLayout()
        self.backend_selector = QComboBox()
        self.backend_selector.addItem("本地 Ministral", "ministral")
        self.backend_selector.addItem("DeepSeek API", "deepseek")
        model_bar.addWidget(self.backend_selector)
        self.model_path = QLineEdit("mistralai/Ministral-3-3B-Instruct-2512")
        self.model_path.setPlaceholderText("Hugging Face 缓存 ID 或本地模型目录")
        self.browse_button = QPushButton("选择目录")
        self.browse_button.clicked.connect(self._browse)
        self.load_button = QPushButton("加载模型")
        self.load_button.clicked.connect(self._load_model)
        self.unload_button = QPushButton("卸载模型")
        self.unload_button.clicked.connect(self.runner.unload_model)
        self.model_status = QLabel("模型：未加载")
        for widget in (
            self.model_path,
            self.browse_button,
            self.load_button,
            self.unload_button,
            self.model_status,
        ):
            model_bar.addWidget(widget)
        layout.addLayout(model_bar)
        self.api_panel = QWidget()
        api_form = QFormLayout(self.api_panel)
        self.api_url = QLineEdit(
            os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        )
        self.api_key = QLineEdit()
        self.api_key.setEchoMode(QLineEdit.EchoMode.Password)
        self.api_key.setPlaceholderText("留空读取 DEEPSEEK_API_KEY；仅本次运行使用")
        self.api_thinking = QCheckBox("启用思考模式")
        self.api_thinking.setChecked(True)
        self.api_effort = QComboBox()
        self.api_effort.addItems(["low", "high", "max"])
        self.api_effort.setCurrentText("high")
        self.api_timeout = QSpinBox()
        self.api_timeout.setRange(1, 3600)
        self.api_timeout.setValue(180)
        self.api_timeout.setSuffix(" s")
        options = QHBoxLayout()
        for widget in (
            self.api_thinking,
            QLabel("思考强度"),
            self.api_effort,
            QLabel("请求超时"),
            self.api_timeout,
        ):
            options.addWidget(widget)
        api_form.addRow("API 地址", self.api_url)
        api_form.addRow("API Key", self.api_key)
        api_form.addRow(options)
        note = QLabel(
            "DeepSeek 将接收任务、对话历史、无人机状态和第一视角图片；启用 JSON Output。思考模式建议至少 4096 个生成 token，在 Agent 设置中修改。"
        )
        note.setWordWrap(True)
        api_form.addRow(note)
        layout.addWidget(self.api_panel)
        self.api_panel.hide()
        self.backend_selector.currentIndexChanged.connect(self._backend_changed)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        left = QWidget()
        left_layout = QVBoxLayout(left)
        navigation = QHBoxLayout()
        self.previous_camera = QPushButton("◀")
        self.next_camera = QPushButton("▶")
        self.camera_label = QLabel("第一视角")
        self.camera_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.previous_camera.clicked.connect(lambda: self._switch_camera(-1))
        self.next_camera.clicked.connect(lambda: self._switch_camera(1))
        navigation.addWidget(self.previous_camera)
        navigation.addWidget(self.camera_label, 1)
        navigation.addWidget(self.next_camera)
        left_layout.addLayout(navigation)
        self.view_tabs = QTabBar()
        for _, title in self.cameras:
            self.view_tabs.addTab(title)
        self.view_tabs.setExpanding(True)
        self.view_tabs.currentChanged.connect(self._select_camera)
        left_layout.addWidget(self.view_tabs)
        self.preview = CameraPreview("正在启动仿真与相机……")
        self.preview.camera_move.connect(self._move_world_camera)
        self.preview.fov_steps.connect(
            lambda steps: self.camera_fov.setValue(
                self.camera_fov.value() - round(steps * 3)
            )
        )
        self.preview.reset_camera.connect(self._reset_world_camera)
        self.preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview.setMinimumSize(320, 320)
        self.preview.setStyleSheet(
            "background:#161b22; color:#d4d9e0; border-radius:6px;"
        )
        left_layout.addWidget(self.preview, 1)
        self.depth_legend = QLabel("深度图：白色近、黑色 ≥10 m 或无回波；相机前向 Z 深度。此预览独立于模型深度开关。")
        self.depth_legend.setWordWrap(True)
        left_layout.addWidget(self.depth_legend)
        self.depth_legend.hide()
        self.camera_controls = QWidget()
        camera_layout = QVBoxLayout(self.camera_controls)
        camera_layout.setContentsMargins(0, 0, 0, 0)
        camera_buttons = QHBoxLayout()
        camera_buttons.addWidget(QLabel("FOV"))
        self.camera_fov = QSpinBox()
        self.camera_fov.setRange(5, 150)
        self.camera_fov.setValue(60)
        self.camera_fov.setSuffix("°")
        self.camera_fov.valueChanged.connect(self._set_world_fov)
        camera_buttons.addWidget(self.camera_fov)
        camera_buttons.addStretch()
        reset_camera = QPushButton("重置视角")
        reset_camera.clicked.connect(self._reset_world_camera)
        camera_buttons.addWidget(reset_camera)
        camera_layout.addLayout(camera_buttons)
        self.camera_pose = QLabel()
        self.camera_pose.setWordWrap(True)
        camera_layout.addWidget(self.camera_pose)
        help_text = QLabel(
            "左键旋转 · 右键平移 · 滚轮/中键缩放 · Ctrl+滚轮调整 FOV · Shift+拖动切换水平操作 · 双击重置"
        )
        help_text.setWordWrap(True)
        camera_layout.addWidget(help_text)
        left_layout.addWidget(self.camera_controls)
        self.camera_controls.hide()
        telemetry = QFormLayout()
        self.status_labels = {}
        for key, title in (
            ("control", "控制权"),
            ("flight", "无人机状态"),
            ("agent", "Agent 状态"),
            ("time", "仿真时间"),
            ("position", "世界位置 NWU / m"),
            ("yaw", "Yaw / rad"),
        ):
            label = QLabel("—")
            telemetry.addRow(title, label)
            self.status_labels[key] = label
        left_layout.addLayout(telemetry)
        splitter.addWidget(left)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        self.tabs = QTabWidget()
        right_layout.addWidget(self.tabs)
        conversation = QWidget()
        conversation_layout = QVBoxLayout(conversation)
        self.tabs.addTab(conversation, "对话")
        self.chat = QTextBrowser()
        self.chat.setOpenExternalLinks(False)
        conversation_layout.addWidget(self.chat, 1)
        self.input = QPlainTextEdit()
        self.input.setPlaceholderText("输入任务，例如：起飞后向北移动 0.5 米并悬停。")
        self.input.setMaximumHeight(90)
        conversation_layout.addWidget(self.input)
        buttons = QHBoxLayout()
        self.stop_button = QPushButton("终止本轮任务")
        self.stop_button.clicked.connect(
            lambda: self.runner.cancel("用户终止任务，已请求保持当前位置。")
        )
        self.send_button = QPushButton("发送任务")
        self.send_button.clicked.connect(self._send)
        buttons.addWidget(self.stop_button)
        buttons.addStretch()
        buttons.addWidget(self.send_button)
        conversation_layout.addLayout(buttons)

        log_page = QWidget()
        log_layout = QVBoxLayout(log_page)
        hint = QLabel("累计模型输入、原始输出、解析、工具和观测 JSON；RGB 仅记录元数据。")
        hint.setWordWrap(True)
        log_layout.addWidget(hint)
        self.model_output = QPlainTextEdit()
        self.model_output.setReadOnly(True)
        self.model_output.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        log_layout.addWidget(self.model_output, 1)
        log_buttons = QHBoxLayout()
        export = QPushButton("导出 JSONL")
        export.clicked.connect(self._export_log)
        clear = QPushButton("清空日志")
        clear.clicked.connect(self._clear_log)
        log_buttons.addWidget(export)
        log_buttons.addWidget(clear)
        log_layout.addLayout(log_buttons)
        self.tabs.addTab(log_page, "JSON 日志")

        self._build_settings()
        splitter.addWidget(right)
        splitter.setSizes([520, 660])
        layout.addWidget(splitter, 1)
        self.setCentralWidget(root)

    def _build_settings(self):
        page = QWidget()
        layout = QVBoxLayout(page)
        hint = QLabel(
            "修改后点击应用，对下一轮任务生效；运行中不可应用。修改 system prompt 时请保留 JSON 工具协议。最大执行轮数包含格式修正和 FINAL 回复。"
        )
        hint.setWordWrap(True)
        layout.addWidget(hint)
        layout.addWidget(QLabel("System prompt"))
        self.system_prompt = QPlainTextEdit()
        self.system_prompt.setMinimumHeight(220)
        layout.addWidget(self.system_prompt, 1)
        form = QFormLayout()
        self.include_depth = QCheckBox("向模型添加第一视角深度信息")
        form.addRow(self.include_depth)
        self.setting_fields = {}
        for key, title, minimum, maximum, help_text in (
            ("max_decisions", "最大执行轮数", 1, 10000, "每轮任务最多调用模型的次数，包含格式修正及 FINAL。"),
            ("max_new_tokens", "单次生成 token 上限", 1, 32768, "过小可能截断航点 JSON，过大会增加推理开销。"),
            ("max_parse_retries", "JSON 修正重试次数", 0, 100, "连续非法指令允许请求修正的次数，0 表示直接报错。"),
            ("depth_size", "模型深度分辨率（正方形）", 8, 1024, "默认 56×56，独立于模型 RGB 和界面预览分辨率。"),
            (
                "history_messages",
                "历史消息数上限",
                1,
                10000,
                "历史 user/assistant 消息数目标上限，不含 system prompt；JSON 修正时始终保留当前观测及后续消息。",
            ),
            (
                "max_context_chars",
                "上下文字符上限",
                1,
                10000000,
                "历史文本字符数上限；不是模型 token 数，也不包含 system prompt 和图像。",
            ),
        ):
            field = QSpinBox()
            field.setRange(minimum, maximum)
            field.setToolTip(help_text)
            self.setting_fields[key] = field
            form.addRow(title, field)
        layout.addLayout(form)
        self.include_depth.toggled.connect(self.setting_fields["depth_size"].setEnabled)
        buttons = QHBoxLayout()
        defaults = QPushButton("恢复默认值")
        defaults.clicked.connect(lambda: self._show_config(RunnerConfig()))
        self.apply_settings = QPushButton("应用设置")
        self.apply_settings.clicked.connect(self._apply_settings)
        buttons.addWidget(defaults)
        buttons.addWidget(self.apply_settings)
        layout.addLayout(buttons)
        self.settings_status = QLabel("设置仅保存在本次运行中。")
        self.settings_status.setWordWrap(True)
        layout.addWidget(self.settings_status)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(page)
        self.tabs.addTab(scroll, "Agent 设置")
        self._show_config(self.runner.get_config())

    def _show_config(self, config):
        self.system_prompt.setPlainText(config.system_prompt)
        self.include_depth.setChecked(config.include_depth)
        self.setting_fields["depth_size"].setEnabled(config.include_depth)
        for key, field in self.setting_fields.items():
            field.setValue(getattr(config, key))

    def _apply_settings(self):
        try:
            config = RunnerConfig(
                system_prompt=self.system_prompt.toPlainText(),
                include_depth=self.include_depth.isChecked(),
                **{key: field.value() for key, field in self.setting_fields.items()},
            )
            self.runner.configure(config)
            self.settings_status.setText("已应用，下一轮任务使用新设置。")
            self._append_chat("system", "Agent 设置已更新，对下一轮任务生效。")
        except (ValueError, RuntimeError) as exc:
            self.settings_status.setText(str(exc))

    def _clear_log(self):
        self._log_records.clear()
        self.model_output.clear()

    def _export_log(self):
        path, _ = QFileDialog.getSaveFileName(
            self, "导出完整日志", "agent_log.jsonl", "JSON Lines (*.jsonl)"
        )
        if path:
            try:
                with open(path, "w", encoding="utf-8") as stream:
                    for record in self._log_records:
                        stream.write(json.dumps(record, ensure_ascii=False) + "\n")
            except OSError as exc:
                self._append_chat("error", f"日志导出失败：{exc}")

    def _browse(self):
        path = QFileDialog.getExistingDirectory(self, "选择本地模型目录")
        if path:
            self.model_path.setText(path)

    def _backend_changed(self):
        remote = self.backend_selector.currentData() == "deepseek"
        self.api_panel.setVisible(remote)
        self.browse_button.setVisible(not remote)
        self.load_button.setText("配置 API" if remote else "加载模型")
        self.model_path.setText(
            "deepseek-flash" if remote else "mistralai/Ministral-3-3B-Instruct-2512"
        )
        self.model_path.setPlaceholderText(
            "DeepSeek 模型名（需支持图像）" if remote else "Hugging Face 缓存 ID 或本地模型目录"
        )
        if remote and self.runner.get_config().max_new_tokens == 384:
            self.runner.configure(
                replace(self.runner.get_config(), max_new_tokens=4096)
            )
            self.setting_fields["max_new_tokens"].setValue(4096)
            self.settings_status.setText(
                "已将本地默认 384 token 调整为 DeepSeek 建议的 4096；其他设置不变。"
            )

    def _load_model(self):
        try:
            spec = self.model_path.text().strip()
            if self.backend_selector.currentData() == "deepseek":
                spec = DeepSeekConfig(
                    model=spec,
                    base_url=self.api_url.text().strip(),
                    api_key=self.api_key.text().strip(),
                    thinking=self.api_thinking.isChecked(),
                    reasoning_effort=self.api_effort.currentText(),
                    timeout_seconds=self.api_timeout.value(),
                )
            self.runner.load_model(spec)
            self._refresh()
        except ValueError as exc:
            self._append_chat("error", str(exc))

    def _send(self):
        text = self.input.toPlainText().strip()
        if text:
            self.runner.submit(text)
            self.input.clear()

    def _switch_camera(self, direction):
        self._select_camera((self._camera_index + direction) % len(self.cameras))

    def _select_camera(self, index):
        self._camera_index = index
        self.view_tabs.blockSignals(True)
        self.view_tabs.setCurrentIndex(index)
        self.view_tabs.blockSignals(False)
        camera, title = self.cameras[self._camera_index]
        self.camera_label.setText(title)
        self.preview.set_interactive(camera == "world_camera")
        self.camera_controls.setVisible(camera == "world_camera")
        self.depth_legend.setVisible(camera == "depth_camera")
        self.preview.clear()
        self.preview.setText("切换相机……")
        try:
            future = self.runtime.set_camera(camera)
            future.add_done_callback(self._camera_changed)
        except Exception as exc:
            self._append_chat("error", str(exc))

    def _move_world_camera(self, action, dx, dy):
        try:
            self.runtime.move_world_camera(action, dx, dy).add_done_callback(
                self._camera_changed
            )
        except Exception as exc:
            self._append_chat("error", str(exc))

    def _set_world_fov(self, value):
        try:
            self.runtime.set_world_camera_fov(value).add_done_callback(
                self._camera_changed
            )
        except Exception as exc:
            self._append_chat("error", str(exc))

    def _reset_world_camera(self):
        try:
            self.runtime.reset_world_camera().add_done_callback(self._camera_changed)
        except Exception as exc:
            self._append_chat("error", str(exc))

    def _camera_changed(self, future):
        if future.exception() is not None:
            from runtime.events import RuntimeEvent

            self._bridge.received.emit(
                RuntimeEvent("chat", {"role": "error", "text": str(future.exception())})
            )

    def _append_chat(self, role, text):
        titles = {
            "user": "你",
            "assistant": "Agent",
            "error": "运行错误",
            "tool": "工具",
            "system": "系统",
        }
        color = {"error": "#c0392b", "user": "#2376b7", "tool": "#8e44ad"}.get(
            role, "#35495e"
        )
        self.chat.append(
            f'<p><b style="color:{color}">{titles.get(role, role)}</b><br>{html.escape(text).replace(chr(10), "<br>")}</p>'
        )
        self.chat.verticalScrollBar().setValue(self.chat.verticalScrollBar().maximum())

    @Slot(object)
    def _on_event(self, event):
        if self._closing:
            return
        if event.kind == "chat":
            self._append_chat(event.data["role"], event.data["text"])
        elif event.kind == "agent_log":
            self._log_records.append(event.data)
            self.model_output.appendPlainText(
                json.dumps(event.data, ensure_ascii=False, indent=2)
            )

    def _refresh(self):
        if self._closing:
            if self._shutdown_done.is_set():
                if self._shutdown_error:
                    self._closing = False
                    self.setEnabled(True)
                    self._append_chat("error", self._shutdown_error)
                    self._shutdown_error = None
                    self._shutdown_done.clear()
                else:
                    self.close()
            return
        state = self.runtime.get_status()
        model = self.runner.get_status()
        labels = {
            "unloaded": "未加载",
            "loading": "加载中",
            "ready": "就绪",
            "unloading": "卸载中",
            "error": "错误",
        }
        self.model_status.setText(
            "模型：" + labels.get(model["model_state"], model["model_state"])
        )
        busy = model["busy"]
        editable = not busy and model["model_state"] != "ready"
        self.backend_selector.setEnabled(editable)
        self.model_path.setEnabled(editable)
        self.browse_button.setEnabled(editable)
        self.api_panel.setEnabled(editable)
        self.apply_settings.setEnabled(not busy)
        self.load_button.setEnabled(not busy and model["model_state"] != "ready")
        self.unload_button.setEnabled(not busy and model["model_state"] == "ready")
        self.send_button.setEnabled(not busy)
        self.input.setEnabled(not busy)
        self.stop_button.setEnabled(model["running_turn"])
        source = state.get("authority_source")
        if source != self._authority_source:
            self._authority_source = source
            if source == "no_joystick":
                self._append_chat(
                    "system", "启动时未检测到手柄，已自动启用 Agent 控制。加载模型后即可发送任务，使用“终止本轮任务”按钮可停止。"
                )
        self.status_labels["control"].setText(
            "AGENT / 无手柄自动授权"
            if source == "no_joystick" and state.get("authorized")
            else ("AGENT / Axis4=1" if state.get("authorized") else "MANUAL / Axis4=0")
        )
        self.status_labels["flight"].setText(state.get("flight_state", "—"))
        self.status_labels["agent"].setText(state.get("agent_state", "—").upper())
        self.status_labels["time"].setText(f'{state.get("sim_time", 0.):.2f} s')
        camera_state = state.get("world_camera")
        if camera_state is not None:
            self.camera_fov.blockSignals(True)
            self.camera_fov.setValue(round(camera_state["fov"]))
            self.camera_fov.blockSignals(False)
            position = ", ".join(f"{value:.2f}" for value in camera_state["position"])
            self.camera_pose.setText(
                f'相机位置 ({position}) m · Az {camera_state["azimuth"]:.1f}° · El {camera_state["elevation"]:.1f}° · 距离 {camera_state["distance"]:.2f} m'
            )
        if "position" in state:
            self.status_labels["position"].setText(
                " / ".join(f"{v:.3f}" for v in state["position"])
            )
            self.status_labels["yaw"].setText(f'{state["yaw"]:.3f}')
        if state.get("camera") == self.cameras[self._camera_index][0]:
            image = self.runtime.get_preview()
            if image is not None:
                data = image.convert("RGB").tobytes()
                qimage = QImage(
                    data,
                    image.width,
                    image.height,
                    image.width * 3,
                    QImage.Format.Format_RGB888,
                ).copy()
                pixel_ratio = self.preview.devicePixelRatioF()
                pixmap = QPixmap.fromImage(qimage).scaled(
                    self.preview.size() * pixel_ratio,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation,
                )
                pixmap.setDevicePixelRatio(pixel_ratio)
                self.preview.setPixmap(pixmap)

    def _shutdown(self):
        try:
            self.runner.close()
            self.runtime.close()
        except Exception as exc:
            self._shutdown_error = str(exc)
        finally:
            self._shutdown_done.set()

    def closeEvent(self, event):
        if self._shutdown_done.is_set() and self._shutdown_error is None:
            self._timer.stop()
            event.accept()
        else:
            event.ignore()
            if not self._closing:
                self._closing = True
                self.setEnabled(False)
                self.model_status.setText("正在终止任务并释放资源……")
                threading.Thread(
                    target=self._shutdown, name="desktop-shutdown", daemon=True
                ).start()
