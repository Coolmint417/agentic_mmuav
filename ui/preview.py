"""Qt mouse input for the desktop world camera; no MuJoCo access here."""
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import QLabel, QSizePolicy


class CameraPreview(QLabel):
    camera_move = Signal(str, float, float)
    fov_steps = Signal(float)
    reset_camera = Signal()

    def __init__(self, text=""):
        super().__init__(text)
        self._interactive = False
        self._last = None
        self.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)

    def set_interactive(self, enabled):
        self._interactive = enabled
        self._last = None
        self.setCursor(
            Qt.CursorShape.OpenHandCursor if enabled else Qt.CursorShape.ArrowCursor
        )

    def _inside_image(self, position):
        pixmap = self.pixmap()
        if pixmap is None or pixmap.isNull():
            return False
        size = pixmap.deviceIndependentSize()
        left, top = (self.width() - size.width()) / 2, (
            self.height() - size.height()
        ) / 2
        return (
            left <= position.x() < left + size.width()
            and top <= position.y() < top + size.height()
        )

    def mousePressEvent(self, event):
        if (
            self._interactive
            and self._inside_image(event.position())
            and event.button()
            in (
                Qt.MouseButton.LeftButton,
                Qt.MouseButton.RightButton,
                Qt.MouseButton.MiddleButton,
            )
        ):
            self._last = event.position()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()
        else:
            super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if not self._interactive or self._last is None:
            return super().mouseMoveEvent(event)
        delta = event.position() - self._last
        self._last = event.position()
        shift = bool(event.modifiers() & Qt.KeyboardModifier.ShiftModifier)
        if event.buttons() & Qt.MouseButton.RightButton:
            action = "move_horizontal" if shift else "move"
        elif event.buttons() & Qt.MouseButton.LeftButton:
            action = "rotate_horizontal" if shift else "rotate"
        elif event.buttons() & Qt.MouseButton.MiddleButton:
            action = "zoom"
        else:
            return
        pixmap = self.pixmap()
        scale = max(
            1,
            pixmap.deviceIndependentSize().height()
            if pixmap and not pixmap.isNull()
            else self.height(),
        )
        self.camera_move.emit(action, delta.x() / scale, delta.y() / scale)
        event.accept()

    def mouseReleaseEvent(self, event):
        self._last = None
        if self._interactive:
            self.setCursor(Qt.CursorShape.OpenHandCursor)
        super().mouseReleaseEvent(event)

    def wheelEvent(self, event):
        if self._interactive and self._inside_image(event.position()):
            steps = (
                event.angleDelta().y() / 120
                if event.angleDelta().y()
                else event.pixelDelta().y() / 40
            )
            if event.modifiers() & Qt.KeyboardModifier.ControlModifier:
                self.fov_steps.emit(steps)
            else:
                self.camera_move.emit("zoom", 0, -0.05 * steps)
            event.accept()
        else:
            event.ignore()

    def mouseDoubleClickEvent(self, event):
        if (
            self._interactive
            and self._inside_image(event.position())
            and event.button() == Qt.MouseButton.LeftButton
        ):
            self._last = None
            self.reset_camera.emit()
            event.accept()
        else:
            super().mouseDoubleClickEvent(event)
