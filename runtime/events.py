"""Framework-independent events for desktop and headless clients."""
from dataclasses import dataclass
import threading
from typing import Any


@dataclass(frozen=True)
class RuntimeEvent:
    kind: str
    data: Any


class EventBus:
    def __init__(self):
        self._lock = threading.Lock()
        self._listeners = []

    def subscribe(self, callback):
        with self._lock:
            self._listeners.append(callback)

    def publish(self, kind, data):
        with self._lock:
            listeners = tuple(self._listeners)
        for listener in listeners:
            try:
                listener(RuntimeEvent(kind, data))
            except Exception:
                # Presentation errors must not interrupt vehicle control.
                continue


def describe_error(error):
    text = str(error)
    if (
        isinstance(error, MemoryError)
        or "outofmemory" in type(error).__name__.lower()
        or "out of memory" in text.lower()
    ):
        return "内存/显存不足（OOM）。本轮已停止，可降低图像分辨率、上下文或生成长度后重试。\n" + text
    return f"{type(error).__name__}: {text}"
