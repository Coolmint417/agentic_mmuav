"""Backend selection without importing GPU or HTTP inference dependencies."""
from dataclasses import dataclass, field
import math
from urllib.parse import urlparse


@dataclass(frozen=True)
class DeepSeekConfig:
    model: str = "deepseek-flash"
    base_url: str = "https://api.deepseek.com"
    api_key: str = field(default="", repr=False)
    thinking: bool = True
    reasoning_effort: str = "high"
    timeout_seconds: float = 180.0

    def __post_init__(self):
        if not isinstance(self.api_key, str):
            raise ValueError("API Key 必须是字符串。")
        if not isinstance(self.model, str) or not self.model.strip():
            raise ValueError("DeepSeek 模型名不能为空。")
        url = urlparse(self.base_url)
        if (
            url.scheme not in {"http", "https"}
            or not url.netloc
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("请输入有效的 API base URL（不包含密钥）。")
        if not isinstance(self.thinking, bool) or self.reasoning_effort not in {
            "low",
            "high",
            "max",
        }:
            raise ValueError("无效的 DeepSeek 思考设置。")
        if (
            isinstance(self.timeout_seconds, bool)
            or not isinstance(self.timeout_seconds, (int, float))
            or not math.isfinite(self.timeout_seconds)
            or self.timeout_seconds <= 0
        ):
            raise ValueError("API 超时必须是正数。")


def load_backend(spec):
    if isinstance(spec, DeepSeekConfig):
        from .deepseek import DeepSeekBackend

        return DeepSeekBackend(spec)
    from .ministral import MinistralBackend

    return MinistralBackend(spec)
