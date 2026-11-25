"""Qwen (DashScope OpenAI-compatible) adapter; existing agent JSON tools remain local."""
from .backends import QwenConfig
from .openai_compat import OpenAICompatBackend, image_messages


class QwenBackend(OpenAICompatBackend):
    provider_name = "Qwen"
    env_api_key = "DASHSCOPE_API_KEY"
    config_type = QwenConfig

    def _build_payload(self, prepared, max_new_tokens):
        # DashScope rejects response_format json_object when enable_thinking
        # is true, so JSON mode implies non-thinking mode. In thinking mode
        # the JSON system prompt and the runner's parse-retry loop enforce
        # the response format instead.
        payload = {
            "model": self.config.model.strip(),
            "messages": prepared,
            "max_tokens": max_new_tokens,
            "stream": False,
            "enable_thinking": bool(self.config.thinking),
        }
        if not self.config.thinking:
            payload["response_format"] = {"type": "json_object"}
        return payload
