"""DeepSeek Chat Completions adapter; existing agent JSON tools remain local."""
from .backends import DeepSeekConfig
from .openai_compat import OpenAICompatBackend, image_messages


class DeepSeekBackend(OpenAICompatBackend):
    provider_name = "DeepSeek"
    env_api_key = "DEEPSEEK_API_KEY"
    config_type = DeepSeekConfig

    def _build_payload(self, prepared, max_new_tokens):
        payload = {
            "model": self.config.model.strip(),
            "messages": prepared,
            "max_tokens": max_new_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
            "thinking": {"type": "enabled" if self.config.thinking else "disabled"},
        }
        if self.config.thinking:
            payload["reasoning_effort"] = self.config.reasoning_effort
        return payload
