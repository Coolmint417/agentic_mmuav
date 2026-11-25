"""MiMo Chat Completions adapter; existing agent JSON tools remain local."""
from .backends import MiMoConfig
from .openai_compat import OpenAICompatBackend, image_messages


class MiMoBackend(OpenAICompatBackend):
    provider_name = "MiMo"
    env_api_key = "MIMO_API_KEY"
    config_type = MiMoConfig

    def _build_payload(self, prepared, max_new_tokens):
        # MiMo follows the OpenAI-compatible surface but uses
        # max_completion_tokens and does not accept reasoning_effort.
        return {
            "model": self.config.model.strip(),
            "messages": prepared,
            "max_completion_tokens": max_new_tokens,
            "stream": False,
            "response_format": {"type": "json_object"},
            "thinking": {"type": "enabled" if self.config.thinking else "disabled"},
        }
