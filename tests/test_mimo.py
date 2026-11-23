"""MiMo wire protocol via MockTransport; no real API calls."""
import base64
from io import BytesIO
import json
import unittest
from unittest.mock import patch

import httpx
from PIL import Image

from agent.backends import MiMoConfig, load_backend
from agent.mimo import MiMoBackend


def response(text='{"final":"完成"}', **kwargs):
    return httpx.Response(
        200,
        json={
            "model": "mimo-v2.6-pro",
            "choices": [
                {
                    "message": {
                        "content": text,
                        "reasoning_content": "model reasoning",
                    },
                    "finish_reason": kwargs.get("finish_reason", "stop"),
                }
            ],
            "usage": {"prompt_tokens": 300, "completion_tokens": 20},
        },
    )


class MiMoTests(unittest.TestCase):
    def backend(self, handler, **kwargs):
        return MiMoBackend(
            MiMoConfig(api_key="test-secret", **kwargs),
            transport=httpx.MockTransport(handler),
        )

    def test_payload_uses_max_completion_tokens_and_thinking(self):
        requests = []

        def handler(request):
            requests.append(request)
            return response()

        backend = self.backend(handler)
        messages = [
            {"role": "system", "content": "Return JSON"},
            {"role": "user", "content": "task and observation"},
            {"role": "assistant", "content": "bad JSON"},
            {"role": "user", "content": "repair JSON"},
        ]
        result = backend.generate(
            messages=messages,
            image=Image.new("RGB", (32, 24)),
            image_message_index=1,
            max_new_tokens=4096,
        )
        request = requests[0]
        data = json.loads(request.content)
        self.assertEqual(
            str(request.url), "https://api.xiaomimimo.com/v1/chat/completions"
        )
        self.assertEqual(request.headers["Authorization"], "Bearer test-secret")
        self.assertEqual(data["max_completion_tokens"], 4096)
        self.assertNotIn("max_tokens", data)
        self.assertNotIn("reasoning_effort", data)
        self.assertEqual(data["thinking"], {"type": "enabled"})
        self.assertEqual(data["response_format"], {"type": "json_object"})
        self.assertNotIn("tools", data)
        self.assertNotIn("temperature", data)
        blocks = data["messages"][1]["content"]
        image = Image.open(
            BytesIO(base64.b64decode(blocks[1]["image_url"]["url"].split(",", 1)[1]))
        )
        self.assertEqual(image.size, (32, 24))
        self.assertEqual(data["messages"][-1]["content"], "repair JSON")
        self.assertEqual(messages[1]["content"], "task and observation")
        self.assertEqual(result["text"], '{"final":"完成"}')
        self.assertEqual(result["usage"]["completion_tokens"], 20)
        self.assertEqual(result["reasoning_content"], "model reasoning")
        self.assertNotIn("test-secret", repr(backend.config))

    def test_disabled_thinking_env_key_unload_and_load_backend(self):
        calls = []

        def handler(request):
            calls.append(json.loads(request.content))
            return response()

        with patch.dict("os.environ", {"MIMO_API_KEY": "env-key"}):
            configured = load_backend(MiMoConfig())
            self.assertIsInstance(configured, MiMoBackend)
            configured.unload()
            backend = MiMoBackend(
                MiMoConfig(thinking=False), transport=httpx.MockTransport(handler)
            )
        backend.generate("task")
        self.assertEqual(calls[0]["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", calls[0])
        self.assertIn("JSON", calls[0]["messages"][0]["content"])
        backend.unload()
        with self.assertRaises(RuntimeError):
            backend.generate("task")
        with patch.dict("os.environ", {"MIMO_API_KEY": ""}), self.assertRaises(
            ValueError
        ):
            MiMoBackend()

    def test_http_errors_empty_truncated_and_malformed_responses(self):
        for status in (400, 401, 402, 403, 429, 500):
            with self.subTest(status=status):
                backend = self.backend(
                    lambda r: httpx.Response(status, json={"error": "test-secret"})
                )
                with self.assertRaises(RuntimeError) as error:
                    backend.generate("JSON task")
                self.assertIn(str(status), str(error.exception))
                self.assertNotIn("test-secret", str(error.exception))
        for reply, hint in (
            (response(""), "空 content"),
            (response(finish_reason="length"), "token 上限"),
            (httpx.Response(200, json={}), "响应格式"),
        ):
            with self.subTest(hint=hint), self.assertRaisesRegex(RuntimeError, hint):
                self.backend(lambda r: reply).generate("JSON task")

    def test_config_validation(self):
        for kwargs in (
            {"model": " "},
            {"base_url": "https://user:pass@example.com"},
            {"base_url": "ftp://example.com"},
            {"thinking": "yes"},
            {"timeout_seconds": 0},
            {"timeout_seconds": float("nan")},
        ):
            with self.subTest(**kwargs), self.assertRaises(ValueError):
                MiMoConfig(**kwargs)


if __name__ == "__main__":
    unittest.main()
