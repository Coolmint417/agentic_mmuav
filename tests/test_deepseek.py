"""DeepSeek wire protocol, image ownership and cancellation without paid calls."""
import asyncio
import base64
from io import BytesIO
import json
import threading
import unittest
from unittest.mock import patch

import httpx
from PIL import Image

from agent.backends import DeepSeekConfig, load_backend
from agent.deepseek import DeepSeekBackend
from agent.protocol import GenerationCancelled
from tests.test_runner import running_system, wait_until


def response(text='{"final":"完成"}', **kwargs):
    return httpx.Response(
        200,
        json={
            "model": "deepseek-flash",
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


class DeepSeekTests(unittest.TestCase):
    def backend(self, handler, **kwargs):
        return DeepSeekBackend(
            DeepSeekConfig(api_key="test-secret", **kwargs),
            transport=httpx.MockTransport(handler),
        )

    def test_image_binding_json_thinking_and_no_history_mutation(self):
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
        self.assertEqual(str(request.url), "https://api.deepseek.com/chat/completions")
        self.assertEqual(request.headers["Authorization"], "Bearer test-secret")
        self.assertEqual(data["thinking"], {"type": "enabled"})
        self.assertEqual(data["reasoning_effort"], "high")
        self.assertEqual(data["response_format"], {"type": "json_object"})
        self.assertNotIn("tools", data)
        self.assertNotIn("temperature", data)
        self.assertEqual(data["max_tokens"], 4096)
        blocks = data["messages"][1]["content"]
        image = Image.open(
            BytesIO(base64.b64decode(blocks[1]["image_url"]["url"].split(",", 1)[1]))
        )
        self.assertEqual(image.size, (32, 24))
        self.assertEqual(data["messages"][-1]["content"], "repair JSON")
        self.assertEqual(messages[1]["content"], "task and observation")
        self.assertEqual(result["text"], '{"final":"完成"}')
        self.assertEqual(result["usage"]["completion_tokens"], 20)
        self.assertNotIn("test-secret", repr(backend.config))

    def test_disabled_thinking_env_key_and_unload(self):
        calls = []

        def handler(request):
            calls.append(json.loads(request.content))
            return response()

        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": "env-key"}):
            configured = load_backend(DeepSeekConfig())
            self.assertIsInstance(configured, DeepSeekBackend)
            configured.unload()
            backend = DeepSeekBackend(
                DeepSeekConfig(thinking=False), transport=httpx.MockTransport(handler)
            )
        backend.generate("task")
        self.assertEqual(calls[0]["thinking"], {"type": "disabled"})
        self.assertNotIn("reasoning_effort", calls[0])
        self.assertIn("JSON", calls[0]["messages"][0]["content"])
        backend.unload()
        with self.assertRaises(RuntimeError):
            backend.generate("task")
        with patch.dict("os.environ", {"DEEPSEEK_API_KEY": ""}), self.assertRaises(
            ValueError
        ):
            DeepSeekBackend()

    def test_http_errors_empty_truncated_and_malformed_responses(self):
        for status in (400, 401, 402, 429, 500):
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

    def test_cancel_before_and_during_request_and_total_timeout(self):
        entered, exited, cancel = (
            threading.Event(),
            threading.Event(),
            threading.Event(),
        )

        async def handler(request):
            entered.set()
            try:
                await asyncio.Future()
            finally:
                exited.set()

        backend = self.backend(handler)
        cancel.set()
        with self.assertRaises(GenerationCancelled):
            backend.generate("JSON task", cancel_event=cancel)
        self.assertFalse(entered.is_set())
        cancel.clear()
        errors = []

        def generate():
            try:
                backend.generate("JSON task", cancel_event=cancel)
            except Exception as exc:
                errors.append(exc)

        worker = threading.Thread(target=generate)
        worker.start()
        try:
            self.assertTrue(entered.wait(2))
            cancel.set()
            worker.join(2)
            self.assertFalse(worker.is_alive())
            self.assertTrue(exited.is_set())
            self.assertIsInstance(errors[0], GenerationCancelled)
        finally:
            cancel.set()
            worker.join(2)
        with self.assertRaises(TimeoutError):
            self.backend(handler, timeout_seconds=0.05).generate("JSON task")

    def test_runner_feedback_usage_logging_and_authority(self):
        calls = []

        def handler(request):
            calls.append(json.loads(request.content))
            return response(
                '{"tool":"land"}' if len(calls) == 1 else '{"final":"尚未起飞"}'
            )

        backend = self.backend(handler)
        with running_system(backend) as (runtime, runner, remote, events):
            runner.submit("降落")
            wait_until(lambda: not runner.get_status()["busy"])
            self.assertEqual(len(calls), 2)
            feedback = json.loads(calls[1]["messages"][-1]["content"][0]["text"])
            self.assertEqual(feedback["tool_result"]["status"], "error")
            self.assertTrue(
                all("reasoning_content" not in m for m in calls[1]["messages"])
            )
            self.assertTrue(
                any(
                    e.kind == "agent_log"
                    and e.data["kind"] == "model_output"
                    and e.data["data"]["metadata"]["usage"]["completion_tokens"] == 20
                    for e in events
                )
            )
            self.assertNotIn("test-secret", str(events))
        calls.clear()
        with running_system(self.backend(handler), authorized=False) as (
            runtime,
            runner,
            remote,
            events,
        ):
            self.assertFalse(runner.submit("起飞"))
            self.assertEqual(calls, [])

    def test_axis4_revocation_aborts_http_and_no_tool_runs(self):
        entered, exited = threading.Event(), threading.Event()

        async def handler(request):
            entered.set()
            try:
                await asyncio.Future()
            finally:
                exited.set()

        with running_system(self.backend(handler), realtime=True) as (
            runtime,
            runner,
            remote,
            events,
        ):
            runner.submit("起飞")
            self.assertTrue(entered.wait(10))
            start = runtime.get_status()["sim_time"]
            wait_until(lambda: runtime.get_status()["sim_time"] > start + 0.05)
            remote.axes[4] = 0.0
            wait_until(lambda: not runner.get_status()["busy"])
            self.assertTrue(exited.is_set())
            self.assertEqual(runtime.get_status()["agent_state"], "end")
            self.assertFalse(
                any(
                    e.kind == "agent_log" and e.data["kind"] == "tool_call"
                    for e in events
                )
            )
