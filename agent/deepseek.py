"""DeepSeek Chat Completions adapter; existing agent JSON tools remain local."""
import asyncio
import base64
from io import BytesIO
import os
import time

import httpx

from .backends import DeepSeekConfig
from .protocol import GenerationCancelled


def image_messages(messages, image, index, extra_images=None):
    result = [dict(message) for message in messages]
    if image is None:
        return result
    index = len(result) - 1 if index is None else index
    if (
        isinstance(index, bool)
        or not isinstance(index, int)
        or not 0 <= index < len(result)
        or result[index].get("role") != "user"
    ):
        raise ValueError("图片必须绑定到对应的 user 观测消息。")
    buffer = BytesIO()
    image.convert("RGB").save(buffer, format="PNG")
    url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
    result[index]["content"] = [
        {"type": "text", "text": result[index]["content"]},
        {"type": "image_url", "image_url": {"url": url}},
    ]
    for name, attachment in (extra_images or {}).items():
        buffer = BytesIO()
        attachment.convert("RGB").save(buffer, format="PNG")
        url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode(
            "ascii"
        )
        result[index]["content"].extend(
            [
                {
                    "type": "text",
                    "text": f"Image attachment: {name}. Interpret using observation metadata.",
                },
                {"type": "image_url", "image_url": {"url": url}},
            ]
        )
    return result


class DeepSeekBackend:
    def __init__(self, config=None, *, transport=None):
        self.config = config or DeepSeekConfig()
        self._api_key = (
            self.config.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
        ).strip()
        if not self._api_key:
            raise ValueError("请填写 DeepSeek API Key，或设置 DEEPSEEK_API_KEY 环境变量。")
        if not self._api_key.isascii() or any(
            character.isspace() for character in self._api_key
        ):
            raise ValueError("API Key 必须是无空白的 ASCII 字符串。")
        # Optional MockTransport for tests. No request is made during setup.
        self._transport = transport
        self._closed = False

    async def _request(self, payload, cancel_event):
        async with httpx.AsyncClient(
            timeout=self.config.timeout_seconds, transport=self._transport
        ) as client:
            task = asyncio.create_task(
                client.post(
                    self.config.base_url.rstrip("/") + "/chat/completions",
                    headers={"Authorization": "Bearer " + self._api_key},
                    json=payload,
                )
            )
            try:
                deadline = time.monotonic() + self.config.timeout_seconds
                while not task.done():
                    if cancel_event is not None and cancel_event.is_set():
                        raise GenerationCancelled("DeepSeek 请求已取消，后续输出将被丢弃。")
                    if time.monotonic() >= deadline:
                        raise TimeoutError("DeepSeek 请求超时，请稍后重试或增大 API 超时。")
                    await asyncio.wait({task}, timeout=0.05)
                return await task
            finally:
                if not task.done():
                    task.cancel()
                await asyncio.gather(task, return_exceptions=True)

    def generate(
        self,
        prompt=None,
        *,
        system=None,
        messages=None,
        image=None,
        extra_images=None,
        image_message_index=None,
        max_new_tokens=4096,
        cancel_event=None,
    ):
        if self._closed:
            raise RuntimeError("DeepSeek 后端已关闭。")
        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelled("请求已取消。")
        if messages is None:
            if not isinstance(prompt, str):
                raise ValueError("prompt 必须是字符串。")
            messages = ([{"role": "system", "content": system}] if system else []) + [
                {"role": "user", "content": prompt}
            ]
        elif prompt is not None or system is not None:
            raise ValueError("messages 与 prompt/system 不能同时提供。")
        if not messages or messages[-1].get("role") != "user":
            raise ValueError("请求必须以 user 消息结束。")
        prepared = image_messages(messages, image, image_message_index, extra_images)
        if not any("json" in str(m.get("content", "")).lower() for m in messages):
            prepared.insert(
                0,
                {
                    "role": "system",
                    "content": "Return one JSON object using the required agent decision schema.",
                },
            )
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
        started = time.perf_counter()
        try:
            response = asyncio.run(self._request(payload, cancel_event))
        except httpx.TimeoutException:
            raise RuntimeError("DeepSeek 网络请求超时。") from None
        except httpx.RequestError:
            raise RuntimeError("无法连接 DeepSeek API，请检查 API 地址和网络。") from None
        if cancel_event is not None and cancel_event.is_set():
            raise GenerationCancelled("请求已取消，输出已丢弃。")
        if response.is_error:
            hints = {
                401: "API Key 无效",
                402: "余额不足",
                403: "无访问权限",
                429: "请求限流",
                400: "请求参数或模型不支持当前图像/思考设置",
            }
            raise RuntimeError(
                f"DeepSeek HTTP {response.status_code}：{hints.get(response.status_code, '服务错误，请稍后重试')}。"
            )
        try:
            data = response.json()
            choice = data["choices"][0]
            message = choice["message"]
            if not isinstance(message, dict):
                raise ValueError()
            text = message.get("content") or ""
            finish_reason = choice.get("finish_reason")
            if not isinstance(text, str):
                raise ValueError()
        except (ValueError, KeyError, IndexError, TypeError):
            raise RuntimeError("DeepSeek 返回了无法识别的响应格式。") from None
        if finish_reason == "length":
            raise RuntimeError(
                "DeepSeek 达到生成 token 上限，未执行输出；请在 Agent 设置中提高单次生成 token 上限，或关闭思考模式。"
            )
        if not text.strip():
            raise RuntimeError("DeepSeek 返回空 content，请增大生成 token 上限或调整提示词后重试。")
        return {
            "text": text.strip(),
            "model": data.get("model", self.config.model),
            "finish_reason": finish_reason,
            "usage": data.get("usage", {}),
            "seconds": time.perf_counter() - started,
            "reasoning_content": message.get("reasoning_content") or "",
        }

    def unload(self):
        self._closed = True
        self._api_key = ""
