"""Shared OpenAI-compatible Chat Completions transport for remote backends."""
import asyncio
import base64
from io import BytesIO
import os
import time

import httpx

from .protocol import GenerationCancelled
from .messages import image_groups


def image_messages(messages, image, index, extra_images=None, message_images=None):
    result = [dict(message) for message in messages]
    for position, attachments in image_groups(
            messages, image, index, extra_images, message_images).items():
        blocks = [{"type": "text", "text": result[position]["content"]}]
        for name, attachment in attachments.items():
            buffer = BytesIO()
            attachment.convert("RGB").save(buffer, format="PNG")
            url = "data:image/png;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")
            if name != "rgb":
                blocks.append({
                    "type": "text",
                    "text": f"Image attachment: {name}. Interpret using observation metadata.",
                })
            blocks.append({"type": "image_url", "image_url": {"url": url}})
        result[position]["content"] = blocks
    return result


class OpenAICompatBackend:
    """Common HTTP, cancellation, parsing and error handling for chat APIs."""

    provider_name = "API"
    env_api_key = ""
    config_type = None

    def __init__(self, config=None, *, transport=None):
        self.config = config or self.config_type()
        self._api_key = (
            self.config.api_key or os.environ.get(self.env_api_key, "")
        ).strip()
        if not self._api_key:
            raise ValueError(
                f"请填写 {self.provider_name} API Key，或设置 {self.env_api_key} 环境变量。"
            )
        if not self._api_key.isascii() or any(
            character.isspace() for character in self._api_key
        ):
            raise ValueError("API Key 必须是无空白的 ASCII 字符串。")
        # Optional MockTransport for tests. No request is made during setup.
        self._transport = transport
        self._closed = False

    def _build_payload(self, prepared, max_new_tokens):
        raise NotImplementedError

    async def _request(self, payload, cancel_event):
        provider = self.provider_name
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
                        raise GenerationCancelled(
                            f"{provider} 请求已取消，后续输出将被丢弃。"
                        )
                    if time.monotonic() >= deadline:
                        raise TimeoutError(
                            f"{provider} 请求超时，请稍后重试或增大 API 超时。"
                        )
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
        message_images=None,
        max_new_tokens=4096,
        cancel_event=None,
    ):
        provider = self.provider_name
        if self._closed:
            raise RuntimeError(f"{provider} 后端已关闭。")
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
        prepared = image_messages(
            messages, image, image_message_index, extra_images, message_images
        )
        if not any("json" in str(m.get("content", "")).lower() for m in messages):
            prepared.insert(
                0,
                {
                    "role": "system",
                    "content": "Return one JSON object using the required agent decision schema.",
                },
            )
        payload = self._build_payload(prepared, max_new_tokens)
        started = time.perf_counter()
        try:
            response = asyncio.run(self._request(payload, cancel_event))
        except httpx.TimeoutException:
            raise RuntimeError(f"{provider} 网络请求超时。") from None
        except httpx.RequestError:
            raise RuntimeError(
                f"无法连接 {provider} API，请检查 API 地址和网络。"
            ) from None
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
                f"{provider} HTTP {response.status_code}："
                f"{hints.get(response.status_code, '服务错误，请稍后重试')}。"
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
            raise RuntimeError(f"{provider} 返回了无法识别的响应格式。") from None
        if finish_reason == "length":
            raise RuntimeError(
                f"{provider} 达到生成 token 上限，未执行输出；请在 Agent 设置中提高单次生成 token 上限，或关闭思考模式。"
            )
        if not text.strip():
            raise RuntimeError(
                f"{provider} 返回空 content，请增大生成 token 上限或调整提示词后重试。"
            )
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
