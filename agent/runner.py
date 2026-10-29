"""UI-independent model/JSON/tool/observation orchestration; no physics stepping."""
from dataclasses import dataclass
from datetime import datetime, timezone
import json
import queue
import threading

from runtime.events import EventBus, describe_error
from .protocol import GenerationCancelled, SYSTEM_PROMPT, parse_decision
from .backends import DeepSeekConfig, load_backend as _load_backend


@dataclass(frozen=True)
class RunnerConfig:
    max_new_tokens: int = 384
    max_decisions: int = 16
    max_parse_retries: int = 2
    history_messages: int = 12
    max_context_chars: int = 24000
    system_prompt: str = SYSTEM_PROMPT
    include_depth: bool = False
    depth_size: int = 56

    def __post_init__(self):
        if not isinstance(self.system_prompt, str) or not self.system_prompt.strip():
            raise ValueError("System prompt must be a nonempty string")
        for name, value in vars(self).items():
            if name in {"system_prompt", "include_depth"}:
                continue
            if (
                isinstance(value, bool)
                or not isinstance(value, int)
                or value < (0 if name == "max_parse_retries" else 1)
            ):
                raise ValueError(f"Invalid runner setting: {name}")
        if not isinstance(self.include_depth, bool) or not 8 <= self.depth_size <= 1024:
            raise ValueError("深度开关必须为布尔值，深度分辨率需在 8～1024 之间。")


class AgentRunner:
    """One model worker and one active user turn. All env access goes through runtime."""

    def __init__(
        self,
        environment_runtime,
        *,
        events=None,
        config=None,
        backend_factory=_load_backend,
    ):
        self.runtime = environment_runtime
        self.events = events or EventBus()
        self.config = config or RunnerConfig()
        self._factory = backend_factory
        self._jobs = queue.Queue()
        self._lock = threading.Lock()
        self._cancel = threading.Event()
        self._backend = None
        self._busy = False
        self._running_turn = False
        self._closing = False
        self._model_state = "unloaded"
        self._conversation = []
        self._thread = threading.Thread(
            target=self._work, name="agent-runner", daemon=True
        )
        self.runtime.set_cancel_callback(self.cancel)
        self._thread.start()

    def _chat(self, text, role="system"):
        self.events.publish("chat", {"role": role, "text": text})

    def get_status(self):
        with self._lock:
            return {
                "busy": self._busy,
                "running_turn": self._running_turn,
                "model_state": self._model_state,
            }

    def get_config(self):
        with self._lock:
            return self.config

    def configure(self, config):
        """Apply an immutable configuration between jobs, never during a turn."""
        if not isinstance(config, RunnerConfig):
            raise TypeError("Expected RunnerConfig")
        with self._lock:
            if self._busy or self._closing:
                raise RuntimeError("请等待当前任务结束后再应用设置。")
            self.config = config

    def _log(self, kind, turn_id, data):
        # Serialize now: later history compaction must not mutate earlier records.
        record = json.loads(
            json.dumps(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "turn_id": turn_id,
                    "kind": kind,
                    "data": data,
                },
                ensure_ascii=False,
                allow_nan=False,
            )
        )
        self.events.publish("agent_log", record)

    def _enqueue(self, kind, value=None):
        with self._lock:
            if self._closing or self._busy:
                rejected = True
            else:
                rejected = False
                self._busy = True
                self._running_turn = kind == "turn"
                if kind == "turn":
                    self._cancel.clear()
                self._jobs.put((kind, value))
        if rejected:
            self._chat("当前任务尚未结束，请等待或先终止任务。")
        return not rejected

    def load_model(self, path="mistralai/Ministral-3-3B-Instruct-2512"):
        if self.get_status()["model_state"] == "ready":
            self._chat("模型已加载；更换模型前请先卸载。")
            return False
        return self._enqueue("load", path)

    def unload_model(self):
        return self._enqueue("unload")

    def reload_environment(self, scene_id):
        """Serialize map changes with generation; retain backend, discard old history."""
        return self._enqueue("reload", scene_id)

    def submit(self, text):
        text = text.strip()
        if not text:
            return False
        self._chat(text, "user")
        status = self.runtime.get_status()
        if not status.get("ready"):
            self._chat("仿真环境尚未就绪。" + (status.get("error") or ""), "error")
            return False
        if not status.get("authorized"):
            self._chat("Axis4 未授权：请将遥控器 Axis4 切换为 1。本次消息不会发送给模型。")
            return False
        if self.get_status()["model_state"] != "ready":
            self._chat("请先加载模型，本次消息未发送给模型。")
            return False
        return self._enqueue("turn", text)

    def cancel(self, reason="用户终止任务"):
        first = not self._cancel.is_set()
        self._cancel.set()
        self.runtime.cancel_turn(reason)
        if first and self.get_status()["running_turn"]:
            self._chat(reason)

    def _check_cancelled(self):
        if self._cancel.is_set():
            raise GenerationCancelled("本轮已取消，已丢弃模型输出。")

    def _bounded(self, history, *, required_index=None):
        # Keep the current observation with its task/RGB even after JSON retries.
        start = max(0, len(history) - self.config.history_messages)
        if required_index is not None:
            start = min(start, required_index)
        result = history[start:]
        while (
            len(result) > 1
            and (required_index is None or start < required_index)
            and sum(len(item["content"]) for item in result)
            > self.config.max_context_chars
        ):
            result = result[1:]
            start += 1
        if sum(len(item["content"]) for item in result) > self.config.max_context_chars:
            raise ValueError("当前输入或观测过长，请缩短任务描述或观测内容。")
        return result

    @staticmethod
    def _append_observation(history, packet, *, task=None, tool_result=None):
        # Old text observations remain useful; old image attachments do not.
        for item in history:
            if item["role"] == "user":
                try:
                    previous = json.loads(item["content"])
                    observation = (
                        previous.get("observation")
                        if isinstance(previous, dict)
                        else None
                    )
                    if isinstance(observation, dict) and "rgb" in observation:
                        observation["rgb"] = {"not_retained": True}
                        if "depth" in observation:
                            observation["depth"] = {"not_retained": True}
                        item["content"] = json.dumps(previous, ensure_ascii=False)
                except (ValueError, TypeError):
                    pass
        value = {"observation": packet.to_dict(embed_images=False)}
        if task is not None:
            value["task"] = task
        if tool_result is not None:
            value["tool_result"] = tool_result
        history.append(
            {
                "role": "user",
                "content": json.dumps(value, ensure_ascii=False, allow_nan=False),
            }
        )

    def _run_turn(self, text):
        self._check_cancelled()
        (turn_id, packet,) = self.runtime.begin_turn(
            include_depth=self.config.include_depth, depth_size=self.config.depth_size
        )
        history = [dict(item) for item in self._conversation]
        self._append_observation(history, packet, task=text)
        observation_index = len(history) - 1
        self._log("observation", turn_id, json.loads(history[-1]["content"]))
        parse_failures = 0
        for _ in range(self.config.max_decisions):
            self._check_cancelled()
            self.runtime.validate_turn(turn_id)
            inputs = packet.to_model_inputs()
            bounded_history = self._bounded(history, required_index=observation_index)
            messages = [
                {"role": "system", "content": self.config.system_prompt},
                *bounded_history,
            ]
            image_message_index = (
                next(
                    index
                    for index, item in enumerate(messages)
                    if item is history[observation_index]
                )
                if inputs.get("image") is not None
                else None
            )
            self.events.publish("phase", "THINK")
            self._log(
                "model_input",
                turn_id,
                {
                    "messages": messages,
                    "max_new_tokens": self.config.max_new_tokens,
                    "image_message_index": image_message_index,
                    "image_size": list(inputs["image"].size)
                    if inputs.get("image") is not None
                    else None,
                    "extra_image_sizes": {
                        name: list(image.size)
                        for name, image in inputs.get("extra_images", {}).items()
                    },
                },
            )
            answer = self._backend.generate(
                messages=messages,
                image=inputs.get("image"),
                image_message_index=image_message_index,
                max_new_tokens=self.config.max_new_tokens,
                cancel_event=self._cancel,
                **(
                    {"extra_images": inputs["extra_images"]}
                    if "extra_images" in inputs
                    else {}
                ),
            )
            raw = answer["text"]
            self._log(
                "model_output",
                turn_id,
                {
                    "raw": raw,
                    "cancel_requested": self._cancel.is_set(),
                    "metadata": {
                        key: answer[key]
                        for key in (
                            "model",
                            "finish_reason",
                            "usage",
                            "seconds",
                            "input_tokens",
                            "generated_tokens",
                            "tokens_per_second",
                            "reasoning_content",
                        )
                        if key in answer
                    },
                },
            )
            self._check_cancelled()
            self.runtime.validate_turn(turn_id)
            self.events.publish("model_output", raw)
            history.append({"role": "assistant", "content": raw})
            try:
                decision = parse_decision(raw)
            except (ValueError, TypeError) as exc:
                self._log("parse_error", turn_id, {"error": str(exc)})
                parse_failures += 1
                if parse_failures > self.config.max_parse_retries:
                    raise ValueError("模型多次未返回合法 JSON 指令：" + str(exc)) from exc
                self._chat("模型指令格式有误，正在请求修正。")
                history.append(
                    {
                        "role": "user",
                        "content": "JSON 校验错误（"
                        + type(exc).__name__
                        + "）："
                        + str(exc)
                        + "\n请修正上一条回复的 JSON 格式，保持原任务和判断不变。"
                        + '\n只输出一个符合工具协议的 JSON 对象；若使用 final，其值必须是字符串，例如 {"final":"任务已完成。"}。',
                    }
                )
                continue
            parse_failures = 0
            self._log("decision", turn_id, decision)
            if "final" in decision:
                self.runtime.finish_turn(turn_id)
                self._conversation = self._bounded(history)
                self._chat(decision["final"], "assistant")
                return
            self._chat("调用工具：" + json.dumps(decision, ensure_ascii=False), "tool")
            self._log("tool_call", turn_id, decision)
            result = self.runtime.execute_tool(decision, turn_id).result()
            self._log("tool_result", turn_id, result)
            self._check_cancelled()
            self._chat("工具结果：" + json.dumps(result, ensure_ascii=False), "tool")
            packet = self.runtime.collect_observation(turn_id)
            self._append_observation(history, packet, task=text, tool_result=result)
            observation_index = len(history) - 1
            self._log("observation", turn_id, json.loads(history[-1]["content"]))
        raise RuntimeError("已达到本轮最大决策次数，任务停止。")

    def _unload(self):
        backend, self._backend = self._backend, None
        try:
            unload = getattr(backend, "unload", None)
            if unload is not None:
                unload()
        finally:
            self._model_state = "unloaded"
            self._conversation.clear()

    def _work(self):
        try:
            while True:
                kind, value = self._jobs.get()
                if kind == "shutdown":
                    break
                try:
                    if self._closing:
                        continue
                    if kind == "load":
                        self._model_state = "loading"
                        remote = isinstance(value, DeepSeekConfig)
                        self._chat(
                            "正在配置 DeepSeek API……"
                            if remote
                            else "正在加载本地模型，首次请求可能需要编译推理算子……"
                        )
                        self._backend = self._factory(value)
                        self._model_state = "ready"
                        self._chat(
                            "DeepSeek API 已配置；首次授权任务时验证网络和密钥。"
                            if remote
                            else "模型已加载，可以发送任务。"
                        )
                    elif kind == "unload":
                        self._model_state = "unloading"
                        self._unload()
                        self._chat("模型已卸载。")
                    elif kind == "reload":
                        self.runtime.reload_scene(value)
                        self._conversation.clear()
                        self._log("environment_reload", None, {"scene_id": value})
                        self._chat(
                            "环境已重载：" + value
                            + "。无人机与仿真时间已重置，模型对话历史已清空。"
                        )
                    elif kind == "turn":
                        self._run_turn(value)
                except GenerationCancelled:
                    self._log("cancelled", None, {"job": kind})
                    self.runtime.cancel_turn("本轮已取消")
                except Exception as exc:
                    self._log(
                        "error", None, {"job": kind, "error": describe_error(exc)}
                    )
                    if kind in {"load", "unload"}:
                        self._backend = None
                        self._model_state = "error"
                    if kind == "turn":
                        self.runtime.cancel_turn("运行错误")
                    self._chat(describe_error(exc), "error")
                finally:
                    with self._lock:
                        self._busy = self._running_turn = False
                    self.events.publish("runner_idle", None)
        finally:
            try:
                self._unload()
            except Exception as exc:
                self._chat("模型释放失败：" + describe_error(exc), "error")

    def close(self, *, wait=True):
        should_cancel = False
        with self._lock:
            if not self._closing:
                self._closing = True
                should_cancel = True
                self._jobs.put(("shutdown", None))
        if should_cancel:
            self.cancel("应用正在关闭")
        if wait:
            self._thread.join(timeout=30)
            if self._thread.is_alive():
                raise TimeoutError("模型计算尚未退出，请等待当前 GPU 运算完成")
