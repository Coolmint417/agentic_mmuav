# Qwen（通义千问）API 后端

保持原有 THINK → EXECUTE → OBSERVATION → THINK / END 回路，API 仅负责生成 JSON 决策。
`agent/backends.py` 根据配置延迟导入本地或 HTTP 后端；选择 Qwen 不加载 Transformers 或 CUDA 模型。
`agent/qwen.py` 通过 DashScope 的 OpenAI 兼容 Chat Completions 接口请求生成，实现与本地 /
DeepSeek / MiMo 后端相同的 `generate(messages, image, image_message_index, max_new_tokens,
cancel_event)` / `unload()` 接口。公共 HTTP、取消、解析与错误映射在 `agent/openai_compat.py`，
Qwen 只覆盖请求体。

## 启动

```bash
conda activate mm_planner
python app.py --backend qwen
```

在界面输入 API Key，或启动前通过环境变量 `DASHSCOPE_API_KEY` 配置。
密钥输入框隐藏内容，不保存到文件，不作为命令行参数，不进入日志。
HTTP 依赖与 DeepSeek 相同（`requirements-deepseek.txt` 中的 httpx），另需原有仿真及 PySide6 依赖。

界面可选本地 Ministral、DeepSeek、MiMo 或 Qwen；模型就绪后先卸载再更改后端设置。
“配置 API”仅验证本地配置和密钥是否存在，不发送请求，不验证余额和服务端权限；
首次发送获授权的任务时才访问 API。“卸载模型”关闭此客户端，不操作云端模型。

默认设置：

| 参数 | 默认值 | 配置方式 |
| --- | --- | --- |
| 模型 | qwen3-vl-plus | 顶部模型名或 `--model-path` |
| API base URL | https://dashscope.aliyuncs.com/compatible-mode/v1 | 界面、`--api-base-url` 或 QWEN_BASE_URL |
| 思考模式 | 关闭 | 界面或 `--thinking` / `--no-thinking` |
| 请求总超时 | 180 秒 | 界面或 `--api-timeout` |
| 单次生成 token | 4096 | Agent 设置或 `--max-new-tokens` |

在已有窗口切换到 Qwen 时，本地默认 384 token 会自动提高至 4096；其他自定义值保持。
base URL 可以是服务根地址或 `/v1` 地址，后端追加 `/chat/completions`。
自定义模型必须实际支持图像输入，不会在服务拒绝图像时静默降级成纯文本。

## JSON Output 与思考模式

DashScope 对 `enable_thinking=true` 的请求返回 400：JSON mode 不能与思考模式同时启用。
因此：

- 默认 `thinking=false`：请求体含 `response_format={"type":"json_object"}` 与
  `enable_thinking=false`，输出更稳定、更快。
- 界面勾选“启用思考模式”后：请求体只发送 `enable_thinking=true`，不发送 `response_format`；
  模型依靠 system prompt 的 JSON 协议与解析重试输出 JSON，延迟更高、偶发重试。

思考强度选项对 Qwen 不生效：请求体不发送 `reasoning_effort`。

示例：

```bash
python app.py --backend qwen
python app.py --backend qwen --model-path qwen3-vl-flash --max-new-tokens 8192
python app.py --backend qwen --thinking --max-new-tokens 8192
```

## 与 DeepSeek / MiMo 的协议差异

- 生成上限字段为 `max_tokens`（DashScope OpenAI 兼容接口），不发送 `max_completion_tokens`。
- 思考开关使用布尔字段 `enable_thinking`，不使用 `thinking` 对象，也不发送 `reasoning_effort`。
- 仅在非思考模式发送 `response_format={"type":"json_object"}`；思考模式省略该字段。
- 图片按 `image_message_index` 附在对应 user 观测消息的 `content` 数组中，使用 PNG/base64 data URL；
  base64 单张不超过 50 MiB。
- 沿用本仓库的 JSON 工具协议，不向 API 提交原生 `tools`；只把最终 `content` 交给解析器，
  不把 `reasoning_content` 误当工具指令，也不回传到后续历史。
- 模型输出日志的 metadata 包含模型、usage、finish_reason、耗时和返回的 reasoning_content；
  不包含 Authorization 或图片 base64。
- 空 content、生成长度截断、HTTP 错误和网络超时会在对话框报错并结束本轮；截断输出不执行。
  不自动重试付费请求，避免不明确的重复费用。
- 有手柄时 Axis4 未授权不发任务；桌面启动未检测到手柄时自动授权 Agent。
  撤销授权或点击终止后，异步请求约每 50 ms 检查取消，取消正在等待的 HTTP 操作并丢弃输出；
  客户端取消不能保证供应商立刻终止计算或免除已产生的费用。

## 测试

```bash
MUJOCO_GL=egl QT_QPA_PLATFORM=offscreen OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m unittest tests.test_qwen tests.test_mimo tests.test_deepseek \
  tests.test_runner tests.test_desktop_ui -v
```

测试使用 httpx.MockTransport 模拟 HTTP 与真实 MuJoCo runtime，覆盖请求体字段（max_tokens、
enable_thinking、response_format 条件）、图像绑定、错误/空输出/截断、取消、授权与 UI 后端切换；
不使用真实密钥或付费调用。

## 备注

DashScope 北京地域 OpenAI 兼容 base URL 为 `https://dashscope.aliyuncs.com/compatible-mode/v1`，
新加坡地域为 `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`；如使用新版工作空间域名，
可用 `QWEN_BASE_URL` 覆盖。模型名与服务端能力以阿里云百炼文档为准。
本机 WSL 可直连 `dashscope.aliyuncs.com`（未带密钥时返回 401）。
