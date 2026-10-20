# DeepSeek API 后端

保持原有 THINK → EXECUTE → OBSERVATION → THINK / END 回路，API 仅负责生成 JSON 决策。
`agent/backends.py` 根据配置延迟导入本地或 HTTP 后端；选择 DeepSeek 不加载 Transformers 或 CUDA 模型。
`agent/deepseek.py` 通过 HTTP Chat Completions API 请求生成，实现与本地后端相同的
`generate(messages, image, image_message_index, max_new_tokens, cancel_event)` / `unload()` 接口。

## 启动

```bash
conda activate mm_planner
python app.py --backend deepseek
```

在界面输入 API Key，或启动前通过环境变量 `DEEPSEEK_API_KEY` 配置。
密钥输入框隐藏内容，不保存到文件，不作为命令行参数，不进入日志。
HTTP 依赖为 `requirements-deepseek.txt` 中的 httpx；当前 mm_planner 已安装 0.28.1。
其他环境可运行 `python -m pip install -r requirements-deepseek.txt`，另需原有仿真及 PySide6 依赖。

界面可选本地 Ministral 或 DeepSeek，模型就绪后先卸载再更改后端设置。
“配置 API”仅验证本地配置和密钥是否存在，不发送请求，不验证余额和服务端权限；
首次发送获授权的任务时才访问 API。“卸载模型”关闭此客户端，不操作云端模型。

默认设置：

| 参数 | 默认值 | 配置方式 |
| --- | --- | --- |
| 模型 | deepseek-flash | 顶部模型名或 `--model-path` |
| API base URL | https://api.deepseek.com | 界面、`--api-base-url` 或 DEEPSEEK_BASE_URL |
| 思考模式 | 开启 | 界面或 `--no-thinking` / `--thinking` |
| 思考强度 | high | 界面或 `--reasoning-effort low/high/max` |
| 请求总超时 | 180 秒 | 界面或 `--api-timeout` |
| 单次生成 token | 4096 | Agent 设置或 `--max-new-tokens` |

在已有窗口切换到 DeepSeek 时，本地默认 384 token 会自动提高至 4096；其他自定义值保持。
思考强度仅在思考开启时发送，不发送 temperature/top_p 等采样参数。
base URL 可以是服务根地址或 `/v1` 地址，后端追加 `/chat/completions`。
自定义模型必须实际支持图像输入，不会在服务拒绝图像时静默降级成纯文本。
配置了 API 后，授权任务会发送 system prompt、裁剪后的历史、任务、状态及当前第一视角 PNG。

示例：

```bash
python app.py --backend deepseek --max-new-tokens 8192 --reasoning-effort high
python app.py --backend deepseek --no-thinking --max-new-tokens 1024
```

## 协议、日志和取消

- 图片按 `image_message_index` 附在对应 user 观测消息的 `content` 数组中，使用 PNG/base64 data URL。
  JSON 修正消息继续留在历史末尾，图片不会移到修正提示上。
  开启深度后增加第二张 PNG 和 depth 名称说明，默认 56×56；米制距离编码和 3×3 区域最小值
  放在观测 JSON 中。设置及解释见 [深度观测](../observations/README.md#深度观测)。
- 请求固定 `response_format={"type":"json_object"}`。自定义提示词不含 JSON 字样时追加格式要求。
  服务端 JSON Output 不代替本地字段、状态、有限数值和航点时间校验。
- 沿用本仓库的 JSON 工具协议，不向 API 提交原生 `tools`。只将最终 `content` 交给解析器，
  不把 `reasoning_content` 误当工具指令，也不回传到后续历史。
  按官方思考文档，不携带 `tools` 时无须回传思考内容。
- 模型输出日志的 metadata 包含模型、usage、finish_reason、耗时和返回的 reasoning_content；
  不包含 Authorization 或图片 base64。导出的日志会包含对话与模型思考文本。
- 空 content、生成长度截断、HTTP 错误和网络超时会在对话框报错并结束本轮；截断输出不执行。
  不自动重试付费请求，避免不明确的重复费用。
- 有手柄时 Axis4 未授权不发任务；桌面启动未检测到手柄时自动授权 Agent。
  撤销授权或点击终止后，异步请求约每 50 ms 检查取消，
  取消正在等待的 HTTP 操作并丢弃输出；仿真线程持续运行。
  客户端取消不能保证供应商立刻终止计算或免除已产生的费用。
- 请求过程中仿真继续运行，模型回复时使用的观测可能已有延迟；不据此保证避障或控制精度。

## 测试

```bash
MUJOCO_GL=egl QT_QPA_PLATFORM=offscreen OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m unittest tests.test_deepseek tests.test_runner tests.test_desktop_ui -v
```

测试使用 httpx.MockTransport 模拟 HTTP 与真实 MuJoCo runtime，覆盖图像绑定、JSON 参数、
思考开关、错误/空输出/截断、取消、授权、工具反馈及日志。不使用真实密钥或付费调用。

## 文档依据

- [图像理解](https://api-docs.deepseek.com/zh-cn/guides/vision/)
- [思考模式](https://api-docs.deepseek.com/zh-cn/guides/thinking_mode)
- [JSON Output](https://api-docs.deepseek.com/zh-cn/guides/json_mode)

用户提供的“图像理解”HTML 的 canonical URL 和正文实际指向思考模式，
图像协议已补充核对上述官方图像理解页面。模型名和服务端能力以实际部署为准。
