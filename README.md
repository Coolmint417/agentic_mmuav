# Agentic MMUAV

起降工具已实现：模型可调用 `{"tool": "takeoff"}` 或 `{"tool": "land"}`，无需参数。
工具复用环境自动起降逻辑，并在完成后同步 agent 与无人机状态机。
调用方法及完成判据见 [tools/README.md](tools/README.md)。

基于现有 MuJoCo 无人机环境，逐步搭建本地 VLM 驱动的具身 agent。
当前提供本地 Ministral 3B 和 DeepSeek API 推理、起降与航点工具、agent 与无人机状态机。
已实现独立 Environment Runtime、模型 Agent Runner 和 PySide6 桌面界面。

中文总体架构、两个状态机及模型通信格式见 [工程架构图](docs/architecture.md)。

现有 **5 张地图**，可在前端顶部的“仿真地图”下拉框选择并点击“重载环境”，
或使用 `python app.py --map map3` 指定初始地图。重载保留模型，重置无人机与仿真时间并清空模型对话历史；
任务运行中需先终止并等待退出。布局、材质及扩展方法见 [地图说明](docs/maps.md)。

## 启动桌面界面

首次安装完整环境（Linux，包含 CUDA 12.8 的本地模型后端）：

```bash
conda env create -f requirement.yaml
```

该文件参照当前 `mm_planner` 的核心依赖版本编写，不包含本机环境绝对路径。
若已有同名环境，可用 `conda env create -n mm_planner_new -f requirement.yaml` 创建新环境，
或用 `conda env update -n mm_planner -f requirement.yaml` 更新现有环境。
环境安装不包含模型权重、API Key 或 NVIDIA 驱动；首次使用本地模型还需准备 Hugging Face
模型缓存并运行下文的 `scripts/cache_fp8_kernel.py`。仅使用 DeepSeek 不需要本地模型权重。
桌面显示和 EGL 渲染依赖主机可用的图形驱动；此配置针对当前 Linux 工程。

```bash
conda activate mm_planner
python app.py
```

点击“加载模型”，有手柄时将 Axis4 切为1后发送任务；启动时未检测到手柄则自动授权 Agent。
自动授权只开放任务输入，不自动推理或起飞；可用界面的“终止本轮任务”停止任务。
未授权的消息不会发送给模型；
OOM 等运行错误会显示在对话框。左侧用 ◀ / ▶ 切换第一视角和世界视角，模型始终使用第一视角。
世界视角下可左键旋转、右键平移、滚轮缩放、Ctrl+滚轮调整 FOV，双击或点击按钮重置视角。
界面预览默认 1280×960，发送给模型的第一视角默认保持 224×224；两者可独立配置。
预览标签还可查看第一视角深度；Agent 设置可开启发送深度图并选择分辨率，默认 56×56。
也可用 `python app.py --backend deepseek --depth --depth-size 56` 启动。
详细架构、参数与验证方法见 [runtime/README.md](runtime/README.md)。

DeepSeek API 可通过界面顶部后端选择启用，也可直接启动：

```bash
python app.py --backend deepseek
```

在界面填写 API Key 或预先设置 `DEEPSEEK_API_KEY` 环境变量，点击“配置 API”。
默认模型 `deepseek-flash`，支持第一视角图片、思考模式和 JSON Output；默认生成上限 4096 token。
该后端会将任务、对话历史、状态和第一视角图片发送至所配置的 API。
详细配置、取消行为及测试范围见 [agent/DEEPSEEK.md](agent/DEEPSEEK.md)。

## 手动与 agent 控制

无人机始终使用位置控制。Axis4=1 授权 agent，屏蔽其他遥控器输入；
Axis4 从1切到0时终止 agent（END），取消轨迹/自动起降并进入 HOVERING，恢复手动控制。
手动模式仅有 ON_GROUND / HOVERING，HOVERING 允许摇杆控制运动；Axis7 负责手动起降并同步飞行状态。
原 FLYING 状态已改名为 FLY_AGENT，专用于授权后的 agent 轨迹执行；Axis5 不再使用。
`MMUAVEnv.set_model_stop_callback()` 已接入 runner 的生成取消事件。
完整状态规则和入口见 [env/README.md](env/README.md)。

## 在 mm_planner 中运行

```bash
conda activate mm_planner
python -m pip install -r requirements-ministral.txt
# 首次部署需联网缓存 FP8 CUDA 算子；之后测试完全离线。
python scripts/cache_fp8_kernel.py
python scripts/test_ministral.py
```

默认复用 Hugging Face 本地缓存中的
`mistralai/Ministral-3-3B-Instruct-2512`，不下载模型权重，也不依赖 foundation 环境。
模型加载使用 `local_files_only=True`、`trust_remote_code=False`、
原生 FP8 权重、BF16 非量化模块以及 SDPA attention；需要可用的 CUDA GPU，
计算能力至少 8.9。本机已有 PyTorch `2.10.0+cu128`，依赖文件不重新安装 PyTorch。
FP8 算子固定到 v4 的 commit `8c178950e8710e26a2210e4e909cd02dc16c8715`；
缓存脚本下载整个小型源码快照，以满足 huggingface-hub 的离线完整性检查。
后端为固定的 Transformers 5.18 设置对应算子映射；升级依赖时需重新验证此兼容设置。

可以显式指定已有模型目录：

```bash
python scripts/test_ministral.py --model-path /path/to/local/snapshot
```

测试结果写入 `artifacts/ministral_smoke.json`，相机图像写入
`artifacts/mujoco_fpv.png`。测试失败返回非零退出码。四项测试分别检查：

- 文本算术生成是否得到 `5`。
- NWU 坐标下向北移动 0.5 米，经过单层 Markdown 包装清理和 JSON 校验后，
  是否得到目标 `[0.5, 0, 1]`。
- 红色图像输入是否识别为 `red`。
- 从现有场景渲染 `fpv_camera` 图像并完成 VLM 描述。

最后一项只检查渲染与视觉推理链路，不衡量描述准确率、避障或飞行成功率。
报告中的生成耗时包括 prefill 和 decode；首个请求可能包含 Triton 编译开销。
本机测试中，首个请求的算子编译/调优约需 50 秒，后续相机描述请求约 2.5 秒。
仿真步长为 0.002 秒（500 Hz），模型适合低频规划层。

最初仅提示 `NWU means x=north, y=west, z=up` 时，模型输出了 Markdown 代码块，
并将北向位移误写为负 Y。原始失败结果保留在 `artifacts/ministral_baseline.json`。
当前测试明确约束北向增加 X、西向增加 Y，以及纯 JSON 输出；
即使单次样例通过，也不能据此推断模型具备可靠的导航或避障能力。
实际模型仍可能添加代码块。`agent/waypoint.py` 只允许清理一个完整的 JSON 代码块，
然后校验字段、坐标数量和有限数值；不从混杂文本中任意提取对象。
该校验未包含场景边界、碰撞、速度和轨迹可行性，尚不能直接执行目标。
最终 `ministral_smoke.json` 的航点项是在实际 GPU 输出上重新执行解析器校验；
只修改解析器后没有重复模型推理，报告的 `validation` 字段记录了这一点。
未清理包装时的严格 JSON 检查结果保留在 `artifacts/ministral_strict_json.json`。

## Observation 模块

`observations/` 提供可替换的观测模块：当前位置、姿态（wxyz 四元数和弧度制 RPY）、
飞行状态、从启动起的仿真时间，以及默认 224×224 的第一视角 RGB。
可配置分辨率、相机，并追加传感器或替换整套 provider。

`packet.to_json()` 输出带 PNG/base64 的完整 JSON；
`backend.generate(**packet.to_model_inputs())` 将状态 JSON 与 PIL 图像传给当前本地模型。
`collect_for_agent(provider, environment)` 负责采集及 OBSERVATION → THINK 的状态推进，
采集时失去 Axis4 授权或发生轮次切换会丢弃观测。
完整字段、调用及扩展方式见 [observations/README.md](observations/README.md)。

## 推理后端

```python
from PIL import Image
from agent.ministral import MinistralBackend

backend = MinistralBackend()  # 只加载一次
result = backend.generate("当前画面有哪些障碍物？", image=Image.open("artifacts/mujoco_fpv.png"))
print(result["text"])
```

`agent/ministral.py` 同时支持纯文本和单图输入。默认图像最长边 448 像素，
可以通过 `MinistralBackend(image_size=...)` 调整。缓存的模型快照缺少
`preprocessor_config.json`，因此从模型配置读取 14 像素 patch 和 2 倍 spatial merge，
显式构造 PixtralProcessor，避免使用不匹配的默认 patch 设置。

模型资料：[官方模型卡](https://huggingface.co/mistralai/Ministral-3-3B-Instruct-2512)。

## 仓库中的控制链路

- `env/scene.xml`：场景，包含 `mmuav.xml` 和 `assets/map1.xml`。
- `env/mmuav_env.py`：手柄输入、位置控制回调、控制权切换与起降/轨迹入口。
- `env/controller/`：SE3 位置控制、姿态控制、角速度 PID、滤波和电机混控。
- `env/trajectory_optimizer/`：将位置/yaw 航点拟合为平滑分段多项式。

世界坐标使用 **NWU**：x 北、y 西、z 上；yaw 使用弧度。
仿真状态可从 `d.qpos[:3]` 和 `d.qvel[:3]` 读取，相机包括
`fpv_camera`、`fpv_camera_far`、`world_camera`。

未来闭环可以使用以下路径：

```text
任务 + 状态 + 相机图像
  → Ministral 低频规划
  → JSON 校验 / 航点约束
  → PolynomialTrajectoryOptimizer
  → set_planner_target_provider
  → SE3 / Rate PID / Mixer
  → MuJoCo stepping
  → 新观测与任务反馈
```

控制接口说明：

1. `set_planner_target_provider(provider)` 仅在位置模式、且自动起降没有接管时生效。
   环境固定使用位置控制，provider 只在 Axis4 授权 agent 后生效。
2. provider 接收 `m, d, current_rotation, omega, dt`，返回对象需包含
   `position, velocity, acceleration, yaw`；可选 `yaw_rate`，但当前代码会直接读取 `omega`。
   `MMUAVEnv.follow_trajectory()` 已适配轨迹状态及 body-frame yaw 前馈。
3. 模型推理应在规划线程/进程执行；MuJoCo 控制回调只读取已生成的轨迹，
   Environment Runtime 持续推进仿真和处理 Axis4，模型 runner 独立运行，不直接调用 env.step。

另：轨迹 README 中 yaw 的 C4/四阶目标描述与当前配置不一致；
实际 `config_trajectory_optimizer.py` 使用 yaw 连续阶数 2、目标导数阶数 2。

