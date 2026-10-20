# Environment Runtime / Agent Runner / PySide6

从工程根目录运行：

```bash
conda activate mm_planner
python app.py
```

桌面依赖见 `requirements-desktop.txt`，当前 mm_planner 已安装 PySide6 6.11。
模型与 FP8 算子沿用本地缓存；首次部署其他环境时参照根 README 安装并缓存算子。
也可选择 DeepSeek API 后端：`python app.py --backend deepseek`，不加载本地模型。
API Key、思考模式、JSON Output、图像输入和请求取消说明见 [DeepSeek 后端](../agent/DEEPSEEK.md)。

## 操作

1. 启动后仿真独立运行，默认约 1 倍实时速度，遥控器可在手动模式控制无人机。
2. 选择本地模型目录或保留默认缓存 ID，点击“加载模型”。加载过程在工作线程执行。
3. 有手柄时将 Axis4 切为1，在右侧输入任务。启动时未检测到手柄则自动授权 Agent，
   左侧显示“无手柄自动授权”，加载模型或配置 API 后可直接输入任务。
   未授权时对话框提示，模型不会收到 prompt。
4. 左侧只显示一个相机画面，通过标签页或 ◀ / ▶ 在第一视角 RGB、世界视角、第一视角深度间切换。
   相机切换只影响显示，模型始终接收第一视角观测。
   世界视角支持鼠标交互，见下表；下方显示相机位置、azimuth、elevation 和距离。
5. “终止本轮任务”取消推理和工具；Axis4 撤销还会立即恢复手动控制。
6. 一轮 FINAL / 终止 / 错误处理结束后才能输入下一轮。模型加载、卸载期间也不启动任务。

| 世界视角操作 | 效果 |
| --- | --- |
| 左键拖动 | 绕观察中心旋转，改变 azimuth / elevation 和相机位置 |
| 右键拖动 | 平移观察中心和相机 |
| Shift + 左键 / 右键拖动 | MuJoCo 的水平旋转 / 水平平移操作 |
| 滚轮 / 中键拖动 | 改变相机到观察中心的距离 |
| Ctrl + 滚轮 / FOV 数值框 | 调整垂直视场角，范围 5°～150° |
| 左键双击 / “重置视角” | 恢复 scene.xml 中 world_camera 的初始位置、方向和 FOV |

鼠标从实际图像区域开始操作；图像外的黑边不接收相机操作。
切换到第一视角再切回时保留调整结果；重启应用后恢复 XML 初始配置，不写回 XML。
预览和 UI 刷新默认约 30 FPS，实际速度取决于渲染性能。
界面预览默认 **1280×960**，模型观测默认 **224×224**，由独立的 renderer 渲染。
第一视角和世界视角的界面预览都使用高分辨率；模型只接收低分辨率第一视角。
窗口放大或使用高 DPI 显示器时，显示按设备像素比缩放，保持画面比例。
深度预览同样使用界面预览分辨率，白近黑远、固定 0～10 m 显示范围；
与发送给模型的 56×56 深度图分辨率独立，详情见 [深度观测](../observations/README.md#深度观测)。
`runtime/preview.py` 使用私有 `MjvCamera` 和预览渲染场景，通过 `mjv_moveCamera` 实现原生交互；
相机操作只在环境 owner 线程执行。FOV 只改预览场景投影，不改变模型中的固定相机，
因此不会改变无人机姿态、控制目标或 Agent 的第一视角观测。

界面显示模型状态、控制权、两套状态机、位置、yaw、仿真时间，右侧分为三个标签页：

- **对话**：输入任务、查看回复和工具结果、终止任务。
- **JSON 日志**：按时间累计每次模型输入、原始输出（包括非法 JSON）、解析结果/错误、
  工具调用/结果、观测、取消和运行错误。每条记录含 UTC 时间、类型和轮次标识；
  不属于具体轮次的运行错误/取消记录轮次为 null。
  输入记录包含完整 system prompt 和本次历史文本；RGB 记录元数据和尺寸，不复制图像二进制。
  支持导出 JSONL、清空；本次应用进程中不自动截断日志，关闭前请导出需要保留的记录。
- **Agent 设置**：完整 system prompt 编辑及下列参数。点击“应用设置”后生效，
  工作线程忙碌时禁止应用，避免一轮内的协议和限制变化。无需重新加载模型。
  “恢复默认值”只填入默认值，仍需点击应用。设置仅保存在本次运行中。

| 参数 | 默认值 | 含义 |
| --- | --- | --- |
| 最大执行轮数 | 16 | 单个用户任务的模型调用总次数，包含 JSON 修正及 FINAL |
| 单次生成 token 上限 | 384 | 限制每次输出长度，过小可能截断航点 JSON |
| JSON 修正重试次数 | 2 | 连续非法 JSON 允许请求修正的次数 |
| 历史消息数上限 | 12 | 历史 user/assistant 消息数目标上限，不含 system；JSON 修正期间保留当前观测及其后的消息 |
| 上下文字符上限 | 24000 | 历史文本字符上限，不含 system 和图像；不是 token 限制 |
| 添加深度信息 | 关闭 | 启用后把第一视角深度作为第二张图片与 RGB 一起发送 |
| 模型深度分辨率 | 56 | 正方形边长，范围 8～1024，单位像素 |

默认 system prompt 位于 `agent/protocol.py` 的 `SYSTEM_PROMPT`，设置页显示当前有效全文。
它规定单 JSON 输出、三个飞行工具及 FINAL、工具状态限制、NWU 绝对坐标、
相对工具启动的航点时间、弧度 yaw，以及必须依据工具反馈判断成功。
自定义 prompt 不改变 Python 侧 JSON 解析器和工具校验协议。

方向指令约定：北/南/西/东对应世界坐标的 +X/-X/+Y/-Y；前/后/左/右按任务开始时的 yaw
换算为世界坐标，默认保持高度和朝向。例如 yaw=pi/2 时向前是 +Y，向北仍是 +X。
模型需在首次规划时确定绝对目标，绕行或重试时不重复累加位移。
默认提示词要求遇到障碍先判断路径是否受阻，并考虑分段绕行、观测和重规划。
这属于模型规划策略：当前轨迹优化器没有障碍约束、碰撞检测或可靠的单目深度估计，
不能因此视为已实现可靠自主避障；未知空间也不能仅靠提示词判定可通行。
正在运行的窗口不会自动更新 prompt，重启应用或在设置页点击“恢复默认值”后“应用设置”。

OOM、模型目录错误、JSON 重试失败、工具及渲染异常都会回显到对话框。
终止是协作式取消：无法强杀已提交的 GPU 运算或 IPOPT 调用，但仿真和手动控制独立运行，
迟到的模型/规划结果不能启动工具或重新接管。UI 在计算退出前不开始新一轮任务。

可以减小推理负载：

```bash
python app.py --max-new-tokens 128 --observation-size 168
python app.py --model-path /path/to/local/model
python app.py --no-preview
python app.py --preview-width 1920 --preview-height 1440 --observation-size 224
python app.py --preview-width 960 --preview-height 720 --preview-fps 20
python app.py --backend deepseek --depth --depth-size 56
```

`--no-preview` 只关闭界面预览，模型的 RGB 观测仍需要可用的渲染后端。
入口默认在导入 MuJoCo 前设置 `MUJOCO_GL=egl`，可通过外部环境变量覆盖。

## 线程与职责

| 部件 | 职责 |
| --- | --- |
| `runtime/environment.py` | 唯一的仿真 owner 线程：step、Axis4、状态机、工具任务、观测、GL renderer |
| `agent/runner.py` | 模型工作线程：加载/卸载、多轮消息、模型通信、JSON parse、工具和观测调度 |
| 轨迹规划 executor | 执行 IPOPT 优化，等待期间环境仍保持位置并处理遥控器 |
| `ui/main_window.py` | Qt 主线程：显示缓存快照/图像，发送任务和停止请求，不访问 MjData |

Runner 不调用 `env.step()`，只调用 runtime 的任务/观测 API。
运行时通过命令队列/Future 提交任务与返回结果；runner 等待 Future 不阻塞环境。
工具的 `iter_execute()` 每次让出控制，runtime 推进仿真后恢复任务。
原同步 `execute()` 保留给无 runtime 的测试/脚本，不能与运行服务同时用于同一个环境。

所有 state machine 修改和 renderer 生命周期都归环境线程。
模型线程通过 `threading.Event` 接收取消，Transformers `StoppingCriteria` 在生成步之间检查它。
Axis4 由环境持续轮询；runner 接收任务时检查快照，在开始每次推理前再次请求环境读取实际开关。
用 `turn_id` 检查模型决策/工具结果所属轮次，防止迟到结果覆盖当前控制。
预览只缓存最新帧，不在 Qt 消息队列中积压 RGB 图像。

## 模型协议与循环

模型每次返回一个完整 JSON 对象：

```json
{"tool":"takeoff"}
```

```json
{"tool":"land"}
```

```json
{"tool":"fly_waypoint","waypoint":[{"time":2.0,"position":[-1.8,2.0,1.5],"yaw":0.0}]}
```

```json
{"final":"任务已完成。"}
```

Runner 保存有限长度的多轮历史，并在每次工具结果后加入最新状态和 RGB。
只把最新一组 RGB/可选深度图交给视觉处理器，旧图像不保留；历史中的观测保留文本状态。
非法 JSON 默认最多请求修正2次；一轮最多16次决策，避免无限调用。
修正提示包含具体的 JSON 语法/协议校验错误，以及“请修正上一条回复的 JSON 格式，保持原任务和判断不变”。
图片通过显式消息索引绑定到最新观测，不会附到后面的格式修正提示。
裁剪历史时始终保留当前观测（含任务）及后续修正消息，即使暂时超过历史消息数目标上限；
若这些必要消息仍超过上下文字符上限，则报错结束，避免静默丢失任务或错配图片。
日志 `model_input.image_message_index` 为图片所属消息在 `messages` 中的零起始索引。
工具拒绝或执行失败会作为反馈给模型，让模型调整计划或给出解释。

## 验证

```bash
MUJOCO_GL=egl QT_QPA_PLATFORM=offscreen OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 \
  python -m unittest discover -s tests -v
python scripts/smoke_desktop_ui.py
python scripts/smoke_desktop_model.py
```

单元/集成测试使用脚本化假模型验证真实 MuJoCo 工具闭环、Axis4、OOM、模型生命周期、JSON重试、
对话历史、后台规划取消、相机切换和 Qt 关闭。
`smoke_desktop_ui.py` 不加载模型、不执行飞行工具，输出 `artifacts/desktop_ui_smoke.png`。
`smoke_desktop_model.py` 使用真实缓存 Ministral 模型和明确模拟的 Axis4=1，只请求 FINAL，
输出 `artifacts/desktop_model_smoke.json`；它不验证模型自主飞行能力。
生产入口 `app.py` 启动时检测手柄：设备可用时仍由真实 Axis4 授权；不可用时使用无手柄 Agent 输入。
无手柄模式不自动起飞或发送 prompt，界面终止按钮仍可结束本轮。接回手柄后重启应用即可恢复 Axis4 控制。
该选项仅由桌面入口显式开启，独立 `MMUAVEnv()` 保持原默认行为；
需要复用时可传入 `auto_agent_without_joystick=True`。
