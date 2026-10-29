# 无人机 Agent 工程架构

本文对应当前代码。图中实线表示已实现的调用或数据流，虚线表示待接入模块或注释关联。Mermaid 图可在 GitHub 或支持 Mermaid 的 Markdown 预览中查看。

## 1. 总体架构与线程分工

```mermaid
flowchart TB
    USER["用户：自然语言任务 / 终止任务 / 参数设置"]
    UI["PySide6 界面 · Qt 主线程<br/>对话 / JSON 日志 / Agent 设置<br/>第一视角 RGB / 深度 / 可交互世界视角"]
    USER --> UI

    subgraph MODEL["模型后端 · 统一生成接口"]
        HF["已实现：Hugging Face Transformers<br/>本地 Ministral 3B · GPU 推理<br/>视觉处理器接收 PIL 图像"]
        DS["已实现：DeepSeek API<br/>HTTP Chat Completions<br/>文本 + image_url / JSON Output"]
        LLAMA["待接入：本地 llama.cpp HTTP<br/>GGUF 模型由服务端加载<br/>量化计算由服务端完成<br/>图像支持取决于模型及视觉适配"]
        IF["统一接口：generate(...)<br/>输入：消息、图像、生成上限、取消事件<br/>输出：text 及可选推理统计"]
        IF <--> HF
        IF <--> DS
        IF -.-> LLAMA
    end

    RUN["Agent Runner · 模型工作线程<br/>加载 / 卸载后端 · 历史消息管理<br/>JSON 解析与修正 · 工具 / 观测调度<br/>最大决策次数 · 协作式取消"]
    UI -->|"任务 / 配置 / 停止请求"| RUN
    RUN <-->|"多模态消息 / 决策 JSON 文本"| IF

    subgraph OWNER["Environment Runtime · 唯一仿真 owner 线程"]
        RT["命令队列 / Future<br/>持续推进仿真与工具任务<br/>检查授权和 turn_id"]
        AS["AgentStateMachine<br/>对话轮次状态<br/>THINK / EXECUTE / OBSERVATION / END"]
        FS["FlightStateMachine<br/>控制权：MANUAL / AGENT<br/>ON_GROUND / HOVERING / FLY_AGENT"]
        TOOLS["本地工具注册表<br/>takeoff / land / fly_waypoint<br/>iter_execute 协作执行"]
        OBS["可替换 ObservationProvider<br/>位置 / 姿态 / 飞行状态 / 仿真时间<br/>第一视角 RGB · 可选深度 · 扩展传感器"]
        ENV["MuJoCo / MMUAVEnv<br/>自动起降 · 位置目标 provider<br/>SE3 控制回路 · env.step()"]
        PREVIEW["独立预览渲染器<br/>缓存最新图像与状态快照"]
        RT --> AS
        RT --> FS
        RT --> TOOLS
        RT --> OBS
        RT --> ENV
        TOOLS -->|"起降命令 / 轨迹目标"| ENV
        TOOLS -->|"执行完成，更新状态"| AS
        TOOLS -->|"飞行阶段切换"| FS
        ENV -->|"实际状态 / 同一时刻的观测快照"| OBS
        ENV --> PREVIEW
    end

    PLAN["轨迹规划 executor · 独立工作线程<br/>多项式轨迹优化 / IPOPT<br/>支持不等时间间隔航点"]
    TOOLS <-->|"规划请求 / 优化后的轨迹"| PLAN
    RC["遥控器输入<br/>Axis4：Agent 授权<br/>Axis7：手动起降"]
    RC --> ENV
    ENV -->|"轮询授权并同步两套状态机"| FS
    ENV -->|"撤权：禁用并终止当前轮次"| AS
    RUN <-->|"任务 / 结果 / 观测：命令队列与 Future"| RT
    RT -->|"撤权通知 → 取消事件"| RUN
    PREVIEW -->|"缓存快照 / 最新预览帧"| UI
    RUN -->|"EventBus：对话、阶段、JSON 日志、错误"| UI

    classDef planned fill:#fff4d6,stroke:#b7791f,stroke-dasharray:5 5;
    class LLAMA planned;
```

Runner 不直接调用 `env.step()`，也不直接修改状态机。桌面运行时，状态机修改、工具任务推进、观测采集及 renderer 生命周期均归环境线程；轨迹优化等待和模型推理期间，仿真仍持续运行。Qt 线程读取缓存并发送请求，不访问实时 `MjData`。工具的同步 `execute()` 仅保留给无 runtime 的独立脚本和测试。

对应入口：[`app.py`](../app.py)、[`agent/runner.py`](../agent/runner.py)、[`agent/backends.py`](../agent/backends.py)、[`runtime/environment.py`](../runtime/environment.py)、[`ui/main_window.py`](../ui/main_window.py)。llama.cpp 目前尚无后端适配器或界面选项；图中仅标出将来的接入位置，后续需适配统一 `generate` 接口。

## 2. Agent 状态机：一轮对话的生命周期

```mermaid
stateDiagram-v2
    direction LR
    state "END：本轮结束 / 初始待命" as END
    state "THINK：模型生成、JSON 解析、工具选择" as THINK
    state "EXECUTE：实际执行本地工具" as EXECUTE
    state "OBSERVATION：采集工具后的状态与图像" as OBSERVATION

    [*] --> END
    END --> THINK: 已授权、模型就绪、worker 空闲；start_turn
    THINK --> THINK: JSON 非法且未超限；携具体错误请求修正
    THINK --> EXECUTE: 合法 tool 决策；complete_think(TOOL)
    THINK --> END: 合法 final 决策；complete_think(FINAL)
    EXECUTE --> OBSERVATION: 工具成功或可反馈的失败；complete_execute
    OBSERVATION --> THINK: 采集成功并复查授权与轮次；complete_observation
    THINK --> END: 终止 / 撤权 / 运行错误 / 次数超限
    EXECUTE --> END: 终止 / 撤权 / 未恢复的运行错误
    OBSERVATION --> END: 终止 / 撤权 / 运行错误
```

源码：[`agent/state_machine.py`](../agent/state_machine.py)。状态类只负责状态与转换校验，模型调用和任务取消由 runner/runtime 实现。

- `END` 是初始状态；只有 `END` 且 Agent 已启用才能 `start_turn()`，每轮递增 `turn_id`。应用还要求模型就绪、上一项后台工作已退出，因此逻辑上进入 `END` 不代表可以立即并发发起下一轮。
- `THINK` 包含生成、JSON 解析和工具调用决策；实际工具执行属于 `EXECUTE`。JSON 修正不切换状态，保留原任务和观测，返回具体错误与“请修正上一条回复的 JSON 格式，保持原任务和判断不变”。当前解析器也兼容单个完整的 Markdown JSON 代码块。
- 初次 `begin_turn()` 在进入 `THINK` 后采集初始观测；只有工具后的反馈采集走 `OBSERVATION`。观测采集函数失败时本身保留该状态，桌面 runner 捕获异常后请求终止本轮。
- `terminate(reason)` 可从任意状态进入 `END`；`set_enabled(False)` 同时终止并禁止开始新轮次。恢复授权只重新启用 Agent，不自动继续旧轮次。
- 取消是协作式的：HF 在生成步间检查事件，HTTP 后端取消请求；已提交的 GPU / IPOPT 运算无法保证立即退出。`turn_id` 与授权复查阻止迟到结果执行工具或覆盖新轮次。

## 3. 无人机状态机与手动 / Agent 控制权

```mermaid
stateDiagram-v2
    direction LR
    state "ON_GROUND：在地面" as GROUND
    state "HOVERING：位置控制 / 悬停" as HOVER
    state "FLY_AGENT：执行 Agent 航点轨迹" as FLY

    [*] --> GROUND
    GROUND --> HOVER: 自动起飞完成；Agent takeoff 或手动 Axis7
    HOVER --> GROUND: 自动降落完成；Agent land 或手动 Axis7
    HOVER --> FLY: Agent 已授权；fly_waypoint 规划后开始执行
    FLY --> HOVER: 最终航点稳定悬停完成；或成功安装中断保持指令
```

```mermaid
flowchart TB
    DETECT{"启动时检测到手柄？"}
    DETECT -->|"是"| AXIS{"持续读取 Axis4"}
    DETECT -->|"否；桌面入口启用自动授权"| AGENT
    AXIS -->|"大于 0.5，视作 1"| AGENT["AGENT 控制权<br/>启用 Agent 状态机<br/>只使用 Axis4，屏蔽其余遥控器控制输入"]
    AXIS -->|"未授权"| MANUAL["MANUAL 控制权<br/>Agent 禁用；提交任务时提示，不发送 prompt<br/>仅允许 ON_GROUND / HOVERING"]
    AGENT -->|"Axis4 从授权变为未授权"| REVOKE["清除轨迹与自动起降目标<br/>锁定当前位置与 yaw<br/>Agent → END，取消当前模型 / 工具任务<br/>飞行状态 → HOVERING"]
    REVOKE --> MANUAL
    MANUAL -->|"Axis4 重新授权"| AGENT
    MANUAL --> RC["Axis7 边沿：触发自动起飞 / 降落<br/>HOVERING 下摇杆可移动无人机<br/>控制器始终采用位置模式"]
```

源码：[`env/flight_state_machine.py`](../env/flight_state_machine.py)、[`env/mmuav_env.py`](../env/mmuav_env.py)。控制权是 `ControlMode` 字段，与三种飞行状态共同维护；不是第四种飞行状态。

- 禁止 `ON_GROUND ↔ FLY_AGENT` 直接切换，必须经过 `HOVERING`。手动模式不能进入 `FLY_AGENT`；手动移动仍标为 `HOVERING`。
- 起飞 / 降落执行期间保留原飞行状态，完成后才变为目标状态。`fly_waypoint` 仅从 `HOVERING` 开始，规划时保持位置，执行轨迹时进入 `FLY_AGENT`。
- 最终航点的位置、速度、yaw、角速度均满足容差，且连续稳定 `hover_seconds`（默认 1 秒）后，航点工具返回成功：飞行状态进入 `HOVERING`，Agent 进入 `OBSERVATION`。中间航点不逐个悬停。
- 工具超时 / 异常会尝试安装当前位置保持指令；不能把失败或超时视为到达目标。保持失败时可能保留原飞行状态，并在结果中报告 `hold_error`。撤权则优先恢复手动目标，旧工具不得覆盖它。
- 当前撤权实现会强制进入 `HOVERING`，即使之前在地面；该状态表示位置保持控制语义，不是接触传感器对离地或稳定性的确认。
- 桌面入口无手柄时自动授权，既不自动起飞也不自动推理；终止按钮仍可用。此行为由 `auto_agent_without_joystick=True` 开启，独立构造环境的默认行为不同。

### 两个状态机如何配合

| 事件 | Agent 状态 | 无人机状态 |
| --- | --- | --- |
| 接收任务并采集初始观测 | END → THINK | 保持当前状态 |
| 选择 takeoff | THINK → EXECUTE | 执行时 ON_GROUND，完成后 HOVERING |
| 选择 fly_waypoint | THINK → EXECUTE | 规划时 HOVERING，飞行时 FLY_AGENT |
| 航点完成并稳定悬停 | EXECUTE → OBSERVATION | FLY_AGENT → HOVERING |
| 选择 land 并完成 | THINK → EXECUTE → OBSERVATION | HOVERING → ON_GROUND |
| 工具后观测采集完成 | OBSERVATION → THINK | 保持当前状态 |
| 模型输出 final | THINK → END | 保持当前状态；不会自动降落 |
| 用户终止 | 任意状态 → END | 取消运动并保持位置；运动中转为 HOVERING |
| Axis4 撤销授权 | 任意状态 → END，并禁用 | 强制 HOVERING，恢复手动控制 |

## 4. 模型与 Agent 的通信格式

```mermaid
flowchart LR
    TASK["原任务 + 有限长度历史<br/>system prompt：工具协议 / NWU 坐标 / 规划规则"]
    PACK["ObservationPacket<br/>JSON：位置、姿态、状态、仿真时间<br/>附件：RGB 224×224<br/>可选深度：默认 56×56"]
    MSG["Runner 组装 messages<br/>system / user / assistant<br/>user.content 为任务、观测、工具结果的 JSON 字符串<br/>图像索引绑定对应观测消息"]
    ADAPT["后端转换<br/>HF：视觉处理器 + PIL<br/>DeepSeek：文本 + PNG base64 image_url<br/>待接入 llama.cpp：HTTP 多模态适配"]
    OUT["统一返回：text 字符串<br/>可附 model / usage / 耗时 / 推理内容"]
    PARSE{"parse_decision<br/>解析并校验协议"}
    TOOL["tool 决策<br/>本地执行工具 → 结果 + 新观测"]
    FINAL["final 决策<br/>向用户显示文本 → END"]
    FIX["非法 JSON / 字段 / 参数<br/>具体错误 + 保持任务与判断的修正要求"]
    TASK --> MSG
    PACK --> MSG
    MSG --> ADAPT --> OUT --> PARSE
    PARSE -->|"合法工具"| TOOL
    PARSE -->|"合法最终回答"| FINAL
    PARSE -->|"非法且重试未超限"| FIX
    FIX --> MSG
    TOOL --> PACK
```

### 发给模型的文本与图像

以下是首条 `user.content` 解析后的示例；在实际 `messages` 中该对象被序列化为 JSON 字符串。图像通过附件传送，不把 base64 混入文本 prompt。

```json
{
  "task": "向北飞行 1 米",
  "observation": {
    "type": "uav_observation",
    "schema_version": 1,
    "sim_time_seconds": 12.5,
    "coordinate_frame": "NWU",
    "position_m": [0.0, 0.0, 1.0],
    "attitude": {
      "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
      "euler_rpy_rad": [0.0, 0.0, 0.0],
      "rotation_convention": "body_to_world; Rz(yaw) Ry(pitch) Rx(roll)"
    },
    "flight_state": "hovering",
    "rgb": {
      "camera": "fpv_camera",
      "width": 224,
      "height": 224,
      "color_space": "RGB",
      "mime_type": "image/png",
      "encoding": "attachment",
      "attachment": "rgb"
    }
  }
}
```

工具后的同类消息额外带 `tool_result`，并再次带上原 `task`。最新 RGB / 可选深度始终绑定该条观测；JSON 修正消息不会改变绑定。旧观测文本保留，旧图像标记 `not_retained`，不重复发送附件。`ObservationPacket.to_json()` 也支持自包含 PNG/base64 JSON，但桌面 runner 使用附件路径。

深度默认不发送，开启后附加 `observation.depth` 元数据和第二张图：相机前向 Z 深度（米），灰度固定编码 0～10 米，白近黑远，附 3×3 区域最近深度等统计。模型观测分辨率与界面预览分辨率独立。未来点云可通过替换 `ObservationProvider` 或 `extra_sensors → extras` 扩展，当前尚未实现点云观测。

### 模型返回的决策

每次只返回下列四类对象之一，无额外字段；当前使用自定义 JSON 决策协议，不是 OpenAI 原生 `tool_calls` / `role=tool` 消息协议。DeepSeek 的 JSON Output 只要求对象格式，工具字段和参数合法性仍由本地校验。

```json
{"tool":"takeoff"}
```

```json
{"tool":"land"}
```

```json
{"tool":"fly_waypoint","waypoint":[{"time":3.0,"position":[0.4,0.0,1.0],"yaw":0.0},{"time":5.5,"position":[1.0,0.0,1.0],"yaw":0.0}]}
```

```json
{"final":"已完成任务。"}
```

`waypoint` 必须是显式非空数组，数量由模型决定；`time` 是相对此次工具开始的秒数，必须为正且严格递增，间隔可不同。`position` 是世界系绝对坐标（米），NWU 的 +X 向北、+Y 向西、+Z 向上；`yaw` 单位为弧度。前后左右需按任务开始时的 yaw 换算为世界目标，不能直接当世界轴方向。

`tool_result` 的字段随工具及结果不同，状态包含 `success`、`timeout`、`error`、`terminated`。例如航点成功结果带实际位置、yaw、误差、执行时长和稳定悬停时长。Runner 将结果作为观测消息中的 JSON 数据回传，模型应据实际状态判断完成，而不是把合法调用或 `final` 文本当作已到达的证明。

通信实现见 [`agent/protocol.py`](../agent/protocol.py)、[`agent/messages.py`](../agent/messages.py)、[`observations/base.py`](../observations/base.py)、[`agent/ministral.py`](../agent/ministral.py)、[`agent/deepseek.py`](../agent/deepseek.py)。当前轨迹优化器不包含障碍约束或碰撞检测；RGB / 深度与提示词支持模型规划，但不构成可靠避障保证。
