# 起飞与降落工具

`TakeoffTool` 和 `LandTool` 接受 JSON 字符串或 dict，只允许以下无参数调用。
单个完整的 JSON 代码块包装也可以；额外字段（包括 `arguments`）会被拒绝。

```json
{"tool": "takeoff"}
```

```json
{"tool": "land"}
```

| 工具 | 调用前无人机状态 | 成功后无人机状态 | 成功后 agent 状态 |
| --- | --- | --- | --- |
| takeoff | ON_GROUND | HOVERING | OBSERVATION |
| land | HOVERING | ON_GROUND | OBSERVATION |

调用前 Axis4 必须为1，agent 必须处于 EXECUTE。飞行中先结束航点运动回到悬停，再调用降落。
执行期间保留原飞行状态，操作成功后才切换。

## 复用环境控制逻辑

工具直接调用 `MMUAVEnv.start_auto_takeoff()` / `start_auto_land()`，
复用 `mmuav_env.py` 内的自动控制和完成判据，不复制起降控制算法：

- 起飞保持当前 X/Y，目标 Z 为 `clip(max(1.0, current_z + 0.8), 0, 10)`。
  高度误差小于 0.1 m、线速度小于 1 m/s 后完成，并锁定起飞目标位置。
- 降落保持 X/Y，以 0.25 m/s 下移目标高度，最低世界 Z=0。
  线速度小于 0.08 m/s 连续 1 秒后完成，锁定当前实际位置。
  平台降落采用实际位置；沿用速度判据，不增加接触判定或停桨逻辑。

环境始终使用位置模式，并在启动工具前校验 Axis4 授权。
自动起降完成后更新位置 provider，避免重新读取旧航点或起飞前的悬停目标。

## 调用

```python
from agent.state_machine import AgentStateMachine, ThinkOutcome
from env.mmuav_env import MMUAVEnv
from tools import TakeoffTool, LandTool

agent = AgentStateMachine()
with MMUAVEnv(agent_state_machine=agent) as environment:
    # 实际遥控器 Axis4 必须为1；手动模式下 start_turn() 会被拒绝。
    environment.require_agent_control()
    agent.start_turn()
    for Tool in (TakeoffTool, LandTool):
        agent.complete_think(ThinkOutcome.TOOL)
        result = Tool(environment, agent).execute({"tool": Tool.name})
        print(result)
        if result["status"] != "success":
            break
        agent.complete_observation()
    # 主循环继续处理观测，或在 THINK 中输出最终回答并进入 END。
```

`AutoFlightConfig(timeout_seconds=60.0)` 可以在构造工具时调整等待超时。
该配置是工程内部设置，不是模型调用参数。

非法调用返回 error 并进入 OBSERVATION，飞行状态不变。
已开始的起降若超时、异常或终止，会尝试停止自动起降并安装当前位置悬停指令。
安装成功标记 HOVERING，表示保持指令已生效，不表示已经稳定或确认离地；
安装失败保留原状态并返回 `hold_error`。主动终止保留 agent 的 END，
通过 `turn_id` 防止旧工具结果推进新轮次。

桌面应用通过 Environment Runtime 驱动工具的 `iter_execute()`，工具和 runner 都不直接推进仿真。
同步 `execute()` 仅作为无 runtime 的脚本/测试兼容入口。运行服务会持续控制无人机。
Axis4 在执行中撤销时，环境立即将 agent 置 END、飞行状态置 HOVERING 并恢复手动控制。
工具返回 terminated，不再安装悬停 provider 或覆盖手动目标。

## 验证

```bash
conda activate mm_planner
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

测试覆盖地面起飞→降落→再次起飞、平台降落后再次起飞、状态切换、非法调用、
超时、异常和终止，并同时运行航点工具与控制权切换测试。
第一版pyside界面左侧的第一视角和世界视角可以不同时出现，用户可以用一个类似左右箭头的控件切换是哪个视角。同时runner也应该读取一下axis4的值，如果不授权的话直接在对话框中返回相应提示，直接不给模型发promp


## 爬行工具 crawl

```json
{"tool":"crawl","steps":2,"attached":true}
```

- `steps`：模型决定的正整数，一个完整四阶段步态周期算一步，约 3.2 仿真秒。
- `attached`：必须为 JSON 布尔值。`true` 表示结束后吸附并进入 `ADHERED`；
  `false` 表示释放四个吸盘并进入 `ON_GROUND`，碰撞支撑始终保留。
- 仅允许 `ON_GROUND` 或 `ADHERED` 调用，并要求 Axis4 授权和 agent `EXECUTE`。
  `ADHERED` 启动时先经过 `ON_GROUND`，再进入 `CRAWLING`；完成时也经过
  `ON_GROUND`，按参数决定是否进入 `ADHERED`。其他状态返回 `error`，不执行动作。
- 沿机体前方爬行，不提供转向和后退。步数不是距离；收腿收尾期间仍可能发生位移。
  模型需要结合执行后的观测判断实际位置。
- 复用环境步态，由环境在阶段边界计步。计满后先收腿至中立位、恢复全部吸盘并等待，
  再应用最终吸附参数。成功后 agent 从 `EXECUTE` 进入 `OBSERVATION`。
- `CrawlTool.iter_execute()` 交给 Runtime 驱动，不直接调用 `env.step()`；
  同步 `execute()` 仅供独立脚本或测试使用。工具已经注册到 Runtime 和模型 JSON 协议、默认提示词。
- Axis4 撤销、用户终止、生成器关闭或异常时，请求环境完成收尾并保持吸附，
  不安装空中悬停指令。撤销授权后 agent 保持 `END`，环境继续独立执行物理收尾。
- 为防止释放吸盘时穿入表面，碰撞始终开启。吸附的一组吸盘使用
  `CRAWL_ATTACHED_FRICTION=3.0`，释放的一组使用 `CRAWL_RELEASE_FRICTION=0.05`，
  状态切换时同步更新滑动摩擦和吸附力；收尾后也按最终吸附状态保持对应摩擦。
  两种状态均提高接触优先级，使吸盘摩擦设置覆盖平台默认值。扭转、滚动摩擦沿用 XML。

无参数、缺失参数、额外字段、非正整数步数及 `attached: 0/1` 都会被拒绝。
模型必须使用 `true/false`。吸附状态下起飞仍会被拒绝，需要先完成一次以脱附结束的爬行。
