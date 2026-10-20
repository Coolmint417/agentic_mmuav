THE VEHICLE IS CONSTRUCTED IN NWU COORDINATION

## 飞行状态机

`flight_state_machine.py` 统一记录飞行状态和控制权。默认 MANUAL / ON_GROUND。

```text
在地面（ON_GROUND） ↔ 悬停/手动运动（HOVERING） ↔ agent 轨迹飞行（FLY_AGENT）
```

MANUAL 下仅允许 ON_GROUND / HOVERING。AGENT 授权后才允许进入 FLY_AGENT，
它不能直接与 ON_GROUND 切换；非法切换抛出 `InvalidFlightTransition`。
`MMUAVEnv`、控制回调与三个 tools 使用同一实例，自动起降完成时由环境更新状态。

## 遥控器与控制权

飞行始终使用位置控制（SE3 → Rate PID → Mixer）。不再读取 Axis5 或切换姿态/定高模式。

| 信号 | 手动模式 | agent 模式 |
| --- | --- | --- |
| Axis4 | >0.5（开关为1）授权 agent | ≤0.5（开关为0）立即撤销授权 |
| Axis0/1/2/3 | HOVERING 时控制水平速度、升降速度和 yaw | 不读取，不影响目标 |
| Axis7 | 负→正起飞；正→负降落 | 不读取 |
| Axis5 | 不读取 | 不读取 |

手动 ON_GROUND 下运动摇杆不推进飞行目标，需由 Axis7 起飞进入 HOVERING。
HOVERING 允许手动飞行，不要求速度为零。Axis7 自动起降期间沿用原来的自动控制和完成判据。

Axis4 授权只启用 agent，不自动创建新对话，也不立即进入 FLY_AGENT。
桌面入口额外开启 `MMUAVEnv(auto_agent_without_joystick=True)`：若启动时真实
`JoystickReader` 无可用设备，使用持续授予 Agent 权限的输入适配器；初始仍是 ON_GROUND / END，
不自动起飞或启动模型。手柄可用时不启用该适配器，仍完整遵循 Axis4。
独立环境默认不开启自动授权；测试注入的遥控器仍按自身 Axis4 行为运行。
agent 的 THINK/等待和起降工具仍可处于 HOVERING 或 ON_GROUND；`fly_waypoint` 执行时进入 FLY_AGENT。
撤销授权会：

1. 禁用 agent 并使其进入 END，拒绝新对话及后续状态推进。
2. 清除轨迹 provider 和正在执行的自动起降，锁定实际位置与 yaw。
3. 飞行状态进入 HOVERING，恢复手动输入；切换当步先保持位置，下一步响应运动摇杆。
4. 同步 Axis7 当前开关状态，避免回放接管期间的起降翻转。

重新授权后 agent 保持 END，由调用方重新 `start_turn()`；旧工具结果不能重启已取消的轨迹。

## 使用

从工程根目录使用：

```python
from env.flight_state_machine import FlightState, FlightStateMachine

machine = FlightStateMachine()
machine.transition_to(FlightState.HOVERING)
machine.set_agent_authorized(True)  # 纯状态机示例；仿真中由 Axis4 驱动
machine.transition_to(FlightState.FLY_AGENT)
print(machine.state)
print(machine.allowed_transitions)  # 此时只允许转到悬停
```

`can_transition_to(target)` 检查转换是否允许。
仿真入口是 `MMUAVEnv(agent_state_machine=agent)`，也可用 `environment.agent_state_machine`。
三个工具会绑定同一 agent 状态机，手动模式下不能启动工具。

`environment.set_model_stop_callback(callback)` 预留模型取消接口。Axis4 撤销时先禁用 agent、
清除控制目标，再调用 callback。Environment Runtime 已接入 AgentRunner 的取消事件，
推理在独立线程运行，生成结果按轮次校验；环境持续 stepping 和处理遥控器。
独立脚本仍需持续 stepping 或调用 `environment.poll_control_authority()`。
每个仿真控制步都会轮询 Axis4，工具在步间检查终止和权限。

测试可传入 `joystick_reader` 模拟遥控器，不能用自动安装 provider 的方式绕过 Axis4。
