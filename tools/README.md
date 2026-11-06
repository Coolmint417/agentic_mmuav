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
- `attached`：必须为 JSON 布尔值。完成后运动状态均为 `ON_GROUND`，
  `true` 将 `sucker_states` 设置为四个 true；`false` 设置为四个 false，碰撞支撑始终保留。
- 仅允许 `ON_GROUND` 调用，并要求 Axis4 授权和 agent `EXECUTE`。
  执行时进入 `CRAWLING`，完成收腿后返回 `ON_GROUND`，吸盘开关独立保存。其他状态返回 error。
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
模型必须使用 `true/false`。表面支撑静止时可调用 takeoff，起飞前会通过统一状态接口释放四吸盘。


## 大角度自主吸附 aggressive_perch

```json
{"tool":"aggressive_perch","position":[2.0,0.0,1.5],"yaw":0.0,"wall_angle_deg":90.0,"time":2.0}
```

入口仅接受上述字段：`position` 是世界 NWU 坐标系中的一个**表面接触区域中心**（米），
不是机体中心；`yaw` 单位为弧度；`wall_angle_deg` 是平面相对水平面的倾角，范围 30～90 度；
`time` 是从轨迹执行开始到计划接触时刻的仿真秒数，范围 `(0,2]`，另有接触稳定确认时间。
不接受航点数组。要求 Axis4 授权、agent EXECUTE、无人机 HOVERING 且没有自动起降任务。

按本工程约定，期望姿态 `R_T=Rz(yaw) Ry(-angle)`，朝向机器人自由空间的表面外法线：

```text
n = [-sin(angle)*cos(yaw), -sin(angle)*sin(yaw), cos(angle)]
a(T) = alpha*n + model.opt.gravity
8 <= alpha <= 20  # m/s²，由优化器联合求解
v(T) = -0.4*n
```

角度代入 sin/cos 时转换为弧度。yaw=0、angle=90° 对应从较小 X 一侧接近墙面；
yaw=pi/2 则从较小 Y 一侧接近。机体 +Z 指向外法线，底部吸盘朝向墙内，因此接触速度与
外法线方向相反。若机器人初始点不在输入所定义的外法线一侧，工具会返回 error。

沿用论文 IV-B 式 (10)、(11) 的方向约束，比推力不再固定为 12，而是作为额外决策变量
与多项式系数一起最小化 snap 代价。默认界限为 8～20 m/s²，上界同时受电机最大推力/质量限制；
下界用于保留末端姿态跟踪余量。实验中放宽到 6 曾出现姿态跟踪失败，因此默认采用 8。
终端轻触速度仍为 0.4 m/s，位于论文给定的 0.2～0.8 m/s 区间。

提示词要求距墙法向距离至少 2 m，强调快速接近，约 2 m 距离优先选 1.5～1.8 s，最多 2 s。
工具额外硬校验 `clearance/time >= 1 m/s`，不满足即返回错误，不执行轨迹。
这是平均法向接近速度，不是触壁瞬间的速度，不会把接触速度提高到 1 m/s。
`AggressivePerchConfig` 可调整比推力范围、平均速度门槛、`minimum_height=0.25` 等内部参数；
模型调用仍只有原来的 position/yaw/wall_angle_deg/time。

执行过程：

1. 打开四个吸盘的 adhesion，并设置高摩擦，保持原悬停控制，随后才请求优化器规划。
2. 从模型吸盘 mesh 的底面求四吸盘接触区域相对机体原点的偏移，转换表面点为机体目标；
   再加入默认 2 cm 的法向预压参考偏移，用来克服跟踪误差并促成接触。这是参考轨迹偏移，
   不修改墙面、不关闭碰撞、不强制写入机器人位姿。
3. 调用现有 PolynomialTrajectoryOptimizer 连接起点与唯一目标，固定末端位置、法向速度，
   以 `a(T)=alpha*n+gravity` 约束加速度方向并求解 alpha，两端 jerk 为零。
   对参考机体原点施加世界 Z≥0.25 m 下界，先在每段 101 个时刻施加线性约束，
   再求导并对归一化多项式求根，检查采样点之间的真正极值；有越界就补约束重求解，
   最多 8 次仍未通过则报错。随后按 201 个时间样本检查前馈推力不超过电机总推力限制。
   Z 约束保持原末端加速度方向，不会把竖直墙面接触所需的 -g 改为零。
4. 通过 SE3/角速度控制器执行；飞行状态进入 FLY_AGENT。专用轨迹 provider 保留末端加速度，
   不使用 fly_waypoint 末端清零逻辑，并加入由 jerk 得到的姿态角速度前馈。
   吸附执行期间临时使用 `kx=40, kv=12, kR=12`，结束或中断恢复原增益。
5. 实际检测到至少两个吸盘与目标法线一致的外部表面接触、机体 +Z 对齐及位置误差合格后停桨，
   然后要求四个吸盘均接触、线速度与角速度合格并持续 0.3 仿真秒。
6. 只有确认上述稳定吸附，才返回 success，运动状态 FLY_AGENT → ON_GROUND（表面支撑），agent EXECUTE → OBSERVATION；
   保持四吸盘开启并清除轨迹 provider，旋翼输出为零。

未接触、只有部分吸盘接触或持续不稳定，均不因时间到达而报告成功。默认接触超时是轨迹时间之后
1 秒，超时返回 timeout。规划失败、超时、用户终止及 Axis4 撤销都会清理本次吸附和临时控制设置；
仍有控制权时安装当前位置悬停指令，撤权时不覆盖手动保持目标。

SE3 姿态构造会正规化旋转矩阵，并在吸附模式下使用连续的机体 Y 轴参考，避免俯仰接近 90°
时航向向量与推力方向共线造成姿态翻转。原有航点工具接口和末端悬停行为保持不变。

验证包含真实有限尺寸 box 表面上 30°、60°、75°、90° 与非零 yaw 的吸附，停桨后继续仿真确认接触保持；
另覆盖不存在表面时的 timeout、规划失败、撤权、生成器关闭，以及完整 Runner → 工具 → 观测回路。
这些是指定初始距离约 2.1 m、轨迹时间 2 s 的测试工况；另验证日志中的两处约 1 m 高起点，
1.5～2 s 的六组工况均可吸附，结果见 artifacts/diagnostics/aggressive_perch_fast/physical_sweep.json。
这些验证不代表任意起点、任意时间都能成功，参考 Z 下界也不是实际全机体碰撞保证。
本工具未实现论文中的 RL 修正器、全程扭矩/角速度约束或碰撞避障；推力采样检查也不是连续时间可行性证明。


## 独立四吸盘状态

运动状态只有 `ON_GROUND / HOVERING / FLY_AGENT / CRAWLING`。
`FlightStateMachine.sucker_states` 是独立的四元 bool 元组，固定顺序：
左前 `sucker_upleft`、左后 `sucker_bottomleft`、右后 `sucker_bottomright`、右前 `sucker_upright`。
它表示 adhesion 开启命令，不是接触传感器；空中也可开启，开启不等于吸附成功。

```python
machine.set_sucker_states([True, False, True, False])
machine.set_sucker_state(1, True)  # 0-based，单独修改左后吸盘
# 模拟器 owner 线程内立即同步物理控制及摩擦：
environment.set_sucker_states([True, True, True, True])
# 外部线程经 Runtime 提交，独立于 Agent 对话和授权状态：
runtime.set_sucker_states([False, False, False, False]).result()
```

所有爬行步态、遥控器启停、起飞释放、吸附准备及失败清理均通过同一向量更新。
控制回调每步根据该向量应用四路 adhesion 和对应高/低摩擦，因此直接修改 FSM 命令也会在下一步生效。
爬行控制仍按步态周期继续写入后续吸盘命令。吸盘 setter 本身不改变运动状态或控制权。
`ON_GROUND` 表示表面支撑静止，包含大角度墙面吸附；稳定 perch 从 FLY_AGENT 直接进入 ON_GROUND，
无需伪造中间悬停状态。保持这一状态时旋翼关闭，开启/释放任意吸盘不会自动启动旋翼。
普通地面待机撤权保持 ON_GROUND；自动起飞/降落中撤权仍取消自动任务并恢复悬停控制。
当前观测、过程采样、工具结果及 Runtime/UI 状态均回报 `sucker_states`，完整观测另含 `sucker_order`。


工具结果的 `planning` 字段记录优化后的比推力及界限、参考最低 Z、Z 下界、最大采样推力、
平均法向速度与轨迹时间。成功和执行超时均保留这些字段，便于判断规划与跟踪的问题。
更换优化器后，原来的固定 12 m/s²、3～4 s 低空单段轨迹不再是默认执行方案。
