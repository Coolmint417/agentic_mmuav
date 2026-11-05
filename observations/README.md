# 无人机 Observation

此目录提供可替换的观测接口、MuJoCo 默认实现和模型输入打包逻辑。
观测按需采集，不恢复 clip 录制，不写入视频或数据集。

## 默认字段

`MuJoCoObservationProvider` 采集：

| 字段 | 含义 |
| --- | --- |
| `type` / `schema_version` | `uav_observation` / `1` |
| `sim_time_seconds` | 当前环境从启动起累计的仿真秒数，即 `MjData.time`；不从观测模块创建时重新计时 |
| `coordinate_frame` | NWU：x 北、y 西、z 上 |
| `position_m` | 机体 Body 自由关节参考点的世界位置 `[x,y,z]`，单位米 |
| `attitude.quaternion_wxyz` | 机体到世界坐标的单位四元数，顺序 w/x/y/z |
| `attitude.euler_rpy_rad` | `[roll,pitch,yaw]`，单位弧度，旋转矩阵约定为 `Rz(yaw) Ry(pitch) Rx(roll)` |
| `flight_state` | 共享飞行状态机的实际值：通常为 `on_ground` 或 `hovering`；运动中采集会如实返回 `fly_agent` |
| `rgb` | 第一视角 `fpv_camera` 的 RGB 图像及其元数据，默认 224×224 |
| `depth` | 可选的同一快照第一视角深度图、距离编码及 3×3 区域最小深度；默认 56×56 |
| `extras` | 配置扩展传感器时才出现的附加 JSON 数据 |

采集在私有 `MjData` 副本上刷新相机位姿，确保 RGB 和运动数据属于同一快照。
不会 stepping，不修改实际 `qpos/qvel/ctrl`，不触发全局控制回调。
必须在仿真 owner 线程、两次 stepping 之间调用；不支持同时在另一线程修改环境数据。

## 独立采集与 JSON 传输

```python
from env.mmuav_env import MMUAVEnv
from observations import MuJoCoObservationProvider, ObservationConfig

with MMUAVEnv() as environment:
    with MuJoCoObservationProvider(
        environment, ObservationConfig(width=224, height=224, camera_name="fpv_camera")
    ) as observer:
        packet = observer.collect()
        json_prompt = packet.to_json()
```

`collect()` 可用于手动模式调试，不主动修改 agent 状态。
分辨率和相机在构造模块时配置，修改后重新创建 provider 即可。
模块持有可复用的离屏 renderer，需要 `close()` 或使用 `with`，并在关闭环境前释放。

`to_json()` 返回完整、可独立传输的 JSON。RGB 以无损 PNG/base64 存入 `rgb.data`：

```json
{
  "type": "uav_observation",
  "schema_version": 1,
  "sim_time_seconds": 6.0,
  "coordinate_frame": "NWU",
  "position_m": [-2.0, 2.0, 1.0],
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
    "encoding": "base64",
    "data": "...PNG base64..."
  }
}
```

## 回传本地模型与状态切换

本地 Transformers 通过视觉处理器接收图片，所以应调用 `to_model_inputs()`：

```python
from observations import collect_for_agent

# environment 和 backend 已创建；此前工具已将 agent 推进到 OBSERVATION。
with MuJoCoObservationProvider(environment) as observer:
    packet = collect_for_agent(observer, environment)
    # 成功采集后已进入 THINK；图像单独交给视觉处理器。
    result = backend.generate(**packet.to_model_inputs())
```

`to_model_inputs()` 返回 `prompt`（状态 JSON + 图像附件描述）和 `image`（PIL RGB），
可直接传给已有 `MinistralBackend.generate()`。base64 不进入文本 token；RGB 仍通过视觉编码器输入。
开启深度时还返回 `extra_images={"depth": PIL.Image}`；本地 Ministral 和 DeepSeek 后端
按 RGB、depth 顺序发送两张图片，并保持它们绑定到同一条观测消息。

## 深度观测

```python
config = ObservationConfig(include_depth=True, depth_size=56)
with MuJoCoObservationProvider(environment, config) as observer:
    packet = observer.collect()
    # packet.images["rgb"]: 224×224; packet.images["depth"]: 56×56
    # packet.data["depth"]["sector_min_m"]: 3×3 区域的最近可见表面深度
```

MuJoCo renderer 直接输出米制相机 Z 深度，表示沿相机光轴的深度，非到像素表面的欧氏距离。
RGB 与深度使用同一个私有 MjData 快照，深度单独渲染到所选分辨率，默认 56×56。
深度编码为可供视觉模型读取的三通道灰度 PNG，固定 0～10 m 范围：
`I = round(255 * (1 - clip(depth_m, 0, 10) / 10))`，白近、黑远；
黑色同时表示超出 10 m 显示范围或无回波，不能据此认定空间可通行。
不是每帧按 min/max 重新归一化，因此不同观测的灰度尺度一致，量化间隔约 3.9 cm。
原始浮点深度不作为 56×56 数组塞入 prompt；附带米制 `minimum_visible_m` 和
`sector_min_m` 九个数值（图像从上到下、从左到右的 3×3 区域最小可见深度，空区域为 null），
以及 FOV、单位、范围和像素解码规则。区域统计保留实际距离，不受 10 m 灰度裁剪限制。
这种图像和区域最小值不等于点云、占据图或碰撞检测，也不能保证整机轨迹无碰撞。

在 Agent 设置中勾选深度开关、选择边长 8～1024，点击应用后对下一轮生效。
默认关闭发送以保持原行为，开启后的默认分辨率为 56×56。
预览区“第一视角深度”标签始终可用，与模型深度开关、模型分辨率独立。
模型服务可能重采样图片，56×56 输入并不意味着固定的视觉 token 数。

`collect_for_agent(provider, environment)` 要求共享 agent 处于 OBSERVATION 且 Axis4 已授权。
采集失败时保留 OBSERVATION；采集成功后复查权限、状态和 `turn_id`，再进入 THINK。
Axis4 撤销或轮次已变化时抛出 `ObservationCancelled`，不交付过期观测，也不推进新轮次。
它只完成观测和状态推进，不启动推理。桌面应用由 Environment Runtime 在 owner 线程调用它，
AgentRunner 经服务 API 获取观测，并在每次推理前检查权限；环境同时持续处理 Axis4。

## 替换与扩展

整套实验可替换为任意 `ObservationProvider` 子类，只需实现 `collect() -> ObservationPacket`，
有设备资源时再实现 `close()`。`collect_for_agent()` 与具体 provider 无关。
`observations.base` 不依赖 MuJoCo，其他环境或点云实验可复用同一传输和状态推进接口。

给默认观测追加传感器可传入 `extra_sensors`：

```python
with MuJoCoObservationProvider(environment, extra_sensors={
    "linear_velocity_mps": lambda model, snapshot: snapshot.qvel[:3].tolist(),
}) as observer:
    packet = observer.collect()
    # packet.data["extras"]["linear_velocity_mps"]
```

每个传感器函数接收 `(model, snapshot_data)`，必须只读，返回有限的 JSON 值。
未来点云可通过同一入口返回坐标系、单位、点数组等；NumPy 数组需先 `.tolist()`。
附加数据归入 `extras`，不覆盖核心字段。需要新图像模态或不同数据源时可替换 provider。

## 测试

无窗口运行时，在导入 MuJoCo 前选择 EGL 后端：

```bash
conda activate mm_planner
MUJOCO_GL=egl OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 python -m unittest discover -s tests -v
```

测试包含真实 FPV 渲染、PNG/JSON 往返、分辨率调整、姿态/时间、采集副作用检查、
扩展及替换 provider、采集失败、Axis4 撤销和轮次隔离，以及现有控制/工具回归测试。


## 工具执行过程观测

Environment Runtime 在工具执行期间按仿真时间采样，默认每 1 秒一次。
Agent 设置页的“过程观测采样间隔”可修改该值，下一轮生效；最大执行轮数默认 32，
历史消息数上限默认 24。过程帧没有数量上限，不裁剪、不均匀抽帧，也不发送过程深度。
采样 RGB 使用模型观测分辨率（默认 224×224），与界面高分辨率预览独立。

`observations/process.py` 的 `ProcessObservation` 缓存单次工具数据；每张 RGB 与运动数据
来自 provider 的同一份私有 MjData 快照。采样不调用实时控制回调、不推进仿真。
当前观测新增 `velocity_m_s`（世界系）和 `angular_velocity_rad_s_body`（机体系）字段。
工具结束后 observation 增加以下数据，并通过命名附件发送过程 RGB：

```json
{
  "process": {
    "tool": "fly_waypoint",
    "sample_interval_seconds": 1.0,
    "start_time_seconds": 10.0,
    "end_time_seconds": 14.5,
    "coordinate_frame": "NWU",
    "angular_velocity_frame": "body",
    "frames": [{
      "time": 11.0,
      "elapsed_seconds": 1.0,
      "position": [1.0, 2.0, 1.5],
      "velocity": [0.3, 0.0, 0.0],
      "rpy": [0.0, 0.01, 0.0],
      "angular_velocity": [0.0, 0.0, 0.0],
      "state": "fly_agent",
      "rgb_attachment": "process_rgb_000000"
    }]
  }
}
```

position 单位 m，velocity 单位 m/s，rpy 单位 rad，角速度单位 rad/s，time 为启动后的仿真秒。
帧按时间排序，首帧在工具开始后的一个采样间隔；工具不足一个间隔时 frames 为空。
只记录实际经过的仿真快照，不为漏过的时间点复制图片；结束时刻的帧由当前观测表示。

DeepSeek 将每个过程 RGB 编码为 PNG/base64 image_url，附带附件名；HF 使用相同顺序的
命名 [IMG] 占位符。两者都将图像绑定到本次观测消息，JSON 重试时仍保持原绑定。
当前帧可继续按既有配置发送深度图，过程采样强制禁用深度。

刚结束工具的过程帧会用于本次推理与 JSON 修正重试。加入新观测或结束当前任务时，
删除旧过程逐帧数据及过程图片，不发送 `sample_count` 摘要。
每次观测的关键帧 RGB 和可选深度继续随其原始 user 消息保留，连同原位置、时间、工具结果
一起参与后续请求；历史消息被数量/字符预算裁剪时，其图片也同步不再发送。
Runner 内部 `_images` 保存实际图像，传输时重建 `message_images` 索引，不将图像对象写入
HTTP 消息或 JSON 日志。DeepSeek 为每条观测构造自己的多模态 content；HF 按消息顺序
排列各条观测的 [IMG] 和像素数据。
取消、Axis4 撤销、新任务及环境重载会清理过程缓存，避免跨工具、跨轮次混用。
为遵守过程帧无数量上限，当前工具逐帧数据不受历史文本字符预算裁剪；
API/本地模型的上下文、请求体与显存限制仍然适用。日志包含逐帧传感数据、附件名和尺寸，
不写入图片 Base64。图片只存在本次运行内存中。
