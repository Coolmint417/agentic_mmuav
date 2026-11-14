# 单帧平面观测 detect_planes

模型使用现有自定义 JSON 协议自主调用只读工具：

```json
{"tool":"detect_planes"}
```

不接受参数，不改变飞行目标、飞行状态、吸盘或控制权；任何已授权飞行状态都可调用。工具仍占用一次 Agent 决策，复用 THINK→EXECUTE→OBSERVATION→THINK、turn_id、取消和最大决策次数机制。

## 运行与配置

在 mm_planner 环境安装几何依赖：

```bash
python -m pip install -r requirements-planes.txt
python app.py --backend deepseek
```

本次已在本机 mm_planner 安装 `open3d-cpu==0.20.0`，使用其原生 `segment_plane`；SciPy 用于二维凸包与标注标签位置。Open3D CPU wheel 的 Python 3.13 支持见 [官方发布说明](https://github.com/isl-org/Open3D/releases)。新的 conda / desktop 安装清单均包含 requirements-planes.txt；缺失依赖时工具返回明确 error，不伪造检测结果。无需新增视觉模型。

## 实现文件

| 文件 | 改动 |
| --- | --- |
| observations/camera.py、observations/planes.py | 不可变 RGB-D 快照、标定、纯几何算法和标注 |
| observations/mujoco_observation.py、observations/base.py | owner 同步采集、renderer 生命周期、缓存快照交付 |
| tools/detect_planes.py、tools/tasks.py、tools/__init__.py | 工具接口、保存配置、纯计算请求及导出 |
| runtime/environment.py | 注册、共用 executor、取消及单次 mask/boundary 缓存 |
| agent/protocol.py、agent/runner.py | 自定义 JSON 协议、提示词和配置传递；复用现有多图后端 |
| ui/main_window.py | 默认勾选的保存开关、检测宽高设置 |
| requirements-planes.txt、requirements-desktop.txt、requirement.yaml | 几何依赖 |
| scripts/detect_planes.py | 人工检查样例和离线 NPZ 重算 |
| tests/test_planes.py、tests/test_plane_tool.py、tests/test_desktop_ui.py | 几何、仿真、HTTP、取消和 UI 验证 |
| README.md、tools/README.md、observations/README.md、docs/architecture.md、docs/changelog.md、本文件 | 使用方式、架构与局限 |

在 **Agent 设置** 中：

- 平面检测 RGB-D 宽度/高度：默认 **640×480**，可分别调整 16～2048。
- “detect_planes 完成后保存标注 RGB 和几何 JSON”：**默认勾选**。
- 修改后点击“应用设置”，下一轮任务生效。

检测分辨率与普通模型深度图（默认 56×56）、普通 RGB 和 UI 预览相互独立。640×480 深度仅在本地用于点云计算，不转换为发送给模型的深度图片，也不依赖“向模型添加深度信息”开关。该次检测反馈只含对应快照的 RGB 和 annotated_rgb。关闭保存仍会检测并发送真实标注图，只有本地写盘被关闭。

默认保存到仓库根目录 `artifacts/planes/`：

```text
obs_<唯一ID>.png    # 原生检测分辨率的 annotated RGB
obs_<唯一ID>.json   # 几何结果、原图尺寸、图像引用和各阶段耗时
```

## 同步与线程

1. MuJoCo owner 线程复制 MjData，并只在副本执行 mj_fwdPosition。
2. 从同一副本、同一相机渲染 RGB 与米制 Z-depth，读取相机位姿和传感数据。产生 observation_id 和 sim_time_seconds。
3. RGB、depth、K、光学系到世界系变换被复制为只读 NumPy 数组；几何 worker 不持有实时 MjData 或 renderer。
4. 复用已有轨迹 executor 处理 ComputationRequest，主仿真继续步进，无新增仿真或渲染线程。
5. 回到 owner 后复查权限与轮次，缓存本次 ObservationPacket，并在既有 collect_observation 通道交付。不会重新采一张新 RGB 再叠加旧 mask。

返回的时间是**采集时刻**，不是算法结束时刻；模型提示词明确这一点。图像处理期间仿真仍会前进，模型不能假定图像是计算结束时的最新视图。被取消或撤权的结果不发布，也不污染新轮次；取消信号在算法阶段之间及每次平面提取之间检查，正在执行的一次原生 RANSAC 调用不保证立即抢占。若取消恰好发生在 PNG 写盘期间，可能留下未交付的调试文件，不会作为模型反馈或几何缓存使用。

## 标定和几何

目前支持本工程的透视 fovy 相机。正交投影、显式 sensorsize 内参覆盖会返回不支持的错误，避免静默使用错误标定。

```text
fx = fy = H / (2*tan(fovy/2))
cx = (W-1)/2, cy = (H-1)/2
```

u/v 是左上像素中心为 (0,0) 的数组索引。当前 fovy=90°、640×480 对应 fx=fy=240、cx=319.5、cy=239.5。

渲染器输出米制光轴 Z-depth，反投影为右/下/前的光学坐标。MuJoCo camera 是右/上/后，因此：

```text
R_world_optical = cam_xmat @ diag(1, -1, -1)
t_world_optical = cam_xpos
p_world = R_world_optical @ [(u-cx)*Z/fx, (v-cy)*Z/fy, Z] + t_world_optical
```

cam_xpos 已包含相机相对机体前移 0.3 m 的外参；不可再重复补偿。相机位姿不由图像推断，来自同一快照的 MuJoCo camera 变换。过滤 NaN、Inf、非正/近裁剪范围深度及远裁剪背景。

流水线为：反投影→体素下采样→迭代多平面 Open3D RANSAC→PCA→原分辨率 mask 重建→唯一像素分配→原始支持点 PCA/二维凸包→半透明标注。

算法配置集中在 `observations/planes.py:PlaneDetectionConfig`：默认 max_planes=5、voxel_size_m=0.03、distance_threshold_m=0.015、mask_threshold_m=0.02、ransac_iterations=500、min_inliers=80、min_support_fraction=0.01；最多取 30000 个体素点用于 RANSAC，mask 仍在完整原始有效点云上重建。

每个候选使用平面局部二维范围和光轴深度范围约束原始 mask；重叠像素按归一化平面残差归属至多一个平面。采用原始支持点细化最终法向和质量，法向统一朝向相机；相机恰在平面上或支持近乎共线的退化候选被跳过。

不按法向方向筛选；水平面、天花板、墙面和任意倾斜面使用同一算法。倾角为 acos(abs(normal.z))，0°水平、90°竖直。相机到平面的距离是到无限平面的垂距，不是 Z-depth，也不是到有限边界的最近距离。

## 返回与图像绑定

tool_result 包含每个 P1/P2/... 的：世界 NWU 法向、平面 offset、观测支持点质心、倾角、相机垂距、内点数/比例、RMSE、凸包面积、局部二维 extent/切向轴，以及原始检测分辨率下的 bbox。另有原生尺寸、observation_id、时间、阶段耗时和标注附件引用。不发送点云、mask 全数组、boundary 全数组或图片 base64 文本。

mask、局部凸包 boundary 和基底保留在 runtime 最新一次检测缓存中，供未来扩展。新轮次、取消或重载清空；没有跨帧关联或长期地图。

标注在原始 RGB 分辨率绘制，再与对应 RGB 一起等比例缩放到模型输入尺寸范围。默认 640×480 会缩为 224×168，避免直接把 4:3 mask 叠到不同水平视场的 224×224 原有 RGB 上。

既有 ObservationPacket、历史图像绑定、HF 和 DeepSeek 多图发送机制直接承载 named attachment `annotated_rgb`，与 RGB 处于同一 user 观测消息。历史保留机制与其他观测关键帧相同。DeepSeek 的 HTTP content 中包含标注图的真实 PNG image_url，不只是 JSON image_id 字符串；HF 保持对应 IMG 占位符顺序。

P1/P2 等编号只在本次 observation_id 内有效，不是跨帧物体 ID。模型应结合用户描述与标注 RGB 选择目标，不可只按最大面积或某个倾角推断用户意图。工具不自动选贴附点、不判断吸附可行性、不改变控制器。

## 人工检查与离线重算

```bash
MUJOCO_GL=egl python -m scripts.detect_planes --map map6 --output artifacts/plane_demo
```

脚本默认在离线场景将机体初始位姿设为 [1.05,2,3]、yaw=0 后采集一次；不执行飞行工具，也不加载模型。支持 --position x y z、--yaw、--width、--height。
输出标注 PNG、原始 RGB、几何 JSON 和含 RGB、浮点深度、K、T、时间及裁剪范围的 *_snapshot.npz。

```bash
python -m scripts.detect_planes --snapshot artifacts/plane_demo/<ID>_snapshot.npz --output artifacts/plane_recompute
```

重算不初始化 MuJoCo，可用于独立调参。最终原生 RANSAC 在不同线程数/软件版本下可能有差异，不承诺逐位一致。

## 验证与局限

测试覆盖水平/竖直/30°/45°/60° 平面、相机侧法向、噪声和离群点、多平面/唯一 mask、无效深度、空点云、下采样后的原始像素对齐，以及已知 MuJoCo 表面的光轴深度、法向和点到真实平面的距离。
端到端测试使用实际 runtime、renderer、算法和 DeepSeekBackend 的模拟 HTTP，检查 user 消息中真实 PNG、JSON/图像观测 ID、保存开关、只读控制行为、仿真继续及迟到取消；没有付费在线 API 或真实模型语义选择实验。

**没有实现 Connected Components、DBSCAN、语义分割、SLAM 或多视角融合。**共面不连续区域仍可能合并；局部范围/深度筛选只能降低风险。凸包会跨越孔洞、凹陷和遮挡，面积仅是观测支持的近似，**凸包不等于真实可贴附区域**。曲面或细杆也可能在距离阈值内拟合出局部平面，应结合 RMSE、支持范围和图像判断；质量指标不是语义概率。
