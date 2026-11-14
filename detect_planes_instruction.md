# UAV Agent：单帧 `detect_planes()` 几何感知工具开发说明

> 面向 Codex 的实施指令。请先阅读仓库现有代码与 `architecture.md`，在现有架构中增量实现；不要另建一套 Agent、仿真运行时、渲染线程或模型通信协议。

## 1. 目标与范围

为当前 MuJoCo UAV Agent 增加一个可由大模型调用的只读工具 `detect_planes()`。工具基于**一次同步的 RGB + Depth + Camera Pose 观测快照**，自动检测视野中任意倾角的多个平面，输出其几何参数、有限二维支持区域及与 RGB 严格对齐的半透明标注图。将精简 JSON 工具结果和 annotated RGB 一起送回 Agent，让模型能把用户描述的目标表面与 `plane_id` 对应起来，供未来贴附工具使用。

**本次采用的完整流水线：**

```text
同一仿真时刻的 RGB + metric Z-depth + 相机内外参
    → 深度有效性检查与相机坐标系反投影
    → 轻量下采样（用于 RANSAC）
    → 迭代多平面 RANSAC
    → 每个平面内点的 PCA/SVD 精细拟合
    → 在原始分辨率深度点云上重新计算平面支持 mask
    → 平面局部二维投影与 Boundary / Extent 估计
    → mask 与 RGB 对齐，生成 annotated RGB（区域颜色 + plane_id）
    → JSON 结构化结果 + annotated RGB 附件 → Agent
```

**明确不做：** Spatial Connected Components / DBSCAN、RGB 语义分割、跨帧配准与多视角融合、SLAM、长期 Plane Memory、NBV/主动感知、贴附点选取、吸附可行性判断、修改飞行控制器。允许为未来扩展预留接口，但不得把这些功能作为本次必需依赖。

本工具要检测**所有朝向**的平面，包括竖直、倾斜、水平、天花板。不得以 `normal.z ≈ 0` 等条件事先过滤。`inclination_deg` 仅是输出属性。

## 2. 与现有工程的集成约束

先检查以下现有模块及其实际实现，再确定最小改动位置（路径以仓库现状为准）：

- `runtime/environment.py`：Environment Runtime 是唯一拥有实时 MuJoCo `MjData`、renderer 生命周期及工具推进权的 owner 线程。
- `observations/base.py` 及 ObservationProvider 实现：读取统一时刻的 RGB、可选 depth、相机/机体状态。
- `agent/runner.py`、`agent/protocol.py`、`agent/messages.py`：Agent 决策循环、工具结果、观测消息和图像附件。
- 现有本地工具注册表与 `iter_execute` 执行路径；`agent/deepseek.py` 或 `agent/backends.py` 等现有后端。
- `ui/main_window.py`：日志与视觉调试预览（可先不增加复杂 UI）。

**必须遵守：**

1. Agent Runner 不直接调用 `env.step()`，不访问实时 `MjData`，不在模型线程中触碰 MuJoCo renderer。获取快照必须通过已有 Environment Runtime 的命令队列/Future 或已有 ObservationProvider 通道。
2. 获取 RGB、Depth、Camera Pose 的快照时，三者要对应同一仿真状态；为数据加 `observation_id` / `sim_time_seconds`，避免旧 mask 叠加在新 RGB 上。
3. 几何计算可以在快照产生后对**不可变的 NumPy 副本**进行，不必长期占据仿真 owner 线程；如果计算较慢，遵循现有 executor/worker 模式异步处理，仿真继续运行。不可为了实现工具再创建第二个可读写 MuJoCo 的线程。
4. 工具是**只读感知工具**：不可修改飞行状态机、飞行目标、控制权或仿真世界。
5. 遵守现有 `turn_id`、取消、授权复查和最大工具调用次数机制；迟到结果不得推进已取消轮次。
6. 沿用现有**自定义 JSON 决策协议**，不要擅自换成 OpenAI 原生 `tool_calls`。工具结果应该进入既有 `tool_result` / observation 消息流。
7. 注意当前已有 `ObservationPacket` 的 RGB 附件、可选深度（默认不发送）、模型输入尺寸与界面预览尺寸分离等机制。**不要为了本工具强制发送整张 depth 图给模型**；应本地分析深度，再只发送结构化几何结果和 annotated RGB。
8. 如果现有消息系统只能在每次观测附带单张 RGB，优先扩展已有附件/观测封装，使 `annotated_rgb` 替代或附加于该次工具反馈所绑定的 RGB；不得把图片字节写进 prompt 的 JSON 文本。保持工具结果与图片同一 `observation_id`。

## 3. 建议的内部数据结构

尽量使用 dataclass / TypedDict / Pydantic（与仓库风格一致），拆成纯算法层与 Agent 接口层：

```python
@dataclass
class CameraSnapshot:
    observation_id: str
    sim_time_seconds: float
    rgb: np.ndarray             # H × W × 3, RGB uint8
    depth_m: np.ndarray         # H × W, metric optical-axis Z-depth, float
    K: np.ndarray               # 3 × 3, matching depth resolution
    T_world_optical: np.ndarray # 4 × 4, camera optical frame -> world

@dataclass
class DetectedPlane:
    plane_id: str               # P1, P2, ...; valid within this observation
    normal_world: np.ndarray    # unit normal, oriented toward camera
    offset_world_m: float       # n·x + d = 0
    centroid_world: np.ndarray  # observed supporting points centroid
    distance_to_camera_m: float # absolute perpendicular distance
    inclination_deg: float      # 0 horizontal, 90 vertical
    inlier_count: int
    inlier_ratio: float
    fit_rmse_m: float
    observed_area_m2: float     # approximate; not true full surface area
    extent_uv_m: tuple[float, float]
    bbox_xyxy: tuple[int, int, int, int]
    mask: np.ndarray            # H × W binary mask (internal only)
    boundary_uv: np.ndarray     # optional 2D plane-local boundary polygon
```

字段可以根据现有编码规范调整，但 JSON 中的坐标系、单位、法向朝向约定必须固定且可验证。

## 4. 相机与坐标系：最重要的前置验证

### 4.1 Depth 含义

确认 `mujoco.Renderer.enable_depth_rendering()` 或现有采集接口输出的是**米制光轴 Z-depth**，不是直接来自 OpenGL 的非线性 depth buffer，也不是欧氏射线距离。用前方已知距离的 box 进行数值校验，避免静默使用错误深度定义。

### 4.2 反投影

使用与当前 depth 分辨率对应的真实相机内参：

\[
Z=D(u,v),\quad X=\frac{u-c_x}{f_x}Z,\quad Y=\frac{v-c_y}{f_y}Z.
\]

这里定义**光学坐标系**：`+X` 向右、`+Y` 向下、`+Z` 向前。MuJoCo camera 的自身局部轴通常不是这一约定（其视线沿 `-Z`），须显式写出 `T_world_optical` 对应的轴变换，绝不可直接把光学系的反投影结果与 MuJoCo 的 camera rotation 混用。

\[
p_W=R_{W\leftarrow C_{opt}}p_{C_{opt}}+t_{W\leftarrow C_{opt}}.
\]

如果当前工程使用的是 NWU 世界系，平面输出也统一为 **NWU world**；法向和位置都标明 `coordinate_frame: "world_nwu"`。不要将相机坐标的法向误写成世界系。

### 4.3 同步和分辨率

- RGB 与 depth 必须对应同一 camera 或已标定配准，且采集时刻一致。
- 处理完整分辨率 depth 时保存原始 `(u,v)` 索引；若对 depth 下采样，务必同步缩放内参。
- 标注图应在对应原始 RGB 分辨率上绘制，再按现有后端输入需要缩放。不要将低分辨率 mask 无说明地套到另一尺寸上。
- 过滤 NaN、Inf、非正深度、远裁剪面背景值（按已有 renderer 约定）。

## 5. 核心几何算法

### 5.1 迭代多平面 RANSAC

对有效点云体素降采样；对剩余点进行 RANSAC，获得平面初值 \((n,d)\) 与 inliers：

\[
\|n\|=1,\quad n^\top p+d=0,\qquad e(p)=|n^\top p+d|.
\]

- `distance_threshold` 可配置，建议初始 `0.01–0.02 m`（MuJoCo 理想深度；最终需要按深度质量调参）。
- 建议 `ransac_n=3`、`num_iterations=300–1000`；不是性能承诺。
- 将当前平面的 inliers 从**工作点云**中移除，继续寻找下一个平面；设置 `max_planes`、`min_inliers`、`min_support_fraction` 等退出条件。
- 小平面可能因为点数不足被忽略，应保留可配置参数，不将“最大的平面”直接定义为目标平面。
- 推荐优先使用 Open3D 的 `segment_plane`；如果仓库已依赖其他点云库，尽量复用。

### 5.2 PCA/SVD 精细拟合

用当前平面 inliers 重新估计：

\[
\bar p=\frac{1}{M}\sum_i p_i,\quad
C=\frac1M\sum_i(p_i-\bar p)(p_i-\bar p)^\top.
\]

对 \(C\) 求特征分解，最小特征值对应的单位特征向量为法向 \(n\)，令 \(d=-n^\top\bar p\)。或对中心化点矩阵做 SVD，取最小右奇异向量。

**法向统一定向：** 给定当前相机世界位置 \(c_{cam}\) 和平面点 \(\bar p\)，若 \(n^\top(c_{cam}-\bar p)<0\)，则令 \(n\leftarrow-n\)，同时 \(d\leftarrow-d\)。确保平面法向始终朝向相机所在一侧；在异常的穿越/相机恰好位于平面上情形给出处理规则。

输出拟合质量：

\[
\operatorname{RMSE}=\sqrt{\frac1M\sum_i(n^\top p_i+d)^2}.
\]

同时保留 inlier 数、inlier 比率；它们是**几何支持质量**，不是校准后的概率。不要生成类似“95% 是墙”的未经验证置信度。

### 5.3 原始分辨率重建 Plane Mask

RANSAC 使用下采样点，所以不能直接把下采样后的 `inliers` 当作原始 RGB 像素索引。

用精细拟合后的 \((n,d)\) 在**原始分辨率且有效的 depth 点云**上重新计算：

\[
M_i(u,v)=\mathbf 1\{|n_i^\top P_W(u,v)+d_i|<\epsilon_{mask}\}.
\]

须限制到与 RANSAC 候选支持点相符的可见局部区域，避免将远处不相关的共面像素全部吞并。由于本版本明确不做 Connected Components，可使用**候选点的局部包围区域/平面坐标范围 + 深度一致性**作为轻量筛选，并将边界断裂、共面物体误并风险写入已知限制。

若不同平面候选的 mask 有交叠，统一根据每个像素到各候选平面的归一化残差分配最多一个 `plane_id`，或将歧义像素设为未标注。不可在最终可视化中同一像素同时属于两个平面。

### 5.4 Boundary & Extent（不含 Connected Components）

将每个平面的支持点投影到该平面的局部二维坐标：

1. 取单位法向 \(n\) 与中心 \(c\)。
2. 构造两个正交切向轴 \(e_1,e_2\)，使 \(e_1^\top n=e_2^\top n=e_1^\top e_2=0\)。
3. 对每个点 \(p\) 得到 \((s,t)=(e_1^\top(p-c),e_2^\top(p-c))\)。
4. **V0 推荐二维 Convex Hull** 得到近似边界 polygon；同时求二维 AABB / extent 和粗略面积。
5. 将 `polygon` 的局部二维顶点恢复到 world 进行调试；图像中的边界尽量优先来自支持 mask 的轮廓而不是“把凸包当作完全可接触区域”。

> **限制说明：** Convex Hull 可能填补遮挡、凹陷和空洞；这是本次刻意接受的简化。`observed_area_m2` 只能作为大致支持范围，不得宣称凸包内部均可吸附，更不能凭此生成最终接触点。无 Connected Components 时，多个共面的不连续区域仍可能被合并。

### 5.5 倾角与距离

水平面倾角定义为：

\[
\alpha=\arccos(|n_W^\top[0,0,1]^\top|)\cdot\frac{180}{\pi}.
\]

0° 为水平面（地板或天花板），90° 为竖直面。对任意倾斜度均输出，不做朝向预筛选。

若返回 `distance_to_camera_m`，明确是**相机光心到无限平面的垂直距离** \(|n^\top c_{cam}+d|\)，不是到有限平面多边形的最近距离，也不是前向射线深度。可选额外返回 `centroid_range_m`，避免误解。

## 6. Annotated RGB 视觉输出

使用本次快照的原始 RGB 作为背景，按每个平面的原始分辨率 mask 绘制：

- 不同、稳定、易区分的颜色（同一 `plane_id` 在当前观测各处一致）。
- 半透明填充（建议 alpha 0.20–0.35，保留纹理细节）。
- 清晰边缘轮廓，区域内部放 `P1/P2/...` 标签；标签位置选择 mask 内部可靠位置，避免完全遮住用户描述的关键对象。
- 非平面区域不涂色，允许存储不带覆盖的原始 RGB 供调试。
- 如果 RGB 与 depth 分辨率不同，先做确定的坐标对齐，不能直接叠加。
- `plane_id` 只保证**单次观测内**唯一；不得假设下一帧相同编号表示同一物理表面。

工具结果中只提供图像的内部引用或按当前 Agent 后端支持的真实图像附件进行发送，不将 base64 串混在 `tool_result` 的 JSON 字符串中。若当前协议/后端不支持在工具反馈里附图，则对现有消息装配做最小兼容扩展，并确保 DeepSeek 实际能看到标注图，而非仅在 UI 显示。

可以额外在 UI 预览/日志里显示 annotated RGB；此预览不得访问实时 renderer。

## 7. Agent 工具接口和返回示例

建议无参调用（使用最新同步快照）：

```json
{"tool":"detect_planes"}
```

工具返回 JSON **示例**（数值仅作结构示意）：

```json
{
  "tool": "detect_planes",
  "status": "success",
  "observation_id": "obs_000125",
  "sim_time_seconds": 12.50,
  "coordinate_frame": "world_nwu",
  "normal_convention": "unit_normal_points_toward_camera",
  "planes": [
    {
      "plane_id": "P1",
      "normal_world": [-0.7071, 0.0, 0.7071],
      "offset_world_m": 0.35,
      "centroid_world_m": [2.2, 0.5, 1.4],
      "inclination_deg": 45.0,
      "distance_to_camera_m": 1.6,
      "inlier_count": 3275,
      "inlier_ratio": 0.41,
      "fit_rmse_m": 0.007,
      "observed_area_m2": 2.15,
      "extent_uv_m": [1.5, 1.8],
      "bbox_xyxy": [25, 20, 150, 205],
      "mask_id": "obs_000125_P1"
    },
    {
      "plane_id": "P2",
      "normal_world": [-1.0, 0.0, 0.0],
      "offset_world_m": 3.2,
      "centroid_world_m": [3.2, -0.8, 1.6],
      "inclination_deg": 90.0,
      "distance_to_camera_m": 2.7,
      "inlier_count": 1640,
      "inlier_ratio": 0.21,
      "fit_rmse_m": 0.009,
      "observed_area_m2": 1.05,
      "extent_uv_m": [1.0, 1.3],
      "bbox_xyxy": [155, 35, 223, 200],
      "mask_id": "obs_000125_P2"
    }
  ],
  "visualization": {
    "kind": "annotated_rgb",
    "image_id": "obs_000125_plane_overlay",
    "attached_to_agent_message": true
  }
}
```

- 示例中几何数值不保证相互构成真实场景，请以算法结果为准。
- JSON 应精简，不包含原始三维点集、全部 mask 像素或编码后的图片；mask/boundary 保留在工具侧缓存供未来复用。
- 没检测到平面时返回 `status: "success"`、`planes: []` 和必要的原因/质量信息，不要伪造平面。
- 图像/深度缺失、相机参数不一致、数值无效或异常时返回可解释的 error；不要静默返回假结果。
- 更新 system prompt / 工具说明：模型应根据 `plane_id` 与附带标注 RGB 完成语言目标到平面的选择；平面几何给的是测量结果而不是已验证的吸附可行性；模型不应凭 JSON 中的大小或倾角自行认定哪个平面是用户指定的目标。

## 8. 性能要求与参数配置

- 纯算法层与 Agent 工具层分离，便于用离线 `rgb/depth/K/pose` 快照测试。
- 初始建议：`max_planes=5`、`voxel_size_m=0.02–0.04`、`ransac_distance_threshold_m=0.01–0.02`、`ransac_iterations=500`；**这些是初始调参范围，不是硬编码标准**。
- 参数暴露到统一配置（或工具默认配置），不要散落在多份代码里。
- 记录分阶段耗时：反投影、RANSAC、PCA、mask 重建、boundary、RGB 标注、总耗时。
- 优先保证正确性；不得阻塞 MuJoCo 的实时 step 循环来等待耗时拟合。
- 没有安装 Open3D 时给出清晰依赖错误，更新 `requirements`/安装说明；不要无理由引入多个重型视觉模型。

## 9. 必需的验证与验收

### 9.1 算法单元测试

至少建立合成平面点集测试：

1. 水平、竖直和 30°/45°/60° 倾斜平面，验证 `inclination_deg` 与法向误差；法向正负一致性按“朝向相机”规范测试。
2. 含随机离群点和少量深度噪声的平面，验证 RANSAC + PCA 的鲁棒性。
3. 同一视野内有 2–3 个不同朝向平面，验证多平面提取和 ID/mask 一一对应。
4. 有无效深度、远裁剪背景、mask 重叠或空点云时不崩溃。
5. 经过体素降采样后，仍能得到与原始 RGB 对齐的正确 mask。
6. 边界、bbox、extent 和倾角的单位与坐标系一致。

### 9.2 MuJoCo 仿真集成测试

在简单 MuJoCo 场景里放置姿态已知的 box/plane surface，用 geom 的世界位姿作为**仅用于离线验证的 ground truth**，比较：

\[
E_n=\arccos(|\hat n^\top n_{GT}|),\qquad
E_d=\left|\hat n^\top c_{GT}+\hat d\right|.
\]

注意 `E_d` 检查真实平面点到估计平面的距离，和“两个以不同符号约定存储的 offset 直接相减”不同。再检查 annotated RGB 区域是否与实际可见面重合。可根据实验条件设定可接受误差，不必提前承诺固定精度。

### 9.3 Agent 端到端测试

- DeepSeek 在单次任务中能够发出 `{"tool":"detect_planes"}`。
- Environment Runtime 生成同步快照，工具执行并返回 JSON。
- 模型真的收到并能查看 annotated RGB，而不只是收到图像引用字符串。
- 同一个 JSON 里的 `P1/P2/...` 与图像标签一致；模型能够依据“左侧/右侧/靠近某物体的平面”等指令选择正确 plane ID。
- 取消、撤权和模型轮次变更时不会执行过期工具结果。
- 无图像或检测失败时系统给出结构化反馈，不阻断后续安全运行。

## 10. 交付要求

请 Codex 在仓库中：

1. 实现 `detect_planes()` 的核心算法模块和 Agent 工具注册，保持原有代码风格。
2. 提供用于人工检查的 RGB 标注图与 JSON 输出；最好附带一份离线测试入口或可复现示例。
3. 增加必要的单元测试及 MuJoCo 集成测试。
4. 更新工具协议说明、依赖、配置和 README/开发文档。
5. 在最终总结中列出修改文件、运行方法、测试结果、现有局限；**特别说明未实现 Connected Components，凸包不等于真实可贴附区域**。

**验收核心标准：** 给定一个 MuJoCo 当前时刻 RGB-D 快照，工具能自动检测同视野内任意倾角的多个几何平面，以世界坐标返回可解释的法向/位置/倾角/有限边界/质量信息；生成与 RGB 严格对齐的半透明 `P1/P2/...` 标注图，并通过现有 Agent 消息链路把 JSON 和实际图像同时提供给 DeepSeek，用于根据用户语言选定目标平面。本阶段不做贴附控制或多视角感知。
