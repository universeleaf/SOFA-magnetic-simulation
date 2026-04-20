# SOFA 导丝导航仿真项目技术总览

本文档面向 `NotebookLM + Canvas` / PPT 制作，按“当前代码实际实现”梳理整个项目的算法、数据流、物理建模、碰撞处理、磁导航逻辑和已知局限。文中尽量区分三件事：

1. 现在代码真实在做什么。
2. 这些实现来自哪些文件、哪些接口。
3. 哪些地方是工程近似，而不是严格的连续体力学真值模型。

## 1. 项目目标与当前架构

这个项目的目标是：在 SOFA 中构建一个“血管网格 + 中心线路径 + 可磁导航导丝”的仿真场景，用于展示导丝在血管腔内的推进、碰撞约束、磁头转向和路径跟踪。

当前主路径不是直接使用原生 `elasticRod.cpp/.h` 作为运行时后端，而是采用：

- Python 场景层：负责搭建 SOFA scene graph、加载血管和中心线、构建 beam 导丝、接线可视化和安全约束。
- SOFA Beam/BeamAdapter 路径：负责导丝主体的梁力学、质量、隐式积分与接触约束。
- 原生 C++ 插件 `ExternalMagneticForceField`：负责磁场方向计算、前视点目标、磁力矩和等效磁力分配。
- Python Controller：负责几何安全兜底、虚拟鞘管、调试可视化、相机跟随，以及一部分推进相关的“命令量”管理。

一句话概括现在的控制分工：

- 磁导航主算法：在 C++。
- 碰撞主链：在 SOFA 接触管线。
- 防穿壁二级兜底：在 Python Controller。
- 推进显示语义：当前更接近运动学送丝，而不是纯粹的恒力推进。

## 2. 代码目录与模块职责

当前你做 PPT 时最值得展示的模块关系如下。

### 2.1 Python 主模块

- `guidewire_scene/main.py`
  - 场景入口。
  - 导出 `createScene`。
- `guidewire_scene/scene.py`
  - 场景总装。
  - 根节点、求解器、血管、中心线、导丝、碰撞节点、视觉节点、控制器都在这里挂接。
- `guidewire_scene/config.py`
  - 所有核心参数。
  - 包括 `option.txt` 解析、导丝物性、接触参数、路由选择、磁导航调参、相机参数等。
- `guidewire_scene/geometry.py`
  - 血管 OBJ 读取。
  - 路径/中心线加载。
  - 腔体半径估计。
  - 初始导丝姿态生成。
- `guidewire_scene/math_utils.py`
  - 路径弧长累计、插值、切向、四元数工具、Data 读写工具。
- `guidewire_scene/sofa_builders.py`
  - 接触管理器。
  - 全导丝碰撞节点。
  - 相机创建。
- `guidewire_scene/controller.py`
  - Python 端控制器。
  - 现在主要做安全几何约束和调试可视化，不再是主磁场求解器。

### 2.2 C++ 原生插件模块

- `guidewire_scene/native/ElasticRodGuidewire/include/ElasticRodGuidewire/ExternalMagneticForceField.h`
- `guidewire_scene/native/ElasticRodGuidewire/src/ExternalMagneticForceField.cpp`

这是当前场景里最关键的磁导航模块。它是一个真正注册到 SOFA ObjectFactory 的 `ForceField<Rigid3d>` 组件，不是 Python 手搓的磁力更新。

### 2.3 学长代码与参考来源

当前项目里“学长代码”的关系要分两类说：

- `elasticRod.cpp/.h` 与 `ElasticRodGuidewireParameters.h`
  - 主要用于提供参数语义和分段刚度语义。
  - 例如“最后 5 条边是磁性边”“head/body 两段使用不同模量”。
- `guidewire_scene/references/externalMagneticForce.cpp/.h`
  - 这是原始磁力算法参考实现。
  - 但它不是当前运行时直接调用的类。
  - 当前做法是“保留其算法语义，再重包成 SOFA 原生插件 `ExternalMagneticForceField`”。

也就是说，当前运行时不是直接实例化参考版 `externalMagneticForce`，而是把它的核心思路迁移进新的 SOFA `ForceField` 组件。

## 3. 场景 Scene Graph 结构

场景的主装配在 `guidewire_scene/scene.py`。

### 3.1 根节点

根节点挂了以下关键组件：

- `FreeMotionAnimationLoop`
- `GenericConstraintSolver`
- `CollisionPipeline`
- `BruteForceBroadPhase`
- `BVHNarrowPhase`
- `LocalMinDistance`
- `DefaultContactManager` 或 `CollisionResponse`

这意味着当前接触求解是标准 SOFA 的“自由运动 + 接触约束修正”框架。

### 3.2 Vessel 节点

血管节点包含：

- `MeshOBJLoader`
- `MeshTopology`
- `MechanicalObject<Vec3d>`
- `TriangleCollisionModel(moving=0, simulated=0, bothSide=1)`
- 半透明可视化 `OglModel`

这里血管被当成静态三角面壳体。

### 3.3 Centerline 节点

中心线节点包含：

- `MechanicalObject<Vec3d>`
- `EdgeSetTopologyContainer`
- 蓝色 `OglModel`

它不参与真实碰撞，只是导航目标和几何参考。

### 3.4 Guidewire 节点

导丝主物理节点包含：

- `EulerImplicitSolver`
- `BTDLinearSolver`
- `MechanicalObject<Rigid3d>`
- `EdgeSetTopologyContainer`
- `BeamInterpolation`
- `AdaptiveBeamForceFieldAndMass`
- `LinearSolverConstraintCorrection`
- `ConstantForceField(name="proximalPushForce")`
- `ExternalMagneticForceField`（如果插件加载成功）

这说明导丝不是点质量链，而是 `Rigid3d` 梁离散模型。每个节点既有平移，也有姿态四元数。

### 3.5 可视化辅助节点

- 灰色主体 `BodyVisual`
- 红色磁头 `MagneticHead`
- 黄色目标点 `TargetMarker`
- 橙色箭头 `MagneticForceArrow`

其中黄色点和橙色箭头主要是调试展示，不等于“真实接触点”或“真实物理合力方向”，后文会专门解释。

## 4. 血管模型与中心线路径

## 4.1 血管几何

血管来自：

- `guidewire_scene/assets/vessel.obj`

几何读取逻辑在 `geometry.py::_load_obj_vertices_faces()`：

- 读取 OBJ 顶点 `v`
- 读取三角面 `f`

运行时血管既作为视觉模型，也作为静态三角碰撞表面。

## 4.2 路径选择机制

当前中心线路径并不是实时从网格中心线提取，而是优先读取预提取路线文件：

- 路由索引在 `guidewire_scene/config.py`
- 路由文件位于 `guidewire_scene/assets/centerline/extracted_paths/routes/`

当前默认路线：

- `SELECTED_ROUTE_NAME = 'right_outer_main'`
- 对应文件：`route_rightmost_lower_inlet_to_rightmost_upper_outlet.npy`

### 4.3 路径合法性校验

在 `config.py` 中有一组路线合法性规则：

- 入口必须位于血管下方区域。
- 出口必须位于血管上方区域。
- 整条路线的纵向跨度必须足够大。

这样做是为了避免选到“入口出口都在下方”的错误路线。

### 4.4 中心线加载与重采样

`geometry.py::_load_centerline()` 的流程是：

1. 优先读取 `CENTERLINE_FILE`。
2. 如果路径点顺序是“上到下”，就自动反转成“下到上”。
3. 删除重复点。
4. 按约 `2 mm` 间距重采样。

重采样的意义是：

- 最近段查找更稳定。
- 前视点查找更平滑。
- 腔体半径估计更均匀。

### 4.5 腔体半径估计

当前血管并没有严格的体素化腔体模型，而是通过 `geometry.py::_lumen_profile()` 近似估计每个中心线点的“局部可用半径”：

- 对于中心线上的每个点，计算到所有血管网格顶点的距离。
- 取最近的 `k` 个顶点。
- 用一个分位数估计局部半径。

这是一种工程近似，不是精确的最近三角面法向管腔半径。

## 5. 导丝物理模型

## 5.1 导丝离散方式

导丝使用的是 SOFA beam 路径，不是完整原生 `elasticRod` 求解器。

每个导丝节点是一个 `Rigid3d`：

- 平移：`x, y, z`
- 姿态：四元数 `qx, qy, qz, qw`

相邻节点之间由 beam 插值和 beam 力场连接。

## 5.2 Beam 插值

`BeamInterpolation` 负责定义梁的截面和材料参数：

- 截面形状：圆截面
- 半径：`WIRE_RADIUS_MM`
- 杨氏模量：按边分段
- 泊松比：`WIRE_POISSON`

## 5.3 分段刚度语义

`config.py::segmented_young(edge_count)` 保留了 `elasticRod.cpp` 的核心分段语义：

- 前 `ne - magnetic_edge_count` 条边使用主体刚度。
- 最后 `magnetic_edge_count` 条边使用磁头刚度。

当前默认：

- 磁性边数 `MAGNETIC_HEAD_EDGES = 5`
- 也就是最后 5 条边属于磁性头段。

## 5.4 当前导丝物性参数

当前代码中，物性参数的来源是“`option.txt` + 插件头文件默认值 + 用户后续覆盖”的组合，不是只认一个来源。

当前实际运行值以 `guidewire_scene/config.py` 为准：

- 时间步长 `DT = 0.003 s`
- 导丝总长 `WIRE_TOTAL_LENGTH_MM = 250 mm`
- 节点数 `WIRE_NODE_COUNT = 201`
- 半径 `WIRE_RADIUS_MM = 0.32 mm`
- 质量密度 `WIRE_MASS_DENSITY = 7800 kg/m^3`
- 泊松比 `WIRE_POISSON = 0.33`
- 主体杨氏模量 `WIRE_BODY_YOUNG_MODULUS_PA = 2.1e11 Pa`
- 磁头杨氏模量 `WIRE_HEAD_YOUNG_MODULUS_PA = 8.0e10 Pa`
- 主体剪切模量 `WIRE_BODY_SHEAR_MODULUS_PA = 7.9e10 Pa`
- 磁头剪切模量 `WIRE_HEAD_SHEAR_MODULUS_PA = 3.0e10 Pa`

这里有一个汇报时必须说清的点：

- `option.txt` 原始值中，杨氏模量更低。
- 但当前场景为了更接近“高刚度金属导丝”，对力学参数采用了插件头文件语义对应的高刚度覆盖值。

也就是说，`option.txt` 是运行和接触参数的重要来源，但机械参数不是 100% 原样照抄 `option.txt`。

## 5.5 质量与变形计算

真正让导丝发生弯曲、扭转和惯性响应的，是：

- `AdaptiveBeamForceFieldAndMass`
- `EulerImplicitSolver`
- `BTDLinearSolver`

物理含义可以简单讲成：

- 梁力场负责内部弹性响应和质量分布。
- 隐式积分负责时间推进。
- 线性求解器负责每步求解增量。
- 接触修正和磁力场共同影响当前时刻的广义力。

## 6. 导丝初始状态生成

初始姿态生成在 `geometry.py::_initial_wire_state()`。

逻辑是：

1. 先定义一个“初始磁头弧长位置” `INITIAL_TIP_ARC_MM`。
2. 再额外沿入口方向前送 `INITIAL_ENTRY_ADVANCE_MM = 2.5 mm`。
3. 对每个节点：
   - 如果该节点已经进入中心线正弧长范围，就放到中心线对应位置。
   - 如果还在入口外，就沿入口方向延长成一条直线。
4. 每个节点的姿态通过 `_quat_from_z_to()` 让本地 z 轴对齐到导丝切向。
5. 检查“已经进入血管内的节点”到血管表面的最小净间隙是否足够。
6. 如果净间隙不够，就把初始磁头位置往回退 `1 mm` 再试。

这样做的目标是：

- 避开入口边缘最尖锐的三角毛刺。
- 让仿真一开始就减少入口处的首帧冲突。

## 7. 碰撞检测与非穿透机制

这是汇报里最重要的一节之一。

当前“不穿血管壁”不是靠单一算法保证，而是三层机制共同作用。

### 7.1 第一层：SOFA 标准碰撞管线

根节点上的碰撞管线是：

- `CollisionPipeline`
- `BruteForceBroadPhase`
- `BVHNarrowPhase`
- `LocalMinDistance`
- `FrictionContactConstraint`

其中：

- `LocalMinDistance` 用于近距离接触检测。
- `FrictionContactConstraint` 用于生成接触约束。
- `GenericConstraintSolver` 用于求解这些约束。

当前关键参数：

- `alarmDistance = 0.2 mm`
- `contactDistance = 0.1 mm`
- `mu = 0.01`
- `maxIterations = 101`
- `tolerance = 1e-3`

### 7.2 第二层：全导丝统一 CollisionNode

`sofa_builders.py::_add_full_collision()` 明确创建了一个只负责整根导丝碰撞的节点：

- `MechanicalObject<Vec3d>`
- `RigidMapping(template="Rigid3d,Vec3d")`
- `LineCollisionModel`
- `PointCollisionModel`

关键点：

- 这是整根导丝统一碰撞节点，不是只有红色磁头碰撞。
- 使用 `RigidMapping` 把 `Rigid3d` 导丝节点映射到碰撞几何点。
- `LineCollisionModel` 负责线段级接触。
- `PointCollisionModel` 负责点级接触。
- `selfCollision=1`，所以导丝自身也能参与一定的自碰撞约束。

这正是为了解决早期那种“红头受约束、灰体穿墙”的问题。

### 7.3 第三层：Python 二级安全投影

即便有 SOFA 接触管线，当前代码仍在 `controller.py::_constrain_wire()` 里加了一层几何兜底。

它不是主碰撞求解器，而是二级安全修正。

其逻辑分三部分：

#### 7.3.1 虚拟鞘管

对于还没进入血管的体外段节点：

- 横向位置锁回初始插入直线。
- 只允许沿插入方向运动。
- 横向速度和角速度会被抑制。

目的：

- 抑制入口外段欧拉屈曲。
- 减少入口前方的团丝和倒退。

#### 7.3.2 腔内安全投影

对于已经进入血管的节点：

- 根据中心线弧长位置估计该处局部腔体半径。
- 如果某个节点径向超出允许腔体，就将其投影回允许边界。

这个逻辑本质上是“近似的腔内圆管投影”，而不是对真实三角网格做严格点面最近投影。

#### 7.3.3 段长夹持

为了避免局部段长因数值抖动突然拉得过长或压得过短，控制器还会对相邻节点段长做夹持：

- 最小段长比：`0.97 * restSpacing`
- 最大段长比：`1.03 * restSpacing`

这一步是工程稳定化手段，本质上是在用几何方式抑制非物理锯齿弯折。

### 7.4 当前非穿透机制的结论

所以如果老师问“你们怎么保证导丝不穿出血管外面”，可以准确回答：

- 第一层是真实接触约束：SOFA 的碰撞检测 + 摩擦接触约束求解。
- 第二层是整根导丝统一碰撞映射，确保主体和磁头都在同一碰撞链里。
- 第三层是 Python 几何兜底：体外虚拟鞘管 + 腔内安全投影 + 段长夹持。

同时也必须诚实补一句：

- 这第三层是工程近似，并不等价于“真实血管壁连续体接触模型”。

## 8. 磁场力与磁导航算法

这部分一定要按“当前真实运行版本”来讲。

## 8.1 主磁导航不在 Python，而在 C++ 插件

当前磁导航主算法在：

- `guidewire_scene/native/ElasticRodGuidewire/src/ExternalMagneticForceField.cpp`

Python `controller.py` 已经不再负责主磁力计算。它只读取 C++ 导出的调试数据。

### 8.2 C++ 组件的数据接口

`ExternalMagneticForceField` 的主要 Data 包括：

- `tubeNodes`
- `brVector`
- `baVectorRef`
- `muZero`
- `rodRadius`
- `magneticEdgeCount`
- `lookAheadDistance`
- `fieldSmoothingAlpha`
- `maxFieldTurnAngleDeg`
- `fieldRampTime`
- `lateralForceScale`
- `debugTargetPoint`
- `debugBaVector`
- `debugForceVector`

其中最重要的是：

- `tubeNodes`：导航中心线。
- `brVector`：参考剩磁方向。
- `baVectorRef`：没有足够中心线信息时的默认场方向。
- `magneticEdgeCount`：受磁段边数。

## 8.3 磁导航目标的求法：前视点 Look-ahead

`computeLookAheadDirection()` 的流程如下：

1. 取磁头尖端位置 `tipPos`。
2. 在整条中心线 `tubeNodes` 上寻找离尖端最近的线段投影点。
3. 得到该投影对应的中心线弧长 `bestS`。
4. 向前推进一个固定前视弧长 `lookAheadDistance`。
5. 插值得到前视目标点 `targetPoint`。
6. 用 `targetPoint - tipPos` 作为期望导航方向 `desiredBa`。
7. 如果前视目标点离尖端太近，就退化成附近中心线切向差分方向。

当前默认前视距离：

- `MAGNETIC_LOOKAHEAD_DISTANCE_MM = 2.5 mm`

这意味着磁头不是盯着最近点，而是盯着“沿中心线向前 2.5 mm 的目标”。

## 8.4 为什么磁场方向不会瞬间对准目标

很多时候视觉上会感觉“目标点在右边，但磁场方向没有立刻完全向右”。这是因为 C++ 力场故意做了三重平滑限制。

### 8.4.1 最大转角限制

`rotateToward(from, to, maxAngle)` 限制了磁场方向每一步最多只能转：

- `MAGNETIC_MAX_TURN_ANGLE_DEG = 3.5°`

这意味着即使目标方向一下子变很多，施加场方向也只会逐步逼近。

### 8.4.2 一阶低通平滑

`blendDirection(a, b, alpha)` 做方向滤波：

- `MAGNETIC_FIELD_SMOOTHING_ALPHA = 0.12`

也就是说当前场方向并不是直接等于目标方向，而是：

- 一部分保留上一步方向
- 一部分向新的目标方向靠拢

### 8.4.3 启动爬升 Ramp

`fieldRampTime = 0.30 s`

仿真开始前 `0.3 s` 内，磁场强度会从 0 逐渐爬升到完整值，避免一开启动就把磁头突然掰弯。

所以你在汇报时如果要解释“为什么磁头回正不是瞬间发生”，可以说：

- 为了数值稳定，当前磁场方向被限制为“最近中心线段前视方向的平滑、限速版本”，而不是理想中的无延迟瞬时对齐。

## 8.5 磁力矩与等效节点力的计算

`computeMagneticForces()` 是实际把磁场方向转成节点广义力的地方。

### 8.5.1 受磁范围

只作用于最后 `magneticEdgeCount = 5` 条边，也就是红色磁头部分。

主体灰色部分本身不直接受磁场力。

### 8.5.2 当前边框架与初始边框架

对于每条磁性边：

- 先重建当前局部坐标系 `current(m1,m2,m3)`
- 再调用初始化时缓存的 `rest(m1,m2,m3)`

这样可以把参考剩磁 `brVector` 从“初始磁化方向”映射到“当前导丝姿态方向”。

### 8.5.3 磁矩与力矩

代码里的核心关系可以概括成：

- 截面积 `A = π r^2`
- 当前磁矩 `m` 由参考剩磁在当前局部基底上重建
- 磁力矩 `tau ≈ (A * L / muZero) * (m × Ba)`

这里的 `Ba` 是当前施加的平滑磁场方向。

### 8.5.4 力矩转成等效节点力

因为 beam 节点上更容易施加的是节点广义力，所以代码把力矩转成一对等效力：

- `pairForce = torque × edgeVec / |edge|^2`

它的物理直觉是：

- 给一段磁性梁施加一对相反节点力，形成等效弯矩。

### 8.5.5 附加横向 steering force

除了纯力矩，代码还增加了一个横向 steering force：

1. 将期望方向 `desiredBa` 投影到当前边方向的法平面上。
2. 得到横向单位方向 `lateralDir`。
3. 根据失配程度 `misalignment` 调节横向力大小。
4. 用 `lateralForceScale` 放大这部分辅助转向作用。

当前默认：

- `MAGNETIC_LATERAL_FORCE_SCALE = 1.25`

这意味着当前磁导航并不只是“纯力矩驱动”，而是“磁力矩 + 横向辅助拉偏”的组合。

### 8.5.6 力和转矩的节点分配

每条磁性边的节点力并不是平均分，而是向更远端略偏重地分配：

- 平移力分到边的两个端点。
- 旋转载荷也分到两个端点。
- 越靠近最远端，权重越高。

目的：

- 让磁头末端比磁头根部更容易体现可见转向。

## 8.6 C++ 力场的切线刚度近似

`ExternalMagneticForceField` 不只是 `addForce()`，还实现了：

- `addDForce()`
- `addKToMatrix()`

这两部分通过有限差分对当前磁力模型做局部线性化，给隐式求解器提供近似切线刚度。

这一步的意义很大：

- 如果只有外力没有导数，隐式求解的收敛性会更差。
- 当前实现虽然不是解析 Jacobian，但用数值差分给出了可用的局部刚度近似。

## 8.7 调试箭头不等于真实合力方向

这一点必须明确说清。

在 `controller.py::_sync_debug_visuals()` 中：

- 优先使用 `target_pull = normalize(target_point - tip_pos)` 作为箭头方向。
- 如果这个方向不可用，才退回到 C++ 导出的 `debugForceVector`。

所以现在屏幕上显示的橙色箭头，更接近：

- “当前想把磁头引向哪里的方向”

而不是严格意义上的：

- “当前磁头真实受到的总合力方向”

这也正是为什么有时你会看到“目标点在右边，但箭头与实际弯曲不完全一致”。

## 9. Python Controller 的实际职责

`GuidewireNavigationController` 现在已经不是主导航器，而是一个“场景安全与调试协调器”。

### 9.1 每一步的执行顺序

`onAnimateBeginStep(dt)` 的核心流程是：

1. `_advance_commanded_push(dt)`
   - 累积命令送丝量。
2. `_update_estimated_push_mm()`
   - 更新名义推进量。
3. `_constrain_wire()`
   - 做虚拟鞘管、腔内投影、段长夹持。
4. 再次 `_update_estimated_push_mm()`
5. 读取当前磁头姿态和速度。
6. `_update_wall_contact_state()`
   - 检查磁头是否贴壁。
7. `_sync_debug_visuals()`
   - 更新黄色目标点和橙色箭头。
8. `_update_camera_follow()`
   - 可选相机跟随。
9. `_update_steering_state()`
   - 计算磁头方向与目标方向夹角。
10. `_update_push_force(dt)`
   - 当前这里实际把近端力场清零。

### 9.2 一个非常关键的现实：当前推进不是纯恒力推进

虽然场景里仍然存在：

- `ConstantForceField(name="proximalPushForce")`

但当前 `controller.py::_update_push_force()` 里实际上写的是：

- `total_force = 0.0`
- 每个近端节点的力都被清零

同时控制器又在每帧推进：

- `commanded_push_mm += speed * dt`

并把这部分命令推进量用于虚拟鞘管位置和安全链。

所以严格来说，当前推进主逻辑已经切回：

- 固定速度的运动学送丝语义

而不是：

- 近端真实恒力推入后由阻尼自然平衡出速度

这一点你给老师汇报时最好直接说清楚，不然很容易被追问“为什么 ConstantForceField 还在但速度是人为固定的”。

### 9.3 碰壁检测

碰壁检测主要看磁头末端若干探针节点的壁净间隙：

- 进入贴壁态阈值：`0.05 mm`
- 退出贴壁态阈值：`0.15 mm`

这是一个带滞回的判定，避免在阈值附近频繁抖动切换。

### 9.4 转向失配检测

控制器还会计算：

- 当前磁头方向 `tipDir`
- 目标方向 `desiredDir = normalize(target_point - tip_pos)`
- 两者夹角 `steering_angle_deg`

然后用两个阈值定义“是否转向失配”：

- 进入失配：`20°`
- 退出失配：`12°`

当前这个状态主要用于诊断和日志，不再直接控制磁场方向。

## 10. 当前与 `option.txt` / 学长参数接口的关系

## 10.1 `option.txt` 真正接管了什么

当前 `config.py` 会解析：

- `rodRadius`
- `RodLength`
- `numVertices`
- `youngM`
- `noMagYoungM`
- `Poisson`
- `density`
- `deltaTime`
- `speed`
- `col_limit`
- `mu`
- `tol`
- `maxIter`
- `gVector`
- `baVector`
- `brVector`
- `muZero`
- `tubeRadius`

这些参数被拆成两类：

- 直接接管运行参数：时间步长、接触距离、求解器容差、摩擦、磁参考向量等。
- 作为参考但允许覆盖：机械刚度、密度、半径、推进速度等。

## 10.2 当前没有完全照搬 `option.txt` 的地方

汇报时不要说成“全部参数都严格来自 option.txt”，因为当前代码里确实有覆盖：

- 半径：最终取 `max(option, plugin default, 0.32)`
- 密度：使用 `7800 kg/m^3`
- 泊松比：使用 `0.33`
- head/body 杨氏模量：使用更高的插件默认值
- 摩擦系数：`mu = min(optionMu, 0.01)`，所以实际会压到 `0.01`
- 推进速度：当前强行设置为 `10 mm/s`，覆盖了 `option.txt` 中的 `1 mm/s`

这在论文式表述里属于“参数对齐后再做工程性二次调参”。

## 10.3 与 `elasticRod.cpp/.h` 的语义对齐

当前与 `elasticRod.cpp/.h` 对齐得最明确的，是以下语义：

- 主体段和磁头段使用不同材料参数。
- 最后 `5` 条边视为磁性边。
- 磁头只在远端受磁场作用，主体不直接受磁力。

但当前运行后端仍然是 beam，不是完整原生 elastic rod 求解器，所以不能说“现在已经完全运行在 elasticRod 本体上”。

## 11. 为什么会出现“回正慢、贴壁后才明显偏转”的观感

这是老师很可能会问的点。

可以从以下 5 个原因解释。

### 11.1 磁场方向被平滑和限速

如前所述：

- 最大每步转角只有 `3.5°`
- 平滑系数只有 `0.12`
- 启动还有 `0.3 s` ramp

所以不会出现“目标点一变，磁头立刻瞬时大幅偏转”的效果。

### 11.2 只有最后 5 条边受磁

磁力只施加在最末端的一小段磁头上，而整根导丝主体不直接受磁。

因此整体导丝的显著偏转需要：

- 先由磁头局部产生弯矩
- 再通过梁结构把曲率逐步传回主体

这天然比“整根都受横向力”慢。

### 11.3 当前箭头更像目标方向，不像真实合力

用户肉眼看到的箭头并不是求解器实际使用的总合力方向，所以会产生“显示上应该往右，但形变没有立即往右”的感受。

### 11.4 推进主逻辑是运动学送丝

由于当前推进不是纯力平衡，而是命令推进量驱动，磁头有时会在“被送进去”的同时再慢慢对齐，而不是像真实力控系统那样先力学平衡后再前进。

### 11.5 Beam 后端仍是工程近似

当前不是严格不可伸长的 Cosserat rod / 原生 elastic rod 全模型，因此会残留一定的：

- 局部压缩
- 数值锯齿弯折
- 团丝倾向

这也是为什么当前代码还要依赖段长夹持和虚拟鞘管来稳定它。

## 12. 当前场景的主要局限

这一节在 PPT 中非常建议保留，因为它能体现你对模型边界是清楚的。

### 12.1 不是全原生 elastic rod

当前主体还是 beam 场景，不是完全使用 `elasticRod.cpp/.h` 的原生 rod 后端。

### 12.2 非穿透带有几何近似兜底

除 SOFA 标准碰撞之外，当前额外用了中心线-半径近似的腔内投影，不是对真实血管内壁做解析接触修正。

### 12.3 推进不是纯力控

虽然保留了 `proximalPushForce` 组件，但当前控制器已经把它清零，推进主逻辑更接近固定速度送丝。

### 12.4 箭头不是严格物理力可视化

橙色箭头更偏“导航目标方向可视化”，不应被当作严格的瞬时总受力方向。

### 12.5 腔体半径来自顶点距离统计

局部 lumen 半径是通过血管顶点邻近统计估计的，不是严格的三角面法向内接圆估计。

## 13. 如果你做 PPT，推荐怎么讲

下面是一套适合 `NotebookLM + Canvas` 自动生成提纲的结构。

### 第 1 页：项目目标

- 在 SOFA 中构建磁导航导丝血管仿真
- 展示导丝推进、转向、碰撞和路径跟踪
- 用于汇报算法结构与工程实现

### 第 2 页：整体架构

- Python 场景层
- Beam 导丝力学层
- SOFA 碰撞/约束层
- C++ 原生磁力场插件层

### 第 3 页：数据与文件结构

- `scene.py`
- `controller.py`
- `config.py`
- `geometry.py`
- `ExternalMagneticForceField.cpp`

### 第 4 页：血管与中心线

- `vessel.obj` 静态血管网格
- 路由文件 `.npy`
- 路径合法性校验
- 2 mm 重采样

### 第 5 页：导丝物理模型

- `Rigid3d` 离散
- `BeamInterpolation`
- `AdaptiveBeamForceFieldAndMass`
- head/body 分段刚度

### 第 6 页：碰撞与非穿透

- 全导丝统一碰撞节点
- 静态三角血管碰撞面
- 接触参数 `alarmDistance/contactDistance/mu`
- Python 二级安全投影

### 第 7 页：磁导航算法

- 最近中心线投影
- 前视点 look-ahead
- 平滑、限速、ramp
- 磁力矩和等效节点力

### 第 8 页：控制器逻辑

- 虚拟鞘管
- 腔内安全投影
- 段长夹持
- 目标点和箭头同步
- 相机跟随

### 第 9 页：关键参数表

建议列：

- 导丝长度、半径、节点数
- 主体/磁头杨氏模量
- `dt`
- `mu`
- `contactDistance`
- `lookAheadDistance`
- `magneticEdgeCount`

### 第 10 页：当前局限与后续方向

- 仍是 beam 近似，不是完整 elastic rod
- 腔内投影是工程兜底
- 推进未完全力控
- 箭头仍需改成真实力方向可视化
- 后续可考虑切换到更严格不可伸长 rod 模型

## 14. 一句结论版

如果你最后只想给老师一句浓缩总结，可以这样讲：

“当前系统采用 SOFA beam 作为导丝连续体骨架，使用全导丝统一碰撞链保证主体与磁头共同受约束，再用原生 C++ `ExternalMagneticForceField` 按中心线前视点实时计算磁场方向和磁力矩，同时辅以 Python 层的虚拟鞘管与腔内几何投影做数值稳定化；因此它是一个可运行、可演示、但仍带有工程近似的磁导航导丝仿真系统。”
