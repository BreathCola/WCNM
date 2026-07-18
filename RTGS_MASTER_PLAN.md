# RT-GS from Clean 3DGS：全阶段开发规范

## 0. 本文档用途

这是本仓库实现 RT-GS 的唯一长期技术标准。

每次开始修改代码前，必须阅读：

```text
AGENTS.md
docs/RTGS_MASTER_PLAN.md
docs/STATUS.md
docs/DECISIONS.md
docs/stages/STAGE_<CURRENT>.md
```

其中：

```text
AGENTS.md                    工作纪律、禁止事项、阶段切换规则
docs/RTGS_MASTER_PLAN.md     本文件：唯一完整技术规范
docs/STATUS.md               当前阶段、已完成内容、当前阻塞、下一项任务
docs/DECISIONS.md            所有论文未公开细节与工程决策
docs/stages/STAGE_*.md       当前阶段的细化任务和验收 checklist
```

这些文件在 **Stage A 的首个提交中创建**，不单独设置 Stage0。

禁止依赖聊天上下文、临时口头约定或未记录的实现假设。
任何影响渲染公式、数据表示、损失、路径定义、训练时序的决定，都必须写入 `docs/DECISIONS.md`。

---

# 1. 最终目标

从一个干净的原始 3D Gaussian Splatting 仓库出发，实现 RT-GS 风格的三分支 Gaussian 表示：

```text
Diffuse / Base Gaussian       D
Reflection Gaussian           R
Transmittance Gaussian        T
```

最终应支持：

```text
输入多视图图像 + 相机位姿 + 透明物体 mask
                ↓
Diffuse surfel rasterization
                ↓
Cd / depth / position / normal / roughness / f0 / ks
                ↓
Reflection ray tracing on R
                ↓
Transmittance first bounce on T
                ↓
Mesh-guided second bounce on D
                ↓
microfacet BRDF / BTDF compositing
                ↓
最终 RT-GS 图像
```

最终颜色：

```text
C = (1 - ks) * Cd + ks * (fr * Cr + ft * Ct)
```

其中：

```text
Cd : diffuse color
Cr : reflection color
Ct : transmittance color
ks : specular blending weight
fr : BRDF 权重
ft : BTDF 权重
```

---

# 2. 全局不可违反原则

## 2.1 必须从 3DGS 演进为 2D Gaussian / surfel

最终 RT-GS 不是原始 3D ellipsoid 3DGS。

Diffuse 分支必须改为 2D Gaussian / surfel 表示，以获得稳定的：

```text
surface position
depth
normal
surface mesh
```

原始 3DGS 的训练入口、数据读取、COLMAP 支持、checkpoint 思路可以保留；但最终 RT-GS 配置必须使用 2D surfel。

---

## 2.2 三套 Gaussian 必须独立

必须存在三个独立模型：

```python
diffuse_model
reflection_model
transmittance_model
```

它们必须具有独立的：

```text
参数
optimizer
scheduler
densification state
pruning state
checkpoint state
导出文件
debug 图
```

禁止用一个 Gaussian 数组加 `type_id` 冒充三套场。

---

## 2.3 训练时不得裁掉外部环境

即使最终只导出玻璃罩和内部物体，训练图像也必须保留外部环境。

原因：

```text
R 分支需要外部环境学习玻璃表面反射；
D 分支需要外部环境支持 second-bounce outside color；
T 分支必须避免把外部背景错误吸收到内部。
```

禁止使用透明物体 mask 屏蔽整张 RGB loss。

mask 只用于：

```text
开启 T 分支
specular constraint
transmittance depth constraint
transparent mesh / two-hit 辅助处理
```

---

## 2.4 透明物体采用薄壳近似

本实现遵循 RT-GS 假设：

```text
透明物体外层近似无限薄；
不实现 Snell 折射；
不实现厚玻璃、液体、棱镜、色散、焦散；
transmittance ray 方向保持为相机入射方向。
```

即：

```python
d_trans = d_cam
```

其中：

```python
d_cam = normalize(surface_position - camera_center)
```

---

## 2.5 每个阶段必须可独立运行、测试和回滚

每完成一个阶段，必须：

```text
1. 更新 STATUS.md；
2. 更新 DECISIONS.md；
3. 更新对应 STAGE_X.md checklist；
4. 提供训练和测试命令；
5. 输出关键 debug 图；
6. 创建可回滚 Git commit；
7. 未通过验收不得进入下一阶段。
```

---

# 3. 数据与坐标标准

## 3.1 输入数据

要求兼容基础 3DGS 数据格式：

```text
scene/
├── images/
├── sparse/
│   └── 0/
│       ├── cameras.bin
│       ├── images.bin
│       └── points3D.bin
├── masks/
│   ├── 00000.png
│   ├── 00001.png
│   └── ...
└── normal_priors/
    ├── 00000.npy
    ├── 00001.npy
    └── ...
```

透明物体 mask：

```text
0：非透明区域
1：透明物体区域
```

建议支持三种 mask：

```text
mask_soft        用于边缘稳定的材质监督
mask_hard        用于启用 transmittance ray
mask_eroded      用于 mesh / depth loss
```

---

## 3.2 方向约定

统一采用：

```python
d_cam = normalize(surface_position - camera_center)
```

含义：

```text
d_cam：从相机指向表面
wo：从表面指向相机
```

因此：

```python
wo = -d_cam
```

反射方向：

```python
d_ref = normalize(
    d_cam - 2.0 * dot(d_cam, normal) * normal
)
```

所有 normal 在 deferred shading 前必须 face-forward：

```python
normal = where(
    dot(normal, wo) < 0,
    -normal,
    normal
)
```

所有 ray origin 必须有 epsilon 偏移：

```python
origin = position + eps * direction
eps = ray_epsilon_scale * scene_radius
```

默认：

```yaml
ray_epsilon_scale: 1e-4
```

---

# 4. 模型数据结构

## 4.1 Diffuse 2D Surfel Gaussian

每个 Diffuse Gaussian 至少包含：

```python
xyz:             [N, 3]
rotation:        [N, 4]
scaling_2d:      [N, 2]
opacity_raw:     [N, 1]

base_color_raw:  [N, 3]
roughness_raw:   [N, 1]
f0_raw:          [N, 3]
ks_raw:          [N, 1]
```

激活函数：

```python
opacity = sigmoid(opacity_raw)

roughness = roughness_min + (
    1.0 - roughness_min
) * sigmoid(roughness_raw)

f0 = sigmoid(f0_raw)
ks = sigmoid(ks_raw)
```

默认：

```yaml
roughness_min: 0.03
f0_channels: 3
```

局部切向量由 rotation 与 2D scaling 导出：

```python
tangent_u
tangent_v
normal = normalize(cross(tangent_u, tangent_v))
```

normal 不得作为与几何完全解耦的自由可学习向量。

---

## 4.2 Reflection Gaussian

Reflection Gaussian 是独立的 2D surfel field。

最低属性：

```python
xyz
rotation
scaling_2d
opacity_raw
color_raw
```

初始化必须支持：

```yaml
reflection_init:
  mode: random_bbox
```

以及可选工程模式：

```yaml
reflection_init:
  mode: uniform_grid_bbox
  grid_resolution: 32
  surfels_per_cell: 5
```

所有初始化策略和实际数量必须写入日志。

---

## 4.3 Transmittance Gaussian

Transmittance Gaussian 也是独立 2D surfel field。

最低属性：

```python
xyz
rotation
scaling_2d
opacity_raw
color_raw
```

必须支持：

```yaml
transmittance_init:
  mode: random_scene_bbox
```

可选工程优化模式：

```yaml
transmittance_init:
  mode: random_transparent_volume
```

该优化模式不是论文明确公开内容，默认关闭；启用时必须在 `DECISIONS.md` 说明。

---

# 5. Renderer 与 Ray Tracer 输出契约

## 5.1 Diffuse Rasterizer 输出

每次 diffuse rasterization 必须输出：

```python
{
    "Cd":           [H, W, 3],
    "alpha":        [H, W, 1],
    "depth":        [H, W, 1],
    "position":     [H, W, 3],
    "normal":       [H, W, 3],
    "roughness":    [H, W, 1],
    "f0":           [H, W, 3],
    "ks":           [H, W, 1],
}
```

所有属性使用相同前到后 alpha compositing：

```text
x = Σ_i x_i * opacity_i * Π_j<i(1 - opacity_j)
```

应用于：

```text
Cd
depth
normal
roughness
f0
ks
```

normal blend 后必须归一化：

```python
normal = normalize(blended_normal)
```

position 优先通过 depth 与相机参数反投影获得：

```python
position = unproject(depth, camera)
```

---

## 5.2 Gaussian Ray Tracer 输出

所有 R、T、second-bounce D ray trace 必须复用同一接口：

```python
color, alpha, expected_depth, hit_mask = raytrace(
    model,
    ray_origins,
    ray_directions,
    return_alpha=True,
    return_depth=True,
    return_hit_mask=True,
)
```

输出：

```python
color:          [M, 3]
alpha:          [M, 1]
expected_depth: [M, 1]
hit_mask:       [M, 1]
```

ray tracer 必须支持对以下变量反向传播：

```text
Gaussian position
Gaussian rotation
Gaussian scaling
Gaussian opacity
Gaussian color
ray origin
ray direction
```

必须支持：

```text
chunked tracing
BVH 或等价加速结构
参数更新后的加速结构同步
CUDA / OptiX 主实现
```

纯 PyTorch brute-force 版本只允许作为小场景测试 fallback。

---

# 6. 材质与 BRDF / BTDF 标准

## 6.1 最终渲染公式

```python
C = (1.0 - ks) * Cd + ks * (
    wr * Cr + wt * Ct
)
```

其中：

```text
wr：反射有效权重
wt：透射有效权重
```

---

## 6.2 GGX / Trowbridge-Reitz NDF

设：

```python
wo = -d_cam
wi = d_ref
h = normalize(wi + wo)

NoV = clamp(dot(normal, wo), eps, 1)
NoL = clamp(dot(normal, wi), eps, 1)
NoH = clamp(dot(normal, h), eps, 1)
VoH = clamp(dot(wo, h), eps, 1)
```

使用：

```python
D = roughness ** 2 / (
    pi * (
        (NoH ** 2) * (roughness ** 2 - 1.0) + 1.0
    ) ** 2
)
```

使用：

```python
F = fresnel_schlick(f0, VoH)
G = smith_ggx(NoV, NoL, roughness)

fr = D * G * F / (4.0 * NoV * NoL + eps)
wr = fr * NoL
```

必须使用完整 microfacet 项，不得使用 split-sum approximation。

---

## 6.3 Roughness 稳定化

roughness 接近零会导致 GGX 数值不稳定。

必须：

```text
1. 将 roughness clamp 到 roughness_min；
2. 支持 PBRT roughness remapping；
3. 防止 D、G、fr 出现 NaN 或 Inf；
4. 记录 roughness 最小值和统计直方图。
```

PBRT remapping 作为可选开关：

```yaml
roughness_remap:
  enabled: true
```

若启用，记录具体公式与 적용位置。

---

## 6.4 透射权重

薄壳近似下：

```python
wt = transparent_mask * (1.0 - F)
```

论文没有完整公开单条 ray 下 BRDF / BTDF、cosine 项、PDF 和积分近似如何折叠。

因此必须提供：

```yaml
bsdf_weight_mode:
  - raw_brdf
  - brdf_times_cosine
  - normalized_energy
```

默认：

```yaml
bsdf_weight_mode: brdf_times_cosine
```

并在 checkpoint metadata 中保存实际使用的模式。

---

# 7. Transmittance 合成标准

第一跳：

```python
Cin, Ain, Din, _ = raytrace(
    transmittance_model,
    first_origin,
    d_trans,
)
```

第二跳：

```python
Cout, Aout, Dout, _ = raytrace(
    diffuse_model,
    second_origin,
    d_trans,
)
```

论文图示中 Inside Color 与 Outside Color 相加，但未公开精确合成式。

默认采用 alpha-over：

```python
Ct = Cin + (1.0 - Ain) * Cout
At = Ain + (1.0 - Ain) * Aout
```

必须支持对照模式：

```yaml
transmittance_compose:
  mode: alpha_over
```

可选：

```yaml
transmittance_compose:
  mode: raw_add
```

若改动默认合成逻辑，必须写入 `DECISIONS.md`。

---

# 8. Loss 标准

## 8.1 RGB Reconstruction

```python
L_rgb = lambda_l1 * L1(render, gt) + \
        lambda_dssim * D_SSIM(render, gt)
```

默认应继承基础 3DGS 的 L1 / D-SSIM 配置。

若使用：

```yaml
lambda_l1: 0.8
lambda_dssim: 0.2
```

必须标记为工程默认值，而非 RT-GS 论文明确给出的比例。

---

## 8.2 Specular Constraint

透明 mask 区域强制 `ks` 接近高 specular 值：

```python
L_spec = mean(
    transparent_mask * relu(K0 - ks)
)
```

默认：

```yaml
K0: 0.9
lambda_spec: 0.2
```

---

## 8.3 Transmittance Depth Constraint

第一跳 T 深度不能越过透明壳背面：

```python
L_depth = mean(
    mask_eroded
    * valid_two_hit
    * relu(Din - t_far)
)
```

默认：

```yaml
lambda_depth: 0.2
```

---

## 8.4 Normal-depth Consistency

```python
Nd = normalize(
    cross(
        finite_diff_x(position),
        finite_diff_y(position)
    )
)

L_norm = mean(
    1.0 - dot(normal, Nd)
)
```

默认：

```yaml
lambda_norm: 0.04
```

---

## 8.5 Monocular Normal Loss

```python
L_mono = mean(
    1.0 - dot(normal, mono_normal)
)
```

默认：

```yaml
lambda_mono: 0.01
```

---

## 8.6 Perceptual Loss

使用 VGG-16 特征：

```python
L_perc = L1(
    vgg(render),
    vgg(ground_truth)
)
```

默认：

```yaml
lambda_perc: 0.01
```

---

## 8.7 总损失

```python
L = (
    L_rgb
    + lambda_spec * L_spec
    + lambda_depth * L_depth
    + lambda_norm * L_norm
    + lambda_mono * L_mono
    + lambda_perc * L_perc
)
```

---

# 9. 开发阶段总览

```text
Stage A：3DGS → 2D Surfel Diffuse Foundation
Stage B：Differentiable Gaussian Ray Tracing + Reflection
Stage C：Transparent Mask + Mesh + Two-Hit Geometry
Stage D：Transmittance Gaussian + Full RT-GS Training
Stage E：稳定化、消融、评测、导出与复现
```

---

# 10. Stage A：2D Surfel Diffuse Foundation

## 10.1 目标

从基础 3DGS 项目出发，完成：

```text
原始 3D Gaussian ellipsoid
        ↓
2D Gaussian / surfel diffuse representation
        ↓
Diffuse rasterization
        ↓
Cd / alpha / depth / position / normal /
roughness / f0 / ks
```

本阶段不实现：

```text
Reflection Gaussian
Transmittance Gaussian
Ray tracing
OptiX
Mesh two-hit
BRDF / BTDF final shading
```

---

## 10.2 Stage A 首个提交必须创建的文件

```text
AGENTS.md
docs/RTGS_MASTER_PLAN.md
docs/STATUS.md
docs/DECISIONS.md
docs/stages/STAGE_A.md
docs/stages/STAGE_B.md
docs/stages/STAGE_C.md
docs/stages/STAGE_D.md
docs/stages/STAGE_E.md
```

其中：

```text
RTGS_MASTER_PLAN.md：复制本规范
STATUS.md：当前阶段、完成项、测试、阻塞、下一步
DECISIONS.md：工程选择与论文未公开内容
STAGE_A.md：本阶段详细 checklist
```

---

## 10.3 Stage A 实现任务

### A-1：重构模型表示

新增：

```text
scene/diffuse_surfel_model.py
```

实现：

```python
class DiffuseSurfelModel:
    xyz
    rotation
    scaling_2d
    opacity
    base_color
    roughness
    f0
    ks
```

必须保留原始 3DGS 模型作为可运行 baseline，但 RT-GS 配置必须使用 `DiffuseSurfelModel`。

---

### A-2：改造 CUDA Rasterizer

实现 2D surfel rasterization。

要求：

```text
支持位置、旋转、两个切向尺度和 opacity；
正确输出 RGB、depth、alpha；
输出 surfel normal；
支持材质属性 alpha blending；
对位置、旋转、缩放、opacity、颜色、材质参数反传。
```

---

### A-3：Material Maps

renderer 必须输出：

```python
Cd
alpha
depth
position
normal
roughness
f0
ks
```

必须输出对应 debug 图：

```text
diffuse_color.png
depth.png
normal.png
roughness.png
f0.png
ks.png
alpha.png
```

---

### A-4：Diffuse-only Loss

启用：

```text
L_rgb
L_norm
L_mono
L_perc
```

不启用：

```text
L_spec
L_depth
Reflection
Transmittance
Ray tracing
```

---

### A-5：Stage A 测试

必须新增：

```text
tests/test_surfel_normal.py
tests/test_material_compositing.py
tests/test_renderer_output_contract.py
tests/test_normal_depth_loss.py
```

必须验证：

```text
1. normal 已归一化；
2. normal face-forward 正确；
3. material maps 使用同一 alpha blend 权重；
4. roughness、f0、ks 无 NaN / Inf；
5. 原始数据集可训练；
6. RGB、depth、normal 都可保存；
7. checkpoint 可恢复。
```

---

## 10.4 Stage A 验收

```text
[ ] RT-GS 配置切换到 2D surfel；
[ ] diffuse-only 可训练；
[ ] renderer 完整输出材质图；
[ ] normal 与 depth 可视化稳定；
[ ] L_norm / L_mono / L_perc 可运行；
[ ] 不存在 R/T/ray tracing 代码路径；
[ ] 所有 debug 图可生成；
[ ] checkpoint 可保存和恢复；
[ ] STATUS.md 指向 Stage B；
[ ] Git commit 完成。
```

---

# 11. Stage B：Ray Tracing 与 Reflection Gaussian

## 11.1 目标

实现：

```text
独立 Reflection Gaussian
+ 可微 Gaussian ray tracer
+ reflection ray generation
+ full microfacet reflection shading
```

本阶段不实现 Transmittance Gaussian 和 mesh-guided second bounce。

---

## 11.2 Stage B 实现任务

### B-1：Ray Tracer 基础

新增：

```text
raytracer/
├── tracer.py
├── ray_utils.py
├── acceleration_structure.py
├── differentiable_raytrace.py
└── tests/
```

必须支持：

```text
D/R/T 通用输入接口；
BVH 加速；
可微传播；
chunked rays；
color / alpha / depth / hit mask 输出。
```

---

### B-2：Reflection Gaussian

新增：

```text
scene/reflection_surfel_model.py
```

创建独立：

```python
reflection_model
optimizer_reflection
densification_state_reflection
```

---

### B-3：Reflection Rays

从 Stage A 的 surface position 与 normal 生成：

```python
d_ref = reflect(d_cam, normal)
```

调用：

```python
Cr, Ar, Dr, hit = raytrace(
    reflection_model,
    surface_position + eps * d_ref,
    d_ref,
)
```

输出：

```text
reflection_color.png
reflection_alpha.png
reflection_depth.png
reflection_hit_mask.png
```

---

### B-4：Full Microfacet Reflection

实现：

```text
GGX / Trowbridge-Reitz D
Schlick Fresnel F
Smith GGX G
full BRDF
不使用 split-sum approximation
```

此阶段最终颜色：

```python
C = (1.0 - ks) * Cd + ks * wr * Cr
```

---

### B-5：Specular Mask Pipeline

实现透明对象 mask 加载、验证和可视化。

加入：

```python
L_spec = mean(
    transparent_mask * relu(K0 - ks)
)
```

默认：

```yaml
K0: 0.9
lambda_spec: 0.2
```

---

## 11.3 Stage B 验收

```text
[ ] 独立 Reflection Gaussian 存在；
[ ] ray tracer 对 R 可运行；
[ ] reflection ray 方向正确；
[ ] Cr / Ar / Dr / hit map 可视化；
[ ] D 与 R 优化器独立；
[ ] full microfacet BRDF 可运行；
[ ] 不使用 split-sum；
[ ] 透明区域 ks 有明显提升；
[ ] 无 NaN / Inf；
[ ] Stage A 功能未退化；
[ ] STATUS.md 指向 Stage C。
```

---

# 12. Stage C：透明物体 Mesh 与 Two-Hit Geometry

## 12.1 目标

实现透明物体的：

```text
mask
surface mesh
front / back intersection
two-hit cache
```

本阶段尚不要求最终 T 分支重建内部物体。

---

## 12.2 Stage C 实现任务

### C-1：透明物体 mask 工具链

新增：

```text
preprocess/
├── make_transparent_masks.py
├── validate_masks.py
└── convert_masks.py
```

要求：

```text
支持直接加载人工 mask；
可选支持 GroundingDINO + SAM2；
支持 mask_soft / mask_hard / mask_eroded；
每帧 mask 与图像尺寸严格一致；
输出 mask overlay。
```

---

### C-2：TSDF Mesh Extraction

从 Diffuse surfel geometry 提取 mesh。

新增：

```text
geometry/
├── tsdf_fusion.py
├── mesh_extraction.py
├── transparent_mesh_filter.py
└── mesh_cleanup.py
```

必须支持：

```text
从训练视角深度融合；
导出 PLY；
mesh 清理；
透明物体相关 component 筛选；
法线统一；
统计 mesh 是否闭合。
```

默认流程：

```text
从 diffuse geometry 提取 mesh
→ 用透明 mask 投影筛选透明壳相关部分
→ 清理碎片
→ 固定 mesh
```

---

### C-3：Two-Hit Ray Intersection

实现：

```python
t_near, t_far, valid_two_hit = intersect_two_hits(
    mesh,
    camera_center,
    camera_ray_direction,
)
```

仅对透明 mask 区域计算或缓存。

必须输出：

```text
near_depth.png
far_depth.png
two_hit_valid.png
back_surface_points.ply
```

---

### C-4：Two-Hit Cache

mesh 固定后，预计算全部训练视角：

```text
cache/mesh_hits/
├── 00000.npz
├── 00001.npz
└── ...
```

每个文件：

```python
{
    "t_near": ...,
    "t_far": ...,
    "valid_two_hit": ...,
    "back_position": ...,
}
```

训练中不得每 iteration 重算 mesh intersection。

---

## 12.3 Stage C 验收

```text
[ ] 透明 mask 加载稳定；
[ ] 可导出透明物体 mesh；
[ ] mesh 不含大面积无关背景；
[ ] two-hit 对透明区域有效；
[ ] 对有效像素始终满足 t_far > t_near；
[ ] back_position 位于透明壳背面；
[ ] cache 可保存、加载与版本校验；
[ ] Stage A/B 未退化；
[ ] STATUS.md 指向 Stage D。
```

---

# 13. Stage D：Transmittance Gaussian 与完整 RT-GS

## 13.1 目标

实现：

```text
Transmittance Gaussian
first-bounce inside tracing
second-bounce outside tracing
Ct 合成
L_depth
完整 D + R + T 联合优化
```

---

## 13.2 Stage D 实现任务

### D-1：Transmittance Gaussian

新增：

```text
scene/transmittance_surfel_model.py
```

创建：

```python
transmittance_model
optimizer_transmittance
densification_state_transmittance
```

T 分支只在透明 mask 区域启用。

---

### D-2：First Bounce

```python
d_trans = d_cam

Cin, Ain, Din, Tin_hit = raytrace(
    transmittance_model,
    position + eps * d_trans,
    d_trans,
)
```

输出：

```text
inside_color.png
inside_alpha.png
inside_depth.png
inside_hit_mask.png
```

---

### D-3：Second Bounce

从 Stage C cache 的背面点出发：

```python
back_position = mesh_cache["back_position"]

Cout, Aout, Dout, Dout_hit = raytrace(
    diffuse_model,
    back_position + eps * d_trans,
    d_trans,
)
```

输出：

```text
outside_color.png
outside_alpha.png
outside_depth.png
outside_hit_mask.png
```

---

### D-4：Transmittance Composite

默认：

```python
Ct = Cin + (1.0 - Ain) * Cout
```

输出：

```text
transmittance_color.png
transmittance_alpha.png
```

---

### D-5：Depth Constraint

在全局训练步数达到 40,000 后启用：

```python
L_depth = mean(
    mask_eroded
    * valid_two_hit
    * relu(Din - t_far)
)
```

输出：

```text
depth_violation.png
```

定义：

```python
depth_violation = relu(Din - t_far)
```

---

### D-6：完整合成

```python
C = (1.0 - ks) * Cd + ks * (
    wr * Cr + wt * Ct
)
```

必须输出分量：

```text
diffuse_contribution.png
reflection_contribution.png
transmittance_contribution.png
final.png
```

---

## 13.3 Stage D 验收

```text
[ ] D / R / T 三套 Gaussian 独立训练；
[ ] Cin 主要呈现内部物体；
[ ] Cout 主要呈现透明罩后方外界；
[ ] Ct 不大面积吸收外部背景；
[ ] L_depth 生效后 Din 大部分不超过 t_far；
[ ] R 分支呈现反射而非普通 diffuse；
[ ] 完整合成可重建原始视角；
[ ] 新视角无明显分支闪烁；
[ ] STATUS.md 指向 Stage E。
```

---

# 14. Stage E：稳定化、评测、消融与导出

## 14.1 目标

完成：

```text
稳定训练
分支级验证
论文式消融
定量评估
模型导出
可复现实验说明
```

---

## 14.2 必须实现的消融

```text
1. w/o transmittance Gaussian
2. w/o specular loss
3. w/o depth loss
4. w/o mesh-guided ray tracing
5. with split-sum approximation
6. full model
```

每种消融输出：

```text
final image
Cd
Cr
Cin
Cout
Ct
ks
Din
depth violation
PSNR / SSIM / LPIPS
```

---

## 14.3 分支级评估

除全图指标外，必须提供：

```text
内部对象区域 PSNR / SSIM / LPIPS
透明物体区域 PSNR / SSIM / LPIPS
背景区域 PSNR / SSIM / LPIPS
reflection 区域视觉对比
跨视角 T 分支一致性
Din <= t_far 比例
transparent mask 中 ks 分布
```

---

## 14.4 导出模式

必须支持：

```text
render_mode:
  final
  diffuse_only
  reflection_only
  transmittance_only
  inside_only
  outside_only
```

必须支持：

```text
export diffuse_model
export reflection_model
export transmittance_model
export mesh
export mesh_hit_cache
```

---

## 14.5 Stage E 验收

```text
[ ] 全部消融可运行；
[ ] 全部模型可导出；
[ ] 全部 render mode 可运行；
[ ] 完整训练可恢复；
[ ] 训练配置写入 checkpoint；
[ ] 最终 README 可从空环境复现；
[ ] 所有工程假设已写入 DECISIONS.md；
[ ] 发布版本有 Git tag。
```

---

# 15. 最终训练时序

这不是开发阶段，而是最终 `train.py` 中的完整训练 schedule。

```text
Step 0–2,999:
    仅训练 D
    rasterization
    C = Cd

Step 3,000–19,999:
    联合训练 D + R
    reflection ray tracing
    C = (1-ks)Cd + ks * wr * Cr

Step 20,000:
    从 D 提取 mesh
    固定 mesh
    预计算所有 two-hit cache

Step 20,001–39,999:
    联合训练 D + R + T
    first-bounce T
    second-bounce D
    不启用 L_depth

Step 40,000–59,999:
    联合训练 D + R + T
    启用 L_depth
    完整 RT-GS loss
```

总训练：

```text
60,000 steps
```

---

# 16. Densification 与 Pruning 标准

三套模型必须独立 densify/prune：

```python
densify_diffuse()
densify_reflection()
densify_transmittance()
```

建议梯度来源：

```text
Diffuse:
    rasterization screen-space gradient
    geometry / normal / depth gradient

Reflection:
    ray tracer 的 3D position gradient
    reflection ray hit coverage

Transmittance:
    透明 mask 区域 ray tracing 3D gradient
    depth violation
    internal-object coverage
```

必须记录：

```text
step
diffuse_count
reflection_count
transmittance_count
diffuse_pruned
reflection_pruned
transmittance_pruned
```

禁止让 T Gaussian 在透明物体外无限制增殖。

---

# 17. Debug 输出标准

每 500 或 1,000 step，对固定验证视角输出：

```text
final.png

diffuse_color.png
diffuse_depth.png
alpha.png
normal.png
roughness.png
f0.png
ks.png

reflection_color.png
reflection_alpha.png
reflection_depth.png
reflection_hit_mask.png

inside_color.png
inside_alpha.png
inside_depth.png
inside_hit_mask.png

outside_color.png
outside_alpha.png
outside_depth.png
outside_hit_mask.png

transmittance_color.png
transmittance_alpha.png

near_depth.png
far_depth.png
two_hit_valid.png
depth_violation.png

diffuse_contribution.png
reflection_contribution.png
transmittance_contribution.png
```

必须同时保存：

```text
ground_truth.png
transparent_mask.png
overlay.png
```

---

# 18. 必须实现的测试

```text
test_surfel_normal.py
test_material_compositing.py
test_renderer_output_contract.py
test_reflection_ray.py
test_microfacet.py
test_mesh_two_hit.py
test_transmittance_composite.py
test_raytrace_gradients.py
test_checkpoint_resume.py
test_toy_scene_end_to_end.py
```

至少验证：

```text
normal 已归一化；
normal face-forward 正确；
alpha compositing 一致；
GGX 无 NaN / Inf；
Fresnel 在掠射角升高；
reflection direction 正确；
two-hit 满足 t_far > t_near；
alpha-over Ct 合成正确；
ray tracer 梯度与 finite difference 基本一致；
checkpoint 可恢复；
合成球壳场景可跑完整 D/R/T。
```

---

# 19. 合成测试场景

必须建立一个最小 synthetic scene：

```text
透明球壳
+ 球内彩色物体
+ 球后彩色背景板
+ 侧面高对比反射物体
```

验收：

```text
T 分支恢复内部物体；
Cout 恢复背景板；
R 分支恢复侧面反射物体；
D 分支不应包含主要内部物体；
去掉 L_depth 后可观察背景泄漏；
去掉 L_spec 后可观察 D 分支抢占内部物体；
去掉 mesh guidance 后出现 two-bounce 错位。
```

---

# 20. 状态与阶段切换规则

每次完成任务后，必须更新：

```markdown
# docs/STATUS.md

Current stage:
Current branch:
Last verified commit:
Completed:
Tests passed:
Known failures:
Current metrics:
Next exact task:
Blocked by:
```

每次出现论文未公开细节或工程取舍，必须新增：

```markdown
# docs/DECISIONS.md

## Decision ID
Date:
Question:
Chosen implementation:
Alternatives:
Why:
Paper fidelity:
Impact:
Required ablation:
```

禁止在下列情况下进入下一阶段：

```text
当前阶段未通过验收；
debug 图缺失；
测试未运行；
未记录关键设计；
Git 工作区仍有未提交关键改动；
基础训练已经退化但未解释。
```

---

# 21. Codex 工作纪律

每次接到任务时：

```text
1. 读取 AGENTS.md；
2. 读取 RTGS_MASTER_PLAN.md；
3. 读取 STATUS.md；
4. 读取 DECISIONS.md；
5. 读取当前 STAGE 文件；
6. 只实现当前阶段允许的内容；
7. 不得提前实现后续阶段；
8. 修改后运行对应测试；
9. 更新文档；
10. 提交 Git。
```

每次回复必须包含：

```text
当前阶段：
本次修改：
未修改内容：
运行过的命令：
测试结果：
生成的 debug 输出：
已知风险：
下一步：
Git commit：
```

禁止：

```text
一次性重写整个仓库；
跳过 2D surfel 直接给 3DGS 加透明度；
把 D/R/T 合并为一个 field；
在训练时裁掉外部环境；
未实现就创建 dummy 输出；
未测试就声称完成；
不记录论文未公开的技术假设。
```

---

# 22. Tao layered early-joint scene-local annex (TAO-LJ-001)

This annex is part of the sole master plan; it is not a second specification.
Tao is an independent scene and must start from global 0. Existing Tao D-only
1k/20k/30k outputs and every TiHuBird checkpoint, mask, geometry release,
two-hit cache, or D/R/T output are historical evidence only and are forbidden
as Tao initialization, scale calibration, proxy geometry, optimizer state, or
resume input.

The Tao input identity is 112 RGB stems `000000`--`000111`, 112 StableNormal
priors, Tao COLMAP cameras/images/points3D, and 66,187 sparse points. Real
DiffusionRenderer inverse priors live under a fresh local Stage-A output and
must bind source/code/model/config/environment and per-file hashes. Its depth is
per-view relative evidence only; metric use requires a per-view depth or
inverse-depth scale/shift fit on projectable Tao COLMAP observations.

Automatic glass masks are review proposals only. They must be scene-agnostic,
contain soft/hard/eroded/overlay/contact-sheet/risk/metric products, and expose
no training or promotion path. Only a later explicitly accepted
`rtgs_tao_reviewed_glass_masks_v1` manifest may authorize geometry. The Tao
geometry builder then fits a six-plane cuboid from Tao COLMAP, Tao reviewed
masks, Tao RGB, and Tao DR normal/depth only. It must predeclare source-resolution
IoU, hard-mask two-hit coverage, 100% finite/order, crossing-count, watertight,
and fixed-view review gates. Runtime uses analytic cuboid intersection; caches
are audit/parity evidence and are never required for a novel view. Failure of
any gate blocks release rather than forcing a cuboid.

The versioned Tao renderer does not alter legacy TiHuBird behavior. Its formal
fast path issues exactly four black-background Gaussian traces in order:
`R_front` to strict-outside R, `T_direct` to strict-inside T,
`R_back_from_T` to the same strict-inside T after one back-face reflection, and
`Cout` to strict-outside D. D-interface position, normal, coverage, tint, ks,
f0, and roughness remain in the D/material namespace and do not form a fourth
radiance field. There is one back internal reflection, no recursion, no Snell
refraction, no dispersion, no caustics, and no multiple bounce.

All composition uses unclamped linear float values:

```text
internal = alpha * ks * (1-Ffront) * Cin
front_reflection = alpha * ks * Ffront * Rfront
back_internal_reflection =
  alpha * ks * (1-Ffront) * (1-Ain) * Fback * Rback
Cout_only =
  alpha * ks * (1-Ffront) * (1-Ain) * (1-Fback) * Cout
reflection = front_reflection + back_internal_reflection
no_reflection = D_direct + internal + Cout_only + uncovered_background
final = no_reflection + reflection
```

`Cin` is already premultiplied over black and must not be multiplied by `Ain`
again. Formal review exports preserve float tensors and write display-clamped
PNGs for final, ground truth, no-reflection, reflection, internal, front/back
reflection, Cout, D-direct, Cin/Ain, internal intrinsic RGB, and internal alpha.
Every export records max/mean linear closure error for
`final == no_reflection + reflection`.

Global-0 initialization uses all Tao COLMAP points for D, frozen analytic proxy
geometry for D-interface, fresh deterministic complete-3-sigma strict-outside R,
and strict-inside Tao COLMAP points plus deterministic support-safe fill for T.
No semantic internal-object mask, Ain positive push, TiHuBird internal loss,
checkpoint, resume, optimizer state, or PLY initialization is permitted.

The default operator is plan-only and must refuse an existing output. Only
explicit `--execute` may dispatch training. The planned schedule is global
1--250 appearance-first joint participation, 251--1,000 low-rate R/T geometry
with no R/T topology changes, and 1,001--3,000 bounded joint refinement, never
3,001+. Review/checkpoint nodes are 0, 100, 250, 500, 1,000, 2,000, and 3,000.
Full-frame RGB is always retained; the glass mask controls transparent paths and
interface supervision only. Unfiltered and ownership-class diagnostics run only
under `no_grad` review nodes.

TAO-LJ-001 stops after infrastructure and review proposal. No formal mask,
geometry release, plan execution, optimizer update, checkpoint, or PLY is
created. Until explicit human mask approval, the required verdict is
`TAO_GLASS_MASK_REVIEW_REQUIRED`; Stage E remains forbidden.

## 22.1 Tao DR geometry mask repair boundary (TAO-DR-MASK-REPAIR-001)

The original per-view threshold proposal is comparison evidence only. It must
not supervise geometry, define a silhouette objective, authorize acceptance, or
provide a fallback. Tao DR normal, relative depth, RGB, basecolor, and diffuse
albedo are instead fused as multi-view geometric likelihoods under Tao COLMAP.
Normal convention is selected by an exhaustive signed-permutation search after
camera-to-world transformation. Raw DR depth is never metric and is never
compared across views; each view independently selects a robust affine depth or
inverse-depth calibration on its projectable COLMAP observations, with weak
calibrations downweighted.

The repair bootstrap is a separate review-only six-plane cuboid implementation,
not the formal Tao geometry builder. It optimizes one finite, positive-extent,
watertight cuboid against fused DR boundary/interior/normal/depth evidence in
COLMAP scale and projects that single fixed geometry into all 112 cameras. A
failed predeclared gate must stop before mask directories are materialized; no
ellipse, convex hull, full-frame, or old-mask fallback is permitted. Soft masks
come from the new projected silhouette distance field.

The v2 schema is deliberately incompatible with formal mask and geometry
loaders. Its only allowed status is `proposal_requires_review`, with
`training_eligible=false`, `promotion_performed=false`, and
`formal_geometry_release=false`. Human approval remains necessary before any
formal Tao mask, geometry release, two-hit cache, or training run.
