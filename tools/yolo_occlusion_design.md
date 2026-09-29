# YOLO 遮挡训练版本 —— 技术设计文档

> 状态：设计已定稿（方案 A + M1/M2 物理分割 + severity 双模式），接口已钉死，等待实现。
> **M1 的遮挡形式已收敛为只做 random_box**，实现级设计见 [yolo_occlusion_m1_design.md](yolo_occlusion_m1_design.md)
> （本文件其余各节仍按全模式描述，M1 以 M1 文档为准）。
> 关联脚本：`train_pick_pointcloud_yolo.py`（stage 3/4 无遮挡，改造基底）、
> `train_pick_pointcloud_occluded.py`（stage 2 遮挡 baseline，参考对象）。

---

## 0. 摘要

**目标**：在 YOLO 掩码路线（stage 3/4）上加入**可控**遮挡，训练并评估策略在「工件被部分遮挡」下的鲁棒性。

**方案一句话**：把 stage 2 遮挡实验的几何挖点（M1 缺失型）搬到 YOLO 管线上，并新增一个正交的掩码膨胀机制（M2 膨胀型）模拟「边缘点被误采入」；两个机制**物理分割**（各自独立模块 / setter / CLI 参数组 / 独立开关），severity 支持**固定值**或**每 episode 在可设范围内随机**两种模式。

**已拍板的决策**：

| # | 决策 | 结论 |
|---|---|---|
| 1 | 遮挡层级 | 方案 A：点级几何（M1），外加一小块掩码级组件（M2），二者物理分割 |
| 2 | 掩码来源 | YOLO-seg（stage 3 每步跑 / stage 4 缓存跑），不是 Isaac 实例分割 |
| 3 | severity 模式 | 固定单值 或 per-episode `U[min, max]`，两种都支持 |
| 4 | 两个遮挡机制 | M1 缺失型（遮掉原有/漏检轴）+ M2 膨胀型（边缘误采/过分割轴），各自独立旋钮 |
| 5 | 实现顺序 | 先 M1（最小闭环），M2 单独做；实验一次只开一个维度 |

**开放问题**（第 10 节）：M2 是否纳入首版、range 下限是否固定 0、可视化旁路是否复用、跑 stage 3 还是 4。

---

## 1. 背景与动机

### 1.1 为什么走 YOLO 路线

真机侧掩码只能是 YOLO 出的。若训练全程吃 Isaac 的完美分割，策略见到的掩码分布和部署时差一截，这是**系统性偏差不是噪声**（spec §11.2），加零均值高斯补不回来。让训练直接吃 YOLO 掩码，这段 gap 从根上不存在（见 `configs/yolo_mask_source.py` docstring）。

### 1.2 YOLO 的两种系统性误差（本设计的两根轴）

`configs/yolo_mask_source.py` docstring 明确并列了两种：

| 误差 | 现象 | 误差轴 | 对应机制 |
|---|---|---|---|
| 漏检（false negative） | 「遮挡处缺一块」 | 数据**缺失** | **M1 缺失型** |
| 过分割（false positive） | 「边缘胖一圈」 | 数据**污染** | **M2 膨胀型** |

这两个是**独立的正交误差轴**，必须分开建模、分开控制，才能保证「失败可归因」（spec 的铁律）。

### 1.3 stage 2 遮挡实验已回答了什么、缺什么

`configs/occlusion.py` 的几何谓词（halfspace/sphere/random_sphere/random_box）回答的是「**完美分割**（Isaac 掩码）下，点云表征在工件被部分遮挡时还能不能学」。它不回答：

1. 掩码换成 YOLO（本身带误差）之后，同样的几何遮挡下还能不能学；
2. 「边缘点被误采入」这种**反向**误差（数据多出来而不是少掉）的影响。

本设计补这两点。

---

## 2. 现有管线（读代码结论，本设计的起点）

### 2.1 数据流（stage 3/4）

```
RGB ─► yolo_masks() ─► mask (B,H,W) bool
                             │
depth ───────────────────────┼─► mask_depth_to_pointcloud()
                             │      ① valid = mask & depth>0 & isfinite
                             │      ② _select_candidates 确定性预筛
                             │      ③ unproject → 基座系
                             │      ④ workspace 裁剪（越界点用首个界内点顶替）
                             │      ⑤ ★ occlusion_keep_fn 钩子（本设计 M1 注入点）
                             │      ⑥ FPS 采样 m_obj=64 点
                             │      ⑦ σ=1cm 高斯噪声
                             ▼
                   物体点 (B, 64, 3) ─┐
夹爪关键点 (B, 6, 3)（正运动学，不经相机）┴─► assemble_point_cloud ─► 扁平 217 维
```

- 观测 217 维 = 物体点 64×3 + 夹爪点 6×3 + 关节角 7。网络结构不变。
- **M2 注入点在 `yolo_masks()` 之后、`mask_depth_to_pointcloud()` 之前**（改 mask）。
- **M1 注入点在第 ⑤ 步钩子**（改点）。两者走 `mask_depth_to_pointcloud` 的**两个不同参数**（`mask` vs `occlusion_keep_fn`），共用反投影是 spec §1.1 强制的，不是耦合。

### 2.2 stage 对照

| stage | 掩码来源 | 相机 data_types | 备注 |
|---|---|---|---|
| 2 | `instance_id_segmentation_fast`（完美） | depth + 分割 | 现有遮挡实验的基底 |
| 3 | YOLO-seg（每步跑） | depth + rgb | 本设计的基底（无遮挡） |
| 4 | YOLO-seg（每 N=5 步跑，掩码缓存） | depth + rgb | 同上 + 掩码缓存，快 5 倍 |

### 2.3 M1 钩子的替换机制（理解 M1 的关键）

`mask_depth_to_pointcloud` 的 `occlusion_keep_fn` 在 workspace 裁剪之后、FPS 之前调用：
`keep=False` 的点**用该条目第一个存活点顶替**（与 workspace 裁剪同一替换法）。于是 FPS 的候选池里遮挡区被「挖空」，64 个采样点全部来自存活区 —— **观测维度不变（仍是 217），信息被削弱**。这就是「遮掉原有的点」的精确语义：不是真的少点，而是采样集中在可见区、遮挡区的几何信息消失。

---

## 3. 核心设计：两个正交遮挡机制

### 3.1 术语

- **M1 缺失型（removal）**：挖掉工件局部的一块（点级几何）。对应漏检轴。复用 `occlusion.py`。
- **M2 膨胀型（over-segmentation）**：掩码边界向外膨胀几像素，边缘外的像素被误采入（掩码级形态学）。对应过分割轴。新增 `mask_augment.py`。

严格说只有 M1 算「遮挡」，M2 是过分割误差；两者**不共用一个 severity 旋钮**，各自独立。

### 3.2 物理分割原则

| | M1 | M2 |
|---|---|---|
| 作用对象 | 点云（反投影后） | 掩码（反投影前） |
| 模块 | `configs/occlusion.py`（复用） | `configs/mask_augment.py`（新增） |
| setter | `set_occlusion_yolo(...)` | 同左函数的 `dilate_*` 参数组（见 5.3） |
| CLI | `--occlusion_*` | `--dilate_*` |
| 注入点 | `occlusion_keep_fn` 钩子 | `mask = dilate_mask(mask, px)` 一行 |

**代码层保证**：改 M1 不可能动到 M2 的逻辑，反之亦然；每个机制可独立 import、独立单测、独立开关。

**代码层保证不了**：效果的独立。管线顺序「膨胀 → 反投影 → 挖点」决定了两个机制的效果**叠加**（M2 加进来的边缘点可能随后被 M1 挖掉）。归因靠实验一次只开一个，不靠代码分割。

### 3.3 M1 缺失型（复用，不改）

几何谓词全部复用 `configs/occlusion.py`，定义在**工件局部系**（工件位姿随机化下 severity 才有确定含义）。模式与 severity 语义照旧：

| mode | severity 语义 | 随机性来源 |
|---|---|---|
| `halfspace` | 沿指定轴切除比例 | 无（确定性） |
| `sphere` | 球半径 / 工件外接球半径 | 无（确定性） |
| `random_sphere` | 同 sphere | 球心每 episode 在包围盒内随机 |
| `random_box` | 删除最近邻候选点比例（严格精确） | 盒心每 episode 从**面向相机侧 GT 表面点**随机，朝向随机 SO(3) |

谓词纯坐标、无 RNG、无 sim 依赖（真机侧可 import），这一层**一行不改**。

### 3.4 M2 膨胀型（新增）

对 YOLO 输出的 bool 掩码做**形态学膨胀**（max filter）：

```
dilate_mask(mask: (B,H,W) bool, radius_px: int) -> (B,H,W) bool
```

- 实现：`max_pool2d(float(mask), kernel=2r+1, stride=1, padding=r) > 0`，纯 tensor、无 sim 依赖，与 `occlusion.py` 同标准（可独立 import、单测）。
- 语义：膨胀出来的像素位于物体轮廓**之外**，其深度是背景深度（桌面/背后/悬空），反投影后成为轮廓外的一圈**误采点** —— 这正是「边缘点被误采入」的物理含义，是特性不是 bug。
- **0 档不是真 0**：YOLO 掩码天生带一点胖边，`dilate_px=0` 时仍有 YOLO 自然过分割量。想要「从真边界开始膨胀」需先腐蚀，那是另一个实验，首版不做。

---

## 4. 文件布局

| 文件 | 动作 | 内容 |
|---|---|---|
| `configs/mask_augment.py` | **新增** | M2：`dilate_mask()` 纯函数 |
| `configs/occlusion.py` | 不动 | M1 谓词已备好 |
| `configs/point_bridge_pointcloud.py` | 不动 | `occlusion_keep_fn` 钩子已备好 |
| `configs/xarm7_pick_pointcloud_env_cfg.py` | **修改** | 新增观测函数 `point_bridge_point_cloud_yolo_occluded`、setter `set_occlusion_yolo`；扩展 `randomize_occlusion_state` |
| `scripts/train/train_pick_pointcloud_yolo_occluded.py` | **新增** | 训练入口：yolo 版脚本 + 遮挡 CLI 参数组 |

---

## 5. 接口定义（钉死）

### 5.1 `configs/mask_augment.py`（新增）

```python
def dilate_mask(mask: torch.Tensor, radius_px: int) -> torch.Tensor:
    """对 bool 掩码做形态学膨胀（max filter）。

    Args:
        mask: (B, H, W) bool，YOLO 输出的二值掩码。
        radius_px: 膨胀半径（像素）。0 = 恒等（原样返回）。

    Returns:
        (B, H, W) bool，膨胀后的掩码。
    """
```

- 实现要点：`torch.nn.functional.max_pool2d`，kernel = `2*radius_px+1`，stride = 1，padding = `radius_px`，输入 `mask.float()`，输出 `> 0`。
- 约束：**不 import 任何 sim / env 依赖**（与 `occlusion.py` 同标准），真机侧可 import。

### 5.2 `configs/occlusion.py`（不动，备忘）

已有 `occlusion_keep(pts_base, obj_pos_b, obj_quat_b, obj_local_min, obj_local_max, mode, severity, axis, center, center_local=None, shape_quat_local=None) -> (B,P) bool keep` 与 `parse_occlusion_center(s)`，本设计直接复用，不改签名。

### 5.3 `configs/xarm7_pick_pointcloud_env_cfg.py`（修改）

#### 5.3.1 新观测函数

```python
def point_bridge_point_cloud_yolo_occluded(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg = SceneEntityCfg("camera_fixed"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    m_obj: int = M_OBJ,
    noise_std: float = NOISE_STD_M,
    yolo_weights: str = DEFAULT_YOLO_WEIGHTS,
    yolo_conf: float = YOLO_CONF,
    occlusion_mode: str = "none",
    occlusion_severity: float = 0.0,
    occlusion_axis: str = "x",
    occlusion_center: tuple[float, float, float] = (0.5, 0.5, 0.5),
    occlusion_severity_range: tuple[float, float] | None = None,
    dilate_px: int = 0,
    dilate_px_range: tuple[int, int] | None = None,
) -> torch.Tensor:
    """YOLO 掩码路线 + M1 几何遮挡 + M2 掩码膨胀。观测仍 217 维。

    结构 = point_bridge_point_cloud_yolo 逐字复制，加两处：
      ① mask = yolo_masks(rgb) 之后：if dilate_px_eff > 0: mask = dilate_mask(mask, dilate_px_eff)
      ② mask_depth_to_pointcloud 调用处：传 occlusion_keep_fn（照抄
        point_bridge_point_cloud_occluded 的构造块）

    dilate_px_eff / severity_eff 取值规则（固定 vs 随机范围，见第 6 节）：
      - dilate_px_range 为 None：dilate_px_eff = dilate_px（标量参数）
      - dilate_px_range 非 None：dilate_px_eff = _OCCLUSION_STATE_BUF["dilate_px"][env_id]
      - severity 同理：range 为 None 用标量，否则从 buffer 读
      - 构造期（首次 reset 前 ObservationManager 探测 shape）buffer 为空：
        给确定性占位（severity_min / dilate_px_min），值无关紧要、只保 shape。
    """
```

**wiring 骨架（只有这几行是新逻辑，其余照抄 yolo 版）**：

```python
    # ── M2：膨胀（一行，委托 mask_augment）────────────────────────────
    if dilate_px_eff > 0:
        mask = dilate_mask(mask, dilate_px_eff)

    # ── M1：遮挡谓词（委托 occlusion，构造块照抄 stage2 occluded 版）──
    keep_fn = None
    if occlusion_mode != "none" and severity_eff > 0.0:
        ...  # obj_pos_b/obj_quat_b、_object_local_bounds、buffer 读 center/quat
        keep_fn = lambda pts_base: occlusion_keep(pts_base, ..., severity=severity_eff, ...)

    candidate_pts, counts = mask_depth_to_pointcloud(
        mask=mask, depth=depth, K=..., cam_pos=..., cam_quat=...,
        m_obj=m_obj, noise_std=noise_std, occlusion_keep_fn=keep_fn,
    )
```

其余（夹爪关键点、`yolo_detect_ratio`/`visible_ratio` 日志、ZOH/empty 处理、`_LAST_POINT_CLOUD` 缓存、`_DEBUG_CAPTURE`）与 `point_bridge_point_cloud_yolo` **逐字一致**。

#### 5.3.2 新 setter

```python
def set_occlusion_yolo(
    env_cfg,
    mode: str = "none",
    severity: float = 0.0,
    axis: str = "x",
    center: tuple[float, float, float] = (0.5, 0.5, 0.5),
    severity_range: tuple[float, float] | None = None,
    dilate_px: int = 0,
    dilate_px_range: tuple[int, int] | None = None,
) -> None:
    """把 stage 3/4 的物体点观测换成带 M1 遮挡 + M2 膨胀的版本。

    必须在 set_observation_stage(cfg, 3 或 4) **之后**调用（它只替换观测函数，
    不重设相机 data_types / YOLO 参数）。相机保持 stage 3/4 的 depth + rgb，
    不要改成 stage 2 的 depth + 分割。
    """
```

- `term.func = point_bridge_point_cloud_yolo_occluded`，`term.params` 按 5.3.1 全量给出。
- 注册 reset 事件 `randomize_occlusion_state` 的条件（**并集**）：`mode in ("random_sphere","random_box")` **或** `severity_range is not None` **或** `dilate_px_range is not None`。
- 不碰 `env_cfg.scene.camera_fixed.data_types`、不碰 `invalidate_instance_lut` / `invalidate_yolo_mask`。

#### 5.3.3 扩展 `randomize_occlusion_state`

现有：random_box 从面向相机侧 GT 表面点抽盒心、SO(3) 朝向；random_sphere 在包围盒内均匀抽球心。**新增两条**（同样只采样真正 reset 的 `env_ids`，写 `_OCCLUSION_STATE_BUF`）：

```python
    if severity_range is not None:
        lo, hi = severity_range
        buf["severity"][env_ids] = lo + torch.rand((n,), device=env.device) * (hi - lo)
    if dilate_px_range is not None:
        dlo, dhi = dilate_px_range
        buf["dilate_px"][env_ids] = torch.randint(dlo, dhi + 1, (n,), device=env.device).float()
```

buffer 结构由 `{"center", "quat"}` 扩为 `{"center", "quat", "severity", "dilate_px"}`。随机性走全局 RNG（`cfg.seed` 播种），与工件位姿 DR 同源，可复现。

**severity 随机与 random_* 的随机中心是两个正交维度**：前者随机「切多少」（幅度），后者随机「切在哪」（位置/朝向），参数上分开、可叠加。

### 5.4 `scripts/train/train_pick_pointcloud_yolo_occluded.py`（新增）

= `train_pick_pointcloud_yolo.py` 副本 + 下列遮挡参数组。`_build_env_cfg` 中顺序：

```python
set_observation_stage(cfg, _STAGE, keep_appearance=args_cli.play)   # 先 stage 3/4
set_occlusion_yolo(cfg, mode=..., severity=..., axis=..., center=...,
                   severity_range=..., dilate_px=..., dilate_px_range=...)   # 后遮挡
```

CLI 参数表（新增部分；`--yolo_weights/--yolo_conf/--yolo_step/--num_envs/--checkpoint/--resume/--play/...` 全部照抄 yolo 版）：

| 参数 | 类型 / 默认 | 说明 |
|---|---|---|
| `--occlusion_mode` | str, `none` | `none/halfspace/sphere/random_sphere/random_box` |
| `--occlusion_severity` | float, `0.0` | M1 固定 severity（range 模式给定时忽略） |
| `--occlusion_axis` | str, `x` | halfspace 切哪一侧（`x/-x/y/-y/z/-z`） |
| `--occlusion_center` | str, `0.5,0.5,0.5` | sphere 球心（归一化分数） |
| `--occlusion_severity_min` / `--occlusion_severity_max` | float, None | **两个都给** → M1 per-episode `U[min,max]` |
| `--dilate_px` | int, `0` | M2 固定膨胀半径（像素） |
| `--dilate_px_min` / `--dilate_px_max` | int, None | **两个都给** → M2 per-episode 整数均匀随机 |

日志目录名把两个机制都编进去，多条曲线进同一张 TensorBoard 对照：

```
log_name = f"xarm7_pick_pointcloud_stage{_STAGE}_occ_{occ_tag}_dil{dil_tag}"
# occ_tag 沿用 occluded 版规则（none/halfspace_x_50/random_box_30…）
# dil_tag = f"{dilate_px}" 或 f"r{dmin}-{dmax}"（range 模式）
```

日志里连带存档 `configs/mask_augment.py`（新文件，复现缺了它说不清膨胀怎么做的）。

---

## 6. severity 固定 vs 随机范围

### 6.1 语义

- **固定模式**：所有 env、所有 episode 同一 severity。干净消融、可对比。
- **范围模式**：severity 每 episode 在 `[min, max]` 内独立均匀采样（per-env）。这是标准 DR：防止过拟合单一遮挡程度；下限 0 时区间自然包含无遮挡 episode，起锚定作用。

### 6.2 采样机制

- **采样时机**：reset 事件（`randomize_occlusion_state`），不是每步抖 —— 一个 episode 内遮挡区必须静止，否则点云每步乱跳。
- **粒度**：per-env 独立采样（只采样真正 reset 的 `env_ids`，其余保留上一 episode 状态）。
- **RNG**：全局 RNG，`cfg.seed` 播种，可复现。

### 6.3 边界情况

- **构造期占位**：ObservationManager 在首次 reset 前会调一次观测函数探测 shape，此时 buffer 为空。给确定性占位（severity 用 `severity_min`、dilate 用 `dilate_px_min`），与现有 center/quat 占位（包围盒中心、单位朝向）同法。
- **固定/范围互斥**：`severity_range` 给定则忽略标量 `severity`；CLI 层保证「`--occlusion_severity` 与 min/max 对不可同时生效」，只给 min 或只给 max 时报错。
- **几何保底**：halfspace 不切光、random_box 有 `num_cand-1` 保底，遮挡本身不会把 `counts` 打到 0，不会误触发 ZOH。

---

## 7. 端到端数据流（M1 + M2 全开时）

```
RGB ─► yolo_masks() ─► mask ──► dilate_mask(mask, dilate_px_eff)   [M2，掩码级]
                                  │
depth ────────────────────────────┼─► mask_depth_to_pointcloud(
                                  │       mask=dilated_mask,        ← M2 的结果从这里进
                                  │       occlusion_keep_fn=fn)     ← M1 从这里进
                                  │       ①~④ 照旧
                                  │       ⑤ fn(pts_frame) → keep；遮挡区用首个存活点顶替  [M1]
                                  │       ⑥ FPS（只在存活区采样 64 点）
                                  │       ⑦ σ=1cm 噪声
                                  ▼
                         物体点 (B,64,3) ──► 217 维观测
```

M1 与 M2 在 `mask_depth_to_pointcloud` 内部外的唯一交汇是两个**不同参数**；内部顺序固定：先膨胀（进池子前）后挖点（池子里），效果叠加但代码零耦合。

---

## 8. 交互与注意事项

1. **M1 severity 语义不干净（与 stage2 版最本质的差别）**：`occlusion_keep` 定义在工件局部系（绝对几何），与 YOLO 掩码覆盖无关。实际可见比例 = **YOLO 覆盖 ∩ 遮挡保留**，severity 不再等于最终缺失比例。日志里盯 `visible_ratio`，别拿它和 stage2 曲线比绝对值。
2. **random_box 盒心来自 GT 表面，不是 YOLO 掩码**：遮挡区可能落在 YOLO 本来就检不到的部分上 → 几何遮挡 no-op，有效遮挡 < severity。stage2 不会发生（Isaac 掩码完整）。想避开就先用 halfspace/sphere（对称谓词，不依赖表面朝向）。
3. **M2 的 0 档不是真 0**：YOLO 天然胖边仍在。dilation 是「额外膨胀量」。
4. **empty / ZOH**：复用 yolo 版既有处理（RGB 快照 + strike 计数，stage3 落盘不崩溃、stage4 强制重跑 YOLO）。M1/M2 都不会把 `counts` 打到 0。
5. **stage 4（缓存掩码）与遮挡正交**：掩码缓存是图像空间、遮挡是对象局部空间，可叠加。但 reset 时 `invalidate_yolo_mask` 与 `randomize_occlusion_state` **两个事件都要注册**（stage 4 分支已注册前者，后者由 `set_occlusion_yolo` 补）。
6. **代码物理分割 ≠ 效果独立**：膨胀 → 反投影 → 挖点的顺序决定了效果叠加。归因靠实验一次开一个。
7. **热启动源**：从 **stage 3/4 的 YOLO checkpoint** 续训，不是 stage 2 的（掩码分布不同，从 stage2 续等于换任务）。
8. **M2 膨胀像素的深度是背景深度**：反投影出的误采点在轮廓外（悬空/桌面/背后），不是贴在表面上——这是「边缘误采」的物理含义，验收时别当成 bug 修。

---

## 9. 实验与消融计划（一次只开一个维度）

全部从 stage 3/4 checkpoint 热启动（`--resume --checkpoint <stage3 run>/model_xxx.pt`），默认 `--num_envs 16`，盯 `yolo_detect_ratio` 与 `visible_ratio` 两条曲线。

1. **M1 固定 severity 扫点**（先 halfspace，语义最干净）：
   ```bash
   # severity 0（对照）/ 0.3 / 0.5 / 0.7
   .../train_pick_pointcloud_yolo_occluded.py --num_envs 16 \
       --checkpoint logs/xarm7_pick_pointcloud_stage3/<run>/model_xxx.pt --resume \
       --occlusion_mode halfspace --occlusion_axis x --occlusion_severity 0.5
   ```
2. **M1 随机范围**：`--occlusion_severity_min 0.0 --occlusion_severity_max 0.7`（与固定 0.5 对照）。
3. **M2 单独扫点**：`--dilate_px 2 / 4 / 6`（`--occlusion_mode none`）。
4. **M1 + M2 叠加**：最后再做，只在与单独跑的曲线对比时使用。

判读规则：曲线掉了，先查当前开了哪个维度、severity 多大；两个都开时不许下结论。

---

## 10. 开放问题（实现前确认）

| # | 问题 | 建议 |
|---|---|---|
| 1 | M2（膨胀型）是否纳入首版 | **先只做 M1**（最小闭环），M2 的 `mask_augment.py` 单独做，更干净 |
| 2 | severity range 下限是否固定 0 | 建议支持 `[min, max]`，默认 min=0（用户说「可设上限」，留 min 为将来排除无遮挡 episode） |
| 3 | 是否复用 stage2 occluded 的可视化旁路（`_OCCLUSION_VIS` / `dense_occlusion_pixel_map`） | 建议复用 M1 部分；M2 的验收 dump 膨胀掩码 diff 图即可 |
| 4 | 跑 stage 3 还是 stage 4 | 建议先 stage 4（快 5 倍、掩码滞后本身与部署一致），stage 3 做终验 |
