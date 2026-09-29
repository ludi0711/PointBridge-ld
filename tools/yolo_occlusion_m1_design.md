# M1 缺失型遮挡（random_box）—— 实现级设计

> 上级规格：[yolo_occlusion_design.md](yolo_occlusion_design.md)（总规格，含 M2 与实验计划）。
> 本文件是其中 §5.3 的实现展开，只覆盖 **M1 缺失型、且遮挡形式只做 random_box**。
> 状态：范围已收敛，接口钉死，等待实现。

---

## 0. 范围与决策记录

| 决策 | 结论 |
|---|---|
| 遮挡形式 | **只做 random_box**：盒心每 episode 从面向相机侧 GT 表面点随机、朝向随机 SO(3)，按各向异性椭球度量删掉离盒心最近的 severity 比例候选点。halfspace / sphere / random_sphere **不实现**（扩展点见 §5） |
| severity 模式 | 固定单值 或 per-episode `U[min, max]`，两种都支持 |
| 观测维度 | 217 不变；网络结构不变；critic 特权状态不变 |
| 掩码来源 | YOLO-seg，stage 3 / stage 4 都支持（`yolo_step` 参数合一，见 §2.2） |
| `occlusion.py` | 一处最小扩展：`severity` 支持 `(B,)` 张量（per-env 随机化的必要条件）。**标量路径代码一字未动**，张量路径为新增分支，向后兼容由构造保证（见 §2.1） |
| M2 膨胀型 | 不在本文件范围，签名留扩展点（见 §5） |

**random_box 的 severity 语义**（沿用 `occlusion.py` docstring）：severity = 删除比例。按随机朝向的各向异性椭球度量（轴长 = 工件局部包围盒 span）删掉离盒心最近的 severity 比例候选点——比例严格精确、删出的是一块连通「阴影」、方向随机、无需搜索。盒心来自相机侧表面点，阴影必落在相机可见面。

---

## 1. 改动清单

| 文件 | 动作 | 内容 |
|---|---|---|
| `configs/occlusion.py` | **修改（唯一动到的既有模块）** | 入口守卫 + random_box 双路径（标量原代码未动 + 张量新分支），约 15 行 |
| `configs/xarm7_pick_pointcloud_env_cfg.py` | **修改** | 新增观测函数 `point_bridge_point_cloud_yolo_occluded`、setter `set_occlusion_yolo`；扩展 `randomize_occlusion_state` 与 `_OCCLUSION_STATE_BUF` |
| `scripts/train/train_pick_pointcloud_yolo_occluded.py` | **新增** | yolo 版训练脚本副本 + random_box 遮挡 CLI 参数组 |
| `configs/point_bridge_pointcloud.py` | 不动 | `occlusion_keep_fn` 钩子已存在 |
| `configs/yolo_mask_source.py` | 不动 | 掩码来源不变 |

---

## 2. 逐处设计

### 2.1 `configs/occlusion.py`：severity 支持 `(B,)`

**为什么必须改**：Isaac Lab 各 env 独立 reset，「每 episode 随机 severity」本质上就是 per-env 随机，severity 必须能按行表达。改动保持模块 docstring 的三条约束：确定性（无 RNG）、纯坐标运算、真机可 import。

**改造点 1 —— 入口守卫**（现第 90-91 行）：

```python
# 原：
if mode == "none" or severity <= 0.0:
    return keep
# 新：
if mode == "none":
    return keep
sev = torch.as_tensor(severity, dtype=pts_base.dtype, device=pts_base.device)
if sev.dim() == 0 and float(sev) <= 0.0:
    return keep
if sev.dim() != 0 and mode != "random_box":
    raise ValueError("(B,) 张量 severity 仅 random_box 支持（per-env 删除比例）；"
                     "halfspace/sphere/random_sphere 请用标量")
# 张量 severity 的零行不提前返回 —— random_box 的 k=0 自然全保留（见下）。
```

- 标量路径的判定条件与原式**逐字等价**（`mode=="none"` 或标量 `severity<=0.0` → 提前返回），只是拆成两步；
- 显式拒绝「非 random_box 模式 + 张量 severity」：既有调用方永远传标量 float，此 raise 对它们不可达；也杜绝了张量在 halfspace 等模式里被错误广播的静默语义变化。

**改造点 2 —— random_box 分支双路径**（现第 126-147 行）：

```python
# 标量 severity：原 topk 路径一字未动（现有 stage2 调用方逐位一致，零风险）
k = min(max(int(round(severity * num_cand)), 0), num_cand - 1)
keep = torch.ones(local.shape[0], num_cand, dtype=torch.bool, device=pts_base.device)
if k > 0:
    _, del_idx = torch.topk(d2, k, dim=1, largest=False)
    keep.scatter_(1, del_idx, torch.zeros_like(del_idx, dtype=torch.bool))

# 新增分支（仅 sev.dim() != 0 可达）：per-env 删除数不同，topk 无法按行取 k
k = (sev * num_cand).round().long().clamp_(0, num_cand - 1)      # (B,)，至少留 1 个存活点
rank = torch.argsort(torch.argsort(d2, dim=1), dim=1)            # (B,P) 稠密秩，0 = 离盒心最近
keep = rank >= k.unsqueeze(1)
```

- **标量路径 = 原代码照抄**，不经过任何新运算 —— 现有 stage 2 遮挡脚本（`--occlusion_mode random_box`）行为逐位不变；
- 张量路径只有新 YOLO 函数（范围模式）能走到，是纯新增行为；
- 改造点 3：docstring 的 `severity` 参数说明更新为 `float | (B,) Tensor`，并注明「随机性仍在上层采样，本层无 RNG；张量仅 random_box 支持」。

### 2.2 新观测函数 `point_bridge_point_cloud_yolo_occluded`

```python
def point_bridge_point_cloud_yolo_occluded(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg = SceneEntityCfg("camera_fixed"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),   # ← 比 yolo 版多（遮挡谓词要读工件位姿）
    m_obj: int = M_OBJ,
    noise_std: float = NOISE_STD_M,
    yolo_weights: str = DEFAULT_YOLO_WEIGHTS,
    yolo_conf: float = YOLO_CONF,
    yolo_step: int | None = None,                            # ← 新：None=stage3 每步跑；int=stage4 缓存跑
    occlusion_mode: str = "random_box",                      # ← 保留开关：mode="none" 即对照
    occlusion_severity: float = 0.0,                         # ← 固定模式
    occlusion_severity_range: tuple[float, float] | None = None,  # ← 范围模式
) -> torch.Tensor:
```

**骨架映射表**（实现 = 逐字抄 + 两处新逻辑）：

| 块 | 来源 | 动作 |
|---|---|---|
| 夹爪关键点 | [point_bridge_point_cloud_yolo:1044-1048](configs/xarm7_pick_pointcloud_env_cfg.py#L1044-L1048) | 逐字抄 |
| 掩码获取 | yolo 版 1058-1060 / cached 版 1204-1223 | `yolo_step is None` 抄前者（每步 YOLO），否则抄后者（缓存分支） |
| keep_fn 构造 | [point_bridge_point_cloud_occluded:2444-2483](configs/xarm7_pick_pointcloud_env_cfg.py#L2444-L2483) | 抄 + severity_eff 改造（见下） |
| 反投影 | yolo 版 1070-1078 | 抄，调用处传 `occlusion_keep_fn=keep_fn` |
| 可视化旁路 | occluded 版 2503-2524 | 抄（`_OCCLUSION_VIS` + `dense_occlusion_pixel_map` 标红被挖像素，验收用） |
| 日志 / ZOH / `_LAST_POINT_CLOUD` / DEBUG | yolo 版 1064、1080-1131 | 逐字抄（stage4 分支照抄 cached 版的 strikes→强制重跑 YOLO 逻辑） |

**新逻辑 1 —— severity_eff 解析**（替换 occluded 版里的标量 `occlusion_severity`）：

```python
# 固定模式：标量，severity=0 或 mode=none 时不构造 keep_fn（对照不变量，见 §3.7）
keep_condition = occlusion_mode != "none" and occlusion_severity > 0.0
severity_eff = occlusion_severity
if occlusion_severity_range is not None:
    keep_condition = occlusion_mode != "none"
    state = _OCCLUSION_STATE_BUF.get(id(env))
    if state is not None and state.get("severity") is not None:
        severity_eff = state["severity"]                       # (B,) per-env
    else:
        severity_eff = occlusion_severity_range[0]             # 构造期占位（shape 无关）
```

**新逻辑 2 —— keep_fn 构造**（照抄 occluded 版 2456-2483，去掉 axis/center，severity 用 `severity_eff`）：

```python
if keep_condition:
    obj_pos_b, obj_quat_b = pose_in_frame(obj.data.root_pos_w, obj.data.root_quat_w,
                                          base_pos_w, base_quat_w)
    obj_lo, obj_hi = _object_local_bounds(env)
    state = _OCCLUSION_STATE_BUF.get(id(env))
    if state is None:                       # 构造期占位：中心=包围盒中心、单位朝向
        center_local = ((obj_lo + obj_hi) * 0.5).unsqueeze(0).expand(num_envs, 3)
        shape_quat_local = torch.tensor([1., 0., 0., 0.], device=device,
                                        dtype=obj_lo.dtype).unsqueeze(0).expand(num_envs, 4)
    else:
        center_local, shape_quat_local = state["center"], state["quat"]

    def occlusion_keep_fn(pts_base: torch.Tensor) -> torch.Tensor:
        return occlusion_keep(pts_base, obj_pos_b, obj_quat_b, obj_lo, obj_hi,
                              mode=occlusion_mode, severity=severity_eff,
                              center_local=center_local, shape_quat_local=shape_quat_local)
```

其余（空掩码 ZOH、strike 计数、`visible_ratio` / `yolo_detect_ratio` 日志）与 yolo 版**逐字一致**——包括「stage4 缓存掩码下连续空检 → 强制重跑 YOLO + RGB 快照」的既有分支。

### 2.3 新 setter + 事件扩展

```python
def set_occlusion_yolo(
    env_cfg,
    severity: float = 0.0,
    severity_range: tuple[float, float] | None = None,
    yolo_step: int | None = None,
    mode: str = "random_box",
) -> None:
```

- **前置条件**：必须在 `set_observation_stage(cfg, 3 或 4)` **之后**调用（它只替换观测函数，不重设相机 data_types / YOLO 参数）。相机保持 depth + rgb。
- `term.func = point_bridge_point_cloud_yolo_occluded`，`term.params` 按 §2.2 签名全量给出。
- **事件注册规则**：
  - `mode != "none"` 且（`severity > 0` 或给了 range）→ 注册 `randomize_occlusion_state`（`params={"occlusion_mode": mode, "severity_range": severity_range}`）。**对照（severity=0 且无 range）不注册** —— 该事件要消耗全局 RNG，注册了会破坏与 stage 3/4 同 seed 的逐位可复现性（比"注册一个无副作用事件"更强）。
  - `yolo_step is not None` → 注册 `invalidate_yolo_mask`（与 `set_observation_stage` 的 stage 4 分支语义对齐；若 stage 4 已注册，同 key 覆盖为等价行为，且该事件不用 RNG，不影响对照复现）。
- 不碰 `env_cfg.scene.camera_fixed.data_types`、不碰 `invalidate_instance_lut`。

**`randomize_occlusion_state` 扩展**（现第 2210-2268 行）：签名追加可选参数 `severity_range: tuple[float, float] | None = None`；random_box 的盒心采样（相机侧 GT 表面 + SO(3) 朝向）**原样保留**，追加：

> 现有注册点（`set_occlusion_stage2` 第 2652 行）不传 `severity_range` → 默认 None → 新代码块永不执行，RNG 消耗顺序与原行为完全一致。stage 2 遮挡脚本零影响。

```python
    if severity_range is not None:
        lo, hi = severity_range
        buf["severity"][env_ids] = lo + torch.rand((n,), device=env.device) * (hi - lo)
```

- `_OCCLUSION_STATE_BUF` 初始化由 `{"center", "quat"}` 扩为 `{"center", "quat", "severity"}`（`severity` 初始 zeros，首次 reset 覆盖）。
- 随机性走全局 RNG（`cfg.seed` 播种），与工件位姿 DR 同源，可复现；只重采真正 reset 的 `env_ids`。

### 2.4 训练脚本 `scripts/train/train_pick_pointcloud_yolo_occluded.py`

= `train_pick_pointcloud_yolo.py` 副本，`_build_env_cfg` 顺序：

```python
set_observation_stage(cfg, _STAGE, keep_appearance=args_cli.play)   # _STAGE = 3 if yolo_step is None else 4
set_occlusion_yolo(cfg, severity=..., severity_range=...,
                   yolo_step=args_cli.yolo_step)                    # 之后注入遮挡
```

**新增 CLI 参数**（其余全部照抄 yolo 版）：

| 参数 | 类型 / 默认 | 说明 |
|---|---|---|
| `--occlusion_severity` | float, `0.0` | 固定删除比例 `[0,1]`。0 = 对照（与 stage 3/4 逐位一致，见 §3.7） |
| `--occlusion_severity_min` / `--occlusion_severity_max` | float, None | **两个都给** → per-episode `U[min,max]`，此时固定值被忽略；只给一个报错 |

**不提供 `--occlusion_mode`**：模式硬编码 `random_box`（对照用 `severity=0`，无需 mode 开关；将来加模式时再补开关，见 §5）。

- 参数校验：`0 ≤ severity ≤ 1`；`0 ≤ min ≤ max ≤ 1`；min/max 成对出现。
- `log_name = f"xarm7_pick_pointcloud_stage{_STAGE}_occ_rbox_{tag}"`：
  `tag = f"{int(round(severity*100))}"`（固定）或 `f"r{int(min*100)}-{int(max*100)}"`（范围）。
- 存档清单 = yolo 版原 4 件 **+ `configs/occlusion.py`**（遮挡谓词是这条路线的核心变量）。

---

## 3. 边界情况清单

1. **构造期占位**：ObservationManager 在首次 reset 前调一次观测函数探测 shape，此时 buffer 为空 → 盒心用包围盒中心、朝向单位四元数、severity 用 `severity_range[0]`。值与 shape 无关，首次 reset 后全部真实采样覆盖。
2. **盒心采样兜底**：相机侧半球无表面点时退回全表面均匀抽（现有代码 2260-2261 行，原样保留）。
3. **severity=0 的行**（范围模式下采样到 0）：`k=0 → rank >= 0` 全保留，谓词自然归零，无需特判。
4. **k 保底**：`clamp(0, num_cand-1)` 保证至少 1 个存活点 → `mask_depth_to_pointcloud` 的 fallback（`first_kept = argmax(keep)`）始终有效，遮挡本身不会把 `counts` 打到 0、不会误触发 ZOH。
5. **空掩码 / ZOH**：复用 yolo 版既有处理（stage3：strike 超限落盘 RGB 快照不崩溃；stage4：强制重跑 YOLO + 快照）。两者与遮挡正交。
6. **no-op 风险（random_box 特有，必须知晓）**：盒心采样用的是**物体 prim 的 GT 表面**，不是 YOLO 掩码。遮挡区可能落在 YOLO 本来就检不到的部分上 → 该 env 的有效删除 < severity。这是「随机盒 + YOLO 漏检叠加」的固有语义，不是 bug；日志的 `visible_ratio` 会如实反映。
7. **对照不变量**：`occlusion_mode="none"` 或固定 `severity=0` 时**不构造 keep_fn**，函数走与 stage 3/4 完全相同的代码路径 → 点云逐位一致（§4 验证项 2 锁死）。
8. **tie-break 差异**：仅新增的 `(B,)` 张量路径存在——argsort 秩与标量 topk 在 d2 完全并列时选择的点可能不同，删除数两路径都严格精确、确定性保持。标量路径是原代码，不受影响（§2.1）。
9. **stage 4 正交性**：掩码缓存是图像空间、遮挡是对象局部空间，可叠加；reset 时 `invalidate_yolo_mask` 与 `randomize_occlusion_state` **双事件都已注册**（前者来自 stage 4 分支，后者来自本 setter）。
10. **热启动源**：从 stage 3/4 的 YOLO checkpoint 续训（`--resume`），不是 stage 2 的。

---

## 4. 验证计划

1. **单测 `occlusion_keep`（纯 torch，无 sim）**：
   - 向后兼容由构造保证（标量路径代码未动），回归以 stage 2 遮挡脚本 smoke run 为准；
   - 新张量路径：删除数 = `round(sev * num_cand)` 精确；`sev=0` 行全保留、`sev=1` 行恰留 1 点；clamp 保底生效；
   - 无并列时标量与 `(B,)` 同值 → 删除集一致（交叉验证两条路径语义等价）；
   - 连通性抽查：球状点云 + 表面盒心 → 删除集单连通。
2. **对照不变量**：`--occlusion_severity 0` 与 yolo 版脚本同 seed 各跑若干步，debug capture 的点云逐位 diff 为 0。
3. **play 可视化**：`--play --occlusion_severity 0.5`，用 `_OCCLUSION_VIS` 旁路 + `dense_occlusion_pixel_map` 在录像帧上标红被挖像素 → 确认阴影形状连通、落在相机可见面、随机方向每 episode 变化。
4. **曲线判读**：`visible_ratio` 下降量 ≈ severity × YOLO 覆盖比例（粗略量级）；`yolo_detect_ratio` **不受影响**（遮挡在点级，不进 YOLO）；ZOH strike 计数不增长。
5. **范围模式复现性**：同 seed 两次运行曲线一致；各 env severity 直方图接近均匀。

---

## 5. 未来扩展点

| 扩展 | 改动量 |
|---|---|
| halfspace / sphere / random_sphere | `occlusion.py` 谓词现成（halfspace/sphere 的 `(B,)` severity 是各一行广播）；脚本加 `--occlusion_mode` 开关 + setter 透传 |
| M2 膨胀型 | 签名预留 `dilate_px` / `dilate_px_range`，插入点 = `yolo_masks` 之后一行；见 [yolo_occlusion_design.md](yolo_occlusion_design.md) §3.4 |
| 真机复用 | `occlusion.py` 保持纯 torch 无 sim 依赖，真机侧只 import 不注入（遮挡本身只在仿真训练脚本里使用） |
