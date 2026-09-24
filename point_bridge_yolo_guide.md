# xArm7 Point Bridge 点表征 PPO —— YOLO-seg 掩码路线

> 整理日期：2026-08-15  
> 适用仓库：`gx-VA-isaaclab`（Isaac Lab + rsl_rl）

---

## 一、路线概述

**Point Bridge** 是一套把相机深度图 + 分割掩码转成点云来训 PPO 的表征方案，设计
目标是让仿真训练和真机部署共用**同一条反投影-FPS-噪声管线**（规格 §1.1 禁止两份
实现），从而消除仿真→真机的点云分布 gap。

**YOLO-seg 路线**（stage 3）是在这个基础上把掩码来源从 Isaac 的完美实例分割换成
YOLO 网络的推理结果。动机：真机侧只能用 YOLO 出掩码；若训练全程用完美分割，YOLO
的边缘偏胖、遮挡处缺块这类**系统性偏差**在 sim2real 时就会暴露出来，加高斯噪声
补不回非零均值的偏差（规格 §11.2）。把 YOLO 直接放进训练循环，这段 gap 从根上
不存在。

---

## 二、观测向量布局

```
Actor obs（217 维）
  物体点云   M_OBJ=64 个点 × 3 = 192 维
  夹爪关键点  N_ROBOT=6 个点 × 3 =  18 维
  关节角      7 维
  ─────────────────────────────────────
  合计       217 维

Critic privileged（35 维，仅训练时可见）
  物体位姿 + 速度 + 夹爪状态等特权信息
```

点云由 `mask_depth_to_pointcloud` 生成，内含反投影 → FPS → 0.01m 高斯噪声三步。
**这个函数仿真和真机各调一次，绝不允许存在两份实现。**

---

## 三、训练 Stage 说明

| Stage | 掩码来源 | 相机数据类型 | 典型用途 |
|-------|----------|--------------|----------|
| -2 | 无相机（privileged state） | — | 快速验证奖励/网络是否通 |
| -1 | 无相机（privileged state） | — | 基线策略训练 |
| 0 | GT 分割（可视化） | depth + instance_seg | 调试点云可视化 |
| 1 | GT 分割（FPS 采样） | depth + instance_seg | 点云管线冒烟 |
| 2 | GT 分割（完整相机管线） | depth + instance_seg | 主训练，学会点云→动作 |
| **3** | **YOLO-seg** | **depth + rgb** | **sim2real 微调，消除掩码 gap** |

Stage 3 把相机的 `instance_id_segmentation_fast` 整路摘掉，换成 RGB。分割图没有
任何消费者了，相机显存持平（省了 AOV 渲染开销）。

**推荐流程：先跑到 stage 2 收敛，再从 stage 2 checkpoint 续训 stage 3。**
Stage 2 已把"点云→动作"学会，stage 3 只需要几百次迭代适应掩码分布的变化。

---

## 四、YOLO 模型

### 使用模型

`20260815.pt`（YOLO26n-seg，end2end=True，3.05M 参数，imgsz=640，batch=12，epochs=600）

```
train_metrics: precision(M)=1.0  recall(M)=0.9988  mAP50(M)=0.995  mAP50-95(M)=0.9813
```

### 为什么选 20260815.pt 而不是 best.pt

两个模型都是 YOLO26n-seg / end2end 架构，在 560 帧仿真图上的统计对比：

| 指标 | best.pt | 20260815.pt |
|------|---------|-------------|
| 漏检（0 det） | 1/560 | **0**/560 |
| 单检（1 det） | 541/560 | 547/560 |
| 多检（≥2 det，end2end 重复） | 18/560 | 13/560 |
| conf 均值 | 0.9307 | 0.9344 |
| 掩码中位面积 | 9406 px | 12370 px |
| 掩码最小面积 | 5364 px | 8202 px |

两个数字上差不多，但**决定性差异在遮挡处理**：

当夹爪压在工件中间时，`best.pt` 把遮挡物当成物体边界，只分割出工件一侧（半块）。
`20260815.pt` 返回完整工件并在夹爪处挖缺口（正确行为）。

半块工件反投影出的点云质心会偏出约半个工件长度，这是比边缘胖瘦严重得多的**系统性误差**，
不随训练迭代收敛。多检（end2end 重复）的质心误差实测 <1mm，可以接受。

### end2end 重复检出的处理

YOLO26 是 NMS-free 架构，`predict(iou=...)` 完全无效。一对一匹配未收敛时同一目标
会出现两个近乎重合的掩码（IoU≈0.995）。本方案对单类单目标场景取 **conf 最高的一个**，
实测 max_conf 与并集的质心差 <1mm，不需要合并逻辑。

---

## 五、环境配置

### ultralytics 隔离安装（推荐，不碰 isaaclab 环境）

```bash
~/miniconda3/envs/isaaclab/bin/python -m pip install --no-deps \
    --target /home/gxai/Desktop/CZR/.yolo_deps \
    ultralytics==8.4.69 polars nvidia-ml-py ultralytics-thop
```

- 共 21MB、4 个**纯 Python** 包，无编译扩展，不存在 ABI 问题
- isaaclab 的 site-packages **一个文件都不改**，删掉目录即完全复原
- 实测 torch 2.8.0 / numpy 2.4.6 / scipy 1.11.4 全部原样

**`--no-deps` 是关键**：不加它 pip 会把 Isaac 的 scipy 1.11.4 升成 1.17.1，可能
破坏 Isaac 内部的数值依赖。

### 自动挂载

`configs/yolo_mask_source.py` 在首次调用 `get_yolo()` 时自动把 `.yolo_deps` 追加
到 `sys.path` 末尾（环境自带的包优先）。训练脚本不需要手动设置任何路径。

若想换目录，设环境变量 `YOLO_DEPS_DIR` 即可：

```bash
export YOLO_DEPS_DIR=/your/custom/path
```

---

## 六、关键文件

| 文件 | 作用 |
|------|------|
| [configs/yolo_mask_source.py](configs/yolo_mask_source.py) | YOLO 封装：模型缓存、RGB 预处理、批量推理、去重。其余代码调这里，不直接 import ultralytics |
| [configs/xarm7_pick_pointcloud_env_cfg.py](configs/xarm7_pick_pointcloud_env_cfg.py) | 环境配置；`set_observation_stage(cfg, 3)` 分支在里面 |
| [configs/point_bridge_pointcloud.py](configs/point_bridge_pointcloud.py) | 反投影/FPS/噪声/零阶保持的唯一实现（§1.1 禁止两份） |
| [scripts/train/train_pick_pointcloud_yolo.py](scripts/train/train_pick_pointcloud_yolo.py) | Stage 3 训练入口，默认 num_envs=16，硬编码 _STAGE=3 |
| [scripts/train/train_pick_pointcloud.py](scripts/train/train_pick_pointcloud.py) | Stage -2/-1/0/1/2 训练入口（参考） |
| `/home/gxai/Desktop/CZR/.yolo_deps/` | ultralytics 隔离目录（21MB，不在仓库里） |
| `/home/gxai/Desktop/CZR/20260815.pt` | 当前使用的 YOLO 权重（不在仓库里） |

每次训练结束后，log 目录会自动存档四个文件快照：
`xarm7_pick_pointcloud_env_cfg.py` / `point_bridge_pointcloud.py` /
`yolo_mask_source.py` / `pointnet_actor_critic.py`，方便事后复现。

---

## 七、训练命令

### Stage 3：从 stage 2 checkpoint 续训（推荐）

```bash
cd /home/gxai/Desktop/CZR/gx-VA-isaaclab && \
~/IsaacLab/isaaclab.sh -p scripts/train/train_pick_pointcloud_yolo.py \
    --num_envs 16 \
    --checkpoint logs/xarm7_pick_pointcloud_stage2/2026-08-15_13-26-32/model_15100.pt \
    --resume --max_iterations 500 --headless
```

`--max_iterations` 是相对值：`tot_iter = start_iter + num_learning_iterations`，
从 iter=15100 续训 500 步后停在 15600。

### Stage 3：回放验证

```bash
~/IsaacLab/isaaclab.sh -p scripts/train/train_pick_pointcloud_yolo.py \
    --play --num_envs 1 \
    --checkpoint logs/xarm7_pick_pointcloud_stage3/<run>/model_xxx.pt
```

### Stage 2：参考命令（GT 分割，先跑这个再续 stage 3）

```bash
~/IsaacLab/isaaclab.sh -p scripts/train/train_pick_pointcloud.py \
    --stage 2 --num_envs 64 --headless
```

---

## 八、性能数据

**YOLO 推理开销（RTX 5090，20260815.pt，imgsz=640）**

| num_envs (B) | 每步耗时 | 每 env 耗时 |
|:---:|:---:|:---:|
| 8 | 43.5 ms | 5.43 ms |
| 16 | 70.1 ms | 4.38 ms |
| 32 | 130.4 ms | 4.07 ms |
| 64 | 273.8 ms | 4.28 ms |

**瓶颈是 ultralytics 的逐图 Python 后处理**（掩码上采样、Results 对象构造），
不是网络本身，因此几乎不随 batch 摊薄。仿真一步约 20ms，YOLO 在 64 env 时慢 14×。

**YOLO 显存**：额外约 2GB，与渲染器共用同一块 GPU。显存紧时先降 num_envs。

---

## 九、注意事项

**1. 建议从 stage 2 续训，不要从头训**  
Stage 3 的慢不适合从头训。Stage 2 已完成点云→动作的学习，stage 3 只需几百次
迭代让策略适应 YOLO 掩码分布（边缘偏胖、遮挡处缺块），成本低得多。

**2. 零阶保持**（§11.1）  
检测失败（YOLO 返回空掩码）时，`point_bridge_point_cloud_yolo` 会复用上一步的
点云。最多允许 30 步连续空掩码（`_MAX_EMPTY_STRIKES = 30`），超过后重置为均匀
采样的保底点云。策略永远不会收到空点集。

**3. GPU 全程不下显卡**  
Isaac 相机缓冲 → `rgb_to_yolo_input`（permute + /255）→ `model.predict()`（GPU
tensor 输入）→ masks.data（GPU tensor）→ 反投影，全程不调 `.cpu()` 或 numpy。

**4. conf 门限调整**  
`YOLO_CONF = 0.25` 与真机侧保持一致。调高会增加漏检（走零阶保持），调低会增加
误检。可以用 `--yolo_conf` 命令行参数覆盖，但要与真机侧同步修改。

**5. 换权重文件**  
用 `--yolo_weights /path/to/new.pt` 覆盖默认路径。权重会和其他配置文件一起存入
log 目录快照。

---

## 十、快速检查清单

训练前确认：

- [ ] `.yolo_deps` 目录存在（`ls /home/gxai/Desktop/CZR/.yolo_deps`）
- [ ] `20260815.pt` 存在（`ls /home/gxai/Desktop/CZR/20260815.pt`）
- [ ] stage 2 checkpoint 路径正确（`logs/xarm7_pick_pointcloud_stage2/.../model_15100.pt`）
- [ ] 磁盘剩余空间足够（每个 checkpoint ~700MB，建议 >20GB 余量）
- [ ] `--headless` 已加（无显示器环境必须）

