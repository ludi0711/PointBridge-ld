# M1 random_box 遮挡 —— 标准训练命令

对应脚本：`scripts/train/train_pick_pointcloud_yolo_occluded.py`
（YOLO 掩码路线 stage 3/4 + random_box 遮挡，设计见 [yolo_occlusion_m1_design.md](yolo_occlusion_m1_design.md)）。

以下命令直接 `python` 启动（本机 conda 环境已激活；装了 isaaclab.sh 的机器换成
`isaaclab.sh -p` 即可）。**日志一律存数据盘**：训练命令带
`--log_root /root/autodl-tmp/logs`（与 stage2/3/4 惯例一致，TensorBoard:6006
已经扫这个目录，新曲线直接进同一张图）。热启动源是 **stage 3/4 的 checkpoint**
（不是 stage 2 的 —— 掩码来源不同）。

---

## 1. 固定 severity（最常用）

```bash
# 从 stage 3 的 checkpoint 续训，固定删 50% 最近邻候选点
python scripts/train/train_pick_pointcloud_yolo_occluded.py \
    --num_envs 16 --log_root /root/autodl-tmp/logs \
    --checkpoint /root/autodl-tmp/logs/xarm7_pick_pointcloud_stage3/<run>/model_xxx.pt \
    --resume --max_iterations 500 --occlusion_severity 0.5 --headless

# 30% 遮挡
python scripts/train/train_pick_pointcloud_yolo_occluded.py \
    --num_envs 16 --log_root /root/autodl-tmp/logs \
    --checkpoint /root/autodl-tmp/logs/xarm7_pick_pointcloud_stage3/<run>/model_xxx.pt \
    --resume --max_iterations 500 --occlusion_severity 0.3 --headless
```

## 2. stage 4 基底（每 5 步跑一次 YOLO，省 4/5 的 YOLO 开销）

```bash
# 本机现存最新 stage4 基线：2026-09-23_19-31-41/model_10500.pt（= 10500 次迭代）
python scripts/train/train_pick_pointcloud_yolo_occluded.py \
    --num_envs 16 --log_root /root/autodl-tmp/logs --yolo_step 5 \
    --checkpoint /root/autodl-tmp/logs/xarm7_pick_pointcloud_stage4/2026-09-23_19-31-41/model_10500.pt \
    --resume --max_iterations 3000 --seed 0 --occlusion_severity 0.5 --headless
```

> 注意：`--yolo_step` 必须与热启动 checkpoint 的 stage 匹配 —— 给 `--yolo_step`
> 就走 stage 4（缓存掩码），不给就走 stage 3（每步跑 YOLO）。

> `--max_iterations` 在 `--resume` 时是**增量**：rsl_rl 里
> `tot_iter = 当前 checkpoint 的 iter + max_iterations`。上面从 model_10500 续
> 训、给 3000，实际训到 13500（每 100 步存一次 → model_10600 … model_15500）。
> 正式微调 3000 够（完整从零训默认预算 12000，微调取其 1/4）；曲线还在涨就再加。
>
> `--seed` 固定随机种子（不给会打 "Seed not set"、不保证可复现）。整组消融
> （对照 + 30/50/70 + 范围）用**同一个 seed** 才公平。**从零训**（不热启动）就
> 去掉 `--checkpoint --resume`，把 `--max_iterations` 改成 `12000`。

## 3. 每 episode 随机 severity（范围模式）

```bash
# 每 episode 在 20%..60% 内独立采样
python scripts/train/train_pick_pointcloud_yolo_occluded.py \
    --num_envs 16 --log_root /root/autodl-tmp/logs \
    --checkpoint /root/autodl-tmp/logs/xarm7_pick_pointcloud_stage3/<run>/model_xxx.pt \
    --resume --max_iterations 500 \
    --occlusion_severity_min 0.2 --occlusion_severity_max 0.6 --headless
```

`--occlusion_severity_min/max` 必须**成对给出**（只给一个报错），且满足
`0 <= min <= max <= 1`。此时 `--occlusion_severity` 被忽略。

## 4. 对照（无遮挡）

```bash
# severity=0：不构造遮挡谓词、不注册遮挡事件，与 stage 3/4 逐位一致（同 seed 可复现）
# stage 4 对照 —— 与 §2 的遮挡曲线同 checkpoint、同 seed、同迭代数，只差 severity
python scripts/train/train_pick_pointcloud_yolo_occluded.py \
    --num_envs 16 --log_root /root/autodl-tmp/logs --yolo_step 5 \
    --checkpoint /root/autodl-tmp/logs/xarm7_pick_pointcloud_stage4/2026-09-23_19-31-41/model_10500.pt \
    --resume --max_iterations 3000 --seed 0 --headless
```

对照的用途：证明"遮挡管线本身无副作用"——日志目录名带 `_occ_none`，可与
`xarm7_pick_pointcloud_stage4` 的曲线直接重叠对比。注意对照必须和遮挡曲线
**同 stage、同 seed、同 `--max_iterations`**，这样两条曲线的唯一差异就是 severity。

## 5. 回放看遮挡效果

> **⚠ 本容器没有显示服务器（无 DISPLAY，也没有 X/VNC）**：Isaac Lab 检测不到显示时
> 会自动给 Kit 加 `--no-window`，此模式下首次 reset 会长时间占满单核 CPU（实测
> 15 分钟以上仍无窗口、无输出）—— **play 在这里开不出窗口，不要用它验证遮挡**。
> 看遮挡效果请直接用 §5.1 的 headless 可视化脚本（十几秒出结果，帧/mp4/CSV 落盘）。

```bash
# 单 env 回放，固定 50% 遮挡（阴影形状每 episode 随机）
~/IsaacLab/isaaclab.sh -p scripts/train/train_pick_pointcloud_yolo_occluded.py \
    --play --num_envs 1 \
    --checkpoint logs/xarm7_pick_pointcloud_stage3_occ_rbox_50/<run>/model_xxx.pt \
    --occlusion_severity 0.5

# stage 4 + 范围遮挡的回放
~/IsaacLab/isaaclab.sh -p scripts/train/train_pick_pointcloud_yolo_occluded.py \
    --play --num_envs 1 --yolo_step 5 \
    --checkpoint logs/xarm7_pick_pointcloud_stage4_occ_rbox_r20-60/<run>/model_xxx.pt \
    --occlusion_severity_min 0.2 --occlusion_severity_max 0.6
```

### 5.1 遮挡点云标红可视化（推荐）

`tools/validate_pick_pointcloud_yolo_occlusion.py` —— **不改任何既有文件**，运行时
打开 env_cfg 里埋好的 `_OCCLUSION_VIS_ENABLED` 调试旁路（关闭时零开销、零行为差异），
把被挖掉的候选点/区域标红落盘：半透明红=遮挡区域整体形状，红点=被挖候选点，
绿点=存活候选点。

```bash
# 零动作：纯看阴影形状随工件位姿变化（遮挡与策略无关，不需要 checkpoint）
cd /root/gx-va/gx-VA-isaaclab_v1 && python tools/validate_pick_pointcloud_yolo_occlusion.py \
    --headless --num_envs 4 --steps 60 --occlusion_severity 0.5 \
    --out /root/autodl-tmp/logs/occ_vis_rbox50

# 加载 checkpoint 用训练策略走真实接近轨迹（看抓取过程中的遮挡变化）
python tools/validate_pick_pointcloud_yolo_occlusion.py \
    --headless --num_envs 4 --steps 60 --occlusion_severity 0.5 \
    --checkpoint /root/autodl-tmp/logs/xarm7_pick_pointcloud_stage3_occ_rbox_50/<run>/model_xxx.pt \
    --out /root/autodl-tmp/logs/occ_vis_rbox50

# stage 4 + 范围遮挡（--yolo_step 与要加载 checkpoint 的 stage 一致）
python tools/validate_pick_pointcloud_yolo_occlusion.py \
    --headless --num_envs 4 --steps 60 --yolo_step 5 \
    --occlusion_severity_min 0.2 --occlusion_severity_max 0.6 --out /root/autodl-tmp/logs/occ_vis_r2060
```

输出：`frames/frame_<step>_env<i>.png`（imageio 不可用时）+ `env_<i>.mp4` +
`occ_stats.csv`（每步每 env：被挖候选点比例、遮挡像素比例、YOLO 检出率）。
被挖候选点比例应 ≈ severity —— 这是遮挡生效的最直接数值校验。

## 6. 消融实验标准组合

| 曲线名（日志目录后缀） | 命令要点 |
|---|---|
| `stage3_occ_none` | 什么都不加（对照） |
| `stage3_occ_rbox_30` | `--occlusion_severity 0.3` |
| `stage3_occ_rbox_50` | `--occlusion_severity 0.5` |
| `stage3_occ_rbox_70` | `--occlusion_severity 0.7` |
| `stage3_occ_rbox_r20-60` | `--occlusion_severity_min 0.2 --occlusion_severity_max 0.6` |
| `stage4_occ_*` | 同上一行加 `--yolo_step 5`（对比缓存掩码叠加遮挡） |

同一条 TensorBoard 里对照 `visible_ratio`（遮挡生效它应下降，量级 ≈ severity ×
YOLO 覆盖比例）与 `yolo_detect_ratio`（遮挡在点级、不进 YOLO，应不受影响）。

## 7. 后台 nohup 训练（SSH 断开不中断）

正式跑建议 nohup 挂后台，断了 SSH 也不停。日志重定向到数据盘的 console log
（与 stage2 惯例一致），PID 记下来方便 kill：

```bash
cd /root/gx-va/gx-VA-isaaclab_v1 && nohup python scripts/train/train_pick_pointcloud_yolo_occluded.py \
    --num_envs 16 --log_root /root/autodl-tmp/logs --yolo_step 5 \
    --checkpoint /root/autodl-tmp/logs/xarm7_pick_pointcloud_stage4/2026-09-23_19-31-41/model_10500.pt \
    --resume --max_iterations 3000 --seed 0 \
    --occlusion_severity 0.5 --headless \
    > /root/autodl-tmp/logs/stage4_occ_rbox50_console.log 2>&1 &
echo $! > /root/autodl-tmp/logs/stage4_occ_rbox50.pid
```

- 看进度：`tail -f /root/autodl-tmp/logs/stage4_occ_rbox50_console.log`；曲线看
  TensorBoard:6006（日志本体落在
  `/root/autodl-tmp/logs/xarm7_pick_pointcloud_stage4_occ_rbox_50/<时间戳>/`）。
- 停掉：Omniverse 收到 SIGTERM 会卡住，直接 `kill -9 $(cat /root/autodl-tmp/logs/stage4_occ_rbox50.pid)`。
- 其余 severity 曲线：改 `--occlusion_severity 0.3/0.7`，范围模式改
  `--occlusion_severity_min 0.2 --occlusion_severity_max 0.6`；`--seed` 与
  `--max_iterations` 保持跟上面一致，只动 severity。

## 参数速查

| 参数 | 默认 | 说明 |
|---|---|---|
| `--occlusion_severity` | `0.0` | 固定删除比例 `[0,1]`；`0` = 对照 |
| `--occlusion_severity_min` / `--occlusion_severity_max` | None | 成对给出 → 每 episode `U[min,max]` |
| `--yolo_step` | None | 不给 = stage 3（每步跑 YOLO）；给 N = stage 4（每 N 步缓存掩码） |
| `--num_envs` | `16` | YOLO 约 4.3ms/env 且不随 batch 摊薄，开大了主要在等它 |
| `--checkpoint` + `--resume` | — | 热启动，checkpoint 必须是 stage 3/4 的 |
| `--log_root` | `<项目>/logs` | 日志根目录；本机一律 `/root/autodl-tmp/logs`（数据盘，TensorBoard:6006 已挂这个目录） |
| `--max_iterations` | 配置值（12000） | `--resume` 时是**增量**（加在 checkpoint 的 iter 上）；微调给 3000，从零训给 12000 |
| `--seed` | None | 固定随机种子；不给 = "Seed not set"、不保证可复现。整组消融用同一个 seed |

校验规则：`--occlusion_severity` 必须在 `[0,1]`；`--yolo_step >= 1`；
min/max 成对且 `0 <= min <= max <= 1`。违反任意一条直接报错退出。
