# stage2 遮挡实验 —— 训练 / 查看命令（直接复制）

> 脚本：`scripts/train/train_pick_pointcloud_occluded.py`（stage2 遮挡版，不改原训练脚本）
> python：`/root/gx-va/bin/python`　日志根：`/root/autodl-tmp/logs`

---

## 一、96G 卡（主实验：512 envs + 2 minibatch）

四个遮挡梯度，只改 `--occlusion_mode` / `--occlusion_severity`，其余参数完全一致（保证曲线可比）。

### ① baseline（无遮挡）
/root/gx-va/gx-VA-isaaclab_v1/scripts/train/train_pick_pointcloud_occluded.py
```bash
nohup /root/gx-va/bin/python \
  /root/gx-va/gx-VA-isaaclab_v1/scripts/train/train_pick_pointcloud_occluded.py \
  --headless \
  --num_envs 512 --num_mini_batches 4 --seed 300 \
  --occlusion_mode none --occlusion_severity 0.0 \
  --max_iterations 12000 \
  --log_root /root/autodl-tmp/logs \
  > /root/autodl-tmp/logs/occ_none_v1_512mb4_console.log 2>&1 &
```

### ② 25% 半空间遮挡（切掉工件 +x 一侧 25%）

```bash
nohup /root/gx-va/bin/python \
  /root/gx-va/gx-VA-isaaclab_v1/scripts/train/train_pick_pointcloud_occluded.py \
  --headless \
  --num_envs 512 --num_mini_batches 2 --seed 300 \
  --occlusion_mode halfspace --occlusion_severity 0.25 --occlusion_axis x \
  --max_iterations 12000 \
  --log_root /root/autodl-tmp/logs \
  > /root/autodl-tmp/logs/occ_halfspace25_512mb2_console.log 2>&1 &
```

### ③ 50% 半空间遮挡

```bash
nohup /root/gx-va/bin/python \
  /root/gx-va/gx-VA-isaaclab_v1/scripts/train/train_pick_pointcloud_occluded.py \
  --headless \
  --num_envs 512 --num_mini_batches 2 --seed 300 \
  --occlusion_mode halfspace --occlusion_severity 0.50 --occlusion_axis x \
  --max_iterations 12000 \
  --log_root /root/autodl-tmp/logs \
  > /root/autodl-tmp/logs/occ_halfspace50_512mb2_console.log 2>&1 &
```

### ④ 75% 半空间遮挡

```bash
nohup /root/gx-va/bin/python \
  /root/gx-va/gx-VA-isaaclab_v1/scripts/train/train_pick_pointcloud_occluded.py \
  --headless \
  --num_envs 512 --num_mini_batches 2 --seed 300 \
  --occlusion_mode halfspace --occlusion_severity 0.75 --occlusion_axis x \
  --max_iterations 12000 \
  --log_root /root/autodl-tmp/logs \
  > /root/autodl-tmp/logs/occ_halfspace75_512mb2_console.log 2>&1 &
```

> 想用球形遮挡，把 `--occlusion_mode halfspace --occlusion_axis x` 换成
> `--occlusion_mode sphere --occlusion_center 0.5,0.5,0.5` 即可。

### ⑤ 随机位置遮挡（圆形 / 方形，解决"固定方向学出偏置"）

遮挡区**每 episode 重采位置**（方形还重采朝向），模型无法记住"工件某侧永远缺失"。
`severity` 语义：圆形=球半径/外接球（球心每 episode 在工件包围盒内随机）；
方形=最近邻删除比例（盒心每 episode 从面向相机的那侧表面点随机抽，朝向随机
SO(3) 作为椭球主轴 —— 删掉的必是一块相机可见面的连通阴影，比例严格等于
severity）。

```bash
# 圆形：球心每 episode 在工件包围盒内随机
nohup /root/gx-va/bin/python \
  /root/gx-va/gx-VA-isaaclab_v1/scripts/train/train_pick_pointcloud_occluded.py \
  --headless \
  --num_envs 512 --num_mini_batches 2 --seed 300 \
  --occlusion_mode random_sphere --occlusion_severity 0.25 \
  --max_iterations 12000 \
  --log_root /root/autodl-tmp/logs \
  > /root/autodl-tmp/logs/occ_randsphere25_512mb2_console.log 2>&1 &
```

```bash
# 方形：盒心（相机侧表面）+ 朝向（随机 SO(3) 椭球主轴）每 episode 随机，按最近邻删 severity 比例
nohup /root/gx-va/bin/python \
  /root/gx-va/gx-VA-isaaclab_v1/scripts/train/train_pick_pointcloud_occluded.py \
  --headless \
  --num_envs 512 --num_mini_batches 2 --seed 300 \
  --occlusion_mode random_box --occlusion_severity 0.25 \
  --max_iterations 12000 \
  --log_root /root/autodl-tmp/logs \
  > /root/autodl-tmp/logs/occ_randbox25_512mb2_console.log 2>&1 &
```

> 换比例只改 `--occlusion_severity`；`--occlusion_axis` / `--occlusion_center` 对
> `random_*` 不生效（会被忽略）。run 目录名带 `random_sphere_<pct>` / `random_box_<pct>`。

### ⑥ 冒烟测试（先确认链路不崩，再上 512 env）

```bash
/root/gx-va/bin/python \
  /root/gx-va/gx-VA-isaaclab_v1/scripts/train/train_pick_pointcloud_occluded.py \
  --headless --num_envs 8 --num_mini_batches 4 --seed 300 \
  --occlusion_mode random_box --occlusion_severity 0.25 \
  --max_iterations 3 \
  --log_root /root/autodl-tmp/logs
```

跑 3 个 iteration 正常结束即链路 OK。rsl_rl 每次训练结束都会存一个
`model_<最后迭代>.pt`（3 次迭代约是 `model_2.pt`），供下面画面验证用。

### ⑦ 画面验证（看遮挡是不是"随机一大块"）

冒烟拿到 checkpoint 后，用验证脚本录一段**带遮挡标注**的视频：

```bash
/root/gx-va/bin/python \
  /root/gx-va/gx-VA-isaaclab_v1/tools/validate_pick_pointcloud_stage1.py \
  --stage 2 --occlusion_mode random_box --occlusion_severity 0.25 \
  --debug_vis_occlusion \
  --episodes 3 \
  --checkpoint /root/autodl-tmp/logs/xarm7_pick_pointcloud_stage2_occ_random_box_25/<run>/model_2.pt
```

- `--debug_vis_occlusion`：录像里半透明红=遮挡区域，红点=被挖候选点，绿点=存活候选点。
- 圆形换成 `--occlusion_mode random_sphere`。
- 输出在 `/root/autodl-tmp/validation/.../rollout.mp4`（`--out_dir` 可指定）。
- `<run>` 换成实际时间戳目录：`ls -lt /root/autodl-tmp/logs/xarm7_pick_pointcloud_stage2_occ_random_box_25/`

---

## 二、32G 卡（备份 / 冒烟：128 envs + 2 minibatch）

```bash
nohup /root/gx-va/bin/python \
  /root/gx-va/gx-VA-isaaclab_v1/scripts/train/train_pick_pointcloud_occluded.py \
  --headless \
  --num_envs 128 --num_mini_batches 2 --seed 300 \
  --occlusion_mode none --occlusion_severity 0.0 \
  --max_iterations 12000 \
  --log_root /root/autodl-tmp/logs \
  > /root/autodl-tmp/logs/occ_none_128mb2_console.log 2>&1 &
```

> 要跑遮挡，同样只改 `--occlusion_mode` / `--occlusion_severity` 两个参数。
> ⚠️ 32G 卡和 96G 卡的曲线（num_envs 不同）不能混比，主实验统一放 96G 卡。

---

## 三、TensorBoard（6008 = 32G 卡 / 6006 = 96G 卡）

```bash
# 32G 卡（6008，一般已常驻）
nohup tensorboard --logdir /root/autodl-tmp/logs --host 0.0.0.0 --port 6008 \
  > /root/autodl-tmp/logs/tensorboard_6008.log 2>&1 &

# 96G 卡（6006）
nohup tensorboard --logdir /root/autodl-tmp/logs --host 0.0.0.0 --port 6006 \
  > /root/autodl-tmp/logs/tensorboard_6006.log 2>&1 &
```

浏览器访问 `http://<服务器公网IP>:6008` 或 `:6006`。`--logdir` 指向日志根目录，
所有遮挡 run（none/25/50/75）并进同一张图对比。

---

## 四、查看运行状态

```bash
# 实时看控制台输出
tail -f /root/autodl-tmp/logs/occ_none_512mb2_console.log

# 已跑多少 iteration
grep -c "Mean reward" /root/autodl-tmp/logs/occ_none_512mb2_console.log

# 看显存（每 2 秒刷新）
watch -n 2 nvidia-smi

# 确认训练进程在跑
ps aux | grep train_pick_pointcloud_occluded | grep -v grep
```

---

## 五、查看配置文件 / 代码

```bash
# ① 训练脚本全部 CLI 参数（一行一个）
grep -n "add_argument" /root/gx-va/gx-VA-isaaclab_v1/scripts/train/train_pick_pointcloud_occluded.py
```

```bash
# ② 遮挡几何定义（halfspace / sphere / severity 语义）
cat /root/gx-va/gx-VA-isaaclab_v1/configs/occlusion.py
```

```bash
# ③ PPO 超参（步数 / minibatch / 学习率 / 迭代数等）
grep -nE "num_steps_per_env|num_mini_batches|num_learning_epochs|learning_rate|max_iterations|clip_param|gamma|lam|entropy_coef" \
  /root/gx-va/gx-VA-isaaclab_v1/configs/agents/rsl_rl_ppo_cfg.py
```

```bash
# ④ 点云桥关键函数（反投影 / 遮挡注入点 / FPS）
grep -nE "def |occlusion_keep_fn|mask_depth_to_pointcloud|farthest_point_sample" \
  /root/gx-va/gx-VA-isaaclab_v1/configs/point_bridge_pointcloud.py
```

```bash
# ⑤ 环境配置关键段（观测 / 遮挡开关 / success 终止 / 抓取目标）
grep -nE "def |set_occlusion_stage2|point_bridge_point_cloud_occluded|set_success_termination|set_grasp_target_offset" \
  /root/gx-va/gx-VA-isaaclab_v1/configs/xarm7_pick_pointcloud_env_cfg.py
```

```bash
# ⑥ 某次 run 实际用的配置快照（脚本已自动把 config 拷进 log 目录）
ls -lt /root/autodl-tmp/logs/xarm7_pick_pointcloud_stage2_occ_none/
```

> 直接看整个文件：`cat <文件路径>` 或 `less <文件路径>`。

---

## 六、其他常用命令

```bash
# 停训练（先确认只有它在跑）
pkill -f train_pick_pointcloud_occluded

# 停 tensorboard
pkill -f tensorboard

# 检查端口是否被占（无输出 = 可用）
ss -tlnp | grep 6006
ss -tlnp | grep 6008

# 列出日志目录（按时间找最新 run）
ls -lt --time-style=+%m-%d_%H:%M /root/autodl-tmp/logs
```

---

## 七、参数速查

| 参数 | 默认 | 说明 |
|---|---|---|
| `--num_envs` | 64 | 并行环境数（96G 卡用 512，32G 卡用 128） |
| `--num_mini_batches` | 4 | minibatch 数（本实验固定 2） |
| `--num_steps_per_env` | 24 | rollout 步数，**保持 24 别动** |
| `--seed` | 配置值 | 本实验固定 300 |
| `--max_iterations` | 12000 | 训练迭代数 |
| `--occlusion_mode` | none | `none` / `halfspace` / `sphere` / `random_sphere` / `random_box` |
| `--occlusion_severity` | 0.0 | 0~1，遮挡程度：`halfspace`=切除比例，`sphere`/`random_sphere`=球半径/外接球，`random_box`=最近邻删除比例（`random_*` 下每 episode 位置/朝向随机） |
| `--occlusion_axis` | x | halfspace 用：`x`/`-x`/`y`/`-y`/`z`/`-z` |
| `--occlusion_center` | 0.5,0.5,0.5 | sphere 用：球心局部归一化分数（`random_*` 忽略） |
| `--enable_success_termination` | 关 | 开 reach_success 提前 done |
| `--grasp_offset_cm` | 3.0 | 抓取目标抬到工件上方 cm |
| `--log_root` | <项目>/logs | 日志根目录 |
| `--headless` | — | 无显示运行，必加 |

**minibatch = num_envs × num_steps_per_env / num_mini_batches**
- 96G 卡：512 × 24 / 2 = **6144** ✓
- 32G 卡：128 × 24 / 2 = 1536

---

## 八、关键教训（别踩）

1. **`--num_steps_per_env` 保持 24**。实测 96 会让「每单位数据更新次数 ÷4」，训练卡在
   "乱撞手臂"局部最优出不来（96 万步时 reward 还 -194、collision 95%）。
2. **"全碰撞退出 / 超时≈0" 不是收敛**。早期 action noise 还高，agent 乱挥手臂就会撞；
   有没有学会要看 `reaching_fine` / `reach_success_bonus`，不是 collision。
3. **整组实验同卡同配置**，只改 `--occlusion_mode` / `--occlusion_severity`。
4. 96G 卡代码没 `--num_steps_per_env` 开关、32G 卡有——但两边都用默认 24，所以**命令里都别带这个参数**。

---

## 九、仿真验证（画面验证 + 批量统计误差）

> 以下命令默认在 `/root/gx-va/gx-VA-isaaclab_new` 目录下执行，python 统一用
> `/root/gx-va/bin/python`（本机 isaaclab 环境）。训练脚本也在本树：
> `scripts/train/train_pick_pointcloud_occluded.py`。

训练拿到 checkpoint 后，仿真侧有两层验证：

| 工具 | 作用 | 出什么 |
|---|---|---|
| `tools/validate_pick_pointcloud_stage1.py` | 单环境回放策略、录视频、可叠加遮挡可视化 | `rollout.mp4`（或 PNG 帧） |
| `tools/eval_pointcloud_compare.py` | 外层调度：并行跑两版 checkpoint 逐集结算误差 | 逐集 CSV + 统计报表 + 图 |

### ① 画面验证（录一段带遮挡标注的视频）

stage 2 的 `--occlusion_*` **必须与训练该 checkpoint 时一致**（决定哪些点被挖掉，观测分布才对齐）。
baseline 时 `--occlusion_mode none`（内部走同一观测函数，只是不注入过滤）。

```bash
# stage 2 + 遮挡，显式指定 checkpoint，录 3 集 + 遮挡可视化
/root/gx-va/bin/python tools/validate_pick_pointcloud_stage1.py \
  --stage 2 \
  --checkpoint /root/autodl-tmp/logs/xarm7_pick_pointcloud_stage2_occ_random_box_25/2026-09-01_15-34-14/model_6600.pt \
  --occlusion_mode random_box --occlusion_severity 0.25 \
  --debug_vis_occlusion \
  --episodes 3 --video_fps 25
```

```bash
# 只给 --run_dir 自动取该 run 的最终 checkpoint；都不给则交互式选 run + checkpoint
/root/gx-va/bin/python tools/validate_pick_pointcloud_stage1.py \
  --stage 2 --run_dir 2026-09-01_15-34-14 \
  --occlusion_mode random_box --occlusion_severity 0.25 --debug_vis_occlusion
```

- `--debug_vis_occlusion`：半透明红=遮挡形状，红点=被挖候选点，绿点=存活候选点。
- 输出默认在 `/root/autodl-tmp/validation/xarm7_pick_pointcloud_stage2_<run>_<ckpt>_<时间戳>/rollout.mp4`（`--out_dir` 可指定）。
- `--seed` 只决定评测时工件随机摆放序列，与训练无需一致（权重从 checkpoint 确定性加载）。

### ② 批量并行统计误差（A/B 两版对比，配对/非配对自适应）

`eval_pointcloud_compare.py` 只做调度（普通 python 即可），内部用
`CONDA_PREFIX=/root/gx-va /root/IsaacLab/isaaclab.sh -p` 起 worker；跑完自动调 stats 出报表+图。

```bash
# 默认：遮挡训练版(random_box 25%) vs baseline(无遮挡)，50 envs × 1000 集，严格配对
/root/gx-va/bin/python tools/eval_pointcloud_compare.py \
  --num_envs 50 --total_episodes 1000 --seed 0
```

```bash
# 只跑单版（出单版统计），或换 checkpoint / 遮挡参数
/root/gx-va/bin/python tools/eval_pointcloud_compare.py \
  --only occ \
  --occ_ckpt /root/autodl-tmp/logs/xarm7_pick_pointcloud_stage2_occ_random_box_25/2026-09-01_15-34-14/model_6600.pt \
  --occlusion_mode random_box --occlusion_severity 0.25 \
  --num_envs 50 --total_episodes 1000 --seed 0
```

关键参数（两版必须完全一致，RNG 流才同构）：

- `--num_envs` / `--total_episodes` / `--seed`：并行环境数、每版总集数、种子。
- `--occlusion_mode` / `--occlusion_severity`：评估时注入的遮挡，须与训练该 checkpoint 时一致。
- `--lockstep`（默认开）：关闭碰撞终止、所有集跑满 1200 步超时 → 两版严格逐集配对；
  `--no-lockstep` 退回「碰撞即终止」的分布对齐（碰撞率改用 ever_collided）。
- `--only {occ,base,both}`：只跑单版还是两版对比；`--no_stats`：只出原始 CSV 不出报表/图。
- `--occ_ckpt` / `--base_ckpt`：两个 checkpoint（默认已在脚本里写死，本机都存在）。

统计口径（worker 落盘 CSV 列）：距离量到【抓取目标点】`grasp_target_pos_w`（非物体中心），
姿态误差=180° 夹持对称角；成功三档 `reach_1cm_3deg` / `reach_0p5cm_3deg` / `reach_pos_1cm`
+ `ever_lifted`；误差 `dist_best_cm` / `ori_best_deg`（最优）与 `dist_final_cm` / `ori_final_deg`（终止帧）。

输出（默认 `logs/eval_pointcloud/`）：

- 每版一份逐集 CSV + `<tag>.meta.json`；
- `*_stats/report.txt`（第 1 块单版指标 + Wilson 95% CI；第 2 块 Δ=A−B）、`summary.csv`、
  `stats.json`、`figs/*.png`（成功率条形图 / 误差箱线 / 误差 CDF / 成功散点）。

配对 vs 非配对自动判定：按存下的 `obj_init_pos` 对齐率 ≥99% → 配对（McNemar + 配对 bootstrap），
否则非配对（两样本 bootstrap + Cliff's delta）。

### ③ 手动跑 worker / stats（一般不用，compare 已自动调度）

```bash
# 手动起一个 worker（逐集 CSV）
CONDA_PREFIX=/root/gx-va /root/IsaacLab/isaaclab.sh -p tools/eval_pointcloud_worker.py \
  --checkpoint /root/autodl-tmp/logs/xarm7_pick_pointcloud_stage2_occ_random_box_25/2026-09-01_15-34-14/model_6600.pt \
  --occlusion_mode random_box --occlusion_severity 0.25 \
  --num_envs 50 --total_episodes 1000 --seed 0 \
  --tag occ25 --out /root/autodl-tmp/eval_occ/occ25.csv
```

```bash
# 手动对两份 CSV 出统计（配对/非配对自适应）
/root/gx-va/bin/python tools/stats_pointcloud_occlusion.py \
  --csv_a <A.csv> --label_a "遮挡训练(25%)" \
  --csv_b <B.csv> --label_b "baseline(无遮挡)" \
  --out_dir <out_stats_dir>
```

---

## 十、stage4（YOLO 掩码）训练 —— 与 stage2 baseline 参数对齐

> 脚本：`scripts/train/train_pick_pointcloud_yolo.py`
> python：`/root/gx-va/bin/python`　日志根：`/root/autodl-tmp/logs`
> 续训源：stage2 baseline = **occ_none**（`--occlusion_mode none`）：
> `/root/autodl-tmp/logs/xarm7_pick_pointcloud_stage2_occ_none/2026-09-16_14-32-26/`

**与 stage2 对齐的参数**（多数走默认，命令里不用写）：
`--grasp_offset_cm 3.0`（已从 occ_none 控制台确认 `[Target] ... 3.0 cm`）·
`--enable_success_termination` 关 · `--no_joint_vel` 不传（critic 35 维）·
`num_steps_per_env=24` · `num_mini_batches=4`（config 默认，脚本不暴露）·
`--seed 300` · `--log_root /root/autodl-tmp/logs` · `--headless`

**必须发散的一个：`--num_envs`**。stage2 用 512，但 YOLO 约 4.3ms/env **不随 batch
摊薄**，512 env = 每步 ~2.2s 全在等 YOLO。降到 **16**（脚本默认）。minibatch 会从
3072 缩到 96（16×24/4），但这是 finetune（几百迭代），不是从头训，可接受。

### ① 冒烟（先确认 YOLO 链路不崩，再上正式）

```bash
/root/gx-va/bin/python \
  /root/gx-va/gx-VA-isaaclab_v1/scripts/train/train_pick_pointcloud_yolo.py \
  --headless --num_envs 8 --seed 300 \
  --yolo_step 5 --max_iterations 3 \
  --checkpoint /root/autodl-tmp/logs/xarm7_pick_pointcloud_stage2_occ_none/2026-09-16_14-32-26/model_8800.pt \
  --resume \
  --log_root /root/autodl-tmp/logs
```

跑 3 个 iteration 正常结束、控制台头部打印 `[Stage] 4 ... 每 5 步跑一次` 即链路 OK。

### ② 正式 finetune（stage4 = 加 `--yolo_step 5`）

```bash
nohup /root/gx-va/bin/python \
  /root/gx-va/gx-VA-isaaclab_v1/scripts/train/train_pick_pointcloud_yolo.py \
  --headless \
  --num_envs 16 --seed 300 \
  --yolo_step 5 --max_iterations 500 \
  --checkpoint /root/autodl-tmp/logs/xarm7_pick_pointcloud_stage2_occ_none/2026-09-16_14-32-26/model_8800.pt \
  --resume \
  --log_root /root/autodl-tmp/logs \
  > /root/autodl-tmp/logs/stage4_yolo5_16env_console.log 2>&1 &
```

> `--max_iterations 500`（不是 12000）：stage2 已学会「点云→动作」，这里只需适应
> YOLO 掩码分布，几百迭代通常就够。
> `--checkpoint` 换 stage2 baseline 最终权重：等 occ_none 跑到 `model_12000.pt` 后用
> 它（现在最新是 `model_8800.pt`，仍在训练）。
> 日志落在 `/root/autodl-tmp/logs/xarm7_pick_pointcloud_stage4/<时间戳>/`。

### ③ 回放验证

```bash
/root/gx-va/bin/python \
  /root/gx-va/gx-VA-isaaclab_v1/scripts/train/train_pick_pointcloud_yolo.py \
  --play --num_envs 1 --yolo_step 5 \
  --checkpoint /root/autodl-tmp/logs/xarm7_pick_pointcloud_stage4/<run>/model_500.pt
```

### stage3 vs stage4（只差一个参数）

- 不给 `--yolo_step` → stage3：每步都跑 YOLO（50Hz，贵，不符合真机节奏）。
- 给 `--yolo_step 5` → stage4：每 5 步跑一次 YOLO、缓存掩码（深度每步新鲜），
  正好 = 真机「50Hz 控制 + 10Hz YOLO」。**部署训练用 stage4**。
