# xArm7 Pick 任务 TensorBoard 曲线说明

> 适用 run：`logs/xarm7_pick_pointcloud_stage2_occ_*`（以及同源的 stage1/stage10 点云训练）。
> 数据源：`rsl_rl` 写入的 `events.out.tfevents.*`，共 25 条 scalar。

## 读图前的两个关键前提

1. **`Episode_Reward/*` 是"每个 episode 内该项奖励的累计和"**（跨 64 个环境取平均），
   不是瞬时值、也不是均值。所以量级是几百上千，不是 0~1。
2. **`Episode_Termination/*` 是"该原因触发终止的环境比例"**（0~1）。

---

## 1. Train/* —— 全局训练健康度（先看这里）

| 曲线 | 实际意义 | 健康判读 |
|---|---|---|
| `mean_reward` | 每 episode 总奖励（所有项累加）的滑动均值，**唯一综合指标** | 应单调上升后平台 |
| `mean_episode_length` | 每 episode 存活步数。满 1200 = 全程无碰撞、无提前终止 | 长期贴满 1200 |
| `mean_reward/time` | 同 `mean_reward`，横轴换成墙钟时间 | 看单位时间的收益斜率 |
| `mean_episode_length/time` | 同上 | — |

当前 run 收敛到 `mean_reward ≈ 5100`、`mean_episode_length ≈ 1200`，属健康水平。

---

## 2. Episode_Reward/* —— 奖励项逐项分解（定位问题用）

### 位置三档（`object_ee_distance`，`exp(-dist/std)`）

`dist` = EE 到"工件正上方 3cm 目标点"的距离。

| 项 | weight | std | 满值时 dist | 收敛值 |
|---|---|---|---|---|
| `reaching_coarse` | 35 | 0.50 | 远距离粗引导 | 33.4/35（饱和） |
| `reaching_mid` | 55 | 0.15 | ~15cm 内精修 | 48.8/55 |
| `reaching_fine` | 80 | 0.04 | 4cm 内 | 60.1/80 → **残余 ~1.15cm** |

### 姿态三档（`ee_orientation_alignment`，`exp(-angle/std)`）

`angle` = EE 与工件姿态的对称角误差（考虑 180° 绕 Z 抓取对称）。

| 项 | weight | std | 满值时角误差 | 收敛值 |
|---|---|---|---|---|
| `ee_orientation_coarse` | 30 | 0.50 | 粗对齐 | 21.9/30 |
| `ee_orientation_mid` | 40 | 0.15 | ~8.6° 内 | 20.9/40 |
| `ee_orientation_fine` | 70 | 0.08 | 4.6° 内 | 27.5/70 → **残余 ~4.3°** |

> 反推公式：`angle = -std · ln(reward/weight)`。fine 停在 27.5/70 = 39%，即 EE 姿态角误差
> 卡在 ~4.3°，恰好在 std=0.08 rad(4.6°) 边缘、未饱和 —— 这是姿态抖动/未精确对齐的直接信号。

### 辅助项

| 项 | 实际意义 | 收敛值 |
|---|---|---|
| `reach_success_bonus` | 二元成功奖励（阈值 1cm + 3°）。**当前 weight=0，恒为 0**，是预留开关 | 0 |
| `action_rate` | `-10 × ‖a_t − a_{t-1}‖²`，惩罚动作突变 | -0.43（动作已平滑） |
| `collision_penalty` | 夹爪接触力 > 0.5N 时 −1000 | 0（无碰撞） |
| `lock_joint_step_end` | **恒 0 的副作用项**：只在算奖励前把关节锁回缓存 q_target，本身返回 0，不改总奖励 | 0 |

---

## 3. Episode_Termination/* —— 终止原因占比（0~1）

| 曲线 | 意义 | 健康判读 |
|---|---|---|
| `lock_joint_step_end` | 副作用项，恒 0（同上） | 忽略 |
| `collision` | 夹爪碰撞（接触力>0.5N）导致**提前终止** | ≈0 |
| `time_out` | 到达最大 1200 步超时终止 | ≈1 |

若 `collision` 上升、`time_out` 下降，说明策略在碰东西。

---

## 4. Loss/* —— PPO 训练损失

### `value_function`（要往 0 压的"真误差"）

Critic 价值估计的均方误差，**衡量 V 函数准不准**。越小越准。
- 当前 stage2 点云 run 收敛到 ~6.5 震荡（特权 base 能到 0.15），原因是点云策略噪声大、状态部分可观测。

### `surrogate`（不是误差，是"策略改进幅度"）

PPO 截断代理目标（取负号后的 loss 形式），核心代码：

```python
ratio            = exp(new_log_prob - old_log_prob)      # π_new / π_old
surrogate        = -A * ratio
surrogate_clipped = -A * clamp(ratio, 0.8, 1.2)          # clip_param=0.2
surrogate_loss   = max(surrogate, surrogate_clipped).mean()
```

- **ratio**：新/旧策略对"当时实际动作"的概率比。1=不变，1.2=提了 20%，0.8=降了 20%。
- **A（advantage）**：GAE 优势（gamma=0.98, lam=0.95），已归一化到均值 0 / 标准差 1。
- **A·ratio**：策略梯度目标，最大化它。A>0 推 ratio 上，A<0 推 ratio 下。
- **裁剪 [0.8, 1.2]**：PPO 的"近端约束"，把单次更新对任一动作概率的改动封顶在 ±20%，
  取 `min(A·ratio, A·clip)` 避免带噪单轨迹过度押注。
- **负号 + max**：rsl_rl 把"最大化目标"翻成"最小化 loss"，所以 `max(-A·ratio, -A·clip)`。

**怎么读**：越负 = 本次更新改进越狠；它是每迭代重算的移动目标，会波动，不追求归零。
它和 `value_function` 是一对：value 看 critic 准不准，surrogate 看策略还在不在改。
收敛时 advantage→0，surrogate 落进小振幅震荡。

### `entropy`

策略熵（熵系数 entropy_coef=0.006 加权），**探索度**。太高=乱，太低=早熟；应随训练缓慢下降。

### `learning_rate`

自适应 LR（`schedule="adaptive"`，按 desired_kl=0.01 调：KL 超 2×目标降 1/1.5，低于目标一半升 1.5×，
初始 1e-3）。正常应从 1e-3 缓慢衰减；长期震荡说明 KL 不稳。

> 总损失 = `surrogate + value_loss_coef·value_loss − entropy_coef·entropy`，当前 value_loss_coef=0.5。

---

## 5. Policy/mean_noise_std

高斯策略动作噪声的标准差均值，反映"还在探索还是已收敛"。训练后期应随 entropy 一起缩小（动作趋于确定）。

---

## 6. Perf/* —— 性能（不直接影响策略好坏）

| 曲线 | 意义 |
|---|---|
| `total_fps` | 总吞吐（steps/秒） |
| `collection time` | rollout 采样耗时 |
| `learning_time` | 策略更新耗时 |

`collection time` 远大于 `learning_time` 是正常的（仿真采样是瓶颈）。stage2 相机 + 点云反投影会拖慢 collection。掉 fps 只影响训练速度，不影响最终质量。

---

## 一句话判读顺序

1. 看 `mean_reward` 涨没涨、`mean_episode_length` 有没有贴 1200。
2. 拆 `Episode_Reward/*`，找哪一项没饱和（如 `ee_orientation_fine` 停在 27/70 → 姿态没对齐）。
3. 怀疑价值估计时才看 `value_function`；想判断策略是否还在进步看 `surrogate` 趋势。
