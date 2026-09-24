# xArm7 Pick 项目修改记录

项目路径：`/home/gxai/IsaacLab/czr/zjj/usd_test/`

---

## 版本全览

| 版本 | 配置文件 | MDP 文件 | 核心变化 |
|------|----------|----------|----------|
| v1 | `xarm7_pick_env_cfg_liftcube.py` | `xarm7_pick_mdp_liftcube.py` | 基础版本，夹爪二值控制，小动作幅度 |
| v2 | `xarm7_pick_env_cfg_liftcube_v2.py` | `xarm7_pick_mdp_liftcube_v2.py` | 夹爪连续控制，动作幅度 ±60°，EE 条件奖励 |
| v3 | `xarm7_pick_env_cfg_liftcube_v3.py` | 复用 v2 | 机械臂改增量控制，episode 延长到 5s |
| v4 | `xarm7_pick_env_cfg_liftcube_v4.py` | 复用 v2 | 观测精简到 22 维，回归绝对控制 |
| v5 | `xarm7_pick_env_cfg_liftcube_v5.py` | 复用 v2 | 成功条件重构，xyz<1cm，成功奖励 2000 |
| v6 | `xarm7_pick_env_cfg_liftcube_v6.py` | 复用 v2 | 机械臂增量控制 ±2°/步，episode 延长到 5s |

---

## v1 — 基础版本

**文件**：`xarm7_pick_env_cfg_liftcube.py` + `xarm7_pick_mdp_liftcube.py`

### 动作空间
| 关节 | 类型 | 范围 |
|------|------|------|
| joint1~7 | 绝对位置控制 | scale=0.2（约 ±11°） |
| drive_joint | **二值**控制（Binary） | 0（开） / 0.85（闭） |

### 观测空间（41 维）
`joint_pos(13) + joint_vel(13) + ee_pos(3) + obj_pos(3) + object_lifted(1) + actions(8)`

### 奖励配置
| 奖励项 | weight | 说明 |
|--------|--------|------|
| `reaching_object` | 10.0 | exp(-dist/0.3) |
| `ee_orientation` | 10.0 | 固定目标姿态 (0,0,1,0) |
| `lift_soft` | 30.0 | z>0.75 引导到 0.85m（仅 z，无 EE 条件） |
| `lift_hard` | 60.0 | z>0.75 且 xy<10cm（std=0.05，无 EE 条件） |
| `lift_success_bonus` | 200.0 | z>0.85 且 xy<3cm |
| `action_rate` | -1e-4 | 课程学习动态调整 |
| `joint_vel` | -1e-4 | 课程学习动态调整 |

### 终止条件
- `time_out`：超时（2s/100步）
- `object_dropping`：工件 z < 0.62m
- `lift_success`：z > 0.85m 且 xy_dist < 5cm

### 其他
- 有课程学习：`action_rate` / `joint_vel` 权重动态从 -1e-4 增大到 -1e-1
- ee_frame 开启 debug_vis（可视化坐标系）

---

## v2 — 夹爪连续控制 + EE 条件奖励

**文件**：`xarm7_pick_env_cfg_liftcube_v2.py` + `xarm7_pick_mdp_liftcube_v2.py`

### 相对 v1 的变化

#### 动作空间
| 关节 | v1 | v2 |
|------|----|----|
| joint1~7 | scale=0.2（±11°） | scale=1.047（**±60°**） |
| drive_joint | 二值（开/闭） | **连续** [0, 0.85]（scale=0.425, offset=0.425） |

#### 奖励变化
- `ee_orientation`：目标姿态从**固定** `(0,0,1,0)` 改为**动态**（工件当前姿态 × offset），适应任意朝向工件
- `lift_soft`：加入 EE 条件（ee_dist<4cm，ee_angle<10°），最低触发高度从 0.75m 降到 **0.73m**
- `lift_hard`：xy 阈值从 10cm **放宽到 30cm**，std 从 0.05 **提高到 0.1**，加入 EE 条件，启用 xyz 三维距离（`use_xyz_dist=True`）
- 去掉课程学习，`action_rate` / `joint_vel` 固定为 -1e-4

#### MDP 函数更新（v2 新增）
- `object_goal_distance`：加入 `ee_dist_threshold`、`ee_angle_threshold_deg`、`use_xyz_dist` 参数
- `object_reached_lift_height`：加入 `env.scene.env_origins` 修正（多环境 xy 坐标正确性）
- `gripper_close_near_object`：靠近工件时奖励夹爪闭合

### 奖励配置（v2 最终版）
| 奖励项 | weight | 说明 |
|--------|--------|------|
| `reaching_object` | 10.0 | exp(-dist/0.3) |
| `ee_orientation` | 10.0 | 姿态对齐（动态目标） |
| `lift_soft` | 30.0 | z>0.73 + EE<4cm + 角度<10°，引导到 0.85m（仅 z） |
| `lift_hard` | 60.0 | z>0.73 且 xy<30cm + EE<4cm + 角度<10°，xyz 距离 |
| `lift_success_bonus` | 200.0 | z>0.85 且 xy<3cm |
| `action_rate` | -1e-4 | 固定 |
| `joint_vel` | -1e-4 | 固定 |

---

## 2026-03-20 — v2 抖动优化

**文件**：`configs/xarm7_pick_env_cfg_liftcube_v2.py`

- `action_rate` weight：`-1e-4` → `-0.01`
- `joint_vel` weight：`-1e-4` → `-0.01`
- **原因**：策略可以夹取工件，但机械臂存在明显抖动，平滑惩罚权重过小
- **Checkpoint**：从 `model_3700.pt`（`2026-03-19_15-32-45`）继续训练

---

## v3 — 关节角增量控制

**文件**：`xarm7_pick_env_cfg_liftcube_v3.py`（MDP 复用 v2）

### 相对 v2 的变化

#### 动作空间
| 关节 | v2 | v3 |
|------|----|----|
| joint1~7 | 绝对位置，scale=1.047 | **增量控制**，每步最大 ±0.5°（≈ ±0.00873 rad） |
| drive_joint | 连续绝对 [0, 0.85] | 不变 |

#### 其他
- episode 时长从 2s（100步）延长到 **5s（250步）**，给增量控制足够步数完成任务
- `action_rate` / `joint_vel` weight 调整为 -1e-3
- 观测、奖励结构与 v2 相同

> **备注**：v3 采用增量控制，运动更平滑但收敛较慢，最终未成为主线版本。

---

## v4 — 观测精简

**文件**：`xarm7_pick_env_cfg_liftcube_v4.py`（MDP 复用 v2）

### 相对 v2 的变化

#### 观测空间（22 维，从 41 维精简）
| 去掉 | 保留 |
|------|------|
| `joint_vel`（13维） | `joint_pos`（只保留 joint1~7 + drive_joint，**8 维**） |
| `object_lifted`（1维） | `ee_pos`（3维） |
| | `obj_pos`（3维） |
| | `actions`（8维） |

- 动作恢复 v2 绝对控制（scale=1.047），不用 v3 增量控制
- episode 恢复 2s/100步
- 奖励配置与 v2 相同
- ee_frame 关闭 debug_vis

---

## 2026-03-22 — v6 机械臂增量控制

**新增文件**：
- `configs/xarm7_pick_env_cfg_liftcube_v6.py`
- `scripts/train_pick_liftcube_v6.py`

### 相对 v5 的变化

#### 动作空间
| 关节 | v5 | v6 |
|------|----|----|
| joint1~7 | 绝对位置，scale=1.047（±60°） | **增量控制**，每步最大 ±2°（≈ ±0.03491 rad） |
| drive_joint | 绝对位置连续 [0, 0.85] | 不变（夹爪保持绝对控制，增量夹爪累积误差难控） |

#### 其他
- episode 时长从 2s（100步）延长到 **5s（250步）**
  - ±2°/步 × 250步 = 最大 500° 行程，足够完成任务
- 奖励、观测、终止条件与 v5 完全相同

---

## 2026-03-22 — v5 成功条件重构

**新增文件**：
- `configs/xarm7_pick_env_cfg_liftcube_v5.py`
- `scripts/train_pick_liftcube_v5.py`

**MDP 新增函数**（`configs/xarm7_pick_mdp_liftcube_v2.py`）：
- `object_reached_lift_xyz`：xyz 三维距离 < 阈值 且 EE 贴近工件，返回 float(0/1)，兼容 RewTerm 和 DoneTerm

### 相对 v4 的变化

**问题根源**：
1. `lift_success_bonus`（RewTerm）与 `lift_success`（DoneTerm）判断逻辑几乎相同，在终止同一步触发，bonus 只起一步作用，意义有限
2. soft/hard 奖励不约束或宽松约束 xy，但终止条件要求 xy < 5cm，目标不一致，导致策略可能在 hard 奖励高但终止条件达不到的区域徘徊

**修改**：
- 去掉 `lift_success_bonus`（RewTerm w=200）
- 新增 `lift_success_reward`（RewTerm w=2000），与终止条件完全相同的判据，成功时给大奖励
- `lift_success` 终止条件从 `z>0.85 & xy<5cm` 改为 **`xyz_dist<3cm & EE_dist<4cm`**（与 lift_hard 目标一致）

### 奖励配置（v5）
| 奖励项 | weight | 说明 |
|--------|--------|------|
| `reaching_object` | 10.0 | exp(-dist/0.3) |
| `ee_orientation` | 10.0 | 姿态对齐（动态目标） |
| `lift_soft` | 30.0 | z>0.73 + EE<4cm + 角度<10°，引导到 0.85m（仅 z） |
| `lift_hard` | 60.0 | z>0.73 且 xy<30cm + EE<4cm + 角度<10°，xyz 距离 |
| `lift_success_reward` | 2000.0 | xyz<1cm 且 EE<4cm（一次性成功奖励） |
| `action_rate` | -1e-4 | 固定 |
| `joint_vel` | -1e-4 | 固定 |

### 终止条件（v5）
- `time_out`：超时（2s/100步）
- `object_dropping`：工件 z < 0.62m
- `lift_success`：**xyz_dist < 1cm 且 EE_dist < 4cm**
