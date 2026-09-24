# DeGuV → Isaac Lab 迁移规划

> 路线：**移植整套 DrQ-v2（重）** + **单固定相机（12通道）** + **舍弃 Theia 改用 DeGuV 的 CNN 编码器**
>
> 目标：在 `gx-VA-isaaclab` 的 xArm7 spatial-camfix 环境上，跑通 DeGuV 的深度引导掩码视觉 RL（含泛化评估）。

---

## 0. 核心定位

迁移后，`xarm7_pick_vision_spatial_camfix_env_cfg.py` **从 PPO 训练环境降级为纯仿真器**：
- 只负责出图（RGB+深度）、执行 7 维关节动作、给 reward / done。
- **不再经过 Theia**，观测改为相机原始 `(12, H, W)` uint8（3帧 × (RGB 3 + 深度 1)）。
- DeGuV 的算法、训练循环、replay buffer、泛化评估全部从 `DeGuV-main/` 搬过来。

工作量分布：**90% 在写 Isaac→dm_env 的 wrapper**，其余基本是搬运 + 剥依赖。

---

## 1. 三个核心接口矛盾（迁移的全部难点）

| 接口 | DeGuV 要的 | Isaac 给的 | 适配 |
|------|-----------|-----------|------|
| 环境 API | dm_env：`reset()/step()→TimeStep`，`observation_spec/action_spec` | gym 风格 `ManagerBasedRLEnv`，`(obs,rew,done,...)` | wrapper 转 dm_env |
| 并行度 | 单环境 | 向量化（默认 64） | wrapper 强制 `num_envs=1`，取 `[0]` |
| 观测 | `(12,H,W)` uint8：3帧×(RGB+深度) | 1166 维 Theia 向量 | 绕过 Theia，直接读 `camera.data.output` |
| 深度 | RGB+深度对齐 | 固定相机**只有 RGB** | env_cfg 给 `camera_fixed` 加 `distance_to_camera` |
| 动作 | dm_env 连续 `[-1,1]` | 7维关节 relative action | wrapper 转 shape/范围 |

---

## 2. 文件清单

### 2.1 从 DeGuV-main 搬过来（基本原样）
- `algos/deguv.py`、`algos/info_nce.py`、`algos/drqv2.py`
- `train.py`、`replay_buffer.py`、`utils.py`、`logger.py`、`video.py`、`rl_utils.py`
- `cfgs/deguv_config.yaml`（改任务参数）
- `assets/`（random_overlay 用的 Places 背景图——域随机化必需）

### 2.2 全新写（最核心）
- **`wrappers/isaac_deguv_wrapper.py`** —— 把 `XArm7PickLiftCubePlayEnvCfg` 包成 dm_env：
  - 强制单环境
  - 出 `(12,H,W)`：固定相机 RGB(3) + 深度(1)，frame-stack=3 → 12 通道
  - 深度归一化到 uint8（DeGuV Encoder 假设输入 /255）
  - 动作 `(7,)` ↔ Isaac `(1,7)` 转换
  - `done/reward/discount` 透传，组装 `ExtendedTimeStep`
  - `observation_spec()`→`BoundedArray((12,H,W),uint8)`，`action_spec()`→7维 `[-1,1]`

### 2.3 改 env_cfg（小改，且要可逆/可开关）
- `camera_fixed.data_types`：`["rgb"]` → `["rgb", "distance_to_camera"]`
- Theia 观测项不再使用（wrapper 直接读相机，不走 ObservationManager 的 1166 维）。
  - **不要删 Theia 代码**——加开关 / 新建一个 minimal 观测配置类，保留 PPO 版可用。

---

## 3. 关键技术风险（按严重度）

1. **🔴 事件循环嵌套**：Isaac `simulation_app` 必须主线程持续 step；DeGuV `Workspace.train()` 是 hydra 主循环 + replay loader 多进程。两套循环嫁接是最易卡死/崩的点。
   - 缓解：wrapper 内同步驱动 Isaac step；replay loader 进程数先设 0/1 调试。
2. **🟠 速度**：单环境 + 每帧 render RGB+深度 + replay 落盘，比现在 64-env PPO 慢一个量级。1M 帧可能很久。
   - 缓解：先小规模（num_train_frames 调小）跑通，再考虑 GPU 端 replay。
3. **🟠 深度尺度**：Isaac `distance_to_camera` 是米为单位 float，DeGuV Encoder 吃 uint8 /255。需定一个归一化范围（如 clip 到 [0, near~far] 再 ×255）。Masker 对深度尺度敏感，要调。
4. **🟡 依赖剥离**：DeGuV `utils.py`/`train.py` 夹带 dm_env / robosuite / MuJoCo / quaternion import，云端要么装、要么剥。

---

## 4. 分阶段执行（每阶段可独立验证）

### 阶段 0：环境与依赖
- [ ] 云端确认 Isaac Lab + DeGuV 依赖（dm_env, hydra-core, wandb, dm_tree）可共存
- [ ] 剥离 DeGuV 里 robosuite/MuJoCo/habitat 死引用（只留 dmc/robo adapter 用得到的）
- [ ] 准备 `assets/` 背景图（random_overlay 必需）

### 阶段 1：写 wrapper（核心）
- [ ] `isaac_deguv_wrapper.py`：reset/step/spec 跑通
- [ ] **用随机动作验证**：能出 `(12,H,W)` 观测、done 正确、reward 透传——**先不接 DeGuV 算法**
- [ ] 深度归一化方案定下来（可视化看 mask 输入是否合理）

### 阶段 2：接 DeGuV 算法
- [ ] env_cfg 加 `camera_fixed` 深度
- [ ] 改 `train.py` 的 `setup()`：加 `env == 'isaac'` 分支，调 wrapper
- [ ] 跑通几百步不崩，replay buffer 正常写读
- [ ] 确认 Encoder 输入通道 = 12-3 = 9（`obs_shape[0]-3`，DeGuV 默认），Masker 吃深度

### 阶段 3：训练调通
- [ ] 小规模训练，看 reward 上升、mask 可视化合理
- [ ] 调深度归一化 / Masker lr / 增强强度

### 阶段 4：泛化评估
- [ ] 移植 `eval.py`，对接你已有的光照/纹理/相机随机化（替代 robo_setting 的 easy/medium/hard）
- [ ] 录 mask 可视化视频

---

## 5. 已确认的事实（供执行时参考）

- 你的训练框架：RSL-RL PPO（`OnPolicyRunner`）——**迁移后不再用**
- 现观测：1166 = theia_fixed(576)+theia_wrist(576)+joint(7)+action(7)——**迁移后弃用**
- 相机原图可取：`base_env.scene["camera_fixed"].data.output["rgb"]` / `["distance_to_camera"]`
- 固定相机当前 `data_types=["rgb"]`，**缺深度，必须加**
- 腕部相机已有 `distance_to_camera`（本路线不用腕部）
- 动作：`KinematicRelativeJointDirectAction`，7 维，scale=0.6°
- DeGuV Encoder：`obs_shape[0]-3` 通道（9）过 CNN，Masker 吃 1 通道深度，输出 mask 乘回 RGB

---

## 6. 待定/需注意

- **深度归一化范围**：需根据桌面场景的实际深度分布定（相机高 1.8m，桌面 0.815m → 深度约 1m 量级）
- **frame_stack**：DeGuV 默认 3。Isaac 单环境需在 wrapper 内自己维护帧队列
- **action_repeat**：DeGuV 默认 2；你的 env decimation 已是 2，注意别叠加
- **episode 长度**：你的 `episode_length_s=24` → step 数要和 DeGuV 的 `num_train_frames` 协调
