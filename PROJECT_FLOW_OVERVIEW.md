# xArm7「Point Bridge 点表征」工程 —— 顶层流程与代码映射

> **总纲（一句话框架）**：在抵达最终整流程训练（stage 3）之前，逐个 stage 变换观测形式来训练，
> 用每一档的成败**定位**问题出在哪一环——这就是 stage 的作用（不只是验证，更是归因）。
> 每个 stage 都经历**同一套训练主循环**，这就是训练架构的作用。
> 每个 stage 喂的观测不同，由**观测管线**提供（这是阶梯上唯一的变量）；
> **场景、奖励、控制器**保持不变（被控常量），最后在 stage 3 抵达真机分布。

> 整理日期：2026-08-19
> 依据：规格文档 [point_bridge_rl_observation_spec.md](point_bridge_rl_observation_spec.md)（12 节设计规格）、
> 路线文档 [point_bridge_yolo_guide.md](point_bridge_yolo_guide.md)、仓库 `gx-VA-isaaclab` 全部代码与文档。
> 目标：把"参考文献里的流程"与"实际工程的代码模块/文件"逐段对应起来，建立工程共性认知。

---

## 一、一句话定位

把 Point Bridge（arXiv 2601.16212）的**「深度图 + 分割掩码 → 点云」观测表征**移植到一个已跑通的
xArm7 pick RL 任务（Isaac Lab + RSL-RL PPO）上，最终目标是 **sim2real**：仿真训练与真机部署
共用**同一条**「反投影 → 采点 → 噪声」管线（规格 §1.1 的强制要求），真机侧用 **YOLO-seg + RealSense D435**
替代仿真渲染器的完美实例分割。

整个工程的灵魂约束只有一句话：**掩码来源、深度来源可以在两侧不同，但从"掩码内采点"往后的每一步必须
是同一个函数。**

---

## 二、工程全景：四条主线 × 一个 stage 阶梯

### 2.1 主线一览

| 主线 | 观测 | 入口脚本 | 环境配置 | 状态 |
|---|---|---|---|---|
| A. 位姿真值 | 21 维低维状态 | [train_pick_pose.py](gx-VA-isaaclab/scripts/train/train_pick_pose.py) | [xarm7_pick_pose_env_cfg.py](gx-VA-isaaclab/configs/xarm7_pick_pose_env_cfg.py) | 已收敛基线 |
| A'. 位姿 + CRT 接触夹爪（写入/抬升任务） | 21/22 维 | [train_pick_pose_lift.py](gx-VA-isaaclab/scripts/train/train_pick_pose_lift.py) | [xarm7_pick_pose_crt_write_env_cfg.py](gx-VA-isaaclab/configs/xarm7_pick_pose_crt_write_env_cfg.py) / [xarm7_pick_pose_lift_crt_write_env_cfg.py](gx-VA-isaaclab/configs/xarm7_pick_pose_lift_crt_write_env_cfg.py) | 接触富集任务 |
| B. 视觉编码器（Theia + Point-BERT） | 782 / 398 / 1166 维 | [train_pick_vision.py](gx-VA-isaaclab/scripts/train/train_pick_vision.py) 及 3 个变体 | [xarm7_pick_vision_env_cfg.py](gx-VA-isaaclab/configs/xarm7_pick_vision_env_cfg.py) 等 4 个 | 早期主线，Point Bridge 取代了它 |
| **C. Point Bridge 点云（当前主线）** | 217 维（stage 10 为 409） | [train_pick_pointcloud.py](gx-VA-isaaclab/scripts/train/train_pick_pointcloud.py) / [train_pick_pointcloud_yolo.py](gx-VA-isaaclab/scripts/train/train_pick_pointcloud_yolo.py) | [xarm7_pick_pointcloud_env_cfg.py](gx-VA-isaaclab/configs/xarm7_pick_pointcloud_env_cfg.py) | 规格文档的直接实现 |
| D. 真机部署工具链 | — | [deploy_pointcloud_policy_xarm7.py](gx-VA-isaaclab/scripts/deploy/deploy_pointcloud_policy_xarm7.py) 等 | 复用 C 的共用函数 | 与 C 配套 |

> B 主线已经被 C 取代（观测从 1166 维 Theia 特征换成 217 维点云），但 B 的环境/奖励/场景代码仍是
> C 的模板来源——C 复用 vision 系的 Actions/Rewards/Terminations 类。

### 2.2 Point Bridge 主线的 stage 阶梯（观测表征的进化验证链）

这是规格 §7「分阶段验证」在工程里的落地方式。stage 是**观测表征的档位**，从"信息上限"逐级降级到
"与真机同分布"，每一步都可单独训练、可归因：

| Stage | 掩码/观测来源 | 相机 data_types | 入口 | 用途 |
|---|---|---|---|---|
| -2 | privileged 状态（28 维，无速度） | 无相机 | train_pick_pointcloud.py `--stage -2` | 网络/奖励 sanity 上界 |
| -1 | privileged 状态（35 维） | 无相机 | 同上 `--stage -1` | 基线策略（也是 YOLO 数据集的采集策略） |
| 0 | GT 分割（`noise_std=0`） | depth + seg + rgb | 同上 `--stage 0` | 点云可视化验收（规格 §7 阶段 0，强制 num_envs=1） |
| 1 | GT 网格表面点 + GT 位姿（零噪声） | 摘相机 | 同上 `--stage 1` | **信息上限测试**（规格 §7 阶段 1：点表征不经过相机） |
| 2 | GT 实例分割完整管线（σ=1cm） | depth + instance_seg | 同上 `--stage 2`（默认） | **主训练**（规格 §7 阶段 2） |
| 3 | **YOLO-seg 掩码** | depth + rgb | train_pick_pointcloud_yolo.py（硬编码 `_STAGE=3`，:156） | sim2real 微调，消除掩码 gap |
| 4 | YOLO + 掩码缓存 | 同 stage 3 | 同上 `--yolo_step` 触发 | 省 YOLO 推理开销的变体 |
| 10 | stage 2 + **腕部相机点云**（整帧无分割，+192 维） | 固定相机同 stage 2 + 腕部 depth | train_pick_pointcloud.py `--stage 10` | 补充末端近场稠密几何 |

stage 生效的唯一入口：`set_observation_stage(cfg, stage)` ——
[xarm7_pick_pointcloud_env_cfg.py:1625-1762](gx-VA-isaaclab/configs/xarm7_pick_pointcloud_env_cfg.py#L1625-L1762)。

### 2.3 stage 阶梯的意义：把复杂问题拆成可独立判决的命题

> stage 的作用是让复杂问题得到拆解——具体拆法如下。

**① 四个关键档位的名词解释**（观测信息从"上帝视角"逐级降级到"真机同分布"）：

| 档位 | 观测里装的是什么 | 一句话理解 |
|---|---|---|
| privileged（特权状态，stage -2/-1） | 35 维：物体位姿 + 末端位姿 + 关节角 + 关节速度 + 上一动作 | 直接读仿真器内部状态的"上帝视角"——真机上没有任何传感器能直接给出这些量 |
| GT 点（stage 1） | 64 物体点（网格表面 GT 采样）+ 6 夹爪点 + 关节角 | 点云的形态已与最终一致，但点的来源仍是上帝——无相机、无噪声、点全完美 |
| 仿真相机（stage 2） | 同样的 70 点，但从渲染深度 + 实例分割反投影而来，带 σ=1cm 噪声 | 点的来源换成仿真感知管线：有遮挡、有噪声、有误差；掩码仍是渲染器完美分割 |
| 真机掩码（stage 3） | 管线同上，只把掩码来源换成真机 YOLO 推理 | 观测的最后一个仿真专属组件被换掉，分布从此与部署时一致 |

**② 顺序为什么是"从假到真"**：观测是一条信息管道 `世界 → 传感器 → 掩码 → 点云 → 网络`，
四个阶段从管道末端开始，**每次把"离真机最远的那一个组件"换成真机的**——每步只换一个组件，
所以每步成败只属于那一个组件；方向固定为"从假到真"，保证走到任何一步时，它下游的所有组件都已被验证过。

**③ 拆解为什么必要**（本质）：

- 真机成功是一串假设的**合取**：`控制可学 ∧ 表征充分 ∧ 感知管线保真 ∧ 掩码可替代 ∧ …`——任何一环为假整体为假，且 RL 没有"部分得分"；
- RL 失败的观测是**退化**的：reward 平线兼容所有原因，单次复合实验（直接训最终形态）信息量≈0；
- 阶梯把合取拆成**每步只动一个变量的独立判决**，失败原因在逻辑上唯一；同时区分科学假设（stage 1 失败 = 表征方向错，止损）与工程假设（stage 2 失败 = 管线有 bug，修代码）；
- 每个边界记录的落差 = 该环节**牺牲信息的真实价格**（性能预算表）——真机出问题时定点投资，不需重新诊断；
- **事后二分补不出来**：成本最坏指数级；且部分失败特征（如表征不充分）在复合实验里会被噪声掩盖，可辨识性不可逆。

---

## 三、训练主循环：所有入口共享的同一套骨架

8 个训练脚本结构完全同构（「一条主线 = 入口脚本 + env_cfg + agent_cfg 三件套」）：

```
train_*.py (argparse: --num_envs/--play/--checkpoint/--resume/--load_run/--max_iterations/--seed)
 └─ AppLauncher
     ├─ 构建 env_cfg，点云线在此调用 set_observation_stage(cfg, stage)
     ├─ ManagerBasedRLEnv(cfg)                     ← 全部用 ManagerBased，无 DirectRLEnv
     ├─ RslRlVecEnvWrapper(env)
     └─ OnPolicyRunner(env, agent_cfg.to_dict())   ← rsl_rl；点云线先 register_with_rsl_rl()
         ├─ [--resume] runner.load(checkpoint / load_run)
         └─ runner.learn(max_iterations, init_at_random_ep_len=True)
             └─ 按 save_interval 每 100 迭代存 model_<iter>.pt（OnPolicyRunner 负责，脚本不管）
```

点云线 PPO 超参（[rsl_rl_ppo_cfg.py](gx-VA-isaaclab/configs/agents/rsl_rl_ppo_cfg.py) 的
`XArm7PickPointCloudPPORunnerCfg` :1440-1481）：

| 项 | 值 | 项 | 值 |
|---|---|---|---|
| num_steps_per_env | 24 | lr | 3e-4（adaptive） |
| num_learning_epochs | 5 | num_mini_batches | 4 |
| gamma / lam | 0.98 / 0.95 | clip_param | 0.2 |
| entropy_coef | 0.002 | desired_kl | 0.008 |
| max_iterations | 12000 | save_interval | 100 |
| empirical_normalization | **False**（actor 侧） | obs_groups | `{"policy":["policy"], "critic":["critic"]}` |

工程习惯（在 [training_and_eval.md](gx-VA-isaaclab/docs/training_and_eval.md) 中约定）：

- **no-success horizon 默认开启**：成功不提前终止，episode 跑满；`--enable_success_termination` 恢复旧行为。
- **`--max_iterations` 是相对值**：`tot_iter = start_iter + num_learning_iterations`（续训语义）。
- **复现快照**：训练脚本在启动时把关键源文件拷进 log 目录（yolo 版拷 4 个：
  env_cfg / point_bridge_pointcloud / yolo_mask_source / pointnet_actor_critic，
  [train_pick_pointcloud_yolo.py:258-268](gx-VA-isaaclab/scripts/train/train_pick_pointcloud_yolo.py#L258-L268)）。
- log 目录：`logs/xarm7_pick_pointcloud_stage{stage}/时间戳/`。

### 环境内部步进与控制（与文档 [isaac_control_whitebox.md](gx-VA-isaaclab/docs/isaac_control_whitebox.md) 对应）

- `decimation = 2`：每个 policy step 跑 2 个 physics substep；`sim.render_interval = decimation`（渲染与控制步长对齐，规格 §4.1④）。
- 动作是**自定义的 `KinematicRelativeJointDirectAction`**（[xarm7_pick_pose_env_cfg.py:133](gx-VA-isaaclab/configs/xarm7_pick_pose_env_cfg.py#L133)）：
  `delta_q = clip(action × 0.6°, ±0.6°)` → `q_target = q_current + delta_q` →
  **`write_joint_state_to_sim(q_target, vel=0)` 直接写关节状态**（不走 PhysX drive/actuator 追踪），
  step 末 `lock_to_target_after_physics()` 锁回 q_target。7 维关节增量、**无夹爪动作**（夹爪在 CRT 线自动闭合 / 其他线固定）。
  这是与 Isaac 原生 JointPositionAction 的本质区别（state 路线 vs target 路线），真机部署用
  `set_servo_angle_j` 对应同语义。

---

## 四、观测管线：从相机到网络输入（工程核心）

### 4.1 数据流总览

```
TiledCamera camera_fixed (640×480, fx=fy=615, convention="ros",
                          depth_clipping_behavior="zero", distance_to_image_plane)
data_types 由 stage 裁剪: stage 0/1/2/10 → [depth, instance_seg];  stage 3/4 → [depth, rgb]
        │
   ┌────┴──────────────────┬──────────────────────────┐
   │ depth+seg             │ rgb → YOLO               │ depth（腕部, 仅 stage 10）
   ▼                       ▼                          ▼
point_bridge_point_cloud  point_bridge_point_cloud_yolo   point_bridge_wrist_point_cloud
   build_instance_id_lut    yolo_masks() (conf最高去重,     全帧 mask（无分割）
   (camera.data.info)       全程 GPU 不下卡)
        │                       │
        └───────────┬───────────┘
                    ▼
        mask_depth_to_pointcloud   ← ★ 唯一共用实现（规格 §1.1）
        valid 过滤 → 确定性预筛(≤512) → 确定性 FPS(64) → 手写反投影
        → 基座系 SE(3) 变换 → 工作空间裁剪 → 高斯噪声(仅物体点)
        → (points (B,64,3), counts (B,))
                    │
        gripper_keypoints(ee 位姿 → 6 点, 无噪声)   ← 不经过相机，零渲染开销
                    ▼
        assemble_point_cloud → [物体 192 | 夹爪 18] + joint_pos 7
                    │
        policy 组: (B, 217)   [stage 10 加腕部 192 → 409]
        critic 组: privileged_state (B, 35) = 物体位姿7 ⊕ ee位姿7 ⊕ 关节角7 ⊕ 关节速度7 ⊕ 上一动作7
```

### 4.2 各环节的代码位置

| 环节 | 文件:行号 | 要点 |
|---|---|---|
| 相机配置（内参/外参/data_types/裁剪） | [xarm7_pick_pointcloud_env_cfg.py:311-347](gx-VA-isaaclab/configs/xarm7_pick_pointcloud_env_cfg.py#L311-L347) | `make_pointcloud_camera_cfg`；规格 §4.1 四条全部满足 |
| 标定常量 | 同文件 :138-139 `CALIB_CAMERA_*`（基座系原始值）与 :150/:167 `FIXED_CAMERA_*`（换算到 env 系的 OffsetCfg 值） | 见 §7.2 的常量溯源链 |
| 外参自检 | 同文件 :217-269 `_verify_camera_extrinsics()` | import 时自动校验 |
| GT 分割观测函数 | 同文件 :769+ `point_bridge_point_cloud`（含零阶保持 `_MAX_EMPTY_STRIKES=30`） | GT 路线超限 raise RuntimeError |
| YOLO 观测函数 | 同文件 :949-1238 `point_bridge_point_cloud_yolo` / `_cached` | 仅换掩码来源，下游共用 |
| 腕部观测函数 | 同文件 :829 `point_bridge_wrist_point_cloud` | 整帧深度采 64 点，无分割 |
| YOLO 封装 | [yolo_mask_source.py](gx-VA-isaaclab/configs/yolo_mask_source.py) | `.yolo_deps` 自动挂载、模型缓存、`yolo_masks` :186（conf 最高去重、nearest 插值、>0.5 阈值）、`YOLO_CONF=0.25`、imgsz=640 |
| **共用主函数** | [point_bridge_pointcloud.py:294-397](gx-VA-isaaclab/configs/point_bridge_pointcloud.py#L294-L397) `mask_depth_to_pointcloud` | 见下 |
| 夹爪关键点 | 同文件 :118-125 布局、:400 `gripper_keypoints` | 与规格 §4.3 表逐行一致，默认**不加噪声** |
| 组装/归一化 | 同文件 :445 `assemble_point_cloud`、:459 `normalize_points` | 固定 min-max 仿射，**在网络内做** |
| 实例 ID 查表 | 同文件 :487 `build_instance_id_lut` | 解决 TiledCamera 共享 info dict、多 mesh、邻 env 入画三个坑 |
| 观测组定义 | [xarm7_pick_pointcloud_env_cfg.py:1893-1968](gx-VA-isaaclab/configs/xarm7_pick_pointcloud_env_cfg.py#L1893-L1968) | policy 组 217 / critic 组 35；点云必须在最前、物体点先于夹爪点 |
| 不变量测试 | [test_point_bridge_pointcloud.py](gx-VA-isaaclab/tests/test_point_bridge_pointcloud.py)（645 行，**无需 Isaac Sim 即可跑**） | 规格 §4.5 + §12 确定性全覆盖 |

### 4.3 网络侧：PointNetActorCritic

[model/pointnet_actor_critic.py](gx-VA-isaaclab/model/pointnet_actor_critic.py)（rsl_rl 兼容类，
经 `register_with_rsl_rl` 注入）：

```
actor:
  观测 (B,210) ──切分──> obj_pts (B,64,3) ─┐
                                          ├─ 同一个 PointNetEncoder（共享权重，各跑一次 = 双 token）
                      robot_pts (B,6,3) ──┘   in_dim=3, hidden=(64,128,256), out_dim=512
                                          → 两个 embedding concat [512+512]
                                          → + joint_pos 7 → MLP [256,256] (elu) → 7 维关节增量
critic:
  privileged_state (B,35) → MLP [512,256,128] → 1 维 value
```

- PointNetEncoder：共享 MLP（`nn.Linear` 对点维广播）+ LayerNorm + ReLU + **max pool** + proj；**无 T-Net、无 BatchNorm**（规格 §5.2 的警告被严格执行）。
- 归一化 `normalize_points` 在网络内部做，min/max 注册为 buffer 随 checkpoint 保存；actor 侧关闭 EmpiricalNormalization（观测保持米制）。
- critic 吃 35 维 privileged 状态（规格 §5.3 的意图实现——"保留原观测"字面无法执行，原 Theia 观测已整体移除）。
- 超参由 [rsl_rl_ppo_cfg.py:1395-1436](gx-VA-isaaclab/configs/agents/rsl_rl_ppo_cfg.py#L1395-L1436) 的 `RslRlPointNetActorCriticCfg` 提供，stage 10 时训练脚本把 `num_wrist_points` 置 64。

### 4.4 为什么点云要加噪声（σ=1cm，仅物体点）

- **根因：仿真深度完美、真机深度有噪声。** 渲染深度零误差；真机 RGB-D 误差随距离增长
  （[point_bridge_pointcloud.py:93-94](gx-VA-isaaclab/configs/point_bridge_pointcloud.py#L93-L94) 的 TODO：
  ~1.2m 处实测若为 2-3cm 必须上调 `NOISE_STD_M`）。若训练点全部几何完美，策略会学会依赖精确几何做决策，
  而这些精度真机上不存在——部署时观测落在训练分布之外。
- **噪声 = 把训练分布撑开、覆盖部署分布**：训练时点就带 ~1cm 抖动，策略天然容忍真实传感器的随机抖动。
- **幅度必须对标真机实测**：σ=0.01m 取自参考工作；[calibrate_depth_noise.py](gx-VA-isaaclab/scripts/camera/calibrate_depth_noise.py)
  实测真机逐点深度噪声，与 `NOISE_STD_M` 对比给出结论。原理：噪声幅度 ≈ 真机传感器噪声 → 训练分布覆盖部署分布；
  小了覆盖不住，大了白牺牲信息。
- **噪声只治方差、不治偏差**：零均值高斯覆盖不了 YOLO 掩码的系统性偏差（规格 §11.2"偏差不是噪声"）。
  所以 stage 2（噪声管传感器方差）之外还需要 stage 3（真 YOLO 进训练环管掩码偏差）——两者分工，缺一不可。
- **只加在物体点**：夹爪点在真机侧来自编码器 + 正运动学，精度远高于 RGB-D，加噪是"比现实更悲观"；
  参考实现同样只对 object points 加噪（修正了规格 §12 早先"夹爪点也加"的自行指定）。
- **受控性**：确定性 FPS（规格 §12）已消除采样抖动这一随机源，点云噪声只剩"故意加的高斯噪声"一个来源——
  加多少、加在哪完全可控，失败可归因。

---

## 五、场景 / MDP / 控制器层

### 5.1 场景继承树（真正的类继承只发生在场景层）

```
InteractiveSceneCfg
└── LiftWorkbenchSceneCfg            lift_workbench_scene_cfg.py:41   ← 共用基类（地面/工作台/桌 z=0.815/工件/灯；robot、相机留 None 占位）
    ├── PickPoseSceneCfg             pose_env_cfg.py:763              ← 位姿主线（无相机）
    │   └── PickPoseCRTWriteSceneCfg crt_write_env_cfg.py:451         ← 换 CRT 接触夹爪机器人 USD
    │       └── PickPoseLiftCRTWriteSceneCfg  lift_crt_write:161      ← 工件换动力学可抓取版（gongjian_graspable.usd，重力开）
    └── XArm7PickLiftCubeSceneCfg（vision/nopc/spatial/pointcloud 四个文件**各自独立定义**，互不共享）
```

**MDP 配置类（Actions/Observations/Events/Rewards/Terminations）在各 env 文件里基本是复制粘贴式重复声明**，
只有 CRT 系通过 import 复用 pose 的 `ActionsCfg`。**改奖励权重必须逐文件改**——这是工程里最大的坑之一。

### 5.2 奖励结构（各主线 v88 同步，权重逐文件重复）

| 项 | 函数 | 权重 |
|---|---|---|
| reaching_coarse / mid / fine | `object_ee_distance` exp(-d/0.50, 0.15, 0.04) | 35 / 55 / 80 |
| ee_orientation_coarse / mid / fine | `ee_orientation_alignment`（夹爪 Z 轴 180° 对称） | 30 / 40 / 70 |
| reach_success_bonus | 0.01m / 3° 判据 | **0**（只留监控指标） |
| action_rate | `mdp.action_rate_l2` | −10 |
| collision_penalty | 夹爪接触力 > 0.5N | −1000 |
| （lift 线追加）object_lift_height / ee_yaw_free_tilt | 抬升进度 / 绕工具轴 yaw 自由姿态 | +120 / +35 |

共用 MDP 函数库是 [xarm7_pick_liftcube_mdp.py](gx-VA-isaaclab/configs/xarm7_pick_liftcube_mdp.py)
（含翻面任务等备选函数）。success 判定：`ee_reached_object` 位置 <0.01m 且姿态 <3°；
`grasp_target_pos_w`（:84）是全项目抓取目标点的唯一入口（目标位姿不单独采样，目标 = 当前工件位姿 + `q_offset=(0,0,0,1)` 朝下抓）。

### 5.3 随机化与课程

- reset 随机化：`reset_table_height_and_object_pose` —— 工件 x/y ±0.10~0.125m、yaw ±90°、桌高增量、灯光/贴图（vision/spatial 另有图像增强）。
- Isaac 侧**没有 curriculum**（各 `CurriculumCfg` 为空）；课程逻辑只存在于参考实现
  [mujoco_env/](gx-VA-isaaclab/mujoco_env/) 的两个 robosuite 文件（原工作的课程 pickplace 环境 V4.1/V4.3，Isaac 侧注释以它们为准绳对齐）。

### 5.4 CRT 控制器（A' 主线）

[controller/crt_gripper_contact_controller.py](gx-VA-isaaclab/configs/controller/crt_gripper_contact_controller.py)
是夹爪的 set-target PD **接触控制器**（非直接力矩）：auto-close（EE 距工件 <12mm 且姿态 <1° 触发）、
pad 法向力 hold（默认 12N）、30N 力伺服、左右同步限位；臂侧控制与 pose 主线完全相同。
[crt_gripper_runtime.py](gx-VA-isaaclab/configs/controller/crt_gripper_runtime.py) 是无 omni 依赖的
eval/回放运行时助手。

---

## 六、真机部署链路（sim2real 全流程）

```
[仿真侧]                                        [真机侧]
① 采集策略轨迹 RGB                              ⑥ 手眼标定（AprilTag，RMS<10mm）
   collect_policy_rgb_episodes.py                  → camera_to_right_arm_base.json
   （跑 stage -1 privileged 策略，相机纯旁观）        → 转抄成 CALIB_CAM_* 常量
   → logs/policy_rgb_episodes/（560 帧 + index.csv）   → 换算 FIXED_CAM_* 喂仿真渲染器
        │                                         校验三件套：
② 数据集打包                                         transition.py（反投影+归一化核对）
   tools/collect_yolo_dataset.py                     mark_camera_position.py（视口画球）
   （仿真分层 250 + 真机均匀 250）                     compare_sim_real_camera.py（并排目视）
   → yolo_labeling_<日期>/ + manifest.csv                │
        │ 上传平台打标 + 训练 YOLO26n-seg          ⑦ 深度噪声实测
③ 评测                                              calibrate_depth_noise.py → 对比 NOISE_STD_M
   eval_yolo_weights.py（mAP / 560 帧无标签体检）
   test_yolo_camera.py（真机实时）                      │
        │                                        ⑧（备选）手动 ROI
④ 仿真训练 stage 2 收敛 → checkpoint                manual_roi_annotate.py → roi_mask.json/.png
   （再续训 stage 3 适应 YOLO 掩码分布）               manual_roi_pointcloud.py（早期无 YOLO 路线）
        └────────────────────────────┬──────────────────┘
⑤ deploy_pointcloud_policy_xarm7.py
   D435 → 掩码（现场手标 polygon 或 YOLO 确认）→ build_pointcloud
   → build_obs（217 维）→ PointNetActorCritic（训练同一类）
   → 动作缩放 0.6°、限位外 1° 裕量、空掩码 hold → set_servo_angle_j（50Hz）
   → 日志 logs/deploy_pointcloud/（45 列 CSV + 纯 RGB 录像 + 掩码备份）
```

**真机侧复用了同一份实现（规格 §1.1 成立）**：真机点云唯一入口
[real_roi_pointcloud.py](gx-VA-isaaclab/scripts/camera/real_roi_pointcloud.py) 的 `build_pointcloud`
直接调 `configs.point_bridge_pointcloud.mask_depth_to_pointcloud`；部署脚本同样复用
`gripper_keypoints` 与 `yolo_mask_source.yolo_masks`（与 stage 3 训练同函数）。

---

## 七、参考文献流程 vs 实际实现：偏离清单

这是"看过论文/规格后读代码"最容易困惑的地方——实现比规格新，且有几处**有意偏离**（代码内注释都写了理由）：

| # | 规格（参考文献） | 实际实现 | 性质 |
|---|---|---|---|
| 1 | §3.2 每点 4 维含类型通道 `TYPE_OBJ/TYPE_ROBOT` | **3 维纯 xyz，无类型通道**；身份由"物体/夹爪各过一次**同一** encoder 的双 token"承担（对齐参考实现 `pb.py` 的原做法，比类型通道更硬的结构性保证） | 有意偏离（规格自己把"两个独立 PointNet"列为可选变体） |
| 2 | §1.1 签名 `→ [B, m_obj, 4]` | 返回 `(points (B,m_obj,3), counts (B,))` 元组，counts 供 visible_ratio 与零阶保持 | 扩展 |
| 3 | §4.2 步骤 2 优先用内置 `unproject_depth` | 手写 K⁻¹ `unproject_pixels`（只算掩码像素，省 236MB 中间张量），由逐元素对比测试背书 | 等价替代 |
| 4 | §12 自行指定"夹爪点也加噪声" | 夹爪点默认**不加噪声**（对齐参考实现：只对 object points 加噪） | 修正（读到了参考实现源码） |
| 5 | §4.2/§11.1 零阶保持"复用上一帧+计数报警" | 在 env 观测函数层实现（`_MAX_EMPTY_STRIKES=30`）；GT 路线超限 raise，YOLO 路线超限存诊断图+继续 | 位置不同 |
| 6 | §11.1 真机掩码内缩 20% | **未实现** | 未做 |
| 7 | §2.2 相机分辨率从 84×84 起试 | 直接 640×480（真机 D435 标定内参 fx=fy=615）；clipping_range (0.05, 20) | 偏离 |
| 8 | §5.1 单编码器 [128,128]→256 | 双 token 编码器 (64,128,256)→512（对齐参考实现 dp3_encoder.py）；注释说明原规格把 GPT hidden dim 与 PointNet 输出搞混了 | 修正 |
| 9 | §5.3 "critic 保留原观测不动" | 原 Theia 观测已整体移除，critic 改为 35 维 privileged GT 状态 | 意图实现 |
| 10 | §7 阶段 3（掩码增强）"本版不做" | 已做并扩展：stage 3/4（YOLO 训练内）、stage 10（腕部相机）、stage -1/-2（privileged 上界） | 规格外新增 |
| 11 | 参考实现用数据集 stats 归一化 | 固定工作空间 min-max 仿射（RL 无预采数据集），且在网络内做 | 改编 |
| 12 | §4.2 直接 FPS | 先等间距预筛到 512 候选再 FPS（确定性保持，性能优化） | 优化 |
| 13 | §3.2 常量 POINT_DIM=4 | POINT_DIM=3 | 同 #1 |

**两个文档级的过时文字**（读代码时的陷阱）：env_cfg 模块头注释写"287 维"实际是 217 维；
env_cfg :709 docstring 的 `(B, P, 4)` 应为 3 维。

---

## 八、工程共性认知（约定与套路）

读这个工程的代码时，先记住这十条，很多"奇怪"的地方立刻能看懂：

1. **一条主线 = 入口脚本 + env_cfg + agent_cfg 三件套**，三者在一个脚本里显式绑定；MDP 类在各文件重复声明，改权重要逐文件同步。
2. **唯一共用函数原则**：`mask_depth_to_pointcloud` 是仿真/真机/YOLO 各路唯一实现，任何新入口（如腕部、ROI）都只在上游换掩码/深度来源，下游不动。
3. **stage 阶梯 = 归因工具**：从信息上限逐级降级观测，每级单独训练，失败可定位到"表征 vs 算法 vs 掩码分布"。
4. **确定性原则**：FPS 起点固定、预筛用 stable 排序、无任何 RNG——因为 H=1 没有时序平滑，采样抖动会直接变成观测噪声（有专门测试 `test_static_scene_is_stable`）。
5. **坐标系约定**（写错就是规格 §9 的"点云镜像 90°"）：
   - 深度 = `distance_to_image_plane`（沿相机 z 轴，不是到光心的欧氏距离）；`depth_clipping_behavior="zero"`；
   - 相机系 = OpenCV/ROS 光学系，取姿态必须用 `camera.data.quat_w_ros`（不是 `quat_w_world`）；
   - 点云一律在**机器人基座系**、单位**米**；`pose_in_frame` 统一做世界系→基座系（自动抵消多环境 env_origins）。
6. **常量溯源链**：真机标定 → `transition.py` 校验 → `CALIB_CAMERA_*`（基座系原值）→ `FIXED_CAMERA_*`（env 系渲染值）→ import 时 `_verify_camera_extrinsics()` 自检。
7. **privileged critic 原则**：critic 只在训练时存在，永远给特权信息；actor 降级观测的成败因此可单独归因。
8. **观测扁平布局契约**：`[物体点 192 | 夹爪点 18 | 低维向量...]`，网络按长度切分（有测试 `test_flat_observation_layout_matches_network_split` 保证），改顺序必先看 `PointNetActorCritic._encode_actor`。
9. **log 目录 = 复现单元**：时间戳目录 + 关键源文件快照 + model_<iter>.pt；工具链按"策略时间戳"组织输出（[policy_log_paths.py](gx-VA-isaaclab/tools/policy_log_paths.py)）。
10. **视觉/相机调试三件套**：`mark_camera_position`（渲染器实际相机位姿画球）→ `compare_sim_real_camera`（并排目视）→ `transition.py` 式数值核对；外参对齐是 sim2real 的第一道关。

---

## 九、关键文件速查表

### 训练与算法
| 文件 | 职责 |
|---|---|
| [scripts/train/train_pick_pointcloud.py](gx-VA-isaaclab/scripts/train/train_pick_pointcloud.py) | Point Bridge 主入口（stage -2~2/10），含 stage 0 可视化 |
| [scripts/train/train_pick_pointcloud_yolo.py](gx-VA-isaaclab/scripts/train/train_pick_pointcloud_yolo.py) | stage 3/4 入口（YOLO 掩码线），四文件快照 |
| [scripts/train/train_pick_pose.py](gx-VA-isaaclab/scripts/train/train_pick_pose.py) / [_lift.py](gx-VA-isaaclab/scripts/train/train_pick_pose_lift.py) | 位姿真值线（含 CRT）训练/回放 |
| [scripts/train/train_pick_vision.py](gx-VA-isaaclab/scripts/train/train_pick_vision.py) 等 4 个 | 视觉编码器线（Theia + Point-BERT 变体） |
| [configs/agents/rsl_rl_ppo_cfg.py](gx-VA-isaaclab/configs/agents/rsl_rl_ppo_cfg.py) | 所有 PPO runner 配置（含大量历史版本 V10~V58） |
| [scripts/eval/eval_pick_pose.py](gx-VA-isaaclab/scripts/eval/eval_pick_pose.py) | 独立评估（可手定工件位姿） |

### 观测管线（Point Bridge 核心）
| 文件 | 职责 |
|---|---|
| [configs/point_bridge_pointcloud.py](gx-VA-isaaclab/configs/point_bridge_pointcloud.py) | ★ 共用点云管线唯一实现（反投影/FPS/噪声/关键点/归一化/实例 ID） |
| [configs/xarm7_pick_pointcloud_env_cfg.py](gx-VA-isaaclab/configs/xarm7_pick_pointcloud_env_cfg.py) | 点云环境：相机标定、观测函数、stage 分支、观测组 |
| [configs/yolo_mask_source.py](gx-VA-isaaclab/configs/yolo_mask_source.py) | YOLO 封装（deps 挂载、缓存、批量推理、去重） |
| [model/pointnet_actor_critic.py](gx-VA-isaaclab/model/pointnet_actor_critic.py) | PointNet actor + privileged critic |
| [tests/test_point_bridge_pointcloud.py](gx-VA-isaaclab/tests/test_point_bridge_pointcloud.py) | 不变量测试（无需 Isaac Sim） |
| [configs/pointbert_encoder.py](gx-VA-isaaclab/configs/pointbert_encoder.py) | Point-BERT 独立推理（视觉主线遗留，注意其深度语义是 distance_to_camera） |

### 场景 / 任务
| 文件 | 职责 |
|---|---|
| [configs/lift_workbench_scene_cfg.py](gx-VA-isaaclab/configs/lift_workbench_scene_cfg.py) | 共用场景基类（桌顶 z=0.815、工件 gongjian、灯） |
| [configs/xarm7_pick_liftcube_mdp.py](gx-VA-isaaclab/configs/xarm7_pick_liftcube_mdp.py) | 全项目唯一共用 MDP 函数库 |
| [configs/xarm7_pick_pose_env_cfg.py](gx-VA-isaaclab/configs/xarm7_pick_pose_env_cfg.py) | 位姿线环境 + 自定义动作类 `KinematicRelativeJointDirectAction` |
| [configs/controller/crt_gripper_contact_controller.py](gx-VA-isaaclab/configs/controller/crt_gripper_contact_controller.py) | CRT 接触夹爪控制器 |
| [mujoco_env/*.py](gx-VA-isaaclab/mujoco_env/) | 参考实现（robosuite 课程 pickplace V4.1/V4.3），Isaac 线的对照基准 |

### 真机部署与工具
| 文件 | 职责 |
|---|---|
| [scripts/deploy/deploy_pointcloud_policy_xarm7.py](gx-VA-isaaclab/scripts/deploy/deploy_pointcloud_policy_xarm7.py) | 点云策略真机部署主脚本（50Hz 伺服 + 全套安全机制） |
| [scripts/camera/real_roi_pointcloud.py](gx-VA-isaaclab/scripts/camera/real_roi_pointcloud.py) | 真机点云唯一入口（复用共用函数） |
| [scripts/train/collect_policy_rgb_episodes.py](gx-VA-isaaclab/scripts/train/collect_policy_rgb_episodes.py) | 策略轨迹 RGB 采集（YOLO 数据集仿真图源） |
| [tools/collect_yolo_dataset.py](gx-VA-isaaclab/tools/collect_yolo_dataset.py) / [eval_yolo_weights.py](gx-VA-isaaclab/tools/eval_yolo_weights.py) / [test_yolo_camera.py](gx-VA-isaaclab/tools/test_yolo_camera.py) | YOLO 数据打包 / 权重评测 / 真机实时评测 |
| [scripts/camera/read_realsense_camera_params.py](gx-VA-isaaclab/scripts/camera/read_realsense_camera_params.py) / [calibrate_depth_noise.py](gx-VA-isaaclab/scripts/camera/calibrate_depth_noise.py) | 相机参数读取 / 深度噪声实测 |
| [scripts/train/compare_sim_real_camera.py](gx-VA-isaaclab/scripts/train/compare_sim_real_camera.py) / [mark_camera_position.py](gx-VA-isaaclab/scripts/train/mark_camera_position.py) | 仿真-真机相机对齐校验工具 |
| [transition.py](transition.py) | 相机外参校验 + 归一化推导（顶层的草稿脚本，常量已固化进 env_cfg） |

### 文档
| 文件 | 内容 |
|---|---|
| [point_bridge_rl_observation_spec.md](point_bridge_rl_observation_spec.md) | 设计规格（12 节）：观测、网络、验证、归因、部署约束 |
| [point_bridge_yolo_guide.md](point_bridge_yolo_guide.md) | YOLO 路线（stage 3）使用指南 |
| [docs/training_and_eval.md](gx-VA-isaaclab/docs/training_and_eval.md) / [deployment.md](gx-VA-isaaclab/docs/deployment.md) / [camera.md](gx-VA-isaaclab/docs/camera.md) / [tools_and_logs.md](gx-VA-isaaclab/docs/tools_and_logs.md) / [analysis.md](gx-VA-isaaclab/docs/analysis.md) | 训练/eval、21D 策略部署、相机工具、日志 schema、分析画图 |
| [docs/isaac_control_whitebox.md](gx-VA-isaaclab/docs/isaac_control_whitebox.md) | Isaac 控制链源码打白（action → PhysX 全链路） |
| [docs/deguv_migration_plan.md](gx-VA-isaaclab/docs/deguv_migration_plan.md) | DeGuV → Isaac 迁移规划（**规划文档，尚未实施**） |
| [CHANGELOG.md](gx-VA-isaaclab/CHANGELOG.md) / [README_LIFTCUBE_TRAINING.md](gx-VA-isaaclab/README_LIFTCUBE_TRAINING.md) | 早期 liftcube 版本演进史（v1~v6，现已并入当前配置体系） |

### 已知遗留物（非活跃代码）
- `data/v12_endstates*.npy`：全仓库无引用，死数据。
- `roi_mask.json/.png`：早期手动 ROI 路线产物，部署已改用现场 polygon / YOLO。
- `logs/collect_300k.log`：256 env 大规模采集尝试的 OOM 日志（目标 30 万帧未跑成，落地数据是 560 帧 policy_rgb_episodes）。
- `scene_cfg` 里的 `WORKPIECE_USD_PATH` 等 Linux 绝对路径常量（/home/gxai/...）未使用，各 env 用自己的 `GONGJIAN_USD`。
- vision/nopc/spatial 三个文件的同名类互不共享，改动不会互相影响。

---

## 十、一分钟串讲（把全流程过一遍）

1. 真机上用 D435 拍工件，AprilTag 标定出 camera→base，人工抄进 [xarm7_pick_pointcloud_env_cfg.py](gx-VA-isaaclab/configs/xarm7_pick_pointcloud_env_cfg.py)（`CALIB_CAMERA_*`），换算成仿真相机 OffsetCfg（`FIXED_CAMERA_*`），[transition.py](transition.py) 与视口工具验证。
2. 仿真里 `--stage 2` 训练：TiledCamera 出 depth + 实例分割 → [point_bridge_pointcloud.py](gx-VA-isaaclab/configs/point_bridge_pointcloud.py) 的 `mask_depth_to_pointcloud` 反投影出 64 个物体点 + 6 个夹爪关键点 → 217 维观测进 PointNet actor（critic 吃 35 维特权状态）→ 7 维关节增量动作直接写关节状态。
3. 真机上先采策略轨迹图（560 帧）→ 混合真机图打标 → 训练 YOLO26n-seg（20260815.pt）→ 从 stage 2 checkpoint 续训 stage 3（YOLO 换掉实例分割，其余管线一字不动）。
4. [deploy_pointcloud_policy_xarm7.py](gx-VA-isaaclab/scripts/deploy/deploy_pointcloud_policy_xarm7.py) 上真机：D435 + YOLO（或手标 polygon）→ **同一个** `mask_depth_to_pointcloud` → 同一个网络 → 50Hz 伺服，动作缩放 0.6°、空掩码 hold、限位裕量拒发。
5. 每一步都有对应日志（训练 TensorBoard / 部署 45 列 CSV / YOLO 560 帧统计），失败按规格 §9 归因树定位。
