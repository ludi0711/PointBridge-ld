# gx-VA-isaaclab

xArm7 操作任务的 Isaac Lab 实验工作区。当前分两条主线：

- **位姿真值训练**：用真实工件位姿作为观测，训练 pick 策略
- **视觉编码器训练**：用相机图像 + Theia/Point-BERT 视觉特征作为观测

---

## 环境依赖

### 前置条件

- Isaac Sim（已安装在 `/home/gxai/IsaacLab/_isaac_sim`）
- Isaac Lab（已安装在 `/home/gxai/IsaacLab`）
- conda 环境 `isaaclab`

### 激活环境

```bash
conda activate isaaclab
```

如果 Isaac Lab 尚未安装，参考官方文档：

```bash
cd /home/gxai/IsaacLab
./isaaclab.sh --install
```

### 外部资产

所有 USD 资产已包含在仓库 `assets/xarm7/` 目录中，拉取后即可训练：

- `assets/xarm7/XARM-WITH-GRIP-NEW-FALAN.usd` — 机器人
- `assets/xarm7/gongjian.usd` — 工件
- `assets/xarm7/textures/` — 工件贴图

视觉训练额外需要模型权重（不在仓库中）：

| 资产 | 路径 |
|------|------|
| Theia 模型 | `/home/gxai/IsaacLab/theia_tiny` |
| Point-BERT 权重 | `/home/gxai/IsaacLab/Point_BERT/Point-BERT.pth` |

---

## 共同训练开关

`train_pick_pose.py` 和 `train_pick_vision.py` 默认都关闭 success 提前终止，使用 no-success horizon 训练；需要恢复旧式“到达即 done”时，两个脚本都使用同一个开关：

```bash
--enable_success_termination
```

共同 success 参数为距离 `0.01 m`、姿态 `3.0 deg`、`q_offset=(0.0,0.0,0.0,1.0)`，夹爪绕局部 Z 轴 180 deg 视为等价姿态。pose 和 vision 的 shaping reward 数值按 vision 配置对齐；详细表格见 `docs/training_and_eval.md`。

---

## 位姿真值训练

用真实工件位姿（28 维状态观测）训练 xArm7 pick 策略。

**入口文件**

```
scripts/train/train_pick_pose.py       # 训练/回放脚本
configs/xarm7_pick_pose_env_cfg.py           # 环境配置
configs/agents/rsl_rl_ppo_cfg.py       # PPO runner 配置
```

**训练**

```bash
cd /home/gxai/IsaacLab
python -u gx-VA-isaaclab/scripts/train/train_pick_pose.py \
    --num_envs 1000 \
    --headless \
    --max_iterations 1200
```

**回放**

```bash
python -u gx-VA-isaaclab/scripts/train/train_pick_pose.py \
    --num_envs 1 \
    --play \
    --checkpoint logs/xarm7_pick_pose/<timestamp>/model_<iter>.pt \
    --play_max_steps 1200 \
    --print_every 10
```

---

## 视觉编码器训练

用双相机 Theia 视觉特征、腕部点云特征、关节位置和上一帧动作（782 维观测）训练 pick 策略。

**入口文件**

```
scripts/train/train_pick_vision.py     # 训练/回放脚本
configs/xarm7_pick_vision_env_cfg.py  # 环境配置（含 Theia + Point-BERT 编码器）
configs/agents/rsl_rl_ppo_cfg.py       # PPO runner 配置
```

**训练**

```bash
source /home/gxai/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab
cd /home/lsz/rl_robot/gx-VA-isaaclab

python -u scripts/train/train_pick_vision.py \
    --num_envs 64 \
    --headless \
    --enable_cameras \
    --max_iterations 20000
```

**最小试跑**

如果只想确认入口能拉起来，先跑 1 个环境、1 次迭代：

```bash
source /home/gxai/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab
cd /home/lsz/rl_robot/gx-VA-isaaclab

python -u scripts/train/train_pick_vision.py \
    --num_envs 1 \
    --headless \
    --enable_cameras \
    --max_iterations 1
```

注意：视觉训练需要 `--enable_cameras`。当前脚本会自动把 `args_cli.enable_cameras` 设为 `True`，命令里保留这个参数是为了启动意图更明确；显存占用较大，建议 `num_envs` 不超过 64。

---

## 目录结构

```
gx-VA-isaaclab/
├── assets/xarm7/                # 机器人与夹爪 USD/URDF
├── configs/                     # 环境、场景、MDP、PPO 配置
│   ├── agents/                  # RSL-RL PPO runner 配置
│   ├── xarm7_pick_pose_env_cfg.py     # 位姿真值训练环境
│   ├── xarm7_pick_vision_env_cfg.py  # 视觉训练环境
│   ├── xarm7_pick_liftcube_mdp.py      # 共用 MDP 函数
│   ├── lift_workbench_scene_cfg.py     # 共用场景基类
│   └── pointbert_encoder.py     # Point-BERT 编码器
├── scripts/
│   ├── train/                   # 训练入口（两条主线）
│   │   ├── train_pick_pose.py   # 位姿真值训练
│   │   └── train_pick_vision.py # 视觉编码器训练
│   └── camera/                  # 相机相关工具
├── mujoco_env/                  # MuJoCo 环境封装（备用）
└── tools/                       # USD 检查工具
```
