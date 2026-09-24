# 训练和评估

## 共同开关与奖励参数

`train_pick_pose.py` 和 `train_pick_vision.py` 的 success termination 开关保持一致：默认关闭 `reach_success` 提前终止，episode 跑到 timeout；如果需要恢复旧式“到达即 done”，两个脚本都使用同一个参数：

```bash
--enable_success_termination
```

共同 success / reward 参数：

| 参数 | 值 |
|------|----|
| success 距离阈值 | `0.01 m` |
| success 姿态阈值 | `3.0 deg` |
| `q_offset` | `(0.0, 0.0, 0.0, 1.0)`，wxyz |
| 夹爪姿态等价 | `(1,0,0,0)` 与绕局部 Z 轴 180 deg 的 `(0,0,0,1)` |
| `reach_success_bonus` | `weight=0.0`，只保留指标项，不额外加奖励 |

pose 和 vision 的 shaping reward 数值按 vision 配置对齐：

| reward term | `std` / 参数 | `weight` |
|-------------|--------------|----------|
| `reaching_coarse` | `std=0.50` | `35.0` |
| `reaching_mid` | `std=0.15` | `55.0` |
| `reaching_fine` | `std=0.04` | `80.0` |
| `ee_orientation_coarse` | `std=0.50` | `30.0` |
| `ee_orientation_mid` | `std=0.15` | `40.0` |
| `ee_orientation_fine` | `std=0.08` | `70.0` |
| `action_rate` | `mdp.action_rate_l2` | `-10` |
| `collision_penalty` | `threshold=0.5` | `-1000.0` |

pose 环境仍保留 `collision` termination；success termination 的开关行为与 vision 一致。

## 低维 pose 策略训练

脚本：`scripts/train/train_pick_pose.py`

```bash
python -u scripts/train/train_pick_pose.py   --num_envs 1000   --headless   --max_iterations 1200
```

## 低维 pose 旧版 play 回放

仍可使用，但新评估建议用 `scripts/eval/eval_pick_pose.py`。

```bash
python -u scripts/train/train_pick_pose.py   --num_envs 1   --play   --checkpoint logs/xarm7_pick_pose/<time>/model_1199.pt   --play_max_steps 1200   --print_every 10
```

## 独立 eval 推理

脚本：`scripts/eval/eval_pick_pose.py`

支持手动设置工件在 robot base-link 下的位姿。
位置单位：m。四元数顺序：`qw,qx,qy,qz`。
`--max_steps` 是单条轨迹 / episode 的步数上限；`--max_episodes` 控制跑几条轨迹。

```bash
python -u scripts/eval/eval_pick_pose.py   --checkpoint logs/xarm7_pick_pose/2026-05-20_15-45-57/model_1200.pt   --max_steps 400  --max_episodes 1  --object_pose_base "0.3,0.1,0.05,0.0,1.0,0.0,0.0"
```

也可以拆成位置和欧拉角：

```bash
python -u scripts/eval/eval_pick_pose.py   --headless   --checkpoint logs/xarm7_pick_pose/<time>/model_1199.pt   --object_pos_base "0.0,-0.44,0.033"   --object_rpy_base_deg "0,0,0"
```

## 视觉策略训练

脚本：`scripts/train/train_pick_vision.py`

训练：

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

最小试跑：

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

回放：

```bash
source /home/gxai/miniconda3/etc/profile.d/conda.sh
conda activate isaaclab
cd /home/lsz/rl_robot/gx-VA-isaaclab

python -u scripts/train/train_pick_vision.py \
    --num_envs 1 \
    --play \
    --checkpoint logs/xarm7_pick_liftcube_vision/<time>/model_<iter>.pt
```
