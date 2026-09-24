# xArm7 Pick 任务 - Lift-Cube 风格大规模训练

## 训练配置

### 规模参数
- **迭代次数**: 200,000 次（20 万次）
- **并行环境**: 1,000 个
- **总训练步数**: 200,000 × 24 × 1,000 = **4.8B 步**（48 亿步）
- **保存间隔**: 每 1,000 次迭代保存一次模型

## 启动训练

### 完整训练
```bash
cd /home/gxai/IsaacLab/czr/zjj/usd_test

# 20 万次迭代，1000 个环境
python scripts/train_pick_liftcube.py --num_envs 1000 --headless
```

### 测试配置（快速验证）
```bash
# 1000 次迭代，100 个环境，验证配置正确
python scripts/train_pick_liftcube.py --num_envs 100 --headless --max_iterations 1000
```

### 从检查点恢复
```bash
python scripts/train_pick_liftcube.py --num_envs 1000 --headless \
  --resume --load_run logs/xarm7_pick_liftcube/2026-03-12_xx-xx-xx
```
## 监控指标

### 关键奖励
```
reaching_object:              0 → 1.0   (接近物体)
lifting_object:               0 → 15.0  (举起物体)
object_goal_tracking:         0 → 16.0  (粗粒度跟踪)
object_goal_tracking_fine:    0 → 5.0   (细粒度跟踪)
```

### 成功标准
- **10k 次**: reaching_object > 0.8
- **50k 次**: lifting_object > 5.0
- **100k 次**: lifting_object > 12.0
- **200k 次**: 总奖励 > 30.0

### 终止条件
```
time_out:         100% → 50%   (超时比例下降)
lift_success:     0% → 50%     (成功比例上升)
object_dropping:  0% → 0%      (保持为 0)
```

## 日志与模型

### 日志位置
```
logs/xarm7_pick_liftcube/YYYY-MM-DD_HH-MM-SS/
├── model_1000.pt      # 每 1000 次保存
├── model_2000.pt
├── ...
├── model_200000.pt
└── summaries/         # TensorBoard 日志
```

### 查看训练曲线
```bash
tensorboard --logdir logs/xarm7_pick_liftcube
```

### 测试模型
```bash
# 测试 10k 次迭代的模型
python scripts/train_pick_liftcube.py --num_envs 50 --play \
  --checkpoint logs/xarm7_pick_liftcube/xxx/model_10000.pt

# 测试最终模型
python scripts/train_pick_liftcube.py --num_envs 50 --play \
  --checkpoint logs/xarm7_pick_liftcube/xxx/model_200000.pt
```


## 优化建议

### 如果训练太慢
1. 减少环境数量（1000 → 512）
2. 减少迭代次数（200k → 100k）
3. 增大 save_interval（1000 → 5000）

### 如果 GPU 内存不足
1. 减少环境数量（1000 → 512 或 256）
2. 减小网络大小（[256,128,64] → [128,64,32]）

### 如果训练不收敛
1. 降低 lifting 阈值（0.86 → 0.83）
2. 增加 lifting 权重（15.0 → 30.0）
3. 增大动作 scale（0.2 → 0.3）

## 预期结果

### 最终性能
- **成功率**: > 80%
- **平均奖励**: > 30.0
- **举起时间**: < 3 秒

### 与原版对比
| 指标 | 原版（MuJoCo v4.1） | Lift-Cube 风格 |
|------|-------------------|---------------|
| 奖励设计 | max() 组合 | 加权求和 |
| 训练步数 | 4.9B | 4.8B |
| 夹爪学习 | 显式判断 | 隐式学习 |
| 目标跟踪 | 无 | 双层引导 |

## 注意事项

1. **训练时间长**：20 万次迭代需要 15-20 天，请确保：
   - 电源稳定
   - 系统不会自动休眠
   - 有足够的存储空间

2. **定期检查**：建议每天检查一次训练进度：
   ```bash
   # 查看最新日志
   tail -f logs/xarm7_pick_liftcube/xxx/log.txt

   # 查看 TensorBoard
   tensorboard --logdir logs/xarm7_pick_liftcube
   ```

3. **中断恢复**：如果训练中断，使用 `--resume` 恢复：
   ```bash
   python scripts/train_pick_liftcube.py --num_envs 1000 --headless \
     --resume --load_run logs/xarm7_pick_liftcube/xxx
   ```

4. **模型备份**：定期备份重要的检查点：
   ```bash
   # 备份 10k, 50k, 100k, 200k 的模型
   cp logs/xarm7_pick_liftcube/xxx/model_10000.pt backups/
   cp logs/xarm7_pick_liftcube/xxx/model_50000.pt backups/
   ```

## 开始训练

```bash
cd /home/gxai/IsaacLab/czr/zjj/usd_test
python scripts/train_pick_liftcube.py --num_envs 1000 --headless
```

祝训练顺利！🚀
