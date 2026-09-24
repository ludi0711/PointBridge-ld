# 日志分析

## 动作日志对比

脚本：`scripts/analysis/plot_action_log_compare.py`

```bash
python scripts/analysis/plot_action_log_compare.py   --input isaac_sim=test_log/xarm7_pick_pose-<time>/xarm7_pick_pose_<time>.csv   --input xarm7_sim=test_log/xarm7_pick_pose-<time>/sim2sim_deploy_state_pose_<time>.csv   --value cmd_delta_deg
```

常用 `--value`：

- `cmd_delta_deg`
- `raw_action`
- `actual_delta_deg`
- `obs`

## TCP / 工件位姿对比

脚本：`scripts/analysis/plot_tcp_object_pose_compare.py`

```bash
python scripts/analysis/plot_tcp_object_pose_compare.py   --input isaac_sim=test_log/xarm7_pick_pose-<time>/xarm7_pick_pose_<time>.csv   --input sdk_rollout=test_log/xarm7_pick_pose-<time>/sdk_obs_rollout_<time>.csv
```

## Eval vs sim2real HTML 轨迹对比

脚本：`scripts/analysis/plot_eval_deploy_html.py`

用于把一份 eval CSV 和一份 sim2real deploy CSV 画到同一个自包含 HTML 里，包含：

- 距离误差曲线：`dist_cm` vs `obj_rel_dist_mm / 10`
- 姿态误差曲线：`ori_err_deg` vs `obj_rel_ori_err_deg`
- 可拖动 3D 轨迹：TCP/EE 与目标位置 XYZ

默认读取当前调试日志目录，并用 `--eval_tcp_frame auto` 自动判断旧日志是否需要 target offset 校正：

```bash
python scripts/analysis/plot_eval_deploy_html.py --episode 0
```

指定输入和输出：

```bash
python scripts/analysis/plot_eval_deploy_html.py \
  --eval_csv test_log/xarm7_pick_pose-2026-05-18_20-02-24/eval_pick_pose_2026-05-18_20-02-24.csv \
  --deploy_csv test_log/xarm7_pick_pose-2026-05-18_20-02-24/sim2real_deploy_state_pose_2026-05-18_20-02-24_run02.csv \
  --episode 0 \
  --output test_log/xarm7_pick_pose-2026-05-18/eval_vs_sim2real_deploy_curves.html
```

旧 eval CSV 如果是在 `tcp_*` 里记录了 `ee_frame` 的 target offset 帧，而不是 source/link7 帧，也可以显式指定把 TCP 换回 source/link7：

```bash
python scripts/analysis/plot_eval_deploy_html.py \
  --eval_csv test_log/xarm7_pick_pose-2026-05-18_20-02-24/eval_pick_pose_2026-05-18_20-02-24.csv \
  --deploy_csv test_log/xarm7_pick_pose-2026-05-18_20-02-24/sim2real_deploy_state_pose_2026-05-18_20-02-24_run02.csv \
  --episode 0 \
  --eval_tcp_frame target-offset \
  --eval_tcp_offset_m 0.177 \
  --output test_log/xarm7_pick_pose-2026-05-18_20-02-24/eval_vs_sim2real_deploy_curves.html
```

查看 HTML：

```bash
cd /home/lsz/rl_robot/gx-VA-isaaclab
python -m http.server 8000
```

本地通过 VS Code 端口转发或 SSH `-L` 转发后打开：

```text
http://localhost:8000/test_log/xarm7_pick_pose-<time>/eval_vs_sim2real_deploy_curves.html
```

坐标系约定：

- 新版 `eval_pick_pose.py` 和 `train_pick_pose.py --play` 生成的 `tcp_x_m/tcp_y_m/tcp_z_m`、`object_x_m/object_y_m/object_z_m` 已统一为 robot `base_link` 坐标。
- 这里的 `base_link` 固化为 pick-pose 机器人 Robot root 坐标系，由 `robot.init_state.pos = (0.9, 4.6, 0.8)` 和 `robot.init_state.rot = (0.707, 0.0, 0.0, -0.707)` 定义，不再额外叠加任何坐标变换。
- 新版 `tcp_*` 记录的是 `ee_frame` source/link7，在没有额外 xArm TCP offset 时与 sim2real deploy CSV 的 `ee_*` 对齐；训练观测里的 `object_pose_relative_to_ee` 仍然使用带 `0.177m` offset 的 target TCP 帧。
- sim2real deploy CSV 的 `ee_x_mm/ee_y_mm/ee_z_mm` 是 xArm base 坐标，脚本会转成 m 后对比。
- sim2real deploy CSV 的 `obj_rel_x_mm/obj_rel_y_mm/obj_rel_z_mm` 是目标相对 EE/TCP 的向量，脚本会使用 `ee_qw/ee_qx/ee_qy/ee_qz` 旋回 base 坐标后得到目标位置。
- 旧版 eval/play CSV 里的 `tcp_*` 曾经可能是 `world - env_origin` 或 `ee_frame target offset`，不要直接和新版 sim2real HTML 轨迹混用；能重新跑 eval/play 时优先重新生成 CSV。
