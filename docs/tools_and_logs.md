# 工具和日志

## 通用工具

- `tools/policy_log_paths.py`：根据 checkpoint 时间组织日志路径。
- `tools/inspect_usd.py`：打印 USD 文件中的 prim 路径。
- `tools/utils/pick_pose_eval_utils.py`：eval/deploy 复用的数学和环境辅助函数。
- `tools/logs/pick_pose_csv.py`：pick-pose CSV schema 和写入逻辑。

查看 USD：

```bash
python tools/inspect_usd.py /path/to/file.usd
```

## Pick Pose CSV 日志

当前使用方：

- `scripts/train/train_pick_pose.py --play`
- `scripts/eval/eval_pick_pose.py`

两者共用：

```python
from tools.logs.pick_pose_csv import PickPoseCsvRow, open_pick_pose_csv
```

默认输出目录：

```text
test_log/xarm7_pick_pose-<policy_time>/
```

常见文件：

- `xarm7_pick_pose_<policy_time>.csv`
- `eval_pick_pose_<policy_time>.csv`

## 维护日志字段

修改日志字段时同步改：

- `PickPoseCsvRow`
- `build_pick_pose_header()`
- `format_pick_pose_row()`
- 本文档

## 字段分组

基本状态：

- `episode`：episode 编号
- `step`：episode 内步数
- `g_step`：全局步数
- `done`：是否结束
- `timeout`：是否超时

策略数据：

- `mdp_obs_00` ... `mdp_obs_20`：21 维策略观测
- `mdp_raw_action_j1` ... `mdp_raw_action_j7`：策略原始动作

误差和终止：

- `dist_cm`：TCP 到工件距离，单位 cm
- `ori_err_deg`：姿态误差，单位 deg
- `terminal_dist_cm`：done 瞬间缓存距离
- `terminal_ori_err_deg`：done 瞬间缓存姿态误差
- `terminal_success`：是否成功
- `terminal_source`：`cached_terminal` 或 `live_scene`

TCP 位姿：

- `tcp_x_m`, `tcp_y_m`, `tcp_z_m`：TCP 位置，单位 m
- `tcp_qw`, `tcp_qx`, `tcp_qy`, `tcp_qz`：TCP 姿态，四元数 wxyz

工件位姿：

- `object_x_m`, `object_y_m`, `object_z_m`：工件位置，单位 m
- `object_qw`, `object_qx`, `object_qy`, `object_qz`：工件姿态，四元数 wxyz

关节执行：

- `cmd_delta_j*_deg`：命令关节增量，单位 deg
- `actual_delta_j*_deg`：实际关节增量，单位 deg
- `delta_error_j*_deg`：`actual_delta - cmd_delta`
- `joint*_deg`：step 后关节角，单位 deg

格式：

- obs/action/TCP/object：8 位小数
- 距离、姿态误差：6 位小数
- 关节角和增量：6 位小数
- 缺失 terminal 值写空字符串
