# 部署和诊断

这些脚本会连接 xArm SDK。第一次运行建议加 `--dry_run` 和 `--step_mode`。

`--real` 只影响日志命名，不会自动切换 SDK 机器人模式。

## 闭环部署

脚本：`scripts/deploy/deploy_policy_pose.py`

读取 SDK 关节/TCP，重建 21D obs，策略推理后发送关节增量目标。

```bash
python -u scripts/deploy/deploy_policy_pose.py   --ckpt /home/lsz/rl_robot/gx-VA-isaaclab/logs/xarm7_pick_pose/2026-05-22_20-44-56/model_3900.pt  --ip 192.168.73.229   --step_mode --object_pos_base 0.30 0.10 0.05   --object_quat_base 0.0 1.0 0.0 0.0 --step 400
```

确认输出正常后再移除 `--dry_run`。

## FoundationPose + PickPose 实机全流程

这套流程用于 D435 头部相机 + FoundationPose 工件位姿 + `deploy_policy_pose.py`
闭环 reach + CTM2F110 夹爪夹取 + lift。

当前默认硬件参数：

- xArm IP：`192.168.73.229`
- D435 serial：`254622072913`
- FoundationPose shm：`foundationpose_multi_pose`
- 标定目录：`third_party/gx-foundationpose/camera_calibration/head_zero_d435_20260607_recalib`
- 工件 mesh：`/home/gxai/Desktop/cbd/FoundationPose/mesh/gongjian/gongjian.obj`
- 策略 ckpt：`logs/xarm7_pick_pose/2026-05-22_20-44-56/model_9999.pt`
- FP object frame 到训练 frame 的修正已固化在代码里：`[0.9984, 0.0334, 0.0149, -0.0436]`，正常部署不要再传 `--foundationpose_object_frame_y_deg -90`。

### 1. 清理旧 FP 进程

如果要重新启动 FP，先停掉旧进程：

```bash
pgrep -af 'foundationpose|FoundationPose|run_realtime.py'
kill -TERM <PID1> <PID2>
```

确认没有残留：

```bash
pgrep -af 'foundationpose|FoundationPose|run_realtime.py'
```

### 2. 确认相机和 TCP

确认 D435 serial：

```bash
rs-enumerate-devices | grep -E "Name|Serial Number"
```

确认 xArm 当前 TCP offset 和 TCP 位姿。这个 TCP 必须和部署时使用的一致：

```bash
cd /home/lsz/rl_robot/gx-VA-isaaclab

python3 - <<'PY'
import sys
sys.path.insert(0, "/home/lsz/rl_robot/gx-VA-isaaclab/xArm_Python_SDK_master")
from xarm.wrapper import XArmAPI

arm = XArmAPI("192.168.73.229")
try:
    print("err_warn:", arm.get_err_warn_code())
    print("tcp_offset:", arm.tcp_offset)
    print("position_deg:", arm.get_position(is_radian=False))
finally:
    arm.disconnect()
PY
```

CTM2F110 夹爪 payload 建议保持：`0.59015 kg`，CoG `[0, 0, 72] mm`。见下方“CRT CTM2F110 夹爪 payload”。

### 3. 重新手眼标定

先清空旧标定结果，只删标定目录下生成的 JSON：

```bash
cd /home/lsz/rl_robot/gx-VA-isaaclab

rm -f third_party/gx-foundationpose/camera_calibration/head_zero_d435_20260607_recalib/camera_to_right_arm_base.json \
      third_party/gx-foundationpose/camera_calibration/head_zero_d435_20260607_recalib/camera_to_j2.json \
      third_party/gx-foundationpose/camera_calibration/head_zero_d435_20260607_recalib/board_to_link7.json \
      third_party/gx-foundationpose/camera_calibration/head_zero_d435_20260607_recalib/T_j2_to_r_arm.json \
      third_party/gx-foundationpose/camera_calibration/head_zero_d435_20260607_recalib/calibration_raw_data.json
```

量 AprilTag 码本体正方形边长，单位米。`--tag-size` 必须是黑白码正方形边长，不是整张纸，也不含白边。

启动标定，下面以 `0.08 m` 为例，实际使用时替换成实测值。必须先 `cd` 到标定输出目录；脚本会把 JSON 写到当前目录。

```bash
cd /home/lsz/rl_robot/gx-VA-isaaclab/third_party/gx-foundationpose/camera_calibration/head_zero_d435_20260607_recalib
conda activate foundationpose

python -u /home/lsz/rl_robot/gx-VA-isaaclab/third_party/gx-foundationpose/camera_calibration/calibrate_with_robot.py \
  --ip 192.168.73.229 \
  --arm B \
  --tag-size 0.06 \
  --tag-id 0 \
  --camera-serial 254622072913 \
  --robot-pose-source sdk \
  --xarm-sdk-dir /home/lsz/rl_robot/gx-VA-isaaclab/xArm_Python_SDK_master \
  --stable-pos-std-mm 3 \
  --stable-rot-std-deg 1.0 \
  --capture-frames 20 \
  --capture-timeout-s 18
```

采集要求：

- 采 `15-20` 组。
- 每组位置和 wrist 姿态都要不同，不要只平移。
- tag 必须刚性固定在末端，不能被夹爪夹软、滑动或弯曲。
- tag 要清楚、尽量大、不要贴画面边缘。
- 每次按 `SPACE`/回车时，终端会打印 `[trigger] tcp at SPACE/ENTER ...` 和 `[capture] tcp used ...`，用来核对机械臂面板上的 TCP。

采完按 `q` 结束。合格结果应尽量接近：

- `RMS pos < 10 mm`
- `board-to-TCP position spread < 5-10 mm`

生成文件：

```bash
ls -lh
```

应包含：

- `camera_to_right_arm_base.json`
- `camera_to_j2.json`
- `board_to_link7.json`
- `T_j2_to_r_arm.json`
- `calibration_raw_data.json`

后续 FP 使用 `camera_to_right_arm_base.json`。如果标定时没有先 `cd` 到标定输出目录，结果会写在当前目录；启动 FP 的 `--camera_calib` 必须指向实际存在的那份 JSON。

### 4. 启动 FoundationPose

```bash
cd /home/lsz/rl_robot/gx-VA-isaaclab
conda activate foundationpose

python -u scripts/camera/run_realtime.py \
  --mesh_file /home/gxai/Desktop/cbd/FoundationPose/mesh/gongjian/gongjian.obj \
  --camera_calib third_party/gx-foundationpose/camera_calibration/head_zero_d435_20260607_recalib/camera_to_right_arm_base.json \
  --serial 254622072913 \
  --width 1280 \
  --height 720 \
  --est_refine_iter 2 \
  --track_refine_iter 1 \
  --timing_print_interval_s 0 \
  --udp_host 192.168.73.201 \
  --udp_port 5005
```

默认只发布本机共享内存；如果 FoundationPose 跑在 205，需要把位姿 UDP 转发给 201，就保留上面的 `--udp_host/--udp_port`。只做本机共享内存测试时可以删掉这两个参数。

窗口里按 `SPACE` 进入工件选择：左键点多边形顶点，右键结束当前工件，`SPACE`/回车开始 tracking。第一个工件对应 deploy 的 `--foundationpose_object_index 0`。

确认共享内存已创建：

```bash
ls -lh /dev/shm | grep foundationpose
```

### 5. 验证策略实际收到的工件位姿

另开终端：

```bash
cd /home/lsz/rl_robot/gx-VA-isaaclab
conda activate isaaclab

python3 scripts/deploy/monitor_foundationpose_obs.py \
  --ip 192.168.73.229 \
  --object-index 0 \
  --interval 0.5
```

看 `[4] object relative pose in TCP frame`：

- 最佳夹取点附近，`rel_pos_tcp_mm` 应接近 0。
- 这套代码默认已使用本次实机固化的 frame 修正，正常情况下不传 `--foundationpose-object-frame-y-deg -90`。
- 如果重新换工件 frame 或训练 frame，可以在最佳夹取姿态下看 `candidate_object_frame_quat_if_this_pose_is_ideal`，再用 `--foundationpose_object_frame_quat W X Y Z` 覆盖。

### 6. 自动 reach + 夹取 + lift

正式部署前确保 FP 已经 tracking，且 monitor 输出正常。

推荐使用 `--gripper_auto_close_mode z`：只等 TCP z 低于阈值再夹取，避免 dist/ori 阈值提前触发导致夹不到。

```bash
cd /home/lsz/rl_robot/gx-VA-isaaclab
conda activate isaaclab

python -u scripts/deploy/deploy_policy_pose.py \
  --ckpt logs/xarm7_pick_pose/2026-05-22_20-44-56/model_9999.pt \
  --ip 192.168.73.229 \
  --real \
  --foundationpose \
  --foundationpose_shm_name foundationpose_multi_pose \
  --foundationpose_object_index 0 \
  --foundationpose_max_age_s 0.5 \
  --enable_gripper \
  --gripper_auto_close \
  --gripper_auto_close_mode z \
  --gripper_trigger_dist_mm 20 \
  --gripper_trigger_ori_deg 15 \
  --gripper_trigger_steps 3 \
  --gripper_force_close_below_tcp_z_mm -10 \
  --gripper_init_open_speed 60 \
  --gripper_init_open_torque 60 \
  --gripper_close_speed 20 \
  --gripper_close_torque 40 \
  --gripper_close_hold_s 0.8 \
  --gripper_lift_mm 200 \
  --gripper_lift_speed 50 \
  --gripper_lift_acc 500 \
  --steps 1000
```

当前行为：

- 初始化时自动打开夹爪。
- 策略闭环 reach。
- `--gripper_auto_close_mode z` 下只在 `tcp_z <= -10 mm` 时夹取。
- 夹爪闭合后等待 `0.8s`。
- 沿 xArm base 的 z 方向抬升 `200mm`。
- lift 完成后退出 deploy。

### 6.1 eval 里的同步 lift

`scripts/eval/eval_pick_pose_crt_write.py` 现在也支持和真机 deploy 一样的 TCP z 向抬升：

- 默认 `--lift_mode cartesian_z`
- 默认 `--lift_height_mm 200`，也可以写成 `--gripper_lift_mm 200`
- 默认 `--lift_cartesian_step_mm 5`，每个控制步重新做一次小段 IK，lift 过程中保持夹爪当前朝向
- 内部使用 Pinocchio + `assets/crt_ctm2f110_gripper_visualization/external/xarm_description/meshes/xarm7.urdf`
- TCP offset 默认 `--lift_ik_tcp_z_offset_m 0.177`，和 `ee_frame` 的 177mm 保持一致
- 旧的 joint2 抬升保留为 fallback：`--lift_mode joint2_delta --lift_joint2_delta_deg -8`

示例：

```bash
python -u scripts/eval/eval_pick_pose_crt_write.py \
  --checkpoint logs/xarm7_pick_pose/2026-05-22_20-44-56/model_9999.pt \
  --dynamic_object_eval \
  --auto_start_policy \
  --lift_after_auto_close \
  --lift_mode cartesian_z \
  --lift_height_mm 200 \
  --lift_cartesian_step_mm 5
```

如果想人工确认每一步，在命令末尾加：

```bash
--step_mode
```

### 7. 常见问题

- deploy 报 `FileNotFoundError: /foundationpose_multi_pose`：FP 没启动、还没 tracking，或共享内存被旧进程/resource tracker 删掉。杀掉旧 FP 后重启。
- 最佳夹取点位置对，但姿态差约 `90°`：检查 frame 修正。当前默认使用固化 quat，不要再叠加 `--foundationpose_object_frame_y_deg -90`。
- 还没到 z 阈值就夹：确认命令里有 `--gripper_auto_close_mode z`。默认 `either` 会被 dist/ori 阈值提前触发。
- 标定后 base 位姿偏很多：优先检查 `--tag-size` 是否量了 AprilTag 码本体边长、tag 是否刚性固定、标定 residual 是否大于 `10mm`。
- 夹爪启动状态不一致：deploy 带 `--enable_gripper` 时会在初始化阶段自动 open。

## CRT CTM2F110 夹爪 payload

当前实机使用 CRT CTM2F110 夹爪时，xArm manual mode 如果 payload 仍是默认
`[0, [0, 0, 0]]`，容易触发 `C37 Abnormal movement in Manual Mode`。已按
`crt_ctm2f110_gripper_visualization` 里的 URDF/USD 质量数据计算并在实机回读确认：

- 质量：`0.59015 kg`
- 重心：`[0.0, 0.0, 72.0] mm`
- 重力方向：`[0.0, 0.0, -1.0]`
- 安装方向：`[0.0, -0.0]`

计算来源：

- `crt_ctm2f110_gripper_visualization/urdf/urdf_sync/crt_ctm2f110.xacro`
- `crt_ctm2f110_gripper_visualization/urdf/urdf_sync/crt_ctm2f110_model_macro.xacro`
- 总质量约 `0.590150 kg`
- 手指开合导致质心 z 约在 `69.4~71.9 mm`，实机使用 `72.0 mm`

运行时设置，不保存到控制器：

```bash
cd /home/lsz/rl_robot/gx-VA-isaaclab

python3 - <<'PY'
import sys
import time

sys.path.insert(0, "/home/lsz/rl_robot/gx-VA-isaaclab/third_party/xArm_Python_SDK_master")
from xarm.wrapper import XArmAPI

arm = XArmAPI("192.168.73.229", is_radian=True, enable_report=True, report_type="rich")
try:
    time.sleep(0.5)
    print("before tcp_load:", arm.tcp_load)
    print("set_tcp_load:", arm.set_tcp_load(0.59015, [0.0, 0.0, 72.0], wait=True))
    time.sleep(0.5)
    print("after tcp_load:", arm.tcp_load)
    print("err_warn:", arm.get_err_warn_code())
finally:
    arm.disconnect()
PY
```

稳定验证后再永久保存：

```bash
cd /home/lsz/rl_robot/gx-VA-isaaclab

python3 - <<'PY'
import sys

sys.path.insert(0, "/home/lsz/rl_robot/gx-VA-isaaclab/third_party/xArm_Python_SDK_master")
from xarm.wrapper import XArmAPI

arm = XArmAPI("192.168.73.229", is_radian=True)
try:
    print("set_tcp_load:", arm.set_tcp_load(0.59015, [0.0, 0.0, 72.0], wait=True))
    print("save_conf:", arm.save_conf())
finally:
    arm.disconnect()
PY
```

如果末端还额外挂了 AprilTag 板、转接件或工件，需要把额外质量和重心重新合并后再设置。

## 从 obs CSV 开环回放

脚本：`scripts/deploy/deploy_policy_pose_from_obs_log.py`

从 play CSV 读取 `mdp_obs_00..20`，送入策略，再把动作发给 xArm。

```bash
python -u scripts/deploy/deploy_policy_pose_from_obs_log.py   --ckpt logs/xarm7_pick_pose/<time>/model_1199.pt   --obs_csv test_log/xarm7_pick_pose-<time>/xarm7_pick_pose_<time>.csv   --ip 192.168.73.229   --dry_run   --step_mode   --steps 10
```

## Isaac 首帧 + SDK 反馈滚动

脚本：`scripts/deploy/rollout_isaac_obs_via_xarm_sdk.py`

```bash
python -u scripts/deploy/rollout_isaac_obs_via_xarm_sdk.py   --ckpt logs/xarm7_pick_pose/<time>/model_1199.pt   --isaac_csv test_log/xarm7_pick_pose-<time>/xarm7_pick_pose_<time>.csv   --ip 192.168.73.229   --dry_run   --step_mode   --steps 10   --object_mode isaac   --action_source policy
```

常用选项：

- `--object_mode isaac`：使用 Isaac CSV 中的工件相对位姿
- `--object_mode sdk`：用 SDK TCP 和工件 base pose 重建
- `--action_source policy`：重新推理动作
- `--action_source logged`：使用 Isaac 日志里的动作

## 动作跟踪测试

脚本：`scripts/deploy/test_policy_pose_action_tracking.py`

比较命令关节增量和 SDK 实测关节增量。

```bash
python -u scripts/deploy/test_policy_pose_action_tracking.py   --ckpt logs/xarm7_pick_pose/<time>/model_1199.pt   --ip 192.168.73.229   --dry_run   --steps 50   --object_pos_base 0.20 0.00 0.12   --object_quat_base 0.0 1.0 0.0 0.0
```
