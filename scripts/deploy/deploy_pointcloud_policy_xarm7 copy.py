#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""部署 xArm7 stage2 点云策略：D435 深度 → 基座系点云 → PointNet → 关节增量伺服。

完整链路::

    RealSense D435 (640x480@30, 深度硬件对齐到彩色)
      → 掩码：手标多边形（--mask_source polygon，默认）
              或 YOLO-seg（--mask_source yolo，需人工确认后后台线程持续刷新）
      → mask_depth_to_pointcloud()      ← **仿真训练时用的同一个函数**
            确定性 FPS → 反投影 → 基座系 → 工作空间裁剪 → 噪声
      → 物体点 64x3 (192)
    xArm7 编码器
      → joint_pos(7) → joint_pos_rel = q - q_default   (7)
      → TCP 位姿 → gripper_keypoints() 正运动学 → 夹爪点 6x3 (18)
      → 217 维观测 = 物体点(192) ⊕ 夹爪点(18) ⊕ joint_pos_rel(7)
      → PointNetActorCritic.act_inference → 7 维关节增量
      → delta_q = clip(raw * scale, ±clip)，scale = clip = 0.6°
        q_target = clip(q + delta_q, 关节限位)
      → set_servo_angle_j，50 Hz (dt=0.02s = sim dt 0.01 x decimation 2)

与 privileged_base 部署脚本（GSworld/.../deploy_pose_policy_xarm7.py）的**关键差异**
—— 照抄那份会错，这三条都是"不报错、只是行为悄悄不对"的类型：

    1. **actor 不做 obs 归一化**。训练配置 ``actor_obs_normalization=False``
       （rsl_rl_ppo_cfg.py:1454，理由：EmpiricalNormalization 逐维统计，而 FPS 后
       第 k 个点无固定语义，且它会削掉绝对位置尺度 —— 绝对位置正是全部任务信息）。
       checkpoint 里也确实只有 ``critic_obs_normalizer.*``。观测**保持米制原值**
       直接进网络。
    2. **点云归一化在网络内部**。``point_norm_min/point_norm_range`` 是 registered
       buffer，``_encode_actor`` 对物体点和夹爪点用**同一套** min/range 做仿射。
       外部预先归一化会归两次。
    3. **最后 7 维是 joint_pos_rel 而非绝对关节角**：``mdp.joint_pos_rel`` =
       当前关节角 − 默认关节角（env_cfg.py:1194）。喂绝对角度会整体偏一个常量。

策略类**直接实例化训练时那个 PointNetActorCritic**，不手工重搭网络：点云
encoder 是双 token 共享权重结构，手搭极易在切分偏移或共享关系上出错，而那种错
会得到一个照样能跑、只是行为不对的网络。

安全（沿用参考脚本的那套，逐条对齐）:

    - 点云无效（掩码内无有效深度 / visible < 门槛）→ hold，不输出新动作
    - 关节限位外（留 1° 裕量）→ 拒发该步
    - 任何异常 → set_state(0) 停止，切回 Mode 0 再断开
    - **步进模式**（``--step_mode``，首次上机强烈建议）：启动即暂停，按空格执行
      ``--step_n`` 步后自动暂停并 hold
    - ``--dry_run`` 全链路跑通但**不发任何指令**，先用它验证观测数值

深度冻结（``--freeze_depth`` / 运行中按 ``f``）：把当前深度图留存，此后物体点
都从这张不变的图算。**闭环下这意味着观测与现实脱节** —— 工件被挪走，点还停在
原处，机械臂会朝记忆中的位置去抓。仅用于"工件确定不动、想排除深度噪声"的对照
实验，画面和日志都会持续标红。按 ``u`` 解冻。

掩码来源二选一（``--mask_source``）:

    polygon  手标多边形，默认。见下面"工件掩码"一节。
    yolo     YOLO-seg 自动分割，与训练 stage 3 调**同一个** ``yolo_masks``。
             启动时跑一次并**等人工确认**（y 接受 / r 重检 / c 降门限 / q 取消）
             —— YOLO 可能把桌面反光或夹爪当成工件，而那类错误不报错：掩码有面积、
             深度也有效，反投影出的点云看着正常只是位置错了。确认后由后台线程持续
             刷新掩码（不占控制环时间），空检出保留上一张（零阶保持）。
             ``--yolo_freeze`` 可锁定确认时那一张不再刷新。

             上机前先用 ``scripts/camera/test_yolo_realtime.py`` 看真机画面上的
             检出率与 valid 像素数 —— 那份脚本的 20260815.pt 指标是在仿真渲染图上
             统计的，真机的曝光与反光不一样。

工件掩码：**默认每次启动现场标注**（左键沿轮廓点一圈，回车确认）。挪动工件做测试
时这是最省事的路径 —— 挪完直接跑，当场圈一下即可，不用先跑采集脚本存文件再传路径。
标完会自动存一份带时间戳的备份到 logs/deploy_pointcloud/，出问题能复现当时用的掩码。
工件位置不变时可以用 ``--polygon <json>`` 复用上次的，跳过标注。

注意多边形是**像素坐标**，绑死在画面位置上：工件挪了而仍用旧标注，掩码会框到空桌面，
且**不会报错** —— 要么 visible 不足持续 hold，要么框到桌面算出一朵假点云喂给策略。
所以挪过工件就别加 ``--polygon``。

启动顺序刻意是"相机 → 标注 → 连机械臂"：标注要花时间，放在使能之后机械臂就得带电
停在初始位姿干等，且标注中途退出会变成一次带电异常退出。

用法::

    conda activate gx_va_deploy

    # ① 先干跑，不连机械臂不发指令，只看观测数值对不对（现场标注）
    python scripts/deploy/deploy_pointcloud_policy_xarm7.py \
        --ckpt logs/xarm7_pick_pointcloud_stage2/2026-08-13_19-10-02/model_12000.pt \
        --dry_run

    # ② 上机，步进模式，一次一步
    python scripts/deploy/deploy_pointcloud_policy_xarm7.py \
        --ckpt logs/xarm7_pick_pointcloud_stage2/2026-08-13_19-10-02/model_12000.pt \
        --robot_ip 192.168.73.229 --step_mode

    # ③ 连续运行
    python scripts/deploy/deploy_pointcloud_policy_xarm7.py \
        --ckpt logs/xarm7_pick_pointcloud_stage2/2026-08-13_19-10-02/model_12000.pt \
        --robot_ip 192.168.73.229

    # ④ 工件没挪，复用上次标注跳过手标
    python scripts/deploy/deploy_pointcloud_policy_xarm7.py \
        --ckpt logs/.../model_12000.pt --robot_ip 192.168.73.229 \
        --polygon logs/deploy_pointcloud/polygon_20260814_143000.json

    # ⑤ YOLO 掩码替掉手标（启动时人工确认），先干跑
    python scripts/deploy/deploy_pointcloud_policy_xarm7.py \
        --ckpt logs/.../model_40300.pt --mask_source yolo --dry_run

    # ⑥ YOLO 掩码上机，步进模式
    python scripts/deploy/deploy_pointcloud_policy_xarm7.py \
        --ckpt logs/.../model_40300.pt --mask_source yolo \
        --robot_ip 192.168.73.229 --step_mode

录像：**默认开**，控制环一起来就录，存到 logs/deploy_pointcloud/video_<时间戳>.mp4。
录的是相机**纯 RGB**，不含点云投影、轮廓、状态文字那些叠加层 —— 所以录下来的视频
可以直接回灌 YOLO 或重新标注，等价于当时喂给策略的真实输入。只在新帧到达时写一帧，
视频时长≈真实时长，不受步进模式暂停或控制环空转影响。``--record <路径>`` 改输出位置，
``--no_record`` 关掉。异常退出也会走 release —— 少了这步 mp4 缺 moov box 整个文件打不开。

标注键位：左键加点  右键/退格删点  回车确认(>=3点)  r 重来  q 取消
运行键位：SPACE=暂停/继续（步进模式下=执行N步）  R=回初始位姿  H=手动hold
          F=冻结深度  U=解冻  S=截图  Q=退出
"""

from __future__ import annotations

import argparse
import csv
import math
import os
import select
import sys
import threading
import time
from pathlib import Path

import numpy as np

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))
sys.path.insert(0, _PROJECT_DIR)


# ── 常量（与训练配置严格一致，勿改） ─────────────────────────────────────────
ACT_DIM = 7
CONTROL_DT = 0.02          # sim dt 0.01 x decimation 2 = 50Hz

# 动作缩放：configs/xarm7_pick_vision_spatial_camfix_env_cfg.py 的
# _ARM_ACTION_SCALE / _ARM_CLIP_RAD，两者都是 radians(0.6)
TRAIN_ACTION_SCALE_DEG = 0.6
TRAIN_ACTION_CLIP_DEG = 0.6

# 默认关节角 = 训练时的 INIT_JOINT_POS（同上文件 line 140）。
# joint_pos_rel 以它为基准，**改这里等于改观测**。
DEFAULT_JOINT_POS_DEG = [-0.4, -58.4, -0.1, 17.5, -0.2, 75.8, -0.2]
# 真机运动起始位姿。与 DEFAULT_JOINT_POS_DEG 数值相同（训练即从此处起步），
# 但语义不同：一个是 obs 基准，一个是开机回零目标，分开写以免日后改错。
INIT_JOINT_POS_DEG = [-0.4, -58.4, -0.1, 17.5, -0.2, 75.8, -0.2]
INIT_SPEED_DEG = 20.0

JOINT_LIMITS_LOW_DEG = [-180, -118, -180, -11, -97, -180, -180]
JOINT_LIMITS_HIGH_DEG = [180, 118, 180, 225, 97, 180, 180]
JOINT_LIMITS_LOW_RAD = np.deg2rad(JOINT_LIMITS_LOW_DEG).astype(np.float32)
JOINT_LIMITS_HIGH_RAD = np.deg2rad(JOINT_LIMITS_HIGH_DEG).astype(np.float32)

# 标定外参（**基座系**原始值）。与 real_roi_pointcloud.py 同源，启动时核对。
#
# 2026-09-14 同步 2026-09-01 重标（同一台相机 serial 254622072913，源文件
# /home/lsz/gx_pose/config/camera_to_base.json）。用了旧标定的 checkpoint 直接
# 部署会整体偏 11.34 mm —— 不报错，只让点云悄悄错位。
CALIB_CAM_POS_IN_BASE_M = (0.39141, -0.781319, 0.588075)
CALIB_CAM_QUAT_IN_BASE_WXYZ = (-0.523262, 0.851472, 0.006784, -0.033852)


# ── 自动寻找 xArm SDK ────────────────────────────────────────────────────────
def _add_xarm_sdk_to_path():
    candidates = []
    env_path = os.environ.get("XARM_SDK_DIR", "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())
    candidates += [
        Path("/home/gxai/Desktop/Deploy/third_party/xArm_Python_SDK_master"),
        Path(_PROJECT_DIR) / "third_party" / "xArm_Python_SDK_master",
    ]
    for candidate in candidates:
        if (candidate / "xarm").exists():
            sys.path.insert(0, str(candidate))
            print(f"[SDK] 使用 xArm SDK: {candidate}")
            return
    print("[SDK] 未找到 xArm SDK，可用 export XARM_SDK_DIR=... 指定（--dry_run 不需要）")


_add_xarm_sdk_to_path()

try:
    from xarm.wrapper import XArmAPI
except ImportError:
    XArmAPI = None


def quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """RPY(rad) → 四元数 wxyz。xArm get_position 给的是 RPY。"""
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    return np.array([
        cr * cp * cy + sr * sp * sy,
        sr * cp * cy - cr * sp * sy,
        cr * sp * cy + sr * cp * sy,
        cr * cp * sy - sr * sp * cy,
    ], dtype=np.float32)


# ── 策略 ─────────────────────────────────────────────────────────────────────
class PointCloudPolicy:
    """217 维点云观测 → 7 维关节增量。直接复用训练时的 PointNetActorCritic。

    **不手工重搭网络**：点云 encoder 是"物体点和夹爪点各过一次同一个 encoder"的
    双 token 共享权重结构，手搭很容易在切分偏移或权重共享上出错，而那种错会得到
    一个照样能前向、只是行为不对的网络 —— 在真机上表现为机械臂动作诡异但不报错。
    """

    def __init__(self, ckpt_path: str, device: str = "cuda"):
        import torch

        from configs.point_bridge_pointcloud import (
            M_OBJ, N_ROBOT, NUM_POINTS, POINT_DIM,
        )
        from model.pointnet_actor_critic import PointNetActorCritic

        print(f"[策略] 加载: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt["model_state_dict"]

        self.obs_dim = NUM_POINTS * POINT_DIM + 7      # 192 + 18 + 7 = 217
        actor_in = state_dict["actor.0.weight"].shape[1]
        critic_in = state_dict["critic.0.weight"].shape[1]

        # actor.0 的输入 = 两个 encoder embedding + 低维向量。
        # 对不上说明 checkpoint 不是这套配置训出来的，早报错好过在真机上试。
        emb = state_dict["point_encoder.proj.weight"].shape[0]
        expect = emb * 2 + 7
        if actor_in != expect:
            hint = ""
            if actor_in == emb * 3 + 7:
                # stage 10 是三个 embedding（物体 / 夹爪 / 腕部）。真机上没有腕部
                # 相机，喂不出第三路点云，所以这里只能明确拒绝而不是凑维度。
                hint = (
                    "\n  形状恰好等于 embedding x3 + 7 —— 这看起来是 **stage 10** 的权重"
                    "（多一路腕部相机点云）。本脚本没有腕部相机数据源，无法部署；"
                    "请用 stage 2/3/4 的权重。"
                )
            raise SystemExit(
                f"[策略] checkpoint 与本脚本假设的观测布局不符：\n"
                f"  actor.0 输入 {actor_in}，按 embedding {emb}x2 + 7 维低维应为 {expect}\n"
                f"  这个脚本只支持 stage2 点云策略（物体点 {M_OBJ} + 夹爪点 {N_ROBOT}）"
                f"{hint}"
            )

        # PointNetActorCritic 从 obs 字典**推断**维度（它按 obs[group].shape[-1]
        # 累加），没有 num_actor_obs 这类参数。部署时没有真环境，用形状正确的
        # 零张量喂进去即可 —— 构造只读 shape，不读数值。
        dummy_obs = {
            "policy": torch.zeros(1, self.obs_dim),
            "critic": torch.zeros(1, critic_in),
        }
        self.model = PointNetActorCritic(
            obs=dummy_obs,
            obs_groups={"policy": ["policy"], "critic": ["critic"]},
            num_actions=ACT_DIM,
            num_object_points=M_OBJ,
            num_robot_points=N_ROBOT,
            point_dim=POINT_DIM,
            actor_obs_normalization=False,   # 与训练一致，见模块 docstring 第 1 条
            critic_obs_normalization=True,
            actor_hidden_dims=[256, 256],
            critic_hidden_dims=[512, 256, 128],
            activation="elu",
        )
        missing, unexpected = self.model.load_state_dict(state_dict, strict=False)
        # 允许 critic/std 相关的差异（部署用不到），但 actor 与 encoder 必须齐全
        crit = [k for k in missing if k.startswith(("actor.", "point_encoder.", "point_norm"))]
        if crit:
            raise SystemExit(f"[策略] actor/encoder 权重缺失: {crit}")
        if missing:
            print(f"[策略] 忽略缺失(非 actor 路径): {missing}")
        if unexpected:
            print(f"[策略] 忽略多余键: {unexpected}")

        self.model.to(device).eval()
        self.device = device
        print(f"[策略] 加载完成 obs_dim={self.obs_dim} act_dim={ACT_DIM} "
              f"iter={ckpt.get('iter')}  embedding={emb}")
        print(f"[策略] 点云归一化 min={self.model.point_norm_min.tolist()} "
              f"range={self.model.point_norm_range.tolist()}  (网络内部完成)")

    def predict(self, obs_np: np.ndarray) -> np.ndarray:
        import torch

        if obs_np.shape != (self.obs_dim,):
            raise ValueError(f"[策略] obs 维度错误: {obs_np.shape} != ({self.obs_dim},)")
        obs = torch.as_tensor(obs_np, dtype=torch.float32,
                              device=self.device).unsqueeze(0)
        with torch.no_grad():
            # act_inference 收的是**按观测组分键的字典**（get_actor_obs 会
            # torch.cat(obs[group] for group in obs_groups["policy"])），不是扁平
            # 张量。内部会走 actor_obs_normalizer（此处为 Identity）再 _encode_actor
            # （内部做点云归一化），与训练前向完全同路。
            action = self.model.act_inference({"policy": obs})
        return action.squeeze(0).cpu().numpy().astype(np.float32)


# ── 相机后台取流（控制环 50Hz 不被 30fps 相机阻塞） ───────────────────────────
class ThreadedFrameGrabber:
    def __init__(self, camera):
        self.camera = camera
        self._lock = threading.Lock()
        self._latest = None       # (rgb, depth, seq)
        self._seq = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="rs-grabber", daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            try:
                rgb, depth = self.camera.get_frame()
                if rgb is not None and depth is not None:
                    with self._lock:
                        self._seq += 1
                        self._latest = (rgb, depth, self._seq)
            except RuntimeError:
                time.sleep(0.005)

    def latest(self):
        with self._lock:
            return self._latest

    def stop(self):
        self._stop.set()


# ── xArm 工具 ────────────────────────────────────────────────────────────────
def get_joint_positions(arm) -> np.ndarray:
    code, raw = arm.get_servo_angle(is_radian=True)
    if code != 0 or raw is None:
        raise RuntimeError(f"[xArm] get_servo_angle 失败: code={code}")
    return np.asarray(raw[:7], dtype=np.float32)


def get_tcp_pose_base(arm):
    """TCP 位姿（基座系）：位置 m，姿态 wxyz。控制器上需已配置 TCP offset。"""
    code, pose = arm.get_position(is_radian=True)
    if code != 0 or pose is None:
        raise RuntimeError(f"[xArm] get_position 失败: code={code}")
    pose = np.asarray(pose[:6], dtype=np.float32)
    return (pose[:3] * 0.001).astype(np.float32), quat_from_euler_xyz(
        float(pose[3]), float(pose[4]), float(pose[5]))


def enter_position_mode(arm):
    arm.set_mode(0); arm.set_state(0); time.sleep(0.2)
    print("[xArm] 已切换 Mode 0")


def enter_servo_mode(arm):
    arm.set_mode(1); arm.set_state(0); time.sleep(0.3)
    print("[xArm] 已切换 Mode 1 连续伺服")


def recover_servo_mode(arm):
    print("[xArm] 尝试恢复 Mode 1 ...")
    arm.clean_error(); arm.clean_warn(); arm.motion_enable(enable=True)
    arm.set_mode(1); arm.set_state(0); time.sleep(0.1)


def check_joint_safe(joint_rad: np.ndarray, margin_deg: float = 1.0) -> bool:
    m = np.deg2rad(margin_deg)
    return bool(np.all(joint_rad > JOINT_LIMITS_LOW_RAD + m)
                and np.all(joint_rad < JOINT_LIMITS_HIGH_RAD - m))


# ── 观测构建 ─────────────────────────────────────────────────────────────────
def build_obs(object_points: np.ndarray, ee_pos_m: np.ndarray,
              ee_quat_wxyz: np.ndarray, joint_rad: np.ndarray,
              default_joint_rad: np.ndarray) -> "tuple[np.ndarray, np.ndarray]":
    """217 维观测：物体点(192) ⊕ 夹爪点(18) ⊕ joint_pos_rel(7)。

    布局与 env_cfg.py 的 PolicyCfg 逐项对齐 —— 顺序由 ObsGroup 里字段的声明顺序
    决定，网络按**固定偏移**切分点云，错一位整条链路就废了。

    夹爪点由 TCP 位姿正运动学算出（``gripper_keypoints``，与仿真同一函数），
    不经过相机。所以即使深度被冻结，这 18 维仍然是实时的 —— 否则策略连自己的
    夹爪往哪走都看不见，闭环直接失效。

    最后 7 维是 ``joint_pos_rel`` = 当前关节角 − 默认关节角，**不是绝对角度**。

    Returns:
        ``(obs 217维, robot_pts 6x3)``，后者供可视化叠加用。
    """
    import torch

    from configs.point_bridge_pointcloud import gripper_keypoints

    robot_pts = gripper_keypoints(
        torch.from_numpy(ee_pos_m.astype(np.float32)).unsqueeze(0),
        torch.from_numpy(ee_quat_wxyz.astype(np.float32)).unsqueeze(0),
        noise_std=0.0,          # 参考实现只对物体点加噪，夹爪点来自 FK 不加
    )[0].numpy()

    joint_pos_rel = (joint_rad - default_joint_rad).astype(np.float32)
    obs = np.concatenate([
        object_points.reshape(-1).astype(np.float32),   # 64 x 3 = 192
        robot_pts.reshape(-1).astype(np.float32),       #  6 x 3 =  18
        joint_pos_rel,                                  #           7
    ]).astype(np.float32)
    return obs, robot_pts


# ── YOLO 掩码（--mask_source yolo）─────────────────────────────────────────────
def yolo_mask_once(rgb: np.ndarray, weights: str, conf: float, device: str
                   ) -> "tuple[np.ndarray, float, int]":
    """对单帧 RGB 跑一次 YOLO，返回 ``(mask, best_conf, n_det)``。

    调 ``configs/yolo_mask_source.py`` 的 ``yolo_masks`` —— **与训练 stage 3 同一个
    函数**。掩码来源是这条路线唯一变的东西，它必须只有一份实现，否则真机与训练的
    掩码分布又会悄悄分叉，那正是 stage 3 要消掉的 gap。

    conf 与检出数是从同一次推理的 Results 里顺手取的诊断量（不进观测），用于确认
    界面和日志。
    """
    import torch

    from configs.yolo_mask_source import yolo_masks

    rgb_t = torch.from_numpy(rgb).unsqueeze(0).to(device)
    # return_details 让 conf / 检出数从**同一次**推理里带出来。分两次跑会把延迟
    # 翻倍，而 50Hz 控制环对延迟是硬约束。
    mask_t, _, det = yolo_masks(rgb_t, weights=weights, conf=conf,
                               return_details=True)
    return (mask_t[0].cpu().numpy(),
            float(det["best_conf"][0]),
            int(det["n_det"][0]))


def confirm_yolo_mask(camera, weights: str, conf: float, device: str,
                      min_visible: int, max_wait_s: float = 300.0):
    """跑 YOLO 出掩码，**等用户人工确认**后才返回。取消返回 None。

    为什么必须人工确认：YOLO 在真机上可能把桌面反光、夹爪、旁边的杂物当成工件，
    而这类错误**不会报错** —— 掩码有面积、深度也有效，反投影出的点云看起来完全
    正常，只是位置错了，然后策略照着错的点云去抓。手标多边形至少是人画的，换成
    YOLO 就必须补一道人眼确认。

    与手标一样放在**连接机械臂之前**：确认要花时间，放在使能之后机械臂就得带电
    干等，且中途取消会变成一次带电异常退出。

    返回 ``(mask, polygon_like_contour, best_conf, n_det)``；``polygon_like_contour``
    是掩码最大外轮廓，仅供运行时画绿线用（不参与采样）。

    键位：y/回车 接受   r 重新检测   c 调低门限重试   q 取消
    """
    import cv2

    win = "confirm YOLO mask"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    print("\n[确认] YOLO 掩码需人工确认：y/回车 接受   r 重新检测   "
          "c 降低 conf 门限   q 取消")

    cur_conf = float(conf)
    t0 = time.time()
    mask = None
    best_conf = 0.0
    n_det = 0
    need_infer = True

    try:
        while time.time() - t0 < max_wait_s:
            rgb, depth = camera.get_frame()
            if rgb is None or depth is None:
                continue

            if need_infer:
                mask, best_conf, n_det = yolo_mask_once(rgb, weights, cur_conf, device)
                need_infer = False
                area = int(mask.sum())
                valid = int((mask & np.isfinite(depth) & (depth > 0)).sum())
                print(f"[检测] det={n_det} conf={best_conf:.3f} area={area}px "
                      f"valid={valid} (conf_th={cur_conf:.2f})")

            area = int(mask.sum())
            valid = int((mask & np.isfinite(depth) & (depth > 0)).sum())

            disp = rgb[..., ::-1].copy()
            if area > 0:
                ov = disp.copy()
                ov[mask] = (0, 0, 255)
                disp = cv2.addWeighted(disp, 0.7, ov, 0.3, 0)
                cnts, _ = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(disp, cnts, -1, (0, 255, 0), 2)
                ys, xs = np.nonzero(mask)
                cv2.circle(disp, (int(xs.mean()), int(ys.mean())), 5, (255, 255, 0), -1)

            ok = n_det > 0 and valid >= min_visible
            hint = (f"det={n_det} conf={best_conf:.3f} area={area}px valid={valid} "
                    f"th={cur_conf:.2f} | y:accept r:redo c:lower-th q:cancel")
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 22), (0, 0, 0), -1)
            cv2.putText(disp, hint, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.40,
                        (255, 255, 255) if ok else (0, 200, 255), 1, cv2.LINE_AA)
            if not ok:
                msg = ("NO DETECTION" if n_det == 0
                       else f"valid {valid} < min_visible {min_visible}")
                cv2.rectangle(disp, (0, 22), (disp.shape[1], 44), (0, 0, 200), -1)
                cv2.putText(disp, f"!! {msg} -- do NOT accept !!", (5, 38),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.42, (255, 255, 255), 1,
                            cv2.LINE_AA)
            cv2.imshow(win, disp)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("y"), ord("Y"), 13, 10):
                if not ok:
                    # 明确拒绝放行：接受一张空掩码等于让策略吃零阶保持的旧点云，
                    # 或者直接吃一朵位置错误的点云 —— 都不该在上机前被默许。
                    print(f"[确认] 当前掩码不合格（det={n_det} valid={valid}），"
                          f"不允许接受。按 r 重试或 c 降门限，q 取消。")
                    continue
                cnts, _ = cv2.findContours(mask.astype(np.uint8),
                                           cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                contour = max(cnts, key=cv2.contourArea).reshape(-1, 2) if cnts else None
                print(f"[确认] 已接受 YOLO 掩码 area={area}px valid={valid} "
                      f"conf={best_conf:.3f}")
                return mask, contour, best_conf, n_det, cur_conf
            elif key == ord("r"):
                need_infer = True
            elif key == ord("c"):
                cur_conf = max(0.05, round(cur_conf - 0.05, 2))
                need_infer = True
                print(f"[门限] conf={cur_conf:.2f}（调低会增加误检，确认画面再接受）")
            elif key in (27, ord("q")):
                return None
        print("[确认] 超时未确认")
        return None
    finally:
        cv2.destroyWindow(win)


class YoloMaskWorker:
    """后台线程持续跑 YOLO 刷新掩码。控制环只读最新一张，不等推理。

    **为什么必须异步**：50Hz 控制周期只有 20ms，而 YOLO 单帧在 5090 上约 5~15ms、
    真机分辨率下可能更久（用 ``test_yolo_realtime.py`` 量出的 p95 就是判据）。放在
    控制环里同步跑，一旦超过 20ms 整个伺服节拍就被拖慢，机械臂动作会变顿。

    异步的语义代价：掩码比深度**旧几帧**。工件静止时这不影响什么（掩码本来就几乎
    不变）；这也正是训练 stage 4 的语义 —— 每 N 步刷一次掩码、深度每步都新，那份
    docstring 里写明"真机部署本来也是 50Hz 控制配 10Hz 的 YOLO 更新"。

    掩码**只在检出合格时才更新**：空检出直接丢弃、保留上一张（等价于规格 §11.1 的
    零阶保持，只是发生在掩码层而不是点云层）。连续空检计数暴露给控制环做告警。
    """

    def __init__(self, camera_grabber, weights: str, conf: float, device: str,
                 initial_mask: np.ndarray, min_area: int = 1):
        self.grabber = camera_grabber
        self.weights = weights
        self.conf = conf
        self.device = device
        self.min_area = min_area

        self._lock = threading.Lock()
        self._mask = initial_mask
        self._meta = {"conf": 0.0, "n_det": 0, "age": 0, "empty_streak": 0,
                      "lat_ms": 0.0, "updates": 0}
        self._last_seq = -1     # 只对新帧跑推理，见 _run
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, name="yolo-mask", daemon=True)

    def start(self):
        self._thread.start()

    def _run(self):
        while not self._stop.is_set():
            latest = self.grabber.latest()
            if latest is None:
                time.sleep(0.005)
                continue
            rgb, _depth, seq = latest
            # 同一帧不重复推理：相机 30fps，YOLO 若更快就会在同一张 RGB 上反复跑，
            # 结果逐字相同却白占 GPU —— 而这块 GPU 同时在跑策略前向。
            if seq == self._last_seq:
                time.sleep(0.002)
                continue
            self._last_seq = seq
            try:
                t0 = time.perf_counter()
                mask, best_conf, n_det = yolo_mask_once(
                    rgb, self.weights, self.conf, self.device)
                lat = (time.perf_counter() - t0) * 1000.0
            except Exception as exc:            # 推理异常不该拖垮控制环
                with self._lock:
                    self._meta["empty_streak"] += 1
                print(f"\n[YOLO] 推理异常，保留上一张掩码: {type(exc).__name__}: {exc}")
                time.sleep(0.05)
                continue

            with self._lock:
                if int(mask.sum()) >= self.min_area:
                    self._mask = mask
                    self._meta.update(conf=best_conf, n_det=n_det, lat_ms=lat,
                                      empty_streak=0,
                                      updates=self._meta["updates"] + 1)
                else:
                    self._meta["empty_streak"] += 1
                    self._meta["lat_ms"] = lat

    def latest(self) -> "tuple[np.ndarray, dict]":
        with self._lock:
            return self._mask, dict(self._meta)

    def stop(self):
        self._stop.set()


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="xArm7 stage2 点云策略真机部署（D435 深度 → 点云 → PointNet → 伺服）")
    p.add_argument("--ckpt", type=str, required=True, help="stage2 训练出的 .pt")
    p.add_argument("--mask_source", type=str, default="polygon",
                   choices=("polygon", "yolo"),
                   help="工件掩码来源。polygon=手标多边形（默认）；"
                        "yolo=YOLO-seg 自动分割（启动时需人工确认，之后后台线程持续刷新）")
    p.add_argument("--yolo_weights", type=str, default=None,
                   help="YOLO-seg 权重，默认用 configs/yolo_mask_source.py 里的")
    p.add_argument("--yolo_conf", type=float, default=None,
                   help="YOLO 检出门限，默认 0.25（与训练 stage 3 同一门限）")
    p.add_argument("--yolo_freeze", action="store_true",
                   help="接受确认后**冻结掩码**，运行中不再刷新。工件确定不动时用它"
                        "排除 YOLO 抖动；工件会动就别加 —— 掩码会框到旧位置。")
    p.add_argument("--polygon", type=str, default=None,
                   help="工件多边形：\"x,y x,y ...\" 或 pointcloud_meta.json 路径。"
                        "**不给就在启动时现场标注**（挪动工件后直接跑，不用先存文件）。")
    p.add_argument("--save_polygon", type=str, default=None,
                   help="现场标注后另存到指定路径；默认自动存带时间戳的备份")
    p.add_argument("--robot_ip", type=str, default=None, help="xArm7 控制箱 IP")
    p.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu"))

    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--serial", type=str, default=None)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--intrinsics", type=str, default="real", choices=("real", "sim"))

    p.add_argument("--noise_std", type=float, default=0.0,
                   help="物体点高斯噪声(米)。真机深度自带噪声，仿真加噪正是为了模拟它，"
                        "这里默认 0；只在对照实验时传 0.01。")
    p.add_argument("--no_workspace_filter", action="store_true")
    p.add_argument("--min_visible", type=int, default=200,
                   help="掩码内有效像素低于此值就 hold —— 点云已退化成一小撮重复点，"
                        "此时的观测不可信，不该驱动机械臂。")

    p.add_argument("--dt", type=float, default=CONTROL_DT)
    p.add_argument("--steps", type=int, default=-1, help="跑够步数退出；-1=一直跑")
    p.add_argument("--action_scale_deg", type=float, default=TRAIN_ACTION_SCALE_DEG)
    p.add_argument("--action_clip_deg", type=float, default=TRAIN_ACTION_CLIP_DEG)

    p.add_argument("--dry_run", action="store_true",
                   help="全链路跑通但不连机械臂、不发任何指令。首次务必先跑这个。")
    p.add_argument("--step_mode", action="store_true",
                   help="步进模式：启动即暂停，按空格执行 --step_n 步后自动暂停")
    p.add_argument("--step_n", type=int, default=1)
    p.add_argument("--no_move_init", action="store_true", help="不回初始位姿")
    p.add_argument("--freeze_depth", action="store_true",
                   help="启动就冻结深度图。观测将与现实脱节，仅用于工件确定不动的对照实验。")
    p.add_argument("--no_display", action="store_true")
    p.add_argument("--no_log", action="store_true")
    p.add_argument("--log", type=str, default=None)
    p.add_argument("--no_record", action="store_true",
                   help="不录像。默认开录，存纯 RGB（不含叠加的点/轮廓/文字）。")
    p.add_argument("--record", type=str, default=None,
                   help="录像输出路径（.mp4）。默认 logs/deploy_pointcloud/video_<时间戳>.mp4")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    import cv2
    import torch

    from configs.point_bridge_pointcloud import M_OBJ
    from scripts.camera.real_roi_pointcloud import (
        _verify_extrinsics, build_pointcloud, project_to_pixels,
    )
    from scripts.camera.realsense_camera import RealsenseCamera
    from scripts.camera.view_real_depth import (
        SIM_CANONICAL, colorize_depth, parse_polygon_arg, polygon_to_mask,
        report_intrinsics, select_polygon,
    )

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[警告] CUDA 不可用，退回 CPU")
        args.device = "cpu"
    if not args.dry_run:
        if XArmAPI is None:
            raise SystemExit("[错误] 无法导入 xArmAPI；先用 --dry_run 验证，或设 XARM_SDK_DIR")
        if not args.robot_ip:
            raise SystemExit("[错误] 上机必须给 --robot_ip（或用 --dry_run）")

    # YOLO 参数：默认取 configs/yolo_mask_source.py 的常量，保证与训练 stage 3
    # 用同一个权重和同一个门限。只在 --mask_source yolo 时 import，多边形路线
    # 不该因为缺 ultralytics 就跑不起来。
    yolo_weights = yolo_conf = None
    if args.mask_source == "yolo":
        from configs.yolo_mask_source import DEFAULT_YOLO_WEIGHTS, YOLO_CONF

        yolo_weights = args.yolo_weights or DEFAULT_YOLO_WEIGHTS
        yolo_conf = args.yolo_conf if args.yolo_conf is not None else YOLO_CONF
        if not os.path.exists(yolo_weights):
            raise SystemExit(f"[错误] 找不到 YOLO 权重: {yolo_weights}")

    _verify_extrinsics()
    policy = PointCloudPolicy(args.ckpt, device=args.device)

    action_scale_rad = np.deg2rad(args.action_scale_deg).astype(np.float32)
    action_clip_rad = np.deg2rad(args.action_clip_deg).astype(np.float32)
    init_q_rad = np.deg2rad(INIT_JOINT_POS_DEG).astype(np.float32)
    default_q_rad = np.deg2rad(DEFAULT_JOINT_POS_DEG).astype(np.float32)

    # build_pointcloud 从 argparse 命名空间取参数，这里补齐它要的字段
    class _PCArgs:
        m_obj = M_OBJ
        noise_std = args.noise_std
        no_workspace_filter = args.no_workspace_filter
    pc_args = _PCArgs()

    print("\n" + "=" * 72)
    print(f"[策略]   {args.ckpt}")
    print(f"[xArm]   ip={args.robot_ip} dry_run={args.dry_run} dt={args.dt}s")
    print(f"[动作]   scale=clip={args.action_scale_deg}° (与训练一致)")
    print(f"[观测]   物体点 {M_OBJ}x3 + 夹爪点 6x3 + joint_pos_rel 7 = {policy.obs_dim}")
    print(f"[基准]   joint_pos_rel 基准角 = {DEFAULT_JOINT_POS_DEG}")
    if args.mask_source == "yolo":
        print(f"[掩码]   YOLO-seg  weights={yolo_weights}  conf={yolo_conf}  "
              f"{'冻结' if args.yolo_freeze else '后台持续刷新'}")
    else:
        print("[掩码]   手标多边形")
    if args.step_mode:
        print(f"[键盘]   步进模式：SPACE=执行 {args.step_n} 步  R=回初始  H=hold  "
              f"F=冻结深度  U=解冻  Q=退出")
    else:
        print("[键盘]   SPACE=暂停/继续  R=回初始  H=hold  F=冻结深度  U=解冻  Q=退出")
    print("=" * 72 + "\n")

    arm = camera = grabber = csv_file = writer = yolo_worker = None
    video = None
    video_frames = 0

    try:
        # ── 相机 ──
        # 顺序：先相机、先标注，**最后**才连机械臂。标注要花几十秒到几分钟，
        # 把它放在使能之后，机械臂就得一直带电停在初始位姿等你画完 —— 既没必要，
        # 也让"标注中途按 q 退出"变成一次带电异常退出。现在退出时臂还没连。
        camera = RealsenseCamera(width=args.width, height=args.height, fps=args.fps,
                                 align_on_device=True, serial_number=args.serial)
        camera.start()
        intr = report_intrinsics(camera)
        if args.intrinsics == "sim":
            K = np.array([[SIM_CANONICAL["fx"], 0, SIM_CANONICAL["cx"]],
                          [0, SIM_CANONICAL["fy"], SIM_CANONICAL["cy"]],
                          [0, 0, 1]], dtype=np.float64)
            print("[内参] 强制使用仿真 canonical")
        else:
            K = np.asarray(camera.K_color, dtype=np.float64)

        print(f"[预热] 丢弃前 {args.warmup} 帧...")
        for _ in range(args.warmup):
            camera.get_frame()

        # ── 掩码来源：YOLO 或手标多边形 ──────────────────────────────────
        # 两条路都必须在 grabber 起线程**之前**完成首次交互：select_polygon 与
        # confirm_yolo_mask 都自己调 cam.get_frame()，和后台取流线程会抢同一个
        # pipeline。
        polygon = None          # YOLO 路线下为 None，仅影响画绿线与备份格式
        yolo_contour = None
        if args.mask_source == "yolo":
            if args.no_display:
                raise SystemExit(
                    "[错误] --mask_source yolo 需要人工确认掩码，不能配 --no_display")
            print("\n[掩码] YOLO-seg 自动分割（与训练 stage 3 同一个 yolo_masks）")
            print(f"       weights={yolo_weights}  conf={yolo_conf}")
            confirmed = confirm_yolo_mask(
                camera, yolo_weights, yolo_conf, args.device, args.min_visible)
            if confirmed is None:
                raise SystemExit(
                    "[错误] 未确认 YOLO 掩码，退出（未连接机械臂前退出是安全的）")
            mask, yolo_contour, _c0, _n0, yolo_conf = confirmed
            print(f"[掩码] YOLO 掩码已确认，面积 {int(mask.sum())} px")
            if args.yolo_freeze:
                print("       --yolo_freeze：运行中不再刷新掩码（工件必须确定不动）")
            else:
                print("       运行中由后台线程持续刷新；空检出保留上一张（零阶保持）")
        else:
            if args.polygon:
                polygon = parse_polygon_arg(args.polygon)
                print(f"[掩码] 复用已存多边形: {args.polygon}")
                print("       工件若已挪动，这份标注就失效了 —— 掩码会框到桌面，"
                      "算出的假点云不会报错。挪过就别加 --polygon。")
            else:
                if args.no_display:
                    raise SystemExit(
                        "[错误] --no_display 时无法现场标注，请用 --polygon 指定")
                print("\n[标注] 未给 --polygon，现场标注工件轮廓")
                polygon = select_polygon(camera)
                if polygon is None:
                    raise SystemExit("[错误] 未标注有效多边形，退出（未连接机械臂前退出是安全的）")
                spec = " ".join(f"{x},{y}" for x, y in polygon)
                print(f"[标注] 完成 {len(polygon)} 点  (复用: --polygon \"{spec}\")")

            mask = polygon_to_mask(polygon, args.height, args.width)
            print(f"[掩码] 多边形 {len(polygon)} 点，面积 {int(mask.sum())} px")

        # 运行时画在画面上的轮廓：多边形路线用顶点，YOLO 路线用掩码外轮廓
        poly_arr = (np.asarray(polygon, dtype=np.int32) if polygon is not None
                    else (np.asarray(yolo_contour, dtype=np.int32)
                          if yolo_contour is not None else None))

        # 标注后立刻取一帧做基线：掩码内有效像素太少的话，点云是退化的，
        # 这时候上机没有意义 —— 与其等控制环里反复 hold，不如现在就说清楚。
        _rgb0, _d0 = camera.get_frame()
        _valid0 = int((mask & np.isfinite(_d0) & (_d0 > 0)).sum())
        print(f"[基线] 掩码内有效深度像素 {_valid0}/{int(mask.sum())}")
        if _valid0 < args.min_visible:
            print(f"  [警告] 低于 --min_visible={args.min_visible}，控制环会持续 hold。"
                  f"多边形圈大一点，或确认工件确实在画面里。")

        # 现场标注/确认的存一份带时间戳的备份：出了问题要能复现当时用的是哪个掩码。
        # YOLO 路线存的是掩码 png + 元数据（掩码不是几个顶点，存不成多边形）。
        if polygon is not None and not args.polygon:
            from datetime import datetime
            import json as _json

            out = (Path(args.save_polygon) if args.save_polygon else
                   Path(_PROJECT_DIR) / "logs" / "deploy_pointcloud" /
                   f"polygon_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(_json.dumps({
                "polygon": [list(p) for p in polygon],
                "mask_pixels": int(mask.sum()),
                "valid_depth_px": _valid0,
            }, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[标注] 已备份 → {out}")
        elif args.mask_source == "yolo":
            from datetime import datetime
            import json as _json

            stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            bak = Path(_PROJECT_DIR) / "logs" / "deploy_pointcloud"
            bak.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(bak / f"yolomask_{stamp}.png"),
                        (mask.astype(np.uint8) * 255))
            (bak / f"yolomask_{stamp}.json").write_text(_json.dumps({
                "weights": yolo_weights,
                "conf": yolo_conf,
                "mask_pixels": int(mask.sum()),
                "valid_depth_px": _valid0,
                "frozen": bool(args.yolo_freeze),
            }, indent=2, ensure_ascii=False), encoding="utf-8")
            print(f"[掩码] 已备份 → {bak / f'yolomask_{stamp}.png'}")

        # ── 连接机械臂（标注完成后才使能，见上文顺序说明） ──
        if not args.dry_run:
            arm = XArmAPI(args.robot_ip, is_radian=True, check_joint_limit=False)
            arm.clean_error(); arm.clean_warn()
            arm.motion_enable(enable=True); arm.set_state(0)
            print(f"[xArm] 已连接 {args.robot_ip} "
                  f"(warn={arm.warn_code} error={arm.error_code})")
            if not args.no_move_init:
                print(f"[xArm] 回初始位姿: {np.round(INIT_JOINT_POS_DEG, 2)}°")
                enter_position_mode(arm)
                code = arm.set_servo_angle(angle=init_q_rad.tolist(),
                                           speed=np.deg2rad(INIT_SPEED_DEG),
                                           is_radian=True, wait=True)
                if code != 0:
                    raise RuntimeError(f"[xArm] 初始位姿移动失败: code={code}")
            enter_servo_mode(arm)

        grabber = ThreadedFrameGrabber(camera)
        grabber.start()
        time.sleep(1.0)

        # YOLO 掩码后台线程。必须在 grabber 起来之后 —— 它从 grabber 读帧，不自己
        # 碰相机 pipeline。--yolo_freeze 时不起线程，掩码保持确认时那一张。
        if args.mask_source == "yolo" and not args.yolo_freeze:
            yolo_worker = YoloMaskWorker(
                grabber, yolo_weights, yolo_conf, args.device, initial_mask=mask)
            yolo_worker.start()
            print("[YOLO] 掩码刷新线程已启动")

        # ── 日志 ──
        if not args.no_log:
            if args.log is None:
                from datetime import datetime
                log_dir = Path(_PROJECT_DIR) / "logs" / "deploy_pointcloud"
                log_dir.mkdir(parents=True, exist_ok=True)
                args.log = str(log_dir /
                               f"rollout_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv")
            csv_file = open(args.log, "w", newline="", encoding="utf-8")
            writer = csv.writer(csv_file)
            header = ["step", "t_s", "loop_ms", "policy_ms", "seq", "visible_px",
                      "paused", "manual_hold", "depth_frozen", "hold_reason",
                      "send_code"]
            header += [f"obj_centroid_{a}" for a in "xyz"]
            header += [f"ee_pos_{a}" for a in "xyz"]
            header += [f"ee_quat_{a}" for a in "wxyz"]
            header += ["mask_px", "yolo_conf", "yolo_ndet", "yolo_empty_streak"]
            header += [f"joint_rel_j{i}_deg" for i in range(1, 8)]
            header += [f"raw_action_j{i}" for i in range(1, 8)]
            header += [f"cmd_delta_j{i}_deg" for i in range(1, 8)]
            header += [f"target_j{i}_deg" for i in range(1, 8)]
            header += [f"q_before_j{i}_deg" for i in range(1, 8)]
            writer.writerow(header)
            print(f"[日志] {args.log}")

        # ── 录像 ──
        # 存的是相机原始 RGB，不带任何叠加层（点云投影、轮廓、状态文字都不写进去），
        # 这样录下来的视频可以直接回灌 YOLO 或重新标注，等价于当时的真实输入。
        # 只在新帧到达时写一帧（seq 变化），所以视频时长≈真实时长，不受控制环
        # 空转或步进模式暂停的影响。
        if not args.no_record:
            from datetime import datetime

            if args.record is None:
                vid_dir = Path(_PROJECT_DIR) / "logs" / "deploy_pointcloud"
                vid_dir.mkdir(parents=True, exist_ok=True)
                args.record = str(
                    vid_dir / f"video_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
            else:
                Path(args.record).parent.mkdir(parents=True, exist_ok=True)
            video = cv2.VideoWriter(
                args.record, cv2.VideoWriter_fourcc(*"mp4v"),
                float(args.fps), (args.width, args.height))
            if not video.isOpened():
                # 编码器缺失不该拖垮上机流程，降级成不录并说清楚
                print(f"[录像] 打不开编码器，已关闭录像: {args.record}")
                video = None
            else:
                print(f"[录像] {args.record}  ({args.width}x{args.height} @{args.fps}fps)")

        # ── 控制环 ──
        paused = args.step_mode
        step_remaining = 0
        manual_hold = False
        step_count = 0
        frozen_depth = None
        frozen_at_step = None
        last_seq = -1
        video_last_seq = -1     # 录像独立计数：last_seq 会被掩码更新强制回退
        pts = None
        n_vis = 0
        policy_ms = 0.0
        screenshot_pending = False
        next_t = time.perf_counter()
        loop_start = next_t

        if args.freeze_depth:
            print("[冻结] --freeze_depth：取到首帧后立即冻结")

        while args.steps < 0 or step_count < args.steps:
            t0 = time.perf_counter()

            # ── 键盘 ──
            key = -1
            if not args.no_display:
                key = cv2.waitKey(1) & 0xFF
            elif select.select([sys.stdin], [], [], 0)[0]:
                line = sys.stdin.readline()
                key = ord(line[0]) if line else -1

            if key in (ord("q"), ord("Q"), 27):
                print("\n[退出] 用户按键退出")
                break
            elif key == ord(" "):
                if args.step_mode:
                    if paused:
                        step_remaining = max(1, args.step_n)
                        paused = False
                        print(f"\n[步进] 执行 {step_remaining} 步")
                else:
                    paused = not paused
                    print(f"\n[状态] {'已暂停' if paused else '继续运行'}")
            elif key == ord("r"):
                print("\n[重置] 回初始位姿（先暂停策略输出）")
                paused = True
                if not args.dry_run:
                    enter_position_mode(arm)
                    arm.set_servo_angle(angle=init_q_rad.tolist(),
                                        speed=np.deg2rad(INIT_SPEED_DEG),
                                        is_radian=True, wait=True)
                    enter_servo_mode(arm)
            elif key == ord("h"):
                manual_hold = not manual_hold
                print(f"\n[保持] {'手动 hold 开' if manual_hold else '手动 hold 关'}")
            elif key == ord("u"):
                if frozen_depth is not None:
                    frozen_depth, frozen_at_step = None, None
                    print(f"\n[解冻] 第 {step_count} 步起恢复实时深度")
            elif key == ord("s"):
                screenshot_pending = True

            # ── 取帧 → 点云 ──
            latest = grabber.latest()
            if latest is None:
                time.sleep(0.005)
                continue
            rgb, depth, seq = latest

            # 纯 RGB 落盘。放在任何叠加绘制之前，且用 rgb 本体不用副本 ——
            # 下游的 display 分支自己 .copy() 过，不会污染这里写出去的帧。
            if video is not None and seq != video_last_seq:
                video_last_seq = seq
                # ascontiguousarray：::-1 出来是负 stride 视图，cv2 要连续内存
                video.write(np.ascontiguousarray(rgb[..., ::-1]))    # RGB → BGR
                video_frames += 1

            if key == ord("f"):
                frozen_depth, frozen_at_step = depth.copy(), step_count
                print(f"\n[冻结] 第 {step_count} 步深度图已冻结 —— 观测将与现实脱节")
            elif frozen_depth is None and args.freeze_depth:
                frozen_depth, frozen_at_step = depth.copy(), step_count
                print(f"[冻结] 第 {step_count} 步深度图已冻结（--freeze_depth）")

            # YOLO 路线：取后台线程最新的掩码。它比深度旧几帧 —— 与训练 stage 4
            # 的语义一致（掩码低频刷新、深度每步都新），工件静止时无影响。
            yolo_meta = None
            if yolo_worker is not None:
                new_mask, yolo_meta = yolo_worker.latest()
                if new_mask is not mask:
                    mask = new_mask
                    last_seq = -1        # 掩码换了，强制重算点云

            src_depth = frozen_depth if frozen_depth is not None else depth
            # 冻结时深度不变，但掩码与 FPS 照常每帧重跑；实时时只在新帧到达才重算
            if frozen_depth is not None or seq != last_seq:
                last_seq = seq
                pts, n_vis = build_pointcloud(src_depth, mask, K, pc_args)

            # ── 读机械臂状态 ──
            if args.dry_run:
                # 干跑没有真实关节，用默认位姿占位。观测数值仍可核对点云那 192 维，
                # 但夹爪点固定不动 —— 别据此判断闭环行为。
                q_before = default_q_rad.copy()
                ee_pos_m = np.array([0.4, 0.0, 0.3], dtype=np.float32)
                ee_quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
            else:
                q_before = get_joint_positions(arm)
                ee_pos_m, ee_quat = get_tcp_pose_base(arm)

            # ── 是否 hold ──
            pts_bad = pts is None or n_vis < args.min_visible
            hold = paused or manual_hold or pts_bad
            hold_reason = ("paused" if paused else
                           "manual" if manual_hold else
                           f"visible<{args.min_visible}" if pts_bad else "")

            raw_action = np.zeros(ACT_DIM, dtype=np.float32)
            obs = None
            robot_pts = None
            policy_ms = 0.0
            if not hold:
                obs, robot_pts = build_obs(pts, ee_pos_m, ee_quat, q_before,
                                           default_q_rad)
                t_pol = time.perf_counter()
                raw_action = policy.predict(obs)
                policy_ms = (time.perf_counter() - t_pol) * 1000.0

            cmd_delta_rad = np.clip(raw_action * action_scale_rad,
                                    -action_clip_rad, action_clip_rad).astype(np.float32)
            target_rad = np.clip(q_before + cmd_delta_rad,
                                 JOINT_LIMITS_LOW_RAD,
                                 JOINT_LIMITS_HIGH_RAD).astype(np.float32)

            send_code = -1
            if not args.dry_run:
                if not check_joint_safe(target_rad):
                    print("\n[安全] 目标关节角逼近限位，本步拒发")
                    target_rad = q_before
                send_code = arm.set_servo_angle_j(
                    angles=target_rad.tolist(), speed=np.deg2rad(100.0),
                    mvacc=np.deg2rad(1000.0), mvtime=0)
                if send_code != 0:
                    print(f"\n[警告] set_servo_angle_j 失败: {send_code}，恢复模式")
                    recover_servo_mode(arm)

            # ── 可视化 ──
            if not args.no_display:
                bgr = rgb[..., ::-1].copy()
                depth_vis, _, _ = colorize_depth(src_depth, 0.3, 1.5)
                # YOLO 路线画当前掩码的实时轮廓（掩码每帧可能变），多边形路线画固定顶点
                if yolo_worker is not None or args.mask_source == "yolo":
                    cnts, _ = cv2.findContours(mask.astype(np.uint8),
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                    for img in (bgr, depth_vis):
                        cv2.drawContours(img, cnts, -1, (0, 255, 0), 2)
                elif poly_arr is not None:
                    for img in (bgr, depth_vis):
                        cv2.polylines(img, [poly_arr], True, (0, 255, 0), 2)

                if pts is not None:
                    uvz = project_to_pixels(pts, K)
                    for u, v, z in uvz:
                        if z > 0 and 0 <= u < args.width and 0 <= v < args.height:
                            cv2.circle(bgr, (int(u), int(v)), 2, (0, 0, 255), -1)
                            cv2.circle(depth_vis, (int(u), int(v)), 2, (0, 0, 255), -1)
                # 夹爪点画青色，与 stage0/回放脚本同一套配色
                if robot_pts is not None:
                    for u, v, z in project_to_pixels(robot_pts, K):
                        if z > 0 and 0 <= u < args.width and 0 <= v < args.height:
                            cv2.circle(bgr, (int(u), int(v)), 4, (255, 255, 0), -1)

                state = "HOLD" if hold else "RUN"
                info = (f"step {step_count}  {state}  visible {n_vis}px  "
                        f"policy {policy_ms:.1f}ms")
                if hold_reason:
                    info += f"  ({hold_reason})"
                if yolo_meta is not None:
                    info += (f"  | yolo conf {yolo_meta['conf']:.2f} "
                             f"det {yolo_meta['n_det']} {yolo_meta['lat_ms']:.0f}ms")
                    if yolo_meta["empty_streak"] > 0:
                        info += f" empty x{yolo_meta['empty_streak']}"
                for img in (bgr, depth_vis):
                    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
                    cv2.putText(img, info, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                                (0, 200, 255) if hold else (255, 255, 255), 1,
                                cv2.LINE_AA)

                # 冻结态画得刺眼：此时点云与画面里的现实脱节，闭环下正是撞机前提
                if frozen_depth is not None:
                    warn = (f"!! FROZEN depth @step {frozen_at_step} "
                            f"(age {step_count - frozen_at_step}) -- u to unfreeze !!")
                    for img in (bgr, depth_vis):
                        cv2.rectangle(img, (0, 22), (img.shape[1], 46), (0, 0, 200), -1)
                        cv2.putText(img, warn, (5, 39), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.42, (255, 255, 255), 1, cv2.LINE_AA)
                        cv2.rectangle(img, (1, 1), (img.shape[1] - 2, img.shape[0] - 2),
                                      (0, 0, 200), 3)

                combined = np.hstack([bgr, depth_vis])
                cv2.imshow("xArm7 PointCloud Policy Deploy", combined)
                if screenshot_pending:
                    shot = Path(_PROJECT_DIR) / "logs" / "deploy_pointcloud"
                    shot.mkdir(parents=True, exist_ok=True)
                    fp = shot / f"gui_{time.strftime('%Y%m%d_%H%M%S')}_s{step_count}.png"
                    cv2.imwrite(str(fp), combined)
                    print(f"\n[截图] 已保存 {fp}")
                    screenshot_pending = False

            # ── 步进模式自动暂停 ──
            if args.step_mode and not paused:
                step_remaining -= 1
                if step_remaining <= 0:
                    paused = True
                    print("\n[步进] 已自动暂停（hold），按空格执行下一步")

            loop_ms = (time.perf_counter() - t0) * 1000.0

            if writer is not None:
                cen = pts.mean(0) if pts is not None else np.zeros(3)
                row = [step_count, f"{time.perf_counter() - loop_start:.4f}",
                       f"{loop_ms:.2f}", f"{policy_ms:.2f}", seq, n_vis,
                       int(paused), int(manual_hold), int(frozen_depth is not None),
                       hold_reason, send_code]
                row += [f"{v:.5f}" for v in cen]
                row += [f"{v:.5f}" for v in ee_pos_m]
                row += [f"{v:.5f}" for v in ee_quat]
                row += [int(mask.sum()),
                        f"{yolo_meta['conf']:.4f}" if yolo_meta else "",
                        yolo_meta["n_det"] if yolo_meta else "",
                        yolo_meta["empty_streak"] if yolo_meta else ""]
                row += [f"{v:.4f}" for v in np.rad2deg(q_before - default_q_rad)]
                row += [f"{v:.5f}" for v in raw_action]
                row += [f"{v:.5f}" for v in np.rad2deg(cmd_delta_rad)]
                row += [f"{v:.4f}" for v in np.rad2deg(target_rad)]
                row += [f"{v:.4f}" for v in np.rad2deg(q_before)]
                writer.writerow(row)

            if step_count % 50 == 0:
                tag = f" [FROZEN@{frozen_at_step}]" if frozen_depth is not None else ""
                print(f"step {step_count}  {'HOLD' if hold else 'RUN'}  "
                      f"visible={n_vis}  loop={loop_ms:.1f}ms{tag}")
            step_count += 1

            # 保持 50Hz
            next_t += args.dt
            sleep_s = next_t - time.perf_counter()
            if sleep_s > 0:
                time.sleep(sleep_s)
            else:
                next_t = time.perf_counter()

    except KeyboardInterrupt:
        print("\n[退出] Ctrl-C")
    finally:
        # 先停机械臂再收其它资源：异常路径下这一步最要紧
        if arm is not None:
            try:
                arm.set_state(0)
                arm.set_mode(0)
                arm.set_state(0)
                arm.disconnect()
                print("[xArm] 已停止并断开")
            except Exception as exc:
                print(f"[xArm] 收尾异常: {type(exc).__name__}: {exc}")
        # YOLO 线程先停：它从 grabber 读帧，反过来的顺序会让它读到已停的相机
        if yolo_worker is not None:
            yolo_worker.stop()
            _, _m = yolo_worker.latest()
            print(f"[YOLO] 已停止（掩码更新 {_m['updates']} 次）")
        if grabber is not None:
            grabber.stop()
        if camera is not None:
            camera.stop()
        if csv_file is not None:
            csv_file.close()
            print(f"[日志] 已写入 {args.log}")
        # release 必须走到，否则 mp4 缺 moov box，整个文件打不开
        if video is not None:
            video.release()
            print(f"[录像] 已写入 {args.record}  ({video_frames} 帧，"
                  f"约 {video_frames / max(1, args.fps):.1f}s)")
        if not args.no_display:
            import cv2 as _cv2
            _cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
