#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""通过 xArm SDK 闭环部署 21D pick-pose 策略。"""

import os
import sys
sys.path = [p for p in sys.path if "_isaac_sim" not in p]

import time
import argparse
import tty
import termios
from multiprocessing import resource_tracker, shared_memory
from pathlib import Path

_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _CURRENT_DIR.parents[1]
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from tools.policy_log_paths import infer_policy_time_from_checkpoint, make_policy_output_path

import numpy as np
import torch
import torch.nn as nn



# =============================================================================
# 自动寻找 xArm SDK
# =============================================================================

def _add_xarm_sdk_to_path():
    candidates = []

    env_path = os.environ.get("XARM_SDK_DIR", "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())

    script_dir = Path(__file__).resolve().parent
    candidates += [
        _PROJECT_DIR / "third_party" / "xArm_Python_SDK_master",
        script_dir / "third_party" / "xArm_Python_SDK_master",
        script_dir.parent / "third_party" / "xArm_Python_SDK_master",
        script_dir.parent.parent / "third_party" / "xArm_Python_SDK_master",
        Path("/home/gxai/Desktop/embodied-ai-project/modules/control/scripts/third_party/xArm_Python_SDK_master"),
        Path("/home/gxai/Desktop/zjj/embodied-ai-project/modules/control/scripts/third_party/xArm_Python_SDK_master"),
        Path("/home/gxai/Desktop/test_pickandprice/embodied-ai-project/modules/control/scripts/third_party/xArm_Python_SDK_master"),
        Path("/home/gxai/Desktop/task_node_neo/embodied-ai-project/modules/control/scripts/third_party/xArm_Python_SDK_master"),
    ]

    for p in candidates:
        if (p / "xarm").exists():
            sys.path.insert(0, str(p))
            print(f"[SDK] 使用 xArm SDK: {p}")
            return p

    print("[SDK] 未在常见位置找到 xArm SDK。")
    print("[SDK] 可设置：export XARM_SDK_DIR=/path/to/xArm_Python_SDK_master")
    return None


_add_xarm_sdk_to_path()

try:
    from xarm.wrapper import XArmAPI
except Exception as exc:
    print("[Error] 无法导入 xArmAPI。")
    print(f"原始错误: {type(exc).__name__}: {exc}")
    raise

try:
    from ctm2f110_gripper import CTM2F110Gripper
except Exception as exc:
    CTM2F110Gripper = None
    _GRIPPER_IMPORT_ERROR = exc
else:
    _GRIPPER_IMPORT_ERROR = None


LOG_NAME = "xarm7_pick_pose"

VERSION_TAG = "v87_relative_pose_joint_step_direct_0p3deg"

INIT_JOINTS_DEG = [-0.4, -47.6, 0.0, 1.8, -0.2, 49.4, -0.1]
DEFAULT_JOINT_POS_RAD = np.deg2rad(INIT_JOINTS_DEG).astype(np.float32)

INIT_SPEED_DEG = 20.0

JOINT_LIMITS_LOW = np.deg2rad(
    [-180, -118, -180, -11, -97, -180, -180]
).astype(np.float32)

JOINT_LIMITS_HIGH = np.deg2rad(
    [180, 118, 180, 225, 97, 180, 180]
).astype(np.float32)

CONTROL_DT = 0.02
OBS_DIM = 21
ACT_DIM = 7
RSL_RL_OBS_NORM_EPS = 1.0e-2

TRAIN_ACTION_SCALE_DEG = 0.6
TRAIN_ACTION_CLIP_DEG = 0.6

DEFAULT_Q_OFFSET_WXYZ = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float32)
DEFAULT_FOUNDATIONPOSE_OBJECT_FRAME_QUAT_WXYZ = np.array(
    [0.9984, 0.0334, 0.0149, -0.0436], dtype=np.float32
)
DEFAULT_GRASP_SYMMETRY_QUATS_WXYZ = np.asarray(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ],
    dtype=np.float32,
)
FOUNDATIONPOSE_SHM_NAME = "foundationpose_multi_pose"
FOUNDATIONPOSE_HEADER_BYTES = 12
FOUNDATIONPOSE_POSE_BYTES = 128
FOUNDATIONPOSE_MAX_OBJECTS = 10


def _getch() -> str:
    fd = sys.stdin.fileno()
    old = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        ch = sys.stdin.read(1)
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old)
    return ch


def _wait_for_step() -> bool:
    print("  [SPACE=执行 / q=退出] ", end="", flush=True)
    while True:
        ch = _getch()
        if ch == " ":
            print()
            return True
        if ch in ("q", "Q", "\x03", "\x04"):
            print("\n[Step] 用户退出")
            return False


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    n = np.linalg.norm(q)
    if n < 1.0e-8:
        raise ValueError("四元数范数过小")
    return (q / n).astype(np.float32)


def quat_mul(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    q1 = quat_normalize(q1)
    q2 = quat_normalize(q2)
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def quat_inv(q: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=np.float32)


def quat_mul_raw(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product, wxyz. 不做 normalize，用于向量旋转。"""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dtype=np.float32,
    )


def quat_apply_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """把 base/world 向量 v 旋转到 q 对应坐标系下，保持向量长度。"""
    q = quat_normalize(q)
    q_inv = quat_inv(q)
    vq = np.array([0.0, v[0], v[1], v[2]], dtype=np.float32)
    return quat_mul_raw(quat_mul_raw(q_inv, vq), q)[1:4]

def quat_from_euler_xyz(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)

    return quat_normalize(
        np.array(
            [
                cr * cp * cy + sr * sp * sy,
                sr * cp * cy - cr * sp * sy,
                cr * sp * cy + sr * cp * sy,
                cr * cp * sy - sr * sp * cy,
            ],
            dtype=np.float32,
        )
    )


def quat_from_rot_matrix(rot: np.ndarray) -> np.ndarray:
    """Convert a 3x3 rotation matrix to a normalized wxyz quaternion."""
    r = np.asarray(rot, dtype=np.float64)
    if r.shape != (3, 3) or not np.all(np.isfinite(r)):
        raise ValueError("rotation matrix must be finite and shape=(3, 3)")

    trace = float(np.trace(r))
    if trace > 0.0:
        s = np.sqrt(trace + 1.0) * 2.0
        qw = 0.25 * s
        qx = (r[2, 1] - r[1, 2]) / s
        qy = (r[0, 2] - r[2, 0]) / s
        qz = (r[1, 0] - r[0, 1]) / s
    elif r[0, 0] > r[1, 1] and r[0, 0] > r[2, 2]:
        s = np.sqrt(1.0 + r[0, 0] - r[1, 1] - r[2, 2]) * 2.0
        qw = (r[2, 1] - r[1, 2]) / s
        qx = 0.25 * s
        qy = (r[0, 1] + r[1, 0]) / s
        qz = (r[0, 2] + r[2, 0]) / s
    elif r[1, 1] > r[2, 2]:
        s = np.sqrt(1.0 + r[1, 1] - r[0, 0] - r[2, 2]) * 2.0
        qw = (r[0, 2] - r[2, 0]) / s
        qx = (r[0, 1] + r[1, 0]) / s
        qy = 0.25 * s
        qz = (r[1, 2] + r[2, 1]) / s
    else:
        s = np.sqrt(1.0 + r[2, 2] - r[0, 0] - r[1, 1]) * 2.0
        qw = (r[1, 0] - r[0, 1]) / s
        qx = (r[0, 2] + r[2, 0]) / s
        qy = (r[1, 2] + r[2, 1]) / s
        qz = 0.25 * s

    return quat_normalize(np.array([qw, qx, qy, qz], dtype=np.float32))


def quat_to_angle_deg(q: np.ndarray) -> float:
    q = quat_normalize(q)
    w = abs(float(q[0]))
    xyz_norm = float(np.linalg.norm(q[1:4]))
    angle = 2.0 * np.arctan2(xyz_norm, max(w, 1.0e-8))
    return float(np.rad2deg(angle))


def _actor_layer_index(key: str) -> int:
    try:
        return int(key.split(".")[1])
    except Exception:
        return 10**9


class V87RelativePoseJointStepPolicy:
    def __init__(self, ckpt_path: str, device: str = "cpu", obs_norm_clip: float = 0.0):
        print(f"[Policy] 加载: {ckpt_path}")
        d = torch.load(ckpt_path, map_location=device)
        sd = d["model_state_dict"]

        actor_weight_keys = sorted(
            [k for k in sd if k.startswith("actor.") and k.endswith(".weight")],
            key=_actor_layer_index,
        )
        if len(actor_weight_keys) == 0:
            raise RuntimeError("[Policy] checkpoint 中没有找到 actor.*.weight")

        layers = []
        for i, key in enumerate(actor_weight_keys):
            w = sd[key]
            b = sd[key.replace(".weight", ".bias")]

            lin = nn.Linear(w.shape[1], w.shape[0])
            lin.weight = nn.Parameter(w)
            lin.bias = nn.Parameter(b)
            layers.append(lin)

            if i < len(actor_weight_keys) - 1:
                layers.append(nn.ELU())

        self.actor = nn.Sequential(*layers).to(device).eval()

        if "actor_obs_normalizer._mean" not in sd or "actor_obs_normalizer._std" not in sd:
            raise RuntimeError("[Policy] checkpoint 中没有 actor_obs_normalizer._mean/_std")

        self.obs_mean = sd["actor_obs_normalizer._mean"].to(device)
        self.obs_std = sd["actor_obs_normalizer._std"].to(device)
        self.device = device
        self.obs_norm_clip = float(obs_norm_clip)

        obs_dim = sd[actor_weight_keys[0]].shape[1]
        act_dim = sd[actor_weight_keys[-1]].shape[0]

        print(f"[Policy] 加载完成  obs_dim={obs_dim}  act_dim={act_dim}")
        assert obs_dim == OBS_DIM, f"期望 {OBS_DIM} 维观测，checkpoint 为 {obs_dim} 维"
        assert act_dim == ACT_DIM, f"期望 {ACT_DIM} 维动作，checkpoint 为 {act_dim} 维"

    def predict(self, obs_np: np.ndarray) -> np.ndarray:
        if obs_np.shape != (OBS_DIM,):
            raise ValueError(f"[Policy] obs 维度错误: {obs_np.shape}, 期望 {(OBS_DIM,)}")

        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=self.device).unsqueeze(0)
        obs_norm = (obs - self.obs_mean) / (self.obs_std + RSL_RL_OBS_NORM_EPS)
        if self.obs_norm_clip > 0.0:
            obs_norm = torch.clamp(obs_norm, -self.obs_norm_clip, self.obs_norm_clip)

        with torch.no_grad():
            action = self.actor(obs_norm).squeeze(0).cpu().numpy().astype(np.float32)

        return action


class FoundationPoseShmReader:
    """Read base-frame object poses published by scripts/camera/run_realtime.py."""

    def __init__(self, name: str, object_index: int, max_age_s: float):
        self.name = str(name)
        self.object_index = int(object_index)
        self.max_age_s = float(max_age_s)
        if self.object_index < 0 or self.object_index >= FOUNDATIONPOSE_MAX_OBJECTS:
            raise ValueError(
                f"--foundationpose_object_index must be in [0, {FOUNDATIONPOSE_MAX_OBJECTS - 1}], "
                f"got {self.object_index}"
            )
        self.shm = shared_memory.SharedMemory(name=self.name)
        # This process only consumes the FoundationPose shared memory. Without
        # unregistering, Python's resource_tracker can unlink the producer-owned
        # /dev/shm name when deploy exits, making later reconnects fail.
        resource_tracker.unregister(self.shm._name, "shared_memory")
        expected_size = FOUNDATIONPOSE_HEADER_BYTES + FOUNDATIONPOSE_POSE_BYTES * FOUNDATIONPOSE_MAX_OBJECTS
        if self.shm.size < expected_size:
            self.close()
            raise RuntimeError(
                f"[FoundationPose] shared memory '{self.name}' is too small: "
                f"{self.shm.size} bytes, expected >= {expected_size}"
            )
        print(
            f"[FoundationPose] connected shm={self.name} object_index={self.object_index} "
            f"max_age_s={self.max_age_s:.3f}"
        )

    def close(self) -> None:
        if getattr(self, "shm", None) is not None:
            self.shm.close()
            self.shm = None

    def read_pose_base(self) -> tuple[np.ndarray, np.ndarray, float, float, int]:
        if self.shm is None:
            raise RuntimeError("[FoundationPose] shared memory is closed")

        buf = self.shm.buf
        num_objects = int(np.frombuffer(buf[0:4], dtype=np.int32, count=1)[0])
        timestamp = float(np.frombuffer(buf[4:12], dtype=np.float64, count=1)[0])
        if num_objects <= 0:
            raise RuntimeError("[FoundationPose] no object pose has been published yet")
        if self.object_index >= num_objects:
            raise RuntimeError(
                f"[FoundationPose] object_index={self.object_index} but only {num_objects} object(s) published"
            )
        if not np.isfinite(timestamp) or timestamp <= 0.0:
            raise RuntimeError(f"[FoundationPose] invalid timestamp: {timestamp}")

        age_s = time.time() - timestamp
        if self.max_age_s >= 0.0 and age_s > self.max_age_s:
            raise RuntimeError(
                f"[FoundationPose] stale pose: age={age_s:.3f}s > max_age_s={self.max_age_s:.3f}s"
            )

        offset = FOUNDATIONPOSE_HEADER_BYTES + self.object_index * FOUNDATIONPOSE_POSE_BYTES
        pose = np.frombuffer(buf[offset : offset + FOUNDATIONPOSE_POSE_BYTES], dtype=np.float64, count=16)
        pose = pose.reshape(4, 4).copy()
        if not np.all(np.isfinite(pose)):
            raise RuntimeError("[FoundationPose] pose contains non-finite values")

        pos_base_m = pose[:3, 3].astype(np.float32)
        quat_base_wxyz = quat_from_rot_matrix(pose[:3, :3])
        return pos_base_m, quat_base_wxyz, age_s, timestamp, num_objects


def object_pose_relative_to_ee_np(
    object_pos_base_m: np.ndarray,
    object_quat_base_wxyz: np.ndarray,
    ee_pos_base_m: np.ndarray,
    ee_quat_base_wxyz: np.ndarray,
    q_offset_wxyz: np.ndarray = DEFAULT_Q_OFFSET_WXYZ,
    grasp_symmetry_quats_wxyz: np.ndarray = DEFAULT_GRASP_SYMMETRY_QUATS_WXYZ,
) -> np.ndarray:
    obj_q = quat_normalize(object_quat_base_wxyz)
    ee_q = quat_normalize(ee_quat_base_wxyz)
    q_offset = quat_normalize(q_offset_wxyz)

    rel_pos_ee = quat_apply_inverse(ee_q, object_pos_base_m - ee_pos_base_m)

    rel_quat_candidates = []
    for sym_q in np.asarray(grasp_symmetry_quats_wxyz, dtype=np.float32).reshape(-1, 4):
        offset = quat_mul(q_offset, sym_q)
        target_quat_base = quat_mul(obj_q, offset)
        rel_quat = quat_mul(quat_inv(ee_q), target_quat_base)
        rel_quat_candidates.append(quat_normalize(rel_quat))

    rel_quats = np.stack(rel_quat_candidates, axis=0)
    best_idx = int(np.argmax(np.abs(rel_quats[:, 0])))
    rel_quat = rel_quats[best_idx]

    if rel_quat[0] < 0.0:
        rel_quat = -rel_quat

    return np.concatenate([rel_pos_ee, rel_quat]).astype(np.float32)


def build_obs(
    joint_rad: np.ndarray,
    object_rel_pose: np.ndarray,
    last_action_clipped: np.ndarray,
) -> np.ndarray:
    joint_pos_rel = joint_rad.astype(np.float32) - DEFAULT_JOINT_POS_RAD

    obs = np.concatenate(
        [
            joint_pos_rel.astype(np.float32),
            object_rel_pose.astype(np.float32),
            last_action_clipped.astype(np.float32),
        ]
    ).astype(np.float32)

    if obs.shape != (OBS_DIM,):
        raise RuntimeError(f"[Obs] 观测维度错误: {obs.shape}, 期望 {(OBS_DIM,)}")

    return obs


def enter_position_mode(arm: XArmAPI):
    arm.set_mode(0)
    arm.set_state(0)
    time.sleep(0.2)
    print("[Arm] 已切换到 Mode 0")


def enter_servo_mode(arm: XArmAPI):
    arm.set_mode(1)
    arm.set_state(0)
    time.sleep(0.3)
    print("[Arm] 已切换到 Mode 1 连续伺服模式")


def recover_servo_mode(arm: XArmAPI):
    print("[Arm] 尝试恢复 Mode 1 ...")
    arm.clean_error()
    arm.clean_warn()
    arm.motion_enable(enable=True)
    arm.set_mode(1)
    arm.set_state(0)
    time.sleep(0.1)


def check_joint_safe(joint_rad: np.ndarray, margin_deg: float = 1.0) -> bool:
    margin = np.deg2rad(margin_deg)
    return bool(
        np.all(joint_rad > JOINT_LIMITS_LOW + margin)
        and np.all(joint_rad < JOINT_LIMITS_HIGH - margin)
    )


def format_deg(values_rad: np.ndarray, ndigits: int = 3) -> list:
    return np.rad2deg(values_rad).round(ndigits).tolist()


def format_np(values: np.ndarray, ndigits: int = 3) -> list:
    return np.asarray(values, dtype=np.float32).round(ndigits).tolist()


def get_tcp_pose_base(arm: XArmAPI):
    code, pose = arm.get_position(is_radian=True)
    if code != 0 or pose is None:
        raise RuntimeError(f"[Arm] get_position failed: {code}")

    pose = np.asarray(pose[:6], dtype=np.float32)
    ee_pos_m = pose[:3] * 0.001
    ee_quat = quat_from_euler_xyz(float(pose[3]), float(pose[4]), float(pose[5]))
    return ee_pos_m.astype(np.float32), ee_quat.astype(np.float32), pose


def get_joint_positions(arm: XArmAPI) -> np.ndarray:
    code, raw_angles = arm.get_servo_angle(is_radian=True)
    if code != 0 or raw_angles is None:
        raise RuntimeError(f"[Arm] get_servo_angle failed: {code}")
    return np.asarray(raw_angles[:7], dtype=np.float32)


def execute_gripper_close_and_lift(
    arm: XArmAPI,
    gripper: "CTM2F110Gripper",
    args: argparse.Namespace,
    raw_tcp_pose: np.ndarray,
) -> int:
    """Close the CTM2F110 gripper and lift the current TCP in base Z."""
    print("\n=== Auto grasp trigger ===")
    print(
        f"[Gripper] close speed={args.gripper_close_speed} "
        f"torque={args.gripper_close_torque} hold={args.gripper_close_hold_s:.2f}s"
    )
    print(
        f"[Lift] dz={args.gripper_lift_mm:.1f}mm "
        f"speed={args.gripper_lift_speed:.1f}mm/s acc={args.gripper_lift_acc:.1f}mm/s^2"
    )

    if args.dry_run:
        pose = np.asarray(raw_tcp_pose[:6], dtype=np.float32)
        print(
            "[DryRun] would close gripper and lift TCP "
            f"from z={pose[2]:.2f}mm to z={pose[2] + args.gripper_lift_mm:.2f}mm"
        )
        return 0

    gripper.start_closing(
        speed=args.gripper_close_speed,
        torque=args.gripper_close_torque,
    )
    time.sleep(max(0.0, float(args.gripper_close_hold_s)))

    # Use a fresh TCP pose after the close delay; the policy loop may have been
    # running in servo mode just before the trigger.
    code, pose = arm.get_position(is_radian=True)
    if code != 0 or pose is None:
        raise RuntimeError(f"[Lift] get_position before lift failed: code={code}")
    pose = np.asarray(pose[:6], dtype=np.float32)
    target_z = float(pose[2] + args.gripper_lift_mm)

    enter_position_mode(arm)
    lift_code = arm.set_position(
        x=float(pose[0]),
        y=float(pose[1]),
        z=target_z,
        roll=float(pose[3]),
        pitch=float(pose[4]),
        yaw=float(pose[5]),
        speed=float(args.gripper_lift_speed),
        mvacc=float(args.gripper_lift_acc),
        is_radian=True,
        wait=True,
        timeout=float(args.gripper_lift_timeout_s),
    )
    print(f"[Lift] set_position code={lift_code} target_z={target_z:.2f}mm")
    if lift_code != 0:
        raise RuntimeError(f"[Lift] set_position failed: code={lift_code}")
    print("[Auto] close + lift done; gripper close command remains latched for holding.")
    return lift_code


def parse_args():
    parser = argparse.ArgumentParser(
        description="sim2real v87 state-pose deployment, 21D obs, ±0.6deg/step"
    )

    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--ip", type=str, default="192.168.73.229")
    parser.add_argument("--real", action="store_true", help="Tag outputs as sim2real data only; does not switch SDK robot mode.")
    parser.add_argument("--step_mode", action="store_true")
    parser.add_argument("--steps", type=int, default=250)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--obs_norm_clip", type=float, default=0.0, help="normalized obs clamp; 0 disables it to match Isaac inference")

    parser.add_argument("--action_scale_deg", type=float, default=TRAIN_ACTION_SCALE_DEG)
    parser.add_argument("--action_clip_deg", type=float, default=TRAIN_ACTION_CLIP_DEG)

    parser.add_argument("--log", type=str, default=None, help="Output CSV path. Default: test_log/xarm7_pick_pose-<policy_time>/<sim2real|sim2sim>_deploy_state_pose_<policy_time>.csv")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_move_init", action="store_true")
    pose_group = parser.add_mutually_exclusive_group(required=True)
    pose_group.add_argument(
        "--object_rel_pose",
        type=float,
        nargs=7,
        metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"),
        help="直接传入训练观测 object_pose_relative_to_ee，位置单位 m，四元数 wxyz。",
    )
    pose_group.add_argument(
        "--object_pos_base",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="工件在 xArm base 坐标系下的位置，单位 m。需要同时传 --object_quat_base。",
    )
    pose_group.add_argument(
        "--foundationpose",
        action="store_true",
        help="从 FoundationPose 共享内存读取工件在 xArm base 坐标系下的 4x4 位姿。",
    )

    parser.add_argument(
        "--object_quat_base",
        type=float,
        nargs=4,
        metavar=("W", "X", "Y", "Z"),
        help="工件在 xArm base 坐标系下的四元数，wxyz。仅 --object_pos_base 模式需要。",
    )

    parser.add_argument(
        "--foundationpose_shm_name",
        type=str,
        default=FOUNDATIONPOSE_SHM_NAME,
        help="FoundationPose 共享内存名字，默认读取 scripts/camera/run_realtime.py 发布的 foundationpose_multi_pose。",
    )
    parser.add_argument(
        "--foundationpose_object_index",
        type=int,
        default=0,
        help="读取共享内存中的第几个工件位姿，0 表示第一个框选/注册的工件。",
    )
    parser.add_argument(
        "--foundationpose_max_age_s",
        type=float,
        default=0.25,
        help="FoundationPose 位姿最大允许延迟，单位 s；设为负数表示不检查 stale。",
    )
    parser.add_argument(
        "--foundationpose_object_frame_y_deg",
        type=float,
        default=None,
        help="FP 工件坐标系到训练工件坐标系的局部 Y 轴旋转修正，单位 deg；不传则使用这次实机对准固化的默认四元数。",
    )
    parser.add_argument(
        "--foundationpose_object_frame_quat",
        type=float,
        nargs=4,
        default=None,
        metavar=("W", "X", "Y", "Z"),
        help="FP 工件坐标系到训练工件坐标系的完整右乘四元数修正，wxyz；传入后优先于 --foundationpose_object_frame_y_deg。",
    )

    parser.add_argument(
        "--q_offset",
        type=float,
        nargs=4,
        default=DEFAULT_Q_OFFSET_WXYZ.tolist(),
        metavar=("W", "X", "Y", "Z"),
        help="和训练 object_pose_relative_to_ee 里的 q_offset 对齐，默认 [0,0,0,1]。",
    )

    parser.add_argument(
        "--set_tcp_offset_mm",
        type=float,
        nargs=6,
        default=None,
        metavar=("X", "Y", "Z", "RX", "RY", "RZ"),
        help="可选：运行前设置 xArm TCP offset，单位 mm/rad。只有确认方向一致时再使用。",
    )

    parser.add_argument(
        "--continue_on_pose_error",
        action="store_true",
        help="读取 TCP 位姿失败时尝试恢复并继续。默认遇到异常退出。",
    )

    parser.add_argument("--enable_gripper", action="store_true", help="启用 CTM2F110 夹爪控制。")
    parser.add_argument("--gripper_slave_id", type=int, default=1)
    parser.add_argument("--gripper_host_id", type=int, default=9)
    parser.add_argument("--gripper_auto_close", action="store_true", help="相对位姿误差进入阈值后自动闭合夹爪并抬升。")
    parser.add_argument(
        "--gripper_auto_close_mode",
        choices=("either", "pose", "z"),
        default="either",
        help="自动闭合触发模式：either=pose 阈值或 TCP z 任一满足；pose=只看位姿阈值；z=只看 TCP z 阈值。",
    )
    parser.add_argument("--gripper_trigger_dist_mm", type=float, default=15.0, help="自动闭合位置阈值，object_rel 距离 mm。")
    parser.add_argument("--gripper_trigger_ori_deg", type=float, default=30.0, help="自动闭合姿态阈值 deg；设 >=180 基本等于忽略姿态。")
    parser.add_argument("--gripper_trigger_steps", type=int, default=3, help="连续满足阈值多少步后触发自动闭合。")
    parser.add_argument("--gripper_force_close_below_tcp_z_mm", type=float, default=-5.0, help="硬安全触发：TCP z <= 该阈值时无视姿态/距离阈值，立即夹取并抬升；默认 -5mm。")
    parser.add_argument("--gripper_close_speed", type=int, default=20)
    parser.add_argument("--gripper_close_torque", type=int, default=40)
    parser.add_argument("--gripper_close_hold_s", type=float, default=0.8)
    parser.add_argument("--gripper_init_open_speed", type=int, default=60, help="启用夹爪时，deploy 初始化阶段自动打开夹爪的速度。")
    parser.add_argument("--gripper_init_open_torque", type=int, default=60, help="启用夹爪时，deploy 初始化阶段自动打开夹爪的力矩。")
    parser.add_argument("--gripper_init_open_timeout_s", type=float, default=4.0, help="初始化自动打开夹爪的等待超时。")
    parser.add_argument("--gripper_lift_mm", type=float, default=200.0)
    parser.add_argument("--gripper_lift_speed", type=float, default=50.0)
    parser.add_argument("--gripper_lift_acc", type=float, default=500.0)
    parser.add_argument("--gripper_lift_timeout_s", type=float, default=8.0)

    return parser.parse_args()


def _output_path(path_str: str | None, checkpoint: str, run_label: str) -> Path:
    if path_str:
        path = Path(path_str).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    policy_time = infer_policy_time_from_checkpoint(checkpoint)
    return make_policy_output_path(
        _PROJECT_DIR / "test_log",
        LOG_NAME,
        policy_time,
        f"{run_label}_deploy_state_pose",
    )


def main():
    args = parse_args()

    run_label = "sim2real" if args.real else "sim2sim"
    log_path = _output_path(args.log, args.ckpt, run_label)

    if args.object_pos_base is not None and args.object_quat_base is None:
        raise ValueError("使用 --object_pos_base 时必须同时提供 --object_quat_base")
    if args.object_quat_base is not None and args.object_pos_base is None:
        raise ValueError("--object_quat_base 只能和 --object_pos_base 一起使用")
    if args.gripper_auto_close and not args.enable_gripper:
        raise ValueError("使用 --gripper_auto_close 时必须同时传 --enable_gripper")
    if args.enable_gripper and CTM2F110Gripper is None:
        raise RuntimeError(f"无法导入 CTM2F110Gripper: {type(_GRIPPER_IMPORT_ERROR).__name__}: {_GRIPPER_IMPORT_ERROR}")
    if args.gripper_trigger_steps < 1:
        raise ValueError("--gripper_trigger_steps 必须 >= 1")

    action_scale_rad = np.deg2rad(args.action_scale_deg).astype(np.float32)
    action_clip_rad = np.deg2rad(args.action_clip_deg).astype(np.float32)

    q_offset = quat_normalize(np.asarray(args.q_offset, dtype=np.float32))
    foundationpose_object_frame_correction = None
    foundationpose_object_frame_correction_desc = None
    if args.foundationpose:
        if args.foundationpose_object_frame_quat is not None:
            foundationpose_object_frame_correction = quat_normalize(
                np.asarray(args.foundationpose_object_frame_quat, dtype=np.float32)
            )
            foundationpose_object_frame_correction_desc = "right-multiply explicit quat"
        elif args.foundationpose_object_frame_y_deg is not None:
            foundationpose_object_frame_correction = quat_from_euler_xyz(
                0.0,
                float(np.deg2rad(args.foundationpose_object_frame_y_deg)),
                0.0,
            )
            foundationpose_object_frame_correction_desc = (
                f"right-multiply Ry({args.foundationpose_object_frame_y_deg:.1f} deg)"
            )
        else:
            foundationpose_object_frame_correction = quat_normalize(
                DEFAULT_FOUNDATIONPOSE_OBJECT_FRAME_QUAT_WXYZ
            )
            foundationpose_object_frame_correction_desc = "right-multiply calibrated default quat"

    if args.object_rel_pose is not None:
        object_rel_pose_fixed = np.asarray(args.object_rel_pose, dtype=np.float32)
        object_rel_pose_fixed[3:7] = quat_normalize(object_rel_pose_fixed[3:7])
        pose_mode = "object_rel_pose_direct"
        object_pos_base_m = None
        object_quat_base = None
    elif args.foundationpose:
        object_rel_pose_fixed = None
        pose_mode = "foundationpose_shm_to_rel"
        object_pos_base_m = None
        object_quat_base = None
    else:
        object_rel_pose_fixed = None
        pose_mode = "object_pose_base_to_rel"
        object_pos_base_m = np.asarray(args.object_pos_base, dtype=np.float32)
        object_quat_base = quat_normalize(np.asarray(args.object_quat_base, dtype=np.float32))

    print("\n========== sim2real v87 deploy state pose ==========")
    print(f"[Config] VERSION={VERSION_TAG}")
    print(f"[Config] OBS_DIM={OBS_DIM}, ACT_DIM={ACT_DIM}")
    print(f"[Config] CONTROL_DT={CONTROL_DT:.3f}s")
    print(f"[Config] action_scale = {args.action_scale_deg:.3f} deg")
    print(f"[Config] action_clip  = ±{args.action_clip_deg:.3f} deg")
    print(f"[Config] obs_norm_clip = {args.obs_norm_clip:.3f} (0=disabled, match Isaac inference)")
    print("[Config] action: raw=网络原始归一化动作(无单位), scaled_delta_deg=raw*action_scale, final_delta_deg=限幅后执行增量")
    print(f"[Config] pose_mode={pose_mode}")
    print(f"[Config] run_label={run_label} (--real only changes log naming; SDK mode is not switched)")
    print(f"[Config] dry_run={args.dry_run}")
    print(f"[Config] enable_gripper={args.enable_gripper} auto_close={args.gripper_auto_close}")
    if args.gripper_auto_close:
        print(
            f"[Config] gripper mode={args.gripper_auto_close_mode}; "
            f"[Config] gripper trigger: dist<={args.gripper_trigger_dist_mm:.1f}mm "
            f"ori<={args.gripper_trigger_ori_deg:.1f}deg stable_steps={args.gripper_trigger_steps}; "
            f"force_close_if_tcp_z<={args.gripper_force_close_below_tcp_z_mm:.1f}mm; "
            f"close speed={args.gripper_close_speed} torque={args.gripper_close_torque}; "
            f"lift={args.gripper_lift_mm:.1f}mm"
        )
    print(f"[Output] log={log_path}")
    if object_rel_pose_fixed is not None:
        print(f"[Config] object_rel_pos_mm = {format_np(object_rel_pose_fixed[:3] * 1000.0, 3)}")
        print(f"[Config] object_rel_quat_wxyz = {format_np(object_rel_pose_fixed[3:7], 6)}")
    elif args.foundationpose:
        print(
            f"[Config] foundationpose_shm={args.foundationpose_shm_name} "
            f"object_index={args.foundationpose_object_index} max_age_s={args.foundationpose_max_age_s:.3f}"
        )
        print(
            "[Config] foundationpose object frame correction: "
            f"{foundationpose_object_frame_correction_desc}, "
            f"quat_wxyz={format_np(foundationpose_object_frame_correction, 6)}"
        )
        print("[Config] object base pose will be read live from FoundationPose each control step.")
    else:
        print(f"[Config] object_pos_base_mm = {format_np(object_pos_base_m * 1000.0, 3)}")
        print(f"[Config] object_quat_base_wxyz = {format_np(object_quat_base, 6)}")
    print("====================================================\n")

    if abs(args.action_scale_deg - TRAIN_ACTION_SCALE_DEG) > 1e-6 or abs(args.action_clip_deg - TRAIN_ACTION_CLIP_DEG) > 1e-6:
        print("[Warn] 当前部署 action_scale/action_clip 与训练环境不一致，v87 默认应为 ±0.6°/step。\n")

    arm = None
    gripper = None
    csv_f = None
    foundationpose_reader = None

    try:
        if args.foundationpose:
            foundationpose_reader = FoundationPoseShmReader(
                args.foundationpose_shm_name,
                object_index=args.foundationpose_object_index,
                max_age_s=args.foundationpose_max_age_s,
            )

        policy = V87RelativePoseJointStepPolicy(args.ckpt, device=args.device, obs_norm_clip=args.obs_norm_clip)

        print(f"\n[Main] 连接 xArm: {args.ip}  run_label={run_label}")
        arm = XArmAPI(args.ip, is_radian=True, check_joint_limit=False)
        print("[Arm] --real 只把日志标为 sim2real；不会调用 set_simulation_robot() 切换 SDK 模式。")

        arm.clean_error()
        arm.clean_warn()
        arm.motion_enable(enable=True)
        arm.set_state(0)

        if args.enable_gripper:
            gripper = CTM2F110Gripper(
                arm,
                slave_id=args.gripper_slave_id,
                host_id=args.gripper_host_id,
            )
            print(
                f"[Gripper] CTM2F110 enabled slave_id={args.gripper_slave_id} "
                f"host_id={args.gripper_host_id}"
            )
            try:
                print(f"[Gripper] status={gripper.status()}")
                print(f"[Gripper] finger1={gripper.feedback(finger=1)}")
                print(f"[Gripper] finger2={gripper.feedback(finger=2)}")
            except Exception as exc:
                print(f"[Warn] 读取夹爪初始反馈失败: {type(exc).__name__}: {exc}")

            print(
                f"[Gripper] init open speed={args.gripper_init_open_speed} "
                f"torque={args.gripper_init_open_torque} timeout={args.gripper_init_open_timeout_s:.1f}s"
            )
            if args.dry_run:
                print("[DryRun] would open gripper at init")
            else:
                gripper.open(
                    speed=args.gripper_init_open_speed,
                    torque=args.gripper_init_open_torque,
                    wait=True,
                    timeout=args.gripper_init_open_timeout_s,
                )
                try:
                    print(f"[Gripper] after init open status={gripper.status()}")
                    print(f"[Gripper] after init open finger1={gripper.feedback(finger=1)}")
                    print(f"[Gripper] after init open finger2={gripper.feedback(finger=2)}")
                except Exception as exc:
                    print(f"[Warn] 读取夹爪打开后反馈失败: {type(exc).__name__}: {exc}")

        if args.set_tcp_offset_mm is not None:
            print(f"[Arm] 设置 TCP offset: {args.set_tcp_offset_mm}")
            code = arm.set_tcp_offset(args.set_tcp_offset_mm, is_radian=True)
            print(f"[Arm] set_tcp_offset code={code}")
            time.sleep(0.2)

        if not args.no_move_init:
            print("\n=== Phase 1: Move to init joints (Mode 0) ===")
            enter_position_mode(arm)

            init_rad = np.deg2rad(INIT_JOINTS_DEG).tolist()
            code = arm.set_servo_angle(
                angle=init_rad,
                speed=np.deg2rad(INIT_SPEED_DEG),
                is_radian=True,
                wait=True,
            )
            if code != 0:
                print(f"[Error] 移到初始位置失败 code={code}")
                return

            print(f"[Main] 已到达训练初始位置: {INIT_JOINTS_DEG} deg")
            time.sleep(0.5)
        else:
            print("[Main] 跳过自动回初始位，将从当前关节角开始部署。")

        print("\n=== Phase 2: Switch to Mode 1 ===")
        enter_servo_mode(arm)

        last_action_clipped = np.zeros(ACT_DIM, dtype=np.float32)
        prev_target_rad = None
        gripper_ready_count = 0
        gripper_auto_done = False

        mode_str = "【步进模式】SPACE=执行 / Q=退出" if args.step_mode else f"【连续模式】{1 / CONTROL_DT:.0f} Hz"
        print(f"\n[Main] {mode_str}")
        print("-" * 120)

        csv_f = open(log_path, "w", encoding="utf-8")
        csv_f.write(
            "step,"
            "j1_deg,j2_deg,j3_deg,j4_deg,j5_deg,j6_deg,j7_deg,"
            "jrel1_deg,jrel2_deg,jrel3_deg,jrel4_deg,jrel5_deg,jrel6_deg,jrel7_deg,"
            "ee_x_mm,ee_y_mm,ee_z_mm,ee_qw,ee_qx,ee_qy,ee_qz,"
            "obj_base_x_mm,obj_base_y_mm,obj_base_z_mm,obj_base_qw,obj_base_qx,obj_base_qy,obj_base_qz,foundationpose_age_ms,"
            "obj_rel_x_mm,obj_rel_y_mm,obj_rel_z_mm,obj_rel_qw,obj_rel_qx,obj_rel_qy,obj_rel_qz,"
            "obj_rel_dist_mm,obj_rel_ori_err_deg,"
            "last_clip_norm0,last_clip_norm1,last_clip_norm2,last_clip_norm3,last_clip_norm4,last_clip_norm5,last_clip_norm6,"
            "raw_act0,raw_act1,raw_act2,raw_act3,raw_act4,raw_act5,raw_act6,"
            "scaled_delta_deg0,scaled_delta_deg1,scaled_delta_deg2,scaled_delta_deg3,scaled_delta_deg4,scaled_delta_deg5,scaled_delta_deg6,"
            "clip_norm0,clip_norm1,clip_norm2,clip_norm3,clip_norm4,clip_norm5,clip_norm6,"
            "final_delta_deg0,final_delta_deg1,final_delta_deg2,final_delta_deg3,final_delta_deg4,final_delta_deg5,final_delta_deg6,"
            "target1_deg,target2_deg,target3_deg,target4_deg,target5_deg,target6_deg,target7_deg,"
            "track_err1_deg,track_err2_deg,track_err3_deg,track_err4_deg,track_err5_deg,track_err6_deg,track_err7_deg,"
            "t_pose_ms,t_inf_ms,loop_ms,code\n"
        )

        step = 0
        while True:
            if not args.step_mode and step >= args.steps:
                print(f"[Main] 已完成 {args.steps} 步，退出")
                break

            t0 = time.perf_counter()

            try:
                joint_rad = get_joint_positions(arm)

                ee_pos_base_m, ee_quat_base, raw_tcp_pose = get_tcp_pose_base(arm)

                foundationpose_age_s = float("nan")
                foundationpose_timestamp = float("nan")
                foundationpose_num_objects = 0
                object_pos_base_current_m = None
                object_quat_base_current = None

                if foundationpose_reader is not None:
                    (
                        object_pos_base_current_m,
                        object_quat_base_current,
                        foundationpose_age_s,
                        foundationpose_timestamp,
                        foundationpose_num_objects,
                    ) = foundationpose_reader.read_pose_base()
                    object_quat_base_current = quat_mul(
                        object_quat_base_current,
                        foundationpose_object_frame_correction,
                    )
                elif object_rel_pose_fixed is None:
                    object_pos_base_current_m = object_pos_base_m
                    object_quat_base_current = object_quat_base

                if object_rel_pose_fixed is None:
                    object_rel_pose = object_pose_relative_to_ee_np(
                        object_pos_base_current_m,
                        object_quat_base_current,
                        ee_pos_base_m,
                        ee_quat_base,
                        q_offset_wxyz=q_offset,
                    )
                else:
                    object_rel_pose = object_rel_pose_fixed.copy()

            except Exception as exc:
                print(f"[Warn] 读取关节/TCP/FoundationPose 位姿失败: {type(exc).__name__}: {exc}")
                if args.continue_on_pose_error:
                    recover_servo_mode(arm)
                    continue
                raise

            t1 = time.perf_counter()

            if not check_joint_safe(joint_rad, margin_deg=0.5):
                print("[Error] 当前关节角已接近安全限位，停止部署。")
                print(f"        joint_deg={format_deg(joint_rad, 3)}")
                break

            track_err_deg = np.zeros(ACT_DIM, dtype=np.float32)
            if prev_target_rad is not None:
                track_err_deg = np.rad2deg(joint_rad - prev_target_rad).astype(np.float32)
                max_track_err = float(np.max(np.abs(track_err_deg)))
                if max_track_err > 0.5:
                    print(f"[Warn] 上一步实机跟踪误差较大: {track_err_deg.round(3).tolist()} deg")

            obs = build_obs(joint_rad, object_rel_pose, last_action_clipped)
            action = policy.predict(obs)
            t2 = time.perf_counter()

            action_clipped = np.clip(action, -1.0, 1.0).astype(np.float32)
            scaled_delta_rad = action.astype(np.float32) * action_scale_rad
            delta_rad = np.clip(scaled_delta_rad, -action_clip_rad, action_clip_rad).astype(np.float32)

            target_rad = np.clip(joint_rad + delta_rad, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH).astype(np.float32)
            scaled_delta_deg = np.rad2deg(scaled_delta_rad).astype(np.float32)
            delta_deg = np.rad2deg(delta_rad).astype(np.float32)
            joint_rel_deg = np.rad2deg(joint_rad - DEFAULT_JOINT_POS_RAD).astype(np.float32)

            obj_rel_dist_m = float(np.linalg.norm(object_rel_pose[:3]))
            obj_rel_dist_mm = obj_rel_dist_m * 1000.0
            obj_rel_ori_err_deg = quat_to_angle_deg(object_rel_pose[3:7])
            ee_pos_base_mm = ee_pos_base_m * 1000.0
            object_base_pos_mm = np.full(3, np.nan, dtype=np.float32)
            object_base_quat = np.full(4, np.nan, dtype=np.float32)
            if object_pos_base_current_m is not None and object_quat_base_current is not None:
                object_base_pos_mm = object_pos_base_current_m * 1000.0
                object_base_quat = object_quat_base_current.astype(np.float32)
            foundationpose_age_ms = foundationpose_age_s * 1000.0 if np.isfinite(foundationpose_age_s) else float("nan")
            obj_rel_pos_mm = object_rel_pose[:3] * 1000.0
            obj_rel_quat = object_rel_pose[3:7]

            gripper_threshold_ok = False
            gripper_trigger_now = False
            gripper_force_z_trigger = False
            gripper_trigger_reason = ""
            if args.gripper_auto_close and not gripper_auto_done:
                gripper_threshold_ok = (
                    obj_rel_dist_mm <= float(args.gripper_trigger_dist_mm)
                    and obj_rel_ori_err_deg <= float(args.gripper_trigger_ori_deg)
                )
                gripper_ready_count = gripper_ready_count + 1 if gripper_threshold_ok else 0
                gripper_pose_trigger = gripper_ready_count >= int(args.gripper_trigger_steps)
                if gripper_pose_trigger and args.gripper_auto_close_mode in ("either", "pose"):
                    gripper_trigger_now = True
                    gripper_trigger_reason = (
                        f"threshold reached for {gripper_ready_count} consecutive step(s): "
                        f"dist={obj_rel_dist_mm:.2f}mm <= {args.gripper_trigger_dist_mm:.2f}mm, "
                        f"ori={obj_rel_ori_err_deg:.2f}deg <= {args.gripper_trigger_ori_deg:.2f}deg"
                    )
                tcp_z_mm = float(raw_tcp_pose[2])
                gripper_force_z_trigger = tcp_z_mm <= float(args.gripper_force_close_below_tcp_z_mm)
                if gripper_force_z_trigger and args.gripper_auto_close_mode in ("either", "z"):
                    gripper_trigger_now = True
                    gripper_trigger_reason = (
                        f"SAFETY tcp_z={tcp_z_mm:.2f}mm <= "
                        f"{args.gripper_force_close_below_tcp_z_mm:.2f}mm; force close + lift"
                    )

            if args.step_mode:
                print(f"\n[Step {step:4d}]")
                print(f"  joints deg   cur={format_deg(joint_rad, 3)}")
                print(f"               rel={format_np(joint_rel_deg, 3)}")
                print(f"  tcp          pos_mm={format_np(ee_pos_base_mm, 3)}  quat_wxyz={format_np(ee_quat_base, 6)}")
                if object_pos_base_current_m is not None and object_quat_base_current is not None:
                    fp_age_text = f"  fp_age={foundationpose_age_ms:.1f}ms" if foundationpose_reader is not None else ""
                    print(
                        f"  object base  pos_mm={format_np(object_base_pos_mm, 3)}  "
                        f"quat_wxyz={format_np(object_base_quat, 6)}"
                        f"{fp_age_text}"
                    )
                print(
                    f"  object rel   pos_mm={format_np(obj_rel_pos_mm, 3)}  "
                    f"quat_wxyz={format_np(obj_rel_quat, 6)}  "
                    f"dist={obj_rel_dist_mm:.2f}mm  ori={obj_rel_ori_err_deg:.2f}deg"
                )
                print(f"  action       last_clip_norm={format_np(last_action_clipped, 5)}")
                print(f"               raw={format_np(action, 5)}")
                print(f"               scaled_delta_deg={format_np(scaled_delta_deg, 4)}")
                print(f"               final_delta_deg={format_np(delta_deg, 4)}  clip_norm={format_np(action_clipped, 5)}")
                print(f"  target       target_deg={format_deg(target_rad, 3)}")
                print(f"               track_err_deg={format_np(track_err_deg, 4)}")
                print(f"  timing       pose={((t1 - t0) * 1e3):.1f}ms  inf={((t2 - t1) * 1e3):.1f}ms")
                if args.gripper_auto_close and not gripper_auto_done:
                    print(
                        f"  gripper_auto mode={args.gripper_auto_close_mode} ok={gripper_threshold_ok} "
                        f"stable={gripper_ready_count}/{args.gripper_trigger_steps} "
                        f"threshold=({args.gripper_trigger_dist_mm:.1f}mm, {args.gripper_trigger_ori_deg:.1f}deg) "
                        f"force_z={gripper_force_z_trigger}"
                    )

            if gripper_trigger_now:
                print(f"[Auto] grasp trigger: {gripper_trigger_reason}")
                execute_gripper_close_and_lift(arm, gripper, args, raw_tcp_pose)
                gripper_auto_done = True
                break

            if args.step_mode:
                if not _wait_for_step():
                    break

            send_code = 0
            if not args.dry_run:
                send_code = arm.set_servo_angle_j(
                    angles=target_rad.tolist(),
                    speed=np.deg2rad(100.0),
                    mvacc=np.deg2rad(1000.0),
                    mvtime=0,
                )

                if send_code != 0:
                    print(f"[Warn] set_servo_angle_j failed: {send_code}，尝试恢复...")
                    recover_servo_mode(arm)

            prev_target_rad = target_rad.copy()

            if args.step_mode and not args.dry_run:
                time.sleep(CONTROL_DT)
                try:
                    cur_rad = get_joint_positions(arm)
                    arm.set_servo_angle_j(
                        angles=cur_rad.tolist(),
                        speed=np.deg2rad(100.0),
                        mvacc=np.deg2rad(1000.0),
                        mvtime=0,
                    )
                except Exception as exc:
                    print(f"[Warn] 步进保持当前位置失败: {type(exc).__name__}: {exc}")

            last_action_clipped = action_clipped.copy()

            elapsed_ms = (time.perf_counter() - t0) * 1000.0

            if not args.step_mode and step % 10 == 0:
                fp_age_text = f" fp_age={foundationpose_age_ms:.1f}ms" if foundationpose_reader is not None else ""
                print(
                    f"[step {step:4d}] "
                    f"obj_rel_mm={format_np(obj_rel_pos_mm, 2)} "
                    f"dist={obj_rel_dist_mm:.1f}mm ori={obj_rel_ori_err_deg:.2f}deg | "
                    f"scaled={format_np(scaled_delta_deg, 3)} deg "
                    f"final={format_np(delta_deg, 3)} deg | "
                    f"track_err={format_np(track_err_deg, 3)} deg | "
                    f"pose={(t1 - t0) * 1e3:.1f}ms "
                    f"inf={(t2 - t1) * 1e3:.1f}ms "
                    f"loop={elapsed_ms:.1f}ms{fp_age_text} | code={send_code}"
                )

            csv_f.write(
                f"{step},"
                + ",".join(f"{v:.6f}" for v in np.rad2deg(joint_rad))
                + ","
                + ",".join(f"{v:.6f}" for v in joint_rel_deg)
                + ","
                + ",".join(f"{v:.6f}" for v in ee_pos_base_mm)
                + ","
                + ",".join(f"{v:.8f}" for v in ee_quat_base)
                + ","
                + ",".join(f"{v:.6f}" for v in object_base_pos_mm)
                + ","
                + ",".join(f"{v:.8f}" for v in object_base_quat)
                + f",{foundationpose_age_ms:.3f},"
                + ",".join(f"{v:.6f}" for v in obj_rel_pos_mm)
                + ","
                + ",".join(f"{v:.8f}" for v in obj_rel_quat)
                + f",{obj_rel_dist_mm:.6f},{obj_rel_ori_err_deg:.8f},"
                + ",".join(f"{v:.8f}" for v in last_action_clipped)
                + ","
                + ",".join(f"{v:.8f}" for v in action)
                + ","
                + ",".join(f"{v:.8f}" for v in scaled_delta_deg)
                + ","
                + ",".join(f"{v:.8f}" for v in action_clipped)
                + ","
                + ",".join(f"{v:.8f}" for v in delta_deg)
                + ","
                + ",".join(f"{v:.6f}" for v in np.rad2deg(target_rad))
                + ","
                + ",".join(f"{v:.6f}" for v in track_err_deg)
                + f",{(t1 - t0) * 1e3:.3f}"
                + f",{(t2 - t1) * 1e3:.3f}"
                + f",{elapsed_ms:.3f}"
                + f",{send_code}\n"
            )
            csv_f.flush()

            step += 1

            if not args.step_mode:
                sleep_t = CONTROL_DT - (time.perf_counter() - t0)
                if sleep_t > 0:
                    time.sleep(sleep_t)
                else:
                    print(f"[Warn] step {step} 超时 {-sleep_t * 1e3:.1f} ms，实际控制频率低于 50Hz")

    except KeyboardInterrupt:
        print("\n[Main] 用户中断")

    finally:
        if foundationpose_reader is not None:
            foundationpose_reader.close()

        if csv_f is not None:
            csv_f.close()
            print(f"[Main] 日志已保存: {log_path}")

        if arm is not None:
            try:
                arm.set_mode(0)
                arm.set_state(0)
                print("[Main] 已切回 Mode 0")
                arm.disconnect()
                print("[Main] 已断开机械臂连接")
            except Exception as exc:
                print(f"[Warn] 退出时机械臂复位/断开异常: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
