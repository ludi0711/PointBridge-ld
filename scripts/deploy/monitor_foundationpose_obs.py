#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Print the FoundationPose object pose exactly as deploy_policy_pose uses it."""

import argparse
import os
import sys
import time
from multiprocessing import resource_tracker, shared_memory
from pathlib import Path

import numpy as np


CURRENT_DIR = Path(__file__).resolve().parent
PROJECT_DIR = CURRENT_DIR.parents[1]

FOUNDATIONPOSE_SHM_NAME = "foundationpose_multi_pose"
FOUNDATIONPOSE_HEADER_BYTES = 12
FOUNDATIONPOSE_POSE_BYTES = 128
FOUNDATIONPOSE_MAX_OBJECTS = 10

INIT_JOINTS_DEG = [-0.4, -47.6, 0.0, 1.8, -0.2, 49.4, -0.1]
DEFAULT_JOINT_POS_RAD = np.deg2rad(INIT_JOINTS_DEG).astype(np.float32)
OBS_DIM = 21
ACT_DIM = 7

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


def add_xarm_sdk_to_path() -> None:
    candidates = []
    env_path = os.environ.get("XARM_SDK_DIR", "").strip()
    if env_path:
        candidates.append(Path(env_path).expanduser())

    candidates += [
        PROJECT_DIR / "third_party" / "xArm_Python_SDK_master",
        CURRENT_DIR / "third_party" / "xArm_Python_SDK_master",
        CURRENT_DIR.parent / "third_party" / "xArm_Python_SDK_master",
        CURRENT_DIR.parent.parent / "third_party" / "xArm_Python_SDK_master",
        Path("/home/gxai/Desktop/embodied-ai-project/modules/control/scripts/third_party/xArm_Python_SDK_master"),
        Path("/home/gxai/Desktop/zjj/embodied-ai-project/modules/control/scripts/third_party/xArm_Python_SDK_master"),
        Path("/home/gxai/Desktop/test_pickandprice/embodied-ai-project/modules/control/scripts/third_party/xArm_Python_SDK_master"),
        Path("/home/gxai/Desktop/task_node_neo/embodied-ai-project/modules/control/scripts/third_party/xArm_Python_SDK_master"),
    ]

    for path in candidates:
        if (path / "xarm").exists():
            sys.path.insert(0, str(path))
            print(f"[SDK] using xArm SDK: {path}")
            return

    raise RuntimeError("xArm SDK not found; set XARM_SDK_DIR=/path/to/xArm_Python_SDK_master")


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    n = np.linalg.norm(q)
    if n < 1.0e-8:
        raise ValueError("quaternion norm is too small")
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
    r = np.asarray(rot, dtype=np.float64)
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


def rot_matrix_to_rpy_xyz(rot: np.ndarray) -> np.ndarray:
    r = np.asarray(rot, dtype=np.float64)
    sy = np.sqrt(r[0, 0] * r[0, 0] + r[1, 0] * r[1, 0])
    singular = sy < 1.0e-6
    if not singular:
        roll = np.arctan2(r[2, 1], r[2, 2])
        pitch = np.arctan2(-r[2, 0], sy)
        yaw = np.arctan2(r[1, 0], r[0, 0])
    else:
        roll = np.arctan2(-r[1, 2], r[1, 1])
        pitch = np.arctan2(-r[2, 0], sy)
        yaw = 0.0
    return np.rad2deg(np.array([roll, pitch, yaw], dtype=np.float64))


def quat_to_rot_matrix(q: np.ndarray) -> np.ndarray:
    w, x, y, z = quat_normalize(q).astype(np.float64)
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=np.float64,
    )


def quat_to_rpy_deg(q: np.ndarray) -> np.ndarray:
    return rot_matrix_to_rpy_xyz(quat_to_rot_matrix(q))


def quat_to_angle_deg(q: np.ndarray) -> float:
    q = quat_normalize(q)
    w = abs(float(q[0]))
    xyz_norm = float(np.linalg.norm(q[1:4]))
    angle = 2.0 * np.arctan2(xyz_norm, max(w, 1.0e-8))
    return float(np.rad2deg(angle))


def quat_angle_between_deg(q1: np.ndarray, q2: np.ndarray) -> float:
    return quat_to_angle_deg(quat_mul(quat_inv(q1), q2))


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


def build_obs(joint_rad: np.ndarray, object_rel_pose: np.ndarray, last_action_clipped: np.ndarray) -> np.ndarray:
    joint_pos_rel = joint_rad.astype(np.float32) - DEFAULT_JOINT_POS_RAD
    obs = np.concatenate(
        [
            joint_pos_rel.astype(np.float32),
            object_rel_pose.astype(np.float32),
            last_action_clipped.astype(np.float32),
        ]
    ).astype(np.float32)
    if obs.shape != (OBS_DIM,):
        raise RuntimeError(f"obs shape={obs.shape}, expected {(OBS_DIM,)}")
    return obs


class FoundationPoseSharedMemoryReader:
    def __init__(self, name: str, object_index: int, max_age_s: float):
        self.name = name
        self.object_index = int(object_index)
        self.max_age_s = float(max_age_s)
        self.shm = shared_memory.SharedMemory(name=self.name)
        # Consumer only: do not let this monitor unlink the producer-owned name.
        resource_tracker.unregister(self.shm._name, "shared_memory")

    def close(self) -> None:
        if self.shm is not None:
            self.shm.close()
            self.shm = None

    def read(self):
        buf = self.shm.buf
        num_objects = int(np.frombuffer(buf[0:4], dtype=np.int32, count=1)[0])
        timestamp = float(np.frombuffer(buf[4:12], dtype=np.float64, count=1)[0])
        if num_objects <= 0:
            raise RuntimeError("FoundationPose shared memory exists, but no object pose has been published yet")
        if self.object_index >= num_objects:
            raise RuntimeError(f"object_index={self.object_index}, but num_objects={num_objects}")
        age_s = time.time() - timestamp
        if self.max_age_s >= 0.0 and age_s > self.max_age_s:
            raise RuntimeError(f"stale FoundationPose pose: age={age_s:.3f}s > {self.max_age_s:.3f}s")

        offset = FOUNDATIONPOSE_HEADER_BYTES + self.object_index * FOUNDATIONPOSE_POSE_BYTES
        pose = np.frombuffer(buf[offset : offset + FOUNDATIONPOSE_POSE_BYTES], dtype=np.float64, count=16)
        pose = pose.reshape(4, 4).copy()
        if not np.all(np.isfinite(pose)):
            raise RuntimeError("FoundationPose pose contains non-finite values")

        obj_pos_base_m = pose[:3, 3].astype(np.float32)
        obj_quat_base = quat_from_rot_matrix(pose[:3, :3])
        obj_rpy_deg = rot_matrix_to_rpy_xyz(pose[:3, :3]).astype(np.float32)
        return pose, obj_pos_base_m, obj_quat_base, obj_rpy_deg, age_s, timestamp, num_objects


def get_tcp_pose_base(arm):
    code, pose = arm.get_position(is_radian=True)
    if code != 0 or pose is None:
        raise RuntimeError(f"xArm get_position failed: code={code}")
    pose = np.asarray(pose[:6], dtype=np.float32)
    ee_pos_m = pose[:3] * 0.001
    ee_quat = quat_from_euler_xyz(float(pose[3]), float(pose[4]), float(pose[5]))
    return ee_pos_m.astype(np.float32), ee_quat.astype(np.float32), pose


def get_joint_positions(arm) -> np.ndarray:
    code, raw_angles = arm.get_servo_angle(is_radian=True)
    if code != 0 or raw_angles is None:
        raise RuntimeError(f"xArm get_servo_angle failed: code={code}")
    return np.asarray(raw_angles[:7], dtype=np.float32)


def fmt(values, ndigits=2):
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 0:
        return f"{float(arr):.{ndigits}f}"
    flat = arr.reshape(-1)
    return "[" + ", ".join(f"{float(v):.{ndigits}f}" for v in flat) + "]"


def print_block(
    step,
    fp_data,
    tcp_data,
    joint_rad,
    last_action,
    object_frame_correction,
    object_frame_correction_desc,
    full_matrix=False,
):
    pose_base, obj_pos_base_m, obj_quat_base, obj_rpy_deg, age_s, timestamp, num_objects = fp_data
    ee_pos_base_m, ee_quat_base, raw_tcp_pose, tcp_offset = tcp_data
    obj_quat_train_base = quat_mul(obj_quat_base, object_frame_correction)
    obj_rpy_train_deg = quat_to_rpy_deg(obj_quat_train_base)

    object_rel_pose = object_pose_relative_to_ee_np(
        obj_pos_base_m,
        obj_quat_train_base,
        ee_pos_base_m,
        ee_quat_base,
    )
    obs = build_obs(joint_rad, object_rel_pose, last_action)

    joint_rel_rad = obs[0:7]
    obj_rel_obs = obs[7:14]
    last_action_obs = obs[14:21]
    rel_pos_mm = object_rel_pose[:3] * 1000.0
    rel_quat = object_rel_pose[3:7]

    print("\n" + "=" * 100)
    print(f"[step {step}] FP age={age_s * 1000.0:.2f} ms  timestamp={timestamp:.2f}  num_objects={num_objects}")

    print("[1] FP raw object pose in base_link / xArm base  (from shared memory matrix)")
    print(f"    xyz_mm={fmt(obj_pos_base_m * 1000.0)}")
    print(f"    quat_wxyz={fmt(obj_quat_base)}")
    print(f"    rpy_xyz_deg={fmt(obj_rpy_deg)}")
    if full_matrix:
        print("    T_base_object=")
        print(np.array2string(pose_base, precision=2, suppress_small=True))

    print("[2] xArm7/deploy received object pose  (after FP -> training frame correction)")
    print(f"    correction={object_frame_correction_desc}")
    print(f"    object_base_pos_m={fmt(obj_pos_base_m)}")
    print(f"    object_base_pos_mm={fmt(obj_pos_base_m * 1000.0)}")
    print(f"    object_base_quat_wxyz={fmt(obj_quat_train_base)}")
    print(f"    object_base_rpy_xyz_deg={fmt(obj_rpy_train_deg)}")

    print("[3] TCP/end-effector pose in xArm base  (from xArm SDK get_position)")
    print(f"    controller_tcp_offset={fmt(tcp_offset)}")
    print(f"    tcp_xyz_mm={fmt(ee_pos_base_m * 1000.0)}")
    print(f"    tcp_raw_rpy_rad={fmt(raw_tcp_pose[3:6])}")
    print(f"    tcp_rpy_xyz_deg={fmt(np.rad2deg(raw_tcp_pose[3:6]))}")
    print(f"    tcp_quat_wxyz={fmt(ee_quat_base)}")

    print("[4] object relative pose in TCP frame  (this is obs[7:14])")
    print(f"    rel_pos_tcp_mm={fmt(rel_pos_mm)}")
    print(f"    rel_quat_wxyz={fmt(rel_quat)}")
    print(f"    rel_rpy_xyz_deg={fmt(quat_to_rpy_deg(rel_quat))}")
    print(f"    rel_dist_mm={float(np.linalg.norm(rel_pos_mm)):.2f}")
    print(f"    rel_ori_angle_deg={quat_to_angle_deg(rel_quat):.2f}")
    if np.all(np.isfinite(tcp_offset[:3])) and np.all(np.abs(tcp_offset[3:6]) < 1.0e-6):
        candidate_tcp_offset = tcp_offset[:3] + rel_pos_mm
        print(
            "    candidate_tcp_offset_xyz_mm_if_this_pose_is_ideal="
            f"{fmt(candidate_tcp_offset)}"
        )
    print("    candidate_object_frame_quat_if_this_pose_is_ideal:")
    for idx, sym_q in enumerate(DEFAULT_GRASP_SYMMETRY_QUATS_WXYZ):
        target_offset = quat_mul(DEFAULT_Q_OFFSET_WXYZ, sym_q)
        candidate_corr = quat_mul(
            quat_mul(quat_inv(obj_quat_base), ee_quat_base),
            quat_inv(target_offset),
        )
        if candidate_corr[0] < 0.0:
            candidate_corr = -candidate_corr
        delta_deg = quat_angle_between_deg(object_frame_correction, candidate_corr)
        print(
            f"      sym={idx} quat_wxyz={fmt(candidate_corr, 4)} "
            f"delta_from_current_deg={delta_deg:.2f}"
        )

    print("[5] policy obs fields used by deploy, split by group")
    print(f"    joint_rel_rad={fmt(joint_rel_rad)}")
    print(f"    joint_rel_deg={fmt(np.rad2deg(joint_rel_rad))}")
    print(f"    obs_object_rel={fmt(obj_rel_obs)}")
    print(f"    obs_last_action={fmt(last_action_obs)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Monitor FoundationPose pose as xArm deploy obs.")
    parser.add_argument("--ip", default="192.168.73.229", help="xArm controller IP")
    parser.add_argument("--shm-name", default=FOUNDATIONPOSE_SHM_NAME)
    parser.add_argument("--object-index", type=int, default=0)
    parser.add_argument("--max-age-s", type=float, default=-1.0, help="negative disables stale-pose check")
    parser.add_argument(
        "--foundationpose-object-frame-y-deg",
        type=float,
        default=None,
        help="FP object frame -> training object frame local-Y correction. If omitted, use calibrated default quaternion.",
    )
    parser.add_argument(
        "--foundationpose-object-frame-quat",
        type=float,
        nargs=4,
        default=None,
        metavar=("W", "X", "Y", "Z"),
        help="Explicit FP object frame -> training frame right-multiply quaternion, wxyz. Overrides local-Y correction.",
    )
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--steps", type=int, default=0, help="0 means run until Ctrl-C")
    parser.add_argument("--full-matrix", action="store_true", help="also print full T_base_object matrix")
    return parser.parse_args()


def main():
    args = parse_args()
    add_xarm_sdk_to_path()
    from xarm.wrapper import XArmAPI
    if args.foundationpose_object_frame_quat is not None:
        object_frame_correction = quat_normalize(np.asarray(args.foundationpose_object_frame_quat, dtype=np.float32))
        object_frame_correction_desc = f"right-multiply explicit quat {fmt(object_frame_correction, 4)}"
    elif args.foundationpose_object_frame_y_deg is not None:
        object_frame_correction = quat_from_euler_xyz(
            0.0,
            float(np.deg2rad(args.foundationpose_object_frame_y_deg)),
            0.0,
        )
        object_frame_correction_desc = f"right-multiply Ry({args.foundationpose_object_frame_y_deg:.2f} deg)"
    else:
        object_frame_correction = quat_normalize(DEFAULT_FOUNDATIONPOSE_OBJECT_FRAME_QUAT_WXYZ)
        object_frame_correction_desc = f"right-multiply calibrated default quat {fmt(object_frame_correction, 4)}"

    reader = None
    arm = None
    last_action = np.zeros(ACT_DIM, dtype=np.float32)

    try:
        reader = FoundationPoseSharedMemoryReader(args.shm_name, args.object_index, args.max_age_s)
        print(f"[FP] connected shm={args.shm_name} object_index={args.object_index}")

        arm = XArmAPI(args.ip)
        print(f"[xArm] connected ip={args.ip}")
        print(f"[xArm] err_warn={arm.get_err_warn_code()}")

        step = 0
        while args.steps <= 0 or step < args.steps:
            step += 1
            try:
                fp_data = reader.read()
                tcp_offset = np.asarray(getattr(arm, "tcp_offset", [np.nan] * 6), dtype=np.float32)
                tcp_data = (*get_tcp_pose_base(arm), tcp_offset)
                joint_rad = get_joint_positions(arm)
                print_block(
                    step,
                    fp_data,
                    tcp_data,
                    joint_rad,
                    last_action,
                    object_frame_correction,
                    object_frame_correction_desc,
                    full_matrix=args.full_matrix,
                )
            except Exception as exc:
                print(f"[warn] {type(exc).__name__}: {exc}")
            time.sleep(max(0.01, args.interval))
    except KeyboardInterrupt:
        print("\n[Main] interrupted")
    finally:
        if reader is not None:
            reader.close()
        if arm is not None:
            arm.disconnect()


if __name__ == "__main__":
    main()
