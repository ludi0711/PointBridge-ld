#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 Isaac 首帧 obs 初始化，并用 xArm SDK 反馈继续滚动策略。"""

import argparse
import csv
import sys
import time
from pathlib import Path

import numpy as np

_CURRENT_DIR = Path(__file__).resolve().parent
_PROJECT_DIR = _CURRENT_DIR.parents[1]
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from tools.policy_log_paths import (
    fallback_policy_time,
    infer_policy_time_from_checkpoint,
    infer_policy_time_from_csv,
    make_policy_output_path,
)

from deploy_policy_pose import (
    ACT_DIM,
    CONTROL_DT,
    DEFAULT_JOINT_POS_RAD,
    DEFAULT_Q_OFFSET_WXYZ,
    INIT_SPEED_DEG,
    JOINT_LIMITS_HIGH,
    JOINT_LIMITS_LOW,
    OBS_DIM,
    TRAIN_ACTION_CLIP_DEG,
    TRAIN_ACTION_SCALE_DEG,
    V87RelativePoseJointStepPolicy,
    XArmAPI,
    _wait_for_step,
    build_obs,
    enter_position_mode,
    enter_servo_mode,
    format_deg,
    format_np,
    get_tcp_pose_base,
    get_joint_positions,
    object_pose_relative_to_ee_np,
    quat_inv,
    quat_mul,
    quat_mul_raw,
    quat_normalize,
    recover_servo_mode,
)


LOG_NAME = "xarm7_pick_pose"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Use Isaac first obs as initial xArm obs, then roll out through xArm SDK feedback."
    )
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--isaac_csv", type=str, required=True)
    parser.add_argument("--ip", type=str, default="192.168.73.229")
    parser.add_argument("--real", action="store_true", help="Tag outputs as sim2real data only; does not switch SDK robot mode.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--obs_norm_clip", type=float, default=0.0)
    parser.add_argument("--action_scale_deg", type=float, default=TRAIN_ACTION_SCALE_DEG)
    parser.add_argument("--action_clip_deg", type=float, default=TRAIN_ACTION_CLIP_DEG)
    parser.add_argument("--start_row", type=int, default=0)
    parser.add_argument("--steps", type=int, default=-1)
    parser.add_argument("--sample_dt", type=float, default=CONTROL_DT)
    parser.add_argument("--step_mode", action="store_true")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument("--no_move_init", action="store_true")
    parser.add_argument("--ignore_done", action="store_true")
    parser.add_argument("--log", type=str, default=None, help="Default: test_log/xarm7_pick_pose-<policy_time>/sdk_obs_rollout_<sim2real|sim2sim>_<policy_time>.csv")
    parser.add_argument(
        "--object_mode",
        choices=("sdk", "hold", "isaac"),
        default="sdk",
        help="sdk recomputes object_relative_pose from SDK TCP; hold keeps first pose; isaac uses each row as oracle.",
    )
    parser.add_argument(
        "--action_source",
        choices=("policy", "logged"),
        default="policy",
        help="policy runs inference on SDK-built obs; logged sends Isaac mdp_raw_action_j*.",
    )
    return parser.parse_args()


def _boolish(value: str | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def _output_path(path_str: str | None, checkpoint: str, input_csv: str, run_label: str) -> Path:
    if path_str:
        path = Path(path_str).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        return path

    fallback = infer_policy_time_from_csv(input_csv, LOG_NAME) or fallback_policy_time("unknown_policy")
    policy_time = infer_policy_time_from_checkpoint(checkpoint, fallback=fallback)
    return make_policy_output_path(_PROJECT_DIR / "test_log", LOG_NAME, policy_time, f"sdk_obs_rollout_{run_label}")


def _load_rows(path: str, start_row: int, steps: int):
    rows = []
    with Path(path).expanduser().open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {path}")
        obs_cols = [f"mdp_obs_{i:02d}" for i in range(OBS_DIM)]
        raw_cols = [f"mdp_raw_action_j{i}" for i in range(1, ACT_DIM + 1)]
        cmd_cols = [f"cmd_delta_j{i}_deg" for i in range(1, ACT_DIM + 1)]
        missing = [c for c in obs_cols + raw_cols if c not in reader.fieldnames]
        if missing:
            raise RuntimeError(f"CSV missing columns: {missing[:8]}")
        old_extra = f"mdp_obs_{OBS_DIM:02d}"
        if old_extra in reader.fieldnames:
            raise RuntimeError(
                f"CSV looks like an old 28D velocity-observation log because it contains {old_extra}; "
                "regenerate it with the 21D no-velocity config."
            )
        has_cmd = all(c in reader.fieldnames for c in cmd_cols)

        for csv_row, row in enumerate(reader):
            if csv_row < start_row:
                continue
            if steps >= 0 and len(rows) >= steps:
                break
            obs = np.asarray([float(row[c]) for c in obs_cols], dtype=np.float32)
            logged_raw = np.asarray([float(row[c]) for c in raw_cols], dtype=np.float32)
            logged_cmd = None
            if has_cmd:
                logged_cmd = np.asarray([float(row[c]) for c in cmd_cols], dtype=np.float32)
            rows.append((csv_row, row, obs, logged_raw, logged_cmd))
    if not rows:
        raise RuntimeError(f"No rows loaded from {path}")
    return rows


def _quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    q = quat_normalize(q)
    vq = np.array([0.0, v[0], v[1], v[2]], dtype=np.float32)
    return quat_mul_raw(quat_mul_raw(q, vq), quat_inv(q))[1:4]


def _infer_object_base_pose_from_rel(
    object_rel_pose: np.ndarray,
    ee_pos_base_m: np.ndarray,
    ee_quat_base_wxyz: np.ndarray,
    q_offset_wxyz: np.ndarray = DEFAULT_Q_OFFSET_WXYZ,
):
    rel_pos = object_rel_pose[:3].astype(np.float32)
    rel_quat = quat_normalize(object_rel_pose[3:7])
    ee_quat = quat_normalize(ee_quat_base_wxyz)
    q_offset = quat_normalize(q_offset_wxyz)

    object_pos_base_m = ee_pos_base_m.astype(np.float32) + _quat_apply(ee_quat, rel_pos)
    target_quat_base = quat_mul(ee_quat, rel_quat)
    object_quat_base = quat_mul(target_quat_base, quat_inv(q_offset))
    return object_pos_base_m.astype(np.float32), object_quat_base.astype(np.float32)


def _write_header(writer):
    header = ["csv_row", "episode", "step", "g_step", "done", "timeout", "send_code", "loop_ms"]
    header += ["tcp_x_m", "tcp_y_m", "tcp_z_m", "tcp_qw", "tcp_qx", "tcp_qy", "tcp_qz"]
    header += ["object_x_m", "object_y_m", "object_z_m", "object_qw", "object_qx", "object_qy", "object_qz"]
    header += [f"sdk_obs_{i:02d}" for i in range(OBS_DIM)]
    header += [f"isaac_obs_{i:02d}" for i in range(OBS_DIM)]
    header += [f"obs_diff_{i:02d}" for i in range(OBS_DIM)]
    header += [f"policy_raw_action_j{i}" for i in range(1, ACT_DIM + 1)]
    header += [f"logged_raw_action_j{i}" for i in range(1, ACT_DIM + 1)]
    header += [f"raw_action_diff_j{i}" for i in range(1, ACT_DIM + 1)]
    header += [f"cmd_delta_j{i}_deg" for i in range(1, ACT_DIM + 1)]
    header += [f"logged_cmd_delta_j{i}_deg" for i in range(1, ACT_DIM + 1)]
    header += [f"cmd_delta_diff_j{i}_deg" for i in range(1, ACT_DIM + 1)]
    header += [f"q_before_j{i}_deg" for i in range(1, ACT_DIM + 1)]
    header += [f"target_j{i}_deg" for i in range(1, ACT_DIM + 1)]
    header += [f"q_after_j{i}_deg" for i in range(1, ACT_DIM + 1)]
    header += [f"actual_delta_j{i}_deg" for i in range(1, ACT_DIM + 1)]
    header += [f"track_err_j{i}_deg" for i in range(1, ACT_DIM + 1)]
    writer.writerow(header)


def _write_row(writer, item):
    logged_cmd = item["logged_cmd_delta_deg"]
    cmd_diff = None if logged_cmd is None else item["cmd_delta_deg"] - logged_cmd
    values = [
        item["csv_row"],
        item["episode"],
        item["step"],
        item["g_step"],
        int(item["done"]),
        int(item["timeout"]),
        item["send_code"],
        f"{item['loop_ms']:.3f}",
    ]
    values += [f"{v:.8f}" for v in item["tcp_pos_m"]]
    values += [f"{v:.8f}" for v in item["tcp_quat_wxyz"]]
    values += [f"{v:.8f}" for v in item["object_pos_m"]]
    values += [f"{v:.8f}" for v in item["object_quat_wxyz"]]
    values += [f"{v:.8f}" for v in item["sdk_obs"]]
    values += [f"{v:.8f}" for v in item["isaac_obs"]]
    values += [f"{v:.8f}" for v in item["sdk_obs"] - item["isaac_obs"]]
    values += [f"{v:.8f}" for v in item["policy_raw_action"]]
    values += [f"{v:.8f}" for v in item["logged_raw_action"]]
    values += [f"{v:.8f}" for v in item["policy_raw_action"] - item["logged_raw_action"]]
    values += [f"{v:.8f}" for v in item["cmd_delta_deg"]]
    values += ["" if logged_cmd is None else f"{v:.8f}" for v in (logged_cmd if logged_cmd is not None else np.zeros(ACT_DIM))]
    values += ["" if cmd_diff is None else f"{v:.8f}" for v in (cmd_diff if cmd_diff is not None else np.zeros(ACT_DIM))]
    values += [f"{v:.8f}" for v in item["q_before_deg"]]
    values += [f"{v:.8f}" for v in item["target_deg"]]
    values += [f"{v:.8f}" for v in item["q_after_deg"]]
    values += [f"{v:.8f}" for v in item["actual_delta_deg"]]
    values += [f"{v:.8f}" for v in item["track_err_deg"]]
    writer.writerow(values)


def main():
    args = parse_args()
    run_label = "sim2real" if args.real else "sim2sim"
    rows = _load_rows(args.isaac_csv, args.start_row, args.steps)
    out_path = _output_path(args.log, args.ckpt, args.isaac_csv, run_label)

    action_scale_rad = np.deg2rad(args.action_scale_deg).astype(np.float32)
    action_clip_rad = np.deg2rad(args.action_clip_deg).astype(np.float32)

    first_obs = rows[0][2]
    init_q_rad = np.clip(DEFAULT_JOINT_POS_RAD + first_obs[:ACT_DIM], JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH)
    first_object_rel_pose = first_obs[7:14].copy()

    print("\n========== SDK rollout from Isaac first obs ==========")
    print(f"[Input]  isaac_csv={args.isaac_csv}")
    print(f"[Input]  rows={len(rows)} start_row={args.start_row}")
    print(f"[Policy] ckpt={args.ckpt} action_source={args.action_source}")
    print(f"[Obs]    object_mode={args.object_mode}")
    print(f"[Arm]    ip={args.ip} run_label={run_label} dry_run={args.dry_run}")
    print("[Arm]    --real only changes log naming; set_simulation_robot() is not called.")
    print(f"[Init]   q0={format_deg(init_q_rad, 3)}")
    print(f"[Output] {out_path}")
    print("======================================================\n")

    policy = None
    if args.action_source == "policy":
        policy = V87RelativePoseJointStepPolicy(args.ckpt, device=args.device, obs_norm_clip=args.obs_norm_clip)

    arm = None
    last_action_clipped = first_obs[14:21].copy().astype(np.float32)
    object_pos_base_m = None
    object_quat_base = None

    try:
        arm = XArmAPI(args.ip, is_radian=True, check_joint_limit=False)
        arm.clean_error()
        arm.clean_warn()
        arm.motion_enable(enable=True)
        arm.set_state(0)

        if not args.no_move_init:
            print("[Arm] Move to Isaac first-observation joints.")
            enter_position_mode(arm)
            code = arm.set_servo_angle(
                angle=init_q_rad.tolist(),
                speed=np.deg2rad(INIT_SPEED_DEG),
                is_radian=True,
                wait=True,
            )
            if code != 0:
                raise RuntimeError(f"set_servo_angle init failed: {code}")
            print(f"[Arm] init reached: {format_deg(init_q_rad, 3)}")
        else:
            print("[Arm] Skip init move; start from current SDK joints.")

        enter_servo_mode(arm)

        object_pos_base_m = None
        object_quat_base = None
        if args.object_mode == "sdk":
            ee_pos0_m, ee_quat0, _ = get_tcp_pose_base(arm)
            object_pos_base_m, object_quat_base = _infer_object_base_pose_from_rel(
                first_object_rel_pose,
                ee_pos0_m,
                ee_quat0,
            )
            print(f"[Obs] inferred object_pos_base_m={format_np(object_pos_base_m, 6)}")
            print(f"[Obs] inferred object_quat_base={format_np(object_quat_base, 6)}")

        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            _write_header(writer)

            for rollout_step, (csv_row, csv_data, isaac_obs, logged_raw, logged_cmd) in enumerate(rows):
                loop_t0 = time.perf_counter()
                done = _boolish(csv_data.get("done"))
                timeout = _boolish(csv_data.get("timeout"))

                q_before = get_joint_positions(arm)
                ee_pos_base_m, ee_quat_base, _ = get_tcp_pose_base(arm)
                if args.object_mode == "sdk":
                    object_rel_pose = object_pose_relative_to_ee_np(
                        object_pos_base_m,
                        object_quat_base,
                        ee_pos_base_m,
                        ee_quat_base,
                        q_offset_wxyz=DEFAULT_Q_OFFSET_WXYZ,
                    )
                elif args.object_mode == "isaac":
                    object_rel_pose = isaac_obs[7:14]
                else:
                    object_rel_pose = first_object_rel_pose

                if object_pos_base_m is None or object_quat_base is None:
                    object_pos_log_m, object_quat_log = _infer_object_base_pose_from_rel(
                        object_rel_pose,
                        ee_pos_base_m,
                        ee_quat_base,
                    )
                else:
                    object_pos_log_m = object_pos_base_m
                    object_quat_log = object_quat_base

                sdk_obs = build_obs(q_before, object_rel_pose, last_action_clipped)

                if args.action_source == "logged":
                    raw_action = logged_raw.copy()
                else:
                    raw_action = policy.predict(sdk_obs)

                cmd_delta_rad_unclipped = raw_action.astype(np.float32) * action_scale_rad
                cmd_delta_rad = np.clip(cmd_delta_rad_unclipped, -action_clip_rad, action_clip_rad).astype(np.float32)
                cmd_delta_deg = np.rad2deg(cmd_delta_rad).astype(np.float32)
                target_rad = np.clip(q_before + cmd_delta_rad, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH).astype(np.float32)

                print(
                    f"[row {csv_row:05d} sdk_step {rollout_step:04d}] "
                    f"cmd_delta={format_np(cmd_delta_deg, 4)} deg target={format_deg(target_rad, 3)}"
                )
                print(f"  obs_diff_max={float(np.max(np.abs(sdk_obs - isaac_obs))):.6g} raw_diff={format_np(raw_action - logged_raw, 6)}")

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
                        print(f"[Warn] set_servo_angle_j failed: {send_code}; recover mode 1")
                        recover_servo_mode(arm)

                if args.sample_dt > 0.0:
                    time.sleep(args.sample_dt)

                q_after = get_joint_positions(arm)
                tcp_pos_m, tcp_quat_wxyz, _ = get_tcp_pose_base(arm)
                actual_delta_rad = (q_after - q_before).astype(np.float32)
                actual_delta_deg = np.rad2deg(actual_delta_rad).astype(np.float32)
                track_err_deg = np.rad2deg(q_after - target_rad).astype(np.float32)
                last_action_clipped = np.clip(raw_action, -1.0, 1.0).astype(np.float32)

                _write_row(
                    writer,
                    {
                        "csv_row": csv_row,
                        "episode": csv_data.get("episode", ""),
                        "step": csv_data.get("step", ""),
                        "g_step": csv_data.get("g_step", ""),
                        "done": done,
                        "timeout": timeout,
                        "send_code": send_code,
                        "loop_ms": (time.perf_counter() - loop_t0) * 1000.0,
                        "tcp_pos_m": tcp_pos_m,
                        "tcp_quat_wxyz": tcp_quat_wxyz,
                        "object_pos_m": object_pos_log_m,
                        "object_quat_wxyz": object_quat_log,
                        "sdk_obs": sdk_obs,
                        "isaac_obs": isaac_obs,
                        "policy_raw_action": raw_action,
                        "logged_raw_action": logged_raw,
                        "cmd_delta_deg": cmd_delta_deg,
                        "logged_cmd_delta_deg": logged_cmd,
                        "q_before_deg": np.rad2deg(q_before).astype(np.float32),
                        "target_deg": np.rad2deg(target_rad).astype(np.float32),
                        "q_after_deg": np.rad2deg(q_after).astype(np.float32),
                        "actual_delta_deg": actual_delta_deg,
                        "track_err_deg": track_err_deg,
                    },
                )
                f.flush()

                if done and not args.ignore_done:
                    print(f"[Done] CSV row {csv_row} has done=1; stop. Use --ignore_done to continue.")
                    break

    finally:
        if arm is not None:
            try:
                arm.set_mode(0)
                arm.set_state(0)
                arm.disconnect()
                print("[Arm] mode 0 and disconnected")
            except Exception as exc:
                print(f"[Warn] cleanup failed: {type(exc).__name__}: {exc}")


if __name__ == "__main__":
    main()
