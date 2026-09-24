#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""从 Isaac play CSV 回放 obs，并通过 xArm SDK 发送策略动作。"""

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
    INIT_JOINTS_DEG,
    INIT_SPEED_DEG,
    JOINT_LIMITS_HIGH,
    JOINT_LIMITS_LOW,
    OBS_DIM,
    TRAIN_ACTION_CLIP_DEG,
    TRAIN_ACTION_SCALE_DEG,
    V87RelativePoseJointStepPolicy,
    XArmAPI,
    _wait_for_step,
    enter_position_mode,
    enter_servo_mode,
    format_deg,
    format_np,
    get_joint_positions,
    recover_servo_mode,
)


LOG_NAME = "xarm7_pick_pose"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Replay mdp_obs from train_pick_pose --play CSV into policy and send actions to xArm."
    )
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--obs_csv", type=str, required=True, help="CSV produced by train_pick_pose.py --play with mdp_obs columns.")
    parser.add_argument("--ip", type=str, default="192.168.73.229")
    parser.add_argument("--real", action="store_true", help="Tag outputs as sim2real data only; does not switch SDK robot mode.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--obs_norm_clip", type=float, default=0.0, help="0 disables extra normalized-obs clamp, matching Isaac inference.")
    parser.add_argument("--action_scale_deg", type=float, default=TRAIN_ACTION_SCALE_DEG)
    parser.add_argument("--action_clip_deg", type=float, default=TRAIN_ACTION_CLIP_DEG)
    parser.add_argument("--start_row", type=int, default=0, help="First CSV data row to replay, 0-based after header.")
    parser.add_argument("--steps", type=int, default=-1, help="Number of rows to replay. -1 means until CSV end or done row.")
    parser.add_argument("--sample_dt", type=float, default=CONTROL_DT)
    parser.add_argument("--step_mode", action="store_true", help="Wait for SPACE before sending each row action.")
    parser.add_argument("--dry_run", action="store_true", help="Compute and log actions without sending to xArm.")
    parser.add_argument("--no_move_init", action="store_true")
    parser.add_argument("--ignore_done", action="store_true", help="Continue after rows whose done column is true.")
    parser.add_argument("--log", type=str, default=None, help="Output CSV path. Default: test_log/xarm7_pick_pose-<policy_time>/obs_replay_deploy_<sim2real|sim2sim>_<policy_time>.csv")
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
    return make_policy_output_path(_PROJECT_DIR / "test_log", LOG_NAME, policy_time, f"obs_replay_deploy_{run_label}")


def _obs_columns(fieldnames: list[str]) -> list[str]:
    cols = [f"mdp_obs_{i:02d}" for i in range(OBS_DIM)]
    missing = [c for c in cols if c not in fieldnames]
    if missing:
        raise RuntimeError(
            "obs_csv 缺少 mdp_obs 字段；请先用更新后的 train_pick_pose.py --play 生成 CSV。"
            f" 缺少: {missing[:5]}{'...' if len(missing) > 5 else ''}"
        )
    old_extra = f"mdp_obs_{OBS_DIM:02d}"
    if old_extra in fieldnames:
        raise RuntimeError(
            f"obs_csv 看起来仍是旧 28D vel 观测日志，包含 {old_extra}；"
            "请重新用 21D no-vel 配置生成 play CSV。"
        )
    return cols


def _raw_action_columns(fieldnames: list[str]) -> list[str]:
    cols = [f"mdp_raw_action_j{i}" for i in range(1, ACT_DIM + 1)]
    return cols if all(c in fieldnames for c in cols) else []


def _load_rows(path: str, start_row: int, steps: int):
    with Path(path).expanduser().open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV 没有 header: {path}")
        obs_cols = _obs_columns(reader.fieldnames)
        raw_cols = _raw_action_columns(reader.fieldnames)
        rows = []
        for i, row in enumerate(reader):
            if i < start_row:
                continue
            if steps >= 0 and len(rows) >= steps:
                break
            obs = np.asarray([float(row[c]) for c in obs_cols], dtype=np.float32)
            logged_raw = None
            if raw_cols:
                logged_raw = np.asarray([float(row[c]) for c in raw_cols], dtype=np.float32)
            rows.append((i, row, obs, logged_raw))
    if not rows:
        raise RuntimeError(f"没有可回放的 CSV 行: {path}")
    return rows


def _write_header(writer):
    header = [
        "csv_row", "episode", "step", "g_step", "done", "timeout", "send_code", "loop_ms",
    ]
    header += [f"mdp_obs_{i:02d}" for i in range(OBS_DIM)]
    header += [f"policy_raw_action_j{i}" for i in range(1, ACT_DIM + 1)]
    header += [f"logged_raw_action_j{i}" for i in range(1, ACT_DIM + 1)]
    header += [f"raw_action_diff_j{i}" for i in range(1, ACT_DIM + 1)]
    header += [f"cmd_delta_j{i}_deg" for i in range(1, ACT_DIM + 1)]
    header += [f"q_before_j{i}_deg" for i in range(1, ACT_DIM + 1)]
    header += [f"target_j{i}_deg" for i in range(1, ACT_DIM + 1)]
    header += [f"q_after_j{i}_deg" for i in range(1, ACT_DIM + 1)]
    header += [f"actual_delta_j{i}_deg" for i in range(1, ACT_DIM + 1)]
    header += [f"track_err_j{i}_deg" for i in range(1, ACT_DIM + 1)]
    writer.writerow(header)


def _write_row(writer, item):
    logged_raw = item["logged_raw_action"]
    raw_diff = item["raw_action_diff"]
    values = [
        item["csv_row"], item["episode"], item["step"], item["g_step"],
        int(item["done"]), int(item["timeout"]), item["send_code"], f"{item['loop_ms']:.3f}",
    ]
    values += [f"{v:.8f}" for v in item["obs"]]
    values += [f"{v:.8f}" for v in item["policy_raw_action"]]
    values += ["" if logged_raw is None else f"{v:.8f}" for v in (logged_raw if logged_raw is not None else np.zeros(ACT_DIM))]
    values += ["" if raw_diff is None else f"{v:.8f}" for v in (raw_diff if raw_diff is not None else np.zeros(ACT_DIM))]
    values += [f"{v:.8f}" for v in item["cmd_delta_deg"]]
    values += [f"{v:.8f}" for v in item["q_before_deg"]]
    values += [f"{v:.8f}" for v in item["target_deg"]]
    values += [f"{v:.8f}" for v in item["q_after_deg"]]
    values += [f"{v:.8f}" for v in item["actual_delta_deg"]]
    values += [f"{v:.8f}" for v in item["track_err_deg"]]
    writer.writerow(values)


def main():
    args = parse_args()
    run_label = "sim2real" if args.real else "sim2sim"
    rows = _load_rows(args.obs_csv, args.start_row, args.steps)
    out_path = _output_path(args.log, args.ckpt, args.obs_csv, run_label)
    action_scale_rad = np.deg2rad(args.action_scale_deg).astype(np.float32)
    action_clip_rad = np.deg2rad(args.action_clip_deg).astype(np.float32)

    print("\n========== deploy pose policy from recorded MDP obs ==========")
    print(f"[Input]  obs_csv={args.obs_csv}")
    print(f"[Input]  rows={len(rows)} start_row={args.start_row}")
    print(f"[Policy] ckpt={args.ckpt}")
    print(f"[Arm]    ip={args.ip} run_label={run_label} dry_run={args.dry_run}")
    print("[Arm]    --real only changes log naming; set_simulation_robot() is not called.")
    print(f"[Action] scale={args.action_scale_deg:.3f}deg clip=+/-{args.action_clip_deg:.3f}deg")
    print(f"[Output] {out_path}")
    print("============================================================\n")

    policy = V87RelativePoseJointStepPolicy(args.ckpt, device=args.device, obs_norm_clip=args.obs_norm_clip)
    arm = None

    try:
        arm = XArmAPI(args.ip, is_radian=True, check_joint_limit=False)
        arm.clean_error()
        arm.clean_warn()
        arm.motion_enable(enable=True)
        arm.set_state(0)

        if not args.no_move_init:
            print("[Arm] Move to training init joints first.")
            enter_position_mode(arm)
            code = arm.set_servo_angle(
                angle=np.deg2rad(INIT_JOINTS_DEG).tolist(),
                speed=np.deg2rad(INIT_SPEED_DEG),
                is_radian=True,
                wait=True,
            )
            if code != 0:
                raise RuntimeError(f"set_servo_angle init failed: {code}")
            print(f"[Arm] init joints reached: {INIT_JOINTS_DEG} deg")
        else:
            print("[Arm] Skip init move; start from current joints.")

        enter_servo_mode(arm)

        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            _write_header(writer)

            for replay_step, (csv_row, csv_data, obs, logged_raw) in enumerate(rows):
                loop_t0 = time.perf_counter()
                done = _boolish(csv_data.get("done"))
                timeout = _boolish(csv_data.get("timeout"))

                raw_action = policy.predict(obs)
                raw_diff = None if logged_raw is None else raw_action - logged_raw
                cmd_delta_rad_unclipped = raw_action.astype(np.float32) * action_scale_rad
                cmd_delta_rad = np.clip(cmd_delta_rad_unclipped, -action_clip_rad, action_clip_rad).astype(np.float32)

                q_before = get_joint_positions(arm)
                target_rad = np.clip(q_before + cmd_delta_rad, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH).astype(np.float32)
                cmd_delta_deg = np.rad2deg(cmd_delta_rad).astype(np.float32)

                print(
                    f"[row {csv_row:05d} step {replay_step:04d}] "
                    f"cmd_delta={format_np(cmd_delta_deg, 4)} deg "
                    f"target={format_deg(target_rad, 3)} done={done}"
                )
                if logged_raw is not None:
                    print(f"  raw_diff_vs_log={format_np(raw_diff, 6)}")

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
                actual_delta_deg = np.rad2deg(q_after - q_before).astype(np.float32)
                track_err_deg = np.rad2deg(q_after - target_rad).astype(np.float32)

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
                        "obs": obs,
                        "policy_raw_action": raw_action,
                        "logged_raw_action": logged_raw,
                        "raw_action_diff": raw_diff,
                        "cmd_delta_deg": cmd_delta_deg,
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
