#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""测试 xArm SDK 执行动作时的关节跟踪误差。"""

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from deploy_policy_pose import (
    ACT_DIM,
    CONTROL_DT,
    DEFAULT_Q_OFFSET_WXYZ,
    INIT_JOINTS_DEG,
    INIT_SPEED_DEG,
    JOINT_LIMITS_HIGH,
    JOINT_LIMITS_LOW,
    OBS_DIM,
    TRAIN_ACTION_CLIP_DEG,
    TRAIN_ACTION_SCALE_DEG,
    V87RelativePoseJointStepPolicy,
    XArmAPI,
    build_obs,
    enter_position_mode,
    enter_servo_mode,
    format_deg,
    get_joint_positions,
    get_tcp_pose_base,
    object_pose_relative_to_ee_np,
    quat_normalize,
    recover_servo_mode,
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Plot commanded policy action vs measured xArm joint forward-difference action."
    )
    parser.add_argument("--ckpt", type=str, required=True)
    parser.add_argument("--ip", type=str, default="192.168.73.229")
    parser.add_argument("--real", action="store_true", help="Tag outputs as sim2real data only; does not switch SDK robot mode.")
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--obs_csv", type=str, default=None, help="Use mdp_obs rows from a train_pick_pose --play CSV instead of live robot/object observations.")
    parser.add_argument("--episode", type=int, default=1, help="Episode id to replay from --obs_csv. Default: 1.")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--obs_norm_clip", type=float, default=0.0, help="normalized obs clamp; 0 disables it to match Isaac inference")
    parser.add_argument("--sample_dt", type=float, default=CONTROL_DT)
    parser.add_argument("--action_scale_deg", type=float, default=TRAIN_ACTION_SCALE_DEG)
    parser.add_argument("--action_clip_deg", type=float, default=TRAIN_ACTION_CLIP_DEG)
    parser.add_argument("--dry_run", action="store_true", help="Do not send target joints; useful for checking policy output only.")
    parser.add_argument("--no_move_init", action="store_true")
    parser.add_argument("--log_dir", type=str, default="test_log")
    parser.add_argument("--tag", type=str, default="action_tracking_pose")
    parser.add_argument("--continue_on_pose_error", action="store_true")

    pose_group = parser.add_mutually_exclusive_group(required=False)
    pose_group.add_argument(
        "--object_rel_pose",
        type=float,
        nargs=7,
        metavar=("X", "Y", "Z", "QW", "QX", "QY", "QZ"),
        help="Object pose relative to TCP/EE, position in m, quaternion wxyz.",
    )
    pose_group.add_argument(
        "--object_pos_base",
        type=float,
        nargs=3,
        metavar=("X", "Y", "Z"),
        help="Object position in xArm base frame, unit m. Requires --object_quat_base.",
    )
    parser.add_argument(
        "--object_quat_base",
        type=float,
        nargs=4,
        metavar=("W", "X", "Y", "Z"),
        help="Object quaternion in xArm base frame, wxyz.",
    )
    parser.add_argument(
        "--q_offset",
        type=float,
        nargs=4,
        default=DEFAULT_Q_OFFSET_WXYZ.tolist(),
        metavar=("W", "X", "Y", "Z"),
    )
    return parser.parse_args()


def make_output_paths(log_dir: str, tag: str):
    root = Path(log_dir)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    base = root / f"{tag}_{stamp}"
    return base.with_suffix(".csv"), Path(f"{base}_action_norm.png"), Path(f"{base}_delta_deg.png")


def load_recorded_obs(csv_path: str, episode: int, max_steps: int):
    path = Path(csv_path).expanduser()
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"CSV has no header: {path}")
        obs_cols = [f"mdp_obs_{i:02d}" for i in range(OBS_DIM)]
        missing = [c for c in obs_cols if c not in reader.fieldnames]
        if missing:
            raise RuntimeError(f"CSV missing mdp_obs columns: {missing[:5]}")
        old_extra = f"mdp_obs_{OBS_DIM:02d}"
        if old_extra in reader.fieldnames:
            raise RuntimeError(
                f"CSV looks like an old 28D velocity-observation log because it contains {old_extra}; "
                "regenerate it with the 21D no-velocity config."
            )

        for csv_row, row in enumerate(reader):
            try:
                row_ep = int(row.get("episode", -1))
            except Exception:
                row_ep = -1
            if row_ep != episode:
                continue

            obs = np.asarray([float(row[c]) for c in obs_cols], dtype=np.float32)
            rows.append((csv_row, row, obs))
            if max_steps >= 0 and len(rows) >= max_steps:
                break

    if not rows:
        raise RuntimeError(f"No rows found for episode={episode} in {path}")
    return rows


def init_pose_args(args):
    if args.object_rel_pose is None and args.object_pos_base is None:
        raise ValueError("Live-observation mode requires --object_rel_pose or --object_pos_base. Use --obs_csv to replay recorded MDP obs.")
    if args.object_pos_base is not None and args.object_quat_base is None:
        raise ValueError("--object_pos_base requires --object_quat_base")

    q_offset = quat_normalize(np.asarray(args.q_offset, dtype=np.float32))
    if args.object_rel_pose is not None:
        object_rel_pose_fixed = np.asarray(args.object_rel_pose, dtype=np.float32)
        object_rel_pose_fixed[3:7] = quat_normalize(object_rel_pose_fixed[3:7])
        object_pos_base_m = None
        object_quat_base = None
        pose_mode = "object_rel_pose_direct"
    else:
        object_rel_pose_fixed = None
        object_pos_base_m = np.asarray(args.object_pos_base, dtype=np.float32)
        object_quat_base = quat_normalize(np.asarray(args.object_quat_base, dtype=np.float32))
        pose_mode = "object_pose_base_to_rel"
    return pose_mode, q_offset, object_rel_pose_fixed, object_pos_base_m, object_quat_base


def write_header(writer):
    header = ["step", "t_s", "loop_ms", "send_code"]
    header += [f"q_before_j{i}_deg" for i in range(1, ACT_DIM + 1)]
    header += [f"q_after_j{i}_deg" for i in range(1, ACT_DIM + 1)]
    header += [f"target_j{i}_deg" for i in range(1, ACT_DIM + 1)]
    header += [f"raw_action_j{i}" for i in range(1, ACT_DIM + 1)]
    header += [f"cmd_action_norm_j{i}" for i in range(1, ACT_DIM + 1)]
    header += [f"actual_action_norm_j{i}" for i in range(1, ACT_DIM + 1)]
    header += [f"cmd_delta_j{i}_deg" for i in range(1, ACT_DIM + 1)]
    header += [f"actual_delta_j{i}_deg" for i in range(1, ACT_DIM + 1)]
    header += [f"track_err_j{i}_deg" for i in range(1, ACT_DIM + 1)]
    writer.writerow(header)


def write_row(writer, row):
    values = [row["step"], f"{row['t_s']:.6f}", f"{row['loop_ms']:.3f}", row["send_code"]]
    for key in (
        "q_before_deg",
        "q_after_deg",
        "target_deg",
        "raw_action",
        "cmd_action_norm",
        "actual_action_norm",
        "cmd_delta_deg",
        "actual_delta_deg",
        "track_err_deg",
    ):
        values.extend(f"{v:.8f}" for v in row[key])
    writer.writerow(values)


def plot_results(rows, action_png: Path, delta_png: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    steps = np.asarray([r["step"] for r in rows], dtype=np.int32)
    cmd_action = np.stack([r["cmd_action_norm"] for r in rows])
    actual_action = np.stack([r["actual_action_norm"] for r in rows])
    cmd_delta = np.stack([r["cmd_delta_deg"] for r in rows])
    actual_delta = np.stack([r["actual_delta_deg"] for r in rows])

    def _plot_pair(y_cmd, y_actual, ylabel, path):
        fig, axes = plt.subplots(ACT_DIM, 1, figsize=(13, 14), sharex=True)
        for j, ax in enumerate(axes):
            ax.plot(steps, y_cmd[:, j], label="cmd", lw=1.4)
            ax.plot(steps, y_actual[:, j], label="actual", lw=1.1)
            ax.axhline(0.0, color="0.7", lw=0.7)
            ax.set_ylabel(f"j{j + 1}\n{ylabel}")
            ax.grid(True, alpha=0.3)
            if j == 0:
                ax.legend(loc="upper right")
        axes[-1].set_xlabel("step")
        fig.tight_layout()
        fig.savefig(path, dpi=160)
        plt.close(fig)

    _plot_pair(cmd_action, actual_action, "action", action_png)
    _plot_pair(cmd_delta, actual_delta, "deg", delta_png)


def main():
    args = parse_args()
    run_label = "sim2real" if args.real else "sim2sim"
    recorded_obs_rows = None
    if args.obs_csv is not None:
        recorded_obs_rows = load_recorded_obs(args.obs_csv, args.episode, args.steps)
        pose_mode = f"recorded_mdp_obs_ep{args.episode}"
        q_offset = object_rel_pose_fixed = object_pos_base_m = object_quat_base = None
    else:
        pose_mode, q_offset, object_rel_pose_fixed, object_pos_base_m, object_quat_base = init_pose_args(args)
    csv_path, action_png, delta_png = make_output_paths(args.log_dir, f"{args.tag}_{run_label}")

    action_scale_rad = np.deg2rad(args.action_scale_deg).astype(np.float32)
    action_clip_rad = np.deg2rad(args.action_clip_deg).astype(np.float32)

    print("\n========== pose policy action tracking test ==========")
    print(f"[Config] ckpt={args.ckpt}")
    print(f"[Config] ip={args.ip} run_label={run_label} dry_run={args.dry_run}")
    print("[Config] --real only changes log naming; set_simulation_robot() is not called.")
    if recorded_obs_rows is not None:
        print(f"[Config] obs_csv={args.obs_csv} episode={args.episode} rows={len(recorded_obs_rows)}")
    print(f"[Config] steps={args.steps} sample_dt={args.sample_dt:.4f}s pose_mode={pose_mode}")
    print(f"[Config] action_scale={args.action_scale_deg:.3f}deg action_clip=+/-{args.action_clip_deg:.3f}deg")
    print(f"[Config] obs_norm_clip={args.obs_norm_clip:.3f} (0=disabled, match Isaac inference)")
    print(f"[Output] csv={csv_path}")
    print(f"[Output] action_png={action_png}")
    print(f"[Output] delta_png={delta_png}")
    print("====================================================\n")

    policy = V87RelativePoseJointStepPolicy(args.ckpt, device=args.device, obs_norm_clip=args.obs_norm_clip)
    arm = None
    rows = []

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

        last_action_clipped = np.zeros(ACT_DIM, dtype=np.float32)
        t_start = time.perf_counter()

        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            write_header(writer)

            loop_count = len(recorded_obs_rows) if recorded_obs_rows is not None else args.steps
            for step in range(loop_count):
                loop_t0 = time.perf_counter()

                try:
                    q_before = get_joint_positions(arm)
                    if recorded_obs_rows is not None:
                        csv_row, csv_data, obs = recorded_obs_rows[step]
                    else:
                        ee_pos_base_m, ee_quat_base, _ = get_tcp_pose_base(arm)

                        if object_rel_pose_fixed is None:
                            object_rel_pose = object_pose_relative_to_ee_np(
                                object_pos_base_m,
                                object_quat_base,
                                ee_pos_base_m,
                                ee_quat_base,
                                q_offset_wxyz=q_offset,
                            )
                        else:
                            object_rel_pose = object_rel_pose_fixed.copy()

                        obs = build_obs(q_before, object_rel_pose, last_action_clipped)
                except Exception as exc:
                    print(f"[Warn] read obs/joint/TCP failed: {type(exc).__name__}: {exc}")
                    if args.continue_on_pose_error:
                        recover_servo_mode(arm)
                        continue
                    raise
                raw_action = policy.predict(obs)
                policy_action_clipped = np.clip(raw_action, -1.0, 1.0).astype(np.float32)
                cmd_delta_rad_unclipped = raw_action.astype(np.float32) * action_scale_rad
                cmd_delta_rad = np.clip(cmd_delta_rad_unclipped, -action_clip_rad, action_clip_rad).astype(np.float32)
                target_rad = np.clip(q_before + cmd_delta_rad, JOINT_LIMITS_LOW, JOINT_LIMITS_HIGH).astype(np.float32)

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
                actual_delta_rad = (q_after - q_before).astype(np.float32)
                cmd_delta_deg = np.rad2deg(cmd_delta_rad).astype(np.float32)
                cmd_action_norm = (cmd_delta_deg / max(args.action_scale_deg, 1.0e-6)).astype(np.float32)
                actual_delta_deg = np.rad2deg(actual_delta_rad).astype(np.float32)
                actual_action_norm = (actual_delta_deg / max(args.action_scale_deg, 1.0e-6)).astype(np.float32)
                track_err_deg = np.rad2deg(q_after - target_rad).astype(np.float32)

                row = {
                    "step": step,
                    "t_s": time.perf_counter() - t_start,
                    "loop_ms": (time.perf_counter() - loop_t0) * 1000.0,
                    "send_code": send_code,
                    "q_before_deg": np.rad2deg(q_before).astype(np.float32),
                    "q_after_deg": np.rad2deg(q_after).astype(np.float32),
                    "target_deg": np.rad2deg(target_rad).astype(np.float32),
                    "raw_action": raw_action.astype(np.float32),
                    "cmd_action_norm": cmd_action_norm,
                    "actual_action_norm": actual_action_norm,
                    "cmd_delta_deg": cmd_delta_deg,
                    "actual_delta_deg": actual_delta_deg,
                    "track_err_deg": track_err_deg,
                }
                rows.append(row)
                write_row(writer, row)
                f.flush()

                last_action_clipped = policy_action_clipped.copy()

                if step % 10 == 0:
                    max_err = float(np.max(np.abs(track_err_deg)))
                    prefix = f"[step {step:4d}]"
                    if recorded_obs_rows is not None:
                        prefix = f"[ep {args.episode} csv_row {csv_row:05d} step {step:4d}]"
                    print(
                        f"{prefix} "
                        f"cmd_delta={np.round(cmd_delta_deg, 3).tolist()} deg | "
                        f"actual_delta={np.round(actual_delta_deg, 3).tolist()} deg | "
                        f"max_track_err={max_err:.3f} deg | code={send_code}"
                    )

        if rows:
            try:
                plot_results(rows, action_png, delta_png)
                print(f"[Plot] saved: {action_png}")
                print(f"[Plot] saved: {delta_png}")
            except Exception as exc:
                print(f"[Warn] plot failed, CSV is still saved: {type(exc).__name__}: {exc}")

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
