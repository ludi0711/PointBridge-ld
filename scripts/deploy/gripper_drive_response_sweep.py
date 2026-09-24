#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Sweep CRT gripper actuator gains and measure step response."""

from __future__ import annotations

import argparse
import csv
import math
import sys
from datetime import datetime
from pathlib import Path

from isaaclab.app import AppLauncher

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

DEFAULT_FIXED_SCENE_USD = (
    REPO_ROOT
    / "assets"
    / "crt_ctm2f110_gripper_visualization"
    / "scene_config_files"
    / "visualized_lift_v20_scene_articulation_fixed.usd"
)
DEFAULT_RAW_SCENE_USD = (
    REPO_ROOT
    / "assets"
    / "crt_ctm2f110_gripper_visualization"
    / "scene_config_files"
    / "visualized_lift_v20_scene.usd"
)
DEFAULT_WRAPPER_USD = Path("/tmp/gx_va_crt_gripper_response_robot.usd")
GRIPPER_OPEN_DEG = 57.0
GRIPPER_CLOSED_DEG = 0.0
GRIPPER_JOINT_TARGET_MULTIPLIERS = {
    "leftfinger_joint": 1.0,
    "rightfinger_joint": 1.0,
    "left_kckle_joint": 1.0,
    "right_kckle_joint": 1.0,
    "leftinn_joint": -1.0,
    "rightinn_joint": -1.0,
}
PRIMARY_JOINT_NAMES = ["leftfinger_joint", "rightfinger_joint"]
CONTACT_REPORT_BODY_PATHS = (
    "/Robot/link7/tool/assembly/links/left_pad",
    "/Robot/link7/tool/assembly/links/right_pad",
)


def default_source_usd() -> Path:
    return DEFAULT_FIXED_SCENE_USD if DEFAULT_FIXED_SCENE_USD.exists() else DEFAULT_RAW_SCENE_USD


parser = argparse.ArgumentParser(description="CRT gripper actuator step-response sweep.")
parser.add_argument("--source-usd", type=Path, default=default_source_usd())
parser.add_argument("--source-robot-prim", default="/World/XArm7WithCRTGripper")
parser.add_argument("--wrapper-usd", type=Path, default=DEFAULT_WRAPPER_USD)
parser.add_argument("--out-dir", type=Path, default=None)
parser.add_argument("--settle-time", type=float, default=0.30)
parser.add_argument("--hold-time", type=float, default=1.20)
parser.add_argument("--velocity-limit", type=float, default=10.0)
parser.add_argument("--step-deg", type=float, default=10.0, help="Native joint step size in display degrees from the current settled pose.")
parser.add_argument("--joint-units", choices=("rad", "deg"), default="deg", help="Interpret gripper joint tensor values/targets as radians or native degrees.")
parser.add_argument("--primary-only", action="store_true", help="Command only left/right primary finger joints during the response test.")
parser.add_argument(
    "--candidates",
    nargs="*",
    default=[
        "260,90,40",
        "400,120,40",
        "600,150,40",
        "800,180,40",
        "1000,220,40",
        "600,180,60",
        "800,240,60",
    ],
    help="Candidate triples stiffness,damping,effort.",
)
parser.add_argument("--plot", action="store_true", default=True)
parser.add_argument("--no-plot", dest="plot", action="store_false")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch
from pxr import PhysxSchema, Sdf, Usd, UsdGeom

from isaaclab.envs import ManagerBasedRLEnv
from configs.xarm7_pick_pose_crt_write_env_cfg import PickPoseCRTWriteEnvCfg_PLAY


def _write_robot_wrapper_usd(source_usd: Path, source_prim: str, wrapper_usd: Path) -> Path:
    source_usd = source_usd.expanduser().resolve()
    wrapper_usd = wrapper_usd.expanduser().resolve()
    if not source_usd.exists():
        raise FileNotFoundError(source_usd)
    wrapper_usd.parent.mkdir(parents=True, exist_ok=True)
    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, Sdf.Path("/Robot")).GetPrim()
    stage.SetDefaultPrim(root)
    if not root.GetReferences().AddReference(str(source_usd), Sdf.Path(source_prim)):
        raise RuntimeError(f"Failed to add robot reference: {source_usd}:{source_prim}")
    for body_path in CONTACT_REPORT_BODY_PATHS:
        prim = stage.GetPrimAtPath(body_path)
        if prim.IsValid():
            api = PhysxSchema.PhysxContactReportAPI.Apply(prim)
            api.CreateThresholdAttr().Set(0.0)
    stage.GetRootLayer().Export(str(wrapper_usd))
    print(f"[OK] Robot wrapper: {wrapper_usd}", flush=True)
    return wrapper_usd


def _parse_candidates(values: list[str]) -> list[tuple[float, float, float]]:
    out = []
    for value in values:
        parts = [float(v.strip()) for v in value.split(",")]
        if len(parts) != 3:
            raise ValueError(f"Candidate must be stiffness,damping,effort: {value}")
        out.append((parts[0], parts[1], parts[2]))
    return out


def _make_env_cfg(robot_usd: Path, stiffness: float, damping: float, effort: float, velocity_limit: float):
    env_cfg = PickPoseCRTWriteEnvCfg_PLAY()
    env_cfg.scene.num_envs = 1
    env_cfg.scene.ground = None
    env_cfg.scene.robot.spawn.usd_path = str(robot_usd.expanduser().resolve())
    # Headless local kit lacks the optional MDL material extension; the drive test does not need scene visuals.
    for scene_attr in ("robot_support_box", "workpiece_table"):
        scene_item = getattr(env_cfg.scene, scene_attr, None)
        if scene_item is not None and getattr(scene_item, "spawn", None) is not None:
            try:
                scene_item.spawn.visual_material = None
            except Exception:
                pass
    # This is a free-space actuator response test. Move/disable scene contacts so table/object contact
    # does not masquerade as poor PD tracking.
    for scene_attr in ("workpiece_table", "object"):
        scene_item = getattr(env_cfg.scene, scene_attr, None)
        if scene_item is None:
            continue
        try:
            scene_item.init_state.pos = [10.0, 10.0, -10.0]
        except Exception:
            pass
        try:
            scene_item.spawn.collision_props.collision_enabled = False
        except Exception:
            pass
    # Keep unrelated managers quiet and deterministic.
    try:
        env_cfg.events.reset_table_and_object = None
        env_cfg.events.randomize_table_color = None
        env_cfg.events.randomize_light_intensity = None
    except Exception:
        pass
    for name in ("crt_gripper_primary", "crt_gripper_mimic"):
        actuator = env_cfg.scene.robot.actuators[name]
        actuator.stiffness = float(stiffness)
        actuator.damping = float(damping)
        actuator.effort_limit_sim = float(effort)
        actuator.velocity_limit_sim = float(velocity_limit)
    return env_cfg


def _display_to_native(angle_deg: float) -> float:
    return math.radians(float(angle_deg)) if args_cli.joint_units == "rad" else float(angle_deg)


def _native_to_display(values: torch.Tensor) -> torch.Tensor:
    return torch.rad2deg(values) if args_cli.joint_units == "rad" else values


def _scalar_target(angle_deg: float, multipliers: torch.Tensor, device) -> torch.Tensor:
    scalar = torch.tensor([[_display_to_native(float(angle_deg))]], device=device, dtype=torch.float32)
    return scalar * multipliers.view(1, -1)


def _set_gripper_target(robot, joint_ids: list[int], target: torch.Tensor):
    zero_vel = torch.zeros_like(target)
    try:
        robot.set_joint_position_target(target, joint_ids=joint_ids)
    except TypeError:
        full_target = robot.data.joint_pos.clone()
        full_target[:, joint_ids] = target
        robot.set_joint_position_target(full_target)
    try:
        robot.set_joint_velocity_target(zero_vel, joint_ids=joint_ids)
    except TypeError:
        full_vel = torch.zeros_like(robot.data.joint_vel)
        full_vel[:, joint_ids] = 0.0
        robot.set_joint_velocity_target(full_vel)


def _sim_step(env: ManagerBasedRLEnv, dt: float):
    env.scene.write_data_to_sim()
    env.sim.step(render=False)
    env.scene.update(dt)


def _actual_scalar_deg(robot, joint_ids: list[int], multipliers: torch.Tensor, primary_cols: list[int]) -> tuple[float, float, float]:
    values = robot.data.joint_pos[0, joint_ids] / multipliers
    values_deg = _native_to_display(values).detach().cpu().numpy()
    left = float(values_deg[primary_cols[0]])
    right = float(values_deg[primary_cols[1]])
    mean = float(np.mean(values_deg[primary_cols]))
    return left, right, mean


def _metric_summary(t: np.ndarray, y: np.ndarray, start_value: float, target_value: float) -> dict[str, float]:
    target_delta = float(target_value - start_value)
    direction = 1.0 if target_delta >= 0.0 else -1.0
    step_abs = max(abs(target_delta), 1.0e-9)
    response = (y - float(start_value)) * direction
    final_value = float(y[-1])
    peak_response = float(np.max(response))
    overshoot_pct = max(0.0, (peak_response - step_abs) / step_abs * 100.0)

    def first_cross(level: float) -> float:
        idx = np.nonzero(response >= level * step_abs)[0]
        return float(t[idx[0]]) if idx.size else float("nan")

    t10 = first_cross(0.10)
    t90 = first_cross(0.90)
    rise_time = t90 - t10 if math.isfinite(t10) and math.isfinite(t90) else float("nan")
    band = max(0.02 * step_abs, 0.05)
    err = np.abs(y - target_value)
    settling_time = float("nan")
    for idx in range(len(t)):
        if np.all(err[idx:] <= band):
            settling_time = float(t[idx])
            break
    return {
        "final_error_deg": float(target_value - final_value),
        "overshoot_pct": float(overshoot_pct),
        "rise_time_10_90_s": float(rise_time),
        "settling_time_2pct_s": float(settling_time),
        "max_abs_error_deg": float(np.max(np.abs(target_value - y))),
    }


def _write_csv(rows: list[dict], path: Path):
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def _plot(raw_rows: list[dict], out_dir: Path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    groups = {}
    for row in raw_rows:
        groups.setdefault((row["candidate"], row["direction"]), []).append(row)
    for direction in ("negative_step", "positive_step"):
        fig, ax = plt.subplots(figsize=(12, 7), constrained_layout=True)
        for (candidate, group_direction), rows in groups.items():
            if group_direction != direction:
                continue
            t = np.asarray([r["time_s"] for r in rows], dtype=float)
            actual = np.asarray([r["actual_mean_deg"] for r in rows], dtype=float)
            target = np.asarray([r["target_deg"] for r in rows], dtype=float)
            ax.plot(t, actual, lw=1.2, label=candidate)
            if candidate == sorted({key[0] for key in groups})[0]:
                ax.plot(t, target, "--", lw=1.0, color="black", label="target")
        ax.set_title(f"CRT gripper {direction} step response")
        ax.set_xlabel("time (s)")
        ax.set_ylabel("pad scalar angle (deg)")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=8)
        fig.savefig(out_dir / f"gripper_{direction}_step_response.png", dpi=160)
        plt.close(fig)


def main():
    candidates = _parse_candidates(args_cli.candidates)
    out_dir = args_cli.out_dir
    if out_dir is None:
        out_dir = REPO_ROOT / "logs" / "gripper_drive_response" / datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    robot_usd = _write_robot_wrapper_usd(args_cli.source_usd, args_cli.source_robot_prim, args_cli.wrapper_usd)

    raw_rows: list[dict] = []
    summary_rows: list[dict] = []
    for stiffness, damping, effort in candidates:
        candidate_name = f"k{stiffness:g}_d{damping:g}_f{effort:g}"
        print(f"[CAND] {candidate_name}", flush=True)
        env_cfg = _make_env_cfg(robot_usd, stiffness, damping, effort, args_cli.velocity_limit)
        env = ManagerBasedRLEnv(cfg=env_cfg)
        try:
            dt = float(env.physics_dt)
            settle_steps = max(1, int(round(float(args_cli.settle_time) / dt)))
            hold_steps = max(1, int(round(float(args_cli.hold_time) / dt)))
            robot = env.scene["robot"]
            joint_names_order = list(GRIPPER_JOINT_TARGET_MULTIPLIERS.keys())
            joint_ids, joint_names = robot.find_joints(joint_names_order, preserve_order=True)
            joint_ids = [int(v) for v in list(joint_ids)]
            joint_names = [str(v) for v in list(joint_names)]
            print(f"[INFO] gripper joint_ids={joint_ids} names={joint_names}", flush=True)
            try:
                for act_name, act in robot.actuators.items():
                    if "gripper" in act_name:
                        print(
                            f"[INFO] actuator {act_name}: joints={getattr(act, 'joint_indices', None)} "
                            f"stiff={getattr(act, 'stiffness', None)} damp={getattr(act, 'damping', None)} "
                            f"effort={getattr(act, 'effort_limit_sim', None)}",
                            flush=True,
                        )
            except Exception as exc:
                print(f"[WARN] actuator print failed: {exc}", flush=True)
            if joint_names != joint_names_order:
                print(f"[WARN] gripper joint order={joint_names}", flush=True)
            multipliers = torch.tensor(
                [float(GRIPPER_JOINT_TARGET_MULTIPLIERS[name]) for name in joint_names],
                device=env.device,
                dtype=torch.float32,
            )
            primary_cols = [joint_names.index(name) for name in PRIMARY_JOINT_NAMES]
            command_joint_ids = joint_ids
            command_multipliers = multipliers
            if args_cli.primary_only:
                command_joint_ids = [joint_ids[col] for col in primary_cols]
                command_multipliers = multipliers[primary_cols]
                print(f"[INFO] primary-only command ids={command_joint_ids}", flush=True)

            for direction, step_sign in (("negative_step", -1.0), ("positive_step", 1.0)):
                env.reset()
                # Hold all non-tested joints at their current pose; otherwise the arm's implicit PD target
                # remains at its default buffer value and arm motion contaminates the gripper response.
                full_hold_target = robot.data.joint_pos.clone()
                robot.set_joint_position_target(full_hold_target)
                robot.set_joint_velocity_target(torch.zeros_like(full_hold_target))
                _sim_step(env, dt)
                # Test the native implicit drive response around the current USD pose.
                # The semantic 57deg/0deg gripper aperture is not the same as every USD joint's native position.
                start_target = robot.data.joint_pos[:, command_joint_ids].clone()
                _set_gripper_target(robot, command_joint_ids, start_target)
                for _ in range(settle_steps):
                    _set_gripper_target(robot, command_joint_ids, start_target)
                    _sim_step(env, dt)
                left0_deg, right0_deg, start_deg = _actual_scalar_deg(robot, joint_ids, multipliers, primary_cols)
                start_target = robot.data.joint_pos[:, command_joint_ids].clone()
                step_delta = _display_to_native(float(args_cli.step_deg) * float(step_sign))
                step_target = start_target + step_delta * command_multipliers.view(1, -1)
                target_deg = start_deg + float(args_cli.step_deg) * float(step_sign)
                _set_gripper_target(robot, command_joint_ids, step_target)
                t_series = []
                y_series = []
                for step in range(hold_steps + 1):
                    _set_gripper_target(robot, command_joint_ids, step_target)
                    _sim_step(env, dt)
                    left_deg, right_deg, mean_deg = _actual_scalar_deg(robot, joint_ids, multipliers, primary_cols)
                    t = (step + 1) * dt
                    if step == 0:
                        try:
                            print(
                                f"[DBG] {candidate_name} {direction} target_buf="
                                f"{robot.data.joint_pos_target[0, command_joint_ids].detach().cpu().tolist()} "
                                f"sim_target={robot._joint_pos_target_sim[0, command_joint_ids].detach().cpu().tolist()} "
                                f"pos={robot.data.joint_pos[0, command_joint_ids].detach().cpu().tolist()} "
                                f"vel={robot.data.joint_vel[0, command_joint_ids].detach().cpu().tolist()} "
                                f"computed_tau={robot.data.computed_torque[0, command_joint_ids].detach().cpu().tolist()} "
                                f"applied_tau={robot.data.applied_torque[0, command_joint_ids].detach().cpu().tolist()}",
                                flush=True,
                            )
                        except Exception as exc:
                            print(f"[DBG] failed: {exc}", flush=True)
                    raw_rows.append(
                        {
                            "candidate": candidate_name,
                            "stiffness": stiffness,
                            "damping": damping,
                            "effort": effort,
                            "velocity_limit": float(args_cli.velocity_limit),
                            "step_size_deg": float(args_cli.step_deg),
                            "joint_units": args_cli.joint_units,
                            "direction": direction,
                            "time_s": t,
                            "target_deg": target_deg,
                            "actual_left_deg": left_deg,
                            "actual_right_deg": right_deg,
                            "actual_mean_deg": mean_deg,
                            "error_deg": target_deg - mean_deg,
                        }
                    )
                    t_series.append(t)
                    y_series.append(mean_deg)
                metrics = _metric_summary(np.asarray(t_series), np.asarray(y_series), start_deg, target_deg)
                summary = {
                    "candidate": candidate_name,
                    "stiffness": stiffness,
                    "damping": damping,
                    "effort": effort,
                    "velocity_limit": float(args_cli.velocity_limit),
                    "step_size_deg": float(args_cli.step_deg),
                    "joint_units": args_cli.joint_units,
                    "direction": direction,
                    "start_left_deg": left0_deg,
                    "start_right_deg": right0_deg,
                    "start_deg": start_deg,
                    "target_deg": target_deg,
                    **metrics,
                }
                summary_rows.append(summary)
                print(
                    f"[STEP] {candidate_name} {direction}: "
                    f"final_err={summary['final_error_deg']:+.3f}deg "
                    f"overshoot={summary['overshoot_pct']:.2f}% "
                    f"rise={summary['rise_time_10_90_s']:.3f}s "
                    f"settle={summary['settling_time_2pct_s']:.3f}s",
                    flush=True,
                )
        finally:
            env.close()

    _write_csv(raw_rows, out_dir / "gripper_drive_response_raw.csv")
    _write_csv(summary_rows, out_dir / "gripper_drive_response_summary.csv")
    if args_cli.plot:
        _plot(raw_rows, out_dir)
    scored = []
    for stiffness, damping, effort in candidates:
        name = f"k{stiffness:g}_d{damping:g}_f{effort:g}"
        rows = [row for row in summary_rows if row["candidate"] == name]
        if len(rows) != 2:
            continue
        settle_vals = [row["settling_time_2pct_s"] for row in rows]
        rise_vals = [row["rise_time_10_90_s"] for row in rows]
        overshoot_vals = [row["overshoot_pct"] for row in rows]
        final_err_vals = [abs(row["final_error_deg"]) for row in rows]
        # Penalize unsettle/nan heavily; prefer low overshoot and fast response.
        settle_score = sum(v if math.isfinite(v) else 10.0 for v in settle_vals) / len(settle_vals)
        rise_score = sum(v if math.isfinite(v) else 10.0 for v in rise_vals) / len(rise_vals)
        overshoot_score = max(overshoot_vals)
        final_score = max(final_err_vals)
        score = settle_score + 0.5 * rise_score + 0.05 * overshoot_score + 0.1 * final_score
        scored.append((score, name, stiffness, damping, effort, settle_score, rise_score, overshoot_score, final_score))
    scored.sort(key=lambda item: item[0])
    if scored:
        best = scored[0]
        print(
            "[BEST] "
            f"{best[1]} score={best[0]:.4f} avg_settle={best[5]:.4f}s avg_rise={best[6]:.4f}s "
            f"max_overshoot={best[7]:.2f}% max_final_err={best[8]:.4f}deg",
            flush=True,
        )
    print(f"[OK] Raw CSV: {out_dir / 'gripper_drive_response_raw.csv'}", flush=True)
    print(f"[OK] Summary CSV: {out_dir / 'gripper_drive_response_summary.csv'}", flush=True)
    print(f"[OK] Out dir: {out_dir}", flush=True)


if __name__ == "__main__":
    try:
        try:
            main()
        except BaseException as exc:
            print(f"[ERROR] {type(exc).__name__}: {exc!r}", flush=True)
            raise
    finally:
        simulation_app.close()
