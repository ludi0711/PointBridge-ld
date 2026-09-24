#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""CSV logging helpers for pick-pose play/eval runs."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence, TextIO

from tools.policy_log_paths import (
    fallback_policy_time,
    infer_policy_time_from_checkpoint,
    make_policy_output_path,
    policy_log_dir,
)


@dataclass(frozen=True)
class PickPoseCsvRow:
    episode: int
    step: int
    g_step: int
    done: bool
    timeout: bool
    obs: Sequence[float]
    raw_action: Sequence[float]
    dist_cm: float
    ori_err_deg: float
    terminal_dist_cm: float | None
    terminal_ori_err_deg: float | None
    terminal_success: bool
    terminal_source: str
    tcp_pos_m: Sequence[float]
    tcp_quat_wxyz: Sequence[float]
    object_pos_m: Sequence[float]
    object_quat_wxyz: Sequence[float]
    cmd_delta_deg: Sequence[float]
    actual_delta_deg: Sequence[float]
    delta_error_deg: Sequence[float]
    joint_deg: Sequence[float]
    pad_force_l_n: float | None = None
    pad_force_r_n: float | None = None
    pad_force_total_n: float | None = None
    gripper_target_deg: float | None = None
    gripper_actual_left_deg: float | None = None
    gripper_actual_right_deg: float | None = None
    gripper_actual_mean_deg: float | None = None
    gripper_tracking_error_deg: float | None = None
    gripper_force_pid_u_deg: float | None = None
    gripper_force_pid_delta_deg: float | None = None
    gripper_actual_step_deg: float | None = None
    gripper_target_left_deg: float | None = None
    gripper_target_right_deg: float | None = None
    gripper_left_contact: float | None = None
    gripper_right_contact: float | None = None
    gripper_both_contact: float | None = None
    gripper_closed_mask: float | None = None
    gripper_hold_mask: float | None = None
    gripper_force_servo_active: float | None = None
    gripper_actual_vel_mean_deg_s: float | None = None
    gripper_effort_limit_min: float | None = None
    gripper_effort_limit_max: float | None = None
    gripper_chain_span_m: float | None = None
    gripper_pad_distance_m: float | None = None
    gripper_chain_broken: bool | None = None


@dataclass
class PickPoseCsvLog:
    path: Path
    file: TextIO
    writer: Any

    def write_row(self, row: PickPoseCsvRow) -> None:
        self.writer.writerow(format_pick_pose_row(row))

    def flush(self) -> None:
        self.file.flush()

    def close(self) -> None:
        self.file.close()


def build_pick_pose_header(obs_dim: int, action_dim: int = 7, arm_dim: int = 7) -> list[str]:
    header = ["episode", "step", "g_step", "done", "timeout"]
    header += [f"mdp_obs_{i:02d}" for i in range(obs_dim)]
    header += [f"mdp_raw_action_{i:02d}" for i in range(action_dim)]
    header += [
        "dist_cm",
        "ori_err_deg",
        "terminal_dist_cm",
        "terminal_ori_err_deg",
        "terminal_success",
        "terminal_source",
    ]
    header += ["tcp_x_m", "tcp_y_m", "tcp_z_m", "tcp_qw", "tcp_qx", "tcp_qy", "tcp_qz"]
    header += ["object_x_m", "object_y_m", "object_z_m", "object_qw", "object_qx", "object_qy", "object_qz"]
    header += [f"cmd_delta_j{i}_deg" for i in range(1, arm_dim + 1)]
    header += [f"actual_delta_j{i}_deg" for i in range(1, arm_dim + 1)]
    header += [f"delta_error_j{i}_deg" for i in range(1, arm_dim + 1)]
    header += [f"joint{i}_deg" for i in range(1, arm_dim + 1)]
    header += [
        "pad_force_l_n",
        "pad_force_r_n",
        "pad_force_total_n",
        "gripper_target_deg",
        "gripper_actual_left_deg",
        "gripper_actual_right_deg",
        "gripper_actual_mean_deg",
        "gripper_tracking_error_deg",
        "gripper_force_pid_u_deg",
        "gripper_force_pid_delta_deg",
        "gripper_actual_step_deg",
        "gripper_target_left_deg",
        "gripper_target_right_deg",
        "gripper_left_contact",
        "gripper_right_contact",
        "gripper_both_contact",
        "gripper_closed_mask",
        "gripper_hold_mask",
        "gripper_force_servo_active",
        "gripper_actual_vel_mean_deg_s",
        "gripper_effort_limit_min",
        "gripper_effort_limit_max",
        "gripper_chain_span_m",
        "gripper_pad_distance_m",
        "gripper_chain_broken",
    ]
    return header


def format_pick_pose_row(row: PickPoseCsvRow) -> list[Any]:
    return (
        [row.episode, row.step, row.g_step, int(row.done), int(row.timeout)]
        + _format_seq(row.obs, 8)
        + _format_seq(row.raw_action, 8)
        + [
            f"{row.dist_cm:.6f}",
            f"{row.ori_err_deg:.6f}",
            _format_optional(row.terminal_dist_cm, 6),
            _format_optional(row.terminal_ori_err_deg, 6),
            int(row.terminal_success),
            row.terminal_source,
        ]
        + _format_seq(row.tcp_pos_m, 8)
        + _format_seq(row.tcp_quat_wxyz, 8)
        + _format_seq(row.object_pos_m, 8)
        + _format_seq(row.object_quat_wxyz, 8)
        + _format_seq(row.cmd_delta_deg, 6)
        + _format_seq(row.actual_delta_deg, 6)
        + _format_seq(row.delta_error_deg, 6)
        + _format_seq(row.joint_deg, 6)
        + [
            _format_optional(row.pad_force_l_n, 6),
            _format_optional(row.pad_force_r_n, 6),
            _format_optional(row.pad_force_total_n, 6),
            _format_optional(row.gripper_target_deg, 6),
            _format_optional(row.gripper_actual_left_deg, 6),
            _format_optional(row.gripper_actual_right_deg, 6),
            _format_optional(row.gripper_actual_mean_deg, 6),
            _format_optional(row.gripper_tracking_error_deg, 6),
            _format_optional(row.gripper_force_pid_u_deg, 6),
            _format_optional(row.gripper_force_pid_delta_deg, 6),
            _format_optional(row.gripper_actual_step_deg, 6),
            _format_optional(row.gripper_target_left_deg, 6),
            _format_optional(row.gripper_target_right_deg, 6),
            _format_optional(row.gripper_left_contact, 6),
            _format_optional(row.gripper_right_contact, 6),
            _format_optional(row.gripper_both_contact, 6),
            _format_optional(row.gripper_closed_mask, 6),
            _format_optional(row.gripper_hold_mask, 6),
            _format_optional(row.gripper_force_servo_active, 6),
            _format_optional(row.gripper_actual_vel_mean_deg_s, 6),
            _format_optional(row.gripper_effort_limit_min, 6),
            _format_optional(row.gripper_effort_limit_max, 6),
            _format_optional(row.gripper_chain_span_m, 6),
            _format_optional(row.gripper_pad_distance_m, 6),
            "" if row.gripper_chain_broken is None else int(bool(row.gripper_chain_broken)),
        ]
    )


def open_pick_pose_csv(
    *,
    root: str | Path,
    log_name: str,
    checkpoint_path: str | Path | None,
    file_prefix: str,
    obs_dim: int,
    action_dim: int = 7,
    arm_dim: int = 7,
    fallback_prefix: str,
    log_dir_label: str,
    log_file_label: str,
) -> PickPoseCsvLog:
    fallback = fallback_policy_time(fallback_prefix)
    policy_time = infer_policy_time_from_checkpoint(checkpoint_path, fallback=fallback)
    path = make_policy_output_path(root, log_name, policy_time, file_prefix)

    csv_file = open(path, "w", newline="", encoding="utf-8")
    writer = csv.writer(csv_file)
    writer.writerow(build_pick_pose_header(obs_dim, action_dim=action_dim, arm_dim=arm_dim))

    print(f"[PolicyTime] {policy_time}")
    print(f"[{log_dir_label}] {policy_log_dir(root, log_name, policy_time)}")
    print(f"[{log_file_label}] {path}")
    return PickPoseCsvLog(path=path, file=csv_file, writer=writer)


def _format_seq(values: Sequence[float], precision: int) -> list[str]:
    return [f"{float(v):.{precision}f}" for v in values]


def _format_optional(value: float | None, precision: int) -> str:
    return "" if value is None else f"{float(value):.{precision}f}"
