#!/usr/bin/env python3
"""对比多份日志中的 TCP 和工件位姿曲线。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "test_log" / "same_initial_action_compare" / "isaac_sim_vs_xarm7_sim"
DEFAULT_ISAAC_CSV = REPO_ROOT / "test_log" / "xarm7_pick_pose_2026-05-18_16-41-45.csv"
DEFAULT_XARM7_CSV = REPO_ROOT / "test_log" / "sdk_obs_rollout_2026-05-18_18-10-50.csv"
POSE_FIELD_NAMES = [
    "tcp_x_m", "tcp_y_m", "tcp_z_m", "tcp_qw", "tcp_qx", "tcp_qy", "tcp_qz",
    "object_x_m", "object_y_m", "object_z_m", "object_qw", "object_qx", "object_qy", "object_qz",
]


@dataclass
class PoseColumns:
    pos_cols: list[str]
    quat_cols: list[str]
    pos_scale: float


@dataclass
class Dataset:
    label: str
    path: Path
    rows: dict[tuple[int, ...], np.ndarray]
    tcp_cols: PoseColumns
    object_cols: PoseColumns
    total_rows: int
    kept_rows: int
    duplicate_keys: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Align logs and plot TCP/object absolute poses.")
    parser.add_argument("--isaac-csv", type=Path, default=DEFAULT_ISAAC_CSV)
    parser.add_argument("--xarm7-csv", type=Path, default=DEFAULT_XARM7_CSV)
    parser.add_argument(
        "--input",
        action="append",
        default=None,
        metavar="LABEL=CSV",
        help="Optional repeated input spec. Overrides --isaac-csv/--xarm7-csv.",
    )
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument(
        "--align-by",
        choices=("episode_step", "g_step", "step", "csv_row"),
        default="episode_step",
    )
    parser.add_argument("--episode", type=int, default=None)
    parser.add_argument("--max-frames", type=int, default=-1)
    parser.add_argument("--include-done", action="store_true")
    return parser.parse_args()


def boolish(value: str | None) -> bool:
    return str(value).strip().lower() in {"1", "true", "t", "yes", "y"}


def int_field(row: dict[str, str], name: str, fallback: int | None = None) -> int:
    value = row.get(name, "")
    if value == "" or value is None:
        if fallback is None:
            raise KeyError(name)
        return fallback
    return int(float(value))


def make_key(row: dict[str, str], row_index: int, align_by: str) -> tuple[int, ...]:
    if align_by == "episode_step":
        return (int_field(row, "episode"), int_field(row, "step"))
    if align_by == "g_step":
        return (int_field(row, "g_step"),)
    if align_by == "step":
        return (int_field(row, "step"),)
    if align_by == "csv_row":
        return (int_field(row, "csv_row", row_index),)
    raise ValueError(f"Unsupported align key: {align_by}")


def key_names(align_by: str) -> list[str]:
    if align_by == "episode_step":
        return ["episode", "step"]
    return [align_by]


def x_axis(keys: list[tuple[int, ...]], align_by: str) -> tuple[np.ndarray, str]:
    if align_by == "episode_step":
        episodes = {key[0] for key in keys}
        if len(episodes) == 1:
            return np.asarray([key[1] for key in keys], dtype=np.int64), "step"
    if align_by in {"g_step", "step", "csv_row"}:
        return np.asarray([key[0] for key in keys], dtype=np.int64), align_by
    return np.arange(len(keys), dtype=np.int64), "aligned frame"


def _pose_candidates(pose_name: str) -> list[PoseColumns]:
    if pose_name == "tcp":
        prefixes = ("tcp", "ee")
    elif pose_name == "object":
        prefixes = ("object", "obj")
    else:
        raise ValueError(pose_name)

    candidates: list[PoseColumns] = []
    for prefix in prefixes:
        candidates.append(
            PoseColumns(
                [f"{prefix}_x_m", f"{prefix}_y_m", f"{prefix}_z_m"],
                [f"{prefix}_qw", f"{prefix}_qx", f"{prefix}_qy", f"{prefix}_qz"],
                1.0,
            )
        )
        candidates.append(
            PoseColumns(
                [f"{prefix}_x_mm", f"{prefix}_y_mm", f"{prefix}_z_mm"],
                [f"{prefix}_qw", f"{prefix}_qx", f"{prefix}_qy", f"{prefix}_qz"],
                0.001,
            )
        )
    return candidates


def find_pose_cols(fieldnames: list[str], label: str, pose_name: str) -> PoseColumns:
    fields = set(fieldnames)
    for candidate in _pose_candidates(pose_name):
        if all(col in fields for col in candidate.pos_cols + candidate.quat_cols):
            return candidate

    pose_like = [
        name for name in fieldnames
        if any(token in name.lower() for token in ("tcp", "ee_", "object", "obj"))
    ]
    raise RuntimeError(
        f"{label}: cannot find absolute {pose_name} pose columns. "
        "Rerun Isaac play/xArm rollout with TCP logging enabled. "
        f"Available pose-like columns: {pose_like[:64]}"
    )


def normalize_quat(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = float(np.linalg.norm(quat))
    if norm < 1.0e-12:
        return quat
    quat = quat / norm
    if quat[0] < 0.0:
        quat = -quat
    return quat


def quat_angle_deg(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    dots = np.abs(np.sum(a * b, axis=-1))
    dots = np.clip(dots, -1.0, 1.0)
    return np.rad2deg(2.0 * np.arccos(dots))


def read_pose(row: dict[str, str], cols: PoseColumns) -> tuple[np.ndarray, np.ndarray]:
    pos = np.asarray([float(row[col]) for col in cols.pos_cols], dtype=np.float64) * cols.pos_scale
    quat = normalize_quat(np.asarray([float(row[col]) for col in cols.quat_cols], dtype=np.float64))
    return pos, quat


def load_dataset(label: str, path: Path, args: argparse.Namespace) -> Dataset:
    path = path.expanduser().resolve()
    rows: dict[tuple[int, ...], np.ndarray] = {}
    total_rows = 0
    kept_rows = 0
    duplicate_keys = 0

    with path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise RuntimeError(f"{label}: CSV has no header: {path}")
        tcp_cols = find_pose_cols(reader.fieldnames, label, "tcp")
        object_cols = find_pose_cols(reader.fieldnames, label, "object")

        for row_index, row in enumerate(reader):
            total_rows += 1
            if args.episode is not None:
                try:
                    if int_field(row, "episode") != args.episode:
                        continue
                except KeyError:
                    raise RuntimeError(f"{label}: --episode was set, but CSV has no episode column") from None
            if not args.include_done and (boolish(row.get("done")) or boolish(row.get("timeout"))):
                continue

            key = make_key(row, row_index, args.align_by)
            tcp_pos, tcp_quat = read_pose(row, tcp_cols)
            object_pos, object_quat = read_pose(row, object_cols)
            values = np.concatenate([tcp_pos, tcp_quat, object_pos, object_quat], axis=0)
            kept_rows += 1
            if key in rows:
                duplicate_keys += 1
                continue
            rows[key] = values

    if not rows:
        raise RuntimeError(f"{label}: no rows loaded from {path}")
    return Dataset(label, path, rows, tcp_cols, object_cols, total_rows, kept_rows, duplicate_keys)


def input_specs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if not args.input:
        return [("isaac_sim", args.isaac_csv), ("xarm7_sim", args.xarm7_csv)]

    specs: list[tuple[str, Path]] = []
    seen: set[str] = set()
    for item in args.input:
        if "=" not in item:
            raise RuntimeError(f"--input must be LABEL=CSV, got: {item}")
        label, path_str = item.split("=", 1)
        label = label.strip()
        if not label:
            raise RuntimeError(f"--input label is empty: {item}")
        if label in seen:
            raise RuntimeError(f"Duplicate --input label: {label}")
        seen.add(label)
        specs.append((label, Path(path_str).expanduser()))
    if len(specs) < 2:
        raise RuntimeError("Need at least two --input entries to compare.")
    return specs


def aligned_keys(datasets: list[Dataset], max_frames: int) -> list[tuple[int, ...]]:
    common = set(datasets[0].rows)
    for ds in datasets[1:]:
        common &= set(ds.rows)
    keys = sorted(common)
    if max_frames >= 0:
        keys = keys[:max_frames]
    if not keys:
        labels = ", ".join(ds.label for ds in datasets)
        raise RuntimeError(f"No common frames after alignment for: {labels}")
    return keys


def import_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting. Try: pip install matplotlib") from exc
    return plt


def pose_slice(pose_name: str) -> tuple[slice, slice]:
    if pose_name == "tcp":
        return slice(0, 3), slice(3, 7)
    if pose_name == "object":
        return slice(7, 10), slice(10, 14)
    raise ValueError(pose_name)


def plot_components(
    out_path: Path,
    keys: list[tuple[int, ...]],
    arrays: dict[str, np.ndarray],
    align_by: str,
    pose_name: str,
    group: str,
) -> None:
    plt = import_pyplot()
    x, xlabel = x_axis(keys, align_by)
    pos_slc, quat_slc = pose_slice(pose_name)
    if group == "xyz":
        slc = pos_slc
        labels = ["x_m", "y_m", "z_m"]
    elif group == "quat":
        slc = quat_slc
        labels = ["qw", "qx", "qy", "qz"]
    else:
        raise ValueError(group)

    fig, axes = plt.subplots(len(labels), 1, figsize=(14, max(7, len(labels) * 1.9)), sharex=True, constrained_layout=True)
    if len(labels) == 1:
        axes = [axes]
    for idx, ax in enumerate(axes):
        for label, values in arrays.items():
            ax.plot(x, values[:, slc][:, idx], label=label, linewidth=1.35)
        ax.grid(True, alpha=0.3)
        ax.set_ylabel(labels[idx])
    axes[0].legend(loc="upper right", ncols=min(3, len(arrays)))
    axes[0].set_title(f"{pose_name} {group} aligned by {align_by} ({len(keys)} frames)")
    axes[-1].set_xlabel(xlabel)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def _set_equal_3d_axes(ax, points: np.ndarray) -> None:
    mins = np.nanmin(points, axis=0)
    maxs = np.nanmax(points, axis=0)
    centers = (mins + maxs) * 0.5
    radius = max(float(np.max(maxs - mins)) * 0.5, 1.0e-6)
    ax.set_xlim(centers[0] - radius, centers[0] + radius)
    ax.set_ylim(centers[1] - radius, centers[1] + radius)
    ax.set_zlim(centers[2] - radius, centers[2] + radius)


def plot_trajectory_3d(
    out_path: Path,
    arrays: dict[str, np.ndarray],
    pose_name: str,
) -> None:
    plt = import_pyplot()
    pos_slc, _ = pose_slice(pose_name)
    fig = plt.figure(figsize=(9, 8), constrained_layout=True)
    ax = fig.add_subplot(111, projection="3d")
    all_points = []
    for label, values in arrays.items():
        xyz = values[:, pos_slc]
        all_points.append(xyz)
        ax.plot(xyz[:, 0], xyz[:, 1], xyz[:, 2], label=label, linewidth=1.6)
        ax.scatter(xyz[0, 0], xyz[0, 1], xyz[0, 2], s=30, marker="o")
        ax.scatter(xyz[-1, 0], xyz[-1, 1], xyz[-1, 2], s=36, marker="x")
    _set_equal_3d_axes(ax, np.concatenate(all_points, axis=0))
    ax.set_xlabel("x_m")
    ax.set_ylabel("y_m")
    ax.set_zlabel("z_m")
    ax.set_title(f"{pose_name} xyz trajectory")
    ax.legend(loc="upper right")
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_diff(
    out_path: Path,
    keys: list[tuple[int, ...]],
    arrays: dict[str, np.ndarray],
    align_by: str,
    pose_name: str,
) -> None:
    labels = list(arrays)
    base_label = labels[0]
    base = arrays[base_label]
    pos_slc, quat_slc = pose_slice(pose_name)
    x, xlabel = x_axis(keys, align_by)
    plt = import_pyplot()

    fig, axes = plt.subplots(5, 1, figsize=(14, 10), sharex=True, constrained_layout=True)
    for label in labels[1:]:
        diff_xyz = arrays[label][:, pos_slc] - base[:, pos_slc]
        for idx, suffix in enumerate(("x", "y", "z")):
            axes[idx].plot(x, diff_xyz[:, idx], label=f"{label} - {base_label}", linewidth=1.25)
            axes[idx].set_ylabel(f"d{suffix}_m")
        axes[3].plot(x, np.linalg.norm(diff_xyz, axis=1), label=f"{label} - {base_label}", linewidth=1.25)
        axes[4].plot(x, quat_angle_deg(arrays[label][:, quat_slc], base[:, quat_slc]), label=f"{label} vs {base_label}", linewidth=1.25)
    for ax in axes:
        ax.axhline(0.0, color="0.35", linewidth=0.8)
        ax.grid(True, alpha=0.3)
    axes[3].set_ylabel("pos_err_m")
    axes[4].set_ylabel("ori_err_deg")
    axes[0].legend(loc="upper right", ncols=min(2, len(labels) - 1))
    axes[0].set_title(f"{pose_name} diff vs {base_label} ({len(keys)} frames)")
    axes[-1].set_xlabel(xlabel)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_aligned_csv(
    out_path: Path,
    keys: list[tuple[int, ...]],
    arrays: dict[str, np.ndarray],
    align_by: str,
) -> None:
    labels = list(arrays)
    base_label = labels[0]
    header = ["aligned_index"] + key_names(align_by)
    for label in labels:
        header += [f"{label}_{name}" for name in POSE_FIELD_NAMES]
    for label in labels[1:]:
        for pose_name in ("tcp", "object"):
            header += [
                f"{label}_minus_{base_label}_{pose_name}_dx_m",
                f"{label}_minus_{base_label}_{pose_name}_dy_m",
                f"{label}_minus_{base_label}_{pose_name}_dz_m",
                f"{label}_vs_{base_label}_{pose_name}_pos_err_m",
                f"{label}_vs_{base_label}_{pose_name}_ori_err_deg",
            ]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for index, key in enumerate(keys):
            row: list[str | int] = [index] + list(key)
            for label in labels:
                row += [f"{v:.8f}" for v in arrays[label][index]]
            for label in labels[1:]:
                for pose_name in ("tcp", "object"):
                    pos_slc, quat_slc = pose_slice(pose_name)
                    diff_xyz = arrays[label][index, pos_slc] - arrays[base_label][index, pos_slc]
                    pos_err = float(np.linalg.norm(diff_xyz))
                    ori_err = float(quat_angle_deg(arrays[label][index:index + 1, quat_slc], arrays[base_label][index:index + 1, quat_slc])[0])
                    row += [f"{v:.8f}" for v in diff_xyz]
                    row += [f"{pos_err:.8f}", f"{ori_err:.8f}"]
            writer.writerow(row)


def format_array(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{v:.6g}" for v in values) + "]"


def summary_lines(
    datasets: list[Dataset],
    keys: list[tuple[int, ...]],
    arrays: dict[str, np.ndarray],
    align_by: str,
    outputs: list[Path],
) -> list[str]:
    labels = list(arrays)
    base_label = labels[0]
    lines = [
        "TCP/object pose comparison",
        f"align_by={align_by}",
        f"aligned_frames={len(keys)}",
        f"first_key={keys[0]}",
        f"last_key={keys[-1]}",
        "",
        "Inputs:",
    ]
    for ds in datasets:
        lines.append(
            f"- {ds.label}: rows={ds.total_rows}, kept={ds.kept_rows}, "
            f"unique_keys={len(ds.rows)}, duplicates={ds.duplicate_keys}, path={ds.path}"
        )
        lines.append(f"  tcp_cols={ds.tcp_cols.pos_cols + ds.tcp_cols.quat_cols}")
        lines.append(f"  object_cols={ds.object_cols.pos_cols + ds.object_cols.quat_cols}")

    lines += ["", f"Diff stats vs {base_label}:"]
    for label in labels[1:]:
        for pose_name in ("tcp", "object"):
            pos_slc, quat_slc = pose_slice(pose_name)
            diff_xyz = arrays[label][:, pos_slc] - arrays[base_label][:, pos_slc]
            pos_err = np.linalg.norm(diff_xyz, axis=1)
            ori_err = quat_angle_deg(arrays[label][:, quat_slc], arrays[base_label][:, quat_slc])
            lines.append(f"- {label} {pose_name}: mean_abs_xyz_m={format_array(np.mean(np.abs(diff_xyz), axis=0))}")
            lines.append(f"- {label} {pose_name}: max_abs_xyz_m={format_array(np.max(np.abs(diff_xyz), axis=0))}")
            lines.append(f"- {label} {pose_name}: mean_pos_err_m={float(np.mean(pos_err)):.6g}, max_pos_err_m={float(np.max(pos_err)):.6g}")
            lines.append(f"- {label} {pose_name}: mean_ori_err_deg={float(np.mean(ori_err)):.6g}, max_ori_err_deg={float(np.max(ori_err)):.6g}")

    lines += ["", "Outputs:"]
    for path in outputs:
        lines.append(f"- {path}")
    return lines


def main() -> None:
    args = parse_args()
    datasets = [load_dataset(label, path, args) for label, path in input_specs(args)]
    keys = aligned_keys(datasets, args.max_frames)
    arrays = {ds.label: np.stack([ds.rows[key] for key in keys], axis=0) for ds in datasets}

    args.out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"pose_{args.align_by}"
    outputs = [
        args.out_dir / f"{stem}_tcp_xyz_overlay.png",
        args.out_dir / f"{stem}_tcp_quat_overlay.png",
        args.out_dir / f"{stem}_object_xyz_overlay.png",
        args.out_dir / f"{stem}_object_quat_overlay.png",
        args.out_dir / f"{stem}_tcp_xyz_3d.png",
        args.out_dir / f"{stem}_object_xyz_3d.png",
        args.out_dir / f"{stem}_tcp_diff_vs_sim.png",
        args.out_dir / f"{stem}_object_diff_vs_sim.png",
        args.out_dir / f"{stem}_aligned.csv",
        args.out_dir / f"{stem}_summary.txt",
    ]

    plot_components(outputs[0], keys, arrays, args.align_by, "tcp", "xyz")
    plot_components(outputs[1], keys, arrays, args.align_by, "tcp", "quat")
    plot_components(outputs[2], keys, arrays, args.align_by, "object", "xyz")
    plot_components(outputs[3], keys, arrays, args.align_by, "object", "quat")
    plot_trajectory_3d(outputs[4], arrays, "tcp")
    plot_trajectory_3d(outputs[5], arrays, "object")
    plot_diff(outputs[6], keys, arrays, args.align_by, "tcp")
    plot_diff(outputs[7], keys, arrays, args.align_by, "object")
    write_aligned_csv(outputs[8], keys, arrays, args.align_by)
    lines = summary_lines(datasets, keys, arrays, args.align_by, outputs)
    outputs[9].write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    try:
        main()
    except RuntimeError as exc:
        raise SystemExit(str(exc)) from None
