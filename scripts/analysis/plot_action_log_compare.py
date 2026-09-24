#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""对比多份动作日志并绘制关节动作曲线。"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np


ACT_DIM = 7
OBS_DIM = 21
OBS_LABELS = [
    "qrel_j1",
    "qrel_j2",
    "qrel_j3",
    "qrel_j4",
    "qrel_j5",
    "qrel_j6",
    "qrel_j7",
    "obj_x",
    "obj_y",
    "obj_z",
    "obj_qw",
    "obj_qx",
    "obj_qy",
    "obj_qz",
    "last_j1",
    "last_j2",
    "last_j3",
    "last_j4",
    "last_j5",
    "last_j6",
    "last_j7",
]
REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LOG_DIR = REPO_ROOT / "test_log"
DEFAULT_COMPARE_ROOT = DEFAULT_LOG_DIR / "same_initial_action_compare"

DEFAULT_SIM_CSV = DEFAULT_LOG_DIR / "xarm7_pick_pose_2026-05-18_16-41-45.csv"
DEFAULT_REAL_NORENDER_CSV = DEFAULT_LOG_DIR / "obs_replay_deploy_2026-05-18_16-45-48.csv"
DEFAULT_REAL_CSV = DEFAULT_LOG_DIR / "obs_replay_deploy_2026-05-18_16-48-09.csv"


@dataclass
class Dataset:
    label: str
    path: Path
    action_cols: list[str]
    rows: dict[tuple[int, ...], np.ndarray]
    total_rows: int
    kept_rows: int
    duplicate_keys: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Align sim/real CSV logs by frame key, then plot the 7 action dimensions "
            "for sim, real_norender, and real."
        )
    )
    parser.add_argument("--sim-csv", type=Path, default=DEFAULT_SIM_CSV)
    parser.add_argument("--real-norender-csv", type=Path, default=DEFAULT_REAL_NORENDER_CSV)
    parser.add_argument("--real-csv", type=Path, default=DEFAULT_REAL_CSV)
    parser.add_argument(
        "--input",
        action="append",
        default=None,
        metavar="LABEL=CSV",
        help="Optional repeated input spec, for example --input isaac_sim=sim.csv --input xarm7_sim=real.csv. Overrides the three default files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "Optional output base directory. Default: "
            "test_log/same_initial_action_compare/<sim2sim|sim2real>, "
            "then the script creates a value subfolder such as cmd_delta or raw_action."
        ),
    )
    parser.add_argument(
        "--align-by",
        choices=("episode_step", "g_step", "step", "csv_row"),
        default="episode_step",
        help="Frame key used for alignment. Default: episode_step.",
    )
    parser.add_argument(
        "--value",
        choices=("raw_action", "cmd_delta_deg", "actual_delta_deg", "obs"),
        default="cmd_delta_deg",
        help=("What to compare. raw_action uses action columns; *_delta_deg uses matching delta columns; obs uses sdk_obs or mdp_obs."),
    )
    parser.add_argument(
        "--real-raw-source",
        choices=("policy", "logged"),
        default="policy",
        help="For --value raw_action, choose policy_raw_action_j* or logged_raw_action_j* in real logs.",
    )
    parser.add_argument(
        "--episode",
        type=int,
        default=None,
        help="Optional episode filter before alignment.",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=-1,
        help="Optional limit after alignment. -1 keeps all aligned frames.",
    )
    parser.add_argument(
        "--include-done",
        action="store_true",
        help="Keep rows whose done or timeout flag is true. Default drops them to avoid reset jumps.",
    )
    return parser.parse_args()


def numbered_cols(prefix: str, suffix: str = "") -> list[str]:
    return [f"{prefix}{i}{suffix}" for i in range(1, ACT_DIM + 1)]


def obs_cols(prefix: str) -> list[str]:
    return [f"{prefix}_{i:02d}" for i in range(OBS_DIM)]


def candidate_action_cols(label: str, args: argparse.Namespace) -> list[list[str]]:
    if args.value == "obs":
        return [
            obs_cols("sdk_obs"),
            obs_cols("mdp_obs"),
            obs_cols("isaac_obs"),
            obs_cols("obs"),
        ]
    if args.value == "cmd_delta_deg":
        return [numbered_cols("cmd_delta_j", "_deg")]
    if args.value == "actual_delta_deg":
        return [numbered_cols("actual_delta_j", "_deg")]

    if args.real_raw_source == "policy":
        return [
            numbered_cols("mdp_raw_action_j"),
            numbered_cols("raw_action_j"),
            numbered_cols("policy_raw_action_j"),
        ]
    return [
        numbered_cols("mdp_raw_action_j"),
        numbered_cols("raw_action_j"),
        numbered_cols("logged_raw_action_j"),
    ]


def find_action_cols(fieldnames: list[str], label: str, args: argparse.Namespace) -> list[str]:
    for cols in candidate_action_cols(label, args):
        if all(col in fieldnames for col in cols):
            if args.value == "obs":
                prefix = cols[0].rsplit("_", 1)[0]
                old_extra = f"{prefix}_{OBS_DIM:02d}"
                if old_extra in fieldnames:
                    raise RuntimeError(
                        f"{label}: found {old_extra}; this looks like an old 28D velocity-observation log. "
                        "Regenerate logs with the 21D no-velocity config before comparing obs."
                    )
            return cols

    actionish = [
        name
        for name in fieldnames
        if "action" in name
        or name.startswith("cmd_delta_j")
        or name.startswith("actual_delta_j")
        or name.startswith("mdp_obs_")
        or name.startswith("sdk_obs_")
        or name.startswith("isaac_obs_")
    ]
    raise RuntimeError(
        f"{label}: cannot find {args.value} columns in CSV. "
        f"Available matching columns: {actionish[:48]}"
    )


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


def infer_compare_name(input_paths: list[Path]) -> str:
    for token in ("sim2real", "sim2sim"):
        if any(token in path.name.lower() for path in input_paths):
            return token
    return "sim2sim"


def value_subdir(value: str) -> str:
    return {
        "raw_action": "raw_action",
        "cmd_delta_deg": "cmd_delta",
        "actual_delta_deg": "actual_delta",
        "obs": "obs",
    }[value]


def output_dir(args: argparse.Namespace, specs: list[tuple[str, Path]]) -> Path:
    base_dir = args.out_dir
    if base_dir is None:
        compare_name = infer_compare_name([path for _, path in specs])
        base_dir = DEFAULT_COMPARE_ROOT / compare_name

    subdir = value_subdir(args.value)
    if base_dir.name == subdir:
        return base_dir
    return base_dir / subdir


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
        action_cols = find_action_cols(reader.fieldnames, label, args)

        for row_index, row in enumerate(reader):
            total_rows += 1
            if args.episode is not None:
                try:
                    if int_field(row, "episode") != args.episode:
                        continue
                except KeyError:
                    raise RuntimeError(f"{label}: --episode was set, but CSV has no episode column")

            if not args.include_done and (boolish(row.get("done")) or boolish(row.get("timeout"))):
                continue

            key = make_key(row, row_index, args.align_by)
            action = np.asarray([float(row[col]) for col in action_cols], dtype=np.float64)
            kept_rows += 1

            if key in rows:
                duplicate_keys += 1
                continue
            rows[key] = action

    if not rows:
        raise RuntimeError(f"{label}: no rows loaded from {path}")

    return Dataset(
        label=label,
        path=path,
        action_cols=action_cols,
        rows=rows,
        total_rows=total_rows,
        kept_rows=kept_rows,
        duplicate_keys=duplicate_keys,
    )


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


def x_axis(keys: list[tuple[int, ...]], align_by: str) -> tuple[np.ndarray, str]:
    if align_by == "episode_step":
        episodes = {key[0] for key in keys}
        if len(episodes) == 1:
            return np.asarray([key[1] for key in keys], dtype=np.int64), "step"
    if align_by in {"g_step", "step", "csv_row"}:
        return np.asarray([key[0] for key in keys], dtype=np.int64), align_by
    return np.arange(len(keys), dtype=np.int64), "aligned frame"


def write_aligned_csv(
    out_path: Path,
    keys: list[tuple[int, ...]],
    arrays: dict[str, np.ndarray],
    align_by: str,
    value: str,
) -> None:
    labels = list(arrays)
    base_label = labels[0]
    names = key_names(align_by)
    dim = next(iter(arrays.values())).shape[1]
    suffixes = value_suffixes(value, dim)

    header = ["aligned_index"] + names
    for label in labels:
        header += [f"{label}_{value}_{suffix}" for suffix in suffixes]
    for label in labels[1:]:
        header += [f"{label}_minus_{base_label}_{value}_{suffix}" for suffix in suffixes]

    with out_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        for index, key in enumerate(keys):
            row: list[str | int] = [index] + list(key)
            for label in labels:
                row += [f"{v:.8f}" for v in arrays[label][index]]
            for label in labels[1:]:
                diff = arrays[label][index] - arrays[base_label][index]
                row += [f"{v:.8f}" for v in diff]
            writer.writerow(row)


def import_pyplot():
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise RuntimeError("matplotlib is required for plotting. Try: pip install matplotlib") from exc
    return plt



def value_suffixes(value: str, dim: int) -> list[str]:
    if value == "obs":
        if dim == OBS_DIM:
            return [f"{i:02d}_{name}" for i, name in enumerate(OBS_LABELS)]
        return [f"{i:02d}" for i in range(dim)]
    return [f"j{i}" for i in range(1, dim + 1)]


def obs_groups(dim: int) -> list[tuple[str, slice]]:
    if dim != OBS_DIM:
        return [("all", slice(0, dim))]
    return [
        ("joint_pos_rel", slice(0, 7)),
        ("object_rel_pose", slice(7, 14)),
        ("last_action", slice(14, 21)),
    ]


def plot_overlay(
    out_path: Path,
    keys: list[tuple[int, ...]],
    arrays: dict[str, np.ndarray],
    align_by: str,
    value: str,
) -> None:
    plt = import_pyplot()
    x, xlabel = x_axis(keys, align_by)
    dim = next(iter(arrays.values())).shape[1]
    suffixes = value_suffixes(value, dim)

    fig, axes = plt.subplots(dim, 1, figsize=(14, max(7, min(22, dim * 1.8))), sharex=True, constrained_layout=True)
    if dim == 1:
        axes = [axes]
    for idx, ax in enumerate(axes):
        for label, values in arrays.items():
            ax.plot(x, values[:, idx], label=label, linewidth=1.35)
        ax.axhline(0.0, color="0.55", linewidth=0.7)
        ax.set_ylabel(suffixes[idx])
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="upper right", ncols=min(3, len(arrays)))
    axes[0].set_title(f"{value} aligned by {align_by} ({len(keys)} frames)")
    axes[-1].set_xlabel(xlabel)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_diff(
    out_path: Path,
    keys: list[tuple[int, ...]],
    arrays: dict[str, np.ndarray],
    align_by: str,
    value: str,
) -> None:
    labels = list(arrays)
    base_label = labels[0]
    plt = import_pyplot()
    x, xlabel = x_axis(keys, align_by)
    dim = arrays[base_label].shape[1]
    suffixes = value_suffixes(value, dim)

    fig, axes = plt.subplots(dim, 1, figsize=(14, max(7, min(22, dim * 1.8))), sharex=True, constrained_layout=True)
    if dim == 1:
        axes = [axes]
    for idx, ax in enumerate(axes):
        for label in labels[1:]:
            diff = arrays[label][:, idx] - arrays[base_label][:, idx]
            ax.plot(x, diff, label=f"{label} - {base_label}", linewidth=1.25)
        ax.axhline(0.0, color="0.35", linewidth=0.8)
        ax.set_ylabel(suffixes[idx])
        ax.grid(True, alpha=0.3)

    axes[0].legend(loc="upper right", ncols=min(2, len(labels) - 1))
    axes[0].set_title(f"{value} difference vs {base_label} ({len(keys)} frames)")
    axes[-1].set_xlabel(xlabel)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_obs_group_overlays(
    out_dir: Path,
    stem: str,
    keys: list[tuple[int, ...]],
    arrays: dict[str, np.ndarray],
    align_by: str,
) -> list[Path]:
    plt = import_pyplot()
    x, xlabel = x_axis(keys, align_by)
    outputs: list[Path] = []
    dim = next(iter(arrays.values())).shape[1]

    for group_name, slc in obs_groups(dim):
        group_dim = slc.stop - slc.start
        suffixes = value_suffixes("obs", dim)[slc]
        fig, axes = plt.subplots(
            group_dim,
            1,
            figsize=(14, max(7, group_dim * 1.8)),
            sharex=True,
            constrained_layout=True,
        )
        if group_dim == 1:
            axes = [axes]
        for local_idx, ax in enumerate(axes):
            dim_idx = slc.start + local_idx
            for label, values in arrays.items():
                ax.plot(x, values[:, dim_idx], label=label, linewidth=1.35)
            ax.axhline(0.0, color="0.55", linewidth=0.7)
            ax.set_ylabel(suffixes[local_idx])
            ax.grid(True, alpha=0.3)
        axes[0].legend(loc="upper right", ncols=min(3, len(arrays)))
        axes[0].set_title(f"obs {group_name} aligned by {align_by} ({len(keys)} frames)")
        axes[-1].set_xlabel(xlabel)
        path = out_dir / f"{stem}_{group_name}_overlay.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(path)
    return outputs


def plot_obs_group_diffs(
    out_dir: Path,
    stem: str,
    keys: list[tuple[int, ...]],
    arrays: dict[str, np.ndarray],
    align_by: str,
) -> list[Path]:
    labels = list(arrays)
    base_label = labels[0]
    plt = import_pyplot()
    x, xlabel = x_axis(keys, align_by)
    outputs: list[Path] = []
    dim = arrays[base_label].shape[1]

    for group_name, slc in obs_groups(dim):
        group_dim = slc.stop - slc.start
        suffixes = value_suffixes("obs", dim)[slc]
        fig, axes = plt.subplots(
            group_dim,
            1,
            figsize=(14, max(7, group_dim * 1.8)),
            sharex=True,
            constrained_layout=True,
        )
        if group_dim == 1:
            axes = [axes]
        for local_idx, ax in enumerate(axes):
            dim_idx = slc.start + local_idx
            for label in labels[1:]:
                diff = arrays[label][:, dim_idx] - arrays[base_label][:, dim_idx]
                ax.plot(x, diff, label=f"{label} - {base_label}", linewidth=1.25)
            ax.axhline(0.0, color="0.35", linewidth=0.8)
            ax.set_ylabel(suffixes[local_idx])
            ax.grid(True, alpha=0.3)
        axes[0].legend(loc="upper right", ncols=min(2, len(labels) - 1))
        axes[0].set_title(f"obs {group_name} diff vs {base_label} ({len(keys)} frames)")
        axes[-1].set_xlabel(xlabel)
        path = out_dir / f"{stem}_{group_name}_diff_vs_sim.png"
        fig.savefig(path, dpi=180)
        plt.close(fig)
        outputs.append(path)
    return outputs


def format_array(values: np.ndarray) -> str:
    return "[" + ", ".join(f"{v:.6g}" for v in values) + "]"


def summary_lines(
    datasets: list[Dataset],
    keys: list[tuple[int, ...]],
    arrays: dict[str, np.ndarray],
    align_by: str,
    value: str,
    outputs: list[Path],
) -> list[str]:
    labels = list(arrays)
    base_label = labels[0]
    lines = [
        "Action log comparison",
        f"value={value}",
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
            f"unique_keys={len(ds.rows)}, duplicates={ds.duplicate_keys}, "
            f"cols={ds.action_cols}, path={ds.path}"
        )

    lines += ["", f"Diff stats vs {base_label}:"]
    for label in labels[1:]:
        diff = arrays[label] - arrays[base_label]
        mae = np.mean(np.abs(diff), axis=0)
        max_abs = np.max(np.abs(diff), axis=0)
        lines.append(f"- {label}: mean_abs={format_array(mae)}")
        lines.append(f"- {label}: max_abs={format_array(max_abs)}")

    if value == "obs" and next(iter(arrays.values())).shape[1] == OBS_DIM:
        lines += ["", "Obs dim labels:"]
        for index, name in enumerate(OBS_LABELS):
            lines.append(f"- obs_{index:02d}: {name}")

    lines += ["", "Outputs:"]
    for path in outputs:
        lines.append(f"- {path}")
    return lines


def input_specs(args: argparse.Namespace) -> list[tuple[str, Path]]:
    if not args.input:
        return [
            ("isaac_sim", args.sim_csv),
            ("xarm7_sim_norender", args.real_norender_csv),
            ("xarm7_sim", args.real_csv),
        ]

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


def main() -> None:
    args = parse_args()
    specs = input_specs(args)
    datasets = [
        load_dataset(label, path, args)
        for label, path in specs
    ]
    keys = aligned_keys(datasets, args.max_frames)
    arrays = {
        ds.label: np.stack([ds.rows[key] for key in keys], axis=0)
        for ds in datasets
    }

    out_dir = output_dir(args, specs)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{args.value}_{args.align_by}"
    overlay_png = out_dir / f"{stem}_overlay.png"
    diff_png = out_dir / f"{stem}_diff_vs_sim.png"
    aligned_csv = out_dir / f"{stem}_aligned.csv"
    summary_txt = out_dir / f"{stem}_summary.txt"

    plot_overlay(overlay_png, keys, arrays, args.align_by, args.value)
    plot_diff(diff_png, keys, arrays, args.align_by, args.value)
    write_aligned_csv(aligned_csv, keys, arrays, args.align_by, args.value)

    outputs = [overlay_png, diff_png, aligned_csv, summary_txt]
    if args.value == "obs":
        outputs += plot_obs_group_overlays(out_dir, stem, keys, arrays, args.align_by)
        outputs += plot_obs_group_diffs(out_dir, stem, keys, arrays, args.align_by)

    lines = summary_lines(
        datasets=datasets,
        keys=keys,
        arrays=arrays,
        align_by=args.align_by,
        value=args.value,
        outputs=outputs,
    )
    summary_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
