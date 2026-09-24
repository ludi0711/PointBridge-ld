#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""点云遮挡任务 两版对比评估（外层调度）。

依次用 isaaclab.sh 启动 eval_pointcloud_worker.py 评两份 checkpoint：
    A. 遮挡训练版（默认 random_box 25% 训练出的模型）
    B. 无遮挡 baseline（stage2_occ_none）
两版走【完全相同】的评估参数：同 seed、同 num_envs、同遮挡注入（25% random_box）、
同抓取目标偏移 —— 这是 seed 对齐的基础。

seed 对齐的两个层级（详见 worker docstring）：
    1) 默认（非 lockstep）：两版消费同一条全局 RNG 流（同 seed + 同 env 数 + 同遮挡
       参数 + 观测噪声关闭），初始工件位姿与遮挡位置来自【同一随机序列】。但碰撞
       终止时刻策略相关，reset 节奏会逐步错位 → 只能保证"同一分布"。
    2) --lockstep（默认开）：关闭碰撞终止，所有 episode 跑到 1200 步超时，两版
       reset 时刻完全一致 → 逐集同一初始位姿 + 同一遮挡位置，严格配对。
       此时碰撞率改用 ever_collided（任意帧接触力超阈）。
       stats 脚本会按存下的 obj_init_pos 校验对齐情况，自适应选择配对/非配对检验。

用法（普通 python 即可，本脚本只做 subprocess 调度）::

    python tools/eval_pointcloud_compare.py \
        --num_envs 50 --total_episodes 1000 --seed 0

换遮挡参数（后续圆心遮挡等）同样在这里选::

    python tools/eval_pointcloud_compare.py \
        --occlusion_mode random_sphere --occlusion_severity 0.25 \
        --occ_ckpt <对应训练checkpoint> --num_envs 50

交互式选择 checkpoint（带 -i，其余参数照旧走命令行，解析后逐个选两个
checkpoint，并复核/微调全部参数，避免每次手打完整路径）::

    python tools/eval_pointcloud_compare.py -i --num_envs 50 --seed 0

默认 checkpoint：
    occ:  logs/xarm7_pick_pointcloud_stage2_occ_random_box_25/2026-09-01_15-34-14/model_6600.pt
    base: logs/xarm7_pick_pointcloud_stage2_occ_none/2026-08-27_23-11-05/model_7200.pt
"""

import argparse
import glob
import os
import re
import subprocess
import sys
import tempfile
from datetime import datetime

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_CURRENT_DIR)

ISAACLAB_SH_DEFAULT = "/root/IsaacLab/isaaclab.sh"
WORKER = os.path.join(_CURRENT_DIR, "eval_pointcloud_worker.py")
STATS = os.path.join(_CURRENT_DIR, "stats_pointcloud_occlusion.py")

DEFAULT_OCC_CKPT = (
    "/root/autodl-tmp/logs/xarm7_pick_pointcloud_stage2_occ_random_box_30"
    "/2026-09-06_10-24-58/model_5100.pt"
)
DEFAULT_BASE_CKPT = (
    "/root/autodl-tmp/logs/xarm7_pick_pointcloud_stage2_occ_none"
    "/2026-08-27_23-11-05/model_7200.pt"
)

_OCC_CHOICES = ("none", "halfspace", "sphere", "random_sphere", "random_box")

# 交互选 checkpoint 时扫描的日志根目录（结构 run/<时间戳>/model_<step>.pt）
_LOG_ROOTS = [
    "/root/autodl-tmp/logs",
    os.path.join(_PROJECT_DIR, "logs"),
]

_BOOL_MAP = {"true": True, "1": True, "yes": True, "false": False, "0": False, "no": False}


def _iter_checkpoints(log_roots):
    """扫描日志根目录，返回 [(run_name, timestamp, step, full_path), ...]，按序排列。"""
    out = []
    seen = set()
    for root in log_roots:
        root = os.path.expanduser(root)
        if not os.path.isdir(root):
            continue
        for run_name in sorted(os.listdir(root)):
            run_dir = os.path.join(root, run_name)
            if not os.path.isdir(run_dir):
                continue
            for ts_name in sorted(os.listdir(run_dir)):
                ts_dir = os.path.join(run_dir, ts_name)
                if not os.path.isdir(ts_dir):
                    continue
                for fn in sorted(os.listdir(ts_dir)):
                    m = re.match(r"model_(\d+)\.pt$", fn)
                    if not m:
                        continue
                    step = int(m.group(1))
                    key = (run_name, ts_name, step)
                    if key in seen:
                        continue
                    seen.add(key)
                    out.append((run_name, ts_name, step, os.path.join(ts_dir, fn)))
    out.sort(key=lambda e: (e[0], e[1], e[2]))
    return out


def _run_of(path):
    """从 checkpoint 全路径提取 run 名（倒数第二级目录）。"""
    return os.path.basename(os.path.dirname(os.path.dirname(path)))


def _select_checkpoint(label, log_roots, default_path=None):
    """交互式逐级选择 checkpoint：run → 时间戳 → model 文件。

    每步都支持：回车=默认值、粘贴完整路径(以 / 开头)、输入编号。
    """
    print(f"\n{'─' * 70}")
    print(f"[交互] 选择 {label} 的 checkpoint")
    print(f"  日志目录: {', '.join(log_roots)}")
    print(f"  默认值:   {default_path or '(无)'}")
    print("  规则: 输入编号逐步选择；直接回车=默认；任意一步粘贴完整路径(/开头)直接采用")
    print("─" * 70)

    entries = _iter_checkpoints(log_roots)
    if not entries:
        raw = input(f"[交互] 日志目录下未发现任何 model_*.pt，请直接输入 {label} 的 .pt 路径: ").strip()
        return raw or default_path

    default_run = _run_of(default_path) if default_path else None

    # ── Level 1: 选 run ──
    run_names = sorted({e[0] for e in entries})
    while True:
        print(f"\n  {label} · 第 1 步 / 3：选择 run")
        for i, r in enumerate(run_names, 1):
            n = sum(1 for e in entries if e[0] == r)
            mark = "  ★默认" if r == default_run else ""
            print(f"    {i:>2}. {r}  ({n} 个 checkpoint){mark}")
        raw = input(f"  → 编号(1..{len(run_names)}) / 回车=默认 / 完整路径: ").strip()
        if raw == "":
            if default_path and os.path.isfile(default_path):
                return default_path
            print("    (无默认值) 请输入编号或完整路径。")
            continue
        if raw.startswith("/"):
            return raw
        if raw.isdigit() and 1 <= int(raw) <= len(run_names):
            chosen_run = run_names[int(raw) - 1]
            break
        print(f"    无效输入，请输入 1..{len(run_names)} 或完整路径。")

    # ── Level 2: 选时间戳 ──
    ts_names = sorted({e[1] for e in entries if e[0] == chosen_run})
    while True:
        print(f"\n  {label} · 第 2 步 / 3：选择 {chosen_run} 的时间戳")
        for i, t in enumerate(ts_names, 1):
            n = sum(1 for e in entries if e[0] == chosen_run and e[1] == t)
            print(f"    {i:>2}. {t}  ({n} 个 checkpoint)")
        raw = input(f"  → 编号(1..{len(ts_names)}) / 回车=默认 / 完整路径: ").strip()
        if raw == "":
            if default_path and os.path.isfile(default_path):
                return default_path
            print("    (无默认值) 请输入编号或完整路径。")
            continue
        if raw.startswith("/"):
            return raw
        if raw.isdigit() and 1 <= int(raw) <= len(ts_names):
            chosen_ts = ts_names[int(raw) - 1]
            break
        print(f"    无效输入，请输入 1..{len(ts_names)} 或完整路径。")

    # ── Level 3: 选 model 文件（按 step 编号，网格展示避免过长）──
    models = sorted((e for e in entries if e[0] == chosen_run and e[1] == chosen_ts),
                    key=lambda e: e[2])
    steps = [step for _, _, step, _ in models]
    while True:
        print(f"\n  {label} · 第 3 步 / 3：选择 {chosen_run}/{chosen_ts} 的模型"
              f"（共 {len(models)} 个，按 step 排序）")
        for i in range(0, len(steps), 8):
            print("    " + "  ".join(f"model_{s}" for s in steps[i:i + 8]))
        raw = input(f"  → 输入步数(如 5100) / 回车=默认 / 完整路径: ").strip()
        if raw == "":
            if default_path and os.path.isfile(default_path):
                return default_path
            print("    (无默认值) 请输入步数或完整路径。")
            continue
        if raw.startswith("/"):
            return raw
        if raw.isdigit():
            n = int(raw)
            for _, _, step, path in models:
                if step == n:
                    return path
            print(f"    找不到 model_{n}.pt（可用步数见上表）。")
            continue
        print("    无效输入。")

    return default_path  # 理论不可达


def _review_and_maybe_edit(args, occ_ckpt, base_ckpt):
    """交互模式最后一步：打印完整参数清单，允许 参数=值 微调；回车即开始。"""
    editable = {
        "num_envs": int,
        "total_episodes": int,
        "seed": int,
        "occlusion_mode": str,
        "occlusion_severity": float,
        "occlusion_axis": str,
        "occlusion_center": str,
        "grasp_offset_cm": float,
        "lift_height": float,
        "collision_force_thresh": float,
        "reset_grace_steps": int,
    }
    while True:
        lock = "lockstep(严格配对)" if args.lockstep else "分布对齐(碰撞终止)"
        print(f"\n{'─' * 70}\n[交互] 参数复核（回车直接开始）")
        print(f"  协议:     num_envs={args.num_envs}  episodes={args.total_episodes}  "
              f"seed={args.seed}  {lock}")
        print(f"  遮挡注入: mode={args.occlusion_mode}  severity={args.occlusion_severity:.2f}  "
              f"axis={args.occlusion_axis}  center={args.occlusion_center}")
        print(f"  任务口径: grasp_offset_cm={args.grasp_offset_cm}  "
              f"lift_height={args.lift_height}  collision_thresh={args.collision_force_thresh}")
        if args.only in ("occ", "both"):
            print(f"  A(遮挡训练): {occ_ckpt}")
        if args.only in ("base", "both"):
            print(f"  B(baseline):  {base_ckpt}")
        print(f"  可改参数: {', '.join(editable)}；lockstep 用 true/false")
        raw = input(f"{'─' * 70}\n  回车开始 / q 退出 / 参数=值 修改: ").strip()
        if raw == "":
            return
        if raw.lower() in ("q", "quit", "exit"):
            raise SystemExit("[交互] 已取消，未启动评估。")
        if "=" in raw:
            k, v = (x.strip() for x in raw.split("=", 1))
            if k == "lockstep":
                if v.lower() not in _BOOL_MAP:
                    print(f"  [交互] lockstep 取值应为 true/false，got {v}")
                else:
                    args.lockstep = _BOOL_MAP[v.lower()]
                continue
            if k not in editable:
                print(f"  [交互] 未知参数 {k}；可改: {', '.join(editable)}, lockstep")
                continue
            if k == "occlusion_mode" and v not in _OCC_CHOICES:
                print(f"  [交互] occlusion_mode 取值应在 {_OCC_CHOICES}")
                continue
            try:
                setattr(args, k, editable[k](v))
                print(f"  [交互] 已更新 {k} = {getattr(args, k)}")
            except ValueError as ex:
                print(f"  [交互] 值非法 {v} ({ex})")
            continue
        print("  [交互] 无法识别；请输入 回车 / q / 参数=值。")


def _occ_tag(mode, severity, axis):
    """与训练脚本一致的遮挡标签（none 时返回 none）。"""
    if mode == "none" or severity <= 0.0:
        return "none"
    if mode == "halfspace":
        return f"halfspace{axis}_{int(round(severity * 100))}"
    return f"{mode}_{int(round(severity * 100))}"


def run_worker(label, checkpoint, out_csv, args):
    """用 isaaclab.sh 启动一次 worker，跑满 total_episodes，逐集 CSV 落盘。"""
    cmd = [
        args.isaaclab_sh, "-p", WORKER,
        "--checkpoint", checkpoint,
        "--tag", label,
        "--occlusion_mode", args.occlusion_mode,
        "--occlusion_severity", str(args.occlusion_severity),
        "--occlusion_axis", args.occlusion_axis,
        "--occlusion_center", args.occlusion_center,
        "--grasp_offset_cm", str(args.grasp_offset_cm),
        "--lift_height", str(args.lift_height),
        "--collision_force_thresh", str(args.collision_force_thresh),
        "--num_envs", str(args.num_envs),
        "--total_episodes", str(args.total_episodes),
        "--seed", str(args.seed),
        "--reset_grace_steps", str(args.reset_grace_steps),
        "--out", out_csv,
    ]
    if args.lockstep:
        cmd.append("--lockstep")
    print(f"\n{'=' * 78}\n[compare] 评估 {label}: {checkpoint}\n{'=' * 78}", flush=True)
    print("[compare] CMD:", " ".join(cmd), flush=True)
    # isaaclab.sh 用 CONDA_PREFIX 定位 python（本机是 /root/gx-va 这个 isaaclab 环境）
    env = dict(os.environ)
    env["CONDA_PREFIX"] = args.isaaclab_conda_env
    ret = subprocess.run(cmd, cwd=_PROJECT_DIR, env=env)
    if ret.returncode != 0:
        print(f"[compare][WARN] {label} worker 退出码非 0: {ret.returncode}", flush=True)
    if not os.path.exists(out_csv):
        print(f"[compare][ERROR] {label} 未产出 CSV: {out_csv}", flush=True)
        return None
    n_rows = sum(1 for _ in open(out_csv, encoding="utf-8")) - 1
    print(f"[compare] {label} 落盘 {n_rows} 个 episode: {out_csv}", flush=True)
    return out_csv


def main():
    parser = argparse.ArgumentParser(description="点云遮挡两版对比评估（外层调度）")
    parser.add_argument("-i", "--interactive", action="store_true",
                        help="交互模式：其余参数照常走命令行，解析后逐级选择两个 checkpoint，"
                             "并复核/微调全部参数后再启动")
    parser.add_argument("--logs_dir", nargs="+", default=None,
                        help="交互扫描的日志根目录（默认 /root/autodl-tmp/logs 与项目 logs/）")
    # ── 评估协议 ──
    parser.add_argument("--num_envs", type=int, default=50,
                        help="并行环境数（两版一致，RNG 流才同构）")
    parser.add_argument("--total_episodes", type=int, default=1000,
                        help="每版评估的 episode 数")
    parser.add_argument("--seed", type=int, default=0,
                        help="随机种子（两版一致）")
    parser.add_argument("--lockstep", action=argparse.BooleanOptionalAction, default=True,
                        help="关闭碰撞终止（全部超时 1200 步）→ 两版严格逐集配对。"
                             "默认开；--no-lockstep 退回分布对齐（碰撞终止保留）")
    parser.add_argument("--reset_grace_steps", type=int, default=0,
                        help="开局宽限步数：<=该步数内碰撞终止的 episode 判为无效局丢弃。"
                             "配对比较必须为 0（默认）")
    # ── 遮挡注入（两版一致，与训练该 checkpoint 时一致）──
    parser.add_argument("--occlusion_mode", type=str, default="random_box",
                        choices=_OCC_CHOICES)
    parser.add_argument("--occlusion_severity", type=float, default=0.25)
    parser.add_argument("--occlusion_axis", type=str, default="x")
    parser.add_argument("--occlusion_center", type=str, default="0.5,0.5,0.5")
    # ── 任务口径 ──
    parser.add_argument("--grasp_offset_cm", type=float, default=3.0)
    parser.add_argument("--lift_height", type=float, default=0.73)
    parser.add_argument("--collision_force_thresh", type=float, default=0.5)
    # ── 两个 checkpoint 与调度 ──
    parser.add_argument("--occ_ckpt", type=str, default=None,
                        help="遮挡训练版 checkpoint（不传且带 -i 时交互选择；否则用默认）")
    parser.add_argument("--base_ckpt", type=str, default=None,
                        help="无遮挡 baseline checkpoint（不传且带 -i 时交互选择；否则用默认）")
    parser.add_argument("--label_occ", type=str, default="occluded(25% random_box)",
                        help="统计报表里 A 版的显示名")
    parser.add_argument("--label_base", type=str, default="baseline(无遮挡)",
                        help="统计报表里 B 版的显示名")
    parser.add_argument("--only", choices=["occ", "base", "both"], default="both",
                        help="只跑单版（单版统计）还是两版对比")
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(_PROJECT_DIR, "logs", "eval_pointcloud"))
    parser.add_argument("--isaaclab_sh", type=str, default=ISAACLAB_SH_DEFAULT,
                        help="isaaclab.sh 路径")
    parser.add_argument("--isaaclab_conda_env", type=str, default="/root/gx-va",
                        help="isaaclab 所在 conda 环境路径（isaaclab.sh 靠 CONDA_PREFIX 选 python）")
    parser.add_argument("--no_stats", action="store_true",
                        help="跑完后不自动调 stats 脚本出报表/图")
    args = parser.parse_args()

    if not os.path.isfile(args.isaaclab_sh):
        raise SystemExit(f"[compare][ERROR] isaaclab.sh 不存在: {args.isaaclab_sh} "
                         f"（可用 --isaaclab_sh 指定）")
    if not os.path.isfile(os.path.join(args.isaaclab_conda_env, "bin", "python")):
        raise SystemExit(f"[compare][ERROR] conda 环境无 python: {args.isaaclab_conda_env} "
                         f"（可用 --isaaclab_conda_env 指定 isaaclab 所在环境）")

    # ── 解析两个 checkpoint：交互模式逐个选择，否则用命令行/默认 ──
    occ_ckpt = args.occ_ckpt or DEFAULT_OCC_CKPT
    base_ckpt = args.base_ckpt or DEFAULT_BASE_CKPT
    if args.interactive:
        log_roots = args.logs_dir or _LOG_ROOTS
        if args.only in ("occ", "both") and not args.occ_ckpt:
            occ_ckpt = _select_checkpoint("A(遮挡训练版)", log_roots, DEFAULT_OCC_CKPT)
        if args.only in ("base", "both") and not args.base_ckpt:
            base_ckpt = _select_checkpoint("B(baseline)", log_roots, DEFAULT_BASE_CKPT)
        _review_and_maybe_edit(args, occ_ckpt, base_ckpt)

    # ── 参数校验（交互模式可能在复核时改了值，故放在交互之后）──
    if not (0.0 <= args.occlusion_severity <= 1.0):
        raise SystemExit(
            f"[compare][ERROR] --occlusion_severity 必须在 [0,1]，"
            f"got {args.occlusion_severity}"
        )
    if args.reset_grace_steps > 0 and args.lockstep:
        raise SystemExit(
            "[compare][ERROR] --lockstep 配对比较要求 --reset_grace_steps 0："
            "开局过滤会因策略不同丢弃不同 episode，破坏逐集对应。"
        )

    os.makedirs(args.out_dir, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    prefix = f"stage2_occ_{_occ_tag(args.occlusion_mode, args.occlusion_severity, args.occlusion_axis)}_seed{args.seed}_{stamp}"
    csv_occ = os.path.join(args.out_dir, f"{prefix}_occtrained.csv")
    csv_base = os.path.join(args.out_dir, f"{prefix}_baseline.csv")

    print("=" * 78)
    print("[compare] 两版点云遮挡对比评估（seed 对齐）")
    print(f"[compare] 遮挡注入: mode={args.occlusion_mode} "
          f"severity={args.occlusion_severity:.2f}（两版一致）")
    print(f"[compare] 协议: num_envs={args.num_envs} "
          f"episodes={args.total_episodes} seed={args.seed}")
    print(f"[compare] 终止: {'lockstep（碰撞终止关闭→严格配对）' if args.lockstep else '碰撞即终止（分布对齐）'}")
    print(f"[compare] A(遮挡训练): {occ_ckpt}")
    print(f"[compare] B(baseline):  {base_ckpt}")
    print(f"[compare] 输出目录: {args.out_dir}")
    print("=" * 78, flush=True)

    res_occ = res_base = None

    if args.only in ("occ", "both"):
        if os.path.isfile(occ_ckpt):
            res_occ = run_worker("occ25", occ_ckpt, csv_occ, args)
        else:
            print(f"[compare][ERROR] 遮挡版 checkpoint 不存在: {occ_ckpt}")

    if args.only in ("base", "both"):
        if os.path.isfile(base_ckpt):
            res_base = run_worker("baseline", base_ckpt, csv_base, args)
        else:
            print(f"[compare][ERROR] baseline checkpoint 不存在: {base_ckpt}")

    # ── 自动出统计报表 + 图 ─────────────────────────────────────────────────
    produced = [p for p in (res_occ, res_base) if p is not None]
    if produced and not args.no_stats:
        stats_out = os.path.join(args.out_dir, f"{prefix}_stats")
        cmd = [sys.executable, STATS,
               "--csv_a", produced[0], "--label_a", args.label_occ,
               "--out_dir", stats_out]
        if len(produced) == 2 and args.only == "both":
            cmd += ["--csv_b", produced[1], "--label_b", args.label_base]
        print(f"\n[compare] 自动出统计报表: {' '.join(cmd)}", flush=True)
        ret = subprocess.run(cmd, cwd=_PROJECT_DIR)
        if ret.returncode != 0:
            print(f"[compare][WARN] stats 脚本退出码非 0: {ret.returncode}，"
                  f"可手动重跑上述命令", flush=True)
        else:
            print(f"[compare] 报表已出: {stats_out}", flush=True)

    print("\n[compare] 完成。原始样本 CSV：")
    for p in produced:
        print(f"  - {p}")
    if args.only == "both" and not produced:
        print("[compare][ERROR] 两版均未产出结果。")
        sys.exit(1)


if __name__ == "__main__":
    main()
