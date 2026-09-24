#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""两版 reach 评估对比（外层调度）。

依次用 isaaclab.sh 启动 eval_reach_single.py 评估两个版本（各自独立 Isaac 进程，
避免 782/398 网络与 env_cfg 冲突），收集两份 JSON，打印对比表并存 CSV。

口径：
    - 碰撞即终止（collision + time_out 终止保留，reach_success 关闭）。
    - 成功 = episode 结束前【曾进入过】阈值（三档：1cm/3°、0.5cm/3°、纯位置1cm）。
    - 碰撞 = 该 episode 因碰撞 terminated。
    - 成功与碰撞独立统计。

用法（用普通 python 或 isaaclab.sh 都行，本脚本只做 subprocess 调度）：
    python scripts/train/eval_reach_compare.py \
        --num_envs 200 --total_episodes 1000

两个 checkpoint 已写死在下方 VERSIONS，可用 --vision_ckpt / --nopc_ckpt 覆盖。
"""

import argparse
import csv
import json
import os
import subprocess
import sys
import tempfile

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))

ISAACLAB_SH = "/home/gxai/IsaacLab/isaaclab.sh"
WORKER = os.path.join(_CURRENT_DIR, "eval_reach_single.py")

# 默认评估的两个 checkpoint
DEFAULT_VISION_CKPT = os.path.join(
    _PROJECT_DIR, "logs", "xarm7_pick_liftcube_vision",
    "2026-06-16_22-06-03", "model_19999.pt",
)
DEFAULT_NOPC_CKPT = os.path.join(
    _PROJECT_DIR, "logs", "xarm7_pick_liftcube_vision_nopc",
    "2026-06-17_16-13-38", "model_11300.pt",
)


def run_worker(version: str, checkpoint: str, num_envs: int,
               total_episodes: int, seed: int, out_json: str,
               fix_table_height: bool = False, reset_grace_steps: int = 2) -> dict:
    cmd = [
        ISAACLAB_SH, "-p", WORKER,
        "--version", version,
        "--checkpoint", checkpoint,
        "--num_envs", str(num_envs),
        "--total_episodes", str(total_episodes),
        "--seed", str(seed),
        "--reset_grace_steps", str(reset_grace_steps),
        "--out", out_json,
    ]
    if fix_table_height:
        cmd.append("--fix_table_height")
    print(f"\n{'='*70}\n[compare] 评估 {version}: {checkpoint}\n{'='*70}", flush=True)
    print("[compare] CMD:", " ".join(cmd), flush=True)
    ret = subprocess.run(cmd, cwd=_PROJECT_DIR)
    if ret.returncode != 0:
        print(f"[compare][WARN] {version} worker 退出码非 0: {ret.returncode}", flush=True)
    if not os.path.exists(out_json):
        print(f"[compare][ERROR] {version} 未产出结果 JSON: {out_json}", flush=True)
        return None
    with open(out_json, "r", encoding="utf-8") as f:
        return json.load(f)


def _fmt_row(label, vis, nop):
    def g(d, *keys):
        if d is None:
            return None
        cur = d
        for k in keys:
            cur = cur.get(k, {}) if isinstance(cur, dict) else None
            if cur is None:
                return None
        return cur
    return label, vis, nop


def print_table(res_vision: dict, res_nopc: dict):
    def rate(d, key):
        if d is None:
            return "  N/A  "
        return f"{d['rates'][key]*100:6.2f}%"

    def cnt(d, key):
        if d is None:
            return "N/A"
        return str(d["counts"][key])

    def epi(d):
        return "N/A" if d is None else str(d["episodes_completed"])

    def meanv(d, key):
        return "N/A" if d is None else f"{d[key]:.3f}"

    print("\n" + "=" * 78)
    print(" reach 评估对比  (成功=结束前曾进入阈值；碰撞=因碰撞terminated)")
    print("=" * 78)
    print(f"{'指标':<28}{'vision(782/点云)':>24}{'nopc(398/无点云)':>24}")
    print("-" * 78)
    def discarded(d):
        return "N/A" if d is None else str(d.get("discarded_early_collision", 0))

    print(f"{'有效 episodes':<28}{epi(res_vision):>24}{epi(res_nopc):>24}")
    print(f"{'丢弃(开局碰撞无效局)':<28}{discarded(res_vision):>24}{discarded(res_nopc):>24}")
    print("-" * 78)
    for label, key in [
        ("成功率 1cm/3°", "reach_1cm_3deg"),
        ("成功率 0.5cm/3°", "reach_0p5cm_3deg"),
        ("成功率 纯位置1cm", "reach_pos_1cm"),
        ("碰撞率", "collision"),
        ("超时率", "timeout"),
    ]:
        v = f"{rate(res_vision, key)} ({cnt(res_vision, key)})"
        n = f"{rate(res_nopc, key)} ({cnt(res_nopc, key)})"
        print(f"{label:<28}{v:>24}{n:>24}")
    print("-" * 78)
    print(f"{'平均最优 dist(cm)':<28}"
          f"{meanv(res_vision,'mean_best_dist_cm'):>24}"
          f"{meanv(res_nopc,'mean_best_dist_cm'):>24}")
    print(f"{'平均最优 ori(deg)':<28}"
          f"{meanv(res_vision,'mean_best_ori_deg'):>24}"
          f"{meanv(res_nopc,'mean_best_ori_deg'):>24}")
    print("=" * 78 + "\n")


def save_csv(path, res_vision, res_nopc):
    rows = []
    for ver, d in [("vision", res_vision), ("nopc", res_nopc)]:
        if d is None:
            continue
        r = {"version": ver, "checkpoint": d["checkpoint"],
             "episodes": d["episodes_completed"],
             "mean_best_dist_cm": d["mean_best_dist_cm"],
             "mean_best_ori_deg": d["mean_best_ori_deg"]}
        for k, v in d["counts"].items():
            r[f"count_{k}"] = v
        for k, v in d["rates"].items():
            r[f"rate_{k}"] = v
        rows.append(r)
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[compare] CSV 已保存: {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="两版 reach 评估对比（外层调度）")
    parser.add_argument("--num_envs", type=int, default=200)
    parser.add_argument("--total_episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--vision_ckpt", type=str, default=DEFAULT_VISION_CKPT)
    parser.add_argument("--nopc_ckpt", type=str, default=DEFAULT_NOPC_CKPT)
    parser.add_argument("--only", choices=["vision", "nopc", "both"], default="both")
    parser.add_argument("--fix_table_height", action="store_true",
                        help="固定桌面高度，不做高度随机化（两版都生效）。")
    parser.add_argument("--reset_grace_steps", type=int, default=2,
                        help="开局宽限步数：<=该步数内碰撞结束的 episode 判为无效局丢弃。"
                             "设 0 关闭。")
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(_PROJECT_DIR, "logs", "eval_reach"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="eval_reach_")

    res_vision = res_nopc = None

    if args.only in ("vision", "both"):
        if os.path.exists(args.vision_ckpt):
            res_vision = run_worker("vision", args.vision_ckpt, args.num_envs,
                                    args.total_episodes, args.seed,
                                    os.path.join(tmp, "vision.json"),
                                    fix_table_height=args.fix_table_height,
                                    reset_grace_steps=args.reset_grace_steps)
        else:
            print(f"[compare][ERROR] vision checkpoint 不存在: {args.vision_ckpt}")

    if args.only in ("nopc", "both"):
        if os.path.exists(args.nopc_ckpt):
            res_nopc = run_worker("nopc", args.nopc_ckpt, args.num_envs,
                                  args.total_episodes, args.seed,
                                  os.path.join(tmp, "nopc.json"),
                                  fix_table_height=args.fix_table_height,
                                  reset_grace_steps=args.reset_grace_steps)
        else:
            print(f"[compare][ERROR] nopc checkpoint 不存在: {args.nopc_ckpt}")

    print_table(res_vision, res_nopc)
    save_csv(os.path.join(args.out_dir, "eval_reach_compare.csv"), res_vision, res_nopc)


if __name__ == "__main__":
    main()
