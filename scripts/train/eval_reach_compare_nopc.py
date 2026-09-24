#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""两个 nopc(398维无点云) checkpoint 的 reach 评估对比（新 vs 旧）。

依次用 isaaclab.sh 启动 eval_reach_single.py（两次都用 --version nopc，只是
checkpoint 不同），收集两份 JSON，打印对比表并存 CSV。

口径（与 eval_reach_single 一致）：
    - 碰撞即终止（collision + time_out 终止保留，reach_success 关闭）。
    - 成功 = episode 结束前【曾进入过】阈值（三档：1cm/3°、0.5cm/3°、纯位置1cm）。
    - 碰撞 = 该 episode 因碰撞 terminated。
    - 可选 --fix_table_height 固定桌面高度；--reset_grace_steps 丢弃开局碰撞无效局。

用法：
    python scripts/train/eval_reach_compare_nopc.py \
        --num_envs 200 --total_episodes 1000 --fix_table_height
"""

import argparse
import csv
import json
import os
import subprocess
import tempfile

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))

ISAACLAB_SH = "/home/gxai/IsaacLab/isaaclab.sh"
WORKER = os.path.join(_CURRENT_DIR, "eval_reach_single.py")

# 默认对比：新 nopc vs 旧 nopc
DEFAULT_NEW_CKPT = os.path.join(
    _PROJECT_DIR, "logs", "xarm7_pick_liftcube_vision_nopc",
    "2026-06-19_17-19-44", "model_7400.pt",
)
DEFAULT_OLD_CKPT = os.path.join(
    _PROJECT_DIR, "logs", "xarm7_pick_liftcube_vision_nopc",
    "2026-06-17_16-13-38", "model_11300.pt",
)


def run_worker(label, checkpoint, num_envs, total_episodes, seed, out_json,
               fix_table_height=False, reset_grace_steps=2):
    """两个槽都按 nopc 加载（398维网络），只是 checkpoint / 输出不同。"""
    cmd = [
        ISAACLAB_SH, "-p", WORKER,
        "--version", "nopc",
        "--checkpoint", checkpoint,
        "--num_envs", str(num_envs),
        "--total_episodes", str(total_episodes),
        "--seed", str(seed),
        "--reset_grace_steps", str(reset_grace_steps),
        "--out", out_json,
    ]
    if fix_table_height:
        cmd.append("--fix_table_height")
    print(f"\n{'='*70}\n[compare-nopc] 评估 [{label}]: {checkpoint}\n{'='*70}", flush=True)
    print("[compare-nopc] CMD:", " ".join(cmd), flush=True)
    ret = subprocess.run(cmd, cwd=_PROJECT_DIR)
    if ret.returncode != 0:
        print(f"[compare-nopc][WARN] [{label}] worker 退出码非 0: {ret.returncode}", flush=True)
    if not os.path.exists(out_json):
        print(f"[compare-nopc][ERROR] [{label}] 未产出结果 JSON: {out_json}", flush=True)
        return None
    with open(out_json, "r", encoding="utf-8") as f:
        d = json.load(f)
    d["_label"] = label
    return d


def print_table(res_new, res_old, new_ckpt, old_ckpt):
    def rate(d, key):
        return "  N/A  " if d is None else f"{d['rates'][key]*100:6.2f}%"

    def cnt(d, key):
        return "N/A" if d is None else str(d["counts"][key])

    def epi(d):
        return "N/A" if d is None else str(d["episodes_completed"])

    def disc(d):
        return "N/A" if d is None else str(d.get("discarded_early_collision", 0))

    def meanv(d, key):
        return "N/A" if d is None else f"{d[key]:.3f}"

    print("\n" + "=" * 86)
    print(" nopc reach 评估对比  (成功=结束前曾进入阈值；碰撞=因碰撞terminated)")
    print("=" * 86)
    print(f"  NEW : {new_ckpt}")
    print(f"  OLD : {old_ckpt}")
    print("-" * 86)
    print(f"{'指标':<28}{'NEW (新)':>28}{'OLD (旧)':>28}")
    print("-" * 86)
    print(f"{'有效 episodes':<28}{epi(res_new):>28}{epi(res_old):>28}")
    print(f"{'丢弃(开局碰撞无效局)':<28}{disc(res_new):>28}{disc(res_old):>28}")
    print("-" * 86)
    for label, key in [
        ("成功率 1cm/3°", "reach_1cm_3deg"),
        ("成功率 0.5cm/3°", "reach_0p5cm_3deg"),
        ("成功率 纯位置1cm", "reach_pos_1cm"),
        ("碰撞率", "collision"),
        ("超时率", "timeout"),
    ]:
        v = f"{rate(res_new, key)} ({cnt(res_new, key)})"
        o = f"{rate(res_old, key)} ({cnt(res_old, key)})"
        print(f"{label:<28}{v:>28}{o:>28}")
    print("-" * 86)
    print(f"{'平均最优 dist(cm)':<28}"
          f"{meanv(res_new,'mean_best_dist_cm'):>28}"
          f"{meanv(res_old,'mean_best_dist_cm'):>28}")
    print(f"{'平均最优 ori(deg)':<28}"
          f"{meanv(res_new,'mean_best_ori_deg'):>28}"
          f"{meanv(res_old,'mean_best_ori_deg'):>28}")
    print("=" * 86 + "\n")


def save_csv(path, res_new, res_old):
    rows = []
    for tag, d in [("new", res_new), ("old", res_old)]:
        if d is None:
            continue
        r = {"slot": tag, "checkpoint": d["checkpoint"],
             "episodes": d["episodes_completed"],
             "discarded_early_collision": d.get("discarded_early_collision", 0),
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
    print(f"[compare-nopc] CSV 已保存: {path}", flush=True)


def main():
    parser = argparse.ArgumentParser(description="两个 nopc checkpoint reach 评估对比")
    parser.add_argument("--num_envs", type=int, default=200)
    parser.add_argument("--total_episodes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--new_ckpt", type=str, default=DEFAULT_NEW_CKPT)
    parser.add_argument("--old_ckpt", type=str, default=DEFAULT_OLD_CKPT)
    parser.add_argument("--fix_table_height", action="store_true",
                        help="固定桌面高度，不做高度随机化（两个 checkpoint 都生效）。")
    parser.add_argument("--reset_grace_steps", type=int, default=2,
                        help="开局宽限步数：<=该步数内碰撞结束的 episode 判为无效局丢弃。设 0 关闭。")
    parser.add_argument("--out_dir", type=str,
                        default=os.path.join(_PROJECT_DIR, "logs", "eval_reach"))
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    tmp = tempfile.mkdtemp(prefix="eval_reach_nopc_")

    res_new = res_old = None
    if os.path.exists(args.new_ckpt):
        res_new = run_worker("NEW", args.new_ckpt, args.num_envs, args.total_episodes,
                             args.seed, os.path.join(tmp, "new.json"),
                             fix_table_height=args.fix_table_height,
                             reset_grace_steps=args.reset_grace_steps)
    else:
        print(f"[compare-nopc][ERROR] NEW checkpoint 不存在: {args.new_ckpt}")

    if os.path.exists(args.old_ckpt):
        res_old = run_worker("OLD", args.old_ckpt, args.num_envs, args.total_episodes,
                             args.seed, os.path.join(tmp, "old.json"),
                             fix_table_height=args.fix_table_height,
                             reset_grace_steps=args.reset_grace_steps)
    else:
        print(f"[compare-nopc][ERROR] OLD checkpoint 不存在: {args.old_ckpt}")

    print_table(res_new, res_old, args.new_ckpt, args.old_ckpt)
    save_csv(os.path.join(args.out_dir, "eval_reach_compare_nopc.csv"), res_new, res_old)


if __name__ == "__main__":
    main()
