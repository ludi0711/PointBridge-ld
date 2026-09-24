#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""点云遮挡评估统计（第 1 块单版统计 + 第 2 块 A/B 对比，配对/非配对自适应）。

输入：eval_pointcloud_worker.py 落盘的逐集 CSV（一版一份，或两份对比）。
输出（--out_dir 下）：
    report.txt   完整统计报表（第 1 块：单版指标 + Wilson 95% CI；第 2 块：Δ=A−B 检验）
    summary.csv  每个指标的均值/率、Δ、CI、p、检验方法
    stats.json   机器可读的同一份结果
    figs/*.png   成功率条形图（Wilson CI）、误差箱线图、误差 CDF、成功/失败散点

配对 vs 非配对（自适应，不用手工选）：
    两份 CSV 按 episode_index 对齐，比较每集的 obj_init_pos（工件初始位置）：
      - 对齐比例 ≥ 99%（--lockstep 跑法，两版逐集同一初始位姿+同一遮挡）→ 配对检验：
        二值指标 McNemar + 配对 bootstrap；连续指标配对 bootstrap（Δ 的 95% CI 与 p）。
      - 否则 → 非配对检验：两样本 bootstrap（Δ 的 CI 与 p）+ Cliff's delta 效应量。
    判定依据（含最大偏差）写入报表，供核对。

统计函数手写实现（环境无 scipy）：Wilson 二项区间、McNemar（χ²(1) 用 erfc 精确算）、
Cliff's delta、配对/非配对 bootstrap（固定种子，可复现）。

用法::

    python tools/stats_pointcloud_occlusion.py \
        --csv_a logs/eval_pointcloud/xxx_occtrained.csv --label_a "遮挡训练(25%)" \
        --csv_b logs/eval_pointcloud/xxx_baseline.csv  --label_b "baseline(无遮挡)" \
        --out_dir logs/eval_pointcloud/xxx_stats
"""

import argparse
import csv
import json
import math
import os
import sys

import numpy as np

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    _MPL = True
except Exception:
    _MPL = False

# ── dataviz 参考调色板（light，surface #fcfcfb）───────────────────────────────
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
BASELINE = "#c3c2b7"
A_COLOR = "#2a78d6"   # categorical slot 1（蓝）→ A 版（遮挡训练）
B_COLOR = "#eb6834"   # categorical slot 2（橙）→ B 版（baseline）
GOOD = "#0ca30c"      # status good
CRIT = "#d03b3b"      # status critical

_BOOT_SEED = 20260903
_N_BOOT = 4000

BINARY_METRICS = [
    ("reach_1cm_3deg", "reach 1cm/3°", "成功率 1cm/3°", True),
    ("reach_0p5cm_3deg", "reach 0.5cm/3°", "成功率 0.5cm/3°", True),
    ("reach_pos_1cm", "reach pos 1cm", "成功率 纯位置1cm", True),
    ("ever_lifted", "ever lifted", "曾举起率", True),
    ("collision", "collision终止", "碰撞终止率", False),
    ("timeout", "timeout终止", "超时率", False),
    ("ever_collided", "ever collided", "任意帧碰撞率(超阈)", False),
]
CONTINUOUS_METRICS = [
    ("dist_best_cm", "best dist (cm)", "最优距离 cm", False),
    ("ori_best_deg", "best ori (deg)", "最优姿态误差 deg", False),
    ("dist_final_cm", "final dist (cm)", "终止时距离 cm", False),
    ("ori_final_deg", "final ori (deg)", "终止时姿态误差 deg", False),
    ("ep_len", "episode len (steps)", "存活步数", None),
    ("reward_total", "total reward", "总奖励", True),
]
BINARY_KEYS = [m[0] for m in BINARY_METRICS]
CONTINUOUS_KEYS = [m[0] for m in CONTINUOUS_METRICS]

# ── 无 scipy 的统计手写实现 ───────────────────────────────────────────────────

def wilson_ci(k, n, z=1.96):
    """二项比例 Wilson 95% CI。"""
    if n <= 0:
        return (0.0, 0.0)
    p = k / n
    d = 1.0 + z * z / n
    c = (p + z * z / (2.0 * n)) / d
    h = z * math.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n)) / d
    return (max(0.0, c - h), min(1.0, c + h))


def mcnemar(xa, xb):
    """配对二值的 McNemar 检验（含连续性校正），返回双侧 p。xa/xb 为 0/1 对齐数组。"""
    b_ = int(((xa == 1) & (xb == 0)).sum())
    c_ = int(((xa == 0) & (xb == 1)).sum())
    if b_ + c_ == 0:
        return 1.0
    chi2 = (abs(b_ - c_) - 1.0) ** 2 / (b_ + c_)
    return math.erfc(math.sqrt(chi2 / 2.0))  # χ²(1) 生存函数


def cliff_delta(x, y):
    """Cliff's delta（非配对效应量，∈[-1,1]；x>y 记正）。"""
    x = np.sort(np.asarray(x, dtype=float))
    y = np.sort(np.asarray(y, dtype=float))
    n_y = y.size
    if x.size == 0 or n_y == 0:
        return float("nan")
    lt = np.searchsorted(y, x, side="left").sum()   # y < x 的对数
    gt = (n_y - np.searchsorted(y, x, side="right")).sum()  # y > x 的对数
    return float((lt - gt) / (x.size * n_y))


def _bootstrap_diff(xa, xb, paired):
    """Δ = mean(A) − mean(B) 的 bootstrap 95% CI 与双侧 p（固定种子）。"""
    rng = np.random.default_rng(_BOOT_SEED)
    if paired:
        d = np.asarray(xa, dtype=float) - np.asarray(xb, dtype=float)
        n = d.size
        if n == 0:
            return (float("nan"), float("nan"), float("nan"), 1.0)
        draws = rng.choice(d, size=(_N_BOOT, n), replace=True).mean(axis=1)
        obs = float(d.mean())
    else:
        xa = np.asarray(xa, dtype=float)
        xb = np.asarray(xb, dtype=float)
        if xa.size == 0 or xb.size == 0:
            return (float("nan"), float("nan"), float("nan"), 1.0)
        da = rng.choice(xa, size=(_N_BOOT, xa.size), replace=True).mean(axis=1)
        db = rng.choice(xb, size=(_N_BOOT, xb.size), replace=True).mean(axis=1)
        draws = da - db
        obs = float(xa.mean() - xb.mean())
    lo, hi = float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))
    # 双侧 bootstrap p：零分布 = S* − S_obs（中心化到 0），双侧尾部用 |obs|。
    centered = draws - obs
    t = abs(obs)
    p = 2.0 * min(float((centered <= -t).mean()), float((centered >= t).mean()))
    p = min(1.0, p)
    return (obs, lo, hi, p)


def _describe(x):
    x = np.asarray(x, dtype=float)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"n": 0}
    return {
        "n": int(x.size),
        "mean": float(x.mean()),
        "std": float(x.std(ddof=1)) if x.size > 1 else 0.0,
        "min": float(x.min()),
        "p05": float(np.percentile(x, 5)),
        "p25": float(np.percentile(x, 25)),
        "p50": float(np.percentile(x, 50)),
        "p75": float(np.percentile(x, 75)),
        "p95": float(np.percentile(x, 95)),
        "max": float(x.max()),
    }


# ── 数据载入与配对校验 ────────────────────────────────────────────────────────

def load_csv(path):
    """读 worker CSV → {列名: float ndarray}（空串=NaN）。"""
    with open(path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        names = list(reader.fieldnames)
        raw = {n: [] for n in names}
        for row in reader:
            for n in names:
                raw[n].append(row.get(n, "").strip())
    data = {}
    for n, vals in raw.items():
        out = np.full(len(vals), np.nan)
        for i, v in enumerate(vals):
            if v != "":
                try:
                    out[i] = float(v)
                except ValueError:
                    out[i] = np.nan
        data[n] = out
    return data


def pairing_analysis(a, b, tol):
    """按 episode_index 对齐两份 CSV，校验 obj_init_pos 是否逐集一致。"""
    ia = {int(v): i for i, v in enumerate(a["episode_index"]) if np.isfinite(v)}
    ib = {int(v): i for i, v in enumerate(b["episode_index"]) if np.isfinite(v)}
    common = sorted(set(ia) & set(ib))
    if not common:
        return {"common": 0, "aligned": 0, "frac": 0.0, "paired": False,
                "max_d": float("nan"), "mean_d": float("nan"),
                "idx_a": None, "idx_b": None}
    idx_a = np.array([ia[k] for k in common])
    idx_b = np.array([ib[k] for k in common])
    pa = np.stack([a["obj_init_pos_x"][idx_a],
                   a["obj_init_pos_y"][idx_a],
                   a["obj_init_pos_z"][idx_a]], axis=-1)
    pb = np.stack([b["obj_init_pos_x"][idx_b],
                   b["obj_init_pos_y"][idx_b],
                   b["obj_init_pos_z"][idx_b]], axis=-1)
    dist = np.linalg.norm(pa - pb, axis=-1)
    aligned = int((dist <= tol).sum())
    frac = aligned / len(common)
    paired = frac >= 0.99 and len(common) >= 100
    return {"common": len(common), "aligned": aligned, "frac": frac,
            "paired": paired, "max_d": float(dist.max()),
            "mean_d": float(dist.mean()), "idx_a": idx_a, "idx_b": idx_b}


def _fmt_p(p):
    return "<0.001" if p < 0.001 else f"{p:.3f}"


def _fmt_pct(r):
    return f"{r * 100:.1f}%"


# ── 图 ────────────────────────────────────────────────────────────────────────

def _cjk_font():
    """系统有 CJK 字体才用中文标签，否则退回英文（matplotlib 默认字体无中文）。"""
    if not _MPL:
        return None
    for f in font_manager.fontManager.ttflist:
        if any(k in f.name for k in ("Noto Sans CJK", "Noto Serif CJK", "WenQuanYi",
                                     "Source Han", "AR PL", "SimHei", "SimSun",
                                     "Microsoft YaHei", "PingFang")):
            return f.name
    return None


_ZH_FONT = _cjk_font() if _MPL else None


def _fig_label(label, fallback):
    """图内标签：无中文字体时把用户标签转 ASCII（去中文），全中文则退回 fallback。"""
    if _ZH_FONT is not None:
        return label
    ascii_lbl = "".join(ch for ch in label if ord(ch) < 128).strip()
    if ascii_lbl.endswith("()"):
        ascii_lbl = ascii_lbl[:-2].rstrip()
    return ascii_lbl or fallback


def make_plots(a, b, label_a, label_b, pairing, out_dir):
    """成功率条形 / 误差箱线 / 误差 CDF / 成功散点。两版或单版都支持。"""
    if not _MPL:
        print("[stats][WARN] matplotlib 不可用，跳过绘图。", flush=True)
        return []
    zh = _ZH_FONT is not None
    lab = lambda en, cn: cn if zh else en
    plt.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
        "axes.edgecolor": BASELINE, "axes.labelcolor": INK,
        "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
        "grid.color": GRID, "font.family": "sans-serif",
        "axes.grid": False, "figure.dpi": 150,
    })
    if zh:
        plt.rcParams["font.family"] = [_ZH_FONT, "DejaVu Sans"]

    two = b is not None
    fig_label_a = _fig_label(label_a, "A")
    fig_label_b = _fig_label(label_b, "B")
    paths = []
    os.makedirs(os.path.join(out_dir, "figs"), exist_ok=True)

    # 1) 成功率条形图（4 档阈值，Wilson CI 误差棒）
    groups = [("reach_1cm_3deg", "1cm/3°"), ("reach_0p5cm_3deg", "0.5cm/3°"),
              ("reach_pos_1cm", "pos 1cm"), ("ever_lifted", "lifted")]
    fig, ax = plt.subplots(figsize=(8, 4.6))
    x = np.arange(len(groups))
    w = 0.38 if two else 0.5
    offsets = [-w / 2, w / 2] if two else [0.0]
    colors = [A_COLOR, B_COLOR] if two else [A_COLOR]
    labels = [fig_label_a, fig_label_b] if two else [fig_label_a]
    for off, c, labv, series in zip(offsets, colors, labels,
                                     ([a, b] if two else [a])):
        rates, lo, hi = [], [], []
        for key, _ in groups:
            v = series[key]
            v = v[np.isfinite(v)]
            k = int((v == 1).sum())
            n = v.size
            rates.append(k / max(n, 1) * 100.0)
            wl, wu = wilson_ci(k, n)
            lo.append((k / max(n, 1) - wl) * 100.0)
            hi.append((wu - k / max(n, 1)) * 100.0)
        ax.bar(x + off, rates, w, color=c, label=labv,
               yerr=[lo, hi], capsize=3,
               error_kw=dict(ecolor=MUTED, lw=1), zorder=3)
        for xi, r in zip(x + off, rates):
            ax.text(xi, r + 0.6, f"{r:.1f}%", ha="center",
                    va="bottom", fontsize=8, color=INK)
    ax.set_xticks(x, [g[1] for g in groups], fontsize=9)
    ax.set_ylabel(lab("rate (%)", "比率 (%)"))
    ax.set_title(lab("Success / lift rates (Wilson 95% CI)",
                     "成功/举起率（Wilson 95% 置信区间）"), fontsize=11)
    ax.set_ylim(0, 105)
    ax.grid(axis="y", color=GRID, lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    if two:
        ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    p = os.path.join(out_dir, "figs", "fig1_success_rates.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # 2) 误差箱线图 2x2
    fig, axes = plt.subplots(2, 2, figsize=(9, 6.5))
    for ax, (key, en, cn, _) in zip(axes.ravel(), CONTINUOUS_METRICS[:4]):
        series = []
        for s in ([a, b] if two else [a]):
            v = s[key]
            series.append(v[np.isfinite(v)])
        bp = ax.boxplot(
            series, tick_labels=([fig_label_a, fig_label_b] if two else [fig_label_a]),
            patch_artist=True, widths=0.45,
            medianprops=dict(color=INK, lw=1.5),
            whiskerprops=dict(color=INK, lw=1),
            capprops=dict(color=INK, lw=1),
            boxprops=dict(color=INK, lw=1),
            flierprops=dict(marker="o", markersize=3, markerfacecolor=MUTED,
                            markeredgecolor="none", alpha=0.6),
        )
        for patch, c in zip(bp["boxes"], [A_COLOR, B_COLOR] if two else [A_COLOR]):
            patch.set_facecolor(c)
            patch.set_alpha(0.85)
        ax.set_ylabel(lab(en, cn), fontsize=9)
        ax.grid(axis="y", color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    fig.suptitle(lab("Error distribution (best / final)", "误差分布（最优/终止时）"),
                 fontsize=12, color=INK)
    fig.tight_layout()
    p = os.path.join(out_dir, "figs", "fig2_error_box.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # 3) 误差 CDF（best dist / best ori 两面板）
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for ax, (key, en, cn, _) in zip(axes, [CONTINUOUS_METRICS[0], CONTINUOUS_METRICS[1]]):
        for c, labv, s in zip(([A_COLOR, B_COLOR] if two else [A_COLOR]),
                              ([fig_label_a, fig_label_b] if two else [fig_label_a]),
                              ([a, b] if two else [a])):
            v = np.sort(s[key][np.isfinite(s[key])])
            n = v.size
            if n:
                ax.plot(v, np.arange(1, n + 1) / n, drawstyle="steps-post",
                        color=c, lw=2, label=labv)
        ax.set_xlabel(lab(en, cn), fontsize=9)
        ax.set_ylabel("CDF", fontsize=9)
        ax.grid(color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        if two:
            ax.legend(frameon=False, fontsize=9)
    fig.suptitle(lab("Cumulative distribution of best errors",
                     "最优误差累积分布"), fontsize=12)
    fig.tight_layout()
    p = os.path.join(out_dir, "figs", "fig3_error_cdf.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    # 4) 成功/失败散点（x=best dist, y=best ori；status 色 + 图例标签）
    ncols = 2 if two else 1
    fig, axes = plt.subplots(1, ncols, figsize=(5.2 * ncols, 4.4))
    if ncols == 1:
        axes = [axes]
    for ax, labv, s in zip(axes, ([fig_label_a, fig_label_b] if two else [fig_label_a]),
                           ([a, b] if two else [a])):
        d = s["dist_best_cm"]
        o = s["ori_best_deg"]
        r = s["reach_1cm_3deg"]
        m = np.isfinite(d) & np.isfinite(o) & np.isfinite(r)
        ok = (r[m] == 1)
        ax.scatter(d[m][ok], o[m][ok], s=8, alpha=0.55, color=GOOD,
                   linewidths=0, label=lab("reach 1cm/3° (success)", "成功（曾达 1cm/3°）"))
        ax.scatter(d[m][~ok], o[m][~ok], s=8, alpha=0.55, color=CRIT,
                   linewidths=0, label=lab("no reach (fail)", "失败（未达）"))
        ax.set_xlabel(lab("best dist (cm)", "最优距离 (cm)"), fontsize=9)
        ax.set_ylabel(lab("best ori (deg)", "最优姿态误差 (deg)"), fontsize=9)
        ax.set_title(labv, fontsize=10, color=INK)
        ax.grid(color=GRID, lw=0.7)
        ax.set_axisbelow(True)
        for s_ in ("top", "right"):
            ax.spines[s_].set_visible(False)
        ax.legend(frameon=False, fontsize=8, markerscale=1.6)
    fig.tight_layout()
    p = os.path.join(out_dir, "figs", "fig4_success_scatter.png")
    fig.savefig(p, bbox_inches="tight")
    plt.close(fig)
    paths.append(p)

    return paths


# ── 主流程 ────────────────────────────────────────────────────────────────────

def main():
    global _N_BOOT
    parser = argparse.ArgumentParser(description="点云遮挡评估统计（配对/非配对自适应）")
    parser.add_argument("--csv_a", type=str, required=True, help="A 版（遮挡训练）CSV")
    parser.add_argument("--label_a", type=str, default="A")
    parser.add_argument("--csv_b", type=str, default=None, help="B 版（baseline）CSV，缺省=单版统计")
    parser.add_argument("--label_b", type=str, default="B")
    parser.add_argument("--out_dir", type=str, required=True)
    parser.add_argument("--pair_tol_m", type=float, default=0.005,
                        help="配对校验阈值：对齐后 obj_init_pos 偏差 ≤ 该值(米) 视为同一集")
    parser.add_argument("--n_boot", type=int, default=_N_BOOT)
    args = parser.parse_args()
    _N_BOOT = args.n_boot

    os.makedirs(args.out_dir, exist_ok=True)
    a = load_csv(args.csv_a)
    b = load_csv(args.csv_b) if args.csv_b else None
    two = b is not None

    pairing = None
    if two:
        pairing = pairing_analysis(a, b, args.pair_tol_m)

    lines = []
    w = lines.append
    w("=" * 78)
    w("点云遮挡评估统计报告")
    w("=" * 78)
    w(f"A: {args.label_a}")
    w(f"   CSV: {args.csv_a}   episodes: {int(np.isfinite(a['episode_index']).sum())}")
    if two:
        w(f"B: {args.label_b}")
        w(f"   CSV: {args.csv_b}   episodes: {int(np.isfinite(b['episode_index']).sum())}")

    # ── 第 1 块：单版统计 ────────────────────────────────────────────────────
    w("")
    w("-" * 78)
    w("第 1 块  单版统计（25% 遮挡条件下的绝对水平）")
    w("-" * 78)
    single = {}
    for tag, s in ([("A", a)] + ([("B", b)] if two else [])):
        rates, cont = {}, {}
        for key, en, cn, _ in BINARY_METRICS:
            v = s[key]
            v = v[np.isfinite(v)]
            k = int((v == 1).sum())
            n = int(v.size)
            lo, hi = wilson_ci(k, n)
            rates[key] = {"k": k, "n": n, "rate": k / max(n, 1),
                          "ci": [lo, hi]}
        for key, en, cn, _ in CONTINUOUS_METRICS:
            cont[key] = _describe(s[key])
        single[tag] = {"binary": rates, "continuous": cont}

        label = args.label_a if tag == "A" else args.label_b
        w(f"")
        w(f"  [{tag}] {label}")
        for key, en, cn, _ in BINARY_METRICS:
            r = rates[key]
            w(f"      {cn:<26s} {_fmt_pct(r['rate']):>8s}  "
              f"[{_fmt_pct(r['ci'][0])}, {_fmt_pct(r['ci'][1])}]  ({r['k']}/{r['n']})")
        for key, en, cn, _ in CONTINUOUS_METRICS:
            d = cont[key]
            if d.get("n", 0) == 0:
                w(f"      {cn:<26s} 无有效样本")
            else:
                w(f"      {cn:<26s} mean {d['mean']:8.3f} ± {d['std']:7.3f}  "
                  f"min {d['min']:6.2f}  p50 {d['p50']:6.2f}  p95 {d['p95']:6.2f}  "
                  f"max {d['max']:6.2f}  (n={d['n']})")

    # ── 第 2 块：A vs B 对比 ─────────────────────────────────────────────────
    compare_binary, compare_cont = [], []
    if two:
        w("")
        w("-" * 78)
        w("第 2 块  A vs B 对比（Δ = A − B；误差/碰撞类指标 Δ<0 表示 A 更优）")
        w("-" * 78)
        if pairing["paired"]:
            w(f"配对校验: {pairing['aligned']}/{pairing['common']} 集 obj_init_pos 逐集一致"
              f"（对齐率 {pairing['frac']*100:.1f}%，最大偏差 {pairing['max_d']*1000:.3f} mm）")
            w("  → 严格配对，采用配对检验（McNemar / 配对 bootstrap）")
        else:
            w(f"配对校验: 仅 {pairing['aligned']}/{pairing['common']} 集对齐"
              f"（对齐率 {pairing['frac']*100:.1f}%，最大偏差 {pairing['max_d']*1000:.3f} mm）")
            w("  → 未逐集配对，采用非配对检验（两样本 bootstrap + Cliff's delta）")
        idx_a = pairing["idx_a"] if pairing["paired"] else None
        idx_b = pairing["idx_b"] if pairing["paired"] else None

        w("")
        w("  二值指标:")
        for key, en, cn, _ in BINARY_METRICS:
            if pairing["paired"]:
                xa = a[key][idx_a]
                xb = b[key][idx_b]
            else:
                xa = a[key][np.isfinite(a[key])]
                xb = b[key][np.isfinite(b[key])]
            ra = float((xa == 1).sum()) / max(xa.size, 1)
            rb = float((xb == 1).sum()) / max(xb.size, 1)
            if pairing["paired"]:
                diff, lo, hi, p = _bootstrap_diff(xa, xb, paired=True)
                p_mcn = mcnemar(xa, xb)
                b_only = int(((xa == 0) & (xb == 1)).sum())
                c_only = int(((xa == 1) & (xb == 0)).sum())
                test = f"McNemar p={_fmt_p(p_mcn)} (A胜{c_only}集/B胜{b_only}集)"
            else:
                diff, lo, hi, p = _bootstrap_diff(xa, xb, paired=False)
                test = f"bootstrap p={_fmt_p(p)}"
            w(f"      {cn:<26s} A {_fmt_pct(ra):>8s}  B {_fmt_pct(rb):>8s}  "
              f"Δ {diff*100:+7.2f}%  CI[{lo*100:+6.2f}%, {hi*100:+6.2f}%]  {test}")
            compare_binary.append({
                "metric": key, "rate_a": ra, "rate_b": rb,
                "diff": diff, "ci": [lo, hi], "p": p, "test": test,
            })

        w("")
        w("  连续指标（Δ 的 95% bootstrap CI）:")
        for key, en, cn, _ in CONTINUOUS_METRICS:
            if pairing["paired"]:
                xa = a[key][idx_a]
                xb = b[key][idx_b]
            else:
                xa = a[key][np.isfinite(a[key])]
                xb = b[key][np.isfinite(b[key])]
            xa = xa[np.isfinite(xa)]
            xb = xb[np.isfinite(xb)]
            diff, lo, hi, p = _bootstrap_diff(xa, xb, paired=pairing["paired"])
            eff = (float(np.mean(xa - xb) / max(np.std(xa - xb, ddof=1), 1e-9))
                   if pairing["paired"]
                   else cliff_delta(xa, xb))
            eff_name = "d_z" if pairing["paired"] else "Cliff's δ"
            w(f"      {cn:<26s} A {xa.mean():8.3f}  B {xb.mean():8.3f}  "
              f"Δ {diff:+8.3f}  CI[{lo:+8.3f}, {hi:+8.3f}]  p={_fmt_p(p)}  "
              f"{eff_name}={eff:+.3f}")
            compare_cont.append({
                "metric": key, "mean_a": float(xa.mean()), "mean_b": float(xb.mean()),
                "diff": diff, "ci": [lo, hi], "p": p,
                "effect": eff, "effect_name": eff_name,
            })

        # 配对模式下 McNemar 分歧集数也值得单独报告（遮挡下谁赢的集更多）
        if pairing["paired"]:
            w("")
            w("  配对分歧集数（同一初始条件两版结果不同）:")
            for key, en, cn, _ in BINARY_METRICS:
                xa = a[key][idx_a]
                xb = b[key][idx_b]
                w(f"      {cn:<26s} A成B败 {int(((xa==1)&(xb==0)).sum()):>5d}  "
                  f"A败B成 {int(((xa==0)&(xb==1)).sum()):>5d}  "
                  f"两成 {int(((xa==1)&(xb==1)).sum()):>5d}  "
                  f"两败 {int(((xa==0)&(xb==0)).sum()):>5d}")

    # ── 图 ───────────────────────────────────────────────────────────────────
    fig_paths = make_plots(a, b, args.label_a, args.label_b, pairing, args.out_dir)
    if fig_paths:
        w("")
        w("-" * 78)
        w("图:")
        for p in fig_paths:
            w(f"  {p}")
        fla = _fig_label(args.label_a, "A")
        flb = _fig_label(args.label_b, "B")
        if (fla != args.label_a) or (two and flb != args.label_b):
            w(f"  （本机无中文字体，图内标签已转 ASCII："
              f"“{args.label_a}” → “{fla}”"
              + (f'，“{args.label_b}” → “{flb}”' if two else "") +
              "；报表文字不受影响）")

    report_path = os.path.join(args.out_dir, "report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    # ── summary.csv / stats.json ─────────────────────────────────────────────
    csv_rows = []
    for tag, s in ([("A", a)] + ([("B", b)] if two else [])):
        label = args.label_a if tag == "A" else args.label_b
        for key, en, cn, _ in BINARY_METRICS:
            r = single[tag]["binary"][key]
            csv_rows.append({"version": tag, "metric": key, "type": "binary",
                             "label": label, "n": r["n"], "k": r["k"],
                             "rate": f"{r['rate']:.6f}",
                             "ci_lo": f"{r['ci'][0]:.6f}",
                             "ci_hi": f"{r['ci'][1]:.6f}"})
        for key, en, cn, _ in CONTINUOUS_METRICS:
            d = single[tag]["continuous"][key]
            csv_rows.append({"version": tag, "metric": key, "type": "continuous",
                             "label": label, "n": d.get("n", 0),
                             "mean": f"{d['mean']:.6f}" if d.get("n") else "",
                             "std": f"{d['std']:.6f}" if d.get("n") else "",
                             "p50": f"{d['p50']:.6f}" if d.get("n") else ""})
    if two:
        for r in compare_binary:
            csv_rows.append({"version": "diff", "metric": r["metric"], "type": "binary_diff",
                             "label": "A-B", "n": "",
                             "diff": f"{r['diff']:.6f}",
                             "ci_lo": f"{r['ci'][0]:.6f}",
                             "ci_hi": f"{r['ci'][1]:.6f}",
                             "p": f"{r['p']:.6f}", "test": r["test"]})
        for r in compare_cont:
            csv_rows.append({"version": "diff", "metric": r["metric"], "type": "cont_diff",
                             "label": "A-B", "n": "",
                             "diff": f"{r['diff']:.6f}",
                             "ci_lo": f"{r['ci'][0]:.6f}",
                             "ci_hi": f"{r['ci'][1]:.6f}",
                             "p": f"{r['p']:.6f}",
                             "effect": f"{r['effect']:.6f}",
                             "effect_name": r["effect_name"]})
    fieldnames = ["version", "type", "label", "metric", "n", "k", "rate",
                  "mean", "std", "p50", "diff", "ci_lo", "ci_hi", "p",
                  "effect", "effect_name", "test"]
    with open(os.path.join(args.out_dir, "summary.csv"), "w", newline="",
              encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        wr.writeheader()
        wr.writerows(csv_rows)

    stats_json = {
        "inputs": {"csv_a": args.csv_a, "csv_b": args.csv_b,
                   "label_a": args.label_a, "label_b": args.label_b},
        "pairing": ({k: v for k, v in pairing.items()
                     if k not in ("idx_a", "idx_b")} if pairing else None),
        "single": single,
        "compare": {"binary": compare_binary, "continuous": compare_cont},
        "figures": fig_paths,
    }
    with open(os.path.join(args.out_dir, "stats.json"), "w", encoding="utf-8") as f:
        json.dump(stats_json, f, ensure_ascii=False, indent=2, default=float)

    print("\n".join(lines), flush=True)
    print(f"\n[stats] 报表: {report_path}", flush=True)


if __name__ == "__main__":
    main()
