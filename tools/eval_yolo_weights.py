#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""检验 YOLO 权重的训练效果。

两种模式，按传入参数自动选择：

  1. 带标签验证（给 --data 指向 data.yaml）
     跑 ultralytics 的 val，输出 mAP50 / mAP50-95 / P / R；分割模型同时给出
     box 和 mask 两套指标。这是唯一能定量说明"训练得好不好"的方式。

  2. 无标签体检（给 --source 指向图片目录）
     没有真值时退而求其次：统计检出率、置信度分布、掩码面积占比，并按文件名
     前缀（sim_ / real_）分组对比。检出率在两个域上差距大，就是 sim2real 差距
     的直接信号。可视化结果存盘供人工翻看。

用法（ultralytics 装在 sam2anno / sam6d / foundationpose 环境里）：
  P=/home/gxai/miniconda3/envs/sam2anno/bin/python
  $P eval_yolo_weights.py --data /path/to/data.yaml          # 定量验证
  $P eval_yolo_weights.py --source /path/to/images           # 无标签体检
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


TOOLS_DIR = Path(__file__).resolve().parent
DEFAULT_WEIGHTS = TOOLS_DIR / "best (1).pt"
DEFAULT_OUTPUT_ROOT = TOOLS_DIR / "logs" / "yolo_eval"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
# 与 collect_yolo_dataset.py 的命名对齐，用于分域统计
DOMAIN_PREFIXES = ("sim", "real")


def fail(msg: str) -> None:
    sys.exit(f"[错误] {msg}")


def load_model(weights: Path):
    """延迟导入 ultralytics，好让缺依赖时给出可操作的提示而不是裸 traceback。"""
    if not weights.is_file():
        fail(f"权重文件不存在: {weights}")
    try:
        from ultralytics import YOLO
    except ModuleNotFoundError:
        fail(
            "当前 Python 没装 ultralytics。已知装了的环境：\n"
            "       /home/gxai/miniconda3/envs/sam2anno/bin/python   (8.4.69, 最新)\n"
            "       /home/gxai/miniconda3/envs/sam6d/bin/python      (8.0.135)\n"
            "       /home/gxai/miniconda3/envs/foundationpose/bin/python (8.0.120)\n"
            "       注意 yolov8 环境名字有误导性，里面并没有 ultralytics。"
        )
    return YOLO(str(weights))


def domain_of(name: str) -> str:
    """从文件名前缀推断域；认不出就归到 all。"""
    head = name.split("_", 1)[0].lower()
    return head if head in DOMAIN_PREFIXES else "all"


def describe(values: list[float]) -> dict[str, float]:
    """给一组数的分位概览，空列表返回空 dict 而不是抛异常。"""
    if not values:
        return {}
    ordered = sorted(values)
    return {
        "count": len(ordered),
        "mean": round(statistics.fmean(ordered), 4),
        "min": round(ordered[0], 4),
        "p25": round(ordered[len(ordered) // 4], 4),
        "median": round(statistics.median(ordered), 4),
        "p75": round(ordered[len(ordered) * 3 // 4], 4),
        "max": round(ordered[-1], 4),
    }


def metrics_block(box_or_seg) -> dict[str, float]:
    """从 ultralytics 的 Metric 对象里取标量指标，跨版本字段名有差异故逐个兜底。"""
    out = {}
    for key, attr in (("mAP50", "map50"), ("mAP50-95", "map"),
                      ("precision", "mp"), ("recall", "mr")):
        value = getattr(box_or_seg, attr, None)
        if value is not None:
            try:
                out[key] = round(float(value), 4)
            except (TypeError, ValueError):
                pass
    return out


def run_val(model, args) -> dict:
    """带标签验证：直接复用 ultralytics 的 val，指标最权威。"""
    data_path = args.data.expanduser().resolve()
    if not data_path.is_file():
        fail(f"data.yaml 不存在: {data_path}")

    print(f"=== 带标签验证 ===\ndata  : {data_path}\nsplit : {args.split}")
    results = model.val(
        data=str(data_path),
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        batch=args.batch,
        project=str(args.output_root),
        name=args.run_name,
        exist_ok=True,
        plots=True,
        verbose=False,
    )

    report = {"mode": "val", "data": str(data_path), "split": args.split}
    box = metrics_block(getattr(results, "box", None)) if getattr(results, "box", None) else {}
    seg = metrics_block(getattr(results, "seg", None)) if getattr(results, "seg", None) else {}
    if box:
        report["box"] = box
        print("\n--- 检测框指标 ---")
        for k, v in box.items():
            print(f"  {k:<10} {v}")
    if seg:
        report["mask"] = seg
        print("\n--- 分割掩码指标 ---")
        for k, v in seg.items():
            print(f"  {k:<10} {v}")
    if not box and not seg:
        print("[警告] 没能从 val 结果里读到指标，检查 data.yaml 的标签路径是否正确。")

    speed = getattr(results, "speed", None)
    if isinstance(speed, dict):
        report["speed_ms"] = {k: round(float(v), 2) for k, v in speed.items()}
        print("\n--- 单张耗时 (ms) ---")
        for k, v in report["speed_ms"].items():
            print(f"  {k:<12} {v}")

    save_dir = getattr(results, "save_dir", None)
    if save_dir:
        report["save_dir"] = str(save_dir)
        print(f"\n曲线与混淆矩阵: {save_dir}")
    return report


def collect_images(source: Path) -> list[Path]:
    if source.is_file():
        return [source]
    if not source.is_dir():
        fail(f"图片来源不存在: {source}")
    # 打标集常见两种布局：直接一堆图，或套一层 images/
    base = source / "images" if (source / "images").is_dir() else source
    files = sorted(p for p in base.rglob("*") if p.suffix.lower() in IMAGE_SUFFIXES)
    if not files:
        fail(f"目录下没有图片: {base}")
    return files


def run_inspect(model, args) -> dict:
    """无标签体检：统计检出率与置信度，按域分组对比。"""
    files = collect_images(args.source.expanduser().resolve())
    if args.limit and args.limit < len(files):
        step = len(files) / args.limit  # 均匀抽样，避免只看到排序靠前的那一批
        files = [files[int(i * step)] for i in range(args.limit)]

    out_dir = args.output_root / args.run_name
    vis_dir = out_dir / "vis"
    if args.save_vis:
        vis_dir.mkdir(parents=True, exist_ok=True)
    else:
        out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== 无标签体检 ===\n图片  : {len(files)} 张\nconf  : {args.conf}")
    print("[提示] 没有真值标签，以下是检出率与置信度统计，不是 mAP。\n"
          "       要定量评估请用 --data 指向 data.yaml。\n")

    per_domain: dict[str, dict] = defaultdict(
        lambda: {"images": 0, "hit": 0, "conf": [], "inst": [], "area": []}
    )
    rows = []

    for i, path in enumerate(files, start=1):
        result = model.predict(
            str(path), imgsz=args.imgsz, conf=args.conf, iou=args.iou,
            device=args.device, verbose=False,
        )[0]

        boxes = getattr(result, "boxes", None)
        confs = [float(c) for c in boxes.conf] if boxes is not None and len(boxes) else []
        domain = domain_of(path.name)
        stat = per_domain[domain]
        stat["images"] += 1
        stat["inst"].append(len(confs))
        if confs:
            stat["hit"] += 1
            stat["conf"].append(max(confs))  # 每张图取最高分，反映"能否找到目标"

        # 掩码面积占比：分割模型独有，异常小/大都提示掩码质量有问题
        area_ratio = None
        masks = getattr(result, "masks", None)
        if masks is not None and getattr(masks, "data", None) is not None and len(masks.data):
            m = masks.data
            area_ratio = float(m.any(dim=0).sum().item()) / float(m.shape[-1] * m.shape[-2])
            stat["area"].append(area_ratio)

        rows.append({
            "filename": path.name,
            "domain": domain,
            "instances": len(confs),
            "conf_max": round(max(confs), 4) if confs else "",
            "mask_area_ratio": round(area_ratio, 5) if area_ratio is not None else "",
            "source_path": str(path),
        })

        if args.save_vis:
            result.save(filename=str(vis_dir / f"{path.stem}_pred.jpg"))
        if i % 100 == 0 or i == len(files):
            print(f"  已处理 {i}/{len(files)}")

    return summarize_inspect(per_domain, rows, out_dir, args)


def summarize_inspect(per_domain: dict, rows: list[dict], out_dir: Path, args) -> dict:
    """打印分域统计、落盘明细 CSV，返回报告 dict。"""
    import csv

    summary = {}
    print("\n--- 分域统计 ---")
    header = f"  {'域':<6} {'图片':>5} {'检出':>5} {'检出率':>7} {'置信度均值':>10} {'实例均值':>8}"
    print(header)
    for domain in sorted(per_domain):
        s = per_domain[domain]
        n, hit = s["images"], s["hit"]
        rate = hit / n if n else 0.0
        conf_mean = statistics.fmean(s["conf"]) if s["conf"] else 0.0
        inst_mean = statistics.fmean(s["inst"]) if s["inst"] else 0.0
        print(f"  {domain:<6} {n:>5} {hit:>5} {rate:>6.1%} "
              f"{conf_mean:>10.3f} {inst_mean:>8.2f}")
        summary[domain] = {
            "images": n,
            "detected": hit,
            "miss": n - hit,
            "detect_rate": round(rate, 4),
            "conf_max_stats": describe(s["conf"]),
            "instances_per_image": round(inst_mean, 3),
            "mask_area_ratio_stats": describe(s["area"]),
        }

    # sim/real 都在时给出域间差距，这是 sim2real 最直接的读数
    if {"sim", "real"} <= set(summary):
        gap = summary["sim"]["detect_rate"] - summary["real"]["detect_rate"]
        summary["sim_vs_real"] = {
            "detect_rate_gap": round(gap, 4),
            "note": "正值表示仿真图检出率更高，差距大说明模型在真实图上泛化不足",
        }
        print(f"\n  仿真 - 真实 检出率差距: {gap:+.1%}")

    # 漏检清单：最该人工看的样本
    missed = [r["filename"] for r in rows if r["instances"] == 0]
    if missed:
        preview = ", ".join(missed[:8]) + (" ..." if len(missed) > 8 else "")
        print(f"\n[注意] {len(missed)} 张图未检出任何目标: {preview}")

    inst_counts = Counter(r["instances"] for r in rows)
    print(f"\n--- 每图实例数分布 ---")
    for k in sorted(inst_counts):
        print(f"  {k} 个: {inst_counts[k]} 张")

    out_dir.mkdir(parents=True, exist_ok=True)
    detail_csv = out_dir / "per_image.csv"
    with detail_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n逐图明细: {detail_csv}")
    if args.save_vis:
        print(f"可视化  : {out_dir / 'vis'}")

    return {
        "mode": "inspect",
        "source": str(args.source),
        "images_evaluated": len(rows),
        "conf_threshold": args.conf,
        "by_domain": summary,
        "instances_histogram": {str(k): v for k, v in sorted(inst_counts.items())},
        "missed_images": missed,
        "per_image_csv": str(detail_csv),
    }


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="检验 YOLO 权重效果：给 --data 走 mAP 验证，给 --source 走无标签体检",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog="ultralytics 装在 sam2anno / sam6d / foundationpose 环境，用对应的 python 运行。",
    )
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="权重文件路径")
    p.add_argument("--data", type=Path, default=None, help="data.yaml 路径，给了就走带标签验证")
    p.add_argument("--source", type=Path, default=None, help="图片目录或单张图，走无标签体检")
    p.add_argument("--split", default="val", choices=["train", "val", "test"], help="验证用的 split")
    p.add_argument("--imgsz", type=int, default=640, help="推理分辨率")
    p.add_argument("--conf", type=float, default=0.25, help="置信度阈值")
    p.add_argument("--iou", type=float, default=0.7, help="NMS IoU 阈值")
    p.add_argument("--batch", type=int, default=16, help="验证 batch size")
    p.add_argument("--device", default=None, help="设备，如 0 或 cpu，默认自动")
    p.add_argument("--limit", type=int, default=0, help="体检模式最多看几张（0=全部，均匀抽样）")
    p.add_argument("--save-vis", action="store_true", help="体检模式保存带框/掩码的可视化图")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="输出根目录")
    p.add_argument("--run-name", default=None, help="本次运行的子目录名，默认按时间戳")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.data is None and args.source is None:
        fail("至少给一个：--data 指向 data.yaml（定量验证），或 --source 指向图片目录（无标签体检）。")
    if not 0.0 <= args.conf <= 1.0:
        fail("--conf 必须在 0 到 1 之间")

    args.weights = args.weights.expanduser()
    args.output_root = args.output_root.expanduser().resolve()
    mode = "val" if args.data is not None else "inspect"
    if args.run_name is None:
        args.run_name = f"{mode}_{datetime.now():%Y%m%d_%H%M%S}"

    model = load_model(args.weights)
    print(f"权重  : {args.weights}")
    print(f"任务  : {model.task}")
    print(f"类别  : {model.names}\n")

    report = run_val(model, args) if mode == "val" else run_inspect(model, args)

    report.update({
        "weights": str(args.weights),
        "task": model.task,
        "names": {str(k): v for k, v in model.names.items()},
        "imgsz": args.imgsz,
        "iou": args.iou,
        "created_at": datetime.now().isoformat(timespec="seconds"),
    })

    out_dir = args.output_root / args.run_name
    out_dir.mkdir(parents=True, exist_ok=True)
    report_path = out_dir / "eval_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\n=== 完成 ===\n报告: {report_path}")


if __name__ == "__main__":
    main()
