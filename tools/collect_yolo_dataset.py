#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""收集仿真 / 真实 RGB 图片，输出可直接上传 YOLO 平台打标的图片集。

仿真图片按 episode 分层采样（借助 index.csv 的 episode/shot 元信息），保证工件
位姿与时间点覆盖最广；真实图片按均匀步长采样，避开连续视频帧的近重复。
两边都保持原始格式不转码，附带 manifest.csv 便于打标后回溯来源。
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path


SIM_FRAMES_DIR = Path("/home/gxai/Desktop/CZR/gx-VA-isaaclab/logs/policy_rgb_episodes/frames")
REAL_ROOT_DIR = Path("/mnt/shared/gxpose/datasets/foundationpose_real_cleaned")
DEFAULT_OUTPUT_ROOT = Path.home() / "Desktop"

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg"}
PNG_MAGIC = b"\x89PNG\r\n\x1a\n"
JPEG_MAGIC = b"\xff\xd8\xff"

META_FIELDS = [
    "episode", "env", "shot", "sim_time_s",
    "obj_x", "obj_y", "obj_z", "obj_qw", "obj_qx", "obj_qy", "obj_qz",
]
MANIFEST_FIELDS = ["filename", "domain", "session", "source_name", "source_path"] + META_FIELDS


def evenly_spaced(total: int, count: int, phase: float = 0.0) -> list[int]:
    """在 [0, total) 上取 count 个均匀下标；phase∈[0,1) 把整组下标平移不到一个步长。"""
    if count <= 0 or total <= 0:
        return []
    if count >= total:
        return list(range(total))
    step = total / count
    shift = phase * step
    return [min(total - 1, int(j * step + shift)) for j in range(count)]


def looks_like_image(path: Path) -> bool:
    """轻量校验：magic bytes + JPEG 尾标记，挡掉截断或损坏的文件（不依赖 PIL）。"""
    try:
        size = path.stat().st_size
        if size < 64:
            return False
        with path.open("rb") as f:
            head = f.read(8)
            if head.startswith(PNG_MAGIC):
                return True
            if not head.startswith(JPEG_MAGIC):
                return False
            f.seek(-2, 2)
            return f.read(2) == b"\xff\xd9"
    except OSError:
        return False


def episode_key(value: str) -> tuple[int, int | str]:
    """episode 字段按数值排序，非数值退回字符串排序。"""
    try:
        return (0, int(value))
    except (TypeError, ValueError):
        return (1, str(value))


def build_sim_pool(frames_dir: Path) -> tuple[list[dict[str, str]], str]:
    """返回 (候选行, 来源说明)。优先用 index.csv 带上 episode/位姿元信息。"""
    index_csv = frames_dir.parent / "index.csv"
    if index_csv.is_file():
        with index_csv.open(newline="", encoding="utf-8") as f:
            rows = [r for r in csv.DictReader(f) if r.get("image")]
        pool = []
        for row in rows:
            path = frames_dir / row["image"]
            if path.is_file():
                entry = {"path": str(path), "source_name": row["image"]}
                entry.update({k: row.get(k, "") for k in META_FIELDS})
                pool.append(entry)
        if pool:
            note = f"index.csv ({len(pool)}/{len(rows)} 行命中磁盘文件)"
            return pool, note

    files = sorted(p for p in frames_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
    pool = [{"path": str(p), "source_name": p.name} for p in files]
    return pool, f"目录文件名排序 ({len(pool)} 张，无 index.csv 元信息)"


def build_real_pool(root: Path) -> tuple[list[dict[str, str]], str]:
    """把各 session 的 rgb 首尾相接成一条序列。

    传 .../session_xxx/rgb 则只收该 session；传数据集根目录则按 session 名排序收全部。
    后续用单一步长扫过整条序列，各 session 的张数自然成比例、步长统一。
    """
    if root.name == "rgb":
        sessions = [root]
    else:
        sessions = sorted(p / "rgb" for p in root.iterdir() if p.is_dir() and (p / "rgb").is_dir())
    if not sessions:
        return [], f"{root} 下没有找到 rgb 子目录"

    pool, counts = [], []
    for rgb_dir in sessions:
        name = rgb_dir.parent.name if rgb_dir.name == "rgb" else rgb_dir.name
        files = sorted(p for p in rgb_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)
        pool += [{"path": str(p), "source_name": p.name, "session": name} for p in files]
        counts.append(f"{name}:{len(files)}")
    return pool, f"{len(sessions)} 个 session 顺序拼接 ({', '.join(counts)})"


def sample_stratified(pool: list[dict[str, str]], count: int, rng: random.Random) -> list[dict[str, str]]:
    """按 episode 分层采样：配额均摊到每个 episode，episode 内的 shot 均匀铺开。

    每个 episode 的起始相位随机，避免 200 张图全部落在同样几个 shot 上。
    """
    if count >= len(pool):
        return list(pool)

    groups: dict[str, list[dict[str, str]]] = {}
    for entry in pool:
        groups.setdefault(entry.get("episode", ""), []).append(entry)

    keys = sorted(groups, key=episode_key)
    if len(keys) <= 1:
        return [pool[i] for i in evenly_spaced(len(pool), count)]

    picked: list[dict[str, str]] = []
    n_groups = len(keys)
    for i, key in enumerate(keys):
        # 差分取整：配额之和精确等于 count，且余数均匀散布在各 episode 上
        quota = (i + 1) * count // n_groups - i * count // n_groups
        rows = groups[key]
        for j in evenly_spaced(len(rows), quota, phase=rng.random()):
            picked.append(rows[j])

    if len(picked) < count:  # 某些 episode 图片数少于配额时补齐
        chosen = {id(e) for e in picked}
        for entry in pool:
            if len(picked) >= count:
                break
            if id(entry) not in chosen:
                picked.append(entry)
                chosen.add(id(entry))

    picked.sort(key=lambda e: (episode_key(e.get("episode", "")), e["source_name"]))
    return picked[:count]


def sample_uniform(pool: list[dict[str, str]], count: int) -> list[dict[str, str]]:
    """均匀步长采样，用于连续视频帧：整段铺满，天然避开相邻近重复帧。"""
    return [pool[i] for i in evenly_spaced(len(pool), count)]


def copy_batch(
    picked: list[dict[str, str]],
    domain: str,
    images_dir: Path,
    dry_run: bool,
) -> list[dict[str, str]]:
    """拷贝并统一重命名为 <domain>_0001.<原后缀>，返回 manifest 行。"""
    records = []
    for i, entry in enumerate(picked, start=1):
        src = Path(entry["path"])
        filename = f"{domain}_{i:04d}{src.suffix.lower()}"
        if not dry_run:
            shutil.copy2(src, images_dir / filename)
        record = {
            "filename": filename,
            "domain": domain,
            "session": entry.get("session", ""),
            "source_name": entry["source_name"],
            "source_path": entry["path"],
        }
        record.update({k: entry.get(k, "") for k in META_FIELDS})
        records.append(record)
    return records


def prepare_pool(
    label: str,
    directory: Path,
    builder,
) -> tuple[list[dict[str, str]], list[str], str]:
    """构建候选池并剔除损坏文件，返回 (可用池, 被剔除的文件名, 来源说明)。"""
    if not directory.is_dir():
        sys.exit(f"[错误] {label}目录不存在: {directory}")
    pool, note = builder(directory)
    if not pool:
        sys.exit(f"[错误] {label}目录里没有找到图片: {directory}")
    good, bad = [], []
    for entry in pool:
        if looks_like_image(Path(entry["path"])):
            good.append(entry)
        else:
            bad.append(entry["source_name"])
    if not good:
        sys.exit(f"[错误] {label}目录下所有文件都无法识别为有效图片: {directory}")
    return good, bad, note


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="收集仿真与真实 RGB 图片，输出 YOLO 打标用图片集",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--sim-dir", type=Path, default=SIM_FRAMES_DIR, help="仿真图片目录")
    p.add_argument("--real-dir", type=Path, default=REAL_ROOT_DIR,
                   help="真实图片根目录（收全部 session）或单个 session 的 rgb 目录")
    p.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT, help="输出根目录")
    p.add_argument("--output-name", default=None, help="输出文件夹名，默认 yolo_labeling_<日期>")
    p.add_argument("--sim-count", type=int, default=250, help="仿真图片采样数量")
    p.add_argument("--real-count", type=int, default=250, help="真实图片采样数量")
    p.add_argument("--seed", type=int, default=42, help="采样随机种子，固定以便复现")
    p.add_argument("--dry-run", action="store_true", help="只打印采样结果，不落盘")
    p.add_argument("--overwrite", action="store_true", help="允许写入已存在且非空的输出目录")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.sim_count < 0 or args.real_count < 0:
        sys.exit("[错误] 采样数量不能为负数")
    if args.sim_count + args.real_count == 0:
        sys.exit("[错误] 仿真与真实采样数量不能同时为 0")

    rng = random.Random(args.seed)
    sim_dir = args.sim_dir.expanduser().resolve()
    real_dir = args.real_dir.expanduser().resolve()

    sim_pool, sim_bad, sim_note = prepare_pool("仿真", sim_dir, build_sim_pool)
    real_pool, real_bad, real_note = prepare_pool("真实", real_dir, build_real_pool)

    sim_picked = sample_stratified(sim_pool, args.sim_count, rng)
    real_picked = sample_uniform(real_pool, args.real_count)

    print("=== 数据源 ===")
    print(f"仿真: {sim_dir}")
    print(f"      候选 {len(sim_pool)} 张，来源 {sim_note}")
    print(f"真实: {real_dir}")
    print(f"      候选 {len(real_pool)} 张，来源 {real_note}")
    for label, bad in (("仿真", sim_bad), ("真实", real_bad)):
        if bad:
            preview = ", ".join(bad[:5]) + (" ..." if len(bad) > 5 else "")
            print(f"[警告] {label}目录跳过 {len(bad)} 个无效/损坏文件: {preview}")

    print("\n=== 采样结果 ===")
    print(f"仿真 {len(sim_picked)}/{args.sim_count} 张（按 episode 分层，seed={args.seed}）")
    if sim_picked and sim_picked[0].get("episode"):
        eps = {e["episode"] for e in sim_picked}
        shots = sorted({e["shot"] for e in sim_picked}, key=episode_key)
        print(f"     覆盖 {len(eps)} 个 episode，shot 取值 {shots}")
    print(f"真实 {len(real_picked)}/{args.real_count} 张（均匀步长 "
          f"{len(real_pool) / max(1, args.real_count):.2f}）")
    real_by_session = Counter(e.get("session", "") for e in real_picked)
    if real_by_session:
        detail = ", ".join(f"{k or '-'}:{v}" for k, v in sorted(real_by_session.items()))
        print(f"     各 session 分布 {detail}")
    for label, picked, want in (("仿真", sim_picked, args.sim_count), ("真实", real_picked, args.real_count)):
        if len(picked) < want:
            print(f"[警告] {label}候选不足，只取到 {len(picked)} 张（需要 {want} 张）")

    out_name = args.output_name or f"yolo_labeling_{datetime.now():%Y%m%d}"
    out_dir = (args.output_root.expanduser().resolve() / out_name)
    images_dir = out_dir / "images"

    if args.dry_run:
        print(f"\n[dry-run] 未写入任何文件。目标目录会是: {out_dir}")
        for label, picked in (("仿真", sim_picked), ("真实", real_picked)):
            for entry in picked[:3]:
                print(f"  {label} <- {entry['source_name']}")
            if len(picked) > 3:
                print(f"  {label} ... 共 {len(picked)} 张")
        return

    if images_dir.exists() and any(images_dir.iterdir()) and not args.overwrite:
        sys.exit(f"[错误] 输出目录已存在且非空: {images_dir}\n"
                 f"       换个 --output-name，或加 --overwrite 覆盖同名文件。")
    images_dir.mkdir(parents=True, exist_ok=True)

    records = copy_batch(sim_picked, "sim", images_dir, dry_run=False)
    records += copy_batch(real_picked, "real", images_dir, dry_run=False)

    manifest_path = out_dir / "manifest.csv"
    with manifest_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(records)

    report = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "seed": args.seed,
        "output_dir": str(out_dir),
        "total_images": len(records),
        "sim": {
            "source_dir": str(sim_dir),
            "source_note": sim_note,
            "available": len(sim_pool),
            "requested": args.sim_count,
            "collected": len(sim_picked),
            "strategy": "stratified_by_episode",
            "skipped_invalid": sim_bad,
        },
        "real": {
            "source_dir": str(real_dir),
            "source_note": real_note,
            "available": len(real_pool),
            "requested": args.real_count,
            "collected": len(real_picked),
            "strategy": "uniform_stride",
            "stride": round(len(real_pool) / max(1, args.real_count), 2),
            "by_session": dict(sorted(real_by_session.items())),
            "skipped_invalid": real_bad,
        },
    }
    report_path = out_dir / "collect_report.json"
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")

    total_bytes = sum((images_dir / r["filename"]).stat().st_size for r in records)
    print(f"\n=== 完成 ===")
    print(f"图片   {len(records)} 张 -> {images_dir}")
    print(f"清单   {manifest_path}")
    print(f"报告   {report_path}")
    print(f"占用   {total_bytes / 1024 / 1024:.1f} MB")


if __name__ == "__main__":
    main()
