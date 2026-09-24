#!/usr/bin/env python3
"""把固定相机的三路 data_types 各自存成图，肉眼确认每一路都对。

三路分别是（configs/xarm7_pick_pointcloud_env_cfg.py:315）::

    distance_to_image_plane        深度   → 反投影必需
    instance_id_segmentation_fast  实例ID → 掩码必需
    rgb                            彩色   → 只有 stage 0 当底图用

本脚本**只读不改**：不碰任何训练配置，不写 logs 以外的地方。用于回答
"这三路各自长什么样、RGB 是不是真的没用到"。

掩码那一路刻意复用环境自己的 ``_get_instance_lut``，而不是另写一份查表 ——
这样看到的掩码就是训练时真正用的那个，有偏差能当场暴露。

输出（默认 logs/camera_data_types/）::

    env{i}_1_rgb.png          原始彩色
    env{i}_2_depth.png        深度伪彩（turbo，附 min/max 标注）
    env{i}_3_segmentation.png 实例 ID 伪彩（每个 ID 一种颜色）
    env{i}_4_mask.png         工件掩码（白=工件），标注像素数
    env{i}_5_overlay.png      掩码红色叠在 RGB 上，看掩码是否贴合工件
    combined.png              全部拼一张

用法::

    ~/IsaacLab/isaaclab.sh -p scripts/train/dump_camera_data_types.py
    ~/IsaacLab/isaaclab.sh -p scripts/train/dump_camera_data_types.py --num_envs 4
"""

import argparse
import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))
sys.path.insert(0, _PROJECT_DIR)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="导出固定相机三路 data_types 的可视化")
parser.add_argument("--num_envs", type=int, default=2, help="导出前几个 env")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--warmup",
    type=int,
    default=6,
    help="reset 后空步数。RTX 需要几帧收敛，设 0 会拿到半渲染的图。",
)
parser.add_argument(
    "--out",
    type=str,
    default=os.path.join(_PROJECT_DIR, "logs", "camera_data_types"),
)
parser.add_argument("--env_spacing", type=float, default=6.0)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv

from configs.xarm7_pick_pointcloud_env_cfg import (
    XArm7PickPointCloudPlayEnvCfg,
    _get_instance_lut,
)


def _colorize_depth(depth: np.ndarray) -> tuple[np.ndarray, float, float]:
    """深度 → turbo 伪彩。只按**有效像素**定标，否则背景的 0 会把动态范围压死。"""
    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        return np.zeros((*depth.shape, 3), np.uint8), 0.0, 0.0
    lo, hi = float(depth[valid].min()), float(depth[valid].max())
    norm = np.zeros_like(depth, dtype=np.float32)
    if hi > lo:
        norm[valid] = (depth[valid] - lo) / (hi - lo)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~valid] = 0  # 无效像素涂黑，和真实的近距离区分开
    return img, lo, hi


def _colorize_segmentation(seg: np.ndarray) -> np.ndarray:
    """实例 ID → 伪彩。ID 是任意整数，用固定调色板按出现顺序上色。"""
    out = np.zeros((*seg.shape, 3), np.uint8)
    ids = np.unique(seg)
    rng = np.random.default_rng(0)  # 固定种子，多次运行配色一致
    for k, sid in enumerate(ids):
        if sid == 0:
            continue  # 0 视作背景，留黑
        color = rng.integers(60, 256, size=3)
        out[seg == sid] = color
    return out


def _label(img: np.ndarray, text: str) -> np.ndarray:
    """左上角加一条黑底白字，避免文字落在浅色区域看不清。"""
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, text, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                (255, 255, 255), 1, cv2.LINE_AA)
    return img


def main() -> None:
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    import random as _py_random

    _py_random.seed(args_cli.seed)

    cfg = XArm7PickPointCloudPlayEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.scene.env_spacing = args_cli.env_spacing
    cfg.seed = args_cli.seed

    env = ManagerBasedRLEnv(cfg=cfg)
    camera = env.scene["camera_fixed"]

    os.makedirs(args_cli.out, exist_ok=True)

    env.reset()
    zero_action = torch.zeros(
        (env.num_envs, env.action_manager.total_action_dim), device=env.device
    )
    for _ in range(args_cli.warmup):
        env.step(zero_action)

    out = camera.data.output
    print("=" * 78)
    print(f"相机实际输出的 data_types: {list(out.keys())}")
    for k, v in out.items():
        print(f"  {k:34s} shape={tuple(v.shape)}  dtype={v.dtype}")
    print("=" * 78)

    depth_raw = out.get("distance_to_image_plane")
    seg_raw = out.get("instance_id_segmentation_fast")
    rgb_raw = out.get("rgb")

    # 掩码走环境自己的查表，保证与训练一致
    lut = _get_instance_lut(env, camera) if seg_raw is not None else None

    rows = []
    for i in range(env.num_envs):
        tiles = []

        # ── 1. RGB ──────────────────────────────────────────────────────
        rgb_bgr = None
        if rgb_raw is not None:
            arr = rgb_raw[i].detach().cpu().numpy()
            if arr.shape[-1] == 4:
                arr = arr[..., :3]
            if arr.dtype != np.uint8:
                hi = float(np.nanmax(arr)) if arr.size else 0.0
                scale = 1.0 if hi > 1.5 else 255.0
                arr = np.clip(np.nan_to_num(arr) * scale, 0, 255).astype(np.uint8)
            rgb_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
            cv2.imwrite(os.path.join(args_cli.out, f"env{i}_1_rgb.png"), rgb_bgr)
            tiles.append(_label(rgb_bgr, "1. rgb (stage0 only)"))

        # ── 2. 深度 ─────────────────────────────────────────────────────
        depth_np = None
        if depth_raw is not None:
            d = depth_raw[i]
            d = d[..., 0] if d.dim() == 3 else d
            depth_np = d.detach().cpu().numpy()
            img, lo, hi = _colorize_depth(depth_np)
            cv2.imwrite(os.path.join(args_cli.out, f"env{i}_2_depth.png"), img)
            tiles.append(_label(img, f"2. depth {lo:.2f}~{hi:.2f}m"))

        # ── 3. 分割 ─────────────────────────────────────────────────────
        seg_np = None
        if seg_raw is not None:
            s = seg_raw[i]
            s = s[..., 0] if s.dim() == 3 else s
            seg_np = s.detach().cpu().numpy().astype(np.int64)
            img = _colorize_segmentation(seg_np)
            cv2.imwrite(os.path.join(args_cli.out, f"env{i}_3_segmentation.png"), img)
            tiles.append(_label(img, f"3. seg ({len(np.unique(seg_np))} ids)"))

        # ── 4. 工件掩码 ─────────────────────────────────────────────────
        mask_np = None
        if seg_np is not None and lut is not None:
            seg_t = torch.from_numpy(seg_np).to(lut.device)
            seg_idx = seg_t.long().clamp_(min=0, max=lut.numel() - 1)
            mask_np = (lut[seg_idx] == (i + 1)).cpu().numpy()
            n_px = int(mask_np.sum())
            img = np.zeros((*mask_np.shape, 3), np.uint8)
            img[mask_np] = (255, 255, 255)
            # 规格 §2.2 的门槛是 200~500，低于它点云会退化
            ok = "OK" if n_px >= 200 else "LOW!"
            cv2.imwrite(os.path.join(args_cli.out, f"env{i}_4_mask.png"), img)
            tiles.append(_label(img, f"4. object mask {n_px}px {ok}"))

        # ── 5. 掩码叠加 ─────────────────────────────────────────────────
        if mask_np is not None and rgb_bgr is not None:
            ov = rgb_bgr.copy()
            red = np.zeros_like(ov)
            red[mask_np] = (0, 0, 255)
            ov = cv2.addWeighted(ov, 1.0, red, 0.55, 0)
            cv2.imwrite(os.path.join(args_cli.out, f"env{i}_5_overlay.png"), ov)
            tiles.append(_label(ov, "5. mask over rgb"))

        if tiles:
            h = min(t.shape[0] for t in tiles)
            tiles = [cv2.resize(t, (int(t.shape[1] * h / t.shape[0]), h)) for t in tiles]
            rows.append(np.hstack(tiles))

        # 数值摘要，图看不出的问题这里能看出来
        if depth_np is not None:
            v = np.isfinite(depth_np) & (depth_np > 0)
            print(f"env{i}  深度有效像素 {int(v.sum())}/{depth_np.size} "
                  f"({100*v.mean():.1f}%)  范围 {depth_np[v].min():.3f}~"
                  f"{depth_np[v].max():.3f} m" if v.any() else f"env{i}  深度全无效!")
        if mask_np is not None:
            n = int(mask_np.sum())
            if n:
                ys, xs = np.nonzero(mask_np)
                dv = depth_np[mask_np] if depth_np is not None else np.array([0.0])
                dv = dv[np.isfinite(dv) & (dv > 0)]
                print(f"env{i}  工件掩码 {n}px  bbox x[{xs.min()}~{xs.max()}] "
                      f"y[{ys.min()}~{ys.max()}]  工件深度 "
                      f"{dv.min():.3f}~{dv.max():.3f} m" if dv.size else
                      f"env{i}  工件掩码 {n}px 但深度全无效!")
            else:
                print(f"env{i}  工件掩码为空! 分割或 LUT 有问题")

    if rows:
        w = max(r.shape[1] for r in rows)
        rows = [np.pad(r, ((0, 0), (0, w - r.shape[1]), (0, 0))) for r in rows]
        cv2.imwrite(os.path.join(args_cli.out, "combined.png"), np.vstack(rows))

    print("=" * 78)
    print(f"输出目录: {args_cli.out}")
    print("  combined.png 是拼图，单独的图见 env*_[1-5]_*.png")
    print("=" * 78)

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close()
