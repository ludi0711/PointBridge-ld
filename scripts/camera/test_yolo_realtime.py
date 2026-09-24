#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实时看 YOLO-seg 在**真机**画面上的分割效果。不连机械臂、不出动作、不建点云。

这是三步上机流程的第一步：先确认 YOLO 在真实相机画面上认得出工件，再谈用它替掉
手标多边形。之所以要单独测，是因为 `20260815.pt` 的那份指标（漏检 0/560、
mAP50-95 0.98）是在**仿真渲染图**上统计的，真机的曝光、白平衡、桌面反光、镜头
畸变都不一样 —— 仿真上完美不等于真机上能用。

看什么：

    det 数     >1 就是 end2end 的重复检出（已知现象，取 conf 最高那个，质心差
               <1mm）。恒为 0 说明画面漂出了 YOLO 的训练分布。
    conf       真机上普遍比仿真低是正常的；低于 --conf 门限就整帧漏检。
    area       掩码面积(px)。仿真上中位 12370 px，真机明显偏小说明只分割出了一部分。
    valid      掩码内**深度有效**的像素数。这一项直接预告部署时的 hold 行为 ——
               部署脚本的 `--min_visible`(默认 200) 就是拿这个数做门控，低于它
               控制环持续 hold。掩码再漂亮，深度是空洞也没用。
    lat        单帧 YOLO 耗时(ms)。50Hz 控制周期是 20ms，若这里明显超过 20ms，
               部署时 YOLO 必须放到后台线程异步跑（第三步的脚本就是这么做的）。

**与部署路径的关系**：本脚本从 ``configs/yolo_mask_source.py`` 取
``get_yolo`` / ``rgb_to_yolo_input``（权重缓存、预热、RGB 量纲判断那套全部同源），
但自己做 predict 与取最高 conf，为的是把所有检出都显示出来做诊断。部署脚本走的是
同一模块的 ``yolo_masks``。反投影管线（规格 §1.1 只允许一份）本脚本完全不碰。

用法::

    conda activate gx_va_deploy

    # 默认权重 20260815.pt
    python scripts/camera/test_yolo_realtime.py

    # 换权重 / 调门限
    python scripts/camera/test_yolo_realtime.py --weights /path/to/new.pt --conf 0.15

键位：q 退出   s 存图   [ / ] 调 conf 门限   m 切换掩码填充   d 存诊断 npz
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

import numpy as np

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))
sys.path.insert(0, _PROJECT_DIR)

def parse_args() -> argparse.Namespace:
    from configs.yolo_mask_source import DEFAULT_YOLO_WEIGHTS, YOLO_CONF, YOLO_IMGSZ

    p = argparse.ArgumentParser(
        description="实时查看 YOLO-seg 在真机相机画面上的分割效果（不连机械臂）")
    p.add_argument("--weights", type=str, default=DEFAULT_YOLO_WEIGHTS,
                   help=f"YOLO-seg 权重，默认 {DEFAULT_YOLO_WEIGHTS}")
    p.add_argument("--conf", type=float, default=YOLO_CONF,
                   help=f"检出置信度门限，默认 {YOLO_CONF}（与训练/部署同一门限）")
    p.add_argument("--imgsz", type=int, default=YOLO_IMGSZ,
                   help=f"推理尺寸，默认 {YOLO_IMGSZ}。模型在 640 上训的，别改")
    p.add_argument("--device", type=str, default="cuda", choices=("cuda", "cpu"))

    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--serial", type=str, default=None)
    p.add_argument("--warmup", type=int, default=30)

    p.add_argument("--min_visible", type=int, default=200,
                   help="部署脚本的 hold 门限，这里只用于把 valid 标红做预警")
    p.add_argument("--depth_range", type=float, nargs=2, default=(0.3, 1.5),
                   metavar=("LO", "HI"), help="深度伪彩固定色标区间(米)")
    p.add_argument("--max_frames", type=int, default=0, help="跑够帧数退出；0=一直跑")
    p.add_argument("--no_window", action="store_true",
                   help="无显示器时用：只打印统计，不开窗口")
    p.add_argument("--out", type=str,
                   default=os.path.join(_PROJECT_DIR, "logs", "yolo_realtime"))
    return p.parse_args()


def main() -> None:
    args = parse_args()

    import cv2
    import torch

    from configs.yolo_mask_source import get_yolo, rgb_to_yolo_input
    from scripts.camera.realsense_camera import RealsenseCamera
    from scripts.camera.view_real_depth import colorize_depth, report_intrinsics

    if args.device == "cuda" and not torch.cuda.is_available():
        print("[警告] CUDA 不可用，退回 CPU（YOLO 会慢很多）")
        args.device = "cpu"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print("\n" + "=" * 72)
    print(f"[YOLO] {args.weights}")
    print(f"[参数] conf={args.conf}  imgsz={args.imgsz}  device={args.device}")
    print(f"[相机] {args.width}x{args.height}@{args.fps}")
    print("=" * 72 + "\n")

    # 先加载模型再开相机：权重路径写错的话立刻报错，不用先等相机起流
    model = get_yolo(args.weights, device=args.device)

    camera = RealsenseCamera(width=args.width, height=args.height, fps=args.fps,
                            align_on_device=True, serial_number=args.serial)
    camera.start()
    report_intrinsics(camera)

    print(f"[预热] 丢弃前 {args.warmup} 帧...")
    for _ in range(args.warmup):
        camera.get_frame()

    if not args.no_window:
        print("\n[键位] q 退出  s 存图  [ / ] 调 conf  m 切换填充  d 存诊断 npz\n")

    conf = float(args.conf)
    fill_mask = True
    frame_i = 0
    # 统计量按帧累积，退出时汇总 —— 单帧数字会跳，看的是分布
    stat_det, stat_conf, stat_area, stat_valid, stat_lat = [], [], [], [], []

    try:
        while args.max_frames <= 0 or frame_i < args.max_frames:
            rgb, depth = camera.get_frame()
            if rgb is None or depth is None:
                continue

            # ── YOLO 推理 ──
            # rgb_to_yolo_input 与训练/部署同源：切 alpha、permute、量纲判断都在里面
            x = rgb_to_yolo_input(
                torch.from_numpy(rgb).unsqueeze(0).to(args.device))
            t0 = time.perf_counter()
            results = model.predict(x, imgsz=args.imgsz, conf=conf,
                                    device=args.device, verbose=False)
            if args.device == "cuda":
                torch.cuda.synchronize()
            lat_ms = (time.perf_counter() - t0) * 1000.0

            res = results[0]
            n_det = 0 if res.masks is None else len(res.masks)

            mask = np.zeros((args.height, args.width), dtype=bool)
            best_conf = 0.0
            all_conf: list[float] = []
            if n_det > 0:
                all_conf = [float(c) for c in res.boxes.conf]
                best = int(torch.argmax(res.boxes.conf))
                best_conf = all_conf[best]
                m = res.masks.data[best]
                if tuple(m.shape) != (args.height, args.width):
                    # nearest 而非双线性：双线性会在边缘造出 0~1 中间值，阈值化后
                    # 掩码边界发生亚像素漂移。与 yolo_mask_source 里逐字一致。
                    m = torch.nn.functional.interpolate(
                        m[None, None].float(), size=(args.height, args.width),
                        mode="nearest")[0, 0]
                mask = (m > 0.5).cpu().numpy()

            area = int(mask.sum())
            # 掩码内深度有效的像素数 —— 部署时 --min_visible 门控的就是这个量
            valid = int((mask & np.isfinite(depth) & (depth > 0)).sum())

            stat_det.append(n_det)
            stat_conf.append(best_conf)
            stat_area.append(area)
            stat_valid.append(valid)
            stat_lat.append(lat_ms)

            if frame_i % 30 == 0:
                flag = ""
                if n_det == 0:
                    flag = "  <== 漏检"
                elif valid < args.min_visible:
                    flag = f"  <== valid<{args.min_visible}，部署时会 hold"
                print(f"frame {frame_i:5d}  det={n_det}  conf={best_conf:.3f}  "
                      f"area={area:6d}px  valid={valid:5d}  lat={lat_ms:5.1f}ms{flag}")

            # ── 可视化 ──
            if not args.no_window:
                bgr = rgb[..., ::-1].copy()
                depth_vis, _, _ = colorize_depth(
                    depth, args.depth_range[0], args.depth_range[1])

                if area > 0:
                    if fill_mask:
                        ov = bgr.copy()
                        ov[mask] = (0, 0, 255)
                        bgr = cv2.addWeighted(bgr, 0.65, ov, 0.35, 0)
                    cnts, _ = cv2.findContours(mask.astype(np.uint8),
                                               cv2.RETR_EXTERNAL,
                                               cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(bgr, cnts, -1, (0, 255, 0), 2)
                    cv2.drawContours(depth_vis, cnts, -1, (0, 255, 0), 2)
                    ys, xs = np.nonzero(mask)
                    cv2.circle(bgr, (int(xs.mean()), int(ys.mean())), 5,
                               (255, 255, 0), -1)

                ok = n_det > 0 and valid >= args.min_visible
                info = (f"det={n_det} conf={best_conf:.3f} area={area}px "
                        f"valid={valid} lat={lat_ms:.1f}ms conf_th={conf:.2f}")
                for img in (bgr, depth_vis):
                    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
                    cv2.putText(img, info, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                                (255, 255, 255) if ok else (0, 200, 255), 1,
                                cv2.LINE_AA)

                # 漏检和深度不足画得刺眼：这两种都会让部署环节静默 hold
                warn = None
                if n_det == 0:
                    warn = "!! NO DETECTION -- policy would hold !!"
                elif valid < args.min_visible:
                    warn = f"!! valid {valid} < min_visible {args.min_visible} !!"
                elif n_det > 1:
                    warn = f"note: {n_det} dets (end2end duplicate, using max conf)"
                if warn is not None:
                    red = n_det == 0 or valid < args.min_visible
                    for img in (bgr, depth_vis):
                        cv2.rectangle(img, (0, 22), (img.shape[1], 44),
                                      (0, 0, 200) if red else (0, 120, 120), -1)
                        cv2.putText(img, warn, (5, 38), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.42, (255, 255, 255), 1, cv2.LINE_AA)

                combined = np.hstack([bgr, depth_vis])
                cv2.imshow("YOLO-seg realtime (real camera)", combined)

                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), ord("Q"), 27):
                    print("\n[退出] 用户按键退出")
                    break
                elif key == ord("s"):
                    fp = out_dir / f"yolo_{time.strftime('%Y%m%d_%H%M%S')}_f{frame_i}.png"
                    cv2.imwrite(str(fp), combined)
                    print(f"[截图] {fp}")
                elif key == ord("["):
                    conf = max(0.01, round(conf - 0.05, 2))
                    print(f"[门限] conf={conf:.2f}")
                elif key == ord("]"):
                    conf = min(0.95, round(conf + 0.05, 2))
                    print(f"[门限] conf={conf:.2f}")
                elif key == ord("m"):
                    fill_mask = not fill_mask
                elif key == ord("d"):
                    fp = out_dir / f"diag_{time.strftime('%Y%m%d_%H%M%S')}_f{frame_i}.npz"
                    np.savez_compressed(fp, rgb=rgb, depth=depth, mask=mask,
                                        confs=np.asarray(all_conf, dtype=np.float32),
                                        conf_th=conf, n_det=n_det)
                    print(f"[诊断] {fp}（含 rgb/depth/mask/confs，可离线复算）")

            frame_i += 1

    except KeyboardInterrupt:
        print("\n[退出] Ctrl-C")
    finally:
        camera.stop()
        if not args.no_window:
            import cv2 as _cv2
            _cv2.destroyAllWindows()

        if stat_det:
            det = np.asarray(stat_det)
            cf = np.asarray(stat_conf)
            ar = np.asarray(stat_area)
            va = np.asarray(stat_valid)
            la = np.asarray(stat_lat)
            n = len(det)
            print("\n" + "=" * 72)
            print(f"[汇总] {n} 帧")
            print(f"  漏检(det=0)   {int((det == 0).sum()):5d} / {n}  "
                  f"({100.0 * (det == 0).mean():.1f}%)")
            print(f"  单检(det=1)   {int((det == 1).sum()):5d} / {n}")
            print(f"  重复(det>=2)  {int((det >= 2).sum()):5d} / {n}  "
                  f"(end2end 已知现象，取 max conf)")
            hit = det > 0
            if hit.any():
                print(f"  conf   均值 {cf[hit].mean():.3f}  最小 {cf[hit].min():.3f}")
                print(f"  area   中位 {int(np.median(ar[hit])):6d}px  "
                      f"最小 {int(ar[hit].min()):6d}px")
                print(f"  valid  中位 {int(np.median(va[hit])):6d}   "
                      f"最小 {int(va[hit].min()):6d}")
                bad = int((va < args.min_visible).sum())
                print(f"  valid < min_visible({args.min_visible})  {bad} / {n}  "
                      f"({100.0 * bad / n:.1f}%)  ← 部署时这些帧会 hold")
            print(f"  延迟   均值 {la.mean():.1f}ms  p95 "
                  f"{np.percentile(la, 95):.1f}ms  最大 {la.max():.1f}ms")
            if la.mean() > 20.0:
                print("  [注意] 均值超过 50Hz 的 20ms 周期 —— 部署时 YOLO 必须异步跑"
                      "（deploy 脚本的 --mask_source yolo 已经是后台线程）")
            print("=" * 72)


if __name__ == "__main__":
    main()
