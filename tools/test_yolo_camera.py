#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""YOLO 权重实时相机测试脚本。

从 RealSense D435 取流，推理 YOLO 模型，实时显示检测结果并统计效果指标。
用于验证训练好的权重在实际场景下的泛化能力。

输出指标：
  - 检出率（有目标的帧占比）
  - 置信度分布（P25/P50/P75）
  - 掩码面积占比（分割模型）
  - FPS（推理速度）

可选录制模式，保存检出/漏检样本 + 逐帧统计 JSON 供后续分析。

用法::

    conda run -n gx_va_deploy python tools/test_yolo_camera.py
    conda run -n gx_va_deploy python tools/test_yolo_camera.py --conf 0.15
    conda run -n gx_va_deploy python tools/test_yolo_camera.py --record --record-dir tools/logs/camera_test_001
"""

from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime
from pathlib import Path

# 确保有 DISPLAY 环境变量才能显示窗口
if "DISPLAY" not in os.environ:
    os.environ["DISPLAY"] = ":0"  # 默认使用主显示器


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent
DEFAULT_WEIGHTS = SCRIPT_DIR / "best.pt"
DEFAULT_RECORD_ROOT = SCRIPT_DIR / "logs" / "camera_test"

# 相机包装类在 scripts/camera/ 下，需加入 sys.path
CAMERA_MODULE_DIR = PROJECT_ROOT / "scripts" / "camera"
if str(CAMERA_MODULE_DIR) not in sys.path:
    sys.path.insert(0, str(CAMERA_MODULE_DIR))

# RealSense D435 序列号（你机器上的）
D435_SERIAL = "254622072913"


def fail(msg: str) -> None:
    sys.exit(f"[错误] {msg}")


def check_deps() -> None:
    """前置检查：ultralytics、pyrealsense2、cv2，给出可操作的提示。"""
    missing = []
    try:
        import cv2  # noqa: F401
    except ModuleNotFoundError:
        missing.append("opencv-python")
    try:
        import pyrealsense2  # noqa: F401
    except ModuleNotFoundError:
        missing.append("pyrealsense2")
    try:
        from ultralytics import YOLO  # noqa: F401
    except ModuleNotFoundError:
        missing.append("ultralytics")

    if missing:
        fail(
            f"当前 Python 缺依赖: {', '.join(missing)}。\n"
            "       建议用 gx_va_deploy 环境运行（已装全）:\n"
            "       conda run -n gx_va_deploy python tools/test_yolo_camera.py\n\n"
            "       或在 camera 环境补装 ultralytics:\n"
            "       conda run -n camera pip install ultralytics"
        )


def draw_detections(frame, result, conf_threshold: float):
    """在图像上绘制 YOLO 检测结果：掩码 + 框 + 置信度标签。

    Args:
        frame: (H, W, 3) uint8 RGB
        result: YOLO Results 对象
        conf_threshold: 置信度阈值

    Returns:
        annotated: 标注后的 RGB 图像
        stats: {n_instances, max_conf, mask_area_ratio}
    """
    import cv2
    import numpy as np

    overlay = frame.copy()
    stats = {"n_instances": 0, "max_conf": 0.0, "mask_area_ratio": 0.0}

    if result.boxes is None or len(result.boxes) == 0:
        return overlay, stats

    cpu = result.cpu()
    boxes = cpu.boxes.xyxy.numpy()
    confs = cpu.boxes.conf.numpy()

    stats["n_instances"] = len(boxes)
    stats["max_conf"] = float(confs.max()) if len(confs) else 0.0

    # 半透明红色掩码
    if result.masks is not None:
        masks_data = cpu.masks.data.numpy()
        H, W = frame.shape[:2]
        union_mask = np.zeros((H, W), dtype=bool)

        for i, m in enumerate(masks_data):
            if m.shape != (H, W):
                m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
            mask_bool = m > 0.5
            union_mask |= mask_bool
            # 半透明红色
            overlay[mask_bool] = (0.6 * overlay[mask_bool] + 0.4 * np.array([255, 40, 40])).astype(np.uint8)

        stats["mask_area_ratio"] = float(union_mask.sum()) / float(H * W)

    # 边界框与标签
    for i, (box, conf) in enumerate(zip(boxes, confs)):
        x1, y1, x2, y2 = box.astype(int)
        label = f"target {conf:.2f}"

        # 红色边界框
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (255, 0, 0), 2)

        # 文字背景
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (x1, y1 - th - 4), (x1 + tw, y1), (255, 0, 0), -1)
        cv2.putText(overlay, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return overlay, stats


class TestRecorder:
    """录制检出/漏检样本与逐帧统计。"""

    def __init__(self, root_dir: Path, max_hit: int = 50, max_miss: int = 20):
        self.root = root_dir
        self.max_hit = max_hit
        self.max_miss = max_miss

        self.hit_dir = root_dir / "detected"
        self.miss_dir = root_dir / "missed"
        self.hit_dir.mkdir(parents=True, exist_ok=True)
        self.miss_dir.mkdir(parents=True, exist_ok=True)

        self.hit_count = 0
        self.miss_count = 0
        self.frame_log = []

    def save_frame(self, rgb, frame_id: int, detected: bool, conf: float, area_ratio: float) -> None:
        """保存样本图像（RGB uint8）。"""
        import cv2

        if detected and self.hit_count < self.max_hit:
            path = self.hit_dir / f"frame_{frame_id:05d}_conf{conf:.2f}.jpg"
            cv2.imwrite(str(path), rgb[..., ::-1])  # RGB → BGR
            self.hit_count += 1
        elif not detected and self.miss_count < self.max_miss:
            path = self.miss_dir / f"frame_{frame_id:05d}_miss.jpg"
            cv2.imwrite(str(path), rgb[..., ::-1])
            self.miss_count += 1

        self.frame_log.append({
            "frame_id": frame_id,
            "detected": detected,
            "conf_max": round(conf, 4) if detected else 0.0,
            "mask_area_ratio": round(area_ratio, 5) if detected else 0.0,
        })

    def finalize(self, total_time: float, fps_avg: float) -> None:
        """保存统计 JSON。"""
        report = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "total_frames": len(self.frame_log),
            "detected_frames": sum(1 for f in self.frame_log if f["detected"]),
            "missed_frames": sum(1 for f in self.frame_log if not f["detected"]),
            "total_time_sec": round(total_time, 2),
            "fps_avg": round(fps_avg, 2),
            "samples_saved": {"detected": self.hit_count, "missed": self.miss_count},
            "per_frame": self.frame_log,
        }
        path = self.root / "test_stats.json"
        path.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"[录制] 统计已保存: {path}")
        print(f"[录制] 检出样本 {self.hit_count} 张: {self.hit_dir}")
        print(f"[录制] 漏检样本 {self.miss_count} 张: {self.miss_dir}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="建议用 gx_va_deploy 环境运行（已装全依赖）。",
    )
    p.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS, help="YOLO 权重文件")
    p.add_argument("--serial", default=D435_SERIAL, help=f"RealSense 序列号（默认 D435: {D435_SERIAL}）")
    p.add_argument("--width", type=int, default=640, help="相机分辨率宽")
    p.add_argument("--height", type=int, default=480, help="相机分辨率高")
    p.add_argument("--conf", type=float, default=0.25, help="YOLO 置信度阈值")
    p.add_argument("--imgsz", type=int, default=640, help="YOLO 推理分辨率")
    p.add_argument("--device", default="0", help="推理设备，0=GPU, cpu=CPU")
    p.add_argument("--record", action="store_true", help="录制检出/漏检样本与统计 JSON")
    p.add_argument("--record-dir", type=Path, default=None, help="录制输出目录，默认按时间戳")
    p.add_argument("--max-hit", type=int, default=50, help="录制模式最多保存几张检出样本")
    p.add_argument("--max-miss", type=int, default=20, help="录制模式最多保存几张漏检样本")
    p.add_argument("--headless", action="store_true", help="无头模式：不显示窗口，只录制和输出统计")
    return p.parse_args()


def print_summary(total_frames: int, detected_frames: int, confs: list[float],
                  area_ratios: list[float], fps_avg: float) -> None:
    """退出时打印统计汇总。"""
    miss = total_frames - detected_frames
    rate = detected_frames / total_frames if total_frames else 0.0

    print("\n" + "=" * 50)
    print("=== 测试汇总 ===")
    print(f"总帧数      : {total_frames}")
    print(f"检出帧数    : {detected_frames}  ({rate:.1%})")
    print(f"漏检帧数    : {miss}  ({1 - rate:.1%})")

    if confs:
        sorted_confs = sorted(confs)
        p25 = sorted_confs[len(sorted_confs) // 4]
        p50 = sorted_confs[len(sorted_confs) // 2]
        p75 = sorted_confs[len(sorted_confs) * 3 // 4]
        print(f"置信度 P25/P50/P75: {p25:.2f} / {p50:.2f} / {p75:.2f}")
    else:
        print("置信度: (无检出)")

    if area_ratios:
        avg_area = statistics.fmean(area_ratios)
        print(f"掩码面积占比均值: {avg_area:.3%}")

    print(f"平均 FPS    : {fps_avg:.1f}")
    print("=" * 50)


def main() -> int:
    args = parse_args()
    check_deps()

    args.weights = args.weights.expanduser()
    if not args.weights.is_file():
        fail(f"权重文件不存在: {args.weights}")

    # 录制目录
    recorder = None
    if args.record:
        if args.record_dir is None:
            args.record_dir = DEFAULT_RECORD_ROOT / f"{datetime.now():%Y%m%d_%H%M%S}"
        args.record_dir = args.record_dir.expanduser().resolve()
        recorder = TestRecorder(args.record_dir, max_hit=args.max_hit, max_miss=args.max_miss)
        print(f"[录制] 输出目录: {args.record_dir}")

    # 加载模型
    from ultralytics import YOLO
    model = YOLO(str(args.weights))
    print(f"[YOLO] 权重 : {args.weights}")
    print(f"[YOLO] 任务 : {model.task}")
    print(f"[YOLO] 类别 : {model.names}")

    # 启动相机
    from realsense_camera import RealsenseCamera
    cam = RealsenseCamera(
        width=args.width, height=args.height, fps=30,
        align_on_device=True, serial_number=args.serial
    )
    K = cam.start()
    print(f"[相机] fx={K[0,0]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

    # OpenCV 窗口
    import cv2
    # OpenCV 窗口（非无头模式才创建）
    if not args.headless:
        window_name = "YOLO Camera Test (Press 'q' to quit)"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window_name, 1280, 720)

    # 统计变量
    frame_times = []
    total_frames = 0
    detected_frames = 0
    all_confs = []
    all_area_ratios = []
    start_time = time.time()

    if args.headless:
        print("[运行] 无头模式：不显示窗口，Ctrl+C 停止\n")
    else:
        print("[运行] 实时检测已启动，按 'q' 退出\n")

    try:
        while True:
            t0 = time.time()

            rgb, depth = cam.get_frame()
            if rgb is None:
                print("[相机] 取帧失败，重试")
                continue

            total_frames += 1

            # YOLO 推理（ultralytics 期望 BGR）
            res = model.predict(
                rgb[..., ::-1], imgsz=args.imgsz, conf=args.conf,
                device=args.device, verbose=False
            )[0]

            # 绘制 + 统计
            annotated, stats = draw_detections(rgb, res, args.conf)
            detected = stats["n_instances"] > 0

            if detected:
                detected_frames += 1
                all_confs.append(stats["max_conf"])
                if stats["mask_area_ratio"] > 0:
                    all_area_ratios.append(stats["mask_area_ratio"])

            # 录制样本
            if recorder:
                recorder.save_frame(rgb, total_frames, detected,
                                   stats["max_conf"], stats["mask_area_ratio"])

            # FPS 统计
            frame_times.append(time.time() - t0)
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps = len(frame_times) / sum(frame_times) if frame_times else 0.0

            # 叠加 FPS 和检出状态（非无头模式才绘制）
            if not args.headless:
                status_color = (0, 255, 0) if detected else (128, 128, 128)
                status_text = f"FPS: {fps:.1f} | Frame: {total_frames} | Detected: {detected_frames}"
                cv2.putText(annotated, status_text, (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2, cv2.LINE_AA)

                # 显示（OpenCV 用 BGR）
                cv2.imshow(window_name, annotated[..., ::-1])

                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
            else:
                # 无头模式：每 30 帧打印一次进度
                if total_frames % 30 == 0:
                    print(f"[进度] 帧数: {total_frames} | 检出: {detected_frames} | FPS: {fps:.1f}")

    except KeyboardInterrupt:
        print("\n[中断]")
    finally:
        elapsed = time.time() - start_time
        fps_avg = total_frames / elapsed if elapsed > 0 else 0.0

        cam.stop()
        if not args.headless:
            cv2.destroyAllWindows()

        print_summary(total_frames, detected_frames, all_confs, all_area_ratios, fps_avg)

        if recorder:
            recorder.finalize(elapsed, fps_avg)

    return 0


if __name__ == "__main__":
    sys.exit(main())

