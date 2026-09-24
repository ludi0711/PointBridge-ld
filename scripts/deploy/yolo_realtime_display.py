#!/usr/bin/env python3
"""YOLO 工件检测 + RealSense 实时画面标注。

用 OpenCV 窗口实时显示相机画面，叠加 YOLO 检测结果：
  - 分割掩码（半透明红色）
  - 边界框 + 类别标签
  - 多任务头输出：selection_score、side、back_probability

使用 D435 相机（fx=605.4，与仿真标定的 615.0 同量级）。

用法::

    conda run -n gx_va_deploy python scripts/deploy/yolo_realtime_display.py
    conda run -n gx_va_deploy python scripts/deploy/yolo_realtime_display.py --conf 0.15
"""

from __future__ import annotations

import argparse
import os
import sys
import time

# YOLO fork 必须比 pip 装的 ultralytics 先进 sys.path
YOLO_REPO = "/home/gxai/Desktop/GSworld/gongjian_multitask_training"
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.dirname(_SCRIPT_DIR)
_PROJECT_DIR = os.path.dirname(_SCRIPTS_DIR)
for _p in (YOLO_REPO, _PROJECT_DIR, os.path.join(_SCRIPTS_DIR, "camera")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

def _drop_isaac_paths() -> list[str]:
    """摘掉 Isaac Sim 路径，避免 filelock 3.13.1 盖掉环境里的 3.20.1。"""
    bad = [p for p in sys.path if "_isaac_sim" in p or "isaac-sim" in p.lower()]
    for p in bad:
        sys.path.remove(p)
    for mod in ("filelock",):
        loaded = sys.modules.get(mod)
        if loaded is not None and "_isaac_sim" in (getattr(loaded, "__file__", "") or ""):
            del sys.modules[mod]
    return bad

_DROPPED = _drop_isaac_paths()

import cv2
import numpy as np

DEFAULT_WEIGHTS = os.path.join(YOLO_REPO, "0806_weights", "best.pt")
D435_SERIAL = "254622072913"
D405_SERIAL = "230322270207"

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--serial", default=D435_SERIAL,
                   help=f"相机序列号。D435={D435_SERIAL}（默认），D405={D405_SERIAL}")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--conf", type=float, default=0.25, help="YOLO 置信度阈值")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--device", default="0")
    return p.parse_args()


def draw_detections(frame: np.ndarray, result, class_names: dict) -> np.ndarray:
    """在图像上绘制 YOLO 检测结果：掩码 + 框 + 标签 + 多任务字段。

    Args:
        frame: (H, W, 3) uint8 RGB
        result: YOLO Results 对象
        class_names: {0: 'gongjian', 1: 'kong'}

    Returns:
        标注后的图像（RGB uint8）
    """
    overlay = frame.copy()

    if result.boxes is None or len(result.boxes) == 0:
        return overlay

    cpu = result.cpu()
    boxes = cpu.boxes.xyxy.numpy()  # (N, 4)
    confs = cpu.boxes.conf.numpy()
    cls = cpu.boxes.cls.int().numpy()

    # 多任务字段
    g = cpu.gongjian
    has_multitask = g is not None and len(getattr(g, "selection_score", [])) > 0

    # 掩码先画（半透明）
    if result.masks is not None:
        masks_data = cpu.masks.data.numpy()  # (N, h_net, w_net)
        H, W = frame.shape[:2]
        for i in range(len(masks_data)):
            m = masks_data[i]
            if m.shape != (H, W):
                m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
            mask_bool = m > 0.5

            # 颜色：工件=红、孔=蓝
            color = (255, 40, 40) if cls[i] == 0 else (40, 100, 255)
            overlay[mask_bool] = (0.6 * overlay[mask_bool] + 0.4 * np.array(color)).astype(np.uint8)

    # 框和标签
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i].astype(int)
        conf = confs[i]
        c = int(cls[i])

        # 框颜色
        box_color = (255, 0, 0) if c == 0 else (0, 120, 255)  # BGR for cv2
        cv2.rectangle(overlay, (x1, y1), (x2, y2), box_color, 2)

        # 标签文本
        label = f"{class_names[c]} {conf:.2f}"
        if has_multitask and c == 0 and i < len(g.selection_score):
            score = float(g.selection_score[i])
            side = g.side[i] if i < len(g.side) else "?"
            label += f" | score={score:.2f} {side}"

        # 文字背景
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(overlay, (x1, y1 - th - 4), (x1 + tw, y1), box_color, -1)
        cv2.putText(overlay, label, (x1, y1 - 2), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (255, 255, 255), 1, cv2.LINE_AA)

    return overlay


def main() -> int:
    args = parse_args()

    if _DROPPED:
        print(f"[环境] 已从 sys.path 摘掉 {len(_DROPPED)} 条 Isaac Sim 路径")

    from ultralytics import YOLO
    import ultralytics
    print(f"[YOLO] fork: {ultralytics.__file__}")
    if not ultralytics.__file__.startswith(YOLO_REPO):
        print("[警告] 用的不是定制 fork，result.gongjian 可能不存在")
    model = YOLO(args.weights)
    print(f"[YOLO] 权重 {args.weights}  类别 {model.names}")

    from realsense_camera import RealsenseCamera
    cam = RealsenseCamera(width=args.width, height=args.height, fps=30,
                          align_on_device=True, serial_number=args.serial)
    K = cam.start()
    print(f"[对比] 仿真标定 fx=615.0 cx=320.0 cy=240.0 / "
          f"实测 fx={K[0,0]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

    window_name = "YOLO Real-time Detection (Press 'q' to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1280, 720)

    frame_times = []

    try:
        print("[运行] 实时检测已启动，按 'q' 退出")
        while True:
            t0 = time.time()

            rgb, depth = cam.get_frame()
            if rgb is None:
                print("[相机] 取帧失败，重试")
                continue

            # YOLO 推理 —— ultralytics 的 preprocess() 期望 BGR 输入（OpenCV 惯例），
            # 会自动做 BGR→RGB 翻转。RealsenseCamera 返回 RGB，所以要先转回 BGR。
            res = model.predict(rgb[..., ::-1], imgsz=args.imgsz, conf=args.conf,
                                device=args.device, verbose=False)[0]

            # 绘制检测结果
            annotated = draw_detections(rgb, res, model.names)

            # FPS 统计
            frame_times.append(time.time() - t0)
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps = len(frame_times) / sum(frame_times) if frame_times else 0.0

            # 叠加 FPS
            cv2.putText(annotated, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)

            # BGR 转换后显示（OpenCV 用 BGR）
            cv2.imshow(window_name, annotated[..., ::-1])

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

    except KeyboardInterrupt:
        print("\n[退出]")
    finally:
        cam.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
