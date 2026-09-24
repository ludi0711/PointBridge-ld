#!/usr/bin/env python3
"""手动标注工件 ROI 并保存掩码。

启动后显示相机 RGB 图像，用鼠标拖拽矩形框选工件区域，
按 's' 保存掩码 + 内参到 JSON，按 'q' 退出。

用法::

    conda activate gx_va_deploy
    cd /home/gxai/Desktop/CZR/gx-VA-isaaclab
    python scripts/deploy/manual_roi_annotate.py
    python scripts/deploy/manual_roi_annotate.py --serial 230322270207  # D405
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# 添加项目路径
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.dirname(_SCRIPT_DIR)
_PROJECT_DIR = os.path.dirname(_SCRIPTS_DIR)
sys.path.insert(0, os.path.join(_SCRIPTS_DIR, "camera"))

import cv2
import numpy as np

D435_SERIAL = "254622072913"
D405_SERIAL = "230322270207"

# 全局变量：多边形点集
polygon_points = []
current_frame = None


def mouse_callback(event, x, y, flags, param):
    """鼠标回调：左键添加多边形顶点，右键完成。"""
    global polygon_points

    if event == cv2.EVENT_LBUTTONDOWN:
        polygon_points.append((x, y))
        print(f"[点 {len(polygon_points)}] ({x}, {y})")
    elif event == cv2.EVENT_RBUTTONDOWN:
        if len(polygon_points) >= 3:
            print(f"[完成] 多边形共 {len(polygon_points)} 个点")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--serial", default=D435_SERIAL,
                   help=f"相机序列号。D435={D435_SERIAL}（默认），D405={D405_SERIAL}")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--output", default="roi_mask.json",
                   help="输出文件路径（JSON 格式）")
    return p.parse_args()


def main() -> int:
    args = parse_args()
    global current_frame

    from realsense_camera import RealsenseCamera
    cam = RealsenseCamera(width=args.width, height=args.height, fps=30,
                          align_on_device=True, serial_number=args.serial)
    K = cam.start()
    print(f"[相机] 内参 fx={K[0,0]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

    window_name = "Manual ROI Annotation (Left click: add point, Right click: close polygon, 's': save, 'q': quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(window_name, mouse_callback)

    print("[提示] 左键点击添加多边形顶点，右键闭合多边形，按 's' 保存，'r' 重置，'q' 退出")

    try:
        while True:
            rgb, depth = cam.get_frame()
            if rgb is None:
                print("[相机] 取帧失败，重试")
                time.sleep(0.1)
                continue

            current_frame = rgb.copy()
            display = current_frame.copy()

            # 绘制多边形顶点和边
            if len(polygon_points) > 0:
                # 绘制已有顶点
                for i, pt in enumerate(polygon_points):
                    cv2.circle(display, pt, 4, (0, 255, 0), -1)
                    cv2.putText(display, str(i+1), (pt[0]+8, pt[1]-8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

                # 绘制边
                if len(polygon_points) > 1:
                    pts = np.array(polygon_points, dtype=np.int32)
                    cv2.polylines(display, [pts], False, (0, 255, 0), 2)

                # 如果多边形已闭合（>=3个点），绘制填充预览
                if len(polygon_points) >= 3:
                    pts = np.array(polygon_points, dtype=np.int32)
                    overlay = display.copy()
                    cv2.fillPoly(overlay, [pts], (0, 255, 0))
                    display = cv2.addWeighted(display, 0.7, overlay, 0.3, 0)

                # 显示点数
                cv2.putText(display, f"Points: {len(polygon_points)}", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            # BGR 显示
            cv2.imshow(window_name, display[..., ::-1])

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("[退出]")
                break
            elif key == ord('r'):
                polygon_points.clear()
                print("[重置] 多边形已清空")
            elif key == ord('s'):
                if len(polygon_points) < 3:
                    print("[错误] 至少需要 3 个点形成多边形")
                    continue

                # 生成多边形掩码
                H, W = rgb.shape[:2]
                mask = np.zeros((H, W), dtype=np.uint8)
                pts = np.array(polygon_points, dtype=np.int32)
                cv2.fillPoly(mask, [pts], 255)

                if mask.sum() < 100:
                    print("[错误] 多边形区域太小")
                    continue

                # 计算边界框（用于加载时快速定位）
                x_coords = [pt[0] for pt in polygon_points]
                y_coords = [pt[1] for pt in polygon_points]
                x1, y1 = min(x_coords), min(y_coords)
                x2, y2 = max(x_coords), max(y_coords)

                # 保存到 JSON
                output_path = Path(args.output)
                output_data = {
                    "polygon": [[int(pt[0]), int(pt[1])] for pt in polygon_points],
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                    "mask_shape": [int(H), int(W)],
                    "intrinsics": {
                        "fx": float(K[0, 0]),
                        "fy": float(K[1, 1]),
                        "cx": float(K[0, 2]),
                        "cy": float(K[1, 2]),
                    },
                    "serial": args.serial,
                    "resolution": [int(W), int(H)],
                }

                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w") as f:
                    json.dump(output_data, f, indent=2)

                # 保存掩码图像
                mask_path = output_path.with_suffix(".png")
                cv2.imwrite(str(mask_path), mask)

                print(f"[保存] 多边形顶点: {len(polygon_points)} 个")
                print(f"[保存] 掩码像素: {mask.sum() // 255}")
                print(f"[保存] JSON: {output_path}")
                print(f"[保存] 掩码图: {mask_path}")
                break

    except KeyboardInterrupt:
        print("\n[中断]")
    finally:
        cam.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
