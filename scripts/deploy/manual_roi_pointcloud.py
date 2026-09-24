#!/usr/bin/env python3
"""基于手动标注的 ROI 掩码采样点云（实时）。

加载 manual_roi_annotate.py 保存的掩码，对实时深度图采样生成点云。
左侧显示 RGB + ROI 框 + 采样点反投影，右侧显示 3D 点云。

用法::

    conda activate gx_va_deploy
    cd /home/gxai/Desktop/CZR/gx-VA-isaaclab
    python scripts/deploy/manual_roi_pointcloud.py
    python scripts/deploy/manual_roi_pointcloud.py --mask roi_mask.json --m-obj 128
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
for _p in (_PROJECT_DIR, os.path.join(_SCRIPTS_DIR, "camera")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

import cv2
import numpy as np
import torch


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mask", default="roi_mask.json",
                   help="ROI 掩码 JSON 文件路径")
    p.add_argument("--frame", choices=("camera", "base"), default="camera",
                   help="点云坐标系。base 用仿真那套未标定外参")
    p.add_argument("--m-obj", type=int, default=None,
                   help="采样点数，默认取仿真的 M_OBJ")
    p.add_argument("--noise-std", type=float, default=0.0,
                   help="高斯噪声（米）")
    return p.parse_args()


def build_pointcloud(mask, depth, K, args, m_obj):
    """掩码 + 深度 → 点云。复用仿真管线。"""
    from configs.point_bridge_pointcloud import mask_depth_to_pointcloud

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mask_t = torch.from_numpy(mask).to(dev).unsqueeze(0)
    depth_t = torch.from_numpy(depth).float().to(dev).unsqueeze(0)
    K_t = torch.from_numpy(K).float().to(dev).unsqueeze(0)

    if args.frame == "camera":
        cam_pos = torch.zeros(1, 3, device=dev)
        cam_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev)
        ws_z, ws_r = None, None
    else:
        from configs.xarm7_pick_pointcloud_env_cfg import (
            FIXED_CAMERA_POSITION_M, FIXED_CAMERA_QUATERNION_WXYZ,
        )
        from configs.point_bridge_pointcloud import (
            WORKSPACE_Z_RANGE_M, WORKSPACE_RADIUS_XY_M,
        )
        cam_pos = torch.tensor([FIXED_CAMERA_POSITION_M], device=dev, dtype=torch.float32)
        cam_quat = torch.tensor([FIXED_CAMERA_QUATERNION_WXYZ], device=dev, dtype=torch.float32)
        ws_z, ws_r = WORKSPACE_Z_RANGE_M, WORKSPACE_RADIUS_XY_M

    pts, counts = mask_depth_to_pointcloud(
        mask=mask_t, depth=depth_t, K=K_t,
        cam_pos=cam_pos, cam_quat=cam_quat,
        m_obj=m_obj, noise_std=args.noise_std,
        workspace_z_range_m=ws_z, workspace_radius_xy_m=ws_r,
    )
    return pts[0].cpu().numpy(), int(counts[0])


def project_points_to_image(pts, K):
    """将 3D 点投影回图像坐标。"""
    valid = pts[:, 2] > 0
    pts_valid = pts[valid]
    if len(pts_valid) == 0:
        return np.empty((0, 2), dtype=int)

    xy_norm = pts_valid[:, :2] / pts_valid[:, 2:3]
    ones = np.ones((len(xy_norm), 1))
    homo = np.hstack([xy_norm, ones])
    uv_homo = (K @ homo.T).T
    uv = uv_homo[:, :2].astype(int)
    return uv


def render_split_view(rgb, roi, polygon, pts, counts, K, fps, m_obj, frame_name):
    """左右分栏：左=RGB+ROI框+采样点反投影，右=3D点云。"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    H, W = rgb.shape[:2]
    overlay = rgb.copy()

    # 绘制多边形或矩形
    if polygon is not None:
        pts_poly = np.array(polygon, dtype=np.int32)
        cv2.polylines(overlay, [pts_poly], True, (0, 255, 0), 2)
        cv2.fillPoly(overlay, [pts_poly], (0, 255, 0))
        overlay = cv2.addWeighted(rgb, 0.7, overlay, 0.3, 0)
    elif roi is not None:
        x1, y1, x2, y2 = roi
        cv2.rectangle(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(overlay, f"ROI: {x2-x1}x{y2-y1}", (x1, y1 - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # 反投影采样点回图像
    if pts is not None and len(pts) > 0 and frame_name == "camera":
        uv = project_points_to_image(pts, K)
        for u, v in uv:
            if 0 <= u < W and 0 <= v < H:
                cv2.circle(overlay, (u, v), 2, (255, 0, 0), -1)

    # 统计信息
    cv2.putText(overlay, f"FPS: {fps:.1f}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(overlay, f"Points: {counts}/{m_obj}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    # 右栏：3D 点云
    fig = plt.Figure(figsize=(6, 4.5), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    # 固定显示范围（不随点云变化）
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_zlim(0, 0.6)

    # 绘制原点（红色球）
    ax.scatter([0], [0], [0], c='red', s=100, marker='o', edgecolors='darkred', linewidths=2)

    # 绘制原点平面（XY平面，Z=0）
    plane_size = 0.3
    xx, yy = np.meshgrid(np.linspace(-plane_size, plane_size, 10),
                         np.linspace(-plane_size, plane_size, 10))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray', edgecolor='none')

    # 绘制点云（如果有）
    if pts is not None and len(pts) > 0:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 2], cmap="viridis", s=12)

    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(f"Point Cloud ({frame_name} frame)")

    # 渲染到 numpy
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    plot_img = buf.reshape(canvas.get_width_height()[::-1] + (4,))[:, :, :3]
    plt.close(fig)

    # 调整尺寸
    plot_h, plot_w = plot_img.shape[:2]
    if plot_h != H:
        plot_img = cv2.resize(plot_img, (int(plot_w * H / plot_h), H))

    combined = np.hstack([overlay, plot_img])
    return combined


def show_interactive_3d(pts, frame_name):
    """打开独立的交互式 3D 窗口（可旋转、缩放）。"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 固定显示范围（与实时视图一致）
    ax.set_xlim(-0.3, 0.3)
    ax.set_ylim(-0.3, 0.3)
    ax.set_zlim(0, 0.6)

    # 绘制原点（红色球）
    ax.scatter([0], [0], [0], c='red', s=100, marker='o', edgecolors='darkred', linewidths=2)

    # 绘制原点平面（XY平面，Z=0）
    plane_size = 0.3
    xx, yy = np.meshgrid(np.linspace(-plane_size, plane_size, 10),
                         np.linspace(-plane_size, plane_size, 10))
    zz = np.zeros_like(xx)
    ax.plot_surface(xx, yy, zz, alpha=0.2, color='gray', edgecolor='none')

    # 绘制点云
    ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 2], cmap="viridis", s=12)

    ax.set_xlabel("X (m)")
    ax.set_ylabel("Y (m)")
    ax.set_zlabel("Z (m)")
    ax.set_title(f"Interactive Point Cloud ({frame_name} frame)\n可旋转、缩放")

    plt.show()


def main() -> int:
    args = parse_args()

    from configs.point_bridge_pointcloud import M_OBJ
    m_obj = args.m_obj if args.m_obj is not None else M_OBJ

    # 加载掩码和内参
    mask_path = Path(args.mask)
    if not mask_path.exists():
        print(f"[错误] 掩码文件不存在: {mask_path}")
        print(f"[提示] 先运行 manual_roi_annotate.py 生成掩码")
        return 1

    with open(mask_path, "r") as f:
        mask_data = json.load(f)

    # 兼容新旧格式：多边形或矩形
    polygon = mask_data.get("polygon")  # [[x, y], ...] or None
    roi = mask_data.get("bbox", mask_data.get("roi"))  # [x1, y1, x2, y2]
    K = np.array([
        [mask_data["intrinsics"]["fx"], 0, mask_data["intrinsics"]["cx"]],
        [0, mask_data["intrinsics"]["fy"], mask_data["intrinsics"]["cy"]],
        [0, 0, 1]
    ])
    serial = mask_data["serial"]
    width, height = mask_data["resolution"]

    # 加载掩码图像
    mask_img_path = mask_path.with_suffix(".png")
    if not mask_img_path.exists():
        print(f"[错误] 掩码图像不存在: {mask_img_path}")
        return 1
    mask_img = cv2.imread(str(mask_img_path), cv2.IMREAD_GRAYSCALE)
    mask = mask_img > 127  # bool array

    print(f"[掩码] ROI: {roi}, 像素数: {mask.sum()}")
    print(f"[内参] fx={K[0,0]:.1f} cx={K[0,2]:.1f} cy={K[1,2]:.1f}")

    # 启动相机
    from realsense_camera import RealsenseCamera
    cam = RealsenseCamera(width=width, height=height, fps=30,
                          align_on_device=True, serial_number=serial)
    cam.start()

    window_name = "Manual ROI Point Cloud (Press 'q' to quit, 'p' for interactive 3D)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 480)

    frame_times = []
    latest_pts = None  # 保存最新点云用于交互窗口

    try:
        print("[运行] 实时点云可视化已启动，按 'q' 退出，'p' 打开交互 3D 窗口")
        while True:
            t0 = time.time()

            rgb, depth = cam.get_frame()
            if rgb is None:
                print("[相机] 取帧失败，重试")
                continue

            # 点云采样
            pts, counts = build_pointcloud(mask, depth, K, args, m_obj)
            latest_pts = pts  # 保存用于交互窗口

            # FPS 统计
            frame_times.append(time.time() - t0)
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps = len(frame_times) / sum(frame_times) if frame_times else 0.0

            # 渲染
            combined = render_split_view(rgb, roi, polygon, pts, counts, K, fps, m_obj, args.frame)

            # BGR 显示
            cv2.imshow(window_name, combined[..., ::-1])

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('p'):
                # 打开交互式 3D 窗口
                if latest_pts is not None and len(latest_pts) > 0:
                    print("[交互] 打开独立 3D 窗口，关闭后继续实时流...")
                    show_interactive_3d(latest_pts, args.frame)
                else:
                    print("[交互] 当前没有点云数据")

    except KeyboardInterrupt:
        print("\n[退出]")
    finally:
        cam.stop()
        cv2.destroyAllWindows()

    return 0


if __name__ == "__main__":
    sys.exit(main())
