#!/usr/bin/env python3
"""YOLO 分割 + RealSense 深度 → 点云实时可视化。

在 OpenCV 窗口实时显示：
  左侧：RGB + 掩码叠加 + 被选中点的反投影位置（红点）
  右侧：3D 点云散点图（实时刷新）

共享 Point Bridge 的 mask_depth_to_pointcloud，保证与仿真管线一致。

用法::

    conda run -n gx_va_deploy python scripts/deploy/yolo_pointcloud_realtime.py
    conda run -n gx_va_deploy python scripts/deploy/yolo_pointcloud_realtime.py --conf 0.3
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
import torch

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
    p.add_argument("--frame", choices=("camera", "base"), default="camera",
                   help="点云坐标系。base 用仿真那套**未标定**外参，仅看量级。")
    p.add_argument("--m-obj", type=int, default=None,
                   help="采样点数，默认取仿真的 M_OBJ 以保持一致")
    p.add_argument("--noise-std", type=float, default=0.0,
                   help="高斯噪声（米）。目视核对默认 0；仿真训练用 0.01")
    return p.parse_args()


def pick_target(result, erosion_ratio=0.1):
    """从 YOLO 结果里挑出工件掩码。

    Args:
        result: YOLO Results 对象
        erosion_ratio: 掩码收缩比例，0.1 表示沿边界内收 10%

    Returns:
        (mask_best, mask_all, info, best_idx):
            mask_best 为最优实例掩码（H,W bool，用于点云），
            mask_all 为所有实例掩码（H,W bool，用于可视化），
            info 为所有检测信息列表，
            best_idx 为最优实例在 info 中的索引
    """
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return None, None, [], None

    cpu = result.cpu()
    cls = cpu.boxes.cls.int().tolist()
    obj_idx = [i for i, c in enumerate(cls) if c == 0]
    if not obj_idx:
        return None, None, [], None

    info = []
    g = cpu.gongjian
    for i in obj_idx:
        rec = {"index": i, "conf": float(cpu.boxes.conf[i])}
        if g is not None and len(getattr(g, "selection_score", [])) > i:
            rec["selection_score"] = float(g.selection_score[i])
            rec["side"] = g.side[i] if len(g.side) > i else "?"
        info.append(rec)

    # 找出最优实例
    key = "selection_score" if "selection_score" in info[0] else "conf"
    best_idx = max(range(len(info)), key=lambda i: info[i].get(key, 0.0))
    best_i = info[best_idx]["index"]

    # masks.data 是 (N, h, w) 且 h/w 是网络输入尺度，需要缩回原图
    md = cpu.masks.data.numpy()
    H, W = result.orig_shape

    # 所有实例掩码（用于可视化）
    mask_all = np.zeros((H, W), dtype=bool)
    for rec in info:
        i = rec["index"]
        m = md[i]
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        mask_all |= m > 0.5

    # 最优实例掩码（用于点云，需要腐蚀）
    m_best = md[best_i]
    if m_best.shape != (H, W):
        m_best = cv2.resize(m_best.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
    mask_best = m_best > 0.5

    # 形态学腐蚀：掩码内收指定比例
    if erosion_ratio > 0 and mask_best.any():
        area = mask_best.sum()
        equiv_radius = np.sqrt(area / np.pi)
        kernel_size = max(3, int(equiv_radius * erosion_ratio * 2))
        kernel_size = kernel_size + (1 - kernel_size % 2)  # 确保为奇数
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        mask_best = cv2.erode(mask_best.astype(np.uint8), kernel, iterations=1).astype(bool)

    return mask_best, mask_all, info, best_idx


def build_pointcloud(mask, depth, K, args, m_obj):
    """掩码 + 深度 → 点云。**调用与仿真完全相同的共享函数。**"""
    from configs.point_bridge_pointcloud import mask_depth_to_pointcloud

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mask_t = torch.from_numpy(mask).to(dev).unsqueeze(0)              # (1,H,W)
    depth_t = torch.from_numpy(depth).float().to(dev).unsqueeze(0)    # (1,H,W)
    K_t = torch.from_numpy(K).float().to(dev).unsqueeze(0)            # (1,3,3)

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
    """将 3D 点投影回图像坐标（相机光学系 → 像素坐标）。

    Args:
        pts: (N, 3) 相机坐标系下的点 [x, y, z]
        K: (3, 3) 内参矩阵

    Returns:
        uv: (N, 2) 像素坐标 [u, v]，Z<=0 的点会被过滤掉
    """
    valid = pts[:, 2] > 0
    pts_valid = pts[valid]
    if len(pts_valid) == 0:
        return np.empty((0, 2), dtype=int)

    # [x, y, z] → [u, v, 1] = K @ [x/z, y/z, 1]
    xy_norm = pts_valid[:, :2] / pts_valid[:, 2:3]  # (N, 2)
    ones = np.ones((len(xy_norm), 1))
    homo = np.hstack([xy_norm, ones])  # (N, 3)
    uv_homo = (K @ homo.T).T  # (N, 3)
    uv = uv_homo[:, :2].astype(int)
    return uv


def draw_all_detections(overlay, result, info, best_idx):
    """在 overlay 上绘制所有检测实例的 bbox 和标签。

    非最优实例：蓝色框，灰色标签
    最优实例：绿色框+粗线，白底绿字标签
    """
    if not info or result.boxes is None:
        return

    cpu = result.cpu()
    md = cpu.masks.data.numpy()
    H, W = overlay.shape[:2]

    for rank, rec in enumerate(info):
        i = rec["index"]
        is_best = (rank == best_idx)

        # 掩码半透明叠加
        m = md[i]
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        mask_i = m > 0.5
        if is_best:
            # 最优：绿色叠加
            overlay[mask_i] = (0.4 * overlay[mask_i] + 0.6 * np.array([40, 220, 40])).astype(np.uint8)
        else:
            # 其他：蓝色叠加，透明度低
            overlay[mask_i] = (0.7 * overlay[mask_i] + 0.3 * np.array([40, 40, 220])).astype(np.uint8)

        # Bounding box
        box = cpu.boxes.xyxy[i].tolist()
        x1, y1, x2, y2 = int(box[0]), int(box[1]), int(box[2]), int(box[3])
        color = (0, 220, 0) if is_best else (180, 100, 0)   # RGB
        thickness = 3 if is_best else 1
        cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)

        # 标签
        score_val = rec.get("selection_score", rec["conf"])
        side_str = f" {rec['side']}" if "side" in rec else ""
        label = f"★ score={score_val:.3f}{side_str}" if is_best else f"score={score_val:.3f}{side_str}"
        font_scale = 0.5
        font = cv2.FONT_HERSHEY_SIMPLEX
        (tw, th), baseline = cv2.getTextSize(label, font, font_scale, 1)
        ty = max(y1 - 6, th + 4)
        if is_best:
            cv2.rectangle(overlay, (x1, ty - th - 4), (x1 + tw + 4, ty + baseline), (0, 220, 0), -1)
            cv2.putText(overlay, label, (x1 + 2, ty - 2), font, font_scale, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            cv2.putText(overlay, label, (x1 + 2, ty - 2), font, font_scale, (180, 180, 255), 1, cv2.LINE_AA)


def render_split_view(rgb, result, mask_best, pts, counts, K, info, best_idx, fps, m_obj, frame_name):
    """左右分栏：左=RGB+所有检测框（最优高亮）+采样点反投影，右=3D点云。"""
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    H, W = rgb.shape[:2]

    # 左栏：先画所有实例的掩码和框，再叠加最优实例的采样点反投影
    overlay = rgb.copy()
    if result is not None and info:
        draw_all_detections(overlay, result, info, best_idx)

    # 反投影采样点回图像（只在相机系下有意义，只显示最优实例的点）
    if pts is not None and len(pts) > 0 and frame_name == "camera":
        uv = project_points_to_image(pts, K)
        for u, v in uv:
            if 0 <= u < W and 0 <= v < H:
                cv2.circle(overlay, (u, v), 2, (0, 255, 0), -1)

    # 叠加 FPS 和统计信息
    n_det = len(info) if info else 0
    cv2.putText(overlay, f"FPS: {fps:.1f}  Det: {n_det}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(overlay, f"Points: {counts}/{m_obj}", (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    if info and best_idx is not None:
        best = info[best_idx]
        txt = f"best conf={best['conf']:.3f}"
        if "selection_score" in best:
            txt += f" score={best['selection_score']:.3f} {best.get('side', '')}"
        cv2.putText(overlay, txt, (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 255, 200), 1, cv2.LINE_AA)

    # 右栏：matplotlib 3D 点云
    fig = plt.Figure(figsize=(6, 4.5), dpi=100)
    ax = fig.add_subplot(111, projection="3d")

    if pts is not None and len(pts) > 0:
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 2], cmap="viridis", s=12)
        c = pts.mean(axis=0)
        r = max(np.ptp(pts, axis=0).max() * 0.5, 0.01)
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(f"Point Cloud ({frame_name} frame)")

    # 渲染 matplotlib 到 numpy 数组
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    canvas = FigureCanvasAgg(fig)
    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
    plot_img = buf.reshape(canvas.get_width_height()[::-1] + (4,))[:, :, :3]  # drop alpha
    plt.close(fig)

    # 调整尺寸使两栏高度一致
    plot_h, plot_w = plot_img.shape[:2]
    if plot_h != H:
        plot_img = cv2.resize(plot_img, (int(plot_w * H / plot_h), H))

    # 水平拼接
    combined = np.hstack([overlay, plot_img])
    return combined


def main() -> int:
    args = parse_args()

    from configs.point_bridge_pointcloud import M_OBJ
    m_obj = args.m_obj if args.m_obj is not None else M_OBJ

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

    window_name = "YOLO Point Cloud Real-time (Press 'q' to quit)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, 1600, 480)

    frame_times = []

    try:
        print("[运行] 实时点云可视化已启动，按 'q' 退出")
        while True:
            t0 = time.time()

            rgb, depth = cam.get_frame()
            if rgb is None:
                print("[相机] 取帧失败，重试")
                continue

            # ultralytics 的 preprocess() 期望 BGR 输入，会自动翻转成 RGB。
            # RealsenseCamera 返回 RGB，所以要先转回 BGR。
            res = model.predict(rgb[..., ::-1], imgsz=args.imgsz, conf=args.conf,
                                device=args.device, verbose=False)[0]
            # mask_best: 最优实例掩码（用于点云采集，已腐蚀）
            # mask_all:  所有实例掩码（用于可视化显示）
            # info:      所有实例的检测信息列表
            # best_idx:  最优实例在 info 中的索引
            mask_best, mask_all, info, best_idx = pick_target(res)

            if mask_best is None:
                pts, counts = None, 0
            else:
                # 点云只采最优实例的区域
                pts, counts = build_pointcloud(mask_best, depth, K, args, m_obj)

            # FPS 统计
            frame_times.append(time.time() - t0)
            if len(frame_times) > 30:
                frame_times.pop(0)
            fps = len(frame_times) / sum(frame_times) if frame_times else 0.0

            # 渲染分栏视图：传入原始 result 以便绘制所有实例
            combined = render_split_view(rgb, res, mask_best, pts, counts, K,
                                         info, best_idx, fps, m_obj, args.frame)

            # BGR 转换后显示（OpenCV 用 BGR）
            cv2.imshow(window_name, combined[..., ::-1])

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
