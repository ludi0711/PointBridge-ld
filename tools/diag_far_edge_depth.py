#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实机 D435 远端边深度丢失诊断 —— 验证"部署点云向相机偏置"的根因。

背景
====
训练用仿真完美深度，工件远端轮廓也能采到点；部署用 D435 真实深度，远端边因为
掠射角 + 边缘空洞，深度大量返回 0。FPS（最远点采样）又偏爱挑极端点，于是部署点云
在远端被截断、质心向相机偏。本脚本在**像素级 + 点云级**把这个机制坐实，并量出
"远端缺失带有多宽"，直接标定训练侧该加的远边腐蚀参数。

设计原则
========
- **不修改任何现有代码**：相机、掩码、内参、外参、反投影全部 import 复用
  ``scripts/camera/*`` 与 ``configs/point_bridge_pointcloud.py``（与训练/部署同源）。
- 反投影**不自己写**，只调共享 ``mask_depth_to_pointcloud``——否则证明不了
  "真机路径 == 训练路径"，而这是唯一值得测的事。
- 深度语义沿用既有约定：z-depth（distance_to_image_plane），``0 == 无效``。

分阶段（交互菜单 / 也可 --run_all 一键跑）
==========================================
  [0] 相机就绪 + 内参/外参对照   起相机、实机内参 vs 仿真 canonical、外参核对
  [1] 标注工件 ROI              鼠标手标多边形（或 --polygon 复用）
  [2] 单帧远边诊断（核心）       近端 vs 远端空洞率对照 + 远端内收量(带宽 mm)
  [3] 稳定性采集（N 帧）         远端缺失是否稳定出现（系统性 vs 随机）
  [4] 点云级验证 + 汇总报告      反投影点云远端截断/质心偏置 + 最终训练建议

运行环境（必须 gx_va_deploy）：既要 pyrealsense2 也要 isaaclab。

    conda activate gx_va_deploy
    python tools/diag_far_edge_depth.py                 # 交互菜单
    python tools/diag_far_edge_depth.py --run_all \
        --polygon "310,220 380,225 385,275 305,270" --frames 30   # 无头一键
    python tools/diag_far_edge_depth.py --selftest       # 不连相机，验证几何逻辑

结果文件默认写到 /root/autodl-tmp/far_edge_diag/（--out 可改）。
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_CURRENT_DIR)
sys.path.insert(0, _PROJECT_DIR)

# 结果默认落盘目录（要求：/root/autodl-tmp 内新建文件夹）。
DEFAULT_OUT = "/home/gxai/Desktop/ld/PointBridge/gx-VA-isaaclab_new/tools/logs/occ_close"

# 判定阈值。
RECESSION_SIGNIFICANT_MM = 1.0    # 远端内收超过 1mm 才算"有意义的远边缺失"
FAR_NEAR_RATIO_SIGNIFICANT = 2.0  # 远/近空洞率比值超过 2 才判"单侧"

# =============================================================================
# 纯 numpy 几何工具（不依赖 isaaclab / pyrealsense2，可离线 selftest）
# 约定与 real_roi_pointcloud.project_to_pixels 完全一致：
#   R = cam->base 旋转矩阵；base->cam 用 (P - cam_pos) @ R；cam->base 用 p @ R.T + cam_pos
# =============================================================================


def unproject_pixels_to_base(uv, z, K, cam_pos, R):
    """像素 (N,2) + z-depth (N,) → 基座系 (N,3)。

    z-depth = 到成像平面的距离（不是到光心欧氏距离），与训练一致。
    """
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    X = (uv[:, 0] - cx) / fx * z
    Y = (uv[:, 1] - cy) / fy * z
    Z = z
    cam = np.stack([X, Y, Z], axis=1)
    return cam @ R.T + cam_pos


def project_base_to_pixels(P, K, cam_pos, R):
    """基座系 (N,3) → 像素 (N,2)。是 unproject 的逆运算。"""
    cam = (P - cam_pos) @ R
    fx, fy, cx, cy = float(K[0, 0]), float(K[1, 1]), float(K[0, 2]), float(K[1, 2])
    Z = np.clip(cam[:, 2], 1e-6, None)
    u = fx * cam[:, 0] / Z + cx
    v = fy * cam[:, 1] / Z + cy
    return np.stack([u, v], axis=1)


def compute_far_axis(depth, mask, K, cam_pos, R):
    """求"远离相机"在图像里的方向 (du, dv)，以及工件锚点。

    方法：用外参直接计算，不依赖深度梯度。
    ─────────────────────────────────────────────────────────────────────
    深度梯度方向容易被相机俯仰角主导（y 方向梯度始终比 x 大），导致工件
    在桌面内旋转时 du/dv 仍锁定在 (0, -1)，远近分半方向判断错误。

    正确做法：
      1. 从掩码中位深度反投影出工件质心在基座系的 3D 坐标 P_obj。
      2. 从 P_obj 指向 cam_pos 的向量就是"3D 空间里指向相机"的方向。
      3. 把该方向投影到桌面（去掉基座系 Z 分量），取反得到"桌面内远离
         相机"的方向 d3_flat。这与工件在桌面上怎么旋转无关。
      4. 把 d3_flat 从基座系变换回相机系，再投影到图像平面，得到
         图像里的 (du, dv)。

    consistency 改为验证指标：把 (du, dv) 投回 3D 再计算与 d3_flat
    的夹角余弦，接近 1 说明外参和深度数据一致。
    """
    valid = mask & np.isfinite(depth) & (depth > 0)
    vy, vx = np.nonzero(valid)
    if vy.size == 0:
        return None
    d_med = float(np.median(depth[valid]))
    cu, cv = float(vx.mean()), float(vy.mean())

    # 工件质心在基座系的 3D 坐标
    P_obj = unproject_pixels_to_base(
        np.array([[cu, cv]]), np.array([d_med]), K, cam_pos, R
    )[0]

    # 基座系内"工件质心 → 相机"方向
    d_to_cam = np.asarray(cam_pos, dtype=np.float64) - P_obj
    # 投影到桌面（去掉 Z 分量），"远离相机"取反
    d_flat = -np.array([d_to_cam[0], d_to_cam[1], 0.0], dtype=np.float64)
    n_flat = float(np.linalg.norm(d_flat))
    if n_flat < 1e-6:
        # 工件正好在相机正下方，桌面内方向退化；回退到深度梯度
        d_flat = -d_to_cam
        n_flat = float(np.linalg.norm(d_flat)) or 1.0
    d_flat /= n_flat

    # 把基座系方向转到相机系：cam = (P - cam_pos) @ R，方向只用 R
    d_cam = d_flat @ R          # shape (3,)

    # 投影到图像平面：忽略 Z 分量，用 fx/fy 缩放
    fx, fy = float(K[0, 0]), float(K[1, 1])
    z_cam = d_cam[2]
    if abs(z_cam) < 1e-6:
        # 方向几乎平行于图像平面，退化情况
        du_raw, dv_raw = d_cam[0] * fx, d_cam[1] * fy
    else:
        # 透视投影：(x/z)*fx, (y/z)*fy
        du_raw = d_cam[0] / z_cam * fx
        dv_raw = d_cam[1] / z_cam * fy
    mag = float(np.hypot(du_raw, dv_raw)) or 1.0
    du, dv = du_raw / mag, dv_raw / mag

    # consistency：(du, dv) 方向与 d3_flat 在图像上投影的余弦
    d3 = P_obj - np.asarray(cam_pos, dtype=np.float64)
    n3 = float(np.linalg.norm(d3)) or 1.0
    d3 = d3 / n3
    consistency = float(abs(du * (-d_flat[0]) + dv * (-d_flat[1])))  # 近似一致性

    return {"cu": cu, "cv": cv, "d_med": d_med, "P_obj": P_obj,
            "d3": d3, "du": du, "dv": dv, "consistency": consistency}


def split_near_far(mask, axis):
    """按远轴投影正负把掩码切成近半 / 远半。返回 (mask_near, mask_far)。"""
    cu, cv, du, dv = axis["cu"], axis["cv"], axis["du"], axis["dv"]
    yy, xx = np.nonzero(mask)
    s = (xx - cu) * du + (yy - cv) * dv
    near = np.zeros_like(mask)
    far = np.zeros_like(mask)
    near[yy[s <= 0], xx[s <= 0]] = True
    far[yy[s > 0], xx[s > 0]] = True
    return near, far


def half_hole_ratio(depth, mask_half):
    """半区空洞率 = 1 - 有效像素/掩码像素。返回 (hole_ratio, n_valid, n_total)。"""
    valid = mask_half & np.isfinite(depth) & (depth > 0)
    n_valid = int(valid.sum())
    n_total = int(mask_half.sum())
    ratio = (1.0 - n_valid / n_total) if n_total else 1.0
    return ratio, n_valid, n_total


def recession_width(mask, valid, axis, K):
    """远端内收量（像素 + mm）。

    多边形是"真实边界"，有效掩码是"实际测到深度的边界"。远边丢深度时，有效掩码的
    远边界会内收进多边形远边界以内，内收宽度即缺失带宽度。

    多边形边界取 max（沿远轴的最远像素就是手标轮廓的远端尖点，即真实边界；
    用 98% 分位会因远端尖帽像素少而系统性内缩，把内收量读小）。有效边界取
    98% 分位（D435 边缘偶有飞点把有效深度溅出真实边界之外，98% 分位免疫）。

    返回 (recession_px, recession_mm, tip_gap_px) 或 None（无有效像素时）。
    tip_gap_px = max(s_poly) - p98(s_poly)，>8px 时提示多边形远端可能标出界。
    """
    vy, vx = np.nonzero(valid)
    if vy.size == 0:
        return None
    cu, cv, du, dv = axis["cu"], axis["cv"], axis["du"], axis["dv"]
    yy, xx = np.nonzero(mask)
    s_poly = (xx - cu) * du + (yy - cv) * dv
    s_val = (vx - cu) * du + (vy - cv) * dv
    tip = float(s_poly.max())
    p98_poly = float(np.percentile(s_poly, 98))
    p98_val = float(np.percentile(s_val, 98))
    rec_px = tip - p98_val
    fx = float(K[0, 0])
    pix_scale_m = axis["d_med"] / fx          # 该深度处 1px ≈ 多少米
    rec_mm = rec_px * pix_scale_m * 1000.0
    return rec_px, rec_mm, tip - p98_poly


def depth_bin_profile(depth, mask, n_bins=10):
    """有效像素按深度等分箱（近→远），空洞按最近有效像素深度归箱，算每箱空洞率。

    这是"外参无关"的交叉验证：远 = 深度更大。空洞 depth=0 无法直接分箱，用
    scipy cKDTree 找最近有效像素借深度。scipy 缺失时退化（只给有效像素剖面）。
    """
    valid = mask & np.isfinite(depth) & (depth > 0)
    vy, vx = np.nonzero(valid)
    if vy.size == 0:
        return None
    dval = depth[vy, vx]
    edges = np.quantile(dval, np.linspace(0.0, 1.0, n_bins + 1))
    in_edges = edges[1:-1]
    vbin = np.digitize(dval, in_edges)                 # 0..n_bins-1

    hbin = None
    holes = mask & ~valid
    if holes.any():
        try:
            from scipy.spatial import cKDTree
            hy, hx = np.nonzero(holes)
            tree = cKDTree(np.stack([vy, vx], axis=1))
            _, idx = tree.query(np.stack([hy, hx], axis=1), k=1)
            hdepth = depth[vy[idx], vx[idx]]
            hbin = np.digitize(hdepth, in_edges)
        except Exception:
            hbin = None

    rows = []
    for i in range(n_bins):
        nv = int((vbin == i).sum())
        nh = int((hbin == i).sum()) if hbin is not None else 0
        rows.append({"bin": i, "depth_lo": float(edges[i]), "depth_hi": float(edges[i + 1]),
                     "valid": nv, "hole": nh,
                     "hole_ratio": nh / (nv + nh) if (nv + nh) else 0.0})
    return rows


def hole_centroid_offset(mask, valid):
    """空洞质心相对掩码质心的偏移（px），量"空洞是否单侧聚集"。"""
    hy, hx = np.nonzero(mask & ~valid)
    py, px = np.nonzero(mask)
    if hy.size == 0 or py.size == 0:
        return None
    hc = np.array([hx.mean(), hy.mean()])
    pc = np.array([px.mean(), py.mean()])
    off = hc - pc
    return {"offset_px": off.tolist(),
            "mag_px": float(np.linalg.norm(off))}


# =============================================================================
# 命令行参数
# =============================================================================


def parse_args():
    p = argparse.ArgumentParser(description="实机 D435 远端边深度丢失诊断（分阶段交互）")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--serial", type=str, default=None, help="多相机时指定序列号")
    p.add_argument("--warmup", type=int, default=30, help="丢弃前 N 帧（自动曝光/滤波收敛）")
    p.add_argument("--polygon", type=str, default=None,
                   help="复用已标多边形：\"x,y x,y ...\" 或 meta.json 路径；不给则现场手标")
    p.add_argument("--depth_range", type=float, nargs=2, default=(0.3, 1.5),
                   metavar=("LO", "HI"), help="深度伪彩固定色标区间(米)")
    p.add_argument("--bins", type=int, default=10, help="S2 深度分箱数")
    p.add_argument("--frames", type=int, default=30, help="S3 稳定性采集帧数")
    p.add_argument("--m_obj", type=int, default=64)
    p.add_argument("--noise_std", type=float, default=0.0,
                   help="S4 点云反投影的高斯噪声；默认 0 便于核对几何")
    p.add_argument("--intrinsics", type=str, default="real", choices=("real", "sim"),
                   help="real=实机实测内参；sim=仿真 canonical（隔离内参差异用）")
    p.add_argument("--no_workspace_filter", action="store_true",
                   help="S4 关掉工作空间裁剪，看原始分布")
    p.add_argument("--no_window", action="store_true", help="不弹窗（SSH/无显示器）")
    p.add_argument("--run_all", action="store_true", help="一键顺序跑 S0→S4（需 --polygon）")
    p.add_argument("--selftest", action="store_true", help="不连相机，离线验证几何逻辑")
    p.add_argument("--out", type=str, default=DEFAULT_OUT,
                   help=f"结果输出目录（默认 {DEFAULT_OUT}）")
    return p.parse_args()


# =============================================================================
# 打印辅助
# =============================================================================


def _sec(title):
    print("\n" + "=" * 78)
    print("  " + title)
    print("=" * 78)


def _sub(title):
    print("\n── " + title + " " + "-" * max(0, 66 - len(title)))


# =============================================================================
# 会话与各阶段
# =============================================================================


class Session:
    """跨阶段共享的会话状态。"""

    def __init__(self, args):
        self.args = args
        self.cam = None
        self.K = None          # 反投影用内参 (3,3) float64
        self.R = None          # cam->base 旋转矩阵 (3,3)
        self.intr = None       # 实机内参 dict
        self.polygon = None
        self.poly_arr = None
        self.mask = None       # bool (H, W)
        self.last_rgb = None
        self.last_depth = None
        self.axis = None       # S2 算出的远轴（供 S4 复用）
        self.results = {}      # 各阶段结果汇总


def _load_deps():
    """懒加载相机/掩码/反投影依赖（真机 gx_va_deploy 环境才装）。"""
    from scripts.camera.realsense_camera import RealsenseCamera
    from scripts.camera.view_real_depth import (
        SIM_CANONICAL, MIN_MASK_PIXELS, colorize_depth, select_polygon,
        polygon_to_mask, parse_polygon_arg, quat_to_matrix, report_intrinsics,
        roi_depth_stats, polygon_bbox,
    )
    from scripts.camera.real_roi_pointcloud import (
        CALIB_CAM_POS_IN_BASE_M, CALIB_CAM_QUAT_IN_BASE_WXYZ, _verify_extrinsics,
    )
    return dict(
        RealsenseCamera=RealsenseCamera, SIM_CANONICAL=SIM_CANONICAL,
        MIN_MASK_PIXELS=MIN_MASK_PIXELS, colorize_depth=colorize_depth,
        select_polygon=select_polygon, polygon_to_mask=polygon_to_mask,
        parse_polygon_arg=parse_polygon_arg, quat_to_matrix=quat_to_matrix,
        report_intrinsics=report_intrinsics, roi_depth_stats=roi_depth_stats,
        polygon_bbox=polygon_bbox, CALIB_CAM_POS_IN_BASE_M=CALIB_CAM_POS_IN_BASE_M,
        CALIB_CAM_QUAT_IN_BASE_WXYZ=CALIB_CAM_QUAT_IN_BASE_WXYZ,
        _verify_extrinsics=_verify_extrinsics,
    )


def stage0_camera(sess, dep):
    """[0] 相机就绪 + 内参/外参对照。"""
    _sec("[S0] 相机就绪 + 内参/外参对照")
    dep["_verify_extrinsics"]()
    args = sess.args
    sess.cam = dep["RealsenseCamera"](width=args.width, height=args.height, fps=args.fps,
                                      align_on_device=True, serial_number=args.serial)
    sess.cam.start()
    sess.intr = dep["report_intrinsics"](sess.cam)
    if args.intrinsics == "sim":
        sc = dep["SIM_CANONICAL"]
        sess.K = np.array([[sc["fx"], 0, sc["cx"]],
                           [0, sc["fy"], sc["cy"]],
                           [0, 0, 1]], dtype=np.float64)
        print("[内参] 强制使用仿真 canonical")
    else:
        sess.K = np.asarray(sess.cam.K_color, dtype=np.float64)
    sess.R = dep["quat_to_matrix"](dep["CALIB_CAM_QUAT_IN_BASE_WXYZ"])
    print(f"[外参] 基座系 pos={dep['CALIB_CAM_POS_IN_BASE_M']}")
    print(f"[预热] 丢弃前 {args.warmup} 帧...")
    for _ in range(args.warmup):
        sess.cam.get_frame()
    print("[S0] 完成")


def stage1_annotate(sess, dep):
    """[1] 标注工件 ROI。"""
    _sec("[S1] 标注工件 ROI")
    args = sess.args
    polygon = dep["parse_polygon_arg"](args.polygon) if args.polygon else None
    if polygon is None:
        if args.no_window:
            raise SystemExit("[S1] 无窗口环境请用 --polygon \"x,y x,y ...\" 指定")
        print("\n左键沿工件轮廓点一圈，回车确认（>=3 点）。")
        polygon = dep["select_polygon"](sess.cam)
        if polygon is None:
            raise SystemExit("[S1] 未标注有效多边形，退出")
        spec = " ".join(f"{x},{y}" for x, y in polygon)
        print(f"[手标] {len(polygon)} 点  (复用: --polygon \"{spec}\")")
    else:
        print(f"[掩码] 复用已存多边形（{len(polygon)} 点）")
    sess.polygon = polygon
    sess.poly_arr = np.asarray(polygon, dtype=np.int32)
    sess.mask = dep["polygon_to_mask"](polygon, args.height, args.width)
    print(f"[掩码] 多边形面积 {int(sess.mask.sum())} px")
    if int(sess.mask.sum()) < dep["MIN_MASK_PIXELS"]:
        print(f"  [警告] 面积 < {dep['MIN_MASK_PIXELS']} px，采样会大量重复，"
              f"建议多边形圈大一点或工件靠近相机")


def stage2_diagnose(sess, dep):
    """[2] 单帧远边诊断（核心）：近端 vs 远端对照 + 远端内收量。"""
    _sec("[S2] 单帧远边诊断（近端 vs 远端对照）")
    if sess.mask is None:
        print("[S2] 需要先做 S1 标注；自动先跑 S1...")
        stage1_annotate(sess, dep)
    rgb, depth = sess.cam.get_frame()
    if rgb is None or depth is None:
        print("[S2] 取帧失败")
        return
    sess.last_rgb, sess.last_depth = rgb, depth
    args = sess.args
    mask, K, R = sess.mask, sess.K, sess.R
    cam_pos = np.asarray(dep["CALIB_CAM_POS_IN_BASE_M"], dtype=np.float64)

    valid = mask & np.isfinite(depth) & (depth > 0)

    # ── 整体统计 ──
    stats = dep["roi_depth_stats"](depth, mask)
    _sub("整体深度统计")
    print(f"  掩码 {stats['n_total']} px   有效 {stats['n_valid']} px   "
          f"空洞率 {100 * stats['hole_ratio']:.1f}%")
    if stats["n_valid"] == 0:
        print("[S2] 掩码内无有效深度 —— 多边形可能圈到桌面外，或相机/工件距离异常。")
        return
    print(f"  深度  min {stats['min']:.3f}  median {stats['median']:.3f}  "
          f"max {stats['max']:.3f} m   std {stats['std']:.4f} m")

    # ── 路线 A：外参空间分半 ──
    axis = compute_far_axis(depth, mask, K, cam_pos, R)
    if axis is None:
        print("[S2] 无法计算远轴")
        return
    sess.axis = axis
    near, far = split_near_far(mask, axis)
    hr_near, nv_near, nt_near = half_hole_ratio(depth, near)
    hr_far, nv_far, nt_far = half_hole_ratio(depth, far)
    rec = recession_width(mask, valid, axis, K)

    _sub("路线 A · 外参空间分半（近端 vs 远端）")
    print(f"  远方向（图像）du={axis['du']:+.3f}  dv={axis['dv']:+.3f}   "
          f"中位深度 {axis['d_med']:.3f} m   一致性 {axis.get('consistency', 1.0):.2f}")
    if axis.get("consistency", 1.0) < 0.5:
        print("  [警告] 远方向一致性低（工件正面朝向相机/形状复杂），"
              "路线 A 分半意义有限，以路线 B 为准")
    print(f"  近半: 掩码 {nt_near} px  有效 {nv_near} px  空洞率 {100 * hr_near:.1f}%")
    print(f"  远半: 掩码 {nt_far} px  有效 {nv_far} px  空洞率 {100 * hr_far:.1f}%")
    ratio = (hr_far / hr_near) if hr_near > 1e-9 else float("inf")
    print(f"  远/近空洞率比 = {ratio:.2f}")
    if rec is not None:
        print(f"  远端内收量 = {rec[0]:.1f} px  ≈ {rec[1]:.2f} mm  "
              f"（有效掩码远边界比多边形远边界内收）")
        if rec[2] > 8.0:
            print(f"  [警告] 多边形远端尖帽占比大（max-p98 = {rec[2]:.1f} px），"
                  f"远端可能标出界，内收量被高估")

    # ── 路线 B：深度分箱剖面（外参无关交叉验证）──
    _sub("路线 B · 深度分箱剖面（近→远，外参无关）")
    rows = depth_bin_profile(depth, mask, args.bins)
    if rows is not None:
        print(f"  {'箱':>3} {'深度区间(m)':>18} {'有效':>6} {'空洞':>6} {'空洞率':>8}")
        for r in rows:
            print(f"  {r['bin']:>3} {r['depth_lo']:>7.3f}~{r['depth_hi']:<7.3f} "
                  f"{r['valid']:>6} {r['hole']:>6} {100 * r['hole_ratio']:>7.1f}%")
        far_ratio = rows[-1]["hole_ratio"]
        near_ratio = rows[0]["hole_ratio"]
        print(f"  最远箱空洞率 {100 * far_ratio:.1f}%  vs 最近箱 {100 * near_ratio:.1f}%")

    off = hole_centroid_offset(mask, valid)
    if off is not None:
        print(f"  空洞质心偏移 = {off['mag_px']:.1f} px  "
              f"方向 {off['offset_px']}（大 = 空洞单侧聚集，非均匀绕边）")

    # ── 判定 ──
    _sub("S2 判定")
    rec_mm = rec[1] if rec is not None else 0.0
    if hr_far > FAR_NEAR_RATIO_SIGNIFICANT * hr_near and rec_mm > RECESSION_SIGNIFICANT_MM:
        verdict = (f"单侧远边深度缺失：远半空洞率 {100 * hr_far:.1f}% ≫ 近半 "
                   f"{100 * hr_near:.1f}%，远端内收 {rec_mm:.2f} mm。\n"
                   f"    → 训练应做【确定性远边腐蚀】，band ≈ {rec_mm:.1f} mm")
    elif (hr_far > 0.05 and hr_near > 0.05
          and abs(hr_far - hr_near) < 0.5 * max(hr_far, hr_near)):
        verdict = (f"轮廓边缘空洞（近远接近对称）：近 {100 * hr_near:.1f}% / 远 "
                   f"{100 * hr_far:.1f}%。\n    → 训练应做【轮廓边缘腐蚀（对称）】")
    else:
        verdict = ("未见明显远边缺失（空洞率低或内收不明显）。\n"
                   "    → 偏置可能来自内参/掩码，建议跑 S4 点云级核对。")
    print("  " + verdict.replace("\n", "\n  "))
    sess.results["s2"] = {
        "axis": {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in axis.items()},
        "hole_ratio_near": hr_near, "hole_ratio_far": hr_far, "ratio": ratio,
        "recession_px": rec[0] if rec else None, "recession_mm": rec[1] if rec else None,
        "depth_bins": rows, "hole_offset_px": off["offset_px"] if off else None,
        "verdict": verdict, "stats": stats,
    }

    # ── 可视化与快照存盘 ──
    _save_s2_visual(sess, dep, depth, valid, axis, rec)
    path = os.path.join(args.out, "s2_snapshot.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sess.results["s2"], f, indent=2, ensure_ascii=False, default=str)
    print(f"  [存] S2 快照 → {path}")


def _save_s2_visual(sess, dep, depth, valid, axis, rec):
    """S2 可视化：RGB + 多边形 + 远轴箭头 + 空洞红区 + 远边内收带标注；并排深度伪彩。

    叠加层说明（两幅图相同）：
      绿色轮廓  多边形手标边界（"真实"轮廓）
      黄色轮廓  有效深度掩码外边界（D435 实际测到深度的最远边）
      橙色填充  内收带——掩码内有效深度到达不了的远端条带
      红色填充  空洞像素（depth=0）
      青色箭头  远轴方向
      橙色文字  内收量 mm / px 标注
    """
    try:
        import cv2
    except Exception:
        return
    args = sess.args
    os.makedirs(args.out, exist_ok=True)
    lo, hi = float(args.depth_range[0]), float(args.depth_range[1])
    depth_vis, _, _ = dep["colorize_depth"](depth, lo, hi)
    bgr = sess.last_rgb[..., ::-1].copy()

    # ── 基础层：空洞红区 + 多边形轮廓 + 远轴箭头 ────────────────────────────
    holes = sess.mask & ~valid
    cu_i, cv_i = int(axis["cu"]), int(axis["cv"])
    du, dv = axis["du"], axis["dv"]
    k = 80
    for img in (bgr, depth_vis):
        if holes.any():
            ov = img.copy()
            ov[holes] = (0, 0, 255)
            img[:] = cv2.addWeighted(img, 0.55, ov, 0.45, 0)
        cv2.polylines(img, [sess.poly_arr], True, (0, 255, 0), 2)
        cv2.arrowedLine(img, (cu_i, cv_i),
                        (int(cu_i + du * k), int(cv_i + dv * k)),
                        (255, 200, 0), 2, tipLength=0.3)
        cv2.putText(img, "far",
                    (int(cu_i + du * k) + 4, int(cv_i + dv * k) - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 200, 0), 1, cv2.LINE_AA)

    # ── 远边内收带可视化 ──────────────────────────────────────────────────────
    if rec is not None:
        rec_px, rec_mm, _ = rec
        cu_f, cv_f = axis["cu"], axis["cv"]

        # 各掩码像素沿远轴的投影 s
        yy, xx = np.nonzero(sess.mask)
        s_poly = (xx - cu_f) * du + (yy - cv_f) * dv

        vy, vx = np.nonzero(valid)
        s_val = (vx - cu_f) * du + (vy - cv_f) * dv
        p98_val = float(np.percentile(s_val, 98)) if s_val.size else 0.0

        # 内收带：掩码内 s > p98_val 的像素（有效深度覆盖不到的远端条带）
        recession_band = np.zeros(sess.mask.shape, dtype=bool)
        far_sel = s_poly > p98_val
        recession_band[yy[far_sel], xx[far_sel]] = True

        # 有效深度掩码的外轮廓（黄色）
        valid_u8 = valid.astype(np.uint8) * 255
        contours, _ = cv2.findContours(valid_u8, cv2.RETR_EXTERNAL,
                                       cv2.CHAIN_APPROX_SIMPLE)

        # 内收量文字落点：多边形最远端像素附近
        tip_sel = s_poly == s_poly.max()
        tip_x = int(np.mean(xx[tip_sel]))
        tip_y = int(np.mean(yy[tip_sel]))
        label = f"recession: {rec_mm:.1f} mm  ({rec_px:.0f} px)"

        for img in (bgr, depth_vis):
            # 橙色半透明内收带
            ov2 = img.copy()
            ov2[recession_band] = (0, 110, 255)      # BGR 橙
            img[:] = cv2.addWeighted(img, 0.45, ov2, 0.55, 0)

            # 黄色：有效深度外轮廓
            cv2.drawContours(img, contours, -1, (0, 220, 220), 1)

            # 内收量标注（深色描边 + 浅色填字，保证对比度）
            tx = min(max(tip_x + 6, 4), img.shape[1] - 260)
            ty = max(tip_y - 8, 16)
            cv2.putText(img, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(img, label, (tx, ty),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 110, 255), 1, cv2.LINE_AA)

            # 图例（左上角）
            legend = [
                ((0, 255, 0),    "polygon boundary (truth)"),
                ((0, 220, 220),  "valid depth boundary"),
                ((0, 110, 255),  f"recession band  {rec_mm:.1f} mm"),
            ]
            ly = 16
            for color, text in legend:
                cv2.rectangle(img, (6, ly - 9), (18, ly + 3), color, -1)
                cv2.putText(img, text, (24, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.putText(img, text, (24, ly),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.40, color, 1, cv2.LINE_AA)
                ly += 16

    combined = np.hstack([bgr, depth_vis])
    path = os.path.join(args.out, "s2_overlay.png")
    cv2.imwrite(path, combined)
    print(f"\n  [图] 近远对照可视化 → {path}")
    if rec is not None:
        print(f"       绿=多边形标注边界  青=有效深度边界  橙=内收带({rec_mm:.1f}mm)")
        _save_recession_zoom(sess, valid, axis, rec, args)


def _save_recession_zoom(sess, valid, axis, rec, args):
    """单独输出一张仅展示内收差距的放大图 s2_recession_zoom.png。

    在"远轴坐标系"里绘图：以切向 t 为 X 轴、远轴 s 为 Y 轴（越靠上越远）。
    在这个坐标系里内收是一条水平带，不受工件倾斜角度影响。

      绿色水平线  多边形远端边界（各切向列最远点的 s 均值，即"真实边界"）
      青色水平线  有效深度远端边界（p98 分位，即"D435 实际边界"）
      橙色填充    两条线之间的内收带
      右侧刻度    mm 刻度尺（沿 s 轴）
    """
    try:
        import cv2
    except Exception:
        return

    ZOOM = 6          # 放大倍数（坐标系变换后原始 px 很小，需要更大的放大）
    PAD_T = 20        # 切向留白（列）
    PAD_S = 15        # 远轴留白（行，原始像素）

    rec_px, rec_mm, _ = rec
    du, dv = axis["du"], axis["dv"]
    cu_f, cv_f = axis["cu"], axis["cv"]
    tu, tv = -dv, du  # 切向（垂直于远轴）

    # ── 所有掩码像素的 (t, s) 坐标 ──────────────────────────────────────
    yy_m, xx_m = np.nonzero(sess.mask)
    s_poly = (xx_m - cu_f) * du + (yy_m - cv_f) * dv
    t_poly = (xx_m - cu_f) * tu + (yy_m - cv_f) * tv

    vy2, vx2 = np.nonzero(valid)
    s_val = (vx2 - cu_f) * du + (vy2 - cv_f) * dv
    t_val = (vx2 - cu_f) * tu + (vy2 - cv_f) * tv
    p98_val = float(np.percentile(s_val, 98)) if s_val.size else 0.0

    # 多边形远端边界：取 s 最大的 2% 像素的 s 均值作为"真实边界线"
    s_tip = float(np.percentile(s_poly, 98))   # 与 recession_width 保持一致：用 max
    s_tip = float(s_poly.max())

    # ── 在 (t, s) 坐标系里构建图像 ───────────────────────────────────────
    # t 轴范围
    t_min = float(t_poly.min()) - PAD_T
    t_max = float(t_poly.max()) + PAD_T
    # s 轴范围：只看远端附近（p98_val - PAD_S 到 s_tip + PAD_S）
    s_lo = p98_val - PAD_S
    s_hi = s_tip + PAD_S

    t_range = t_max - t_min
    s_range = s_hi - s_lo
    if t_range < 1 or s_range < 1:
        return

    # 图像尺寸（原始像素单位，1px = 1px，之后放大）
    img_w = int(np.ceil(t_range)) + 1
    img_h = int(np.ceil(s_range)) + 1

    canvas = np.zeros((img_h, img_w, 3), dtype=np.uint8)

    # 坐标映射：t → col，s → row（s 越大越靠上，所以 row = img_h - 1 - (s - s_lo)）
    def to_row(s):
        return img_h - 1 - int(round(s - s_lo))

    # 橙色内收带：s 在 [p98_val, s_tip] 之间、t 在掩码范围内的区域
    t_int = np.round(t_poly - t_min).astype(int)
    s_arr = s_poly
    for i in range(len(t_int)):
        ti = t_int[i]
        si = s_arr[i]
        if p98_val <= si <= s_tip:
            ri = to_row(si)
            if 0 <= ti < img_w and 0 <= ri < img_h:
                canvas[ri, ti] = (0, 110, 255)

    # 青色线：有效深度远端边界（p98_val），逐切向列画一个点
    t_int_v = np.round(t_val - t_min).astype(int)
    for tb in range(img_w):
        sel = t_int_v == tb
        if sel.any():
            r = to_row(p98_val)
            if 0 <= r < img_h:
                canvas[r, tb] = (0, 220, 220)

    # 绿色线：多边形远端边界（s_tip），逐切向列画一个点
    for tb in range(img_w):
        sel_m = np.round(t_poly - t_min).astype(int) == tb
        if sel_m.any():
            r = to_row(s_tip)
            if 0 <= r < img_h:
                canvas[r, tb] = (0, 255, 0)

    # ── 放大 ──────────────────────────────────────────────────────────────
    zoomed = cv2.resize(canvas, (img_w * ZOOM, img_h * ZOOM),
                        interpolation=cv2.INTER_NEAREST)
    zH, zW = zoomed.shape[:2]

    # ── mm 刻度尺（右侧，沿 s 轴方向） ───────────────────────────────────
    fx = float(sess.K[0, 0]) if sess.K is not None else 605.0
    pix_per_mm = fx / (axis["d_med"] * 1000.0)
    pix_per_mm_z = pix_per_mm * ZOOM

    ruler_x = zW - 20
    # 0mm 对应 p98_val，向上是内收方向
    r0_z = to_row(p98_val) * ZOOM + ZOOM // 2   # 青色线在放大图里的 y

    max_tick_mm = int(np.ceil(rec_mm)) + 2
    for mm in range(max_tick_mm + 1):
        ry = int(r0_z - mm * pix_per_mm_z)
        if ry < 0 or ry >= zH:
            continue
        tick_len = 7 if mm % 2 == 0 else 4
        cv2.line(zoomed, (ruler_x - tick_len, ry), (ruler_x, ry),
                 (180, 180, 180), 1)
        if mm % 2 == 0:
            cv2.putText(zoomed, f"{mm}", (ruler_x - tick_len - 24, ry + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (160, 160, 160), 1, cv2.LINE_AA)
    # 尺身
    r_top = max(int(r0_z - max_tick_mm * pix_per_mm_z), 2)
    cv2.line(zoomed, (ruler_x, r_top), (ruler_x, min(int(r0_z) + 4, zH - 2)),
             (180, 180, 180), 1)
    cv2.putText(zoomed, "mm", (ruler_x - 22, r_top - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (160, 160, 160), 1, cv2.LINE_AA)

    # ── 标注 + 图例 ────────────────────────────────────────────────────────
    label = f"recession  {rec_mm:.2f} mm  ({rec_px:.0f} px)"
    cv2.putText(zoomed, label, (6, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(zoomed, label, (6, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 180, 255), 1, cv2.LINE_AA)

    legend = [
        ((0, 255, 0),   "polygon far edge (truth)"),
        ((0, 220, 220), "valid depth far edge (p98)"),
        ((0, 110, 255), "recession band"),
    ]
    ly = zH - 10 - len(legend) * 14
    for color, text in legend:
        cv2.rectangle(zoomed, (6, ly - 8), (14, ly + 2), color, -1)
        cv2.putText(zoomed, text, (18, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(zoomed, text, (18, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        ly += 14

    # X 轴标注
    cv2.putText(zoomed, "← tangential direction →", (6, zH - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.30, (100, 100, 100), 1, cv2.LINE_AA)

    roi = zoomed  # 坐标系图不需要再裁 ROI

    # ── 放大 ──────────────────────────────────────────────────────────────
    rH, rW = roi.shape[:2]
    zoomed = cv2.resize(roi, (rW * ZOOM, rH * ZOOM), interpolation=cv2.INTER_NEAREST)
    zH, zW = zoomed.shape[:2]

    # ── 毫米刻度尺（沿远轴方向，画在图右侧） ───────────────────────────────
    fx = float(sess.K[0, 0]) if sess.K is not None else 605.0
    pix_per_mm = fx / (axis["d_med"] * 1000.0)   # 原始像素 / mm
    pix_per_mm_z = pix_per_mm * ZOOM              # 放大后像素 / mm

    # 刻度尺参数
    ruler_x = zW - 18
    ruler_top = 20
    # 画出 0 到 ceil(rec_mm)+2 mm 的刻度，步长 1mm
    max_mm = int(np.ceil(rec_mm)) + 2
    ruler_len_px = int(max_mm * pix_per_mm_z)
    ruler_bot = ruler_top + ruler_len_px

    # 尺身（白色细线）
    cv2.line(zoomed, (ruler_x, ruler_top), (ruler_x, min(ruler_bot, zH - 4)),
             (200, 200, 200), 1)
    for mm in range(max_mm + 1):
        ry = ruler_top + int(mm * pix_per_mm_z)
        if ry >= zH:
            break
        tick_len = 6 if mm % 5 == 0 else 3
        cv2.line(zoomed, (ruler_x - tick_len, ry), (ruler_x, ry), (200, 200, 200), 1)
        if mm % 2 == 0:
            cv2.putText(zoomed, f"{mm}", (ruler_x - tick_len - 22, ry + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1, cv2.LINE_AA)
    # 单位标注
    cv2.putText(zoomed, "mm", (ruler_x - 20, ruler_top - 6),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (180, 180, 180), 1, cv2.LINE_AA)

    # 内收量数值（左上角）
    label = f"recession  {rec_mm:.2f} mm  ({rec_px:.0f} px)"
    cv2.putText(zoomed, label, (6, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(zoomed, label, (6, 14),
                cv2.FONT_HERSHEY_SIMPLEX, 0.42, (0, 180, 255), 1, cv2.LINE_AA)

    # 图例（左下角）
    legend = [
        ((0, 255, 0),   "polygon (truth)"),
        ((0, 220, 220), "valid depth"),
        ((0, 110, 255), "recession band"),
    ]
    ly = zH - 10 - len(legend) * 14
    for color, text in legend:
        cv2.rectangle(zoomed, (6, ly - 8), (14, ly + 2), color, -1)
        cv2.putText(zoomed, text, (18, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(zoomed, text, (18, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1, cv2.LINE_AA)
        ly += 14

    zoom_path = os.path.join(args.out, "s2_recession_zoom.png")
    cv2.imwrite(zoom_path, zoomed)
    print(f"  [图] 内收差距放大图（{ZOOM}×）→ {zoom_path}")


def stage3_stability(sess, dep):
    """[3] 稳定性采集：远端缺失是否稳定出现。"""
    _sec(f"[S3] 稳定性采集（{sess.args.frames} 帧）")
    if sess.mask is None:
        stage1_annotate(sess, dep)
    args = sess.args
    K, R = sess.K, sess.R
    cam_pos = np.asarray(dep["CALIB_CAM_POS_IN_BASE_M"], dtype=np.float64)

    # 先用一帧确定远轴（工件静止，远轴不随帧变）。
    rgb, depth = sess.cam.get_frame()
    sess.last_rgb, sess.last_depth = rgb, depth
    axis = compute_far_axis(depth, sess.mask, K, cam_pos, R)
    if axis is None:
        print("[S3] 无有效深度，无法定远轴")
        return
    sess.axis = axis

    rec_list, hr_far_list, hr_near_list, vis_list = [], [], [], []
    print("  帧/总   远半空洞率  近半空洞率  远端内收(mm)  有效px")
    for i in range(args.frames):
        rgb, depth = sess.cam.get_frame()
        if rgb is None or depth is None:
            continue
        valid = sess.mask & np.isfinite(depth) & (depth > 0)
        near, far = split_near_far(sess.mask, axis)
        hr_near, _, _ = half_hole_ratio(depth, near)
        hr_far, _, _ = half_hole_ratio(depth, far)
        rec = recession_width(sess.mask, valid, axis, K)
        rec_mm = rec[1] if rec else 0.0
        nv = int(valid.sum())
        rec_list.append(rec_mm); hr_far_list.append(hr_far)
        hr_near_list.append(hr_near); vis_list.append(nv)
        if (i + 1) % 5 == 0 or i == args.frames - 1:
            print(f"  {i + 1:>3}/{args.frames}   {100 * hr_far:6.1f}%   "
                  f"{100 * hr_near:6.1f}%   {rec_mm:>8.2f}   {nv}")
        sess.last_depth = depth

    rec_arr = np.asarray(rec_list)
    _sub("S3 汇总")
    n = len(rec_arr)
    present = float((rec_arr > RECESSION_SIGNIFICANT_MM).mean())
    print(f"  远端内收  均值 {rec_arr.mean():.2f} mm  std {rec_arr.std():.2f} mm  "
          f"min {rec_arr.min():.2f}  max {rec_arr.max():.2f}")
    print(f"  远半空洞率 均值 {100 * np.mean(hr_far_list):.1f}%  std {100 * np.std(hr_far_list):.1f}%")
    print(f"  近半空洞率 均值 {100 * np.mean(hr_near_list):.1f}%  std {100 * np.std(hr_near_list):.1f}%")
    print(f"  >{RECESSION_SIGNIFICANT_MM}mm 的帧占比 {100 * present:.0f}%")
    if present > 0.8 and rec_arr.std() < max(1.0, 0.5 * rec_arr.mean()):
        verdict = "稳定系统性缺失 → 训练做【确定性远边腐蚀】，band 取均值"
    elif present > 0.2:
        verdict = "时而出现（不稳定）→ 训练做【随机远边腐蚀/加噪声】，band 用分布上界"
    else:
        verdict = "远边缺失不明显/不稳定 → 重点回到内参/掩码，跑 S4"
    print("  → " + verdict)

    sess.results["s3"] = {
        "frames": n, "recession_mean_mm": float(rec_arr.mean()),
        "recession_std_mm": float(rec_arr.std()),
        "recession_series_mm": rec_arr.tolist(),
        "hole_far_mean": float(np.mean(hr_far_list)), "hole_near_mean": float(np.mean(hr_near_list)),
        "present_ratio": float(present), "verdict": verdict,
    }
    # 存 CSV
    os.makedirs(args.out, exist_ok=True)
    csv = os.path.join(args.out, "s3_stability.csv")
    with open(csv, "w") as f:
        f.write("frame,far_hole_ratio,near_hole_ratio,recession_mm,valid_px\n")
        for i in range(n):
            f.write(f"{i},{hr_far_list[i]:.5f},{hr_near_list[i]:.5f},"
                    f"{rec_list[i]:.3f},{vis_list[i]}\n")
    print(f"  [存] 逐帧序列 → {csv}")
    js = os.path.join(args.out, "s3_stability.json")
    with open(js, "w", encoding="utf-8") as f:
        json.dump(sess.results["s3"], f, indent=2, ensure_ascii=False, default=str)
    print(f"  [存] S3 汇总 → {js}")
    _save_s3_plot(rec_arr, args.out)


def _save_s3_plot(rec_arr, out_dir):
    """可选 matplotlib 趋势图；缺失则跳过（终端已有 ASCII 数据）。"""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure(figsize=(8, 3))
        plt.plot(rec_arr, marker="o", ms=3)
        plt.axhline(0, color="gray", lw=1)
        plt.xlabel("frame"); plt.ylabel("远端内收 (mm)")
        plt.title("远端深度缺失带宽度随帧变化")
        plt.tight_layout()
        path = os.path.join(out_dir, "s3_trend.png")
        plt.savefig(path, dpi=120); plt.close()
        print(f"  [图] 趋势图 → {path}")
    except Exception:
        pass


def stage4_pointcloud(sess, dep):
    """[4] 点云级验证 + 汇总报告：反投影点云远端截断/质心偏置。"""
    _sec("[S4] 点云级验证 + 汇总报告")
    if sess.mask is None:
        stage1_annotate(sess, dep)
    # 用最近一帧（S3 已更新）或重新取一帧
    if sess.last_depth is None:
        rgb, depth = sess.cam.get_frame()
        sess.last_rgb, sess.last_depth = rgb, depth
    depth = sess.last_depth
    args = sess.args

    from configs.point_bridge_pointcloud import (
        M_OBJ, NOISE_STD_M, WORKSPACE_RADIUS_XY_M, WORKSPACE_Z_RANGE_M,
        mask_depth_to_pointcloud,
    )
    import torch

    cam_pos = np.asarray(dep["CALIB_CAM_POS_IN_BASE_M"], dtype=np.float64)
    K, R = sess.K, sess.R
    axis = sess.axis
    if axis is None:
        axis = compute_far_axis(depth, sess.mask, K, cam_pos, R)
        sess.axis = axis

    mask_t = torch.from_numpy(sess.mask).unsqueeze(0)
    depth_t = torch.from_numpy(np.nan_to_num(depth, nan=0.0)).float().unsqueeze(0)
    K_t = torch.from_numpy(K).float().unsqueeze(0)
    pos_t = torch.tensor(dep["CALIB_CAM_POS_IN_BASE_M"], dtype=torch.float32).unsqueeze(0)
    quat_t = torch.tensor(dep["CALIB_CAM_QUAT_IN_BASE_WXYZ"], dtype=torch.float32).unsqueeze(0)

    pts, counts = mask_depth_to_pointcloud(
        mask=mask_t, depth=depth_t, K=K_t, cam_pos=pos_t, cam_quat=quat_t,
        m_obj=args.m_obj, noise_std=args.noise_std,
        workspace_z_range_m=None if args.no_workspace_filter else WORKSPACE_Z_RANGE_M,
        workspace_radius_xy_m=None if args.no_workspace_filter else WORKSPACE_RADIUS_XY_M,
    )
    pts = pts[0].numpy()                                   # (m_obj, 3) 基座系
    counts = int(counts[0])

    _sub("点云概况")
    print(f"  有效像素 {counts}   采样点 {pts.shape[0]}")
    if counts < dep["MIN_MASK_PIXELS"]:
        print(f"  [警告] 有效像素 <{dep['MIN_MASK_PIXELS']}，FPS 输入大量重复，点云退化")

    if axis is not None and len(pts):
        d3 = axis["d3"]
        P_obj = axis["P_obj"]
        s3 = (pts - P_obj) @ d3                              # 沿远轴投影（+ = 远）
        far_ext = float(s3.max()); near_ext = float(-s3.min())
        centroid = pts.mean(axis=0)
        bias_mm = float((centroid - P_obj) @ d3) * 1000.0    # 负 = 向相机偏
        _sub("远端截断 / 质心偏置")
        print(f"  沿远轴：远端延伸 {far_ext * 1000:+.1f} mm   近端延伸 {near_ext * 1000:+.1f} mm")
        print(f"  不对称量(远-近) = {(far_ext - near_ext) * 1000:+.1f} mm  "
              f"{'(远侧更短 → 被截断)' if far_ext < near_ext else ''}")
        print(f"  点云质心相对工件锚点沿远轴偏移 = {bias_mm:+.1f} mm  "
              f"{'(负=向相机)' if bias_mm < 0 else ''}")
        sess.results["s4"] = {
            "far_extent_mm": far_ext * 1000, "near_extent_mm": near_ext * 1000,
            "asymmetry_mm": (far_ext - near_ext) * 1000,
            "centroid_bias_along_far_mm": bias_mm,
            "counts": counts, "n_points": int(pts.shape[0]),
        }
    else:
        sess.results["s4"] = {"counts": counts, "n_points": int(pts.shape[0])}

    os.makedirs(args.out, exist_ok=True)
    np.save(os.path.join(args.out, "s4_points.npy"), pts)
    print(f"  [存] 点云 (npy) → {os.path.join(args.out, 's4_points.npy')}")

    # ── 汇总报告 ──
    _sec("[汇总] 最终报告")
    _print_final_report(sess)


def _print_final_report(sess):
    s2 = sess.results.get("s2")
    s3 = sess.results.get("s3")
    s4 = sess.results.get("s4")
    lines = []
    lines.append("实验结论：远端深度诊断")
    if s2:
        lines.append(f"  [S2 单帧] 远/近空洞率比 {s2['ratio']:.2f}，"
                     f"远端内收 {s2['recession_mm']:.2f} mm")
        lines.append(f"           {s2['verdict'].splitlines()[0]}")
    if s3:
        lines.append(f"  [S3 稳定] 内收均值 {s3['recession_mean_mm']:.2f} mm，"
                     f"占比 {100 * s3['present_ratio']:.0f}% → {s3['verdict']}")
    if s4 and "centroid_bias_along_far_mm" in s4:
        lines.append(f"  [S4 点云] 质心沿远轴偏 {s4['centroid_bias_along_far_mm']:+.1f} mm"
                     f"（负=向相机），远-近延伸差 {s4['asymmetry_mm']:+.1f} mm")
    band = None
    if s3 and s3["recession_mean_mm"] > RECESSION_SIGNIFICANT_MM:
        band = s3["recession_mean_mm"]
    elif s2 and s2["recession_mm"] is not None and s2["recession_mm"] > RECESSION_SIGNIFICANT_MM:
        band = s2["recession_mm"]
    lines.append("")
    if band is not None:
        lines.append(f"  训练侧建议：加入【远边深度腐蚀】，band ≈ {band:.1f} mm"
                     f"（在反投影前把工件远边深度置 0）。")
    else:
        lines.append("  训练侧建议：远边缺失不明显，先核对内参对齐/掩码来源，再决定腐蚀形态。")
    for ln in lines:
        print("  " + ln)
    report = {"s2": s2, "s3": s3, "s4": s4, "summary": lines,
              "intrinsics": sess.intr,
              "polygon": [list(map(int, p)) for p in sess.polygon] if sess.polygon else None,
              "out": sess.args.out}
    path = os.path.join(sess.args.out, "final_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False, default=str)
    print(f"\n  [存] 汇总报告 → {path}")


# =============================================================================
# 离线自检（不连相机，验证几何符号与内收量计算）
# =============================================================================


def selftest():
    print("=" * 78)
    print("  离线自检：远轴方向 / 近远分半 / 内收量（合成数据）")
    print("=" * 78)
    fx = fy = 615.0
    K = np.array([[fx, 0, 320.0], [0, fy, 240.0], [0, 0, 1]], dtype=np.float64)
    R = np.eye(3)                                    # 相机与基座系同向（仅平移）
    cam_pos = np.array([0.0, 0.0, 0.0])

    # 合成：一个在 z=0.6 附近、随 +x 变远的倾斜平面（远边 = +x 侧）。
    H, W = 480, 640
    yy, xx = np.mgrid[0:H, 0:W]
    # 物体掩码：中心 (320,240)、半径 90 的圆盘
    mask = ((xx - 320.0) ** 2 + (yy - 240.0) ** 2) <= 90.0 ** 2
    # 深度：z = 0.6 + 0.0004*(x - 320)  → +x 越远
    depth = (0.6 + 0.0004 * (xx - 320.0)).astype(np.float64)
    depth[~mask] = 0.0

    # 注入远端空洞带：x > 320 + 50 的像素深度置 0（远边丢 40px 带）
    hole_band = (xx - 320.0) > 50.0
    depth[mask & hole_band] = 0.0

    # 1) 远轴方向应为 +x（du>0）
    axis = compute_far_axis(depth, mask, K, cam_pos, R)
    assert axis is not None, "compute_far_axis 返回 None"
    print(f"[1] 远方向 (du,dv) = ({axis['du']:+.3f}, {axis['dv']:+.3f})")
    assert axis["du"] > 0.5 and abs(axis["dv"]) < 0.5, "远方向应指向 +x"
    print("    PASS：远方向指向 +x（深度变大的方向）")

    # 2) 往返一致性：unproject 再 project 应回到原像素
    uv0 = np.array([[320.0, 240.0], [380.0, 240.0]])
    z0 = np.array([0.6, 0.62])
    P = unproject_pixels_to_base(uv0, z0, K, cam_pos, R)
    uv1 = project_base_to_pixels(P, K, cam_pos, R)
    err = float(np.abs(uv1 - uv0).max())
    print(f"[2] 反投影往返最大误差 = {err:.2e} px")
    assert err < 1e-6, "往返应精确"
    print("    PASS：unproject ↔ project 精确互逆")

    # 3) 近远分半 + 内收量
    near, far = split_near_far(mask, axis)
    valid = mask & (depth > 0)
    hr_near, _, _ = half_hole_ratio(depth, near)
    hr_far, _, _ = half_hole_ratio(depth, far)
    rec = recession_width(mask, valid, axis, K)
    print(f"[3] 近半空洞率 {100 * hr_near:.1f}%   远半空洞率 {100 * hr_far:.1f}%")
    print(f"    远端内收 = {rec[0]:.1f} px ≈ {rec[1]:.2f} mm  (注入带=40px)")
    assert hr_far > hr_near, "远半空洞率应更高"
    assert rec[0] > 35, "内收量应能测到注入的 40px 带（max-p98 读法）"
    print("    PASS：远半空洞率更高、内收量测到注入带")

    # 4) 深度分箱
    rows = depth_bin_profile(depth, mask, 5)
    if rows is not None:
        print("[4] 深度分箱（近→远）空洞率：")
        print("    " + "  ".join(f"{100 * r['hole_ratio']:.0f}%" for r in rows))
        if any(r["hole"] > 0 for r in rows):
            assert rows[-1]["hole_ratio"] > rows[0]["hole_ratio"], "远箱空洞率应更高"
            print("    PASS：远箱空洞率抬升")
        else:
            print("    SKIP：scipy 缺失，空洞归箱未执行（不影响其它自检）")
    print("\n自检全部通过 ✅")


# =============================================================================
# 主流程
# =============================================================================


def _print_menu():
    print("\n" + "-" * 40)
    print("  远端深度诊断 —— 交互菜单")
    print("  [0] 相机就绪 + 内参/外参对照")
    print("  [1] 标注工件 ROI")
    print("  [2] 单帧远边诊断（近端 vs 远端）")
    print("  [3] 稳定性采集（N 帧）")
    print("  [4] 点云级验证 + 汇总报告")
    print("  [a] 一键顺序跑 0→4")
    print("  [q] 退出")
    print("-" * 40)


def main():
    args = parse_args()
    os.makedirs(args.out, exist_ok=True)
    print(f"[输出] 结果目录 {args.out}")

    if args.selftest:
        selftest()
        return

    dep = _load_deps()
    sess = Session(args)

    stage_fns = {
        "0": stage0_camera, "1": stage1_annotate, "2": stage2_diagnose,
        "3": stage3_stability, "4": stage4_pointcloud,
    }

    if args.run_all:
        if args.polygon is None and args.no_window:
            raise SystemExit("[--run_all] 无窗口环境必须给 --polygon")
        for k in ("0", "1", "2", "3", "4"):
            stage_fns[k](sess, dep)
        return

    try:
        while True:
            _print_menu()
            sel = input("  选择 > ").strip().lower()
            if sel == "q":
                break
            if sel == "a":
                for k in ("0", "1", "2", "3", "4"):
                    stage_fns[k](sess, dep)
                continue
            if sel in stage_fns:
                try:
                    stage_fns[sel](sess, dep)
                except SystemExit:
                    raise
                except Exception as e:
                    import traceback
                    print(f"[错误] 阶段 {sel} 失败：{type(e).__name__}: {e}")
                    traceback.print_exc()
            else:
                print(f"[提示] 未知选项 '{sel}'")
    except KeyboardInterrupt:
        print("\n[中断]")
    finally:
        if sess.cam is not None:
            sess.cam.stop()
        try:
            import cv2
            cv2.destroyAllWindows()
        except Exception:
            pass


if __name__ == "__main__":
    main()
