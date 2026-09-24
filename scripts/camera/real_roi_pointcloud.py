#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实机 D435 多边形掩码 → 基座系物体点云（与仿真共用同一个反投影函数）。

和同目录 view_real_depth.py 的分工::

    view_real_depth.py --stage 1   只看 RGB/深度数据流，确认实机这一路本身对
    view_real_depth.py --stage 2   手标多边形 + 掩码内深度统计/采样，仍停在像素系
    本脚本                          手标多边形 → 基座系 3D 点云      ← 再往前一步

工件掩码由手标多边形填充而来，顶替仿真里的实例分割掩码。用多边形而不是矩形：
矩形必然把工件周围的桌面框进来，那些像素深度有效，会被当成工件点采走，在基座系
里表现为工件周围一圈贴着桌面的假点。

关键点：反投影**不自己写**，直接调 configs/point_bridge_pointcloud.py 的
``mask_depth_to_pointcloud`` —— 那个函数的 docstring 明写"仿真与真机共用此函数"。
自己照公式重写一遍会得到一朵看着也对的点云，但它证明不了真机路径和训练路径一致，
而这恰恰是唯一值得测的东西。共用之后，FPS 采样、工作空间裁剪、噪声、以及
"有效像素少于 cap 时按取模循环补齐"这些行为全都自动对齐。

外参用标定给的**基座系**原始值（configs 里的 CALIB_CAMERA_*），不是换算到 env
根之后的那一组 —— 后者是喂给 Isaac 的 OffsetCfg 用的，真机没有 env 根这个概念。
两组值混用不会报错，只会让点云整体偏掉，所以这里显式取前者。

判读要点：
  - 点贴在工件表面，不飘不散
  - 工件在桌面上，基座系 z 应落在 WORKSPACE_Z_RANGE_M 内且大致等于桌面高度
  - visible_counts >= 200，低于此 FPS 输入大量重复，点云退化成一小撮点
  - 遮挡工件时被挡那块的点消失

冻结深度（``f`` 键 / ``--freeze_at_start``）：把某一帧的深度图**留存**下来，此后
每帧都从这张不变的深度图重新采点。冻结的是深度图不是点云 —— 掩码和 FPS 照常每帧
重跑，所以取到的 64 个点每帧仍然不同，只是底图不变。冻结点云则连采样随机性都没了。

冻结态下点云与画面里的现实是**脱节**的：工件被挪走，点还停在原处。用它驱动机械臂
就是朝记忆中的位置去抓。所以冻结时画面加红框红条、日志带 [FROZEN@n] 标记、meta 里
记 ``depth_frozen`` —— 这个状态必须一眼看得出来，按 ``u`` 可随时解冻。

用法（**必须用 gx_va_deploy 环境**：既要 pyrealsense2 也要 isaaclab —— 后者是
``point_bridge_pointcloud`` 的依赖，共用仿真函数的代价就是它）::

    conda activate gx_va_deploy

    # 左键沿工件轮廓点一圈，回车确认，实时看点云
    # 运行中 f=冻结当前深度图  u=解冻  q=退出
    python scripts/camera/real_roi_pointcloud.py

    # 复用上次标好的多边形
    python scripts/camera/real_roi_pointcloud.py \
        --polygon "310,220 380,225 385,275 305,270"
    python scripts/camera/real_roi_pointcloud.py \
        --polygon logs/real_roi_pointcloud/pointcloud_meta.json

    # 标完就冻结，看同一张深度图上 FPS 每帧采出的点有多大差异
    python scripts/camera/real_roi_pointcloud.py \
        --polygon logs/real_roi_pointcloud/pointcloud_meta.json --freeze_at_start

    # 无显示器，存图 + 存点云
    python scripts/camera/real_roi_pointcloud.py \
        --polygon logs/real_roi_pointcloud/pointcloud_meta.json \
        --no_window --save_frames --save_points --max_frames 30
"""

import argparse
import json
import os
import sys
import time

import numpy as np

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))
sys.path.insert(0, _PROJECT_DIR)

import torch

from configs.point_bridge_pointcloud import (
    M_OBJ,
    NOISE_STD_M,
    WORKSPACE_RADIUS_XY_M,
    WORKSPACE_Z_RANGE_M,
    mask_depth_to_pointcloud,
)
from scripts.camera.realsense_camera import RealsenseCamera
from scripts.camera.view_real_depth import (
    MIN_MASK_PIXELS,
    SIM_CANONICAL,
    colorize_depth,
    parse_polygon_arg,
    polygon_to_mask,
    report_intrinsics,
    select_polygon,
)

# 标定外参（**基座系**原始值）。取自 configs/xarm7_pick_pointcloud_env_cfg.py
# 的 CALIB_CAMERA_* —— 那里注释写明这一组是"供追溯 / 真机部署用"。
# 从 configs 直接 import 会连带拉起 isaaclab（真机环境未必装），故此处复制常量，
# 并在启动时与 configs 核对一次（见 _verify_extrinsics）。
#
# 2026-09-14 同步 2026-09-01 重标（同一台相机 serial 254622072913，源文件
# /home/lsz/gx_pose/config/camera_to_base.json；Quality: pos_rms 1.19mm,
# rot_rms 0.29deg, PASS）。注意 JSON 里是 xyzw，此处是 wxyz，已换序。
CALIB_CAM_POS_IN_BASE_M = (0.39141, -0.781319, 0.588075)
CALIB_CAM_QUAT_IN_BASE_WXYZ = (-0.523262, 0.851472, 0.006784, -0.033852)


def parse_args():
    p = argparse.ArgumentParser(description="实机多边形掩码 → 基座系点云（共用仿真反投影）")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--serial", type=str, default=None)
    p.add_argument("--warmup", type=int, default=30)
    p.add_argument("--polygon", type=str, default=None,
                   help="跳过手标，直接给多边形顶点：\"x1,y1 x2,y2 ...\"；"
                        "也可传上次存的 pointcloud_meta.json 路径。")
    p.add_argument("--m_obj", type=int, default=M_OBJ,
                   help=f"输出点数，默认与仿真 M_OBJ={M_OBJ} 一致")
    p.add_argument("--noise_std", type=float, default=0.0,
                   help=f"高斯噪声(米)。默认 0 便于核对几何；想看训练时的真实观测"
                        f"传 {NOISE_STD_M}。真机深度本身已有噪声，仿真加噪正是为了"
                        f"模拟它，所以这里再叠一层只在对照实验时才有意义。")
    p.add_argument("--no_workspace_filter", action="store_true",
                   help="关掉工作空间裁剪。标定或摆放不对时点会被裁到界内，"
                        "看起来'正常'反而掩盖问题 —— 排查时先关掉看原始分布。")
    p.add_argument("--depth_range", type=float, nargs=2, default=(0.3, 1.5),
                   metavar=("LO", "HI"), help="深度伪彩固定色标区间(米)")
    p.add_argument("--intrinsics", type=str, default="real", choices=("real", "sim"),
                   help="real=用实机实测内参(默认)；sim=强行用仿真 canonical，"
                        "用于隔离'内参差异'是不是点云偏移的来源")
    p.add_argument("--freeze_at_start", action="store_true",
                   help="标完多边形立刻冻结深度图，不必再按 f。等价于开跑就进冻结态。")
    p.add_argument("--max_frames", type=int, default=0)
    p.add_argument("--save_frames", action="store_true")
    p.add_argument("--save_points", action="store_true",
                   help="把每帧点云存成 npy，供离线与仿真点云对比")
    p.add_argument("--no_window", action="store_true")
    p.add_argument("--out", type=str,
                   default=os.path.join(_PROJECT_DIR, "logs", "real_roi_pointcloud"))
    return p.parse_args()


def _verify_extrinsics() -> None:
    """与 configs 里的标定值核对，防止那边改了这边忘了同步。

    **用文本解析而不是 import**：``xarm7_pick_pointcloud_env_cfg`` 依赖
    ``omni.client``（完整 Isaac Sim 运行时，只在 isaaclab.sh 下存在），在真机
    conda 环境里永远导不进来。而 ``point_bridge_pointcloud`` 只依赖
    ``isaaclab.utils.math``，纯 conda 装的 isaaclab 就够 —— 两者依赖深度不同，
    别把它们当成一回事。

    解析失败时**不静默放行**：使用可能过期的外参只会让点云悄悄偏掉，比直接
    报错难查得多。
    """
    import ast
    import re

    cfg_path = os.path.join(_PROJECT_DIR, "configs",
                            "xarm7_pick_pointcloud_env_cfg.py")
    try:
        src = open(cfg_path, encoding="utf-8").read()
        got = {}
        for name in ("CALIB_CAMERA_POSITION_IN_BASE_M",
                     "CALIB_CAMERA_QUATERNION_IN_BASE_WXYZ"):
            mt = re.search(rf"^{name}\s*=\s*(\([^)]*\))", src, re.M)
            if mt is None:
                raise ValueError(f"未在 {cfg_path} 找到 {name}")
            got[name] = ast.literal_eval(mt.group(1))
    except Exception as e:
        raise SystemExit(
            f"[错误] 无法从 configs 解析标定外参：{type(e).__name__}: {e}\n"
            f"  文件: {cfg_path}\n"
            "  拒绝用未经核对的外参继续 —— 偏移不会报错，只会让点云悄悄错位。"
        )

    dp = float(np.abs(np.asarray(CALIB_CAM_POS_IN_BASE_M)
                      - np.asarray(got["CALIB_CAMERA_POSITION_IN_BASE_M"])).max())
    dq = float(np.abs(np.asarray(CALIB_CAM_QUAT_IN_BASE_WXYZ)
                      - np.asarray(got["CALIB_CAMERA_QUATERNION_IN_BASE_WXYZ"])).max())
    if dp > 1e-6 or dq > 1e-6:
        raise SystemExit(
            "[错误] 外参与 configs 不一致，点云会整体偏移：\n"
            f"  本文件 : pos={CALIB_CAM_POS_IN_BASE_M} "
            f"quat={CALIB_CAM_QUAT_IN_BASE_WXYZ}\n"
            f"  configs: pos={got['CALIB_CAMERA_POSITION_IN_BASE_M']} "
            f"quat={got['CALIB_CAMERA_QUATERNION_IN_BASE_WXYZ']}\n"
            "  请同步本文件顶部的 CALIB_CAM_* 常量。"
        )
    print("[核对] 外参与 configs 一致")


def build_pointcloud(depth: np.ndarray, mask: np.ndarray, K: np.ndarray, args
                     ) -> "tuple[np.ndarray, int]":
    """工件掩码 → 基座系点云。反投影全部交给仿真那份函数。

    ``mask`` 是手标多边形填充出来的，顶替仿真里的实例分割掩码。多边形能贴着
    工件轮廓收边，剩下的偏差主要来自标注精度本身，而不是几何上必然混进来的
    一圈桌面像素 —— 后者是矩形框的固有问题。

    Returns:
        ``(points, visible_count)``，points 形状 (m_obj, 3)，基座系，单位米。
    """
    dev = torch.device("cpu")
    mask_t = torch.from_numpy(mask).unsqueeze(0).to(dev)
    depth_t = torch.from_numpy(np.nan_to_num(depth, nan=0.0)).float().unsqueeze(0).to(dev)
    K_t = torch.from_numpy(K).float().unsqueeze(0).to(dev)
    pos_t = torch.tensor(CALIB_CAM_POS_IN_BASE_M, dtype=torch.float32).unsqueeze(0)
    quat_t = torch.tensor(CALIB_CAM_QUAT_IN_BASE_WXYZ, dtype=torch.float32).unsqueeze(0)

    pts, counts = mask_depth_to_pointcloud(
        mask=mask_t,
        depth=depth_t,
        K=K_t,
        cam_pos=pos_t,
        cam_quat=quat_t,
        m_obj=args.m_obj,
        noise_std=args.noise_std,
        workspace_z_range_m=None if args.no_workspace_filter else WORKSPACE_Z_RANGE_M,
        workspace_radius_xy_m=None if args.no_workspace_filter else WORKSPACE_RADIUS_XY_M,
    )
    return pts[0].numpy(), int(counts[0])


def project_to_pixels(points: np.ndarray, K: np.ndarray) -> np.ndarray:
    """基座系点 → 像素坐标，用于把点叠回 RGB 自检反投影是否自洽。

    这是 build_pointcloud 的**逆运算**。两者一致时点会精确落回工件上；不一致
    说明外参或内参用错了 —— 这是本脚本能自查的最有力的一条。
    """
    from scripts.camera.view_real_depth import quat_to_matrix

    R = quat_to_matrix(CALIB_CAM_QUAT_IN_BASE_WXYZ)
    t = np.asarray(CALIB_CAM_POS_IN_BASE_M, dtype=np.float64)
    # base → cam：R 是 cam→base，故取转置
    pts_cam = (points.astype(np.float64) - t) @ R
    z = np.clip(pts_cam[:, 2], 1e-6, None)
    u = pts_cam[:, 0] / z * K[0, 0] + K[0, 2]
    v = pts_cam[:, 1] / z * K[1, 1] + K[1, 2]
    return np.stack([u, v, pts_cam[:, 2]], axis=-1)


def main() -> None:
    args = parse_args()
    import cv2

    _verify_extrinsics()

    show_window = not args.no_window
    if args.save_frames or args.save_points:
        os.makedirs(args.out, exist_ok=True)

    cam = RealsenseCamera(width=args.width, height=args.height, fps=args.fps,
                          align_on_device=True, serial_number=args.serial)
    cam.start()

    try:
        intr = report_intrinsics(cam)
        if args.intrinsics == "sim":
            K = np.array([[SIM_CANONICAL["fx"], 0, SIM_CANONICAL["cx"]],
                          [0, SIM_CANONICAL["fy"], SIM_CANONICAL["cy"]],
                          [0, 0, 1]], dtype=np.float64)
            print("[内参] 强制使用仿真 canonical")
        else:
            K = np.asarray(cam.K_color, dtype=np.float64)

        print(f"[外参] 基座系 pos={CALIB_CAM_POS_IN_BASE_M} "
              f"quat={CALIB_CAM_QUAT_IN_BASE_WXYZ}")
        ws = "关闭" if args.no_workspace_filter else \
             f"z{WORKSPACE_Z_RANGE_M} r<={WORKSPACE_RADIUS_XY_M}"
        print(f"[管线] m_obj={args.m_obj}  noise={args.noise_std}  工作空间裁剪={ws}")

        print(f"[预热] 丢弃前 {args.warmup} 帧...")
        for _ in range(args.warmup):
            cam.get_frame()

        polygon = parse_polygon_arg(args.polygon) if args.polygon else None
        if polygon is None:
            if not show_window:
                print("[错误] 无窗口环境请用 --polygon \"x,y x,y ...\" 指定")
                return
            polygon = select_polygon(cam)
            if polygon is None:
                print("[错误] 未标注有效多边形")
                return
            spec = " ".join(f"{x},{y}" for x, y in polygon)
            print(f"[手标] 多边形 {len(polygon)} 点  (复用: --polygon \"{spec}\")")
        mask = polygon_to_mask(polygon, args.height, args.width)
        poly_arr = np.asarray(polygon, dtype=np.int32)
        print(f"[掩码] 多边形面积 {int(mask.sum())} px")

        frame = 0
        t0 = time.time()
        vis_lo, vis_hi = float(args.depth_range[0]), float(args.depth_range[1])

        # 冻结的深度图。非 None 时反投影一律走它，实时帧只用来显示 RGB。
        # 冻结的是**深度图**而不是点云：掩码和 FPS 每帧照常重跑，所以点的选取
        # 仍是随机的、每帧不同，只有被采的那张深度底图不变。冻结点云的话每帧
        # 拿到的是同一组 64 个点，连采样噪声都没有了。
        frozen_depth = None
        frozen_at = None

        while True:
            rgb, depth = cam.get_frame()
            if rgb is None or depth is None:
                continue
            frame += 1

            if frozen_depth is None and args.freeze_at_start:
                frozen_depth, frozen_at = depth.copy(), frame
                print(f"[冻结] 第 {frame} 帧深度图已冻结（--freeze_at_start）")

            # 反投影的输入：冻结态用旧图，否则用当前帧
            src_depth = frozen_depth if frozen_depth is not None else depth
            pts, n_vis = build_pointcloud(src_depth, mask, K, args)

            bgr = rgb[..., ::-1].copy()
            depth_vis, d_lo, d_hi = colorize_depth(src_depth, vis_lo, vis_hi)
            for img in (bgr, depth_vis):
                cv2.polylines(img, [poly_arr], True, (0, 255, 0), 2)

            # 点云投回像素系：反投影自洽的话红点应精确落在工件上
            if len(pts):
                uvz = project_to_pixels(pts, K)
                for u, v, z in uvz:
                    if z > 0 and 0 <= u < args.width and 0 <= v < args.height:
                        cv2.circle(bgr, (int(u), int(v)), 2, (0, 0, 255), -1)
                        cv2.circle(depth_vis, (int(u), int(v)), 2, (0, 0, 255), -1)

            ok = "OK" if n_vis >= MIN_MASK_PIXELS else "LOW!"
            if len(pts):
                info = (f"frame {frame}  visible {n_vis}px {ok}  "
                        f"z {pts[:, 2].min():.3f}~{pts[:, 2].max():.3f}m  "
                        f"depth {d_lo:.2f}~{d_hi:.2f}m")
            else:
                info = f"frame {frame}  ROI 内无有效深度!"

            for img in (bgr, depth_vis):
                cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
                cv2.putText(img, info, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1, cv2.LINE_AA)

            # 冻结态画得刺眼一点：此时点云已经和画面里的现实脱节，工件被挪动
            # 也不会反映到点上。真机闭环下这正是撞机的前提，不能只在角落写一行小字。
            if frozen_depth is not None:
                age = frame - frozen_at
                warn = f"!! FROZEN depth @frame {frozen_at} (age {age}f) -- press f to refresh !!"
                for img in (bgr, depth_vis):
                    cv2.rectangle(img, (0, 22), (img.shape[1], 46), (0, 0, 200), -1)
                    cv2.putText(img, warn, (5, 39), cv2.FONT_HERSHEY_SIMPLEX, 0.42,
                                (255, 255, 255), 1, cv2.LINE_AA)
                    cv2.rectangle(img, (1, 1), (img.shape[1] - 2, img.shape[0] - 2),
                                  (0, 0, 200), 3)

            combined = np.hstack([bgr, depth_vis])
            if show_window:
                cv2.imshow("real ROI -> base-frame pointcloud", combined)
                key = cv2.waitKey(1) & 0xFF
                if key in (27, ord("q")):
                    break
                if key == ord("f"):
                    # 每次按 f 都取**当前**帧重新冻结，等于"刷新记忆"
                    frozen_depth, frozen_at = depth.copy(), frame
                    print(f"[冻结] 第 {frame} 帧深度图已冻结")
                elif key == ord("u"):
                    frozen_depth, frozen_at = None, None
                    print(f"[解冻] 第 {frame} 帧起恢复实时深度")
            if args.save_frames:
                cv2.imwrite(os.path.join(args.out, f"frame_{frame:05d}.png"), combined)
            if args.save_points:
                np.save(os.path.join(args.out, f"points_{frame:05d}.npy"), pts)
            if frame % 30 == 0:
                tag = f"  [FROZEN@{frozen_at}]" if frozen_depth is not None else ""
                print(f"{info}{tag}  ({frame / (time.time() - t0):.1f} fps)")
            if args.max_frames and frame >= args.max_frames:
                break

        os.makedirs(args.out, exist_ok=True)
        meta = {
            "intrinsics_used": args.intrinsics,
            "K": K.tolist(),
            "real_intrinsics": intr,
            "extrinsics_base_frame": {
                "pos": list(CALIB_CAM_POS_IN_BASE_M),
                "quat_wxyz": list(CALIB_CAM_QUAT_IN_BASE_WXYZ),
            },
            "polygon": [list(p) for p in polygon],
            "mask_pixels": int(mask.sum()),
            "m_obj": args.m_obj,
            "noise_std": args.noise_std,
            "workspace_filter": not args.no_workspace_filter,
            # 存盘的 npy 是实时深度还是冻结深度算出来的，事后从点云本身看不出来 ——
            # 不记这一条，那批点云回头就没法判读了。
            "depth_frozen": frozen_depth is not None,
            "frozen_at_frame": frozen_at,
            "frames": frame,
        }
        meta_path = os.path.join(args.out, "pointcloud_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"\n[输出] 元数据 → {meta_path}")

    finally:
        cam.stop()
        if show_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
