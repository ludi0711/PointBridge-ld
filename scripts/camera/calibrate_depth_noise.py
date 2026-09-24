#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""固定摆放、单次框选、连采多帧 → 标定仿真该加多大的深度噪声（NOISE_STD_M）。

**测的是什么**

仿真里 ``NOISE_STD_M`` 给点云每个坐标加高斯噪声，用意是让训练时的点云"脏"得和
真机一样。要标定它，就得知道真机的点云有多脏 —— 具体说是**同一个静止工件、同一
个掩码，帧与帧之间抖多少**。工件不动、标注不变，那么点云的一切变化都只能来自
D435 的深度测量本身，这正是 NOISE_STD_M 对标的东西。

所以流程就是：摆好工件不要动 → 框一次多边形 → 连采几百帧 → 统计逐点抖动。

**为什么不能只看质心**

质心会把噪声平均掉：64 个点各自抖 sigma，质心只抖 sigma/8。之前测到的"质心跳动
7.3mm"因此**严重低估**了单点噪声。这里统计逐点残差，不是质心。

**逐点残差怎么算**

FPS 是确定性的（规格 §12：起始点固定），同一张深度图 + 同一个掩码必然采出同样
的点。所以帧间第 i 个点可以直接对位相减 —— 它们对应工件上大致相同的位置。但深度
一抖，FPS 的选点也会跟着漂，对位就不严格了。故同时给两个口径：

  - ``paired`` : 逐点对位残差。**这是 sigma 的估计值。**
  - ``nn``     : 每点到参考帧的最近邻距离。只用来判断 FPS 选点稳不稳 ——
                 它系统性高估约 1.3 倍（残差成了两帧噪声之差，方差翻倍），
                 不能拿来当 sigma。

用法（**必须 gx_va_deploy 环境**）::

    conda activate gx_va_deploy

    # 摆好工件别动，框一次，采 300 帧
    python scripts/camera/calibrate_depth_noise.py

    # 采久一点
    python scripts/camera/calibrate_depth_noise.py --frames 600

    # 复用上次的多边形，不用再框
    python scripts/camera/calibrate_depth_noise.py \\
        --polygon logs/real_roi_pointcloud/pointcloud_meta.json

    # 只分析上次存的数据，不开相机
    python scripts/camera/calibrate_depth_noise.py --analyze_only
"""

import argparse
import json
import os
import sys

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
    parse_polygon_arg,
    polygon_to_mask,
    report_intrinsics,
    select_polygon,
)

# 与 real_roi_pointcloud.py 同源的标定外参（基座系原始值）。
# 那边有 _verify_extrinsics 与 configs 对账，直接复用，不再抄一份逻辑。
from scripts.camera.real_roi_pointcloud import (
    CALIB_CAM_POS_IN_BASE_M,
    CALIB_CAM_QUAT_IN_BASE_WXYZ,
    _verify_extrinsics,
)


def parse_args():
    p = argparse.ArgumentParser(description="标定真机深度噪声水平 → NOISE_STD_M")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--serial", type=str, default=None)
    p.add_argument("--warmup", type=int, default=30,
                   help="丢弃开头若干帧。D435 刚开流时自动曝光未收敛，深度明显更脏，"
                        "算进去会高估噪声。")
    p.add_argument("--frames", type=int, default=300,
                   help="连采帧数。300 帧 @30fps = 10 秒，足够了；"
                        "sigma 的估计误差约 1/sqrt(2*F*M)，F=300/M=64 时已 <1%%。")
    p.add_argument("--polygon", type=str, default=None,
                   help="复用已存多边形：\"x,y x,y ...\" 或 meta.json 路径。"
                        "不给就现场框。")
    p.add_argument("--m_obj", type=int, default=M_OBJ)
    p.add_argument("--out", type=str,
                   default=os.path.join(_PROJECT_DIR, "logs", "depth_noise_calib"))
    p.add_argument("--analyze_only", action="store_true",
                   help="跳过采集，直接分析 --out 里已有的数据")
    return p.parse_args()


def build_pointcloud(depth, mask, K, m_obj):
    """与 real_roi_pointcloud.build_pointcloud 同参数，但**强制 noise_std=0**。

    要测的就是真机自带的噪声，再叠一层仿真噪声等于自己污染自己的测量。
    """
    mask_t = torch.from_numpy(mask).unsqueeze(0)
    depth_t = torch.from_numpy(np.nan_to_num(depth, nan=0.0)).float().unsqueeze(0)
    K_t = torch.from_numpy(K).float().unsqueeze(0)
    pts, counts = mask_depth_to_pointcloud(
        mask=mask_t,
        depth=depth_t,
        K=K_t,
        cam_pos=torch.tensor(CALIB_CAM_POS_IN_BASE_M, dtype=torch.float32).unsqueeze(0),
        cam_quat=torch.tensor(CALIB_CAM_QUAT_IN_BASE_WXYZ, dtype=torch.float32).unsqueeze(0),
        m_obj=m_obj,
        noise_std=0.0,
        workspace_z_range_m=WORKSPACE_Z_RANGE_M,
        workspace_radius_xy_m=WORKSPACE_RADIUS_XY_M,
    )
    return pts[0].numpy(), int(counts[0])


def collect(args) -> dict:
    import cv2

    _verify_extrinsics()

    cam = RealsenseCamera(width=args.width, height=args.height, fps=args.fps,
                          align_on_device=True, serial_number=args.serial)
    cam.start()
    os.makedirs(args.out, exist_ok=True)

    try:
        intr = report_intrinsics(cam)
        K = np.asarray(cam.K_color, dtype=np.float64)
        print(f"[预热] 丢弃前 {args.warmup} 帧...")
        for _ in range(args.warmup):
            cam.get_frame()

        if args.polygon:
            polygon = parse_polygon_arg(args.polygon)
            print(f"[掩码] 复用已存多边形: {args.polygon}")
            print("       工件若已挪动，这份标注就失效了 —— 掩码会框到桌面，"
                  "测出来的是桌面的噪声，不报错。挪过就别加 --polygon。")
        else:
            print("\n把工件摆好，**采集期间不要碰它、不要碰相机**。")
            print("左键沿工件轮廓点一圈，回车确认。")
            polygon = select_polygon(cam)
            if polygon is None:
                raise SystemExit("[错误] 未标注有效多边形")
        mask = polygon_to_mask(polygon, args.height, args.width)
        print(f"[掩码] 多边形面积 {int(mask.sum())} px")

        print(f"[采集] 连采 {args.frames} 帧（约 {args.frames/args.fps:.0f} 秒），"
              "保持静止...")
        pts_list, vis_list = [], []
        for fi in range(args.frames):
            rgb, depth = cam.get_frame()
            if rgb is None or depth is None:
                continue
            pts, n_vis = build_pointcloud(depth, mask, K, args.m_obj)
            pts_list.append(pts)
            vis_list.append(n_vis)
            if (fi + 1) % 50 == 0:
                print(f"    {fi + 1}/{args.frames}  visible={n_vis}px")

        if not pts_list:
            raise SystemExit("[错误] 未采到任何有效帧")

        arr = np.stack(pts_list)                      # (F, M, 3)
        np.save(os.path.join(args.out, "pts.npy"), arr)
        vis_mean = float(np.mean(vis_list))
        vis_min = int(np.min(vis_list))
        if vis_mean < MIN_MASK_PIXELS:
            print(f"[警告] 平均有效像素 {vis_mean:.0f} < {MIN_MASK_PIXELS}，"
                  "FPS 输入大量重复，噪声估计不可信")

        record = {
            "K": K.tolist(), "real_intrinsics": intr,
            "polygon": [list(map(int, p)) for p in polygon],
            "mask_px": int(mask.sum()),
            "frames": len(pts_list),
            "visible_mean": vis_mean, "visible_min": vis_min,
            "m_obj": args.m_obj,
        }
        meta_path = os.path.join(args.out, "calib_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(record, f, indent=2, ensure_ascii=False)
        print(f"\n[输出] {arr.shape} → {os.path.join(args.out, 'pts.npy')}")
        print(f"[输出] {meta_path}")
        return record

    finally:
        cam.stop()
        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def _nn_dist(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """a 中每点到 b 的最近邻距离。点数只有 64，暴力算即可。

    **这个口径系统性高估噪声**，合成数据实测约 1.3 倍：点间距远大于噪声时最近邻
    几乎总配回"自己那个点"，于是残差是**两帧噪声之差**（方差翻倍，sqrt2），再被
    min 操作拉低一点，净剩 ~1.3。所以它只用来交叉验证 FPS 选点稳不稳，不作为
    sigma 的估计值 —— 估计值一律取 paired。
    """
    d = np.linalg.norm(a[:, None, :] - b[None, :, :], axis=-1)
    return d.min(axis=1)


def _stat(x: np.ndarray, name: str):
    """逐点三维距离 → 等效单轴 sigma。

    高斯各向同性时 E[|d|^2] = 3*sigma^2，所以 sigma = rms/sqrt(3)。
    NOISE_STD_M 是**单轴**标准差，直接拿三维距离对标会高估 sqrt(3)=1.73 倍。
    """
    if not len(x):
        print(f"  {name}: (无数据)")
        return None
    rms = float(np.sqrt((x ** 2).mean()))
    print(f"  {name}: 中位 {np.median(x)*1000:6.2f}mm  RMS {rms*1000:6.2f}mm  "
          f"p95 {np.percentile(x,95)*1000:6.2f}mm  "
          f"→ 等效单轴 sigma {rms/np.sqrt(3)*1000:5.2f}mm")
    return rms / np.sqrt(3)


def analyze(args, record: dict) -> None:
    path = os.path.join(args.out, "pts.npy")
    if not os.path.exists(path):
        raise SystemExit(f"[错误] 找不到 {path}")
    arr = np.load(path)                               # (F, M, 3)
    F, M, _ = arr.shape

    print("\n" + "=" * 70)
    print("  分析")
    print("=" * 70)
    print(f"  {F} 帧 x {M} 点，平均有效像素 {record.get('visible_mean', 0):.0f}"
          f"（最低 {record.get('visible_min', 0)}）")

    c = arr.reshape(-1, 3).mean(axis=0)
    cam_d = float(np.linalg.norm(c - np.array(CALIB_CAM_POS_IN_BASE_M)))
    print(f"  质心(基座系) x={c[0]:+.3f} y={c[1]:+.3f} z={c[2]:+.3f}"
          f"   距相机 {cam_d:.3f} m")

    # ── 逐点残差 ──
    # 参考帧的两种取法，重排时结果差很多：
    #   逐点时间平均 arr.mean(0) —— 索引稳定时是最优估计（平均掉噪声）；但重排时
    #                              它把工件上**不同位置**的点平均到了一起，参考本身
    #                              就是个不存在的虚点，最近邻口径也会跟着虚高。
    #   单帧 arr[0]             —— 带一份噪声（使残差方差翻倍），但至少是真实点集。
    # 所以最近邻对**单帧**算，对位仍对时间平均算（重排时它本来就要被弃用）。
    ref = arr.mean(axis=0)
    paired = np.linalg.norm(arr - ref[None], axis=-1).ravel()
    step = max(1, F // 50)
    nn = np.concatenate([_nn_dist(arr[f], arr[0]) for f in range(1, F, step)])

    print("\n── 帧间抖动（= 传感器噪声）──")
    s_paired = _stat(paired, "  对位")
    s_nn = _stat(nn, "  最近邻  ★")
    # 两个口径谁大谁小，直接决定该信哪个：
    #   paired >> nn  → FPS 选点在帧间重排，同一索引已不是同一物理点。paired 测的
    #                   是"点在工件表面上滑动多远"，与深度噪声无关，必须弃用。
    #   nn >> paired  → 不该发生；真出现说明点集本身在变形。
    # 第一版这里写成 `s_nn > 1.8 * s_paired`，方向反了，恰好漏掉了真实会发生的
    # 那一种，实测 300 帧数据时静默放行了一个 92mm 的假 sigma。
    reorder = bool(s_paired and s_nn and s_paired > 1.8 * s_nn)
    if reorder:
        print(f"\n  [重排] 对位({s_paired*1000:.1f}mm) >> 最近邻({s_nn*1000:.1f}mm)，"
              f"{s_paired/s_nn:.1f} 倍。")
        print("         FPS 在帧间重排了选点：同一索引的点已不对应同一物理位置，")
        print("         对位残差测的是'点沿工件表面滑了多远'，不是深度噪声。")

    # ── 排序口径：对重排免疫 ──
    # 把每帧的点按坐标字典序重排后再逐点对位。工件静止时点集整体不变，排序给出
    # 一个与 FPS 输出顺序无关的规范序，于是"同一索引"重新变得有意义。
    # 它不完美：点在空间中挨得近时，噪声会让排序名次互换，残差被高估一点。但在
    # 重排已经把对位口径彻底毁掉的情况下，这是唯一还能用的逐点估计。
    srt = np.stack([f[np.lexsort((f[:, 2], f[:, 1], f[:, 0]))] for f in arr])
    srt_ref = srt.mean(axis=0)
    sorted_resid = np.linalg.norm(srt - srt_ref[None], axis=-1).ravel()
    print("\n── 排序后（对 FPS 重排免疫）──")
    s_sorted = _stat(sorted_resid, "  排序对位" + ("  ★" if reorder else "  "))

    # 最终 sigma。重排时用排序口径，否则用对位。
    # 最近邻不作为估计值：它系统性高估约 1.3 倍（见 _nn_dist），只用来触发重排告警。
    sigma = s_sorted if reorder else s_paired
    src = "排序对位" if reorder else "对位"

    # ── 分轴 ──
    # 用与 sigma 同一口径的残差，否则重排时长轴方向会虚高到工件尺寸量级，
    # 看着像"各向异性 20 倍"，其实只是点在长条上滑动。
    resid = ((srt - srt_ref[None]) if reorder else (arr - ref[None])).reshape(-1, 3)
    per_axis = resid.std(axis=0)
    print(f"\n── 分轴 sigma（基座系，{src}口径）──")
    print(f"  x {per_axis[0]*1000:5.2f}mm   y {per_axis[1]*1000:5.2f}mm   "
          f"z {per_axis[2]*1000:5.2f}mm")
    aniso = float(per_axis.max() / max(per_axis.min(), 1e-9))
    if aniso > 2.0:
        print(f"  各向异性 {aniso:.1f}x —— 噪声明显不是各向同性的，而 NOISE_STD_M "
              "是三轴同一个值。用最大轴取值会让另外两轴过噪，用平均则最脏那轴欠噪。")
        if reorder:
            print("  [保留怀疑] 排序口径下残差仍受排序名次互换影响，长轴方向偏大"
                  "可能是残余伪影而非真实各向异性。")
    else:
        print(f"  各向异性 {aniso:.1f}x —— 接近各向同性，单一 NOISE_STD_M 够用。")

    # ── 与顺序无关的稳定性检查 ──
    # 这几个量与点的顺序无关，能独立证明"工件到底动没动"。它们小 = 点云整体稳定
    # = 前面那些大数字确实只是索引在换，而不是工件真的在动。
    cen = arr.mean(axis=1)
    ext = arr.max(axis=1) - arr.min(axis=1)
    print(f"\n── 与顺序无关的稳定性 ──")
    print(f"  质心抖动 std     x {cen.std(0)[0]*1000:5.2f}  y {cen.std(0)[1]*1000:5.2f}"
          f"  z {cen.std(0)[2]*1000:5.2f} mm")
    print(f"  包围盒尺寸 std   x {ext.std(0)[0]*1000:5.2f}  y {ext.std(0)[1]*1000:5.2f}"
          f"  z {ext.std(0)[2]*1000:5.2f} mm")
    print("  这两项小 = 工件没动、点云整体形状稳定（与点的顺序无关，重排不影响）。")

    # ── 结论 ──
    print("\n" + "=" * 70)
    print("  结论")
    print("=" * 70)
    print(f"  当前仿真 NOISE_STD_M = {NOISE_STD_M} m ({NOISE_STD_M*1000:.1f}mm，单轴)")
    if sigma is None:
        print("  [无有效数据]")
        return
    print(f"  实测传感器 sigma     = {sigma:.4f} m ({sigma*1000:.2f}mm，单轴，{src}口径)")
    if reorder:
        print(f"  （对位口径给出的 {s_paired*1000:.1f}mm 是重排伪影，已弃用）")

    r = NOISE_STD_M / sigma
    print()
    if r > 1.5:
        print(f"  → 仿真噪声是实测的 {r:.1f} 倍，**偏大**。")
        print("    这个方向本身不会让策略失效 —— 真机比训练时更干净，策略只会觉得"
              "世界变简单了。代价是精度上限：训练时被迫学会容忍 10mm 的抖动，就学不出"
              "依赖亚毫米细节的策略。")
        print(f"    想贴合实测就改成 {sigma:.4f}；但先想清楚要不要留余量 —— "
              "换个工件材质、换个光照、工件摆到量程更远处，真机噪声都会变大，"
              "调太紧会让这些情况变成分布外。")
    elif r < 0.67:
        print(f"  → 仿真噪声只有实测的 {r:.2f} 倍，**偏小**。")
        print("    这是会直接掉性能的方向：策略在真机上会遇到训练时没见过的脏点云。")
        print(f"    建议至少调到 {sigma:.4f}，留余量则更高。")
    else:
        print(f"  → 与实测同量级（比值 {r:.2f}），不用改。")

    print("\n  测量范围提醒：这次只测了工件在一个固定位置、一种姿态下的噪声。")
    print("  随机化框是 0.25x0.25 + yaw ±90°，工件摆到框的远端、或转成大角度时，")
    print("  可见面积和入射角都会变，噪声未必一样。要覆盖就换个位置再跑一次。")


def main() -> None:
    args = parse_args()
    if args.analyze_only:
        meta_path = os.path.join(args.out, "calib_meta.json")
        if not os.path.exists(meta_path):
            raise SystemExit(f"[错误] 找不到 {meta_path}，先跑一次采集")
        record = json.load(open(meta_path, encoding="utf-8"))
    else:
        record = collect(args)
    analyze(args, record)


if __name__ == "__main__":
    main()
