#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""实机 D435 数据流查看 + 工件深度采样（YOLO 之前的替身）。

分两阶段，用 ``--stage`` 切：

    stage 1  只看数据流：RGB / 深度伪彩并排显示，读实机内参并与仿真 canonical
             内参对照，确认实机这一路本身是对的。
    stage 2  点选多边形圈住工件 → 记录掩码内深度统计 → 之后每帧在掩码内**采样**
             M 个点，模拟 point_bridge 训练时的取点行为。

为什么要有 stage 2：仿真里工件像素由实例分割给出，真机没有分割就只能靠 YOLO。
YOLO 先略过，用手标的多边形顶替。多边形是静止的，所以只有工件不动时才有意义，
但足以回答"真机深度在工件那块区域的数值/噪声/空洞率是什么样"。这个问题和
YOLO 好不好用无关，先单独测掉。

用多边形而不是矩形：矩形必然把工件周围的桌面框进来，那些像素深度有效，会被当成
工件点采走，统计和采样都被污染。多边形能贴着轮廓收边，与实例分割掩码形状同类。

坐标约定与训练路径对齐（见 configs/xarm7_pick_pointcloud_env_cfg.py）:
  - 深度取"沿光轴到成像平面的距离"(z-depth)，与仿真的
    ``distance_to_image_plane`` 同义。RealSense 对齐后的深度本来就是 z-depth，
    不需要再换算 —— 别误当成到光心的欧氏距离。
  - 深度 0 表示无效（没打到东西 / 超量程），与仿真
    ``depth_clipping_behavior="zero"`` 一致，统一按 0 过滤。

用法::

    # 阶段一：看数据流
    python scripts/camera/view_real_depth.py

    # 阶段一，无显示器时存图自检
    python scripts/camera/view_real_depth.py --save_frames --max_frames 1

    # 阶段二：左键沿工件轮廓点一圈，回车确认
    python scripts/camera/view_real_depth.py --stage 2

    # 复用上次标好的多边形，跳过手标
    python scripts/camera/view_real_depth.py --stage 2 \
        --polygon "310,220 380,225 385,275 305,270"
    python scripts/camera/view_real_depth.py --stage 2 \
        --polygon logs/real_camera/real_camera_meta.json
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

from scripts.camera.realsense_camera import RealsenseCamera

# 仿真侧 canonical 针孔模型。这里**只读不用**，用于和实机实测内参对照 ——
# 两者差得远就说明"实机和仿真摆放一致"这个前提不成立，后面全白做。
# 取自 configs/xarm7_pick_pointcloud_env_cfg.py 的 CANONICAL_* 常量。
SIM_CANONICAL = {"width": 640, "height": 480, "fx": 615.0, "fy": 615.0,
                 "cx": 320.0, "cy": 240.0}

# 规格 §2.2 的掩码像素门槛。低于它点云会退化成一小撮重复点。
MIN_MASK_PIXELS = 200


def parse_args():
    p = argparse.ArgumentParser(description="实机 D435 RGB/深度查看与工件深度采样")
    p.add_argument("--stage", type=int, default=1, choices=(1, 2),
                   help="1=只看数据流；2=点选多边形圈住工件并采样深度")
    p.add_argument("--width", type=int, default=640)
    p.add_argument("--height", type=int, default=480)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--serial", type=str, default=None,
                   help="多相机时指定序列号；仿真内参标的是 254622072913")
    p.add_argument("--warmup", type=int, default=30,
                   help="丢弃前 N 帧。自动曝光和深度滤波要几十帧才稳。")
    p.add_argument("--num_samples", type=int, default=64,
                   help="stage 2 每帧在掩码内采样的点数，默认与 M_OBJ 一致")
    p.add_argument("--polygon", type=str, default=None,
                   help="跳过手标，直接给多边形顶点：\"x1,y1 x2,y2 x3,y3 ...\"；"
                        "也可传上次存的 JSON 路径。")
    p.add_argument("--depth_range", type=float, nargs=2, default=(0.3, 1.5),
                   metavar=("LO", "HI"),
                   help="深度伪彩的固定色标区间(米)。默认 0.3~1.5 覆盖桌面作业距离"
                        "（标定相机离基座约 1.0m）。区间外钳位不丢弃。")
    p.add_argument("--auto_range", action="store_true",
                   help="改回逐帧 min/max 自适应。会闪，只在不知道该设多少时用一次"
                        "看看实际范围，然后填回 --depth_range。")
    p.add_argument("--max_frames", type=int, default=0, help="跑够帧数退出；0=一直跑")
    p.add_argument("--save_frames", action="store_true", help="每帧存盘（无显示器时用）")
    p.add_argument("--out", type=str,
                   default=os.path.join(_PROJECT_DIR, "logs", "real_camera"))
    p.add_argument("--no_window", action="store_true", help="不弹窗，只存盘/打印")
    return p.parse_args()


def colorize_depth(depth: np.ndarray, lo: float, hi: float
                   ) -> "tuple[np.ndarray, float, float]":
    """深度 → turbo 伪彩(BGR)，按**固定**的 [lo, hi] 区间定标。

    刻意不用逐帧 min/max 自适应：那样每帧的色标都不一样，场景里任何一点风吹草动
    （手伸进画面、远处出现反光空洞）都会把整幅图的配色重映射一遍，看起来就是闪。
    固定区间后同一个颜色在任何一帧都对应同一个深度，帧间才可比 —— 这也是把它当
    测量工具而不是好看图的前提。

    区间外的值**钳位**而不是丢弃：近于 lo 的一律最红、远于 hi 的一律最蓝，
    仍然可见，只是不再参与拉伸。返回的 lo/hi 是本帧有效像素的真实 min/max，
    只用于标题显示，不影响配色。

    与 scripts/train/dump_camera_data_types.py 的差别就在这里：那个是离线存图，
    一帧一个色标无所谓；这个是实时流，必须固定。
    """
    import cv2

    valid = np.isfinite(depth) & (depth > 0)
    if not valid.any():
        return np.zeros((*depth.shape, 3), np.uint8), 0.0, 0.0
    d_lo, d_hi = float(depth[valid].min()), float(depth[valid].max())

    norm = np.zeros_like(depth, dtype=np.float32)
    if hi > lo:
        norm[valid] = np.clip((depth[valid] - lo) / (hi - lo), 0.0, 1.0)
    img = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    img[~valid] = 0  # 无效像素涂黑，与真实的近距离区分开
    return img, d_lo, d_hi


def select_polygon(cam, max_wait_s: float = 300.0) -> "list[tuple[int, int]] | None":
    """在实时画面上点选多边形圈出工件。返回顶点列表，取消则 None。

    用多边形而不是矩形：矩形一定会把工件周围的桌面框进去，那些像素深度有效，
    会被当成工件点采走 —— 点云里就混进一圈贴着桌面的点。多边形能贴着工件轮廓
    收边，是 YOLO 分割接上之前最接近真实掩码的替身。

    走**实时**画面而不是单帧快照：标注时能看到工件是否被自动曝光带偏、深度是否
    正在闪，边标边确认。代价是画面在动，手要稳一点。

    操作：左键加点 / 右键或退格删上一个 / 回车确认（>=3 点）/ r 重来 / q 取消。
    """
    import cv2

    pts: "list[tuple[int, int]]" = []

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            pts.append((x, y))
        elif event == cv2.EVENT_RBUTTONDOWN and pts:
            pts.pop()

    win = "annotate object polygon"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.setMouseCallback(win, on_mouse)
    print("\n[手标] 左键加点，右键删点，回车确认(>=3点)，r 重来，q 取消")

    t0 = time.time()
    try:
        while time.time() - t0 < max_wait_s:
            rgb, _ = cam.get_frame()
            if rgb is None:
                continue
            disp = rgb[..., ::-1].copy()

            if pts:
                arr = np.array(pts, dtype=np.int32)
                if len(pts) >= 3:
                    # 填充预览用叠加而不是直接涂，否则挡住工件没法判断边贴不贴
                    ov = disp.copy()
                    cv2.fillPoly(ov, [arr], (0, 255, 0))
                    disp = cv2.addWeighted(disp, 0.75, ov, 0.25, 0)
                if len(pts) >= 2:
                    cv2.polylines(disp, [arr], len(pts) >= 3, (0, 255, 0), 2)
                for i, p in enumerate(pts):
                    cv2.circle(disp, p, 4, (0, 255, 0), -1)
                    cv2.putText(disp, str(i + 1), (p[0] + 8, p[1] - 8),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            hint = f"points={len(pts)}  L:add  R:undo  Enter:ok  r:reset  q:cancel"
            cv2.rectangle(disp, (0, 0), (disp.shape[1], 22), (0, 0, 0), -1)
            cv2.putText(disp, hint, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                        (255, 255, 255), 1, cv2.LINE_AA)
            cv2.imshow(win, disp)

            key = cv2.waitKey(1) & 0xFF
            if key in (13, 10):                      # Enter
                if len(pts) >= 3:
                    return pts
                print("[手标] 至少要 3 个点才能围成多边形")
            elif key in (8, 127):                    # Backspace / Delete
                if pts:
                    pts.pop()
            elif key == ord("r"):
                pts.clear()
            elif key in (27, ord("q")):              # Esc / q
                return None
        print("[手标] 超时未确认")
        return None
    finally:
        cv2.destroyWindow(win)


def polygon_to_mask(polygon, height: int, width: int) -> np.ndarray:
    """多边形顶点 → bool 掩码。顶替仿真里的实例分割掩码。"""
    import cv2

    mask = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(mask, [np.asarray(polygon, dtype=np.int32)], 1)
    return mask.astype(bool)


def polygon_bbox(polygon) -> "tuple[int, int, int, int]":
    """多边形外接矩形 (x, y, w, h)，只用于画框和显示，不参与采样。"""
    arr = np.asarray(polygon, dtype=np.int32)
    x, y = int(arr[:, 0].min()), int(arr[:, 1].min())
    return x, y, int(arr[:, 0].max()) - x + 1, int(arr[:, 1].max()) - y + 1


def parse_polygon_arg(spec: str) -> "list[tuple[int, int]]":
    """``--polygon`` 的值 → 顶点列表。支持 "x1,y1 x2,y2 ..." 或 JSON 文件路径。

    JSON 认两种形状：``[[x,y],...]``，或本脚本存的 ``{"polygon": [[x,y],...]}``。
    """
    if os.path.isfile(spec):
        with open(spec, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("polygon", data.get("roi_polygon"))
        if not data:
            raise ValueError(f"{spec} 里没找到 polygon 字段")
        return [(int(a), int(b)) for a, b in data]

    pts = []
    for tok in spec.replace(";", " ").split():
        a, b = tok.split(",")
        pts.append((int(a), int(b)))
    if len(pts) < 3:
        raise ValueError(f"多边形至少 3 个点，得到 {len(pts)}")
    return pts


def quat_to_matrix(q) -> np.ndarray:
    """四元数 (w,x,y,z) → 3x3 旋转矩阵。

    与 configs/xarm7_pick_pointcloud_env_cfg.py 的 ``_quat_to_matrix`` 同式
    （也就是 isaaclab 的 ``matrix_from_quat``），在这里重写一份只是为了让真机
    脚本不必依赖 isaaclab —— 真机环境未必装得上。顺序是 wxyz，**不是** xyzw，
    传错不会报错，只会让点云绕轴转一个角度。
    """
    w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def report_intrinsics(cam: RealsenseCamera) -> dict:
    """打印实机内参并与仿真 canonical 对照。"""
    K = cam.K_color
    fx, fy = float(K[0, 0]), float(K[1, 1])
    cx, cy = float(K[0, 2]), float(K[1, 2])

    print("=" * 78)
    print("实机彩色相机内参（深度已对齐到彩色，故反投影用这一组）")
    print(f"  fx={fx:.2f}  fy={fy:.2f}  cx={cx:.2f}  cy={cy:.2f}")
    print(f"  depth_scale={cam.depth_scale}  (原始 uint16 乘它得米)")
    print("-" * 78)
    print("仿真 canonical（configs/xarm7_pick_pointcloud_env_cfg.py）")
    print(f"  fx={SIM_CANONICAL['fx']:.2f}  fy={SIM_CANONICAL['fy']:.2f}"
          f"  cx={SIM_CANONICAL['cx']:.2f}  cy={SIM_CANONICAL['cy']:.2f}")
    print("-" * 78)

    dfx, dfy = fx - SIM_CANONICAL["fx"], fy - SIM_CANONICAL["fy"]
    dcx, dcy = cx - SIM_CANONICAL["cx"], cy - SIM_CANONICAL["cy"]
    print(f"差值  dfx={dfx:+.2f}  dfy={dfy:+.2f}  dcx={dcx:+.2f}  dcy={dcy:+.2f}")
    # canonical fx 取 615 是**刻意**比实测 605 大的（FOV 略窄，保证每个 canonical
    # 像素都落在原始帧内）。所以 ~10 的差是设计值，不是错误。
    if abs(dfx) > 25 or abs(dfy) > 25:
        print("  [警告] 焦距差 >25px，超出 canonical 的设计余量，检查分辨率是否为 640x480")
    else:
        print("  焦距差在设计余量内（canonical 刻意取大约 10px 以留边界裕度）")
    if abs(dcx) > 20 or abs(dcy) > 20:
        print("  [警告] 主点差 >20px。仿真强制主点居中，实机偏心过大会让反投影系统性偏斜")
    print("=" * 78)

    return {"fx": fx, "fy": fy, "cx": cx, "cy": cy,
            "depth_scale": float(cam.depth_scale),
            "width": cam.width, "height": cam.height}


def roi_depth_stats(depth: np.ndarray, mask: np.ndarray) -> dict:
    """掩码内深度统计。空洞率是这里最该看的数 —— 它决定采样能不能取满点。"""
    valid = mask & np.isfinite(depth) & (depth > 0)
    n_valid = int(valid.sum())
    n_total = int(mask.sum())
    stats = {"n_total": n_total, "n_valid": n_valid,
             "hole_ratio": float(1.0 - n_valid / n_total) if n_total else 1.0}
    if n_valid:
        v = depth[valid]
        stats.update({
            "min": float(v.min()), "max": float(v.max()),
            "mean": float(v.mean()), "median": float(np.median(v)),
            "std": float(v.std()),
        })
    return stats


def sample_roi_depth(depth: np.ndarray, mask: np.ndarray, num_samples: int, rng
                     ) -> np.ndarray:
    """在掩码内**有效**深度像素中随机采 num_samples 个点。

    返回 (N, 3) 的 (u, v, z)，z 单位米。有效点不足时**有放回**采样补齐 ——
    这与仿真侧点数不足时的重复填充行为一致（都会让点云退化成一小撮重复点，
    所以 n_valid 才是真正要盯的指标，不是返回的点数）。有效点为 0 时返回空数组。
    """
    valid = mask & np.isfinite(depth) & (depth > 0)
    vy, vx = np.nonzero(valid)
    if vy.size == 0:
        return np.zeros((0, 3), np.float32)

    replace = vy.size < num_samples
    idx = rng.choice(vy.size, size=num_samples, replace=replace)
    return np.stack([vx[idx].astype(np.float32),
                     vy[idx].astype(np.float32),
                     depth[vy[idx], vx[idx]].astype(np.float32)], axis=-1)


def main() -> None:
    args = parse_args()
    import cv2

    rng = np.random.default_rng(0)
    show_window = not args.no_window
    if args.save_frames:
        os.makedirs(args.out, exist_ok=True)

    cam = RealsenseCamera(width=args.width, height=args.height, fps=args.fps,
                          align_on_device=True, serial_number=args.serial)
    cam.start()

    try:
        intr = report_intrinsics(cam)

        # 预热：自动曝光/深度滤波要几十帧才稳，太早取帧会拿到偏暗或空洞很多的图
        print(f"[预热] 丢弃前 {args.warmup} 帧...")
        for _ in range(args.warmup):
            cam.get_frame()

        polygon = parse_polygon_arg(args.polygon) if args.polygon else None
        mask = None
        if args.stage == 2 and polygon is None:
            if not show_window:
                print("[错误] stage 2 需要手标；无窗口环境请用 --polygon \"x,y x,y ...\"")
                return
            polygon = select_polygon(cam)
            if polygon is None:
                print("[错误] 未标注有效多边形，退出")
                return
            spec = " ".join(f"{x},{y}" for x, y in polygon)
            print(f"[手标] 多边形 {len(polygon)} 点  (复用: --polygon \"{spec}\")")
        if polygon is not None:
            mask = polygon_to_mask(polygon, args.height, args.width)
            print(f"[掩码] 多边形面积 {int(mask.sum())} px")

        # stage 2 的基线：标完框先记一次深度统计，后续每帧与它对照，
        # 用来看深度是在漂移还是只是逐帧噪声。
        baseline = None
        frame = 0
        # 循环一帧都没跑成时（相机拔了/取帧一直失败）下面存 meta 仍要用，先给定值
        vis_lo, vis_hi = float(args.depth_range[0]), float(args.depth_range[1])
        t0 = time.time()

        while True:
            rgb, depth = cam.get_frame()
            if rgb is None or depth is None:
                continue
            frame += 1

            # 固定色标：同一颜色在任何一帧都对应同一深度，否则整幅图逐帧重映射会闪。
            # --auto_range 传 (min,max) 退化回自适应，仅用于摸清实际范围。
            if args.auto_range:
                v = np.isfinite(depth) & (depth > 0)
                vis_lo, vis_hi = ((float(depth[v].min()), float(depth[v].max()))
                                  if v.any() else (0.0, 1.0))
            else:
                vis_lo, vis_hi = float(args.depth_range[0]), float(args.depth_range[1])
            depth_vis, lo, hi = colorize_depth(depth, vis_lo, vis_hi)
            bgr = rgb[..., ::-1].copy()

            valid = np.isfinite(depth) & (depth > 0)
            # 标题里同时给"本帧真实范围"和"当前色标区间"——前者是数据，后者是显示设定，
            # 混在一起看会误判成深度在跳。
            info = (f"frame {frame}  depth {lo:.2f}~{hi:.2f}m  "
                    f"[scale {vis_lo:.2f}~{vis_hi:.2f}]  "
                    f"valid {100.0 * valid.mean():.1f}%")

            if args.stage == 2:
                stats = roi_depth_stats(depth, mask)
                if baseline is None and stats["n_valid"] > 0:
                    baseline = stats
                    print("\n" + "=" * 78)
                    print("[基线] 标注时刻的掩码内深度")
                    print(f"  有效 {stats['n_valid']}/{stats['n_total']} "
                          f"(空洞率 {100 * stats['hole_ratio']:.1f}%)")
                    print(f"  深度 {stats['min']:.3f}~{stats['max']:.3f} m  "
                          f"中位数 {stats['median']:.3f}  std {stats['std']:.4f}")
                    if stats["n_valid"] < MIN_MASK_PIXELS:
                        print(f"  [警告] 有效像素 <{MIN_MASK_PIXELS}，采样会大量重复点，"
                              f"点云会退化 —— 多边形圈大一点或让工件离相机近一点")
                    print("=" * 78 + "\n")

                pts = sample_roi_depth(depth, mask, args.num_samples, rng)
                poly_arr = np.asarray(polygon, dtype=np.int32)
                for img in (bgr, depth_vis):
                    cv2.polylines(img, [poly_arr], True, (0, 255, 0), 2)
                    for u, v, _z in pts:
                        cv2.circle(img, (int(u), int(v)), 2, (0, 0, 255), -1)

                if stats["n_valid"]:
                    drift = ""
                    if baseline is not None:
                        drift = f"  漂移 {stats['median'] - baseline['median']:+.4f}m"
                    info += (f"  | 掩码有效 {stats['n_valid']}/{stats['n_total']}"
                             f"  中位 {stats['median']:.3f}m"
                             f"  std {stats['std']:.4f}{drift}")
                else:
                    info += "  | 掩码内无有效深度!"

            for img in (bgr, depth_vis):
                cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
                cv2.putText(img, info, (5, 15), cv2.FONT_HERSHEY_SIMPLEX, 0.45,
                            (255, 255, 255), 1, cv2.LINE_AA)

            combined = np.hstack([bgr, depth_vis])

            if show_window:
                cv2.imshow("real D435  left=RGB  right=depth", combined)
                if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
                    break
            if args.save_frames:
                cv2.imwrite(os.path.join(args.out, f"frame_{frame:05d}.png"), combined)
            if frame % 30 == 0:
                print(f"{info}  ({frame / (time.time() - t0):.1f} fps)")
            if args.max_frames and frame >= args.max_frames:
                break

        # 内参与 stage2 基线存盘，供后续反投影/对照使用
        meta = {"intrinsics": intr, "sim_canonical": SIM_CANONICAL,
                "depth_vis_range": [vis_lo, vis_hi]}
        if args.stage == 2:
            meta["polygon"] = [list(p) for p in polygon]
            meta["mask_pixels"] = int(mask.sum())
            meta["baseline"] = baseline
        os.makedirs(args.out, exist_ok=True)
        meta_path = os.path.join(args.out, "real_camera_meta.json")
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2, ensure_ascii=False)
        print(f"\n[输出] 内参/基线 → {meta_path}")
        if args.save_frames:
            print(f"[输出] 已存 {frame} 帧 → {args.out}")

    finally:
        cam.stop()
        if show_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
