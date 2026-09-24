#!/usr/bin/env python3
"""YOLO 分割 + RealSense 深度 → Point Bridge 点云，目视核对部署链路。

这是阶段 2 在真机上的对应物：仿真用渲染器的 ``instance_id_segmentation_fast``
拿掩码，真机用 YOLO 实例分割拿掩码，之后**共用同一个**
:func:`configs.point_bridge_pointcloud.mask_depth_to_pointcloud`（规格 §1.1）。
共用是关键 —— 反投影、FPS、裁剪、噪声的任何差异都会变成 sim2real gap，而这类
gap 很难在下游发现，因为策略照样会输出看起来合理的动作。

坐标系（默认 ``--frame camera``）：点在**相机光学系**（+X 右、+Y 下、+Z 前）。
为什么默认不给基座系：手眼标定文件在本仓库里不存在，而 ``xarm_va`` 的
``EXTRINSICS_CONFIRMED`` 至今为 False。用未确认的外参画基座系点云，会把"分割/
深度对不对"和"外参对不对"两个问题混在一起，而这个脚本只想回答前者。
``--frame base`` 可以用仿真那套外参转到基座系，但它是**未标定**的，只能看量级。

用法::

    conda run -n gx_va_deploy python scripts/deploy/test_yolo_pointcloud.py
    conda run -n gx_va_deploy python scripts/deploy/test_yolo_pointcloud.py --save out.png
    conda run -n gx_va_deploy python scripts/deploy/test_yolo_pointcloud.py --live
"""

from __future__ import annotations

import argparse
import os
import sys

# YOLO fork 必须比任何 pip 装的 ultralytics 先进 sys.path，否则拿不到
# GongjianSegment26 多任务头，result.gongjian 会不存在。
YOLO_REPO = "/home/gxai/Desktop/GSworld/gongjian_multitask_training"
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SCRIPTS_DIR = os.path.dirname(_SCRIPT_DIR)
_PROJECT_DIR = os.path.dirname(_SCRIPTS_DIR)
for _p in (YOLO_REPO, _PROJECT_DIR, os.path.join(_SCRIPTS_DIR, "camera")):
    if os.path.isdir(_p) and _p not in sys.path:
        sys.path.insert(0, _p)

def _drop_isaac_paths() -> list[str]:
    """把 Isaac Sim 的捆绑包路径从 ``sys.path`` 里摘掉。

    从 isaaclab 的 conda shell 里启动时（含 ``conda run -n gx_va_deploy``），
    ``PYTHONPATH`` 会把 Isaac 的 ``pip_prebundle`` 带进来。那里面的 filelock 是
    3.13.1，没有 ``AsyncFileLock``，而 YOLO fork 的 ``data/converter.py`` 要用它
    —— 于是 import 直接失败。Isaac 的路径排在前面就会盖掉环境里的新版本。

    这个脚本纯粹是真机侧的，不需要任何 Isaac 运行时（共享的点云函数只用
    ``isaaclab.utils.math``，是纯 torch 实现，装在 gx_va_deploy 里）。所以最干净
    的处理是直接摘掉，让脚本对调用方的 shell 免疫。
    """
    bad = [p for p in sys.path if "_isaac_sim" in p or "isaac-sim" in p.lower()]
    for p in bad:
        sys.path.remove(p)
    # 已经被 Isaac 那份污染过的模块要清掉，否则后续 import 拿的还是缓存
    for mod in ("filelock",):
        loaded = sys.modules.get(mod)
        if loaded is not None and "_isaac_sim" in (getattr(loaded, "__file__", "") or ""):
            del sys.modules[mod]
    return bad


_DROPPED = _drop_isaac_paths()

import numpy as np
import torch

DEFAULT_WEIGHTS = os.path.join(YOLO_REPO, "0806_weights", "best.pt")
D435_SERIAL = "254622072913"   # fx=605.4，与仿真标定的 615.0 同量级
D405_SERIAL = "230322270207"   # fx=390.7，视场差很多，不是仿真对标的那台
DEFAULT_SERIAL = D435_SERIAL


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--weights", default=DEFAULT_WEIGHTS)
    p.add_argument("--serial", default=DEFAULT_SERIAL,
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
    p.add_argument("--save", default=None, help="存图路径，不给则弹窗显示")
    p.add_argument("--live", action="store_true", help="连续取帧刷新，Ctrl-C 退出")
    p.add_argument("--all-instances", action="store_true",
                   help="合并所有工件实例的掩码；默认只取 selection_score 最高的那个")
    return p.parse_args()


def pick_target(result, all_instances: bool):
    """从 YOLO 结果里挑出工件掩码。

    模型是两类：0=gongjian（工件）、1=kong（孔）。点云只要工件，孔要排除 ——
    孔的掩码在工件内部，混进去会让点云出现空腔，而 FPS 会把点分配到那些边缘上。

    多任务头额外给 ``selection_score``（该抓哪个）和 ``side``（正/反面）。默认
    取 score 最高的单个实例，对应真机"先抓最该抓的那个"，也与仿真里场上只有一
    个工件的设定一致。
    """
    if result.masks is None or result.boxes is None or len(result.boxes) == 0:
        return None, []

    cpu = result.cpu()
    cls = cpu.boxes.cls.int().tolist()
    obj_idx = [i for i, c in enumerate(cls) if c == 0]
    if not obj_idx:
        return None, []

    info = []
    g = cpu.gongjian
    for i in obj_idx:
        rec = {"index": i, "conf": float(cpu.boxes.conf[i])}
        # 多任务字段在纯分割权重上可能为空，逐个兜底而不是整体假定存在
        if g is not None and len(getattr(g, "selection_score", [])) > i:
            rec["selection_score"] = float(g.selection_score[i])
            rec["side"] = g.side[i] if len(g.side) > i else "?"
            rec["back_probability"] = float(g.back_probability[i])
        info.append(rec)

    if all_instances:
        chosen = obj_idx
    else:
        key = "selection_score" if "selection_score" in info[0] else "conf"
        best = max(info, key=lambda r: r.get(key, 0.0))
        chosen = [best["index"]]

    # masks.data 是 (N, h, w) 且 h/w 是网络输入尺度，需要缩回原图
    md = cpu.masks.data.numpy()
    H, W = result.orig_shape
    mask = np.zeros((H, W), dtype=bool)
    import cv2
    for i in chosen:
        m = md[i]
        if m.shape != (H, W):
            m = cv2.resize(m.astype(np.float32), (W, H), interpolation=cv2.INTER_NEAREST)
        mask |= m > 0.5
    return mask, info


def build_pointcloud(mask, depth, K, args, m_obj):
    """掩码 + 深度 → 点云。**调用与仿真完全相同的共享函数。**"""
    from configs.point_bridge_pointcloud import mask_depth_to_pointcloud

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    mask_t = torch.from_numpy(mask).to(dev).unsqueeze(0)              # (1,H,W)
    depth_t = torch.from_numpy(depth).float().to(dev).unsqueeze(0)    # (1,H,W)
    K_t = torch.from_numpy(K).float().to(dev).unsqueeze(0)            # (1,3,3)

    if args.frame == "camera":
        # 相机系：位姿为单位变换，函数内的"目标系"就是相机光学系本身。
        cam_pos = torch.zeros(1, 3, device=dev)
        cam_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]], device=dev)
        # 工作空间裁剪的边界是**基座系**下定义的，相机系里没有意义，必须关掉，
        # 否则会把本来正确的点全裁掉（这正是之前测试踩过的坑）。
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


def render(rgb, depth, mask, pts, counts, info, args, m_obj):
    """四栏图：RGB+掩码 / 深度+掩码轮廓 / 三维点云 / 数值诊断。"""
    import matplotlib
    if args.save:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=(16, 9))

    ax = fig.add_subplot(2, 2, 1)
    ov = rgb.copy()
    if mask is not None:
        ov[mask] = (0.45 * ov[mask] + 0.55 * np.array([255, 40, 40])).astype(np.uint8)
    ax.imshow(ov)
    ax.set_title(f"RGB + YOLO 掩码（{mask.sum() if mask is not None else 0} px）")
    ax.axis("off")

    ax = fig.add_subplot(2, 2, 2)
    dv = depth.copy()
    fin = np.isfinite(dv) & (dv > 0)
    if fin.any():
        lo, hi = np.percentile(dv[fin], [2, 98])
        dv = np.clip((dv - lo) / max(hi - lo, 1e-6), 0, 1)
    dv[~fin] = 0.0
    ax.imshow(dv, cmap="viridis")
    if mask is not None:
        # 只画轮廓不填充，方便看掩码边缘是否压在深度跳变处
        ax.contour(mask.astype(float), levels=[0.5], colors="r", linewidths=1.2)
    ax.set_title("深度（2-98 分位拉伸）+ 掩码轮廓")
    ax.axis("off")

    ax = fig.add_subplot(2, 2, 3, projection="3d")
    if pts is not None and len(pts):
        ax.scatter(pts[:, 0], pts[:, 1], pts[:, 2], c=pts[:, 2], cmap="viridis", s=18)
        # 等比例轴：不等比会让形状看起来是错的，这个图的唯一用途就是看形状
        c = pts.mean(axis=0)
        r = max(np.ptp(pts, axis=0).max() * 0.5, 0.01)
        ax.set_xlim(c[0] - r, c[0] + r)
        ax.set_ylim(c[1] - r, c[1] + r)
        ax.set_zlim(c[2] - r, c[2] + r)
    ax.set_xlabel("X (m)"); ax.set_ylabel("Y (m)"); ax.set_zlabel("Z (m)")
    ax.set_title(f"点云 {args.frame} 系（{m_obj} 点，FPS 采样）")

    ax = fig.add_subplot(2, 2, 4)
    ax.axis("off")
    lines = [f"坐标系: {args.frame}" + ("   ← 未标定外参，仅看量级"
                                        if args.frame == "base" else ""),
             f"有效点数(counts): {counts} / {m_obj}",
             f"噪声 sigma: {args.noise_std} m", ""]
    if mask is not None:
        npx = int(mask.sum())
        dvalid = int((mask & fin).sum())
        lines += [f"掩码像素: {npx}",
                  f"其中深度有效: {dvalid}"
                  + (f"  ({100.0 * dvalid / npx:.1f}%)" if npx else ""), ""]
    if pts is not None and len(pts):
        lines += ["点云范围 (m):",
                  f"  X {pts[:,0].min():+.3f} .. {pts[:,0].max():+.3f}",
                  f"  Y {pts[:,1].min():+.3f} .. {pts[:,1].max():+.3f}",
                  f"  Z {pts[:,2].min():+.3f} .. {pts[:,2].max():+.3f}",
                  f"尺寸 (m): {np.ptp(pts, axis=0).round(3).tolist()}", ""]
    lines.append(f"YOLO 工件实例: {len(info)}")
    for r in info[:5]:
        s = f"  #{r['index']} conf={r['conf']:.3f}"
        if "selection_score" in r:
            s += f" score={r['selection_score']:.3f} side={r['side']}"
        lines.append(s)
    ax.text(0.0, 1.0, "\n".join(lines), va="top", ha="left",
            family="monospace", fontsize=10, transform=ax.transAxes)

    fig.tight_layout()
    if args.save:
        fig.savefig(args.save, dpi=110)
        print(f"[存图] {args.save}")
        plt.close(fig)
    else:
        plt.show()


def main() -> int:
    args = parse_args()

    from configs.point_bridge_pointcloud import M_OBJ
    m_obj = args.m_obj if args.m_obj is not None else M_OBJ

    if _DROPPED:
        print(f"[环境] 已从 sys.path 摘掉 {len(_DROPPED)} 条 Isaac Sim 路径"
              f"（它们的 filelock 3.13.1 会让 YOLO fork 导入失败）")

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

    try:
        while True:
            rgb, depth = cam.get_frame()
            if rgb is None:
                print("[相机] 取帧失败，重试")
                continue

            # ultralytics 的 preprocess() 期望 BGR 输入，会自动翻转成 RGB。
            # RealsenseCamera.get_frame() 返回 RGB，所以传入前要转回 BGR。
            res = model.predict(rgb[..., ::-1], imgsz=args.imgsz, conf=args.conf,
                                device=args.device, verbose=False)[0]
            mask, info = pick_target(res, args.all_instances)

            if mask is None:
                print("[YOLO] 未检出工件（class 0）。调低 --conf 或检查画面里有没有工件。")
                pts, counts = None, 0
            else:
                pts, counts = build_pointcloud(mask, depth, K, args, m_obj)
                print(f"[点云] 掩码 {int(mask.sum())} px → 有效 {counts}/{m_obj} 点"
                      f"  尺寸 {np.ptp(pts, axis=0).round(3).tolist()} m")
                if counts == 0:
                    # 掩码有像素但一个点都没活下来，几乎一定是深度侧的问题
                    print("[警告] 掩码非空但有效点为 0：掩码区域深度全为 0/NaN。"
                          "常见原因是工件太近（D405 最近 ~7cm、D435 ~28cm）或反光。")

            render(rgb, depth, mask, pts, counts, info, args, m_obj)
            if not args.live:
                break
    except KeyboardInterrupt:
        print("\n[退出]")
    finally:
        cam.stop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
