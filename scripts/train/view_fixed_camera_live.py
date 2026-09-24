#!/usr/bin/env python3
"""实时 GUI 窗口：显示标定固定相机拍到的画面。

与同目录的 ``dump_fixed_camera_rgb.py`` 是一对：那个落盘单帧、可在无显示器的机器
上跑；这个开窗口连续刷新，适合一边动机械臂一边看视角。

用途和 dump 脚本一样 —— ``xarm7_pick_pointcloud_env_cfg`` 的外参来自
``forced_solution: true`` 的手眼标定（``EXTRINSICS_CONFIRMED`` 至今为 False），
点云管线不会因为视角偏了而报错，只能靠肉眼确认相机到底在拍哪儿。

用法（必须用 Isaac Lab 的 python，且需要有显示器 / X11）::

    ~/IsaacLab/isaaclab.sh -p scripts/train/view_fixed_camera_live.py
    ~/IsaacLab/isaaclab.sh -p scripts/train/view_fixed_camera_live.py --view rgb+depth+mask
    ~/IsaacLab/isaaclab.sh -p scripts/train/view_fixed_camera_live.py --motion random

窗口内按键：
    q / ESC   退出
    s         把当前帧存到 --out 目录
    空格      暂停 / 继续仿真步进（画面仍然刷新）
"""

import argparse
import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))
sys.path.insert(0, _PROJECT_DIR)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="实时显示 xarm7 点云环境标定固定相机的画面")
parser.add_argument(
    "--view",
    type=str,
    default="rgb",
    help=(
        "显示哪些通道，用 + 连接：rgb / depth / mask。例如 'rgb+depth'、"
        "'rgb+depth+mask'。多个通道横向拼接在同一个窗口里。"
    ),
)
parser.add_argument(
    "--motion",
    type=str,
    default="zero",
    choices=("zero", "random"),
    help=(
        "机械臂怎么动。zero=保持初始位姿不动（看静态视角用）；"
        "random=小幅随机动作，用来确认机械臂进画面时会不会挡住工件。"
    ),
)
parser.add_argument(
    "--scale",
    type=float,
    default=1.0,
    help="窗口显示缩放。640x480 在高分屏上偏小，可给 1.5。不影响相机实际分辨率。",
)
parser.add_argument(
    "--out",
    type=str,
    default=os.path.join(_PROJECT_DIR, "logs", "camera_preview"),
    help="按 s 存图时的输出目录",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 相机是本脚本的全部目的；程序化桌面的 PreviewSurface(MDL) 材质也只在渲染型 Kit
# experience 下注册 —— 两个理由都要求 enable_cameras。
args_cli.enable_cameras = True
args_cli.num_envs = 1
# 这里刻意**不**强制 headless：用户可能想同时开 Isaac 的 3D 视口对照着看。
# 我们自己的 cv2 窗口与 headless 无关，headless 只影响 Isaac 自带视口。

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv

from configs.xarm7_pick_pointcloud_env_cfg import (
    FIXED_CAMERA_CONVENTION,
    FIXED_CAMERA_POSITION_M,
    FIXED_CAMERA_QUATERNION_WXYZ,
    OBJECT_PRIM_PATTERN,
    XArm7PickPointCloudPlayEnvCfg,
)

_WINDOW = "xArm7 fixed camera (calibrated)"


def _to_uint8_rgb(rgb: torch.Tensor) -> np.ndarray:
    """相机 rgb 输出 → (H, W, 3) uint8。

    Isaac Lab 的 tiled camera 视版本可能给 uint8 [0,255] 或 float [0,1]，也可能带
    alpha 通道。统一收口，免得看到全黑画面却以为是外参错了。
    """
    arr = rgb.detach().cpu().numpy()
    if arr.ndim == 4:
        arr = arr[0]
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        hi = float(np.nanmax(arr)) if arr.size else 0.0
        # >1.5 说明本来就是 0~255 的 float，不能再乘 255
        scale = 1.0 if hi > 1.5 else 255.0
        arr = np.clip(np.nan_to_num(arr) * scale, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _depth_to_bgr(depth_raw: torch.Tensor) -> tuple[np.ndarray, str]:
    """深度 → 伪彩 BGR 图 + 一行统计文本。"""
    d = depth_raw[0, ..., 0] if depth_raw.dim() == 4 else depth_raw[0]
    d = d.detach().cpu().numpy()
    # depth_clipping_behavior="zero"：0 代表"没打到东西"，必须排除在归一化之外，
    # 否则整张图被拉到近乎全白。
    valid = np.isfinite(d) & (d > 0)
    vis = np.zeros_like(d, dtype=np.float32)
    if valid.any():
        lo, hi = d[valid].min(), d[valid].max()
        vis[valid] = (d[valid] - lo) / max(hi - lo, 1e-6)
        text = f"depth {lo:.2f}~{hi:.2f}m  valid={int(valid.sum())}"
    else:
        text = "depth: NO VALID PIXELS"
    bgr = cv2.applyColorMap((vis * 255).astype(np.uint8), cv2.COLORMAP_TURBO)
    # 无效区域强制涂黑，避免 colormap 把 0 染成鲜艳的颜色误导判断
    bgr[~valid] = 0
    return bgr, text


def _object_ids(id_to_labels: dict) -> list[int]:
    """从 idToLabels 里挑出属于工件的实例 ID。"""
    target = OBJECT_PRIM_PATTERN.format(env=0)
    return [
        int(k)
        for k, v in id_to_labels.items()
        if target in (v if isinstance(v, str) else str(v))
    ]


def _label(img: np.ndarray, text: str) -> np.ndarray:
    """在图片左上角压一行字，带黑底避免白背景上看不清。"""
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 22), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1,
                cv2.LINE_AA)
    return img


def main() -> None:
    views = [v.strip() for v in args_cli.view.split("+") if v.strip()]
    unknown = [v for v in views if v not in ("rgb", "depth", "mask")]
    if unknown:
        raise ValueError(f"--view 只支持 rgb/depth/mask，收到未知项: {unknown}")
    if not views:
        raise ValueError("--view 不能为空")

    cfg = XArm7PickPointCloudPlayEnvCfg()
    cfg.scene.num_envs = 1
    # 注意：不调用 set_observation_stage()。stage 1/-1 会把 camera_fixed 摘掉，
    # 而这台相机正是本脚本的全部目的；默认 cfg 走的就是完整相机管线。
    env = ManagerBasedRLEnv(cfg=cfg)

    os.makedirs(args_cli.out, exist_ok=True)

    print("=" * 78)
    print("固定相机外参（相对机器人基座系）")
    print(f"  位置 (m)      : {FIXED_CAMERA_POSITION_M}")
    print(f"  四元数 (wxyz) : {FIXED_CAMERA_QUATERNION_WXYZ}")
    print(f"  约定          : {FIXED_CAMERA_CONVENTION}")
    print("-" * 78)
    print("窗口按键：q/ESC=退出   s=存当前帧   空格=暂停/继续")
    print("=" * 78)

    env.reset()
    camera = env.scene["camera_fixed"]
    action_dim = env.action_manager.total_action_dim

    cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)

    paused = False
    step = 0
    saved_n = 0
    printed_pose = False

    while simulation_app.is_running():
        if not paused:
            if args_cli.motion == "random":
                # 小幅动作：动作空间已在 camfix 里限幅（±0.6°/step），这里再乘 0.3
                # 只是让画面变化平缓些，便于观察遮挡关系。
                action = torch.randn((env.num_envs, action_dim), device=env.device) * 0.3
            else:
                action = torch.zeros((env.num_envs, action_dim), device=env.device)
            _, _, terminated, truncated, _ = env.step(action)
            step += 1
            if bool(terminated[0]) or bool(truncated[0]):
                env.reset()
                step = 0

        out = camera.data.output

        if not printed_pose:
            # 相机在世界系的实际位姿。update_latest_camera_pose=True 时这是渲染器
            # 真正用的值，与上面的标称 offset 对不上就说明有别的东西写过位姿。
            print(f"[实际] camera pos_w      : {camera.data.pos_w[0].cpu().numpy()}")
            print(f"[实际] camera quat_w_ros : {camera.data.quat_w_ros[0].cpu().numpy()}")
            printed_pose = True

        panels = []

        if "rgb" in views:
            rgb = out.get("rgb")
            if rgb is None:
                raise RuntimeError(
                    "相机没有 rgb 输出。检查 make_pointcloud_camera_cfg 的 data_types 是否仍含 'rgb'。"
                )
            bgr = cv2.cvtColor(_to_uint8_rgb(rgb), cv2.COLOR_RGB2BGR)
            panels.append(_label(bgr, f"RGB  step={step}{'  [PAUSED]' if paused else ''}"))

        if "depth" in views:
            depth_raw = out.get("distance_to_image_plane")
            if depth_raw is None:
                panels.append(_label(np.zeros((480, 640, 3), np.uint8), "depth: N/A"))
            else:
                bgr, text = _depth_to_bgr(depth_raw)
                panels.append(_label(bgr, text))

        if "mask" in views:
            seg_raw = out.get("instance_id_segmentation_fast")
            info = camera.data.info.get("instance_id_segmentation_fast")
            id_to_labels = info.get("idToLabels") if isinstance(info, dict) else None
            if seg_raw is None or not id_to_labels:
                panels.append(_label(np.zeros((480, 640, 3), np.uint8), "mask: N/A"))
            else:
                seg = seg_raw[0, ..., 0] if seg_raw.dim() == 4 else seg_raw[0]
                seg = seg.detach().cpu().numpy()
                ids = _object_ids(id_to_labels)
                mask = np.isin(seg, ids) if ids else np.zeros_like(seg, bool)
                bgr = np.zeros((*mask.shape, 3), np.uint8)
                bgr[mask] = (0, 255, 0)
                # 掩码像素数是规格 §2.2 的验收项，直接标在画面上省得回头翻日志
                panels.append(_label(bgr, f"object mask  px={int(mask.sum())} (need 200~500+)"))

        # 各通道分辨率一致（同一台相机），直接横向拼
        canvas = panels[0] if len(panels) == 1 else np.hstack(panels)
        if abs(args_cli.scale - 1.0) > 1e-3:
            canvas = cv2.resize(
                canvas, None, fx=args_cli.scale, fy=args_cli.scale,
                interpolation=cv2.INTER_NEAREST,
            )
        cv2.imshow(_WINDOW, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        if key == ord(" "):
            paused = not paused
        if key == ord("s"):
            path = os.path.join(args_cli.out, f"live_{saved_n:03d}.png")
            cv2.imwrite(path, canvas)
            saved_n += 1
            print(f"已保存: {path}")

        # 用户点窗口右上角叉号关掉时也要退出，否则会留个没有窗口的死循环
        if cv2.getWindowProperty(_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close()
