#!/usr/bin/env python3
"""仿真固定相机 vs 真机 D435：并排实时对比。

这是 real2sim 对齐的核心目视工具 —— 外参数值对不对，最终要靠"两边看到的画面像
不像"来判断。数值残差再漂亮，画面对不上就是没对齐。

左=仿真（标定内外参渲染），右=真机 D435（serial 254622072913，即标定所用那台）。

看什么：
    桌面边缘、工件、机械臂在两幅画面里的**位置和大小**应大致重合。
    - 整体平移  → 外参位置偏了
    - 整体旋转  → 外参姿态偏了
    - 大小不一  → 内参 fx/fy 不匹配，或分辨率/FOV 没对齐
    - 透视不同  → 相机高度或俯仰角错了

分辨率：刻意让真机也出 640x480，与仿真的 canonical 分辨率一致。默认的 1280x720
是 16:9，与仿真的 4:3 直接并排会因 FOV 不同而误判成外参错误。

用法（需要 GUI + 已连接 D435）::

    ~/IsaacLab/isaaclab.sh -p scripts/train/compare_sim_real_camera.py
    ~/IsaacLab/isaaclab.sh -p scripts/train/compare_sim_real_camera.py --mode blend
    ~/IsaacLab/isaaclab.sh -p scripts/train/compare_sim_real_camera.py --no_real   # 无相机时只看仿真

窗口按键：
    q / ESC   退出
    s         存当前对比图
    b         切换 并排 / 混合叠加
    [ / ]     混合模式下调整仿真占比
    e         切换 边缘叠加（把真机的边缘画到仿真图上，最容易看出错位）
    空格      暂停 / 继续仿真步进
"""

import argparse
import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))
sys.path.insert(0, _PROJECT_DIR)

from isaaclab.app import AppLauncher

# 与标定所用的全局相机一致（scripts/deploy/manual_roi_annotate.py 里的 D435_SERIAL）
D435_SERIAL = "254622072913"

parser = argparse.ArgumentParser(description="仿真固定相机与真机 D435 并排实时对比")
parser.add_argument("--serial", type=str, default=D435_SERIAL, help="真机相机序列号")
parser.add_argument(
    "--no_real",
    action="store_true",
    help="不连真机，只显示仿真画面（没插相机时用来确认仿真侧正常）",
)
parser.add_argument(
    "--mode",
    type=str,
    default="side",
    choices=("side", "blend"),
    help="side=并排；blend=半透明叠加（错位时会看到重影，比并排更灵敏）",
)
parser.add_argument("--alpha", type=float, default=0.5, help="blend 模式下仿真的占比")
parser.add_argument("--scale", type=float, default=1.0, help="窗口整体缩放")
parser.add_argument(
    "--motion",
    type=str,
    default="zero",
    choices=("zero", "random"),
    help="仿真里机械臂是否运动。对比静态场景时用 zero。",
)
parser.add_argument(
    "--out",
    type=str,
    default=os.path.join(_PROJECT_DIR, "logs", "camera_preview"),
    help="按 s 存图的目录",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True
args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv

from configs.xarm7_pick_pointcloud_env_cfg import (
    CANONICAL_HEIGHT,
    CANONICAL_WIDTH,
    CALIB_CAMERA_POSITION_IN_BASE_M,
    FIXED_CAMERA_POSITION_M,
    XArm7PickPointCloudPlayEnvCfg,
)

_WINDOW = "SIM (left) vs REAL D435 (right)"


def _to_uint8_rgb(rgb: torch.Tensor) -> np.ndarray:
    """相机 rgb 输出 → (H, W, 3) uint8，兼容 uint8/float 与带 alpha 的情况。"""
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


def _label(img: np.ndarray, text: str, color=(255, 255, 255)) -> np.ndarray:
    img = img.copy()
    cv2.rectangle(img, (0, 0), (img.shape[1], 24), (0, 0, 0), -1)
    cv2.putText(img, text, (6, 17), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return img


def _open_real_camera():
    """连真机 D435。失败时返回 None 并说明原因 —— 不让仿真侧跟着挂掉。"""
    if args_cli.no_real:
        return None
    sys.path.insert(0, os.path.join(_PROJECT_DIR, "scripts", "camera"))
    try:
        from realsense_camera import RealsenseCamera
    except Exception as e:
        print(f"[真机] 导入 RealsenseCamera 失败: {e}")
        return None
    try:
        # 640x480 与仿真 canonical 分辨率一致。若用默认的 1280x720(16:9)，
        # 与仿真的 4:3 并排会因 FOV 不同被误读成外参错误。
        cam = RealsenseCamera(
            width=CANONICAL_WIDTH,
            height=CANONICAL_HEIGHT,
            fps=30,
            align_on_device=True,
            serial_number=args_cli.serial,
        )
        cam.start()
        return cam
    except Exception as e:
        print(f"[真机] 启动相机失败: {e}")
        print("       检查：相机是否插好、是否被其它进程占用、序列号是否正确。")
        return None


def main() -> None:
    cfg = XArm7PickPointCloudPlayEnvCfg()
    cfg.scene.num_envs = 1
    # 不调 set_observation_stage：stage 1/-1 会摘掉 camera_fixed，而它正是对比对象
    env = ManagerBasedRLEnv(cfg=cfg)
    camera = env.scene["camera_fixed"]

    real_cam = _open_real_camera()
    os.makedirs(args_cli.out, exist_ok=True)

    print("=" * 78)
    print("仿真 vs 真机 对比")
    print(f"  仿真相机 (env系)  : {FIXED_CAMERA_POSITION_M}")
    print(f"  标定值   (基座系) : {CALIB_CAMERA_POSITION_IN_BASE_M}")
    print(f"  真机序列号        : {args_cli.serial}"
          f"{'  [未连接，仅显示仿真]' if real_cam is None else ''}")
    print(f"  分辨率            : {CANONICAL_WIDTH}x{CANONICAL_HEIGHT} (两侧一致)")
    print("-" * 78)
    print("按键: q=退出  s=存图  b=并排/叠加  e=边缘叠加  [ ]=调透明度  空格=暂停")
    print("=" * 78)

    env.reset()
    action_dim = env.action_manager.total_action_dim
    zero_action = torch.zeros((env.num_envs, action_dim), device=env.device)

    cv2.namedWindow(_WINDOW, cv2.WINDOW_NORMAL)

    mode = args_cli.mode
    alpha = float(np.clip(args_cli.alpha, 0.0, 1.0))
    edge_overlay = False
    paused = False
    saved_n = 0
    last_real = None

    while simulation_app.is_running():
        if not paused:
            if args_cli.motion == "random":
                action = torch.randn((env.num_envs, action_dim), device=env.device) * 0.3
            else:
                action = zero_action
            _, _, terminated, truncated, _ = env.step(action)
            if bool(terminated[0]) or bool(truncated[0]):
                env.reset()

        sim_rgb = camera.data.output.get("rgb")
        if sim_rgb is None:
            raise RuntimeError(
                "仿真相机没有 rgb 输出。检查 make_pointcloud_camera_cfg 的 data_types。"
            )
        sim_bgr = cv2.cvtColor(_to_uint8_rgb(sim_rgb), cv2.COLOR_RGB2BGR)

        # 真机取帧。取不到就沿用上一帧，避免画面闪烁掩盖对比效果。
        if real_cam is not None:
            try:
                rgb, _ = real_cam.get_frame()
                if rgb is not None:
                    last_real = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            except Exception as e:
                print(f"[真机] 取帧失败: {e}")

        if last_real is not None:
            real_bgr = last_real
            if real_bgr.shape[:2] != sim_bgr.shape[:2]:
                # 分辨率不一致时缩放到仿真尺寸。注意这**不能**修正 FOV 差异 ——
                # 若真机没能按 640x480(4:3) 出流，这里的缩放会拉伸画面，对比结果
                # 只能看大致构图，不能用来判断外参。
                real_bgr = cv2.resize(real_bgr, (sim_bgr.shape[1], sim_bgr.shape[0]))
        else:
            real_bgr = np.zeros_like(sim_bgr)
            cv2.putText(real_bgr, "NO REAL CAMERA", (40, sim_bgr.shape[0] // 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2, cv2.LINE_AA)

        if edge_overlay and last_real is not None:
            # 真机边缘画到仿真图上：错位时边缘会明显偏离仿真里的对应结构，
            # 比半透明叠加更容易定位是平移还是旋转。
            gray = cv2.cvtColor(real_bgr, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 60, 160)
            canvas = sim_bgr.copy()
            canvas[edges > 0] = (0, 255, 0)
            canvas = _label(canvas, "EDGE OVERLAY: green = real-camera edges on sim image")
        elif mode == "blend" and last_real is not None:
            blended = cv2.addWeighted(sim_bgr, alpha, real_bgr, 1.0 - alpha, 0.0)
            canvas = _label(blended, f"BLEND  sim={alpha:.2f} / real={1-alpha:.2f}"
                                     "   ([ / ] to adjust)")
        else:
            left = _label(sim_bgr, "SIM (calibrated extrinsics)", (120, 220, 255))
            right = _label(real_bgr, f"REAL D435 {args_cli.serial}", (120, 255, 160))
            # 中间加一条分隔线，避免两幅画面边界处被误读成图像内容
            sep = np.full((sim_bgr.shape[0], 2, 3), 60, np.uint8)
            canvas = np.hstack([left, sep, right])

        if paused:
            canvas = _label(canvas, "[PAUSED]  press space to resume", (0, 220, 255))

        if abs(args_cli.scale - 1.0) > 1e-3:
            canvas = cv2.resize(canvas, None, fx=args_cli.scale, fy=args_cli.scale,
                                interpolation=cv2.INTER_AREA)
        cv2.imshow(_WINDOW, canvas)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), 27):
            break
        elif key == ord(" "):
            paused = not paused
        elif key == ord("b"):
            mode = "blend" if mode == "side" else "side"
        elif key == ord("e"):
            edge_overlay = not edge_overlay
        elif key == ord("["):
            alpha = max(0.0, alpha - 0.05)
        elif key == ord("]"):
            alpha = min(1.0, alpha + 0.05)
        elif key == ord("s"):
            path = os.path.join(args_cli.out, f"sim_vs_real_{saved_n:03d}.png")
            cv2.imwrite(path, canvas)
            saved_n += 1
            print(f"已保存: {path}")

        if cv2.getWindowProperty(_WINDOW, cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()
    if real_cam is not None:
        try:
            real_cam.stop()
        except Exception:
            pass
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
