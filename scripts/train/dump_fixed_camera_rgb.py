#!/usr/bin/env python3
"""导出标定固定相机的 RGB 视角图，用于人眼确认外参对不对。

用途：``xarm7_pick_pointcloud_env_cfg`` 里的外参来自 ``forced_solution: true``
的手眼标定（旋转激励 26.0° < 要求的 45°，``EXTRINSICS_CONFIRMED`` 至今为 False）。
点云管线本身不会因为视角偏了而报错 —— 掩码非空就照样出点。所以在跑阶段 1/2 之前
先用肉眼看一眼这个相机到底在拍哪儿，是最省时间的一次核对。

看什么：
  - 工件（Object）在不在画面里、离画面中心远不远
  - 桌面占据画面的比例是否与 §标定几何 的"1.264 m 宽覆盖"一致
  - 机械臂有没有整个糊在镜头上、或者干脆拍到天上（ROS/world 约定混用的典型症状）

用法（必须用 Isaac Lab 的 python）::

    ./isaaclab.sh -p scripts/train/dump_fixed_camera_rgb.py
    ./isaaclab.sh -p scripts/train/dump_fixed_camera_rgb.py --settle_steps 30 --save_depth

默认 headless，不弹窗口，只落盘 PNG。
"""

import argparse
import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))
sys.path.insert(0, _PROJECT_DIR)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="把 xarm7 点云环境里那台标定固定相机的 RGB 存成图片"
)
parser.add_argument(
    "--out",
    type=str,
    default=os.path.join(_PROJECT_DIR, "logs", "camera_preview"),
    help="输出目录，默认 logs/camera_preview/",
)
parser.add_argument(
    "--settle_steps",
    type=int,
    default=12,
    help=(
        "落盘前先空跑多少个控制步。首帧渲染器的光照/材质常常还没收敛（画面偏暗或"
        "缺纹理），空跑几步再抓图更能代表训练时策略真正看到的画面。"
    ),
)
parser.add_argument(
    "--save_depth",
    action="store_true",
    help="额外存一张深度伪彩图。深度是点云的真正来源，RGB 只是给人看的旁证。",
)
parser.add_argument(
    "--save_mask",
    action="store_true",
    help="额外存一张工件实例掩码图，并打印掩码像素数（规格 §2.2 门槛 200~500）。",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 相机是本脚本的全部目的；且程序化桌面用 PreviewSurface(MDL) 材质，只在渲染型 Kit
# experience 下注册 —— 两个理由都要求 enable_cameras。
args_cli.enable_cameras = True
# 单环境即可，多开只是浪费显存
args_cli.num_envs = 1
# 不需要交互窗口，落盘就行；用户显式传了 --headless 也不冲突
if not getattr(args_cli, "headless", False):
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

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


def _to_uint8_rgb(rgb: torch.Tensor) -> np.ndarray:
    """相机 rgb 输出 → (H, W, 3) uint8。

    Isaac Lab 的 tiled camera 视版本可能给 uint8 [0,255] 或 float [0,1]，也可能带
    第 4 个 alpha 通道，这里统一收口，避免存出一张全黑图还以为是外参错了。
    """
    arr = rgb.detach().cpu().numpy()
    if arr.ndim == 4:  # (B, H, W, C) → 取第 0 个环境
        arr = arr[0]
    if arr.shape[-1] == 4:  # 丢掉 alpha
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        # float 路径：>1.0 说明本来就是 0~255 的 float，不要再乘 255
        hi = float(np.nanmax(arr)) if arr.size else 0.0
        scale = 1.0 if hi > 1.5 else 255.0
        arr = np.clip(np.nan_to_num(arr) * scale, 0, 255).astype(np.uint8)
    return arr


def main() -> None:
    cfg = XArm7PickPointCloudPlayEnvCfg()
    cfg.scene.num_envs = 1

    # 注意：不调用 set_observation_stage()。stage 1/-1 会把 camera_fixed 摘掉，
    # 而本脚本的全部目的就是那台相机；默认 cfg 走的正是完整相机管线。
    env = ManagerBasedRLEnv(cfg=cfg)

    os.makedirs(args_cli.out, exist_ok=True)

    print("=" * 78)
    print("固定相机外参（相对机器人基座系）")
    print(f"  位置 (m)      : {FIXED_CAMERA_POSITION_M}")
    print(f"  四元数 (wxyz) : {FIXED_CAMERA_QUATERNION_WXYZ}")
    print(f"  约定          : {FIXED_CAMERA_CONVENTION}")
    print("=" * 78)

    env.reset()

    camera = env.scene["camera_fixed"]
    action_dim = env.action_manager.total_action_dim
    # 零动作：让机械臂保持初始位姿不动，抓到的画面才是可复现的
    zero_action = torch.zeros((env.num_envs, action_dim), device=env.device)

    for _ in range(max(args_cli.settle_steps, 1)):
        env.step(zero_action)

    # 相机在世界系的实际位姿。update_latest_camera_pose=True 时这是渲染器用的真值，
    # 与上面打印的标称 offset 对不上就说明有别的东西写过位姿。
    print(f"[实际] camera pos_w      : {camera.data.pos_w[0].cpu().numpy()}")
    print(f"[实际] camera quat_w_ros : {camera.data.quat_w_ros[0].cpu().numpy()}")

    out = camera.data.output
    saved = []

    rgb = out.get("rgb")
    if rgb is None:
        raise RuntimeError(
            "相机没有输出 rgb。检查 make_pointcloud_camera_cfg 的 data_types 是否仍含 'rgb'。"
        )

    import imageio.v2 as imageio

    rgb_np = _to_uint8_rgb(rgb)
    rgb_path = os.path.join(args_cli.out, "fixed_camera_rgb.png")
    imageio.imwrite(rgb_path, rgb_np)
    saved.append(rgb_path)
    print(f"[RGB]  {rgb_np.shape[1]}x{rgb_np.shape[0]}  非黑像素占比="
          f"{float((rgb_np.max(axis=-1) > 8).mean()):.3f}")

    if args_cli.save_depth:
        depth_raw = out.get("distance_to_image_plane")
        if depth_raw is None:
            print("[警告] 相机没有 distance_to_image_plane 输出，跳过深度图。")
        else:
            d = depth_raw[0, ..., 0] if depth_raw.dim() == 4 else depth_raw[0]
            d = d.detach().cpu().numpy()
            # depth_clipping_behavior="zero"：0 表示"没打到东西"，不能参与归一化，
            # 否则整张图会被拉到近乎全白。
            valid = np.isfinite(d) & (d > 0)
            vis = np.zeros_like(d, dtype=np.float32)
            if valid.any():
                lo, hi = d[valid].min(), d[valid].max()
                vis[valid] = (d[valid] - lo) / max(hi - lo, 1e-6)
                print(f"[深度] 有效像素={int(valid.sum())}  范围={lo:.3f}~{hi:.3f} m")
            else:
                print("[警告] 深度图没有任何有效像素 —— 相机很可能在拍虚空。")
            depth_path = os.path.join(args_cli.out, "fixed_camera_depth.png")
            imageio.imwrite(depth_path, (vis * 255).astype(np.uint8))
            saved.append(depth_path)

    if args_cli.save_mask:
        seg_raw = out.get("instance_id_segmentation_fast")
        info = camera.data.info.get("instance_id_segmentation_fast")
        id_to_labels = info.get("idToLabels") if isinstance(info, dict) else None
        if seg_raw is None or not id_to_labels:
            print("[警告] 没有实例分割输出，跳过掩码图。")
        else:
            seg = seg_raw[0, ..., 0] if seg_raw.dim() == 4 else seg_raw[0]
            seg = seg.detach().cpu().numpy()
            target = OBJECT_PRIM_PATTERN.format(env=0)
            # idToLabels 的 value 可能是 str，也可能是 {"class": path} 形式的 dict
            obj_ids = [
                int(k)
                for k, v in id_to_labels.items()
                if target in (v if isinstance(v, str) else str(v))
            ]
            if not obj_ids:
                print(f"[警告] 没有实例 ID 匹配 {target!r}，掩码为空。")
                print(f"       可用标签样例: {sorted(set(map(str, id_to_labels.values())))[:6]}")
            mask = np.isin(seg, obj_ids)
            print(f"[掩码] 工件像素数={int(mask.sum())}  (规格 §2.2 门槛 200~500)")
            mask_path = os.path.join(args_cli.out, "fixed_camera_object_mask.png")
            imageio.imwrite(mask_path, (mask * 255).astype(np.uint8))
            saved.append(mask_path)

    print("-" * 78)
    for p in saved:
        print(f"已保存: {p}")

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
