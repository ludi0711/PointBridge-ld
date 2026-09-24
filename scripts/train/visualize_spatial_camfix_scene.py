#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""xArm7 spatial-camfix 环境可视化脚本（平移到 xy=(0,0) 后的布局确认）。

参考 train_pick_vision_spatial_camfix.py 的 --play 模式，但不加载策略、不画奖励曲线，
只做纯场景可视化：

  - 单环境构建 camfix Play 环境，在 Isaac Sim 视口里渲染新布局
  - 用零动作（机械臂保持初始关节角）持续 step，方便观察
  - 在 link7、工件、固定相机位置放彩色 marker 球
  - 一个 matplotlib 窗口实时显示 固定相机 / 腕部相机 的 RGB
  - 启动时打印各资产世界坐标，确认机械臂基座已落在 xy≈(0,0)

用法：
    /home/gxai/IsaacLab/isaaclab.sh -p scripts/train/visualize_spatial_camfix_scene.py

按 Ctrl-C 退出。
"""

import argparse
import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))
sys.path.insert(0, _PROJECT_DIR)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="xArm7 spatial-camfix 环境可视化（平移到 xy=(0,0) 布局确认）"
)
parser.add_argument("--seed", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 环境观测依赖 camera_fixed / camera_wrist 的 Theia 特征，必须启用相机管线。
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

from configs.xarm7_pick_vision_spatial_camfix_env_cfg import (
    XArm7PickLiftCubePlayEnvCfg,
    _augment_image,
)


def _augment_rgb_for_display(rgb, key):
    """对单帧相机 RGB 套用与观测一致的 _augment_image，返回 uint8 (H,W,3) 供显示。

    传相机名作 key → 读同一轮缓存的增强参数，显示的模糊/噪声与训练观测完全一致
    （同一 episode 内固定，reset 才变）。输入 rgb: (H,W,>=3)，uint8 或 float。
    """
    x = rgb[..., :3]
    if x.dtype == torch.uint8:
        x = x.float() / 255.0
    else:
        x = x.float()
    x = x.permute(2, 0, 1).unsqueeze(0)        # (1,3,H,W)
    x = _augment_image(x, key=key)              # 高斯模糊 + 噪声（读当前轮缓存参数）
    x = x.squeeze(0).permute(1, 2, 0)           # (H,W,3)
    return (x.clamp(0.0, 1.0) * 255.0).to(torch.uint8).cpu().numpy()


def _make_sphere_marker(stage, prim_path: str, color: tuple, radius: float = 0.012):
    """在场景里放一个自发光小球 marker，返回其 translate op。"""
    from pxr import UsdGeom, UsdShade, Gf, Sdf
    sphere = UsdGeom.Sphere.Define(stage, prim_path)
    sphere.GetRadiusAttr().Set(radius)
    mat = UsdShade.Material.Define(stage, prim_path + "/mat")
    shader = UsdShade.Shader.Define(stage, prim_path + "/mat/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(sphere).Bind(mat)
    xf = UsdGeom.Xformable(sphere)
    op = xf.AddTranslateOp()
    return op


def _print_layout(base_env):
    """打印各资产世界坐标，确认平移生效（机械臂基座应≈ xy=(0,0)）。"""
    scene = base_env.scene
    robot = scene["robot"]
    table = scene["workpiece_table"]
    obj   = scene["object"]

    def _xyz(t):
        return f"({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f})"

    print("\n========== 场景布局（env_0 世界坐标）==========")
    print(f"  机械臂基座 Robot       : {_xyz(robot.data.root_pos_w[0].cpu().numpy())}")
    print(f"  工件桌  WorkpieceTable : {_xyz(table.data.root_pos_w[0].cpu().numpy())}")
    print(f"  工件    Object         : {_xyz(obj.data.root_pos_w[0].cpu().numpy())}")

    import omni.usd
    from pxr import UsdGeom, Usd
    stage = omni.usd.get_context().get_stage()
    cam_prim = stage.GetPrimAtPath("/World/envs/env_0/CameraFixed")
    if cam_prim.IsValid():
        t = UsdGeom.Xformable(cam_prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        ).ExtractTranslation()
        print(f"  固定相机 CameraFixed   : ({t[0]:+.3f}, {t[1]:+.3f}, {t[2]:+.3f})")
    print("===============================================\n")


def main():
    # ── 1. 单环境 + 关闭会终止 episode 的 termination（持续观察，不自动 reset）──
    env_cfg = XArm7PickLiftCubePlayEnvCfg()
    env_cfg.scene.num_envs = 1
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    env_cfg.terminations.collision = None
    env_cfg.terminations.time_out = None
    env_cfg.terminations.reach_success = None

    # 纯可视化不需要策略，跳过 Theia 视觉编码器（避免缺模型时崩在观测管理器初始化）。
    # 仅删除这两个观测项，相机本身仍正常 spawn / 渲染，双相机 RGB 照常显示。
    env_cfg.observations.policy.theia_feat_fixed = None
    env_cfg.observations.policy.theia_feat_wrist = None

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    base_env = env.unwrapped
    device = env.device

    obj      = base_env.scene["object"]
    ee_frame = base_env.scene["ee_frame"]

    # ── 2. marker 球（link7 末端 / 工件 / 固定相机）──────────────────────────
    import omni.usd
    from pxr import Gf, UsdGeom, Usd
    stage = omni.usd.get_context().get_stage()
    ee_marker_op        = _make_sphere_marker(stage, "/World/Markers/ee_offset",  color=(1.0, 0.0, 0.0))
    obj_marker_op       = _make_sphere_marker(stage, "/World/Markers/obj_target", color=(0.0, 0.3, 1.0), radius=0.03)
    cam_fixed_marker_op = _make_sphere_marker(stage, "/World/Markers/cam_fixed",  color=(1.0, 0.8, 0.0), radius=0.02)

    # ── 3. 双相机 RGB 显示窗口（上排原图，下排增强后）────────────────────────
    import matplotlib.pyplot as plt
    fig_cam, axes = plt.subplots(2, 2, figsize=(10, 9))
    (ax_fixed, ax_wrist), (ax_fixed_aug, ax_wrist_aug) = axes
    ax_fixed.axis("off");     ax_fixed.set_title("Fixed Camera (RGB raw)", fontsize=10)
    ax_wrist.axis("off");     ax_wrist.set_title("Wrist Camera (RGB raw)", fontsize=10)
    ax_fixed_aug.axis("off"); ax_fixed_aug.set_title("Fixed Camera (blur+noise)", fontsize=10)
    ax_wrist_aug.axis("off"); ax_wrist_aug.set_title("Wrist Camera (blur+noise)", fontsize=10)
    _dummy = np.zeros((224, 224, 3), dtype=np.uint8)
    fixed_handle     = ax_fixed.imshow(_dummy)
    wrist_handle     = ax_wrist.imshow(_dummy)
    fixed_aug_handle = ax_fixed_aug.imshow(_dummy)
    wrist_aug_handle = ax_wrist_aug.imshow(_dummy)
    fig_cam.suptitle("xArm7 spatial-camfix 场景可视化（xy=(0,0) 布局；下排=观测增强）", fontsize=11)
    fig_cam.tight_layout()
    plt.ion(); plt.pause(0.1)

    # ── 4. reset，打印布局，进入观察循环 ─────────────────────────────────────
    obs, _ = env.reset()
    cam_fixed_prim = stage.GetPrimAtPath("/World/envs/env_0/CameraFixed")
    _print_layout(base_env)

    # 零动作：机械臂在动作链路里保持当前关节角（KinematicRelativeJointDirectAction）。
    num_actions = getattr(base_env.action_manager, "total_action_dim", None) or 7
    zero_action = torch.zeros((1, num_actions), device=device)

    import time
    RESET_PERIOD_S = 2.0   # 每 2 秒自动 reset 一次，轮换光照/桌面颜色/相机等随机化
    last_reset_t = time.time()

    print("[Vis] 场景运行中。在 Isaac Sim 视口观察布局；matplotlib 窗口显示双相机 RGB。")
    print(f"[Vis] 每 {RESET_PERIOD_S:.0f} 秒自动 reset 一次（重采样光照随机化）。按 Ctrl-C 退出。")

    g_step = 0
    try:
        while simulation_app.is_running():
            # 定时 reset：触发所有 mode="reset" 的 EventTerm（含光照随机化）。
            if time.time() - last_reset_t >= RESET_PERIOD_S:
                obs, _ = env.reset()
                last_reset_t = time.time()

            obs, _, dones, _ = env.step(zero_action)

            ee_pos  = ee_frame.data.target_pos_w[0, 0].cpu().numpy()
            obj_pos = obj.data.root_pos_w[0].cpu().numpy()
            ee_marker_op.Set(Gf.Vec3d(float(ee_pos[0]), float(ee_pos[1]), float(ee_pos[2])))
            obj_marker_op.Set(Gf.Vec3d(float(obj_pos[0]), float(obj_pos[1]), float(obj_pos[2])))

            if cam_fixed_prim.IsValid():
                t = UsdGeom.Xformable(cam_fixed_prim).ComputeLocalToWorldTransform(
                    Usd.TimeCode.Default()
                ).ExtractTranslation()
                cam_fixed_marker_op.Set(Gf.Vec3d(t[0], t[1], t[2]))

            # 每 3 步刷新一次相机画面，省开销。
            if g_step % 3 == 0:
                rgb_fixed = base_env.scene["camera_fixed"].data.output["rgb"]
                if rgb_fixed is not None and rgb_fixed.shape[0] > 0:
                    fixed_handle.set_data(rgb_fixed[0, ..., :3].cpu().numpy().astype(np.uint8))
                    fixed_aug_handle.set_data(_augment_rgb_for_display(rgb_fixed[0], "camera_fixed"))
                rgb_wrist = base_env.scene["camera_wrist"].data.output["rgb"]
                if rgb_wrist is not None and rgb_wrist.shape[0] > 0:
                    wrist_handle.set_data(rgb_wrist[0, ..., :3].cpu().numpy().astype(np.uint8))
                    wrist_aug_handle.set_data(_augment_rgb_for_display(rgb_wrist[0], "camera_wrist"))
                fig_cam.canvas.draw_idle()
                plt.pause(0.001)

            g_step += 1

    except KeyboardInterrupt:
        print("\n[Vis] 用户中断。")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nError: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
