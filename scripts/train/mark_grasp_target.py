#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""在 Isaac Sim 视口里标出**训练用的抓取目标点**，以及工件根原点作为参考。

2026-08-17 起训练目标点已改为**工件根原点正上方 3 cm**（训练侧开关
``--grasp_offset_cm``，默认 3.0）。本脚本的绿球就是那个点，与奖励
``object_ee_distance`` / success ``ee_reached_object`` 共用同一个
``custom_mdp.grasp_target_pos_w``，不在这里重算公式。

为什么要标：``root_pos_w`` 的"根原点"是建模时 pivot 的位置，可能在几何中心、也可能
在底面或某个角上。目标点落在工件的哪一处，决定了策略最终把 TCP 送到哪里，而它在配置
文件里是看不出来的，只能把球放进场景里用眼睛确认。

标三个球：

    绿球（半径 10mm）  = **当前训练目标点** = 根原点正上方 offset_cm，与奖励同源
    红球（半径 12mm）  = 工件根原点 object.data.root_pos_w（08-17 前的目标，仅参考）
    粉球（半径 8mm）   = TCP，ee_frame.data.target_pos_w（link7 + [0,0,0.177]）

**粉球与绿球重合就是任务目标本身** —— 奖励里比的就是这两个点的距离，success 判据是
重合到 1 cm 以内且姿态差 3° 以内。粉球刻意比目标球小：重合时它陷进去而不是把目标球
挡掉，一眼能分清"进去了"还是"停在旁边"。

**3 cm 偏移是在工件局部系里做的**，再由工件四元数转到世界系。这样工件 yaw 随机化
（±90°）时偏移会跟着工件转 —— 而这正是"沿工件自身某个轴"的语义。

局部偏移取 **-z 而不是 +z**：工件 ``init_state.rot = [0, 1, 0, 0]`` 是绕 X 轴
180°，它把局部 +z 翻到了世界 -z（朝下）。所以要得到"世界向上"，局部得走 -z。这条
符号很容易搞反，且搞反了画面上球会沉到工件下面 —— 因此脚本每次 reset 都把实际的
世界系 Δz 打出来，为负就大字告警。

marker 每帧跟随实时位姿刷新，不用初始值：工件每次 reset 都被
``reset_table_height_and_object_pose`` 挪位（xy ±0.125 m）并转 yaw（±90°），桌面
高度也在随机化，写死初始值的球会停在原地不动。

**只是可视化，不改任何训练配置**：不动奖励、不动观测、不动动作。机械臂全程零动作
（``KinematicRelativeJointDirectAction`` 下等于保持当前关节角），方便逐帧核对几何。

用法::

    # 默认 stage 1（不渲染相机，最轻，视口照常可看）
    ~/IsaacLab/isaaclab.sh -p scripts/train/mark_grasp_target.py

    # 想同时看相机管线出的点云就用 stage 2
    ~/IsaacLab/isaaclab.sh -p scripts/train/mark_grasp_target.py --stage 2

    # 改偏移距离 / 不自动 reset（工件停在一处慢慢看）
    ~/IsaacLab/isaaclab.sh -p scripts/train/mark_grasp_target.py \
        --offset_cm 5 --reset_period_s 0

``--offset_cm`` 要与训练用的 ``--grasp_offset_cm`` 保持一致，否则视口里看的目标点
不是训练在优化的那个点。

视口里找不到球时：Isaac Sim 左侧 Stage 树展开 ``/World/Markers``，双击
``grasp_target`` 可以让视口聚焦过去。

本脚本机械臂零动作，所以粉球停在初始位姿不会去追绿球 —— 想看策略把两球对上的过程，
用 ``play_pointcloud_overlay.py`` 或训练脚本的 ``--play`` 加载 checkpoint。
"""

import argparse
import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))
sys.path.insert(0, _PROJECT_DIR)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="标出训练抓取目标点（工件根原点上方 offset_cm）及根原点参考位置")
parser.add_argument("--stage", type=int, default=1, choices=(1, 2),
                    help="1=GT 采点不渲染相机(默认，最轻)；2=完整相机管线")
parser.add_argument("--offset_cm", type=float, default=3.0,
                    help="训练目标点(绿球)在根原点上方多少 cm，默认 3。"
                         "要与训练的 --grasp_offset_cm 一致")
parser.add_argument("--reset_period_s", type=float, default=4.0,
                    help="自动 reset 周期(秒)，看工件随机化后 marker 是否跟得上。"
                         "给 0 则不自动 reset，工件停在一处便于细看")
parser.add_argument("--seed", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# 程序化桌面用 PreviewSurface(MDL)，必须渲染型 Kit experience；视口本身也要它
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from configs.xarm7_pick_pointcloud_env_cfg import (
    XArm7PickPointCloudPlayEnvCfg,
    set_observation_stage,
)
from isaaclab.envs import ManagerBasedRLEnv

import configs.xarm7_pick_liftcube_mdp as custom_mdp

def _make_sphere_marker(stage, prim_path: str, color: tuple, radius: float = 0.012):
    """在场景里放一个自发光小球 marker，返回其 translate op。

    与 ``visualize_spatial_camfix_scene.py`` / ``train_pick_vision_bmw.py`` 里的同名
    函数逐字一致（自发光才能在任何光照 DR 下都看得清）。没有抽到公共模块是因为那两个
    脚本各自独立，这里保持同样的写法以免读代码时以为有什么不同。
    """
    from pxr import Gf, Sdf, UsdGeom, UsdShade

    sphere = UsdGeom.Sphere.Define(stage, prim_path)
    sphere.GetRadiusAttr().Set(radius)
    mat = UsdShade.Material.Define(stage, prim_path + "/mat")
    shader = UsdShade.Shader.Define(stage, prim_path + "/mat/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(sphere).Bind(mat)
    return UsdGeom.Xformable(sphere).AddTranslateOp()


def main() -> None:
    env_cfg = XArm7PickPointCloudPlayEnvCfg()
    env_cfg.scene.num_envs = 1
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    # keep_appearance=True：视口要看画面，桌面贴图与光照保留
    set_observation_stage(env_cfg, args_cli.stage, keep_appearance=True)

    # 纯观察：关掉会自动结束 episode 的 termination，工件不会因为超时或碰撞被重置。
    # 自动 reset 由本脚本按 --reset_period_s 自己控制。
    env_cfg.terminations.time_out = None
    if getattr(env_cfg.terminations, "reach_success", None) is not None:
        env_cfg.terminations.reach_success = None
    if getattr(env_cfg.terminations, "collision", None) is not None:
        env_cfg.terminations.collision = None

    env = ManagerBasedRLEnv(cfg=env_cfg)
    base_env = env.unwrapped
    device = base_env.device

    obj = base_env.scene["object"]
    ee_frame = base_env.scene["ee_frame"]

    import omni.usd
    from pxr import Gf

    stage = omni.usd.get_context().get_stage()
    # 红 = 工件根原点（参考位置，2026-08-17 之前的训练目标）；
    # 绿 = 当前训练目标点，即根原点上方 offset_cm（见 set_grasp_target_offset_cm）
    target_op = _make_sphere_marker(
        stage, "/World/Markers/grasp_target", color=(1.0, 0.0, 0.0), radius=0.012)
    above_op = _make_sphere_marker(
        stage, "/World/Markers/grasp_target_above", color=(0.0, 1.0, 0.2), radius=0.010)
    # 粉球 = TCP，即 ee_frame.data.target_pos_w[:, 0, :]（link7 + [0,0,0.177]）。
    # 这**正是**奖励里比距离的那个点，所以"粉球与绿球重合"就是任务目标；
    # success 判据是重合到 1 cm 以内且姿态差 3° 以内。
    # 半径取 8mm（比目标球小）：重合时粉球会陷进去而不是把它挡住，
    # 一眼能看出是"进去了"还是"停在旁边"。
    tcp_op = _make_sphere_marker(
        stage, "/World/Markers/tcp", color=(1.0, 0.35, 0.7), radius=0.008)

    offset_m = args_cli.offset_cm / 100.0
    # 偏移语义与训练完全同源：直接调 custom_mdp.grasp_target_pos_w，不在这里重算。
    # 局部 -z 的由来见那个函数的注释（工件 rot=[0,1,0,0] 把局部 +z 翻到世界 -z）。
    # 标记脚本自己复制一份偏移公式是最容易埋坑的地方 —— 视口里看着对，训练里瞄的
    # 却是另一个点。
    offset_local = (0.0, 0.0, -offset_m)

    print("\n" + "=" * 74)
    print(f"[标记] 绿球 = **当前训练目标点** = 工件根原点正上方 "
          f"{args_cli.offset_cm:.1f} cm")
    print(f"        奖励 object_ee_distance / success ee_reached_object 都用它，"
          f"经 custom_mdp.grasp_target_pos_w 同源计算")
    print(f"        训练侧开关：--grasp_offset_cm（默认 3.0，与本脚本 --offset_cm "
          f"对应）")
    print("[标记] 红球 = 工件根原点 object.data.root_pos_w"
          "（08-17 之前的训练目标，现仅作参考）")
    print("[标记] 粉球 = TCP（ee_frame，link7+0.177m）—— 任务是让它与**绿球**重合，"
          "success 判据 1cm/3°")
    print(f"[阶段] stage {args_cli.stage}"
          f"{'（不渲染相机）' if args_cli.stage == 1 else '（完整相机管线）'}")
    print(f"[重置] 每 {args_cli.reset_period_s:.0f} s 自动 reset"
          if args_cli.reset_period_s > 0 else "[重置] 不自动 reset")
    print("[视口] 找不到球：Stage 树展开 /World/Markers，双击 grasp_target 聚焦")
    print("=" * 74 + "\n")

    obs, _ = env.reset()

    zero_action = torch.zeros(
        (base_env.num_envs, base_env.action_manager.total_action_dim), device=device)

    import time

    last_reset_t = time.time()
    step = 0
    reported = False

    try:
        while simulation_app.is_running():
            obs, _, _, _, _ = env.step(zero_action)

            obj_pos = obj.data.root_pos_w[0]            # (3,) 世界系

            # 走训练用的同一个函数：工件 yaw 随机化时偏移随之旋转，这正是"沿工件
            # 自身轴"的语义（而不是恒定的世界 +z）。
            above_pos = custom_mdp.grasp_target_pos_w(obj, offset_local)[0]
            world_offset = above_pos - obj_pos

            ee_pos = ee_frame.data.target_pos_w[0, 0, :]    # TCP，世界系

            target_op.Set(Gf.Vec3d(*obj_pos.cpu().numpy().astype(float)))
            above_op.Set(Gf.Vec3d(*above_pos.cpu().numpy().astype(float)))
            tcp_op.Set(Gf.Vec3d(*ee_pos.cpu().numpy().astype(float)))

            # 每次 reset 后报一次实测值：符号错了这里立刻看得见
            if not reported:
                reported = True
                dz = float(world_offset[2])
                d_target = float(torch.norm(obj_pos - ee_pos))
                d_above = float(torch.norm(above_pos - ee_pos))
                print(f"[step {step:5d}] "
                      f"根原点(红) ({obj_pos[0]:+.3f}, {obj_pos[1]:+.3f}, "
                      f"{obj_pos[2]:+.3f})"
                      f"   训练目标(绿) ({above_pos[0]:+.3f}, {above_pos[1]:+.3f}, "
                      f"{above_pos[2]:+.3f})")
                print(f"            TCP(粉) ({ee_pos[0]:+.3f}, {ee_pos[1]:+.3f}, "
                      f"{ee_pos[2]:+.3f})   世界系偏移 Δ=({world_offset[0]:+.4f}, "
                      f"{world_offset[1]:+.4f}, {world_offset[2]:+.4f}) m")
                print(f"            粉→绿 {d_above:.3f} m（success 需 <0.010）   "
                      f"粉→红 {d_target:.3f} m（仅参考）")
                if dz < 0:
                    print("  [!! 方向反了 !!] 世界系 Δz 为负 —— 绿球在工件**下方**。"
                          "grasp_target_pos_w 的偏移符号需要翻过来（-z 改 +z）。")
                elif abs(dz - offset_m) > 1e-3:
                    # yaw 只绕世界 z 转，不该改变 Δz 的大小；对不上说明工件姿态里
                    # 还有别的旋转分量（例如 USD 自带的额外朝向）
                    print(f"  [注意] Δz={dz:+.4f} 与设定 {offset_m:.4f} 不等 —— "
                          f"工件姿态含非 yaw 旋转分量，绿球不是竖直正上方。")

            if args_cli.reset_period_s > 0 and \
                    time.time() - last_reset_t > args_cli.reset_period_s:
                obs, _ = env.reset()
                last_reset_t = time.time()
                reported = False        # 新位姿再报一次
            step += 1

    except KeyboardInterrupt:
        print("\n[退出] Ctrl-C")
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
