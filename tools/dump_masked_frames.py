#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""抓「成功掩码」帧 —— 判断训练吃到的 YOLO 掩码到底对不对。

``yolo_empty_debug`` 只存了**空检失败**的帧，成功帧一张没留。光看失败帧只能知道
"什么时候检不出来"，不知道"检出来的时候贴得准不准"——而后者才是判断训练是否正确的
关键：掩码贴到机械臂 / 反光 / 桌面上，策略照样会学到错的东西。

本脚本短跑 stage4 环境，对每个 **YOLO 检出成功（detected=True）** 的 env 存一张
RGB+掩码叠加图（绿色高亮 = 掩码区域，文件名带 conf 和检出数），供人工核对。
顺带统计检出率，并少量存几张 miss 帧做对照。

要点：
  - 相机这一路和训练完全一致：``rgb = env.scene["camera_fixed"].data.output["rgb"]``，
    掩码 = ``yolo_masks(rgb)``（就是训练里用的同一个函数、同一份权重）。
  - 掩码叠加是**逐像素绿色高亮 + 边界描边**，偏没偏一眼能看出来。

用法::

    # 零动作短跑（工件随机摆放，8 个 env 8 种姿态）
    YOLO_DEPS_DIR=/root/autodl-tmp/yolo_deps /root/gx-va/bin/python \
      tools/dump_masked_frames.py --headless --num_envs 8 --steps 30

    # 用训练出的策略走真实接近轨迹（能看到夹爪自遮挡时刻的掩码表现）
    ... --checkpoint /root/autodl-tmp/logs/xarm7_pick_pointcloud_stage4/<run>/model_9300.pt

    # 输出目录（数据盘）
    ... --out /root/autodl-tmp/mask_ok_vis

注意：需要**空闲 GPU**——训练跑满 98% 显存时先别启动本脚本，会 OOM。
"""

import argparse
import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_CURRENT_DIR)  # tools/ 的上一级 = 项目根
sys.path.insert(0, _PROJECT_DIR)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="抓成功掩码帧，核对训练吃到的 YOLO 掩码")
parser.add_argument("--num_envs", type=int, default=8)
parser.add_argument("--steps", type=int, default=30, help="总步数")
parser.add_argument("--reset_every", type=int, default=0,
                    help="每 N 步手动 reset 一次换新工件姿态；0=不手动 reset（靠 episode 自然循环）")
parser.add_argument("--out", type=str, default="/root/autodl-tmp/mask_ok_vis")
parser.add_argument("--checkpoint", type=str, default=None,
                    help="策略权重。给则走真实接近轨迹，不给则零动作（工件随机摆放）")
parser.add_argument("--max_ok", type=int, default=200, help="最多存多少张成功帧")
parser.add_argument("--max_miss", type=int, default=20, help="最多存多少张 miss 对照帧")
parser.add_argument("--yolo_weights", type=str, default=None)
parser.add_argument("--yolo_conf", type=float, default=None)
parser.add_argument("--seed", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
import numpy as np
from PIL import Image

from configs.xarm7_pick_pointcloud_env_cfg import XArm7PickPointCloudEnvCfg
from configs.point_bridge_pointcloud import M_OBJ, N_ROBOT, POINT_DIM
from configs.yolo_mask_source import (
    DEFAULT_YOLO_WEIGHTS,
    YOLO_CONF,
    yolo_masks,
)
from isaaclab.envs import ManagerBasedRLEnv


def _rgb_to_np_uint8(img: torch.Tensor) -> np.ndarray:
    """(H,W,3/4) 张量 → (H,W,3) numpy uint8。"""
    img = img.detach()
    if img.shape[-1] == 4:
        img = img[..., :3]
    img = img.float()
    if img.numel() and float(img.max()) > 1.5:
        img = img / 255.0
    return (img.clamp(0.0, 1.0) * 255.0).round().to(torch.uint8).cpu().numpy()


def _overlay(rgb_u8: np.ndarray, mask: torch.Tensor) -> np.ndarray:
    """RGB 上叠加绿色掩码高亮 + 边界描边。"""
    arr = rgb_u8.astype(np.float32).copy()
    m = mask.detach().cpu().numpy().astype(bool)

    # 边界：mask 与其相邻像素不一致的点（简单的 4 邻域检测，不依赖 scipy）
    boundary = m & ~(
        np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1)
    )

    green = np.array([0.0, 255.0, 0.0])
    # 掩码内部：6 成绿 + 4 成原图；边界：纯绿描边
    arr[m] = arr[m] * 0.4 + green * 0.6
    arr[boundary] = green
    return np.clip(arr, 0, 255).astype(np.uint8)


def main():
    cfg = XArm7PickPointCloudEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        cfg.seed = args_cli.seed

    env = ManagerBasedRLEnv(cfg=cfg)

    weights = args_cli.yolo_weights or DEFAULT_YOLO_WEIGHTS
    conf = args_cli.yolo_conf if args_cli.yolo_conf is not None else YOLO_CONF

    os.makedirs(args_cli.out, exist_ok=True)

    # 可选：加载策略走真实轨迹（rsl_rl 需要 RslRlVecEnvWrapper 包一层）
    policy = None
    vec_env = None
    if args_cli.checkpoint is not None:
        from model.pointnet_actor_critic import register_with_rsl_rl
        from configs.agents.rsl_rl_ppo_cfg import XArm7PickPointCloudPPORunnerCfg
        from rsl_rl.runners import OnPolicyRunner
        from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

        register_with_rsl_rl()
        agent_cfg = XArm7PickPointCloudPPORunnerCfg()
        # 网络侧点云形状与环境侧一致，从同一份常量推导（与训练脚本相同）
        agent_cfg.policy.num_object_points = M_OBJ
        agent_cfg.policy.num_robot_points = N_ROBOT
        agent_cfg.policy.point_dim = POINT_DIM
        vec_env = RslRlVecEnvWrapper(env)
        runner = OnPolicyRunner(
            vec_env, agent_cfg.to_dict(), log_dir=args_cli.out, device=env.device
        )
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=env.device)
        print(f"[策略] 已加载 {args_cli.checkpoint}")

    print(f"[掩码] weights={weights}  conf={conf}")
    print(f"[输出] {args_cli.out}")

    if vec_env is not None:
        obs, _ = vec_env.reset()
    else:
        obs, _ = env.reset()

    n_ok = n_miss = n_checked = 0

    try:
        for step in range(args_cli.steps):
            rgb = env.scene["camera_fixed"].data.output["rgb"]  # (B,H,W,3/4)
            mask, detected, details = yolo_masks(
                rgb, weights=weights, conf=conf, return_details=True
            )
            best_conf = details["best_conf"].cpu().numpy()
            n_det = details["n_det"].cpu().numpy()

            for i in range(args_cli.num_envs):
                if n_ok >= args_cli.max_ok and n_miss >= args_cli.max_miss:
                    break
                n_checked += 1
                rgb_u8 = _rgb_to_np_uint8(rgb[i])
                if detected[i] and n_ok < args_cli.max_ok:
                    vis = _overlay(rgb_u8, mask[i])
                    fn = (f"ok_env{i}_conf{best_conf[i]:.2f}_nd{int(n_det[i])}"
                          f"_s{step}.png")
                    Image.fromarray(vis).save(os.path.join(args_cli.out, fn))
                    n_ok += 1
                elif not detected[i] and n_miss < args_cli.max_miss:
                    fn = f"miss_env{i}_s{step}.png"
                    Image.fromarray(rgb_u8).save(os.path.join(args_cli.out, fn))
                    n_miss += 1

            if n_ok >= args_cli.max_ok and n_miss >= args_cli.max_miss:
                break

            if policy is not None:
                with torch.no_grad():
                    actions = policy(obs)
            else:
                actions = torch.zeros(
                    (args_cli.num_envs, env.action_manager.total_action_dim),
                    device=env.device,
                )

            if args_cli.reset_every and (step + 1) % args_cli.reset_every == 0:
                if vec_env is not None:
                    obs, _ = vec_env.reset()
                else:
                    obs, _ = env.reset()
            elif vec_env is not None:
                obs, _, _, _ = vec_env.step(actions)
            else:
                env.step(actions)  # 零动作：不需要 obs，5 元组返回值直接丢弃
    finally:
        env.close()

    total = n_ok + n_miss
    rate = n_ok / max(1, n_checked) * 100
    print("\n" + "=" * 60)
    print(f"完成。共检查 {n_checked} 个 env·步，检出成功 {n_ok}（{rate:.0f}%），"
          f"miss {n_miss}")
    print(f"图片在: {args_cli.out}")
    print("看 ok_*.png：绿色高亮应紧贴铝件轮廓。若绿色贴到机械臂/桌面/反光上，"
          "说明 YOLO 掩码错了，训练在吃脏数据。")
    print("=" * 60)


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close()
