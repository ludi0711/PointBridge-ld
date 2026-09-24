#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""核对工件随机化的**实际**落点分布，确认配置改动真的生效。

改随机化范围有两个各自独立、且都不报错的失败模式：

  1. 只改了 ``x_range/y_range`` 没改 ``object.init_state`` —— 框大了但仍偏在老
     中心。曲线上看不出来，只有真机摆到新位置时策略才突然不行。
  2. 平移方向搞反 —— 基座系 +x 对应 env 系 **-y**（基座 rot 绕 z -90°）。写成
     env +y 同样不报错，工件只是挪到了离基座更近处。

所以这里不看配置里写了什么，而是真的 reset 若干次、读 ``object`` 的 root pose，
统计基座系下的实际范围，与期望值对账。

用法（必须走 isaaclab.sh，本模块依赖 omni.client）::

    ~/IsaacLab/isaaclab.sh -p scripts/train/check_randomization_range.py --resets 40

只建环境、reset、读位姿，不训练、不写 checkpoint。
"""

import argparse
import os
import sys

import numpy as np

_PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _PROJECT_DIR)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="核对工件随机化实际落点")
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--resets", type=int, default=40,
                    help="reset 次数；样本数 = num_envs * resets")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch  # noqa: E402

from configs.xarm7_pick_pointcloud_env_cfg import (  # noqa: E402
    XArm7PickPointCloudEnvCfg,
    set_observation_stage,
)
from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402

# 期望值。与 configs 里的两处旋钮对应，改配置后同步改这里 —— 对不上就该报警。
EXPECT_CENTER_ENV_Y = -0.49      # object.init_state 的 y
EXPECT_HALF = 0.125              # x_range/y_range 的半宽
# 基座位姿（camfix 场景）：env 原点 + (0,0,0.822)，绕 z -90°
BASE_POS = np.array([0.0, 0.0, 0.822])
BASE_QUAT_WXYZ = (0.707, 0.0, 0.0, -0.707)


def quat_to_matrix(q):
    w, x, y, z = q
    n = (w * w + x * x + y * y + z * z) ** 0.5
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])


def main() -> None:
    cfg = XArm7PickPointCloudEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.sim.device = args_cli.device if hasattr(args_cli, "device") else "cuda:0"
    # stage 2 = 完整点云管线，与实际训练一致。随机化与 stage 无关，但保持一致
    # 可以顺带确认 stage 切换没把事件项覆盖掉。
    set_observation_stage(cfg, 2)

    env = ManagerBasedRLEnv(cfg=cfg)
    obj = env.scene["object"]

    # 先读 default_root_state，这是随机化的**中心**，直接暴露旋钮 1 是否生效
    default_xy = obj.data.default_root_state[:, :2].clone().cpu().numpy()
    origins = env.scene.env_origins[:, :2].cpu().numpy()
    center_env = (default_xy - origins).mean(axis=0) if default_xy.shape == origins.shape \
        else default_xy.mean(axis=0)
    print(f"\n[中心] default_root_state (env系, 去掉env_origin) = "
          f"x={center_env[0]:+.4f} y={center_env[1]:+.4f}")

    samples = []
    for i in range(args_cli.resets):
        env.reset()
        xy = obj.data.root_pos_w[:, :2].cpu().numpy() - origins
        samples.append(xy)
    env.close()

    pts = np.concatenate(samples, axis=0)
    print(f"[样本] {len(pts)} 个 ({args_cli.num_envs} env x {args_cli.resets} reset)\n")

    # env 系统计
    print("── env 系 ──")
    for k, ax in ((0, "x"), (1, "y")):
        lo, hi = pts[:, k].min(), pts[:, k].max()
        print(f"  {ax}: {lo:+.4f} ~ {hi:+.4f}   (span {hi - lo:.4f}, "
              f"mean {pts[:, k].mean():+.4f})")

    # 基座系统计 —— 真机部署看的是这一组
    R = quat_to_matrix(BASE_QUAT_WXYZ)
    pts3 = np.column_stack([pts, np.full(len(pts), 0.833)])
    base = (pts3 - BASE_POS) @ R
    print("\n── 基座系（真机部署用） ──")
    for k, ax in ((0, "x"), (1, "y")):
        lo, hi = base[:, k].min(), base[:, k].max()
        print(f"  {ax}: {lo:+.4f} ~ {hi:+.4f}   (span {hi - lo:.4f}, "
              f"mean {base[:, k].mean():+.4f})")

    # 对账。用 span 而不是 min/max 单独判：均匀采样在有限样本下取不满边界，
    # 但 span 会随样本数稳定逼近 2*half，偏差主要来自采样而非配置错误。
    print("\n── 对账 ──")
    ok = True
    exp_span = 2 * EXPECT_HALF
    for k, ax in ((0, "x"), (1, "y")):
        span = pts[:, k].max() - pts[:, k].min()
        good = abs(span - exp_span) < 0.02
        ok &= good
        print(f"  {ax} span {span:.4f} vs 期望 {exp_span:.4f}  "
              f"{'OK' if good else '<<< 不符'}")
    dy = abs(center_env[1] - EXPECT_CENTER_ENV_Y)
    good = dy < 1e-3
    ok &= good
    print(f"  中心 env y {center_env[1]:+.4f} vs 期望 {EXPECT_CENTER_ENV_Y:+.4f}  "
          f"{'OK' if good else '<<< 不符'}")
    base_x = (np.array([0.0, center_env[1], 0.833]) - BASE_POS) @ R
    print(f"  → 基座系中心 x={base_x[0]:+.4f} y={base_x[1]:+.4f}  "
          f"（期望 x=+0.4900）")

    print(f"\n{'[通过] 配置改动已生效' if ok else '[失败] 配置与期望不符，别开训'}")


if __name__ == "__main__":
    main()
    simulation_app.close()
