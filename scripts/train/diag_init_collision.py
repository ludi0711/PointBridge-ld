#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""初始化碰撞诊断：完全随机化跑 N 次 reset，找出“一初始化就碰撞”的工件位姿并可视化。

做什么：
    - 用 nopc 环境（完全随机化，含桌面高度随机），多环境并发，反复 reset。
    - 每次 reset 后让物理 settle 几步（--settle_steps，动作给 0，不加载策略），
      读 link7 接触力；若 > 阈值，判定为“初始化即碰撞”。
    - 记录每次 reset 的工件位姿（相对环境原点的 x,y,z 与 yaw）及是否碰撞、接触力。
    - 输出 CSV，并画 3D 散点图（碰撞=红，正常=蓝）+ 三视投影。

为什么不加载策略：
    只关心“随机化初始摆放本身是否穿插/接触”，与策略无关，动作恒为 0。

用法：
    /home/gxai/IsaacLab/isaaclab.sh -p scripts/train/diag_init_collision.py \
        --num_envs 200 --total_resets 10000 --settle_steps 2
"""

import argparse
import csv
import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="初始化碰撞诊断 + 工件位姿 3D 可视化")
parser.add_argument("--num_envs", type=int, default=200, help="并行环境数")
parser.add_argument("--total_resets", type=int, default=10000, help="总采样的 reset 次数")
parser.add_argument("--settle_steps", type=int, default=2,
                    help="reset 后让物理沉淀的步数，再读接触力判定（动作恒 0）")
parser.add_argument("--force_thresh", type=float, default=0.5, help="碰撞接触力阈值(N)")
parser.add_argument("--fix_table_height", action="store_true",
                    help="固定桌面高度，不做高度随机化（height_range=0）。"
                         "工件 x/y/yaw、相机、光照、桌色随机化仍保留。")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--out_dir", type=str,
                    default=os.path.join(_PROJECT_DIR, "logs", "diag_init_collision"))
parser.add_argument("--no_plot", action="store_true", help="只存 CSV，不画图（无显示环境用）")
parser.add_argument("--enable_cameras_render", action="store_true",
                    help="启用相机渲染管线（默认关闭）。本诊断只用接触力+工件位姿，"
                         "不需要相机；关闭可省 ~6GB 显存、避免与他人共卡时 OOM。")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = True
# 诊断不读 RGB/深度，默认不启动相机渲染管线（省显存、防 OOM）。
args_cli.enable_cameras = bool(args_cli.enable_cameras_render)

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from isaaclab.envs import ManagerBasedRLEnv
from configs.xarm7_pick_vision_nopc_env_cfg import XArm7PickLiftCubeEnvCfg


def _contact_force_per_env(base_env) -> torch.Tensor:
    sensor = base_env.scene.sensors["grip_contact"]
    forces = sensor.data.net_forces_w_history          # (N, hist, bodies, 3)
    return forces.norm(dim=-1).max(dim=1).values.sum(dim=-1)  # (N,)


def _object_pose_rel(base_env) -> tuple:
    """工件相对环境原点的位置 (N,3) 与 yaw (N,)。"""
    obj = base_env.scene["object"]
    pos_w = obj.data.root_pos_w                         # (N,3) 世界系
    pos_rel = pos_w - base_env.scene.env_origins        # 去掉各 env 栅格偏移
    quat = obj.data.root_quat_w                         # (N,4) wxyz
    w, x, y, z = quat.unbind(-1)
    yaw = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))
    return pos_rel, yaw


def main():
    env_cfg = XArm7PickLiftCubeEnvCfg()     # 完全随机化（不关任何随机）
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    # 不需要策略、不需要终止逻辑干扰采样；关掉 collision/reach_success/time_out 终止，
    # 由本脚本自己控制 reset 节奏（每 settle 后强制全体 reset）。
    for name in ("collision", "reach_success", "time_out"):
        if hasattr(env_cfg.terminations, name):
            setattr(env_cfg.terminations, name, None)

    # 可选：固定桌面高度（只关高度随机化，其余随机化保留）。
    if args_cli.fix_table_height:
        ev = env_cfg.events
        if hasattr(ev, "reset_table_and_object") and ev.reset_table_and_object is not None:
            ev.reset_table_and_object.params["height_range"] = 0.0
            print("[diag] 桌面高度随机化已关闭（height_range=0）", flush=True)
        else:
            print("[diag][WARN] 未找到 reset_table_and_object event，无法固定桌面高度",
                  flush=True)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    base_env = env.unwrapped
    device = env.device
    N = args_cli.num_envs
    act_dim = env.action_manager.total_action_dim
    zero_act = torch.zeros(N, act_dim, device=device)

    rows = []  # 每条: (x, y, z, yaw_deg, force, collided)
    collided_count = 0
    sampled = 0
    target = args_cli.total_resets

    print(f"[diag] 开始：目标 {target} 次 reset，{N} 并发，settle={args_cli.settle_steps}",
          flush=True)

    while sampled < target and simulation_app.is_running():
        # 全体 reset → 一批新的随机初始化
        env.reset()

        # 让物理沉淀几步（动作恒 0），使初始穿插的接触力显现并稳定
        for _ in range(max(1, args_cli.settle_steps)):
            env.step(zero_act)

        force = _contact_force_per_env(base_env)               # (N,)
        pos_rel, yaw = _object_pose_rel(base_env)              # (N,3), (N,)
        collided = force > args_cli.force_thresh               # (N,)

        take = min(N, target - sampled)
        f_cpu = force[:take].detach().cpu().numpy()
        p_cpu = pos_rel[:take].detach().cpu().numpy()
        y_cpu = (yaw[:take] * 180.0 / 3.14159265358979).detach().cpu().numpy()
        c_cpu = collided[:take].detach().cpu().numpy()

        for i in range(take):
            rows.append((
                float(p_cpu[i, 0]), float(p_cpu[i, 1]), float(p_cpu[i, 2]),
                float(y_cpu[i]), float(f_cpu[i]), bool(c_cpu[i]),
            ))
        collided_count += int(c_cpu.sum())
        sampled += take

        if sampled % (N * 5) < N:
            print(f"[diag] 进度 {sampled}/{target} | 累计初始碰撞 {collided_count} "
                  f"({100.0*collided_count/max(sampled,1):.2f}%)", flush=True)

    env.close()

    # ── 存 CSV ────────────────────────────────────────────────────────────
    # 固定/随机桌高分开存，避免互相覆盖。
    _suffix = "fixedh" if args_cli.fix_table_height else "randh"
    os.makedirs(args_cli.out_dir, exist_ok=True)
    csv_path = os.path.join(args_cli.out_dir, f"init_collision_samples_{_suffix}.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["x", "y", "z", "yaw_deg", "contact_force_N", "collided"])
        for r in rows:
            w.writerow(r)
    rate = 100.0 * collided_count / max(sampled, 1)
    print(f"\n[diag] 完成。采样 {sampled} 次，初始碰撞 {collided_count} 次（{rate:.2f}%）。",
          flush=True)
    print(f"[diag] CSV: {csv_path}", flush=True)

    # ── 画图 ──────────────────────────────────────────────────────────────
    if args_cli.no_plot:
        return
    try:
        import matplotlib
        matplotlib.use("Agg")  # 无界面后端，直接存 PNG
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
        import numpy as np

        arr = np.array([(r[0], r[1], r[2], r[5]) for r in rows], dtype=float)
        xs, ys, zs, col = arr[:, 0], arr[:, 1], arr[:, 2], arr[:, 3] > 0.5
        ok = ~col

        fig = plt.figure(figsize=(15, 11))
        _table_tag = "固定桌高" if args_cli.fix_table_height else "桌高随机"
        fig.suptitle(
            f"初始化碰撞诊断 [{_table_tag}]  样本={sampled}  碰撞={collided_count} ({rate:.2f}%)  "
            f"阈值={args_cli.force_thresh}N  settle={args_cli.settle_steps}",
            fontsize=12,
        )

        # 3D 散点
        ax3d = fig.add_subplot(2, 2, 1, projection="3d")
        ax3d.scatter(xs[ok], ys[ok], zs[ok], s=3, c="steelblue", alpha=0.25, label="正常")
        ax3d.scatter(xs[col], ys[col], zs[col], s=8, c="red", alpha=0.7, label="初始碰撞")
        ax3d.set_xlabel("x (m)"); ax3d.set_ylabel("y (m)"); ax3d.set_zlabel("z (m)")
        ax3d.set_title("工件初始位置 3D 分布")
        ax3d.legend(loc="upper right", fontsize=8)

        # 俯视 XY
        axxy = fig.add_subplot(2, 2, 2)
        axxy.scatter(xs[ok], ys[ok], s=3, c="steelblue", alpha=0.2)
        axxy.scatter(xs[col], ys[col], s=8, c="red", alpha=0.7)
        axxy.set_xlabel("x (m)"); axxy.set_ylabel("y (m)")
        axxy.set_title("俯视 (X-Y)"); axxy.grid(True, alpha=0.3); axxy.set_aspect("equal")

        # 侧视 XZ
        axxz = fig.add_subplot(2, 2, 3)
        axxz.scatter(xs[ok], zs[ok], s=3, c="steelblue", alpha=0.2)
        axxz.scatter(xs[col], zs[col], s=8, c="red", alpha=0.7)
        axxz.set_xlabel("x (m)"); axxz.set_ylabel("z (m)")
        axxz.set_title("侧视 (X-Z)"); axxz.grid(True, alpha=0.3)

        # 侧视 YZ
        axyz = fig.add_subplot(2, 2, 4)
        axyz.scatter(ys[ok], zs[ok], s=3, c="steelblue", alpha=0.2)
        axyz.scatter(ys[col], zs[col], s=8, c="red", alpha=0.7)
        axyz.set_xlabel("y (m)"); axyz.set_ylabel("z (m)")
        axyz.set_title("侧视 (Y-Z)"); axyz.grid(True, alpha=0.3)

        fig.tight_layout(rect=[0, 0, 1, 0.96])
        png_path = os.path.join(args_cli.out_dir, f"init_collision_3d_{_suffix}.png")
        fig.savefig(png_path, dpi=130)
        print(f"[diag] 图已保存: {png_path}", flush=True)
    except Exception as exc:
        print(f"[diag][WARN] 画图失败（CSV 已存）：{type(exc).__name__}: {exc}", flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nError: {exc}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
