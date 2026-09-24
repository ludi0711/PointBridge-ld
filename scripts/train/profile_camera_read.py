#!/usr/bin/env python3
"""独立脚本：测量 stage2 里 camera.data.output 访问的耗时。

用法（在 CZR 项目根目录下）：
    conda activate isaaclab
    cd /home/gxai/Desktop/CZR/gx-VA-isaaclab
    python scripts/train/profile_camera_read.py --num_envs 128 --headless --enable_cameras

输出格式与 preplace 任务的 [PC_PROF] 一致，方便直接对比。
"""

import argparse
import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))
sys.path.insert(0, _PROJECT_DIR)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Profile camera.data.output access in stage2")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--num_steps", type=int, default=300, help="Number of steps to profile")
parser.add_argument("--print_every", type=int, default=100, help="Print summary every N steps")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import time
import torch

from configs.xarm7_pick_pointcloud_env_cfg import (
    XArm7PickPointCloudEnvCfg,
    set_observation_stage,
)
from isaaclab.envs import ManagerBasedRLEnv

# stage 2: 完整相机管线
cfg = XArm7PickPointCloudEnvCfg()
cfg.scene.num_envs = args_cli.num_envs
set_observation_stage(cfg, stage=2)

env = ManagerBasedRLEnv(cfg=cfg)
obs, _ = env.reset()
action_dim = env.action_manager.total_action_dim

# 找到 camera_fixed
camera = env.scene.sensors.get("camera_fixed")
if camera is None:
    print("[ERROR] camera_fixed not found in scene sensors")
    env.close()
    simulation_app.close()
    sys.exit(1)

print(f"[PROF] camera resolution: {camera.cfg.width}x{camera.cfg.height}")
print(f"[PROF] data_types: {camera.cfg.data_types}")
print(f"[PROF] num_envs: {args_cli.num_envs}")
print(f"[PROF] profiling {args_cli.num_steps} steps, printing every {args_cli.print_every}...")

timing = {
    "output_attr": 0.0,
    "depth_get": 0.0,
    "seg_get": 0.0,
    "depth_sync": 0.0,
    "seg_sync": 0.0,
    "total_camera": 0.0,
    "step_total": 0.0,
}
n = 0

for step in range(args_cli.num_steps):
    actions = torch.zeros(args_cli.num_envs, action_dim, device=env.device)

    t_step_start = time.perf_counter()
    obs, rew, done, trunc, info = env.step(actions)
    t_after_step = time.perf_counter()

    # 单独计时 camera.data.output 访问
    t0 = time.perf_counter()
    output = camera.data.output
    ta = time.perf_counter()
    depth_raw = output.get("distance_to_image_plane")
    tb = time.perf_counter()
    seg_raw = output.get("instance_id_segmentation_fast")
    tc = time.perf_counter()
    if depth_raw is not None:
        _ = depth_raw.shape
    td = time.perf_counter()
    if seg_raw is not None:
        _ = seg_raw.shape
    te = time.perf_counter()

    timing["output_attr"] += (ta - t0)
    timing["depth_get"] += (tb - ta)
    timing["seg_get"] += (tc - tb)
    timing["depth_sync"] += (td - tc)
    timing["seg_sync"] += (te - td)
    timing["total_camera"] += (te - t0)
    timing["step_total"] += (t_after_step - t_step_start)
    n += 1

    if n % args_cli.print_every == 0:
        H = camera.cfg.height
        W = camera.cfg.width
        print(
            f"[PC_PROF] steps={n} | "
            f"camera_read={timing['total_camera']/n*1000:.2f}ms"
            f"  (output_attr={timing['output_attr']/n*1000:.2f}ms"
            f"  depth_get={timing['depth_get']/n*1000:.2f}ms"
            f"  seg_get={timing['seg_get']/n*1000:.2f}ms"
            f"  depth_sync={timing['depth_sync']/n*1000:.2f}ms"
            f"  seg_sync={timing['seg_sync']/n*1000:.2f}ms) | "
            f"step_total={timing['step_total']/n*1000:.2f}ms  "
            f"(B={args_cli.num_envs}, H={H}, W={W})"
        )

env.close()
simulation_app.close()

