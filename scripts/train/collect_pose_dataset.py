#!/usr/bin/env python3
"""采集「固定相机 RGB → 工件位姿」数据集（光照 + 工件位姿双随机化）。

每次 ``env.reset()`` 触发一整套域随机化（工件 xy/yaw + 每 env 独立光照），
稳定几帧后存**一张**图。并行 N 个 env 则一轮产出 N 帧，各帧的光照和工件位姿
互不相同 —— 光源已是 ``{ENV_REGEX_NS}/Light``，逐 env 采样。

位姿存**机械臂基座系**：真机手眼标定结果就在这个系里，sim2real 直接可比。
同时把 env 局部系也记进 metadata，出问题时能和配置常量直接对照。

关于 warmup：reset 后光照属性刚写进 USD，RTX 需要几帧才收敛。省掉 warmup
会拿到上一轮的光照 —— 图像和标签就对不上了。默认 4 帧，别调到 0。

用法::

    ~/IsaacLab/isaaclab.sh -p scripts/train/collect_pose_dataset.py \
        --num_frames 1000 --num_envs 8 --seed 42 --out logs/pose_dataset_v1

输出::

    out/
    ├── frames/frame_000000/{rgb.png, object_pose.npy, metadata.json}
    ├── summary.npz
    ├── dataset_info.json     相机内外参 + 随机化范围（换机器复现全靠它）
    └── preview_grid.png      抽检拼图（--preview）
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))
sys.path.insert(0, _PROJECT_DIR)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="采集固定相机 RGB + 工件位姿数据集")
parser.add_argument("--num_frames", type=int, default=1000, help="目标总帧数")
parser.add_argument(
    "--num_envs",
    type=int,
    default=8,
    help="并行环境数。每轮 reset 产出 num_envs 帧，光照条件数 = 总轮数。"
         "调大更快但光照种类变少（总帧数固定时）。",
)
parser.add_argument("--seed", type=int, default=42, help="随机种子")
parser.add_argument(
    "--warmup",
    type=int,
    default=4,
    help="reset 后空步数，等 RTX 渲染收敛。设 0 会拿到上一轮的光照。",
)
parser.add_argument(
    "--env_spacing",
    type=float,
    default=6.0,
    help="env 间距。默认 6.0 确保邻居不入画（2.5 时邻桌可能出现在画面边缘）。",
)
parser.add_argument("--out", type=str, default=os.path.join(_PROJECT_DIR, "logs", "pose_dataset"))
parser.add_argument("--preview", type=int, default=8, help="抽检拼图帧数，0 表示不生成")
parser.add_argument("--io_workers", type=int, default=8, help="异步写盘线程数")
parser.add_argument(
    "--rgb_only",
    action="store_true",
    default=True,
    help="只渲染 RGB，关掉深度和分割两路缓冲。显存降到约 1/3，"
         "是能否开大 num_envs 的关键。默认开启。",
)
parser.add_argument(
    "--keep_all_data_types",
    dest="rgb_only",
    action="store_false",
    help="保留深度/分割（本脚本用不到，只在排查渲染问题时才需要）",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.enable_cameras = True
args_cli.headless = True  # 采集不需要 GUI，headless 更快

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2
import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.utils.math import subtract_frame_transforms

from configs.xarm7_pick_pointcloud_env_cfg import (
    CANONICAL_HEIGHT,
    CANONICAL_WIDTH,
    FIXED_CAMERA_CONVENTION,
    XArm7PickPointCloudPlayEnvCfg,
    canonical_intrinsic_matrix,
)
from configs.xarm7_pick_vision_spatial_camfix_env_cfg import (
    INIT_JOINT_POS,
    _LIGHT_COLOR_TEMP_RANGE,
    _LIGHT_EXPOSURE_RANGE,
    _LIGHT_MAX,
    _LIGHT_MIN,
    _LIGHT_RADIUS_RANGE,
)


def _to_uint8_rgb(rgb: torch.Tensor) -> np.ndarray:
    """相机 rgb 输出 → (N, H, W, 3) uint8，兼容 float 与带 alpha 的情况。"""
    arr = rgb.detach().cpu().numpy()
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        hi = float(np.nanmax(arr)) if arr.size else 0.0
        # >1.5 说明本来就是 0~255 的 float，不能再乘 255
        scale = 1.0 if hi > 1.5 else 255.0
        arr = np.clip(np.nan_to_num(arr) * scale, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _write_frame(frame_dir: str, rgb: np.ndarray, pose_base: np.ndarray, meta: dict) -> None:
    """写一帧的三个文件。在线程池里跑，主循环不等 IO。"""
    os.makedirs(frame_dir, exist_ok=True)
    cv2.imwrite(os.path.join(frame_dir, "rgb.png"), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    np.save(os.path.join(frame_dir, "object_pose.npy"), pose_base.astype(np.float32))
    with open(os.path.join(frame_dir, "metadata.json"), "w") as f:
        json.dump(meta, f, indent=2)


def main() -> None:
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    # 域随机化用的是 configs 里的 `random` 模块（_random），必须单独播种，
    # 否则 --seed 对光照/工件位姿不起作用。
    import random as _py_random

    _py_random.seed(args_cli.seed)

    cfg = XArm7PickPointCloudPlayEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.scene.env_spacing = args_cli.env_spacing
    cfg.seed = args_cli.seed

    # 只保留 rgb。相机默认还开了 distance_to_image_plane 和
    # instance_id_segmentation_fast，各自占一份和 RGB 同量级的 GPU 缓冲，
    # 而采集只用得到 RGB —— 关掉这两路，显存占用降到约 1/3，
    # 是 num_envs 能开多大的决定性因素（256 env 全开必 OOM）。
    if args_cli.rgb_only:
        for cam_name in ("camera_fixed", "camera"):
            cam_cfg = getattr(cfg.scene, cam_name, None)
            if cam_cfg is not None and hasattr(cam_cfg, "data_types"):
                cam_cfg.data_types = ["rgb"]

    env = ManagerBasedRLEnv(cfg=cfg)
    camera = env.scene["camera_fixed"]
    robot = env.scene["robot"]
    obj = env.scene["object"]

    num_envs = env.num_envs
    num_rounds = (args_cli.num_frames + num_envs - 1) // num_envs
    total_frames = num_rounds * num_envs

    frames_dir = os.path.join(args_cli.out, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    print("=" * 78)
    print("数据采集")
    print(f"  输出目录      : {args_cli.out}")
    print(f"  并行环境数    : {num_envs}   env_spacing={args_cli.env_spacing}")
    print(f"  轮数 x 每轮帧 : {num_rounds} x {num_envs} = {total_frames} 帧")
    print(f"  光照条件数    : {num_rounds}  (同一轮内各 env 光照独立，轮间重新采样)")
    print(f"  图像          : {CANONICAL_WIDTH}x{CANONICAL_HEIGHT}  camera_fixed")
    print(f"  位姿坐标系    : 机械臂基座系 (object_pose.npy)")
    print(f"  warmup        : {args_cli.warmup} 帧")
    print(f"  rgb_only      : {args_cli.rgb_only}"
          f"{'  (已关闭深度/分割，省显存)' if args_cli.rgb_only else '  (全部 data_types，显存占用高)'}")
    print(f"  seed          : {args_cli.seed}")
    print("=" * 78)

    action_dim = env.action_manager.total_action_dim
    zero_action = torch.zeros((num_envs, action_dim), device=env.device)

    poses_base, poses_env, light_params, frame_idx_list = [], [], [], []
    preview_tiles = []
    t0 = time.time()

    pool = ThreadPoolExecutor(max_workers=args_cli.io_workers)
    futures = []
    frame_id = 0

    for rnd in range(num_rounds):
        env.reset()
        for _ in range(args_cli.warmup):
            env.step(zero_action)

        rgb_batch = _to_uint8_rgb(camera.data.output["rgb"])

        # 工件位姿 → 基座系。基座有 -90° 偏航，必须做完整位姿变换，
        # 不能只减位置（姿态也会变）。
        pos_b, quat_b = subtract_frame_transforms(
            robot.data.root_pos_w, robot.data.root_quat_w,
            obj.data.root_pos_w, obj.data.root_quat_w,
        )
        pos_e = obj.data.root_pos_w - env.scene.env_origins
        quat_e = obj.data.root_quat_w

        pos_b_np = pos_b.detach().cpu().numpy()
        quat_b_np = quat_b.detach().cpu().numpy()
        pos_e_np = pos_e.detach().cpu().numpy()
        quat_e_np = quat_e.detach().cpu().numpy()
        lp = getattr(env, "_light_params", [[0.0] * 4] * num_envs)

        ts = time.time()
        for i in range(num_envs):
            pose_base = np.concatenate([pos_b_np[i], quat_b_np[i]])
            pose_env = np.concatenate([pos_e_np[i], quat_e_np[i]])
            meta = {
                "frame": frame_id,
                "round": rnd,
                "env_index": i,
                "timestamp_utc": ts,
                "object_position_base_xyz": [round(float(v), 6) for v in pos_b_np[i]],
                "object_quaternion_base_wxyz": [round(float(v), 6) for v in quat_b_np[i]],
                "object_position_env_xyz": [round(float(v), 6) for v in pos_e_np[i]],
                "object_quaternion_env_wxyz": [round(float(v), 6) for v in quat_e_np[i]],
                "light": {
                    "intensity": round(float(lp[i][0]), 3),
                    "color_temperature": round(float(lp[i][1]), 3),
                    "exposure": round(float(lp[i][2]), 4),
                    "radius": round(float(lp[i][3]), 5),
                },
                "image_shape_hwc": list(rgb_batch[i].shape),
            }
            frame_dir = os.path.join(frames_dir, f"frame_{frame_id:06d}")
            futures.append(
                pool.submit(_write_frame, frame_dir, rgb_batch[i].copy(), pose_base, meta)
            )

            poses_base.append(pose_base)
            poses_env.append(pose_env)
            light_params.append([float(v) for v in lp[i]])
            frame_idx_list.append(frame_id)

            if args_cli.preview and len(preview_tiles) < args_cli.preview:
                tile = cv2.resize(
                    cv2.cvtColor(rgb_batch[i], cv2.COLOR_RGB2BGR), (320, 240)
                )
                cv2.rectangle(tile, (0, 0), (320, 34), (0, 0, 0), -1)
                cv2.putText(tile, f"f{frame_id} I={lp[i][0]:.0f} T={lp[i][1]:.0f}K",
                            (4, 14), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                            (255, 255, 255), 1, cv2.LINE_AA)
                cv2.putText(tile,
                            f"obj_base ({pos_b_np[i][0]:+.3f},{pos_b_np[i][1]:+.3f})",
                            (4, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                            (120, 220, 255), 1, cv2.LINE_AA)
                preview_tiles.append(tile)

            frame_id += 1

        if (rnd + 1) % max(1, num_rounds // 20) == 0 or rnd == num_rounds - 1:
            el = time.time() - t0
            fps = frame_id / el if el > 0 else 0.0
            eta = (total_frames - frame_id) / fps if fps > 0 else 0.0
            print(f"  [{frame_id:6d}/{total_frames}] "
                  f"{100*frame_id/total_frames:5.1f}%  {fps:.1f} 帧/s  ETA {eta:.0f}s")

    print("等待写盘完成 ...")
    for fu in futures:
        fu.result()  # 让异常浮出来，别静默丢帧
    pool.shutdown()

    poses_base = np.asarray(poses_base, dtype=np.float32)
    poses_env = np.asarray(poses_env, dtype=np.float32)
    light_params = np.asarray(light_params, dtype=np.float32)
    frame_indices = np.asarray(frame_idx_list, dtype=np.int32)

    np.savez(
        os.path.join(args_cli.out, "summary.npz"),
        object_poses_base=poses_base,
        object_poses_env=poses_env,
        light_params=light_params,
        frame_indices=frame_indices,
        task="XArm7PickPointCloudPlay",
        seed=args_cli.seed,
        num_frames=total_frames,
        pose_frame="robot_base",
        light_param_order="intensity,color_temperature,exposure,radius",
    )

    # canonical_intrinsic_matrix 返回**扁平的 9 元素行主序列表**，不是嵌套 3x3
    K = canonical_intrinsic_matrix(CANONICAL_WIDTH, CANONICAL_HEIGHT)
    K33 = [[float(K[3 * r + c]) for c in range(3)] for r in range(3)]
    cam_pos_w = camera.data.pos_w[0].detach().cpu().numpy()
    cam_quat_w = camera.data.quat_w_world[0].detach().cpu().numpy()
    cam_pos_b, cam_quat_b = subtract_frame_transforms(
        robot.data.root_pos_w[:1], robot.data.root_quat_w[:1],
        camera.data.pos_w[:1], camera.data.quat_w_world[:1],
    )
    info = {
        "task": "XArm7PickPointCloudPlay",
        "seed": args_cli.seed,
        "num_frames": total_frames,
        "num_envs": num_envs,
        "num_rounds": num_rounds,
        "num_light_conditions": num_rounds * num_envs,
        "warmup_steps": args_cli.warmup,
        "env_spacing": args_cli.env_spacing,
        "rgb_only": bool(args_cli.rgb_only),
        "image": {
            "width": CANONICAL_WIDTH,
            "height": CANONICAL_HEIGHT,
            "camera": "camera_fixed",
            "convention": FIXED_CAMERA_CONVENTION,
        },
        "camera_intrinsics": {
            "fx": K33[0][0], "fy": K33[1][1],
            "cx": K33[0][2], "cy": K33[1][2],
            "matrix": K33,
        },
        "camera_extrinsics": {
            "position_env_xyz": [float(v) for v in
                                 (cam_pos_w - env.scene.env_origins[0].cpu().numpy())],
            "quaternion_world_wxyz": [float(v) for v in cam_quat_w],
            "position_base_xyz": [float(v) for v in cam_pos_b[0].cpu().numpy()],
            "quaternion_base_wxyz": [float(v) for v in cam_quat_b[0].cpu().numpy()],
        },
        "pose_frame": "robot_base",
        "pose_layout": "[x, y, z, qw, qx, qy, qz]",
        "init_joint_pos_deg": {k: round(float(np.degrees(v)), 2)
                               for k, v in INIT_JOINT_POS.items()},
        "randomization": {
            "object_x_range": [-0.10, 0.10],
            "object_y_range": [-0.10, 0.10],
            "object_yaw_range_rad": [-1.5707963, 1.5707963],
            "light_intensity": [_LIGHT_MIN, _LIGHT_MAX],
            "light_color_temperature": list(_LIGHT_COLOR_TEMP_RANGE),
            "light_exposure": list(_LIGHT_EXPOSURE_RANGE),
            "light_radius": list(_LIGHT_RADIUS_RANGE),
        },
    }
    with open(os.path.join(args_cli.out, "dataset_info.json"), "w") as f:
        json.dump(info, f, indent=2)

    if preview_tiles:
        cols = 4
        while len(preview_tiles) % cols:
            preview_tiles.append(np.zeros_like(preview_tiles[0]))
        rows = [np.hstack(preview_tiles[r:r + cols])
                for r in range(0, len(preview_tiles), cols)]
        cv2.imwrite(os.path.join(args_cli.out, "preview_grid.png"), np.vstack(rows))

    el = time.time() - t0
    print("=" * 78)
    print(f"完成: {total_frames} 帧, 耗时 {el:.1f}s ({total_frames/el:.1f} 帧/s)")
    print(f"  工件位置(基座系) x {poses_base[:,0].min():+.3f}~{poses_base[:,0].max():+.3f}"
          f"  y {poses_base[:,1].min():+.3f}~{poses_base[:,1].max():+.3f}")
    print(f"  光照 intensity   {light_params[:,0].min():.0f}~{light_params[:,0].max():.0f}")
    print(f"  光照 色温        {light_params[:,1].min():.0f}~{light_params[:,1].max():.0f} K")
    print(f"  唯一光照组合数   {len(np.unique(light_params, axis=0))} / {total_frames}")
    print(f"  输出            {args_cli.out}")
    print("=" * 78)

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
