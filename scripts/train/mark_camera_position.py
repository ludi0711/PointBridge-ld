#!/usr/bin/env python3
"""在 Isaac 视口里用红色小球标出标定固定相机的位置。

配套 ``view_fixed_camera_live.py``：那个脚本给你看"相机拍到什么"，这个给你看
"相机在哪儿"。两个视角合起来才能判断外参对不对 —— 只看相机画面时，一个偏了
30 cm 的相机和一个正确的相机可能都拍得到工件。

标出来的东西：
    红球      相机光心位置
    蓝球      光轴前方 0.3 m 处的一点（红→蓝的连线方向就是相机朝向）
    绿球      光轴与桌面的交点（相机到底在瞄桌面的哪个位置）

**位置取自 ``camera.data.pos_w`` 而不是配置里的 offset 常量。**
这是有意的：``update_latest_camera_pose=True`` 时 ``pos_w`` 是渲染器真正在用的
位姿，若有别的代码写过相机位姿，只有它会反映出来。拿常量画球等于自己骗自己 ——
球永远落在"标称"位置，永远看不出错位。脚本会把两者都打印出来供对照。

用法（需要 GUI，不能加 --headless）::

    ~/IsaacLab/isaaclab.sh -p scripts/train/mark_camera_position.py
    ~/IsaacLab/isaaclab.sh -p scripts/train/mark_camera_position.py --radius 0.05

看什么：相机球应该悬在桌子斜上方、朝工件方向俯视（配置注释说光轴偏垂直 25.1°，
在 1.215 m 处打到桌面 (0.132, -0.359, 0.815)）。绿球若不在桌面上、或红球跑到地
底下/天上，就是外参或坐标系约定错了。
"""

import argparse
import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))
sys.path.insert(0, _PROJECT_DIR)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="在 Isaac 视口里用红球标出固定相机位置")
parser.add_argument("--radius", type=float, default=0.04, help="小球半径(米)，默认 0.04")
parser.add_argument(
    "--axis_len",
    type=float,
    default=0.3,
    help="朝向指示球距光心多远(米)。红球→蓝球的方向即光轴方向。",
)
parser.add_argument(
    "--table_z",
    type=float,
    default=0.815,
    help="桌面高度(米)，用于求光轴与桌面的交点。默认 0.815 与场景一致。",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 相机是被标记的对象，必须实例化；程序化桌面的 PreviewSurface(MDL) 材质也要求
# 渲染型 Kit experience。
args_cli.enable_cameras = True
args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

import isaaclab.sim as sim_utils
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.utils.math import quat_apply, quat_apply_inverse

from configs.xarm7_pick_pointcloud_env_cfg import (
    CALIB_CAMERA_POSITION_IN_BASE_M,
    FIXED_CAMERA_CONVENTION,
    FIXED_CAMERA_POSITION_M,
    FIXED_CAMERA_QUATERNION_WXYZ,
    XArm7PickPointCloudPlayEnvCfg,
)


def _make_markers(radius: float) -> VisualizationMarkers:
    """三色小球。放在 /Visuals 下，不参与物理、不进任何观测。"""
    cfg = VisualizationMarkersCfg(
        prim_path="/Visuals/CameraPoseMarkers",
        markers={
            "camera": sim_utils.SphereCfg(
                radius=radius,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
            ),
            "forward": sim_utils.SphereCfg(
                radius=radius * 0.5,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.3, 1.0)),
            ),
            "hit": sim_utils.SphereCfg(
                radius=radius * 0.6,
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 1.0, 0.2)),
            ),
        },
    )
    return VisualizationMarkers(cfg)


def main() -> None:
    cfg = XArm7PickPointCloudPlayEnvCfg()
    cfg.scene.num_envs = 1
    # 不调 set_observation_stage：stage 1/-1 会把 camera_fixed 摘掉，而它正是本
    # 脚本要标记的对象。
    env = ManagerBasedRLEnv(cfg=cfg)

    camera = env.scene["camera_fixed"]
    robot = env.scene["robot"]
    markers = _make_markers(args_cli.radius)

    env.reset()
    # 先走一步让相机位姿缓冲填充。reset 之后 pos_w 可能还是初值。
    action_dim = env.action_manager.total_action_dim
    zero_action = torch.zeros((env.num_envs, action_dim), device=env.device)
    env.step(zero_action)

    cam_pos_w = camera.data.pos_w[:1]        # (1, 3) 世界系
    cam_quat_w = camera.data.quat_w_ros[:1]  # (1, 4) ROS 光学系

    # ROS 光学系里 +Z 是光轴前方（+X 右、+Y 下）。用 world 约定的四元数配这个向量
    # 会指错方向 —— 这正是配置注释里警告过的坑。
    forward_local = torch.tensor(
        [[0.0, 0.0, 1.0]], device=cam_pos_w.device, dtype=cam_pos_w.dtype
    )
    forward_w = quat_apply(cam_quat_w, forward_local)
    fwd_pos_w = cam_pos_w + forward_w * args_cli.axis_len

    # 光轴与桌面 z=table_z 的交点。方向朝下(dz<0)时才有交点。
    dz = float(forward_w[0, 2])
    if dz < -1e-6:
        t = (args_cli.table_z - float(cam_pos_w[0, 2])) / dz
        hit_w = cam_pos_w + forward_w * t
    else:
        # 光轴没朝下 → 打不到桌面。这本身就是个强信号：相机在拍天上。
        t = float("nan")
        hit_w = fwd_pos_w.clone()

    base_pos_w = robot.data.root_pos_w[:1]
    base_quat_w = robot.data.root_quat_w[:1]
    # 世界系 → 基座系：必须做逆旋转，不能只做减法。基座有 90° 的 yaw
    # (rot=0.707,0,0,-0.707)，只减位置会得到一组"看着像但完全不对"的数字。
    cam_pos_b = quat_apply_inverse(base_quat_w, cam_pos_w - base_pos_w)

    print("=" * 78)
    print("相机位姿对照")
    print(f"  标定值 (基座系)     : {CALIB_CAMERA_POSITION_IN_BASE_M}")
    print(f"  实测值 (基座系)     : {cam_pos_b[0].cpu().numpy()}   ← 应与上一行一致")
    print("-" * 78)
    print(f"  配置 offset (env系) : {FIXED_CAMERA_POSITION_M}")
    print(f"  实际 pos_w (世界系) : {cam_pos_w[0].cpu().numpy()}")
    print(f"  机器人基座 pos_w    : {base_pos_w[0].cpu().numpy()}")
    print(f"  相机离桌面高度      : {float(cam_pos_w[0, 2]) - args_cli.table_z:+.3f} m"
          "   ← 必须为正，负值=相机埋在桌子里")
    print("-" * 78)
    print(f"  光轴方向 (世界系)   : {forward_w[0].cpu().numpy()}")
    print(f"  光轴与水平面夹角    : {float(torch.asin(forward_w[0, 2]).rad2deg()):+.1f} 度"
          "   ← 应约 -27.4 度（负=朝下看桌面）")
    if dz < -1e-6:
        hit_b = quat_apply_inverse(base_quat_w, hit_w - base_pos_w)
        print(f"  光轴打到桌面(世界系): {hit_w[0].cpu().numpy()}  (距离 {t:.3f} m)")
        print(f"  该点在基座系        : {hit_b[0].cpu().numpy()}")
    else:
        print("  [警告] 光轴没有朝下，打不到桌面 —— 相机很可能在拍天上。")
    print("-" * 78)
    print("红球=相机光心   蓝球=光轴前方   绿球=光轴与桌面交点")
    print("在 Isaac 视口里用鼠标转视角查看。关闭窗口或 Ctrl+C 退出。")
    print("=" * 78)

    translations = torch.cat([cam_pos_w, fwd_pos_w, hit_w], dim=0)
    marker_indices = torch.tensor([0, 1, 2], device=translations.device)

    while simulation_app.is_running():
        # 每帧重画：相机位姿 DR 虽已关闭，但若将来加回来，球会自动跟着动。
        markers.visualize(translations=translations, marker_indices=marker_indices)
        env.step(zero_action)

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
