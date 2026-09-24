#!/usr/bin/env python3
"""直接显示 Isaac Sim 相机 RGB 画面，无点云、无深度、无掩码。

用法::

    conda activate isaaclab
    cd /home/gxai/Desktop/CZR/gx-VA-isaaclab
    python scripts/deploy/view_camera_rgb.py
"""

import argparse
import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))
sys.path.insert(0, _PROJECT_DIR)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Isaac Sim 相机 RGB 实时查看器")
parser.add_argument("--num_envs", type=int, default=1, help="环境数量（固定为1）")
parser.add_argument(
    "--save",
    type=str,
    default=None,
    help="把第一帧 RGB 存成 PNG 后退出（无显示器时用来自检）",
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
args_cli.num_envs = 1

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

import isaaclab.sim as sim_utils
from isaaclab.assets import (
    Articulation,
    ArticulationCfg,
    AssetBaseCfg,
    RigidObject,
    RigidObjectCfg,
)
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors.camera import Camera, TiledCameraCfg
from isaaclab.sim import SimulationContext, SimulationCfg
from isaaclab.utils import configclass

# 从环境配置导入相机参数
from configs.xarm7_pick_pointcloud_env_cfg import (
    CANONICAL_WIDTH,
    CANONICAL_HEIGHT,
    FIXED_CAMERA_POSITION_M,
    FIXED_CAMERA_QUATERNION_WXYZ,
    FIXED_CAMERA_CONVENTION,
    FIXED_CAMERA_CLIPPING_RANGE_M,
    ROBOT_BASE_POSITION_W,
    ROBOT_BASE_QUATERNION_W_WXYZ,
    canonical_intrinsic_matrix,
)

# 机器人 USD（项目自带资产，与训练环境同一份）
ROBOT_USD = os.path.join(_PROJECT_DIR, "assets/xarm7", "XARM-WITH-GRIP-NEW-FALAN.usd")

# 初始关节位置（与训练环境一致）
INIT_JOINT_POS = {
    "joint1": math.radians(-0.4),
    "joint2": math.radians(-47.6),
    "joint3": math.radians(0.0),
    "joint4": math.radians(1.8),
    "joint5": math.radians(30.5),
    "joint6": math.radians(0.0),
    "joint7": math.radians(0.0),
}


@configclass
class CameraViewSceneCfg(InteractiveSceneCfg):
    """最小场景：机器人 + 物体 + 相机。"""

    # 地面
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    # 机器人（xArm7）
    robot: ArticulationCfg = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=ROBOT_USD,
            activate_contact_sensors=True,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                max_depenetration_velocity=5.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=False,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=8,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            # 基座位姿必须与标定外参的换算基准一致，否则相机会拍到空地
            pos=ROBOT_BASE_POSITION_W,
            rot=ROBOT_BASE_QUATERNION_W_WXYZ,
            joint_pos=INIT_JOINT_POS,
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["joint[1-7]"],
                effort_limit=200.0,
                velocity_limit=100.0,
                stiffness=800.0,
                damping=80.0,
            ),
        },
    )

    # 桌子（与 camfix 场景一致，顶面 z=0.815）
    workpiece_table: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WorkpieceTable",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, -0.52, 0.4075),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
        spawn=sim_utils.MeshCuboidCfg(
            size=(1.2, 1.2, 0.815),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.4, 0.25, 0.15),
                metallic=0.1,
                roughness=0.8,
            ),
        ),
    )

    # 物体（简单立方体，摆在桌面上、相机视野内）
    object: RigidObjectCfg = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        spawn=sim_utils.CuboidCfg(
            size=(0.06, 0.06, 0.03),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.05),
            collision_props=sim_utils.CollisionPropertiesCfg(),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.8, 0.2, 0.2)),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.0, -0.44, 0.833),
            rot=(1.0, 0.0, 0.0, 0.0),
        ),
    )

    # 相机（只采集 RGB）
    # prim_path 必须挂在 env 根下，与训练环境一致 —— FIXED_CAMERA_POSITION_M
    # 存的是**换算到 env 根坐标系**后的值，挂到机器人 link 下会落在错误位置。
    camera: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/CameraFixed",
        update_period=0.0,
        offset=TiledCameraCfg.OffsetCfg(
            pos=FIXED_CAMERA_POSITION_M,
            rot=FIXED_CAMERA_QUATERNION_WXYZ,
            convention=FIXED_CAMERA_CONVENTION,
        ),
        data_types=["rgb"],
        # 内参从标定矩阵推导，不要手填 focal_length / aperture
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=canonical_intrinsic_matrix(CANONICAL_WIDTH, CANONICAL_HEIGHT),
            width=CANONICAL_WIDTH,
            height=CANONICAL_HEIGHT,
            clipping_range=FIXED_CAMERA_CLIPPING_RANGE_M,
        ),
        width=CANONICAL_WIDTH,
        height=CANONICAL_HEIGHT,
    )

    # 光源
    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            intensity=2000.0,
            color=(0.75, 0.75, 0.75),
        ),
    )


def main():
    # 先建仿真上下文（InteractiveScene 构造时要读 sim.device），再建场景
    sim = SimulationContext(SimulationCfg(dt=1.0 / 60.0, device="cuda:0"))
    sim.set_camera_view(eye=(2.0, 2.0, 2.0), target=(0.0, 0.0, 0.5))

    scene_cfg = CameraViewSceneCfg(num_envs=1, env_spacing=2.0)
    scene = InteractiveScene(scene_cfg)

    # 播放时间线，之后 sim.step() 才会真正渲染
    sim.reset()

    print("=" * 78)
    print("[Camera RGB Viewer] 相机 RGB 实时查看器")
    print(f"[分辨率]  {CANONICAL_WIDTH}x{CANONICAL_HEIGHT}")
    print(f"[位置]    {FIXED_CAMERA_POSITION_M}")
    print(f"[姿态]    {FIXED_CAMERA_QUATERNION_WXYZ} ({FIXED_CAMERA_CONVENTION})")
    print("=" * 78)

    # Matplotlib 交互窗口（--save 模式不开窗）
    fig = ax = None
    if args_cli.save is None:
        fig, ax = plt.subplots(figsize=(10, 7.5))
        ax.axis("off")
        ax.set_title("Isaac Sim Camera RGB (按 Ctrl+C 退出)", fontsize=12)
        plt.ion()
        plt.show(block=False)

    im = None
    step = 0

    # Reset 场景
    scene.reset()
    scene.write_data_to_sim()
    sim.step()
    scene.update(sim.get_physics_dt())

    def _grab_rgb():
        """取一帧 RGB，统一成 (H, W, 3)。"""
        rgb = scene["camera"].data.output["rgb"][0].cpu().numpy()
        if rgb.shape[-1] == 4:
            rgb = rgb[..., :3]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0.0, 1.0)
        return rgb

    # --save：渲染管线要几帧才稳定（首帧常是全黑），预热后存图退出
    if args_cli.save is not None:
        for _ in range(30):
            scene.write_data_to_sim()
            sim.step()
            scene.update(sim.get_physics_dt())
        rgb = _grab_rgb()
        import matplotlib.image as mpimg

        mpimg.imsave(args_cli.save, rgb)
        print(f"[保存] {args_cli.save}  shape={rgb.shape} dtype={rgb.dtype} "
              f"range=[{rgb.min()}, {rgb.max()}] mean={rgb.mean():.4f}")
        return

    try:
        while simulation_app.is_running():
            # 零动作，让机器人静止
            scene.robot.write_joint_state_to_sim(
                scene.robot.data.default_joint_pos,
                scene.robot.data.default_joint_vel,
            )
            scene.write_data_to_sim()

            # Step 仿真
            sim.step()

            # 更新场景缓冲（含相机）
            scene.update(sim.get_physics_dt())

            # 读取 RGB
            rgb = _grab_rgb()

            # 显示
            if im is None:
                im = ax.imshow(rgb)
            else:
                im.set_data(rgb)

            ax.set_title(
                f"Isaac Sim Camera RGB (step {step})\n"
                f"分辨率: {rgb.shape[1]}x{rgb.shape[0]}  范围: [{rgb.min():.2f}, {rgb.max():.2f}]",
                fontsize=11,
            )
            fig.canvas.draw_idle()
            plt.pause(0.001)

            step += 1

    except KeyboardInterrupt:
        print("\n[退出] 用户中断")

    plt.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
