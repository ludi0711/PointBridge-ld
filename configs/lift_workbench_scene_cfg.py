# Copyright (c) 2024, Isaac Lab Project
# SPDX-License-Identifier: BSD-3-Clause
"""xArm7 Lift Workbench 场景配置

主要可调入口：
  1. 机械臂初始关节角  →  子类 robot.init_state.joint_pos（在 env cfg 中覆盖）
  2. 桌子高度         →  WORKPIECE_TABLE_HEIGHT（本文件顶部常量）
  3. 工件大小         →  OBJECT_SCALE（本文件顶部常量）
"""

import os
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, ArticulationCfg, RigidObjectCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass

# ── 可调参数 ───────────────────────────────────────────────────────────────────

# 桌子高度（cuboid 中心 z 坐标）。
# 桌面顶面 z = WORKPIECE_TABLE_HEIGHT + WORKPIECE_TABLE_HALF_HEIGHT
# 工件初始 z 应略高于桌面顶面（加上工件自身半高）。
WORKPIECE_TABLE_HEIGHT      = 0.4075  # 桌子中心 z（m）= 桌高/2 = 0.815/2
WORKPIECE_TABLE_HALF_HEIGHT = 0.4075  # 桌子半高 = size_z/2（m），底面 z=0 贴地
WORKPIECE_TABLE_TOP_Z       = WORKPIECE_TABLE_HEIGHT + WORKPIECE_TABLE_HALF_HEIGHT  # = 0.815（比机械臂基座 0.8m 高 1.5cm）

# 工件初始放置高度（z）= 桌面顶面(0.815) + 工件半高(scale=1.0时约0.067)
OBJECT_INIT_Z = 0.882

# 工件缩放比例（xyz 均匀缩放）
OBJECT_SCALE  = (1.0, 1.0, 1.0)

# ── 路径 ──────────────────────────────────────────────────────────────────────
_CURRENT_DIR      = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WORKBENCH_USD_PATH  = os.path.join(_CURRENT_DIR, "usd/bmw_workbench_static.usd")
WORKPIECE_USD_PATH  = "/home/gxai/IsaacLab/xarm7/mesh/gongjian/gongjian.usd"


@configclass
class LiftWorkbenchSceneCfg(InteractiveSceneCfg):
    """Lift Workbench 场景配置：桌高、工件大小集中在顶部常量，初始关节角在 env cfg 中覆盖。"""

    # 地面
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.0, 0.0, 0.0]),
        spawn=sim_utils.GroundPlaneCfg(physics_material=None),
    )

    # 工装台（不用可保留，环境配置中将其设为 None）
    workbench = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Workbench",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.5, 0.0, 0.0],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=WORKBENCH_USD_PATH,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    # 机械臂底座支撑圆柱
    robot_support_box = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RobotSupportBox",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.9, 4.6, 0.4],
            rot=[0.707, 0.0, 0.0, -0.707],
        ),
        spawn=sim_utils.CylinderCfg(
            radius=0.1,
            height=0.8,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.7,
                dynamic_friction=0.6,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.3, 0.3, 0.35),
                metallic=0.2,
                roughness=0.7,
            ),
        ),
    )

    # ── 工件桌面 ──────────────────────────────────────────────────────────────
    # 调整桌高：修改顶部 WORKPIECE_TABLE_HEIGHT / WORKPIECE_TABLE_HALF_HEIGHT
    workpiece_table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WorkpieceTable",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.9, 4.0, WORKPIECE_TABLE_HEIGHT],
            rot=[0.707, 0.0, 0.0, -0.707],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, WORKPIECE_TABLE_HALF_HEIGHT * 2),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.7,
                dynamic_friction=0.6,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.4, 0.25, 0.15),
                metallic=0.1,
                roughness=0.8,
            ),
        ),
    )

    # ── 工件 ──────────────────────────────────────────────────────────────────
    # 调整工件大小：修改顶部 OBJECT_SCALE
    # 调整工件初始高度：修改顶部 OBJECT_INIT_Z
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.9, 4.16, OBJECT_INIT_Z],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=WORKPIECE_USD_PATH,
            scale=OBJECT_SCALE,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
                linear_damping=0.5,
                angular_damping=0.5,
                max_linear_velocity=1.0,
                max_angular_velocity=1.0,
                solver_position_iteration_count=16,
                solver_velocity_iteration_count=1,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.5),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    robot: ArticulationCfg = None
    ee_frame: FrameTransformerCfg = None

    light = AssetBaseCfg(
        prim_path="/World/Light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=3000.0,
        ),
    )
