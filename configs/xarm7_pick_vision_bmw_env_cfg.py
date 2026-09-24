# Copyright (c) 2024, Isaac Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""xArm7 Pick 视觉环境 —— 带 BMW 工装台版本

在 xarm7_pick_vision_env_cfg.py 基础上，将 workbench 替换为
assets/bwmgongzhuang/bmwgongzhuang.usdc（静态背景，不参与物理仿真）。
其余场景、观测、奖励、训练配置与原版完全一致。
"""

import math
import os
import random as _random

import torch
import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

from .xarm7_pick_vision_env_cfg import (
    XArm7PickLiftCubeSceneCfg,
    XArm7PickLiftCubeEnvCfg,
    XArm7PickLiftCubePlayEnvCfg,
    XArm7PickLiftCubeEventCfg,
)

_CONFIGS_DIR       = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR       = os.path.dirname(_CONFIGS_DIR)
BMW_GONGZHUANG_USD = os.path.join(_PROJECT_DIR, "assets/bwmgongzhuang/bmwgongzhuang.usdc")


def randomize_gongzhuang(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    x_range: tuple = (-0.10, 0.10),
    y_range: tuple = (-0.10, 0.10),
    yaw_range: tuple = (-math.pi / 2, math.pi / 2),
) -> None:
    import omni.usd
    from pxr import Gf, UsdShade

    gongzhuang = env.scene["workbench"]
    table      = env.scene["workpiece_table"]
    device     = env.device
    n          = len(env_ids)

    # 桌面 root_pos 在质心（half-height=0.4075），顶面 = root_pos_z + 0.4075
    _TABLE_HALF_H = 0.4075
    table_top_z   = table.data.root_pos_w[env_ids, 2] + _TABLE_HALF_H  # 世界坐标顶面 z

    root_state = gongzhuang.data.default_root_state[env_ids].clone()
    root_state[:, 0] += torch.zeros(n, device=device).uniform_(*x_range)
    root_state[:, 1] += torch.zeros(n, device=device).uniform_(*y_range)
    # table_top_z 是世界坐标，转成局部坐标再统一加 env_origin
    root_state[:, 2]  = table_top_z - env.scene.env_origins[env_ids, 2]

    yaw    = torch.zeros(n, device=device).uniform_(*yaw_range)
    zeros  = torch.zeros(n, device=device)
    q_yaw  = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
    root_state[:, 3:7] = math_utils.quat_mul(q_yaw, root_state[:, 3:7])

    root_state[:, :3] += env.scene.env_origins[env_ids]
    root_state[:, 7:]  = 0.0
    gongzhuang.write_root_state_to_sim(root_state, env_ids=env_ids)

    # 颜色随机化
    stage = omni.usd.get_context().get_stage()
    for env_id in env_ids.tolist():
        shader_path = f"/World/envs/env_{env_id}/BmwGongzhuang/root/_materials/Metal005/Principled_BSDF"
        shader_prim = stage.GetPrimAtPath(shader_path)
        if not shader_prim.IsValid():
            continue
        color = Gf.Vec3f(_random.random(), _random.random(), _random.random())
        UsdShade.Shader(shader_prim).GetInput("diffuseColor").Set(color)


@configclass
class XArm7PickLiftCubeBmwEventCfg(XArm7PickLiftCubeEventCfg):
    randomize_gongzhuang = EventTerm(
        func=randomize_gongzhuang, mode="reset"
    )


@configclass
class XArm7PickLiftCubeBmwSceneCfg(XArm7PickLiftCubeSceneCfg):
    """在原有视觉场景基础上加入 BMW 工装台（kinematic rigid body，可动态移动）。"""

    workbench = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/BmwGongzhuang",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.9, 4.0, 0.815],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=BMW_GONGZHUANG_USD,
            scale=(1.0, 1.0, 1.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )


@configclass
class XArm7PickLiftCubeBmwEnvCfg(XArm7PickLiftCubeEnvCfg):
    """带 BMW 工装台的完整训练环境配置。"""

    scene:  XArm7PickLiftCubeBmwSceneCfg = XArm7PickLiftCubeBmwSceneCfg(num_envs=64, env_spacing=2.5)
    events: XArm7PickLiftCubeBmwEventCfg = XArm7PickLiftCubeBmwEventCfg()


@configclass
class XArm7PickLiftCubeBmwPlayEnvCfg(XArm7PickLiftCubeBmwEnvCfg):
    """带 BMW 工装台的单环境测试配置。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs    = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.observations.policy.pointbert_feat_wrist.params["noise_std"] = 0.0
