# Copyright (c) 2024, Isaac Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""xArm7 Pick LiftCube 环境配置

- Theia 和 Point-BERT 推理改用 FP16（half precision）
  模型权重/激活值均为 float16，显存减半，Tensor Core 加速
  输出 feature 转回 float32 再送入 MLP，不影响策略网络精度
- 观测维度 782，包含双相机视觉特征、腕部点云特征、关节位置和上一帧动作
"""

import math
import os
import random as _random

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import FrameTransformerCfg, TiledCameraCfg, ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

import isaaclab.envs.mdp as mdp

from . import xarm7_pick_liftcube_mdp as custom_mdp
from .lift_workbench_scene_cfg import LiftWorkbenchSceneCfg
from .pointbert_encoder import (
    get_pointbert_model,
    depth_to_pointcloud,
    sample_to_fixed_points,
)
from .xarm7_pick_pose_env_cfg import (
    KinematicRelativeJointDirectActionCfg,
    lock_robot_to_cached_joint_target_reward,
    lock_robot_to_cached_joint_target_done,
    contact_force_done,
    last_clipped_action,
)

# ── 常量 ──────────────────────────────────────────────────────────────────────
_ARM_ACTION_SCALE = math.radians(0.6)
_ARM_CLIP_RAD     = math.radians(0.6)

_ARM_JOINT_LIMITS_LOW = torch.tensor(
    [math.radians(v) for v in [-180, -118, -180, -11, -97, -180, -180]],
    dtype=torch.float32,
)
_ARM_JOINT_LIMITS_HIGH = torch.tensor(
    [math.radians(v) for v in [180, 118, 180, 225, 97, 180, 180]],
    dtype=torch.float32,
)

_CONFIGS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_CONFIGS_DIR)
_ASSETS_DIR  = os.path.join(_PROJECT_DIR, "assets/xarm7")

GONGJIAN_USD        = os.path.join(_ASSETS_DIR, "gongjian.usd")
ROBOT_USD           = os.path.join(_ASSETS_DIR, "XARM-WITH-GRIP-NEW-FALAN.usd")
THEIA_MODEL_PATH    = "/home/gxai/IsaacLab/theia_tiny"
POINTBERT_CKPT_PATH = "/home/gxai/IsaacLab/Point_BERT/Point-BERT.pth"

# ── D435 对齐相机内参（224×224，FOV≈43.3°）──────────────────────────────────
_FIXED_CAM_FX = 281.6
_FIXED_CAM_FY = 281.6
_FIXED_CAM_CX = 111.5
_FIXED_CAM_CY = 111.5

# 腕部相机内参 —— D405 实测（serial=230322270207）
_WRIST_CAM_FX = 182.3
_WRIST_CAM_FY = 182.1
_WRIST_CAM_CX = 111.1
_WRIST_CAM_CY = 111.3

# D405 腕部相机仿真内参（PinholeCameraCfg）
_D435_FOCAL_LENGTH        = 1.93
_D435_HORIZONTAL_APERTURE = 2.3712

# 点云参数
_POINTCLOUD_NUM_POINTS = 1024
_POINTCLOUD_NOISE_STD  = 0.002

# 相机基础位姿
_CAM_POS_BASE = (0.5000, 4.0200, 1.8000)
_CAM_ROT_BASE = (0.7002, 0.0984, -0.0984, -0.7002)

# 随机化范围
_CAM_POS_RANGE   = (-0.05, 0.05)
_CAM_ROT_RANGE   = math.radians(6.0)
_LIGHT_MIN       = 0.0
_LIGHT_MAX       = 3000.0
_TABLE_HEIGHT_RAND = 0.05
_TABLE_BASE_H    = 0.4075

# 平行夹爪绕自身局部 Z 轴旋转 180° 后只是左右指交换，视为等价夹持姿态。
# 和 pick-pose 奖励 / success 判定保持同一套姿态语义。
_GRASP_SYMMETRY_QUATS_WXYZ = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)

INIT_JOINT_POS = {
    "joint1": math.radians(-0.4),
    "joint2": math.radians(-47.6),
    "joint3": math.radians(0.0),
    "joint4": math.radians(1.8),
    "joint5": math.radians(-0.2),
    "joint6": math.radians(49.4),
    "joint7": math.radians(-0.1),
}

# ── 机器人配置 ─────────────────────────────────────────────────────────────────
XARM7_GRIP_ON_FALAN_CFG = ArticulationCfg(
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
    init_state=ArticulationCfg.InitialStateCfg(joint_pos=INIT_JOINT_POS),
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


# ── 域随机化函数 ────────────────────────────────────────────────────────────────

def reset_table_height_and_object_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    x_range: tuple = (-0.10, 0.10),
    y_range: tuple = (-0.10, 0.10),
    yaw_range: tuple = (-math.pi / 2, math.pi / 2),
    height_range: float = _TABLE_HEIGHT_RAND,
) -> None:
    obj    = env.scene["object"]
    table  = env.scene["workpiece_table"]
    device = env.device
    n      = len(env_ids)

    delta_h = torch.zeros(n, device=device).uniform_(-height_range, height_range)

    table_state = table.data.default_root_state[env_ids].clone()
    table_state[:, 2] += delta_h
    table_state[:, :3] += env.scene.env_origins[env_ids]
    table_state[:, 7:] = 0.0
    table.write_root_pose_to_sim(table_state[:, :7], env_ids=env_ids)

    root_state = obj.data.default_root_state[env_ids].clone()
    root_state[:, 0] += torch.zeros(n, device=device).uniform_(*x_range)
    root_state[:, 1] += torch.zeros(n, device=device).uniform_(*y_range)
    root_state[:, 2] += delta_h

    yaw    = torch.zeros(n, device=device).uniform_(*yaw_range)
    zeros  = torch.zeros(n, device=device)
    q_yaw  = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
    root_state[:, 3:7] = math_utils.quat_mul(q_yaw, root_state[:, 3:7])
    root_state[:, :3] += env.scene.env_origins[env_ids]
    root_state[:, 7:]  = 0.0
    obj.write_root_state_to_sim(root_state, env_ids=env_ids)


def randomize_camera_pose(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    bx, by, bz   = _CAM_POS_BASE
    bw, bi, bj, bk = _CAM_ROT_BASE

    for env_id in env_ids.tolist():
        cam_prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/CameraFixed")
        if not cam_prim.IsValid():
            continue

        px = bx + _random.uniform(*_CAM_POS_RANGE)
        py = by + _random.uniform(*_CAM_POS_RANGE)
        pz = bz + _random.uniform(*_CAM_POS_RANGE)

        dp = _random.uniform(-_CAM_ROT_RANGE, _CAM_ROT_RANGE)
        dy = _random.uniform(-_CAM_ROT_RANGE, _CAM_ROT_RANGE)
        dr = _random.uniform(-_CAM_ROT_RANGE, _CAM_ROT_RANGE)

        qp      = Gf.Quatf(math.cos(dp / 2), math.sin(dp / 2), 0.0, 0.0)
        qy      = Gf.Quatf(math.cos(dy / 2), 0.0, 0.0, math.sin(dy / 2))
        qr      = Gf.Quatf(math.cos(dr / 2), 0.0, math.sin(dr / 2), 0.0)
        q_base  = Gf.Quatf(bw, bi, bj, bk)
        q_final = q_base * qp * qy * qr
        q_final.Normalize()

        xformable = UsdGeom.Xformable(cam_prim)
        ops = {op.GetOpName(): op for op in xformable.GetOrderedXformOps()}
        if "xformOp:translate" in ops:
            ops["xformOp:translate"].Set(Gf.Vec3d(px, py, pz))
        if "xformOp:orient" in ops:
            ops["xformOp:orient"].Set(
                Gf.Quatd(q_final.GetReal(), *q_final.GetImaginary())
            )


def randomize_light_intensity(_env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    import omni.usd
    from pxr import UsdLux

    if len(env_ids) == 0:
        return
    stage = omni.usd.get_context().get_stage()
    light_prim = stage.GetPrimAtPath("/World/Light")
    if not light_prim.IsValid():
        return
    intensity = _random.uniform(_LIGHT_MIN, _LIGHT_MAX)
    UsdLux.LightAPI(light_prim).GetIntensityAttr().Set(float(intensity))


def randomize_table_color(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    import omni.usd
    from pxr import Gf, UsdShade

    stage = omni.usd.get_context().get_stage()
    for env_id in env_ids.tolist():
        shader_path = f"/World/envs/env_{env_id}/WorkpieceTable/geometry/material/Shader"
        shader_prim = stage.GetPrimAtPath(shader_path)
        if not shader_prim.IsValid():
            continue
        color = Gf.Vec3f(_random.random(), _random.random(), _random.random())
        UsdShade.Shader(shader_prim).GetInput("diffuseColor").Set(color)


# ── FP16 Theia 模型单例 ────────────────────────────────────────────────────────

_theia_model_cache: dict = {}
_norm_cache: dict = {}


def _get_theia_model_fp16(model_path: str, device: str):
    key = (model_path, device)
    if key not in _theia_model_cache:
        from transformers import AutoModel
        model = AutoModel.from_pretrained(
            model_path,
            trust_remote_code=True,
            local_files_only=True,
        ).eval().to(device).half()   # FP16
        for p in model.parameters():
            p.requires_grad_(False)
        _theia_model_cache[key] = model
    return _theia_model_cache[key]


def theia_visual_feature(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg = SceneEntityCfg("camera_fixed"),
    model_path: str = THEIA_MODEL_PATH,
) -> torch.Tensor:
    """RGB → Theia-tiny(FP16) → (B, 192) float32。"""
    device = env.device
    model  = _get_theia_model_fp16(model_path, device)
    camera = env.scene[camera_cfg.name]

    rgb = camera.data.output["rgb"]

    if rgb.dtype == torch.uint8:
        x = rgb[..., :3].permute(0, 3, 1, 2).float() / 255.0
    else:
        x = rgb[..., :3].permute(0, 3, 1, 2).float()

    if device not in _norm_cache:
        _norm_cache[device] = (
            torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1),
            torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1),
        )
    mean, std = _norm_cache[device]
    x = (x - mean) / std

    with torch.no_grad():
        out  = model.backbone.model(pixel_values=x.half(), interpolate_pos_encoding=True)
        feat = out.last_hidden_state[:, 1:].mean(dim=1).float()   # (B, 192) fp32
    return feat


# ── FP16 Point-BERT 模型单例 ───────────────────────────────────────────────────

_pointbert_fp16_cache: dict = {}


def _get_pointbert_model_fp16(ckpt_path: str, device: str):
    key = (ckpt_path, device)
    if key not in _pointbert_fp16_cache:
        model = get_pointbert_model(ckpt_path, device).half()   # FP16
        _pointbert_fp16_cache[key] = model
    return _pointbert_fp16_cache[key]


def pointbert_wrist_feature(
    env: ManagerBasedRLEnv,
    camera_cfg:  SceneEntityCfg = SceneEntityCfg("camera_wrist"),
    ckpt_path:   str   = POINTBERT_CKPT_PATH,
    fx:          float = _WRIST_CAM_FX,
    fy:          float = _WRIST_CAM_FY,
    cx:          float = _WRIST_CAM_CX,
    cy:          float = _WRIST_CAM_CY,
    num_points:  int   = _POINTCLOUD_NUM_POINTS,
    noise_std:   float = _POINTCLOUD_NOISE_STD,
) -> torch.Tensor:
    """腕部相机深度图 → 点云 → Point-BERT(FP16) → (B, 384) float32。"""
    device = env.device
    model  = _get_pointbert_model_fp16(ckpt_path, device)

    camera = env.scene[camera_cfg.name]

    depth = camera.data.output.get("distance_to_camera")
    if depth is None:
        raise ValueError(
            "camera_wrist 未输出 distance_to_camera。"
            "请确认 TiledCameraCfg.data_types 包含 'distance_to_camera'。"
        )

    depth_hw = depth[..., 0].float()   # (B, H, W)
    pts_all  = depth_to_pointcloud(depth_hw, fx, fy, cx, cy)   # (B, H*W, 3)
    pts = sample_to_fixed_points(
        pts_all, depth_hw, num_points,
        depth_min=0.01, depth_max=2.0,
    )   # (B, num_points, 3)

    if noise_std > 0.0:
        pts = pts + torch.randn_like(pts) * noise_std

    with torch.no_grad():
        feat = model(pts.half()).float()   # (B, 384) fp32
    return feat


##
# 场景配置
##


@configclass
class XArm7PickLiftCubeSceneCfg(LiftWorkbenchSceneCfg):
    """双相机（腕部+固定）+ grip 碰撞检测 + 全套域随机化场景。"""

    robot = XARM7_GRIP_ON_FALAN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (0.9, 4.6, 0.8)
    robot.init_state.rot = (0.707, 0.0, 0.0, -0.707)

    workbench = None

    workpiece_table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WorkpieceTable",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.9, 4.0, 0.4075],
            rot=[0.707, 0.0, 0.0, -0.707],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 0.815),
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

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.9, 4.16, 0.833],
            rot=[0.0, 1.0, 0.0, 0.0],
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=GONGJIAN_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/link7",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/link7",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.177]),
            ),
        ],
    )

    grip_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/link7",
        history_length=3,
        track_air_time=False,
    )

    camera_wrist = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/link7/CameraWrist",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.11, 0.0, 0.0),
            rot=(1.0, 0.0, 0.0, 0.0),
            convention="ros",
        ),
        data_types=["rgb", "distance_to_camera"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_D435_FOCAL_LENGTH,
            horizontal_aperture=_D435_HORIZONTAL_APERTURE,
            clipping_range=(0.01, 5.0),
        ),
        width=224,
        height=224,
    )

    camera_fixed = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/CameraFixed",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(0.5000, 4.0200, 1.8000),
            rot=(0.7002, 0.0984, -0.0984, -0.7002),
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_D435_FOCAL_LENGTH,
            horizontal_aperture=_D435_HORIZONTAL_APERTURE,
            clipping_range=(0.1, 1.0e5),
        ),
        width=224,
        height=224,
    )


##
# MDP 配置
##


@configclass
class XArm7PickLiftCubeActionsCfg:
    arm_action = KinematicRelativeJointDirectActionCfg(
        asset_name="robot",
        joint_names=["joint[1-7]"],
        scale=_ARM_ACTION_SCALE,
        clip=_ARM_CLIP_RAD,
        joint_limits_low=tuple(_ARM_JOINT_LIMITS_LOW.tolist()),
        joint_limits_high=tuple(_ARM_JOINT_LIMITS_HIGH.tolist()),
    )


def make_reach_success_done_term() -> DoneTerm:
    """Create the optional success termination used by the legacy vision setup."""
    return DoneTerm(
        func=custom_mdp.ee_reached_object,
        params={
            "threshold":           0.01,
            "angle_threshold_deg": 3.0,
            "object_cfg":          SceneEntityCfg("object"),
            "ee_frame_cfg":        SceneEntityCfg("ee_frame"),
            "q_offset":            (0.0, 0.0, 0.0, 1.0),
            "grasp_symmetry_quats": _GRASP_SYMMETRY_QUATS_WXYZ,
        },
    )


def set_success_termination_enabled(env_cfg: ManagerBasedRLEnvCfg, enabled: bool) -> None:
    """Toggle success termination without changing the vision reward weights."""
    env_cfg.terminations.reach_success = make_reach_success_done_term() if enabled else None


@configclass
class XArm7PickLiftCubeObservationsCfg:
    """782 维观测配置（FP16 编码器）
    theia_fixed(192) + theia_wrist(192) + pointbert_wrist(384) + joint_pos(7) + last_action(7)
    """

    @configclass
    class PolicyCfg(ObsGroup):

        theia_feat_fixed = ObsTerm(
            func=theia_visual_feature,
            params={
                "camera_cfg": SceneEntityCfg("camera_fixed"),
                "model_path": THEIA_MODEL_PATH,
            },
        )

        theia_feat_wrist = ObsTerm(
            func=theia_visual_feature,
            params={
                "camera_cfg": SceneEntityCfg("camera_wrist"),
                "model_path": THEIA_MODEL_PATH,
            },
        )

        pointbert_feat_wrist = ObsTerm(
            func=pointbert_wrist_feature,
            params={
                "camera_cfg": SceneEntityCfg("camera_wrist"),
                "ckpt_path":  POINTBERT_CKPT_PATH,
                "fx":         _WRIST_CAM_FX,
                "fy":         _WRIST_CAM_FY,
                "cx":         _WRIST_CAM_CX,
                "cy":         _WRIST_CAM_CY,
                "num_points": _POINTCLOUD_NUM_POINTS,
                "noise_std":  _POINTCLOUD_NOISE_STD,
            },
        )

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint[1-7]"])},
        )

        actions = ObsTerm(func=last_clipped_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class XArm7PickLiftCubeEventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_table_and_object = EventTerm(
        func=reset_table_height_and_object_pose,
        mode="reset",
        params={
            "x_range":      (-0.10, 0.10),
            "y_range":      (-0.10, 0.10),
            "yaw_range":    (-math.pi / 2, math.pi / 2),
            "height_range": _TABLE_HEIGHT_RAND,
        },
    )

    randomize_table_color     = EventTerm(func=randomize_table_color,     mode="reset")
    randomize_camera_pose     = EventTerm(func=randomize_camera_pose,     mode="reset")
    randomize_light_intensity = EventTerm(func=randomize_light_intensity, mode="reset")


@configclass
class XArm7PickLiftCubeRewardsCfg:
    """v88 同步奖励配置。

    训练目标：
      1. 不改变训练动作链路，不加入动作平滑；
      2. 强化 2~4 cm 区间内的位置精修梯度；
      3. success 判定阈值为 1 cm / 3 deg；success termination 默认关闭，可按需开启；
      4. action_rate 惩罚减弱，避免策略在目标附近不敢修正。
    """

    lock_joint_step_end = RewTerm(
        func=lock_robot_to_cached_joint_target_reward,
        weight=1.0,
    )

    reaching_coarse = RewTerm(
        func=custom_mdp.object_ee_distance,
        params={
            "std":          0.50,
            "object_cfg":   SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=35.0,
    )

    reaching_mid = RewTerm(
        func=custom_mdp.object_ee_distance,
        params={
            "std":          0.15,
            "object_cfg":   SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=55.0,
    )

    reaching_fine = RewTerm(
        func=custom_mdp.object_ee_distance,
        params={
            "std":          0.04,
            "object_cfg":   SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=80.0,
    )

    ee_orientation_coarse = RewTerm(
        func=custom_mdp.ee_orientation_alignment,
        params={
            "std":          0.50,
            "q_offset":     (0.0, 0.0, 0.0, 1.0),
            "grasp_symmetry_quats": _GRASP_SYMMETRY_QUATS_WXYZ,
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg":   SceneEntityCfg("object"),
        },
        weight=30.0,
    )

    ee_orientation_mid = RewTerm(
        func=custom_mdp.ee_orientation_alignment,
        params={
            "std":          0.15,
            "q_offset":     (0.0, 0.0, 0.0, 1.0),
            "grasp_symmetry_quats": _GRASP_SYMMETRY_QUATS_WXYZ,
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg":   SceneEntityCfg("object"),
        },
        weight=40.0,
    )

    ee_orientation_fine = RewTerm(
        func=custom_mdp.ee_orientation_alignment,
        params={
            "std":          0.08,
            "q_offset":     (0.0, 0.0, 0.0, 1.0),
            "grasp_symmetry_quats": _GRASP_SYMMETRY_QUATS_WXYZ,
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg":   SceneEntityCfg("object"),
        },
        weight=70.0,
    )

    reach_success_bonus = RewTerm(
        func=custom_mdp.ee_reached_object,
        params={
            "threshold":           0.01,
            "angle_threshold_deg": 3.0,
            "object_cfg":          SceneEntityCfg("object"),
            "ee_frame_cfg":        SceneEntityCfg("ee_frame"),
            "q_offset":            (0.0, 0.0, 0.0, 1.0),
            "grasp_symmetry_quats": _GRASP_SYMMETRY_QUATS_WXYZ,
        },
        weight=0.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-10)

    collision_penalty = RewTerm(
        func=custom_mdp.contact_force_penalty,
        params={"sensor_name": "grip_contact", "threshold": 0.5},
        weight=-1000.0,
    )


@configclass
class XArm7PickLiftCubeTerminationsCfg:
    lock_joint_step_end = DoneTerm(func=lock_robot_to_cached_joint_target_done)

    collision = DoneTerm(
        func=contact_force_done,
        params={"sensor_name": "grip_contact", "threshold": 0.5},
    )

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # Default matches pick-pose no-success training: run until timeout.
    # Use set_success_termination_enabled(..., True) to restore early success done.
    reach_success = None


@configclass
class XArm7PickLiftCubeCurriculumCfg:
    pass


##
# 环境配置
##


@configclass
class XArm7PickLiftCubeEnvCfg(ManagerBasedRLEnvCfg):
    """xArm7 Pick LiftCube 环境：FP16 编码器，782 维观测，两段式奖励，全套 DR。"""

    scene:        XArm7PickLiftCubeSceneCfg  = XArm7PickLiftCubeSceneCfg(num_envs=64, env_spacing=2.5)
    observations: XArm7PickLiftCubeObservationsCfg   = XArm7PickLiftCubeObservationsCfg()
    actions:      XArm7PickLiftCubeActionsCfg        = XArm7PickLiftCubeActionsCfg()
    events:       XArm7PickLiftCubeEventCfg          = XArm7PickLiftCubeEventCfg()
    rewards:      XArm7PickLiftCubeRewardsCfg        = XArm7PickLiftCubeRewardsCfg()
    terminations: XArm7PickLiftCubeTerminationsCfg   = XArm7PickLiftCubeTerminationsCfg()
    curriculum:   XArm7PickLiftCubeCurriculumCfg     = XArm7PickLiftCubeCurriculumCfg()

    def __post_init__(self):
        self.decimation        = 2
        self.episode_length_s  = 24.0

        self.sim.dt              = 0.01
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity               = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 8
        self.sim.physx.gpu_total_aggregate_pairs_capacity      = 1024 * 1024 * 4
        self.sim.physx.friction_correlation_distance           = 0.00625
        self.sim.physx.gpu_max_rigid_patch_count               = 1024 * 1024 * 4
        self.sim.physx.gpu_max_rigid_patch_count               = 1024 * 1024


@configclass
class XArm7PickLiftCubePlayEnvCfg(XArm7PickLiftCubeEnvCfg):
    """测试配置：单环境，关闭训练噪声。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs    = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        self.observations.policy.pointbert_feat_wrist.params["noise_std"] = 0.0
