# Copyright (c) 2024, Isaac Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""xArm7 Pick LiftCube 环境配置（无点云，camfix：全局相机内参修正版）

- camfix：全局相机(D435)改用 aperture=1.5303(fx≈282.5, HFOV≈43.3°)，对齐真实 D435；
  旧 spatial 误用 D405 的 2.3712(FOV 63°)。腕部相机不变。

- Theia 推理改用 FP16（half precision）
  模型权重/激活值均为 float16，显存减半，Tensor Core 加速
  输出 feature 转回 float32 再送入 MLP，不影响策略网络精度
- 观测维度 1166，包含双相机视觉特征（mean+spatial-softmax 池化），关节位置和上一帧动作（无点云）
"""

import math
import os
import random as _random

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
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

GONGJIAN_USD     = os.path.join(_ASSETS_DIR, "gongjian.usd")
ROBOT_USD        = os.path.join(_ASSETS_DIR, "XARM-WITH-GRIP-NEW-FALAN.usd")
THEIA_MODEL_PATH = os.path.join(_PROJECT_DIR, "model/theia_tiny")

# 桌面纹理库（domain randomization）：reset 时 70% 概率从中挑一张贴图、30% 纯色随机。
_TEXTURES_DIR   = os.path.join(_PROJECT_DIR, "data/textures")
_TABLE_TEXTURES = sorted(
    os.path.join(_TEXTURES_DIR, f)
    for f in (os.listdir(_TEXTURES_DIR) if os.path.isdir(_TEXTURES_DIR) else [])
    if f.lower().endswith((".png", ".jpg", ".jpeg"))
)
_TABLE_TEXTURE_PROB   = 0.70                  # 贴纹理的概率（其余走纯色）
_TABLE_TEX_SCALE_RANGE = (0.5, 4.0)           # UV 重复次数（纹理大小随机）
_TABLE_TEX_ROT_RANGE   = (0.0, 360.0)         # UV 旋转角度（度）
# 每张纹理预建一个静态材质（GPU 资源在首次 reset 时一次性建好；之后 reset 只切绑定+改 UV，
# 不再运行时新建/重连 shader 节点，避免触发 RTX/Vulkan GPU pagefault crash）。
_table_material_pool: dict = {}               # { env_id: {...} }

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

# 腕部相机 D405 仿真内参（PinholeCameraCfg）
# fx = focal/aperture*224 = 182.3, HFOV≈63.1°，与 D405 实测一致。
_WRIST_FOCAL_LENGTH        = 1.93
_WRIST_HORIZONTAL_APERTURE = 2.3712

# 全局相机 D435 仿真内参（camfix 修正）
# 真实 D435 (224 center-crop): fx≈282.5, HFOV≈43.3°。
# 旧 spatial 误用了 D405 的 aperture(2.3712 → FOV 63°，过宽)。
# 此处 aperture=1.5303 使 fx=282.5、HFOV=43.25°，对齐真实 D435。
_FIXED_FOCAL_LENGTH        = 1.93
_FIXED_HORIZONTAL_APERTURE = 1.5303

# 兼容旧名（部分代码可能引用）。指向腕部值，但下方相机 spawn 已改用各自独立常量。
_D435_FOCAL_LENGTH        = _WRIST_FOCAL_LENGTH
_D435_HORIZONTAL_APERTURE = _WRIST_HORIZONTAL_APERTURE

# 相机基础位姿（机械臂基座平移到 xy=(0,0) 后整体平移 Δ=(-0.9, -4.6)）
_CAM_POS_BASE = (-0.4000, -0.5800, 1.8000)
_CAM_ROT_BASE = (0.7002, 0.0984, -0.0984, -0.7002)

# 随机化范围
_CAM_POS_RANGE   = (-0.05, 0.05)
_CAM_ROT_RANGE   = math.radians(6.0)
_LIGHT_MIN       = 1000.0
_LIGHT_MAX       = 50000.0
# 球形点光源随机化范围（每次 reset 重采样）
_LIGHT_COLOR_TEMP_RANGE = (2500.0, 9000.0)   # 色温 K
_LIGHT_EXPOSURE_RANGE   = (-1.0, 2.0)        # 曝光（2 的幂次乘到 intensity）
_LIGHT_RADIUS_RANGE     = (0.095, 0.2)       # 球半径 m
# 相机图像增强（送入 Theia 前，对观测生效，作为域随机化提升 sim2real 鲁棒性）
# 每轮（reset）采样一次，整个 episode 内固定；下次 reset 再变化。两个相机各自独立采样。
_IMG_AUG_ENABLE        = True
_IMG_BLUR_SIGMA_RANGE  = (0.5, 2.5)          # 高斯模糊 sigma（像素）
_IMG_BLUR_KERNEL       = 9                    # 高斯核大小（奇数）
_IMG_NOISE_STD_RANGE   = (0.05, 0.20)        # 高斯噪声 std（作用在 [0,1] 图像上）
_TABLE_HEIGHT_RAND = 0.00
_TABLE_BASE_H    = 0.4075

# 平行夹爪绕自身局部 Z 轴旋转 180° 后只是左右指交换，视为等价夹持姿态。
# 和 pick-pose 奖励 / success 判定保持同一套姿态语义。
_GRASP_SYMMETRY_QUATS_WXYZ = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)

INIT_JOINT_POS = {
    "joint1": math.radians(-0.4),
    "joint2": math.radians(-58.4),
    "joint3": math.radians(-0.1),
    "joint4": math.radians(17.5),
    "joint5": math.radians(-0.2),
    "joint6": math.radians(75.8),
    "joint7": math.radians(-0.2),
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
    """随机化球形点光源：色温 / 强度 / 曝光 / 半径（每次 reset 重采样）。

    逐 env 独立采样 —— 光源现在是 ``{ENV_REGEX_NS}/Light``，每个 env 一盏。
    并行时 N 个 env 会拿到 N 组不同的光照条件，而不是共用一组。

    采样值回写到 ``_env._light_params``（shape (num_envs, 4)，列依次为
    intensity / color_temperature / exposure / radius），供数据采集脚本记录。
    没有这个，事后就无法按光照条件给数据分组。
    """
    import omni.usd
    from pxr import UsdLux

    if len(env_ids) == 0:
        return
    stage = omni.usd.get_context().get_stage()

    # 惰性建表：这里不用 torch 是因为要写进 json，numpy/python float 更省事
    if not hasattr(_env, "_light_params"):
        _env._light_params = [[0.0] * 4 for _ in range(_env.num_envs)]

    for env_id in env_ids.tolist():
        light_prim = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/Light")
        if not light_prim.IsValid():
            continue

        light_api  = UsdLux.LightAPI(light_prim)
        sphere_api = UsdLux.SphereLight(light_prim)

        # 强度
        intensity = _random.uniform(_LIGHT_MIN, _LIGHT_MAX)
        light_api.GetIntensityAttr().Set(float(intensity))

        # 色温（需先启用 enableColorTemperature，spawn 已开启；此处再确保为 True）
        color_temp = _random.uniform(*_LIGHT_COLOR_TEMP_RANGE)
        light_api.GetEnableColorTemperatureAttr().Set(True)
        light_api.GetColorTemperatureAttr().Set(float(color_temp))

        # 曝光
        exposure = _random.uniform(*_LIGHT_EXPOSURE_RANGE)
        light_api.GetExposureAttr().Set(float(exposure))

        # 半径
        radius = _random.uniform(*_LIGHT_RADIUS_RANGE)
        sphere_api.GetRadiusAttr().Set(float(radius))

        _env._light_params[env_id] = [
            float(intensity), float(color_temp), float(exposure), float(radius)
        ]


def _find_table_mesh(stage, env_id: int):
    """返回桌子 mesh 的 UsdGeom.Mesh（找不到返回 None）。"""
    from pxr import UsdGeom
    p = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/WorkpieceTable/geometry/mesh")
    if p and p.IsValid() and p.IsA(UsdGeom.Mesh):
        return UsdGeom.Mesh(p)
    tbl = stage.GetPrimAtPath(f"/World/envs/env_{env_id}/WorkpieceTable")
    for q in (tbl.GetAllChildren() if tbl and tbl.IsValid() else []):
        for r in [q] + list(q.GetAllChildren()):
            if r.IsA(UsdGeom.Mesh):
                return UsdGeom.Mesh(r)
    return None


def _ensure_table_uv(stage, env_id: int) -> None:
    """给桌子 mesh author 一个 faceVarying 的 primvars:st（顶面 x/y 平面投影 UV）。

    MeshCuboidCfg 生成的 mesh 不带 UV；无 UV 则 UsdUVTexture 采样退化、纹理贴不上。
    这里按顶点 (x,y) 归一化到 [0,1] 作为 UV，UsdTransform2d 再叠加 scale/rotation。
    """
    from pxr import UsdGeom, Sdf, Vt

    mesh = _find_table_mesh(stage, env_id)
    if mesh is None:
        return
    api = UsdGeom.PrimvarsAPI(mesh.GetPrim())
    if api.HasPrimvar("st"):
        return  # 已有 UV，不重复

    points = mesh.GetPointsAttr().Get()
    fvi    = mesh.GetFaceVertexIndicesAttr().Get()
    if not points or not fvi:
        return

    xs = [p[0] for p in points]
    ys = [p[1] for p in points]
    minx, maxx = min(xs), max(xs)
    miny, maxy = min(ys), max(ys)
    spanx = (maxx - minx) or 1.0
    spany = (maxy - miny) or 1.0

    # faceVarying：每个 faceVertex 一个 UV，按其顶点的 (x,y) 投影。
    uvs = [
        ((points[i][0] - minx) / spanx, (points[i][1] - miny) / spany)
        for i in fvi
    ]
    st = api.CreatePrimvar("st", Sdf.ValueTypeNames.TexCoord2fArray,
                           interpolation=UsdGeom.Tokens.faceVarying)
    st.Set(Vt.Vec2fArray([(float(u), float(v)) for (u, v) in uvs]))


def _ensure_table_material_pool(stage, env_id: int) -> dict:
    """为某 env 一次性预建桌面材质池：N 个纹理材质 + 1 个纯色材质。

    每个材质静态接好完整网络（Shader←UVTexture←Transform2d←PrimvarReader），纹理 file
    在建池时就写死。reset 时不再新建任何 shader，只切换桌子 mesh 的材质绑定 + 改 UV
    transform，避免运行时改材质网络触发 GPU pagefault。返回该 env 的池字典（已缓存）。
    """
    from pxr import Gf, Sdf, UsdShade

    if env_id in _table_material_pool:
        return _table_material_pool[env_id]

    # 桌子 mesh 补 UV（MeshCuboid 默认无 st primvar，否则纹理贴不上）
    _ensure_table_uv(stage, env_id)

    pool_root = f"/World/envs/env_{env_id}/TableMatPool"

    def _make_preview_shader(mat_path):
        mat = UsdShade.Material.Define(stage, mat_path)
        shd = UsdShade.Shader.Define(stage, mat_path + "/Shader")
        shd.CreateIdAttr("UsdPreviewSurface")
        shd.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        shd.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.1)
        mat.CreateSurfaceOutput().ConnectToSource(shd.ConnectableAPI(), "surface")
        return mat, shd

    # ── 纯色材质（reset 时改其 diffuseColor 常量）──
    color_mat, color_shd = _make_preview_shader(f"{pool_root}/ColorMat")
    color_shd.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(0.5, 0.5, 0.5))

    # ── N 个纹理材质（每张 png 一个；reset 时只改 UVTransform 的 scale/rotation）──
    tex_mats = []
    for i, tex_file in enumerate(_TABLE_TEXTURES):
        mp = f"{pool_root}/TexMat_{i:03d}"
        mat, shd = _make_preview_shader(mp)

        reader = UsdShade.Shader.Define(stage, mp + "/STReader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        xform = UsdShade.Shader.Define(stage, mp + "/UVTransform")
        xform.CreateIdAttr("UsdTransform2d")
        xform.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(reader.GetOutput("result"))
        xform.CreateInput("rotation", Sdf.ValueTypeNames.Float).Set(0.0)
        xform.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(1.0, 1.0))
        xform.CreateInput("translation", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(0.0, 0.0))
        xform.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        tex = UsdShade.Shader.Define(stage, mp + "/DiffuseTex")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(Sdf.AssetPath(tex_file))
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(xform.GetOutput("result"))
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

        shd.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.GetOutput("rgb"))
        tex_mats.append({"mat_path": mp, "xform_path": mp + "/UVTransform"})

    pool = {
        "color_mat":   {"mat_path": f"{pool_root}/ColorMat", "shader_path": f"{pool_root}/ColorMat/Shader"},
        "tex_mats":    tex_mats,
        "mesh_path":   f"/World/envs/env_{env_id}/WorkpieceTable/geometry/mesh",
    }
    _table_material_pool[env_id] = pool
    return pool


def randomize_table_color(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """桌面域随机化：30% 纯色随机，70% 从预建纹理材质池里选一张（大小/旋转随机）。

    所有材质在首次调用时一次性预建好（_ensure_table_material_pool）；此处每轮只：
      - 纯色：改 ColorMat 的 diffuseColor 常量，绑定 ColorMat
      - 纹理：随机选一个 TexMat，改其 UVTransform 的 scale/rotation，绑定该 TexMat
    不再运行时新建 shader 节点 → 避免 GPU pagefault。
    """
    import omni.usd
    from pxr import Gf, UsdShade

    stage = omni.usd.get_context().get_stage()
    for env_id in env_ids.tolist():
        pool = _ensure_table_material_pool(stage, env_id)

        mesh = _find_table_mesh(stage, env_id)
        if mesh is None:
            continue
        mesh_prim = mesh.GetPrim()

        use_texture = (len(pool["tex_mats"]) > 0) and (_random.random() < _TABLE_TEXTURE_PROB)

        if not use_texture:
            color_shd = UsdShade.Shader(stage.GetPrimAtPath(pool["color_mat"]["shader_path"]))
            color_shd.GetInput("diffuseColor").Set(
                Gf.Vec3f(_random.random(), _random.random(), _random.random())
            )
            target = UsdShade.Material(stage.GetPrimAtPath(pool["color_mat"]["mat_path"]))
        else:
            entry = _random.choice(pool["tex_mats"])
            xform = UsdShade.Shader(stage.GetPrimAtPath(entry["xform_path"]))
            scale = _random.uniform(*_TABLE_TEX_SCALE_RANGE)
            rot   = _random.uniform(*_TABLE_TEX_ROT_RANGE)
            xform.GetInput("scale").Set(Gf.Vec2f(scale, scale))
            xform.GetInput("rotation").Set(float(rot))
            target = UsdShade.Material(stage.GetPrimAtPath(entry["mat_path"]))

        UsdShade.MaterialBindingAPI(mesh_prim).Bind(target)


# ── FP16 Theia 模型单例 ────────────────────────────────────────────────────────

_theia_model_cache: dict = {}
_norm_cache: dict = {}

# 当前轮的图像增强参数（每相机一组）：{ camera_name: {"sigma": float, "noise_std": float} }
# 由 randomize_image_aug（mode="reset"）在每次 reset 时重采样；episode 内固定。
_img_aug_params: dict = {}


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


def _spatial_softmax(patch_tokens: torch.Tensor) -> torch.Tensor:
    """对 patch token 做 spatial-softmax，提取每通道的空间期望坐标。

    输入: (B, N, C)，N=196 个 patch（14×14 网格），C=192 通道。
    输出: (B, 2*C)，每通道 (x, y) 归一化期望坐标 ∈ [-1, 1]，沿通道拼接。
    无可训练参数，保持 frozen。
    """
    B, N, C = patch_tokens.shape
    H = W = int(round(N ** 0.5))                       # 196 -> 14×14
    # (B, N, C) -> (B, C, H, W)
    fmap = patch_tokens.permute(0, 2, 1).reshape(B, C, H, W)

    # 每个 (通道) 的 H*W 特征图做 softmax 得到空间概率分布
    attn = torch.softmax(fmap.reshape(B, C, H * W), dim=-1)  # (B, C, H*W)

    # 归一化网格坐标 ∈ [-1, 1]
    ys = torch.linspace(-1.0, 1.0, H, device=patch_tokens.device)
    xs = torch.linspace(-1.0, 1.0, W, device=patch_tokens.device)
    grid_y, grid_x = torch.meshgrid(ys, xs, indexing="ij")
    grid_x = grid_x.reshape(1, 1, H * W)
    grid_y = grid_y.reshape(1, 1, H * W)

    exp_x = (attn * grid_x).sum(dim=-1)                # (B, C)
    exp_y = (attn * grid_y).sum(dim=-1)                # (B, C)
    return torch.cat([exp_x, exp_y], dim=-1)           # (B, 2*C)


def _gaussian_kernel1d(sigma: float, ksize: int, device) -> torch.Tensor:
    """生成 1D 高斯核（已归一化），shape (ksize,)。sigma<=0 时退化为冲激（不模糊）。"""
    half = (ksize - 1) / 2.0
    xs = torch.arange(ksize, device=device, dtype=torch.float32) - half
    if sigma <= 1e-6:
        k = torch.zeros(ksize, device=device, dtype=torch.float32)
        k[ksize // 2] = 1.0
        return k
    k = torch.exp(-(xs ** 2) / (2.0 * sigma * sigma))
    return k / k.sum()


def _sample_img_aug_params(key: str) -> dict:
    """为某相机采样一组增强参数并写入缓存。返回该组参数。"""
    p = {
        "sigma":     _random.uniform(*_IMG_BLUR_SIGMA_RANGE),
        "noise_std": _random.uniform(*_IMG_NOISE_STD_RANGE),
    }
    _img_aug_params[key] = p
    return p


def _augment_image(x: torch.Tensor, key: str = "default") -> torch.Tensor:
    """对 (B, 3, H, W)、值域 [0,1] 的图像做高斯模糊 + 高斯噪声。

    增强强度（blur sigma / noise std）从 _img_aug_params[key] 读取，由
    randomize_image_aug 在每次 reset 时重采样 —— 即「每轮一致，reset 才变」。
    若缓存缺失（首帧/未注册 event），当场采样一次并存入。frozen，无可训练参数。
    """
    if not _IMG_AUG_ENABLE:
        return x

    C = x.shape[1]
    device = x.device

    p = _img_aug_params.get(key) or _sample_img_aug_params(key)
    sigma     = p["sigma"]
    noise_std = p["noise_std"]

    # ── 高斯模糊（可分离卷积，先横后竖）──
    if sigma > 1e-6:
        k1d = _gaussian_kernel1d(sigma, _IMG_BLUR_KERNEL, device)
        pad = _IMG_BLUR_KERNEL // 2
        kh = k1d.view(1, 1, 1, _IMG_BLUR_KERNEL).expand(C, 1, 1, _IMG_BLUR_KERNEL)
        kv = k1d.view(1, 1, _IMG_BLUR_KERNEL, 1).expand(C, 1, _IMG_BLUR_KERNEL, 1)
        x = torch.nn.functional.conv2d(x, kh, padding=(0, pad), groups=C)
        x = torch.nn.functional.conv2d(x, kv, padding=(pad, 0), groups=C)

    # ── 高斯噪声 ──
    if noise_std > 1e-6:
        x = x + torch.randn_like(x) * noise_std

    return x.clamp_(0.0, 1.0)


def randomize_image_aug(_env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """每次 reset 重采样两个相机的图像增强强度（episode 内固定）。"""
    if len(env_ids) == 0:
        return
    for cam_name in ("camera_fixed", "camera_wrist"):
        _sample_img_aug_params(cam_name)


def theia_visual_feature(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg = SceneEntityCfg("camera_fixed"),
    model_path: str = THEIA_MODEL_PATH,
) -> torch.Tensor:
    """RGB → Theia-tiny(FP16) → (B, 576) float32。

    spatial 版池化：patch mean(192) ⊕ spatial-softmax(384) = 576 维。
    保留全局语义(mean) + 加入空间定位(每通道注意力期望坐标)。frozen，无可训练参数。
    """
    device = env.device
    model  = _get_theia_model_fp16(model_path, device)
    camera = env.scene[camera_cfg.name]

    rgb = camera.data.output["rgb"]

    if rgb.dtype == torch.uint8:
        x = rgb[..., :3].permute(0, 3, 1, 2).float() / 255.0
    else:
        x = rgb[..., :3].permute(0, 3, 1, 2).float()

    # 图像增强（高斯模糊 + 高斯噪声），作用在 [0,1] 图像上，归一化前。
    # key=相机名：每相机各自一组强度，整轮固定（reset 时重采样）。
    x = _augment_image(x, key=camera_cfg.name)

    if device not in _norm_cache:
        _norm_cache[device] = (
            torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1),
            torch.tensor([0.5, 0.5, 0.5], device=device).view(1, 3, 1, 1),
        )
    mean, std = _norm_cache[device]
    x = (x - mean) / std

    with torch.no_grad():
        out    = model.backbone.model(pixel_values=x.half(), interpolate_pos_encoding=True)
        patch  = out.last_hidden_state[:, 1:].float()       # (B, 196, 192)
        feat_mean = patch.mean(dim=1)                       # (B, 192) 全局语义
        feat_sp   = _spatial_softmax(patch)                 # (B, 384) 空间定位
        feat = torch.cat([feat_mean, feat_sp], dim=-1)      # (B, 576) fp32
    return feat


##
# 场景配置
##


@configclass
class XArm7PickLiftCubeSceneCfg(LiftWorkbenchSceneCfg):
    """双相机（腕部+固定）+ grip 碰撞检测 + 全套域随机化场景。"""

    robot = XARM7_GRIP_ON_FALAN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (0.0, 0.0, 0.822)
    robot.init_state.rot = (0.707, 0.0, 0.0, -0.707)

    # 覆盖基类支撑柱坐标。几何链（桌面顶 = 0.815）：
    #   柱底 0.816  ← 离桌面留 1 mm 间隙，避免与桌面共面导致接触求解抖动
    #   柱顶 0.822  ← 机械臂基座坐在这里，比桌面高 7 mm
    #   高度 = 0.822 - 0.816 = 0.006，中心 z = (0.816 + 0.822) / 2 = 0.819
    #
    # 2026-08-12：基座离桌面由 30 mm 改为 7 mm，柱高相应由 29 mm 改为 6 mm。
    # 那 1 mm 间隙按原样保留 —— 它防的是共面接触抖动，柱子变矮并不消除这个问题。
    # 但注意间隙占比从 1/29(3.4%) 升到 1/7(14.3%)，柱子实际"悬空"的相对量变大了；
    # 柱子是 kinematic + disable_gravity，不靠接触支撑，所以功能上无影响。
    robot_support_box = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/RobotSupportBox",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.0, 0.0, 0.819],
            rot=[0.707, 0.0, 0.0, -0.707],
        ),
        spawn=sim_utils.CylinderCfg(
            # 半径 0.08 而非 0.10：基座在 (0,0)，桌面 +y 边缘在 +0.08，半径 0.1 的
            # 柱子会伸出桌沿 2 cm 悬空。0.08 正好内切于桌边。
            radius=0.08,
            height=0.006,
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

    workbench = None

    # 桌面 1.2 x 1.2 m，厚度 0.815（顶面 z=0.815 不变）。
    # 机械臂基座在 (0, 0)，位于桌子 +y 边缘往内 8 cm：
    #   +y 边缘 = +0.08，-y 边缘 = 0.08 - 1.2 = -1.12
    #   中心 y  = (0.08 + (-1.12)) / 2 = -0.52
    #   x 方向基座居中 → 中心 x = 0，x 范围 -0.6 ~ +0.6
    # 高度不变：中心 z = 0.815/2 = 0.4075，顶面仍是 0.815。
    workpiece_table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WorkpieceTable",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.0, -0.52, 0.4075],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.MeshCuboidCfg(
            size=(1.2, 1.2, 0.815),
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

    # ── 白墙 ──────────────────────────────────────────────────────────────
    # 房间 2.00 x 2.00 m，只建两面（另两侧敞开），高度 3.0 m。
    #
    # 2026-08-12：墙体整体顺时针旋转 90°，即"贴着桌子那条边"从桌面 +y 边换成 +x 边。
    # 桌面 x -0.60~0.60，y -1.12~0.08：
    #   贴合墙  内表面 x = +0.60（贴桌面 +x 边），墙体在 x > 0.60 一侧
    #   偏移墙  内表面 y = 0.08 + 0.49 = +0.57（距桌面 +y 边 0.49 m），墙体在 y > 0.57
    #   房间范围 x -1.40~0.60，y -1.43~0.57，跨度均为 2.00 ✓
    #
    # 墙厚 0.02，**朝房间外**长出去，内表面精确落在上述平面，不侵占 2.00 m 净空。
    # 两面墙在拐角处各自延长覆盖对方厚度（跨度 2.02），避免拐角漏缝。
    wall_x_pos = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallXPos",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[0.61, -0.42, 1.5],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.MeshCuboidCfg(
            # y 跨度 2.02：从房间下沿 -1.43 到偏移墙外表面 +0.59，把拐角补满
            size=(0.02, 2.02, 3.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.92, 0.92, 0.92),
                metallic=0.0,
                roughness=0.9,
            ),
        ),
    )

    wall_y_pos = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/WallYPos",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=[-0.39, 0.58, 1.5],
            rot=[1.0, 0.0, 0.0, 0.0],
        ),
        spawn=sim_utils.MeshCuboidCfg(
            # x 跨度 2.02：从房间左沿 -1.40 到贴合墙外表面 +0.62
            size=(2.02, 0.02, 3.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.92, 0.92, 0.92),
                metallic=0.0,
                roughness=0.9,
            ),
        ),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.0, -0.44, 0.833],
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
            focal_length=_WRIST_FOCAL_LENGTH,
            horizontal_aperture=_WRIST_HORIZONTAL_APERTURE,
            clipping_range=(0.01, 5.0),
        ),
        width=224,
        height=224,
    )

    camera_fixed = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/CameraFixed",
        offset=TiledCameraCfg.OffsetCfg(
            pos=(-0.4000, -0.5800, 1.8000),
            rot=(0.7002, 0.0984, -0.0984, -0.7002),
            convention="world",
        ),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=_FIXED_FOCAL_LENGTH,
            horizontal_aperture=_FIXED_HORIZONTAL_APERTURE,
            clipping_range=(0.1, 1.0e5),
        ),
        width=224,
        height=224,
    )

    # 覆盖基类穹顶光：改用球形点光源，放在桌子中心正上方 3m（桌面顶 0.815 + 3 = 3.815）。
    # 只在 camfix 生效，不动基类（其他 env 仍用 DomeLight）。
    # 色温/强度/曝光/半径在 randomize_light_intensity 里按 reset 随机化；此处为初值。
    #
    # 2026-08-12：prim 路径从全局的 /World/Light 移到 {ENV_REGEX_NS}/Light。
    # 原来整个 stage 只有**一盏**灯（实测遍历确认），被所有 env 共用：
    #   ① 并行时 N 个 env 只能拿到 1 种光照，光照随机化的样本量被 num_envs 除掉；
    #   ② 灯固定在 env_0 桌子上方，其他 env 的桌子在 x=±2.5/±5.0…，离灯越来越远，
    #      入射角和距离衰减都不同 —— 各 env 的光照条件本就不等价。
    # 放进 env 命名空间后由 clone 逐 env 复制，两个问题一并消除。
    # init_state.pos 是 env 局部坐标，clone 时自动加上各自的 env_origin。
    light = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Light",
        spawn=sim_utils.SphereLightCfg(
            radius=0.1,
            color=(1.0, 1.0, 1.0),
            intensity=25000.0,
            enable_color_temperature=True,
            color_temperature=6500.0,
            normalize=True,
            exposure=0.0,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, -0.6, 3.815)),
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
    """1166 维观测配置（FP16 编码器，mean+spatial-softmax 池化，无点云）
    theia_fixed(576) + theia_wrist(576) + joint_pos(7) + last_action(7)
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
    randomize_image_aug       = EventTerm(func=randomize_image_aug,       mode="reset")


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
    """xArm7 Pick LiftCube 环境：FP16 编码器，1166 维观测（mean+spatial-softmax 池化，无点云），两段式奖励，全套 DR。"""

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
