# Copyright (c) 2024, Isaac Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""xArm7 Pick 环境配置 —— Point Bridge 风格点云观测。

规格来源：``point_bridge_rl_observation_spec.md``。

相对 ``xarm7_pick_vision_spatial_camfix_env_cfg`` 的改动，只有观测与相机两项：

    观测   1166 维双相机 Theia 特征  →  287 维（点云 280 ⊕ 关节角 7）
    相机   224x224 未标定视角        →  640x480 标定内外参，出深度 + 实例分割

奖励、动作空间与限幅、终止条件、物理参数、PPO 超参全部从 camfix 直接 import
复用，一行不改（规格 §0 的改动范围表）。

**相机内外参来源（与 camfix 的差异，务必知晓）**

camfix 用的那组位姿在 ``xarm_va/assets/cameras.py`` 里被标注为
``LEGACY_CZR_CAMERA_*``，数值逐位相同，注释写明它是 "hand-placed scene prop,
never measured against hardware" —— 即从未标定。本配置改用该模块里真正标定过的
那组（AprilTag 手眼标定 + 机器人基座位姿复合得到的世界系位姿）。

内参也不是缩放关系：camfix 的 fx=282.5 相当于把 D435 整帧按高度 resize 到 224
再当正方形图用，等效把水平 FOV 砍到与垂直相同（43.3°x43.3°）；标定值是真正的
4:3、54.98°x42.64°。camfix 注释里 "224 center-crop" 的说法与实际不符 ——
center-crop 不改变像素焦距。

因此本配置与 camfix 的 checkpoint **完全不兼容**，必须从零训练。

**外参的状态（2026-08-12 更新）**

现用外参来自 ``GSworld/GSWorld/tools/camera_calibration/results/
global_20260812_104936/camera_to_right_arm_base.json``，已**通过**质量门
（``quality_gate.pass: true``），替代了此前那份 ``forced_solution: true`` 的标定：

    位置 RMS 2.99 mm / 最大 6.81 mm，旋转 RMS 1.16° / 最大 2.35°
    20 组采样中 18 组为内点（RANSAC + 两轮精炼，剔除 marker 滑动的早期样本）
    DANIILIDIS / TSAI / HORAUD 三法互相一致到 0.21° RMS
    leave-one-out 稳定性 0.26° RMS

多方法一致性与 LOO 稳定性是这份标定比旧版可信的关键 —— 旧版"残差漂亮但欠激励"
的问题恰恰在这两项上暴露不出来，而这里两项都过了。

门控备注写明是"在位姿 31-50 子集上用放宽阈值求解"，因此仍不宜当作最终可部署
外参；但作为仿真起点，它比旧值可信得多。真机表现与仿真不符时，这里仍是首查点。
"""

from __future__ import annotations

import math
import os

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import quat_apply, quat_apply_inverse, random_orientation

import isaaclab.envs.mdp as mdp

from .point_bridge_pointcloud import (
    FPS_CANDIDATE_CAP,
    M_OBJ,
    M_WRIST,
    N_ROBOT,
    NOISE_STD_M,
    NUM_POINTS,
    POINT_DIM,
    assemble_point_cloud,
    build_instance_id_lut,
    dense_occlusion_pixel_map,
    farthest_point_sample,
    pose_in_frame,
    gripper_keypoints,
    mask_depth_to_pointcloud,
)
from .occlusion import occlusion_keep

# 奖励 / 动作 / 终止 / DR 全部复用 camfix，不重新定义（规格 §0：这些"不动"）
from .xarm7_pick_vision_spatial_camfix_env_cfg import (
    GONGJIAN_USD,
    XArm7PickLiftCubeActionsCfg,
    XArm7PickLiftCubeCurriculumCfg,
    XArm7PickLiftCubeRewardsCfg,
    XArm7PickLiftCubeSceneCfg,
    XArm7PickLiftCubeTerminationsCfg,
    last_clipped_action,
    randomize_light_intensity,
    reset_table_height_and_object_pose,
    set_success_termination_enabled,  # noqa: F401  (训练脚本从本模块 import)
    _TABLE_HEIGHT_RAND,
)

# YOLO 掩码源（stage 3）。这里只 import 薄封装，ultralytics 本身是在
# get_yolo() 里**延迟**加载的 —— 否则 stage -2/-1/0/1/2 这些用不到 YOLO 的路线
# 也会被迫要求 isaaclab 环境里装了 ultralytics。
from .yolo_mask_source import (
    DEFAULT_YOLO_STEP,
    DEFAULT_YOLO_WEIGHTS,
    YOLO_CONF,
    yolo_masks,
)


# ──────────────────────────────────────────────────────────────────────────────
# 标定相机参数
# ──────────────────────────────────────────────────────────────────────────────
# 内参：xarm_va/assets/canonical_camera.py 的 canonical 针孔模型。
# D435 实测 (serial 254622072913, 640x480, fw 5.17.0.10) fx=605.38/fy=604.96，
# canonical fx 取 615（略大 → FOV 略窄）以保证每个 canonical 像素都严格落在原始
# 帧内、不靠边界复制伪造像素。Omniverse 要求方形像素 + 主点居中，故 fx==fy、
# cx=W/2、cy=H/2。
CANONICAL_WIDTH = 640
CANONICAL_HEIGHT = 480
CANONICAL_FX = 615.0
CANONICAL_FY = 615.0
CANONICAL_CX = 320.0
CANONICAL_CY = 240.0

# 外参：标定给的是**相对机器人基座系**的位姿，但 TiledCameraCfg.OffsetCfg 是
# "w.r.t. the parent frame"，而 prim_path="{ENV_REGEX_NS}/CameraFixed" 的父级是
# env 根（≈世界系），**不是**机器人基座。直接把基座系数值填进 OffsetCfg 会让相机
# 落在错误的地方 —— 具体表现是 z=0.592 < 桌面 0.815，相机被埋进桌子里 22 cm。
#
# 因此这里存的是换算到 env 根坐标系之后的值。换算链（见本文件末尾的自检代码）：
#     基座在世界系 pos=(0,0,0.845), rot=(0.707,0,0,-0.707)   ← camfix 场景定义
#     cam_w  = base_pos + R_base @ cam_base
#     R_cam_w = R_base @ R_cam_base
#
# 标定原始值（基座系，供追溯 / 真机部署用，**不要**直接填进 OffsetCfg）。
#
# 2026-09-01 重标（camera_to_base.v1，parent=xarm_base / child=camera_color，serial
# 254622072913 同一台，130 raw / 89 active poses，质量门 PASS）。源文件：
# /home/lsz/gx_pose/config/camera_to_base.json
#     quality: position_rms 1.19 mm / max 2.54 mm，rotation_rms 0.29 deg
#
# 与上一组（0.394, -0.771, 0.592）/ (-0.518078, ...) 相差 11.34 mm / 1.043 deg ——
# 远超本次标定自身的 1.19 mm 噪声，是相机被拆卸重装后的真实位移，不是抖动。
#
# 注意 JSON 里是 xyzw（orientation_quaternion_xyzw），此处存的是 **wxyz**，已换序。
CALIB_CAMERA_POSITION_IN_BASE_M = (0.39141, -0.781319, 0.588075)
CALIB_CAMERA_QUATERNION_IN_BASE_WXYZ = (-0.523262, 0.851472, 0.006784, -0.033852)

# 机器人基座在世界系的位姿，**必须**与 camfix 场景里 robot.init_state 保持一致。
# 改了那边就必须同步改这里，否则相机会静默错位 —— 下面的 _verify_camera_extrinsics()
# 会在 import 时直接读 camfix 的场景配置来核对，改漏了会启动即报错。
ROBOT_BASE_POSITION_W = (0.0, 0.0, 0.822)
ROBOT_BASE_QUATERNION_W_WXYZ = (0.707, 0.0, 0.0, -0.707)

# env 根坐标系下的相机位姿 —— 这才是 OffsetCfg 要的东西。
# 2026-08-12：基座由 z=0.845 降到 0.822（离桌面 7 mm），相机随之下移 23 mm，
# z 由 1.437 变为 1.414。姿态不变（基座只改平移，R_base 不变）。
# 2026-09-14：随 2026-09-01 重标更新。下面这组是 **同一台相机**（serial
# 254622072913）重标后的结果，基座没动，故换算基准 ROBOT_BASE_* 未变。
FIXED_CAMERA_POSITION_M = (-0.781319, -0.39141, 1.410075)

# ROS 光学系四元数（+X 右，+Y 下，+Z 前），即标定本身所在的约定。
#
# 必须与 convention="ros" 成对使用。Isaac Lab 的 convention="world" 是
# USD/OpenGL 相机轴（+Y 上、-Z 前），把 ROS 四元数喂给它会让相机转 180° 拍空天
# —— xarm_va 的注释记录了这个坑已经踩过一次。camfix 用的是 world + 另一组四元
# 数，两者不可混搭。
#
# 与位置同理，这是 R_base @ R_cam_base 换算到 env 根之后的值，不是标定原始值。
#
# 2026-08-12 修正两处独立错误：
#   ① 旧四元数 (0.662,-0.158,0.179,0.710) 与标定文件相差 120°，光轴朝上 +62.4°
#      （拍天花板）而不是朝下 -27.4°（拍桌面）。
#   ② 位置与姿态都误把"基座系"数值当成了 OffsetCfg 的"父级系"数值。
# 两处都不会让管线报错 —— 掩码非空、点云照常生成，只有肉眼看画面或在视口里标出
# 相机位置才能发现。这正是 mark_camera_position.py 存在的理由。
#
# 2026-09-14：随 2026-09-01 重标更新，与上面的 CALIB_CAMERA_* 成对。
FIXED_CAMERA_QUATERNION_WXYZ = (0.393939, -0.606879, 0.597285, -0.346065)
def _base_frame_to_env_frame(
    pos_b: tuple[float, float, float],
    quat_b: tuple[float, float, float, float],
) -> tuple[tuple[float, ...], tuple[float, ...]]:
    """标定的基座系位姿 → env 根坐标系位姿（OffsetCfg 要的那个系）。

    存在的理由：上面 ``FIXED_CAMERA_*`` 是硬编码的换算结果，可读、可 diff、不依赖
    导入顺序；但机器人基座位姿一旦在 camfix 里改动，硬编码值就会静默失效 —— 而
    "相机错位"不会让任何断言失败，只会让点云悄悄偏掉。所以这里在 import 时重算
    一遍并比对，把静默错位变成启动即报错。
    """
    import numpy as np

    def _quat_to_matrix(q):
        w, x, y, z = np.asarray(q, dtype=np.float64) / np.linalg.norm(q)
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ])

    def _matrix_to_quat(R):
        t = np.trace(R)
        if t > 0:
            s = np.sqrt(t + 1.0) * 2
            q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
        else:
            i = int(np.argmax(np.diag(R)))
            if i == 0:
                s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
                q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
            elif i == 1:
                s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
                q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
            else:
                s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
                q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
        q = np.asarray(q) / np.linalg.norm(q)
        # 四元数双覆盖：q 与 -q 表示同一旋转，统一取 w>=0 以便和硬编码值逐位比对
        return q if q[0] >= 0 else -q

    R_base = _quat_to_matrix(ROBOT_BASE_QUATERNION_W_WXYZ)
    base_pos = np.asarray(ROBOT_BASE_POSITION_W, dtype=np.float64)

    pos_w = base_pos + R_base @ np.asarray(pos_b, dtype=np.float64)
    quat_w = _matrix_to_quat(R_base @ _quat_to_matrix(quat_b))
    return tuple(pos_w), tuple(quat_w)


def _verify_camera_extrinsics(tol_m: float = 1.0e-3, tol_quat: float = 1.0e-3) -> None:
    """启动自检：硬编码的 env 系外参必须与标定值重算的结果一致。

    两道检查：
      ① ROBOT_BASE_* 必须与 camfix 场景里 robot.init_state 的实际值一致 ——
         直接读那边的配置对象，不信任本文件里手抄的副本。基座位姿改了却漏改这里
         是最容易发生的一类错误（2026-08-12 把基座从 0.845 降到 0.822 时就险些
         漏掉），而它的后果是相机静默错位、点云整体偏移、训练照跑不报错。
      ② 由标定值重算的 env 系位姿必须与硬编码的 FIXED_CAMERA_* 一致。

    容差 1 mm / 1e-3 四元数分量，只抓结构性错误，不是数值精度检查（硬编码值本身
    就是六位小数的舍入）。
    """
    import numpy as np

    # ① 基座位姿与 camfix 对账。放在函数内 import 避免模块级循环依赖。
    #
    # 必须**实例化**再读：@configclass 把类成员转成带 default_factory 的 dataclass
    # 字段，值只在实例上，直接取 XArm7PickLiftCubeSceneCfg.robot 会 AttributeError。
    # 实例化只构造 cfg 对象（不碰 USD、不需要 sim app），代价可忽略。
    from .xarm7_pick_vision_spatial_camfix_env_cfg import XArm7PickLiftCubeSceneCfg

    _scene = XArm7PickLiftCubeSceneCfg()
    actual_pos = tuple(_scene.robot.init_state.pos)
    actual_rot = tuple(_scene.robot.init_state.rot)
    dbp = float(np.abs(np.asarray(actual_pos) - np.asarray(ROBOT_BASE_POSITION_W)).max())
    dbr = float(np.abs(np.asarray(actual_rot) - np.asarray(ROBOT_BASE_QUATERNION_W_WXYZ)).max())
    if dbp > tol_m or dbr > tol_quat:
        raise RuntimeError(
            "机器人基座位姿与 camfix 场景不一致，相机外参的换算基准已失效。\n"
            f"  camfix 实际值 : pos={actual_pos} rot={actual_rot}\n"
            f"  本文件记录值 : pos={ROBOT_BASE_POSITION_W} rot={ROBOT_BASE_QUATERNION_W_WXYZ}\n"
            "请更新 ROBOT_BASE_*，并用新基座位姿重算 FIXED_CAMERA_*。"
        )

    # ② 外参换算对账
    pos, quat = _base_frame_to_env_frame(
        CALIB_CAMERA_POSITION_IN_BASE_M, CALIB_CAMERA_QUATERNION_IN_BASE_WXYZ
    )
    dp = float(np.abs(np.asarray(pos) - np.asarray(FIXED_CAMERA_POSITION_M)).max())
    dq = float(np.abs(np.asarray(quat) - np.asarray(FIXED_CAMERA_QUATERNION_WXYZ)).max())
    if dp > tol_m or dq > tol_quat:
        raise RuntimeError(
            "固定相机外参与标定值不一致，相机会静默错位。\n"
            f"  由标定值重算 : pos={tuple(round(v, 6) for v in pos)} "
            f"quat={tuple(round(v, 6) for v in quat)}\n"
            f"  文件里硬编码 : pos={FIXED_CAMERA_POSITION_M} quat={FIXED_CAMERA_QUATERNION_WXYZ}\n"
            f"  最大偏差     : 位置 {dp:.6f} m / 四元数 {dq:.6f}\n"
            "机器人基座位姿(ROBOT_BASE_*)若在 camfix 里改过，请用重算值更新 FIXED_CAMERA_*。"
        )


_verify_camera_extrinsics()


FIXED_CAMERA_CONVENTION = "ros"

# 标定得到的观察几何（2026-09-14 随 2026-09-01 重标重算）：
# 相机在世界系 (-0.781, -0.391, 1.410)，高出桌面(z=0.815) 0.595 m；光轴朝下
# 与水平面成 26.8°（相对铅垂 63.2°），在 1.322 m 处打到桌面 (0.396, -0.306, 0.815)。
# 工件顶面约 10x6 cm → 掩码远超规格 §2.2 的 200~500 门槛。
#
# 与上一组（2026-08-12）相比俯仰几乎没变（27.4° → 26.8°），但指向横向移动了约
# 1.1 cm —— 与两版标定之间 11.34 mm 的平移差一致。这正是"同一台相机重装后必须
# 重标"的原因：俯仰对得上会让人误以为外参没问题，实际落点已经偏了。
#
# 旧注释曾写"光轴偏垂直 25.1°、1.215 m 处打到 (0.132,-0.359,0.815)"，那是配更早
# 一组外参的，已随 2026-08-12 与本次修正一并更新。
FIXED_CAMERA_CLIPPING_RANGE_M = (0.05, 20.0)

# 物体 prim 路径模板，用于每次 reset 后重查实例 ID（规格 §4.1③）。
OBJECT_PRIM_PATTERN = "/World/envs/env_{env}/Object"


def canonical_intrinsic_matrix(
    width: int = CANONICAL_WIDTH, height: int = CANONICAL_HEIGHT
) -> list[float]:
    """行主序 3x3 canonical 内参，可按分辨率等比缩放。

    缩放保持投影模型不变，因此换分辨率时 FOV 不变。**必须保持 4:3**：非等比
    缩放会让 fx != fy，违反 Omniverse 的方形像素约束。规格 §2.2 建议的 84x84
    正是这种情形，不能直接照用。
    """
    scale_x = width / CANONICAL_WIDTH
    scale_y = height / CANONICAL_HEIGHT
    if abs(CANONICAL_FX * scale_x - CANONICAL_FY * scale_y) > 1.0e-4:
        raise ValueError(
            f"Non-uniform rescale gives fx != fy ({CANONICAL_FX * scale_x} vs "
            f"{CANONICAL_FY * scale_y}), which Omniverse pinhole cameras cannot "
            f"represent. Keep the 4:3 aspect ratio of {CANONICAL_WIDTH}x{CANONICAL_HEIGHT}."
        )
    return [
        CANONICAL_FX * scale_x, 0.0, CANONICAL_CX * scale_x,
        0.0, CANONICAL_FY * scale_y, CANONICAL_CY * scale_y,
        0.0, 0.0, 1.0,
    ]


def make_pointcloud_camera_cfg(
    width: int = CANONICAL_WIDTH,
    height: int = CANONICAL_HEIGHT,
) -> TiledCameraCfg:
    """出深度 + 实例分割的标定固定相机（规格 §4.1）。"""

    return TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/CameraFixed",
        update_period=0.0,
        offset=TiledCameraCfg.OffsetCfg(
            pos=FIXED_CAMERA_POSITION_M,
            rot=FIXED_CAMERA_QUATERNION_WXYZ,
            convention=FIXED_CAMERA_CONVENTION,
        ),
        # distance_to_image_plane：沿相机 z 轴到成像平面的距离。
        # 不能用 distance_to_camera（到光心的欧氏距离）—— 用错不报错，只会让边缘
        # 像素系统性偏差（规格 §4.1①）。
        data_types=["distance_to_image_plane", "instance_id_segmentation_fast", "rgb"],
        # 必须 False：True 会把 ID 映射成 uint8 四通道彩色图，拿不到原始 ID。
        colorize_instance_id_segmentation=False,
        spawn=sim_utils.PinholeCameraCfg.from_intrinsic_matrix(
            intrinsic_matrix=canonical_intrinsic_matrix(width, height),
            width=width,
            height=height,
            clipping_range=FIXED_CAMERA_CLIPPING_RANGE_M,
        ),
        width=width,
        height=height,
        # 必改项。默认 "none" 时超出最大范围的像素对 distance_to_image_plane
        # 返回 NaN，NaN 会静默传进 PointNet 把梯度打成 NaN（规格 §4.1② / §9 首
        # 行）。设 "zero" 后统一过滤 0 值 —— 0 同时代表"没打到东西"，一并滤掉。
        depth_clipping_behavior="zero",
        # 相机位姿若被 set_world_poses() 写过，不开这个标志则 data.pos_w /
        # data.quat_w_ros 会一直报告标称 offset。本配置关掉了相机位姿 DR，但反
        # 投影严格依赖这两个值，保持开启以免将来加回 DR 时静默错位。
        update_latest_camera_pose=True,
    )


# ── 腕部相机（stage 10）───────────────────────────────────────────────────────
# 位姿 / 分辨率 / 内参逐字沿用 camfix 的 camera_wrist（用户要求"参考之前的腕部
# 相机"）。camfix 那一份是给 Theia RGB 用的，这里只要深度。
WRIST_CAMERA_PRIM_PATH = "{ENV_REGEX_NS}/Robot/link7/CameraWrist"
WRIST_CAMERA_POSITION_M = (0.11, 0.0, 0.0)
WRIST_CAMERA_QUATERNION_WXYZ = (1.0, 0.0, 0.0, 0.0)
WRIST_CAMERA_CONVENTION = "ros"
WRIST_CAMERA_WIDTH = 224
WRIST_CAMERA_HEIGHT = 224
# 与 camfix 的 _WRIST_FOCAL_LENGTH / _WRIST_HORIZONTAL_APERTURE 相同。
WRIST_FOCAL_LENGTH = 1.93
WRIST_HORIZONTAL_APERTURE = 2.3712
# 近端 1cm：腕部相机离夹爪很近，裁太远会把夹爪连同它挡住的物体一起裁掉。
WRIST_CAMERA_CLIPPING_RANGE_M = (0.01, 5.0)


def make_wrist_pointcloud_camera_cfg() -> TiledCameraCfg:
    """只出深度的腕部相机（stage 10）。

    与 :func:`make_pointcloud_camera_cfg` 的三处刻意差异：

    **只有深度一路 data_type。** 腕部不做分割（见 point_bridge_pointcloud 里
    ``M_WRIST`` 的说明），也没有 RGB 消费者，因此单 AOV —— 224x224 的单路缓冲比
    固定相机的 640x480 双路便宜一个量级，这是 stage 10 仍能开大 num_envs 的原因。

    **不走 canonical_intrinsic_matrix()。** 那个函数强制 4:3（非等比缩放会让
    fx != fy，Omniverse 的方形像素表示不了），224x224 直接触发它的 ValueError。
    这里改用 ``PinholeCameraCfg(focal_length, horizontal_aperture)`` 让 Isaac 自己
    从物理镜头参数算内参，与 camfix 同法；下游反投影读的是
    ``camera.data.intrinsic_matrices``，谁生成的都一样。

    **update_latest_camera_pose=True 在这里是硬要求，不是保险。** 固定相机那份
    开着它只为将来加回位姿 DR；腕部相机**本身就随臂运动**，不开这个标志时
    ``data.pos_w`` / ``data.quat_w_ros`` 会一直报告初始的标称 offset，于是每一个
    腕部点都被按错误外参反投影 —— 而且不报任何错，点云看着"有值"、训练照跑，
    只是几何全错。这是本阶段最容易静默失败的一处，验收第 1 步专门查它。
    """

    return TiledCameraCfg(
        prim_path=WRIST_CAMERA_PRIM_PATH,
        update_period=0.0,
        offset=TiledCameraCfg.OffsetCfg(
            pos=WRIST_CAMERA_POSITION_M,
            rot=WRIST_CAMERA_QUATERNION_WXYZ,
            convention=WRIST_CAMERA_CONVENTION,
        ),
        # distance_to_image_plane，不是 camfix 那一份的 distance_to_camera：
        # 后者是到光心的欧氏距离，喂给针孔反投影会让边缘像素系统性外扩
        # （规格 §4.1①）。camfix 用它没问题 —— 那边深度只进 Theia，不做反投影。
        data_types=["distance_to_image_plane"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=WRIST_FOCAL_LENGTH,
            horizontal_aperture=WRIST_HORIZONTAL_APERTURE,
            clipping_range=WRIST_CAMERA_CLIPPING_RANGE_M,
        ),
        width=WRIST_CAMERA_WIDTH,
        height=WRIST_CAMERA_HEIGHT,
        # 与固定相机同理：默认 "none" 会给超范围像素返回 NaN，NaN 静默毁梯度。
        # 腕部相机的远端只有 5m，画面里超范围的像素比固定相机更常见。
        depth_clipping_behavior="zero",
        update_latest_camera_pose=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 桌面固定贴图（替代已移除的颜色/纹理随机化）
# ──────────────────────────────────────────────────────────────────────────────

# 1:1 桌面贴图：1254x1254 方图，正好覆盖 1.2x1.2 m 桌面。
# 用 UV scale=(1,1) + clamp 包裹，使整张图不重复、不平铺地铺满桌面顶面。
# 路径软定位（不写死绝对路径）：环境变量 ``TABLE_TEXTURE_PATH`` 优先，其次仓库内
# ``tools/table.png``，最后退回旧机器的绝对路径兜底。
_TOOLS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools")
)


def _resolve_table_texture() -> str:
    """按 环境变量 → 仓库 tools → 旧机器兜底 的顺序解析桌面贴图路径。"""
    for candidate in (
        os.environ.get("TABLE_TEXTURE_PATH"),
        os.path.join(_TOOLS_DIR, "table.png"),
        "/home/gxai/Desktop/GSworld/table.png",
    ):
        if candidate and os.path.exists(candidate):
            return candidate
    return os.path.join(_TOOLS_DIR, "table.png")


TABLE_TEXTURE_PATH = _resolve_table_texture()

# 贴图绕 UV 原点旋转的角度（度，逆时针）。桌面是正方形、贴图也是正方形，所以 90°
# 的整数倍不会产生拉伸或露白。
TABLE_TEXTURE_ROTATION_DEG = 90.0


def _uv_rotation_offset(rotation_deg: float) -> tuple[float, float]:
    """求把旋转后的 UV 方块平移回 [0,1]² 所需的 translation。

    UsdTransform2d 绕**原点**旋转，而 UV 方块是 [0,1]²（不是以原点为中心），所以
    旋转后整块会跑偏：90° 跑到 u∈[-1,0]，180° 跑到 u,v∈[-1,0]，270° 跑到 v∈[-1,0]。
    配 wrap=clamp 时跑偏的后果不是"看不见贴图"，而是边缘一个像素被拉满整张桌子 ——
    看上去像纯色，很容易被误判成"贴图没生效"。

    这里按旋转后四角的最小值取反作为平移量，对任意角度都成立（非 90° 倍数时方块
    不再轴对齐，平移只保证落在第一象限，会露白 —— 但本配置只用 90° 倍数）。
    """
    import numpy as np

    th = np.radians(rotation_deg)
    R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
    rotated = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]) @ R.T
    return float(-rotated[:, 0].min()), float(-rotated[:, 1].min())


# 已建好贴图材质的 env，避免每次 reset 重建 shader 网络。
# 运行时重建材质网络会触发 RTX/Vulkan GPU pagefault（camfix 的注释记录了这个坑），
# 所以这里也遵循"一次建好、之后只复用"的做法。
_TABLE_TEX_APPLIED: set = set()


def apply_table_texture(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """给桌子顶面贴上固定的 1:1 贴图。

    与 camfix 的 ``randomize_table_color`` 不同，这里**不做任何随机化**：一张图、
    scale=(1,1)、无旋转，整张贴满 1.2x1.2 的桌面。

    UV 来自 camfix 的 ``_ensure_table_uv``：它把顶面顶点的 (x, y) 归一化到 [0,1]，
    正好就是 1:1 贴合所需的映射 —— 桌子是正方形、贴图也是正方形(1254x1254)，因此
    不会有拉伸。

    wrap 用 ``clamp`` 而不是 ``repeat``：UV 恰好落在 [0,1] 边界上时，repeat 会因
    浮点误差在桌沿采样到对侧像素，形成一条细缝。clamp 没有这个问题。

    幂等：材质只在首次为某个 env 建一次，之后的 reset 直接返回。
    """
    import omni.usd
    from pxr import Gf, Sdf, UsdShade

    from .xarm7_pick_vision_spatial_camfix_env_cfg import (
        _ensure_table_uv,
        _find_table_mesh,
    )

    if not os.path.isfile(TABLE_TEXTURE_PATH):
        raise FileNotFoundError(
            f"桌面贴图不存在: {TABLE_TEXTURE_PATH}\n"
            "改 TABLE_TEXTURE_PATH，或把图放到该路径。"
        )

    stage = omni.usd.get_context().get_stage()

    for env_id in env_ids.tolist():
        if env_id in _TABLE_TEX_APPLIED:
            continue

        # MeshCuboidCfg 生成的 mesh 不带 UV，没有 UV 纹理就贴不上
        _ensure_table_uv(stage, env_id)

        mesh = _find_table_mesh(stage, env_id)
        if mesh is None:
            continue

        mat_path = f"/World/envs/env_{env_id}/TableFixedTexMat"
        mat = UsdShade.Material.Define(stage, mat_path)
        shd = UsdShade.Shader.Define(stage, mat_path + "/Shader")
        shd.CreateIdAttr("UsdPreviewSurface")
        shd.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.8)
        shd.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.1)
        mat.CreateSurfaceOutput().ConnectToSource(shd.ConnectableAPI(), "surface")

        reader = UsdShade.Shader.Define(stage, mat_path + "/STReader")
        reader.CreateIdAttr("UsdPrimvarReader_float2")
        reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        # UV 旋转 90°。UsdTransform2d 的语义是
        #     result = rot(theta) @ (in * scale) + translation
        # 绕原点转 90° 会把 [0,1]² 转到 u ∈ [-1,0]，整块跑出贴图范围（配 clamp 会
        # 变成一条边像素被拉满整个桌面）。因此必须补一个 translation=(1,0) 把它平
        # 移回 [0,1]²。已验证四角精确落回 (0,0)/(1,0)/(1,1)/(0,1)。
        xform = UsdShade.Shader.Define(stage, mat_path + "/UVTransform")
        xform.CreateIdAttr("UsdTransform2d")
        xform.CreateInput("in", Sdf.ValueTypeNames.Float2).ConnectToSource(
            reader.GetOutput("result")
        )
        xform.CreateInput("rotation", Sdf.ValueTypeNames.Float).Set(TABLE_TEXTURE_ROTATION_DEG)
        xform.CreateInput("scale", Sdf.ValueTypeNames.Float2).Set(Gf.Vec2f(1.0, 1.0))
        xform.CreateInput("translation", Sdf.ValueTypeNames.Float2).Set(
            Gf.Vec2f(*_uv_rotation_offset(TABLE_TEXTURE_ROTATION_DEG))
        )
        xform.CreateOutput("result", Sdf.ValueTypeNames.Float2)

        tex = UsdShade.Shader.Define(stage, mat_path + "/DiffuseTex")
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(
            Sdf.AssetPath(TABLE_TEXTURE_PATH)
        )
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            xform.GetOutput("result")
        )
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("clamp")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("clamp")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)

        shd.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex.GetOutput("rgb")
        )

        UsdShade.MaterialBindingAPI(mesh.GetPrim()).Bind(mat)
        _TABLE_TEX_APPLIED.add(env_id)


# ──────────────────────────────────────────────────────────────────────────────
# 观测函数
# ──────────────────────────────────────────────────────────────────────────────

_INSTANCE_LUT_CACHE: dict = {}
_LAST_POINT_CLOUD: dict = {}
_EMPTY_MASK_STRIKES: dict = {}

# 阶段 0 可视化：id(env) 在 _DEBUG_ENABLED 里时，观测函数顺手把中间量存进
# _DEBUG_CAPTURE。用独立的开关集合而不是"字典非空"作判据 —— 空字典是 falsy，
# 拿它当开关会导致第一帧永远不记录。训练路径不碰这两个容器。
_DEBUG_ENABLED: set = set()
_DEBUG_CAPTURE: dict = {}

# 遮挡可视化：id(env) 在 _OCCLUSION_VIS_ENABLED 里时，stage2 遮挡观测函数把
# keep 掩码 + 候选像素索引存进 _OCCLUSION_VIS，供验证脚本在录像帧上标红被挖掉的
# 区域。纯只读旁路，训练路径不碰这两个容器。
_OCCLUSION_VIS_ENABLED: set = set()
_OCCLUSION_VIS: dict = {}

# 阶段 1（规格 §7）用的物体表面采样点缓存：{ id(env): (num_surface, 3) }，
# 局部物体坐标系。
_SURFACE_POINTS_CACHE: dict = {}

# 连续多少步整批掩码全空就报错停下。零阶保持只是为了扛住偶发单帧丢失；持续为空
# 说明相机、prim 路径或分割配置错了，继续训练只会喂给策略一串陈旧点云。
_MAX_EMPTY_STRIKES = 30

# stage 4 专用：YOLO 掩码缓存。每 yolo_step 步才重新跑一次 YOLO，中间步复用缓存
# 掩码 + 当前深度帧做反投影。不影响 stage 3 的任何路径。
_LAST_YOLO_MASK: dict = {}    # { id(env): (B, H, W) bool }
_YOLO_STEP_COUNTER: dict = {}  # { id(env): int }，距上次运行 YOLO 的步数

# stage 10 专用：腕部相机的零阶保持与空帧计数。
#
# **key 是 (id(env), camera_name)，不是 id(env)** —— 上面那些字典全都只用 id(env)，
# 因为历史上每个 env 只有一个相机。stage 10 加了第二个相机后，如果腕部沿用同一批
# 字典，两个相机会互相覆盖对方的"上一帧"：全局那一路复用到腕部的点、腕部复用到
# 全局的点，且**不报错**（形状都是 (B, 64, 3)）。故这里另开两个带相机名的字典，
# 现有的全局路径一行不动 —— stage -2..4 的行为逐位不变。
_LAST_WRIST_POINTS: dict = {}   # { (id(env), name): (B, M_WRIST, 3) }
_WRIST_EMPTY_STRIKES: dict = {}  # { (id(env), name): int }


def invalidate_instance_lut(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """reset 时丢弃实例 ID 查表缓存，强制下一帧重查（规格 §4.1③）。"""
    _INSTANCE_LUT_CACHE.pop(id(env), None)


_YOLO_EMPTY_DUMP_COUNT: dict = {}
_MAX_YOLO_EMPTY_DUMPS = 500


def _resolve_yolo_empty_debug_dir() -> str:
    """空检快照目录。默认直接写数据盘 ``/root/autodl-tmp``，可用 ``YOLO_EMPTY_DEBUG_DIR`` 覆盖。

    之前这里写相对路径 ``"logs/yolo_empty_debug"``，落到哪个盘取决于训练进程的 CWD
    （本机训练在 ``/root/gx-va`` 启动，落在**系统盘**），无上限时会填满系统盘；上一版
    是靠在系统盘放一个指向数据盘的 symlink 顶过去的。改成绝对路径直接写数据盘后，
    不再依赖 CWD、也不需要 symlink，删掉旧 symlink 也不影响。
    """
    return os.environ.get("YOLO_EMPTY_DEBUG_DIR", "/root/autodl-tmp/yolo_empty_debug")


def _save_yolo_empty_rgb(
    rgb_raw: torch.Tensor,
    key: int,
    env_idx: int = 0,
    mask_px: int | None = None,
) -> None:
    """YOLO 持续空检时存一张 RGB 快照，供事后核对画面是否漂出训练分布。

    ``env_idx`` **必须**是真正出问题的那个 env 的序号。之前这里硬写 ``rgb_raw[0]``，
    而判据是 ``empty.any()``（任意 env 为空），于是存下来的往往是健康 env 的画面 ——
    图里工件清清楚楚，反而把排查带偏。

    ``mask_px`` 是该 env 的掩码像素数，用来区分两种完全不同的故障：

    - ``mask_px == 0``：YOLO 真的没检出 —— 查画面/权重/conf。
    - ``mask_px > 0``：掩码有，但反投影后没有一个点通过有效性检查或工作空间裁剪
      —— 查深度（是否为 0/NaN）、相机外参、``WORKSPACE_Z_RANGE_M``。

    最多存 ``_MAX_YOLO_EMPTY_DUMPS`` 张就不再写盘 —— 这个函数是在故障路径上被调用
    的，如果画面真的一直不对，无上限地写图会填满磁盘（之前踩过一次 disk full）。
    500 张 480x640 PNG 约 100~250MB，量级可接受。
    """
    n = _YOLO_EMPTY_DUMP_COUNT.get(key, 0)
    if n >= _MAX_YOLO_EMPTY_DUMPS:
        return
    _YOLO_EMPTY_DUMP_COUNT[key] = n + 1

    try:
        import os
        import time

        out_dir = _resolve_yolo_empty_debug_dir()
        os.makedirs(out_dir, exist_ok=True)
        img = rgb_raw[env_idx]
        if img.dim() == 3 and img.shape[0] in (3, 4):
            img = img.permute(1, 2, 0)
        img = img[..., :3].detach().float()
        if float(img.max()) <= 1.0:
            img = img * 255.0
        arr = img.clamp(0, 255).to(torch.uint8).cpu().numpy()

        stamp = time.strftime("%H%M%S")
        tag = "nomask" if mask_px == 0 else f"maskpx{mask_px}"
        path = os.path.join(out_dir, f"yolo_empty_{stamp}_env{env_idx}_{tag}_{n}.png")
        try:
            import imageio.v2 as imageio

            imageio.imwrite(path, arr)
        except ImportError:
            from PIL import Image

            Image.fromarray(arr).save(path)

        if mask_px == 0:
            why = "掩码为空 —— YOLO 没检出，查画面/权重/conf"
        elif mask_px is None:
            why = "掩码像素数未知"
        else:
            why = (
                f"掩码有 {mask_px} px 但反投影后无有效点 —— 查深度是否为 0/NaN、"
                "相机外参、WORKSPACE_Z_RANGE_M 裁剪"
            )
        print(
            f"[YOLO] 持续空检 env={env_idx}：{why}\n"
            f"       快照 -> {path}（第 {n + 1}/{_MAX_YOLO_EMPTY_DUMPS} 张）"
        )
    except Exception as exc:  # noqa: BLE001
        # 存图失败绝不能把训练带下去 —— 这本来就是个诊断辅助。
        print(f"[YOLO] 空检快照保存失败（忽略）: {exc}")


def _get_instance_lut(env: ManagerBasedRLEnv, camera) -> torch.Tensor | None:
    """取（必要时重建）``instance id -> env 序号 + 1`` 查表。"""
    key = id(env)
    cached = _INSTANCE_LUT_CACHE.get(key)
    if cached is not None:
        return cached

    info = camera.data.info.get("instance_id_segmentation_fast")
    if not info:
        return None
    id_to_labels = info.get("idToLabels") if isinstance(info, dict) else None
    if not id_to_labels:
        return None

    lut = build_instance_id_lut(
        id_to_labels=id_to_labels,
        prim_path_pattern=OBJECT_PRIM_PATTERN,
        num_envs=env.num_envs,
        device=env.device,
    )
    if int(lut.max()) == 0:
        # 一个 ID 都没匹配上：prim 路径模板与实际 stage 不符，早失败胜过静默喂
        # 空点云。
        raise RuntimeError(
            "No instance segmentation ID matched the object prim pattern "
            f"{OBJECT_PRIM_PATTERN!r}. Available labels: {sorted(set(map(str, id_to_labels.values())))[:8]}"
        )
    _INSTANCE_LUT_CACHE[key] = lut
    return lut


def point_bridge_point_cloud(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg = SceneEntityCfg("camera_fixed"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    m_obj: int = M_OBJ,
    noise_std: float = NOISE_STD_M,
) -> torch.Tensor:
    """掩码深度 + 夹爪关键点 → 扁平点云观测 (B, (M_OBJ + N_ROBOT) * 3)。

    所有点在**机器人基座系**、单位米，每点仅 xyz。布局是
    ``[物体点 M_OBJ*3 | 夹爪点 N_ROBOT*3]``：网络侧按这两段切开、各自过一次
    同一个 PointNet（对齐参考实现的双 token 做法），因此不需要类型通道。

    返回扁平张量而非 (B, P, 4)：rsl_rl 3.0.1 的观测组必须是 1D
    （``ActorCritic`` 里有 ``len(obs[group].shape) == 2`` 的断言），由
    ``PointNetActorCritic.get_actor_obs`` 在网络侧 reshape 回 (B, P, 4)。

    遮挡为什么自动正确（规格 §1.2）：``instance_id_segmentation_fast`` 是逐像素
    的，被夹爪挡住的像素其值是夹爪的 prim ID 而不是物体的，因此
    ``mask == obj_id`` 天然不含被遮挡区域；物体自遮挡同理（同一条视线只有正面
    表面被渲染）。结果是单视角可见表面点，与真机 YOLO-seg 的行为一致。不需要手
    写任何可见性剔除 —— 渲染器的 z-buffer 已经算完了。
    """
    camera = env.scene[camera_cfg.name]
    robot = env.scene[robot_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]

    device = env.device
    num_envs = env.num_envs
    key = id(env)

    base_pos_w = robot.data.root_pos_w
    base_quat_w = robot.data.root_quat_w

    # ── 夹爪关键点（不经相机，零渲染开销；规格 §4.3）────────────────────────
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]
    ee_pos_b, ee_quat_b = pose_in_frame(
        ee_pos_w, ee_quat_w, base_pos_w, base_quat_w
    )
    # 夹爪点不加噪声：参考实现 read_data/mimiclabs.py 只对 object points 加噪，
    # robot points 路径上没有任何噪声（正运动学精度远高于 RGB-D 深度）。
    robot_pts = gripper_keypoints(ee_pos_b, ee_quat_b, noise_std=0.0)

    # ── 物体点 ──────────────────────────────────────────────────────────────
    depth_raw = camera.data.output.get("distance_to_image_plane")
    seg_raw = camera.data.output.get("instance_id_segmentation_fast")
    lut = _get_instance_lut(env, camera) if seg_raw is not None else None

    obj_pts = None
    if depth_raw is not None and seg_raw is not None and lut is not None:
        depth = depth_raw[..., 0] if depth_raw.dim() == 4 else depth_raw
        seg = seg_raw[..., 0] if seg_raw.dim() == 4 else seg_raw

        # 查表：物体像素为 env 序号 + 1，其余为 0。这样逐 env 不同的 ID、同一
        # 物体下多个 mesh 子 prim 的多个 ID 都被覆盖，邻居 env 入画也被排除。
        seg_idx = seg.long().clamp_(min=0, max=lut.numel() - 1)
        expected = torch.arange(num_envs, device=device).add_(1).view(-1, 1, 1)
        mask = lut[seg_idx] == expected

        cam_pos_b, cam_quat_b = pose_in_frame(
            camera.data.pos_w, camera.data.quat_w_ros, base_pos_w, base_quat_w
        )

        candidate_pts, counts = mask_depth_to_pointcloud(
            mask=mask,
            depth=depth,
            K=camera.data.intrinsic_matrices,
            cam_pos=cam_pos_b,
            cam_quat=cam_quat_b,
            m_obj=m_obj,
            noise_std=noise_std,
        )

        # 可见点占比：只进日志不进观测（规格 §4.2 / §8）。
        env.extras["visible_ratio"] = (
            counts.float() / float(m_obj)
        ).clamp_(max=1.0).mean()

        # 阶段 0 可视化用的中间量。只在开关打开时记录，训练时不占显存。
        if key in _DEBUG_ENABLED:
            _DEBUG_CAPTURE[key] = {
                "mask_pixels": mask.flatten(1).sum(dim=1).detach(),
                "visible_counts": counts.detach(),
                "depth": depth.detach(),
                "cam_pos_b": cam_pos_b.detach(),
                "cam_quat_b": cam_quat_b.detach(),
                "K": camera.data.intrinsic_matrices.detach(),
            }

        previous = _LAST_POINT_CLOUD.get(key)
        empty = counts == 0
        if bool(empty.any()):
            if previous is None:
                # 首帧就没有任何可见点：不能复用上一帧，只能报错。喂空点集给策略
                # 是规格 §11.1 明确禁止的。
                raise RuntimeError(
                    "The object mask is empty on the very first observation, so there is no "
                    "previous point cloud to hold. Check the camera extrinsics, the object "
                    "prim pattern, and that --enable_cameras is set."
                )
            candidate_pts = torch.where(
                empty.view(-1, 1, 1), previous[:, :m_obj, :], candidate_pts
            )
            strikes = _EMPTY_MASK_STRIKES.get(key, 0) + 1
            _EMPTY_MASK_STRIKES[key] = strikes
            if strikes > _MAX_EMPTY_STRIKES:
                raise RuntimeError(
                    f"The object mask has been empty for {strikes} consecutive steps in at "
                    "least one environment. Zero-order hold is only meant to cover isolated "
                    "dropped frames; this indicates a broken camera or segmentation setup."
                )
        else:
            _EMPTY_MASK_STRIKES[key] = 0

        obj_pts = candidate_pts

    if obj_pts is None:
        # 相机缓冲尚未填充（首个 reset 之前 observation manager 会先跑一次）。
        # 复用上一帧，没有上一帧则给零 —— 这一帧不会进入任何 rollout。
        previous = _LAST_POINT_CLOUD.get(key)
        if previous is not None:
            obj_pts = previous[:, :m_obj, :]
        else:
            obj_pts = torch.zeros(num_envs, m_obj, POINT_DIM, device=device)

    point_cloud = assemble_point_cloud(obj_pts, robot_pts)
    _LAST_POINT_CLOUD[key] = point_cloud.detach()

    if key in _DEBUG_ENABLED:
        _DEBUG_CAPTURE.setdefault(key, {})["points"] = point_cloud.detach()

    return point_cloud.reshape(num_envs, -1)


def point_bridge_wrist_point_cloud(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg = SceneEntityCfg("camera_wrist"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    m_wrist: int = M_WRIST,
    noise_std: float = NOISE_STD_M,
    min_depth_m: float = 0.0,
) -> torch.Tensor:
    """腕部相机整帧深度 → 扁平点云观测 (B, M_WRIST * 3)，**不做分割**。

    与 :func:`point_bridge_point_cloud` 的唯一实质差别是掩码：这里传一个
    ``depth > min_depth_m`` 的全帧掩码，而不是 ``instance_id == obj_id``。反投影、
    确定性 FPS、工作空间裁剪、噪声全部复用同一份
    :func:`mask_depth_to_pointcloud` —— 规格 §1.1 只允许存在一份反投影实现。

    点同样在**机器人基座系**、单位米。选基座系而不是相机系，是为了让腕部点与全局
    点、夹爪点处在同一坐标系里：三者可比才谈得上"拼接"，网络也能共用同一套
    ``POINT_NORM_MIN/MAX`` 归一化。代价是腕部点的数值随臂运动而变（相机系下则是
    静止的局部几何），但那正是任务信息 —— 末端在哪、物体相对基座在哪。

    没有夹爪关键点：那 6 个点由全局那一路提供，不重复进观测。

    Args:
        min_depth_m: 近端截断（米）。默认 0 = 不截断。腕部相机装在 link7 前 11cm
            朝夹爪方向看，夹爪本体会占据近场一大片，64 个点里可能有相当比例落在
            夹爪上。默认**不裁** —— 夹爪点与物体点在同一帧同一次采样里，它们的相对
            关系正是伺服需要的信息，裁掉反而丢东西。留这个旋钮是为了万一 stage 0
            可视化发现夹爪点占比过高（比如 >50%），可以一行调整。
    """
    camera = env.scene[camera_cfg.name]
    robot = env.scene[robot_cfg.name]

    device = env.device
    num_envs = env.num_envs
    # 带相机名的 key：与全局那一路的缓存彻底分开，见 _LAST_WRIST_POINTS 的说明。
    key = (id(env), camera_cfg.name)

    depth_raw = camera.data.output.get("distance_to_image_plane")

    wrist_pts = None
    if depth_raw is not None:
        depth = depth_raw[..., 0] if depth_raw.dim() == 4 else depth_raw

        # 全帧掩码。mask_depth_to_pointcloud 内部还会再过一遍
        # `depth > 0 & isfinite(depth)`，所以 min_depth_m=0 时这里等价于全 True，
        # 无效像素不会漏进去。
        if min_depth_m > 0.0:
            mask = depth > min_depth_m
        else:
            mask = torch.ones_like(depth, dtype=torch.bool)

        # 腕部相机随臂运动，这两个值必须是**当前**位姿。相机 cfg 里的
        # update_latest_camera_pose=True 就是为此，缺了它这里会拿到标称 offset。
        cam_pos_b, cam_quat_b = pose_in_frame(
            camera.data.pos_w,
            camera.data.quat_w_ros,
            robot.data.root_pos_w,
            robot.data.root_quat_w,
        )

        candidate_pts, counts = mask_depth_to_pointcloud(
            mask=mask,
            depth=depth,
            K=camera.data.intrinsic_matrices,
            cam_pos=cam_pos_b,
            cam_quat=cam_quat_b,
            m_obj=m_wrist,
            noise_std=noise_std,
        )

        env.extras["wrist_visible_ratio"] = (
            counts.float() / float(m_wrist)
        ).clamp_(max=1.0).mean()

        # 零阶保持。整帧采样下 counts=0 意味着**全画面**都没有有效深度（相机全部
        # 打在裁剪范围外，或深度缓冲还没填），这比掩码路线罕见得多，但兜底逻辑一样
        # 要有 —— 喂空点集给策略是规格 §11.1 明确禁止的。
        previous = _LAST_WRIST_POINTS.get(key)
        empty = counts == 0
        if bool(empty.any()):
            if previous is None:
                raise RuntimeError(
                    "The wrist camera produced no valid depth on the very first observation, "
                    "so there is no previous point cloud to hold. Check that "
                    "--enable_cameras is set, that the wrist camera prim path "
                    f"{WRIST_CAMERA_PRIM_PATH!r} exists, and that "
                    "depth_clipping_behavior is 'zero'."
                )
            candidate_pts = torch.where(empty.view(-1, 1, 1), previous, candidate_pts)
            strikes = _WRIST_EMPTY_STRIKES.get(key, 0) + 1
            _WRIST_EMPTY_STRIKES[key] = strikes
            if strikes > _MAX_EMPTY_STRIKES:
                raise RuntimeError(
                    f"The wrist camera has produced no valid depth for {strikes} consecutive "
                    "steps in at least one environment. For a full-frame (unsegmented) "
                    "sampler this almost always means the camera itself is broken, not that "
                    "the object left the view."
                )
        else:
            _WRIST_EMPTY_STRIKES[key] = 0

        wrist_pts = candidate_pts

    if wrist_pts is None:
        # 相机缓冲尚未填充（首个 reset 之前 observation manager 会先跑一次）。
        previous = _LAST_WRIST_POINTS.get(key)
        wrist_pts = (
            previous
            if previous is not None
            else torch.zeros(num_envs, m_wrist, POINT_DIM, device=device)
        )

    _LAST_WRIST_POINTS[key] = wrist_pts.detach()

    if id(env) in _DEBUG_ENABLED:
        _DEBUG_CAPTURE.setdefault(id(env), {})["wrist_points"] = wrist_pts.detach()

    return wrist_pts.reshape(num_envs, -1)


def point_bridge_point_cloud_yolo(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg = SceneEntityCfg("camera_fixed"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    m_obj: int = M_OBJ,
    noise_std: float = NOISE_STD_M,
    yolo_weights: str = DEFAULT_YOLO_WEIGHTS,
    yolo_conf: float = YOLO_CONF,
) -> torch.Tensor:
    """与 :func:`point_bridge_point_cloud` 完全相同，只把掩码来源换成 YOLO-seg。

    唯一的差别是这三行的替换（规格 §1.1 只允许有一份反投影实现，所以下游的
    ``mask_depth_to_pointcloud`` / FPS / 噪声 / 零阶保持全部原样复用）::

        原：seg = camera.data.output["instance_id_segmentation_fast"]
            mask = lut[seg] == env_id + 1
        今：rgb = camera.data.output["rgb"]
            mask, _ = yolo_masks(rgb)

    这么做的意义在于**训练时策略见到的掩码分布 == 真机部署时的掩码分布**。Isaac
    的分割是逐像素完美的，YOLO 的边缘会胖一圈、遮挡处会缺一块，而且这是系统性偏差
    不是零均值噪声（规格 §11.2）—— 靠加高斯噪声补不回来，只能让训练直接吃它。

    相机这一路只需要 ``rgb`` + ``distance_to_image_plane``，实例分割整路可以摘掉
    （见 :func:`set_observation_stage` 的 stage 3 分支）。

    检出为空时不在这里兜底 —— 落到下面既有的零阶保持分支，与 GT 分割路线共用同一套
    故障处理和 ``_MAX_EMPTY_STRIKES`` 计数，行为完全一致。
    """
    camera = env.scene[camera_cfg.name]
    robot = env.scene[robot_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]

    device = env.device
    num_envs = env.num_envs
    key = id(env)

    base_pos_w = robot.data.root_pos_w
    base_quat_w = robot.data.root_quat_w

    # ── 夹爪关键点（与 GT 路线逐字一致）──────────────────────────────────────
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]
    ee_pos_b, ee_quat_b = pose_in_frame(ee_pos_w, ee_quat_w, base_pos_w, base_quat_w)
    robot_pts = gripper_keypoints(ee_pos_b, ee_quat_b, noise_std=0.0)

    # ── 物体点：掩码来自 YOLO ───────────────────────────────────────────────
    depth_raw = camera.data.output.get("distance_to_image_plane")
    rgb_raw = camera.data.output.get("rgb")

    obj_pts = None
    if depth_raw is not None and rgb_raw is not None:
        depth = depth_raw[..., 0] if depth_raw.dim() == 4 else depth_raw

        mask, detected = yolo_masks(
            rgb_raw, weights=yolo_weights, conf=yolo_conf
        )

        # 检出率进日志（不进观测）。这是这条路线最该盯的一个量：它掉下去说明
        # 渲染画面漂出了 YOLO 的训练分布，点云会大面积走零阶保持。
        env.extras["yolo_detect_ratio"] = detected.float().mean()

        cam_pos_b, cam_quat_b = pose_in_frame(
            camera.data.pos_w, camera.data.quat_w_ros, base_pos_w, base_quat_w
        )

        candidate_pts, counts = mask_depth_to_pointcloud(
            mask=mask,
            depth=depth,
            K=camera.data.intrinsic_matrices,
            cam_pos=cam_pos_b,
            cam_quat=cam_quat_b,
            m_obj=m_obj,
            noise_std=noise_std,
        )

        env.extras["visible_ratio"] = (
            counts.float() / float(m_obj)
        ).clamp_(max=1.0).mean()

        if key in _DEBUG_ENABLED:
            _DEBUG_CAPTURE[key] = {
                "mask_pixels": mask.flatten(1).sum(dim=1).detach(),
                "visible_counts": counts.detach(),
                "depth": depth.detach(),
                "cam_pos_b": cam_pos_b.detach(),
                "cam_quat_b": cam_quat_b.detach(),
                "K": camera.data.intrinsic_matrices.detach(),
            }

        previous = _LAST_POINT_CLOUD.get(key)
        empty = counts == 0
        if bool(empty.any()):
            if previous is None:
                # ObservationManager 在 env 初始化时会调用一次每个观测函数来探测输出
                # shape，此时相机尚未完成首次渲染，RGB 是全零张量，YOLO 检不出任何东西。
                # 与 GT 路线保持一致：obj_pts 留 None，落到下面的零兜底，返回形状正确的
                # 零张量。这一帧不会进入任何 rollout。
                pass  # obj_pts remains None → zero fallback below
            else:
                candidate_pts = torch.where(
                    empty.view(-1, 1, 1), previous[:, :m_obj, :], candidate_pts
                )
                strikes = _EMPTY_MASK_STRIKES.get(key, 0) + 1
                _EMPTY_MASK_STRIKES[key] = strikes
                if strikes > _MAX_EMPTY_STRIKES:
                    # 连续空检超限：存一张 RGB 快照到 logs/ 供事后排查，然后清零
                    # 计数器继续训练，不崩溃。空检期间零阶保持点云，episode 会自然
                    # timeout；PhysX 短暂 GPU 抖动通常在几步内自愈。
                    bad = int(torch.argmax(empty.to(torch.uint8)))
                    _save_yolo_empty_rgb(
                        rgb_raw, key, bad, int(mask[bad].sum())
                    )
                    _EMPTY_MASK_STRIKES[key] = 0
                obj_pts = candidate_pts
        else:
            _EMPTY_MASK_STRIKES[key] = 0
            obj_pts = candidate_pts

    if obj_pts is None:
        previous = _LAST_POINT_CLOUD.get(key)
        if previous is not None:
            obj_pts = previous[:, :m_obj, :]
        else:
            obj_pts = torch.zeros(num_envs, m_obj, POINT_DIM, device=device)

    point_cloud = assemble_point_cloud(obj_pts, robot_pts)
    _LAST_POINT_CLOUD[key] = point_cloud.detach()

    if key in _DEBUG_ENABLED:
        _DEBUG_CAPTURE.setdefault(key, {})["points"] = point_cloud.detach()

    return point_cloud.reshape(num_envs, -1)


def invalidate_yolo_mask(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    """reset 时丢弃 YOLO 掩码缓存，强制下一帧重跑 YOLO（stage 4）。

    必须挂在 reset 事件上：reset 会把工件挪到新的随机位置，旧掩码指向的是上一
    episode 的像素区域，继续复用会框到空桌面 —— 而且**不会报错**，只是反投影出
    一朵位置错误的点云喂给策略。
    """
    _LAST_YOLO_MASK.pop(id(env), None)
    _YOLO_STEP_COUNTER.pop(id(env), None)


def point_bridge_point_cloud_yolo_cached(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg = SceneEntityCfg("camera_fixed"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    m_obj: int = M_OBJ,
    noise_std: float = NOISE_STD_M,
    yolo_weights: str = DEFAULT_YOLO_WEIGHTS,
    yolo_conf: float = YOLO_CONF,
    yolo_step: int = 5,
) -> torch.Tensor:
    """与 :func:`point_bridge_point_cloud_yolo` 相同，但每 ``yolo_step`` 步才跑一次 YOLO。

    **缓存的是掩码，不是深度**，这一点是本函数正确性的全部所在::

        每步：  当前深度帧 + 缓存掩码 → 反投影 → 点云      ← 观测持续更新
        每 N 步：当前 RGB → YOLO → 新掩码存入缓存

    如果连深度一起缓存（即"记录第一帧的深度信息，后续都用它采样"），策略每步会
    收到完全相同的静止点云，感知不到自己正在靠近工件 —— 那等价于部署脚本里的
    ``--freeze_depth``，文档明确标注它只用于开环对照实验。所以这里深度**每步都是
    新的**，只有"工件在图像里的哪些像素"这个判断被复用。

    ``yolo_step`` 的合理性：抓取前工件静止不动，掩码在连续几帧内几乎不变；夹爪
    逐渐遮挡工件带来的掩码变化在厘米量级以内，小于 FPS 下采样与 ``NOISE_STD_M``
    引入的不确定性。真机部署本来也是 50Hz 控制配 10Hz 的 YOLO 更新，训练侧这样做
    反而与部署更一致。

    reset 时缓存必须失效，见 :func:`invalidate_yolo_mask`。
    """
    camera = env.scene[camera_cfg.name]
    robot = env.scene[robot_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]

    device = env.device
    num_envs = env.num_envs
    key = id(env)

    base_pos_w = robot.data.root_pos_w
    base_quat_w = robot.data.root_quat_w

    # ── 夹爪关键点（与 GT / stage3 路线逐字一致）─────────────────────────────
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]
    ee_pos_b, ee_quat_b = pose_in_frame(ee_pos_w, ee_quat_w, base_pos_w, base_quat_w)
    robot_pts = gripper_keypoints(ee_pos_b, ee_quat_b, noise_std=0.0)

    depth_raw = camera.data.output.get("distance_to_image_plane")
    rgb_raw = camera.data.output.get("rgb")

    obj_pts = None
    if depth_raw is not None and rgb_raw is not None:
        depth = depth_raw[..., 0] if depth_raw.dim() == 4 else depth_raw

        # ── 掩码：每 yolo_step 步重跑，其余步复用缓存 ────────────────────────
        cached_mask = _LAST_YOLO_MASK.get(key)
        counter = _YOLO_STEP_COUNTER.get(key, 0)
        # 缓存形状必须与当前 batch 对齐 —— num_envs 变了（换 run / play 模式）就
        # 必须重跑，否则下面 mask_depth_to_pointcloud 会广播出静默的错误结果。
        stale = (
            cached_mask is None
            or cached_mask.shape[0] != num_envs
            or cached_mask.shape[-2:] != depth.shape[-2:]
        )
        if stale or counter <= 0:
            mask, detected = yolo_masks(rgb_raw, weights=yolo_weights, conf=yolo_conf)
            _LAST_YOLO_MASK[key] = mask
            _YOLO_STEP_COUNTER[key] = max(1, int(yolo_step)) - 1
            env.extras["yolo_detect_ratio"] = detected.float().mean()
            env.extras["yolo_ran"] = torch.ones((), device=device)
        else:
            mask = cached_mask
            _YOLO_STEP_COUNTER[key] = counter - 1
            env.extras["yolo_ran"] = torch.zeros((), device=device)

        cam_pos_b, cam_quat_b = pose_in_frame(
            camera.data.pos_w, camera.data.quat_w_ros, base_pos_w, base_quat_w
        )

        candidate_pts, counts = mask_depth_to_pointcloud(
            mask=mask,
            depth=depth,
            K=camera.data.intrinsic_matrices,
            cam_pos=cam_pos_b,
            cam_quat=cam_quat_b,
            m_obj=m_obj,
            noise_std=noise_std,
        )

        env.extras["visible_ratio"] = (
            counts.float() / float(m_obj)
        ).clamp_(max=1.0).mean()

        if key in _DEBUG_ENABLED:
            _DEBUG_CAPTURE[key] = {
                "mask_pixels": mask.flatten(1).sum(dim=1).detach(),
                "visible_counts": counts.detach(),
                "depth": depth.detach(),
                "cam_pos_b": cam_pos_b.detach(),
                "cam_quat_b": cam_quat_b.detach(),
                "K": camera.data.intrinsic_matrices.detach(),
            }

        previous = _LAST_POINT_CLOUD.get(key)
        empty = counts == 0
        if bool(empty.any()):
            if previous is None:
                pass  # 首次 shape 探测帧，落到下面的零兜底
            else:
                candidate_pts = torch.where(
                    empty.view(-1, 1, 1), previous[:, :m_obj, :], candidate_pts
                )
                strikes = _EMPTY_MASK_STRIKES.get(key, 0) + 1
                _EMPTY_MASK_STRIKES[key] = strikes
                if strikes > _MAX_EMPTY_STRIKES:
                    # 缓存掩码下持续空检，额外的可能原因是掩码过期（工件已移动而
                    # 缓存未失效）。强制下一帧重跑 YOLO，再存图记录。
                    _LAST_YOLO_MASK.pop(key, None)
                    _YOLO_STEP_COUNTER[key] = 0
                    bad = int(torch.argmax(empty.to(torch.uint8)))
                    _save_yolo_empty_rgb(
                        rgb_raw, key, bad, int(mask[bad].sum())
                    )
                    _EMPTY_MASK_STRIKES[key] = 0
                obj_pts = candidate_pts
        else:
            _EMPTY_MASK_STRIKES[key] = 0
            obj_pts = candidate_pts

    if obj_pts is None:
        previous = _LAST_POINT_CLOUD.get(key)
        if previous is not None:
            obj_pts = previous[:, :m_obj, :]
        else:
            obj_pts = torch.zeros(num_envs, m_obj, POINT_DIM, device=device)

    point_cloud = assemble_point_cloud(obj_pts, robot_pts)
    _LAST_POINT_CLOUD[key] = point_cloud.detach()

    if key in _DEBUG_ENABLED:
        _DEBUG_CAPTURE.setdefault(key, {})["points"] = point_cloud.detach()

    return point_cloud.reshape(num_envs, -1)


def privileged_state(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """critic 用的 privileged 状态，35 维（规格 §5.3）。

    布局：物体位姿 (7) ⊕ 末端位姿 (7) ⊕ 关节位置 (7) ⊕ 关节速度 (7) ⊕ 上一动作 (7)。
    位置分量均在机器人基座系。

    规格 §5.3 的字面要求是"critic 保留原观测不动"，但原观测是 1166 维双相机
    Theia 特征，本次已按用户决定整体移除，无法字面执行。这里按其**意图**实现：
    critic 拿未降级的完整状态。理由不变 —— critic 只在训练时存在、不需要部署，
    给它 privileged 信息不违反任何约束；而若 actor 与 critic 同时降级，一旦不
    收敛就无法区分是 actor 观测不足还是 critic 观测不足。这是本方案最重要的调试
    杠杆。实际上 GT 位姿比原来的视觉 critic 更强。
    """
    obj = env.scene[object_cfg.name]
    robot = env.scene[robot_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]

    base_pos_w = robot.data.root_pos_w
    base_quat_w = robot.data.root_quat_w

    obj_pos_b, obj_quat_b = pose_in_frame(
        obj.data.root_pos_w, obj.data.root_quat_w, base_pos_w, base_quat_w
    )
    ee_pos_b, ee_quat_b = pose_in_frame(
        ee_frame.data.target_pos_w[:, 0, :],
        ee_frame.data.target_quat_w[:, 0, :],
        base_pos_w,
        base_quat_w,
    )

    return torch.cat(
        [
            obj_pos_b,
            obj_quat_b,
            ee_pos_b,
            ee_quat_b,
            robot.data.joint_pos[:, :7],
            robot.data.joint_vel[:, :7],
            last_clipped_action(env)[:, :7],
        ],
        dim=-1,
    )


def privileged_state_no_vel(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """去掉关节速度的 privileged 状态，28 维。

    布局：物体位姿 (7) ⊕ 末端位姿 (7) ⊕ 关节位置 (7) ⊕ 上一动作 (7)。
    与 :func:`privileged_state` 的唯一差别是**不含 joint_vel**。

    为什么单独做一版：关节速度是**真机难以可靠获得**的一路 —— 编码器差分噪声大，
    滤波又引入滞后。若这一版与 35 维版学得一样好，说明速度对本任务并非必需，
    sim2real 时可以不依赖它；若明显变差，则说明策略确实在用速度做阻尼/预测，
    真机侧就必须把这一路做出来。

    另一个动机：``last_action`` 已经隐含了近期的运动趋势，与 joint_vel 存在信息
    重叠。去掉后观测更接近真机能稳定提供的量。
    """
    obj = env.scene[object_cfg.name]
    robot = env.scene[robot_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]

    base_pos_w = robot.data.root_pos_w
    base_quat_w = robot.data.root_quat_w

    obj_pos_b, obj_quat_b = pose_in_frame(
        obj.data.root_pos_w, obj.data.root_quat_w, base_pos_w, base_quat_w
    )
    ee_pos_b, ee_quat_b = pose_in_frame(
        ee_frame.data.target_pos_w[:, 0, :],
        ee_frame.data.target_quat_w[:, 0, :],
        base_pos_w,
        base_quat_w,
    )

    return torch.cat(
        [
            obj_pos_b,
            obj_quat_b,
            ee_pos_b,
            ee_quat_b,
            robot.data.joint_pos[:, :7],
            last_clipped_action(env)[:, :7],
        ],
        dim=-1,
    )


# ──────────────────────────────────────────────────────────────────────────────
# 阶段 1：绕过相机的 GT 采点（规格 §7 阶段 1）
# ──────────────────────────────────────────────────────────────────────────────

_MESH_POINTS_CACHE: dict = {}


def _object_surface_points_local(env: ManagerBasedRLEnv, m_obj: int) -> torch.Tensor:
    """从场景里实际的物体 USD 网格采 ``m_obj`` 个表面点，返回物体局部系坐标。

    刻意**不用** ``xarm_va/assets/data/*.npz`` 里那两份预采点：它们是对
    ``workpiece_m.usd`` 采的，而本项目场景装的是 ``gongjian.usd``，两者网格不同，
    直接拿来会让阶段 1 的"信息上限"建立在错误几何上 —— 而阶段 1 的全部意义就是
    排除几何以外的因素，用错网格会让这个基准失去意义。

    顶点按 x 坐标等间距抽取（确定性，无 RNG），再交给 FPS 摊匀。顶点不是按面积
    均匀的，但阶段 1 只需要一个"位置信息完整、无遮挡、无噪声"的上界参照，顶点
    分布的轻微不均匀不影响这个用途。
    """
    key = (id(env), m_obj)
    cached = _MESH_POINTS_CACHE.get(key)
    if cached is not None:
        return cached

    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    root_path = OBJECT_PRIM_PATTERN.format(env=0)
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        raise RuntimeError(f"Object prim not found at {root_path!r} for stage-1 GT sampling.")

    root_inv = UsdGeom.Xformable(root_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    ).GetInverse()

    collected: list[torch.Tensor] = []
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        if not pts:
            continue
        # mesh 局部 → 世界 → 物体根局部
        to_obj = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        ) * root_inv
        collected.append(
            torch.tensor(
                [tuple(to_obj.Transform(p)) for p in pts],
                dtype=torch.float32,
                device=env.device,
            )
        )

    if not collected:
        raise RuntimeError(f"No UsdGeom.Mesh found under {root_path!r} for stage-1 GT sampling.")

    verts = torch.cat(collected, dim=0)
    # 确定性等间距抽取到 FPS 候选上限，再 FPS 摊匀
    cap = min(verts.shape[0], FPS_CANDIDATE_CAP)
    step = max(verts.shape[0] // cap, 1)
    cand = verts[::step][:cap].unsqueeze(0)
    idx = farthest_point_sample(cand, min(m_obj, cand.shape[1]))
    sampled = cand[0, idx[0]]
    if sampled.shape[0] < m_obj:   # 网格顶点少于 m_obj：有放回补齐
        pad = sampled[torch.arange(m_obj, device=env.device) % sampled.shape[0]]
        sampled = pad
    _MESH_POINTS_CACHE[key] = sampled
    return sampled


def point_bridge_point_cloud_gt(
    env: ManagerBasedRLEnv,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    m_obj: int = M_OBJ,
    noise_std: float = 0.0,
) -> torch.Tensor:
    """规格 §7 阶段 1：绕过相机，用 GT 物体位姿把网格表面点变换到基座系。

    这是该表征的**信息上限** —— 无遮挡、无掩码误差、无深度噪声、全表面可见。

    验收：训练奖励曲线应与原版接近。若这一阶段就学不出来，说明点表征无法表达该
    任务，**不要往下走**，回到规格 §8 的位姿回归测试。

    默认 ``noise_std=0``（规格 §7 阶段 1 明确"不加噪声"）。
    """
    robot = env.scene[robot_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    obj = env.scene[object_cfg.name]

    base_pos_w = robot.data.root_pos_w
    base_quat_w = robot.data.root_quat_w
    num_envs = env.num_envs

    local = _object_surface_points_local(env, m_obj)              # (m_obj, 3)
    local = local.unsqueeze(0).expand(num_envs, m_obj, 3)

    # 物体局部 → 世界：先按 GT 姿态旋转，再加 GT 位置
    obj_quat = obj.data.root_quat_w.unsqueeze(1).expand(num_envs, m_obj, 4)
    pts_w = quat_apply(obj_quat, local) + obj.data.root_pos_w.unsqueeze(1)

    # 世界 → 基座系
    base_quat_exp = base_quat_w.unsqueeze(1).expand(num_envs, m_obj, 4)
    pts_b = quat_apply_inverse(base_quat_exp, pts_w - base_pos_w.unsqueeze(1))

    if noise_std > 0.0:
        pts_b = pts_b + torch.randn_like(pts_b) * noise_std

    obj_pts = pts_b

    ee_pos_b, ee_quat_b = pose_in_frame(
        ee_frame.data.target_pos_w[:, 0, :],
        ee_frame.data.target_quat_w[:, 0, :],
        base_pos_w,
        base_quat_w,
    )
    # 夹爪点不加噪声（参考实现只对 object points 加噪）
    robot_pts = gripper_keypoints(ee_pos_b, ee_quat_b, noise_std=0.0)

    point_cloud = assemble_point_cloud(obj_pts, robot_pts)
    key = id(env)
    _LAST_POINT_CLOUD[key] = point_cloud.detach()
    if key in _DEBUG_ENABLED:
        _DEBUG_CAPTURE.setdefault(key, {})["points"] = point_cloud.detach()
    return point_cloud.reshape(num_envs, -1)


def set_privileged_baseline(env_cfg, keep_appearance: bool = False) -> None:
    """privileged base：**actor 与 critic 都吃 GT 状态**，彻底移除点云与相机。

    这是整条链路的**最上界**，比阶段 1 还高一层：

        privileged base   状态真值 → 不受任何表征限制
        阶段 1            点表征 + GT 位姿 → 受点表征限制，不受感知限制
        阶段 2            点表征 + 掩码深度 → 两者都受限

    **用途：把"环境/奖励能不能学"与"表征够不够用"彻底分开。**

    这一版若学不出来，问题一定在环境本身 —— 奖励权重、动作限幅（±0.6°/step）、
    episode 长度、碰撞终止阈值、初始位姿分布 —— 而与点云、相机、外参、分割全都
    无关。反之若这一版学得很好而阶段 1 崩，才轮到怀疑点表征。

    没有这一版的话，阶段 1 不收敛会有两种无法区分的解释：点表征不足，或环境根本
    不可解。规格 §9 的失败归因树默认环境是可解的（因为它假定"原版"已经跑通），
    但我们换了相机、换了观测、换了网络，这个前提值得单独验证一次。

    观测：``policy`` 与 ``critic`` 两组都只含 :func:`privileged_state`（35 维）。
    两组内容相同但都保留，这样 rsl_rl 的 ``obs_groups`` 映射不用为这一版特殊处理，
    且 actor/critic 各自的归一化统计互不干扰。

    奖励、动作空间与限幅、终止条件、物理参数、场景（除相机）全部不动 —— 这是本
    版有意义的前提：它必须与阶段 1/2 面对**同一个** MDP，只有观测不同。
    """
    # 点云项整个去掉。ObservationManager 会跳过 None 的 term。
    env_cfg.observations.policy.point_cloud = None
    # 腕部点云同理。默认本来就是 None，这里显式写一遍是为了防"先调 stage 10 再调本
    # setter"的调用顺序留下一路孤立的腕部观测（那会让 privileged base 不再是纯 GT）。
    env_cfg.observations.policy.wrist_point_cloud = None
    # 关节角也去掉：privileged_state 里已含关节位置与速度，留着就是重复通道。
    env_cfg.observations.policy.joint_pos = None
    # actor 直接吃与 critic 相同的 GT 状态
    env_cfg.observations.policy.state = ObsTerm(func=privileged_state)

    # 无任何消费者读相机 → 摘掉，省下每步 640x480xN_env 的渲染。
    # --enable_cameras 仍必须开：程序化桌面的 PreviewSurface 材质经 MDL 创建，
    # 只在渲染型 Kit experience 下注册。
    env_cfg.scene.camera_fixed = None
    # 腕部相机同理（默认已是 None，显式写一遍以防调用顺序意外把它装上）。
    env_cfg.scene.camera_wrist = None
    env_cfg.events.invalidate_instance_lut = None

    # 外观类 DR 全部摘掉：本版观测是纯 GT 状态向量，一个像素都不读，贴图与光照
    # 对观测的影响严格为零，留着就是每次 reset 白跑一遍材质重绑定 / USD 属性写入。
    #
    # 与阶段 1/2 的差别仅限**渲染外观**，MDP（奖励、动作、终止、物理）完全不变，
    # 因此不破坏"三档面对同一个 MDP"这个前提 —— 那正是本版可比性的基础。
    #
    # 注意 apply_table_texture 是幂等的（材质只在首次为某 env 建一次），所以摘掉它
    # 省下的主要是 reset 时的字典查表与 USD 遍历，不是大头；真正的大头是相机，
    # 而相机在上面已经摘掉了。因此 keep_appearance=True 的额外开销可以忽略。
    #
    # keep_appearance：``--play`` 回放时置 True。回放**就是为了看画面**，桌面没
    # 贴图、光照恒定会让它和阶段 1/2 的视觉效果对不上，也没法和真机比对。回放是
    # 单环境、不训练的场景，那点开销无所谓。
    if not keep_appearance:
        env_cfg.events.apply_table_texture = None
        env_cfg.events.randomize_light_intensity = None


def set_privileged_baseline_no_vel(env_cfg, keep_appearance: bool = False) -> None:
    """privileged base 的**无关节速度**变体：actor 与 critic 都吃 28 维 GT 状态。

    与 :func:`set_privileged_baseline` 的唯一差别是观测函数换成
    :func:`privileged_state_no_vel`（少了 joint_vel 那 7 维），其余（摘相机、摘
    点云、摘关节角、外观 DR）完全一致。

    **用途：单独测量关节速度对本任务的贡献。** 与 stage -1 并排训练，两条曲线
    的差就是速度那 7 维带来的收益。这个数值直接决定 sim2real 时要不要在真机上
    把关节速度做出来 —— 编码器差分噪声大、滤波有滞后，能不依赖它最好。

    MDP（奖励、动作、终止、物理、场景）与 stage -1 / 1 / 2 完全相同，只有观测
    不同 —— 这是两版可比的前提。
    """
    set_privileged_baseline(env_cfg, keep_appearance=keep_appearance)
    # 复用上面的全部摘除逻辑，只把观测函数换掉。两组都换：本版的定位是
    # "整条链路在无速度信息下的上界"，critic 保留速度会让它测不到想测的东西。
    env_cfg.observations.policy.state = ObsTerm(func=privileged_state_no_vel)
    env_cfg.observations.critic.state = ObsTerm(func=privileged_state_no_vel)


def set_critic_joint_vel_enabled(env_cfg, enabled: bool) -> None:
    """开关 critic 那 7 维关节速度。只动 critic，policy 组本来就不含速度。

    stage 1/2 的 actor 观测是"点云 + 关节角"，从来没有关节速度（见 PolicyCfg
    的说明）。速度唯一的入口是 critic 的 :func:`privileged_state`（35 维）。
    因此"stage 2 无速度版"= 把 critic 换成 28 维的
    :func:`privileged_state_no_vel`，actor 一个字节都不用改。

    critic 只在训练时用来算优势函数，推理时不参与 —— 所以这个开关不影响部署，
    改的是"训练信号里允不允许出现真机拿不到的量"。

    对 stage -1/-2 调用无意义（那两版的 actor/critic 由各自的 setter 成对设定），
    调用方应只在 stage 1/2 上使用。
    """
    env_cfg.observations.critic.state = ObsTerm(
        func=privileged_state if enabled else privileged_state_no_vel
    )


def set_grasp_target_offset_cm(env_cfg, offset_cm: float) -> None:
    """把抓取目标点从工件根原点抬到其**世界系正上方** ``offset_cm`` 厘米。

    ``offset_cm=0`` 即历史行为（瞄工件根原点），此时不给任何项写 params，
    数值上逐位等同于改动前。

    实现上偏移是在**工件局部系**下的 -z（见
    :func:`custom_mdp.grasp_target_pos_w` 的推导：场景里 object.init_state.rot =
    [0,1,0,0] 把局部 +z 翻成了世界 -z），所以偏移会随工件 yaw 随机化一起转 ——
    这正是要的：目标点始终在工件正上方，而不是在某个固定世界方向上。

    一次性改**三处**，它们必须同步：
      1. reaching_coarse / mid / fine 三个距离奖励；
      2. reach_success_bonus（虽然默认 weight=0，但一旦调权重就要对齐）；
      3. terminations.reach_success（若已由 set_success_termination_enabled 开启）。
    漏掉任何一处都会导致"奖励在爬 A 点、success 在判 B 点"，训练曲线好看但
    success 永远上不去。调用顺序要求：**在 set_success_termination_enabled 之后**
    调用，否则那个 setter 会新建一个不带偏移的 DoneTerm 把这里的设置盖掉。

    姿态类奖励（ee_orientation_*）刻意不加偏移：它们比的是四元数夹角，与目标点
    平移无关。碰撞惩罚、action_rate 同理。
    """
    offset_local = (0.0, 0.0, -offset_cm / 100.0)

    for name in ("reaching_coarse", "reaching_mid", "reaching_fine",
                 "reach_success_bonus"):
        term = getattr(env_cfg.rewards, name, None)
        if term is not None:
            term.params["target_offset_local"] = offset_local

    done_term = getattr(env_cfg.terminations, "reach_success", None)
    if done_term is not None:
        done_term.params["target_offset_local"] = offset_local


def disable_point_cloud_noise(env_cfg) -> None:
    """把**所有**点云观测项的 ``noise_std`` 归零（回放 / 阶段 0 可视化用）。

    为什么不能只靠 ``XArm7PickPointCloudPlayEnvCfg.__post_init__``：那里只能碰
    ``point_cloud``，因为 ``wrist_point_cloud`` 在 ``__post_init__`` 跑的时候还是
    ``None`` —— 它是 :func:`set_observation_stage` 之后才挂上的。于是 stage 10 回放
    会出现"全局点无噪声、腕部点带 1cm 噪声"的错配，而验收标准是"点贴在表面，容差
    约 3 sigma"，带噪声的那一路看着就像外参标错了。

    调用时机：**set_observation_stage 之后**。对不存在的项静默跳过。
    """
    for name in ("point_cloud", "wrist_point_cloud"):
        term = getattr(env_cfg.observations.policy, name, None)
        if term is not None and "noise_std" in term.params:
            term.params["noise_std"] = 0.0


def set_observation_stage(env_cfg, stage: int, keep_appearance: bool = False) -> None:
    """按规格 §7 切换观测来源。就地修改 ``env_cfg``。

    - ``stage=-2``：privileged base 的**无关节速度**变体（28 维），用于测量速度
      那一路的贡献。见 :func:`set_privileged_baseline_no_vel`。
    - ``stage=-1``：**privileged base** —— actor 与 critic 都吃 GT 状态，无点云、
      无相机。用于验证环境/奖励本身可解，见 :func:`set_privileged_baseline`。
    - ``stage=0``：完整相机管线，噪声关闭（只做可视化，要看点是否贴在表面）。
    - ``stage=1``：绕过相机的 GT 采点，无噪声（点表征的信息上限）。
    - ``stage=2``：完整相机管线 + sigma=1cm 噪声（规格默认）。相机只保留深度与
      实例分割两路，RGB 摘掉（训练路径无消费者，省约 1/3 相机显存）。
    - ``stage=3``：与 stage 2 相同，但**掩码来自 YOLO-seg 而非 Isaac 实例分割**
      （规格 §11.1/§11.2）。相机改为深度 + RGB，实例分割整路摘掉。每步要跑一次
      YOLO（实测约 4.3ms/env，几乎不随 batch 摊薄），故适合小 num_envs 微调，
      不适合从头训。
    - ``stage=10``：stage 2 **加一路腕部相机**。全局相机仍走 GT 实例分割出物体点，
      腕部相机不做分割、在整帧深度里采 64 点（见 ``M_WRIST`` 的说明）。观测
      217 → 409 维，网络侧多一个独立权重的腕部 PointNet。编号跳到 10 而不是接
      5：3/4 是"换掩码来源"这条 sim2real 支线，10 开的是"加传感器"这条新支线，
      两者互不依赖，将来 10 也可以再叠 YOLO。

    ``keep_appearance`` 仅对 ``stage=-1``/``-2`` 有意义：回放时保留桌面贴图与光照
    DR，使画面与阶段 1/2 一致。训练时保持 False。
    """
    if stage not in (-2, -1, 0, 1, 2, 3, 4, 10):
        raise ValueError(f"stage must be -2, -1, 0, 1, 2, 3, 4 or 10; got {stage}")

    if stage == -2:
        set_privileged_baseline_no_vel(env_cfg, keep_appearance=keep_appearance)
        return

    if stage == -1:
        set_privileged_baseline(env_cfg, keep_appearance=keep_appearance)
        return

    term = env_cfg.observations.policy.point_cloud
    if stage == 1:
        term.func = point_bridge_point_cloud_gt
        term.params = {
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "robot_cfg": SceneEntityCfg("robot"),
            "object_cfg": SceneEntityCfg("object"),
            "m_obj": M_OBJ,
            "noise_std": 0.0,
        }
        # 阶段 1 没有任何消费者读相机，留着它就是每步白渲染一份 640x480xN_env。
        # 直接从场景里摘掉 —— xarm_va 的 XArmReachSceneNoCameraCfg 是同一手法
        # （那里的理由是 1024 env 的 tiled camera 会吃光 32 GB 显存）。
        #
        # 注意 --enable_cameras 仍然必须开：程序化桌面用的 PreviewSurface 材质
        # 经由 MDL 创建，只在渲染型 Kit experience 下才注册，否则材质 prim 返回
        # null、资产创建直接失败。
        env_cfg.scene.camera_fixed = None
        # 实例 ID 查表事件只对相机路线有意义，一并去掉（它本身无害，只是 pop 一个
        # 字典键，但留着会让"阶段 1 不依赖分割"这件事显得含糊）。
        env_cfg.events.invalidate_instance_lut = None
    elif stage == 3:
        # 掩码来源换成 YOLO-seg（规格 §11.1/§11.2）。反投影及其下游一律复用
        # GT 路线那一份，见 point_bridge_point_cloud_yolo 的 docstring。
        term.func = point_bridge_point_cloud_yolo
        term.params = {
            "camera_cfg": SceneEntityCfg("camera_fixed"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "robot_cfg": SceneEntityCfg("robot"),
            "m_obj": M_OBJ,
            "noise_std": NOISE_STD_M,
            "yolo_weights": DEFAULT_YOLO_WEIGHTS,
            "yolo_conf": YOLO_CONF,
        }
        # **实例分割整路摘掉** —— YOLO 只要 RGB，分割图没有任何消费者了。
        # 与 stage 2 相比是"两路换两路"（深度+分割 → 深度+RGB），相机显存持平，
        # 省下的是分割 AOV 的渲染开销。colorize_instance_id_segmentation、
        # instance LUT 那些坑也一并不用管。
        env_cfg.scene.camera_fixed.data_types = [
            "distance_to_image_plane",
            "rgb",
        ]
        # LUT 失效事件只服务实例分割，这条路线上它已无意义。
        env_cfg.events.invalidate_instance_lut = None
    elif stage == 4:
        # 与 stage 3 完全同路，只是每 yolo_step 步才跑一次 YOLO，中间步复用缓存
        # 掩码配当前深度帧。缓存掩码而非深度 —— 详见函数 docstring。
        term.func = point_bridge_point_cloud_yolo_cached
        term.params = {
            "camera_cfg": SceneEntityCfg("camera_fixed"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "robot_cfg": SceneEntityCfg("robot"),
            "m_obj": M_OBJ,
            "noise_std": NOISE_STD_M,
            "yolo_weights": DEFAULT_YOLO_WEIGHTS,
            "yolo_conf": YOLO_CONF,
            "yolo_step": DEFAULT_YOLO_STEP,
        }
        env_cfg.scene.camera_fixed.data_types = [
            "distance_to_image_plane",
            "rgb",
        ]
        env_cfg.events.invalidate_instance_lut = None
        # **必须**挂上掩码失效事件：reset 会把工件挪到新位置，旧掩码指向上一
        # episode 的像素区域，复用会静默地反投影出位置错误的点云。
        env_cfg.events.invalidate_yolo_mask = EventTerm(
            func=invalidate_yolo_mask, mode="reset"
        )
    else:
        term.func = point_bridge_point_cloud
        term.params = {
            "camera_cfg": SceneEntityCfg("camera_fixed"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "robot_cfg": SceneEntityCfg("robot"),
            "m_obj": M_OBJ,
            "noise_std": 0.0 if stage == 0 else NOISE_STD_M,
        }
        if stage in (2, 10):
            # 训练路径**没有任何 RGB 消费者** —— 观测是深度反投影出的 xyz 点，
            # 读 RGB 的只有 _run_stage0_visualization()（拿它当投影底图）。
            # 每一路 data_type 各占一份 640x480xN_env 的 GPU 缓冲，摘掉 RGB
            # 直接省下约 1/3 相机显存，决定 num_envs 能开多大。
            #
            # 不对 stage 0 生效：那一阶段要靠 RGB 看点是否贴在物体表面。
            env_cfg.scene.camera_fixed.data_types = [
                "distance_to_image_plane",
                "instance_id_segmentation_fast",
            ]

        if stage == 10:
            # 腕部相机：默认场景里是 None（camfix 那份被显式摘掉了），只在本阶段
            # 装回来，所以 stage -2..4 一份腕部渲染都不会白跑。
            env_cfg.scene.camera_wrist = make_wrist_pointcloud_camera_cfg()
            env_cfg.observations.policy.wrist_point_cloud = ObsTerm(
                func=point_bridge_wrist_point_cloud,
                params={
                    "camera_cfg": SceneEntityCfg("camera_wrist"),
                    "robot_cfg": SceneEntityCfg("robot"),
                    "m_wrist": M_WRIST,
                    "noise_std": NOISE_STD_M,
                    "min_depth_m": 0.0,
                },
            )


# ──────────────────────────────────────────────────────────────────────────────
# 阶段 0 可视化支持（规格 §7 阶段 0）
# ──────────────────────────────────────────────────────────────────────────────

def debug_capture(env: ManagerBasedRLEnv) -> dict | None:
    """打开中间量记录并返回上一帧的快照；数据未就绪时返回 ``None``。

    首次调用只是打开开关（观测函数下一帧才会写入），因此返回 ``None`` 是正常的。
    仅用于阶段 0，训练路径不调用，不占显存。
    """
    key = id(env)
    _DEBUG_ENABLED.add(key)
    cap = _DEBUG_CAPTURE.get(key)
    if not cap or "points" not in cap:
        return None

    out = dict(cap)
    if "visible_counts" in cap:
        out["visible_ratio"] = (cap["visible_counts"].float() / float(M_OBJ)).clamp_(max=1.0)
    if "depth" in cap:
        # 深度图转成可 imshow 的形式：0（无效）留作最小值，其余线性映射
        d = cap["depth"].clone()
        finite = torch.isfinite(d) & (d > 0)
        if bool(finite.any()):
            lo = d[finite].min()
            hi = d[finite].max()
            d = torch.where(finite, (d - lo) / (hi - lo).clamp(min=1e-6), torch.zeros_like(d))
        out["depth_vis"] = d.cpu().numpy()
    return out


def project_points_to_pixels(
    env: ManagerBasedRLEnv,
    points_base: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把基座系点投影回固定相机的像素坐标，供阶段 0 目视核对。

    这是反投影的逆运算，因此能捕捉到"外参换算错 / 相机约定不匹配"这类错误：若
    投影回去的点没有落在渲染图里物体所在的位置，说明前向管线就是错的。

    Args:
        points_base: 基座系点，形状 (B, P, 3)。

    Returns:
        ``(uv, in_front)``：``uv`` 形状 (B, P, 2) 像素坐标；``in_front`` 形状
        (B, P) 标记相机前方（z > 0）的点，投影只对这些点有意义。
    """
    cap = _DEBUG_CAPTURE.get(id(env)) or {}
    if "cam_pos_b" not in cap:
        raise RuntimeError(
            "Camera pose has not been captured yet. Call debug_capture(env) and step the "
            "environment once before projecting."
        )

    cam_pos_b = cap["cam_pos_b"]
    cam_quat_b = cap["cam_quat_b"]
    K = cap["K"]

    B, P, _ = points_base.shape
    quat = cam_quat_b.unsqueeze(1).expand(B, P, 4)
    # 基座系 → 相机光学系（cam pose 的逆变换）
    pts_cam = quat_apply_inverse(quat, points_base - cam_pos_b.unsqueeze(1))

    z = pts_cam[..., 2]
    in_front = z > 1.0e-6
    z_safe = torch.where(in_front, z, torch.ones_like(z))

    if K.dim() == 2:
        K = K.unsqueeze(0).expand(B, 3, 3)
    fx = K[:, 0, 0].unsqueeze(-1)
    fy = K[:, 1, 1].unsqueeze(-1)
    cx = K[:, 0, 2].unsqueeze(-1)
    cy = K[:, 1, 2].unsqueeze(-1)

    u = pts_cam[..., 0] / z_safe * fx + cx
    v = pts_cam[..., 1] / z_safe * fy + cy
    return torch.stack([u, v], dim=-1), in_front


# ──────────────────────────────────────────────────────────────────────────────
# 场景
# ──────────────────────────────────────────────────────────────────────────────

@configclass
class XArm7PickPointCloudSceneCfg(XArm7PickLiftCubeSceneCfg):
    """camfix 场景，换标定相机、默认去掉腕部相机、工件初始位沿基座 +x 平移 5cm。

    腕部相机默认移除的理由：stage -2..4 的点云只由固定相机产生，Theia 已整体移除，
    没有任何消费者读它的 RGB。留着就是每步白渲染一份 224x224x64env。

    stage 10 会把它装回来（:func:`make_wrist_pointcloud_camera_cfg`，只出深度），
    因此这里保持 ``None`` 而不是删掉字段 —— 有这个字段在，装回来只是一次赋值。

    工件初始位在**本类里覆盖**而不是去改 camfix 的 ``object``：那一份被 pose /
    nopc / vision 等多个配置共享，就地改会连带挪动那些实验的工件，且让它们已有的
    checkpoint 与配置对不上 —— 而这两件事都不报错，只是结果悄悄变了。
    """

    camera_wrist = None
    camera_fixed: TiledCameraCfg = make_pointcloud_camera_cfg()

    # env 系 y 减小 = 基座系 x 增大（基座 rot=(0.707,0,0,-0.707)，绕 z -90°），
    # 所以 -0.44 → -0.49 就是"基座 +x 方向平移 5cm"。搞反方向不会报错，工件只是
    # 挪到了离基座更近的地方，看曲线看不出来。
    #
    # 其余字段（rot / spawn / prim_path）必须与 camfix 的 object 逐字一致 ——
    # RigidObjectCfg 是整体替换而非字段级合并，漏写 spawn 会 spawn 不出网格。
    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.0, -0.49, 0.833],
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


# ──────────────────────────────────────────────────────────────────────────────
# 观测
# ──────────────────────────────────────────────────────────────────────────────

@configclass
class XArm7PickPointCloudObservationsCfg:
    """policy 217 维（物体点 192 ⊕ 夹爪点 18 ⊕ 关节角 7），critic 35 维 privileged。

    点云每点仅 xyz（无类型通道），布局 ``[物体 64*3 | 夹爪 6*3 | 关节角 7]``。
    网络侧按前两段各过一次同一个 PointNet，两个 embedding 拼接后再接 MLP。

    policy 组里**不含**（规格 §3.1，严格对齐 Point Bridge 的 H=1 设定）：
    关节速度、上一动作、夹爪开合度、物体速度、力/力矩、观测历史堆叠。

    若出现动作抖动 / 目标附近震荡 / 学出过度保守的慢速解，按规格 §10 在此处补
    低维向量（第一优先关节速度，第二优先上一动作），**不要提高 H** —— 两帧点云
    拼成更大点集会破坏语义（PointNet 置换不变，分不清哪帧）。补的维度会自动被
    PointNetActorCritic 拼到 PointNet 输出之后，无需改网络代码。

    stage 10 会额外挂上 ``wrist_point_cloud``（腕部整帧采样 64x3），布局变成
    ``[物体 192 | 夹爪 18 | 腕部 192 | 关节角 7]`` = 409 维，网络侧多一个独立权重的
    腕部 encoder。默认关闭，因此 stage -2..4 的维度与行为逐位不变。

    ``joint_angles`` 本身就是相对原工作的必要偏离：原工作动作空间是末端位姿，
    夹爪关键点已完整编码末端状态；本任务动作空间是关节角，而 7 自由度臂存在零
    空间（同一末端位姿对应无穷多关节构型），不给关节角会导致"指令关节但观测不到
    关节"的不自洽。
    """

    @configclass
    class PolicyCfg(ObsGroup):

        # 顺序要紧：点云必须在最前且**物体点在夹爪点之前**，PointNetActorCritic
        # 按 OBJ_FEATURE_LEN / ROBOT_FEATURE_LEN 依次切出两段，剩下的当低维向量。
        point_cloud = ObsTerm(
            func=point_bridge_point_cloud,
            params={
                "camera_cfg": SceneEntityCfg("camera_fixed"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "robot_cfg": SceneEntityCfg("robot"),
                "m_obj": M_OBJ,
                "noise_std": NOISE_STD_M,
            },
        )

        # stage 10 专用（腕部相机整帧采样点），默认关闭。
        #
        # 位置在 point_cloud **之后**、joint_pos **之前**，因为观测项的声明顺序就是
        # 扁平布局，而网络按固定偏移切分：[物体 | 夹爪 | 腕部 | 低维向量]。挪到
        # joint_pos 后面会让网络把 7 维关节角当成腕部点的一部分去 reshape，
        # 维度对不上时报形状错，维度恰好对上时**静默算错**。
        #
        # 与 state 同理写成显式字段而不是运行时挂属性：顺序必须在类定义里可读可审。
        wrist_point_cloud = None

        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint[1-7]"])},
        )

        # privileged base（stage=-1）专用，默认关闭。声明成显式字段而不是运行时
        # 动态挂属性：观测项的**顺序**决定扁平布局，而网络按固定偏移切分点云，
        # 顺序必须在类定义里写死、可读可审。
        state = None

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):

        state = ObsTerm(func=privileged_state)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# ──────────────────────────────────────────────────────────────────────────────
# 事件
# ──────────────────────────────────────────────────────────────────────────────

@configclass
class XArm7PickPointCloudEventCfg:
    """camfix 的 DR 减去相机位姿随机化与图像增强。

    相机位姿 DR 关闭：规格 §2.1 的消融显示视角匹配时三任务 23/30、21/30、24/30，
    视角随机化后掉到 12/30、12/30、18/30（平均约 47%）。而且点云是在基座系下的
    绝对米制坐标，抖动相机等于直接给几何加系统性偏差，与 §4.4 的零均值噪声不是
    一回事。

    图像增强移除：那是给 Theia 的 RGB 用的，本配置不出 RGB。掩码级增强
    （腐蚀/膨胀/随机切块）属于规格 §11.2 的后续阶段，本版不加。

    桌面颜色/纹理 DR 已移除（2026-08-12）：本配置不出 RGB，观测只来自深度 + 实例
    分割，桌面外观对观测零影响，留着纯属每次 reset 白跑一遍材质重绑定。光照 DR
    保留 —— 它同样不影响深度与分割，但成本极低，且将来若加回 RGB 分支不用重接。
    """

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # 工件随机化：0.25x0.25 方框，中心 = object.init_state（见 SceneCfg 里的覆盖，
    # 基座系 x=+0.49）。2026-08-14 从 0.20x0.20 @ base x=+0.44 扩到这一组，理由是
    # 真机实测工件常摆在基座 x 更远处，原范围压不住。
    #
    # 改这里必须同步改 SceneCfg 的 object.init_state —— 随机化是**相对
    # default_root_state 的偏移**（reset_table_height_and_object_pose 里
    # `root_state[:, 0] += uniform(*x_range)`），范围和中心是两个独立的旋钮，
    # 只改一个只会得到"框大了但还偏在老地方"。
    #
    # 与旧 checkpoint 不兼容：model_12000.pt 是在 0.20x0.20 @ x=+0.44 上训的，
    # 必须 stage -1 与 stage 2 都重训。
    reset_table_and_object = EventTerm(
        func=reset_table_height_and_object_pose,
        mode="reset",
        params={
            "x_range": (-0.125, 0.125),
            "y_range": (-0.125, 0.125),
            "yaw_range": (-math.pi / 2, math.pi / 2),
            "height_range": _TABLE_HEIGHT_RAND,
        },
    )

    # 桌面颜色/纹理随机化已移除（2026-08-12，用户要求），改为固定 1:1 贴图。
    # 本配置的观测是深度 + 实例分割反投影出的点云，不含 RGB，桌面外观对观测没有
    # 任何影响 —— 贴图纯粹是为了视口 / sim-real 对比时看着与真机一致。
    apply_table_texture = EventTerm(func=apply_table_texture, mode="reset")

    randomize_light_intensity = EventTerm(func=randomize_light_intensity, mode="reset")

    # 规格 §4.1③：实例 ID 每次 reset 后重查，不要硬编码。
    invalidate_instance_lut = EventTerm(func=invalidate_instance_lut, mode="reset")


# ──────────────────────────────────────────────────────────────────────────────
# 环境
# ──────────────────────────────────────────────────────────────────────────────

@configclass
class XArm7PickPointCloudEnvCfg(ManagerBasedRLEnvCfg):
    """点云观测环境：actor 吃 287 维点表征，critic 吃 35 维 privileged 状态。"""

    scene: XArm7PickPointCloudSceneCfg = XArm7PickPointCloudSceneCfg(
        num_envs=64, env_spacing=2.5
    )
    observations: XArm7PickPointCloudObservationsCfg = XArm7PickPointCloudObservationsCfg()
    actions: XArm7PickLiftCubeActionsCfg = XArm7PickLiftCubeActionsCfg()
    events: XArm7PickPointCloudEventCfg = XArm7PickPointCloudEventCfg()
    rewards: XArm7PickLiftCubeRewardsCfg = XArm7PickLiftCubeRewardsCfg()
    terminations: XArm7PickLiftCubeTerminationsCfg = XArm7PickLiftCubeTerminationsCfg()
    curriculum: XArm7PickLiftCubeCurriculumCfg = XArm7PickLiftCubeCurriculumCfg()

    def __post_init__(self):
        # 与 camfix 逐项一致（规格 §0：物理参数与算法配置不动）。
        self.decimation = 2
        self.episode_length_s = 24.0

        self.sim.dt = 0.01
        # 渲染间隔与控制步长对齐，否则会拿到上一帧的深度（规格 §4.1④）。
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 8
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.physx.gpu_max_rigid_patch_count = 1024 * 1024


@configclass
class XArm7PickPointCloudPlayEnvCfg(XArm7PickPointCloudEnvCfg):
    """回放 / 阶段 0 可视化：单环境，关闭观测噪声。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
        # 阶段 0 验收要看点是否贴在物体表面，容差 ≈ NOISE_STD x 3。带噪声看不清，
        # 因此回放时关掉。
        #
        # 腕部点（stage 10）**管不到**：那个观测项要等 set_observation_stage 才存在，
        # 此刻还是 None。调用方在 set_observation_stage 之后再调一次
        # disable_point_cloud_noise(cfg) 才能把两路都归零。
        self.observations.policy.point_cloud.params["noise_std"] = 0.0


# ──────────────────────────────────────────────────────────────────────────────
# stage2 遮挡实验（仿真专用；不改动任何既有 stage 的分支）
# ──────────────────────────────────────────────────────────────────────────────

_OBJECT_LOCAL_BOUNDS_CACHE: dict = {}


def _object_local_bounds(env: ManagerBasedRLEnv) -> tuple[torch.Tensor, torch.Tensor]:
    """读物体 USD 网格顶点，返回工件局部包围盒 ``(min, max)``，形状各 (3,)。

    遮挡谓词需要"工件的局部范围"来把 severity 锚定成"工件尺寸的比例"（切 25% /
    球半径 50% 外接球），而不是绝对米数。与阶段 1 的
    :func:`_object_surface_points_local` 同源（都读同一份 gongjian.usd 顶点、转到
    物体根局部系），但这里取**全部顶点的精确 min/max**，而不是 FPS 抽样的近似。

    网格是静态的，按 ``id(env)`` 缓存一次；工件被随机化的只是位姿（root pose），
    局部几何不变，缓存不会过期。
    """
    key = id(env)
    cached = _OBJECT_LOCAL_BOUNDS_CACHE.get(key)
    if cached is not None:
        return cached

    import omni.usd
    from pxr import Usd, UsdGeom

    stage = omni.usd.get_context().get_stage()
    root_path = OBJECT_PRIM_PATTERN.format(env=0)
    root_prim = stage.GetPrimAtPath(root_path)
    if not root_prim.IsValid():
        raise RuntimeError(
            f"Object prim not found at {root_path!r} for occlusion bounds."
        )

    root_inv = UsdGeom.Xformable(root_prim).ComputeLocalToWorldTransform(
        Usd.TimeCode.Default()
    ).GetInverse()

    collected: list[torch.Tensor] = []
    for prim in Usd.PrimRange(root_prim):
        if not prim.IsA(UsdGeom.Mesh):
            continue
        mesh = UsdGeom.Mesh(prim)
        pts = mesh.GetPointsAttr().Get()
        if not pts:
            continue
        to_obj = UsdGeom.Xformable(prim).ComputeLocalToWorldTransform(
            Usd.TimeCode.Default()
        ) * root_inv
        collected.append(
            torch.tensor(
                [tuple(to_obj.Transform(p)) for p in pts],
                dtype=torch.float32,
                device=env.device,
            )
        )

    if not collected:
        raise RuntimeError(
            f"No UsdGeom.Mesh found under {root_path!r} for occlusion bounds."
        )

    verts = torch.cat(collected, dim=0)
    lo = verts.min(dim=0).values
    hi = verts.max(dim=0).values
    _OBJECT_LOCAL_BOUNDS_CACHE[key] = (lo, hi)
    return lo, hi


_OCCLUSION_STATE_BUF: dict = {}

# random_box 盒心采样用的表面点池大小。复用 _object_surface_points_local 的确定性
# FPS 缓存（cache key 含 m_obj，与阶段 1 的 M_OBJ 条目互不干扰），比 64 个点的
# 中心粒度更细。
_SURF_CENTER_POOL = 256


def randomize_occlusion_state(
    env: ManagerBasedRLEnv, env_ids: torch.Tensor, occlusion_mode: str
) -> None:
    """每次 reset 为随机遮挡（random_sphere/random_box）重采球心/盒心与盒朝向。

    采样结果写进 ``_OCCLUSION_STATE_BUF[id(env)]``，观测函数据此构造遮挡谓词。

    random_sphere：球心在工件局部包围盒 ``[lo, hi]`` 内均匀采样（原语义）。
    random_box：盒心从**面向相机的那侧表面点**里均匀抽一个 —— 保证盒心一定落在
    工件上、阴影必在相机可见面（不会再采到包围盒空角/背面而"全绿"）；朝向走完整
    SO(3)（``random_orientation``），作为各向异性椭球度量（轴长=工件局部包围盒
    span）的主轴方向，**与工件轴解耦**。

    相机侧筛选在工件局部系做：把相机位置转到工件局部系，取"以工件中心为原点、
    指向相机"的半球表面点。本事件注册在所有 reset 事件之后（set_occlusion_stage2
    追加），因此这里读到的 obj.data.root_pos_w 已是本 episode 随机化后的位姿
    （write_root_state_to_sim 会立即同步内部缓冲）。camera.data.pos_w 可能滞后
    一 episode 的相机抖动（randomize_camera_pose 直写 USD、不更新缓冲），但抖动
    仅 ±5cm/±6°，对半球筛选无影响。

    随机性走全局 RNG（multinomial / random_orientation），由 ``cfg.seed`` 播种，
    与工件位姿 DR 同源，给定 seed 可复现。只重采真正 reset 的环境，其余保留上一
    episode 的状态。
    """
    num_envs = env.num_envs
    buf = _OCCLUSION_STATE_BUF.setdefault(
        id(env),
        {
            "center": torch.zeros(num_envs, 3, device=env.device),
            "quat": torch.zeros(num_envs, 4, device=env.device),
        },
    )
    n = env_ids.numel()
    if n == 0:
        return

    if occlusion_mode == "random_box":
        # 盒心从面向相机的表面点抽：保证一定落在工件上、且在相机看得见的那面
        surf = _object_surface_points_local(env, m_obj=_SURF_CENTER_POOL)  # (P, 3) 工件局部系
        lo, hi = _object_local_bounds(env)
        obj_center = (lo + hi) * 0.5                                      # (3,)
        obj = env.scene["object"]
        cam = env.scene["camera_fixed"]
        cam_pos_obj = quat_apply_inverse(
            obj.data.root_quat_w[env_ids],
            cam.data.pos_w[env_ids] - obj.data.root_pos_w[env_ids],
        )                                                                 # (n, 3)
        dir_obj = cam_pos_obj - obj_center.unsqueeze(0)                   # (n, 3) 指向相机的方向
        facing = torch.einsum("sp,np->ns", surf - obj_center.unsqueeze(0), dir_obj) > 0.0
        weights = facing.to(torch.float32)                                # (n, P) 面向点权重 1，其余 0
        none_facing = weights.sum(dim=1) <= 0.0                           # 兜底：半球为空时退回全表面
        weights = torch.where(none_facing.unsqueeze(1), torch.ones_like(weights), weights)
        idx = torch.multinomial(weights, 1).squeeze(1)                    # (n,) 面向点里均匀抽一个
        buf["center"][env_ids] = surf[idx]
    else:  # random_sphere：包围盒均匀采样（原语义不变）
        lo, hi = _object_local_bounds(env)
        buf["center"][env_ids] = lo + torch.rand((n, 3), device=env.device) * (hi - lo)
    # 朝向只有 random_box 会读（球旋转对称）；采样不动，保持 RNG 流结构简单
    buf["quat"][env_ids] = random_orientation(n, env.device)


_EMPTY_MASK_SNAPSHOT_DIR = "/root/autodl-tmp/empty_mask_debug"


def _write_gray_png(path: str, arr) -> None:
    """8-bit 灰度 PNG（纯 stdlib，无 PIL/cv2 依赖）。"""
    import struct
    import zlib

    h, w = arr.shape
    raw = b"".join(b"\x00" + arr[i].tobytes() for i in range(h))

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 0, 0, 0, 0)  # 8-bit 灰度
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_chunk(b"IHDR", ihdr))
        f.write(_chunk(b"IDAT", zlib.compress(raw, 6)))
        f.write(_chunk(b"IEND", b""))


def _write_rgb_png(path: str, arr) -> None:
    """8-bit RGB PNG（纯 stdlib，无 PIL/cv2 依赖）。"""
    import struct
    import zlib

    h, w, _ = arr.shape
    raw = b"".join(b"\x00" + arr[i].tobytes() for i in range(h))

    def _chunk(tag: bytes, data: bytes) -> bytes:
        return (
            struct.pack(">I", len(data))
            + tag
            + data
            + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)
        )

    ihdr = struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)  # 8-bit RGB
    with open(path, "wb") as f:
        f.write(b"\x89PNG\r\n\x1a\n")
        f.write(_chunk(b"IHDR", ihdr))
        f.write(_chunk(b"IDAT", zlib.compress(raw, 6)))
        f.write(_chunk(b"IEND", b""))


def _dump_empty_mask_snapshot(
    env, camera, obj, mask, depth, seg, lut, empty, counts, strikes
) -> None:
    """空掩码 env 的相机通道 + 工件位姿落盘，肉眼确认掩码为何变空。

    只在 strikes 触顶（即将抛错）时调用一次。纯诊断、best-effort：任何失败都被
    上层 try/except 吞掉，不影响原本的 RuntimeError。
    """
    import os
    import time

    import numpy as np

    out_dir = os.path.join(
        _EMPTY_MASK_SNAPSHOT_DIR, time.strftime("%Y-%m-%d_%H-%M-%S")
    )
    os.makedirs(out_dir, exist_ok=True)

    empty_idx = empty.nonzero(as_tuple=False).flatten().cpu().tolist()
    for e in empty_idx[:4]:  # 512 env 可能同时多个空，只存前 4 个
        d = depth[e].detach().cpu().float().numpy()
        m = mask[e].detach().cpu().bool().numpy()
        s = seg[e].detach().cpu().numpy()
        obj_pos = obj.data.root_pos_w[e].detach().cpu().float().numpy()
        obj_ids = (lut == (e + 1)).nonzero(as_tuple=False).flatten().cpu().tolist()

        d_valid = np.isfinite(d) & (d > 0.0)
        if d_valid.any():
            lo = float(d[d_valid].min())
            hi = float(d[d_valid].max())
            d_gray = np.where(
                d_valid, (d - lo) / max(hi - lo, 1e-6) * 255.0, 0.0
            ).astype(np.uint8)
        else:
            d_gray = np.zeros(d.shape, dtype=np.uint8)
        m_gray = m.astype(np.uint8) * 255

        # 上色：工件 instance=绿，其它非零 instance（臂/夹爪/桌面）=红，背景=黑。
        seg_rgb = np.zeros((s.shape[0], s.shape[1], 3), dtype=np.uint8)
        if obj_ids:
            obj_px = s == obj_ids[0]
            other_px = (s != 0) & (~obj_px)
            seg_rgb[..., 1] = np.where(obj_px, np.uint8(255), np.uint8(0))
            seg_rgb[..., 0] = np.where(other_px, np.uint8(255), np.uint8(0))
        else:
            seg_rgb[..., 0] = np.where(s != 0, np.uint8(255), np.uint8(0))

        tag = f"env{e:04d}"
        _write_gray_png(os.path.join(out_dir, f"{tag}_depth.png"), d_gray)
        _write_gray_png(os.path.join(out_dir, f"{tag}_mask.png"), m_gray)
        _write_rgb_png(os.path.join(out_dir, f"{tag}_seg.png"), seg_rgb)

        rgb = camera.data.output.get("rgb")
        if rgb is not None:
            r = rgb[e].detach().cpu().float().numpy()
            if r.ndim == 3 and r.shape[-1] >= 3:
                r = r[..., :3]
                if float(r.max()) <= 1.0:
                    r = (r * 255.0).astype(np.uint8)
                else:
                    r = r.astype(np.uint8)
                _write_rgb_png(os.path.join(out_dir, f"{tag}_rgb.png"), r)

        np.savez(
            os.path.join(out_dir, f"{tag}.npz"),
            depth=d,
            mask=m,
            seg=s,
            obj_root_pos_w=obj_pos,
            obj_instance_ids=np.asarray(obj_ids, dtype=np.int64),
            visible_count=int(counts[e].item()),
        )
        print(
            f"[empty-mask] snapshot env={e} mask_px={int(m.sum())} "
            f"depth_valid_px={int(d_valid.sum())} visible_count={int(counts[e].item())} "
            f"obj_ids={obj_ids} obj_pos={obj_pos.tolist()} -> {out_dir}",
            flush=True,
        )


def point_bridge_point_cloud_occluded(
    env: ManagerBasedRLEnv,
    camera_cfg: SceneEntityCfg = SceneEntityCfg("camera_fixed"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    m_obj: int = M_OBJ,
    noise_std: float = NOISE_STD_M,
    occlusion_mode: str = "none",
    occlusion_severity: float = 0.0,
    occlusion_axis: str = "x",
    occlusion_center: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> torch.Tensor:
    """stage2 完整相机管线 + 坐标级工件遮挡（仿真实验专用）。

    与 :func:`point_bridge_point_cloud`（stage 2）**逐字同构**，唯一差别是反投影
    之后把一部分工件点按几何谓词挖掉：掩码、深度、反投影、workspace 裁剪、FPS、
    噪声、零阶保持全部原样复用 :func:`mask_depth_to_pointcloud`，遮挡通过它的
    ``occlusion_keep_fn`` 可选钩子注入 —— 因此 ``occlusion_mode='none'`` 时本函数
    与 stage 2 逐位一致，可作对照。

    遮挡只作用在**物体点**上；夹爪 6 关键点、关节角、critic 特权状态都不受影响。
    观测维度仍是 217，网络结构不变，因此可从 stage2 baseline checkpoint 热启动。
    """
    camera = env.scene[camera_cfg.name]
    robot = env.scene[robot_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    obj = env.scene[object_cfg.name]

    device = env.device
    num_envs = env.num_envs
    key = id(env)

    base_pos_w = robot.data.root_pos_w
    base_quat_w = robot.data.root_quat_w

    # ── 夹爪关键点（与 stage 2 逐字一致）──────────────────────────────────────
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]
    ee_pos_b, ee_quat_b = pose_in_frame(ee_pos_w, ee_quat_w, base_pos_w, base_quat_w)
    robot_pts = gripper_keypoints(ee_pos_b, ee_quat_b, noise_std=0.0)

    # ── 遮挡谓词（工件局部系）。mode != 'none' 且 severity>0 时才构造 ──────────
    occlusion_keep_fn = None
    if occlusion_mode != "none" and occlusion_severity > 0.0:
        obj_pos_b, obj_quat_b = pose_in_frame(
            obj.data.root_pos_w, obj.data.root_quat_w, base_pos_w, base_quat_w
        )
        obj_lo, obj_hi = _object_local_bounds(env)

        # 随机遮挡：从 reset 事件采好的状态里读球心/盒心与盒朝向。球旋转对称，
        # 朝向只有 random_box 会读；但两者中心都必须来自这里，否则遮挡区位置不随机。
        center_local = None
        shape_quat_local = None
        if occlusion_mode in ("random_sphere", "random_box"):
            state = _OCCLUSION_STATE_BUF.get(id(env))
            if state is None:
                # 构造期（ObservationManager 推断观测维度）会先于首次 reset 调用本函数，
                # 此时 reset 事件还没采样。给一个确定性占位（中心=包围盒中心、单位朝向），
                # 观测维度与值无关；首次 reset 后 randomize_occlusion_state 写入真实采样。
                center_local = ((obj_lo + obj_hi) * 0.5).unsqueeze(0).expand(num_envs, 3)
                shape_quat_local = torch.tensor(
                    [1.0, 0.0, 0.0, 0.0], device=device, dtype=obj_lo.dtype
                ).unsqueeze(0).expand(num_envs, 4)
            else:
                center_local = state["center"]
                shape_quat_local = state["quat"]

        def occlusion_keep_fn(pts_base: torch.Tensor) -> torch.Tensor:
            return occlusion_keep(
                pts_base,
                obj_pos_b,
                obj_quat_b,
                obj_lo,
                obj_hi,
                mode=occlusion_mode,
                severity=occlusion_severity,
                axis=occlusion_axis,
                center=occlusion_center,
                center_local=center_local,
                shape_quat_local=shape_quat_local,
            )

    # ── 物体点 ──────────────────────────────────────────────────────────────
    depth_raw = camera.data.output.get("distance_to_image_plane")
    seg_raw = camera.data.output.get("instance_id_segmentation_fast")
    lut = _get_instance_lut(env, camera) if seg_raw is not None else None

    obj_pts = None
    if depth_raw is not None and seg_raw is not None and lut is not None:
        depth = depth_raw[..., 0] if depth_raw.dim() == 4 else depth_raw
        seg = seg_raw[..., 0] if seg_raw.dim() == 4 else seg_raw

        seg_idx = seg.long().clamp_(min=0, max=lut.numel() - 1)
        expected = torch.arange(num_envs, device=device).add_(1).view(-1, 1, 1)
        mask = lut[seg_idx] == expected

        cam_pos_b, cam_quat_b = pose_in_frame(
            camera.data.pos_w, camera.data.quat_w_ros, base_pos_w, base_quat_w
        )

        occ_debug = _OCCLUSION_VIS.setdefault(key, {}) if key in _OCCLUSION_VIS_ENABLED else None
        candidate_pts, counts = mask_depth_to_pointcloud(
            mask=mask,
            depth=depth,
            K=camera.data.intrinsic_matrices,
            cam_pos=cam_pos_b,
            cam_quat=cam_quat_b,
            m_obj=m_obj,
            noise_std=noise_std,
            occlusion_keep_fn=occlusion_keep_fn,
            occlusion_debug=occ_debug,
        )
        # 可视化旁路：另算一张稠密遮挡图（逐像素），供录像画"遮挡整体形状"。
        if occ_debug is not None and occlusion_keep_fn is not None:
            occ_debug["occluded_map"] = dense_occlusion_pixel_map(
                mask=mask,
                depth=depth,
                K=camera.data.intrinsic_matrices,
                cam_pos=cam_pos_b,
                cam_quat=cam_quat_b,
                occlusion_keep_fn=occlusion_keep_fn,
            ).detach()

        env.extras["visible_ratio"] = (
            counts.float() / float(m_obj)
        ).clamp_(max=1.0).mean()

        if key in _DEBUG_ENABLED:
            _DEBUG_CAPTURE[key] = {
                "mask_pixels": mask.flatten(1).sum(dim=1).detach(),
                "visible_counts": counts.detach(),
                "depth": depth.detach(),
                "cam_pos_b": cam_pos_b.detach(),
                "cam_quat_b": cam_quat_b.detach(),
                "K": camera.data.intrinsic_matrices.detach(),
            }

        previous = _LAST_POINT_CLOUD.get(key)
        empty = counts == 0
        if bool(empty.any()):
            if previous is None:
                raise RuntimeError(
                    "The object mask is empty on the very first observation, so there is no "
                    "previous point cloud to hold. Check the camera extrinsics, the object "
                    "prim pattern, and that --enable_cameras is set."
                )
            candidate_pts = torch.where(
                empty.view(-1, 1, 1), previous[:, :m_obj, :], candidate_pts
            )
            strikes = _EMPTY_MASK_STRIKES.get(key, 0) + 1
            _EMPTY_MASK_STRIKES[key] = strikes
            if strikes > _MAX_EMPTY_STRIKES:
                # 区分"臂物理遮挡工件"与"工件失位 / 相机损坏"两种空 mask。
                # 工件是 kinematic（永不动）：若它仍停在 default_root_state 附近且该环境
                # 深度大体有效，空 mask 只能是被机械臂挡住 —— 继续 ZOH 即可（工件没动，
                # 上一帧点云仍是正确的），而不是当成"相机/分割坏了"直接抛错。
                obj_local = obj.data.root_pos_w - env.scene.env_origins
                at_spawn = (
                    (obj_local - obj.data.default_root_state[:, :3])
                    .abs()
                    .max(dim=1)
                    .values
                    <= 0.2
                )
                depth_ok = (torch.isfinite(depth) & (depth > 0.0)).flatten(1).any(dim=1)
                genuine = empty & (~at_spawn | ~depth_ok)
                if bool(genuine.any()):
                    # 真正的故障：工件失位或该环境深度全无效。照旧落盘快照再抛错。
                    try:
                        _dump_empty_mask_snapshot(
                            env=env,
                            camera=camera,
                            obj=obj,
                            mask=mask,
                            depth=depth,
                            seg=seg,
                            lut=lut,
                            empty=genuine,
                            counts=counts,
                            strikes=strikes,
                        )
                    except Exception as _exc:  # noqa: BLE001 诊断失败不能吞掉真正的报错
                        print(f"[empty-mask] snapshot failed (ignored): {_exc}", flush=True)
                    raise RuntimeError(
                        f"The object mask has been empty for {strikes} consecutive steps in at "
                        "least one environment while the object is not at its spawn pose or the "
                        "depth is entirely invalid. This indicates a broken camera or "
                        "segmentation setup (physical occlusion by the arm is excluded)."
                    )
                # 全是物理遮挡（工件仍在 spawn 且深度有效）：不抛错，重置计数，继续 ZOH。
                _EMPTY_MASK_STRIKES[key] = 0
        else:
            _EMPTY_MASK_STRIKES[key] = 0

        obj_pts = candidate_pts

    if obj_pts is None:
        previous = _LAST_POINT_CLOUD.get(key)
        if previous is not None:
            obj_pts = previous[:, :m_obj, :]
        else:
            obj_pts = torch.zeros(num_envs, m_obj, POINT_DIM, device=device)

    point_cloud = assemble_point_cloud(obj_pts, robot_pts)
    _LAST_POINT_CLOUD[key] = point_cloud.detach()

    if key in _DEBUG_ENABLED:
        _DEBUG_CAPTURE.setdefault(key, {})["points"] = point_cloud.detach()

    return point_cloud.reshape(num_envs, -1)


def set_occlusion_stage2(
    env_cfg,
    mode: str = "none",
    severity: float = 0.0,
    axis: str = "x",
    center: tuple[float, float, float] = (0.5, 0.5, 0.5),
) -> None:
    """把 stage2 的物体点观测换成带坐标遮挡的版本（仿真实验专用）。

    **必须在 :func:`set_observation_stage`(cfg, 2) 之后调用**：它只替换观测函数，
    不重设相机 / 掩码 / 噪声等 stage 2 已有的设置，也不碰任何其他 stage 的分支。

    ``mode='none'`` 或 ``severity=0`` 时仍走带遮挡的观测函数，但内部不注入任何
    过滤，行为与 stage 2 逐位一致 —— 用作"遮挡管线本身无副作用"的对照。

    ``mode='random_sphere'/'random_box'`` 时额外注册一个 ``mode="reset"`` 事件
    （:func:`randomize_occlusion_state`），每个 episode 重采球心/盒心与盒朝向：
    random_sphere 球心在包围盒内随机；random_box 盒心从面向相机的表面点随机、
    朝向随机 SO(3)（各向异性椭球度量主轴），不与工件轴对齐。
    """
    term = env_cfg.observations.policy.point_cloud
    term.func = point_bridge_point_cloud_occluded
    term.params = {
        "camera_cfg": SceneEntityCfg("camera_fixed"),
        "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        "robot_cfg": SceneEntityCfg("robot"),
        "object_cfg": SceneEntityCfg("object"),
        "m_obj": M_OBJ,
        "noise_std": NOISE_STD_M,
        "occlusion_mode": mode,
        "occlusion_severity": severity,
        "occlusion_axis": axis,
        "occlusion_center": center,
    }
    # 随机遮挡需要在每个 episode 开始时重采遮挡区的位置/朝向；只在随机模式下注册，
    # 避免给 halfspace/sphere/none（以及其它 stage）白加 reset 开销。
    if mode in ("random_sphere", "random_box"):
        env_cfg.events.randomize_occlusion_state = EventTerm(
            func=randomize_occlusion_state,
            mode="reset",
            params={"occlusion_mode": mode},
        )
