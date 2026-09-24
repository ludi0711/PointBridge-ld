#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Manual eval for xArm7 CRT-gripper pick-pose policy with baseline arm write control."""

import argparse
import os
import sys
from pathlib import Path

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from tools.logs.pick_pose_csv import PickPoseCsvRow, open_pick_pose_csv

# AppLauncher 之前不要 import isaaclab.sim / isaaclab.envs / pxr / omni。
from isaaclab.app import AppLauncher


VERSION_TAG = "eval_pick_pose_crt_write"
LOG_NAME = "xarm7_pick_pose_crt_write"
ENV_CFG_FILE = os.path.join(_PROJECT_DIR, "configs", "xarm7_pick_pose_crt_write_env_cfg.py")
TEST_LOG_DIR = os.path.join(_PROJECT_DIR, "test_log")
PLAY_MDP_OBS_DIM = 21
PLAY_ACTION_DIM = 7

# ── 回放参数 ───────────────────────────────────────────────────────────────────
# 纯策略回放建议保持 1.0，即完全不平滑，真实看训练出来的 policy 行为。
# 如果想看部署平滑效果，可以改成 0.25~0.5。
PLAY_ACTION_SMOOTH_ALPHA = 1.0

# 默认关闭 stop-on-best，避免刚开始距离还很远时被误判为 best 并锁死。
PLAY_ENABLE_STOP_ON_BEST = False

# 如果手动启用 stop-on-best，只有进入这个邻域后才允许停。
PLAY_STOP_ENABLE_DIST_CM = 3.0
PLAY_STOP_ENABLE_ORI_DEG = 15.0
PLAY_STOP_DIST_WEIGHT = 0.9
PLAY_STOP_ORI_WEIGHT = 0.1
PLAY_STOP_DIST_REF_CM = 1.0
PLAY_STOP_ORI_REF_DEG = 6.0
PLAY_MIN_SCORE_PATIENCE = 20
PLAY_MIN_SCORE_EPS = 0.001

# 与环境 reward / success 的 q_offset 保持一致，格式为 w,x,y,z。
PLAY_ORI_Q_OFFSET_WXYZ = "0.0,0.0,0.0,1.0"
# 平行夹爪绕局部 Z 轴 180° 后视为同一个夹持姿态。
PLAY_GRASP_SYMMETRY_QUATS_WXYZ = (
    "1.0,0.0,0.0,0.0",
    "0.0,0.0,0.0,1.0",
)


DEFAULT_FIXED_SCENE_USD = (
    Path(_PROJECT_DIR)
    / "assets"
    / "crt_ctm2f110_gripper_visualization"
    / "scene_config_files"
    / "visualized_lift_v20_scene_articulation_fixed.usd"
)
DEFAULT_RAW_SCENE_USD = (
    Path(_PROJECT_DIR)
    / "assets"
    / "crt_ctm2f110_gripper_visualization"
    / "scene_config_files"
    / "visualized_lift_v20_scene.usd"
)
DEFAULT_TRAINING_ROBOT_USD = Path(_PROJECT_DIR) / "assets" / "xarm7" / "gx_va_crt_training_robot_write_train.usd"
DEFAULT_WRAPPER_USD = Path("/tmp/gx_va_crt_training_robot_write_train.usd")
DEFAULT_OBJECT_WRAPPER_USD = Path("/tmp/gx_va_gongjian_dynamic_safe_crt_write.usd")
DEFAULT_GRASPABLE_OBJECT_USD = Path(_PROJECT_DIR) / "assets" / "xarm7" / "gongjian_graspable.usd"
DEFAULT_XARM7_PINOCCHIO_URDF = (
    Path(_PROJECT_DIR)
    / "assets"
    / "crt_ctm2f110_gripper_visualization"
    / "external"
    / "xarm_description"
    / "meshes"
    / "xarm7.urdf"
)
OBJECT_COLLISION_APPROXIMATION = "convexHull"
WORKPIECE_GRIP_SIDE_LENGTH = 0.18
WORKPIECE_GRIP_SIDE_THICKNESS = 0.006
WORKPIECE_GRIP_SIDE_HEIGHT = 0.030
WORKPIECE_GRIP_SIDE_Z_OFFSET = 0.004
WORKPIECE_CONTACT_OFFSET = 0.004
GRIPPER_CONTACT_OFFSET = 0.003
GRIPPER_PAD_CONTACT_SIZE = (0.0080, 0.0500, 0.0320)
GRIPPER_PAD_CONTACT_PATHS = (
    "/Robot/link7/tool/assembly/links/left_pad",
    "/Robot/link7/tool/assembly/links/right_pad",
)
REST_OFFSET = 0.0
GRIPPER_STATIC_FRICTION = 2.5
GRIPPER_DYNAMIC_FRICTION = 2.0
WORKPIECE_MAIN_STATIC_FRICTION = 0.6
WORKPIECE_MAIN_DYNAMIC_FRICTION = 0.45
WORKPIECE_GRIP_STATIC_FRICTION = 3.0
WORKPIECE_GRIP_DYNAMIC_FRICTION = 2.5
CONTACT_TORSIONAL_PATCH_RADIUS = 0.025
CONTACT_MIN_TORSIONAL_PATCH_RADIUS = 0.010
CONTACT_REPORT_BODY_PATHS = (
    "/Robot/link7",
    "/Robot/link7/tool/assembly/links/left_pad",
    "/Robot/link7/tool/assembly/links/right_pad",
    "/Robot/link7/tool/assembly/links/leftout_Link",
    "/Robot/link7/tool/assembly/links/rightout_Link",
    "/Robot/link7/tool/assembly/links/leftinn_Link",
    "/Robot/link7/tool/assembly/links/rightinn_Link",
    "/Robot/link7/tool/assembly/links/left_kckle",
    "/Robot/link7/tool/assembly/links/right_kckle",
)
OBJECT_CONTACT_FILTER_PATHS = ["{ENV_REGEX_NS}/Object"]
PAD_PROXY_SIZE = GRIPPER_PAD_CONTACT_SIZE
PAD_PROXY_LOCAL_OFFSETS = {
    "left": (-0.0009999997, 0.0004093610, 0.0000033076),
    # The right/magenta side used to contact too late and wedge the object
    # downward. Shift it toward the pad contact face so collision starts before
    # it has already penetrated the workpiece side band.
    "right": (0.0030000000, 0.0004101116, 0.0000033076),
}
USE_EXTERNAL_PAD_PROXIES = False
PAD_PROXY_ASSETS = {
    "left": "left_pad_proxy",
    "right": "right_pad_proxy",
}
PAD_CONTACT_SENSOR_BODIES = {
    "left_pad_contact": "/Robot/link7/tool/assembly/links/left_pad",
    "right_pad_contact": "/Robot/link7/tool/assembly/links/right_pad",
}
PAD_PROXY_PRIMS = {
    "left": "LeftPadProxy",
    "right": "RightPadProxy",
}
PAD_PROXY_MASS = 5.0
PAD_PROXY_ATTRACTOR_STIFFNESS = 6000.0
PAD_PROXY_ATTRACTOR_DAMPING = 650.0
PAD_PROXY_ATTRACTOR_FORCE_LIMIT = 600.0
GRIPPER_CHAIN_BODY_PATTERNS = (
    "left_pad",
    "right_pad",
    "leftout_Link",
    "rightout_Link",
    "leftinn_Link",
    "rightinn_Link",
    "left_kckle",
    "right_kckle",
)
GRIPPER_CHAIN_MAX_SPAN_M = 0.35
GRIPPER_PAD_DISTANCE_MAX_M = 0.20


def default_source_usd() -> Path:
    return DEFAULT_FIXED_SCENE_USD if DEFAULT_FIXED_SCENE_USD.exists() else DEFAULT_RAW_SCENE_USD


parser = argparse.ArgumentParser(
    description="xArm7 CRT Pick 位姿策略 eval，arm=baseline write，gripper=set，手动 Start Pick"
)
parser.add_argument("--num_envs", type=int, default=None, help="训练环境数量；play 默认 1，train 默认 64")
parser.add_argument("--play", action="store_true", help="兼容参数；eval 脚本会强制启用 play 模式")
parser.add_argument("--auto_start_policy", action="store_true", help="进入 eval 后立即开始策略输出；默认等控制面板 Start Pick")
parser.add_argument("--checkpoint", type=str, default=None, help="checkpoint 路径")
parser.add_argument("--resume", action="store_true", help="训练时从 checkpoint 或 load_run 恢复")
parser.add_argument("--load_run", type=str, default=None, help="rsl_rl runner.load 的路径")
parser.add_argument("--max_iterations", type=int, default=None, help="最大训练迭代次数")
parser.add_argument("--seed", type=int, default=None, help="随机种子")
parser.add_argument(
    "--enable_success_termination",
    action="store_true",
    help="启用 reach_success 提前 done；默认关闭，使用 no-success horizon 训练。",
)
parser.add_argument(
    "--use_policy_gripper",
    action="store_true",
    help="eval 22 obs / 8 actions 的策略夹爪版本；默认保持旧策略 21 obs / 7 actions，夹爪由脚本/auto-close 控制。",
)
parser.add_argument("--play_max_steps", type=int, default=-1, help="eval 单次 episode 最大步数；到达后 reset env，不退出；-1 表示只按环境 done/reset")
parser.add_argument("--eval_episode_length_s", type=float, default=3600.0, help="eval 环境 horizon 秒数；默认拉长，避免未手动 Start Pick 就 8s/400step timeout")
parser.add_argument("--print_every", type=int, default=10, help="回放每隔多少步打印一次")
parser.add_argument("--test_log_dir", type=str, default=TEST_LOG_DIR, help="回放 CSV 保存目录")
parser.add_argument("--robot_usd", type=Path, default=None, help="直接使用这个 CRT robot USD，跳过 wrapper 生成；默认使用 assets/xarm7/gx_va_crt_training_robot_write_train.usd")
parser.add_argument("--source_usd", type=Path, default=None, help="包含 CRT robot prim 的源场景 USD")
parser.add_argument("--source_robot_prim", default="/World/XArm7WithCRTGripper", help="源 USD 里的 robot prim")
parser.add_argument("--wrapper_usd", type=Path, default=None, help="生成的训练 robot wrapper USD")
parser.add_argument("--dynamic_object_eval", action="store_true", help="把工件切成 dynamic，用于检查夹爪接触/抓取；默认保持 baseline kinematic 工件")
parser.add_argument("--arm_set_mode", action="store_true", help="机械臂也改用 set_joint_position_target，不直接 write 状态；用于和 baseline write 对比")
parser.add_argument("--object_usd", type=Path, default=None, help="dynamic eval 的工件 USD；默认用 assets/xarm7/gongjian_graspable.usd")
parser.add_argument("--object_wrapper_usd", type=Path, default=DEFAULT_OBJECT_WRAPPER_USD, help="dynamic-safe 工件 wrapper USD")
parser.add_argument("--object_mass", type=float, default=0.05, help="dynamic eval 工件质量 kg；默认先降到 50g 方便验证夹取")
parser.add_argument("--object_friction", type=float, default=3.0, help="dynamic eval 工件摩擦系数")
parser.add_argument("--hold_after_auto_close", action="store_true", help="进入自动闭合窗口后保持机械臂当前关节角；dynamic_object_eval 会默认启用")
parser.add_argument("--lift_after_auto_close", action="store_true", help="自动闭合后抬工件；默认用 Pinocchio IK 沿 TCP/base z 抬高指定高度")
parser.add_argument("--policy_lift_after_auto_close", action="store_true", help="动态 eval 时自动闭合后继续由策略控制机械臂 lift；禁用 eval 的 hold/脚本 lift")
parser.add_argument("--lift_after_auto_close_delay", type=float, default=0.5, help="左右 pad 接触力都达到阈值后再稳定等待多久开始抬，单位秒")
parser.add_argument("--lift_contact_force_threshold", type=float, default=12.0, help="lift 触发前左右 pad 法向力都必须达到的阈值，单位 N")
parser.add_argument("--lift_mode", choices=("cartesian_z", "joint2_delta"), default="cartesian_z", help="自动夹取后的抬升方式；cartesian_z 与真机 deploy 的 gripper_lift_mm 对齐")
parser.add_argument("--lift_height_mm", "--gripper_lift_mm", dest="lift_height_mm", type=float, default=200.0, help="cartesian_z 模式下 TCP 沿 base z 抬高的距离，单位 mm；与 deploy --gripper_lift_mm 同义")
parser.add_argument("--lift_cartesian_step_mm", type=float, default=5.0, help="cartesian_z 模式每个控制步的 TCP z 小步长，单位 mm；越小越能保持 lift 过程姿态")
parser.add_argument("--lift_joint2_delta_deg", type=float, default=-8.0, help="joint2_delta fallback 模式下 joint2 的总增量；方向反了可设为正数")
parser.add_argument("--lift_ik_urdf", type=Path, default=DEFAULT_XARM7_PINOCCHIO_URDF, help="Pinocchio IK 使用的 xArm7 URDF")
parser.add_argument("--lift_ik_tcp_z_offset_m", type=float, default=0.177, help="Pinocchio IK 的 TCP 相对 link7 的本地 z 偏移，单位 m；需与 ee_frame offset 保持一致")
parser.add_argument("--lift_ik_max_iters", type=int, default=120, help="Pinocchio IK 每次求解最大迭代次数")
parser.add_argument("--lift_ik_damping", type=float, default=1.0e-4, help="Pinocchio IK 阻尼最小二乘 damping")
parser.add_argument("--lift_ik_tol_pos_mm", type=float, default=1.0, help="Pinocchio IK 位置收敛阈值，单位 mm")
parser.add_argument("--lift_ik_tol_ori_deg", type=float, default=2.0, help="Pinocchio IK 姿态保持阈值，单位 deg")
parser.add_argument("--object_frame_axis_length", type=float, default=0.08, help="play 推理时绘制工件坐标轴长度，单位 m；设 <=0 关闭")
parser.add_argument("--hide_contact_boxes", action="store_true", help="隐藏新增的 gripper pad / workpiece side collision debug boxes")
parser.add_argument("--gripper_force_target", type=float, default=None, help="覆盖夹爪力反馈目标，单位 N")
parser.add_argument("--gripper_force_push_effort", type=float, default=None, help="覆盖 force servo 中未达力侧的 effort limit，单位 N*m")
parser.add_argument("--gripper_force_kp", type=float, default=None, help="覆盖夹爪力反馈 Kp，单位 deg/N")
parser.add_argument("--gripper_force_ki", type=float, default=None, help="覆盖夹爪力反馈 Ki，单位 deg/(N*s)")
parser.add_argument("--gripper_force_kd", type=float, default=None, help="覆盖夹爪力反馈 Kd，单位 deg*s/N")
parser.add_argument("--gripper_force_deadband", type=float, default=None, help="覆盖夹爪力反馈死区，单位 N")
parser.add_argument("--gripper_force_max_step", type=float, default=None, help="覆盖夹爪力反馈每步最大目标变化，单位 deg")
parser.add_argument("--gripper_force_hold_error", type=float, default=None, help="覆盖过力侧保持的小预压角，单位 deg；越小接触力越低")
parser.add_argument("--gripper_force_release_step", type=float, default=None, help="覆盖过力侧 target 每步最大释放量，单位 deg；0 表示瞬时释放到预压目标")
parser.add_argument("--gripper_hold_force", type=float, default=None, help="覆盖夹爪进入 hold/限力所需的左右 pad 法向力阈值，单位 N")
parser.add_argument("--gripper_hard_warn_force", type=float, default=None, help="覆盖单侧大力报警阈值，单位 N")
parser.add_argument("--gripper_approach_effort", type=float, default=None, help="覆盖闭合接触前 effort limit，单位 N*m")
parser.add_argument("--gripper_hold_effort", type=float, default=None, help="覆盖 hold 后 effort limit，单位 N*m")
parser.add_argument("--gripper_hard_hold_effort", type=float, default=None, help="覆盖单侧超过 hard warn 后的 effort limit，单位 N*m")
parser.add_argument("--gripper_velocity_limit", type=float, default=None, help="覆盖夹爪关节速度限，单位 rad/s")
parser.add_argument("--gripper_sync_max_lead", type=float, default=None, help="覆盖夹爪同步允许的最大实际开合领先量，单位 deg")
parser.add_argument("--gripper_free_close_step", type=float, default=None, help="覆盖无/单侧接触时夹爪闭合目标步长，单位 deg/step")
parser.add_argument("--gripper_contact_close_step", type=float, default=None, help="覆盖双侧接触且 force servo 关闭时夹爪闭合目标步长，单位 deg/step")
parser.add_argument("--gripper_auto_close_dist_cm", type=float, default=None, help="覆盖夹爪控制器内部自动闭合距离阈值，单位 cm")
parser.add_argument("--gripper_auto_close_ori_deg", type=float, default=None, help="覆盖夹爪控制器内部自动闭合姿态阈值，单位 deg")
parser.add_argument("--arm_hold_auto_close_dist_cm", type=float, default=None, help="覆盖 eval 中机械臂进入 hold 的距离阈值，单位 cm；默认跟随夹爪闭合距离")
parser.add_argument("--arm_hold_auto_close_ori_deg", type=float, default=None, help="覆盖 eval 中机械臂进入 hold 的姿态阈值，单位 deg；默认跟随夹爪闭合姿态")
parser.add_argument("--gripper_chain_max_span_m", type=float, default=GRIPPER_CHAIN_MAX_SPAN_M, help="断链检测：gripper body 到 link7 的最大允许距离，单位 m")
parser.add_argument("--gripper_pad_distance_max_m", type=float, default=GRIPPER_PAD_DISTANCE_MAX_M, help="断链检测：左右 pad 最大允许距离，单位 m")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
USE_POLICY_GRIPPER = bool(args_cli.use_policy_gripper)
if USE_POLICY_GRIPPER:
    PLAY_MDP_OBS_DIM = 22
    PLAY_ACTION_DIM = 8
args_cli.play = True
if args_cli.num_envs is None:
    args_cli.num_envs = 1

# 使用低维状态观测，不需要相机。
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# ---- AppLauncher 之后再 import Isaac Lab / RSL-RL / torch 等 ----
import math
import traceback
from datetime import datetime
import shutil

import numpy as np
import torch
from pxr import Gf, PhysxSchema, Sdf, Usd, UsdGeom, UsdPhysics, UsdShade
import omni.usd
try:
    from omni.isaac.dynamic_control import _dynamic_control
except Exception:
    _dynamic_control = None
try:
    import omni.ui as ui
except Exception:
    ui = None

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.sensors import ContactSensorCfg
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from configs.xarm7_pick_pose_env_cfg import set_success_termination_enabled
from configs.xarm7_pick_pose_crt_write_env_cfg import (
    AUTO_GRIPPER_CLOSE_DIST_M,
    AUTO_GRIPPER_CLOSE_ORI_DEG,
    AUTO_GRIPPER_CLOSE_TIME_S,
    GRIPPER_CLOSE_VELOCITY_LIMIT,
    GRIPPER_FORCE_CONTROL_DEADBAND_N,
    GRIPPER_FORCE_CONTROL_ENABLED,
    GRIPPER_FORCE_CONTROL_KP_DEG_PER_N,
    GRIPPER_FORCE_CONTROL_KI_DEG_PER_N_S,
    GRIPPER_FORCE_CONTROL_KD_DEG_S_PER_N,
    GRIPPER_FORCE_CONTROL_INTEGRAL_LIMIT_N_S,
    GRIPPER_FORCE_CONTROL_MAX_STEP_DEG,
    GRIPPER_FORCE_CONTROL_TARGET_N,
    GRIPPER_FREE_CLOSE_FORCE_N,
    GRIPPER_FREE_CLOSE_STEP_DEG,
    GRIPPER_CONTACT_CLOSE_STEP_DEG,
    GRIPPER_HARD_NORMAL_FORCE_N,
    GRIPPER_HOLD_EFFORT,
    GRIPPER_HOLD_NORMAL_FORCE_N,
    GRIPPER_MIMIC_EFFORT,
    GRIPPER_MIMIC_STIFFNESS,
    GRIPPER_PRIMARY_EFFORT,
    GRIPPER_PRIMARY_STIFFNESS,
    GRIPPER_SYNC_ENABLED,
    GRIPPER_SYNC_MAX_LEAD_DEG,
    OBJECT_POSE_Z_OFFSET_M,
    CRTSetActionsCfg,
    PickPoseCRTWriteEnvCfg as PickPoseEnvCfg,
    PickPoseCRTWriteEnvCfg_PLAY as PickPoseEnvCfg_PLAY,
    _ARM_ACTION_SCALE,
    _ARM_CLIP_RAD,
    _ARM_JOINT_LIMITS_HIGH,
    _ARM_JOINT_LIMITS_LOW,
    _compute_reach_error_metrics,
)
from configs.controller import (
    CRTGripperEvalGate,
    get_arm_action_term as _get_arm_action_term,
    set_gripper_auto_close_enabled as _set_gripper_auto_close_enabled,
)
from configs.agents.rsl_rl_ppo_cfg import PickPosePPORunnerCfg
from tools.utils.pick_pose_eval_utils import get_ee_target_pose_w, pose_in_robot_base


def _write_robot_wrapper_usd(source_usd: Path, source_prim: str, wrapper_usd: Path) -> Path:
    source_usd = source_usd.expanduser().resolve()
    wrapper_usd = wrapper_usd.expanduser().resolve()
    if not source_usd.exists():
        raise FileNotFoundError(source_usd)
    wrapper_usd.parent.mkdir(parents=True, exist_ok=True)

    source_stage = Usd.Stage.Open(str(source_usd))
    if source_stage is None:
        raise RuntimeError(f"Could not open source USD: {source_usd}")
    if not source_stage.GetPrimAtPath(source_prim).IsValid():
        raise RuntimeError(f"Robot prim not found in source USD: {source_prim}")

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, Sdf.Path("/Robot")).GetPrim()
    stage.SetDefaultPrim(root)
    if not root.GetReferences().AddReference(str(source_usd), Sdf.Path(source_prim)):
        raise RuntimeError(f"Failed to add robot reference: {source_usd}:{source_prim}")

    contact_count = 0
    for body_path in CONTACT_REPORT_BODY_PATHS:
        body_prim = stage.GetPrimAtPath(body_path)
        if not body_prim.IsValid():
            print(f"[WARN] Contact reporter body not found in wrapper: {body_path}", flush=True)
            continue
        if body_prim.HasAPI(PhysxSchema.PhysxContactReportAPI):
            contact_api = PhysxSchema.PhysxContactReportAPI.Get(stage, body_prim.GetPrimPath())
        else:
            contact_api = PhysxSchema.PhysxContactReportAPI.Apply(body_prim)
        contact_api.CreateThresholdAttr().Set(0.0)
        contact_count += 1

    gripper_material = _define_physics_material(
        stage,
        Sdf.Path("/Robot/PhysicsMaterials/gripper_high_friction"),
        GRIPPER_STATIC_FRICTION,
        GRIPPER_DYNAMIC_FRICTION,
    )
    gripper_pad_box_count = _add_gripper_pad_contact_boxes(stage, gripper_material)
    gripper_material_count = _bind_material_to_collision_subtree(
        stage,
        Sdf.Path("/Robot/link7/tool/assembly"),
        gripper_material,
        GRIPPER_CONTACT_OFFSET,
    )

    stage.GetRootLayer().Export(str(wrapper_usd))
    print(f"[OK] Training robot USD wrapper: {wrapper_usd}", flush=True)
    print(f"[OK] Contact reporter enabled on {contact_count} CRT bodies", flush=True)
    print(
        f"[OK] Gripper contact material: collisions={gripper_material_count}, "
        f"pad_boxes={gripper_pad_box_count}, "
        f"mu={GRIPPER_STATIC_FRICTION:.2f}/{GRIPPER_DYNAMIC_FRICTION:.2f}, "
        f"contactOffset={GRIPPER_CONTACT_OFFSET:.4f}",
        flush=True,
    )
    return wrapper_usd


def _resolve_robot_usd() -> Path:
    if args_cli.robot_usd is not None:
        robot_usd = args_cli.robot_usd.expanduser().resolve()
        if not robot_usd.exists():
            raise FileNotFoundError(robot_usd)
        return robot_usd

    if args_cli.source_usd is None and args_cli.wrapper_usd is None:
        robot_usd = DEFAULT_TRAINING_ROBOT_USD.expanduser().resolve()
        if not robot_usd.exists():
            raise FileNotFoundError(robot_usd)
        return robot_usd

    source_usd = args_cli.source_usd if args_cli.source_usd is not None else default_source_usd()
    wrapper_usd = args_cli.wrapper_usd if args_cli.wrapper_usd is not None else DEFAULT_WRAPPER_USD
    return _write_robot_wrapper_usd(source_usd, args_cli.source_robot_prim, wrapper_usd)



def _set_prim_attr(prim, name: str, type_name, value):
    attr = prim.GetAttribute(name)
    if not attr:
        attr = prim.CreateAttribute(name, type_name)
    attr.Set(value)


def _define_physics_material(
    stage,
    material_path: Sdf.Path,
    static_friction: float,
    dynamic_friction: float,
) -> UsdShade.Material:
    parent_path = material_path.GetParentPath()
    if not stage.GetPrimAtPath(parent_path).IsValid():
        UsdGeom.Scope.Define(stage, parent_path)
    material = UsdShade.Material.Define(stage, material_path)
    material_api = UsdPhysics.MaterialAPI.Apply(material.GetPrim())
    material_api.CreateStaticFrictionAttr().Set(float(static_friction))
    material_api.CreateDynamicFrictionAttr().Set(float(dynamic_friction))
    material_api.CreateRestitutionAttr().Set(0.0)
    PhysxSchema.PhysxMaterialAPI.Apply(material.GetPrim())
    _set_prim_attr(material.GetPrim(), "physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token, "max")
    _set_prim_attr(material.GetPrim(), "physxMaterial:restitutionCombineMode", Sdf.ValueTypeNames.Token, "min")
    _set_prim_attr(material.GetPrim(), "physxMaterial:improvePatchFriction", Sdf.ValueTypeNames.Bool, True)
    return material


def _set_contact_patch_attrs(prim, contact_offset: float):
    PhysxSchema.PhysxCollisionAPI.Apply(prim)
    _set_prim_attr(prim, "physxCollision:contactOffset", Sdf.ValueTypeNames.Float, float(contact_offset))
    _set_prim_attr(prim, "physxCollision:restOffset", Sdf.ValueTypeNames.Float, REST_OFFSET)
    _set_prim_attr(prim, "physxCollision:torsionalPatchRadius", Sdf.ValueTypeNames.Float, CONTACT_TORSIONAL_PATCH_RADIUS)
    _set_prim_attr(prim, "physxCollision:minTorsionalPatchRadius", Sdf.ValueTypeNames.Float, CONTACT_MIN_TORSIONAL_PATCH_RADIUS)


def _bind_material_to_collision_subtree(stage, root_path: Sdf.Path, material: UsdShade.Material, contact_offset: float) -> int:
    count = 0
    for prim in stage.Traverse():
        if not prim.GetPath().HasPrefix(root_path):
            continue
        if not prim.HasAPI(UsdPhysics.CollisionAPI):
            continue
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
        _set_contact_patch_attrs(prim, contact_offset)
        count += 1
    return count


def _add_gripper_pad_contact_boxes(stage, material: UsdShade.Material | None = None) -> int:
    """Enable collision on the actual pad visual meshes.

    Extra child cube colliders are fragile here because the CRT pad rigid bodies
    use reset-xform absolute transforms. PhysX debug can display those child
    cubes away from the rendered pad. The visual mesh is already under the pad
    rigid body, so using it as a convex hull keeps collision and rendering on
    the same path.
    """
    created = 0
    for pad_path_str in GRIPPER_PAD_CONTACT_PATHS:
        pad_path = Sdf.Path(pad_path_str)
        pad_prim = stage.GetPrimAtPath(pad_path)
        if not pad_prim.IsValid():
            print(f"[WARN] Gripper pad prim not found for collision: {pad_path}", flush=True)
            continue

        old_box_path = pad_path.AppendChild(f"{pad_path.name}_isaac_box_collision")
        if stage.GetPrimAtPath(old_box_path).IsValid():
            stage.RemovePrim(old_box_path)

        collision_path = pad_path.AppendChild("visual")
        collision_prim = stage.GetPrimAtPath(collision_path)
        if not collision_prim.IsValid():
            print(f"[WARN] Gripper pad visual mesh not found for collision: {collision_path}", flush=True)
            continue

        collision_api = UsdPhysics.CollisionAPI.Apply(collision_prim)
        collision_api.CreateCollisionEnabledAttr().Set(True)
        mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(collision_prim)
        mesh_collision_api.CreateApproximationAttr().Set(UsdPhysics.Tokens.convexHull)
        if material is not None:
            UsdShade.MaterialBindingAPI.Apply(collision_prim).Bind(material)
        _set_contact_patch_attrs(collision_prim, GRIPPER_CONTACT_OFFSET)
        print(
            f"[OK] Gripper pad collision prim: {collision_path} | "
            f"approx=convexHull | contactOffset={GRIPPER_CONTACT_OFFSET:.4f}",
            flush=True,
        )
        created += 1
    if created:
        print(f"[OK] Gripper rubber pad visual collisions: meshes={created}", flush=True)
    return created


def _bbox_center_dims_np(stage, root_path: Sdf.Path):
    prim = stage.GetPrimAtPath(root_path)
    if not prim.IsValid():
        return None, None
    try:
        bbox_cache = UsdGeom.BBoxCache(
            Usd.TimeCode.Default(),
            [UsdGeom.Tokens.default_, UsdGeom.Tokens.render, UsdGeom.Tokens.proxy],
        )
        world_range = bbox_cache.ComputeWorldBound(prim).ComputeAlignedBox()
        min_pt = np.array(world_range.GetMin(), dtype=float)
        max_pt = np.array(world_range.GetMax(), dtype=float)
        if not np.all(np.isfinite(min_pt)) or not np.all(np.isfinite(max_pt)):
            return None, None
        dims = max_pt - min_pt
        if np.any(dims <= 1.0e-6):
            return None, None
        return 0.5 * (min_pt + max_pt), dims
    except Exception as exc:
        print(f"[WARN] Workpiece bbox failed: {type(exc).__name__}: {exc}", flush=True)
        return None, None


def _world_point_to_prim_local(stage, prim_path: Sdf.Path, world_point: np.ndarray) -> np.ndarray:
    prim = stage.GetPrimAtPath(prim_path)
    if not prim.IsValid():
        return np.asarray(world_point, dtype=float)
    try:
        xform_cache = UsdGeom.XformCache(Usd.TimeCode.Default())
        local_to_world = xform_cache.GetLocalToWorldTransform(prim)
        local = local_to_world.GetInverse().Transform(Gf.Vec3d(float(world_point[0]), float(world_point[1]), float(world_point[2])))
        return np.array([float(local[0]), float(local[1]), float(local[2])], dtype=float)
    except Exception as exc:
        print(f"[WARN] Workpiece local transform failed: {type(exc).__name__}: {exc}", flush=True)
        return np.asarray(world_point, dtype=float)


def _add_workpiece_grip_side_bands(stage) -> int:
    body_path = Sdf.Path("/Root/Root/gongjian")
    body_prim = stage.GetPrimAtPath(body_path)
    if not body_prim.IsValid():
        print(f"[WARN] Workpiece body not found for grip side bands: {body_path}", flush=True)
        return 0

    for child in list(body_prim.GetChildren()):
        if child.GetName().startswith("grip_side_collision_"):
            stage.RemovePrim(child.GetPath())

    bbox_center, bbox_dims = _bbox_center_dims_np(stage, body_path)
    if bbox_center is None or bbox_dims is None:
        print("[WARN] Workpiece grip side bands skipped: invalid bbox", flush=True)
        return 0

    side_length = min(float(bbox_dims[0]) * 0.55, WORKPIECE_GRIP_SIDE_LENGTH)
    side_thickness = min(max(WORKPIECE_GRIP_SIDE_THICKNESS, 0.001), max(float(bbox_dims[1]) * 0.2, 0.001))
    side_height = min(max(WORKPIECE_GRIP_SIDE_HEIGHT, 0.001), max(float(bbox_dims[2]) * 0.95, 0.001))
    # Conservative placement: sit just inside the +/-Y side. This avoids a
    # protruding helper box that can pre-contact and lock the dynamic body.
    y_half = max(0.0, 0.5 * float(bbox_dims[1]) - 0.5 * side_thickness)

    created = 0
    for label, sign in (("neg_y", -1.0), ("pos_y", 1.0)):
        world_center = np.array(
            [
                float(bbox_center[0]),
                float(bbox_center[1]) + sign * y_half,
                float(bbox_center[2]) + WORKPIECE_GRIP_SIDE_Z_OFFSET,
            ],
            dtype=float,
        )
        local_center = _world_point_to_prim_local(stage, body_path, world_center)
        collision_path = body_path.AppendChild(f"grip_side_collision_{label}")
        cube = UsdGeom.Cube.Define(stage, collision_path)
        cube.CreateSizeAttr().Set(1.0)
        if args_cli.hide_contact_boxes:
            UsdGeom.Imageable(cube.GetPrim()).MakeInvisible()
            UsdGeom.Imageable(cube.GetPrim()).CreatePurposeAttr().Set(UsdGeom.Tokens.guide)
        else:
            cube.CreateVisibilityAttr().Set(UsdGeom.Tokens.inherited)
            cube.CreateDisplayColorAttr().Set([Gf.Vec3f(1.0, 0.8, 0.0)])
            cube.CreateDisplayOpacityAttr().Set([0.35])
        cube_xform = UsdGeom.Xformable(cube.GetPrim())
        cube_xform.ClearXformOpOrder()
        cube_xform.AddTranslateOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(float(local_center[0]), float(local_center[1]), float(local_center[2]))
        )
        cube_xform.AddScaleOp(UsdGeom.XformOp.PrecisionDouble).Set(
            Gf.Vec3d(float(side_length), float(side_thickness), float(side_height))
        )
        UsdPhysics.CollisionAPI.Apply(cube.GetPrim()).CreateCollisionEnabledAttr().Set(True)
        PhysxSchema.PhysxCollisionAPI.Apply(cube.GetPrim())
        _set_prim_attr(cube.GetPrim(), "physxCollision:contactOffset", Sdf.ValueTypeNames.Float, WORKPIECE_CONTACT_OFFSET)
        _set_prim_attr(cube.GetPrim(), "physxCollision:restOffset", Sdf.ValueTypeNames.Float, REST_OFFSET)
        created += 1

    print(
        f"[OK] Workpiece grip side bands={created}, axis=Y, "
        f"centerY=±{y_half:.4f} m, "
        f"size=({side_length:.3f}, {side_thickness:.3f}, {side_height:.3f}) m, "
        f"zOffset={WORKPIECE_GRIP_SIDE_Z_OFFSET:+.3f} m",
        flush=True,
    )
    return created

def _write_dynamic_safe_object_usd(source_usd: Path, wrapper_usd: Path) -> Path:
    source_usd = source_usd.expanduser().resolve()
    wrapper_usd = wrapper_usd.expanduser().resolve()
    if not source_usd.exists():
        raise FileNotFoundError(source_usd)
    wrapper_usd.parent.mkdir(parents=True, exist_ok=True)

    source_stage = Usd.Stage.Open(str(source_usd))
    if source_stage is None:
        raise RuntimeError(f"Could not open object USD: {source_usd}")
    source_default = source_stage.GetDefaultPrim()
    source_prim_path = source_default.GetPath() if source_default and source_default.IsValid() else Sdf.Path("/Root")

    stage = Usd.Stage.CreateInMemory()
    root = UsdGeom.Xform.Define(stage, Sdf.Path("/Root")).GetPrim()
    stage.SetDefaultPrim(root)
    if not root.GetReferences().AddReference(str(source_usd), source_prim_path):
        raise RuntimeError(f"Failed to add object reference: {source_usd}:{source_prim_path}")

    for prim_path in ("/Root/Root/gongjian", "/Root/Root/gongjian/gongjian/mesh_"):
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            prim = stage.OverridePrim(prim_path)
        mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
        mesh_collision.CreateApproximationAttr().Set(OBJECT_COLLISION_APPROXIMATION)

    aluminum_physics = stage.GetPrimAtPath("/Root/Root/gongjian/Aluminum_Physics")
    if not aluminum_physics.IsValid():
        aluminum_physics = stage.OverridePrim("/Root/Root/gongjian/Aluminum_Physics")
    aluminum_physics.SetActive(False)

    workpiece_main_material = _define_physics_material(
        stage,
        Sdf.Path("/Root/PhysicsMaterials/workpiece_main_moderate_friction"),
        WORKPIECE_MAIN_STATIC_FRICTION,
        WORKPIECE_MAIN_DYNAMIC_FRICTION,
    )
    workpiece_main_material_count = _bind_material_to_collision_subtree(
        stage,
        Sdf.Path("/Root/Root/gongjian"),
        workpiece_main_material,
        WORKPIECE_CONTACT_OFFSET,
    )
    grip_side_count = _add_workpiece_grip_side_bands(stage)
    workpiece_grip_material = _define_physics_material(
        stage,
        Sdf.Path("/Root/PhysicsMaterials/workpiece_grip_side_high_friction"),
        WORKPIECE_GRIP_STATIC_FRICTION,
        WORKPIECE_GRIP_DYNAMIC_FRICTION,
    )
    grip_side_material_count = 0
    for label in ("neg_y", "pos_y"):
        prim = stage.GetPrimAtPath(Sdf.Path(f"/Root/Root/gongjian/grip_side_collision_{label}"))
        if not prim.IsValid():
            continue
        UsdShade.MaterialBindingAPI.Apply(prim).Bind(workpiece_grip_material)
        _set_contact_patch_attrs(prim, WORKPIECE_CONTACT_OFFSET)
        grip_side_material_count += 1

    stage.GetRootLayer().Export(str(wrapper_usd))
    print(f"[OK] Dynamic-safe object wrapper: {wrapper_usd}", flush=True)
    print(
        f"[OK] Object collision approximation={OBJECT_COLLISION_APPROXIMATION}; "
        f"grip side bands={grip_side_count}; "
        f"main materials={workpiece_main_material_count}, "
        f"main_mu={WORKPIECE_MAIN_STATIC_FRICTION:.2f}/{WORKPIECE_MAIN_DYNAMIC_FRICTION:.2f}; "
        f"grip materials={grip_side_material_count}, "
        f"grip_mu={WORKPIECE_GRIP_STATIC_FRICTION:.2f}/{WORKPIECE_GRIP_DYNAMIC_FRICTION:.2f}; "
        "disabled /Root/Root/gongjian/Aluminum_Physics",
        flush=True,
    )
    return wrapper_usd


def _resolve_object_usd(default_object_usd: str) -> Path:
    if args_cli.object_usd is None:
        graspable_usd = DEFAULT_GRASPABLE_OBJECT_USD.expanduser().resolve()
        if graspable_usd.exists():
            print(f"[OK] Dynamic object USD: {graspable_usd}", flush=True)
            return graspable_usd
        print(f"[WARN] Graspable object USD not found, fallback to env config object: {graspable_usd}", flush=True)
        source_usd = Path(default_object_usd)
    else:
        source_usd = args_cli.object_usd

    source_usd = source_usd.expanduser().resolve()
    if source_usd.name == DEFAULT_GRASPABLE_OBJECT_USD.name:
        print(f"[OK] Dynamic object USD: {source_usd}", flush=True)
        return source_usd
    return _write_dynamic_safe_object_usd(source_usd, args_cli.object_wrapper_usd)


def _resolve_checkpoint(path_str: str | None) -> str | None:
    """把 checkpoint 路径解析成可读路径。"""
    if not path_str:
        return None

    p = Path(path_str).expanduser()
    if p.is_file():
        return str(p)

    p2 = Path("/home/gxai/IsaacLab") / path_str
    if p2.is_file():
        return str(p2)

    p3 = Path(_PROJECT_DIR).parent.parent / path_str
    if p3.is_file():
        return str(p3)

    raise FileNotFoundError(
        f"找不到 checkpoint: {path_str}\n"
        f"尝试过: {p}, {p2}, {p3}"
    )


def _make_env_cfg(robot_usd: Path):
    """创建 eval 环境配置。"""
    env_cfg = PickPoseEnvCfg_PLAY()
    env_cfg.episode_length_s = max(0.1, float(args_cli.eval_episode_length_s))
    if args_cli.arm_set_mode:
        env_cfg.actions = CRTSetActionsCfg()
    env_cfg.scene.robot.spawn.usd_path = str(robot_usd.expanduser().resolve())

    if args_cli.dynamic_object_eval:
        # Match the high-PD dynamic-object path: keep startup local/offline and
        # avoid the default ground asset stalling scene creation.
        env_cfg.scene.ground = None
        object_usd_path = _resolve_object_usd(env_cfg.scene.object.spawn.usd_path)
        env_cfg.scene.object.spawn.usd_path = str(object_usd_path)
        using_graspable_object = object_usd_path.expanduser().resolve().name == DEFAULT_GRASPABLE_OBJECT_USD.name
        env_cfg.scene.object.spawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(
            kinematic_enabled=False,
            disable_gravity=False,
            linear_damping=0.35,
            angular_damping=0.35,
            max_linear_velocity=2.0,
            max_angular_velocity=8.0,
            max_depenetration_velocity=1.5,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=4,
        )
        env_cfg.scene.object.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=max(0.01, float(args_cli.object_mass)))
        env_cfg.scene.object.spawn.collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=True)
        if using_graspable_object:
            # The graspable wrapper has only one hidden collision body, so make
            # the spawn-level material match its high-friction USD material.
            object_static_friction = WORKPIECE_GRIP_STATIC_FRICTION
            object_dynamic_friction = WORKPIECE_GRIP_DYNAMIC_FRICTION
        else:
            object_static_friction = WORKPIECE_MAIN_STATIC_FRICTION
            object_dynamic_friction = WORKPIECE_MAIN_DYNAMIC_FRICTION
        env_cfg.scene.object.spawn.physics_material = sim_utils.RigidBodyMaterialCfg(
            static_friction=object_static_friction,
            dynamic_friction=object_dynamic_friction,
            restitution=0.0,
        )
        if hasattr(env_cfg.scene.object.spawn, "activate_contact_sensors"):
            env_cfg.scene.object.spawn.activate_contact_sensors = True
        for sensor_name, body_path in PAD_CONTACT_SENSOR_BODIES.items():
            setattr(
                env_cfg.scene,
                sensor_name,
                ContactSensorCfg(
                    prim_path=f"{{ENV_REGEX_NS}}{body_path}",
                    filter_prim_paths_expr=OBJECT_CONTACT_FILTER_PATHS,
                    track_friction_forces=True,
                    track_contact_points=True,
                    max_contact_data_count_per_prim=16,
                    history_length=3,
                ),
            )

        if USE_EXTERNAL_PAD_PROXIES:
            env_cfg.scene.left_pad_proxy = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/LeftPadProxy",
                init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.0, -10.0]),
                spawn=sim_utils.CuboidCfg(
                    size=PAD_PROXY_SIZE,
                    activate_contact_sensors=True,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,
                        disable_gravity=True,
                        max_depenetration_velocity=2.0,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=PAD_PROXY_MASS),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=True,
                        contact_offset=GRIPPER_CONTACT_OFFSET,
                        rest_offset=REST_OFFSET,
                    ),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=GRIPPER_STATIC_FRICTION,
                        dynamic_friction=GRIPPER_DYNAMIC_FRICTION,
                        restitution=0.0,
                    ),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(0.0, 0.85, 1.0),
                        opacity=0.35,
                    ),
                ),
            )
            env_cfg.scene.right_pad_proxy = RigidObjectCfg(
                prim_path="{ENV_REGEX_NS}/RightPadProxy",
                init_state=RigidObjectCfg.InitialStateCfg(pos=[0.0, 0.0, -10.1]),
                spawn=sim_utils.CuboidCfg(
                    size=PAD_PROXY_SIZE,
                    activate_contact_sensors=True,
                    rigid_props=sim_utils.RigidBodyPropertiesCfg(
                        kinematic_enabled=True,
                        disable_gravity=True,
                        max_depenetration_velocity=2.0,
                    ),
                    mass_props=sim_utils.MassPropertiesCfg(mass=PAD_PROXY_MASS),
                    collision_props=sim_utils.CollisionPropertiesCfg(
                        collision_enabled=True,
                        contact_offset=GRIPPER_CONTACT_OFFSET,
                        rest_offset=REST_OFFSET,
                    ),
                    physics_material=sim_utils.RigidBodyMaterialCfg(
                        static_friction=GRIPPER_STATIC_FRICTION,
                        dynamic_friction=GRIPPER_DYNAMIC_FRICTION,
                        restitution=0.0,
                    ),
                    visual_material=sim_utils.PreviewSurfaceCfg(
                        diffuse_color=(1.0, 0.0, 0.85),
                        opacity=0.35,
                    ),
                ),
            )

    if bool(getattr(args_cli, "headless", False)):
        for scene_attr in ("robot_support_box", "workpiece_table", "left_pad_proxy", "right_pad_proxy"):
            scene_item = getattr(env_cfg.scene, scene_attr, None)
            if scene_item is not None and getattr(scene_item, "spawn", None) is not None:
                try:
                    scene_item.spawn.visual_material = None
                except Exception:
                    pass

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    else:
        env_cfg.scene.num_envs = 1 if args_cli.play else 64

    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    action_cfg = getattr(getattr(env_cfg, "actions", None), "arm_action", None)
    if action_cfg is not None:
        action_cfg.policy_gripper_enabled = USE_POLICY_GRIPPER
        gripper_overrides = (
            ("gripper_force_target", "force_control_target_n"),
            ("gripper_force_push_effort", "force_control_push_effort_limit"),
            ("gripper_force_kp", "force_control_kp_deg_per_n"),
            ("gripper_force_ki", "force_control_ki_deg_per_n_s"),
            ("gripper_force_kd", "force_control_kd_deg_s_per_n"),
            ("gripper_force_deadband", "force_control_deadband_n"),
            ("gripper_force_max_step", "force_control_max_step_deg"),
            ("gripper_force_hold_error", "force_control_hold_error_deg"),
            ("gripper_force_release_step", "force_control_release_step_deg"),
            ("gripper_hold_force", "hold_normal_force_n"),
            ("gripper_hard_warn_force", "hard_normal_force_n"),
            ("gripper_approach_effort", "approach_effort_limit"),
            ("gripper_hold_effort", "hold_effort_limit"),
            ("gripper_hard_hold_effort", "hard_hold_effort_limit"),
            ("gripper_velocity_limit", "velocity_limit"),
            ("gripper_sync_max_lead", "gripper_sync_max_lead_deg"),
            ("gripper_free_close_step", "free_close_step_deg"),
            ("gripper_contact_close_step", "contact_close_step_deg"),
        )
        for arg_name, cfg_name in gripper_overrides:
            value = getattr(args_cli, arg_name, None)
            if value is not None:
                setattr(action_cfg, cfg_name, float(value))
                print(f"[GripperOverride] {cfg_name}={float(value):.6g}", flush=True)
        if args_cli.gripper_auto_close_dist_cm is not None:
            value = max(0.0, float(args_cli.gripper_auto_close_dist_cm)) / 100.0
            action_cfg.auto_close_dist_m = value
            print(f"[GripperOverride] auto_close_dist_m={value:.6g}", flush=True)
        if args_cli.gripper_auto_close_ori_deg is not None:
            value = max(0.0, float(args_cli.gripper_auto_close_ori_deg))
            action_cfg.auto_close_ori_deg = value
            print(f"[GripperOverride] auto_close_ori_deg={value:.6g}", flush=True)

    set_success_termination_enabled(env_cfg, args_cli.enable_success_termination)

    return env_cfg


def _effective_gripper_close_dist_m() -> float:
    if args_cli.gripper_auto_close_dist_cm is not None:
        return max(0.0, float(args_cli.gripper_auto_close_dist_cm)) / 100.0
    return float(AUTO_GRIPPER_CLOSE_DIST_M)


def _effective_gripper_close_ori_deg() -> float:
    if args_cli.gripper_auto_close_ori_deg is not None:
        return max(0.0, float(args_cli.gripper_auto_close_ori_deg))
    return float(AUTO_GRIPPER_CLOSE_ORI_DEG)


def _effective_arm_hold_close_dist_m() -> float:
    if args_cli.arm_hold_auto_close_dist_cm is not None:
        return max(0.0, float(args_cli.arm_hold_auto_close_dist_cm)) / 100.0
    return _effective_gripper_close_dist_m()


def _effective_arm_hold_close_ori_deg() -> float:
    if args_cli.arm_hold_auto_close_ori_deg is not None:
        return max(0.0, float(args_cli.arm_hold_auto_close_ori_deg))
    return _effective_gripper_close_ori_deg()


def _print_run_header(log_dir: str, num_envs: int, robot_usd: Path):
    gripper_close_dist_m = _effective_gripper_close_dist_m()
    gripper_close_ori_deg = _effective_gripper_close_ori_deg()
    print(f"\n========== {VERSION_TAG} ==========")
    print(f"[Version] {VERSION_TAG}")
    print(f"[Action] scale = {math.degrees(_ARM_ACTION_SCALE):.4f} deg")
    print(f"[Action] clip  = ±{math.degrees(_ARM_CLIP_RAD):.4f} deg")
    print(f"[Action] dim   = {PLAY_ACTION_DIM} ({'7 arm + 1 gripper' if USE_POLICY_GRIPPER else '7 arm'})")
    arm_mode = "set_joint_position_target, no write" if args_cli.arm_set_mode else "baseline write_joint_state_to_sim + lock"
    gripper_mode = "policy 0/1 action" if USE_POLICY_GRIPPER else "script/auto-close outside policy action"
    print(f"[Mode]   arm={arm_mode} | gripper={gripper_mode}")
    print(
        f"[Gripper] close <= {gripper_close_dist_m * 1000.0:.1f} mm / "
        f"{gripper_close_ori_deg:.1f} deg | ramp={AUTO_GRIPPER_CLOSE_TIME_S:.2f}s | "
        f"vel_limit={GRIPPER_CLOSE_VELOCITY_LIMIT:.1f} rad/s | "
        f"effort primary/mimic={GRIPPER_PRIMARY_EFFORT:.1f}/{GRIPPER_MIMIC_EFFORT:.1f}"
    )
    print(f"[Robot]  {robot_usd}")
    object_mode = "dynamic" if args_cli.dynamic_object_eval else "kinematic baseline"
    print(f"[Object] {object_mode}")
    if args_cli.dynamic_object_eval:
        print(f"[Object] mass={args_cli.object_mass:.3f} kg | main_mu={WORKPIECE_MAIN_STATIC_FRICTION:.2f}/{WORKPIECE_MAIN_DYNAMIC_FRICTION:.2f} | grip_mu={WORKPIECE_GRIP_STATIC_FRICTION:.2f}/{WORKPIECE_GRIP_DYNAMIC_FRICTION:.2f}")
    print(f"[Log]    {log_dir}")
    print(f"[Env]    num_envs = {num_envs}")
    print(f"[Config] {ENV_CFG_FILE}")
    success_term_status = (
        "enabled"
        if args_cli.enable_success_termination
        else "disabled (no-success default)"
    )
    print(f"[Reward] success termination: {success_term_status}")
    if abs(float(OBJECT_POSE_Z_OFFSET_M)) > 1.0e-12:
        print(f"[Target] object pose z offset = {OBJECT_POSE_Z_OFFSET_M * 1000.0:.1f} mm")
    if args_cli.play and args_cli.lift_after_auto_close:
        if args_cli.lift_mode == "cartesian_z":
            print(
                f"[Play] lift-after-auto-close: TCP base-z += {args_cli.lift_height_mm:.1f} mm "
                f"with stepwise Pinocchio IK after L/R pad force >= {args_cli.lift_contact_force_threshold:.2f} N "
                f"for {args_cli.lift_after_auto_close_delay:.2f}s, "
                f"cart_step={args_cli.lift_cartesian_step_mm:.1f} mm, "
                f"joint ramp <= {math.degrees(_ARM_CLIP_RAD):.2f} deg/step"
            )
        else:
            print(
                f"[Play] lift-after-auto-close: joint2 += {args_cli.lift_joint2_delta_deg:.2f} deg "
                f"after L/R pad force >= {args_cli.lift_contact_force_threshold:.2f} N "
                f"for {args_cli.lift_after_auto_close_delay:.2f}s, "
                f"ramp <= {math.degrees(_ARM_CLIP_RAD):.2f} deg/step"
            )
    if args_cli.play:
        print(
            "[Play] pure policy replay | "
            f"smooth_alpha={PLAY_ACTION_SMOOTH_ALPHA}, "
            f"stop_on_best={PLAY_ENABLE_STOP_ON_BEST}, "
            f"play_max_steps={args_cli.play_max_steps}"
        )
        if PLAY_ENABLE_STOP_ON_BEST:
            print(
                "[Play] stop-on-best is enabled only near target | "
                f"enable_dist<{PLAY_STOP_ENABLE_DIST_CM} cm, "
                f"enable_ori<{PLAY_STOP_ENABLE_ORI_DEG} deg, "
                f"patience={PLAY_MIN_SCORE_PATIENCE}"
            )
    print("====================================\n")


def _as_first_env_numpy(value, expected_dim: int, name: str) -> np.ndarray:
    """Extract first-env 1D float array from Tensor, dict, or TensorDict-like observations."""
    if torch.is_tensor(value):
        tensor = value
    elif isinstance(value, dict):
        tensor = None
        for key in ("policy", "obs", "observations"):
            if key in value:
                tensor = value[key]
                break
        if tensor is None:
            tensors = [v for v in value.values() if torch.is_tensor(v)]
            if len(tensors) == 1:
                tensor = tensors[0]
        if tensor is None:
            raise TypeError(f"Cannot extract {name}: dict keys={list(value.keys())}")
        return _as_first_env_numpy(tensor, expected_dim, name)
    else:
        tensor = None
        for key in ("policy", "obs", "observations"):
            try:
                candidate = value.get(key, None)
            except Exception:
                candidate = None
            if candidate is not None:
                tensor = candidate
                break
        if tensor is None:
            try:
                values = [v for v in value.values() if torch.is_tensor(v)]
            except Exception:
                values = []
            if len(values) == 1:
                tensor = values[0]
        if tensor is None:
            raise TypeError(f"Cannot extract {name}: type={type(value).__name__}")
        return _as_first_env_numpy(tensor, expected_dim, name)

    arr = tensor.detach().float().cpu().numpy()
    if arr.ndim == 2:
        arr = arr[0]
    elif arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.shape[0] != expected_dim:
        raise ValueError(f"{name} dim mismatch: got {arr.shape[0]}, expected {expected_dim}")
    return arr.astype(np.float32, copy=False)


def _parse_quat_wxyz(text: str, device: torch.device | str) -> torch.Tensor:
    vals = [float(v.strip()) for v in text.split(",") if v.strip()]
    if len(vals) != 4:
        raise ValueError(f"PLAY_ORI_Q_OFFSET_WXYZ 必须是 4 个数，w,x,y,z 格式，当前为: {text}")
    q = torch.tensor(vals, device=device, dtype=torch.float32)
    q = q / torch.clamp(torch.linalg.norm(q), min=1.0e-8)
    return q


def _quat_mul_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """四元数乘法，输入/输出均为 w,x,y,z。"""
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _normalize_quat(q: torch.Tensor) -> torch.Tensor:
    return q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1.0e-8)


def _get_ee_quat_w(ee_frame) -> torch.Tensor:
    """从 FrameTransformer 读取末端四元数，返回 shape=(4,) 的 w,x,y,z。"""
    data = ee_frame.data
    quat = None
    for name in ("target_quat_w", "frame_quat_w", "source_quat_w"):
        if hasattr(data, name):
            quat = getattr(data, name)
            break
    if quat is None:
        raise RuntimeError("无法从 ee_frame.data 中读取 target_quat_w/frame_quat_w/source_quat_w")
    if quat.ndim == 3:
        return quat[0, 0].detach().clone()
    if quat.ndim == 2:
        return quat[0].detach().clone()
    raise RuntimeError(f"未知 ee quat 维度: {quat.shape}")


def _orientation_error_deg(
    ee_quat_w: torch.Tensor,
    obj_quat_w: torch.Tensor,
    q_offset_wxyz: torch.Tensor,
    grasp_symmetry_quats_wxyz: torch.Tensor,
) -> float:
    """末端姿态到最近 180° 对称夹持目标的角度误差，单位 deg。"""
    ee_q = _normalize_quat(ee_quat_w.view(1, 4)).view(4)
    obj_q = _normalize_quat(obj_quat_w.view(1, 4)).view(4)
    offset_q = _normalize_quat(q_offset_wxyz.view(1, 4)).view(4)
    sym_q = _normalize_quat(grasp_symmetry_quats_wxyz.view(-1, 4))

    offsets = _quat_mul_wxyz(offset_q.view(1, 4).expand_as(sym_q), sym_q)
    offsets = _normalize_quat(offsets)
    desired_q = _quat_mul_wxyz(obj_q.view(1, 4).expand_as(offsets), offsets)
    desired_q = _normalize_quat(desired_q)

    # abs(dot) 避免 q 和 -q 双覆盖跳变；max 表示对称候选中取最小姿态误差。
    dots = torch.abs(torch.sum(ee_q.view(1, 4) * desired_q, dim=-1))
    dot = torch.clamp(torch.max(dots), -1.0, 1.0)
    angle_rad = 2.0 * torch.acos(dot)
    return float(torch.rad2deg(angle_rad).item())


def _extract_timeout(extras, dones: torch.Tensor) -> torch.Tensor:
    """从 extras 中提取 timeout 标志；没有则返回全 False。"""
    done_mask = dones.bool().view(-1)
    time_outs = None
    if isinstance(extras, dict):
        time_outs = extras.get("time_outs", None)

    if time_outs is None:
        return torch.zeros_like(done_mask, dtype=torch.bool, device=done_mask.device)
    if not torch.is_tensor(time_outs):
        time_outs = torch.as_tensor(time_outs, dtype=torch.bool, device=done_mask.device)
    else:
        time_outs = time_outs.bool().to(device=done_mask.device)

    time_outs = time_outs.view(-1)
    if time_outs.numel() != done_mask.numel():
        return torch.zeros_like(done_mask, dtype=torch.bool, device=done_mask.device)
    return time_outs



def _read_terminal_cache(base_env, env_id: int = 0):
    """读取环境在 termination 阶段缓存的 reset 前终止误差。

    返回：
        terminal_dist_cm: float | None
        terminal_ori_err_deg: float | None
        terminal_success: bool

    只有配套环境文件把 reach_success 替换为
    ee_reached_object_cached_terminal 后，这里才能读到有效值。
    """
    dist = getattr(base_env, "_terminal_dist_cm", None)
    ori = getattr(base_env, "_terminal_ori_err_deg", None)
    success = getattr(base_env, "_terminal_success", None)

    if dist is None or ori is None or success is None:
        return None, None, False

    try:
        terminal_dist_cm = float(dist[env_id].detach().cpu().item())
        terminal_ori_err_deg = float(ori[env_id].detach().cpu().item())
        terminal_success = bool(success[env_id].detach().cpu().item())
        return terminal_dist_cm, terminal_ori_err_deg, terminal_success
    except Exception:
        return None, None, False


def _get_cmd_delta_deg(base_env, actions: torch.Tensor) -> np.ndarray:
    """读取本 step 的命令关节增量，优先使用 action term 的 processed_actions。"""
    term = _get_arm_action_term(base_env)
    if term is not None and hasattr(term, "processed_actions"):
        try:
            cmd_delta_rad = term.processed_actions[0, :7].detach()
            return torch.rad2deg(cmd_delta_rad).cpu().numpy()
        except Exception:
            pass

    # 兜底：用 action*scale/clip 估算命令增量；夹爪动作不参与关节误差日志。
    act_np = actions[0, :7].detach().cpu().numpy()
    return np.clip(
        act_np * math.degrees(_ARM_ACTION_SCALE),
        -math.degrees(_ARM_CLIP_RAD),
        math.degrees(_ARM_CLIP_RAD),
    )


def _write_arm_joint_state(base_env, robot, joint_ids, arm_q_target: torch.Tensor):
    """仅在 write 模式的手动 hold/lift 中使用：写回指定 7 关节并清零速度。"""
    joint_pos = robot.data.joint_pos.clone()
    joint_vel = robot.data.joint_vel.clone()

    joint_pos[:, joint_ids] = arm_q_target.view(1, -1).to(device=joint_pos.device, dtype=joint_pos.dtype)
    joint_vel[:, joint_ids] = 0.0

    try:
        robot.set_joint_position_target(joint_pos[:, joint_ids], joint_ids=joint_ids)
    except Exception:
        try:
            robot.set_joint_position_target(joint_pos)
        except Exception:
            pass

    try:
        robot.set_joint_velocity_target(torch.zeros_like(joint_pos[:, joint_ids]), joint_ids=joint_ids)
    except Exception:
        try:
            robot.set_joint_velocity_target(torch.zeros_like(joint_vel))
        except Exception:
            pass

    robot.write_joint_state_to_sim(joint_pos, joint_vel)
    try:
        base_env.scene.write_data_to_sim()
    except Exception:
        pass
    try:
        base_env.scene.update(0.0)
    except Exception:
        pass


def _arm_action_towards_joint_target(robot, joint_ids, arm_q_target: torch.Tensor, action_like: torch.Tensor) -> torch.Tensor:
    """set 模式 hold/lift 用正常 action 通道追目标，避免在回放循环里写 joint state。"""
    q_current = robot.data.joint_pos[0, joint_ids].detach()
    q_target = arm_q_target.to(device=q_current.device, dtype=q_current.dtype)
    delta_q = torch.clamp(q_target - q_current, -float(_ARM_CLIP_RAD), float(_ARM_CLIP_RAD))
    scale = max(abs(float(_ARM_ACTION_SCALE)), 1.0e-9)
    actions = torch.zeros_like(action_like)
    actions[0, : len(joint_ids)] = delta_q / scale
    return actions


def _quat_wxyz_to_rot_matrix_np(quat_wxyz: np.ndarray) -> np.ndarray:
    q = np.asarray(quat_wxyz, dtype=float)
    norm = float(np.linalg.norm(q))
    if norm <= 1.0e-12:
        return np.eye(3, dtype=float)
    w, x, y, z = q / norm
    return np.array(
        [
            [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
            [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
            [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
        ],
        dtype=float,
    )


class PinocchioTcpLiftIK:
    """Small Pinocchio IK helper for eval lift: keep TCP pose orientation, raise base-z."""

    def __init__(
        self,
        urdf_path: Path,
        tcp_z_offset_m: float,
        joint_limits_low: torch.Tensor,
        joint_limits_high: torch.Tensor,
        max_iters: int,
        damping: float,
        tol_pos_mm: float,
        tol_ori_deg: float,
    ):
        try:
            import pinocchio as pin
        except Exception as exc:
            raise RuntimeError(
                "Pinocchio is required for --lift_mode cartesian_z. "
                "Run this eval from the isaaclab conda environment."
            ) from exc

        self.pin = pin
        self.urdf_path = Path(urdf_path).expanduser().resolve()
        if not self.urdf_path.exists():
            raise FileNotFoundError(f"Pinocchio IK URDF not found: {self.urdf_path}")

        self.model = pin.buildModelFromUrdf(str(self.urdf_path))
        if not self.model.existFrame("link7"):
            raise RuntimeError(f"Pinocchio IK URDF has no link7 frame: {self.urdf_path}")

        link7_frame_id = self.model.getFrameId("link7")
        link7_frame = self.model.frames[link7_frame_id]
        tcp_offset = pin.SE3(np.eye(3), np.array([0.0, 0.0, float(tcp_z_offset_m)], dtype=float))
        try:
            tcp_frame = pin.Frame(
                "eval_tcp",
                link7_frame.parentJoint,
                link7_frame_id,
                tcp_offset,
                pin.FrameType.OP_FRAME,
            )
        except TypeError:
            tcp_frame = pin.Frame("eval_tcp", link7_frame.parentJoint, tcp_offset, pin.FrameType.OP_FRAME)
        self.tcp_frame_id = self.model.addFrame(tcp_frame)
        self.data = self.model.createData()

        if self.model.nq != 7 or self.model.nv != 7:
            raise RuntimeError(f"Pinocchio IK expects 7-DoF xArm7 URDF, got nq={self.model.nq}, nv={self.model.nv}")
        expected_names = [f"joint{i}" for i in range(1, 8)]
        actual_names = [str(name) for name in self.model.names[1:8]]
        if actual_names != expected_names:
            print(f"[LiftIK] warning: unexpected Pinocchio joint order: {actual_names}", flush=True)

        self.lower = joint_limits_low.detach().cpu().numpy().astype(float).reshape(7)
        self.upper = joint_limits_high.detach().cpu().numpy().astype(float).reshape(7)
        self.max_iters = max(1, int(max_iters))
        self.damping = max(1.0e-8, float(damping))
        self.tol_pos_m = max(0.0, float(tol_pos_mm)) / 1000.0
        self.tol_ori_rad = math.radians(max(0.0, float(tol_ori_deg)))
        self.max_step_rad = max(1.0e-4, min(0.25, float(_ARM_CLIP_RAD) * 4.0))

        print(
            f"[LiftIK] Pinocchio ready | urdf={self.urdf_path} | "
            f"tcp_offset_z={float(tcp_z_offset_m):.3f} m | frame=eval_tcp",
            flush=True,
        )

    def _forward_pose(self, q: np.ndarray):
        q = np.asarray(q, dtype=float).reshape(7)
        self.pin.forwardKinematics(self.model, self.data, q)
        self.pin.updateFramePlacements(self.model, self.data)
        pose = self.data.oMf[self.tcp_frame_id]
        return pose.translation.copy(), pose.rotation.copy()

    def solve_pose(self, q_seed: np.ndarray, target_pos: np.ndarray, target_rot: np.ndarray) -> dict:
        q = np.asarray(q_seed, dtype=float).reshape(7).copy()
        q = np.minimum(np.maximum(q, self.lower), self.upper)
        start_pos, start_rot = self._forward_pose(q)
        target_pos = np.asarray(target_pos, dtype=float).reshape(3).copy()
        target_rot = np.asarray(target_rot, dtype=float).reshape(3, 3).copy()

        pos_err_norm = float("inf")
        ori_err_norm = float("inf")
        iters = 0
        for it in range(self.max_iters):
            iters = it + 1
            self.pin.forwardKinematics(self.model, self.data, q)
            self.pin.computeJointJacobians(self.model, self.data, q)
            self.pin.updateFramePlacements(self.model, self.data)
            current = self.data.oMf[self.tcp_frame_id]

            pos_err = target_pos - current.translation
            # LOCAL_WORLD_ALIGNED frame Jacobian uses world-aligned angular rows.
            ori_err = self.pin.log3(target_rot @ current.rotation.T)
            pos_err_norm = float(np.linalg.norm(pos_err))
            ori_err_norm = float(np.linalg.norm(ori_err))
            if pos_err_norm <= self.tol_pos_m and ori_err_norm <= self.tol_ori_rad:
                break

            err = np.concatenate([pos_err, ori_err]).astype(float)
            jac = self.pin.getFrameJacobian(
                self.model,
                self.data,
                self.tcp_frame_id,
                self.pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
            )
            lhs = jac @ jac.T + (self.damping * self.damping) * np.eye(6)
            try:
                dq = jac.T @ np.linalg.solve(lhs, err)
            except np.linalg.LinAlgError:
                dq = np.linalg.pinv(jac, rcond=1.0e-4) @ err
            dq_norm = float(np.linalg.norm(dq))
            if dq_norm > self.max_step_rad:
                dq *= self.max_step_rad / max(dq_norm, 1.0e-12)
            q = self.pin.integrate(self.model, q, dq)
            q = np.minimum(np.maximum(q, self.lower), self.upper)

        final_pos, final_rot = self._forward_pose(q)
        final_pos_err = target_pos - final_pos
        final_ori_err = self.pin.log3(target_rot @ final_rot.T)
        pos_err_norm = float(np.linalg.norm(final_pos_err))
        ori_err_norm = float(np.linalg.norm(final_ori_err))
        success = pos_err_norm <= max(self.tol_pos_m, 1.0e-9) and ori_err_norm <= max(self.tol_ori_rad, 1.0e-9)
        return {
            "q": q.astype(np.float32),
            "success": success,
            "iters": iters,
            "start_pos": start_pos,
            "start_rot": start_rot,
            "target_pos": target_pos,
            "target_rot": target_rot,
            "final_pos": final_pos,
            "final_rot": final_rot,
            "pos_err_m": pos_err_norm,
            "ori_err_rad": ori_err_norm,
        }

    def solve_lift(self, q_seed: np.ndarray, lift_height_m: float) -> dict:
        q = np.asarray(q_seed, dtype=float).reshape(7).copy()
        q = np.minimum(np.maximum(q, self.lower), self.upper)
        start_pos, start_rot = self._forward_pose(q)
        target_pos = start_pos + np.array([0.0, 0.0, max(0.0, float(lift_height_m))], dtype=float)
        return self.solve_pose(q, target_pos, start_rot)


class ObjectFrameAxesViz:
    """Viewport helper that draws the live RigidObject root frame."""

    def __init__(self, axis_length: float, line_width: float = 0.008):
        self.axis_length = max(0.0, float(axis_length))
        self.line_width = max(0.001, float(line_width))
        self.enabled = self.axis_length > 0.0
        self.stage = None
        self.curves = {}
        if not self.enabled:
            return
        try:
            self.stage = omni.usd.get_context().get_stage()
            root_path = Sdf.Path("/World/ObjectRootFrameAxes")
            UsdGeom.Xform.Define(self.stage, root_path)
            for name, color in (
                ("x", Gf.Vec3f(1.0, 0.0, 0.0)),
                ("y", Gf.Vec3f(0.0, 1.0, 0.0)),
                ("z", Gf.Vec3f(0.0, 0.25, 1.0)),
            ):
                curve = UsdGeom.BasisCurves.Define(self.stage, root_path.AppendChild(f"axis_{name}"))
                curve.CreateTypeAttr().Set(UsdGeom.Tokens.linear)
                curve.CreateCurveVertexCountsAttr().Set([2])
                curve.CreateWidthsAttr().Set([self.line_width, self.line_width])
                curve.CreateDisplayColorAttr().Set([color])
                curve.CreatePointsAttr().Set([Gf.Vec3f(0.0, 0.0, 0.0), Gf.Vec3f(0.0, 0.0, 0.0)])
                self.curves[name] = curve
            print(
                f"[ObjectFrameViz] enabled | root=/World/ObjectRootFrameAxes | "
                f"length={self.axis_length:.3f} m | x=red y=green z=blue",
                flush=True,
            )
        except Exception as exc:
            self.enabled = False
            print(f"[ObjectFrameViz] disabled: {type(exc).__name__}: {exc}", flush=True)

    def update(self, pos_w, quat_wxyz) -> None:
        if not self.enabled:
            return
        try:
            pos = np.asarray(pos_w.detach().cpu().numpy() if torch.is_tensor(pos_w) else pos_w, dtype=float).reshape(3)
            quat = np.asarray(
                quat_wxyz.detach().cpu().numpy() if torch.is_tensor(quat_wxyz) else quat_wxyz,
                dtype=float,
            ).reshape(4)
            rot = _quat_wxyz_to_rot_matrix_np(quat)
            for idx, name in enumerate(("x", "y", "z")):
                end = pos + rot[:, idx] * self.axis_length
                self.curves[name].GetPointsAttr().Set(
                    [
                        Gf.Vec3f(float(pos[0]), float(pos[1]), float(pos[2])),
                        Gf.Vec3f(float(end[0]), float(end[1]), float(end[2])),
                    ]
                )
        except Exception as exc:
            self.enabled = False
            print(f"[ObjectFrameViz] update disabled: {type(exc).__name__}: {exc}", flush=True)

def _compute_stop_score(
    dist_cm: float,
    ori_err_deg: float,
    dist_weight: float,
    ori_weight: float,
    dist_ref_cm: float,
    ori_ref_deg: float,
) -> float:
    dist_ref_cm = max(float(dist_ref_cm), 1.0e-6)
    ori_ref_deg = max(float(ori_ref_deg), 1.0e-6)
    return float(dist_weight * (dist_cm / dist_ref_cm) + ori_weight * (ori_err_deg / ori_ref_deg))



def _force_normal_sum(tensor: torch.Tensor | None, axis_w: torch.Tensor | None, env_index: int = 0) -> float:
    if tensor is None:
        return 0.0
    try:
        row = tensor[env_index]
    except Exception:
        return 0.0
    if row.numel() == 0:
        return 0.0
    forces = torch.nan_to_num(row.detach(), nan=0.0, posinf=0.0, neginf=0.0).reshape(-1, 3)
    if axis_w is None:
        normal = torch.linalg.norm(forces, dim=-1).sum()
    else:
        normal = torch.abs(torch.sum(forces.to(axis_w.device) * axis_w.view(1, 3), dim=-1)).sum()
    return float(normal.cpu())


def _pad_axis_w(scene, env_index: int = 0) -> torch.Tensor | None:
    try:
        robot = scene["robot"]
        body_ids, _ = robot.find_bodies(["left_pad", "right_pad"], preserve_order=True)
        body_ids = [int(v) for v in list(body_ids)]
        if len(body_ids) < 2:
            return None
        left = robot.data.body_pos_w[env_index, body_ids[0]].detach()
        right = robot.data.body_pos_w[env_index, body_ids[1]].detach()
        axis = right - left
        norm = torch.linalg.norm(axis)
        if float(norm.cpu()) <= 1.0e-8:
            return None
        return axis / norm
    except Exception:
        return None


def _pad_normal_force_n(scene, sensor_name: str, axis_w: torch.Tensor | None, env_index: int = 0) -> float:
    try:
        sensor = scene[sensor_name]
    except Exception:
        try:
            sensor = scene.sensors[sensor_name]
        except Exception:
            return 0.0
    data = getattr(sensor, "data", None)
    if data is None:
        return 0.0
    for attr_name in ("force_matrix_w", "net_forces_w", "net_forces_w_history"):
        normal = _force_normal_sum(getattr(data, attr_name, None), axis_w, env_index)
        if normal > 1.0e-6:
            return normal
    return 0.0


def _pad_normal_forces_n(scene) -> tuple[float, float, float]:
    axis_w = _pad_axis_w(scene)
    left = _pad_normal_force_n(scene, "left_pad_contact", axis_w)
    right = _pad_normal_force_n(scene, "right_pad_contact", axis_w)
    return left, right, left + right


def _resolve_gripper_chain_body_ids(robot) -> tuple[list[int], list[str], int | None]:
    body_ids: list[int] = []
    body_names: list[str] = []
    seen: set[int] = set()
    try:
        ids, names = robot.find_bodies(list(GRIPPER_CHAIN_BODY_PATTERNS), preserve_order=True)
        for body_id, body_name in zip(list(ids), list(names), strict=False):
            body_id = int(body_id)
            if body_id in seen:
                continue
            seen.add(body_id)
            body_ids.append(body_id)
            body_names.append(str(body_name))
    except Exception as exc:
        print(f"[GripperChain] body resolve skipped: {type(exc).__name__}: {exc}", flush=True)
    anchor_body_id = None
    try:
        anchor_ids, _ = robot.find_bodies(["link7"], preserve_order=True)
        anchor_ids = [int(v) for v in list(anchor_ids)]
        if anchor_ids:
            anchor_body_id = anchor_ids[0]
    except Exception:
        anchor_body_id = None
    return body_ids, body_names, anchor_body_id


def _gripper_chain_metrics(
    robot,
    chain_body_ids: list[int],
    anchor_body_id: int | None,
    pad_body_ids: dict[str, int] | None,
    *,
    max_span_m: float,
    max_pad_distance_m: float,
    env_index: int = 0,
) -> tuple[float, float, bool]:
    if not chain_body_ids or anchor_body_id is None:
        return float("nan"), float("nan"), False
    try:
        body_pos = robot.data.body_pos_w[env_index].detach()
        ids_tensor = torch.as_tensor(chain_body_ids, device=body_pos.device, dtype=torch.long)
        chain_pos = body_pos.index_select(0, ids_tensor)
        anchor_pos = body_pos[int(anchor_body_id)]
        finite = bool(torch.isfinite(chain_pos).all().item() and torch.isfinite(anchor_pos).all().item())
        if not finite:
            return float("nan"), float("nan"), True
        span = torch.linalg.norm(chain_pos - anchor_pos.view(1, 3), dim=-1).max()
        span_m = float(span.detach().cpu().item())
        pad_distance_m = float("nan")
        if pad_body_ids and "left" in pad_body_ids and "right" in pad_body_ids:
            left_pos = body_pos[int(pad_body_ids["left"])]
            right_pos = body_pos[int(pad_body_ids["right"])]
            if bool(torch.isfinite(left_pos).all().item() and torch.isfinite(right_pos).all().item()):
                pad_distance_m = float(torch.linalg.norm(left_pos - right_pos).detach().cpu().item())
            else:
                return span_m, float("nan"), True
        broken = span_m > max(0.0, float(max_span_m))
        if math.isfinite(pad_distance_m):
            broken = broken or pad_distance_m > max(0.0, float(max_pad_distance_m))
        return span_m, pad_distance_m, bool(broken)
    except Exception:
        return float("nan"), float("nan"), False


class ContactForcePlotWindow:
    def __init__(self, history_len: int = 240):
        self.history_len = int(history_len)
        self.left: list[float] = []
        self.right: list[float] = []
        self.total: list[float] = []
        self.enabled = ui is not None and not bool(getattr(args_cli, "headless", False))
        self.window = None
        self.plot_frame = None
        self.label = None
        if not self.enabled:
            return
        try:
            self.window = ui.Window("CRT Pad Contact Force", width=620, height=320, visible=True)
            with self.window.frame:
                with ui.VStack(spacing=6):
                    ui.Label("Pad normal contact force (N): left=blue, right=amber, total=red", height=24)
                    self.label = ui.Label("L 0.00  R 0.00  T 0.00 N", height=22)
                    self.plot_frame = ui.Frame(height=230)
                    self.plot_frame.set_build_fn(self._build_plot)
        except Exception as exc:
            self.enabled = False
            print(f"[ContactForceWindow] disabled: {type(exc).__name__}: {exc}", flush=True)

    def _series(self, values: list[float]) -> list[float]:
        if not values:
            return [0.0, 0.0]
        return [float(v) if math.isfinite(float(v)) else 0.0 for v in values]

    def _plot(self, values: list[float], color: int, high: float):
        try:
            ui.Plot(ui.Type.LINE, *self._series(values), min=0.0, max=high, height=68, style={"color": color})
        except Exception:
            ui.Label(f"plot unavailable latest={values[-1] if values else 0.0:.2f}", height=68)

    def _build_plot(self):
        high = max(5.0, max(self.total[-self.history_len:] or [0.0]) * 1.2)
        with ui.VStack(spacing=4):
            self._plot(self.left[-self.history_len:], 0xFF4EA1FF, high)
            self._plot(self.right[-self.history_len:], 0xFFFFC04D, high)
            self._plot(self.total[-self.history_len:], 0xFFFF5A5F, high)

    def update(self, left_n: float, right_n: float, total_n: float):
        self.left.append(float(left_n))
        self.right.append(float(right_n))
        self.total.append(float(total_n))
        if len(self.total) > self.history_len:
            self.left = self.left[-self.history_len:]
            self.right = self.right[-self.history_len:]
            self.total = self.total[-self.history_len:]
        if not self.enabled:
            return
        if self.label is not None:
            self.label.text = f"L {left_n:.2f}  R {right_n:.2f}  T {total_n:.2f} N"
        if self.plot_frame is not None:
            self.plot_frame.rebuild()


class GripperPDPlotWindow:
    def __init__(self, base_env, history_len: int = 240):
        self.base_env = base_env
        self.history_len = int(history_len)
        self.target: list[float] = []
        self.left: list[float] = []
        self.right: list[float] = []
        self.mean: list[float] = []
        self.error: list[float] = []
        self.u_deg: list[float] = []
        self.delta_deg: list[float] = []
        self.actual_step_deg: list[float] = []
        self._prev_mean_deg: float | None = None
        self.enabled = ui is not None and not bool(getattr(args_cli, "headless", False))
        self.window = None
        self.plot_frame = None
        self.label = None
        self.action_term = None
        self.last_values: dict[str, float] = {}
        if not self.enabled:
            return
        try:
            self.action_term = self._find_controller()
            self.window = ui.Window("CRT Gripper Raw PD Output", width=720, height=620, visible=True)
            with self.window.frame:
                with ui.VStack(spacing=6):
                    ui.Label("RAW force-PD output and lower-level pad execution, deg", height=24)
                    ui.Label("u_deg=purple raw PD output; delta=cyan clipped target step; actual step=green", height=22)
                    self.label = ui.Label("u 0.00 | delta 0.00 | actual_step 0.00 | target 0.00 mean 0.00 err 0.00", height=22)
                    self.plot_frame = ui.Frame(height=520)
                    self.plot_frame.set_build_fn(self._build_plot)
            print("[GripperPDPlotWindow] enabled", flush=True)
        except Exception as exc:
            self.enabled = False
            print(f"[GripperPDPlotWindow] disabled: {type(exc).__name__}: {exc}", flush=True)

    def _find_controller(self):
        term = _get_arm_action_term(self.base_env)
        controller = getattr(term, "gripper_controller", None) if term is not None else None
        if controller is not None and hasattr(controller, "debug_plot_values"):
            self.action_term = controller
            return controller
        if term is not None and hasattr(term, "debug_plot_values"):
            self.action_term = term
            return term
        if self.action_term is not None:
            return self.action_term
        return None

    def _series(self, values: list[float]) -> list[float]:
        if not values:
            return [0.0, 0.0]
        return [float(v) if math.isfinite(float(v)) else 0.0 for v in values]

    def _plot(self, values: list[float], color: int, low: float, high: float, height: int = 58):
        try:
            ui.Plot(ui.Type.LINE, *self._series(values), min=low, max=high, height=height, style={"color": color})
        except Exception:
            ui.Label(f"plot unavailable latest={values[-1] if values else 0.0:.2f}", height=height)

    def _append(self, attr: str, value: float) -> None:
        values = getattr(self, attr)
        values.append(float(value))
        if len(values) > self.history_len:
            setattr(self, attr, values[-self.history_len:])

    def _build_plot(self):
        angle_values = self.target[-self.history_len:] + self.left[-self.history_len:] + self.right[-self.history_len:] + self.mean[-self.history_len:]
        angle_high = max(5.0, max(angle_values or [0.0]) * 1.08)
        angle_low = min(0.0, min(angle_values or [0.0]) * 1.08)
        err_abs = max(1.0, max([abs(v) for v in self.error[-self.history_len:]] or [0.0]) * 1.2)
        u_abs = max(1.0, max([abs(v) for v in self.u_deg[-self.history_len:]] or [0.0]) * 1.2)
        step_abs = max(0.5, max([abs(v) for v in (self.delta_deg[-self.history_len:] + self.actual_step_deg[-self.history_len:])] or [0.0]) * 1.2)
        with ui.VStack(spacing=4):
            ui.Label("RAW PD OUTPUT: u_deg = Kp*e + Ki*I + Kd*D", height=18)
            self._plot(self.u_deg[-self.history_len:], 0xFFB388FF, -u_abs, u_abs, height=72)
            ui.Label("lower-level command vs execution per frame: clipped delta_deg / actual_mean_step", height=18)
            self._plot(self.delta_deg[-self.history_len:], 0xFF00D4FF, -step_abs, step_abs, height=52)
            self._plot(self.actual_step_deg[-self.history_len:], 0xFF5DDB7A, -step_abs, step_abs, height=52)
            ui.Label("position target sent to drive vs actual pad joint mean", height=18)
            self._plot(self.target[-self.history_len:], 0xFF4EA1FF, angle_low, angle_high, height=52)
            self._plot(self.mean[-self.history_len:], 0xFFFFFFFF, angle_low, angle_high, height=52)
            ui.Label("actual pad joint scalar: left / right", height=18)
            self._plot(self.left[-self.history_len:], 0xFF5DDB7A, angle_low, angle_high, height=42)
            self._plot(self.right[-self.history_len:], 0xFFFFC04D, angle_low, angle_high, height=42)
            ui.Label("position tracking error: target - actual_mean", height=18)
            self._plot(self.error[-self.history_len:], 0xFFFF5A5F, -err_abs, err_abs, height=52)

    def update(self) -> None:
        controller = self._find_controller()
        if controller is None or not hasattr(controller, "debug_plot_values"):
            return
        values = controller.debug_plot_values(0)
        target = float(values.get("target_deg", 0.0))
        left = float(values.get("actual_left_deg", 0.0))
        right = float(values.get("actual_right_deg", 0.0))
        mean = float(values.get("actual_mean_deg", 0.0))
        error = float(values.get("tracking_error_deg", target - mean))
        u_deg = float(values.get("u_deg", 0.0))
        delta_deg = float(values.get("delta_deg", 0.0))
        actual_step_deg = 0.0 if self._prev_mean_deg is None else mean - self._prev_mean_deg
        self.last_values = dict(values)
        self.last_values.update({
            "target_deg": target,
            "actual_left_deg": left,
            "actual_right_deg": right,
            "actual_mean_deg": mean,
            "tracking_error_deg": error,
            "u_deg": u_deg,
            "delta_deg": delta_deg,
            "actual_step_deg": actual_step_deg,
        })
        self._prev_mean_deg = mean
        self._append("target", target)
        self._append("left", left)
        self._append("right", right)
        self._append("mean", mean)
        self._append("error", error)
        self._append("u_deg", u_deg)
        self._append("delta_deg", delta_deg)
        self._append("actual_step_deg", actual_step_deg)
        if not self.enabled:
            return
        if self.label is not None:
            self.label.text = (
                f"u {u_deg:.2f} | delta {delta_deg:.2f} | actual_step {actual_step_deg:.2f} | "
                f"target {target:.2f} mean {mean:.2f} err {error:.2f} deg"
            )
        if self.plot_frame is not None:
            self.plot_frame.rebuild()


class PadProxyOffsetWindow:
    def __init__(self):
        self.enabled = ui is not None and not bool(getattr(args_cli, "headless", False)) and bool(args_cli.dynamic_object_eval)
        self.window = None
        self.labels: dict[tuple[str, int], object] = {}
        if not self.enabled:
            return
        try:
            self.window = ui.Window("CRT Pad Proxy Offset", width=620, height=360, visible=True)
            self._build()
            print("[PadProxyOffsetWindow] enabled: tune left/right proxy local offsets in mm", flush=True)
        except Exception as exc:
            self.enabled = False
            print(f"[PadProxyOffsetWindow] disabled: {type(exc).__name__}: {exc}", flush=True)

    def _build(self):
        with self.window.frame:
            with ui.VStack(spacing=6):
                ui.Label("Pad proxy local offset, mm. Left=cyan, Right=magenta", height=24)
                ui.Label("Drag while watching contact boxes and force plot; values apply immediately.", height=22)
                ui.Line(height=2)
                self._side_group("left", "Left / cyan")
                ui.Line(height=2)
                self._side_group("right", "Right / magenta")

    def _side_group(self, side: str, title: str):
        ui.Label(title, height=22)
        for axis_idx, axis_name in enumerate(("X", "Y", "Z")):
            self._offset_slider(side, axis_idx, axis_name)

    def _offset_slider(self, side: str, axis_idx: int, axis_name: str):
        current_mm = float(PAD_PROXY_LOCAL_OFFSETS[side][axis_idx]) * 1000.0
        with ui.HStack(height=28, spacing=8):
            ui.Label(f"{axis_name}", width=28)
            value_label = ui.Label(f"{current_mm:+.2f} mm", width=90)
            slider = ui.FloatSlider(min=-12.0, max=12.0, step=0.25, height=18)
            slider.model.set_value(current_mm)
            self.labels[(side, axis_idx)] = value_label

            def on_change(model, item_side=side, item_axis=axis_idx):
                value_mm = float(model.get_value_as_float())
                old = list(PAD_PROXY_LOCAL_OFFSETS[item_side])
                old[item_axis] = value_mm / 1000.0
                PAD_PROXY_LOCAL_OFFSETS[item_side] = tuple(old)
                label = self.labels.get((item_side, item_axis))
                if label is not None:
                    label.text = f"{value_mm:+.2f} mm"

            slider.model.add_value_changed_fn(on_change)


class RuntimeTuningPanel:
    """Small live tuning UI for CRT gripper PD stiffness and contact friction."""

    GRIPPER_JOINT_NAMES = (
        "leftfinger_joint",
        "rightfinger_joint",
        "left_kckle_joint",
        "right_kckle_joint",
        "leftinn_joint",
        "rightinn_joint",
    )
    GRIPPER_ACTUATOR_NAMES = ("crt_gripper_primary", "crt_gripper_mimic")

    def __init__(self, base_env, robot):
        self.base_env = base_env
        self.robot = robot
        self.enabled = ui is not None and not bool(getattr(args_cli, "headless", False))
        self.window = None
        self.status_label = None
        self.value_labels: dict[str, object] = {}
        self.sliders: dict[str, object] = {}
        self.gripper_joint_ids: list[int] = []
        self.gripper_joint_names: list[str] = []
        self.last_material_counts = {"gripper": 0, "object": 0}
        self.values = {
            "gripper_stiffness": float(max(GRIPPER_PRIMARY_STIFFNESS, GRIPPER_MIMIC_STIFFNESS)),
            "gripper_static_friction": float(GRIPPER_STATIC_FRICTION),
            "gripper_dynamic_friction": float(GRIPPER_DYNAMIC_FRICTION),
            "object_static_friction": float(WORKPIECE_GRIP_STATIC_FRICTION),
            "object_dynamic_friction": float(WORKPIECE_GRIP_DYNAMIC_FRICTION),
        }
        if not self.enabled:
            return
        self._resolve_gripper_joints()
        try:
            self.window = ui.Window("CRT Runtime Tuning", width=520, height=260, visible=True)
            self._build()
            self._set_status("Ready: sliders apply immediately.")
            print(
                "[RuntimeTuningPanel] enabled: gripper stiffness/friction + object friction",
                flush=True,
            )
        except Exception as exc:
            self.enabled = False
            print(f"[RuntimeTuningPanel] disabled: {type(exc).__name__}: {exc}", flush=True)

    def _resolve_gripper_joints(self) -> None:
        try:
            joint_ids, joint_names = self.robot.find_joints(list(self.GRIPPER_JOINT_NAMES), preserve_order=True)
        except TypeError:
            joint_ids, joint_names = self.robot.find_joints(list(self.GRIPPER_JOINT_NAMES))
        except Exception as exc:
            print(f"[RuntimeTuningPanel] gripper joint lookup failed: {type(exc).__name__}: {exc}", flush=True)
            return
        self.gripper_joint_ids = [int(v) for v in list(joint_ids)]
        self.gripper_joint_names = [str(v) for v in list(joint_names)]

    def _build(self) -> None:
        with self.window.frame:
            with ui.VStack(spacing=6):
                ui.Label("Live CRT tuning: gripper stiffness and PhysX friction", height=22)
                ui.Label("Applies to current simulation only; not saved to the config file.", height=20)
                ui.Line(height=2)
                self._slider("Grip Stiff", "gripper_stiffness", 20.0, 1200.0, self.values["gripper_stiffness"], 5.0, 1)
                self._slider("Grip Fric S", "gripper_static_friction", 0.0, 8.0, self.values["gripper_static_friction"], 0.05, 2)
                self._slider("Grip Fric D", "gripper_dynamic_friction", 0.0, 8.0, self.values["gripper_dynamic_friction"], 0.05, 2)
                self._slider("Obj Fric S", "object_static_friction", 0.0, 8.0, self.values["object_static_friction"], 0.05, 2)
                self._slider("Obj Fric D", "object_dynamic_friction", 0.0, 8.0, self.values["object_dynamic_friction"], 0.05, 2)
                ui.Line(height=2)
                self.status_label = ui.Label("", height=24)

    def _slider(
        self,
        label: str,
        key: str,
        low: float,
        high: float,
        value: float,
        step: float,
        precision: int,
    ) -> None:
        with ui.HStack(height=28, spacing=8):
            ui.Label(label, width=96)
            slider = ui.FloatSlider(min=low, max=high, step=step, height=18)
            value_label = ui.Label(f"{float(value):.{precision}f}", width=76)
        slider.model.set_value(float(value))
        slider.model.add_value_changed_fn(
            lambda model, item=key, item_precision=precision: self._on_slider(
                item,
                float(model.get_value_as_float()),
                item_precision,
            )
        )
        self.sliders[key] = slider
        self.value_labels[key] = value_label

    def _set_status(self, message: str) -> None:
        if self.status_label is not None:
            self.status_label.text = str(message)

    def _on_slider(self, key: str, value: float, precision: int) -> None:
        value = max(0.0, float(value))
        self.values[key] = value
        label = self.value_labels.get(key)
        if label is not None:
            label.text = f"{value:.{precision}f}"
        try:
            if key == "gripper_stiffness":
                self._apply_gripper_stiffness(value)
            elif key.startswith("gripper_"):
                self._apply_material_friction(
                    "gripper",
                    self.values["gripper_static_friction"],
                    self.values["gripper_dynamic_friction"],
                )
            elif key.startswith("object_"):
                self._apply_material_friction(
                    "object",
                    self.values["object_static_friction"],
                    self.values["object_dynamic_friction"],
                )
        except Exception as exc:
            self._set_status(f"Update failed: {type(exc).__name__}: {exc}")
            print(f"[RuntimeTuningPanel] update failed: {type(exc).__name__}: {exc}", flush=True)

    def _fill_actuator_tensor(self, actuator_name: str, attr_name: str, value: float) -> None:
        actuators = getattr(self.robot, "actuators", {})
        actuator = actuators.get(actuator_name) if isinstance(actuators, dict) else None
        if actuator is None:
            return
        tensor = getattr(actuator, attr_name, None)
        if torch.is_tensor(tensor):
            tensor.fill_(float(value))
        elif tensor is not None:
            try:
                setattr(actuator, attr_name, float(value))
            except Exception:
                pass

    def _apply_gripper_stiffness(self, stiffness: float) -> None:
        if not self.gripper_joint_ids:
            self._set_status("No gripper joints found; stiffness not applied.")
            return
        stiffness = max(0.0, float(stiffness))
        for actuator_name in self.GRIPPER_ACTUATOR_NAMES:
            self._fill_actuator_tensor(actuator_name, "stiffness", stiffness)
        self.robot.write_joint_stiffness_to_sim(stiffness, joint_ids=self.gripper_joint_ids)
        try:
            self.base_env.scene.write_data_to_sim()
        except Exception:
            pass
        self._set_status(f"Grip stiffness = {stiffness:.1f} on {len(self.gripper_joint_ids)} joints")
        print(
            f"[RuntimeTuningPanel] gripper stiffness={stiffness:.2f} joints={self.gripper_joint_names}",
            flush=True,
        )

    def _stage(self):
        try:
            return omni.usd.get_context().get_stage()
        except Exception:
            return None

    def _is_material_prim(self, prim) -> bool:
        try:
            if str(prim.GetTypeName()) == "Material":
                return True
        except Exception:
            pass
        try:
            return bool(prim.HasAPI(UsdPhysics.MaterialAPI))
        except Exception:
            return False

    def _material_matches(self, prim, kind: str) -> bool:
        path = str(prim.GetPath()).lower()
        if kind == "gripper":
            return (
                "gripper_high_friction" in path
                or ("/robot/" in path and "physicsmaterials" in path and "gripper" in path)
                or "leftpadproxy" in path
                or "rightpadproxy" in path
            )
        if kind == "object":
            return (
                ("/object/" in path and ("material" in path or "physicsmaterials" in path))
                or "workpiece" in path
                or "gongjian" in path and "material" in path
                or "grasp_collision" in path
                or "grip_side" in path and "material" in path
            )
        return False

    def _apply_material_friction(self, kind: str, static_friction: float, dynamic_friction: float) -> None:
        stage = self._stage()
        if stage is None:
            self._set_status("No USD stage; material update skipped.")
            return
        static_friction = max(0.0, float(static_friction))
        dynamic_friction = max(0.0, float(dynamic_friction))
        count = 0
        for prim in stage.Traverse():
            if not self._is_material_prim(prim) or not self._material_matches(prim, kind):
                continue
            material_api = UsdPhysics.MaterialAPI.Apply(prim)
            material_api.CreateStaticFrictionAttr().Set(static_friction)
            material_api.CreateDynamicFrictionAttr().Set(dynamic_friction)
            material_api.CreateRestitutionAttr().Set(0.0)
            PhysxSchema.PhysxMaterialAPI.Apply(prim)
            _set_prim_attr(prim, "physxMaterial:frictionCombineMode", Sdf.ValueTypeNames.Token, "max")
            _set_prim_attr(prim, "physxMaterial:restitutionCombineMode", Sdf.ValueTypeNames.Token, "min")
            _set_prim_attr(prim, "physxMaterial:improvePatchFriction", Sdf.ValueTypeNames.Bool, True)
            count += 1
        self.last_material_counts[kind] = count
        label = "gripper" if kind == "gripper" else "object"
        self._set_status(f"{label} mu = {static_friction:.2f}/{dynamic_friction:.2f}, mats={count}")
        print(
            f"[RuntimeTuningPanel] {label} friction static/dynamic={static_friction:.3f}/{dynamic_friction:.3f} material_count={count}",
            flush=True,
        )


class ContactForceLimitPanel:
    """Runtime UI for the gripper contact-force limiting state machine."""

    def __init__(self, base_env, force_limit_state: dict[str, float], manual_control_state: dict[str, bool] | None = None):
        self.base_env = base_env
        self.force_limit_state = force_limit_state
        self.manual_control_state = manual_control_state if manual_control_state is not None else {}
        self.enabled = ui is not None and not bool(getattr(args_cli, "headless", False))
        self.window = None
        self.status_label = None
        self.value_labels: dict[str, object] = {}
        self.sliders: dict[str, object] = {}
        self.precisions: dict[str, int] = {}
        self._syncing_ui = False
        self.action_term = None
        if not self.enabled:
            return
        try:
            self.action_term = self._find_action_term()
            self.window = ui.Window("CRT Contact Force Limit", width=560, height=720, visible=True)
            self._build()
            self._set_status("Ready: force limits apply immediately.")
            print("[ContactForceLimitPanel] enabled", flush=True)
        except Exception as exc:
            self.enabled = False
            print(f"[ContactForceLimitPanel] disabled: {type(exc).__name__}: {exc}", flush=True)

    def _find_action_term(self):
        term = _get_arm_action_term(self.base_env)
        controller = getattr(term, "gripper_controller", None) if term is not None else None
        if controller is not None and hasattr(controller, "_hold_effort_limit"):
            self.action_term = controller
            return controller
        if term is not None and hasattr(term, "_hold_effort_limit"):
            self.action_term = term
            return term
        if self.action_term is not None:
            return self.action_term
        return term

    def _value_from_term(self, attr_name: str, default_value: float) -> float:
        term = self._find_action_term()
        if term is None:
            return float(default_value)
        return float(getattr(term, attr_name, default_value))

    def _build(self) -> None:
        with self.window.frame:
            with ui.VStack(spacing=6):
                ui.Label("Live force-limit tuning for CRT gripper contact", height=22)
                ui.Label("Close/Hold effort is joint effort limit; forces are pad normal thresholds in N.", height=20)
                with ui.HStack(height=28, spacing=8):
                    start_pick_btn = ui.Button("Start Pick", height=24)
                    start_pick_btn.set_clicked_fn(self._start_pick)
                    reset_env_btn = ui.Button("Reset Env", height=24)
                    reset_env_btn.set_clicked_fn(self._reset_env)
                ui.Line(height=2)
                self._slider("Close MaxF", "approach_effort_limit", 1.0, 120.0, self._value_from_term("_approach_effort_limit", GRIPPER_PRIMARY_EFFORT), 1.0, 1)
                self._slider("Hold MaxF", "hold_effort_limit", 1.0, 120.0, self._value_from_term("_hold_effort_limit", GRIPPER_HOLD_EFFORT), 1.0, 1)
                self._slider("Hold Force", "hold_normal_force_n", 0.0, 80.0, self._value_from_term("_hold_normal_force_n", GRIPPER_HOLD_NORMAL_FORCE_N), 0.5, 1)
                self._slider("Hard Warn", "hard_normal_force_n", 0.0, 250.0, self._value_from_term("_hard_normal_force_n", GRIPPER_HARD_NORMAL_FORCE_N), 1.0, 1)
                self._slider("Lift Force", "lift_force_threshold_n", 0.0, 80.0, float(self.force_limit_state.get("lift_force_threshold_n", args_cli.lift_contact_force_threshold)), 0.5, 1)
                ui.Line(height=2)
                self._slider("Free Force", "free_close_force_n", 0.0, 5.0, self._value_from_term("_free_close_force_n", GRIPPER_FREE_CLOSE_FORCE_N), 0.1, 2)
                self._slider("Free Step", "free_close_step_deg", 0.0, 20.0, math.degrees(self._value_from_term("_free_close_step_rad", math.radians(GRIPPER_FREE_CLOSE_STEP_DEG))), 0.25, 2)
                self._slider("Contact Step", "contact_close_step_deg", 0.0, 5.0, math.degrees(self._value_from_term("_contact_close_step_rad", math.radians(GRIPPER_CONTACT_CLOSE_STEP_DEG))), 0.05, 2)
                ui.Line(height=2)
                self._slider("Servo Target", "force_control_target_n", 0.0, 80.0, self._value_from_term("_force_control_target_n", GRIPPER_FORCE_CONTROL_TARGET_N), 0.5, 1)
                self._slider("Servo Kp", "force_control_kp_deg_per_n", 0.0, 5.00, self._value_from_term("_force_control_kp_deg_per_n", GRIPPER_FORCE_CONTROL_KP_DEG_PER_N), 0.005, 3)
                self._slider("Servo Ki", "force_control_ki_deg_per_n_s", 0.0, 0.20, self._value_from_term("_force_control_ki_deg_per_n_s", GRIPPER_FORCE_CONTROL_KI_DEG_PER_N_S), 0.005, 3)
                self._slider("Servo Kd", "force_control_kd_deg_s_per_n", 0.0, 0.20, self._value_from_term("_force_control_kd_deg_s_per_n", GRIPPER_FORCE_CONTROL_KD_DEG_S_PER_N), 0.005, 3)
                self._slider("I Limit", "force_control_integral_limit_n_s", 0.0, 200.0, self._value_from_term("_force_control_integral_limit_n_s", GRIPPER_FORCE_CONTROL_INTEGRAL_LIMIT_N_S), 1.0, 1)
                self._slider("Servo Dead", "force_control_deadband_n", 0.0, 10.0, self._value_from_term("_force_control_deadband_n", GRIPPER_FORCE_CONTROL_DEADBAND_N), 0.25, 2)
                self._slider("Servo Step", "force_control_max_step_deg", 0.0, 20.0, math.degrees(self._value_from_term("_force_control_max_step_rad", math.radians(GRIPPER_FORCE_CONTROL_MAX_STEP_DEG))), 0.1, 2)
                self._slider("Hold Err", "force_control_hold_error_deg", 0.0, 3.0, math.degrees(self._value_from_term("_force_control_hold_error_rad", math.radians(GRIPPER_FORCE_CONTROL_HOLD_ERROR_DEG))), 0.05, 2)
                self._slider("Vel Limit", "velocity_limit", 0.1, 10.0, self._value_from_term("_velocity_limit", GRIPPER_CLOSE_VELOCITY_LIMIT), 0.1, 2)
                self._slider("Sync Lead", "gripper_sync_max_lead_deg", 0.0, 10.0, math.degrees(self._value_from_term("_gripper_sync_max_lead_rad", math.radians(GRIPPER_SYNC_MAX_LEAD_DEG))), 0.25, 2)
                with ui.HStack(height=28, spacing=8):
                    servo_btn = ui.Button("Toggle Servo", height=24)
                    servo_btn.set_clicked_fn(self._toggle_force_servo)
                    sync_btn = ui.Button("Toggle Sync", height=24)
                    sync_btn.set_clicked_fn(self._toggle_sync)
                    rearm_btn = ui.Button("Re-arm Hold", height=24)
                    rearm_btn.set_clicked_fn(self._rearm_hold)
                ui.Line(height=2)
                self.status_label = ui.Label("", height=24)

    def _slider(self, label: str, key: str, low: float, high: float, value: float, step: float, precision: int) -> None:
        self.precisions[key] = precision
        with ui.HStack(height=28, spacing=8):
            ui.Label(label, width=104)
            slider = ui.FloatSlider(min=low, max=high, step=step, height=18)
            value_label = ui.Label(f"{float(value):.{precision}f}", width=76)
        slider.model.set_value(float(value))
        slider.model.add_value_changed_fn(
            lambda model, item=key: self._on_slider(item, float(model.get_value_as_float()))
        )
        self.sliders[key] = slider
        self.value_labels[key] = value_label

    def _set_status(self, message: str) -> None:
        if self.status_label is not None:
            self.status_label.text = str(message)

    def _set_value_label(self, key: str, value: float) -> None:
        label = self.value_labels.get(key)
        if label is not None:
            precision = self.precisions.get(key, 1)
            label.text = f"{float(value):.{precision}f}"

    def _set_slider_silent(self, key: str, value: float) -> None:
        slider = self.sliders.get(key)
        self._set_value_label(key, value)
        if slider is None:
            return
        self._syncing_ui = True
        try:
            slider.model.set_value(float(value))
        finally:
            self._syncing_ui = False

    def _on_slider(self, key: str, value: float) -> None:
        if self._syncing_ui:
            return
        value = max(0.0, float(value))
        term = self._find_action_term()
        try:
            if key == "lift_force_threshold_n":
                self.force_limit_state["lift_force_threshold_n"] = value
                self._set_value_label(key, value)
                self._set_status(f"Lift force threshold = {value:.1f} N")
                return

            if term is None or not hasattr(term, "_set_gripper_effort_limits"):
                self._set_status("Action term unavailable; force limit not applied.")
                return

            if key == "approach_effort_limit":
                term._approach_effort_limit = value
                self._set_value_label(key, value)
                term._set_gripper_effort_limits()
                self._set_status(f"Close effort limit = {value:.1f}")
            elif key == "hold_effort_limit":
                term._hold_effort_limit = value
                self._set_value_label(key, value)
                term._set_gripper_effort_limits()
                self._set_status(f"Hold effort limit = {value:.1f}")
            elif key == "hold_normal_force_n":
                term._hold_normal_force_n = value
                if getattr(term, "_hard_normal_force_n", 0.0) < value:
                    term._hard_normal_force_n = value
                    self._set_slider_silent("hard_normal_force_n", value)
                self._reset_hard_warn(term)
                self._set_value_label(key, value)
                self._set_status(f"Hold force threshold = {value:.1f} N")
            elif key == "hard_normal_force_n":
                hold = float(getattr(term, "_hold_normal_force_n", GRIPPER_HOLD_NORMAL_FORCE_N))
                value = max(value, hold)
                term._hard_normal_force_n = value
                self._reset_hard_warn(term)
                self._set_slider_silent(key, value)
                self._set_status(f"Hard force warning = {value:.1f} N")
            elif key == "free_close_force_n":
                term._free_close_force_n = value
                self._set_value_label(key, value)
                self._set_status(f"Free close until pad force > {value:.2f} N")
            elif key == "free_close_step_deg":
                term._free_close_step_rad = math.radians(value)
                self._set_value_label(key, value)
                self._set_status(f"Free close step = {value:.2f} deg/step")
            elif key == "contact_close_step_deg":
                term._contact_close_step_rad = math.radians(value)
                self._set_value_label(key, value)
                self._set_status(f"Contact close step = {value:.2f} deg/step")
            elif key == "force_control_target_n":
                term._force_control_target_n = value
                self._set_value_label(key, value)
                self._set_status(f"Force servo target = {value:.1f} N avg-pad")
            elif key == "force_control_kp_deg_per_n":
                term._force_control_kp_deg_per_n = value
                self._set_value_label(key, value)
                self._set_status(f"Force servo Kp = {value:.3f} deg/N")
            elif key == "force_control_ki_deg_per_n_s":
                term._force_control_ki_deg_per_n_s = value
                self._set_value_label(key, value)
                self._set_status(f"Force servo Ki = {value:.3f} deg/(N*s)")
            elif key == "force_control_kd_deg_s_per_n":
                term._force_control_kd_deg_s_per_n = value
                self._set_value_label(key, value)
                self._set_status(f"Force servo Kd = {value:.3f} deg*s/N")
            elif key == "force_control_integral_limit_n_s":
                term._force_control_integral_limit_n_s = value
                integral = getattr(term, "_gripper_force_servo_integral_n_s", None)
                if torch.is_tensor(integral) and value > 0.0:
                    integral.clamp_(min=-value, max=value)
                self._set_value_label(key, value)
                self._set_status(f"Force servo I limit = {value:.1f} N*s")
            elif key == "force_control_deadband_n":
                term._force_control_deadband_n = value
                self._set_value_label(key, value)
                self._set_status(f"Force servo deadband = {value:.2f} N")
            elif key == "force_control_max_step_deg":
                term._force_control_max_step_rad = math.radians(value)
                self._set_value_label(key, value)
                self._set_status(f"Force servo max step = {value:.2f} deg/step")
            elif key == "force_control_hold_error_deg":
                term._force_control_hold_error_rad = math.radians(value)
                self._set_value_label(key, value)
                self._set_status(f"Force servo hold error = {value:.2f} deg")
            elif key == "velocity_limit":
                term._velocity_limit = value
                if hasattr(term, "_set_gripper_velocity_limits"):
                    term._set_gripper_velocity_limits()
                self._set_value_label(key, value)
                self._set_status(f"Gripper velocity limit = {value:.2f} rad/s")
            elif key == "gripper_sync_max_lead_deg":
                term._gripper_sync_max_lead_rad = math.radians(value)
                self._set_value_label(key, value)
                self._set_status(f"Gripper sync lead = {value:.2f} deg")
            print(
                f"[ContactForceLimitPanel] {key}={value:.3f}",
                flush=True,
            )
        except Exception as exc:
            self._set_status(f"Update failed: {type(exc).__name__}: {exc}")
            print(f"[ContactForceLimitPanel] update failed: {type(exc).__name__}: {exc}", flush=True)

    def _reset_hard_warn(self, term) -> None:
        mask = getattr(term, "_gripper_hard_force_warn_mask", None)
        if torch.is_tensor(mask):
            mask[:] = False

    def _toggle_force_servo(self) -> None:
        term = self._find_action_term()
        if term is None:
            self._set_status("Action term unavailable; cannot toggle servo.")
            return
        enabled = not bool(getattr(term, "_force_control_enabled", GRIPPER_FORCE_CONTROL_ENABLED))
        term._force_control_enabled = enabled
        servo_init = getattr(term, "_gripper_force_servo_initialized", None)
        if torch.is_tensor(servo_init):
            servo_init[:] = False
        self._set_status(f"Force servo {'enabled' if enabled else 'disabled'}")
        print(f"[ContactForceLimitPanel] force servo {'enabled' if enabled else 'disabled'}", flush=True)

    def _toggle_sync(self) -> None:
        term = self._find_action_term()
        if term is None:
            self._set_status("Action term unavailable; cannot toggle sync.")
            return
        enabled = not bool(getattr(term, "_gripper_sync_enabled", GRIPPER_SYNC_ENABLED))
        term._gripper_sync_enabled = enabled
        self._set_status(f"Gripper sync {'enabled' if enabled else 'disabled'}")
        print(f"[ContactForceLimitPanel] gripper sync {'enabled' if enabled else 'disabled'}", flush=True)

    def _rearm_hold(self) -> None:
        term = self._find_action_term()
        if term is None:
            self._set_status("Action term unavailable; cannot re-arm.")
            return
        try:
            hold_mask = getattr(term, "_gripper_force_hold_mask", None)
            warn_mask = getattr(term, "_gripper_hard_force_warn_mask", None)
            if torch.is_tensor(hold_mask):
                hold_mask[:] = False
            if torch.is_tensor(warn_mask):
                warn_mask[:] = False
            if hasattr(term, "_reset_force_servo_state"):
                term._reset_force_servo_state()
            if hasattr(term, "_set_gripper_effort_limits"):
                term._set_gripper_effort_limits()
            self._set_status("Hold latch cleared; close effort re-applied.")
            print("[ContactForceLimitPanel] re-armed hold latch", flush=True)
        except Exception as exc:
            self._set_status(f"Re-arm failed: {type(exc).__name__}: {exc}")
            print(f"[ContactForceLimitPanel] re-arm failed: {type(exc).__name__}: {exc}", flush=True)

    def _start_pick(self) -> None:
        self.manual_control_state["start_pick_requested"] = True
        self._set_status("Start Pick requested: policy output starts on next step.")
        print("[ContactForceLimitPanel] Start Pick requested: policy output", flush=True)

    def _reset_env(self) -> None:
        self.manual_control_state["reset_env_requested"] = True
        self._set_status("Reset Env requested; applying in play loop.")
        print("[ContactForceLimitPanel] Reset Env requested", flush=True)


def _force_pad_proxy_display_colors(base_env) -> None:
    if not USE_EXTERNAL_PAD_PROXIES or not args_cli.dynamic_object_eval or bool(args_cli.hide_contact_boxes):
        return
    try:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        colors = {
            "LeftPadProxy": (Gf.Vec3f(0.0, 0.85, 1.0), "cyan"),
            "RightPadProxy": (Gf.Vec3f(1.0, 0.0, 0.85), "magenta"),
        }

        def material_for(proxy_name: str, color: Gf.Vec3f, env_index: int):
            mat_path = Sdf.Path(f"/World/Looks/{proxy_name}_debug_mat_env_{env_index}")
            parent = mat_path.GetParentPath()
            if not stage.GetPrimAtPath(parent).IsValid():
                UsdGeom.Scope.Define(stage, parent)
            material = UsdShade.Material.Define(stage, mat_path)
            shader = UsdShade.Shader.Define(stage, mat_path.AppendChild("PreviewSurface"))
            shader.CreateIdAttr("UsdPreviewSurface")
            shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(color)
            shader.CreateInput("opacity", Sdf.ValueTypeNames.Float).Set(0.55)
            shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.35)
            material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
            return material

        num_envs = int(getattr(base_env.scene.cfg, "num_envs", 1))
        for env_index in range(num_envs):
            env_path = Sdf.Path(f"/World/envs/env_{env_index}")
            for proxy_name, (color, _label) in colors.items():
                root = stage.GetPrimAtPath(env_path.AppendChild(proxy_name))
                if not root.IsValid():
                    continue
                material = material_for(proxy_name, color, env_index)
                for prim in Usd.PrimRange(root):
                    if prim.IsA(UsdGeom.Gprim):
                        gprim = UsdGeom.Gprim(prim)
                        gprim.CreateDisplayColorAttr().Set([color])
                        gprim.CreateDisplayOpacityAttr().Set([0.55])
                        UsdShade.MaterialBindingAPI.Apply(prim).Bind(material)
        print("[PadContactProxy] colors/materials: LeftPadProxy=cyan, RightPadProxy=magenta", flush=True)
    except Exception as exc:
        print(f"[PadContactProxy] color override skipped: {type(exc).__name__}: {exc}", flush=True)


def _filter_pad_proxy_robot_collisions(base_env) -> None:
    if not USE_EXTERNAL_PAD_PROXIES or not args_cli.dynamic_object_eval:
        return
    if getattr(base_env, "_crt_proxy_robot_collision_filtered", False):
        return
    try:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        # This script is a one-env dynamic check path. Filter proxy-vs-robot so
        # external kinematic pad proxies do not collide with the gripper itself.
        for env_index in range(int(getattr(base_env.scene.cfg, "num_envs", 1))):
            env_path = Sdf.Path(f"/World/envs/env_{env_index}")
            robot_path = env_path.AppendChild("Robot")
            for proxy_name in ("LeftPadProxy", "RightPadProxy"):
                proxy_path = env_path.AppendChild(proxy_name)
                proxy_prim = stage.GetPrimAtPath(proxy_path)
                robot_prim = stage.GetPrimAtPath(robot_path)
                if not proxy_prim.IsValid() or not robot_prim.IsValid():
                    continue
                api = UsdPhysics.FilteredPairsAPI.Apply(proxy_prim)
                rel = api.CreateFilteredPairsRel()
                rel.AddTarget(robot_path)
        base_env._crt_proxy_robot_collision_filtered = True
        print("[PadContactProxy] filtered proxy-vs-robot collisions; proxies still collide with Object", flush=True)
    except Exception as exc:
        print(f"[PadContactProxy] collision filter failed: {type(exc).__name__}: {exc}", flush=True)


def _dc_handle_is_valid(handle) -> bool:
    if handle is None:
        return False
    invalid = getattr(_dynamic_control, "INVALID_HANDLE", None) if _dynamic_control is not None else None
    return invalid is None or handle != invalid


def _dc_transform_from_wxyz(pos: np.ndarray, quat_wxyz: np.ndarray):
    q = np.asarray(quat_wxyz, dtype=float).reshape(4)
    q_norm = float(np.linalg.norm(q))
    if q_norm <= 1.0e-12:
        q = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
    else:
        q = q / q_norm
    return _dynamic_control.Transform(
        (float(pos[0]), float(pos[1]), float(pos[2])),
        (float(q[1]), float(q[2]), float(q[3]), float(q[0])),
    )


def _configure_pad_proxy_physx_prims(base_env) -> None:
    if not args_cli.dynamic_object_eval:
        return
    try:
        stage = omni.usd.get_context().get_stage()
        if stage is None:
            return
        num_envs = int(getattr(base_env.scene.cfg, "num_envs", 1))
        colors = {
            "left": Gf.Vec3f(0.0, 0.85, 1.0),
            "right": Gf.Vec3f(1.0, 0.0, 0.85),
        }
        for env_index in range(num_envs):
            env_path = Sdf.Path(f"/World/envs/env_{env_index}")
            for side, prim_name in PAD_PROXY_PRIMS.items():
                root = stage.GetPrimAtPath(env_path.AppendChild(prim_name))
                if not root.IsValid():
                    continue
                for prim in Usd.PrimRange(root):
                    if prim.IsA(UsdGeom.Gprim):
                        gprim = UsdGeom.Gprim(prim)
                        if args_cli.hide_contact_boxes:
                            UsdGeom.Imageable(prim).MakeInvisible()
                        else:
                            gprim.CreateDisplayColorAttr().Set([colors[side]])
                            gprim.CreateDisplayOpacityAttr().Set([0.55])
        print(
            "[PadContactProxy] deploy-style kinematic proxies configured | "
            f"size=({PAD_PROXY_SIZE[0]:.3f}, {PAD_PROXY_SIZE[1]:.3f}, {PAD_PROXY_SIZE[2]:.3f}) m | "
            "pose route=IsaacLab tensor write",
            flush=True,
        )
    except Exception as exc:
        print(f"[PadContactProxy] visual configure skipped: {type(exc).__name__}: {exc}", flush=True)


class KinematicPadContactProxyFollower:
    """External kinematic pad proxies whose collision bodies follow live pad poses."""

    def __init__(self, base_env, robot, pad_body_ids: dict[str, int] | None):
        self.enabled = False
        self.base_env = base_env
        self.robot = robot
        self.pad_body_ids = pad_body_ids or {}
        self.prev_pos: dict[tuple[int, str], np.ndarray] = {}
        self.warned = False
        self.num_envs = int(getattr(base_env.scene.cfg, "num_envs", 1))
        if not USE_EXTERNAL_PAD_PROXIES or not args_cli.dynamic_object_eval or not self.pad_body_ids:
            return
        _configure_pad_proxy_physx_prims(base_env)
        self.enabled = True
        print(
            "[PadContactProxy] kinematic follower enabled: "
            "the visible proxy and collision body are the same RigidObject",
            flush=True,
        )

    def reset(self) -> None:
        self.prev_pos.clear()

    def _write_proxy_tensor(
        self,
        side: str,
        env_index: int,
        pos: np.ndarray,
        quat_wxyz: np.ndarray,
        linear_vel: np.ndarray,
    ) -> None:
        try:
            proxy = self.base_env.scene[PAD_PROXY_ASSETS[side]]
            device = proxy.data.root_pos_w.device
            dtype = proxy.data.root_pos_w.dtype
            root_pose = torch.cat((proxy.data.root_pos_w.clone(), proxy.data.root_quat_w.clone()), dim=-1)
            root_pose[env_index, 0:3] = torch.as_tensor(pos, device=device, dtype=dtype)
            root_pose[env_index, 3:7] = torch.as_tensor(quat_wxyz, device=device, dtype=dtype)
            proxy.write_root_pose_to_sim(root_pose)
            if hasattr(proxy, "write_root_velocity_to_sim") and hasattr(proxy.data, "root_vel_w"):
                root_vel = proxy.data.root_vel_w.clone()
                root_vel[env_index, 0:3] = torch.as_tensor(linear_vel, device=device, dtype=dtype)
                root_vel[env_index, 3:6] = 0.0
                proxy.write_root_velocity_to_sim(root_vel)
        except Exception as exc:
            if not self.warned:
                print(f"[PadContactProxy] tensor follow failed once: {type(exc).__name__}: {exc}", flush=True)
                self.warned = True

    def update(self, dt: float, snap: bool = False) -> None:
        if not self.enabled:
            return
        sim_dt = max(float(dt), 1.0e-6)
        body_pos_w = self.robot.data.body_pos_w
        body_quat_w = self.robot.data.body_quat_w
        active_envs = min(self.num_envs, int(body_pos_w.shape[0]))
        for env_index in range(active_envs):
            for side, body_id in self.pad_body_ids.items():
                key = (env_index, side)
                pad_pos = body_pos_w[env_index, body_id].detach().cpu().numpy().astype(float)
                pad_quat = body_quat_w[env_index, body_id].detach().cpu().numpy().astype(float)
                rot = _quat_wxyz_to_rot_matrix_np(pad_quat)
                local = np.asarray(PAD_PROXY_LOCAL_OFFSETS[side], dtype=float)
                proxy_pos = pad_pos + rot @ local
                prev = self.prev_pos.get(key)
                linear_vel = np.zeros(3, dtype=float) if prev is None or snap else (proxy_pos - prev) / sim_dt
                self._write_proxy_tensor(side, env_index, proxy_pos, pad_quat, linear_vel)
                self.prev_pos[key] = proxy_pos

def _run_play(env, runner):
    print("[BOOT] _run_play entered", flush=True)
    ckpt = _resolve_checkpoint(args_cli.checkpoint)
    print(f"[BOOT] checkpoint resolved: {ckpt}", flush=True)
    if ckpt is not None:
        print(f"[Checkpoint] loading: {ckpt}")
        runner.load(ckpt)
    else:
        print("[Checkpoint] 未指定 checkpoint，使用随机策略。")

    policy = runner.get_inference_policy(device=env.device)

    base_env = env.unwrapped
    obj = base_env.scene["object"]
    ee_frame = base_env.scene["ee_frame"]
    robot = base_env.scene["robot"]
    object_frame_viz = ObjectFrameAxesViz(args_cli.object_frame_axis_length)

    joint_ids, joint_names = robot.find_joints(["joint[1-7]"])
    joint_ids = list(joint_ids)
    if len(joint_ids) != 7:
        raise RuntimeError(f"回放日志期望找到 7 个机械臂关节，但实际找到 {len(joint_ids)} 个: {joint_names}")
    print(f"[Play] logging arm joints: {joint_names}")
    joint_name_list = [str(name) for name in joint_names]
    joint2_col = joint_name_list.index("joint2") if "joint2" in joint_name_list else 1
    pad_body_ids = None
    if args_cli.dynamic_object_eval:
        try:
            body_ids, body_names = robot.find_bodies(["left_pad", "right_pad"], preserve_order=True)
            pad_body_ids = {"left": int(body_ids[0]), "right": int(body_ids[1])}
            print(f"[PadContactProxy] USD pad collision boxes on bodies={body_names}", flush=True)
        except Exception as exc:
            print(f"[PadContactProxy] disabled: {type(exc).__name__}: {exc}", flush=True)
    gripper_chain_body_ids, gripper_chain_body_names, gripper_chain_anchor_body_id = _resolve_gripper_chain_body_ids(robot)
    if gripper_chain_body_ids:
        print(
            f"[GripperChain] monitoring {len(gripper_chain_body_ids)} bodies from link7 | "
            f"max_span<={args_cli.gripper_chain_max_span_m:.3f} m | "
            f"pad_dist<={args_cli.gripper_pad_distance_max_m:.3f} m | "
            f"bodies={gripper_chain_body_names}",
            flush=True,
        )
    else:
        print("[GripperChain] warning: no gripper bodies resolved; chain-break check disabled", flush=True)
    gripper_chain_warned = False
    contact_force_window = ContactForcePlotWindow()
    gripper_pd_plot_window = GripperPDPlotWindow(base_env)
    runtime_tuning_panel = RuntimeTuningPanel(base_env, robot)
    contact_proxy_follower = None

    smooth_alpha = float(np.clip(PLAY_ACTION_SMOOTH_ALPHA, 0.0, 1.0))
    q_offset_wxyz = _parse_quat_wxyz(PLAY_ORI_Q_OFFSET_WXYZ, env.device)
    grasp_symmetry_quats_wxyz = torch.stack(
        [_parse_quat_wxyz(text, env.device) for text in PLAY_GRASP_SYMMETRY_QUATS_WXYZ],
        dim=0,
    )

    # stop-on-best 默认关闭；启用时也必须等进入近目标区才允许 stop。
    raw_dist_w = max(0.0, float(PLAY_STOP_DIST_WEIGHT))
    raw_ori_w = max(0.0, float(PLAY_STOP_ORI_WEIGHT))
    if raw_dist_w + raw_ori_w <= 1.0e-8:
        raw_dist_w, raw_ori_w = 1.0, 0.0
    dist_w = raw_dist_w / (raw_dist_w + raw_ori_w)
    ori_w = raw_ori_w / (raw_dist_w + raw_ori_w)
    min_score_patience = max(1, int(PLAY_MIN_SCORE_PATIENCE))
    min_score_eps = max(0.0, float(PLAY_MIN_SCORE_EPS))

    env_step_dt = float(getattr(base_env, "step_dt", 0.02))

    obs, _ = env.reset()
    _force_pad_proxy_display_colors(base_env)
    _filter_pad_proxy_robot_collisions(base_env)
    if USE_EXTERNAL_PAD_PROXIES:
        contact_proxy_follower = KinematicPadContactProxyFollower(base_env, robot, pad_body_ids)
        contact_proxy_follower.update(env_step_dt, snap=True)
    if args_cli.dynamic_object_eval:
        print(f"[ContactBoxViz] {'hidden' if args_cli.hide_contact_boxes else 'visible'} | USD pad collision boxes=cyan/magenta | workpiece Y side bands=yellow", flush=True)
        print(f"[PadContactForce] sensors={list(PAD_CONTACT_SENSOR_BODIES.keys())}", flush=True)
    csv_log = open_pick_pose_csv(
        root=args_cli.test_log_dir,
        log_name=LOG_NAME,
        checkpoint_path=ckpt,
        file_prefix=LOG_NAME,
        obs_dim=PLAY_MDP_OBS_DIM,
        action_dim=PLAY_ACTION_DIM,
        fallback_prefix="random_policy",
        log_dir_label="PlayLogDir",
        log_file_label="PlayLog",
    )

    ep = 0
    step = 0
    g_step = 0
    last_smoothed_actions = torch.zeros(1, PLAY_ACTION_DIM, device=env.device, dtype=torch.float32)

    # 只有手动启用 stop-on-best 时才使用这些变量。
    best_score = float("inf")
    best_q_arm = robot.data.joint_pos[0, joint_ids].detach().clone()
    policy_hold_q_arm = best_q_arm.detach().clone()
    no_improve_steps = 0
    holding_best = False

    policy_gate = CRTGripperEvalGate(auto_start_policy=bool(args_cli.auto_start_policy))
    policy_lift_after_auto_close = bool(args_cli.policy_lift_after_auto_close)
    auto_hold_after_close = bool(
        (args_cli.hold_after_auto_close or args_cli.dynamic_object_eval or args_cli.lift_after_auto_close)
        and not policy_lift_after_auto_close
    )
    lift_after_auto_close = bool(args_cli.lift_after_auto_close and not policy_lift_after_auto_close)
    lift_mode = str(args_cli.lift_mode)
    lift_height_m = max(0.0, float(args_cli.lift_height_mm) / 1000.0)
    lift_cartesian_step_m = max(0.0005, float(args_cli.lift_cartesian_step_mm) / 1000.0)
    lift_force_threshold_n = max(0.0, float(args_cli.lift_contact_force_threshold))
    force_limit_state = {"lift_force_threshold_n": lift_force_threshold_n}
    contact_force_limit_panel = ContactForceLimitPanel(base_env, force_limit_state, policy_gate.requests)
    _set_gripper_auto_close_enabled(base_env, enabled=(policy_gate.policy_started and not USE_POLICY_GRIPPER))
    lift_stable_steps = max(0, int(round(max(0.0, float(args_cli.lift_after_auto_close_delay)) / max(env_step_dt, 1.0e-6))))
    lift_joint2_total_delta_rad = math.radians(float(args_cli.lift_joint2_delta_deg))
    lift_joint2_step_rad_abs = abs(float(_ARM_CLIP_RAD))
    if lift_joint2_step_rad_abs <= 1.0e-9:
        lift_joint2_step_rad_abs = abs(lift_joint2_total_delta_rad)
    lift_ik_solver = None
    if lift_after_auto_close and lift_mode == "cartesian_z":
        lift_ik_solver = PinocchioTcpLiftIK(
            urdf_path=args_cli.lift_ik_urdf,
            tcp_z_offset_m=float(args_cli.lift_ik_tcp_z_offset_m),
            joint_limits_low=_ARM_JOINT_LIMITS_LOW,
            joint_limits_high=_ARM_JOINT_LIMITS_HIGH,
            max_iters=int(args_cli.lift_ik_max_iters),
            damping=float(args_cli.lift_ik_damping),
            tol_pos_mm=float(args_cli.lift_ik_tol_pos_mm),
            tol_ori_deg=float(args_cli.lift_ik_tol_ori_deg),
        )
    lift_force_ready_g_step = None
    lift_started = False
    lift_progress_rad = 0.0
    lift_target_q_arm = None
    lift_ik_start_pos = None
    lift_ik_target_rot = None
    lift_ik_start_z_m = None
    lift_ik_target_z_m = None
    auto_close_hold_dist_m = _effective_arm_hold_close_dist_m()
    auto_close_hold_ori_deg = _effective_arm_hold_close_ori_deg()
    print("[Eval] start | policy is paused; press Start Pick to begin policy output; Ctrl+C exits")
    if not policy_gate.policy_started:
        print("[Eval] waiting for Start Pick: arm is held still and gripper control is gated")
    if policy_lift_after_auto_close:
        print(
            "[Eval] policy-lift-after-auto-close enabled: dynamic object can be lifted by policy; "
            "eval hold and scripted lift are disabled"
        )
    if auto_hold_after_close:
        print(
            f"[Eval] hold-after-auto-close enabled after policy starts and reaches close window "
            f"({auto_close_hold_dist_m * 100.0:.2f} cm/{auto_close_hold_ori_deg:.2f} deg)"
        )
    if lift_after_auto_close:
        if lift_mode == "cartesian_z":
            print(
                f"[Play] cartesian lift enabled: close after 1cm/1deg, "
                f"then wait L/R pad force >= {lift_force_threshold_n:.2f} N for {lift_stable_steps} steps, "
                f"TCP base-z += {lift_height_m * 1000.0:.1f} mm using stepwise Pinocchio IK "
                f"({lift_cartesian_step_m * 1000.0:.1f} mm/step)"
            )
        else:
            print(
                f"[Play] joint2 lift enabled: close after 1cm/1deg, "
                f"then wait L/R pad force >= {lift_force_threshold_n:.2f} N for {lift_stable_steps} steps, "
                f"joint2 += {args_cli.lift_joint2_delta_deg:.2f} deg, "
                f"step <= {math.degrees(lift_joint2_step_rad_abs):.2f} deg"
            )

    try:
        while simulation_app.is_running():
            if policy_gate.policy_started and args_cli.play_max_steps >= 0 and step >= args_cli.play_max_steps:
                obs, _ = env.reset()
                if USE_EXTERNAL_PAD_PROXIES and contact_proxy_follower is not None:
                    contact_proxy_follower.reset()
                    contact_proxy_follower.update(env_step_dt, snap=True)
                ep += 1
                step = 0
                last_smoothed_actions.zero_()
                best_score = float("inf")
                best_q_arm = robot.data.joint_pos[0, joint_ids].detach().clone()
                policy_hold_q_arm = best_q_arm.detach().clone()
                policy_gate.reset_after_env_reset()
                _set_gripper_auto_close_enabled(base_env, enabled=(policy_gate.policy_started and not USE_POLICY_GRIPPER))
                no_improve_steps = 0
                holding_best = False
                lift_force_ready_g_step = None
                lift_started = False
                lift_progress_rad = 0.0
                lift_target_q_arm = None
                lift_ik_start_pos = None
                lift_ik_target_rot = None
                lift_ik_start_z_m = None
                lift_ik_target_z_m = None
                gripper_chain_warned = False
                print(f"[Eval] reached --play_max_steps={args_cli.play_max_steps}; reset env | ep={ep}, g={g_step}", flush=True)
                continue

            if policy_gate.consume_reset_env():
                obs, _ = env.reset()
                if USE_EXTERNAL_PAD_PROXIES and contact_proxy_follower is not None:
                    contact_proxy_follower.reset()
                    contact_proxy_follower.update(env_step_dt, snap=True)
                ep += 1
                step = 0
                last_smoothed_actions.zero_()
                best_score = float("inf")
                best_q_arm = robot.data.joint_pos[0, joint_ids].detach().clone()
                policy_hold_q_arm = best_q_arm.detach().clone()
                policy_gate.reset_after_env_reset()
                _set_gripper_auto_close_enabled(base_env, enabled=(policy_gate.policy_started and not USE_POLICY_GRIPPER))
                no_improve_steps = 0
                holding_best = False
                lift_force_ready_g_step = None
                lift_started = False
                lift_progress_rad = 0.0
                lift_target_q_arm = None
                lift_ik_start_pos = None
                lift_ik_target_rot = None
                lift_ik_start_z_m = None
                lift_ik_target_z_m = None
                gripper_chain_warned = False
                print(f"[Panel] Reset Env applied | ep={ep}, g={g_step}", flush=True)
                continue

            if policy_gate.consume_start_policy():
                policy_gate.start_policy()
                step = 0
                holding_best = False
                best_q_arm = robot.data.joint_pos[0, joint_ids].detach().clone()
                policy_hold_q_arm = best_q_arm.detach().clone()
                lift_force_ready_g_step = None
                lift_started = False
                lift_progress_rad = 0.0
                lift_target_q_arm = None
                lift_ik_start_pos = None
                lift_ik_target_rot = None
                lift_ik_start_z_m = None
                lift_ik_target_z_m = None
                last_smoothed_actions.zero_()
                _set_gripper_auto_close_enabled(base_env, enabled=(not USE_POLICY_GRIPPER))
                print(f"[Eval] Start Pick applied | policy output enabled | step={step}, g={g_step}", flush=True)

            with torch.no_grad():
                obs_mdp_np = _as_first_env_numpy(obs, PLAY_MDP_OBS_DIM, "policy obs")
                if policy_gate.policy_started:
                    raw_actions = policy(obs)
                else:
                    raw_actions = torch.zeros(1, PLAY_ACTION_DIM, device=env.device, dtype=torch.float32)
                raw_actions_mdp_np = _as_first_env_numpy(raw_actions, PLAY_ACTION_DIM, "raw action")

            if not policy_gate.policy_started:
                if args_cli.arm_set_mode:
                    actions = _arm_action_towards_joint_target(robot, joint_ids, policy_hold_q_arm, raw_actions)
                else:
                    _write_arm_joint_state(base_env, robot, joint_ids, policy_hold_q_arm)
                    actions = torch.zeros_like(raw_actions)
            elif holding_best:
                if args_cli.arm_set_mode:
                    actions = _arm_action_towards_joint_target(robot, joint_ids, best_q_arm, raw_actions)
                else:
                    _write_arm_joint_state(base_env, robot, joint_ids, best_q_arm)
                    actions = torch.zeros_like(raw_actions)
            else:
                actions = smooth_alpha * raw_actions + (1.0 - smooth_alpha) * last_smoothed_actions
                last_smoothed_actions = actions.detach().clone()

            _filter_pad_proxy_robot_collisions(base_env)
            if USE_EXTERNAL_PAD_PROXIES and contact_proxy_follower is not None:
                contact_proxy_follower.update(env_step_dt, snap=False)
            q_before = robot.data.joint_pos[0, joint_ids].detach().clone()
            obs, rewards, dones, extras = env.step(actions)
            if USE_EXTERNAL_PAD_PROXIES and contact_proxy_follower is not None:
                contact_proxy_follower.update(env_step_dt, snap=False)
            q_after = robot.data.joint_pos[0, joint_ids].detach().clone()

            joint_deg = torch.rad2deg(q_after).cpu().numpy()
            actual_delta_deg = torch.rad2deg(q_after - q_before).cpu().numpy()
            cmd_delta_deg = _get_cmd_delta_deg(base_env, actions)
            delta_error_deg = actual_delta_deg - cmd_delta_deg

            ee_pos_w, ee_quat = get_ee_target_pose_w(ee_frame)
            obj_root_pos_w = obj.data.root_pos_w[0].detach().clone()
            obj_quat = obj.data.root_quat_w[0].detach().clone()
            object_frame_viz.update(obj_root_pos_w, obj_quat)
            pad_force_l_n, pad_force_r_n, pad_force_total_n = _pad_normal_forces_n(base_env.scene)
            contact_force_window.update(pad_force_l_n, pad_force_r_n, pad_force_total_n)
            gripper_pd_plot_window.update()
            gripper_debug_values = getattr(gripper_pd_plot_window, "last_values", {}) or {}
            gripper_chain_span_m, gripper_pad_distance_m, gripper_chain_broken = _gripper_chain_metrics(
                robot,
                gripper_chain_body_ids,
                gripper_chain_anchor_body_id,
                pad_body_ids,
                max_span_m=float(args_cli.gripper_chain_max_span_m),
                max_pad_distance_m=float(args_cli.gripper_pad_distance_max_m),
            )
            if gripper_chain_broken and not gripper_chain_warned:
                gripper_chain_warned = True
                print(
                    f"[GripperChainBreak] span={gripper_chain_span_m:.4f} m "
                    f"pad_dist={gripper_pad_distance_m:.4f} m | "
                    f"limits=({args_cli.gripper_chain_max_span_m:.4f}, {args_cli.gripper_pad_distance_max_m:.4f}) m | "
                    f"step={step}, g={g_step}",
                    flush=True,
                )
            lift_force_threshold_n = max(
                0.0,
                float(force_limit_state.get("lift_force_threshold_n", lift_force_threshold_n)),
            )
            obj_pos_w = obj_root_pos_w.clone()
            obj_pos_w[2] += float(OBJECT_POSE_Z_OFFSET_M)

            # CSV 里的 tcp_* / object_* 统一记录 robot base_link 坐标系下的位姿。
            # tcp_* 使用 ee_frame target/TCP，包含配置里的 0.177 m offset。
            ee_pos, ee_quat_np = pose_in_robot_base(robot, ee_pos_w, ee_quat)
            obj_pos, obj_quat_np = pose_in_robot_base(robot, obj_pos_w, obj_quat)
            dist_cm = float(np.linalg.norm(obj_pos - ee_pos) * 100.0)
            ori_err_deg = _orientation_error_deg(
                ee_quat,
                obj_quat,
                q_offset_wxyz,
                grasp_symmetry_quats_wxyz,
            )

            done_bool = bool(dones[0].item()) if torch.is_tensor(dones) else bool(dones[0])
            time_outs = _extract_timeout(extras, dones)
            timeout_bool = bool(time_outs[0].item())

            terminal_dist_cm, terminal_ori_err_deg, terminal_success = _read_terminal_cache(base_env, env_id=0)

            # done=True 的那一帧，Isaac Lab / wrapper 可能已经自动 reset。
            # 因此 done 行优先使用环境 termination 阶段缓存的 reset 前误差。
            if done_bool and terminal_dist_cm is not None and terminal_ori_err_deg is not None:
                log_dist_cm = terminal_dist_cm
                log_ori_err_deg = terminal_ori_err_deg
                terminal_source = "cached_terminal"
            else:
                log_dist_cm = dist_cm
                log_ori_err_deg = ori_err_deg
                terminal_source = "live_scene"

            auto_close_ready = (
                policy_gate.policy_started
                and dist_cm <= auto_close_hold_dist_m * 100.0
                and ori_err_deg <= auto_close_hold_ori_deg
            )
            if auto_hold_after_close and auto_close_ready and (not holding_best):
                holding_best = True
                best_q_arm = q_after.detach().clone()
                lift_force_ready_g_step = None
                lift_started = False
                lift_progress_rad = 0.0
                lift_target_q_arm = None
                lift_ik_start_pos = None
                lift_ik_target_rot = None
                lift_ik_start_z_m = None
                lift_ik_target_z_m = None
                if not args_cli.arm_set_mode:
                    _write_arm_joint_state(base_env, robot, joint_ids, best_q_arm)
                print(
                    f"[Hold@auto-close] hold arm pose | "
                    f"step={step}, g={g_step}, dist={dist_cm:.4f} cm, ori={ori_err_deg:.4f} deg"
                )

            lift_contact_ready = (
                lift_after_auto_close
                and holding_best
                and pad_force_l_n >= lift_force_threshold_n
                and pad_force_r_n >= lift_force_threshold_n
            )
            if lift_after_auto_close and holding_best and (not lift_started):
                if lift_contact_ready:
                    if lift_force_ready_g_step is None:
                        lift_force_ready_g_step = g_step
                        print(
                            f"[Lift@force-ready] L={pad_force_l_n:.2f} N R={pad_force_r_n:.2f} N "
                            f">= {lift_force_threshold_n:.2f} N | stable wait={lift_stable_steps} steps"
                        )
                else:
                    if lift_force_ready_g_step is not None:
                        print(
                            f"[Lift@force-lost] reset stable timer | "
                            f"L={pad_force_l_n:.2f} N R={pad_force_r_n:.2f} N "
                            f"threshold={lift_force_threshold_n:.2f} N"
                        )
                    lift_force_ready_g_step = None

            lift_has_motion = (
                lift_height_m > 1.0e-6
                if lift_mode == "cartesian_z"
                else abs(lift_joint2_total_delta_rad) > 1.0e-9
            )
            lift_ready = (
                lift_after_auto_close
                and holding_best
                and lift_force_ready_g_step is not None
                and (g_step - lift_force_ready_g_step) >= lift_stable_steps
                and lift_has_motion
            )
            if lift_ready and (not lift_started):
                if lift_mode == "cartesian_z":
                    if lift_ik_solver is None:
                        raise RuntimeError("lift_mode=cartesian_z requires PinocchioTcpLiftIK")
                    seed_q = best_q_arm.detach().cpu().numpy().astype(float)
                    start_pos, start_rot = lift_ik_solver._forward_pose(seed_q)
                    lift_ik_start_pos = start_pos.copy()
                    lift_ik_target_rot = start_rot.copy()
                    lift_ik_start_z_m = float(start_pos[2])
                    lift_ik_target_z_m = lift_ik_start_z_m + lift_height_m
                    lift_target_q_arm = None
                    lift_started = True
                    print(
                        f"[Lift@auto-close] stepwise cartesian_z start | total={lift_height_m * 1000.0:.1f} mm, "
                        f"tcp_z={lift_ik_start_z_m * 1000.0:.1f}->{lift_ik_target_z_m * 1000.0:.1f} mm | "
                        f"cart_step={lift_cartesian_step_m * 1000.0:.1f} mm | "
                        f"hold current TCP orientation | step={step}, g={g_step}",
                        flush=True,
                    )
                else:
                    lift_started = True
                    print(
                        f"[Lift@auto-close] joint2 ramp start | total={args_cli.lift_joint2_delta_deg:.2f} deg, "
                        f"step<={math.degrees(lift_joint2_step_rad_abs):.2f} deg | step={step}, g={g_step}",
                        flush=True,
                    )

            if lift_mode == "cartesian_z":
                lift_remaining_metric = 0.0
                lift_progress_text = "0.0 mm"
                if lift_ik_solver is not None and lift_ik_start_z_m is not None and lift_ik_target_z_m is not None:
                    pin_pos, _ = lift_ik_solver._forward_pose(best_q_arm.detach().cpu().numpy().astype(float))
                    progress_m = float(np.clip(float(pin_pos[2]) - lift_ik_start_z_m, 0.0, lift_height_m))
                    lift_remaining_metric = max(0.0, lift_ik_target_z_m - float(pin_pos[2]))
                    lift_progress_text = f"{progress_m * 1000.0:.1f}/{lift_height_m * 1000.0:.1f} mm"
            else:
                lift_remaining_metric = abs(lift_joint2_total_delta_rad - lift_progress_rad)
                lift_progress_text = f"{math.degrees(lift_progress_rad):.2f} deg"

            lift_pause_threshold = 1.0e-6 if lift_mode == "cartesian_z" else 1.0e-9
            if (
                lift_after_auto_close
                and lift_started
                and lift_remaining_metric > lift_pause_threshold
                and not lift_contact_ready
            ):
                print(
                    f"[Lift@force-lost] pause lift | L={pad_force_l_n:.2f} N R={pad_force_r_n:.2f} N "
                    f"threshold={lift_force_threshold_n:.2f} N | progress={lift_progress_text}",
                    flush=True,
                )
                lift_force_ready_g_step = None
                lift_started = False

            if lift_ready and lift_started:
                if lift_mode == "cartesian_z":
                    if (
                        lift_ik_solver is not None
                        and lift_ik_start_pos is not None
                        and lift_ik_target_rot is not None
                        and lift_ik_start_z_m is not None
                        and lift_ik_target_z_m is not None
                    ):
                        seed_q = best_q_arm.detach().cpu().numpy().astype(float)
                        current_pos, _ = lift_ik_solver._forward_pose(seed_q)
                        current_progress_m = float(np.clip(float(current_pos[2]) - lift_ik_start_z_m, 0.0, lift_height_m))
                        if current_progress_m < lift_height_m - 1.0e-6:
                            desired_progress_m = min(lift_height_m, current_progress_m + lift_cartesian_step_m)
                            ik_result = None
                            step_target_q_arm = None
                            for _attempt in range(7):
                                target_pos = np.asarray(lift_ik_start_pos, dtype=float).copy()
                                target_pos[2] = lift_ik_start_z_m + desired_progress_m
                                ik_result = lift_ik_solver.solve_pose(seed_q, target_pos, lift_ik_target_rot)
                                step_target_q_arm = torch.as_tensor(
                                    ik_result["q"],
                                    device=best_q_arm.device,
                                    dtype=best_q_arm.dtype,
                                )
                                max_delta_rad = float(torch.max(torch.abs(step_target_q_arm - best_q_arm)).detach().cpu().item())
                                if max_delta_rad <= float(_ARM_CLIP_RAD) * 1.05 or desired_progress_m <= current_progress_m + 0.0006:
                                    break
                                desired_progress_m = current_progress_m + 0.5 * (desired_progress_m - current_progress_m)

                            if step_target_q_arm is not None:
                                lift_target_q_arm = step_target_q_arm
                                remaining_q = lift_target_q_arm - best_q_arm
                                if float(torch.max(torch.abs(remaining_q)).detach().cpu().item()) > 1.0e-9:
                                    step_q = torch.clamp(remaining_q, -float(_ARM_CLIP_RAD), float(_ARM_CLIP_RAD))
                                    lift_q_arm = best_q_arm.detach().clone() + step_q
                                    lift_q_arm = torch.clamp(
                                        lift_q_arm,
                                        _ARM_JOINT_LIMITS_LOW.to(device=lift_q_arm.device, dtype=lift_q_arm.dtype),
                                        _ARM_JOINT_LIMITS_HIGH.to(device=lift_q_arm.device, dtype=lift_q_arm.dtype),
                                    )
                                    best_q_arm = lift_q_arm
                                    if not args_cli.arm_set_mode:
                                        _write_arm_joint_state(base_env, robot, joint_ids, best_q_arm)
                                    pin_pos, pin_rot = lift_ik_solver._forward_pose(best_q_arm.detach().cpu().numpy().astype(float))
                                    pin_progress_mm = (float(pin_pos[2]) - lift_ik_start_z_m) * 1000.0
                                    lift_ori_err_deg = math.degrees(float(np.linalg.norm(lift_ik_solver.pin.log3(lift_ik_target_rot @ pin_rot.T))))
                                    done_tag = " done" if (lift_ik_target_z_m - float(pin_pos[2])) <= 1.0e-4 else ""
                                    step_deg = torch.rad2deg(step_q).detach().cpu().numpy()
                                    print(
                                        f"[Lift@auto-close]{done_tag} cartesian_z progress={pin_progress_mm:.1f}/{lift_height_m * 1000.0:.1f} mm | "
                                        f"target_step={desired_progress_m * 1000.0:.1f} mm | "
                                        f"ori_err={lift_ori_err_deg:.2f} deg | "
                                        f"ik_pos_err={ik_result['pos_err_m'] * 1000.0:.2f} mm | "
                                        f"step_deg={np.array2string(step_deg, precision=2, suppress_small=True)} | step={step}, g={g_step}",
                                        flush=True,
                                    )
                else:
                    remaining_rad = lift_joint2_total_delta_rad - lift_progress_rad
                    if abs(remaining_rad) > 1.0e-9:
                        step_rad = math.copysign(min(abs(remaining_rad), lift_joint2_step_rad_abs), remaining_rad)
                        lift_q_arm = best_q_arm.detach().clone()
                        joint2_before = float(lift_q_arm[joint2_col].detach().cpu().item())
                        lift_q_arm[joint2_col] += step_rad
                        joint2_low = float(_ARM_JOINT_LIMITS_LOW[joint2_col].detach().cpu().item())
                        joint2_high = float(_ARM_JOINT_LIMITS_HIGH[joint2_col].detach().cpu().item())
                        lift_q_arm[joint2_col] = torch.clamp(lift_q_arm[joint2_col], joint2_low, joint2_high)
                        actual_step_rad = float(lift_q_arm[joint2_col].detach().cpu().item()) - joint2_before
                        best_q_arm = lift_q_arm
                        lift_progress_rad += actual_step_rad
                        if abs(actual_step_rad) <= 1.0e-9:
                            lift_progress_rad = lift_joint2_total_delta_rad
                        if not args_cli.arm_set_mode:
                            _write_arm_joint_state(base_env, robot, joint_ids, best_q_arm)
                        joint2_deg = math.degrees(float(best_q_arm[joint2_col].detach().cpu().item()))
                        progress_deg = math.degrees(lift_progress_rad)
                        done_tag = " done" if abs(lift_joint2_total_delta_rad - lift_progress_rad) <= 1.0e-6 else ""
                        print(
                            f"[Lift@auto-close]{done_tag} joint2 target={joint2_deg:.2f} deg | "
                            f"progress={progress_deg:.2f}/{args_cli.lift_joint2_delta_deg:.2f} deg | "
                            f"step_delta={math.degrees(actual_step_rad):.2f} deg | step={step}, g={g_step}",
                            flush=True,
                        )

            if PLAY_ENABLE_STOP_ON_BEST and (not holding_best):
                near_enough_for_stop = (dist_cm < PLAY_STOP_ENABLE_DIST_CM) and (ori_err_deg < PLAY_STOP_ENABLE_ORI_DEG)
                stop_score = _compute_stop_score(
                    dist_cm=dist_cm,
                    ori_err_deg=ori_err_deg,
                    dist_weight=dist_w,
                    ori_weight=ori_w,
                    dist_ref_cm=PLAY_STOP_DIST_REF_CM,
                    ori_ref_deg=PLAY_STOP_ORI_REF_DEG,
                )

                if near_enough_for_stop:
                    if stop_score < best_score - min_score_eps:
                        best_score = stop_score
                        best_q_arm = q_after.detach().clone()
                        no_improve_steps = 0
                    else:
                        no_improve_steps += 1

                    if no_improve_steps >= min_score_patience:
                        holding_best = True
                        if not args_cli.arm_set_mode:
                            _write_arm_joint_state(base_env, robot, joint_ids, best_q_arm)
                        print(
                            f"[Stop@best-score] hold best pose | "
                            f"step={step}, dist={dist_cm:.3f} cm, ori={ori_err_deg:.3f} deg, score={stop_score:.6f}"
                        )
                else:
                    # 还没进入近目标区，不允许 stop，也不累计 no_improve。
                    best_score = float("inf")
                    no_improve_steps = 0

            csv_log.write_row(
                PickPoseCsvRow(
                    episode=ep,
                    step=step,
                    g_step=g_step,
                    done=done_bool,
                    timeout=timeout_bool,
                    obs=obs_mdp_np,
                    raw_action=raw_actions_mdp_np,
                    dist_cm=log_dist_cm,
                    ori_err_deg=log_ori_err_deg,
                    terminal_dist_cm=terminal_dist_cm,
                    terminal_ori_err_deg=terminal_ori_err_deg,
                    terminal_success=terminal_success,
                    terminal_source=terminal_source,
                    tcp_pos_m=ee_pos,
                    tcp_quat_wxyz=ee_quat_np,
                    object_pos_m=obj_pos,
                    object_quat_wxyz=obj_quat_np,
                    cmd_delta_deg=cmd_delta_deg,
                    actual_delta_deg=actual_delta_deg,
                    delta_error_deg=delta_error_deg,
                    joint_deg=joint_deg,
                    pad_force_l_n=pad_force_l_n,
                    pad_force_r_n=pad_force_r_n,
                    pad_force_total_n=pad_force_total_n,
                    gripper_target_deg=gripper_debug_values.get("target_deg"),
                    gripper_actual_left_deg=gripper_debug_values.get("actual_left_deg"),
                    gripper_actual_right_deg=gripper_debug_values.get("actual_right_deg"),
                    gripper_actual_mean_deg=gripper_debug_values.get("actual_mean_deg"),
                    gripper_tracking_error_deg=gripper_debug_values.get("tracking_error_deg"),
                    gripper_force_pid_u_deg=gripper_debug_values.get("u_deg"),
                    gripper_force_pid_delta_deg=gripper_debug_values.get("delta_deg"),
                    gripper_actual_step_deg=gripper_debug_values.get("actual_step_deg"),
                    gripper_target_left_deg=gripper_debug_values.get("target_left_deg"),
                    gripper_target_right_deg=gripper_debug_values.get("target_right_deg"),
                    gripper_left_contact=gripper_debug_values.get("left_contact"),
                    gripper_right_contact=gripper_debug_values.get("right_contact"),
                    gripper_both_contact=gripper_debug_values.get("both_contact"),
                    gripper_closed_mask=gripper_debug_values.get("closed_mask"),
                    gripper_hold_mask=gripper_debug_values.get("hold_mask"),
                    gripper_force_servo_active=gripper_debug_values.get("force_servo_active"),
                    gripper_actual_vel_mean_deg_s=gripper_debug_values.get("actual_vel_mean_deg_s"),
                    gripper_effort_limit_min=gripper_debug_values.get("effort_limit_min"),
                    gripper_effort_limit_max=gripper_debug_values.get("effort_limit_max"),
                    gripper_chain_span_m=gripper_chain_span_m,
                    gripper_pad_distance_m=gripper_pad_distance_m,
                    gripper_chain_broken=gripper_chain_broken,
                )
            )
            csv_log.flush()
            if gripper_chain_broken and args_cli.dynamic_object_eval:
                print("[Play] abort: gripper chain-break check failed; reject these parameters.", flush=True)
                break

            periodic_print = args_cli.print_every > 0 and (g_step % args_cli.print_every == 0)
            success_print = done_bool and not timeout_bool
            if periodic_print or success_print:
                tag = "SUCCESS " if success_print else ""
                print(
                    f"[{tag}ep {ep:03d} step {step:04d} g {g_step:06d}] "
                    f"dist={log_dist_cm:.3f} cm | ori={log_ori_err_deg:.3f} deg | "
                    f"cmd_mean={float(np.mean(np.abs(cmd_delta_deg))):.3f} deg | "
                    f"act_mean={float(np.mean(np.abs(actual_delta_deg))):.3f} deg | "
                    f"err_max={float(np.max(np.abs(delta_error_deg))):.3f} deg | "
                    f"done={done_bool} | timeout={timeout_bool} | terminal_success={terminal_success} | source={terminal_source}"
                )

            step += 1
            g_step += 1

            if done_bool:
                print(
                    f"[EpisodeDone] ep={ep} steps={step} "
                    f"final_dist={log_dist_cm:.3f} cm | final_ori={log_ori_err_deg:.3f} deg | "
                    f"success={terminal_success} | timeout={timeout_bool} | source={terminal_source}"
                )
                obs, _ = env.reset()
                if USE_EXTERNAL_PAD_PROXIES and contact_proxy_follower is not None:
                    contact_proxy_follower.reset()
                    contact_proxy_follower.update(env_step_dt, snap=True)
                ep += 1
                step = 0
                last_smoothed_actions.zero_()
                best_score = float("inf")
                best_q_arm = robot.data.joint_pos[0, joint_ids].detach().clone()
                policy_hold_q_arm = best_q_arm.detach().clone()
                policy_gate.reset_after_env_reset()
                _set_gripper_auto_close_enabled(base_env, enabled=(policy_gate.policy_started and not USE_POLICY_GRIPPER))
                no_improve_steps = 0
                holding_best = False
                lift_force_ready_g_step = None
                lift_started = False
                lift_progress_rad = 0.0
                lift_target_q_arm = None
                lift_ik_start_pos = None
                lift_ik_target_rot = None
                lift_ik_start_z_m = None
                lift_ik_target_z_m = None
                policy_gate.clear_requests()

    finally:
        csv_log.close()


def _tensorboard_iteration_from_log_args(args, kwargs, runner) -> int:
    locs = args[0] if args and isinstance(args[0], dict) else kwargs.get("locs")
    if isinstance(locs, dict):
        for key in ("it", "iteration", "current_learning_iteration"):
            value = locs.get(key)
            if value is not None:
                try:
                    return int(value)
                except Exception:
                    pass
    for attr in ("current_learning_iteration", "iter", "iteration"):
        value = getattr(runner, attr, None)
        if value is not None:
            try:
                return int(value)
            except Exception:
                pass
    return 0


def _patch_runner_tensorboard_error_logging(runner):
    """Add pose error scalars to the same TensorBoard writer used by RSL-RL."""
    if getattr(runner, "_pick_pose_error_logging_patched", False):
        return

    base_env = getattr(getattr(runner, "env", None), "unwrapped", None)
    if base_env is None:
        print("[TensorBoard] skip pose error scalars: runner.env.unwrapped not found")
        return

    original_log = runner.log

    def log_with_pose_errors(*args, **kwargs):
        result = original_log(*args, **kwargs)

        writer = getattr(runner, "writer", None)
        if writer is None:
            return result

        try:
            with torch.no_grad():
                dist_m, ori_err_deg = _compute_reach_error_metrics(base_env)
                dist_cm = dist_m * 100.0
                step = _tensorboard_iteration_from_log_args(args, kwargs, runner)
                writer.add_scalar("PoseError/dist_cm", float(dist_cm.mean().item()), step)
                writer.add_scalar("PoseError/ori_err_deg", float(ori_err_deg.mean().item()), step)
        except Exception as exc:
            if not getattr(runner, "_pick_pose_error_logging_warned", False):
                print(f"[TensorBoard] failed to log pose errors once: {exc}")
                runner._pick_pose_error_logging_warned = True

        return result

    runner.log = log_with_pose_errors
    runner._pick_pose_error_logging_patched = True

def _run_train(runner, agent_cfg):
    _patch_runner_tensorboard_error_logging(runner)

    if args_cli.resume:
        if args_cli.checkpoint is not None:
            ckpt = _resolve_checkpoint(args_cli.checkpoint)
            if ckpt is not None:
                print(f"[Checkpoint] resume from: {ckpt}")
                runner.load(ckpt)
        elif args_cli.load_run is not None:
            print(f"[Checkpoint] resume from load_run: {args_cli.load_run}")
            runner.load(args_cli.load_run)
        else:
            print("[Resume] --resume specified but no --checkpoint/--load_run given. Start from scratch.")

    runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)


def main():
    print("[BOOT] Resolving CRT robot USD", flush=True)
    robot_usd = _resolve_robot_usd()
    print("[BOOT] Building PickPoseCRTWriteEnvCfg", flush=True)
    env_cfg = _make_env_cfg(robot_usd)

    agent_cfg = PickPosePPORunnerCfg()
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    log_dir = os.path.join(
        "logs",
        LOG_NAME,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    os.makedirs(log_dir, exist_ok=True)

    if os.path.isfile(ENV_CFG_FILE):
        shutil.copy(ENV_CFG_FILE, os.path.join(log_dir, "env_cfg.py"))
    shutil.copy(Path(__file__).resolve(), os.path.join(log_dir, "eval_script.py"))

    print("[BOOT] Creating ManagerBasedRLEnv", flush=True)
    env = ManagerBasedRLEnv(cfg=env_cfg)
    print("[BOOT] Wrapping RSL-RL env", flush=True)
    env = RslRlVecEnvWrapper(env)

    print("[BOOT] Creating OnPolicyRunner", flush=True)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=env.device)
    print("[BOOT] OnPolicyRunner ready", flush=True)

    print("[BOOT] Printing run header", flush=True)
    _print_run_header(log_dir, env_cfg.scene.num_envs, robot_usd)
    print("[BOOT] Entering eval loop", flush=True)

    try:
        _run_play(env, runner)
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nError: {exc}")
        traceback.print_exc()
    finally:
        simulation_app.close()
