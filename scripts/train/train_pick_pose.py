#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""xArm7 pick-pose 训练和旧版 --play 回放入口。"""

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


VERSION_TAG = "pick_pose"
LOG_NAME = "xarm7_pick_pose"
BASE_ENV_CFG_FILE = os.path.join(_PROJECT_DIR, "configs", "xarm7_pick_pose_env_cfg.py")
CRT_ENV_CFG_FILE = os.path.join(_PROJECT_DIR, "configs", "xarm7_pick_pose_crt_write_env_cfg.py")
ENV_CFG_FILE = BASE_ENV_CFG_FILE
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


parser = argparse.ArgumentParser(
    description="xArm7 Pick 位姿真值训练/纯策略回放"
)
parser.add_argument("--num_envs", type=int, default=None, help="训练环境数量；play 默认 1，train 默认 64")
parser.add_argument("--play", action="store_true", help="回放模式")
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
    "--use_crt_gripper",
    action="store_true",
    help="使用 CRT 夹爪机器人模型；默认策略仍为 21 obs / 7 actions，夹爪不进入策略动作空间。",
)
parser.add_argument(
    "--use_policy_gripper",
    action="store_true",
    help="让策略接管 CRT 夹爪；隐含 --use_crt_gripper，并切到 22 obs / 8 actions。",
)
parser.add_argument("--play_max_steps", type=int, default=-1, help="回放最大总步数，-1 表示一直运行")
parser.add_argument("--print_every", type=int, default=10, help="回放每隔多少步打印一次")
parser.add_argument("--test_log_dir", type=str, default=TEST_LOG_DIR, help="回放 CSV 保存目录")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
USE_POLICY_GRIPPER = bool(args_cli.use_policy_gripper)
USE_CRT_GRIPPER = bool(args_cli.use_crt_gripper or USE_POLICY_GRIPPER)

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

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from configs.xarm7_pick_pose_env_cfg import (
    PickPoseEnvCfg,
    PickPoseEnvCfg_PLAY,
    _ARM_ACTION_SCALE,
    _ARM_CLIP_RAD,
    _compute_reach_error_metrics,
    set_success_termination_enabled,
)

if USE_CRT_GRIPPER:
    from configs.xarm7_pick_pose_crt_write_env_cfg import (
        PickPoseCRTWriteEnvCfg,
        PickPoseCRTWriteEnvCfg_PLAY,
    )
else:
    PickPoseCRTWriteEnvCfg = None
    PickPoseCRTWriteEnvCfg_PLAY = None

if USE_CRT_GRIPPER:
    VERSION_TAG = "pick_pose_crt_write_policy_gripper" if USE_POLICY_GRIPPER else "pick_pose_crt_write_arm_only"
    LOG_NAME = "xarm7_pick_pose_crt_write" if USE_POLICY_GRIPPER else "xarm7_pick_pose_crt_write_arm_only"
    ENV_CFG_FILE = CRT_ENV_CFG_FILE
    PLAY_MDP_OBS_DIM = 22 if USE_POLICY_GRIPPER else 21
    PLAY_ACTION_DIM = 8 if USE_POLICY_GRIPPER else 7

from configs.agents.rsl_rl_ppo_cfg import PickPosePPORunnerCfg
from tools.utils.pick_pose_eval_utils import get_ee_target_pose_w, pose_in_robot_base


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


def _make_env_cfg():
    """创建训练或回放环境配置。"""
    if USE_CRT_GRIPPER:
        env_cfg = PickPoseCRTWriteEnvCfg_PLAY() if args_cli.play else PickPoseCRTWriteEnvCfg()
        env_cfg.actions.arm_action.policy_gripper_enabled = USE_POLICY_GRIPPER
    else:
        env_cfg = PickPoseEnvCfg_PLAY() if args_cli.play else PickPoseEnvCfg()

    if args_cli.num_envs is not None:
        env_cfg.scene.num_envs = args_cli.num_envs
    else:
        env_cfg.scene.num_envs = 1 if args_cli.play else 64

    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    set_success_termination_enabled(env_cfg, args_cli.enable_success_termination)

    return env_cfg


def _print_run_header(log_dir: str, num_envs: int):
    print(f"\n========== {VERSION_TAG} ==========")
    print(f"[Version] {VERSION_TAG}")
    print(f"[Action] scale = {math.degrees(_ARM_ACTION_SCALE):.4f} deg")
    print(f"[Action] clip  = ±{math.degrees(_ARM_CLIP_RAD):.4f} deg")
    action_surface = "7 arm + 1 gripper" if USE_POLICY_GRIPPER else "7 arm"
    print(f"[Action] dim   = {PLAY_ACTION_DIM} ({action_surface})")
    print(f"[Gripper] CRT model = {'enabled' if USE_CRT_GRIPPER else 'disabled'} | policy gripper = {'enabled' if USE_POLICY_GRIPPER else 'disabled'}")
    print(f"[Log]    {log_dir}")
    print(f"[Env]    num_envs = {num_envs}")
    print(f"[Config] {ENV_CFG_FILE}")
    success_term_status = (
        "enabled"
        if args_cli.enable_success_termination
        else "disabled (no-success default)"
    )
    print(f"[Reward] success termination: {success_term_status}")
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


def _get_arm_action_term(base_env):
    """从 action_manager 中读取 arm_action，用于获得 processed_actions。"""
    manager = base_env.action_manager

    if hasattr(manager, "get_term"):
        try:
            term = manager.get_term("arm_action")
            if hasattr(term, "processed_actions"):
                return term
        except Exception:
            pass

    for attr in ("_terms", "_action_terms"):
        terms = getattr(manager, attr, None)
        if isinstance(terms, dict) and "arm_action" in terms:
            term = terms["arm_action"]
            if hasattr(term, "processed_actions"):
                return term

    terms = getattr(manager, "_terms", None)
    if isinstance(terms, (list, tuple)):
        for term in terms:
            if hasattr(term, "processed_actions"):
                return term

    return None


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
    """仅在手动启用 stop-on-best 时使用：写回指定 7 关节并清零速度。"""
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


def _run_play(env, runner):
    ckpt = _resolve_checkpoint(args_cli.checkpoint)
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

    joint_ids, joint_names = robot.find_joints(["joint[1-7]"])
    joint_ids = list(joint_ids)
    if len(joint_ids) != 7:
        raise RuntimeError(f"回放日志期望找到 7 个机械臂关节，但实际找到 {len(joint_ids)} 个: {joint_names}")
    print(f"[Play] logging arm joints: {joint_names}")

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

    obs, _ = env.reset()
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
    no_improve_steps = 0
    holding_best = False

    print("[Play] start | pure policy replay by default | press Ctrl+C to exit")

    try:
        while simulation_app.is_running():
            if args_cli.play_max_steps >= 0 and g_step >= args_cli.play_max_steps:
                print(f"[Play] reached --play_max_steps={args_cli.play_max_steps}, exit.")
                break

            with torch.no_grad():
                obs_mdp_np = _as_first_env_numpy(obs, PLAY_MDP_OBS_DIM, "policy obs")
                raw_actions = policy(obs)
                raw_actions_mdp_np = _as_first_env_numpy(raw_actions, PLAY_ACTION_DIM, "raw action")

            if holding_best:
                _write_arm_joint_state(base_env, robot, joint_ids, best_q_arm)
                actions = torch.zeros_like(raw_actions)
            else:
                actions = smooth_alpha * raw_actions + (1.0 - smooth_alpha) * last_smoothed_actions
                last_smoothed_actions = actions.detach().clone()

            q_before = robot.data.joint_pos[0, joint_ids].detach().clone()
            obs, rewards, dones, extras = env.step(actions)
            q_after = robot.data.joint_pos[0, joint_ids].detach().clone()

            joint_deg = torch.rad2deg(q_after).cpu().numpy()
            actual_delta_deg = torch.rad2deg(q_after - q_before).cpu().numpy()
            cmd_delta_deg = _get_cmd_delta_deg(base_env, actions)
            delta_error_deg = actual_delta_deg - cmd_delta_deg

            ee_pos_w, ee_quat = get_ee_target_pose_w(ee_frame)
            obj_pos_w = obj.data.root_pos_w[0].detach()
            obj_quat = obj.data.root_quat_w[0].detach().clone()

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
                )
            )
            csv_log.flush()

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
                ep += 1
                step = 0
                last_smoothed_actions.zero_()
                best_score = float("inf")
                best_q_arm = robot.data.joint_pos[0, joint_ids].detach().clone()
                no_improve_steps = 0
                holding_best = False

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
    env_cfg = _make_env_cfg()

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

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=env.device)

    _print_run_header(log_dir, env_cfg.scene.num_envs)

    try:
        if args_cli.play:
            _run_play(env, runner)
        else:
            _run_train(runner, agent_cfg)
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
