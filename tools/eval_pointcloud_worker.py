#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""点云遮挡任务 并行批量评估 worker（内层，由 eval_pointcloud_compare.py 调度）。

一次只评一个 checkpoint + 一组遮挡参数：num_envs 个并行环境跑满 total_episodes 就
停，逐 episode 结算误差，原始样本逐行写 CSV；统计与配对对比在
stats_pointcloud_occlusion.py 里做。

口径（沿用 eval_reach_single.py，误差参考点换成点云任务的抓取目标点）：
  - 碰撞即终止：collision + time_out 保留，reach_success 关闭（成功不提前 done，
    靠"曾达到"标志统计）。lock_joint_step_end 保留（done 恒 False）。
  - 距离量到【抓取目标点】custom_mdp.grasp_target_pos_w（工件局部系偏移
    grasp_offset_cm，与训练一致），**不量物体中心**。
  - 姿态误差 = 对称角 custom_mdp._symmetric_orientation_angle（180° 夹持对称，
    q_offset 单位四元数，与训练 rewards 一致）。
  - 成功三档（曾达到）：1cm/3°、0.5cm/3°、纯位置 1cm；另记 ever_lifted（工件高度
    超过 lift_height 的任意帧）。
  - dist_final / ori_final = 终止前最后一帧的值（结算帧）。
  - ever_collided = 任意帧接触力超阈（与训练的 collision 终止同一力算法），在
    --lockstep（关闭碰撞终止）下它就是"碰撞率"的来源。

seed 对齐（与 baseline 同 seed + 同 num_envs + 同遮挡参数）：
  - 遮挡中心/朝向采样（randomize_occlusion_state）与工件位姿 DR 同走全局 torch
    RNG、由 cfg.seed 播种；策略 forward 在 torch.no_grad() 下不消耗 RNG；本 worker
    走 PlayEnvCfg（观测噪声已关），观测路径不消耗 RNG。
  - 严格逐集配对需要两版 episode 结束时刻完全一致（reset 事件按"发生的先后"消耗
    RNG 流）。默认碰撞终止下两版碰撞时刻不同 → 只能保证"同一随机流/同一分布"；
    --lockstep（关闭碰撞终止、全部 1200 步超时）可保证两版逐集同一初始位姿+同一
    遮挡位置，实现严格配对。统计脚本会按存下的 obj_init_pos 校验对齐情况，自适应
    选择配对/非配对检验。
  - 因此 paired 运行时 --reset_grace_steps 必须为 0（默认 0）：开局碰撞过滤会因
    策略不同丢弃不同集，破坏逐集对应。

用法（必须用 isaaclab 环境；本机 conda 环境在 /root/gx-va）::

    CONDA_PREFIX=/root/gx-va /root/IsaacLab/isaaclab.sh -p tools/eval_pointcloud_worker.py \
        --checkpoint /root/autodl-tmp/logs/xarm7_pick_pointcloud_stage2_occ_random_box_25/2026-09-01_15-34-14/model_6600.pt \
        --occlusion_mode random_box --occlusion_severity 0.25 \
        --num_envs 50 --total_episodes 1000 --seed 0 \
        --tag occ25 --out /root/autodl-tmp/eval_occ/occ25.csv
"""

import argparse
import csv
import json
import math
import os
import sys
import tempfile
from datetime import datetime

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_CURRENT_DIR)
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from isaaclab.app import AppLauncher

# ── CLI 选项 ──────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="点云遮挡任务并行批量评估 worker（逐 episode 原始样本落盘 CSV）"
)
parser.add_argument("--checkpoint", type=str, required=True,
                    help="要评估的 checkpoint（model_*.pt，绝对/相对路径）")
parser.add_argument("--tag", type=str, default="run",
                    help="本次评估的标签，写入 meta JSON 与默认输出文件名")
parser.add_argument("--occlusion_mode", type=str, default="random_box",
                    choices=("none", "halfspace", "sphere", "random_sphere", "random_box"),
                    help="遮挡几何（须与训练该 checkpoint 时一致；none=无遮挡基线）")
parser.add_argument("--occlusion_severity", type=float, default=0.25,
                    help="遮挡程度 0.0..1.0（必须与训练时一致，否则观测对不上）")
parser.add_argument("--occlusion_axis", type=str, default="x",
                    help="halfspace 用：切哪一侧（x|-x|y|-y|z|-z，工件局部系）")
parser.add_argument("--occlusion_center", type=str, default="0.5,0.5,0.5",
                    help="sphere 用：球心在工件局部包围盒内的归一化分数 'cx,cy,cz'")
parser.add_argument("--grasp_offset_cm", type=float, default=3.0,
                    help="抓取目标点相对工件正上方偏移 cm（训练默认 3.0）")
parser.add_argument("--lift_height", type=float, default=0.73,
                    help="ever_lifted 判据：工件 z 超过该高度（米）")
parser.add_argument("--num_envs", type=int, default=50,
                    help="并行环境数（与对比的另一版保持一致，RNG 流才同构）")
parser.add_argument("--total_episodes", type=int, default=1000,
                    help="评估的总 episode 数")
parser.add_argument("--seed", type=int, default=0,
                    help="环境随机种子（与对比的另一版保持一致）")
parser.add_argument("--reset_grace_steps", type=int, default=0,
                    help="开局宽限步数：<=该步数内碰撞终止的 episode 判为无效局丢弃。"
                         "配对比较必须为 0（默认）")
parser.add_argument("--lockstep", action="store_true",
                    help="关闭碰撞终止（全部 episode 跑到 1200 步超时）→ 两版 reset "
                         "时刻完全一致，严格逐集配对。碰撞率改用 ever_collided")
parser.add_argument("--collision_force_thresh", type=float, default=0.5,
                    help="ever_collided 的接触力阈值（N），与训练 collision 终止一致")
parser.add_argument("--out", type=str, default=None,
                    help="CSV 输出路径；默认 logs/eval_pointcloud/<tag>_<时间戳>.csv")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 启动重型 AppLauncher 之前先校验参数（失败早退）
_ckpt = os.path.abspath(os.path.expanduser(args_cli.checkpoint))
if not os.path.isfile(_ckpt):
    raise SystemExit(f"[错误] checkpoint 不存在: {_ckpt}")
if not (0.0 <= args_cli.occlusion_severity <= 1.0):
    raise SystemExit(
        f"[错误] --occlusion_severity 必须在 [0,1]，got {args_cli.occlusion_severity}"
    )
if args_cli.num_envs < 1 or args_cli.total_episodes < 1:
    raise SystemExit("[错误] --num_envs / --total_episodes 必须 ≥ 1")
if args_cli.out is None:
    _stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    args_cli.out = os.path.join(
        _PROJECT_DIR, "logs", "eval_pointcloud", f"{args_cli.tag}_{_stamp}.csv"
    )
args_cli.out = os.path.abspath(args_cli.out)
os.makedirs(os.path.dirname(args_cli.out), exist_ok=True)

args_cli.headless = True          # 统计跑一律无头离屏
args_cli.enable_cameras = True    # 程序化桌面材质只在渲染型 Kit experience 下注册
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import configs.xarm7_pick_liftcube_mdp as custom_mdp
from configs.agents.rsl_rl_ppo_cfg import XArm7PickPointCloudPPORunnerCfg
from configs.occlusion import parse_occlusion_center
from configs.point_bridge_pointcloud import M_OBJ, N_ROBOT, POINT_DIM
from configs.xarm7_pick_pointcloud_env_cfg import (
    _OCCLUSION_STATE_BUF,
    XArm7PickPointCloudPlayEnvCfg,
    disable_point_cloud_noise,
    set_grasp_target_offset_cm,
    set_observation_stage,
    set_occlusion_stage2,
    set_success_termination_enabled,
)
from model.pointnet_actor_critic import register_with_rsl_rl

# ── 落盘列（固定顺序，stats 脚本按名读）────────────────────────────────────────
CSV_COLUMNS = [
    "episode_index", "env_index", "seed",
    "obj_init_pos_x", "obj_init_pos_y", "obj_init_pos_z",
    "obj_init_quat_w", "obj_init_quat_x", "obj_init_quat_y", "obj_init_quat_z",
    "occ_center_x", "occ_center_y", "occ_center_z",
    "occ_quat_w", "occ_quat_x", "occ_quat_y", "occ_quat_z",
    "dist_best_cm", "ori_best_deg", "dist_final_cm", "ori_final_deg",
    "reach_1cm_3deg", "reach_0p5cm_3deg", "reach_pos_1cm", "ever_lifted",
    "collision", "timeout", "ever_collided", "ep_len", "reward_total",
]

# 抓取目标偏移（工件局部系），与 set_grasp_target_offset_cm 同公式
_OFFSET_LOCAL = (0.0, 0.0, -args_cli.grasp_offset_cm / 100.0)


def _build_env_cfg():
    """镜像 training 的 play 路径：PlayEnvCfg + stage2 + 遮挡 + 关噪声 + 无 success 终止。"""
    cfg = XArm7PickPointCloudPlayEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.seed = args_cli.seed

    set_observation_stage(cfg, 2, keep_appearance=True)
    # 即使 mode='none' 也调用 —— 训练时就走 point_bridge_point_cloud_occluded 这个
    # 观测函数，评估必须用同一个函数，否则观测与 checkpoint 对不上。
    set_occlusion_stage2(
        cfg,
        mode=args_cli.occlusion_mode,
        severity=args_cli.occlusion_severity,
        axis=args_cli.occlusion_axis,
        center=parse_occlusion_center(args_cli.occlusion_center),
    )
    disable_point_cloud_noise(cfg)
    set_success_termination_enabled(cfg, False)          # 成功不提前 done
    if args_cli.lockstep:
        cfg.terminations.collision = None                # 全部跑满 1200 步超时
    set_grasp_target_offset_cm(cfg, args_cli.grasp_offset_cm)  # 必须在 success setter 之后
    return cfg


def _dist_cm_ori_deg(base_env):
    """TCP→抓取目标点距离(cm) 与对称姿态误差(deg)。返回 (N,) (N,)。"""
    obj = base_env.scene["object"]
    ee_frame = base_env.scene["ee_frame"]
    target = custom_mdp.grasp_target_pos_w(obj, _OFFSET_LOCAL)
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    dist_cm = torch.norm(target - ee_pos, dim=-1) * 100.0

    ee_quat = ee_frame.data.target_quat_w[:, 0, :]
    q_obj = obj.data.root_quat_w
    ori_rad = custom_mdp._symmetric_orientation_angle(
        ee_quat, q_obj,
        q_offset=(0.0, 0.0, 0.0, 1.0),
        grasp_symmetry_quats=custom_mdp._GRASP_SYMMETRY_QUATS_WXYZ,
    )
    return dist_cm, ori_rad * (180.0 / math.pi)


def _contact_force_per_env(base_env):
    """逐 env 的夹爪接触力幅值(N)，与训练 collision 终止的 contact_force_penalty
    同算法。返回 (N,)。"""
    sensor = base_env.scene.sensors["grip_contact"]
    forces = sensor.data.net_forces_w_history          # (N, hist, bodies, 3)
    return forces.norm(dim=-1).max(dim=1).values.sum(dim=-1)  # (N,)


def main():
    env_cfg = _build_env_cfg()
    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    base_env = env.unwrapped
    device = env.device
    N = args_cli.num_envs

    # ── 载入策略（与 training 的 play 路径完全同路）───────────────────────────
    agent_cfg = XArm7PickPointCloudPPORunnerCfg()
    agent_cfg.policy.num_object_points = M_OBJ
    agent_cfg.policy.num_robot_points = N_ROBOT
    agent_cfg.policy.point_dim = POINT_DIM
    agent_cfg.policy.num_wrist_points = 0              # stage 2 无腕部分支
    register_with_rsl_rl()
    _tmp = tempfile.mkdtemp(prefix="eval_pointcloud_")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=_tmp, device=device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=device)

    # ── episode 内累计标志（"曾达到"；碰撞/超时按结束方式判定）────────────────
    ever_1cm3deg = torch.zeros(N, dtype=torch.bool, device=device)
    ever_05cm3deg = torch.zeros(N, dtype=torch.bool, device=device)
    ever_pos1cm = torch.zeros(N, dtype=torch.bool, device=device)
    ever_lift = torch.zeros(N, dtype=torch.bool, device=device)
    ever_collide = torch.zeros(N, dtype=torch.bool, device=device)
    best_dist = torch.full((N,), 1e9, device=device)
    best_ori = torch.full((N,), 1e9, device=device)
    last_dist = torch.zeros(N, device=device)
    last_ori = torch.zeros(N, device=device)
    ep_steps = torch.zeros(N, dtype=torch.long, device=device)
    ep_reward = torch.zeros(N, device=device)
    # 每集开局快照：初始位姿（配对校验键）+ 遮挡中心/朝向
    init_pos = torch.zeros(N, 3, device=device)
    init_quat = torch.zeros(N, 4, device=device)
    occ_center = torch.full((N, 3), float("nan"), device=device)
    occ_quat = torch.full((N, 4), float("nan"), device=device)

    grace = max(0, args_cli.reset_grace_steps)

    # ── 结算 + 落盘 ───────────────────────────────────────────────────────────
    csv_f = open(args_cli.out, "w", newline="", encoding="utf-8")
    writer = csv.DictWriter(csv_f, fieldnames=CSV_COLUMNS)
    writer.writeheader()

    totals = {
        "episodes": 0,
        "reach_1cm_3deg": 0,
        "reach_0p5cm_3deg": 0,
        "reach_pos_1cm": 0,
        "ever_lifted": 0,
        "collision": 0,
        "timeout": 0,
        "ever_collided": 0,
        "discarded_early_collision": 0,
    }
    sum_best_dist = 0.0
    sum_best_ori = 0.0

    def _capture_episode_start(env_ids_tensor):
        """env_ids_tensor 为 True 的 env 刚进入新 episode：快照初始位姿 + 遮挡状态。"""
        idx = torch.nonzero(env_ids_tensor, as_tuple=False).flatten()
        if idx.numel() == 0:
            return
        obj = base_env.scene["object"]
        init_pos[idx] = obj.data.root_pos_w[idx]
        init_quat[idx] = obj.data.root_quat_w[idx]
        if args_cli.occlusion_mode in ("random_sphere", "random_box"):
            state = _OCCLUSION_STATE_BUF.get(id(base_env))
            if state is not None:
                occ_center[idx] = state["center"][idx]
                occ_quat[idx] = state["quat"][idx]

    def _settle(env_mask, collided_mask, timeout_mask):
        """结算刚结束的 episode：写 CSV 行，然后清掉这些 env 的累计标志。"""
        nonlocal sum_best_dist, sum_best_ori
        idx = torch.nonzero(env_mask, as_tuple=False).flatten()
        if idx.numel() == 0:
            return 0

        # 无效局：开局宽限内碰撞终止（默认 grace=0 关闭；配对比较必须关）
        if grace > 0:
            invalid = collided_mask & env_mask & (ep_steps <= grace)
        else:
            invalid = torch.zeros_like(env_mask)
        valid = env_mask & (~invalid)
        n_invalid = int(invalid.sum().item())
        totals["discarded_early_collision"] += n_invalid

        vidx = torch.nonzero(valid, as_tuple=False).flatten().cpu().tolist()
        n = len(vidx)
        for j in vidx:
            ep = totals["episodes"]
            row = {
                "episode_index": ep,
                "env_index": j,
                "seed": args_cli.seed,
            }
            for k, name in enumerate(("x", "y", "z")):
                row[f"obj_init_pos_{name}"] = f"{init_pos[j, k].item():.6f}"
            for k, name in enumerate(("w", "x", "y", "z")):
                row[f"obj_init_quat_{name}"] = f"{init_quat[j, k].item():.6f}"
            if args_cli.occlusion_mode in ("random_sphere", "random_box") \
                    and not bool(torch.isnan(occ_center[j, 0]).item()):
                for k, name in enumerate(("x", "y", "z")):
                    row[f"occ_center_{name}"] = f"{occ_center[j, k].item():.6f}"
                for k, name in enumerate(("w", "x", "y", "z")):
                    row[f"occ_quat_{name}"] = f"{occ_quat[j, k].item():.6f}"
            else:
                for name in ("x", "y", "z"):
                    row[f"occ_center_{name}"] = ""
                for name in ("w", "x", "y", "z"):
                    row[f"occ_quat_{name}"] = ""

            bd = min(float(best_dist[j].item()), 1e4)
            bo = min(float(best_ori[j].item()), 1e4)
            row.update({
                "dist_best_cm": f"{bd:.4f}",
                "ori_best_deg": f"{bo:.4f}",
                "dist_final_cm": f"{last_dist[j].item():.4f}",
                "ori_final_deg": f"{last_ori[j].item():.4f}",
                "reach_1cm_3deg": int(ever_1cm3deg[j].item()),
                "reach_0p5cm_3deg": int(ever_05cm3deg[j].item()),
                "reach_pos_1cm": int(ever_pos1cm[j].item()),
                "ever_lifted": int(ever_lift[j].item()),
                "collision": int(collided_mask[j].item()),
                "timeout": int(timeout_mask[j].item()),
                "ever_collided": int(ever_collide[j].item()),
                "ep_len": int(ep_steps[j].item()),
                "reward_total": f"{ep_reward[j].item():.4f}",
            })
            writer.writerow(row)

            totals["episodes"] += 1
            totals["reach_1cm_3deg"] += row["reach_1cm_3deg"]
            totals["reach_0p5cm_3deg"] += row["reach_0p5cm_3deg"]
            totals["reach_pos_1cm"] += row["reach_pos_1cm"]
            totals["ever_lifted"] += row["ever_lifted"]
            totals["collision"] += row["collision"]
            totals["timeout"] += row["timeout"]
            totals["ever_collided"] += row["ever_collided"]
            sum_best_dist += bd
            sum_best_ori += bo

        # 清掉所有刚结束 env（含无效局）的累计标志，开始新 episode
        ever_1cm3deg[idx] = False
        ever_05cm3deg[idx] = False
        ever_pos1cm[idx] = False
        ever_lift[idx] = False
        ever_collide[idx] = False
        best_dist[idx] = 1e9
        best_ori[idx] = 1e9
        ep_steps[idx] = 0
        ep_reward[idx] = 0
        return n

    # ── 主循环 ────────────────────────────────────────────────────────────────
    print("=" * 78)
    print(f"[eval] checkpoint: {args_cli.checkpoint}")
    print(f"[eval] 遮挡: mode={args_cli.occlusion_mode} "
          f"severity={args_cli.occlusion_severity:.2f}")
    if args_cli.lockstep:
        print("[eval] lockstep: 碰撞终止已关闭（全部 1200 步超时 → 严格逐集配对）")
    print(f"[eval] num_envs={N}  total_episodes={args_cli.total_episodes}  "
          f"seed={args_cli.seed}  grace={grace}")
    print(f"[eval] 落盘: {args_cli.out}")
    print("=" * 78, flush=True)

    obs, _ = env.reset()
    # 首帧（frame 0）计入本集：快照 + 指标
    _capture_episode_start(torch.ones(N, dtype=torch.bool, device=device))
    dist_cm, ori_deg = _dist_cm_ori_deg(base_env)
    lifted = base_env.scene["object"].data.root_pos_w[:, 2] > args_cli.lift_height
    ever_1cm3deg |= (dist_cm <= 1.0) & (ori_deg <= 3.0)
    ever_05cm3deg |= (dist_cm <= 0.5) & (ori_deg <= 3.0)
    ever_pos1cm |= dist_cm <= 1.0
    ever_lift |= lifted
    ever_collide |= _contact_force_per_env(base_env) > args_cli.collision_force_thresh
    best_dist = torch.minimum(best_dist, dist_cm)
    best_ori = torch.minimum(best_ori, ori_deg)
    last_dist = dist_cm
    last_ori = ori_deg

    step = 0
    while totals["episodes"] < args_cli.total_episodes and simulation_app.is_running():
        with torch.no_grad():
            actions = policy(obs)
        obs, rewards, dones, _ = env.step(actions)

        ep_steps += 1
        ep_reward += rewards      # done env 该步 reward 已被 reset 清零，不影响旧集

        # 先读结束方式（step 内部已对 done 的 env 计算 terminated/time_out）。
        # terminated=True 表示非超时终止（本环境里即碰撞 collision）。
        done_mask = collided_mask = timeout_mask = None
        if isinstance(dones, torch.Tensor) and dones.any():
            done_mask = dones.bool().flatten()
            terminated = getattr(base_env, "reset_terminated", None)
            timed_out = getattr(base_env, "reset_time_outs", None)
            collided_mask = (
                terminated.bool().flatten() if terminated is not None
                else torch.zeros(N, dtype=torch.bool, device=device)
            )
            timeout_mask = (
                timed_out.bool().flatten() if timed_out is not None
                else done_mask & (~collided_mask)
            )

        # 结算 done 的 episode：用【截至上一帧】的累计标志（终止发生前的生命周期），
        # 必须在"更新当前帧指标"之前结算，否则 reset 后的新局初值会污染旧 episode。
        if done_mask is not None:
            _settle(done_mask, collided_mask, timeout_mask)
            # done 的 env 已被 step 内部 reset，当前帧即新 episode 的 frame 0
            _capture_episode_start(done_mask)

        # 更新当前帧指标（刚 reset 的 env 其标志已清零，这里写入新集首帧）
        dist_cm, ori_deg = _dist_cm_ori_deg(base_env)
        lifted = base_env.scene["object"].data.root_pos_w[:, 2] > args_cli.lift_height
        ever_1cm3deg |= (dist_cm <= 1.0) & (ori_deg <= 3.0)
        ever_05cm3deg |= (dist_cm <= 0.5) & (ori_deg <= 3.0)
        ever_pos1cm |= dist_cm <= 1.0
        ever_lift |= lifted
        ever_collide |= _contact_force_per_env(base_env) > args_cli.collision_force_thresh
        best_dist = torch.minimum(best_dist, dist_cm)
        best_ori = torch.minimum(best_ori, ori_deg)
        last_dist = dist_cm
        last_ori = ori_deg

        step += 1
        if step % 50 == 0:
            print(f"[eval] episodes={totals['episodes']}/{args_cli.total_episodes} "
                  f"(step={step})", flush=True)

    csv_f.close()
    env.close()

    # ── meta JSON（参数 + 汇总口径）────────────────────────────────────────────
    done_count = totals["episodes"]
    meta = {
        "tag": args_cli.tag,
        "checkpoint": args_cli.checkpoint,
        "occlusion_mode": args_cli.occlusion_mode,
        "occlusion_severity": args_cli.occlusion_severity,
        "occlusion_axis": args_cli.occlusion_axis,
        "occlusion_center": args_cli.occlusion_center,
        "grasp_offset_cm": args_cli.grasp_offset_cm,
        "lift_height": args_cli.lift_height,
        "num_envs": N,
        "seed": args_cli.seed,
        "total_episodes": args_cli.total_episodes,
        "lockstep": args_cli.lockstep,
        "reset_grace_steps": grace,
        "collision_force_thresh": args_cli.collision_force_thresh,
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "episodes_completed": done_count,
        "discarded_early_collision": totals["discarded_early_collision"],
        "counts": {k: totals[k] for k in (
            "reach_1cm_3deg", "reach_0p5cm_3deg", "reach_pos_1cm",
            "ever_lifted", "collision", "timeout", "ever_collided")},
        "rates": {k: totals[k] / max(done_count, 1) for k in (
            "reach_1cm_3deg", "reach_0p5cm_3deg", "reach_pos_1cm",
            "ever_lifted", "collision", "timeout", "ever_collided")},
        "mean_best_dist_cm": sum_best_dist / max(done_count, 1),
        "mean_best_ori_deg": sum_best_ori / max(done_count, 1),
    }
    meta_path = os.path.splitext(args_cli.out)[0] + ".meta.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"[eval] 完成，{done_count} 个 episode 落盘: {args_cli.out}", flush=True)
    print(f"[eval] meta: {meta_path}", flush=True)
    print(json.dumps(meta, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nError: {exc}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
