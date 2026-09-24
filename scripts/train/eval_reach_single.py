#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""单版本并发评估（内层 worker）。

由外层 eval_reach_compare.py 用 isaaclab.sh 启动，每次只评一个版本。

口径（与用户确认一致）：
    - 碰撞即终止：保留 collision 终止 + time_out 终止（和训练/部署一致）。
      reach_success 终止关闭（成功不提前 done，靠“曾达到”标志统计）。
      lock_joint_step_end 保留（其 done 恒 False，仅做末端锁定）。
    - 成功 = episode 结束（碰撞或超时）之前【曾进入过】阈值（任意一帧满足即记一次）。
      统计三档：
          reach_1cm_3deg   : dist <= 1.0cm 且 ori <= 3deg
          reach_0p5cm_3deg : dist <= 0.5cm 且 ori <= 3deg
          reach_pos_1cm    : dist <= 1.0cm（不看姿态，参考）
    - 碰撞：该 episode 是否【因碰撞而 terminated】（reset_terminated 区分碰撞 vs 超时）。
    - 成功与碰撞是独立标志：一个 episode 可同时“曾达到阈值”且“因碰撞结束”。

并发：用 num_envs 个并行环境，跑满 total_episodes 就停。结果写到 --out JSON。
"""

import argparse
import json
import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="reach 评估 worker（单版本并发）")
parser.add_argument("--version", choices=["vision", "nopc"], required=True,
                    help="vision=782维有点云；nopc=398维无点云")
parser.add_argument("--checkpoint", type=str, required=True)
parser.add_argument("--num_envs", type=int, default=200, help="并行环境数")
parser.add_argument("--total_episodes", type=int, default=1000, help="评估的总 episode 数")
parser.add_argument("--out", type=str, required=True, help="结果 JSON 输出路径")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--fix_table_height", action="store_true",
                    help="评估时固定桌面高度，不做高度随机化（height_range=0）。"
                         "工件 x/y/yaw、相机、光照、桌色随机化仍保留。")
parser.add_argument("--reset_grace_steps", type=int, default=2,
                    help="开局宽限步数：若某 episode 在 <= 该步数内就因碰撞 terminated，"
                         "判定为随机化初始穿插导致的无效局，整局丢弃（不计入任何统计）。"
                         "设为 0 关闭该过滤。")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

args_cli.headless = True          # 评估一律无头
args_cli.enable_cameras = True    # 视觉观测需要相机管线

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

import configs.xarm7_pick_liftcube_mdp as custom_mdp
from isaaclab.managers import SceneEntityCfg

# 阈值（与训练判定一致）
COLLISION_FORCE_THRESH = 0.5
_GRASP_SYM = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)
_Q_OFFSET = (0.0, 0.0, 0.0, 1.0)


def _load_env_and_agent():
    """按版本加载对应 env_cfg(PlayEnvCfg) 与 PPO agent cfg。"""
    if args_cli.version == "vision":
        from configs.xarm7_pick_vision_env_cfg import XArm7PickLiftCubePlayEnvCfg
        from configs.agents.rsl_rl_ppo_cfg import XArm7PickLiftCubePPORunnerCfg as AgentCfg
    else:  # nopc
        from configs.xarm7_pick_vision_nopc_env_cfg import XArm7PickLiftCubePlayEnvCfg
        from configs.agents.rsl_rl_ppo_cfg import XArm7PickPPORunnerCfg_V58 as AgentCfg
    return XArm7PickLiftCubePlayEnvCfg, AgentCfg


def _dist_cm_and_ori_deg(base_env):
    """逐 env 计算 TCP→工件 距离(cm) 与最小对称姿态误差(deg)。返回 (N,) (N,)。"""
    obj = base_env.scene["object"]
    ee_frame = base_env.scene["ee_frame"]
    obj_pos = obj.data.root_pos_w                       # (N,3)
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]        # (N,3)
    dist_cm = torch.norm(obj_pos - ee_pos, dim=-1) * 100.0

    ee_quat = ee_frame.data.target_quat_w[:, 0, :]
    q_obj = obj.data.root_quat_w
    ori_rad = custom_mdp._symmetric_orientation_angle(
        ee_quat, q_obj, q_offset=_Q_OFFSET, grasp_symmetry_quats=_GRASP_SYM,
    )                                                   # (N,) rad
    ori_deg = ori_rad * (180.0 / 3.14159265358979)
    return dist_cm, ori_deg


def _contact_force_per_env(base_env):
    """逐 env 的 link7 接触力幅值(N)，与 contact_force_penalty 同算法。返回 (N,)。"""
    sensor = base_env.scene.sensors["grip_contact"]
    forces = sensor.data.net_forces_w_history          # (N, hist, bodies, 3)
    return forces.norm(dim=-1).max(dim=1).values.sum(dim=-1)  # (N,)


def main():
    PlayEnvCfg, AgentCfg = _load_env_and_agent()

    env_cfg = PlayEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs
    env_cfg.seed = args_cli.seed

    # 碰撞即终止：保留 collision 与 time_out 终止（和训练/部署一致）。
    # 只关闭 reach_success（成功不提前 done，靠“曾达到”标志统计）。
    # lock_joint_step_end 保留（其 done 恒 False，仅做末端锁定，与训练动作链路一致）。
    if hasattr(env_cfg.terminations, "reach_success"):
        env_cfg.terminations.reach_success = None
    # collision 与 time_out 保留。

    # 可选：固定桌面高度（只关高度随机化，其余随机化保留）。
    if args_cli.fix_table_height:
        ev = env_cfg.events
        if hasattr(ev, "reset_table_and_object") and ev.reset_table_and_object is not None:
            ev.reset_table_and_object.params["height_range"] = 0.0
            print(f"[eval:{args_cli.version}] 桌面高度随机化已关闭（height_range=0）", flush=True)
        else:
            print(f"[eval:{args_cli.version}][WARN] 未找到 reset_table_and_object event，"
                  f"无法固定桌面高度", flush=True)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    base_env = env.unwrapped
    device = env.device
    N = args_cli.num_envs

    # 加载策略
    import tempfile
    agent_cfg = AgentCfg()
    _tmp = tempfile.mkdtemp(prefix=f"eval_{args_cli.version}_")
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=_tmp, device=device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=device)
    print(f"[eval:{args_cli.version}] checkpoint 已加载: {args_cli.checkpoint}", flush=True)

    # episode 内累计标志（“曾达到”阈值；碰撞按结束方式判定，见下）
    ever_1cm_3deg = torch.zeros(N, dtype=torch.bool, device=device)
    ever_0p5cm_3deg = torch.zeros(N, dtype=torch.bool, device=device)
    ever_pos_1cm = torch.zeros(N, dtype=torch.bool, device=device)
    best_dist_cm = torch.full((N,), 1e9, device=device)
    best_ori_deg = torch.full((N,), 1e9, device=device)
    # 自维护的 per-env 步数计数器（不依赖 Isaac 内部 episode_length_buf，
    # 因为后者在 step 内对 done 的 env 已被清零，外部读不到“碰撞时存活步数”）。
    ep_steps = torch.zeros(N, dtype=torch.long, device=device)

    grace = max(0, args_cli.reset_grace_steps)

    # 全局完成 episode 的计数器（各档命中数）
    totals = {
        "episodes": 0,
        "reach_1cm_3deg": 0,
        "reach_0p5cm_3deg": 0,
        "reach_pos_1cm": 0,
        "collision": 0,    # 因碰撞 terminated 的 episode 数
        "timeout": 0,      # 因超时结束的 episode 数
        "discarded_early_collision": 0,  # 开局宽限内碰撞→无效局，已丢弃
    }
    sum_best_dist = 0.0
    sum_best_ori = 0.0

    obs, _ = env.reset()

    def _accumulate(env_mask, collided_mask, timeout_mask):
        """结算 env_mask 为 True（刚结束）的 episode。
        若 episode 在 <= grace 步内因碰撞结束，判为随机化初始穿插的无效局，整局丢弃。
        无论有效/无效，最后都清掉这些 env 的累计标志。"""
        idx = torch.nonzero(env_mask, as_tuple=False).flatten()
        if idx.numel() == 0:
            return 0

        # 无效局：碰撞结束 且 存活步数 <= grace
        if grace > 0:
            invalid = collided_mask & env_mask & (ep_steps <= grace)
        else:
            invalid = torch.zeros_like(env_mask)
        valid_mask = env_mask & (~invalid)

        n_invalid = int(invalid.sum().item())
        totals["discarded_early_collision"] += n_invalid

        vidx = torch.nonzero(valid_mask, as_tuple=False).flatten()
        n = int(vidx.numel())
        if n > 0:
            totals["episodes"] += n
            totals["reach_1cm_3deg"] += int(ever_1cm_3deg[vidx].sum().item())
            totals["reach_0p5cm_3deg"] += int(ever_0p5cm_3deg[vidx].sum().item())
            totals["reach_pos_1cm"] += int(ever_pos_1cm[vidx].sum().item())
            totals["collision"] += int(collided_mask[vidx].sum().item())
            totals["timeout"] += int(timeout_mask[vidx].sum().item())
            nonlocal sum_best_dist, sum_best_ori
            sum_best_dist += float(best_dist_cm[vidx].clamp(max=1e4).sum().item())
            sum_best_ori += float(best_ori_deg[vidx].clamp(max=1e4).sum().item())

        # 清掉所有刚结束 env（含无效局）的累计标志与步数，开始新 episode
        ever_1cm_3deg[idx] = False
        ever_0p5cm_3deg[idx] = False
        ever_pos_1cm[idx] = False
        best_dist_cm[idx] = 1e9
        best_ori_deg[idx] = 1e9
        ep_steps[idx] = 0
        return n

    target = args_cli.total_episodes
    step = 0
    while totals["episodes"] < target and simulation_app.is_running():
        with torch.no_grad():
            actions = policy(obs)
        obs, _, dones, _ = env.step(actions)

        # 本 episode 步数 +1（在结算之前，使 done 的 env 的 ep_steps = 该局总步数）。
        ep_steps += 1

        # 先读结束方式（step 内部已对 done 的 env 计算 terminated/time_out）。
        # terminated=True 表示非超时终止（本环境里即碰撞 collision）。
        done_mask = collided_mask = timeout_mask = None
        if isinstance(dones, torch.Tensor) and dones.any():
            done_mask = dones.bool().flatten()
            terminated = getattr(base_env, "reset_terminated", None)
            timed_out = getattr(base_env, "reset_time_outs", None)
            collided_mask = (terminated.bool().flatten() if terminated is not None
                             else torch.zeros(N, dtype=torch.bool, device=device))
            timeout_mask = (timed_out.bool().flatten() if timed_out is not None
                            else done_mask & (~collided_mask))

        # 结算 done 的 episode：用【截至上一帧】的累计标志（碰撞/超时发生前的生命周期），
        # 必须在“更新当前帧指标”之前结算，否则 reset 后的新局初值会污染旧 episode。
        if done_mask is not None:
            _accumulate(done_mask, collided_mask, timeout_mask)

        # 更新当前帧指标。对刚 reset 的 env，其标志已在 _accumulate 内清零，
        # 这里写入的是新 episode 的第一帧，归属正确。
        dist_cm, ori_deg = _dist_cm_and_ori_deg(base_env)
        ever_1cm_3deg |= (dist_cm <= 1.0) & (ori_deg <= 3.0)
        ever_0p5cm_3deg |= (dist_cm <= 0.5) & (ori_deg <= 3.0)
        ever_pos_1cm |= dist_cm <= 1.0
        best_dist_cm = torch.minimum(best_dist_cm, dist_cm)
        best_ori_deg = torch.minimum(best_ori_deg, ori_deg)

        step += 1
        if step % 50 == 0:
            print(f"[eval:{args_cli.version}] episodes={totals['episodes']}/{target} "
                  f"(step={step})", flush=True)

    # 收尾：done_count 只含【有效】episode；无效局（开局碰撞）已在结算时剔除。
    done_count = totals["episodes"]
    result = {
        "version": args_cli.version,
        "checkpoint": args_cli.checkpoint,
        "num_envs": N,
        "reset_grace_steps": grace,
        "episodes_completed": done_count,
        "discarded_early_collision": totals["discarded_early_collision"],
        "counts": {
            "reach_1cm_3deg": totals["reach_1cm_3deg"],
            "reach_0p5cm_3deg": totals["reach_0p5cm_3deg"],
            "reach_pos_1cm": totals["reach_pos_1cm"],
            "collision": totals["collision"],
            "timeout": totals["timeout"],
        },
        "rates": {
            "reach_1cm_3deg": totals["reach_1cm_3deg"] / max(done_count, 1),
            "reach_0p5cm_3deg": totals["reach_0p5cm_3deg"] / max(done_count, 1),
            "reach_pos_1cm": totals["reach_pos_1cm"] / max(done_count, 1),
            "collision": totals["collision"] / max(done_count, 1),
            "timeout": totals["timeout"] / max(done_count, 1),
        },
        "mean_best_dist_cm": sum_best_dist / max(done_count, 1),
        "mean_best_ori_deg": sum_best_ori / max(done_count, 1),
    }

    with open(args_cli.out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"[eval:{args_cli.version}] 完成，结果写入: {args_cli.out}", flush=True)
    print(json.dumps(result, ensure_ascii=False, indent=2), flush=True)

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nError: {exc}", flush=True)
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
