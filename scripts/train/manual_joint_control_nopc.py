#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""xArm7 手动关节控制调试器（nopc 398 维环境）

用一个 matplotlib GUI 窗口手动控制 7 个关节（滑块=绝对目标角，单位 度），
实时绘制：
    - 每个奖励项的 per-step reward 曲线
    - 总 reward 曲线
    - link7 接触力（N）曲线 + 当前数值
    - TCP→工件 距离(cm) 与 姿态误差(deg) 曲线
    - 碰撞 / 成功 状态灯（达到判定条件会亮，但【不会终止】episode）

与训练动作链路保持一致：
    滑块给出目标角 q_des，每个 RL step 喂入
        action = clip((q_des - q_current) / 0.6°, -1, +1)
    经 KinematicRelativeJointDirectAction 转成 ±0.6°/step 的增量目标。

关键：本脚本【关闭所有会真正终止 episode 的 termination】，
      done 只用于点亮状态灯，绝不触发 env 内部 reset。

用法：
    /home/gxai/IsaacLab/isaaclab.sh -p scripts/train/manual_joint_control_nopc.py \
        [--checkpoint logs/.../model_xxxx.pt]   # checkpoint 可选；不给则纯手动

    勾选 "policy" 复选框可切换为“由策略接管”，再次取消则回到手动滑块。
"""

import argparse
import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="xArm7 手动关节控制调试器（nopc 398 维，done 不终止）"
)
parser.add_argument("--checkpoint", type=str, default=None,
                    help="可选：加载策略 checkpoint，用于 policy 接管模式。")
parser.add_argument("--seed", type=int, default=None)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 需要相机管线（环境观测依赖 camera_fixed / camera_wrist 的 Theia 特征）。
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import math

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from configs.xarm7_pick_vision_nopc_env_cfg import (
    XArm7PickLiftCubePlayEnvCfg,
    INIT_JOINT_POS,
    _ARM_ACTION_SCALE,        # rad，= radians(0.6)
    _ARM_JOINT_LIMITS_LOW,    # tensor(7,) rad
    _ARM_JOINT_LIMITS_HIGH,   # tensor(7,) rad
)
from configs.agents.rsl_rl_ppo_cfg import XArm7PickPPORunnerCfg_V58

# 成功判定阈值（与环境 reach_success / ee_reached_object 一致）。
SUCCESS_DIST_CM = 1.0
SUCCESS_ORI_DEG = 3.0
# 碰撞判定阈值（与 collision_penalty / contact_force_done 一致）。
COLLISION_FORCE_THRESH = 0.5

JOINT_NAMES = [f"joint{i}" for i in range(1, 8)]


# ── 四元数工具（wxyz）─────────────────────────────────────────────────────────
def _quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(-1)
    w2, x2, y2, z2 = q2.unbind(-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def _quat_inv(q: torch.Tensor) -> torch.Tensor:
    q = q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1.0e-8)
    return torch.cat((q[..., 0:1], -q[..., 1:4]), dim=-1)


def _extract_ee_pose(ee_frame):
    data = ee_frame.data
    pos = None
    for name in ("target_pos_w", "source_pos_w", "frame_pos_w"):
        if hasattr(data, name):
            pos = getattr(data, name)
            if pos.ndim == 3:
                pos = pos[:, 0, :]
            break
    quat = None
    for name in ("target_quat_w", "source_quat_w", "frame_quat_w"):
        if hasattr(data, name):
            quat = getattr(data, name)
            if quat.ndim == 3:
                quat = quat[:, 0, :]
            break
    if pos is None or quat is None:
        raise RuntimeError("无法从 ee_frame.data 读取 TCP 位姿。")
    return pos, quat


def _reach_metrics(base_env) -> dict:
    """TCP→工件 距离(cm) 与最小对称姿态误差(deg)。"""
    obj = base_env.scene["object"]
    ee_frame = base_env.scene["ee_frame"]
    obj_pos = obj.data.root_pos_w[0]
    obj_quat = obj.data.root_quat_w[0]
    ee_pos_all, ee_quat_all = _extract_ee_pose(ee_frame)
    ee_pos, ee_quat = ee_pos_all[0], ee_quat_all[0]

    dist_cm = float((torch.linalg.norm(obj_pos - ee_pos) * 100.0).item())

    # 夹爪 180° 对称：取两种等价姿态里误差更小的。
    best = 180.0
    for sym in ((1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0, 1.0)):
        q_sym = torch.tensor(sym, device=ee_quat.device, dtype=ee_quat.dtype)
        rel = _quat_mul(_quat_inv(_quat_mul(ee_quat, q_sym)), obj_quat)
        rel = rel / torch.clamp(torch.linalg.norm(rel), min=1.0e-8)
        w = torch.abs(rel[0])
        xyz = torch.linalg.norm(rel[1:4])
        ang = float(torch.rad2deg(2.0 * torch.atan2(xyz, torch.clamp(w, min=1.0e-8))).item())
        best = min(best, ang)
    return {"dist_cm": dist_cm, "ori_deg": best}


def _contact_force(base_env) -> float:
    """link7 接触力幅值（N），与 contact_force_penalty 同一算法。"""
    sensor = base_env.scene.sensors["grip_contact"]
    forces = sensor.data.net_forces_w_history       # (envs, hist, bodies, 3)
    mag = forces.norm(dim=-1).max(dim=1).values.sum(dim=-1)  # (envs,)
    return float(mag[0].item())


def main():
    # ── 1. 构造环境：单环境 + 关闭所有真正会终止的 termination ──────────────
    env_cfg = XArm7PickLiftCubePlayEnvCfg()
    env_cfg.scene.num_envs = 1
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    # done 不终止：清掉碰撞 / 超时 / 成功终止。
    # 保留 lock_joint_step_end（其 done 恒为 False，仅做末端二次锁定副作用，
    # 与训练动作链路一致）。
    env_cfg.terminations.collision = None
    env_cfg.terminations.time_out = None
    env_cfg.terminations.reach_success = None

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    base_env = env.unwrapped
    device = env.device

    robot = base_env.scene["robot"]
    joint_ids, _ = robot.find_joints(["joint[1-7]"])
    joint_ids = list(joint_ids)

    limits_low_deg = [math.degrees(v) for v in _ARM_JOINT_LIMITS_LOW.tolist()]
    limits_high_deg = [math.degrees(v) for v in _ARM_JOINT_LIMITS_HIGH.tolist()]
    init_deg = [math.degrees(INIT_JOINT_POS[n]) for n in JOINT_NAMES]
    scale_rad = float(_ARM_ACTION_SCALE)      # 0.6° in rad

    # ── 2. 可选策略（policy 接管模式）──────────────────────────────────────
    policy = None
    if args_cli.checkpoint is not None:
        import tempfile
        agent_cfg = XArm7PickPPORunnerCfg_V58()
        _tmp_log = tempfile.mkdtemp(prefix="manual_joint_ctrl_")
        runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=_tmp_log, device=device)
        runner.load(args_cli.checkpoint)
        policy = runner.get_inference_policy(device=device)
        print(f"[Manual] 已加载 checkpoint: {args_cli.checkpoint}（可在 GUI 勾选 policy 接管）")
    else:
        print("[Manual] 未提供 checkpoint：纯手动模式。")

    # ── 3. matplotlib GUI ──────────────────────────────────────────────────
    import matplotlib.pyplot as plt
    from matplotlib.widgets import Slider, Button, CheckButtons

    plt.ion()

    # 控制窗口：7 个关节滑块 + 复位按钮 + policy 接管复选框
    fig_ctrl = plt.figure("Joint Control", figsize=(7.5, 6.0))
    fig_ctrl.subplots_adjust(left=0.30, right=0.92, top=0.94, bottom=0.20)
    fig_ctrl.suptitle("手动关节控制（滑块=目标角°）", fontsize=11)

    sliders = []
    target_deg = list(init_deg)  # 当前目标角（度）
    for i, name in enumerate(JOINT_NAMES):
        ax_s = fig_ctrl.add_axes([0.30, 0.86 - i * 0.095, 0.60, 0.045])
        s = Slider(
            ax_s, f"{name}", limits_low_deg[i], limits_high_deg[i],
            valinit=init_deg[i], valfmt="%.1f°",
        )
        sliders.append(s)

    def _on_slide(_val):
        for i, s in enumerate(sliders):
            target_deg[i] = s.val
    for s in sliders:
        s.on_changed(_on_slide)

    ax_reset = fig_ctrl.add_axes([0.30, 0.05, 0.25, 0.06])
    btn_reset = Button(ax_reset, "复位到初始角")

    state = {"policy_on": False, "do_env_reset": False}

    def _on_reset(_evt):
        for i, s in enumerate(sliders):
            s.set_val(init_deg[i])
        state["do_env_reset"] = True
    btn_reset.on_clicked(_on_reset)

    ax_chk = fig_ctrl.add_axes([0.62, 0.04, 0.28, 0.08])
    chk = CheckButtons(ax_chk, ["policy 接管"], [False])

    def _on_chk(_label):
        state["policy_on"] = chk.get_status()[0]
        if state["policy_on"] and policy is None:
            print("[Manual] 未加载 checkpoint，无法启用 policy 接管。")
            chk.set_active(0)  # 反勾
            state["policy_on"] = False
    chk.on_clicked(_on_chk)

    # 监控窗口：奖励曲线 / 接触力 / 距离·姿态 / 状态灯
    rm = base_env.reward_manager
    term_names = rm.active_terms
    HISTORY = 300

    fig_mon, (axr, axc, axd) = plt.subplots(3, 1, figsize=(11, 8), sharex=True,
                                            num="Reward / Contact / Reach Monitor")
    fig_mon.subplots_adjust(top=0.90)
    cmap = plt.get_cmap("tab10")
    colors = {n: cmap(i % 10) for i, n in enumerate(term_names)}

    steps_hist = []
    rew_hist = {n: [] for n in term_names}
    total_hist = []
    force_hist = []
    dist_hist = []
    ori_hist = []

    lines_rew = {n: axr.plot([], [], label=n, color=colors[n], lw=1)[0] for n in term_names}
    line_total, = axr.plot([], [], "k-", lw=1.6, label="total")
    axr.set_ylabel("per-step reward")
    axr.legend(loc="upper left", fontsize=6, ncol=3)
    axr.grid(True, alpha=0.3)

    line_force, = axc.plot([], [], color="crimson", lw=1.3, label="link7 contact (N)")
    axc.axhline(COLLISION_FORCE_THRESH, color="gray", ls="--", lw=1.0,
                label=f"thresh={COLLISION_FORCE_THRESH}N")
    axc.set_ylabel("contact force (N)")
    axc.legend(loc="upper left", fontsize=7)
    axc.grid(True, alpha=0.3)

    line_dist, = axd.plot([], [], color="steelblue", lw=1.2, label="dist (cm)")
    line_ori, = axd.plot([], [], color="darkorange", lw=1.2, label="ori (deg)")
    axd.axhline(SUCCESS_DIST_CM, color="steelblue", ls=":", lw=0.9)
    axd.axhline(SUCCESS_ORI_DEG, color="darkorange", ls=":", lw=0.9)
    axd.set_ylabel("dist / ori")
    axd.set_xlabel("step")
    axd.legend(loc="upper left", fontsize=7)
    axd.grid(True, alpha=0.3)

    # ── 4. 主循环 ────────────────────────────────────────────────────────────
    obs, _ = env.reset()
    g_step = 0

    print("[Manual] 运行中。拖动 Joint Control 窗口里的滑块控制关节。")
    print("[Manual] 碰撞/成功只点亮状态灯与标题，不会终止 episode。")

    try:
        while simulation_app.is_running():
            # 手动触发的 env.reset（复位按钮）——这是唯一允许的 reset。
            if state["do_env_reset"]:
                obs, _ = env.reset()
                state["do_env_reset"] = False
                g_step = 0
                for k in (steps_hist, total_hist, force_hist, dist_hist, ori_hist):
                    k.clear()
                for n in term_names:
                    rew_hist[n].clear()

            # 4.1 生成动作
            if state["policy_on"] and policy is not None:
                with torch.no_grad():
                    actions = policy(obs)
            else:
                # 滑块目标角(度) → 目标角(rad) → 相对当前关节角的归一化 action
                q_cur = robot.data.joint_pos[0, joint_ids]                       # (7,) rad
                q_des = torch.tensor([math.radians(v) for v in target_deg],
                                     device=device, dtype=q_cur.dtype)           # (7,) rad
                delta = q_des - q_cur
                act = torch.clamp(delta / max(scale_rad, 1e-8), -1.0, 1.0)       # (7,) in [-1,1]
                actions = act.unsqueeze(0)

            # 4.2 step（termination 已清空，永不 reset）
            obs, _, dones, _ = env.step(actions)

            # 4.3 读取奖励 / 接触力 / reach 指标
            step_vals = {n: rm._step_reward[0, i].item() for i, n in enumerate(term_names)}
            total = sum(step_vals.values())
            force_n = _contact_force(base_env)
            metrics = _reach_metrics(base_env)

            collided = force_n > COLLISION_FORCE_THRESH
            success = (metrics["dist_cm"] <= SUCCESS_DIST_CM
                       and metrics["ori_deg"] <= SUCCESS_ORI_DEG)

            steps_hist.append(g_step)
            for n in term_names:
                rew_hist[n].append(step_vals[n])
            total_hist.append(total)
            force_hist.append(force_n)
            dist_hist.append(metrics["dist_cm"])
            ori_hist.append(metrics["ori_deg"])

            # 4.4 刷新曲线（每 3 步刷一次，省开销）
            if g_step % 3 == 0:
                xs = steps_hist[-HISTORY:]
                for n in term_names:
                    lines_rew[n].set_data(xs, rew_hist[n][-HISTORY:])
                line_total.set_data(xs, total_hist[-HISTORY:])
                line_force.set_data(xs, force_hist[-HISTORY:])
                line_dist.set_data(xs, dist_hist[-HISTORY:])
                line_ori.set_data(xs, ori_hist[-HISTORY:])
                for ax in (axr, axc, axd):
                    ax.relim()
                    ax.autoscale_view()

                col_txt = "■ COLLISION" if collided else "□ collision"
                suc_txt = "■ SUCCESS" if success else "□ success"
                mode_txt = "POLICY" if state["policy_on"] else "MANUAL"
                fig_mon.suptitle(
                    f"[{mode_txt}] step={g_step} | total_rew={total:8.2f} | "
                    f"force={force_n:6.3f}N  {col_txt} | "
                    f"dist={metrics['dist_cm']:5.2f}cm ori={metrics['ori_deg']:5.2f}°  {suc_txt}",
                    fontsize=10,
                    color=("red" if collided else ("green" if success else "black")),
                )
                fig_mon.canvas.draw_idle()
                fig_ctrl.canvas.draw_idle()
                plt.pause(0.001)

            g_step += 1

    except KeyboardInterrupt:
        print("\n[Manual] 用户中断。")
    finally:
        env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"\nError: {exc}")
        import traceback
        traceback.print_exc()
    finally:
        simulation_app.close()
