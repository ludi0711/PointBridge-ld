#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""xArm7 pick-pose 独立推理入口。"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))
if _PROJECT_DIR not in sys.path:
    sys.path.insert(0, _PROJECT_DIR)

from tools.logs.pick_pose_csv import PickPoseCsvRow, open_pick_pose_csv

# Do not import Isaac Lab modules before AppLauncher.
from isaaclab.app import AppLauncher


LOG_NAME = "xarm7_pick_pose"
ENV_CFG_FILE = os.path.join(_PROJECT_DIR, "configs", "xarm7_pick_pose_env_cfg.py")
TEST_LOG_DIR = os.path.join(_PROJECT_DIR, "test_log")
MDP_OBS_DIM = 21
ORI_Q_OFFSET_WXYZ = "0.0,0.0,0.0,1.0"
GRASP_SYMMETRY_QUATS_WXYZ = (
    "1.0,0.0,0.0,0.0",
    "0.0,0.0,0.0,1.0",
)


parser = argparse.ArgumentParser(description="xArm7 pick pose eval runner")
parser.add_argument("--checkpoint", type=str, required=True, help="Policy checkpoint path.")
parser.add_argument("--num_envs", type=int, default=1, help="Number of envs. The CSV logs env 0.")
parser.add_argument("--seed", type=int, default=None, help="Random seed.")
parser.add_argument(
    "--max_steps",
    type=int,
    default=-1,
    help="Max steps per episode/trajectory, -1 means use environment terminations only.",
)
parser.add_argument("--max_episodes", type=int, default=1, help="Number of episodes to run, -1 means run forever.")
parser.add_argument(
    "--max_total_steps",
    type=int,
    default=-1,
    help="Optional global step cap across episodes, -1 disables it.",
)
parser.add_argument("--print_every", type=int, default=10, help="Print every N global steps, <=0 disables periodic prints.")
parser.add_argument("--test_log_dir", type=str, default=TEST_LOG_DIR, help="CSV output root.")
parser.add_argument("--action_smooth_alpha", type=float, default=1.0, help="1.0 keeps pure policy actions.")
parser.add_argument(
    "--object_pose_base",
    type=str,
    default=None,
    help="Manual object pose in robot base-link frame: x,y,z,qw,qx,qy,qz.",
)
parser.add_argument(
    "--object_pos_base",
    type=str,
    default=None,
    help="Manual object position in robot base-link frame: x,y,z.",
)
parser.add_argument(
    "--object_quat_base",
    type=str,
    default=None,
    help="Manual object quaternion in robot base-link frame: qw,qx,qy,qz.",
)
parser.add_argument(
    "--object_rpy_base_deg",
    type=str,
    default=None,
    help="Manual object Euler angle in robot base-link frame: roll,pitch,yaw in deg.",
)
parser.add_argument(
    "--keep_reset_randomization",
    action="store_true",
    help="Keep reset table/object randomization even when a manual object pose is supplied.",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# Low-dimensional state policy does not need cameras.
args_cli.enable_cameras = False

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

# Isaac Lab / RSL-RL imports must happen after AppLauncher.
import math
import shutil
import traceback
from datetime import datetime

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from configs.agents.rsl_rl_ppo_cfg import PickPosePPORunnerCfg
from configs.xarm7_pick_pose_env_cfg import PickPoseEnvCfg_PLAY, _ARM_ACTION_SCALE, _ARM_CLIP_RAD
from tools.utils.pick_pose_eval_utils import (
    ManualObjectPose,
    as_first_env_numpy,
    extract_timeout,
    get_cmd_delta_deg,
    get_ee_target_pose_w,
    manual_object_pose_from_args,
    object_pose_in_base,
    orientation_error_deg,
    parse_quat_wxyz,
    pose_in_robot_base,
    read_terminal_cache,
    read_terminal_done_reason,
    read_terminal_joint_cache,
    read_terminal_pose_cache,
    reset_eval_env,
)


def _resolve_checkpoint(path_str: str | None) -> str:
    if not path_str:
        raise ValueError("--checkpoint is required for eval.")

    candidates = [
        Path(path_str).expanduser(),
        Path(_PROJECT_DIR) / path_str,
        Path("/home/gxai/IsaacLab") / path_str,
        Path(_PROJECT_DIR).parent.parent / path_str,
    ]
    for path in candidates:
        if path.is_file():
            return str(path)

    tried = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"Cannot find checkpoint: {path_str}\nTried:\n{tried}")


def _make_env_cfg(manual_pose: ManualObjectPose | None):
    env_cfg = PickPoseEnvCfg_PLAY()
    env_cfg.scene.num_envs = int(args_cli.num_envs)
    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    if manual_pose is not None and not args_cli.keep_reset_randomization:
        reset_event = getattr(env_cfg.events, "reset_table_and_object", None)
        params = getattr(reset_event, "params", None)
        if isinstance(params, dict):
            params["x_range"] = (0.0, 0.0)
            params["y_range"] = (0.0, 0.0)
            params["yaw_range"] = (0.0, 0.0)
            params["height_range"] = 0.0

    return env_cfg


def _print_header(log_dir: str, env_cfg, manual_pose: ManualObjectPose | None):
    print("\n========== pick_pose_eval ==========")
    print(f"[Action] scale = {math.degrees(_ARM_ACTION_SCALE):.4f} deg")
    print(f"[Action] clip  = +/-{math.degrees(_ARM_CLIP_RAD):.4f} deg")
    print(f"[Log]    {log_dir}")
    print(f"[Env]    num_envs = {env_cfg.scene.num_envs}")
    print(f"[Config] {ENV_CFG_FILE}")
    print(
        f"[Eval]   max_steps_per_episode={args_cli.max_steps}, "
        f"max_episodes={args_cli.max_episodes}, "
        f"max_total_steps={args_cli.max_total_steps}, "
        f"smooth_alpha={args_cli.action_smooth_alpha}"
    )
    if manual_pose is not None:
        print(f"[ObjectBase] pos_m={manual_pose.pos_base_m.round(6).tolist()}")
        print(f"[ObjectBase] quat_wxyz={manual_pose.quat_base_wxyz.round(8).tolist()}")
    print("====================================\n")


def _run_eval(env, runner, checkpoint_path: str, manual_pose: ManualObjectPose | None):
    print(f"[Checkpoint] loading: {checkpoint_path}")
    runner.load(checkpoint_path)
    policy = runner.get_inference_policy(device=env.device)

    base_env = env.unwrapped
    obj = base_env.scene["object"]
    ee_frame = base_env.scene["ee_frame"]
    robot = base_env.scene["robot"]

    joint_ids, joint_names = robot.find_joints(["joint[1-7]"])
    joint_ids = list(joint_ids)
    if len(joint_ids) != 7:
        raise RuntimeError(f"Expected 7 arm joints, got {len(joint_ids)}: {joint_names}")
    print(f"[Eval] logging arm joints: {joint_names}")

    q_offset_wxyz = parse_quat_wxyz(ORI_Q_OFFSET_WXYZ, env.device)
    grasp_symmetry_quats_wxyz = torch.stack(
        [parse_quat_wxyz(text, env.device) for text in GRASP_SYMMETRY_QUATS_WXYZ],
        dim=0,
    )

    obs = reset_eval_env(env, base_env, robot, obj, manual_pose)
    if manual_pose is not None:
        pos_base, quat_base = object_pose_in_base(robot, obj)
        print(f"[ObjectBaseApplied] pos_m={pos_base.round(6).tolist()}")
        print(f"[ObjectBaseApplied] quat_wxyz={quat_base.round(8).tolist()}")

    csv_log = open_pick_pose_csv(
        root=args_cli.test_log_dir,
        log_name=LOG_NAME,
        checkpoint_path=checkpoint_path,
        file_prefix="eval_pick_pose",
        obs_dim=MDP_OBS_DIM,
        fallback_prefix="unknown_policy",
        log_dir_label="EvalLogDir",
        log_file_label="EvalLog",
    )
    smooth_alpha = float(np.clip(args_cli.action_smooth_alpha, 0.0, 1.0))

    ep = 0
    step = 0
    g_step = 0
    last_smoothed_actions = torch.zeros(1, 7, device=env.device, dtype=torch.float32)

    print("[Eval] start | press Ctrl+C to exit")
    try:
        while simulation_app.is_running():
            if args_cli.max_total_steps >= 0 and g_step >= args_cli.max_total_steps:
                print(f"[Eval] reached --max_total_steps={args_cli.max_total_steps}, exit.")
                break
            if args_cli.max_steps >= 0 and step >= args_cli.max_steps:
                print(f"[EpisodeStepLimit] ep={ep} reached --max_steps={args_cli.max_steps}.")
                ep += 1
                if args_cli.max_episodes >= 0 and ep >= args_cli.max_episodes:
                    print(f"[Eval] reached --max_episodes={args_cli.max_episodes}, exit.")
                    break
                obs = reset_eval_env(env, base_env, robot, obj, manual_pose)
                step = 0
                last_smoothed_actions.zero_()
                continue

            with torch.no_grad():
                obs_mdp_np = as_first_env_numpy(obs, MDP_OBS_DIM, "policy obs")
                raw_actions = policy(obs)
                raw_actions_mdp_np = as_first_env_numpy(raw_actions, 7, "raw action")

            actions = smooth_alpha * raw_actions + (1.0 - smooth_alpha) * last_smoothed_actions
            last_smoothed_actions = actions.detach().clone()

            q_before = robot.data.joint_pos[0, joint_ids].detach().clone()
            obs, rewards, dones, extras = env.step(actions)
            q_after = robot.data.joint_pos[0, joint_ids].detach().clone()

            done_bool = bool(dones[0].item()) if torch.is_tensor(dones) else bool(dones[0])
            time_outs = extract_timeout(extras, dones)
            timeout_bool = bool(time_outs[0].item())
            terminal_joint_pos = read_terminal_joint_cache(base_env, env_id=0) if done_bool else None
            if terminal_joint_pos is not None:
                q_after = terminal_joint_pos.to(device=q_before.device, dtype=q_before.dtype)

            joint_deg = torch.rad2deg(q_after).cpu().numpy()
            actual_delta_deg = torch.rad2deg(q_after - q_before).cpu().numpy()
            cmd_delta_deg = get_cmd_delta_deg(base_env, actions, _ARM_ACTION_SCALE, _ARM_CLIP_RAD)
            delta_error_deg = actual_delta_deg - cmd_delta_deg
            if done_bool and terminal_joint_pos is None:
                actual_delta_deg = np.full_like(cmd_delta_deg, np.nan)
                delta_error_deg = np.full_like(cmd_delta_deg, np.nan)

            ee_pos_w, ee_quat = get_ee_target_pose_w(ee_frame)
            obj_pos_w = obj.data.root_pos_w[0].detach()
            obj_quat = obj.data.root_quat_w[0].detach().clone()

            # CSV 里的 tcp_* / object_* 统一记录 robot base_link 坐标系下的位姿。
            # tcp_* 使用 ee_frame target/TCP，包含配置里的 0.177 m offset。
            ee_pos, ee_quat_np = pose_in_robot_base(robot, ee_pos_w, ee_quat)
            obj_pos, obj_quat_np = pose_in_robot_base(robot, obj_pos_w, obj_quat)
            dist_cm = float(np.linalg.norm(obj_pos - ee_pos) * 100.0)
            ori_err_deg = orientation_error_deg(
                ee_quat,
                obj_quat,
                q_offset_wxyz,
                grasp_symmetry_quats_wxyz,
            )

            terminal_dist_cm, terminal_ori_err_deg, terminal_success = read_terminal_cache(base_env, env_id=0)
            terminal_pose = read_terminal_pose_cache(base_env, env_id=0) if done_bool else None
            done_reason = read_terminal_done_reason(base_env, env_id=0) if done_bool else "running"
            if done_bool and not done_reason:
                if terminal_success:
                    done_reason = "success"
                elif timeout_bool:
                    done_reason = "timeout"
                else:
                    done_reason = "non_timeout_done"
            if terminal_pose is not None:
                term_ee_pos_w, term_ee_quat, term_obj_pos_w, term_obj_quat = terminal_pose
                ee_pos, ee_quat_np = pose_in_robot_base(robot, term_ee_pos_w, term_ee_quat)
                obj_pos, obj_quat_np = pose_in_robot_base(robot, term_obj_pos_w, term_obj_quat)
                dist_cm = float(np.linalg.norm(obj_pos - ee_pos) * 100.0)
                ori_err_deg = orientation_error_deg(
                    term_ee_quat,
                    term_obj_quat,
                    q_offset_wxyz,
                    grasp_symmetry_quats_wxyz,
                )

            if done_bool and terminal_dist_cm is not None and terminal_ori_err_deg is not None:
                log_dist_cm = terminal_dist_cm
                log_ori_err_deg = terminal_ori_err_deg
                terminal_source = "cached_terminal"
            else:
                log_dist_cm = dist_cm
                log_ori_err_deg = ori_err_deg
                terminal_source = "live_scene"

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
            done_print = done_bool
            if periodic_print or done_print:
                tag = "SUCCESS " if terminal_success else ("DONE " if done_bool else "")
                act_mean = float(np.nanmean(np.abs(actual_delta_deg))) if not np.all(np.isnan(actual_delta_deg)) else float("nan")
                err_max = float(np.nanmax(np.abs(delta_error_deg))) if not np.all(np.isnan(delta_error_deg)) else float("nan")
                print(
                    f"[{tag}ep {ep:03d} step {step:04d} g {g_step:06d}] "
                    f"dist={log_dist_cm:.3f} cm | ori={log_ori_err_deg:.3f} deg | "
                    f"cmd_mean={float(np.mean(np.abs(cmd_delta_deg))):.3f} deg | "
                    f"act_mean={act_mean:.3f} deg | "
                    f"err_max={err_max:.3f} deg | "
                    f"done={done_bool} | reason={done_reason} | timeout={timeout_bool} | "
                    f"terminal_success={terminal_success} | source={terminal_source}"
                )

            step += 1
            g_step += 1

            if done_bool:
                print(
                    f"[EpisodeDone] ep={ep} steps={step} "
                    f"final_dist={log_dist_cm:.3f} cm | final_ori={log_ori_err_deg:.3f} deg | "
                    f"success={terminal_success} | reason={done_reason} | timeout={timeout_bool} | source={terminal_source}"
                )
                ep += 1
                if args_cli.max_episodes >= 0 and ep >= args_cli.max_episodes:
                    print(f"[Eval] reached --max_episodes={args_cli.max_episodes}, exit.")
                    break
                obs = reset_eval_env(env, base_env, robot, obj, manual_pose)
                step = 0
                last_smoothed_actions.zero_()

    finally:
        csv_log.close()


def main():
    manual_pose = manual_object_pose_from_args(args_cli)
    checkpoint_path = _resolve_checkpoint(args_cli.checkpoint)
    env_cfg = _make_env_cfg(manual_pose)

    agent_cfg = PickPosePPORunnerCfg()
    log_dir = os.path.join("logs", LOG_NAME, f"eval_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}")
    os.makedirs(log_dir, exist_ok=True)
    if os.path.isfile(ENV_CFG_FILE):
        shutil.copy(ENV_CFG_FILE, os.path.join(log_dir, "env_cfg.py"))

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=env.device)

    _print_header(log_dir, env_cfg, manual_pose)
    try:
        _run_eval(env, runner, checkpoint_path, manual_pose)
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
