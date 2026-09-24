#!/usr/bin/env python3
"""xArm7 Point Bridge —— stage2 遮挡实验训练 / 回放入口。

在 stage2 完整相机管线（实例分割掩码 + 深度反投影 + σ=1cm 噪声）之上，对反投影到
基座系的物体点云按**工件局部系坐标**挖掉集中的一块（半空间 / 球），测点云表征在
工件被部分遮挡时还能不能学。观测维度仍 217、网络结构不变，因此可从 stage2 baseline
checkpoint 热启动（--checkpoint <path> --resume）。

与 train_pick_pointcloud.py 的关系：本脚本是它的**副本**，只做两处改动 ——
（1）固定 stage=2；（2）加遮挡开关。不修改原脚本，二者可并存。

遮挡方式（severity 语义见 configs/occlusion.py）：
    --occlusion_mode      none | halfspace | sphere | random_sphere | random_box
    --occlusion_severity  0.0..1.0（0=不遮；random_* 下每 episode 位置/朝向随机）
    --occlusion_axis      x|-x|y|-y|z|-z   （halfspace，切哪一侧，工件局部系）
    --occlusion_center    cx,cy,cz         （sphere，局部归一化分数）

交互项：遮挡方式 / 程度 / --num_envs / --seed / --num_mini_batches / --num_steps_per_env / --log_root
全部走命令行参数。
"""

import argparse
import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))
sys.path.insert(0, _PROJECT_DIR)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="stage2 遮挡实验：actor 吃 217 维点云（物体 64x3 + 夹爪 6x3 + 关节角 7），critic 吃 35 维特权状态"
)
parser.add_argument("--num_envs", type=int, default=64)
parser.add_argument("--play", action="store_true")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--load_run", type=str, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument(
    "--num_mini_batches",
    type=int,
    default=None,
    help=(
        "PPO minibatch 数，默认 None=沿用配置文件（4）。"
        "保持 minibatch≈6000 时取 num_envs*24/6000，例如 1024→4 / 2500→10 / 5000→20。"
    ),
)
parser.add_argument(
    "--num_steps_per_env",
    type=int,
    default=None,
    help=(
        "每环境每迭代 rollout 步数，默认 None=沿用配置文件（24）。"
        "内存吃紧、envs 加不上去时用它拉大 batch："
        "minibatch = num_envs*num_steps_per_env/num_mini_batches。"
        "例如 128 envs + 96 步 + 2 minibatch = 6144。"
    ),
)
parser.add_argument(
    "--occlusion_mode",
    type=str,
    default="none",
    choices=("none", "halfspace", "sphere", "random_sphere", "random_box"),
    help=(
        "遮挡几何：none=不遮，halfspace=切掉工件一侧，sphere=挖掉一块球，"
        "random_sphere=圆形随机位置，random_box=相机侧表面随机盒心+随机朝向，"
        "按最近邻比例挖连通阴影（每 episode 重采）。"
    ),
)
parser.add_argument(
    "--occlusion_severity",
    type=float,
    default=0.0,
    help=(
        "遮挡程度 0.0..1.0（halfspace=切除比例，sphere/random_sphere=球半径占外接球比例，"
        "random_box=删除最近邻候选点比例）。"
    ),
)
parser.add_argument(
    "--occlusion_axis",
    type=str,
    default="x",
    help="halfspace 用：切哪一侧（x|-x|y|-y|z|-z，工件局部系）。",
)
parser.add_argument(
    "--occlusion_center",
    type=str,
    default="0.5,0.5,0.5",
    help="sphere 用：球心在工件局部包围盒内的归一化分数，逗号分隔 'cx,cy,cz'。",
)
parser.add_argument(
    "--no_joint_vel",
    action="store_true",
    help="去掉 critic 的 7 维关节速度（35 维 → 28 维）。actor 本来就不含速度。",
)
parser.add_argument(
    "--enable_success_termination",
    action="store_true",
    help="启用 reach_success 提前 done；默认关闭（与 camfix 训练一致）。",
)
parser.add_argument(
    "--grasp_offset_cm",
    type=float,
    default=3.0,
    help="抓取目标点抬到工件正上方多少厘米（世界系向上）。默认 3.0。",
)
parser.add_argument(
    "--log_root",
    type=str,
    default=None,
    help="日志根目录；默认 <项目>/logs。要存数据盘就传这个。",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from configs.xarm7_pick_pointcloud_env_cfg import (
    XArm7PickPointCloudEnvCfg,
    XArm7PickPointCloudPlayEnvCfg,
    disable_point_cloud_noise,
    set_critic_joint_vel_enabled,
    set_grasp_target_offset_cm,
    set_observation_stage,
    set_occlusion_stage2,
    set_success_termination_enabled,
)
from configs.occlusion import parse_occlusion_center
from configs.point_bridge_pointcloud import M_OBJ, N_ROBOT, NUM_POINTS, POINT_DIM
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from model.pointnet_actor_critic import register_with_rsl_rl


def _build_env_cfg():
    cfg = XArm7PickPointCloudPlayEnvCfg() if args_cli.play else XArm7PickPointCloudEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        cfg.seed = args_cli.seed

    # 固定 stage 2：完整相机管线（深度 + 实例分割，无 RGB）。回放保留桌面/光照。
    set_observation_stage(cfg, 2, keep_appearance=args_cli.play)
    # stage 2 之上注入遮挡（必须在 set_observation_stage 之后 —— 它替换观测函数）。
    set_occlusion_stage2(
        cfg,
        mode=args_cli.occlusion_mode,
        severity=args_cli.occlusion_severity,
        axis=args_cli.occlusion_axis,
        center=parse_occlusion_center(args_cli.occlusion_center),
    )
    if args_cli.no_joint_vel:
        set_critic_joint_vel_enabled(cfg, False)
    if args_cli.play:
        disable_point_cloud_noise(cfg)
    set_success_termination_enabled(cfg, args_cli.enable_success_termination)
    set_grasp_target_offset_cm(cfg, args_cli.grasp_offset_cm)
    return cfg


def main():
    if not (0.0 <= args_cli.occlusion_severity <= 1.0):
        raise SystemExit(
            f"--occlusion_severity 必须在 [0,1]，got {args_cli.occlusion_severity}"
        )
    _occ_active = args_cli.occlusion_mode != "none" and args_cli.occlusion_severity > 0.0

    env_cfg = _build_env_cfg()

    print("=" * 78)
    print("[Stage] 2（完整相机管线）+ 工件坐标遮挡")
    print(
        f"[Obs]   物体点 {M_OBJ}x{POINT_DIM} + 夹爪点 {N_ROBOT}x{POINT_DIM} "
        f"= {NUM_POINTS * POINT_DIM} 维点云 + 关节角 7 = 217 维"
    )
    print(
        f"[Critic] privileged 状态 {'28' if args_cli.no_joint_vel else '35'} 维"
        f"{'（去关节速度）' if args_cli.no_joint_vel else '（含关节速度）'}"
    )
    if _occ_active:
        if args_cli.occlusion_mode == "halfspace":
            print(
                f"[Occ]   halfspace  axis={args_cli.occlusion_axis}  "
                f"severity={args_cli.occlusion_severity:.2f}（切除该侧比例）"
            )
        elif args_cli.occlusion_mode == "sphere":
            print(
                f"[Occ]   sphere  center={args_cli.occlusion_center}  "
                f"severity={args_cli.occlusion_severity:.2f}（球半径/外接球）"
            )
        elif args_cli.occlusion_mode == "random_sphere":
            print(
                f"[Occ]   random_sphere  severity={args_cli.occlusion_severity:.2f}"
                f"（圆形，球心每 episode 随机）"
            )
        else:
            print(
                f"[Occ]   random_box  severity={args_cli.occlusion_severity:.2f}"
                f"（相机侧盒心+随机朝向，最近邻删除该比例候选点）"
            )
    else:
        print("[Occ]   无遮挡（== stage2 baseline，作对照）")
    print(
        f"[Reward] success termination: "
        f"{'enabled' if args_cli.enable_success_termination else 'disabled (no-success default)'}"
    )
    if args_cli.grasp_offset_cm != 0.0:
        print(f"[Target] 抓取目标点 = 工件正上方 {args_cli.grasp_offset_cm:.1f} cm")
    else:
        print("[Target] 抓取目标点 = 工件根原点（旧行为）")
    print("=" * 78)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    from datetime import datetime

    from configs.agents.rsl_rl_ppo_cfg import XArm7PickPointCloudPPORunnerCfg

    agent_cfg = XArm7PickPointCloudPPORunnerCfg()
    agent_cfg.policy.num_object_points = M_OBJ
    agent_cfg.policy.num_robot_points = N_ROBOT
    agent_cfg.policy.point_dim = POINT_DIM
    agent_cfg.policy.num_wrist_points = 0

    # 日志目录名：把遮挡方式/程度编进去，多条曲线可进同一张 TensorBoard 对照。
    if _occ_active:
        if args_cli.occlusion_mode == "halfspace":
            occ_tag = (
                f"halfspace{args_cli.occlusion_axis}"
                f"_{int(round(args_cli.occlusion_severity * 100))}"
            )
        else:
            occ_tag = (
                f"{args_cli.occlusion_mode}"
                f"_{int(round(args_cli.occlusion_severity * 100))}"
            )
    else:
        occ_tag = "none"
    log_name = f"xarm7_pick_pointcloud_stage2_occ_{occ_tag}"
    if args_cli.no_joint_vel:
        log_name += "_no_vel"

    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.num_mini_batches is not None:
        agent_cfg.algorithm.num_mini_batches = args_cli.num_mini_batches
    if args_cli.num_steps_per_env is not None:
        agent_cfg.num_steps_per_env = args_cli.num_steps_per_env

    log_root = args_cli.log_root if args_cli.log_root else os.path.join(_PROJECT_DIR, "logs")
    log_dir = os.path.join(
        log_root,
        log_name,
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    os.makedirs(log_dir, exist_ok=True)

    import shutil

    for src in (
        os.path.join(_PROJECT_DIR, "configs", "xarm7_pick_pointcloud_env_cfg.py"),
        os.path.join(_PROJECT_DIR, "configs", "point_bridge_pointcloud.py"),
        os.path.join(_PROJECT_DIR, "configs", "occlusion.py"),
        os.path.join(_PROJECT_DIR, "model", "pointnet_actor_critic.py"),
        os.path.join(_CURRENT_DIR, "train_pick_pointcloud_occluded.py"),
    ):
        if os.path.isfile(src):
            shutil.copy(src, os.path.join(log_dir, os.path.basename(src)))

    # OnPolicyRunner 用 eval(class_name) 解析策略类，必须在构造 runner 之前注入。
    register_with_rsl_rl()

    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=env.device)

    if args_cli.play:
        if args_cli.checkpoint is not None:
            runner.load(args_cli.checkpoint)
        else:
            print("Warning: No checkpoint specified. Using random policy.")
        policy = runner.get_inference_policy(device=env.device)

        obs, _ = env.reset()
        while simulation_app.is_running():
            with torch.no_grad():
                actions = policy(obs)
            obs, _, dones, _ = env.step(actions)
            if dones[0]:
                obs, _ = env.reset()
    else:
        if args_cli.resume:
            if args_cli.checkpoint is not None:
                runner.load(args_cli.checkpoint)
            elif args_cli.load_run is not None:
                runner.load(args_cli.load_run)
        runner.learn(
            num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
        )

    env.close()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close()
