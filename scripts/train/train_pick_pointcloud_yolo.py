#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""xArm7 Point Bridge 点表征 PPO —— **掩码来自 YOLO-seg**（stage 3）。

与 ``train_pick_pointcloud.py`` 的唯一区别是**掩码怎么来**：

    train_pick_pointcloud.py   instance_id_segmentation_fast（Isaac 逐像素完美分割）
    本脚本                      RGB ─► YOLO-seg ─► mask

反投影、FPS、噪声、零阶保持、网络、PPO 超参全部原样复用 —— 规格 §1.1 只允许存在
一份 ``mask_depth_to_pointcloud``，仿真和真机必须调同一个函数。

**为什么值得慢十倍去换这个掩码**

真机侧掩码只能是 YOLO 出的。训练若全程吃 Isaac 的完美分割，策略见到的掩码分布和
部署时差一截：YOLO 边缘会胖一圈、遮挡处会缺一块，而且这是**系统性偏差不是噪声**
（规格 §11.2）—— sigma=1cm 的零均值高斯补不回非零均值的偏差。让训练直接吃 YOLO
的掩码，这段 gap 从根上不存在。

**相机这一路：分割整路已摘掉**

stage 3 的相机是 ``distance_to_image_plane`` + ``rgb``，**没有实例分割**。YOLO 只
要 RGB，分割图没有任何消费者了。与 stage 2 的"深度+分割"是两路换两路，相机显存
持平。

**开销：这是本脚本最需要先知道的事**

RTX 5090 实测（20260815.pt，imgsz=640）::

    B=8   43.5ms   5.43ms/env
    B=16  70.1ms   4.38ms/env
    B=64 273.8ms   4.28ms/env

**几乎不随 batch 摊薄** —— 瓶颈是 ultralytics 逐图的 Python 后处理，不是网络。而
仿真一步只要 20ms。所以：

    · 默认 num_envs=16（不是 64）。开 64 每步 274ms，一次 PPO 迭代要几分钟。
    · YOLO 另占约 2GB 显存，和渲染器抢。显存紧时先降 num_envs。
    · **强烈建议从 stage 2 的 checkpoint 续训**，而不是从头训。stage 2 已经把
      "点云→动作"学会了，这里只需要适应掩码分布的变化，几百次迭代通常就够。

**环境要求：已装好，不需要动 isaaclab 环境**

ultralytics 装在独立目录 ``/home/gxai/Desktop/CZR/.yolo_deps``（21MB，4 个纯
Python 包），由 ``configs/yolo_mask_source.py`` 自动挂 ``sys.path``。isaaclab 的
site-packages **一个文件都没改** —— 删掉那个目录就完全复原。

若要重装::

    ~/miniconda3/envs/isaaclab/bin/python -m pip install --no-deps \\
        --target /home/gxai/Desktop/CZR/.yolo_deps \\
        ultralytics==8.4.69 polars nvidia-ml-py ultralytics-thop

用法::

    # 从 stage 2 的 checkpoint 续训（推荐）
    ~/IsaacLab/isaaclab.sh -p scripts/train/train_pick_pointcloud_yolo.py \\
        --num_envs 16 --checkpoint logs/xarm7_pick_pointcloud_stage2/<run>/model_xxx.pt \\
        --resume --max_iterations 500 --headless

    # 回放看效果
    ~/IsaacLab/isaaclab.sh -p scripts/train/train_pick_pointcloud_yolo.py \\
        --play --num_envs 1 --checkpoint logs/xarm7_pick_pointcloud_stage3/<run>/model_xxx.pt
"""

import argparse
import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))
sys.path.insert(0, _PROJECT_DIR)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="xArm7 Point Bridge 点表征 PPO —— 掩码来自 YOLO-seg（stage 3）"
)
# 默认 16 而不是 64：YOLO 每步约 4.3ms/env 且不随 batch 摊薄，64 env 每步 274ms，
# 而仿真本身只要 20ms。开大了绝大部分时间在等 YOLO。
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--play", action="store_true")
parser.add_argument("--checkpoint", type=str, default=None)
parser.add_argument("--resume", action="store_true")
parser.add_argument("--load_run", type=str, default=None)
parser.add_argument("--max_iterations", type=int, default=None)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument(
    "--yolo_weights",
    type=str,
    default=None,
    help="YOLO-seg 权重。默认用 configs/yolo_mask_source.py 里的 20260815.pt。",
)
parser.add_argument(
    "--yolo_conf",
    type=float,
    default=None,
    help="检出置信度门限，默认 0.25。调高会增加漏检（走零阶保持），调低会增加误检。",
)
parser.add_argument(
    "--yolo_step",
    type=int,
    default=None,
    help=(
        "给出这个参数即切到 **stage 4**：每 N 步才跑一次 YOLO，中间步复用缓存掩码"
        "配当前深度帧（缓存掩码而非深度，观测每步仍在更新）。N=5 时 YOLO 均摊开销"
        "降到 1/5。不给这个参数则跑 stage 3（每步都跑 YOLO）。"
    ),
)
parser.add_argument(
    "--no_joint_vel",
    action="store_true",
    help="去掉 critic 的 7 维关节速度（35 维 → 28 维）。actor 本来就不含速度。",
)
parser.add_argument(
    "--enable_success_termination",
    action="store_true",
    help="启用 reach_success 提前 done；默认关闭，与既有训练保持一致。",
)
parser.add_argument(
    "--grasp_offset_cm",
    type=float,
    default=3.0,
    help=(
        "抓取目标点抬到工件正上方多少厘米（世界系向上，随工件 yaw 一起转）。"
        "默认 3.0 = 绿球位置。给 0 则回到工件根原点（旧行为）。"
    ),
)
parser.add_argument(
    "--log_root",
    type=str,
    default=None,
    help="日志根目录；默认 <项目>/logs。要存数据盘就传这个（对齐 stage2 的惯例）。",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# 相机是观测来源，且程序化桌面用 PreviewSurface(MDL)，两者都要求渲染型 Kit experience
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from configs.xarm7_pick_pointcloud_env_cfg import (
    XArm7PickPointCloudEnvCfg,
    XArm7PickPointCloudPlayEnvCfg,
    set_critic_joint_vel_enabled,
    set_grasp_target_offset_cm,
    set_observation_stage,
    set_success_termination_enabled,
)
from configs.point_bridge_pointcloud import M_OBJ, N_ROBOT, NUM_POINTS, POINT_DIM
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from model.pointnet_actor_critic import register_with_rsl_rl

# 给了 --yolo_step 就走 stage 4（缓存掩码），否则 stage 3（每步跑 YOLO）。
_STAGE = 3 if args_cli.yolo_step is None else 4


def _build_env_cfg():
    cfg = (
        XArm7PickPointCloudPlayEnvCfg() if args_cli.play else XArm7PickPointCloudEnvCfg()
    )
    cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        cfg.seed = args_cli.seed

    # stage 3/4：掩码换成 YOLO，相机改为深度 + RGB（实例分割整路摘掉）。
    # keep_appearance 对这两个 stage 无效（那只作用于 stage -1/-2），桌面贴图与光照 DR
    # 本来就保留着 —— 对这条路线尤其重要，YOLO 吃的就是这张 RGB。
    set_observation_stage(cfg, _STAGE, keep_appearance=args_cli.play)

    # 命令行覆盖 YOLO 参数。必须在 set_observation_stage **之后** —— 它会重设
    # term.params，放前面会被整个覆盖掉。
    term = cfg.observations.policy.point_cloud
    if args_cli.yolo_weights is not None:
        term.params["yolo_weights"] = args_cli.yolo_weights
    if args_cli.yolo_conf is not None:
        term.params["yolo_conf"] = args_cli.yolo_conf
    if args_cli.yolo_step is not None:
        if args_cli.yolo_step < 1:
            raise ValueError(f"--yolo_step 必须 >= 1；给的是 {args_cli.yolo_step}")
        term.params["yolo_step"] = args_cli.yolo_step

    if args_cli.no_joint_vel:
        set_critic_joint_vel_enabled(cfg, False)
    set_success_termination_enabled(cfg, args_cli.enable_success_termination)
    # 必须在 set_success_termination_enabled **之后** —— 那个 setter 会新建一个
    # 不带偏移的 DoneTerm，放前面会被盖掉，导致 success 判的点和奖励的点不一致。
    set_grasp_target_offset_cm(cfg, args_cli.grasp_offset_cm)
    return cfg


def main():
    env_cfg = _build_env_cfg()
    term = env_cfg.observations.policy.point_cloud

    print("=" * 78)
    if _STAGE == 4:
        print(f"[Stage] 4  (完整相机管线，**掩码来自 YOLO-seg，每 "
              f"{term.params['yolo_step']} 步跑一次**)")
    else:
        print(f"[Stage] {_STAGE}  (完整相机管线，**掩码来自 YOLO-seg**)")
    print(f"[Obs]   物体点 {M_OBJ}x{POINT_DIM} + 夹爪点 {N_ROBOT}x{POINT_DIM} "
          f"= {NUM_POINTS * POINT_DIM} 维点云（双 token，无类型通道）")
    if args_cli.no_joint_vel:
        print("[Critic] privileged 状态 28 维 —— 已去掉关节速度")
    else:
        print("[Critic] privileged 状态 35 维（含关节速度）")
    print(f"[YOLO]  weights={term.params['yolo_weights']}")
    print(f"        conf={term.params['yolo_conf']}   "
          f"相机 data_types={env_cfg.scene.camera_fixed.data_types}")
    if _STAGE == 4:
        n = term.params["yolo_step"]
        print(f"        掩码缓存 {n} 步（深度每步都是新的，缓存的只是掩码），"
              f"YOLO 均摊开销 ≈ 1/{n}")
    print(f"[Env]   num_envs={args_cli.num_envs}   "
          f"（YOLO 约 4.3ms/env 且不随 batch 摊薄，开大了主要在等它）")
    print(f"[Reward] success termination: "
          f"{'enabled' if args_cli.enable_success_termination else 'disabled'}")
    if args_cli.grasp_offset_cm != 0.0:
        print(f"[Target] 抓取目标点 = 工件正上方 {args_cli.grasp_offset_cm:.1f} cm"
              f"（工件局部 z={-args_cli.grasp_offset_cm / 100.0:+.3f} m，随 yaw 转）")
    else:
        print("[Target] 抓取目标点 = 工件根原点（旧行为）")
    if not args_cli.resume and not args_cli.play:
        print("[提示] 没有 --resume。建议从 stage 2 的 checkpoint 续训 —— stage 2 已学会"
              "『点云→动作』，这里只需适应掩码分布的变化。")
        if args_cli.grasp_offset_cm != 0.0:
            print("       注意：既有 stage 2 checkpoint 是瞄工件根原点训的，目标点变了"
                  "之后续训相当于换任务，前期 reward 会先掉一段再爬。")
    print("=" * 78)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    from datetime import datetime

    from configs.agents.rsl_rl_ppo_cfg import XArm7PickPointCloudPPORunnerCfg

    agent_cfg = XArm7PickPointCloudPPORunnerCfg()
    # 网络侧的点云形状必须与环境侧一致，从同一份常量推导而不是各写一遍
    agent_cfg.policy.num_object_points = M_OBJ
    agent_cfg.policy.num_robot_points = N_ROBOT
    agent_cfg.policy.point_dim = POINT_DIM

    log_name = f"xarm7_pick_pointcloud_stage{_STAGE}"
    if args_cli.no_joint_vel:
        log_name += "_no_vel"

    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    log_root = (
        args_cli.log_root if args_cli.log_root else os.path.join(_PROJECT_DIR, "logs")
    )
    log_dir = os.path.join(log_root, log_name, datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    os.makedirs(log_dir, exist_ok=True)

    import shutil

    # 连 yolo_mask_source.py 一起存档：掩码怎么来的是这条路线的核心变量，
    # 事后复现时缺了它就说不清当时用的是哪个权重、哪个 conf。
    for src in (
        os.path.join(_PROJECT_DIR, "configs", "xarm7_pick_pointcloud_env_cfg.py"),
        os.path.join(_PROJECT_DIR, "configs", "point_bridge_pointcloud.py"),
        os.path.join(_PROJECT_DIR, "configs", "yolo_mask_source.py"),
        os.path.join(_PROJECT_DIR, "model", "pointnet_actor_critic.py"),
    ):
        shutil.copy(src, os.path.join(log_dir, os.path.basename(src)))

    # OnPolicyRunner 用 eval(class_name) 在它自己的模块 globals 里解析策略类，
    # 必须在构造 runner 之前注入。
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
