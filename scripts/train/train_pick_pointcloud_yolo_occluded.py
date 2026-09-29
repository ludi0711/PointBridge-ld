#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""xArm7 Point Bridge 点表征 PPO —— **掩码来自 YOLO-seg（stage 3/4）+ random_box 遮挡**。

= ``train_pick_pointcloud_yolo.py``（YOLO 掩码路线，无遮挡）的**副本**，唯一差别是
在反投影后的物体点云上注入 random_box 几何遮挡（M1 缺失型，设计：
``tools/yolo_occlusion_m1_design.md``）：盒心每 episode 从面向相机的表面点随机、
朝向随机 SO(3)，按各向异性椭球度量删掉离盒心最近的 severity 比例候选点。

与 ``train_pick_pointcloud_occluded.py``（stage 2 实例分割 baseline）的关系：它是
"完美分割 + 遮挡"，本脚本是"YOLO 掩码 + 遮挡" —— 测的是掩码本身带误差时，几何
遮挡下还能不能学。反投影、FPS、噪声、零阶保持、网络、PPO 超参全部原样复用。

severity 两种给法：
    --occlusion_severity 0.5              固定删除比例（0 = 对照，与 stage 3/4 逐位一致）
    --occlusion_severity_min 0.2          每 episode 在 [min,max] 内独立采样
    --occlusion_severity_max 0.6          （两个都给才生效，此时固定值被忽略）

遮挡只作用在物体点上，观测维度仍 217、网络结构不变。热启动源是 stage 3/4 的
checkpoint（**不是** stage 2 的 —— 掩码来源不同）。

用法::

    # 从 stage 3/4 的 checkpoint 续训（推荐），固定 severity 0.5
    ~/IsaacLab/isaaclab.sh -p scripts/train/train_pick_pointcloud_yolo_occluded.py \\
        --num_envs 16 --checkpoint logs/xarm7_pick_pointcloud_stage3/<run>/model_xxx.pt \\
        --resume --max_iterations 500 --occlusion_severity 0.5 --headless

    # 每 episode 在 20%..60% 内随机
    ~/IsaacLab/isaaclab.sh -p scripts/train/train_pick_pointcloud_yolo_occluded.py \\
        --num_envs 16 --checkpoint logs/xarm7_pick_pointcloud_stage4/<run>/model_xxx.pt \\
        --resume --yolo_step 5 --occlusion_severity_min 0.2 --occlusion_severity_max 0.6

    # 回放看遮挡效果（录像帧上标红被挖像素需另行打开 _OCCLUSION_VIS_ENABLED）
    ~/IsaacLab/isaaclab.sh -p scripts/train/train_pick_pointcloud_yolo_occluded.py \\
        --play --num_envs 1 --checkpoint logs/xarm7_pick_pointcloud_stage3_occ_rbox_50/<run>/model_xxx.pt \\
        --occlusion_severity 0.5
"""

import argparse
import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))
sys.path.insert(0, _PROJECT_DIR)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="xArm7 Point Bridge 点表征 PPO —— YOLO-seg 掩码（stage 3/4）+ random_box 遮挡"
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
        "配当前深度帧。N=5 时 YOLO 均摊开销降到 1/5。不给这个参数则跑 stage 3"
        "（每步都跑 YOLO）。"
    ),
)
parser.add_argument(
    "--occlusion_severity",
    type=float,
    default=0.0,
    help=(
        "random_box 固定删除比例 0.0..1.0：删掉离盒心最近的该比例候选点。"
        "0 = 对照（与 stage 3/4 逐位一致）。"
    ),
)
parser.add_argument(
    "--occlusion_severity_min",
    type=float,
    default=None,
    help="与 --occlusion_severity_max 成对给出：每 episode 在 [min,max] 内独立采样。",
)
parser.add_argument(
    "--occlusion_severity_max",
    type=float,
    default=None,
    help="与 --occlusion_severity_min 成对给出：每 episode 在 [min,max] 内独立采样。",
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
    set_occlusion_yolo,
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

    if args_cli.yolo_step is not None and args_cli.yolo_step < 1:
        raise ValueError(f"--yolo_step 必须 >= 1；给的是 {args_cli.yolo_step}")

    # severity 校验（main 里打印前还要再查一次固定值；这里只查范围模式成对性）。
    severity_range = None
    if args_cli.occlusion_severity_min is not None or args_cli.occlusion_severity_max is not None:
        if (
            args_cli.occlusion_severity_min is None
            or args_cli.occlusion_severity_max is None
        ):
            raise ValueError(
                "--occlusion_severity_min 与 --occlusion_severity_max 必须成对给出；"
                f"min={args_cli.occlusion_severity_min} max={args_cli.occlusion_severity_max}"
            )
        if not (
            0.0
            <= args_cli.occlusion_severity_min
            <= args_cli.occlusion_severity_max
            <= 1.0
        ):
            raise ValueError(
                "--occlusion_severity_min/max 必须满足 0 <= min <= max <= 1；"
                f"got min={args_cli.occlusion_severity_min} max={args_cli.occlusion_severity_max}"
            )
        severity_range = (
            args_cli.occlusion_severity_min,
            args_cli.occlusion_severity_max,
        )

    # stage 3/4：掩码换成 YOLO，相机改为深度 + RGB（实例分割整路摘掉）。
    # keep_appearance 对这两个 stage 无效（那只作用于 stage -1/-2），桌面贴图与光照 DR
    # 本来就保留着 —— 对这条路线尤其重要，YOLO 吃的就是这张 RGB。
    set_observation_stage(cfg, _STAGE, keep_appearance=args_cli.play)

    # stage 3/4 之上注入 random_box 遮挡（必须在 set_observation_stage **之后** ——
    # 它替换观测函数）。对照（severity=0 且无范围）时不构造谓词、不注册遮挡事件，
    # 与 stage 3/4 逐位一致。
    set_occlusion_yolo(
        cfg,
        severity=args_cli.occlusion_severity,
        severity_range=severity_range,
        yolo_step=args_cli.yolo_step,
    )

    # 命令行覆盖 YOLO 参数。必须在 set_occlusion_yolo **之后** —— 它会重设
    # term.params，放前面会被整个覆盖掉。
    term = cfg.observations.policy.point_cloud
    if args_cli.yolo_weights is not None:
        term.params["yolo_weights"] = args_cli.yolo_weights
    if args_cli.yolo_conf is not None:
        term.params["yolo_conf"] = args_cli.yolo_conf

    if args_cli.no_joint_vel:
        set_critic_joint_vel_enabled(cfg, False)
    set_success_termination_enabled(cfg, args_cli.enable_success_termination)
    # 必须在 set_success_termination_enabled **之后** —— 那个 setter 会新建一个
    # 不带偏移的 DoneTerm，放前面会被盖掉，导致 success 判的点和奖励的点不一致。
    set_grasp_target_offset_cm(cfg, args_cli.grasp_offset_cm)
    return cfg


def main():
    if not (0.0 <= args_cli.occlusion_severity <= 1.0):
        raise SystemExit(
            f"--occlusion_severity 必须在 [0,1]，got {args_cli.occlusion_severity}"
        )
    severity_range = None
    if args_cli.occlusion_severity_min is not None:
        severity_range = (
            args_cli.occlusion_severity_min,
            args_cli.occlusion_severity_max,
        )
    _occ_active = severity_range is not None or args_cli.occlusion_severity > 0.0

    env_cfg = _build_env_cfg()
    term = env_cfg.observations.policy.point_cloud

    print("=" * 78)
    if _STAGE == 4:
        print(f"[Stage] 4  (完整相机管线，**掩码来自 YOLO-seg，每 "
              f"{term.params['yolo_step']} 步跑一次**) + random_box 遮挡")
    else:
        print(f"[Stage] {_STAGE}  (完整相机管线，**掩码来自 YOLO-seg**) + random_box 遮挡")
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
    if _occ_active:
        if severity_range is not None:
            lo, hi = severity_range
            print(f"[Occ]   random_box  severity ~ U[{lo:.2f}, {hi:.2f}]"
                  f"（相机侧盒心+随机朝向，每 episode 重采，删最近邻该比例候选点）")
        else:
            print(f"[Occ]   random_box  severity={args_cli.occlusion_severity:.2f}"
                  f"（相机侧盒心+随机朝向，删最近邻该比例候选点）")
    else:
        print("[Occ]   无遮挡（== stage 3/4 baseline，作对照）")
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
        print("[提示] 没有 --resume。建议从 stage 3/4 的 checkpoint 续训（**不是**"
              "stage 2 的 —— 掩码来源不同）。")
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

    # 日志目录名：把遮挡程度编进去，多条曲线可进同一张 TensorBoard 对照。
    if _occ_active:
        if severity_range is not None:
            lo, hi = severity_range
            occ_tag = f"rbox_r{int(round(lo * 100))}-{int(round(hi * 100))}"
        else:
            occ_tag = f"rbox_{int(round(args_cli.occlusion_severity * 100))}"
    else:
        occ_tag = "none"
    log_name = f"xarm7_pick_pointcloud_stage{_STAGE}_occ_{occ_tag}"
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

    # 连 yolo_mask_source.py 和 occlusion.py 一起存档：掩码怎么来的、遮挡谓词长
    # 什么样，是这条路线的核心变量，事后复现时缺了它们就说不清。
    for src in (
        os.path.join(_PROJECT_DIR, "configs", "xarm7_pick_pointcloud_env_cfg.py"),
        os.path.join(_PROJECT_DIR, "configs", "point_bridge_pointcloud.py"),
        os.path.join(_PROJECT_DIR, "configs", "yolo_mask_source.py"),
        os.path.join(_PROJECT_DIR, "configs", "occlusion.py"),
        os.path.join(_PROJECT_DIR, "model", "pointnet_actor_critic.py"),
        os.path.join(_CURRENT_DIR, "train_pick_pointcloud_yolo_occluded.py"),
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
