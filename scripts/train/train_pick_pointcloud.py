#!/usr/bin/env python3
"""xArm7 Point Bridge 点表征 PPO 训练 / 回放入口。

观测（规格 point_bridge_rl_observation_spec.md §3.1）：

    actor  : 物体点 64x3(192) + 夹爪点 6x3(18) + joint_pos 7 → 217 维
             物体点与夹爪点各过一次同一个 PointNet（双 token，对齐参考实现），
             两个 512 维 embedding 拼接后接 MLP。每点仅 xyz，无类型通道。
    critic : privileged 状态 35 维（GT 物体位姿 / EE 位姿 / 关节角 / 关节速度 / 上一动作）

相机换成 xarm_va 的标定内外参（640x480 canonical，fx=fy=615，ROS 光学系外参），
不再走 camfix 那套未标定的 legacy 视角。Theia RGB 特征全部移除。

阶段化验证入口（规格 §7，按顺序不要跳）：

    --stage 0   可视化。单环境 + 点云投回渲染图，不训练。
    --stage 1   信息上限。绕过相机，从物体网格 GT 采点、无噪声。
    --stage 2   完整管线。掩码 + 深度反投影 + sigma=1cm 噪声。
    --stage 10  stage 2 + 腕部相机。腕部**不分割**，在整帧深度里采 64 点（基座
                系）；网络分成全局 encoder（物体点 + 夹爪点，共享权重）与腕部
                encoder（独立权重），三个 embedding 拼接。观测 217 → 409 维，
                因此**无法从任何既有 checkpoint 续训**。

阶段 1 学不出来说明点表征无法表达该任务，不要往下走；阶段 1 好而阶段 2 崩，首查
外参是否用了真机实参、掩码像素数是否够、噪声幅度。
"""

import argparse
import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))
sys.path.insert(0, _PROJECT_DIR)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="xArm7 Point Bridge 点表征 PPO — actor 吃 70x4 点云 + 关节角，critic 吃 privileged 状态"
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
        "PPO minibatch 数，默认 None=沿用配置文件（num_mini_batches=4）。"
        "传具体值则强制覆盖。保持 minibatch≈6000 时取 num_envs*24/6000，"
        "例如 1024→4 / 5000→20 / 6000→24 / 10000→40。"
    ),
)
parser.add_argument(
    "--stage",
    type=int,
    default=2,
    choices=(-2, -1, 0, 1, 2, 10),
    help=(
        "验证阶段：-2=privileged base 去掉关节速度（28 维，测速度那一路的贡献），"
        "-1=privileged base（actor+critic 都吃 GT 状态，验证环境可解），"
        "0=可视化(不训练)，1=GT 采点信息上限，2=完整相机管线，"
        "10=stage 2 + 腕部相机（腕部不分割、整帧采 64 点，双 PointNet，409 维）。"
    ),
)
parser.add_argument(
    "--no_joint_vel",
    action="store_true",
    help=(
        "去掉 critic 的 7 维关节速度（35 维 → 28 维），只对 stage 1/2 有效。"
        "actor 本来就不含速度，所以这是『整条链路彻底不用关节速度』的开关。"
        "stage -2 已是无速度版，无需再加此项。"
    ),
)
parser.add_argument(
    "--enable_success_termination",
    action="store_true",
    help="启用 reach_success 提前 done；默认关闭，与 camfix 训练保持一致的 no-success horizon。",
)
parser.add_argument(
    "--grasp_offset_cm",
    type=float,
    default=3.0,
    help=(
        "抓取目标点抬到工件正上方多少厘米（世界系向上，随工件 yaw 一起转）。"
        "默认 3.0 = 绿球位置。给 0 则回到工件根原点（旧行为，model_40300.pt "
        "等既有 checkpoint 就是在 0 上训的，续训时要显式写 --grasp_offset_cm 0）。"
    ),
)
parser.add_argument(
    "--log_root",
    type=str,
    default=None,
    help=(
        "日志根目录；默认 <项目>/logs。要存数据盘就传这个，"
        "例如 --log_root /autodl-pub/data/gx-va/logs"
    ),
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# 相机是观测来源，且程序化桌面用 PreviewSurface(MDL)，两者都要求渲染型 Kit experience
args_cli.enable_cameras = True
if args_cli.stage == 0:
    args_cli.num_envs = 1

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
    set_success_termination_enabled,
)
from configs.point_bridge_pointcloud import (
    M_OBJ,
    M_WRIST,
    N_ROBOT,
    NUM_POINTS,
    POINT_DIM,
)
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from model.pointnet_actor_critic import register_with_rsl_rl


def _build_env_cfg():
    cfg = (
        XArm7PickPointCloudPlayEnvCfg()
        if (args_cli.play or args_cli.stage == 0)
        else XArm7PickPointCloudEnvCfg()
    )
    cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        cfg.seed = args_cli.seed

    # 回放时保留桌面贴图与光照 DR：回放就是为了看画面，光秃秃的桌面既没法和阶段
    # 1/2 对照，也没法和真机比对。训练时摘掉（stage -1 观测不含任何像素）。
    set_observation_stage(cfg, args_cli.stage, keep_appearance=args_cli.play)
    # set_observation_stage 之后再动 critic —— stage 1/2 分支会重设观测项，
    # 放前面会被覆盖。stage -1/-2 的 actor/critic 由各自 setter 成对设定，
    # 不在这里插手（-2 本身就是无速度版）。
    if args_cli.no_joint_vel and args_cli.stage in (1, 2, 10):
        set_critic_joint_vel_enabled(cfg, False)
    # 回放 / 阶段 0 要把**所有**点云路的噪声归零。PlayEnvCfg.__post_init__ 只能管到
    # point_cloud —— wrist_point_cloud 是 set_observation_stage 之后才存在的，
    # 必须在这里补一刀，否则 stage 10 回放会是"全局无噪声、腕部带 1cm 噪声"。
    if args_cli.play or args_cli.stage == 0:
        disable_point_cloud_noise(cfg)
    set_success_termination_enabled(cfg, args_cli.enable_success_termination)
    # 必须在 set_success_termination_enabled **之后** —— 那个 setter 会新建一个
    # 不带偏移的 DoneTerm，放前面会被盖掉，导致 success 判的点和奖励的点不一致。
    set_grasp_target_offset_cm(cfg, args_cli.grasp_offset_cm)
    return cfg


def _run_stage0_visualization(env) -> None:
    """规格 §7 阶段 0：把生成的点云投回渲染图，人工确认几何正确。

    验收项（不通过就不要继续往下走，约 90% 的问题在这一步暴露）：
      - 物体点贴在物体表面
      - 夹爪点跟随夹爪刚性移动
      - 夹爪遮挡物体时被挡区域的点消失
      - 无点堆积在原点或相机光心
      - 掩码像素数 >= 200~500（规格 §2.2）
    """
    import matplotlib.pyplot as plt
    import numpy as np

    from configs.xarm7_pick_pointcloud_env_cfg import (
        debug_capture,
        project_points_to_pixels,
    )

    base_env = env.unwrapped

    fig, (ax_img, ax_3d) = plt.subplots(
        1, 2, figsize=(13, 5), subplot_kw=None, gridspec_kw={"width_ratios": [1, 1]}
    )
    ax_3d.remove()
    ax_3d = fig.add_subplot(1, 2, 2, projection="3d")
    ax_img.axis("off")
    ax_img.set_title("投影回渲染图：物体点(红) / 夹爪点(青)", fontsize=10)
    ax_3d.set_title("基座系点云", fontsize=10)
    plt.ion()
    plt.show(block=False)

    obs, _ = env.reset()
    step = 0

    while simulation_app.is_running():
        with torch.no_grad():
            # 阶段 0 不评估策略，用零动作让机械臂停住便于逐帧核对几何。
            # 想手动移动机械臂时改用随机动作即可。
            actions = torch.zeros(
                (env.num_envs, base_env.action_manager.total_action_dim),
                device=env.device,
            )
        obs, _, dones, _ = env.step(actions)

        cap = debug_capture(base_env)
        if cap is None:
            step += 1
            continue

        pts_base = cap["points"][0, :, :3]
        # 点云不再有类型通道，身份由**位置**决定：前 M_OBJ 个是物体点，其后是夹爪点。
        # 这与网络侧的切分方式一致（见 PointNetActorCritic._encode_actor）。
        is_obj_t = torch.arange(pts_base.shape[0], device=pts_base.device) < M_OBJ

        # 优先显示 RGB，降级到深度可视化
        if "rgb" in cap:
            rgb_like = cap["rgb"]  # shape (B, H, W, 3), 范围 [0, 1]
        else:
            rgb_like = cap["depth_vis"]

        uv, in_front = project_points_to_pixels(base_env, cap["points"][:1, :, :3])
        uv = uv[0].cpu().numpy()
        in_front = in_front[0].cpu().numpy()
        is_obj = is_obj_t.cpu().numpy()

        ax_img.clear()
        ax_img.axis("off")
        ax_img.set_title(
            f"掩码像素={int(cap['visible_counts'][0])}"
            f"  可见率={float(cap['visible_ratio'][0]):.2f}"
            f"  (§2.2 门槛 200~500)",
            fontsize=10,
        )
        ax_img.imshow(rgb_like[0])
        sel_obj = in_front & is_obj
        sel_rob = in_front & (~is_obj)
        ax_img.scatter(uv[sel_obj, 0], uv[sel_obj, 1], s=6, c="red", label="object")
        ax_img.scatter(uv[sel_rob, 0], uv[sel_rob, 1], s=14, c="cyan", label="gripper")
        ax_img.legend(loc="upper right", fontsize=7)

        p = pts_base.cpu().numpy()
        ax_3d.clear()
        ax_3d.set_title("基座系点云（红=物体 青=夹爪）", fontsize=10)
        ax_3d.scatter(p[is_obj, 0], p[is_obj, 1], p[is_obj, 2], s=6, c="red")
        ax_3d.scatter(p[~is_obj, 0], p[~is_obj, 1], p[~is_obj, 2], s=25, c="cyan")
        ax_3d.set_xlabel("x"); ax_3d.set_ylabel("y"); ax_3d.set_zlabel("z")
        # 原点标出来，便于发现"点堆在原点"这个典型故障
        ax_3d.scatter([0], [0], [0], s=40, c="black", marker="x")

        fig.canvas.draw_idle()
        plt.pause(0.001)

        step += 1
        if dones[0]:
            obs, _ = env.reset()
            step = 0


def main():
    env_cfg = _build_env_cfg()

    _STAGE_LABEL = {
        -2: "privileged base 无关节速度（28 维）",
        -1: "privileged base（actor+critic 都吃 GT 状态）",
        0: "可视化",
        1: "GT 采点信息上限",
        2: "完整相机管线",
        10: "完整相机管线 + 腕部相机（双 PointNet）",
    }
    print("=" * 78)
    print(f"[Stage] {args_cli.stage}  ({_STAGE_LABEL[args_cli.stage]})")
    if args_cli.stage == -2:
        print("[Obs]   privileged 状态 28 维（GT 物体位姿 / EE 位姿 / 关节角 / 上一动作）")
        print("        与 stage -1 的差就是关节速度那 7 维的贡献。")
    elif args_cli.stage == -1:
        print("[Obs]   privileged 状态 35 维（GT 物体位姿 / EE 位姿 / 关节角 / 关节速度 / 上一动作）")
        print("        无点云、无相机。这一版学不出来 → 问题在环境而非表征。")
    else:
        print(f"[Obs]   物体点 {M_OBJ}x{POINT_DIM} + 夹爪点 {N_ROBOT}x{POINT_DIM} "
              f"= {NUM_POINTS * POINT_DIM} 维点云（双 token，无类型通道）")
        if args_cli.stage == 10:
            _n_wrist_dim = M_WRIST * POINT_DIM
            print(f"        + 腕部点 {M_WRIST}x{POINT_DIM} = {_n_wrist_dim} 维"
                  f"（整帧采样、**不分割**，基座系）")
            print(f"        policy 总计 {NUM_POINTS * POINT_DIM + _n_wrist_dim + 7} 维 "
                  f"= [物体 {M_OBJ * POINT_DIM} | 夹爪 {N_ROBOT * POINT_DIM} | "
                  f"腕部 {_n_wrist_dim} | 关节角 7]")
            print("        网络：全局 encoder（物体+夹爪共享权重）+ 腕部 encoder"
                  "（独立权重），三个 embedding 拼接")
        if args_cli.no_joint_vel and args_cli.stage in (1, 2, 10):
            print("[Critic] privileged 状态 28 维 —— 已去掉关节速度")
        else:
            print("[Critic] privileged 状态 35 维（含关节速度）")
    print(f"[Reward] success termination: "
          f"{'enabled' if args_cli.enable_success_termination else 'disabled (no-success default)'}")
    if args_cli.grasp_offset_cm != 0.0:
        print(f"[Target] 抓取目标点 = 工件正上方 {args_cli.grasp_offset_cm:.1f} cm"
              f"（工件局部 z={-args_cli.grasp_offset_cm / 100.0:+.3f} m，随 yaw 转）"
              f" —— 与既有 checkpoint 的目标点不同，不可直接续训")
    else:
        print("[Target] 抓取目标点 = 工件根原点（旧行为）")
    print("=" * 78)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    if args_cli.stage == 0:
        _run_stage0_visualization(env)
        env.close()
        return

    from datetime import datetime

    if args_cli.stage in (-1, -2):
        # privileged base：内置 ActorCritic（纯 MLP），没有点云要编码
        from configs.agents.rsl_rl_ppo_cfg import XArm7PickPrivilegedBasePPORunnerCfg

        agent_cfg = XArm7PickPrivilegedBasePPORunnerCfg()
        # 分开存日志，两版曲线可在同一张 TensorBoard 里直接对照
        log_name = (
            "xarm7_pick_privileged_base_no_vel"
            if args_cli.stage == -2
            else "xarm7_pick_privileged_base"
        )
    else:
        from configs.agents.rsl_rl_ppo_cfg import XArm7PickPointCloudPPORunnerCfg

        agent_cfg = XArm7PickPointCloudPPORunnerCfg()
        # 网络侧的点云形状必须与环境侧一致，从同一份常量推导而不是各写一遍
        agent_cfg.policy.num_object_points = M_OBJ
        agent_cfg.policy.num_robot_points = N_ROBOT
        agent_cfg.policy.point_dim = POINT_DIM
        # 只有 stage 10 有腕部分支。其余阶段保持 0 —— 那是"网络结构与既有
        # checkpoint 逐位兼容"的开关，不能顺手改成永远 M_WRIST。
        agent_cfg.policy.num_wrist_points = M_WRIST if args_cli.stage == 10 else 0
        log_name = f"xarm7_pick_pointcloud_stage{args_cli.stage}"
        if args_cli.no_joint_vel:
            # 分开存日志，有速度/无速度两条曲线可在同一张 TensorBoard 里直接对照
            log_name += "_no_vel"

    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations
    if args_cli.num_mini_batches is not None:
        agent_cfg.algorithm.num_mini_batches = args_cli.num_mini_batches

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
