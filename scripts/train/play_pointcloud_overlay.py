#!/usr/bin/env python3
"""策略回放 + 把**送进 PointNet 的那份点云**叠在 RGB 上。

和已有两个脚本的分工::

    train_pick_pointcloud.py --stage 0   零动作 + 投影，机械臂不动，不加载权重
    train_pick_pointcloud.py --play      策略驱动，但不做投影
    本脚本                                策略驱动 + 投影            ← 两者的交集

看的是"策略实际吃到的观测"，因此点**必须**取自 ``debug_capture``，不能自己照着
公式重算一遍。那份数据里已经含了 FPS 采样、sigma=1cm 噪声、workspace 裁剪，以及
"本帧可见点为 0 时回退到上一帧"的兜底（env_cfg 第 617-620 行）。自己重算会得到
另一朵点云，看着对也证明不了训练路径是对的。

关于 RGB：stage 2 的相机被摘掉了 rgb 一路（env_cfg 第 957 行，为省显存）。本脚本
在**运行时的 cfg 对象**上把它加回来，不改任何源码，训练路径不受影响。代价是每 env
多约 0.9 MB 显存（640x480x3），回放 1 个 env 无所谓 —— 但也因此**不能用本脚本
测训练时的显存占用**。

判读要点（和 stage 0 同一套验收标准）：
  - 红点贴在工件表面，不是飘在旁边或糊成一团
  - 青点跟着夹爪刚性移动
  - 夹爪挡住工件时，被挡那块的红点消失
  - 没有点堆在原点（黑色 x 标记处）或相机光心
  - 掩码像素数 >= 200（低于此点云会退化成一小撮重复点）

用法::

    # 弹窗实时看
    ~/IsaacLab/isaaclab.sh -p scripts/train/play_pointcloud_overlay.py \
        --checkpoint logs/xarm7_pick_pointcloud_stage2/2026-08-13_19-10-02/model_11500.pt

    # 无显示环境，存帧
    ~/IsaacLab/isaaclab.sh -p scripts/train/play_pointcloud_overlay.py \
        --checkpoint <ckpt> --save_frames --headless --max_steps 300
"""

import argparse
import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))
sys.path.insert(0, _PROJECT_DIR)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="策略回放，并把送进 PointNet 的点云投影叠加到 RGB 上"
)
parser.add_argument("--checkpoint", type=str, required=True, help="训练出的 .pt 权重")
parser.add_argument("--num_envs", type=int, default=1, help="回放环境数，看图建议 1")
parser.add_argument(
    "--stage",
    type=int,
    default=2,
    choices=(1, 2, 10),
    help="必须与训练该 checkpoint 时一致，否则网络形状对不上直接报错。"
         "stage 10 的叠加图仍然只画全局点（腕部点属于另一个相机，投到固定相机的"
         "像素平面上没有意义）。",
)
parser.add_argument("--seed", type=int, default=None)
parser.add_argument(
    "--max_steps", type=int, default=0, help="跑够步数退出；0 表示一直跑"
)
parser.add_argument(
    "--save_frames", action="store_true", help="把每帧叠加图存盘（headless 下用这个）"
)
parser.add_argument(
    "--out",
    type=str,
    default=os.path.join(_PROJECT_DIR, "logs", "pointcloud_overlay"),
)
parser.add_argument(
    "--no_window", action="store_true", help="不弹 matplotlib 窗口，只存盘"
)
parser.add_argument("--env_spacing", type=float, default=6.0)
parser.add_argument(
    "--grasp_offset_cm",
    type=float,
    default=3.0,
    help="抓取目标点在工件上方多少 cm，必须与训练该 checkpoint 时用的值一致。"
         "回放 08-17 之前的 checkpoint（如 model_40300.pt）要传 0。",
)
parser.add_argument(
    "--noise_std",
    type=float,
    default=0.0,
    help="物体点高斯噪声(米)。默认 0 便于核对几何 —— sigma=1cm 在 0.5m 处约合 "
         "12 像素，足以把边缘点推出掩码，看起来像反投影错了。想看训练时的真实"
         "观测就传 0.01。",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 相机是观测来源，且程序化桌面用 PreviewSurface(MDL)，两者都要求渲染型 Kit experience
args_cli.enable_cameras = True

# AppLauncher 会消费并改写 args_cli 里的启动参数，之后再读 headless 拿到的不是
# 原始值。要判断能不能弹窗，必须在这之前快照。
_HEADLESS = bool(getattr(args_cli, "headless", False))
# 只存盘不看窗口时，等价于 headless —— 少起一个 GUI，省显存也更快。
if args_cli.no_window:
    args_cli.headless = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import numpy as np
import torch

from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from rsl_rl.runners import OnPolicyRunner

from configs.point_bridge_pointcloud import M_OBJ, M_WRIST, N_ROBOT, POINT_DIM
from configs.xarm7_pick_pointcloud_env_cfg import (
    XArm7PickPointCloudPlayEnvCfg,
    _get_instance_lut,
    debug_capture,
    project_points_to_pixels,
    set_grasp_target_offset_cm,
    set_observation_stage,
    set_success_termination_enabled,
)
from model.pointnet_actor_critic import register_with_rsl_rl


def _to_uint8_rgb(rgb: torch.Tensor) -> np.ndarray:
    """相机 rgb 输出 → (N, H, W, 3) uint8，兼容 float 与带 alpha 的情况。"""
    arr = rgb.detach().cpu().numpy()
    if arr.shape[-1] == 4:
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        hi = float(np.nanmax(arr)) if arr.size else 0.0
        # >1.5 说明本来就是 0~255 的 float，不能再乘 255
        scale = 1.0 if hi > 1.5 else 255.0
        arr = np.clip(np.nan_to_num(arr) * scale, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(arr)


def _build_env_cfg():
    cfg = XArm7PickPointCloudPlayEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.scene.env_spacing = args_cli.env_spacing
    if args_cli.seed is not None:
        cfg.seed = args_cli.seed

    set_observation_stage(cfg, args_cli.stage)
    set_success_termination_enabled(cfg, False)
    # 回放要用**与训练同一个**目标点，否则打印的距离/success 是按另一个点算的。
    # 回放旧 checkpoint（瞄工件根原点训的）时要显式 --grasp_offset_cm 0。
    set_grasp_target_offset_cm(cfg, args_cli.grasp_offset_cm)

    # set_observation_stage 会无条件重写 term.params，把 PlayEnvCfg 里设好的
    # noise_std=0 又覆盖成 NOISE_STD_M（stage 2 分支）。必须在它之后再设一次。
    # 不关噪声的话，1cm 的抖动在图上约 12 像素，边缘的红点会被推到掩码外，
    # 看起来像反投影错了 —— 实际只是噪声。
    cfg.observations.policy.point_cloud.params["noise_std"] = args_cli.noise_std

    # 腕部那一路同理（stage 10）。它是 set_observation_stage 之后才存在的观测项，
    # PlayEnvCfg.__post_init__ 管不到，不在这里补就会"全局无噪声、腕部带噪声"。
    wrist_term = getattr(cfg.observations.policy, "wrist_point_cloud", None)
    if wrist_term is not None:
        wrist_term.params["noise_std"] = args_cli.noise_std

    # stage 2 把 rgb 摘掉了（省显存），这里加回来当叠加底图。
    # 只动运行时对象，不碰 set_observation_stage 的源码。
    cam = getattr(cfg.scene, "camera_fixed", None)
    if cam is not None and hasattr(cam, "data_types"):
        if "rgb" not in cam.data_types:
            cam.data_types = list(cam.data_types) + ["rgb"]
    elif args_cli.stage == 1:
        # stage 1 绕过相机（camera_fixed 被置 None），点从物体网格 GT 采样。
        # 没有相机就没有 RGB 底图，也没有外参可做投影 —— 只能看 3D 那半张图。
        print("[警告] stage 1 没有相机，无 RGB 底图，只显示 3D 点云。")
    return cfg


def main() -> None:
    env_cfg = _build_env_cfg()

    print("=" * 78)
    print(f"[Stage]      {args_cli.stage}")
    print(f"[Checkpoint] {args_cli.checkpoint}")
    print(f"[Obs]        物体点 {M_OBJ}x{POINT_DIM} + 夹爪点 {N_ROBOT}x{POINT_DIM}")
    print("=" * 78)

    env = ManagerBasedRLEnv(cfg=env_cfg)
    base_env = env.unwrapped
    camera = env.scene["camera_fixed"] if "camera_fixed" in env.scene.keys() else None
    env = RslRlVecEnvWrapper(env)

    from configs.agents.rsl_rl_ppo_cfg import XArm7PickPointCloudPPORunnerCfg

    agent_cfg = XArm7PickPointCloudPPORunnerCfg()
    # 网络形状必须与环境侧同源，不能各写一遍
    agent_cfg.policy.num_object_points = M_OBJ
    agent_cfg.policy.num_robot_points = N_ROBOT
    agent_cfg.policy.point_dim = POINT_DIM
    # 必须与训练时一致，否则 actor 第一层形状对不上，runner.load 直接报错。
    agent_cfg.policy.num_wrist_points = M_WRIST if args_cli.stage == 10 else 0

    # OnPolicyRunner 用 eval(class_name) 在自己的模块 globals 里解析策略类，
    # 必须在构造 runner 之前注入。
    register_with_rsl_rl()
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.device)
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.device)

    import matplotlib

    show_window = not args_cli.no_window and not _HEADLESS
    if not show_window:
        matplotlib.use("Agg")  # 必须在 pyplot 导入前设定
    import matplotlib.pyplot as plt

    if args_cli.save_frames:
        os.makedirs(args_cli.out, exist_ok=True)

    fig = plt.figure(figsize=(16, 9))
    # 上排：RGB 叠加 + 基座系 3D；下排：深度 / 分割 / 掩码
    ax_img = fig.add_subplot(2, 3, 1)
    ax_3d = fig.add_subplot(2, 3, 3, projection="3d")
    ax_depth = fig.add_subplot(2, 3, 4)
    ax_seg = fig.add_subplot(2, 3, 5)
    ax_mask = fig.add_subplot(2, 3, 6)
    for _a in (ax_img, ax_depth, ax_seg, ax_mask):
        _a.axis("off")
    fig.tight_layout()
    if show_window:
        plt.ion()
        plt.show(block=False)

    obs, _ = env.reset()
    step = 0
    saved = 0
    # 实例 LUT 内部有缓存，取一次即可；相机未就绪时为 None，下面每帧再补取。
    lut = None

    while simulation_app.is_running():
        with torch.no_grad():
            actions = policy(obs)
        obs, _, dones, _ = env.step(actions)
        step += 1

        # 首次调用只是打开开关，观测函数下一帧才写入，返回 None 是正常的
        cap = debug_capture(base_env)
        if cap is None:
            continue

        if lut is None and camera is not None:
            lut = _get_instance_lut(base_env, camera)

        pts_base = cap["points"][0, :, :3]
        # 点云没有类型通道，身份由**位置**决定：前 M_OBJ 个是物体点，其后是夹爪点。
        # 与网络侧 PointNetActorCritic._encode_actor 的切分方式一致。
        is_obj = (torch.arange(pts_base.shape[0], device=pts_base.device) < M_OBJ).cpu().numpy()
        p = pts_base.cpu().numpy()

        # ── 投影一次，五格共用 ────────────────────────────────────────────
        # 下排三格与 RGB 出自同一相机、同一分辨率，像素坐标可直接复用。
        uv = sel_obj = sel_rob = None
        if camera is not None:
            uv_t, in_front_t = project_points_to_pixels(
                base_env, cap["points"][:1, :, :3]
            )
            uv = uv_t[0].cpu().numpy()
            in_front = in_front_t[0].cpu().numpy()
            # 落在画面外的点画出来只会挤在边缘造成误判，剔掉
            h_img = camera.cfg.height
            w_img = camera.cfg.width
            inside = (
                in_front
                & (uv[:, 0] >= 0) & (uv[:, 0] < w_img)
                & (uv[:, 1] >= 0) & (uv[:, 1] < h_img)
            )
            sel_obj = inside & is_obj
            sel_rob = inside & (~is_obj)

        def _scatter(ax):
            """把物体点(红)/夹爪点(青)叠到任意一格上。"""
            if uv is None:
                return
            ax.scatter(uv[sel_obj, 0], uv[sel_obj, 1], s=5, c="red")
            ax.scatter(uv[sel_rob, 0], uv[sel_rob, 1], s=12, c="cyan")

        # ── 左图：投影叠加 ────────────────────────────────────────────────
        ax_img.clear()
        ax_img.axis("off")
        if camera is not None and "rgb" in camera.data.output:
            rgb = _to_uint8_rgb(camera.data.output["rgb"])[0]
            ax_img.imshow(rgb)
            ax_img.scatter(uv[sel_obj, 0], uv[sel_obj, 1], s=6, c="red", label="object")
            ax_img.scatter(uv[sel_rob, 0], uv[sel_rob, 1], s=14, c="cyan", label="gripper")
            ax_img.legend(loc="upper right", fontsize=7)

            n_px = int(cap["mask_pixels"][0]) if "mask_pixels" in cap else -1
            ratio = float(cap["visible_ratio"][0]) if "visible_ratio" in cap else -1.0
            flag = "OK" if n_px >= 200 else "LOW!"
            ax_img.set_title(
                f"step {step}  掩码={n_px}px {flag}  可见率={ratio:.2f}"
                f"  画面内点={int(inside.sum())}/{len(inside)}",
                fontsize=10,
            )
        else:
            ax_img.text(0.5, 0.5, "无 RGB（stage 1 不用相机）",
                        ha="center", va="center", fontsize=12)
            ax_img.set_title(f"step {step}", fontsize=10)

        # ── 下排：深度 / 分割 / 掩码 ──────────────────────────────────────
        # 这三格是"送进反投影的原始三路"，左上的红点就是从它们算出来的。
        # 点飘了先看这里：深度空洞 → 点缺失；掩码碎 → 点粘到背景。
        for _a in (ax_depth, ax_seg, ax_mask):
            _a.clear()
            _a.axis("off")

        if camera is not None:
            out = camera.data.output

            # 深度：只按有效像素定标，否则背景的 0 会把动态范围压死
            draw = out.get("distance_to_image_plane")
            if draw is not None:
                d = draw[0]
                d = d[..., 0] if d.dim() == 3 else d
                d = d.detach().cpu().numpy()
                valid = np.isfinite(d) & (d > 0)
                if valid.any():
                    lo, hi = float(d[valid].min()), float(d[valid].max())
                    norm = np.zeros_like(d, dtype=np.float32)
                    if hi > lo:
                        norm[valid] = (d[valid] - lo) / (hi - lo)
                    ax_depth.imshow(norm, cmap="turbo", vmin=0.0, vmax=1.0)
                    _scatter(ax_depth)
                    ax_depth.set_title(f"深度 {lo:.2f}~{hi:.2f}m", fontsize=9)
                else:
                    ax_depth.set_title("深度全无效!", fontsize=9)

            # 分割：实例 ID 是任意整数，用 nipy_spectral 拉开相邻 ID 的色差
            sraw = out.get("instance_id_segmentation_fast")
            seg_np = None
            if sraw is not None:
                s = sraw[0]
                s = s[..., 0] if s.dim() == 3 else s
                seg_np = s.detach().cpu().numpy().astype(np.int64)
                ax_seg.imshow(seg_np, cmap="nipy_spectral", interpolation="nearest")
                _scatter(ax_seg)
                ax_seg.set_title(f"实例分割 ({len(np.unique(seg_np))} ids)", fontsize=9)

            # 掩码：走环境自己的 LUT，保证和训练用的是同一份
            if seg_np is not None and lut is not None:
                seg_t = torch.from_numpy(seg_np).to(lut.device)
                seg_idx = seg_t.long().clamp_(min=0, max=lut.numel() - 1)
                # 物体像素为 env 序号+1；这里只看 env 0
                mask_np = (lut[seg_idx] == 1).cpu().numpy()
                n_px = int(mask_np.sum())
                ax_mask.imshow(mask_np, cmap="gray", vmin=0, vmax=1)
                _scatter(ax_mask)
                # 规格 §2.2 门槛 200px，低于此点云会退化成一小撮重复点
                ax_mask.set_title(
                    f"工件掩码 {n_px}px {'OK' if n_px >= 200 else 'LOW!'}", fontsize=9
                )

        # ── 右上：基座系 3D 点云 ──────────────────────────────────────────
        ax_3d.clear()
        ax_3d.set_title("基座系点云（红=物体 青=夹爪 黑x=原点）", fontsize=10)
        ax_3d.scatter(p[is_obj, 0], p[is_obj, 1], p[is_obj, 2], s=6, c="red")
        ax_3d.scatter(p[~is_obj, 0], p[~is_obj, 1], p[~is_obj, 2], s=25, c="cyan")
        # 原点标出来，便于发现"点堆在原点"这个典型故障
        ax_3d.scatter([0], [0], [0], s=40, c="black", marker="x")
        ax_3d.set_xlabel("x"); ax_3d.set_ylabel("y"); ax_3d.set_zlabel("z")

        if show_window:
            fig.canvas.draw_idle()
            plt.pause(0.001)
        if args_cli.save_frames:
            fig.savefig(os.path.join(args_cli.out, f"frame_{saved:05d}.png"), dpi=90)
            saved += 1

        if args_cli.max_steps and step >= args_cli.max_steps:
            break
        if bool(dones[0]):
            obs, _ = env.reset()

    if args_cli.save_frames:
        print(f"已存 {saved} 帧 → {args_cli.out}")
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
