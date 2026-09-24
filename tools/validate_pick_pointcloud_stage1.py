#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""xArm7 点云策略仿真验证 + 录像（stage 1 / stage 2 通用）。

用某次训练的某个 checkpoint，在 Isaac Lab 里回放策略，把固定相机画面录成 mp4 存到
autodl-tmp 下新建的文件夹里。``--stage`` 决定观测来源与扫哪个 run 目录：``1`` =
GT 物体表面采点（信息上限，``xarm7_pick_pointcloud_stage1``），``2`` = 完整相机管线
（``xarm7_pick_pointcloud_stage2_occ_<tag>``，tag 由遮挡参数编码）。

**选 checkpoint 有三种方式（优先顺序从高到低）：**

  1. ``--checkpoint /绝对/或/相对/model_2000.pt``  —— 精确指定某个 .pt
  2. ``--run_dir 2026-08-26_10-03-28``（或绝对路径）—— 指定 run，自动取它的最终 checkpoint
  3. 都不给 —— 交互式列出当前 stage 的 run 目录下所有 run，再列出该 run 下所有
     ``model_*.pt``，按序号选。

路径一律 ``os.path.abspath`` 归一化：**相对路径也能用**（不再因为漏写前导 ``/``
而 404），并且会在启动重型的 AppLauncher 之前就校验文件存在、失败直接早退。

**seed 不需要和训练一致**：策略权重从 checkpoint 确定性加载，与 seed 无关；seed
只决定评测时每个 episode 工件的随机摆放序列。交互式模式下会提示输入 seed（回车
=100）；想复现同一次评测就用同一个 seed，想多测几组就换不同 seed。

**stage 2 的遮挡参数必须与训练该 checkpoint 时一致**。观测函数是
``point_bridge_point_cloud_occluded``，遮挡方式/程度/轴/球心决定了哪些点被挖掉；
验证时给错 ``--occlusion_*`` 会让观测与 checkpoint 训练时看到的分布不同，策略行为
不可比。baseline（``--occlusion_mode none``）时也走同一观测函数（内部不注入过滤）。

关键点（照抄 training 的 play 路径，只多录像这一件事）：

  - **stage 1 的观测不读相机**。``set_observation_stage(stage=1)`` 会把
    ``camera_fixed`` 从场景里摘掉（观测是 ``point_bridge_point_cloud_gt``，从物体
    mesh 的 GT 位姿采点）。录像需要画面，因此这里在 ``set_observation_stage`` 之后
    把 ``camera_fixed`` 重新装回来 —— 它只供录像，**不参与观测**，不影响策略行为。
  - **stage 2 观测走相机深度 + 实例分割**，且 ``--occlusion_*`` 会注入坐标遮挡；
    遮挡在 :func:`set_observation_stage` 之后、:func:`disable_point_cloud_noise` 之前
    注入，与训练脚本 ``train_pick_pointcloud_occluded.py`` 的顺序一致。
  - 回放用 ``XArm7PickPointCloudPlayEnvCfg``（单环境、关观测噪声），并
    ``keep_appearance=True`` 保留桌面贴图与光照 DR，画面与训练阶段一致。
  - 策略加载走与 training 完全相同的 ``OnPolicyRunner`` 路径（``register_with_rsl_rl``
    → ``runner.load(checkpoint)`` → ``get_inference_policy``），不手搭网络。
  - 录像走固定相机 RGB，headless 离屏渲染，``imageio`` 写 mp4；``imageio`` 的
    ffmpeg 后端不可用时退级保存 PNG 帧序列（``<out>/frames/``），保证总有画面落盘。

用法（在 isaaclab python 下）::

    # stage 1（默认）
    python tools/validate_pick_pointcloud_stage1.py
    python tools/validate_pick_pointcloud_stage1.py --run_dir 2026-08-25_14-20-40
    python tools/validate_pick_pointcloud_stage1.py --checkpoint /root/autodl-tmp/logs/xarm7_pick_pointcloud_stage1/2026-08-25_14-20-40/model_2000.pt

    # stage 2 baseline（无遮挡）
    python tools/validate_pick_pointcloud_stage1.py --stage 2
    python tools/validate_pick_pointcloud_stage1.py --stage 2 --checkpoint /root/autodl-tmp/logs/xarm7_pick_pointcloud_stage2_occ_none/2026-08-27_14-29-34/model_2800.pt

    # stage 2 + 遮挡（须与训练时一致）
    python tools/validate_pick_pointcloud_stage1.py --stage 2 --occlusion_mode halfspace --occlusion_severity 0.25 --occlusion_axis x
    python tools/validate_pick_pointcloud_stage1.py --stage 2 --occlusion_mode sphere --occlusion_severity 0.5 --occlusion_center 0.5,0.5,0.5

    python tools/validate_pick_pointcloud_stage1.py --stage 2 --episodes 5 --video_fps 30
"""

import argparse
import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_CURRENT_DIR)
sys.path.insert(0, _PROJECT_DIR)

from isaaclab.app import AppLauncher

# autodl-tmp 根目录。训练时 --log_root 用的是 /root/autodl-tmp/logs，这里也存这里。
_AUTODL_TMP = "/root/autodl-tmp"
_LOGS_DIR = os.path.join(_AUTODL_TMP, "logs")


def _occ_tag(mode: str, severity: float, axis: str) -> str:
    """把遮挡参数编码成 log 目录名里那段 tag，与训练脚本逐字一致。

    训练脚本 ``scripts/train/train_pick_pointcloud_occluded.py`` 里：
        halfspace → halfspace{axis}_{severity%}，sphere/random_sphere/random_box →
        {mode}_{severity%}，none → none
    这里必须保持一致，否则 stage2 的 run 目录扫不到。
    """
    if mode != "none" and severity > 0.0:
        if mode == "halfspace":
            return f"halfspace{axis}_{int(round(severity * 100))}"
        return f"{mode}_{int(round(severity * 100))}"
    return "none"


def _runs_dir_for_stage(
    stage: int,
    occ_mode: str = "none",
    occ_severity: float = 0.0,
    occ_axis: str = "x",
) -> str:
    """按 stage（+遮挡参数）返回该阶段所有 run 所在的目录。

    stage 2 的 run 目录名带遮挡 tag（``xarm7_pick_pointcloud_stage2_occ_<tag>``），
    由训练脚本在写入时编码，验证时必须用同一套编码去扫。
    """
    if stage == 1:
        return os.path.join(_LOGS_DIR, "xarm7_pick_pointcloud_stage1")
    if stage == 2:
        tag = _occ_tag(occ_mode, occ_severity, occ_axis)
        return os.path.join(_LOGS_DIR, f"xarm7_pick_pointcloud_stage2_occ_{tag}")
    if stage == 10:
        return os.path.join(_LOGS_DIR, "xarm7_pick_pointcloud_stage10")
    # 其余 stage（0/3/4 及 -1/-2）目前没有对应的点云 run 目录，回退到 stage1 目录；
    # 交互式选择仍可用，或直接 --checkpoint 显式指定。
    return os.path.join(_LOGS_DIR, "xarm7_pick_pointcloud_stage1")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="xArm7 点云策略仿真验证 + 录像（--stage 选 1/2，2 可叠加遮挡）"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="精确指定某个 .pt（相对/绝对路径均可）。不给则交互式选择，或看 --run_dir",
    )
    parser.add_argument(
        "--run_dir",
        type=str,
        default=None,
        help=(
            "run 目录名或路径（如 2026-08-25_14-20-40，或它的绝对路径）。"
            "给这个会自动取该 run 的最终 checkpoint，无需再给 --checkpoint"
        ),
    )
    parser.add_argument(
        "--stage",
        type=int,
        default=1,
        choices=(-2, -1, 0, 1, 2, 3, 4, 10),
        help="观测 stage（默认 1 = GT 物体表面采点；2 = 完整相机管线）",
    )
    parser.add_argument(
        "--occlusion_mode",
        type=str,
        default="none",
        choices=("none", "halfspace", "sphere", "random_sphere", "random_box"),
        help=(
            "stage 2 遮挡几何：none=不遮，halfspace=切掉工件一侧，sphere=挖掉一块球，"
            "random_sphere=圆形随机位置，random_box=相机侧表面随机盒心+随机朝向，"
            "按最近邻删除 severity 比例候选点（与训练脚本一致）"
        ),
    )
    parser.add_argument(
        "--occlusion_severity",
        type=float,
        default=0.0,
        help="stage 2 遮挡程度 0.0..1.0（必须与训练该 checkpoint 时一致，否则观测对不上）",
    )
    parser.add_argument(
        "--occlusion_axis",
        type=str,
        default="x",
        help="halfspace 用：切哪一侧（x|-x|y|-y|z|-z，工件局部系）",
    )
    parser.add_argument(
        "--occlusion_center",
        type=str,
        default="0.5,0.5,0.5",
        help="sphere 用：球心在工件局部包围盒内的归一化分数 'cx,cy,cz'",
    )
    parser.add_argument(
        "--debug_vis_occlusion",
        action="store_true",
        help="stage2 + 遮挡时叠加可视化：半透明红=遮挡形状，红点=被挖候选点，绿点=存活候选点",
    )
    parser.add_argument(
        "--grasp_offset_cm",
        type=float,
        default=3.0,
        help="抓取目标点相对工件正上方偏移 cm（训练默认 3.0）",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=100,
        help="环境随机种子（与训练一致）",
    )
    parser.add_argument(
        "--episodes",
        type=int,
        default=3,
        help="录制多少个 episode（每次 reset 工件位置会重新随机化）",
    )
    parser.add_argument(
        "--video_fps",
        type=int,
        default=25,
        help="输出视频帧率；控制频率 50Hz，会按该帧率降采样录制",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default=None,
        help=f"输出目录；默认在 {_AUTODL_TMP}/validation 下新建带 run 名+checkpoint 名的文件夹",
    )
    AppLauncher.add_app_launcher_args(parser)
    return parser


# ── 交互式选择 ──────────────────────────────────────────────────────────────

def _list_checkpoints(run_dir: str):
    """run 目录下所有 model_*.pt，按迭代次数升序。"""
    if not os.path.isdir(run_dir):
        return []
    ckpts = [
        f for f in os.listdir(run_dir)
        if f.startswith("model_") and f.endswith(".pt")
    ]

    def _key(f: str):
        try:
            return int(f[len("model_"):-len(".pt")])
        except ValueError:
            return -1

    return sorted(ckpts, key=_key)


def _latest_checkpoint(runs_dir: str) -> str:
    """在 ``runs_dir`` 下找最新 run 的最终 checkpoint（按时间戳目录名升序取最后一个）。

    目录名是 ``%Y-%m-%d_%H-%M-%S`` 时间戳，字符串升序即时间升序，``reversed`` 后从
    最新开始找，跳过没有 ``model_*.pt`` 的目录。
    """
    if not os.path.isdir(runs_dir):
        raise SystemExit(f"[错误] run 目录不存在: {runs_dir}")
    runs = sorted(
        d for d in os.listdir(runs_dir)
        if os.path.isdir(os.path.join(runs_dir, d))
    )
    for r in reversed(runs):
        ckpts = _list_checkpoints(os.path.join(runs_dir, r))
        if ckpts:
            return os.path.join(runs_dir, r, ckpts[-1])
    raise SystemExit(f"[错误] {runs_dir} 下没有任何 model_*.pt，无法自动选 checkpoint")


def _ask_int(prompt: str, default: int, max_val: int) -> int:
    try:
        raw = input(f"{prompt} [{default}]: ").strip()
    except EOFError:  # 非交互 stdin（如管道/nohup），退回默认
        return default
    if raw == "":
        return default
    try:
        v = int(raw)
    except ValueError:
        print(f"  → 不是数字，用默认 {default}")
        return default
    if not 0 <= v <= max_val:
        print(f"  → 序号越界，用默认 {default}")
        return default
    return v


def _ask_seed(default: int = 100) -> int:
    """交互式输入 seed（≥0 的整数）。

    只决定评测时工件的随机摆放序列，与 checkpoint 权重无关（权重确定性地从
    checkpoint 加载）。回车用默认；想复现同一次评测用同一个 seed，想多测几组就换。
    """
    try:
        raw = input(
            f"输入 seed（回车=默认 {default}；只影响工件随机摆放，无需与训练一致）: "
        ).strip()
    except EOFError:  # 非交互 stdin（如管道/nohup），退回默认
        return default
    if raw == "":
        return default
    try:
        v = int(raw)
    except ValueError:
        print(f"  → 不是数字，用默认 {default}")
        return default
    if v < 0:
        print(f"  → seed 需 ≥ 0，用默认 {default}")
        return default
    return v


def _pick_checkpoint_interactively(runs_dir: str) -> str:
    """交互式选 run + checkpoint，返回绝对路径。"""
    print("=" * 70)
    print("交互式选择 checkpoint")
    print("=" * 70)

    if not os.path.isdir(runs_dir):
        latest = _latest_checkpoint(runs_dir)
        print(f"[提示] 找不到 run 目录 {runs_dir}，改用最新 {latest}")
        return latest

    runs = sorted(
        [d for d in os.listdir(runs_dir)
         if os.path.isdir(os.path.join(runs_dir, d))],
        reverse=True,  # 时间戳，新的在前
    )
    if not runs:
        latest = _latest_checkpoint(runs_dir)
        print(f"[提示] {runs_dir} 下没有 run，改用最新 {latest}")
        return latest

    print(f"\n可用的 run 目录（新→旧，来自 {runs_dir}）：")
    finals = {}
    for i, r in enumerate(runs):
        ckpts = _list_checkpoints(os.path.join(runs_dir, r))
        final = ckpts[-1] if ckpts else "(无 model_*.pt)"
        finals[r] = ckpts
        print(f"  [{i}] {r}   最终: {final}")

    idx = _ask_int("\n选 run 序号", default=0, max_val=len(runs) - 1)
    run_name = runs[idx]
    run_dir = os.path.join(runs_dir, run_name)

    ckpts = finals[run_name]
    if not ckpts:
        latest = _latest_checkpoint(runs_dir)
        print(f"[提示] {run_dir} 下没有 model_*.pt，改用最新 {latest}")
        return latest

    print(f"\n{run_dir} 下的 checkpoint：")
    for i, c in enumerate(ckpts):
        print(f"  [{i}] {c}")

    idx = _ask_int(
        "选 checkpoint 序号（默认最后一个=最终模型）",
        default=len(ckpts) - 1,
        max_val=len(ckpts) - 1,
    )
    return os.path.join(run_dir, ckpts[idx])


def _resolve_checkpoint(args, runs_dir: str) -> str:
    """把用户输入收敛成一条存在的绝对 checkpoint 路径，失败即早退。"""
    if args.checkpoint is not None:
        p = os.path.abspath(os.path.expanduser(args.checkpoint))
    elif args.run_dir is not None:
        run_dir = os.path.abspath(os.path.expanduser(args.run_dir))
        # 允许只给 run 名（相对该 stage 的 runs_dir）
        if not os.path.isdir(run_dir):
            run_dir = os.path.join(runs_dir, args.run_dir)
        ckpts = _list_checkpoints(run_dir)
        if not ckpts:
            raise SystemExit(f"[错误] run 目录里没有 model_*.pt: {run_dir}")
        p = os.path.join(run_dir, ckpts[-1])
    else:
        p = _pick_checkpoint_interactively(runs_dir)

    if not os.path.isfile(p):
        raise SystemExit(
            f"[错误] checkpoint 不存在: {p}\n"
            f"       检查路径拼写；相对路径会以当前工作目录为基准解析。"
        )
    return p


def _to_uint8_rgb(rgb):
    """相机 rgb 输出 → (H, W, 3) uint8。

    Isaac Lab 的 tiled camera 视版本可能给 uint8 [0,255] 或 float [0,1]，也可能带
    第 4 个 alpha 通道，这里统一收口（与 dump_fixed_camera_rgb.py 同一逻辑）。
    """
    import numpy as np

    arr = rgb.detach().cpu().numpy()
    if arr.ndim == 4:  # (B, H, W, C) → 取第 0 个环境
        arr = arr[0]
    if arr.shape[-1] == 4:  # 丢掉 alpha
        arr = arr[..., :3]
    if arr.dtype != np.uint8:
        # float 路径：>1.0 说明本来就是 0~255 的 float，不要再乘 255
        hi = float(np.nanmax(arr)) if arr.size else 0.0
        scale = 1.0 if hi > 1.5 else 255.0
        arr = np.clip(np.nan_to_num(arr) * scale, 0, 255).astype(np.uint8)
    return arr


def _draw_dots(out, px, color):
    """在 float 帧上把 (N,2) 像素坐标画成 3x3 色块。"""
    import numpy as np

    if px is None or px.size == 0:
        return out
    H, W = out.shape[:2]
    u = np.clip(px[:, 0].astype(np.int64), 0, W - 1)
    v = np.clip(px[:, 1].astype(np.int64), 0, H - 1)
    for du in (-1, 0, 1):
        for dv in (-1, 0, 1):
            uu = np.clip(u + du, 0, W - 1)
            vv = np.clip(v + dv, 0, H - 1)
            out[vv, uu, 0] = color[0]
            out[vv, uu, 1] = color[1]
            out[vv, uu, 2] = color[2]
    return out


def _overlay_occlusion(frame, dbg):
    """叠加遮挡可视化：半透明红=遮挡整体形状，红点=被挖候选点，绿点=存活候选点。"""
    import numpy as np

    if dbg is None:
        return frame
    H, W = frame.shape[:2]
    out = frame.copy().astype(np.float32)

    occ_map = dbg.get("occluded_map")
    if occ_map is not None and occ_map.size:
        occ_map = np.asarray(occ_map, dtype=bool)
        if occ_map.shape[0] == H and occ_map.shape[1] == W:
            blend = 0.35 * occ_map[..., None].astype(np.float32)
            red = np.zeros_like(out)
            red[..., 0] = 255.0
            out = out * (1.0 - blend) + red * blend

    out = _draw_dots(out, dbg.get("occluded_px"), (255, 0, 0))
    out = _draw_dots(out, dbg.get("kept_px"), (0, 255, 0))

    return np.clip(out, 0, 255).astype(np.uint8)


def main() -> None:
    import numpy as np
    import torch

    from configs.agents.rsl_rl_ppo_cfg import XArm7PickPointCloudPPORunnerCfg
    from configs.occlusion import parse_occlusion_center
    from configs.point_bridge_pointcloud import M_OBJ, M_WRIST, N_ROBOT, POINT_DIM
    from configs.xarm7_pick_pointcloud_env_cfg import (
        _OCCLUSION_VIS,
        _OCCLUSION_VIS_ENABLED,
        XArm7PickPointCloudPlayEnvCfg,
        disable_point_cloud_noise,
        make_pointcloud_camera_cfg,
        set_grasp_target_offset_cm,
        set_observation_stage,
        set_occlusion_stage2,
    )
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
    from model.pointnet_actor_critic import register_with_rsl_rl
    from rsl_rl.runners import OnPolicyRunner

    # ── 环境配置（镜像 training 的 play 路径）─────────────────────────────────
    cfg = XArm7PickPointCloudPlayEnvCfg()
    cfg.scene.num_envs = 1
    if args_cli.seed is not None:
        cfg.seed = args_cli.seed

    set_observation_stage(cfg, args_cli.stage, keep_appearance=True)
    if args_cli.stage == 2:
        # stage 2 遮挡管线：与训练脚本 train_pick_pointcloud_occluded.py 完全同路。
        # 即使 mode='none' 也调用 —— 训练时就走 point_bridge_point_cloud_occluded
        # 这个观测函数（内部 mode='none' 不注入过滤，行为与 stage2 基线一致），
        # 验证必须用同一个函数，否则观测与 checkpoint 对不上。
        set_occlusion_stage2(
            cfg,
            mode=args_cli.occlusion_mode,
            severity=args_cli.occlusion_severity,
            axis=args_cli.occlusion_axis,
            center=parse_occlusion_center(args_cli.occlusion_center),
        )
    # stage 1 会摘掉 camera_fixed（观测走 GT 采点、不读相机）。录像需要画面，
    # 重新装回来 —— 它只供录像，观测函数 point_bridge_point_cloud_gt 不碰它。
    if cfg.scene.camera_fixed is None:
        cfg.scene.camera_fixed = make_pointcloud_camera_cfg()
    else:
        # stage 2/10 会把 camera_fixed 的 data_types 摘到只剩深度+分割（训练路径
        # 没有 RGB 消费者，省显存）。录像要读 camera.data.output["rgb"]，这里把
        # RGB 补回来 —— 只多一路渲染缓冲，观测函数反投影深度不读 rgb，不影响策略。
        _dt = cfg.scene.camera_fixed.data_types
        if _dt is not None and "rgb" not in _dt:
            cfg.scene.camera_fixed.data_types = list(_dt) + ["rgb"]
    disable_point_cloud_noise(cfg)
    set_grasp_target_offset_cm(cfg, args_cli.grasp_offset_cm)

    control_dt = cfg.sim.dt * cfg.decimation  # 0.02s = 50Hz
    record_every = max(1, int(round((1.0 / control_dt) / args_cli.video_fps)))

    # ── 输出目录（带上 run 名 + checkpoint 名，不同验证不再混淆）─────────────
    from datetime import datetime

    if args_cli.out_dir is None:
        run_name = os.path.basename(os.path.dirname(args_cli.checkpoint))
        ckpt_name = os.path.splitext(os.path.basename(args_cli.checkpoint))[0]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        args_cli.out_dir = os.path.join(
            _AUTODL_TMP, "validation",
            f"xarm7_pick_pointcloud_stage{args_cli.stage}_{run_name}_{ckpt_name}_{stamp}",
        )
    os.makedirs(args_cli.out_dir, exist_ok=True)
    video_path = os.path.join(args_cli.out_dir, "rollout.mp4")

    # ── 建环境 + 载入策略（与 training 的 play 完全同路）─────────────────────
    env = ManagerBasedRLEnv(cfg=cfg)
    base_env = env  # 录像从这里拿相机
    env = RslRlVecEnvWrapper(env)
    camera = base_env.scene["camera_fixed"]

    # 遮挡可视化：把 base_env 注册进遮挡调试容器，观测函数才会顺手记录 keep 掩码。
    # 纯只读旁路，不影响点云观测本身。
    if args_cli.debug_vis_occlusion:
        _OCCLUSION_VIS_ENABLED.add(id(base_env))
    occ_debug_holder = {"dbg": None}  # 每步更新：遮挡调试信息（实心区域 + 被挖/存活候选点）

    agent_cfg = XArm7PickPointCloudPPORunnerCfg()
    agent_cfg.policy.num_object_points = M_OBJ
    agent_cfg.policy.num_robot_points = N_ROBOT
    agent_cfg.policy.point_dim = POINT_DIM
    # 只有 stage 10 有腕部分支；其余阶段保持 0（网络结构与 checkpoint 逐位兼容）
    agent_cfg.policy.num_wrist_points = M_WRIST if args_cli.stage == 10 else 0

    register_with_rsl_rl()
    runner = OnPolicyRunner(
        env, agent_cfg.to_dict(), log_dir=args_cli.out_dir, device=env.device
    )
    runner.load(args_cli.checkpoint)
    policy = runner.get_inference_policy(device=env.device)

    # ── 录像 writer ───────────────────────────────────────────────────────────
    import imageio.v2 as imageio

    writer = None
    frames_dir = None
    try:
        writer = imageio.get_writer(video_path, fps=args_cli.video_fps)
    except Exception as exc:  # noqa: BLE001 —— ffmpeg 后端缺失时退级，不阻断验证
        print(f"[录像] imageio 打不开 mp4（{type(exc).__name__}: {exc}），退级保存 PNG 帧序列")
        frames_dir = os.path.join(args_cli.out_dir, "frames")
        os.makedirs(frames_dir, exist_ok=True)

    def _read_occ_debug():
        """从观测函数暂存的调试信息里解出遮挡可视化的三样东西。

        返回 dict：occluded_map (H,W) bool 稠密遮挡区域、occluded_px (N,2) 被挖
        候选点 (u,v)、kept_px (M,2) 存活候选点 (u,v)。未开开关或无遮挡时返回 None。
        """
        if not args_cli.debug_vis_occlusion:
            return None
        dbg = _OCCLUSION_VIS.get(id(base_env))
        if not dbg or "keep" not in dbg:
            return None
        keep = dbg["keep"].detach().cpu().numpy()          # (B, cap)
        cand_idx = dbg["cand_idx"].detach().cpu().numpy()  # (B, cap)
        W = int(dbg["W"])
        keep0 = keep[0].astype(bool)                       # (cap,)
        idx = cand_idx[0]
        u = idx % W
        v = idx // W
        out = {
            "occluded_px": np.stack([u[~keep0], v[~keep0]], axis=-1),
            "kept_px": np.stack([u[keep0], v[keep0]], axis=-1),
        }
        occ_map = dbg.get("occluded_map")
        if occ_map is not None:
            out["occluded_map"] = occ_map.detach().cpu().numpy()
        return out

    def _record(frame_idx: int) -> int:
        rgb = camera.data.output.get("rgb")
        if rgb is None:
            return frame_idx
        frame = _to_uint8_rgb(rgb)
        frame = _overlay_occlusion(frame, occ_debug_holder["dbg"])
        if writer is not None:
            writer.append_data(frame)
        else:
            imageio.imwrite(
                os.path.join(frames_dir, f"frame_{frame_idx:06d}.png"), frame
            )
        return frame_idx + 1

    print("=" * 78)
    print("[验证] stage {} 点云策略回放 + 录像".format(args_cli.stage))
    print(f"[checkpoint] {args_cli.checkpoint}")
    print(f"[观测]       物体点 {M_OBJ}x{POINT_DIM} + 夹爪点 {N_ROBOT}x{POINT_DIM}"
          f" + 关节角 7 = {M_OBJ * POINT_DIM + N_ROBOT * POINT_DIM + 7} 维")
    if args_cli.stage == 2:
        if args_cli.occlusion_mode != "none" and args_cli.occlusion_severity > 0.0:
            if args_cli.occlusion_mode == "halfspace":
                print(
                    f"[遮挡]       halfspace  axis={args_cli.occlusion_axis}  "
                    f"severity={args_cli.occlusion_severity:.2f}"
                )
            elif args_cli.occlusion_mode == "sphere":
                print(
                    f"[遮挡]       sphere  center={args_cli.occlusion_center}  "
                    f"severity={args_cli.occlusion_severity:.2f}"
                )
            elif args_cli.occlusion_mode == "random_sphere":
                print(
                    f"[遮挡]       random_sphere  severity={args_cli.occlusion_severity:.2f}"
                    f"（圆形，球心每 episode 随机）"
                )
            else:
                print(
                    f"[遮挡]       random_box  severity={args_cli.occlusion_severity:.2f}"
                    f"（方形，盒心+朝向每 episode 随机）"
                )
        else:
            print("[遮挡]       无（stage2 baseline）")
    print(f"[抓取目标]   工件正上方 {args_cli.grasp_offset_cm:.1f} cm")
    if args_cli.debug_vis_occlusion:
        if (
            args_cli.stage == 2
            and args_cli.occlusion_mode != "none"
            and args_cli.occlusion_severity > 0.0
        ):
            print("[可视化]    半透明红=遮挡形状，红点=被挖候选点，绿点=存活候选点")
        else:
            print("[可视化]    已开 --debug_vis_occlusion，但当前无遮挡，画面不会标红")
    print(f"[录像]       {video_path}")
    print(f"              {args_cli.video_fps} fps，每 {record_every} 控制步取一帧")
    print(f"[计划]       录制 {args_cli.episodes} 个 episode")
    print("=" * 78)

    obs, _ = env.reset()
    frame_idx = 0
    occ_debug_holder["dbg"] = _read_occ_debug()
    frame_idx = _record(frame_idx)  # 首帧（reset 后的初始场景）

    episode_idx = 0
    step_count = 0
    ep_steps = 0
    ep_reward = 0.0

    while simulation_app.is_running() and episode_idx < args_cli.episodes:
        with torch.no_grad():
            actions = policy(obs)
        obs, rewards, dones, _ = env.step(actions)

        ep_steps += 1
        step_count += 1
        ep_reward += float(rewards[0])

        occ_debug_holder["dbg"] = _read_occ_debug()

        if step_count % record_every == 0:
            frame_idx = _record(frame_idx)

        if bool(dones[0]):
            print(
                f"[episode {episode_idx + 1}/{args_cli.episodes}] "
                f"steps={ep_steps}  reward={ep_reward:.2f}  累计帧={frame_idx}"
            )
            episode_idx += 1
            ep_steps = 0
            ep_reward = 0.0
            if episode_idx < args_cli.episodes:
                obs, _ = env.reset()
                frame_idx = _record(frame_idx)

    # ── 收尾 ──────────────────────────────────────────────────────────────────
    if writer is not None:
        writer.close()
        print(f"[录像] 已写入 {video_path}  ({frame_idx} 帧，"
              f"约 {frame_idx / max(1, args_cli.video_fps):.1f}s)")
    else:
        print(f"[录像] 已保存 {frame_idx} 帧 PNG → {frames_dir}")
    print(f"[完成] 输出目录: {args_cli.out_dir}")

    env.close()


if __name__ == "__main__":
    args_cli = _build_parser().parse_args()

    # 先解析 checkpoint（交互式或命令行），失败就早退，别白启动重 AppLauncher。
    fully_interactive = args_cli.checkpoint is None and args_cli.run_dir is None
    runs_dir = _runs_dir_for_stage(
        args_cli.stage,
        occ_mode=args_cli.occlusion_mode,
        occ_severity=args_cli.occlusion_severity,
        occ_axis=args_cli.occlusion_axis,
    )
    args_cli.checkpoint = _resolve_checkpoint(args_cli, runs_dir)
    if fully_interactive:
        args_cli.seed = _ask_seed(default=args_cli.seed)

    # 相机是录像来源；程序化桌面 PreviewSurface(MDL) 只在渲染型 Kit experience 下注册。
    args_cli.enable_cameras = True
    args_cli.num_envs = 1
    # 服务器无显示，固定离屏渲染（与训练 --headless 一致）
    if not getattr(args_cli, "headless", False):
        args_cli.headless = True

    app_launcher = AppLauncher(args_cli)
    simulation_app = app_launcher.app

    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback

        traceback.print_exc()
    finally:
        simulation_app.close()
