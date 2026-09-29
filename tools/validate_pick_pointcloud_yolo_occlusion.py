#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""M1 random_box 遮挡可视化 —— 把被挖掉的区域/候选点标红落盘。

**增量式验证脚本：不修改任何既有文件。** 只复用 ``xarm7_pick_pointcloud_env_cfg.py``
里早就埋好的调试旁路 ``_OCCLUSION_VIS_ENABLED / _OCCLUSION_VIS``：在 env 构造之后
运行时把 ``id(env)`` 加进开关集合，M1 观测函数就会在每步把 ``keep`` 掩码、候选像素
索引和稠密遮挡图抄出来（该旁路关闭时零开销、零行为差异，不开启时与训练逐位一致）。

观测管线与 ``train_pick_pointcloud_yolo_occluded.py`` 完全同源：
``set_observation_stage`` + ``set_occlusion_yolo`` 同一个 setter、同一份 YOLO 权重。
帧上标注（画法沿用 stage-1 验证脚本）：

  * 半透明红区域 = 稠密遮挡图（遮挡阴影的整体形状与位置）
  * 红点 = 被挖掉的候选点（keep=False）
  * 绿点 = 存活候选点

用法::

    # 固定 50% 遮挡（stage 3，每步跑 YOLO），4 env 跑 60 步，零动作（机械臂不动，
    # 纯看阴影形状随工件位姿变化 —— 遮挡与策略无关，不加载 checkpoint 也完全成立）
    python tools/validate_pick_pointcloud_yolo_occlusion.py \
        --headless --num_envs 4 --steps 60 --occlusion_severity 0.5 \
        --out logs/occ_vis_rbox50

    # 加载 checkpoint 用训练出的策略走（看夹爪接近、抓取过程中的遮挡变化）
    python tools/validate_pick_pointcloud_yolo_occlusion.py \
        --headless --num_envs 4 --steps 60 --occlusion_severity 0.5 \
        --checkpoint logs/xarm7_pick_pointcloud_stage3_occ_rbox_50/<run>/model_19.pt \
        --out logs/occ_vis_rbox50

    # stage 4（每 5 步缓存掩码）+ 每 episode 范围遮挡
    python tools/validate_pick_pointcloud_yolo_occlusion.py \
        --headless --num_envs 4 --steps 60 --yolo_step 5 \
        --occlusion_severity_min 0.2 --occlusion_severity_max 0.6 --out logs/occ_vis_r2060

输出（--out 下）：
    frames/frame_<step>_env<i>.png   每步每 env 一张叠加帧
    env_<i>.mp4                      同上的连续视频（imageio 打不开时退级 PNG）
    occ_stats.csv                    每步每 env：被挖候选点比例、稠密遮挡像素比例、YOLO 检出

注意：
  * 需要与训练相同的启动环境（isaaclab python + ``YOLO_DEPS_DIR``）。
  * ``--yolo_step`` 应与要加载 checkpoint 的 stage 一致；只看遮挡形状则无此约束。
  * num_envs > 1 时任一 env 达 episode 末即全部 reset（复用训练回放循环的语义）。
"""

import argparse
import csv
import os
import sys

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_CURRENT_DIR)
sys.path.insert(0, _PROJECT_DIR)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="M1 random_box 遮挡可视化 —— 被挖区域/候选点标红落盘"
)
parser.add_argument("--num_envs", type=int, default=4)
parser.add_argument("--steps", type=int, default=60)
parser.add_argument("--checkpoint", type=str, default=None,
                    help="可选：加载训练 checkpoint 用策略驱动（不给则零动作）。")
parser.add_argument("--seed", type=int, default=None)
parser.add_argument(
    "--yolo_step", type=int, default=None,
    help="不给 = stage 3（每步跑 YOLO）；给 N = stage 4（每 N 步缓存掩码）。"
)
parser.add_argument(
    "--occlusion_severity", type=float, default=0.5,
    help="random_box 固定删除比例 [0,1]；0 = 无遮挡。",
)
parser.add_argument("--occlusion_severity_min", type=float, default=None,
                    help="与 --occlusion_severity_max 成对给出：每 episode U[min,max]。")
parser.add_argument("--occlusion_severity_max", type=float, default=None)
parser.add_argument("--out", type=str, default=None,
                    help="输出目录；默认 logs/occ_vis_<标签>_<时间戳>。")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
# 相机是观测来源，headless 也要离屏渲染 RGB
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch

from configs.xarm7_pick_pointcloud_env_cfg import (
    XArm7PickPointCloudEnvCfg,
    _OCCLUSION_VIS,
    _OCCLUSION_VIS_ENABLED,
    set_observation_stage,
    set_occlusion_yolo,
)
from configs.point_bridge_pointcloud import M_OBJ, N_ROBOT, POINT_DIM
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper

# 阶段与遮挡参数校验（与训练脚本同一套规则）
_STAGE = 3 if args_cli.yolo_step is None else 4
if _STAGE == 4 and args_cli.yolo_step < 1:
    raise SystemExit(f"--yolo_step 必须 >= 1；给的是 {args_cli.yolo_step}")
if not (0.0 <= args_cli.occlusion_severity <= 1.0):
    raise SystemExit(f"--occlusion_severity 必须在 [0,1]，got {args_cli.occlusion_severity}")
severity_range = None
if args_cli.occlusion_severity_min is not None or args_cli.occlusion_severity_max is not None:
    if (args_cli.occlusion_severity_min is None
            or args_cli.occlusion_severity_max is None):
        raise SystemExit("--occlusion_severity_min/max 必须成对给出")
    if not (0.0 <= args_cli.occlusion_severity_min
            <= args_cli.occlusion_severity_max <= 1.0):
        raise SystemExit(
            f"--occlusion_severity_min/max 必须满足 0 <= min <= max <= 1；"
            f"got {args_cli.occlusion_severity_min} / {args_cli.occlusion_severity_max}"
        )
    severity_range = (args_cli.occlusion_severity_min, args_cli.occlusion_severity_max)


def _to_uint8_rgb(arr):
    """相机 rgb 输出 → (H, W, 3) uint8。

    与 stage-1 验证脚本同一逻辑：兼容 uint8/float 与可能的 alpha 通道。
    """
    import numpy as np

    if hasattr(arr, "detach"):  # torch 张量 → numpy
        arr = arr.detach().cpu().numpy()
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


def _read_occ_debug(base_env, env_i):
    """从观测函数暂存的调试信息解出第 env_i 个环境的遮挡可视化三样东西。

    返回 (frame_annot, row)：
      frame_annot: dict 可喂给 _overlay_occlusion（无遮挡/旁路未开时 None）
      row:        list 供 stats CSV（kept 比例、遮挡图比例；缺省 None）
    """
    import numpy as np

    dbg = _OCCLUSION_VIS.get(id(base_env))
    if not dbg or "keep" not in dbg:
        return None, None
    keep = dbg["keep"].detach().cpu().numpy()          # (B, cap)
    cand_idx = dbg["cand_idx"].detach().cpu().numpy()  # (B, cap)
    W = int(dbg["W"])
    H = int(dbg["H"])
    keep_i = keep[env_i].astype(bool)
    idx = cand_idx[env_i]
    u = idx % W
    v = idx // W
    out = {
        "occluded_px": np.stack([u[~keep_i], v[~keep_i]], axis=-1),
        "kept_px": np.stack([u[keep_i], v[keep_i]], axis=-1),
    }
    occ_map = dbg.get("occluded_map")
    if occ_map is not None:
        out["occluded_map"] = occ_map.detach().cpu().numpy()[env_i]
    kept_ratio = float(keep_i.mean())
    occ_ratio = float(out["occluded_map"].astype(np.float32).mean()) if "occluded_map" in out else None
    return out, [kept_ratio, occ_ratio]


def _overlay_occlusion(frame, annot):
    """叠加遮挡可视化：半透明红=遮挡整体形状，红点=被挖候选点，绿点=存活候选点。"""
    import numpy as np

    if annot is None:
        return frame
    H, W = frame.shape[:2]
    out = frame.copy().astype(np.float32)

    occ_map = annot.get("occluded_map")
    if occ_map is not None and occ_map.size:
        occ_map = np.asarray(occ_map, dtype=bool)
        if occ_map.shape[0] == H and occ_map.shape[1] == W:
            blend = 0.35 * occ_map[..., None].astype(np.float32)
            red = np.zeros_like(out)
            red[..., 0] = 255.0
            out = out * (1.0 - blend) + red * blend

    out = _draw_dots(out, annot.get("occluded_px"), (255, 0, 0))
    out = _draw_dots(out, annot.get("kept_px"), (0, 255, 0))

    return np.clip(out, 0, 255).astype(np.uint8)


def main():
    import numpy as np

    from datetime import datetime

    from configs.agents.rsl_rl_ppo_cfg import XArm7PickPointCloudPPORunnerCfg
    from model.pointnet_actor_critic import register_with_rsl_rl

    cfg = XArm7PickPointCloudEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    if args_cli.seed is not None:
        cfg.seed = args_cli.seed

    # 与训练脚本同源：先切 stage（替换观测函数），再注入遮挡（替换观测函数为带遮挡版）
    set_observation_stage(cfg, _STAGE, keep_appearance=False)
    set_occlusion_yolo(
        cfg,
        severity=args_cli.occlusion_severity,
        severity_range=severity_range,
        yolo_step=args_cli.yolo_step,
    )
    term = cfg.observations.policy.point_cloud

    base_env = ManagerBasedRLEnv(cfg=cfg)
    # 运行时打开调试旁路（构造后、任何 reset 之前）。id(env) 与观测函数里的
    # key=id(env) 一致 —— 不修改任何既有文件，旁路关闭时行为与训练逐位一致。
    _OCCLUSION_VIS_ENABLED.add(id(base_env))
    env = RslRlVecEnvWrapper(base_env)
    camera = base_env.scene["camera_fixed"]

    # ── 输出目录 ──────────────────────────────────────────────
    occ_tag = ""
    if severity_range is not None:
        lo, hi = severity_range
        occ_tag = f"r{int(round(lo * 100))}-{int(round(hi * 100))}"
    else:
        occ_tag = f"rbox{int(round(args_cli.occlusion_severity * 100))}"
    if args_cli.out is None:
        args_cli.out = os.path.join(
            _PROJECT_DIR, "logs",
            f"occ_vis_stage{_STAGE}_{occ_tag}_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}",
        )
    frames_dir = os.path.join(args_cli.out, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    print("=" * 78)
    print(f"[可视化] M1 random_box 遮挡 —— stage {_STAGE}（YOLO 掩码）")
    print(f"[Occ]    severity="
          + (f"U[{severity_range[0]:.2f},{severity_range[1]:.2f}]" if severity_range
             else f"{args_cli.occlusion_severity:.2f}")
          + f"   YOLO weights={term.params['yolo_weights']}")
    print(f"[Env]    num_envs={args_cli.num_envs}  steps={args_cli.steps}  "
          f"checkpoint={args_cli.checkpoint or '无（零动作）'}")
    print(f"[输出]   {args_cli.out}")

    # ── 可选：加载 checkpoint 用策略驱动（不动则零动作）──────────
    policy = None
    if args_cli.checkpoint is not None:
        ckpt = args_cli.checkpoint
        if not os.path.exists(ckpt):
            raise SystemExit(f"[错误] checkpoint 不存在: {ckpt}")
        agent_cfg = XArm7PickPointCloudPPORunnerCfg()
        agent_cfg.policy.num_object_points = M_OBJ
        agent_cfg.policy.num_robot_points = N_ROBOT
        agent_cfg.policy.point_dim = POINT_DIM
        register_with_rsl_rl()
        from rsl_rl.runners import OnPolicyRunner

        runner = OnPolicyRunner(
            env, agent_cfg.to_dict(),
            log_dir=os.path.join(args_cli.out, "runner_scratch"),
            device=env.device,
        )
        runner.load(ckpt)
        policy = runner.get_inference_policy(device=env.device)
        print(f"[策略]   已加载 {ckpt}")

    # ── 录像 writer（mp4 失败退级 PNG 帧序列）──────────────────
    import imageio.v2 as imageio

    writers = {}
    for i in range(args_cli.num_envs):
        try:
            writers[i] = imageio.get_writer(
                os.path.join(args_cli.out, f"env_{i}.mp4"), fps=10
            )
        except Exception as exc:  # imageio 无 ffmpeg 等
            writers[i] = None
            print(f"[录像] imageio 打不开 env_{i}.mp4（{type(exc).__name__}），退级 PNG 帧序列")

    csv_path = os.path.join(args_cli.out, "occ_stats.csv")
    csv_file = open(csv_path, "w", newline="")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(
        ["step", "env", "kept_ratio", "occluded_map_ratio", "yolo_detect_ratio"]
    )

    def _dump_frames(step: int) -> None:
        rgb = camera.data.output.get("rgb")
        if rgb is None:
            return
        arr = rgb.detach().cpu().numpy()                      # (B, H, W, C)
        yolo_ratio = None
        for i in range(args_cli.num_envs):
            frame = _to_uint8_rgb(arr[i])
            annot, row = _read_occ_debug(base_env, i)
            frame = _overlay_occlusion(frame, annot)
            if writers[i] is not None:
                writers[i].append_data(frame)
            else:
                imageio.imwrite(
                    os.path.join(frames_dir, f"frame_{step:06d}_env{i}.png"), frame
                )
            if i == 0:
                yolo_ratio = base_env.extras.get("yolo_detect_ratio")
                yolo_ratio = float(yolo_ratio) if yolo_ratio is not None else None
            kept, occ = (row if row is not None else (None, None))
            csv_writer.writerow([step, i, kept, occ, yolo_ratio])
        csv_file.flush()

    # ── 主循环：与训练回放同构（dones 触发整批 reset）──────────
    obs, _ = env.reset()
    try:
        for step in range(args_cli.steps):
            with torch.no_grad():
                if policy is not None:
                    actions = policy(obs)
                else:
                    actions = torch.zeros((args_cli.num_envs, 7), device=env.device)
            obs, _, dones, _ = env.step(actions)
            _dump_frames(step)
            if bool(dones.any()):
                obs, _ = env.reset()
        print(f"[完成]   {args_cli.steps} 步已落盘 → {args_cli.out}")
    finally:
        for w in writers.values():
            if w is not None:
                w.close()
        csv_file.close()
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
