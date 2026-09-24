#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""跑 privileged 策略（stage -1），过程中每 1 秒存一张固定相机 RGB，每 10 秒 reset 一轮。

与 ``collect_pose_dataset.py`` 的区别：那个是**静态**采集 —— reset、等渲染收敛、
存一张、再 reset，工件位姿由随机化决定，机械臂始终在初始位。这个脚本存的是
**策略执行过程中**的画面，同一轮里 10 张图对应机械臂从初始位逼近工件的 10 个不同
时刻。要的是"运动中的样子"，那就必须让策略真的动起来。

**为什么用 stage -1 的策略来产图**

model_3800.pt 是 35 维纯 GT 状态的 privileged 策略，观测里没有任何像素。所以相机
在这里是个纯旁观者 —— 它拍什么都不会反过来影响策略动作。这正是想要的：策略给出
一条**合理的、成功率高的**轨迹，相机沿途记录，得到的图天然覆盖"好轨迹上会遇到的
视觉状态"，而不是随机乱挥的姿势。

**stage -1 会把相机摘掉，必须补回来**

``set_privileged_baseline`` 里有一句 ``env_cfg.scene.camera_fixed = None`` ——
训练时没有任何消费者读相机，摘掉能省下每步 640x480xN_env 的渲染。但这里就是要读
它，所以调完 ``set_observation_stage`` 之后必须把 camera_fixed 装回去。装回去还
不够：``keep_appearance=False`` 会同时摘掉桌面贴图和光照随机化，拍出来是一张没有
贴图、光照恒定的桌子，和 stage 1/2 及真机都对不上。所以 ``keep_appearance=True``。

**时序**

sim dt 0.01 x decimation 2 = 0.02s，即 50Hz。故 1 秒 = 50 步，10 秒 = 500 步，
每轮 10 张图。环境自身 episode_length_s=24.0，比 10 秒长 —— 换句话说 10 秒这个
边界只能靠脚本主动 reset，超时截断不会替你做。反过来，若策略提前把 episode 结束
了（成功/失败终止），本轮就不足 10 张，metadata 里的 ``truncated_early`` 会标出来。

用法::

    ~/IsaacLab/isaaclab.sh -p scripts/train/collect_policy_rgb_episodes.py \\
        --checkpoint logs/xarm7_pick_privileged_base/2026-08-14_16-38-55/model_3800.pt \\
        --episodes 50 --num_envs 8 --headless

输出::

    out/
    ├── frames/001.png 002.png 003.png ...   全部平铺在一个文件夹，连号
    ├── index.csv            每张图 → 轮次/时刻/工件位姿（人能直接翻的那份）
    ├── summary.npz          每帧的工件位姿 / EE 位姿 / 关节角 / 时刻
    └── dataset_info.json    相机内外参 + 采集参数

图片是**跨轮次连号**的：001~010 是第一轮的 10 个时刻，011~020 是第二轮，以此类推。
但轮次被提前终止时那一轮不足 10 张，连号不会留空 —— 所以别用 "编号//10" 反推轮次，
要看 index.csv / summary.npz 里的 ``episode`` 列。
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))
sys.path.insert(0, _PROJECT_DIR)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="策略执行过程中的固定相机 RGB 采集")
parser.add_argument("--checkpoint", type=str,
                    default=os.path.join(
                        _PROJECT_DIR, "logs", "xarm7_pick_privileged_base",
                        "2026-08-14_16-38-55", "model_3800.pt"),
                    help="stage -1 的 privileged checkpoint（35 维观测）")
parser.add_argument("--episodes", type=int, default=50,
                    help="总轮数（所有 env 合计）。每轮产出 episode_s/interval_s 张图。")
parser.add_argument("--num_envs", type=int, default=8,
                    help="并行环境数。每次 reset 同时开 num_envs 轮，各 env 的工件"
                         "位姿和光照独立。")
parser.add_argument("--episode_s", type=float, default=10.0, help="每轮时长（秒）")
parser.add_argument("--interval_s", type=float, default=1.0, help="存图间隔（秒）")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--env_spacing", type=float, default=6.0,
                    help="env 间距。默认 6.0 确保邻桌不入画（2.5 时会出现在画面边缘）。")
parser.add_argument("--warmup", type=int, default=4,
                    help="reset 后空步数，等 RTX 把新光照渲染收敛。设 0 会拿到上一轮"
                         "的光照，图像和标签对不上。这几步不计入 10 秒。")
parser.add_argument("--out", type=str,
                    default=os.path.join(_PROJECT_DIR, "logs", "policy_rgb_episodes"))
parser.add_argument("--io_workers", type=int, default=8)
parser.add_argument("--rgb_only", action="store_true", default=True,
                    help="只渲染 RGB，关掉深度与分割两路缓冲。显存降到约 1/3，"
                         "是 num_envs 能开多大的决定性因素。默认开启。")
parser.add_argument("--keep_all_data_types", dest="rgb_only", action="store_false",
                    help="保留深度/分割（本脚本用不到，排查渲染问题时才需要）")

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

# 要读相机就必须开渲染，和 train_pick_pointcloud.py 一样强制打开。
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

from isaaclab.envs import ManagerBasedRLEnv  # noqa: E402
from isaaclab.utils.math import subtract_frame_transforms  # noqa: E402
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper  # noqa: E402
from rsl_rl.runners import OnPolicyRunner  # noqa: E402

from configs.xarm7_pick_pointcloud_env_cfg import (  # noqa: E402
    CANONICAL_HEIGHT,
    CANONICAL_WIDTH,
    XArm7PickPointCloudEnvCfg,
    canonical_intrinsic_matrix,
    make_pointcloud_camera_cfg,
    set_observation_stage,
    set_success_termination_enabled,
)
from model.pointnet_actor_critic import register_with_rsl_rl  # noqa: E402


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


def _write_png(path: str, rgb: np.ndarray) -> None:
    """写一张图。目录由主流程一次性建好，这里不再 makedirs —— 平铺成一个文件夹后
    每张图都要建一次目录纯属浪费，而且多线程并发 makedirs 没有意义。"""
    cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))


def _build_env_cfg():
    """stage -1 的环境 + 补回固定相机。

    这里刻意**不用** ``XArm7PickPointCloudPlayEnvCfg``：那一版把 num_envs 钉死成 1，
    而采集要靠并行 env 提速。观测噪声在下面单独关掉，不依赖 Play 版本。
    """
    cfg = XArm7PickPointCloudEnvCfg()
    cfg.scene.num_envs = args_cli.num_envs
    cfg.scene.env_spacing = args_cli.env_spacing
    cfg.seed = args_cli.seed

    # keep_appearance=True：桌面贴图与光照随机化要保留 —— 采的就是画面，
    # 摘掉它们拍出来的是一张没贴图、光照恒定的桌子，和 stage 1/2 及真机都对不上。
    set_observation_stage(cfg, -1, keep_appearance=True)
    # 与训练时保持一致：checkpoint 是在无成功终止的设定下训的，这里打开会让轨迹
    # 提前结束，10 秒轮次凑不满。
    set_success_termination_enabled(cfg, False)

    # set_privileged_baseline 里 `env_cfg.scene.camera_fixed = None`（训练时没有
    # 消费者读相机，摘掉省渲染）。这里就是要读，所以装回去。必须在
    # set_observation_stage **之后**，否则会被它清成 None。
    cfg.scene.camera_fixed = make_pointcloud_camera_cfg()
    if args_cli.rgb_only:
        # 相机默认还开着 distance_to_image_plane 和 instance_id_segmentation_fast，
        # 各占一份和 RGB 同量级的 GPU 缓冲。只存 RGB 就关掉，显存降到约 1/3。
        cfg.scene.camera_fixed.data_types = ["rgb"]

    # invalidate_instance_lut 保持 None（set_privileged_baseline 已置空）：它服务的
    # 是分割 LUT 缓存，而 rgb_only 下根本不出分割图。策略也不读点云，不需要它。

    # 观测噪声关掉。策略吃的是 GT 状态，加噪只会让轨迹变差，而这里要的是"好轨迹
    # 上的画面"。
    cfg.observations.policy.enable_corruption = False
    return cfg


def main() -> None:
    torch.manual_seed(args_cli.seed)
    np.random.seed(args_cli.seed)
    # 域随机化走 configs 里的 `random` 模块，必须单独播种，否则 --seed 对光照和
    # 工件位姿不起作用。
    import random as _py_random

    _py_random.seed(args_cli.seed)

    if not os.path.exists(args_cli.checkpoint):
        raise SystemExit(f"[错误] 找不到 checkpoint: {args_cli.checkpoint}")

    cfg = _build_env_cfg()
    base_env = ManagerBasedRLEnv(cfg=cfg)
    env = RslRlVecEnvWrapper(base_env)

    camera = base_env.scene["camera_fixed"]
    robot = base_env.scene["robot"]
    obj = base_env.scene["object"]
    num_envs = base_env.num_envs

    # 控制周期。不写死 0.02 —— 从配置推，改了 decimation 这里自动跟上。
    control_dt = base_env.cfg.sim.dt * base_env.cfg.decimation
    steps_per_shot = int(round(args_cli.interval_s / control_dt))
    steps_per_ep = int(round(args_cli.episode_s / control_dt))
    shots_per_ep = steps_per_ep // steps_per_shot
    num_rounds = (args_cli.episodes + num_envs - 1) // num_envs
    total_eps = num_rounds * num_envs

    # 文件名位宽。至少 3 位（001.png），图多了就加宽到刚好装得下总数。
    # 位宽不够时 %03d 会自然溢出成 4 位，文件名不会撞，但按字符串排序时
    # "1000" 会排到 "999" 前面 —— 一次性按总数定宽就没这个问题。
    total_shots = total_eps * shots_per_ep
    name_digits = max(3, len(str(total_shots)))

    # stage -1 用 privileged base 的 runner 配置，与训练时同一份，否则网络结构
    # 对不上、load 会报 shape mismatch。
    from configs.agents.rsl_rl_ppo_cfg import XArm7PickPrivilegedBasePPORunnerCfg

    agent_cfg = XArm7PickPrivilegedBasePPORunnerCfg()
    # OnPolicyRunner 用 eval(class_name) 在它自己的模块 globals 里解析策略类，
    # 必须在构造 runner 之前注入。
    register_with_rsl_rl()
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=None, device=env.device)
    runner.load(args_cli.checkpoint)
    # 返回的是 ActorCritic.act_inference，它内部会先过 actor_obs_normalizer 再进
    # 网络。model_3800.pt 里确实存了这套 35 维的 mean/var，靠上面的 runner.load
    # 恢复 —— 所以顺序不能反：先 load 再取 policy。反过来拿到的是未加载的归一化
    # 统计量，动作整体跑偏，而且不报错。
    policy = runner.get_inference_policy(device=env.device)

    frames_dir = os.path.join(args_cli.out, "frames")
    os.makedirs(frames_dir, exist_ok=True)

    print("=" * 78)
    print("策略执行过程 RGB 采集（stage -1 privileged）")
    print(f"  checkpoint    : {args_cli.checkpoint}")
    print(f"  输出目录      : {args_cli.out}")
    print(f"  并行环境数    : {num_envs}   env_spacing={args_cli.env_spacing}")
    print(f"  控制周期      : {control_dt*1000:.0f}ms ({1/control_dt:.0f}Hz)")
    print(f"  每轮          : {args_cli.episode_s}s = {steps_per_ep} 步，"
          f"每 {args_cli.interval_s}s({steps_per_shot} 步) 存一张 → {shots_per_ep} 张/轮")
    print(f"  轮数 x 每轮   : {num_rounds} x {num_envs} = {total_eps} 轮，"
          f"共 {total_eps * shots_per_ep} 张")
    print(f"  图像          : {CANONICAL_WIDTH}x{CANONICAL_HEIGHT}  camera_fixed")
    print(f"  rgb_only      : {args_cli.rgb_only}")
    print(f"  seed          : {args_cli.seed}")
    print("=" * 78)

    pool = ThreadPoolExecutor(max_workers=args_cli.io_workers)
    futures = []
    rec_name = []
    rec_ep, rec_env, rec_shot, rec_t = [], [], [], []
    rec_obj_base, rec_ee_base, rec_qpos, rec_done = [], [], [], []
    t0 = time.time()
    n_img = 0

    for rnd in range(num_rounds):
        obs, _ = env.reset()
        # reset 后光照属性刚写进 USD，RTX 要几帧才收敛。这几步用策略动作而不是零
        # 动作：零动作会让机械臂在 t=0 之前就偏离初始位，第一张图和后面对不上。
        # 但它们不计入 10 秒 —— 计时从 warmup 之后开始。
        for _ in range(args_cli.warmup):
            with torch.no_grad():
                obs, _, _, _ = env.step(policy(obs))

        # 本轮里哪些 env 已经被环境自己终止了。终止后 Isaac Lab 会自动 reset 那个
        # env，画面跳回初始位 —— 再存图就把两轮混进同一个 ep 目录了。所以标记之后
        # 不再为它存图。
        ep_alive = np.ones(num_envs, dtype=bool)
        shots_taken = np.zeros(num_envs, dtype=np.int32)

        for step in range(steps_per_ep):
            with torch.no_grad():
                actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

            # 先存图再处理 dones：本步的画面对应的是 step 之后的状态，而 dones 说
            # 的是"这一步之后该 env 被重置了"。顺序反了会把重置后的画面记成本轮的。
            if (step + 1) % steps_per_shot == 0:
                shot_idx = (step + 1) // steps_per_shot - 1
                sim_t = (step + 1) * control_dt
                rgb_batch = _to_uint8_rgb(camera.data.output["rgb"])

                # 工件与 EE 位姿 → 基座系。基座有 -90° 偏航，必须做完整位姿变换，
                # 不能只减位置（姿态也会变）。
                obj_p, obj_q = subtract_frame_transforms(
                    robot.data.root_pos_w, robot.data.root_quat_w,
                    obj.data.root_pos_w, obj.data.root_quat_w,
                )
                # EE 位姿走 ee_frame 这个 FrameTransformer，不是 find_bodies：
                # TCP 是 link7 上带偏移的一个虚拟点，body 位姿差着那段偏移，
                # 直接拿 link7 会系统性偏几厘米。奖励项用的也是 ee_frame。
                ee_frame = base_env.scene["ee_frame"]
                ee_p, ee_q = subtract_frame_transforms(
                    robot.data.root_pos_w, robot.data.root_quat_w,
                    ee_frame.data.target_pos_w[:, 0, :],
                    ee_frame.data.target_quat_w[:, 0, :],
                )
                obj_p_np = obj_p.cpu().numpy(); obj_q_np = obj_q.cpu().numpy()
                ee_p_np = ee_p.cpu().numpy(); ee_q_np = ee_q.cpu().numpy()
                qpos_np = robot.data.joint_pos.cpu().numpy()

                for i in range(num_envs):
                    if not ep_alive[i]:
                        continue
                    ep_id = rnd * num_envs + i
                    # 全部图片平铺在一个文件夹里，从 001.png 连号往下排。
                    # 编号只由 n_img 这一个计数器给出，不掺 ep/env —— 掺进去就会
                    # 在提前终止的轮次上留空号。图和轮次的对应关系记在 summary.npz
                    # 的 image_name/episode 里，不靠文件名编码。
                    name = f"{n_img + 1:0{name_digits}d}.png"
                    path = os.path.join(frames_dir, name)
                    futures.append(pool.submit(_write_png, path, rgb_batch[i].copy()))
                    rec_name.append(name)
                    rec_ep.append(ep_id); rec_env.append(i)
                    rec_shot.append(shot_idx); rec_t.append(sim_t)
                    rec_obj_base.append(np.concatenate([obj_p_np[i], obj_q_np[i]]))
                    rec_ee_base.append(np.concatenate([ee_p_np[i], ee_q_np[i]]))
                    rec_qpos.append(qpos_np[i])
                    rec_done.append(False)
                    shots_taken[i] += 1
                    n_img += 1

            d = dones.cpu().numpy().astype(bool).reshape(-1)
            ep_alive &= ~d

        el = time.time() - t0
        fps = n_img / el if el > 0 else 0.0
        eta = (total_eps * shots_per_ep - n_img) / fps if fps > 0 else 0.0
        n_full = int((shots_taken == shots_per_ep).sum())
        print(f"  [轮 {rnd+1:4d}/{num_rounds}] 图 {n_img}/{total_eps*shots_per_ep}  "
              f"完整轮 {n_full}/{num_envs}  {fps:.1f} 图/s  ETA {eta:.0f}s")

    print("等待写盘完成 ...")
    for fu in futures:
        fu.result()  # 让异常浮出来，别静默丢帧
    pool.shutdown()

    rec_ep_a = np.asarray(rec_ep, dtype=np.int32)
    # 每轮实际存了几张。不足 shots_per_ep 的说明策略把 episode 提前终止了，
    # 训 student 时这些短轮次要不要用是个判断题，所以显式记下来。
    counts = np.bincount(rec_ep_a, minlength=total_eps)
    np.savez(
        os.path.join(args_cli.out, "summary.npz"),
        # 图片平铺成一个文件夹后，"第几张图属于第几轮"只能靠这两列对应，
        # 不再能从路径读出来。image_name 与 frames/ 下的文件名逐条对齐。
        image_name=np.asarray(rec_name),
        episode=rec_ep_a,
        env_index=np.asarray(rec_env, dtype=np.int32),
        shot_index=np.asarray(rec_shot, dtype=np.int32),
        sim_time_s=np.asarray(rec_t, dtype=np.float32),
        object_pose_base=np.asarray(rec_obj_base, dtype=np.float32),
        ee_pose_base=np.asarray(rec_ee_base, dtype=np.float32),
        joint_pos=np.asarray(rec_qpos, dtype=np.float32),
        shots_per_episode=counts.astype(np.int32),
        expected_shots_per_episode=shots_per_ep,
        pose_frame="robot_base",
        pose_layout="xyz + quat_wxyz",
    )

    # 平铺之后光看文件名已经读不出这张图是哪一轮、第几秒了，所以额外写一份纯文本
    # 索引。npz 里有同样的内容，但那个得写代码才能翻 —— 抽查标注、和别人对数据时
    # 用的是这一份。
    csv_path = os.path.join(args_cli.out, "index.csv")
    with open(csv_path, "w", encoding="utf-8") as f:
        f.write("image,episode,env,shot,sim_time_s,"
                "obj_x,obj_y,obj_z,obj_qw,obj_qx,obj_qy,obj_qz\n")
        for k in range(len(rec_name)):
            p = rec_obj_base[k]
            f.write(f"{rec_name[k]},{rec_ep[k]},{rec_env[k]},{rec_shot[k]},"
                    f"{rec_t[k]:.3f}," + ",".join(f"{v:.6f}" for v in p) + "\n")

    K = canonical_intrinsic_matrix(CANONICAL_WIDTH, CANONICAL_HEIGHT)
    K33 = [[float(K[3 * r + c]) for c in range(3)] for r in range(3)]
    cam_p, cam_q = subtract_frame_transforms(
        robot.data.root_pos_w[:1], robot.data.root_quat_w[:1],
        camera.data.pos_w[:1], camera.data.quat_w_world[:1],
    )
    info = {
        "checkpoint": args_cli.checkpoint,
        "stage": -1,
        "camera": "camera_fixed",
        "image_size_wh": [CANONICAL_WIDTH, CANONICAL_HEIGHT],
        "intrinsics_K": K33,
        "camera_position_base_xyz": [float(v) for v in cam_p[0].cpu().numpy()],
        "camera_quaternion_base_wxyz": [float(v) for v in cam_q[0].cpu().numpy()],
        "control_dt_s": float(control_dt),
        "episode_s": args_cli.episode_s,
        "interval_s": args_cli.interval_s,
        "steps_per_episode": steps_per_ep,
        "steps_per_shot": steps_per_shot,
        "shots_per_episode": shots_per_ep,
        "num_envs": num_envs,
        "episodes": total_eps,
        "images": int(n_img),
        "image_layout": "flat",
        "image_name_format": f"%0{name_digits}d.png (从 1 起，跨轮次连号)",
        "seed": args_cli.seed,
        "warmup_steps": args_cli.warmup,
        "rgb_only": bool(args_cli.rgb_only),
        "incomplete_episodes": int((counts != shots_per_ep).sum()),
    }
    with open(os.path.join(args_cli.out, "dataset_info.json"), "w", encoding="utf-8") as f:
        json.dump(info, f, indent=2, ensure_ascii=False)

    print("=" * 78)
    print(f"[完成] {n_img} 张图 / {total_eps} 轮 → {frames_dir}")
    if rec_name:
        print(f"       {rec_name[0]} ... {rec_name[-1]}（平铺连号）")
    print(f"[索引] {csv_path}")
    n_bad = int((counts != shots_per_ep).sum())
    if n_bad:
        print(f"[注意] {n_bad}/{total_eps} 轮不足 {shots_per_ep} 张 —— "
              "这些轮被环境提前终止（超时/失败）。编号是连的、不留空号，"
              "所以别用编号反推轮次，看 index.csv 的 episode 列。")
    print("=" * 78)

    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
