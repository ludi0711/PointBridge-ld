#!/usr/bin/env python3
"""xArm7 LiftCube 视觉策略训练和回放入口。"""

import argparse
import sys
import os

_CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(os.path.dirname(_CURRENT_DIR))
sys.path.insert(0, _PROJECT_DIR)

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(
    description="xArm7 Pick LiftCube 视觉训练 — FP16 编码器，398 维（无点云），两段式奖励"
)
parser.add_argument("--num_envs",        type=int,   default=64)
parser.add_argument("--play",            action="store_true")
parser.add_argument("--checkpoint",      type=str,   default=None)
parser.add_argument("--resume",          action="store_true")
parser.add_argument("--load_run",        type=str,   default=None)
parser.add_argument("--max_iterations",  type=int,   default=None)
parser.add_argument("--seed",            type=int,   default=None)
parser.add_argument(
    "--enable_success_termination",
    action="store_true",
    help="启用 reach_success 提前 done；默认关闭，使用 no-success horizon 训练。",
)

AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import torch
from configs.xarm7_pick_vision_nopc_env_cfg import (
    XArm7PickLiftCubeEnvCfg,
    XArm7PickLiftCubePlayEnvCfg,
    set_success_termination_enabled,
)
from rsl_rl.runners import OnPolicyRunner
from isaaclab_rl.rsl_rl import RslRlVecEnvWrapper
from isaaclab.envs import ManagerBasedRLEnv


def _make_sphere_marker(stage, prim_path: str, color: tuple, radius: float = 0.012):
    import omni.usd  # noqa: F401
    from pxr import UsdGeom, UsdShade, Gf, Sdf
    sphere = UsdGeom.Sphere.Define(stage, prim_path)
    sphere.GetRadiusAttr().Set(radius)
    mat = UsdShade.Material.Define(stage, prim_path + "/mat")
    shader = UsdShade.Shader.Define(stage, prim_path + "/mat/shader")
    shader.CreateIdAttr("UsdPreviewSurface")
    shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    shader.CreateInput("emissiveColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*color))
    mat.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    UsdShade.MaterialBindingAPI(sphere).Bind(mat)
    xf = UsdGeom.Xformable(sphere)
    op = xf.AddTranslateOp()
    return op


def main():
    if args_cli.play:
        env_cfg = XArm7PickLiftCubePlayEnvCfg()
    else:
        env_cfg = XArm7PickLiftCubeEnvCfg()
    env_cfg.scene.num_envs = args_cli.num_envs

    if args_cli.seed is not None:
        env_cfg.seed = args_cli.seed

    set_success_termination_enabled(env_cfg, args_cli.enable_success_termination)
    _success_term_status = (
        "enabled"
        if args_cli.enable_success_termination
        else "disabled (no-success default)"
    )
    print(f"[Reward] success termination: {_success_term_status}")

    env = ManagerBasedRLEnv(cfg=env_cfg)
    env = RslRlVecEnvWrapper(env)

    from configs.agents.rsl_rl_ppo_cfg import XArm7PickPPORunnerCfg_V58
    from datetime import datetime

    agent_cfg = XArm7PickPPORunnerCfg_V58()
    if args_cli.max_iterations is not None:
        agent_cfg.max_iterations = args_cli.max_iterations

    log_dir = os.path.join(
        _PROJECT_DIR, "logs", "xarm7_pick_liftcube_vision_nopc",
        datetime.now().strftime("%Y-%m-%d_%H-%M-%S"),
    )
    os.makedirs(log_dir, exist_ok=True)
    import shutil
    shutil.copy(
        os.path.join(_PROJECT_DIR, "configs", "xarm7_pick_vision_nopc_env_cfg.py"),
        os.path.join(log_dir, "env_cfg.py"),
    )
    runner = OnPolicyRunner(env, agent_cfg.to_dict(), log_dir=log_dir, device=env.device)

    if args_cli.play:
        if args_cli.checkpoint is not None:
            runner.load(args_cli.checkpoint)
        else:
            print("Warning: No checkpoint specified. Using random policy.")
        policy = runner.get_inference_policy(device=env.device)

        import matplotlib.pyplot as plt
        import numpy as np

        base_env  = env.unwrapped
        rm        = base_env.reward_manager
        term_names = rm.active_terms
        HISTORY   = 200

        global_steps      = []
        step_reward_hist  = {name: [] for name in term_names}
        cumul_reward_hist = {name: [] for name in term_names}
        total_cumul_hist  = []
        episode_boundaries = []
        dist_hist         = []

        _obj      = base_env.scene["object"]
        _ee_frame = base_env.scene["ee_frame"]

        fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(14, 11), sharex=True)
        fig.suptitle("Reward Monitor (Play — FP16 编码器，398维 无点云)", fontsize=12)
        cmap   = plt.get_cmap("tab10")
        colors = {name: cmap(i % 10) for i, name in enumerate(term_names)}

        fig_cam, (ax_fixed, ax_wrist) = plt.subplots(1, 2, figsize=(10, 5))
        ax_fixed.axis("off"); ax_fixed.set_title("Fixed Camera (RGB)", fontsize=10)
        ax_wrist.axis("off"); ax_wrist.set_title("Wrist Camera (RGB)", fontsize=10)
        _dummy = np.zeros((224, 224, 3), dtype=np.uint8)
        _fixed_handle = ax_fixed.imshow(_dummy)
        _wrist_handle = ax_wrist.imshow(_dummy)
        fig_cam.tight_layout()
        plt.figure(fig_cam.number); plt.pause(0.05)
        plt.figure(fig.number)

        lines_step  = {name: ax1.plot([], [], label=name, color=colors[name], lw=1)[0] for name in term_names}
        lines_cumul = {name: ax2.plot([], [], label=name, color=colors[name], lw=1)[0] for name in term_names}
        line_total, = ax3.plot([], [], "k-", lw=1.5, label="total cumul")
        line_dist,  = ax4.plot([], [], color="steelblue", lw=1.2, label="dist (cm)")

        ax1.set_ylabel("Per-step Reward")
        ax2.set_ylabel("Cumul per Term (episode)")
        ax3.set_ylabel("Total Cumul (episode)")
        ax4.set_ylabel("Dist (cm)"); ax4.set_xlabel("Step")
        for ax in (ax1, ax2):
            ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax3.legend(loc="upper right", fontsize=7)
        ax4.legend(loc="upper right", fontsize=7)
        for ax in (ax1, ax2, ax3, ax4):
            ax.grid(True, alpha=0.3)
        plt.tight_layout(); plt.ion(); plt.pause(0.1)

        vlines = {1: [], 2: [], 3: [], 4: []}

        def _update_plot():
            xs = global_steps[-HISTORY:]
            for name in term_names:
                lines_step[name].set_data(xs, step_reward_hist[name][-HISTORY:])
                lines_cumul[name].set_data(xs, cumul_reward_hist[name][-HISTORY:])
            line_total.set_data(xs, total_cumul_hist[-HISTORY:])
            line_dist.set_data(xs, dist_hist[-HISTORY:])
            for vl in vlines[1] + vlines[2] + vlines[3] + vlines[4]:
                try: vl.remove()
                except Exception: pass
            for k in vlines: vlines[k].clear()
            recent_b = [b for b in episode_boundaries if b >= g_step - HISTORY]
            for b in recent_b:
                for k, ax in zip([1, 2, 3, 4], [ax1, ax2, ax3, ax4]):
                    vlines[k].append(ax.axvline(b, color="gray", ls="--", alpha=0.4, lw=0.8))
            for ax in (ax1, ax2, ax3, ax4):
                ax.relim(); ax.autoscale_view()

        import omni.usd as _omni_usd
        from pxr import Gf as _Gf, UsdGeom as _UsdGeom, Usd as _Usd
        _stage = _omni_usd.get_context().get_stage()
        _ee_marker_op       = _make_sphere_marker(_stage, "/World/Markers/ee_offset",   color=(1.0, 0.0, 0.0))
        _obj_marker_op      = _make_sphere_marker(_stage, "/World/Markers/obj_target",  color=(0.0, 0.3, 1.0), radius=0.03)
        _cam_fixed_marker_op = _make_sphere_marker(_stage, "/World/Markers/cam_fixed",  color=(1.0, 0.8, 0.0), radius=0.02)

        obs, _ = env.reset()
        # 固定相机 prim 在 reset 后才确保 transform 已写入 USD
        _cam_fixed_prim = _stage.GetPrimAtPath("/World/envs/env_0/CameraFixed")
        step   = 0
        g_step = 0

        while simulation_app.is_running():
            with torch.no_grad():
                actions = policy(obs)
            obs, _, dones, _ = env.step(actions)

            step_vals   = {name: rm._step_reward[0, i].item() for i, name in enumerate(term_names)}
            cumul_vals  = {name: rm._episode_sums[name][0].item() for name in term_names}
            total_cumul = sum(cumul_vals.values())

            global_steps.append(g_step)
            for name in term_names:
                step_reward_hist[name].append(step_vals[name])
                cumul_reward_hist[name].append(cumul_vals[name])
            total_cumul_hist.append(total_cumul)

            _ee_pos  = _ee_frame.data.target_pos_w[0, 0].cpu().numpy()
            _obj_pos = _obj.data.root_pos_w[0].cpu().numpy()
            dist_hist.append(float(np.linalg.norm(_obj_pos - _ee_pos) * 100))

            _ee_marker_op.Set(_Gf.Vec3d(float(_ee_pos[0]),  float(_ee_pos[1]),  float(_ee_pos[2])))
            _obj_marker_op.Set(_Gf.Vec3d(float(_obj_pos[0]), float(_obj_pos[1]), float(_obj_pos[2])))

            # 固定相机：静态 prim，USD transform 可读
            if _cam_fixed_prim.IsValid():
                _t = _UsdGeom.Xformable(_cam_fixed_prim).ComputeLocalToWorldTransform(_Usd.TimeCode.Default()).ExtractTranslation()
                _cam_fixed_marker_op.Set(_Gf.Vec3d(_t[0], _t[1], _t[2]))

            _rgb_fixed = base_env.scene["camera_fixed"].data.output["rgb"]
            if _rgb_fixed is not None and _rgb_fixed.shape[0] > 0:
                _fixed_handle.set_data(_rgb_fixed[0, ..., :3].cpu().numpy().astype(np.uint8))

            _rgb_wrist = base_env.scene["camera_wrist"].data.output["rgb"]
            if _rgb_wrist is not None and _rgb_wrist.shape[0] > 0:
                _wrist_handle.set_data(_rgb_wrist[0, ..., :3].cpu().numpy().astype(np.uint8))

            fig_cam.canvas.draw_idle()

            step += 1; g_step += 1

            if dones[0]:
                episode_boundaries.append(g_step)
                _update_plot()
                ep_total   = sum(sum(step_reward_hist[name][-step:]) for name in term_names)
                ep_num     = len(episode_boundaries)
                is_success = step < 250
                label      = "SUCCESS" if is_success else "TIMEOUT"
                fig.suptitle(
                    f"Episode {ep_num}  [{label}]  steps={step}  ep_reward≈{ep_total:.2f}"
                    f"  |  在图表窗口按任意键继续...",
                    fontsize=10, color="green" if is_success else "red",
                )
                plt.pause(0.1); plt.waitforbuttonpress(timeout=-1)
                fig.suptitle("Reward Monitor (Play — FP16 编码器，398维 无点云)", fontsize=12, color="black")
                obs, _ = env.reset()
                step = 0

            elif g_step % 5 == 0:
                _update_plot(); plt.pause(0.001)
    else:
        if args_cli.resume:
            if args_cli.checkpoint is not None:
                runner.load(args_cli.checkpoint)
            elif args_cli.load_run is not None:
                runner.load(args_cli.load_run)
        runner.learn(num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True)

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
