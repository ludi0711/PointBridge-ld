# Copyright (c) 2024, Isaac Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""
xArm7 Pick 任务环境配置：位姿真值训练（relative_pose + joint_step_direct，0.6°/step）

环境特性：
    action_dim = 7
    delta_q = clip(action * 0.6°, -0.6°, +0.6°)
    q_target = q_current_at_action_time + delta_q
    apply_actions() 在 decimation=2 的两个 physics substep 中保持同一个 q_target

观测：
    21 维 = joint_pos_rel(7) + object_pose_relative_to_ee(7)
             + last_action_clipped(7)

成功条件：
    position error < 1 cm，orientation error < 3 deg

关键原则：
    1. 本文件是环境配置文件，不是 AppLauncher 启动脚本；
    2. 由配套 scripts/train_pick_pose.py 在 AppLauncher 启动后导入；
    3. 训练环境中不加入动作平滑、不加入最小距离 stop 逻辑；
    4. 动作平滑、按距离/姿态 stop 等逻辑只建议放在 play / 部署端。
"""

import math
import os
import random as _random
from dataclasses import MISSING

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.sensors import FrameTransformerCfg, ContactSensorCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

import isaaclab.envs.mdp as mdp

from . import xarm7_pick_liftcube_mdp as custom_mdp
from .lift_workbench_scene_cfg import LiftWorkbenchSceneCfg

# ── 常量 ──────────────────────────────────────────────────────────────────────
# 训练阶段保持干净动作语义：无动作平滑。
# 策略输出相对当前关节角的增量目标，直接写入版本采用 0.6°/step。
_ARM_ACTION_SCALE = math.radians(0.6)
_ARM_CLIP_RAD     = math.radians(0.6)

_CONFIGS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.dirname(_CONFIGS_DIR)
_ASSETS_DIR  = os.path.join(_PROJECT_DIR, "assets/xarm7")

GONGJIAN_USD     = os.path.join(_ASSETS_DIR, "gongjian.usd")
ROBOT_USD    = os.path.join(_ASSETS_DIR, "XARM-WITH-GRIP-NEW-FALAN.usd")
_LIGHT_MIN = 0.0
_LIGHT_MAX = 3000.0

_TABLE_HEIGHT_RAND = 0.00
_TABLE_BASE_H      = 0.4075

# 平行夹爪绕自身局部 Z 轴旋转 180° 后只是左右指交换，视为等价夹持姿态。
# 四元数格式为 w,x,y,z；第一项为原姿态，第二项为 180° 对称姿态。
_GRASP_SYMMETRY_QUATS_WXYZ = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)

INIT_JOINT_POS = {
    "joint1": math.radians(-0.4),
    "joint2": math.radians(-47.6),
    "joint3": math.radians(0.0),
    "joint4": math.radians(1.8),
    "joint5": math.radians(-0.2),
    "joint6": math.radians(49.4),
    "joint7": math.radians(-0.1),
}

# 和部署脚本保持一致，作为额外安全限位
_ARM_JOINT_LIMITS_LOW = torch.tensor(
    [math.radians(v) for v in [-180, -118, -180, -11, -97, -180, -180]],
    dtype=torch.float32,
)
_ARM_JOINT_LIMITS_HIGH = torch.tensor(
    [math.radians(v) for v in [180, 118, 180, 225, 97, 180, 180]],
    dtype=torch.float32,
)


# ── 机器人配置 ─────────────────────────────────────────────────────────────────
XARM7_GRIP_NEW_FALAN_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=ROBOT_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # 直接关节步进 + step 末端锁定。
            # 真机位置伺服会补偿重力并保持目标角；仿真端关闭机器人重力，
            # 并在每个 env.step 结束阶段再次锁回 q_target，减少 physics step 漂移。
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos=INIT_JOINT_POS),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-7]"],
            effort_limit_sim=200.0,
            velocity_limit_sim=100.0,
            stiffness=800.0,
            damping=80.0,
        ),
    },
)


# ── 自定义动作项：相对关节增量目标直接写入 ───────────────────────────────────────
class KinematicRelativeJointDirectAction(ActionTerm):
    """相对当前关节角的关节增量目标，直接写入关节状态。

    本版训练动作链路：
        delta_q = clip(action * scale, -clip, +clip)
        q_next  = q_current + delta_q
        joint_vel = 0
        write_joint_state_to_sim(q_next, joint_vel)

    注意：
        processed_actions 的单位是 rad，含义是本 step 实际直接写入的 delta_q。
        训练阶段不做动作平滑。
    """

    cfg: "KinematicRelativeJointDirectActionCfg"

    def __init__(self, cfg: "KinematicRelativeJointDirectActionCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)

        self._asset = env.scene[cfg.asset_name]
        self._joint_ids, self._joint_names = self._asset.find_joints(cfg.joint_names)
        if len(self._joint_ids) == 0:
            raise RuntimeError(f"[KinematicRelativeJointDirectAction] 找不到关节: {cfg.joint_names}")

        self._joint_ids = list(self._joint_ids)
        self._num_joints = len(self._joint_ids)
        self._scale = float(cfg.scale)
        self._clip = float(cfg.clip)

        self._raw_actions = torch.zeros(
            self.num_envs, self._num_joints, device=self.device, dtype=torch.float32
        )
        self._processed_actions = torch.zeros_like(self._raw_actions)

        # decimation=2 时，process_actions() 每个 RL step 只调用一次，
        # apply_actions() 会在两个 physics substep 中各调用一次。
        # 所以 q_target 必须在 process_actions() 中只计算一次并缓存。
        self._target_joint_pos = torch.zeros_like(self._raw_actions)
        self._zero_joint_vel_target = torch.zeros_like(self._raw_actions)
        self._target_initialized = False

        self._warned_position_target_sync = False
        self._warned_velocity_target_sync = False

        low = torch.as_tensor(cfg.joint_limits_low, device=self.device, dtype=torch.float32)
        high = torch.as_tensor(cfg.joint_limits_high, device=self.device, dtype=torch.float32)
        if low.numel() != self._num_joints or high.numel() != self._num_joints:
            raise RuntimeError(
                f"joint_limits_low/high 维度应为 {self._num_joints}，"
                f"当前 low={low.numel()}, high={high.numel()}"
            )
        self._joint_limits_low = low.view(1, self._num_joints)
        self._joint_limits_high = high.view(1, self._num_joints)

        print(
            "[KinematicRelativeJointDirectAction] enabled | "
            f"joints={self._joint_names} | "
            f"scale={math.degrees(self._scale):.4f} deg | "
            f"clip=±{math.degrees(self._clip):.4f} deg | "
            "NO_SMOOTH_TRAINING | NO_DOUBLE: target cached once in process_actions"
        )

    @property
    def action_dim(self) -> int:
        return self._num_joints

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        """把策略动作转换为本个 env.step 的一次性关节目标。"""
        self._raw_actions[:] = actions

        # 训练阶段不做动作平滑：action 直接对应本 step 的关节增量。
        delta_q = actions * self._scale
        delta_q = torch.clamp(delta_q, -self._clip, self._clip)
        self._processed_actions[:] = delta_q

        q_current = self._asset.data.joint_pos[:, self._joint_ids]
        q_target = q_current + delta_q

        # 额外硬限位，避免写入超过 xArm 安全范围。
        q_target = torch.maximum(torch.minimum(q_target, self._joint_limits_high), self._joint_limits_low)
        self._target_joint_pos[:] = q_target
        self._target_initialized = True

    def _sync_position_target(self, joint_pos: torch.Tensor, q_target: torch.Tensor):
        """同步隐式执行器 position target，避免 PD target 与直接写入状态打架。"""
        try:
            self._asset.set_joint_position_target(q_target, joint_ids=self._joint_ids)
        except TypeError:
            try:
                full_pos_target = joint_pos.clone()
                full_pos_target[:, self._joint_ids] = q_target
                self._asset.set_joint_position_target(full_pos_target)
            except Exception as exc:
                if not self._warned_position_target_sync:
                    print(
                        "[KinematicRelativeJointDirectAction] position target sync skipped: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    self._warned_position_target_sync = True
        except Exception as exc:
            if not self._warned_position_target_sync:
                print(
                    "[KinematicRelativeJointDirectAction] position target sync skipped: "
                    f"{type(exc).__name__}: {exc}"
                )
                self._warned_position_target_sync = True

    def _sync_velocity_target(self, joint_vel: torch.Tensor):
        """同步隐式执行器 velocity target 为 0，减少残余速度目标干扰。"""
        try:
            self._asset.set_joint_velocity_target(self._zero_joint_vel_target, joint_ids=self._joint_ids)
        except TypeError:
            try:
                full_vel_target = joint_vel.clone()
                full_vel_target[:, self._joint_ids] = 0.0
                self._asset.set_joint_velocity_target(full_vel_target)
            except Exception as exc:
                if not self._warned_velocity_target_sync:
                    print(
                        "[KinematicRelativeJointDirectAction] velocity target sync skipped: "
                        f"{type(exc).__name__}: {exc}"
                    )
                    self._warned_velocity_target_sync = True
        except Exception as exc:
            if not self._warned_velocity_target_sync:
                print(
                    "[KinematicRelativeJointDirectAction] velocity target sync skipped: "
                    f"{type(exc).__name__}: {exc}"
                )
                self._warned_velocity_target_sync = True

    def apply_actions(self):
        joint_pos = self._asset.data.joint_pos.clone()
        joint_vel = self._asset.data.joint_vel.clone()

        if not self._target_initialized:
            self._target_joint_pos[:] = joint_pos[:, self._joint_ids]
            self._target_initialized = True
        q_target = self._target_joint_pos

        joint_pos[:, self._joint_ids] = q_target
        joint_vel[:, self._joint_ids] = 0.0

        self._sync_position_target(joint_pos, q_target)
        self._sync_velocity_target(joint_vel)
        self._asset.write_joint_state_to_sim(joint_pos, joint_vel)

    def lock_to_target_after_physics(self):
        """在 physics step 结束后再次把机器人 7 个关节锁回本 step 的 q_target。"""
        if not self._target_initialized:
            return

        joint_pos = self._asset.data.joint_pos.clone()
        joint_vel = self._asset.data.joint_vel.clone()
        q_target = self._target_joint_pos

        joint_pos[:, self._joint_ids] = q_target
        joint_vel[:, self._joint_ids] = 0.0

        self._sync_position_target(joint_pos, q_target)
        self._sync_velocity_target(joint_vel)
        self._asset.write_joint_state_to_sim(joint_pos, joint_vel)


@configclass
class KinematicRelativeJointDirectActionCfg(ActionTermCfg):
    """KinematicRelativeJointDirectAction 配置。"""

    class_type: type = KinematicRelativeJointDirectAction

    asset_name: str = MISSING
    joint_names: list[str] = MISSING
    scale: float = _ARM_ACTION_SCALE
    clip: float = _ARM_CLIP_RAD
    joint_limits_low: tuple[float, ...] = tuple(_ARM_JOINT_LIMITS_LOW.tolist())
    joint_limits_high: tuple[float, ...] = tuple(_ARM_JOINT_LIMITS_HIGH.tolist())


# ── step 末端锁定函数：用于在 physics step 后再次锁回 q_target ────────────────
def _get_arm_action_term(env: ManagerBasedRLEnv):
    """尽量兼容不同 Isaac Lab 版本，从 action_manager 中取出名为 arm_action 的 ActionTerm。"""
    manager = env.action_manager

    if hasattr(manager, "get_term"):
        try:
            term = manager.get_term("arm_action")
            if hasattr(term, "lock_to_target_after_physics"):
                return term
        except Exception:
            pass

    for attr in ("_terms", "_action_terms"):
        terms = getattr(manager, attr, None)
        if isinstance(terms, dict) and "arm_action" in terms:
            term = terms["arm_action"]
            if hasattr(term, "lock_to_target_after_physics"):
                return term

    terms = getattr(manager, "_terms", None)
    if isinstance(terms, (list, tuple)):
        for term in terms:
            if hasattr(term, "lock_to_target_after_physics"):
                return term

    return None


def _lock_robot_to_cached_joint_target(env: ManagerBasedRLEnv) -> None:
    """在 env.step 后半段调用，把机器人关节锁回本 step 缓存的 q_target。"""
    term = _get_arm_action_term(env)
    if term is None:
        return

    term.lock_to_target_after_physics()

    try:
        env.scene.write_data_to_sim()
    except Exception:
        pass
    try:
        env.scene.update(0.0)
    except Exception:
        pass


def lock_robot_to_cached_joint_target_reward(env: ManagerBasedRLEnv) -> torch.Tensor:
    """零奖励项：副作用是在 reward 计算前锁回 q_target，不改变总奖励。"""
    _lock_robot_to_cached_joint_target(env)
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)


def lock_robot_to_cached_joint_target_done(env: ManagerBasedRLEnv) -> torch.Tensor:
    """零终止项：副作用是在 termination 计算前锁回 q_target，不触发终止。"""
    _lock_robot_to_cached_joint_target(env)
    return torch.zeros(env.num_envs, device=env.device, dtype=torch.bool)


def _cache_terminal_state(
    env: ManagerBasedRLEnv,
    valid_mask: torch.Tensor,
    done_reason: str,
    success: torch.Tensor,
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    q_offset: tuple[float, float, float, float],
    grasp_symmetry_quats: tuple[tuple[float, float, float, float], ...],
) -> None:
    """Cache pre-reset terminal state for eval logging."""
    valid_mask = valid_mask.detach().bool().clone()
    dist_m, ori_err_deg = _compute_reach_error_metrics(
        env,
        object_cfg=object_cfg,
        ee_frame_cfg=ee_frame_cfg,
        q_offset=q_offset,
        grasp_symmetry_quats=grasp_symmetry_quats,
    )
    env._terminal_cache_valid = valid_mask
    env._terminal_done_reason = [done_reason if bool(v) else "" for v in valid_mask.detach().cpu().tolist()]
    env._terminal_dist_cm = (dist_m * 100.0).detach().clone()
    env._terminal_ori_err_deg = ori_err_deg.detach().clone()
    env._terminal_success = success.detach().bool().clone()

    obj = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]
    env._terminal_object_pos_w = obj.data.root_pos_w.detach().clone()
    env._terminal_object_quat_w = obj.data.root_quat_w.detach().clone()
    ee_pos_w, ee_quat_w = _frame_pose_for_terminal_cache(ee_frame)
    if ee_pos_w is not None and ee_quat_w is not None:
        env._terminal_ee_pos_w = ee_pos_w
        env._terminal_ee_quat_w = ee_quat_w

    try:
        robot = env.scene["robot"]
        joint_ids, _ = robot.find_joints(["joint[1-7]"])
        env._terminal_joint_pos = robot.data.joint_pos[:, list(joint_ids)].detach().clone()
    except Exception:
        pass


def contact_force_done(
    env: ManagerBasedRLEnv,
    sensor_name: str,
    threshold: float = 0.5,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    q_offset: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    grasp_symmetry_quats: tuple[tuple[float, float, float, float], ...] = _GRASP_SYMMETRY_QUATS_WXYZ,
) -> torch.Tensor:
    """碰撞终止项：复用 collision_penalty 的接触力阈值，触碰后立即结束 episode。"""
    collided = custom_mdp.contact_force_penalty(
        env,
        sensor_name=sensor_name,
        threshold=threshold,
    ).bool()
    if torch.any(collided):
        _cache_terminal_state(
            env,
            valid_mask=collided,
            done_reason="collision",
            success=torch.zeros_like(collided, dtype=torch.bool),
            object_cfg=object_cfg,
            ee_frame_cfg=ee_frame_cfg,
            q_offset=q_offset,
            grasp_symmetry_quats=grasp_symmetry_quats,
        )
    return collided


def time_out_cached_terminal(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    q_offset: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    grasp_symmetry_quats: tuple[tuple[float, float, float, float], ...] = _GRASP_SYMMETRY_QUATS_WXYZ,
) -> torch.Tensor:
    """Timeout termination with pre-reset terminal-state cache for eval logging."""
    timed_out = env.episode_length_buf >= env.max_episode_length
    if torch.any(timed_out):
        _cache_terminal_state(
            env,
            valid_mask=timed_out,
            done_reason="timeout",
            success=torch.zeros_like(timed_out, dtype=torch.bool),
            object_cfg=object_cfg,
            ee_frame_cfg=ee_frame_cfg,
            q_offset=q_offset,
            grasp_symmetry_quats=grasp_symmetry_quats,
        )
    return timed_out


def last_clipped_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """返回上一帧真实执行后的动作，而不是策略原始输出。"""
    term = _get_arm_action_term(env)
    if term is None or not hasattr(term, "processed_actions"):
        return mdp.last_action(env)

    processed = term.processed_actions
    arm_action = torch.clamp(processed[:, :7] / max(_ARM_ACTION_SCALE, 1.0e-8), -1.0, 1.0)
    if processed.shape[-1] <= 7:
        return arm_action

    # CRT gripper variants append a normalized 0/1 close command after the arm action.
    gripper_action = torch.clamp(processed[:, 7:], 0.0, 1.0)
    return torch.cat((arm_action, gripper_action), dim=-1)


def object_pose_env_frame(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """工件在各自环境局部世界坐标系中的绝对位姿。

    多环境会把每个 env 平铺到不同 world origin。这里从 world position 中减去
    env origin，保留同一任务坐标系下的绝对位置；姿态仍使用 world quaternion(wxyz)。
    """
    obj = env.scene[object_cfg.name]
    pos = obj.data.root_pos_w - env.scene.env_origins
    quat = obj.data.root_quat_w
    return torch.cat((pos, quat), dim=-1)


def _target_quat_candidates_with_grasp_symmetry(
    obj_quat_w: torch.Tensor,
    q_offset: tuple[float, float, float, float],
    grasp_symmetry_quats: tuple[tuple[float, float, float, float], ...] = _GRASP_SYMMETRY_QUATS_WXYZ,
) -> torch.Tensor:
    """返回 shape=(num_envs, num_sym, 4) 的对称等价目标姿态。"""
    offset = torch.tensor(q_offset, dtype=torch.float32, device=obj_quat_w.device)
    offset = offset / torch.clamp(torch.linalg.norm(offset), min=1.0e-8)

    sym = torch.tensor(grasp_symmetry_quats, dtype=torch.float32, device=obj_quat_w.device)
    sym = sym / torch.clamp(torch.linalg.norm(sym, dim=-1, keepdim=True), min=1.0e-8)

    # 先应用原来的 grasp offset，再绕夹爪局部 Z 轴应用 180° 等价旋转。
    offsets = math_utils.quat_mul(offset.unsqueeze(0).expand_as(sym), sym)
    offsets = offsets / torch.clamp(torch.linalg.norm(offsets, dim=-1, keepdim=True), min=1.0e-8)

    num_envs = obj_quat_w.shape[0]
    num_sym = offsets.shape[0]
    obj = obj_quat_w[:, None, :].expand(num_envs, num_sym, 4).reshape(-1, 4)
    off = offsets[None, :, :].expand(num_envs, num_sym, 4).reshape(-1, 4)
    targets = math_utils.quat_mul(obj, off).view(num_envs, num_sym, 4)
    return targets / torch.clamp(torch.linalg.norm(targets, dim=-1, keepdim=True), min=1.0e-8)


def _nearest_symmetric_relative_quat(
    ee_quat_w: torch.Tensor,
    target_quat_candidates_w: torch.Tensor,
) -> torch.Tensor:
    """在对称候选姿态中选取离当前 TCP 最近的相对四元数。"""
    num_envs, num_sym, _ = target_quat_candidates_w.shape
    ee_inv = math_utils.quat_inv(ee_quat_w)
    ee_inv = ee_inv[:, None, :].expand(num_envs, num_sym, 4).reshape(-1, 4)
    targets = target_quat_candidates_w.reshape(-1, 4)
    rel_quats = math_utils.quat_mul(ee_inv, targets).view(num_envs, num_sym, 4)
    rel_quats = rel_quats / torch.clamp(torch.linalg.norm(rel_quats, dim=-1, keepdim=True), min=1.0e-8)

    best_ids = torch.argmax(torch.abs(rel_quats[:, :, 0]), dim=1)
    rel_quat = rel_quats[torch.arange(num_envs, device=ee_quat_w.device), best_ids]

    # 四元数双覆盖消歧，避免同一个姿态在观测里正负跳变。
    sign = torch.where(rel_quat[:, 0:1] < 0.0, -1.0, 1.0)
    return rel_quat * sign


def object_pose_relative_to_ee(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    q_offset: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    grasp_symmetry_quats: tuple[tuple[float, float, float, float], ...] = _GRASP_SYMMETRY_QUATS_WXYZ,
) -> torch.Tensor:
    """工件相对 TCP/夹爪的位姿。

    返回 7 维：
        [0:3] 工件相对 TCP 的位置，表达在 TCP 坐标系下；
        [3:7] TCP 到最近的对称等价目标夹取姿态的相对四元数误差，wxyz。

    姿态目标使用 q_obj * q_offset，并把夹爪绕局部 Z 轴 180° 后的姿态
    视为同一个夹持方式。观测、reward 和 success 都取最近的对称候选姿态。
    """
    obj = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]

    obj_pos_w = obj.data.root_pos_w
    obj_quat_w = obj.data.root_quat_w
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]

    rel_pos_ee = math_utils.quat_apply_inverse(ee_quat_w, obj_pos_w - ee_pos_w)
    target_quats_w = _target_quat_candidates_with_grasp_symmetry(
        obj_quat_w,
        q_offset=q_offset,
        grasp_symmetry_quats=grasp_symmetry_quats,
    )
    rel_quat = _nearest_symmetric_relative_quat(ee_quat_w, target_quats_w)

    return torch.cat((rel_pos_ee, rel_quat), dim=-1)


def _compute_reach_error_metrics(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    q_offset: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    grasp_symmetry_quats: tuple[tuple[float, float, float, float], ...] = _GRASP_SYMMETRY_QUATS_WXYZ,
):
    """计算当前末端到工件目标位姿的距离误差和姿态误差。

    返回：
        dist_m:      shape=(num_envs,), 单位 m
        ori_err_deg: shape=(num_envs,), 单位 deg

    该函数只用于日志缓存，不改变奖励或终止逻辑。
    """
    obj = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]

    obj_pos_w = obj.data.root_pos_w
    obj_quat_w = obj.data.root_quat_w
    ee_pos_w = ee_frame.data.target_pos_w[:, 0, :]
    ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]

    dist_m = torch.linalg.norm(obj_pos_w - ee_pos_w, dim=-1)

    target_quat_w = _target_quat_candidates_with_grasp_symmetry(
        obj_quat_w,
        q_offset=q_offset,
        grasp_symmetry_quats=grasp_symmetry_quats,
    )
    rel_quat = _nearest_symmetric_relative_quat(ee_quat_w, target_quat_w)

    # 四元数双覆盖处理：q 和 -q 表示同一姿态，因此取 abs(w)。
    w = torch.abs(rel_quat[:, 0])
    xyz_norm = torch.linalg.norm(rel_quat[:, 1:4], dim=-1)
    angle_rad = 2.0 * torch.atan2(xyz_norm, torch.clamp(w, min=1.0e-8))
    ori_err_deg = torch.rad2deg(angle_rad)

    return dist_m, ori_err_deg


def _frame_pose_for_terminal_cache(ee_frame):
    data = ee_frame.data

    pos = None
    for name in ("target_pos_w", "frame_pos_w", "source_pos_w"):
        if hasattr(data, name):
            pos = getattr(data, name)
            pos = pos[:, 0, :] if pos.ndim == 3 else pos
            break
    if pos is None:
        return None, None

    quat = None
    for name in ("target_quat_w", "frame_quat_w", "source_quat_w"):
        if hasattr(data, name):
            quat = getattr(data, name)
            quat = quat[:, 0, :] if quat.ndim == 3 else quat
            break
    if quat is None:
        return None, None

    return pos.detach().clone(), quat.detach().clone()


def ee_reached_object_cached_terminal(
    env: ManagerBasedRLEnv,
    threshold: float = 0.01,
    angle_threshold_deg: float = 3.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    q_offset: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    grasp_symmetry_quats: tuple[tuple[float, float, float, float], ...] = _GRASP_SYMMETRY_QUATS_WXYZ,
) -> torch.Tensor:
    """成功终止函数：保持原成功判定，同时缓存 reset 前的终止瞬间误差。

    这个函数返回值仍然由 custom_mdp.ee_reached_object 决定，
    因此不会改变原来的 done 条件，只是额外把 termination 阶段的
    距离误差、姿态误差和 success 标志缓存到 env 上，供回放脚本读取。
    """
    dist_m, ori_err_deg = _compute_reach_error_metrics(
        env,
        object_cfg=object_cfg,
        ee_frame_cfg=ee_frame_cfg,
        q_offset=q_offset,
        grasp_symmetry_quats=grasp_symmetry_quats,
    )

    success = custom_mdp.ee_reached_object(
        env,
        threshold=threshold,
        angle_threshold_deg=angle_threshold_deg,
        object_cfg=object_cfg,
        ee_frame_cfg=ee_frame_cfg,
        q_offset=q_offset,
        grasp_symmetry_quats=grasp_symmetry_quats,
    )

    # termination manager 在 reset 之前调用，因此这里缓存的是 done 瞬间状态。
    if torch.any(success):
        _cache_terminal_state(
            env,
            valid_mask=success,
            done_reason="success",
            success=success,
            object_cfg=object_cfg,
            ee_frame_cfg=ee_frame_cfg,
            q_offset=q_offset,
            grasp_symmetry_quats=grasp_symmetry_quats,
        )

    return success


# ── 域随机化函数 ────────────────────────────────────────────────────────────────
def reset_table_height_and_object_pose(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    x_range: tuple = (-0.10, 0.10),
    y_range: tuple = (-0.10, 0.10),
    yaw_range: tuple = (-math.pi / 2, math.pi / 2),
    height_range: float = _TABLE_HEIGHT_RAND,
) -> None:
    obj    = env.scene["object"]
    table  = env.scene["workpiece_table"]
    device = env.device
    n      = len(env_ids)

    delta_h = torch.zeros(n, device=device).uniform_(-height_range, height_range)

    table_state = table.data.default_root_state[env_ids].clone()
    table_state[:, 2] += delta_h
    table_state[:, :3] += env.scene.env_origins[env_ids]
    table_state[:, 7:] = 0.0
    table.write_root_pose_to_sim(table_state[:, :7], env_ids=env_ids)

    root_state = obj.data.default_root_state[env_ids].clone()
    root_state[:, 0] += torch.zeros(n, device=device).uniform_(*x_range)
    root_state[:, 1] += torch.zeros(n, device=device).uniform_(*y_range)
    root_state[:, 2] += delta_h

    yaw   = torch.zeros(n, device=device).uniform_(*yaw_range)
    zeros = torch.zeros(n, device=device)
    q_yaw = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
    root_state[:, 3:7] = math_utils.quat_mul(q_yaw, root_state[:, 3:7])
    root_state[:, :3] += env.scene.env_origins[env_ids]
    root_state[:, 7:]  = 0.0
    obj.write_root_state_to_sim(root_state, env_ids=env_ids)


def randomize_light_intensity(_env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    import omni.usd
    from pxr import UsdLux

    if len(env_ids) == 0:
        return
    stage      = omni.usd.get_context().get_stage()
    light_prim = stage.GetPrimAtPath("/World/Light")
    if not light_prim.IsValid():
        return
    UsdLux.LightAPI(light_prim).GetIntensityAttr().Set(float(_random.uniform(_LIGHT_MIN, _LIGHT_MAX)))


def randomize_table_color(env: ManagerBasedRLEnv, env_ids: torch.Tensor) -> None:
    import omni.usd
    from pxr import Gf, UsdShade

    stage = omni.usd.get_context().get_stage()
    for env_id in env_ids.tolist():
        shader_path = f"/World/envs/env_{env_id}/WorkpieceTable/geometry/material/Shader"
        shader_prim = stage.GetPrimAtPath(shader_path)
        if not shader_prim.IsValid():
            continue
        color = Gf.Vec3f(_random.random(), _random.random(), _random.random())
        UsdShade.Shader(shader_prim).GetInput("diffuseColor").Set(color)


##
# 场景配置
##
@configclass
class PickPoseSceneCfg(LiftWorkbenchSceneCfg):
    """状态观测场景：新法兰机器人 + 工件 + grip 碰撞检测。"""

    robot = XARM7_GRIP_NEW_FALAN_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (0.9, 4.6, 0.8)
    robot.init_state.rot = (0.707, 0.0, 0.0, -0.707)

    workbench = None

    workpiece_table = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/WorkpieceTable",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.9, 4.0, 0.4075],
            rot=[0.707, 0.0, 0.0, -0.707],
        ),
        spawn=sim_utils.CuboidCfg(
            size=(1.0, 1.0, 0.815),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=0.7,
                dynamic_friction=0.6,
                restitution=0.0,
            ),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.4, 0.25, 0.15),
                metallic=0.1,
                roughness=0.8,
            ),
        ),
    )

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.9, 4.16, 0.833],
            rot=[0.0, 1.0, 0.0, 0.0],
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=GONGJIAN_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=True,
                disable_gravity=True,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )

    ee_frame = FrameTransformerCfg(
        prim_path="{ENV_REGEX_NS}/Robot/link7",
        debug_vis=False,
        target_frames=[
            FrameTransformerCfg.FrameCfg(
                prim_path="{ENV_REGEX_NS}/Robot/link7",
                name="end_effector",
                offset=OffsetCfg(pos=[0.0, 0.0, 0.177]),
            ),
        ],
    )

    grip_contact = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/link7",
        history_length=3,
        track_air_time=False,
    )


##
# MDP 配置
##
@configclass
class ActionsCfg:
    arm_action = KinematicRelativeJointDirectActionCfg(
        asset_name="robot",
        joint_names=["joint[1-7]"],
        scale=_ARM_ACTION_SCALE,
        clip=_ARM_CLIP_RAD,
        joint_limits_low=tuple(_ARM_JOINT_LIMITS_LOW.tolist()),
        joint_limits_high=tuple(_ARM_JOINT_LIMITS_HIGH.tolist()),
    )


def make_reach_success_done_term() -> DoneTerm:
    """Create the optional success termination shared with the vision setup."""
    return DoneTerm(
        func=ee_reached_object_cached_terminal,
        params={
            "threshold":           0.01,
            "angle_threshold_deg": 3.0,
            "object_cfg":          SceneEntityCfg("object"),
            "ee_frame_cfg":        SceneEntityCfg("ee_frame"),
            "q_offset":            (0.0, 0.0, 0.0, 1.0),
            "grasp_symmetry_quats": _GRASP_SYMMETRY_QUATS_WXYZ,
        },
    )


def set_success_termination_enabled(env_cfg: ManagerBasedRLEnvCfg, enabled: bool) -> None:
    """Toggle success termination without changing the reward terms."""
    env_cfg.terminations.reach_success = make_reach_success_done_term() if enabled else None


@configclass
class ObservationsCfg:
    """观测配置 — 21 维。

    joint_pos_rel(7) + object_pose_relative_to_ee(7) + last_action_clipped(7)
    """

    @configclass
    class PolicyCfg(ObsGroup):
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel,
            params={"asset_cfg": SceneEntityCfg("robot", joint_names=["joint[1-7]"])},
        )

        object_relative_pose = ObsTerm(
            func=object_pose_relative_to_ee,
            params={
                "object_cfg": SceneEntityCfg("object"),
                "ee_frame_cfg": SceneEntityCfg("ee_frame"),
                "q_offset": (0.0, 0.0, 0.0, 1.0),
                "grasp_symmetry_quats": _GRASP_SYMMETRY_QUATS_WXYZ,
            },
        )

        actions = ObsTerm(func=last_clipped_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_table_and_object = EventTerm(
        func=reset_table_height_and_object_pose,
        mode="reset",
        params={
            "x_range":      (-0.10, 0.10),
            "y_range":      (-0.10, 0.10),
            "yaw_range":    (-math.pi / 2, math.pi / 2),
            "height_range": _TABLE_HEIGHT_RAND,
        },
    )

    randomize_table_color     = EventTerm(func=randomize_table_color,     mode="reset")
    randomize_light_intensity = EventTerm(func=randomize_light_intensity, mode="reset")


@configclass
class RewardsCfg:
    """位姿真值训练奖励函数。

    训练目标：
      1. 不改变训练动作链路，不加入动作平滑；
      2. 强化 2~4 cm 区间内的位置精修梯度；
      3. success 判定阈值为 1 cm / 3 deg；success termination 默认关闭，可按需开启；
      4. reward 数值与 vision 配置保持一致。
    """

    lock_joint_step_end = RewTerm(
        func=lock_robot_to_cached_joint_target_reward,
        weight=1.0,
    )

    reaching_coarse = RewTerm(
        func=custom_mdp.object_ee_distance,
        params={
            "std":          0.50,
            "object_cfg":   SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=35.0,
    )

    reaching_mid = RewTerm(
        func=custom_mdp.object_ee_distance,
        params={
            "std":          0.15,
            "object_cfg":   SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=55.0,
    )

    reaching_fine = RewTerm(
        func=custom_mdp.object_ee_distance,
        params={
            "std":          0.04,
            "object_cfg":   SceneEntityCfg("object"),
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
        },
        weight=80.0,
    )

    ee_orientation_coarse = RewTerm(
        func=custom_mdp.ee_orientation_alignment,
        params={
            "std":          0.50,
            "q_offset":     (0.0, 0.0, 0.0, 1.0),
            "grasp_symmetry_quats": _GRASP_SYMMETRY_QUATS_WXYZ,
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg":   SceneEntityCfg("object"),
        },
        weight=30.0,
    )

    ee_orientation_mid = RewTerm(
        func=custom_mdp.ee_orientation_alignment,
        params={
            "std":          0.15,
            "q_offset":     (0.0, 0.0, 0.0, 1.0),
            "grasp_symmetry_quats": _GRASP_SYMMETRY_QUATS_WXYZ,
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg":   SceneEntityCfg("object"),
        },
        weight=40.0,
    )

    ee_orientation_fine = RewTerm(
        func=custom_mdp.ee_orientation_alignment,
        params={
            "std":          0.08,
            "q_offset":     (0.0, 0.0, 0.0, 1.0),
            "grasp_symmetry_quats": _GRASP_SYMMETRY_QUATS_WXYZ,
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg":   SceneEntityCfg("object"),
        },
        weight=70.0,
    )

    reach_success_bonus = RewTerm(
        func=custom_mdp.ee_reached_object,
        params={
            "threshold":           0.01,
            "angle_threshold_deg": 3.0,
            "object_cfg":          SceneEntityCfg("object"),
            "ee_frame_cfg":        SceneEntityCfg("ee_frame"),
            "q_offset":            (0.0, 0.0, 0.0, 1.0),
            "grasp_symmetry_quats": _GRASP_SYMMETRY_QUATS_WXYZ,
        },
        weight=0.0,
    )

    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-10)

    collision_penalty = RewTerm(
        func=custom_mdp.contact_force_penalty,
        params={"sensor_name": "grip_contact", "threshold": 0.5},
        weight=-1000.0,
    )


@configclass
class TerminationsCfg:
    lock_joint_step_end = DoneTerm(func=lock_robot_to_cached_joint_target_done)

    collision = DoneTerm(
        func=contact_force_done,
        params={"sensor_name": "grip_contact", "threshold": 0.5},
    )

    time_out = DoneTerm(
        func=time_out_cached_terminal,
        time_out=True,
        params={
            "object_cfg":          SceneEntityCfg("object"),
            "ee_frame_cfg":        SceneEntityCfg("ee_frame"),
            "q_offset":            (0.0, 0.0, 0.0, 1.0),
            "grasp_symmetry_quats": _GRASP_SYMMETRY_QUATS_WXYZ,
        },
    )

    # Default matches vision no-success training: run until timeout.
    # Use set_success_termination_enabled(..., True) to restore early success done.
    reach_success = None


@configclass
class CurriculumCfg:
    pass


##
# 环境配置
##
@configclass
class PickPoseEnvCfg(ManagerBasedRLEnvCfg):
    """xArm7 Pick 位姿真值训练 — 28维状态观测，相对关节增量目标直接写入，scale=0.6°，clip=±0.6°。"""

    scene:        PickPoseSceneCfg = PickPoseSceneCfg(num_envs=64, env_spacing=2.5)
    observations: ObservationsCfg  = ObservationsCfg()
    actions:      ActionsCfg       = ActionsCfg()
    events:       EventCfg         = EventCfg()
    rewards:      RewardsCfg       = RewardsCfg()
    terminations: TerminationsCfg  = TerminationsCfg()
    curriculum:   CurriculumCfg    = CurriculumCfg()

    def __post_init__(self):
        self.decimation        = 2
        self.episode_length_s  = 8

        self.sim.dt              = 0.01
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity               = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 8
        self.sim.physx.gpu_total_aggregate_pairs_capacity      = 1024 * 1024 * 4
        self.sim.physx.friction_correlation_distance           = 0.00625
        self.sim.physx.gpu_max_rigid_patch_count               = 1024 * 1024 * 4


@configclass
class PickPoseEnvCfg_PLAY(PickPoseEnvCfg):
    """回放/测试配置：单环境，关闭训练噪声。"""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs    = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
