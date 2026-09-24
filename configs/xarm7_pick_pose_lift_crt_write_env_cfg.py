# Copyright (c) 2024, Isaac Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""CRT auto-close pick-pose training config with learned lift.

This variant intentionally keeps the pick-pose action/observation surface:

    obs_dim = 21
    action_dim = 7 arm joints

The CRT gripper stays outside the policy action space and uses the same
simulation auto-close trigger as the existing CRT pick-pose config.  There is
no scripted post-grasp lift; the arm policy receives an object-height reward
and must learn the lift motion itself.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

from .xarm7_pick_pose_env_cfg import (
    _GRASP_SYMMETRY_QUATS_WXYZ,
    _TABLE_HEIGHT_RAND,
    EventCfg as PickPoseEventCfg,
    RewardsCfg as PickPoseRewardsCfg,
    _target_quat_candidates_with_grasp_symmetry,
)
from .xarm7_pick_pose_crt_write_env_cfg import (
    AUTO_GRIPPER_CLOSE_DIST_M,
    AUTO_GRIPPER_CLOSE_ORI_DEG,
    PickPoseCRTWriteEnvCfg,
    PickPoseCRTWriteSceneCfg,
    _compute_reach_error_metrics,
    object_pose_relative_to_ee_with_z_offset,
)


_PROJECT_DIR = Path(__file__).resolve().parents[1]
GRASPABLE_OBJECT_USD = str(_PROJECT_DIR / "assets" / "xarm7" / "gongjian_graspable.usd")

LIFT_OBJECT_MASS_KG = 0.05
LIFT_TARGET_DELTA_M = 0.25
LIFT_HEIGHT_REWARD_WEIGHT = 120.0

# Yaw is allowed; this term only rewards keeping the tool approach axis aligned
# with the grasp target axis, which suppresses roll/pitch drift in the air.
EE_YAW_FREE_TILT_STD_RAD = math.radians(8.0)
EE_YAW_FREE_TILT_REWARD_WEIGHT = 35.0


def _ensure_lift_ref_buffer(env: ManagerBasedRLEnv) -> torch.Tensor:
    ref = getattr(env, "_pick_pose_lift_initial_object_z", None)
    if ref is None or ref.shape[0] != env.num_envs or ref.device != env.device:
        ref = torch.zeros(env.num_envs, device=env.device, dtype=torch.float32)
        env._pick_pose_lift_initial_object_z = ref
    return ref


def reset_table_height_and_object_pose_with_lift_ref(
    env: ManagerBasedRLEnv,
    env_ids: torch.Tensor,
    x_range: tuple = (-0.10, 0.10),
    y_range: tuple = (-0.10, 0.10),
    yaw_range: tuple = (-math.pi / 2, math.pi / 2),
    height_range: float = _TABLE_HEIGHT_RAND,
) -> None:
    """Reset table/object exactly like pick-pose and cache this episode's object z0."""
    obj = env.scene["object"]
    table = env.scene["workpiece_table"]
    device = env.device
    n = len(env_ids)

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

    yaw = torch.zeros(n, device=device).uniform_(*yaw_range)
    zeros = torch.zeros(n, device=device)
    q_yaw = math_utils.quat_from_euler_xyz(zeros, zeros, yaw)
    root_state[:, 3:7] = math_utils.quat_mul(q_yaw, root_state[:, 3:7])

    ref = _ensure_lift_ref_buffer(env)
    ref[env_ids] = root_state[:, 2].detach()

    root_state[:, :3] += env.scene.env_origins[env_ids]
    root_state[:, 7:] = 0.0
    obj.write_root_state_to_sim(root_state, env_ids=env_ids)


def object_lift_height_progress(
    env: ManagerBasedRLEnv,
    target_lift_m: float = LIFT_TARGET_DELTA_M,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Linear lift progress reward: x = clamp((z - z0) / target_lift_m, 0, 1)."""
    obj = env.scene[object_cfg.name]
    z_local = obj.data.root_pos_w[:, 2] - env.scene.env_origins[:, 2]

    ref = getattr(env, "_pick_pose_lift_initial_object_z", None)
    if ref is None or ref.shape[0] != env.num_envs:
        ref = obj.data.default_root_state[:, 2].to(device=env.device, dtype=z_local.dtype)
    else:
        ref = ref.to(device=env.device, dtype=z_local.dtype)

    denom = max(float(target_lift_m), 1.0e-6)
    x = (z_local - ref) / denom
    return torch.clamp(x, 0.0, 1.0)


def ee_yaw_free_tilt_alignment(
    env: ManagerBasedRLEnv,
    std: float = EE_YAW_FREE_TILT_STD_RAD,
    q_offset: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    grasp_symmetry_quats: tuple[tuple[float, float, float, float], ...] = _GRASP_SYMMETRY_QUATS_WXYZ,
) -> torch.Tensor:
    """Reward roll/pitch stability while allowing free yaw about the tool axis."""
    ee_frame = env.scene[ee_frame_cfg.name]
    obj = env.scene[object_cfg.name]

    ee_quat_w = ee_frame.data.target_quat_w[:, 0, :]
    target_quat_w = _target_quat_candidates_with_grasp_symmetry(
        obj.data.root_quat_w,
        q_offset=q_offset,
        grasp_symmetry_quats=grasp_symmetry_quats,
    )[:, 0, :]

    local_z = torch.zeros((env.num_envs, 3), device=env.device, dtype=torch.float32)
    local_z[:, 2] = 1.0
    ee_axis_w = math_utils.quat_apply(ee_quat_w, local_z)
    target_axis_w = math_utils.quat_apply(target_quat_w, local_z)

    dot = torch.sum(ee_axis_w * target_axis_w, dim=-1)
    angle = torch.acos(torch.clamp(dot, -1.0, 1.0))
    return torch.exp(-angle / max(float(std), 1.0e-6))


@configclass
class PickPoseLiftCRTWriteSceneCfg(PickPoseCRTWriteSceneCfg):
    """CRT pick-pose scene with a dynamic graspable workpiece."""

    # Isaac Sim 5.1 sometimes fails to spawn PreviewSurface material for this
    # inherited helper cylinder. The robot is gravity-disabled, so the support
    # cylinder is not needed for training physics.
    robot_support_box = None

    object = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/Object",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=[0.9, 4.16, 0.833],
            rot=[0.0, 1.0, 0.0, 0.0],
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=GRASPABLE_OBJECT_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                linear_damping=0.35,
                angular_damping=0.35,
                max_linear_velocity=2.0,
                max_angular_velocity=8.0,
                max_depenetration_velocity=1.5,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=4,
            ),
            mass_props=sim_utils.MassPropertiesCfg(mass=LIFT_OBJECT_MASS_KG),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
        ),
    )


@configclass
class PickPoseLiftEventCfg(PickPoseEventCfg):
    reset_table_and_object = EventTerm(
        func=reset_table_height_and_object_pose_with_lift_ref,
        mode="reset",
        params={
            "x_range": (-0.10, 0.10),
            "y_range": (-0.10, 0.10),
            "yaw_range": (-math.pi / 2, math.pi / 2),
            "height_range": _TABLE_HEIGHT_RAND,
        },
    )


@configclass
class PickPoseLiftRewardsCfg(PickPoseRewardsCfg):
    object_lift_height = RewTerm(
        func=object_lift_height_progress,
        params={
            "target_lift_m": LIFT_TARGET_DELTA_M,
            "object_cfg": SceneEntityCfg("object"),
        },
        weight=LIFT_HEIGHT_REWARD_WEIGHT,
    )

    ee_yaw_free_tilt = RewTerm(
        func=ee_yaw_free_tilt_alignment,
        params={
            "std": EE_YAW_FREE_TILT_STD_RAD,
            "q_offset": (0.0, 0.0, 0.0, 1.0),
            "grasp_symmetry_quats": _GRASP_SYMMETRY_QUATS_WXYZ,
            "ee_frame_cfg": SceneEntityCfg("ee_frame"),
            "object_cfg": SceneEntityCfg("object"),
        },
        weight=EE_YAW_FREE_TILT_REWARD_WEIGHT,
    )


@configclass
class PickPoseLiftCRTWriteEnvCfg(PickPoseCRTWriteEnvCfg):
    """Pick-pose + auto-close gripper + learned lift reward."""

    scene: PickPoseLiftCRTWriteSceneCfg = PickPoseLiftCRTWriteSceneCfg(num_envs=64, env_spacing=2.5)
    events: PickPoseLiftEventCfg = PickPoseLiftEventCfg()
    rewards: PickPoseLiftRewardsCfg = PickPoseLiftRewardsCfg()

    def __post_init__(self):
        super().__post_init__()
        # Keep table collision/friction, but skip visual material creation in
        # this Kit build to avoid null PreviewSurface prim failures.
        if self.scene.workpiece_table is not None and self.scene.workpiece_table.spawn is not None:
            self.scene.workpiece_table.spawn.visual_material = None
        self.actions.arm_action.policy_gripper_enabled = False
        self.actions.arm_action.auto_close_dist_m = AUTO_GRIPPER_CLOSE_DIST_M
        self.actions.arm_action.auto_close_ori_deg = AUTO_GRIPPER_CLOSE_ORI_DEG
        self.observations.policy.object_relative_pose.func = object_pose_relative_to_ee_with_z_offset


@configclass
class PickPoseLiftCRTWriteEnvCfg_PLAY(PickPoseLiftCRTWriteEnvCfg):
    """Play config for the learned-lift CRT auto-close variant."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
