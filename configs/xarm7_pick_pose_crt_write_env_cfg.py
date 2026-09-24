# Copyright (c) 2024, Isaac Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""CRT-gripper pick-pose config with baseline arm write control.

This keeps the original pick-pose training data flow for the arm:

    delta_q = clip(action * 0.6 deg, +/-0.6 deg)
    q_target = q_current_at_action_time + delta_q
    write_joint_state_to_sim(q_target, 0)
    lock arm joints back to q_target at step end

Only the CRT gripper is controlled through actuator targets. By default the
policy remains arm-only (21 obs / 7 actions) and the gripper can auto-close.
When policy_gripper_enabled is set, the policy surface becomes 22 obs / 8
actions with one normalized gripper command: 0=open, 1=close.
"""

from __future__ import annotations

import math
from pathlib import Path

import torch

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass
from isaaclab.utils import math as math_utils

from .controller import (
    DEFAULT_CRT_GRIPPER_PARAMS,
    CRTGripperContactControllerParams,
    make_crt_gripper_contact_controller,
)
from .xarm7_pick_pose_env_cfg import (
    INIT_JOINT_POS,
    _ARM_ACTION_SCALE,
    _ARM_CLIP_RAD,
    _ARM_JOINT_LIMITS_HIGH,
    _ARM_JOINT_LIMITS_LOW,
    _GRASP_SYMMETRY_QUATS_WXYZ,
    _nearest_symmetric_relative_quat,
    _target_quat_candidates_with_grasp_symmetry,
    ActionsCfg as BaseActionsCfg,
    KinematicRelativeJointDirectAction,
    KinematicRelativeJointDirectActionCfg,
    PickPoseEnvCfg,
    PickPoseEnvCfg_PLAY,
    PickPoseSceneCfg,
)


_PROJECT_DIR = Path(__file__).resolve().parents[1]
CRT_TRAINING_ROBOT_USD = str(_PROJECT_DIR / "assets" / "xarm7" / "gx_va_crt_training_robot_write_train.usd")

# Temporary grasp-pose shim for the CRT-write eval path: every place that
# consumes the measured object pose as a target sees it 5 mm higher in world Z.
# The simulated object's physical pose is unchanged.
OBJECT_POSE_Z_OFFSET_M = 0.0
CRT_EPISODE_LENGTH_S = 8.0

# Arm actuator values use the high-PD response tested in the set-control path.
# Write mode still writes joint state directly; set mode relies on these drives.
ARM_EFFORT = 200.0
ARM_STIFFNESS = 3000.0
ARM_DAMPING = 160.0
ARM_VELOCITY_LIMIT = 100.0

# Shared CRT gripper controller defaults. Override the fields on the ActionCfg
# when a reach task needs different trigger thresholds or force-servo gains.
CRT_GRIPPER_PARAMS = DEFAULT_CRT_GRIPPER_PARAMS

GRIPPER_OPEN_DEG = CRT_GRIPPER_PARAMS.open_deg
GRIPPER_CLOSED_DEG = CRT_GRIPPER_PARAMS.closed_deg
AUTO_GRIPPER_CLOSE_DIST_M = CRT_GRIPPER_PARAMS.auto_close_dist_m
AUTO_GRIPPER_CLOSE_ORI_DEG = CRT_GRIPPER_PARAMS.auto_close_ori_deg
AUTO_GRIPPER_CLOSE_TIME_S = CRT_GRIPPER_PARAMS.auto_close_time_s
GRIPPER_FREE_CLOSE_FORCE_N = CRT_GRIPPER_PARAMS.free_close_force_n
GRIPPER_FREE_CLOSE_STEP_DEG = CRT_GRIPPER_PARAMS.free_close_step_deg
GRIPPER_CONTACT_CLOSE_STEP_DEG = CRT_GRIPPER_PARAMS.contact_close_step_deg
GRIPPER_CLOSE_VELOCITY_LIMIT = CRT_GRIPPER_PARAMS.velocity_limit
GRIPPER_PRIMARY_STIFFNESS = CRT_GRIPPER_PARAMS.primary_stiffness
GRIPPER_PRIMARY_DAMPING = CRT_GRIPPER_PARAMS.primary_damping
GRIPPER_PRIMARY_EFFORT = CRT_GRIPPER_PARAMS.primary_effort
GRIPPER_HOLD_EFFORT = CRT_GRIPPER_PARAMS.hold_effort_limit
GRIPPER_HARD_HOLD_EFFORT = CRT_GRIPPER_PARAMS.hard_hold_effort_limit
GRIPPER_HOLD_NORMAL_FORCE_N = CRT_GRIPPER_PARAMS.hold_normal_force_n
GRIPPER_HARD_NORMAL_FORCE_N = CRT_GRIPPER_PARAMS.hard_normal_force_n
GRIPPER_FORCE_CONTROL_ENABLED = CRT_GRIPPER_PARAMS.force_control_enabled
GRIPPER_FORCE_CONTROL_TARGET_N = CRT_GRIPPER_PARAMS.force_control_target_n
GRIPPER_FORCE_CONTROL_PUSH_EFFORT = CRT_GRIPPER_PARAMS.force_control_push_effort_limit
GRIPPER_FORCE_CONTROL_KP_DEG_PER_N = CRT_GRIPPER_PARAMS.force_control_kp_deg_per_n
GRIPPER_FORCE_CONTROL_KI_DEG_PER_N_S = CRT_GRIPPER_PARAMS.force_control_ki_deg_per_n_s
GRIPPER_FORCE_CONTROL_KD_DEG_S_PER_N = CRT_GRIPPER_PARAMS.force_control_kd_deg_s_per_n
GRIPPER_FORCE_CONTROL_INTEGRAL_LIMIT_N_S = CRT_GRIPPER_PARAMS.force_control_integral_limit_n_s
GRIPPER_FORCE_CONTROL_DEADBAND_N = CRT_GRIPPER_PARAMS.force_control_deadband_n
GRIPPER_FORCE_CONTROL_MAX_STEP_DEG = CRT_GRIPPER_PARAMS.force_control_max_step_deg
GRIPPER_FORCE_CONTROL_HOLD_ERROR_DEG = CRT_GRIPPER_PARAMS.force_control_hold_error_deg
GRIPPER_FORCE_CONTROL_RELEASE_STEP_DEG = CRT_GRIPPER_PARAMS.force_control_release_step_deg
GRIPPER_SYNC_ENABLED = CRT_GRIPPER_PARAMS.sync_enabled
GRIPPER_SYNC_MAX_LEAD_DEG = CRT_GRIPPER_PARAMS.sync_max_lead_deg
GRIPPER_MIMIC_STIFFNESS = CRT_GRIPPER_PARAMS.mimic_stiffness
GRIPPER_MIMIC_DAMPING = CRT_GRIPPER_PARAMS.mimic_damping
GRIPPER_MIMIC_EFFORT = CRT_GRIPPER_PARAMS.mimic_effort

GRIPPER_PRIMARY_JOINT_TARGET_MULTIPLIERS = CRT_GRIPPER_PARAMS.primary_joint_target_multipliers_dict()
GRIPPER_MIMIC_JOINT_TARGET_MULTIPLIERS = CRT_GRIPPER_PARAMS.mimic_joint_target_multipliers_dict()
GRIPPER_JOINT_TARGET_MULTIPLIERS = CRT_GRIPPER_PARAMS.joint_target_multipliers_dict()

def gripper_init_joint_pos() -> dict[str, float]:
    return CRT_GRIPPER_PARAMS.init_joint_pos()


def _object_pos_w_with_target_offset(obj_pos_w: torch.Tensor) -> torch.Tensor:
    if abs(OBJECT_POSE_Z_OFFSET_M) <= 1.0e-12:
        return obj_pos_w
    offset = torch.zeros_like(obj_pos_w)
    offset[:, 2] = float(OBJECT_POSE_Z_OFFSET_M)
    return obj_pos_w + offset


def object_pose_relative_to_ee_with_z_offset(
    env: ManagerBasedRLEnv,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    q_offset: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0),
    grasp_symmetry_quats: tuple[tuple[float, float, float, float], ...] = _GRASP_SYMMETRY_QUATS_WXYZ,
) -> torch.Tensor:
    obj = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]

    obj_pos_w = _object_pos_w_with_target_offset(obj.data.root_pos_w)
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
    obj = env.scene[object_cfg.name]
    ee_frame = env.scene[ee_frame_cfg.name]

    obj_pos_w = _object_pos_w_with_target_offset(obj.data.root_pos_w)
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
    w = torch.abs(rel_quat[:, 0])
    xyz_norm = torch.linalg.norm(rel_quat[:, 1:4], dim=-1)
    angle_rad = 2.0 * torch.atan2(xyz_norm, torch.clamp(w, min=1.0e-8))
    ori_err_deg = torch.rad2deg(angle_rad)
    return dist_m, ori_err_deg


CRT_XARM7_WRITE_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=CRT_TRAINING_ROBOT_USD,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=True,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            **INIT_JOINT_POS,
            **gripper_init_joint_pos(),
        }
    ),
    actuators={
        "arm": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-7]"],
            effort_limit_sim=ARM_EFFORT,
            velocity_limit_sim=ARM_VELOCITY_LIMIT,
            stiffness=ARM_STIFFNESS,
            damping=ARM_DAMPING,
        ),
        "crt_gripper_primary": ImplicitActuatorCfg(
            joint_names_expr=list(GRIPPER_PRIMARY_JOINT_TARGET_MULTIPLIERS.keys()),
            effort_limit_sim=GRIPPER_PRIMARY_EFFORT,
            velocity_limit_sim=GRIPPER_CLOSE_VELOCITY_LIMIT,
            stiffness=GRIPPER_PRIMARY_STIFFNESS,
            damping=GRIPPER_PRIMARY_DAMPING,
        ),
        "crt_gripper_mimic": ImplicitActuatorCfg(
            joint_names_expr=list(GRIPPER_MIMIC_JOINT_TARGET_MULTIPLIERS.keys()),
            effort_limit_sim=GRIPPER_MIMIC_EFFORT,
            velocity_limit_sim=GRIPPER_CLOSE_VELOCITY_LIMIT,
            stiffness=GRIPPER_MIMIC_STIFFNESS,
            damping=GRIPPER_MIMIC_DAMPING,
        ),
    },
)


class CRTWriteArmSetGripperAction(KinematicRelativeJointDirectAction):
    """Baseline arm write action plus policy-controlled set-target CRT gripper."""

    cfg: "CRTWriteArmSetGripperActionCfg"

    def __init__(self, cfg: "CRTWriteArmSetGripperActionCfg", env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self._object_name = str(cfg.object_name)
        self._ee_frame_name = str(cfg.ee_frame_name)
        gripper_params = CRT_GRIPPER_PARAMS.with_overrides(
            auto_close_dist_m=float(cfg.auto_close_dist_m),
            auto_close_ori_deg=float(cfg.auto_close_ori_deg),
            auto_close_time_s=float(cfg.auto_close_time_s),
            free_close_force_n=float(cfg.free_close_force_n),
            free_close_step_deg=float(cfg.free_close_step_deg),
            contact_close_step_deg=float(cfg.contact_close_step_deg),
            approach_effort_limit=float(cfg.approach_effort_limit),
            hold_effort_limit=float(cfg.hold_effort_limit),
            hard_hold_effort_limit=float(cfg.hard_hold_effort_limit),
            hold_normal_force_n=float(cfg.hold_normal_force_n),
            hard_normal_force_n=float(cfg.hard_normal_force_n),
            force_control_enabled=bool(cfg.force_control_enabled),
            force_control_target_n=float(cfg.force_control_target_n),
            force_control_push_effort_limit=float(cfg.force_control_push_effort_limit),
            force_control_kp_deg_per_n=float(cfg.force_control_kp_deg_per_n),
            force_control_ki_deg_per_n_s=float(cfg.force_control_ki_deg_per_n_s),
            force_control_kd_deg_s_per_n=float(cfg.force_control_kd_deg_s_per_n),
            force_control_integral_limit_n_s=float(cfg.force_control_integral_limit_n_s),
            force_control_deadband_n=float(cfg.force_control_deadband_n),
            force_control_max_step_deg=float(cfg.force_control_max_step_deg),
            force_control_hold_error_deg=float(cfg.force_control_hold_error_deg),
            force_control_release_step_deg=float(cfg.force_control_release_step_deg),
            velocity_limit=float(cfg.velocity_limit),
            sync_enabled=bool(cfg.gripper_sync_enabled),
            sync_max_lead_deg=float(cfg.gripper_sync_max_lead_deg),
        )
        self.gripper_controller = make_crt_gripper_contact_controller(
            env=self._env,
            asset=self._asset,
            device=self.device,
            num_envs=self.num_envs,
            object_ee_error_fn=self._object_ee_error,
            params=gripper_params,
            action_label=self._action_label(),
            arm_route_label=self._arm_route_label(),
        )
        self._policy_gripper_enabled = bool(cfg.policy_gripper_enabled)
        self.gripper_controller.set_auto_close_enabled(not self._policy_gripper_enabled)

        self._policy_action_dim = self._num_joints + int(self._policy_gripper_enabled)
        self._raw_actions_with_gripper = torch.zeros(
            self.num_envs, self._policy_action_dim, device=self.device, dtype=torch.float32
        )
        self._processed_actions_with_gripper = torch.zeros_like(self._raw_actions_with_gripper)
        self._last_gripper_close_command = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

        if self._policy_gripper_enabled:
            print(
                "[CRTGripperAction] policy action = 7 arm deltas + 1 gripper command "
                "(0=open, 1=close); internal auto-close disabled"
            )
        else:
            print(
                "[CRTGripperAction] policy action = 7 arm deltas only; "
                "gripper kept outside policy action space with auto-close available"
            )

    def _action_label(self) -> str:
        return "CRTWriteArmSetGripperAction"

    def _arm_route_label(self) -> str:
        return "write_joint_state_to_sim + lock"

    def _object_ee_error(self) -> tuple[torch.Tensor, torch.Tensor]:
        return _compute_reach_error_metrics(
            self._env,
            object_cfg=SceneEntityCfg(self._object_name),
            ee_frame_cfg=SceneEntityCfg(self._ee_frame_name),
        )

    @property
    def action_dim(self) -> int:
        return int(getattr(self, "_policy_action_dim", int(getattr(self, "_num_joints", 7))))

    @property
    def raw_actions(self) -> torch.Tensor:
        if hasattr(self, "_raw_actions_with_gripper"):
            return self._raw_actions_with_gripper
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        if hasattr(self, "_processed_actions_with_gripper"):
            return self._processed_actions_with_gripper
        return self._processed_actions

    def reset(self, env_ids=None) -> None:
        self.gripper_controller.reset(env_ids)
        if env_ids is None:
            self._raw_actions_with_gripper.zero_()
            self._processed_actions_with_gripper.zero_()
            self._last_gripper_close_command[:] = False
            self._target_initialized = False
            return

        self._raw_actions_with_gripper[env_ids] = 0.0
        self._processed_actions_with_gripper[env_ids] = 0.0
        self._last_gripper_close_command[env_ids] = False
        if self._target_initialized:
            self._target_joint_pos[env_ids] = self._asset.data.joint_pos[env_ids][:, self._joint_ids]

    def process_actions(self, actions: torch.Tensor):
        arm_actions = actions[:, : self._num_joints]
        KinematicRelativeJointDirectAction.process_actions(self, arm_actions)

        self._raw_actions_with_gripper[:, : self._num_joints] = arm_actions
        self._processed_actions_with_gripper[:, : self._num_joints] = self._processed_actions

        if self._policy_gripper_enabled:
            if actions.shape[-1] > self._num_joints:
                gripper_raw = actions[:, self._num_joints]
            else:
                gripper_raw = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
            gripper_01 = torch.clamp(gripper_raw, 0.0, 1.0)
            close_command = gripper_01 >= 0.5
            close_edges = close_command & (~self._last_gripper_close_command)
            open_edges = (~close_command) & self._last_gripper_close_command
            if torch.any(close_edges):
                self.gripper_controller.request_close(close_edges)
            if torch.any(open_edges):
                self.gripper_controller.request_open(open_edges)
            self._last_gripper_close_command[:] = close_command
            self._raw_actions_with_gripper[:, self._num_joints] = gripper_raw
            self._processed_actions_with_gripper[:, self._num_joints] = close_command.to(dtype=torch.float32)

        self.gripper_controller.update_target()

    def apply_actions(self):
        super().apply_actions()
        self.gripper_controller.apply_targets()

    def lock_to_target_after_physics(self):
        super().lock_to_target_after_physics()
        self.gripper_controller.apply_targets()

class CRTSetArmSetGripperAction(CRTWriteArmSetGripperAction):
    """Same policy action surface, but the arm is driven only by PD set targets."""

    cfg: "CRTSetArmSetGripperActionCfg"

    def _action_label(self) -> str:
        return "CRTSetArmSetGripperAction"

    def _arm_route_label(self) -> str:
        return "set_joint_position_target, no write_joint_state_to_sim"

    def _ensure_arm_target_initialized(self):
        if not self._target_initialized:
            self._target_joint_pos[:] = self._asset.data.joint_pos[:, self._joint_ids]
            self._target_initialized = True

    def _set_arm_position_and_velocity_targets(self):
        joint_pos = self._asset.data.joint_pos.clone()
        joint_vel = self._asset.data.joint_vel.clone()
        q_target = self._target_joint_pos
        joint_pos[:, self._joint_ids] = q_target
        joint_vel[:, self._joint_ids] = 0.0
        self._sync_position_target(joint_pos, q_target)
        self._sync_velocity_target(joint_vel)

    def apply_actions(self):
        self._ensure_arm_target_initialized()
        self._set_arm_position_and_velocity_targets()
        self.gripper_controller.apply_targets()

    def lock_to_target_after_physics(self):
        if self._target_initialized:
            self._set_arm_position_and_velocity_targets()
        self.gripper_controller.apply_targets()


@configclass
class CRTWriteArmSetGripperActionCfg(KinematicRelativeJointDirectActionCfg):
    """Baseline write arm action plus 0/1 policy gripper config."""

    class_type: type = CRTWriteArmSetGripperAction

    object_name: str = "object"
    ee_frame_name: str = "ee_frame"
    auto_close_dist_m: float = AUTO_GRIPPER_CLOSE_DIST_M
    auto_close_ori_deg: float = AUTO_GRIPPER_CLOSE_ORI_DEG
    auto_close_time_s: float = AUTO_GRIPPER_CLOSE_TIME_S
    free_close_force_n: float = GRIPPER_FREE_CLOSE_FORCE_N
    free_close_step_deg: float = GRIPPER_FREE_CLOSE_STEP_DEG
    contact_close_step_deg: float = GRIPPER_CONTACT_CLOSE_STEP_DEG
    approach_effort_limit: float = GRIPPER_PRIMARY_EFFORT
    hold_effort_limit: float = GRIPPER_HOLD_EFFORT
    hard_hold_effort_limit: float = GRIPPER_HARD_HOLD_EFFORT
    hold_normal_force_n: float = GRIPPER_HOLD_NORMAL_FORCE_N
    hard_normal_force_n: float = GRIPPER_HARD_NORMAL_FORCE_N
    force_control_enabled: bool = GRIPPER_FORCE_CONTROL_ENABLED
    force_control_target_n: float = GRIPPER_FORCE_CONTROL_TARGET_N
    force_control_push_effort_limit: float = GRIPPER_FORCE_CONTROL_PUSH_EFFORT
    force_control_kp_deg_per_n: float = GRIPPER_FORCE_CONTROL_KP_DEG_PER_N
    force_control_ki_deg_per_n_s: float = GRIPPER_FORCE_CONTROL_KI_DEG_PER_N_S
    force_control_kd_deg_s_per_n: float = GRIPPER_FORCE_CONTROL_KD_DEG_S_PER_N
    force_control_integral_limit_n_s: float = GRIPPER_FORCE_CONTROL_INTEGRAL_LIMIT_N_S
    force_control_deadband_n: float = GRIPPER_FORCE_CONTROL_DEADBAND_N
    force_control_max_step_deg: float = GRIPPER_FORCE_CONTROL_MAX_STEP_DEG
    force_control_hold_error_deg: float = GRIPPER_FORCE_CONTROL_HOLD_ERROR_DEG
    force_control_release_step_deg: float = GRIPPER_FORCE_CONTROL_RELEASE_STEP_DEG
    velocity_limit: float = GRIPPER_CLOSE_VELOCITY_LIMIT
    policy_gripper_enabled: bool = False
    gripper_sync_enabled: bool = GRIPPER_SYNC_ENABLED
    gripper_sync_max_lead_deg: float = GRIPPER_SYNC_MAX_LEAD_DEG


@configclass
class CRTSetArmSetGripperActionCfg(CRTWriteArmSetGripperActionCfg):
    """PD set-target arm action plus set-target gripper config."""

    class_type: type = CRTSetArmSetGripperAction


@configclass
class PickPoseCRTWriteSceneCfg(PickPoseSceneCfg):
    """Baseline pick-pose scene with the CRT gripper robot USD."""

    robot = CRT_XARM7_WRITE_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
    robot.init_state.pos = (0.9, 4.6, 0.8)
    robot.init_state.rot = (0.707, 0.0, 0.0, -0.707)


@configclass
class CRTWriteActionsCfg(BaseActionsCfg):
    arm_action = CRTWriteArmSetGripperActionCfg(
        asset_name="robot",
        joint_names=["joint[1-7]"],
        scale=_ARM_ACTION_SCALE,
        clip=_ARM_CLIP_RAD,
        joint_limits_low=tuple(_ARM_JOINT_LIMITS_LOW.tolist()),
        joint_limits_high=tuple(_ARM_JOINT_LIMITS_HIGH.tolist()),
        object_name="object",
        ee_frame_name="ee_frame",
        auto_close_dist_m=AUTO_GRIPPER_CLOSE_DIST_M,
        auto_close_ori_deg=AUTO_GRIPPER_CLOSE_ORI_DEG,
        auto_close_time_s=AUTO_GRIPPER_CLOSE_TIME_S,
    )


@configclass
class CRTSetActionsCfg(BaseActionsCfg):
    arm_action = CRTSetArmSetGripperActionCfg(
        asset_name="robot",
        joint_names=["joint[1-7]"],
        scale=_ARM_ACTION_SCALE,
        clip=_ARM_CLIP_RAD,
        joint_limits_low=tuple(_ARM_JOINT_LIMITS_LOW.tolist()),
        joint_limits_high=tuple(_ARM_JOINT_LIMITS_HIGH.tolist()),
        object_name="object",
        ee_frame_name="ee_frame",
        auto_close_dist_m=AUTO_GRIPPER_CLOSE_DIST_M,
        auto_close_ori_deg=AUTO_GRIPPER_CLOSE_ORI_DEG,
        auto_close_time_s=AUTO_GRIPPER_CLOSE_TIME_S,
    )


@configclass
class PickPoseCRTWriteEnvCfg(PickPoseEnvCfg):
    """Baseline pick-pose training with CRT gripper and arm write control."""

    scene: PickPoseCRTWriteSceneCfg = PickPoseCRTWriteSceneCfg(num_envs=64, env_spacing=2.5)
    actions: CRTWriteActionsCfg = CRTWriteActionsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.episode_length_s = CRT_EPISODE_LENGTH_S
        self.observations.policy.object_relative_pose.func = object_pose_relative_to_ee_with_z_offset


@configclass
class PickPoseCRTWriteEnvCfg_PLAY(PickPoseCRTWriteEnvCfg):
    """Play config for the CRT write-control variant."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 1
        self.scene.env_spacing = 2.5
        self.observations.policy.enable_corruption = False
