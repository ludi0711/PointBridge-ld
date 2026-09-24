"""CRT gripper contact controller used by the pick-pose env configs."""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass, replace

import torch


@dataclass(frozen=True)
class CRTGripperContactControllerParams:
    """Tuned CRT gripper parameters shared by reach training and deploy checks.

    The defaults are the headless-tested values from the grasp/lift sweep:
    600/60/40 drive response, 30 N force target, 12 mm / 1 deg auto-close,
    and 8 deg sync lead. Angles are expressed in degrees unless noted.
    """

    open_deg: float = 57.0
    closed_deg: float = 0.0
    auto_close_dist_m: float = 0.012
    auto_close_ori_deg: float = 1.0
    auto_close_time_s: float = 0.45
    free_close_force_n: float = 1.0
    free_close_step_deg: float = 8.0
    contact_close_step_deg: float = 1.5
    velocity_limit: float = 10.0

    primary_stiffness: float = 600.0
    primary_damping: float = 60.0
    primary_effort: float = 40.0
    mimic_stiffness: float = 600.0
    mimic_damping: float = 60.0
    mimic_effort: float = 40.0

    approach_effort_limit: float = 40.0
    hold_effort_limit: float = 30.0
    hard_hold_effort_limit: float = 30.0
    hold_normal_force_n: float = 12.0
    hard_normal_force_n: float = 80.0
    force_control_enabled: bool = True
    force_control_target_n: float = 30.0
    force_control_push_effort_limit: float = 40.0
    force_control_kp_deg_per_n: float = 0.08
    force_control_ki_deg_per_n_s: float = 0.0
    force_control_kd_deg_s_per_n: float = 0.0
    force_control_integral_limit_n_s: float = 50.0
    force_control_deadband_n: float = 2.0
    force_control_max_step_deg: float = 1.0
    force_control_hold_error_deg: float = 0.35
    force_control_release_step_deg: float = 0.0
    sync_enabled: bool = True
    sync_max_lead_deg: float = 8.0

    primary_joint_target_multipliers: tuple[tuple[str, float], ...] = (
        ("leftfinger_joint", 1.0),
        ("rightfinger_joint", 1.0),
    )
    mimic_joint_target_multipliers: tuple[tuple[str, float], ...] = (
        ("left_kckle_joint", 1.0),
        ("right_kckle_joint", 1.0),
        ("leftinn_joint", -1.0),
        ("rightinn_joint", -1.0),
    )
    gripper_actuator_names: tuple[str, ...] = ("crt_gripper_primary", "crt_gripper_mimic")
    left_pad_body_name: str = "left_pad"
    right_pad_body_name: str = "right_pad"
    left_pad_contact_sensor_name: str = "left_pad_contact"
    right_pad_contact_sensor_name: str = "right_pad_contact"

    def with_overrides(self, **kwargs) -> "CRTGripperContactControllerParams":
        clean = {key: value for key, value in kwargs.items() if value is not None}
        return replace(self, **clean)

    def primary_joint_target_multipliers_dict(self) -> dict[str, float]:
        return dict(self.primary_joint_target_multipliers)

    def mimic_joint_target_multipliers_dict(self) -> dict[str, float]:
        return dict(self.mimic_joint_target_multipliers)

    def joint_target_multipliers_dict(self) -> dict[str, float]:
        return {
            **self.primary_joint_target_multipliers_dict(),
            **self.mimic_joint_target_multipliers_dict(),
        }

    def init_joint_pos(self) -> dict[str, float]:
        return {
            name: math.radians(self.open_deg * multiplier)
            for name, multiplier in self.joint_target_multipliers_dict().items()
        }


DEFAULT_CRT_GRIPPER_PARAMS = CRTGripperContactControllerParams()


class CRTGripperContactController:
    """Set-target gripper controller with contact-force hold, force servo, and sync limiting."""

    def __init__(
        self,
        *,
        env,
        asset,
        device,
        num_envs: int,
        object_ee_error_fn: Callable[[], tuple[torch.Tensor, torch.Tensor]],
        joint_target_multipliers: Mapping[str, float],
        primary_joint_target_multipliers: Mapping[str, float],
        open_deg: float,
        closed_deg: float,
        auto_close_dist_m: float,
        auto_close_ori_deg: float,
        auto_close_time_s: float,
        free_close_force_n: float,
        free_close_step_deg: float,
        contact_close_step_deg: float,
        approach_effort_limit: float,
        hold_effort_limit: float,
        hard_hold_effort_limit: float,
        hold_normal_force_n: float,
        hard_normal_force_n: float,
        force_control_enabled: bool,
        force_control_target_n: float,
        force_control_push_effort_limit: float,
        force_control_kp_deg_per_n: float,
        force_control_ki_deg_per_n_s: float,
        force_control_kd_deg_s_per_n: float,
        force_control_integral_limit_n_s: float,
        force_control_deadband_n: float,
        force_control_max_step_deg: float,
        force_control_hold_error_deg: float,
        force_control_release_step_deg: float,
        velocity_limit: float,
        gripper_actuator_names: tuple[str, ...],
        sync_enabled: bool,
        sync_max_lead_deg: float,
        action_label: str,
        arm_route_label: str,
        left_pad_body_name: str = "left_pad",
        right_pad_body_name: str = "right_pad",
        left_pad_contact_sensor_name: str = "left_pad_contact",
        right_pad_contact_sensor_name: str = "right_pad_contact",
    ):
        self._env = env
        self._asset = asset
        self.device = device
        self.num_envs = int(num_envs)
        self._object_ee_error_fn = object_ee_error_fn
        self._joint_target_multipliers = dict(joint_target_multipliers)
        self._primary_joint_target_multipliers = dict(primary_joint_target_multipliers)
        self._action_label = str(action_label)
        self._arm_route_label = str(arm_route_label)
        self._left_pad_body_name = str(left_pad_body_name)
        self._right_pad_body_name = str(right_pad_body_name)
        self._left_pad_contact_sensor_name = str(left_pad_contact_sensor_name)
        self._right_pad_contact_sensor_name = str(right_pad_contact_sensor_name)

        self._auto_close_dist_m = float(auto_close_dist_m)
        self._auto_close_ori_deg = float(auto_close_ori_deg)
        self._auto_close_enabled = True
        self._auto_close_time_s = float(auto_close_time_s)
        self._free_close_force_n = max(0.0, float(free_close_force_n))
        self._free_close_step_rad = math.radians(max(0.0, float(free_close_step_deg)))
        self._contact_close_step_rad = math.radians(max(0.0, float(contact_close_step_deg)))
        self._approach_effort_limit = max(0.0, float(approach_effort_limit))
        self._hold_effort_limit = max(0.0, float(hold_effort_limit))
        self._hard_hold_effort_limit = max(0.0, float(hard_hold_effort_limit))
        self._hold_normal_force_n = max(0.0, float(hold_normal_force_n))
        self._hard_normal_force_n = max(self._hold_normal_force_n, float(hard_normal_force_n))
        self._force_control_enabled = bool(force_control_enabled)
        self._force_control_target_n = max(0.0, float(force_control_target_n))
        self._force_control_push_effort_limit = max(0.0, float(force_control_push_effort_limit))
        self._force_control_kp_deg_per_n = max(0.0, float(force_control_kp_deg_per_n))
        self._force_control_ki_deg_per_n_s = max(0.0, float(force_control_ki_deg_per_n_s))
        self._force_control_kd_deg_s_per_n = max(0.0, float(force_control_kd_deg_s_per_n))
        self._force_control_integral_limit_n_s = max(0.0, float(force_control_integral_limit_n_s))
        self._force_control_deadband_n = max(0.0, float(force_control_deadband_n))
        self._force_control_max_step_rad = math.radians(max(0.0, float(force_control_max_step_deg)))
        self._force_control_hold_error_rad = math.radians(max(0.0, float(force_control_hold_error_deg)))
        self._force_control_release_step_rad = math.radians(max(0.0, float(force_control_release_step_deg)))
        self._velocity_limit = max(0.0, float(velocity_limit))
        self._gripper_actuator_names = tuple(str(name) for name in gripper_actuator_names)
        self._gripper_sync_enabled = bool(sync_enabled)
        self._gripper_sync_max_lead_rad = math.radians(max(0.0, float(sync_max_lead_deg)))
        self._auto_close_alpha_step = self._env_step_dt_s() / max(self._auto_close_time_s, 1.0e-6)
        self._auto_close_alpha_step = min(1.0, max(0.0, self._auto_close_alpha_step))

        self._gripper_ids: dict[str, int] = {}
        missing_gripper_joints: list[str] = []
        for joint_name in self._joint_target_multipliers:
            ids, _ = self._asset.find_joints([joint_name])
            ids = [int(v) for v in list(ids)]
            if ids:
                self._gripper_ids[joint_name] = ids[0]
            else:
                missing_gripper_joints.append(joint_name)
        if missing_gripper_joints:
            raise RuntimeError(f"[{self._action_label}] missing gripper joints: {missing_gripper_joints}")

        self._gripper_joint_ids = [self._gripper_ids[name] for name in self._joint_target_multipliers]
        self._gripper_joint_names = list(self._joint_target_multipliers.keys())
        self._gripper_target_multipliers = torch.tensor(
            [float(self._joint_target_multipliers[name]) for name in self._gripper_joint_names],
            device=self.device,
            dtype=torch.float32,
        )
        self._gripper_joint_sides = torch.tensor(
            [1 if name.startswith("right") else 0 for name in self._gripper_joint_names],
            device=self.device,
            dtype=torch.long,
        )
        self._left_gripper_cols = [idx for idx, name in enumerate(self._gripper_joint_names) if name.startswith("left")]
        self._right_gripper_cols = [idx for idx, name in enumerate(self._gripper_joint_names) if name.startswith("right")]
        self._primary_gripper_joint_names = [
            name for name in self._primary_joint_target_multipliers if name in self._gripper_ids
        ]
        self._primary_gripper_joint_ids = [self._gripper_ids[name] for name in self._primary_gripper_joint_names]
        self._left_pad_body_id = None
        self._right_pad_body_id = None
        try:
            body_ids, _ = self._asset.find_bodies([self._left_pad_body_name, self._right_pad_body_name], preserve_order=True)
            body_ids = [int(v) for v in list(body_ids)]
            if len(body_ids) >= 2:
                self._left_pad_body_id = body_ids[0]
                self._right_pad_body_id = body_ids[1]
        except Exception:
            pass

        self._gripper_target_joint_pos = torch.zeros(
            self.num_envs,
            len(self._gripper_joint_ids),
            device=self.device,
            dtype=torch.float32,
        )
        self._zero_gripper_vel_target = torch.zeros_like(self._gripper_target_joint_pos)
        self._gripper_target_initialized = False
        self._gripper_closed_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._gripper_force_hold_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._gripper_hard_force_warn_mask = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._gripper_close_alpha = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._gripper_force_servo_integral_n_s = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float32)
        self._gripper_force_servo_prev_error_n = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float32)
        self._gripper_force_servo_initialized = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._gripper_force_servo_prev_error_initialized = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        self._gripper_side_effort_hold_mask = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.bool)
        self._gripper_side_effort_hard_mask = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.bool)
        self._debug_force_servo_control_deg = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._debug_force_servo_delta_deg = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._debug_gripper_scalar_target_deg = torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        self._gripper_closed_pos = math.radians(float(closed_deg))
        self._gripper_open_pos = math.radians(float(open_deg))
        self._gripper_scalar_target = torch.full(
            (self.num_envs,),
            self._gripper_open_pos,
            device=self.device,
            dtype=torch.float32,
        )
        self._gripper_side_scalar_target = torch.full(
            (self.num_envs, 2),
            self._gripper_open_pos,
            device=self.device,
            dtype=torch.float32,
        )
        self._closed_deg = float(closed_deg)
        self._warned_gripper_position_target = False
        self._warned_gripper_velocity_target = False

        print(
            f"[{self._action_label}] enabled | "
            f"arm route={self._arm_route_label} | "
            f"gripper_joints={self._gripper_joint_names} | "
            f"auto_close={self._auto_close_dist_m * 1000.0:.1f} mm/{self._auto_close_ori_deg:.1f} deg | "
            f"close_ramp={self._auto_close_time_s:.2f}s | "
            f"free_close={math.degrees(self._free_close_step_rad):.1f}deg/step below {self._free_close_force_n:.1f}N | "
            f"contact_close={math.degrees(self._contact_close_step_rad):.1f}deg/step | "
            f"q_close={self._closed_deg:.1f}deg with effort limiting | "
            f"pad_normal_hold={self._hold_normal_force_n:.1f}N hard_warn={self._hard_normal_force_n:.1f}N | "
            f"force_servo={'on' if self._force_control_enabled else 'off'} target={self._force_control_target_n:.1f}N push_effort={self._force_control_push_effort_limit:.1f} | "
            f"pid=({self._force_control_kp_deg_per_n:.3f},{self._force_control_ki_deg_per_n_s:.3f},{self._force_control_kd_deg_s_per_n:.3f}) "
            f"hold_err={math.degrees(self._force_control_hold_error_rad):.2f}deg release_step={math.degrees(self._force_control_release_step_rad):.2f}deg | "
            f"sync={'on' if self._gripper_sync_enabled else 'off'} lead={math.degrees(self._gripper_sync_max_lead_rad):.1f}deg | "
            f"vel_limit={self._velocity_limit:.2f}rad/s | "
            f"effort approach/hold/hard={self._approach_effort_limit:.1f}/{self._hold_effort_limit:.1f}/{self._hard_hold_effort_limit:.1f} | "
            "gripper route=set_joint_position_target"
        )

    def _env_step_dt_s(self) -> float:
        cfg = getattr(self._env, "cfg", None)
        if cfg is not None:
            sim_cfg = getattr(cfg, "sim", None)
            sim_dt = getattr(sim_cfg, "dt", None) if sim_cfg is not None else None
            decimation = getattr(cfg, "decimation", None)
            if sim_dt is not None and decimation is not None:
                return float(sim_dt) * float(decimation)
        step_dt = getattr(self._env, "step_dt", None)
        if step_dt is not None:
            return float(step_dt)
        return 0.02

    def reset(self, env_ids=None) -> None:
        if env_ids is None:
            self._gripper_closed_mask[:] = False
            self._gripper_force_hold_mask[:] = False
            self._gripper_hard_force_warn_mask[:] = False
            self._gripper_close_alpha[:] = 0.0
            self._gripper_scalar_target[:] = self._gripper_open_pos
            self._gripper_side_scalar_target[:] = self._gripper_open_pos
            self._gripper_force_servo_initialized[:] = False
            self._gripper_side_effort_hold_mask[:] = False
            self._gripper_side_effort_hard_mask[:] = False
            self._gripper_force_servo_integral_n_s[:] = 0.0
            self._gripper_force_servo_prev_error_n[:] = 0.0
            self._gripper_force_servo_prev_error_initialized[:] = False
            self._set_gripper_effort_limits()
            self._set_gripper_velocity_limits()
            self._gripper_target_initialized = False
            return
        self._gripper_closed_mask[env_ids] = False
        self._gripper_force_hold_mask[env_ids] = False
        self._gripper_hard_force_warn_mask[env_ids] = False
        self._gripper_close_alpha[env_ids] = 0.0
        self._gripper_scalar_target[env_ids] = self._gripper_open_pos
        self._gripper_side_scalar_target[env_ids] = self._gripper_open_pos
        self._gripper_force_servo_initialized[env_ids] = False
        self._gripper_side_effort_hold_mask[env_ids] = False
        self._gripper_side_effort_hard_mask[env_ids] = False
        self._gripper_force_servo_integral_n_s[env_ids] = 0.0
        self._gripper_force_servo_prev_error_n[env_ids] = 0.0
        self._gripper_force_servo_prev_error_initialized[env_ids] = False
        self._set_gripper_effort_limits()
        self._set_gripper_velocity_limits()
        if self._gripper_target_initialized:
            self._gripper_target_joint_pos[env_ids] = self._asset.data.joint_pos[env_ids][:, self._gripper_joint_ids]
            self._gripper_side_scalar_target[env_ids] = self._gripper_side_live_scalar()[env_ids]
            self._gripper_scalar_target[env_ids] = torch.mean(self._gripper_side_scalar_target[env_ids], dim=-1)

    def set_auto_close_enabled(self, enabled: bool) -> None:
        self._auto_close_enabled = bool(enabled)

    def request_close(self, env_ids=None) -> None:
        if env_ids is None:
            self._gripper_closed_mask[:] = True
            self._gripper_force_hold_mask[:] = False
            self._gripper_hard_force_warn_mask[:] = False
            self._reset_force_servo_state()
        else:
            self._gripper_closed_mask[env_ids] = True
            self._gripper_force_hold_mask[env_ids] = False
            self._gripper_hard_force_warn_mask[env_ids] = False
            self._reset_force_servo_state(env_ids)
        self._set_gripper_effort_limits()

    def request_open(self, env_ids=None) -> None:
        if env_ids is None:
            self._gripper_closed_mask[:] = False
            self._gripper_force_hold_mask[:] = False
            self._gripper_hard_force_warn_mask[:] = False
            self._gripper_close_alpha[:] = 0.0
            self._gripper_scalar_target[:] = self._gripper_open_pos
            self._gripper_side_scalar_target[:] = self._gripper_open_pos
            self._reset_force_servo_state()
        else:
            self._gripper_closed_mask[env_ids] = False
            self._gripper_force_hold_mask[env_ids] = False
            self._gripper_hard_force_warn_mask[env_ids] = False
            self._gripper_close_alpha[env_ids] = 0.0
            self._gripper_scalar_target[env_ids] = self._gripper_open_pos
            self._gripper_side_scalar_target[env_ids] = self._gripper_open_pos
            self._reset_force_servo_state(env_ids)
        self._set_gripper_effort_limits()

    def update_target(self) -> None:
        dist_m, ori_err_deg = self._object_ee_error_fn()
        if self._auto_close_enabled:
            close_now = (dist_m <= self._auto_close_dist_m) & (ori_err_deg <= self._auto_close_ori_deg)
        else:
            close_now = torch.zeros_like(self._gripper_closed_mask)
        self._gripper_closed_mask |= close_now

        left_n, right_n = self._gripper_pad_normal_forces_n()
        if self._hold_normal_force_n > 0.0:
            both_hold_force = (left_n >= self._hold_normal_force_n) & (right_n >= self._hold_normal_force_n)
            prev_hold = self._gripper_force_hold_mask.clone()
            live_hold = both_hold_force & self._gripper_closed_mask
            new_hold = live_hold & (~prev_hold)
            released_hold = prev_hold & (~live_hold)
            self._gripper_force_hold_mask[:] = live_hold
            if torch.any(new_hold):
                ids = torch.nonzero(new_hold, as_tuple=False).flatten().detach().cpu().tolist()
                left_vals = left_n[new_hold].detach().cpu().tolist()
                right_vals = right_n[new_hold].detach().cpu().tolist()
                print(
                    f"[GripperForceLimit] live envs={ids} | "
                    f"left={left_vals} N right={right_vals} N | "
                    f"effort {self._approach_effort_limit:.2f}->{self._hold_effort_limit:.2f}",
                    flush=True,
                )
            if torch.any(new_hold) or torch.any(released_hold):
                self._set_gripper_effort_limits()
                if torch.any(new_hold):
                    try:
                        actual_limits = self._asset.data.joint_effort_limits[new_hold][:, self._gripper_joint_ids].detach().cpu().tolist()
                        print(f"[GripperForceLimit] actual joint_effort_limits={actual_limits}", flush=True)
                    except Exception as exc:
                        print(f"[GripperForceLimit] actual limit read failed: {type(exc).__name__}: {exc}", flush=True)

        active = self._gripper_closed_mask
        gripper_side_target = self._gripper_side_scalar_target.clone()
        if torch.any(active):
            left_contact = left_n > self._free_close_force_n
            right_contact = right_n > self._free_close_force_n
            both_contact = left_contact & right_contact
            servo_start_force_n = max(self._free_close_force_n, self._hold_normal_force_n)
            both_servo_ready = (left_n >= servo_start_force_n) & (right_n >= servo_start_force_n)
            if self._force_control_enabled:
                servo_ready = self._gripper_force_servo_initialized | both_servo_ready | self._gripper_force_hold_mask
            else:
                servo_ready = torch.zeros_like(active)
            free_close = active & (~both_contact) & (~servo_ready) & (~self._gripper_force_hold_mask)
            # Light dual contact keeps closing both fingers slowly for self-centering.
            contact_close = active & both_contact & (~servo_ready) & (~self._gripper_force_hold_mask)
            close_step = torch.zeros_like(self._gripper_scalar_target)
            close_step[free_close] = self._free_close_step_rad
            close_step[contact_close] = self._contact_close_step_rad
            gripper_side_target[active] = gripper_side_target[active] - close_step[active, None]
            gripper_side_target = torch.clamp(
                gripper_side_target,
                min=min(self._gripper_closed_pos, self._gripper_open_pos),
                max=max(self._gripper_closed_pos, self._gripper_open_pos),
            )
        gripper_side_target = self._apply_gripper_force_servo(gripper_side_target, left_n, right_n)
        gripper_side_target = self._apply_gripper_sync_limit(gripper_side_target, left_n, right_n)
        self._gripper_side_scalar_target[:] = gripper_side_target
        gripper_scalar_target = torch.mean(gripper_side_target, dim=-1)
        self._gripper_scalar_target[:] = gripper_scalar_target
        self._debug_gripper_scalar_target_deg[:] = torch.rad2deg(gripper_scalar_target)
        for col, joint_name in enumerate(self._joint_target_multipliers):
            side = int(self._gripper_joint_sides[col].detach().cpu().item())
            self._gripper_target_joint_pos[:, col] = gripper_side_target[:, side] * float(self._joint_target_multipliers[joint_name])
        self._gripper_target_initialized = True

    def apply_targets(self) -> None:
        self._ensure_gripper_target_initialized()
        self._set_gripper_position_target()
        self._set_gripper_velocity_target()

    def _contact_sensor_normal_force_sum(self, sensor_name: str, axis: torch.Tensor | None) -> torch.Tensor:
        try:
            sensor = self._env.scene[sensor_name]
        except Exception:
            try:
                sensor = self._env.scene.sensors[sensor_name]
            except Exception:
                return torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        data = getattr(sensor, "data", None)
        if data is None:
            return torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)
        for attr_name in ("force_matrix_w", "net_forces_w", "net_forces_w_history"):
            tensor = getattr(data, attr_name, None)
            if tensor is None:
                continue
            try:
                forces = torch.nan_to_num(tensor.detach().to(device=self.device), nan=0.0, posinf=0.0, neginf=0.0)
                forces = forces.reshape(self.num_envs, -1, 3)
                if axis is None:
                    normal_n = torch.linalg.norm(forces, dim=-1).sum(dim=-1)
                else:
                    normal_n = torch.abs(torch.sum(forces * axis[:, None, :], dim=-1)).sum(dim=-1)
                if torch.any(normal_n > 1.0e-6):
                    return normal_n.float()
            except Exception:
                continue
        return torch.zeros(self.num_envs, device=self.device, dtype=torch.float32)

    def _pad_axis_w(self) -> torch.Tensor | None:
        if self._left_pad_body_id is None or self._right_pad_body_id is None:
            return None
        try:
            left_pos = self._asset.data.body_pos_w[:, self._left_pad_body_id, :]
            right_pos = self._asset.data.body_pos_w[:, self._right_pad_body_id, :]
            axis = right_pos - left_pos
            norm = torch.linalg.norm(axis, dim=-1, keepdim=True)
            if not torch.any(norm > 1.0e-8):
                return None
            return axis / torch.clamp(norm, min=1.0e-8)
        except Exception:
            return None

    def _gripper_pad_normal_forces_n(self) -> tuple[torch.Tensor, torch.Tensor]:
        axis = self._pad_axis_w()
        left_n = self._contact_sensor_normal_force_sum(self._left_pad_contact_sensor_name, axis)
        right_n = self._contact_sensor_normal_force_sum(self._right_pad_contact_sensor_name, axis)
        return left_n.float(), right_n.float()

    def _gripper_side_live_scalar(self) -> torch.Tensor:
        live_scalar = self._asset.data.joint_pos[:, self._gripper_joint_ids] / self._gripper_target_multipliers.view(1, -1)
        live_scalar = torch.nan_to_num(live_scalar, nan=0.0, posinf=0.0, neginf=0.0)
        out = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float32)
        if self._left_gripper_cols:
            out[:, 0] = live_scalar[:, self._left_gripper_cols].mean(dim=-1)
        else:
            out[:, 0] = live_scalar.mean(dim=-1)
        if self._right_gripper_cols:
            out[:, 1] = live_scalar[:, self._right_gripper_cols].mean(dim=-1)
        else:
            out[:, 1] = live_scalar.mean(dim=-1)
        return out

    def _apply_gripper_sync_limit(
        self,
        gripper_side_target: torch.Tensor,
        left_n: torch.Tensor | None = None,
        right_n: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if not self._gripper_sync_enabled or self._gripper_sync_max_lead_rad <= 0.0:
            return gripper_side_target
        try:
            synced = gripper_side_target.clone()
            left = gripper_side_target[:, 0]
            right = gripper_side_target[:, 1]
            left_too_closed = left < (right - self._gripper_sync_max_lead_rad)
            right_too_closed = right < (left - self._gripper_sync_max_lead_rad)
            synced[:, 0] = torch.where(left_too_closed, right - self._gripper_sync_max_lead_rad, synced[:, 0])
            synced[:, 1] = torch.where(right_too_closed, left - self._gripper_sync_max_lead_rad, synced[:, 1])
            return torch.clamp(
                synced,
                min=min(self._gripper_closed_pos, self._gripper_open_pos),
                max=max(self._gripper_closed_pos, self._gripper_open_pos),
            )
        except Exception:
            return gripper_side_target

    def _apply_gripper_force_servo(
        self,
        gripper_side_target: torch.Tensor,
        left_n: torch.Tensor,
        right_n: torch.Tensor,
    ) -> torch.Tensor:
        if not self._force_control_enabled or self._force_control_target_n <= 0.0:
            self._reset_force_servo_state()
            return gripper_side_target
        force_n = torch.stack((left_n, right_n), dim=-1)
        servo_start_force_n = max(self._free_close_force_n, self._hold_normal_force_n)
        contact = force_n >= servo_start_force_n
        both_contact = torch.all(contact, dim=-1)
        active = self._gripper_closed_mask & (
            self._gripper_force_hold_mask | both_contact | self._gripper_force_servo_initialized
        )
        if not torch.any(active):
            self._reset_force_servo_state()
            return gripper_side_target

        newly_active = active & (~self._gripper_force_servo_initialized)
        if torch.any(newly_active):
            self._gripper_force_servo_initialized[newly_active] = True
        inactive = ~active
        if torch.any(inactive):
            self._reset_force_servo_state(inactive)

        side_effort_hold = active[:, None] & (force_n >= (self._force_control_target_n - self._force_control_deadband_n))
        side_effort_hard = active[:, None] & (force_n >= self._hard_normal_force_n)
        self._gripper_side_effort_hold_mask[active] = side_effort_hold[active]
        self._gripper_side_effort_hard_mask[active] = side_effort_hard[active]
        self._set_gripper_effort_limits()

        side_target = gripper_side_target.clone()
        if self._force_control_hold_error_rad > 0.0:
            live_side = self._gripper_side_live_scalar()
            preload_target = live_side - self._force_control_hold_error_rad
            high_force = force_n >= (self._force_control_target_n + self._force_control_deadband_n)
            high_active = active[:, None] & high_force
            release_target = preload_target
            if self._force_control_release_step_rad > 0.0:
                release_target = torch.minimum(
                    preload_target,
                    side_target + self._force_control_release_step_rad,
                )
            side_target = torch.where(high_active & (side_target < preload_target), release_target, side_target)

        error_n = torch.clamp(self._force_control_target_n - force_n, min=0.0)
        if self._force_control_deadband_n > 0.0:
            in_deadband = error_n <= self._force_control_deadband_n
            error_n = torch.where(in_deadband, torch.zeros_like(error_n), error_n)
        dt_s = max(self._env_step_dt_s(), 1.0e-6)
        self._gripper_force_servo_integral_n_s[active] += error_n[active] * dt_s
        if self._force_control_integral_limit_n_s > 0.0:
            self._gripper_force_servo_integral_n_s.clamp_(
                min=-self._force_control_integral_limit_n_s,
                max=self._force_control_integral_limit_n_s,
            )
        prev_ready = active & self._gripper_force_servo_prev_error_initialized
        derivative_n_per_s = torch.zeros_like(error_n)
        derivative_n_per_s[prev_ready] = (
            error_n[prev_ready] - self._gripper_force_servo_prev_error_n[prev_ready]
        ) / dt_s
        self._gripper_force_servo_prev_error_n[active] = error_n[active]
        self._gripper_force_servo_prev_error_initialized[active] = True
        control_deg = (
            self._force_control_kp_deg_per_n * error_n
            + self._force_control_ki_deg_per_n_s * self._gripper_force_servo_integral_n_s
            + self._force_control_kd_deg_s_per_n * derivative_n_per_s
        )
        control_deg = torch.clamp(control_deg, min=0.0)
        delta_rad = -control_deg * (math.pi / 180.0)
        if self._force_control_max_step_rad > 0.0:
            delta_rad = torch.clamp(delta_rad, -self._force_control_max_step_rad, 0.0)
        side_target[active] = side_target[active] + delta_rad[active]
        side_target = torch.clamp(
            side_target,
            min=min(self._gripper_closed_pos, self._gripper_open_pos),
            max=max(self._gripper_closed_pos, self._gripper_open_pos),
        )
        self._debug_force_servo_control_deg[active] = torch.max(control_deg[active], dim=-1).values
        self._debug_force_servo_delta_deg[active] = torch.rad2deg(torch.mean(delta_rad[active], dim=-1))
        return side_target

    def _reset_force_servo_state(self, env_mask: torch.Tensor | None = None) -> None:
        if env_mask is None:
            self._gripper_force_servo_initialized[:] = False
            self._gripper_side_effort_hold_mask[:] = False
            self._gripper_side_effort_hard_mask[:] = False
            self._gripper_force_servo_integral_n_s[:] = 0.0
            self._gripper_force_servo_prev_error_n[:] = 0.0
            self._gripper_force_servo_prev_error_initialized[:] = False
            self._debug_force_servo_control_deg[:] = 0.0
            self._debug_force_servo_delta_deg[:] = 0.0
            return
        self._gripper_force_servo_initialized[env_mask] = False
        self._gripper_side_effort_hold_mask[env_mask] = False
        self._gripper_side_effort_hard_mask[env_mask] = False
        self._gripper_force_servo_integral_n_s[env_mask] = 0.0
        self._gripper_force_servo_prev_error_n[env_mask] = 0.0
        self._gripper_force_servo_prev_error_initialized[env_mask] = False
        self._debug_force_servo_control_deg[env_mask] = 0.0
        self._debug_force_servo_delta_deg[env_mask] = 0.0

    def debug_plot_values(self, env_index: int = 0) -> dict[str, float]:
        env_index = int(env_index)
        out = {
            "target_deg": 0.0,
            "actual_left_deg": 0.0,
            "actual_right_deg": 0.0,
            "actual_mean_deg": 0.0,
            "tracking_error_deg": 0.0,
            "u_deg": 0.0,
            "delta_deg": 0.0,
        }
        try:
            actual = self._asset.data.joint_pos[env_index, self._gripper_joint_ids] / self._gripper_target_multipliers
            actual = torch.nan_to_num(actual, nan=0.0, posinf=0.0, neginf=0.0)
            actual_mean_deg = float(torch.rad2deg(actual.mean()).detach().cpu().item())
            out["actual_mean_deg"] = actual_mean_deg
            if self._gripper_target_initialized:
                target = self._gripper_target_joint_pos[env_index] / self._gripper_target_multipliers
                target = torch.nan_to_num(target, nan=0.0, posinf=0.0, neginf=0.0)
                target_deg = float(torch.rad2deg(target.mean()).detach().cpu().item())
                side_target = self._gripper_side_scalar_target[env_index]
                out["target_left_deg"] = float(torch.rad2deg(side_target[0]).detach().cpu().item())
                out["target_right_deg"] = float(torch.rad2deg(side_target[1]).detach().cpu().item())
            else:
                target_deg = actual_mean_deg
                out["target_left_deg"] = target_deg
                out["target_right_deg"] = target_deg
            out["target_deg"] = target_deg
            out["tracking_error_deg"] = target_deg - actual_mean_deg
        except Exception:
            pass
        try:
            primary_scalars = []
            for joint_id, joint_name in zip(self._primary_gripper_joint_ids, self._primary_gripper_joint_names, strict=False):
                multiplier = float(self._joint_target_multipliers.get(joint_name, 1.0))
                scalar = self._asset.data.joint_pos[env_index, joint_id] / multiplier
                primary_scalars.append(float(torch.rad2deg(scalar).detach().cpu().item()))
            if primary_scalars:
                out["actual_left_deg"] = primary_scalars[0]
                out["actual_right_deg"] = primary_scalars[-1] if len(primary_scalars) > 1 else primary_scalars[0]
        except Exception:
            pass
        try:
            out["u_deg"] = float(self._debug_force_servo_control_deg[env_index].detach().cpu().item())
            out["delta_deg"] = float(self._debug_force_servo_delta_deg[env_index].detach().cpu().item())
        except Exception:
            pass
        try:
            left_n, right_n = self._gripper_pad_normal_forces_n()
            left_f = float(left_n[env_index].detach().cpu().item())
            right_f = float(right_n[env_index].detach().cpu().item())
            out["left_contact"] = float(left_f > self._free_close_force_n)
            out["right_contact"] = float(right_f > self._free_close_force_n)
            out["both_contact"] = float((left_f > self._free_close_force_n) and (right_f > self._free_close_force_n))
            out["closed_mask"] = float(bool(self._gripper_closed_mask[env_index].detach().cpu().item()))
            out["hold_mask"] = float(bool(self._gripper_force_hold_mask[env_index].detach().cpu().item()))
            out["force_servo_active"] = float(bool(self._gripper_force_servo_initialized[env_index].detach().cpu().item()))
        except Exception:
            pass
        try:
            vel = self._asset.data.joint_vel[env_index, self._gripper_joint_ids] / self._gripper_target_multipliers
            vel = torch.nan_to_num(vel, nan=0.0, posinf=0.0, neginf=0.0)
            out["actual_vel_mean_deg_s"] = float(torch.rad2deg(vel.mean()).detach().cpu().item())
        except Exception:
            pass
        try:
            limits = self._asset.data.joint_effort_limits[env_index, self._gripper_joint_ids]
            limits = torch.nan_to_num(limits, nan=0.0, posinf=0.0, neginf=0.0)
            out["effort_limit_min"] = float(limits.min().detach().cpu().item())
            out["effort_limit_max"] = float(limits.max().detach().cpu().item())
        except Exception:
            pass
        return out

    def _fill_gripper_actuator_tensor(self, attr_name: str, value: float) -> None:
        actuators = getattr(self._asset, "actuators", {})
        if not isinstance(actuators, Mapping):
            return
        for actuator_name in self._gripper_actuator_names:
            actuator = actuators.get(actuator_name)
            if actuator is None:
                continue
            attr_value = getattr(actuator, attr_name, None)
            if torch.is_tensor(attr_value):
                attr_value.fill_(float(value))
            elif attr_value is not None:
                try:
                    setattr(actuator, attr_name, float(value))
                except Exception:
                    pass

    def _set_gripper_effort_limits(self) -> None:
        if not self._gripper_joint_ids:
            return
        actuator_effort = max(self._approach_effort_limit, self._force_control_push_effort_limit)
        self._fill_gripper_actuator_tensor("effort_limit_sim", actuator_effort)
        self._fill_gripper_actuator_tensor("effort_limit", actuator_effort)
        effort = torch.full(
            (self.num_envs, len(self._gripper_joint_ids)),
            self._approach_effort_limit,
            device=self.device,
            dtype=torch.float32,
        )
        side_hold = self._gripper_side_effort_hold_mask.clone()
        side_hard = self._gripper_side_effort_hard_mask.clone()
        if torch.any(self._gripper_force_hold_mask & (~self._gripper_force_servo_initialized)):
            side_hold[self._gripper_force_hold_mask & (~self._gripper_force_servo_initialized)] = True
        side_push = self._gripper_force_servo_initialized[:, None] & (~side_hold) & (~side_hard)
        if torch.any(side_hold | side_hard | side_push):
            hard_effort = min(self._hold_effort_limit, self._hard_hold_effort_limit)
            for col in range(len(self._gripper_joint_ids)):
                side = int(self._gripper_joint_sides[col].detach().cpu().item())
                effort[:, col] = torch.where(
                    side_push[:, side],
                    torch.full_like(effort[:, col], self._force_control_push_effort_limit),
                    effort[:, col],
                )
                effort[:, col] = torch.where(
                    side_hold[:, side],
                    torch.full_like(effort[:, col], self._hold_effort_limit),
                    effort[:, col],
                )
                effort[:, col] = torch.where(
                    side_hard[:, side],
                    torch.full_like(effort[:, col], hard_effort),
                    effort[:, col],
                )
        try:
            self._asset.write_joint_effort_limit_to_sim(effort, joint_ids=self._gripper_joint_ids)
        except Exception as exc:
            print(f"[GripperForceLimit] tensor effort write failed: {type(exc).__name__}: {exc}", flush=True)
            try:
                fallback = self._hard_hold_effort_limit if torch.any(side_hard) else (self._hold_effort_limit if torch.any(side_hold) else (self._force_control_push_effort_limit if torch.any(side_push) else self._approach_effort_limit))
                self._asset.write_joint_effort_limit_to_sim(float(fallback), joint_ids=self._gripper_joint_ids)
            except Exception as exc2:
                print(f"[GripperForceLimit] scalar effort write failed: {type(exc2).__name__}: {exc2}", flush=True)

    def _set_gripper_velocity_limits(self) -> None:
        if not self._gripper_joint_ids or self._velocity_limit <= 0.0:
            return
        self._fill_gripper_actuator_tensor("velocity_limit_sim", self._velocity_limit)
        self._fill_gripper_actuator_tensor("velocity_limit", self._velocity_limit)
        velocity = torch.full(
            (self.num_envs, len(self._gripper_joint_ids)),
            self._velocity_limit,
            device=self.device,
            dtype=torch.float32,
        )
        try:
            self._asset.write_joint_velocity_limit_to_sim(velocity, joint_ids=self._gripper_joint_ids)
        except Exception as exc:
            print(f"[GripperVelocityLimit] tensor velocity write failed: {type(exc).__name__}: {exc}", flush=True)
            try:
                self._asset.write_joint_velocity_limit_to_sim(float(self._velocity_limit), joint_ids=self._gripper_joint_ids)
            except Exception as exc2:
                print(f"[GripperVelocityLimit] scalar velocity write failed: {type(exc2).__name__}: {exc2}", flush=True)

    def _set_gripper_position_target(self) -> None:
        try:
            self._asset.set_joint_position_target(self._gripper_target_joint_pos, joint_ids=self._gripper_joint_ids)
        except TypeError:
            try:
                full_target = self._asset.data.joint_pos.clone()
                full_target[:, self._gripper_joint_ids] = self._gripper_target_joint_pos
                self._asset.set_joint_position_target(full_target)
            except Exception as exc:
                self._warn_gripper_once("position", exc)
        except Exception as exc:
            self._warn_gripper_once("position", exc)

    def _set_gripper_velocity_target(self) -> None:
        try:
            self._asset.set_joint_velocity_target(self._zero_gripper_vel_target, joint_ids=self._gripper_joint_ids)
        except TypeError:
            try:
                full_target = torch.zeros_like(self._asset.data.joint_vel)
                full_target[:, self._gripper_joint_ids] = 0.0
                self._asset.set_joint_velocity_target(full_target)
            except Exception as exc:
                self._warn_gripper_once("velocity", exc)
        except Exception as exc:
            self._warn_gripper_once("velocity", exc)

    def _warn_gripper_once(self, target_type: str, exc: Exception) -> None:
        flag = "_warned_gripper_position_target" if target_type == "position" else "_warned_gripper_velocity_target"
        if not getattr(self, flag):
            print(
                f"[{self._action_label}] gripper {target_type} target skipped once: "
                f"{type(exc).__name__}: {exc}"
            )
            setattr(self, flag, True)

    def _ensure_gripper_target_initialized(self) -> None:
        if not self._gripper_target_initialized:
            self._gripper_target_joint_pos[:] = self._asset.data.joint_pos[:, self._gripper_joint_ids]
            self._gripper_side_scalar_target[:] = self._gripper_side_live_scalar()
            self._gripper_scalar_target[:] = torch.mean(self._gripper_side_scalar_target, dim=-1)
            self._gripper_target_initialized = True

def make_crt_gripper_contact_controller(
    *,
    env,
    asset,
    device,
    num_envs: int,
    object_ee_error_fn: Callable[[], tuple[torch.Tensor, torch.Tensor]],
    params: CRTGripperContactControllerParams = DEFAULT_CRT_GRIPPER_PARAMS,
    action_label: str = "CRTGripperContactController",
    arm_route_label: str = "external arm action",
) -> CRTGripperContactController:
    """Create a CRT gripper contact controller from the shared parameter object."""

    return CRTGripperContactController(
        env=env,
        asset=asset,
        device=device,
        num_envs=num_envs,
        object_ee_error_fn=object_ee_error_fn,
        joint_target_multipliers=params.joint_target_multipliers_dict(),
        primary_joint_target_multipliers=params.primary_joint_target_multipliers_dict(),
        open_deg=params.open_deg,
        closed_deg=params.closed_deg,
        auto_close_dist_m=params.auto_close_dist_m,
        auto_close_ori_deg=params.auto_close_ori_deg,
        auto_close_time_s=params.auto_close_time_s,
        free_close_force_n=params.free_close_force_n,
        free_close_step_deg=params.free_close_step_deg,
        contact_close_step_deg=params.contact_close_step_deg,
        approach_effort_limit=params.approach_effort_limit,
        hold_effort_limit=params.hold_effort_limit,
        hard_hold_effort_limit=params.hard_hold_effort_limit,
        hold_normal_force_n=params.hold_normal_force_n,
        hard_normal_force_n=params.hard_normal_force_n,
        force_control_enabled=params.force_control_enabled,
        force_control_target_n=params.force_control_target_n,
        force_control_push_effort_limit=params.force_control_push_effort_limit,
        force_control_kp_deg_per_n=params.force_control_kp_deg_per_n,
        force_control_ki_deg_per_n_s=params.force_control_ki_deg_per_n_s,
        force_control_kd_deg_s_per_n=params.force_control_kd_deg_s_per_n,
        force_control_integral_limit_n_s=params.force_control_integral_limit_n_s,
        force_control_deadband_n=params.force_control_deadband_n,
        force_control_max_step_deg=params.force_control_max_step_deg,
        force_control_hold_error_deg=params.force_control_hold_error_deg,
        force_control_release_step_deg=params.force_control_release_step_deg,
        velocity_limit=params.velocity_limit,
        gripper_actuator_names=params.gripper_actuator_names,
        sync_enabled=params.sync_enabled,
        sync_max_lead_deg=params.sync_max_lead_deg,
        action_label=action_label,
        arm_route_label=arm_route_label,
        left_pad_body_name=params.left_pad_body_name,
        right_pad_body_name=params.right_pad_body_name,
        left_pad_contact_sensor_name=params.left_pad_contact_sensor_name,
        right_pad_contact_sensor_name=params.right_pad_contact_sensor_name,
    )
