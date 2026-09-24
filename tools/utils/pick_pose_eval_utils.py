#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared helpers for pick-pose eval/deploy scripts.

Keep this module free of AppLauncher/IsaacLab app startup side effects. Import it
from scripts only after AppLauncher has been created if the caller is also
importing Isaac Lab runtime modules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class ManualObjectPose:
    """Object pose expressed in the robot base-link frame."""

    pos_base_m: np.ndarray
    quat_base_wxyz: np.ndarray


def parse_float_list(text: str, count: int, name: str) -> list[float]:
    parts = text.replace(",", " ").split()
    if len(parts) != count:
        raise ValueError(f"{name} expects {count} values, got {len(parts)}: {text}")
    return [float(v) for v in parts]


def quat_from_euler_xyz_np(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    q = np.array(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        dtype=np.float32,
    )
    return q / max(float(np.linalg.norm(q)), 1.0e-8)


def normalize_np_quat(q: np.ndarray, name: str) -> np.ndarray:
    q = np.asarray(q, dtype=np.float32)
    norm = float(np.linalg.norm(q))
    if norm < 1.0e-8:
        raise ValueError(f"{name} quaternion norm is too small")
    return q / norm


def manual_object_pose_from_args(args: Any) -> ManualObjectPose | None:
    """Parse manual object pose CLI fields from an argparse namespace-like object."""

    if args.object_pose_base is not None:
        if args.object_pos_base is not None or args.object_quat_base is not None or args.object_rpy_base_deg is not None:
            raise ValueError("--object_pose_base cannot be combined with split object pose arguments.")
        values = parse_float_list(args.object_pose_base, 7, "--object_pose_base")
        pos = np.asarray(values[:3], dtype=np.float32)
        quat = normalize_np_quat(np.asarray(values[3:7], dtype=np.float32), "--object_pose_base")
        return ManualObjectPose(pos_base_m=pos, quat_base_wxyz=quat)

    if args.object_pos_base is None:
        if args.object_quat_base is not None or args.object_rpy_base_deg is not None:
            raise ValueError("Set --object_pos_base when using split object orientation arguments.")
        return None

    pos = np.asarray(parse_float_list(args.object_pos_base, 3, "--object_pos_base"), dtype=np.float32)
    if args.object_quat_base is not None and args.object_rpy_base_deg is not None:
        raise ValueError("Use only one of --object_quat_base or --object_rpy_base_deg.")
    if args.object_quat_base is not None:
        quat = normalize_np_quat(
            np.asarray(parse_float_list(args.object_quat_base, 4, "--object_quat_base"), dtype=np.float32),
            "--object_quat_base",
        )
    elif args.object_rpy_base_deg is not None:
        rpy_deg = parse_float_list(args.object_rpy_base_deg, 3, "--object_rpy_base_deg")
        quat = quat_from_euler_xyz_np(*np.deg2rad(np.asarray(rpy_deg, dtype=np.float32)))
    else:
        quat = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    return ManualObjectPose(pos_base_m=pos, quat_base_wxyz=quat)


def as_first_env_numpy(value: Any, expected_dim: int, name: str) -> np.ndarray:
    """Extract first-env 1D float array from tensor/dict/TensorDict-like observations."""

    if torch.is_tensor(value):
        tensor = value
    elif isinstance(value, dict):
        tensor = None
        for key in ("policy", "obs", "observations"):
            if key in value:
                tensor = value[key]
                break
        if tensor is None:
            tensors = [v for v in value.values() if torch.is_tensor(v)]
            if len(tensors) == 1:
                tensor = tensors[0]
        if tensor is None:
            raise TypeError(f"Cannot extract {name}: dict keys={list(value.keys())}")
        return as_first_env_numpy(tensor, expected_dim, name)
    else:
        tensor = None
        for key in ("policy", "obs", "observations"):
            try:
                candidate = value.get(key, None)
            except Exception:
                candidate = None
            if candidate is not None:
                tensor = candidate
                break
        if tensor is None:
            try:
                values = [v for v in value.values() if torch.is_tensor(v)]
            except Exception:
                values = []
            if len(values) == 1:
                tensor = values[0]
        if tensor is None:
            raise TypeError(f"Cannot extract {name}: type={type(value).__name__}")
        return as_first_env_numpy(tensor, expected_dim, name)

    arr = tensor.detach().float().cpu().numpy()
    if arr.ndim == 2:
        arr = arr[0]
    elif arr.ndim != 1:
        arr = arr.reshape(-1)
    if arr.shape[0] != expected_dim:
        raise ValueError(f"{name} dim mismatch: got {arr.shape[0]}, expected {expected_dim}")
    return arr.astype(np.float32, copy=False)


def parse_quat_wxyz(text: str, device: torch.device | str) -> torch.Tensor:
    values = parse_float_list(text, 4, "quaternion")
    q = torch.tensor(values, device=device, dtype=torch.float32)
    return q / torch.clamp(torch.linalg.norm(q), min=1.0e-8)


def normalize_quat(q: torch.Tensor) -> torch.Tensor:
    return q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1.0e-8)


def quat_mul_wxyz(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)
    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_inv_wxyz(q: torch.Tensor) -> torch.Tensor:
    q = normalize_quat(q)
    return torch.cat((q[..., 0:1], -q[..., 1:4]), dim=-1)


def quat_apply_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    q = normalize_quat(q)
    zeros = torch.zeros(v.shape[:-1] + (1,), dtype=v.dtype, device=v.device)
    vq = torch.cat((zeros, v), dim=-1)
    return quat_mul_wxyz(quat_mul_wxyz(q, vq), quat_inv_wxyz(q))[..., 1:4]


def quat_apply_inverse_wxyz(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    return quat_apply_wxyz(quat_inv_wxyz(q), v)


def pose_in_robot_base(robot, pos_w: torch.Tensor, quat_w: torch.Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Convert a world-frame pose into robot root/base_link coordinates for env 0."""
    root_pos_w = robot.data.root_pos_w[0].detach()
    root_quat_w = normalize_quat(robot.data.root_quat_w[0].detach().view(1, 4)).view(4)
    pos_w = pos_w.detach().view(3)
    quat_w = normalize_quat(quat_w.detach().view(1, 4)).view(4)

    pos_base = quat_apply_inverse_wxyz(root_quat_w.view(1, 4), (pos_w - root_pos_w).view(1, 3)).view(3)
    quat_base = normalize_quat(quat_mul_wxyz(quat_inv_wxyz(root_quat_w).view(1, 4), quat_w.view(1, 4))).view(4)
    return pos_base.cpu().numpy(), quat_base.cpu().numpy()


def _first_env_frame_value(value: torch.Tensor, name: str) -> torch.Tensor:
    if value.ndim == 3:
        return value[0, 0].detach().clone()
    if value.ndim == 2:
        return value[0].detach().clone()
    raise RuntimeError(f"Unknown {name} shape: {value.shape}")


def get_ee_source_pose_w(ee_frame) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the FrameTransformer source pose for env 0, falling back to target pose.

    The source pose is the Robot/link7 pose in the current pick-pose scene and
    matches the xArm SDK get_position frame when no extra TCP offset is set.
    """
    data = ee_frame.data

    pos = None
    for name in ("source_pos_w", "frame_pos_w", "target_pos_w"):
        if hasattr(data, name):
            pos = _first_env_frame_value(getattr(data, name), name)
            break
    if pos is None:
        raise RuntimeError("Cannot read source_pos_w/frame_pos_w/target_pos_w from ee_frame.data")

    quat = None
    for name in ("source_quat_w", "frame_quat_w", "target_quat_w"):
        if hasattr(data, name):
            quat = _first_env_frame_value(getattr(data, name), name)
            break
    if quat is None:
        raise RuntimeError("Cannot read source_quat_w/frame_quat_w/target_quat_w from ee_frame.data")

    return pos, normalize_quat(quat.view(1, 4)).view(4)


def get_ee_target_pose_w(ee_frame) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the FrameTransformer target/TCP pose for env 0, falling back to source pose."""
    data = ee_frame.data

    pos = None
    for name in ("target_pos_w", "frame_pos_w", "source_pos_w"):
        if hasattr(data, name):
            pos = _first_env_frame_value(getattr(data, name), name)
            break
    if pos is None:
        raise RuntimeError("Cannot read target_pos_w/frame_pos_w/source_pos_w from ee_frame.data")

    quat = None
    for name in ("target_quat_w", "frame_quat_w", "source_quat_w"):
        if hasattr(data, name):
            quat = _first_env_frame_value(getattr(data, name), name)
            break
    if quat is None:
        raise RuntimeError("Cannot read target_quat_w/frame_quat_w/source_quat_w from ee_frame.data")

    return pos, normalize_quat(quat.view(1, 4)).view(4)


def get_ee_quat_w(ee_frame) -> torch.Tensor:
    data = ee_frame.data
    quat = None
    for name in ("target_quat_w", "frame_quat_w", "source_quat_w"):
        if hasattr(data, name):
            quat = getattr(data, name)
            break
    if quat is None:
        raise RuntimeError("Cannot read target_quat_w/frame_quat_w/source_quat_w from ee_frame.data")
    if quat.ndim == 3:
        return quat[0, 0].detach().clone()
    if quat.ndim == 2:
        return quat[0].detach().clone()
    raise RuntimeError(f"Unknown ee quat shape: {quat.shape}")


def orientation_error_deg(
    ee_quat_w: torch.Tensor,
    obj_quat_w: torch.Tensor,
    q_offset_wxyz: torch.Tensor,
    grasp_symmetry_quats_wxyz: torch.Tensor,
) -> float:
    ee_q = normalize_quat(ee_quat_w.view(1, 4)).view(4)
    obj_q = normalize_quat(obj_quat_w.view(1, 4)).view(4)
    offset_q = normalize_quat(q_offset_wxyz.view(1, 4)).view(4)
    sym_q = normalize_quat(grasp_symmetry_quats_wxyz.view(-1, 4))

    offsets = quat_mul_wxyz(offset_q.view(1, 4).expand_as(sym_q), sym_q)
    offsets = normalize_quat(offsets)
    desired_q = quat_mul_wxyz(obj_q.view(1, 4).expand_as(offsets), offsets)
    desired_q = normalize_quat(desired_q)

    dots = torch.abs(torch.sum(ee_q.view(1, 4) * desired_q, dim=-1))
    dot = torch.clamp(torch.max(dots), -1.0, 1.0)
    angle_rad = 2.0 * torch.acos(dot)
    return float(torch.rad2deg(angle_rad).item())


def extract_timeout(extras: Any, dones: torch.Tensor) -> torch.Tensor:
    done_mask = dones.bool().view(-1)
    time_outs = extras.get("time_outs", None) if isinstance(extras, dict) else None
    if time_outs is None:
        return torch.zeros_like(done_mask, dtype=torch.bool, device=done_mask.device)
    if not torch.is_tensor(time_outs):
        time_outs = torch.as_tensor(time_outs, dtype=torch.bool, device=done_mask.device)
    else:
        time_outs = time_outs.bool().to(device=done_mask.device)
    time_outs = time_outs.view(-1)
    if time_outs.numel() != done_mask.numel():
        return torch.zeros_like(done_mask, dtype=torch.bool, device=done_mask.device)
    return time_outs


def clear_terminal_cache(base_env) -> None:
    for name in (
        "_terminal_cache_valid",
        "_terminal_done_reason",
        "_terminal_dist_cm",
        "_terminal_ori_err_deg",
        "_terminal_success",
        "_terminal_ee_pos_w",
        "_terminal_ee_quat_w",
        "_terminal_object_pos_w",
        "_terminal_object_quat_w",
        "_terminal_joint_pos",
    ):
        if hasattr(base_env, name):
            try:
                delattr(base_env, name)
            except Exception:
                setattr(base_env, name, None)


def _terminal_cache_is_valid(base_env, env_id: int) -> bool:
    valid = getattr(base_env, "_terminal_cache_valid", None)
    if valid is None:
        return True
    try:
        return bool(valid[env_id].detach().cpu().item())
    except Exception:
        return False


def read_terminal_cache(base_env, env_id: int = 0):
    if not _terminal_cache_is_valid(base_env, env_id):
        return None, None, False
    dist = getattr(base_env, "_terminal_dist_cm", None)
    ori = getattr(base_env, "_terminal_ori_err_deg", None)
    success = getattr(base_env, "_terminal_success", None)
    if dist is None or ori is None or success is None:
        return None, None, False
    try:
        terminal_dist_cm = float(dist[env_id].detach().cpu().item())
        terminal_ori_err_deg = float(ori[env_id].detach().cpu().item())
        terminal_success = bool(success[env_id].detach().cpu().item())
        return terminal_dist_cm, terminal_ori_err_deg, terminal_success
    except Exception:
        return None, None, False


def read_terminal_done_reason(base_env, env_id: int = 0) -> str | None:
    if not _terminal_cache_is_valid(base_env, env_id):
        return None
    reason = getattr(base_env, "_terminal_done_reason", None)
    if reason is None:
        return None
    try:
        if isinstance(reason, (list, tuple)):
            value = reason[env_id]
        else:
            value = reason
        value = str(value)
        return value or None
    except Exception:
        return None


def read_terminal_joint_cache(base_env, env_id: int = 0):
    if not _terminal_cache_is_valid(base_env, env_id):
        return None
    joint_pos = getattr(base_env, "_terminal_joint_pos", None)
    if joint_pos is None:
        return None
    try:
        return joint_pos[env_id].detach().clone()
    except Exception:
        return None


def read_terminal_pose_cache(base_env, env_id: int = 0):
    if not _terminal_cache_is_valid(base_env, env_id):
        return None
    ee_pos = getattr(base_env, "_terminal_ee_pos_w", None)
    ee_quat = getattr(base_env, "_terminal_ee_quat_w", None)
    obj_pos = getattr(base_env, "_terminal_object_pos_w", None)
    obj_quat = getattr(base_env, "_terminal_object_quat_w", None)
    if ee_pos is None or ee_quat is None or obj_pos is None or obj_quat is None:
        return None
    try:
        return (
            ee_pos[env_id].detach().clone(),
            normalize_quat(ee_quat[env_id].detach().view(1, 4)).view(4),
            obj_pos[env_id].detach().clone(),
            normalize_quat(obj_quat[env_id].detach().view(1, 4)).view(4),
        )
    except Exception:
        return None


def get_arm_action_term(base_env):
    manager = base_env.action_manager
    if hasattr(manager, "get_term"):
        try:
            term = manager.get_term("arm_action")
            if hasattr(term, "processed_actions"):
                return term
        except Exception:
            pass

    for attr in ("_terms", "_action_terms"):
        terms = getattr(manager, attr, None)
        if isinstance(terms, dict) and "arm_action" in terms:
            term = terms["arm_action"]
            if hasattr(term, "processed_actions"):
                return term

    terms = getattr(manager, "_terms", None)
    if isinstance(terms, (list, tuple)):
        for term in terms:
            if hasattr(term, "processed_actions"):
                return term
    return None


def get_cmd_delta_deg(
    base_env,
    actions: torch.Tensor,
    action_scale_rad: float,
    action_clip_rad: float,
) -> np.ndarray:
    term = get_arm_action_term(base_env)
    if term is not None and hasattr(term, "processed_actions"):
        try:
            cmd_delta_rad = term.processed_actions[0].detach()
            return torch.rad2deg(cmd_delta_rad).cpu().numpy()
        except Exception:
            pass

    act_np = actions[0].detach().cpu().numpy()
    return np.clip(
        act_np * np.rad2deg(action_scale_rad),
        -np.rad2deg(action_clip_rad),
        np.rad2deg(action_clip_rad),
    )


def refresh_observations(env, base_env, fallback_obs):
    for owner in (env, base_env):
        getter = getattr(owner, "get_observations", None)
        if getter is None:
            continue
        try:
            obs = getter()
            if isinstance(obs, tuple):
                obs = obs[0]
            if isinstance(obs, dict) and "policy" in obs:
                return obs["policy"]
            return obs
        except Exception:
            pass

    obs_manager = getattr(base_env, "observation_manager", None)
    compute = getattr(obs_manager, "compute", None)
    if compute is not None:
        try:
            obs = compute()
            if isinstance(obs, dict) and "policy" in obs:
                return obs["policy"]
            return obs
        except Exception:
            pass

    return fallback_obs


def apply_manual_object_pose(base_env, robot, obj, manual_pose: ManualObjectPose):
    device = base_env.device
    env_ids = torch.arange(base_env.scene.env_origins.shape[0], device=device, dtype=torch.long)
    n = int(env_ids.numel())

    pos_base = torch.as_tensor(manual_pose.pos_base_m, dtype=torch.float32, device=device).view(1, 3).expand(n, 3)
    quat_base = torch.as_tensor(manual_pose.quat_base_wxyz, dtype=torch.float32, device=device).view(1, 4).expand(n, 4)
    quat_base = normalize_quat(quat_base)

    root_pos_w = robot.data.root_pos_w[env_ids]
    root_quat_w = normalize_quat(robot.data.root_quat_w[env_ids])

    object_pos_w = root_pos_w + quat_apply_wxyz(root_quat_w, pos_base)
    object_quat_w = normalize_quat(quat_mul_wxyz(root_quat_w, quat_base))

    root_state = obj.data.default_root_state[env_ids].clone()
    root_state[:, :3] = object_pos_w
    root_state[:, 3:7] = object_quat_w
    root_state[:, 7:] = 0.0
    obj.write_root_state_to_sim(root_state, env_ids=env_ids)

    try:
        base_env.scene.write_data_to_sim()
    except Exception:
        pass
    try:
        base_env.scene.update(0.0)
    except Exception:
        pass


def object_pose_in_base(robot, obj) -> tuple[np.ndarray, np.ndarray]:
    obj_pos_w = obj.data.root_pos_w[0].detach()
    obj_quat_w = obj.data.root_quat_w[0].detach()
    return pose_in_robot_base(robot, obj_pos_w, obj_quat_w)


def reset_eval_env(env, base_env, robot, obj, manual_pose: ManualObjectPose | None):
    clear_terminal_cache(base_env)
    obs, _ = env.reset()
    if manual_pose is not None:
        apply_manual_object_pose(base_env, robot, obj, manual_pose)
        obs = refresh_observations(env, base_env, obs)
    return obs
