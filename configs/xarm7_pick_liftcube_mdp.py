"""自定义 MDP 函数 - Lift-Cube 风格奖励设计

参考 Isaac Lab 标准 Lift-Cube 任务的奖励设计：
- 加权求和：total_reward = Σ(weight × reward_term)
- 分离奖励项：reaching + lifting + goal_tracking
- 双层引导：粗粒度 + 细粒度目标跟踪
- 隐式学习：不显式判断夹爪状态，让策略自己学习

奖励项设计：
1. reaching_object:     接近物体奖励，weight=1.0
2. lifting_object:      举起物体奖励（二值），weight=15.0
3. object_goal_tracking: 目标位置跟踪（粗粒度），weight=16.0
4. object_goal_tracking_fine: 目标位置跟踪（细粒度），weight=5.0
"""

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


_GRASP_SYMMETRY_QUATS_WXYZ = (
    (1.0, 0.0, 0.0, 0.0),
    (0.0, 0.0, 0.0, 1.0),
)


def _normalize_quat(q: torch.Tensor) -> torch.Tensor:
    return q / torch.clamp(torch.linalg.norm(q, dim=-1, keepdim=True), min=1.0e-8)


def _symmetric_orientation_angle(
    ee_quat: torch.Tensor,
    q_obj: torch.Tensor,
    q_offset: tuple,
    grasp_symmetry_quats: tuple = _GRASP_SYMMETRY_QUATS_WXYZ,
) -> torch.Tensor:
    """夹爪局部 Z 轴 180° 对称等价时，返回最小姿态误差(rad)。"""
    from isaaclab.utils.math import quat_mul

    ee_quat = _normalize_quat(ee_quat)
    q_obj = _normalize_quat(q_obj)
    offset = torch.tensor(q_offset, dtype=torch.float32, device=ee_quat.device)
    offset = _normalize_quat(offset.view(1, 4)).view(4)
    sym = torch.tensor(grasp_symmetry_quats, dtype=torch.float32, device=ee_quat.device)
    sym = _normalize_quat(sym)

    offsets = quat_mul(offset.unsqueeze(0).expand_as(sym), sym)
    offsets = _normalize_quat(offsets)

    num_envs = q_obj.shape[0]
    num_sym = offsets.shape[0]
    obj = q_obj[:, None, :].expand(num_envs, num_sym, 4).reshape(-1, 4)
    off = offsets[None, :, :].expand(num_envs, num_sym, 4).reshape(-1, 4)
    targets = quat_mul(obj, off).view(num_envs, num_sym, 4)
    targets = _normalize_quat(targets)

    dots = torch.abs((ee_quat[:, None, :] * targets).sum(dim=-1)).clamp(0.0, 1.0)
    best_dot = torch.max(dots, dim=1).values
    return 2.0 * torch.acos(best_dot)


##
# 抓取目标点
##

# 工件局部系下的抓取目标偏移，单位 m。(0,0,0) = 工件根原点，即历史行为。
#
# 为什么是**局部**系而不是世界系 z：工件每次 reset 会绕 z 转 ±90°（见
# XArm7PickPointCloudEventCfg.reset_table_and_object 的 yaw_range），偏移必须跟着
# 工件转，否则同一个"上方 3cm"在不同 yaw 下会落到工件的不同侧面。
#
# 为什么 z 取**负**值才是世界系向上：场景里 object.init_state.rot = [0,1,0,0]，
# 即绕 X 轴 180°，把工件局部 +z 翻成了世界 -z。所以"世界向上 3cm" = 局部 -0.03。
# 这一条已用 scripts/train/mark_grasp_target.py 在视口里核对过（绿球在红球上方）。
_GRASP_TARGET_OFFSET_LOCAL_M = (0.0, 0.0, 0.0)


def grasp_target_pos_w(
    object: RigidObject,
    offset_local: tuple = (0.0, 0.0, 0.0),
) -> torch.Tensor:
    """抓取目标点的世界坐标 = 工件根原点 + 局部偏移旋到世界系。

    ``offset_local`` 为 (0,0,0) 时直接返回 ``root_pos_w``，**不做任何张量运算**，
    保证不加偏移的旧配置在数值上逐位等同于改动前。

    这是全项目唯一一处计算抓取目标点的地方 —— 奖励、success 判据、终止条件、
    可视化标记都必须走这里。目标点定义散成多份是这类改动最容易埋的坑：改了奖励
    忘了改 success，训练曲线一路涨而 success 永远是 0。
    """
    if offset_local == (0.0, 0.0, 0.0):
        return object.data.root_pos_w

    from isaaclab.utils.math import quat_apply

    offset = torch.tensor(
        offset_local, dtype=torch.float32, device=object.data.root_pos_w.device
    ).view(1, 3).expand_as(object.data.root_pos_w)
    return object.data.root_pos_w + quat_apply(object.data.root_quat_w, offset)


##
# 观测函数
##


def ee_position_in_robot_root_frame(env: "ManagerBasedRLEnv", ee_frame_cfg: SceneEntityCfg) -> torch.Tensor:
    """末端执行器位置（相对于机械臂基座）"""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene["robot"]
    return ee_frame.data.target_pos_w[:, 0, :] - robot.data.root_pos_w


def ee_orientation(env: "ManagerBasedRLEnv", ee_frame_cfg: SceneEntityCfg) -> torch.Tensor:
    """末端执行器姿态（四元数）"""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return ee_frame.data.target_quat_w[:, 0, :]


def object_position_relative_to_ee(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """工件相对于末端执行器的位置"""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    return object.data.root_pos_w - ee_frame.data.target_pos_w[:, 0, :]


def object_lifted_binary(
    env: "ManagerBasedRLEnv",
    minimal_height: float,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """工件是否被举起（二值）"""
    object: RigidObject = env.scene[object_cfg.name]
    obj_height = object.data.root_pos_w[:, 2]
    return (obj_height > minimal_height).float().unsqueeze(-1)


##
# 奖励函数（Lift-Cube 风格）
##


def object_ee_distance(
    env: "ManagerBasedRLEnv",
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    target_offset_local: tuple = _GRASP_TARGET_OFFSET_LOCAL_M,
) -> torch.Tensor:
    """接近目标点奖励 - 使用指数核函数，梯度更陡，迫使策略真正靠近

    ``target_offset_local``：工件局部系下的目标偏移，见 :func:`grasp_target_pos_w`。
    默认 (0,0,0) 即瞄工件根原点。
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    obj_pos = grasp_target_pos_w(object, target_offset_local)
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    distance = torch.norm(obj_pos - ee_pos, dim=-1)

    # 指数核：exp(-dist / std)，距离越近奖励越高，远处趋近 0。
    reward = torch.exp(-distance / std)

    return reward


def ee_orientation_alignment(
    env: "ManagerBasedRLEnv",
    std: float = 0.5,
    q_offset: tuple = (0.0, 0.0, 1.0, 0.0),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    grasp_symmetry_quats: tuple = _GRASP_SYMMETRY_QUATS_WXYZ,
) -> torch.Tensor:
    """末端姿态奖励：目标姿态允许夹爪绕局部 Z 轴 180° 对称。"""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    ee_quat = ee_frame.data.target_quat_w[:, 0, :]       # (N, 4) wxyz
    q_obj = object.data.root_quat_w                       # (N, 4) wxyz
    angle_diff = _symmetric_orientation_angle(
        ee_quat,
        q_obj,
        q_offset=q_offset,
        grasp_symmetry_quats=grasp_symmetry_quats,
    )

    return torch.exp(-angle_diff / std)


def object_ee_distance_linear(
    env: "ManagerBasedRLEnv",
    d_max: float = 0.5,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """线性距离奖励：clamp(1 - d/d_max, 0, 1)，全程常数梯度，避免 Gaussian 梯度死区。"""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    d = torch.norm(object.data.root_pos_w - ee_frame.data.target_pos_w[:, 0, :], dim=-1)
    return (1.0 - d / d_max).clamp(min=0.0)


def ee_orientation_alignment_linear(
    env: "ManagerBasedRLEnv",
    angle_max: float = 3.14159265,
    q_offset: tuple = (0.0, 0.0, 0.0, 1.0),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    grasp_symmetry_quats: tuple = _GRASP_SYMMETRY_QUATS_WXYZ,
) -> torch.Tensor:
    """线性姿态奖励：对 180° 对称候选姿态取最小角度。"""
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    ee_quat = ee_frame.data.target_quat_w[:, 0, :]
    q_obj = object.data.root_quat_w
    angle_diff = _symmetric_orientation_angle(
        ee_quat,
        q_obj,
        q_offset=q_offset,
        grasp_symmetry_quats=grasp_symmetry_quats,
    )
    return (1.0 - angle_diff / angle_max).clamp(min=0.0)


def object_is_lifted(
    env: "ManagerBasedRLEnv",
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    max_ee_distance: float = 0.04,
    max_angle_deg: float = 20.0,
) -> torch.Tensor:
    """举起物体奖励 - 纯高度二值奖励"""
    object: RigidObject = env.scene[object_cfg.name]
    return (object.data.root_pos_w[:, 2] > minimal_height).float()


def object_goal_distance(
    env: "ManagerBasedRLEnv",
    std: float,
    minimal_height: float,
    target_height: float,
    xy_threshold: float = None,
    ee_dist_threshold: float = 0.04,
    ee_angle_threshold_deg: float = 10.0,
    use_xyz_dist: bool = False,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """目标位置跟踪奖励 - 需满足：z>minimal_height + EE距工件<4cm + EE姿态误差<10°

    use_xyz_dist=True 时：distance_reward 使用工件到目标 xyz 三维距离（hard 模式）
    use_xyz_dist=False 时：distance_reward 只使用 z 方向距离（soft 模式）
    """
    from isaaclab.utils.math import quat_mul

    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    obj_pos = object.data.root_pos_w
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]

    # 距离奖励
    if use_xyz_dist:
        # hard：工件到目标 xyz 三维距离，目标 = 默认 xy + 目标高度
        target_xy = object.data.default_root_state[:, :2] + env.scene.env_origins[:, :2]
        target_pos = torch.cat([target_xy, torch.full((obj_pos.shape[0], 1), target_height, device=obj_pos.device)], dim=-1)
        dist = torch.norm(obj_pos - target_pos, dim=-1)
    else:
        # soft：只看 z 方向距离
        dist = torch.abs(obj_pos[:, 2] - target_height)

    distance_reward = 1.0 - torch.tanh(dist / std)

    # 条件1：工件高度
    is_lifted = obj_pos[:, 2] > minimal_height

    # 条件2：EE 位置距工件 < ee_dist_threshold
    ee_to_obj = torch.norm(obj_pos - ee_pos, dim=-1)
    ee_close = ee_to_obj < ee_dist_threshold

    # 条件3：EE 姿态误差 < ee_angle_threshold_deg
    ee_quat = ee_frame.data.target_quat_w[:, 0, :]
    q_obj = object.data.root_quat_w
    offset = torch.tensor((0.0, 0.0, 1.0, 0.0), dtype=torch.float32, device=ee_quat.device)
    offset = offset.unsqueeze(0).expand_as(q_obj)
    target_quat = quat_mul(q_obj, offset)
    dot = torch.abs((ee_quat * target_quat).sum(dim=-1)).clamp(0.0, 1.0)
    angle_diff_deg = 2.0 * torch.acos(dot) * (180.0 / 3.14159265)
    ee_angle_ok = angle_diff_deg < ee_angle_threshold_deg

    condition = is_lifted & ee_close & ee_angle_ok

    if xy_threshold is not None:
        target_default_xy = object.data.default_root_state[:, :2] + env.scene.env_origins[:, :2]
        xy_dist = torch.norm(obj_pos[:, :2] - target_default_xy, dim=-1)
        condition = condition & (xy_dist < xy_threshold)

    return condition.float() * distance_reward


def gripper_close_near_object(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=["drive_joint"]),
    dist_threshold: float = 0.10,
) -> torch.Tensor:
    """EE 靠近工件时奖励夹爪闭合 - 打破靠近不夹取的局部最优"""
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    robot: Articulation = env.scene[robot_cfg.name]

    dist = torch.norm(object.data.root_pos_w - ee_frame.data.target_pos_w[:, 0, :], dim=-1)
    near = (dist < dist_threshold).float()

    # drive_joint: 0=全开，0.85=全闭，归一化到 [0, 1]
    gripper_pos = robot.data.joint_pos[:, robot_cfg.joint_ids[0]]
    gripper_closed_ratio = (gripper_pos / 0.85).clamp(0.0, 1.0)

    return near * gripper_closed_ratio





def object_is_lifted_simple(
    env: "ManagerBasedRLEnv",
    minimal_height: float,
    xy_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """lift_success_bonus — z>minimal_height 且 xy 距目标 < xy_threshold"""
    object: RigidObject = env.scene[object_cfg.name]
    obj_pos = object.data.root_pos_w
    target_xy = object.data.default_root_state[:, :2] + env.scene.env_origins[:, :2]
    xy_dist = torch.norm(obj_pos[:, :2] - target_xy, dim=-1)
    return ((obj_pos[:, 2] > minimal_height) & (xy_dist < xy_threshold)).float()


def object_reached_lift_height(
    env: "ManagerBasedRLEnv",
    minimal_height: float,
    xy_threshold: float,
    object_cfg: SceneEntityCfg,
) -> torch.Tensor:
    """成功终止条件 — z>minimal_height 且 xy 距目标 < xy_threshold"""
    object: RigidObject = env.scene[object_cfg.name]
    obj_pos = object.data.root_pos_w
    target_xy = object.data.default_root_state[:, :2] + env.scene.env_origins[:, :2]
    xy_dist = torch.norm(obj_pos[:, :2] - target_xy, dim=-1)
    return (obj_pos[:, 2] > minimal_height) & (xy_dist < xy_threshold)


def object_reached_lift_xyz(
    env: "ManagerBasedRLEnv",
    target_height: float,
    xyz_threshold: float,
    ee_dist_threshold: float = 0.04,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """成功终止条件 — 工件 xyz 三维距离目标 < xyz_threshold 且 EE 贴近工件

    返回 bool 张量，用于 DoneTerm。
    目标位置 = 工件初始 xy + target_height。
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    obj_pos = object.data.root_pos_w
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]

    target_xy = object.data.default_root_state[:, :2] + env.scene.env_origins[:, :2]
    target_pos = torch.cat(
        [target_xy, torch.full((obj_pos.shape[0], 1), target_height, device=obj_pos.device)],
        dim=-1,
    )

    xyz_dist = torch.norm(obj_pos - target_pos, dim=-1)
    ee_to_obj = torch.norm(obj_pos - ee_pos, dim=-1)

    return (xyz_dist < xyz_threshold) & (ee_to_obj < ee_dist_threshold)


def object_reached_lift_xyz_reward(
    env: "ManagerBasedRLEnv",
    target_height: float,
    xyz_threshold: float,
    ee_dist_threshold: float = 0.04,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """成功奖励 — 与 object_reached_lift_xyz 判据相同，返回 float 用于 RewTerm。"""
    return object_reached_lift_xyz(
        env, target_height, xyz_threshold, ee_dist_threshold, object_cfg, ee_frame_cfg
    ).float()


##
# 翻面任务函数
##


def object_is_lifted(
    env: "ManagerBasedRLEnv",
    minimal_height: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """工件是否被举起（返回 float (N,)，用于 RewTerm）。"""
    object: RigidObject = env.scene[object_cfg.name]
    return (object.data.root_pos_w[:, 2] > minimal_height).float()


def object_quat_w(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """工件当前四元数（世界系，wxyz），用于观测。"""
    object: RigidObject = env.scene[object_cfg.name]
    return object.data.root_quat_w  # (N, 4)


def _get_flip_target_quat(
    env: "ManagerBasedRLEnv",
    object_cfg: SceneEntityCfg,
    flip_axis: str,
    device: torch.device,
) -> torch.Tensor:
    """计算翻面目标四元数：初始姿态绕指定轴旋转 180°。

    翻转四元数（wxyz）：
      绕 Y 轴 180° → (0, 0, 1, 0)
      绕 X 轴 180° → (0, 1, 0, 0)
    """
    from isaaclab.utils.math import quat_mul

    object: RigidObject = env.scene[object_cfg.name]
    q_init = object.data.default_root_state[:, 3:7].to(device)  # (N, 4) wxyz

    if flip_axis == "y":
        q_flip = torch.tensor([[0.0, 0.0, 1.0, 0.0]], device=device).expand(q_init.shape[0], -1)
    else:  # x
        q_flip = torch.tensor([[0.0, 1.0, 0.0, 0.0]], device=device).expand(q_init.shape[0], -1)

    return quat_mul(q_init, q_flip)  # (N, 4)


def _quat_angle_error(q_current: torch.Tensor, q_target: torch.Tensor) -> torch.Tensor:
    """两个四元数之间的旋转角误差（弧度），范围 [0, π]。"""
    dot = (q_current * q_target).sum(dim=-1).abs().clamp(0.0, 1.0)
    return 2.0 * torch.acos(dot)


def flip_orientation_reward(
    env: "ManagerBasedRLEnv",
    std: float = 0.5,
    minimal_height: float = 0.73,
    flip_axis: str = "y",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """翻转姿态奖励 — 工件被举起时，奖励姿态朝目标翻转方向靠近。

    使用线性进度奖励：(π - angle_error) / π，范围 [0, 1]。
    初始角差 π → 奖励 0；翻转完成 → 奖励 1。
    全程有梯度，避免 Gaussian 在大角度时梯度消失。
    """
    import math as _math
    object: RigidObject = env.scene[object_cfg.name]
    obj_pos = object.data.root_pos_w
    obj_quat = object.data.root_quat_w

    q_target = _get_flip_target_quat(env, object_cfg, flip_axis, obj_quat.device)
    angle_error = _quat_angle_error(obj_quat, q_target)

    is_lifted = (obj_pos[:, 2] > minimal_height).float()
    progress = (_math.pi - angle_error) / _math.pi   # [0, 1]，起点有梯度
    return is_lifted * progress.clamp(0.0, 1.0)


def flip_place_reward(
    env: "ManagerBasedRLEnv",
    table_height: float = 0.72,
    z_std: float = 0.03,
    ori_std: float = 0.3,
    xy_threshold: float = 0.15,
    flip_axis: str = "y",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """放置奖励 — 工件回到桌面附近（z ≈ table_height）且姿态已翻转时给分。

    奖励 = exp(-(z_err/z_std)²) * exp(-(angle_err/ori_std)²) * xy_ok
    """
    object: RigidObject = env.scene[object_cfg.name]
    obj_pos = object.data.root_pos_w
    obj_quat = object.data.root_quat_w

    q_target = _get_flip_target_quat(env, object_cfg, flip_axis, obj_quat.device)
    angle_error = _quat_angle_error(obj_quat, q_target)

    z_err = obj_pos[:, 2] - table_height
    target_xy = object.data.default_root_state[:, :2] + env.scene.env_origins[:, :2]
    xy_dist = torch.norm(obj_pos[:, :2] - target_xy, dim=-1)

    ori_reward = torch.exp(-(angle_error / ori_std) ** 2)
    z_reward = torch.exp(-(z_err / z_std) ** 2)
    xy_ok = (xy_dist < xy_threshold).float()

    return ori_reward * z_reward * xy_ok


def object_flip_success(
    env: "ManagerBasedRLEnv",
    table_height: float = 0.72,
    z_tolerance: float = 0.05,
    angle_threshold: float = 0.52,  # ≈30°
    xy_threshold: float = 0.15,
    vel_threshold: float = 0.05,
    flip_axis: str = "y",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """翻面成功终止条件（返回 bool）：
    工件回到桌面 + 姿态翻转误差 < 30° + xy 偏移 < 15cm + 速度稳定。
    """
    object: RigidObject = env.scene[object_cfg.name]
    obj_pos = object.data.root_pos_w
    obj_quat = object.data.root_quat_w
    obj_vel = object.data.root_lin_vel_w

    q_target = _get_flip_target_quat(env, object_cfg, flip_axis, obj_quat.device)
    angle_error = _quat_angle_error(obj_quat, q_target)

    target_xy = object.data.default_root_state[:, :2] + env.scene.env_origins[:, :2]
    xy_dist = torch.norm(obj_pos[:, :2] - target_xy, dim=-1)
    vel_mag = torch.norm(obj_vel, dim=-1)

    on_table = (obj_pos[:, 2] > table_height - z_tolerance) & (obj_pos[:, 2] < table_height + z_tolerance)
    flipped = angle_error < angle_threshold
    xy_ok = xy_dist < xy_threshold
    stable = vel_mag < vel_threshold

    return on_table & flipped & xy_ok & stable


def object_flip_success_reward(
    env: "ManagerBasedRLEnv",
    table_height: float = 0.72,
    z_tolerance: float = 0.05,
    angle_threshold: float = 0.52,
    xy_threshold: float = 0.15,
    vel_threshold: float = 0.05,
    flip_axis: str = "y",
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """翻面成功奖励（返回 float），与 object_flip_success 判据相同，用于 RewTerm。"""
    return object_flip_success(
        env, table_height, z_tolerance, angle_threshold, xy_threshold, vel_threshold, flip_axis, object_cfg
    ).float()


def ee_reached_object(
    env: "ManagerBasedRLEnv",
    threshold: float,
    angle_threshold_deg: float = 180.0,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    q_offset: tuple = (0.0, 0.0, 0.0, 1.0),
    grasp_symmetry_quats: tuple = _GRASP_SYMMETRY_QUATS_WXYZ,
    target_offset_local: tuple = _GRASP_TARGET_OFFSET_LOCAL_M,
) -> torch.Tensor:
    """EE 到达目标点附近的终止条件；姿态允许 180° 对称夹持。

    ``target_offset_local`` 必须与奖励里用的**同一个值**，否则奖励在爬一个点、
    success 在判另一个点。
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    obj_pos = grasp_target_pos_w(object, target_offset_local)
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]
    dist = torch.norm(obj_pos - ee_pos, dim=-1)
    pos_ok = dist < threshold

    if angle_threshold_deg >= 180.0:
        return pos_ok

    ee_quat = ee_frame.data.target_quat_w[:, 0, :]
    q_obj = object.data.root_quat_w
    angle_deg = _symmetric_orientation_angle(
        ee_quat,
        q_obj,
        q_offset=q_offset,
        grasp_symmetry_quats=grasp_symmetry_quats,
    ) * (180.0 / 3.14159265)
    angle_ok = angle_deg < angle_threshold_deg

    return pos_ok & angle_ok


def ee_orientation_error_quat(
    env: "ManagerBasedRLEnv",
    q_offset: tuple = (0.0, 0.0, 1.0, 0.0),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """EE 与目标姿态的相对旋转四元数（4维，wxyz），用于观测。

    target = q_obj * q_offset
    error  = q_ee^{-1} * target
    返回 error 四元数，当 EE 完全对齐时趋近于 (1,0,0,0)。
    """
    from isaaclab.utils.math import quat_mul, quat_inv
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]

    ee_quat  = ee_frame.data.target_quat_w[:, 0, :]       # (N,4) wxyz
    q_obj    = object.data.root_quat_w                     # (N,4) wxyz

    offset   = torch.tensor(q_offset, dtype=torch.float32, device=ee_quat.device)
    offset   = offset.unsqueeze(0).expand_as(q_obj)
    q_target = quat_mul(q_obj, offset)                     # (N,4)

    q_err = quat_mul(quat_inv(ee_quat), q_target)          # (N,4)
    # 保证 w >= 0（双覆盖消歧）
    sign  = torch.sign(q_err[:, 0:1]).clamp(min=0.0) * 2 - 1
    sign[sign == 0] = 1.0
    return q_err * sign                                    # (N,4)


def object_holding_reward(
    env: "ManagerBasedRLEnv",
    minimal_height: float,
    ee_dist_threshold: float = 0.04,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """持握奖励 — 工件在空中（z > minimal_height）且 EE 贴近工件时每步给分。

    用于塑形"抓住不放"的行为，防止策略通过弹射工件来刷瞬时距离奖励。
    """
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]

    obj_pos = object.data.root_pos_w
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]

    is_lifted = obj_pos[:, 2] > minimal_height
    ee_close = torch.norm(obj_pos - ee_pos, dim=-1) < ee_dist_threshold

    return (is_lifted & ee_close).float()


##
# 举升任务函数（基于 buffer 初始 EE 姿态）
##


def ee_to_lift_target(
    env: "ManagerBasedRLEnv",
    target_z: float,
    std: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """EE 到举升目标位置的距离奖励。

    目标位置 = (buffer 存储的 EE 初始 XY, target_z)，
    引导 EE 在保持 XY 不变的情况下垂直抬升到 target_z。
    需配合状态缓冲环境使用（env._init_ee_pos_world 由其 _reset_idx 写入）。
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos = ee_frame.data.target_pos_w[:, 0, :]              # (N, 3)

    init_xy = env._init_ee_pos_world[:, :2]                   # (N, 2)
    target_pos = torch.cat(
        [init_xy, torch.full((ee_pos.shape[0], 1), target_z, device=ee_pos.device)],
        dim=-1,
    )                                                          # (N, 3)

    dist = torch.norm(ee_pos - target_pos, dim=-1)
    return torch.exp(-dist / std)


def ee_orientation_fixed_target(
    env: "ManagerBasedRLEnv",
    std: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """EE 与 buffer 初始四元数对齐的姿态奖励。

    目标四元数 = env._init_ee_quat（由状态缓冲环境的 _reset_idx 写入），
    即预训练对齐策略输出后的 EE 朝向，举升过程中保持不变。
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_quat = ee_frame.data.target_quat_w[:, 0, :]            # (N, 4) wxyz

    target_quat = env._init_ee_quat                           # (N, 4) wxyz
    dot = torch.abs((ee_quat * target_quat).sum(dim=-1)).clamp(0.0, 1.0 - 1e-7)
    angle = 2.0 * torch.acos(dot)
    return torch.exp(-angle / std)


def lift_reached(
    env: "ManagerBasedRLEnv",
    target_z: float,
    dist_threshold: float,
    angle_threshold_deg: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """举升到达终止条件（返回 bool）。

    同时满足：
      - EE 到目标位置 (init_xy, target_z) 的距离 < dist_threshold
      - EE 四元数与初始姿态的角度误差 < angle_threshold_deg
    """
    import math as _math
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    ee_pos  = ee_frame.data.target_pos_w[:, 0, :]             # (N, 3)
    ee_quat = ee_frame.data.target_quat_w[:, 0, :]            # (N, 4) wxyz

    init_xy = env._init_ee_pos_world[:, :2]
    target_pos = torch.cat(
        [init_xy, torch.full((ee_pos.shape[0], 1), target_z, device=ee_pos.device)],
        dim=-1,
    )
    pos_ok = torch.norm(ee_pos - target_pos, dim=-1) < dist_threshold

    target_quat = env._init_ee_quat
    dot = torch.abs((ee_quat * target_quat).sum(dim=-1)).clamp(0.0, 1.0 - 1e-7)
    angle_deg = 2.0 * torch.acos(dot) * (180.0 / _math.pi)
    angle_ok = angle_deg < angle_threshold_deg

    return pos_ok & angle_ok


def lift_reached_reward(
    env: "ManagerBasedRLEnv",
    target_z: float,
    dist_threshold: float,
    angle_threshold_deg: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """举升到达奖励（float 版，供 RewTerm 使用）。"""
    return lift_reached(
        env, target_z, dist_threshold, angle_threshold_deg, ee_frame_cfg
    ).float()


def contact_force_penalty(
    env: "ManagerBasedRLEnv",
    sensor_name: str,
    threshold: float = 0.5,
) -> torch.Tensor:
    """接触力惩罚 — sensor_name 对应的 ContactSensor 受到超过阈值的合力时返回 1.0。

    将此函数挂载到工件或桌面的 ContactSensor，当机械臂与其发生碰撞时给予负奖励。
    返回值：(num_envs,)，有碰撞为 1.0，否则为 0.0。
    """
    from isaaclab.sensors import ContactSensor
    sensor: ContactSensor = env.scene.sensors[sensor_name]
    # net_forces_w_history: (num_envs, history_len, num_bodies, 3)
    forces = sensor.data.net_forces_w_history
    # 取历史最大合力幅值，对所有 body 求和
    force_mag = forces.norm(dim=-1).max(dim=1).values.sum(dim=-1)  # (num_envs,)
    return (force_mag > threshold).float()
