"""Point Bridge 风格点表征 —— 仿真与真机共用的点云管线。

规格来源：``point_bridge_rl_observation_spec.md``。

本模块是规格 §1.1 要求的**唯一共用实现**。仿真训练与真机部署调用同一份
:func:`mask_depth_to_pointcloud`，两侧的差异只在它上游：

    掩码来源    仿真 = instance_id_segmentation_fast   真机 = YOLO-seg
    深度来源    仿真 = distance_to_image_plane          真机 = RGB-D 深度图

从"掩码内采点"往后（确定性 FPS → 反投影 → 目标坐标系 → 类型通道 → 噪声）全部
在本函数内，不允许在任何一侧另写一份。

坐标约定（写错这里就是规格 §9 里"点云整体镜像或旋转 90°"那一行）：

    深度      沿相机 z 轴到成像平面的距离（``distance_to_image_plane``），
              **不是**到光心的欧氏距离（``distance_to_camera``）。
    相机系    OpenCV / ROS 光学系：+X 右，+Y 下，+Z 前。
              这与 ``unproject_depth`` 的输出一致，因此在 Isaac Lab 侧取相机
              姿态必须用 ``camera.data.quat_w_ros``，不能用 ``quat_w_world``
              （后者是 USD/OpenGL 的 +Y 上 / -Z 前）。
    cam_pose  传入的 ``cam_pos`` / ``cam_quat`` 是**相机在目标坐标系（机器人
              基座系）中的位姿**，不是世界系。真机标定文件给的就是
              camera→base，直接可用；仿真侧需先把世界系位姿换算过来，见
              :func:`pose_in_frame`。

单位一律为米。
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import (
    matrix_from_quat,
    quat_apply,
    quat_apply_inverse,
    quat_mul,
)


# ──────────────────────────────────────────────────────────────────────────────
# 常量（规格 §3.2）
# ──────────────────────────────────────────────────────────────────────────────

M_OBJ = 64          # 每物体点数。参考实现的默认是 128（num_points_per_obj），
                    # 消融显示 10 / 64 / 128 表现相近，这里取 64 省算力。
N_ROBOT = 6         # 夹爪关键点数，布局见 GRIPPER_KEYPOINTS_LOCAL_M。
POINT_DIM = 3       # xyz。参考实现的 PointNetEncoderXYZ 是 in_channels=3。
NUM_POINTS = M_OBJ + N_ROBOT   # 70

# 观测扁平布局：[物体点 M_OBJ*3 | 夹爪点 N_ROBOT*3 | 低维向量...]
# 网络侧按这两个长度切开、各自过一次 encoder（见下方"双 token"说明）。
OBJ_FEATURE_LEN = M_OBJ * POINT_DIM        # 192
ROBOT_FEATURE_LEN = N_ROBOT * POINT_DIM    # 18
POINT_FEATURE_LEN = OBJ_FEATURE_LEN + ROBOT_FEATURE_LEN   # 210

# ── 腕部相机点云（stage 10）────────────────────────────────────────────────────
# 腕部这一路**不做分割**：直接在整帧有效深度里确定性采 M_WRIST 个点。理由有三：
#
#   1. 无掩码 = 无 sim2real 掩码差。全局那一路仿真用 instance_id_segmentation、
#      真机用 YOLO-seg，两者的边界抖动与漏检分布不同，是 stage 3/4 的主要 gap；
#      腕部只要深度，真机一个 RGB-D 相机直接给出同样的东西。
#   2. 空掩码问题自动消失。腕部相机贴在 link7 前 11cm 往夹爪方向看，画面里永远
#      有桌面或臂体，counts=0 几乎不可能发生 —— 不必依赖零阶保持兜底。
#   3. 开销与掩码大小无关。FPS_CANDIDATE_CAP 的等间距预筛把候选压到 512，之后
#      的 FPS 成本固定，所以"整帧采样"并不比"掩码内采样"贵。
#
# 代价是腕部点里混着桌面与臂体，物体点不再被单独标出 —— 由网络自己从近场几何里
# 学。这是刻意的取舍：腕部分支的价值在"末端附近的稠密局部几何"，而不在语义。
M_WRIST = 64
WRIST_FEATURE_LEN = M_WRIST * POINT_DIM    # 192

# ── 为什么没有类型通道 ────────────────────────────────────────────────────────
# 参考实现 (pointbridge-main/point_bridge/agent/pb.py) 用**同一个 encoder 跑两次
# 独立前向**，物体点和夹爪点各产出一个 token，身份由 token 的结构位置携带：
#
#     past_robot_tracks  = self.encoder(past_robot_tracks)    # 一个 token
#     past_object_tracks = self.encoder(past_object_tracks)   # 另一个 token
#
# 因此两组点从不进入同一次 max pool，不存在"池化后分不清谁是谁"的问题，也就不需要
# 逐点类型标签。这比类型通道更硬：类型通道要靠网络自己学会读第 4 维，而分开编码是
# 结构性保证。
#
# 我们没有参考实现的 GPT trunk，所以两个 embedding 用 concat 而非序列拼接，效果
# 等价（单步策略下 transformer 退化为 MLP）。
#
# 参考实现的 num_robot_points = 8 + 1 = 9，布局见 robot_utils/common/
# franka_gripper_points.py（Franka Hand 两指 + 两条横线，全部在 x=0 平面内）。
# 本项目的夹爪不同，沿用下方自定的 6 点布局。

# 规格 §4.4：0.01 是原工作的值，与参考实现的 noise_std: 0.01 一致。
# TODO(real2real): 真机 D435 在 ~1.2 m 处的实测深度误差若为 2-3 cm，必须按实测
# 上调，不要照抄这个数。上调后仿真需重训。
NOISE_STD_M = 0.01

# ── 点云归一化（对齐参考实现的 preprocess["past_tracks"]）────────────────────
# 参考实现把点云做 min-max 归一化到 [0, 1]，物体点与夹爪点**共用同一套 stats**：
#
#     (x - min) / (max - min + 1e-5)
#
# 它的 min/max 来自整个演示数据集。RL 没有预先数据集，因此改用**固定的工作空间
# 边界**：确定性、不随策略漂移，且是纯仿射变换 —— 相对几何关系与绝对位置信息全部
# 保留，只是把数值缩放到适合网络的量级。
#
# 这与 EmpiricalNormalization 有本质区别：后者逐维统计、随策略分布漂移，且会把
# 绝对位置的尺度信息削掉。actor 侧仍然不用它。
POINT_NORM_MIN = (-1.0, -1.0, -0.10)
POINT_NORM_MAX = (1.0, 1.0, 0.60)

# FPS 前的确定性预筛上限。掩码像素可能有几千个，直接对全量跑 64 轮 FPS 偏贵；
# 先按栅格顺序等间距抽到这个数，再 FPS。等间距抽取不含随机性，因此不破坏
# 规格 §12 要求的"采样必须确定性"。
FPS_CANDIDATE_CAP = 512

# 夹爪关键点局部布局（规格 §4.3），相对 ee_frame 的 TCP。
# 约束：至少 3 点非共线，否则末端姿态无法从点集恢复。下列 6 点张成三维。
GRIPPER_KEYPOINTS_LOCAL_M = (
    (0.00, 0.00, 0.00),    # TCP 原点
    (0.00, 0.00, -0.03),   # 接近轴 3 cm
    (0.00, 0.00, -0.06),   # 接近轴 6 cm
    (0.03, 0.00, 0.00),    # 侧向 +x
    (-0.03, 0.00, 0.00),   # 侧向 -x
    (0.00, 0.03, 0.00),    # 侧向 +y，保证姿态可观测
)

# 基座系工作空间裁剪（规格 §4.2 步骤 4，可选但推荐）。
# 机器人基座在世界系 z=0.845、桌面顶 z=0.815，故桌面在基座系约 z=-0.03。
# 取值刻意宽松：目的是丢掉地面与远处背景，不是精修物体。
WORKSPACE_Z_RANGE_M = (-0.10, 0.60)
WORKSPACE_RADIUS_XY_M = 1.0


# ──────────────────────────────────────────────────────────────────────────────
# 反投影
# ──────────────────────────────────────────────────────────────────────────────

def unproject_pixels(
    u: torch.Tensor,
    v: torch.Tensor,
    z: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """把选中的像素按针孔模型反投影到相机光学系。

    等价于 ``isaaclab.utils.math.unproject_depth(..., is_ortho=True)``，但只算
    选中的像素而不是整幅图 —— 640x480 x 64 env 的全图反投影会产生 ~236 MB 的
    中间张量，而我们真正需要的只有几百个掩码像素。

    ``tests`` 里有一条测试把本函数与内置 ``unproject_depth`` 逐元素对比，这是
    规格 §4.2 步骤 2"优先用内置实现避免手写 K⁻¹ 的符号错误"的等价保障：数值
    由内置实现背书，性能由本实现提供。

    Args:
        u: 像素列坐标（浮点）。形状 (B, P)。
        v: 像素行坐标（浮点）。形状 (B, P)。
        z: 沿相机 z 轴的深度，米。形状 (B, P)。
        intrinsics: 内参矩阵。形状 (B, 3, 3) 或 (3, 3)。

    Returns:
        相机光学系下的点，形状 (B, P, 3)。
    """
    if intrinsics.dim() == 2:
        intrinsics = intrinsics.unsqueeze(0).expand(u.shape[0], 3, 3)

    fx = intrinsics[:, 0, 0].unsqueeze(-1)
    fy = intrinsics[:, 1, 1].unsqueeze(-1)
    cx = intrinsics[:, 0, 2].unsqueeze(-1)
    cy = intrinsics[:, 1, 2].unsqueeze(-1)

    x = (u - cx) / fx * z
    y = (v - cy) / fy * z
    return torch.stack([x, y, z], dim=-1)


def pose_in_frame(
    pos_w: torch.Tensor,
    quat_w: torch.Tensor,
    frame_pos_w: torch.Tensor,
    frame_quat_w: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """把任意世界系位姿换算到目标坐标系（本项目里目标系恒为机器人基座系）。

    相机、末端执行器、物体三者都走这个函数 —— 它只是一次 SE(3) 逆变换，与被变换
    的对象是什么无关。

    用于相机时，``quat_w`` 必须来自 ``camera.data.quat_w_ros``（不是
    ``quat_w_world``）：本模块下游的反投影输出在 OpenCV/ROS 光学系，两者必须同
    系，否则就是规格 §9 里"点云整体镜像或旋转 90°"那一行。

    ``frame_*`` 用机器人 root 的实际位姿而不是硬编码的基座常量，这样多环境下每个
    env 的 ``env_origins`` 平移会自动被减掉，无需另行处理。

    仿真专用。真机侧的手眼标定直接给出 camera→base，不需要这一步。
    """
    # 先归一化：下面用共轭当逆，只对单位四元数成立。场景配置里的基座姿态常写成
    # (0.707, 0, 0, -0.707)（范数 0.99985，不是 1），直接用共轭会引入约 1.6e-4 m
    # 的位置误差。物理引擎回读的 root_quat_w 是归一化的，但配置常量不是，所以在
    # 这里统一处理而不是依赖调用方。
    frame_quat_w = frame_quat_w / frame_quat_w.norm(dim=-1, keepdim=True)
    quat_w = quat_w / quat_w.norm(dim=-1, keepdim=True)

    rel_pos = quat_apply_inverse(frame_quat_w, pos_w - frame_pos_w)
    frame_quat_inv = frame_quat_w * torch.tensor(
        [1.0, -1.0, -1.0, -1.0], device=frame_quat_w.device, dtype=frame_quat_w.dtype
    )
    rel_quat = quat_mul(frame_quat_inv, quat_w)
    return rel_pos, rel_quat


# 旧名保留：本函数最初只用于相机，改名后沿用别名以免外部引用断裂。
camera_pose_in_frame = pose_in_frame


# ──────────────────────────────────────────────────────────────────────────────
# 确定性采点
# ──────────────────────────────────────────────────────────────────────────────

def _select_candidates(
    valid: torch.Tensor,
    cap: int = FPS_CANDIDATE_CAP,
) -> tuple[torch.Tensor, torch.Tensor]:
    """从有效像素中确定性地取最多 ``cap`` 个候选。

    ``valid`` 里为 True 的像素按栅格顺序排在前面（stable 排序保证顺序稳定，
    不引入随机性），然后按等间距取 ``cap`` 个；有效像素不足 ``cap`` 时用取模
    循环补齐（即有放回重复，规格 §4.2 的边界情况处理）。

    Args:
        valid: 有效像素掩码。形状 (B, H*W)。

    Returns:
        ``(indices, counts)``：``indices`` 形状 (B, cap) 为像素平铺索引；
        ``counts`` 形状 (B,) 为每个 batch 条目的真实有效像素数，供
        ``visible_ratio`` 日志与空掩码兜底使用。
    """
    B, _ = valid.shape
    device = valid.device

    counts = valid.sum(dim=1)                                  # (B,)
    # stable=True：有效像素之间保持原栅格顺序 → 全过程无 RNG
    order = torch.argsort(valid.to(torch.uint8), dim=1, descending=True, stable=True)

    # 每个条目在自己的 [0, count) 区间内等间距取点；count=0 时置 1 避免除零，
    # 这些条目由调用方按"复用上一帧"处理。
    safe_counts = counts.clamp(min=1).unsqueeze(-1)             # (B, 1)
    steps = torch.arange(cap, device=device).unsqueeze(0)       # (1, cap)
    # 有效点多于 cap 时铺满整个区间；少于 cap 时循环重复
    stride = torch.where(safe_counts > cap, safe_counts.float() / cap, torch.ones_like(safe_counts.float()))
    picks = (steps.float() * stride).long() % safe_counts       # (B, cap)

    indices = torch.gather(order, 1, picks)
    return indices, counts


def farthest_point_sample(points: torch.Tensor, num_samples: int) -> torch.Tensor:
    """批量最远点采样（FPS），确定性。

    起点固定为第 0 个候选点，全过程无随机数，因此静止场景下同一份掩码总是得到
    同一组点。规格 §12 要求确定性 FPS 而非"均匀随机 + FPS"：H=1 的观测没有任何
    时序平滑能吸收采样抖动，掩码边界的逐帧抖动叠加随机重采样会让静止物体的点云
    持续跳变。

    Args:
        points: 候选点。形状 (B, P, 3)。
        num_samples: 目标点数，须 <= P。

    Returns:
        选中点的索引，形状 (B, num_samples)。
    """
    B, P, _ = points.shape
    device = points.device

    selected = torch.zeros(B, num_samples, dtype=torch.long, device=device)
    # 到已选集合的最近距离，初始化为 +inf
    min_dist = torch.full((B, P), float("inf"), device=device)
    current = torch.zeros(B, dtype=torch.long, device=device)   # 确定性起点

    batch_idx = torch.arange(B, device=device)
    for i in range(num_samples):
        selected[:, i] = current
        centroid = points[batch_idx, current].unsqueeze(1)      # (B, 1, 3)
        dist = torch.sum((points - centroid) ** 2, dim=-1)      # (B, P)
        min_dist = torch.minimum(min_dist, dist)
        current = torch.argmax(min_dist, dim=1)

    return selected


# ──────────────────────────────────────────────────────────────────────────────
# 共用主函数（规格 §1.1）
# ──────────────────────────────────────────────────────────────────────────────

def mask_depth_to_pointcloud(
    mask: torch.Tensor,
    depth: torch.Tensor,
    K: torch.Tensor,
    cam_pos: torch.Tensor,
    cam_quat: torch.Tensor,
    m_obj: int = M_OBJ,
    noise_std: float = NOISE_STD_M,
    workspace_z_range_m: tuple[float, float] | None = WORKSPACE_Z_RANGE_M,
    workspace_radius_xy_m: float | None = WORKSPACE_RADIUS_XY_M,
    occlusion_keep_fn=None,
    occlusion_debug=None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """掩码 + 深度 → 目标坐标系下的物体点云。**仿真与真机共用此函数。**

    流程严格按规格 §4.2：过滤无效深度 → 确定性 FPS → 反投影 → 变换到目标系
    → 工作空间裁剪 → 加类型通道 → 加高斯噪声。

    Args:
        mask: 目标物体掩码。形状 (B, H, W)，bool。
        depth: 沿相机 z 轴到成像平面的深度，米。形状 (B, H, W)。
              必须是 ``distance_to_image_plane``（或真机 RGB-D 的 z 深度），
              不能是到光心的欧氏距离。
        K: 内参矩阵。形状 (B, 3, 3) 或 (3, 3)。
        cam_pos: 相机在目标坐标系中的位置。形状 (B, 3)。
        cam_quat: 相机在目标坐标系中的姿态 (w, x, y, z)，OpenCV/ROS 光学系。
                 形状 (B, 4)。
        m_obj: 输出点数。
        noise_std: 高斯噪声标准差（米），仅作用于 xyz。
        workspace_z_range_m: 目标系 z 裁剪范围，None 表示不裁剪。
        workspace_radius_xy_m: 目标系 xy 平面半径裁剪，None 表示不裁剪。
        occlusion_keep_fn: 可选。目标系（基座系）候选点 -> 布尔 keep 掩码 (B, P)
            的可调用对象。非 None 时在 workspace 裁剪之后、FPS 之前调用，把
            ``keep=False`` 的点用该条目第一个存活点顶替（与 workspace 裁剪同一
            替换法），从而把遮挡区从 FPS 候选池里挖掉。默认 None = 不遮挡，
            既有 stage 行为逐位不变。仿真遮挡实验专用，真机侧不传。
        occlusion_debug: 可选 dict。非 None 且遮挡开启时，往里写 ``keep`` (B, cap)
            布尔掩码、``cand_idx`` (B, cap) 候选像素平铺索引以及 ``W``/``H``，
            供仿真验证脚本把被挖掉的像素在录像帧上标红。默认 None = 不记录，
            不参与计算图、不改返回的点云/counts。仿真实验专用，真机侧不传。

    Returns:
        ``(points, visible_counts)``：

        - ``points`` 形状 (B, m_obj, 3)，仅 xyz（无类型通道，对齐参考实现的
          ``in_channels=3``；身份由"分开编码"承担，见模块顶部说明）。
        - ``visible_counts`` 形状 (B,)，每个条目通过全部有效性检查的像素数。
          用于 ``visible_ratio`` 日志与空掩码告警，**不进观测**（规格 §4.2）。
          为 0 的条目其 ``points`` 无意义，调用方须复用上一帧。
    """
    B, H, W = mask.shape
    device = depth.device

    # ── 有效性：掩码内、深度为正、非 NaN/Inf ────────────────────────────────
    # depth 为 0 同时代表"超出裁剪范围"（depth_clipping_behavior="zero"）和
    # "没打到任何东西"，两者都该滤掉，正好一起处理。规格 §4.1②：若相机配置
    # 用了默认的 "none"，超范围像素会是 NaN，会静默把 PointNet 梯度打成 NaN。
    valid = mask & (depth > 0.0) & torch.isfinite(depth)

    valid_flat = valid.reshape(B, H * W)
    depth_flat = depth.reshape(B, H * W)

    # ── 确定性预筛 → FPS ───────────────────────────────────────────────────
    cand_idx, counts = _select_candidates(valid_flat)           # (B, cap), (B,)

    u = (cand_idx % W).to(torch.float32)
    v = (cand_idx // W).to(torch.float32)
    z = torch.gather(depth_flat, 1, cand_idx)

    # 反投影到相机光学系
    pts_cam = unproject_pixels(u, v, z, K)                      # (B, cap, 3)

    # 变换到目标坐标系（机器人基座系）
    rot = matrix_from_quat(cam_quat)                            # (B, 3, 3)
    pts_frame = torch.einsum("bij,bpj->bpi", rot, pts_cam) + cam_pos.unsqueeze(1)

    # 工作空间裁剪：越界点搬到已选集合的质心附近会污染 max pool，因此改为把它们
    # 标成"距离上不可能被 FPS 优先选中"是不可靠的做法。这里直接用替换法：越界点
    # 用该条目的第一个候选点顶替，保证 FPS 输入始终在工作空间内。
    if workspace_z_range_m is not None or workspace_radius_xy_m is not None:
        in_ws = torch.ones(B, pts_frame.shape[1], dtype=torch.bool, device=device)
        if workspace_z_range_m is not None:
            z_lo, z_hi = workspace_z_range_m
            in_ws &= (pts_frame[..., 2] >= z_lo) & (pts_frame[..., 2] <= z_hi)
        if workspace_radius_xy_m is not None:
            r2 = pts_frame[..., 0] ** 2 + pts_frame[..., 1] ** 2
            in_ws &= r2 <= workspace_radius_xy_m ** 2
        # 每条目取第一个在界内的点作为顶替源；整条都越界时保持原值（由 counts=0
        # 之外的上层告警覆盖）。
        first_ok = torch.argmax(in_ws.to(torch.uint8), dim=1)   # (B,)
        fallback = pts_frame[torch.arange(B, device=device), first_ok].unsqueeze(1)
        pts_frame = torch.where(in_ws.unsqueeze(-1), pts_frame, fallback)

        # counts 要反映裁剪后真正可用的像素数。不能直接对候选数组求 in_ws.sum()：
        # 候选是带重复的（有效像素少于 cap 时按取模循环补齐），越界像素会被数很多次
        # 而不是一次，min() 因此几乎永远不生效。
        #
        # counts <= cap 时 picks = steps % counts，所以候选槽位 0..counts-1 恰好各
        # 对应一个不同的有效像素 —— 在这个前缀上求和是精确的。counts > cap 时候选是
        # 等间距子采样，用其界内比例外推。
        cap_len = in_ws.shape[1]
        slot = torch.arange(cap_len, device=device).unsqueeze(0)          # (1, cap)
        prefix = slot < counts.clamp(max=cap_len).unsqueeze(-1)           # (B, cap)
        exact = (in_ws & prefix).sum(dim=1)                               # (B,)
        approx = (counts.to(torch.float32) * in_ws.to(torch.float32).mean(dim=1)).long()
        counts = torch.where(counts <= cap_len, exact, approx)

    # ── 遮挡（可选，默认关闭；仿真实验专用）──────────────────────────────────
    # 在 workspace 裁剪之后、FPS 之前挖掉遮挡区：keep=False 的点用该条目第一个
    # 存活点顶替（与上面 workspace 裁剪的替换法一致）。这样 FPS 的候选池里遮挡区
    # 被"挖空"，m_obj 个采样点全部来自存活区 —— 观测维度不变，内容被削弱。
    if occlusion_keep_fn is not None:
        keep = occlusion_keep_fn(pts_frame)                     # (B, cap) bool
        # 可视化旁路：只把已算好的 keep/候选像素索引抄出去（.detach 只读），
        # 不改 pts_frame、不参与计算图、不影响返回的点云与 counts。
        if occlusion_debug is not None:
            occlusion_debug["keep"] = keep.detach()
            occlusion_debug["cand_idx"] = cand_idx.detach()
            occlusion_debug["W"] = W
            occlusion_debug["H"] = H
        first_kept = torch.argmax(keep.to(torch.uint8), dim=1)   # (B,)
        kept_fallback = pts_frame[torch.arange(B, device=device), first_kept].unsqueeze(1)
        pts_frame = torch.where(keep.unsqueeze(-1), pts_frame, kept_fallback)

    fps_idx = farthest_point_sample(pts_frame, m_obj)           # (B, m_obj)
    pts = torch.gather(pts_frame, 1, fps_idx.unsqueeze(-1).expand(B, m_obj, 3))

    # ── 噪声（仅物体点；参考实现只对 object points 加噪）────────────────────
    if noise_std > 0.0:
        pts = pts + torch.randn_like(pts) * noise_std

    return pts, counts


def dense_occlusion_pixel_map(
    mask: torch.Tensor,
    depth: torch.Tensor,
    K: torch.Tensor,
    cam_pos: torch.Tensor,
    cam_quat: torch.Tensor,
    occlusion_keep_fn,
) -> torch.Tensor:
    """把遮挡谓词作用到每个有效物体像素，返回稠密 (B, H, W) bool 遮挡图。

    仅供仿真验证脚本可视化"遮挡的整体形状"。与观测/训练无关，真机侧不调用。
    观测只为 512 个候选点反投影，这里为所有有效像素反投影——验证单环境离线跑
    可接受（640x480 x 1 env ≈ 30 万点，一次向量化反投影即可）。

    Args:
        mask: 目标物体掩码。形状 (B, H, W)，bool。
        depth: 沿相机 z 轴的平面深度（``distance_to_image_plane``），米。形状 (B, H, W)。
        K: 内参矩阵。形状 (B, 3, 3) 或 (3, 3)。
        cam_pos / cam_quat: 相机在基座系的位姿。形状 (B, 3) / (B, 4)。
        occlusion_keep_fn: (B, P, 3) 基座系点 -> (B, P) bool keep 掩码的可调用对象。

    Returns:
        (B, H, W) bool，True = 该像素被遮挡。
    """
    B, H, W = mask.shape
    device = depth.device

    valid = mask & (depth > 0.0) & torch.isfinite(depth)

    vv, uu = torch.meshgrid(
        torch.arange(H, device=device),
        torch.arange(W, device=device),
        indexing="ij",
    )
    u = uu.unsqueeze(0).expand(B, H, W).reshape(B, H * W).to(torch.float32)
    v = vv.unsqueeze(0).expand(B, H, W).reshape(B, H * W).to(torch.float32)
    z = depth.reshape(B, H * W)
    valid_flat = valid.reshape(B, H * W)

    # 无效像素用 dummy 值反投影（结果随后丢弃），避免 NaN 进矩阵乘法
    u = torch.where(valid_flat, u, torch.zeros_like(u))
    v = torch.where(valid_flat, v, torch.zeros_like(v))
    z = torch.where(valid_flat, z, torch.ones_like(z))

    pts_cam = unproject_pixels(u, v, z, K)                      # (B, H*W, 3)
    rot = matrix_from_quat(cam_quat)                            # (B, 3, 3)
    pts_frame = torch.einsum("bij,bpj->bpi", rot, pts_cam) + cam_pos.unsqueeze(1)

    keep = occlusion_keep_fn(pts_frame)                         # (B, H*W) bool
    occluded_flat = valid_flat & (~keep)
    return occluded_flat.reshape(B, H, W)


def gripper_keypoints(
    ee_pos: torch.Tensor,
    ee_quat: torch.Tensor,
    noise_std: float = 0.0,
    local_offsets_m: tuple[tuple[float, float, float], ...] = GRIPPER_KEYPOINTS_LOCAL_M,
) -> torch.Tensor:
    """由末端位姿正运动学算出夹爪关键点（规格 §4.3）。

    不经过相机，零渲染开销。

    **默认不加噪声**（``noise_std=0.0``）。规格 §12 曾把"噪声是否加在夹爪点上"
    列为论文未明说、由规格自行取"加"；但参考实现给了明确答案：
    ``read_data/mimiclabs.py`` 只对 object points 加噪
    （``noise_object_points`` / ``noise_std=0.01``），robot points 路径上没有任何
    噪声。这也说得通 —— 夹爪点来自正运动学与编码器，精度远高于 RGB-D 深度。

    已知盲点：``local_offsets_m`` 是常量，不随夹爪开合变化，因此夹爪开合度不被
    编码在关键点里。本版接受此盲点（与原工作一致）；若任务需要多次开合，按
    规格 §10 在观测里补一维开合度标量。

    Args:
        ee_pos: 末端位置（目标坐标系）。形状 (B, 3)。
        ee_quat: 末端姿态 (w, x, y, z)（目标坐标系）。形状 (B, 4)。
        noise_std: 高斯噪声标准差（米）。默认 0，对齐参考实现。
        local_offsets_m: 相对末端系的固定偏移。

    Returns:
        形状 (B, N_ROBOT, 3)，仅 xyz。
    """
    B = ee_pos.shape[0]
    device = ee_pos.device
    n = len(local_offsets_m)

    local = torch.tensor(local_offsets_m, device=device, dtype=ee_pos.dtype)   # (n, 3)
    local = local.unsqueeze(0).expand(B, n, 3)

    quat = ee_quat.unsqueeze(1).expand(B, n, 4)
    pts = quat_apply(quat, local) + ee_pos.unsqueeze(1)         # (B, n, 3)

    if noise_std > 0.0:
        pts = pts + torch.randn_like(pts) * noise_std

    return pts


def assemble_point_cloud(
    object_points: torch.Tensor,
    robot_points: torch.Tensor,
) -> torch.Tensor:
    """按观测约定拼接物体点与夹爪点。形状 (B, M_OBJ + N_ROBOT, 3)。

    这里的拼接**只是为了塞进一个扁平观测张量**，网络侧会按
    ``OBJ_FEATURE_LEN`` / ``ROBOT_FEATURE_LEN`` 重新切开、各自过一次 encoder。
    两组点不会进入同一次 max pool，因此顺序不承载语义，但**切分位置依赖这个顺序**
    （物体点在前），改动前先看 ``PointNetActorCritic._encode_actor``。
    """
    return torch.cat([object_points, robot_points], dim=1)


def normalize_points(
    points: torch.Tensor,
    norm_min: tuple[float, float, float] = POINT_NORM_MIN,
    norm_max: tuple[float, float, float] = POINT_NORM_MAX,
) -> torch.Tensor:
    """固定 min-max 仿射归一化到约 [0, 1]（对齐参考实现的 ``past_tracks`` 预处理）。

    物体点与夹爪点**必须共用同一套 min/max**，否则两者的几何关系会被不同的缩放
    破坏 —— 参考实现也是共用一套数据集 stats。

    不做 clamp：落在工作空间外的点得到 [0,1] 之外的值，这是有意的。clamp 会把
    远处的点全压到边界上、丢失"有多远"的信息，而超界本身是有效信号。

    Args:
        points: 形状 (..., 3)，米制，机器人基座系。

    Returns:
        同形状，归一化后。
    """
    lo = torch.tensor(norm_min, device=points.device, dtype=points.dtype)
    hi = torch.tensor(norm_max, device=points.device, dtype=points.dtype)
    return (points - lo) / (hi - lo + 1.0e-5)


# ──────────────────────────────────────────────────────────────────────────────
# 实例分割 ID 解析
# ──────────────────────────────────────────────────────────────────────────────

def build_instance_id_lut(
    id_to_labels: dict,
    prim_path_pattern: str,
    num_envs: int,
    device: torch.device | str,
) -> torch.Tensor:
    """构建 ``instance id -> env 序号 + 1`` 的查表张量。

    规格 §4.1③ 要求实例 ID 每次 reset 后重查、不要硬编码。TiledCamera 这里还有
    两个额外的坑：

    1. ``camera.data.info`` 对 TiledCamera 是**整个 tiled buffer 共享的一个
       dict**（``Camera`` 才是 per-env 列表）。而各环境的物体是不同 prim
       （``env_0/Object``、``env_1/Object`` …），所以 ID 是**逐环境不同**的，
       不能取单个标量 id 去比较。
    2. 一个 ``Object`` prim 下可能有多个 mesh 子 prim，各自拿到不同的 ID，
       因此单个环境也可能对应**多个** ID，掩码要取它们的并集。

    用查表而不是逐 ID 比较，上面两点自然都被覆盖。顺带还解决了邻居环境入画的
    问题：env_spacing 2.5 m 配 55° FOV，邻桌很可能出现在画面里，但 prim path
    精确匹配天然把它们排除在外。

    Args:
        id_to_labels: ``camera.data.info["instance_id_segmentation_fast"]``
            里的 ``idToLabels``，形如 ``{"2": "/World/envs/env_0/Object/..."}``。
        prim_path_pattern: 物体 prim 路径模板，用 ``{env}`` 占位环境序号，
            例如 ``"/World/envs/env_{env}/Object"``。匹配采用前缀比较，所以
            子 mesh 会一并命中。
        num_envs: 环境数。
        device: 输出张量所在设备。

    Returns:
        形状 (max_id + 1,) 的 int32 张量：物体像素处为 ``env 序号 + 1``，
        其余为 0。用法 ``mask = lut[seg] == env_idx + 1``。
    """
    numeric_ids = {}
    for key, label in id_to_labels.items():
        try:
            numeric_ids[int(key)] = str(label)
        except (TypeError, ValueError):
            continue

    max_id = max(numeric_ids) if numeric_ids else 0
    lut = torch.zeros(max_id + 1, dtype=torch.int32, device=device)

    prefixes = [(env, prim_path_pattern.format(env=env)) for env in range(num_envs)]
    for seg_id, label in numeric_ids.items():
        for env, prefix in prefixes:
            if label == prefix or label.startswith(prefix + "/"):
                lut[seg_id] = env + 1
                break

    return lut
