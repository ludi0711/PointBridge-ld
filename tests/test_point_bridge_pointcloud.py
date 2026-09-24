"""点云管线的坐标系不变量测试（规格 §4.5）。

对应 ``point_bridge_rl_observation_spec.md`` §4.5 列出的全部不变量，外加两条
针对本实现的回归测试（反投影与内置实现对齐、相对位姿换算的四元数归一化）。

规格 §7 阶段 0 说约 90% 的问题出在"外参换算错、深度单位错、相机约定不匹配、
实例 ID 查错"这四类。本文件覆盖前三类的纯数学部分（不需要启动 Isaac Sim），
第四类由 :func:`test_instance_lut_*` 覆盖。剩下的"点是否真的贴在物体表面"必须
在渲染器里看，属于阶段 0 的可视化环节，不在这里。

运行（不需要 Isaac Sim，但需要能 import isaaclab.utils.math）::

    conda run -n gx_va_deploy python -m pytest tests/test_point_bridge_pointcloud.py -v

注意 ``isaaclab`` 那个 conda 环境里 numpy 2.4.6 与 scipy 1.11.4 的 ABI 不兼容
（``import isaaclab.utils`` 会在 trimesh→scipy 处炸），与本改动无关。用
``gx_va_deploy`` 跑测试。
"""

from __future__ import annotations

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from isaaclab.utils.math import (  # noqa: E402
    matrix_from_quat,
    quat_apply,
    quat_from_euler_xyz,
    quat_mul,
    unproject_depth,
)

from configs.point_bridge_pointcloud import (  # noqa: E402
    GRIPPER_KEYPOINTS_LOCAL_M,
    M_OBJ,
    N_ROBOT,
    NOISE_STD_M,
    NUM_POINTS,
    OBJ_FEATURE_LEN,
    POINT_DIM,
    POINT_NORM_MAX,
    POINT_NORM_MIN,
    ROBOT_FEATURE_LEN,
    assemble_point_cloud,
    build_instance_id_lut,
    camera_pose_in_frame,
    farthest_point_sample,
    gripper_keypoints,
    mask_depth_to_pointcloud,
    normalize_points,
    unproject_pixels,
)


# ──────────────────────────────────────────────────────────────────────────────
# 辅助
# ──────────────────────────────────────────────────────────────────────────────

H, W = 24, 32
FX = FY = 40.0
CX, CY = W / 2.0, H / 2.0


def make_K() -> torch.Tensor:
    return torch.tensor([[FX, 0.0, CX], [0.0, FY, CY], [0.0, 0.0, 1.0]])


def make_scene(depth_value: float = 1.2, patch: tuple[int, int, int, int] = (8, 15, 10, 20)):
    """一块矩形掩码区域 + 常深度。返回 ``(mask, depth)``，形状 (1, H, W)。"""
    v0, v1, u0, u1 = patch
    mask = torch.zeros(1, H, W, dtype=torch.bool)
    mask[0, v0:v1, u0:u1] = True
    depth = torch.zeros(1, H, W)
    # 加一点深度变化，避免完全共面导致 FPS 出现大量精确并列
    vv = torch.arange(H).view(1, H, 1).float()
    uu = torch.arange(W).view(1, 1, W).float()
    depth[:] = depth_value + 0.002 * vv + 0.001 * uu
    return mask, depth


def identity_cam(batch: int = 1):
    pos = torch.zeros(batch, 3)
    quat = torch.zeros(batch, 4)
    quat[:, 0] = 1.0
    return pos, quat


# 用 identity_cam 时点落在目标系 z≈+1.2（相机在原点、朝 +Z 看），这在真实场景里
# 是不可能的位置，会被默认的 WORKSPACE_Z_RANGE_M 全部裁掉。只关心采样/类型通道
# 等与裁剪无关的行为时，用这个把裁剪关掉。
NO_CLIP = {"workspace_z_range_m": None, "workspace_radius_xy_m": None}


def assert_point_sets_close(a: torch.Tensor, b: torch.Tensor, atol: float):
    """比较两组点的**集合**是否一致，不要求逐行对应。

    FPS 只在"无精确并列"时才对刚体变换严格等变：``argmax`` 遇到并列取首个索引，
    而并列关系会被 1e-7 量级的浮点差异改写，于是选中的是另一个几乎重合的候选点。
    这不是缺陷 —— 规格 §12 要求的是"同一输入给出同一输出"（由
    ``test_pipeline_is_deterministic`` 覆盖，静止场景下深度逐位相同，输出逐位
    相同），而不是"选中的索引对刚体变换不变"。

    因此这里用双向 Hausdorff 距离：两组点必须互相覆盖到 ``atol`` 以内。实测偏差
    在毫米级，远低于 NOISE_STD_M = 1 cm 的噪声地板，且 PointNet 的对称池化本身
    对点序不敏感。
    """
    d = torch.cdist(a, b)
    fwd = d.min(dim=1).values.max()
    bwd = d.min(dim=0).values.max()
    assert max(fwd, bwd) < atol, f"Hausdorff fwd={fwd:.2e} bwd={bwd:.2e} atol={atol:.2e}"


# ──────────────────────────────────────────────────────────────────────────────
# 反投影正确性
# ──────────────────────────────────────────────────────────────────────────────

def test_unproject_matches_builtin():
    """自写的稀疏反投影必须与 ``unproject_depth`` 逐元素相同。

    规格 §4.2 步骤 2 要求优先用内置实现，因为手写 K⁻¹ 极易搞错相机约定的符号。
    我们为性能只算掩码像素（全图 640x480x64env 的中间张量约 236 MB），所以用这
    条测试把数值正确性挂回内置实现。

    注意 ``unproject_depth`` 内部用 ``meshgrid(indexing="ij")`` 配
    ``depth.transpose_(1, 2)``，其平铺输出是**转置**的像素顺序（先 u 后 v），
    reshape 时必须按 (W, H) 而不是 (H, W)。这个顺序差异本身就是规格 §9 里
    "点云整体镜像或旋转 90°"的一种常见来源。
    """
    K = make_K()
    depth = torch.rand(1, H, W) + 0.5

    ref = unproject_depth(depth, K).reshape(1, W, H, 3).permute(0, 2, 1, 3)

    vv, uu = torch.meshgrid(
        torch.arange(H).float(), torch.arange(W).float(), indexing="ij"
    )
    mine = unproject_pixels(
        uu.reshape(1, -1), vv.reshape(1, -1), depth.reshape(1, -1), K
    ).reshape(1, H, W, 3)

    assert torch.allclose(ref, mine, atol=1e-6), (ref - mine).abs().max()


def test_depth_is_along_camera_z():
    """深度语义必须是"沿相机 z 轴到成像平面"，不是到光心的欧氏距离。

    规格 §4.1①：用错 ``distance_to_camera`` 不会报错，只让边缘像素系统性偏差。
    判据：所有点的 z 分量恰好等于输入深度（欧氏距离语义下边缘像素会小于它）。
    """
    K = make_K()
    depth = torch.full((1, H, W), 1.3)
    vv, uu = torch.meshgrid(
        torch.arange(H).float(), torch.arange(W).float(), indexing="ij"
    )
    pts = unproject_pixels(uu.reshape(1, -1), vv.reshape(1, -1), depth.reshape(1, -1), K)
    assert torch.allclose(pts[..., 2], torch.full_like(pts[..., 2], 1.3), atol=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# 相对位姿换算
# ──────────────────────────────────────────────────────────────────────────────

def test_camera_pose_in_frame_matches_two_step_transform():
    """先转世界系再转基座系，应与"直接用相对位姿转"结果一致。"""
    K = make_K()
    mask, depth = make_scene()

    cam_pos_w = torch.tensor([[-0.3796, -0.4213, 1.9148]])
    cam_quat_w = quat_from_euler_xyz(
        torch.tensor([2.6]), torch.tensor([0.05]), torch.tensor([1.2])
    )
    base_pos_w = torch.tensor([[0.0, 0.0, 0.845]])
    # 刻意用场景配置里那个非归一化的常量（范数 0.99985）
    base_quat_w = torch.tensor([[0.707, 0.0, 0.0, -0.707]])

    rel_pos, rel_quat = camera_pose_in_frame(cam_pos_w, cam_quat_w, base_pos_w, base_quat_w)

    torch.manual_seed(0)
    direct, _ = mask_depth_to_pointcloud(
        mask, depth, K, rel_pos, rel_quat, noise_std=0.0,
        workspace_z_range_m=None, workspace_radius_xy_m=None,
    )
    torch.manual_seed(0)
    in_world, _ = mask_depth_to_pointcloud(
        mask, depth, K, cam_pos_w, cam_quat_w, noise_std=0.0,
        workspace_z_range_m=None, workspace_radius_xy_m=None,
    )

    bq = base_quat_w / base_quat_w.norm(dim=-1, keepdim=True)
    bq_inv = bq * torch.tensor([1.0, -1.0, -1.0, -1.0])
    expect = quat_apply(
        bq_inv.expand(M_OBJ, 4), in_world[0, :, :3] - base_pos_w
    )
    # 集合比较：两条路径的浮点误差会改写 FPS 的并列顺序，见 assert_point_sets_close。
    # 1 mm 容差，远低于 NOISE_STD_M = 1 cm。
    assert_point_sets_close(direct[0, :, :3], expect, atol=1.0e-3)


def test_camera_pose_in_frame_normalizes_quaternion():
    """非归一化的基座四元数不得引入位置误差（共轭当逆只对单位四元数成立）。"""
    cam_pos_w = torch.tensor([[-0.3796, -0.4213, 1.9148]])
    cam_quat_w = quat_from_euler_xyz(
        torch.tensor([2.6]), torch.tensor([0.05]), torch.tensor([1.2])
    )
    base_pos_w = torch.tensor([[0.0, 0.0, 0.845]])

    raw = torch.tensor([[0.707, 0.0, 0.0, -0.707]])              # 范数 0.99985
    unit = raw / raw.norm(dim=-1, keepdim=True)

    p_raw, q_raw = camera_pose_in_frame(cam_pos_w, cam_quat_w, base_pos_w, raw)
    p_unit, q_unit = camera_pose_in_frame(cam_pos_w, cam_quat_w, base_pos_w, unit)

    assert torch.allclose(p_raw, p_unit, atol=1e-6)
    assert torch.allclose(q_raw, q_unit, atol=1e-6)


def test_env_origins_cancel_in_base_frame():
    """多环境下 ``env_origins`` 的平移必须被自动消掉（规格 §4.5：点在基座系）。"""
    K = make_K()
    mask, depth = make_scene()
    mask = mask.repeat(2, 1, 1)
    depth = depth.repeat(2, 1, 1)

    cam_off = torch.tensor([-0.3796, -0.4213, 1.9148])
    cam_quat = quat_from_euler_xyz(
        torch.tensor([2.6]), torch.tensor([0.05]), torch.tensor([1.2])
    ).repeat(2, 1)
    base_quat = torch.tensor([[0.707, 0.0, 0.0, -0.707]]).repeat(2, 1)

    # env 1 整体平移 2.5 m，相机与基座一起动
    origins = torch.tensor([[0.0, 0.0, 0.0], [2.5, 0.0, 0.0]])
    cam_pos_w = cam_off.unsqueeze(0) + origins
    base_pos_w = torch.tensor([[0.0, 0.0, 0.845]]) + origins

    rel_pos, rel_quat = camera_pose_in_frame(cam_pos_w, cam_quat, base_pos_w, base_quat)
    torch.manual_seed(0)
    pts, _ = mask_depth_to_pointcloud(mask, depth, K, rel_pos, rel_quat, noise_std=0.0)

    # env 1 的 cam_pos_w - base_pos_w 要在 2.5 m 上做减法，浮点抵消留下 ~1e-7 m
    # 残差，足以改写 FPS 的并列顺序，因此用集合比较。1 mm 容差远低于 1 cm 噪声。
    assert_point_sets_close(pts[0, :, :3], pts[1, :, :3], atol=1.0e-3)
    assert pts.shape == (2, M_OBJ, 3)


# ──────────────────────────────────────────────────────────────────────────────
# 采样确定性（规格 §12）
# ──────────────────────────────────────────────────────────────────────────────

def test_fps_is_deterministic():
    torch.manual_seed(0)
    pts = torch.rand(3, 200, 3)
    a = farthest_point_sample(pts, M_OBJ)
    b = farthest_point_sample(pts, M_OBJ)
    assert torch.equal(a, b)


def test_fps_spreads_better_than_prefix():
    """FPS 应比"取前 N 个"覆盖更大范围，否则它没起作用。"""
    torch.manual_seed(0)
    pts = torch.rand(1, 300, 3)
    idx = farthest_point_sample(pts, M_OBJ)
    picked = pts[0, idx[0]]
    prefix = pts[0, :M_OBJ]
    assert picked.std(dim=0).mean() >= prefix.std(dim=0).mean()


def test_static_scene_is_stable():
    """同一份掩码 + 深度必须得到同一份点云（规格 §9"静止场景下点云跳变"）。"""
    K = make_K()
    mask, depth = make_scene()
    cam_pos, cam_quat = identity_cam()

    torch.manual_seed(0)
    a, ca = mask_depth_to_pointcloud(mask, depth, K, cam_pos, cam_quat, noise_std=0.0)
    torch.manual_seed(0)
    b, cb = mask_depth_to_pointcloud(mask, depth, K, cam_pos, cam_quat, noise_std=0.0)
    assert torch.equal(a, b)
    assert torch.equal(ca, cb)


def test_fps_point_set_is_rotation_equivariant():
    """旋转相机后，选出的点集应与"先选点再旋转"基本一致。

    FPS 只依赖点间距离，而刚体旋转保距，所以点**集合**在数学上不变。但
    ``argmax`` 在浮点并列时的取胜者会变，因此少数点会被换成邻近的候选点。这里
    用集合距离（每点到另一集合的最近距离）判定，容差取 5 mm —— 远小于
    ``NOISE_STD_M`` = 1 cm，即该抖动被观测噪声完全淹没，不影响策略。

    完全共面的掩码（所有候选深度相同）会产生大量精确并列，是最坏情况；
    :func:`make_scene` 刻意给了微小深度梯度以贴近真实工件。
    """
    K = make_K()
    mask, depth = make_scene()
    cam_pos = torch.zeros(1, 3)

    q_a = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    q_b = quat_from_euler_xyz(
        torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([0.4])
    )

    torch.manual_seed(0)
    pa, _ = mask_depth_to_pointcloud(
        mask, depth, K, cam_pos, q_a, noise_std=0.0,
        workspace_z_range_m=None, workspace_radius_xy_m=None,
    )
    torch.manual_seed(0)
    pb, _ = mask_depth_to_pointcloud(
        mask, depth, K, cam_pos, q_b, noise_std=0.0,
        workspace_z_range_m=None, workspace_radius_xy_m=None,
    )

    # 把 b 的点转回 a 的朝向后比较集合
    rot_inv = matrix_from_quat(q_b).transpose(1, 2)
    back = torch.einsum("bij,bpj->bpi", rot_inv, pb[:, :, :3])

    d = torch.cdist(back[0], pa[0, :, :3])
    assert d.min(dim=1).values.max() < 5e-3, d.min(dim=1).values.max()


# ──────────────────────────────────────────────────────────────────────────────
# 数值卫生（规格 §4.5 / §9）
# ──────────────────────────────────────────────────────────────────────────────

def test_zero_and_nan_depth_excluded():
    """0 与 NaN 深度必须被滤掉，且不得泄漏进输出。

    规格 §4.1②：``depth_clipping_behavior="none"`` 时超范围像素是 NaN，会静默把
    PointNet 梯度打成 NaN；0 同时代表"超范围"和"没打到东西"。
    """
    K = make_K()
    mask, depth = make_scene()
    depth[0, 8, 10] = 0.0
    depth[0, 9, 11] = float("nan")
    cam_pos, cam_quat = identity_cam()

    pts, counts = mask_depth_to_pointcloud(
        mask, depth, K, cam_pos, cam_quat, noise_std=0.0,
        workspace_z_range_m=None, workspace_radius_xy_m=None,
    )
    assert int(counts[0]) == int(mask.sum()) - 2
    assert torch.isfinite(pts).all()


def test_no_points_at_optical_center():
    """规格 §4.5：不得有点堆积在原点或相机光心。

    无效深度直接反投影会得到 (0,0,0)，一整团错点会彻底污染 max pooling
    （规格 §11.1 对真机侧的同一告警）。
    """
    K = make_K()
    mask, depth = make_scene()
    depth[0, 8, 10] = 0.0
    cam_pos = torch.tensor([[0.3, -0.2, 1.0]])
    _, cam_quat = identity_cam()

    pts, _ = mask_depth_to_pointcloud(
        mask, depth, K, cam_pos, cam_quat, noise_std=0.0,
        workspace_z_range_m=None, workspace_radius_xy_m=None,
    )
    dist_to_center = (pts[0, :, :3] - cam_pos).norm(dim=-1)
    assert dist_to_center.min() > 1e-3


def test_workspace_clip_removes_outliers_and_updates_count():
    """越界点必须被剔除，且 ``visible_counts`` 要反映裁剪后的真实可用像素数。

    counts 不能直接对候选数组求和：候选带重复（有效像素少于 cap 时取模循环补齐），
    越界像素会被数很多次。
    """
    K = make_K()
    mask, depth = make_scene()
    # 掩码内插几个极远像素（模拟深度穿到背景几何）
    depth[0, 8, 10] = 18.0
    depth[0, 8, 11] = 18.0
    depth[0, 8, 12] = 18.0
    cam_pos, cam_quat = identity_cam()

    _, counts_noclip = mask_depth_to_pointcloud(
        mask, depth, K, cam_pos, cam_quat, noise_std=0.0,
        workspace_z_range_m=None, workspace_radius_xy_m=None,
    )
    pts, counts_clip = mask_depth_to_pointcloud(
        mask, depth, K, cam_pos, cam_quat, noise_std=0.0,
        workspace_z_range_m=(-5.0, 5.0), workspace_radius_xy_m=None,
    )
    assert int(counts_clip[0]) == int(counts_noclip[0]) - 3
    assert pts[0, :, 2].max() < 5.0


def test_output_is_finite_and_shaped():
    K = make_K()
    mask, depth = make_scene()
    cam_pos, cam_quat = identity_cam()
    pts, counts = mask_depth_to_pointcloud(mask, depth, K, cam_pos, cam_quat)
    assert pts.shape == (1, M_OBJ, 3)
    assert counts.shape == (1,)
    assert torch.isfinite(pts).all()


def test_empty_mask_reports_zero_count():
    """空掩码必须让 counts 归零，供上层"复用上一帧并计数报警"（规格 §4.2）。"""
    K = make_K()
    _, depth = make_scene()
    mask = torch.zeros(1, H, W, dtype=torch.bool)
    cam_pos, cam_quat = identity_cam()
    pts, counts = mask_depth_to_pointcloud(mask, depth, K, cam_pos, cam_quat)
    assert int(counts[0]) == 0
    assert torch.isfinite(pts).all()   # 仍不得产生 NaN


def test_fewer_valid_than_m_obj_is_padded():
    """有效像素少于 M_OBJ 时按有放回重复补齐（规格 §4.2 边界情况）。"""
    K = make_K()
    _, depth = make_scene()
    mask = torch.zeros(1, H, W, dtype=torch.bool)
    mask[0, 5:8, 5:8] = True                  # 9 个像素 < 64
    cam_pos, cam_quat = identity_cam()

    # 关掉工作空间裁剪：这个夹具的相机在原点朝 +Z 看，点落在基座系 z=+1.2，
    # 会被 WORKSPACE_Z_RANGE_M=(-0.1, 0.6) 全部裁掉。本测试考察的是补齐行为。
    pts, counts = mask_depth_to_pointcloud(
        mask, depth, K, cam_pos, cam_quat, noise_std=0.0,
        workspace_z_range_m=None, workspace_radius_xy_m=None,
    )
    assert int(counts[0]) == 9
    assert pts.shape == (1, M_OBJ, 3)
    assert torch.isfinite(pts).all()
    unique = torch.unique(pts[0, :, :3], dim=0)
    assert unique.shape[0] <= 9


# ──────────────────────────────────────────────────────────────────────────────
# 观测布局、噪声、归一化
# ──────────────────────────────────────────────────────────────────────────────

def test_point_dim_is_three_no_type_channel():
    """每点必须是 3 维纯 xyz。

    参考实现 ``PointNetEncoderXYZ`` 是 ``in_channels=3``；身份由"物体点与夹爪点
    分开编码成两个 token"承担，不靠逐点类型标签。若这里变回 4 维，网络侧的切分
    偏移会全部错位。
    """
    K = make_K()
    mask, depth = make_scene()
    cam_pos, cam_quat = identity_cam()

    obj, _ = mask_depth_to_pointcloud(mask, depth, K, cam_pos, cam_quat)
    rob = gripper_keypoints(torch.zeros(1, 3), torch.tensor([[1.0, 0.0, 0.0, 0.0]]))

    assert obj.shape == (1, M_OBJ, 3)
    assert rob.shape == (1, N_ROBOT, 3)
    assert POINT_DIM == 3


def test_flat_observation_layout_matches_network_split():
    """扁平布局必须是 [物体点 | 夹爪点]，且长度常量与实际切分一致。

    网络按 ``OBJ_FEATURE_LEN`` / ``ROBOT_FEATURE_LEN`` 切开扁平观测，切错就会把
    夹爪点当物体点喂进去 —— 不会报错，只会静默学不出来。
    """
    K = make_K()
    mask, depth = make_scene()
    cam_pos, cam_quat = identity_cam()

    obj, _ = mask_depth_to_pointcloud(mask, depth, K, cam_pos, cam_quat, noise_std=0.0)
    rob = gripper_keypoints(torch.zeros(1, 3), torch.tensor([[1.0, 0.0, 0.0, 0.0]]))
    cloud = assemble_point_cloud(obj, rob)
    flat = cloud.reshape(1, -1)

    assert cloud.shape == (1, NUM_POINTS, 3)
    assert OBJ_FEATURE_LEN == M_OBJ * 3
    assert ROBOT_FEATURE_LEN == N_ROBOT * 3
    assert flat.shape[-1] == OBJ_FEATURE_LEN + ROBOT_FEATURE_LEN

    # 按网络的切法还原，必须逐元素等于原始两组点
    obj_back = flat[:, :OBJ_FEATURE_LEN].view(1, M_OBJ, 3)
    rob_back = flat[:, OBJ_FEATURE_LEN:].view(1, N_ROBOT, 3)
    assert torch.equal(obj_back, obj)
    assert torch.equal(rob_back, rob)


def test_noise_applied_to_object_points_only():
    """噪声只加在物体点上；夹爪点默认无噪声。

    参考实现 ``read_data/mimiclabs.py`` 只对 object points 加噪
    （``noise_object_points`` / ``noise_std=0.01``），robot points 路径上没有任何
    噪声 —— 夹爪点来自正运动学，精度远高于 RGB-D 深度。
    """
    K = make_K()
    mask, depth = make_scene()
    cam_pos, cam_quat = identity_cam()

    torch.manual_seed(0)
    clean, _ = mask_depth_to_pointcloud(mask, depth, K, cam_pos, cam_quat, noise_std=0.0)
    torch.manual_seed(0)
    noisy, _ = mask_depth_to_pointcloud(
        mask, depth, K, cam_pos, cam_quat, noise_std=NOISE_STD_M
    )
    delta = (noisy - clean).std().item()
    assert 0.5 * NOISE_STD_M < delta < 2.0 * NOISE_STD_M, delta

    # 夹爪点：默认调用不得引入任何随机性
    ee_pos = torch.tensor([[0.1, -0.3, 0.4]])
    ee_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    a = gripper_keypoints(ee_pos, ee_quat)
    b = gripper_keypoints(ee_pos, ee_quat)
    assert torch.equal(a, b), "gripper keypoints must be noise-free by default"


def test_normalize_points_is_shared_affine():
    """归一化必须是仿射、且物体点与夹爪点共用同一套 min/max。

    共用是关键：两组点若用不同缩放，它们之间的几何关系（夹爪离物体多远）就会被
    破坏。参考实现也是共用一套数据集 stats。
    """
    lo = torch.tensor(POINT_NORM_MIN)
    hi = torch.tensor(POINT_NORM_MAX)

    # 边界点映射到约 0 / 1
    at_min = normalize_points(lo.view(1, 1, 3))
    at_max = normalize_points(hi.view(1, 1, 3))
    assert torch.allclose(at_min, torch.zeros_like(at_min), atol=1e-5)
    assert torch.allclose(at_max, torch.ones_like(at_max), atol=1e-3)

    # 仿射性：归一化保持共线与比例关系
    p = torch.randn(4, 7, 3) * 0.3
    q = torch.randn(4, 7, 3) * 0.3
    t = 0.37
    lhs = normalize_points(t * p + (1 - t) * q)
    rhs = t * normalize_points(p) + (1 - t) * normalize_points(q)
    assert torch.allclose(lhs, rhs, atol=1e-6)

    # 不 clamp：超界的点得到 [0,1] 之外的值（保留"有多远"的信息）
    far = normalize_points(torch.tensor([[[0.0, 0.0, 5.0]]]))
    assert far.max() > 1.0


# ──────────────────────────────────────────────────────────────────────────────
# 夹爪关键点（规格 §4.3）
# ──────────────────────────────────────────────────────────────────────────────

def test_gripper_keypoints_shape_and_tcp():
    ee_pos = torch.tensor([[0.1, -0.3, 0.4]])
    ee_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    pts = gripper_keypoints(ee_pos, ee_quat, noise_std=0.0)
    assert pts.shape == (1, N_ROBOT, 3)
    assert torch.allclose(pts[0, 0, :3], ee_pos[0], atol=1e-6)   # 第 0 点是 TCP


def test_gripper_keypoints_move_rigidly():
    """规格 §4.5：夹爪点应随末端刚性移动。"""
    ee_quat = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    a = gripper_keypoints(torch.tensor([[0.1, -0.3, 0.4]]), ee_quat, noise_std=0.0)
    b = gripper_keypoints(torch.tensor([[0.2, -0.3, 0.4]]), ee_quat, noise_std=0.0)
    shift = b[..., :3] - a[..., :3]
    assert torch.allclose(shift, torch.tensor([0.1, 0.0, 0.0]).expand_as(shift), atol=1e-6)


def test_gripper_keypoints_preserve_pairwise_distance_under_rotation():
    """旋转末端只应旋转点集，不改变点间距（刚体变换）。"""
    ee_pos = torch.tensor([[0.1, -0.3, 0.4]])
    q_a = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
    q_b = quat_from_euler_xyz(
        torch.tensor([0.3]), torch.tensor([-0.2]), torch.tensor([0.7])
    )
    a = gripper_keypoints(ee_pos, q_a, noise_std=0.0)[0, :, :3]
    b = gripper_keypoints(ee_pos, q_b, noise_std=0.0)[0, :, :3]
    assert torch.allclose(torch.cdist(a, a), torch.cdist(b, b), atol=1e-6)


def test_gripper_keypoints_span_3d():
    """规格 §4.3 约束：至少 3 点非共线，否则末端姿态无法从点集恢复。"""
    local = torch.tensor(GRIPPER_KEYPOINTS_LOCAL_M)
    centered = local - local.mean(dim=0)
    assert torch.linalg.matrix_rank(centered, tol=1e-6) == 3


def test_gripper_keypoints_rotate_with_ee_orientation():
    """姿态必须真的作用到关键点上（否则姿态在观测里不可见）。"""
    ee_pos = torch.zeros(1, 3)
    q = quat_from_euler_xyz(
        torch.tensor([0.0]), torch.tensor([0.0]), torch.tensor([torch.pi / 2])
    )
    pts = gripper_keypoints(ee_pos, q, noise_std=0.0)
    local = torch.tensor(GRIPPER_KEYPOINTS_LOCAL_M).unsqueeze(0)
    expect = quat_apply(q.expand(N_ROBOT, 4), local[0])
    assert torch.allclose(pts[0, :, :3], expect, atol=1e-6)


# ──────────────────────────────────────────────────────────────────────────────
# 实例 ID 查表（规格 §4.1③）
# ──────────────────────────────────────────────────────────────────────────────

def test_instance_lut_maps_each_env():
    labels = {
        "1": "/World/GroundPlane",
        "2": "/World/envs/env_0/Object",
        "3": "/World/envs/env_1/Object",
        "4": "/World/envs/env_2/Object",
        "5": "/World/envs/env_0/WorkpieceTable",
    }
    lut = build_instance_id_lut(labels, "/World/envs/env_{env}/Object", 3, "cpu")
    assert int(lut[2]) == 1      # env 0 -> 1
    assert int(lut[3]) == 2
    assert int(lut[4]) == 3
    assert int(lut[1]) == 0      # 地面排除
    assert int(lut[5]) == 0      # 桌子排除


def test_instance_lut_unions_submeshes():
    """一个 Object prim 下的多个 mesh 会拿到不同 ID，掩码要取并集。"""
    labels = {
        "7": "/World/envs/env_0/Object/geometry/mesh_0",
        "8": "/World/envs/env_0/Object/geometry/mesh_1",
    }
    lut = build_instance_id_lut(labels, "/World/envs/env_{env}/Object", 1, "cpu")
    assert int(lut[7]) == 1
    assert int(lut[8]) == 1


def test_instance_lut_prefix_guard():
    """``env_10`` 不得被 ``env_1`` 的前缀误匹配。"""
    labels = {
        "2": "/World/envs/env_1/Object",
        "3": "/World/envs/env_10/Object",
    }
    lut = build_instance_id_lut(labels, "/World/envs/env_{env}/Object", 2, "cpu")
    assert int(lut[2]) == 2      # env_1 -> 2
    assert int(lut[3]) == 0      # env_10 超出 num_envs=2，不匹配


def test_instance_lut_ignores_non_numeric_keys():
    labels = {"BACKGROUND": "BACKGROUND", "2": "/World/envs/env_0/Object"}
    lut = build_instance_id_lut(labels, "/World/envs/env_{env}/Object", 1, "cpu")
    assert int(lut[2]) == 1


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
