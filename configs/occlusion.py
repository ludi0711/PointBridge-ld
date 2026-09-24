"""坐标级工件遮挡 —— 为 stage2 遮挡实验提供纯几何谓词（仿真实验专用）。

规格与动机：
  本实验要回答"点云表征在工件被部分遮挡时还能不能学"。遮挡必须是**集中的一块**
  （连通的几何区域），不是随机摊开的散点，因此用几何谓词（半空间 / 球）在反投影
  后的基座系点云上按坐标挖掉一块，而不是在掩码上随机置零。

  所有谓词在**工件局部系**下定义。理由：工件位姿是随机化的（0.25x0.25 平移 + yaw），
  若在基座系绝对坐标定义遮挡，无法保证"遮住工件的 25%"——工件挪一下就遮不到；而在
  局部系里"遮住 +x 半边"永远指工件的同一块，severity 才有确定含义。

  本模块不 import 任何 sim / env 依赖（``isaaclab.utils.math`` 是纯 torch），因此
  真机侧也能 import，不破坏"唯一共用实现"的约束。遮挡本身只在仿真训练脚本里注入。

  **确定性**：所有谓词都是纯坐标运算、无 RNG，保持规格 §12 的采样确定性。
  random_sphere/random_box 的随机性**不在此层**：球心/盒心/朝向由上层（env 事件）
  每 episode 采样后作为张量传入，本层仍是确定性坐标运算。

severity 语义（逐模式）：
  halfspace  severity = 沿指定轴切除的比例。0=不遮，0.5=切掉半边，1=切光。
             axis 用带符号的轴名表示切哪一侧：'x' 切 +x 侧，'-x' 切 -x 侧。
  sphere     severity = 遮挡球半径占工件外接球半径（对角线一半）的比例。0=不遮，
             1=半径盖满整个外接球（几乎全遮）。center 是球心在局部包围盒内的归一化
             位置（0..1 分数），默认 (0.5,0.5,0.5)=工件中心。
  random_sphere（圆形）severity = 遮挡球半径占工件外接球半径的比例（同 sphere）。
             球心由上层每 episode 在工件局部包围盒内随机采样（center_local），
             位置随机、无固定方向。
  random_box          severity = 删除比例：按随机朝向的各向异性椭球度量，删掉离盒心
             最近的 severity 比例候选点（比例严格精确、连通"阴影"、方向随机，无需
             搜索）。盒心（center_local）每 episode 从**面向相机的那侧表面点**随机
             抽一个（保证阴影落在相机可见面）；盒朝向（shape_quat_local）随机
             SO(3)，作为椭球主轴方向，**不与工件轴对齐**——工件局部系只用于定位与
             定标，不约束椭球方向。
"""

from __future__ import annotations

import torch

from isaaclab.utils.math import quat_apply_inverse


def _base_to_object_local(
    pts_base: torch.Tensor,
    obj_pos_b: torch.Tensor,
    obj_quat_b: torch.Tensor,
) -> torch.Tensor:
    """基座系点 → 工件局部系。

    ``obj_quat_b`` 是工件在基座系的姿态，局部 = 逆旋转(点 - 工件位置)。
    """
    B, P, _ = pts_base.shape
    q = obj_quat_b.unsqueeze(1).expand(B, P, 4)
    return quat_apply_inverse(q, pts_base - obj_pos_b.unsqueeze(1))


def occlusion_keep(
    pts_base: torch.Tensor,
    obj_pos_b: torch.Tensor,
    obj_quat_b: torch.Tensor,
    obj_local_min: torch.Tensor,
    obj_local_max: torch.Tensor,
    mode: str = "none",
    severity: float = 0.0,
    axis: str = "x",
    center: tuple[float, float, float] = (0.5, 0.5, 0.5),
    center_local: torch.Tensor | None = None,
    shape_quat_local: torch.Tensor | None = None,
) -> torch.Tensor:
    """返回布尔 keep 掩码 (B, P)：True=保留，False=被遮挡剔除。

    Args:
        pts_base: 候选点，基座系。形状 (B, P, 3)。
        obj_pos_b / obj_quat_b: 工件位姿（基座系）。形状 (B, 3) / (B, 4)。
        obj_local_min / obj_local_max: 工件局部包围盒。形状 (3,)。
        mode: 'none' | 'halfspace' | 'sphere' | 'random_sphere' | 'random_box'。
        severity: 遮挡程度，见模块 docstring 的逐模式语义。
        axis: halfspace 用，带符号轴名 'x'/'-x'/'y'/'-y'/'z'/'-z'。
        center: sphere 用，球心在局部包围盒内的归一化分数 (cx, cy, cz)。
        center_local: random_sphere/random_box 用。已采样好的球心/盒心，工件局部系，
            形状 (B, 3)，米制。random_box 的盒心来自面向相机的那侧表面点。
            由上层每 episode 采样传入，本函数保持无 RNG。
        shape_quat_local: random_box 用。已采样好的随机 SO(3) 朝向，工件局部系，
            形状 (B, 4) (w,x,y,z)。作为各向异性椭球度量（轴长=工件局部包围盒 span）
            的主轴方向，与工件轴解耦。
    """
    keep = torch.ones(
        pts_base.shape[0], pts_base.shape[1], dtype=torch.bool, device=pts_base.device
    )
    if mode == "none" or severity <= 0.0:
        return keep

    lo = obj_local_min.to(pts_base.device)
    hi = obj_local_max.to(pts_base.device)
    span = (hi - lo).clamp(min=1.0e-6)

    if mode == "halfspace":
        neg = axis.startswith("-")
        a = axis.lstrip("-")
        if len(a) != 1 or a not in "xyz":
            raise ValueError(
                f"halfspace axis must be one of x/-x/y/-y/z/-z; got {axis!r}"
            )
        i = "xyz".index(a)
        local = _base_to_object_local(pts_base, obj_pos_b, obj_quat_b)
        if neg:
            cut = lo[i] + severity * span[i]
            keep = local[..., i] >= cut
        else:
            cut = hi[i] - severity * span[i]
            keep = local[..., i] <= cut
    elif mode == "sphere":
        local = _base_to_object_local(pts_base, obj_pos_b, obj_quat_b)
        c = lo + torch.tensor(center, device=pts_base.device, dtype=pts_base.dtype) * span
        radius = (torch.norm(span) / 2.0) * severity
        d2 = ((local - c) ** 2).sum(dim=-1)
        keep = d2 > radius * radius  # 球内被遮，球外保留
    elif mode == "random_sphere":
        if center_local is None:
            raise ValueError("random_sphere requires center_local (B,3) tensor")
        local = _base_to_object_local(pts_base, obj_pos_b, obj_quat_b)
        c = center_local.to(pts_base.device, dtype=pts_base.dtype).unsqueeze(1)  # (B,1,3)
        radius = (torch.norm(span) / 2.0) * severity
        d2 = ((local - c) ** 2).sum(dim=-1)
        keep = d2 > radius * radius  # 球内被遮，球外保留
    elif mode == "random_box":
        if center_local is None or shape_quat_local is None:
            raise ValueError(
                "random_box requires center_local (B,3) and shape_quat_local (B,4)"
            )
        local = _base_to_object_local(pts_base, obj_pos_b, obj_quat_b)
        c = center_local.to(pts_base.device, dtype=pts_base.dtype).unsqueeze(1)  # (B,1,3)
        q = shape_quat_local.to(pts_base.device, dtype=pts_base.dtype).unsqueeze(1)
        q = q.expand(local.shape[0], local.shape[1], 4)          # (B, P, 4)
        p_box = quat_apply_inverse(q, local - c)                 # (B, P, 3)，椭球自身系
        # 随机各向异性度量：主轴方向 = 随机 q（SO(3)），轴长 = 工件局部包围盒 span。
        # 按该度量删掉离盒心最近的 severity 比例候选点 —— 比例严格精确（无需搜索）、
        # 删掉的是一块连通"阴影"、方向随机；盒心来自相机侧表面点，阴影必在可见面。
        d2 = ((p_box / span) ** 2).sum(dim=-1)                   # (B, P)
        num_cand = local.shape[1]
        # 至少留 1 个存活点：否则上层 mask_depth_to_pointcloud 的 fallback
        # （first_kept = argmax(keep)）会顶替到一个被删点上。
        k = min(max(int(round(severity * num_cand)), 0), num_cand - 1)
        keep = torch.ones(local.shape[0], num_cand, dtype=torch.bool, device=pts_base.device)
        if k > 0:
            _, del_idx = torch.topk(d2, k, dim=1, largest=False)  # (B, k) 度量下最近 k 个
            keep.scatter_(1, del_idx, torch.zeros_like(del_idx, dtype=torch.bool))
    else:
        raise ValueError(
            f"unknown occlusion mode {mode!r}; expected "
            "none/halfspace/sphere/random_sphere/random_box"
        )

    return keep


def parse_occlusion_center(s: str) -> tuple[float, float, float]:
    """解析 'cx,cy,cz' 字符串为三元组（归一化分数，各分量在 [0,1]）。"""
    parts = [float(x) for x in s.split(",")]
    if len(parts) != 3:
        raise ValueError(f"center 需要 3 个逗号分隔的数，got {s!r}")
    if any(not (0.0 <= x <= 1.0) for x in parts):
        raise ValueError(f"center 分量应在 [0,1] 内，got {parts}")
    return (parts[0], parts[1], parts[2])
