# Copyright (c) 2024, Isaac Lab Project
# SPDX-License-Identifier: BSD-3-Clause

"""Point-BERT 独立推理模块（无需 knn_cuda / pointnet2_ops）

从 Point-BERT.pth 加载 transformer_q（MaskTransformer 查询编码器），
提供单一接口：
    model = get_pointbert_model(ckpt_path, device)
    feat  = model(xyz)   # xyz: (B, N, 3) → feat: (B, 384)

架构（与 Point-BERT 原论文完全对齐，全 PyTorch 实现）：
  输入   : (B, N_pts, 3)         — N_pts=1024 的点云
  分组   : FPS(num_group=64) + KNN(group_size=32) → (B, 64, 32, 3)
  编码   : PointNet Conv1d(3→128→256 → max → 512→256) per group → (B, 64, 256)
  映射   : reduce_dim  Linear(256→384) → (B, 64, 384)
  位置嵌 : MLP(3→128→384) on center coordinates
  变换器 : 12×PreNorm-Attention + FFN (d=384, heads=6, mlp_ratio=4)
  输出   : LayerNorm(CLS token) → (B, 384)

加载键值对应（checkpoint base_model.transformer_q.* → model.*）：
  encoder.*      ←→ encoder.*
  reduce_dim.*   ←→ reduce_dim.*
  cls_token      ←→ cls_token
  cls_pos        ←→ cls_pos
  mask_token     ←→ mask_token
  pos_embed.*    ←→ pos_embed.*
  blocks.blocks  ←→ blocks.blocks  （TransformerEncoder 内的 ModuleList）
  norm.*         ←→ norm.*
"""

from __future__ import annotations

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

# ──────────────────────────────────────────────────────────────────────────────
# 纯 PyTorch FPS + KNN（无外部依赖）
# ──────────────────────────────────────────────────────────────────────────────

def _fps(xyz: torch.Tensor, npoint: int) -> torch.Tensor:
    """Farthest Point Sampling.

    Args:
        xyz    : (B, N, 3)
        npoint : 采样点数 G

    Returns:
        centers: (B, G, 3)
    """
    B, N, _ = xyz.shape
    device   = xyz.device

    centers  = torch.zeros(B, npoint, 3, device=device, dtype=xyz.dtype)
    dist     = torch.full((B, N), float("inf"), device=device, dtype=xyz.dtype)
    farthest = torch.randint(0, N, (B,), device=device)

    batch_idx = torch.arange(B, device=device)
    for i in range(npoint):
        sel             = xyz[batch_idx, farthest]   # (B, 3)
        centers[:, i]   = sel
        d               = ((xyz - sel.unsqueeze(1)) ** 2).sum(-1)  # (B, N)
        dist            = torch.min(dist, d)
        farthest        = dist.argmax(-1)             # (B,)

    return centers


def _knn_query(xyz: torch.Tensor, centers: torch.Tensor, k: int) -> torch.Tensor:
    """K-Nearest Neighbors（纯 PyTorch topk）。

    Args:
        xyz    : (B, N, 3)  — all points
        centers: (B, G, 3)  — query centers
        k      : int

    Returns:
        idx : (B, G, k)  — indices into xyz
    """
    # squared distance (B, G, N)
    diff  = centers.unsqueeze(2) - xyz.unsqueeze(1)   # (B, G, N, 3)
    sq    = (diff ** 2).sum(-1)                        # (B, G, N)
    _, idx = sq.topk(k, dim=-1, largest=False)         # (B, G, k)
    return idx


# ──────────────────────────────────────────────────────────────────────────────
# 点云分组
# ──────────────────────────────────────────────────────────────────────────────

class _Group(nn.Module):
    """FPS + KNN 点云分组，等价于 dvae.Group，无外部依赖。
    全程 torch.gather 操作，张量留在 GPU，无 Python for-loop 收集邻域。
    """

    def __init__(self, num_group: int, group_size: int):
        super().__init__()
        self.num_group  = num_group
        self.group_size = group_size

    def forward(self, xyz: torch.Tensor):
        """
        Args:
            xyz: (B, N, 3)
        Returns:
            neighborhood: (B, G, K, 3)  — 局部坐标（中心已减去）
            centers:      (B, G, 3)
        """
        B, N, _ = xyz.shape
        G, K     = self.num_group, self.group_size

        centers  = _fps(xyz, G)                                 # (B, G, 3)
        idx      = _knn_query(xyz, centers, K)                  # (B, G, K)

        # 向量化收集邻域点：torch.gather，无 Python 循环
        idx_flat = idx.reshape(B, G * K)                        # (B, G*K)
        # gather: 对 dim=1（N 维）按 idx_flat 取点
        nbhd = xyz.gather(
            1,
            idx_flat.unsqueeze(-1).expand(-1, -1, 3)            # (B, G*K, 3)
        ).reshape(B, G, K, 3)

        # 局部坐标：减去中心
        nbhd = nbhd - centers.unsqueeze(2)                      # (B, G, K, 3)
        return nbhd, centers


# ──────────────────────────────────────────────────────────────────────────────
# 局部补丁编码器（PointNet 风格）
# ──────────────────────────────────────────────────────────────────────────────

class _Encoder(nn.Module):
    """与 dvae.Encoder 完全相同，用于匹配 checkpoint encoder.* 键。"""

    def __init__(self, encoder_channel: int = 256):
        super().__init__()
        self.encoder_channel = encoder_channel
        self.first_conv = nn.Sequential(
            nn.Conv1d(3, 128, 1),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Conv1d(128, 256, 1),
        )
        self.second_conv = nn.Sequential(
            nn.Conv1d(512, 512, 1),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Conv1d(512, encoder_channel, 1),
        )

    def forward(self, point_groups: torch.Tensor) -> torch.Tensor:
        """
        Args:
            point_groups: (B, G, K, 3)
        Returns:
            tokens: (B, G, encoder_channel)
        """
        B, G, K, _ = point_groups.shape
        x  = point_groups.reshape(B * G, K, 3).permute(0, 2, 1)   # (BG, 3, K)
        f  = self.first_conv(x)                                     # (BG, 256, K)
        fg = f.max(dim=2, keepdim=True)[0].expand(-1, -1, K)       # (BG, 256, K)
        f  = self.second_conv(torch.cat([fg, f], dim=1))            # (BG, C, K)
        return f.max(dim=2)[0].reshape(B, G, self.encoder_channel)  # (B, G, C)


# ──────────────────────────────────────────────────────────────────────────────
# Transformer 组件
# ──────────────────────────────────────────────────────────────────────────────

class _Attention(nn.Module):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = False):
        super().__init__()
        self.num_heads = num_heads
        self.scale     = (dim // num_heads) ** -0.5
        self.qkv       = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.proj      = nn.Linear(dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        return self.proj(x)


class _Mlp(nn.Module):
    def __init__(self, dim: int, mlp_ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * mlp_ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class _Block(nn.Module):
    """与 Point_BERT.Block 等价（drop_path=0，用 nn.Identity 替代 DropPath）。"""

    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.norm1     = nn.LayerNorm(dim)
        self.attn      = _Attention(dim, num_heads, qkv_bias=False)
        self.norm2     = nn.LayerNorm(dim)
        self.mlp       = _Mlp(dim, mlp_ratio)
        self.drop_path = nn.Identity()   # 推理时无需随机深度

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class _TransformerEncoder(nn.Module):
    """与 Point_BERT.TransformerEncoder 等价，键名为 blocks.blocks.*。"""

    def __init__(self, embed_dim: int, depth: int, num_heads: int):
        super().__init__()
        self.blocks = nn.ModuleList([
            _Block(embed_dim, num_heads) for _ in range(depth)
        ])

    def forward(self, x: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
        for blk in self.blocks:
            x = blk(x + pos)
        return x


# ──────────────────────────────────────────────────────────────────────────────
# 完整推理模块（对应 checkpoint 的 transformer_q）
# ──────────────────────────────────────────────────────────────────────────────

class _MaskTransformerInference(nn.Module):
    """MaskTransformer 推理子集，输出原始 CLS 特征 (B, trans_dim)。

    模块名称与 checkpoint transformer_q.* 完全对应，
    支持 strict=False 加载（忽略 cls_head / lm_head / mask_token 等预训练专属权重）。
    """

    def __init__(
        self,
        trans_dim:    int = 384,
        depth:        int = 12,
        num_heads:    int = 6,
        encoder_dims: int = 256,
        cls_dim:      int = 512,
        num_tokens:   int = 8192,
    ):
        super().__init__()

        # ── 局部补丁编码器 ──────────────────────────────────────────────────
        self.encoder    = _Encoder(encoder_channel=encoder_dims)
        self.reduce_dim = nn.Linear(encoder_dims, trans_dim)

        # ── Learnable tokens（保留以匹配 checkpoint 键）──────────────────
        self.cls_token  = nn.Parameter(torch.zeros(1, 1, trans_dim))
        self.cls_pos    = nn.Parameter(torch.zeros(1, 1, trans_dim))
        self.mask_token = nn.Parameter(torch.zeros(1, 1, trans_dim))

        # ── 位置嵌入 MLP (3→128→trans_dim) ─────────────────────────────
        self.pos_embed = nn.Sequential(
            nn.Linear(3, 128),
            nn.GELU(),
            nn.Linear(128, trans_dim),
        )

        # ── Transformer ──────────────────────────────────────────────────
        self.blocks = _TransformerEncoder(trans_dim, depth, num_heads)
        self.norm   = nn.LayerNorm(trans_dim)

        # ── 预训练专属头（仅为加载权重，推理不使用）────────────────────
        self.cls_head = nn.Sequential(
            nn.Linear(trans_dim, cls_dim),
            nn.GELU(),
            nn.Linear(cls_dim, cls_dim),
        )
        self.lm_head = nn.Linear(trans_dim, num_tokens)

    def forward(
        self,
        neighborhood: torch.Tensor,
        center:       torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            neighborhood: (B, G, K, 3)  — 局部坐标
            center:       (B, G, 3)     — 全局坐标
        Returns:
            cls_feat: (B, trans_dim=384)
        """
        B = neighborhood.shape[0]

        tokens  = self.reduce_dim(self.encoder(neighborhood))        # (B, G, D)
        pos     = self.pos_embed(center)                              # (B, G, D)

        cls_tok = self.cls_token.expand(B, -1, -1)                   # (B, 1, D)
        cls_pos = self.cls_pos.expand(B, -1, -1)                     # (B, 1, D)

        x   = torch.cat([cls_tok, tokens], dim=1)                    # (B, G+1, D)
        pos = torch.cat([cls_pos, pos], dim=1)                       # (B, G+1, D)

        x = self.blocks(x, pos)
        x = self.norm(x)
        return x[:, 0]   # CLS token → (B, 384)


# ──────────────────────────────────────────────────────────────────────────────
# 顶层包装器（对应 checkpoint 的 base_model）
# ──────────────────────────────────────────────────────────────────────────────

class PointBERT(nn.Module):
    """公开接口：输入原始点云，输出 (B, 384) CLS 特征。

    调用示例：
        model = PointBERT()
        model.load_from_checkpoint('/home/gxai/IsaacLab/Point_BERT/Point-BERT.pth', 'cuda')
        feat = model(xyz)   # xyz: (B, N, 3)
    """

    NUM_GROUP  = 64    # FPS 采样中心数 G
    GROUP_SIZE = 32    # 每组 KNN 点数 K
    NUM_POINTS = 1024  # 标准输入点数

    def __init__(self):
        super().__init__()
        self.group_divider = _Group(self.NUM_GROUP, self.GROUP_SIZE)
        self.transformer_q = _MaskTransformerInference()

    def load_from_checkpoint(self, ckpt_path: str, device: str = "cpu"):
        """从 Point-BERT.pth 加载 transformer_q 权重。"""
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        bm   = ckpt["base_model"]

        # 提取 transformer_q.* 子集，去掉前缀
        tq_state = {
            k[len("transformer_q."):]: v
            for k, v in bm.items()
            if k.startswith("transformer_q.")
        }

        result = self.transformer_q.load_state_dict(tq_state, strict=False)
        missing_critical = [
            k for k in result.missing_keys
            if not any(tag in k for tag in ("cls_head", "lm_head", "mask_token", "drop_path"))
        ]
        if missing_critical:
            print(f"[PointBERT] 警告：关键权重缺失 → {missing_critical}")
        else:
            print(f"[PointBERT] 权重加载成功（跳过预训练专属头）")

        self.to(device)
        self.eval()
        for p in self.parameters():
            p.requires_grad_(False)
        return self

    @torch.no_grad()
    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        """
        Args:
            xyz: (B, N, 3)  — 点云（相机坐标系，单位 m）
        Returns:
            feat: (B, 384)
        """
        neighborhood, center = self.group_divider(xyz)
        return self.transformer_q(neighborhood, center)


# ──────────────────────────────────────────────────────────────────────────────
# 单例缓存
# ──────────────────────────────────────────────────────────────────────────────

_pointbert_cache: dict = {}


def get_pointbert_model(ckpt_path: str, device: str) -> PointBERT:
    """懒加载 Point-BERT 单例，避免重复加载（线程不安全，单进程 Isaac Lab 场景下够用）。"""
    key = (ckpt_path, device)
    if key not in _pointbert_cache:
        model = PointBERT().load_from_checkpoint(ckpt_path, device)
        _pointbert_cache[key] = model
    return _pointbert_cache[key]


# ──────────────────────────────────────────────────────────────────────────────
# 深度图 → 点云工具函数
# ──────────────────────────────────────────────────────────────────────────────

def depth_to_pointcloud(
    depth: torch.Tensor,
    fx: float,
    fy: float,
    cx: float,
    cy: float,
) -> torch.Tensor:
    """将 distance_to_camera 深度图反投影为点云（相机坐标系）。

    IsaacLab 的 distance_to_camera 是欧氏距离（非 z 分量），
    需先换算为 z 深度再反投影。

    Args:
        depth : (B, H, W)  — distance_to_camera（单位 m）
        fx, fy, cx, cy    — 相机内参（像素单位，224×224 分辨率下）

    Returns:
        pts : (B, H*W, 3)  — 点云 (x, y, z)（相机坐标系）
    """
    B, H, W = depth.shape
    device   = depth.device

    u = torch.arange(W, device=device, dtype=torch.float32)
    v = torch.arange(H, device=device, dtype=torch.float32)
    uu, vv = torch.meshgrid(u, v, indexing="xy")              # (H, W) each

    # 归一化方向
    u_n = (uu - cx) / fx   # (H, W)
    v_n = (vv - cy) / fy   # (H, W)

    # distance_to_camera → z 深度
    scale = torch.sqrt(1.0 + u_n ** 2 + v_n ** 2)            # (H, W)
    z = depth / scale.unsqueeze(0)                             # (B, H, W)
    x = u_n.unsqueeze(0) * z                                   # (B, H, W)
    y = v_n.unsqueeze(0) * z                                   # (B, H, W)

    pts = torch.stack([x, y, z], dim=-1)                       # (B, H, W, 3)
    return pts.reshape(B, H * W, 3)


def sample_to_fixed_points(
    pts_all:    torch.Tensor,
    depth_hw:   torch.Tensor,
    num_points: int,
    depth_min:  float = 0.01,
    depth_max:  float = 2.0,
) -> torch.Tensor:
    """对每个 batch 条目独立采样到固定数量点（全向量化，无 Python for-loop）。

    策略：
      valid 点得分 ∈ [1, 2)，invalid 得分 ∈ [0, 1)
      按得分降序排列 → valid 点全部排在前面（内部随机）
      用 modulo 索引实现：M≥num_points 时随机取子集，M<num_points 时循环补齐

    Args:
        pts_all   : (B, H*W, 3)   — 所有点（含无效点）
        depth_hw  : (B, H, W)     — 原始深度图（用于有效性判断）
        num_points: int
        depth_min : 最小有效深度 (m)
        depth_max : 最大有效深度 (m)

    Returns:
        pts : (B, num_points, 3)
    """
    B, NHW, _ = pts_all.shape
    device    = pts_all.device

    # ── 有效掩码 ───────────────────────────────────────────────────────────
    valid = (
        (depth_hw > depth_min) & (depth_hw < depth_max)
    ).reshape(B, NHW).float()                                  # (B, NHW) float

    # ── 随机打分：valid ∈ [1,2)，invalid ∈ [0,1) ──────────────────────────
    rand  = torch.rand(B, NHW, device=device)
    score = rand + valid                                        # valid 分数 > invalid

    # 按降序排列：valid 点全部排在前面，内部顺序随机 → (B, NHW)
    order = score.argsort(dim=1, descending=True)

    # ── 每项有效点数（至少为 1，防止 div-by-zero）──────────────────────────
    n_valid = valid.long().sum(dim=1).clamp(min=1)             # (B,)

    # ── modulo 索引：超过 n_valid 的位置循环回头 ──────────────────────────
    sample_range = torch.arange(num_points, device=device).unsqueeze(0)  # (1, N)
    take_pos = sample_range % n_valid.unsqueeze(1)             # (B, num_points)

    # 用 take_pos 在排序后的 order 里取最终点索引
    point_idx = order.gather(1, take_pos)                      # (B, num_points)

    # ── gather 真实坐标 ────────────────────────────────────────────────────
    result = pts_all.gather(
        1,
        point_idx.unsqueeze(-1).expand(-1, -1, 3)              # (B, num_points, 3)
    )
    return result
