#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""用 YOLO-seg 替掉 Isaac 的实例分割，作为点云管线的掩码来源（规格 §11.1/§11.2）。

**为什么要有这个模块**

规格 §1.1 只允许存在**一份** ``mask_depth_to_pointcloud``：仿真和真机必须调同一个
函数，否则两边的反投影会悄悄长歪。所以"换分割"能动的只有**掩码怎么来**这一件事，
反投影、FPS、噪声、零阶保持全都不许碰。本模块就只负责那一件事::

    仿真原路线   instance_id_segmentation_fast ─┐
    本模块       RGB ─► YOLO-seg ──────────────┴─► mask ─► mask_depth_to_pointcloud

把 mask 换成 YOLO 出的那张，下游一个字都不用改 —— 这正是 §1.1 要的效果。

**训练时就用 YOLO，图什么**

真机侧掩码只能是 YOLO 出的。若训练全程吃 Isaac 的完美分割，策略见到的掩码分布和
部署时差一截：YOLO 的边缘会胖一圈、遮挡处会缺一块，而且这是**系统性偏差不是噪声**
（规格 §11.2）—— 加零均值高斯噪声补不回来。让训练直接吃 YOLO 的掩码，这段 gap 就
从根上不存在了。代价是每步要跑一次网络，见下面的开销一节。

**RGB 进、分割出：这条路线可以彻底关掉分割那一路**

原 stage 2 相机开 ``distance_to_image_plane`` + ``instance_id_segmentation_fast``。
换成 YOLO 之后分割图**没有任何消费者**了，可以整路摘掉，改成深度 + RGB。两路换两路，
相机显存持平；而且省下的是分割那一路的渲染开销（AOV 渲染并不比 RGB 便宜）。
``colorize_instance_id_segmentation`` 之类的坑也一并不用管了。

**为什么不做 CPU 往返**

``model.predict()` 直接吃 GPU 上的 ``(B,3,H,W)`` float 张量，masks 也在 GPU 上按
原分辨率返回 —— 实测 ``masks.data.device == cuda:0``、``shape == (n,480,640)``。
所以 Isaac 相机缓冲 → YOLO → 掩码 → 反投影全程不下显存。中间只要一次
``permute`` + ``/255``，没有 numpy、没有 ``.cpu()``。

**开销（RTX 5090 实测，20260815.pt，imgsz=640）**

    B=8   43.5ms   5.43ms/env
    B=16  70.1ms   4.38ms/env
    B=32 130.4ms   4.07ms/env
    B=64 273.8ms   4.28ms/env

**几乎不随 batch 摊薄** —— 瓶颈是 ultralytics 逐图的 Python 后处理（掩码上采样、
Results 对象构造），不是网络本身。而仿真一步只要 20ms，所以 64 env 会慢十几倍。
结论：这条路线**适合小 num_envs 的微调/验证，不适合从头训**。建议 8~16 env，
从 stage 2 的 checkpoint 续训。

**去重：end2end 头会重复检出**

YOLO26 是 NMS-free 架构（``head.end2end=True``），``predict(iou=...)`` 完全无效。
一对一匹配没收敛透时同一个目标会出两个近乎重合的掩码（实测 mask IoU 0.995）。
560 帧里 13 帧如此。单类单目标场景下按 **conf 最高**取一个即可 —— 实测取 max_conf
与取并集的质心差 <1mm，不值得为此写合并逻辑。

**批内对齐：第 i 张图的检出必须回到第 i 个 env**

``predict`` 返回的 ``Results`` 列表与输入 batch 顺序一一对应，直接按下标取。检出
为空的那些 env 掩码留全 False，交给下游的零阶保持处理（规格 §11.1 明确要求"绝不
给策略送空点集"）。

**ultralytics 装在哪：独立目录，不碰 isaaclab 环境**

ultralytics 只在 sam2anno 环境里有（torch 2.12），训练在 isaaclab（torch 2.8）。
但**不需要**把它装进 isaaclab —— 装进一个独立目录再挂 ``sys.path`` 即可::

    ~/miniconda3/envs/isaaclab/bin/python -m pip install --no-deps \\
        --target /home/gxai/Desktop/CZR/.yolo_deps \\
        ultralytics==8.4.69 polars nvidia-ml-py ultralytics-thop

共 21MB、4 个**纯 Python** 包（无编译扩展，不存在 ABI 问题），isaaclab 的
site-packages 一个文件都不改，删掉目录就完全复原。实测挂上后 torch 2.8.0 /
numpy 2.4.6 / scipy 1.11.4 全部原样，YOLO 推理正常。

``--no-deps`` 是关键：不加它 pip 会顺手把 Isaac 的 scipy 1.11.4 换成 1.17.1
（torch 倒是不会动，它依赖里只写了 ``torch>=1.8``，2.8 已满足）。

目录位置可用环境变量 ``YOLO_DEPS_DIR`` 覆盖。若你更愿意直接装进 isaaclab 环境，
那样也能工作 —— ``_ensure_deps_on_path`` 把独立目录追加在 ``sys.path`` **末尾**，
环境自带的优先。

**为什么不用 PYTHONPATH 直接借 sam2anno 那份**

试过，不行：sam2anno 是 py3.10，它的 numpy/cv2 是编译扩展，在 py3.11 解释器里
``import numpy`` 直接报错。只有纯 Python 包能跨小版本借用。
"""

from __future__ import annotations

import os
import sys

import torch

# 权重默认值。20260815.pt 相比旧的 best.pt：560 帧仿真图上漏检 1→0、重复 18→13，
# 且遮挡处理是对的 —— 夹爪压在工件中间时旧模型只分割出一侧半块（把遮挡物当成了
# 物体边界），新模型返回完整工件并在夹爪处挖缺口。半块工件反投影出的点云质心会朝
# 一侧偏出半个工件长度，这是比边缘胖瘦严重得多的系统性错误。
# 权重路径软定位（不写死绝对路径）：环境变量 ``YOLO_WEIGHTS`` 优先，其次仓库内
# ``tools/20260815.pt``，最后退回旧机器的绝对路径兜底。换 checkout 目录 / 换机器
# 都不用改代码；三者都不存在时返回 tools 下的默认位置，让 ``get_yolo`` 报清晰错误。
_TOOLS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "tools")
)


def _resolve_yolo_weights() -> str:
    """按 环境变量 → 仓库 tools → 旧机器兜底 的顺序解析 YOLO 权重路径。"""
    for candidate in (
        os.environ.get("YOLO_WEIGHTS"),
        os.path.join(_TOOLS_DIR, "20260815.pt"),
        "/home/gxai/Desktop/CZR/20260815.pt",
    ):
        if candidate and os.path.exists(candidate):
            return candidate
    return os.path.join(_TOOLS_DIR, "20260815.pt")


DEFAULT_YOLO_WEIGHTS = _resolve_yolo_weights()

# 与训练一致，别改。模型在 640 上训的，换尺寸会掉点。
YOLO_IMGSZ = 640

# 低于它不算检出。与 test_yolo_seg.py / 真机侧保持同一门限。
YOLO_CONF = 0.25

# stage 4 默认的 YOLO 运行间隔：每 N 步跑一次，中间步复用缓存掩码配当前深度帧。
# 5 的依据是抓取前工件静止、掩码在连续几帧内变化在厘米量级以内，小于 FPS 下采样
# 与 NOISE_STD_M 引入的不确定性；同时把 YOLO 的均摊开销压到 1/5。
DEFAULT_YOLO_STEP = 5

_MODEL_CACHE: dict = {}

# ultralytics 装在**独立目录**里，不进 isaaclab 环境本体（21MB，4 个纯 Python 包）。
# 这样 isaaclab 的 site-packages 一个文件都不改，删掉这个目录就完全复原。
# 目录不存在时不报错 —— 也许用户就是直接装进环境里了，那样同样能 import。
_YOLO_DEPS_DIR = os.environ.get(
    "YOLO_DEPS_DIR", "/home/gxai/Desktop/CZR/.yolo_deps"
)


def _ensure_deps_on_path() -> None:
    """把独立依赖目录挂到 sys.path 末尾。

    追加到**末尾**而不是开头：万一 isaaclab 环境里已经有同名包，优先用环境自己的，
    这个目录只作兜底。这样两种安装方式都能工作，且不会互相顶掉。
    """
    if os.path.isdir(_YOLO_DEPS_DIR) and _YOLO_DEPS_DIR not in sys.path:
        sys.path.append(_YOLO_DEPS_DIR)


def get_yolo(weights: str = DEFAULT_YOLO_WEIGHTS, device: str = "cuda:0"):
    """加载并缓存 YOLO。同一权重只加载一次 —— 每步重新构造要几百毫秒。"""
    key = (weights, device)
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached

    if not os.path.exists(weights):
        raise FileNotFoundError(f"找不到 YOLO 权重: {weights}")
    _ensure_deps_on_path()
    try:
        from ultralytics import YOLO
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            f"import ultralytics 失败（已尝试 {_YOLO_DEPS_DIR}）。\n"
            "推荐装法 —— 装进独立目录，**不碰 isaaclab 环境本体**（21MB，删目录即复原）：\n"
            f"  ~/miniconda3/envs/isaaclab/bin/python -m pip install --no-deps \\\n"
            f"      --target {_YOLO_DEPS_DIR} \\\n"
            "      ultralytics==8.4.69 polars nvidia-ml-py ultralytics-thop\n"
            "（--no-deps 是关键：不加它 pip 会把 Isaac 的 scipy 1.11.4 换成 1.17.1。）"
        ) from exc

    model = YOLO(weights)
    # 预热：第一次推理要建 CUDA graph / 分配工作区，几百毫秒。放在这里而不是让
    # 它落在第一个训练步上，免得第一步的耗时统计和后面对不上。
    dummy = torch.zeros(1, 3, 480, 640, device=device)
    model.predict(dummy, imgsz=YOLO_IMGSZ, conf=YOLO_CONF, device=device, verbose=False)

    head = model.model.model[-1]
    print(
        f"[YOLO] {os.path.basename(weights)}  head={type(head).__name__}  "
        f"end2end={getattr(head, 'end2end', False)}  names={model.names}"
    )
    _MODEL_CACHE[key] = model
    return model


def rgb_to_yolo_input(rgb: torch.Tensor) -> torch.Tensor:
    """Isaac 相机 RGB → YOLO 要的 ``(B,3,H,W)`` float 0~1，全程留在 GPU 上。

    Isaac 的 ``camera.data.output["rgb"]`` 是 ``(B,H,W,3或4)``：带 alpha 时要切掉，
    dtype 可能是 uint8 也可能是 0~1 的 float。用最大值判断量纲 —— 直接除 255 会把
    本来就是 0~1 的 float 打成全黑，而全黑图 YOLO 什么都检不出来，且不报错。
    """
    x = rgb
    if x.shape[-1] == 4:
        x = x[..., :3]
    x = x.permute(0, 3, 1, 2).contiguous().float()
    if x.numel() and float(x.max()) > 1.5:
        x = x / 255.0
    return x.clamp_(0.0, 1.0)


@torch.no_grad()
def yolo_masks(
    rgb: torch.Tensor,
    weights: str = DEFAULT_YOLO_WEIGHTS,
    conf: float = YOLO_CONF,
    imgsz: int = YOLO_IMGSZ,
    return_details: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """一批 RGB → 每个 env 一张 bool 掩码。

    返回 ``(mask (B,H,W) bool, detected (B,) bool)``。``detected`` 为 False 的
    env 掩码全 False —— 不在这里做兜底，交给 ``point_bridge_point_cloud`` 里既有的
    零阶保持逻辑，那是规格 §11.1 规定的位置，两处都做会互相掩盖故障。

    单类单目标：每张图按 conf 最高取**一个**掩码。end2end 头的重复检出（同一目标
    两个近乎重合的掩码）就是靠这一步去掉的。

    ``return_details=True`` 时额外返回第三项 ``details``：
    ``{"best_conf": (B,) float tensor, "n_det": (B,) int tensor}``。真机部署的确认
    界面与日志要显示 conf 和检出数，而那些量本来就在这一次 ``predict`` 的
    ``Results`` 里 —— 让调用方再跑一次推理只为读 conf 会把延迟翻倍，而延迟在 50Hz
    控制环里是硬约束。默认 False，训练路径的返回签名和行为一个字都不变。
    """
    device = rgb.device
    x = rgb_to_yolo_input(rgb)
    num_envs, _, height, width = x.shape

    model = get_yolo(weights, device=str(device))
    results = model.predict(
        x, imgsz=imgsz, conf=conf, device=str(device), verbose=False
    )

    mask = torch.zeros(num_envs, height, width, dtype=torch.bool, device=device)
    detected = torch.zeros(num_envs, dtype=torch.bool, device=device)
    best_conf = torch.zeros(num_envs, dtype=torch.float32, device=device)
    n_det = torch.zeros(num_envs, dtype=torch.int32, device=device)

    # Results 列表与输入 batch 严格同序，按下标回填即可。
    for i, res in enumerate(results):
        if res.masks is None or len(res.masks) == 0:
            continue
        data = res.masks.data  # (n, h, w)，已在 GPU 上
        confs = res.boxes.conf
        best = int(torch.argmax(confs))
        n_det[i] = len(data)
        best_conf[i] = confs[best]
        m = data[best]
        if m.shape != (height, width):
            # 掩码分辨率与原图不一致时缩回去。用 nearest —— 双线性会在边缘造出
            # 0~1 的中间值，阈值化后掩码边界发生亚像素漂移。
            m = torch.nn.functional.interpolate(
                m[None, None].float(), size=(height, width), mode="nearest"
            )[0, 0]
        mask[i] = m > 0.5
        detected[i] = bool(mask[i].any())

    # PhysX GPU 与 PyTorch 共用同一块 GPU。ultralytics predict() 在自己的 CUDA
    # stream 上提交操作，若不显式同步直接返回，PhysX 下一步启动 kernel 时这些操作
    # 可能还未执行完，导致 "GPU compressContactStage1 fail to launch kernel" /
    # "Scene state is corrupted" 错误。synchronize 确保 YOLO 的所有 GPU 操作在
    # 本函数返回前完全结束，PhysX 接管 GPU 时状态干净。
    torch.cuda.synchronize(device)

    if return_details:
        return mask, detected, {"best_conf": best_conf, "n_det": n_det}
    return mask, detected
