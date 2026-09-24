#!/usr/bin/env python3
"""FoundationPose + RealSense 实时位姿估计入口。"""

import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SCRIPTS_DIR = os.path.dirname(SCRIPT_DIR)
PROJECT_DIR = os.path.dirname(SCRIPTS_DIR)
FOUNDATIONPOSE_DIR = os.environ.get(
    "FOUNDATIONPOSE_DIR",
    os.path.join(PROJECT_DIR, "third_party", "gx-foundationpose"),
)
FOUNDATIONPOSE_REALTIME_DIR = os.path.join(FOUNDATIONPOSE_DIR, "realtime")

# Keep scripts/camera first so the local RealSense wrapper wins, then add the
# FoundationPose repo for estimater.py/Utils.py and realtime/roi_selector.py.
for path in (SCRIPTS_DIR, FOUNDATIONPOSE_DIR, FOUNDATIONPOSE_REALTIME_DIR):
    if os.path.isdir(path) and path not in sys.path:
        sys.path.append(path)

import argparse
import cv2
import numpy as np
import time
from datetime import datetime
import trimesh
import json
import socket

from estimater import FoundationPose, ScorePredictor, PoseRefinePredictor
from Utils import draw_posed_3d_box, draw_xyz_axis, set_logging_format, set_seed
from realsense_camera import RealsenseCamera
from roi_selector import ROISelector

try:
    import nvdiffrast.torch as dr
except ImportError:
    print("[错误] 请安装 nvdiffrast")
    sys.exit(1)

from multiprocessing import shared_memory

MULTI_POSE_SHM_NAME = "foundationpose_multi_pose"


class RealtimePoseEstimator:
    def __init__(self, mesh_file, est_refine_iter=5, track_refine_iter=2, debug=1, debug_dir=None, use_original_origin=False):
        """
        初始化实时位姿估计器 - 支持单个工件
        Args:
            use_original_origin: 是否使用obj文件的原始坐标系原点（默认False，使用几何中心）
        """
        self.mesh_file = mesh_file
        self.est_refine_iter = est_refine_iter
        self.track_refine_iter = track_refine_iter
        self.debug = debug
        self.use_original_origin = use_original_origin
        
        # 设置 debug 目录
        if debug_dir is None:
            code_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            debug_dir = os.path.join(code_dir, 'debug_realtime')
        self.debug_dir = debug_dir
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # 状态
        self.initialized = False
        self.pose = None  # 相机坐标系下的位姿
        self.frame_count = 0
        
        self._init_estimator()
        
    def _init_estimator(self):
        """初始化 FoundationPose 估计器"""
        print(f"[初始化] 加载模型: {self.mesh_file}")
        self.mesh = trimesh.load(self.mesh_file)
        
        # 计算 bbox 用于可视化
        to_origin, extents = trimesh.bounds.oriented_bounds(self.mesh)
        self.to_origin = to_origin
        self.bbox = np.stack([-extents/2, extents/2], axis=0).reshape(2, 3)
        
        print("[初始化] 加载神经网络...")
        self.scorer = ScorePredictor()
        self.refiner = PoseRefinePredictor()
        self.glctx = dr.RasterizeCudaContext()
        
        self.est = FoundationPose(
            model_pts=self.mesh.vertices,
            model_normals=self.mesh.vertex_normals,
            mesh=self.mesh,
            scorer=self.scorer,
            refiner=self.refiner,
            glctx=self.glctx,
            debug=self.debug,
            debug_dir=self.debug_dir,
            use_original_origin=self.use_original_origin  # 传递参数
        )
        
        if self.use_original_origin:
            print("[初始化] 使用obj文件的原始坐标系原点")
        else:
            print("[初始化] 使用几何中心作为原点")
        
        print("[初始化] FoundationPose 准备完成")

    def register(self, rgb, depth, mask, K):
        """
        首帧注册，初始化位姿
        """
        print("[注册] 正在计算初始位姿...")
        self.pose = self.est.register(
            K=K, 
            rgb=rgb, 
            depth=depth, 
            ob_mask=mask, 
            iteration=self.est_refine_iter
        )
        self.initialized = True
        print("[注册] 初始化完成")
        return self.pose
    
    def track(self, rgb, depth, K):
        """
        跟踪模式，更新位姿
        """
        if not self.initialized:
            raise RuntimeError("请先调用 register() 初始化")
        
        self.pose = self.est.track_one(
            rgb=rgb,
            depth=depth,
            K=K,
            iteration=self.track_refine_iter
        )
        self.frame_count += 1
        return self.pose
    
    def visualize(self, rgb, K, line_color=(0, 255, 0)):
        """
        可视化位姿结果（优化版，带坐标轴）
        """
        if self.pose is None:
            return rgb
        
        # 直接在原图上绘制，避免copy
        center_pose = self.pose @ np.linalg.inv(self.to_origin)
        
        # 绘制 3D bbox
        rgb = draw_posed_3d_box(K, img=rgb, ob_in_cam=center_pose, bbox=self.bbox, line_color=line_color)
        
        # 快速绘制坐标轴（优化版，使用center_pose保持一致）
        rgb = self._draw_xyz_axis_fast(rgb, center_pose, K, scale=0.05, thickness=3)
        
        return rgb
        
        # Alpha混合
        rgb_blended = (rgb * (1 - mask * alpha) + color_rendered * mask * alpha).astype(np.uint8)
        
        return rgb_blended
    
    def _draw_xyz_axis_fast(self, rgb, ob_in_cam, K, scale=0.05, thickness=3):
        """快速绘制XYZ坐标轴（与原版draw_xyz_axis完全一致的坐标系和颜色）"""
        from Utils import project_3d_to_2d
        
        # 定义坐标轴（与原版完全一致）
        origin = np.array([0, 0, 0, 1], dtype=float)
        xx = np.array([1, 0, 0, 1], dtype=float)
        yy = np.array([0, 1, 0, 1], dtype=float)
        zz = np.array([0, 0, 1, 1], dtype=float)
        # 只对前3个元素（xyz坐标）应用scale，保持齐次坐标为1
        xx[:3] = xx[:3] * scale
        yy[:3] = yy[:3] * scale
        zz[:3] = zz[:3] * scale
        
        # 投影到2D
        origin_2d = tuple(project_3d_to_2d(origin, K, ob_in_cam))
        xx_2d = tuple(project_3d_to_2d(xx, K, ob_in_cam))
        yy_2d = tuple(project_3d_to_2d(yy, K, ob_in_cam))
        zz_2d = tuple(project_3d_to_2d(zz, K, ob_in_cam))
        
        # 绘制坐标轴（RGB格式，与原版颜色对应）
        # 原版BGR: X=(0,0,255)红, Y=(0,255,0)绿, Z=(255,0,0)蓝
        # RGB格式: X=(255,0,0)红, Y=(0,255,0)绿, Z=(0,0,255)蓝
        cv2.line(rgb, origin_2d, xx_2d, (255, 0, 0), thickness, cv2.LINE_AA)  # X轴-红色
        cv2.line(rgb, origin_2d, yy_2d, (0, 255, 0), thickness, cv2.LINE_AA)  # Y轴-绿色
        cv2.line(rgb, origin_2d, zz_2d, (0, 0, 255), thickness, cv2.LINE_AA)  # Z轴-蓝色
        
        return rgb


class MultiObjectTracker:
    """多工件跟踪器 — 串行注册 + 原图跟踪 + 碰撞重注册"""
    def __init__(self, mesh_file, est_refine_iter=5, track_refine_iter=2,
                 debug=1, camera_to_base_file=None, use_original_origin=False,
                 udp_host=None, udp_port=5005):
        self.mesh_file = mesh_file
        self.est_refine_iter = est_refine_iter
        self.track_refine_iter = track_refine_iter
        self.debug = debug
        self.use_original_origin = use_original_origin  # 传递给estimator

        # 加载相机到基坐标系的变换
        self.camera_to_base = self.load_camera_to_base_transform(camera_to_base_file)

        # 预加载 mesh 和渲染资源（所有 estimator 共享同一个 mesh）
        self._init_render_resources()

        # 工件估计器列表
        self.estimators = []
        # 每个工件的初始 mask（用于重注册时的 fallback）
        self.initial_masks = []
        self.colors = [
            (0, 255, 0),    # 绿色
            (255, 0, 0),    # 蓝色
            (0, 0, 255),    # 红色
            (255, 255, 0),  # 青色
            (255, 0, 255),  # 品红
            (0, 255, 255),  # 黄色
        ]

        # 共享内存
        self.shm = None
        self.setup_shared_memory()

        # 可选 UDP 发送：给跨机器部署使用，默认关闭，保留原共享内存路径。
        self.udp_addr = None
        self.udp_sock = None
        self.udp_frame_id = 0
        self.last_udp_error_print = 0.0
        if udp_host:
            self.udp_addr = (str(udp_host), int(udp_port))
            self.udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            self.udp_sock.setblocking(False)
            print(f"[UDP] 已启用 FoundationPose 位姿发送: {self.udp_addr[0]}:{self.udp_addr[1]}")

    def _init_render_resources(self):
        """预加载 mesh 渲染资源，用于遮挡消除时渲染精确 mask"""
        from Utils import make_mesh_tensors
        import torch

        mesh = trimesh.load(self.mesh_file)
        # 与 FoundationPose 内部一致：centered mesh
        max_xyz = mesh.vertices.max(axis=0)
        min_xyz = mesh.vertices.min(axis=0)
        self.model_center = (min_xyz + max_xyz) / 2
        self.mesh_centered = mesh.copy()
        self.mesh_centered.vertices = self.mesh_centered.vertices - self.model_center.reshape(1, 3)
        self.mesh_tensors_for_render = make_mesh_tensors(self.mesh_centered, device='cuda')
        self.glctx_render = dr.RasterizeCudaContext()

        # 碰撞检测阈值：工件直径的 30%
        from Utils import compute_mesh_diameter
        self.mesh_diameter = compute_mesh_diameter(model_pts=self.mesh_centered.vertices, n_sample=10000)
        self.collision_threshold = self.mesh_diameter * 0.3
        print(f"[初始化] mesh直径: {self.mesh_diameter:.4f}m, 碰撞阈值: {self.collision_threshold:.4f}m")

    def load_camera_to_base_transform(self, json_file):
        """加载相机到基坐标系的变换矩阵"""
        if json_file is None:
            raise ValueError("必须提供 --camera_calib；不要用单位矩阵跑实机 FoundationPose。")

        json_file = os.path.expanduser(str(json_file))
        if not os.path.exists(json_file):
            raise FileNotFoundError(
                f"相机标定文件不存在: {json_file}. "
                "请确认 --camera_calib 指向 camera_to_right_arm_base.json。"
            )

        with open(json_file, 'r') as f:
            data = json.load(f)

        transform = np.array(data['transform_matrix'], dtype=np.float64)
        if transform.shape != (4, 4) or not np.all(np.isfinite(transform)):
            raise ValueError(f"标定 transform_matrix 非法: shape={transform.shape}")
        print(f"[标定] 已加载相机到基坐标系变换: {json_file}")
        print(transform)
        return transform

    def setup_shared_memory(self):
        """设置共享内存用于ROS通信"""
        try:
            try:
                old_shm = shared_memory.SharedMemory(name=MULTI_POSE_SHM_NAME)
                old_shm.close()
                old_shm.unlink()
            except:
                pass

            size = 4 + 8 + 128 * 10
            self.shm = shared_memory.SharedMemory(
                name=MULTI_POSE_SHM_NAME,
                create=True,
                size=size
            )
            print(f"[共享内存] 已创建: {MULTI_POSE_SHM_NAME}")
        except Exception as e:
            print(f"[警告] 创建共享内存失败: {e}")
            self.shm = None

    # ================================================================
    # 注册阶段：串行注册，每注册一个就渲染擦除，再注册下一个
    # ================================================================

    def add_objects_sequential(self, rgb, depth, masks, K):
        """
        串行注册多个工件。
        注册第1个 → 渲染擦除第1个 → 用擦除后的图像注册第2个 → ...
        这样每个工件的 refiner 只能看到当前工件（前面的已被擦除），
        避免 refiner 把所有位姿拉向最上方的工件。
        """
        rgb_current = rgb.copy()
        depth_current = depth.copy()

        for i, mask in enumerate(masks):
            print(f"[注册] 正在注册工件 {i+1}/{len(masks)}...")
            estimator = RealtimePoseEstimator(
                mesh_file=self.mesh_file,
                est_refine_iter=self.est_refine_iter,
                track_refine_iter=self.track_refine_iter,
                debug=self.debug,
                use_original_origin=self.use_original_origin  # 传递参数
            )
            estimator.register(rgb_current, depth_current, mask, K)
            self.estimators.append(estimator)
            self.initial_masks.append(mask.copy())

            # 渲染擦除当前工件，为下一个工件准备干净的图像
            if i < len(masks) - 1:
                rgb_current, depth_current = self._render_erase(
                    rgb_current, depth_current, estimator, K
                )
                print(f"[注册] 工件 {i+1} 已擦除，准备注册工件 {i+2}")

        print(f"[注册] 全部 {len(masks)} 个工件注册完成")

    # ================================================================
    # 跟踪阶段：用原始图像跟踪 + 碰撞检测 + 自动重注册
    # ================================================================

    def track_all(self, rgb, depth, K, enable_occlusion_removal=True):
        """
        跟踪所有工件。

        策略：
        1. 用原始（未修改的）图像跟踪所有工件
           — track_one 依赖帧间一致性，不能喂修改过的图像
        2. 检测碰撞（两个工件的位姿是否重叠到一起）
        3. 如果碰撞，对碰撞的工件做重注册（用擦除后的图像 + register）
        """
        if not enable_occlusion_removal:
            for estimator in self.estimators:
                estimator.track(rgb, depth, K)
            return

        # Phase 1: 用原始图像跟踪所有工件
        for estimator in self.estimators:
            estimator.track(rgb, depth, K)

        # Phase 2: 碰撞检测 — 找出位姿重叠的工件
        colliding_indices = self._detect_collisions()

        if colliding_indices:
            print(f"[碰撞] 检测到工件位姿重叠: {[i+1 for i in colliding_indices]}, 触发重注册")
            # Phase 3: 对碰撞的工件做重注册
            for idx in colliding_indices:
                self._re_register_object(idx, rgb, depth, K)

    def _detect_collisions(self):
        """
        检测哪些工件的3D位置过于接近（位姿塌缩到同一个工件上）。
        返回需要重注册的工件索引列表（保留编号小的，重注册编号大的）。
        """
        positions = []
        for est in self.estimators:
            if est.pose is not None:
                positions.append(est.pose[:3, 3].copy())
            else:
                positions.append(None)

        colliding = set()
        n = len(positions)
        for i in range(n):
            if positions[i] is None:
                continue
            for j in range(i + 1, n):
                if positions[j] is None:
                    continue
                dist = np.linalg.norm(positions[i] - positions[j])
                if dist < self.collision_threshold:
                    # 保留编号小的（一般是最上方、最可靠的），重注册编号大的
                    colliding.add(j)

        return sorted(colliding)

    def _re_register_object(self, idx, rgb, depth, K):
        """
        对碰撞的工件做重注册：
        1. 渲染擦除所有其他工件
        2. 用擦除后的图像 + 掩码调用 register
        """
        # 渲染擦除所有其他（非碰撞的）工件
        rgb_clean = rgb.copy()
        depth_clean = depth.copy()
        for i, est in enumerate(self.estimators):
            if i != idx and est.pose is not None:
                rgb_clean, depth_clean = self._render_erase(
                    rgb_clean, depth_clean, est, K
                )

        # 生成当前位姿对应的近似 mask
        mask = self._generate_mask_from_pose(self.estimators[idx], K, rgb.shape[:2])

        # 如果生成的 mask 太小，fallback 到初始 mask
        if mask.sum() < 100 and idx < len(self.initial_masks):
            mask = self.initial_masks[idx]

        # 重注册（用较少的迭代次数以保持实时性）
        est = self.estimators[idx]
        est.pose = est.est.register(
            K=K, rgb=rgb_clean, depth=depth_clean,
            ob_mask=mask, iteration=3
        )
        est.initialized = True
        print(f"[重注册] 工件 {idx+1} 重注册完成")

    def _generate_mask_from_pose(self, estimator, K, image_shape):
        """
        从当前位姿渲染 mesh 得到近似 mask（用于重注册时提供给 register）。
        """
        import torch
        from Utils import nvdiffrast_render

        H, W = image_shape
        pose_for_render = estimator.est.pose_last

        if pose_for_render is None:
            # fallback: 全图 mask
            return np.ones((H, W), dtype=np.uint8)

        ob_in_cam = pose_for_render.reshape(1, 4, 4).float().cuda()
        extra = {}
        _, depth_rendered, _ = nvdiffrast_render(
            K=K, H=H, W=W,
            ob_in_cams=ob_in_cam,
            glctx=self.glctx_render,
            mesh_tensors=self.mesh_tensors_for_render,
            get_normal=False,
            use_light=False,
            extra=extra,
        )
        depth_rendered_np = depth_rendered[0].data.cpu().numpy()
        mask = (depth_rendered_np > 0.001).astype(np.uint8)

        # 适当膨胀，给 register 更大的搜索范围
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
        mask = cv2.dilate(mask, kernel, iterations=2)

        return mask

    # ================================================================
    # 渲染擦除：用 nvdiffrast 渲染精确 mesh 轮廓，深度感知擦除
    # ================================================================

    def _render_erase(self, rgb, depth, estimator, K):
        """
        使用 nvdiffrast 渲染 mesh 得到像素级 mask，擦除该工件。

        关键改进：深度感知擦除
        — 只擦除 |渲染深度 - 实际深度| < 容差 的像素
        — 这样不会误擦除被遮挡的其他工件（它们的深度与渲染深度不匹配）
        """
        if estimator.pose is None:
            return rgb, depth

        try:
            import torch
            from Utils import nvdiffrast_render

            H, W = rgb.shape[:2]
            pose_for_render = estimator.est.pose_last

            if pose_for_render is None:
                return rgb, depth

            ob_in_cam = pose_for_render.reshape(1, 4, 4).float().cuda()

            extra = {}
            _, depth_rendered, _ = nvdiffrast_render(
                K=K, H=H, W=W,
                ob_in_cams=ob_in_cam,
                glctx=self.glctx_render,
                mesh_tensors=self.mesh_tensors_for_render,
                get_normal=False,
                use_light=False,
                extra=extra,
            )

            depth_rendered_np = depth_rendered[0].data.cpu().numpy()

            # ---- 深度感知擦除 ----
            # 渲染区域
            render_valid = depth_rendered_np > 0.001
            # 实际深度有效
            depth_valid = depth > 0.001
            # 深度匹配：渲染深度与实际深度接近 → 该像素确实属于这个工件
            # 容差 = 工件直径的 5%（适应不同大小的工件）
            depth_tolerance = max(self.mesh_diameter * 0.05, 0.005)  # 至少 5mm
            depth_match = np.abs(depth_rendered_np - depth) < depth_tolerance

            # 最终擦除 mask：渲染有效 & 深度有效 & 深度匹配
            erase_mask = (render_valid & depth_valid & depth_match).astype(np.uint8)

            if erase_mask.sum() < 10:
                # 如果深度感知方式匹配太少，fallback 到仅用渲染 mask
                erase_mask = render_valid.astype(np.uint8)
                if erase_mask.sum() < 10:
                    return rgb, depth

            # 轻微膨胀覆盖边缘
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            erase_mask_dilated = cv2.dilate(erase_mask, kernel, iterations=1)
            erase_bool = erase_mask_dilated.astype(bool)

            # --- RGB 擦除：用 inpaint 填充 ---
            inpaint_mask = (erase_mask_dilated * 255).astype(np.uint8)
            bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            bgr_inpainted = cv2.inpaint(bgr, inpaint_mask, inpaintRadius=3,
                                         flags=cv2.INPAINT_TELEA)
            rgb_result = cv2.cvtColor(bgr_inpainted, cv2.COLOR_BGR2RGB)

            # --- 深度擦除：设为 0（无效），让后续 register 的 depth 过滤自动忽略 ---
            depth_result = depth.copy()
            depth_result[erase_bool] = 0

            return rgb_result, depth_result

        except Exception as e:
            print(f"[警告] 渲染擦除失败: {e}")
            import traceback
            traceback.print_exc()
            return rgb, depth

    # ================================================================
    # 工具方法
    # ================================================================

    def get_poses_in_base_frame(self):
        """获取所有工件在基坐标系下的位姿"""
        poses_base = []
        for estimator in self.estimators:
            if estimator.pose is not None:
                pose_base = self.camera_to_base @ estimator.pose
                poses_base.append(pose_base)
        return poses_base

    def publish_poses(self):
        """发布位姿到共享内存，并可选通过 UDP 发到另一台机器。"""
        poses_base = self.get_poses_in_base_frame()
        num_objects = len(poses_base)

        if num_objects == 0:
            return

        timestamp_s = time.time()

        if self.shm is not None:
            try:
                offset = 0
                self.shm.buf[offset:offset+4] = np.array([num_objects], dtype=np.int32).tobytes()
                offset += 4
                self.shm.buf[offset:offset+8] = np.array([timestamp_s], dtype=np.float64).tobytes()
                offset += 8
                for pose in poses_base:
                    self.shm.buf[offset:offset+128] = pose.astype(np.float64).tobytes()
                    offset += 128
            except Exception as e:
                print(f"[错误] 发布位姿到共享内存失败: {e}")

        self.publish_poses_udp(poses_base, timestamp_s)

    def publish_poses_udp(self, poses_base, timestamp_s):
        """Send base-frame object poses as one UDP JSON packet."""
        if self.udp_sock is None or self.udp_addr is None:
            return

        try:
            payload = {
                "schema": "foundationpose_multi_pose.v1",
                "timestamp": float(timestamp_s),
                "frame_id": int(self.udp_frame_id),
                "frame": "xarm_base",
                "num_objects": int(len(poses_base)),
                "poses": [
                    {
                        "index": int(i),
                        "T_base_object": pose.astype(np.float64).tolist(),
                    }
                    for i, pose in enumerate(poses_base)
                ],
            }
            data = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            self.udp_sock.sendto(data, self.udp_addr)
            self.udp_frame_id = (self.udp_frame_id + 1) % (2**32)
        except Exception as e:
            now = time.time()
            if now - self.last_udp_error_print > 2.0:
                print(f"[警告] UDP 位姿发送失败: {e}")
                self.last_udp_error_print = now

    def visualize(self, rgb, K):
        """可视化所有工件"""
        vis = rgb.copy()
        for i, estimator in enumerate(self.estimators):
            line_color = self.colors[i % len(self.colors)]
            vis = estimator.visualize(vis, K, line_color=line_color)
        return vis

    def reset(self):
        """重置所有工件"""
        self.estimators = []
        self.initial_masks = []
        print("[重置] 已清除所有工件")

    def cleanup(self):
        """清理资源"""
        if self.udp_sock is not None:
            self.udp_sock.close()
            self.udp_sock = None
            print("[清理] UDP socket 已关闭")
        if self.shm is not None:
            self.shm.close()
            try:
                self.shm.unlink()
            except:
                pass
            print("[清理] 共享内存已关闭")


def main():
    parser = argparse.ArgumentParser(description="FoundationPose 实时多工件位姿估计")
    parser.add_argument('--mesh_file', type=str, required=True, help='物体 mesh 文件路径')
    parser.add_argument('--use_original_origin', action='store_true', help='使用obj文件的原始坐标系原点（默认使用几何中心）')
    parser.add_argument('--est_refine_iter', type=int, default=5, help='注册时的优化迭代次数')
    parser.add_argument('--track_refine_iter', type=int, default=3, help='跟踪时的优化迭代次数')
    parser.add_argument('--debug', type=int, default=1, help='调试级别')
    parser.add_argument('--width', type=int, default=1280, help='相机宽度')
    parser.add_argument('--height', type=int, default=720, help='相机高度')
    parser.add_argument('--serial', type=str, default='318122304321', help='相机序列号（多相机时使用）')
    parser.add_argument('--occlusion_removal', action='store_true', help='启用遮挡消除模式')
    parser.add_argument('--udp_host', type=str, default=None, help='可选：把 base 坐标系下的 FoundationPose 位姿用 UDP JSON 发到该主机/IP。')
    parser.add_argument('--udp_port', type=int, default=5005, help='UDP 目标端口，默认 5005。')
    parser.add_argument(
        '--timing_print_interval_s',
        type=float,
        default=1.0,
        help='跟踪 timing 打印间隔；默认 1 秒一次，设为 0 或负数关闭。',
    )
    parser.add_argument('--camera_calib', type=str,
                       default='/home/gxai/桌面/cbd/FoundationPose/camera_calibration/camera_to_base_final.json',
                       help='相机到基坐标系标定文件')
    args = parser.parse_args()

    set_logging_format()
    set_seed(0)

    # 初始化相机（使用硬件对齐，更快）
    camera = RealsenseCamera(width=args.width, height=args.height, align_on_device=True, serial_number=args.serial)
    K = camera.start()

    # 初始化多工件跟踪器
    tracker = MultiObjectTracker(
        mesh_file=args.mesh_file,
        est_refine_iter=args.est_refine_iter,
        track_refine_iter=args.track_refine_iter,
        debug=args.debug,
        camera_to_base_file=args.camera_calib,
        use_original_origin=args.use_original_origin,  # 传递参数
        udp_host=args.udp_host,
        udp_port=args.udp_port,
    )

    # ROI 选择器
    roi_selector = ROISelector()

    print("\n" + "="*60)
    print("FoundationPose 实时多工件位姿估计（串行注册 + 碰撞检测）")
    print("="*60)
    print("操作说明:")
    print("  空格键  - 暂停并框选工件（请从最上方工件开始框选！）")
    print("           左键: 添加顶点")
    print("           右键: 完成当前工件")
    print("           空格/Enter: 开始跟踪")
    print("  R键     - 重置所有工件")
    print("  Q键     - 退出")
    print("="*60)
    print(f"相机标定文件: {args.camera_calib}")
    print(f"遮挡消除模式: {'开启' if args.occlusion_removal else '关闭'}")
    if args.udp_host:
        print(f"UDP 发送目标: {args.udp_host}:{args.udp_port}")
    print("="*60 + "\n")

    paused = True
    fps_list = []
    last_timing_print = 0.0
    window_name = "FoundationPose Multi-Object Tracking"

    cv2.namedWindow(window_name)
    print("[提示] 按空格键开始框选工件")

    try:
        while True:
            t0 = time.time()

            # 1. 获取帧
            rgb, depth = camera.get_frame()
            if rgb is None:
                continue
            t1 = time.time()

            if paused:
                vis = rgb[..., ::-1].copy()
                cv2.putText(vis, "PAUSED - Press SPACE to select objects",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
                t2 = t1
                t3 = t1
                t3_5 = t1
                t4_vis = t1
            else:
                if len(tracker.estimators) > 0:
                    t2 = time.time()

                    # 2. FoundationPose 推理
                    tracker.track_all(rgb, depth, K,
                                      enable_occlusion_removal=args.occlusion_removal)

                    import torch
                    torch.cuda.synchronize()
                    t3 = time.time()

                    # 3. 发布位姿
                    tracker.publish_poses()
                    t3_5 = time.time()

                    # 4. 可视化
                    vis = tracker.visualize(rgb, K)
                    t4_vis = time.time()
                else:
                    vis = rgb
                    t2 = t1
                    t3 = t1
                    t3_5 = t1
                    t4_vis = t1

            display = vis[..., ::-1].copy()
            t4 = time.time()

            time_total = (t4 - t0) * 1000
            time_read = (t1 - t0) * 1000
            time_inference = (t3 - t2) * 1000

            fps = 1000.0 / max(time_total, 1e-6)
            fps_list.append(fps)
            if len(fps_list) > 30:
                fps_list.pop(0)
            avg_fps = np.mean(fps_list)

            status = f"FPS: {avg_fps:.1f} | Objects: {len(tracker.estimators)}"
            if len(tracker.estimators) > 0:
                status += " | TRACKING"
            cv2.putText(display, status, (10, args.height - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

            if (
                not paused
                and len(tracker.estimators) > 0
                and args.timing_print_interval_s > 0.0
                and time.time() - last_timing_print >= args.timing_print_interval_s
            ):
                last_timing_print = time.time()
                print(f"总: {time_total:.1f}ms | 读图: {time_read:.1f}ms | 推理: {time_inference:.1f}ms")

            cv2.imshow(window_name, display)
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                break

            elif key == 32:  # 空格
                if paused:
                    masks = roi_selector.select_multiple_masks(rgb, window_name=window_name)
                    if masks is not None and len(masks) > 0:
                        # 收集所有有效 mask
                        refined_masks = []
                        for mask in masks:
                            refined_mask = ROISelector.refine_mask_with_depth(mask, depth)
                            if refined_mask.sum() > 100:
                                refined_masks.append(refined_mask)

                        if refined_masks:
                            # 关键：串行注册（注册一个、擦除一个、再注册下一个）
                            tracker.add_objects_sequential(rgb, depth, refined_masks, K)
                            paused = False
                            print(f"[状态] 开始跟踪 {len(tracker.estimators)} 个工件")
                        else:
                            print("[取消] 没有有效的工件区域")
                    else:
                        print("[取消] 未选择工件")
                else:
                    paused = True
                    print("[状态] 已暂停")

            elif key == ord('r'):
                paused = True
                tracker.reset()
                print("[状态] 已重置，请重新框选工件")

    except KeyboardInterrupt:
        print("\n[中断] 用户终止")
    finally:
        camera.stop()
        tracker.cleanup()
        cv2.destroyAllWindows()
        print("[结束] 程序退出")


if __name__ == "__main__":
    main()
