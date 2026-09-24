# RealSense 相机封装类 - 直接连接本机
# 支持 D455/D435 等型号
# 在本机进行深度对齐，减少RealSense处理负担

import pyrealsense2 as rs
import numpy as np
import cv2


class RealsenseCamera:
    def __init__(self, width=1280, height=720, fps=30, align_on_device=True, serial_number=None):
        """
        初始化 RealSense 相机
        Args:
            width: 图像宽度
            height: 图像高度
            fps: 帧率
            align_on_device: 是否在设备上对齐（True=硬件对齐，推荐）
            serial_number: 相机序列号（可选，用于多相机场景）
        """
        self.width = width
        self.height = height
        self.fps = fps
        self.align_on_device = align_on_device
        self.serial_number = serial_number
        
        self.pipeline = rs.pipeline()
        self.config = rs.config()
        
        # 如果指定了序列号，则启用该设备
        if serial_number:
            self.config.enable_device(serial_number)
            print(f"[相机] 指定使用序列号: {serial_number}")
        
        # 配置流
        self.config.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        self.config.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        
        # 对齐方式
        if align_on_device:
            # 在RealSense设备上对齐（硬件加速，快）
            self.align = rs.align(rs.stream.color)
            print("[相机] 使用硬件深度对齐")
        else:
            # 在本机对齐（软件，慢）
            self.align = None
            print("[相机] 使用软件深度对齐")
        
        self.profile = None
        self.K_color = None  # 彩色相机内参
        self.K_depth = None  # 深度相机内参
        self.depth_scale = None
        self.depth_intrinsics = None
        self.color_intrinsics = None
        
    def start(self):
        """启动相机并预热"""
        print(f"[相机] 正在启动 RealSense {self.width}x{self.height}@{self.fps}fps...")
        self.profile = self.pipeline.start(self.config)
        
        # 获取深度比例
        depth_sensor = self.profile.get_device().first_depth_sensor()
        self.depth_scale = depth_sensor.get_depth_scale()
        print(f"[相机] 深度比例: {self.depth_scale}")
        
        # 预热
        print("[相机] 预热中...")
        for _ in range(30):
            self.pipeline.wait_for_frames()
        
        # 获取内参
        frames = self.pipeline.wait_for_frames()
        
        # 获取彩色和深度内参
        color_frame = frames.get_color_frame()
        depth_frame = frames.get_depth_frame()
        
        self.color_intrinsics = color_frame.profile.as_video_stream_profile().get_intrinsics()
        self.depth_intrinsics = depth_frame.profile.as_video_stream_profile().get_intrinsics()
        
        # 彩色相机内参矩阵
        self.K_color = np.array([
            [self.color_intrinsics.fx, 0, self.color_intrinsics.ppx],
            [0, self.color_intrinsics.fy, self.color_intrinsics.ppy],
            [0, 0, 1]
        ])
        
        # 深度相机内参矩阵
        self.K_depth = np.array([
            [self.depth_intrinsics.fx, 0, self.depth_intrinsics.ppx],
            [0, self.depth_intrinsics.fy, self.depth_intrinsics.ppy],
            [0, 0, 1]
        ])
        
        print(f"[相机] 启动完成")
        print(f"[相机] 彩色内参:\n{self.K_color}")
        if not self.align_on_device:
            print(f"[相机] 深度内参:\n{self.K_depth}")
        
        return self.K_color
    
    def align_depth_to_color_manual(self, depth, depth_intrinsics, color_intrinsics):
        """
        手动将深度图对齐到彩色图（在本机CPU上）
        优化版本：预计算映射表
        """
        # 如果映射表未初始化，则创建
        if not hasattr(self, '_map_x') or not hasattr(self, '_map_y'):
            h, w = depth.shape
            
            # 计算内参比例
            fx_ratio = color_intrinsics.fx / depth_intrinsics.fx
            fy_ratio = color_intrinsics.fy / depth_intrinsics.fy
            cx_offset = color_intrinsics.ppx - depth_intrinsics.ppx * fx_ratio
            cy_offset = color_intrinsics.ppy - depth_intrinsics.ppy * fy_ratio
            
            # 预计算映射表（只需计算一次）
            u_coords, v_coords = np.meshgrid(np.arange(w), np.arange(h))
            self._map_x = ((u_coords - cx_offset) / fx_ratio).astype(np.float32)
            self._map_y = ((v_coords - cy_offset) / fy_ratio).astype(np.float32)
            
            print("[相机] 深度对齐映射表已预计算")
        
        # 使用预计算的映射表进行快速重映射
        aligned_depth = cv2.remap(depth, self._map_x, self._map_y, cv2.INTER_NEAREST)
        
        return aligned_depth
    
    def get_frame(self):
        """
        获取一帧数据
        Returns:
            rgb: (H, W, 3) uint8, RGB格式
            depth: (H, W) float32, 单位米，已对齐到彩色
        """
        frames = self.pipeline.wait_for_frames()
        
        if self.align_on_device:
            # 设备端对齐
            aligned = self.align.process(frames)
            depth_frame = aligned.get_depth_frame()
            color_frame = aligned.get_color_frame()
        else:
            # 本机端对齐
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
        
        if not depth_frame or not color_frame:
            return None, None
        
        # BGR -> RGB
        color = np.asanyarray(color_frame.get_data())
        rgb = color[..., ::-1].copy()
        
        # 深度转米
        depth = np.asanyarray(depth_frame.get_data()).astype(np.float32)
        depth = depth * self.depth_scale
        
        # 如果不在设备上对齐，则在本机对齐
        if not self.align_on_device:
            depth = self.align_depth_to_color_manual(depth, self.depth_intrinsics, self.color_intrinsics)
        
        return rgb, depth
    
    def stop(self):
        """停止相机"""
        self.pipeline.stop()
        print("[相机] 已停止")
