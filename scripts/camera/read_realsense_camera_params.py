# read_realsense_camera_params.py
# -*- coding: utf-8 -*-

"""读取 RealSense 相机参数并保存为 JSON。"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pyrealsense2 as rs


def np_to_list(x):
    """numpy 数组转 list，方便保存 JSON"""
    return np.asarray(x).tolist()


def make_K_from_intrinsics(intr):
    """
    根据 RealSense intrinsics 构造相机内参矩阵 K

    K =
    [ fx   0   cx ]
    [  0  fy   cy ]
    [  0   0    1 ]
    """
    K = np.array(
        [
            [intr.fx, 0.0, intr.ppx],
            [0.0, intr.fy, intr.ppy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return K


def intrinsics_to_dict(intr):
    """
    RealSense intrinsics 转成字典

    intr 中主要包含：
    width, height: 图像宽高
    fx, fy: 焦距，单位是像素
    ppx, ppy: 主点坐标，也就是 cx, cy
    model: 畸变模型
    coeffs: 畸变系数
    """
    K = make_K_from_intrinsics(intr)

    return {
        "width": int(intr.width),
        "height": int(intr.height),
        "fx": float(intr.fx),
        "fy": float(intr.fy),
        "ppx_cx": float(intr.ppx),
        "ppy_cy": float(intr.ppy),
        "distortion_model": str(intr.model),
        "distortion_coeffs": [float(c) for c in intr.coeffs],
        "K": np_to_list(K),
    }


def extrinsics_to_dict(extr):
    """
    RealSense extrinsics 转成字典

    extr.rotation 是长度为 9 的数组，表示 3x3 旋转矩阵
    extr.translation 是长度为 3 的数组，单位通常是米

    外参含义：
    P_target = R * P_source + t
    """
    R = np.array(extr.rotation, dtype=np.float64).reshape(3, 3)
    t = np.array(extr.translation, dtype=np.float64).reshape(3, 1)

    T = np.eye(4, dtype=np.float64)
    T[:3, :3] = R
    T[:3, 3:4] = t

    return {
        "rotation_3x3": np_to_list(R),
        "translation_3x1_meter": np_to_list(t),
        "T_4x4": np_to_list(T),
        "baseline_meter": float(np.linalg.norm(t)),
    }


def get_device_info(device):
    """读取 RealSense 设备基本信息"""
    info = {}

    items = {
        "name": rs.camera_info.name,
        "serial_number": rs.camera_info.serial_number,
        "firmware_version": rs.camera_info.firmware_version,
        "product_id": rs.camera_info.product_id,
        "product_line": rs.camera_info.product_line,
    }

    for key, camera_info in items.items():
        try:
            if device.supports(camera_info):
                info[key] = device.get_info(camera_info)
            else:
                info[key] = None
        except Exception:
            info[key] = None

    return info


def warmup_camera(pipeline, num_frames=30):
    """预热相机，避免刚启动时曝光和深度不稳定"""
    for _ in range(num_frames):
        pipeline.wait_for_frames()


def read_one_camera_params(serial_number, width, height, fps, warmup_frames=30):
    """
    读取单台 RealSense 相机的全部参数

    serial_number:
        指定要启动的 RealSense 序列号

    width, height, fps:
        启动 RGB 和 Depth 流的分辨率和帧率
    """

    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_device(serial_number)

    # 开启深度流
    config.enable_stream(
        rs.stream.depth,
        width,
        height,
        rs.format.z16,
        fps,
    )

    # 开启彩色流
    config.enable_stream(
        rs.stream.color,
        width,
        height,
        rs.format.bgr8,
        fps,
    )

    print("=" * 80)
    print(f"[启动] 正在启动 RealSense，序列号: {serial_number}")
    print(f"[启动] 分辨率: {width}x{height}, FPS: {fps}")

    profile = None

    try:
        profile = pipeline.start(config)

        device = profile.get_device()
        device_info = get_device_info(device)

        print(f"[设备] 名称: {device_info.get('name')}")
        print(f"[设备] 序列号: {device_info.get('serial_number')}")
        print(f"[设备] 固件版本: {device_info.get('firmware_version')}")
        print(f"[设备] 产品线: {device_info.get('product_line')}")

        # 读取深度尺度
        depth_sensor = device.first_depth_sensor()
        depth_scale = float(depth_sensor.get_depth_scale())

        print(f"[深度] depth_scale = {depth_scale}")

        # 预热
        print(f"[预热] 等待 {warmup_frames} 帧...")
        warmup_camera(pipeline, warmup_frames)

        # 等待一帧，确保 profile 可用
        frames = pipeline.wait_for_frames()
        depth_frame = frames.get_depth_frame()
        color_frame = frames.get_color_frame()

        if not depth_frame:
            raise RuntimeError("没有获取到 depth_frame，请检查深度流是否正常。")

        if not color_frame:
            raise RuntimeError("没有获取到 color_frame，请检查彩色流是否正常。")

        # 获取 stream profile
        depth_stream_profile = (
            depth_frame.profile.as_video_stream_profile()
        )
        color_stream_profile = (
            color_frame.profile.as_video_stream_profile()
        )

        # 获取内参
        depth_intr = depth_stream_profile.get_intrinsics()
        color_intr = color_stream_profile.get_intrinsics()

        # 获取外参
        # 含义：把深度相机坐标系下的 3D 点变换到彩色相机坐标系
        depth_to_color_extr = depth_stream_profile.get_extrinsics_to(
            color_stream_profile
        )

        # 含义：把彩色相机坐标系下的 3D 点变换到深度相机坐标系
        color_to_depth_extr = color_stream_profile.get_extrinsics_to(
            depth_stream_profile
        )

        color_intr_dict = intrinsics_to_dict(color_intr)
        depth_intr_dict = intrinsics_to_dict(depth_intr)

        depth_to_color_dict = extrinsics_to_dict(depth_to_color_extr)
        color_to_depth_dict = extrinsics_to_dict(color_to_depth_extr)

        # 汇总结果
        result = {
            "device_info": device_info,
            "stream_config": {
                "width": int(width),
                "height": int(height),
                "fps": int(fps),
                "depth_format": "z16",
                "color_format": "bgr8",
            },
            "depth_scale_meter_per_unit": depth_scale,
            "color_camera": color_intr_dict,
            "depth_camera": depth_intr_dict,
            "extrinsics": {
                "depth_to_color": depth_to_color_dict,
                "color_to_depth": color_to_depth_dict,
            },
        }

        print("\n[彩色相机内参 K_color]")
        print(np.array(color_intr_dict["K"]))

        print("\n[深度相机内参 K_depth]")
        print(np.array(depth_intr_dict["K"]))

        print("\n[深度相机畸变参数]")
        print(depth_intr_dict["distortion_model"])
        print(depth_intr_dict["distortion_coeffs"])

        print("\n[彩色相机畸变参数]")
        print(color_intr_dict["distortion_model"])
        print(color_intr_dict["distortion_coeffs"])

        print("\n[depth -> color 外参矩阵 T_depth_to_color]")
        print(np.array(depth_to_color_dict["T_4x4"]))

        print("\n[color -> depth 外参矩阵 T_color_to_depth]")
        print(np.array(color_to_depth_dict["T_4x4"]))

        print(f"\n[基线长度] {depth_to_color_dict['baseline_meter']} m")

        return result

    finally:
        if profile is not None:
            pipeline.stop()
            time.sleep(0.5)
            print(f"[停止] 相机 {serial_number} 已停止")


def list_connected_realsense_devices():
    """枚举所有连接到本机的 RealSense 设备"""
    ctx = rs.context()
    devices = ctx.query_devices()

    serial_numbers = []

    print("=" * 80)
    print(f"[检测] 当前检测到 {len(devices)} 台 RealSense 设备")

    for i, dev in enumerate(devices):
        info = get_device_info(dev)
        serial = info.get("serial_number")

        print("-" * 80)
        print(f"[设备 {i}]")
        print(f"名称: {info.get('name')}")
        print(f"序列号: {info.get('serial_number')}")
        print(f"固件版本: {info.get('firmware_version')}")
        print(f"产品线: {info.get('product_line')}")
        print(f"产品 ID: {info.get('product_id')}")

        if serial is not None:
            serial_numbers.append(serial)

    return serial_numbers


def save_json(data, save_path):
    """保存 JSON 文件"""
    save_path = Path(save_path)
    save_path.parent.mkdir(parents=True, exist_ok=True)

    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(
            data,
            f,
            ensure_ascii=False,
            indent=4,
        )

    print("=" * 80)
    print(f"[保存] 相机参数已保存到: {save_path.resolve()}")


def parse_args():
    parser = argparse.ArgumentParser(
        description="读取 Intel RealSense 彩色相机和深度相机参数"
    )

    parser.add_argument(
        "--width",
        type=int,
        default=1280,
        help="图像宽度，默认 1280",
    )

    parser.add_argument(
        "--height",
        type=int,
        default=720,
        help="图像高度，默认 720",
    )

    parser.add_argument(
        "--fps",
        type=int,
        default=30,
        help="帧率，默认 30",
    )

    parser.add_argument(
        "--serial",
        type=str,
        default=None,
        help="指定某一台 RealSense 的序列号。不指定则读取所有连接的 RealSense。",
    )

    parser.add_argument(
        "--output",
        type=str,
        default="realsense_camera_params.json",
        help="输出 JSON 文件路径",
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=30,
        help="预热帧数，默认 30",
    )

    return parser.parse_args()


def main():
    args = parse_args()

    all_results = {}

    if args.serial is not None:
        # 只读取指定序列号的相机
        serial_numbers = [args.serial]
    else:
        # 自动读取所有连接的 RealSense 相机
        serial_numbers = list_connected_realsense_devices()

    if len(serial_numbers) == 0:
        print("[错误] 没有检测到 RealSense 相机，请检查 USB 连接和驱动。")
        return

    for serial in serial_numbers:
        try:
            params = read_one_camera_params(
                serial_number=serial,
                width=args.width,
                height=args.height,
                fps=args.fps,
                warmup_frames=args.warmup,
            )

            all_results[serial] = params

        except Exception as e:
            print("=" * 80)
            print(f"[错误] 读取相机 {serial} 参数失败")
            print(f"[错误信息] {repr(e)}")
            print(
                "[建议] 如果是分辨率或帧率不支持，尝试："
                "python read_realsense_camera_params.py --width 640 --height 480 --fps 30"
            )

    save_json(all_results, args.output)


if __name__ == "__main__":
    main()