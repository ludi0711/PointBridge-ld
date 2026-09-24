# 相机工具

## 读取 RealSense 参数

脚本：`scripts/camera/read_realsense_camera_params.py`

读取 RealSense 内参、外参和 depth scale，保存为 JSON。

```bash
python scripts/camera/read_realsense_camera_params.py   --output camera_params.json
```

## 实时位姿估计

脚本：`scripts/camera/run_realtime.py`

FoundationPose + RealSense 实时位姿估计。

```bash
python scripts/camera/run_realtime.py   --mesh_file /path/to/object.obj   --width 1280   --height 720   --serial 318122304321   --camera_calib camera_params.json
```


## FoundationPose UDP 传输

205 上跑 FoundationPose 时，可以把标定后的物体位姿通过 UDP 发给 201。不开 `--udp_host` 时仍然只走原来的本机共享内存；加上 `--udp_host` 后，每帧跟踪成功会额外发送一包 UDP JSON。

```bash
python scripts/camera/run_realtime.py \
  --mesh_file /path/to/object.obj \
  --camera_calib /path/to/camera_to_base.json \
  --serial 318122304321 \
  --width 1280 \
  --height 720 \
  --udp_host <201的IP> \
  --udp_port 5005
```

UDP 数据格式为单个 UTF-8 JSON datagram：

```json
{
  "schema": "foundationpose_multi_pose.v1",
  "timestamp": 1710000000.123,
  "frame_id": 0,
  "frame": "xarm_base",
  "num_objects": 1,
  "poses": [
    {
      "index": 0,
      "T_base_object": [
        [1.0, 0.0, 0.0, 0.123],
        [0.0, 1.0, 0.0, 0.456],
        [0.0, 0.0, 1.0, 0.789],
        [0.0, 0.0, 0.0, 1.0]
      ]
    }
  ]
}
```

`T_base_object` 是经过 `--camera_calib` 外参转换后的 xArm/base 坐标系下物体 4x4 位姿矩阵，平移单位为米。

## 相机封装

`realsense_camera.py` 是 RealSense 相机封装模块，通常由其他脚本导入。
