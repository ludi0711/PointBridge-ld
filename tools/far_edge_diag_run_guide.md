# 远端边深度丢失诊断 —— 如何运行与开展实验

> 实验方案与判读：[far_edge_diag_experiment.md](./far_edge_diag_experiment.md)
> 脚本：[diag_far_edge_depth.py](./diag_far_edge_depth.py)

---

## 1. 环境与前置条件

```bash
conda activate gx_va_deploy        # 必须：既要 pyrealsense2 又要 isaaclab
cd /root/gx-va/gx-VA-isaaclab_new  # 脚本会 sys.path 挂到项目根，请从任意目录运行
```

前置条件：
- D435 已插入且被系统识别（`rs-enumerate-devices` 能列出设备）；
- 相机标定位姿没变（脚本 S0 会与 `configs` 里的标定值核对）；
- 手眼标定文件在 `configs/` 下的路径与 `real_roi_pointcloud.py` 中一致（S0 自动校验）。

**本脚本不修改任何其它代码**：相机、掩码、内参、外参、反投影全部 import 复用 `scripts/camera/*` 与 `configs/point_bridge_pointcloud.py`。

结果文件默认写入 `/root/autodl-tmp/far_edge_diag/`（新建文件夹，可用 `--out` 改）。

---

## 2. 先离线自检（不连相机，1 秒）

```bash
python tools/diag_far_edge_depth.py --selftest
```

用合成数据验证四件事：远方向符号、反投影往返互逆、近远分半与内收量测量、深度分箱。全 PASS 再上机。

---

## 3. 实机交互模式（推荐，分阶段可重复）

```bash
python tools/diag_far_edge_depth.py
```

进入菜单：

```
  [0] 相机就绪 + 内参/外参对照
  [1] 标注工件 ROI
  [2] 单帧远边诊断（近端 vs 远端）
  [3] 稳定性采集（N 帧）
  [4] 点云级验证 + 汇总报告
  [a] 一键顺序跑 0→4
  [q] 退出
```

### 开展实验的推荐顺序（一次典型实机实验）

1. **摆位**：把工件摆到**与部署抓取相同的典型位姿**（同距离、同姿态），相机不动。工件距相机建议 0.4–1.0 m（在 `--depth_range` 默认色标内）。
2. **按 0**：起相机。核对命令行打印的实机内参与仿真 canonical（fx=fy≈615、cx≈320、cy≈240）偏差不大；外参校验通过。
3. **按 1**：弹出窗口后，**沿工件可见轮廓**左键点一圈（≥3 点），回车确认。终端会打印多边形面积——面积 < 200 px 就回车重标或让工件靠近些。
4. **按 2**（核心，可反复按多帧看）：观察
   - 路线 A：远/近空洞率比是否 > 2；远端内收多少 mm；
   - 路线 B：深度分箱空洞率是否越远越高（两条路线一致才可信）；
   - 终端判定行直接给出训练侧动作。弹窗/落盘的 `s2_overlay.png` 上空洞标红、箭头指远，一眼可验。
5. **按 3**：连采 30 帧（`--frames` 可改），看缺失是否稳定。
6. **按 4**：真机反投影 64 点，看质心沿远轴偏移（负 = 向相机）。
7. 收工 `q`（相机自动停止）。

### 无显示器（SSH）时

交互手标不可用，用 `--polygon` 传入坐标（或 `meta.json` 路径）跳过 S1 手标，配合 `--no_window`：

```bash
python tools/diag_far_edge_depth.py --no_window \
    --polygon "310,220 380,225 385,275 305,270"
```

也可以直接一键跑（见下节）。

---

## 4. 无头一键模式

```bash
python tools/diag_far_edge_depth.py --run_all \
    --polygon "310,220 380,225 385,275 305,270" \
    --frames 30 --out /root/autodl-tmp/far_edge_diag
```

顺序执行 S0→S4，实时打印各阶段中间结果与判定，结束后在输出目录生成全部结果文件。

---

## 5. 常用参数

| 参数 | 默认 | 说明 |
|---|---|---|
| `--width/--height/--fps` | 640/480/30 | 相机分辨率帧率（与部署一致） |
| `--serial` | 无 | 多相机时指定序列号 |
| `--warmup` | 30 | 预热丢帧数（自动曝光收敛） |
| `--polygon` | 无 | 复用多边形："x,y x,y ..." 或 meta.json 路径 |
| `--depth_range LO HI` | 0.3 1.5 | 深度伪彩色标区间(米)，按工件距离调 |
| `--bins` | 10 | S2 深度分箱数 |
| `--frames` | 30 | S3 采集帧数 |
| `--intrinsics real/sim` | real | 反投影用实机内参 / 仿真 canonical（隔离内参差异） |
| `--noise_std` | 0.0 | S4 反投影高斯噪声，默认 0 便于核对几何 |
| `--no_workspace_filter` | 关 | 关掉 S4 工作空间裁剪，看原始分布 |
| `--no_window` | 关 | 不弹窗（SSH） |
| `--run_all` | 关 | 一键顺序跑 S0→S4（需 `--polygon`） |
| `--selftest` | 关 | 离线几何自检 |
| `--out` | `/root/autodl-tmp/far_edge_diag` | 结果输出目录 |

---

## 6. 实机操作检查单

- [ ] `gx_va_deploy` 环境；`rs-enumerate-devices` 能看到 D435
- [ ] 相机支架/位姿自上次标定后没动过（S0 会核对）
- [ ] 工件摆位与部署抓取典型位姿一致；ROI 内只有工件（别把夹具/桌面圈进去）
- [ ] `--selftest` 通过
- [ ] 工件静止时跑 S3（人不要扶着工件）
- [ ] 多摆位复测：换 2–3 个典型工件位姿各跑一遍 S2，看内收量是否一致（一致才说明 band 可全局用）
- [ ] 实验完查看 `/root/autodl-tmp/far_edge_diag/` 下 `final_report.json` 的 band 建议值

---

## 7. 常见问题

- **掩码内无有效深度（S2 直接返回）**：多边形圈到桌面外/背景，或距离超量程（D435 建议 0.3–1.5 m），或工件表面全反光。重标多边形或换摆位。
- **近远空洞率都高且对称**：整体反光/分割问题，不是远边问题——按实验方案第 4 节第 2 分支理解，跑 S4 核对。
- **两条路线矛盾**（A 说远边缺、B 说没有）：优先怀疑外参标定过期，重新做手眼标定后复测；也可 `--intrinsics sim` 试一次隔离内参。
- **S4 点云偏移小但部署仍偏**：上游诊断排除嫌疑，偏置在观测编码/策略下游，另查。
- **弹窗卡死/没有 X**：`--no_window --polygon ...`。
- **多台相机混淆**：`--serial <序列号>` 指定。
