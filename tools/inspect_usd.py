"""读取 USD 文件，打印所有 prim 路径（不依赖 pxr，通过 omni.usd 或 usda 文本解析）"""
import sys
import subprocess
import os

usd_path = sys.argv[1] if len(sys.argv) > 1 else "/home/gxai/IsaacLab/czr/tj.usd"
print(f"\n=== 检查 USD: {usd_path} ===")

# 方法1：用 usdcat 转换为 usda 文本格式
usdcat_candidates = [
    "/home/gxai/.local/share/ov/pkg/isaac-sim-4.5.0/kit/python/bin/usdcat",
    "/home/gxai/.local/share/ov/pkg/isaac-sim-4.2.0/kit/python/bin/usdcat",
    "/home/gxai/.local/share/ov/pkg/isaac_sim-2023.1.1/kit/python/bin/usdcat",
    "/home/gxai/.local/share/ov/pkg/isaac_sim-4.0.0/kit/python/bin/usdcat",
    "/home/gxai/.local/share/ov/pkg/isaac_sim-4.1.0/kit/python/bin/usdcat",
]

# 查找实际存在的 usdcat
usdcat = None
for c in usdcat_candidates:
    if os.path.exists(c):
        usdcat = c
        break

# 也用 find 搜索
if not usdcat:
    result = subprocess.run(
        ["find", "/home/gxai/.local/share/ov", "-name", "usdcat", "-type", "f"],
        capture_output=True, text=True, timeout=10
    )
    candidates = result.stdout.strip().splitlines()
    if candidates:
        usdcat = candidates[0]

if usdcat:
    print(f"使用 usdcat: {usdcat}\n")
    result = subprocess.run([usdcat, "--flatten", usd_path], capture_output=True, text=True, timeout=30)
    if result.returncode == 0:
        print(result.stdout[:8000])  # 只打印前8000字符
    else:
        print("usdcat 错误:", result.stderr[:2000])
else:
    print("未找到 usdcat，尝试用 strings 提取...")
    result = subprocess.run(["strings", usd_path], capture_output=True, text=True)
    lines = [l.strip() for l in result.stdout.splitlines() if len(l.strip()) > 2]
    for l in lines:
        print(l)
