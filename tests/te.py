import csv, numpy as np
rows = list(csv.DictReader(open('/root/gx-va/gx-VA-isaaclab_v1/logs/deploy_pointcloud/rollout_20260917_135920.csv')))
def f(r,k):
    try: return float(r[k])
    except: return np.nan
obj = np.array([f(rows[-1],'obj_centroid_x'), f(rows[-1],'obj_centroid_y'), f(rows[-1],'obj_centroid_z')])
ee  = np.array([f(rows[-1],'ee_pos_x'), f(rows[-1],'ee_pos_y'), f(rows[-1],'ee_pos_z')])
true = np.array([0.0, 0.0, 0.0])   # ← 填你量到的真实位置
axis = np.array([-0.0647, 0.8906, -0.4501]); axis /= np.linalg.norm(axis)

def report(name, err):
    print(f'{name}: 误差 {err*1000} mm, 模长 {np.linalg.norm(err)*1000:.1f} mm')
    print(f'   沿光轴分量 {err@axis*1000:+.1f} mm, 垂直光轴 {np.linalg.norm(err-(err@axis)*axis)*1000:.1f} mm')

report('感知误差 obj-true', obj - true)   # 是深度/外参的问题
report('夹取误差 ee-true ', ee  - true)   # 你肉眼看到的"夹偏"