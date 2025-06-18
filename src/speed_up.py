import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from tqdm import tqdm  # 进度条库
import threading
import psutil
import time

log_file = open("timing_log.txt", "w")

# 网格长宽各乘2，点数由201 -> 401
x = np.linspace(0, 1.0, 401)
y = np.linspace(0, 1.0, 401)
X, Y = np.meshgrid(x, y)
T_1 = np.zeros_like(X)
x_1 = np.linspace(0.005, 0.995, 399)  # 对应的x坐标数组，长度399
q_1 = np.zeros_like(x_1)
T_2 = np.zeros_like(X)
q_2 = np.zeros_like(x_1)

# 边界条件
T_top = 0
T_left = 0
T_right = 0
T_bottom = 100

# 迭代次数
iterations = 100000

# 设置边界条件
T_1[0, :] = T_bottom
T_1[-1, :] = T_top
T_1[:, 0] = T_right
T_1[:, -1] = T_left

T_2[0, :] = T_bottom
T_2[-1, :] = T_top
T_2[:, 0] = T_right
T_2[:, -1] = T_left

def print_memory_usage():
    process = psutil.Process(os.getpid())
    mem = process.memory_info().rss / 1024 / 1024
    print(f"[Memory] Usage: {mem:.2f} MB")

# 缺陷区域掩码，True表示不更新（缺陷处）
mask_1 = np.zeros_like(T_1, dtype=bool)
mask_1[200:, 160:242] = True  # 纵向缺陷

mask_2 = np.zeros_like(T_2, dtype=bool)
mask_2[160:242, 200:] = True  # 横向缺陷

def updateT1():
    start_time = time.time()
    for _ in range(iterations):
        T_old = T_1.copy()
        T_new = 0.25 * (T_old[2:, 1:-1] + T_old[:-2, 1:-1] + T_old[1:-1, 2:] + T_old[1:-1, :-2])
        mask_update = ~mask_1[1:-1, 1:-1]
        T_1[1:-1, 1:-1][mask_update] = T_new[mask_update]
        if _ % 2000 == 0:
            print_memory_usage()
            # print(_)
            elapsed = time.time() - start_time
            print(f"[T1] Iter {_} time: {elapsed:.4f} s")
            log_file.write(f"[T1] Iter {_} time: {elapsed:.4f} s")
            start_time = time.time()

# 迭代计算 T_2，带进度条和矢量化
def updateT2():
    start_time2 = time.time()
    for _ in range(iterations):
        T_old = T_2.copy()
        T_new = 0.25 * (T_old[2:, 1:-1] + T_old[:-2, 1:-1] + T_old[1:-1, 2:] + T_old[1:-1, :-2])
        mask_update = ~mask_2[1:-1, 1:-1]
        T_2[1:-1, 1:-1][mask_update] = T_new[mask_update]
        if _ % 2000 == 0:
            print_memory_usage()
            # print(_)
            elapsed2 = time.time() - start_time2
            print(f"[T1] Iter {_} time: {elapsed2:.4f} s")
            log_file.write(f"[T1] Iter {_} time: {elapsed2:.4f} s")
            start_time2 = time.time()

thread1 = threading.Thread(target=updateT1)
thread2 = threading.Thread(target=updateT2)

thread1.start()
thread2.start()

thread1.join()
thread2.join()

log_file.close()

# for _ in tqdm(range(iterations), desc='Iterating T_1'):
#     T_old = T_1.copy()
#     T_new = 0.25 * (T_old[2:, 1:-1] + T_old[:-2, 1:-1] + T_old[1:-1, 2:] + T_old[1:-1, :-2])
#     mask_update = ~mask_1[1:-1, 1:-1]
#     T_1[1:-1, 1:-1][mask_update] = T_new[mask_update]

# # 迭代计算 T_2，带进度条和矢量化
# for _ in tqdm(range(iterations), desc='Iterating T_2'):
#     T_old = T_2.copy()
#     T_new = 0.25 * (T_old[2:, 1:-1] + T_old[:-2, 1:-1] + T_old[1:-1, 2:] + T_old[1:-1, :-2])
#     mask_update = ~mask_2[1:-1, 1:-1]
#     T_2[1:-1, 1:-1][mask_update] = T_new[mask_update]

# 计算温度梯度 q
for i in range(1, 400):
    q_1[i-1] = T_1[0, i] - T_1[1, i]
    q_2[i-1] = T_2[0, i] - T_2[1, i]

print(np.trapezoid(q_1, x_1))
print(np.trapezoid(q_2, x_1))

# 可视化并保存单独的图像文件

figsize=(12,12)

# 1. vertical defect
fig1, ax1 = plt.subplots(figsize=figsize)
cs1 = ax1.contourf(X, Y, T_1, 50, cmap='jet')
rect = Rectangle((0.4, 0.5), 0.2, 0.5, linewidth=1, edgecolor='white', facecolor='none')
ax1.add_patch(rect)
ax1.set_aspect('equal')
ax1.set_title('vertical defect')
fig1.colorbar(cs1, ax=ax1)
fig1.savefig('../result/vertical_defect.png')
plt.close(fig1)

# 2. horizontal defect
fig2, ax2 = plt.subplots(figsize=figsize)
cs2 = ax2.contourf(X, Y, T_2, 50, cmap='jet')
rect2 = Rectangle((0.5, 0.4), 0.5, 0.2, linewidth=1, edgecolor='white', facecolor='none')
ax2.add_patch(rect2)
ax2.set_aspect('equal')
ax2.set_title('horizontal defect')
fig2.colorbar(cs2, ax=ax2)
fig2.savefig('../result/horizontal_defect.png')
plt.close(fig2)

# 3. bottom temperature gradient 2
fig3, ax3 = plt.subplots()
ax3.plot(x_1, q_1)
ax3.set_title('vertical bottom temperature gradient')
fig3.savefig('../result/vertical_bottom_temperature_gradient.png')
plt.close(fig3)

# 4. bottom temperature gradient 3
fig4, ax4 = plt.subplots()
ax4.plot(x_1, q_2)
ax4.set_title('horizon bottom temperature gradient')
fig4.savefig('../result/horizon_bottom_temperature_gradient.png')
plt.close(fig4)
