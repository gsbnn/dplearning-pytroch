# 安装依赖：pip install fastdtw numpy matplotlib scipy

import numpy as np
import matplotlib.pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

# 1. 生成数据
t_A = np.linspace(0, 2*np.pi, 50)
sequence_A = np.sin(t_A) + np.random.normal(0, 0.1, 50)
t_B = np.linspace(0, 2*np.pi, 30)
sequence_B = np.sin(t_B * 0.8) * 1.2 + np.random.normal(0, 0.1, 30)

# 2. 计算DTW路径
distance, path = fastdtw(
    sequence_A.reshape(-1, 1),
    sequence_B.reshape(-1, 1),
    dist=euclidean
)
path = np.array(path)
idx_A, idx_B = path[:, 0], path[:, 1]

# 3. 对齐并可视化
aligned_A = sequence_A[idx_A]
aligned_B = sequence_B[idx_B]

plt.figure(figsize=(12, 6))
plt.subplot(2,1,1)
plt.plot(sequence_A, label="Sensor A", alpha=0.5)
plt.plot(sequence_B, label="Sensor B", alpha=0.5)
plt.legend()

plt.subplot(2,1,2)
plt.plot(aligned_A, label="Aligned A", marker='o')
plt.plot(aligned_B, label="Aligned B", marker='x')
plt.legend()
plt.show()