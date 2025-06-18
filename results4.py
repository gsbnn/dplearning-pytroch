import numpy as np
import pandas as pd
import matplotlib
from matplotlib import pyplot as plt

# 解决中文字体下坐标轴负数的负号显示问题
plt.rcParams['axes.unicode_minus'] = False
# 中文字体
font1 = matplotlib.font_manager.FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=10)
# 英文字体
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 10,
}

# 导入数据
mic_path = 'data\mic_result.xlsx'
drop_label = ["底物浓度", "菌体浓度", "产物浓度"]
mic = pd.read_excel(mic_path, header=0, index_col=0)
mic = mic.drop(index=drop_label, columns=drop_label)
label = mic.columns.values # 获取列名
data = np.array(mic)
data = np.around(data, 2)

# 设置图片格式
fig, axes = plt.subplots(1, 1, figsize=(10, 6))
axes.set_xticks(np.arange(len(label)), labels=label, rotation=45, rotation_mode='anchor', ha='right', fontproperties=font1)
axes.set_yticks(np.arange(len(label)), labels=label, fontproperties=font1)
for i in range(len(label)):
    for j in range(len(label)):
        color = 'w' if data[i, j] < 0.95 else 'k'
        axes.text(j, i, data[i, j], ha='center', va='center', color=color, fontproperties=font2)

cobar = fig.colorbar(axes.imshow(data, cmap='viridis'), ax=axes)
cobar.set_ticklabels(cobar.ax.get_yticklabels(), fontproperties=font2) # colorbar是一种特殊的axes
fig.savefig('E:\Deep_Learning\picture\\results4.svg', dpi=600, bbox_inches = 'tight')
plt.show()