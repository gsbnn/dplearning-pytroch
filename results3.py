import matplotlib
from matplotlib import pyplot as plt

# 解决中文字体下坐标轴负数的负号显示问题
plt.rcParams['axes.unicode_minus'] = False
# 中文字体
font1 = matplotlib.font_manager.FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=15)
# 英文字体
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 15,
}

x = [0.01, 0.05, 0.1, 0.15, 0.2]
rmse = [0.0155, 0.0086, 0.0099, 0.0131, 0.0160]
r2 = [0.9989, 0.9996, 0.9995, 0.9990, 0.9986]
fig, axes = plt.subplots(1, 2)
axes[0].plot(x, rmse, 'b-o')
axes[0].set_xlabel("样本相似度阈值",  fontproperties=font1)
axes[0].set_ylabel("RMSE", fontproperties=font2)
axes[0].tick_params(labelsize=15)
labels = axes[0].get_xticklabels() + axes[0].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
axes[1].plot(x, r2, 'b-o')
axes[1].set_xlabel("样本相似度阈值",  fontproperties=font1)
axes[1].set_ylabel(r"$R^2$", fontsize=15)
axes[1].tick_params(labelsize=15)
labels = axes[1].get_xticklabels() + axes[1].get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
fig.set_size_inches(15, 6)
fig.subplots_adjust(wspace=0.3)
fig.savefig('E:\Deep_Learning\picture\\results3.svg', dpi=600, bbox_inches = 'tight')
plt.show()