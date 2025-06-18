import pandas as pd
import matplotlib
from matplotlib import pyplot as plt
import matplotlib.patches as patches

# 解决中文字体下坐标轴负数的负号显示问题
plt.rcParams['axes.unicode_minus'] = False
# 中文字体
font1 = matplotlib.font_manager.FontProperties(fname=r"C:\Windows\Fonts\simsun.ttc", size=15)
# 英文字体
font2 = {'family' : 'Times New Roman',
'weight' : 'normal',
'size'   : 15,
}

def config_figure(axes, xlabel, ylabel, xlim, ylim, font, title="", xscale='linear', yscale='linear'):
    """设置图片属性"""
    axes.set_title(title, fontproperties=font)
    axes.set_xlabel(xlabel, fontproperties=font)
    axes.set_ylabel(ylabel, fontproperties=font)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.grid()


win = 4
length = 400

JGCNpe_path = "data\JITL_GCN_青霉素浓度.xlsx"
JGCNce_path = "data\JITL_GCN_菌体浓度.xlsx"
JRVMpe_path = "data\JITL_RVM_青霉素浓度.xlsx"
JRVMce_path = "data\JITL_RVM_菌体浓度.xlsx"
GCNpe_path = "data\GCN_青霉素浓度.xlsx"
GCNce_path = "data\GCN_菌体浓度.xlsx"

JGCNpeData = pd.read_excel(JGCNpe_path, header=None)
JGCNceData = pd.read_excel(JGCNce_path, header=None)
JRVMpeData = pd.read_excel(JRVMpe_path, header=None)
JRVMceData = pd.read_excel(JRVMce_path, header=None)
GCNpeData = pd.read_excel(GCNpe_path, header=None)
GCNceData = pd.read_excel(GCNce_path, header=None)

JGCNx = [i for i in range(win, len(JGCNpeData) + win)]
JRVMx = [i for i in range(0, len(JRVMpeData))]
GCNx = [i for i in range(win, len(GCNpeData) + win)]
axes1_args = [0.07, 0.085, 0.9, 0.85]
axes2_args = [0.5, 0.25, 0.3, 0.2]

xlabel = "采样时间/h"
ylabel1 = "青霉素浓度/g/h"
ylabel2 = "菌体浓度/g/h"
ylabel3 = "误差"

fig1 = plt.figure()
lfig1, peerraxes = plt.subplots(1, 1)
fig2 = plt.figure()
lfig2, ceerraxes = plt.subplots(1, 1)

pepredaxes = fig1.add_axes(axes1_args)
lpepredaxes = fig1.add_axes(axes2_args)
perect = patches.Rectangle((190, 0.78), 30, 0.25, linewidth=2, edgecolor='k', facecolor='none')
cepredaxes = fig2.add_axes(axes1_args)
lcepredaxes = fig2.add_axes(axes2_args)
cerect = patches.Rectangle((190, 10), 30, 2, linewidth=2, edgecolor='k', facecolor='none')
lJGCNx = JGCNx[190-win: 190-win+30]
lGCNx = GCNx[190-win: 190-win+30]
lJRVMx = JRVMx[190: 190+30]

config_figure(pepredaxes, xlabel, ylabel1, [0, length], [0, 1.4], font1)
config_figure(cepredaxes, xlabel, ylabel2, [0, length], [0, 13], font1)
config_figure(peerraxes, xlabel, ylabel3, [0, length], [-0.12, 0.08], font1)
config_figure(ceerraxes, xlabel, ylabel3, [0, length], [-0.7, 0.7], font1)

pepredaxes.add_patch(perect)
pepredaxes.plot(JGCNx, JGCNpeData[0], 'b-', label="Real Value") # 真实值
pepredaxes.plot(JGCNx, JGCNpeData[1], 'r--', label="JITL-GCN")
pepredaxes.plot(JRVMx, JRVMpeData[0], 'g-.', label="JITL-RVM")
pepredaxes.plot(GCNx, GCNpeData[0], 'm:', label="GCN")
lpepredaxes.plot(lJGCNx, JGCNpeData[0][190-win: 190-win+30], 'b-', label="Real Value")
lpepredaxes.plot(lJGCNx, JGCNpeData[1][190-win: 190-win+30], 'r--', label="JITL-GCN")
lpepredaxes.plot(lJRVMx, JRVMpeData[0][190: 190+30], 'g-.', label="JITL-RVM")
lpepredaxes.plot(lGCNx, GCNpeData[0][190-win: 190-win+30], 'm:', label="GCN")
perect_x1 = perect.get_x()
perect_x2 = perect.get_x() + perect.get_width()
perect_y = perect.get_y() + perect.get_height()
lpepredaxes_x1 = (axes2_args[0] - axes1_args[0]) * length / axes1_args[2]
lpepredaxes_x2 = (axes2_args[0] + axes2_args[2] - axes1_args[0]) * length / axes1_args[2]
lpepredaxes_y = (axes2_args[1] + axes2_args[3] - axes1_args[1]) * 1.4 / axes1_args[3]
pepredaxes.plot((perect_x1, lpepredaxes_x1), (perect_y, lpepredaxes_y), 'k-')
pepredaxes.plot((perect_x2, lpepredaxes_x2), (perect_y, lpepredaxes_y), 'k-')

cepredaxes.add_patch(cerect)
cepredaxes.plot(JGCNx, JGCNceData[0], 'b-', label="Real Value") # 真实值
cepredaxes.plot(JGCNx, JGCNceData[1], 'r--', label="JITL-GCN")
cepredaxes.plot(JRVMx, JRVMceData[0], 'g-.', label="JITL-RVM")
cepredaxes.plot(GCNx, GCNceData[0], 'm:', label="GCN")
lcepredaxes.plot(lJGCNx, JGCNceData[0][190-win: 190-win+30], 'b-', label="Real Value") # 真实值
lcepredaxes.plot(lJGCNx, JGCNceData[1][190-win: 190-win+30], 'r--', label="JITL-GCN")
lcepredaxes.plot(lJRVMx, JRVMceData[0][190: 190+30], 'g-.', label="JITL-RVM")
lcepredaxes.plot(lGCNx, GCNceData[0][190-win: 190-win+30], 'm:', label="GCN")
cerect_x1 = cerect.get_x()
cerect_x2 = cerect.get_x() + cerect.get_width()
cerect_y = cerect.get_y() + cerect.get_height()
lcepredaxes_x1 = (axes2_args[0] - axes1_args[0]) * length / axes1_args[2]
lcepredaxes_x2 = (axes2_args[0] + axes2_args[2] - axes1_args[0]) * length / axes1_args[2]
lcepredaxes_y = (axes2_args[1] + axes2_args[3] - axes1_args[1]) * 13 / axes1_args[3]
cepredaxes.plot((cerect_x1, lcepredaxes_x1), (cerect_y, lcepredaxes_y), 'k-')
cepredaxes.plot((cerect_x2, lcepredaxes_x2), (cerect_y, lcepredaxes_y), 'k-')


peerraxes.plot(JGCNx, JGCNpeData[2], 'r--*', label="JITL-GCN")
peerraxes.plot(JRVMx, JRVMpeData[1], 'g-.>', label="JITL-RVM")
peerraxes.plot(GCNx, GCNpeData[1], 'm:o', label="GCN")

ceerraxes.plot(JGCNx, JGCNceData[2], 'r--*', label="JITL-GCN")
ceerraxes.plot(JRVMx, JRVMceData[1], 'g-.>', label="JITL-RVM")
ceerraxes.plot(GCNx, GCNceData[1], 'm:o', label="GCN")

pepredaxes.legend(prop=font2) # 图例字体
pepredaxes.tick_params(labelsize=18) # 坐标刻度
labels = pepredaxes.get_xticklabels() + pepredaxes.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
lines = pepredaxes.get_lines() # 设置线宽
[line.set_linewidth(2) for line in lines]
cepredaxes.legend(prop=font2)
cepredaxes.tick_params(labelsize=18)
labels = cepredaxes.get_xticklabels() + cepredaxes.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
lines = cepredaxes.get_lines()
[line.set_linewidth(2) for line in lines]
peerraxes.legend(prop=font2)
peerraxes.tick_params(labelsize=18)
labels = peerraxes.get_xticklabels() + peerraxes.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
ceerraxes.legend(prop=font2)
ceerraxes.tick_params(labelsize=18)
labels = ceerraxes.get_xticklabels() + ceerraxes.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
lpepredaxes.tick_params(labelsize=15)
labels = lpepredaxes.get_xticklabels() + lpepredaxes.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
lcepredaxes.tick_params(labelsize=15)
labels = lcepredaxes.get_xticklabels() + lcepredaxes.get_yticklabels()
[label.set_fontname('Times New Roman') for label in labels]
[fig.set_size_inches(10, 6) for fig in [fig1, lfig1, fig2, lfig2]] # 图片尺寸
fig1.savefig("picture\Pepred.svg", dpi=600, bbox_inches='tight')
lfig1.savefig("picture\Cepred.svg", dpi=600, bbox_inches='tight')
fig2.savefig("picture\Peerror.svg", dpi=600, bbox_inches='tight')
lfig2.savefig("picture\Ceerror.svg", dpi=600, bbox_inches='tight')
plt.show()