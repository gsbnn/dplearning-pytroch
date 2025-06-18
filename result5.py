import numpy as np
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
'size'   : 20,
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

def plot_fig(fig, axes_args, X, Y, xlabel, ylabel, xlim, ylim, 
             cfont, efont, labels, linewidth=2, title="", 
             fmts=('b-', 'r--'), ticklabelszie=18, 
             xscale='linear', yscale='linear'):
    """画多条曲线"""
    fig.set_size_inches(10, 6)
    axes = fig.add_axes(axes_args)
    config_figure(axes, xlabel, ylabel, xlim, ylim, cfont)
    for x, y, label, fmt in zip(X, Y, labels, fmts):
        axes.plot(x, y, fmt, linewidth=linewidth, label=label)
    axes.legend(prop=efont) # 图例字体
    axes.tick_params(labelsize=ticklabelszie) # 坐标刻度
    ticklabels = axes.get_xticklabels() + axes.get_yticklabels()
    [ticklabel.set_fontname(efont['family']) for ticklabel in ticklabels]
    return axes

def create_fig(dpi, path, axes_args, X, Y, xlabel, ylabel, xlim, ylim, 
             cfont, efont, labels, linewidth=2, title="", 
             fmts=('b-', 'r--'), ticklabelszie=18, 
             xscale='linear', yscale='linear'):
    fig = plt.figure()
    axes = plot_fig(fig, axes_args, X, Y, xlabel, ylabel, xlim, ylim, 
                    cfont, efont, labels, linewidth=linewidth, title=title, 
                    fmts=fmts, ticklabelszie=ticklabelszie, 
                    xscale=xscale, yscale=yscale)
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    return fig, axes

win = 4
length = 400

JGCNpe_path = "data\JITL_GCN_青霉素浓度.xlsx"
JGCNce_path = "data\JITL_GCN_菌体浓度.xlsx"
JRVMpe_path = "data\JITL_RVM_青霉素浓度.xlsx"
JRVMce_path = "data\JITL_RVM_菌体浓度.xlsx"
GCNpe_path = "data\GCN_青霉素浓度.xlsx"
GCNce_path = "data\GCN_菌体浓度.xlsx"

JGCNpeData = np.array(pd.read_excel(JGCNpe_path, header=None)).T
JGCNceData = np.array(pd.read_excel(JGCNce_path, header=None)).T
JRVMpeData = np.array(pd.read_excel(JRVMpe_path, header=None)).T
JRVMceData = np.array(pd.read_excel(JRVMce_path, header=None)).T
GCNpeData = np.array(pd.read_excel(GCNpe_path, header=None)).T
GCNceData = np.array(pd.read_excel(GCNce_path, header=None)).T
x1 = [[i for i in range(win, JGCNpeData.shape[1] + win)]]
x2 = [[i for i in range(0, JRVMpeData.shape[1])]]
x3 = [[i for i in range(win, GCNpeData.shape[1] + win)]]
JGCNx = np.array(x1 * 2)
JRVMx = np.array(x2 * 2)
GCNx = np.array(x3 * 2)
errorx = x1 + x2 + x3
errory1 = [JGCNpeData[-1], JRVMpeData[-1], GCNpeData[-1]]
errory2 = [JGCNceData[-1], JRVMceData[-1], GCNceData[-1]]
axes1_args = [0.07, 0.085, 0.9, 0.85]
axes2_args = [0.5, 0.25, 0.3, 0.2]
    
xlabel = "采样时间/h"
ylabel1 = "青霉素浓度/g/h"
ylabel2 = "菌体浓度/g/h"
ylabel3 = "误差"



JGCNpe = create_fig(600, 'picture\JGCNPepred.svg', axes1_args, JGCNx, JGCNpeData[0: 2], 
                    xlabel, ylabel1, [0, length], [0, 1.4], font1, font2, 
                    labels=["Real Value", "JITL-GCN"])

JGCNce = create_fig(600, 'picture\JGCNCepred.svg', axes1_args, JGCNx, JGCNceData[0: 2], 
                    xlabel, ylabel2, [0, length], [0, 13], 
                    font1, font2, labels=["Real Value", "JITL-GCN"])

JRVMpe = create_fig(600, 'picture\JRVMPepred.svg', axes1_args, JRVMx, JRVMpeData[0: 2], 
                    xlabel, ylabel1, [0, length], [0, 1.4], font1, font2, 
                    labels=["Real Value", "JITL-RVM"], fmts=('b-', 'g-'))

JRVMce = create_fig(600, 'picture\JRVMCepred.svg', axes1_args, JRVMx, JRVMceData[0: 2], 
                    xlabel, ylabel2, [0, length], [0, 13], 
                    font1, font2, labels=["Real Value", "JITL-RVM"], fmts=('b-', 'g-'))

GCNpe = create_fig(600, 'picture\GCNPepred.svg', axes1_args, GCNx, GCNpeData[0: 2], 
                    xlabel, ylabel1, [0, length], [0, 1.4], font1, font2, 
                    labels=["Real Value", "GCN"], fmts=('b-', 'm:'))

GCNce = create_fig(600, 'picture\GCNCepred.svg', axes1_args, GCNx, GCNceData[0: 2], 
                    xlabel, ylabel2, [0, length], [0, 13], 
                    font1, font2, labels=["Real Value", "GCN"], fmts=('b-', 'm:'))

errorpe = create_fig(600, 'picture\errorpe.svg', axes1_args, errorx, errory1, 
                     xlabel, ylabel3, [0, length], [-0.12, 0.08], 
                     font1, font2, labels=["JITL-GCN", "JITL-RVM", "GCN"], fmts=('r--*', 'g-.>', "m:o"))

errorce = create_fig(600, 'picture\errorce.svg', axes1_args, errorx, errory2, 
                     xlabel, ylabel3, [0, length], [-0.7, 0.7], 
                     font1, font2, labels=["JITL-GCN", "JITL-RVM", "GCN"], fmts=('r--*', 'g-.>', "m:o"))
plt.show()