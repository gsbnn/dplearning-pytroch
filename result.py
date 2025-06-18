import pandas as pd
from matplotlib import pyplot as plt
from pylab import mpl

# 设置显示中文字体
mpl.rcParams["font.sans-serif"] = ["SimHei"]
# 解决中文字体下坐标轴负数的负号显示问题
plt.rcParams['axes.unicode_minus'] = False 

win = 4
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

_, pepredaxes = plt.subplots(1, 1)
_, peerraxes = plt.subplots(1, 1)
_, cepredaxes = plt.subplots(1, 1)
_, ceerrdaxes = plt.subplots(1, 1)

pepredaxes.plot(JGCNx, JGCNpeData[0], 'b-', label="Real Value") # 真实值
pepredaxes.plot(JGCNx, JGCNpeData[1], 'r--', label="JITL-GCN")
pepredaxes.plot(JRVMx, JRVMpeData[0], 'g-.', label="JITL-RVM")
pepredaxes.plot(GCNx, GCNpeData[0], 'm:', label="GCN")
pepredaxes.set_title("青霉素浓度预测曲线")
pepredaxes.set_xlabel("采样时间/h")
pepredaxes.set_ylabel("青霉素浓度/g/h")

cepredaxes.plot(JGCNx, JGCNceData[0], 'b-', label="Real Value") # 真实值
cepredaxes.plot(JGCNx, JGCNceData[1], 'r--', label="JITL-GCN")
cepredaxes.plot(JRVMx, JRVMceData[0], 'g-.', label="JITL-RVM")
cepredaxes.plot(GCNx, GCNceData[0], 'm:', label="GCN")
cepredaxes.set_title("菌体浓度预测曲线")
cepredaxes.set_xlabel("采样时间/h")
cepredaxes.set_ylabel("菌体浓度/g/h")

peerraxes.plot(JGCNx, JGCNpeData[2], 'r--*', label="JITL-GCN")
peerraxes.plot(JRVMx, JRVMpeData[1], 'g-.>', label="JITL-RVM")
peerraxes.plot(GCNx, GCNpeData[1], 'm:o', label="GCN")
peerraxes.set_title("青霉素浓度预测误差曲线")
peerraxes.set_xlabel("采样时间/h")
peerraxes.set_ylabel("误差")

ceerrdaxes.plot(JGCNx, JGCNceData[2], 'r--*', label="JITL-GCN")
ceerrdaxes.plot(JRVMx, JRVMceData[1], 'g-.>', label="JITL-RVM")
ceerrdaxes.plot(GCNx, GCNceData[1], 'm:o', label="GCN")
ceerrdaxes.set_title("菌体浓度预测误差曲线")
ceerrdaxes.set_xlabel("采样时间/h")
ceerrdaxes.set_ylabel("误差")

pepredaxes.legend()
cepredaxes.legend()
peerraxes.legend()
ceerrdaxes.legend()
plt.show()