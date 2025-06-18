import torch
from torch import nn
from torch.nn import functional as F 
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean

def get_dataset(data_file_path, select_channels, label):
    """获取数据集，维度[20000, 16]"""
    data = pd.read_excel(data_file_path, header=0, index_col=None, usecols=select_channels) # 忽略采样序号和热流速度
    target_data = torch.tensor(data[label].values).reshape(-1, 1) # torch.Size([20000, 1])
    input_data = torch.tensor(data.drop(label, axis=1).values) # torch.Size([20000, 15])
    dataset = torch.cat([target_data, input_data], 1) # torch.Size([20000, 16])
    return dataset.to(dtype=torch.float32) # 确定好精度

def batch_iter(dataset, batch_size):
    """产生批数据"""
    for i in range((len(dataset) + batch_size - 1) // batch_size):
        index = i * batch_size
        if index + batch_size > len(dataset)-1:
            yield dataset[index:, 1:], dataset[index:, 0] # torch.Size([batch_size, 15]) torch.Size([batch_size])
        else:
            yield dataset[index: index+batch_size, 1:], dataset[index: index+batch_size, 0]

def create_data_win(data, win_size): # (改)
    """产生窗口数据"""
    win_data_list = []
    for i in range(int(len(data) / win_size)):
        win_data_list.append(data[i*win_size: (i+1)*win_size])
    return torch.stack(win_data_list, dim=0)

def get_distance(data, curret_sample): # （改）
    """计算窗口样本的距离之和"""
    close_fn = torch.abs(curret_sample - data)
    close_fn = torch.exp(-1* close_fn)
    close_fn = close_fn.sum(dim=-1) / data.shape[-1]
    distance = torch.sum(1 / (close_fn + 1e-5) -1, dim=-1)
    return distance

def data_normal(data, dim=0, keepdim=False, bias=1e-6):
    """数据标准化"""
    data_mean = data.mean(dim=dim, keepdim=keepdim)
    data_std = data.std(dim=dim, keepdim=keepdim) + bias
    return (data - data_mean) / data_std, data_mean, data_std

def get_similar_data(dataset, curret_data): # （改）
    """获取历史相似样本"""
    # 计算距离
    distance = get_distance(dataset, curret_data)
    indexs = distance.argsort()
    return indexs[0: int(len(indexs) * 0.025)]

def graphcov(X, adjmatrix):
    """图卷积"""
    degree = adjmatrix.sum(dim=1)
    degree_inv_sqrt = degree.pow(-0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
    adjmatrix_norm = adjmatrix * degree_inv_sqrt.reshape(-1, 1)
    adjmatrix_norm = adjmatrix_norm * degree_inv_sqrt
    return torch.matmul(adjmatrix_norm, X)

class GCNCov(nn.Module): # (改)
    """图卷积块"""
    def __init__(self, in_features, out_features, adjmatrix, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('adjmatrix', adjmatrix)

    def forward(self, X):
        Y = graphcov(X, self.adjmatrix)
        Y = self.linear(Y)
        Y = F.relu(Y)
        return self.dropout(Y)
    
class GCNGRU(nn.Module): # (改)
    """GCN+GRU"""
    def __init__(self, in_features, varible_num, hidden_features, out_features, num_layers, adjmatrix, dropout=0.2):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.gcn1 = GCNCov(in_features, in_features, adjmatrix, dropout)
        self.gcn2 = GCNCov(in_features, in_features, adjmatrix, dropout)
        self.gru = nn.GRU(varible_num, hidden_features, num_layers, 
                          batch_first=True, dropout=dropout)
        self.fc =nn.Linear(hidden_features, out_features)

    def forward(self, X, states):
        Y = self.gcn2(self.gcn1(X)) + X
        Y = Y.transpose(1, 2) # [batch_size, win_size, hidden_features]
        Y, _ = self.gru(Y, states)
        Y = Y.reshape((-1, Y.shape[-1]))  # [batch_size*win_size, hidden_features]
        Y = self.fc(Y)
        return Y
    
    def init_states(self, device, batch_size):
        return torch.zeros((self.num_layers, batch_size, 
                            self.hidden_features), device=device)

def grad_clipping(net, theta):
    """裁剪梯度"""
    if isinstance(net, nn.Module):
        params = [p for p in net.parameters() if p.requires_grad]
    else:
        params = net.params
    # 这里的范数是所有参数的范数之和
    norm = torch.sqrt(sum(torch.sum((p.grad ** 2)) for p in params))
    if norm > theta:
        for param in params:
            param.grad[:] *= theta / norm # param.grad[:]原位修改
    
def config_figure(axes, xlabel, ylabel, xlim, ylim, xscale='linear', yscale='linear'):
    """设置图片属性"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.grid()

def model_train(net, X, y, lr, wd, num_epochs, device):
    """训练模型"""
    x_list = []
    y_list = []
    # 权重初始化
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    # 将模型参数和数据移到GPU
    print('tarining on:', device)
    net.to(device)
    X, y = X.to(device), y.to(device)
    # 定义优化函数和损失函数
    optimizer = torch.optim.SGD(net.parameters(), lr, weight_decay=wd)
    loss = nn.MSELoss(reduction='mean')
    # 训练
    for i in range(num_epochs):
        state = net.init_states(device, len(X))
        net.train()
        optimizer.zero_grad()
        y_model = net(X, state)
        l = loss(y_model, y.reshape(y_model.shape))
        l.backward()
#        grad_clipping(net, 1)
        optimizer.step()
        x_list.append(i+1)
        y_list.append(l.item()) # tensor--->int
    return x_list, y_list

def model_test_gpu(net, X, y,  device=None):
    """计算预测误差"""
    print('test on: ', next(iter(net.parameters())).device)
    if isinstance(net, nn.Module):
        net.eval() # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    with torch.no_grad():
        X = X.to(device)
        y = y.to(device)
        state = net.init_states(device, 1)
        y_model = net(X, state)
        y_pred = y_model.reshape(-1)[-1]
        error = y_pred - y
    return error, y_pred

def just_intime_learning(hist_dataset, test_dataset, batch_size, 
                         win_size, n_nodes, gru_hidden, 
                         gru_layers, adjmatrix, dropout,
                         lr, num_epochs, per_batch, device):
    """即时学习+图卷积网络"""
    error_list = []
    ypred_list = []
    yreal_list = []
    i = 0
    for data in create_data_win(test_dataset, win_size):
        net = GCNGRU(win_size, n_nodes, gru_hidden, 1, gru_layers, adjmatrix, dropout)
        index_list, similar_data, similar_label = get_similar_data(hist_dataset, batch_size, win_size, per_batch, data[:, 1:])
        print('sample: ', i)
        model_train(net, similar_data, similar_label, lr, wd, num_epochs, per_batch, device)
        test_data = data[:, 1:].t().unsqueeze(dim=0)
        test_label = data[-1, 0]
        error, y_model = model_test_gpu(net, test_data, test_label)
        error_list.append(error)
        ypred_list.append(y_model.item())
        yreal_list.append(test_label.item())
        i += 1
    error = torch.cat(error_list, 0)
    y_pred = torch.tensor(ypred_list)
    y_real = torch.tensor(yreal_list)
    rmse = torch.sqrt((error ** 2).sum() / error.shape[0])
    r2 = 1 - (error ** 2).sum() / ((y_pred - y_real.mean()) ** 2).sum()
    return rmse, r2, y_real, y_pred

def get_adjmatrix(mic_path, label):
    """获取邻接矩阵"""
    mic = pd.read_excel(mic_path, header=0, index_col=0)
    mic = np.array(mic.drop(index=label, columns=label))
    adjmatrix = np.zeros(mic.shape)
    adjmatrix[mic > 0.5] = 1
    adjmatrix = torch.tensor(adjmatrix, dtype=torch.float32)
    print("邻接矩阵：", adjmatrix)
    return adjmatrix

if __name__ == '__main__':
    # 超参数
    batch_size = 400
    win_size = 4 # 5
    lr = 0.1 # 0.1
    wd = 0
    num_epochs = 200
    dropout = 0 # 0
    gru_hidden = 128 # 128
    gru_layers = 2

    # 加载历史数据数据集
    history_data_path = 'data\pensimdata_full_variable_50_batch_本科.xlsx'
    test_data_path = 'data\pensimdata_full_variable_50_batch_本科2.xlsx'
#    select_channels = "D:I,K:N,P,Q"
#    select_channels = "B:N,P:Q"
#    select_channels = "B:E,G,I:N,P:Q" #（改）
    select_channels = "B:F,G,J:N,P:Q"
#    label = "产物浓度"
    label = "底物浓度"
    hist_dataset = get_dataset(history_data_path, select_channels, label)[0: 20 * batch_size] # (改)
    test_dataset = get_dataset(test_data_path, select_channels, label)
    curret_sample = test_dataset[2*400+0: 2*400+0+win_size]

    # 邻接矩阵
    mic_path = 'data\mic_result.xlsx'
#    drop_label = ["曝气率", "搅拌速率", "底物流加温度", "溶解氧浓度", "产物浓度", "培养液体积", "PH值", "反应罐温度"]
    drop_label = ["底物浓度", "菌体浓度", "产物浓度"] #（改）
#    drop_label = label
    adjmatrix = get_adjmatrix(mic_path, drop_label)
    n_nodes = len(adjmatrix)

    # 损失图
    loss_fig, loss_axes = plt.subplots(1, 1, figsize=(6, 4))
    config_figure(loss_axes, 'epoch', 'loss', [1, num_epochs], [0, 1.5])
#    _, axes = plt.subplots(1, 1, figsize=(6, 4))
#    config_figure(axes, 'time', None, [win_size-1, batch_size-1], [-2.5, 2.5])

    # 标准化 （改）
    hist_dataset_norm, hist_dataset_mean, hist_dataset_std = data_normal(hist_dataset[:, 1:], dim=0)
    curret_data_norm = (curret_sample[:, 1:] - hist_dataset_mean) / hist_dataset_std
    print("历史数据均值:",hist_dataset_mean)
    # 窗口数据 （改）
    win_histdataset = create_data_win(hist_dataset, win_size)
    win_histdata_norm = (win_histdataset[:, :, 1:] - hist_dataset_mean) / hist_dataset_std
    print("窗口数据集大小：", win_histdataset.shape)
    # 获取历史相似数据 （改）
    indexs = get_similar_data(win_histdata_norm, curret_data_norm)
    similar_data = win_histdataset[:, :, 1:][indexs].transpose(-2, -1)
    similar_label = win_histdataset[:, :, 0][indexs].reshape(-1)
    print("相似数据索引：", indexs)
    print("训练集大小：", similar_data.shape, similar_label.shape)
#    print(similar_data[0], similar_label[0 : win_size])

    # 数据变形 (改)
    curret_data = curret_sample[:, 1:].t()
    curret_label = curret_sample[:, 0]

    # 数据标准化 （改）
    similar_data_norm, similar_data_mean, similar_data_std = data_normal(similar_data, (0, -1), True)
    similar_label_norm, similar_label_mean, similar_label_std = data_normal(similar_label)
    curret_data_norm = (curret_data - similar_data_mean) / similar_data_std
    curret_label_norm = (curret_label - similar_label_mean) / similar_label_std
#    print("相似样本均值：", similar_data_mean)
    print("标准化测试样本大小：", curret_data_norm.shape, curret_label_norm.shape)

    # 模型
    new_net = GCNGRU(win_size, n_nodes, gru_hidden, 1, gru_layers, adjmatrix, dropout)

    # 训练
    x_list, loss_list = model_train(new_net, similar_data_norm, similar_label_norm, lr, wd, num_epochs, 'cuda:0') # (改)
    print("train loss: ", loss_list[-1])
    loss_axes.plot(x_list, loss_list, label='loss')

    # 测试
    error, y_model = model_test_gpu(new_net, curret_data_norm, curret_label_norm[-1]) # (改)
    print('relavalue: ', curret_label[-1])
    print('prediction: ', y_model.cpu() * similar_label_std + similar_label_mean) # (改)
    print('error: ', error.cpu() * similar_label_std) # (改)
    loss_axes.legend()
#    rmse, r2, y_real, y_pred = just_intime_learning(net, hist_dataset, test_dataset, batch_size, win_size, lr, num_epochs, per_batch, 'cuda:0')
#    print('RMSE: ', rmse, 'R2: ', r2)
#    print('real value: ', y_real)
#    print('predict value: ', y_pred)
#    x = np.arange(win_size - 1, batch_size)
#    axes.plot(x, y_real.detach().numpy(), 'b-', label="real value")
#    axes.plot(x, y_pred.detach().numpy(), 'r-', label='predict value')
#    axes.legend()
    plt.show()