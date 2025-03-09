import torch
from torch import nn
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def get_dataset(data_file_path, select_channels, label):
    """获取数据集"""
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

def create_data_win(data, win_size):
    """产生窗口数据"""
    for i in range(len(data) - win_size + 1):
        yield data[i: i+win_size]

def get_distance(data, curret_sample):
    """计算窗口样本的距离之和"""
    close_fn = torch.abs(curret_sample - data)
    close_fn = torch.exp(-1* close_fn)
    close_fn = close_fn.sum(dim=1) / data.shape[1]
    distance = torch.sum(1 / (close_fn + 1e-5) -1, dim=0, keepdim=True)
    return distance

def get_similar_data(dataset, batch_size, win_size, curret_data):
    """获取历史相似样本"""
    similar_data = []
    similar_label = []
    index_list = []
    for data, labels in batch_iter(dataset, batch_size):
        distance_list = []
        # 计算距离
        for win_data in create_data_win(data, win_size):
            distance = get_distance(win_data, curret_data)
            distance_list.append(distance)
        distances = torch.cat(distance_list)
        # 获取相似数据
        index = distances.argmin(dim=0) # 每个批次的最近似窗口数据
        similar_data.append(data[index: index+win_size].t())
        similar_label.append(labels[index+win_size-1].unsqueeze(dim=0))
        index_list.append(index)
        # 整理成tensor
        data = torch.stack(similar_data, dim=0)
        labels = torch.cat(similar_label)
    return index_list, data, labels # data[batch_size, nodes, in_feature], labels[batch_size]

def GraphCov(X, adjmatrix):
    """图卷积"""
    degree = adjmatrix.sum(dim=1)
    degree_inv_sqrt = degree.pow(-0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
    adjmatrix_norm = adjmatrix * degree_inv_sqrt.reshape(-1, 1)
    adjmatrix_norm = adjmatrix_norm * degree_inv_sqrt
    return torch.matmul(adjmatrix_norm, X)

class GCNConv2d(nn.Module):
    """图卷积层"""
    def __init__(self, adjmatrix):
        super().__init__()
        self.register_buffer('adjmatrix', adjmatrix)
    def forward(self, X):
        Y = GraphCov(X, self.adjmatrix)
        return Y
    
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
    bias_params = [p for name, p in net.named_parameters() if 'weight' in name]
    others = [p for name, p in net.named_parameters() if 'weight' not in name]
    optimizer = torch.optim.SGD([{'params': bias_params, 'weight_decay': wd},  # 使用权重衰减
                                 {'params': others}], lr=lr)
    loss = nn.MSELoss()
    # 训练
    for i in range(num_epochs):
        net.train()
        optimizer.zero_grad()
        y_model = net(X)
        l = loss(y_model, y.reshape(y_model.shape))
        l.backward()
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
        y_model = net(X)
        error = y_model - y.reshape(y_model.shape)
    return error, y_model

def just_intime_learning(net, dataset, test_data, label, batch_size, win_size, lr, num_epochs, device):
    """即时学习+图卷积网络"""
    error_list = []
    for data in test_data:
        index_list, similar_data, similar_label = get_similar_data(dataset, batch_size, win_size, data)
        epoch_list, loss_list = model_train(net, similar_data, similar_label, lr, num_epochs, device)
        error = model_test_gpu(net, test_data, label)
        error_list.append(error)
    rmse = torch.cat(error_list, 0)
    rmse = torch.sqrt((rmse ** 2).sum() / rmse.shape[0])
    return rmse

def get_adjmatrix(mic_path, label):
    """获取邻接矩阵"""
    mic = pd.read_excel(mic_path, header=0, index_col=0)
    mic = np.array(mic.drop(index=label, columns=label))
    adjmatrix = np.zeros(mic.shape)
    adjmatrix[mic > 0.4] = 1
    adjmatrix = torch.tensor(adjmatrix, dtype=torch.float32)
    print("邻接矩阵：", adjmatrix)
    return adjmatrix

if __name__ == '__main__':
    # 超参数
    batch_size = 400
    win_size = 4
    lr = 0.01
    wd = 1
    num_epochs = 50

    # 加载历史数据数据集
    history_data_path = 'data\pensimdata_full_variable_50_batch_本科.xlsx'
    test_data_path = 'data\pensimdata_full_variable_50_batch_本科2.xlsx'
    select_channels = "B:N,P:Q"
    label = "产物浓度"
    hist_dataset = get_dataset(history_data_path, select_channels, label)
    test_dataset = get_dataset(test_data_path, select_channels, label)
    curret_sample = test_dataset[0*400+222: 0*400+222+win_size]
    print("当前样本：", curret_sample)

    # 邻接矩阵
    mic_path = 'data\mic_result.xlsx'
    adjmatrix = get_adjmatrix(mic_path, label)
    n_nodes = len(adjmatrix)

    # 模型
    net = nn.Sequential(GCNConv2d(adjmatrix), nn.Linear(win_size, win_size), 
                        nn.BatchNorm1d(n_nodes), nn.ReLU(), 
                        GCNConv2d(adjmatrix), nn.Linear(win_size, win_size), 
                        nn.BatchNorm1d(n_nodes), nn.ReLU(), 
                        GCNConv2d(adjmatrix), nn.Linear(win_size, win_size), 
                        nn.BatchNorm1d(n_nodes), nn.ReLU(), 
                        nn.Flatten(),
                        nn.Linear(n_nodes * win_size, 128), 
                        nn.BatchNorm1d(128), 
                        nn.ReLU(), 
                        nn.Linear(128, 1))
    
    # 损失图
    loss_fig, loss_axes = plt.subplots(1, 1, figsize=(6, 4))
    config_figure(loss_axes, 'epoch', 'loss', [1, num_epochs], [0, 0.5])

    # 标准化
    data_mean = hist_dataset.mean(dim=0)
    data_std = hist_dataset.std(dim=0)
    hist_dataset = (hist_dataset - data_mean) / (data_std + 1e-5)
    curret_sample = (curret_sample - data_mean) / (data_std + 1e-5)
    print("特征均值：", data_mean)
    print("当前归一化样本：", curret_sample)
    
    # 获取历史相似数据
    index_list, similar_data, similar_label = get_similar_data(hist_dataset, batch_size, win_size, curret_sample[-1, 1:])
    print("相似样本索引：", index_list)

    # 训练
    curret_data = curret_sample[:, 1:].t().unsqueeze(dim=0)
    curret_label = curret_sample[-1, 0].unsqueeze(dim=0)
    x_list, loss_list = model_train(net, similar_data, similar_label, lr, wd, num_epochs, 'cuda:0')
    print("train loss: ", loss_list[-1])
    loss_axes.plot(x_list, loss_list, label='loss')
    error, y_model = model_test_gpu(net, curret_data, curret_label)
    print('relavalue: ', curret_label)
    print('prediction: ', y_model)
    print('error: ', error)
    loss_axes.legend()
    plt.show()