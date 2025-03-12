import torch
from torch import nn
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def get_dataset(data_file_path, select_channels, label):
    """获取数据集，维度[20000, 16]"""
    data = pd.read_excel(data_file_path, header=0, index_col=None, usecols=select_channels) # 忽略采样序号和热流速度
    target_data = torch.tensor(data[label].values).reshape(-1, 1) # torch.Size([20000, 1])
    input_data = torch.tensor(data.drop(label, axis=1).values) # torch.Size([20000, 15])
    dataset = torch.cat([target_data, input_data], 1) # torch.Size([20000, 16])
    return dataset.to(dtype=torch.float32) # 确定好精度

def batch_iter(dataset, batch_size):
    """产生批数据，维度[]"""
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

def get_similar_data(dataset, batch_size, win_size, per_batch, curret_data):
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
        for i in range(per_batch):
            new_index = index + i * win_size
            similar_data.append(data[new_index: new_index+win_size].t())
            similar_label.append(labels[new_index+win_size-1].unsqueeze(dim=0))
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

def model_train(net, X, y, lr, wd, num_epochs, per_batch, device):
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
        for j in range(per_batch):
            net.train()
            optimizer.zero_grad()
            index = [k * per_batch + j for k in range(len(X) // per_batch)]
            y_model = net(X[index])
            l = loss(y_model, y[index].reshape(y_model.shape))
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

def just_intime_learning(net, hist_dataset, test_dataset, batch_size, win_size, lr, num_epochs, per_batch, device):
    """即时学习+图卷积网络"""
    error_list = []
    ypred_list = []
    yreal_list = []
    i = 0
    for data in create_data_win(test_dataset, win_size):
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
    win_size = 8
    lr = 0.25
    wd = 2.7
    num_epochs = 15
    per_batch = 1

    # 加载历史数据数据集
    history_data_path = 'data\pensimdata_full_variable_50_batch_本科.xlsx'
    test_data_path = 'data\pensimdata_full_variable_50_batch_本科2.xlsx'
    select_channels = "D:I,K:N,P,Q"
    label = "产物浓度"
    hist_dataset = get_dataset(history_data_path, select_channels, label)
#    test_dataset = get_dataset(test_data_path, select_channels, label)[0: batch_size]
    test_dataset = get_dataset(test_data_path, select_channels, label)
    curret_sample = test_dataset[22*400+52: 22*400+52+win_size]
    print("当前样本：", curret_sample)

    # 邻接矩阵
    mic_path = 'data\mic_result.xlsx'
#    drop_label = ["曝气率", "搅拌速率", "底物流加温度", "溶解氧浓度", "产物浓度", "培养液体积", "PH值", "反应罐温度"]
    drop_label = ["曝气率", "搅拌速率", "产物浓度", "培养液体积"]
    adjmatrix = get_adjmatrix(mic_path, drop_label)
    n_nodes = len(adjmatrix)

    # 模型
    net = nn.Sequential(GCNConv2d(adjmatrix), 
                        nn.Linear(win_size, 16), 
                        nn.Dropout(), 
                        nn.ReLU(), 
                        GCNConv2d(adjmatrix), 
                        nn.Linear(16, 16), 
                        nn.Dropout(),  
                        nn.ReLU(), 
                        nn.Flatten(),
                        nn.Linear(n_nodes * 16, 16), 
                        nn.Dropout(), 
                        nn.ReLU(), 
                        nn.Linear(16, 1)
    )
    
    # 损失图
    loss_fig, loss_axes = plt.subplots(1, 1, figsize=(6, 4))
    config_figure(loss_axes, 'epoch', 'loss', [1, num_epochs], [0, 0.5])
#    _, axes = plt.subplots(1, 1, figsize=(6, 4))
#    config_figure(axes, 'time', None, [win_size-1, batch_size-1], [-2.5, 2.5])
    # 标准化
    data_mean = hist_dataset.mean(dim=0)
    data_std = hist_dataset.std(dim=0)
    hist_dataset = (hist_dataset - data_mean) / (data_std + 1e-5)
#    test_dataset = (test_dataset - data_mean) / (data_std + 1e-5)
    curret_sample = (curret_sample - data_mean) / (data_std + 1e-5)
    print("当前归一化样本：", curret_sample)
    
    # 获取历史相似数据
    index_list, similar_data, similar_label = get_similar_data(hist_dataset, batch_size, win_size, per_batch, curret_sample[-1, 1:])
    print("相似数据索引：", index_list)
    print("训练集大小：", similar_data.shape)

    # 数据变形
    curret_data = curret_sample[:, 1:].t().unsqueeze(dim=0)
    curret_label = curret_sample[-1, 0].unsqueeze(dim=0)

    x_list, loss_list = model_train(net, similar_data, similar_label, lr, wd, num_epochs, per_batch, 'cuda:0')
    print("train loss: ", loss_list[-1])
    loss_axes.plot(x_list, loss_list, label='loss')
    error, y_model = model_test_gpu(net, curret_data, curret_label)
    print('relavalue: ', curret_label)
    print('prediction: ', y_model)
    print('error: ', error)
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