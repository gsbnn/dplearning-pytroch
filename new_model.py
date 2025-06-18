import torch
import gc
from torch import nn
from torch.nn import functional as F
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

def get_dataset(data_file_path, select_channels, label):
    """读取数据集"""
    data = pd.read_excel(data_file_path, header=0, index_col=None, usecols=select_channels)
    target_data = torch.tensor(data[label].values).reshape(-1, 1) # torch.Size([20000, 1])
    input_data = torch.tensor(data.drop(label, axis=1).values) # torch.Size([20000, 11])
    dataset = torch.cat([target_data, input_data], 1) # torch.Size([20000, 12])
    return dataset.to(dtype=torch.float32) # 确定好精度

def batch_iter(dataset, batch_size):
    """产生批数据"""
    for i in range((len(dataset) + batch_size - 1) // batch_size):
        index = i * batch_size
        if index + batch_size > len(dataset)-1:
            yield dataset[index:, 1:], dataset[index:, 0] # torch.Size([batch_size, 11]) torch.Size([batch_size])
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
    # 获取每个批次的相似样本
    for data, labels in batch_iter(dataset, batch_size):
        distance_list = []
        # 计算距离
        for win_data in create_data_win(data, win_size):
            distance = get_distance(win_data, curret_data)
            distance_list.append(distance)
        distances = torch.cat(distance_list)
        # 获取相似样本
        index = distances.argmin(dim=0) # 每个批次的最近似窗口数据
        for i in range(per_batch):
            new_index = index + i * win_size
            similar_data.append(data[new_index: new_index+win_size].t())
            similar_label.append(labels[new_index+win_size-1].unsqueeze(dim=0))
        index_list.append(index)
        # 整理
        data = torch.stack(similar_data, dim=0)
        labels = torch.cat(similar_label)
    return index_list, data, labels # data[批量, 节点, win_size（time_steps）], labels[批量]

def normalize_adj(adj, add_self_loop=True):
    if add_self_loop:
        adj = adj + torch.eye(adj.size(0), device=adj.device)
    degree = adj.sum(dim=1)
    degree_inv_sqrt = degree.pow(-0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
    adj = adj * degree_inv_sqrt.view(-1, 1)
    adj = adj * degree_inv_sqrt.view(1, -1)
    return adj

class GCNCov(nn.Module):
    """图卷积块 + 残差连接"""
    def __init__(self, in_features, out_features, adjmatrix, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.need_proj = in_features != out_features
        if self.need_proj:
            self.residual_proj = nn.Linear(in_features, out_features)
        adj_norm = normalize_adj(adjmatrix)
        self.register_buffer('adjmatrix_norm', adj_norm)

    def forward(self, X):
        out = torch.matmul(self.adjmatrix_norm, X)
        out = self.linear(out)
        # 残差连接
        res = self.residual_proj(X) if self.need_proj else X
        out = F.relu(out + res)
        return self.dropout(out)

class GCNGRU(nn.Module):
    """GCN + GRU"""
    def __init__(self, in_features, variable_num, hidden_features, out_features, num_layers, adjmatrix, dropout=0.2):
        super().__init__()
        self.in_features = in_features
        self.hidden_features = hidden_features
        self.out_features = out_features
        self.num_layers = num_layers
        self.gcn1 = GCNCov(in_features, in_features, adjmatrix, dropout)
        self.gcn2 = GCNCov(in_features, in_features, adjmatrix, dropout)
        self.gru = nn.GRU(variable_num, hidden_features, num_layers,
                          batch_first=True, dropout=dropout)
        self.fc =nn.Linear(hidden_features, out_features)


    def forward(self, X, states):
        Y = self.gcn2(self.gcn1(X))
        Y = Y.transpose(1, 2)  # [batch, time_steps, features]
        Y, _ = self.gru(Y, states)
        Y = self.fc(Y[:, -1, :])
        return Y

    def init_states(self, device, batch_size):
        return torch.zeros((self.num_layers, batch_size, self.hidden_features), device=device)
    
def config_figure(axes, xlabel, ylabel, xlim, ylim, xscale='linear', yscale='linear'):
    """设置图片属性"""
    axes.set_xlabel(xlabel)
    axes.set_ylabel(ylabel)
    axes.set_xlim(xlim)
    axes.set_ylim(ylim)
    axes.set_xscale(xscale)
    axes.set_yscale(yscale)
    axes.grid()

def model_train(net, optimizer, loss, X, y, num_epochs, per_batch, device):
    """训练模型"""
    x_list = []
    loss_list = []
    # 权重初始化
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    # 将模型参数和数据移到GPU
    net.to(device)
    X, y = X.to(device), y.to(device)
    # 训练
    for i in range(num_epochs):
        state = net.init_states(device, 50)
        for j in range(per_batch):
            net.train()
            optimizer.zero_grad()
            index = [k * per_batch + j for k in range(len(X) // per_batch)]
            y_model = net(X[index], state)
            l = loss(y_model, y[index].reshape(y_model.shape))
            l.backward()
            optimizer.step()
        x_list.append(i+1)
        loss_list.append(l.item()) # tensor--->int
    return x_list, loss_list

def model_test_gpu(net, X, y, device=None):
    """计算预测误差"""
    if isinstance(net, nn.Module):
        net.eval() # 设置为评估模式
        if not device:
            device = next(iter(net.parameters())).device
    with torch.no_grad():
        X = X.to(device)
        y = y.to(device)
        state = net.init_states(device, 1)
        y_model = net(X, state)
        error = y_model - y.reshape(y_model.shape)
    return error, y_model

def just_intime_learning(hist_dataset, test_dataset, batch_size, 
                         win_size, n_nodes, gru_hidden, 
                         gru_layers, adjmatrix, dropout,
                         lr, wd, num_epochs, per_batch, device):
    """即时学习+图卷积网络"""
    error_list = []
    ypred_list = []
    yreal_list = []
    i = 0
    for data in create_data_win(test_dataset, win_size):
        # 每次创建新模型
        net = GCNGRU(win_size, n_nodes, gru_hidden, 1, gru_layers, adjmatrix, dropout)
        weight_params = [p for name, p in net.named_parameters() if 'weight' in name]
        others = [p for name, p in net.named_parameters() if 'weight' not in name]
        optimizer = torch.optim.SGD([{'params': weight_params, 'weight_decay': wd},  # 使用权重衰减
                                 {'params': others}], lr=lr)

        loss = nn.MSELoss()
        print('sample: ', i)
        index_list, similar_data, similar_label = get_similar_data(hist_dataset, batch_size, win_size, per_batch, data[:, 1:])
        model_train(net, optimizer, loss, similar_data, similar_label, num_epochs, per_batch, device)
        test_data = data[:, 1:].t().unsqueeze(dim=0)
        test_label = data[-1, 0]
        error, y_model = model_test_gpu(net, test_data, test_label)
        error_list.append(error)
        ypred_list.append(y_model.item())
        yreal_list.append(test_label.item())
        i += 1
        # 丢弃模型
        del net, optimizer, loss
        gc.collect()
        torch.cuda.empty_cache()
    # 统计预测值，真实值，误差，计算RMSE,R2
    error = torch.cat(error_list, 0).cpu()
    y_pred = torch.tensor(ypred_list)
    y_real = torch.tensor(yreal_list)
    rmse = torch.sqrt((error ** 2).sum() / error.shape[0])
    r2 = 1 - (error ** 2).sum() / ((y_pred - y_real.mean()) ** 2).sum()
    return rmse, r2, y_real, y_pred, error

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
    win_size = 5 # time_steps
    lr = 0.05
    wd = 0 # 权重衰减系数
    num_epochs = 200 # 训练轮数
    per_batch = 1 # 每个批次只选一个相似样本，共50个相似样本
    dropout = 0.5 
    gru_hidden = 64
    gru_layers = 2

    # 加载历史数据数据集，根据MIC结果，忽略："采样时刻", "曝气率", "搅拌速率", "培养液体积", "热流速度"这几个变量
    history_data_path = 'data\pensimdata_full_variable_50_batch_本科.xlsx'
    test_data_path = 'data\pensimdata_full_variable_50_batch_本科2.xlsx'
    select_channels = "B:E,G:H,J:N,P:Q" #（改, 菌体浓度）
#    select_channels = "D:I,K:N,P,Q"
#    label = "产物浓度"
    label = "菌体浓度" # (改, 菌体浓度)
    # 历史数据
    hist_dataset = get_dataset(history_data_path, select_channels, label)
    # 测试数据
    test_dataset = get_dataset(test_data_path, select_channels, label)[7*batch_size: 7*batch_size+batch_size]

    # 邻接矩阵
    mic_path = 'data\mic_result.xlsx'
#    drop_label = ["曝气率", "搅拌速率", "产物浓度", "培养液体积"]
#    drop_label = ["曝气率", "搅拌速率", "菌体浓度", "培养液体积"]
    drop_label = ["底物浓度", "菌体浓度", "产物浓度"] #（改）
    adjmatrix = get_adjmatrix(mic_path, drop_label)
    n_nodes = len(adjmatrix) # 节点数

    # 图像配置
    _, axes = plt.subplots(1, 2, figsize=(6, 4))
    config_figure(axes[0], 'time', 'Penicillin concentration/(g/L)', [win_size-1, batch_size-1], [0, 2.0])
    config_figure(axes[1], 'time', 'Error/(g/L)', [win_size-1, batch_size-1], [-0.2, 0.2])
    # 即时学习
    rmse, r2, y_real, y_pred, error = just_intime_learning(hist_dataset, test_dataset, batch_size, 
                         win_size, n_nodes, gru_hidden, 
                         gru_layers, adjmatrix, dropout,
                         lr, wd, num_epochs, per_batch, 'cuda:0')
    print('RMSE: ', rmse, 'R2: ', r2)
    print('real value: ', y_real)
    print('predict value: ', y_pred)
    x = np.arange(win_size - 1, batch_size)
    axes[0].plot(x, y_real.detach().numpy(), 'b-', label='real value')
    axes[0].plot(x, y_pred.detach().numpy(), 'r-', label='predict value')
    axes[1].plot(x, error.detach().numpy(), 'b-')
    axes[0].legend()
    plt.show()
    Y = torch.cat((y_real.reshape(-1, 1), y_pred.reshape(-1, 1), error.reshape(-1, 1)), dim=1)
    Y = pd.DataFrame(Y.detach().numpy())
    Y.to_excel('data\GCN_Output.xlsx', index=False, header=False)