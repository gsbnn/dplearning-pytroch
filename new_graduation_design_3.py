import torch
import gc
from torch import nn
from torch.nn import functional as F 
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

def data_normal(data, dim=0, keepdim=False, bias=1e-6):
    """数据标准化"""
    data_mean = data.mean(dim=dim, keepdim=keepdim)
    data_std = data.std(dim=dim, keepdim=keepdim) + bias
    return (data - data_mean) / data_std, data_mean, data_std

def get_similar_data(dataset, batch_size, win_size, curret_data):
    """获取历史相似样本"""
    similar_data = []
    similar_label = []
    index_list = []
    for data, labels in batch_iter(dataset, batch_size):
        distance_list = []
        # 数据标准化
        data_norm, data_mean, data_std = data_normal(data, 0)
        curret_data_norm = (curret_data - data_mean) / data_std
        # 计算距离
        for win_data in create_data_win(data_norm, win_size):
            distance = get_distance(win_data, curret_data_norm)
            distance_list.append(distance)
        distances = torch.cat(distance_list)
        # 获取相似数据
        indexs = distances.argsort(dim=0)[0: min_batch] # 每个批次的最近似窗口数据
        index_min = indexs.min()
        index_max = indexs.max()
        for i in range(100):
            index = index_min + i * win_size
            if index + win_size <=  index_max + win_size:
                similar_data.append(data[index: index + win_size].t())
                similar_label.append(labels[index : index + win_size])
            else:
                break
        index_list.append((index_min.item(), index_max.item() + win_size))
    # 整理成tensor
    data = torch.stack(similar_data, dim=0)
    labels = torch.cat(similar_label)
    return index_list, data, labels # data[batch_size, nodes, in_feature], labels[batch_size*win_size]

def graphcov(X, adjmatrix):
    """图卷积"""
    degree = adjmatrix.sum(dim=1)
    degree_inv_sqrt = degree.pow(-0.5)
    degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
    adjmatrix_norm = adjmatrix * degree_inv_sqrt.reshape(-1, 1)
    adjmatrix_norm = adjmatrix_norm * degree_inv_sqrt
    return torch.matmul(adjmatrix_norm, X)

class GCNCov(nn.Module):
    """图卷积块"""
    def __init__(self, in_features, out_features, adjmatrix, dropout=0.2):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer('adjmatrix', adjmatrix)

    def forward(self, X):
        Y = graphcov(X, self.adjmatrix)
        Y = self.linear(Y)
        # 残差连接
        Y = F.relu(Y + X)
        return self.dropout(Y)
    
class GCNGRU(nn.Module):
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
        Y = self.gcn2(self.gcn1(X))
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

def model_train(net, optimizer, loss, X, y, num_epochs, device):
    """训练模型"""
    x_list = []
    y_list = []
    # 权重初始化
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    # 将模型参数和数据移到GPU
#    print('tarining on:', device)
    net.to(device)
    X, y = X.to(device), y.to(device)
    # 训练
    for i in range(num_epochs):
        state = net.init_states(device, len(X))
        net.train()
        optimizer.zero_grad()
        y_model = net(X, state)
        l = loss(y_model, y.reshape(y_model.shape))
        l.backward()
        optimizer.step()
        x_list.append(i+1)
        y_list.append(l.item()) # tensor--->int
    return x_list, y_list

def model_test_gpu(net, X, y, device=None):
    """计算预测误差"""
#    print('test on: ', next(iter(net.parameters())).device)
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
                         lr, wd, num_epochs, device):
    """即时学习+图卷积网络"""
    error_list = []
    ypred_list = []
    yreal_list = []
    i = 0
    for data in create_data_win(test_dataset, win_size):
        print('sample: ', i)
        # 获取相似样本
        index_list, similar_data, similar_label = get_similar_data(hist_dataset, batch_size, win_size, data[:, 1:])
        # 数据标准化
        test_data = data[:, 1:].t().unsqueeze(dim=0)
        test_label = data[:, 0]
        similar_data_norm, similar_data_mean, similar_data_std = data_normal(similar_data, (0, 2), True)
        similar_label_norm, similar_label_mean, similar_label_std = data_normal(similar_label)
        test_data_norm = (test_data - similar_data_mean) / similar_data_std
        test_label_norm = (test_label - similar_label_mean) / similar_label_std
        # 模型
        net = GCNGRU(win_size, n_nodes, gru_hidden, 1, gru_layers, adjmatrix, dropout)
        optimizer = torch.optim.SGD(net.parameters(), lr, weight_decay=wd)
        loss = nn.MSELoss()
        # 训练
        model_train(net, optimizer, loss, similar_data_norm, similar_label_norm, 
                    num_epochs, device)
        # 测试
        error, y_model = model_test_gpu(net, test_data_norm, test_label_norm[-1])
        error_list.append(error.cpu().item() * similar_label_std)
        ypred_list.append(y_model.cpu().item() * similar_label_std + similar_label_mean)
        yreal_list.append(test_label[-1].item())
        i += 1
        # 丢弃模型
        del net, optimizer, loss
        gc.collect()
        torch.cuda.empty_cache()
    error = torch.tensor(error_list)
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
    win_size = 5
    lr = 0.1 # 0.002
    wd = 0.1 # 0
    num_epochs = 200 # 100
    dropout = 0 # 0
    gru_hidden = 128 # 128
    gru_layers = 2

    # 加载历史数据数据集
    history_data_path = 'data\pensimdata_full_variable_50_batch_本科.xlsx'
    test_data_path = 'data\pensimdata_full_variable_50_batch_本科2.xlsx'
#    select_channels = "B:E,G,I:N,P:Q" #（改, 产物浓度）
    select_channels = "B:E,G:H,J:N,P:Q" #（改, 菌体浓度）
#    select_channels = "B:F,G,J:N,P:Q" #（改, 底物浓度）
#    label = "产物浓度"
    label = "菌体浓度" # (改, 菌体浓度)
#    label = "底物浓度" # (改, 底物浓度)
    hist_dataset = get_dataset(history_data_path, select_channels, label)
    test_dataset = get_dataset(test_data_path, select_channels, label)[9*batch_size: 9*batch_size+batch_size]
    min_batch = 1


    # 邻接矩阵
    mic_path = 'data\mic_result.xlsx'
    drop_label = ["底物浓度", "菌体浓度", "产物浓度"] #（改）
    adjmatrix = get_adjmatrix(mic_path, drop_label)
    n_nodes = len(adjmatrix)

    # 损失图
    _, axes = plt.subplots(1, 2, figsize=(6, 4))
    config_figure(axes[0], 'Sampling time\h', 'Penicillin concentration/(g/L)', [win_size-1, batch_size-1], [0, 14.0])
    config_figure(axes[1], 'Sampling time\h', 'Error/(g/L)', [win_size-1, batch_size-1], [-1.0, 1.0])

    rmse, r2, y_real, y_pred, error = just_intime_learning(hist_dataset, test_dataset, batch_size, 
                         win_size, n_nodes, gru_hidden, 
                         gru_layers, adjmatrix, dropout,
                         lr, wd, num_epochs, 'cuda:0')
    print('RMSE: ', rmse, 'R2: ', r2)
    print('real value: ', y_real)
    print('predict value: ', y_pred)
    print('error: ', error)
    x = np.arange(win_size - 1, batch_size)
    axes[0].plot(x, y_real.detach().numpy(), 'b-', label='real value')
    axes[0].plot(x, y_pred.detach().numpy(), 'r-', label='predict value')
    axes[1].plot(x, error.detach().numpy(), 'b-')
    axes[0].legend()
    plt.show()
    # Y = torch.cat((y_real.reshape(-1, 1), y_pred.reshape(-1, 1), error.reshape(-1, 1)), dim=1)
    # Y = pd.DataFrame(Y.detach().numpy())
    # Y.to_excel('data\JITL_GCN_Output.xlsx', index=False, header=False)