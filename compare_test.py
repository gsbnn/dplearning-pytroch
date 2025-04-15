import torch
import gc
from torch import nn
from torch.utils import data
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

def get_win_data(dataset, win_size):
    """产生窗口数据"""
    win_list =[]
    win_label_list = []
    for i in range(len(dataset) - win_size + 1):
        win_list.append(dataset[i: i + win_size, 1:].t())
        win_label_list.append(dataset[i: i + win_size, 0])
    win_data = torch.stack(win_list)
    win_label = torch.stack(win_label_list)
    return win_data, win_label # [batch_size, n_nodes, time_steps] [batch_size, time_steps]

def data_normal(data, dim=0, keepdim=False, bias=1e-6):
    """数据标准化"""
    data_mean = data.mean(dim=dim, keepdim=keepdim)
    data_std = data.std(dim=dim, keepdim=keepdim) + bias
    return (data - data_mean) / data_std, data_mean, data_std

def create_data_iter(win_dataset, batch_size, workers=4, is_train=True):
    """创建数据迭代器"""
    dataset = data.TensorDataset(*win_dataset)
    return data.DataLoader(dataset, batch_size, shuffle=is_train, drop_last=True, num_workers=workers) 
# tuple([batch_size, n_nodes, time_steps], [batch_size, time_steps])

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
        Y = Y.transpose(1, 2) # [batch_size, time_steps, in_features]
        Y, _ = self.gru(Y, states) #[batch_size, time_steps, hidden_features]
        Y = self.fc(Y) # [batch_size, time_steps, out_features]
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

def model_train(net, train_iter, num_epochs, device):
    """训练模型"""
    x_list = []
    y_list = []
    # 权重初始化
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.xavier_uniform_(m.weight)
    net.apply(init_weights)
    loss = nn.MSELoss()
    optimizer = torch.optim.SGD(net.parameters(), lr, weight_decay=wd)
    # 将模型参数和数据移到GPU
#    print('tarining on:', device)
    net.to(device)
    # 训练
    for i in range(num_epochs):
        for X, y in train_iter:
            X, y = X.to(device), y.to(device)
            state = net.init_states(device, len(X))
            net.train()
            optimizer.zero_grad()
            y_model = net(X, state)
            l = loss(y_model, y.reshape(y_model.shape))
            l.backward()
            optimizer.step()
        x_list.append(i+1)
        y_list.append(l.cpu().item()) # tensor--->int
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
        state = net.init_states(device, len(X))
        y_model = net(X, state)
        y_pred = y_model.reshape(y.shape)[:, win_size-1]
        error = y_pred - y[:, win_size-1]
    return error.cpu(), y_pred.cpu(), y[:, win_size-1].cpu()

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
    lr = 0.002 # 0.002
    wd = 0 # 0
    num_epochs = 100 # 100
    dropout = 0 # 0
    gru_hidden = 256 # 128
    gru_layers = 2

    # 加载历史数据数据集
    history_data_path = 'data\pensimdata_full_variable_50_batch_本科3.xlsx'
    test_data_path = 'data\pensimdata_full_variable_50_batch_本科4.xlsx'
#    select_channels = "D:I,K:N,P,Q"
    select_channels = "B:N,P:Q"
    label = "产物浓度"
    hist_dataset = get_dataset(history_data_path, select_channels, label)
    test_dataset = get_dataset(test_data_path, select_channels, label)[34*batch_size: 34*batch_size+batch_size]
#    curret_sample = test_dataset[27*400+274: 27*400+274+win_size]
    min_batch = 1
#    batch = len(hist_dataset) // batch_size * min_batch

    # 邻接矩阵
    mic_path = 'data\mic_result.xlsx'
#    drop_label = ["曝气率", "搅拌速率", "底物流加温度", "溶解氧浓度", "产物浓度", "培养液体积", "PH值", "反应罐温度"]
#    drop_label = ["曝气率", "搅拌速率", "产物浓度", "培养液体积"]
    drop_label = label
    adjmatrix = get_adjmatrix(mic_path, drop_label)
    n_nodes = len(adjmatrix)

    # 构造数据集
    win_dataset = get_win_data(hist_dataset, win_size)
    train_iter = create_data_iter(win_dataset, 512, 4, True)
    test_data = get_win_data(test_dataset, win_size)

    # 模型
    net = GCNGRU(win_size, n_nodes, gru_hidden, 1, 2, adjmatrix, dropout)

    # 损失图
    _, axes = plt.subplots(2, 2, figsize=(6, 4))
    config_figure(axes[0, 0], 'time', 'Penicillin concentration/(g/L)', [win_size-1, batch_size-1], [0, 2.0])
    config_figure(axes[0, 1], 'time', 'Error/(g/L)', [win_size-1, batch_size-1], [-0.2, 0.2])
    config_figure(axes[1, 0], 'epochs', 'loss', [0, num_epochs], [-0.2, 0.2])

    # 训练
    epochs, loss = model_train(net, train_iter, num_epochs, 'cuda:0')
    # 预测
    error, y_pred, y_real = model_test_gpu(net, test_data[0], test_data[1], 'cuda:0')
    rmse = torch.sqrt((error ** 2).sum() / error.shape[0])
    r2 = 1 - (error ** 2).sum() / ((y_pred - y_real.mean()) ** 2).sum()
    print('RMSE: ', rmse, 'R2: ', r2)
    print('real value: ', y_real)
    print('predict value: ', y_pred)
    print('error: ', error)
    x = np.arange(win_size - 1, batch_size)
    axes[0, 0].plot(x, y_real.detach().numpy(), 'b-', label='real value')
    axes[0, 0].plot(x, y_pred.detach().numpy(), 'r-', label='predict value')
    axes[0, 1].plot(x, error.detach().numpy(), 'b-')
    axes[1, 0].plot(epochs, loss, 'b-')
    axes[0, 0].legend()
    plt.show()