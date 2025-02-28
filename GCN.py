import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
import math


class GraphConvolution(nn.Module):
    """
    支持batch处理的改进版图卷积层
    """

    def __init__(self, in_features, out_features, bias=True):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input, adj):
        """
        input: (batch_size, n_nodes, in_features)
        adj: (n_nodes, n_nodes)
        """
        batch_size, n_nodes, _ = input.shape

        # 计算归一化邻接矩阵: D^(-1/2) * A * D^(-1/2)
        degree = torch.sum(adj, dim=1)
        degree_inv_sqrt = degree.pow(-0.5)
        # 防止除0
        degree_inv_sqrt[torch.isinf(degree_inv_sqrt)] = 0
        adj_norm = adj * degree_inv_sqrt[:, None]
        adj_norm = adj_norm * degree_inv_sqrt[None, :]

        # 直接使用torch.matmul进行批量矩阵乘法
        # input: (batch_size, n_nodes, in_features) 乘以 weight: (in_features, out_features)
        support = torch.matmul(input, self.weight)  # (batch_size, n_nodes, out_features)
        # 归一化邻接矩阵为 (n_nodes, n_nodes) 会自动广播到batch维度
        output = torch.matmul(adj_norm, support)
        if self.bias is not None:
            output += self.bias.unsqueeze(0).unsqueeze(0)

        return output


class GCNGRU(nn.Module):
    """
    完整GCN+GRU模型
    """

    def __init__(self, n_feat, gcn_hidden, gru_hidden, n_nodes, adj_matrix, dropout=0.2):
        super(GCNGRU, self).__init__()
        # 固定对称邻接矩阵注册为buffer
        self.register_buffer('adj', adj_matrix)

        # GCN层堆叠
        self.gc1 = GraphConvolution(n_feat, gcn_hidden)
        self.gc2 = GraphConvolution(gcn_hidden, gcn_hidden)

        # GRU时序处理
        self.gru = nn.GRU(
            input_size=n_nodes * gcn_hidden,  # GRU输入为节点数×GCN隐藏维度
            hidden_size=gru_hidden,
            batch_first=True,
            num_layers=2,
            dropout=dropout
        )

        # 输出层
        self.fc = nn.Sequential(
            nn.Linear(gru_hidden, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # 输出最终预测值
        )

        self.dropout = dropout

    def forward(self, x):
        """
        输入尺寸: (batch, time_steps, n_nodes, n_feat)
        输出尺寸: (batch, 1)
        """
        batch_size, time_steps = x.size(0), x.size(1)

        # 各时间步GCN处理
        gcn_outputs = []
        for t in range(time_steps):
            x_t = x[:, t, :, :]  # (batch, nodes, feat)

            # 两层GCN
            h = F.relu(self.gc1(x_t, self.adj))
            h = F.dropout(h, self.dropout, training=self.training)
            h = F.relu(self.gc2(h, self.adj))

            # 展平节点特征 (batch, nodes*hidden)
            h_flat = h.view(batch_size, -1)
            gcn_outputs.append(h_flat.unsqueeze(1))  # 每个时间步的输出

        # 构建GRU输入序列 (batch, time, features)
        gru_input = torch.cat(gcn_outputs, dim=1)

        # GRU处理
        gru_out, _ = self.gru(gru_input)

        # 最终预测：使用GRU输出的最后一个时间步
        output = self.fc(gru_out[:, -1, :])  # 使用最后一个时间步的输出
        return output


# 测试数据和模型
if __name__ == "__main__":
    # 模拟数据
    n_nodes = 10  # 假设有10个节点（青霉素发酵过程的变量数目）
    time_steps = 50  # 假设有50个时间步
    input_dim = 5  # 每个节点的特征维度
    hidden_dim = 16
    output_dim = 8  # GCN层输出的维度
    gru_hidden_dim = 32  # GRU的隐藏层维度
    dropout = 0.5
    upper_tri = torch.randint(0, 2, (n_nodes, n_nodes))
    upper_tri = torch.triu(upper_tri, diagonal=1)
    # 对称化
    adj_matrix = upper_tri + upper_tri.t() # 保证邻接矩阵对称

    # 模拟输入数据
    x_data = torch.randn(32, time_steps, n_nodes, input_dim)  # 假设batch_size=32, 随机生成数据
    y_data = torch.randn(32, 1)  # 假设我们只预测一个目标变量

    # 构建模型
    model = GCNGRU(input_dim, hidden_dim, gru_hidden_dim, n_nodes, adj_matrix, dropout)

    # 前向传播
    output = model(x_data)

    print(f"Model output shape: {output.shape}")  # 输出的形状应该是 (batch_size, 1)

    # 假设我们计算损失
    criterion = nn.MSELoss()
    loss = criterion(output, y_data)
    print(f"Loss: {loss.item()}")
