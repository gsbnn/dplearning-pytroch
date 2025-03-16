import torch
from torch import nn
from d2l import torch as d2l
from test import load_data_fashion_mnist
from test import train_ch3

# 定义模型
# PyTorch不会隐式地调整输入的形状。因此，
# 我们在线性层前定义了展平层（flatten），来调整网络输入的形状
net = nn.Sequential(nn.Flatten(), nn.Linear(784, 10))

# 初始化参数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std = 0.01)
net.apply(init_weights)

# nn.CrossEntropyLoss()的input和traget
# input:单个样本(C)，批样本(batch_size, C)，高维批样本(batch_size, C, d1, d2,..., dk)
# traget:类别索引or类别概率
# 类别索引：单个样本()，批样本(batch_size)，高维样本(batch_size, d1, d2,..., dk)；值：[0, C)
# 类别概率:维度与input一致；值(0~1)
loss = nn.CrossEntropyLoss(reduction='none')

# 优化器
trainer = torch.optim.SGD(net.parameters(), lr=0.1)

# 设置超参数
num_epochs = 10
batch_size = 256

if __name__ == '__main__':
    train_iter, test_iter = load_data_fashion_mnist(batch_size) # 加载数据
    train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer) # 训练