import torch
from torch import nn
from test import load_data_fashion_mnist
from test import train_ch3
from d2l import torch as d2l



# 初始化模型参数
num_inputs, num_outputs, num_hiddens = 784, 10, 256
W1 = nn.Parameter(torch.randn(
    num_inputs, num_hiddens, requires_grad=True) * 0.01)
b1 = nn.Parameter(torch.zeros(num_hiddens, requires_grad=True))
W2 = nn.Parameter(torch.randn(
    num_hiddens, num_outputs, requires_grad=True) * 0.01)
b2 = nn.Parameter(torch.zeros(num_outputs, requires_grad=True))

param = [W1, b1, W2, b2]

# 定义激活函数ReLU
def relu(X):
    a = torch.zeros_like(X)
    return torch.max(X, a)

# 定义模型
def net(X):
    X = X.reshape(-1, num_inputs)
    H = relu(X@W1 + b1)
    return (H@W2 + b2)

# 定义损失函数
loss = nn.CrossEntropyLoss(reduction='none')

# 训练 
num_epochs, lr = 10, 0.1
updater = torch.optim.SGD(param, lr=lr)
batch_size = 256

if __name__ == '__main__':
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    train_ch3(net, train_iter, test_iter, loss, num_epochs, updater)
