import torch
from torch import nn
from d2l import torch as d2l
import test

dropout1, dropout2 = 0.2, 0.5
# 定义模型
net = nn.Sequential(nn.Flatten(),
                    nn.Linear(784, 256), 
                    nn.ReLU(), 
                    nn.Dropout(dropout1), 
                    nn.Linear(256, 256), 
                    nn.ReLU(), 
                    nn.Dropout(dropout2), 
                    nn.Linear(256, 10))
# 初始化模型参数
def init_weights(m):
    if type(m) == nn.Linear:
        nn.init.normal_(m.weight, std=0.01)

net.apply(init_weights)

num_epochs, lr, batch_size = 10, 0.5, 256

# 定义损失函数
loss = nn.CrossEntropyLoss(reduction='none')

# 定义优化算法
trainer = torch.optim.SGD(net.parameters(), lr=lr)

# 训练
if __name__ == '__main__':
    train_iter, test_iter = test.load_data_fashion_mnist(batch_size)
    test.train_ch3(net, train_iter, test_iter, loss, num_epochs, trainer)