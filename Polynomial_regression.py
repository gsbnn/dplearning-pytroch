import math
import numpy as np
import torch
from torch import nn
from d2l import torch as d2l
import test

max_degree = 20
n_train, n_test = 100, 100 # 训练和测试数据集大小
true_w = np.zeros(max_degree)
true_w[0:4] = np.array([5, 1.2, -3.4, 5.6])

# 构造数据集
features = np.random.normal(size=(n_train + n_test, 1)) 
np.random.shuffle(features)
poly_features = np.power(features, np.arange(max_degree).reshape(1, -1))
for i in range(max_degree):
    poly_features[:, i] /= math.gamma(i + 1) # gamma(n) = (n-1)! 样本特征：(n_train + n_test, max_degree)
labels = np.dot(poly_features, true_w)
labels += np.random.normal(scale=0.1, size=labels.shape) # 样本标签(n_train + n_test)

# ndarray转换成tensor
true_w, features, poly_features, labels = [torch.tensor(x, dtype=torch.float32) 
                                           for x in [true_w, features, poly_features, labels]]

def evaluate_loss(net, data_iter, loss):
    """评估给定数据集上模型的损失"""
    metric = d2l.Accumulator(2)
    for X, y in data_iter:
        out = net(X)
        y = y.reshape(out.shape)
        l = loss(out, y)
        metric.add(l.sum(), l.numel())
    return metric[0] / metric[1]

# 定义训练函数
def train(train_features, test_features, train_labels, test_labels, 
          num_epochs=400):
    loss = nn.MSELoss(reduction='none')
    input_shape = train_features.shape[-1] # 控制模型参数维度
    net = nn.Sequential(nn.Linear(input_shape, 1, bias=False))
    batch_size = min(10, train_labels.shape[0])
    train_iter = d2l.load_array((train_features, train_labels.reshape(-1, 1)), 
                                batch_size)
    test_iter = d2l.load_array((test_features, test_labels.reshape(-1, 1)), 
                                batch_size, is_train=False)
    trainer = torch.optim.SGD(net.parameters(), lr=0.01)
    #animator = test.Animator(xlabel = 'epoch', ylabel='loss', yscale='log', 
    #                    xlim=[1, num_epochs], ylim=[1e-3, 1e2], 
    #                    legned=['train', 'test'])
    for epoch in range(num_epochs):
        test.train_epoch_ch3(net, train_iter, loss, trainer)
    #    if epoch == 0 or (epoch + 1) % 20 == 0:
    #        animator.add(epoch + 1, (evaluate_loss(net, train_iter, loss), 
    #                                 evaluate_loss(net, test_iter, loss)))
    return evaluate_loss(net, train_iter, loss), evaluate_loss(net, test_iter, loss)
    # print('weight:', net[0].weight.data.numpy())
    # d2l.plt.show()

animator = test.Animator(xlabel = 'order of poly', ylabel='loss', yscale='log',
                        xlim=[4, max_degree], ylim=[1e-3, 1e-1], 
                        legned=['train', 'test'])
for i in range(4, max_degree+1):
    train_loss, test_loss = train(poly_features[:n_train, :i], poly_features[n_train:, :i], 
            labels[:n_train], labels[n_train:], num_epochs=1500)
    animator.add(i, (train_loss, test_loss))

d2l.plt.show()