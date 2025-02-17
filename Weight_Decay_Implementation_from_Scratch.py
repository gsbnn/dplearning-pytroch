import torch
from torch import nn
from d2l import torch as d2l
from test import Animator

# 生成数据
n_train, n_test, num_inputs, batch_size = 20, 100, 200, 5
true_w, true_b = torch.ones(num_inputs, 1) * 0.01, 0.05
train_data = d2l.synthetic_data(true_w, true_b, n_train) # X(n_train, num_inputs) y(n_train, 1)
train_iter = d2l.load_array(train_data, batch_size)
test_data = d2l.synthetic_data(true_w, true_b, n_test)
test_iter = d2l.load_array(test_data, batch_size, is_train=False)

# 初始化模型参数，将参数装载list中
def init_params():
    w = torch.normal(0, 1, size=(num_inputs, 1), requires_grad=True)
    b = torch.zeros(1, requires_grad=True)
    return [w, b]

# 定义L2范数惩罚
def l2_penalty(w):
    return torch.sum(w.pow(2)) / 2

def train(lambd):
    w, b = init_params()
    net, loss = lambda X: d2l.linreg(X, w, b), d2l.squared_loss
    num_epochs, lr = 100, 0.003
    animator = Animator(xlabel='epochs  lambda=' + str(lambd), ylabel='loss', yscale='log', 
                        xlim=[5, num_epochs], legned=['train', 'test'])
    for epoch in range(num_epochs):
        for X, y in train_iter:
            # 增加L2范数惩罚项
            l = loss(net(X), y) + lambd * l2_penalty(w) # (batch_size, 1)
            l.sum().backward()
            d2l.sgd([w, b], lr, batch_size)
        if(epoch + 1) % 5 == 0:
            animator.add(epoch + 1, (d2l.evaluate_loss(net, train_iter, loss), 
                                     d2l.evaluate_loss(net, test_iter, loss)))

train(lambd=0)
train(lambd=3)
d2l.plt.show()