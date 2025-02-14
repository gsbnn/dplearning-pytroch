import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from IPython import display
from d2l import torch as d2l
from matplotlib import animation

def get_dataloader_workers(): 
    """使用4个进程来读取数据"""
    return 4
  
def load_data_fashion_mnist(batch_size, resize=None):
    """下载Fashion-MNIST数据集，然后将其加载到内存中"""
    # torchvision.transform模块功能如下：
    # 1.支持PIL Image or ndarray和tensor之间相互转换
    # 2.支持tensor的dtype转换

    # torchvision.transforms.ToTensor()转换结果如下：
    # 1.图片数据（data）：tensor，torch.size([C,H,W])，C为通道数
    # 2.图片标签（labels）：int类型
    trans = [transforms.ToTensor()] 
    if resize:
        trans.insert(0, transforms.Resize(resize))
    trans = transforms.Compose(trans)

    
    # 根据定义的转换规则trans，将数据转换到mnist_train和mnist_test
    # 具体过程为，索引数据时，__getitem__()输出为：tuple(tensor(C, H, W), int)
    mnist_train = torchvision.datasets.FashionMNIST( 
        root="data", train=True, transform=trans, download=True
    )
    mnist_test = torchvision.datasets.FashionMNIST(
        root="data", train=False, transform=trans, download=True
    )

    # data.DataLoader()接收tuple类型
    # 并将tuple中每个元素按batch_size打包成一个tensor
    # 最后将每个元素的tensor装进一个列表
    # 数据类型变化如下：
    # tuple(tensor, int)--->list[tensor, tensor]
    # 数据维度变化如下：
    # tuple(data(C, H, W), labels(None))--->list[data(batch_size, C, H, W), labels(batch_size)]
    return (data.DataLoader(mnist_train, batch_size, shuffle=True, 
                            num_workers=get_dataloader_workers()), 
            data.DataLoader(mnist_test, batch_size, shuffle=False, 
                            num_workers=get_dataloader_workers()))

def softmax(X):
    # (batch_size, q)--->(batch_size, q)
    X_exp = torch.exp(X) # (batch_size, q)
    partition = X_exp.sum(dim=1, keepdim=True) # (batch_size, 1)
    return X_exp / partition # 运用广播机制

def net(X):
    # (batch_size, C, H, w)--->(batch_size, q)
    return softmax(torch.matmul(X.reshape(-1, W.shape[0]), W) + b) # (batch_size, q)

def corss_entropy(y_hat, y):
    # (batch_size, q)--->(batch_size)
    return - torch.log(y_hat[range(len(y_hat)), y])
# return - torch.log(y_hat[:, y])

def accuracy(y_hat, y):
    """计算预测正确的数量"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = y_hat.argmax(axis=1)
    cmp = y_hat.type(y.dtype) == y
    return float(cmp.type(y.dtype).sum())

def evaluuate_accuracy(net, data_iter):
    """计算在指定数据集上模型的精度"""
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        for X, y in data_iter:
            metric.add(accuracy(net(X), y), y.numel())
    return metric[0] / metric[1]

class Accumulator():
    """在n个变量上累加"""
    def __init__(self, n):
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]
    
    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
    

def updater(batch_size):
    return d2l.sgd([W, b], lr, batch_size)

def train_epoch_ch3(net, train_iter, loss, updater):
    """训练模型一个迭代周期"""
    if isinstance(net, torch.nn.Module):
        net.train()
    
    # 训练损失总和、训练准确度总和、样本数
    metric = Accumulator(3)
    for X, y in train_iter:
        # 计算梯度并更新参数
        y_hat = net(X)
        l = loss(y_hat, y)
        if isinstance(updater, torch.optim.Optimizer):
            # 使用PyTorch内置的优化器和损失函数
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        else:
            # 使用定制的优化器和损失函数
            l.sum().backward()
            updater(X.shape[0])
        metric.add(float(l.sum()), accuracy(y_hat, y), y.numel())
    # 返回平均训练损失和训练精度    
    return metric[0] / metric[2], metric[1] / metric[2]

class Animator():
    """在动画中绘制数据"""
    def __init__(self, xlabel=None, ylabel=None, legned=None, xlim=None, 
                 ylim=None, xscale='linear', yscale='linear', 
                 fmts=('-', 'm--', 'g-.', 'r:'), nrows=1, ncols=1, 
                 figsize=(3.5, 2.5)):
        # 增量地绘制多条线
        if legned is None:
            legned = []
        self.fig, self.axes = d2l.plt.subplots(nrows, ncols, figsize=figsize)
        if nrows * ncols == 1:
            self.axes = [self.axes, ]
        # 使用lambda函数捕获参数
        self.config_axes = lambda: d2l.set_axes(self.axes[0], xlabel, ylabel, xlim, ylim, 
                                                xscale, yscale, legned)
        self.X, self.Y, self.fmts = None, None, fmts
    
    def add(self, x, y):
        # 向图表中添加多个数据点
        if not hasattr(y, "__len__"):
            y = [y]
        n = len(y)
        if not hasattr(x, "__len__"):
            x = [x] * n
        if not self.X:
            self.X = [[] for _ in range(n)]
        if not self.Y:
            self.Y = [[] for _ in range(n)]
        for i, (a, b) in enumerate(zip(x, y)):
            if a is not None and b is not None:
                self.X[i].append(a)
                self.Y[i].append(b)
        self.axes[0].cla() # Clear the Axes
        for x, y, fmt in zip(self.X, self.Y, self.fmts):
            self.axes[0].plot(x, y, fmt)
        self.config_axes()
        # display.display(self.fig)
        # display.clear_output(wait=True)

def train_ch3(net, train_iter, test_iter, loss, num_epochs, updater):
    """训练模型"""
    animator = Animator(xlabel='epoch', xlim=[1, num_epochs], ylim=[0.3, 0.9],
                        legned=['train loss', 'train acc', 'test acc'])
    for epoch in range(num_epochs):
        train_metrics = train_epoch_ch3(net, train_iter, loss, updater)
        test_acc = evaluuate_accuracy(net, test_iter)
        animator.add(epoch + 1, train_metrics + (test_acc,))
    d2l.plt.show()

def predict_ch3(net, test_iter, n=6):
    for X, y in test_iter:
        break
    trues = d2l.get_fashion_mnist_labels(y)
    preds = d2l.get_fashion_mnist_labels(net(X).argmax(axis=1))
    print(trues[0:n])
    print(preds[0:n])

num_inputs = 784
num_outputs = 10
batch_size = 256
num_epochs = 10
lr = 0.1
W = torch.normal(0, 0.01, size=(num_inputs, num_outputs), requires_grad=True) # 注意开启计算图记录
b = torch.zeros(num_outputs, requires_grad=True) # b是一个一维Tensor

if __name__=='__main__':
    train_iter, test_iter = load_data_fashion_mnist(batch_size)
    train_ch3(net, train_iter, test_iter, corss_entropy, num_epochs, updater)
    predict_ch3(net, test_iter)