import torch
import torchvision
from torch.utils import data
from torchvision import transforms
from d2l import torch as d2l

def get_fashion_mnist_labels(labels):
    """返回Fashion-MNIST数据集的文本标签"""
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels] # 返回一个列表

def show_images(imgs, num_rows, num_cols, titles=None, scale=1.5):
    """绘制图像列表"""
    figsize = (num_cols * scale, num_rows * scale)
    _, axes = d2l.plt.subplots(num_rows, num_cols, figsize=figsize) # subplots()将Axes作为(num_rows, num_cols)的ndarray返回
    axes = axes.flatten() # 将二维降至一维
    for i, (ax, img) in enumerate(zip(axes, imgs)):
        if torch.is_tensor(img):
            ax.imshow(img.numpy()) # 绘制图片
        else:
            # PIL图片
            ax.imshow(img)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if titles:
            ax.set_title(titles[i])
    return axes

#X, y = next(iter(data.DataLoader(mnist_train, batch_size=18))) # data.DataLoader()调用mnist_train.__getitem__()进行取值，因此X是的维度是(18, 1, 28, 28)
#axes = show_images(X.reshape(18, 28, 28), 2, 9, titles=get_fashion_mnist_labels(y))
#d2l.plt.show()



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

if __name__ == '__main__':
    train_iter, test_iter = load_data_fashion_mnist(32, resize=64)
    train_data = next(iter(train_iter))
    print(train_data[0].shape, train_data[1].shape)
    train_data_dim = train_data[0].reshape(-1, train_data[0].shape[2]**2) # reshape机制？
    print(train_data_dim.shape)


#for X, y in train_iter:
#    print(X.shape, X.dtype, y.shape, y.dtype)
#    break