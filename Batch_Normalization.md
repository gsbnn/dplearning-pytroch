# 批量归一化  

## 定义
从形式上来说，用$x\in \mathcal B$表示一个来自小批量$\mathcal B$的输入，批量规范化$\text{BN}$根据以下表达式转换$\bold x$：  
$$
\begin{align}
\text{BN}(\bold x) &=\pmb\gamma\odot\frac{\bold x - \pmb{\hat\mu}_\mathcal B}{\pmb{\hat\sigma}_\mathcal B}+\pmb\beta\\
\pmb{\hat\mu}_\mathcal B &=\frac{1}{|\mathcal B|}\sum_{\bold x\in\mathcal B}\bold x\\
\pmb{\hat\sigma}_\mathcal B^2 &=\frac{1}{|\mathcal B|}\sum_{\bold x\in\mathcal B}(\bold x - \pmb{\hat\mu}_\mathcal B)^2+\epsilon
\end{align}
$$
$\pmb{\hat\mu}_\mathcal B$是小批量$\mathcal B$的样本均值， $\pmb{\hat\sigma}_\mathcal B^2$是小批量$\mathcal B$的样本标准差，**$\pmb\gamma$和$\pmb\beta$是需要与其他模型参数一起学习的参数**，它们的形状与样本相同。批量规范化将每一层主动居中，并将它们重新调整为给定的平均值和大小（通过$\pmb{\hat\mu}_\mathcal B$和$\pmb{\hat\sigma}_\mathcal B$）。

## 作用
批量归一化有两个作用：  

1）添加噪音，防止过拟合，是正则化的一种，不用暂退法一起使用。  
2）加速深层网络收敛速度。  

## 应用  
- 全连接层和卷积层输出上，激活函数之前。  
- 全连接层和卷积层输入上。  
- 对于全连接层，作用在特征维，即对样本的每个特征归一化。  
- 对于卷积层，作用在通道维，将每个像素看作样本，通道看作特征。 

另外，批量规范化层在”训练模式“（通过小批量统计数据规范化）和“预测模式”（通过数据集统计规范化）中的功能不同。在训练过程中，我们无法得知使用整个数据集来估计平均值和方差，所以只能根据每个小批次的平均值和方差不断训练模型。而在预测模式下，可以根据整个数据集精确计算批量规范化所需的平均值和方差。