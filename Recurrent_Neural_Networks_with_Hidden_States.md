# 有隐状态的循环神经网络  

## 模型
假设在在时间步$t$处有小批量输入$\bold X_t\in\mathbb R^{n\times d}$。换言之，有$n$个序列样本，每个序列样本在时间步$t$处的样本特征数是$d$，因此完整的数据集是三维张量$\mathsf X\in\mathbb R^{n\times t\times d}$。  

当前时间步的隐藏状态由上一时间步的隐藏状态和当前时间步的输入决定：
$$
\bold H_t = \phi(\bold X_t\bold W_{xh}+\bold H_{t-1}\bold W_{hh}+\bold b_h)
$$  
其中$\bold W_{xh}\in\mathbb R^{d\times h}，\bold W_{hh}\in\mathbb R^{h\times h}，\bold H_{t-1},\bold H_t\in\mathbb R^{n\times h}$。

与MLP相比，多添加了一项$\bold H_{t-1}\bold W_{hh}$，由于在当前时间步中，隐状态使用的定义与前一个时间步中使用的定义相同，因此 上式的计算是循环的（recurrent）。于是基于循环计算的隐状态神经网络被命名为 循环神经网络（recurrent neural network）。在循环神经网络中执行上式计算的层称为循环层（recurrent layer）。 

对于时间步t，输出层的输出类似于多层感知机中的计算：  
$$
\bold O_t=\bold H_t\bold W_{hq}+\bold b_q 
$$
其中$\bold W_{hq}\in\mathbb R^{h\times q}$。  

结构示意图如下：  
![具有隐状态的循环神经网络](picture\rnn_with_hidden_states.jpg)  
隐状态中$\bold X_t\bold W_{xh}+\bold H_{t-1}\bold W_{hh}$的计算，相当于$\bold X_t$和$\bold H_{t-1}$的拼接与$\bold W_{xh}$和$\bold W_{hh}$的拼接的矩阵乘法。具体如下：
```python
torch.matmul(torch.cat((X, H), 1), torch.cat((W_xh, W_hh), 0))
```

两个语言模型中应用RNN的例子：  
例一  
![rnn例子1](picture\rnn_example_1.jpg)  
例二  
![rnn例子2](picture\rnn_example_2.jpg)  

## 评价模型好坏（损失函数）——困惑度（Perplexity）  
语言模型的目标是：
$$  
P(x_1,\ldots,x_n)=P(x_1)P(x_2\mid x_1)P(x_3\mid x_1,x_2)\cdots P(x_n\mid x_1,\ldots,x_{n-1})=\prod_{t=1}^n P(x_t\mid x_1,\ldots,x_{t-1})
$$  
因此，我们可以通过一个序列中所有的n个词元的交叉熵损失的平均值来衡量：  
$$
\frac 1n\sum_{t=1}^n -\ln P(x_t\mid x_{t-1},\ldots,x_1)
$$
自然语言处理的科学家更喜欢使用一个叫做困惑度（perplexity）的量。简而言之，它是上式的指数：  
$$
\exp\left(-\frac 1n\sum_{t=1}^n \ln P(x_t\mid x_{t-1},\ldots,x_1)\right)
$$ 

## 梯度裁剪