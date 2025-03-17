# 有隐状态的循环神经网络  

## RNN模型
假设在在时间步$t$处有小批量输入$\bold X_t\in\mathbb R^{n\times d}$。换言之，有$n$个序列样本，每个序列样本在时间步$t$处的样本特征数是$d$，因此完整的数据集是三维张量$\mathsf X\in\mathbb R^{n\times T\times d}$或者$\mathsf X\in\mathbb R^{T\times n\times d}$，$T$为时间步长个数。  

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
其中$\bold W_{hq}\in\mathbb R^{h\times q}，\bold O_t\in\mathbb R^{n\times q}$，$q$为词表长度，也就是词的类别数。  

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
其中，一个序列在时间步$t$处的概率为：
$$
P(x_t\mid x_1,\ldots,x_{t-1})=\prod_{i=1}^q\hat x_{ti}^{x_{ti}}
$$
因此，我们可以通过**一个序列**中所有的n个词元的交叉熵损失的平均值来衡量：  
$$
L=\frac 1n\sum_{t=1}^n l(\hat x_t,x_t) \\
$$
其中，**一个序列在时间步$t$处**的交叉损失为:
$$
l(\hat x_t,x_t)=-\ln P(x_t\mid x_{t-1},\ldots,x_1)=-\sum_{i=1}^{q} x_{ti}\ln \hat x_{ti}
$$
自然语言处理的科学家更喜欢使用一个叫做困惑度（perplexity）的量。简而言之，它是上式的指数：  
$$
\exp\left(-\frac 1n\sum_{t=1}^n \ln P(x_t\mid x_{t-1},\ldots,x_1)\right)
$$   
如果有$m$个序列那么最终的损失函数为：  
$$
\frac 1m\sum_{j=1}^m L_j=\frac 1m\sum_{j=1}^m\frac 1n\sum_{t=1}^n l(\hat x_t,x_t)
$$  
**总结：文本序列预测本质也是分类问题，一个小批量共进行了$m\times n$次分类。**

## 梯度分析  

循环层的计算过程如下所示：
```python
def rnn(inputs, state, params):
    # inputs的形状：（时间步数量， 批量大小， 词表大小）
    W_xh, W_hh, b_h, W_hq, b_q = params
    H, = state
    outputs = []

    # 对时间步进行循环
    # X的形状：（批量大小， 词表大小）
    for X in inputs:
        H = torch.tanh(torch.mm(X, W_xh) + torch.mm(H, W_hh) + b_h)
        Y = torch.mm(H, W_hq) + b_q
        # 输出形状是（时间步数×批量大小，词表大小）
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)
```  
我们从一个描述循环神经网络工作原理的简化模型开始，此模型忽略了隐状态的特性及其更新方式的细节。这里的数学表示没有像过去那样明确地区分标量、向量和矩阵，因为这些细节对于分析并不重要，反而只会使本小节中的符号变得混乱。
我们分别使用$w_h$和$w_o$来表示隐藏层和输出层的权重。每个时间步的隐状态和输出可以写为：  
$$
\begin{align}
&h_t = f(x_t, h_{t-1}, w_h) \\
&o_t = g(h_t, w_o)
\end{align}
$$
平均交叉熵损失函数为：  
$$
L(x_1,\ldots,x_T,y_1,\ldots,y_T,w_h,w_o)=\frac 1T\sum_{t=1}^T l(y_t,o_t)
$$  
反向传播：  
$$
\frac{\partial L}{\partial w_h}=\frac 1T\sum_{t=1}^T\frac{\partial l(y_t, o_t)}{\partial o_t}\frac{\partial g(h_t, w_o)}{\partial h_t}\frac{\partial h_t}{\partial w_h}
$$  
其中$h_t$既依赖于$h_{t-1}$又依赖于$w_h$，其中$h_{t-1}$的计算也依赖于$w_h$。因此，使用链式法则产生：  
$$
\frac{\partial h_t}{\partial w_h}=\frac{\partial f(x_t, h_{t-1}, w_h)}{\partial w_h}+\frac{\partial f(x_t, h_{t-1}, w_h)}{\partial h_{t-1}}\frac{\partial h_{t-1}}{\partial w_h}+\cdots
$$
依次类推，最终得到：  
$$
\frac{\partial h_t}{\partial w_h}=\frac{\partial f(x_t, h_{t-1}, w_h)}{\partial w_h}+\sum_{i=1}^{t-1}\left(\prod_{j=i+1}^t\frac{\partial f(x_j,h_{j-1},w_h)}{\partial h_{j-1}}\right)\frac{\partial f(x_i, h_{i-1},w_h)}{\partial w_h}
$$  
当$t$很大时这个链就会变得很长，这样的连续矩阵乘法会造成数值稳定性问题，因为初始条件的微小变化就可能会对结果产生巨大的影响。  

## GRU模型  

$$
\begin{align}
\bold R_t &= \sigma(\bold X_t\bold W_{xr}+\bold H_{t-1}\bold W_{hr}+\bold b_r) \\
\bold Z_t &= \sigma(\bold X_t\bold W_{xz}+\bold H_{t-1}\bold W_{hz}+\bold b_z) \\
\bold {\tilde H}_t &= \tanh(\bold X_t\bold W_{xh}+(\bold R_t\odot\bold H_{t-1})\bold W_{hh}+\bold b_h) \\
\bold H_t &= \bold Z_t\odot\bold H_{t-1}+(1-\bold Z_t)\odot\bold {\tilde H}_t
\end{align}
$$  
看$\bold H_t$的公式，当更新门$\bold Z_t$接近1时，模型就倾向只保留旧状态。此时，来自$\bold X_t$的信息基本上被忽略，从而有效地跳过依赖链条中的时间步$t$。相反，当$\bold Z_t$接近0时，新的隐状态$\bold H_t$就会接近候选隐状态$\bold {\tilde H}_t$，$\bold {\tilde H}_t$由$\bold R_t$决定采纳多少新输入的$\bold X_t$的信息。