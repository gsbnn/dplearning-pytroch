# Softmax回归  
1.Softmax回归适用于分类问题，对于类别，也就是labels，采用独热编码（one‐hot encoding）。独热编码是一个向量，它的分量和类别一样多。类别对应的分量设置为1，其他所有分量设置为0。  

2.网络结构  
softmax网络softmax回归是一个单层神经网络。由于计算每个输出o1、 o2和o3取决于所有输入x1、 x2、 x3和x4，所以softmax回归的输出层也是全连接层。  
![softmax网络结构](picture\softmax_structure.jpg)  
假设输入一个样本$\bold x(x_1, x_2, x_3, x_4)$，那么输出$\bold o(o_1, o_2, o_3)$可写作：
$$
o_1 = x_1w_{11}+x_2w_{12}+x_3w_{13}+x_4w_{14}+b_1 \\
o_2 = x_1w_{21}+x_2w_{22}+x_3w_{23}+x_4w_{24}+b_2 \\
o_3 = x_1w_{31}+x_2w_{32}+x_3w_{33}+x_4w_{44}+b_3 \\
$$

写成向量形式为$\bold{o=Wx+b}$  
现在进行扩展，假设我们读取了一个批量的样本$\bold X$，其中特征维度（输入数量）为$d$，批量大小为$n$。此外，假设我们在输出中有$q$个类别。那么小批量样本的特征为$\bold{X}\in\mathbb R^{n\times d}$，权重为$\bold{W}\in\mathbb R^{d\times q}$，偏置为$\bold{b}\in\mathbb R^{1\times q}$。数学表达式如下：  
$$\bold{O = XW+b}$$  
$\bold{X}\in\mathbb R^{n\times d}$的每一行是一个样本$\bold x$，$\bold{W}\in\mathbb R^{d\times q}$的每一列是一个权重$\bold w$，共$q$列表示网络的$q$个输出神经元，$\bold{XW}\in\mathbb R^{n\times q}$每一行是一个样本$\bold x$经过网络的输出结果$\bold o$。

3.输出结果概率化：  
$$
\bold{\hat Y}=\text{softmax}(\bold O) \\
\bold{\hat y}=\text{softmax}(\bold o) \text{，其中 } \hat y_j = \frac{\text{exp}(o_j)}{\sum_{k=1}^q\text{softmax}(o_k)} \\ 
$$  
4.损失函数采用交叉熵损失，具体推导如下所示  

a.对数似然  
假设整个数据集$\bold{\{X, Y\}}$具有$n$个样本，其中第$i$个样本由特征向量$\bold x^{(i)}$和独热标签向量$\bold y^{(i)}$组成。那么假设某次试验的结果为$\{\bold x^{(i)}, \bold y^{(i)}\}$，则其发生的概率可以表示为：     
$$
P(\bold y^{(i)}\mid\bold x^{(i)})=P_1^{y_1}P_2^{y_2}\cdots P_q^{y_q}=\prod_{j=1}^qP_j^{y_j}\text{，其中 }P_1,P_2,\dots,P_q为\bold y^{(i)}中每个元素的发生概率。
$$  
根据极大似然估计法调整参数$P_1,P_2,\dots,P_q$应使得上述概率最大，而参数$P_1,P_2,\dots,P_q$我们用预测值$\bold{\hat y}^{(i)}$代替，因此：  
$$
P(\bold y^{(i)}\mid\bold x^{(i)})=\prod_{j=1}^q\hat y_j^{y_j}  
$$   
推广到整个数据集则有：  
$$
P(\bold Y\mid\bold X)=\prod_{i=1}^nP(\bold y^{(i)}\mid\bold x^{(i)})  
$$
根据最大似然估计，我们最大化$P(\bold Y\mid\bold X)$，相当于最小化负对数似然： 
$$
-\ln P(\bold Y\mid \bold X) = \sum_{i=1}^n-\ln P(\bold y^{(i)}\mid \bold x^{(i)})=\sum_{i=1}^nl(\bold y^{(i)},\bold{\hat y}^{(i)})
$$  
其中，**单个样本**的损失函数为：
$$
l(\bold y^{(i)},\bold{\hat y}^{(i)})=-\sum_{j=1}^q y_j\ln{\hat y_j}
$$  
通常被称为交叉熵损失。  

b.损失函数的梯度  
$$
\begin{align}
l(\bold y,\bold{\hat y})&=-\sum_{j=1}^q y_j\ln\frac{\text{exp}(o_j)}{\sum_{k=1}^q\text{exp}(o_k)} \\
&=\sum_{j=1}^q y_j \ln\sum_{k=1}^q\text{exp}(o_k)-\sum_{j=1}^q y_jo_j \\
&=\ln\sum_{k=1}^q\text{exp}(o_k)-\sum_{j=1}^q y_jo_j \\
\end{align}
$$  
考虑相对于任何未规范化的预测$o_j$的导数，我们得到：
$$
\partial_{o_j}l(\bold y,\bold{\hat y})=\frac{\text{exp}(o_j)}{\sum_{k=1}^q\text{exp}(o_k)}-y_j=\text{softmax}(o_j)-y_j
$$  
换句话说，导数是我们softmax模型分配的概率与实际发生的情况（由独热标签向量表示）之间的差异。 

c.熵  
如果我们不能完全预测每一个事件，那么我们有时可能会感到“惊异”。克劳德·香农决定用信息量$\log\frac{1}{P(j)} = −\log P(j)$来量化这种惊异程度。在观察一个事件j时，并赋予它（主观）概率$P(j)$。当我们赋予一个事件较低的概率时，我们的惊异会更大，该事件的信息量也就更大。  
熵是当分配的概率真正匹配数据生成过程时的信息量的期望：
$$
H[P]=\sum_j -P(j)\log P(j)
$$  
那么什么是交叉熵？交叉熵从$P$到$Q$，记为$H(P, Q)$。我们可以把交叉熵想象为“主观概率为$Q$的观察者在看到根据概率$P$生成的数据时的预期惊异”。当$P = Q$时，交叉熵达到最低。  