# 数值稳定性  
1.神经网络反向传播——链式法则  
考虑如下$d$层神经网络：
$$
\bold h^t = f_t(\bold h^{t-1}) \quad \text{and}\quad y = l \circ f_d\circ\ldots\circ f_1(\bold x)
$$  
$\bold h^t$表示第$t$层的输出，$\bold h^d$表示最后的输出$\bold o$
计算损失$l$关于参数$\bold W_t$的梯度：  
$$
\frac{\partial l}{\partial\bold W_t}=\frac{\partial l}{\partial\bold h^d}\underbrace{\frac{\partial h^d}{\partial\bold h^{d-1}}\cdots\frac{\partial h^{t+1}}{\partial\bold h^t}}_{d-t\text{次矩阵乘法}}\frac{\partial h^t}{\partial\bold W_t}
$$  
连续的矩阵乘法导致梯度爆炸或梯度消失。  

2.梯度爆炸  
(1)以MLP为例：  
$$
\begin{align}
&f_t(\bold h^{t-1}) = \sigma(\bold W^t\bold h^{t-1})\quad\sigma\text{是激活函数}\\
&\frac{\partial h^{t+1}}{\partial h^t}=\text{diag}\left(\sigma'(\bold W^{t+1}\bold h^t)\right)(\bold W^{t+1})^\top \quad \sigma'\text{是}\sigma\text{的导函数}\\
&\prod_{i=t}^{d-1}\frac{\partial h^{i+1}}{\partial h^i}=\prod_{i=t}^{d-1}\text{diag}\left(\sigma'(\bold W^{i+1}\bold h^i)\right)(\bold W^{i+1})^\top
\end{align}
$$ 
使用ReLU作为激活函数：  
$$
\sigma(x)=\text{max}(0, x)\quad\text{and}\quad
\sigma'(x)=
\begin{cases}
1&x>0\\
0&\text{其他}
\end{cases}
$$  
$\prod_{i=t}^{d-1}\frac{\partial h^{i+1}}{\partial h^i}=\prod_{i=t}^{d-1}\text{diag}\left(\sigma'(\bold W^{i+1}\bold h^i)\right)(\bold W^{i+1})^\top$的一些元素来自于$\prod_{i=t}^{d-1}(\bold W^{i+1})^\top$，如果$d-t$很大，那么值将会很大。  

(2)梯度爆炸的问题  
- 超出值域（inf）
- 对学习率敏感
    - 学习率太大，梯度过大  
    - 学习率太小，训练无进展
    - 训练过程中动态调整学习率  

3.梯度消失  
(1)使用$\text{sigmod}$作为激活函数  
如果$d-t$很大，$\prod_{i=t}^{d-1}\text{diag}\left(\sigma'(\bold W^{i+1}\bold h^i)\right)$会很小。   

(2)梯度消失的问题  
- 梯度变为0 
- 训练没有进展
    - 不管如何选择学习率
- 对于底部层尤其严重
    - 仅顶部层训练的很好
    - 无法让神经网络更深  

3.让数值稳定的办法  
- 目标：让梯度值在合理范围内，例如[1e-6,1e3]
- 将乘法变加法，例如ResNet，LSTM
- 梯度归一化，梯度裁剪
- 合理的权重初始和激活函数  

4.合理的权重初始化  
- 将每次的输出和梯度都看作随机变量  
- 让它们的均值和方差保持一致  
以MLP为例  

- 假设
    - $w_{i,j}^t$是独立同分布，那么$E[w_{i,j}^t]=0,\;\text{Var}[w_{i,j}^t]=\gamma_t $  
    - $h_j^{t-1}$独立于$w_{i,j}^t$
    - $t,i,j$分别表示第$t$层，第$i$个神经元，权重向量中第$j$个元素，$h_j^{t-1}$表示第$t-1$层的输出  

**假设没有激活函数：**  
(1)正向输出的均值和方差  
$$
E[h_i^t]=E\left[\sum_j w_{i,j}^th_j^{t-1}\right]=\sum E[w_{i,j}^t]E[h_j^{t-1}]=0
$$  
当前假设下每层输出的均值可以保持一致，接下来研究每次输出的方差  
$$
\begin{align}
\text{Var}[h_i^t]&=E\left[\left(\sum_j w_{i,j}^th_j^{t-1}\right)^2\right]\\
&=E\left[\sum_j \left(w_{i,j}^t\right)^2\left(h_j^{t-1}\right)^2+\sum_{j\neq k}w_{i,j}^tw_{i,k}^th_j^{t-1}h_k^{t-1}\right]\\
&=\sum_j E\left[\left(w_{i,j}^t\right)^2\right]E\left[\left(h_j^{t-1}\right)^2\right]\\
&=\sum_j \text{Var}[w_{i,j}^t]\text{Var}[h_j^{t-1}]\\
&=n_{t-1}\gamma_t\text{Var}[h_j^{t-1}]
\end{align}
$$  
要想使每层输出的方差保持一致，需要满足$n_{t-1}\gamma_t=1$。  

(2)反向传播的梯度的均值和方差：  
与正向情况类似：  
$$
\begin{align}
&\frac{\partial l}{\partial\bold h^{t-1}}=\frac{\partial l}{\partial\bold h^t}\bold W^t \Rightarrow \left(\frac{\partial l}{\partial\bold h^{t-1}}\right)^\top=(\bold W^t)^\top\left(\frac{\partial l}{\partial\bold h^t}\right)^\top\\
&E\left[\frac{\partial l}{\partial h_i^{t-1}}\right]=0 \quad \text{Var}\left[\frac{\partial l}{\partial h_j^{t-1}}\right]=n_t\gamma_t\text{Var}\left[\frac{\partial l}{\partial h_j^t}\right]
\end{align}
$$  
要想使每层的反向梯度的方差保持一致，需要满足$n_t\gamma_t=1$。  
$n_{t-1}，n_t$分别是第$t-1，t$层的输出$\bold h_{t-1}，\bold h_t$的维度。  

(3)初始化方案——Xavier初始化  
$n_{t-1}\gamma_t=1$和$n_t\gamma_t=1$难以同时满足，但可以使得方差$\gamma_t$满足：$\gamma_t = 2/(n_{t-1}+n_t)$。  
**因此，只需要根据第$t$层的输入输出维度调整该层权重$\bold W_t$的方差即可。**   
例如：$\bold W_t$满足正态分布$\mathcal N(0,\sqrt{2/(n_{t-1}+n_t)})$ 或者均匀分布$\mathcal U(-\sqrt{6/(n_{t-1}+n_t)},\sqrt{6/(n_{t-1}+n_t)})$等。

**假设存在激活函数：**
假设是线性激活函数$\sigma(x)=\alpha x+\beta$，那么正向输出和反向梯度的均值和方差仍然要保持一致，那么只能使$\alpha=1,\beta=0$，因此$\sigma(x)=x$。检查常见的激活函数：  
$$
\begin{align}
\text{sigmoid}(x)&=\frac 12+\frac x4-\frac {x^3}{48}+O(x^5)\\
\text{tanh}(x)&=0+x-\frac {x^3}{3}+O(x^5)\\
\text{relu}(x)&=0+x \quad  x\geq 0\\
\end{align}
$$  
需要调整$\text{sigmoid}$为:  
$$4\times\text{sigmoid}(x)-2$$  
可以显著降低$\text{sigmoid}$的梯度消失问题。 


### 总结
合理的权重初始值和激活函数的选取可以提升数值稳定性。  