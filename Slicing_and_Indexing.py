import torch


# 纯索引（index）
# 所有index_arry的相同位置元素构成索引坐标
# 结果的维度与index_arry保持一致
# 结果是原tensor的一个copy
# index_arry可以应用广播机制
x = torch.arange(0, 20, dtype=torch.float32).reshape(4, 5)
print(x)
print(x[range(4), [0, 2, 4, 1]])






# 索引+切片（index + slice）
# 所有index_arry的相同位置元素构成索引坐标
# 每个索引坐标与slice覆盖的每个切片坐标进行组合
# 最终的索引顺序如下：
# 第一个索引坐标与第一个切片坐标组合成最终坐标，最终坐标最高维度不变，然后寻找剩余坐标；
# 寻找完毕后，第二个索引坐标与第二个切片坐标组合成最终坐标，最终坐标最高维度不变，然后寻找剩余坐标；
# 依次类推，直到找到所有坐标。
# 结果的维度是变化的
# 结果是原tensor的一个copy
print(x[:, [0, 2, 4, 1]])


y = torch.arange(0, 60, dtype=torch.float32).reshape(3, 4, 5)
print(y)
print(y[:, [0, 1], [0, 1]]) # 000 011 100 111 200 211
print(y[[0, 1], 0:3, [0, 1]]) # 000 010 020 101 111 121
print(y[[0, 1], [0, 1], 0:3]) # 000 001 002 110 111 112
print(y[0:2, 0:2, [0, 1]]) # 000 001 010 011 100 101 110 111
print(y[[[2, 1],[0, 1]], 0:3, [[0, 1],[1, 2]]]) # 200 210 220 100 110 120 001 011 021 102 112 122

X = torch.arange(1, 26, dtype=torch.float32).reshape(5, 5)
X.requires_grad_(True)
b = torch.arange(0, 5, dtype=torch.float32, requires_grad=True)
l = torch.matmul(X, b)
print(l)
l.sum().backward()
print(b.grad)
b.grad.zero_()
l.sum().backward()
print(b.grad)