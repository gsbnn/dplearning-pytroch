import torch

# 初始化一个张量（tensor）
x = torch.arange(12, dtype=torch.float32)
print(x,'\r\n', "x的元素个数为：", x.numel(),'\r\n', "x的形状为：", x.shape)

X = x.reshape(3, 4)
print(X)

y = torch.zeros((2, 3, 4)) # 构造两个3*4的张量，元素全为0
print(y)

y = torch.ones((2, 3, 4))
print(y)

y = torch.randn(3, 4) # 构建一个形状为3*4的张量，其中元素取自标准正态分布
print(y)

y = torch.tensor([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print(y)

# 张量索引和切片
print(X[-1]) # 输出X的最后一行
print(X[1:3]) # 输出X的第2行到第3行,不包括第4行
X[:2, :] = 12 # 将X的前两行所有元素赋值为12
print(X)

# 张量拼接
X = torch.arange(12, dtype=torch.float32).reshape(3, 4)
Y = torch.tensor([[2.0, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])
print("沿行拼接：\r\n", torch.cat((X,Y), dim=0)) # 沿行拼接X和Y
print("沿列拼接：\r\n", torch.cat((X,Y), dim=1)) # 沿列拼接X和Y

# 逻辑运算
print("判断X和Y中的元素是否相等：\r\n", X == Y)
print("判断X>Y的元素：\r\n", X > Y)

# Broadcasting机制，用于处理形状不同的张量的运算
a = torch.arange(3).reshape(3, 1)
b = torch.arange(2).reshape(1, 2)
print(a, '\r\n', b, '\r\n', a + b) # Broadcasting机制将a沿列扩展，b沿行扩展

# 原地修改
print("Y的地址为：", id(Y))
Y[:] = X + Y # X+=Y
print("执行Y[:] = X + Y后，Y的地址为：", id(Y))
# 执行Y = X + Y后，Y的地址发生了变化
Y = X + Y 
print("执行Y = X + Y后，Y的地址为：", id(Y))
