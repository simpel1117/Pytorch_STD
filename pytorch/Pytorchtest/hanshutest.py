import torch
import numpy as np



#print(torch.__version__)
# x = torch.rand(5,3, dtype = torch.double)
# y = torch.randn_like(x, dtype = torch.float)
# print(x)
# print(y)
# print(y.size())
# print(x+y)
# print(torch.add(x,y))
# print(x)
# print(x[1,:])
# print(x[:,1])

 ### view 可改变矩阵的维度
x = torch.randn(4,4)
y = x.view(16)
z = x.view(-1,8) #-1 表示自动做计算

print(x.size(), y.size(), z.size())

 ### 与numpy合作

a =torch.ones(5)
b = a.numpy()

c = np.ones(5)
d = torch.from_numpy(c)
print(b)
print(c)


####autograd机制

x = torch.randn(3, 4, requires_grad=True)
b = torch.randn(3, 4, requires_grad=True)
# t1 = torch.add(x , b)
t = x + b
y = t.sum()

#print("t1:",t1)
#print("t:" ,t)
print(y.backward()) # 最后一步调用反向传播
print(b.grad) # b 的梯度

print("x:{0}\n b:{1}\n t: {2}".format(x.requires_grad, b.requires_grad, t.requires_grad))


