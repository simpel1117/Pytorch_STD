import torch
import torch.nn as nn
import numpy as np

class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LinearRegressionModel,self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        out =self.linear(x)
        return out


input_dim = 1
output_dim = 1
model = LinearRegressionModel(input_dim, output_dim)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

epochs =500
learning_rate = 0.001
optimizer = torch.optim.SGD(model.parameters(), lr = learning_rate)



#损失函数MSE常用在回归任务
ceriterion = nn.MSELoss()

x_values = [ i for i in range(11)]
x_train = np.array(x_values,dtype =np.float32)

#reshape 把数据转换成reshape的格式
x_train = x_train.reshape(-1,1)


y_values = [ 2*i+1 for i in x_values ]
y_train = np.array(y_values,dtype =np.float32)
y_train = y_train.reshape(-1,1)



for epoch in range(epochs):
    epoch +=1
    inputs = torch.from_numpy(x_train).to(device) # numpy 数据转换成torch
    labels = torch.from_numpy(y_train).to(device)

    # 梯度清零
    optimizer.zero_grad()
    # 前向传播
    outputs = model(inputs)
    #计算损失
    loss = ceriterion(outputs, labels)
    #反向传播
    loss.backward()
    #更新权重
    optimizer.step()

    if epoch % 50 ==0:
        print( "epoch{}, loss{}".format(epoch, loss.item()))


# ###模型预测
# predicted = model (torch.from_numpy(x_train).requires_grad_()).data.numpy() #.data.numpy 数据转会numpy
# print(predicted)


# ## 模型保存
# torch.save(model.state_dict(),"model.pkl")
# ## model加载
# model.load_state_dict(torch.load("model.pkl"))
