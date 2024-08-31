'''
https://pytorch.org/tutorials/intermediate/pruning_tutorial.html
'''
 
import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
 
# 选定设备
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
 
# 构建LeNet网络模型
class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=6, kernel_size=3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
 
    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
 
model = LeNet().to(device=device)  # 模型传入cpu/gpu


module = model.conv1  # 选定为第一个卷积层为操作的模块
print(list(module.named_parameters()))


#############################################################

#  ####### 选择需要剪枝的参数，随机非结构化剪掉30%的参数
# prune.random_unstructured(module, name="weight", amount=0.3)
#   #剪枝后查看named_parameters（不变）：
# print(list(module.named_parameters()))  # 查看第一层卷积层的参数，包括weight和bias

#    # 查看本次剪枝的掩码，所有剪枝操作的掩码都存储与named_buffers中（其中0和1分别代表丢弃和保留该wei位置的参数）
# print(list(module.named_buffers()))

#    #查看剪枝后权重参数（对应掩码位置为0的位置上的参数已经被替换为0）：
# print(module.weight)

###############################################################

###############################################################

##### 迭代剪枝 此操作将剪掉2个node的weight,总的剪除数量为50%
# prune.ln_structured(module, name="weight", amount=0.5, n=2, dim=0)

#  #查看剪枝后的权重
# print(module.weight)

###############################################################
# ###########对整个模型进行剪枝
new_model = LeNet()
for name, module in new_model.named_modules():
    # 在所有的2d卷积层中使用l1正则非结构剪枝20%的权重
    if isinstance(module, torch.nn.Conv2d):
        prune.l1_unstructured(module, name='weight', amount=0.2)
    # 在所有的线性层使用l1正则非结构剪枝40%的权重
    elif isinstance(module, torch.nn.Linear):
        prune.l1_unstructured(module, name='weight', amount=0.4)
 
print(dict(new_model.named_buffers()).keys()) 









########## 全局剪枝
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class LeNet(nn.Module):
    def __init__(self):
        super(LeNet, self).__init__()
        # 1 input image channel, 6 output channels, 3x3 square conv kernel
        self.conv1 = nn.Conv2d(1, 6, 3)
        self.conv2 = nn.Conv2d(6, 16, 3)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 5x5 image dimension
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, int(x.nelement() / x.shape[0]))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

model = LeNet().to(device=device)

model = LeNet()

parameters_to_prune = (
    (model.conv1, 'weight'),
    (model.conv2, 'weight'),
    (model.fc1, 'weight'),
    (model.fc2, 'weight'),
    (model.fc3, 'weight'),
)

prune.global_unstructured(
    parameters_to_prune,
    pruning_method=prune.L1Unstructured,
    amount=0.2,
)
# 计算卷积层和整个模型的稀疏度
# 其实调用的是 Tensor.numel 内内函数，返回输入张量中元素的总数
print(
    "Sparsity in conv1.weight: {:.2f}%".format(
        100. * float(torch.sum(model.conv1.weight == 0))
        / float(model.conv1.weight.nelement())
    )
)
print(
    "Global sparsity: {:.2f}%".format(
        100. * float(
            torch.sum(model.conv1.weight == 0)
            + torch.sum(model.conv2.weight == 0)
            + torch.sum(model.fc1.weight == 0)
            + torch.sum(model.fc2.weight == 0)
            + torch.sum(model.fc3.weight == 0)
        )
        / float(
            model.conv1.weight.nelement()
            + model.conv2.weight.nelement()
            + model.fc1.weight.nelement()
            + model.fc2.weight.nelement()
            + model.fc3.weight.nelement()
        )
    )
)
# 程序运行结果
"""
Sparsity in conv1.weight: 3.70%
Global sparsity: 20.00%
"""