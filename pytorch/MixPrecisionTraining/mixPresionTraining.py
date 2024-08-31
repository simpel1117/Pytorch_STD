import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler


"""

FP16（16位浮点数）：

FP16 是一种半精度浮点数格式，它使用16位（2字节）来表示一个浮点数。
它的格式通常包括1位符号位、5位指数位和10位尾数位。
由于指数位较少，FP16能够表示的数值范围比FP32小，但它需要的内存和计算资源也更少。
FP16在深度学习中被用于加速计算和节省内存，尤其是在支持FP16运算的硬件上。


FP32（32位浮点数）：
FP32 是一种单精度浮点数格式，它使用32位（4字节）来表示一个浮点数。
它的格式包括1位符号位、8位指数位和23位尾数位。
相比于FP16，FP32能够表示更大范围的数值，具有更高的精度，但也需要更多的内存和计算资源。
FP32是最常用的浮点数类型，适用于广泛的科学计算和工程应用。


在深度学习中，使用FP16进行训练可以显著减少模型的内存占用，加快数据传输和计算速度，尤其是在配备有Tensor Core的NVIDIA GPU上。
然而，由于FP16的数值范围较小，可能会导致数值下溢（underflow）或精度损失，
因此在训练过程中可能需要一些特殊的技术（如梯度缩放和混合精度训练）来确保模型的数值稳定性和最终精度。

                        

"""
class SimpleMLP(nn.Module):
    def __init__(self):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x




##########使用autocast()上下文管理器来指定哪些操作应该使用FP16执行：
#### autocast()将模型的前向传播和损失计算转换为FP16格式。然而，反向传播仍然是在FP32精度下进行的，这是为了保持数值稳定性。
model = SimpleMLP().cuda()
model.train()
scaler = GradScaler()

for epoch in range(num_epochs):
    for batch in data_loader:
        x, y = batch
        x, y = x.cuda(), y.cuda()

        with autocast():
            outputs = model(x)
            loss = criterion(outputs, y)

        # 反向传播和权重更新
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()



"""
GradScaler 是 PyTorch 中 torch.cuda.amp 模块提供的一个工具，它用于帮助进行混合精度训练。在混合精度训练中，我们通常使用 FP16 来存储模型的权重和进行前向计算，以减少内存占用和加速计算。

然而，FP16 的数值范围比 FP32 小，这可能导致在梯度计算和权重更新时出现数值下溢（underflow），即梯度的数值变得非常小，以至于在 FP16 格式下无法有效表示。

GradScaler 通过在反向传播之前自动放大（scale up）梯度的值来解决这个问题。然后，在执行权重更新之后，GradScaler 会将放大的梯度缩放（scale down）回原来的大小。这个过程确保了即使在 FP16 格式下，梯度的数值也能保持在可表示的范围内，从而避免了数值下溢的问题。

"""

scaler = torch.cuda.amp.GradScaler()
for inputs, targets in dataloader:
    with autocast():
        outputs = model(inputs)
        loss = loss_fn(outputs, targets)
    scaler.scale(loss).backward()  # 放大梯度
    scaler.step(optimizer)  # 应用缩放后的梯度进行权重更新
    scaler.update()  # 更新缩放因子

torch.save(model.state_dict(), 'model.pth')
model.load_state_dict(torch.load('model.pth'))


"""
可能出现的问题:

数值稳定性问题：使用FP16可能会导致数值下溢（underflow），即非常小的数值在FP16格式中无法有效表示，变成零。
由于FP16的精度较低，可能会在训练过程中引入舍入误差，影响模型的收敛和最终性能。

硬件兼容性：并非所有的硬件都支持FP16运算。在没有专门Tensor Core的GPU上，使用FP16可能不会带来预期的性能提升。
一些旧的或低端的GPU可能完全不支持FP16，这意味着混合精度训练无法在这些硬件上使用。

软件和库的支持： 一些深度学习框架和库可能没有完全支持混合精度训练，或者对FP16的支持不够成熟，这可能需要额外的工作来集成或调试。

模型和数据类型的转换：在混合精度训练中，需要在FP32和FP16之间转换数据类型，这可能需要仔细管理以避免精度损失。
某些操作可能需要显式地转换为FP32来保证数值稳定性，例如梯度缩放和权重更新。

调试和分析困难：使用混合精度训练可能会使得模型的调试和性能分析更加复杂，因为需要跟踪哪些操作是在FP16下执行的，哪些是在FP32下执行的。
模型泛化能力：

在某些情况下，混合精度训练可能会影响模型的泛化能力，尤其是在模型对精度非常敏感的情况下。
为了解决这些问题，研究人员和工程师通常会采用一些策略，如使用数值稳定的算法、确保正确的数据类型转换、使用支持混合精度训练的深度学习框架和库，以及在必要时进行模型微调。此外，对于特别需要高精度的任务，可能会选择使用全精度（FP32）训练，以避免潜在的精度问题。


"""