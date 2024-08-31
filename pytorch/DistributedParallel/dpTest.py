import torch 
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

class MyModel(nn.Module):
    """Some Information about MyModule"""
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(10, 10)
    def forward(self, x):

        return self.fc(x)
    

model = nn.DataParallel(MyModel()).cuda()

dataset = TensorDataset(torch.randn(100,10), torch.randn(100,10))
dataloader = DataLoader(dataset, batch_size =32)

for inputs, targets in dataloader:
    inputs, targets = inputs.cuda(), targets.cuda()
    outputs = model(inputs)