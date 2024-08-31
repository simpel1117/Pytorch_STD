import torch 
from torch import nn 
import torch.nn.functional as F
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader  
from torchinfo import summary
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt




 ##### 设置随机种子 便于复现

torch.manual_seed(0)
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

##### 使用cuDNN 加速卷积预算
torch.backends.cudnn.bechmark = True


train_dataset = torchvision.datasets.MNIST(
    root="dataset/",
    train=True,
    transform=transforms.ToTensor(),
    download=True
    )

test_dataset = torchvision.datasets.MNIST(
    root="dataset/",
    train=False,
    transform=transforms.ToTensor(),
    download=True
)

### 生成dataloader
train_loader = DataLoader(dataset=train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=32, shuffle=True)


class TeacherModel(nn.Module):
    def __init__(self, in_channels=1,num_classes=10):
        super(TeacherModel,self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(784, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, num_classes)
        self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = x.view(-1,784)
        x = self.fc1(x)
        x = self.dropout(x)
        x = self.relu(x)

        x = self.fc2(x)
        x = self.dropout(x)
        x = self.relu(x)


        x = self.fc3(x)

        return x

teacher_model = TeacherModel()
teacher_model = teacher_model.to(device)

# summary(model)

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# epochs = 6
# for epoch in range(epochs):
#     model.train()
#     for data,targets in tqdm(train_loader):
#         data = data.to(device)
#         targets = targets.to(device)

#         preds = model(data)
#         loss = criterion(preds, targets)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

# model.eval()
# num_correct = 0
# num_samples = 0

# with torch.no_grad():
#     for x, y in test_loader:
#         x = x.to(device)
#         y = y.to(device)

#         preds = model(x)
#         predictions = preds.max(1).indices
#         num_correct += (predictions == y).sum()
#         num_samples += predictions.size(0)
#     acc =(num_correct/num_samples).item()

# model.train()
# print("Epoch:{}\tAccuracy:{:.4f}".format(epoch+1,acc))





class StudentModel(nn.Module):
    def __init__(self, in_channels=1,num_classes=10):
        super(StudentModel,self).__init__()
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(784, 20)
        self.fc2 = nn.Linear(20, 20)
        self.fc3 = nn.Linear(20, num_classes)
        #self.dropout = nn.Dropout(p=0.5)
    
    def forward(self, x):
        x = x.view(-1,784)
        x = self.fc1(x)
       # x = self.dropout(x)
        x = self.relu(x)

        x = self.fc2(x)
      #  x = self.dropout(x)
        x = self.relu(x)

        x = self.fc3(x)

        return x

model = StudentModel()
model = model.to(device)

# #summary(model)

# criterion = nn.CrossEntropyLoss()
# optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


# epochs = 6
# for epoch in range(epochs):
#     model.train()
#     for data,targets in tqdm(train_loader):
#         data = data.to(device)
#         targets = targets.to(device)

#         preds = model(data)
#         loss = criterion(preds, targets)

#         optimizer.zero_grad()
#         loss.backward()
#         optimizer.step()

# model.eval()
# num_correct = 0
# num_samples = 0

# with torch.no_grad():
#     for x, y in test_loader:
#         x = x.to(device)
#         y = y.to(device)

#         preds = model(x)
#         predictions = preds.max(1).indices
#         num_correct += (predictions == y).sum()
#         num_samples += predictions.size(0)
#     acc =(num_correct/num_samples).item()

# model.train()
# print("Epoch:{}\tAccuracy:{:.4f}".format(epoch+1,acc))






### 准备好训练好的教师模型
teacher_model.eval()
model = StudentModel()
model = model.to(device)
model.train()

##### Temp越大越soft
temp = 7

hard_loss = nn.CrossEntropyLoss()
alpha = 0.3

soft_loss = nn.KLDivLoss(reduction="batchmean")

optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)


epochs = 6
for epoch in range(epochs):
    model.train()
    for data,targets in tqdm(train_loader):
        data = data.to(device)
        targets = targets.to(device)
        
        with torch.no_grad():
            teacher_preds = model(data)
        student_preds = model(data)
        
        ####### student_loss 也叫hardloss
        student_loss = hard_loss(student_preds, targets)
        

        ####### ditillation_loss 也叫softloss
        ditillation_loss = soft_loss(
            F.softmax(student_preds/ temp, dim=1),
            F.softmax(teacher_preds/ temp, dim=1)
        )
       
        loss = alpha* student_loss +(1 - alpha)*ditillation_loss
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

model.eval()
num_correct = 0
num_samples = 0

with torch.no_grad():
    for x, y in test_loader:
        x = x.to(device)
        y = y.to(device)

        preds = model(x)
        predictions = preds.max(1).indices
        num_correct += (predictions == y).sum()
        num_samples += predictions.size(0)
    acc =(num_correct/num_samples).item()

model.train()
print("Epoch:{}\tAccuracy:{:.4f}".format(epoch+1,acc))





































#    ##################  知识蒸馏T如何 选择

# logits  = np.array([-5,2,7,9])
# #### 普通的Softmax T=1
# softmax_1 = np.exp(logits) /sum(np.exp(logits))

# #print(softmax_1)

# # plt.plot(softmax_1, label = "softmax_1")
# # plt.legend()
# # plt.show()

# ######  知识蒸馏T=3
# T3 = 3
# softmax_3 = np.exp(logits/T3) / sum(np.exp(logits/T3))

# ######  知识蒸馏T=5
# T5 = 5
# softmax_5 = np.exp(logits/T5) / sum(np.exp(logits/T5))

# ######  知识蒸馏T=10
# T10 = 10
# softmax_10 = np.exp(logits/T10) / sum(np.exp(logits/T10))

# ######  知识蒸馏T=100
# T100 = 100
# softmax_100 = np.exp(logits/T100) / sum(np.exp(logits/T100))

# plt.plot(softmax_1, label="T=1")
# plt.plot(softmax_3, label="T=3")
# plt.plot(softmax_5, label="T=5")
# plt.plot(softmax_10, label="T=10")
# plt.plot(softmax_100, label="T=100")
# plt.legend()
# plt.show()
