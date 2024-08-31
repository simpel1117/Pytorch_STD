import torch 

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = torch.nn.Linear(3,3,bias=False)
        self.relu = torch.nn.ReLU()
        self.linear2= torch.nn.Linear(3,1,bias=False)

    def forward(self, inputs):
        outputs = self.linear1(inputs)
        outputs = self.relu(outputs)
        outputs = self.linear2(outputs)
        return outputs
    

weights = torch.tensor(([1.1], [2.2], [3.3]))
torch.manual_seed(123)
training_features = torch.randn(12000,3)
training_labels = training_features@weights

torch.manual_seed(123)
test_features = torch.randn(12000,3)
test_labels = test_features@weights

model = MyModel()
optimizer = torch.optim.Adam(model.parameters(),lr=0.1)

for i in range(100):
    preds = model (training_features)
    loss = torch.nn.functional.mse_loss(preds,training_labels)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()



model.eval()
with torch.no_grad():
    preds = model (test_features)
    mse = torch.nn.functional.mse_loss(preds, test_labels)
    print(f'float32 model testing loss:{mse.item():.3f}')

#### 动态量化模型
model_int8 = torch.ao.quantization.quantize_dynamic(model,{torch.nn.Linear}, dtype = torch.qint8)

#### 测试量化后的模型
with torch.no_grad():
    preds = model_int8(test_features)
    mse = torch.nn.functional.mse_loss(preds, test_labels)
    print(f"init8 model testing loss:{mse.item():.3f}")

print("float32 model linear1 parameter:\n", model.linear1.weight)
print("init8 model linear1 parameter(int8):\n", (torch.int_repr(model_int8.linear1.weight())))
print("init8 model linear1 parameter\n", model_int8.linear1.weight())
