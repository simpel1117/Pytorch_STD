######卷积层:
## 参数量： (kernel*kernel) *channel_input*channel_output
## 计算量： (kernel*kernel*map*map) *channel_input*channel_output

######全连接层
# 参数量＝计算量＝weight_in*weight_out


##### 换算
#一般一个参数是值一个float，也就是４个字节 1kb=1024字节

##### 计算量(FLOPs)和参数量(Params)
## FLOPs，FLOP时指浮点运算次数，s是指秒，即每秒浮点运算次数的意思
#  FLOPs = Kh * Kw * Cin * Cout * H * W


##参数量 Params : 是指网络模型中需要训练的参数总数。  params = Kh × Kw × Cin × Cout


#### 参考链接 https://blog.csdn.net/weixin_40826634/article/details/128164063

#1 KB（千字节） = 1024字节
#1 MB（兆字节） = 1024 KB = 1024 * 1024字节
#32位浮点数（单精度）：每个参数需要4个字节（8位一个字节，共32位）来存储。
#64位浮点数（双精度）：每个参数需要8个字节（8位一个字节，共64位）来存储。

#当我们提到“XXB”（例如6B、34B）这样的术语时，它通常指的是模型的参数量，
# 其中“B”代表“Billion”，即“十亿”。M Milloin 百万。 K Thousand 千 。因此，6B表示模型有6十亿（即6亿）个参数，而34B表示模型有34十亿（即34亿）个参数。




 ################ 第一种方法：thop： pip install thop

# -- coding: utf-8 --
import torch
import torchvision
from thop import profile

# Model
print('==> Building model..')
model = torchvision.models.alexnet(pretrained=False)

dummy_input = torch.randn(1, 3, 224, 224)
flops, params = profile(model, (dummy_input,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))




##########第二种方法：ptflops
# -- coding: utf-8 --
import torchvision
from ptflops import get_model_complexity_info

model = torchvision.models.alexnet(pretrained=False)
flops, params = get_model_complexity_info(model, (3, 224, 224), as_strings=True, print_per_layer_stat=True)
print('flops: ', flops, 'params: ', params)




###### 第三种方法：pytorch_model_summary
import torch
import torchvision
from pytorch_model_summary import summary

# Model
print('==> Building model..')
model = torchvision.models.alexnet(pretrained=False)

dummy_input = torch.randn(1, 3, 224, 224)
print(summary(model, dummy_input, show_input=False, show_hierarchical=False))





#######  第四种方法：参数总量和可训练参数总量
import torch
import torchvision
from pytorch_model_summary import summary

# Model
print('==> Building model..')
model = torchvision.models.alexnet(pretrained=False)

pytorch_total_params = sum(p.numel() for p in model.parameters())
trainable_pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

print('Total - ', pytorch_total_params)
print('Trainable - ', trainable_pytorch_total_params)


######输入数据对模型的参数量和计算量的影响


# -- coding: utf-8 --
import torch
import torchvision
from thop import profile

# Model
print('==> Building model..')
model = torchvision.models.alexnet(pretrained=False)

dummy_input = torch.randn(1, 3, 224, 224)
#dummy_input = torch.randn(1, 3, 512, 512)
#dummy_input = torch.randn(8, 3, 224, 224)
flops, params = profile(model, (dummy_input,))
print('flops: ', flops, 'params: ', params)
print('flops: %.2f M, params: %.2f M' % (flops / 1000000.0, params / 1000000.0))
