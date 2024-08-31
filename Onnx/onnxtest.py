import onnx
import onnxruntime as ort
import torch 
import io
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import numpy as np
from PIL import Image
from torchvision.transforms import transforms
import time
from onnx import numpy_helper



onnx_file = 'resnet50.onnx'
save_dir = 'resnet50.pt'

Resnet50 = torchvision.models.resnet50(pretrained=True)
torch.save(Resnet50.state_dict(),save_dir)
#print(Resnet50)


# pytorch 推理

# batch_size =1
# load_model = torchvision.models.resnet50()
# load_model.load_state_dict(torch.load(save_dir))

# # load_model.eval()
# x = torch.randn(batch_size, 3, 224, 224, requires_grad=True)
# torch_out = load_model(x)
# # print(torch_out)


# #导出onnx格式
# torch.onnx.export(load_model,
#                   x,
#                   onnx_file,
#                   export_params=True,
#                   opset_version=15,
#                   do_constant_folding=True,
#                   input_names = ['conv1'],
#                   output_names = ['fc'],
#                   dynamic_axes = {'conv1': {0:'batch_size'},
#                                 'fc': {0:'batch_size'}})

# ort_session = ort.InferenceSession(onnx_file,providers=['CPUExecutionProvider'])
# def to_numpy(tensor):
#     return tensor.detach().cpu().numpy() if tensor.requires_grad else tensor.cpu().numpy()

# ort_inputs = { ort_session.get_inputs()[0].name:to_numpy(x)}
# ort_outs = ort_session.run(None, ort_inputs)

# #比较pytroch 和 onnx runtime 得出精度
# np.testing.assert_allclose(to_numpy(torch_out), ort_outs[0], rtol=1e-03, atol=1e-05)
# print("OK")
# # 我们可以使用异常处理的方法进行检验
# try:
#     # 当我们的模型不可用时，将会报出异常
#     onnx.checker.check_model(onnx_file)
# except onnx.checker.ValidationError as e:
#     print("The model is invalid: %s"%e)
# else:
#     # 模型可用时，将不会报出异常，并会输出“The model is valid!”
#     print("The model is valid!")


# image = Image.open("cat.jpg")
# image = image.resize((224,224))
# image = image.convert("RGB")
# image.save("katz.jpg")

categories = []
# Read the categories
with open("imagenet_classes.txt", "r") as f:
    categories = [s.strip() for s in f.readlines()]
    
# def get_class_name(probabilities):
#     # Show top categories per image
#     top5_prob, top5_catid = torch.topk(probabilities, 5)
#     for i in range(top5_prob.size(0)):
#         print(categories[top5_catid[i]], top5_prob[i].item())
    

def pre_image(image_file):
    input_image = Image.open(image_file)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485,0.465,0.406], std =[0.229,0.224,0.225]),
    ])

    inputs_tensor =preprocess(input_image)
    inputs = inputs_tensor.unsqueeze(0)
    #inputs= inputs.cpu().detach().numpy()
    return inputs

# #inference with model
# # 先加载模型结构
# resnet50 = torchvision.models.resnet50()   
# # 在加载模型权重
# resnet50.load_state_dict(torch.load(save_dir))
# resnet50.eval()  
# #推理
# input_batch = pre_image('katze.jpg')

# # move the input and model to GPU for speed if available
# # print("GPU Availability: ", torch.cuda.is_available())
# # if torch.cuda.is_available():
# #     input_batch = input_batch.to('cuda')
# #     resnet50.to('cuda')
    
# with torch.no_grad():
#     output = resnet50(input_batch)
# # Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
# # print(output[0])
# # The output has unnormalized scores. To get probabilities, you can run a softmax on it.
# probabilities = torch.nn.functional.softmax(output[0], dim=0)
# get_class_name(probabilities)

#benchmark 性能
# latency = []
# for i in range(10):
#     with torch.no_grad():
#         start = time.time()
#         output = resnet50(input_batch)
#         probabilities = torch.nn.functional.softmax(output[0], dim=0)
#         top5_prob, top5_catid = torch.topk(probabilities, 5)
#         # for catid in range(top5_catid.size(0)):
#         #     print(categories[catid])
#         latency.append(time.time() - start)
#     print("{} model inference CPU time:cost {} ms".format(str(i),format(sum(latency) * 1000 / len(latency), '.2f')))

session_fp32 = ort.InferenceSession(onnx_file, providers=['CPUExecutionProvider'])
# session_fp32 = onnxruntime.InferenceSession("resnet50.onnx", providers=['CUDAExecutionProvider'])
# session_fp32 = onnxruntime.InferenceSession("resnet50.onnx", providers=['OpenVINOExecutionProvider'])

def softmax(x):
    """Compute softmax values for each sets of scores in x."""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


latency = []
def run_sample(session, categories, inputs):
    start = time.time()
    input_arr = inputs
    ort_outputs = session.run([], {'conv1':input_arr})[0]
    output = ort_outputs.flatten()
    output = softmax(output) # this is optional
    top5_catid = np.argsort(-output)[:5]
    # for catid in top5_catid:
    #     print(categories[catid])
    latency.append(time.time() - start)
    return ort_outputs

input_tensor = pre_image('katze.jpg')
input_arr = input_tensor.cpu().detach().numpy()
for i in range(10):
    ort_output = run_sample(session_fp32, categories, input_arr)
    print("{} ONNX Runtime CPU Inference time = {} ms".format(str(i),format(sum(latency) * 1000 / len(latency), '.2f')))