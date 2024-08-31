import torch
from torchvision.models import resnet18
import torch_pruning as tp





##########  Torch-Pruning 是一个专用于torch的模型剪枝库，其基于DepGraph 
# 技术分析出模型layer中的依赖关系。 
# DepGraph 与现有的修剪方法（如 Magnitude Pruning 或 Taylor Pruning）相结合可以达到良好的剪枝效果。

model = resnet18(pretrained=True).eval()

# 1. build dependency graph for resnet18
DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1,3,224,224))

# 2. Specify the to-be-pruned channels. Here we prune those channels indexed by [2, 6, 9].
group = DG.get_pruning_group( model.conv1, tp.prune_conv_out_channels, idxs=[2, 6, 9] )

# 3. prune all grouped layers that are coupled with model.conv1 (included).
print(group)
if DG.check_pruning_group(group): # avoid full pruning, i.e., channels=0.
    group.prune()
    
# 4. Save & Load
model.zero_grad() # We don't want to store gradient information
torch.save(model, 'model.pth') # without .state_dict
model = torch.load('model.pth') # load the model object
