
import  torch
import numpy as np
from torchvision.transforms import transforms
from torch.utils.data import DataLoader, Dataset

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

print("okkkk")

