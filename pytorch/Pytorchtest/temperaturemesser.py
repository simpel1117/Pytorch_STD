import torch 
import numpy as np
import pandas as pd 
import matplotlib.pyplot as plt 
import torch.optim as optim 
import warnings
import datetime
warnings.filterwarnings("ignore")


features = pd.read_csv("temps.csv")
#看下数据结构
print(features.head())
print("数据维度",features.shape)


