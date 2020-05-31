from efficientnet import model as efficientnet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
from sklearn.metrics import f1_score, precision_score, recall_score
from matplotlib import pyplot as plt
from PIL import Image
import os
import shutil
import copy
import re
import time

# 第一次训练模型需要从EfficientNet获取模型，之后不需要，直接读之前训练保存的pkl文件
# 现在本地测试，所以把多GPU设置注释掉。之后要设置：batch_size=128,num_workers=8,.from_name("efficientnet-b7")
CLASSNUMBER = 53
model = efficientnet.EfficientNet.from_name("efficientnet-b7")
model._fc.out_features = CLASSNUMBER
# print(model._fc.in_features)
# print("分类数量",model._fc.out_features)
# 把模型放到GPU上
# model=model.cuda()
# 使用多GPU

print(model)
print("-"*80)

model = nn.DataParallel(model)
print(model)
