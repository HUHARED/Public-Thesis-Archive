
from efficientnet import model as efficientnet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.models as models
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
CLASSNUMBER = 31
NUM_WORKERS = 32*5
TRAINBATCHSIZE = 256 * 5
model = torch.load("./test4_resnet_epochnumber_30.pkl")
model = model.cuda()


dfAllFilePath = pd.read_csv("./all_image_path.csv")
imageFilesList = dfAllFilePath["breed"].tolist()
classToIndex = {
    x: i for i, x in enumerate(dfAllFilePath["breed"].unique())
}
IndexToClass = {
    i: x for i, x in enumerate(dfAllFilePath["breed"].unique())
}
normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

dsTransforms = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])

# 训练、验证、测试集的划分要按照不同品种来划分
"""
首先把40个不同品种的所有路径、品种分别保存到40个DataFrame中。然后40个DataFrame放到1个List中。
由于之前统计过不同种类的图片数量，所以直接用数量来对大DataFrame切片
"""
breedsList = list(IndexToClass.values())
dfBreedsNumber = pd.read_csv("./breeds_number.csv")
dfBreedsDict = {}
begin = 0
end = 0
for i in range(len(dfBreedsNumber)):
    breedName = dfBreedsNumber.iat[i, 0]
    end = dfBreedsNumber.iat[i, 2]+end
    tempDf = dfAllFilePath[begin:end]
    dfBreedsDict[breedName] = tempDf
    begin = end

# 用于存储53个品种的文件地址链接的字典已经创建完成，接下来就分别划分训练、验证、测试集，然后添加到分别添加到大的训练、验证、测试集中
dfTrain = pd.DataFrame(columns=["path", "breed"])
dfValidate = dfTrain.copy()
dfTest = dfTrain.copy()
for ele in list(dfBreedsDict.keys()):
    # ele是DataFrame
    tempDf = dfBreedsDict[ele]
    tempTrain, tempValidate, tempTest = np.split(
        tempDf.sample(frac=1, random_state=42),
        [
            int(0.6*len(tempDf)),
            int(0.8*len(tempDf)),
        ]
    )
    dfTrain = dfTrain.append(tempTrain, ignore_index=True)
    dfValidate = dfValidate.append(tempValidate, ignore_index=True)
    dfTest = dfTest.append(tempTest, ignore_index=True)


# 创建Dataset对象
class CatDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        imgPath = self.df.iat[idx, 0]
        image = Image.open(imgPath)
        if image.mode != "RGB":
            image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)
        breedIndex = classToIndex[self.df.iat[idx, 1]]
        return [image, breedIndex]


# 创建训练、验证、测试集的Dataset
dsTrain = CatDataset(dfTrain, transform=dsTransforms)
dsValidate = CatDataset(dfValidate, transform=dsTransforms)
dsTest = CatDataset(dfTest, transform=dsTransforms)
# 创建训练、验证、测试集的Dataloader
dlTrain = DataLoader(
    dsTrain,
    batch_size=TRAINBATCHSIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,
    drop_last=True
)
dlValidate = DataLoader(
    dsValidate,
    batch_size=32,
    shuffle=True,
    num_workers=NUM_WORKERS,
    drop_last=True
)
dlTest = DataLoader(
    dsTest,
    batch_size=32,
    shuffle=True,
    num_workers=NUM_WORKERS,
    drop_last=True
)
dataloaders = {
    "train": dlTrain,
    "validate": dlValidate,
    "test": dlTest
}
datasets = {
    "train": dsTrain,
    "validate": dsValidate,
    "test": dsTest
}


# 创建检查验证、训练集预测效果的函数
def test_analysis():
    """ 要做的是把验证、测试集的不同品种的查准、查全率分别打印保存到csv """
    sinceWhole = time.time()
    model.train(False)
    for phase in ["validate", "test"]:
        if phase == "test":
            print("测试集分析")
            with torch.no_grad():
                labelsAll = []
                predsAll = []
                for data in dataloaders[phase]:
                    inputs, labels = data
                    inputs = inputs.float().cuda()
                    labels = labels.cuda()
                    outputs = model(inputs)
                    _, preds = torch.max(outputs.data, dim=1)
                    labelsAll.extend(labels.cpu().numpy().tolist())
                    predsAll.extend(preds.cpu().numpy().tolist())
            print("验证集预测完毕，开始单独分析各品种预测结果")
            # 验证集的所有预测结果和标签都保存好了，接下来要做的事将所有种类的准确度打印，完成后，再打印当前种类被预测为其他种类的分别概率
            # 创建一个列表保存53个列表
            # 而这53个列表分别保存当前列表元素索引值所对应种类的名称，总数量，预测正确率，误预测值列表。
            # 误预测值列表包含了当前种类被错误预测的种类标签
            listAllClassIdentifyResult = []  # 包含了所有种类预测结果的列表

            for clsIndex in range(CLASSNUMBER):
                """ 循环处理每个种类 """
                listThisClassLabel = []  # 当前品种标签列表
                listThisClassPred = []  # 当前品种预测值的列表
                corrects = 0  # 正确数量
                listThisClassMisidentifiedAs = []  # 误预测值列表
                for seek in range(len(labelsAll)):
                    """ 循环处理总标签列表 """
                    if labelsAll[seek] == clsIndex:
                        """ 总标签列表的值如果等于品种索引，就把标签添加到当前品种标签列表，把预测值添加到当前品种预测值的列表 """
                        listThisClassLabel.append(labelsAll[seek])
                        listThisClassPred.append(predsAll[seek])
                lenThisClassLabel = len(listThisClassLabel)
                for i in range(lenThisClassLabel):
                    """ 比较当前品种标签和预测值，将总正确预测数量和误预测值记下来 """
                    if listThisClassLabel[i] == listThisClassPred[i]:
                        corrects += 1
                    else:
                        listThisClassMisidentifiedAs.append(
                            listThisClassPred[i])
                """ 根据总正确预测数量和当前品种图片总数计算预测正确率 """
                if lenThisClassLabel != 0:
                    classPredictionRate = round(
                        corrects / lenThisClassLabel, 2)
                else:
                    classPredictionRate = 0.0
                """ 构建最终结果列表，包含品种名，图片数，预测正确率，误预测值列表 """
                """ print(IndexToClass[clsIndex], len(
                    listThisClassLabel), classPredictionRate, listThisClassMisidentifiedAs) """
                temp = [IndexToClass.get(clsIndex), lenThisClassLabel,
                        classPredictionRate, listThisClassMisidentifiedAs]
                listAllClassIdentifyResult.append(temp)

    timeWhole = time.time()-sinceWhole
    print("Training complete in {:.0f}m {:.0f}s".format(
        timeWhole//60, timeWhole % 60))

    return listAllClassIdentifyResult


# 将大体结果保存
listAllClassIdentifyResult = test_analysis()
temp = []
for ele in listAllClassIdentifyResult:
    temp.append([ele[0], ele[1], round(ele[2], 2)])

dfBreedsPredicionResults = pd.DataFrame(
    temp, columns=["breedName", "imageNumber", "PredicionRate"])
dfBreedsPredicionResults.to_csv("./test_dataset_predicion_results.csv", index=False)

# 将所有品种对应的误预测值分别保存到一个csv中（而所有csv保存到一个文件夹中），用于以后分析。
for i in range(len(listAllClassIdentifyResult)):
    temp = listAllClassIdentifyResult[i][3]
    temp = pd.DataFrame(temp, columns=["MisidentifiedAs"])
    if IndexToClass.get(i) is None:
        temp.to_csv("./each_test_analysis/"+str(i)+".csv", index=False)
    else:
        temp.to_csv("./each_test_analysis/"+str(IndexToClass.get(i))+".csv", index=False)






