


from efficientnet import model as efficientnet
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
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

os.environ['CUDA_VISIBLE_DEVICES'] = "0,3,4,5,6"

# 第一次训练模型需要从EfficientNet获取模型，之后不需要，直接读之前训练保存的pkl文件
CLASSNUMBER = 31
NUM_WORKERS = 4*5
TRAINBATCHSIZE = 40 * 5
EPOCHNUMBER = 51
# 因为下载预训练参数很慢，所以先试试非预训练的模型。
# model=efficientnet.EfficientNet.from_pretrained("efficientnet-b8")
# model=efficientnet.EfficientNet.from_name("efficientnet-b8")
# model._fc.out_features=CLASSNUMBER
# model._fc.out_features = CLASSNUMBER
# print(model._fc.in_features)
# print("分类数量",model._fc.out_features)

model=torch.load("././test5_efficientnet_epochnumber_51.pkl")

""" 用DistributedDataParallel """
torch.distributed.init_process_group(backend="nccl")
model = model.cuda()
model = nn.parallel.DistributedDataParallel(model)


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
# 由于数据集不同种类图片数量相差太大，所以训练、验证、测试集的划分要按照不同品种来划分
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
    pin_memory=True,
    drop_last=True
)
dlValidate = DataLoader(
    dsValidate,
    batch_size=32,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    drop_last=True
)
dlTest = DataLoader(
    dsTest,
    batch_size=32,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
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

# 创建损失计算函数、参数优化器、学习率调度器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adadelta(model.parameters(), lr=1)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, mode="max", verbose=True,patience=5,threshold=0.01)
""" 换一个学习率优化器 """
# scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=9)

# 定义训练函数，注意，由于要在服务器上训练，所以每次调用训练函数后，加一行保存模型的代码。


def train_model(model, criterion, optimizer, scheduler, epochNumber=42):
    sinceWhole = time.time()
    bestModelState = model.state_dict()
    bestF1 = 0.0
    bestPrecision = 0.0
    bestRecall = 0.0
    nowLr = 0.0

    for epoch in range(epochNumber):
        sinceThisEpoch = time.time()
        print("Epoch {}/{}".format(epoch, epochNumber))
        print("-"*20)
        for phase in ["train", "validate"]:
            if phase == "train":
                model.train(True)
            else:
                model.train(False)

            if phase == "train":
                labelsAll = []
                predsAll = []
                for data in dataloaders[phase]:
                    inputs, labels = data
                    inputs = inputs.float().cuda()
                    labels = labels.cuda()
                    optimizer.zero_grad()

                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs.data, dim=1)
                    labelsAll.extend(labels.cpu().numpy().tolist())
                    predsAll.extend(preds.cpu().numpy().tolist())

                    loss.backward()
                    optimizer.step()

                f1Train = f1_score(labelsAll, predsAll, average="macro")
                precsionTrain = precision_score(
                    labelsAll, predsAll, average="macro")
                recallTrain = recall_score(
                    labelsAll, predsAll, average="macro")

            if phase == "validate":
                labelsAll = []
                predsAll = []
                with torch.no_grad():
                    for data in dataloaders[phase]:
                        inputs, labels = data
                        inputs = inputs.float().cuda()
                        labels = labels.cuda()

                        outputs = model(inputs)
                        _, preds = torch.max(outputs.data, dim=1)
                        labelsAll.extend(labels.cpu().numpy().tolist())
                        predsAll.extend(preds.cpu().numpy().tolist())
                f1Validate = f1_score(labelsAll, predsAll, average="macro")
                precsionValidate = precision_score(
                    labelsAll, predsAll, average="macro")
                recallValidate = recall_score(
                    labelsAll, predsAll, average="macro")
                stepSign = f1Validate
                scheduler.step(stepSign)
                """ 打印学习率变化 """
                # scheduler.step()
                """ lrState = scheduler.state_dict() """

                """ 之后用个if函数判断，仅在学习率变化时打印 """

                """ for key in lrState.keys():
                    print(key, "=", lrState[key]) """

                if f1Validate > bestF1:
                    bestF1 = f1Validate
                    bestPrecision = precsionValidate
                    bestRecall = recallValidate
                    bestModelState = model.state_dict()

        # 一个Epoch完成后
        timeUsedThisEpoch = time.time()-sinceThisEpoch
        print("Train - F1: {:.9f}\tprecision: {:.9f}\trecall: {:.9f}\nValidation - F1: {:.9f}\tprecision: {:.9f}\trecall: {:.9f}\nin {:.0f}m {:.0f}s".format(
            f1Train,
            precsionTrain,
            recallTrain,
            f1Validate,
            precsionValidate,
            recallValidate,
            timeUsedThisEpoch//60, timeUsedThisEpoch % 60
        ))
        print()

    # 训练、验证都完成后
    timeWhole = time.time()-sinceWhole
    print("Training complete in {:.0f}m {:.0f}s".format(
        timeWhole//60, timeWhole % 60))
    print("Best Validation - F1: {:.9f}\tprecision: {:.9f}\trecall: {:.9f}".format(
        bestF1,
        bestPrecision,
        bestRecall
    ))
    print()

    model.load_state_dict(bestModelState)
    return model


model = train_model(model, criterion, optimizer,
                    scheduler, epochNumber=EPOCHNUMBER)

savePath = "./test5_efficientnet_epochnumber_"+str(EPOCHNUMBER)+".pkl"
torch.save(model.module, savePath)


