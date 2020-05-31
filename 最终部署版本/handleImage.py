import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

import pandas as pd
from PIL import Image
import numpy as np
import cv2

import base64
import io
import time

# dfEncyclopediaLink = pd.read_csv("./index_and_other_31breeds.csv")
dfEncyclopediaLink = pd.read_csv("./index_and_other_39breeds.csv")

model = torch.load("./MyModel_39breeds.pkl",map_location='cpu')
# model.cpu()

normalize = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)
imageTransform = transforms.Compose([
    transforms.Resize(224),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    normalize
])



def check_cat_number(imgByteStream):
    """ 先用YOLO判断图片中猫的数量，非1只就直接返回 """

    """ 设置网络模型 """
    confidenceDefault = 0.5  # 设置置信度阈值
    thresholdDefault = 0.3  # 设置使用NMS时的阈值
    labelsPath = "./yolo-coco/coco.names"  # 设置标签路径
    LABELS = open(labelsPath).read().strip().split(
        "\n")  # 打开标签文件，处理。返回一个list，包含了80个类名
    weightsPath = "./yolo-coco/yolov3.weights"
    configPath = "./yolo-coco/yolov3.cfg"
    net = cv2.dnn.readNetFromDarknet(
        configPath, weightsPath)  # 读取存储在Darknet模型中的参数、权重

    ln = net.getLayerNames()  # 获取网络所有层的名称,包含卷积层、relu层等等
    ln = [ln[i[0] - 1] for i in net.getUnconnectedOutLayers()]  # 获取输出图层的索引

    """ 读取图像 """
    temp = base64.b64encode(imgByteStream)
    temp = base64.b64decode(temp)
    temp = io.BytesIO(temp)
    image = Image.open(temp)
    # 由于 PIL.Image 使用的颜色模式是 RGB ，而 OpenCV 使用的是 BGR ，所以我们在二者间进行格式转换的时候使用了 cv2.COLOR_RGB2BGR 控制颜色模式的转换。
    image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)  # 设置网络的新输入值
    layerOutputs = net.forward(ln)  # 正向传递，返回指定层第一个输出的blob
    boxes = []  # 边框
    confidences = []  # 置信度
    classIDs = []  # 类标签
    for output in layerOutputs:
        for detection in output:
            scores = detection[5:]  # 取第6-85个值（共80个）即不同种类对应的置信度
            classID = np.argmax(scores)  # 取出置信度列表中的最大值的索引
            confidence = scores[classID]  # 然后按索引取出最大值的具体数值
            if confidence > confidenceDefault:
                box = detection[0:4] * np.array([W, H, W, H])
                (centerX, centerY, width, height) = box.astype("int")
                x = int(centerX - (width / 2))
                y = int(centerY - (height / 2))
                boxes.append([x, y, int(width), int(height)])
                confidences.append(float(confidence))
                classIDs.append(classID)
    idxs = cv2.dnn.NMSBoxes(
        boxes, confidences, confidenceDefault, thresholdDefault)

    catCount = 0
    if len(idxs) > 0:
        # 循环使用NMS确定idx
        for i in idxs.flatten():
            if LABELS[classIDs[i]] == "cat":
                catCount += 1
        return catCount
    else:
        # 如果图片里没有COCO数据集中的东西：
        return 0


def check_cat_class(imgByteStream):
    """ 
    接受request.files.get("file").read()的图像字节流，将其转为 PIL.Image 
    然后进行预测，返回[品种名，品种中文名]
    """

    """ timeBigen = time.time() """
    temp = base64.b64encode(imgByteStream)
    temp = base64.b64decode(temp)
    temp = io.BytesIO(temp)
    img = Image.open(temp)

    if img.mode != "RGB":
        img = img.convert("RGB")

    inputImg = imageTransform(img)
    inputImg = torch.unsqueeze(inputImg, 0)
    # inputImg = inputImg.float().cuda()
    inputImg = inputImg.float()
    # inputImg = inputImg.cuda()

    output = model(inputImg)
    # print(output.size())
    # print(output)

    """ timeUsed = time.time() - timeBigen
    print("time: {:.0f}m {:.0f}s".format(
        timeUsed//60, timeUsed % 60
    )) """

    """ 将原本只返回最大值的代码更改为返回若干个值，先把所有置信度大于0.1的值都返回，然后再考虑排序的事 """
    dictForPreds = {}
    confLevelList = F.softmax(output, dim=1)  # 获取置信度
    # print(confLevelList)
    # 返回置信度大于0.1的数值和索引，保存到字典
    for i in range(len(confLevelList[0])):
        temp = confLevelList[0][i]
        if temp >= 0.1:
            dictForPreds[i] = round(temp.item(), 2)
    # 如果没有数值大于0.1的话,就只把可信度最高的放入字典。
    if len(dictForPreds) == 0:
        _, preds = torch.max(confLevelList, dim=1)
        preds = preds.item()
        dictForPreds[preds] = confLevelList[0][preds]
    # print(dictForPreds)
    predList=sorted(dictForPreds.items(),key=lambda ele: ele[1],reverse=True)
    """ 排序好的[索引:置信度]已经完成，接下来要将品种名，百科都放到一个List里，然后返回出去 """
    predInfoList = []
    for i in range(len(predList)):
        index = predList[i][0]

        confiLivel = predList[i][1]
        breedEn = dfEncyclopediaLink.iat[index, 1]
        breedCh = dfEncyclopediaLink.iat[index,2]
        encyclopediaPet = dfEncyclopediaLink.iat[index, 3]
        encyclopediaWiki = dfEncyclopediaLink.iat[index, 4]
        encyclopediaBaidu = dfEncyclopediaLink.iat[index, 5]
        temp = [
            breedEn,
            breedCh,
            encyclopediaPet,
            encyclopediaWiki,
            encyclopediaBaidu,
            confiLivel
        ]
        predInfoList.append(temp)

    return predInfoList

    """ 以下原始代码先别动 """
    """ _, preds = torch.max(confLevelList, dim=1)
    preds = preds.item()
    breedEn = dfEncyclopediaLink.iloc[preds, 1]
    breedCh = dfEncyclopediaLink.iloc[preds, 2]
    encyclopediaPet = dfEncyclopediaLink.iloc[preds, 3]
    encyclopediaWiki = dfEncyclopediaLink.iloc[preds, 4]
    encyclopediaBaidu = dfEncyclopediaLink.iloc[preds, 5]

    predBreed1 = [breedEn, breedCh, encyclopediaPet,
                  encyclopediaWiki, encyclopediaBaidu]

    return predBreed1 """
