# 调用需要的模块
import numpy as np
import argparse
import time
import cv2
import os
import shutil
import pandas as pd
listImagePath = pd.read_csv("./dataset_one_path.csv")["path"].tolist()
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


def handle_image(imgPath, movePath):
    image = cv2.imread(imgPath)
    (H, W) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(
        image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)  # 设置网络的新输入值
    layerOutputs = net.forward(ln)  # 正向传递，返回指定层第一个输出的blob
    boxes = []  # 边框
    confidences = []  # 置信度
    classIDs = []  # 类标签
    # 每个输出层
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

    if catCount > 1:
        shutil.move(imgPath, movePath)


count = 0
movePath = "./images/twocat_one/"
startTime = time.time()

for ele in listImagePath:
    count += 1
    if count % 1000 == 0:
        print("处理到了第{}张图片".format(count))
    if count <= 32000:
        continue
    # 由于之前测试删掉了一些图片，导致csv中有的文件在文件夹中找不到，所以增加个判定
    if os.path.exists(ele) == False:
        continue
    handle_image(ele, movePath)


useTime = time.time()-startTime

print("总耗时{:.0f}m {:.0f}s".format(
    useTime//60, useTime % 60
)
)
