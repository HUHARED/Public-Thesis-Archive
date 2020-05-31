import os
import pandas as pd
path = "./images/wrongimages_two/"
listAllFiles = []
for root, dirs, files in os.walk(path):
    for ele in files:
        listAllFiles.append(ele)

dfFilesName = pd.DataFrame(listAllFiles)
dfFilesName = dfFilesName.rename(columns={0: "name"})
dfFilesName.to_csv("./dataset_wrongimages_two.csv")
