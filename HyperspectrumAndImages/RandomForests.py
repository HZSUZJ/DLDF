import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import cv2
import numpy as np

datas = pd.read_csv('freshness.csv')
imagePaths = datas.values[:, 0]
hyDatas = datas.values[:, 2:]
data = []
for i in range(len(imagePaths)):
    path = imagePaths[i]
    hyd = hyDatas[i]
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(image, (22, 21),
                     interpolation=cv2.INTER_CUBIC)
    img = img.flatten()
    hyd = np.array(hyd)
    tmp = np.append(img, hyd)
    data.append(((tmp / 255).flatten()))
data = np.array(data)
label = datas.values[:, 1]
label = label.astype('int')
train_datas, test_datas, train_labels, test_labels = train_test_split(data, label, test_size=0.3, random_state=1)
rfc = RandomForestClassifier(n_estimators=101, random_state=21, max_depth=12)
rfc = rfc.fit(train_datas, train_labels)
