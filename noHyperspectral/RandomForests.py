import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import cv2
import numpy as np

datas = pd.read_csv('freshness.csv')
imagePaths = datas.values[:, 0]
data = []
for path in imagePaths:
    image = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
    img = cv2.resize(image, (22, 21),
                     interpolation=cv2.INTER_CUBIC)
    data.append(((img / 255).flatten()))
data = np.array(data)
label = datas.values[:, 1]
label = label.astype('int')
train_datas, test_datas, train_labels, test_labels = train_test_split(data, label, test_size=0.3, random_state=1)
rfc = RandomForestClassifier(n_estimators=101, random_state=21, max_depth=12)
rfc = rfc.fit(train_datas, train_labels)
