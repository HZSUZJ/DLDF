import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
import cv2
from sklearn.model_selection import train_test_split

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
rng = np.random.RandomState(1)
adb = AdaBoostClassifier(DecisionTreeClassifier(max_depth=6),
                         n_estimators=300, random_state=rng)
adb.fit(train_datas, train_labels)
