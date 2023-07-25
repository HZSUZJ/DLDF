import numpy as np
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
import pandas as pd
from sklearn.model_selection import train_test_split

datas = pd.read_csv('freshness.csv')
data = datas.values[:, 1:463]
label = datas.values[:, 0]
train_datas, test_datas, train_labels, test_labels = train_test_split(data, label, test_size=0.3, random_state=1)
rng = np.random.RandomState(1)
adb = AdaBoostClassifier(DecisionTreeClassifier(max_depth=6),
                         n_estimators=300, random_state=rng)
adb.fit(train_datas, train_labels)
