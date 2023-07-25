from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.tree import DecisionTreeClassifier

datas = pd.read_csv('freshness.csv')
data = datas.values[:, 1:463]
label = datas.values[:, 0]
train_datas, test_datas, train_labels, test_labels = train_test_split(data, label, test_size=0.3, random_state=1)
dtc = DecisionTreeClassifier(max_depth=6)
dtc.fit(train_datas, train_labels)
