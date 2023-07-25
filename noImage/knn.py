import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

datas = pd.read_csv('freshness.csv')
data = datas.values[:, 1:463]
label = datas.values[:, 0]
train_datas, test_datas, train_labels, test_labels = train_test_split(data, label, test_size=0.3, random_state=1)
knn = KNeighborsClassifier(n_neighbors=2)
knn.fit(train_datas, train_labels)
