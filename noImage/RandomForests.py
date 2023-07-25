import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

datas = pd.read_csv('freshness.csv')
data = datas.values[:, 1:463]
label = datas.values[:, 0]
train_datas, test_datas, train_labels, test_labels = train_test_split(data, label, test_size=0.3, random_state=1)
rfc = RandomForestClassifier(n_estimators=101, random_state=21, max_depth=12)
rfc = rfc.fit(train_datas, train_labels)
