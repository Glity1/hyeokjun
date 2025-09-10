import numpy as np 
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression     # 유일하게 Regression이지만 분류모델
from sklearn.tree import DecisionTreeClassifier         # 분류 모델
from sklearn.ensemble import RandomForestClassifier


#1. 데이터
path = './_data/dacon/diabetes/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

x = train_csv.drop(['Outcome'], axis=1)
x = x.replace(0, np.nan)
x = x.fillna(train_csv.mean())
y = train_csv['Outcome']

model_list =  [
    LinearSVC(C=0.3, max_iter=10000),
    LogisticRegression(max_iter=1000),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(n_estimators=100)
]

for model in model_list:
    name = model.__class__.__name__
    model.fit(x, y)
    results = model.score(x, y) 
    print(f"{name} 정확도: {results:.4f}")

#4. 평가, 예측

# LinearSVC 정확도: 0.7485
# LogisticRegression 정확도: 0.7653
# DecisionTreeClassifier 정확도: 1.0000
# RandomForestClassifier 정확도: 1.0000
                