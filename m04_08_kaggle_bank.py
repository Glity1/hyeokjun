import numpy as np 
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression     # 유일하게 Regression이지만 분류모델
from sklearn.tree import DecisionTreeClassifier         # 분류 모델
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

#1. 데이터
path = './_data/kaggle/bank/'

train_csv = pd.read_csv(path +'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)
submission_csv = pd.read_csv(path + 'sample_submission.csv', index_col=0)

le = LabelEncoder()
for col in ['Geography', 'Gender']:
    train_csv[col] = le.fit_transform(train_csv[col])
    test_csv[col] = le.transform(test_csv[col])

train_csv = train_csv.drop(["CustomerId","Surname"], axis=1)
test_csv = test_csv.drop(["CustomerId","Surname"], axis=1)

x = train_csv.drop(['Exited'], axis=1)
y = train_csv['Exited']

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

# LinearSVC 정확도: 0.7017
# LogisticRegression 정확도: 0.7856
# DecisionTreeClassifier 정확도: 0.9997
# RandomForestClassifier 정확도: 0.9996
                