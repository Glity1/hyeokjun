import numpy as np 
import pandas as pd
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression     # 유일하게 Regression이지만 분류모델
from sklearn.tree import DecisionTreeClassifier         # 분류 모델
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import MinMaxScaler,StandardScaler, LabelEncoder, RobustScaler

#1. 데이터
path = './_data/kaggle/otto/'  
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0) 

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']             

le = LabelEncoder() # 문자열을 숫자로 현재 y에 들어간 train_csv의 target은 문자열임
y = le.fit_transform(y)

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

# LinearSVC 정확도: 0.7504
# LogisticRegression 정확도: 0.7678
# DecisionTreeClassifier 정확도: 0.5282
# RandomForestClassifier 정확도: 1.0000
                