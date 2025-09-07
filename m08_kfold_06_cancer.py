import numpy as np 
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.datasets import load_breast_cancer

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target           


n_split = 10
kfold = KFold(n_splits=n_split, shuffle=True, random_state=222)  # 몇번 접을껀가 n_split 만큼
# kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=222)  # 분류에서만 쓴다.

#2. 모델구성    
model = HistGradientBoostingRegressor()
# model = RandomForestRegressor()

#3. 컴파일, 훈련
scores = cross_val_score(model, x, y, cv=kfold)   # fit까지 포함됨.
print('acc : ', scores, '\n 평균 acc : ', round(np.mean(scores),4))                     # \n 줄바꿈


#4. 평가, 예측
# results = model.score(x,y)
# print(results)

# acc :  [0.82634986 0.90252779 0.92017824 0.92959243 0.88144146 0.83245775
#  0.8561398  0.82187018 0.85228193 0.87806345]
#  평균 acc :  0.8701