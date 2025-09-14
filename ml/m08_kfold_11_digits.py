import numpy as np 
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.datasets import load_digits


#1. 데이터
datasets = load_digits()
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

# acc :  [0.89656162 0.91425524 0.92306468 0.88706753 0.88998359 0.83288752
#  0.85703872 0.88096483 0.88425339 0.86004371]
#  평균 acc :  0.8826