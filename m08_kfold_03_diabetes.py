import numpy as np 
from sklearn.datasets import load_diabetes                               
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor


#1. 데이터
x,y = load_diabetes(return_X_y=True)

n_split = 12
kfold = KFold(n_splits=n_split, shuffle=True, random_state=222)  # 몇번 접을껀가 n_split 만큼
# kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=222)  # 분류에서만 쓴다.

#2. 모델구성    
model = HistGradientBoostingRegressor()
# model = RandomForestRegressor()

#3. 컴파일, 훈련
scores = cross_val_score(model, x, y, cv=kfold)   # fit까지 포함됨.
print('score : ', scores, '\n 평균 score : ', round(np.mean(scores),4))                     # \n 줄바꿈


#4. 평가, 예측
# results = model.score(x,y)
# print(results)

# score :  [0.42606503 0.47297321 0.42713086 0.42432442 0.33625731 0.27266792
#  0.434167   0.1768003  0.57483574 0.33577393 0.52057302 0.0042266 ]
#  평균 score :  0.3671
                