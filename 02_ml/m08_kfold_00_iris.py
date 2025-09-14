import numpy as np 
from sklearn.datasets import load_iris
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier

#1. 데이터
x,y = load_iris(return_X_y=True)

n_split = 10
# kfold = KFold(n_splits=n_split, shuffle=True, random_state=222)  # 몇번 접을껀가 n_split 만큼
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=222)  # 몇번 접을껀가 n_split 만큼

#2. 모델구성    
model = MLPClassifier()

#3. 컴파일, 훈련
scores = cross_val_score(model, x, y, cv=kfold)   # fit까지 포함됨.
print('acc : ', scores, '\n 평균 acc : ', round(np.mean(scores),4))                     # \n 줄바꿈


#4. 평가, 예측
# results = model.score(x,y)
# print(results)
                