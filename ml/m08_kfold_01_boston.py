import numpy as np 
import pandas as pd
data_url = "http://lib.stat.cmu.edu/datasets/boston"
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
x = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
y = raw_df.values[1::2, 2]
# from sklearn.datasets import load_boston
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor


#1. 데이터
# x,y = load_boston(return_X_y=True)

n_split = 7
kfold = KFold(n_splits=n_split, shuffle=True, random_state=222)  # 몇번 접을껀가 n_split 만큼

#2. 모델구성    
model = HistGradientBoostingRegressor()

#3. 컴파일, 훈련
scores = cross_val_score(model, x, y, cv=kfold)   # fit까지 포함됨.
print('score : ', scores, '\n 평균 score : ', round(np.mean(scores),4))                     # \n 줄바꿈


# score :  [0.91802135 0.87647    0.82059763 0.87988013 0.90759858 0.82906047
#  0.87081656]
#  평균 score :  0.8718