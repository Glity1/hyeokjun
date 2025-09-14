import numpy as np 
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from sklearn.model_selection import train_test_split


#1. 데이터
path = './_data/kaggle/santander/'  
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0) 

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']             

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=74, stratify=y
)

scaler = RobustScaler()
x = scaler.fit(x)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)

n_split = 10
kfold = KFold(n_splits=n_split, shuffle=True, random_state=74)  # 몇번 접을껀가 n_split 만큼
# kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=222)  # 분류에서만 쓴다.

#2. 모델구성    
model = HistGradientBoostingRegressor()
# model = RandomForestRegressor()

#3. 컴파일, 훈련
scores = cross_val_score(model, x_train, y_train, cv=kfold)   # fit까지 포함됨.
print('score : ', scores, '\n 평균 score : ', round(np.mean(scores),4))                     # \n 줄바꿈


#4. 평가, 예측
# results = model.score(x,y)
# print(results)

# score :  [0.18737158 0.17984171 0.18962242 0.19782979 0.18966566 0.186118
# 0.19646623 0.18496493 0.18630935 0.19451104]
# 평균 score :  0.1893