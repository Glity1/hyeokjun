import numpy as np 
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor


#1. 데이터
path = ('./_data/kaggle/bike/bike-sharing-demand/')
train_csv = pd.read_csv(path + ('train.csv'), index_col=0)
test_csv = pd.read_csv(path + ('test.csv'), index_col=0)

x = train_csv.drop(['count', 'casual', 'registered'], axis=1)
y = train_csv['count']              


n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=222)  # 몇번 접을껀가 n_split 만큼

#2. 모델구성    
model = HistGradientBoostingRegressor()

#3. 컴파일, 훈련
scores = cross_val_score(model, x, y, cv=kfold)   # fit까지 포함됨.
print('score : ', scores, '\n 평균 score : ', round(np.mean(scores),4))                     # \n 줄바꿈

# score :  [0.33228028 0.36749399 0.36682292 0.33909    0.36702131] 
#  평균 score :  0.3545   