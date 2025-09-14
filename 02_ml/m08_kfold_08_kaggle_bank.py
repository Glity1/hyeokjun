import numpy as np 
import pandas as pd
from sklearn.model_selection import KFold, cross_val_score, StratifiedKFold
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier, HistGradientBoostingRegressor
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


n_split = 4
kfold = KFold(n_splits=n_split, shuffle=True, random_state=72)  # 몇번 접을껀가 n_split 만큼
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

# acc :  [0.414037   0.41865178 0.41251949 0.39626292 0.42693425 0.40560714
#  0.41845132 0.40253652 0.41010953 0.40553615]
#  평균 acc :  0.4111