from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
path = ('./Study25/_data/kaggle/bike/bike-sharing-demand/') 
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
print(x)        # [10886 rows x 8 columns]
y = train_csv['count']
print(y.shape)  # (10886,)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=333, # stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True)

#2. 모델
model = MLPRegressor()

#3. 훈련
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('R2 : ', scores, '\n평균 R2 : ', np.round(np.mean(scores), 3))

y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)
r2 = r2_score(y_test, y_pred)
print('cross_val_predict R2 : ', r2)

# R2 :  [0.31171719 0.29881956 0.3101264  0.30192567 0.31404838] 
# 평균 R2 :  0.307
# cross_val_predict R2 :  0.24479591970015346