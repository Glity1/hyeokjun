from sklearn.datasets import fetch_covtype
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = fetch_covtype(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=333, # stratify=y
)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

n_split = 5
kfold = StratifiedKFold(n_splits=n_split, shuffle=True)

#2. 모델
model = MLPClassifier()

#3. 훈련
scores = cross_val_score(model, x_train, y_train, cv=kfold)
print('acc : ', scores, '\n평균 acc : ', np.round(np.mean(scores), 3))

y_pred = cross_val_predict(model, x_test, y_test, cv=kfold)
acc = accuracy_score(y_test, y_pred)
print('cross_val_predict ACC : ', acc)

# acc :  [0.86571933 0.86911856 0.86811816 0.86604204 0.86624498] 
# 평균 acc :  0.867
# cross_val_predict ACC :  0.8454084662186002