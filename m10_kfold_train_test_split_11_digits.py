from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split, cross_val_score, cross_val_predict, KFold
from sklearn.metrics import accuracy_score, r2_score
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier, MLPRegressor
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_digits(return_X_y=True)

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

# R2 :  [0.78662151 0.76510212 0.77739507 0.77534242 0.77014414] 
# 평균 R2 :  0.775
# cross_val_predict R2 :  0.7251248847118875