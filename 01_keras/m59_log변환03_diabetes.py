#47_00 copy
import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestRegressor, GradientBoostingClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler, MinMaxScaler
from sklearn. metrics import r2_score
import time
import random
import matplotlib.pyplot as plt
import xgboost as xgb

seed = 123
random.seed(seed)
np.random.seed(seed)

#1. 데이터
datasets = load_diabetes()
x = datasets.data
y = datasets.target

log_x = np.log1p(x) 
log_y = np.log1p(y) 

results = {}

# 1. 기본 (x, y 원본)
x1, y1 = x,y
x_train, x_test, y_train, y_test = train_test_split(
    x1, y1, random_state=seed)
model = RandomForestRegressor()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
results["log 변환 전 score : "] = r2_score(y_test, y_pred)

# 2. y만 log
x2, y2 = x,log_y
x_train, x_test, y_train, y_test = train_test_split(
    x2, y2, random_state=seed)
model = RandomForestRegressor()
model.fit(x_train, y_train)
y_pred_log = model.predict(x_test)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)
results["y만 log 변환 score : "] = r2_score(y_true, y_pred)
# 3. x만 log
x3, y3 = log_x, y
x_train, x_test, y_train, y_test = train_test_split(
    x3, y3, random_state=seed)
model = RandomForestRegressor()
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
results["x만 log 변환 score : "] = r2_score(y_test, y_pred)
# 4. x, y 모두 log
x4, y4 = log_x, log_y
x_train, x_test, y_train, y_test = train_test_split(
    x2, y2, random_state=seed)
model = RandomForestRegressor()
model.fit(x_train, y_train)
y_pred_log = model.predict(x_test)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)
results["x,y log 변환 score : "] = r2_score(y_true, y_pred)

# 결과 출력
for k, v in results.items():
    print(f"{k} R2 Score: {v:.4f}")
    
# RandomForestRegressor 모델로
# log 변환 전 score :  R2 Score: 0.4797
# y만 log 변환 score :  R2 Score: 0.4683
# x만 log 변환 score :  R2 Score: 0.4869
# x,y log 변환 score :  R2 Score: 0.4518