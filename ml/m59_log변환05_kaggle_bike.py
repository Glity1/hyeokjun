#47_00 copy
import numpy as np
import pandas as pd
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
path = ('./_data/kaggle/bike/') 
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

x = train_csv.drop(['casual', 'registered', 'count'], axis=1)
y = train_csv['count']

log_x = np.log1p(x)
log_y = np.log1p(y)

# 결과 저장용
results = {}

# 1. 기본 (x, y 원본)
x1, y1 = x, y
x_train, x_test, y_train, y_test = train_test_split(x1, y1, test_size=0.2, random_state=seed)
model = RandomForestRegressor(random_state=seed)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
results["log 변환 전 score : "] = r2_score(y_test, y_pred)

# 2. y만 log
x2, y2 = x, log_y
x_train, x_test, y_train, y_test = train_test_split(x2, y2, test_size=0.2, random_state=seed)
model = RandomForestRegressor(random_state=seed)
model.fit(x_train, y_train)
y_pred_log = model.predict(x_test)    # 학습한것의 예측값 까지는 log변환한거 그대로
y_pred = np.expm1(y_pred_log)         # 역변환
y_true = np.expm1(y_test)             # 평가 대상도 역변환해야 함
results["y만 log 변환 score : "] = r2_score(y_true, y_pred)

# 3. x만 log
x3, y3 = log_x, y
x_train, x_test, y_train, y_test = train_test_split(x3, y3, test_size=0.2, random_state=seed)
model = RandomForestRegressor(random_state=seed)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)
results["x만 log 변환 score : "] = r2_score(y_test, y_pred)

# 4. x, y 모두 log
x4, y4 = log_x, log_y
x_train, x_test, y_train, y_test = train_test_split(x4, y4, test_size=0.2, random_state=seed)
model = RandomForestRegressor(random_state=seed)
model.fit(x_train, y_train)
y_pred_log = model.predict(x_test)
y_pred = np.expm1(y_pred_log)
y_true = np.expm1(y_test)
results["x,y log 변환 score : "] = r2_score(y_true, y_pred)

# 결과 출력
for k, v in results.items():
    print(f"{k} R2 Score: {v:.4f}")

# RandomForestRegressor 모델로
# log 변환 전 score :  R2 Score: 0.2659
# y만 log 변환 score :  R2 Score: 0.2323
# x만 log 변환 score :  R2 Score: 0.2672
# x,y log 변환 score :  R2 Score: 0.2326
























