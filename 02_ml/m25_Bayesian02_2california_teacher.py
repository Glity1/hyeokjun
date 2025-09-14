import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV
from xgboost.callback import EarlyStopping as XGB_EarlyStopping
import time
import joblib
import warnings
warnings.filterwarnings('ignore')
import random

seed = 222
random.seed(seed)
np.random.seed(seed)

#1. 데이터
x, y = fetch_california_housing(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=222,)

scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

beyesian_params = {
    'n_estimators' : (100, 500),
    'learning_rate' : (0.001, 0.5),
    'max_depth' : (3,10),
    # 'num_leaves' : (24,40),
    'min_child_weight' : (1, 10),
    # 'min_child_samples' : (10, 200),
    'gamma' : (0, 5),
    'colsample_bytree' : (0.5, 1),
    'colsample_bylevel' : (0.5, 1),
    'subsample' : (0.5, 2),
    # 'max_bin' : (9, 500),
    'reg_lambda' : (0, 100),  # default 1 // L2 정규화 // 릿지
    'reg_alpha' : (0, 10),    # default 0 // L1 정규화 // 라쏘
}

#2. 모델
def xgb_hamsu(n_estimators, learning_rate, max_depth, min_child_weight, subsample, gamma, colsample_bytree, colsample_bylevel, reg_lambda, reg_alpha):
    params = {
        'n_estimators':int(n_estimators),
        'learning_rate':learning_rate,
        'max_depth': int(max_depth),
        'min_child_weight': min_child_weight,
        'subsample': max(min(subsample,1), 0),
        'gamma' : gamma,
        'colsample_bytree': colsample_bytree,
        'colsample_bylevel': colsample_bylevel,
        'reg_lambda': max(reg_lambda, 0),
        'reg_alpha': reg_alpha,
    }
    
    model = XGBRegressor(**params, n_jobs=-1,early_stopping_rounds=30)
    
    model.fit(
        x_train, y_train,
        eval_set=[(x_test, y_test)],
        verbose=1
    )
    y_pred = model.predict(x_test)
    result = r2_score(y_test, y_pred)
    
    return result

optimizer = BayesianOptimization(
    f = xgb_hamsu,                   # 함수는 y_function을 쓴다.
    pbounds=beyesian_params,
    random_state=seed,
)

n_iter = 20
start = time.time()
optimizer.maximize(init_points=10,         # 초기 훈련 5번     # 총 25번 돌려라
                   n_iter=n_iter)         # 반복 훈련 n_iter번
end = time.time()

print(" 최적의 값 : ", optimizer.max)
print(n_iter, ' 번 걸린 시간 : ', round(end - start, 2), '초')


#  최적의 값 :  {'target': 0.8493238724978146, 'params': {'n_estimators': 341.70618562117255, 'learning_rate': 0.5, 'max_depth': 5.28012573544958, 'min_child_weight': 2.1366711246697396, 'gamma': 0.0, 'colsample_bytree': 1.0, 'colsample_bylevel': 0.5, 'subsample': 2.0, 'reg_lambda': 91.87705838123945, 'reg_alpha': 6.520576130730861}}
# 130  번 걸린 시간 :  184.77 초