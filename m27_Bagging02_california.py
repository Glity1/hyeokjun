import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from sklearn.datasets import fetch_california_housing
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
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import BaggingRegressor, RandomForestRegressor

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

#2. 모델
# model = DecisionTreeRegressor()

model = BaggingRegressor(DecisionTreeRegressor(),
                         n_estimators=100,
                         n_jobs=-1,
                         random_state=seed,
                        #  bootstrap=True,
                         )
# model = RandomForestRegressor(random_state=seed)

#3. 훈련
model.fit(x_train, y_train)

#4. 평가, 예측
results = model.score(x_test, y_test)
print('최종점수 : ', results)

# DecisionTreeRegressor
# 최종점수 :  0.6077178616735723

# Bagging
# 최종점수 :  0.8129224395750682
# Bagging 에서 bootstrap=False 일 때  # 샘플데이터 중복 비허용
# 최종점수 :  0.6282169382461609


# RandomForestRegressor
# 최종점수 :  0.8124780028585497


