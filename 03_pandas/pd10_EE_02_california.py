#47_00 copy
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_california_housing
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler, MinMaxScaler
from sklearn. metrics import r2_score
from sklearn.preprocessing import PolynomialFeatures
import time
import random
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.covariance import EllipticEnvelope

seed = 123
random.seed(seed)
np.random.seed(seed)

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

outliers = EllipticEnvelope(contamination=0.1)
outliers.fit(x)
results = outliers.predict(x)

mask = results == 1
x = x[mask]
y = y[mask]

pf = PolynomialFeatures(degree=2, include_bias=True, interaction_only=False)  #각각의 컬럼 데이터값끼리의 곱하기만 출력 제곱을 빼면 성능이 좀 떨어짐.
x_pf = pf.fit_transform(x)

x_train, x_test, y_train, y_test = train_test_split(
    x_pf, y, test_size=0.2, random_state=seed, #stratify=y,
)

#2. 모델
es = xgb.callback.EarlyStopping(
    rounds = 50,
    # metric_name = 'mlogloss',
    data_name = 'validation_0',
    # save_best = True,
)

model = XGBRegressor(
    random_state=seed)

model.fit(x_train, y_train,
          eval_set = [(x_test, y_test)],
          verbose = 0,
          )    

print('r2 : ', model.score(x_test, y_test))   # acc2 :  0.9666666666666667
# print(model.feature_importances_)

# 중요도가 낮은거 순서대로 제거해서 성능을 보고싶을 때.

thresholds = np.sort(model.feature_importances_)  # sort 오름차순 정렬 낮은게 젤 앞으로 점점 커지는것
# print(thresholds) 

#[0.02818759 0.03586521 0.12753496 0.80841225]

from sklearn.feature_selection import SelectFromModel

for i in thresholds : 
    selection = SelectFromModel(model, threshold=i, prefit=False)
    # threshold가 i값 이상인것을 모두 훈련시킨다.
    # i 0일 때 4개 전부다 
    #   1일 때 처음 1개 빼고 3개
    
    # threshold가 i값 이상인것을 모두 훈련시킨다.
    # prefit = False : 모델이 아직 학습되지 않았을 때, fit 호출해서 훈련한다.(기본값)
    # prefit = True : 이미 학습된 모델을 전달할 때.
    
    select_x_train = selection.fit_transform(x_train, y_train)
    select_x_test = selection.transform(x_test)
    
    # print(select_x_train.shape)

    select_model = XGBRegressor(
        random_state=seed
        )
    
    select_model.fit(select_x_train, y_train,
          eval_set = [(select_x_test, y_test)],
          verbose = 0,
          )    
    
    select_y_pred = select_model.predict(select_x_test)
    score = r2_score(y_test, select_y_pred)
    print('Thresh=%.7f, n=%dscore, R2: %.4f%%' %(i, select_x_train.shape[1], score*100))

# r2 :  0.8240050631284823
# Thresh=0.0000000, n=45score, R2: 82.4005%
# Thresh=0.0000000, n=45score, R2: 82.4005%