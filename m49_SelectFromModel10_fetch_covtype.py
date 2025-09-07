#47_00 copy
import numpy as np
import pandas as pd
from sklearn.datasets import fetch_covtype
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBRegressor, XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, RobustScaler, StandardScaler, MinMaxScaler
from sklearn. metrics import r2_score, accuracy_score
import time
import random
import matplotlib.pyplot as plt
import xgboost as xgb

seed = 123
random.seed(seed)
np.random.seed(seed)

#1. 데이터
datasets = fetch_covtype()
x = datasets.data
y = datasets.target   

le = LabelEncoder()
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed, #stratify=y
)
#2. 모델
es = xgb.callback.EarlyStopping(
    rounds = 50,
    # metric_name = 'mlogloss',
    data_name = 'validation_0',
    # save_best = True,
)

model = XGBClassifier(
    random_state=seed)

model.fit(x_train, y_train,
          eval_set = [(x_test, y_test)],
          verbose = 0,
          )    

print('acc1 : ', model.score(x_test, y_test))   # acc2 :  0.9666666666666667
# print(model.feature_importances_)

# 중요도가 낮은거 순서대로 제거해서 성능을 보고싶을 때.

thresholds = np.sort(model.feature_importances_)  # sort 오름차순 정렬 낮은게 젤 앞으로 점점 커지는것
# print(thresholds) 

#[0.02818759 0.03586521 0.12753496 0.80841225]

from sklearn.feature_selection import SelectFromModel

for i in thresholds : 
    selection = SelectFromModel(model, threshold=i, prefit=True)
    # threshold가 i값 이상인것을 모두 훈련시킨다.
    # i 0일 때 4개 전부다 
    #   1일 때 처음 1개 빼고 3개
    
    # threshold가 i값 이상인것을 모두 훈련시킨다.
    # prefit = False : 모델이 아직 학습되지 않았을 때, fit 호출해서 훈련한다.(기본값)
    # prefit = True : 이미 학습된 모델을 전달할 때.
    
    select_x_train = selection.fit_transform(x_train, y_train)
    select_x_test = selection.transform(x_test)
    
    # print(select_x_train.shape)

    select_model = XGBClassifier(
        random_state=seed
        )
    
    select_model.fit(select_x_train, y_train,
          eval_set = [(select_x_test, y_test)],
          verbose = 0,
          )    
    
    select_y_pred = select_model.predict(select_x_test)
    score = accuracy_score(y_test, select_y_pred)
    print('Thresh=%.3f, n=%dscore, ACC: %.4f%%' %(i, select_x_train.shape[1], score*100))

# acc1 :  0.8708294966567128
# Thresh=0.000, n=54score, ACC: 87.0829%
# Thresh=0.002, n=53score, ACC: 87.0829%
# Thresh=0.003, n=52score, ACC: 86.9487%
# Thresh=0.003, n=51score, ACC: 87.1182%
# Thresh=0.004, n=50score, ACC: 87.1699%
# Thresh=0.004, n=49score, ACC: 87.1931%
# Thresh=0.004, n=48score, ACC: 87.2499%
# Thresh=0.005, n=47score, ACC: 87.4736%
# Thresh=0.005, n=46score, ACC: 87.2301%
# Thresh=0.005, n=45score, ACC: 86.9091%
# Thresh=0.005, n=44score, ACC: 87.0262%
# Thresh=0.006, n=43score, ACC: 87.0520%
# Thresh=0.006, n=42score, ACC: 87.3377%
# Thresh=0.006, n=41score, ACC: 87.0107%
# Thresh=0.007, n=40score, ACC: 87.1010%
# Thresh=0.007, n=39score, ACC: 87.0511%
# Thresh=0.007, n=38score, ACC: 87.2800%
# Thresh=0.007, n=37score, ACC: 87.0786%
# Thresh=0.007, n=36score, ACC: 86.8532%
# Thresh=0.009, n=35score, ACC: 87.0227%
# Thresh=0.011, n=34score, ACC: 85.7930%
# Thresh=0.011, n=33score, ACC: 86.1441%
# Thresh=0.011, n=32score, ACC: 86.1105%
# Thresh=0.011, n=31score, ACC: 85.6794%
# Thresh=0.012, n=30score, ACC: 85.3171%
# Thresh=0.013, n=29score, ACC: 81.0349%
# Thresh=0.013, n=28score, ACC: 80.8103%
# Thresh=0.013, n=27score, ACC: 78.0686%
# Thresh=0.013, n=26score, ACC: 78.1727%
# Thresh=0.014, n=25score, ACC: 72.5739%