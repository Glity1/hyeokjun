#47_00 copy
import numpy as np
import pandas as pd
from sklearn.datasets import load_digits 
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
datasets = load_digits()
x = datasets.data
y = datasets.target    

le = LabelEncoder() 
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=123, train_size=0.8,
    stratify=y 
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

# acc1 :  0.975
# Thresh=0.000, n=64score, ACC: 97.5000%
# Thresh=0.000, n=64score, ACC: 97.5000%
# Thresh=0.000, n=64score, ACC: 97.5000%
# Thresh=0.000, n=64score, ACC: 97.5000%
# Thresh=0.000, n=64score, ACC: 97.5000%
# Thresh=0.000, n=64score, ACC: 97.5000%
# Thresh=0.000, n=64score, ACC: 97.5000%
# Thresh=0.000, n=64score, ACC: 97.5000%
# Thresh=0.000, n=64score, ACC: 97.5000%
# Thresh=0.000, n=64score, ACC: 97.5000%
# Thresh=0.000, n=64score, ACC: 97.5000%
# Thresh=0.000, n=64score, ACC: 97.5000%
# Thresh=0.000, n=64score, ACC: 97.5000%
# Thresh=0.003, n=51score, ACC: 97.5000%
# Thresh=0.004, n=50score, ACC: 97.2222%
# Thresh=0.004, n=49score, ACC: 97.7778%
# Thresh=0.004, n=48score, ACC: 97.2222%
# Thresh=0.005, n=47score, ACC: 97.5000%
# Thresh=0.005, n=46score, ACC: 97.5000%
# Thresh=0.005, n=45score, ACC: 97.2222%
# Thresh=0.005, n=44score, ACC: 96.9444%
# Thresh=0.005, n=43score, ACC: 96.9444%
# Thresh=0.006, n=42score, ACC: 96.3889%
# Thresh=0.006, n=41score, ACC: 96.6667%
# Thresh=0.007, n=40score, ACC: 96.6667%
# Thresh=0.007, n=39score, ACC: 96.3889%
# Thresh=0.008, n=38score, ACC: 96.1111%
# Thresh=0.008, n=37score, ACC: 96.3889%
# Thresh=0.009, n=36score, ACC: 95.5556%
# Thresh=0.009, n=35score, ACC: 95.8333%
# Thresh=0.009, n=34score, ACC: 95.5556%
# Thresh=0.009, n=33score, ACC: 96.1111%
# Thresh=0.010, n=32score, ACC: 95.8333%
# Thresh=0.010, n=31score, ACC: 95.8333%
# Thresh=0.010, n=30score, ACC: 96.1111%
# Thresh=0.011, n=29score, ACC: 96.1111%
# Thresh=0.011, n=28score, ACC: 96.1111%
# Thresh=0.011, n=27score, ACC: 95.2778%
# Thresh=0.011, n=26score, ACC: 95.2778%
# Thresh=0.012, n=25score, ACC: 95.5556%
# Thresh=0.013, n=24score, ACC: 96.1111%
# Thresh=0.014, n=23score, ACC: 96.3889%
# Thresh=0.016, n=22score, ACC: 95.8333%
# Thresh=0.017, n=21score, ACC: 96.3889%
# Thresh=0.018, n=20score, ACC: 95.8333%
# Thresh=0.019, n=19score, ACC: 96.3889%
# Thresh=0.019, n=18score, ACC: 95.2778%
# Thresh=0.019, n=17score, ACC: 95.5556%
# Thresh=0.024, n=16score, ACC: 94.4444%
# Thresh=0.025, n=15score, ACC: 94.7222%
# Thresh=0.026, n=14score, ACC: 94.7222%
# Thresh=0.028, n=13score, ACC: 90.5556%
# Thresh=0.029, n=12score, ACC: 90.0000%
# Thresh=0.034, n=11score, ACC: 87.5000%
# Thresh=0.034, n=10score, ACC: 86.1111%
# Thresh=0.040, n=9score, ACC: 81.9444%
# Thresh=0.041, n=8score, ACC: 77.7778%
# Thresh=0.041, n=7score, ACC: 68.3333%
# Thresh=0.043, n=6score, ACC: 68.6111%
# Thresh=0.045, n=5score, ACC: 60.2778%
# Thresh=0.049, n=4score, ACC: 51.6667%
# Thresh=0.064, n=3score, ACC: 40.0000%
# Thresh=0.065, n=2score, ACC: 38.3333%
# Thresh=0.074, n=1score, ACC: 28.8889%























