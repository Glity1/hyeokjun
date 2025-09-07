#47_00 copy
import numpy as np
import pandas as pd
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
path = './_data/kaggle/otto/'  
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0) 

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']             

le = LabelEncoder() 
y = le.fit_transform(y)

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=123, train_size=0.8,  stratify=y
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

# acc1 :  0.8116515837104072
# Thresh=0.002, n=93score, ACC: 81.1652%
# Thresh=0.002, n=92score, ACC: 81.0763%
# Thresh=0.002, n=91score, ACC: 81.0278%
# Thresh=0.002, n=90score, ACC: 81.0116%
# Thresh=0.002, n=89score, ACC: 80.7531%
# Thresh=0.002, n=88score, ACC: 81.0763%
# Thresh=0.002, n=87score, ACC: 81.3348%
# Thresh=0.003, n=86score, ACC: 81.2702%
# Thresh=0.003, n=85score, ACC: 80.9874%
# Thresh=0.003, n=84score, ACC: 81.0440%
# Thresh=0.003, n=83score, ACC: 81.0520%
# Thresh=0.003, n=82score, ACC: 81.1005%
# Thresh=0.003, n=81score, ACC: 81.1732%
# Thresh=0.003, n=80score, ACC: 80.7531%
# Thresh=0.003, n=79score, ACC: 80.8096%
# Thresh=0.003, n=78score, ACC: 80.9632%
# Thresh=0.003, n=77score, ACC: 80.9793%
# Thresh=0.003, n=76score, ACC: 81.0520%
# Thresh=0.003, n=75score, ACC: 81.0278%
# Thresh=0.003, n=74score, ACC: 80.8258%
# Thresh=0.003, n=73score, ACC: 80.9632%
# Thresh=0.003, n=72score, ACC: 80.7773%
# Thresh=0.003, n=71score, ACC: 81.2056%
# Thresh=0.004, n=70score, ACC: 80.8581%
# Thresh=0.004, n=69score, ACC: 81.0682%
# Thresh=0.004, n=68score, ACC: 80.9712%
# Thresh=0.004, n=67score, ACC: 80.9793%
# Thresh=0.004, n=66score, ACC: 80.5915%
# Thresh=0.004, n=65score, ACC: 80.6238%
# Thresh=0.004, n=64score, ACC: 80.6803%
# Thresh=0.004, n=63score, ACC: 80.5349%
# Thresh=0.004, n=62score, ACC: 80.8258%
# Thresh=0.004, n=61score, ACC: 80.6399%
# Thresh=0.004, n=60score, ACC: 80.5915%
# Thresh=0.004, n=59score, ACC: 80.6803%
# Thresh=0.004, n=58score, ACC: 80.6642%
# Thresh=0.005, n=57score, ACC: 80.4460%
# Thresh=0.005, n=56score, ACC: 80.3652%
# Thresh=0.005, n=55score, ACC: 80.4218%
# Thresh=0.005, n=54score, ACC: 80.1309%
# Thresh=0.005, n=53score, ACC: 79.8966%
# Thresh=0.005, n=52score, ACC: 79.9531%
# Thresh=0.005, n=51score, ACC: 79.8885%
# Thresh=0.005, n=50score, ACC: 79.7188%
# Thresh=0.005, n=49score, ACC: 79.8239%
# Thresh=0.005, n=48score, ACC: 79.6057%
# Thresh=0.005, n=47score, ACC: 79.6218%
# Thresh=0.005, n=46score, ACC: 79.7511%
# Thresh=0.006, n=45score, ACC: 79.4360%
# Thresh=0.006, n=44score, ACC: 79.2421%
# Thresh=0.006, n=43score, ACC: 79.0320%
# Thresh=0.007, n=42score, ACC: 79.1774%
# Thresh=0.007, n=41score, ACC: 79.0078%
# Thresh=0.007, n=40score, ACC: 78.9754%
# Thresh=0.007, n=39score, ACC: 78.6118%























