#49_06 copy
import numpy as np
import pandas as pd
from sklearn.datasets import load_breast_cancer
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
import warnings
warnings.filterwarnings('ignore')

seed = 222
random.seed(seed)
np.random.seed(seed)

#1. 데이터
datasets = load_breast_cancer()
x = datasets.data
y = datasets.target
feature_names = datasets.feature_names

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=seed, #stratify=y,
)


#2. 모델
es = xgb.callback.EarlyStopping(
    rounds = 50,
    metric_name = 'logloss',
    data_name = 'validation_0',
    # save_best = True,
)

model = XGBClassifier(
    n_estimators = 500,
    max_depth = 6,
    gamma = 0,
    min_child_weight = 0,
    subsample = 0.4,
    reg_alpha = 0,
    reg_lambda = 1,
    eval_metric = 'logloss',  # 다중분류 : mlogloss, merror, 
                              # 이진분류 : logloss, error
                              # 회귀 : rmse, mae, mrsle
                              # 2.1.1 버전 이후로 fit에서 모델로 위치이동.
    callbacks = [es],
    random_state=seed)

model.fit(x_train, y_train,
          eval_set = [(x_test, y_test)],
          verbose = 0,
          )    

print('acc : ', model.score(x_test, y_test))   # acc2 :  0.9666666666666667

# print(model.feature_importances_)
# aaa = model.get_booster().get_score(importance_type='weight') = split, 빈도수 개념
# {'f0': 95.0, 'f1': 56.0, 'f2': 12.0, 'f3': 3.0, 'f4': 23.0, 'f5': 16.0, 'f6': 22.0, 
#  'f7': 35.0, 'f8': 23.0, 'f9': 8.0, 'f10': 9.0, 'f11': 9.0, 'f12': 4.0, 'f13': 23.0, 
#  'f14': 14.0, 'f15': 10.0, 'f16': 8.0, 'f17': 7.0, 'f18': 18.0, 'f19': 12.0, 
#  'f20': 39.0, 'f21': 24.0, 'f22': 37.0, 'f23': 5.0, 'f24': 20.0, 'f25': 4.0, 
#  'f26': 23.0, 'f27': 43.0, 'f28': 14.0, 'f29': 11.0}  
# dictionary형태로 제공
aaa = model.get_booster().get_score(importance_type='gain')   # gain으로 중요도 기여도 판별
 
# {'f0': 0.12738777697086334, 'f1': 0.6071306467056274, 'f2': 0.08608654141426086, 
#  'f3': 1.151373267173767, 'f4': 0.16465474665164948, 'f5': 0.1376906931400299, 
#  'f6': 0.07445260882377625, 'f7': 1.4920015335083008, 'f8': 0.3002316951751709, 
#  'f9': 0.2602378726005554, 'f10': 1.050227165222168, 'f11': 0.12306038290262222, 
#  'f12': 0.29094719886779785, 'f13': 0.5654060244560242, 'f14': 0.33709150552749634, 
#  'f15': 0.9213720560073853, 'f16': 0.07099147140979767, 'f17': 0.21485702693462372, 
#  'f18': 0.29548996686935425, 'f19': 0.24784411489963531, 'f20': 2.305630683898926, 
#  'f21': 1.1028623580932617, 'f22': 2.435417652130127, 'f23': 1.841024398803711, 
#  'f24': 0.531847357749939, 'f25': 0.009670689702033997, 'f26': 0.7949649691581726, 
#  'f27': 4.114499092102051, 'f28': 0.16948430240154266, 'f29': 0.19079570472240448}
print(aaa)
# exit()

# 중요도가 낮은거 순서대로 제거해서 성능을 보고싶을 때.

# key_values = list(aaa.items())
# print(key_values)
# keys = list(aaa.keys())
# print(keys)
# values = list(aaa.values())
# print(values)
# # [0.12738777697086334, 0.6071306467056274, 0.08608654141426086, 1.151373267173767, 0.16465474665164948, 0.1376906931400299, 0.07445260882377625, 1.4920015335083008, 0.3002316951751709, 0.2602378726005554, 1.050227165222168, 0.12306038290262222, 0.29094719886779785, 0.5654060244560242, 0.33709150552749634, 0.9213720560073853, 0.07099147140979767, 0.21485702693462372, 0.29548996686935425, 0.24784411489963531, 2.305630683898926, 1.1028623580932617, 2.435417652130127, 1.841024398803711, 0.531847357749939, 0.009670689702033997, 0.7949649691581726, 4.114499092102051, 
# # 0.16948430240154266, 0.19079570472240448]
values = np.array(list(aaa.values()))
keys = list(aaa.keys())
print(values)
# [0.12738778 0.60713065 0.08608654 1.15137327 0.16465475 0.13769069
#  0.07445261 1.49200153 0.3002317  0.26023787 1.05022717 0.12306038
#  0.2909472  0.56540602 0.33709151 0.92137206 0.07099147 0.21485703
#  0.29548997 0.24784411 2.30563068 1.10286236 2.43541765 1.8410244
#  0.53184736 0.00967069 0.79496497 4.11449909 0.1694843  0.1907957 ]
# exit()
# thresholds = np.sort(values)  # sort 오름차순 정렬 낮은게 젤 앞으로 점점 커지는것
# print(thresholds) 

values_scaled = (values - values.min()) / (values.max() - values.min())
print(values_scaled)

thresholds = np.sort(values_scaled)  # sort 오름차순 정렬 낮은게 젤 앞으로 점점 커지는것
print(thresholds) 

#[0.00967069 0.07099147 0.07445261 0.08608654 0.12306038 0.12738778
#  0.13769069 0.16465475 0.1694843  0.1907957  0.21485703 0.24784411
#  0.26023787 0.2909472  0.29548997 0.3002317  0.33709151 0.53184736
#  0.56540602 0.60713065 0.79496497 0.92137206 1.05022717 1.10286236
#  1.15137327 1.49200153 1.8410244  2.30563068 2.43541765 4.11449909]
 
exit()
from sklearn.feature_selection import SelectFromModel

for i in thresholds : 
    selection = SelectFromModel(model, threshold=i, prefit=False)
    # threshold가 i값 이상인것을 모두 훈련시킨다.
    # i 0일 때 4개 전부다 
    #   1일 때 처음 1개 빼고 3개
    
    # threshold가 i값 이상인것을 모두 훈련시킨다.
    # prefit = False : 모델이 아직 학습되지 않았을 때, fit 호출해서 훈련한다.(기본값)
    # prefit = True : 이미 학습된 모델을 전달할 때.
    
    select_x_train = selection.transform(x_train)
    select_x_test = selection.transform(x_test)
    
    if select_x_train.shape[1] == 0:
        print(f"Thresh={i:.3f}, ❌ No features selected → Skipping")
        continue
    
    # print(select_x_train.shape)

    select_model = XGBClassifier(
        n_estimators = 500,
        max_depth = 6,
        gamma = 0,
        min_child_weight = 0,
        subsample = 0.4,
        reg_alpha = 0,
        reg_lambda = 1,
        eval_metric = 'logloss',  # 다중분류 : mlogloss, merror, 이진분류 : logloss, error
                                # 2.1.1 버전 이후로 fit에서 모델로 위치이동.
        # early_stopping_rounds=30,
        callbacks = [es],
        random_state=seed
        )
    
    select_model.fit(select_x_train, y_train,
          eval_set = [(select_x_test, y_test)],
          verbose = 0,
          )    
    
    select_y_pred = select_model.predict(select_x_test)
    score = accuracy_score(y_test, select_y_pred)
    print('Thresh=%.3f, n=%dscore, ACC: %.4f%%' %(i, select_x_train.shape[1], score*100))















