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

seed = 123
random.seed(seed)
np.random.seed(seed)

#1. 데이터
datasets = fetch_california_housing()
x = datasets.data
y = datasets.target

def outlier(nums):
    q1, q2, q3 = np.percentile(nums, [25, 50, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    loc = np.where((nums > upper) | (nums < lower))
    return loc, iqr, upper, lower

for i in range(x.shape[1]):                                            # aaa.shape[1]의 의미는 2 즉 숫자, 총 열의 개수를 말한다 
    print(f"\n열 {i}에 대한 이상치 검사:")
    nums = x[:, i]
    outlier_loc, iqr, up, low = outlier(nums)
    print('이상치 위치:', outlier_loc)
    plt.figure()
    plt.boxplot(x[:, i])
    plt.axhline(up, color='red', linestyle='--', label='upper bound')
    plt.axhline(low, color='blue', linestyle='--', label='lower bound')
    plt.title(f'열 {i} boxplot')
    plt.legend()
    plt.show()

def remove_outliers(x, y):
    total_outlier_idx = set()
    for i in range(x.shape[1]):
        nums = x[:, i]
        outlier_loc, _, _, _ = outlier(nums)
        total_outlier_idx.update(outlier_loc[0])
    print(f'제거할 이상치 행 수: {len(total_outlier_idx)}')
    mask = np.ones(len(x), dtype=bool)
    mask[list(total_outlier_idx)] = False
    return x[mask], y[mask]

x_clean, y_clean = remove_outliers(x, y)

pf = PolynomialFeatures(degree=2, include_bias=True, interaction_only=False)  #각각의 컬럼 데이터값끼리의 곱하기만 출력 제곱을 빼면 성능이 좀 떨어짐.
x_pf = pf.fit_transform(x_clean)

x_train, x_test, y_train, y_test = train_test_split(
    x_pf, y_clean, test_size=0.2, random_state=seed, #stratify=y,
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


# r2 :  0.8080294336818377
# Thresh=0.000, n=45score, R2: 80.8029%
# Thresh=0.000, n=45score, R2: 80.8029%
# Thresh=0.000, n=45score, R2: 80.8029%
# Thresh=0.000, n=45score, R2: 80.8029%
# Thresh=0.000, n=45score, R2: 80.8029%
# Thresh=0.000, n=45score, R2: 80.8029%
# Thresh=0.000, n=45score, R2: 80.8029%
# Thresh=0.000, n=45score, R2: 80.8029%
# Thresh=0.000, n=45score, R2: 80.8029%
# Thresh=0.002, n=36score, R2: 80.8029%
# Thresh=0.003, n=35score, R2: 80.9104%
# Thresh=0.004, n=34score, R2: 80.6540%
# Thresh=0.005, n=33score, R2: 81.4662%
# Thresh=0.005, n=32score, R2: 80.8679%
# Thresh=0.005, n=31score, R2: 80.9993%
# Thresh=0.005, n=30score, R2: 81.2742%
# Thresh=0.006, n=29score, R2: 80.7424%
# Thresh=0.006, n=28score, R2: 81.4548%
# Thresh=0.006, n=27score, R2: 81.0691%
# Thresh=0.006, n=26score, R2: 81.6268%
# Thresh=0.006, n=25score, R2: 81.5274%
# Thresh=0.006, n=24score, R2: 81.7344%
# Thresh=0.006, n=23score, R2: 81.4923%
# Thresh=0.006, n=27score, R2: 81.0691%
# Thresh=0.006, n=26score, R2: 81.6268%
# Thresh=0.006, n=25score, R2: 81.5274%
# Thresh=0.006, n=24score, R2: 81.7344%
# Thresh=0.006, n=23score, R2: 81.4923%
# Thresh=0.006, n=25score, R2: 81.5274%
# Thresh=0.006, n=24score, R2: 81.7344%
# Thresh=0.006, n=23score, R2: 81.4923%
# Thresh=0.006, n=24score, R2: 81.7344%
# Thresh=0.006, n=23score, R2: 81.4923%
# Thresh=0.006, n=23score, R2: 81.4923%
# Thresh=0.006, n=22score, R2: 81.1737%
# Thresh=0.007, n=21score, R2: 81.3103%
# Thresh=0.007, n=21score, R2: 81.3103%
# Thresh=0.007, n=20score, R2: 81.2827%
# Thresh=0.007, n=19score, R2: 81.2007%
# Thresh=0.008, n=18score, R2: 81.1246%
# Thresh=0.008, n=17score, R2: 81.4518%
# Thresh=0.008, n=16score, R2: 81.7320%
# Thresh=0.010, n=15score, R2: 80.8821%
# Thresh=0.011, n=14score, R2: 81.1812%
# Thresh=0.011, n=13score, R2: 81.5011%
# Thresh=0.020, n=12score, R2: 81.7040%
# Thresh=0.020, n=11score, R2: 81.8006%
# Thresh=0.024, n=10score, R2: 81.1467%
# Thresh=0.027, n=9score, R2: 81.2373%
# Thresh=0.029, n=8score, R2: 81.5054%
# Thresh=0.035, n=7score, R2: 80.6277%
# Thresh=0.036, n=6score, R2: 73.1174%
# Thresh=0.041, n=5score, R2: 59.4515%
# Thresh=0.070, n=4score, R2: 56.1644%
# Thresh=0.085, n=3score, R2: 50.3004%
# Thresh=0.131, n=2score, R2: 50.5765%
# Thresh=0.328, n=1score, R2: 36.3931%




















