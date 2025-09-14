from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import accuracy_score, r2_score
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import PolynomialFeatures
import pandas as pd
import time
import numpy as np

#1. 데이터
path = './_data/kaggle/otto/'  
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0) 

x = train_csv.drop(['target'], axis=1)
y = train_csv['target']             

import matplotlib.pyplot as plt
def outlier(nums):
    q1, q2, q3 = np.percentile(nums, [25, 50, 75])
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    loc = np.where((nums > upper) | (nums < lower))
    return loc, iqr, upper, lower

for i in range(x.shape[1]):                                            # aaa.shape[1]의 의미는 2 즉 숫자, 총 열의 개수를 말한다 
    print(f"\n열 {i}에 대한 이상치 검사:")
    nums = x.iloc[:, i]
    outlier_loc, iqr, up, low = outlier(nums)
    print('이상치 위치:', outlier_loc)
    plt.figure()
    plt.boxplot(x.iloc[:, i])
    plt.axhline(up, color='red', linestyle='--', label='upper bound')
    plt.axhline(low, color='blue', linestyle='--', label='lower bound')
    plt.title(f'열 {i} boxplot')
    plt.legend()
    # plt.show()

def remove_outliers(x, y):
    total_outlier_idx = set()
    for i in range(x.shape[1]):
        nums = x.iloc[:, i]
        outlier_loc, _, _, _ = outlier(nums)
        total_outlier_idx.update(outlier_loc[0])
    print(f'제거할 이상치 행 수: {len(total_outlier_idx)}')
    mask = np.ones(len(x), dtype=bool)
    mask[list(total_outlier_idx)] = False
    return x[mask], y[mask]

x_clean, y_clean = remove_outliers(x, y)
   
# exit()    

le = LabelEncoder() # 문자열을 숫자로 현재 y에 들어간 train_csv의 target은 문자열임
y = le.fit_transform(y_clean)  

pf = PolynomialFeatures(degree=2, include_bias=False, interaction_only=False)  #각각의 컬럼 데이터값끼리의 곱하기만 출력 제곱을 빼면 성능이 좀 떨어짐.
x_pf = pf.fit_transform(x_clean)

x_train, x_test, y_train, y_test = train_test_split(
    x_pf,y, shuffle=True, random_state=123, train_size=0.8
    )

n_split = 3
kfold = StratifiedKFold(n_splits=n_split, shuffle=True, random_state=503)

parameters = [
    {'n_estimators': [100,500], 'max_depth':[6,10,12], 'learning_rate': [0.1, 0.01, 0.001]},    # 18
    {'max_depth': [6,8,10,12], 'learning_rate': [0.1, 0.01, 0.001]},                            # 12
    {'min_child_weight': [2,3,5,10], 'learning_rate': [0.1, 0.01, 0.001]}                       # 12
]

#2. 모델
xgb = XGBClassifier()
model = GridSearchCV(xgb, parameters, cv=kfold,   # 42 * 5 = 210
                     verbose=2,
                     n_jobs=-1,
                     refit=True,    # 1번
                     )  # 총 210번 + 1번 = 211번

#3. 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time() - start

print('\n\n- 최적의 매개변수 : ', model.best_estimator_)
print('- 최적의 파라미터 : ', model.best_params_)

#4. 평가, 예측
print('- best_score : ', model.best_score_)
print('- mode.score : ', model.score(x_test, y_test))

y_pred = model.predict(x_test)
print('- accuracy_score : ', accuracy_score(y_test, y_pred))
print('- 걸린시간 : ', round(end, 2), '초\n\n')

# 최적의 매개변수 :  SVC(C=1, kernel='linear')
# 최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'}
# best_score :  0.9833333333333334
# mode.score :  1.0
# accuracy_score :  1.0
# 0.98 초

# pf 적용 후
#  best_score :  0.8176033396200659
# - mode.score :  0.8241758241758241
# - accuracy_score :  0.8241758241758241
# - 걸린시간 :  602.59 초