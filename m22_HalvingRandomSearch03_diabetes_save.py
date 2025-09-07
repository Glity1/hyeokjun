# m18_00 copy
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.metrics import r2_score, mean_squared_error, accuracy_score
from xgboost import XGBClassifier, XGBRegressor
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.experimental import enable_halving_search_cv           # 정식버전이 아니라서 적어주는 라인
from sklearn.model_selection import HalvingGridSearchCV
import time
import joblib
import warnings
warnings.filterwarnings('ignore')

#1. 데이터
x, y = load_diabetes(return_X_y=True)

x_train, x_test, y_train, y_test = train_test_split(
    x, y, test_size=0.2, random_state=55
)

print(x_train.shape) #(353, 10)

n_split = 5
kfold = KFold(n_splits=n_split, shuffle=True, random_state=333)

parameters = [
    {'n_estimators': [100,500], 'max_depth':[6,10,12], 'learning_rate': [0.1, 0.01, 0.001]},    # 18
    {'max_depth': [6,8,10,12], 'learning_rate': [0.1, 0.01, 0.001]},                            # 12
    {'min_child_weight': [2,3,5,10], 'learning_rate': [0.1, 0.01, 0.001]}                       # 12
]   # 총42개의 경우의 수 중에 10개만 빼서 실행한다.

#2. 모델
xgb = XGBRegressor()

model = HalvingGridSearchCV(xgb, parameters, cv=kfold,   
                     verbose=1,
                     n_jobs=-1,
                     refit=True,       # 최종 한번 더 돌림
                     random_state=333,   
                     # default
                     factor=2,           # 배수 : min_resources * factor //  # 데이터가 크면 factor를 크게 조절 //  가급적이면 데이터전체를 훈련시키게끔 구조를 만든다
                                         #        n_candidates / factor   
                     min_resources=40,   # 1 iter때의 최소 훈련 행의 개수
                     max_resources=350,  # 데이터 행의 개수(n_samples)
                     )  

#3. 훈련
start = time.time()
model.fit(x_train, y_train)
end = time.time() - start

# Fitting 5 folds for each of 10 candidates, totalling 50 fits

print('\n\n- 최적의 매개변수 : ', model.best_estimator_)            # refit=false 상태일 때 print다 안됨
print('- 최적의 파라미터 : ', model.best_params_)

#4. 평가, 예측
print('- best_score : ', model.best_score_)
print('- mode.score : ', model.score(x_test, y_test))
 
y_pred = model.predict(x_test)                                      # 두 predict 두개중에 원하는거 쓰면된다

print("R2 Score :", r2_score(y_test, y_pred))
print("MSE :", mean_squared_error(y_test, y_pred))


print('- 걸린시간 : ', round(end, 2), '초\n\n')

# 최적의 매개변수 :  SVC(C=1, kernel='linear')
# 최적의 파라미터 :  {'C': 1, 'degree': 3, 'kernel': 'linear'}
# best_score :  0.9833333333333334
# mode.score :  1.0
# accuracy_score :  1.0
# 0.98 초

import pandas as pd
print(pd.DataFrame(model.cv_results_).sort_values(              # sort : 정렬
     'rank_test_score',                                         # rank_test_score : 좋은 성능 랭킹 파악 // 를 기준으로 오름차순 정렬
      ascending=True))   

print(pd.DataFrame(model.cv_results_).columns)                                                                # pandas 명령어는 암기하자.
# Index(['mean_fit_time', 'std_fit_time', 'mean_score_time', 'std_score_time',
#        'param_learning_rate', 'param_max_depth', 'param_n_estimators',
#        'param_min_child_weight', 'params', 'split0_test_score',
#        'split1_test_score', 'split2_test_score', 'split3_test_score',
#        'split4_test_score', 'mean_test_score', 'std_test_score',
#        'rank_test_score'],
#       dtype='object')

path = './_save/m15_cv_results/'

# 파일저장

pd.DataFrame(model.cv_results_).sort_values(                    
     'rank_test_score',                                         
      ascending=True).to_csv(path + 'm18_03_rs_cv_results2.csv')

path = './_save/m15_cv_results/'
joblib.dump(model.best_estimator_, path + 'm18_03_best_model2.joblib' )







