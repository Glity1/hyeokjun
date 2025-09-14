import numpy as np
import pandas as pd
from bayes_opt import BayesianOptimization
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import r2_score, accuracy_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler
from lightgbm import LGBMClassifier,LGBMRegressor, early_stopping
import time
import joblib
import warnings
warnings.filterwarnings('ignore')
import random

seed = 222
random.seed(seed)
np.random.seed(seed)

#1. 데이터
path = './_data/dacon/diabetes/'
train_csv = pd.read_csv(path + 'train.csv', index_col=0)
test_csv = pd.read_csv(path + 'test.csv', index_col=0)

x = train_csv.drop(['Outcome'], axis=1)
x = x.replace(0, np.nan)
x = x.fillna(train_csv.mean())
y = train_csv['Outcome'] 

x_train, x_test, y_train, y_test = train_test_split(
    x,y, shuffle=True, random_state=123, train_size=0.8
    )


beyesian_params = {
    'n_estimators' : (100, 500),
    'learning_rate' : (0.001, 0.5),
    'max_depth' : (3,10),
    'num_leaves' : (20, 100),
    'min_child_samples' : (5, 200),
    'feature_fraction' : (0.5, 1),
    'bagging_fraction' : (0.5, 1),
    'lambda_l2' : (0, 100),  
    'lambda_l1' : (0, 10),   
    'min_gain_to_split' : (0, 5), 
}

#2. 모델
def Lgb_hamsu(n_estimators, learning_rate, max_depth, num_leaves, min_child_samples, 
              feature_fraction, bagging_fraction, lambda_l2, lambda_l1, min_gain_to_split):
    params = {
        'n_estimators':int(n_estimators),
        'learning_rate':learning_rate,
        'max_depth': int(max_depth),
        'num_leaves': int(num_leaves), # LightGBM의 주요 파라미터
        'min_child_samples': int(min_child_samples), # int로 변환
        'feature_fraction': feature_fraction, # colsample_bytree 대신
        'bagging_fraction': max(min(bagging_fraction,1), 0), # subsample 대신, 범위는 0~1 사이로 확실히
        'bagging_freq': 1, # bagging_fraction 사용 시 권장
        'lambda_l2': max(lambda_l2, 0), # reg_lambda 대신
        'lambda_l1': lambda_l1,       # reg_alpha 대신
        'min_gain_to_split': min_gain_to_split, # 추가된 파라미터
        'n_jobs': -1,
        'random_state': seed, # 재현성 확보
        'verbose': -1
    }
    
    model = LGBMClassifier(**params)
    
    model.fit(
        x_train, y_train,
        eval_set=[(x_test, y_test)],
        eval_metric='rmse', 
        callbacks=[early_stopping(stopping_rounds=30, verbose=False)], # verbose=False로 콜백 출력 억제
    )
    y_pred = model.predict(x_test, num_iteration=model.best_iteration_)
    result = accuracy_score(y_test, y_pred)
    
    return result

optimizer = BayesianOptimization(
    f = Lgb_hamsu,                   # 함수는 y_function을 쓴다.
    pbounds=beyesian_params,
    random_state=seed,
)

n_iter = 50
start = time.time()
optimizer.maximize(init_points=10,         # 초기 훈련 5번     # 총 25번 돌려라
                   n_iter=n_iter)         # 반복 훈련 n_iter번
end = time.time()

print(" 최적의 값 : ", optimizer.max)
print(n_iter, ' 번 걸린 시간 : ', round(end - start, 2), '초')


# 최적의 값 :  {'target': 0.7404580152671756, 'params': {'n_estimators': 337.80509181923384, 'learning_rate': 0.5, 'max_depth': 3.0, 'num_leaves': 66.69168225587784, 'min_child_samples': 60.604501824562085, 'feature_fraction': 1.0, 'bagging_fraction': 0.5, 'lambda_l2': 0.0, 'lambda_l1': 0.0, 'min_gain_to_split': 5.0}}
# 50  번 걸린 시간 :  24.65 초