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
path = './_data/kaggle/santander/'                                          
train_csv = pd.read_csv(path + 'train.csv', index_col=0)                    
test_csv = pd.read_csv(path + 'test.csv', index_col=0)                      

x = train_csv.drop(['target'], axis=1)                                     
y = train_csv['target']                                                    

x_train, x_test, y_train, y_test = train_test_split(                       
    x, y, test_size=0.2, random_state=74, stratify=y
)

scaler = RobustScaler()
x = scaler.fit(x)
x_train = scaler.transform(x_train)
x_test = scaler.transform(x_test)


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
        'verbose': -1,
        'objective': 'multiclass', 
        'num_class': len(np.unique(y_train))
    }
    
    model = LGBMClassifier(**params)
    
    model.fit(
        x_train, y_train,
        eval_set=[(x_test, y_test)],
        eval_metric='multi_logloss', 
        callbacks=[early_stopping(stopping_rounds=30, verbose=False)], # verbose=False로 콜백 출력 억제
    )
    if hasattr(model, 'best_iteration_') and model.best_iteration_ is not None:
        y_pred = model.predict(x_test, num_iteration=model.best_iteration_)
    else:
        y_pred = model.predict(x_test)
        
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


#   최적의 값 :  {'target': 0.555432495061496, 'params': {'n_estimators': 188.63266780218743, 'learning_rate': 0.43582925614836726, 'max_depth': 3.269104063434592, 'min_child_weight': 9.209413512810201, 'gamma': 0.8176646192520692, 'colsample_bytree': 0.839747823306114, 'colsample_bylevel': 0.5615888737897758, 'subsample': 1.057140220405731, 'reg_lambda': 58.278767137817546, 'reg_alpha': 7.004617487124869}}
# 50  번 걸린 시간 :  10.08 초